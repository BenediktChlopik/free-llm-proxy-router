package strategy

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/models"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/reliability"
)

// StrategySmart uses the most capable model that safely fits the request, while
// keeping routing efficient and avoiding context-limit and tool-calling errors.
//
// It incorporates a "reliable" factor: models with poor historical success
// are demoted, and models with proven records are prioritized.
//
// It works in two stages:
//
//  1. Filter. A model is a candidate only if it
//     - has a context window large enough for the request: estimated prompt
//     (including tool definitions) x safety margin + the reply budget;
//     - supports every capability the request needs: tool calling when the
//     request carries tools or earlier tool calls, vision when it carries
//     images.
//     Models that fail are removed, not demoted — the fallback chain never
//     reaches them.
//
//  2. Rank. The surviving candidates are ordered by size (effective
//     parameters) adjusted for how they have actually performed: each doubling
//     of a model's typical response time costs speed_weight "size doublings"
//     (a model must be about twice as large to justify answering twice as
//     slowly), and a poor success rate costs failure_weight x (1 - rate). So a
//     fast, reliable model beats a bigger one that is slow or often failing.
//     Models never measured get a mild latency penalty, which lets measured
//     fast models lead while untried ones remain fallback candidates. Failure
//     records fade over ~30 minutes, so a demoted model is retried later.
//     Without history the order is purely by size. Among models that score
//     the same, the one with provider-confirmed tool support comes first for
//     tool requests, then the one with the smaller sufficient context window,
//     which keeps long-context models free for the requests that need them.
//
// During fallback the proxy walks this same list, so every retry candidate meets
// the same requirements as the first choice. StrategySmart is Strict: it never
// falls back to an unfiltered router, and when no model qualifies the request
// fails fast with an explanation instead of being retried.
//
// Tunables (proxy.strategy_overrides.smart):
//
//	safety_margin          prompt-size multiplier, token estimates are rough (default 1.15)
//	output_reserve_tokens  reply budget when the client sets no max_tokens (default 1024)
//	unknown_context_tokens window assumed when the catalog has none (default 8192)
//	max_tokens_cap         same cap the proxy applies to the request, so the reply budget matches
//	speed_weight           size doublings charged per doubling of latency (default 1; 0 ignores speed)
//	failure_weight         size doublings charged for a 0% success rate (default 6; 0 ignores failures)
//	unknown_latency_ms     latency assumed for never-measured models (default 1000)
//	fast_latency_ms        latency at or below which no speed penalty applies (default 500)
type StrategySmart struct {
	Tracker *reliability.Tracker
}

func (s *StrategySmart) Name() string { return "smart" }

const (
	defaultSafetyMargin   = 1.15
	defaultOutputReserve  = 1024
	defaultUnknownContext = 8192
	defaultSpeedWeight    = 1.0
	defaultFailureWeight  = 6.0
	defaultUnknownLatency = 1000.0
	defaultFastLatency    = 500.0
	longContextNameHint   = 131072 // window assumed for names that advertise long context
	unknownParamsBillions = 30.0   // models with no parseable size rank as mid-sized
)

// smartCandidate is a model that passed the filter, with its sort keys.
type smartCandidate struct {
	entry          catalog.CatalogEntry
	size           float64 // effective parameters, billions
	score          float64 // log2(size) minus speed and failure penalties; higher ranks first
	window         int     // context window used for the fit check
	confirmedTools bool    // provider says the model supports tools
}

// smartTally counts why models were rejected, for Explain.
type smartTally struct {
	free          int
	noTools       int
	noVision      int
	tooSmall      int
	largestWindow int // largest window among models that had the needed capabilities
}

func (s *StrategySmart) Rank(req Request, entries []catalog.CatalogEntry, enriched *models.EnrichedCatalog) []RankedEntry {
	cands, _ := smartFilter(req, entries, enriched)
	for i := range cands {
		cands[i].score = smartScore(req, cands[i], s.Tracker)
	}

	sort.SliceStable(cands, func(i, j int) bool {
		a, b := cands[i], cands[j]
		if math.Abs(a.score-b.score) > 1e-9 {
			return a.score > b.score // best first
		}
		if req.NeedsTools && a.confirmedTools != b.confirmedTools {
			return a.confirmedTools
		}
		return a.window < b.window // smallest sufficient window first
	})

	ranked := make([]RankedEntry, 0, len(cands))
	for _, c := range cands {
		ranked = append(ranked, RankedEntry{
			Entry:      c.entry,
			ProviderID: c.entry.ProviderID,
			ModelID:    c.entry.ModelID,
			Score:      -c.score,
		})
	}
	return ranked
}

// smartScore is log2 of the model size, less penalties for slowness and
// failures when performance history is available.
func smartScore(req Request, c smartCandidate, tracker *reliability.Tracker) float64 {
	score := math.Log2(c.size)
	if req.Perf == nil && tracker == nil {
		return score
	}

	var latency, success float64
	var calls int64

	if tracker != nil {
		latency, success, calls = tracker.Perf(c.entry.ProviderID, c.entry.ModelID)
	} else if req.Perf != nil {
		latency, success, calls = req.Perf(c.entry.ProviderID, c.entry.ModelID)
	}
	fast := optFloat(req.Options, "fast_latency_ms", defaultFastLatency)
	if fast < 1 {
		fast = defaultFastLatency
	}
	if calls == 0 || latency <= 0 {
		latency = optFloat(req.Options, "unknown_latency_ms", defaultUnknownLatency)
	}
	if latency > fast {
		score -= optFloat(req.Options, "speed_weight", defaultSpeedWeight) * math.Log2(latency/fast)
	}
	if success < 1 {
		score -= optFloat(req.Options, "failure_weight", defaultFailureWeight) * (1 - success)
	}
	return score
}

// Explain implements Strict.
func (s *StrategySmart) Explain(req Request, entries []catalog.CatalogEntry, enriched *models.EnrichedCatalog) string {
	_, t := smartFilter(req, entries, enriched)
	need := smartRequiredContext(req)

	var needs []string
	needs = append(needs, fmt.Sprintf("about %d tokens of context (prompt plus reply budget)", need))
	if req.NeedsTools {
		needs = append(needs, "tool calling")
	}
	if req.NeedsVision {
		needs = append(needs, "image input")
	}

	var why []string
	if t.noTools > 0 {
		why = append(why, fmt.Sprintf("%d lack tool calling", t.noTools))
	}
	if t.noVision > 0 {
		why = append(why, fmt.Sprintf("%d lack image input", t.noVision))
	}
	if t.tooSmall > 0 {
		why = append(why, fmt.Sprintf("%d have too small a context window (largest known: %d tokens)", t.tooSmall, t.largestWindow))
	}

	msg := fmt.Sprintf("no free model meets this request, which needs %s", strings.Join(needs, ", "))
	if len(why) > 0 {
		msg += fmt.Sprintf("; of %d free models, %s", t.free, strings.Join(why, ", "))
	}
	return msg + ". Shorten the request, lower max_tokens, or use a different strategy."
}

// smartFilter applies the context and capability requirements.
func smartFilter(req Request, entries []catalog.CatalogEntry, enriched *models.EnrichedCatalog) ([]smartCandidate, smartTally) {
	var (
		cands []smartCandidate
		t     smartTally
	)
	need := smartRequiredContext(req)
	unknown := optInt(req.Options, "unknown_context_tokens", defaultUnknownContext)
	if unknown < 1 {
		unknown = defaultUnknownContext
	}

	for _, e := range entries {
		if !e.IsFree {
			continue
		}
		t.free++

		confirmed := false
		if req.NeedsTools {
			if _, ok := toolsScore(e); !ok {
				t.noTools++
				continue
			}
			if v, has := e.Metadata["supports_tools"].(bool); has && v {
				confirmed = true
			}
		}
		if req.NeedsVision && !supportsVision(e) {
			t.noVision++
			continue
		}

		window := smartWindow(e, enriched, unknown)
		if window > t.largestWindow {
			t.largestWindow = window
		}
		if window < need {
			t.tooSmall++
			continue
		}

		cands = append(cands, smartCandidate{
			entry:          e,
			size:           smartSize(e),
			window:         window,
			confirmedTools: confirmed,
		})
	}
	return cands, t
}

// smartRequiredContext is the context a model must offer: the estimated prompt
// padded by the safety margin, plus the reply budget.
func smartRequiredContext(req Request) int {
	margin := optFloat(req.Options, "safety_margin", defaultSafetyMargin)
	if margin < 1 {
		margin = 1
	}
	out := req.MaxTokens
	if capTokens := optInt(req.Options, "max_tokens_cap", 0); capTokens > 0 && (out == 0 || out > capTokens) {
		out = capTokens // the proxy clamps max_tokens to this cap before sending
	}
	if out <= 0 {
		out = optInt(req.Options, "output_reserve_tokens", defaultOutputReserve)
	}
	// The epsilon keeps float noise (1000 x 1.15 = 1150.0000000000002) from adding a token.
	return int(math.Ceil(float64(req.EstimatedPromptTokens)*margin-1e-9)) + out
}

// smartWindow returns the context window to check a request against. A window
// the catalog does not know is assumed small (unknown), or long when the model
// name advertises it, so an unverified model is never trusted with a big prompt.
func smartWindow(e catalog.CatalogEntry, enriched *models.EnrichedCatalog, unknown int) int {
	if w := resolveContextWindow(e, enriched); w > 0 {
		return w
	}
	for _, c := range models.DetectCapabilities(e.ModelID) {
		if c == models.CapLongContext && longContextNameHint > unknown {
			return longContextNameHint
		}
	}
	return unknown
}

// smartSize is the model's effective size in billions of parameters.
func smartSize(e catalog.CatalogEntry) float64 {
	p := models.ExtractParams(e.ModelID)
	if p.Effective > 0 {
		return p.Effective
	}
	if p.Billions > 0 {
		return p.Billions
	}
	return unknownParamsBillions
}

// supportsVision reports whether the model accepts image input: an explicit
// catalog flag wins, otherwise the vision patterns in capabilities.yaml decide.
func supportsVision(e catalog.CatalogEntry) bool {
	if v, ok := e.Metadata["supports_vision"].(bool); ok {
		return v
	}
	for _, c := range models.DetectCapabilities(e.ModelID) {
		if c == models.CapVision {
			return true
		}
	}
	return false
}

// resolveContextWindow returns the context window the catalog or the enriched
// catalog knows for e, or 0 when it is unknown.
func resolveContextWindow(e catalog.CatalogEntry, enriched *models.EnrichedCatalog) int {
	if e.ContextWindow > 0 {
		return e.ContextWindow
	}
	if enriched != nil {
		for _, em := range enriched.Models {
			for _, inst := range em.Instances {
				if inst.ProviderID == e.ProviderID && inst.ModelID == e.ModelID && inst.ContextWindow > 0 {
					return inst.ContextWindow
				}
			}
		}
	}
	return 0
}

// optInt reads an integer option; YAML numbers arrive as int or float64.
func optInt(o map[string]any, key string, def int) int {
	switch v := o[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return def
}

// optFloat reads a numeric option.
func optFloat(o map[string]any, key string, def float64) float64 {
	switch v := o[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case int64:
		return float64(v)
	}
	return def
}
