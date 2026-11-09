package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/logging"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/ratelimit"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/reliability"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/strategy"
)

// FallbackChain manages the ordered provider fallback logic.
//
// Fallback order:
//  1. OpenRouter with models[] array (native intra-provider fallback)
//  2. Groq (service_tier only if configured; free orgs reject "auto")
//  3. Gemini (4-dimensional rate limit check)
//  4. Remaining providers (sorted by strategy)
//  5. OpenRouter with model: "openrouter/free" (ultimate fallback)
//
// Special cases:
//   - HuggingFace 503+loading: wait estimated_time, retry SAME provider
//   - Cerebras: pre-emptive slowdown if remaining-requests < 2
type FallbackChain struct {
	Cfg                *config.Config
	Strategy           strategy.Strategy
	Catalog            *catalog.Catalog
	RateLimiter        *ratelimit.GlobalTracker
	GeminiTracker      *ratelimit.GeminiTracker
	ReliabilityTracker *reliability.Tracker
	HTTPClient         *http.Client
	OnModelExcluded    func(providerID, modelID string)
}

// Request is the parsed incoming chat-completions request.
type Request struct {
	Model     string           `json:"model"`
	Messages  []map[string]any `json:"messages"`
	Stream    bool             `json:"stream,omitempty"`
	MaxTokens int              `json:"max_tokens,omitempty"`
	// Raw holds the original request body for passthrough.
	Raw map[string]any `json:"-"`

	// deadline, when set, is the absolute time by which the request must be
	// answered. It is carried across a route change (direct model → strategy
	// chain) so the time already spent counts against fallback.max_wait.
	deadline time.Time
}

// Response wraps an upstream HTTP response.
type Response struct {
	StatusCode int
	Header     http.Header
	Body       []byte
	// UsedModel is the model that actually served the request (from response body).
	UsedModel string
	// Empty is set on a 200 response that carries no usable completion (no
	// choices, no content and no tool calls, or an unparseable body). Such a
	// response is treated as a failed attempt, not as an answer.
	Empty bool
}

// usable reports whether resp is a genuine, non-empty provider completion.
func (r *Response) usable() bool {
	return r != nil && r.StatusCode == http.StatusOK && !r.Empty
}

// ErrNoProviderAvailable is returned by Execute only when Fallback.MaxWait is
// set and no provider produced a real response within that time.
var ErrNoProviderAvailable = errors.New("no provider produced a response")

// NoEligibleModelError is returned by Execute when a strict strategy (see
// strategy.Strict) filtered out every free model, e.g. the prompt is larger than
// any context window or no model supports the tools the request carries.
// Retrying cannot help, so Execute returns it immediately.
type NoEligibleModelError struct {
	Reason string
}

func (e *NoEligibleModelError) Error() string { return e.Reason }

// minRetryDelay is the only wait between passes, and only when the pass that
// just failed returned almost instantly (every candidate skipped on cooldown,
// no provider call made). It keeps such a pass from turning into a busy loop.
// A pass that actually called providers is followed by the next one at once.
const minRetryDelay = 100 * time.Millisecond

// Execute runs the fallback chain for the given request until a provider
// returns a real response.
//
// A single pass over the chain (executePass) is bounded by Fallback.MaxAttempts.
// When a pass finds no provider that answers, Execute waits (exponential
// backoff with jitter, shortened to the next cooldown expiry) and starts a new
// pass with the unmodified original request. It never fabricates a
// completion: it returns either an upstream response, ctx.Err() when the
// client went away, or — only if Fallback.MaxWait is set and exceeded —
// ErrNoProviderAvailable.
//
// Fallback.MaxWait is a hard cap on the whole call: it is applied as a deadline
// to every provider request and wait, so an in-flight call is cut off when the
// time is up instead of being allowed to finish first.
//
// A strict strategy that leaves no eligible model makes Execute return a
// *NoEligibleModelError at once.
func (fc *FallbackChain) Execute(parent context.Context, req Request) (*Response, error) {
	maxWait := fc.Cfg.Fallback.MaxWait
	deadline := req.deadline
	if deadline.IsZero() && maxWait > 0 {
		deadline = time.Now().Add(maxWait)
	}
	ctx := parent
	if !deadline.IsZero() {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(parent, deadline)
		defer cancel()
	}
	timedOut := func(cause error) error {
		return fmt.Errorf("%w within %s: %v", ErrNoProviderAvailable, maxWait, cause)
	}

	// Provider/model pairs that returned an empty response during this request.
	demoted := make(map[string]bool)

	logging.Debug("fallback: execute starting (max_wait=%s, max_attempts=%d)", fc.Cfg.Fallback.MaxWait, fc.Cfg.Fallback.MaxAttempts)
	resp, err := fc.executePass(ctx, req, demoted)
	if err == nil {
		logging.Info("fallback: success")
		return resp, nil
	}
	if cerr := parent.Err(); cerr != nil {
		return nil, cerr // client went away or shutdown; nobody to answer
	}
	var noModel *NoEligibleModelError
	if errors.As(err, &noModel) {
		return nil, err // no retry can make a model eligible
	}
	if !deadline.IsZero() && (time.Until(deadline) <= 0 || ctx.Err() != nil) {
		return nil, timedOut(err)
	}
	if ctx.Err() != nil {
		if cerr := parent.Err(); cerr != nil {
			return nil, cerr
		}
		return nil, timedOut(err)
	}
	return nil, err
}

// retryDelay turns the base wait into the actual one: jittered ±20% to avoid
// synchronised retries, and cut short when a provider/model cooldown expires
// sooner (never below minRetryDelay).
func (fc *FallbackChain) retryDelay(backoff time.Duration) time.Duration {
	d := time.Duration(float64(backoff) * (0.8 + 0.4*rand.Float64()))
	if fc.RateLimiter != nil {
		if rec := fc.RateLimiter.EarliestRecovery(); !rec.IsZero() {
			if until := time.Until(rec) + 50*time.Millisecond; until < d {
				d = until
			}
		}
	}
	if d < minRetryDelay {
		d = minRetryDelay
	}
	return d
}

// sleepCtx waits for d or until ctx is done, returning ctx.Err() in the latter case.
func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// executePass makes one pass over the fallback chain: at most
// Fallback.MaxAttempts provider calls. It returns the first successful
// response, or an error if no provider answered during this pass. req is never
// modified, so every pass sends the original request.
//
// A 200 response without usable content counts as a failure: the provider is
// skipped for the rest of the pass, and its provider/model key is added to
// demoted so that later passes try all other candidates before it again.
// demoted may be nil.
func (fc *FallbackChain) executePass(ctx context.Context, req Request, demoted map[string]bool) (*Response, error) {
	if demoted == nil {
		demoted = make(map[string]bool)
	}
	freeEntries := fc.Catalog.FreeEntries()
	needsTools, needsVision := requestNeeds(req)
	stratReq := strategy.Request{
		Messages:              req.Messages,
		EstimatedPromptTokens: estimateRequestTokens(req),
		MaxTokens:             req.MaxTokens,
		Stream:                req.Stream,
		Model:                 req.Model,
		NeedsTools:            needsTools,
		NeedsVision:           needsVision,
		Options:               fc.Cfg.Proxy.StrategyOverrides[fc.Strategy.Name()],
	}
	if fc.ReliabilityTracker != nil {
		stratReq.Perf = fc.ReliabilityTracker.Perf
	}

	// A strict strategy's ranked list is a hard filter: the chain stays inside
	// it (no unfiltered last-resort router) and gives up at once if it is empty.
	strictStrat, strict := fc.Strategy.(strategy.Strict)

	ranked := demoteRanked(fc.Strategy.Rank(stratReq, freeEntries, nil), demoted)
	if strict && len(ranked) == 0 && len(freeEntries) > 0 {
		return nil, &NoEligibleModelError{Reason: strictStrat.Explain(stratReq, freeEntries, nil)}
	}
	maxAttempts := fc.Cfg.Fallback.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 5
	}

	// The parallel strategy or a strategy configured with parallel fan-out
	// is a real first-response-wins fan-out. Candidate selection is bounded by
	// the strategy's fan_out / parallel setting or maxAttempts, and all requests
	// share a cancellable context so slow losers stop as soon as one succeeds.
	if fc.Strategy.Name() == "parallel" || fc.isParallelOverride() {
		resp, err := fc.executeParallelPass(ctx, req, stratReq, ranked, maxAttempts)
		if err != nil {
			return nil, err
		}
		if resp != nil {
			return resp, nil
		}
		if !strict {
			fallbackResp, fallbackErr := fc.tryOpenRouterFree(ctx, req)
			if fallbackErr != nil {
				return nil, fallbackErr
			}
			if fallbackResp != nil {
				return fallbackResp, nil
			}
		}
		return nil, fmt.Errorf("no provider answered in this parallel/async pass (%d candidates)", maxAttempts)
	}

	// Steps 1–4: OpenRouter models[] first, then one model per provider in
	// ranked order. Failures fall through at once; slow calls are hedged (see
	// runCandidates in hedge.go).
	resp, attempt, err := fc.runCandidates(ctx, req, stratReq, ranked, maxAttempts, demoted)
	if err != nil {
		return nil, err
	}
	if resp != nil {
		return resp, nil
	}

	// Step 5: Ultimate fallback — OpenRouter free router. It picks any model, so
	// it is skipped for strict strategies, whose candidates must all meet the
	// request's requirements.
	if !strict {
		freeResp, freeErr := fc.tryOpenRouterFree(ctx, req)
		if freeErr == nil && freeResp != nil {
			return freeResp, nil
		}
	}

	return nil, fmt.Errorf("no provider answered in this pass (%d attempts)", attempt)
}

type parallelCandidate struct {
	ranked strategy.RankedEntry
	cfg    config.ProviderConfig
	body   map[string]any
}

type parallelResult struct {
	candidate parallelCandidate
	resp      *Response
	err       error
}

// executeParallelPass races the strategy's provider candidates and returns the
// first usable completion. Failures received before the winner are still
// recorded for cooldown and catalog maintenance.
func (fc *FallbackChain) executeParallelPass(ctx context.Context, req Request, stratReq strategy.Request, ranked []strategy.RankedEntry, maxAttempts int) (*Response, error) {
	candidates := make([]parallelCandidate, 0, len(ranked))
	seenProviders := make(map[string]bool)
	for _, r := range ranked {
		if len(candidates) >= maxAttempts || seenProviders[r.ProviderID] {
			continue
		}
		pc := fc.findProvider(r.ProviderID)
		if pc == nil {
			continue
		}
		if fc.RateLimiter != nil && fc.RateLimiter.IsOnCooldown(r.ProviderID) {
			continue
		}
		if fc.RateLimiter != nil && fc.RateLimiter.IsOnCooldown(r.ProviderID+"/"+r.ModelID) {
			continue
		}
		if r.ProviderID == "gemini" && fc.GeminiTracker != nil && !fc.GeminiTracker.CanRequest(r.ModelID, stratReq.EstimatedPromptTokens) {
			continue
		}
		if r.ProviderID == "cerebras" && fc.RateLimiter != nil && !fc.RateLimiter.Provider("cerebras").CanRequest(r.ModelID, 0, 0) {
			continue
		}

		body := fc.buildBody(r.ProviderID, req.Raw, r.ModelID)
		if r.ProviderID == "openrouter" {
			models := fc.collectProviderModels("openrouter", ranked)
			body = BuildOpenRouterRequest(req.Raw, models)
		}
		callCfg := *pc
		if fc.RateLimiter != nil {
			if keys := pc.AllKeys(); len(keys) > 1 {
				if idx := fc.RateLimiter.CurrentKeyIndex(r.ProviderID); idx < len(keys) {
					callCfg.APIKey = keys[idx]
					callCfg.APIKeyEnv = ""
				}
			}
		}
		candidates = append(candidates, parallelCandidate{ranked: r, cfg: callCfg, body: body})
		seenProviders[r.ProviderID] = true
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	callCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	results := make(chan parallelResult, len(candidates))
	for _, candidate := range candidates {
		candidate := candidate
		go func() {
			if candidate.ranked.ProviderID == "cerebras" && fc.Cfg.Fallback.CerebrasRequestSpacingMs > 0 {
				if err := sleepCtx(callCtx, time.Duration(fc.Cfg.Fallback.CerebrasRequestSpacingMs)*time.Millisecond); err != nil {
					results <- parallelResult{candidate: candidate, err: err}
					return
				}
			}
			resp, err := fc.callProvider(callCtx, candidate.cfg, candidate.body)
			results <- parallelResult{candidate: candidate, resp: resp, err: err}
		}()
	}

	var failures []parallelResult
	for range candidates {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case result := <-results:
			if result.err == nil && result.resp != nil && result.resp.usable() {
				for _, failure := range failures {
					fc.recordParallelFailure(failure)
				}
				providerID := result.candidate.ranked.ProviderID
				if fc.ReliabilityTracker != nil {
					fc.ReliabilityTracker.Record(providerID, true)
				}
				if providerID == "gemini" && fc.GeminiTracker != nil {
					fc.GeminiTracker.RecordRequest(result.candidate.ranked.ModelID, stratReq.EstimatedPromptTokens)
				}
				if providerID == "cerebras" && fc.RateLimiter != nil {
					info := ratelimit.ExtractRateLimitInfo("cerebras", result.resp.Header, result.resp.Body)
					if info.RemainingRequests >= 0 && info.RemainingRequests < 2 {
						fc.RateLimiter.Provider("cerebras").PreemptiveSlowdown(info.WaitDuration())
					}
				}
				return result.resp, nil
			}
			if result.err == nil && result.resp != nil && result.resp.StatusCode == http.StatusOK && result.resp.Empty {
				if fc.Catalog != nil {
					fc.Catalog.MarkNeedsReverification(result.candidate.ranked.ProviderID, result.candidate.ranked.ModelID)
				}
			}
			failures = append(failures, result)
		}
	}
	for _, failure := range failures {
		fc.recordParallelFailure(failure)
	}
	return nil, nil
}

func (fc *FallbackChain) recordParallelFailure(result parallelResult) {
	providerID := result.candidate.ranked.ProviderID
	modelID := result.candidate.ranked.ModelID
	resp := result.resp
	if result.err != nil && errors.Is(result.err, context.Canceled) {
		return
	}
	if resp != nil {
		switch classifyUpstreamFailure(resp) {
		case failureNoCredit:
			if resp.StatusCode == http.StatusPaymentRequired {
				fc.excludeModel(providerID, modelID)
			} else if fc.RateLimiter != nil {
				fc.RateLimiter.SetCooldown(providerID, time.Now().Add(noCreditCooldown))
			}
			return
		case failureUnsupportedModel:
			if fc.RateLimiter != nil {
				fc.RateLimiter.SetCooldown(providerID+"/"+modelID, time.Now().Add(unsupportedModelCooldown))
			}
			return
		case failureDailyLimit:
			fc.noteOpenRouterFailure(resp, 0)
			return
		}
		switch resp.StatusCode {
		case http.StatusTooManyRequests:
			if fc.Catalog != nil {
				fc.Catalog.MarkNeedsReverification(providerID, modelID)
			}
			info := ratelimit.ExtractRateLimitInfo(providerID, resp.Header, resp.Body)
			wait := info.WaitDuration()
			if isZeroQuota(resp.Body) {
				wait = zeroQuotaCooldown
			}
			if fc.RateLimiter != nil {
				fc.RateLimiter.SetCooldown(providerID+"/"+modelID, time.Now().Add(wait))
				if result.candidate.cfg.NumKeys() > 1 {
					fc.RateLimiter.RotateKey(providerID, result.candidate.cfg.NumKeys())
				}
			}
		case http.StatusUnauthorized, http.StatusForbidden:
			if fc.RateLimiter != nil {
				fc.RateLimiter.SetCooldown(providerID, time.Now().Add(10*time.Minute))
			}
			if fc.ReliabilityTracker != nil {
				fc.ReliabilityTracker.Record(providerID, false)
			}
		case http.StatusPaymentRequired, http.StatusNotFound:
			fc.excludeModel(providerID, modelID)
		default:
			// Model-specific transient failures don't count against provider health.
		}
	} else if result.err != nil && fc.ReliabilityTracker != nil {
		fc.ReliabilityTracker.Record(providerID, false)
		if fc.RateLimiter != nil && fc.ReliabilityTracker.ShouldCooldown(providerID) {
			fc.RateLimiter.SetCooldown(providerID, time.Now().Add(30*time.Second))
		}
	}
}

// noteOpenRouterFailure reacts to a failed OpenRouter native call: it parks
// OpenRouter until the daily free quota resets, or shrinks the models[] array
// when OpenRouter reports a stricter size limit. sent is the models[] length
// that was sent (0 when not applicable).
func (fc *FallbackChain) noteOpenRouterFailure(resp *Response, sent int) {
	if resp == nil {
		return
	}
	if classifyUpstreamFailure(resp) == failureDailyLimit && fc.RateLimiter != nil {
		wait := openRouterResetWait(resp.Body)
		fc.RateLimiter.SetCooldown(openRouterCooldownKey, time.Now().Add(wait))
		logging.Warn("fallback: openrouter free-models-per-day quota exhausted — skipping OpenRouter for %s", wait.Round(time.Second))
		return
	}
	if limit := modelsArrayLimit(resp); limit > 0 && sent > 0 {
		n := lowerOpenRouterModelsLimit(sent, limit)
		logging.Warn("fallback: openrouter rejected models[] of %d (limit %d) — using %d from now on", sent, limit, n)
	}
}

func (fc *FallbackChain) excludeModel(providerID, modelID string) {
	changed := false
	if fc.Catalog != nil {
		changed = fc.Catalog.ExcludeModel(providerID, modelID)
	}
	if changed && fc.OnModelExcluded != nil {
		fc.OnModelExcluded(providerID, modelID)
	}
}

func (fc *FallbackChain) tryOpenRouterFree(ctx context.Context, req Request) (*Response, error) {
	or := fc.findProvider("openrouter")
	if or == nil {
		return nil, nil
	}
	if fc.RateLimiter != nil && fc.RateLimiter.IsOnCooldown(openRouterCooldownKey) {
		logging.Debug("fallback: openrouter on cooldown — skipping openrouter/free")
		return nil, nil
	}
	logging.Info("fallback: trying openrouter/free ultimate fallback")
	resp, err := fc.callProvider(ctx, *or, OpenRouterFreeRouterRequest(req.Raw))
	if err == nil && resp != nil && resp.usable() {
		if fc.ReliabilityTracker != nil {
			fc.ReliabilityTracker.Record("openrouter", true)
		}
		return resp, nil
	}
	fc.noteOpenRouterFailure(resp, 0)
	if fc.ReliabilityTracker != nil {
		fc.ReliabilityTracker.Record("openrouter", false)
	}
	return nil, err
}

func (fc *FallbackChain) callProvider(ctx context.Context, cfg config.ProviderConfig, body map[string]any) (*Response, error) {
	if d, err := time.ParseDuration(cfg.Timeout); err == nil && d > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, d)
		defer cancel()
	}
	endpoint := strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions"

	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	h, v := cfg.ResolvedAuth()
	if v != "" && v != "Bearer " {
		req.Header.Set(h, v)
	}
	for k, val := range cfg.ExtraHeaders {
		req.Header.Set(k, val)
	}

	resp, err := fc.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if cfg.ID == "gemini" && resp.StatusCode == http.StatusOK {
		respBody = StripGeminiThoughts(respBody)
	}

	// Extract the actual model used (OpenRouter fills this in)
	usedModel := ""
	empty := false
	if resp.StatusCode == http.StatusOK {
		var parsed struct {
			Model string `json:"model"`
		}
		if json.Unmarshal(respBody, &parsed) == nil {
			usedModel = parsed.Model
		}
		empty = isEmptyCompletion(respBody)
	}

	return &Response{
		StatusCode: resp.StatusCode,
		Header:     resp.Header,
		Body:       respBody,
		UsedModel:  usedModel,
		Empty:      empty,
	}, nil
}

// demoteRanked returns ranked with every demoted provider/model moved to the
// end (order otherwise unchanged), so providers that already returned an empty
// response are only retried after all other candidates.
func demoteRanked(ranked []strategy.RankedEntry, demoted map[string]bool) []strategy.RankedEntry {
	if len(demoted) == 0 {
		return ranked
	}
	out := make([]strategy.RankedEntry, 0, len(ranked))
	var last []strategy.RankedEntry
	for _, r := range ranked {
		if demoted[r.ProviderID+"/"+r.ModelID] {
			last = append(last, r)
		} else {
			out = append(out, r)
		}
	}
	return append(out, last...)
}

// isEmptyCompletion reports whether a 200 response body is not a usable chat
// completion: unparseable, without choices, or where no choice carries text
// content, tool calls or a function call. Reasoning-only output does not count
// as an answer.
func isEmptyCompletion(body []byte) bool {
	var c struct {
		Choices []struct {
			Message struct {
				Content      any   `json:"content"`
				ToolCalls    []any `json:"tool_calls"`
				FunctionCall any   `json:"function_call"`
			} `json:"message"`
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &c); err != nil {
		return true
	}
	for _, ch := range c.Choices {
		m := ch.Message
		if len(m.ToolCalls) > 0 || m.FunctionCall != nil {
			return false
		}
		if strings.TrimSpace(ch.Text) != "" || hasContent(m.Content) {
			return false
		}
	}
	return true
}

// hasContent reports whether a message "content" value (string or array of
// parts) holds anything beyond whitespace.
func hasContent(v any) bool {
	switch c := v.(type) {
	case string:
		return strings.TrimSpace(c) != ""
	case []any:
		for _, part := range c {
			m, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if t, isText := m["text"].(string); isText {
				if strings.TrimSpace(t) != "" {
					return true
				}
			} else if len(m) > 0 {
				return true // non-text part (image, audio, …)
			}
		}
	}
	return false
}

func (fc *FallbackChain) isParallelOverride() bool {
	if fc.Strategy == nil {
		return false
	}
	opts := fc.Cfg.Proxy.StrategyOverrides[fc.Strategy.Name()]
	switch v := opts["parallel"].(type) {
	case int:
		return v > 0
	case int64:
		return v > 0
	case float64:
		return v > 0
	case bool:
		return v
	}
	return false
}

func (fc *FallbackChain) findProvider(id string) *config.ProviderConfig {
	return findProviderCfg(fc.Cfg, id)
}

func (fc *FallbackChain) collectProviderModels(providerID string, ranked []strategy.RankedEntry) []string {
	var models []string
	limit := currentOpenRouterModelsLimit()
	for _, r := range ranked {
		if r.ProviderID == providerID {
			models = append(models, r.ModelID)
			if len(models) == limit {
				break
			}
		}
	}
	return models
}

// anthropicOnlyFields are request fields specific to the Anthropic Messages API
// that are not part of the OpenAI spec and will cause 400 errors on other providers.
var anthropicOnlyFields = []string{
	"promptcachekey",
	"prompt_cache_key",
	"thinking",
	"betas",
}

func (fc *FallbackChain) buildBody(providerID string, raw map[string]any, modelID string) map[string]any {
	body := copyMap(raw)
	body["model"] = modelID
	delete(body, "stream") // always non-streaming to providers; fallback requires buffered JSON

	// Strip Anthropic-specific fields that other providers reject with 400.
	for _, f := range anthropicOnlyFields {
		delete(body, f)
	}

	// Apply max_tokens_cap from strategy_overrides if the strategy has one configured
	// and the client hasn't already requested fewer tokens.
	if fc.Strategy != nil {
		if overrides, ok := fc.Cfg.Proxy.StrategyOverrides[fc.Strategy.Name()]; ok {
			if capRaw, ok := overrides["max_tokens_cap"]; ok {
				var cap int
				switch v := capRaw.(type) {
				case int:
					cap = v
				case float64:
					cap = int(v)
				}
				if cap > 0 {
					current := 0
					if mt, ok := body["max_tokens"].(float64); ok {
						current = int(mt)
					}
					if current == 0 || current > cap {
						body["max_tokens"] = cap
					}
				}
			}
		}
	}

	// Provider-native service tier (e.g. Groq), only when explicitly configured:
	// Groq rejects service_tier "auto" for free-tier orgs with a 400.
	if pc := fc.findProvider(providerID); pc != nil && pc.NativeFeatures.ServiceTier != "" {
		body["service_tier"] = pc.NativeFeatures.ServiceTier
	}
	return sanitizeBody(providerID, body)
}

const (
	// maxModelsPerProvider bounds how many models of one provider a single
	// request tries after model-level errors (anything but auth/transport), so a large catalog
	// such as NVIDIA NIM cannot use up every attempt.
	maxModelsPerProvider = 3
	// zeroQuotaCooldown applies when a 429 reports a quota "limit: 0": the
	// model is listed but not on the free tier, so retrying soon is pointless.
	zeroQuotaCooldown = 24 * time.Hour
)

// isZeroQuota reports whether a 429 body says the quota for this model is zero
// (Gemini returns "limit: 0" for models outside the free tier).
func isZeroQuota(body []byte) bool {
	return bytes.Contains(body, []byte("limit: 0,"))
}

// upstreamErrSnippet returns a short " (...)" suffix describing why an upstream
// call failed, for logging. It never includes request data or API keys.
func upstreamErrSnippet(resp *Response, err error) string {
	if err != nil {
		return fmt.Sprintf(" (%v)", err)
	}
	if resp == nil || len(resp.Body) == 0 {
		return ""
	}
	msg := strings.Join(strings.Fields(string(resp.Body)), " ")
	if len(msg) > 200 {
		msg = msg[:200] + "…"
	}
	return " (" + msg + ")"
}

// parseHFLoadingWait extracts the estimated_time from a HuggingFace 503 body.
// Returns 0 if not a loading response.
func parseHFLoadingWait(body []byte) int64 {
	if !strings.Contains(string(body), "loading") {
		return 0
	}
	var hfErr struct {
		EstimatedTime float64 `json:"estimated_time"`
	}
	if json.Unmarshal(body, &hfErr) == nil && hfErr.EstimatedTime > 0 {
		ms := int64(hfErr.EstimatedTime * 1000)
		if ms > 60_000 {
			ms = 60_000 // cap at 60s
		}
		return ms
	}
	return 30_000 // default 30s if not specified
}

const (
	// imageTokenEstimate is a flat allowance per image part; real cost varies by
	// provider and resolution.
	imageTokenEstimate = 800
	// messageOverheadTokens covers role markers and separators per message.
	messageOverheadTokens = 4
)

// estimateRequestTokens gives a rough input token count (4 chars ≈ 1 token). It
// counts text in string and part-list content, tool calls the assistant made
// earlier, images, and the tool/function definitions the request carries —
// which can dominate the prompt in agent workloads.
func estimateRequestTokens(req Request) int {
	total := 0
	for _, m := range req.Messages {
		total += messageOverheadTokens
		switch c := m["content"].(type) {
		case string:
			total += len(c) / 4
		case []any:
			for _, part := range c {
				p, ok := part.(map[string]any)
				if !ok {
					continue
				}
				if text, isText := p["text"].(string); isText {
					total += len(text) / 4
				} else if isImagePart(p) {
					total += imageTokenEstimate
				}
			}
		}
		if calls, ok := m["tool_calls"]; ok {
			if b, err := json.Marshal(calls); err == nil {
				total += len(b) / 4
			}
		}
	}
	for _, key := range []string{"tools", "functions"} {
		if defs, ok := req.Raw[key]; ok {
			if b, err := json.Marshal(defs); err == nil {
				total += len(b) / 4
			}
		}
	}
	return total
}

// requestNeeds reports the capabilities a request requires of the serving
// model: tool calling (tool/function definitions, or tool calls and results in
// the conversation) and vision (image content).
func requestNeeds(req Request) (tools, vision bool) {
	for _, key := range []string{"tools", "functions"} {
		if defs, ok := req.Raw[key].([]any); ok && len(defs) > 0 {
			tools = true
		}
	}
	for _, m := range req.Messages {
		if role, _ := m["role"].(string); role == "tool" || role == "function" {
			tools = true
		}
		if calls, ok := m["tool_calls"].([]any); ok && len(calls) > 0 {
			tools = true
		}
		if parts, ok := m["content"].([]any); ok {
			for _, part := range parts {
				if p, ok := part.(map[string]any); ok && isImagePart(p) {
					vision = true
				}
			}
		}
	}
	return tools, vision
}

// isImagePart reports whether a content part carries an image.
func isImagePart(p map[string]any) bool {
	switch t, _ := p["type"].(string); t {
	case "image_url", "image", "input_image":
		return true
	}
	return false
}
