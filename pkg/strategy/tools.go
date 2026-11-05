package strategy

import (
	"sort"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/models"
)

// StrategyTools is a model group for agents and clients that rely on tool /
// function calling. Unlike the other strategies, which only reorder, it is a
// strict filter: a model that is not known to support tool calling is never a
// candidate, so the proxy cannot answer a tool request with a model that
// ignores the tools.
//
// Tool support is decided in this order:
//  1. catalog Metadata["supports_tools"] (bool), set by scanners that learn it
//     from the provider's model list (OpenRouter's supported_parameters);
//     an explicit false excludes the model.
//  2. the function_calling patterns in capabilities.yaml (name heuristics).
//
// Within the group, models with provider-confirmed support rank first, then
// larger models (tool-call reliability grows with size); sub-3B models last.
type StrategyTools struct{}

func (s *StrategyTools) Name() string { return "tools" }

func (s *StrategyTools) Rank(req Request, entries []catalog.CatalogEntry, enriched *models.EnrichedCatalog) []RankedEntry {
	var ranked []RankedEntry
	for _, e := range entries {
		if !e.IsFree {
			continue
		}
		score, ok := toolsScore(e)
		if !ok {
			continue
		}
		ranked = append(ranked, RankedEntry{
			Entry:      e,
			ProviderID: e.ProviderID,
			ModelID:    e.ModelID,
			Score:      score,
		})
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		return ranked[i].Score < ranked[j].Score
	})
	return ranked
}

// toolsScore returns the score (lower = better) and whether the model is in the group.
func toolsScore(e catalog.CatalogEntry) (float64, bool) {
	base := 0.0
	if v, ok := e.Metadata["supports_tools"].(bool); ok {
		if !v {
			return 0, false
		}
		base = 1.0 // confirmed by the provider
	} else {
		detected := false
		for _, c := range models.DetectCapabilities(e.ModelID) {
			if c == models.CapFunctionCall {
				detected = true
				break
			}
		}
		if !detected {
			return 0, false
		}
		base = 2.0 // inferred from the model name
	}

	params := models.ExtractParams(e.ModelID)
	effective := params.Effective
	if effective == 0 {
		effective = params.Billions
	}
	if effective > 0 && effective < 3 {
		base += 3.0 // tiny models call tools unreliably
	}
	return base + 1.0/(effective+1), true
}
