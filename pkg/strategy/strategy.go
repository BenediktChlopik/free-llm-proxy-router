// Package strategy implements the 15 model-selection strategies.
package strategy

import (
	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/models"
)

// Request holds the information needed to rank models.
type Request struct {
	// Messages is the list of messages in the chat request.
	Messages []map[string]any
	// EstimatedPromptTokens is the approximate token count of the input,
	// including tool/function definitions.
	EstimatedPromptTokens int
	// MaxTokens is the requested output token limit.
	MaxTokens int
	// Stream indicates a streaming request.
	Stream bool
	// Model is the requested model (or "auto").
	Model string
	// NeedsTools is set when the request carries tool/function definitions or
	// earlier tool calls, so the serving model must support tool calling.
	NeedsTools bool
	// NeedsVision is set when the request carries image content.
	NeedsVision bool
	// Options holds the strategy's proxy.strategy_overrides.<name> settings.
	Options map[string]any
}

// RankedEntry pairs a catalog entry with a priority score (lower = higher priority).
type RankedEntry struct {
	Entry    catalog.CatalogEntry
	Score    float64
	ProviderID string
	ModelID  string
}

// Strategy is the interface implemented by all 15 strategies.
type Strategy interface {
	// Name returns the strategy identifier used in config.
	Name() string
	// Rank returns an ordered list of model candidates for the given request.
	// Input: free entries from the catalog + enriched metadata.
	// Output: ordered RankedEntry slice (index 0 = most preferred).
	Rank(req Request, entries []catalog.CatalogEntry, enriched *models.EnrichedCatalog) []RankedEntry
}

// Strict is implemented by strategies whose Rank output is a hard filter, not
// just an ordering: every candidate is guaranteed to meet the request's
// requirements. The fallback chain therefore never leaves the ranked list for a
// strict strategy (no unfiltered last-resort router), and it fails fast instead
// of retrying forever when the filter leaves nothing.
type Strict interface {
	Strategy
	// Explain is called when Rank returned no candidates although the catalog
	// holds free models. It returns a short, client-readable reason.
	Explain(req Request, entries []catalog.CatalogEntry, enriched *models.EnrichedCatalog) string
}
