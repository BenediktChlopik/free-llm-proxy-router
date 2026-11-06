package models

import "time"

// ModelInstance represents one provider's offering of a model.
type ModelInstance struct {
	ProviderID    string  `json:"provider_id"`
	ModelID       string  `json:"model_id"`
	ContextWindow int     `json:"context_window,omitempty"`
	IsFree        bool    `json:"is_free"`
	TierType      string  `json:"tier_type,omitempty"` // "free" | "credit"
}

// EnrichedModel is a model with tier, parameter count, capabilities,
// and cross-provider correlation information.
type EnrichedModel struct {
	Family       string          `json:"family"`
	Tier         ModelTier       `json:"tier"`
	Params       ParamScale      `json:"params"`
	Capabilities []Capability    `json:"capabilities,omitempty"`
	Instances    []ModelInstance `json:"instances"`

	// CanonicalID is the normalised model identifier used for cross-provider matching.
	CanonicalID string `json:"canonical_id"`
}

// EnrichedCatalog is the full set of enriched models persisted to enriched.json.
type EnrichedCatalog struct {
	Version   int             `json:"version"`
	UpdatedAt time.Time       `json:"updated_at"`
	Models    []EnrichedModel `json:"models"`
}



