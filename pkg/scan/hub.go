package scan

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
)

// DefaultHubURL is the community-maintained list of free LLM API providers.
const DefaultHubURL = "https://freellmapihub.com/providers.json"

// hubSlugAliases maps our provider IDs to hub slugs where they differ.
var hubSlugAliases = map[string]string{
	"gemini":       "google-gemini",
	"zai":          "zai-glm",
	"ovh-ai":       "ovhcloud-ai-endpoints",
	"qwen-alibaba": "alibaba-model-studio",
	"fireworks":    "fireworks-ai",
}

// HubList holds the free models per hub provider slug.
type HubList struct {
	// free maps slug -> set of normalized model IDs. A slug that is present with
	// a nil/empty set is not stored: providers whose models_free is null are absent.
	free map[string]map[string]struct{}
}

type hubResponse struct {
	Providers []struct {
		Slug       string   `json:"slug"`
		ModelsFree []string `json:"models_free"`
	} `json:"providers"`
}

// FetchHub downloads and parses providers.json.
func FetchHub(ctx context.Context, client *http.Client, url string) (*HubList, error) {
	if url == "" {
		url = DefaultHubURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: status %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, err
	}
	return ParseHub(body)
}

// ParseHub parses providers.json content.
func ParseHub(data []byte) (*HubList, error) {
	var hr hubResponse
	if err := json.Unmarshal(data, &hr); err != nil {
		return nil, fmt.Errorf("parsing providers.json: %w", err)
	}
	h := &HubList{free: map[string]map[string]struct{}{}}
	for _, p := range hr.Providers {
		if len(p.ModelsFree) == 0 {
			continue // hub does not enumerate free models for this provider
		}
		set := make(map[string]struct{}, len(p.ModelsFree))
		for _, m := range p.ModelsFree {
			set[normalizeModelID(m)] = struct{}{}
		}
		h.free[p.Slug] = set
	}
	if len(h.free) == 0 {
		return nil, fmt.Errorf("providers.json lists no free models")
	}
	return h, nil
}

// Allows reports whether a model should be probed. known is false when the hub
// has no model list for the provider, in which case allowed is true (no filter).
func (h *HubList) Allows(cfg config.ProviderConfig, modelID string) (allowed, known bool) {
	set, ok := h.free[hubSlug(cfg)]
	if !ok {
		return true, false
	}
	_, in := set[normalizeModelID(modelID)]
	return in, true
}

func hubSlug(cfg config.ProviderConfig) string {
	if cfg.HubSlug != "" {
		return cfg.HubSlug
	}
	if s, ok := hubSlugAliases[cfg.ID]; ok {
		return s
	}
	return cfg.ID
}

func normalizeModelID(id string) string {
	return strings.TrimPrefix(strings.ToLower(strings.TrimSpace(id)), "models/")
}
