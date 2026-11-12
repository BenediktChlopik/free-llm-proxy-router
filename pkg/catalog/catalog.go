// Package catalog manages the local free-model catalog.
package catalog

import (
	"encoding/json"
	"slices"
	"sync"
	"time"
)

// CatalogEntry is a single model entry discovered at runtime.
type CatalogEntry struct {
	ProviderID    string    `json:"provider_id"`
	ModelID       string    `json:"model_id"`
	IsFree        bool      `json:"is_free"`
	TierType      string    `json:"tier_type"` // "free" | "credit"
	ContextWindow int       `json:"context_window,omitempty"`
	DiscoveredAt  time.Time `json:"discovered_at"`
	// NeedsReverification is set when a 429 is received on a model marked free.
	NeedsReverification bool `json:"needs_reverification,omitempty"`
	// LastVerifiedAt is the last time this entry was confirmed free.
	LastVerifiedAt time.Time `json:"last_verified_at,omitempty"`
	// Metadata holds provider-specific fields (e.g. pricing, max_tokens).
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Catalog holds all discovered free-model entries.
type Catalog struct {
	mu        sync.RWMutex   `json:"-"`
	Version   int            `json:"version"`
	UpdatedAt time.Time      `json:"updated_at"`
	Entries   []CatalogEntry `json:"entries"`
	// Blocklist contains "provider_id/model_id" pairs that are permanently excluded.
	// Models on this list are removed after every scan (non-existent, chat-incompatible, etc).
	Blocklist []string `json:"blocklist,omitempty"`
}

// FilterBlocklisted removes any entry whose "provider_id/model_id" appears in Blocklist.
func (c *Catalog) FilterBlocklisted() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.Blocklist) == 0 {
		return
	}
	blocked := make(map[string]struct{}, len(c.Blocklist))
	for _, b := range c.Blocklist {
		blocked[b] = struct{}{}
	}
	var kept []CatalogEntry
	for _, e := range c.Entries {
		if _, bad := blocked[e.ProviderID+"/"+e.ModelID]; !bad {
			kept = append(kept, e)
		}
	}
	c.Entries = kept
}

// Block adds a "provider_id/model_id" key to the blocklist if not already present.
func (c *Catalog) Block(providerID, modelID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.blockLocked(providerID, modelID)
}

func (c *Catalog) blockLocked(providerID, modelID string) {
	key := providerID + "/" + modelID
	if slices.Contains(c.Blocklist, key) {
		return
	}
	c.Blocklist = append(c.Blocklist, key)
}

// ByProvider returns all entries for the given provider.
func (c *Catalog) ByProvider(providerID string) []CatalogEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []CatalogEntry
	for _, e := range c.Entries {
		if e.ProviderID == providerID {
			out = append(out, e)
		}
	}
	return out
}

// Find returns the entry for a given provider+model pair, or nil.
func (c *Catalog) Find(providerID, modelID string) *CatalogEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for i := range c.Entries {
		e := c.Entries[i]
		if e.ProviderID == providerID && e.ModelID == modelID {
			copy := e
			return &copy
		}
	}
	return nil
}

// EntriesSnapshot returns a point-in-time copy of the catalog entries.
func (c *Catalog) EntriesSnapshot() []CatalogEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]CatalogEntry(nil), c.Entries...)
}

// BlocklistSnapshot returns a point-in-time copy of the permanent exclusions.
func (c *Catalog) BlocklistSnapshot() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]string(nil), c.Blocklist...)
}

// MarkNeedsReverification flags a model for reverification (e.g. after a 429).
func (c *Catalog) MarkNeedsReverification(providerID, modelID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.Entries {
		if c.Entries[i].ProviderID == providerID && c.Entries[i].ModelID == modelID {
			c.Entries[i].NeedsReverification = true
			return
		}
	}
}

// FreeEntries returns only entries marked as free.
func (c *Catalog) FreeEntries() []CatalogEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []CatalogEntry
	for _, e := range c.Entries {
		if e.IsFree {
			out = append(out, e)
		}
	}
	return out
}

// KeepUnscanned copies entries from prev for every provider that has no
// entries in c (scan failed, key missing, provider down). It returns the IDs
// of the providers that were carried forward. Used by scheduled scans that may
// not have keys for every provider, so a partial scan never wipes models.
func (c *Catalog) KeepUnscanned(prev *Catalog) []string {
	if prev == nil {
		return nil
	}
	previousEntries := prev.EntriesSnapshot()
	c.mu.Lock()
	defer c.mu.Unlock()
	scanned := make(map[string]bool)
	for _, e := range c.Entries {
		scanned[e.ProviderID] = true
	}
	var kept []string
	seen := make(map[string]bool)
	for _, e := range previousEntries {
		if scanned[e.ProviderID] {
			continue
		}
		if !seen[e.ProviderID] {
			seen[e.ProviderID] = true
			kept = append(kept, e.ProviderID)
		}
		c.Entries = append(c.Entries, e)
	}
	return kept
}

// ExcludeModel permanently removes a model from free routing and future scans.
// This is used for definitive upstream responses such as HTTP 402 or 404.
func (c *Catalog) ExcludeModel(providerID, modelID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	changed := false
	kept := c.Entries[:0]
	for _, e := range c.Entries {
		if e.ProviderID == providerID && e.ModelID == modelID {
			changed = true
			continue
		}
		kept = append(kept, e)
	}
	c.Entries = kept
	if changed {
		c.blockLocked(providerID, modelID)
	}
	return changed
}

// RecordProbeResult applies a completed catalog probe safely while requests
// and other probes may be reading the catalog concurrently.
func (c *Catalog) RecordProbeResult(providerID, modelID string, accessible bool, statusCode int, verifiedAt time.Time) bool {
	if accessible {
		c.mu.Lock()
		defer c.mu.Unlock()
		for i := range c.Entries {
			e := &c.Entries[i]
			if e.ProviderID == providerID && e.ModelID == modelID {
				e.NeedsReverification = false
				e.LastVerifiedAt = verifiedAt
				return true
			}
		}
		return false
	}
	if statusCode == 402 || statusCode == 404 {
		return c.ExcludeModel(providerID, modelID)
	}
	return false
}

// MarshalJSON takes a read lock so a concurrent runtime exclusion cannot race
// with catalog persistence.
func (c *Catalog) MarshalJSON() ([]byte, error) {
	type catalogJSON Catalog
	c.mu.RLock()
	defer c.mu.RUnlock()
	return json.Marshal((*catalogJSON)(c))
}
