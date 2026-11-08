// Package reliability tracks per-provider success rates using an
// exponential moving average (EMA).
package reliability

import (
	"math"
	"sync"
	"time"
)

const (
	// emaAlpha is the smoothing factor for the EMA.
	// Higher = faster reaction to recent events, lower = more stable.
	emaAlpha = 0.1

	// defaultSuccessRate is the starting EMA for unknown providers.
	defaultSuccessRate = 1.0

	// Per-model observations are sparse, so they react faster than provider ones.
	callSuccessAlpha = 0.2
	callLatencyAlpha = 0.3

	// penaltyHalfLife is how fast an old failure record stops counting against a
	// model. A demoted model gets no new traffic, so without this decay a single
	// bad hour would exile it for good.
	penaltyHalfLife = 30 * time.Minute
)

// ProviderStats holds the reliability stats for one provider.
type ProviderStats struct {
	ProviderID  string  `json:"provider_id"`
	SuccessRate float64 `json:"success_rate"` // 0.0 – 1.0 EMA
	// LatencyMs is the EMA of successful response times (per-model entries only).
	LatencyMs   float64   `json:"latency_ms,omitempty"`
	TotalCalls  int64     `json:"total_calls"`
	LastUpdated time.Time `json:"last_updated"`
}

// Tracker tracks per-provider reliability using EMA.
type Tracker struct {
	mu    sync.RWMutex
	stats map[string]*ProviderStats
}

// New creates an empty reliability Tracker.
func New() *Tracker {
	return &Tracker{stats: make(map[string]*ProviderStats)}
}

// Record adds an observation (success or failure) for the given provider.
func (t *Tracker) Record(providerID string, success bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	s := t.getOrCreate(providerID)
	s.TotalCalls++
	s.LastUpdated = time.Now()

	val := 0.0
	if success {
		val = 1.0
	}
	s.SuccessRate = emaAlpha*val + (1-emaAlpha)*s.SuccessRate
}

// ModelKey is the tracker key for one provider/model pair. Per-model entries
// live next to the per-provider ones in the same map (and the same file).
func ModelKey(providerID, modelID string) string { return providerID + "/" + modelID }

// RecordCall records one finished call to a specific model. A success updates
// the model's latency EMA; a failure lowers its success rate and leaves the
// latency alone (a timeout says nothing about how fast it normally answers).
func (t *Tracker) RecordCall(providerID, modelID string, success bool, latency time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()

	s := t.getOrCreate(ModelKey(providerID, modelID))
	s.TotalCalls++
	s.LastUpdated = time.Now()
	if success {
		s.SuccessRate = callSuccessAlpha*1.0 + (1-callSuccessAlpha)*s.SuccessRate
		if ms := float64(latency) / float64(time.Millisecond); ms > 0 {
			if s.LatencyMs <= 0 {
				s.LatencyMs = ms
			} else {
				s.LatencyMs = callLatencyAlpha*ms + (1-callLatencyAlpha)*s.LatencyMs
			}
		}
		return
	}
	s.SuccessRate = (1 - callSuccessAlpha) * s.SuccessRate
}

// Perf reports how a model has been performing: its typical response time in
// milliseconds (0 = never measured), its success rate, and the number of calls
// behind those numbers. Failure records fade with penaltyHalfLife. A model with
// no history of its own inherits its provider's success rate.
func (t *Tracker) Perf(providerID, modelID string) (latencyMs, successRate float64, calls int64) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if s, ok := t.stats[ModelKey(providerID, modelID)]; ok {
		success := s.SuccessRate
		if age := time.Since(s.LastUpdated); age > 0 && success < 1 {
			success = 1 - (1-success)*math.Pow(0.5, float64(age)/float64(penaltyHalfLife))
		}
		return s.LatencyMs, success, s.TotalCalls
	}
	if s, ok := t.stats[providerID]; ok {
		return 0, s.SuccessRate, 0
	}
	return 0, defaultSuccessRate, 0
}

// ShouldCooldown returns true when a provider's EMA failure rate exceeds 50%
// over at least 5 observed requests.
func (t *Tracker) ShouldCooldown(providerID string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	s, ok := t.stats[providerID]
	if !ok || s.TotalCalls < 5 {
		return false
	}
	return s.SuccessRate < 0.5
}

// SuccessRate returns the current EMA success rate for a provider.
// Returns defaultSuccessRate for unknown providers.
func (t *Tracker) SuccessRate(providerID string) float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if s, ok := t.stats[providerID]; ok {
		return s.SuccessRate
	}
	return defaultSuccessRate
}

// All returns a snapshot of all provider stats.
func (t *Tracker) All() []ProviderStats {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]ProviderStats, 0, len(t.stats))
	for _, s := range t.stats {
		out = append(out, *s)
	}
	return out
}

// LoadFrom overwrites the tracker state with the given stats slice.
func (t *Tracker) LoadFrom(stats []ProviderStats) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, s := range stats {
		cp := s
		t.stats[s.ProviderID] = &cp
	}
}

func (t *Tracker) getOrCreate(providerID string) *ProviderStats {
	if s, ok := t.stats[providerID]; ok {
		return s
	}
	s := &ProviderStats{
		ProviderID:  providerID,
		SuccessRate: defaultSuccessRate,
	}
	t.stats[providerID] = s
	return s
}
