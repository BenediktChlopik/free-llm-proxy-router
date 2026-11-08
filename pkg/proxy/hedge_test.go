package proxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/ratelimit"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/reliability"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/strategy"
)

// hedgeProvider is a fake upstream. It reads the request body first (so the
// server notices a client cancel), then waits `delay` — or until the client
// goes away — before answering with `status`.
type hedgeProvider struct {
	id        string
	delay     time.Duration
	status    int
	calls     atomic.Int32
	cancelled atomic.Int32
	srv       *httptest.Server
}

func newHedgeProvider(t *testing.T, id string, delay time.Duration, status int) *hedgeProvider {
	t.Helper()
	p := &hedgeProvider{id: id, delay: delay, status: status}
	p.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.calls.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done():
			p.cancelled.Add(1)
			return
		case <-time.After(p.delay):
		}
		w.WriteHeader(p.status)
		if p.status == http.StatusOK {
			_, _ = w.Write([]byte(`{"model":"` + p.id + `","choices":[{"message":{"content":"from ` + p.id + `"}}]}`))
		} else {
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
		}
	}))
	t.Cleanup(p.srv.Close)
	return p
}

func newHedgeChain(hedgeAfter time.Duration, maxInflight int, providers ...*hedgeProvider) *FallbackChain {
	cfg := &config.Config{Fallback: config.FallbackConfig{
		MaxAttempts:      len(providers),
		HedgeAfter:       hedgeAfter,
		HedgeMaxInflight: maxInflight,
	}}
	var ranked listStrategy
	cat := &catalog.Catalog{}
	for _, p := range providers {
		cfg.Providers = append(cfg.Providers, config.ProviderConfig{ID: p.id, BaseURL: p.srv.URL})
		cat.Entries = append(cat.Entries, catalog.CatalogEntry{ProviderID: p.id, ModelID: p.id, IsFree: true})
		ranked = append(ranked, strategy.RankedEntry{ProviderID: p.id, ModelID: p.id})
	}
	return &FallbackChain{
		Cfg:                cfg,
		Strategy:           ranked,
		Catalog:            cat,
		RateLimiter:        ratelimit.NewGlobalTracker(),
		ReliabilityTracker: reliability.New(),
		HTTPClient:         &http.Client{},
	}
}

func hedgeReq() Request {
	return Request{
		Raw:      map[string]any{"messages": []any{}},
		Messages: []map[string]any{{"role": "user", "content": "hi"}},
	}
}

// A provider that hangs is hedged: the next candidate starts after HedgeAfter
// and wins, long before the hung provider's own timeout would have fired.
func TestHedge_SlowProviderIsHedgedAndHedgeWins(t *testing.T) {
	slow := newHedgeProvider(t, "slow", 10*time.Second, http.StatusOK)
	fast := newHedgeProvider(t, "fast", 0, http.StatusOK)
	fc := newHedgeChain(80*time.Millisecond, 2, slow, fast)

	start := time.Now()
	resp, err := fc.executePass(context.Background(), hedgeReq(), nil)
	if err != nil {
		t.Fatalf("pass failed: %v", err)
	}
	if resp.UsedModel != "fast" {
		t.Errorf("UsedModel = %q, want fast", resp.UsedModel)
	}
	if el := time.Since(start); el < 70*time.Millisecond || el > 2*time.Second {
		t.Errorf("answered after %v; want ≈ the 80ms hedge delay", el)
	}
	// The hung call is cancelled once there is a winner.
	deadline := time.Now().Add(2 * time.Second)
	for slow.cancelled.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if slow.cancelled.Load() == 0 {
		t.Error("the slow call was not cancelled after the hedge won")
	}
}

// Hedging must not cancel the original call: if it answers before the hedge,
// the original still wins.
func TestHedge_SlowOriginalStillWinsIfFirst(t *testing.T) {
	first := newHedgeProvider(t, "first", 200*time.Millisecond, http.StatusOK)
	second := newHedgeProvider(t, "second", 10*time.Second, http.StatusOK)
	fc := newHedgeChain(50*time.Millisecond, 2, first, second)

	resp, err := fc.executePass(context.Background(), hedgeReq(), nil)
	if err != nil {
		t.Fatalf("pass failed: %v", err)
	}
	if resp.UsedModel != "first" {
		t.Errorf("UsedModel = %q, want first", resp.UsedModel)
	}
	if second.calls.Load() != 1 {
		t.Errorf("hedge calls = %d, want 1 (it should have been started)", second.calls.Load())
	}
}

// A failure never waits for the hedge timer: the next candidate starts at once.
func TestHedge_FailureFallsThroughImmediately(t *testing.T) {
	bad := newHedgeProvider(t, "bad", 0, http.StatusInternalServerError)
	good := newHedgeProvider(t, "good", 0, http.StatusOK)
	fc := newHedgeChain(5*time.Second, 2, bad, good)

	start := time.Now()
	resp, err := fc.executePass(context.Background(), hedgeReq(), nil)
	if err != nil {
		t.Fatalf("pass failed: %v", err)
	}
	if resp.UsedModel != "good" {
		t.Errorf("UsedModel = %q, want good", resp.UsedModel)
	}
	if el := time.Since(start); el > time.Second {
		t.Errorf("fallback took %v; a failure must not wait for hedge_after", el)
	}
}

// With hedging disabled the chain is strictly sequential: the second candidate
// is never started while the first is still running.
func TestHedge_DisabledIsSequential(t *testing.T) {
	first := newHedgeProvider(t, "first", 250*time.Millisecond, http.StatusOK)
	second := newHedgeProvider(t, "second", 0, http.StatusOK)
	fc := newHedgeChain(0, 2, first, second)

	resp, err := fc.executePass(context.Background(), hedgeReq(), nil)
	if err != nil {
		t.Fatalf("pass failed: %v", err)
	}
	if resp.UsedModel != "first" {
		t.Errorf("UsedModel = %q, want first", resp.UsedModel)
	}
	if second.calls.Load() != 0 {
		t.Errorf("second provider was called %d times with hedging off", second.calls.Load())
	}
}

// HedgeMaxInflight bounds the quota spent on hedges.
func TestHedge_MaxInflightIsRespected(t *testing.T) {
	a := newHedgeProvider(t, "a", 400*time.Millisecond, http.StatusOK)
	b := newHedgeProvider(t, "b", 400*time.Millisecond, http.StatusOK)
	c := newHedgeProvider(t, "c", 400*time.Millisecond, http.StatusOK)
	fc := newHedgeChain(40*time.Millisecond, 2, a, b, c)

	resp, err := fc.executePass(context.Background(), hedgeReq(), nil)
	if err != nil {
		t.Fatalf("pass failed: %v", err)
	}
	if resp.UsedModel != "a" {
		t.Errorf("UsedModel = %q, want a (started first, same latency)", resp.UsedModel)
	}
	if got := c.calls.Load(); got != 0 {
		t.Errorf("third provider started (%d calls) although max in-flight is 2", got)
	}
	if b.calls.Load() != 1 {
		t.Errorf("hedge provider calls = %d, want 1", b.calls.Load())
	}
}

// When every candidate fails, the pass reports it instead of hanging.
func TestHedge_AllFailReturnsError(t *testing.T) {
	a := newHedgeProvider(t, "a", 0, http.StatusInternalServerError)
	b := newHedgeProvider(t, "b", 30*time.Millisecond, http.StatusBadGateway)
	fc := newHedgeChain(10*time.Millisecond, 2, a, b)

	if _, err := fc.executePass(context.Background(), hedgeReq(), nil); err == nil {
		t.Fatal("expected an error when every provider fails")
	}
}

// A hedge that fails with a definitive status (404) still has its side effects
// applied even though another call wins the race.
func TestHedge_LoserFailureEffectsAreApplied(t *testing.T) {
	winner := newHedgeProvider(t, "winner", 150*time.Millisecond, http.StatusOK)
	gone := newHedgeProvider(t, "gone", 0, http.StatusNotFound)
	fc := newHedgeChain(30*time.Millisecond, 2, winner, gone)

	resp, err := fc.executePass(context.Background(), hedgeReq(), nil)
	if err != nil {
		t.Fatalf("pass failed: %v", err)
	}
	if resp.UsedModel != "winner" {
		t.Fatalf("UsedModel = %q, want winner", resp.UsedModel)
	}
	if fc.Catalog.Find("gone", "gone") != nil {
		t.Error("404 model should have been excluded from the catalog")
	}
}
