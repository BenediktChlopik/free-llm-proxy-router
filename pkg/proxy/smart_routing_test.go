package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/ratelimit"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/reliability"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/strategy"
)

func TestRequestNeeds(t *testing.T) {
	cases := []struct {
		name         string
		req          Request
		tools, image bool
	}{
		{"plain", userReq(nil), false, false},
		{"empty tools array", userReq(map[string]any{"tools": []any{}}), false, false},
		{"tool definitions", userReq(map[string]any{"tools": []any{map[string]any{"type": "function"}}}), true, false},
		{"legacy functions", userReq(map[string]any{"functions": []any{map[string]any{"name": "f"}}}), true, false},
		{"tool result in history", Request{Messages: []map[string]any{{"role": "tool", "content": "42"}}}, true, false},
		{"earlier tool call", Request{Messages: []map[string]any{{"role": "assistant", "tool_calls": []any{map[string]any{"id": "1"}}}}}, true, false},
		{"image part", Request{Messages: []map[string]any{{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "what is this"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:..."}},
		}}}}, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tools, vision := requestNeeds(c.req)
			if tools != c.tools || vision != c.image {
				t.Errorf("requestNeeds = (tools %v, vision %v), want (%v, %v)", tools, vision, c.tools, c.image)
			}
		})
	}
}

func TestEstimateRequestTokensCountsToolSchemasAndParts(t *testing.T) {
	base := Request{Messages: []map[string]any{{"role": "user", "content": strings.Repeat("a", 400)}}}
	baseTokens := estimateRequestTokens(base)
	if baseTokens < 100 || baseTokens > 110 {
		t.Errorf("400 chars + overhead = %d tokens, want ~104", baseTokens)
	}

	withTools := base
	withTools.Raw = map[string]any{"tools": []any{map[string]any{"type": "function", "function": map[string]any{
		"name": "search", "description": strings.Repeat("d", 4000),
	}}}}
	if got := estimateRequestTokens(withTools); got < baseTokens+1000 {
		t.Errorf("tool schema not counted: %d vs base %d", got, baseTokens)
	}

	parts := Request{Messages: []map[string]any{{"role": "user", "content": []any{
		map[string]any{"type": "text", "text": strings.Repeat("b", 400)},
		map[string]any{"type": "image_url"},
	}}}}
	if got := estimateRequestTokens(parts); got < 100+imageTokenEstimate {
		t.Errorf("part-list content under-counted: %d", got)
	}
}

// hangUntilClientLeaves is an upstream that never answers. It reads the request
// body first: Go's server only notices a client disconnect (and cancels
// r.Context()) once the body has been consumed, and httptest.Server.Close would
// otherwise wait on the hung handler forever.
func hangUntilClientLeaves(w http.ResponseWriter, r *http.Request) {
	io.Copy(io.Discard, r.Body)
	<-r.Context().Done()
}

// strictChain builds a chain that uses the real smart strategy against one test server.
func strictChain(srvURL string, entries []catalog.CatalogEntry) *FallbackChain {
	provs := []config.ProviderConfig{{ID: "openrouter", BaseURL: srvURL}}
	seen := map[string]bool{"openrouter": true}
	for _, e := range entries {
		if !seen[e.ProviderID] {
			seen[e.ProviderID] = true
			provs = append(provs, config.ProviderConfig{ID: e.ProviderID, BaseURL: srvURL})
		}
	}
	return &FallbackChain{
		Cfg: &config.Config{
			Providers: provs,
			Fallback: config.FallbackConfig{
				MaxAttempts:         5,
				RetryBackoffInitial: time.Millisecond,
				RetryBackoffMax:     5 * time.Millisecond,
			},
		},
		Strategy:           &strategy.StrategySmart{},
		Catalog:            &catalog.Catalog{Entries: entries},
		RateLimiter:        ratelimit.NewGlobalTracker(),
		ReliabilityTracker: reliability.New(),
		HTTPClient:         &http.Client{},
	}
}

func rawWithPrompt(chars int, extra map[string]any) map[string]any {
	raw := map[string]any{"messages": []any{map[string]any{"role": "user", "content": strings.Repeat("x", chars)}}}
	for k, v := range extra {
		raw[k] = v
	}
	return raw
}

func reqFrom(raw map[string]any) Request {
	return Request{
		Raw:      raw,
		Messages: []map[string]any{raw["messages"].([]any)[0].(map[string]any)},
	}
}

func TestExecute_StrictStrategyFailsFastWhenNothingFits(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(`{"choices":[{"message":{"content":"should never be asked"}}]}`))
	}))
	defer srv.Close()

	fc := strictChain(srv.URL, []catalog.CatalogEntry{
		{ProviderID: "groq", ModelID: "small-70b", IsFree: true, ContextWindow: 8192},
	})
	// ~100k tokens of prompt: no known window holds it. No max_wait is set, so
	// without fail-fast this would retry forever.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	start := time.Now()
	resp, err := fc.Execute(ctx, reqFrom(rawWithPrompt(400_000, nil)))

	var noModel *NoEligibleModelError
	if resp != nil || !errors.As(err, &noModel) {
		t.Fatalf("want (nil, *NoEligibleModelError), got (%v, %v)", resp, err)
	}
	if !strings.Contains(noModel.Reason, "context") {
		t.Errorf("reason does not explain the context problem: %q", noModel.Reason)
	}
	if time.Since(start) > time.Second {
		t.Errorf("took %s; must fail fast, not retry", time.Since(start))
	}
	if calls.Load() != 0 {
		t.Errorf("provider was called %d times for a request no model can hold", calls.Load())
	}
}

func TestExecute_StrictStrategyFallbackStaysInEligibleSet(t *testing.T) {
	var mu sync.Mutex
	models := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		m, _ := body["model"].(string)
		mu.Lock()
		models[m]++
		mu.Unlock()
		http.Error(w, `{"error":{"message":"overloaded"}}`, http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	fc := strictChain(srv.URL, []catalog.CatalogEntry{
		{ProviderID: "p1", ModelID: "fits-70b", IsFree: true, ContextWindow: 131072, Metadata: map[string]any{"supports_tools": true}},
		{ProviderID: "p2", ModelID: "too-small-405b", IsFree: true, ContextWindow: 8192, Metadata: map[string]any{"supports_tools": true}},
		{ProviderID: "p3", ModelID: "no-tools-120b", IsFree: true, ContextWindow: 131072, Metadata: map[string]any{"supports_tools": false}},
	})
	fc.Cfg.Fallback.MaxWait = 150 * time.Millisecond

	// ~20k-token prompt carrying a tool: only fits-70b qualifies.
	raw := rawWithPrompt(80_000, map[string]any{"tools": []any{map[string]any{"type": "function"}}})
	_, err := fc.Execute(context.Background(), reqFrom(raw))
	if !errors.Is(err, ErrNoProviderAvailable) {
		t.Fatalf("err = %v, want ErrNoProviderAvailable after max_wait", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if models["fits-70b"] == 0 {
		t.Error("the eligible model was never tried")
	}
	for m := range models {
		if m != "fits-70b" {
			t.Errorf("fallback reached %q, which does not meet the request's requirements (all calls: %v)", m, models)
		}
	}
	if models["openrouter/free"] != 0 {
		t.Error("the unfiltered openrouter/free router was used by a strict strategy")
	}
}

func TestExecute_MaxWaitIsAHardCapOnInFlightCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(hangUntilClientLeaves))
	defer srv.Close()

	fc, _, _ := newTestChain(srv.URL, "gemini", "m") // no provider timeout, no client timeout
	fastBackoff(fc)
	fc.Cfg.Fallback.MaxWait = 150 * time.Millisecond

	start := time.Now()
	resp, err := fc.Execute(context.Background(), userReq(nil))
	elapsed := time.Since(start)

	if resp != nil || !errors.Is(err, ErrNoProviderAvailable) {
		t.Fatalf("want (nil, ErrNoProviderAvailable), got (%v, %v)", resp, err)
	}
	if elapsed < 140*time.Millisecond || elapsed > time.Second {
		t.Errorf("returned after %s, want ~150ms: max_wait must cut off the hanging provider call", elapsed)
	}
}

func TestExecute_ClientDisconnectStillBeatsMaxWait(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(hangUntilClientLeaves))
	defer srv.Close()

	fc, _, _ := newTestChain(srv.URL, "gemini", "m")
	fc.Cfg.Fallback.MaxWait = 5 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()

	_, err := fc.Execute(ctx, userReq(nil))
	if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrNoProviderAvailable) {
		t.Errorf("err = %v, want the client's own context error, not a max_wait timeout", err)
	}
}

// chatServer is a Server wired to a catalog and the real strategy registry.
func chatServer(cfg *config.Config, entries []catalog.CatalogEntry) *Server {
	reg := strategy.NewRegistry(reliability.New(), ratelimit.NewGlobalTracker(), "", "", 3, 5)
	return NewServer(cfg, &catalog.Catalog{Entries: entries}, reg, ratelimit.NewGlobalTracker(), nil, reliability.New())
}

func post(s *Server, body map[string]any) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	s.handleChatCompletions(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b)))
	return rec
}

func TestHandler_SmartStrategyReturns400WhenNoModelFits(t *testing.T) {
	cfg := &config.Config{
		Providers: []config.ProviderConfig{{ID: "groq", BaseURL: "http://127.0.0.1:1"}},
		Proxy:     config.ProxyConfig{CacheTTL: 0},
		Fallback:  config.FallbackConfig{MaxAttempts: 3},
	}
	s := chatServer(cfg, []catalog.CatalogEntry{{ProviderID: "groq", ModelID: "small-8b", IsFree: true, ContextWindow: 8192}})

	rec := post(s, map[string]any{
		"model":    "smart",
		"messages": []any{map[string]any{"role": "user", "content": strings.Repeat("y", 400_000)}},
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body %s", rec.Code, rec.Body)
	}
	var out struct {
		Error struct{ Type, Code, Message string } `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Error.Code != "no_eligible_model" || out.Error.Type != "invalid_request_error" || out.Error.Message == "" {
		t.Errorf("error body = %+v", out.Error)
	}
}

func TestHandler_DirectModelRouteHonoursMaxWait(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(hangUntilClientLeaves))
	defer srv.Close()

	cfg := &config.Config{
		Providers: []config.ProviderConfig{{ID: "groq", BaseURL: srv.URL}},
		Fallback:  config.FallbackConfig{MaxAttempts: 3, MaxWait: 150 * time.Millisecond},
	}
	s := chatServer(cfg, []catalog.CatalogEntry{{ProviderID: "groq", ModelID: "pinned-70b", IsFree: true}})

	start := time.Now()
	rec := post(s, map[string]any{
		"model":    "groq/pinned-70b",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body %s", rec.Code, rec.Body)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("answered after %s; max_wait (150ms) was not enforced on the direct route", elapsed)
	}
}
