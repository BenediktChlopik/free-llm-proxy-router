package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
)

func TestFallbackChain_429SkipsProviderModels(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		
		// The first model of the provider 429s.
		if body["model"] == "limited-model-1" {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		
		// If the logic is correct, the second model of the SAME provider 
		// should NOT be called because the whole provider is now skipped.
		if body["model"] == "limited-model-2" {
			w.Write([]byte(`{"model":"limited-model-2","choices":[{"message":{"content":"should not be reached"}}]}`))
			return
		}
		
		w.Write([]byte(`{"model":"other-provider-model","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	// Another provider
	otherSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"model":"other-provider-model","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer otherSrv.Close()

	fc, _, _ := newTestChain(srv.URL, "p1", "limited-model-1")
	fc.Cfg.Providers = []config.ProviderConfig{
		{ID: "p1", BaseURL: srv.URL},
		{ID: "p2", BaseURL: otherSrv.URL},
	}
	
	fc.Strategy = listStrategy{
		{ProviderID: "p1", ModelID: "limited-model-1"},
		{ProviderID: "p1", ModelID: "limited-model-2"},
		{ProviderID: "p2", ModelID: "other-provider-model"},
	}
	
	resp, err := fc.Execute(context.Background(), Request{Messages: []map[string]any{{"role": "user", "content": "hi"}}})
	
	if err != nil {
		t.Fatalf("expected success from p2, got err: %v", err)
	}
	if resp.UsedModel != "other-provider-model" {
		t.Errorf("expected other-provider-model, got %s", resp.UsedModel)
	}
	if callCount != 2 {
		t.Errorf("expected 2 calls (limited-model-1 and other-provider-model), got %d", callCount)
	}
}
