package proxy

import "testing"

func TestBuildOpenRouterRequestCapsModelsAtThree(t *testing.T) {
	raw := map[string]any{"model": "auto", "temperature": 0.2, "stream": true}
	models := []string{"model-1", "model-2", "model-3", "model-4", "model-5"}

	body := BuildOpenRouterRequest(raw, models)
	got, ok := body["models"].([]string)
	if !ok {
		t.Fatalf("models field has type %T, want []string", body["models"])
	}
	if len(got) != 3 {
		t.Fatalf("models has %d items, want 3: %v", len(got), got)
	}
	for i, want := range models[:3] {
		if got[i] != want {
			t.Errorf("models[%d] = %q, want %q", i, got[i], want)
		}
	}
	if _, exists := body["stream"]; exists {
		t.Error("stream should be removed for buffered fallback")
	}
	if raw["model"] != "auto" || raw["stream"] != true {
		t.Errorf("original request was mutated: %v", raw)
	}

	if len(BuildOpenRouterRequest(raw, nil)) != len(raw) {
		t.Error("empty fallback list should return the original request")
	}
}

func TestOpenRouterFallbackLimitConstantIsThree(t *testing.T) {
	if maxOpenRouterFallbackModels != 3 {
		t.Fatalf("maxOpenRouterFallbackModels = %d, want 3", maxOpenRouterFallbackModels)
	}
}
