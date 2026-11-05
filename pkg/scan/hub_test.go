package scan

import (
	"testing"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
)

func TestHubAllows(t *testing.T) {
	h, err := ParseHub([]byte(`{"providers":[
		{"slug":"google-gemini","models_free":["gemini-2.5-flash"]},
		{"slug":"groq","models_free":["openai/gpt-oss-120b"]},
		{"slug":"mistral","models_free":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		id, model      string
		allowed, known bool
	}{
		{"gemini", "models/gemini-2.5-flash", true, true},
		{"gemini", "gemini-2.5-pro", false, true},
		{"groq", "OpenAI/GPT-OSS-120B", true, true},
		{"groq", "llama-3.3-70b-versatile", false, true},
		{"mistral", "mistral-small", true, false},
		{"unknown", "x", true, false},
	}
	for _, c := range cases {
		a, k := h.Allows(config.ProviderConfig{ID: c.id}, c.model)
		if a != c.allowed || k != c.known {
			t.Errorf("%s/%s = (%v,%v), want (%v,%v)", c.id, c.model, a, k, c.allowed, c.known)
		}
	}
}
