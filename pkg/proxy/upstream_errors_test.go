package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestUpstreamErrorHelpers(t *testing.T) {
	c := func(code int, b string) upstreamFailure {
		return classifyUpstreamFailure(&Response{StatusCode: code, Body: []byte(b)})
	}
	if c(429, `{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}`) != failureNoCredit {
		t.Error("zai")
	}
	if c(400, `Model 'x' does not support tools.`) != failureUnsupportedModel {
		t.Error("tools")
	}
	if c(429, `Rate limit exceeded: free-models-per-day`) != failureDailyLimit {
		t.Error("daily")
	}
	if c(429, `tokens per minute`) != failureOther {
		t.Error("other")
	}
	if modelsArrayLimit(&Response{StatusCode: 400, Body: []byte(`'models' array must have 3 items or fewer.`)}) != 3 {
		t.Error("lim")
	}
	if lowerOpenRouterModelsLimit(3, 3) != 2 {
		t.Error("lower")
	}
	var body map[string]any
	json.Unmarshal([]byte(`{"messages":[{"role":"assistant","content":null,"reasoning_details":[1],"tool_calls":[{"id":"a","function":{"name":"f","arguments":null}}]}],"tools":[{"function":{"parameters":null,"x":{"default":null,"type":"string"}}}]}`), &body)
	out := sanitizeBody("gemini", copyMap(body))
	b, _ := json.Marshal(out)
	s := string(b)
	for _, bad := range []string{"null", "reasoning_details"} {
		if strings.Contains(s, bad) {
			t.Error(bad, s)
		}
	}
	if !strings.Contains(s, "skip_thought_signature_validator") {
		t.Error("sig")
	}
	if !strings.Contains(string(mustJSONBytes(body)), "reasoning_details") {
		t.Error("mutated original")
	}
	g := sanitizeBody("groq", copyMap(body))
	if strings.Contains(string(mustJSONBytes(g)), "reasoning_details") {
		t.Error("groq")
	}
}
func mustJSONBytes(v any) []byte { b, _ := json.Marshal(v); return b }
