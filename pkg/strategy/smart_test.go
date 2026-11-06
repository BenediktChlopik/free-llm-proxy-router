package strategy

import (
	"strings"
	"testing"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
)

func tools(v bool) map[string]any { return map[string]any{"supports_tools": v} }

func ids(r []RankedEntry) []string {
	out := make([]string, len(r))
	for i, e := range r {
		out[i] = e.ModelID
	}
	return out
}

func smartEntries() []catalog.CatalogEntry {
	return []catalog.CatalogEntry{
		{ProviderID: "a", ModelID: "big-70b", IsFree: true, ContextWindow: 131072, Metadata: tools(true)},
		{ProviderID: "a", ModelID: "mid-32b", IsFree: true, ContextWindow: 32768, Metadata: tools(true)},
		{ProviderID: "a", ModelID: "huge-405b", IsFree: true, ContextWindow: 8192, Metadata: tools(true)},
		{ProviderID: "a", ModelID: "tiny-1b", IsFree: true, ContextWindow: 131072, Metadata: tools(true)},
		{ProviderID: "a", ModelID: "notools-120b", IsFree: true, ContextWindow: 131072, Metadata: tools(false)},
		{ProviderID: "a", ModelID: "unknown-ctx-200b", IsFree: true, Metadata: tools(true)},
		{ProviderID: "a", ModelID: "paid-500b", IsFree: false, ContextWindow: 1000000, Metadata: tools(true)},
	}
}

func TestSmart_FiltersByContextThenLargestFirst(t *testing.T) {
	// ~20k prompt x 1.15 + 1000 reply = 24,000 tokens: rules out the 8k models,
	// including the biggest one (huge-405b) and the unknown-window model.
	req := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000}
	got := ids((&StrategySmart{}).Rank(req, smartEntries(), nil))
	want := []string{"notools-120b", "big-70b", "mid-32b", "tiny-1b"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("order = %v, want %v", got, want)
	}
}

func TestSmart_ToolRequestExcludesNonToolModels(t *testing.T) {
	req := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000, NeedsTools: true}
	got := ids((&StrategySmart{}).Rank(req, smartEntries(), nil))
	want := []string{"big-70b", "mid-32b", "tiny-1b"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("order = %v, want %v (notools-120b must be filtered out)", got, want)
	}
}

func TestSmart_ShortRequestKeepsLargestModelsRegardlessOfWindow(t *testing.T) {
	req := Request{EstimatedPromptTokens: 100, MaxTokens: 256}
	got := ids((&StrategySmart{}).Rank(req, smartEntries(), nil))
	if len(got) == 0 || got[0] != "huge-405b" {
		t.Errorf("order = %v, want huge-405b first (it fits a short request)", got)
	}
}

func TestSmart_EqualSizePrefersSmallerSufficientWindow(t *testing.T) {
	entries := []catalog.CatalogEntry{
		{ProviderID: "long", ModelID: "same-70b", IsFree: true, ContextWindow: 1000000},
		{ProviderID: "short", ModelID: "same-70b", IsFree: true, ContextWindow: 65536},
	}
	got := (&StrategySmart{}).Rank(Request{EstimatedPromptTokens: 1000, MaxTokens: 256}, entries, nil)
	if len(got) != 2 || got[0].ProviderID != "short" {
		t.Errorf("got %+v, want the smaller-window provider first so long-context capacity stays free", got)
	}
}

func TestSmart_UnknownWindowIsAssumedSmallButConfigurable(t *testing.T) {
	entries := []catalog.CatalogEntry{{ProviderID: "a", ModelID: "unknown-70b", IsFree: true, Metadata: tools(true)}}
	big := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000}
	if got := (&StrategySmart{}).Rank(big, entries, nil); len(got) != 0 {
		t.Errorf("unverified model accepted for a 24k request: %v", ids(got))
	}
	big.Options = map[string]any{"unknown_context_tokens": 65536}
	if got := (&StrategySmart{}).Rank(big, entries, nil); len(got) != 1 {
		t.Errorf("model rejected despite unknown_context_tokens=65536: %v", ids(got))
	}
}

func TestSmart_ReplyBudgetUsesMaxTokensCapAndReserve(t *testing.T) {
	// No max_tokens: the 1024 default reserve applies. 7000*1.15 = 8050 + 1024 > 8192.
	req := Request{EstimatedPromptTokens: 7000}
	if need := smartRequiredContext(req); need != 8050+1024 {
		t.Errorf("need = %d, want %d", need, 8050+1024)
	}
	// A configured cap replaces a larger client max_tokens, as the proxy clamps it.
	req = Request{EstimatedPromptTokens: 1000, MaxTokens: 100000, Options: map[string]any{"max_tokens_cap": 512}}
	if need := smartRequiredContext(req); need != 1150+512 {
		t.Errorf("need = %d, want %d", need, 1150+512)
	}
}

func TestSmart_VisionRequestNeedsVisionModel(t *testing.T) {
	entries := []catalog.CatalogEntry{
		{ProviderID: "a", ModelID: "text-70b", IsFree: true, ContextWindow: 131072},
		{ProviderID: "a", ModelID: "eyes-11b", IsFree: true, ContextWindow: 131072, Metadata: map[string]any{"supports_vision": true}},
	}
	got := ids((&StrategySmart{}).Rank(Request{EstimatedPromptTokens: 900, MaxTokens: 256, NeedsVision: true}, entries, nil))
	if len(got) != 1 || got[0] != "eyes-11b" {
		t.Errorf("got %v, want only eyes-11b", got)
	}
}

func TestSmart_IsStrictAndExplainsEmptyResult(t *testing.T) {
	var s Strategy = &StrategySmart{}
	strict, ok := s.(Strict)
	if !ok {
		t.Fatal("StrategySmart must implement Strict so the chain stays inside its filter")
	}
	req := Request{EstimatedPromptTokens: 500000, MaxTokens: 1000, NeedsTools: true}
	entries := smartEntries()
	if got := s.Rank(req, entries, nil); len(got) != 0 {
		t.Fatalf("expected no eligible model, got %v", ids(got))
	}
	msg := strict.Explain(req, entries, nil)
	for _, want := range []string{"tool calling", "context", "lack tool calling", "too small a context window", "131072"} {
		if !strings.Contains(msg, want) {
			t.Errorf("explanation %q missing %q", msg, want)
		}
	}
}

func TestSmart_DoesNotMutateInput(t *testing.T) {
	entries := smartEntries()
	first := entries[0].ModelID
	(&StrategySmart{}).Rank(Request{EstimatedPromptTokens: 100, MaxTokens: 100}, entries, nil)
	if entries[0].ModelID != first {
		t.Error("Rank reordered the caller's slice")
	}
}
