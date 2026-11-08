package strategy

import (
	"strings"
	"testing"
)

type perfRow struct {
	latency, success float64
	calls            int64
}

func perfOf(m map[string]perfRow) PerfLookup {
	return func(provider, model string) (float64, float64, int64) {
		if r, ok := m[model]; ok {
			return r.latency, r.success, r.calls
		}
		return 0, 1, 0
	}
}

func rank(req Request) string {
	return strings.Join(ids((&StrategySmart{}).Rank(req, smartEntries(), nil)), ",")
}

func TestSmartPerf_NoHistoryKeepsSizeOrder(t *testing.T) {
	base := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000}
	withPerf := base
	withPerf.Perf = perfOf(nil)
	if a, b := rank(base), rank(withPerf); a != b {
		t.Errorf("order changed without history: %s vs %s", a, b)
	}
}

func TestSmartPerf_FastModelBeatsSlowerBiggerOne(t *testing.T) {
	// big-70b answers in 12s; mid-32b (about half the size) in 0.4s.
	req := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000, Perf: perfOf(map[string]perfRow{
		"big-70b": {12000, 1, 20},
		"mid-32b": {400, 1, 20},
	})}
	got := rank(req)
	if !strings.HasPrefix(got, "notools-120b,mid-32b,big-70b") && !strings.HasPrefix(got, "mid-32b,notools-120b,big-70b") {
		t.Errorf("order = %s; fast mid-32b should rank ahead of slow big-70b", got)
	}
	if strings.Index(got, "mid-32b") > strings.Index(got, "big-70b") {
		t.Errorf("order = %s; mid-32b must come before big-70b", got)
	}
}

func TestSmartPerf_FailingModelIsDemoted(t *testing.T) {
	req := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000, NeedsTools: true, Perf: perfOf(map[string]perfRow{
		"big-70b": {600, 0.2, 10}, // fast but mostly failing
		"mid-32b": {600, 1.0, 10},
	})}
	if got := rank(req); !strings.HasPrefix(got, "mid-32b,") {
		t.Errorf("order = %s; reliable mid-32b should lead the failing big-70b", got)
	}
}

func TestSmartPerf_WeightsCanDisableThePreference(t *testing.T) {
	req := Request{EstimatedPromptTokens: 20000, MaxTokens: 1000, NeedsTools: true,
		Options: map[string]any{"speed_weight": 0.0, "failure_weight": 0.0},
		Perf: perfOf(map[string]perfRow{
			"big-70b": {30000, 0.1, 10},
		})}
	if got := rank(req); !strings.HasPrefix(got, "big-70b,") {
		t.Errorf("order = %s; with both weights 0 the largest model must lead", got)
	}
}
