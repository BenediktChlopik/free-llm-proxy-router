package reliability

import (
	"testing"
	"time"
)

func TestRecordCall_TracksLatencyAndSuccessPerModel(t *testing.T) {
	tr := New()
	tr.RecordCall("groq", "fast", true, 400*time.Millisecond)
	tr.RecordCall("groq", "fast", true, 600*time.Millisecond)
	tr.RecordCall("groq", "flaky", false, 0)
	tr.RecordCall("groq", "flaky", false, 0)

	lat, succ, calls := tr.Perf("groq", "fast")
	if calls != 2 || succ != 1 {
		t.Errorf("fast: calls=%d success=%v, want 2 / 1", calls, succ)
	}
	if lat < 400 || lat > 600 {
		t.Errorf("fast latency = %v, want between the two samples", lat)
	}
	_, succ, _ = tr.Perf("groq", "flaky")
	if succ > 0.7 {
		t.Errorf("flaky success = %v, want it well below 1 after two failures", succ)
	}
	if lat, _, _ := tr.Perf("groq", "flaky"); lat != 0 {
		t.Errorf("failures must not set latency, got %v", lat)
	}
}

func TestPerf_UnknownModelInheritsProviderSuccess(t *testing.T) {
	tr := New()
	for i := 0; i < 10; i++ {
		tr.Record("shaky", false)
	}
	lat, succ, calls := tr.Perf("shaky", "never-seen")
	if lat != 0 || calls != 0 || succ >= 1 {
		t.Errorf("got lat=%v success=%v calls=%d; want provider's low success and no latency", lat, succ, calls)
	}
	if _, succ, _ := tr.Perf("nobody", "x"); succ != 1 {
		t.Errorf("fully unknown success = %v, want 1", succ)
	}
}

func TestPerf_FailurePenaltyFades(t *testing.T) {
	tr := New()
	for i := 0; i < 5; i++ {
		tr.RecordCall("p", "m", false, 0)
	}
	_, fresh, _ := tr.Perf("p", "m")
	tr.mu.Lock()
	tr.stats[ModelKey("p", "m")].LastUpdated = time.Now().Add(-2 * time.Hour)
	tr.mu.Unlock()
	_, faded, _ := tr.Perf("p", "m")
	if !(faded > fresh+0.3) {
		t.Errorf("success after 2h = %v, fresh = %v; the penalty should have mostly faded", faded, fresh)
	}
}
