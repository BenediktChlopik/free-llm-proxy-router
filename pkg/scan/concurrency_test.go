package scan

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
)

func TestDispatcherScansProvidersConcurrently(t *testing.T) {
	var active atomic.Int32
	var maxActive atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		current := active.Add(1)
		for previous := maxActive.Load(); current > previous && !maxActive.CompareAndSwap(previous, current); previous = maxActive.Load() {
		}
		defer active.Add(-1)
		time.Sleep(80 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer server.Close()

	providers := make([]config.ProviderConfig, 4)
	for i := range providers {
		providers[i] = config.ProviderConfig{ID: fmt.Sprintf("test-provider-%d", i), BaseURL: server.URL}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	started := time.Now()
	if _, err := NewDispatcherWithWorkers(4).ScanAll(ctx, providers); err != nil {
		t.Fatalf("ScanAll returned error: %v", err)
	}
	elapsed := time.Since(started)
	if got := maxActive.Load(); got < 2 {
		t.Fatalf("maximum concurrent scans = %d, want at least 2", got)
	}
	if elapsed >= 3*time.Second {
		t.Errorf("four scans took unexpectedly long: %s", elapsed)
	}
}
