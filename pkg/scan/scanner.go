// Package scan implements provider-specific free-model discovery.
package scan

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
)

// ProviderScanner is implemented by each provider's scanner.
type ProviderScanner interface {
	ScanFreeModels(ctx context.Context, cfg config.ProviderConfig) ([]catalog.CatalogEntry, error)
}

// Dispatcher selects the correct ProviderScanner for each provider.
type Dispatcher struct {
	client  *http.Client
	workers int
}

// NewDispatcher creates a Dispatcher with a default HTTP client.
func NewDispatcher() *Dispatcher {
	return NewDispatcherWithWorkers(8)
}

// NewDispatcherWithWorkers creates a dispatcher with a bounded number of
// provider scans running concurrently. A zero or negative value uses 8 workers.
func NewDispatcherWithWorkers(workers int) *Dispatcher {
	if workers <= 0 {
		workers = 8
	}
	return &Dispatcher{
		client:  &http.Client{Timeout: 30 * time.Second},
		workers: workers,
	}
}

// ScanAll runs all enabled provider scanners and merges the results.
func (d *Dispatcher) ScanAll(ctx context.Context, providers []config.ProviderConfig) (*catalog.Catalog, error) {
	type scanJob struct {
		index    int
		provider config.ProviderConfig
	}
	type scanResult struct {
		index      int
		providerID string
		entries    []catalog.CatalogEntry
		err        error
	}

	cat := &catalog.Catalog{Version: 1, UpdatedAt: time.Now()}
	jobs := make(chan scanJob)
	results := make(chan scanResult, len(providers))
	workers := d.workers
	if workers <= 0 {
		workers = 8
	}
	if workers > len(providers) {
		workers = len(providers)
	}
	if workers == 0 {
		return cat, nil
	}

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				if err := ctx.Err(); err != nil {
					results <- scanResult{index: job.index, providerID: job.provider.ID, err: err}
					continue
				}
				if !job.provider.IsEnabled() {
					if job.provider.APIKeyEnv != "" {
						log.Printf("scan: skipping %s — no API key (set %s in .env, or use --all-providers)", job.provider.ID, job.provider.APIKeyEnv)
					} else {
						log.Printf("scan: skipping %s — disabled (use --all-providers to scan anyway)", job.provider.ID)
					}
					continue
				}
				scanner := d.scannerFor(job.provider.ID)
				if scanner == nil {
					log.Printf("scan: no scanner registered for provider %q — skipping", job.provider.ID)
					continue
				}
				log.Printf("scan: scanning provider %s ...", job.provider.ID)
				entries, err := scanner.ScanFreeModels(ctx, job.provider)
				results <- scanResult{index: job.index, providerID: job.provider.ID, entries: entries, err: err}
			}
		}()
	}

	go func() {
		defer close(jobs)
		for i, provider := range providers {
			select {
			case <-ctx.Done():
				return
			case jobs <- scanJob{index: i, provider: provider}:
			}
		}
	}()
	go func() {
		wg.Wait()
		close(results)
	}()

	// Store results by config index so concurrent completion does not make the
	// catalog order nondeterministic between runs.
	ordered := make([][]catalog.CatalogEntry, len(providers))
	for result := range results {
		if result.err != nil {
			if result.err != context.Canceled && result.err != context.DeadlineExceeded {
				log.Printf("scan: %s error: %v — continuing", result.providerID, result.err)
			}
			continue
		}
		ordered[result.index] = result.entries
		log.Printf("scan: %s → %d entries", result.providerID, len(result.entries))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, entries := range ordered {
		cat.Entries = append(cat.Entries, entries...)
	}
	return cat, nil
}

// scannerFor returns the ProviderScanner for the given provider ID.
func (d *Dispatcher) scannerFor(providerID string) ProviderScanner {
	switch providerID {
	case "openrouter":
		return &OpenRouterScanner{Client: d.client}
	case "groq":
		return &GroqScanner{Client: d.client}
	case "gemini":
		return &GeminiScanner{Client: d.client}
	case "huggingface":
		return &HuggingFaceScanner{Client: d.client}
	case "github-models":
		return &GitHubScanner{Client: d.client}
	default:
		return &GenericScanner{Client: d.client}
	}
}

// buildAuthHeader returns the (header-name, value) pair for a provider.
func buildAuthHeader(cfg config.ProviderConfig) (string, string) {
	return cfg.ResolvedAuth()
}

// makeRequest creates a GET request with provider auth headers applied.
func makeRequest(ctx context.Context, url string, cfg config.ProviderConfig) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	h, v := buildAuthHeader(cfg)
	if v != "" && v != "Bearer " {
		req.Header.Set(h, v)
	}

	for k, val := range cfg.ExtraHeaders {
		req.Header.Set(k, val)
	}
	return req, nil
}
