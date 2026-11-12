// free-llm-scan: Free model discovery and catalog management.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/spf13/cobra"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/scan"
)

var (
	cfgFile     string
	catalogPath string
)

// loadConfig loads the config and applies the --catalog override.
func loadConfig() (*config.Config, error) {
	cfg, err := config.Load(cfgFile)
	if err != nil {
		return nil, fmt.Errorf("loading config: %w", err)
	}
	if catalogPath != "" {
		cfg.Catalog.Path = catalogPath
	}
	return cfg, nil
}

func main() {
	root := &cobra.Command{
		Use:   "free-llm-scan",
		Short: "Discover and manage free LLM model catalog",
	}
	root.PersistentFlags().StringVarP(&cfgFile, "config", "c", "", "config file path")
	root.PersistentFlags().StringVar(&catalogPath, "catalog", "", "catalog.json path (overrides catalog.path in config)")

	root.AddCommand(
		updateCmd(),
		probeCmd(),
		refreshLLMCmd(),
		exportMDCmd(),
	)

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// updateCmd scans all enabled providers and writes catalog.json.
func updateCmd() *cobra.Command {
	var keepUnscanned, allProviders bool
	var workers int
	cmd := &cobra.Command{
		Use:   "update",
		Short: "Scan all enabled providers and update catalog.json",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}

			// Load existing catalog to preserve blocklist across scans.
			existing, _ := catalog.Load(cfg.Catalog.Path)

			ctx, cancel := context.WithTimeout(cmd.Context(), 2*time.Minute)
			defer cancel()

			providers := cfg.Providers
			if allProviders {
				// Discovery is read-only, so scan providers that are off for routing too.
				providers = make([]config.ProviderConfig, len(cfg.Providers))
				copy(providers, cfg.Providers)
				for i := range providers {
					providers[i].ForceEnabled = true
				}
			}

			dispatcher := scan.NewDispatcherWithWorkers(workers)
			cat, err := dispatcher.ScanAll(ctx, providers)
			if err != nil {
				return fmt.Errorf("scanning providers: %w", err)
			}

			// Carry forward the blocklist from the previous catalog.
			if existing != nil {
				cat.Blocklist = existing.Blocklist
			}
			cat.FilterBlocklisted()

			if keepUnscanned {
				if kept := cat.KeepUnscanned(existing); len(kept) > 0 {
					log.Printf("scan: no results from %v — kept their previous entries", kept)
				}
			}
			// Carry-forward must never resurrect a model excluded by a previous
			// definitive 402/404 response.
			cat.FilterBlocklisted()

			// Correlate cross-provider families
			correlated := scan.Correlate(cat.Entries)
			log.Printf("scan: found %d model families across %d entries", len(correlated), len(cat.Entries))

			if err := catalog.Save(cat, cfg.Catalog.Path); err != nil {
				return fmt.Errorf("saving catalog: %w", err)
			}
			log.Printf("scan: wrote %d entries to %s", len(cat.Entries), cfg.Catalog.Path)

			if err := catalog.PushAfterScan(cfg.Catalog.Path, catalog.GitSyncConfig(cfg.Catalog.GitSync)); err != nil {
				log.Printf("gitsync: push failed (non-fatal): %v", err)
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&keepUnscanned, "keep-unscanned", false,
		"keep previous entries for providers that returned nothing (missing key, outage)")
	cmd.Flags().BoolVar(&allProviders, "all-providers", false,
		"also scan providers disabled by missing .env keys or local-provider toggles")
	cmd.Flags().IntVar(&workers, "workers", 8, "number of provider scans to run concurrently")
	return cmd
}

// probeCmd probes models that need reverification.
func probeCmd() *cobra.Command {
	var workers int
	cmd := &cobra.Command{
		Use:   "probe",
		Short: "Probe all models flagged as needing reverification",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			cat, err := catalog.Load(cfg.Catalog.Path)
			if err != nil {
				return fmt.Errorf("loading catalog: %w", err)
			}

			pending := catalog.PendingReverification(cat)
			if len(pending) == 0 {
				log.Printf("probe: no models need reverification")
				return nil
			}

			log.Printf("probe: checking all %d pending models with up to %d workers...", len(pending), workers)

			prober := &scan.Prober{Client: &http.Client{Timeout: 20 * time.Second}}
			providerByID := make(map[string]config.ProviderConfig, len(cfg.Providers))
			for _, provider := range cfg.Providers {
				providerByID[provider.ID] = provider
			}
			type probeJob struct {
				entry    catalog.CatalogEntry
				provider config.ProviderConfig
			}
			type probeResult struct {
				entry  catalog.CatalogEntry
				result scan.ProbeResult
			}
			jobs := make(chan probeJob)
			results := make(chan probeResult, len(pending))
			workerCount := min(max(workers, 1), len(pending))

			ctx := cmd.Context()
			var wg sync.WaitGroup
			for i := 0; i < workerCount; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for job := range jobs {
						results <- probeResult{entry: job.entry, result: prober.Probe(ctx, job.entry, job.provider)}
					}
				}()
			}
			go func() {
				defer close(jobs)
				for _, entry := range pending {
					provider, ok := providerByID[entry.ProviderID]
					if !ok || !provider.IsEnabled() {
						continue
					}
					select {
					case <-ctx.Done():
						return
					case jobs <- probeJob{entry: entry, provider: provider}:
					}
				}
			}()
			go func() {
				wg.Wait()
				close(results)
			}()

			updated := 0
			for item := range results {
				result := item.result
				if result.IsAccessible {
					cat.RecordProbeResult(item.entry.ProviderID, item.entry.ModelID, true, result.StatusCode, time.Now())
					log.Printf("probe: %s/%s ✓ still free", item.entry.ProviderID, item.entry.ModelID)
					updated++
				} else if result.StatusCode == http.StatusPaymentRequired || result.StatusCode == http.StatusNotFound {
					cat.ExcludeModel(item.entry.ProviderID, item.entry.ModelID)
					log.Printf("probe: %s/%s ✗ excluded from catalog (status %d)", item.entry.ProviderID, item.entry.ModelID, result.StatusCode)
					updated++
				} else {
					message := result.ErrMsg
					if len(message) > 160 {
						message = message[:160] + "…"
					}
					log.Printf("probe: %s/%s ? status %d (keeping as free)%s", item.entry.ProviderID, item.entry.ModelID, result.StatusCode, func() string {
						if message == "" {
							return ""
						}
						return ": " + message
					}())
				}
			}

			if err := ctx.Err(); err != nil {
				return err
			}
			if err := catalog.Save(cat, cfg.Catalog.Path); err != nil {
				return fmt.Errorf("saving catalog: %w", err)
			}
			log.Printf("probe: updated %d entries", updated)
			return nil
		},
	}
	cmd.Flags().IntVar(&workers, "workers", 12, "number of concurrent model probes")
	return cmd
}

// refreshLLMCmd uses an LLM to diff the catalog against live provider info.
func refreshLLMCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "refresh-llm",
		Short: "Use LLM to check for free tier changes and update catalog diff",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			cat, err := catalog.Load(cfg.Catalog.Path)
			if err != nil {
				return fmt.Errorf("loading catalog: %w", err)
			}

			promptFile := cfg.Refresh.PromptFile
			if promptFile == "" {
				promptFile = config.FindDataFile("refresh-prompt.txt", cfgFile)
			}

			refresher := &scan.LLMRefresher{
				Client:     &http.Client{Timeout: 60 * time.Second},
				PromptFile: promptFile,
				Cfg:        &cfg.Refresh,
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()

			diff, err := refresher.Refresh(ctx, cat, cfg.Providers)
			if err != nil {
				return fmt.Errorf("LLM refresh: %w", err)
			}

			log.Printf("refresh: changed=%d, added=%d, removed_confidence=%d",
				len(diff.Changed), len(diff.Added), len(diff.RemovedConfidence))

			scan.MergeDiff(cat, diff)

			if err := catalog.Save(cat, cfg.Catalog.Path); err != nil {
				return fmt.Errorf("saving catalog: %w", err)
			}
			log.Printf("refresh: catalog updated at %s", cfg.Catalog.Path)
			return nil
		},
	}
}

// exportMDCmd renders the catalog as a Markdown page of free models.
func exportMDCmd() *cobra.Command {
	var out string
	cmd := &cobra.Command{
		Use:   "export-md",
		Short: "Render the catalog as a Markdown list of free models",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			cat, err := catalog.Load(cfg.Catalog.Path)
			if err != nil {
				return fmt.Errorf("loading catalog: %w", err)
			}
			w := os.Stdout
			if out != "" {
				f, err := os.Create(out)
				if err != nil {
					return err
				}
				defer f.Close()
				w = f
			}
			return cat.WriteMarkdown(w)
		},
	}
	cmd.Flags().StringVarP(&out, "output", "o", "", "write to file instead of stdout")
	return cmd
}
