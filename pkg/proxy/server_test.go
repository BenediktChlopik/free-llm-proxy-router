package proxy

import (
	"path/filepath"
	"testing"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/catalog"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
)

func TestPersistExcludedModelWritesCatalogAndBlocklist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "catalog.json")
	cat := &catalog.Catalog{
		Version: 1,
		Entries: []catalog.CatalogEntry{{ProviderID: "test-provider", ModelID: "paid-model", IsFree: true}},
	}
	cfg := &config.Config{Catalog: config.CatalogConfig{Path: path}}
	srv := NewServer(cfg, cat, nil, nil, nil, nil)

	srv.persistExcludedModel("test-provider", "paid-model")

	persisted, err := catalog.Load(path)
	if err != nil {
		t.Fatalf("loading persisted catalog: %v", err)
	}
	if len(persisted.Entries) != 0 {
		t.Fatalf("persisted entries = %v, want excluded model removed", persisted.Entries)
	}
	if len(persisted.Blocklist) != 1 || persisted.Blocklist[0] != "test-provider/paid-model" {
		t.Fatalf("persisted blocklist = %v", persisted.Blocklist)
	}
}

func TestUpdateCatalogRetainsExistingBlocklist(t *testing.T) {
	existing := &catalog.Catalog{Entries: []catalog.CatalogEntry{{
		ProviderID: "test-provider", ModelID: "missing-model", IsFree: true,
	}}}
	existing.ExcludeModel("test-provider", "missing-model")
	srv := NewServer(&config.Config{}, existing, nil, nil, nil, nil)

	// Simulate a catalog watcher delivering a stale snapshot that still lists
	// the model after an in-flight request excluded it.
	srv.UpdateCatalog(&catalog.Catalog{Entries: []catalog.CatalogEntry{{
		ProviderID: "test-provider", ModelID: "missing-model", IsFree: true,
	}}})

	current := srv.catalog.Load()
	if len(current.FreeEntries()) != 0 {
		t.Fatalf("stale reload resurrected an excluded model: %v", current.FreeEntries())
	}
	if len(current.BlocklistSnapshot()) != 1 {
		t.Fatalf("blocklist lost across reload: %v", current.BlocklistSnapshot())
	}
}
