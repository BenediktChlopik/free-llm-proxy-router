package catalog

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func bigCatalog(n int) *Catalog {
	c := &Catalog{Version: 1}
	for i := 0; i < n; i++ {
		c.Entries = append(c.Entries, CatalogEntry{
			ProviderID: fmt.Sprintf("p%d", i%12),
			ModelID:    fmt.Sprintf("model-%d", i),
			IsFree:     true,
			TierType:   "free",
		})
	}
	return c
}

func watchTemp(t *testing.T, initial *Catalog) (path string, got chan *Catalog) {
	t.Helper()
	path = filepath.Join(t.TempDir(), "catalog.json")
	if initial != nil {
		if err := Save(initial, path); err != nil {
			t.Fatal(err)
		}
	}
	got = make(chan *Catalog, 8)
	stop, err := Watch(path, func(c *Catalog) { got <- c })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(stop)
	return path, got
}

func TestWatch_ReloadsQuicklyAfterSave(t *testing.T) {
	path, got := watchTemp(t, bigCatalog(10))

	start := time.Now()
	if err := Save(bigCatalog(521), path); err != nil {
		t.Fatal(err)
	}
	select {
	case c := <-got:
		if n := len(c.FreeEntries()); n != 521 {
			t.Errorf("reloaded %d models, want 521", n)
		}
		// Debounce is 40ms; leave generous headroom for slow CI.
		if el := time.Since(start); el > 250*time.Millisecond {
			t.Errorf("hot-reload took %v, want well under 250ms", el)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("catalog was not hot-reloaded")
	}
}

func TestWatch_IgnoresUnchangedContent(t *testing.T) {
	path, got := watchTemp(t, bigCatalog(50))

	// Rewriting identical bytes (and merely touching the file) is not a change.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	select {
	case <-got:
		t.Fatal("reloaded although the content did not change")
	case <-time.After(300 * time.Millisecond):
	}
}

func TestWatch_SkipsBrokenFileThenRecovers(t *testing.T) {
	path, got := watchTemp(t, bigCatalog(5))

	// A non-atomic writer caught mid-write leaves invalid JSON behind.
	if err := os.WriteFile(path, []byte(`{"version":1,"entries":[`), 0o644); err != nil {
		t.Fatal(err)
	}
	select {
	case <-got:
		t.Fatal("a broken catalog must not replace the active one")
	case <-time.After(300 * time.Millisecond):
	}

	if err := Save(bigCatalog(7), path); err != nil {
		t.Fatal(err)
	}
	select {
	case c := <-got:
		if n := len(c.Entries); n != 7 {
			t.Errorf("entries = %d, want 7", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("watcher did not recover after the file became valid")
	}
}

func TestLoad_MissingFileIsEmptyCatalog(t *testing.T) {
	c, err := Load(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil || len(c.Entries) != 0 {
		t.Fatalf("Load(missing) = %v, %v; want empty catalog", c, err)
	}
}

func BenchmarkLoad521(b *testing.B) {
	path := filepath.Join(b.TempDir(), "catalog.json")
	if err := Save(bigCatalog(521), path); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Load(path); err != nil {
			b.Fatal(err)
		}
	}
}
