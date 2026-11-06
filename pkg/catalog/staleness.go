package catalog

import (
	"os"
	"time"
)

// IsStale returns true if the catalog file is older than maxAge.
func IsStale(path string, maxAge time.Duration) bool {
	info, err := os.Stat(expandHome(path))
	if err != nil {
		return true // missing = stale
	}
	return time.Since(info.ModTime()) > maxAge
}


// PendingReverification returns the list of model IDs (per provider) that need
// re-verification after unexpected 429s.
func PendingReverification(c *Catalog) []CatalogEntry {
	var out []CatalogEntry
	for _, e := range c.Entries {
		if e.NeedsReverification {
			out = append(out, e)
		}
	}
	return out
}
