package catalog

import (
	"crypto/sha256"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
)

// reloadDebounce is how long the watcher waits after the last change event
// before reloading. Atomic writers (Save, git-sync) produce one event burst of
// a few milliseconds, so this only needs to cover that burst; a larger value
// is pure added latency before new models are routable.
const reloadDebounce = 40 * time.Millisecond

// Watch starts a file watcher on catalogPath and calls onChange whenever the
// file is written or replaced (e.g. by git-sync, free-llm-scan update, or a
// manual copy).  The returned stop function cancels the watcher.
//
// A change event whose file content is identical to what was last delivered is
// ignored, so touching the file or rewriting it unchanged costs nothing. A file
// that cannot be parsed (e.g. a non-atomic copy caught half-way) or that has
// disappeared is skipped and the current catalog stays active; the next write
// event triggers another attempt.
func Watch(catalogPath string, onChange func(*Catalog)) (stop func(), err error) {
	expanded := expandHome(catalogPath)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}

	// Watch the directory so we catch atomic renames (tmp → real).
	if err := watcher.Add(filepath.Dir(expanded)); err != nil {
		watcher.Close()
		return nil, err
	}

	// Fingerprint of the content the proxy started with, so the first no-op
	// event does not trigger a reload.
	var lastSum [sha256.Size]byte
	if data, rerr := os.ReadFile(expanded); rerr == nil {
		lastSum = sha256.Sum256(data)
	}

	go func() {
		debounce := time.NewTimer(0)
		<-debounce.C

		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if event.Name != expanded && filepath.Base(event.Name) != filepath.Base(expanded) {
					continue
				}
				if event.Has(fsnotify.Write) || event.Has(fsnotify.Create) || event.Has(fsnotify.Rename) {
					debounce.Reset(reloadDebounce)
				}
			case watchErr, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Printf("catalog watcher error: %v", watchErr)
			case <-debounce.C:
				data, rerr := os.ReadFile(expanded)
				if rerr != nil {
					if !os.IsNotExist(rerr) {
						log.Printf("catalog reload error: %v", rerr)
					}
					continue
				}
				sum := sha256.Sum256(data)
				if sum == lastSum {
					continue
				}
				cat, perr := parseCatalog(data, catalogPath)
				if perr != nil {
					log.Printf("catalog reload error: %v", perr)
					continue
				}
				lastSum = sum
				onChange(cat)
			}
		}
	}()

	return func() { watcher.Close() }, nil
}
