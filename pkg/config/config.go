// Package config loads and watches the free-llm-proxy-router configuration file.
package config

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
)

// Config is the top-level configuration struct.
type Config struct {
	Proxy     ProxyConfig      `mapstructure:"proxy"`
	Catalog   CatalogConfig    `mapstructure:"catalog"`
	Refresh   RefreshConfig    `mapstructure:"refresh"`
	Fallback  FallbackConfig   `mapstructure:"fallback"`
	Providers []ProviderConfig `mapstructure:"providers"`
}

// ProxyConfig holds proxy server settings.
type ProxyConfig struct {
	Port      int    `mapstructure:"port"`
	AuthToken string `mapstructure:"auth_token"`
	CacheTTL  int    `mapstructure:"cache_ttl"`
	LogLevel  string `mapstructure:"log_level"`
	Strategy  string `mapstructure:"strategy"`

	// StrategyOverrides allows per-strategy config tweaks.
	StrategyOverrides map[string]map[string]any `mapstructure:"strategy_overrides"`

	// Similar holds model_family for the "similar" strategy.
	Similar map[string]string `mapstructure:"similar"`

	// Agents maps profile names to per-agent routing configuration.
	// An agent sets model: "<profile-name>" and the proxy resolves it here.
	Agents map[string]AgentProfile `mapstructure:"agents"`
}

// AgentProfile defines routing behaviour for a named agent.
// Agents identify themselves by sending model: "<profile-name>" in their request.
//
//	strategy   — which routing strategy to use (adaptive, speed, coding, …)
//	model      — pin to a specific model (e.g. "groq/llama-3.3-70b-versatile");
//	             takes precedence over strategy when set
//	defaults   — parameter defaults applied when the client did not set them
//	             (temperature, max_tokens, top_p, stop, …)
//	overrides  — parameters always applied, overriding whatever the client sent
type AgentProfile struct {
	Strategy  string         `mapstructure:"strategy"`
	Model     string         `mapstructure:"model"`
	Defaults  map[string]any `mapstructure:"defaults"`
	Overrides map[string]any `mapstructure:"overrides"`
}

// CatalogConfig holds paths and staleness settings for the local model catalog.
type CatalogConfig struct {
	Path            string        `mapstructure:"path"`
	EnrichedPath    string        `mapstructure:"enriched_path"`
	MaxAgeHours     int           `mapstructure:"max_age_hours"`
	AutoScanOnStart bool          `mapstructure:"auto_scan_on_start"`
	GitSync         GitSyncConfig `mapstructure:"git_sync"`
}

// GitSyncConfig controls catalog synchronisation via git or a raw remote URL.
// See pkg/catalog/gitsync.go for full documentation.
type GitSyncConfig struct {
	Enabled bool `mapstructure:"enabled"`
	// Scanner role (one machine only):
	RepoPath      string `mapstructure:"repo_path"`
	CatalogInRepo string `mapstructure:"catalog_in_repo"`
	AutoPush      bool   `mapstructure:"auto_push"`
	ScanInterval  string `mapstructure:"scan_interval"`
	// Replica role (all other machines):
	RemoteURL    string `mapstructure:"remote_url"`
	PullInterval string `mapstructure:"pull_interval"`
}

// RefreshConfig holds the LLM-powered refresh settings.
type RefreshConfig struct {
	Model       string `mapstructure:"model"`
	Provider    string `mapstructure:"provider"`
	Schedule    string `mapstructure:"schedule"`
	PromptFile  string `mapstructure:"prompt_file"`
	WebSearch   bool   `mapstructure:"web_search"`
	OutputMerge string `mapstructure:"output_merge"`
}

// FallbackConfig controls retry and fallback behaviour.
type FallbackConfig struct {
	RetryOn429 bool `mapstructure:"retry_on_429"`
	RetryOn5xx bool `mapstructure:"retry_on_5xx"`
	// MaxAttempts caps provider calls per pass over the fallback chain. It no
	// longer ends the request: after a pass fails, the chain backs off and
	// starts over until a provider answers.
	MaxAttempts              int `mapstructure:"max_attempts"`
	CerebrasRequestSpacingMs int `mapstructure:"cerebras_request_spacing_ms"`
	// RetryBackoffInitial and RetryBackoffMax are deprecated and ignored: failed
	// passes are now retried immediately, without backoff.
	RetryBackoffInitial time.Duration `mapstructure:"retry_backoff_initial"`
	RetryBackoffMax     time.Duration `mapstructure:"retry_backoff_max"`
	// MaxWait is the longest a request may take to get an answer: a hard cap on
	// the whole request, including provider calls already in flight, retries and
	// backoff waits. 0 (default) means no limit: the request is retried until a
	// provider answers or the client disconnects. When it elapses, the proxy
	// answers with a 503 error, never with a fabricated completion.
	MaxWait time.Duration `mapstructure:"max_wait"`
}

// mu guards the current config pointer for concurrent hot-reload.
var mu sync.RWMutex
var current *Config

// Load reads the configuration from the given file path.
// If path is empty, it searches for config.yaml in the current directory
// and in ~/.free-llm-proxy-router/.
//
// Before returning, Load looks for .env and .secrets files in standard
// locations and populates them into the dotenv map so that ResolvedAuth()
// can find API keys without requiring them in the real environment.
func Load(path string) (*Config, error) {
	v := viper.New()

	setDefaults(v)

	configDir := ""
	if path != "" {
		v.SetConfigFile(path)
		configDir = filepath.Dir(path)
	} else {
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath(".")
		v.AddConfigPath(expandHome("~/.free-llm-proxy-router"))
		v.AddConfigPath("configs")
	}

	// Load .env / .secrets files before AutomaticEnv so real env vars win.
	loadDotenv(defaultEnvPaths(configDir)...)

	v.AutomaticEnv()

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("reading config: %w", err)
		}
		// No config.yaml anywhere: fall back to the shipped example so the
		// provider list isn't silently empty (which makes scans find nothing).
		if path == "" {
			for _, ex := range []string{"configs/config.example.yaml", "config.example.yaml"} {
				if _, statErr := os.Stat(ex); statErr == nil {
					v.SetConfigFile(ex)
					if rerr := v.ReadInConfig(); rerr == nil {
						log.Printf("config: no config.yaml found; using %s (copy it to ~/.free-llm-proxy-router/config.yaml to customize)", ex)
						break
					}
				}
			}
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshalling config: %w", err)
	}

	mu.Lock()
	current = &cfg
	mu.Unlock()

	return &cfg, nil
}

// Get returns the currently-loaded config (safe for concurrent use).
func Get() *Config {
	mu.RLock()
	defer mu.RUnlock()
	return current
}

// Watch starts watching the config file for changes and calls onChange whenever
// the file changes.  The returned stop function cancels the watcher.
func Watch(path string, onChange func(*Config)) (stop func(), err error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("creating fsnotify watcher: %w", err)
	}

	target := path
	configDir := ""
	if target == "" {
		// Match Load's search order when the CLI uses its default config lookup.
		// Watching only ./config.yaml misses configs/config.yaml and the user
		// config under ~/.free-llm-proxy-router/.
		search := viper.New()
		search.SetConfigName("config")
		search.SetConfigType("yaml")
		search.AddConfigPath(".")
		search.AddConfigPath(expandHome("~/.free-llm-proxy-router"))
		search.AddConfigPath("configs")
		if searchErr := search.ReadInConfig(); searchErr == nil {
			target = search.ConfigFileUsed()
		} else {
			// Also watch the conventional local path so creating config.yaml
			// after startup can trigger a reload.
			target = "config.yaml"
		}
	} else {
		configDir = filepath.Dir(target)
	}
	watchFiles := make(map[string]struct{})
	watchDirs := make(map[string]struct{})
	addWatchFile := func(file string, required bool) error {
		abs, absErr := filepath.Abs(file)
		if absErr != nil {
			return absErr
		}
		watchFiles[filepath.Clean(abs)] = struct{}{}
		dir := filepath.Dir(abs)
		if _, exists := watchDirs[dir]; exists {
			return nil
		}
		if stat, statErr := os.Stat(dir); statErr != nil || !stat.IsDir() {
			if required {
				if statErr != nil {
					return statErr
				}
				return fmt.Errorf("%s is not a directory", dir)
			}
			return nil
		}
		if addErr := watcher.Add(dir); addErr != nil {
			if required {
				return addErr
			}
			return nil
		}
		watchDirs[dir] = struct{}{}
		return nil
	}
	if err := addWatchFile(target, true); err != nil {
		watcher.Close()
		return nil, fmt.Errorf("watching %s: %w", target, err)
	}
	for _, envFile := range defaultEnvPaths(configDir) {
		_ = addWatchFile(envFile, false)
	}

	go func() {
		var debounce *time.Timer
		var debounceC <-chan time.Time
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				abs, absErr := filepath.Abs(event.Name)
				if absErr != nil {
					continue
				}
				if _, watched := watchFiles[filepath.Clean(abs)]; !watched {
					continue
				}
				if event.Has(fsnotify.Write) || event.Has(fsnotify.Create) || event.Has(fsnotify.Rename) || event.Has(fsnotify.Remove) {
					if debounce == nil {
						debounce = time.NewTimer(200 * time.Millisecond)
						debounceC = debounce.C
					} else {
						if !debounce.Stop() {
							select {
							case <-debounce.C:
							default:
							}
						}
						debounce.Reset(200 * time.Millisecond)
					}
				}
			case watchErr, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Printf("config watcher error: %v", watchErr)
			case <-debounceC:
				cfg, loadErr := Load(path)
				if loadErr != nil {
					log.Printf("config reload error: %v", loadErr)
				} else {
					log.Printf("config reloaded from %s (including .env/.secrets)", path)
					onChange(cfg)
				}
				debounceC = nil
			}
		}
	}()

	return func() { watcher.Close() }, nil
}

// setDefaults installs sensible defaults into a viper instance.
func setDefaults(v *viper.Viper) {
	v.SetDefault("proxy.port", 8080)
	v.SetDefault("proxy.cache_ttl", 300)
	v.SetDefault("proxy.log_level", "info")
	v.SetDefault("proxy.strategy", "smart")
	v.SetDefault("catalog.path", expandHome("~/.free-llm-proxy-router/catalog.json"))
	v.SetDefault("catalog.enriched_path", expandHome("~/.free-llm-proxy-router/enriched.json"))
	v.SetDefault("catalog.max_age_hours", 24)
	v.SetDefault("fallback.retry_on_429", true)
	v.SetDefault("fallback.retry_on_5xx", true)
	v.SetDefault("fallback.max_attempts", 5)
	v.SetDefault("fallback.cerebras_request_spacing_ms", 100)
	v.SetDefault("fallback.retry_backoff_initial", "250ms")
	v.SetDefault("fallback.retry_backoff_max", "2s")
	v.SetDefault("fallback.max_wait", "0s")
	v.SetDefault("refresh.schedule", "weekly")
	v.SetDefault("refresh.output_merge", "conservative")
}

// expandHome replaces a leading ~ with the user's home directory.
func expandHome(path string) string {
	if len(path) >= 2 && path[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	}
	return path
}

// lookupEnv returns the value for key, checking real env vars first,
// then falling back to any value loaded from .env / .secrets files.
func lookupEnv(key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return dotenvLookup(key)
}
