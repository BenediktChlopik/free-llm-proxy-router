package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMissingKey(t *testing.T) {
	t.Setenv("FLPR_TEST_KEY", "")
	needsKey := ProviderConfig{ID: "groq", APIKeyEnv: "FLPR_TEST_KEY"}
	if !needsKey.MissingKey() {
		t.Error("provider with api_key_env and no key should report MissingKey")
	}

	t.Setenv("FLPR_TEST_KEY", "gsk_test")
	if needsKey.MissingKey() {
		t.Error("provider with key in env should not report MissingKey")
	}

	literal := ProviderConfig{ID: "x", APIKeyEnv: "FLPR_UNSET_KEY", APIKey: "literal"}
	if literal.MissingKey() {
		t.Error("literal api_key should count as configured")
	}

	keyless := ProviderConfig{ID: "ollama-local"}
	if keyless.MissingKey() {
		t.Error("keyless provider (no api_key_env) must never report MissingKey")
	}
}

func TestProviderIsEnabledFromEnvironment(t *testing.T) {
	t.Setenv("FLPR_TEST_KEY", "")
	provider := ProviderConfig{ID: "groq", APIKeyEnv: "FLPR_TEST_KEY"}
	if provider.IsEnabled() {
		t.Fatal("provider without its configured API key must be disabled")
	}
	t.Setenv("FLPR_TEST_KEY", "gsk_test")
	if !provider.IsEnabled() {
		t.Fatal("provider with its configured API key must be enabled")
	}
}

func TestProviderEnableEnv(t *testing.T) {
	provider := ProviderConfig{ID: "ollama-local", EnableEnv: "FLPR_OLLAMA_ENABLED", APIKey: "ollama"}
	t.Setenv("FLPR_OLLAMA_ENABLED", "false")
	if provider.IsEnabled() {
		t.Fatal("local provider must be disabled when its enable toggle is false")
	}
	t.Setenv("FLPR_OLLAMA_ENABLED", "true")
	if !provider.IsEnabled() {
		t.Fatal("local provider must be enabled when its enable toggle is true")
	}
}

func TestProviderIsEnabledFromDotenvFile(t *testing.T) {
	const key = "FLPR_TEST_DOTENV_PROVIDER_KEY"
	path := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(path, []byte(key+"=test-token\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	loadDotenv(path)
	defer loadDotenv()
	t.Setenv(key, "")

	provider := ProviderConfig{ID: "test-provider", APIKeyEnv: key}
	if !provider.IsEnabled() {
		t.Fatal("provider with a key in .env must be enabled")
	}
}
