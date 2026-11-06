// Package logging provides leveled logging backed by the stdlib log package.
// It respects the proxy.log_level config option so that debug-level messages
// are suppressed unless the operator sets log_level to "debug" in their config.
package logging

import (
	"fmt"
	"os"
	"strings"
)

const (
	LevelDebug = "debug"
	LevelInfo  = "info"
	LevelWarn  = "warn"
)

var currentLevel string

// Set sets the minimum log level. Called once during startup after config is
// loaded. Valid values: "debug", "info", "warn". Defaults to "info".
func Set(level string) {
	currentLevel = strings.ToLower(level)
}

// Debug logs a message at the debug level (only emitted if log_level == "debug").
func Debug(format string, args ...any) {
	if !shouldEmit("debug") {
		return
	}
	fmt.Fprintf(os.Stderr, "[debug] "+format+"\n", args...)
}

// Info logs a message at the info level (always emitted).
func Info(format string, args ...any) {
	if !shouldEmit("info") {
		return
	}
	fmt.Fprintf(os.Stderr, "[info]  "+format+"\n", args...)
}

// Warn logs a message at the warn level (always emitted).
func Warn(format string, args ...any) {
	if !shouldEmit("warn") {
		return
	}
	fmt.Fprintf(os.Stderr, "[warn]  "+format+"\n", args...)
}

func shouldEmit(min string) bool {
	return levelAtLeast(currentLevel, min)
}

func levelAtLeast(current, minimum string) bool {
	order := map[string]int{
		"debug": 0,
		"info":  1,
		"warn":  2,
	}
	// Lower number = more verbose. We emit if current level is at least as verbose as minimum.
	// e.g., if current="debug" (0) and minimum="info" (1): 0 <= 1 => true (emit)
	return order[strings.ToLower(current)] <= order[strings.ToLower(minimum)]
}
