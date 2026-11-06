package proxy

import (
	"net/http"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/logging"
)

// authMiddleware checks the optional proxy auth token.
// If no token is configured, all requests pass through.
// API keys from upstream providers are NEVER logged.
func authMiddleware(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := r.Header.Get("Authorization")
		if got == "Bearer "+token || got == token {
			next.ServeHTTP(w, r)
			return
		}
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
	})
}

// loggingMiddleware logs request method, path, query params, status, duration,
// and body size (for POST requests). It does NOT log Authorization headers,
// request bodies, or API keys.
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}

		// Capture request details for debug logging
		query := r.URL.RawQuery
		if query != "" {
			logging.Debug("http: incoming %s %s?%s", r.Method, r.URL.Path, query)
		} else {
			logging.Debug("http: incoming %s %s", r.Method, r.URL.Path)
		}

		// Log body size for POST/PUT requests (not the content)
		if r.Method == http.MethodPost || r.Method == http.MethodPut {
			if r.ContentLength > 0 {
				logging.Debug("http: request body size=%d bytes", r.ContentLength)
			} else if r.ContentLength == -1 {
				logging.Debug("http: request body size=unknown (chunked)")
			}
		}

		next.ServeHTTP(rw, r)

		elapsed := time.Since(start)
		logging.Info("%s %s %d %v", r.Method, r.URL.Path, rw.status, elapsed)
	})
}

// recoveryMiddleware catches panics and returns a 500.
func recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				logging.Warn("panic: %v", rec)
				http.Error(w, `{"error":"internal server error"}`, http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

// Flush lets streaming handlers (SSE keep-alives) flush through the logging wrapper.
func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
