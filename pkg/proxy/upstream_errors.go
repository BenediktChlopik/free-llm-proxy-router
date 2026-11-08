package proxy

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Cooldowns applied to failures that retrying immediately cannot fix.
const (
	// noCreditCooldown applies when a provider reports an empty balance or a
	// missing resource package on a non-402 status (e.g. Z.ai returns 429 with
	// code 1113). The account, not the model, is the problem.
	noCreditCooldown = time.Hour
	// unsupportedModelCooldown applies when a model rejects the request shape
	// itself (no tool support, not a chat model).
	unsupportedModelCooldown = 6 * time.Hour
	// openRouterDailyLimitCooldown is used when OpenRouter reports the daily
	// free-model quota is spent and gives no usable reset time.
	openRouterDailyLimitCooldown = time.Hour
	// openRouterCooldownKey is the cooldown key for OpenRouter's native
	// models[] / openrouter/free routes, which are not tied to a single model.
	openRouterCooldownKey = "openrouter"
)

// upstreamFailure classifies an upstream error body.
type upstreamFailure int

const (
	failureOther upstreamFailure = iota
	// failureNoCredit: insufficient balance / credits / resource package.
	failureNoCredit
	// failureUnsupportedModel: the model cannot serve this kind of request
	// (does not support tools, is not a chat model).
	failureUnsupportedModel
	// failureDailyLimit: OpenRouter free-models-per-day quota exhausted.
	failureDailyLimit
)

// classifyUpstreamFailure inspects a non-200 response and reports which
// well-known, non-transient failure it is.
func classifyUpstreamFailure(resp *Response) upstreamFailure {
	if resp == nil || resp.StatusCode == http.StatusOK || len(resp.Body) == 0 {
		return failureOther
	}
	body := strings.ToLower(string(resp.Body))
	switch {
	case strings.Contains(body, "free-models-per-day"):
		return failureDailyLimit
	case strings.Contains(body, "insufficient balance"),
		strings.Contains(body, "insufficient_quota"),
		strings.Contains(body, "no resource package"),
		strings.Contains(body, "credits required"),
		strings.Contains(body, "please recharge"),
		strings.Contains(body, "top up your balance"):
		return failureNoCredit
	case strings.Contains(body, "does not support tools"),
		strings.Contains(body, "does not support tool"),
		strings.Contains(body, "toolcall' object is not subscriptable"),
		strings.Contains(body, "does not support chat"):
		return failureUnsupportedModel
	}
	return failureOther
}

var itemsLimitRe = regexp.MustCompile(`must have (\d+) items? or fewer`)

// modelsArrayLimit extracts N from OpenRouter's "'models' array must have N
// items or fewer." 400 error. It returns 0 when resp is not that error.
func modelsArrayLimit(resp *Response) int {
	if resp == nil || resp.StatusCode != http.StatusBadRequest {
		return 0
	}
	if m := itemsLimitRe.FindSubmatch(resp.Body); m != nil {
		if n, err := strconv.Atoi(string(m[1])); err == nil {
			return n
		}
	}
	return 0
}

// openRouterResetWait returns how long to wait for OpenRouter's daily quota to
// reset, using the X-RateLimit-Reset header (epoch milliseconds) embedded in
// the error metadata. It falls back to openRouterDailyLimitCooldown.
func openRouterResetWait(body []byte) time.Duration {
	var parsed struct {
		Error struct {
			Metadata struct {
				Headers map[string]string `json:"headers"`
			} `json:"metadata"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &parsed) == nil {
		if v := parsed.Error.Metadata.Headers["X-RateLimit-Reset"]; v != "" {
			if ms, err := strconv.ParseInt(v, 10, 64); err == nil && ms > 0 {
				if d := time.Until(time.UnixMilli(ms)); d > 0 && d < 48*time.Hour {
					return d
				}
			}
		}
	}
	return openRouterDailyLimitCooldown
}

// geminiSkipThoughtSignature is Google's documented dummy value that lets
// function calls replayed from another model pass thought-signature validation.
const geminiSkipThoughtSignature = "skip_thought_signature_validator"

// sanitizeBody adapts an OpenAI-style request body to what providerID accepts.
// It never mutates body's nested values; messages and tools are copied when
// they need changes.
//
//   - Non-OpenRouter providers reject "reasoning_details" (and the other
//     OpenRouter-only reasoning fields) on assistant messages with a 400.
//   - Gemini rejects null content/arguments ("Value is not a string: null"),
//     null schema nodes ("Value is not a struct: null"), and tool calls without
//     a thought signature.
func sanitizeBody(providerID string, body map[string]any) map[string]any {
	if providerID == "openrouter" {
		return body
	}
	if msgs, ok := body["messages"].([]any); ok {
		out := make([]any, len(msgs))
		for i, m := range msgs {
			msg, isMap := m.(map[string]any)
			if !isMap {
				out[i] = m
				continue
			}
			out[i] = sanitizeMessage(providerID, msg)
		}
		body["messages"] = out
	} else if msgs, ok := body["messages"].([]map[string]any); ok {
		out := make([]any, len(msgs))
		for i, msg := range msgs {
			out[i] = sanitizeMessage(providerID, msg)
		}
		body["messages"] = out
	}
	if providerID == "gemini" {
		for _, key := range []string{"tools", "functions"} {
			if v, ok := body[key]; ok {
				body[key] = stripNulls(v)
			}
		}
		for _, key := range []string{"tool_choice", "function_call", "response_format"} {
			if v, ok := body[key]; ok && v == nil {
				delete(body, key)
			}
		}
	}
	return body
}

func sanitizeMessage(providerID string, msg map[string]any) map[string]any {
	out := make(map[string]any, len(msg))
	for k, v := range msg {
		switch k {
		case "reasoning_details", "reasoning", "reasoning_content":
			continue
		}
		out[k] = v
	}
	if providerID != "gemini" {
		return out
	}
	// Gemini refuses null values anywhere in a message.
	for k, v := range out {
		if v == nil {
			if k == "content" {
				out[k] = ""
			} else {
				delete(out, k)
			}
		}
	}
	calls, ok := out["tool_calls"].([]any)
	if !ok {
		return out
	}
	newCalls := make([]any, len(calls))
	for i, c := range calls {
		call, isMap := c.(map[string]any)
		if !isMap {
			newCalls[i] = c
			continue
		}
		call = copyMap(call)
		if fn, isFn := call["function"].(map[string]any); isFn {
			fn = copyMap(fn)
			if args, present := fn["arguments"]; present && args == nil {
				fn["arguments"] = "{}"
			}
			call["function"] = fn
		}
		if _, has := call["extra_content"]; !has {
			call["extra_content"] = map[string]any{
				"google": map[string]any{"thought_signature": geminiSkipThoughtSignature},
			}
		}
		newCalls[i] = call
	}
	out["tool_calls"] = newCalls
	return out
}

// stripNulls returns v with every null map value removed, recursively. Maps and
// slices are copied, never modified in place.
func stripNulls(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			if val == nil {
				continue
			}
			out[k] = stripNulls(val)
		}
		return out
	case []any:
		out := make([]any, 0, len(t))
		for _, val := range t {
			out = append(out, stripNulls(val))
		}
		return out
	}
	return v
}

// openRouterModelsLimit is the largest models[] array sent to OpenRouter. It
// starts at maxOpenRouterFallbackModels and shrinks if OpenRouter reports a
// stricter limit.
var openRouterModelsLimit atomic.Int32

func init() { openRouterModelsLimit.Store(maxOpenRouterFallbackModels) }

// currentOpenRouterModelsLimit returns the active models[] size cap (>= 1).
func currentOpenRouterModelsLimit() int {
	if n := int(openRouterModelsLimit.Load()); n >= 1 {
		return n
	}
	return 1
}

// lowerOpenRouterModelsLimit records that OpenRouter rejected a models[] array
// of size sent while advertising limit, and returns the new cap. The cap always
// ends strictly below sent so the next attempt cannot repeat the same mistake.
func lowerOpenRouterModelsLimit(sent, limit int) int {
	n := limit
	if sent <= n {
		n = sent - 1
	}
	if n < 1 {
		n = 1
	}
	openRouterModelsLimit.Store(int32(n))
	return n
}
