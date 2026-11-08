package proxy

import (
	"context"
	"net/http"
	"time"

	"github.com/kaiser-data/free-llm-proxy-router/pkg/config"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/logging"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/ratelimit"
	"github.com/kaiser-data/free-llm-proxy-router/pkg/strategy"
)

// attempt is one provider call made on behalf of a request.
type attempt struct {
	r        strategy.RankedEntry
	provider *config.ProviderConfig // configured provider, before key-slot substitution
	cfg      config.ProviderConfig  // what is actually called (active key slot applied)
	body     map[string]any
	native   bool // OpenRouter models[] request
	sent     int  // number of models sent in a native request
}

// attemptResult is the outcome of an attempt, delivered over a channel.
type attemptResult struct {
	attempt
	resp *Response
	err  error
}

// runCandidates walks the strategy's ranked candidates and returns the first
// usable response (steps 1–4 of the chain; see FallbackChain).
//
// It is built for the fastest possible fallback:
//
//   - A candidate that fails (429, 5xx, transport error, empty completion) is
//     replaced by the next one immediately.
//   - A candidate that is merely slow is hedged: after Fallback.HedgeAfter of
//     silence the next candidate starts *alongside* it. The slow call is not
//     cancelled — if it answers first it still wins. At most
//     Fallback.HedgeMaxInflight calls are in flight at once.
//   - The first usable completion wins; the remaining calls are cancelled.
//
// With HedgeAfter == 0 no hedging happens and candidates are tried strictly
// one after another.
//
// The second result is the number of candidate calls started (the OpenRouter
// models[] call is not counted, as before). A nil response with a nil error
// means no candidate answered in this pass.
func (fc *FallbackChain) runCandidates(ctx context.Context, req Request, stratReq strategy.Request, ranked []strategy.RankedEntry, maxAttempts int, demoted map[string]bool) (*Response, int, error) {
	hedgeAfter := fc.Cfg.Fallback.HedgeAfter
	maxInflight := fc.Cfg.Fallback.HedgeMaxInflight
	if maxInflight <= 0 {
		maxInflight = 2
	}

	raceCtx, cancelRace := context.WithCancel(ctx)
	defer cancelRace()

	// Buffered for every call this pass can start, so a cancelled loser never
	// blocks on send and its goroutine exits right away.
	results := make(chan attemptResult, maxAttempts+2)
	inflight := 0
	inflightProviders := make(map[string]bool)

	launch := func(a attempt) {
		inflight++
		inflightProviders[a.r.ProviderID] = true
		go func() {
			resp, err := fc.callProvider(raceCtx, a.cfg, a.body)
			results <- attemptResult{attempt: a, resp: resp, err: err}
		}()
	}
	received := func(res attemptResult) {
		inflight--
		delete(inflightProviders, res.r.ProviderID)
	}

	// wait blocks for the next finished call. With hedge set it also returns
	// (zero, false, nil) when hedgeAfter passes with nothing finished and there
	// is room for another call, which tells the caller to start the next
	// candidate while the current ones keep running.
	wait := func(hedge bool) (attemptResult, bool, error) {
		var timer <-chan time.Time
		if hedge && hedgeAfter > 0 && inflight < maxInflight {
			t := time.NewTimer(hedgeAfter)
			defer t.Stop()
			timer = t.C
		}
		select {
		case res := <-results:
			received(res)
			return res, true, nil
		case <-timer:
			return attemptResult{}, false, nil
		case <-ctx.Done():
			return attemptResult{}, false, ctx.Err()
		}
	}

	// Once a winner is known, calls that had already failed still taught us
	// something (cooldowns, retired models); apply it. Calls still running are
	// cancelled by cancelRace, and their cancellation is not a provider failure.
	defer func() {
		for {
			select {
			case res := <-results:
				received(res)
				if res.err != nil || res.resp == nil {
					continue
				}
				if res.resp.usable() {
					if res.r.ProviderID == "gemini" && fc.GeminiTracker != nil {
						fc.GeminiTracker.RecordRequest(res.r.ModelID, stratReq.EstimatedPromptTokens)
					}
					continue
				}
				if !res.native {
					fc.applyFailureEffects(res.r, res.provider, res.resp)
				}
			default:
				return
			}
		}
	}()

	failedProviders := make(map[string]bool)
	providerTries := make(map[string]int)

	// process handles one finished call. It returns done=true when the pass is
	// over: either with a usable response or with an error.
	process := func(res attemptResult) (resp *Response, done bool, err error) {
		r := res.r
		resp, callErr := res.resp, res.err

		if res.native {
			if callErr == nil && resp.StatusCode == http.StatusOK {
				fc.ReliabilityTracker.Record("openrouter", true)
				return resp, true, nil
			}
			if ctx.Err() != nil {
				return nil, true, ctx.Err()
			}
			logging.Debug("fallback: openrouter models[] failed — continuing%s", upstreamErrSnippet(resp, callErr))
			fc.noteOpenRouterFailure(resp, res.sent)
			fc.ReliabilityTracker.Record("openrouter", false)
			return nil, false, nil
		}

		if callErr == nil && resp.StatusCode == http.StatusOK && resp.Empty {
			// The provider answered 200 but with nothing in it. Treat it like a
			// failure, but prefer *other providers* before asking this one again:
			// skip it for the rest of this pass and demote it for later passes.
			logging.Debug("fallback: %s/%s returned an empty completion — trying other providers first", r.ProviderID, r.ModelID)
			demoted[r.ProviderID+"/"+r.ModelID] = true
			failedProviders[r.ProviderID] = true
			return nil, false, nil
		}

		if callErr == nil && resp.StatusCode == http.StatusOK {
			// Success path.
			if r.ProviderID == "cerebras" {
				info := ratelimit.ExtractRateLimitInfo("cerebras", resp.Header, resp.Body)
				if info.RemainingRequests >= 0 && info.RemainingRequests < 2 {
					tracker := fc.RateLimiter.Provider("cerebras")
					tracker.PreemptiveSlowdown(info.WaitDuration())
				}
			}
			fc.ReliabilityTracker.Record(r.ProviderID, true)
			if r.ProviderID == "gemini" && fc.GeminiTracker != nil {
				fc.GeminiTracker.RecordRequest(r.ModelID, stratReq.EstimatedPromptTokens)
			}
			return resp, true, nil
		}

		// HuggingFace 503+loading: wait and retry SAME provider (cold start).
		// Only when nothing else is running: blocking here would otherwise hold
		// up a hedge that might be about to answer.
		if r.ProviderID == "huggingface" && resp != nil && resp.StatusCode == 503 && inflight == 0 {
			if waitMs := parseHFLoadingWait(resp.Body); waitMs > 0 {
				logging.Info("fallback: huggingface model loading, waiting %dms", waitMs)
				if serr := sleepCtx(ctx, time.Duration(waitMs)*time.Millisecond); serr != nil {
					return nil, true, serr
				}
				resp, callErr = fc.callProvider(ctx, *res.provider, res.body)
				if callErr == nil && resp.StatusCode == http.StatusOK {
					fc.ReliabilityTracker.Record(r.ProviderID, true)
					return resp, true, nil
				}
			}
		}
		if ctx.Err() != nil {
			// Client went away; not the provider's fault.
			return nil, true, ctx.Err()
		}

		modelLevel, accountLevel := fc.applyFailureEffects(r, res.provider, resp)
		statusCode := 0
		if resp != nil {
			statusCode = resp.StatusCode
		}
		providerTries[r.ProviderID]++
		if modelLevel && providerTries[r.ProviderID] < maxModelsPerProvider {
			// The provider itself is fine; try its next ranked model.
			logging.Debug("fallback: %s/%s status %d — trying next model%s", r.ProviderID, r.ModelID, statusCode, upstreamErrSnippet(resp, callErr))
			return nil, false, nil
		}
		logging.Debug("fallback: %s/%s status %d — skipping provider%s", r.ProviderID, r.ModelID, statusCode, upstreamErrSnippet(resp, callErr))
		// Model-level failures say nothing about the provider's health, so
		// only auth and transport errors count against its reliability.
		if !modelLevel && !accountLevel {
			fc.ReliabilityTracker.Record(r.ProviderID, false)
			if fc.ReliabilityTracker.ShouldCooldown(r.ProviderID) {
				fc.RateLimiter.SetCooldown(r.ProviderID, time.Now().Add(30*time.Second))
				logging.Warn("fallback: %s reliability below 50%% — short cooldown", r.ProviderID)
			}
		}
		failedProviders[r.ProviderID] = true // don't try other models from this provider
		return nil, false, nil
	}

	// pump waits for the call just launched (or any earlier one) to finish, or
	// for the hedge timer. done=false means: start the next candidate now.
	pump := func() (*Response, bool, error) {
		res, got, werr := wait(true)
		if werr != nil {
			return nil, true, werr
		}
		if !got {
			logging.Debug("fallback: no answer after %s — hedging with the next candidate", hedgeAfter)
			return nil, false, nil
		}
		return process(res)
	}

	// Step 1: OpenRouter with models[] array (native fallback). Skipped while
	// OpenRouter's free quota is known to be exhausted.
	if or := fc.findProvider("openrouter"); or != nil && !fc.RateLimiter.IsOnCooldown(openRouterCooldownKey) {
		if models := fc.collectProviderModels("openrouter", ranked); len(models) > 0 {
			launch(attempt{
				r:        strategy.RankedEntry{ProviderID: "openrouter"},
				provider: or,
				cfg:      *or,
				body:     BuildOpenRouterRequest(req.Raw, models),
				native:   true,
				sent:     len(models),
			})
			if resp, done, err := pump(); done {
				return resp, 0, err
			}
		}
	}

	// Step 2–4: One model per provider in ranked order.
	//
	// Key invariant: once a provider returns any error, ALL remaining entries
	// for that provider are skipped immediately. This prevents a large provider
	// (e.g. NVIDIA NIM with 172 models) from exhausting maxAttempts before
	// faster, more reliable providers (Groq, Cerebras, Gemini) are even tried.
	attempts := 0
	for _, r := range ranked {
		if attempts >= maxAttempts {
			break
		}
		// Skip OpenRouter entries — already tried above via models[] array.
		if r.ProviderID == "openrouter" {
			continue
		}
		// Skip providers that already returned an error this request, and
		// providers that already have a call running (a hedge never doubles up
		// on the provider it is hedging against).
		if failedProviders[r.ProviderID] || inflightProviders[r.ProviderID] {
			continue
		}

		providerCfg := fc.findProvider(r.ProviderID)
		if providerCfg == nil {
			continue
		}

		// Skip any provider currently on rate-limit cooldown (auth errors).
		if fc.RateLimiter.IsOnCooldown(r.ProviderID) {
			logging.Debug("fallback: %s on cooldown — skipping provider", r.ProviderID)
			failedProviders[r.ProviderID] = true
			continue
		}
		// Skip this specific model if it is rate-limited, but keep trying other models.
		if fc.RateLimiter.IsOnCooldown(r.ProviderID + "/" + r.ModelID) {
			logging.Debug("fallback: %s/%s rate-limited — skipping model", r.ProviderID, r.ModelID)
			continue
		}

		// Cerebras pre-emptive check (RPS-level, finer than cooldown).
		if r.ProviderID == "cerebras" {
			tracker := fc.RateLimiter.Provider("cerebras")
			if !tracker.CanRequest(r.ModelID, 0, 0) {
				logging.Debug("fallback: cerebras pre-emptive skip (rate limited)")
				failedProviders[r.ProviderID] = true
				continue
			}
			spacingMs := fc.Cfg.Fallback.CerebrasRequestSpacingMs
			if spacingMs > 0 {
				if err := sleepCtx(ctx, time.Duration(spacingMs)*time.Millisecond); err != nil {
					return nil, attempts, err
				}
			}
		}

		// Gemini 4-dimensional rate check.
		if r.ProviderID == "gemini" && fc.GeminiTracker != nil {
			if !fc.GeminiTracker.CanRequest(r.ModelID, stratReq.EstimatedPromptTokens) {
				logging.Debug("fallback: gemini %s rate limited — skipping", r.ModelID)
				failedProviders[r.ProviderID] = true
				continue
			}
		}

		body := fc.buildBody(r.ProviderID, req.Raw, r.ModelID)
		// Apply active key slot when multiple API keys are configured.
		callCfg := *providerCfg
		if keys := providerCfg.AllKeys(); len(keys) > 1 {
			if idx := fc.RateLimiter.CurrentKeyIndex(r.ProviderID); idx < len(keys) {
				callCfg.APIKey = keys[idx]
				callCfg.APIKeyEnv = ""
			}
		}
		launch(attempt{r: r, provider: providerCfg, cfg: callCfg, body: body})
		attempts++

		if resp, done, err := pump(); done {
			return resp, attempts, err
		}
	}

	// No candidate left to start: wait for whatever is still running.
	for inflight > 0 {
		res, _, werr := wait(false)
		if werr != nil {
			return nil, attempts, werr
		}
		if resp, done, err := process(res); done {
			return resp, attempts, err
		}
	}
	return nil, attempts, nil
}

// applyFailureEffects records what a non-successful provider response means:
// cooldowns, API-key rotation, and permanent exclusion of retired models. It
// reports whether the failure is specific to the model (so the provider's other
// models are still worth trying) and whether it affects the whole account.
//
// Only auth errors and transport failures (resp == nil) mean the provider
// itself is unusable. Any other status is treated as model-specific: free tiers
// often fail one model (quota, retired, overloaded) while the provider's other
// models still answer.
func (fc *FallbackChain) applyFailureEffects(r strategy.RankedEntry, providerCfg *config.ProviderConfig, resp *Response) (modelLevel, accountLevel bool) {
	modelLevel = resp != nil
	if resp == nil {
		return modelLevel, false
	}
	switch classifyUpstreamFailure(resp) {
	case failureNoCredit:
		if resp.StatusCode != http.StatusPaymentRequired {
			// e.g. Z.ai answers 429 / code 1113 for an empty balance: every
			// model on this key fails, so retrying soon is pointless.
			modelLevel, accountLevel = false, true
			fc.RateLimiter.SetCooldown(r.ProviderID, time.Now().Add(noCreditCooldown))
			logging.Warn("fallback: %s has no credit/resource package — cooldown %s", r.ProviderID, noCreditCooldown)
		}
	case failureUnsupportedModel:
		// The model cannot serve this request shape (no tool support,
		// not a chat model); stop picking it for a while.
		fc.RateLimiter.SetCooldown(r.ProviderID+"/"+r.ModelID, time.Now().Add(unsupportedModelCooldown))
		logging.Info("fallback: %s/%s cannot serve this request — model cooldown %s", r.ProviderID, r.ModelID, unsupportedModelCooldown)
	}
	if accountLevel {
		return modelLevel, accountLevel
	}
	switch resp.StatusCode {
	case 429:
		fc.Catalog.MarkNeedsReverification(r.ProviderID, r.ModelID)
		info := ratelimit.ExtractRateLimitInfo(r.ProviderID, resp.Header, resp.Body)
		waitDur := info.WaitDuration()
		if isZeroQuota(resp.Body) {
			// Quota "limit: 0" means the model is not on this key's free tier.
			waitDur = zeroQuotaCooldown
		}
		until := time.Now().Add(waitDur)
		// Cooldown on the specific model, not the whole provider.
		fc.RateLimiter.SetCooldown(r.ProviderID+"/"+r.ModelID, until)
		// Rotate to the next API key slot if multiple keys are configured.
		if providerCfg != nil && providerCfg.NumKeys() > 1 {
			newIdx := fc.RateLimiter.RotateKey(r.ProviderID, providerCfg.NumKeys())
			logging.Info("fallback: %s/%s rate limited — rotated to key slot %d", r.ProviderID, r.ModelID, newIdx+1)
		} else {
			logging.Info("fallback: %s/%s rate limited — model cooldown %s", r.ProviderID, r.ModelID, waitDur.Round(time.Second))
		}
	case 401, 403:
		modelLevel = false
		fc.RateLimiter.SetCooldown(r.ProviderID, time.Now().Add(10*time.Minute))
		logging.Warn("fallback: %s auth error %d — cooldown 10min", r.ProviderID, resp.StatusCode)
	case http.StatusPaymentRequired, http.StatusNotFound:
		fc.excludeModel(r.ProviderID, r.ModelID)
		logging.Info("fallback: %s model %s returned %d — excluded from free catalog", r.ProviderID, r.ModelID, resp.StatusCode)
	}
	return modelLevel, accountLevel
}
