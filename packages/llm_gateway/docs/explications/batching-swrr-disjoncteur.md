# Batching, SWRR, circuit breaker

Three mechanisms decide, for each batch, *with whom* it leaves, *when*, and *at what
size*. They share the same goal: getting the most out of free-tier providers with small and
heterogeneous quotas, without ever degrading a decision.

## Micro-batching

**The batch key.** `compute_batch_key(request)` = `<catégorie>:MD5(catégorie, parameters
triés, force_provider, min_tpm_required)`. Two requests only merge if they are
perfectly compatible: same template, same parameters (hence same `prompt_variant`,
same `max_tokens`), same provider constraint. `context` is **not** part of the key: it
is re-injected into each agent block by the mobility template.

**The queue.** One Redis queue per key, sorted by priority score (lower = more urgent).
The score comes from the category; without a category that defines one, the fallback constant
9 999 999 999 puts the batch at the end of the queue. The worker pops the most urgent first.

**The window.** On the API side, if the queue reaches `get_dispatch_threshold` — `batch_target_agents`
(10) bounded by the largest `batch_max_agents` of the providers, or the exact capacity of the
forced provider — the dispatch is immediate. Otherwise a `SETNX` flag (TTL = `batch_delay_seconds`
+ 30) guarantees that exactly one dispatch delayed by `batch_delay_seconds` (3 s) is scheduled,
whatever the arrival order of concurrent requests. The 3 s come from a measurement:
prompt inter-arrival p50 = 1.4 s on the 2026-07-10 run; 1 s captured almost nothing.

**The cap at pop.** On the worker side, the actual batch size is `batch_max_agents` **of the
selected provider**:

```
tokens_per_agent  = assumed_prompt_tokens + assumed_output_tokens        # 2 200 + 800 = 3 000
batch_max_agents  = max(1, min(tpm_limit // 3 000  (or rpm_limit if no TPM),
                               max_tokens_per_request // 3 000  (ou max_batch_agents),
                               rpm_limit, max_batch_agents = 20))
```

Mistral (500 000 TPM, 60 RPM) → 20; `google_gemma42_key1` (16 000 TPM) → 5;
`groq_openai_120_key1` (8 000 TPM) → 2. This is why the dispatch threshold does not take the
*minimum* of the providers (1, because of the small TPMs): the accumulation window would
never come into play. After a successful batch, if tasks remain in the queue, the worker immediately
relaunches a `process_batch_task` for the same key.

## SWRR — Smooth Weighted Round-Robin

`core/selection.build_swrr_sequence(weights)` builds a rotation sequence once:
100 slots distributed in proportion to the weights (at least one per provider), interleaved
in the NGINX manner to avoid bursts on the same provider. `{"mistral": 4.0,
"openai": 1.0, "google_gemini31_key1": 1.0}` gives a sequence where Mistral comes back two times out of
three, never twice in a row if another one is available. The `LoadBalancer` keeps a
circular cursor under lock; **`weight: 0` removes the provider from the sequence** (out of
rotation, usable only with `force_provider`).

The weight convention is in `providers.yaml`: `weight = min(rpm, tpm / 3000) / 15`. The
weight follows the capacity actually sustainable, not the advertised RPM.

**Selection.** `select_provider(force, min_tpm, min_output)` walks the sequence from
the cursor; for each candidate: known provider, `tpm_limit ≥ min_tpm` (if both are
defined), `max_output_tokens ≥ min_output` (same), then `limiter.try_reserve`. The first one that
reserves wins. None over a full round → `RuntimeError` "All LLM providers are
saturated…", which the worker handles by local waiting (`provider_wait_seconds`, one attempt every
`saturation_poll_seconds`) then `saturation_retries` retries. After that, **busy is not
down**: if an eligible provider is neither in cooldown, nor deactivated, nor at its daily quota, the batch
keeps waiting for the window instead of being abandoned; it also waits for an instance in
cooldown that reopens in time. **Eligible** means: the pinned instance, otherwise the admitted
instances, otherwise all — an instance excluded by the restriction does not make the batch "busy".

Every wait is **bounded by the client's** (`client_wait_seconds`, or the `wait_timeout` of
the pinned instance, minus `client_wait_margin_seconds`), measured from the batch's first attempt:
beyond it, the result would no longer serve anyone. The batch is then returned **qualified** —
`surcharge_fournisseur` (5xx, per-minute 429, window that does not free up in time) or
`quota_journalier` — with a resume time, instead of the client's silent "Timeout expiré".

## Atomic RPM/TPM reservation

`RedisRateLimiter.try_reserve` first checks, outside the script: not deactivated, not in
cooldown, `active_workers < concurrency_limit`, daily quota not exhausted, current RPM under
the limit. Then **a Lua script** does in a single operation:

1. **smoothing**: refusal (`-1`) if the last request is less than `min_interval = 60 /
   rpm_limit` seconds old — no burst at the start of the window;
2. **TPM reservation**: `INCRBY tpm:<p>` of the estimated tokens in the 60 s sliding window
   ; overflow → rollback and refusal (`0`); skipped if there is no `tpm_limit`;
3. **RPM reservation**: `INCR rpm:<p>`; overflow → rollback of both and refusal;
4. timestamp of the last request.

The initial TPM reservation is the flat rate `tpm_estimate_per_request = batch_max_agents ×
3 000`, set before the batch content is known. Once the prompt is rendered, the worker
**readjusts** it to the estimated actual cost (`caractères / token_chars_ratio` + `n_agents ×
assumed_output_tokens`); after the response, it readjusts it to the billed consumption. A
failed call **gives back** RPM and TPM (`release_slot`) so that errors do not consume
quota. If the actual exceeds the reservation by more than 25 %, a WARNING suggests revisiting
`token_chars_ratio` / `assumed_output_tokens`.

**Daily quotas.** `rpd_limit` is counted at reservation (`rpd:<p>:<jourUTC>`),
`tpd_limit` afterwards on the actual tokens (`record_tokens`). The first overflow sets
a `quota_exhausted:<p>` flag with a TTL until the reset of the provider's day (`quota_reset_tz`,
Pacific midnight for Google): the provider leaves the rotation
without being re-probed every `disable_timeout` seconds.

## Cooldown, deactivation, switch

Three exclusion durations, from shortest to longest:

| Mechanism | Trigger | Duration | Where |
|---|---|---|---|
| **cooldown** | 5xx or truncation → 60 s; 429 → delay announced by the provider (bounded to [10, 3600] s, default 60); switch → `provider_switch_cooldown_seconds` (30 s) | Redis key `cooldown:<p>` with TTL | `try_reserve` refuses as long as the key exists |
| **deactivation** | 30 consecutive failed calls (`record_failure`) | provider's `disable_timeout` (180 s by default, 120 for most instances); counter reset to zero | `disabled:<p>` with TTL; `llm_provider_state` = 1 |
| **daily quota** | `rpd_limit`/`tpd_limit` reached, **or** "per day" 429 from the provider | until the provider's reset | `quota_exhausted:<p>` |

**The switch** (`_switch_provider_or_fail`) handles errors that are due to the *model*
rather than to its load: response outside the schema, unrecoverable 4xx, prompt too large for
the per-request capacity. The culprit gets the short cooldown, the batch goes back to the queue and is
replayed without `force_provider`; the rotation then picks another model. At most
`min(max_retries, len(providers) − 1)` switches, then final failure: a truly
invalid request does not loop.

On the client side, the **SDK circuit breaker** plays the symmetric role: after 10 consecutive failures
it suspends submissions (without failing them) and re-probes every 60 s; the
simulation waits for quota renewal rather than replacing the model's decisions
with a fallback ([Python SDK](../reference/sdk-python.md)).

## Learning `max_output_tokens`

The `max_tokens` sent to the provider is `min(parameters.max_tokens (défaut 4096) × n_agents,
settings.max_output_tokens (16 384), provider.max_output_tokens)`, then trimmed to fit
in `max_tokens_per_request − prompt_estimé` (`_fit_request_budget`); if even
`min_output_tokens` (512) no longer fits, `ProviderCapacityError` before the call — the 413 is
avoided and the batch goes elsewhere (38 HTTP 413s on the 2026-07-11 run motivated this safeguard).

When a provider answers 400 "`max_tokens` must be less than or equal to N" (Groq),
"supports at most N completion tokens" (OpenAI) or "… limited to N" (Google), the worker
**learns N**: `learn_provider_max_output_tokens` updates the process config and
rewrites the line `max_output_tokens: N  # auto-ajusté le AAAA-MM-JJ (HTTP 400 du provider)`
in `providers.yaml` (surgical edit, comments preserved, atomic write), then
replays the batch after 1 s. If the limit was already known, no retry: switch. The rewritten
file is the one **of the installed package** (assumption H8) — in development and in the
containers it is the repository file, mounted; on a wheel installation it would be
`site-packages`, and a read-only file system produces an `[ALARME]` without
blocking the batch.
