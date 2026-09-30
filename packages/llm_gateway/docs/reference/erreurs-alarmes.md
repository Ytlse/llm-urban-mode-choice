# Errors and alarms

## Exception hierarchy

Declared in `adapters/base.py`, except `ProviderCapacityError` (`worker/task_worker.py`).

```
Exception
├── ProviderError(provider, status_code, message, error_type="unknown", ratelimit_reset=None)
│   ├── ProviderServerError      # HTTP 5xx — and a synthesised 503 on a truncated output (finish_reason=length / MAX_TOKENS)
│   └── ProviderClientError      # HTTP 4xx — 429, 400 max_tokens, 401, 402, 413…
├── ProviderParseError(provider, raw, detail)   # JSON illisible, clé `agents` absente, item hors AgentResponse, boucle de répétition
└── ProviderCapacityError(provider, prompt_tokens_est, max_tokens_per_request)   # levée AVANT l'appel HTTP
```

`error_type` is the key of the `llm_provider_errors_by_type_total` family: the first ten
words of the provider's message (`extract_error_type`), `max_tokens_truncation` for a
truncation, `parse error …` for a parse. `ratelimit_reset` comes from `retry-after`,
`x-ratelimit-reset-tokens`, `x-ratelimit-reset-requests`, `x-ratelimit-reset`, or from the JSON
body of a 429 (Gemini: `error.details[].retryDelay` or "Please retry in 11.1s").

## What the worker does with each error

`process_batch_task` (`worker/task_worker.py`), in the order of the `except` clauses:

| Error | Reaction | Then |
|---|---|---|
| **No provider available** (`RuntimeError` from `select_provider`) | local wait `resilience.provider_wait_seconds` (8 s), one attempt every `saturation_poll_seconds` (2 s); `saturation_retries` (2) Celery `retry`s spaced by `saturation_retry_seconds` (12 s) | **eligible** = the pinned instance, otherwise the `instances_admises`, otherwise all (an excluded instance, free or not, does not count). The batch **keeps waiting** if an eligible one is merely **busy** (full RPM/TPM window, smoothing, concurrency; `[ALARME] Providers occupés` once, `alarme:providers_occupes`) or in a cooldown **that reopens before the client gives up** — always bounded by the client's wait (`resilience.client_wait_seconds` 120 s, or `wait_timeout` of the pinned instance, minus `client_wait_margin_seconds` 10 s, measured from the batch's first attempt). Otherwise, or if `abandon_when_busy`: `[ALARME] Tous les providers LLM saturés ou indisponibles`, `alarme:providers_satures`, **failure** of up to 100 tasks, **qualified**: `quota_journalier` if all eligible instances are at their daily quota (resume at the reset), `surcharge_fournisseur` otherwise (resume at the nearest known reopening), no kind if no eligible instance is known |
| `ProviderCapacityError` | RPM/TPM slot given back, `capacity_reroute_total:<p>` | **switch** |
| **Network error** (timeout, connection refused, truncated response) | classified by `BaseAdapter._post` as `ProviderServerError` 504 / 503 / 502, `error_type` `network_timeout`, `network_connect`, `network_protocol` | same path as a 5xx: cooldown and `retry` with backoff |
| `ProviderServerError` (5xx, truncation) | provider cooldown **60 s**, Celery `retry` after `min(backoff_base_seconds × 2^tentative, 30 s)` as long as `retries < max_retries` (50) **and** the client is still waiting after this delay | otherwise failure `error_kind="surcharge_fournisseur"`, `resume_at` = end of the cooldown, `[ALARME] Surcharge fournisseur`, `alarme:surcharge_fournisseur` — returned **before** the client's "Timeout expiré" |
| `ProviderClientError` **429** (per minute) | cooldown = delay announced by the provider (`_parse_ratelimit_reset_seconds`, +2 s margin, bounded to [10, 3600] s, default 60), same backoff and `retry`, same bound by the client's wait | otherwise failure `surcharge_fournisseur`, `resume_at` = end of the cooldown, same alarm |
| `ProviderClientError` **400 max_tokens** | if the message gives a limit N (Groq, OpenAI, Google formats) **and** it is stricter than the known one: `learn_provider_max_output_tokens` (in-memory config + `providers.yaml`), `retry` after 1 s | if the limit was already known: **switch** (so as not to loop on the same 400) |
| `ProviderClientError` other 4xx | — | **switch** |
| `ProviderParseError` | WARNING log with the first 300 characters of the raw output | **switch**; as a last resort, failure with `error = "<détail>\nRaw LLM response:\n<brut>"` |
| unexpected `RuntimeError` | `logger.exception` | failure |
| any other `Exception` | `logger.exception` | failure |

**Switch** (`_switch_provider_or_fail`): the faulty provider gets a cooldown of
`provider_switch_cooldown_seconds` (30 s), the batch is put back in the queue and replayed **without**
`force_provider` so that the rotation picks another model. Number of switches bounded to
`min(max_retries, max(1, len(providers) − 1))`: a truly invalid request ends up
failing on all providers.

In addition, whatever the type: each failed call increments the provider's consecutive
error counter; at **30**, the provider is deactivated for `disable_timeout` seconds
(180 by default) and the counter reset to zero. A success resets the counter to zero.

A failed task is persisted as `failed` with its `error`, published on Pub/Sub (the
long-poll returns immediately) and logged as ERROR: `Tâche échouée | task_id=…
error=…`.

## `[ALARME]` convention

A **confirmed** anomaly — threshold crossed, repeated error — is logged as ERROR with the
`[ALARME]` prefix, on **rising edge** (once per episode, not at each occurrence), and
counts in the Prometheus family `alarme_total{source}`:

- in a process that exposes `/metrics` (GAMA controller, SDK): `fire_alarme(source)`
  (`telemetry/alarms.py`), next to the `logger.error("[ALARME] …")`. The counter is **created at the
  first call**, never at import; if the family already exists in the registry (API
  process), it is kept out of the registry to avoid `DuplicateTimeseries`;
- in the Celery worker (no `/metrics`): `metrics.incr("alarme:<source>")` in Redis,
  read back by the API collector under the same family name.

`source` is a short and stable slug (low cardinality). `make error` from the repository
root lists the `[ALARME]` entries of the run's logs.

## Alarm sources in the code (2026-09-07)

| Where | Source / counter | Trigger | Re-arming |
|---|---|---|---|
| `worker/task_worker.py` | `alarme:providers_occupes` | after `saturation_retries`, the eligible providers are busy but not down: the batch waits for the window instead of being abandoned (rule of 2026-09-07, Prompt_Minimaliste run: half of the requests lost on an instance forced to 15 RPM) | rising edge: once per batch, at attempt `saturation_retries` |
| `worker/task_worker.py` | `alarme:providers_satures` | all eligible providers unavailable (cooldown, deactivation, quota) after the wait; the tasks of the queue fail | at each episode |
| `sdk/client.py` | `gateway_llm` | 10 tasks failed in a row on the client side; arms the backpressure | first success |
| `sdk/client.py` | `gateway_llm_circuit` | circuit breaker open (10 consecutive failures); submissions suspended | successful probe |
| `mobility_llm/categories/itinary_multi_agent.py` | `alarme:mode_label_mismatch` | more than 5 % of mode labels in disagreement over ≥ 200 checked options | never (the Redis counter persists: a single alarm per lifetime of the hash) |
| `config/settings.py` | log only | provider not found in `providers.yaml`, or file not writable, when persisting a learned `max_output_tokens` | — |
| `adapters/google_adapter.py` | log only | 3 consecutive `MAX_TOKENS` truncations on an instance | first clean completion |
| `mobility_llm/mode_choice.py` | log only | probability vector missing or summing to zero → uniform fallback, the model's decision is lost | — |
| `mobility_core/zone_resolver.py` | log only | more than 15 % of points outside the fine-zone layer over ≥ 200 points (expected ~5 %) | below 8 % |

The "log only" alarms do not count in `alarme_total`: they are visible in the
logs and through `make error`, not in Grafana.

## Error logs

- `<telemetry.workdir>/llm_errors.jsonl`: one line per failed LLM call (`time`, `task_id`,
  `provider`, `error_type`, `error_message`, `http_status`, `ratelimit_reset`).
- Redis ring buffer `llm:recent_errors` (latest 50), served by `GET /errors/recent`.
- `llm_call_completed | … status=failed` as ERROR at each failed call, `status=success`
  as INFO at each successful call: a silent worker is not a healthy worker.
