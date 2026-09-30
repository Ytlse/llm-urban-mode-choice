# Tune the quotas

The gateway works with free-tier providers whose quotas are small and different. This
guide says which setting bounds what, how to read a `batch_max_agents` of 2, and how to
refresh the figures. The default values are those of `config/settings.py` and of
`config/providers.yaml`; the full list is in [Settings](../reference/reglages.md).

## The four quotas of a provider

| Field | Window | How it is applied |
|---|---|---|
| `rpm_limit` | sliding 60 s | atomic reservation before each call (Lua script); smoothing: two calls less than `60 / rpm` s apart are refused (at 15 RPM: 4 s) |
| `tpm_limit` | sliding 60 s | reservation of the **estimated** tokens before the call, readjusted on the rendered prompt then on the actual consumption; overflow → refusal, the balancer moves on to the next provider |
| `rpd_limit` | provider's day | counted at reservation; reached → provider set aside until its reset (`quota_exhausted`), no longer re-probed |
| `tpd_limit` | provider's day | counted **after** the call on the billed tokens; same setting aside |
| `quota_reset_tz` | — | time zone where the provider places midnight. Default `UTC`; `America/Los_Angeles` on the Google instances |

A failed call gives back its RPM and TPM reservation: errors do not consume quota.
The daily counters are Redis keys with a 25 h TTL; the 60 s windows are reset to zero
at API startup (lifespan), never by a worker.

### The time zone of the daily reset matters

The Gemini free tier places midnight in **Pacific** time: its billing day
ends at 09:00 in Paris in summer, 08:00 in winter. Counting in UTC therefore emptied the counters seven
hours too early. On 2026-09-08, at 08:44, `/health` announced 49 requests out of 500 while
Google was refusing for exceeding the 500 — the instance looked available and an experiment
waited fifteen minutes on a closed key. Hence `quota_reset_tz`, per instance.

### The local counter is only a safeguard

It only sees the traffic of this gateway, whereas a key is also consumed by
`scripts/synthesis/*` and `prompt_calibration`: it undercounts by construction. What is
authoritative is the provider's response. A 429 whose body designates a **daily** quota
(`quotaId` in `…PerDay…`) sets the instance aside until the reset, without consulting the counter.

The `retryDelay` that Gemini then returns is to be thrown away: 0.7 s to 57 s recorded for a window that
only reopened seven hours later. It is only taken into account for a per-minute rate 429,
where it is correct.

## How many agents in a batch

The computation is done once, when `Settings` is built, for each provider:

```
tokens_per_agent = assumed_prompt_tokens + assumed_output_tokens     = 2 200 + 800 = 3 000
tpm_bound        = tpm_limit // tokens_per_agent    (rpm_limit si pas de tpm_limit)
req_bound        = max_tokens_per_request // tokens_per_agent    (max_batch_agents si absent)
batch_max_agents = max(1, min(tpm_bound, req_bound, rpm_limit, max_batch_agents = 20))
tpm_estimate_per_request = batch_max_agents × tokens_per_agent    (si tpm_limit)
```

On the shipped file:

| Provider | `tpm_limit` | `max_tokens_per_request` | `rpm_limit` | → `batch_max_agents` | TPM estimate/request |
|---|---|---|---|---|---|
| `mistral` | 500 000 | — | 60 | min(166, 20, 60, 20) = **20** | 60 000 |
| `openai` | 200 000 | — | 15 | min(66, 20, 15, 20) = **15** | 45 000 |
| `google_gemini31_key1` | 250 000 | — | 15 | **15** | 45 000 |
| `google_gemma42_key1` | 16 000 | 16 000 | 30 | min(5, 5, 30, 20) = **5** | 15 000 |
| `groq_openai_120_key1` | 8 000 | 8 000 | 30 | min(2, 2, 30, 20) = **2** | 6 000 |
| `cerebras_gptoss120b_key1` | 30 000 | — | 5 | min(10, 20, 5, 20) = **5** | 15 000 |

These values can be read in the log at startup (`Provider 'x' — batch_max_agents=… tpm_estimate_per_request=…`)
and in `llm-gateway config show`.

**Exclusion at startup.** A provider with `max_tokens_per_request` is set aside if
`max_tokens_per_request < batch_max_agents × assumed_prompt_tokens + min_output_tokens`: for
`groq_openai_120_key1`, 2 × 2,200 + 512 = 4,912 ≤ 8,000, it stays. The message is a WARNING
"Provider 'x' excluded: not enough capacity (…)" with the computation.

**Two thresholds, two roles.** The API decides *when* to dispatch with `get_dispatch_threshold`:
`batch_target_agents` (10) bounded by the **largest** provider — not the smallest, otherwise the
threshold would be 1 and the accumulation window would never be used. The worker decides *how many*
to take with the `batch_max_agents` **of the provider it has just reserved**. A batch of 10
queued tasks can therefore leave in one call to Mistral or in five to Groq.

## The accumulation window

`batch_delay_seconds` = 3 s. An isolated request waits for this delay before leaving; it is the
`P4_4_ms` visible in `timing_p5`. The setting is based on a measurement of the 2026-07-10 run
(prompt inter-arrival p50 = 1.4 s, p90 of throughput 40 prompts/min): at 1 s the merge
captured almost nothing, at 3 s the batching rate (`llm_agents_batched_total /
llm_prompts_sent_total`) rises. Lengthening it adds latency to each decision; shortening it
multiplies the calls and hence the RPM pressure.

## Token estimates, and how they were measured

Three numbers drive the TPM reservation; they are **measured**, not chosen:

| Setting | Value | Measurement (comments of `settings.py`) |
|---|---|---|
| `assumed_prompt_tokens` | 2 200 | history of `tokens_in` per agent + 10 % margin |
| `assumed_output_tokens` | 800 | 2026-07-10 run: 1,282 input tokens + 320 output ≈ 1,600 per agent, p90 ≈ 2,400; 3,000 keeps ~25 % margin. A hard-coded 4,096 gave 6,296/agent and crushed `batch_max_agents` to 1 on the small TPMs |
| `token_chars_ratio` | 3.0 | 2026-07-10 run, 427 exchanges (FR prompts + JSON): p50 = 3.24, p10 = 3.05 → 3.0 leaves ~8 % margin |

The worker readjusts the reservation in two steps: at rendering, `prompt_chars / 3,0 + n_agents ×
800` replaces the flat rate (an STM reflection at ~4,500 input tokens per agent reserves its
true cost, a small batch gives back headroom); after the response, the billed consumption
replaces the estimate. Two WARNINGs flag a drift: `tokens_in dépasse
assumed_prompt_tokens` (per agent) and `Estimation TPM sous-évaluée de N tokens` (actual >
reserved × 1.25). Seeing them often = remeasure on `llm_exchanges.jsonl` and adjust.

## The output budget

```
max_tokens = min(parameters.max_tokens (défaut 4096) × n_agents,
                 settings.max_output_tokens (16 384),
                 provider.max_output_tokens)
max_tokens = min(max_tokens, max_tokens_per_request − estimated_prompt)     # if the provider has a per-request capacity
```

If `max_tokens_per_request − prompt_estimé < min_output_tokens` (512), the batch does not leave:
`ProviderCapacityError`, slot given back, switch to another provider,
`llm_capacity_reroute_total` incremented. This is the 413 safeguard: on the 2026-07-11 run, 38
requests left doomed towards the small quotas because the static estimate
underestimated the reflection prompts by a factor of 2.

The per-provider `max_output_tokens` also bounds the selection: a request with
`parameters.max_tokens = 8000` will not go to a provider capped at 4,096.

### `max_output_tokens` is learned

When a provider answers HTTP 400 with its limit ("`max_tokens` must be less than or equal
to 8192" at Groq, "supports at most 16384 completion tokens" at OpenAI, "…limited to
8192" at Google), `learn_provider_max_output_tokens`:

1. updates the config of the current process (the next batch is capped);
2. stores the value in the **learned limits store** (port `LearnedLimits`): Redis hash
   `llm_gateway:learned:max_output_tokens` shared between the API and the workers
   (`LLM_GATEWAY_LEARNED_LIMITS=redis`, default), JSON file without Redis (`file`,
   `LLM_GATEWAY_LEARNED_LIMITS_FILE`), or nothing (`none`);
3. returns `True` if the limit is new and stricter: the batch is replayed after 1 s.
   Otherwise `False`: the worker switches to another provider instead of looping on the same
   400.

At startup, each process merges the learned limits on top of the providers
file (`apply_learned_limits`): a learned limit can only **tighten** a declared cap,
never widen it. The providers file is never rewritten by the
gateway any more; to make a learned limit permanent, copy it into `max_output_tokens`.

!!! note "If the store is unreachable"
    Learning adjusts the in-memory config, replays the batch, and logs
    `[ALARME] Impossible de mémoriser la limite apprise`: the other processes will relearn it
    at their first 400. Nothing is lost, only relearned.

## Refresh the quotas: `make providers`

From the repository root, `scripts/providers/refresh.py` reads the **actual** quotas and
rewrites `config/llm_gateway/providers.yaml` (surgical edit, comments preserved):

```bash
make providers DRY_RUN=1      # bilan sans écrire
make providers                # écrit
```

| Adapter | Source | Fields updated |
|---|---|---|
| mistral, groq, cerebras | a probe request (`max_tokens=1`) → `x-ratelimit-*` headers | mistral: `rpm_limit` (bounded to 60, documented rate 1 req/s), `tpm_limit`, `tpd_limit` forced to 3 × (1 bn / 30) = 100 M tokens/day (safeguard: the free tier is monthly, not exposed); groq: `rpd_limit`, `tpm_limit` (RPM and TPD absent from the headers); cerebras: all four |
| google | Cloud Quotas API (`gcloud auth print-access-token`) | `rpm_limit`, `tpm_limit`, `rpd_limit` per model family |
| openai, or any instance without a key | — | skipped with a warning |

The script recomputes `weight = min(rpm, tpm/3000)/15` as soon as `rpm_limit` or `tpm_limit`
changes, aligns `max_tokens_per_request` on `tpm_limit` when the field exists, appends at the end
of the file any new text model found by `GET /models` (in rotation if RPD ≥ 100, otherwise
`weight: 0`), comments out with the date a `default_model` that has disappeared, and never deletes or relaxes
anything silently: a failed probe leaves the instance intact with an `[ALARME]`
in the report. A block already present, even commented out, is never re-added.

## Read the current state

- `GET /health`: `current_rpm`, `daily_requests`, `daily_tokens`, `cooldown`,
  `quota_exhausted`, `available` per provider.
- `GET /metrics`: `llm_provider_state`, `llm_provider_daily_usage_ratio`,
  `llm_provider_quota_exhausted`, `celery_worker_utilization_ratio`,
  `llm_task_queue_depth_by_category`.
- Worker log: `Quota journalier RPD épuisé (compteur local) — provider écarté jusqu'au reset |
  provider=… used=… limit=… reset_in=…s`.
