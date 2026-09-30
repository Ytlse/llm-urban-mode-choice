# Prometheus metrics

Two processes expose metrics. The **API** serves `GET /metrics`: its own
counters, plus the **worker**'s counters read back from Redis by a collector (the Celery
worker has no HTTP server). The **SDK** declares its own in the consuming
process (the GAMA controller, scraped on `:8002`).

## How the worker counts

The worker writes into the Redis hash `wmetrics` (`HINCRBY`), with names
`metrique:label[:label2]` — `llm_calls_ok_total:mistral`, `mode_by_distance:car:2-5km`. The
`WorkerMetricsCollector` collector (`api/metrics.py`) does **a single `HGETALL` per
scrape** and projects each prefix onto a family. The counters have no TTL: they
survive restarts, a `FLUSHDB` resets them to zero.

## API side — direct counter

| Family | Type | Labels | Meaning |
|---|---|---|---|
| `gama_agents_received_total` | counter | `category` | agents received in `POST /tasks`, before grouping |

## API side — read back from Redis, generic

| Family | Labels | Redis key | Meaning |
|---|---|---|---|
| `llm_provider_calls_ok_total` | `provider` | `llm_calls_ok_total:<p>` | successful LLM calls; configured providers without calls appear at 0 |
| `llm_provider_calls_err_total` | `provider` | `llm_calls_err_total:<p>` | failed calls |
| `llm_prompts_sent_total` | `category` | `prompts_sent_total:<c>` | prompts sent (one per batch) |
| `llm_agents_batched_total` | `category` | `agents_batched_total:<c>` | agents merged into these prompts; the ratio of the two is the batching rate |
| `llm_tokens_in_total` | `provider` | `tokens_in_total:<p>` (and `__all__`) | billed input tokens |
| `llm_tokens_out_total` | `provider` | `tokens_out_total:<p>` (and `__all__`) | output tokens |
| `llm_provider_errors_by_type_total` | `provider`, `error_type` | `llm_errors_by_type:<p>:<type>` | `error_type` = first 10 words of the provider's message (`rate limit reached for model…`), `max_tokens_truncation`, `parse error …` |
| `llm_capacity_reroute_total` | `provider` | `capacity_reroute_total:<p>` | batches replayed elsewhere because the rendered prompt exceeded `max_tokens_per_request` (413 avoided) |
| `alarme_total` | `source` | `alarme:<source>` | `[ALARME]` alarms of the worker and of the bundles; known sources: `providers_satures`, `mode_label_mismatch` |

## API side — read back from Redis, declared by the bundles

The gateway knows none of these families: each category bundle **declares** the
counters that its `observe` hook feeds (`CategoryBundle.metric_families`, one
`MetricFamilySpec` per family: Prometheus name, help, Redis prefix, labels) and the collector
renders them generically from the `<préfixe>:<label1>:<label2>…` keys of the hash. A gateway without
a bundle exposes nothing more than the generic families above. Those of the
`mobility_llm` bundle (`METRIC_FAMILIES`) keep the names read by Grafana dashboards 04 and 07:

| Family | Labels | Redis key | Meaning |
|---|---|---|---|
| `llm_transport_mode_chosen_total` | `mode` | `transport_mode_chosen:<m>` | main mode of the most probable option (fine vocabulary: `metro`, `tram`, `bus`, `train`, `car`, `cycling`, `walking`…) |
| `llm_mode_probability_pct_total` | `mode` | `mode_probability_pct:<m>` | sum of the probabilities (in %) per **canonical** mode (`public_transport`, `train`, `car`, `cycling`, `walking`, `motorbike`, `other`), modes not offered included at 0 |
| `llm_mode_label_checked_total` | — | `mode_label_checked` | options whose mode label copied by the LLM was compared with the actual option |
| `llm_mode_label_mismatch_total` | — | `mode_label_mismatch` | disagreements; expected ratio ≈ 0 |
| `llm_trip_distance_bracket_total` | `bracket` | `trip_distance_bracket:<b>` | `0-1km`, `1-2km`, `2-5km`, `5-10km`, `10-20km`, `20-50km`, `>50km` |
| `llm_mode_by_distance_total` | `mode`, `bracket` | `mode_by_distance:<m>:<b>` | cross-tabulation |
| `llm_mode_by_provider_total` | `mode`, `provider` | `mode_by_provider:<m>:<p>` | cross-tabulation |
| `llm_chosen_index_total` | `index` | `chosen_index:<i>` | position of the chosen option (0 = first offered) |

## API side — provider state (gauges, computed at scrape)

| Family | Labels | Meaning |
|---|---|---|
| `llm_provider_state` | `provider` | 0 = no API key, 1 = temporarily deactivated (30 consecutive errors), 2 = cooldown, 3 = active; all providers of `providers.yaml` appear |
| `llm_provider_disable_ttl_seconds` | `provider` | seconds before reactivation (max of the deactivation TTL and the cooldown) |
| `llm_task_queue_depth` | `batch_key` | `PENDING` tasks per `<catégorie>:<md5>` queue |
| `llm_task_queue_depth_by_category` | `category` | same, aggregated by key prefix |
| `celery_worker_utilization_ratio` | `provider` | `active_workers / concurrency_limit` (1.0 = saturated) |
| `llm_provider_info` | `provider`, `model`, `adapter` | always 1; carries the labels |
| `llm_provider_rpm_limit`, `llm_provider_rpd_limit`, `llm_provider_tpd_limit` | `provider` | configured limits (0 = unlimited) |
| `llm_provider_requests_today`, `llm_provider_tokens_today` | `provider` | consumption of the UTC day |
| `llm_provider_daily_usage_ratio` | `provider` | `requests_today / rpd_limit` (absent if no `rpd_limit`) |
| `llm_provider_quota_exhausted` | `provider` | 1 if set aside until the reset of its day (`quota_reset_tz`) |

## SDK side

Declared at import of `llm_gateway.sdk.client`:

| Family | Type | Labels | Meaning |
|---|---|---|---|
| `llm_task_e2e_duration_seconds` | histogram (1, 2, 5, 10, 30, 60, 120 s) | `category` | `POST /tasks` → terminal state, seen from the client |
| `llm_gateway_circuit_open` | gauge | — | 1 = circuit breaker open |
| `llm_gateway_circuit_waiters` | gauge | — | suspended submissions |
| `alarme_total` | counter | `source` | created at the first `fire_alarme`; sources `gateway_llm`, `gateway_llm_circuit`, plus those of the controller (`backlog`, `event_loop`, `cache_llm`…) |

`alarme_total` carries the **same name** on both sides: on the API side it is served by the
Redis collector, on the controller side by the SDK counter. In a process that already exposes
the family (the API itself, if it imports the SDK), the SDK counter is kept out of the registry
to avoid `DuplicateTimeseries` — the alarms remain visible in the logs.

## What is not a metric

The raw error messages (text) are in the ring buffer `llm:recent_errors`, served by
`GET /errors/recent`. The full exchanges (prompt, response, tokens, `sim_ts`) are in
`<telemetry.workdir>/llm_exchanges.jsonl`, the errors in `llm_errors.jsonl`.
