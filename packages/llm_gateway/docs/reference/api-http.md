# HTTP API

Six endpoints, defined in `api/routes.py`. The contract did not change with the split into
three packages: the GAMA controller and the SDK consume it as is. The dependencies (store,
queue, balancer, registry) are read from `request.app.state.deps`, composed by
`create_app()`.

## `POST /tasks` — submit a batch

Body: `LLMRequest`.

| Field | Type | Role |
|---|---|---|
| `category` | `str`, required | must be declared by a registered bundle |
| `agents` | list of items, **at least one** | each item carries `agent_id` (int accepted, converted to str); the other fields are validated by the category's item model |
| `parameters` | object, default `{}` | passed to the template; part of the batch key; `max_tokens` (per-task budget, default 4096 on the worker side), `temperature` (default 0.7), `prompt_variant` are read from it |
| `force_provider` | `str | None` | bypasses the rotation; part of the batch key |
| `min_tpm_required` | `int | None` | excludes the providers whose `tpm_limit` is lower; part of the batch key |
| `context` | `str | None` | global context (weather, traffic), available in the template |

Processing: the registry resolves the category, validates the items, computes the priority
score, persists the task (`PENDING`), adds it to the `<catégorie>:<md5>` queue and schedules
the dispatch — immediate if the queue reaches `get_dispatch_threshold` (10 by default), otherwise
delayed by `batch_delay_seconds` (3 s) with a SETNX flag that guarantees a single delayed
dispatch per cycle.

| Code | When | Body |
|---|---|---|
| **202** | task accepted | `{"task_id": "<uuid>", "status": "pending", "provider_used": null, "message": "Tâche acceptée. Pollez GET /tasks/<id> pour le résultat."}` |
| **422** | unknown category | `{"detail": "Catégorie inconnue 'x'. Catégories enregistrées : [...]"}` |
| **422** | item refused by the category's model | `{"detail": "Items invalides pour la catégorie 'x' : [erreurs pydantic]"}` |
| **422** | malformed body (FastAPI) | empty `agents`, missing `category`… |

## `GET /tasks/{task_id}` — status (polling)

| Code | Body |
|---|---|
| **200** | `TaskStatusResponse` (below) |
| **404** | `{"detail": "Tâche '<id>' introuvable ou expirée."}` |

## `GET /tasks/{task_id}/wait` — long-poll

Query parameter `timeout` (seconds, default 120, **bounded to [1, 300]**). Blocks
until the terminal state (`success` or `failed`) via Redis Pub/Sub, with reconnection if the
socket is interrupted before the end of the budget. On expiry, returns the last known
state (hence `pending` or `running`: it is up to the client to handle this case; the SDK returns it as
`failed` with `error="Timeout expiré"`).

| Code | Body |
|---|---|
| **200** | `TaskStatusResponse` |
| **404** | unknown task |

### `TaskStatusResponse`

| Field | Type | Filled by |
|---|---|---|
| `task_id` | `str` | API |
| `status` | `pending` · `running` · `success` · `failed` | API then worker |
| `created_at`, `updated_at` | datetime UTC | API / worker |
| `result` | list of `AgentResponse` or `null` | worker, on success: the elements of the response whose `agent_id` belongs to this task |
| `error` | `str | null` | worker, on failure |
| `provider_used` | `str | null` | worker |
| `latency_ms` | `float | null` | duration of the batch's LLM call |
| `timing_p5` | object or `null` | `P4_4_ms` queue wait, `P5_1_ms` wait for a provider, `P5_3_ms` prompt rendering, `P5_4_ms` LLM call, `P5_5_ms` demultiplexing, `provider`, `retries`, `tokens_in`, `tokens_out` (share of the task) |

`AgentResponse`: `agent_id` (str) and, all optional, `probabilities`
(list of `{index, mode, probability, reason}`), `chosen_index`, `mode`, `reason`,
`summary`; `extra="allow"` — every field of the category's schema is kept.

## `GET /health`

```json
{"status": "ok",
 "providers": {"mistral": {"current_rpm": 3, "rpm_limit": 60, "active_tasks": 1, "usage_pct": 5.0,
                           "cooldown": false, "daily_requests": 412, "rpd_limit": null,
                           "daily_tokens": 1203411, "tpd_limit": 100000000,
                           "quota_exhausted": false, "available": true}}}
```

`available` = neither deactivated, nor in cooldown, nor daily quota exhausted, current RPM under the
limit, active workers under `concurrency_limit`. Only the providers **with a key** appear.

`available` answers "can take a request *now*": it is also `false` when the
provider is merely **busy** (`active_tasks` ≥ `concurrency_limit`, the normal case of a local
model with one call at a time). A client that wants to know whether the provider is **out of service** reads
`disabled` and `cooldown`, not `available` — the experiment platform learned this on 2026-09-08.

## `GET /errors/recent`

Parameter `limit` (default 50, bounded to [1, 50]). The latest LLM errors read from the Redis ring
buffer `llm:recent_errors` (50 entries max), from most recent to oldest:

```json
[{"time": "2026-09-07T06:12:41+00:00", "provider": "groq_openai_120",
  "error_type": "rate limit reached for model …", "http_status": 429,
  "message": "[groq_openai_120] HTTP 429: …", "task_id": "batch_3f1c…_2"}]
```

Consumed by the "Dernières erreurs" panel of the Grafana cockpit (Prometheus only stores
numeric data).

## `GET /metrics`

Prometheus export (`text/plain; version=0.0.4`). Generation reads Redis
synchronously and is isolated in a thread. Catalogue: [Metrics](metriques.md).

## OpenAPI

The FastAPI application publishes `/docs` (Swagger UI), `/redoc` and `/openapi.json`. Outside
a server:

```python
from llm_gateway import create_app
schema = create_app().openapi()      # needs a reachable Redis for build_deps; otherwise pass in-memory deps=…
```

The declared title is still "LLM Unified Communication Module", version "1.0.0" (value
hard-coded in `api/app.py`, distinct from `llm_gateway.__version__`).

## CORS

`allow_origins=["*"]`, all methods, all headers — to be restricted in production;
authentication is ticket 036.
