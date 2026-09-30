# Python SDK

`llm_gateway.sdk.client.LLMGatewayClient`, re-exported by the facade: `from llm_gateway import
LLMGatewayClient, TaskResult, TaskTiming`. Asynchronous (httpx), a single `AsyncClient`
reused between calls; `aclose()` when the consumer stops. Detailed signatures:
[Python API](api-python.md).

## Constructor

```python
LLMGatewayClient(
    base_url="http://localhost:8000", *,
    wait_timeout=120.0, dialogue_log_file=None, transport=None,
    backpressure_max_inflight=0, backpressure_release_ratio=0.2,
    circuit_failure_threshold=10, circuit_probe_interval=60.0,
)
```

| Parameter | Default | Role |
|---|---|---|
| `base_url` | `http://localhost:8000` | root of the gateway; the trailing `/` is removed |
| `wait_timeout` | `120.0` s | value passed to `GET /tasks/{id}/wait?timeout=`; the httpx read timeout is `wait_timeout + 30` |
| `dialogue_log_file` | `None` | text file where each request **and** its response are appended; disabled by default since 1.3.0, opt-in |
| `transport` | `None` | injectable `httpx.AsyncBaseTransport` (tests: `MockTransport`) |
| `backpressure_max_inflight` | `0` (disabled) | when the "10 consecutive failures" alarm is active, new submissions wait for the in-flight tasks to fall back under `release_ratio × max_inflight` |
| `backpressure_release_ratio` | `0.2` | release threshold (at least 1) |
| `circuit_failure_threshold` | `10` | consecutive failures that open the circuit breaker; `0` disables |
| `circuit_probe_interval` | `60.0` s | interval between two probes when the circuit breaker is open |

!!! warning "`prompt_dialogue.log` contains potential personal data"
    The dialogue log writes the full payload (hence the personas' profiles: age,
    occupation, place of residence, history) and the model's response, in clear text, in the
    current directory of the process. It is not versioned (`*.log` in `.gitignore`) and,
    since 1.3.0, it is **disabled by default**: only pass it when debugging, with a path
    in the run folder. The worker-side exchange log follows the same rule
    (`telemetry.exchanges_enabled`, redactor `telemetry.redactor`).

## `execute(request) -> TaskResult`

`request`: an `LLMRequest` or a `dict` of the same shape. Sequence: passage through the circuit breaker
→ possible backpressure wait → `POST /tasks` (3 attempts on a transient network
error, waiting 1 s then 2 s) → `GET /tasks/{id}/wait` → metrics and log.

Does **not** raise on a task failure: the `TaskResult` carries `status` and `error`. Raises
`httpx.HTTPStatusError` if the gateway refuses the submission with a **4xx** (invalid payload
or unknown category: programming error, not counted as a failure). A **5xx** at
submission or a gateway unreachable after 3 attempts return a `TaskResult` as `failed`
and count as a failure (alarm, backpressure, circuit breaker).

### `TaskResult`

| Field | Type | Meaning |
|---|---|---|
| `status` | `TaskStatus` | `success` or `failed`; an expired long-poll without a terminal state becomes `failed` with `error="Timeout expiré"` |
| `agents` | `list[AgentResponse]` | the results of this task, aligned on its `agent_id`s |
| `error` | `str | None` | message from the worker or the client (`Gateway error 503 à la soumission`, `Gateway LLM injoignable (ConnectError)`, `Réponse gateway non-JSON (HTTP …)`) |
| `provider_used` | `str | None` | provider instance that served the batch |
| `timing` | `TaskTiming | None` | `post_ms` (duration of the POST), `wait_ms` (duration of the long-poll), `timing_p5` (worker segments) |
| `task_id` | `str | None` | identifier on the gateway side |
| `ok` | property | `status is SUCCESS and bool(agents)` |

## Circuit breaker

After `circuit_failure_threshold` consecutive failures, the circuit breaker opens: calls to
`execute` are **suspended** (they wait, they do not fail, no decision is
degraded). Every `circuit_probe_interval` seconds, one of the waiting callers becomes
the probe and goes through; if it succeeds, the circuit breaker closes again and all the suspended calls
resume on the nominal path; if it fails, the next probe waits a full
interval. Any success resets the failure counter to zero.

This policy is the project's: under quota shortage, we wait for the renewal, we do not
replace the model's decision with a fallback.

Log: `[circuit] Sonde vers le gateway LLM…`, `[circuit] Gateway LLM rétabli après Ns…`.
Alarms: `[ALARME] Gateway LLM : 10 tâches échouées d'affilée…` (source `gateway_llm`) then
`[ALARME] Disjoncteur gateway LLM OUVERT…` (source `gateway_llm_circuit`), both on
rising edge.

## Backpressure

Independent of the circuit breaker and **disabled by default** (`backpressure_max_inflight=0`).
When it is armed by the 10-failure alarm, a new submission waits for the
number of in-flight tasks to fall back to `max(1, int(max_inflight × release_ratio))`, then the
backpressure disarms and the backlog resumes at once. The first success disarms it too.
The circuit breaker's probe is not subject to it.

## Client-side Prometheus metrics

Declared at import of `llm_gateway.sdk.client` (not of the `llm_gateway` facade) in the
default registry of the consuming process — it is the GAMA controller, scraped on `:8002`,
that exposes them:

| Family | Type | Labels | Meaning |
|---|---|---|---|
| `llm_task_e2e_duration_seconds` | histogram, buckets 1 · 2 · 5 · 10 · 30 · 60 · 120 s | `category` | duration `POST /tasks` → terminal state |
| `llm_gateway_circuit_open` | gauge 0/1 | — | circuit breaker open |
| `llm_gateway_circuit_waiters` | gauge | — | submissions suspended behind the circuit breaker |
| `alarme_total` | counter | `source` | created at the first `fire_alarme`, see [Errors and alarms](erreurs-alarmes.md) |

The counters per mode, per index or per provider are exposed on the worker side, not here.

## What the SDK does not do

It has no synchronous API. It does not know the categories (it does not validate the payload
before sending: it is the API that answers 422). It does not draw the mode from the distribution
returned by `itinary_multi_agent`: that is `mobility_llm.mode_choice.draw_index`, on the
simulation side.
