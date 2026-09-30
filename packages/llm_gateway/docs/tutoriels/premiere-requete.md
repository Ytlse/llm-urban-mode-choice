# First request

From scratch to a validated LLM response, locally, with the `perception_filter` category of the
`mobility_llm` bundle. Duration: about ten minutes, most of it spent waiting for the models.

Prerequisites: Python 3.12, Docker (for Redis), at least one API key from a provider in
`providers.yaml` (Mistral, Groq, Google, Cerebras or OpenAI).

## 1. Install the packages

From the repository root:

```bash
python -m pip install -e ./packages/mobility_core -e "./packages/llm_gateway[test]" -e ./packages/mobility_llm
```

The order does not matter; `mobility_llm` declares `llm-gateway>=1.3` and
`mobility-core>=0.1` as dependencies. The editable install is enough for the
`llm_gateway.categories` entry point to be visible to `importlib.metadata`.

## 2. Start Redis

```bash
docker run -d --name redis-gateway -p 6379:6379 redis:7-alpine
```

The gateway expects Redis on `redis://localhost:6379/0` (tasks, queues, quotas), `/1`
(Celery broker) and `/2` (Celery results). These three URLs are settings
(`LLM_GATEWAY_REDIS__URL`, `LLM_GATEWAY_EXECUTOR__CELERY_BROKER_URL`, `LLM_GATEWAY_EXECUTOR__CELERY_RESULT_BACKEND`; the old unprefixed names are still read with a warning).

## 3. Provide an API key

Keys are never in `providers.yaml`: they come from the environment, one per
provider, in the form `PROVIDER_KEYS__<nom>`. The name is that of the instance in
`providers.yaml` (`google_gemini31_key2`), otherwise that of the adapter (`mistral`, `groq`, `google`,
`cerebras`, `openai`).

```bash
cp packages/llm_gateway/.env.example .env      # toutes les variables lues, commentées
export PROVIDER_KEYS__mistral=...      # ou éditer .env puis `set -a; source .env`
llm-gateway config validate
```

Expected: `OK — 1 provider(s) avec clé : mistral`. Instances without a key are set aside at
startup with a warning; they block nothing.

## 4. Look at what the gateway can serve

```bash
llm-gateway categories
```

With `mobility_llm` installed:

```
itinary_multi_agent   (bundle=mobility, items=AgentSpec)
ltm_self_reflection   (bundle=mobility, items=AgentSpec)
perception_filter     (bundle=mobility, items=AgentSpec)
stm_reflection        (bundle=mobility, items=AgentSpec)
```

!!! note "Without a bundle, nothing is served"
    If you only install `llm_gateway`, the command prints "No category registered
    (no bundle under the llm_gateway.categories entry point)." and returns code 1. The API
    starts anyway, but **every** `POST /tasks` request gets a 422:
    `Catégorie inconnue 'perception_filter'. Catégories enregistrées : []`. The gateway
    contains no prompt; this is intended ([ADR 0002](../adr/0002-hook-de-categorie.md)).

## 5. Start the API and the worker

Two terminals:

```bash
llm-gateway serve --port 8000
```

```bash
llm-gateway worker --loglevel info --concurrency 25 --pool threads
```

These are the same commands as in `docker-compose.yml` (`uvicorn llm_gateway.main:app` and
`celery -A llm_gateway.worker.task_worker.celery_app worker … -P threads`). At startup,
the API resets the RPM/TPM windows and the registry logs
`Bundle de catégories enregistré | bundle=mobility categories=[…]`.

```bash
curl -s localhost:8000/health | python -m json.tool
```

Each provider with a key appears with `current_rpm`, `rpm_limit`, `available`.

## 6. Submit a batch

`perception_filter` turns a persona's profile into a short first-person
narrative. Its item model is `AgentSpec`: only `agent_id` and `perception` are
required, everything else has a default value.

```bash
curl -s -X POST localhost:8000/tasks -H 'Content-Type: application/json' -d '{
  "category": "perception_filter",
  "agents": [
    {"agent_id": "ag_1", "perception": "34 ans, cadre, Compans-Caffarelli, sans voiture, abonnée Tisséo."}
  ]
}'
```

Immediate response, code **202**:

```json
{"task_id": "3f1c…", "status": "pending", "provider_used": null,
 "message": "Tâche acceptée. Pollez GET /tasks/3f1c… pour le résultat."}
```

What just happened on the API side: the category was found in the registry, the item
validated by `AgentSpec`, the task persisted, then added to its batch's queue. A single
task is below the immediate dispatch threshold (10): the dispatch is scheduled in
`batch_delay_seconds` = 3 s to let other compatible requests aggregate.

## 7. Wait for the result

```bash
curl -s "localhost:8000/tasks/<task_id>/wait?timeout=120" | python -m json.tool
```

Long-poll on Redis Pub/Sub: the response arrives as soon as the worker publishes the terminal state.
Expected, after the 3 s window plus the model latency:

```json
{
  "task_id": "3f1c…",
  "status": "success",
  "result": [{"agent_id": "ag_1", "summary": "I'm 34, an executive living near Compans-Caffarelli…"}],
  "provider_used": "mistral",
  "latency_ms": 1843.2,
  "timing_p5": {"P4_4_ms": 3012.4, "P5_1_ms": 0.3, "P5_3_ms": 4.1, "P5_4_ms": 1843.2, "P5_5_ms": 0.1,
                "provider": "mistral", "retries": 0, "tokens_in": 412, "tokens_out": 96}
}
```

`P4_4_ms` is the wait in the queue (the 3 s window), `P5_4_ms` the LLM call. The
`summary` field is the one required by this category's output schema (`categories/perception_filter/output_schema.json`); a response outside the schema
would have been replayed on another provider and, failing that, returned as `status: failed` with the
raw output in `error`.

## 8. See what was sent

The worker logs each exchange (rendered prompt, response, tokens) as JSONL in
`<telemetry.workdir>/llm_exchanges.jsonl` (`LLM_GATEWAY_TELEMETRY__WORKDIR`, the worker's current directory by default) and the errors
in `llm_errors.jsonl`. `GET /metrics` exposes the Prometheus counters, including
`llm_prompts_sent_total{category="perception_filter"}` and `llm_agents_batched_total`.

## 9. The same thing from Python

```python
import asyncio
from llm_gateway import LLMGatewayClient, LLMRequest

async def main() -> None:
    client = LLMGatewayClient("http://localhost:8000", wait_timeout=90.0, dialogue_log_file=None)
    result = await client.execute(LLMRequest(
        category="perception_filter",
        agents=[{"agent_id": "ag_1", "perception": "34 ans, cadre, sans voiture, abonnée Tisséo."}],
    ))
    print(result.status, result.provider_used, result.agents[0].summary if result.ok else result.error)
    await client.aclose()

asyncio.run(main())
```

`dialogue_log_file=None` disables the local dialogue log (by default
`prompt_dialogue.log` in the current directory, which contains the payloads sent — hence
potentially personal data). Details: [Python SDK](../reference/sdk-python.md).

## What can go wrong

| Symptom | Likely cause |
|---|---|
| `422 Catégorie inconnue … Catégories enregistrées : []` | no bundle installed in the environment of the **API** |
| `422 Items invalides pour la catégorie 'perception_filter'` | `perception` is missing (`AgentSpec` model) |
| `status: failed`, `Providers saturés ou indisponibles après 8s` | no provider with a key, or all in cooldown/quota: see `/health` |
| the response takes 3 s longer than expected | accumulation window `batch_delay_seconds` — normal for an isolated request |
| `Fournisseur 'x' exclu : clé API manquante` | `PROVIDER_KEYS__x` missing; the name must be that of the instance or of the adapter |
