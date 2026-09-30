# llm-gateway

Asynchronous multi-provider gateway for structured LLM calls. A client submits a batch
of items per **category** (`POST /tasks`), the gateway groups compatible batches into a
single prompt (micro-batching), chooses a provider according to its quotas (SWRR, atomic
RPM/TPM reservation in Redis), calls the model, validates the JSON response against the
category's schema and hands each result back to its `agent_id`. The gateway **contains no
prompt**: templates, schemas and domain rules are brought by *bundles* installed
alongside it.

Version: `llm_gateway.__version__` = 1.3.0 · Python ≥ 3.12 · licence: Apache-2.0
(see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

## The three sibling packages

| Package | Role | Depends on |
|---|---|---|
| `llm_gateway` (this folder) | FastAPI API, Celery worker, provider adapters, balancer, content-free Jinja2 engine, telemetry, SDK | nothing in the repository |
| `mobility_core` | domain of the EMC² Toulouse survey: rings, fine zones, mode hierarchy, bicycle, housing, population calibration | nothing in the repository |
| `mobility_llm` | the four LLM categories of the mobility simulation: persona, templates, schemas, prompt variants, mode choice, domain metrics; registers with the gateway through an entry point | the other two |

The rule is checked by import-linter (`.importlinter` at the root, 5 contracts):
`llm_gateway` imports neither `mobility_core`, nor `mobility_llm`, nor the old
`llm_module` shell; `llm_gateway.core` and `llm_gateway.ports` import no infra.

## Installation

From the repository root, in editable mode:

```bash
python -m pip install -e ./packages/mobility_core -e "./packages/llm_gateway[test]" -e ./packages/mobility_llm
```

Gateway extras: `test` (pytest, fakeredis[lua], hypothesis), `dev` (ruff, mypy,
import-linter, pre-commit), `docs` (mkdocs-material, mkdocstrings), `monitoring` (flower).

Docker image (context = repository root, the three packages embedded, non-root user):

```bash
docker build -f packages/llm_gateway/Dockerfile .
```

## Quick start

```bash
docker run -d -p 6379:6379 redis:7-alpine        # 1. Redis
cp packages/llm_gateway/.env.example .env         # 2. clés : PROVIDER_KEYS__<nom>=...
llm-gateway config validate                       #    → « OK — N provider(s) avec clé »
llm-gateway categories                            # 3. categories brought by the bundles
llm-gateway serve --port 8000                     # 4. API (uvicorn)
llm-gateway worker --concurrency 25 --pool threads   # 5. worker Celery, autre terminal
```

Equivalents used by `docker-compose.yml` (services `api` and `worker`):
`uvicorn llm_gateway.main:app --host 0.0.0.0 --port 8000` and
`celery -A llm_gateway.worker.task_worker.celery_app worker --loglevel=info --concurrency=25 -P threads`.

`GET /health` lists the RPM state of each provider; `GET /metrics` exposes Prometheus.

## Categories come from bundles

The gateway discovers its categories through the `llm_gateway.categories` entry point. A bundle
is a `CategoryBundle` (directory of `<catégorie>.md.j2` templates, output schemas
file, optional `prompts.yaml`) that declares `CategorySpec`s (item model,
priority function, metrics observation hook). The registry validates each
category **at startup**: missing template or schema → `ValueError` before the first
request.

With no bundle installed, `llm-gateway categories` returns code 1 and **every request is
refused with 422** ("Unknown category … Registered categories: []"). The
`mobility_llm` package brings `itinary_multi_agent`, `perception_filter`, `stm_reflection` and
`ltm_self_reflection`. To write your own: `docs/guides/ajouter-une-categorie.md`.

## Python SDK

```python
import asyncio
from llm_gateway import LLMGatewayClient, LLMRequest

async def main() -> None:
    client = LLMGatewayClient("http://localhost:8000", wait_timeout=90.0, dialogue_log_file=None)
    result = await client.execute(LLMRequest(
        category="perception_filter",
        agents=[{"agent_id": "ag_1", "perception": "34 ans, cadre, sans voiture, abonnée TC."}],
    ))
    if result.ok:
        print(result.provider_used, result.agents[0].summary, result.timing.wait_ms)
    else:
        print(result.status, result.error)
    await client.aclose()

asyncio.run(main())
```

`execute` does not raise on a task failure (`TaskResult.status`/`error` carry it); it
raises `httpx.HTTPStatusError` if the gateway refuses the submission (4xx). The client's
circuit breaker suspends submissions after 10 consecutive failures and re-probes every 60 s —
details in `docs/reference/sdk-python.md`.

## Tree

```
llm_gateway/
├── pyproject.toml            # version dynamique (llm_gateway.__version__), extras, pytest, coverage ≥ 80 %
├── Dockerfile                # image api/worker/flower — contexte : racine du dépôt
├── mkdocs.yml, docs/         # site de documentation (mkdocs-material, français)
├── tests/                    # unit / contract / integration / e2e (marqueurs par dossier)
└── src/llm_gateway/
    ├── __init__.py           # façade paresseuse : LLMGatewayClient, LLMRequest, CategoryBundle…
    ├── cli.py                # llm-gateway serve | worker | config validate|show | categories
    ├── main.py               # app = create_app() pour uvicorn
    ├── core/                 # domaine pur : models, batching (clé de lot), selection (SWRR)
    ├── ports/                # Protocol : TaskStore, BatchQueue, RateLimiter, MetricsSink, LLMAdapter, category
    ├── infra/redis/, infra/memory/   # implémentations des ports (Lua Redis / pur Python)
    ├── adapters/             # openai, mistral, google, groq, cerebras — @register_adapter
    ├── balancer/router.py    # LoadBalancer : séquence SWRR + réservation via RateLimiter
    ├── prompts/              # engine (Jinja2, no content), registry (bundles validated at startup)
    ├── api/                  # create_app, routes, deps (build_deps), metrics (collecteur Redis)
    ├── worker/               # create_celery_app, runtime (build_worker_runtime), task_worker
    ├── sdk/client.py         # LLMGatewayClient, TaskResult, TaskTiming
    ├── telemetry/            # logger (loguru, journaux JSONL), alarms (fire_alarme)
    ├── testing/              # echo_bundle, build_registry, FakeAdapter, ports mémoire
    └── config/               # settings.py (pydantic-settings), providers.yaml
```

## Documentation

mkdocs site (Diátaxis: tutorial, guides, reference, explanations, ADR):

```bash
python -m pip install -e "./packages/llm_gateway[docs]"
cd packages/llm_gateway && mkdocs serve      # http://127.0.0.1:8000 ; `mkdocs build --strict` en CI
```

Tests: `make test-gateway` (from the root) or `cd packages/llm_gateway && pytest -m "not e2e"`.
Change history: `CHANGELOG.md`.
