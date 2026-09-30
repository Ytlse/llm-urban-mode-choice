# Testing

The gateway suite is arranged in four tiers, each in its own folder of `tests/`. The
folder **sets the marker** (`tests/conftest.py`); `--strict-markers` refuses any marker
not declared in `pyproject.toml`. 233 tests collected on 2026-09-07; required coverage
80 % (ratchet: 74 % on the morning of 2026-09-07, 81 % after lot C of iteration 2) (`fail_under`), `testing/`, `cli.py` and `main.py` excluded from the computation.

| Tier | Folder | What it requires | What it covers |
|---|---|---|---|
| `unit` | `tests/unit/` | nothing | models, batch key and priority, SWRR sequence (including hypothesis properties), configuration (`batch_max_agents`, dispatch threshold), tolerant parser and its corpus, worker helpers (413 safeguard, 429 delays, learned 400 limit), SDK on `httpx.MockTransport` (typed result, dialogue log, backpressure, circuit breaker) |
| `contract` | `tests/contract/` | nothing; real Redis if `LLM_GATEWAY_TEST_REDIS_URL` | the **same suite** on each implementation of the ports `TaskStore`, `BatchQueue`, `RateLimiter`, `MetricsSink`: memory, fakeredis (Lua scripts via lupa), real Redis |
| `integration` | `tests/integration/` | nothing | the API composed on memory ports (202, 422 unknown category, 422 invalid item, 404, `/health`), the full path of a batch in the worker with `FakeAdapter` (demultiplexing, `agent_id` realignment, failing `observe` hook), the Google adapter on a simulated transport (truncations, thinking tokens) |
| `e2e` | `tests/e2e/` | a gateway and real providers (`LLM_GATEWAY_E2E_URL`) | the folder exists and is marked; it contains **no test** today — the end-to-end test lives in `mobility_llm/tests/e2e/test_perception.py` |

## Run

```bash
cd llm_gateway
pytest                                   # tout ; e2e sautés faute de LLM_GATEWAY_E2E_URL
pytest -m "not e2e" --cov                # what the CI does, with coverage
pytest -m unit                           # un étage
pytest tests/contract -k rate_limiter    # un contrat
LLM_GATEWAY_TEST_REDIS_URL=redis://localhost:6379/15 pytest tests/contract   # + backend Redis réel
LLM_GATEWAY_E2E_URL=http://localhost:8000 pytest -m e2e                      # exige api + worker + clés
```

From the repository root, with the project interpreter (`PKG_PYTHON`, default
`services/llm-agents/.venv/bin/python`):

| Target | Effect |
|---|---|
| `make test-gateway` | `pytest` in `llm_gateway/` |
| `make test-mobility` | `pytest` in `mobility_core/` then `mobility_llm/` |
| `make test-all` | the three packages then `lint-imports` |
| `make lint` | `ruff check` on the three packages |
| `make typecheck` | strict `mypy` on `core`, `ports`, `sdk` |
| `make lint-imports` | the 5 contracts of `.importlinter` |

`asyncio_mode = "auto"`: an `async def` test function runs without a decorator.
`filterwarnings = error::DeprecationWarning:llm_gateway.*`: the library never consumes
its own deprecated shims — a `DeprecationWarning` emitted by `llm_gateway` makes
the test fail.

## The port contracts

`tests/contract/conftest.py` parametrises the `ports` fixture over `BACKENDS = ["memory",
"fakeredis"]`, plus `"redis"` if the variable is defined (dedicated database, `flushdb` before each
test). A contract test only calls the methods of the `Protocol`s of `llm_gateway.ports`.
This is what guarantees that a consumer can replace Redis with memory (tests, embedded
mode) without any change of observable behaviour. The CI adds a Redis 7 service to
exercise the real implementation, Lua scripts included.

## The `llm_gateway.testing` tooling

Importable by any consumer (`pip install llm-gateway` is enough; pytest is not
required):

| Name | Role |
|---|---|
| `echo_bundle()` | a bundle with one `echo` category: the model repeats each item (`{"agent_id", "summary"}`) |
| `build_registry(*bundles)` | a `CategoryRegistry` built by hand, without an entry point; without argument: `echo` alone |
| `FakeAdapter(responder=None)` | adapter without network; answers for each `agent_id=…` found in the prompt; `responder(aid) -> dict` makes the response; `.calls` keeps the `InternalRequest`s received |
| `InMemoryTaskStore`, `InMemoryBatchQueue`, `InMemoryRateLimiter`, `InMemoryMetricsSink` | the in-memory ports, re-exported from `infra.memory` |
| `ECHO_TEMPLATES_DIR`, `ECHO_SCHEMAS_FILE`, `load_echo_schema()` | the files of the echo bundle |

Recipe for a full in-memory API (`tests/integration/conftest.py`):

```python
store = InMemoryTaskStore(); limiter = InMemoryRateLimiter(settings.providers)
deps = GatewayDeps(settings=settings, store=store, queue=InMemoryBatchQueue(store), limiter=limiter,
                   metrics=InMemoryMetricsSink(), balancer=LoadBalancer(settings.providers, limiter),
                   registry=build_registry())
app = create_app(settings, deps=deps)          # puis httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
```

The Celery dispatch is replaced by a stub (`monkeypatch.setattr(tw, "process_batch_task",
stub)`): no broker is contacted. A batch is replayed directly with
`tw._execute_batch(runtime, tasks, "batch_test", "fake")` after `monkeypatch.setattr(tw,
"get_adapter", lambda name: fake)`.

## Add a case to the parser corpus

`tests/data/llm_outputs.json` lists real or reconstructed LLM outputs; each entry
carries `expect` = number of expected agents, or `"error"` if `_parse_output` must refuse
(`ProviderParseError`). A new breaking or repairable output format is documented there, not
in an ad hoc test.

## Quality and CI

`.github/workflows/ci.yml`, triggered on `llm_gateway/**`, `mobility_core/**`,
`mobility_llm/**`, `.importlinter`:

| Job | Content |
|---|---|
| `quality` | `ruff check`, `ruff format --check` (informative, H10), strict `mypy` on `core`/`ports`/`sdk`, `lint-imports` |
| `test-gateway` | `pytest -m "not e2e" --cov` with a Redis service (`LLM_GATEWAY_TEST_REDIS_URL=redis://localhost:6379/15`), `coverage.xml` report as artefact |
| `test-mobility` | `mobility_core` then `mobility_llm` (`-m "not e2e"`) |
| `build` | `python -m build` of the three packages, wheels and sdist as artefacts |
| `docs` | `mkdocs build --strict` in `llm_gateway/` |

Local safety net: `.pre-commit-config.yaml` (ruff `--fix`, gitleaks, check-yaml,
end-of-file-fixer, trailing-whitespace) restricted to the three packages — `pip install
pre-commit && pre-commit install`.

The sibling packages' suites: `mobility_core` (199 tests, markers `unit` and
`needs_data`; the parity tests skip themselves without the `zf_zones.gpkg` layer) and
`mobility_llm` (134 tests, `unit` and `e2e`).
