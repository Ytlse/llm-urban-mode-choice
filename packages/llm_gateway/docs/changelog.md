# Changelog

!!! note "Copy"
    This page is a copy; the source is `CHANGELOG.md` at the root of the `llm_gateway`
    package. In case of discrepancy, the root file is authoritative.

Format: `## [version] - AAAA-MM-JJ`, most recent entries first. The tone is that of use:
what the change enables or modifies for whoever uses it. The files touched are in git.

## [1.1.0] - 2026-09-07

The gateway becomes a generic library: it no longer knows about Toulouse mobility.
The prompts, the persona model, the mode choice and the domain metrics live in
`mobility_llm`, the survey domain in `mobility_core` (ticket 037, iteration 1).

**Before:** a single `llm_module` package mixing gateway and domain. Templates, schemas and
prompt variants were shipped in the package; `AgentSpec` (perception, trajectories,
feelings…) was imposed on every category; the worker imported the mode choice and counted
mode disagreements; the API refused an item without `perception` even for a category
that does not use it.
**After:** the gateway only knows `AgentItem` (an `agent_id`, the rest free). Each
category is declared by a *bundle* (`CategoryBundle` / `CategorySpec`) discovered through
the `llm_gateway.categories` entry point: it is the bundle that validates the items, sets the
priority of a batch and observes the response for its metrics. With no bundle installed, every
request is refused with 422 and the list of known categories (empty).

### What changes for a consumer

- **Import:** `from llm_gateway import LLMGatewayClient, LLMRequest` replaces
  `llm_module.sdk`. The facade is lazy: importing `llm_gateway` loads neither httpx, nor
  prometheus_client, nor FastAPI; each name is resolved on first access.
- **Deprecated shell:** `llm_module` remains importable, each module re-exports its
  successor and emits a `DeprecationWarning`; removal planned for 2.0.
- **Readable version:** `llm_gateway.__version__` (and `pip show llm-gateway`).
- **CLI:** `llm-gateway serve | worker | config validate | config show | categories`.
  `config show` masks secrets; `categories` returns 1 when no bundle is registered.
- **Category validated at startup:** missing template or schema → `ValueError` when the
  registry is built, no longer at the first request.

### What changes in operation

- **JSON repair:** `json-repair` replaces `demjson3` for malformed LLM outputs
  (trailing commas, single quotes, missing braces, stray text).
- **Alarm counter:** the SDK's Prometheus family `alarme_total{source}` is created at the
  first `fire_alarme`, never at import; in the API process, where the Redis collector
  already exposes this family, the counter is kept out of the registry. API startup no longer
  crashes on `DuplicateTimeseries`.
- **Docker image:** build context at the repository root, the three packages installed,
  unprivileged user `gateway`.
- **`.env.example`** shipped with every variable read (keys `PROVIDER_KEYS__<nom>`,
  Redis, logs).

### Tests and quality

- Four tiers (`unit`, `contract`, `integration`, `e2e`), marker set by the folder,
  `--strict-markers`. The port contracts run on memory, fakeredis (Lua scripts) and
  a real Redis when `LLM_GATEWAY_TEST_REDIS_URL` is defined.
- Sub-package `llm_gateway.testing`: `echo` bundle, `build_registry`, `FakeAdapter`, memory
  ports — a consumer tests its bundle without Redis or network.
- Hypothesis properties (SWRR sequence, batch key), corpus of LLM outputs against the parser.
- Required coverage: 80 % (`fail_under`). GitHub Actions CI: lint, strict mypy on
  `core`/`ports`/`sdk`, 5 import-linter contracts, tests with a service Redis, wheels,
  `mkdocs build --strict`.

### Postponed

`LLM_GATEWAY_` prefix for environment variables and moving `providers.yaml` out of the
package; generic OpenAI-compatible adapter; executor without Celery; authentication
(ticket 036); licence.

## [1.0.0] - 2026-07-07

Restructuring of `llm_module` into a ports & adapters package: explicit composition
(`create_app`, `build_deps`), no more side effects at import (the reset of the RPM windows
becomes a step of the API lifespan), worker counters in a single Redis hash,
httpx client shared per adapter, typed SDK (`TaskResult`). Report:
`docs/arch/llm-module-package-refactor.md` in the repository.
