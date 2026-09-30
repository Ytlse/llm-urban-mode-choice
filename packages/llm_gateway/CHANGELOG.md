# Changelog — llm-gateway

Format: `## [version] - AAAA-MM-JJ`, most recent entries first. The tone is that of use:
what the change enables or modifies for whoever uses it. The files touched are in git.

## [Unreleased] - 2026-09-25

### A question already asked word for word is served again without a call

`LLMRequest.espace_rejeu` (optional, set by `LLMGatewayClient(espace_rejeu=…)` on every call)
names a replay space. The worker records in it the response of each served task under the
fingerprint of its exact prompt: category, messages rendered for the task alone, parameters,
routing constraints. The API serves that response again to any task with the same fingerprint in
the same space, before the queue: no batch, no slot, no call. `provider_used` stays the original
provider, and the new `rejeu` field (task, status response, `TaskResult`) names the space. The
exchange log records the replayed response under `rejeu_ab:<fournisseur>`. Setting
`rejeu.dir` (`LLM_GATEWAY_REJEU__DIR`); without it, or without a space, nothing changes.

### A provider overload is reported to the client before it gives up

The "busy is not down" rule now only counts the instances **eligible** for the batch: the
pinned one, otherwise the `instances_admises`, otherwise all. An excluded instance, free or not,
no longer makes a batch wait that it will never serve. Every wait of the worker — full window,
cooldown that reopens in time, retry of a 5xx or of a per-minute 429 — is bounded by the
client's (`resilience.client_wait_seconds`, 120 s, or the `wait_timeout` of the pinned instance,
minus `client_wait_margin_seconds`, 10 s), measured from the batch's first attempt. Beyond that,
the task fails with `error_kind="surcharge_fournisseur"` and `resume_at`, or `quota_journalier`
if all eligible instances are at their daily quota. New alarm `alarme:surcharge_fournisseur`.
A client that waits more than 120 s without pinning an instance must raise `client_wait_seconds`.

## [Unreleased] - 2026-09-24

### A request says who issued it

`LLMRequest.origine` (optional) names the issuing client; `LLMGatewayClient(origine=…)` sets it
on all its calls, like `instances_admises`. The worker writes it into each record of
`llm_exchanges.jsonl` (field `origine`), which lets a client read only its own exchanges there
when several share the same worker. The origin is part of the batch key: two clients are never
merged into the same prompt. A client that does not set it keeps exactly the previous
behaviour.

## [1.3.0] - 2026-09-07 (iteration 2, lots B to E)

### Lot B — inference parameters are resolved in cascade

Temperature, `top_p` and per-task output budget come, in order, from the request
(`parameters`), from the provider (`inference:` block of its entry), then from the global
defaults (`inference` of the settings). The worker no longer has a `0.7` or `4096` literal;
`top_p` is sent to providers when it is defined, never otherwise. An unreadable value in the
request falls through to the next level instead of failing the batch.

### Lot E — each category keeps its parts in its own folder

The prompt engine accepts a template name and a schema file **per category**
(`CategorySpec.template_name`, `schema_path`); `schemas_file` becomes optional. `mobility_llm`
adopts the layout `categories/<nom>/template.md.j2` + `output_schema.json` + `observe.py`;
`prompts.yaml` does not move (prompt_calibration and the experiments cite it). The flat layout
remains possible and both coexist in the same bundle.

### Lot D — telemetry: nothing is written unless asked, and busy is not down

- **Exchange log disabled by default** (`telemetry.exchanges_enabled`): full prompts and
  responses are potential personal data. When enabled, it goes through a configurable
  **redactor** (`telemetry.redactor`, dotted path): masking or hashing of fields, truncation,
  or your own; and it rotates by size (`exchanges_max_bytes`). The format read by
  `make report` does not change. The SDK dialogue log is opt-in too.
- **Non-intrusive logging**: `configure_logging` now only removes loguru's default handler,
  the host's sinks survive; `reset_logging` undoes what it set up.
- **Prometheus families declared by the bundles** (`CategoryBundle.metric_families`): the
  collector no longer knows transport modes, it renders what the bundle declares. The mobility
  names are kept, the Grafana dashboards do not move.
- **The worker no longer abandons a queue when the providers are merely busy.** A full RPM/TPM
  window, smoothing or concurrency are not failures: the batch waits for the window, bounded by
  `max_retries`, with a `providers_occupes` alarm on rising edge. Abandoning remains for
  cooldown, deactivation and daily quota. The wait budget becomes configurable
  (`provider_wait_seconds`, `saturation_poll_seconds`, `saturation_retries`,
  `saturation_retry_seconds`, `abandon_when_busy`).

  **Before:** Prompt_Minimaliste run of 2026-09-07: one instance forced to 15 RPM, half of the
  requests abandoned after 48 s, decisions lost without retry.
  **After:** the batch waits for its slot; the loss no longer comes from the gateway.

### Lot C — a single translator for any OpenAI-compatible API

**Before:** four copied hundred-line adapters (OpenAI, Groq, Cerebras, Mistral), a `ping()`
defined everywhere and called nowhere, a closed list of modules to edit to add a provider, and
network failures that failed the task without retry.
**After:** `OpenAICompatibleAdapter` carries the `/chat/completions` dialect; the four become
settings (`structured_output`, `schema_in_system`) that the instance in the providers file can
override. A compatible provider (Ollama, vLLM, OpenRouter…) is added by configuration alone:
`adapter: openai_compatible`. A third-party package brings its adapter through the
`llm_gateway.adapters` entry point. Timeout, connection refused and truncated response are
`ProviderServerError`s (`network_timeout`, `network_connect`, `network_protocol`): retried with
backoff and cooldown. `ping()` is gone.

## [1.2.0] - 2026-09-07 (iteration 2, lot A)

Settings become a library configuration: grouped, prefixed, layered.

- **`LLM_GATEWAY_` prefix** and groups `redis`, `executor`, `inference`, `batching`, `resilience`,
  `api`, `telemetry` (`LLM_GATEWAY_BATCHING__DELAY_SECONDS`). Six sources in a fixed order:
  constructor, environment, old names (deprecated, warned), `LLM_GATEWAY_CONFIG` file,
  `LLM_GATEWAY_PROFILE` profile (`free-tier`, `paid`), defaults. `PROVIDER_KEYS__<nom>` remains
  the canonical name of API keys.
- **The providers file leaves the package**: `LLM_GATEWAY_PROVIDERS_FILE` designates it, the
  package only ships a `providers.example.yaml`. An unknown key makes startup fail, naming the
  provider and the key; `llm-gateway config schema` publishes the JSON Schema.
- **Learned completion caps** (`max_output_tokens` revealed by an HTTP 400) are no longer
  rewritten into a YAML: they live in Redis (shared between API and workers), a JSON file or
  memory, and are merged at startup without ever widening a declared cap.
- **`GET /config`** and **`GET /config/providers`**: the effective configuration, secrets masked.
- CORS is no longer hard-coded `*`: `api.cors_origins`, empty by default (the compose keeps `["*"]`).
- The threshold for deactivating a provider (`30` consecutive errors, hard-coded in the worker)
  becomes `resilience.disable_after_consecutive_errors`; `circuit_breaker_threshold`, never read,
  is removed.

**Before:** `REDIS_URL`, `APP_WORKDIR` shared with the controller; `providers.yaml` in the package
and rewritten by the worker; flat `Settings`.
**After:** `LLM_GATEWAY_REDIS__URL`, `LLM_GATEWAY_TELEMETRY__WORKDIR`; deployment file
designated, never modified; `settings.batching.delay_seconds`, flat aliases kept for one version.

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
