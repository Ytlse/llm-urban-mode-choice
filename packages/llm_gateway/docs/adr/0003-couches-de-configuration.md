# ADR 0003 — Settings come from six layers, grouped and prefixed

- **Date**: 2026-09-07
- **Status**: accepted (ticket 037, iteration 2, lot A)
- **Scope**: `llm_gateway.config` (settings, sources, providers, learned), the API (`/config`), the compose

## Context

Before lot A, `Settings` was a flat object of twenty fields read from an environment **without
prefix** (`REDIS_URL`, `APP_WORKDIR`, `LOG_LEVEL`…), shared with the controller and the other
services of the same compose: the gateway read variables that were not meant for it, and
a consumer that embedded it inherited its names. The providers file lived **inside
the installed package**, and the worker **rewrote** the learned completion caps into it: a
library that modifies its own files in `site-packages`.

## Decision

1. **A `GatewaySettings` object made of seven groups** (`redis`, `executor`, `inference`,
   `batching`, `resilience`, `api`, `telemetry`) plus the providers file, the keys and
   the learned limits store. Each group is a documented pydantic model; the flat
   aliases (`settings.redis_url`…) remain for one version for consumers.
2. **Six sources, fixed order**: constructor arguments, `LLM_GATEWAY_*` environment
   (`__` between levels), old unprefixed names (legacy source that warns and only speaks
   if the new name is absent), `LLM_GATEWAY_CONFIG` YAML file,
   `LLM_GATEWAY_PROFILE` profile shipped with the package, code defaults.
3. **`PROVIDER_KEYS__<nom>` remains canonical** for API keys: this name is shared with the
   author's `.env`, the compose (logic of the second Google buckets), `make providers` and
   prompt_calibration. Renaming it would have broken four tools for no gain; the prefixed
   form is accepted too.
4. **The providers file leaves the package**: `LLM_GATEWAY_PROVIDERS_FILE` designates it, the
   package only ships an example. Unknown key = failure at load time, provider and key named.
5. **Learned limits go behind a port** (`LearnedLimits`): Redis in production
   (shared API/workers), JSON file without Redis, memory otherwise. Merged at startup on
   top of the configuration, they can only tighten a cap, never widen it.
6. **`GET /config` and `GET /config/providers`** publish the effective configuration, secrets
   masked: the tools that read the YAML can query the gateway.

## Consequences

- The compose switches to prefixed names for `api`, `worker`, `flower`; the author's `.env` does not
  change. The controller reads the providers file at the new path (mounted read-only):
  the startup order does not depend on a call to the API.
- The repository's `providers.yaml` lives in `config/llm_gateway/`; `make providers` writes it there.
- The settings reference is **generated** (`tools/gen_settings_doc.py`): the doc can no longer
  drift from the code.
- Deviation from the iteration plan: the planned `providers` sub-model became three
  top-level fields (`providers_file`, `learned_limits`, `learned_limits_file`) to keep
  `settings.providers` as the dictionary of resolved instances, cited in twenty-two places.

## Alternatives rejected

- **Also renaming `PROVIDER_KEYS__`**: consistent, but four tools and the `.env` to change
  for a secret that bothers no one under this name.
- **Keeping the providers file in the package with an overridable path**: that is what
  allowed the worker to write into `site-packages`; moving it removes the temptation.
- **Querying `/config/providers` from the controller**: fragile at startup (the API is
  not ready yet when the controller reads its settings); the mounted file is enough.
