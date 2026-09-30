# Settings

> Page **generated** by `tools/gen_settings_doc.py` from the pydantic models; rerun it after
> any change to `config/settings.py` or `config/providers.py`.

Settings are **grouped** and come from six sources, from strongest to weakest:
constructor arguments, `LLM_GATEWAY_*` environment (`__` separates the levels:
`LLM_GATEWAY_BATCHING__DELAY_SECONDS`), old unprefixed names (deprecated, one version, one
warning per variable found), YAML file designated by `LLM_GATEWAY_CONFIG`, profile designated by
`LLM_GATEWAY_PROFILE` (`free-tier`, `paid`), code defaults. API keys are read under
`LLM_GATEWAY_PROVIDER_KEYS__<name>` **or** `PROVIDER_KEYS__<name>` (canonical name shared with compose,
`make providers` and prompt_calibration, without warning).

## Top-level fields

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `providers_file` | `Path | NoneType` | `None` | `LLM_GATEWAY_PROVIDERS_FILE` | providers file (deployment configuration). `None` = shipped example loaded with a warning. |
| `learned_limits` | `Literal[redis, file, none]` | `'redis'` | `LLM_GATEWAY_LEARNED_LIMITS` | where learned limits live (`max_output_tokens` revealed by HTTP 400): `redis`, `file`, `none`. |
| `learned_limits_file` | `Path | NoneType` | `None` | `LLM_GATEWAY_LEARNED_LIMITS_FILE` | JSON file path when `learned_limits=file`; default `<telemetry.workdir>/learned_limits.json`. |
| `provider_keys` | `dict[str, SecretStr]` | `empty` | `LLM_GATEWAY_PROVIDER_KEYS__<name>` or `PROVIDER_KEYS__<name>` | API keys per instance or per adapter; resolved as `provider_keys[instance] or provider_keys[adapter]`. |
| `providers` | `dict[str, ProviderConfig]` | `empty` | — | **built** after validation: resolved instances (`ProviderConfig`), not read from the environment. |
| `declared_providers` | `list[str]` | `empty` | — | **built**: names declared in the file, with or without a key. |

## `redis`

Redis connection: task store, batch queues, rate limiter, counters, learned limits.

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `url` | `str` | `'redis://localhost:6379/0'` | `LLM_GATEWAY_REDIS__URL` (ex-`REDIS_URL`) |  |

## `executor`

Batch execution. `celery` today; an in-process executor is planned.

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `kind` | `Literal[celery]` | `'celery'` | `LLM_GATEWAY_EXECUTOR__KIND` |  |
| `celery_broker_url` | `str` | `'redis://localhost:6379/1'` | `LLM_GATEWAY_EXECUTOR__CELERY_BROKER_URL` (ex-`CELERY_BROKER_URL`) |  |
| `celery_result_backend` | `str` | `'redis://localhost:6379/2'` | `LLM_GATEWAY_EXECUTOR__CELERY_RESULT_BACKEND` (ex-`CELERY_RESULT_BACKEND`) |  |

## `inference`

Inference defaults; overridden per provider (`inference:` in the file), then per request.

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `temperature` | `float` | `0.7` | `LLM_GATEWAY_INFERENCE__TEMPERATURE` |  |
| `top_p` | `float | NoneType` | `None` | `LLM_GATEWAY_INFERENCE__TOP_P` |  |
| `max_tokens` | `int` | `4096` | `LLM_GATEWAY_INFERENCE__MAX_TOKENS` |  |
| `thinking_budget` | `int | NoneType` | `None` | `LLM_GATEWAY_INFERENCE__THINKING_BUDGET` |  |
| `thinking_level` | `str | NoneType` | `None` | `LLM_GATEWAY_INFERENCE__THINKING_LEVEL` |  |

## `batching`

Micro-batching: values measured on the July 2026 runs, to be recalibrated elsewhere.

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `max_agents` | `int` | `5` | `LLM_GATEWAY_BATCHING__MAX_AGENTS` (ex-`BATCH_MAX_AGENTS`) |  |
| `delay_seconds` | `float` | `3.0` | `LLM_GATEWAY_BATCHING__DELAY_SECONDS` (ex-`BATCH_DELAY_SECONDS`) |  |
| `target_agents` | `int` | `10` | `LLM_GATEWAY_BATCHING__TARGET_AGENTS` (ex-`BATCH_TARGET_AGENTS`) |  |
| `assumed_prompt_tokens` | `int` | `2200` | `LLM_GATEWAY_BATCHING__ASSUMED_PROMPT_TOKENS` (ex-`ASSUMED_PROMPT_TOKENS`) |  |
| `assumed_output_tokens` | `int` | `800` | `LLM_GATEWAY_BATCHING__ASSUMED_OUTPUT_TOKENS` (ex-`ASSUMED_OUTPUT_TOKENS`) |  |
| `token_chars_ratio` | `float` | `3.0` | `LLM_GATEWAY_BATCHING__TOKEN_CHARS_RATIO` (ex-`TOKEN_CHARS_RATIO`) |  |
| `max_batch_agents` | `int` | `20` | `LLM_GATEWAY_BATCHING__MAX_BATCH_AGENTS` (ex-`MAX_BATCH_AGENTS`) |  |
| `min_output_tokens` | `int` | `512` | `LLM_GATEWAY_BATCHING__MIN_OUTPUT_TOKENS` (ex-`MIN_OUTPUT_TOKENS`) |  |
| `max_output_tokens` | `int` | `16384` | `LLM_GATEWAY_BATCHING__MAX_OUTPUT_TOKENS` (ex-`MAX_OUTPUT_TOKENS`) |  |

## `resilience`

Retries, provider switchover, disabling after consecutive errors.

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `max_retries` | `int` | `50` | `LLM_GATEWAY_RESILIENCE__MAX_RETRIES` (ex-`MAX_RETRIES`) |  |
| `backoff_base_seconds` | `float` | `1.0` | `LLM_GATEWAY_RESILIENCE__BACKOFF_BASE_SECONDS` (ex-`BACKOFF_BASE_SECONDS`) |  |
| `provider_switch_cooldown_seconds` | `int` | `30` | `LLM_GATEWAY_RESILIENCE__PROVIDER_SWITCH_COOLDOWN_SECONDS` (ex-`PROVIDER_SWITCH_COOLDOWN_SECONDS`) |  |
| `disable_after_consecutive_errors` | `int` | `30` | `LLM_GATEWAY_RESILIENCE__DISABLE_AFTER_CONSECUTIVE_ERRORS` |  |
| `provider_wait_seconds` | `float` | `8.0` | `LLM_GATEWAY_RESILIENCE__PROVIDER_WAIT_SECONDS` |  |
| `saturation_poll_seconds` | `float` | `2.0` | `LLM_GATEWAY_RESILIENCE__SATURATION_POLL_SECONDS` |  |
| `saturation_retries` | `int` | `2` | `LLM_GATEWAY_RESILIENCE__SATURATION_RETRIES` |  |
| `saturation_retry_seconds` | `float` | `12.0` | `LLM_GATEWAY_RESILIENCE__SATURATION_RETRY_SECONDS` |  |
| `abandon_when_busy` | `bool` | `False` | `LLM_GATEWAY_RESILIENCE__ABANDON_WHEN_BUSY` |  |
| `client_wait_seconds` | `float` | `120.0` | `LLM_GATEWAY_RESILIENCE__CLIENT_WAIT_SECONDS` |  |
| `client_wait_margin_seconds` | `float` | `10.0` | `LLM_GATEWAY_RESILIENCE__CLIENT_WAIT_MARGIN_SECONDS` |  |

## `api`

HTTP layer: CORS, request size, tokens (ticket 036, not enforced).

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `cors_origins` | `list[str]` | `empty` | `LLM_GATEWAY_API__CORS_ORIGINS` |  |
| `max_request_bytes` | `int` | `2000000` | `LLM_GATEWAY_API__MAX_REQUEST_BYTES` |  |
| `auth_tokens` | `list[SecretStr]` | `empty` | `LLM_GATEWAY_API__AUTH_TOKENS` |  |

## `telemetry`

Logs, run folder, exchange log.

| Field | Type | Default | Variable | Meaning |
|---|---|---|---|---|
| `log_level` | `str` | `'INFO'` | `LLM_GATEWAY_TELEMETRY__LOG_LEVEL` (ex-`LOG_LEVEL`) |  |
| `log_format` | `Literal[text, json]` | `'text'` | `LLM_GATEWAY_TELEMETRY__LOG_FORMAT` |  |
| `service_name` | `str | NoneType` | `None` | `LLM_GATEWAY_TELEMETRY__SERVICE_NAME` (ex-`SERVICE_NAME`) |  |
| `workdir` | `Path` | `PosixPath('.')` | `LLM_GATEWAY_TELEMETRY__WORKDIR` (ex-`APP_WORKDIR`) |  |
| `exchanges_enabled` | `bool` | `False` | `LLM_GATEWAY_TELEMETRY__EXCHANGES_ENABLED` |  |
| `exchanges_file` | `Path | NoneType` | `None` | `LLM_GATEWAY_TELEMETRY__EXCHANGES_FILE` (ex-`LLM_EXCHANGES_FILE`) |  |
| `exchanges_max_bytes` | `int` | `200000000` | `LLM_GATEWAY_TELEMETRY__EXCHANGES_MAX_BYTES` |  |
| `redactor` | `str | NoneType` | `None` | `LLM_GATEWAY_TELEMETRY__REDACTOR` |  |

## Providers file

A `providers:` object whose entries each follow `ProviderEntry`. Any unknown key makes loading
fail, naming the provider and the key. JSON schema: `llm-gateway config schema providers`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `rpm_limit` | `int` | **required** | requests/minute, 60 s sliding window, `60 / rpm` smoothing between two requests |
| `base_url` | `str` | **required** | root of the provider API |
| `default_model` | `str` | **required** | model sent if the request does not impose one |
| `tpm_limit` | `int | NoneType` | `None` | tokens/minute reserved in the same window; bounds `batch_max_agents` |
| `rpd_limit` | `int | NoneType` | `None` | requests/day (UTC); reached → set aside until midnight UTC |
| `quota_reset_tz` | `str` | `'UTC'` |  |
| `tpd_limit` | `int | NoneType` | `None` | tokens/day (UTC), counted afterwards; same setting aside |
| `max_tokens_per_request` | `int | NoneType` | `None` | capacity of a single request (HTTP 413 beyond); bounds the batch and the output budget |
| `max_output_tokens` | `int | NoneType` | `None` | completion ceiling; learned on HTTP 400 and kept by the learned-limits store |
| `thinking_budget_max` | `int | NoneType` | `None` |  |
| `thinking_levels` | `list[str] | NoneType` | `None` |  |
| `weight` | `float` | `1.0` | SWRR weight: `min(rpm_limit, tpm_limit / 3000) / 15` |
| `concurrency_limit` | `int` | `2` | concurrent workers allowed on the instance |
| `wait_timeout` | `float | NoneType` | `None` |  |
| `disable_timeout` | `int` | `180` | duration (s) set aside after `disable_after_consecutive_errors` errors |
| `adapter` | `str` | `''` | adapter name (openai_compatible, openai, mistral, google, groq, cerebras, or a `llm_gateway.adapters` entry point); default = entry name |
| `structured_output` | `Union[Literal[json_schema, json_object, none], NoneType]` | `None` | structured output of the OpenAI-compatible translator: `json_schema`, `json_object`, `none`; None = adapter default |
| `schema_in_system` | `bool | NoneType` | `None` | copy the JSON schema into the system message; None = adapter default |
| `inference` | `InferenceOverrides | NoneType` | `None` | `temperature`, `top_p`, `max_tokens` overrides for this instance |
| `batch_max_agents` | `int | NoneType` | `None` | tolerated for old files, **ignored** with a warning: the gateway computes it |

### `inference` (per-provider overrides)

| Field | Type | Default |
|---|---|---|
| `temperature` | `float | NoneType` | `None` |
| `top_p` | `float | NoneType` | `None` |
| `max_tokens` | `int | NoneType` | `None` |
| `thinking_budget` | `int | NoneType` | `None` |
| `thinking_level` | `str | NoneType` | `None` |

## `ProviderConfig` (resolved instance)

What the gateway handles: the file entry, the injected key and two computed values.
`batch_max_agents = max(1, min(tpm_limit / (assumed_prompt_tokens + assumed_output_tokens),
max_tokens_per_request / same, rpm_limit, batching.max_batch_agents))`;
`tpm_estimate_per_request = batch_max_agents × (assumed_prompt_tokens + assumed_output_tokens)` if `tpm_limit`.

| Field | Type | Default |
|---|---|---|
| `api_key` | `SecretStr` | `''` |
| `rpm_limit` | `int` | **required** |
| `tpm_limit` | `int | NoneType` | `None` |
| `rpd_limit` | `int | NoneType` | `None` |
| `tpd_limit` | `int | NoneType` | `None` |
| `quota_reset_tz` | `str` | `'UTC'` |
| `max_tokens_per_request` | `int | NoneType` | `None` |
| `max_output_tokens` | `int | NoneType` | `None` |
| `thinking_budget_max` | `int | NoneType` | `None` |
| `thinking_levels` | `list[str] | NoneType` | `None` |
| `base_url` | `str` | **required** |
| `default_model` | `str` | **required** |
| `weight` | `float` | `1.0` |
| `batch_max_agents` | `int` | `1` |
| `tpm_estimate_per_request` | `int | NoneType` | `None` |
| `concurrency_limit` | `int` | `2` |
| `wait_timeout` | `float | NoneType` | `None` |
| `disable_timeout` | `int` | `180` |
| `adapter` | `str` | `''` |
| `structured_output` | `Union[Literal[json_schema, json_object, none], NoneType]` | `None` |
| `schema_in_system` | `bool | NoneType` | `None` |
| `inference` | `InferenceOverrides | NoneType` | `None` |

## Old names still read (deprecated)

| Old name | New name |
|---|---|
| `REDIS_URL` | `LLM_GATEWAY_REDIS__URL` |
| `CELERY_BROKER_URL` | `LLM_GATEWAY_EXECUTOR__CELERY_BROKER_URL` |
| `CELERY_RESULT_BACKEND` | `LLM_GATEWAY_EXECUTOR__CELERY_RESULT_BACKEND` |
| `LOG_LEVEL` | `LLM_GATEWAY_TELEMETRY__LOG_LEVEL` |
| `SERVICE_NAME` | `LLM_GATEWAY_TELEMETRY__SERVICE_NAME` |
| `APP_WORKDIR` | `LLM_GATEWAY_TELEMETRY__WORKDIR` |
| `LLM_EXCHANGES_FILE` | `LLM_GATEWAY_TELEMETRY__EXCHANGES_FILE` |
| `MAX_RETRIES` | `LLM_GATEWAY_RESILIENCE__MAX_RETRIES` |
| `BACKOFF_BASE_SECONDS` | `LLM_GATEWAY_RESILIENCE__BACKOFF_BASE_SECONDS` |
| `PROVIDER_SWITCH_COOLDOWN_SECONDS` | `LLM_GATEWAY_RESILIENCE__PROVIDER_SWITCH_COOLDOWN_SECONDS` |
| `BATCH_MAX_AGENTS` | `LLM_GATEWAY_BATCHING__MAX_AGENTS` |
| `BATCH_DELAY_SECONDS` | `LLM_GATEWAY_BATCHING__DELAY_SECONDS` |
| `BATCH_TARGET_AGENTS` | `LLM_GATEWAY_BATCHING__TARGET_AGENTS` |
| `ASSUMED_PROMPT_TOKENS` | `LLM_GATEWAY_BATCHING__ASSUMED_PROMPT_TOKENS` |
| `ASSUMED_OUTPUT_TOKENS` | `LLM_GATEWAY_BATCHING__ASSUMED_OUTPUT_TOKENS` |
| `TOKEN_CHARS_RATIO` | `LLM_GATEWAY_BATCHING__TOKEN_CHARS_RATIO` |
| `MAX_BATCH_AGENTS` | `LLM_GATEWAY_BATCHING__MAX_BATCH_AGENTS` |
| `MIN_OUTPUT_TOKENS` | `LLM_GATEWAY_BATCHING__MIN_OUTPUT_TOKENS` |
| `MAX_OUTPUT_TOKENS` | `LLM_GATEWAY_BATCHING__MAX_OUTPUT_TOKENS` |

`PROVIDER_KEYS__<name>` is not deprecated: it is the canonical name of API keys.
