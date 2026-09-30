"""Generate docs/reference/reglages.md from the pydantic settings models.

Usage: ../llm-agents/.venv/bin/python tools/gen_settings_doc.py
The table is the source of truth of the reference: do not edit it by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from llm_gateway.config.providers import InferenceOverrides, ProviderEntry
from llm_gateway.config.settings import (
    ApiSettings,
    BatchingSettings,
    ExecutorSettings,
    GatewaySettings,
    InferenceSettings,
    ProviderConfig,
    RedisSettings,
    ResilienceSettings,
    TelemetrySettings,
)
from llm_gateway.config.sources import ENV_PREFIX, LEGACY_ENV, new_env_name

OUT = Path(__file__).resolve().parents[1] / "docs" / "reference" / "reglages.md"

GROUPS: list[tuple[str, type[BaseModel], str]] = [
    ("redis", RedisSettings, "Redis connection: task store, batch queues, rate limiter, counters, learned limits."),
    ("executor", ExecutorSettings, "Batch execution. `celery` today; an in-process executor is planned."),
    ("inference", InferenceSettings, "Inference defaults; overridden per provider (`inference:` in the file), then per request."),
    ("batching", BatchingSettings, "Micro-batching: values measured on the July 2026 runs, to be recalibrated elsewhere."),
    ("resilience", ResilienceSettings, "Retries, provider switchover, disabling after consecutive errors."),
    ("api", ApiSettings, "HTTP layer: CORS, request size, tokens (ticket 036, not enforced)."),
    ("telemetry", TelemetrySettings, "Logs, dossier du run, journal des échanges."),
]

LEGACY_BY_PATH = {path: old for old, path in LEGACY_ENV.items()}


def _type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation)).replace("typing.", "")
    args = ", ".join(_type_name(a) for a in get_args(annotation))
    if str(origin).endswith("UnionType") or origin is type(int | None):
        return " | ".join(_type_name(a) for a in get_args(annotation))
    return f"{getattr(origin, '__name__', str(origin))}[{args}]"


def _default(field: Any) -> str:
    if field.default_factory is not None:
        value = field.default_factory()
        return f"`{value!r}`" if value not in ({}, []) else "`empty`"
    if field.default is PydanticUndefined:
        return "**required**"
    v = field.default
    return f"`{v.get_secret_value()!r}`" if hasattr(v, "get_secret_value") else f"`{v!r}`"


def _rows(model: type[BaseModel], group: str | None) -> list[str]:
    rows = []
    for name, field in model.model_fields.items():
        env = new_env_name((group, name)) if group else f"{ENV_PREFIX}{name.upper()}"
        legacy = LEGACY_BY_PATH.get((group, name)) if group else None
        env_cell = f"`{env}`" + (f" (ex-`{legacy}`)" if legacy else "")
        desc = (field.description or "").replace("|", "\\|")
        rows.append(f"| `{name}` | `{_type_name(field.annotation)}` | {_default(field)} | {env_cell} | {desc} |")
    return rows


def main() -> int:
    lines = [
        "# Settings",
        "",
        "> Page **generated** by `tools/gen_settings_doc.py` from the pydantic models; rerun it after",
        "> any change to `config/settings.py` or `config/providers.py`.",
        "",
        "Settings are **grouped** and come from six sources, from strongest to weakest:",
        "constructor arguments, `LLM_GATEWAY_*` environment (`__` separates the levels:",
        "`LLM_GATEWAY_BATCHING__DELAY_SECONDS`), old unprefixed names (deprecated, one version, one",
        "warning per variable found), YAML file designated by `LLM_GATEWAY_CONFIG`, profile designated by",
        "`LLM_GATEWAY_PROFILE` (`free-tier`, `paid`), code defaults. API keys are read under",
        "`LLM_GATEWAY_PROVIDER_KEYS__<name>` **or** `PROVIDER_KEYS__<name>` (canonical name shared with compose,",
        "`make providers` and prompt_calibration, without warning).",
        "",
        "## Top-level fields",
        "",
        "| Field | Type | Default | Variable | Meaning |",
        "|---|---|---|---|---|",
    ]
    top = {
        "providers_file": "providers file (deployment configuration). `None` = shipped example loaded with a warning.",
        "learned_limits": "where learned limits live (`max_output_tokens` revealed by HTTP 400): `redis`, `file`, `none`.",
        "learned_limits_file": "JSON file path when `learned_limits=file`; default `<telemetry.workdir>/learned_limits.json`.",
        "provider_keys": "API keys per instance or per adapter; resolved as `provider_keys[instance] or provider_keys[adapter]`.",
        "providers": "**built** after validation: resolved instances (`ProviderConfig`), not read from the environment.",
        "declared_providers": "**built**: names declared in the file, with or without a key.",
    }
    for name, desc in top.items():
        field = GatewaySettings.model_fields[name]
        env = f"`{ENV_PREFIX}{name.upper()}`" if name in ("providers_file", "learned_limits", "learned_limits_file") else (
            f"`{ENV_PREFIX}PROVIDER_KEYS__<name>` or `PROVIDER_KEYS__<name>`" if name == "provider_keys" else "—")
        lines.append(f"| `{name}` | `{_type_name(field.annotation)}` | {_default(field)} | {env} | {desc} |")
    for group, model, intro in GROUPS:
        lines += ["", f"## `{group}`", "", intro, "", "| Field | Type | Default | Variable | Meaning |", "|---|---|---|---|---|"]
        lines += _rows(model, group)
    lines += [
        "", "## Providers file", "",
        "A `providers:` object whose entries each follow `ProviderEntry`. Any unknown key makes loading",
        "fail, naming the provider and the key. JSON schema: `llm-gateway config schema providers`.",
        "", "| Field | Type | Default | Meaning |", "|---|---|---|---|",
    ]
    entry_desc = {
        "rpm_limit": "requests/minute, 60 s sliding window, `60 / rpm` smoothing between two requests",
        "base_url": "root of the provider API", "default_model": "model sent if the request does not impose one",
        "tpm_limit": "tokens/minute reserved in the same window; bounds `batch_max_agents`",
        "rpd_limit": "requests/day (UTC); reached → set aside until midnight UTC",
        "tpd_limit": "tokens/day (UTC), counted afterwards; same setting aside",
        "max_tokens_per_request": "capacity of a single request (HTTP 413 beyond); bounds the batch and the output budget",
        "max_output_tokens": "completion ceiling; learned on HTTP 400 and kept by the learned-limits store",
        "weight": "SWRR weight: `min(rpm_limit, tpm_limit / 3000) / 15`", "concurrency_limit": "concurrent workers allowed on the instance",
        "disable_timeout": "duration (s) set aside after `disable_after_consecutive_errors` errors",
        "adapter": "adapter name (openai_compatible, openai, mistral, google, groq, cerebras, or a `llm_gateway.adapters` entry point); default = entry name",
        "structured_output": "structured output of the OpenAI-compatible translator: `json_schema`, `json_object`, `none`; None = adapter default",
        "schema_in_system": "copy the JSON schema into the system message; None = adapter default",
        "inference": "`temperature`, `top_p`, `max_tokens` overrides for this instance",
        "batch_max_agents": "tolerated for old files, **ignored** with a warning: the gateway computes it",
    }
    for name, field in ProviderEntry.model_fields.items():
        lines.append(f"| `{name}` | `{_type_name(field.annotation)}` | {_default(field)} | {entry_desc.get(name, '')} |")
    lines += ["", "### `inference` (per-provider overrides)", "", "| Field | Type | Default |", "|---|---|---|"]
    for name, field in InferenceOverrides.model_fields.items():
        lines.append(f"| `{name}` | `{_type_name(field.annotation)}` | {_default(field)} |")
    lines += [
        "", "## `ProviderConfig` (resolved instance)", "",
        "What the gateway handles: the file entry, the injected key and two computed values.",
        "`batch_max_agents = max(1, min(tpm_limit / (assumed_prompt_tokens + assumed_output_tokens),",
        "max_tokens_per_request / same, rpm_limit, batching.max_batch_agents))`;",
        "`tpm_estimate_per_request = batch_max_agents × (assumed_prompt_tokens + assumed_output_tokens)` if `tpm_limit`.",
        "", "| Field | Type | Default |", "|---|---|---|",
    ]
    for name, field in ProviderConfig.model_fields.items():
        lines.append(f"| `{name}` | `{_type_name(field.annotation)}` | {_default(field)} |")
    lines += [
        "", "## Old names still read (deprecated)", "",
        "| Old name | New name |", "|---|---|",
    ]
    for old, path in LEGACY_ENV.items():
        lines.append(f"| `{old}` | `{new_env_name(path)}` |")
    lines += ["", "`PROVIDER_KEYS__<name>` is not deprecated: it is the canonical name of API keys.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT.relative_to(Path.cwd())}: {len(lines)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
