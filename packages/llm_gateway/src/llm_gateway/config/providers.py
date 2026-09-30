"""config/providers.py — the providers file, validated before any startup.

The file (YAML) describes the provider instances: quotas, model, weight, capacity.
It lives **outside the package** (deployment configuration, `LLM_GATEWAY_PROVIDERS_FILE`);
the package only ships an example, `providers.example.yaml`, loaded with a warning when
no file is designated.

An unknown key makes loading fail, naming the provider and the key: a silently ignored
`tpm_limt` made a provider run with a different setting than the one written.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_gateway.core.quota import DEFAUT_FUSEAU_QUOTA

DEFAULT_EXAMPLE_FILE = Path(__file__).resolve().parent / "providers.example.yaml"


class ProvidersConfigError(ValueError):
    """Providers file missing, unreadable or invalid. Never a silent fallback."""


class InferenceOverrides(BaseModel):
    """Per-provider override of inference parameters (cascade: request > provider > default)."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    # Per-provider override: useful to turn thinking off on an instance that bills it
    # or handles it poorly, without touching the others.
    thinking_budget: int | None = None
    thinking_level: str | None = None


class ProviderEntry(BaseModel):
    """A provider instance as written in the file. No computed field here."""

    model_config = ConfigDict(extra="forbid")

    rpm_limit: int
    base_url: str
    default_model: str
    tpm_limit: int | None = None
    rpd_limit: int | None = None   # requests/day: enforced, provider set aside until the quota reset
    # Time zone of the provider's daily reset. Gemini free tier counts its day in
    # Pacific time: with the UTC default, the counters emptied 7 h too early and
    # a refused key looked available (incident of 2026-09-08).
    quota_reset_tz: str = DEFAUT_FUSEAU_QUOTA
    tpd_limit: int | None = None   # tokens/day: enforced, actual tokens counted afterwards
    max_tokens_per_request: int | None = None  # capacity of a single request (HTTP 413 beyond)
    max_output_tokens: int | None = None       # completion cap; learned on HTTP 400 (see learned)
    # THINKING cap of the served model, in thinking tokens. To be read from the provider's
    # documentation, never guessed: a budget above the cap is trimmed SILENTLY on the
    # provider side, and the response only reports the thinking tokens consumed — never
    # the budget applied. Nothing would therefore allow the gap to be recovered afterwards,
    # and the experiment fingerprint would carry a budget that was not applied.
    # Declared → "maximum" becomes selectable in the form and a higher budget is refused
    # upfront. Absent → no maximum is offered (a measurement is not replaced by a
    # plausible number).
    thinking_budget_max: int | None = None
    # Thinking levels accepted by the served model (`minimal`, `low`, `medium`, `high`).
    # Read from the provider's documentation: they vary from one model to another, and
    # requesting a missing level returns 400. Declared → the form only offers those and an
    # unknown level is refused upfront; absent → no level is offered.
    thinking_levels: list[str] | None = None
    weight: float = 1.0
    concurrency_limit: int = 2
    # Maximum CLIENT wait for a task served by this instance (seconds).
    # Caller setting, not routing: the gateway does not read it, the SDK receives it per call
    # (see `sdk.client.execute(wait_timeout=…)`). A local model serves one call at a time:
    # a task's wait is that of the QUEUE, not of generation. None = client default.
    wait_timeout: float | None = None
    disable_timeout: int = 180
    adapter: str = ""
    # Structured output of the OpenAI-compatible adapter: json_schema (native), json_object,
    # none; and injection of the schema into the system message. None = adapter defaults.
    structured_output: Literal["json_schema", "json_object", "none"] | None = None
    schema_in_system: bool | None = None
    inference: InferenceOverrides | None = None
    # Tolerated because old files carry it; recomputed by the settings, never read.
    batch_max_agents: int | None = None   # ignored and flagged on load: the gateway computes it


class ProvidersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderEntry] = Field(default_factory=dict)


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", ())]
        provider = loc[1] if len(loc) > 1 and loc[0] == "providers" else None
        key = ".".join(loc[2:]) if len(loc) > 2 else ".".join(loc)
        where = f"provider {provider!r}, key {key!r}" if provider else f"key {key!r}"
        lines.append(f"  - {where}: {err.get('msg')}")
    return f"Invalid providers file ({path}):\n" + "\n".join(lines)


def load_providers_file(path: Path) -> ProvidersFile:
    """Load and validate the file. Raises ProvidersConfigError, never anything else."""
    path = Path(path)
    if not path.is_file():
        raise ProvidersConfigError(f"Providers file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProvidersConfigError(f"Providers file unreadable ({path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ProvidersConfigError(f"Providers file ({path}): a YAML object is expected")
    try:
        return ProvidersFile.model_validate(data)
    except ValidationError as exc:
        raise ProvidersConfigError(_format_validation_error(path, exc)) from exc


def providers_schema() -> dict[str, Any]:
    """JSON Schema of the providers file (validation in the editor, in CI)."""
    return ProvidersFile.model_json_schema()


def providers_schema_json() -> str:
    return json.dumps(providers_schema(), indent=2, ensure_ascii=False)


__all__ = [
    "DEFAULT_EXAMPLE_FILE",
    "InferenceOverrides",
    "ProviderEntry",
    "ProvidersConfigError",
    "ProvidersFile",
    "load_providers_file",
    "providers_schema",
    "providers_schema_json",
]
