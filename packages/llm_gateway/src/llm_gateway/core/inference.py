"""core/inference.py — the inference parameter cascade, pure.

Three levels, from strongest to weakest: the **request** (`parameters` of the payload), the
**provider** (`inference:` block of its entry in the providers file), the **global
defaults** (`GatewaySettings.inference`). A value missing at one level falls through to the
next; `top_p` and `thinking_budget` may stay `None`, in which case the adapters do not send
them — for `thinking_budget`, this means "provider default", as opposed to `0`, which
disables thinking.

`max_tokens` is a **per-task** budget: the worker multiplies it by the number of agents in the
batch and then bounds it by the provider's ceilings.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class _InferenceLike(Protocol):
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    thinking_budget: int | None
    thinking_level: str | None


class ReglagesReflexionIncompatibles(ValueError):
    """`thinking_level` and `thinking_budget` requested together — the API returns 400."""


@dataclass(frozen=True)
class InferenceParams:
    temperature: float
    top_p: float | None
    max_tokens: int
    # None = not requested: the provider applies its own thinking default. Distinct from
    # 0, which disables it explicitly.
    thinking_budget: int | None = None
    # Thinking level (current API): `minimal` | `low` | `medium` | `high`.
    thinking_level: str | None = None


def _pick(name: str, request: Mapping[str, Any], provider: _InferenceLike | None, defaults: _InferenceLike) -> Any:
    value = request.get(name)
    if value is None and provider is not None:
        value = getattr(provider, name, None)
    if value is None:
        # `None` by default if the defaults object does not declare the key: a key added to
        # the cascade must not break resolution on an older configuration that does not
        # know it. True for `thinking_budget`, and for any future key.
        value = getattr(defaults, name, None)
    return value


def resolve_inference(
    request_parameters: Mapping[str, Any] | None,
    provider_overrides: _InferenceLike | None,
    defaults: _InferenceLike,
) -> InferenceParams:
    """Resolves temperature, top_p and output budget: request > provider > defaults.

    Request values are coerced (a JSON client may send `"0.7"`); an unreadable value is
    ignored and the next level applies, never an exception for an inference
    parameter.
    """
    req = dict(request_parameters or {})
    for key, caster in (("temperature", float), ("top_p", float), ("max_tokens", int),
                        ("thinking_budget", int), ("thinking_level", str)):
        if key in req and req[key] is not None:
            try:
                req[key] = caster(req[key])
            except (TypeError, ValueError):
                req[key] = None
    temperature = _pick("temperature", req, provider_overrides, defaults)
    top_p = _pick("top_p", req, provider_overrides, defaults)
    max_tokens = _pick("max_tokens", req, provider_overrides, defaults)
    budget = _pick("thinking_budget", req, provider_overrides, defaults)
    niveau = _pick("thinking_level", req, provider_overrides, defaults)
    if niveau is not None and budget is not None:
        # The provider returns 400 if both are present. Refusing here names the conflict and
        # the level where it happens, instead of an opaque 400 in the middle of a 3-hour run.
        raise ReglagesReflexionIncompatibles(
            f"thinking_level={niveau!r} and thinking_budget={budget!r} requested together: "
            "the API refuses them jointly (400). Choose one — `thinking_level` is the "
            "current setting, `thinking_budget` is only there for backward compatibility."
        )
    return InferenceParams(
        temperature=float(temperature),
        top_p=None if top_p is None else float(top_p),
        max_tokens=int(max_tokens),
        thinking_budget=None if budget is None else int(budget),
        thinking_level=None if niveau is None else str(niveau),
    )


__all__ = ["InferenceParams", "ReglagesReflexionIncompatibles", "resolve_inference"]
