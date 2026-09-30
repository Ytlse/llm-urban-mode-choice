"""config/learned.py — merging learned limits into the settings, and learning them.

A model's completion cap is often only known at the first HTTP 400. The worker
learns it, stores it in the `LearnedLimits` port, and each process (API, workers, at
startup) merges it over the configuration: a learned limit can only tighten the one
from the file, never widen it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from llm_gateway.ports.learned_limits import LearnedLimits
from llm_gateway.telemetry.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from llm_gateway.config.settings import GatewaySettings

logger = get_logger(__name__)


def apply_learned_limits(settings: GatewaySettings, store: LearnedLimits) -> int:
    """Merge learned limits into `settings.providers`. Return the number of providers adjusted."""
    try:
        learned = store.all_max_output_tokens()
    except Exception as exc:  # store unreachable: start with the configuration alone
        logger.warning(f"Learned limits unreadable at startup (ignored) | error={exc!r}")
        return 0
    adjusted = 0
    for name, limit in learned.items():
        cfg = settings.providers.get(name)
        if cfg is None or limit <= 0:
            continue
        if cfg.max_output_tokens is None or limit < cfg.max_output_tokens:
            logger.info(
                f"Learned completion limit applied | provider={name} "
                f"max_output_tokens={cfg.max_output_tokens} -> {limit}"
            )
            cfg.max_output_tokens = limit
            adjusted += 1
    return adjusted


def learn_provider_max_output_tokens(
    settings: GatewaySettings, store: LearnedLimits, provider_name: str, limit: int
) -> bool:
    """Record the completion limit revealed by a provider HTTP 400.

    Updates the configuration of the current process and the shared store. Returns False if
    the limit was already known (equal or stricter) or if the provider is unknown: the caller
    must then NOT retry, or it would loop on the same 400.
    """
    if limit <= 0:
        return False
    cfg = settings.providers.get(provider_name)
    if cfg is None:
        logger.warning(
            f"max_output_tokens limit learned for an unknown provider | "
            f"provider={provider_name} limit={limit}"
        )
        return False
    if cfg.max_output_tokens is not None and cfg.max_output_tokens <= limit:
        return False
    logger.warning(
        f"Completion limit learned from the provider error — config adjusted | "
        f"provider={provider_name} max_output_tokens={cfg.max_output_tokens} -> {limit}"
    )
    cfg.max_output_tokens = limit
    try:
        store.set_max_output_tokens(provider_name, limit)
    except Exception as exc:
        logger.error(
            f"[ALARME] Could not store the learned limit (the other processes will "
            f"relearn it) | provider={provider_name} limit={limit} error={exc!r}"
        )
    return True


__all__ = ["apply_learned_limits", "learn_provider_max_output_tokens"]
