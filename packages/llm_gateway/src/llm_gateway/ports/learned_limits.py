"""ports/learned_limits.py — what the gateway learns from providers and must remember.

Today a single thing: the real completion cap (`max_output_tokens`) that a
provider reveals through an HTTP 400 "max_tokens must be ≤ N". Before ticket 037, this
value was rewritten in the installed package's YAML; it now lives behind this port
(Redis in production, file or memory without Redis).
"""
from __future__ import annotations

from typing import Protocol


class LearnedLimits(Protocol):
    def get_max_output_tokens(self, provider: str) -> int | None: ...

    def set_max_output_tokens(self, provider: str, limit: int) -> None: ...

    def all_max_output_tokens(self) -> dict[str, int]:
        """All the learned limits, {provider: limit}."""
        ...


__all__ = ["LearnedLimits"]
