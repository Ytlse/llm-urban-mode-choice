"""
ports/llm_adapter.py — Contract of the LLM adapters.

BaseAdapter (llm_gateway.adapters.base) provides the reference implementation;
this protocol formalizes the lifecycle expected by the worker, notably close()
to release the shared httpx client (keep-alive).
"""

from __future__ import annotations

from typing import Protocol

from llm_gateway.core.models import InternalRequest, LLMOutput


class LLMAdapter(Protocol):
    def call(self, request: InternalRequest) -> tuple[LLMOutput, int, int]:
        """Performs the LLM call. Returns (output, tokens_in, tokens_out)."""
        ...

    def close(self) -> None:
        """Releases the resources (shared httpx client)."""
        ...
