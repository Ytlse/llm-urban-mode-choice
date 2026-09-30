"""adapters/cerebras_adapter.py — Cerebras: OpenAI dialect, `json_object`, schema copied into system."""
from __future__ import annotations

from llm_gateway.adapters.base import register_adapter
from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter


@register_adapter
class CerebrasAdapter(OpenAICompatibleAdapter):
    provider_name = "cerebras"
    structured_output = "json_object"
    schema_in_system = True


__all__ = ["CerebrasAdapter"]
