"""adapters/mistral_adapter.py — Mistral: OpenAI dialect, `json_object`, schema copied into system.

Recent models also accept `json_schema`: `structured_output: json_schema` on
the instance is enough to switch, without code.
"""
from __future__ import annotations

from llm_gateway.adapters.base import register_adapter
from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter


@register_adapter
class MistralAdapter(OpenAICompatibleAdapter):
    provider_name = "mistral"
    structured_output = "json_object"
    schema_in_system = True


__all__ = ["MistralAdapter"]
