"""adapters/groq_adapter.py — Groq: OpenAI dialect, `json_object`, schema from the prompt."""
from __future__ import annotations

from llm_gateway.adapters.base import register_adapter
from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter


@register_adapter
class GroqAdapter(OpenAICompatibleAdapter):
    provider_name = "groq"
    structured_output = "json_object"
    schema_in_system = False


__all__ = ["GroqAdapter"]
