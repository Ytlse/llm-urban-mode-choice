"""adapters/openai_adapter.py — OpenAI: native structured output (`response_format: json_schema`)."""
from __future__ import annotations

from typing import Any

from llm_gateway.adapters.base import register_adapter
from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter
from llm_gateway.core.models import InternalRequest

_REASONING_MODELS = ("o1", "o3", "o4", "gpt-5", "gpt-6", "luna", "sol", "terra")


@register_adapter
class OpenAIAdapter(OpenAICompatibleAdapter):
    provider_name = "openai"
    structured_output = "json_schema"
    schema_in_system = False

    def build_payload(self, request: InternalRequest) -> dict[str, Any]:
        payload = super().build_payload(request)
        model = str(payload.get("model") or "").lower()
        if any(token in model for token in _REASONING_MODELS):
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
            if payload.get("temperature") != 1:
                payload.pop("temperature", None)
        return payload


__all__ = ["OpenAIAdapter"]
