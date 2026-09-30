"""
adapters/openai_compatible.py — a single translator for any "/chat/completions" API.

OpenAI, Groq, Cerebras, Mistral, but also Ollama, vLLM, OpenRouter, LM Studio… speak the
same dialect: `POST {base_url}/chat/completions`, `choices[0].message.content`, `usage`.
What varies comes down to two settings, carried by the subclass **or** by the instance in the
providers file (`structured_output`, `schema_in_system`):

- `structured_output`: `json_schema` (native structured output, OpenAI), `json_object`
  (the model promises JSON, the schema travels in the prompt), `none` (nothing);
- `schema_in_system`: copy the JSON schema into the system message, for models that
  do not receive it otherwise (Mistral, Cerebras).

A compatible provider is therefore added **by configuration alone**:

    mon_ollama:
      adapter: openai_compatible
      structured_output: json_object
      base_url: http://ollama:11434/v1
      default_model: qwen3:8b
      rpm_limit: 60
"""
from __future__ import annotations

import json
from typing import Any, Literal

from llm_gateway.adapters.base import (
    BaseAdapter,
    ProviderClientError,
    register_adapter,
)
from llm_gateway.core.models import InternalRequest, LLMOutput

StructuredOutput = Literal["json_schema", "json_object", "none"]


@register_adapter
class OpenAICompatibleAdapter(BaseAdapter):
    provider_name = "openai_compatible"

    # Class defaults; the instance in the providers file can override them.
    structured_output: StructuredOutput = "json_object"
    schema_in_system: bool = False
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"

    # ── Effective settings (instance > class) ────────────────────────────────

    def _instance_config(self) -> Any:
        from llm_gateway.config import get_settings

        return get_settings().providers.get(self._instance_name)

    def _structured_output(self) -> StructuredOutput:
        cfg = self._instance_config()
        value = getattr(cfg, "structured_output", None) if cfg is not None else None
        return value or self.structured_output

    def _schema_in_system(self) -> bool:
        cfg = self._instance_config()
        value = getattr(cfg, "schema_in_system", None) if cfg is not None else None
        return self.schema_in_system if value is None else bool(value)

    # ── Request construction ────────────────────────────────────────────────

    def _messages(self, request: InternalRequest) -> list[dict[str, Any]]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if not self._schema_in_system():
            return messages
        instruction = (
            "\nTu dois répondre UNIQUEMENT en JSON valide, sans markdown, en respectant ce "
            f"schéma : {json.dumps(request.response_schema, ensure_ascii=False)}"
        )
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] = (msg["content"] or "") + instruction
                return messages
        messages.insert(0, {"role": "system", "content": instruction.strip()})
        return messages

    def _response_format(self, request: InternalRequest) -> dict[str, Any] | None:
        mode = self._structured_output()
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": "agents_output", "strict": True, "schema": request.response_schema},
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def build_payload(self, request: InternalRequest) -> dict[str, Any]:
        """The body sent to the provider; exposed for tests and derived adapters."""
        payload: dict[str, Any] = {
            "model": self._resolve_model(request),
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        response_format = self._response_format(request)
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _headers(self) -> dict[str, str]:
        key = self._get_api_key().get_secret_value()
        value = f"{self.auth_scheme} {key}".strip() if self.auth_scheme else key
        return {self.auth_header: value, "Content-Type": "application/json"}

    # ── Call ────────────────────────────────────────────────────────────────

    def _refuser_substitution_de_modele(self, request: InternalRequest, data: dict) -> None:
        """Refuses a response returned by a DIFFERENT model than the one requested.

        Observed on 2026-09-10 on LM Studio: an unknown model identifier produces no
        error — the server answers 200 and serves another loaded model. `qwen3.8-27b-local`
        (nonexistent) was served by `qwen3-vl-8b-instruct-mlx`, an 8B vision model instead of
        a 27B. The archive then carried a wrong model name, without anything reporting it: it
        is exactly the substitution that `allowed_providers` refuses elsewhere
        (llm_agent.py:711).

        The check only applies if the response declares a model. Servers that return none
        pass: refusing on a missing field would block compliant
        providers.
        """
        servi = str(data.get("model") or "").strip()
        if not servi:
            return
        demande = str(self._resolve_model(request)).strip()
        if servi == demande:
            return
        raise ProviderClientError(
            self._instance_name,
            502,
            f"model substitution: {demande!r} requested, {servi!r} served — the measurement "
            f"would carry a wrong model name. Check `default_model` of this instance "
            f"against the models actually exposed by the server.",
        )

    def call(self, request: InternalRequest) -> tuple[LLMOutput, int, int]:
        self._signaler_reflexion_ignoree(request.thinking_budget)
        response = self._post(
            f"{self._get_base_url()}/chat/completions",
            headers=self._headers(),
            json=self.build_payload(request),
        )
        self._raise_for_status(response)
        data = response.json()
        self._check_openai_finish_reason(data)
        self._refuser_substitution_de_modele(request, data)
        raw_content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        return self._parse_output(raw_content), tokens_in, tokens_out


__all__ = ["OpenAICompatibleAdapter", "StructuredOutput"]
