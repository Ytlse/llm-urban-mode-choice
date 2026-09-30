"""
adapters/google_adapter.py — Translator for the Google Gemini API.

Target format (Gemini REST API):
  POST /v1beta/models/{model}:generateContent
  {
    "contents": [{"role": "user", "parts": [{"text": "..."}]}],
    "generationConfig": {
      "responseMimeType": "application/json",
      "responseSchema": {...}
    }
  }

Notable differences vs OpenAI:
  - "system" → separate systemInstruction (not in contents)
  - "assistant" → "model" in the Gemini role
  - Structured Output via generationConfig.responseSchema
  - Usage in usageMetadata (not usage)
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx

from llm_gateway.adapters.base import (
    BaseAdapter,
    ProviderClientError,
    ProviderServerError,
    register_adapter,
)
from llm_gateway.config.settings import get_settings
from llm_gateway.core.models import InternalRequest, LLMOutput
from llm_gateway.telemetry.logger import get_logger

_logger = get_logger(__name__)

# Threshold of consecutive MAX_TOKENS truncations beyond which we move from WARNING
# to an `[ALARME]` ERROR. Fires on the RISING EDGE (a single alarm per
# episode, re-armed by the first success): an isolated truncation is noise, a
# Output reserve when thinking is left to the model (`thinking_budget: -1`): we do not
# know how many thinking tokens it will use, and they are taken from the output budget.
# Value taken from `prompt_calibration` (`mutation_thinking_reserve`), proven on the
# genetic campaign.
RESERVE_REFLEXION = 2048

# series signals an undersized `max_tokens` ceiling or a repetition loop
# — and without this signal, the caller's retry replays it silently until exhaustion.
_MAX_TOKENS_ALARM_THRESHOLD = 3


@register_adapter
class GoogleAdapter(BaseAdapter):
    provider_name = "google"

    # Only adapter that can pass on the thinking depth (`generationConfig
    # .thinkingConfig`). Elsewhere, the setting is reported as not applied.
    applique_reflexion = True

    # Patience ceiling of ONE call. Measured on 2026-07-31 on
    # gemini-3.1-flash-lite-preview, batches of 15 personas from the `train` set with
    # full distribution per persona: 3.6 to 8.8 s per call, 2,742 completion
    # tokens at worst. The margin is therefore two orders of magnitude — it is
    # NOT this timeout that blocked the weighted re-evaluation (see docs/changelog.md
    # of 2026-07-31). Do not lengthen it without a measurement that justifies it: a
    # truly stuck call must eventually give control back.
    request_timeout = float(os.getenv("GOOGLE_ADAPTER_REQUEST_TIMEOUT", "20.0"))

    # Mapping of OpenAI roles → Gemini roles
    ROLE_MAP = {
        "user":      "user",
        "assistant": "model",
        # "system" is handled separately (systemInstruction)
    }

    def __init__(self):
        super().__init__()
        # Consecutive MAX_TOKENS truncations, per adapter instance (the
        # adapters are shared across threads: a lock is enough, the granularity
        # does not need to be exact).
        self._trunc_streak = 0
        self._trunc_lock = threading.Lock()

    def _note_truncation(self, model: str, detail: str) -> None:
        """Counts a truncation and raises the `[ALARME]` on the rising edge."""
        with self._trunc_lock:
            self._trunc_streak += 1
            streak = self._trunc_streak
        if streak == _MAX_TOKENS_ALARM_THRESHOLD:
            _logger.error(
                f"[ALARME] {streak} consecutive MAX_TOKENS truncations | "
                f"provider={self._instance_name} model={model} — the max_tokens "
                f"ceiling is undersized or the model is looping; the retries "
                f"will replay the same truncation. {detail}"
            )

    def _note_completion(self) -> None:
        """Re-arms the rising edge: a clean completion closes the episode."""
        with self._trunc_lock:
            self._trunc_streak = 0

    def _generation_config(self, request: InternalRequest) -> dict[str, Any]:
        """The `generationConfig` sent to the API — extracted from `call` to be testable as is.

        It used to be tested through a COPY in the tests, which did not even contain the level
        setting: the 2026-09-11 defect went to production without a test seeing it.
        A callable method closes that door.

        **Thinking depth** (2026-09-10). Until then no `thinkingConfig` was
        sent, so all Geminis ran with their default, unsteered thinking —
        while `thoughtsTokenCount` was already read and counted. `None` keeps this
        behaviour.

        The setting lives **inside** `thinkingConfig`, never next to it. The level had been
        placed as a sibling of `maxOutputTokens`, trusting the documentation; the API then
        answers 400 "Unknown name "thinking_level" at generation_config" on EVERY call, and the
        whole experiment falls over. Checked against the API on 2026-09-11:

            generationConfig.thinking_level               → 400
            generationConfig.thinkingLevel                → 400
            generationConfig.thinkingConfig.thinkingLevel → 200

        Level and budget are exclusive, and the API says so too ("You can only set only one of
        thinking budget and thinking level"). The `resolve_inference` cascade already refuses
        them together; the `elif` below is the second barrier.
        """
        budget = request.thinking_budget
        niveau = request.thinking_level
        self._refuser_reflexion_hors_plafond(budget)
        self._refuser_niveau_non_supporte(niveau)

        plafond = request.max_tokens
        if niveau is not None and niveau != "minimal":
            # A level does not say how many tokens thinking will take, and it is drawn
            # from the output budget: we reserve the flat amount, as for a dynamic budget.
            plafond = request.max_tokens + RESERVE_REFLEXION
        if budget is not None and budget != 0:
            # Thinking consumes the OUTPUT budget: without a reserve, a generous budget
            # truncates the response (MAX_TOKENS) and the call is lost.
            plafond = request.max_tokens + (budget if budget > 0 else RESERVE_REFLEXION)

        reflexion: dict[str, Any] = {}
        if niveau is not None:
            reflexion = {"thinkingLevel": niveau, "includeThoughts": False}
        elif budget is not None:
            reflexion = {"thinkingBudget": budget, "includeThoughts": False}

        return {
            "temperature":      request.temperature,
            "maxOutputTokens":  plafond,
            **({"topP": request.top_p} if request.top_p is not None else {}),
            **({"thinkingConfig": reflexion} if reflexion else {}),
            "response_mime_type": "application/json",
            "response_json_schema":   self._clean_schema(request.response_schema),
        }

    def call(self, request: InternalRequest) -> tuple[LLMOutput, int, int]:
        api_key = self._get_api_key()
        model   = self._resolve_model(request)

        system_instruction, contents = self._convert_messages(request)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": self._generation_config(request),
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        base_url = self._get_base_url()
        # API key in the x-goog-api-key header (never in the query string: URLs
        # end up in logs and error traces).
        url = f"{base_url}/models/{model}:generateContent"

        started = time.monotonic()
        try:
            response = self._http().post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key.get_secret_value(),
                },
                json=payload,
            )
            self._raise_for_status(response)
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - started
            # The timeout is the ONLY case where we will never know either the finishReason
            # or the tokens produced: we log at least the requested budget and the time
            # actually waited, to tell "generation too long" from
            # "ceiling too short".
            _logger.warning(
                f"Google API timeout after {elapsed:.1f}s "
                f"(limit {self.request_timeout:.0f}s) | provider={self._instance_name} "
                f"model={model} max_tokens_demandes={request.max_tokens} error={exc}"
            )
            # 504 matches Gateway Timeout, eligible for your worker's Retry mechanism
            # _instance_name (e.g. "google_gemma42_key1") and not provider_name ("google"):
            # the cooldown is indexed on the instance name configured in providers.yaml.
            raise ProviderServerError(self._instance_name, 504, f"Request timeout: {exc}", error_type="timeout") from exc

        elapsed = time.monotonic() - started
        data = response.json()

        # ── The three diagnostic quantities ──────────────────────────────────
        # Without them, a batch that "no longer moves" is indistinguishable from a slow
        # or truncated batch, or from a model returning an incomplete but valid answer. They
        # are therefore recorded BEFORE any exception is raised, and repeated in the
        # message of every error.
        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        # The "thinking" tokens of reasoning models are billed AND
        # counted against the maxOutputTokens ceiling: ignoring them underestimated both
        # the consumption and the cause of a truncation.
        thoughts_tokens = usage.get("thoughtsTokenCount", 0) or 0
        tokens_out = completion_tokens + thoughts_tokens

        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderClientError(
                self._instance_name, 400,
                f"No candidate returned after {elapsed:.1f}s "
                f"(tokens in={tokens_in} out={tokens_out}). Data: {data}")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "STOP")

        if "content" not in candidate or "parts" not in candidate["content"]:
            raise ProviderClientError(
                self._instance_name, 400,
                f"Response blocked or empty after {elapsed:.1f}s. "
                f"Reason: {finish_reason} (tokens in={tokens_in} "
                f"completion={completion_tokens} thoughts={thoughts_tokens})")

        self._refuser_substitution_de_modele(model, data)
        raw_content = candidate["content"]["parts"][0]["text"]

        _logger.debug(
            f"Google call | provider={self._instance_name} model={model} "
            f"latence={elapsed:.1f}s finishReason={finish_reason} "
            f"tokens_in={tokens_in} completion={completion_tokens} "
            f"thoughts={thoughts_tokens} plafond={request.max_tokens}"
        )

        if finish_reason == "MAX_TOKENS":
            detail = (f"latence={elapsed:.1f}s completion={completion_tokens} "
                      f"thoughts={thoughts_tokens} plafond={request.max_tokens}")
            if request.thinking_budget is not None and thoughts_tokens and not completion_tokens:
                # Thinking ate the whole response: say which of the two settings to lower,
                # rather than letting one conclude it is a repetition loop.
                detail += (f" · thinking requested={request.thinking_budget} — thinking "
                           f"consumed the budget without leaving a response: lower "
                           f"thinking_budget or raise max_tokens")
            _logger.warning(
                f"Response truncated (MAX_TOKENS) — ceiling reached or repetition "
                f"loop | provider={self._instance_name} model={model} {detail} "
                f"raw_preview={raw_content[:200]!r}"
            )
            self._note_truncation(model, detail)
            raise ProviderServerError(
                self._instance_name, 503,
                f"Output truncated at MAX_TOKENS limit ({detail}) — "
                f"possible repetition loop",
                error_type="max_tokens_truncation",
            )

        self._note_completion()
        return self._parse_output(raw_content), tokens_in, tokens_out

    def _refuser_reflexion_hors_plafond(self, budget: int | None) -> None:
        """Refuses a thinking budget above the instance's DECLARED ceiling.

        The provider would trim it without saying so, and the response only reports the
        thinking tokens consumed — never the budget applied. The experiment would then seal in
        its fingerprint a budget that never happened. Better to refuse the call.

        Without a declared `thinking_budget_max`, no check: we do not substitute a plausible
        figure for a missing measurement.
        """
        if budget is None or budget <= 0:
            return
        cfg = get_settings().providers.get(self._instance_name)
        plafond = getattr(cfg, "thinking_budget_max", None) if cfg else None
        if plafond and budget > int(plafond):
            raise ProviderClientError(
                self._instance_name, 400,
                f"thinking budget {budget} above the declared ceiling {plafond} for "
                f"{getattr(cfg, 'default_model', '?')!r}: the provider would trim it without "
                f"saying so and "
                f"the experiment's fingerprint would carry a budget not applied. Lower "
                f"thinking_budget, or fix thinking_budget_max in providers.yaml.",
            )

    def _refuser_substitution_de_modele(self, demande: str, data: dict) -> None:
        """Refuses a response returned by a DIFFERENT model than the one requested.

        Google returns `modelVersion`. The comparison is a PREFIX, not an equality: the API
        commonly answers with a versioned identifier (`gemini-3.5-flash-001` for
        `gemini-3.5-flash`), which is indeed the requested model. A retired alias served
        by its successor, however, does not start with the requested identifier — the case of
        `gemini-3.1-flash-lite-preview`, shut down on 2026-05-25, yet two September runs
        carry its name in their sealed fingerprint without any error
        having been raised.

        Without `modelVersion` in the response, no check: we do not refuse on a missing
        field.
        """
        servi = str(data.get("modelVersion") or "").strip()
        if not servi or not demande:
            return
        if servi == demande or servi.startswith(f"{demande}-"):
            return
        raise ProviderClientError(
            self._instance_name, 502,
            f"model substitution: {demande!r} requested, {servi!r} served. The measurement "
            f"would carry a wrong model name in its sealed fingerprint. Fix "
            f"`default_model` of this instance — a retired alias is served by its "
            f"successor without anything reporting it.",
        )

    def _refuser_niveau_non_supporte(self, niveau: str | None) -> None:
        """Refuses a thinking level the model does not accept.

        Levels vary from one model to another: `minimal` exists on gemini-3.6-flash and
        3.5-flash-lite, not on 3.7 or 3.8 — asking for it there returns 400. The check only
        applies if `thinking_levels` is declared for the instance: without a declaration, we
        let it through rather than block on a guessed list.
        """
        if niveau is None:
            return
        cfg = get_settings().providers.get(self._instance_name)
        connus = getattr(cfg, "thinking_levels", None) if cfg else None
        if connus and niveau not in connus:
            raise ProviderClientError(
                self._instance_name, 400,
                f"thinking level {niveau!r} not accepted by "
                f"{getattr(cfg, 'default_model', '?')!r}: declared levels "
                f"{', '.join(connus)}. Fix `thinking_level`, or `thinking_levels` in "
                f"providers.yaml if the list is wrong.",
            )

    def _convert_messages(
        self, request: InternalRequest
    ) -> tuple[str, list[dict]]:
        """
        Separates the 'system' message (→ systemInstruction) from the other messages
        and converts the roles to the Gemini format.
        """
        system_text = ""
        contents = []

        for msg in request.messages:
            if msg.role == "system":
                system_text += msg.content + "\n"
                continue
            gemini_role = self.ROLE_MAP.get(msg.role, "user")
            contents.append({
                "role":  gemini_role,
                "parts": [{"text": msg.content}],
            })

        return system_text.strip(), contents

    def _clean_schema(self, schema: dict) -> dict:
        """Recursively removes the fields not supported by Gemini."""
        UNSUPPORTED = {"additionalProperties", "$defs", "$schema", "title"}
        if isinstance(schema, dict):
            return {
                k: self._clean_schema(v)
                for k, v in schema.items()
                if k not in UNSUPPORTED
            }
        if isinstance(schema, list):
            return [self._clean_schema(i) for i in schema]
        return schema
