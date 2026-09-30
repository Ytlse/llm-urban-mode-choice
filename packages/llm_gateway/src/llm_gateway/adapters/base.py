"""
adapters/base.py — Common interface (Adapter Pattern).

Every provider adapter implements this abstract class.
The Worker only knows BaseAdapter — it stays decoupled from third-party SDKs.
"""

from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from typing import Any

try:
    from json_repair import repair_json as _repair_json
except ImportError:  # pragma: no cover - dépendance déclarée, garde-fou
    _repair_json = None

import httpx
from pydantic import ValidationError

from llm_gateway.config import get_settings
from llm_gateway.core.models import AgentResponse, InternalRequest, LLMOutput
from llm_gateway.telemetry.logger import get_logger

_base_logger = get_logger(__name__)


class BaseAdapter(ABC):
    """
    Contract every provider translator must honour.

    Main method: call()
      - Takes an InternalRequest (internal normalised format)
      - Returns (LLMOutput, tokens_in, tokens_out)
      - Raises an exception on a recoverable (5xx) or fatal (4xx) error
    """

    provider_name: str  # Must be set in each subclass (name of the adapter class)

    # Timeout of LLM calls — overridable per adapter (Google: 20s by default).
    request_timeout: float = float(os.getenv("ADAPTER_REQUEST_TIMEOUT", "30.0"))

    # Does it apply `request.thinking_budget`? False by default: asking a thinking depth of
    # an adapter that cannot pass it on would be a setting sealed in the fingerprint and
    # never applied — the exact defect the 2026-09-10 audit found on `temperature` of the
    # antigravity channel. The adapters that can pass it set this to True.
    applique_reflexion: bool = False
    _reflexion_signalee: bool = False

    def _signaler_reflexion_ignoree(self, budget: int | None) -> None:
        """Warns ONCE that a requested thinking depth is not applied."""
        if budget is None or self.applique_reflexion or self._reflexion_signalee:
            return
        self._reflexion_signalee = True
        _logger = __import__("llm_gateway.telemetry.logger", fromlist=["get_logger"]).get_logger(
            __name__
        )
        _logger.warning(
            f"[{self._instance_name}] thinking_budget={budget} requested but this adapter "
            "cannot pass it on: the provider applies its own default thinking. "
            "The setting is sealed in the fingerprint, it is NOT applied."
        )

    def __init__(self):
        # By default, the instance name = the name of the adapter class.
        # get_adapter() replaces it with the configured instance name (e.g. "groq_1").
        self._instance_name: str = self.provider_name
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    def _http(self) -> httpx.Client:
        """Shared httpx client of the instance (keep-alive between calls).

        httpx.Client is thread-safe: Celery workers (-P threads) can use it
        concurrently. Avoids the TCP/TLS handshake on every call
        (~100-300 ms saved per LLM call).
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=self.request_timeout)
        return self._client

    def close(self) -> None:
        """Releases the shared httpx client (LLMAdapter port)."""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    @abstractmethod
    def call(self, request: InternalRequest) -> tuple[LLMOutput, int, int]:
        """
        Performs the HTTP call to the LLM provider.

        Returns:
            Tuple[LLMOutput, tokens_in, tokens_out]

        Raises:
            ProviderServerError  : 5xx error → eligible for backoff retry
            ProviderClientError  : 4xx error → not retryable (bad request)
            ProviderParseError   : invalid JSON response or schema not respected
        """
        ...

    def _post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        """POST to the provider, network errors classified for the worker.

        A timeout, a refused connection or a cut in the middle of the response are
        **transient** failures: they become `ProviderServerError` (retried with backoff
        and cooldown, like a 5xx) instead of falling into the worker's generic
        exception, which failed the task without a retry.
        """
        try:
            return self._http().post(url, headers=headers, json=json)
        except httpx.TimeoutException as exc:
            raise ProviderServerError(
                self._instance_name, 504, f"Deadline exceeded ({self.request_timeout:.0f}s): {exc}",
                error_type="network_timeout",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderServerError(
                self._instance_name, 503, f"Connexion impossible : {exc}", error_type="network_connect",
            ) from exc
        except httpx.RemoteProtocolError as exc:
            raise ProviderServerError(
                self._instance_name, 502, f"Response interrupted: {exc}", error_type="network_protocol",
            ) from exc

    def _resolve_model(self, request: InternalRequest) -> str:
        """Returns the specified model or the provider's default."""
        if request.model:
            return request.model
        return get_settings().providers[self._instance_name].default_model

    def _get_api_key(self):
        return get_settings().providers[self._instance_name].api_key

    def _get_base_url(self) -> str:
        return get_settings().providers[self._instance_name].base_url

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raises a ProviderError if the HTTP status indicates an error.

        Captures the retry delay on 429s — x-ratelimit-reset header
        (Groq/OpenAI) or, failing that, the JSON body (Google Gemini) — and attaches
        it to the exception so that it shows up in llm_errors.jsonl.
        """
        if response.status_code < 400:
            return
        # Priority order: retry-after (HTTP standard, exact) then reset-tokens
        # (Groq 429s concern the TPM/TPD limits, not requests),
        # then the requests/generic variants.
        ratelimit_reset = (
            response.headers.get("retry-after")
            or response.headers.get("x-ratelimit-reset-tokens")
            or response.headers.get("x-ratelimit-reset-requests")
            or response.headers.get("x-ratelimit-reset")
        )
        # Fallback: some providers (Google Gemini) put the retry delay
        # in the JSON body rather than in a header.
        if not ratelimit_reset and response.status_code == 429:
            ratelimit_reset = extract_retry_delay_from_body(response.text)
        error_type = extract_error_type(response.text, response.status_code)
        if response.status_code >= 500:
            raise ProviderServerError(
                self._instance_name, response.status_code, response.text,
                error_type=error_type,
                ratelimit_reset=ratelimit_reset,
            )
        raise ProviderClientError(
            self._instance_name, response.status_code, response.text,
            error_type=error_type,
            ratelimit_reset=ratelimit_reset,
        )

    def _check_openai_finish_reason(self, data: dict) -> None:
        """Detects an output truncated at max_tokens (OpenAI /chat/completions format).

        finish_reason == "length" means generation was cut before the
        end → the JSON is incomplete and json.loads would fail with a misleading
        message (Expecting ',' delimiter...). Instead we raise a typed
        max_tokens_truncation error, eligible for retry (same pattern as google_adapter
        with finishReason == MAX_TOKENS).
        """
        choices = data.get("choices") or [{}]
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "length":
            _base_logger.warning(
                f"Response truncated (finish_reason=length) | provider={self._instance_name} "
                f"completion_tokens={data.get('usage', {}).get('completion_tokens')}"
            )
            raise ProviderServerError(
                self._instance_name, 503,
                "Output truncated at max_tokens limit (finish_reason=length)",
                error_type="max_tokens_truncation",
            )

    def _parse_output(self, raw: str) -> LLMOutput:
        provider = self._instance_name

        # Defensive cleanup: strip Markdown fences (e.g. ```json ... ```)
        raw_clean = raw.strip()
        if raw_clean.startswith('```json'):
            raw_clean = raw_clean[7:]
        if raw_clean.startswith('```'):
            raw_clean = raw_clean[3:]
        if raw_clean.endswith('```'):
            raw_clean = raw_clean[:-3]

        raw_clean = raw_clean.strip()

        # Uses a regex to extract the main dictionary (ignores stray text around it)
        match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
        if match:
            raw_clean = match.group(0)

        # _base_logger.debug(
        #     f"_parse_output start | provider={provider} raw_len={len(raw_clean)} "
        #     f"raw_preview={raw_clean[:300]!r}"
        # )

        # Guard: detect repetition loops (same token repeated > 15 times consecutively)
        words = raw_clean.split()
        if len(words) > 20:
            max_consecutive = 1
            consecutive = 1
            for i in range(1, len(words)):
                if words[i] == words[i - 1]:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1
            if max_consecutive > 15:
                raise ProviderParseError(
                    provider, raw,
                    f"Repetition loop detected ({max_consecutive} consecutive identical tokens)"
                )

        # Step 1 — JSON decode; on failure, tolerant repair (json-repair: trailing commas,
        # single quotes, missing braces, stray text) — designed for LLM outputs, unlike
        # demjson3, which it replaces (ticket 037).
        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError as e:
            repaired = None
            if _repair_json is not None:
                try:
                    repaired = _repair_json(raw_clean, return_objects=True)
                except Exception:  # json_repair does not normally raise; caution
                    repaired = None
            if not isinstance(repaired, dict) or not repaired:
                _base_logger.warning(
                    f"_parse_output FAILED at json.loads (repair impossible) | provider={provider} "
                    f"error={e} raw_preview={raw_clean[:500]!r}"
                )
                raise ProviderParseError(provider, raw, f"JSONDecodeError: {e}") from e
            _base_logger.warning(
                f"_parse_output: json.loads failed, repaired with json-repair | "
                f"provider={provider} original_error={e}"
            )
            data = repaired

        # Step 2 — extract "agents" list
        #
        # ⚠ THE "FIRST LIST THAT COMES" FALLBACK COST A MEASUREMENT. A model that forgets
        # the envelope and returns the agent alone — {"agent_id": …, "severity": …, "modes": []} —
        # has no `agents` key. The old fallback then took the first list of the dictionary,
        # that is `modes`: empty, the task reported "success, zero agents"; non-empty, its
        # items were strings and got discarded one by one. In both cases the
        # judgement was lost WITHOUT a single error being raised (measured on 2026-09-22 on
        # `evenement_jugement`, 19 calls out of 38).
        #
        # Two fallbacks, in this order: first recognise the UNWRAPPED agent, then
        # accept only a list of dictionaries. A list of strings is not a list
        # of agents, and taking it for one yields an empty result instead of an error.
        agents_raw = None
        if "agents" in data:
            agents_raw = data["agents"]
        elif "agent_id" in data:
            _base_logger.warning(
                f"_parse_output: UNWRAPPED response (agent alone, no 'agents' key) — "
                f"rewrapped | provider={provider} keys={list(data.keys())}"
            )
            agents_raw = [data]
        else:
            for cle, v in data.items():
                if isinstance(v, list) and all(isinstance(x, dict) for x in v) and v:
                    _base_logger.warning(
                        f"_parse_output: 'agents' key missing, list of dictionaries taken "
                        f"from '{cle}' | provider={provider} keys={list(data.keys())}"
                    )
                    agents_raw = v
                    break

        if agents_raw is None:
            _base_logger.warning(
                f"_parse_output FAILED: missing 'agents' key (and no alternative list found) | provider={provider} "
                f"top_level_keys={list(data.keys())} raw_preview={raw[:500]!r}"
            )
            raise ProviderParseError(
                provider, raw,
                f"KeyError: 'agents' absent, keys present: {list(data.keys())}"
            )
        if not isinstance(agents_raw, list):
            _base_logger.warning(
                f"_parse_output FAILED: 'agents' is not a list | provider={provider} "
                f"type={type(agents_raw).__name__} value={agents_raw!r}"
            )
            raise ProviderParseError(
                provider, raw,
                f"TypeError: 'agents' is of type {type(agents_raw).__name__}, expected list"
            )

        # Step 3 — build AgentResponse objects
        agents = []
        for idx, item in enumerate(agents_raw):
            if not isinstance(item, dict):
                _base_logger.warning(
                    f"_parse_output: skipping non-dict item at agents[{idx}] | provider={provider} "
                    f"type={type(item).__name__} value={item!r}"
                )
                continue
            try:
                agents.append(AgentResponse(**item))
            except (TypeError, ValidationError) as e:
                _base_logger.warning(
                    f"_parse_output FAILED at AgentResponse construction | provider={provider} "
                    f"index={idx} item={item!r} error={type(e).__name__}: {e}"
                )
                raise ProviderParseError(
                    provider, raw,
                    f"{type(e).__name__} on agents[{idx}]={item!r}: {e}"
                ) from e

        # ⚠ ZERO AGENTS IS NOT A SUCCESS. The gateway then returned `status=success` with an
        # empty result: the caller saw a successful task and no response, without a word
        # in the log to say where the gap came from. A batch is submitted FOR agents;
        # returning none is a generation failure, and it is retryable as such.
        if not agents:
            _base_logger.warning(
                f"_parse_output FAILED: zero agents after parsing | provider={provider} "
                f"top_level_keys={list(data.keys())} raw_preview={raw[:500]!r}"
            )
            raise ProviderParseError(
                provider, raw,
                f"zero agents returned (keys present: {list(data.keys())})",
            )

        return LLMOutput(agents=agents)


# ---------------------------------------------------------------------------
# Adapter-specific exceptions
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base."""
    def __init__(self, provider: str, status_code: int, message: str, error_type: str = "unknown", ratelimit_reset: str | None = None):
        self.provider = provider
        self.status_code = status_code
        self.error_type = error_type
        self.ratelimit_reset = ratelimit_reset
        super().__init__(f"[{provider}] HTTP {status_code}: {message}")


class ProviderServerError(ProviderError):
    """5xx error — eligible for retry with exponential backoff."""
    pass


class ProviderClientError(ProviderError):
    """4xx error — do not retry (invalid auth, quota exceeded, etc.)."""
    pass


class ProviderParseError(Exception):
    """The LLM response does not follow the expected JSON schema."""

    def __init__(self, provider: str, raw: str, detail: str):
        self.provider = provider
        self.raw = raw
        # Metric key: the first 10 words of the parsing detail
        self.error_type = _truncate_to_words(f"parse error {detail}", 10)
        super().__init__(f"[{provider}] Parse error: {detail}")


# ---------------------------------------------------------------------------
# Extraction of the raw error message from the response body
# ---------------------------------------------------------------------------

def _truncate_to_words(text: str, n: int = 10) -> str:
    """Returns the first n words of text, lowercased, without line breaks."""
    cleaned = " ".join(text.split())          # normalises whitespace / \n
    words = cleaned.lower().split()
    return " ".join(words[:n])


def extract_error_type(response_text: str, status_code: int) -> str:
    """
    Returns the first 10 words of the error message as sent back by the provider.

    This text is used directly as a metric key: each unique message
    automatically creates its own counter in Prometheus, with no predefined category.

    Examples:
      "request too large for model llama3-8b-8192 in organization"
      "you exceeded your current quota please check your plan"
      "invalid api key provided"
      "http 500"  (if the body is not valid JSON)
    """
    try:
        body = json.loads(response_text)
        err = body.get("error", {})

        # Format OpenAI / Groq / Mistral : {"error": {"message": "..."}}
        msg = (err.get("message") or "").strip()

        # Google format: the details are sometimes in error.message too
        if not msg:
            msg = (body.get("message") or body.get("error_message") or "").strip()

        if msg:
            return _truncate_to_words(msg, 10)

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Fallback: HTTP code only
    return f"http {status_code}"


def extract_retry_delay_from_body(response_text: str) -> str | None:
    """Extracts the retry delay from the body of a 429 response.

    Used as a fallback when the provider does not send the delay in a header
    (the Google Gemini case, which puts it in the JSON). Looks, in order, for:
      1. The structured field google.rpc.RetryInfo: error.details[].retryDelay
      2. The message text: "Please retry in 11.103190523s." (Gemini) or
         "Please try again in 16m7.68s" / "in 140ms" (Groq)

    Returns the duration string as is (e.g. "16m7.68s", parsable by
    _parse_ratelimit_reset_seconds), or None if nothing is found.
    """
    if not response_text:
        return None

    # 1. Structured field google.rpc.RetryInfo
    try:
        data = json.loads(response_text)
        for detail in data.get("error", {}).get("details", []):
            retry_delay = detail.get("retryDelay")
            if retry_delay:
                return str(retry_delay)  # e.g. "11s"
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # 2. Fallback on the message text — covers "retry in" (Gemini) and
    #    "try again in" (Groq), compound h/m/s/ms formats included.
    m = re.search(r"(?:re)?try(?: again)? in ((?:\d+(?:\.\d+)?(?:h|ms|m|s))+)", response_text)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# Adapter registry (auto-discovery by provider name)
# ---------------------------------------------------------------------------

ADAPTERS_ENTRY_POINT = "llm_gateway.adapters"

_REGISTRY: dict[str, type[BaseAdapter]] = {}

# Instances cached by provider name: the shared httpx client
# (keep-alive) survives from one call to the next instead of being recreated per batch.
_INSTANCES: dict[str, BaseAdapter] = {}
_INSTANCES_LOCK = threading.Lock()


def register_adapter(cls: type[BaseAdapter]) -> type[BaseAdapter]:
    """Registration decorator — used in each concrete adapter."""
    _REGISTRY[cls.provider_name] = cls
    return cls


def get_adapter(provider_name: str) -> BaseAdapter:
    """
    Returns the (cached) adapter matching the provider.

    For multi-instance providers (e.g. groq_1, groq_2), resolves the class
    through the `adapter` field of ProviderConfig, then attaches the instance name
    so that _resolve_model/_get_api_key read the right config.

    Raises:
        KeyError if the provider is not registered.
    """
    inst = _INSTANCES.get(provider_name)
    if inst is not None:
        return inst

    # Class resolution: `adapter` field or the provider name directly
    cfg = get_settings().providers.get(provider_name)
    adapter_key = (cfg.adapter or provider_name) if cfg else provider_name

    if adapter_key not in _REGISTRY:
        _load_adapters()

    if adapter_key not in _REGISTRY:
        raise KeyError(
            f"Unknown adapter for provider '{provider_name}' (adapter='{adapter_key}'). "
            f"Available adapters: {list(_REGISTRY.keys())}"
        )

    with _INSTANCES_LOCK:
        inst = _INSTANCES.get(provider_name)
        if inst is None:
            inst = _REGISTRY[adapter_key]()
            inst._instance_name = provider_name  # points to the right entry in settings.providers
            _INSTANCES[provider_name] = inst
    return inst


def close_all_adapters() -> None:
    """Closes the shared httpx clients of all instances (worker shutdown)."""
    with _INSTANCES_LOCK:
        for inst in _INSTANCES.values():
            inst.close()
        _INSTANCES.clear()


def _load_adapters() -> None:
    """Tries to load each known adapter. Missing imports are logged, not raised."""
    from llm_gateway.telemetry.logger import get_logger
    _logger = get_logger(__name__)

    _known_adapters = {
        "openai_compatible": "llm_gateway.adapters.openai_compatible",
        "openai":   "llm_gateway.adapters.openai_adapter",
        "google":   "llm_gateway.adapters.google_adapter",
        "mistral":  "llm_gateway.adapters.mistral_adapter",
        "groq":     "llm_gateway.adapters.groq_adapter",
        "cerebras": "llm_gateway.adapters.cerebras_adapter",
    }

    import importlib
    for name, module_path in _known_adapters.items():
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning(f"Adapter not available (missing module) | provider={name} reason={e}")

    # Adapters brought by other packages: entry point `llm_gateway.adapters`
    # (name = value of the `adapter` field of the providers file, target = BaseAdapter class).
    from importlib.metadata import entry_points
    for ep in entry_points(group=ADAPTERS_ENTRY_POINT):
        try:
            cls = ep.load()
        except Exception as e:  # a broken package must not block the other adapters
            _logger.error(f"[ALARME] External adapter cannot be loaded | entry_point={ep.name} value={ep.value} error={e!r}")
            continue
        if not (isinstance(cls, type) and issubclass(cls, BaseAdapter)):
            _logger.error(f"[ALARME] Entry point {ep.name!r} ({ep.value}) is not a subclass of BaseAdapter")
            continue
        if not getattr(cls, "provider_name", None):
            cls.provider_name = ep.name
        _REGISTRY.setdefault(cls.provider_name, cls)
        if ep.name != cls.provider_name:
            _REGISTRY.setdefault(ep.name, cls)
