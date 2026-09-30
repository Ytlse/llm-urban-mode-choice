"""
tests/test_worker_helpers.py — Unit tests for the helpers of worker/task_worker.py.

Covers:
  - _fit_request_budget            : 413 safeguard
  - _parse_ratelimit_reset_seconds : parsing of x-ratelimit-reset values
  - _parse_max_tokens_limit        : completion limit learned from a 400
  - learn_provider_max_output_tokens
(the domain helpers — main mode, distance bands, label mismatches — are
tested in mobility_llm/tests/unit/test_itinary_metrics.py)

No Celery/Redis/LLM call.
"""

from datetime import UTC, datetime, timedelta

import pytest

from llm_gateway.worker.task_worker import (
    ProviderCapacityError,
    _fit_request_budget,
    _parse_max_tokens_limit,
    _parse_ratelimit_reset_seconds,
)

# ---------------------------------------------------------------------------
# _fit_request_budget — 413 safeguard (per-request capacity vs rendered prompt)
# ---------------------------------------------------------------------------

class _Cfg:
    def __init__(self, max_tokens_per_request):
        self.max_tokens_per_request = max_tokens_per_request


class TestFitRequestBudget:
    def test_sans_capacite_max_tokens_inchange(self):
        assert _fit_request_budget(_Cfg(None), 5000, 4096, 512, "p") == 4096
        assert _fit_request_budget(None, 5000, 4096, 512, "p") == 4096

    def test_prompt_petit_max_tokens_inchange(self):
        # 8000 - 2000 = 6000 of budget > 4096 requested
        assert _fit_request_budget(_Cfg(8000), 2000, 4096, 512, "p") == 4096

    def test_gros_prompt_rogne_max_tokens(self):
        # Reflection prompt ~4500 tokens on an 8000 provider: 3500 of budget
        assert _fit_request_budget(_Cfg(8000), 4500, 8192, 512, "p") == 3500

    def test_prompt_trop_gros_leve_capacity_error(self):
        # Even the minimal output no longer fits → replay on another provider
        with pytest.raises(ProviderCapacityError) as exc:
            _fit_request_budget(_Cfg(6000), 5800, 4096, 512, "groq_qwen")
        assert exc.value.provider == "groq_qwen"
        assert exc.value.prompt_tokens_est == 5800
        assert exc.value.max_tokens_per_request == 6000


# ---------------------------------------------------------------------------
# _extract_primary_mode
# ---------------------------------------------------------------------------

class TestParseRatelimitResetSeconds:
    # Groq formats (duration)
    def test_seconds_only(self):
        assert _parse_ratelimit_reset_seconds("45s") == 47   # 45 + 2 of margin

    def test_minutes_and_seconds(self):
        result = _parse_ratelimit_reset_seconds("6m0s")
        assert result == 362  # 360 + 2

    def test_minutes_and_decimal_seconds(self):
        result = _parse_ratelimit_reset_seconds("59m17.087999999s")
        # int(59*60 + 17.087...) = 3557, +2 of margin = 3559
        assert result == 3559

    def test_small_seconds_clamped_to_10(self):
        # 3s + 2 = 5, below 10 → must be clamped to 10
        assert _parse_ratelimit_reset_seconds("3s") == 10

    def test_very_long_duration_clamped_to_3600(self):
        # 100 minutes → 6000s > 3600 → clamped
        assert _parse_ratelimit_reset_seconds("100m0s") == 3600

    def test_hours_format_clamped_to_3600(self):
        # Groq daily quota (TPD): delay in hours → clamped to 1h
        assert _parse_ratelimit_reset_seconds("2h37m12.5s") == 3600

    def test_hours_and_seconds_within_clamp(self):
        # 0h5m0s = 300 + 2
        assert _parse_ratelimit_reset_seconds("0h5m0s") == 302

    def test_milliseconds_clamped_to_10(self):
        assert _parse_ratelimit_reset_seconds("140ms") == 10

    def test_bare_seconds_retry_after_header(self):
        # Standard retry-after header: raw number of seconds
        assert _parse_ratelimit_reset_seconds("13") == 15  # 13 + 2

    def test_bare_decimal_seconds(self):
        assert _parse_ratelimit_reset_seconds("58.5") == 60  # int(58.5) + 2

    # Format ISO 8601 (OpenAI)
    def test_iso_8601_near_future(self):
        future = datetime.now(UTC) + timedelta(seconds=58)
        value = future.strftime("%Y-%m-%dT%H:%M:%SZ")
        result = _parse_ratelimit_reset_seconds(value)
        # Must be within roughly [58, 62] (±2 of margin + execution delay)
        assert 10 <= result <= 3600

    def test_iso_8601_past_clamped_to_10(self):
        past = datetime.now(UTC) - timedelta(seconds=30)
        value = past.strftime("%Y-%m-%dT%H:%M:%SZ")
        result = _parse_ratelimit_reset_seconds(value)
        assert result == 10  # negative delta + 2 < 10 → clamped

    # Null or invalid value → default
    def test_none_returns_default(self):
        assert _parse_ratelimit_reset_seconds(None) == 60

    def test_empty_string_returns_default(self):
        assert _parse_ratelimit_reset_seconds("") == 60

    def test_unparseable_string_returns_default(self):
        assert _parse_ratelimit_reset_seconds("not-a-date") == 60

    def test_custom_default(self):
        assert _parse_ratelimit_reset_seconds(None, default=30) == 30


# ---------------------------------------------------------------------------
# _parse_max_tokens_limit
# ---------------------------------------------------------------------------

class TestParseMaxTokensLimit:
    def test_groq_format(self):
        msg = (
            '[groq_llama4] HTTP 400: {"error":{"message":"`max_tokens` must be less '
            "than or equal to `8192`, the maximum value for `max_tokens` is less than "
            'the `context_window` for this model","type":"invalid_request_error","param":"max_tokens"}}'
        )
        assert _parse_max_tokens_limit(msg) == 8192

    def test_groq_format_without_backticks(self):
        assert _parse_max_tokens_limit("max_tokens must be less than or equal to 512") == 512

    def test_openai_format(self):
        msg = "max_tokens is too large: 20000. This model supports at most 16384 completion tokens"
        assert _parse_max_tokens_limit(msg) == 16384

    def test_limited_to_format(self):
        assert _parse_max_tokens_limit("max_output_tokens is limited to 8192 for this model") == 8192

    def test_unrelated_400_returns_none(self):
        assert _parse_max_tokens_limit("Invalid API key provided") is None

    def test_empty_returns_none(self):
        assert _parse_max_tokens_limit("") is None


# ---------------------------------------------------------------------------
# learn_provider_max_output_tokens / providers.yaml persistence
# ---------------------------------------------------------------------------

class TestLearnProviderMaxOutputTokens:
    """The learned limit goes into the store and the process config, no longer into a YAML."""

    def _settings(self, max_output_tokens=None):
        from pydantic import SecretStr

        from llm_gateway.config import ProviderConfig

        cfg = ProviderConfig(
            api_key=SecretStr("k"), rpm_limit=30, base_url="http://x",
            default_model="m", max_output_tokens=max_output_tokens,
        )
        return type("S", (), {"providers": {"groq_llama4": cfg}})(), cfg

    def test_learn_updates_memory_and_store(self):
        from llm_gateway.config import learn_provider_max_output_tokens
        from llm_gateway.testing import InMemoryLearnedLimits

        settings, cfg = self._settings()
        store = InMemoryLearnedLimits()
        assert learn_provider_max_output_tokens(settings, store, "groq_llama4", 8192) is True
        assert cfg.max_output_tokens == 8192
        assert store.get_max_output_tokens("groq_llama4") == 8192

    def test_already_known_limit_returns_false(self):
        from llm_gateway.config import learn_provider_max_output_tokens
        from llm_gateway.testing import InMemoryLearnedLimits

        settings, _ = self._settings(max_output_tokens=8192)
        store = InMemoryLearnedLimits()
        assert learn_provider_max_output_tokens(settings, store, "groq_llama4", 8192) is False
        assert learn_provider_max_output_tokens(settings, store, "groq_llama4", 16384) is False
        assert store.all_max_output_tokens() == {}

    def test_unknown_provider_and_invalid_limit_return_false(self):
        from llm_gateway.config import learn_provider_max_output_tokens
        from llm_gateway.testing import InMemoryLearnedLimits

        settings, _ = self._settings()
        store = InMemoryLearnedLimits()
        assert learn_provider_max_output_tokens(settings, store, "nope", 8192) is False
        assert learn_provider_max_output_tokens(settings, store, "groq_llama4", 0) is False

    def test_apply_learned_limits_only_tightens(self):
        from llm_gateway.config import apply_learned_limits
        from llm_gateway.testing import InMemoryLearnedLimits

        settings, cfg = self._settings(max_output_tokens=16384)
        store = InMemoryLearnedLimits()
        store.set_max_output_tokens("groq_llama4", 8192)
        store.set_max_output_tokens("inconnu", 100)
        assert apply_learned_limits(settings, store) == 1
        assert cfg.max_output_tokens == 8192
        store.set_max_output_tokens("groq_llama4", 65536)   # looser: ignored
        assert apply_learned_limits(settings, store) == 0
        assert cfg.max_output_tokens == 8192


# ---------------------------------------------------------------------------
# _count_mode_mismatches — does the LLM score the option it thinks it scores?
# ---------------------------------------------------------------------------
