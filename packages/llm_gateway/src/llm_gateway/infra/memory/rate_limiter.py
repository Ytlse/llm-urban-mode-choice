"""
infra/memory/rate_limiter.py — InMemoryRateLimiter: RateLimiter port in pure
Python, for tests. 60s RPM window, smoothing and circuit breaker included.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import UTC, datetime

from llm_gateway.config import ProviderConfig
from llm_gateway.core.quota import (
    DEFAUT_FUSEAU_QUOTA,
    next_quota_reset,
    quota_day,
)

RPM_WINDOW_SECONDS = 60


# Quota day and withdrawal duration: in the PROVIDER's time zone (cf. core.quota),
# like the Redis implementation — the two must remain interchangeable.


class InMemoryRateLimiter:
    def __init__(self, providers: dict[str, ProviderConfig]) -> None:
        self._providers = providers
        self._rpm: dict[str, int] = defaultdict(int)
        self._tpm: dict[str, int] = defaultdict(int)
        self._window_start: dict[str, float] = {}
        self._last_req: dict[str, float] = {}
        self._cooldown_until: dict[str, float] = {}
        self._disabled_until: dict[str, float] = {}
        self._consecutive_errors: dict[str, int] = defaultdict(int)
        self._active: dict[str, int] = defaultdict(int)
        # Daily quotas (key = provider|PROVIDER's day) and exhaustion flag.
        self._daily_req: dict[tuple[str, str], int] = defaultdict(int)
        self._daily_tok: dict[tuple[str, str], int] = defaultdict(int)
        self._quota_exhausted_until: dict[str, float] = {}

    # ── RPM reservation ──────────────────────────────────────────────────

    def _maybe_reset_window(self, provider: str) -> None:
        start = self._window_start.get(provider)
        if start is None or time.time() - start >= RPM_WINDOW_SECONDS:
            self._window_start[provider] = time.time()
            self._rpm[provider] = 0
            self._tpm[provider] = 0

    def try_reserve(self, provider: str, est_tokens: int | None = None) -> bool:
        cfg = self._providers.get(provider)
        if cfg is None:
            return False
        if self.is_disabled(provider) or self.is_in_cooldown(provider):
            return False
        if self.active_workers(provider) >= cfg.concurrency_limit:
            return False

        if self._daily_quota_exhausted(provider, cfg):
            return False

        self._maybe_reset_window(provider)
        if self._rpm[provider] >= cfg.rpm_limit:
            return False

        min_interval = RPM_WINDOW_SECONDS / cfg.rpm_limit if cfg.rpm_limit > 0 else 0.0
        last = self._last_req.get(provider)
        if min_interval > 0 and last is not None and time.time() - last < min_interval:
            return False

        # Sliding TPM guard (same rules as the Redis limiter).
        if est_tokens is None:
            est_tokens = cfg.tpm_estimate_per_request or 0
        if cfg.tpm_limit and est_tokens and self._tpm[provider] + est_tokens > cfg.tpm_limit:
            return False

        self._rpm[provider] += 1
        self._tpm[provider] += est_tokens
        self._last_req[provider] = time.time()
        self._daily_req[(provider, quota_day(self._tz(provider)))] += 1
        return True

    # ── Daily quotas (RPD / TPD) ────────────────────────────────────────

    def _tz(self, provider: str) -> str:
        """Time zone of this provider's daily reset (safe default: UTC)."""
        cfg = self._providers.get(provider)
        return getattr(cfg, "quota_reset_tz", DEFAUT_FUSEAU_QUOTA) or DEFAUT_FUSEAU_QUOTA

    def _daily_quota_exhausted(self, provider: str, cfg: ProviderConfig) -> bool:
        """True if the provider was excluded for its daily quota (on the provider's word).

        The local cap (rpd_limit/tpd_limit) is declarative and informative: it no longer
        blocks reservations locally, so that real overruns can be measured.
        Only the presence of a time-based exclusion (set on an actual 429
        via mark_quota_exhausted_until) excludes the provider."""
        return time.time() < self._quota_exhausted_until.get(provider, 0.0)

    def record_tokens(self, provider: str, tokens: int) -> None:
        if tokens > 0:
            self._daily_tok[(provider, quota_day(self._tz(provider)))] += tokens

    def daily_requests_local_seulement(self, provider: str) -> int:
        """Today's requests SEEN FROM HERE. Indicative (ticket 105) — mirrors the Redis twin."""
        return self._daily_req[(provider, quota_day(self._tz(provider)))]

    def daily_tokens_local_seulement(self, provider: str) -> int:
        """Today's tokens SEEN FROM HERE. Indicative (ticket 105) — mirrors the Redis twin."""
        return self._daily_tok[(provider, quota_day(self._tz(provider)))]

    def mark_quota_exhausted_until(
        self, provider: str, kind: str = "rpd", until: datetime | None = None
    ) -> int:
        """Excludes the provider until its day resets — on the provider's word.

        Counterpart of the Redis implementation's `mark_quota_exhausted_until` (cf. its docstring).
        """
        cible = until or next_quota_reset(self._tz(provider))
        ttl = max(1, int((cible - datetime.now(UTC)).total_seconds()))
        self._quota_exhausted_until[provider] = time.time() + ttl
        return ttl

    def is_quota_exhausted(self, provider: str) -> bool:
        return time.time() < self._quota_exhausted_until.get(provider, 0.0)

    def clear_quota_exhausted(self, provider: str) -> None:
        """Lifts the daily quota exhaustion lock for this provider so the API can be retried."""
        self._quota_exhausted_until.pop(provider, None)

    def release_slot(self, provider: str, est_tokens: int | None = None) -> None:
        if self._rpm[provider] > 0:
            self._rpm[provider] -= 1
        if est_tokens is None:
            cfg = self._providers.get(provider)
            est_tokens = cfg.tpm_estimate_per_request if cfg else None
        if est_tokens:
            self._tpm[provider] = max(0, self._tpm[provider] - est_tokens)

    def adjust_tokens(self, provider: str, delta: int) -> None:
        cfg = self._providers.get(provider)
        if cfg is None or not cfg.tpm_limit or delta == 0:
            return
        self._tpm[provider] = max(0, self._tpm[provider] + delta)

    def current_rpm(self, provider: str) -> int:
        self._maybe_reset_window(provider)
        return self._rpm[provider]

    def current_tpm(self, provider: str) -> int:
        self._maybe_reset_window(provider)
        return self._tpm[provider]

    def reset_windows(self) -> None:
        self._rpm.clear()
        self._tpm.clear()
        self._window_start.clear()
        self._last_req.clear()

    # ── Circuit breaker ──────────────────────────────────────────────────

    def cooldown(self, provider: str, seconds: int) -> None:
        self._cooldown_until[provider] = time.time() + seconds

    def is_in_cooldown(self, provider: str) -> bool:
        return time.time() < self._cooldown_until.get(provider, 0.0)

    def cooldown_ttl(self, provider: str) -> int:
        remaining = self._cooldown_until.get(provider, 0.0) - time.time()
        return int(remaining) if remaining > 0 else 0

    def disable(self, provider: str, seconds: int) -> None:
        self._disabled_until[provider] = time.time() + seconds
        self._consecutive_errors[provider] = 0

    def is_disabled(self, provider: str) -> bool:
        return time.time() < self._disabled_until.get(provider, 0.0)

    def disabled_ttl(self, provider: str) -> int:
        remaining = self._disabled_until.get(provider, 0.0) - time.time()
        return int(remaining) if remaining > 0 else 0

    def record_failure(self, provider: str) -> int:
        self._consecutive_errors[provider] += 1
        return self._consecutive_errors[provider]

    def record_success(self, provider: str) -> None:
        self._consecutive_errors[provider] = 0

    # ── Concurrency ──────────────────────────────────────────────────────

    def incr_active(self, provider: str) -> int:
        self._active[provider] += 1
        return self._active[provider]

    def decr_active(self, provider: str) -> int:
        self._active[provider] = max(0, self._active[provider] - 1)
        return self._active[provider]

    def active_workers(self, provider: str) -> int:
        return self._active[provider]
