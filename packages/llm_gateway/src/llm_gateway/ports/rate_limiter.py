"""
ports/rate_limiter.py — Contract for RPM quotas, circuit breaker and concurrency.

Groups all the "provider health" state shared between workers:
  - sliding RPM window with time smoothing (min_interval = ttl / limit)
  - cooldown (429 / 5xx) and temporary deactivation (consecutive errors)
  - active worker counter (concurrency limit per provider)
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class RateLimiter(Protocol):
    # ── RPM reservation ──────────────────────────────────────────────────
    def try_reserve(self, provider: str, est_tokens: int | None = None) -> bool:
        """Atomically reserves an RPM slot (checks deactivation, cooldown,
        concurrency, quota + smoothing). True if the slot is acquired.
        `est_tokens` overrides the static TPM estimate (tpm_estimate_per_request)
        when the real size of the request is known."""
        ...

    def release_slot(self, provider: str, est_tokens: int | None = None) -> None:
        """Gives back the reserved RPM + TPM slots when the LLM call fails.
        `est_tokens` must equal the amount actually reserved (after a
        possible adjust_tokens); default = static estimate."""
        ...

    def adjust_tokens(self, provider: str, delta: int) -> None:
        """Corrects the TPM reservation of the current sliding window:
        delta > 0 when the real request exceeds the estimate made at
        reservation, delta < 0 when it is smaller (frees headroom).
        No effect if the provider has no tpm_limit."""
        ...

    def current_rpm(self, provider: str) -> int: ...

    def current_tpm(self, provider: str) -> int:
        """Estimated tokens reserved in the 60s sliding window (0 if no tpm_limit)."""
        ...

    def reset_windows(self) -> None:
        """Resets the RPM + TPM windows of all providers (API startup)."""
        ...

    # ── Daily quotas (RPD / TPD) ────────────────────────────────────────
    def record_tokens(self, provider: str, tokens: int) -> None:
        """Counts the consumed tokens for enforcing the tpd_limit quota
        (the provider's daily window, cf. `quota_reset_tz`). Called after each
        successful LLM call."""
        ...

    def is_quota_exhausted(self, provider: str) -> bool:
        """True if the provider has exhausted its daily quota (RPD/TPD) and stays
        excluded until its day resets."""
        ...

    def mark_quota_exhausted_until(
        self, provider: str, kind: str = "rpd", until: datetime | None = None
    ) -> int:
        """Excludes the provider until its day resets, on the PROVIDER's word.

        Called on a 429 whose body designates a daily quota: this is the authoritative
        path, since the local counter only sees this gateway's traffic. `until` defaults
        to the next midnight in the provider's time zone. Returns the applied TTL, in
        seconds.
        """
        ...

    def clear_quota_exhausted(self, provider: str) -> None:
        """Lifts the daily quota exhaustion lock for this provider so the API can be retried."""
        ...

    # Ticket 105 — the suffix is part of the contract: these two counters only see the
    # traffic of THIS gateway, while the same API key also serves `scripts/synthesis/*`
    # and `prompt_calibration`. Indicative: they never close a key. Only the
    # provider's 429 does, via `mark_quota_exhausted_until`.
    def daily_requests_local_seulement(self, provider: str) -> int: ...

    def daily_tokens_local_seulement(self, provider: str) -> int: ...

    # ── Circuit breaker ──────────────────────────────────────────────────
    def cooldown(self, provider: str, seconds: int) -> None: ...

    def is_in_cooldown(self, provider: str) -> bool: ...

    def cooldown_ttl(self, provider: str) -> int:
        """Seconds until the cooldown ends (0 if not in cooldown)."""
        ...

    def disable(self, provider: str, seconds: int) -> None: ...

    def is_disabled(self, provider: str) -> bool: ...

    def disabled_ttl(self, provider: str) -> int:
        """Seconds until automatic reactivation (0 if active)."""
        ...

    def record_failure(self, provider: str) -> int:
        """Increments the consecutive errors; returns the new value."""
        ...

    def record_success(self, provider: str) -> None:
        """Resets the consecutive error counter."""
        ...

    # ── Concurrency ──────────────────────────────────────────────────────
    def incr_active(self, provider: str) -> int: ...

    def decr_active(self, provider: str) -> int: ...

    def active_workers(self, provider: str) -> int: ...
