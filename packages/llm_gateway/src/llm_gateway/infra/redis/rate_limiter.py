"""
infra/redis/rate_limiter.py — RedisRateLimiter: RPM quotas, circuit breaker, concurrency.

Implements the RateLimiter port. Reserving an RPM slot combines, in a
single atomic Redis-side operation (Lua script):
  1. the check of the minimum interval since the last request
     (smoothing: min_interval = ttl / limit — avoids bursts at the start of the window)
  2. the slot reservation (INCR + limit check, DECR rollback otherwise)
  3. the update of the last-request timestamp
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import redis as sync_redis

from llm_gateway.config import ProviderConfig
from llm_gateway.core.quota import (
    DEFAUT_FUSEAU_QUOTA,
    next_quota_reset,
    quota_day,
)
from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)

RPM_KEY_PREFIX          = "rpm:"
TPM_KEY_PREFIX          = "tpm:"              # estimated tokens/minute (60s sliding window)
COOLDOWN_KEY_PREFIX     = "cooldown:"
CONS_ERR_KEY_PREFIX     = "cons_err:"
DISABLED_KEY_PREFIX     = "disabled:"
ACTIVE_WORKER_PREFIX    = "active_workers:"
LAST_REQUEST_KEY_PREFIX = "last_req:"
# ⚠ Ticket 105 — "local_seulement" is IN THE KEY NAME, and not only in a
# docstring, because `redis-cli --scan` is what gets read when wondering where a quota stands.
# These two counters see ONLY this gateway's traffic: the same API key is also
# consumed by `scripts/synthesis/*` and `prompt_calibration`. On 2026-09-08 the counter showed
# 49 requests out of 500 while Google was refusing for exceeding the 500; on 2026-09-23 it was
# read as a remaining budget and got a relaunch postponed by a day, for nothing.
# They are INDICATIVE: they block no reservation and no production code reads them.
# The only fact that closes a key is the provider's 429, via `mark_quota_exhausted_until`.
RPD_KEY_PREFIX          = "rpd_local_seulement:"   # requests/day SEEN FROM HERE (provider's tz)
TPD_KEY_PREFIX          = "tpd_local_seulement:"   # tokens/day SEEN FROM HERE (provider's tz)
QUOTA_EXHAUSTED_PREFIX  = "quota_exhausted:"  # provider excluded until ITS day resets

RPM_WINDOW_SECONDS = 60
DAILY_KEY_TTL_SECONDS = 90000  # ~25 h: covers the day + margin, automatic purge


# The daily counters and the exhausted-quota withdrawal are dated in the PROVIDER's
# time zone (cf. llm_gateway.core.quota): counting in UTC against a quota that
# resets on Pacific time emptied the counter 7 h too early, and a key refused
# by Google passed for available (incident of 2026-09-08, experiment stopped at 10%).

# Atomic RPM + TPM reservation (60s sliding window) with time smoothing.
# Order: smoothing → TPM reservation (estimated tokens) → RPM reservation. Any
# step that fails cancels the reservations already made (DECRBY/DECR rollback) so
# that a slot never leaks.
# Return values:
#   -1 : too early — min_interval not respected (smoothing)
#    0 : RPM or TPM quota reached (rollback)
#   >0 : slot reserved successfully (RPM counter value)
_TRY_RESERVE_RPM_SMOOTHED_SCRIPT = r"""
local rpm_key      = KEYS[1]
local last_req_key = KEYS[2]
local tpm_key      = KEYS[3]
local limit        = tonumber(ARGV[1])
local ttl          = tonumber(ARGV[2])
local now          = tonumber(ARGV[3])
local min_interval = tonumber(ARGV[4])
local tpm_limit    = tonumber(ARGV[5])
local est_tokens   = tonumber(ARGV[6])

if min_interval > 0 then
    local last = redis.call('GET', last_req_key)
    if last then
        local elapsed = now - tonumber(last)
        if elapsed < min_interval then
            return -1
        end
    end
end

-- TPM safeguard: reserves the estimated tokens in the 60 s sliding window.
-- Skipped if the provider has no tpm_limit (tpm_limit <= 0).
local tpm_reserved = false
if tpm_limit > 0 and est_tokens > 0 then
    local tcount = redis.call('INCRBY', tpm_key, est_tokens)
    if redis.call('TTL', tpm_key) == -1 then
        redis.call('EXPIRE', tpm_key, ttl)
    end
    if tcount > tpm_limit then
        redis.call('DECRBY', tpm_key, est_tokens)
        return 0
    end
    tpm_reserved = true
end

local count = redis.call('INCR', rpm_key)
if redis.call('TTL', rpm_key) == -1 then
    redis.call('EXPIRE', rpm_key, ttl)
end
if count > limit then
    redis.call('DECR', rpm_key)
    if tpm_reserved then
        redis.call('DECRBY', tpm_key, est_tokens)
    end
    return 0
end

redis.call('SET', last_req_key, tostring(now), 'EX', ttl)
return count
"""


def _rpm_key(provider: str) -> str:
    return f"{RPM_KEY_PREFIX}{provider}"


def _tpm_key(provider: str) -> str:
    return f"{TPM_KEY_PREFIX}{provider}"


def _last_req_key(provider: str) -> str:
    return f"{LAST_REQUEST_KEY_PREFIX}{provider}"


class RedisRateLimiter:
    def __init__(self, sync_client: sync_redis.Redis, providers: dict[str, ProviderConfig]) -> None:
        self._r = sync_client
        self._providers = providers

    # ------------------------------------------------------------------
    # RPM reservation
    # ------------------------------------------------------------------

    def _tz(self, provider: str) -> str:
        """Time zone of this provider's daily reset (safe default: UTC)."""
        cfg = self._providers.get(provider)
        return getattr(cfg, "quota_reset_tz", DEFAUT_FUSEAU_QUOTA) or DEFAUT_FUSEAU_QUOTA

    def try_reserve(self, provider: str, est_tokens: int | None = None) -> bool:
        """
        Checks the preconditions (disabled, cooldown, concurrency, quota)
        then atomically reserves an RPM slot (Lua, with time smoothing).
        `est_tokens` overrides the static TPM estimate when the real size
        of the request is known.
        """
        cfg = self._providers.get(provider)
        if cfg is None:
            return False

        if self.is_disabled(provider):
            return False

        if self.is_in_cooldown(provider):
            return False

        if self.active_workers(provider) >= cfg.concurrency_limit:
            return False

        # Daily quota (RPD/TPD) exhausted → provider excluded until midnight UTC.
        if self._daily_quota_exhausted(provider, cfg):
            return False

        # Optimization: skips the Lua round-trip if the quota is already exceeded.
        if self.current_rpm(provider) >= cfg.rpm_limit:
            return False

        now = time.time()
        limit = cfg.rpm_limit
        min_interval = RPM_WINDOW_SECONDS / limit if limit > 0 else 0.0
        # TPM guard: armed only if the provider declares a tpm_limit AND an
        # estimate (supplied by the caller, otherwise static, computed at build).
        # Otherwise tpm_limit=0 → the script skips it.
        tpm_limit = cfg.tpm_limit or 0
        if est_tokens is None:
            est_tokens = cfg.tpm_estimate_per_request or 0
        result = self._r.eval(
            _TRY_RESERVE_RPM_SMOOTHED_SCRIPT,
            3,
            _rpm_key(provider),
            _last_req_key(provider),
            _tpm_key(provider),
            limit,
            RPM_WINDOW_SECONDS,
            now,
            min_interval,
            tpm_limit,
            est_tokens,
        )
        reserved = int(result) > 0
        if reserved:
            self._incr_daily_requests(provider)
        return reserved

    # ------------------------------------------------------------------
    # Daily quotas (RPD / TPD)
    # ------------------------------------------------------------------

    def _daily_quota_exhausted(self, provider: str, cfg: ProviderConfig) -> bool:
        """True if the provider was excluded for its daily quota (on the provider's word).

        The local cap (rpd_limit/tpd_limit) is declarative and informative: it no longer
        blocks reservations locally, so that real overruns can be measured.
        Only the presence of the QUOTA_EXHAUSTED_PREFIX flag (set on an actual 429
        via mark_quota_exhausted_until) excludes the provider."""
        return bool(self._r.exists(f"{QUOTA_EXHAUSTED_PREFIX}{provider}"))

    def mark_quota_exhausted_until(
        self, provider: str, kind: str = "rpd", until: datetime | None = None
    ) -> int:
        """Excludes the provider until its day resets — on the PROVIDER's word.

        Called on a 429 whose body designates a daily quota. This is the authoritative
        path: the local counter only sees this gateway's traffic, while a key is
        also consumed by `scripts/synthesis/*` and `prompt_calibration` — on 2026-09-08 it
        showed 49 requests out of 500 while Google was refusing for exceeding the 500.

        Returns the applied TTL, in seconds.
        """
        tz = self._tz(provider)
        cible = until or next_quota_reset(tz)
        ttl = max(1, int((cible - datetime.now(UTC)).total_seconds()))
        self._r.set(f"{QUOTA_EXHAUSTED_PREFIX}{provider}", kind, ex=ttl)
        logger.error(
            f"[ALARME] Daily {kind.upper()} quota exhausted, announced by the provider — "
            f"provider excluded until reset | provider={provider} tz={tz} "
            f"reset={cible.isoformat(timespec='seconds')} reset_in={ttl}s"
        )
        return ttl

    def clear_quota_exhausted(self, provider: str) -> None:
        """Lifts the daily quota exhaustion lock for this provider so the API can be retried."""
        self._r.delete(f"{QUOTA_EXHAUSTED_PREFIX}{provider}")
        logger.info(f"Daily quota lock reset | provider={provider}")

    def _incr_daily_requests(self, provider: str) -> None:
        key = f"{RPD_KEY_PREFIX}{provider}:{quota_day(self._tz(provider))}"
        if self._r.incr(key) == 1:
            self._r.expire(key, DAILY_KEY_TTL_SECONDS)

    def record_tokens(self, provider: str, tokens: int) -> None:
        if tokens <= 0:
            return
        key = f"{TPD_KEY_PREFIX}{provider}:{quota_day(self._tz(provider))}"
        if self._r.incrby(key, tokens) == tokens:
            self._r.expire(key, DAILY_KEY_TTL_SECONDS)

    def daily_requests_local_seulement(self, provider: str) -> int:
        """Today's requests SEEN FROM THIS GATEWAY. Indicative — cf. RPD_KEY_PREFIX."""
        val = self._r.get(f"{RPD_KEY_PREFIX}{provider}:{quota_day(self._tz(provider))}")
        return int(val) if val else 0

    def daily_tokens_local_seulement(self, provider: str) -> int:
        """Today's tokens SEEN FROM THIS GATEWAY. Indicative — cf. TPD_KEY_PREFIX."""
        val = self._r.get(f"{TPD_KEY_PREFIX}{provider}:{quota_day(self._tz(provider))}")
        return int(val) if val else 0

    def is_quota_exhausted(self, provider: str) -> bool:
        return self._r.exists(f"{QUOTA_EXHAUSTED_PREFIX}{provider}") > 0

    def release_slot(self, provider: str, est_tokens: int | None = None) -> None:
        """Gives back the reserved RPM + TPM slots when the LLM call fails.
        Prevents errors from consuming quota without a successful call.
        `est_tokens` must equal the amount actually reserved (after a
        possible adjust_tokens); default = static estimate."""
        key = _rpm_key(provider)
        # Decrement only if the key exists (otherwise DECR would create -1)
        if self._r.exists(key):
            self._r.decr(key)
        # Symmetric give-back of the estimated TPM reservation (otherwise the sliding
        # window swells with each failure and over-throttles the provider).
        if est_tokens is None:
            cfg = self._providers.get(provider)
            est_tokens = cfg.tpm_estimate_per_request if cfg else None
        if est_tokens:
            self.adjust_tokens(provider, -est_tokens)

    def adjust_tokens(self, provider: str, delta: int) -> None:
        """Corrects the TPM reservation of the current sliding window (signed
        delta). Clamps to 0 so the window never goes negative (the key
        may have expired between the reservation and the adjustment)."""
        if delta == 0:
            return
        cfg = self._providers.get(provider)
        if cfg is None or not cfg.tpm_limit:
            return
        tpm_key = _tpm_key(provider)
        if delta < 0:
            if self._r.exists(tpm_key) and int(self._r.decrby(tpm_key, -delta)) < 0:
                self._r.set(tpm_key, 0, keepttl=True)
        else:
            if int(self._r.incrby(tpm_key, delta)) == delta:
                # Key created by this INCRBY → set the window TTL.
                self._r.expire(tpm_key, RPM_WINDOW_SECONDS)

    def current_rpm(self, provider: str) -> int:
        val = self._r.get(_rpm_key(provider))
        return int(val) if val else 0

    def current_tpm(self, provider: str) -> int:
        """Estimated tokens reserved in the 60s sliding window (monitoring)."""
        val = self._r.get(_tpm_key(provider))
        return int(val) if val else 0

    def reset_windows(self) -> None:
        """Resets the RPM + TPM windows — an explicit action of the API lifespan."""
        keys = [_rpm_key(name) for name in self._providers]
        keys += [_tpm_key(name) for name in self._providers]
        if keys:
            self._r.delete(*keys)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def cooldown(self, provider: str, seconds: int) -> None:
        self._r.set(f"{COOLDOWN_KEY_PREFIX}{provider}", "1", ex=seconds)

    def is_in_cooldown(self, provider: str) -> bool:
        return self._r.exists(f"{COOLDOWN_KEY_PREFIX}{provider}") > 0

    def cooldown_ttl(self, provider: str) -> int:
        ttl = self._r.ttl(f"{COOLDOWN_KEY_PREFIX}{provider}")
        return ttl if ttl > 0 else 0

    def disable(self, provider: str, seconds: int) -> None:
        """Temporary deactivation — when the TTL expires, Redis deletes the
        key and the provider becomes eligible again."""
        self._r.set(f"{DISABLED_KEY_PREFIX}{provider}", "1", ex=seconds)
        self._r.delete(f"{CONS_ERR_KEY_PREFIX}{provider}")

    def is_disabled(self, provider: str) -> bool:
        return self._r.exists(f"{DISABLED_KEY_PREFIX}{provider}") > 0

    def disabled_ttl(self, provider: str) -> int:
        ttl = self._r.ttl(f"{DISABLED_KEY_PREFIX}{provider}")
        return ttl if ttl > 0 else 0

    def record_failure(self, provider: str) -> int:
        return int(self._r.incr(f"{CONS_ERR_KEY_PREFIX}{provider}"))

    def record_success(self, provider: str) -> None:
        self._r.delete(f"{CONS_ERR_KEY_PREFIX}{provider}")

    # ------------------------------------------------------------------
    # Concurrency (active workers per provider)
    # ------------------------------------------------------------------

    def incr_active(self, provider: str) -> int:
        key = f"{ACTIVE_WORKER_PREFIX}{provider}"
        val = self._r.incr(key)
        if val == 1:
            self._r.expire(key, 600)  # Safety: expires after 10 min if the worker crashes
        return val

    def decr_active(self, provider: str) -> int:
        key = f"{ACTIVE_WORKER_PREFIX}{provider}"
        val = self._r.decr(key)
        if val < 0:
            self._r.set(key, 0)
            val = 0
        return val

    def active_workers(self, provider: str) -> int:
        val = self._r.get(f"{ACTIVE_WORKER_PREFIX}{provider}")
        return int(val) if val else 0
