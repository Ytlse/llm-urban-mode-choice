"""A provider's daily quota window: when it closes, when it reopens.

Two traps, both measured on 2026-09-08 on an experiment stuck at 10%:

1. **The time zone.** The Gemini free tier daily quota resets at *Pacific* midnight,
   not at UTC midnight. The internal counters, indexed on the UTC day, therefore
   emptied 7 h too early: at 08:44 Paris time, Redis showed 49 requests out of
   500 while Google refused for exceeding the 500 (490 counted the day before +
   the morning's requests, in the SAME Pacific day).

2. **The `retryDelay`.** On a daily overrun, Gemini returns a delay of a few
   seconds (0.7 s to 57 s observed) that says NOTHING about the time until the reset.
   Taking it at its word loops: we retry 30 s later, get refused, indefinitely.
   The body however carries the `quotaId` — `…PerDayPerProjectPerModel…` — which settles it.

The internal counter remains an optimistic safeguard: it only sees the traffic that went
through this gateway, while a key is also hit by `scripts/synthesis/*` and `prompt_calibration`.
Only the provider's response is authoritative for declaring a window closed.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, tzinfo

# `quotaId: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"`, and the variants
# met in plain-text messages. Same semantics as `_DAILY_QUOTA_RE` in
# prompt_calibration (standalone repo: the logic is duplicated there, not imported).
DAILY_QUOTA_RE = re.compile(r"per\s*day|requestsperday|tokensperday", re.IGNORECASE)

DEFAUT_FUSEAU_QUOTA = "UTC"


def is_daily_quota_error(corps: str | None) -> bool:
    """Does the refusal concern a DAILY quota (as opposed to a per-minute rate)?

    On a daily quota, the delay announced by the provider must be discarded:
    `next_quota_reset()` gives the reopening time.
    """
    if not corps:
        return False
    return bool(DAILY_QUOTA_RE.search(corps))


def _fuseau(nom: str | None) -> tzinfo:
    """The requested time zone, UTC if the environment cannot resolve it (fail-safe)."""
    if not nom:
        return UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(nom)
    except Exception:  # noqa: BLE001 — fuseau inconnu ou base tzdata absente
        return UTC


def next_quota_reset(fuseau: str | None, maintenant: datetime | None = None) -> datetime:
    """Next midnight in `fuseau`, returned in UTC. Daylight saving changes included."""
    tz = _fuseau(fuseau)
    local = (maintenant or datetime.now(UTC)).astimezone(tz)
    lendemain = (local + timedelta(days=1)).date()
    minuit = datetime(lendemain.year, lendemain.month, lendemain.day, tzinfo=tz)
    return minuit.astimezone(UTC)


def quota_day(fuseau: str | None, maintenant: datetime | None = None) -> str:
    """Current quota day, "YYYYMMDD" in `fuseau` — the key of the day's counters."""
    return (maintenant or datetime.now(UTC)).astimezone(_fuseau(fuseau)).strftime("%Y%m%d")


def seconds_until_quota_reset(fuseau: str | None, maintenant: datetime | None = None) -> int:
    """Seconds before the window reopens (at least 1)."""
    now = maintenant or datetime.now(UTC)
    return max(1, int((next_quota_reset(fuseau, now) - now).total_seconds()))


__all__ = [
    "DAILY_QUOTA_RE",
    "DEFAUT_FUSEAU_QUOTA",
    "is_daily_quota_error",
    "next_quota_reset",
    "quota_day",
    "seconds_until_quota_reset",
]
