"""
load_balancer/router.py — Smart multi-provider router.

Implements:
  - Weighted Round-Robin   : distribution proportional to RPM quotas (NGINX SWRR,
                             pure sequence built by core.selection)
  - Atomic reservation     : delegated to the RateLimiter port (Redis Lua script) — no
                             concurrent worker can exceed the configured rpm_limit
  - Circuit Breaker        : temporary exclusion (cooldown) on 5xx / 429 errors
  - Temporary disabling    : automatic exclusion after N consecutive errors,
                               re-enabled after `disable_timeout` seconds (default 180s)

The constructor does NOT touch Redis: resetting the RPM windows is an explicit
step of the API lifespan (cf. api/app.py), never again an import side
effect.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from llm_gateway.config import ProviderConfig
from llm_gateway.core.selection import build_swrr_sequence
from llm_gateway.ports.rate_limiter import RateLimiter
from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)


class RestrictionInstances(ValueError):
    """The list of allowed instances designates nothing usable — ticket 084.

    ⚠ **A wrong restriction is never "no restriction".** This is the rule that gives the
    batch its value: a typo that meant "help yourself everywhere" would cast doubt on
    every measurement taken under restriction, without a single line reporting it. It is
    therefore raised at selection, before a single call is paid for.
    """


class LoadBalancer:
    """
    Selects the best provider for each request and atomically reserves an RPM slot
    (via the injected RateLimiter) before returning its name.
    """

    def __init__(self, providers: dict[str, ProviderConfig], limiter: RateLimiter,
                 policy: str = "swrr") -> None:
        self._providers = providers
        self._limiter = limiter
        self._policy = policy
        self._lock = threading.Lock()
        self._cursor: int = 0
        self._sequence: list[str] = self._build_sequence()
        if policy == "cascade":
            logger.info(f"CASCADE routing: priority order = {self._cascade()}")

    def _admises(self, admises: Iterable[str] | None) -> frozenset[str] | None:
        """Set of allowed instances, VALIDATED, or `None` when nothing is restricted.

        Ticket 084. Three possible returns and a single silence: `None` (no restriction)
        and a non-empty set. Everything else raises — a list designating no known
        instance, or mixing known and unknown ones. A half-wrong list is a configuration
        error, not an intent to guess.
        """
        if admises is None:
            return None
        demandees = frozenset(str(nom).strip() for nom in admises if str(nom).strip())
        if not demandees:
            # Empty list == no restriction. This is the case of a setting declared but not
            # filled in, and it must behave exactly as before the ticket.
            return None
        inconnues = demandees - set(self._providers)
        if inconnues:
            raise RestrictionInstances(
                f"unknown admitted instances: {sorted(inconnues)} — known: "
                f"{sorted(self._providers)}. A half-wrong list is not a partial "
                f"restriction, it is a configuration error."
            )
        return demandees

    def _cascade(self) -> list[str]:
        """The priority order in cascade mode: the configuration's, duplicates removed.

        The first provider is used up before the next is touched, instead of spreading the
        load: two keys on the same model are thus consumed one after the other.
        """
        vus: list[str] = []
        for nom in self._sequence:
            if nom not in vus:
                vus.append(nom)
        return vus

    # ------------------------------------------------------------------
    # Building the weighted sequence
    # ------------------------------------------------------------------

    def _build_sequence(self) -> list[str]:
        if not self._providers:
            logger.error("No provider available — empty SWRR sequence.")
            return []
        # weight 0 = provider defined but OUT OF ROTATION (build_swrr_sequence
        # guarantees ≥ 1 slot per entry, so filtering must happen here). Still
        # usable via a forced `llm.provider` in the experiment config.
        in_rotation = {name: cfg.weight for name, cfg in self._providers.items()
                       if cfg.weight > 0}
        excluded = sorted(set(self._providers) - set(in_rotation))
        if excluded:
            logger.info(f"Providers out of rotation (weight 0): {excluded}")
        if not in_rotation:
            logger.error("All providers have weight 0 — empty SWRR sequence.")
            return []
        sequence = build_swrr_sequence(in_rotation)
        logger.debug(f"WRR sequence built | sequence={sequence}")
        return sequence

    def rebuild_sequence(self) -> None:
        with self._lock:
            self._sequence = self._build_sequence()
            self._cursor = 0

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def select_provider(
        self,
        force: str | None = None,
        min_tpm: int | None = None,
        min_output: int | None = None,
        admises: Iterable[str] | None = None,
    ) -> str:
        """
        Returns the name of the provider to use and atomically reserves an RPM slot.

        Args:
            force:      If given, bypasses the rotation and uses this provider.
            min_tpm:    If given, excludes providers whose tpm_limit < min_tpm
                        (providers without a TPM limit, tpm_limit=None, remain eligible).
            min_output: If given, excludes providers whose completion ceiling
                        max_output_tokens < min_output (output budget of a task).
                        Providers without a known limit (None) remain eligible.
            admises:    Ticket 084 — LIST of instances allowed to serve this request. The
                        restriction is honoured HERE, before any reservation and thus before
                        a call is paid for. This is what sets it apart from the client filter
                        `allowed_providers`, which rejects an already billed response. `None`
                        or empty list: no restriction, behaviour from before the ticket.

        Raises:
            RuntimeError: If all allowed providers are saturated or unavailable.
            RestrictionInstances: If `admises` designates no known instance, or mixes in
                        unknown ones — a wrong list is never "no restriction".
        """
        _admises = self._admises(admises)

        if force:
            # Two contradictory constraints are a configuration error, not a priority to
            # settle silently: serving the pinned one would ignore the restriction, and
            # the reverse would ignore the pinning. We refuse, and say so.
            if _admises is not None and force not in _admises:
                raise RestrictionInstances(
                    f"forced provider '{force}' outside the admitted instances "
                    f"{sorted(_admises)} — contradictory constraints, neither is "
                    f"settled silently."
                )
            if self._try_reserve(force, min_tpm=min_tpm, min_output=min_output):
                return force
            raise RuntimeError(
                f"Forced provider '{force}' unavailable (quota reached or disabled)."
            )

        if self._policy == "cascade":
            # Cascade: always start again from the first. It is skipped only if it REFUSES
            # (daily quota, cooldown, per-minute rate saturated), and the refusal is logged
            # so that a switchover can be read in the logs.
            ordre = self._cascade()
            if _admises is not None:
                # The cascade ORDER is kept, only the set shrinks: the priority declared
                # in the configuration remains the priority under restriction.
                ordre = [nom for nom in ordre if nom in _admises]
            for rang, candidate in enumerate(ordre):
                if self._try_reserve(candidate, min_tpm=min_tpm, min_output=min_output):
                    if rang:
                        logger.info(
                            f"Cascade: switching to '{candidate}' (rank {rang + 1}/{len(ordre)}) — "
                            f"the previous ones refused: {', '.join(ordre[:rang])}"
                        )
                    return candidate
            raise RuntimeError(
                "All LLM providers saturated or at their concurrency limit. "
                f"Cascade exhausted in order: {', '.join(ordre)}."
                + (
                    f" Restriction in force: {sorted(_admises)} — no fallback outside "
                    f"this set is attempted."
                    if _admises is not None
                    else ""
                )
            )

        # Normal rotation — the lock only protects reading/writing the cursor.
        # _try_reserve() makes its Redis calls OUTSIDE the lock.
        seq_len = len(self._sequence)
        for _ in range(seq_len):
            with self._lock:
                candidate = self._sequence[self._cursor % seq_len]
                self._cursor += 1

            # The cursor advances even on a skipped instance: the weighted sequence is still
            # walked identically, and two requests with different restrictions do not
            # shift each other.
            if _admises is not None and candidate not in _admises:
                continue

            if self._try_reserve(candidate, min_tpm=min_tpm, min_output=min_output):
                return candidate

        raise RuntimeError(
            "All LLM providers saturated or at their concurrency limit. "
            "Retry in a few seconds."
            + (
                f" Restriction in force: {sorted(_admises)} — no fallback outside this "
                f"set is attempted."
                if _admises is not None
                else ""
            )
        )

    def _try_reserve(
        self,
        provider: str,
        min_tpm: int | None = None,
        min_output: int | None = None,
    ) -> bool:
        """
        Applies the routing constraints (known provider, min_tpm, min_output)
        then delegates the atomic reservation (disabling, cooldown, concurrency,
        quota + smoothing) to the RateLimiter.
        """
        cfg = self._providers.get(provider)
        if cfg is None:
            logger.warning(f"Unknown provider in the config | provider={provider}")
            return False

        if min_tpm is not None and cfg.tpm_limit is not None and cfg.tpm_limit < min_tpm:
            return False

        if (
            min_output is not None
            and cfg.max_output_tokens is not None
            and cfg.max_output_tokens < min_output
        ):
            return False

        return self._limiter.try_reserve(provider)

    def get_status(self) -> dict[str, dict]:
        """Snapshot of the RPM counters for monitoring / debug (/health)."""
        status = {}
        for name, cfg in self._providers.items():
            current = self._limiter.current_rpm(name)
            quota_exhausted = self._limiter.is_quota_exhausted(name)
            available = (
                not self._limiter.is_disabled(name)
                and not self._limiter.is_in_cooldown(name)
                and not quota_exhausted
                and current < cfg.rpm_limit
                and self._limiter.active_workers(name) < cfg.concurrency_limit
            )
            status[name] = {
                "current_rpm":     current,
                "disabled":        self._limiter.is_disabled(name),
                "rpm_limit":       cfg.rpm_limit,
                "active_tasks":    self._limiter.active_workers(name),
                "usage_pct":       round(current / cfg.rpm_limit * 100, 1) if cfg.rpm_limit else 0,
                "cooldown":        self._limiter.is_in_cooldown(name),
                # Ticket 105 — INDICATIVE figures: they only see this gateway's traffic
                # and never close a key (only the provider's 429 does).
                # The field names stay stable: the dashboard reads them.
                "daily_requests":  self._limiter.daily_requests_local_seulement(name),
                "rpd_limit":       cfg.rpd_limit,
                "daily_tokens":    self._limiter.daily_tokens_local_seulement(name),
                "tpd_limit":       cfg.tpd_limit,
                "quota_exhausted": quota_exhausted,
                "available":       available,
                # Published so that an experiment's cost estimate converts TRIPS into
                # REQUESTS without copying the formula: the batch ceiling is computed at
                # startup from THIS container's environment, and the value read elsewhere
                # in `providers.yaml` may diverge from it.
                "batch_max_agents":          cfg.batch_max_agents,
                "tpm_estimate_per_request":  cfg.tpm_estimate_per_request,
            }
        return status
