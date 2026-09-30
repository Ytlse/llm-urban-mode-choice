"""Computation of the backpressure applied to the GAMA /sync.

Pure functions, with no dependency on settings or on the FastAPI app, so that they are
unit-testable (tests/test_backpressure.py).

Two families:
  - reactive brake on fill level (``compute_backpressure_interval``,
    ``update_drain_mode``) — the historical safety net;
  - deadline-driven predictive control (ticket 003):
    ``ThroughputEwma`` (smoothed completion throughput), ``time_ewma`` (smoothing of the
    sim/real pace) and ``edf_feasibility`` / ``edf_hold_needed`` (EDF
    feasibility test that decides whether to hold the /sync).

Plus the hold on imminent departure (2026-09-25): ``departures_at_risk`` and
``hold_while`` hold the /sync as long as a departure decision has not been returned,
``late_departure_alarm_transition`` carries the alarm for departures served late.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass


def compute_backpressure_interval(
    in_progress_count: int,
    population_size: int,
    k: float,
    cap: float,
    sans_frein: int = 0,
) -> float:
    """Minimum delay (seconds) between two /sync responses, as a function of the queue fill level.

    Formula: ``cap * min(1, n / population)^k`` — the threshold is relative to the
    population, which guarantees that the maximum brake (cap) is reachable:
    the queue can never exceed the population size, so an absolute threshold
    above it would make the backpressure ineffective (cf. run of
    2026-07-07: backlog 886/901 with a 0.33s brake).

    With k=1.5 and cap=30 (defaults of `settings.world.min_internal_coeff_k/_cap`): ~1s at
    10% backlog, ~4.9s at 30%, ~10.6s at 50%, ~25.2s at 89%, 30s at 100%.

    ⚠ With `edf_enabled` and `predictive_backpressure_enabled` set to True (defaults), this
    cap·ratio^k brake is NOT applied by /sync: EDF predictive holding short-circuits it
    (`handle/application.py`, `predictive` branch). It is only used if one of the two flags
    is False. Drain mode (ratio ≥ `drain_trigger_ratio`) keeps priority in both cases.

    ⚠ FLOOR THRESHOLD, ADDED ON 2026-09-22 — without it, a run with ONE agent braked 30 s at
    EVERY decision. Since the ratio is relative to the population, a single pending activity on
    a population of 1 gives 1/1 = 1, hence the maximum brake. Measured on campaign c3: 8 s
    per simulation step and ~12 min per simulated day, i.e. a one-agent run SLOWER than a
    twenty-agent run (5 s per step). The safeguard calibrated for a thousand agents choked the small ones.

    What the floor states is the real invariant: as long as the number of pending activities
    fits within the SIMULTANEOUS capacity of the gateway, nothing queues, so nothing needs
    to be braked. Below `sans_frein`, all requests are in flight at the same
    time; the queue does not grow. Above it, the formula applies unchanged.

    Measured effect: 1/1 goes from 30.00 s to 0 s, 6/20 from 4.93 s to 0 s, and the regimes that
    really mattered do not move — 300/1000 stays at 4.93 s, 900/1000 at 25.61 s.

    Args:
        in_progress_count: activities awaiting computation (in flight + idle without a plan).
        population_size:   total number of simulated agents.
        k:                 convexity exponent (>1 — larger = later and more abrupt brake).
        cap:               ceiling in seconds (must stay under GAMA's HTTP read timeout).
        sans_frein:        below this number of pending activities, no brake. Equals the
                           simultaneous capacity of the gateway (`world.worker_concurrency`).
                           0 restores the behaviour from before 2026-09-22.
    """
    if population_size <= 0 or in_progress_count <= 0 or cap <= 0:
        return 0.0
    if in_progress_count <= sans_frein:
        return 0.0
    ratio = min(1.0, in_progress_count / population_size)
    return cap * ratio ** k


def update_drain_mode(
    drain_active: bool,
    backlog_ratio: float,
    trigger_ratio: float,
    release_ratio: float,
) -> bool:
    """Hysteresis of the queue drain mode.

    The mode engages when the queue fill level reaches ``trigger_ratio``
    and only releases when it falls back under ``release_ratio`` (release=0.2:
    we wait for the queue to be 80% emptied before handing control back to GAMA).
    A ``trigger_ratio`` <= 0 disables the mechanism.

    Args:
        drain_active:  current state of the drain mode.
        backlog_ratio: queue fill level (pending activities / population).
        trigger_ratio: engagement threshold (rising edge).
        release_ratio: release threshold (must be < trigger_ratio).
    """
    if trigger_ratio <= 0:
        return False
    if drain_active:
        return backlog_ratio > release_ratio
    return backlog_ratio >= trigger_ratio


def backlog_alarm_transition(
    alarm_active: bool,
    backlog_ratio: float,
    trigger_ratio: float,
    release_ratio: float,
    late_count: int,
    overdue_decisions: int,
) -> str:
    """Transition of the ``[ALARME] Backlog critique`` alarm (ticket 010, A2).

    A backlog above the threshold is a saturation only if itinerary
    decisions really suffer: departures served late
    (``late_count``) or plan/refill tasks whose sim deadline has passed
    (``overdue_decisions``). Otherwise — a queue dominated by STM reflections or by
    pre-plannings with a distant deadline, the case of the night drain — the
    operation is nominal: the caller logs the composition at INFO.

    Returns:
      - ``"fire"``    : rising edge of the alarm (ERROR);
      - ``"release"`` : alarm cleared (backlog back under ``release_ratio``);
      - ``"benign"``  : queue above the threshold but nothing urgent pending;
      - ``"none"``    : nothing to do.
    """
    if trigger_ratio <= 0:
        return "none"
    if alarm_active:
        return "release" if backlog_ratio < release_ratio else "none"
    if backlog_ratio < trigger_ratio:
        return "none"
    if late_count > 0 or overdue_decisions > 0:
        return "fire"
    return "benign"


# =============================================================================
# Deadline-driven predictive control (ticket 003)
# =============================================================================


class ThroughputEwma:
    """Completion throughput (tasks/s), exponentially smoothed.

    Exponentially decaying event-rate estimator: each
    completion injects a ``1/tau`` impulse into the accumulator, which decays
    as ``exp(-Δt/tau)``. For a completion process at rate λ, the expectation
    of ``rate()`` converges to λ (dR/dt = -R/tau + λ/tau → R = λ at equilibrium).

    Why EWMA (and not a 5-min sliding average): provider quotas are per
    minute; on exhaustion, the throughput collapses within seconds. A 5-min average
    would keep reporting a healthy throughput → pause triggered too late. The EWMA
    reacts in ~tau.

    Args:
        tau_s:       time constant (s) — larger = smoother, slower.
        floor_per_s: floor of the throughput returned by ``rate`` (never 0: avoids an
                     estimated T = ∞ in the feasibility test).
    """

    def __init__(self, tau_s: float, floor_per_s: float):
        self.tau_s = max(1e-6, float(tau_s))
        self.floor_per_s = max(0.0, float(floor_per_s))
        self._rate = 0.0
        self._last: float | None = None

    def _decay(self, now: float) -> None:
        if self._last is None:
            self._last = now
            return
        dt = now - self._last
        if dt > 0:
            self._rate *= math.exp(-dt / self.tau_s)
            self._last = now

    def mark_completion(self, now: float) -> None:
        """Record the completion of a task at instant ``now`` (monotonic/wall)."""
        self._decay(now)
        self._rate += 1.0 / self.tau_s

    def rate(self, now: float) -> float:
        """Current throughput (tasks/s), bounded by the floor (>= floor_per_s)."""
        self._decay(now)
        return max(self.floor_per_s, self._rate)


def time_ewma(prev_value: float | None, prev_time: float | None,
              sample: float, now: float, tau_s: float) -> float:
    """Time-based EWMA smoothing of a value sampled at irregular instants.

    Used for the pace ``R`` (sim s / real s, sampled at each /sync):
    ``α = 1 - exp(-Δt/tau)`` weights the sample according to the elapsed time,
    so that a widely spaced /sync counts more than a closely spaced one.

    ``prev_value`` None (first sample) returns ``sample`` directly.
    """
    if prev_value is None or prev_time is None:
        return sample
    tau = max(1e-6, tau_s)
    dt = max(0.0, now - prev_time)
    alpha = 1.0 - math.exp(-dt / tau)
    return prev_value + alpha * (sample - prev_value)


@dataclass(frozen=True)
class EdfFeasibility:
    """Result of the EDF feasibility test on a snapshot of the queue.

    hold:            True if at least one deadline risks being missed (with margin).
    t_estimate_s:    worst T_k = N / D — real time to resolve the whole queue.
    min_slack_sim_s: nearest deadline minus the current sim time (SIM seconds,
                     independent of R — serves as a "minimum margin" diagnostic).
    """
    hold: bool
    t_estimate_s: float
    min_slack_sim_s: float


def edf_feasibility(
    deadlines: list[float],
    now_sim: float,
    throughput_per_s: float,
    sim_ratio: float,
    margin: float,
) -> EdfFeasibility:
    """EDF feasibility test: does the queue meet its deadlines at the current throughput?

    With sorted deadlines ``d_1 ≤ … ≤ d_N`` (SIM time), the real throughput ``D``
    (tasks/s) and the pace ``R`` (sim s / real s):

        for k = 1..N:
            T_k     = k / D                   # real time to resolve the k most urgent
            slack_k = (d_k − now_sim) / R     # real time before d_k expires
        hold if ∃k: T_k · margin > slack_k

    Insight: holding the /sync freezes simulated time (it only advances if Python
    responds), so the pause really "buys" time with respect to the deadlines.

    Empty queue, D <= 0 or R <= 0 → no hold (estimated T undefined). O(N log N)
    because of the sort (N ≤ population, negligible at /sync).
    """
    if not deadlines or throughput_per_s <= 0 or sim_ratio <= 0:
        return EdfFeasibility(hold=False, t_estimate_s=0.0, min_slack_sim_s=0.0)

    ordered = sorted(deadlines)
    n = len(ordered)
    t_estimate_s = n / throughput_per_s
    min_slack_sim_s = ordered[0] - now_sim

    hold = False
    for k, d_k in enumerate(ordered, start=1):
        t_k = k / throughput_per_s
        slack_k = (d_k - now_sim) / sim_ratio
        if t_k * margin > slack_k:
            hold = True
            break

    return EdfFeasibility(hold=hold, t_estimate_s=t_estimate_s, min_slack_sim_s=min_slack_sim_s)


def edf_hold_needed(
    deadlines: list[float],
    now_sim: float,
    throughput_per_s: float,
    sim_ratio: float,
    margin: float,
) -> bool:
    """Boolean decision to hold the /sync (cf. ``edf_feasibility``)."""
    return edf_feasibility(deadlines, now_sim, throughput_per_s, sim_ratio, margin).hold


# =============================================================================
# Hold on imminent departure (2026-09-25)
# =============================================================================
#
# On 2026-09-24 (treated arm 2026-09-24_17_50, four agents), the decision for the 17:01 departure
# of agent 286921 stayed in flight for ~70 real seconds — an HTTP 503 "high demand" from
# Google. None of the brakes above saw it: the EDF queue no longer contains a task
# that a consumer has popped to execute it, and the queue ratio (1 out of 4) stayed under
# all thresholds. GAMA ran on until 18:15, the trip left 74 min late, the
# agent's day slipped by one day, and five trips are missing compared with the control.


@dataclass(frozen=True)
class PendingDeparture:
    """Departure decision requested and not yet returned (queued OR being executed).

    key:           unique identifier of the request (order of dispatch to the dispatcher).
    departure_sim: departure time of the trip, in SIMULATED time (same computation as the decision).
    kind:          "plan" | "refill".
    requested_sim: simulated time known at the moment of the request.
    requested_wall: real instant (monotonic) of the request — used to tell for how long
                   the decision has been awaited.
    """

    key: int
    person_id: str
    activity_id: str | None
    purpose: str | None
    departure_sim: float
    kind: str
    requested_sim: float
    requested_wall: float


def departures_at_risk(
    pending: Iterable[PendingDeparture],
    now_sim: float,
    lookahead_sim_s: float,
) -> list[PendingDeparture]:
    """Pending decisions whose departure falls before ``now_sim + lookahead_sim_s``.

    Sorted by increasing departure: the first one is the most urgent. A departure already passed
    (``departure_sim < now_sim``) is always at risk. ``lookahead_sim_s`` <= 0 only keeps
    departures already reached or passed.
    """
    horizon = now_sim + max(0.0, lookahead_sim_s)
    return sorted(
        (p for p in pending if p.departure_sim <= horizon),
        key=lambda p: (p.departure_sim, p.key),
    )


async def hold_while(
    predicate: Callable[[], bool],
    budget_s: float,
    poll_s: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> float:
    """Wait as long as ``predicate()`` is true, at most ``budget_s`` real seconds.

    Same pattern as the drain mode loop: resampling every ``poll_s``
    seconds, exit as soon as the condition drops. Returns the duration actually waited.
    ``budget_s`` <= 0 or condition false on entry: no wait.
    """
    start = clock()
    if budget_s <= 0 or not predicate():
        return 0.0
    while predicate():
        elapsed = clock() - start
        if elapsed >= budget_s:
            break
        await sleep(min(max(poll_s, 0.01), budget_s - elapsed))
    return clock() - start


def late_departure_alarm_transition(
    active: bool,
    late_count: int,
    now_sim: float,
    last_late_sim: float | None,
    rearm_sim_s: float,
) -> str:
    """Transition of the ``[ALARME] Départ servi en retard`` alarm (classic run).

    - ``"fire"``    : first departure served late while the alarm is at rest;
    - ``"release"`` : alarm active and no delay for ``rearm_sim_s`` simulated seconds;
    - ``"none"``    : nothing to do (including a new delay during an ongoing episode,
                      which is logged as WARNING without a new ERROR).
    """
    if late_count > 0:
        return "none" if active else "fire"
    if active and last_late_sim is not None and now_sim - last_late_sim >= rearm_sim_s:
        return "release"
    return "none"
