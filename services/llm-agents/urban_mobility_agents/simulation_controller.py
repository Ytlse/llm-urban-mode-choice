"""
This module defines the main simulation loop (SimulationLoopV1) for the scenario.

Its role is to orchestrate the course of the simulation over time. It handles:
- The synchronisation of the state of the world and of the agents.
- The triggering of the agents' actions (trip planning).
- The processing of the observations of the environment.
- The orchestration of the agents' reflection cycles (short- and long-term memory).

Planning model:
- O(1) fast path: at each arrival, the agent directly triggers its own planning
  through _try_schedule_person → _plan_one (under a concurrency semaphore).
- O(N) fallback path: a Worker scans the whole population every 30 s to
  catch up with the agents whose arrival observation would have been missed.
- Two WebSocket push points towards GAMA:
    Point 1: end of computation (the agent is IDLE when the trip is ready)
    Point 2: receipt of an arrival feedback (the next trip is already Planned)
"""

import asyncio
import contextlib
import dataclasses
import datetime
import heapq
import json
import os
import signal
import time
from collections import Counter as _Compteur
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backpressure import (
    PendingDeparture,
    ThroughputEwma,
    late_departure_alarm_transition,
)
from chaine_activites import activite_suivante
from experiences.decision import (  # ticket 035, spec 02 — la décision unique
    PREFIXE_RECALCUL_HORAIRE,
    PREFIXE_RECALCUL_OFFRE_JOUR,
    SOURCE_EN_VOL,
    SOURCE_HORS_JEU,
    SOURCE_LOCALE,
    Proposition,
    eligibilite,
    modes_vehicules_eligibles,
    plafonner,
    resumer_ecartees,
)
from helper import (
    format_sim_timing,
    humanize_date,
    humanize_duration,
    humanize_time,
    shift_weekend_departure_to_monday,
    to_24h_timestamp_full,
    to_timestamp_based_on_day,
)
from llm import evenements as evenements_module
from llm import foyer as foyer_module
from llm.axes import creneau_de, mode_canonique, normaliser_motif
from llm.gravite import gravite_deterministe, journal_des_composantes
from llm.journal_memoire import journal
from llm.memory import MemoryEntry, MemoryType
from llm_gateway.telemetry.alarms import fire_alarme
from loguru import logger
from models import Activity, BBox, Location, Person, PersonMove, TravelPlan
from prometheus_client import Counter, Gauge, Histogram
from settings import settings
from sim_clock import wall_clock
from text_helper import env_ob_to_text, parse_ob
from text_helper.models.arrival import retard_d_arrivee
from trip_helper import accidents as accidents_module
from trip_helper.base import TripHelper
from trip_helper.school_bus import (
    SCHOOL_BUS_CHOSEN,
    build_school_bus_option,
    is_school_bus_plan,
)
from urban_mobility_agents.agents.llm_agent import (
    ConsolidationMemoryUnavailable,
    Context,
    LlmAgent,
)
from urban_mobility_agents.candidats import (  # noqa: F401 — ré-exportés (ticket 035)
    _METRIC_MODE,
    GROUPE_RAIL,
    MODE_HIERARCHY,
    _primary_mode,
    _select_candidates,
    _selection_group,
    _unknown_metric_modes,
)
from urban_mobility_agents.core.scenario import Action, BaseScenario, Observation
from urban_mobility_agents.utils import filiation, mesures_jour, rejeu_decisions
from urban_mobility_agents.utils.ancre_run import ancrer, jours_ecoules
from urban_mobility_agents.utils.history_log import HistoryStreamLog
from urban_mobility_agents.utils.move_logger import GamaArrivalsLogger, MoveLogger
from urban_mobility_agents.utils.pipeline_logger import PipelineLogger
from urban_mobility_agents.utils.reprise import (
    degeler_si_depasse,
    ecrire_point,
    gel_actif,
)
from urban_mobility_agents.utils.weather_loader import day_weather_outlook, get_weather
from urban_mobility_agents.vehicle_chain import (  # noqa: F401 — ré-exportés (ticket 035, D1)
    _VEHICLE_MODES,
    DRIVING_AGE,
    RETURN_LOCK_MIN_DISTANCE_KM,
    VEHICLE_CHAIN,
    _can_drive,
    _chain_stake_modes,
    _is_car_passenger,
    _orphaned_vehicles,
    _owns_bike,
    _owns_car,
    _owns_vehicle,
    _park_vehicles,
    _road_distance_km,
    _same_place,
    _vehicle_available,
    _vehicle_mode,
    _vehicle_position,
    _vehicles_parked_at,
)
from utils import random_uuid
from world.population import WorldPopulation
from world.world_data import WorldModel

history_logger = HistoryStreamLog.get_instance()


def _format_cache_hit_rates() -> str:
    """Unified log line of the cache rate of the three routing/decision sources.

    OTP, OSMnx and LLM are independent process-wide caches. For each one we display
    hits/lookups (%); 'off' when the source has received no request (cache disabled or
    not yet queried)."""
    from trip_helper.cached_triphelper import get_otp_cache_stats
    from trip_helper.osmnx_direct import get_osmnx_cache_stats

    try:
        from llm.cache import get_llm_cache_stats, get_llm_miss_breakdown

        llm_stats = get_llm_cache_stats()
        llm_misses = get_llm_miss_breakdown()
    except Exception:
        llm_stats = (0, 0)
        llm_misses = {}

    # OTP: 'off' is ambiguous → state why the counter is at 0.
    otp_hits, otp_total = get_otp_cache_stats()
    if otp_total == 0:
        if not settings.gtfs.otp_cache_enabled:
            otp_str = "OTP off (désactivé)"
        else:
            otp_str = "OTP off (aucune requête)"
    else:
        otp_str = f"OTP {100 * otp_hits // otp_total}% ({otp_hits}/{otp_total})"

    def _fmt(name: str, stats: tuple[int, int]) -> str:
        hits, total = stats
        if total == 0:
            return f"{name} off"
        return f"{name} {100 * hits // total}% ({hits}/{total})"

    llm_str = _fmt("LLM", llm_stats)
    if llm_misses:
        _breakdown = ", ".join(
            f"{r}={n}" for r, n in sorted(llm_misses.items(), key=lambda kv: -kv[1])
        )
        llm_str += f" [miss: {_breakdown}]"

    return " · ".join(
        [
            otp_str,
            _fmt("OSMnx", get_osmnx_cache_stats()),
            llm_str,
        ]
    )


PROCESS_PERSON_CALLS = Counter(
    "gama_process_person_calls_total", "Total calls to process_person"
)
# All decisions combined (LLM, semantic cache, single choice, fallback): counted at the
# push of the trip to GAMA — unlike the llm_mode_by_* of the gateway, which only see
# the decisions that went through an actual LLM call.
TRIP_MODE_BY_PURPOSE = Counter(
    "trip_mode_by_purpose_total",
    "Trajets poussés vers GAMA par mode principal et motif d'activité",
    ["mode", "purpose"],
)
EVALUATE_PLAN_CALLS = Counter(
    "gama_evaluate_plan_calls_total", "Total calls to evaluate_and_choose_travel_plan"
)
ACTIONS_CREATED = Counter("gama_actions_created_total", "Total actions created")
PLANNING_LATE = Counter(
    "controller_planning_late_total",
    "Agents dont la date de départ était déjà passée lors de la planification",
)
LOST_ARRIVALS_RECOVERED = Counter(
    "controller_lost_arrivals_recovered_total",
    "Agents récupérés par le watchdog d'arrivée (move poussé jamais suivi d'observation d'arrivée)",
)
ITINERARY_100_COMPLETION = Gauge(
    "agent_itinerary_100_completion_seconds",
    "Durée réelle (secondes) pour traiter 100 itinéraires réussis consécutifs",
)
BOOTSTRAP_DURATION = Gauge(
    "agent_bootstrap_duration_seconds",
    "Durée réelle (secondes) du bootstrap_all_agents (calcul initial des itinéraires au /init)",
)

# Real-time detail of the progress of phase 4 (itinerary bootstrap), for the cockpit.
BOOTSTRAP_ACTIVE = Gauge(
    "agent_bootstrap_active", "Bootstrap en cours (1) ou terminé/inactif (0)"
)
BOOTSTRAP_TOTAL = Gauge(
    "agent_bootstrap_total", "Nombre d'agents éligibles à planifier au bootstrap"
)
BOOTSTRAP_COMPLETED = Gauge(
    "agent_bootstrap_completed",
    "Agents dont le premier itinéraire est calculé (vague 1)",
)
BOOTSTRAP_PROGRESS = Gauge(
    "agent_bootstrap_progress_ratio",
    "Progression de la vague 1 du bootstrap (0..1 = completed/total)",
)
BOOTSTRAP_CACHE_HITS = Gauge(
    "agent_bootstrap_cache_hits",
    "Premiers itinéraires servis depuis le cache LLM pendant le bootstrap",
)
BOOTSTRAP_CACHE_MISSES = Gauge(
    "agent_bootstrap_cache_misses",
    "Premiers itinéraires calculés via LLM (cache miss) pendant le bootstrap",
)
BOOTSTRAP_WAVE = Gauge(
    "agent_bootstrap_wave",
    "Vague d'anticipation courante (1 = premier itinéraire, ≥2 = act[N+k] pré-calculés)",
)
BOOTSTRAP_FUTURE_MOVES = Gauge(
    "agent_bootstrap_future_moves",
    "Trajets futurs (act[N+k]) pré-cachés cumulés pendant le bootstrap",
)
# Détail par vague (1 = premiers itinéraires, ≥2 = anticipation act[N+k]).
# status : planned = candidats de la vague, done = traités (succès ou non),
#          ok = itinéraires obtenus, cache_hit/cache_miss = cache LLM (agents LLM seulement).
BOOTSTRAP_WAVE_MOVES = Gauge(
    "agent_bootstrap_wave_moves", "Trajets par vague du bootstrap", ["wave", "status"]
)
_WAVE_COUNTED_STATUSES = ("done", "ok", "cache_hit", "cache_miss")


def _wave_metrics(wave: int, planned: int) -> dict:
    """Initialises and returns the per-status counters of a bootstrap wave."""
    BOOTSTRAP_WAVE_MOVES.labels(wave=str(wave), status="planned").set(planned)
    counters = {
        s: BOOTSTRAP_WAVE_MOVES.labels(wave=str(wave), status=s)
        for s in _WAVE_COUNTED_STATUSES
    }
    for c in counters.values():
        c.set(0)
    return counters


# Outcome of the mobility decision per planned activity. outcome ∈
#   llm           : plan chosen by the LLM
#   llm_fallback  : LLM without answer (saturation/timeout) → default index (degraded activity)
#   single        : a single itinerary → automatic selection
#   no_solution   : no transport mode connects the OD
#   no_move       : already at destination (no trip)
# phase ∈ {bootstrap, live}: the bootstrap precomputes the itineraries at /init (the agent
# has not « missed » anything yet — it will leave on the fallback anyway). The cockpit only counts
# the live phase as « activities missed for lack of LLM » (cf. panels ③).
# Makes it possible to follow in real time the activities « missed for lack of an LLM answer » (llm_fallback).
ACTIVITY_DECISIONS = Counter(
    "agent_activity_decisions_total",
    "Décisions de mobilité par activité planifiée, ventilées par issue (outcome) et phase",
    ["outcome", "phase"],
)
# selection_method (texte du move-log) → outcome (label métrique, faible cardinalité)
_SELECTION_OUTCOME = {
    "LLM": "llm",
    "LLM Error (Default index)": "llm_fallback",
    "Un seul itinéraire disponible": "single",
    "Pas de solution de déplacement": "no_solution",
    "Pas de déplacement (même localisation)": "no_move",
}

# Métriques goulots d'étranglement
AGENT_SCHEDULING_LAG = Histogram(
    "agent_scheduling_lag_seconds",
    "δ entre scheduled_start_time et envoi de l'action à GAMA (positif = en retard)",
    buckets=[10, 60, 300, 1800, float("inf")],
)
CONTROLLER_SCHEDULING_IN_PROGRESS = Gauge(
    "controller_scheduling_in_progress",
    "Nombre d'agents en attente de décision LLM (scheduling_in_progress=True)",
)
AGENT_LATE_DEPARTURE = Histogram(
    "agent_late_departure_seconds",
    "Retard des agents (sim_time - scheduled_start_time) lors du skip d'activité",
    buckets=[60, 300, 1800, 7200, float("inf")],
)

# Departure punctuality (dashboard 07 · Mobility business). A departure is « on time »
# if the action leaves for GAMA at most LATE_DEPARTURE_TOLERANCE_S after the planned time
# (same lag as AGENT_SCHEDULING_LAG). Live phase only: the bootstrap precomputes
# at /init and does not measure an actual departure.
LATE_DEPARTURE_TOLERANCE_S = 60
DEPARTURE_PUNCTUALITY = Counter(
    "agent_departures_punctuality_total",
    "Départs poussés vers GAMA (phase live) par ponctualité : on_time (lag ≤ 60 s) ou late",
    ["status"],
)
DEPARTURE_DELAY = Histogram(
    "agent_departure_delay_seconds",
    "Retard (s) des seuls départs en retard (lag > 60 s) — sum/count = retard moyen",
    buckets=[120, 300, 900, 1800, 3600, float("inf")],
)
DEPARTURE_DELAY_MAX = Gauge(
    "agent_departure_delay_max_seconds",
    "Retard maximal (s) observé sur un départ depuis le démarrage du contrôleur",
)
for _status in ("on_time", "late"):
    DEPARTURE_PUNCTUALITY.labels(status=_status)


_POPULATION_CHECKPOINT_HOUR = 2 * 3600  # 2:00 AM simulation time

# Simulated time of the resume point (ticket 075). 3 a.m., and not 2 a.m. like the population
# checkpoint: the consolidation floor falls at 10 p.m. and its reflections drain all
# night (ticket 010). Taking the point at 2 a.m. would take it in the middle of this drain, hence on a
# half-written memory — exactly what a resume point must never capture.
_REPRISE_CHECKPOINT_HOUR = 3 * 3600


# Ticket 105 — the lock that arms the experiment stops. `EXPERIMENT_HIBERNATE_ON_QUOTA` is
# the old name (ticket 077, when the quota was the only reason to stop); it is still accepted, because
# it is written in `infra/docker-compose.yml` and in campaign scripts already launched.
# `run_sequential_cohort.py` sets this lock on EVERY campaign: the safeguard is therefore armed
# by default in experiments and silent in an ordinary run, with no lever to think about.
_VERROUS_ARRET_EXPERIENCE = ("EXPERIMENT_STOP_ON_FALLBACK", "EXPERIMENT_HIBERNATE_ON_QUOTA")


def _arret_sur_repli_arme() -> bool:
    """Must the run stop rather than serve a default decision?"""
    return any(os.getenv(nom) == "1" for nom in _VERROUS_ARRET_EXPERIENCE)


def _depart_du_trajet(activite: Activity, base_ts: int) -> tuple[int, int]:
    """Departure time towards `activite` planned from `base_ts`, and the planning delay.

    SINGLE computation, shared by the decision (`_compute_move_for_activity`) and by the registry
    of pending departure decisions: the /sync hold must aim at the time the decision
    will compute, not an approximation. Wrap to D+1 included (24 h time already passed at `base_ts`
    ⇒ next day); the weekend postponement is left to the caller (it logs).
    """
    target_24h = (
        activite.scheduled_start_time
        if activite.scheduled_start_time is not None
        else activite.end_time
    )
    depart = to_timestamp_based_on_day(target_24h_timestamp=target_24h, based_on=base_ts)
    retard_planification = max(0, base_ts - depart)
    if depart < base_ts:
        depart += 86400  # activité du lendemain (bouclage J+1)
    return depart, retard_planification


def depart_prevu(activite: Activity, base_ts: int) -> int:
    """Departure time the decision will keep, weekend postponement included (no log)."""
    depart, _ = _depart_du_trajet(activite, base_ts)
    if settings.agent.no_weekend_departures:
        depart = shift_weekend_departure_to_monday(depart)
    return depart


def _depart_du_move(move: PersonMove) -> int:
    """Departure time of a computed trip — the one `moves.csv` records (« Heure de départ »)."""
    if move.plan is not None and move.plan.start_time:
        return int(move.plan.start_time // 1000)
    return int(move.expected_arrive_at)


def _duree_en_minutes(secondes: float) -> str:
    """« 74 min » — `humanize_duration` renders « 1 hour » for 74 min (it cuts at the first
    « and »): a delay must read to the minute."""
    secondes = int(secondes)
    return f"{secondes // 60} min" if secondes >= 60 else f"{secondes} s"


@dataclass(frozen=True)
class DepartServiEnRetard:
    """Departure decision returned AFTER the departure time (2026-09-25).

    `servi_sim` is the simulated time known to the controller when the decision comes back (last
    /sync, advanced by the observations): a LOWER BOUND of GAMA's time, which can be up to
    one /sync interval ahead. The measured delay can therefore only underestimate —
    never a false alarm.
    """

    person_id: str
    activity_id: str | None
    purpose: str | None
    depart_sim: int
    servi_sim: int
    kind: str

    @property
    def retard_s(self) -> int:
        return self.servi_sim - self.depart_sim


def _next_checkpoint_ts(
    after_ts: int, hour_24h: int = _POPULATION_CHECKPOINT_HOUR
) -> int:
    """Return the absolute timestamp of the next 2 AM occurrence after after_ts."""
    day_start = (after_ts // 86400) * 86400
    candidate = day_start + hour_24h
    if candidate <= after_ts:
        candidate += 86400
    return candidate


def _estimate_fallback_duration(origin, destination) -> int:
    """Estimate travel time in seconds from crow-flies distance at 30 km/h with 1.3 detour factor."""
    road_distance_km = _road_distance_km(origin, destination)
    if road_distance_km is None:
        return 30 * 60
    speed_ms = 30_000 / 3600  # 30 km/h
    return max(5 * 60, int(road_distance_km * 1000 / speed_ms))


def _record_trip_mode(move: PersonMove, activity: Activity | None) -> None:
    """Feeds trip_mode_by_purpose_total for a trip actually pushed to GAMA."""
    if move.plan is None:
        return
    purpose = (
        (activity.purpose if activity else None) or move.purpose or "unknown"
    ).lower()
    TRIP_MODE_BY_PURPOSE.labels(mode=_primary_mode(move.plan), purpose=purpose).inc()


def _agenda_lines(
    person: Person, next_activity: Activity, departure_time: int
) -> list[str]:
    """Remaining trips of the day AFTER the current trip (rolling agenda).

    One line per located future activity: planned time, purpose, distance
    estimated from the previous stop (as the crow flies × 1.3, the convention of
    `_road_distance_km`), and the forecast weather when it differs from that of the
    departure. Stops at the wrap to D+1: the agenda describes THE day, not the
    next one. Never shows past trips — the factual context of a
    trip must not depend on the history (stability of the cache key).
    """
    acts = person.identity.activities or []
    idx = next((i for i, a in enumerate(acts) if a.id == next_activity.id), None)
    if idx is None:
        return []
    now_weather = get_weather(departure_time)
    now_label = now_weather["weather_label"] if now_weather else None
    lines: list[str] = []
    prev = next_activity
    for act in acts[idx + 1 :]:
        if act.location is None:
            prev = act
            continue
        target_24h = (
            act.scheduled_start_time
            if act.scheduled_start_time is not None
            else act.end_time
        )
        ts = to_timestamp_based_on_day(int(target_24h), departure_time)
        if ts <= departure_time:  # 24h time already passed ⇒ activity of the next day
            break
        line = f"{humanize_time(ts)} → {act.purpose}"
        km = _road_distance_km(prev.location, act.location)
        if km:
            line += f" (≈{km:.1f} km)"
        w = get_weather(ts)
        if w and now_label and w["weather_label"] != now_label:
            # English since ticket 074, like everything that reaches the model. This line
            # is the NINTH surface of the switch, and it was not in the inventory:
            # it is only rendered for agents with a vehicle to chain AND when the
            # forecast weather differs from that of the departure — rare enough for a sample
            # check to miss it, frequent enough for it to go into production.
            line += f" — {w['weather_label'].lower()} expected"
        lines.append(line)
        prev = act
    return lines


def _build_anticipation(
    person: Person,
    next_activity: Activity,
    departure_time: int,
) -> dict | None:
    """Anticipation context of the prompt (ticket 014) — None if nothing to show.

    `outlook` is produced for all agents; `agenda` only for those
    who have a vehicle to chain. The position of the vehicles is NO LONGER stated
    in the prompt: the wording (« your bike is with you ») acted
    as an invitation and inflated the bike share by +5.5 points (EMC² measurement,
    run 2026-08-19_13_17). The chain rule now lives in the system
    prompt (variant `expert_m4`), and the availability information remains
    carried by the option set through the locks. `signature` is the
    deterministic concatenation of the texts: it enters the key of the decision
    cache. `trace` feeds the « Anticipation » column of moves.csv
    ("agenda" = rolling agenda present, "meteo" = weather of the day only).
    """
    outlook = day_weather_outlook(departure_time)
    agenda: list[str] = []
    if _chain_stake_modes(person):
        agenda = _agenda_lines(person, next_activity, departure_time)
    if not outlook and not agenda:
        return None
    return {
        "outlook": outlook,
        "agenda": agenda,
        "signature": " | ".join([outlook or "", *agenda]),
        "trace": "agenda" if agenda else "meteo",
    }


def _resume_sources(sources: dict, plans: list) -> str:
    """« enregistree:5,recalculee:2 » — the breakdown of the presented proposals by source (G6)."""
    compte: _Compteur = _Compteur(
        sources.get(id(pl), SOURCE_EN_VOL).split(":")[0] for pl in plans
    )
    return ",".join(f"{k}:{v}" for k, v in sorted(compte.items()))


@dataclass
class _EdfJob:
    """Planning task in the EDF (Earliest Deadline First) queue.

    deadline_sim : deadline in SIMULATED time (departure time of the trip). The closer
                   it is, the higher the priority of the task. Sentinel 0 for a
                   push (already computed, only sending remains → always a priority).
    kind         : "plan" | "refill" | "push" (metrics + feasibility snapshot).
    make_coro    : builds the coroutine to run (lazy: nothing is created if
                   the queue is emptied before consumption → no "coroutine never awaited").
    person_id    : for the logs.
    """

    deadline_sim: float
    kind: str
    make_coro: Callable[[], Coroutine]
    person_id: str


class SimulationLoopV1(BaseScenario):
    MAX_ADJUST_START_TIME = 15 * 60  # 15 minutes

    # Interval of the fallback scan (seconds). This scan catches up with the agents
    # whose arrival observation would have been missed. The normal path is
    # O(1): each agent schedules itself at its arrival.
    _WORKER_SCAN_INTERVAL = 30.0

    def __init__(
        self,
        world_model: "WorldModel",
        trip_helper: "TripHelper" = None,
        agent: Optional["LlmAgent"] = None,
    ):
        self.MAX_ADJUST_START_TIME = (
            settings.agent.max_reschedule_amount or self.MAX_ADJUST_START_TIME
        )
        self._messages = []
        # Departures served late since the last /sync (reset by sync()). Before
        # 2026-09-25, a simple counter fed by `expected_arrive_at < timestamp`, where
        # `timestamp` is the planning BASE (end of the current activity): the condition
        # could not be true, and the 17:01 departure served at 18:15 on 2026-09-24 left
        # `late_since_last_sync=0` throughout.
        self._departs_en_retard: list[DepartServiEnRetard] = []
        self._departs_en_retard_total = 0
        # Ordinary run: rising-edge alarm, re-armed after one simulated hour without delay.
        self._alarme_retard_active = False
        self._dernier_retard_sim: float | None = None
        self._episode_retards = 0
        self._episode_retard_max_s = 0
        # Registry of the departure decisions REQUESTED and not yet RETURNED — in the EDF queue as
        # well as running. The queue alone is not enough: a consumer pops the task
        # before calling the model, and on 2026-09-24 the decision awaited ~70 s (HTTP 503
        # Google) was no longer visible to any brake. Read by the /sync hold (experiments).
        self._decisions_depart: dict[int, PendingDeparture] = {}
        self._decisions_depart_seq = 0
        # /sync holds on imminent departure, for the daily summary.
        self._retenues_depart = 0
        self._retenues_depart_s = 0.0
        self._max_departure_delay_s = (
            0.0  # pire retard observé, alimente DEPARTURE_DELAY_MAX
        )
        self._itinerary_success_count = 0
        self._itinerary_window_start = time.monotonic()
        # True during bootstrap_all_agents: tells the precomputation decisions (/init)
        # from the live decisions (simulation running) apart for the ACTIVITY_DECISIONS metric.
        self._in_bootstrap = False
        # Ticket 077, batch C — mode kept per activity, for the habit journal.
        # It can NOT be reread from the short-term memory buffer at arrival: between the
        # decision and the arrival, a consolidation may have consumed the entry (`remove_batch`),
        # and the trip then disappeared from the habits. Measured on the run of 15/09 at 10:03:
        # 1 to 2 trips logged per agent for about forty arrivals, and the
        # « Mes habitudes » block missing from ALL decision prompts.
        # The key is (agent, activity): it is rewritten from one simulated day to the next, so the
        # table stays bounded by the number of activities of a day.
        self._mode_par_activite: dict[tuple[str, str], str] = {}
        self.model = world_model
        self.trip_helper = trip_helper
        self.agent = agent
        self.next_self_reflection_at = None
        self._stm_reflecting: set[str] = (
            set()
        )  # person_ids with an in-flight STM reflection task
        # EDF deadline (SIM time) of the pending STM reflection, per person_id.
        # Set at the first trigger and KEPT across retries: a
        # gateway failure leaves the STM entries in place, the next sync resubmits
        # with the original deadline → the EDF priority rises at each retry.
        self._stm_reflect_due: dict[str, float] = {}
        # Daily floor (ticket 048): simulated day (ordinal) on which the floor has
        # already submitted a reflection for this agent. It only goes once per day.
        self._stm_floor_day: dict[str, int] = {}
        self._stm_overdue_alarm_on = (
            False  # front montant de l'alarme réflexions en retard
        )
        # Ticket V5 — Concaténation des étapes d'un trajet en un événement unique à l'arrivée
        self._etapes_trajet_en_cours: dict[str, list[dict[str, Any]]] = {}
        # Ticket 071, batch 1 — triggers BY BREAK, to check that this regime stays
        # exceptional. The threshold Θ = 0.7 was set without measurement: no run has ever computed
        # the deterministic severity. The criterion is « less than one trigger per agent and per
        # week »; above it, Θ must go up, otherwise the night consolidation — the regime
        # chosen — becomes the exception in turn and the inference cost follows.
        self._ruptures: list[tuple[float, int]] = []  # (timestamp sim, nombre)
        self._rupture_alarme_on = False  # front montant
        # Declaration at startup of what actually feeds the severity (batch 1). A
        # component without a source contributes zero, and zero is the value of a perfect trip:
        # without this line, the severity would be understated without any symptom appearing.
        journal_des_composantes()
        # Chain consistency of vehicles: denominator (planned returns home)
        # and numerator (returns leaving a vehicle elsewhere) of the orphan rate, plus
        # the rising edge of the associated alarm. A high rate signals that the return
        # lock is not enough and that the catch-up at home carries a notable bias.
        self._vehicle_home_returns = 0
        self._vehicle_orphan_returns = 0
        self._vehicle_orphan_alarm_on = False
        # Recorded trip set (ticket 035, spec 04): served instead of the engines
        # when it covers the trip and the actual time stays within the tolerance of the mode.
        # `None` = historical behaviour (computed on the fly). The counters feed
        # the run's `jeu_stats.json` and the « Jeu enregistré » section of `make report` (G14).
        self.jeu = None
        self.jeu_tolerances: dict = {}
        self._jeu_stats: _Compteur = _Compteur()
        self._jeu_propositions_par_source: _Compteur = _Compteur()
        self._jeu_alarme_sterile_on = False
        self._jeu_stats_depuis_ecriture = 0
        # Calendar of the run (study-area audit, axis A6). The survey only counts
        # weekdays; a weekend departure postponed to Monday PILES UP on the Monday
        # and makes it an atypical day. The postponement is logged at info level every time;
        # the alarm, for its part, only goes off at the first one — that is the event that counts.
        self._weekend_shift_alarm_on = False

        # Worker
        # _worker_sem  : limits the LLM+OTP concurrency (initialised in start_worker)
        # _worker_in_progress : activities whose computation is running (under semaphore or in flight)
        # _current_sim_timestamp : last known timestamp of the simulation
        self._worker_sem: asyncio.Semaphore | None = None
        self._worker_loop_task: asyncio.Task | None = None
        # Fire-and-forget tasks of the scenario (planning, refill, reflections…):
        # cancelled as a block by stop_worker() so that a replaced scenario no longer pushes
        # stale actions to the new simulation.
        self._inflight_tasks: set[asyncio.Task] = set()
        # --- EDF dispatcher (ticket 003) ---
        # Priority queue (min-heap) sorted by (deadline_sim, seq): the planning
        # tasks are served by increasing deadline, not in order of arrival.
        # Consumed by worker_concurrency tasks (_edf_consumer) which REPLACE the
        # semaphore as the concurrency limit. The semaphore is still used by the
        # bootstrap (outside the dispatcher). Initialised in start_worker().
        self._edf_heap: list[tuple[float, int, _EdfJob]] = []
        self._edf_seq: int = 0
        self._edf_event: asyncio.Event | None = None
        self._edf_consumers: list[asyncio.Task] = []
        self._edf_active_jobs = 0
        self._prefixe_job_error: str | None = None
        self._prefixe_sync_applique_ts: int | None = None
        self._prefixe_sync_confirme_ts: int | None = None
        self._prefixe_attente_depuis: float | None = None
        # EWMA of the completion rate (tasks/s): measured at the end of _plan_one /
        # _precompute_one (the full OTP+LLM pipeline is the unit that drains the queue).
        self._throughput: ThroughputEwma | None = None
        self._worker_in_progress: int = 0
        self._current_sim_timestamp: int = 0
        # The client timestamps each submission with the controller's clock, including STM/LTM.
        self.agent.llm_client._horloge_simulee = lambda: self._current_sim_timestamp
        self._push_fn: Callable[[Action], Coroutine] | None = None
        self._next_population_checkpoint_at: int | None = None
        # Ticket 075 — prochain point de reprise (3 h simulées), posé au premier sync.
        self._next_reprise_at: int | None = None
        # Ticket 077 — enquêtes d'affinité modale déclarée (jours simulés déjà sondés)
        self._enquetes_menees: set[int] = set()
        # Ticket 105 — CONSECUTIVE fallbacks (index 0 served instead of a decision of the model).
        # Reset by any decision actually taken: what we want to catch is a
        # series, not an isolated incident that the retries have already absorbed.
        self._replis_consecutifs: int = 0
        # Rising edge of the hibernation: on 2026-09-24, eight consumers in fallback each
        # triggered it in turn, and their concurrent resume points trod on
        # each other (Errno 39). Only the first call stops; the following ones are counted.
        self._hibernation_declenchee: bool = False
        self._hibernations_ignorees: int = 0
        # Suivi temporel : heure réelle franchie à chaque tranche de 24h de temps simulé
        self._sim_start_ts: int | None = None  # premier timestamp simulé observé
        self._sim_real_start: float | None = (
            None  # heure réelle (monotonic) correspondante
        )
        self._next_day_log_at: int | None = None  # prochaine borne 24h à logger
        # Post-pause drain (ticket 010, A3): real time of the last /sync and
        # last count of reflections logged during the silent drain.
        self._last_sync_wall: float | None = None
        self._post_pause_drain_seen: int = 0

        self._brancher_evenements()

        if settings.agent.reschedule_activity__version == 2:
            self.reschedule_amount_function = self.reschedule_amount_v2
            logger.info("Using reschedule activity function version v2")
        else:
            self.reschedule_amount_function = self.reschedule_amount
            logger.info("Using reschedule activity function version v1")

    def _brancher_evenements(self) -> None:
        """Ticket 111 — readers drawn at load time, and relay producer plugged in.

        Before this ticket, the readers were only known at the first `/sync`: a reading on
        day 1 would have been invisible to the bootstrap decisions. The relay producer lives
        here because the controller is the only one holding the population, the LLM client and the
        identities.
        """
        registre = evenements_module.registre()
        if registre is None or registre.evenement.moment != "reveil":
            return
        try:
            registre.lecteurs(self.population.get_people_list())
        except Exception as err:  # noqa: BLE001
            logger.error(
                f"[ALARME] [evenements] lecteurs non tirés au chargement ({err}) : les lignes "
                f"de service ne seront PAS servies."
            )
        if registre.evenement.relais is not None and self.agent is not None:
            registre.brancher_producteur_relais(self._produire_relais)

    async def _produire_relais(self, lecteur_id: str, household_id: str, jour: int):
        """One `evenement_relais` call for this household. Raises `RelaisRefuse` without fallback."""
        from llm import foyer as _foyer
        from llm.evenements import relais as relais_module

        registre = evenements_module.registre()
        lecteur = self.population.people.get(str(lecteur_id))
        if lecteur is None or registre is None:
            raise relais_module._refuser(
                household_id, registre.evenement.evenement_id if registre else "?",
                f"lecteur {lecteur_id} absent de la population",
            )
        lecteurs = registre._lecteurs or {}
        membres = [
            self.population.people[m.person_id]
            for m in _foyer.autres_membres(str(lecteur_id))
            if m.person_id in self.population.people and m.person_id not in lecteurs
        ]
        if not membres:
            # Nothing to pass on is not a refusal: no call, no alarm. The empty relay
            # is traced like the others, so that the resume does not request it again.
            logger.info(
                f"[evenements] relais du foyer {household_id} : le lecteur {lecteur_id} n'a aucun "
                f"autre membre mobile — rien à transmettre, aucun appel."
            )
            return relais_module.RelaisFoyer(
                household_id=str(household_id), lecteur_id=str(lecteur_id),
                lecteur_prenom=relais_module.prenom_de(lecteur),
                evenement_id=registre.evenement.evenement_id, jour_run=int(jour),
            )
        fiches = [
            relais_module.fiche_membre(p, self._modes_habituels(p)) for p in membres
        ]
        e = registre.evenement
        return await relais_module.produire(
            self.agent.llm_client,
            lecteur,
            self.agent.get_person_identity_description(lecteur),
            fiches,
            e.texte_cite.servi if e.texte_cite else "",
            e.evenement_id,
            jour,
            household_id,
            parole_obligatoire=e.relais.parole_obligatoire,
        )

    def _modes_habituels(self, personne) -> list[str]:
        """The modes this member has taken, from most to least frequent, read from their journal."""
        ltm = getattr(self.agent, "long_term_memory", None)
        if ltm is None:
            return []
        try:
            from llm.evenements.relais import modes_habituels

            return modes_habituels(ltm.journal_trajets(str(personne.person_id)))
        except Exception:  # noqa: BLE001 — an incomplete profile is better than a lost relay
            return []

    # -------------------------------------------------------------------------
    # BaseScenario — interface publique Worker
    # -------------------------------------------------------------------------

    def set_push_fn(self, fn: Callable[[Action], Coroutine]) -> None:
        """Inject the coroutine used for direct WebSocket push to GAMA."""
        self._push_fn = fn

    @property
    def worker_in_progress_count(self) -> int:
        """Activities whose computation is running or queued on the semaphore."""
        return self._worker_in_progress

    @property
    def late_since_last_sync(self) -> int:
        """Departure decisions returned AFTER the departure time since the last /sync (reset
        to 0 by sync()). Read by the controller before sync() to feed the counter
        controller_deadline_misses_total and the backlog alarm."""
        return len(self._departs_en_retard)

    @property
    def arret_experience_arme(self) -> bool:
        """Experiment lock (`EXPERIMENT_STOP_ON_FALLBACK=1`): hold on imminent departure
        and stop rather than delay. False in an ordinary run."""
        return _arret_sur_repli_arme()

    def pending_departures(self) -> list[PendingDeparture]:
        """Departure decisions requested and not yet returned (queued or running)."""
        return list(self._decisions_depart.values())

    def _prefixe_pending(self) -> dict[str, int]:
        """State of all the computations that can change a choice or the memory."""
        return {
            "edf_queue": len(self._edf_heap),
            "edf_active": self._edf_active_jobs,
            "reflections": len(self._stm_reflecting),
            "departures": len(self._decisions_depart),
            "workers": self._worker_in_progress,
            "spawned": sum(not t.done() for t in self._inflight_tasks),
        }

    async def attendre_prefixe_quiescent(self, deadline: float, timestamp: int) -> bool:
        """Holds the simulated step until the full drain, or invalidates the run.

        The cache makes the control faster than the treated arm. A simple gradual hold of
        /sync therefore leaves a reflection finished in one arm and in flight in the other.
        The experimental mode requires quiescence in BOTH arms, with no cap that
        would silently release a task still active.
        """
        debut = time.monotonic()
        while True:
            from trip_helper.strict_replay import echec_route_strict

            pending = self._prefixe_pending()
            if (
                self._prefixe_job_error
                or self.agent.llm_client.strict_replay_failure
                or echec_route_strict()
            ):
                cause = (
                    self._prefixe_job_error
                    or self.agent.llm_client.strict_replay_failure
                    or echec_route_strict()
                )
            elif not any(pending.values()):
                logger.info(
                    "[prefixe_commun] pas={} drainé en {:.3f}s",
                    timestamp, time.monotonic() - debut,
                )
                self._prefixe_attente_depuis = None
                return True
            elif time.monotonic() >= deadline:
                if self._prefixe_attente_depuis is None:
                    self._prefixe_attente_depuis = debut
                if (
                    time.monotonic() - self._prefixe_attente_depuis
                    < settings.world.prefixe_commun_stall_timeout_s
                ):
                    logger.info(
                        "[prefixe_commun] pas={} encore en cours après {:.1f}s : {}",
                        timestamp, time.monotonic() - debut, pending,
                    )
                    return False
                cause = f"attente totale épuisée; tâches restantes={pending}"
            else:
                await asyncio.sleep(0.05)
                continue
            logger.error("[ALARME] [prefixe_commun] pas={} interrompu : {}", timestamp, cause)
            await self._declencher_hibernation_propre(
                None, "prefixe_commun", motif="prefixe_commun",
                details={"timestamp": timestamp, "cause": cause, "pending": pending},
                genre=(
                    "rejeu_obligatoire_absent"
                    if self.agent.llm_client.strict_replay_failure else "prefixe_bloque"
                ),
            )
            raise RuntimeError(f"Préfixe commun invalide : {cause}")

    async def synchroniser_prefixe(
        self, timestamp: int, deadline: float,
        *, t_sync: float | None = None, t_parse: float | None = None,
    ) -> bool:
        """Processes each timestamp only once, even after several pending acknowledgements."""
        if self._prefixe_sync_applique_ts is not None and timestamp < self._prefixe_sync_applique_ts:
            raise ValueError(
                f"/sync périmé en préfixe commun : {timestamp} < {self._prefixe_sync_applique_ts}"
            )
        if self._prefixe_sync_applique_ts != timestamp:
            if not await self.attendre_prefixe_quiescent(deadline, timestamp):
                return False
            try:
                await self.sync(timestamp, _t_sync=t_sync, _t_parse=t_parse)
            except Exception as exc:
                await self._declencher_hibernation_propre(
                    None, "prefixe_commun", motif="prefixe_commun",
                    details={"timestamp": timestamp, "cause": repr(exc)},
                )
                raise
            self._prefixe_sync_applique_ts = timestamp
        if not await self.attendre_prefixe_quiescent(deadline, timestamp):
            return False
        self._prefixe_sync_confirme_ts = timestamp
        return True

    def noter_retenue_depart(self, duree_s: float) -> None:
        """Counts a /sync hold on imminent departure (daily summary)."""
        self._retenues_depart += 1
        self._retenues_depart_s += float(duree_s)

    @property
    def activities_to_compute_count(self) -> int:
        """Activities to compute: in flight + Idle agents without a plan (live count).

        The Idle-without-plan count is live and not a snapshot of the last sync:
        the drain mode of /sync resamples this gauge every second while
        it holds the response, and must see the stack go down as pushes happen to
        hand control back to GAMA as soon as the release threshold is reached.
        """
        idle_unplanned = sum(
            1
            for p in self.population.get_people_list()
            if p.state.heading_to is None
            and not p.state.scheduling_in_progress
            and p.state.next_planned_move is None
        )
        return self._worker_in_progress + idle_unplanned

    def count_stuck_agents(self, current_ts: int, threshold_seconds: float) -> int:
        """Number of agents without a successful planning for more than `threshold_seconds`
        of simulated time.

        Single choke point (called at each /sync): an agent is "healthy" if it has a current
        plan, precomputed plans in reserve, or a computation in flight — in which case we
        refresh its last-success timestamp. Otherwise, it is counted as stuck
        as soon as the gap to the last successful plan exceeds the threshold. Agents never
        planned yet are primed at `current_ts` (not stuck until the threshold has
        elapsed since their first observation)."""
        stuck = 0
        for person in self.population.get_people_list():
            st = person.state
            healthy = (
                st.next_planned_move is not None
                or len(st.precomputed_moves) > 0
                or st.scheduling_in_progress
            )
            if healthy or st.last_successful_plan_sim_ts is None:
                st.last_successful_plan_sim_ts = current_ts
            elif current_ts - st.last_successful_plan_sim_ts > threshold_seconds:
                stuck += 1
        return stuck

    async def start_worker(self) -> None:
        """Start the fallback Worker (periodic scan), the EDF dispatcher and the semaphore."""
        concurrency = settings.world.worker_concurrency
        # The semaphore remains the concurrency limit of the BOOTSTRAP (outside the dispatcher,
        # awaited by /init). In steady state, the limit is the number of EDF consumers.
        self._worker_sem = asyncio.Semaphore(concurrency)
        self._throughput = ThroughputEwma(
            tau_s=settings.world.throughput_ewma_tau_s,
            floor_per_s=settings.world.throughput_floor_per_s,
        )
        # Dispatcher EDF : file + N consommateurs. false = spawn direct FIFO historique.
        self._edf_heap = []
        self._edf_seq = 0
        self._edf_event = asyncio.Event()
        self._edf_consumers = []
        self._edf_active_jobs = 0
        self._prefixe_job_error = None
        self._decisions_depart.clear()
        if settings.world.edf_enabled:
            self._edf_consumers = [
                asyncio.create_task(self._edf_consumer(i)) for i in range(concurrency)
            ]
        # The task is kept so that it can be cancelled through stop_worker() on
        # a scenario replacement (successive /init or /test/init).
        self._worker_loop_task = asyncio.create_task(self._worker_loop())
        logger.info(
            f"[worker] Worker started "
            f"(concurrency={concurrency}, edf={settings.world.edf_enabled}, "
            f"fallback_scan_interval={self._WORKER_SCAN_INTERVAL}s)"
        )

    def stop_worker(self) -> None:
        """Cancel the fallback Worker loop, EDF consumers and all in-flight tasks.

        Without cancelling the in-flight tasks, the LLM/OTP planning of the old
        scenario would carry on after a GAMA stop and push its actions to the
        next simulation (same person_ids → stale trips injected). The EDF queue
        is emptied at the same time so as not to run stale tasks."""
        if self._worker_loop_task is not None and not self._worker_loop_task.done():
            self._worker_loop_task.cancel()
            logger.info("[worker] Worker loop cancelled (scenario replaced)")
        self._worker_loop_task = None
        # Cancel the EDF consumers and empty the queue (stale jobs never run).
        for consumer in self._edf_consumers:
            if not consumer.done():
                consumer.cancel()
        _queued = len(self._edf_heap)
        self._edf_consumers = []
        self._edf_heap = []
        if _queued:
            logger.info(f"[edf] {_queued} queued job(s) discarded (scenario replaced)")
        # The queued "reflect" jobs are destroyed without running their finally:
        # purge the tracking state so that the next scenario starts clean.
        self._stm_reflecting.clear()
        self._stm_reflect_due.clear()
        self._stm_floor_day.clear()
        # Same reason for the departure decisions: a job destroyed in the queue never goes
        # through the `finally` that removes it from the registry.
        self._decisions_depart.clear()
        cancelled = 0
        for task in list(self._inflight_tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        self._inflight_tasks.clear()
        if cancelled:
            logger.info(
                f"[worker] {cancelled} in-flight task(s) cancelled (scenario replaced)"
            )

    def _spawn(self, coro) -> asyncio.Task:
        """Launches a fire-and-forget task attached to the scenario (cancellable through stop_worker)."""
        task = asyncio.create_task(coro)
        self._inflight_tasks.add(task)

        def _terminer(fini: asyncio.Task) -> None:
            self._inflight_tasks.discard(fini)
            if not fini.cancelled():
                erreur = fini.exception()
                if erreur is not None:
                    logger.error("[worker] tâche asynchrone échouée : {!r}", erreur)
                    self._prefixe_job_error = f"spawn: {erreur!r}"

        task.add_done_callback(_terminer)
        return task

    async def _consolidation_ou_arret(
        self, coro: Coroutine, person_id: str, categorie: str
    ) -> None:
        """Suspends an experiment rather than carrying on with an incomplete memory."""
        try:
            await coro
        except Exception as exc:  # noqa: BLE001 — toute consolidation manquante invalide le run
            if _arret_sur_repli_arme():
                # 2026-09-29 — the kind tells whether the stop is retried: an overload qualified by the
                # gateway (504 on both keys, 29/09 14:37) yes, an exception of the code no.
                _rejeu_refuse = getattr(
                    getattr(self.agent, "llm_client", None), "strict_replay_failure", None
                )
                _details = {"categorie": categorie, "erreur": f"{type(exc).__name__}: {exc}"}
                if _rejeu_refuse:
                    await self._declencher_hibernation_propre(
                        None, str(person_id), motif="prefixe_commun",
                        details={**_details, "cause": _rejeu_refuse},
                        genre="rejeu_obligatoire_absent",
                    )
                else:
                    await self._declencher_hibernation_propre(
                        None,
                        str(person_id),
                        motif="consolidation_memoire",
                        details=_details,
                        genre=getattr(exc, "genre", None),
                    )
                return
            raise

    # -------------------------------------------------------------------------
    # Dispatcher EDF (Earliest Deadline First)
    # -------------------------------------------------------------------------

    def _dispatch(
        self,
        deadline_sim: float,
        kind: str,
        make_coro: Callable[[], Coroutine],
        person_id: str,
        activity: Activity | None = None,
        departure_sim: float | None = None,
    ) -> None:
        """Routes a planning task: EDF queue if enabled, otherwise direct FIFO spawn.

        The invariants (scheduling_in_progress / precompute_in_progress / _worker_in_progress)
        are set by the CALLER before the call — a queued task counts as « in flight ».

        A departure decision (`plan`, `refill`) is entered in the registry of pending
        decisions until its coroutine ends — success, failure or cancellation. The
        registry therefore also sees the task that the EDF queue has already handed to a consumer.
        `departure_sim`: the departure time the decision will compute (`depart_prevu`); failing
        that, the EDF deadline.
        """
        if kind in ("plan", "refill"):
            self._decisions_depart_seq += 1
            cle = self._decisions_depart_seq
            self._decisions_depart[cle] = PendingDeparture(
                key=cle,
                person_id=str(person_id),
                activity_id=getattr(activity, "id", None),
                purpose=getattr(activity, "purpose", None),
                departure_sim=float(
                    departure_sim if departure_sim is not None else deadline_sim
                ),
                kind=kind,
                requested_sim=float(self._current_sim_timestamp),
                requested_wall=time.monotonic(),
            )
            fabrique = make_coro

            async def _suivie() -> None:
                try:
                    await fabrique()
                finally:
                    self._decisions_depart.pop(cle, None)

            make_coro = _suivie
        if settings.world.edf_enabled and self._edf_event is not None:
            self._edf_seq += 1
            job = _EdfJob(
                deadline_sim=float(deadline_sim),
                kind=kind,
                make_coro=make_coro,
                person_id=person_id,
            )
            heapq.heappush(self._edf_heap, (job.deadline_sim, self._edf_seq, job))
            self._edf_event.set()
        else:
            # Historical behaviour: fire-and-forget, the concurrency is bounded by
            # the semaphore acquired in the coroutine (_worker_concurrency_guard).
            self._spawn(make_coro())

    async def _edf_consumer(self, idx: int) -> None:
        """EDF consumer: pops the most urgent task and runs it.

        The N consumers make up the concurrency limit (they replace the
        semaphore). Safe wake-up pattern: we empty the heap, clear the event,
        retest the heap before sleeping (avoids the lost wake-up if a job arrives just
        after the clear — set() is sticky, wait() returns immediately).
        """
        assert self._edf_event is not None
        while True:
            try:
                while self._edf_heap:
                    _, _, job = heapq.heappop(self._edf_heap)
                    self._edf_active_jobs += 1
                    try:
                        await job.make_coro()
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._prefixe_job_error = f"{job.kind}:{job.person_id}: {e!r}"
                        logger.error(
                            f"[edf] job {job.kind} for {job.person_id} failed: {e}"
                        )
                    finally:
                        self._edf_active_jobs -= 1
                self._edf_event.clear()
                if self._edf_heap:
                    continue
                await self._edf_event.wait()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[edf] consumer {idx} unexpected error: {e}")

    def _worker_concurrency_guard(self):
        """Concurrency context for _plan_one / _precompute_one.

        Under EDF, the concurrency is already bounded by the number of consumers →
        no semaphore (otherwise double limitation). Otherwise (direct FIFO spawn), the
        semaphore bounds the concurrency as it historically did.
        """
        if settings.world.edf_enabled and self._edf_consumers:
            return contextlib.nullcontext()
        return self._worker_sem

    def _mark_completion(self) -> None:
        """Records the completion of a pipeline task (rate D of the EDF queue)."""
        if self._throughput is not None:
            self._throughput.mark_completion(time.monotonic())

    def throughput_per_s(self) -> float:
        """Débit de complétion courant (tâches/s, EWMA) — >= plancher, jamais 0."""
        if self._throughput is None:
            return settings.world.throughput_floor_per_s
        return self._throughput.rate(time.monotonic())

    @property
    def edf_queue_depth(self) -> int:
        """Number of tasks waiting in the EDF queue (tasks being run excluded)."""
        return len(self._edf_heap)

    def edf_snapshot_deadlines(self) -> list[float]:
        """Sorted snapshot of the deadlines (SIM time) of the queued plan/refill/reflect tasks.

        Excludes the pushes (sentinel 0: already computed, drained in ms — including them
        would skew the feasibility test by simulating an already expired deadline).
        The STM reflections (kind "reflect", deadline = wake-up of the agent, ticket 010)
        are included: it is the predictive backpressure that guarantees their deadline
        by holding the /sync if the current rate no longer allows serving them in time.
        """
        return sorted(
            job.deadline_sim
            for _, _, job in self._edf_heap
            if job.kind in ("plan", "refill", "reflect")
        )

    @property
    def pending_reflections_count(self) -> int:
        """STM reflections in the EDF queue or running (ticket 010).

        This is the « incompressible but not urgent » component of the stack: a
        backlog dominated by these tasks during the simulated night is the nominal
        drain, not a saturation (cf. backlog alarm, handle/application.py).
        """
        return len(self._stm_reflecting)

    _FENETRE_RUPTURE_S = 7 * 86400  # une semaine simulée

    def _compter_ruptures(
        self, timestamp: float, par_rupture: set, agents_total: int
    ) -> None:
        """Counts the triggers by break and raises an alarm if the regime stops being rare.

        The threshold Θ was set at 0.7 by reasoning — the line A outage is worth 0.8 in
        deterministic severity, and at Θ = 1.0 the shock studied would not have triggered — but WITHOUT
        measurement, since no run had ever computed `I_det`. This function is the measurement.

        Criterion: less than one trigger per agent and per simulated week. The alarm fires on a
        RISING EDGE, so as not to flood the journal of a sixty-day run.
        """
        if par_rupture:
            self._ruptures.append((float(timestamp), len(par_rupture)))
        # fenêtre glissante d'une semaine SIMULÉE (jamais l'horloge de la machine)
        limite = float(timestamp) - self._FENETRE_RUPTURE_S
        self._ruptures = [(t, n) for t, n in self._ruptures if t >= limite]

        total = sum(n for _, n in self._ruptures)
        if agents_total <= 0:
            return
        par_agent_semaine = total / agents_total

        if par_agent_semaine >= 1.0 and not self._rupture_alarme_on:
            self._rupture_alarme_on = True
            logger.error(
                f"[ALARME] déclenchement par rupture NON EXCEPTIONNEL : {total} sur la "
                f"dernière semaine simulée pour {agents_total} agents, soit "
                f"{par_agent_semaine:.2f} par agent et par semaine (critère : < 1). "
                f"Θ={settings.agent.memoire__theta_gravite_cumulee} est trop bas, ou les "
                f"chocs sont plus fréquents que prévu — la consolidation de nuit n'est plus "
                f"le régime de base et le coût d'inférence suit."
            )
        elif par_agent_semaine < 0.5 and self._rupture_alarme_on:
            # Back to calm at HALF the threshold, not at the threshold: without hysteresis, the alarm
            # would flap at each oscillation around 1.0.
            self._rupture_alarme_on = False
            logger.info(
                f"[gravite] régime de rupture redevenu rare : {par_agent_semaine:.2f} "
                f"déclenchement par agent et par semaine simulée"
            )

    def overdue_decision_count(self, now_sim: float) -> int:
        """Itinerary decisions (plan/refill) pending — in the EDF queue OR
        running — whose departure time has passed: the signal of a REAL saturation,
        an agent is waiting for its departure. Pushes and reflections are left out of the count.

        Read from the registry and no longer from the queue (2026-09-25): a decision popped by
        a consumer and blocked on the model call escaped it."""
        return sum(
            1 for d in self._decisions_depart.values() if d.departure_sim < now_sim
        )

    # -------------------------------------------------------------------------
    # Worker — boucle principale et scan proactif
    # -------------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Fallback scan: catches up with the Idle agents whose observation was missed.

        The normal path is O(1) through _try_schedule_person triggered at each arrival.
        This O(N) scan only runs every _WORKER_SCAN_INTERVAL seconds.
        """
        while True:
            try:
                await asyncio.sleep(self._WORKER_SCAN_INTERVAL)
                await self._scan_and_plan_all_idle()
                self._log_post_pause_drainage()
            except Exception as e:
                logger.error(f"[worker] Unexpected scan error: {e}")

    # No /sync for this long → GAMA paused (end of the simulation_max_days horizon)
    # or stopped: the silent drain of the reflections becomes observable.
    _PAUSE_QUIET_S = 90.0

    def _log_post_pause_drainage(self) -> None:
        """Visibility of the drain of the reflections after the GAMA pause (ticket 010, A3).

        At the end-of-horizon pause (`simulation_max_days`), the controller stays
        alive and the EDF consumers keep serving the queued STM reflections
        — they write to the LTM, useful to the runs that resume this population.
        Nothing interrupts this drain: only `make down` kills the process. These logs
        make the state readable so as not to stop the services too early.
        """
        if self._last_sync_wall is None:
            return
        quiet_s = time.monotonic() - self._last_sync_wall
        if quiet_s < self._PAUSE_QUIET_S:
            self._post_pause_drain_seen = 0
            return
        pending = len(self._stm_reflecting)
        if pending:
            if pending != self._post_pause_drain_seen:
                logger.info(
                    f"[drainage] GAMA silencieux depuis {quiet_s:.0f}s (pause ou fin de run) — "
                    f"{pending} réflexion(s) STM encore en file, écriture LTM en cours : "
                    f"attendre « réflexions épuisées » avant make down"
                )
                self._post_pause_drain_seen = pending
        elif self._post_pause_drain_seen:
            logger.info(
                "[drainage] Réflexions STM épuisées — LTM complète, arrêt sûr (make down)"
            )
            self._post_pause_drain_seen = 0

    # -------------------------------------------------------------------------
    # HORIZON MODEL (rolling pre-planning)
    # -------------------------------------------------------------------------
    # Each agent has a cyclic list of `nb` activities per day
    # (person.identity.activities). An activity is NOT tied to a calendar
    # day: start_time/end_time are 24h offsets; the actual day is
    # deduced from the end timestamp of the previous activity.
    #
    # Invariant aimed at, at all times, for each agent:
    #   - 1 activity IN PROGRESS (the agent is there, or travelling to it);
    #   - ~(nb-1) next activities already PLANNED (trips precomputed in
    #     person.state.precomputed_moves) or BEING planned (refill task
    #     in flight). In other words, the agent's whole day is held
    #     in advance — this is not a one-step horizon.
    #
    # Two phases feed this horizon:
    #   1. BOOTSTRAP (_bootstrap_all_agents, around L992): precomputes the whole
    #      daily cycle in waves act[N+1], act[N+2], … and stops at the full
    #      round (next_act == act_N). On exit, precomputed_moves holds
    #      ~(nb-1) trips, and precomputed_horizon_{act,ts} points to the last one.
    #   2. STEADY STATE (rolling horizon): at each trip consumed
    #      (popleft on precomputed_moves), _refill_precomputed_queue recomputes
    #      EXACTLY one trip beyond the current horizon. Pop 1,
    #      push 1 → the depth ~(nb-1) stays constant.
    #
    # Time base of the chaining: each link starts from the ARRIVAL of the previous
    # link (move.expected_arrive_at), which correctly propagates any
    # day shift (cf. weekend → Monday postponement in _compute_move_for_activity).
    # _try_schedule_next_after is a BACKUP path (empty queue) which, for its part,
    # starts from _current_sim_timestamp.
    # -------------------------------------------------------------------------

    def _try_schedule_next_after(
        self, person: Person, just_started_activity: Activity, timestamp: int
    ) -> bool:
        """Pre-plans act[N+1] using act[N].location as origin. Non-blocking.

        Called after each dispatch (bootstrap or push) to guarantee that the agent
        always has its next trip ready before reaching its destination.
        """
        if (
            person.state.scheduling_in_progress
            or person.state.next_planned_move is not None
        ):
            return False

        activities = person.identity.activities
        if not activities or just_started_activity.location is None:
            return False

        # CYCLIC chain, single implementation shared with the experiment platform
        # (`chaine_activites`): it is the divergence between the two that made the
        # platform miss the return home of each day (ticket 045, A1).
        next_act = activite_suivante(activities, just_started_activity)
        if next_act is None:
            return False

        # Build the full Unix timestamp for when just_started_activity ENDS.
        # Activity.start_time and .end_time are 24h offsets (seconds within a day).
        # If end_time < start_time the activity ends the next calendar day.
        # Using timestamp directly would give the wrong day when, e.g., act[N]=15h46
        # ends at 14h06 next day: to_timestamp_based_on_day(14h06, 5am_sim_start)
        # returns same-day 14h06 (still in the future at 5am) instead of next-day 14h06.
        act_start_ts = to_timestamp_based_on_day(
            int(just_started_activity.start_time), timestamp
        )
        if act_start_ts < timestamp:
            act_start_ts += 86400
        act_end_ts = to_timestamp_based_on_day(
            int(just_started_activity.end_time), act_start_ts
        )
        if act_end_ts < act_start_ts:
            act_end_ts += 86400

        _, _time24h = to_24h_timestamp_full(timestamp)
        person.state.scheduling_in_progress = True
        person.state.scheduling_started_at = _time24h
        self._worker_in_progress += 1
        CONTROLLER_SCHEDULING_IN_PROGRESS.set(self._worker_in_progress)
        # EDF deadline = departure base of the trip (end of the current activity): this
        # act[N+1] pre-planning is legitimately deprioritised against an imminent departure.
        self._dispatch(
            deadline_sim=act_end_ts,
            kind="plan",
            person_id=person.person_id,
            activity=next_act,
            departure_sim=depart_prevu(next_act, act_end_ts),
            make_coro=lambda: self._plan_one(
                person,
                next_act,
                act_end_ts,
                from_location_override=just_started_activity.location,
            ),
        )
        return True

    def _refill_precomputed_queue(self, person: Person) -> None:
        """Triggers the computation of the next activity beyond the current horizon.

        Called after each popleft() on precomputed_moves to keep the queue
        on a rolling 24h horizon. Non-blocking: launches _precompute_one as an asyncio task.
        """
        if person.state.precompute_in_progress:
            return
        horizon_act = person.state.precomputed_horizon_act
        horizon_ts = person.state.precomputed_horizon_ts
        if horizon_act is None or horizon_ts is None:
            return
        # Same shared cyclic chain as everywhere else (ticket 045, A1): it carries
        # the `len <= 1` guard, the matching by `id` and the check of the two locations.
        next_act = activite_suivante(person.identity.activities or [], horizon_act)
        if next_act is None:
            return
        person.state.precompute_in_progress = True
        # Departure base of the refill trip = end of the horizon activity. Brought up here
        # (and not computed in _precompute_one) to serve as EDF deadline: a distant
        # refill (D+1) is naturally deprioritised against an urgent replanning.
        from_act_end_ts = to_timestamp_based_on_day(
            int(horizon_act.end_time), horizon_ts
        )
        if from_act_end_ts < horizon_ts:
            from_act_end_ts += 86400
        self._dispatch(
            deadline_sim=from_act_end_ts,
            kind="refill",
            person_id=person.person_id,
            activity=next_act,
            departure_sim=depart_prevu(next_act, from_act_end_ts),
            make_coro=lambda: self._precompute_one(
                person, horizon_act, from_act_end_ts, next_act
            ),
        )

    async def _precompute_one(
        self, person: Person, from_act: Activity, from_act_end_ts: int, to_act: Activity
    ) -> None:
        """Computes a refill move (bounded concurrency) and appends it to precomputed_moves."""
        try:
            async with self._worker_concurrency_guard():
                move, _ = await self._compute_move_for_activity(
                    person,
                    to_act,
                    from_act_end_ts,
                    from_location_override=from_act.location,
                )
            if move:
                _retard = self._constater_depart_en_retard(person, to_act, move, "refill")
                if _retard is not None and _arret_sur_repli_arme():
                    await self.arreter_pour_decision_en_retard(_retard)
                    return
                person.state.precomputed_moves.append(move)
                person.state.precomputed_horizon_act = to_act
                person.state.precomputed_horizon_ts = move.expected_arrive_at
            else:
                logger.debug(
                    f"[refill] No move for {person.person_id}/{to_act.purpose}"
                )
        except Exception as e:
            logger.debug(f"[refill] Error for {person.person_id}/{to_act.purpose}: {e}")
        finally:
            person.state.precompute_in_progress = False
            self._mark_completion()

    def _try_schedule_person(self, person: Person, timestamp: int) -> bool:
        """Tries to schedule an agent atomically (no await).

        Checks the eligibility conditions, reserves the slot (scheduling_in_progress)
        and launches _plan_one as an asyncio task. Returns True if a task was launched.
        Called in O(1) at each arrival from handle_observation and sync.
        """
        if (
            person.state.heading_to is not None
            or person.state.scheduling_in_progress
            or person.state.next_planned_move is not None
        ):
            return False

        sched = self.population.get_person_default_scheduler(person)
        next_act = sched.next_upcoming_activity(timestamp)
        if next_act is None:
            return False

        _loc = person.state.last_location
        if (
            _loc is not None
            and next_act.location is not None
            and _loc.lat == next_act.location.lat
            and _loc.lon == next_act.location.lon
        ):
            return False

        _, _time24h = to_24h_timestamp_full(timestamp)
        person.state.scheduling_in_progress = True
        person.state.scheduling_started_at = _time24h
        self._worker_in_progress += 1
        CONTROLLER_SCHEDULING_IN_PROGRESS.set(self._worker_in_progress)
        # EDF deadline = current timestamp (the departure is due now → urgent).
        self._dispatch(
            deadline_sim=timestamp,
            kind="plan",
            person_id=person.person_id,
            activity=next_act,
            departure_sim=depart_prevu(next_act, timestamp),
            make_coro=lambda: self._plan_one(person, next_act, timestamp),
        )
        return True

    async def _scan_and_plan_all_idle(self) -> None:
        """Safety scan (every 30s): detects abnormal states and fixes them.

        Nominal state: every travelling agent has its act[N+1] in next_planned_move.
        Abnormal state:
          - Idle agent without next_planned_move nor scheduling_in_progress → WARNING + plans
          - Travelling agent without next_planned_move nor scheduling_in_progress → WARNING + pre-plans N+1
        """
        timestamp = self._current_sim_timestamp
        if timestamp == 0 or self._worker_sem is None:
            return

        _, _time24h = to_24h_timestamp_full(timestamp)
        all_people = self.population.get_people_list()

        idle_fallback: list[tuple[Person, Activity]] = []
        idle_precomputed: list[Person] = []
        moving_fallback_count = 0
        _watchdog_s = int(settings.world.arrival_watchdog_hours * 3600)

        for person in all_people:
            is_idle = person.state.heading_to is None
            has_plan = person.state.next_planned_move is not None
            in_progress = person.state.scheduling_in_progress

            # Lost-arrival watchdog: the agent is "travelling" on the Python side but
            # the arrival deadline of the pushed move has been exceeded by more than the margin.
            # The arrival observation will never come (move lost in a WebSocket
            # cut, never received by GAMA): without a forced restart, the agent would stay
            # inactive forever, invisible to the stack and to the drain.
            if (
                not is_idle
                and _watchdog_s > 0
                and person.state.heading_expected_arrive_at is not None
                and timestamp > person.state.heading_expected_arrive_at + _watchdog_s
            ):
                _overdue = timestamp - person.state.heading_expected_arrive_at
                fire_alarme("arrivee_perdue")
                logger.error(
                    f"[ALARME] Arrivée perdue — person={person.person_id} "
                    f"heading_to={person.state.heading_to} arrivée attendue à "
                    f"{humanize_date(person.state.heading_expected_arrive_at)} "
                    f"(dépassée de {humanize_duration(_overdue)}) : move jamais reçu par "
                    f"GAMA ? Reprise forcée du cycle de l'agent."
                )
                LOST_ARRIVALS_RECOVERED.inc()
                self.population.get_person_default_scheduler(person).finish_activity()
                person.state.heading_expected_arrive_at = None
                # The agent is now Idle: the branches below put it back into
                # the loop (re-push of the plan in hand or replanning).
                is_idle = True

            if is_idle and has_plan and not in_progress:
                # Push previously failed (WebSocket rollback): the agent is Idle with
                # a plan never sent — outside this case, the Idle+Planned state only exists
                # transiently inside a single coroutine.
                logger.warning(
                    f"[worker] ANOMALIE push — person={person.person_id} "
                    f"Idle avec plan non envoyé → nouvelle tentative d'envoi"
                )
                idle_precomputed.append(person)

            elif is_idle and not has_plan and not in_progress:
                # Consommer d'abord la queue pré-calculée (évite un recalcul inutile)
                if person.state.precomputed_moves:
                    person.state.next_planned_move = (
                        person.state.precomputed_moves.popleft()
                    )
                    self._refill_precomputed_queue(person)
                    idle_precomputed.append(person)
                    continue

                # Cas anormal : agent Idle sans aucun plan
                sched = self.population.get_person_default_scheduler(person)
                next_act = sched.next_upcoming_activity(timestamp)
                if next_act is None:
                    continue
                _loc = person.state.last_location
                if (
                    _loc is not None
                    and next_act.location is not None
                    and _loc.lat == next_act.location.lat
                    and _loc.lon == next_act.location.lon
                ):
                    continue
                logger.warning(
                    f"[worker] ANOMALIE idle — person={person.person_id} "
                    f"Idle sans plan → planification de secours pour {next_act.purpose}"
                )
                person.state.scheduling_in_progress = True
                person.state.scheduling_started_at = _time24h
                self._worker_in_progress += 1
                idle_fallback.append((person, next_act))

            elif not is_idle and not has_plan and not in_progress:
                # Abnormal case: travelling agent without a pre-planned act[N+1]
                current_act = person.state.cache_current_activity
                if current_act is not None:
                    logger.warning(
                        f"[worker] ANOMALIE moving — person={person.person_id} "
                        f"en route vers {person.state.heading_to} sans act[N+1] → pré-planification"
                    )
                    self._try_schedule_next_after(person, current_act, timestamp)
                    moving_fallback_count += 1

            # Detection of a silently failed refill: horizon defined, empty queue, no task in flight.
            if (
                person.state.precomputed_horizon_act is not None
                and not person.state.precomputed_moves
                and not person.state.precompute_in_progress
            ):
                logger.warning(
                    f"[worker] ANOMALIE refill — person={person.person_id} "
                    f"horizon défini mais queue vide → refill forcé"
                )
                self._refill_precomputed_queue(person)

        for person in idle_precomputed:
            # Push of an already computed move: deadline 0 = always a priority.
            self._dispatch(
                deadline_sim=0.0,
                kind="push",
                person_id=person.person_id,
                make_coro=lambda _p=person: self._push_planned_move(_p),
            )

        if not idle_fallback and moving_fallback_count == 0:
            return

        if idle_fallback:
            CONTROLLER_SCHEDULING_IN_PROGRESS.set(self._worker_in_progress)
            for person, activity in idle_fallback:
                # Planification de secours : deadline = timestamp (départ dû maintenant).
                self._dispatch(
                    deadline_sim=timestamp,
                    kind="plan",
                    person_id=person.person_id,
                    activity=activity,
                    departure_sim=depart_prevu(activity, timestamp),
                    make_coro=lambda _p=person, _a=activity: self._plan_one(
                        _p, _a, timestamp
                    ),
                )

    async def _plan_one(
        self,
        person: Person,
        activity: Activity,
        timestamp: int,
        from_location_override: Location | None = None,
    ) -> None:
        """Computes the trip of an agent (concurrency bounded by EDF or by the semaphore)."""
        async with self._worker_concurrency_guard():
            try:
                await self._compute_and_store_planned(
                    person,
                    activity,
                    timestamp,
                    from_location_override=from_location_override,
                )
            except Exception as e:
                logger.error(f"[worker] Error for {person.person_id}: {e}")
                person.state.scheduling_in_progress = False
                person.state.scheduling_started_at = None
            finally:
                self._worker_in_progress -= 1
                CONTROLLER_SCHEDULING_IN_PROGRESS.set(self._worker_in_progress)
                self._mark_completion()

    async def _compute_and_store_planned(
        self,
        person: Person,
        activity: Activity,
        timestamp: int,
        from_location_override: Location | None = None,
    ) -> None:
        """Computes the trip, stores it as Planned, then pushes if the agent is IDLE (Point 1)."""
        _pl = PipelineLogger.get()
        _pipeline_rec = (
            _pl.begin(person.person_id, timestamp)
            if (_pl is not None and person.is_llm_based)
            else None
        )
        _dispatched_act: Activity | None = None
        _dispatched_move: PersonMove | None = None
        try:
            PROCESS_PERSON_CALLS.inc()
            move, _ = await self._compute_move_for_activity(
                person,
                activity,
                timestamp,
                from_location_override=from_location_override,
                _pipeline_rec=_pipeline_rec,
            )

            if move:
                _retard = self._constater_depart_en_retard(person, activity, move, "plan")
                if _retard is not None and _arret_sur_repli_arme():
                    # Experiment: the trip would leave after its time. Neither storage nor push —
                    # the run stops, and the resume will decide this departure again in time.
                    await self.arreter_pour_decision_en_retard(_retard)
                    return

                self._itinerary_success_count += 1
                if self._itinerary_success_count >= 100:
                    ITINERARY_100_COMPLETION.set(
                        time.monotonic() - self._itinerary_window_start
                    )
                    self._itinerary_success_count = 0
                    self._itinerary_window_start = time.monotonic()

                # SENDING time = simulated time at the return of the decision, not the planning
                # base (end of the current activity): with the latter, the 17:01
                # departure served at 18:15 on 2026-09-24 counted as « on time ».
                _, _send_time24h = to_24h_timestamp_full(self._current_sim_timestamp)
                _target_24h = (
                    activity.scheduled_start_time or activity.start_time
                ) % 86400
                # Delta normalised into [-43200, +43200]: handles the midnight wrap (a sending at
                # 00:05 for a 23:55 target is worth +600 s, not -85,800 s).
                _lag_s = ((_send_time24h - _target_24h + 43200) % 86400) - 43200
                AGENT_SCHEDULING_LAG.observe(_lag_s)
                if not self._in_bootstrap:
                    if _lag_s > LATE_DEPARTURE_TOLERANCE_S:
                        DEPARTURE_PUNCTUALITY.labels(status="late").inc()
                        DEPARTURE_DELAY.observe(_lag_s)
                        if _lag_s > self._max_departure_delay_s:
                            self._max_departure_delay_s = float(_lag_s)
                            DEPARTURE_DELAY_MAX.set(_lag_s)
                    else:
                        DEPARTURE_PUNCTUALITY.labels(status="on_time").inc()

                # Stocker comme Planned
                person.state.next_planned_move = move

                # Point 1 : l'agent est IDLE → push immédiat
                pushed = await self._push_planned_move(person)
                if pushed:
                    _dispatched_act = move.for_activity
                    _dispatched_move = move
            else:
                logger.warning(f"[worker] No move computed for {person.person_id}")
        finally:
            person.state.scheduling_in_progress = False
            person.state.scheduling_started_at = None

        # Schedule act[N+1] AFTER the finally so that scheduling_in_progress=False is visible.
        # Inside the try, scheduling_in_progress=True blocks _try_schedule_next_after;
        # the same logic applies here as in _bootstrap_one (line ~754).
        if _dispatched_act is not None and person.state.heading_to is not None:
            self._try_schedule_next_after(
                person, _dispatched_act, self._base_du_chainage(_dispatched_move)
            )

    # -------------------------------------------------------------------------
    # Push WebSocket direct (idempotent)
    # -------------------------------------------------------------------------

    async def _push_planned_move(self, person: Person) -> bool:
        """Push the pre-computed planned move to GAMA if agent is IDLE.

        Idempotent: only executes when heading_to is None AND next_planned_move is set.
        Since asyncio is single-threaded, the check+set below is atomic w.r.t. other
        coroutines: no two concurrent calls can both pass the guard.
        """
        move = person.state.next_planned_move
        if move is None or person.state.heading_to is not None:
            return False

        # Réservation atomique avant tout await
        person.state.next_planned_move = None
        activity = move.for_activity
        if activity:
            self.population.get_person_default_scheduler(person).start_on_activity(
                activity=activity
            )

        if self._push_fn:
            action = Action(
                person_id=person.person_id, action=move.model_dump(exclude_none=False)
            )
            ACTIONS_CREATED.inc()
            _record_trip_mode(move, activity)
            _pl = PipelineLogger.get()
            if _pl is not None and person.is_llm_based:
                _pl.mark_enqueued(person.person_id)
            # send_message swallows the sending exceptions and returns False (dead socket,
            # reconnection in progress, "not connected"): the return must be checked just
            # like an exception, otherwise the lost push passes for a success and
            # the agent becomes a zombie (GAMA never received the trip, Python waits for
            # an arrival that will never come — cf. run 2026-07-08, WS 1006 cuts).
            delivered = False
            try:
                delivered = (await self._push_fn(action)) is not False
            except Exception as e:
                logger.error(
                    f"[push] WebSocket send failed for {person.person_id}: {e}"
                )
            if not delivered:
                logger.error(
                    f"[push] Envoi non délivré pour {person.person_id} "
                    f"(WebSocket indisponible) — rollback, le scan retentera le push"
                )
                # Full rollback: agent set back to IDLE and move restored as is —
                # the already computed trip (LLM + OTP) is not lost, the fallback scan
                # will simply retry the push (Idle state + plan present).
                person.state.heading_to = None
                person.state.cache_current_activity = None
                person.state.next_planned_move = move
                return False
            logger.info(
                f"[push] {person.person_id} → {activity.purpose if activity else '?'}"
            )
            if _pl is not None and person.is_llm_based:
                _pl.complete(person.person_id)

        # Arming of the arrival watchdog: if the sim time exceeds this deadline
        # by a margin, the arrival is considered lost and the scan forces the restart.
        person.state.heading_expected_arrive_at = move.expected_arrive_at

        # Load the next move from the precomputed queue if available and trigger
        # the rolling-horizon refill; otherwise reactive computation on the fly.
        if activity is not None:
            if person.state.precomputed_moves:
                person.state.next_planned_move = (
                    person.state.precomputed_moves.popleft()
                )
                self._refill_precomputed_queue(person)
            else:
                self._try_schedule_next_after(
                    person, activity, self._base_du_chainage(move)
                )

        return True

    def _base_du_chainage(self, move: PersonMove) -> int:
        """Reference instant to date the activity that `move` has just opened.

        `_try_schedule_next_after` dates the activity by looking for its next occurrence from
        this instant. Starting from the CURRENT time shifted the day by one day as soon as
        the push arrived after the start of the activity: on 2026-09-24, the 17:01 trip to
        « other » (start 17:48) pushed at 18:15 dated « other » to the next day, the return
        home left on the 25th at 19:07, and five trips were missing against the control. The activity
        starts after the departure of its own trip: the departure is the right anchor. The
        minimum keeps the historical anchor for an early push (nominal case, unchanged).
        """
        return min(int(self._current_sim_timestamp), _depart_du_move(move))

    # -------------------------------------------------------------------------
    # BaseScenario interface
    # -------------------------------------------------------------------------

    @property
    def population(self) -> "WorldPopulation":
        return self.model.population

    @property
    def world_bbox(self) -> BBox:
        return self.model.bbox

    async def _write_population_checkpoint(self, checkpoint_ts: int) -> None:
        # The file name carries the SIMULATED DAY (GAMA wall clock, `sim_clock`):
        # read in the time zone of the process, a late-evening checkpoint was named after the
        # next day and two simulated days ended up in the same file.
        date_str = wall_clock(checkpoint_ts).strftime("%Y-%m-%d")
        fname = f"population_{settings.data.population_size}_checkpoint_{date_str}.json"
        path = str(settings.workdir / fname)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.population.dump_population_snapshot, path)

    def _verifier_filiation(self, timestamp: int) -> None:
        """Did the replay of a child run reproduce its parent, decision by decision?

        Without this check, the saving of batch D is a promise: two arms would believe they share
        a baseline they no longer share, and the gap measured after the shock would bear on two
        different histories. It costs nothing — the decision trace of ticket 090 is written
        anyway during the normal life of any run.
        """
        nom = filiation.parent_declare()
        if not nom:
            return
        try:
            parent_dir = filiation.repertoire_parent(nom, Path(settings.workdir).parent)
            filiation.verifier_reproduction(
                parent_dir, Path(settings.workdir), float(timestamp)
            )
        except Exception as err:  # noqa: BLE001 — a check does not bring down a run
            logger.error(f"[ALARME] [filiation] recette de reproduction impossible : {err}")

    async def _ecrire_point_de_reprise(self, timestamp: int) -> None:
        """Night resume point: long-term memory, journal, anchor, counters.

        The memory metadata are FLUSHED before the copy: they are written lazily
        in bursts (`dirty` mechanism of `MultiUserLongTermMemory`), and copying without a flush
        would capture a state older than the one the simulation has actually reached.
        """
        jour = jours_ecoules(timestamp) + 1
        if gel_actif():
            # Ticket 118, O1 — during the frozen-memory replay, the memory on disk is that
            # of the restored point, not that of the replayed day. Writing here produced a « jour_001 »
            # carrying the memory of day 11 and an empty household (control a13 v5, 28/09 15:23), which
            # overwrote the good point of the first pass. The restored point remains the last valid one.
            logger.info(
                f"[reprise] point du jour {jour} NON écrit : rejeu à mémoire gelée en cours, le "
                f"point restauré reste le dernier valide."
            )
        else:
            try:
                memoire = getattr(self.agent, "long_term_memory", None)
                if memoire is not None:
                    await memoire.aflush_dirty()
                await asyncio.to_thread(
                    ecrire_point,
                    settings.workdir,
                    jour,
                    timestamp,
                    compteurs={
                        "agents": len(self.population.get_people_list()),
                        "souvenirs_par_agent": {
                            pid: len(meta.get("entries", []))
                            for pid, meta in getattr(memoire, "user_metadata", {}).items()
                        }
                        if memoire is not None
                        else {},
                    },
                    foyer=foyer_module.etat_pour_reprise(),
                )
            except Exception as exc:
                # FAIL-OPEN: losing a point costs the replay of one day, bringing down the run
                # costs all sixty.
                logger.error(
                    f"[ALARME] [reprise] écriture du point du jour {jour} impossible ({exc!r}) — "
                    f"une interruption repartirait du point précédent."
                )
        # Ticket 093 — the measures of the closed day, AFTER the resume point: they
        # read the state snapshot there (memory entries, median lifetime), which cannot be
        # rebuilt after the fact. The module is fail-open and off by default: it can
        # neither bring down the simulation, nor slow it down when nobody is measuring.
        if mesures_jour.actif():
            await asyncio.to_thread(mesures_jour.ecrire_mesures_du_jour, settings.workdir)

    # -------------------------------------------------------------------------
    # Départs servis en retard (2026-09-25)
    # -------------------------------------------------------------------------

    def _constater_depart_en_retard(
        self, person: Person, activity: Activity, move: PersonMove, kind: str
    ) -> DepartServiEnRetard | None:
        """Notes that a decision comes back after its departure time, and counts it.

        Reference: the departure time of the trip (« Heure de départ » of `moves.csv`) against the
        simulated time known at the return of the decision. Returns the finding, or None if the departure is
        still ahead.
        """
        depart = _depart_du_move(move)
        servi = int(self._current_sim_timestamp)
        if servi <= depart:
            return None
        retard = DepartServiEnRetard(
            person_id=str(person.person_id),
            activity_id=getattr(activity, "id", None),
            purpose=getattr(activity, "purpose", None),
            depart_sim=depart,
            servi_sim=servi,
            kind=kind,
        )
        self._departs_en_retard.append(retard)
        self._departs_en_retard_total += 1
        PLANNING_LATE.inc()
        AGENT_LATE_DEPARTURE.observe(retard.retard_s)
        logger.warning(
            f"[worker] LATE — person={retard.person_id} activity={retard.purpose} "
            f"({retard.activity_id}) départ={humanize_date(depart)} "
            f"décision rendue à {humanize_date(servi)} "
            f"retard≥{_duree_en_minutes(retard.retard_s)} ({kind})"
        )
        return retard

    async def arreter_pour_decision_en_retard(
        self,
        decision: PendingDeparture | DepartServiEnRetard,
        maintenant_sim: int | None = None,
    ) -> None:
        """Experiment: orderly stop rather than a trip served after its time.

        Two findings lead there:
          - `PendingDeparture` — the /sync held GAMA for its whole budget and the decision has
            still not come back; releasing GAMA would make it pass the departure time;
          - `DepartServiEnRetard` — the decision came back, but after the departure (safety
            net: the trip is neither stored nor pushed).
        """
        maintenant = int(
            maintenant_sim if maintenant_sim is not None else self._current_sim_timestamp
        )
        if isinstance(decision, PendingDeparture):
            attente_s = time.monotonic() - decision.requested_wall
            constat = (
                f"décision du départ de {humanize_date(int(decision.departure_sim))} "
                f"(activité {decision.purpose}, {decision.activity_id}) toujours attendue après "
                f"{attente_s:.0f} s réelles, demandée à {humanize_date(int(decision.requested_sim))} "
                f"— relâcher GAMA à {humanize_date(maintenant)} lui ferait franchir ce départ."
            )
            details = {
                "etat": "en_attente",
                "activity_id": decision.activity_id,
                "purpose": decision.purpose,
                "kind": decision.kind,
                "depart_ts": int(decision.departure_sim),
                "depart": humanize_date(int(decision.departure_sim)),
                "demandee_ts": int(decision.requested_sim),
                "attente_reelle_s": round(attente_s, 1),
                "temps_simule_ts": maintenant,
            }
        else:
            constat = (
                f"décision du départ de {humanize_date(decision.depart_sim)} "
                f"(activité {decision.purpose}, {decision.activity_id}) rendue à "
                f"{humanize_date(decision.servi_sim)}, soit "
                f"{_duree_en_minutes(decision.retard_s)} APRÈS le départ — trajet ni stocké ni poussé."
            )
            details = {
                "etat": "rendue_en_retard",
                "activity_id": decision.activity_id,
                "purpose": decision.purpose,
                "kind": decision.kind,
                "depart_ts": decision.depart_sim,
                "depart": humanize_date(decision.depart_sim),
                "servie_ts": decision.servi_sim,
                "retard_s": decision.retard_s,
            }
        details["constat"] = constat
        await self._declencher_hibernation_propre(
            None, decision.person_id, motif="decision_en_retard", details=details
        )

    def _evaluer_alarme_depart_en_retard(self, timestamp: int) -> None:
        """Ordinary run: `[ALARME]` at the first departure served late, lifted after
        `world.late_departure_alarm_rearm_s` simulated seconds without a new delay.

        In an experiment, the finding stops the run before getting here: no alarm."""
        retards = self._departs_en_retard
        if retards:
            self._dernier_retard_sim = float(timestamp)
        if self.arret_experience_arme:
            return
        rearm_s = settings.world.late_departure_alarm_rearm_s
        transition = late_departure_alarm_transition(
            self._alarme_retard_active,
            len(retards),
            float(timestamp),
            self._dernier_retard_sim,
            rearm_s,
        )
        if retards and self._alarme_retard_active:
            self._episode_retards += len(retards)
            self._episode_retard_max_s = max(
                self._episode_retard_max_s, max(r.retard_s for r in retards)
            )
        if transition == "fire":
            pire = max(retards, key=lambda r: r.retard_s)
            self._alarme_retard_active = True
            self._episode_retards = len(retards)
            self._episode_retard_max_s = pire.retard_s
            fire_alarme("depart_en_retard")
            logger.error(
                f"[ALARME] Départ servi en retard : person={pire.person_id} "
                f"activité={pire.activity_id} ({pire.purpose}) départ prévu "
                f"{humanize_date(pire.depart_sim)} — décision rendue à "
                f"{humanize_date(pire.servi_sim)} sim, retard ≥ "
                f"{_duree_en_minutes(pire.retard_s)} simulées ({len(retards)} départ(s) en "
                f"retard depuis le dernier /sync). Le trajet part après son heure et la suite de "
                f"la journée de l'agent peut glisser. Vérifier la latence LLM (make error, "
                f"llm_errors.jsonl). Dans une expérience (EXPERIMENT_STOP_ON_FALLBACK=1), le run "
                f"se serait arrêté."
            )
        elif transition == "release":
            logger.info(
                f"[ALARME levée] No departure served late for "
                f"{_duree_en_minutes(int(rearm_s))} simulated — episode: "
                f"{self._episode_retards} late departure(s), worst delay "
                f"{_duree_en_minutes(self._episode_retard_max_s)}."
            )
            self._alarme_retard_active = False
            self._episode_retards = 0
            self._episode_retard_max_s = 0

    async def _declencher_hibernation_propre(
        self,
        resume_at: Any,
        person_id: str,
        motif: str = "quota_journalier",
        details: dict | None = None,
        genre: str | None = None,
    ) -> None:
        """Orderly stop of the controller rather than a default decision (077 axis 3, 105).

        `genre` is the kind of error the gateway qualified (`surcharge_fournisseur`,
        `quota_journalier`, …), or `rejeu_obligatoire_absent` for a broken common prefix. With the
        reason, it sets the NATURE of the stop written in the marker (`utils/nature_arret.py`):
        `passager` is retried, `quota` waits for its reopening, `defaut` is not retried. Without
        a kind, a reason that does not state its cause (`decision_absente`, `consolidation_memoire`,
        `enquete_incomplete`) is a defect (2026-09-29).

        Guarantees that no default fallback (index 0) enters the results: rather than
        choosing in place of the model, the run stops.

        Four reasons, and they are NOT all resumed the same way:

        - `quota_journalier` — the provider announces the reopening time; `resume_at`
          carries it, and the marker allows a dated resume.
        - `replis_consecutifs` — a series of decisions the model did not take, whatever
          the cause (on 2026-09-23: upstream saturation, 54 × HTTP 503, no genre_erreur).
          No reopening time exists: `resume_at` is None and the resume is decided by
          hand, on the observed 503 rate.
        - `surcharge_fournisseur` (2026-09-25) — the gateway QUALIFIED the failure: HTTP 5xx or
          per-minute 429 on all allowed instances, batch returned before the client gives up.
          `resume_at` carries the estimated reopening (end of the cooldown), indicative: the
          resume is decided as after fallbacks, on the 503 rate.
        - `decision_en_retard` (2026-09-25) — a departure decision did not come back in time
          despite the /sync hold, or came back after the departure. Same resume as for the
          fallbacks (upstream saturation, no announced time); `details` says which departure, which
          activity and for how long.

        ⚠ The stop goes through a SIGTERM to the process, and NOT through `sys.exit`: this coroutine is
        served by the ASGI, where `SystemExit` is swallowed into a request error without ever returning an
        exit code. The orchestrator of the sequential run, for its part, waits for a 0 to chain on.
        """
        from urban_mobility_agents.utils import rejeu_decisions

        # Set BEFORE the first `await`: the asyncio loop cannot slip a second
        # caller in between the test and the assignment.
        if self._hibernation_declenchee:
            self._hibernations_ignorees += 1
            logger.info(
                f"[hibernation] déjà en cours — appel n° {self._hibernations_ignorees + 1} "
                f"({motif}, {person_id}) ignoré."
            )
            return
        self._hibernation_declenchee = True
        from urban_mobility_agents.utils.nature_arret import (
            DEFAUT,
            PASSAGER,
            nature_arret,
        )

        nature = nature_arret(motif, genre)
        _suite = {
            PASSAGER: "transient overload: the night chain will retry after its wait",
            DEFAUT: "DEFECT, not an overload: no new attempt, fix it before relaunching",
        }.get(nature, "daily quota: no new attempt before the window reopens")
        # A single [ALARME] per stop: the reason's own, just below. This line says what comes next.
        logger.warning(
            f"[hibernation] stop of nature « {nature} » (reason {motif}, error kind "
            f"{genre or 'not qualified'}, {person_id}) — {_suite}."
        )

        if motif == "decision_absente":
            logger.error(
                f"[ALARME] [hibernation] Missing decision for {person_id} after the gateway's "
                f"retries. Stop at the first failure: no default index enters the results."
            )
        elif motif == "replis_consecutifs":
            logger.error(
                f"[ALARME] [hibernation] {self._replis_consecutifs} CONSECUTIVE fallbacks "
                f"(threshold {settings.agent.replis_consecutifs_max}), last one for {person_id}: "
                f"the model no longer decides. Orderly stop of the controller rather than filling "
                f"the choice measurement with defaults. No reopening time: check upstream "
                f"before resuming."
            )
        elif motif == "surcharge_fournisseur":
            logger.error(
                f"[ALARME] [hibernation] Provider overload for {person_id}: the gateway "
                f"returned the decision before the client gave up (HTTP 5xx or per-minute 429 on "
                f"every admitted instance). Orderly stop of the controller rather than a default "
                f"fallback. Estimated reopening: {resume_at}."
            )
        elif motif == "decision_en_retard":
            logger.error(
                f"[ALARME] [hibernation] Late departure avoided for {person_id}: "
                f"{(details or {}).get('constat', 'no finding provided')} Orderly stop of the "
                f"controller rather than serving the trip after its time. No reopening time: "
                f"resume as after fallbacks (upstream saturation)."
            )
        elif motif == "consolidation_memoire":
            logger.error(
                f"[ALARME] [hibernation] Memory consolidation missing for {person_id}: "
                f"{(details or {}).get('categorie', '?')} — "
                f"{(details or {}).get('erreur', 'no detail')}. The buffer stays intact; "
                f"the resume will retry before any decision based on an incomplete memory."
            )
        elif motif == "enquete_incomplete":
            logger.error(
                f"[ALARME] [hibernation] Survey of milestone J{(details or {}).get('jalon', '?')} "
                f"incomplete after its retries ({person_id}). Orderly stop rather than a lost "
                f"answer; the milestone waits in enquete_en_attente_J*.json and will be completed "
                f"on resume, on the same perception."
            )
        elif motif == "prefixe_commun":
            logger.error(
                "[ALARME] [hibernation] Common prefix broken: {}. Arm suspended — a defect, not "
                "an overload: fix the cause, then relaunch the same experiment to resume it.",
                (details or {}).get("cause")
            )
        else:
            logger.warning(
                f"[hibernation] Daily quota exhausted for {person_id} "
                f"(reopening: {resume_at}). Orderly stop of the controller."
            )

        # FAIL-OPEN on the marker, and on it alone: losing the marker costs a resume to
        # name by hand, NOT stopping costs the rest of the run in default fallbacks.
        try:
            en_attente_path = Path(settings.workdir) / "en_attente_quota.json"
            # Ticket 105 — `resume_at` is None on a stop for fallbacks: a saturation
            # announces no reopening time. Writing "None" would suggest a date.
            resume_str = (
                None
                if resume_at is None
                else (resume_at.isoformat() if hasattr(resume_at, "isoformat") else str(resume_at))
            )
            en_attente_path.write_text(
                json.dumps(
                    {
                        "motif": motif,
                        "nature": nature,
                        "cause": genre,
                        "resume_at": resume_str,
                        "replis_consecutifs": self._replis_consecutifs,
                        "person_id": str(person_id),
                        "timestamp": self._current_sim_timestamp,
                        "jour_simule": jours_ecoules(self._current_sim_timestamp) + 1,
                        **({"detail": details} if details else {}),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            logger.info(f"[hibernation] Marqueur d'attente écrit dans {en_attente_path}")
        except OSError as exc:
            logger.error(
                f"[ALARME] [hibernation] marqueur d'attente non écrit ({exc!r}) — la reprise "
                f"devra être nommée à la main (REPRISE=<nom>)."
            )

        # Nothing to flush: `rejeu_decisions.tracer` opens, writes and closes at each decision —
        # the trace is already on disk. We log its summary, which says what the resume
        # will be able to serve again without paying the model for it again.
        logger.info(f"[hibernation] trace de rejeu — {rejeu_decisions.bilan()}")

        try:
            await self._ecrire_point_de_reprise(self._current_sim_timestamp)
        except Exception as exc:
            logger.error(
                f"[ALARME] [hibernation] point de reprise d'urgence non écrit ({exc!r}) — "
                f"la reprise repartirait du point précédent."
            )

        logger.info("[hibernation] SIGTERM au contrôleur — sortie attendue en code 0.")
        os.kill(os.getpid(), signal.SIGTERM)

    async def _juger_evenement_subi(
        self, registre, applique, person, entree_courte, gravite_mesuree: float,
        timestamp: int, detail=None,
    ) -> None:
        """The judgement of a SUFFERED event, off the observation path (ticket 100, Q6).

        Never raises: a judgement that fails leaves the event qualified on the measured fact,
        which remains true — but it leaves an alarm, never a silence.
        """
        try:
            jugement = await evenements_module.juger(
                self.agent.llm_client,
                str(person.person_id),
                self.agent.get_person_identity_description(person),
                applique.texte,
                gravite_deterministe=gravite_mesuree,
                evenement_id=applique.evenement_id,
                jour=applique.jour_run,
            )
        except evenements_module.JugementRefuse:
            # `juger` has already raised the alarm. The exposure took place: it is traced, without its
            # judgement columns — empty, never at zero.
            registre.tracer(
                applique, person.person_id, timestamp, gravite_mesuree, detail
            )
            return
        except Exception as err:  # noqa: BLE001
            logger.error(
                f"[ALARME] [evenements] jugement impossible pour {person.person_id} sur "
                f"« {applique.evenement_id} » : {err}. L'événement reste qualifié sur le seul "
                f"fait mesuré ({gravite_mesuree:.2f}) — depuis la décision D7 c'est le SEUL "
                f"chemin par lequel la gravité déterministe qualifie encore une entrée, et ce "
                f"n'est pas ce qui était déclaré."
            )
            registre.tracer(
                applique, person.person_id, timestamp, gravite_mesuree, detail
            )
            return

        tampon = self.agent.get_short_term_memory(person.person_id).recent_entries
        if entree_courte not in tampon:
            logger.error(
                f"[ALARME] [evenements] the judgement of {person.person_id} on "
                f"« {applique.evenement_id} » arrived AFTER the consolidation: "
                f"the entry had already been consumed, and the event was qualified on the measured "
                f"fact alone ({gravite_mesuree:.2f}) instead of {jugement.importance_retenue:.2f}. "
                f"Do not read that day as a judged day."
            )
            registre.tracer(
                applique, person.person_id, timestamp, gravite_mesuree, detail
            )
            return
        entree_courte.importance = jugement.importance_retenue
        entree_courte.valence = jugement.valence
        registre.tracer(
            applique, person.person_id, timestamp,
            jugement.importance_retenue, detail, jugement=jugement,
        )

    async def _injecter_evenements_du_reveil(self, timestamp: int) -> None:
        """What the agents read this morning, placed before their first decision.

        Ticket 100, batch 2; ticket 111. Nothing when no event is declared, or when
        the declared event is of the `arrivee` moment — the vast majority of runs therefore
        never go through here.

        ⚠ **When.** Launched at EACH synchronisation, outside the resume point block: it
        acts at the first simulation step after midnight, at 00:00, and not at 3 a.m. `dus_au_reveil`
        is idempotent per agent.

        ⚠ **The replay freeze (ticket 075) is checked HERE, and not only downstream.**
        `add_short_term_memory` returns without writing anything as long as the agent replays a day
        already learnt: an article whose publication day falls in the replay window
        would therefore vanish silently, and the protocol would have skipped without a line saying so.
        The refusal is OUTRIGHT and named, and the exposure is declared void (ticket 111).

        FAIL-OPEN for the rest: an injection that fails raises an alarm and lets the
        simulation carry on — but it leaves an alarm, not a silence.
        """
        registre = evenements_module.registre()
        if registre is None or registre.evenement.moment != "reveil":
            return
        try:
            dus = evenements_module.au_reveil(
                registre, list(self.population.people.values()), timestamp
            )
            if not dus:
                return
            if gel_actif():
                logger.error(
                    f"[ALARME] [evenements] {len(dus)} injection(s) au réveil dues à "
                    f"{humanize_date(timestamp)}, alors que le REJEU d'une reprise à chaud "
                    f"est en cours : la mémoire courte n'écrit rien pendant le rejeu, et ces "
                    f"injections seraient perdues sans trace. Le protocole n'a PAS eu lieu ce "
                    f"jour-là — ne pas le compter dans l'analyse."
                )
                for person_id, _ in dus:
                    self._declarer_foyer_non_avenu(registre, person_id, "gel du rejeu")
                return
            for person_id, applique in dus:
                try:
                    lu = await self._injecter_une_lecture(registre, person_id, applique, timestamp)
                    if lu and registre.evenement.relais is not None:
                        await self._injecter_le_relais(registre, person_id, applique, timestamp)
                except Exception as err:  # noqa: BLE001 — one reader does not take the others down
                    logger.error(
                        f"[ALARME] [evenements] injection au réveil de {person_id} impossible "
                        f"à {humanize_date(timestamp)} ({type(err).__name__}: {err}). Son "
                        f"exposition est déclarée non avenue."
                    )
                    self._declarer_foyer_non_avenu(registre, person_id, f"exception : {err}")
        except Exception as err:  # noqa: BLE001
            logger.error(
                f"[ALARME] [evenements] injection au réveil impossible à "
                f"{humanize_date(timestamp)} : {err}. La simulation continue, mais le "
                f"protocole de ce jour-là n'a pas eu lieu."
            )

    def _declarer_foyer_non_avenu(self, registre, lecteur_id: str, motif: str) -> None:
        """The reader AND the members of their household (assumption H7 of ticket 111).

        A transmission that follows from a reading declared void has no more taken place
        than the reading itself.
        """
        registre.declarer_non_avenue(str(lecteur_id), motif)
        if registre.evenement.relais is None:
            return
        from llm import foyer as _foyer

        for membre in _foyer.autres_membres(str(lecteur_id)):
            if registre.foyer_expose(membre.person_id):
                registre.declarer_non_avenue(
                    membre.person_id, f"lecture de {lecteur_id} non avenue ({motif})"
                )

    async def _injecter_une_lecture(self, registre, person_id: str, applique, timestamp: int) -> bool:
        """Le lecteur : jugement, mémoire courte, mémoire longue ATTENDUE. Rend vrai si écrit."""
        personne = self.population.people.get(person_id)
        if personne is None:
            logger.error(
                f"[ALARME] [evenements] lecteur « {person_id} » retenu pour "
                f"« {applique.evenement_id} » mais absent de la population chargée : "
                f"l'injection n'a pas eu lieu."
            )
            self._declarer_foyer_non_avenu(registre, person_id, "absent de la population")
            return False
        # DETERMINISTIC severity of an article: zero, and it is a fact — reading the newspaper inflicts
        # no delay. The agent's judgement (batch 3) is what gives it its weight.
        _texte = evenements_module.entree_de_lecture(applique)
        _importance, _valence, _jugement = await self._juger_a_l_injection(
            registre, personne, applique.texte, applique
        )
        if _importance is None:
            # The alarm was raised by `juger`. The exposure is void: we write NOTHING
            # rather than write an unjudged entry that would count like the others.
            self._declarer_foyer_non_avenu(registre, person_id, "jugement refusé")
            return False
        await self._ecrire_injection(personne, _texte, timestamp, _importance, _valence,
                                     applique.canal, applique.evenement_id)
        registre.tracer(applique, person_id, timestamp, _importance, None, jugement=_jugement)
        logger.info(
            f"[evenements] « {applique.evenement_id} » lu par {person_id} à "
            f"{humanize_date(timestamp)} (jour {applique.jour_run} du run, {applique.raison})"
        )
        return True

    async def _juger_a_l_injection(self, registre, personne, texte: str, applique):
        """`(importance, valence, jugement)`, or `(None, None, None)` if the judgement is refused.

        Judging first and writing afterwards avoids any requalification: the entry is born with its
        FINAL severity, hence with the right lifetime. The force is set at writing
        (`longterm.py`); raising the importance afterwards would not recompute it.

        At 00:00 simulated nothing is waiting — no arrival to process, no agent awake — and
        we are in a background task: the call delays no synchronisation.
        """
        if registre.evenement.jugement == "a_l_injection":
            try:
                jugement = await evenements_module.juger(
                    self.agent.llm_client,
                    str(personne.person_id),
                    self.agent.get_person_identity_description(personne),
                    texte,
                    gravite_deterministe=0.0,
                    evenement_id=applique.evenement_id,
                    jour=applique.jour_run,
                )
            except evenements_module.JugementRefuse:
                return None, None, None
            return jugement.importance_retenue, jugement.valence, jugement
        # Ablation déclarée (`jugement: aucun`) : aucun appel, gravité déterministe nulle.
        return 0.0, "neutre", None

    async def _ecrire_injection(self, personne, texte: str, timestamp: int, importance: float,
                                valence: str, origine: str, evenement_id: str) -> None:
        """Short-term memory then long-term memory, the latter AWAITED (ticket 111).

        (1) SHORT-term memory — so that the evening reflection sees what the agent has read or heard
        and can draw a belief from it. It is the only path to a concept.

        (2) LONG-term memory — the decision reads ONLY the long-term memory, and the evening
        consolidation consumes the short buffer without copying the raw text. Since ticket 111, the
        guarantee for the reading day no longer depends on it (the line is served at the rendering of the
        prompt); it serves the following days, where the memory decides alone. It is awaited and
        no longer launched in the background: the entry is readable as soon as the injection returns.
        """
        _contexte = Context(
            person=personne,
            activity_id=None,
            timestamp=timestamp,
            data={"evenement": evenement_id, "canal": origine},
        )
        self.agent.add_short_term_memory(
            context=_contexte,
            msg=texte,
            timestamp=timestamp,
            importance=importance,
            valence=valence,
            origine=origine,
        )
        await self.agent.aadd_long_term_memory(
            _contexte,
            MemoryEntry(
                content=texte,
                timestamp=wall_clock(timestamp),
                memory_type=MemoryType.CONVERSATION,
                person_id=str(personne.person_id),
                importance=importance,
                valence=valence,
                origine=origine,
            ),
        )

    async def _injecter_le_relais(self, registre, lecteur_id: str, applique, timestamp: int) -> None:
        """Each informed member receives the reader's message — ticket 111, batch 4.

        The relay has most often already been produced the day before, by the first decision of a
        member who asked for it; otherwise it is produced here. Each informed member JUDGES for themselves what
        they heard, with their own identity; the entry carries `origine: entendu` and will never
        go out again (guard G3). A member to whom the reader said nothing receives nothing:
        this is the household's internal control, by the reader's choice.
        """
        hh = (registre._lecteurs or {}).get(str(lecteur_id), ("",))[0]
        if not hh:
            return
        # A single reader relays per household (`lecteurs_par_foyer > 1`): the others have read, they
        # do not pass on a second time what the first one said.
        if registre._lecteur_du_foyer(hh) != str(lecteur_id):
            return
        produit = await registre.relais_du_foyer(hh)
        if produit is None:
            return
        for message in produit.messages:
            if not message.parle:
                continue
            mid = message.destinataire_id
            if mid in registre._entendus:
                continue  # already written to memory: never the same message twice
            membre = self.population.people.get(mid)
            if membre is None:
                logger.error(
                    f"[ALARME] [evenements] informé « {mid} » du foyer {hh} absent de la "
                    f"population chargée : son message n'est pas écrit."
                )
                registre.declarer_non_avenue(mid, "absent de la population")
                continue
            try:
                ligne = evenements_module.ligne_de_foyer(
                    message.texte, produit.lecteur_prenom, message.mineur
                )
                recu = dataclasses.replace(applique, raison=f"relais:{lecteur_id}", texte=ligne)
                importance, valence, jugement = await self._juger_a_l_injection(
                    registre, membre, ligne, recu
                )
                if importance is None:
                    registre.declarer_non_avenue(mid, "jugement du message refusé")
                    continue
                await self._ecrire_injection(membre, ligne, timestamp, importance, valence,
                                             "entendu", applique.evenement_id)
                registre.noter_entendu(mid)
                registre.tracer(
                    recu, mid, timestamp, importance, None, jugement=jugement,
                    extra={
                        "origine": "entendu",
                        "lecteur_id": str(lecteur_id),
                        "household_id": hh,
                        "message": message.texte,
                        "mineur": message.mineur,
                        "directif": message.directif,
                        "familles_directives": list(message.familles),
                    },
                )
                logger.info(
                    f"[evenements] « {applique.evenement_id} » entendu par {mid} de "
                    f"{lecteur_id} (foyer {hh}{', mineur' if message.mineur else ''}) à "
                    f"{humanize_date(timestamp)}"
                )
            except Exception as err:  # noqa: BLE001 — one informed member does not take the others down
                logger.error(
                    f"[ALARME] [evenements] message du foyer {hh} non écrit pour {mid} "
                    f"({type(err).__name__}: {err}). Son exposition est déclarée non avenue."
                )
                registre.declarer_non_avenue(mid, f"exception : {err}")

    async def _tirer_accidents_du_jour(self, timestamp: int) -> None:
        """Draws the accidents of the current simulated day, if the regime is active.

        FAIL-OPEN by construction: a draw that fails raises an alarm and lets the
        simulation carry on. An accident regime is scenery, not a dependency — bringing it
        down in the middle of a run of several hours would cost infinitely more
        than the absence of accidents that day.
        """
        registre = accidents_module.registre()
        if registre is None or self._sim_start_ts is None:
            return
        try:
            if not registre.pret:
                from trip_helper.osmnx_direct import _GraphStore

                graphes, _ = await _GraphStore.get()
                registre.charger_aretes(graphes["drive"])
            jour, debut_jour_ts = accidents_module.jour_simule(
                timestamp, self._sim_start_ts
            )
            registre.tirer_journee(jour, debut_jour_ts)
        except Exception as exc:
            logger.error(
                f"[ALARME] Tirage des accidents impossible à {humanize_date(timestamp)} "
                f"(jour ancré sur {humanize_date(self._sim_start_ts)}) : {exc!r}. "
                "La simulation continue SANS accident pour cette journée."
            )

    async def sync(
        self,
        timestamp: int,
        _t_sync: float | None = None,
        _t_parse: float | None = None,
    ):
        _sync_start = time.monotonic()
        self._last_sync_wall = _sync_start
        all_people = self.population.get_people_list()
        currently_idle = [p for p in all_people if p.state.heading_to is None]
        currently_moving = [p for p in all_people if p.state.heading_to is not None]
        n_planned = sum(1 for p in all_people if p.state.next_planned_move is not None)
        n_sched_in_progress = sum(
            1 for p in all_people if p.state.scheduling_in_progress
        )
        n_unscheduled_idle = sum(
            1
            for p in currently_idle
            if not p.state.scheduling_in_progress and p.state.next_planned_move is None
        )
        logger.info(
            f"[sync] START sim_time={humanize_date(timestamp)} "
            f"total_people={len(all_people)} idle={len(currently_idle)} moving={len(currently_moving)} "
            f"planned={n_planned} sched_in_progress={n_sched_in_progress} unscheduled_idle={n_unscheduled_idle} "
            f"late_since_last_sync={len(self._departs_en_retard)}"
        )
        logger.info(f"[cache] {_format_cache_hit_rates()}")
        self._evaluer_alarme_depart_en_retard(timestamp)
        self._departs_en_retard = []

        # Advance the reference timestamp of the Worker
        self._current_sim_timestamp = timestamp

        # Time tracking: real time at each 24h slice of elapsed simulated time.
        # Anchored on the first observed timestamp (and not on the calendar) to measure
        # the actual throughput of the sim (how much real time = 24 simulated hours).
        if self._sim_start_ts is None:
            self._sim_start_ts = timestamp
            self._sim_real_start = time.monotonic()
            self._next_day_log_at = timestamp + 86400
            # Ticket 075 — the anchor of the run, on which the progression of the weather date, the
            # dating of the memory journal and the resume points depend. `ancrer` does nothing if
            # a resume has already restored its own: that is what prevents the replay from
            # rewinding the weather of all the agents.
            ancrer(timestamp)
            # Weekday of the simulation start (audit A6). The EMC² survey counts
            # the trips of the day before a Tuesday→Saturday interview day: reference day
            # MONDAY to FRIDAY, never a weekend. A comparable run starts on a
            # Monday and lasts at most five simulated days; any other calendar is stated here,
            # at the moment when it is still free to fix it.
            # WALL-CLOCK weekday (`sim_clock`): it is the one GAMA displays and the one
            # whose GTFS timetables are loaded. Reading it in the time zone of the process
            # would have declared a « Friday 11 p.m. » departure as a Saturday.
            weekday = wall_clock(timestamp).isoweekday()
            if weekday >= 6:
                logger.error(
                    f"[ALARME] La simulation démarre un {humanize_date(timestamp)} — un "
                    "WEEK-END. L'enquête de référence ne compte aucun déplacement de "
                    "week-end (jours 1 à 5) : les parts modales de ce run ne sont pas "
                    "comparables à ses cibles. Corrigez `starting_date` dans "
                    "services/GAMA/CityTransport/models/Settings.gaml (un lundi 05:00)."
                )
            elif weekday != 1:
                logger.warning(
                    f"[calendrier] La simulation démarre un {humanize_date(timestamp)} "
                    "(jour de semaine {weekday}), pas un lundi : un run de cinq jours "
                    "franchira le week-end et ses départs seront reportés au lundi."
                )
            else:
                logger.info(
                    f"[calendrier] Simulation démarrée un lundi : "
                    f"{humanize_date(timestamp)} — calendrier conforme à l'enquête "
                    "(jours ouvrés) tant que le run ne dépasse pas cinq jours."
                )
        elif timestamp >= self._next_day_log_at:
            sim_day = (timestamp - self._sim_start_ts) // 86400 + 1
            real_elapsed = int(time.monotonic() - self._sim_real_start)
            logger.info(
                format_sim_timing(
                    "SIM_DAY",
                    sim_day=sim_day,
                    sim_time=humanize_date(timestamp),
                    real_elapsed=humanize_duration(real_elapsed),
                )
            )
            # Departure summary (2026-09-25): ALSO says when all is well — « 0 late »
            # tells a punctual run from a counter that no longer runs.
            logger.info(
                f"[depart] bilan au jour {sim_day} : {self._departs_en_retard_total} départ(s) "
                f"servi(s) en retard depuis le début du run ; {self._retenues_depart} retenue(s) "
                f"/sync sur départ imminent ({self._retenues_depart_s:.0f} s réelles)"
            )
            self._next_day_log_at += 86400

        # Accidents of the simulated day (ticket 070). Idempotent per day: called at
        # each sync, it only draws once. Changes no itinerary duration at this
        # stage — it populates the state of the world and publishes its counters.
        await self._tirer_accidents_du_jour(timestamp)

        # Population checkpoint — once per simulation day at 2 a.m.
        if self._next_population_checkpoint_at is None:
            self._next_population_checkpoint_at = _next_checkpoint_ts(timestamp)
        elif timestamp >= self._next_population_checkpoint_at:
            self._spawn(
                self._write_population_checkpoint(self._next_population_checkpoint_at)
            )
            self._next_population_checkpoint_at += 86400

        # ── Hot resume (ticket 075) ─────────────────────────────────────────────────
        # The THAW first: as long as the clock has not passed the resume point, the agent
        # replays days already learnt and writes nothing. Checked at each sync, not once
        # per day — the thaw must fall at the exact instant, not the next morning.
        # The step clock is noted before: it is the decision time the trace records.
        rejeu_decisions.noter_horloge(timestamp)
        if degeler_si_depasse(timestamp):
            _j = journal()
            if _j is not None:
                _j.degeler("rejeu terminé")
            # Ticket 095, batch D — the check of the child run, at the exact instant when it stops
            # replaying its parent. It is the ONLY moment the comparison makes sense: before, the
            # replay is not over; after, the child writes its own decisions.
            self._verifier_filiation(timestamp)
        # Then the point of the day, at 3 a.m. simulated: after the night drain of the reflections and
        # after the 10 p.m. floor, hence on empty short-term memory buffers.
        if self._next_reprise_at is None:
            self._next_reprise_at = _next_checkpoint_ts(
                timestamp, _REPRISE_CHECKPOINT_HOUR
            )
        elif timestamp >= self._next_reprise_at:
            _ts_point = self._next_reprise_at
            self._next_reprise_at += 86400
            self._spawn(self._ecrire_point_de_reprise(_ts_point))
            # Ticket 100, batch 4 — the household summary, once per simulated day and EVEN AT ZERO.
            # Measurement no. 1 of 078 § 6 comes before all the others: if nothing passes R1,
            # the channel is empty and nothing else makes sense to measure.
            foyer_module.journaliser_compteurs()

        # ── Ticket 100, batch 2 — the `reveil` entry point ─────────────────────────────
        # What the agent read this morning enters BEFORE its first decision of the day. That is
        # what separates this regime from the shock: the article is known while choosing, the shock is suffered
        # after having chosen. The contrast between the two is what chapter 7 measures, and
        # it only exists if both entry points stay in their place.
        #
        # Launched at EACH synchronisation, outside the resume point block: it acts at the
        # first simulation step after midnight, at 00:00 (and not at 3 a.m., as was long
        # written — ticket 111, § 2.4). Idempotent per agent, like the drawing of accidents.
        evenements_module.noter_instant(timestamp)
        self._spawn(self._injecter_evenements_du_reveil(timestamp))

        # Ticket 077 — declared modal affinity survey, at the DECLARED milestones
        # (`EXPERIMENT_SURVEY_DAYS`, default D12/D17/D29/D40).
        #
        # ⚠ Ticket 095, batch B — it is the WHOLE AGENT that is passed, and not only its LLM client.
        # The probe needs the identity narrative and the core memory of the persona: without them,
        # it questions the base model about an age and an occupation, and returns the same answer
        # on day 12 and on day 29.
        from urban_mobility_agents import enquetes as enquetes_module
        _jour_enquete = enquetes_module.is_enquete_due(timestamp, self._enquetes_menees)
        # Ticket 118, O2 — an incomplete milestone is not written: it waits, and a question that
        # stays unanswered after its retries cleanly stops the run (under the experiment
        # lock) so that it resumes later, rather than losing the answer.
        async def _saturation_enquete(jour: int, person_id: str, genre: str | None = None) -> None:
            if _arret_sur_repli_arme():
                _rejeu_refuse = getattr(
                    getattr(self.agent, "llm_client", None), "strict_replay_failure", None
                )
                _details = {"jalon": jour, "categorie": "enquete_affinite"}
                if _rejeu_refuse:
                    await self._declencher_hibernation_propre(
                        None, person_id, motif="prefixe_commun",
                        details={**_details, "cause": _rejeu_refuse},
                        genre="rejeu_obligatoire_absent",
                    )
                else:
                    await self._declencher_hibernation_propre(
                        None,
                        person_id,
                        motif="enquete_incomplete",
                        details=_details,
                        genre=genre,
                    )

        if _jour_enquete is not None:
            self._enquetes_menees.add(_jour_enquete)
            if self.agent and hasattr(self.agent, "llm_client"):
                self._spawn(
                    enquetes_module.executer_enquetes_jalon(
                        _jour_enquete,
                        timestamp,
                        list(self.population.people.values()),
                        self.agent,
                        Path(settings.workdir),
                        en_cas_de_saturation=_saturation_enquete,
                    )
                )
        # A milestone left pending (overload, then resume) is completed as soon as an agent can
        # be questioned, on the perception photographed that evening. Only one at a time.
        if (
            self.agent
            and hasattr(self.agent, "llm_client")
            and enquetes_module.lire_en_attente(Path(settings.workdir)) is not None
        ):
            self._spawn(
                enquetes_module.completer_en_attente(
                    self.agent, Path(settings.workdir), _saturation_enquete
                )
            )

        # --- Phase 2: STM reflection triggered by the volume of entries ---
        # Each reflection goes into the EDF queue (kind "reflect") with as deadline the
        # WAKE-UP of its agent (ticket 010, D2): the LTM must integrate the day before
        # the first decision of the next day, it is the only natural deadline. The
        # evening itinerary decisions mechanically go first, and the stock
        # drains all simulated night in the order of the wake-ups (early risers first).
        # In case of gateway failure, the STM entries stay in place → resubmission at the
        # next sync with the ORIGINAL deadline (_stm_reflect_due), never pushed back.
        if (
            settings.agent.long_term_memory_enabled
            and settings.agent.stm_reflection_min_entries > 0
        ):
            # Daily floor (ticket 048): past the set time, every agent whose
            # buffer is not empty becomes eligible, whatever its fill level, and
            # at most once per simulated day. The volumetric threshold is not touched:
            # the floor is ADDED to it, it does not replace it.
            _floor_day = None
            if settings.agent.stm_reflection_daily_floor_enabled:
                _now_wall = wall_clock(timestamp)
                if _now_wall.hour >= settings.agent.stm_reflection_daily_floor_hour:
                    _floor_day = _now_wall.date().toordinal()

            def _stm_eligible(_person) -> tuple[bool, str]:
                """(eligible, reason) for this agent at the current timestamp.

                THREE conditions, additive — none replaces the others:

                - `seuil`: the count of entries, historical condition;
                - `plancher`: the set time, base consolidation regime (ticket 048);
                - `rupture`: the cumulative severity crosses Θ (ticket 071, batch 1). Mechanism of
                  Park et al. (2023, § 4.2), but with a difference to state: in their work the
                  cumulative threshold is crossed two or three times a day, it is a current
                  regime; here it is EXCEPTIONAL by construction of Θ, reserved for
                  breaks. The normal consolidation remains the night one.
                """
                if not _person.is_llm_based:
                    return False, ""
                if _person.person_id in self._stm_reflecting:
                    return False, ""
                _mem = self.agent.get_short_term_memory(_person.person_id)
                _n = len(_mem.recent_entries)
                if _n >= settings.agent.stm_reflection_min_entries:
                    return True, "seuil"
                if (
                    _n > 0
                    and _mem.gravite_cumulee()
                    >= settings.agent.memoire__theta_gravite_cumulee
                ):
                    return True, "rupture"
                if (
                    _floor_day is not None
                    and _n > 0
                    and self._stm_floor_day.get(_person.person_id) != _floor_day
                ):
                    return True, "plancher"
                return False, ""

            _eligibles = [(p, *_stm_eligible(p)) for p in all_people]
            people_to_reflect = [p for p, _ok, _ in _eligibles if _ok]
            _par_plancher = {
                p.person_id for p, _ok, _m in _eligibles if _ok and _m == "plancher"
            }
            _par_rupture = {
                p.person_id for p, _ok, _m in _eligibles if _ok and _m == "rupture"
            }
            if people_to_reflect:
                # The three reasons are counted SEPARATELY: without that, checking that the break
                # regime stays exceptional — less than one trigger per agent and per
                # week — is impossible without replaying the simulation.
                logger.info(
                    f"[timestamp: {humanize_date(timestamp)}] STM reflection for {len(people_to_reflect)} agents "
                    f"({len(people_to_reflect) - len(_par_plancher) - len(_par_rupture)} par le seuil de {settings.agent.stm_reflection_min_entries} entrées, "
                    f"{len(_par_plancher)} par le plancher journalier, "
                    f"{len(_par_rupture)} par rupture (Θ={settings.agent.memoire__theta_gravite_cumulee})"
                )
                self._compter_ruptures(timestamp, _par_rupture, len(all_people))
                for _p in people_to_reflect:
                    self._stm_reflecting.add(_p.person_id)
                    if _p.person_id in _par_plancher:
                        self._stm_floor_day[_p.person_id] = _floor_day
                    _wake_ts = self.population.get_person_default_scheduler(
                        _p
                    ).next_wakeup_ts(timestamp)
                    if (
                        _wake_ts is None
                    ):  # aucune activité horodatée : fallback +12h sim
                        _wake_ts = (
                            timestamp + settings.agent.stm_reflection_deadline_sim_s
                        )
                    _due = self._stm_reflect_due.setdefault(_p.person_id, _wake_ts)

                    # Ticket 075 — the reason is CAPTURED HERE, at the moment of eligibility:
                    # when the reflection runs (EDF queue, the simulated night), the buffer
                    # will have changed and the reason could no longer be found.
                    if _p.person_id in _par_rupture:
                        _motif = "rupture"
                        _declencheur = (
                            f"gravité cumulée du tampon ≥ Θ = "
                            f"{settings.agent.memoire__theta_gravite_cumulee} — régime "
                            f"exceptionnel, réservé aux chocs"
                        )
                    elif _p.person_id in _par_plancher:
                        _motif = "plancher journalier"
                        _declencheur = (
                            f"heure dite ({settings.agent.stm_reflection_daily_floor_hour} h) "
                            f"et tampon non vide — consolidation de base, une fois par jour "
                            f"simulé au plus"
                        )
                    else:
                        _motif = "seuil"
                        _declencheur = (
                            f"{len(self.agent.get_short_term_memory(_p.person_id).recent_entries)} "
                            f"entrées accumulées, seuil "
                            f"{settings.agent.stm_reflection_min_entries}"
                        )

                    def _make_reflect_coro(
                        _person=_p,
                        _ts=timestamp,
                        _motif=_motif,
                        _declencheur=_declencheur,
                    ):
                        async def _reflect_one():
                            try:
                                await self._consolidation_ou_arret(
                                    self.agent.trigger_short_term_reflection_for_all_people(
                                        timestamp=_ts,
                                        people=[_person],
                                        motif=_motif,
                                        declencheur=_declencheur,
                                    ),
                                    _person.person_id,
                                    "stm_reflection",
                                )
                                _mem = self.agent.get_short_term_memory(
                                    _person.person_id
                                )
                                if (
                                    len(_mem.recent_entries)
                                    < settings.agent.stm_reflection_min_entries
                                ):
                                    # Lot consommé → réflexion aboutie, l'échéance est levée.
                                    self._stm_reflect_due.pop(_person.person_id, None)
                            finally:
                                self._stm_reflecting.discard(_person.person_id)

                        return _reflect_one()

                    self._dispatch(_due, "reflect", _make_reflect_coro, _p.person_id)

            # Alarm (rising edge): reflections still pending beyond their
            # simulated deadline — the guarantee « reflection finished before the wake-up of
            # its agent » (ticket 010, D2) is no longer held.
            _overdue = sum(1 for _d in self._stm_reflect_due.values() if timestamp > _d)
            if _overdue and not self._stm_overdue_alarm_on:
                logger.error(
                    f"[ALARME] {_overdue} réflexion(s) STM au-delà de leur échéance "
                    f"(réveil de l'agent) — la LTM du matin n'intègre pas la veille : "
                    f"file EDF surchargée ou providers saturés (voir make capacity)"
                )
                self._stm_overdue_alarm_on = True
            elif _overdue == 0:
                self._stm_overdue_alarm_on = False

        if settings.agent.long_term_self_reflect_enabled:
            if not self.next_self_reflection_at:
                self.next_self_reflection_at = (
                    timestamp
                    + settings.agent.long_term_self_reflect_interval_days * 24 * 3600
                )
            elif timestamp >= self.next_self_reflection_at:
                logger.info(
                    f"[timestamp: {humanize_date(timestamp)}] Self reflecting the state of the world"
                )
                _duration_days = settings.agent.long_term_self_reflect_window_days
                # Lower bound of the reflection window, in WALL-CLOCK time: it is
                # compared with the `datetime` of the memories, which carry wall-clock fields.
                from_date = wall_clock(timestamp) - datetime.timedelta(
                    days=_duration_days
                )
                from_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
                self.next_self_reflection_at = (
                    timestamp
                    + settings.agent.long_term_self_reflect_interval_days * 24 * 3600
                )
                self._spawn(
                    self._consolidation_ou_arret(
                        self.agent.trigger_long_term_reflection_for_all_people(
                            timestamp=timestamp,
                            from_date=from_date,
                            people=self.population.get_people_list(),
                        ),
                        "population",
                        "ltm_self_reflection",
                    )
                )

        _sync_duration = time.monotonic() - _sync_start
        logger.info(
            f"[sync] END sim_time={humanize_date(timestamp)} "
            f"state_update_duration={_sync_duration:.3f}s worker_backlog={self.worker_in_progress_count}"
        )

    async def trigger_short_term_reflection_for_all(self, timestamp: int):
        people = [p for p in self.population.get_people_list() if p.is_llm_based]

        sem = asyncio.Semaphore(100)

        async def reflect_person(person):
            async with sem:
                await self.agent.trigger_short_term_reflection_for_all_people(
                    timestamp=timestamp, people=[person]
                )

        tasks = [reflect_person(person) for person in people]
        await asyncio.gather(*tasks)

    async def trigger_long_term_reflection_for_all(self, timestamp: int):
        people = self.population.get_people_list()

        sem = asyncio.Semaphore(50)

        async def self_reflect_person(person):
            async with sem:
                await self.agent.reflect_on_long_term_memory(
                    timestamp=timestamp, people=[person]
                )

        tasks = [self_reflect_person(person) for person in people]
        await asyncio.gather(*tasks)

    async def handle_observation(self, observation: Observation):
        """Handle observation data.

        For an arrival: finish_activity, push if Planned (Point 2), and wake up
        the Worker so that it immediately plans the next activity.
        """
        person = self.population.get_person(observation.person_id)
        if not person:
            logger.warning(
                f"[timestamp: {humanize_date(observation.timestamp)}] Person {observation.person_id} not found in population"
            )
            return
        person.state.last_location = observation.location
        on_purpose = person.state.heading_to

        ob_text = env_ob_to_text(
            code=observation.env_ob_code,
            ob=observation.data,
            purpose=on_purpose,
            weather=get_weather(observation.timestamp),
        )

        # Delay suffered, in seconds. Only the arrival observation carries it; elsewhere it is
        # zero, which is a FACT (no delay measured) and not a missing value.
        _retard_observe_s = 0.0

        # ── Ticket 100 — the declared event, if there is one (ticket 079 for the entry point) ───
        # ⚠ NON-NEGOTIABLE RULE: the INJECTED delay is never confused with the
        # MEASURED delay. Two variables, two columns, two fields in the trace. Without this
        # separation, no rereading could tell apart what the simulation
        # produced from what it was made to say, and a hysteresis measurement that does not tell them apart
        # is not publishable.
        _choc = None
        _retard_injecte_s = 0
        _incident_reseau = False
        _correspondance_choc = False

        if observation.env_ob_code in ("arrival", "tc_timeout"):
            _started_at = observation.data.get("started_at")
            _schedule_at = observation.data.get("schedule_at")
            _timed_out = observation.env_ob_code == "tc_timeout"
            # Delay suffered (ticket 071, batch 1), read at the SOURCE and outside any branch: it
            # depends neither on the replanning, nor on the presence of an activity in cache. An
            # early arrival is worth zero and not a negative delay, which would offset a
            # real incident in the same entry.
            if observation.env_ob_code == "arrival":
                try:
                    # ⚠ NOT a simple subtraction: `expected_arrive_at` is computed by GAMA
                    # from the ORIGINAL `schedule_at` and is not recomputed when the departure
                    # is postponed (wrap to D+1, `no_weekend_departures` rule). A Friday
                    # evening trip played on Monday counted 71.6 h of delay whereas it had
                    # arrived early on its own departure. `retard_d_arrivee` removes the
                    # postponement and keeps the ordinary slip. Measured on 2026-09-23.
                    _retard_observe_s = float(
                        retard_d_arrivee(
                            int(observation.data.get("arrive_at", 0)),
                            int(observation.data.get("expected_arrive_at", 0)),
                            observation.data.get("schedule_at"),
                            observation.data.get("started_at"),
                        )
                    )
                except (TypeError, ValueError) as _err:
                    # A malformed observation must not lose the memory — but
                    # neither must it pass silently for a perfect trip.
                    logger.warning(
                        f"[gravite] retard illisible pour {observation.person_id} "
                        f"({_err}) — gravité de retard tenue pour nulle"
                    )
                    _retard_observe_s = 0.0
            # Ticket 100, `arrivee` entry point — the event applies AFTER the decision:
            # the agent chose seeing the nominal offer, it takes the hit afterwards. It is the
            # SUFFERED regime, and it is what makes the day of the event mute on the choice and
            # the following days entirely attributable to the memory.
            # ⚠ The `reveil` entry point — the article known BEFORE deciding — is not here: it lives
            # at the day switch (ticket 100, batch 2).
            _registre_chocs = evenements_module.registre()
            if _registre_chocs is not None and observation.env_ob_code == "arrival":
                # The mode comes from the table set at the DECISION (ticket 077, batch C). Read without
                # consuming it: the habit journal will remove it further down.
                _mode_du_trajet = self._mode_par_activite.get(
                    (person.person_id, str(observation.activity_id))
                )
                _choc = evenements_module.a_l_arrivee(
                    _registre_chocs, person.person_id, _mode_du_trajet, observation.timestamp
                )
                if _choc is not None:
                    _retard_injecte_s = _choc.retard_injecte_s
                    _incident_reseau = _choc.incident_reseau
                    _correspondance_choc = _choc.correspondance_ratee
                    # The text is ADDED to the observation, never substituted: the agent must
                    # keep what the simulation measured, and add to it what it lived.
                    ob_text = evenements_module.joindre(ob_text, _choc)

            await GamaArrivalsLogger.get_instance().log_arrival(
                move_id=str(observation.data.get("moving_id", "")),
                person_id=observation.person_id,
                arrive_at=int(observation.data.get("arrive_at", observation.timestamp)),
                expected_arrive_at=int(
                    observation.data.get("expected_arrive_at", observation.timestamp)
                ),
                started_at=int(_started_at) if _started_at is not None else None,
                schedule_at=int(_schedule_at) if _schedule_at is not None else None,
                timed_out=_timed_out,
                retard_injecte_s=_retard_injecte_s,
            )
            if (
                observation.env_ob_code == "arrival"
                and person.state.cache_current_activity
            ):
                activity = person.state.cache_current_activity
                ob = parse_ob(code=observation.env_ob_code, ob=observation.data)
                if settings.agent.reschedule_activity_departure_time:
                    duration = self.reschedule_amount_function(
                        arrival_late_seconds=ob.late
                    )
                    self.population.get_person_default_scheduler(
                        person
                    ).reschedule_activity(activity, duration)
                    _context = Context(
                        person=person,
                        activity_id=observation.activity_id,
                        timestamp=observation.timestamp,
                        data={
                            "location": observation.location.model_dump(
                                exclude_none=True
                            ),
                            "heading_to": person.state.heading_to,
                            "data": observation.data,
                        },
                    )
                    self.agent.add_short_term_memory(
                        context=_context,
                        msg=f"Because you arrived late, you adjusted your target arrival time for {activity.purpose}. You will now aim to arrive at {humanize_time(activity.scheduled_start_time)}, which is {humanize_duration(activity.start_time - activity.scheduled_start_time)} earlier than originally planned.",
                        timestamp=observation.timestamp,
                    )

            # Transition ACTIVE → IDLE
            self.population.get_person_default_scheduler(person).finish_activity()
            person.state.heading_expected_arrive_at = (
                None  # désarme le watchdog d'arrivée
            )

            # Mettre à jour le timestamp de référence
            self._current_sim_timestamp = max(
                self._current_sim_timestamp, observation.timestamp
            )

            # Point 2: if the next trip is already Planned, immediate push
            pushed = await self._push_planned_move(person)

            # Chemin rapide O(1) : planifier directement cet agent sans scan global
            if not pushed:
                self._try_schedule_person(person, self._current_sim_timestamp)

        # Ticket 071, batch 1 — DETERMINISTIC severity of the observation, computed from what the
        # simulation measured. No call to the model, no judgement: a delay is a delay.
        #   - `arrival` carries the delay suffered, through the `late` property of the observation;
        #   - `tc_timeout` IS the missed connection: the agent saw its vehicle leave.
        # The « constrained mode » component comes through the DECISION path (the context carries
        # `contrainte_chaine`), and « network incident » has no source yet.
        # Ticket 079 — the severity carries the SUM of the two delays, and `incident_reseau`
        # finds here the source that ticket 071 had left pending for it.
        _gravite, _detail = gravite_deterministe(
            retard_s=_retard_observe_s + _retard_injecte_s,
            correspondance_ratee=(
                observation.env_ob_code == "tc_timeout" or _correspondance_choc
            ),
            incident_reseau=_incident_reseau,
        )
        # ⚠ A SINGLE LINE PER EXPOSURE, whatever happens. When a judgement is
        # expected, the trace goes with it — otherwise the exposure would be written twice, once
        # bare and once judged, and any count per exposure would be wrong. When the
        # judgement fails, it is written anyway, without its judgement columns: an
        # exposure that took place is traced, judged or not.
        _jugement_attendu = (
            _choc is not None
            and _registre_chocs is not None
            and _registre_chocs.evenement.jugement == "a_l_injection"
        )
        if _choc is not None and _registre_chocs is not None and not _jugement_attendu:
            _registre_chocs.tracer(
                _choc, person.person_id, observation.timestamp, _gravite, _detail
            )
        if _gravite > 0:
            logger.debug(
                f"[gravite] {person.person_id} — {observation.env_ob_code} : I_det="
                f"{_gravite:.2f} (composantes : {', '.join(_detail.composantes_actives())})"
            )
        if _gravite >= settings.agent.memoire__importance_choc:
            # A shock is rare by construction of the threshold: saying so at INFO level allows
            # finding it in a run journal without replaying the simulation.
            logger.info(
                f"[gravite] CHOC pour {person.person_id} à "
                f"{humanize_date(observation.timestamp)} : I_det={_gravite:.2f} "
                f"({observation.env_ob_code}, retard {int(_retard_observe_s)} s)"
            )

        if observation.env_ob_code != "arrival":
            # V5 concatenation: the micro-steps (transit, transfer, wait_in_stop, vehicle, tc_timeout)
            # are kept to be merged into the final arrival report.
            self._etapes_trajet_en_cours.setdefault(str(person.person_id), []).append({
                "code": observation.env_ob_code,
                "text": ob_text,
                "gravite": _gravite,
                "timestamp": observation.timestamp,
            })
            return

        # Arrival: merge the intermediate steps of the trip into a single report
        _etapes_precedentes = self._etapes_trajet_en_cours.pop(str(person.person_id), [])
        if _etapes_precedentes:
            _resume_etapes = " ; ".join(
                e["text"] for e in _etapes_precedentes if e.get("text")
            )
            if _resume_etapes:
                ob_text = f"Étapes du trajet : {_resume_etapes} — {ob_text}"
            _max_gravite = max((e.get("gravite", 0.0) for e in _etapes_precedentes), default=0.0)
            if _max_gravite > _gravite:
                _gravite = _max_gravite

        _context = Context(
            person=person,
            activity_id=observation.activity_id,
            timestamp=observation.timestamp,
            data={
                "location": observation.location.model_dump(exclude_none=True),
                "heading_to": person.state.heading_to,
                "data": observation.data,
            },
        )
        self.agent.add_short_term_memory(
            context=_context,
            msg=ob_text,
            timestamp=observation.timestamp,
            importance=_gravite,
            origine="vecu",
        )

        # ── Ticket 100, batch 3 — the agent's judgement, SYNCHRONOUS WITH THE INJECTION ────────
        # Q6 put it in a background task, so as not to delay the processing of the arrival: the
        # consolidation was to take as FLOOR the highest importance of the entries
        # it consumes, and raising the short entry before it was consumed was enough.
        #
        # ⚠ THIS REASONING ASSUMED AN EVENING CONSOLIDATION. There is none: the reflection
        # goes at the ENTRY THRESHOLD (`stm_reflection_min_entrees`, 10 by default). The entry of
        # the event is therefore often the one that crosses the threshold — it then triggers
        # the consolidation that consumes it, while its own judgement is in flight. Measured
        # on 2026-09-23: injection at 10:24:08, consolidation at 10:24:09, judgement returned at
        # 10:24:14. Five seconds too late, and the event qualified on the measured fact (0.87)
        # instead of the judged severity (0.75) — that is, under the regime D7 abandoned.
        #
        # The race cannot be won: it depends on the number of trips that precede
        # the event in the day, and the judgement loses it every time the injected
        # entry is the tenth. So we wait for it, which `a_l_injection` announces anyway,
        # and which the `reveil` entry point already does (judgement BEFORE writing). Cost: the
        # processing of THAT arrival waits a few seconds, once or twice per run.
        if _jugement_attendu:
            _entree_courte = self.agent.get_short_term_memory(
                person.person_id
            ).recent_entries[-1]
            await self._juger_evenement_subi(
                _registre_chocs, _choc, person, _entree_courte, _gravite,
                observation.timestamp, _detail,
            )

        # Ticket 071, batch 4 — the completed trip enters the JOURNAL, from which the agent's
        # habits come. It is here, and only here, that the mode kept and the delay ACTUALLY
        # suffered are known together: the mode comes from the decision entry, written earlier
        # in the day, and the delay from the arrival observation.
        if (
            observation.env_ob_code == "arrival"
            and self.agent.long_term_memory is not None
        ):
            try:
                # ⚠ Ticket 077, batch C — the mode of THIS trip, and of no other.
                # It is read from the table set at the DECISION, never from the short-term
                # memory buffer: the latter is emptied by the consolidations, so that
                # the decision entry has often disappeared when the arrival occurs. Going back
                # to the last `axe_objet` of the buffer attributed the arrival to the PREVIOUS trip
                # (run of 075: the 58 returns of agent 609 filed under the « work » purpose
                # of the outbound trip); requiring the activity in the buffer found almost
                # nothing any more (run of 15/09: 2 trips logged for 40 arrivals).
                _mode_retenu = self._mode_par_activite.pop(
                    (person.person_id, str(observation.activity_id)), None
                )
                if _mode_retenu is None:
                    # Nothing is recorded, and the gap shows. Attributing the arrival to an
                    # earlier decision would produce a false habit, which is worse
                    # than a missing habit: the prompt block is read by the model.
                    logger.warning(
                        f"[noyau] arrival without a known chosen mode for "
                        f"{person.person_id} (activity {observation.activity_id}) — "
                        f"trip NOT logged, habits incomplete"
                    )
                else:
                    # The purpose and the time slot come from the ARRIVAL being recorded. Only
                    # the mode comes from the decision: it is the only one of the three things that an
                    # arrival does not carry.
                    _motif = normaliser_motif(
                        observation.data.get("purpose") or person.state.heading_to
                    )
                    self.agent.long_term_memory.noter_trajet(
                        person.person_id,
                        _motif,
                        creneau_de(wall_clock(observation.timestamp)),
                        _mode_retenu,
                        retard_s=_retard_observe_s + _retard_injecte_s,
                    )
            except Exception as _err:  # noqa: BLE001 — the journal must not bring anything down
                logger.warning(
                    f"[noyau] trajet non journalisé pour {person.person_id} ({_err}) — "
                    f"les habitudes de cet agent seront incomplètes"
                )

    def reschedule_amount(self, arrival_late_seconds: int) -> int:
        if arrival_late_seconds <= 0:
            return 0
        amount = min(
            int(abs(arrival_late_seconds) * settings.agent.reschedule_transition_ratio),
            self.MAX_ADJUST_START_TIME,
        )
        amount = amount if arrival_late_seconds > 0 else -amount
        return amount

    def reschedule_amount_v2(self, arrival_late_seconds: int) -> int:
        if arrival_late_seconds <= 0:
            return 0
        k = settings.agent.reschedule_activity_v2__k or 0.02
        arrival_late_minutes = arrival_late_seconds / 60.0
        amount = min(
            k * arrival_late_minutes * arrival_late_minutes * 60,
            self.MAX_ADJUST_START_TIME,
        )
        amount = int(amount) if arrival_late_seconds > 0 else -int(amount)
        return amount

    async def has_messages(self) -> bool:
        return len(self._messages) > 0

    async def pop_all_messages(self) -> list[Action]:
        messages = self._messages.copy()
        self._messages.clear()
        return messages

    async def bootstrap_all_agents(self, timestamp: int):
        """Pre-compute the first upcoming itinerary for every agent.

        Called from /init so GAMA's reflex init blocks on the HTTP response until
        every OTP query is done. Displays a rich progress bar in the console.
        The trips are stored in _messages (publish_loop sends them after /init).
        """
        from rich.console import Console as RichConsole
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        # Initialise the reference timestamp of the Worker from the bootstrap on
        self._current_sim_timestamp = timestamp

        # Marque la phase bootstrap : ACTIVITY_DECISIONS tag phase=bootstrap, cockpit ③
        # n'y compte aucune activité ratée. Réinitialise les jauges d'avancement (cockpit ①).
        self._in_bootstrap = True
        BOOTSTRAP_ACTIVE.set(1)
        BOOTSTRAP_COMPLETED.set(0)
        BOOTSTRAP_PROGRESS.set(0)
        BOOTSTRAP_CACHE_HITS.set(0)
        BOOTSTRAP_CACHE_MISSES.set(0)
        BOOTSTRAP_WAVE.set(1)
        BOOTSTRAP_FUTURE_MOVES.set(0)
        BOOTSTRAP_WAVE_MOVES.clear()  # purges the waves of a possible previous /init

        all_people = self.population.get_people_list()
        _, _time24h = to_24h_timestamp_full(timestamp)

        eligible: list[tuple[Person, Activity]] = []
        for person in all_people:
            if (
                person.state.heading_to is not None
                or person.state.scheduling_in_progress
            ):
                continue
            sched = self.population.get_person_default_scheduler(person)
            next_act = sched.next_upcoming_activity(timestamp)
            if next_act is None:
                continue
            person.state.scheduling_in_progress = True
            person.state.scheduling_started_at = _time24h
            eligible.append((person, next_act))

        total = len(eligible)
        BOOTSTRAP_TOTAL.set(total)
        _w1 = _wave_metrics(1, total)
        logger.info(
            f"[bootstrap] sim_time={humanize_date(timestamp)} "
            f"computing itineraries for {total}/{len(all_people)} agents — GAMA blocked"
        )
        _bootstrap_start = time.monotonic()

        # Dedicated console on stderr, force_terminal=True for display in VSCode
        # even when stdout is captured by uvicorn/hypercorn.
        # Loguru writes to stdout → no interference.
        _rich_console = RichConsole(
            stderr=True,
            force_terminal=True,
            highlight=False,
        )
        _rich_console.print(
            f"\n[bold cyan]Bootstrap :[/bold cyan] calcul de {total} itinéraires initiaux…\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan][bootstrap][/bold cyan]"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("agents planifiés"),
            TimeElapsedColumn(),
            console=_rich_console,
            refresh_per_second=4,
            transient=False,
        ) as progress:
            task_id = progress.add_task("bootstrap", total=total)
            _cache_stats = {"hits": 0, "misses": 0, "done": 0}

            # Smoothing of the burst: bounds the in-flight OTP+LLM pipelines (cf.
            # settings.world.bootstrap_concurrency) so as not to saturate the provider
            # quotas at /init. The tasks are all created at once but
            # only enter the computation in waves as slots are released.
            _boot_sem = asyncio.Semaphore(max(1, settings.world.bootstrap_concurrency))

            async def _bootstrap_one(person: Person, act: Activity):
                _dispatched_act = None
                try:
                    _pl = PipelineLogger.get()
                    _pipeline_rec = (
                        _pl.begin(person.person_id, timestamp)
                        if (_pl is not None and person.is_llm_based)
                        else None
                    )
                    PROCESS_PERSON_CALLS.inc()
                    async with _boot_sem:
                        move, _reason = await self._compute_move_for_activity(
                            person, act, timestamp, _pipeline_rec=_pipeline_rec
                        )
                    if "cache sémantique" in (_reason or ""):
                        _cache_stats["hits"] += 1
                        BOOTSTRAP_CACHE_HITS.set(_cache_stats["hits"])
                        _w1["cache_hit"].inc()
                        logger.info(
                            f"[bootstrap] cache hit — {person.person_id} / {act.purpose}"
                        )
                    elif person.is_llm_based:
                        _cache_stats["misses"] += 1
                        BOOTSTRAP_CACHE_MISSES.set(_cache_stats["misses"])
                        _w1["cache_miss"].inc()
                        logger.info(
                            f"[bootstrap] cache miss — {person.person_id} / {act.purpose}"
                        )
                    if move:
                        _w1["ok"].inc()
                        self._itinerary_success_count += 1
                        if self._itinerary_success_count >= 100:
                            ITINERARY_100_COMPLETION.set(
                                time.monotonic() - self._itinerary_window_start
                            )
                            self._itinerary_success_count = 0
                            self._itinerary_window_start = time.monotonic()
                        ACTIONS_CREATED.inc()
                        _record_trip_mode(move, act)
                        # Chemin bootstrap : push via _messages (publish_loop après /init)
                        self._messages.append(
                            Action(
                                person_id=person.person_id,
                                action=move.model_dump(exclude_none=False),
                            )
                        )
                        self.population.get_person_default_scheduler(
                            person
                        ).start_on_activity(activity=act)
                        if _pl is not None and person.is_llm_based:
                            _pl.mark_enqueued(person.person_id)
                        _dispatched_act = act
                except Exception as _e:
                    logger.debug(
                        f"[bootstrap] Error for {person.person_id}/{act.purpose}: {_e}"
                    )
                finally:
                    person.state.scheduling_in_progress = False
                    person.state.scheduling_started_at = None
                    progress.advance(task_id, 1)
                    _cache_stats["done"] += 1
                    BOOTSTRAP_COMPLETED.set(_cache_stats["done"])
                    _w1["done"].inc()
                    if total:
                        BOOTSTRAP_PROGRESS.set(_cache_stats["done"] / total)
                # Pre-plan act[N+1] with act[N].location as origin as soon as the bootstrap ends
                if _dispatched_act is not None:
                    self._try_schedule_next_after(person, _dispatched_act, timestamp)

            tasks = [
                asyncio.create_task(_bootstrap_one(person, act))
                for person, act in eligible
            ]
            completed = 0
            log_interval = max(1, total // 10)
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception:
                    pass
                completed += 1
                if completed % log_interval == 0 or completed == total:
                    _llm_total = _cache_stats["hits"] + _cache_stats["misses"]
                    _hit_pct = (
                        int(100 * _cache_stats["hits"] / _llm_total)
                        if _llm_total
                        else 0
                    )
                    logger.info(
                        f"[bootstrap] progress {completed}/{total} ({100 * completed // total}%) "
                        f"— cache hits: {_cache_stats['hits']}/{_llm_total} ({_hit_pct}%)"
                    )

        # Wait for all the act[N+1] launched by _try_schedule_next_after to be computed
        # before returning, so that GAMA starts without backpressure at the first /sync.
        if self._worker_in_progress > 0:
            logger.info(
                f"[bootstrap] waiting for {self._worker_in_progress} act[N+1] pre-planning tasks..."
            )
            while self._worker_in_progress > 0:
                await asyncio.sleep(0.5)

        # Precompute act[N+2], act[N+3], ... for each agent and store them in
        # precomputed_moves. This avoids the planning peak during the morning rush
        # hours (100+ agents arriving at the same time who all trigger an OTP computation).
        # Track the full Unix arrival timestamp for the last planned activity per person.
        # This is needed so each wave can compute the correct calendar day for the
        # following activity: act[N+2] must depart AFTER act[N+1] ends, not on the
        # simulation start day.
        person_last_planned_act: dict[str, Activity] = {}
        person_last_planned_ts: dict[str, int] = {}
        for person, act_N in eligible:
            move = person.state.next_planned_move
            if move and move.for_activity:
                person_last_planned_act[person.person_id] = move.for_activity
                person_last_planned_ts[person.person_id] = move.expected_arrive_at

        async def _compute_future_wave_move(
            person: Person,
            from_act: Activity,
            from_arrive_ts: int,
            to_act: Activity,
            wm: dict,
        ) -> None:
            """Computes a bootstrap wave trip under the concurrency semaphore.

            `wm`: agent_bootstrap_wave_moves counters of the wave (cf. _wave_metrics).
            """
            try:
                # Compute when from_act ends: its end_time is a 24h offset, and may
                # be on the next calendar day relative to when the person arrives.
                from_act_end_ts = to_timestamp_based_on_day(
                    int(from_act.end_time), from_arrive_ts
                )
                if from_act_end_ts < from_arrive_ts:
                    from_act_end_ts += 86400
                async with self._worker_sem:
                    move, _reason = await self._compute_move_for_activity(
                        person,
                        to_act,
                        from_act_end_ts,
                        from_location_override=from_act.location,
                    )
                if "cache sémantique" in (_reason or ""):
                    wm["cache_hit"].inc()
                elif person.is_llm_based:
                    wm["cache_miss"].inc()
                if move:
                    wm["ok"].inc()
                    person.state.precomputed_moves.append(move)
                    person_last_planned_act[person.person_id] = to_act
                    person_last_planned_ts[person.person_id] = move.expected_arrive_at
                else:
                    person_last_planned_act.pop(person.person_id, None)
                    person_last_planned_ts.pop(person.person_id, None)
            except Exception as _e:
                logger.debug(
                    f"[bootstrap/wave] Error for {person.person_id}/{to_act.purpose}: {_e}"
                )
                person_last_planned_act.pop(person.person_id, None)
                person_last_planned_ts.pop(person.person_id, None)
            finally:
                wm["done"].inc()

        wave = 2
        while person_last_planned_act:
            wave_batch: list[tuple[Person, Activity, int, Activity]] = []

            for person, act_N in eligible:
                pid = person.person_id
                last_act = person_last_planned_act.get(pid)
                last_ts = person_last_planned_ts.get(pid)
                if last_act is None or last_ts is None:
                    continue
                # Same shared cyclic chain as everywhere else (ticket 045, A1). It
                # adds here the check of the locations that this site did not do: an
                # end without a location produces no computable trip, and the
                # person leaves the tracking instead of entering the wave to fail there.
                next_act = activite_suivante(person.identity.activities or [], last_act)
                if next_act is None:
                    del person_last_planned_act[pid]
                    person_last_planned_ts.pop(pid, None)
                    continue
                if next_act.id == act_N.id:
                    # Cycle complet : on a couvert toutes les activités de la journée
                    del person_last_planned_act[pid]
                    person_last_planned_ts.pop(pid, None)
                    continue
                wave_batch.append((person, last_act, last_ts, next_act))

            if not wave_batch:
                break

            logger.info(
                f"[bootstrap] pre-computing {len(wave_batch)} act[N+{wave}] itineraries (wave {wave})..."
            )
            BOOTSTRAP_WAVE.set(wave)
            _wm = _wave_metrics(wave, len(wave_batch))

            wave_tasks = [
                asyncio.create_task(_compute_future_wave_move(p, fa, fa_ts, ta, _wm))
                for p, fa, fa_ts, ta in wave_batch
            ]
            for coro in asyncio.as_completed(wave_tasks):
                try:
                    await coro
                except Exception:
                    pass
                BOOTSTRAP_FUTURE_MOVES.set(
                    sum(len(p.state.precomputed_moves) for p in all_people)
                )

            wave += 1

        # Initialise the rolling horizon of each agent from the results of the bootstrap.
        # person_last_planned_act/ts hold the last activity computed by the waves.
        for person, act_N in eligible:
            pid = person.person_id
            last_act = person_last_planned_act.get(pid)
            last_ts = person_last_planned_ts.get(pid)
            if last_act is not None and last_ts is not None:
                person.state.precomputed_horizon_act = last_act
                person.state.precomputed_horizon_ts = last_ts

        n_precomputed = sum(len(p.state.precomputed_moves) for p in all_people)
        logger.info(
            f"[bootstrap] all activities pre-planning tasks done ({n_precomputed} future moves pre-cached across {wave - 2} additional wave(s))"
        )

        _bootstrap_duration = time.monotonic() - _bootstrap_start
        BOOTSTRAP_DURATION.set(_bootstrap_duration)
        _planned = sum(1 for p in all_people if p.state.heading_to is not None)
        logger.info(
            f"[bootstrap] done — {total} itineraries computed in {_bootstrap_duration:.2f}s"
        )
        logger.info(f"[cache] {_format_cache_hit_rates()}")
        _rich_console.print(
            f"\n[bold green]✓ Bootstrap terminé[/bold green] — "
            f"[cyan]{_planned}/{total}[/cyan] agents planifiés en "
            f"[yellow]{_bootstrap_duration:.1f}s[/yellow]\n"
        )
        # End of the bootstrap phase: the following decisions are live (cockpit ③ counts from here).
        self._in_bootstrap = False
        BOOTSTRAP_ACTIVE.set(0)

    # -------------------------------------------------------------------------
    # Trip computation (partagé bootstrap + Worker)
    # -------------------------------------------------------------------------

    def _settle_vehicles_at_home(self, person: Person, arrived_at: Activity) -> None:
        """Settles the loop when a planned trip brings the agent back home.

        The return lock covers the vehicles parked at the starting point of the return
        trip, not those left at an intermediate stop (home → work by
        car, work → sport on foot, sport → home by bus: the car sleeps at
        work). These orphans are counted, and by default brought back home: without
        this catch-up, the agent would lose its car for good for all following
        days — a bias far worse than the teleportation corrected here.
        """
        if (arrived_at.purpose or "").lower() != "home":
            return
        self._vehicle_home_returns += 1
        orphans = _orphaned_vehicles(person)
        if not orphans:
            return

        self._vehicle_orphan_returns += 1
        for mode in sorted(orphans):
            VEHICLE_CHAIN.labels(mode=mode, event="orphaned").inc()
            if settings.agent.vehicle_orphan_reset_at_home:
                person.state.planning_vehicle_at.pop(mode, None)
                VEHICLE_CHAIN.labels(mode=mode, event="reset_home").inc()

        # Rising-edge alarm: the catch-up is an acceptable approximation as long as
        # it stays marginal. Beyond the threshold, it drives a significant share of the
        # trips and the measurement of the modal shares depends on it.
        ratio = self._vehicle_orphan_returns / max(1, self._vehicle_home_returns)
        threshold = settings.agent.vehicle_orphan_alarm_ratio
        if (
            self._vehicle_home_returns
            >= settings.agent.vehicle_orphan_alarm_min_returns
            and ratio > threshold
            and not self._vehicle_orphan_alarm_on
        ):
            self._vehicle_orphan_alarm_on = True
            logger.error(
                f"[ALARME] Véhicules orphelins au retour au domicile : "
                f"{self._vehicle_orphan_returns}/{self._vehicle_home_returns} retours "
                f"({100 * ratio:.1f}% > {100 * threshold:.1f}%) — le rattrapage implicite "
                f"au domicile porte une part notable des trajets véhiculés"
            )
            fire_alarme("vehicule_orphelin")
        elif self._vehicle_orphan_alarm_on and ratio < threshold / 2:
            self._vehicle_orphan_alarm_on = False

    # ── Jeu de déplacements enregistré (ticket 035, spec 04) ─────────────────────────
    GROUPES_TOLERANCE = ("walk", "bike", "car", "transit", "rail")

    def charger_jeu(self, chemin: str, info_population) -> None:
        """G1/G2/G5 — loads the designated set, refuses if it is not the population's or
        if the time tolerances are not declared. Raises RuntimeError with the reason."""
        from experiences.experience import ToleranceHoraire
        from experiences.jeu import Jeu

        jeu = Jeu.charger(chemin)
        mismatch = jeu.verifier_population(info_population)
        if mismatch:
            raise RuntimeError(f"trip set refused: {mismatch}")
        declarees = settings.data.jeu_tolerances_horaires or {}
        manquants = [g for g in self.GROUPES_TOLERANCE if g not in declarees]
        if manquants:
            raise RuntimeError(
                "trip set refused: hourly tolerances not declared for "
                f"{manquants} (setting data.jeu_tolerances_horaires — no default in the code, spec 04 G5)"
            )
        self.jeu_tolerances = {
            g: ToleranceHoraire.depuis_yaml(v) for g, v in declarees.items()
        }
        self.jeu = jeu
        self._jeu_stats = _Compteur()
        self._jeu_propositions_par_source = _Compteur()
        couv = jeu.couverture()
        logger.info(
            f"[jeu] Jeu enregistré {jeu.nom!r} chargé (empreinte {str(jeu.empreinte)[:12]}…) : "
            f"{couv['deplacements_couverts']}/{couv['deplacements_attendus']} déplacements couverts, "
            f"tolérances {{{', '.join(f'{g}: {t.type}' + (f' {t.pas_min} min' if t.pas_min else '') for g, t in self.jeu_tolerances.items())}}} — "
            f"régime nominal : aucun appel moteur pour les déplacements couverts"
        )
        self._ecrire_jeu_stats(force=True)

    @staticmethod
    def _groupe_tolerance(plan: TravelPlan) -> str:
        g = _selection_group(plan)
        return "rail" if g == GROUPE_RAIL else g

    @staticmethod
    def _ecart_horaire(departure_time: int, reference_ts: int) -> int:
        """Actual − reference gap, brought back within the day (±12 h): the set holds for each simulated day."""
        e = (int(departure_time) - int(reference_ts)) % 86400
        return e - 86400 if e >= 43200 else e

    def _propositions_du_jeu(
        self, person: Person, next_activity: Activity, departure_time: int
    ) -> dict | None:
        """Recorded proposals of the trip, and the groups to recompute (G3, G5).

        `None` if the set does not cover this trip (source `hors_jeu`, question 9)."""
        ligne = self.jeu.ligne(person.person_id, next_activity.id)
        if ligne is None:
            return None
        propositions = ligne.vers_propositions()
        ecart = self._ecart_horaire(departure_time, ligne.depart_ts)
        presents = {self._groupe_tolerance(p.plan) for p in propositions}
        motifs: dict[str, str] = {}
        for g, tol in self.jeu_tolerances.items():
            if g in presents and tol.hors_tolerance(ecart, ligne.depart_ts):
                motifs[g] = "horaire"
        # Author's decision (2026-09-06, question 18): another simulated day plays the transport
        # offer of THAT day. The transit/rail groups are recomputed as soon as the departure day
        # is not that of the set — unless the equivalence of the two offers has been measured and declared
        # (`EQUIVALENCES.yaml` of the set, cf. `verifier-jours`). Walk, bike, car: served.
        jour_depart = wall_clock(int(departure_time)).date().isoformat()
        if jour_depart != self.jeu.jour_simule:
            # Checked in the GTFS per trip (`verifier-jours`, EQUIVALENCES.yaml): is the
            # timetable of this day identical within the window of the trip? True → served;
            # False → transit recomputed; None (never checked) → transit recomputed too, we assume nothing.
            valide = self.jeu.ligne_valide_le(jour_depart, ligne.cle)
            if valide is not True:
                for g in ("transit", "rail"):
                    if g in presents:
                        motifs[g] = f"offre_jour:{jour_depart}"
                self._jeu_stats[
                    "offre_jour_non_verifiee"
                    if valide is None
                    else "offre_jour_grille_differente"
                ] += 1
        return {
            "ligne": ligne,
            "propositions": propositions,
            "ecart_s": ecart,
            "a_recalculer": set(motifs),
            "motifs": motifs,
        }

    def _fusionner_recalcul(self, du_jeu: dict, recalculees: list) -> tuple[list, dict]:
        """G5/G6/G7 — groupes hors tolérance : recalculés ; autres : servis du jeu ; recalcul stérile compté."""
        a_recalculer = du_jeu["a_recalculer"]
        ecart_min = du_jeu["ecart_s"] // 60
        plans, sources = [], {}
        for prop in du_jeu["propositions"]:
            if self._groupe_tolerance(prop.plan) not in a_recalculer:
                plans.append(prop.plan)
                sources[id(prop.plan)] = prop.source
        motifs = du_jeu.get("motifs") or {}
        for it in recalculees or []:
            g = self._groupe_tolerance(it)
            if g in a_recalculer:
                plans.append(it)
                motif = motifs.get(g, "horaire")
                if motif.startswith("offre_jour"):
                    sources[id(it)] = (
                        f"{PREFIXE_RECALCUL_OFFRE_JOUR}:{motif.split(':', 1)[1]}"
                    )
                else:
                    tol = self.jeu_tolerances.get(g)
                    tol_txt = (
                        tol.type + (f"{tol.pas_min}min" if tol and tol.pas_min else "")
                        if tol
                        else "?"
                    )
                    sources[id(it)] = (
                        f"{PREFIXE_RECALCUL_HORAIRE}:{ecart_min:+d}/{tol_txt}"
                    )

        # G7 — recomputation without effect: same modes and same durations to the minute as the recorded one.
        def _signature(items):
            return sorted((pl.mode_label(), (pl.duration or 0) // 60) for pl in items)

        for g in a_recalculer:
            avant = [
                p.plan
                for p in du_jeu["propositions"]
                if self._groupe_tolerance(p.plan) == g
            ]
            apres = [
                it for it in (recalculees or []) if self._groupe_tolerance(it) == g
            ]
            cle = (
                "recalculs_offre_jour"
                if str(motifs.get(g, "")).startswith("offre_jour")
                else "recalculs_horaire"
            )
            self._jeu_stats[cle] += 1
            if _signature(avant) == _signature(apres):
                self._jeu_stats[
                    "recalcul_sans_effet"
                    if cle == "recalculs_horaire"
                    else "recalcul_offre_jour_sans_effet"
                ] += 1
        rec = self._jeu_stats["recalculs_horaire"]
        part = self._jeu_stats["recalcul_sans_effet"] / rec if rec else 0.0
        seuil = settings.data.jeu_seuil_recalcul_sans_effet
        if rec >= 20 and part > seuil and not self._jeu_alarme_sterile_on:
            self._jeu_alarme_sterile_on = True
            logger.error(
                f"[ALARME] Trip set {self.jeu.nom!r}: {self._jeu_stats['recalcul_sans_effet']}/{rec} hourly "
                f"recomputations without effect ({100 * part:.0f} % > {100 * seuil:.0f} %) — the hourly tolerance "
                f"is too sensitive (G7)"
            )
            fire_alarme("jeu_recalcul_sterile")
        elif self._jeu_alarme_sterile_on and part < seuil / 2:
            self._jeu_alarme_sterile_on = False
        return plans, sources

    def _ecrire_jeu_stats(self, force: bool = False) -> None:
        """`jeu_stats.json` in the run's workdir — read by `make report` (G3, G14)."""
        if self.jeu is None:
            return
        self._jeu_stats_depuis_ecriture += 1
        if not force and self._jeu_stats_depuis_ecriture < 25:
            return
        self._jeu_stats_depuis_ecriture = 0
        jamais = ["offre (événements non implémentés)"]
        if not self._jeu_stats["recalculs_horaire"]:
            jamais.append("horaire")
        if not self._jeu_stats["recalculs_offre_jour"]:
            jamais.append("offre_jour (aucun autre jour simulé que celui du jeu)")
        contenu = {
            "jeu": self.jeu.nom,
            "empreinte": self.jeu.empreinte,
            "population": self.jeu.population.get("nom"),
            "tolerances": {
                g: (t.type if t.type != "pas" else {"pas_min": t.pas_min})
                for g, t in self.jeu_tolerances.items()
            },
            **{k: int(v) for k, v in self._jeu_stats.items()},
            "propositions_par_source": {
                k: int(v) for k, v in self._jeu_propositions_par_source.items()
            },
            "declencheurs_jamais_declenches": jamais,
            "couverture_jeu": self.jeu.couverture(),
        }
        try:
            chemin = Path(settings.app.log_file).parent / "jeu_stats.json"
            tmp = chemin.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(contenu, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8",
            )
            tmp.replace(chemin)
        except OSError as e:
            logger.warning(f"[jeu] jeu_stats.json non écrit : {e}")

    async def _compute_move_for_activity(
        self,
        person: Person,
        next_activity: Activity,
        timestamp: int,
        from_location_override: Location | None = None,
        _pipeline_rec=None,
    ) -> tuple[PersonMove | None, str | None]:
        from_location = (
            from_location_override
            if from_location_override is not None
            else person.state.last_location
        )

        # scheduled_start_time = departure time from the previous activity toward next_activity.
        # Use it as the trip departure; arrive_by=False (depart at this time).
        departure_time, planning_late_s = _depart_du_trajet(next_activity, timestamp)
        # No trip starts at the weekend: a Saturday/Sunday departure is
        # postponed to the following Monday at the same time. The OTP itinerary and the schedule_at on the
        # GAMA side follow from departure_time.
        # ⚠ `expected_arrive_at` DOES NOT FOLLOW — checked on 2026-09-23 on gama_arrivals.csv:
        # schedule_at Friday 19:35, started_at Monday 19:15, expected_arrive_at left at
        # Friday 19:50. The arrival delay was therefore worth 71.6 h for a twelve-minute
        # trip. The postponement is removed at reading, in `retard_d_arrivee` (text_helper).
        if settings.agent.no_weekend_departures:
            shifted = shift_weekend_departure_to_monday(departure_time)
            if shifted != departure_time:
                # Rising-edge alarm (audit A6): the FIRST postponement means the run has
                # crossed a weekend. Each later postponement stays at info level — the event
                # that counts is the crossing, not its volume.
                if not self._weekend_shift_alarm_on:
                    self._weekend_shift_alarm_on = True
                    logger.error(
                        "[ALARME] Le run a franchi un week-end : un départ de samedi/dimanche "
                        "est reporté au lundi suivant (`agent.no_weekend_departures`). Les "
                        "départs reportés s'EMPILENT sur le lundi et en font un jour atypique, "
                        "alors que l'enquête de référence ne compte aucun week-end (jours 1 à 5). "
                        "Un run comparable à l'EMC² démarre un lundi et ne dépasse pas cinq "
                        "jours simulés (`simulation_max_days`). (Alarme émise une seule fois ; "
                        "chaque report reste journalisé en info.)"
                    )
                logger.info(
                    f"[weekend] Départ de {person.person_id} pour {next_activity.purpose} "
                    f"reporté de {humanize_date(departure_time)} à {humanize_date(shifted)} (lundi)"
                )
                departure_time = shifted
        # Exit lock: a vehicle can only be driven where it is parked — and the
        # car, only by someone who has the age and the licence (A2).
        # Ticket 035 (spec 02, RG-1): the exit lock is computed by the single
        # decision — here only so as not to ask OTP for what will be discarded; the full
        # filter (reasons, return lock, events) applies further down through `eligibilite`.
        _is_passenger = _is_car_passenger(person)
        _eligibles = modes_vehicules_eligibles(person, from_location)
        include_car = _eligibles["car"]
        include_bike = _eligibles["bike"]
        # Chain constraint applied to this trip, logged in moves.csv (A4).
        # A single value per line; `passager` takes precedence, then `retour_force`, then
        # `sortie_bloquee`. These lines stay in the scoring: the column explains,
        # it does not filter.
        chain_constraint = ""
        # Ticket 077, batch E5 — where the decision comes from: « cache », « direct », or empty
        # when no model was called (single itinerary, no trip).
        _origine_decision = ""
        # Anticipation (ticket 014) : construit seulement si la décision atteint le
        # LLM — les chemins cache/mono-option n'affichent aucun prompt.
        anticipation: dict | None = None

        same_location = (
            from_location is not None
            and next_activity.location is not None
            and from_location.lat == next_activity.location.lat
            and from_location.lon == next_activity.location.lon
        )
        # Source of each proposal (ticket 035, spec 04 G6): enregistree / locale /
        # recalculee:horaire / en_vol / hors_jeu — traced in moves.csv and jeu_stats.json.
        _sources: dict[int, str] = {}
        _jeu_servi = False
        if same_location:
            itineraries = []
        else:
            _du_jeu = (
                self._propositions_du_jeu(person, next_activity, departure_time)
                if self.jeu is not None
                else None
            )
            if _du_jeu is not None and not _du_jeu["a_recalculer"]:
                # G3 — régime nominal : servi du jeu, AUCUN appel moteur.
                itineraries = [prop.plan for prop in _du_jeu["propositions"]]
                _sources.update(
                    {id(prop.plan): prop.source for prop in _du_jeu["propositions"]}
                )
                _jeu_servi = True
                self._jeu_stats["deplacements_servis"] += 1
            else:
                _timing_sink: dict | None = {} if _pipeline_rec is not None else None
                if _pipeline_rec is not None:
                    _pipeline_rec.T_otp_start = time.time()
                itineraries = await self.trip_helper.get_itineraries(
                    origin=from_location,
                    destination=next_activity.location,
                    departure_time=departure_time,
                    include_car=include_car,
                    include_bike=include_bike,
                    arrive_by=False,
                    _timing_sink=_timing_sink,
                )
                self._jeu_stats["appels_moteur"] += 1
                if _timing_sink:
                    if _timing_sink.get("transit_end") is not None:
                        self._jeu_stats["appels_otp"] += 1
                    if _timing_sink.get("osmnx_end") is not None:
                        self._jeu_stats["appels_osmnx"] += 1
                if _pipeline_rec is not None:
                    _pipeline_rec.T_otp_end = time.time()
                    if _timing_sink:
                        _pipeline_rec.T_transit_sem = _timing_sink.get(
                            "transit_sem_end"
                        )
                        _pipeline_rec.T_transit_end = _timing_sink.get("transit_end")
                        _pipeline_rec.T_osmnx_sem = _timing_sink.get("osmnx_sem_end")
                        _pipeline_rec.T_osmnx_end = _timing_sink.get("osmnx_end")
                if _du_jeu is not None:
                    # G5 — recomputation limited to the groups out of tolerance; the others stay served from the set.
                    itineraries, _maj = self._fusionner_recalcul(_du_jeu, itineraries)
                    _sources.update(_maj)
                    _jeu_servi = True
                elif self.jeu is not None:
                    # Question 9: trip that the set does not cover (precomputation beyond the
                    # day, replanned activity) — legitimate, counted apart, never « enregistré ».
                    self._jeu_stats["hors_jeu"] += 1
                    _sources.update({id(it): SOURCE_HORS_JEU for it in itineraries})

        # Post-filter (OTP/OSMnx sometimes return a mode not requested): carried by
        # `eligibilite`, further down, with the return lock — a single implementation.

        # Synthetic school bus (ticket 030). Injected AFTER the post-filter (never
        # blocked: `_vehicle_mode` returns « transit ») and BEFORE the return lock, so
        # that a pupil who came by car takes the car again (the lock then filters out
        # the school bus), and that a pupil who came by school bus finds it again on the return
        # (no vehicle parked at school → lock inactive). Nothing in case of a non-trip.
        if not same_location and not _jeu_servi:
            school_option = build_school_bus_option(
                person=person,
                from_location=from_location,
                next_activity=next_activity,
                timestamp=timestamp,
                departure_time=departure_time,
            )
            if school_option is not None:
                itineraries = list(itineraries) + [school_option]
                _sources[id(school_option)] = SOURCE_LOCALE

        # Eligibility filter (ticket 035, spec 02): exit lock (post-filter of the modes
        # not requested), return lock at the 1 km threshold, gap reasons and events of
        # the `agent_vehicle_chain_total` metric — the SAME function as the mode without
        # simulator. The choice between the remaining modes stays with the decision-maker.
        _filtre = eligibilite(
            person,
            from_location,
            [
                Proposition(it, _sources.get(id(it), SOURCE_EN_VOL))
                for it in itineraries
            ],
            next_activity.purpose,
            next_activity.location,
        )
        for _mode, _event in _filtre.evenements:
            VEHICLE_CHAIN.labels(mode=_mode, event=_event).inc()
            if _event == "return_failed":
                # No itinerary in the mode of the vehicle (OTP silent, distance beyond
                # bike range…): we let go rather than block the agent — it
                # goes home by another mode and the vehicle becomes an orphan.
                logger.debug(
                    f"[vehicle] Retour au domicile impossible en {_mode} "
                    f"pour {person.person_id} — véhicule laissé sur place"
                )
        chain_constraint = _filtre.contrainte or chain_constraint
        _ecartees = list(_filtre.ecartees)
        itineraries = [prop.plan for prop in _filtre.eligibles]

        for itinerary in itineraries:
            itinerary.purpose = next_activity.purpose

        selection_method = "Undefined"
        provider_info = ""
        reasoning = ""
        faster_itinerary = None
        # Breakdown by mode estimated by the LLM before the draw — empty for the decisions
        # that produce none (single choice, no itinerary, LLM error).
        mode_probabilities: dict = {}

        if not itineraries:
            estimated_duration = _estimate_fallback_duration(
                from_location, next_activity.location
            )
            if same_location:
                selection_method = "Pas de déplacement (même localisation)"
                logger.debug(
                    f"[timestamp: {humanize_date(timestamp)}] Already at destination for {next_activity.purpose} "
                    f"(lat={from_location.lat:.5f},lon={from_location.lon:.5f}) — using fallback duration"
                )
            else:
                selection_method = "Pas de solution de déplacement"
                bbox = self.world_bbox

                def _in_bbox(loc) -> bool:
                    return (
                        loc is not None
                        and bbox.min_lat <= loc.lat <= bbox.max_lat
                        and bbox.min_lon <= loc.lon <= bbox.max_lon
                    )

                origin_ok = _in_bbox(from_location)
                dest_ok = _in_bbox(next_activity.location)
                logger.warning(
                    f"[timestamp: {humanize_date(timestamp)}] Can't get to destination {next_activity.location} by any transport mode, "
                    f"estimated travel time: {humanize_duration(estimated_duration)} | "
                    f"origin_in_bbox={origin_ok} (lat={from_location.lat:.5f},lon={from_location.lon:.5f}) "
                    f"dest_in_bbox={dest_ok}"
                )
            plan = TravelPlan(
                id=random_uuid(),
                start_location=from_location,
                end_location=next_activity.location,
                start_time=departure_time * 1000,
                end_time=(departure_time + estimated_duration) * 1000,
                purpose=next_activity.purpose,
                legs=[],
            )
            plan_index = 0
            reasoning = "Can't find a suitable public transport plan, walk to the destination anyway"
        else:
            plan_index = 0
            reasoning = "Hard to choice, just pick the first one"

            for itinerary in itineraries:
                if (
                    faster_itinerary is None
                    or itinerary.duration < faster_itinerary.duration
                ):
                    faster_itinerary = itinerary

            _retenues, _ecartees_plafond = plafonner(
                [
                    Proposition(it, _sources.get(id(it), SOURCE_EN_VOL))
                    for it in itineraries
                ],
                settings.gtfs.max_trip_candidates,
            )
            _ecartees.extend(_ecartees_plafond)
            itineraries = [prop.plan for prop in _retenues]

            if len(itineraries) == 1:
                reasoning = "Un seul itinéraire disponible, sélection automatique"
                selection_method = "Un seul itinéraire disponible"
                # Ticket 077, batch C — a trip without an alternative remains a TRIP that the
                # memory must know. It does not call the model, so it went through
                # neither of the two places that write a decision entry: 116 of the 514
                # trips of the run of 075 were invisible to the memory, and their arrivals
                # were attributed to the previous trip for lack of a matchable decision.
                if person.is_llm_based and self.agent:
                    self.agent.note_decision_contrainte(
                        Context(
                            person=person,
                            timestamp=timestamp,
                            activity_id=next_activity.id,
                            data={
                                "type": "travel_plan",
                                "contrainte_chaine": chain_constraint or "",
                            },
                        ),
                        itineraries[0],
                        next_activity.purpose,
                    )
            elif person.is_llm_based and self.agent:
                context = Context(
                    person=person,
                    timestamp=timestamp,
                    activity_id=next_activity.id,
                    # `contrainte_chaine` travels with the context (ticket 071, batch 1): it is
                    # here, and only here, that we know whether the agent had to give up a mode.
                    # The short-term memory entry written by the decision uses it for its
                    # deterministic severity.
                    data={
                        "type": "travel_plan",
                        "contrainte_chaine": chain_constraint or "",
                    },
                )
                EVALUATE_PLAN_CALLS.inc()
                if settings.agent.agenda_anticipation_enabled:
                    anticipation = _build_anticipation(
                        person, next_activity, departure_time
                    )
                _trace_decision: dict = {}
                (
                    plan_index,
                    reasoning,
                    provider_info,
                    mode_probabilities,
                ) = await self.agent.evaluate_and_choose_travel_plan(
                    context=context,
                    options=itineraries,
                    destination=next_activity.purpose,
                    departure_time=departure_time,
                    anticipation=anticipation,
                    trace=_trace_decision,
                )
                # A decision served by the cache writes no line in
                # `llm_exchanges.jsonl`: without this column, it cannot be told apart from a
                # direct call in `moves.csv`, and the count of calls saved is made
                # blindly.
                _origine_decision = (
                    "cache" if _trace_decision.get("cache") else "direct"
                )
                if isinstance(plan_index, int) and 0 <= plan_index < len(itineraries):
                    selection_method = "LLM"
                    # Ticket 105 — une décision prise casse la série.
                    self._replis_consecutifs = 0
                else:
                    # The counter stays in the trace to diagnose incidents, but
                    # an experiment stops at the first failure not absorbed by the gateway.
                    self._replis_consecutifs += 1

                    # Ticket 105 — the fallback was only at `debug` level, hence invisible: it took
                    # digging through `moves.csv` by hand on 2026-09-23 to discover that the measurement
                    # window carried 10.3% of them. A decision the model did not take is
                    # a measurement anomaly, it is logged as such.
                    logger.error(
                        f"[ALARME] [repli] décision absente ; index 0 interdit en expérience | "
                        f"agent={person.person_id} instant={humanize_date(timestamp)} "
                        f"destination={next_activity.location} "
                        f"genre_erreur={_trace_decision.get('genre_erreur') or 'aucun'} "
                        f"consecutifs={self._replis_consecutifs}"
                    )

                    # Ticket 077 axis 3, EXTENDED by ticket 105 — the run stops rather than
                    # let index 0 pass for a decision of the model. Two stop
                    # reasons, armed by the same experiment lock:
                    #   • daily quota — the provider says at what time it reopens;
                    #   • any other absence of decision after the gateway's retries —
                    #     immediate stop, because letting index 0 enter moves.csv would
                    #     already contaminate the state and the measurements.
                    if _arret_sur_repli_arme():
                        # 2026-09-25 — the provider overload (5xx, per-minute 429) now arrives
                        # QUALIFIED: the worker returns the batch before the client
                        # expires. It stops at the first failure, like the quota, instead
                        # of waiting for three fallbacks each of which entered the measurement.
                        _genre = _trace_decision.get("genre_erreur")
                        if _genre in ("quota_journalier", "surcharge_fournisseur"):
                            await self._declencher_hibernation_propre(
                                _trace_decision.get("reprise_a"),
                                person.person_id,
                                motif=_genre,
                            )
                            # SIGTERM requested; the pair respects the signature in case the
                            # coroutine gets control back before the process goes down.
                            return None, None
                        # 2026-09-29 — a refused mandatory replay (409 `rejeu_obligatoire_absent`)
                        # arrives here as an exception, WITHOUT a kind: it was filed as
                        # `decision_absente`, which the night chain retried hour after hour,
                        # whereas it recurs identically (control v6 of a13, day 11, three
                        # stops at the same instant). It is the common prefix that is broken.
                        _rejeu_refuse = getattr(
                            getattr(self.agent, "llm_client", None), "strict_replay_failure", None
                        )
                        if _rejeu_refuse:
                            await self._declencher_hibernation_propre(
                                None, person.person_id, motif="prefixe_commun",
                                details={"cause": _rejeu_refuse},
                                genre="rejeu_obligatoire_absent",
                            )
                        else:
                            await self._declencher_hibernation_propre(
                                None, person.person_id, motif="decision_absente", genre=_genre
                            )
                        return None, None
                    plan_index = 0
                    provider_info = ""
                    mode_probabilities = {}
                    selection_method = "LLM Error (Default index)"
                    # The model decided nothing: « direct », computed before knowing we were
                    # falling back, made index 0 pass for a served call (run of 2026-09-24).
                    _origine_decision = "repli"

            plan: TravelPlan = itineraries[plan_index]
            plan.purpose = next_activity.purpose

            # Journalisation Lot B (ticket 030) : car scolaire effectivement retenu.
            if is_school_bus_plan(plan):
                _sb_dir = (
                    "outbound"
                    if (next_activity.purpose or "").lower() == "education"
                    else "return"
                )
                SCHOOL_BUS_CHOSEN.labels(direction=_sb_dir).inc()
                logger.info(f"[school_bus] retenu ({_sb_dir}) par {person.person_id}")

        # Car trip kept by a non-driver: it is a passenger trip
        # (A2). The mode stays « Voiture Privée » in moves.csv — EMC² counts the
        # passenger in « voiture » — and the traceability goes through the
        # « Contrainte de chaîne » column.
        if plan is not None and _is_passenger and _vehicle_mode(plan) == "car":
            VEHICLE_CHAIN.labels(mode="car", event="passenger").inc()
            chain_constraint = "passager"

        # Chain consistency: the vehicle used follows the agent, the others stay parked
        # where they are. The « no trip » case is excluded — the agent has not moved,
        # nor have its vehicles.
        if not same_location and settings.agent.vehicle_chain_enabled:
            _park_vehicles(person, plan, from_location, next_activity.location)
            self._settle_vehicles_at_home(person, next_activity)

        plan_duration_s = (
            max(0, (plan.end_time - plan.start_time) // 1000) if plan is not None else 0
        )
        prepare_before_seconds = max(plan_duration_s, settings.world.time_step)
        # expected_arrive_at = actual arrival time from OTP (departure + trip duration)
        expected_arrive_at = (
            plan.end_time // 1000 if plan is not None else departure_time
        )

        move = PersonMove(
            id=random_uuid(),
            person_id=person.person_id,
            current_time=timestamp,
            expected_arrive_at=expected_arrive_at,
            prepare_before_seconds=prepare_before_seconds,
            purpose=next_activity.purpose,
            target_location=next_activity.location,
            for_activity=next_activity,
            plan=plan,
        )

        if _pipeline_rec is not None:
            _pipeline_rec.plan_selected_index = plan_index
            _pipeline_rec.selection_method = selection_method

        # Outcome of the decision → real-time metric (tracking of the activities degraded
        # for lack of an LLM answer, cf. cockpit ③). The bootstrap precomputes at /init: a
        # fallback there is NOT a missed activity (the agent will leave anyway) → separate phase.
        ACTIVITY_DECISIONS.labels(
            outcome=_SELECTION_OUTCOME.get(selection_method, "other"),
            phase="bootstrap" if self._in_bootstrap else "live",
        ).inc()

        # Ticket 077, batch C — the mode kept is noted HERE, where it is certain, and reread at
        # arrival. `mode_canonique` reads the leg label of the plan: it is the same
        # vocabulary as the axes of the memories.
        if plan is not None and next_activity.id:
            _mode_retenu = mode_canonique(plan.mode_label())
            if _mode_retenu:
                self._mode_par_activite[(person.person_id, str(next_activity.id))] = (
                    _mode_retenu
                )

        _weather = get_weather(timestamp)
        await MoveLogger.get_instance().log_move(
            person=person,
            plan=plan,
            purpose=next_activity.purpose,
            selection_method=selection_method,
            chain_constraint=chain_constraint,
            anticipation=(anticipation or {}).get("trace", ""),
            provider_model=provider_info or "",
            faster_itinerary=faster_itinerary,
            reasoning=reasoning,
            # Ticket 077, batches E2 and E5 — the index kept and the origin of the decision, for
            # EVERY decision and no longer only for those that went through the timing.
            selected_index=plan_index,
            origine_decision=_origine_decision,
            weather_temp=_weather["temperature"] if _weather else None,
            weather_condition=_weather["weather_label"] if _weather else None,
            weather_precip_mm=_weather["precip_mm"] if _weather else None,
            late_s=planning_late_s,
            move_id=move.id,
            simulated_time=timestamp,
            start_time=plan.start_time if plan is not None else None,
            available_options=itineraries,
            activity_id=next_activity.id,
            mode_probabilities=mode_probabilities,
            sources=_resume_sources(_sources, itineraries),
            ecartees=resumer_ecartees(_ecartees),
        )
        if self.jeu is not None:
            for it in itineraries:
                self._jeu_propositions_par_source[
                    _sources.get(id(it), SOURCE_EN_VOL).split(":")[0]
                ] += 1
            self._ecrire_jeu_stats()

        return move, reasoning
