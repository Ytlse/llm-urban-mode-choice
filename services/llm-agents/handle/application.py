"""
FastAPI application for LLM-GAMA integration.

This module provides the HTTP API and WebSocket communication layer between
the GAMA simulation and external LLM (Large Language Model) systems. It handles
world initialization, synchronization, and real-time observation/action exchange.
"""

import asyncio
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import orjson
import uvicorn
from backpressure import (
    backlog_alarm_transition,
    compute_backpressure_interval,
    departures_at_risk,
    edf_feasibility,
    hold_while,
    time_ewma,
    update_drain_mode,
)
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from gama_models import (
    GamaPersonData,
    MessageResponse,
    MessageType,
    WorldInitRequest,
    WorldInitResponse,
    WorldSyncRequest,
)
from handle.websocket import WebSocketClient
from helper import (
    format_sim_timing,
    humanize_date,
    setup_logging,
    to_timestamp_based_on_day,
)
from inputs.population.perimeter import (
    PopulationPerimeter,
    filter_population,
    load_population_perimeter,
    sealed_population_complete,
)
from llm_gateway.telemetry.alarms import fire_alarme
from loguru import logger
from models import Location
from population_utils import (
    ajuster_planning,
    fix_activities,
    merge_consecutive_activities,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from settings import settings
from sim_clock import to_network_datetime, wall_clock
from urban_mobility_agents.core.scenario import BaseScenario, Observation
from urban_mobility_agents.factory.factory import (
    init_dynamic_scenario,
    init_static_data,
)
from urban_mobility_agents.utils.pipeline_logger import PipelineLogger
from urban_mobility_agents.utils import filiation, identite_run, rejeu_decisions
from urban_mobility_agents.utils.routage import journal_du_routage
from urban_mobility_agents.utils.reprise import (
    point_de_reprise_timestamp,
    restaurer_si_demande,
)
from utils import create_background_task

# Counters of the controller endpoints
SYNC_REQUESTS = Counter(
    "controller_sync_requests_total", "Total requêtes /sync reçues de GAMA"
)
INIT_REQUESTS = Counter(
    "controller_init_requests_total", "Total requêtes /init reçues de GAMA"
)

# Métriques de la simulation
SIM_AGENTS_TOTAL = Gauge(
    "gama_sim_agents_total",
    "Nombre total d'agents dans la simulation (défini au /init)",
)
SIM_STEP_INTERVAL = Gauge(
    "gama_sim_step_interval_seconds",
    "Durée réelle entre deux pas de temps GAMA consécutifs (secondes)",
)
SIM_LOGICAL_TIME = Gauge(
    "gama_sim_logical_time_seconds",
    "Horodatage logique courant de la simulation (timestamp Unix GAMA)",
)
SIM_REAL_ELAPSED = Gauge(
    "gama_sim_real_elapsed_seconds",
    "Temps réel écoulé depuis le dernier /init (secondes)",
)
SIM_STEP_COUNT = Gauge(
    "gama_sim_step_count", "Numéro du pas de temps courant depuis le /init"
)
SIM_STEP_LOGICAL_DURATION = Gauge(
    "gama_sim_step_logical_duration_seconds",
    "Durée logique GAMA d'un pas de temps (écart entre deux timestamps consécutifs en secondes de temps simulé)",
)
AGENT_STATES = Gauge(
    "gama_agent_states", "Nombre d'agents par état (inactive/ready/active)", ["state"]
)
SIM_WALL_CLOCK_RATIO = Gauge(
    "sim_wall_clock_ratio",
    "Ratio temps simulé / temps réel entre deux /sync (accélération effective)",
)

# Cockpit — pilotage temps réel de la simulation
CTRL_BACKPRESSURE_INTERVAL = Gauge(
    "controller_backpressure_interval_seconds",
    "Frein backpressure appliqué à la dernière réponse /sync (secondes)",
)
CTRL_BACKLOG_FILL_RATIO = Gauge(
    "controller_backlog_fill_ratio",
    "Remplissage de la pile : activités à calculer / population (1.0 = pile pleine)",
)
CTRL_DRAIN_MODE = Gauge(
    "controller_drain_mode_active",
    "Mode drainage actif (1=/sync retenu jusqu'au vidage de la pile, 0=nominal)",
)
CTRL_AGENTS_STUCK = Gauge(
    "controller_agents_stuck",
    "Agents sans planification réussie depuis plus que le seuil (heures de simulation)",
)
CTRL_INIT_STAGE = Gauge(
    "controller_init_stage",
    "Étape courante de l'init /init: 0=idle,1=prépa population,2=population prête,3=scénario,4=bootstrap itinéraires,5=prêt",
)
CTRL_INIT_PROGRESS = Gauge(
    "controller_init_progress_ratio", "Progression globale de l'init (0..1 = étape/5)"
)

# Ticket 003 — ordonnancement EDF et contre-pression prédictive
CTRL_THROUGHPUT = Gauge(
    "controller_throughput_tasks_per_min",
    "Débit de complétion D (EWMA) × 60 — tâches OTP+LLM drainées par minute",
)
CTRL_T_ESTIMATE = Gauge(
    "controller_t_estimate_seconds",
    "Pire T_k du dernier test de faisabilité EDF (temps réel pour résoudre toute la file)",
)
CTRL_MIN_SLACK = Gauge(
    "controller_min_slack_sim_seconds",
    "Échéance la plus proche en file − temps sim courant (secondes de temps SIMULÉ)",
)
CTRL_PREDICTIVE_HOLD = Gauge(
    "controller_predictive_hold_seconds",
    "Rétention prédictive appliquée à la dernière réponse /sync (secondes)",
)
CTRL_EDF_QUEUE_DEPTH = Gauge(
    "controller_edf_queue_depth",
    "Profondeur de la file EDF (tâches en attente, hors tâches en cours d'exécution)",
)
CTRL_PENDING_REFLECTIONS = Gauge(
    "controller_pending_reflections",
    "Réflexions STM en file EDF ou en cours d'exécution (drainage nocturne, ticket 010)",
)
CTRL_OVERDUE_DECISIONS = Gauge(
    "controller_overdue_decisions",
    "Décisions d'itinéraire (plan/refill) en file EDF ou en cours dont l'heure de départ est dépassée — signal de vraie saturation",
)
CTRL_SYNC_DURATION = Histogram(
    "controller_sync_duration_seconds",
    "Durée de traitement d'une requête /sync (battement de cœur GAMA↔controller)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
)
CTRL_DEADLINE_MISSES = Counter(
    "controller_deadline_misses_total",
    "Décisions de départ rendues après l'heure du départ (temps simulé au retour de la décision) — expose late_since_last_sync",
)
# Retenue sur départ imminent (2026-09-25, expériences seulement)
CTRL_DEPARTURE_HOLD = Gauge(
    "controller_departure_hold_seconds",
    "Retenue /sync sur départ imminent appliquée à la dernière réponse (secondes réelles)",
)
CTRL_DEPARTURE_HOLDS = Counter(
    "controller_departure_holds_total",
    "Retenues /sync sur départ imminent, par issue : relachee (décision rendue), cap (budget épuisé, départ encore loin), arret (run arrêté)",
    ["issue"],
)
CTRL_DEPARTURES_AT_RISK = Gauge(
    "controller_departures_at_risk",
    "Décisions de départ en attente dont le départ tombe dans l'horizon de retenue",
)

_last_sync_wall_time: float = 0.0
_last_sync_response_wall_time: float = 0.0  # timestamp of the last /sync response sent
_sim_init_wall_time: float = 0.0
_sim_step_count: int = 0
_last_logical_time: int = 0
_last_backpressure_in_progress: int = (
    0  # in_progress_count used for the previous sync's sleep
)
_last_backpressure_min_interval: float = (
    0.0  # min_interval computed for the previous sync's sleep
)
_backlog_alarm_active: bool = (
    False  # évite de répéter l'alarme backlog à chaque sync (front montant)
)
_backlog_benign_logged: bool = (
    False  # front montant du log INFO « backlog bénin » (drainage nocturne, ticket 010)
)
_drain_mode_active: bool = False  # drain mode: /sync held as long as the stack is not emptied below the release threshold

# Ticket 003 — state of the predictive control (module-level, reset at /init)
_sim_ratio_ewma: float | None = (
    None  # R lissé : secondes simulées / seconde réelle (EWMA du SIM_WALL_CLOCK_RATIO)
)
_sim_ratio_ewma_time: float | None = None  # instant (wall) du dernier échantillon de R
_throttle_active: bool = False  # régime dégradé notifié à GAMA (topic system/throttle)
_throttle_last_notify_at: float = (
    0.0  # dernier envoi (wall) — rafraîchissement périodique sans spam
)

# eqasim service URL — set via EQASIM_SERVICE_URL env var (default: http://eqasim:8003)
_EQASIM_SERVICE_URL = os.environ.get("EQASIM_SERVICE_URL", "http://eqasim:8003")


async def _trigger_eqasim_generation(
    population_size: int, bbox: tuple[float, float, float, float] | None = None
) -> None:
    """Call the eqasim service to ensure the population JSON is ready.

    Blocks until generation completes (or returns immediately on cache hit).
    Timeout is 30 min to accommodate first-time synpp runs.
    Raises HTTPException on generation failure so /init surfaces the error
    to GAMA rather than crashing later on a missing population file.

    bbox: optional (min_lon, min_lat, max_lon, max_lat) in WGS84 — restricts
    synpp to the communes intersecting this zone so generated profiles stay
    within the simulation area.
    """
    url = f"{_EQASIM_SERVICE_URL}/generate"
    payload: dict = {"population_size": population_size}
    if bbox is not None:
        payload["bbox"] = list(bbox)
    logger.info(
        f"[eqasim] Triggering population generation via {url} (population_size={population_size}, bbox={bbox})"
    )
    try:
        async with httpx.AsyncClient(timeout=1800.0) as client:
            resp = await client.post(url, json=payload)
            body = resp.json()
            if resp.status_code == 200 and body.get("status") == "ok":
                logger.info(f"[eqasim] Population ready — {body.get('file', '')}")
            else:
                exit_code = body.get("exit_code", "?")
                msg = f"[eqasim] Generation failed (exit_code={exit_code}). Check eqasim container logs (OOM if exit_code=137)."
                logger.error(msg)
                raise HTTPException(status_code=503, detail=msg)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            f"[eqasim] Could not reach eqasim service ({exc}); will attempt to load existing file"
        )


def _find_population_json(population_size: int | None = None) -> str | None:
    """Return the path of the eqasim population JSON for the requested size.

    If population_size is given, looks for the exact file
    ``{prefix}population_{population_size}.json`` and returns None if absent.
    If population_size is None, falls back to the largest available file.
    """
    output_dir = settings.data.eqasim_output_dir
    prefix = settings.data.synthetic_file_prefix
    if population_size is not None:
        exact = os.path.join(output_dir, f"{prefix}population_{population_size}.json")
        return exact if os.path.exists(exact) else None
    pattern = re.compile(rf"^{re.escape(prefix)}population_(\d+)\.json$")
    candidates = [
        (int(m.group(1)), os.path.join(output_dir, name))
        for name in os.listdir(output_dir)
        if (m := pattern.match(name))
    ]
    return max(candidates)[1] if candidates else None


def _population_perimeter() -> PopulationPerimeter:
    """The study area of the 453 communes, or an [ALARME] and the exception — never a rectangle in its place."""
    try:
        return load_population_perimeter()
    except Exception as exc:
        logger.error(
            f"[ALARME] [population] Périmètre des 453 communes indisponible ({exc!r}) : le "
            "chargement refuse plutôt que de retomber sur un rectangle. Vérifiez "
            "mobility_core/data/couronne_perimetre.geojson et commune_couronne.json."
        )
        raise


async def _prepare_population(
    population_size: int,
    stop_coords: np.ndarray,
    sim_base_timestamp: int,
    perimeter: PopulationPerimeter | None,
) -> str | None:
    """Ensure workdir/population_{N}.json exists, contains exactly N enriched agents.

    The workdir file is written in eqasim format (not Pydantic) so that both the
    OSMnx route cache and world/population.py can consume it.  Writes are atomic
    (.tmp → rename) so a crash mid-write leaves no corrupted file.

    Idempotence: if workdir/population_{N}.json already exists in enriched eqasim
    format it is used as-is (no re-generation, no re-enrichment).

    Study area (ticket 031, part 2): `perimeter` — the 453 communes of the survey — filters the
    agents by **commune of residence** (`inputs/population/perimeter.py`); an activity outside the
    polygon is counted and raises an alarm but does not discard the agent; a sealed file is loaded whole
    or refused. Before: a 30 km rectangle that discarded the whole 3rd ring.
    """
    import random as _random

    from population_utils import _activity_index_pairs
    from population_utils import scheduling_mode as _sched_mode
    from trip_helper.osmnx_direct import get_direct_plan, init_persistent_cache

    workdir_path = f"{settings.data.population_cache_prefix}{population_size}.json"

    def _init_osmnx_cache() -> None:
        population_name = (
            f"{settings.data.synthetic_file_prefix}population_{population_size}"
        )
        if settings.gtfs.osmnx_cache_enabled:
            cache_dir = os.path.join(
                settings.gtfs.osmnx_persistent_cache_dir, population_name
            )
            init_persistent_cache(cache_dir)

        if settings.gtfs.mode == "OTP" and settings.gtfs.otp_cache_enabled:
            from trip_helper.cached_triphelper import init_otp_persistent_cache

            otp_cache_dir = os.path.join(
                settings.gtfs.otp_persistent_cache_dir, population_name
            )
            init_otp_persistent_cache(otp_cache_dir)

    def _is_eqasim_format(data: list) -> bool:
        # Eqasim identity has no "name" key; Pydantic PersonalIdentity dump does
        return bool(data) and "name" not in data[0].get("identity", {})

    # Init caches before any routing: Pass 2 (regeneration path) must read/write
    # the persistent OSMnx cache, otherwise every regeneration recomputes all routes.
    _init_osmnx_cache()

    data = None
    if os.path.exists(workdir_path):
        with open(workdir_path, encoding="utf-8") as f:
            data = json.load(f)
        if not _is_eqasim_format(data):
            logger.info(
                f"[population] File in old Pydantic format — regenerating: {workdir_path}"
            )
            data = None
        else:
            logger.info(f"[population] File exists — reusing: {workdir_path}")
            return workdir_path

    if data is None:
        sealed = settings.data.population_file
        if sealed:
            # SEALED population: designated explicitly, taken as is, never
            # regenerated. A missing file is a configuration error, not a case where
            # one would « go back » through eqasim — that would replace the seal silently.
            if not os.path.exists(sealed):
                logger.error(
                    f"[ALARME] [population] Population scellée introuvable : {sealed} "
                    "(data.population_file). Rien n'est généré à sa place : corrigez le "
                    "chemin, ou retirez le réglage pour revenir à la recherche par taille."
                )
                return None
            raw_json_path = sealed
            logger.info(
                f"[population] Population scellée : {sealed} (data.population_file)"
            )
        else:
            # Ensure raw eqasim output exists for the exact requested size
            raw_json_path = _find_population_json(population_size)
            if not raw_json_path:
                # No bbox: let eqasim generate the full Toulouse area pool.
                # Bbox filtering is applied below in Python after loading the raw data.
                await _trigger_eqasim_generation(population_size)
                raw_json_path = _find_population_json(population_size)
                if not raw_json_path:
                    logger.error(
                        "[population] No population file found after eqasim generation"
                    )
                    return None

        with open(raw_json_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        # STUDY-AREA filter then draw of exactly population_size (ticket 031, part 2).
        # The residence makes the study area: `household.commune_id` ∈ 453 communes (fallback: trait
        # `residence_zone`, then polygon geometry with alarm). An activity outside the polygon
        # (school, work outside the study area) does NOT discard the agent: it is counted and raises an alarm
        # above a threshold. Before that day, a 30 km rectangle discarded any agent whose
        # residence OR one activity was outside: 79 agents of v3, the whole 3rd ring of v4.
        perimeter_stats = None
        if perimeter is not None:
            raw_data, perimeter_stats = filter_population(
                raw_data, perimeter, source="population"
            )
        if sealed and not sealed_population_complete(
            sealed, len(raw_data), population_size, perimeter_stats
        ):
            # A seal is taken whole: the alarm is raised by sealed_population_complete.
            return None
        if population_size < len(raw_data):
            # Local seeded RNG (does not affect the global state of `random`): the same
            # subset of agents is drawn at each run → identical trips → OSMnx cache
            # reusable on replay.
            _rng = _random.Random(settings.data.population_sample_seed)
            raw_data = _rng.sample(raw_data, population_size)
        data = raw_data
        logger.info(f"[population] Selected {len(data)} agents from eqasim output")

        # Step 2: fix activity sequences + merge consecutive same-purpose/same-location
        fix_count = 0
        for i, person in enumerate(data):
            data[i], fixes = fix_activities(person)
            if fixes:
                fix_count += 1
        n_merged = merge_consecutive_activities(data)
        logger.info(
            f"[population] Step 2: {fix_count} persons fixed, {n_merged} activities merged"
        )

        # Pass 1: public_transport flag
        MAX_WALK_M = 1_500

        def _flag(lon: float, lat: float) -> bool:
            dlat = stop_coords[:, 0] - lat
            dlon = (stop_coords[:, 1] - lon) * math.cos(math.radians(lat))
            return float(np.hypot(dlat, dlon).min()) * 111_320 <= MAX_WALK_M

        pt_count = 0
        for entry in data:
            identity = entry.get("identity", {})
            home = identity.get("home")
            if home and home.get("lon") is not None:
                home["public_transport"] = _flag(home["lon"], home["lat"])
                pt_count += 1
            for act in identity.get("activities", []):
                loc = act.get("location")
                if loc and loc.get("lon") is not None:
                    loc["public_transport"] = _flag(loc["lon"], loc["lat"])
                    pt_count += 1
        logger.info(f"[population] {pt_count} locations flagged with public_transport")

    # Pass 2: compute scheduling-mode travel times (car for car owners, bicycle otherwise)
    # and adjust scheduled_start_time via ajuster_planning.
    # Always executed when generating a new file.
    # Build one scheduling-mode route per activity pair, then call ajuster_planning.
    tasks_meta: list[tuple] = []  # (person_index, prev_i, curr_i)
    coros = []
    for p_idx, entry in enumerate(data):
        activities = entry["identity"].get("activities", [])
        has_car = entry["identity"].get("traits_json", {}).get("number_of_cars", 0) > 0
        mode = _sched_mode(entry)

        for prev_i, curr_i, _ in _activity_index_pairs(activities, has_car):
            prev_act = activities[prev_i]
            curr_act = activities[curr_i]
            prev_loc = prev_act.get("location", {})
            curr_loc = curr_act.get("location", {})
            if not prev_loc or not curr_loc:
                continue
            if prev_loc.get("lon") is None or curr_loc.get("lon") is None:
                continue
            origin = Location(lat=prev_loc["lat"], lon=prev_loc["lon"])
            destination = Location(lat=curr_loc["lat"], lon=curr_loc["lon"])
            origin_end = prev_act.get("end_time") or 0
            departure_unix = to_timestamp_based_on_day(
                int(origin_end)
                if origin_end > 0
                else int(curr_act.get("start_time", 0)),
                sim_base_timestamp,
            )
            # Same clock as the runtime: the timestamp comes from GAMA, so its
            # time is WALL-CLOCK and is read in the time zone of the network, not in that of the
            # process (`sim_clock`). Without it, the pass that adjusts the timetables of
            # the population priced its congestion one hour later than the
            # itineraries these timetables are used to request.
            congestion_dt = to_network_datetime(departure_unix)
            tasks_meta.append((p_idx, prev_i, curr_i))
            coros.append(
                get_direct_plan(
                    origin=origin,
                    destination=destination,
                    trip_mode=mode,
                    departure_time=departure_unix,
                    congestion_dt=congestion_dt,
                )
            )

    logger.info(
        f"[population] Pass 2 OSMnx scheduling: {len(coros)} routes ({len(data)} agents)…"
    )

    BATCH = 200
    results = []
    for i in range(0, len(coros), BATCH):
        batch_results = await asyncio.gather(
            *coros[i : i + BATCH], return_exceptions=True
        )
        results.extend(batch_results)
        logger.info(
            f"[population] Pass 2: {min(i + BATCH, len(coros))}/{len(coros)} routes calculées"
        )

    # Build per-person travel_times dict from results
    person_travel_times: list[dict] = [{} for _ in data]
    osmnx_ok = 0
    for (p_idx, prev_i, curr_i), result in zip(tasks_meta, results):
        if isinstance(result, Exception) or result is None:
            continue
        person_travel_times[p_idx][(prev_i, curr_i)] = result.duration
        osmnx_ok += 1
    logger.info(
        f"[population] Pass 2: {osmnx_ok}/{len(tasks_meta)} scheduling routes computed"
    )

    # Adjust schedules
    sched_errors = 0
    for p_idx, entry in enumerate(data):
        acts = entry.get("identity", {}).get("activities", [])
        try:
            entry["identity"]["activities"] = ajuster_planning(
                workdir_path,
                entry.get("person_id", "?"),
                acts,
                travel_times=person_travel_times[p_idx],
                raise_error=True,
            )
        except ValueError as exc:
            sched_errors += 1
            logger.warning(f"[population] ajuster_planning: {exc}")
    if sched_errors:
        logger.warning(
            f"[population] Step 5: {sched_errors} person(s) with unresolved schedule conflicts"
        )
    else:
        logger.info("[population] Step 5: all schedules adjusted successfully")

    tmp_path = workdir_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.rename(tmp_path, workdir_path)
    logger.info(
        f"[population] Population written to {workdir_path} ({len(data)} agents)"
    )

    return workdir_path


# Set working directory from environment if specified
workdir = os.environ.get("APP_WORKDIR", "")
if workdir:
    settings.update_workdir(workdir)

# Initialize logging
setup_logging(settings)

# Create FastAPI application instance
# ORJSONResponse is the default response class: it computes Content-Length
# and serialises the body with the SAME serialiser (orjson), avoiding the desynchronisation
# that caused "fixed content-length: X, bytes received: Y" on the Java side.
app = FastAPI(default_response_class=ORJSONResponse)


class LoopContainer:
    """
    Container for managing WebSocket communication and message loops.

    This class handles the bidirectional communication between the FastAPI server
    and the GAMA simulation via WebSocket. It manages observation publishing and
    action message handling.
    """

    action_topic = "action/data"
    system_greeting_topic = "system/greeting"
    observation_topic = "observation/data"
    system_log_topic = "system/log"
    system_throttle_topic = "system/throttle"

    def __init__(self):
        self.client = None
        self.scenario = None
        self._worker_task: asyncio.Task | None = None
        # Retry buffer of publish_loop: instance attribute (and not a local variable)
        # so that it can be purged on scenario replacement — otherwise the actions of
        # the old simulation left pending (dead socket at GAMA stop) would be
        # replayed to the new simulation on reconnection.
        self._pending: list = []
        # Initialize WebSocket client for GAMA communication
        self.websocket_client = WebSocketClient(settings.server.gama_ws_url)
        self.websocket_client.on_message = self.handle_message

    def set_scenario(self, scenario: BaseScenario):
        """Set the active simulation scenario, inject push_fn and start the Worker."""
        # Cancel the previous Worker if still running (e.g. successive /test/init calls).
        # stop_worker() cancels the actual _worker_loop of the previous scenario;
        # _worker_task (the start_worker wrapper) has already finished at this stage.
        if self.scenario is not None:
            try:
                self.scenario.stop_worker()
            except Exception as e:
                logger.warning(f"Failed to stop previous scenario worker: {e}")
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

        # Purge the actions of the old scenario still waiting to be sent:
        # the person_ids being identical from one run to the next (same population, same
        # seed), these stale trips would be injected into the new simulation.
        if self._pending:
            logger.warning(
                f"[set_scenario] {len(self._pending)} pending action(s) from previous "
                f"scenario discarded (would have been replayed to the new simulation)"
            )
            self._pending.clear()

        self.scenario = scenario

        # Inject the direct WebSocket push function into the scenario.
        # The boolean of send_json is propagated: send_message swallows the sending
        # exceptions and returns False (dead socket, reconnection in progress) — without this
        # return, _push_planned_move believed the push delivered and the agent became
        # a zombie (GAMA without a trip, Python waiting for an impossible arrival).
        async def _direct_push(action) -> bool:
            return await self.websocket_client.send_json(
                {
                    "topic": self.action_topic,
                    "payload": action.model_dump(),
                }
            )

        scenario.set_push_fn(_direct_push)
        self._worker_task = asyncio.create_task(scenario.start_worker())

    async def greeting(self):
        """Send a greeting message to the WebSocket server"""
        await self.websocket_client.connect()

        greeting_message = {
            "topic": self.system_greeting_topic,
            "payload": {
                "type": "greeting",
                "message": "Hello from FastAPI + WebSocket client!",
            },
        }
        success = await self.websocket_client.send_json(greeting_message)
        if not success:
            logger.error("Failed to send greeting message")

    async def send_log(self, message: str):
        """Envoie un message de progression à GAMA via WebSocket"""
        if self.websocket_client:
            await self.websocket_client.send_json(
                {"topic": self.system_log_topic, "payload": {"message": message}}
            )

    async def send_throttle(self, payload: dict):
        """Notifies GAMA of the backpressure regime (topic system/throttle, ticket 003).

        Same channel as send_log: the `message` field stays self-contained (a GAMA that does not
        handle the topic can treat it as a log). The other fields (rates,
        backlog) feed the UI / the globals of the experiment."""
        if self.websocket_client:
            await self.websocket_client.send_json(
                {
                    "topic": self.system_throttle_topic,
                    "payload": payload,
                }
            )

    async def publish_loop(self):
        """
        Main publishing loop that sends action messages to GAMA via WebSocket.

        Continuously checks for new messages from the scenario and publishes them
        to the GAMA simulation. Handles connection failures and retries.

        `self._pending` survives unexpected exceptions (e.g. model_dump) — the
        unsent messages stay in the buffer and are retried at the next
        iteration; the buffer is purged by set_scenario on scenario change.
        """
        while True:
            try:
                # Only fetch new messages when the pending buffer is empty.
                if (
                    not self._pending
                    and self.scenario
                    and await self.scenario.has_messages()
                ):
                    self._pending = await self.scenario.pop_all_messages()

                sent = 0
                while self._pending:
                    message = self._pending[0]
                    payload = message.model_dump()
                    success = await self.websocket_client.send_json(
                        {
                            "topic": self.action_topic,
                            "payload": payload,
                        }
                    )
                    if not success:
                        # WebSocket not ready — keep in buffer, retry next tick
                        logger.warning(
                            f"WebSocket not connected, will retry {len(self._pending)} pending message(s)"
                        )
                        break
                    # set_scenario may have purged the buffer during the await of the send
                    if self._pending:
                        self._pending.pop(0)
                    sent += 1
                    _pl = PipelineLogger.get()
                    if _pl is not None:
                        _person_id = payload.get("person_id")
                        if _person_id:
                            _pl.complete(_person_id)

                if sent > 0:
                    logger.info(
                        f"WebSocket loop sent {sent} message(s) to {self.action_topic}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket publish loop error: {e}")

            await asyncio.sleep(1)  # Adjust sleep time as needed

    async def handle_message(self, text: str):
        """Handle received Websocket message"""
        try:
            # logger.debug(f"Received: {self.observation_topic} -> {text}")
            await self.process_observation(self.observation_topic, text)

        except Exception as e:
            logger.exception(f"Error handling message: {e}")

    async def process_observation(self, topic: str, payload: str):
        """
        Process observation data received from GAMA simulation.

        Parses the observation payload and forwards it to the scenario for processing.
        Observations contain agent state information for LLM decision making.
        """
        try:
            data = json.loads(payload)
            assert data["topic"] == self.observation_topic, (
                "Invalid topic in observation data"
            )
            observation = Observation(**data["payload"])
            await self.scenario.handle_observation(observation)
        except Exception as e:
            logger.exception(f"Error processing observation: {e}")


class _AgentStateLog:
    """Append-only CSV recording agent state counts per simulation step."""

    _HEADERS = [
        "step",
        "sim_timestamp",
        "sim_time",
        "inactive",
        "ready",
        "active",
        "total",
    ]

    def __init__(self):
        self._path: Path | None = None
        self._initialized = False

    def _ensure_file(self) -> Path:
        if self._path is None:
            self._path = settings.workdir / "gama_results" / "agent_states.csv"
            self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._initialized:
            self._initialized = True
            if not self._path.exists():
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self._HEADERS)
        return self._path

    def record(
        self, step: int, sim_timestamp: int, inactive: int, ready: int, active: int
    ):
        path = self._ensure_file()
        # GAMA WALL-CLOCK time (`sim_clock`): this column is read next to
        # `sim_timestamp` and the controller logs, which carry the same time.
        sim_time = (
            wall_clock(sim_timestamp).strftime("%H:%M:%S") if sim_timestamp > 0 else ""
        )
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    step,
                    sim_timestamp,
                    sim_time,
                    inactive,
                    ready,
                    active,
                    inactive + ready + active,
                ]
            )


_agent_state_log = _AgentStateLog()

# Global loop container instance
loop_container = LoopContainer()
# Initialisation of the static data (GTFS loading, OTP...) at server startup
static_data = init_static_data()
print("===> Données statiques initialisées. En attente de la requête /init de GAMA...")

EVENT_LOOP_LAG = Gauge(
    "controller_event_loop_lag_seconds",
    "Retard maximal observé de l'event loop asyncio sur la dernière fenêtre (stall = la loop n'a pas repris la main à temps)",
)


async def _event_loop_lag_monitor(
    tick: float = 1.0, alarm_threshold: float = 5.0
) -> None:
    """Measures the blockings of the event loop (drift of a periodic sleep).

    A stall > alarm_threshold is traced as ERROR with [ALARME]: it is what
    makes the WebSocket keepalive expire (1006 cut → lost pushes) when it
    exceeds the ping_timeout. The trace gives the instant of recovery — the culprit
    is the operation that has just given control back right before.
    """
    loop = asyncio.get_running_loop()
    while True:
        t0 = loop.time()
        await asyncio.sleep(tick)
        lag = loop.time() - t0 - tick
        EVENT_LOOP_LAG.set(max(0.0, lag))
        if lag > alarm_threshold:
            fire_alarme("event_loop")
            logger.error(
                f"[ALARME] Event loop bloquée pendant ~{lag:.1f}s — opération synchrone "
                f"dans la boucle asyncio (calcul CPU, I/O bloquante ?). Au-delà du "
                f"ping_timeout WebSocket, ces stalls provoquent les coupures 1006."
            )


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.

    Initializes WebSocket connection and starts background tasks for
    real-time communication with GAMA simulation.
    """
    # Ticket 075 — the restoration happens HERE, at the very beginning: the vector index of the memory
    # is opened at the construction of the agent (at `/init`), and replacing its files underneath it
    # would leave a database open on bytes that no longer exist. Without a resume requested,
    # this call does nothing.
    def _empreinte_choc_courante() -> str:
        """L'empreinte du choc déclaré, ou vide. Un choc déplacé change l'expérience."""
        try:
            from llm import chocs as _chocs

            reg = _chocs.registre()
            return getattr(getattr(reg, "choc", None), "empreinte", "") or ""
        except Exception:  # noqa: BLE001 — the identity must never prevent a startup
            return ""

    # Ticket 091 — by default we reuse NOTHING. The resume is requested by naming the run, and
    # only takes place if the identity of the named run describes the same experiment as the current
    # configuration. A missing, unreadable or different identity makes the startup fail by
    # naming what is wrong: reusing the memory of another experiment would produce a wrong
    # measurement that nothing would flag.
    _workdir = Path(settings.workdir)
    _reprise = os.environ.get("REPRISE_RUN", "").strip()

    # Ticket 095, batch C — the per-function routing is read at startup. A binding read
    # nowhere is a binding discovered after the campaign.
    logger.info(f"[routage] instances admises — {journal_du_routage()}")

    # Ticket 095, batch D — CHILD RUN. The common base is played once, at the parent's; each
    # arm inherits it. The lineage is checked BEFORE writing anything: a child that
    # differs from its parent on an undeclared field would inherit a memory produced under
    # other settings, and nothing in the outputs would say so.
    _parent_nom = filiation.parent_declare()
    _parent_dir: Path | None = None
    _champs_libres: tuple[str, ...] = ()
    if _parent_nom:
        _champs_libres = filiation.champs_libres()
    _identite = identite_run.composer(
        settings,
        empreinte_choc=_empreinte_choc_courante(),
        run_parent=_parent_nom,
        champs_libres=_champs_libres,
    )
    if _parent_nom and not _reprise:
        _parent_dir = filiation.repertoire_parent(_parent_nom, _workdir.parent)
        filiation.amorcer(_workdir, _parent_dir, _identite)

    if _reprise:
        identite_run.verifier(_workdir, _identite)
        logger.info(f"[identite] reprise de {_workdir.name} : même expérience, vérifiée champ à champ")
    else:
        identite_run.ecrire(_workdir, _identite)

    # Ticket 090 — the decision trace is OPENED on every run, not only on a resume:
    # it is during normal life that it is written. Opening it only at the resume left it empty,
    # hence inert — defect found on 2026-09-17 on the run 2026-09-17_06_26, zero line traced
    # after two simulated days.
    rejeu_decisions.charger(_workdir)

    # A child restores the point inherited from its parent like an ordinary resume: it is the
    # same freezing mechanism, and the replay of the common days therefore rewrites no memory.
    restaurer_si_demande(
        _workdir,
        reprise_demandee=bool(_reprise) or bool(_parent_dir),
        alarme_si_absent=True,
    )
    # The decision trace (ticket 090) is loaded ONLY on a named and checked resume:
    # without that it would become a cache nobody asked for.
    if _reprise:
        from urban_mobility_agents.utils.reprise import point_de_reprise_timestamp

        rejeu_decisions.charger(_workdir, jusqu_a=point_de_reprise_timestamp())
    await loop_container.greeting()
    create_background_task(loop_container.websocket_client.run_with_reconnect())
    create_background_task(loop_container.publish_loop())
    create_background_task(_event_loop_lag_monitor())


@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event handler - closes WebSocket connections."""
    await loop_container.websocket_client.stop()
    _pl = PipelineLogger.get()
    if _pl is not None:
        _pl.close()


@app.get(
    "/",
    summary="Vérifier le statut du contrôleur",
    description="Vérifie si l'API du contrôleur de simulation (FastAPI) est bien démarrée et en attente de la connexion WebSocket avec GAMA.",
    tags=["Système"],
)
async def root():
    """Root endpoint - returns service status."""
    return {"status": "FastAPI + Websocket running"}


@app.get(
    "/metrics",
    summary="Exporter les métriques Prometheus",
    description="Expose les compteurs d'événements de la simulation GAMA (appels, synchronisations) au format Prometheus.",
    tags=["Système"],
)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/init",
    summary="Initialiser la population du monde",
    description=(
        "Génère et renvoie la liste complète de la population synthétique (avec les coordonnées des domiciles et les caractéristiques des agents) "
        "pour peupler la carte GAMA au lancement de la simulation. "
        "Bloque jusqu'à ce que tous les premiers itinéraires soient calculés (bootstrap), "
        "de sorte que GAMA ne commence pas à avancer avant que chaque agent ait son premier trajet en file."
    ),
    tags=["Simulation"],
)
async def init(request: WorldInitRequest):
    """
    Initialize the simulation world and pre-compute all first itineraries.

    GAMA's reflex init blocks on this HTTP call, so the simulation cannot
    advance until bootstrap_all_agents completes and every agent has a move queued.
    """
    if settings.world.prefixe_commun != bool(request.prefixe_commun):
        raise HTTPException(
            status_code=409,
            detail=(
                "Préfixe commun incohérent entre GAMA et contrôleur : "
                f"GAMA={request.prefixe_commun}, contrôleur={settings.world.prefixe_commun}"
            ),
        )
    INIT_REQUESTS.inc()
    _N_STEPS = 5
    _t_init_start = time.time()

    def _init_stage(step: int) -> None:
        CTRL_INIT_STAGE.set(step)
        CTRL_INIT_PROGRESS.set(step / _N_STEPS)

    _init_stage(1)

    logger.info(
        format_sim_timing("SIM_START", sim_time=humanize_date(request.timestamp))
    )
    logger.info(
        f"INITIALISATION 1/{_N_STEPS} Préparation de la population — sim_time={humanize_date(request.timestamp)}"
    )
    await loop_container.send_log(
        f"[1/{_N_STEPS}] Préparation de la population (génération + enrichissement)..."
    )

    effective_population_size = request.population_size or settings.data.population_size
    stops_df = static_data.gtfs_data.stops[["stop_lat", "stop_lon"]].dropna()
    stop_coords = stops_df.values.astype(float)
    population_json_path = await _prepare_population(
        population_size=effective_population_size,
        stop_coords=stop_coords,
        sim_base_timestamp=request.timestamp,
        perimeter=_population_perimeter(),
    )
    if population_json_path is None:
        raise RuntimeError(
            f"[population] Impossible de préparer la population ({effective_population_size} agents) — "
            "vérifier les logs eqasim et le répertoire eqasim-output."
        )

    _init_stage(2)
    logger.info(
        f"INITIALISATION 2/{_N_STEPS} Population prête — {population_json_path}"
    )
    await loop_container.send_log(f"[2/{_N_STEPS}] Population prête.")

    # Ticket 075 — a simulation that (re)starts replays from its t0. If a resume
    # point exists, the memory on disk is LATER than it: it belongs to an interrupted
    # run, and the replayed days would rewrite it. We bring it back to the point, and freeze
    # until the simulated clock passes it. On a new run, there is no point and
    # this call does nothing. It is here and not at the controller startup: `CONT=1` only
    # restarts GAMA, the `controller` container, for its part, does not go through its startup again.
    _point_restaure = restaurer_si_demande(settings.workdir, reprise_demandee=True)
    rejeu_decisions.charger(settings.workdir, jusqu_a=point_de_reprise_timestamp())

    _init_stage(3)
    logger.info(
        f"INITIALISATION 3/{_N_STEPS} Initialisation du scénario et chargement des agents"
    )
    await loop_container.send_log(f"[3/{_N_STEPS}] Initialisation du scénario...")

    scenario = init_dynamic_scenario(
        static_data,
        sim_base_timestamp=request.timestamp,
        population_size=request.population_size,
        part_of_llm_agents=request.part_of_llm_based_agents
        if request.part_of_llm_based_agents is not None
        else 1.0,
        long_term_memory_enabled=request.long_term_memory_enabled,
        long_term_self_reflect_enabled=request.long_term_self_reflect_enabled,
        simulation_max_days=request.simulation_max_days,
        accidents_enabled=request.accidents_enabled,
    )
    loop_container.set_scenario(scenario)

    # Ticket 118, O1 — the household state of the restored point is reread HERE, after the factory: it
    # resets the household by indexing the population. Reread earlier, it was erased.
    from llm import foyer as _foyer

    _foyer.restaurer_depuis_point(_point_restaure)

    # Ticket 035 (spec 04) — designated recorded trip set (`make run JEU=<name>`):
    # served instead of the engines. Refusal BEFORE any computation if the set is not that of the
    # loaded population or if the time tolerances are not declared (G1, G5).
    if settings.data.jeu_enregistre:
        from experiences.population import info_population

        try:
            _info_pop = info_population(
                settings.data.population_file or population_json_path
            )
            scenario.charger_jeu(settings.data.jeu_enregistre, _info_pop)
        except Exception as e:
            logger.error(
                f"[ALARME] [jeu] Jeu enregistré {settings.data.jeu_enregistre!r} refusé : {e}"
            )
            raise RuntimeError(f"[jeu] jeu enregistré refusé : {e}") from e
        await loop_container.send_log(
            f"[3/{_N_STEPS}] Jeu enregistré {scenario.jeu.nom!r} chargé — régime nominal sans appel moteur."
        )

    if settings.app.pipeline_log_enabled:
        from pathlib import Path

        PipelineLogger.init(Path(settings.app.pipeline_log_file))
        logger.info(f"Pipeline timing log enabled → {settings.app.pipeline_log_file}")

    _init_stage(4)
    logger.info(
        f"INITIALISATION 4/{_N_STEPS} Pré-calcul du premier itinéraire pour chaque agent (bootstrap)"
    )
    await loop_container.send_log(
        f"[4/{_N_STEPS}] Pré-calcul des premiers itinéraires..."
    )

    if request.timestamp > 0:
        await scenario.bootstrap_all_agents(timestamp=request.timestamp)

    people = scenario.population.get_people_list()
    SIM_AGENTS_TOTAL.set(len(people))
    global _sim_init_wall_time, _sim_step_count, _last_logical_time
    global \
        _sim_ratio_ewma, \
        _sim_ratio_ewma_time, \
        _throttle_active, \
        _throttle_last_notify_at
    _sim_init_wall_time = time.time()
    _sim_step_count = 0
    _last_logical_time = max(0, request.timestamp)
    # Reset the state of the predictive control (ticket 003) for a clean run.
    _sim_ratio_ewma = None
    _sim_ratio_ewma_time = None
    _throttle_active = False
    _throttle_last_notify_at = 0.0
    SIM_STEP_COUNT.set(0)
    SIM_REAL_ELAPSED.set(0)
    if request.timestamp > 0:
        SIM_LOGICAL_TIME.set(request.timestamp)
    for _state in ("inactive", "ready", "active"):
        AGENT_STATES.labels(state=_state).set(0)
    AGENT_STATES.labels(state="inactive").set(len(people))

    _init_stage(5)
    _init_duration = time.time() - _t_init_start
    from urban_mobility_agents.simulation_controller import _format_cache_hit_rates

    _cache_str = _format_cache_hit_rates()
    logger.info(
        format_sim_timing(
            "INIT_DONE", agents=len(people), init_duration_s=f"{_init_duration:.0f}"
        )
    )
    logger.info(
        f"INITIALISATION 5/{_N_STEPS} Démarrage de la simulation — {len(people)} agents envoyés à GAMA (init={_init_duration:.0f}s)"
    )
    await loop_container.send_log(
        f"[5/{_N_STEPS}] Démarrage de la simulation — {len(people)} agents | init: {_init_duration:.0f}s | caches: {_cache_str}"
    )

    person_response = [
        GamaPersonData(
            **person.model_dump(),
            location=scenario.population.get_person_home_location(person.person_id),
            name=person.identity.name,
        )
        for person in people
    ]
    return MessageResponse(
        message_type=MessageType.AG_WORLD_INIT,
        data=WorldInitResponse(
            people=person_response,
            num_people=len(people),
            # TODO: remove this
            timestamp=0,
        ),
    )


@app.post(
    "/test/init",
    summary="[TEST] Initialiser le scénario sans GAMA",
    description=(
        "Initialise le scénario directement depuis le fichier de population, sans connexion WebSocket GAMA. "
        "Réservé aux tests de charge et de performance. Ne lance pas le bootstrap des itinéraires."
    ),
    tags=["Test"],
)
async def test_init(population_size: int = None, timestamp: int = 1_775_800_000):
    """Initialize the scenario from the population file without a GAMA WebSocket connection."""
    if population_size is None:
        # Fallback to the largest available file; do NOT use settings (can default to 1)
        largest = _find_population_json(population_size=None)
        if largest is None:
            raise HTTPException(
                status_code=404,
                detail="No population file found. Pass population_size explicitly.",
            )
        m = re.search(r"population_(\d+)\.json$", largest)
        effective_size = int(m.group(1)) if m else settings.data.population_size
    else:
        effective_size = population_size

    stops_df = static_data.gtfs_data.stops[["stop_lat", "stop_lon"]].dropna()
    stop_coords = stops_df.values.astype(float)

    population_json_path = await _prepare_population(
        population_size=effective_size,
        stop_coords=stop_coords,
        sim_base_timestamp=timestamp,
        perimeter=_population_perimeter(),
    )
    if population_json_path is None:
        return MessageResponse(success=False, error="Population preparation failed")

    scenario = init_dynamic_scenario(
        static_data,
        sim_base_timestamp=timestamp,
        population_size=effective_size,
        long_term_memory_enabled=False,
        long_term_self_reflect_enabled=False,
    )
    loop_container.set_scenario(scenario)

    people = scenario.population.get_people_list()
    logger.info(
        f"[test/init] Scénario initialisé — {len(people)} agents, timestamp={timestamp}"
    )
    return MessageResponse(
        success=True,
        data={"num_people": len(people), "population_file": str(population_json_path)},
    )


@app.get(
    "/test/queue_depth",
    summary="[TEST] Nombre d'agents en cours de scheduling",
    tags=["Test"],
)
async def queue_depth():
    """Return the number of agents currently waiting for an LLM/OTP response."""
    if not loop_container.scenario:
        return {"scheduling_in_progress": 0, "total_agents": 0}
    people = loop_container.scenario.population.get_people_list()
    in_progress = sum(1 for p in people if p.state.scheduling_in_progress)
    return {"scheduling_in_progress": in_progress, "total_agents": len(people)}


@app.post(
    "/reflect",
    summary="Déclencher la réflexion forcée des agents",
    description=(
        "Force tous les agents de la simulation à mettre à jour leur état cognitif (réflexion sur leur mémoire) "
        "pour correspondre au timestamp fourni. Utilisé principalement pour le débogage ou la synchronisation manuelle."
    ),
    tags=["Simulation"],
)
async def reflect(request: WorldSyncRequest):
    """
    Reflect the current world state at a specific timestamp.

    Forces all agents to update their state to match the simulation time.
    Used for synchronization and debugging.
    """
    logger.info(f"Reflecting world at timestamp: {request.timestamp}")

    if loop_container.scenario:
        await loop_container.scenario.trigger_short_term_reflection_for_all(
            request.timestamp
        )
        return MessageResponse(
            data="reflected",
            success=True,
        )
    else:
        logger.debug(
            "[/reflect] Scenario not ready yet — init still in progress, skipping"
        )
        return MessageResponse(data="not_ready", success=True)


# Digest de capacité poussé à GAMA (topic system/log) tous les N /sync : signaux
# « live » cheaply available en-process (cache LLM, débit, backlog, activité agents).
_DIGEST_EVERY_N_SYNC = 10


def _build_capacity_digest(
    step: int,
    inactive: int,
    ready: int,
    active: int,
    in_progress: int,
    throughput_per_s: float,
) -> str:
    """Ligne synthétique de capacité (≈ pendant live de scripts/debug/llm_capacity.py)."""
    total = inactive + ready + active
    d_min = throughput_per_s * 60.0
    try:
        from llm.cache import get_llm_cache_stats

        hits, lookups = get_llm_cache_stats()
        cache_str = f"{100 * hits // lookups}% ({hits}/{lookups})" if lookups else "off"
    except Exception:
        cache_str = "n/a"
    fill = in_progress / max(1, total) if total else 0.0
    return (
        f"📊 [cycle {step}] cache LLM {cache_str} · débit {d_min:.0f} req/min · "
        f"backlog {in_progress} ({fill:.0%}) · agents {active} actifs / "
        f"{inactive} inactifs / {total} total"
    )


@app.post(
    "/sync",
    summary="Synchroniser l'état du monde",
    description=(
        "Met à jour l'état du scénario côté Python avec les données de la population inactive (`idle_people`) envoyées par GAMA. "
        "Le contrôleur lit le corps de la requête en texte brut pour contourner les éventuels problèmes de header HTTP/2 (h2c)."
    ),
    tags=["Simulation"],
)
async def sync(raw: Request):
    """Point d'entrée /sync : mesure la durée totale de traitement puis délègue."""
    with CTRL_SYNC_DURATION.time():
        return await _sync_impl(raw)


async def _sync_impl(raw: Request):
    """
    Synchronize the world state with idle population data.

    Reads the raw body to remain compatible with GAMA's Java HTTP client,
    which sends h2c upgrade headers that prevent uvicorn/h11 from reading
    the body. hypercorn handles h2c natively, so the body is always available.
    """
    global \
        _last_sync_wall_time, \
        _last_sync_response_wall_time, \
        _sim_step_count, \
        _last_logical_time, \
        _last_backpressure_in_progress, \
        _last_backpressure_min_interval, \
        _backlog_alarm_active, \
        _backlog_benign_logged, \
        _drain_mode_active
    global \
        _sim_ratio_ewma, \
        _sim_ratio_ewma_time, \
        _throttle_active, \
        _throttle_last_notify_at
    now = time.time()
    real_delta = now - _last_sync_wall_time if _last_sync_wall_time > 0 else 0.0
    if real_delta > 0:
        SIM_STEP_INTERVAL.set(real_delta)
    _last_sync_wall_time = now
    _sim_step_count += 1
    SIM_STEP_COUNT.set(_sim_step_count)
    if _sim_init_wall_time > 0:
        SIM_REAL_ELAPSED.set(now - _sim_init_wall_time)

    SYNC_REQUESTS.inc()
    body = await raw.body()

    if not body:
        logger.warning("[/sync] Empty body received — sync skipped (unknown timestamp)")
        return MessageResponse(data="skipped (empty body)", success=True)

    try:
        data = orjson.loads(body)
        request = WorldSyncRequest(**data)
    except Exception as e:
        logger.error(f"[/sync] JSON parsing error: {e}")
        return ORJSONResponse(status_code=422, content={"detail": str(e)})
    _t_parse_end = time.time()

    logger.info(
        f"Synchronizing world at timestamp: {request.timestamp} ({humanize_date(request.timestamp)})"
    )

    if loop_container.scenario:
        try:
            ready = request.ready_count
            active = request.active_count
            inactive = request.inactive_count
            AGENT_STATES.labels(state="inactive").set(inactive)
            AGENT_STATES.labels(state="ready").set(ready)
            AGENT_STATES.labels(state="active").set(active)
            # Écriture CSV déportée hors de l'event loop (open/write bloquants)
            await asyncio.to_thread(
                _agent_state_log.record,
                _sim_step_count,
                request.timestamp,
                inactive,
                ready,
                active,
            )
        except Exception:
            pass
        in_progress_before_sync = loop_container.scenario.activities_to_compute_count
        # Departures served late (ticket 003, recounted on 2026-09-25): emptied by sync() → read them before.
        # Kept in _late_before_sync: it is also one of the two signals of real
        # saturation of the backlog alarm (ticket 010, A2).
        _late_before_sync = loop_container.scenario.late_since_last_sync
        if settings.world.prefixe_commun:
            scenario = loop_container.scenario
            _deadline = time.monotonic() + settings.world.prefixe_commun_sync_timeout_s

            def _accuse(etat: str) -> MessageResponse:
                return MessageResponse(
                    message_type=MessageType.AG_SYNC,
                    data=f"{etat}:{request.timestamp}", success=True,
                )

            # The response may be « pending ». GAMA asks again for the SAME timestamp;
            # synchroniser_prefixe never reruns scenario.sync() for this step.
            if not await scenario.synchroniser_prefixe(
                request.timestamp, _deadline, t_sync=now, t_parse=_t_parse_end,
            ):
                return _accuse("pending")
            _last_logical_time = request.timestamp
            _last_sync_response_wall_time = time.time()
            return _accuse("synchronized")
        try:
            CTRL_DEADLINE_MISSES.inc(_late_before_sync)
        except Exception:
            pass
        await loop_container.scenario.sync(
            request.timestamp, _t_sync=now, _t_parse=_t_parse_end
        )
        # Advance of GAMA between two /sync (SIMULATED time): measured below, failing that the declared
        # step. It is what GAMA will cross if the response is returned now.
        _pas_sync_sim = float(settings.world.time_step)
        try:
            SIM_LOGICAL_TIME.set(request.timestamp)
            if _last_logical_time > 0 and request.timestamp > _last_logical_time:
                sim_delta = request.timestamp - _last_logical_time
                _pas_sync_sim = float(sim_delta)
                SIM_STEP_LOGICAL_DURATION.set(sim_delta)
                if real_delta > 0:
                    _ratio = sim_delta / real_delta
                    SIM_WALL_CLOCK_RATIO.set(_ratio)
                    # R: smoothed sim/real rhythm (time EWMA), used by the EDF feasibility
                    # test. The raw SIM_WALL_CLOCK_RATIO is noisy from one /sync
                    # to the next (variable brake); the smoothing stabilises the estimated slack.
                    _sim_ratio_ewma = time_ewma(
                        _sim_ratio_ewma,
                        _sim_ratio_ewma_time,
                        _ratio,
                        now,
                        tau_s=settings.world.throughput_ewma_tau_s,
                    )
                    _sim_ratio_ewma_time = now
        except Exception:
            pass
        _last_logical_time = request.timestamp
        in_progress_count = loop_container.scenario.activities_to_compute_count
        if real_delta > 5.0:
            await loop_container.send_log(
                f"[⚠ sync lent] {real_delta:.1f}s depuis le dernier sync — "
                f"tâches utilisées pour le sleep précédent : {_last_backpressure_in_progress} "
                f"(sleep calculé : {_last_backpressure_min_interval:.2f}s)"
            )
        _k = settings.world.min_internal_coeff_k
        _cap = settings.world.min_internal_coeff_cap
        # `sans_frein` is worth the SIMULTANEOUS capacity of the gateway: below it, all the
        # requests are in flight at the same time and the stack does not grow — there is nothing to
        # brake. Without this threshold, a one-agent run took the maximum brake (30 s) at each
        # decision, the 1/1 ratio being worth 1 (c3 campaign of 2026-09-22: 12 min per simulated day).
        min_interval = compute_backpressure_interval(
            in_progress_count,
            settings.data.population_size,
            k=_k,
            cap=_cap,
            sans_frein=settings.world.worker_concurrency,
        )
        logger.info(
            f"Activités à calculer: {in_progress_count} — applying min_interval={min_interval:.2f}s"
        )

        # Capacity digest pushed to GAMA every _DIGEST_EVERY_N_SYNC cycles.
        # Any error is swallowed: the digest must never make a /sync fail.
        if _sim_step_count % _DIGEST_EVERY_N_SYNC == 0:
            try:
                await loop_container.send_log(
                    _build_capacity_digest(
                        step=_sim_step_count,
                        inactive=request.inactive_count,
                        ready=request.ready_count,
                        active=request.active_count,
                        in_progress=in_progress_count,
                        throughput_per_s=loop_container.scenario.throughput_per_s(),
                    )
                )
            except Exception as e:
                logger.debug(f"[digest] non envoyé: {e}")

        # Backlog alarm: aligned on the thresholds of the drain mode (rising edge at
        # drain_trigger_ratio, re-armed when the backlog falls back below drain_release_ratio).
        # Ticket 010 (A2): the threshold alone is no longer enough — a backlog of STM reflections
        # or of pre-plannings with distant deadlines during the simulated night is the
        # nominal drain (run 2026-08-03: 803 tasks, late=0, cache 99%, healthy
        # providers). The ERROR is only emitted if decisions actually suffer:
        # late departures (late_since_last_sync) or plan/refill deadlines exceeded.
        _backlog_ratio = in_progress_count / max(1, settings.data.population_size)
        CTRL_BACKLOG_FILL_RATIO.set(_backlog_ratio)
        _n_reflections = loop_container.scenario.pending_reflections_count
        _n_overdue = loop_container.scenario.overdue_decision_count(request.timestamp)
        CTRL_PENDING_REFLECTIONS.set(_n_reflections)
        CTRL_OVERDUE_DECISIONS.set(_n_overdue)
        _composition = (
            f"{in_progress_count} décisions d'itinéraire (dont {_n_overdue} à échéance dépassée) "
            f"+ {_n_reflections} réflexions STM"
        )
        _transition = backlog_alarm_transition(
            _backlog_alarm_active,
            _backlog_ratio,
            settings.world.drain_trigger_ratio,
            settings.world.drain_release_ratio,
            late_count=_late_before_sync,
            overdue_decisions=_n_overdue,
        )
        if _transition == "fire":
            _backlog_alarm_active = True
            fire_alarme("backlog")
            _alarm_msg = (
                f"[ALARME] Backlog critique : {_composition} "
                f"({_backlog_ratio:.0%} de la population), late_since_last_sync={_late_before_sync} "
                f"— des décisions d'itinéraire sont en souffrance. "
                f"Backpressure actuel min_interval={min_interval:.2f}s "
                f"(coeffs k={_k} cap={_cap}). "
                f"Vérifier les rate limits providers (make error) et le taux de hit du cache LLM."
            )
            logger.error(_alarm_msg)
            await loop_container.send_log(f"⛔ {_alarm_msg}")
        elif _transition == "release":
            _backlog_alarm_active = False
            logger.info(
                f"[ALARME levée] Backlog résorbé : {in_progress_count} activités en attente "
                f"({_backlog_ratio:.0%} de la population)"
            )
        elif _transition == "benign" and not _backlog_benign_logged:
            _backlog_benign_logged = True
            logger.info(
                f"[backlog] Pile à {_backlog_ratio:.0%} sans urgence en souffrance — "
                f"composition : {_composition}, late_since_last_sync=0. "
                f"Drainage nominal (nuit simulée) : la file se vide dans l'ordre des réveils."
            )
        if _backlog_ratio < settings.world.drain_release_ratio:
            _backlog_benign_logged = False
        # Drain mode (hysteresis, on top of the progressive brake): as soon as the stack
        # reaches drain_trigger_ratio, the /sync response is held up to the cap
        # (hard limit: the HTTP read timeout of the GAMA client, a response cannot
        # be held any longer) and the stack is resampled at a
        # short interval to give control back as soon as it is emptied below
        # drain_release_ratio. As long as this threshold is not reached, each following
        # /sync is held in turn: GAMA is throttled to ~1 step per cap.
        _was_draining = _drain_mode_active
        _drain_mode_active = update_drain_mode(
            _drain_mode_active,
            _backlog_ratio,
            settings.world.drain_trigger_ratio,
            settings.world.drain_release_ratio,
        )
        if _drain_mode_active and not _was_draining:
            logger.warning(
                f"[drain] Pile à {_backlog_ratio:.0%} (≥ {settings.world.drain_trigger_ratio:.0%}) : "
                f"les réponses /sync sont retenues jusqu'à {_cap:.0f}s tant que la pile "
                f"n'est pas redescendue sous {settings.world.drain_release_ratio:.0%}"
            )
        # --- Predictive observation (ticket 003): computed at each /sync, even when
        # the predictive control is disabled (observation phase to calibrate tau/margin). ---
        _D = (
            loop_container.scenario.throughput_per_s()
        )  # débit de complétion (tâches/s, EWMA)
        _R = (
            _sim_ratio_ewma if _sim_ratio_ewma is not None else 0.0
        )  # rythme sim/réel lissé
        _deadlines = loop_container.scenario.edf_snapshot_deadlines()
        _feas = edf_feasibility(
            _deadlines, request.timestamp, _D, _R, settings.world.predictive_margin
        )
        CTRL_THROUGHPUT.set(_D * 60.0)
        CTRL_T_ESTIMATE.set(_feas.t_estimate_s)
        CTRL_MIN_SLACK.set(_feas.min_slack_sim_s)
        CTRL_EDF_QUEUE_DEPTH.set(loop_container.scenario.edf_queue_depth)

        # --- Hold on imminent departure (2026-09-25) — EXPERIMENTS ONLY ---
        # As long as a departure decision has not been returned and this departure falls within
        # the `departure_hold_lookahead_s` horizon, the response is held (at most `cap`:
        # HTTP read timeout of GAMA), then held again at the next /sync. None of the
        # brakes below sees a decision being run — on 2026-09-24, a
        # Google 503 held it ~70 s while GAMA raced from 14:30 to 18:15. If, the budget
        # exhausted, releasing GAMA would make it pass the time of a departure still pending,
        # the run stops (hibernation `decision_en_retard`): an experiment does not serve a
        # late trip. Silent in an ordinary run (experiment lock absent).
        _departure_hold_s = 0.0
        _risque_depart = []
        if loop_container.scenario.arret_experience_arme:
            _horizon_depart = settings.world.departure_hold_lookahead_s

            def _departs_a_risque():
                return departures_at_risk(
                    loop_container.scenario.pending_departures(),
                    request.timestamp,
                    _horizon_depart,
                )

            _risque_depart = _departs_a_risque()
            if _risque_depart:
                _tete = _risque_depart[0]
                logger.info(
                    f"[depart] /sync retenu à {humanize_date(request.timestamp)} : "
                    f"{len(_risque_depart)} décision(s) de départ en attente dans l'horizon "
                    f"de {_horizon_depart / 60:.0f} min — la plus urgente : "
                    f"person={_tete.person_id} activité={_tete.purpose} ({_tete.activity_id}) "
                    f"départ {humanize_date(int(_tete.departure_sim))}, demandée à "
                    f"{humanize_date(int(_tete.requested_sim))}"
                )
                _departure_hold_s = await hold_while(
                    lambda: bool(_departs_a_risque()),
                    budget_s=_cap,
                    poll_s=settings.world.drain_poll_interval,
                )
                loop_container.scenario.noter_retenue_depart(_departure_hold_s)
                _restants = _departs_a_risque()
                # What GAMA would cross by the next /sync if it were released.
                _franchis = departures_at_risk(
                    _restants, request.timestamp, _pas_sync_sim
                )
                if not _restants:
                    CTRL_DEPARTURE_HOLDS.labels(issue="relachee").inc()
                    logger.info(
                        f"[depart] décision(s) rendue(s) avant le départ — main rendue à "
                        f"GAMA après {_departure_hold_s:.1f}s"
                    )
                elif _franchis:
                    CTRL_DEPARTURE_HOLDS.labels(issue="arret").inc()
                    CTRL_DEPARTURE_HOLD.set(_departure_hold_s)
                    await loop_container.scenario.arreter_pour_decision_en_retard(
                        _franchis[0], request.timestamp
                    )
                    return MessageResponse(data="arret_experience", success=True)
                else:
                    CTRL_DEPARTURE_HOLDS.labels(issue="cap").inc()
                    logger.info(
                        f"[depart] budget de {_cap:.0f}s épuisé, {len(_restants)} décision(s) "
                        f"encore attendue(s), départ le plus proche "
                        f"{humanize_date(int(_restants[0].departure_sim))} — GAMA avance "
                        f"d'un pas, nouvelle retenue au prochain /sync"
                    )
        CTRL_DEPARTURES_AT_RISK.set(len(_risque_depart))
        CTRL_DEPARTURE_HOLD.set(_departure_hold_s)
        # A response never exceeds `cap`: the following holds only have the remainder.
        _budget_retenue = max(0.0, _cap - _departure_hold_s)

        _predictive_hold_s = 0.0
        if _drain_mode_active:
            # Ultimate safety net (permanent overload > 100% utilisation):
            # freeze by hysteresis ratio, unchanged. Takes precedence over the predictive one.
            _drain_start = time.time()
            _live_ratio = _backlog_ratio
            while _drain_mode_active and (time.time() - _drain_start) < _budget_retenue:
                await asyncio.sleep(settings.world.drain_poll_interval)
                _live_count = loop_container.scenario.activities_to_compute_count
                _live_ratio = _live_count / max(1, settings.data.population_size)
                _drain_mode_active = update_drain_mode(
                    True,
                    _live_ratio,
                    settings.world.drain_trigger_ratio,
                    settings.world.drain_release_ratio,
                )
            if _drain_mode_active:
                logger.warning(
                    f"[drain] Cap {_cap:.0f}s atteint, pile encore à {_live_ratio:.0%} — "
                    f"GAMA sera retenu à nouveau au prochain /sync"
                )
            else:
                logger.info(
                    f"[drain] Pile redescendue à {_live_ratio:.0%} (< {settings.world.drain_release_ratio:.0%}) — "
                    f"main rendue à GAMA après {time.time() - _drain_start:.1f}s"
                )
            min_interval = _cap
        elif (
            settings.world.predictive_backpressure_enabled
            and settings.world.edf_enabled
        ):
            # Predictive hold: the /sync is only held if the EDF feasibility test
            # announces a threatened deadline. Requires the EDF dispatcher (the test reads
            # a snapshot of the heap): without EDF, we fall back on the progressive brake below.
            # Otherwise → immediate response (the progressive brake cap·ratio^k is short-circuited).
            # Holding the /sync freezes the simulated time (hence the deadlines): what unblocks
            # the queue is the drain by the consumers (T_k = k/D goes down) — not
            # the advance of time.
            # R is frozen at entry: during the hold the measured rhythm collapses
            # (no /sync served), it must not feed back into the exit condition.
            _R_frozen = _R
            _live_feas = _feas
            if _feas.hold and _R_frozen > 0:
                _hold_start = time.time()
                while _live_feas.hold and (time.time() - _hold_start) < _budget_retenue:
                    await asyncio.sleep(settings.world.drain_poll_interval)
                    _live_D = loop_container.scenario.throughput_per_s()
                    _live_deadlines = loop_container.scenario.edf_snapshot_deadlines()
                    _live_feas = edf_feasibility(
                        _live_deadlines,
                        request.timestamp,
                        _live_D,
                        _R_frozen,
                        settings.world.predictive_margin,
                    )
                _predictive_hold_s = time.time() - _hold_start
                logger.info(
                    f"[predictive] /sync retenu {_predictive_hold_s:.1f}s "
                    f"(D={_D * 60:.1f}/min, R×{_R_frozen:.1f}, file={len(_deadlines)}, "
                    f"T≈{_feas.t_estimate_s:.0f}s, slack_min={_feas.min_slack_sim_s:.0f}s sim) — "
                    f"{'relâché' if not _live_feas.hold else 'cap atteint'}"
                )
            min_interval = _predictive_hold_s
        elif min_interval > 0 and _last_sync_response_wall_time > 0:
            remaining = min_interval - (time.time() - _last_sync_response_wall_time)
            if remaining > 0:
                await asyncio.sleep(remaining)
        CTRL_PREDICTIVE_HOLD.set(_predictive_hold_s)

        # --- Notification GAMA (topic system/throttle, hystérésis, ticket 003) ---
        _now_wall = time.time()
        _backlog_now = loop_container.scenario.activities_to_compute_count
        if _predictive_hold_s >= settings.world.throttle_notify_threshold_s:
            _need_notify = (
                not _throttle_active
                or (_now_wall - _throttle_last_notify_at)
                >= settings.world.throttle_notify_refresh_s
            )
            if _need_notify:
                _msg = (
                    f"⏳ Simulation bridée : débit LLM {_D * 60:.1f}/min, vitesse sim ×{_R:.1f}, "
                    f"{_backlog_now} tâches en file (T≈{_feas.t_estimate_s / 60:.1f}min)"
                )
                await loop_container.send_throttle(
                    {
                        "active": True,
                        "llm_rate_per_min": round(_D * 60.0, 1),
                        "sim_ratio": round(_R, 1),
                        "backlog": _backlog_now,
                        "t_estimate_s": round(_feas.t_estimate_s, 1),
                        "min_slack_sim_s": round(_feas.min_slack_sim_s, 1),
                        "message": _msg,
                    }
                )
                _throttle_active = True
                _throttle_last_notify_at = _now_wall
        elif _throttle_active:
            # Front descendant : premier /sync servi sans rétention → levée.
            await loop_container.send_throttle(
                {
                    "active": False,
                    "llm_rate_per_min": round(_D * 60.0, 1),
                    "sim_ratio": round(_R, 1),
                    "backlog": _backlog_now,
                    "t_estimate_s": round(_feas.t_estimate_s, 1),
                    "min_slack_sim_s": round(_feas.min_slack_sim_s, 1),
                    "message": "✅ Simulation rétablie : plus de bridage prédictif.",
                }
            )
            _throttle_active = False

        _last_backpressure_in_progress = in_progress_count
        _last_backpressure_min_interval = min_interval
        _last_sync_response_wall_time = time.time()
        CTRL_BACKPRESSURE_INTERVAL.set(min_interval)
        CTRL_DRAIN_MODE.set(1 if _drain_mode_active else 0)
        try:
            _stuck_threshold_s = settings.world.stuck_agent_threshold_hours * 3600
            CTRL_AGENTS_STUCK.set(
                loop_container.scenario.count_stuck_agents(
                    request.timestamp, _stuck_threshold_s
                )
            )
        except Exception:
            pass
        return MessageResponse(data="synchronized", success=True)
    else:
        logger.debug(
            "[/sync] Scenario not ready yet — init still in progress, skipping"
        )
        return MessageResponse(data="not_ready", success=True)


# ── Accident posé à la main (ticket 070, travail F) ──────────────────────────


@app.post(
    "/accidents",
    summary="Poser un accident sur un axe",
    description=(
        "Pose un accident CHOISI sur l'arête du graphe la plus proche d'un point, pour la "
        "durée demandée. Même mécanisme et même monde que les accidents tirés au sort : "
        "seul le déclenchement diffère. C'est de cette pose que viennent les figures — le "
        "tirage aléatoire, à fréquence réelle, ne touche que ~0,6 déplacement par journée "
        "simulée à 1 000 agents et ne peut rien montrer."
    ),
    tags=["Accidents"],
)
async def poser_accident(raw: Request):
    """`{lat, lon, debut_ts, duree_minutes}` → the accident placed, or a reasoned refusal."""
    from trip_helper import accidents as accidents_module

    # Body read raw (GAMA HTTP client / h2c), like /sync and /calibrate.
    body = await raw.body()
    if not body:
        return {"success": False, "error": "corps vide — lat, lon, debut_ts, duree_minutes attendus"}
    try:
        payload = orjson.loads(body)
        lat = float(payload["lat"])
        lon = float(payload["lon"])
        debut_ts = int(payload["debut_ts"])
        duree_minutes = int(payload.get("duree_minutes", 45))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[ALARME] /accidents : charge utile illisible ({exc!r}) — {body[:200]!r}")
        return {"success": False, "error": f"charge utile illisible : {exc}"}

    registre = accidents_module.registre()
    if registre is None:
        # Explicit refusal rather than silence: without it, the experimenter would believe they had
        # placed an accident in a run where the regime is unticked.
        logger.error(
            "[ALARME] /accidents : régime d'accidents DÉSACTIVÉ pour ce run — pose refusée. "
            "Cochez « Accidents sur les axes » dans l'IHM GAMA avant de lancer."
        )
        return {"success": False, "error": "régime d'accidents désactivé pour ce run"}

    accident = registre.poser_manuellement(lat, lon, debut_ts, duree_minutes)
    if accident is None:
        return {"success": False, "error": "pose refusée — voir les alarmes du journal"}
    return {
        "success": True,
        "arete": list(accident.arete),
        "debut_ts": accident.debut_ts,
        "duree_s": accident.duree_s,
        "classe_vitesse": accident.classe_vitesse,
    }


# ── Prompt calibration (triggered from the GAMA UI) ─────────────────────
# Path of the standalone prompt_calibration repository mounted in the container (see
# docker-compose, volume ./prompt_calibration:/app/prompt_calibration). The
# calibration reads its paths relative to this folder (frozen sets, store) and
# through config/gama_container.yaml for prompts.yaml / cerema_values.yaml.
CALIBRATION_DIR = Path("/app/prompt_calibration")

# Handle of the calibration subprocess in progress (only one at a time), and
# lock to make the check-then-launch sequence atomic (without
# it, two concurrent /calibrate requests can both pass the
# "already running" check and each launch their campaign on the same store).
_calibration_proc: subprocess.Popen | None = None
_calibration_lock = asyncio.Lock()


@app.post(
    "/calibrate",
    summary="Lancer la calibration du prompt",
    description=(
        "Démarre une campagne de calibration du prompt système `itinary_multi_agent` "
        "en tâche de fond, avec un nombre de cycles (itérations de la boucle) paramétrable. "
        "L'appel est non bloquant : GAMA reçoit immédiatement l'accusé de démarrage, la "
        "calibration se poursuit dans le conteneur `controller`. Un seul run à la fois."
    ),
    tags=["Calibration"],
)
async def calibrate(raw: Request):
    """Lance `python -m calibration.cli run --iterations N` en sous-processus détaché."""
    global _calibration_proc

    # Body read raw (GAMA HTTP client / h2c), tolerant of an empty body.
    body = await raw.body()
    iterations = 20
    if body:
        try:
            payload = orjson.loads(body)
            iterations = int(payload.get("iterations", iterations))
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "message_type": "calibration_error",
                "error": f"Corps de requête invalide : {exc}",
            }

    if iterations < 1:
        return {
            "success": False,
            "message_type": "calibration_error",
            "error": "Le nombre de cycles doit être >= 1",
        }

    if not CALIBRATION_DIR.exists():
        msg = (
            f"Package de calibration introuvable ({CALIBRATION_DIR}). "
            f"Vérifier le montage ./prompt_calibration:/app/prompt_calibration "
            f"dans docker-compose.yml."
        )
        logger.error(f"[ALARME] /calibrate : {msg}")
        return {"success": False, "message_type": "calibration_error", "error": msg}

    async with _calibration_lock:
        if _calibration_proc is not None and _calibration_proc.poll() is None:
            logger.warning(
                "[/calibrate] Une calibration est déjà en cours — requête ignorée"
            )
            return {
                "success": False,
                "message_type": "calibration_busy",
                "error": "Une calibration est déjà en cours.",
                "data": {"pid": _calibration_proc.pid},
            }

        log_dir = Path("/app/experiments/current")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "calibration.log"
        cmd = [
            sys.executable,
            "-m",
            "calibration.cli",
            "run",
            "--iterations",
            str(iterations),
        ]
        # Config adaptée à la disposition du conteneur (paquets sous /opt, etc.).
        container_config = CALIBRATION_DIR / "config" / "gama_container.yaml"
        if container_config.exists():
            cmd[3:3] = ["--config", str(container_config)]

        log_file = open(log_path, "a", encoding="utf-8")
        try:
            log_file.write(
                f"\n{'=' * 70}\n[{datetime.now().isoformat()}] "
                f"Calibration lancée depuis GAMA — {iterations} cycle(s)\n{'=' * 70}\n"
            )
            log_file.flush()
            _calibration_proc = subprocess.Popen(
                cmd,
                cwd=str(CALIBRATION_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[ALARME] /calibrate : échec du lancement du sous-processus : {exc}"
            )
            return {
                "success": False,
                "message_type": "calibration_error",
                "error": str(exc),
            }
        finally:
            # The subprocess receives its own copy of the descriptor at fork;
            # the parent must close its own so as not to keep it open
            # for the whole (potentially long) lifetime of the run.
            log_file.close()

    logger.info(
        f"[/calibrate] Calibration démarrée (pid={_calibration_proc.pid}, "
        f"cycles={iterations}) — journal : {log_path}"
    )
    return {
        "success": True,
        "message_type": "calibration_started",
        "data": {
            "pid": _calibration_proc.pid,
            "iterations": iterations,
            "log": str(log_path),
        },
    }


if __name__ == "__main__":
    """
    Main entry point for running the FastAPI application.

    Starts the server on host 0.0.0.0 and port 8000.
    This provides the HTTP API for LLM-GAMA integration.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)
