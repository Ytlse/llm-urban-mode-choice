"""
OSMnx direct route planner (foot, bicycle, car).

Optimised for high-frequency async use:
  - Graphs loaded once into a process-wide singleton (disk cache survives restarts).
  - CPU-bound routing dispatched to per-mode ProcessPoolExecutors (bypasses GIL).
  - Each worker process loads only its mode's graph once, then reuses it.
  - OTP-compatible timestamps (Unix seconds) and leg structure.
"""

import asyncio
import functools
import gc
import itertools
import math
import multiprocessing as _mp
import os
import pickle
import sys as _sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import geopandas as gpd
import osmnx as ox
import pandas as pd
import yaml
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram
from shapely.geometry import Point

from models import Location, Transit, TransitLocation, TravelPlan
from geography import (PERIMETER_CACHE_KEY, PERIMETER_GRAPH_LABEL, PRODUCTION_CACHE_KEY_30KM,
                       TOULOUSE_CENTER_DIST_M)
from settings import settings
from trip_helper.congestion_zones import (NODE_ZONE_KEY, ZONE_AGGLO, ZONE_CITY, ZONE_OUTSIDE,
                                          ZoneError, ensure_zones)
from trip_helper.terminal_time import (TERMINAL_ACCESS_ROUTE, TERMINAL_EGRESS_ROUTE,
                                      data_version, terminal_profile)
from utils import create_background_task, random_uuid

# ── HTTP client mode (OSMNX_ENDPOINTS) ───────────────────────────────────────
# When set, get_direct_plan() delegates to remote osmnx microservice replicas
# instead of computing locally. Falls back to local computation when unset.

_osmnx_endpoints_env = os.getenv("OSMNX_ENDPOINTS", "")
if _osmnx_endpoints_env:
    _osmnx_endpoint_list = [e.strip() for e in _osmnx_endpoints_env.split(",") if e.strip()]
    _osmnx_endpoint_iter = itertools.cycle(_osmnx_endpoint_list)
    logger.info(f"OSMnx HTTP mode: {len(_osmnx_endpoint_list)} endpoint(s): {_osmnx_endpoint_list}")
else:
    _osmnx_endpoint_list = []
    _osmnx_endpoint_iter = None

# Limit concurrent in-flight HTTP requests so that surplus requests wait cheaply
# in the asyncio event loop rather than piling up in the osmnx ProcessPoolExecutors.
# Default = replicas × modes (1 worker per mode per replica), overridable via env.
_http_concurrency = int(
    os.getenv("OSMNX_HTTP_CONCURRENCY", str(max(1, len(_osmnx_endpoint_list)) * 3))
)
_OSMNX_HTTP_SEMAPHORE = asyncio.Semaphore(_http_concurrency)
logger.info(f"OSMnx HTTP concurrency limit: {_http_concurrency}")

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "osmnx.yaml"
with _CONFIG_PATH.open() as _f:
    _cfg = yaml.safe_load(_f)

# ── Prometheus ────────────────────────────────────────────────────────────────

OSMNX_OK  = Counter(
    "osmnx_requests_ok_total",
    "OSMnx routing successes",
    ["mode"],
)
OSMNX_ERR = Counter(
    "osmnx_requests_err_total",
    "OSMnx routing failures",
    ["mode", "reason"],
)
TERMINAL_OUT_OF_PERIMETER = Counter(
    "terminal_time_out_of_perimeter_total",
    "Vehicle trip ends whose point is outside the 453 communes of the survey: "
    "the `default` terminal time law is served to them (ticket 028)",
    ["end"],
)
OSMNX_LAT = Histogram(
    "osmnx_request_duration_seconds",
    "OSMnx routing latency",
    ["mode"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)
OSMNX_INFLIGHT = Gauge(
    "osmnx_requests_inflight",
    "OSMnx requests in progress per mode (agents waiting for the response)",
    ["mode"],
)
OSMNX_CACHE_HIT_RATIO = Gauge(
    "osmnx_cache_hit_ratio",
    "Hits / lookups ratio of the persistent OSMnx cache (0-1)",
)

# ── Route markers (make each direct mode's get_code() unique) ─────────────────

DIRECT_ROUTE_MARKER = {
    "foot":     "__DIRECT_FOOT__",
    "bicycle":  "__DIRECT_BICYCLE__",
    "car":      "__DIRECT_CAR__",
}

# ── Persistent SQLite cache (optional, enabled via settings) ──────────────────

_persistent_cache = None  # type: Optional["OsmnxPersistentCache"]

# Process-wide hits/lookups counters for reporting the cache rate in the logs.
_OSMNX_CACHE_HITS = 0
_OSMNX_CACHE_LOOKUPS = 0


def get_osmnx_cache_stats() -> tuple[int, int]:
    """Return the cumulative (hits, lookups) of the persistent OSMnx cache since startup."""
    return _OSMNX_CACHE_HITS, _OSMNX_CACHE_LOOKUPS


def _montage_couvrant(chemin: str, cibles: set[str]) -> str:
    """The longest mount target that contains ``chemin`` (``/`` at worst)."""
    chemin = chemin.rstrip("/") or "/"
    meilleure = "/"
    for cible in cibles:
        c = cible.rstrip("/") or "/"
        if (chemin == c or chemin.startswith(c + "/") or c == "/") and len(c) > len(meilleure):
            meilleure = c
    return meilleure


def _cache_dir_is_mounted(
    cache_dir: str, mountinfo: str = "/proc/self/mountinfo", code_path: str = __file__
) -> Optional[bool]:
    """Is the cache on ITS volume, and not in the code bind? ``None`` if undeterminable.

    The only reliable test in a container: two binds coming from the same host disk
    share their ``st_dev``, so neither ``st_dev`` nor ``os.path.ismount`` tell them
    apart — the mount target is read in ``/proc/self/mountinfo``.
    Outside Linux (macOS host, tests), the question does not arise: ``None``.

    ⚠ We compare the mount that COVERS the cache with the one that covers the code, and not the exact
    path: the controller stores a population's cache in a SUBfolder of the volume
    (``/app/data/cache/osmnx/toulouse_population_20``). The strict equality test declared it
    "outside the volume" whereas it was in it — false alarm of 2026-09-24.
    """
    try:
        with open(mountinfo, encoding="utf-8") as fh:
            # mountinfo escapes spaces in octal (\040): we give them back to the path.
            cibles = {
                line.split()[4].encode().decode("unicode_escape") for line in fh if line.strip()
            }
    except (OSError, IndexError):
        return None
    cache = _montage_couvrant(os.path.realpath(cache_dir), cibles)
    code = _montage_couvrant(os.path.realpath(code_path), cibles)
    return cache != "/" and cache != code


def init_persistent_cache(cache_dir: str) -> None:
    global _persistent_cache
    from trip_helper.osmnx_persistent_cache import OsmnxPersistentCache

    db_path = os.path.join(cache_dir, "osmnx_cache.db")
    existed = os.path.exists(db_path)
    _persistent_cache = OsmnxPersistentCache(cache_dir)
    n_routes = _persistent_cache.count()
    logger.info(
        f"[osmnx-cache] Route cache active: {db_path} — {n_routes} routes in database "
        f"({'fichier existant' if existed else 'FICHIER CRÉÉ'})"
    )

    # The route cache must land on its volume, not in the CODE bind: it is this
    # mount that makes it shared with the bulk populator. An empty cache on an unmounted
    # path means ~2 h 30 of cold routing for 1,000 personas — silent until now.
    mounted = _cache_dir_is_mounted(cache_dir)
    if mounted is False:
        logger.error(
            f"[ALARME] OSMnx route cache outside the volume: {cache_dir} is not a mount "
            f"point — the routes warmed by the populator (data/cache/osmnx/) are "
            f"invisible and this run will recompute everything. Check the "
            f"`./data/cache/osmnx:{cache_dir}` mount of the controller service (docker-compose.yml)."
        )
    elif not existed or n_routes == 0:
        logger.warning(
            f"[osmnx-cache] EMPTY cache at startup ({db_path}) : this run will compute all its "
            f"routes cold. Expected for a new population; otherwise, warm it with "
            f"step 6 of generate_population.ipynb (SKIP_WARMUP = False)."
        )


def _terminal_leg(route_marker: str, at: Location, start_s: int,
                  duration_s: int, step_label: str) -> Transit:
    """Access or egress leg: time, not a routed movement.

    ``is_transfer=True`` is MANDATORY — it is what keeps
    ``TravelPlan.get_code()`` unchanged (cf. its docstring). ``mode=None`` and
    ``distance=None`` are intended too: the leg carries no mode (otherwise
    the option's label would become ``"None,car,None"``) and has no
    network distance to declare.
    """
    loc = TransitLocation(stop="", lat=at.lat, lon=at.lon)
    return Transit(
        start_time=start_s,
        end_time=start_s + duration_s,
        duration=duration_s,
        distance=None,
        mode=None,
        start_location=loc,
        end_location=loc,
        is_transfer=True,
        transit_route=route_marker,
        step_label=step_label,
    )


# ── Ring of a point, for terminal time (ticket 028) ───────────────────────────
#
# Terminal time is SPATIALISED: access is priced on the ORIGIN ring, parking
# on the DESTINATION one. Until tt3 the point was classified by its
# distance to the hypercentre (8 / 20 / 40 km) — that is not the survey's definition,
# which splits by list of communes, and ticket 020 measured 24.4% of personas
# reclassified between the two. Since tt4 the laws are stratified by the survey's
# table, and the point is classified by MEMBERSHIP of the rings (`CommunalZones`,
# the normative footprint): the log, the targets and the terminal time speak of the same
# split.
#
# A point outside the 453 communes receives `hors périmètre` — which is not a ring and
# has no law of its own: `TerminalProfile` serves it the `default` law (all
# trips). Before tt4 it fell into "3rd ring" because "beyond 40 km" had
# no bound: a silent fallback, exactly the vacuity pattern the repository tracks down.
# It is now COUNTED (`terminal_time_out_of_perimeter_total`) and alarmed once.
_out_of_perimeter_alarm_on = False


@functools.lru_cache(maxsize=1)
def _communal_zones():
    """Ring geometry, loaded once.

    ⚠ LAZY import, and it is necessary: the `osmnx` replicas embed this module
    in their image but do NOT have `mobility_core` on their path (they only mount
    `config/`). An import at the top of the file would make them die at startup at the
    next image rebuild — a deferred failure, triggered by an unrelated
    `docker compose build`. In HTTP mode, the replica only computes the
    network duration and never calls `_make_travel_plan`: the import only happens where
    `mobility_core` exists (controller, tests).
    """
    from mobility_core.residence_zone import CommunalZones

    return CommunalZones.load()


def _terminal_zone(lat: float, lon: float, end: str) -> str:
    """Ring of a trip end; `hors périmètre` is counted and alarmed once."""
    global _out_of_perimeter_alarm_on
    from mobility_core.population_reference import OUT_OF_PERIMETER

    zone = _communal_zones().classify(lat, lon)
    if zone == OUT_OF_PERIMETER:
        TERMINAL_OUT_OF_PERIMETER.labels(end=end).inc()
        if not _out_of_perimeter_alarm_on:
            _out_of_perimeter_alarm_on = True
            logger.error(
                "[ALARME] Terminal time: a trip end is outside the 453 communes of "
                f"the survey ({end}, lat≈{lat:.2f} lon≈{lon:.2f}) — the `default` law is "
                "served to it. Expected for a few peripheral destinations; a high "
                "volume signals a population or a scope that has changed. (Alarm emitted "
                "only once; the terminal_time_out_of_perimeter_total counter continues.)")
    return zone


def _make_travel_plan(
    origin: Location,
    destination: Location,
    trip_mode: str,
    departure_time: int,
    duration_s: int,
    distance_m: float,
) -> TravelPlan:
    """Assemble the direct plan: access + routed trip + egress.

    ``duration_s`` is the pure NETWORK TRAVEL time (ticket 013): the access
    and parking time is no longer melted into it by ``_route_sync``, it
    is carried by named legs that the template knows how to break down. This is
    decision T3 — the lie was upstream of the template, it is fixed upstream.

    Walking and public transport have no terminal profile, hence no
    added leg: walking is door to door, and the access walking legs
    of transit are already routed by OTP (criterion 4: no double counting).
    """
    profile = terminal_profile(trip_mode)
    loc_start = TransitLocation(stop="", lat=origin.lat, lon=origin.lon)
    loc_end = TransitLocation(stop="", lat=destination.lat, lon=destination.lon)

    # Terminal time is SPATIALISED (ticket 013 §4.1): access is priced on the
    # ORIGIN ring — that is where the vehicle is parked — and parking
    # on the DESTINATION one, where a space must be found. A 3rd-ring → Toulouse
    # trip therefore pays a rural access and a city-centre parking.
    # The classification is the survey's — membership of the rings, the same
    # definition as the persona's `residence_zone` trait and as the "Lieu de
    # résidence" column of the log (ticket 028): two diverging classifications would
    # bill a city-centre parking to an agent the log places in the 1st
    # ring, an inconsistency invisible in the logs. The import of `mobility_core` stays
    # lazy, in `_communal_zones` — see why over there.
    origin_zone = _terminal_zone(origin.lat, origin.lon, "access")
    dest_zone = _terminal_zone(destination.lat, destination.lon, "egress")

    legs: list[Transit] = []
    cursor = departure_time

    # Draw key of the terminal time (tt3: the duration is drawn from the survey law,
    # not constant). It identifies the TRIP — mode and origin-destination pair to
    # 10⁻⁵ degree, ~1 m — so two distinct trips draw independently and the same
    # trip always draws the same. This last point is not cosmetic: the plans
    # are cached and so are the LLM decisions, an unstable draw would make
    # a run diverge from its resumption and would make the decision cache wrong.
    draw_key = (f"{trip_mode}:{origin.lat:.5f},{origin.lon:.5f}"
                f"→{destination.lat:.5f},{destination.lon:.5f}")
    access_s = profile.access_s(origin_zone, draw_key) if profile is not None else 0
    egress_s = profile.egress_s(dest_zone, draw_key) if profile is not None else 0

    if access_s:
        legs.append(_terminal_leg(TERMINAL_ACCESS_ROUTE, origin, cursor,
                                  access_s, profile.labels["access"]))
        cursor += access_s

    legs.append(Transit(
        start_time=cursor,
        end_time=cursor + duration_s,
        duration=duration_s,
        distance=distance_m,
        mode=trip_mode,
        start_location=loc_start,
        end_location=loc_end,
        is_transfer=False,
        transit_route=DIRECT_ROUTE_MARKER[trip_mode],
        step_label=profile.labels["main"] if profile is not None else None,
    ))
    cursor += duration_s

    if egress_s:
        # The label keeps its `{destination}`: `purpose` is only set on the plan
        # after routing, so the interpolation happens at rendering.
        legs.append(_terminal_leg(TERMINAL_EGRESS_ROUTE, destination, cursor,
                                  egress_s, profile.labels["egress"]))
        cursor += egress_s

    return TravelPlan(
        id=random_uuid(),
        start_location=origin,
        end_location=destination,
        start_time=departure_time,
        end_time=cursor,
        duration=cursor - departure_time,
        distance=distance_m,
        legs=legs,
    )


# ── OSMnx network_type mapping ────────────────────────────────────────────────

_OSMNX_MODES = ("walk", "bike", "drive")
_MODE_TO_OSMNX = {"foot": "walk", "bicycle": "bike", "car": "drive"}

# ── Routing config (loaded from config/osmnx.yaml) ───────────────────────────

_SPEEDS    = _cfg["speeds"]
_FALLBACKS = _cfg["fallbacks"]
_PENALTIES = _cfg["penalties"]
# `park_base` is no longer read: terminal time has left the routing engine
# for `config/terminal_time.yaml` (ticket 013). The key stays in osmnx.yaml,
# neutralised and commented out, so that reading the file does not suggest
# that parking was never modelled.

# ── Congestion tables (TomTom Toulouse, loaded from config/osmnx.yaml) ────────

_DAYS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_HOURS = [f"{h:02d}:00" for h in range(24)]

_cong     = _cfg["congestion"]
_CITY_DF  = pd.DataFrame(_cong["city_raw"],  index=_HOURS, columns=_DAYS)
_METRO_DF = pd.DataFrame(_cong["metro_raw"], index=_HOURS, columns=_DAYS)
_CITY_FF  = float(_cong["city_free_flow_kmh"])
_METRO_FF = float(_cong["metro_free_flow_kmh"])

# ── Distance cutoffs ──────────────────────────────────────────────────────────
# Skip routing entirely when straight-line distance exceeds these thresholds.

_MAX_FOOT_M = 15_000   # 15 km
_MAX_BIKE_M = 30_000   # 30 km

# ── Executor pools ────────────────────────────────────────────────────────────
# Graph building is I/O-bound (OSM download) → ThreadPoolExecutor.
# Routing (NetworkX Dijkstra) is pure-Python CPU-bound → ProcessPoolExecutor
# per mode so each worker process loads only its own graph and bypasses the GIL.
#
# spawn is used on macOS/Windows (fork unsafe with asyncio threads);
# fork is used on Linux (faster startup, COW memory sharing).

_WORKERS_PER_MODE = 1

_IO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="osmnx-io")

# Daemon processes cannot spawn child processes (Python multiprocessing restriction).
# Fall back to ThreadPoolExecutor when running inside a daemon worker (e.g. uvicorn).
_in_daemon = _mp.current_process().daemon

if _in_daemon:
    _MODE_POOLS: dict = {
        mode: ThreadPoolExecutor(max_workers=_WORKERS_PER_MODE, thread_name_prefix=f"osmnx-{mode}")
        for mode in _OSMNX_MODES
    }
else:
    _mp_context = _mp.get_context("fork" if _sys.platform.startswith("linux") else "spawn")
    _MODE_POOLS: dict = {
        mode: ProcessPoolExecutor(max_workers=_WORKERS_PER_MODE, mp_context=_mp_context)
        for mode in _OSMNX_MODES
    }


def _reset_pool(osmnx_mode: str) -> None:
    """Recreate a broken ProcessPoolExecutor for the given mode."""
    old = _MODE_POOLS.get(osmnx_mode)
    if old is not None:
        old.shutdown(wait=False, cancel_futures=True)
    _MODE_POOLS[osmnx_mode] = ProcessPoolExecutor(max_workers=_WORKERS_PER_MODE, mp_context=_mp_context)
    logger.warning(f"[osmnx] ProcessPool for mode={osmnx_mode} recreated after worker crash.")

# ── Graph singleton ───────────────────────────────────────────────────────────

# City and radius of the HISTORICAL graph (30 km disk, downloaded through Overpass). They are now
# only used to rebuild this audit graph if its pickle is missing — never the runtime graph.
_LEGACY_CITY = "Toulouse, France"
_LEGACY_DIST_M = TOULOUSE_CENTER_DIST_M


class GraphMissingError(RuntimeError):
    """The pickle of the configured graph is missing: we refuse to download another one in its place."""


def graph_key() -> str:
    """Key of the OSMnx graph served at runtime (ticket 031, part 2).

    `settings.gtfs.osmnx_graph_key` if set, otherwise the graph of the **polygon of the 453
    communes** (`geography.PERIMETER_CACHE_KEY`). The 30 km disk (`PRODUCTION_CACHE_KEY_30KM`)
    stays loadable for an audit; it is no longer what the simulation routes on.
    """
    return getattr(settings.gtfs, "osmnx_graph_key", None) or PERIMETER_CACHE_KEY


def graph_label(key: str) -> str:
    if key == PERIMETER_CACHE_KEY:
        return PERIMETER_GRAPH_LABEL
    if key == PRODUCTION_CACHE_KEY_30KM:
        return f"disque historique {_LEGACY_CITY} r={_LEGACY_DIST_M} m (audit)"
    return "clé configurée (settings.gtfs.osmnx_graph_key)"


class _GraphStore:
    """Process-wide singleton: load walk/bike/drive graphs + city boundary once."""

    _graphs:   Optional[dict]             = None
    _boundary: Optional[gpd.GeoDataFrame] = None
    _lock:     Optional[asyncio.Lock]     = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _cache_dir(cls) -> Path:
        path = Path(getattr(settings.gtfs, "osmnx_cache_dir", "/app/osmnx_cache"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _build_sync(cls, key: Optional[str] = None) -> tuple[dict, gpd.GeoDataFrame]:
        """Restore the graphs and the boundary of graph `key` from the disk cache.

        The graph of the polygon of the 453 communes is built offline
        (`make osmnx-perimeter-graph`): if it is missing, it is an explicit error — downloading
        a 30 km disk in its place would amount to serving another scope under the same name.
        Only the historical graph (`PRODUCTION_CACHE_KEY_30KM`) keeps its Overpass download
        recipe, for the audit.
        """
        key     = key or graph_key()
        legacy  = key == PRODUCTION_CACHE_KEY_30KM
        g_path  = cls._cache_dir() / f"graphs_{key}.pkl"
        b_path  = cls._cache_dir() / f"boundary_{key}.pkl"

        graphs = None
        if g_path.exists():
            t_load = _time.monotonic()
            logger.info(f"OSMnx: loading graph {key} ({graph_label(key)}) from {g_path.name} "
                        f"({g_path.stat().st_size / 1_048_576:.0f} MB)")
            with g_path.open("rb") as f:
                graphs = pickle.load(f)
            if not all(G.number_of_edges() > 0 for G in graphs.values()):
                if not legacy:
                    raise GraphMissingError(f"OSMnx: the pickle {g_path} carries an empty graph — "
                                            "rebuild it (`make osmnx-perimeter-graph FORCE=1`)")
                logger.warning("OSMnx: cached graphs are empty, clearing cache and re-downloading…")
                g_path.unlink(missing_ok=True)
                graphs = None
            else:
                logger.info(f"OSMnx: graph {key} loaded in {_time.monotonic() - t_load:.1f}s — "
                            + " ; ".join(f"{m}: {G.number_of_nodes()} nœuds / {G.number_of_edges()} arêtes"
                                         for m, G in graphs.items()))

        if graphs is None:
            if not legacy:
                logger.error(
                    f"[ALARME] OSMnx: graph {key} ({graph_label(key)}) not found in "
                    f"{cls._cache_dir()} (expected {g_path.name}). Nothing is downloaded in its place: "
                    "build it with `make osmnx-perimeter-graph` (data/cache/osmnx is mounted "
                    "in the container), or set settings.gtfs.osmnx_graph_key to a cached key.")
                raise GraphMissingError(f"OSMnx graph {key} missing: {g_path}")
            logger.info(f"OSMnx: downloading graphs for {_LEGACY_CITY!r} (r={_LEGACY_DIST_M} m) …")
            graphs = {}
            for osmnx_mode in _OSMNX_MODES:
                G = ox.graph_from_address(_LEGACY_CITY, dist=_LEGACY_DIST_M, network_type=osmnx_mode)
                for _, _, _, data in G.edges(keys=True, data=True):
                    hwy = data.get("highway")
                    if isinstance(hwy, list):
                        hwy = hwy[0]
                    data["speed_kph"] = _SPEEDS[osmnx_mode].get(hwy, _FALLBACKS[osmnx_mode])
                graphs[osmnx_mode] = ox.add_edge_travel_times(G)
            with g_path.open("wb") as f:
                pickle.dump(graphs, f)
            logger.info(f"OSMnx: graphs saved to {g_path.name}")

        if b_path.exists():
            logger.info(f"OSMnx: loading boundary from {b_path.name}")
            with b_path.open("rb") as f:
                boundary = pickle.load(f)
        elif legacy:
            logger.info(f"OSMnx: downloading boundary for {_LEGACY_CITY!r} …")
            boundary = ox.geocode_to_gdf(_LEGACY_CITY).to_crs("EPSG:4326")
            with b_path.open("wb") as f:
                pickle.dump(boundary, f)
            logger.info(f"OSMnx: boundary saved to {b_path.name}")
        else:
            logger.error(f"[ALARME] OSMnx: Toulouse boundary missing for graph {key} "
                         f"({b_path.name}) — `make osmnx-perimeter-graph` copies it from the production "
                         "cache; nothing is geocoded in its place.")
            raise GraphMissingError(f"boundary {b_path} missing")

        # Congestion zones (ticket 031, decision 4): computed once for a graph that has
        # none — the historical 30 km graph downloaded before this change — then
        # cached in the pickle. The polygon graph carries them from its construction.
        t_zones = _time.monotonic()
        counts = ensure_zones(graphs, boundary, log=logger)
        if counts:
            with g_path.open("wb") as f:
                pickle.dump(graphs, f)
            logger.info(f"OSMnx: congestion zones computed in {_time.monotonic() - t_zones:.1f}s and "
                        f"cached in {g_path.name} — "
                        + " ; ".join(f"{m}: " + ", ".join(f"{z} {n}" for z, n in c.items()) for m, c in counts.items()))
        else:
            logger.info("OSMnx: congestion zones already present on the nodes — "
                        + " ; ".join(f"{m}: " + ", ".join(f"{z} {n}" for z, n in _zone_counts(G).items())
                                     for m, G in graphs.items()))

        return graphs, boundary

    @classmethod
    async def get(cls, key: Optional[str] = None) -> tuple[dict, gpd.GeoDataFrame]:
        """Graphs and boundary of graph `key` (default: `graph_key()`), loaded once."""
        # Return cached graphs and boundary if exist
        if cls._graphs is not None:
            return cls._graphs, cls._boundary

        # Else build them in a thread to avoid blocking the event loop, with a lock to prevent concurrent builds
        async with cls._get_lock():
            if cls._graphs is not None:
                return cls._graphs, cls._boundary
            loop = asyncio.get_running_loop()
            cls._graphs, cls._boundary = await loop.run_in_executor(
                _IO_EXECUTOR, cls._build_sync, key or graph_key()
            )
            logger.info("OSMnx: graphs ready.")
        return cls._graphs, cls._boundary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _crow_flies_m(origin: Location, destination: Location) -> float:
    dlat = (destination.lat - origin.lat) * 111_320
    dlon = (destination.lon - origin.lon) * 111_320 * math.cos(math.radians(origin.lat))
    return math.hypot(dlat, dlon)


def _zone_counts(G) -> dict:
    counts: dict = {}
    for _, data in G.nodes(data=True):
        z = data.get(NODE_ZONE_KEY, "?")
        counts[z] = counts.get(z, 0) + 1
    return counts


def _zone_factor(zone: str, dt: datetime) -> float:
    """Congestion factor of an edge according to the zone of its origin node (ticket 031, decision 4).

    ``city`` → TomTom "city" profile, ``agglo`` → "urban area" profile, ``outside`` → 1.0.
    A node without a zone has no guessable factor: explicit error (the zones are set at
    graph loading, cf. `congestion_zones.ensure_zones`).
    """
    if zone == ZONE_OUTSIDE:
        return 1.0
    hour = f"{dt.hour:02d}:00"
    day  = dt.strftime("%a")
    if zone == ZONE_CITY:
        return _CITY_FF / float(_CITY_DF.loc[hour, day])
    if zone == ZONE_AGGLO:
        return _METRO_FF / float(_METRO_DF.loc[hour, day])
    raise ZoneError(f"node without a congestion zone ({zone!r}): the graph has not been prepared "
                    "(congestion_zones.ensure_zones)")


def _congested_travel_time(G, gdf, dt: datetime) -> tuple[float, float, int]:
    """``(congested duration, free-flow duration, accident edges)`` of an itinerary.

    Σ edges travel_time × factor(zone, hour) × factor(accident, hour). ``gdf`` is the
    output of `ox.routing.route_to_gdf` (index (u, v, key)); the zone is that of the origin
    node ``u``. The global factor reported elsewhere is the ratio of the two durations, i.e.
    the mean of the factors weighted by the free-flow time.

    THE ACCIDENT ENTERS HERE, and nowhere else (ticket 070, work C). It is the only
    place in the set-up that knows the edges really used: the GAMA agents
    move as the crow flies, but their speed is set on the duration computed here
    (`Inhabitant.gaml:390`), so a lengthened itinerary does make the agent arrive late.

    ⚠ The agent DOES NOT DETOUR: the path was chosen in free-flow time, before this call. The
    extra cost applies to the path already chosen. Detouring would require recomputing the
    shortest path on modified weights — outside the declared scope.
    """
    from trip_helper import accidents as _accidents

    registre = _accidents.registre()
    ts = int(dt.timestamp()) if registre is not None else 0

    free_s = 0.0
    cong_s = 0.0
    n_accidentees = 0
    for (u, v, _k), tt in zip(gdf.index, gdf["travel_time"].to_numpy(dtype=float)):
        free_s += tt
        facteur = _zone_factor(G.nodes[u].get(NODE_ZONE_KEY), dt)
        if registre is not None:
            f_accident = registre.facteur_arete((u, v), ts)
            if f_accident != 1.0:
                n_accidentees += 1
                facteur *= f_accident
        cong_s += tt * facteur
    return cong_s, free_s, n_accidentees


def _infra_penalty(G, route_nodes: list, osmnx_mode: str) -> float:
    # Penalize routes that traverse infrastructure types considered slower or less desirable.
    counts = {k: 0 for k in _PENALTIES[osmnx_mode]}
    for n in route_nodes[1:-1]:
        tag = G.nodes[n].get("highway")
        if tag in counts:
            counts[tag] += 1
    return sum(counts[k] * _PENALTIES[osmnx_mode][k] for k in counts)


# ── Memory probe (Linux /proc, no-op on other platforms) ─────────────────────

def _proc_mem_mb() -> str:
    """Return a compact RSS/VmPeak/VmSize string from /proc/self/status, or '?' if unavailable."""
    try:
        fields = {}
        with open("/proc/self/status") as fh:
            for line in fh:
                if ":" in line:
                    k, v = line.split(":", 1)
                    fields[k.strip()] = v.strip()
        rss  = fields.get("VmRSS",  "?")
        peak = fields.get("VmPeak", "?")
        vsz  = fields.get("VmSize", "?")
        return f"RSS={rss} VmPeak={peak} VmSize={vsz}"
    except Exception:
        return "mem=unavailable"


# ── Worker-process state (populated lazily on first task, None in main process) ─

_worker_graph:    object                    = None
_worker_boundary: Optional[gpd.GeoDataFrame] = None


def _route_sync_process(
    cache_dir:    str,
    cache_key:    str,
    origin:       Location,
    destination:  Location,
    osmnx_mode:   str,
    congestion_dt: datetime,
) -> Optional[dict]:
    """Entry point for ProcessPoolExecutor workers. Loads its graph once, then routes."""
    global _worker_graph, _worker_boundary
    pid = os.getpid()
    if _worker_graph is None:
        g_path = Path(cache_dir) / f"graphs_{cache_key}.pkl"
        b_path = Path(cache_dir) / f"boundary_{cache_key}.pkl"

        pkl_mb = g_path.stat().st_size / 1_048_576 if g_path.exists() else -1
        logger.info(
            f"[osmnx-worker] pid={pid} mode={osmnx_mode} COLD START — "
            f"pickle={pkl_mb:.0f}MB  {_proc_mem_mb()}"
        )
        t_load = _time.monotonic()

        # Load full pickle then immediately release the other two mode graphs
        # so they can be GC'd before routing begins.
        with g_path.open("rb") as fh:
            _all = pickle.load(fh)
        _worker_graph = _all[osmnx_mode]
        del _all
        gc.collect()

        with b_path.open("rb") as fh:
            _worker_boundary = pickle.load(fh)
        # Congestion zones: normally already in the pickle (_GraphStore caches them);
        # otherwise computed here, without rewriting the pickle from a worker (no write race).
        counts = ensure_zones({osmnx_mode: _worker_graph}, _worker_boundary, log=logger)
        if counts:
            logger.warning(f"[osmnx-worker] pid={pid} mode={osmnx_mode} congestion zones computed "
                           f"hot ({dict(counts[osmnx_mode])}) — the pickle {g_path.name} did not carry them")

        logger.info(
            f"[osmnx-worker] pid={pid} mode={osmnx_mode} READY — "
            f"nodes={_worker_graph.number_of_nodes()} edges={_worker_graph.number_of_edges()} "
            f"load={_time.monotonic()-t_load:.1f}s  {_proc_mem_mb()}"
        )

    t_route = _time.monotonic()
    result = _route_sync(_worker_graph, _worker_boundary, origin, destination, osmnx_mode, congestion_dt)
    # logger.debug(
    #     f"[osmnx-worker] pid={pid} mode={osmnx_mode} "
    #     f"route={_time.monotonic()-t_route:.3f}s  {_proc_mem_mb()}"
    # )
    return result


# Detour applied to the straight-line distance when the graph does not separate the two points
# (same ratio as eqasim's Euclidean approximation: `routed_distance / 1.3`).
SAME_NODE_DETOUR = 1.3


def _same_node_fallback(origin: Location, destination: Location, osmnx_mode: str) -> dict:
    """Duration and distance of a trip too short for the graph: straight line × detour, speed of the mode."""
    dist_m = _crow_flies_m(origin, destination) * SAME_NODE_DETOUR
    speed_ms = float(_FALLBACKS[osmnx_mode]) * 1000.0 / 3600.0
    return {"duration_s": max(1, int(round(dist_m / speed_ms))), "distance_m": dist_m}


def _route_sync(
    G:            object,
    boundary:     gpd.GeoDataFrame,
    origin:       Location,
    destination:  Location,
    osmnx_mode:   str,
    congestion_dt: datetime,
) -> Optional[dict]:
    """Pure-sync OSMnx routing. Returns {duration_s, distance_m} or None.

    "Same node" fallback (ticket 031, decision of 2026-09-03). When origin and destination
    snap to the same node of the graph, there is no itinerary to compute: the trip is
    shorter than the graph mesh. Before, the duration came from a 70 km/h speed whatever
    the mode — a 200 m walk lasted 10 s, displayed "0 minute"; this fallback had
    been written for points OUTSIDE the 30 km graph (98 of the 154 3rd-ring agents of v3),
    a case that disappears with the graph of the polygon of the 453 communes. Now: straight-line
    distance × 1.3 detour, duration at the MODE's fallback speed (`_FALLBACKS`: walk 5,
    bike 14, car 30 km/h), minimum 1 s — the distance returned is that of the detour.

    Congestion (car): per edge, according to the zone of the origin node — `city` (city profile),
    `agglo` (urban area profile), `outside` (1.0) — see `_congested_travel_time` and
    `trip_helper.congestion_zones`. `boundary` is no longer read here: it is used to set the zones
    when the graph is loaded.
    """
    # Find the nearest nodes on the graph to the origin and destination coordinates.
    orig = ox.distance.nearest_nodes(G, origin.lon, origin.lat)
    dest = ox.distance.nearest_nodes(G, destination.lon, destination.lat)

    if orig == dest:
        return _same_node_fallback(origin, destination, osmnx_mode)

    # Compute the fastest route using edge travel times as weights.
    route = ox.routing.shortest_path(G, orig, dest, weight="travel_time")
    if route is None or len(route) < 2:
        return None

    # Convert the route to a GeoDataFrame to aggregate per-edge metrics.
    gdf = ox.routing.route_to_gdf(G, route)

    # Total geometric distance in meters and base travel time in seconds.
    dist_m = float(gdf["length"].sum())
    free_s = float(gdf["travel_time"].sum())

    # Add penalties for infrastructure types that are slower or less desirable.
    infra_s = _infra_penalty(G, route, osmnx_mode)

    # ⚠ No parking time here (ticket 013). What comes out of this
    # function is pure NETWORK TRAVEL time. The former
    # `park_s = _PARK_BASE[mode] * 2` added 4 min (car) / 2 min (bike)
    # INTO this duration: the displayed total contained parking without
    # anything saying so, whereas the public transport options show
    # each walking leg. Terminal time is now a documented exogenous
    # parameter (config/terminal_time.yaml), carried by named legs
    # that the template breaks down — cf. `_make_travel_plan`.

    # Congestion (car only), PER EDGE according to the zone of its origin node — city,
    # urban area, outside (factor 1) — at the departure time (ticket 031, decision 4). A
    # 3rd-ring → Toulouse trip is only congested on its urban part; a 3rd-ring village →
    # village trip is not. Before: a single factor for the whole trip, "city"
    # if one end touched Toulouse, "urban area" otherwise — 1.84 on a Monday at 8 a.m. in open country.
    cong_s = free_s
    if osmnx_mode == "drive":
        cong_s, _free_s, _n_acc = _congested_travel_time(G, gdf, congestion_dt)
        if _n_acc:
            from trip_helper import accidents as _accidents

            _registre = _accidents.registre()
            if _registre is not None:
                _registre.compter_trajet_touche()
                logger.info(
                    f"[accidents] Trip delayed: {_n_acc} accident edge(s) on "
                    f"the itinerary, duration {int(_free_s)} s → {int(cong_s)} s "
                    f"(departure {congestion_dt:%Y-%m-%d %H:%M})"
                )

    # Round up to at least one second and apply all modifiers.
    return {
        "duration_s": max(1, int(cong_s + infra_s)),
        "distance_m": dist_m,
    }


# ── Public API ────────────────────────────────────────────────────────────────

_OSMNX_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=800)

# HTTP session shared between all the requests to the osmnx replicas
# (keep-alive: avoids the cost of session creation + TCP handshake at each request).
_osmnx_http_session: Optional[aiohttp.ClientSession] = None


async def _get_osmnx_session() -> aiohttp.ClientSession:
    global _osmnx_http_session
    if _osmnx_http_session is None or _osmnx_http_session.closed:
        _osmnx_http_session = aiohttp.ClientSession(
            timeout=_OSMNX_HTTP_TIMEOUT,
            connector=aiohttp.TCPConnector(limit=50),
        )
    return _osmnx_http_session


async def _get_direct_plan_http(
    origin: Location,
    destination: Location,
    trip_mode: str,
    congestion_dt: datetime,
    _timing_sink: dict | None = None,
) -> Optional[dict]:
    """POST to the next osmnx replica in the round-robin cycle."""
    import time as _wtime
    url = next(_osmnx_endpoint_iter)
    payload = {
        "origin":       {"lat": origin.lat,      "lon": origin.lon},
        "destination":  {"lat": destination.lat, "lon": destination.lon},
        "mode":         trip_mode,
        "congestion_dt": congestion_dt.isoformat(),
    }
    async with _OSMNX_HTTP_SEMAPHORE:
        if _timing_sink is not None:
            # Keep the latest semaphore acquisition across all parallel OSMnx calls
            _timing_sink["osmnx_sem_end"] = max(
                _timing_sink.get("osmnx_sem_end", 0), _wtime.time()
            )
        session = await _get_osmnx_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()
        return result


async def get_direct_plan(
    origin:        Location,
    destination:   Location,
    trip_mode:     str,      # "foot" | "bicycle" | "car"
    departure_time: int,     # Unix timestamp in seconds
    congestion_dt:  datetime, # Real calendar date+time for congestion lookup
    _timing_sink: dict | None = None,
) -> Optional[TravelPlan]:
    """Async OSMnx direct route. Returns a one-leg TravelPlan or None on failure.

    The graph is that of `graph_key()` — the polygon of the 453 communes by default (ticket 031,
    part 2); no more city nor radius as parameters.
    """
    # Fast reject: skip routing when the straight-line distance exceeds the mode's threshold.
    straight_m = _crow_flies_m(origin, destination)
    if trip_mode == "foot" and straight_m > _MAX_FOOT_M:
        return None
    if trip_mode == "bicycle" and straight_m > _MAX_BIKE_M:
        return None

    # Persistent cache lookup (after fast reject to avoid caching trivial misses).
    #
    # ACCIDENT GUARD (ticket 070, work D). This cache is addressed WITHOUT THE DATE — its key is
    # (version, day of week, slot, mode, coordinates). A duration containing an accident
    # would therefore be served to every Tuesday 8 a.m., including to runs that requested
    # no accident. As long as an accident is active, we neither read nor write: `_p_key` stays
    # `None`, which neutralises at once the read below and the two writes further down.
    #
    # The guard is DELIBERATELY COARSE — an active accident anywhere is enough to
    # bypass the cache, even for an itinerary that does not cross it. We do not know the
    # path before having computed it: filtering finely would require routing first, hence
    # giving up the cache anyway. Accepted cost: at ~1.5 accidents per day of 20 to
    # 90 minutes, one to three hours per simulated day where car routing computes cold.
    _p_key = _p_date = _p_dow = _p_bucket = None
    _accidents_actifs = False
    if _persistent_cache is not None:
        from trip_helper import accidents as _accidents_mod

        _reg = _accidents_mod.registre()
        _accidents_actifs = _reg is not None and _reg.a_des_accidents_actifs(
            int(congestion_dt.timestamp())
        )
    if _persistent_cache is not None and not _accidents_actifs:
        _p_key, _p_date, _p_dow, _p_bucket = _persistent_cache.__class__.make_key(
            congestion_dt, trip_mode,
            round(origin.lat, 5), round(origin.lon, 5),
            round(destination.lat, 5), round(destination.lon, 5),
        )
        entry = await _persistent_cache.lookup_async(_p_key)
        global _OSMNX_CACHE_HITS, _OSMNX_CACHE_LOOKUPS
        _OSMNX_CACHE_LOOKUPS += 1
        if entry.found:
            _OSMNX_CACHE_HITS += 1
            OSMNX_CACHE_HIT_RATIO.set(_OSMNX_CACHE_HITS / _OSMNX_CACHE_LOOKUPS)
            if entry.result is None:
                return None
            return _make_travel_plan(origin, destination, trip_mode, departure_time,
                                     entry.result["duration_s"], entry.result["distance_m"])
        OSMNX_CACHE_HIT_RATIO.set(_OSMNX_CACHE_HITS / _OSMNX_CACHE_LOOKUPS)

    # An active accident also bypasses the historical cache. Before the paper,
    # recomputing this itinerary only in the control arm would break the common prefix.
    from trip_helper.strict_replay import exiger_route_en_cache

    exiger_route_en_cache(departure_time, "OSMnx", _p_key)
    osmnx_mode = _MODE_TO_OSMNX[trip_mode]
    t0 = _time.monotonic()
    OSMNX_INFLIGHT.labels(mode=trip_mode).inc()
    try:
        if _osmnx_endpoint_list:
            # HTTP mode: delegate to a remote osmnx microservice replica.
            result = await _get_direct_plan_http(origin, destination, trip_mode, congestion_dt,
                                                 _timing_sink=_timing_sink)
        else:
            # Local mode: compute in-process (thread or process pool depending on daemon status).
            graphs, boundary = await _GraphStore.get()
            loop = asyncio.get_running_loop()

            if _in_daemon:
                # Thread pool: graphs already in memory, call _route_sync directly.
                fn = functools.partial(
                    _route_sync,
                    graphs[osmnx_mode], boundary, origin, destination, osmnx_mode, congestion_dt,
                )
            else:
                # Process pool: workers load their graph from disk cache independently.
                cache_dir = str(_GraphStore._cache_dir())
                cache_key = graph_key()
                fn = functools.partial(
                    _route_sync_process,
                    cache_dir, cache_key, origin, destination, osmnx_mode, congestion_dt,
                )

            try:
                result = await loop.run_in_executor(_MODE_POOLS[osmnx_mode], fn)
            except BrokenProcessPool:
                _reset_pool(osmnx_mode)
                result = await loop.run_in_executor(_MODE_POOLS[osmnx_mode], fn)
    except Exception as exc:
        OSMNX_ERR.labels(mode=trip_mode, reason=type(exc).__name__).inc()
        logger.exception(f"OSMnx {trip_mode} routing error: {exc}")
        return None
    finally:
        OSMNX_INFLIGHT.labels(mode=trip_mode).dec()

    if result is None:
        OSMNX_ERR.labels(mode=trip_mode, reason="no_route").inc()
        if _persistent_cache is not None and _p_key is not None:
            ecriture = _persistent_cache.store_async(
                _p_key, _p_date, _p_dow, _p_bucket, trip_mode,
                round(origin.lat, 5), round(origin.lon, 5),
                round(destination.lat, 5), round(destination.lon, 5),
                None,
            )
            if settings.world.prefixe_commun:
                await ecriture
            else:
                create_background_task(ecriture)
        return None

    OSMNX_OK.labels(mode=trip_mode).inc()
    OSMNX_LAT.labels(mode=trip_mode).observe(_time.monotonic() - t0)

    if _persistent_cache is not None and _p_key is not None:
        ecriture = _persistent_cache.store_async(
            _p_key, _p_date, _p_dow, _p_bucket, trip_mode,
            round(origin.lat, 5), round(origin.lon, 5),
            round(destination.lat, 5), round(destination.lon, 5),
            result,
        )
        if settings.world.prefixe_commun:
            await ecriture
        else:
            create_background_task(ecriture)

    return _make_travel_plan(origin, destination, trip_mode, departure_time,
                             result["duration_s"], result["distance_m"])


async def warmup(key: Optional[str] = None) -> None:
    """Pre-load OSMnx graphs at application startup to avoid cold-start latency."""
    await _GraphStore.get(key)


if __name__ == "__main__":
    import sys
    import tempfile
    import time as _ttime

    city = "Toulouse, France"
    # Smaller radius by default so the test completes quickly.
    # Pass a custom dist as first arg: python osmnx_direct.py 30000
    dist = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000

    print(f"=== OSMnx save/reload test — {city}, dist={dist} m ===\n")

    # ── Step 1: build (forced, no cache) ──────────────────────────────────────
    print("[1/3] Building graphs from OSMnx (no cache)…")
    t_build_start = _ttime.monotonic()
    graphs_orig: dict = {}
    for mode in _OSMNX_MODES:
        print(f"      {mode}…", end=" ", flush=True)
        t0 = _ttime.monotonic()
        G = ox.graph_from_address(city, dist=dist, network_type=mode)
        for _, _, _, data in G.edges(keys=True, data=True):
            hwy = data.get("highway")
            if isinstance(hwy, list):
                hwy = hwy[0]
            data["speed_kph"] = _SPEEDS[mode].get(hwy, _FALLBACKS[mode])
        graphs_orig[mode] = ox.add_edge_travel_times(G)
        n, e = graphs_orig[mode].number_of_nodes(), graphs_orig[mode].number_of_edges()
        print(f"nodes={n}, edges={e}  ({_ttime.monotonic() - t0:.1f}s)")
        if e == 0:
            print(f"      ERROR: downloaded graph for {mode!r} has no edges!", file=sys.stderr)
            sys.exit(1)
    t_build = _ttime.monotonic() - t_build_start
    print(f"      → build total: {t_build:.1f}s")

    # ── Step 2: pickle save ────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pkl_path = Path(tmp.name)

    print(f"\n[2/3] Saving to {pkl_path}…")
    t0 = _ttime.monotonic()
    with pkl_path.open("wb") as f:
        pickle.dump(graphs_orig, f)
    size_mb = pkl_path.stat().st_size / 1_048_576
    print(f"      Written {size_mb:.1f} MB in {_ttime.monotonic() - t0:.2f}s")

    # ── Step 3: reload and validate ───────────────────────────────────────────
    print(f"\n[3/3] Reloading from {pkl_path}…")
    t0 = _ttime.monotonic()
    with pkl_path.open("rb") as f:
        graphs_loaded = pickle.load(f)
    t_load = _ttime.monotonic() - t0
    print(f"      Loaded in {t_load:.2f}s")

    ok = True
    for mode in _OSMNX_MODES:
        orig_e  = graphs_orig[mode].number_of_edges()
        load_e  = graphs_loaded[mode].number_of_edges()
        status  = "OK" if orig_e == load_e and load_e > 0 else "FAIL"
        print(f"      {mode}: before={orig_e} edges, after={load_e} edges  [{status}]")
        if status == "FAIL":
            ok = False

    pkl_path.unlink(missing_ok=True)

    print()
    if ok:
        print(f"All graphs survived the save/reload cycle correctly. (build {t_build:.1f}s, reload {t_load:.2f}s)")
    else:
        print("ERROR: edge count mismatch after reload — pickle corruption detected.", file=sys.stderr)
        sys.exit(1)
