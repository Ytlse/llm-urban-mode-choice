"""Worker module for parallel OSMnx route computation in ProcessPoolExecutor.

This module must be importable as a top-level module (not a nested function)
for the spawn-based ProcessPoolExecutor to work on macOS.
"""
import gc
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

_graphs = None
_boundary = None
_simulation_date = None  # set by init_worker; date(2024, 1, 8) = Monday by default


def _osmnx_mode(trip_mode: str) -> str:
    return {"foot": "walk", "bicycle": "bike", "car": "drive"}[trip_mode]


def init_worker(
    llmagents_path: str,
    osmnx_cache_dir: str,
    cache_key: str,
    simulation_date_iso: str = "2024-01-08",  # Monday by default
) -> None:
    """Called once per worker process to pre-load OSMnx graphs."""
    global _graphs, _boundary, _simulation_date

    if llmagents_path not in sys.path:
        sys.path.insert(0, llmagents_path)

    _simulation_date = simulation_date_iso

    cache = Path(osmnx_cache_dir)
    t0 = time.monotonic()
    with (cache / f"graphs_{cache_key}.pkl").open("rb") as f:
        _graphs = pickle.load(f)
    gc.collect()
    with (cache / f"boundary_{cache_key}.pkl").open("rb") as f:
        _boundary = pickle.load(f)

    # Node congestion zones (ticket 031, decision 4): carried by the pickle of the polygon
    # graph; computed on the fly otherwise (30 km graph from before the change), without rewriting
    # the pickle from a worker.
    from trip_helper.congestion_zones import ensure_zones
    counts = ensure_zones(_graphs, _boundary)

    pid = os.getpid()
    elapsed = time.monotonic() - t0
    print(f"[worker pid={pid}] graphs loaded in {elapsed:.1f}s  (simulation_date={simulation_date_iso})"
          + (f" — congestion zones computed on the fly: {({m: dict(c) for m, c in counts.items()})}" if counts
             else " — congestion zones read from the pickle"), flush=True)


def compute_route_worker(args: tuple) -> tuple:
    """
    Compute one route.

    args: (origin_lat, origin_lon, dest_lat, dest_lon, trip_mode, hour_of_day)

    Returns: (args, {"duration_s": int, "distance_m": float} | None)
    """
    global _graphs, _boundary

    origin_lat, origin_lon, dest_lat, dest_lon, trip_mode, hour_of_day = args

    from models import Location
    from trip_helper.osmnx_direct import _route_sync

    osmnx_mode = _osmnx_mode(trip_mode)
    origin = Location(lat=origin_lat, lon=origin_lon)
    dest = Location(lat=dest_lat, lon=dest_lon)
    hour = min(int(hour_of_day), 23)
    minute = int((hour_of_day % 1) * 60)
    from datetime import date as _date
    _d = _date.fromisoformat(_simulation_date)
    cdt = datetime(_d.year, _d.month, _d.day, hour, minute)

    result = _route_sync(_graphs[osmnx_mode], _boundary, origin, dest, osmnx_mode, cdt)
    return args, result
