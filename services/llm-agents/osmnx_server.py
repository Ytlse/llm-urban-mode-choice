"""
Standalone OSMnx routing microservice.

Exposes POST /route and GET /health. Graphs are loaded from disk cache at
startup (building from OSM if not cached) and reused across requests.

Running as a dedicated, non-daemon process means ProcessPoolExecutors are
used for routing (true parallelism, no GIL) — unlike the embedded mode
inside the controller where hypercorn workers are daemon processes.
"""

import asyncio
import functools
import os
import time
from contextlib import asynccontextmanager
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from models import Location
from trip_helper.osmnx_direct import (
    _GraphStore,
    _MODE_POOLS,
    _MODE_TO_OSMNX,
    _in_daemon,
    _proc_mem_mb,
    _reset_pool,
    _route_sync,
    _route_sync_process,
    graph_key,
    graph_label,
)

# Two fixed Toulouse points used only for warmup routing.
_WARMUP_ORIGIN = Location(lat=43.6047, lon=1.4442)
_WARMUP_DEST   = Location(lat=43.6200, lon=1.4600)

# Running request counter per mode (best-effort, not thread-safe but fine for logging).
_inflight: dict[str, int] = {m: 0 for m in _MODE_TO_OSMNX}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"[osmnx-server] startup  pid={os.getpid()}  daemon={_in_daemon}  {_proc_mem_mb()}")

    # Graph served: the one of `graph_key()` — the polygon of the 453 communes by default (ticket 031,
    # part 2). Its pickle must be in the mounted cache; nothing is downloaded in its place.
    cache_dir = str(_GraphStore._cache_dir())
    cache_key = graph_key()
    g_path    = Path(cache_dir) / f"graphs_{cache_key}.pkl"
    if not g_path.exists():
        logger.error(
            f"[ALARME] [osmnx-server] graph {cache_key} ({graph_label(cache_key)}) not found: "
            f"{g_path}. The service does not start — mount data/cache/osmnx and build the graph "
            "(`make osmnx-perimeter-graph`).")
        raise RuntimeError(f"OSMnx graph {cache_key} missing: {g_path}")
    logger.info(f"[osmnx-server] graph {cache_key} ({graph_label(cache_key)}) — "
                f"{g_path.name} {g_path.stat().st_size / 1_048_576:.0f} MB in {cache_dir}")
    loop      = asyncio.get_running_loop()

    for mode, osmnx_mode in _MODE_TO_OSMNX.items():
        fn = functools.partial(
            _route_sync_process,
            cache_dir, cache_key,
            _WARMUP_ORIGIN, _WARMUP_DEST,
            osmnx_mode, datetime.now(),
        )
        logger.info(f"[osmnx-server] warmup start  mode={mode}  {_proc_mem_mb()}")
        t0 = time.monotonic()
        try:
            await loop.run_in_executor(_MODE_POOLS[osmnx_mode], fn)
            logger.info(
                f"[osmnx-server] warmup OK  mode={mode}  elapsed={time.monotonic()-t0:.1f}s  {_proc_mem_mb()}"
            )
        except BrokenProcessPool:
            logger.warning(
                f"[osmnx-server] warmup CRASH (BrokenProcessPool)  mode={mode}  "
                f"elapsed={time.monotonic()-t0:.1f}s  {_proc_mem_mb()} — resetting pool"
            )
            _reset_pool(osmnx_mode)
        except Exception as exc:
            logger.warning(
                f"[osmnx-server] warmup ERROR  mode={mode}  exc={exc!r}  "
                f"elapsed={time.monotonic()-t0:.1f}s  {_proc_mem_mb()}"
            )

    logger.info(f"[osmnx-server] all warmups done  {_proc_mem_mb()}")
    yield
    logger.info("[osmnx-server] shutdown")


app = FastAPI(lifespan=lifespan)


class _LatLon(BaseModel):
    lat: float
    lon: float


class _RouteRequest(BaseModel):
    origin: _LatLon
    destination: _LatLon
    mode: str  # "foot" | "bicycle" | "car"
    congestion_dt: datetime


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/route")
async def route(req: _RouteRequest) -> Optional[dict]:
    origin      = Location(lat=req.origin.lat,      lon=req.origin.lon)
    destination = Location(lat=req.destination.lat, lon=req.destination.lon)
    osmnx_mode  = _MODE_TO_OSMNX[req.mode]
    loop        = asyncio.get_running_loop()

    _inflight[req.mode] = _inflight.get(req.mode, 0) + 1
    inflight_now = _inflight[req.mode]
    t0 = time.monotonic()
    # logger.debug(
    #     f"[osmnx-server] request  mode={req.mode}  inflight={inflight_now}  "
    #     f"from=({req.origin.lat:.5f},{req.origin.lon:.5f})  "
    #     f"to=({req.destination.lat:.5f},{req.destination.lon:.5f})"
    # )

    if _in_daemon:
        graphs, boundary = await _GraphStore.get()
        fn = functools.partial(
            _route_sync,
            graphs[osmnx_mode], boundary, origin, destination, osmnx_mode, req.congestion_dt,
        )
    else:
        cache_dir = str(_GraphStore._cache_dir())
        cache_key = graph_key()
        fn = functools.partial(
            _route_sync_process,
            cache_dir, cache_key, origin, destination, osmnx_mode, req.congestion_dt,
        )

    try:
        result = await loop.run_in_executor(_MODE_POOLS[osmnx_mode], fn)
        elapsed = time.monotonic() - t0
        # logger.debug(f"[osmnx-server] done  mode={req.mode}  elapsed={elapsed:.3f}s")
        return result
    except BrokenProcessPool:
        elapsed = time.monotonic() - t0
        logger.warning(
            f"[osmnx-server] BrokenProcessPool  mode={req.mode}  elapsed={elapsed:.1f}s  "
            f"{_proc_mem_mb()} — resetting and retrying"
        )
        _reset_pool(osmnx_mode)
        try:
            result = await loop.run_in_executor(_MODE_POOLS[osmnx_mode], fn)
            logger.info(f"[osmnx-server] retry OK  mode={req.mode}  total={time.monotonic()-t0:.1f}s")
            return result
        except BrokenProcessPool:
            logger.error(
                f"[osmnx-server] BrokenProcessPool AGAIN after reset  mode={req.mode}  "
                f"{_proc_mem_mb()} — returning 503"
            )
            raise HTTPException(status_code=503, detail=f"OSMnx worker crashed for mode={req.mode}")
    finally:
        _inflight[req.mode] = max(0, _inflight.get(req.mode, 1) - 1)
