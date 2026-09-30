"""Synthetic school bus — option factory (ticket 030).

A function, not a class: the option is purely synthetic (no OTP
or OSMnx call), it does not need the :class:`TripHelper` interface. It produces a
single-leg :class:`TravelPlan`, presented to the model like the other options.

Modelling choices (cf. ticket 030, decisions of 2026-09-03):

- **Eligibility** = age 5-17 (liO regulation) + home outside the Tisséo service area (where
  the Region offers the service, proxy ``home.public_transport is False``) + trip
  linked to the ``education`` activity. Neither catchment areas nor distance threshold.
- **Mode / GAMA rendering**: the leg carries ``mode="school_bus"`` (read by all the
  metric tables → counted as transit) and ``transit_route="__DIRECT_CAR__"`` (GAMA
  interpolates it point-to-point like a car, without GAMA editing — GAMA lot out of
  scope, the agent is displayed in red). The stops carry **non-empty names**
  so that ``get_code()`` differs from that of a real car (deduplication
  anti-collision).
- **Duration / schedule / cost**: frozen exogenous parameters of ``config/school_bus.yaml``.

Pure module: YAML read at first call, no mutable state, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from prometheus_client import Counter

from helper import to_timestamp_based_on_day
from models import Activity, Location, Person, Transit, TransitLocation, TravelPlan
from utils import random_uuid

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "school_bus.yaml"

# GAMA interpolates a leg whose ``transit_route`` equals this marker exactly like
# a car (point-to-point, without a GTFS vehicle). We reuse it so as NOT to have
# to touch GAMA (GAMA lot out of scope). Trade-off: red rendering.
SCHOOL_BUS_ROUTE_MARKER = "__DIRECT_CAR__"

SCHOOL_BUS_OPTIONS = Counter(
    "school_bus_options_total",
    "Synthetic school bus options produced (ticket 030)",
    ["direction"],  # 'outbound' | 'return'
)

# "Chosen" part of the logging (Lot B): how many proposed school bus options
# were actually selected by the LLM. Comparing it with
# `school_bus_options_total` gives the adoption rate of the mode.
SCHOOL_BUS_CHOSEN = Counter(
    "school_bus_chosen_total",
    "School bus options selected by the model (ticket 030)",
    ["direction"],  # 'outbound' | 'return'
)


def is_school_bus_plan(plan) -> bool:
    """True if the selected plan is a synthetic school bus (leg mode=school_bus)."""
    return bool(plan) and any(
        (leg.mode or "") == "school_bus" for leg in (getattr(plan, "legs", None) or [])
    )


@dataclass(frozen=True)
class _SchoolBusConfig:
    age_min: int
    age_max: int
    access_minutes: float
    detour_factor: float
    in_vehicle_speed_kmh: float
    ramassage_minutes: float
    schedule_margin_minutes: float


_CONFIG: Optional[_SchoolBusConfig] = None


def _config() -> _SchoolBusConfig:
    global _CONFIG
    if _CONFIG is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _CONFIG = _SchoolBusConfig(
            age_min=int(raw["age_min"]),
            age_max=int(raw["age_max"]),
            access_minutes=float(raw["access_minutes"]),
            detour_factor=float(raw["detour_factor"]),
            in_vehicle_speed_kmh=float(raw["in_vehicle_speed_kmh"]),
            ramassage_minutes=float(raw["ramassage_minutes"]),
            schedule_margin_minutes=float(raw["schedule_margin_minutes"]),
        )
    return _CONFIG


def _age_of(traits: Optional[dict]) -> Optional[int]:
    if not traits:
        return None
    try:
        return int(traits.get("age"))
    except (TypeError, ValueError):
        return None


def _haversine_km(a: Location, b: Location) -> float:
    """Straight-line distance in km between two points (lat/lon in degrees)."""
    r = 6371.0088
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi = math.radians(b.lat - a.lat)
    dlmb = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _same_loc(a: Optional[Location], b: Optional[Location], tol: float = 1e-6) -> bool:
    return (
        a is not None and b is not None
        and abs(a.lat - b.lat) <= tol and abs(a.lon - b.lon) <= tol
    )


def build_school_bus_option(
    person: Person,
    from_location: Optional[Location],
    next_activity: Activity,
    timestamp: int,
    departure_time: int,
) -> Optional[TravelPlan]:
    """Produce the school bus option for this trip, or ``None`` if not eligible.

    Eligible if: persona aged 5-17, home outside the Tisséo service area
    (``home.public_transport is False``), and trip one end of which is
    the persona's ``education`` activity (outbound: destination = school; return:
    origin = school).
    """
    cfg = _config()
    dest = next_activity.location
    if from_location is None or dest is None:
        return None

    # 1. Age (the only individual criterion retained).
    age = _age_of(person.identity.traits_json)
    if age is None or not (cfg.age_min <= age <= cfg.age_max):
        return None

    # 2. Zone: home outside the Tisséo service area (proxy: no Tisséo stop within ≤ 1.5 km).
    home = person.identity.home
    if home is None or home.public_transport is not False:
        return None

    # 3. Study activity of the persona (school destination or known place of study).
    edu = next(
        (a for a in (person.identity.activities or [])
         if (a.purpose or "").lower() == "education" and a.location is not None),
        None,
    )
    if edu is None:
        return None

    # 4. Direction: the destination is the school (outbound) or the origin is (return).
    dest_is_school = (next_activity.purpose or "").lower() == "education"
    origin_is_school = _same_loc(from_location, edu.location)
    if not (dest_is_school or origin_is_school):
        return None

    # 5. Straight-line distance (no graph: zero OSMnx impact).
    d_km = _haversine_km(from_location, dest)

    # 6. Duration: access + (distance × detour / speed) + pick-up.
    dur_s = int(round(
        cfg.access_minutes * 60
        + (d_km * cfg.detour_factor / cfg.in_vehicle_speed_kmh) * 3600
        + cfg.ramassage_minutes * 60
    ))

    # 7. Schedule, aligned on the school activity with the margin.
    margin_s = int(cfg.schedule_margin_minutes * 60)
    if dest_is_school:
        school_24h = next_activity.scheduled_start_time
        if school_24h is None:
            school_24h = next_activity.start_time
        school_abs = to_timestamp_based_on_day(int(school_24h), timestamp)
        end_abs = school_abs - margin_s          # arrives 30 min before the start
        start_abs = end_abs - dur_s
        direction = "outbound"
    else:  # return: leaves 30 min after the end of school
        school_abs = to_timestamp_based_on_day(int(edu.end_time), timestamp)
        start_abs = school_abs + margin_s
        end_abs = start_abs + dur_s
        direction = "return"

    # D+1 loop, like the controller's departure_time computation.
    if start_abs < timestamp:
        start_abs += 86400
        end_abs += 86400

    # 8. Single-leg plan. Named stops → get_code() distinct from a car.
    school_stop = "École"
    home_stop = "Arrêt car scolaire"
    if dest_is_school:
        leg_start = TransitLocation(stop=home_stop, lat=from_location.lat, lon=from_location.lon)
        leg_end = TransitLocation(stop=school_stop, lat=dest.lat, lon=dest.lon)
    else:
        leg_start = TransitLocation(stop=school_stop, lat=from_location.lat, lon=from_location.lon)
        leg_end = TransitLocation(stop=home_stop, lat=dest.lat, lon=dest.lon)

    leg = Transit(
        start_time=int(start_abs) * 1000,
        end_time=int(end_abs) * 1000,
        start_location=leg_start,
        end_location=leg_end,
        is_transfer=False,
        transit_route=SCHOOL_BUS_ROUTE_MARKER,
        shape_id=None,
        duration=dur_s,
        distance=d_km * 1000.0,
        mode="school_bus",
    )
    plan = TravelPlan(
        id=random_uuid(),
        start_location=from_location,
        end_location=dest,
        start_time=int(start_abs) * 1000,
        end_time=int(end_abs) * 1000,
        start_in=int(start_abs - departure_time),
        duration=dur_s,
        distance=d_km * 1000.0,
        legs=[leg],
    )
    SCHOOL_BUS_OPTIONS.labels(direction=direction).inc()
    logger.debug(
        f"[school_bus] option {direction} for {person.person_id} "
        f"(age {age}, {d_km:.1f} km, {dur_s // 60} min)"
    )
    return plan
