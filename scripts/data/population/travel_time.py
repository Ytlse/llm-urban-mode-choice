"""
Travel time calculator for the Toulouse metropolitan area.

Usage
-----
    from travel_time import TravelTimeCalculator
    from datetime import datetime

    calc = TravelTimeCalculator()           # build graphs once (~1 min)
    print(calc.get_travel(
        start=(1.4442, 43.6047),            # (longitude, latitude)
        end=(1.3521, 43.5808),
        mode="drive",
        date=datetime(2024, 5, 6, 8, 0),   # Monday 8 am
        parking_dispo=False,
    ))

    # Or use the bare function with pre-built objects:
    from travel_time import build_graphs, build_city_boundary, get_travel
    graphs   = build_graphs()
    boundary = build_city_boundary()
    get_travel(start, end, "walk", datetime.now(), graphs=graphs, city_boundary=boundary)
"""

import hashlib
import pickle
from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd
from datetime import datetime
from shapely.geometry import Point

_CACHE_DIR = Path(__file__).parent / "cache"

# ──────────────────────────────────────────────────────────────────────────────
# Speed & penalty tables  (same as notebook)
# ──────────────────────────────────────────────────────────────────────────────

MODES = ("walk", "bike", "drive")

_SPEEDS = {
    "walk":  {"residential": 5,  "tertiary": 5,  "footway": 4.5, "pedestrian": 4.5,
              "path": 4.5, "steps": 2, "living_street": 5, "service": 5,
              "unclassified": 5, "secondary": 5, "primary": 5},
    "bike":  {"residential": 15, "tertiary": 18, "cycleway": 20, "living_street": 10,
              "secondary": 16, "primary": 16, "unclassified": 14, "path": 10},
    "drive": {"motorway": 100, "trunk": 80, "primary": 50, "secondary": 45,
              "tertiary": 40, "residential": 25, "unclassified": 30, "service": 15},
}
_FALLBACKS = {"walk": 5, "bike": 14, "drive": 30}

# Seconds added per infrastructure element encountered on the route
_PENALTIES = {
    "drive": {"traffic_signals": 20, "stop": 5, "give_way": 2,
              "crossing": 0,  "mini_roundabout": 3},
    "bike":  {"traffic_signals": 15, "stop": 2, "give_way": 1,
              "crossing": 0,  "mini_roundabout": 2},
    "walk":  {"traffic_signals": 12, "stop": 0, "give_way": 0,
              "crossing": 8,  "mini_roundabout": 0},
}

# Base parking time (seconds, applied once for start + once for end)
_PARK_BASE = {"drive": 120, "bike": 60, "walk": 0}
# Multiplier when no parking is available (search time)
_PARK_NO_SPOT = 4.0

# ──────────────────────────────────────────────────────────────────────────────
# TomTom congestion tables  (source: tomtom.com/traffic-index/toulouse)
# Rows = hours 00-23, columns = Mon Tue Wed Thu Fri Sat Sun
# ──────────────────────────────────────────────────────────────────────────────

_DAYS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_HOURS = [f"{h:02d}:00" for h in range(24)]

# City (commune de Toulouse)
_CITY_RAW = [
    45,45,44,44,43,43,44,  44,43,42,42,42,43,44,  # 00–01
    45,44,43,42,42,42,43,  48,48,47,46,43,43,42,  # 02–03
    54,53,52,51,48,46,44,  57,56,55,54,52,48,45,  # 04–05
    57,57,57,57,57,52,49,  37,37,39,38,44,54,52,  # 06–07
    28,28,32,29,38,55,54,  37,37,40,38,44,51,52,  # 08–09
    45,44,44,44,44,48,50,  45,43,43,43,43,45,48,  # 10–11
    44,42,41,42,41,44,47,  44,43,42,43,44,45,48,  # 12–13
    44,43,42,43,43,45,48,  44,43,42,43,40,44,48,  # 14–15
    38,36,37,36,32,44,48,  31,28,32,29,28,43,47,  # 16–17
    34,31,33,31,33,42,46,  41,39,38,38,38,41,44,  # 18–19
    42,42,41,41,41,41,44,  42,42,41,41,42,41,44,  # 20–21
    42,42,42,42,42,42,44,  44,43,43,42,43,43,45,  # 22–23
]

# Metropolis (Toulouse Métropole)
_METRO_RAW = [
    50,49,48,48,48,48,48,  49,48,47,46,46,48,48,  # 00–01
    50,49,48,47,47,47,47,  53,52,51,50,48,47,47,  # 02–03
    57,56,55,54,52,50,48,  59,58,58,57,56,53,49,  # 04–05
    58,57,57,57,58,56,53,  39,39,42,40,46,58,56,  # 06–07
    32,32,36,33,41,57,57,  41,41,44,42,47,53,55,  # 08–09
    48,46,46,47,47,49,52,  47,46,45,46,46,47,51,  # 10–11
    46,45,43,45,44,47,51,  47,46,45,46,46,48,52,  # 12–13
    47,46,45,46,46,48,53,  46,45,45,46,44,46,53,  # 14–15
    41,39,41,40,37,46,53,  35,33,36,34,34,45,52,  # 16–17
    38,35,37,36,39,45,51,  45,43,43,42,43,45,50,  # 18–19
    46,46,45,45,46,45,50,  46,46,46,46,47,45,49,  # 20–21
    46,46,46,46,47,46,49,  48,47,47,47,47,47,50,  # 22–23
]

CITY_SPEEDS_DF  = pd.DataFrame(
    [_CITY_RAW[i:i+7]  for i in range(0, len(_CITY_RAW),  7)],
    index=_HOURS, columns=_DAYS,
)
METRO_SPEEDS_DF = pd.DataFrame(
    [_METRO_RAW[i:i+7] for i in range(0, len(_METRO_RAW), 7)],
    index=_HOURS, columns=_DAYS,
)

# Best observed speed = free-flow reference (used to compute the congestion ratio)
_CITY_FREE_FLOW  = float(CITY_SPEEDS_DF.values.max())   # 57 km/h
_METRO_FREE_FLOW = float(METRO_SPEEDS_DF.values.max())  # 59 km/h


# ──────────────────────────────────────────────────────────────────────────────
# Graph & boundary builders  (call once, reuse)
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(city: str, dist: int) -> str:
    return hashlib.md5(f"{city}_{dist}".encode()).hexdigest()[:12]


def build_graphs(city: str = "Toulouse, France", dist: int = 15_000) -> dict:
    """Load graphs from disk cache if available, otherwise download and cache them."""
    key = _cache_key(city, dist)
    cache_path = _CACHE_DIR / f"graphs_{key}.pkl"

    if cache_path.exists():
        print(f"Loading graphs from cache ({cache_path.name}) …")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    print(f"Downloading OSM graphs for {city} (radius {dist} m) …")
    graphs = {}
    for mode in MODES:
        G = ox.graph_from_address(city, dist=dist, network_type=mode)
        for _, _, _, data in G.edges(keys=True, data=True):
            hwy = data.get("highway")
            if isinstance(hwy, list):
                hwy = hwy[0]
            data["speed_kph"] = _SPEEDS[mode].get(hwy, _FALLBACKS[mode])
        G = ox.add_edge_travel_times(G)
        graphs[mode] = G

    _CACHE_DIR.mkdir(exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(graphs, f)
    print(f"Graphs saved to cache ({cache_path.name}).")
    return graphs


def build_city_boundary(city: str = "Toulouse, France") -> gpd.GeoDataFrame:
    """Load boundary from disk cache if available, otherwise download and cache it."""
    key = hashlib.md5(city.encode()).hexdigest()[:12]
    cache_path = _CACHE_DIR / f"boundary_{key}.pkl"

    if cache_path.exists():
        print(f"Loading boundary from cache ({cache_path.name}) …")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    print(f"Downloading city boundary for {city} …")
    gdf = ox.geocode_to_gdf(city).to_crs("EPSG:4326")

    _CACHE_DIR.mkdir(exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(gdf, f)
    print(f"Boundary saved to cache ({cache_path.name}).")
    return gdf


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _point_in_city(lon: float, lat: float, boundary: gpd.GeoDataFrame) -> bool:
    return boundary.geometry.iloc[0].contains(Point(lon, lat))


def _congestion(date: datetime, in_city: bool) -> tuple[float, str]:
    """Return (slowdown_factor, label).  factor > 1 → slower than free-flow."""
    day = date.strftime("%a")           # "Mon" … "Sun"
    hour = f"{date.hour:02d}:00"

    if in_city:
        speed = float(CITY_SPEEDS_DF.loc[hour, day])
        factor = _CITY_FREE_FLOW / speed
    else:
        speed = float(METRO_SPEEDS_DF.loc[hour, day])
        factor = _METRO_FREE_FLOW / speed

    ratio = 1.0 / factor  # fraction of free-flow
    if ratio >= 0.85:
        label = "light traffic"
    elif ratio >= 0.70:
        label = "moderate traffic"
    elif ratio >= 0.55:
        label = "heavy traffic"
    else:
        label = "severe congestion"

    return factor, label


def _infra_penalty(G, route_nodes: list, mode: str) -> float:
    """Sum of delay (seconds) from traffic signals, crossings, etc."""
    counts: dict[str, int] = {k: 0 for k in _PENALTIES[mode]}
    for n in route_nodes[1:-1]:
        tag = G.nodes[n].get("highway")
        if tag in counts:
            counts[tag] += 1
    return sum(counts[k] * _PENALTIES[mode][k] for k in counts)


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def get_travel(
    start: tuple,
    end: tuple,
    mode: str,
    date: datetime,
    parking_dispo: bool = True,
    graphs: dict | None = None,
    city_boundary: gpd.GeoDataFrame | None = None,
) -> str:
    """
    Calculate travel time and return a human-readable English description.

    Parameters
    ----------
    start, end      : (longitude, latitude) tuples
    mode            : "walk" | "bike" | "drive"
    date            : datetime – day-of-week and hour are used for congestion
    parking_dispo   : if False, adds a significant parking search penalty (drive only)
    graphs          : pre-built dict from build_graphs() – avoids re-downloading OSM data
    city_boundary   : pre-built GeoDataFrame from build_city_boundary()
    """

    start_time = datetime.now()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")

    if graphs is None:
        graphs = build_graphs()
    if city_boundary is None:
        city_boundary = build_city_boundary()

    G = graphs[mode]
    orig = ox.distance.nearest_nodes(G, start[0], start[1])
    dest = ox.distance.nearest_nodes(G, end[0], end[1])

    route_nodes = ox.routing.shortest_path(G, orig, dest, weight="travel_time")
    if route_nodes is None:
        return f"No {mode} route could be found between the two points."

    gdf = ox.routing.route_to_gdf(G, route_nodes)
    distance_km  = gdf["length"].sum() / 1_000
    free_flow_s  = gdf["travel_time"].sum()

    # Infrastructure penalties (signals, stops, crossings …)
    infra_s = _infra_penalty(G, route_nodes, mode)

    # Parking – applied at destination (and start for drive)
    dest_in_city = _point_in_city(end[0], end[1], city_boundary)
    park_s = 0.0
    if mode in ("drive", "bike"):
        base = _PARK_BASE[mode]
        if mode == "drive" and not parking_dispo and dest_in_city:
            base *= _PARK_NO_SPOT
        park_s = base * 2  # once to park, once to retrieve

    # Congestion factor (drive only – bikes and pedestrians ignore car traffic)
    cong_factor, cong_label = 1.0, ""
    if mode == "drive":
        origin_in_city = _point_in_city(start[0], start[1], city_boundary)
        in_city = origin_in_city or dest_in_city
        cong_factor, cong_label = _congestion(date, in_city)

    total_s  = free_flow_s * cong_factor + infra_s + park_s
    total_min = max(1, round(total_s / 60))

    print(f"Elapsed time {datetime.now() - start_time}")
    return _format(mode, total_min, distance_km, cong_label,
                   dest_in_city, parking_dispo)


# ──────────────────────────────────────────────────────────────────────────────
# Text formatting
# ──────────────────────────────────────────────────────────────────────────────

def _format(mode, minutes, km, cong_label, dest_in_city, parking_dispo):
    labels = {"walk": "On foot", "bike": "By bike", "drive": "By car"}
    t = f"{minutes} minute{'s' if minutes != 1 else ''}"
    d = f"{km:.1f} km"

    msg = f"{labels[mode]}, travel time is {t} for {d}"

    if mode == "drive":
        cong_text = {
            "light traffic":    "traffic is flowing freely",
            "moderate traffic": "moderate traffic expected",
            "heavy traffic":    "significant traffic — expect delays",
            "severe congestion":"severe congestion — major delays likely",
        }
        msg += f", {cong_text.get(cong_label, cong_label)}"

        if dest_in_city and not parking_dispo:
            msg += "; no parking nearby, allow extra time to find a spot"

    return msg + "."


# ──────────────────────────────────────────────────────────────────────────────
# Convenience class (pre-builds everything once)
# ──────────────────────────────────────────────────────────────────────────────

class TravelTimeCalculator:
    """Pre-build OSMnx graphs and boundary once; call get_travel() freely after."""

    def __init__(self, city: str = "Toulouse, France", dist: int = 15_000):
        self.graphs = build_graphs(city, dist)
        self.city_boundary = build_city_boundary(city)
        print("Ready.")

    def get_travel(
        self,
        start: tuple,
        end: tuple,
        mode: str,
        date: datetime,
        parking_dispo: bool = True,
    ) -> str:
        print(f"\nCalculating travel time from {start} to {end} by {mode} at {date.strftime('%a')} (parking dispo: {parking_dispo}) …")
        return get_travel(
            start, end, mode, date, parking_dispo,
            graphs=self.graphs,
            city_boundary=self.city_boundary,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    calc = TravelTimeCalculator()

    start = (1.479092839650062, 43.60289964770299)
    end = (1.431665487237536, 43.605701457103905)

    for mode in ("walk", "bike", "drive"):
        print(calc.get_travel(
            start, end, mode,
            date=datetime(2026, 5, 6, 8, 0),   # Monday 8 am
            parking_dispo=True,
        ))

    # Monday 8 am, no parking
    print(calc.get_travel(start, end, "drive",
                          date=datetime(2026, 5, 4, 8, 0),
                          parking_dispo=False))
    

    # Monday 10 am, parking
    print(calc.get_travel(start, end, "drive",
                          date=datetime(2026, 5, 4, 10, 0),
                          parking_dispo=True))
    
    # Sunday 10 am, parking
    print(calc.get_travel(start, end, "drive",
                          date=datetime(2026, 5, 3, 10, 0),
                          parking_dispo=True))


