"""
Filter TER Occitanie GTFS to trips that serve at least 2 stops within
TOULOUSE_OSM_ROUTES_30K_BBOX from services/llm-agents/geography.py.

This keeps all suburban and regional services (Toulouse↔Labège, Toulouse↔Cerbère…)
while excluding pure pass-through trains that make a single stop at Toulouse Matabiau
(Paris, Bayonne, Pau, etc.).

Output: toulouse/ter_gtfs.zip (OTP loads zip GTFS feeds automatically)

Usage:
    python services/otp-toulouse/tools/filter_ter_gtfs.py
"""

import csv
import io
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOULOUSE_OSM_ROUTES_30K_BBOX = (
    1.085,   # min_lon (west)
    43.336,  # min_lat (south)
    1.815,   # max_lon (east)
    43.868,  # max_lat (north)
)

REPO_ROOT = Path(__file__).resolve().parents[3]  # tools/ → services/otp-toulouse/ → services/ → root
# Source: full TER Occitanie GTFS (download from https://www.data.gouv.fr/fr/datasets/horaires-des-trains-ter/)
# Place the unzipped feed at services/otp-toulouse/source/ter_occitanie/
SRC_DIR = REPO_ROOT / "services" / "otp-toulouse" / "source" / "ter_occitanie"
# OTP only loads GTFS zips sitting at the ROOT of its build directory, which is
# `data/gtfs` (see services/otp-toulouse/Makefile: `--build ../data/gtfs`). Writing the
# zip under services/otp-toulouse/toulouse/ left it outside the build entirely — which is
# one reason the TER has never been in graph.obj.
OUT_ZIP = REPO_ROOT / "data" / "gtfs" / "ter_gtfs.zip"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def in_bbox(lat: float, lon: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = TOULOUSE_OSM_ROUTES_30K_BBOX
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # 1. Stops within bbox
    print("Loading stops…")
    stops_fields, stops_rows = read_csv(SRC_DIR / "stops.txt")
    bbox_stop_ids: set[str] = set()
    for row in stops_rows:
        try:
            lat = float(row["stop_lat"])
            lon = float(row["stop_lon"])
        except (ValueError, KeyError):
            continue
        if in_bbox(lat, lon):
            bbox_stop_ids.add(row["stop_id"])
    print(f"  {len(bbox_stop_ids)} stops within bbox (out of {len(stops_rows)})")

    # 2. Load stop_times grouped by trip_id (sorted by stop_sequence)
    print("Loading stop_times…")
    st_fields, st_rows = read_csv(SRC_DIR / "stop_times.txt")

    trip_stops: dict[str, list[dict]] = {}
    for row in st_rows:
        trip_stops.setdefault(row["trip_id"], []).append(row)

    # Sort each trip's stops by sequence
    for trip_id in trip_stops:
        trip_stops[trip_id].sort(key=lambda r: int(r["stop_sequence"]))

    # 3. Keep trips that serve at least 2 stops within bbox.
    MIN_BBOX_STOPS = 2
    print(f"Filtering trips (≥ {MIN_BBOX_STOPS} stops in bbox)…")
    kept_trip_ids: set[str] = set()
    for trip_id, stops in trip_stops.items():
        n_in_bbox = sum(1 for s in stops if s["stop_id"] in bbox_stop_ids)
        if n_in_bbox >= MIN_BBOX_STOPS:
            kept_trip_ids.add(trip_id)
    print(f"  {len(kept_trip_ids)} trips kept (out of {len(trip_stops)})")

    # 4. Filtered stop_times
    kept_st = [r for r in st_rows if r["trip_id"] in kept_trip_ids]

    # Stops actually referenced by kept trips
    used_stop_ids: set[str] = {r["stop_id"] for r in kept_st}

    # 5. Trips
    trips_fields, trips_rows = read_csv(SRC_DIR / "trips.txt")
    kept_trips = [r for r in trips_rows if r["trip_id"] in kept_trip_ids]
    kept_route_ids = {r["route_id"] for r in kept_trips}
    kept_service_ids = {r["service_id"] for r in kept_trips}

    # 6. Routes
    routes_fields, routes_rows = read_csv(SRC_DIR / "routes.txt")
    kept_routes = [r for r in routes_rows if r["route_id"] in kept_route_ids]
    kept_agency_ids = {r["agency_id"] for r in kept_routes}

    # 7. Calendar dates
    cal_fields, cal_rows = read_csv(SRC_DIR / "calendar_dates.txt")
    kept_cal = [r for r in cal_rows if r["service_id"] in kept_service_ids]

    # 8. Agency (keep all referenced)
    agency_fields, agency_rows = read_csv(SRC_DIR / "agency.txt")
    kept_agency = [r for r in agency_rows if r["agency_id"] in kept_agency_ids]

    # 9. Stops (only those actually used)
    kept_stops = [r for r in stops_rows if r["stop_id"] in used_stop_ids]

    # 10. Transfers (only between kept stops)
    transfers_fields, transfers_rows = read_csv(SRC_DIR / "transfers.txt")
    kept_transfers = [
        r for r in transfers_rows
        if r["from_stop_id"] in used_stop_ids and r["to_stop_id"] in used_stop_ids
    ]

    # 11. feed_info: copy as-is
    feed_fields, feed_rows = read_csv(SRC_DIR / "feed_info.txt")

    # 12. Empty calendar.txt — OTP 2.x uses it as a GTFS feed detection marker
    cal_header = ["service_id","monday","tuesday","wednesday","thursday",
                  "friday","saturday","sunday","start_date","end_date"]

    # ---------------------------------------------------------------------------
    # Write output as zip (OTP requires zip for GTFS feeds)
    # ---------------------------------------------------------------------------
    print(f"Writing filtered GTFS to {OUT_ZIP} …")
    files = {
        "stops.txt":          (stops_fields,   kept_stops),
        "stop_times.txt":     (st_fields,      kept_st),
        "trips.txt":          (trips_fields,   kept_trips),
        "routes.txt":         (routes_fields,  kept_routes),
        "calendar_dates.txt": (cal_fields,     kept_cal),
        "calendar.txt":       (cal_header,     []),
        "agency.txt":         (agency_fields,  kept_agency),
        "transfers.txt":      (transfers_fields, kept_transfers),
        "feed_info.txt":      (feed_fields,    feed_rows),
    }
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, (fields, rows) in files.items():
            zf.writestr(name, csv_bytes(fields, rows))

    print("\nSummary:")
    print(f"  stops          : {len(kept_stops)}")
    print(f"  trips          : {len(kept_trips)}")
    print(f"  stop_times     : {len(kept_st)}")
    print(f"  routes         : {len(kept_routes)}")
    print(f"  calendar_dates : {len(kept_cal)}")
    print(f"  transfers      : {len(kept_transfers)}")
    print(f"  zip size       : {OUT_ZIP.stat().st_size / 1024:.1f} kB")
    print("\nDone. Rebuild OTP graph with: make build-graph")


if __name__ == "__main__":
    if not SRC_DIR.exists():
        print(f"ERROR: source directory not found: {SRC_DIR}", file=sys.stderr)
        sys.exit(1)
    main()
