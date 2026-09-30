"""OTP linking check of a population: zero "Couldn't link" expected (ticket 031, criterion 2).

For each home and each distinct activity location of a population file, queries OTP
(Transmodel v3) for a public transport trip towards a reference point (the Capitole)
at a given time, and counts the `routingErrors` per code — `LOCATION_NOT_FOUND` is the
"Couldn't link" of OTP 2 (point too far from any street of the graph), `OUTSIDE_BOUNDS` a point outside
the extent of the graph, `NO_STOPS_IN_RANGE` a point with no reachable stop (not a graph defect:
the rural 3rd ring has no PT). The instances are queried in rotation.

    services/llm-agents/.venv/bin/python scripts/data/gtfs/otp_link_check.py \
        --population data/population/population_1000_PANEL_v4/population.json \
        --json docs/traces/<trace>/otp_link_check.json

⚠ **`--date-time` bypasses the runtime path.** Passing a local time by hand is
the right convention *for an instrument*, but it is precisely what made the time-zone
defect invisible from 2026-06 to 2026-09-04: this measurement queried OTP in local
time, the runtime sent it the same wall-clock time stamped UTC, and the two were
not talking about the same time. To measure what the agents see, pass
``--gama-timestamp``: the timestamp is then translated by ``sim_clock``, the SAME function
as `trip_helper/otp.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COURONNES_GEOJSON = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "couronne_perimetre.geojson"
# `services/llm-agents/` on the path: `--gama-timestamp` translates the timestamp with `sim_clock`,
# the module the runtime uses. Copying the conversion here would restore the asymmetry
# that hid the defect — an instrument that only fails if IT changes.
if str(REPO_ROOT / "services" / "llm-agents") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "services" / "llm-agents"))

DEFAULT_ENDPOINTS = ["http://localhost:8080/otp/transmodel/v3",
                     "http://localhost:8081/otp/transmodel/v3",
                     "http://localhost:8082/otp/transmodel/v3"]
CAPITOLE = (43.6045, 1.4440)
# The requested modes are those of the runtime (`services/llm-agents/trip_helper/otp.py`): a mode
# missing here would measure a supply the agents do not see, and conversely.
# `legs { mode authority }` is used to count the itineraries that offer a TRAIN —
# the only figure that tells whether adding the `rail` mode changes anything.
QUERY = """
query ($from: Location!, $to: Location!, $dateTime: DateTime, $numTripPatterns: Int) {
  trip(from: $from, to: $to, dateTime: $dateTime, numTripPatterns: $numTripPatterns,
       modes: {accessMode: foot, egressMode: foot, transportModes: [{transportMode: bus}, {transportMode: metro},
               {transportMode: tram}, {transportMode: rail}, {transportMode: cableway}]}) {
    tripPatterns { duration legs { mode authority { id name } } }
    routingErrors { code description inputField }
  }
}
"""
# Leg modes that count as train (transmodel v3).
MODES_TRAIN = {"rail"}


def _points(pop: list) -> tuple[list, list]:
    homes, acts, seen = [], [], set()
    for e in pop:
        ident = e.get("identity") or {}
        h = ident.get("home") or {}
        if h.get("lat") is not None:
            homes.append((e["person_id"], "home", float(h["lat"]), float(h["lon"])))
        for a in ident.get("activities") or []:
            if a.get("purpose") == "home":
                continue
            loc = a.get("location") or {}
            if loc.get("lat") is None:
                continue
            key = (round(float(loc["lat"]), 5), round(float(loc["lon"]), 5))
            if key in seen:
                continue
            seen.add(key)
            acts.append((e["person_id"], a.get("purpose", "?"), float(loc["lat"]), float(loc["lon"])))
    return homes, acts


def classer_par_couronne(points: list, geojson: Path = COURONNES_GEOJSON) -> dict:
    """point → ring of residence (Toulouse, 1st, 2nd, 3rd, or "outside the scope").

    The lack of service is not randomly distributed: this is the question raised
    by the scope report ("without liO, the outer rings only have a
    tenth of their PT supply"). An aggregated total cannot answer it.
    """
    from shapely import contains_xy, prepare
    from shapely.geometry import shape

    if not geojson.exists():
        return {}
    couronnes = []
    for feature in json.loads(geojson.read_text(encoding="utf-8"))["features"]:
        polygone = shape(feature["geometry"])
        prepare(polygone)
        couronnes.append((feature["properties"].get("couronne", "?"), polygone))
    classement = {}
    for index, (_pid, _kind, lat, lon) in enumerate(points):
        nom = "hors perimetre"
        for libelle, polygone in couronnes:
            if contains_xy(polygone, lon, lat):
                nom = libelle
                break
        classement[index] = nom
    return classement


async def _query(session, url, lat, lon, date_time, num_trip_patterns=1):
    variables = {"from": {"coordinates": {"latitude": lat, "longitude": lon}},
                 "to": {"coordinates": {"latitude": CAPITOLE[0], "longitude": CAPITOLE[1]}},
                 "dateTime": date_time, "numTripPatterns": num_trip_patterns}
    async with session.post(url, json={"query": QUERY, "variables": variables}) as resp:
        body = await resp.json()
    trip = (body.get("data") or {}).get("trip") or {}
    errors = [e.get("code") for e in trip.get("routingErrors") or []]
    if "errors" in body and not trip:
        errors = ["GRAPHQL_ERROR:" + str(body["errors"])[:80]]
    motifs = trip.get("tripPatterns") or []
    # Leg modes encountered, and itineraries carrying at least one train.
    modes, avec_train, autorites_train = Counter(), 0, Counter()
    for motif in motifs:
        modes_motif = {(leg.get("mode") or "").lower() for leg in motif.get("legs") or []}
        modes.update(modes_motif)
        if modes_motif & MODES_TRAIN:
            avec_train += 1
            for leg in motif.get("legs") or []:
                if (leg.get("mode") or "").lower() in MODES_TRAIN:
                    autorites_train[((leg.get("authority") or {}).get("name") or "?")] += 1
    return len(motifs), errors, modes, avec_train, autorites_train


async def run(points, endpoints, date_time, concurrency, couronnes=None, num_trip_patterns=1) -> dict:
    import aiohttp

    couronnes = couronnes or {}
    sem = asyncio.Semaphore(concurrency)
    codes, per_kind, no_pattern, link_failures = Counter(), Counter(), Counter(), []
    modes_totaux, autorites_train = Counter(), Counter()
    itineraires, itineraires_train, points_avec_train = 0, 0, 0
    par_couronne = defaultdict(lambda: {"points": 0, "sans_itineraire": 0, "erreurs": Counter(),
                                        "points_avec_train": 0, "itineraires": 0, "itineraires_avec_train": 0})
    t0 = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async def one(i, pt):
            nonlocal itineraires, itineraires_train, points_avec_train
            pid, kind, lat, lon = pt
            async with sem:
                try:
                    n, errs, modes, avec_train, autorites = await _query(
                        session, endpoints[i % len(endpoints)], lat, lon, date_time, num_trip_patterns)
                except Exception as exc:  # network, timeout
                    n, errs, modes, avec_train, autorites = -1, [f"EXC:{type(exc).__name__}"], Counter(), 0, Counter()
            per_kind[kind] += 1
            modes_totaux.update(modes)
            autorites_train.update(autorites)
            couronne = couronnes.get(i)
            if couronne:
                par_couronne[couronne]["points"] += 1
            if n > 0:
                itineraires += n
                if couronne:
                    par_couronne[couronne]["itineraires"] += n
            if avec_train:
                itineraires_train += avec_train
                points_avec_train += 1
                if couronne:
                    par_couronne[couronne]["itineraires_avec_train"] += avec_train
                    par_couronne[couronne]["points_avec_train"] += 1
            if n == 0:
                no_pattern[kind] += 1
                if couronne:
                    par_couronne[couronne]["sans_itineraire"] += 1
            for c in errs:
                codes[c] += 1
                if couronne:
                    par_couronne[couronne]["erreurs"][c] += 1
                if c in ("LOCATION_NOT_FOUND", "OUTSIDE_BOUNDS") or c.startswith("EXC") or c.startswith("GRAPHQL"):
                    link_failures.append({"person_id": pid, "kind": kind, "lat": lat, "lon": lon, "code": c})
        await asyncio.gather(*(one(i, pt) for i, pt in enumerate(points)))
    return {"points": len(points), "par_type": dict(per_kind), "sans_itineraire": dict(no_pattern),
            "routing_errors": dict(codes),
            "num_trip_patterns": num_trip_patterns,
            "itineraires_rendus": itineraires,
            "itineraires_avec_train": itineraires_train,
            "points_avec_au_moins_un_itineraire_train": points_avec_train,
            "modes_de_jambe": dict(modes_totaux.most_common()),
            "autorites_du_train": dict(autorites_train.most_common()),
            "par_couronne": {nom: {**valeurs, "erreurs": dict(valeurs["erreurs"])}
                             for nom, valeurs in sorted(par_couronne.items())},
            "echecs_de_rattachement": link_failures,
            "n_echecs_de_rattachement": len(link_failures), "duree_s": round(time.monotonic() - t0, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument("--date-time", default=None,
                        help="ISO-8601 local time passed as is to OTP (default: "
                             "2026-03-16T08:00:00+01:00, Monday 16 March 2026 8 am). ⚠ "
                             "bypasses the runtime conversion — see --gama-timestamp")
    parser.add_argument("--gama-timestamp", type=int, default=None,
                        help="timestamp published by GAMA (CURRENT_TIMESTAMP, e.g. "
                             "1773637200 = Monday 16 March 2026 5 am wall clock), translated by "
                             "`sim_clock` as `trip_helper/otp.py` does. This is the "
                             "RUNTIME PATH: the only way to measure the supply that "
                             "the agents actually see")
    parser.add_argument("--concurrency", type=int, default=9)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--sans-couronnes", action="store_true",
                        help="do not break the results down by ring of residence")
    parser.add_argument("--num-trip-patterns", type=int, default=1,
                        help="itineraries requested per point. 1 (default) = the linking "
                             "measurement, comparable to earlier readings; 6 = what "
                             "the runtime requests (settings.gtfs.max_trip_candidates), hence "
                             "what the agent is actually offered")
    args = parser.parse_args(argv)

    if args.date_time and args.gama_timestamp is not None:
        parser.error("--date-time and --gama-timestamp designate the same thing in two "
                     "ways: pass only one of them (prefer --gama-timestamp)")
    if args.gama_timestamp is not None:
        # The runtime path, down to the function.
        from sim_clock import network_iso, network_timezone_name, wall_clock

        date_time = network_iso(args.gama_timestamp)
        origine_heure = (f"--gama-timestamp {args.gama_timestamp} → sim_clock "
                         f"(heure murale {wall_clock(args.gama_timestamp).isoformat()}, "
                         f"fuseau du réseau {network_timezone_name()})")
    else:
        date_time = args.date_time or "2026-03-16T08:00:00+01:00"
        origine_heure = "--date-time (heure locale passée à la main, hors runtime)"

    pop = json.loads(args.population.read_text(encoding="utf-8"))
    homes, acts = _points(pop)
    points = homes + acts
    print(f"{len(pop)} personas: {len(homes)} homes, {len(acts)} distinct activity locations → OTP {date_time}",
          file=sys.stderr)
    print(f"requested time: {origine_heure}", file=sys.stderr)
    couronnes = {} if args.sans_couronnes else classer_par_couronne(points)
    result = asyncio.run(run(points, args.endpoints.split(","), date_time, args.concurrency,
                             couronnes, args.num_trip_patterns))
    result.update({"population": str(args.population), "date_time": date_time,
                   "origine_heure": origine_heure, "reference": CAPITOLE})
    print(json.dumps({k: v for k, v in result.items() if k != "echecs_de_rattachement"}, ensure_ascii=False, indent=1))
    if result["echecs_de_rattachement"]:
        print(f"[ALARME] {result['n_echecs_de_rattachement']} point(s) not linked to the OTP graph "
              f"(LOCATION_NOT_FOUND / OUTSIDE_BOUNDS)", file=sys.stderr)
        for f in result["echecs_de_rattachement"][:10]:
            print("  ", f, file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if result["echecs_de_rattachement"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
