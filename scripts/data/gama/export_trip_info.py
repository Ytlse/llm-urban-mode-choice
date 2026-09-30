"""Produces `services/GAMA/CityTransport/includes/trip_info.json` — the runs GAMA operates
— and `shape_lookup.json`, the table through which an itinerary designates those runs.

    services/llm-agents/.venv/bin/python scripts/data/gama/export_trip_info.py
    make gama-trip-info                     # the layers THEN the runs

WHY A RECIPE
------------
`trip_info.json` had none. It was produced by hand by the `__main__`
block of `services/llm-agents/inputs/gtfs/gama.py`, which hard-reads
`../data/gtfs/tisseo_gtfs/` and writes into `../data/exports/gtfs/`. Result
measured on 2026-09-04: the file in service dated from **27 May**, carried
**39,343 runs from Tisséo alone and none with `route_type=2`**, while
`routes.shp` drew 34 TER lines and `stops.shp` 68 stations. GAMA thus drew
a rail system on which **no train would ever run** — and a visible, dead
line reads as a line with no service, not as missing data.

WHAT THE RECIPE GUARANTEES
--------------------------
1. **The three transit systems.** Tisséo, TER and liO, read from the same places as
   `export_gtfs_layers.py` (`FEEDS_DEFAUT`) — hence the same lines in the
   layers and in the runs.
2. **The simulated date is in the calendar.** It is read from
   `Settings.gaml` (`starting_date`), not assumed. Outside the calendar,
   `is_trip_available_today` merely emits a `warn` and schedules
   **no** run at all: the simulation runs, the transit system is empty, and nothing
   says so. The recipe fails rather than deliver that file.
3. **The window fits in the bit mask.** GAMA's calendar is a
   64-bit mask (`assert len(all_dates) <= 64` in `gama.py`, decoded by
   `PublicTransport.gaml` via `trip_calendar_map` and `BITWISE_BIT_VAL`). The
   recipe refuses a wider window, and checks the **actual** span of the
   dates of the merged feed, not only the number of days requested.
4. **Every run has its shape in `routes.shp`.** `PublicTransport.gaml` does
   `route r <- route first_with (each.shape_id = shape_id)` when creating the
   vehicle, then reads `r.route_id`, `r.color` and `r.shape.points`. A `shape_id`
   missing from the layer makes `r` nil: a vehicle without geometry. Runs are
   therefore restricted to the layer's `shape_id`s, and the number of points of
   each shape is **cross-checked** between the layer and the feed — the
   `shape_segments` are indices into `r.shape.points`.
5. **No line type drawn without runs.** The final check is the one the
   safeguard in `PublicTransport.gaml` performs at load time: the `route_type`s of
   `routes.shp` must all carry runs. The recipe exits with an error if this
   is not the case, so the defect shows up here and not five months later.
5. **An itinerary can DESIGNATE these runs.** Next to `trip_info.json`,
   the recipe publishes the table `route_id → {shape_id → {stop_id: stop_sequence}}`
   it actually used: it is the structure consumed by
   `GTFSData.get_shape_id_from_route_info`, on which an agent's boarding of a
   vehicle depends (`Inhabitant.gaml`, `shape_id_list contains each.shape_id`).
   It is taken AS IS from the object that produced the runs — same
   fabricated identifiers, same discarded runs — so that the runtime has
   no rule to reapply. See `services/llm-agents/inputs/gtfs/table_traces.py`.
   Without it, measured on 2026-09-04: 80 of the file's 199 lines (17 TER,
   58 liO coaches, 5 Tisséo circular lines) and 2,277 runs ran without
   any itinerary being able to name them.

THE TWO TRICKY POINTS
---------------------
**`service_id` is prefixed by transit system, and it alone.** The TER and liO annual
feeds number their services `SVC_0001`…: **224 identifiers collide**
between the two. Merged as is, liO coaches would read the trains'
calendar. The prefix is harmless because `service_id` is a join key
with nothing: GAMA reads it only in `trip_calendar_map`, `routes.shp`
carries it without the model reading it, and OTP does not expose it. `shape_id`,
`route_id`, `trip_id` and `stop_id`, on the other hand, **are never renamed**: they are
the join keys with the itineraries returned by OTP (measured: no
collision between the three transit systems).

**TER publishes no geometry.** Its `shapes.txt` has only a header and its
`trips.shape_id` are empty. Shapes are rebuilt by
[`gtfs_traces.py`](gtfs_traces.py) — **one per distinct stop sequence**, like
a GTFS that publishes its shapes — and this module is shared with
`export_gtfs_layers.py` so that both files carry the same
identifiers.

EXIT CODES
----------
    0  file written, all checks held
    1  missing resource (feed, layer, Settings.gaml)
    2  invariant contradicted — the file is NOT written
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.gama import gtfs_traces  # noqa: E402
from scripts.data.gama.export_gtfs_layers import FEEDS_DEFAUT, _a_des_geometries  # noqa: E402

INCLUDES = REPO_ROOT / "services" / "GAMA" / "CityTransport" / "includes"
SETTINGS_GAML = REPO_ROOT / "services" / "GAMA" / "CityTransport" / "models" / "Settings.gaml"

# Maximum length of the calendar window, in days. This was not a choice but a
# CONSTRAINT: GAMA's calendar was a bit mask carried by an integer, and GAMA's integer
# is 32 bits wide — the real limit was therefore 31 dates, not 64, and a sixty-day
# run died on day 32 with "Division by zero" (ticket 075, 2026-09-15). The mask is
# now a string of "0"/"1": it no longer has a maximum width, and this cap is now only
# a common-sense safeguard on the size of the exported feed.
LIMITE_MASQUE = 366

CODE_RESSOURCE = 1
CODE_REFUS = 2

# Tables required by the reader `services/llm-agents/inputs/gtfs/reader.py`.
COLONNES_MINIMALES = {
    "routes.txt": ["route_id", "route_short_name", "route_long_name", "route_type"],
    "trips.txt": ["route_id", "service_id", "trip_id", "direction_id", "shape_id"],
    "stop_times.txt": ["trip_id", "stop_sequence", "stop_id", "arrival_time",
                       "departure_time", "shape_dist_traveled"],
    "stops.txt": ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"],
    "shapes.txt": ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence",
                   "shape_dist_traveled"],
    "calendar_dates.txt": ["service_id", "date", "exception_type"],
}


# ── Reading ───────────────────────────────────────────────────────────────────

def _court(chemin: Path) -> str:
    """Path relative to the repository when possible, absolute otherwise (tests outside it)."""
    try:
        return str(chemin.relative_to(REPO_ROOT))
    except ValueError:
        return str(chemin)


def _lire_csv(chemin: Path) -> list[dict]:
    with open(chemin, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def date_simulee_de_settings(chemin: Path = SETTINGS_GAML) -> date | None:
    """`starting_date <- date([2026,3,16,5,0,0]);` → 2026-03-16.

    The date is READ where the model declares it, not copied into the recipe:
    two sources for the same date end up diverging, and the consequence
    of a divergence is an empty transit system with no error message.
    """
    if not chemin.exists():
        return None
    motif = re.compile(
        r"^\s*date\s+starting_date\s*<-\s*date\(\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
        re.MULTILINE,
    )
    trouve = motif.search(chemin.read_text(encoding="utf-8"))
    if not trouve:
        return None
    a, m, j = (int(g) for g in trouve.groups())
    return date(a, m, j)


def types_de_la_couche(chemin_routes: Path, journal=print) -> tuple[dict[str, int], dict[str, float]]:
    """`routes.shp` → (`shape_id` → number of points, `shape_id` → `route_type`)."""
    import geopandas as gpd

    couche = gpd.read_file(chemin_routes)
    points, types = {}, {}
    for _, entite in couche.iterrows():
        shape_id = str(entite["shape_id"])
        geom = entite.geometry
        if geom is None or geom.is_empty:
            journal(f"[ALARME] tracé {shape_id} sans géométrie dans {chemin_routes.name}")
            continue
        points[shape_id] = len(geom.coords)
        types[shape_id] = float(entite["route_type"])
    return points, types


# ── Geometry ──────────────────────────────────────────────────────────────────

def _metres(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance, in metres.

    `build_trips` uses `shape_dist_traveled` only to recover vertex
    INDICES; only strict growth matters. It is nevertheless computed
    in metres so that the intermediate `shapes.txt` stays readable and
    comparable to that of the transit systems that publish one.
    """
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _cumul(points: list[tuple[float, float]]) -> list[float]:
    cumule, total = [0.0], 0.0
    for (lo1, la1), (lo2, la2) in zip(points, points[1:]):
        total += _metres(lo1, la1, lo2, la2)
        cumule.append(round(total, 1))
    return cumule


# ── The body of the recipe ────────────────────────────────────────────────────

def preparer_reseau(
    reseau: str,
    feed: Path,
    fenetre: set[str],
    shapes_de_la_couche: dict[str, int],
    journal=print,
) -> tuple[dict[str, list[dict]], dict]:
    """Extracts from a feed the part active over the window, ready to be merged."""
    debut = time.monotonic()
    mesures: dict = {"feed": str(feed)}

    calendrier = [l for l in _lire_csv(feed / "calendar_dates.txt") if l["date"] in fenetre]
    exceptions = {l.get("exception_type", "1") for l in calendrier}
    if exceptions - {"1"}:
        journal(f"[ALARME] {reseau} : exception_type {sorted(exceptions - {'1'})} dans "
                f"calendar_dates.txt — le lecteur GAMA n'accepte que 1 (dates explicites)")
        return {}, {**mesures, "refus": "exception_type"}
    services = {l["service_id"] for l in calendrier}

    trips = [l for l in _lire_csv(feed / "trips.txt") if l["service_id"] in services]
    horaires_tous = _lire_csv(feed / "stop_times.txt")

    # ── The shape of each run ────────────────────────────────────────────────
    if _a_des_geometries(feed):
        origine = "gtfs"
        for l in trips:
            l["shape_id"] = (l.get("shape_id") or "").strip()
        sans_trace = [l["trip_id"] for l in trips if not l["shape_id"]]
        if sans_trace:
            journal(f"[ALARME] {reseau} : {len(sans_trace)} course(s) sans shape_id dans un feed "
                    f"qui publie des géométries — aucun véhicule ne roulera pour elles")
        points_par_shape = None
    else:
        origine = "arrets"
        journal(f"    {reseau} : aucune géométrie publiée — tracés reconstruits depuis les arrêts")
        suites = gtfs_traces.suites_depuis_stop_times(horaires_tous)
        traces, course_vers_trace, m_traces = gtfs_traces.traces_par_suite_d_arrets(
            trips, suites, journal=journal)
        mesures["traces_reconstruites"] = m_traces
        coords = {l["stop_id"]: (float(l["stop_lon"]), float(l["stop_lat"]))
                  for l in _lire_csv(feed / "stops.txt")}
        points_par_shape = {}
        for shape_id, suite in traces.items():
            pts = [coords[s] for s in suite if s in coords]
            if len(pts) < 2:
                journal(f"[ALARME] {reseau} : tracé {shape_id} sans coordonnées "
                        f"({len(pts)} point(s) sur {len(suite)} arrêts) — écarté")
                continue
            points_par_shape[shape_id] = (suite, pts, _cumul(pts))
        for l in trips:
            shape_id = course_vers_trace.get(l["trip_id"], "")
            l["shape_id"] = shape_id if shape_id in points_par_shape else ""

    # ── Restriction to the shapes carried by the layer ────────────────────────
    avant = len(trips)
    retenus = [l for l in trips if l["shape_id"] in shapes_de_la_couche]
    hors_couche = sorted({l["shape_id"] for l in trips if l["shape_id"] not in shapes_de_la_couche})
    journal(f"    {reseau} : {len(retenus)}/{avant} course(s) dont le tracé est dans routes.shp "
            f"({len(hors_couche)} tracé(s) hors couche — hors du périmètre, ou sans géométrie)")
    trips = retenus
    trips_retenus = {l["trip_id"] for l in trips}
    shapes_retenues = {l["shape_id"] for l in trips}
    routes_retenues = {l["route_id"] for l in trips}

    # ── Stop times ───────────────────────────────────────────────────────────
    horaires = [l for l in horaires_tous if l["trip_id"] in trips_retenus]
    shape_par_trip = {l["trip_id"]: l["shape_id"] for l in trips}
    if origine == "arrets":
        # The shape's vertices ARE the stops: the distance at the vertex is authoritative.
        index_cumul = {sid: {s: c for s, c in zip(suite, cumule)}
                       for sid, (suite, _pts, cumule) in points_par_shape.items()}
        manquants = 0
        for l in horaires:
            table = index_cumul[shape_par_trip[l["trip_id"]]]
            valeur = table.get(l["stop_id"])
            if valeur is None:
                manquants += 1
                continue
            l["shape_dist_traveled"] = f"{valeur}"
        if manquants:
            journal(f"[ALARME] {reseau} : {manquants} horaire(s) dont l'arrêt n'est pas un sommet "
                    f"de son tracé — la course ne peut pas être placée")
            return {}, {**mesures, "refus": "arret_hors_trace"}
    vides = sum(1 for l in horaires if not str(l.get("shape_dist_traveled") or "").strip())
    if vides:
        journal(f"[ALARME] {reseau} : {vides} horaire(s) sans shape_dist_traveled — "
                f"`build_trips` ne peut pas découper le tracé")
        return {}, {**mesures, "refus": "shape_dist_traveled_absent"}

    # ── Geometries ───────────────────────────────────────────────────────────
    if origine == "arrets":
        shapes = []
        for shape_id in sorted(shapes_retenues):
            _suite, pts, cumule = points_par_shape[shape_id]
            for i, ((lon, lat), dist) in enumerate(zip(pts, cumule)):
                shapes.append({"shape_id": shape_id, "shape_pt_lat": f"{lat}",
                               "shape_pt_lon": f"{lon}", "shape_pt_sequence": str(i),
                               "shape_dist_traveled": f"{dist}"})
    else:
        shapes = [l for l in _lire_csv(feed / "shapes.txt") if l["shape_id"] in shapes_retenues]

    # Cross-check with the layer: the `shape_segments` are indices into
    # `r.shape.points`. A shape shorter in the layer than in the feed makes
    # GAMA run off the list, a longer shape leaves an unreachable stub in it.
    nb_points = {}
    for l in shapes:
        nb_points[l["shape_id"]] = nb_points.get(l["shape_id"], 0) + 1
    desaccords = {sid: (n, shapes_de_la_couche[sid])
                  for sid, n in nb_points.items() if n != shapes_de_la_couche[sid]}
    if desaccords:
        exemples = dict(list(desaccords.items())[:3])
        journal(f"[ALARME] {reseau} : {len(desaccords)} tracé(s) dont le nombre de points diffère "
                f"entre le feed et routes.shp — ex. (feed, couche) {exemples}")
        return {}, {**mesures, "refus": "points_desaccord", "desaccords": len(desaccords)}

    # ── Ancillary tables ─────────────────────────────────────────────────────
    routes = [l for l in _lire_csv(feed / "routes.txt") if l["route_id"] in routes_retenues]
    arrets_retenus = {l["stop_id"] for l in horaires}
    stops = [l for l in _lire_csv(feed / "stops.txt") if l["stop_id"] in arrets_retenus]

    # ── Prefix of the service_id (and of them alone) ─────────────────────────
    services_utiles = {t["service_id"] for t in trips}
    calendrier = [dict(l, service_id=f"{reseau}:{l['service_id']}")
                  for l in calendrier if l["service_id"] in services_utiles]
    for l in trips:
        l["service_id"] = f"{reseau}:{l['service_id']}"

    type_par_route = {l["route_id"]: float(l["route_type"]) for l in routes}
    mesures.update({
        "origine_trace": origine,
        "courses_dans_la_fenetre": avant,
        "courses_retenues": len(trips),
        "traces_retenus": len(shapes_retenues),
        "traces_hors_couche": len(hors_couche),
        "lignes": len(routes),
        "arrets": len(stops),
        "services": len({l["service_id"] for l in calendrier}),
        "duree_s": round(time.monotonic() - debut, 1),
    })
    mesures["courses_par_type"] = _compte_par_type(trips, type_par_route)
    journal(f"    {reseau} : {len(trips):,} course(s), {len(shapes_retenues)} tracé(s), "
            f"{len(routes)} ligne(s), {mesures['services']} service(s) "
            f"en {mesures['duree_s']} s".replace(",", " "))
    return {"routes.txt": routes, "trips.txt": trips, "stop_times.txt": horaires,
            "stops.txt": stops, "shapes.txt": shapes, "calendar_dates.txt": calendrier}, mesures


def _compte_par_type(trips: list[dict], type_par_route: dict[str, float]) -> dict[str, int]:
    compte: dict[str, int] = {}
    for l in trips:
        cle = str(int(type_par_route[l["route_id"]]))
        compte[cle] = compte.get(cle, 0) + 1
    return dict(sorted(compte.items(), key=lambda kv: int(kv[0])))


def ecrire_feed(tables: dict[str, list[dict]], sortie: Path) -> None:
    """Writes the merged feed, columns as a union, `calendar.txt` empty."""
    sortie.mkdir(parents=True, exist_ok=True)
    for nom, lignes in tables.items():
        colonnes = list(COLONNES_MINIMALES.get(nom, []))
        for ligne in lignes:
            for cle in ligne:
                if cle not in colonnes:
                    colonnes.append(cle)
        with open(sortie / nom, "w", encoding="utf-8", newline="") as fh:
            ecrivain = csv.DictWriter(fh, fieldnames=colonnes, extrasaction="ignore")
            ecrivain.writeheader()
            for ligne in lignes:
                ecrivain.writerow({c: ligne.get(c, "") for c in colonnes})
    # The GAMA reader requires an EMPTY calendar.txt (it cannot unfold
    # weekly services): `assert len(data.calendar) == 0` in reader.py.
    (sortie / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n", encoding="utf-8")


def courses_du_jour(donnees: dict, jour: str) -> dict[str, int]:
    """Decodes the bit mask and counts the runs active on that day, by type.

    This is the check that matters: `is_trip_available_today` does exactly this
    computation on the model side. Counting the file's runs without decoding the calendar
    would say "40,000 runs" of a file that operates none.
    """
    calendrier = donnees["calendar"]
    if jour not in calendrier["dates"]:
        return {}
    index_du_jour = calendrier["dates"].index(jour)
    masques = calendrier["data"]
    compte: dict[str, int] = {}
    for trip in donnees["trip_list"]:
        # TEXT mask since 2026-09-15 (ticket 075): one character per date. The old
        # integer mask overflowed GAMA's 32-bit integer beyond the 31st date.
        _masque = masques.get(trip["service_id"], "")
        if len(_masque) > index_du_jour and _masque[index_du_jour] == "1":
            cle = str(int(trip["route_type"]))
            compte[cle] = compte.get(cle, 0) + 1
    return dict(sorted(compte.items(), key=lambda kv: int(kv[0])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feed", action="append", metavar="NOM=CHEMIN",
                        help="network to include (default: tisseo, ter, lio — like the layers)")
    parser.add_argument("--date-simulee", default=None,
                        help="YYYY-MM-DD; by default read from Settings.gaml (starting_date)")
    parser.add_argument("--debut", default=None,
                        help="first day of the window, YYYY-MM-DD (default: the simulated date)")
    parser.add_argument("--jours", type=int, default=LIMITE_MASQUE)
    parser.add_argument("--routes", type=Path, default=INCLUDES / "routes.shp")
    parser.add_argument("--sortie", type=Path, default=INCLUDES / "trip_info.json")
    parser.add_argument("--table-traces", type=Path, default=None,
                        help="shape table the runtime will read "
                             "(default: shape_lookup.json next to --sortie)")
    parser.add_argument("--feed-intermediaire", type=Path, default=None,
                        help="where to keep the merged GTFS (default: temporary, deleted)")
    parser.add_argument("--json", type=Path, default=None, help="writes the measures to this file")
    args = parser.parse_args(argv)

    depart = time.monotonic()

    # ── The simulated date ───────────────────────────────────────────────────
    if args.date_simulee:
        jour_simule = date.fromisoformat(args.date_simulee)
        source_date = "argument --date-simulee"
    else:
        jour_simule = date_simulee_de_settings()
        source_date = f"Settings.gaml ({_court(SETTINGS_GAML)})"
        if jour_simule is None:
            print("[ALARME] starting_date not found in Settings.gaml — "
                  "pass --date-simulee YYYY-MM-DD rather than guessing", file=sys.stderr)
            return CODE_RESSOURCE
    debut = date.fromisoformat(args.debut) if args.debut else jour_simule
    print(f"simulated date: {jour_simule} (read from {source_date})")

    if args.jours > LIMITE_MASQUE:
        print(f"[ALARME] window of {args.jours} days: GAMA's calendar is a {LIMITE_MASQUE}-bit "
              f"bit mask (gama.py, PublicTransport.gaml)", file=sys.stderr)
        return CODE_REFUS
    if args.jours < 1:
        print(f"[ALARME] window of {args.jours} day(s)", file=sys.stderr)
        return CODE_REFUS

    fenetre_dates = [debut + timedelta(days=i) for i in range(args.jours)]
    if jour_simule not in fenetre_dates:
        print(f"[ALARME] the simulated date {jour_simule} is NOT in the window "
              f"{fenetre_dates[0]} → {fenetre_dates[-1]}: off-calendar, GAMA would schedule "
              f"no run and would only say so in a `warn`", file=sys.stderr)
        return CODE_REFUS
    fenetre = {d.strftime("%Y%m%d") for d in fenetre_dates}
    jour_simule_txt = jour_simule.strftime("%Y%m%d")
    print(f"window: {fenetre_dates[0]} → {fenetre_dates[-1]} ({args.jours} days)")

    # ── The feeds ────────────────────────────────────────────────────────────
    demandes = FEEDS_DEFAUT if not args.feed else dict(f.split("=", 1) for f in args.feed)
    feeds: dict[str, Path] = {}
    for reseau, chemin in demandes.items():
        feed = Path(chemin) if Path(chemin).is_absolute() else REPO_ROOT / chemin
        if not (feed / "trips.txt").exists():
            print(f"[ALARME] feed {reseau} not found: {feed}", file=sys.stderr)
            return CODE_RESSOURCE
        feeds[reseau] = feed
    if not args.routes.exists():
        print(f"[ALARME] layer not found: {args.routes} — first run "
              f"scripts/data/gama/export_gtfs_layers.py", file=sys.stderr)
        return CODE_RESSOURCE

    points_couche, types_couche = types_de_la_couche(args.routes)
    types_traces = sorted({int(t) for t in types_couche.values()})
    print(f"réseaux : {', '.join(feeds)}")
    print(f"layer {args.routes.name}: {len(points_couche)} shape(s), "
          f"route_type {types_traces}")

    # ── Transit system by transit system ─────────────────────────────────────
    fusion: dict[str, list[dict]] = {nom: [] for nom in COLONNES_MINIMALES}
    mesures_reseaux: dict[str, dict] = {}
    for reseau, feed in feeds.items():
        tables, mesures = preparer_reseau(reseau, feed, fenetre, points_couche)
        mesures_reseaux[reseau] = mesures
        if not tables:
            print(f"[ALARME] {reseau}: preparation refused ({mesures.get('refus')}) — "
                  f"trip_info.json is not written", file=sys.stderr)
            return CODE_REFUS
        for nom, lignes in tables.items():
            fusion[nom].extend(lignes)

    # ── Checks on the merged feed ────────────────────────────────────────────
    dates_servies = sorted({l["date"] for l in fusion["calendar_dates.txt"]})
    if not dates_servies:
        print("[ALARME] no date served in the window", file=sys.stderr)
        return CODE_REFUS
    premiere = datetime.strptime(dates_servies[0], "%Y%m%d").date()
    derniere = datetime.strptime(dates_servies[-1], "%Y%m%d").date()
    etendue = (derniere - premiere).days + 1
    print(f"merged calendar: {dates_servies[0]} → {dates_servies[-1]} "
          f"({len(dates_servies)} date(s) served, span {etendue} days)")
    if etendue > LIMITE_MASQUE:
        print(f"[ALARME] span of {etendue} days between the first and the last date served: "
              f"`build_calendar_binary_map` builds one bit PER DAY of the interval, not per "
              f"date served — the {LIMITE_MASQUE}-bit mask overflows", file=sys.stderr)
        return CODE_REFUS
    if jour_simule_txt not in dates_servies:
        print(f"[ALARME] la date simulée {jour_simule} n'est servie par aucun réseau de la "
              f"fenêtre — aucune course ne serait planifiée", file=sys.stderr)
        return CODE_REFUS

    for reseau, mesures in mesures_reseaux.items():
        if not mesures.get("courses_retenues"):
            print(f"[ALARME] {reseau} : aucune course retenue — le réseau serait tracé et mort",
                  file=sys.stderr)
            return CODE_REFUS

    # ── The file ─────────────────────────────────────────────────────────────
    temporaire = args.feed_intermediaire
    ephemere = temporaire is None
    if ephemere:
        temporaire = args.sortie.parent / f".feed_fusionne_{datetime.now():%Y%m%d_%H%M%S}"
    ecrire_feed(fusion, temporaire)
    print(f"merged feed written: {temporaire}")

    try:
        sys.path.insert(0, str(REPO_ROOT / "services" / "llm-agents"))
        # Since 2026-09-04, importing `settings` no longer creates a run directory
        # and no longer moves `experiments/current`: that belongs to `claim_run()`,
        # which only the owning process calls. A run in progress is not touched.
        from inputs.gtfs.gama import GamaGTFS  # noqa: E402
        from inputs.gtfs.reader import GTFSData  # noqa: E402
        from inputs.gtfs import table_traces  # noqa: E402

        t0 = time.monotonic()
        print("reading the merged feed …")
        # `table_traces="feed"`: the shape table is RECOMPUTED from the merged
        # feed. The default mode would read the ancillary file — the one this
        # recipe is producing, and which describes the PREVIOUS generation.
        gtfs = GTFSData.from_gtfs_files(str(temporaire), table_traces=GTFSData.SOURCE_FEED)
        print(f"construction des courses ({len(fusion['trips.txt']):,} courses) …"
              .replace(",", " "))
        donnees = GamaGTFS(gtfs).build_data(use_cache=False)
        donnees["trip_list"] = [t.model_dump() for t in donnees["trip_list"]]
        duree_build = round(time.monotonic() - t0, 1)
        print(f"runs built in {duree_build} s")
    finally:
        if ephemere and temporaire.exists():
            shutil.rmtree(temporaire)

    # ── Checks on the produced file ──────────────────────────────────────────
    types_courses = sorted({int(t["route_type"]) for t in donnees["trip_list"]})
    par_type: dict[str, int] = {}
    for t in donnees["trip_list"]:
        cle = str(int(t["route_type"]))
        par_type[cle] = par_type.get(cle, 0) + 1
    par_type = dict(sorted(par_type.items(), key=lambda kv: int(kv[0])))
    du_jour = courses_du_jour(donnees, jour_simule_txt)

    print(f"runs per route_type: {par_type}")
    print(f"runs active on {jour_simule} (bit mask decoded): {du_jour}")

    manquants = [t for t in types_traces if t not in types_courses]
    if manquants:
        print(f"[ALARME] route_type(s) drawn in {args.routes.name} but ABSENT from the runs: "
              f"{manquants} — this is exactly the defect this recipe fixes; "
              f"trip_info.json is NOT written", file=sys.stderr)
        return CODE_REFUS
    muets = [t for t in types_traces if not du_jour.get(str(t))]
    if muets:
        print(f"[ALARME] route_type with NO run active on {jour_simule}: {muets} — "
              f"the line would be visible and dead on the sim day; trip_info.json is NOT written",
              file=sys.stderr)
        return CODE_REFUS

    # ── The shape table the runtime will read ────────────────────────────────
    # It is taken AS IS from the object that has just produced the runs:
    # same `shape_id`s, same discards, same transit systems. No rule is reapplied.
    table, arrets_catalogue = gtfs.table_traces_serialisable()
    couples_courses = {(t["route_id"], t["shape_id"]) for t in donnees["trip_list"]}
    absents = sorted({c for c in couples_courses
                      if c[1] not in table.get(c[0], {})})[:5]
    if absents:
        print(f"[ALARME] {len(absents)} (route_id, shape_id) pair(s) carried by a run "
              f"but missing from the shape table — e.g. {absents}; an itinerary could "
              f"not designate these vehicles. trip_info.json is NOT written",
              file=sys.stderr)
        return CODE_REFUS
    hors_couche = sorted({sid for par_trace in table.values() for sid in par_trace
                          if sid not in points_couche})[:5]
    if hors_couche:
        print(f"[ALARME] the shape table would designate shapes missing from "
              f"{args.routes.name} — e.g. {hors_couche}; `route first_with (each.shape_id = "
              f"shape_id)` would return nil. trip_info.json is NOT written", file=sys.stderr)
        return CODE_REFUS
    print(f"shape table: {len(table)} route_id, "
          f"{sum(len(v) for v in table.values())} shape(s), "
          f"{len(arrets_catalogue)} stop(s) in the catalogue")

    # The freshness witnesses must live in the ancillary file's directory:
    # the runtime looks for them relative to IT, not to an absolute path engraved in
    # the file (the host and the `controller` container do not see the same tree).
    table_sortie = args.table_traces or (args.sortie.parent / table_traces.NOM_FICHIER)
    noms_temoins = [args.routes.name, args.routes.with_suffix(".dbf").name, args.sortie.name]
    for nom in noms_temoins[:-1]:
        if not (table_sortie.parent / nom).exists():
            print(f"[ALARME] freshness witness not found next to {table_sortie}: "
                  f"{nom}. The layer and the runs must live in the SAME directory "
                  f"as the table, otherwise its freshness cannot be verified; "
                  f"trip_info.json is NOT written", file=sys.stderr)
            return CODE_REFUS

    # ── Writing, the old file archived and dated ─────────────────────────────
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    archive = None
    if args.sortie.exists():
        horodatage = datetime.fromtimestamp(args.sortie.stat().st_mtime).strftime("%Y-%m-%d_%H-%M")
        dossier = args.sortie.parent / f"archives_{datetime.now():%Y-%m-%d_%H-%M}"
        dossier.mkdir(parents=True, exist_ok=True)
        archive = dossier / f"{args.sortie.stem}_{horodatage}{args.sortie.suffix}"
        shutil.move(str(args.sortie), str(archive))
        print(f"old file kept: {_court(archive)}")

    with open(args.sortie, "w", encoding="utf-8") as fh:
        json.dump(donnees, fh)
    taille = args.sortie.stat().st_size
    print(f"écrit : {args.sortie} — {taille:,} o ({taille / 1_048_576:.1f} Mo)"
          .replace(",", " "))

    # ── The ancillary file, AFTER the runs: it records their fingerprint ─────
    # Order matters: concordance is proven on the bytes actually written.
    if args.table_traces is None and table_sortie.exists() and archive is not None:
        shutil.move(str(table_sortie), str(archive.parent / table_sortie.name))
        print(f"old table kept: "
              f"{_court(archive.parent / table_sortie.name)}")
    document = table_traces.construire(
        table=table, arrets=arrets_catalogue, dossier_temoins=table_sortie.parent,
        noms_temoins=noms_temoins,
        genere_le=datetime.now().isoformat(timespec="seconds"),
        recette="scripts/data/gama/export_trip_info.py",
        reseaux={nom: {"courses_retenues": m.get("courses_retenues"),
                       "traces_retenus": m.get("traces_retenus"),
                       "lignes": m.get("lignes"),
                       "origine_trace": m.get("origine_trace")}
                 for nom, m in mesures_reseaux.items()},
        comptes_supplementaires={"courses": len(donnees["trip_list"])},
    )
    notes = sorted(document["concordance"]["temoins"])
    attendus = sorted(set(noms_temoins))
    if notes != attendus:
        print(f"[ALARME] freshness witnesses recorded {notes}, expected {attendus} — "
              f"the table would be published without being cross-checkable", file=sys.stderr)
        return CODE_REFUS
    taille_table = table_traces.ecrire(table_sortie, document)
    duree = round(time.monotonic() - depart, 1)
    print(f"écrit : {table_sortie} — {taille_table:,} o "
          f"({taille_table / 1_048_576:.1f} Mo), fraîcheur notée sur {notes} ; "
          f"total {duree} s".replace(",", " "))

    resultat = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "date_simulee": jour_simule.isoformat(),
        "fenetre": {"debut": fenetre_dates[0].isoformat(), "fin": fenetre_dates[-1].isoformat(),
                    "jours_demandes": args.jours, "dates_servies": len(dates_servies),
                    "etendue_jours": etendue},
        "reseaux": mesures_reseaux,
        "couche": {"fichier": str(args.routes), "traces": len(points_couche),
                   "route_types": types_traces},
        "trip_info": {"fichier": str(args.sortie), "octets": taille,
                      "courses": len(donnees["trip_list"]),
                      "courses_par_type": par_type,
                      "courses_actives_le_jour_simule": du_jour,
                      "dates_du_calendrier": len(donnees["calendar"]["dates"]),
                      "services": len(donnees["calendar"]["data"]),
                      "duree_construction_s": duree_build},
        "table_traces": {"fichier": str(table_sortie), "octets": taille_table,
                         "route_id": document["concordance"]["comptes"]["route_id"],
                         "traces": document["concordance"]["comptes"]["traces"],
                         "arrets_catalogue": len(arrets_catalogue),
                         "temoins": document["concordance"]["temoins"]},
        "ancien_fichier": _court(archive) if archive else None,
        "duree_totale_s": duree,
    }
    print(json.dumps(resultat, ensure_ascii=False, indent=1))
    if args.json:
        args.json.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
