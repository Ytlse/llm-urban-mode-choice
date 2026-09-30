"""Exports the GAMA layers of public transport routes and stops (ticket 031, G2).

    services/llm-agents/.venv/bin/python scripts/data/gama/export_gtfs_layers.py

Produces `services/GAMA/CityTransport/includes/routes.shp` and `stops.shp` from **several** GTFS
feeds — Tisséo, TER and liO — where these layers used to carry Tisséo only. The previous layers
are moved aside, timestamped, never deleted.

What GAMA reads, and what fixes the schema (`PublicTransport.gaml`):
`routes.shp` → `color`, `route_type`, `shape_id`, `route_id`; `stops.shp` → `stop_name`,
`stop_id`, `route_type`. The `shape_id` is the join key with the itineraries returned by OTP
(`Inhabitant.gaml`: `shape_id_list contains each.shape_id`): **identifiers are therefore
never prefixed or renamed**, and a collision between two networks raises an `[ALARME]` instead
of being silently arbitrated.

The `includes/` folder is not versioned: this script is the recipe. It does not depend on
`services/llm-agents/settings.py` — importing it from a host script re-points `experiments/current`
and diverts the traces of a running run (ticket 031, open question no. 12).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.gama import gtfs_traces  # noqa: E402

INCLUDES = REPO_ROOT / "services" / "GAMA" / "CityTransport" / "includes"
PERIMETRE = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "couronne_perimetre.geojson"

# The three networks of the scope. Tisséo and the TER in their export in service;
# liO in its annual feed, the only one covering the simulated date (the operator's
# export starts on 1 August 2026).
FEEDS_DEFAUT = {
    "tisseo": "data/gtfs/tisseo_gtfs",
    "ter": "data/gtfs/ter_gtfs",
    "lio": "data/gtfs_year/lio_2026",
}

COLONNES_ROUTES = ["shape_id", "route_id", "service_id", "trip_id", "agency_id", "short_name",
                   "long_name", "color", "text_color", "route_type", "reseau", "trace"]
COLONNES_STOPS = ["stop_id", "stop_name", "location_t", "wheelchair", "route_type", "reseau"]


def _couleur(valeur) -> str:
    """GTFS `route_color` (six hex digits, no hash) into a value readable by GAMA."""
    texte = "" if valeur is None else str(valeur).strip()
    if texte in ("", "nan", "None"):
        return "#000000"
    return texte if texte.startswith("#") else f"#{texte}"


def _a_des_geometries(feed: Path) -> bool:
    """`shapes.txt` present does not mean `shapes.txt` populated.

    The TER feed publishes one that has only its header (73 bytes). Testing only the
    existence of the file sent the TER through the "GTFS geometries" branch,
    which then returned zero paths — without a word.
    """
    chemin = feed / "shapes.txt"
    if not chemin.exists():
        return False
    with open(chemin, encoding="utf-8") as fh:
        fh.readline()  # header
        return bool(fh.readline().strip())


def _lire(feed: Path, nom: str, **kwargs):
    import pandas as pd

    chemin = feed / nom
    if not chemin.exists():
        raise FileNotFoundError(f"{chemin} missing — incomplete feed")
    return pd.read_csv(chemin, dtype=str, **kwargs)


def couches(feeds: dict[str, Path], journal=print):
    """Builds the two GeoDataFrames, network by network."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString

    routes_out, stops_out, comptes = [], [], {}
    for reseau, feed in feeds.items():
        trips = _lire(feed, "trips.txt")
        routes = _lire(feed, "routes.txt")
        stops = _lire(feed, "stops.txt")
        stop_times = _lire(feed, "stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"])

        # ── Routes: one entity per path ───────────────────────────────────────
        # The TER feed publishes no `shapes.txt`. Rather than leaving its
        # routes out of the layer, their path is the polyline of their served
        # stops, marked `trace=arrets` — a reconstruction, not the
        # geometry of the railway track.
        #
        # Since 2026-09-04: **one path per distinct stop sequence**, and no
        # longer one per (route, direction) taken from the most served run. This path
        # is no longer only drawn: it is the geometry along which
        # the `public_vehicle` of `trip_info.json` run, and `build_trips` forces
        # the last segment up to the last point of the path — a Toulouse → Tarbes
        # run placed on the Toulouse → Pau path would travel all the way to
        # Pau. See `gtfs_traces.py`, which is the SINGLE place where these `shape_id`
        # are made, so that the layer and the runs do not diverge.
        if (feed / "shapes.txt").exists() and _a_des_geometries(feed):
            origine_trace = "gtfs"
            shapes = _lire(feed, "shapes.txt")
            shapes["shape_pt_sequence"] = shapes["shape_pt_sequence"].astype(int)
            shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])
            geometries = shapes.groupby("shape_id").apply(
                lambda l: LineString(zip(l["shape_pt_lon"].astype(float), l["shape_pt_lat"].astype(float))),
                include_groups=False,
            )
            trips_traces = trips.copy()
        else:
            origine_trace = "arrets"
            journal(f"    {reseau} : aucun shapes.txt — tracés reconstruits depuis la suite des arrêts")
            suites = gtfs_traces.suites_depuis_stop_times(
                stop_times.to_dict("records"))
            traces_arrets, course_vers_trace, _ = gtfs_traces.traces_par_suite_d_arrets(
                trips.to_dict("records"), suites, journal=journal)
            trips_traces = trips.copy()
            trips_traces["shape_id"] = trips_traces["trip_id"].map(course_vers_trace)
            trips_traces = trips_traces.dropna(subset=["shape_id"])
            coords = stops.set_index("stop_id")[["stop_lon", "stop_lat"]].astype(float)
            lignes = {}
            for shape_id, suite in traces_arrets.items():
                points = [(coords.at[s, "stop_lon"], coords.at[s, "stop_lat"])
                          for s in suite if s in coords.index]
                if len(points) >= 2:
                    lignes[shape_id] = LineString(points)
                else:
                    journal(f"[ALARME] {reseau} : tracé {shape_id} sans coordonnées "
                            f"({len(points)} point(s) sur {len(suite)} arrêts) — écarté")
            geometries = pd.Series(lignes)
            if geometries.empty:
                journal(f"[ALARME] {reseau} : aucun tracé reconstructible depuis les arrêts")
        table = pd.DataFrame({"shape_id": list(geometries.index), "geometry": list(geometries.values)})
        premier_trip = (trips_traces[["route_id", "service_id", "trip_id", "shape_id"]]
                        .groupby("shape_id").agg(lambda x: x.iloc[0]).reset_index())
        table = table.merge(premier_trip, on="shape_id", how="left").merge(routes, on="route_id", how="left")
        table = table.rename(columns={"route_short_name": "short_name", "route_long_name": "long_name",
                                      "route_color": "color", "route_text_color": "text_color"})
        for colonne in COLONNES_ROUTES:
            if colonne not in table.columns:
                table[colonne] = ""
        table["color"] = table["color"].apply(_couleur)
        table["text_color"] = table["text_color"].apply(_couleur)
        table["reseau"] = reseau
        table["trace"] = origine_trace
        routes_out.append(gpd.GeoDataFrame(table[COLONNES_ROUTES + ["geometry"]], crs="EPSG:4326"))

        # ── Stops: those served by a run, with the type of their route ────────
        type_par_trip = trips[["route_id", "trip_id"]].merge(
            routes[["route_id", "route_type"]], on="route_id", how="left")
        type_par_arret = (stop_times.merge(type_par_trip[["trip_id", "route_type"]], on="trip_id", how="left")
                          .dropna(subset=["route_type"]))
        type_par_arret["route_type"] = type_par_arret["route_type"].astype(int)
        type_par_arret = type_par_arret.groupby("stop_id").agg({"route_type": "min"}).reset_index()
        table_arrets = stops.rename(columns={"location_type": "location_t",
                                             "wheelchair_boarding": "wheelchair"})
        for colonne in ("location_t", "wheelchair"):
            if colonne not in table_arrets.columns:
                table_arrets[colonne] = ""
        table_arrets = table_arrets[["stop_id", "stop_name", "location_t", "wheelchair",
                                     "stop_lon", "stop_lat"]].merge(type_par_arret, on="stop_id", how="inner")
        table_arrets["reseau"] = reseau
        stops_out.append(gpd.GeoDataFrame(
            table_arrets[COLONNES_STOPS],
            geometry=gpd.points_from_xy(table_arrets["stop_lon"].astype(float),
                                        table_arrets["stop_lat"].astype(float), z=0),
            crs="EPSG:4326"))
        comptes[reseau] = {"lignes_gtfs": int(len(routes)), "traces": int(len(table)),
                           "origine_trace": origine_trace,
                           "arrets_desservis": int(len(table_arrets)), "feed": str(feed)}
        journal(f"    {reseau} : {len(routes)} ligne(s), {len(table)} tracé(s), "
                f"{len(table_arrets)} arrêt(s) desservi(s)")

    couche_routes = pd.concat(routes_out, ignore_index=True)
    couche_stops = pd.concat(stops_out, ignore_index=True)

    # Identifiers are join keys with OTP: a collision between
    # networks would put an agent aboard another network's vehicle.
    for nom, couche, cle in (("shape_id", couche_routes, "shape_id"), ("stop_id", couche_stops, "stop_id")):
        doublons = couche[couche.duplicated(cle, keep=False)]
        if len(doublons):
            reseaux = sorted(set(doublons["reseau"]))
            journal(f"[ALARME] {len(doublons)} {nom} partagés entre réseaux {reseaux} — "
                    f"la jointure GAMA/OTP est ambiguë sur ces entités")
            comptes.setdefault("collisions", {})[nom] = int(len(doublons))
    return couche_routes, couche_stops, comptes


def restreindre(couche_routes, couche_stops, journal=print):
    """Keeps only what touches the scope of the 453 municipalities.

    liO covers the whole of Occitanie: 2,634 paths, from Perpignan to Millau. Pouring
    them as they are into a GAMA world of 86 × 93 km would bring in
    geometries ten times wider than it. The kept paths are not
    clipped — a route that leaves the scope does so whole, otherwise its
    vehicle would jump from one end of the world to the other.
    """
    import geopandas as gpd

    polygone = gpd.read_file(PERIMETRE).union_all()
    routes_gardees = couche_routes[couche_routes.intersects(polygone)].reset_index(drop=True)
    stops_gardes = couche_stops[couche_stops.within(polygone)].reset_index(drop=True)
    journal(f"    périmètre : {len(routes_gardees)} / {len(couche_routes)} tracé(s) et "
            f"{len(stops_gardes)} / {len(couche_stops)} arrêt(s) touchent les 453 communes")
    return routes_gardees, stops_gardes


def couverture(couche_routes, couche_stops, journal=print) -> dict:
    """What the public transport network really serves of the GAMA world.

    Three measures, because the first one is not enough:
      * the envelope — the test `Settings.gaml` makes at load time. It becomes
        true as soon as a regional network enters the layer, and then says nothing
        more about service coverage: an envelope covering the world does not put a
        single stop in it.
      * the 5 km cells of the world that carry at least one stop;
      * the municipalities of the scope that carry at least one stop — the figure that
        says how many inhabitants can see public transport.
    """
    import geopandas as gpd
    from shapely.geometry import box

    perimetre = gpd.read_file(PERIMETRE)
    monde = box(*perimetre.total_bounds)
    lignes = box(*couche_routes.total_bounds)
    part_enveloppe = lignes.intersection(monde).area / monde.area

    # 5 km cells, in degrees at the latitude of the scope (1° lat = 111.2 km,
    # 1° lon = 111.32 × cos(43.5°) = 80.7 km). Only cells whose centre falls
    # inside the scope count in the denominator: the corners of the
    # bounding rectangle are not study territory.
    from shapely import contains_xy, prepare

    polygone = perimetre.union_all()
    prepare(polygone)
    lon_min, lat_min, lon_max, lat_max = perimetre.total_bounds
    pas_lat, pas_lon = 5.0 / 111.2, 5.0 / 80.7
    n_lat = max(1, int((lat_max - lat_min) / pas_lat) + 1)
    n_lon = max(1, int((lon_max - lon_min) / pas_lon) + 1)
    dans_le_perimetre = {
        (i, j) for i in range(n_lat) for j in range(n_lon)
        if contains_xy(polygone, lon_min + (j + 0.5) * pas_lon, lat_min + (i + 0.5) * pas_lat)
    }
    mailles = {(int((y - lat_min) / pas_lat), int((x - lon_min) / pas_lon))
               for x, y in zip(couche_stops.geometry.x, couche_stops.geometry.y)
               if lon_min <= x <= lon_max and lat_min <= y <= lat_max} & dans_le_perimetre
    part_mailles = len(mailles) / max(1, len(dans_le_perimetre))

    mesure = {"monde_bounds": [round(v, 4) for v in perimetre.total_bounds],
              "lignes_bounds": [round(v, 4) for v in couche_routes.total_bounds],
              "part_enveloppe": round(part_enveloppe, 4),
              "mailles_5km": {"dans_le_perimetre": len(dans_le_perimetre), "avec_arret": len(mailles),
                              "part": round(part_mailles, 4)}}

    zones = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_zones.gpkg"
    if zones.exists():
        zf = gpd.read_file(zones).to_crs("EPSG:4326")
        avec = gpd.sjoin(zf[["geometry"]], couche_stops[["geometry"]], predicate="contains", how="inner")
        servies = int(avec.index.nunique())
        mesure["zones_fines"] = {"total": int(len(zf)), "avec_arret": servies,
                                 "part": round(servies / max(1, len(zf)), 4)}
        journal(f"    couverture : {servies} / {len(zf)} zones fines de l'enquête portent au moins un arrêt")
    journal(f"    couverture : enveloppe {part_enveloppe:.0%} du monde ; "
            f"{len(mailles)} / {len(dans_le_perimetre)} mailles de 5 km du périmètre portent un arrêt "
            f"({part_mailles:.0%})")
    return mesure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feed", action="append", metavar="NAME=PATH",
                        help="network to include (default: tisseo, ter, lio)")
    parser.add_argument("--sortie", type=Path, default=INCLUDES)
    parser.add_argument("--tout", action="store_true",
                        help="do not restrict to the scope of the 453 municipalities")
    parser.add_argument("--json", type=Path, default=None, help="writes the measures into this file")
    args = parser.parse_args(argv)

    demandes = FEEDS_DEFAUT if not args.feed else dict(f.split("=", 1) for f in args.feed)
    feeds = {}
    for reseau, chemin in demandes.items():
        feed = Path(chemin) if Path(chemin).is_absolute() else REPO_ROOT / chemin
        if not (feed / "trips.txt").exists():
            print(f"[ALARME] feed {reseau} introuvable : {feed}", file=sys.stderr)
            return 1
        feeds[reseau] = feed

    print(f"réseaux : {', '.join(feeds)}")
    couche_routes, couche_stops, comptes = couches(feeds)
    comptes["avant_restriction"] = {"traces": int(len(couche_routes)), "arrets": int(len(couche_stops))}
    if not args.tout:
        couche_routes, couche_stops = restreindre(couche_routes, couche_stops)
    mesure = couverture(couche_routes, couche_stops)

    args.sortie.mkdir(parents=True, exist_ok=True)
    horodatage = datetime.now().strftime("%Y-%m-%d_%H-%M")
    archive = args.sortie / f"archives_{horodatage}"
    deplaces = []
    for base in ("routes", "stops"):
        for existant in sorted(args.sortie.glob(f"{base}.*")):
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(existant), str(archive / existant.name))
            deplaces.append(existant.name)
    if deplaces:
        print(f"    previous layers kept in {archive.name}: {len(deplaces)} file(s)")

    couche_routes.to_file(args.sortie / "routes.shp")
    couche_stops.to_file(args.sortie / "stops.shp")

    resultat = {"date": datetime.now().isoformat(timespec="seconds"), "reseaux": comptes,
                "routes_shp": {"entites": int(len(couche_routes))},
                "stops_shp": {"entites": int(len(couche_stops))},
                "couverture": mesure,
                "anciennes_couches": {"dossier": archive.name if deplaces else None,
                                      "fichiers": deplaces}}
    print(json.dumps(resultat, ensure_ascii=False, indent=1))
    if args.json:
        args.json.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
