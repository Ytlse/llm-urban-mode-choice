"""
Assembly of the annual feed: from the chosen days to the GTFS set written to disk.

The atomic unit is not the trip but **the service of one day as a single
export describes it**. A day is therefore never recomposed from
several exports, which rules out by construction the case "the donor day
references a stop or a geometry that is missing".

Three rules govern the identity of objects:

  * The operator's `trip_id` is not stable over the year — the Jaccard
    index between the trips of a Tuesday in March and those of a Tuesday in September
    is 0.00, the two exports using disjoint namespaces.
    The identity used is therefore that of the CONTENT: route, direction, headsign,
    geometry and sequence of timed stops.

  * `(trip, stop times, geometry)` is inseparable. The `shape_dist_traveled`
    of the stop times is calibrated on ITS geometry. Deduplicating geometry
    points on `(shape_id, shape_pt_sequence)` mixes two paths and produces
    a chimera: this is what happened to shape 14846 of the feed in service,
    whose 524 points come from two different exports.

  * Stops, routes and transfers are infrastructure, not
    service: the latest published export is authoritative, and any notable gap is
    flagged rather than silently arbitrated.

The output calendar is rebuilt by SETS OF DATES: trips that
run on exactly the same days share a synthetic `service_id`. This
makes over-supply structurally impossible, keeps `calendar.txt` empty and
`exception_type=1` — the two conditions set by
`services/llm-agents/inputs/gtfs/reader.py` — and compresses the calendar by a factor of ten
compared with one service per trip.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from . import gtfs_io
from .donneurs import EXTRAPOLE, REEL, Provenance
from .gtfs_io import Export
from .offre import IndexExport

COLONNES_TRIPS = [
    "route_id",
    "service_id",
    "trip_id",
    "trip_headsign",
    "direction_id",
    "block_id",
    "shape_id",
    "wheelchair_accessible",
    "bikes_allowed",
]
COLONNES_CALENDAR = ["service_id", "date", "exception_type"]
COLONNES_CALENDAR_HEBDO = [
    "service_id",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "start_date",
    "end_date",
]
COLONNES_FEED_INFO = [
    "feed_id",
    "feed_publisher_name",
    "feed_publisher_url",
    "feed_lang",
    "feed_start_date",
    "feed_end_date",
    "feed_version",
]

# Fields of a stop time that define the service provided. `trip_id` is excluded:
# it is precisely what we try to reconcile across exports.
CHAMPS_HORAIRE_IDENTITE = (
    "stop_sequence",
    "stop_id",
    "arrival_time",
    "departure_time",
    "pickup_type",
    "drop_off_type",
    "stop_headsign",
    "shape_dist_traveled",
)


@dataclass
class Statistiques:
    trips_ecrits: int = 0
    horaires_ecrits: int = 0
    services: int = 0
    lignes_calendrier: int = 0
    trips_fusionnes: int = 0
    trips_forkes: int = 0
    collisions_meme_jour: int = 0
    doublons_de_contenu: int = 0
    shapes_dupliquees: int = 0
    arrets_deplaces: list[tuple[str, float]] = field(default_factory=list)
    lignes_redefinies: int = 0
    orphelins: dict[str, int] = field(default_factory=dict)


def _hacher(*morceaux: str) -> str:
    digest = hashlib.sha256()
    for morceau in morceaux:
        digest.update(morceau.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _cle_horaires(horaires: list[dict[str, str]]) -> str:
    return _hacher(
        *[
            "|".join(h.get(champ, "") for champ in CHAMPS_HORAIRE_IDENTITE)
            for h in horaires
        ]
    )


def cle_contenu(meta: dict[str, str], horaires: list[dict[str, str]], hash_shape: str) -> str:
    """Identity of a trip by what it does, independently of its `trip_id`.

    It is the building block of the service fingerprint used by validation: two
    feeds serve the same day if and only if their multisets of
    content keys coincide.
    """
    return _hacher(
        meta.get("route_id", ""),
        meta.get("direction_id", ""),
        meta.get("trip_headsign", ""),
        hash_shape,
        _cle_horaires(horaires),
    )


def hash_geometrie(points: list[dict[str, str]]) -> str:
    """Fingerprint of a path, over its canonicalised and ordered points."""
    return _hacher(
        *[
            f"{p['shape_pt_lat']}|{p['shape_pt_lon']}|{p.get('shape_dist_traveled', '')}"
            for p in points
        ]
    )


def construire(
    sortie: Path,
    plan: dict[str, Provenance],
    index_par_export: dict[str, IndexExport],
    config: dict,
    identite_feed: dict[str, str],
    journal=print,
) -> Statistiques:
    """Writes the annual feed into `sortie` and returns its counters."""
    stats = Statistiques()
    sortie.mkdir(parents=True, exist_ok=True)
    dec_coord = int(config["canonicalisation"]["decimales_coordonnees"])
    dec_dist = int(config["canonicalisation"]["decimales_distance"])
    deplacement_max = float(config["controles"]["deplacement_arret_max_m"])

    # Days to load, grouped by export: (label → source dates).
    journees: dict[str, set[str]] = {}
    for provenance in plan.values():
        if provenance.mode in (REEL, EXTRAPOLE) and provenance.export:
            journees.setdefault(provenance.export, set()).add(provenance.date_source)

    # Processing order: the oldest export first, so that the `trip_id`
    # kept is the most stable in the repository history.
    ordre = sorted(journees, key=lambda e: (index_par_export[e].export.date_min, e))

    contenu_vers_trip: dict[str, str] = {}
    trip_pris: dict[str, str] = {}
    journee_canon: dict[tuple[str, str], list[str]] = {}
    shapes_vues: dict[str, dict[str, str]] = {}  # shape_id → {hash: shape_id_final}
    trips_sortie: dict[str, dict[str, str]] = {}
    stops_references: set[str] = set()
    routes_referencees: set[str] = set()

    colonnes_horaires = ["trip_id", *CHAMPS_HORAIRE_IDENTITE, "timepoint"]
    colonnes_shapes = [
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
        "shape_dist_traveled",
    ]
    ecrivain_horaires = gtfs_io.EcrivainCSV(sortie / "stop_times.txt", colonnes_horaires)
    ecrivain_shapes = gtfs_io.EcrivainCSV(sortie / "shapes.txt", colonnes_shapes)

    for etiquette in ordre:
        index = index_par_export[etiquette]
        export = index.export
        dates_sources = journees[etiquette]
        trips_voulus: set[str] = set()
        for date in dates_sources:
            trips_voulus.update(index.trips_par_date.get(date, ()))
        journal(
            f"    {etiquette} : {len(dates_sources)} journée(s) retenue(s), "
            f"{len(trips_voulus):,} trip(s) à charger"
        )

        # ── Required geometries ───────────────────────────────────────────────
        shapes_voulues = {
            index.trips[t].get("shape_id", "") for t in trips_voulus
        } - {""}
        points_par_shape: dict[str, list[dict[str, str]]] = {}
        for ligne in gtfs_io.lire(export, "shapes.txt"):
            if ligne["shape_id"] in shapes_voulues:
                points_par_shape.setdefault(ligne["shape_id"], []).append(
                    gtfs_io.canoniser_point_shape(ligne, dec_coord, dec_dist)
                )
        for points in points_par_shape.values():
            points.sort(key=lambda p: int(p["shape_pt_sequence"]))

        hash_par_shape = {
            shape_id: hash_geometrie(points)
            for shape_id, points in points_par_shape.items()
        }

        # ── Required stop times ───────────────────────────────────────────────
        horaires_par_trip: dict[str, list[dict[str, str]]] = {}
        for ligne in gtfs_io.lire(export, "stop_times.txt"):
            trip_id = ligne["trip_id"]
            if trip_id in trips_voulus:
                horaires_par_trip.setdefault(trip_id, []).append(
                    gtfs_io.canoniser_horaire(ligne, dec_dist)
                )
        for horaires in horaires_par_trip.values():
            horaires.sort(key=lambda h: int(h["stop_sequence"]))

        # ── Occurrence rank: two identical runs on the same day ───────────────
        # Two runs with identical content in the same export are the same
        # run when they operate on disjoint days (the operator splits
        # its calendar into periods) — merging them is precisely what
        # compresses the feed. They are two distinct runs when they
        # operate on the SAME day: liO publishes 45 on Monday 14/09/2026, two
        # mission numbers for the same timetable. Confusing them would cut
        # the service of the day, and V2 would reject it.
        # Hence a rank per "slot": each run takes the first slot
        # whose days do not overlap its own. The number of slots is
        # thus the maximum number of identical runs on the same day.
        dates_par_trip: dict[str, set[str]] = {}
        for date in dates_sources:
            for trip_id in index.trips_par_date.get(date, ()):
                if trip_id in trips_voulus:
                    dates_par_trip.setdefault(trip_id, set()).add(date)
        cle_brute_par_trip: dict[str, str] = {}
        for trip_id in sorted(trips_voulus):
            horaires = horaires_par_trip.get(trip_id)
            if not horaires:
                continue
            meta = index.trips[trip_id]
            cle_brute_par_trip[trip_id] = cle_contenu(
                meta, horaires, hash_par_shape.get(meta.get("shape_id", ""), "")
            )
        places_par_cle: dict[str, list[set[str]]] = {}
        rang_par_trip: dict[str, int] = {}
        for trip_id in sorted(cle_brute_par_trip):
            cle_brute = cle_brute_par_trip[trip_id]
            jours = dates_par_trip.get(trip_id, set())
            places = places_par_cle.setdefault(cle_brute, [])
            rang = next((i for i, prises in enumerate(places) if not (prises & jours)), len(places))
            if rang == len(places):
                places.append(set())
            places[rang] |= jours
            rang_par_trip[trip_id] = rang
            if rang:
                stats.doublons_de_contenu += 1

        # ── Identity by content ───────────────────────────────────────────────
        trip_local_vers_canon: dict[str, str] = {}
        for trip_id in sorted(trips_voulus):
            meta = index.trips[trip_id]
            horaires = horaires_par_trip.get(trip_id)
            if not horaires:
                journal(f"[ALARME] {etiquette} : trip {trip_id} sans horaire, écarté")
                continue
            shape_id = meta.get("shape_id", "")
            hash_shape = hash_par_shape.get(shape_id, "")

            # The slot rank enters the key: two exports describing the
            # same run always share it, slot by slot.
            cle_brute = cle_brute_par_trip[trip_id]
            rang = rang_par_trip[trip_id]
            cle = cle_brute if rang == 0 else f"{cle_brute}#{rang + 1}"
            deja = contenu_vers_trip.get(cle)
            if deja is not None:
                trip_local_vers_canon[trip_id] = deja
                stats.trips_fusionnes += 1
                continue

            # The content is new. We keep the operator's trip_id, unless
            # it already designates another content — a real fork, which must stay
            # visible rather than be arbitrated.
            trip_final = trip_id
            if trip_id in trip_pris:
                trip_final = f"{trip_id}__{etiquette}"
                stats.trips_forkes += 1
                journal(
                    f"    fork : trip {trip_id} a un contenu différent dans {etiquette}, "
                    f"conservé sous {trip_final}"
                )
            trip_pris[trip_final] = cle
            contenu_vers_trip[cle] = trip_final
            trip_local_vers_canon[trip_id] = trip_final

            # Geometry: first variant under its original identifier, the
            # following ones duplicated. Never a point-by-point merge.
            shape_final = ""
            if shape_id:
                variantes = shapes_vues.setdefault(shape_id, {})
                shape_final = variantes.get(hash_shape, "")
                if not shape_final:
                    shape_final = shape_id if not variantes else f"{shape_id}__{etiquette}"
                    if variantes:
                        stats.shapes_dupliquees += 1
                        journal(
                            f"    géométrie : shape {shape_id} diverge dans {etiquette} "
                            f"({len(points_par_shape.get(shape_id, []))} points), "
                            f"conservée sous {shape_final}"
                        )
                    variantes[hash_shape] = shape_final
                    for point in points_par_shape.get(shape_id, ()):
                        ecrivain_shapes.ecrire({**point, "shape_id": shape_final})

            sortie_trip = {c: meta.get(c, "") for c in COLONNES_TRIPS}
            sortie_trip["trip_id"] = trip_final
            sortie_trip["shape_id"] = shape_final
            sortie_trip["service_id"] = ""  # assigned at the end
            trips_sortie[trip_final] = sortie_trip
            routes_referencees.add(meta.get("route_id", ""))

            for horaire in horaires:
                ecrivain_horaires.ecrire({**horaire, "trip_id": trip_final})
                stops_references.add(horaire["stop_id"])
            stats.horaires_ecrits += len(horaires)

        for date in dates_sources:
            mappes = [
                trip_local_vers_canon[t]
                for t in index.trips_par_date.get(date, ())
                if t in trip_local_vers_canon
            ]
            distincts = sorted(set(mappes))
            # Safety net: since the occurrence rank distinguishes runs with
            # identical content in the same export, two local runs can
            # no longer land on the same canonical run. If that
            # happened, the service of the day would be cut and V2 would
            # block it — better to name it here.
            if len(distincts) != len(mappes):
                stats.collisions_meme_jour += len(mappes) - len(distincts)
                journal(
                    f"[ALARME] {etiquette} {date} : {len(mappes) - len(distincts)} course(s) de "
                    f"contenu identique le même jour — l'offre de cette journée serait amputée"
                )
            journee_canon[(etiquette, date)] = distincts

    ecrivain_horaires.fermer()
    ecrivain_shapes.fermer()
    stats.trips_ecrits = len(trips_sortie)

    # ── Calendar by sets of dates ────────────────────────────────────────────
    dates_par_trip: dict[str, list[str]] = {}
    for date in sorted(plan):
        provenance = plan[date]
        if provenance.mode not in (REEL, EXTRAPOLE):
            continue
        for trip_final in journee_canon.get((provenance.export, provenance.date_source), ()):
            dates_par_trip.setdefault(trip_final, []).append(date)

    ensembles: dict[tuple[str, ...], list[str]] = {}
    for trip_final, dates in dates_par_trip.items():
        ensembles.setdefault(tuple(dates), []).append(trip_final)

    # Services numbered by decreasing cardinality: SVC_0001 is the most
    # frequent service, which makes the calendar readable to the naked eye.
    ordre_services = sorted(ensembles.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    calendrier: list[dict[str, str]] = []
    for rang, (dates, trips) in enumerate(ordre_services, start=1):
        service_id = f"SVC_{rang:04d}"
        for trip_final in trips:
            trips_sortie[trip_final]["service_id"] = service_id
        for date in dates:
            calendrier.append(
                {"service_id": service_id, "date": date, "exception_type": "1"}
            )
    stats.services = len(ordre_services)
    stats.lignes_calendrier = len(calendrier)

    sans_service = [t for t, l in trips_sortie.items() if not l["service_id"]]
    if sans_service:
        journal(
            f"[ALARME] {len(sans_service)} trip(s) écrit(s) sans aucune date active — "
            f"ils sont retirés de trips.txt, mais leurs horaires et géométries sont "
            f"déjà écrits : la fermeture référentielle les signalera en orphelins"
        )
        for trip_final in sans_service:
            del trips_sortie[trip_final]
        stats.trips_ecrits = len(trips_sortie)

    gtfs_io.ecrire_table(
        sortie / "trips.txt",
        COLONNES_TRIPS,
        trips_sortie.values(),
        tri=lambda l: (l["route_id"], l["service_id"], l["trip_id"]),
    )
    gtfs_io.ecrire_table(
        sortie / "calendar_dates.txt",
        COLONNES_CALENDAR,
        calendrier,
        tri=lambda l: (l["service_id"], l["date"]),
    )
    # empty calendar.txt: the repository reader requires it, and the whole calendar is
    # already carried by calendar_dates.txt.
    gtfs_io.ecrire_table(sortie / "calendar.txt", COLONNES_CALENDAR_HEBDO, [])

    # ── Infrastructure: the latest published export is authoritative ─────────
    _ecrire_infrastructure(
        sortie,
        ordre,
        index_par_export,
        stops_references,
        routes_referencees,
        dec_coord,
        deplacement_max,
        stats,
        journal,
    )

    _ecrire_feed_info(sortie, plan, identite_feed)

    stats.orphelins = _verifier_fermeture(sortie, journal)
    return stats


def _ecrire_infrastructure(
    sortie: Path,
    ordre: list[str],
    index_par_export: dict[str, IndexExport],
    stops_references: set[str],
    routes_referencees: set[str],
    dec_coord: int,
    deplacement_max: float,
    stats: Statistiques,
    journal,
) -> None:
    """Stops, routes, agencies, transfers: latest published version.

    Stops are never suffixed: duplicating a `stop_id` would create two
    distinct platforms in OTP and degrade transfers. A stop that
    moves notably is flagged, not silently arbitrated.
    """
    arrets: dict[str, dict[str, str]] = {}
    lignes: dict[str, dict[str, str]] = {}
    agences: dict[str, dict[str, str]] = {}
    correspondances: dict[tuple[str, str], dict[str, str]] = {}
    colonnes = {"stops": [], "routes": [], "agency": [], "transfers": []}

    for etiquette in ordre:  # from oldest to newest: the last one overwrites
        export = index_par_export[etiquette].export
        for nom, cible in (("stops.txt", "stops"), ("routes.txt", "routes"),
                           ("agency.txt", "agency"), ("transfers.txt", "transfers")):
            for entete in gtfs_io.entetes(export, nom):
                if entete not in colonnes[cible]:
                    colonnes[cible].append(entete)

        for ligne in gtfs_io.lire(export, "stops.txt"):
            canonique = gtfs_io.canoniser_arret(ligne, dec_coord)
            stop_id = canonique["stop_id"]
            precedent = arrets.get(stop_id)
            if precedent is not None:
                ecart = gtfs_io.distance_m(
                    precedent.get("stop_lat", ""), precedent.get("stop_lon", ""),
                    canonique.get("stop_lat", ""), canonique.get("stop_lon", ""),
                )
                if ecart > deplacement_max:
                    stats.arrets_deplaces.append((stop_id, ecart))
            arrets[stop_id] = canonique

        for ligne in gtfs_io.lire(export, "routes.txt"):
            route_id = ligne["route_id"]
            if route_id in lignes and lignes[route_id] != ligne:
                stats.lignes_redefinies += 1
            lignes[route_id] = ligne

        for ligne in gtfs_io.lire(export, "agency.txt"):
            agences[ligne.get("agency_id", "")] = ligne

        for ligne in gtfs_io.lire(export, "transfers.txt"):
            correspondances[(ligne.get("from_stop_id", ""), ligne.get("to_stop_id", ""))] = ligne

    if stats.arrets_deplaces:
        pires = sorted(stats.arrets_deplaces, key=lambda x: -x[1])[:5]
        journal(
            f"[ALARME] {len(stats.arrets_deplaces)} arrêt(s) déplacé(s) de plus de "
            f"{deplacement_max:.0f} m entre exports — les pires : "
            + ", ".join(f"{s} ({d:.0f} m)" for s, d in pires)
        )
    if stats.lignes_redefinies:
        journal(f"    infrastructure : {stats.lignes_redefinies} redéfinition(s) de ligne, dernière version retenue")

    # Closure over parent stations: a kept platform whose station was
    # discarded would leave a dangling reference that OTP reports at every build.
    retenus = set(stops_references)
    for _ in range(4):  # GTFS hierarchies are shallow
        parents = {
            arrets[s].get("parent_station", "")
            for s in retenus
            if s in arrets and arrets[s].get("parent_station")
        } - {""}
        nouveaux = parents - retenus
        if not nouveaux:
            break
        retenus |= nouveaux

    gtfs_io.ecrire_table(
        sortie / "stops.txt",
        colonnes["stops"],
        [a for sid, a in arrets.items() if sid in retenus],
        tri=lambda l: l["stop_id"],
    )
    gtfs_io.ecrire_table(
        sortie / "routes.txt",
        colonnes["routes"],
        [l for rid, l in lignes.items() if rid in routes_referencees],
        tri=lambda l: l["route_id"],
    )
    gtfs_io.ecrire_table(
        sortie / "agency.txt", colonnes["agency"], list(agences.values()),
        tri=lambda l: l.get("agency_id", ""),
    )
    gtfs_io.ecrire_table(
        sortie / "transfers.txt",
        colonnes["transfers"],
        [
            t for (source, cible), t in correspondances.items()
            if source in retenus and cible in retenus
        ],
        tri=lambda l: (l.get("from_stop_id", ""), l.get("to_stop_id", "")),
    )


def _ecrire_feed_info(sortie: Path, plan: dict[str, Provenance], identite: dict[str, str]) -> None:
    dates = sorted(plan)
    gtfs_io.ecrire_table(
        sortie / "feed_info.txt",
        COLONNES_FEED_INFO,
        [
            {
                "feed_id": identite["feed_id"],
                "feed_publisher_name": identite["publisher"],
                "feed_publisher_url": identite["url"],
                "feed_lang": "fr",
                "feed_start_date": dates[0],
                "feed_end_date": dates[-1],
                "feed_version": identite["version"],
            }
        ],
    )


def _verifier_fermeture(sortie: Path, journal) -> dict[str, int]:
    """Checks that every reference of the written feed resolves, in both directions."""
    export = Export(chemin=sortie, etiquette=sortie.name, empreinte="")
    stops = {l["stop_id"] for l in gtfs_io.lire(export, "stops.txt")}
    routes = {l["route_id"] for l in gtfs_io.lire(export, "routes.txt")}
    shapes = {l["shape_id"] for l in gtfs_io.lire(export, "shapes.txt")}
    services = {l["service_id"] for l in gtfs_io.lire(export, "calendar_dates.txt")}

    trips = {}
    orphelins = {"route": 0, "shape": 0, "service": 0, "stop": 0, "trip": 0}
    for ligne in gtfs_io.lire(export, "trips.txt"):
        trips[ligne["trip_id"]] = ligne
        if ligne["route_id"] not in routes:
            orphelins["route"] += 1
        if ligne["shape_id"] and ligne["shape_id"] not in shapes:
            orphelins["shape"] += 1
        if ligne["service_id"] not in services:
            orphelins["service"] += 1
    for ligne in gtfs_io.lire(export, "stop_times.txt"):
        if ligne["trip_id"] not in trips:
            orphelins["trip"] += 1
        if ligne["stop_id"] not in stops:
            orphelins["stop"] += 1

    total = sum(orphelins.values())
    if total:
        journal(f"[ALARME] fermeture référentielle : {total} orphelin(s) — {orphelins}")
    else:
        journal("    fermeture référentielle : aucune référence orpheline")
    return orphelins
