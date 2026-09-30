"""
Extracts a window of a few weeks from a yearly feed.

WHY
---
OTP has no trouble consuming a feed covering the whole year. The GAMA chain
does: the service calendar is encoded there as a 64-bit binary mask —
`assert len(all_dates) <= 64` in `services/llm-agents/inputs/gtfs/gama.py`, decoded on the
model side by `PublicTransport.gaml` (`trip_calendar_map` and `BITWISE_BIT_VAL`).
Beyond 64 dates, the export fails; and `build_trips` scans every trip
for each of them, which makes a yearly feed impractical anyway
(28 MB of `trip_info.json` for 58 days and 39,000 trips).

The window is therefore what GAMA and the runtime see, the yearly feed what
OTP sees. It must contain the simulation date: outside the calendar,
`is_trip_available_today` merely issues a warning and no longer schedules
any run.

USAGE
-----
    make gtfs-window START=2026-03-16 DAYS=64
    python -m scripts.data.gtfs_year.window_feed --source data/gtfs_year/tisseo_2026 \\
        --debut 20260316 --jours 64 --sortie data/gtfs_year/fenetre_gama

The window is written OUTSIDE `data/gtfs/`: installing it into the live feed is
a publication step, described in `docs/arch/gtfs-annee.md`, not the default.

EXIT CODES
----------
    0  window extracted
    1  source not found
    2  empty window, or longer than what the binary mask supports
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.gtfs_year import gtfs_io  # noqa: E402
from scripts.data.gtfs_year.gtfs_io import Export  # noqa: E402

LIMITE_MASQUE = 64

CODE_RESSOURCE = 1
CODE_REFUS = 2


def fenetrer(
    source: Path, debut: str, jours: int, sortie: Path, journal=print
) -> int:
    """Copies into `sortie` the part of the `source` feed active over the window."""
    if not source.exists():
        journal(f"[ALARME] source introuvable : {source}")
        return CODE_RESSOURCE
    if jours > LIMITE_MASQUE:
        journal(
            f"[ALARME] fenêtre de {jours} jours : le masque binaire du modèle GAMA "
            f"n'en supporte que {LIMITE_MASQUE}"
        )
        return CODE_REFUS

    premier = dt.date(int(debut[:4]), int(debut[4:6]), int(debut[6:8]))
    fenetre = {
        (premier + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(jours)
    }
    entree = Export(chemin=source, etiquette=source.name, empreinte="")
    sortie.mkdir(parents=True, exist_ok=True)

    calendrier = [
        ligne for ligne in gtfs_io.lire(entree, "calendar_dates.txt")
        if ligne["date"] in fenetre
    ]
    services = {ligne["service_id"] for ligne in calendrier}
    if not services:
        journal(f"[ALARME] aucune date du feed dans la fenêtre {debut} +{jours} j")
        return CODE_REFUS

    trips = [
        ligne for ligne in gtfs_io.lire(entree, "trips.txt")
        if ligne["service_id"] in services
    ]
    trips_retenus = {ligne["trip_id"] for ligne in trips}
    routes_retenues = {ligne["route_id"] for ligne in trips}
    shapes_retenues = {ligne.get("shape_id", "") for ligne in trips} - {""}

    horaires = [
        ligne for ligne in gtfs_io.lire(entree, "stop_times.txt")
        if ligne["trip_id"] in trips_retenus
    ]
    arrets_retenus = {ligne["stop_id"] for ligne in horaires}

    tables = [
        ("calendar_dates.txt", calendrier, lambda l: (l["service_id"], l["date"])),
        ("trips.txt", trips, lambda l: (l["route_id"], l["service_id"], l["trip_id"])),
        ("stop_times.txt", horaires, lambda l: (l["trip_id"], int(l["stop_sequence"]))),
    ]
    for nom, lignes, tri in tables:
        colonnes = gtfs_io.entetes(entree, nom)
        gtfs_io.ecrire_table(sortie / nom, colonnes, lignes, tri=tri)

    filtres = [
        ("routes.txt", lambda l: l["route_id"] in routes_retenues, lambda l: l["route_id"]),
        ("shapes.txt", lambda l: l["shape_id"] in shapes_retenues,
         lambda l: (l["shape_id"], int(l["shape_pt_sequence"]))),
        ("agency.txt", lambda l: True, lambda l: l.get("agency_id", "")),
        ("calendar.txt", lambda l: False, None),
        ("feed_info.txt", lambda l: True, None),
    ]
    for nom, garder, tri in filtres:
        colonnes = gtfs_io.entetes(entree, nom)
        if not colonnes:
            continue
        gtfs_io.ecrire_table(
            sortie / nom, colonnes, [l for l in gtfs_io.lire(entree, nom) if garder(l)], tri=tri
        )

    # Stops bring their parent stations along, otherwise OTP reports dangling
    # references at every graph build.
    tous_arrets = {l["stop_id"]: l for l in gtfs_io.lire(entree, "stops.txt")}
    retenus = set(arrets_retenus)
    for _ in range(4):
        parents = {
            tous_arrets[s].get("parent_station", "")
            for s in retenus
            if s in tous_arrets and tous_arrets[s].get("parent_station")
        } - {""}
        if not parents - retenus:
            break
        retenus |= parents
    gtfs_io.ecrire_table(
        sortie / "stops.txt",
        gtfs_io.entetes(entree, "stops.txt"),
        [l for sid, l in tous_arrets.items() if sid in retenus],
        tri=lambda l: l["stop_id"],
    )
    colonnes_transferts = gtfs_io.entetes(entree, "transfers.txt")
    if colonnes_transferts:
        gtfs_io.ecrire_table(
            sortie / "transfers.txt",
            colonnes_transferts,
            [
                l for l in gtfs_io.lire(entree, "transfers.txt")
                if l.get("from_stop_id") in retenus and l.get("to_stop_id") in retenus
            ],
            tri=lambda l: (l.get("from_stop_id", ""), l.get("to_stop_id", "")),
        )

    dates = sorted({l["date"] for l in calendrier})
    journal(
        f"    fenêtre {dates[0]} → {dates[-1]} ({len(dates)} date(s) servies sur {jours} demandées) : "
        f"{len(trips):,} trips, {len(horaires):,} horaires, {len(services):,} services, "
        f"{len(retenus):,} arrêts → {sortie}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parseur = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parseur.add_argument("--source", type=Path, required=True, help="yearly feed (directory or zip)")
    parseur.add_argument("--debut", required=True, help="first day of the window, YYYYMMDD")
    parseur.add_argument("--jours", type=int, default=LIMITE_MASQUE)
    parseur.add_argument("--sortie", type=Path, required=True)
    parseur.add_argument("--zip", action="store_true", help="archive the window next to the directory")
    args = parseur.parse_args(argv)

    debut = args.debut.replace("-", "")
    code = fenetrer(args.source, debut, args.jours, args.sortie, print)
    if code == 0 and args.zip:
        archive = gtfs_io.zipper(args.sortie, args.sortie.with_suffix(".zip"))
        print(f"    archive: {archive} ({archive.stat().st_size / 1_048_576:.1f} MB)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
