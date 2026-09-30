"""
Reading, canonicalisation and writing of GTFS data sets.

Everything is read as character strings, never converted to numbers: a `direction_id`
or a GTFS date turned into an integer comes back in a different format and
silently breaks the joins (this is why `STRING_COLUMNS` exists
in `services/llm-agents/inputs/gtfs/reader.py`, applied here to every column).

Canonicalisation is essential before any comparison between exports:
coordinates and distances vary in their last decimals from one
export to the next (`1.579197043824882` → `1.5791970438244147`). Without rounding,
270 stops and 21 geometries look divergent while only 3 of them really
are.

Reading is designed to fit in memory on a yearly data set: the large
files (`stop_times.txt`, `shapes.txt`) are streamed and filtered on
the fly on the identifiers actually kept.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

FICHIERS_GTFS = (
    "agency.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "routes.txt",
    "shapes.txt",
    "stop_times.txt",
    "stops.txt",
    "transfers.txt",
    "trips.txt",
    "feed_info.txt",
    "fare_attributes.txt",
    "fare_rules.txt",
)

csv.field_size_limit(10_000_000)


# ──────────────────────────────────────────────────────────────────────────────
# Exports
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Export:
    """A GTFS export from the operator: a zip or a directory."""

    chemin: Path
    etiquette: str
    empreinte: str
    date_min: str = ""
    date_max: str = ""
    fin_fiable: str = ""
    jours_fiables: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        return self.etiquette


def empreinte_fichier(chemin: Path) -> str:
    """md5 of a zip, or md5 of the GTFS files of a directory."""
    digest = hashlib.md5()
    if chemin.is_file():
        digest.update(chemin.read_bytes())
        return digest.hexdigest()
    for nom in sorted(FICHIERS_GTFS):
        fichier = chemin / nom
        if fichier.exists():
            digest.update(nom.encode())
            digest.update(fichier.read_bytes())
    return digest.hexdigest()


def decouvrir(racines: Iterable[Path], journal=print) -> list[Export]:
    """Inventories the exports, discarding strict duplicates.

    The same export downloaded twice (`… (3).zip` and `… (4).zip`, identical
    md5) must count only once, otherwise the provenance statistics
    are wrong.
    """
    candidats: list[Path] = []
    for racine in racines:
        if not racine.exists():
            journal(f"    exports : {racine} absent, ignoré")
            continue
        if racine.is_file() and racine.suffix == ".zip":
            candidats.append(racine)
            continue
        if (racine / "trips.txt").exists():
            candidats.append(racine)
            continue
        candidats.extend(sorted(racine.glob("*.zip")))
        for sous in sorted(p for p in racine.iterdir() if p.is_dir()):
            if (sous / "trips.txt").exists():
                candidats.append(sous)

    exports: list[Export] = []
    vus: dict[str, str] = {}
    doublons = 0
    for chemin in candidats:
        empreinte = empreinte_fichier(chemin)
        if empreinte in vus:
            doublons += 1
            journal(f"    exports : {chemin.name} identique à {vus[empreinte]}, ignoré")
            continue
        vus[empreinte] = chemin.name
        exports.append(Export(chemin=chemin, etiquette=chemin.stem, empreinte=empreinte))
    journal(f"    exports : {len(exports)} retenus, {doublons} doublon(s) écarté(s)")
    return exports


# ──────────────────────────────────────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────────────────────────────────────


def _flux(export: Export, nom: str) -> io.TextIOWrapper | None:
    if export.chemin.is_file():
        archive = zipfile.ZipFile(export.chemin)
        if nom not in archive.namelist():
            return None
        return io.TextIOWrapper(archive.open(nom), encoding="utf-8-sig", newline="")
    fichier = export.chemin / nom
    if not fichier.exists():
        return None
    return open(fichier, encoding="utf-8-sig", newline="")


def existe(export: Export, nom: str) -> bool:
    flux = _flux(export, nom)
    if flux is None:
        return False
    flux.close()
    return True


def lire(export: Export, nom: str) -> Iterator[dict[str, str]]:
    """Walks through a GTFS file line by line, all as strings."""
    flux = _flux(export, nom)
    if flux is None:
        return
    try:
        for ligne in csv.DictReader(flux):
            yield ligne
    finally:
        flux.close()


def entetes(export: Export, nom: str) -> list[str]:
    flux = _flux(export, nom)
    if flux is None:
        return []
    try:
        return csv.DictReader(flux).fieldnames or []
    finally:
        flux.close()


# ──────────────────────────────────────────────────────────────────────────────
# Canonicalisation
# ──────────────────────────────────────────────────────────────────────────────


def arrondir(valeur: str, decimales: int) -> str:
    """Rounds a textual numeric value, letting the empty value through."""
    if valeur is None or valeur == "":
        return ""
    try:
        return f"{round(float(valeur), decimales):.{decimales}f}"
    except ValueError:
        return valeur


def canoniser_arret(ligne: dict[str, str], decimales: int) -> dict[str, str]:
    sortie = dict(ligne)
    for champ in ("stop_lat", "stop_lon"):
        if champ in sortie:
            sortie[champ] = arrondir(sortie[champ], decimales)
    return sortie


def canoniser_point_shape(ligne: dict[str, str], dec_coord: int, dec_dist: int) -> dict[str, str]:
    sortie = dict(ligne)
    for champ in ("shape_pt_lat", "shape_pt_lon"):
        if champ in sortie:
            sortie[champ] = arrondir(sortie[champ], dec_coord)
    if "shape_dist_traveled" in sortie:
        sortie["shape_dist_traveled"] = arrondir(sortie["shape_dist_traveled"], dec_dist)
    return sortie


def canoniser_horaire(ligne: dict[str, str], dec_dist: int) -> dict[str, str]:
    sortie = dict(ligne)
    if "shape_dist_traveled" in sortie:
        sortie["shape_dist_traveled"] = arrondir(sortie["shape_dist_traveled"], dec_dist)
    return sortie


def distance_m(lat1: str, lon1: str, lat2: str, lon2: str) -> float:
    """Approximate distance in metres between two stop positions.

    Equirectangular: at the scale of a few tens of metres and at the
    latitude of Toulouse, the gap with the haversine formula is negligible.
    """
    import math

    try:
        phi1, lambda1 = math.radians(float(lat1)), math.radians(float(lon1))
        phi2, lambda2 = math.radians(float(lat2)), math.radians(float(lon2))
    except (TypeError, ValueError):
        return 0.0
    x = (lambda2 - lambda1) * math.cos((phi1 + phi2) / 2)
    y = phi2 - phi1
    return math.hypot(x, y) * 6_371_000


# ──────────────────────────────────────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────────────────────────────────────


class EcrivainCSV:
    """Writes a GTFS file as a stream, enforcing a stable set of columns."""

    def __init__(self, chemin: Path, colonnes: list[str]):
        chemin.parent.mkdir(parents=True, exist_ok=True)
        self.chemin = chemin
        self.colonnes = colonnes
        self._fichier = open(chemin, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._fichier, fieldnames=colonnes, extrasaction="ignore", lineterminator="\n"
        )
        self._writer.writeheader()
        self.lignes = 0

    def ecrire(self, ligne: dict[str, str]) -> None:
        self._writer.writerow({c: ligne.get(c, "") for c in self.colonnes})
        self.lignes += 1

    def fermer(self) -> None:
        self._fichier.close()

    def __enter__(self) -> "EcrivainCSV":
        return self

    def __exit__(self, *_exc) -> None:
        self.fermer()


def ecrire_table(
    chemin: Path,
    colonnes: list[str],
    lignes: Iterable[dict[str, str]],
    tri: Callable[[dict[str, str]], tuple] | None = None,
) -> int:
    """Writes a complete table, sorted so that the build is reproducible."""
    materialisees = list(lignes)
    if tri is not None:
        materialisees.sort(key=tri)
    with EcrivainCSV(chemin, colonnes) as ecrivain:
        for ligne in materialisees:
            ecrivain.ecrire(ligne)
        return ecrivain.lignes


def zipper(repertoire: Path, cible: Path) -> Path:
    """Archives a GTFS directory, without timestamps, for a reproducible zip."""
    cible.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cible, "w", zipfile.ZIP_DEFLATED) as archive:
        for nom in sorted(p.name for p in repertoire.glob("*.txt")):
            info = zipfile.ZipInfo(nom, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (repertoire / nom).read_bytes())
    return cible
