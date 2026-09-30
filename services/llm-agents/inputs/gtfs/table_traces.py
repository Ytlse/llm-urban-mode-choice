"""The shape table: what the recipe publishes and what the runtime reads.

WHY THIS ANNEX FILE EXISTS
--------------------------
To make an agent board a vehicle, `Inhabitant.gaml` compares the vehicle's
`shape_id` with the list that the Python side set on the itinerary
leg (`shape_id_list contains each.shape_id`). This list comes from
`GTFSData.get_shape_id_from_route_info`, which queries
`route_id_shape_lookup_map`.

Until 2026-09-04, this table was built from the **primary feed
only** (`settings.gtfs.gtfs_file`, Tisséo), whereas the layers and the
trips have carried **three** networks since the day before. Measured: **80 `route_id`**
(17 TER, 58 liO regional coaches, 5 Tisséo circular lines) and **2,277
trips** ran in GAMA without any itinerary being able to designate them —
`get_shape_id_from_route_info` returned `[]`, which is **indistinguishable** from
"this line has no shape for this pair of stops".

WHY THE RECIPE PUBLISHES, AND DOES NOT LET THE RUNTIME REBUILD
--------------------------------------------------------------
The TER publishes **no geometry** (`shapes.txt` reduced to its header): its
`shape_id` are *fabricated* by `scripts/data/gama/gtfs_traces.py`
(`<route_id>:<sens>:<empreinte de la suite d'arrêts>`). If the runtime rebuilt them
on its own side, two implementations of the same rule would live in
the repository and drift at the first change — the defect that has just been
closed on the bike ownership law (ticket 034, lot 2).

And rebuilding would not be enough: the recipe **discards** trips (shape not
reconstructible, repeated stop, fewer than two stops, shape outside the scope of
`routes.shp`). The runtime would have to reproduce these exclusions identically to
stay in agreement with the layer. It is therefore the recipe that publishes the
mapping it **really** used, and the runtime that reads it.

THE FRESHNESS CHECK
-------------------
The annex file is written next to its two siblings — `routes.shp` (the
geometry GAMA draws) and `trip_info.json` (the trips) — in
`services/GAMA/CityTransport/includes/`, by the same run of the recipe. It records
for each its **size** and its **sha256 fingerprint**, under their bare file
name: the check is done relative to the directory of the annex file
itself, so it holds identically on the host and in the `controller`
container (which mounts `./GAMA` on `/GAMA`) without any absolute path being
burnt in.

Rebuilding the layers alone (`make gama-layers`) changes the fingerprint of
`routes.shp`: the pair is mismatched, and the runtime raises an alarm at loading instead
of serving a table that no longer designates the right shapes. It is this check
that was missing for five months.

The counters are checked in addition to the fingerprints: a truncated or
hand-edited file has the right fingerprints of its siblings and an incomplete
table. A table whose content does not match its own counters
is refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Format version. To be incremented as soon as the STRUCTURE changes: a runtime that
# reads a format it does not know must say so, not misinterpret it.
FORMAT = 1

NOM_FICHIER = "shape_lookup.json"

# The siblings whose fingerprint makes the freshness. `routes.dbf` carries the attributes
# of the layer (including `shape_id`); `routes.shp` the geometries. Both count:
# a layer whose attributes alone change designates other shapes.
TEMOINS = ("routes.shp", "routes.dbf", "trip_info.json")

_BLOC = 1 << 20


class TableTracesInvalide(Exception):
    """The annex table is unusable — with the reason, so that the alarm says it.

    `motif` is a short and stable label (`absente`, `illisible`,
    `format`, `temoin_absent`, `depareillee`, `comptes`): it serves to write
    an actionable alarm message, not to decide on a fallback.
    """

    def __init__(self, motif: str, detail: str):
        super().__init__(detail)
        self.motif = motif
        self.detail = detail


def empreinte_fichier(chemin: Path) -> tuple[int, str] | None:
    """(size in bytes, hexadecimal sha256) — `None` if the file does not exist."""
    chemin = Path(chemin)
    if not chemin.exists():
        return None
    digest = hashlib.sha256()
    taille = 0
    with open(chemin, "rb") as fh:
        while True:
            bloc = fh.read(_BLOC)
            if not bloc:
                break
            taille += len(bloc)
            digest.update(bloc)
    return taille, digest.hexdigest()


def empreintes_des_temoins(dossier: Path, temoins=TEMOINS) -> dict[str, dict]:
    """The fingerprints of the siblings present in `dossier`, by file name.

    A missing control file is not recorded: the recipe cannot promise the
    freshness of a file it has not seen. Loading, on the other hand, requires that
    every **recorded** control file be present and identical.
    """
    dossier = Path(dossier)
    releve: dict[str, dict] = {}
    for nom in temoins:
        mesure = empreinte_fichier(dossier / nom)
        if mesure is not None:
            releve[nom] = {"octets": mesure[0], "sha256": mesure[1]}
    return releve


def comptes_de_la_table(table: dict) -> dict[str, int]:
    """The counters that describe the table — those that loading cross-checks."""
    traces = sum(len(par_trace) for par_trace in table.values())
    arrets_notes = sum(len(stops) for par_trace in table.values()
                       for stops in par_trace.values())
    return {"route_id": len(table), "traces": traces, "couples_trace_arret": arrets_notes}


def construire(
    *,
    table: dict,
    arrets: dict,
    dossier_temoins: Path,
    genere_le: str,
    recette: str,
    noms_temoins=TEMOINS,
    reseaux: dict | None = None,
    comptes_supplementaires: dict | None = None,
) -> dict:
    """The document to write — table, stop catalogue and concordance.

    `noms_temoins` carries the names of the files whose freshness will be cross-checked.
    The recipe passes them explicitly rather than relying on `TEMOINS`:
    a test that names its layer differently must see its REAL layer checked,
    and not zero control files — that is, a freshness that would measure nothing.
    """
    comptes = comptes_de_la_table(table)
    comptes["arrets_catalogue"] = len(arrets)
    if comptes_supplementaires:
        comptes.update(comptes_supplementaires)
    return {
        "format": FORMAT,
        "genere_le": genere_le,
        "recette": recette,
        "concordance": {
            "temoins": empreintes_des_temoins(dossier_temoins, noms_temoins),
            "comptes": comptes,
        },
        "reseaux": reseaux or {},
        "table": table,
        "arrets": arrets,
    }


def ecrire(chemin: Path, document: dict) -> int:
    """Write the document and return its size in bytes."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False)
    return chemin.stat().st_size


def charger(chemin: Path) -> tuple[dict, dict, dict]:
    """Read the annex table, or raise `TableTracesInvalide` with its reason.

    Returns `(table, arrets, journal)` where `table` is
    `route_id → {shape_id → {stop_id: stop_sequence}}` — the exact structure
    consumed by `GTFSData.get_shape_id_from_route_info` — and `arrets` the catalogue
    `stop_id → {stop_name, stop_lat, stop_lon}` of the stops served by these
    shapes, all networks combined.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise TableTracesInvalide(
            "absente",
            f"{chemin} does not exist — the recipe scripts/data/gama/export_trip_info.py "
            f"(make gama-trip-info) never wrote it, or the ./GAMA mount does not make it "
            f"visible")
    try:
        document = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TableTracesInvalide("illisible", f"{chemin} : {exc}") from exc
    if not isinstance(document, dict):
        raise TableTracesInvalide("illisible", f"{chemin} : the document is not a JSON object")

    format_lu = document.get("format")
    if format_lu != FORMAT:
        raise TableTracesInvalide(
            "format",
            f"{chemin} : format {format_lu!r}, this runtime reads format {FORMAT} — "
            f"rerun the recipe")

    concordance = document.get("concordance") or {}
    temoins = concordance.get("temoins") or {}
    if not temoins:
        raise TableTracesInvalide(
            "depareillee",
            f"{chemin} : no freshness control file recorded — impossible to check that the "
            f"table comes from the same generation as routes.shp and trip_info.json")
    ecarts = []
    for nom, attendu in sorted(temoins.items()):
        mesure = empreinte_fichier(chemin.parent / nom)
        if mesure is None:
            raise TableTracesInvalide(
                "temoin_absent",
                f"{chemin.parent / nom} recorded by the table but missing from disk")
        if mesure[0] != attendu.get("octets") or mesure[1] != attendu.get("sha256"):
            ecarts.append(f"{nom} (noté {attendu.get('octets')} o "
                          f"{str(attendu.get('sha256'))[:12]}…, trouvé {mesure[0]} o "
                          f"{mesure[1][:12]}…)")
    if ecarts:
        raise TableTracesInvalide(
            "depareillee",
            f"{chemin} does not come from the same generation as: {', '.join(ecarts)} — "
            f"rerun make gama-trip-info, which rebuilds the layers THEN the trips")

    table_brute = document.get("table")
    if not isinstance(table_brute, dict) or not table_brute:
        raise TableTracesInvalide("illisible", f"{chemin} : table missing or empty")
    table = {
        str(route_id): {
            str(shape_id): {str(stop_id): int(rang) for stop_id, rang in stops.items()}
            for shape_id, stops in par_trace.items()
        }
        for route_id, par_trace in table_brute.items()
    }

    notes = (concordance.get("comptes") or {})
    mesures = comptes_de_la_table(table)
    desaccords = {cle: (notes[cle], mesures[cle]) for cle in mesures
                  if cle in notes and notes[cle] != mesures[cle]}
    if desaccords:
        raise TableTracesInvalide(
            "comptes",
            f"{chemin} : the table read does not match its own counters "
            f"(recorded, read) {desaccords} — file truncated or hand-edited")

    arrets = {
        str(stop_id): {
            "stop_name": str(valeur.get("stop_name", stop_id)),
            "stop_lat": float(valeur["stop_lat"]),
            "stop_lon": float(valeur["stop_lon"]),
        }
        for stop_id, valeur in (document.get("arrets") or {}).items()
    }

    journal = {
        "chemin": str(chemin),
        "genere_le": document.get("genere_le"),
        "recette": document.get("recette"),
        "reseaux": document.get("reseaux") or {},
        "comptes": mesures,
        "arrets_catalogue": len(arrets),
        "temoins": sorted(temoins),
    }
    return table, arrets, journal
