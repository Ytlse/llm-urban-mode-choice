"""Writing the measurement CSVs — ticket 093, lot 2.

TWO REGIMES, AND THE DISTINCTION IS THE HEART OF THE LOT
--------------------------------------------------------
**Flows are rewritten in full, every time.** Trips, modal shares, habits, pool,
concept operations: everything is recomputed from dated, deduplicated logs. A day replayed
after a resume therefore cannot double, and a cut cannot leave a gap in the curve.
Continuity becomes a property of construction, not a precaution one must remember to
take again.

**States are written once and never rewritten.** A state cannot be reconstructed: a memory's
`force` grows at each recall, and an agent's number of entries on day 3 is no longer readable
on day 10. Worse, measured on 2026-09-16 on run `2026-09-16_15_58`: **the replay of a resume
overwrites the checkpoints of days already lived with the frozen state.** Checkpoints `jour_002`
to `jour_009` all carry exactly the same content there — 101 entries, same counters per agent —
because they were rewritten during the replay, while memory was frozen. The growth of the
pool over the first eight days is lost for that run.

Hence the rule: a state line already written is **kept as is**. The first pass
sets it, at the moment the information still exists. That is also what makes case F1 true for
states: they cannot change, since they are not recomputed.

⚠ The directory is created at the FIRST WRITE, never at import: an import that seeds a
run directory is a defect already paid for once in this repository (ticket 075, memory log).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from scripts.analysis.memoire.sources import MODES
from scripts.analysis.mesures.calcul import Mesures, calculer

REPERTOIRE = "mesures"

FICHIER_CHOIX_MODAL = "choix_modal_par_jour.csv"
FICHIER_HABITUDES = "habitudes_par_activite.csv"
FICHIER_MEMOIRE = "memoire_par_jour.csv"
FICHIER_DUREES = "duree_de_vie_par_type.csv"
# Ticket 100 — a single file for both regimes, undergone and read. The ticket 079 name
# (`choc_par_jour.csv`) only described half of what it carries.
FICHIER_CHOC = "evenement_par_jour.csv"
# Analysis of 2026-09-25 — the memory of the event, day after day, for the whole household.
FICHIER_SOUVENIR = "souvenir_evenement_par_jour.csv"
# Snapshot of long-term memory at computation time: OUTSIDE `TABLES`, hence outside the
# continuity check — a concept merged later drops out of it, and that is not a gap.
FICHIER_DERIVES = "souvenirs_derives.csv"
COLONNES_DERIVES = ("evenement_id", "person_id", "role", "doc_id", "type_souvenir",
                    "ecrit_le", "appariement", "mots_retrouves", "enonce")


@dataclass(frozen=True)
class Table:
    """A CSV: its name, its columns, its key, and the STATE columns that are not rewritten."""

    fichier: str
    colonnes: tuple[str, ...]
    cle: tuple[str, ...]
    colonnes_d_etat: tuple[str, ...] = ()


def _colonnes_choix_modal() -> tuple[str, ...]:
    return (
        "jour_simule", "date_simulee", "person_id", "trajets", "trajets_decides",
        "part_decidee", "modes_distincts", *[f"part_{mode}" for mode in MODES],
        "trajets_mode_inconnu",
    )


TABLES = {
    "choix_modal": Table(FICHIER_CHOIX_MODAL, _colonnes_choix_modal(),
                         ("jour_simule", "person_id")),
    "habitudes": Table(
        FICHIER_HABITUDES,
        ("jour_simule", "date_simulee", "person_id", "activite", "motif", "mode", "occurrences",
         "mode_veille", "reprise_veille", "mode_habituel", "conforme_habitude",
         "observations_fenetre"),
        ("jour_simule", "person_id", "activite"),
    ),
    "memoire": Table(
        FICHIER_MEMOIRE,
        ("jour_simule", "date_simulee", "person_id", "rappels", "vivier_min", "vivier_median",
         "vivier_max", "souvenirs_servis", "concepts_crees", "concepts_confirmes",
         "concepts_precises", "concepts_contredits", "operations_hors_vocabulaire",
         "etat_lu_dans", "entrees_ltm"),
        ("jour_simule", "person_id"),
        colonnes_d_etat=("etat_lu_dans", "entrees_ltm"),
    ),
    "durees_de_vie": Table(
        FICHIER_DUREES,
        ("jour_simule", "date_simulee", "person_id", "type_souvenir", "entrees",
         "duree_vie_mediane_jours", "etat_lu_dans"),
        ("jour_simule", "person_id", "type_souvenir"),
        colonnes_d_etat=("entrees", "duree_vie_mediane_jours", "etat_lu_dans"),
    ),
    "chocs": Table(
        FICHIER_CHOC,
        ("jour_simule", "date_simulee", "person_id", "choc_id", "canal", "expositions",
         "minutes_injectees", "incidents_reseau", "correspondances_ratees",
         "souvenir_choc_servi", "decisions_avec_souvenir_choc", "appariement"),
        ("jour_simule", "person_id", "choc_id"),
    ),
    "souvenirs": Table(
        FICHIER_SOUVENIR,
        ("jour_simule", "date_simulee", "person_id", "evenement_id", "role", "informe",
         "jours_depuis_j0", "decisions", "prompts", "decisions_sans_prompt",
         "prompts_avec_souvenir", "souvenir_texte", "souvenir_mots", "via_connaissances",
         "via_changements", "via_rappel"),
        ("jour_simule", "person_id", "evenement_id"),
    ),
}

# Mapping concept operation → column. The columns are fixed and plainly named:
# one column per operation beats an (operation, count) pair that nobody pivots.
_COLONNE_OPERATION = {
    "créé": "concepts_crees",
    "confirmé": "concepts_confirmes",
    "précisé": "concepts_precises",
    "contredit": "concepts_contredits",
}


def _valeur(valeur: Any) -> str:
    """CSV rendering. `None` becomes an EMPTY cell — never a zero.

    In this repository, zero is the perfect score: a vacuity written `0` reads as total
    conformity or as a contradiction that never occurred, and gets quoted as such.
    """
    if valeur is None:
        return ""
    if isinstance(valeur, bool):
        return "1" if valeur else "0"
    if isinstance(valeur, float):
        return f"{valeur:.4f}".rstrip("0").rstrip(".") or "0"
    return str(valeur)


def _lignes(mesures: Mesures, nom: str) -> list[dict[str, Any]]:
    if nom == "choix_modal":
        return [
            {"jour_simule": l.jour_simule, "date_simulee": l.date_simulee,
             "person_id": l.person_id, "trajets": l.trajets,
             "trajets_decides": l.trajets_decides, "part_decidee": l.part_decidee,
             "modes_distincts": l.modes_distincts,
             **{f"part_{mode}": l.parts.get(mode, 0.0) for mode in MODES},
             "trajets_mode_inconnu": l.trajets_mode_inconnu}
            for l in mesures.choix_modal
        ]
    if nom == "habitudes":
        return [
            {"jour_simule": l.jour_simule, "date_simulee": l.date_simulee,
             "person_id": l.person_id, "activite": l.activite, "motif": l.motif,
             "mode": l.mode, "occurrences": l.occurrences, "mode_veille": l.mode_veille,
             "reprise_veille": l.reprise_veille, "mode_habituel": l.mode_habituel,
             "conforme_habitude": l.conforme_habitude,
             "observations_fenetre": l.observations_fenetre}
            for l in mesures.habitudes
        ]
    if nom == "memoire":
        return [
            {"jour_simule": l.jour_simule, "date_simulee": l.date_simulee,
             "person_id": l.person_id, "rappels": l.rappels, "vivier_min": l.vivier_min,
             "vivier_median": l.vivier_median, "vivier_max": l.vivier_max,
             "souvenirs_servis": l.souvenirs_servis,
             **{_COLONNE_OPERATION[op]: valeur for op, valeur in l.operations.items()},
             "operations_hors_vocabulaire": l.operations_hors_vocabulaire,
             "etat_lu_dans": l.etat_lu_dans, "entrees_ltm": l.entrees_ltm}
            for l in mesures.memoire
        ]
    if nom == "durees_de_vie":
        return [
            {"jour_simule": l.jour_simule, "date_simulee": l.date_simulee,
             "person_id": l.person_id, "type_souvenir": l.type_souvenir,
             "entrees": l.entrees, "duree_vie_mediane_jours": l.duree_vie_mediane_jours,
             "etat_lu_dans": l.etat_lu_dans}
            for l in mesures.durees_de_vie
        ]
    if nom == "souvenirs":
        return [{c: getattr(l, c) for c in l.__dataclass_fields__} for l in mesures.souvenirs]
    return [
        {"jour_simule": l.jour_simule, "date_simulee": l.date_simulee,
         "person_id": l.person_id, "choc_id": l.choc_id, "canal": l.canal,
         "expositions": l.expositions,
         "minutes_injectees": l.minutes_injectees, "incidents_reseau": l.incidents_reseau,
         "correspondances_ratees": l.correspondances_ratees,
         "souvenir_choc_servi": l.souvenir_choc_servi,
         "decisions_avec_souvenir_choc": l.decisions_avec_souvenir_choc,
         "appariement": l.appariement}
        for l in mesures.chocs
    ]


def _deja_ecrites(chemin: Path, table: Table) -> dict[tuple[str, ...], dict[str, str]]:
    """Lines of the existing CSV, indexed by their key. Empty if the file does not exist."""
    if not chemin.is_file():
        return {}
    with chemin.open(newline="", encoding="utf-8") as flux:
        return {
            tuple(ligne.get(colonne, "") for colonne in table.cle): ligne
            for ligne in csv.DictReader(flux)
        }


def _ecrire_table(repertoire: Path, table: Table, lignes: Sequence[dict[str, Any]]) -> Path:
    chemin = repertoire / table.fichier
    anciennes = _deja_ecrites(chemin, table)
    with chemin.open("w", newline="", encoding="utf-8") as flux:
        ecrivain = csv.writer(flux)
        ecrivain.writerow(table.colonnes)
        for ligne in lignes:
            clef = tuple(_valeur(ligne.get(colonne)) for colonne in table.cle)
            ancienne = anciennes.get(clef)
            rendue = []
            for colonne in table.colonnes:
                valeur = _valeur(ligne.get(colonne))
                # A state already written is KEPT: it was measured when the information still
                # existed, and recomputing it today would give another value — or the frozen
                # value that a replay left behind.
                if colonne in table.colonnes_d_etat and ancienne is not None:
                    conservee = ancienne.get(colonne, "")
                    valeur = conservee if conservee != "" else valeur
                rendue.append(valeur)
            ecrivain.writerow(rendue)
    return chemin


def ecrire(mesures: Mesures, repertoire: Path | str | None = None) -> dict[str, Path]:
    """Writes the five CSVs and returns their paths, indexed by table name."""
    racine = Path(repertoire) if repertoire else Path(mesures.chemin_run) / REPERTOIRE
    racine.mkdir(parents=True, exist_ok=True)
    chemins = {
        nom: _ecrire_table(racine, table, _lignes(mesures, nom))
        for nom, table in TABLES.items()
    }
    chemins["souvenirs_derives"] = _ecrire_derives(racine, mesures.souvenirs_derives)
    return chemins


def _ecrire_derives(racine: Path, derives: Sequence[Any]) -> Path:
    """Rewritten in full at each computation: it is a snapshot, not a history."""
    chemin = racine / FICHIER_DERIVES
    with chemin.open("w", newline="", encoding="utf-8") as flux:
        ecrivain = csv.writer(flux)
        ecrivain.writerow(COLONNES_DERIVES)
        for d in derives:
            ecrivain.writerow([_valeur(getattr(d, c)) for c in COLONNES_DERIVES])
    return chemin


def mesurer_et_ecrire(chemin_run: Path | str,
                      repertoire: Path | str | None = None) -> dict[str, Path]:
    """The full path: computation then writing. This is what the CLI and the controller call."""
    return ecrire(calculer(chemin_run), repertoire)
