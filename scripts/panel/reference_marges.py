"""reference_marges.py — The reference margins of the population check, assembled and sourced.

    services/llm-agents/.venv/bin/python -m scripts.panel.reference_marges              # prints the margins
    services/llm-agents/.venv/bin/python -m scripts.panel.reference_marges --recompute  # re-freezes the joint target

WHAT IT IS FOR. `control_population.py` compares a synthetic population with the population
surveyed by EMC² 2023. Every compared margin must have a target whose ORIGIN is known
— page of the published report, or recomputation on the microdata with the right weight.
This module is the only place where these targets are assembled; the check only knows
:class:`Marge` objects.

TWO SOURCES, AND THEIR ORDER. (1) The versioned framing `population_emc2_2023.yaml`, read by
`mobility_core.population_reference`, carries the margins PUBLISHED by the AUAT/CEREMA report
(age classes and occupation p. 11, household car ownership p. 21, population by ring
and household size p. 10). (2) The JOINT ring × car ownership target, on a PERSON base,
is published nowhere: the report gives car ownership by ring on a HOUSEHOLD base
(p. 21), and a synthetic population is a sample of persons — multi-car households
weigh their household size in it. It is recomputed from the ProGEDO microdata
(COEP weighting) and FROZEN in `cible_jointe_couronne_motorisation.yaml`, with its
provenance, so that the check runs without the restricted-access microdata.

WHAT HAS NO TARGET, AND SAYS SO. The report publishes neither the breakdown by sex nor the
driving licence holding rate. These two margins exist here with `cible_pct = None`: the
check returns them as `non mesurable — aucune cible publiée`, never a 0. When the microdata
are present, `--recompute` gives a CANDIDATE value for them (P2, P7, COEP), labelled as
such — it is a recomputation, not a publication.

⚠ Two bases never to be confused (population_reference, ticket 020): a PERSON target
is compared with a count of persons (weight 1 per persona); a HOUSEHOLD target is compared with
a count of households (weight 1/size per persona). The base is carried by each margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mobility_core.population_reference import (  # noqa: E402
    COURONNES, COURONNES_FR, couronne_canonique, couronne_population_shares,
    household_targets, population_reference)

logger = logging.getLogger("panel.reference")

JOINT_TARGET = Path(__file__).resolve().parent / "cible_jointe_couronne_motorisation.yaml"
JOINT_VERSION = "cj1"

PROGEDO_STD = (REPO_ROOT / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV"
               / "fichiers_standards")
SIG_DTIR = (REPO_ROOT / "data" / "PROGEDO 2023" / "lil-1750-Documentation" / "SIG"
            / "EMC2_Toulouse_2023_DTIR_17072023.shp")

# Published report: AUAT/CEREMA, « Enquête mobilité 2023 — bassin de vie toulousain »,
# 68 pages. The pages are those of the PDF; every value of the YAML framing was checked there on
# 2026-09-02 (cf. `population_emc2_2023.yaml`, block `sources_publication`).
RAPPORT = ("AUAT/CEREMA, Rapport final EMC² 2023 — bassin de vie toulousain "
           "(68 p., mai 2024)")

# ── Categories, in published order (the order IS the metric for the EMD) ────────

# Age classes of the report (p. 11), keys of the YAML framing and inclusive bounds.
AGE_CLASSES: tuple[tuple[str, str, int, int], ...] = (
    ("5-17 ans", "5-17_ans", 5, 17),
    ("18-24 ans", "18-24_ans", 18, 24),
    ("25-34 ans", "25-34_ans", 25, 34),
    ("35-49 ans", "35-49_ans", 35, 49),
    ("50-64 ans", "50-64_ans", 50, 64),
    ("65 ans et +", "65_ans_et_plus", 65, 200),
)

# Main occupation (p. 11): label, key of the YAML framing, and the persona
# `main_occupation` category (recoding `frames.OCCUPATION_MAP` → cerema key).
OCCUPATIONS: tuple[tuple[str, str, str], ...] = (
    ("Scolaires", "scolaires_jusqu_au_bac", "scolaire"),
    ("Étudiants", "etudiants", "etudiant"),
    ("Actifs temps plein", "actifs_temps_plein", "actif_temps_plein"),
    ("Actifs temps partiel", "actifs_temps_partiel", "actif_temps_partiel"),
    ("En recherche d'emploi", "en_recherche_emploi", "chomeur_recherche_emploi"),
    ("Retraités", "retraites", "Retraité"),
    ("Autres", "autres", "autres"),
)

MOTORISATION: tuple[str, ...] = ("sans voiture", "une voiture", "deux voitures et +")

GENRES: tuple[str, ...] = ("Femmes", "Hommes")

# ── Frozen person margins (cm1): those the report does not publish at this step ─────────
MARGES_TARGET = Path(__file__).resolve().parent / "cibles_marges_personne.yaml"
MARGES_VERSION = "cm1"

AGE5: tuple[str, ...] = ("5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
                         "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
                         "75 et +")
TAILLE_MENAGE: tuple[str, ...] = ("1", "2", "3", "4", "5 et +")
# Labels of the `housing_type` trait set by enrich_housing_type (ticket 019) — they are
# also those of M1 (households file), through the HOUSING recoding of export_housing_type.
LOGEMENT: tuple[str, ...] = ("Individuel isolé", "Individuel accolé", "Petit habitat collectif",
                             "Grand habitat collectif", "Autres")
OUI_NON: tuple[str, ...] = ("Oui", "Non")
_M1_LABEL = {"1": "Individuel isolé", "2": "Individuel accolé", "3": "Petit habitat collectif",
             "4": "Grand habitat collectif", "5": "Autres"}
# P12 « possession d'un abonnement TC valide hier »: any form of pass (free,
# paid, reimbursed or not, unspecified) counts as Oui; only « 4 = Non » counts as Non.
_P12_OUI = {"1", "2", "3", "5", "6"}


def age5_class(age) -> Optional[str]:
    """Five-year class of the report; `None` under 5 years or unknown age."""
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return None
    if a < 5:
        return None
    return AGE5[min((a - 5) // 5, len(AGE5) - 1)]


def taille_menage_class(size) -> Optional[str]:
    try:
        s = int(float(size))
    except (TypeError, ValueError):
        return None
    if s < 1:
        return None
    return TAILLE_MENAGE[min(s, 5) - 1]


#: Stable key of `mobility_core.housing_type` → label published by the report (p. 26).
#: The KEY is the pivot, not the label: it is French and did not move in ticket 074,
#: when the persona label went from « Individuel isolé » to « Detached house ».
_LOGEMENT_PAR_CLE: dict[str, str] = {
    "individuel_isole": "Individuel isolé",
    "individuel_accole": "Individuel accolé",
    "petit_habitat_collectif": "Petit habitat collectif",
    "grand_habitat_collectif": "Grand habitat collectif",
    "autres": "Autres",
}

_logement_inconnus: set[str] = set()


def logement_label(housing_type) -> Optional[str]:
    """Housing label of a persona; `None` if the trait is not set.

    The trait goes through `housing_type.key_for`, which knows BOTH vocabularies: the
    English labels of v6 and the French ones of earlier cohorts. Comparing the persona
    label with the French list would drop a whole v6 cohort into « Autres » —
    and « Autres » IS a published category, so nothing would say that the measure is empty: one
    would read a 99.6-point gap on a single row, without knowing that it comes from the
    vocabulary and not from the draw.
    """
    if housing_type is None or str(housing_type) == "":
        return None
    from mobility_core.housing_type import key_for

    texte = str(housing_type).strip()
    cle = key_for(texte)
    if cle is None:
        # « Autres » stays the fallback — it is a real category — but it is STATED once
        # per unknown value, otherwise a category invented upstream is counted silently.
        if texte not in _logement_inconnus:
            _logement_inconnus.add(texte)
            logger.error("[ALARME] housing: category %r outside both known vocabularies "
                         "(%s) — counted in « Autres »", texte,
                         ", ".join(sorted(_LOGEMENT_PAR_CLE)))
        return "Autres"
    return _LOGEMENT_PAR_CLE[cle]


def oui_non(value) -> Optional[str]:
    if value is None:
        return None
    return "Oui" if bool(value) else "Non"


_OCCUPATION_LABEL_BY_CEREMA = {cerema: label for label, _key, cerema in OCCUPATIONS}


def occupation_label(main_occupation) -> str:
    """Occupation label of the report (p. 11) from the persona's `main_occupation`.

    Goes through the recoding of the scoring engine (`frames.OCCUPATION_MAP`, the 7 FR
    categories of the persona → cerema keys) then to the published label. An unknown value
    falls into « Autres », which IS a published category (5 %) — the check counts such cases apart.
    """
    from scripts.synthesis.frames import OCCUPATION_MAP

    cerema = OCCUPATION_MAP.get(str(main_occupation))
    return _OCCUPATION_LABEL_BY_CEREMA.get(cerema, "Autres") if cerema else "Autres"


def motorisation_class(number_of_cars) -> Optional[str]:
    """Car-ownership class of a number of cars; `None` if unknown."""
    try:
        n = int(float(number_of_cars))
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    return MOTORISATION[min(n, 2)]


def age_class(age) -> Optional[str]:
    """Age class of the report; `None` under 5 years or unknown age."""
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return None
    for label, _key, low, high in AGE_CLASSES:
        if low <= a <= high:
            return label
    return None


@dataclass(frozen=True)
class Marge:
    """A checked variable: its ordered categories and its target, or the lack of a target."""
    nom: str
    unite: str                 # "personne" | "menage"
    echelle: str               # "ordinale" | "nominale"
    modalites: tuple[str, ...]
    cible_pct: Optional[dict[str, float]]
    source_cible: str
    perimetre: str = "2023"
    note: str = ""

    @property
    def mesurable(self) -> bool:
        return self.cible_pct is not None


class ReferenceError(ValueError):
    """Target missing, unreadable or inconsistent. Never a silent fallback."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Frozen joint target ────────────────────────────────────────────────────────

def _couronnes_canoniques(doc: dict) -> dict:
    """Returns the joint target with its rings in the language of the platform.

    The frozen file is a QUOTATION of the survey: it keeps the labels of the report
    (`1ere couronne`), and its `sha256` is published in `empreintes.txt` — translating it
    would break the seal of every cohort already sealed. The translation is therefore done
    here, on reading, as for occupation (`occupation_du_spec`) and housing
    (`key_for`): the boundary is the only place where the two vocabularies meet.

    Without it, the check no longer found a single one of the four rows and refused to
    serve — which is the right behaviour, but for a reason that is not the
    right one (`[ALARME] joint target: row '1st ring' missing`).
    """
    doc = dict(doc)
    if isinstance(doc.get("cible_pct"), dict):
        doc["cible_pct"] = {couronne_canonique(c): v for c, v in doc["cible_pct"].items()}
    marges = doc.get("marges_pct")
    if isinstance(marges, dict) and isinstance(marges.get("couronne"), dict):
        marges = dict(marges)
        marges["couronne"] = {couronne_canonique(c): v
                              for c, v in marges["couronne"].items()}
        doc["marges_pct"] = marges
    return doc


def cible_jointe(path: Path = JOINT_TARGET) -> dict:
    """The frozen joint target, validated: version, sums, categories.

    Raises :class:`ReferenceError` if the file is missing — it is a VERSIONED resource,
    its absence is an anomaly, not a normal case.
    """
    if not path.exists():
        raise ReferenceError(
            f"joint target missing: {path}. It is versioned; regenerating it requires "
            "the ProGEDO microdata: `python -m scripts.panel.reference_marges "
            "--recompute`.")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if doc.get("version") != JOINT_VERSION:
        raise ReferenceError(
            f"joint target {path.name} at version {doc.get('version')!r}, expected "
            f"{JOINT_VERSION!r}: the check does not serve an outdated target « at best ».")
    doc = _couronnes_canoniques(doc)
    cells = doc.get("cible_pct") or {}
    total = 0.0
    for couronne in COURONNES:
        row = cells.get(couronne)
        if not isinstance(row, dict) or set(row) != set(MOTORISATION):
            raise ReferenceError(
                f"joint target: row {couronne!r} missing or unexpected categories")
        total += sum(float(v) for v in row.values())
    if abs(total - 100.0) > 0.05:
        raise ReferenceError(f"joint target: the 12 cells sum to {total:.3f}, not 100")
    return doc


def _flat_joint(doc: dict) -> dict[str, float]:
    return {f"{c} × {m}": float(doc["cible_pct"][c][m])
            for c in COURONNES for m in MOTORISATION}


# Margins of `cibles_marges_personne.yaml`: name → (scale, categories in published order).
MARGES_PERSONNE: dict[str, tuple[str, tuple[str, ...]]] = {
    "age_quinquennal": ("ordinale", AGE5),
    "genre": ("nominale", GENRES),
    "taille_menage_personne": ("ordinale", TAILLE_MENAGE),
    "permis_adultes": ("nominale", OUI_NON),
    "abonnement_tc": ("nominale", OUI_NON),
    "logement": ("nominale", LOGEMENT),
    "immobile": ("nominale", OUI_NON),
}


def cibles_marges(path: Path = MARGES_TARGET) -> dict:
    """The frozen person margins, validated: version, categories, sums to 100."""
    if not path.exists():
        raise ReferenceError(
            f"margin targets missing: {path}. Versioned resource; regenerating it requires the "
            "ProGEDO microdata: `python -m scripts.panel.reference_marges --recompute`.")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if doc.get("version") != MARGES_VERSION:
        raise ReferenceError(
            f"margin targets {path.name} at version {doc.get('version')!r}, expected "
            f"{MARGES_VERSION!r}: an outdated target is not served « at best ».")
    cibles = doc.get("cibles_pct") or {}
    for nom, (_echelle, modalites) in MARGES_PERSONNE.items():
        row = cibles.get(nom)
        if not isinstance(row, dict) or set(row) != set(modalites):
            raise ReferenceError(f"margin targets: « {nom} » missing or unexpected categories "
                                 f"({sorted(row) if isinstance(row, dict) else row})")
        total = sum(float(v) for v in row.values())
        if abs(total - 100.0) > 0.05:
            raise ReferenceError(f"margin targets: « {nom} » sums to {total:.3f}, not 100")
    return doc


# ── Assembling the margins ────────────────────────────────────────────────────

def marges(joint_path: Path = JOINT_TARGET, marges_path: Path = MARGES_TARGET) -> list[Marge]:
    """All the margins of the check, in rendering order."""
    reference = population_reference()
    yaml_src = f"{RAPPORT} p. 11, via population_emc2_2023.yaml"

    ages = reference["population"]["repartition_par_classe_age"]
    age_marge = Marge(
        "classe_age", "personne", "ordinale",
        tuple(label for label, *_ in AGE_CLASSES),
        {label: float(ages[key]) for label, key, *_ in AGE_CLASSES},
        yaml_src, note="population de 5 ans et plus, redressée")

    occ = reference["population"]["repartition_par_occupation_principale"]
    occ_marge = Marge(
        "occupation", "personne", "nominale",
        tuple(label for label, *_ in OCCUPATIONS),
        {label: float(occ[key]) for label, key, _ in OCCUPATIONS},
        yaml_src, note="population de 5 ans et plus, redressée")

    ht = household_targets()
    motor_menage = Marge(
        "motorisation_menage", "menage", "ordinale", MOTORISATION,
        {MOTORISATION[0]: ht["sans_voiture_pct"], MOTORISATION[1]: ht["une_voiture_pct"],
         MOTORISATION[2]: ht["deux_voitures_plus_pct"]},
        f"{RAPPORT} p. 21, via population_emc2_2023.yaml",
        note="base MÉNAGE (COE0) — chaque persona pèse 1/taille de son ménage")

    shares = couronne_population_shares()
    couronne_marge = Marge(
        "couronne", "personne", "nominale", COURONNES,
        {c: float(shares[c]) for c in COURONNES},
        f"{RAPPORT} p. 10 (habitants de 5 ans et +), via population_emc2_2023.yaml",
        note="hors périmètre exclu du dénominateur, compté à part")

    joint_doc = cible_jointe(joint_path)
    joint = _flat_joint(joint_doc)
    joint_marge = Marge(
        "couronne_x_motorisation", "personne", "nominale", tuple(joint),
        joint, joint_doc["provenance"]["source"],
        note="base PERSONNE (COEP) — recalcul microdonnées gelé, non publié tel quel")
    motor_personne = Marge(
        "motorisation_personne", "personne", "ordinale", MOTORISATION,
        {m: float(joint_doc["marges_pct"]["motorisation"][m]) for m in MOTORISATION},
        joint_doc["provenance"]["source"],
        note="base PERSONNE (COEP) — la base d'un comptage de personas à poids 1")

    # Frozen person margins (cm1): those the report does not publish, or not at this step.
    # They are MEASURABLE — their source is a recomputation on microdata, stated as such —
    # and they are the margins that the v3 selection can close (ticket 029).
    cm = cibles_marges(marges_path)
    src_cm = cm["provenance"]["source"] + " — gelé cm1, non publié à ce pas"
    notes = {
        "age_quinquennal": "le rapport publie 6 classes (p. 11), reprises par `classe_age` ; "
                           "le pas de 5 ans est un recalcul",
        "genre": "le rapport ne publie pas la répartition par sexe",
        "taille_menage_personne": "base PERSONNE : chaque persona porte la taille recensée de son "
                                  "ménage ; la taille moyenne publiée (2,08, p. 10) est en base ménage",
        "permis_adultes": "champ : 18 ans et + ; le mot « permis » n'apparaît qu'une fois dans le "
                          "rapport (p. 4)",
        "abonnement_tc": "le rapport publie 26 % d'abonnés (p. 24) ; recodage P12 de la loi "
                         "d'équipement (ticket 016)",
        "logement": "M1 du ménage porté par chaque persona ; les parts publiées p. 26 sont des "
                    "parts de MÉNAGES",
        "immobile": "personnes sans déplacement la veille ; l'export eqasim les écartait avant le "
                    "ticket 029",
    }
    personne_marges = [
        Marge(nom, "personne", echelle, modalites,
              {m: float(cm["cibles_pct"][nom][m]) for m in modalites}, src_cm, note=notes[nom])
        for nom, (echelle, modalites) in MARGES_PERSONNE.items()
    ]

    return [age_marge, occ_marge, motor_personne, motor_menage, couronne_marge,
            joint_marge, *personne_marges]


# ── Recomputation from the microdata (restricted access) ───────────────────────

def recompute_from_microdata() -> dict:
    """Recomputes the joint target and the unpublished candidates from EMC² (ProGEDO).

    Returns the complete YAML document, ready to freeze. Raises :class:`ReferenceError` if the
    microdata are missing: on a machine without ProGEDO access, nothing is recomputed —
    the frozen target is read.
    """
    if not PROGEDO_STD.exists() or not SIG_DTIR.exists():
        raise ReferenceError(
            f"microdata missing ({PROGEDO_STD}) — restricted access lil-1750. The "
            "check uses the frozen target, it does not need them.")
    import geopandas as gpd
    import pandas as pd

    t0 = time.monotonic()
    per_path = PROGEDO_STD / "Toulouse_2023_std_pers.csv"
    men_path = PROGEDO_STD / "Toulouse_2023_std_men.csv"
    per = pd.read_csv(per_path, dtype=str)
    men = pd.read_csv(men_path, dtype=str)
    per["COEP"] = pd.to_numeric(per["COEP"], errors="coerce")
    men["COE0"] = pd.to_numeric(men["COE0"], errors="coerce")
    men["cars"] = pd.to_numeric(men["M6"], errors="coerce").fillna(0)
    men["motor"] = men["cars"].map(motorisation_class)

    dtir = gpd.read_file(SIG_DTIR)
    couronne_of = dict(zip(dtir["NUM_DTIR"].astype(str), dtir["NOM_D2"]))

    joined = per.merge(men[["ZFM", "ECH", "motor"]].rename(columns={"ZFM": "ZFP"}),
                       on=["ZFP", "ECH"], how="left")
    joined["couronne"] = joined["ZFP"].str[:3].map(couronne_of).map(couronne_canonique)
    unmatched = int(joined["motor"].isna().sum())
    unknown_zone = int(joined["couronne"].isna().sum())
    if unmatched or unknown_zone:
        logger.warning("recomputation: %d person(s) with no matched household, %d with no ring — "
                       "excluded from the joint", unmatched, unknown_zone)
    ok = joined.dropna(subset=["motor", "couronne"])
    ok = ok[ok["couronne"].isin(COURONNES)]

    pivot = ok.pivot_table(index="couronne", columns="motor", values="COEP", aggfunc="sum")
    pivot = pivot.reindex(index=list(COURONNES), columns=list(MOTORISATION)).fillna(0.0)
    mass = float(pivot.values.sum())
    cible = {c: {m: round(100.0 * float(pivot.loc[c, m]) / mass, 3) for m in MOTORISATION}
             for c in COURONNES}
    marge_couronne = {c: round(100.0 * float(pivot.loc[c].sum()) / mass, 2) for c in COURONNES}
    marge_motor = {m: round(100.0 * float(pivot[m].sum()) / mass, 2) for m in MOTORISATION}

    # Counter-check: the same car ownership on a HOUSEHOLD base must match p. 21.
    w = men.groupby("motor")["COE0"].sum()
    menage_pct = {m: round(100.0 * float(w.get(m, 0.0)) / float(w.sum()), 2)
                  for m in MOTORISATION}

    # Unpublished candidates: sex and licence, person base.
    per["age"] = pd.to_numeric(per["P4"], errors="coerce")
    sexe = per.groupby("P2")["COEP"].sum()
    genre_pct = {"Femmes": round(100.0 * float(sexe.get("2", 0.0)) / float(sexe.sum()), 2),
                 "Hommes": round(100.0 * float(sexe.get("1", 0.0)) / float(sexe.sum()), 2)}
    adultes = per[per["age"] >= 18]
    permis = adultes.groupby("P7")["COEP"].sum()
    permis_pct = {"Oui": round(100.0 * float(permis.get("1", 0.0)) / float(permis.sum()), 2),
                  "Non": round(100.0 * float(permis.get("2", 0.0)) / float(permis.sum()), 2),
                  "Conduite accompagnée": round(100.0 * float(permis.get("3", 0.0))
                                                / float(permis.sum()), 2)}

    # Counter-check of the published age classes (p. 11), person base aged 5 and over.
    per["classe_age"] = per["age"].map(age_class)
    ages = per.dropna(subset=["classe_age"]).groupby("classe_age")["COEP"].sum()
    age_pct = {label: round(100.0 * float(ages.get(label, 0.0)) / float(ages.sum()), 2)
               for label, *_ in AGE_CLASSES}

    elapsed = time.monotonic() - t0
    logger.info("recalcul microdonnées terminé en %.1fs — %d personnes, %d ménages, "
                "%d personnes dans le joint", elapsed, len(per), len(men), len(ok))
    return {
        "version": JOINT_VERSION,
        "titre": "Cible jointe couronne × motorisation, base PERSONNE, EMC² Toulouse 2023",
        "unite": "personne (poids COEP), population de 5 ans et plus, périmètre 2023",
        "cible_pct": cible,
        "marges_pct": {"couronne": marge_couronne, "motorisation": marge_motor},
        "contre_epreuves": {
            "motorisation_base_menage_pct": menage_pct,
            "motorisation_publiee_p21_pct": {"sans voiture": 19, "une voiture": 45,
                                            "deux voitures et +": 35},
            "classes_age_recalculees_pct": age_pct,
        },
        "candidats_non_publies": {
            "genre_pct": genre_pct,
            "permis_adultes_pct": permis_pct,
            "note": ("Le rapport ne publie ni le sexe ni le permis. Ces valeurs sont un "
                     "RECALCUL sur microdonnées (COEP), servies comme candidats, jamais "
                     "comme cibles publiées."),
        },
        "provenance": {
            "source": ("recalcul microdonnées EMC² 2023 (ProGEDO lil-1750), fichiers "
                       "personnes × ménages, poids COEP, couronne par secteur de tirage "
                       "(DTIR NOM_D2)"),
            "fichiers": {
                per_path.name: _sha256(per_path),
                men_path.name: _sha256(men_path),
                SIG_DTIR.name: _sha256(SIG_DTIR),
            },
            "n_personnes": int(len(per)),
            "n_menages": int(len(men)),
            "n_personnes_dans_le_joint": int(len(ok)),
            "exclues_sans_menage": unmatched,
            "exclues_sans_couronne": unknown_zone,
            "gele_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "par": "scripts/panel/reference_marges.py --recompute",
        },
    }


def recompute_marges_personne() -> dict:
    """Recomputes the person margins (cm1) from EMC²: INTERVIEWED persons (PENQ = 1),
    COEP weight — the base of the weighting and of the published targets for ages 5 and over.

    Raises :class:`ReferenceError` if the microdata are missing.
    """
    if not PROGEDO_STD.exists():
        raise ReferenceError(
            f"microdata missing ({PROGEDO_STD}) — restricted access lil-1750. The "
            "check uses the frozen targets, it does not need them.")
    import pandas as pd

    t0 = time.monotonic()
    per_path = PROGEDO_STD / "Toulouse_2023_std_pers.csv"
    men_path = PROGEDO_STD / "Toulouse_2023_std_men.csv"
    dep_path = PROGEDO_STD / "Toulouse_2023_std_depl.csv"
    per = pd.read_csv(per_path, dtype=str)
    men = pd.read_csv(men_path, dtype=str)
    dep = pd.read_csv(dep_path, dtype=str)
    per["COEP"] = pd.to_numeric(per["COEP"], errors="coerce")
    per["age"] = pd.to_numeric(per["P4"], errors="coerce")

    # ENUMERATED household size: all its members in the persons file, interviewed or not.
    taille = per.groupby(["ZFP", "ECH"]).size().rename("taille").reset_index()
    menage = (men[["ZFM", "ECH", "M1"]].rename(columns={"ZFM": "ZFP"})
              .merge(taille, on=["ZFP", "ECH"], how="left"))
    ndep = (dep.groupby(["ZFD", "ECH", "PER"]).size().rename("ndep").reset_index()
            .rename(columns={"ZFD": "ZFP"}))
    enq = per[per["PENQ"] == "1"].merge(menage, on=["ZFP", "ECH"], how="left")
    enq = enq.merge(ndep, on=["ZFP", "ECH", "PER"], how="left")
    enq["ndep"] = enq["ndep"].fillna(0)

    def _none_if_na(v):
        return None if v is None or (isinstance(v, float) and v != v) else v

    labels = {
        "age_quinquennal": [age5_class(a) for a in enq["age"]],
        "genre": [{"1": "Hommes", "2": "Femmes"}.get(_none_if_na(s)) for s in enq["P2"]],
        "taille_menage_personne": [taille_menage_class(t) for t in enq["taille"]],
        "permis_adultes": [(None if not (a >= 18) or _none_if_na(p) is None
                            else ("Oui" if p == "1" else "Non"))
                           for a, p in zip(enq["age"], enq["P7"])],
        "abonnement_tc": [(None if _none_if_na(p) is None else ("Oui" if str(p) in _P12_OUI else "Non"))
                          for p in enq["P12"]],
        "logement": [_M1_LABEL.get(str(_none_if_na(m))) if _none_if_na(m) is not None else None
                     for m in enq["M1"]],
        "immobile": ["Oui" if n == 0 else "Non" for n in enq["ndep"]],
    }
    cibles: dict[str, dict[str, float]] = {}
    champs: dict[str, int] = {}
    for nom, (_echelle, modalites) in MARGES_PERSONNE.items():
        s = pd.Series(labels[nom], index=enq.index)
        mask = s.notna()
        w = enq.loc[mask, "COEP"]
        g = w.groupby(s[mask]).sum()
        tot = float(w.sum())
        cibles[nom] = {m: round(100.0 * float(g.get(m, 0.0)) / tot, 3) for m in modalites}
        champs[nom] = int(mask.sum())
        # A recoding that lost categories would show here: the whole mass of the field
        # must fall into the published categories.
        if abs(sum(cibles[nom].values()) - 100.0) > 0.05:
            raise ReferenceError(f"recomputation {nom}: the categories sum to "
                                 f"{sum(cibles[nom].values()):.3f}")

    p12 = enq.groupby("P12")["COEP"].sum()
    p12_pct = {str(k): round(100.0 * float(v) / float(p12.sum()), 2) for k, v in p12.items()}
    mobilite = float((enq["ndep"] * enq["COEP"]).sum() / enq["COEP"].sum())
    elapsed = time.monotonic() - t0
    logger.info("recalcul marges personne terminé en %.1fs — %d interrogées ; immobiles %.1f %%, "
                "abonnés TC %.1f %%, permis adultes %.1f %%", elapsed, len(enq),
                cibles["immobile"]["Oui"], cibles["abonnement_tc"]["Oui"],
                cibles["permis_adultes"]["Oui"])
    return {
        "version": MARGES_VERSION,
        "titre": "Marges PERSONNE de l'EMC² Toulouse 2023 non publiées à ce pas, recalculées",
        "unite": "personne interrogée (PENQ = 1, poids COEP), 5 ans et plus, périmètre 2023",
        "cibles_pct": cibles,
        "champs_effectifs": champs,
        "recodages": {
            "age_quinquennal": "P4, classes de 5 ans, 75 et + regroupés",
            "genre": "P2 : 1 Hommes, 2 Femmes",
            "taille_menage_personne": "membres du ménage dans le fichier personnes (recensés, tous "
                                      "âges), classes 1 / 2 / 3 / 4 / 5 et +",
            "permis_adultes": "P7 = 1 → Oui ; 2 (non), 3 (conduite accompagnée) → Non ; champ 18 ans et +",
            "abonnement_tc": "P12 ∈ {1, 2, 3, 5, 6} → Oui ; 4 → Non (même recodage que la loi "
                             "d'équipement, où le fichier standard porte le oui générique en 6)",
            "logement": "M1 du ménage : 1 isolé, 2 accolé, 3 petit collectif, 4 grand collectif, 5 autres",
            "immobile": "0 déplacement dans le fichier déplacements = Oui",
        },
        "controles": {
            "repartition_P12_pct": p12_pct,
            "abonnes_publies_p24_pct": 26,
            "mobilite_moyenne_deplacements_par_personne": round(mobilite, 3),
            "mobilite_publiee": 3.5,
        },
        "provenance": {
            "source": ("recalcul microdonnées EMC² 2023 (ProGEDO lil-1750), personnes interrogées "
                       "(PENQ = 1) × ménages × déplacements, poids COEP"),
            "fichiers": {per_path.name: _sha256(per_path), men_path.name: _sha256(men_path),
                         dep_path.name: _sha256(dep_path)},
            "n_interrogees": int(len(enq)),
            "gele_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "par": "scripts/panel/reference_marges.py --recompute",
        },
    }


def freeze_marges(doc: dict, path: Path = MARGES_TARGET) -> Path:
    header = (
        "# Marges PERSONNE de l'EMC² Toulouse 2023 — recalculées, GELÉES (cm1).\n"
        "#\n"
        "# Générée par `python -m scripts.panel.reference_marges --recompute` depuis les microdonnées\n"
        "# ProGEDO (accès restreint). Ne pas éditer à la main. Ces marges ne sont PAS publiées par\n"
        "# le rapport AUAT (ou pas à ce pas) : leur source est ce recalcul, et le contrôle le dit.\n"
        "# Ce sont les marges que la sélection v3 (ticket 029) referme par échanges de ménages.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False,
                                            default_flow_style=False), encoding="utf-8")
    return path


def _couronnes_francaises(doc: dict) -> dict:
    """The inverse of :func:`_couronnes_canoniques`, for writing the frozen file.

    The frozen file stays in the language of the survey, whatever that of the
    device: its `sha256` is published, and the cohorts already sealed cite it.
    """
    vers_fr = dict(zip(COURONNES, COURONNES_FR))
    doc = dict(doc)
    if isinstance(doc.get("cible_pct"), dict):
        doc["cible_pct"] = {vers_fr.get(c, c): v for c, v in doc["cible_pct"].items()}
    marges = doc.get("marges_pct")
    if isinstance(marges, dict) and isinstance(marges.get("couronne"), dict):
        marges = dict(marges)
        marges["couronne"] = {vers_fr.get(c, c): v for c, v in marges["couronne"].items()}
        doc["marges_pct"] = marges
    return doc


def freeze(doc: dict, path: Path = JOINT_TARGET) -> Path:
    doc = _couronnes_francaises(doc)
    header = (
        "# Cible JOINTE couronne × motorisation — base PERSONNE — EMC² Toulouse 2023.\n"
        "#\n"
        "# GELÉE : générée par `python -m scripts.panel.reference_marges --recompute` depuis\n"
        "# les microdonnées ProGEDO (accès restreint). Ne pas éditer à la main : douze nombres\n"
        "# recopiés dérivent. Le contrôle de population la lit sans avoir besoin des\n"
        "# microdonnées ; `version` est vérifiée à la lecture.\n"
        "#\n"
        "# POURQUOI ELLE EXISTE. Le rapport publie la motorisation PAR MÉNAGE (p. 21). Une\n"
        "# population synthétique est un échantillon de PERSONNES : un ménage multi-motorisé\n"
        "# de 4 y apparaît 4 fois. Sur base personne, « deux voitures et + » pèse 48,7 % et\n"
        "# non 35 %. Allouer des personas sur la base ménage serait une erreur de base.\n"
        "#\n"
        "# Les contre-épreuves retrouvent la page 21 (19,4 / 45,3 / 35,3 pour 19 / 45 / 35)\n"
        "# et la page 11 (classes d'âge) : même microdonnées, mêmes poids, même périmètre.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False,
                                            default_flow_style=False),
                    encoding="utf-8")
    return path


def print_marges(items: list[Marge]) -> None:
    print("═" * 78)
    print("Reference margins of the population check — EMC² Toulouse 2023")
    print("═" * 78)
    for m in items:
        base = "ménage (1/taille)" if m.unite == "menage" else "personne (poids 1)"
        print(f"\n{m.nom}  [{m.echelle} · base {base} · scope {m.perimetre}]")
        print(f"   source : {m.source_cible}")
        if m.note:
            print(f"   note   : {m.note}")
        if not m.mesurable:
            print("   target : NONE — margin not measurable, returned as is")
            continue
        for modalite in m.modalites:
            print(f"   {modalite:34s} {m.cible_pct[modalite]:6.2f} %")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the joint target from the microdata and freeze it")
    parser.add_argument("--out", type=Path, default=JOINT_TARGET,
                        help="frozen file (default: scripts/panel/cible_jointe_…yaml)")
    parser.add_argument("--json", action="store_true", help="margins as JSON on stdout")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    if args.recompute:
        try:
            doc = recompute_from_microdata()
        except ReferenceError as exc:
            logger.error("[ALARME] %s", exc)
            return 2
        path = freeze(doc, args.out)
        logger.info("joint target frozen → %s", path)
        try:
            doc_m = recompute_marges_personne()
        except ReferenceError as exc:
            logger.error("[ALARME] %s", exc)
            return 2
        path_m = freeze_marges(doc_m, MARGES_TARGET)
        logger.info("person margins frozen → %s", path_m)
        for nom, row in doc_m["cibles_pct"].items():
            print(f"  {nom:24s} " + "  ".join(f"{k} {v:5.2f}" for k, v in row.items())
                  + f"   (field {doc_m['champs_effectifs'][nom]})")
        print("  checks :", doc_m["controles"])
        for c in COURONNES:
            row = doc["cible_pct"][c]
            print(f"  {c:14s} " + "  ".join(f"{m} {row[m]:5.2f} %" for m in MOTORISATION))
        print("  car ownership margin (person) :", doc["marges_pct"]["motorisation"])
        print("  household-base counter-check  :",
              doc["contre_epreuves"]["motorisation_base_menage_pct"], "(published 19/45/35)")
        print("  unpublished candidates        : gender", doc["candidats_non_publies"]["genre_pct"],
              "| adult licence", doc["candidats_non_publies"]["permis_adultes_pct"])

    try:
        items = marges(args.out)
    except ReferenceError as exc:
        logger.error("[ALARME] %s", exc)
        return 2
    if args.json:
        print(json.dumps([m.__dict__ for m in items], ensure_ascii=False, indent=1))
    else:
        print_marges(items)
    logger.info("%d margins assembled, %d measurable, %d without a published target",
                len(items), sum(m.mesurable for m in items),
                sum(not m.mesurable for m in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
