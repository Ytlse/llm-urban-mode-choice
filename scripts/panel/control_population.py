"""control_population.py — The test-set population, checked against the EMC² 2023 survey.

    services/llm-agents/.venv/bin/python -m scripts.panel.control_population data/population/x.json
    services/llm-agents/.venv/bin/python -m scripts.panel.control_population x.json --borne 1.0 \\
        --json rapport.json --trace docs/traces/2026-09-02_population_1000_PANEL

WHAT IT CHECKS (§3.1 of the paper template, §2.3 of the paper plan, milestone 0 of the protocol).
A synthetic population is compared, margin by margin, with the surveyed population: age
classes, occupation, car ownership (person base AND household base), ring of residence, then
the ring × car ownership CROSS — the point that protocol §2.2 names as « what remains
to be tested », because a synthesis by margins fails on the joint without any single margin
moving. The targets come from :mod:`scripts.panel.reference_marges`, each with its
source (report page, or frozen microdata recomputation).

WHAT IT RETURNS, FOR EACH CATEGORY. The observed share, its CI95 (Clopper–Pearson), the
target, the gap in points, the TOST verdict at an indifference bound announced in advance
(`--borne`, ± 1 pt by default), and the count. Per margin: goodness-of-fit χ² — required by the
template, published with its Cramér's V and this warning: on 1,000 individuals the χ² does not
settle a 0.4 pt gap, on 13,000 it rejects everything —, EMD on the ordinal, JSD on the
nominal (the EXACT definitions of the scoring engine, imported, never copied).

THREE RULES THAT ARE NOT NEGOTIABLE.
  * A margin without a published target is `non mesurable`, with its reason. Never a 0, never
    a silence — the vacuity pattern the repository hunts down.
  * A category below the minimum count (`--n-min`) is `non mesurable` too: a CI on
    eight individuals is not a measurement.
  * Personas are independent draws with weight 1: the CI is binomial, with no cluster
    effect. This is an ASSUMPTION, stated here, and false for household attributes
    (car ownership, ring) as long as the population carries no household identifier
    — the extended eqasim export sets it; the per-household bootstrap will come with it.

VERDICTS. `conforme` (TOST equivalent, or gap below the bound without significance);
`à corriger` (significant gap on a margin that the stratified selection can close:
ring, car ownership in BOTH bases since rule v5, joint, age, occupation, frozen person
margins); `à publier` (significant gap on a margin that the selection does not close
— activity chains, which are settled in the eqasim matching); `non mesurable`.
Exit code: 0 if every measured axis is conforming, 1 if there is an `à corriger`, 2 if the
population or the reference is unreadable. `seal_population.py` refuses to seal on 1.

The last section is the GAP SUMMARY: everything that is not conforming, with its
amplitude, its nature, and whether sealing can close it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mobility_core.population_reference import (  # noqa: E402
    COURONNES, MIN_AGE, OUT_OF_PERIMETER, household_targets, household_weight)
from scripts.panel.reference_marges import (  # noqa: E402
    AGE_CLASSES, GENRES, JOINT_TARGET, MARGES_PERSONNE, MARGES_TARGET, MOTORISATION,
    OCCUPATIONS, Marge, ReferenceError, age5_class, age_class, cibles_marges, logement_label,
    marges, motorisation_class, occupation_label, oui_non, taille_menage_class)
from scripts.synthesis.frames import OCCUPATION_MAP  # noqa: E402
from scripts.synthesis.sources import import_formule_score  # noqa: E402

logger = logging.getLogger("panel.control")

CONFORME = "conforme"
A_CORRIGER = "à corriger"
A_PUBLIER = "à publier"
NON_MESURABLE = "non mesurable"

TOST_EQUIVALENT = "équivalent"
TOST_INCONCLUSIF = "non concluant"
TOST_ECART = "écart"

# « Schoolchildren with a study activity » line (ticket 031 § 1.2). Reference: EMC² 2023 microdata,
# 90 to 95 % of mobile enrolled 6-17 year-olds have a trip to school on a weekday;
# the eqasim fork targets ≥ 88 % by restricting the ENTD donor days to school days.
# This is not a selection margin: the descent does not swap on this criterion, it is settled
# in the matching (eqasim). Below the threshold, the gap is « à publier ».
SCOLAIRE_AGE_MIN, SCOLAIRE_AGE_MAX = 6, 17
# Survey identifier, not label: `main_occupation` says « Pupil (up to Baccalaureate) »
# since v6 and « Scolaire (jusqu'au Bac) » before. `OCCUPATION_MAP` knows both;
# a literal would have counted 0 schoolchildren on v6 and made the check « conforme » on an
# empty count.
SCOLAIRE_OCCUPATION_KEY = "scolaire"
SCOLAIRES_ETUDES_REFERENCE_PCT = (90.0, 95.0)
SCOLAIRES_ETUDES_SEUIL_PCT = 88.0

# What the stratified selection (seal_population.py) can close: what is drawn.
# `motorisation_menage` belongs to it since rule v5 (2026-09-04): the selection allocates by
# SUB-CELL (cell × present count × declared size), which fixes exactly the
# `1/size` weight of each category — the margin went from a 3.57 to a 0.06 pt gap. A gap
# established on this margin is thus no longer a fact to publish but a defect to correct, and it
# refuses sealing. As long as nothing allocated it (v2 to v4), it came out « à
# publier »: that was true then, it would be a silence now.
REFERMABLE_AU_SCELLEMENT = {"couronne", "motorisation_personne", "couronne_x_motorisation",
                            "classe_age", "occupation", "motorisation_menage", *MARGES_PERSONNE}
NATURE = {
    "classe_age": "composition (tirage)",
    "occupation": "composition (tirage)",
    "motorisation_personne": "composition (tirage)",
    "motorisation_menage": "base ménage (pondération 1/taille) — allouée par sous-cellule (v5)",
    "couronne": "cadre de tirage (Haute-Garonne : 346 des 453 communes)",
    "couronne_x_motorisation": "croisement (cadre de tirage × motorisation)",
    "age_quinquennal": "composition fine (tirage)",
    "genre": "composition (tirage) — cible recalculée, non publiée",
    "taille_menage_personne": "composition des ménages (tirage)",
    "permis_adultes": "trait imputé (loi ticket 017) sur une composition de strates (tirage)",
    "abonnement_tc": "trait imputé (loi ticket 016) sur une composition de strates (tirage)",
    "logement": "trait imputé (loi ticket 019) sur une composition de strates (tirage)",
    "immobile": "chaînes d'activités (ENTD 2008) et export eqasim",
}

# Rows of table §2.1 of methode/PROTOCOLE_SCIENTIFIQUE.md (v1.5, 2026-09-03, papers repository),
# copied to be CROSS-CHECKED — rule 5 of the papers repository README: published figure =
# cross-checked figure.
# (label, reference margin, category, published value). The values of protocol v1.3
# (51.8 / 19.4 / 62.1 / 18.5 / 22.3 / 46.1 / 31.6 / 84.2) had no source and were
# replaced on 2026-09-03; the cross-check flagged them « à consigner », it now compares
# the corrected table.
PROTOCOLE_2_1 = [
    ("Genre — Femmes", "genre", "Femmes", 51.3),
    ("Genre — Hommes", "genre", "Hommes", 48.7),
    ("Âge — 5-17 ans", "classe_age", "5-17 ans", 16.0),
    ("Âge — 18-24 ans", "classe_age", "18-24 ans", 13.0),
    ("Âge — 25-34 ans", "classe_age", "25-34 ans", 14.0),
    ("Âge — 35-49 ans", "classe_age", "35-49 ans", 22.0),
    ("Âge — 50-64 ans", "classe_age", "50-64 ans", 19.0),
    ("Âge — 65 ans et plus", "classe_age", "65 ans et +", 16.0),
    ("Ménages sans voiture", "motorisation_menage", "sans voiture", 19.0),
    ("Ménages avec 1 voiture", "motorisation_menage", "une voiture", 45.0),
    ("Ménages avec 2 voitures et +", "motorisation_menage", "deux voitures et +", 35.0),
    ("Détention du permis (18 ans et +)", "permis_adultes", "Oui", 85.9),
    ("Personnes sans déplacement la veille", "immobile", "Oui", 10.6),
]


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass
class Constat:
    marge: str
    modalite: str
    n: int
    observe_pct: Optional[float]
    ic95: Optional[tuple[float, float]]
    cible_pct: Optional[float]
    ecart_pt: Optional[float]
    tost: Optional[str]
    verdict: str
    raison: Optional[str] = None


@dataclass
class RapportMarge:
    marge: str
    unite: str
    echelle: str
    source_cible: str
    n: int
    n_eff: float
    deff: float
    chi2: Optional[float]
    ddl: Optional[int]
    p_value: Optional[float]
    cramer_v: Optional[float]
    divergence: Optional[float]
    divergence_type: Optional[str]
    ecart_max_pt: Optional[float]
    verdict: str
    raison: Optional[str]
    constats: list[Constat] = field(default_factory=list)


@dataclass
class Persona:
    id: str
    age: Optional[int]
    classe_age: Optional[str]
    genre: Optional[str]
    occupation: Optional[str]
    taille_menage: Optional[float]
    nb_voitures: Optional[int]
    motorisation: Optional[str]
    couronne: Optional[str]
    permis: Optional[bool]
    poids_menage: float
    age5: Optional[str] = None
    taille_cls: Optional[str] = None
    abonnement: Optional[bool] = None
    logement: Optional[str] = None
    immobile: bool = False
    household_id: Optional[str] = None
    n_activites: int = 0
    scolaire: bool = False          # 6-17 years, occupation « Scolaire (jusqu'au Bac) »
    activite_etudes: bool = False   # at least one `education` activity during the day


# ── Loading and normalisation ─────────────────────────────────────────────────

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_population(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("persons") or raw.get("population") or []
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"empty population or unexpected shape: {path}")
    return raw


_GENRE = {"Female": "Femmes", "Male": "Hommes", "female": "Femmes", "male": "Hommes",
          "F": "Femmes", "M": "Hommes", "Femme": "Femmes", "Homme": "Hommes"}


def normalize(records: list[dict], zones=None) -> tuple[list[Persona], Counter]:
    """One row per persona, in the categories of the reference. Discarded cases are counted."""
    counters: Counter = Counter()
    out: list[Persona] = []
    for rec in records:
        identity = rec.get("identity") or {}
        traits = identity.get("traits_json") or {}
        home = identity.get("home") or {}
        pid = str(rec.get("person_id", ""))

        age_raw = traits.get("age")
        try:
            age = int(float(age_raw)) if age_raw is not None else None
        except (TypeError, ValueError):
            age = None
        if age is None:
            counters["sans_age"] += 1
        elif age < MIN_AGE:
            counters["age_sous_5_ans"] += 1
        classe = age_class(age) if age is not None else None

        genre = _GENRE.get(str(traits.get("gender")))
        if genre is None:
            counters["sans_genre"] += 1

        if OCCUPATION_MAP.get(str(traits.get("main_occupation"))) is None:
            counters["occupation_hors_referentiel"] += 1
        occupation = occupation_label(traits.get("main_occupation"))

        taille = traits.get("household_size")
        try:
            taille = float(taille) if taille is not None else None
        except (TypeError, ValueError):
            taille = None
        if taille is None:
            counters["sans_taille_menage"] += 1
        poids = household_weight(taille)

        motor = motorisation_class(traits.get("number_of_cars"))
        if motor is None:
            counters["sans_nombre_de_voitures"] += 1
        try:
            nb_voitures = int(float(traits.get("number_of_cars")))
        except (TypeError, ValueError):
            nb_voitures = None

        couronne = traits.get("residence_zone")
        if couronne in COURONNES or couronne == OUT_OF_PERIMETER:
            pass
        elif zones is not None and home.get("lat") is not None:
            couronne = zones.classify(home.get("lat"), home.get("lon")) or None
            counters["couronne_par_geometrie"] += 1
        else:
            couronne = None
            counters["sans_couronne"] += 1
        if couronne == OUT_OF_PERIMETER:
            counters["hors_perimetre"] += 1

        permis_raw = traits.get("has_driving_license")
        permis = bool(permis_raw) if permis_raw is not None else None

        abo_raw = traits.get("has_pt_subscription")
        abonnement = bool(abo_raw) if abo_raw is not None else None
        if abonnement is None:
            counters["sans_abonnement_tc"] += 1
        logement = logement_label(traits.get("housing_type"))
        if logement is None:
            counters["sans_logement"] += 1
        activities = identity.get("activities") or []
        # Immobile: root flag set by the eqasim export (ticket 029) or, on an earlier
        # population, a day with a single activity (home).
        immobile = bool(rec.get("immobile")) or len(activities) <= 1
        if immobile:
            counters["immobiles"] += 1
        household_id = (rec.get("household") or {}).get("id")
        if not household_id:
            counters["sans_household_id"] += 1
        # Schoolchildren (ticket 031 § 1.2): 6-17 year-olds declared pupils; « study activity » =
        # an `education` purpose in the day. EMC² 2023 counts 90 to 95 % of them on a weekday.
        scolaire = (age is not None and SCOLAIRE_AGE_MIN <= age <= SCOLAIRE_AGE_MAX
                    and OCCUPATION_MAP.get(str(traits.get("main_occupation") or ""))
                    == SCOLAIRE_OCCUPATION_KEY)
        activite_etudes = any(str(a.get("purpose")) == "education" for a in activities)
        if scolaire:
            counters["scolaires_6_17"] += 1

        out.append(Persona(pid, age, classe, genre, occupation, taille, nb_voitures, motor,
                           couronne, permis, poids,
                           age5=age5_class(age) if age is not None else None,
                           taille_cls=taille_menage_class(taille), abonnement=abonnement,
                           logement=logement, immobile=immobile,
                           household_id=str(household_id) if household_id else None,
                           n_activites=len(activities), scolaire=scolaire,
                           activite_etudes=activite_etudes))
    counters["total"] = len(out)
    return out, counters


def modalite_of(p: Persona, marge: str) -> Optional[str]:
    if marge == "classe_age":
        return p.classe_age
    if marge == "occupation":
        return p.occupation
    if marge in ("motorisation_personne", "motorisation_menage"):
        return p.motorisation
    if marge == "couronne":
        return p.couronne if p.couronne in COURONNES else None
    if marge == "couronne_x_motorisation":
        if p.couronne in COURONNES and p.motorisation:
            return f"{p.couronne} × {p.motorisation}"
        return None
    if marge == "genre":
        return p.genre
    if marge == "permis_adultes":
        if p.age is None or p.age < 18 or p.permis is None:
            return None
        return "Oui" if p.permis else "Non"
    if marge == "age_quinquennal":
        return p.age5
    if marge == "taille_menage_personne":
        return p.taille_cls
    if marge == "abonnement_tc":
        return oui_non(p.abonnement)
    if marge == "logement":
        return p.logement
    if marge == "immobile":
        return "Oui" if p.immobile else "Non"
    return None


def households_and_mobility(personas: list[Persona]) -> dict:
    """Households (by `household.id`) and mobility of the cohort — what the margins do not see.

    The declared size counts children under 5, absent from the population by
    construction: « members present / declared » is therefore a floor, and strict equality
    a severe criterion. Both are published; neither is rounded up.
    """
    present: Counter = Counter()
    declared: dict[str, float] = {}
    for p in personas:
        if p.household_id:
            present[p.household_id] += 1
            if p.taille_menage:
                declared.setdefault(p.household_id, p.taille_menage)
    n_hh = len(present)
    complets = sum(1 for h, n in present.items() if h in declared and n >= int(declared[h]))
    membres_declares = sum(int(declared[h]) for h in present if h in declared)
    membres_presents = sum(present.values())
    # Trips per persona. The activity chain is CYCLIC: step 2 of the notebook
    # merges the evening home and the morning home into a single activity (0 chains starting
    # and ending with `home` in the v3 and v4 seals), and step 4+5 routes n pairs for n
    # activities, the last one being the return to the first. A home → work →
    # home day thus makes 2 activities and 2 trips — not n − 1, which forgot the return and
    # undercounted every mobile persona by one trip (measured on 2026-09-03: 2.44 instead
    # of 3.33 per persona on v4, for 3.53 in the survey). An immobile (one activity): 0.
    trips = [p.n_activites if p.n_activites > 1 and not p.immobile else 0 for p in personas]
    n = len(personas) or 1
    mobiles = [t for t in trips if t > 0]
    scolaires = [p for p in personas if p.scolaire]
    scolaires_mobiles = [p for p in scolaires if not p.immobile and p.n_activites > 1]
    scolaires_etudes = [p for p in scolaires_mobiles if p.activite_etudes]
    part_scolaires = (round(100.0 * len(scolaires_etudes) / len(scolaires_mobiles), 1)
                      if scolaires_mobiles else None)
    return {
        "scolaires_6_17": len(scolaires),
        "scolaires_mobiles": len(scolaires_mobiles),
        "scolaires_avec_activite_etudes": len(scolaires_etudes),
        "part_scolaires_avec_etudes_pct": part_scolaires,
        "part_scolaires_avec_etudes_seuil_pct": SCOLAIRES_ETUDES_SEUIL_PCT,
        "n_menages": n_hh,
        "menages_complets_taille_declaree": complets,
        "part_menages_complets_pct": round(100.0 * complets / n_hh, 1) if n_hh else None,
        "membres_presents": membres_presents,
        "membres_declares": membres_declares,
        "part_membres_presents_pct": (round(100.0 * membres_presents / membres_declares, 1)
                                      if membres_declares else None),
        "sans_household_id": sum(1 for p in personas if not p.household_id),
        "deplacements_par_persona": round(sum(trips) / n, 3),
        "deplacements_par_persona_mobile": round(sum(mobiles) / len(mobiles), 3) if mobiles else None,
        "immobiles": sum(1 for p in personas if p.immobile),
        "part_immobiles_pct": round(100.0 * sum(1 for p in personas if p.immobile) / n, 1),
        "reference_enquete": {"deplacements_par_personne": 3.53, "part_immobiles_pct": 10.6,
                              "deplacements_par_personne_mobile": 3.95,
                              "part_scolaires_avec_etudes_pct": list(SCOLAIRES_ETUDES_REFERENCE_PCT),
                              "source": "microdonnées EMC² 2023, PENQ = 1, COEP ; scolaires : 6-17 ans "
                                        "scolarisés mobiles avec un déplacement vers l'école un jour de semaine"},
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def clopper_pearson(k: float, n: float, alpha: float) -> tuple[float, float]:
    """Exact (1 − alpha) CI of a proportion, in %. `n` may be an effective count."""
    if n <= 0:
        return (0.0, 100.0)
    lo = 0.0 if k <= 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (100.0 * lo, 100.0 * hi)


def tost(observe: float, ic90: tuple[float, float], ic95: tuple[float, float],
         cible: float, borne: float) -> str:
    """Equivalence at ± bound: CI90 of the share within [target − bound, target + bound].

    `écart` when the CI95 excludes the target AND the point gap exceeds the bound — the gap
    is both significant and material. In between: `non concluant`.
    """
    if ic90[0] >= cible - borne and ic90[1] <= cible + borne:
        return TOST_EQUIVALENT
    if (ic95[0] > cible or ic95[1] < cible) and abs(observe - cible) > borne:
        return TOST_ECART
    return TOST_INCONCLUSIF


def _metric_functions():
    """`(emd_1d, jsd)` of the scoring engine — imported, never copied."""
    formule_score, error = import_formule_score()
    if formule_score is None:
        raise ReferenceError(
            "EMD/JSD unavailable: " + error + " The divergences must be those of the "
            "score formula (scripts/synthesis/formule_score/metrics.py) — a second "
            "definition would diverge. No local fallback.")
    return formule_score.metrics.emd_1d, formule_score.metrics.jsd


def measure_marge(marge: Marge, personas: list[Persona], borne: float, n_min: int,
                  emd_1d, jsd) -> RapportMarge:
    weighted = marge.unite == "menage"
    counts: dict[str, float] = defaultdict(float)
    raw_counts: Counter = Counter()
    sum_w = sum_w2 = 0.0
    for p in personas:
        m = modalite_of(p, marge.nom)
        if m is None or m not in marge.modalites:
            continue
        w = p.poids_menage if weighted else 1.0
        if w <= 0:
            continue
        counts[m] += w
        raw_counts[m] += 1
        sum_w += w
        sum_w2 += w * w
    n = sum(raw_counts.values())
    n_eff = (sum_w * sum_w / sum_w2) if weighted and sum_w2 > 0 else float(n)
    deff = (n / n_eff) if n_eff > 0 else 1.0

    if not marge.mesurable:
        return RapportMarge(marge.nom, marge.unite, marge.echelle, marge.source_cible, n,
                            n_eff, deff, None, None, None, None, None, None, None,
                            NON_MESURABLE, "aucune cible publiée — " + marge.note,
                            [Constat(marge.nom, m, raw_counts.get(m, 0),
                                     (100.0 * counts.get(m, 0.0) / sum_w) if sum_w else None,
                                     None, None, None, None, NON_MESURABLE,
                                     "aucune cible publiée") for m in marge.modalites])
    if n == 0:
        return RapportMarge(marge.nom, marge.unite, marge.echelle, marge.source_cible, 0,
                            0.0, 1.0, None, None, None, None, None, None, None,
                            NON_MESURABLE, "aucun persona ne porte cette variable", [])

    observed = np.array([counts.get(m, 0.0) for m in marge.modalites])
    obs_pct = 100.0 * observed / observed.sum()
    target = np.array([marge.cible_pct[m] for m in marge.modalites])
    target_pct = 100.0 * target / target.sum()

    constats: list[Constat] = []
    ecart_max = 0.0
    for i, m in enumerate(marge.modalites):
        k_raw = raw_counts.get(m, 0)
        k_eff = obs_pct[i] / 100.0 * n_eff
        if k_raw < n_min:
            constats.append(Constat(marge.nom, m, k_raw, float(obs_pct[i]), None,
                                    float(target_pct[i]), float(obs_pct[i] - target_pct[i]),
                                    None, NON_MESURABLE,
                                    f"effectif {k_raw} < {n_min} : pas d'IC exploitable"))
            continue
        ic95 = clopper_pearson(k_eff, n_eff, 0.05)
        ic90 = clopper_pearson(k_eff, n_eff, 0.10)
        ecart = float(obs_pct[i] - target_pct[i])
        ecart_max = max(ecart_max, abs(ecart))
        verdict_tost = tost(float(obs_pct[i]), ic90, ic95, float(target_pct[i]), borne)
        if verdict_tost == TOST_ECART:
            verdict = A_CORRIGER if marge.nom in REFERMABLE_AU_SCELLEMENT else A_PUBLIER
        else:
            verdict = CONFORME
        constats.append(Constat(marge.nom, m, k_raw, float(obs_pct[i]), ic95,
                                float(target_pct[i]), ecart, verdict_tost, verdict))

    # Goodness-of-fit χ², corrected for the design (first-order Rao–Scott: χ²/DEFF).
    expected = target_pct / 100.0 * n
    mask = expected > 0
    chi2 = float(np.sum((observed[mask] * (n / observed.sum()) - expected[mask]) ** 2
                        / expected[mask])) / max(deff, 1.0)
    ddl = int(mask.sum() - 1)
    p_value = float(stats.chi2.sf(chi2, ddl)) if ddl > 0 else None
    cramer = math.sqrt(chi2 / (n * ddl)) if ddl > 0 and n > 0 else None

    p_obs = obs_pct / 100.0
    q_ref = target_pct / 100.0
    if marge.echelle == "ordinale":
        divergence, div_type = float(emd_1d(p_obs, q_ref)), "EMD (unités de classe)"
    else:
        divergence, div_type = float(jsd(p_obs, q_ref)), "JSD (base 2, [0, 1])"

    verdicts = [c.verdict for c in constats]
    if A_CORRIGER in verdicts:
        verdict = A_CORRIGER
    elif A_PUBLIER in verdicts:
        verdict = A_PUBLIER
    elif all(v == NON_MESURABLE for v in verdicts):
        verdict = NON_MESURABLE
    else:
        verdict = CONFORME
    raison = None
    if verdict in (A_CORRIGER, A_PUBLIER):
        worst = max((c for c in constats if c.ecart_pt is not None), key=lambda c: abs(c.ecart_pt))
        raison = f"{worst.modalite} : {worst.observe_pct:.1f} % contre {worst.cible_pct:.1f} % ({worst.ecart_pt:+.1f} pt)"
    return RapportMarge(marge.nom, marge.unite, marge.echelle, marge.source_cible, n, n_eff,
                        deff, chi2, ddl, p_value, cramer, divergence, div_type, ecart_max,
                        verdict, raison, constats)


def independence_check(personas: list[Persona], n_min_cellule: int) -> dict:
    """The observed joint against the PRODUCT of its observed margins.

    This is the null of a synthesis by margins: if the draw ignored the ring–car
    ownership dependence, the joint would be the product. A high χ² here says that the
    population does carry the dependence — not that it carries the RIGHT one (cf. the margin
    `couronne_x_motorisation`, which compares it with the survey).
    """
    table = np.zeros((len(COURONNES), len(MOTORISATION)))
    for p in personas:
        if p.couronne in COURONNES and p.motorisation:
            table[COURONNES.index(p.couronne), MOTORISATION.index(p.motorisation)] += 1
    n = table.sum()
    if n == 0:
        return {"n": 0, "verdict": NON_MESURABLE}
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        # A missing ring or car-ownership class makes the table degenerate: the χ² does not exist,
        # and saying so is better than an exception in the middle of the check.
        vides = [m for j, m in enumerate(MOTORISATION) if table[:, j].sum() == 0] + \
                [c for i, c in enumerate(COURONNES) if table[i, :].sum() == 0]
        return {"n": int(n), "verdict": NON_MESURABLE,
                "raison": f"modalité(s) sans aucun persona : {vides} — table de contingence dégénérée"}
    chi2, p, ddl, expected = stats.chi2_contingency(table, correction=False)
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    cells = []
    for i, c in enumerate(COURONNES):
        for j, m in enumerate(MOTORISATION):
            cells.append({"cellule": f"{c} × {m}", "n": int(table[i, j]),
                          "attendu_independance": round(float(expected[i, j]), 1),
                          "mesurable": bool(table[i, j] >= n_min_cellule)})
    return {"n": int(n), "chi2": float(chi2), "ddl": int(ddl), "p_value": float(p),
            "cramer_v": float(v), "cellules": cells,
            "lecture": ("dépendance couronne–motorisation présente dans la population"
                        if p < 0.05 else "couronne et motorisation quasi indépendantes")}


def recoupement(rapports: dict[str, RapportMarge], personas: list[Persona],
                joint_doc: dict) -> list[dict]:
    """Each row of the protocol's table §2.1 against the recomputed reference value.

    The reference is the TARGET of the matching margin (frozen `cm1` reports for gender, licence,
    immobile; six classes published p. 11 for age; household base p. 21 for car ownership),
    never the value observed on the population: we check that the manuscript cites the right
    target, not that it cites the population.
    """
    sources = {
        "genre": "recalcul microdonnées (P2, COEP) — non publié (cm1)",
        "classe_age": "rapport p. 11 (6 classes publiées) — population de 5 ans et +",
        "motorisation_menage": "rapport p. 21 (base ménage)",
        "permis_adultes": "recalcul microdonnées (P7 = 1, 18 ans et +, COEP) — non publié (cm1)",
        "immobile": "recalcul microdonnées (aucun déplacement la veille, COEP) — non publié (cm1)",
    }
    ht = household_targets()
    menage_ref = {"sans voiture": ht["sans_voiture_pct"], "une voiture": ht["une_voiture_pct"],
                  "deux voitures et +": ht["deux_voitures_plus_pct"]}
    cand = joint_doc.get("candidats_non_publies", {})

    def cible(marge: str, modalite: str) -> Optional[float]:
        if marge == "motorisation_menage":
            return menage_ref.get(modalite)
        rapport = rapports.get(marge)
        if rapport is not None:
            for c in rapport.constats:
                if c.modalite == modalite and c.cible_pct is not None:
                    return float(c.cible_pct)
        # Fallback: unpublished candidates of the joint framing (populations checked before cm1).
        key = {"genre": "genre_pct", "permis_adultes": "permis_adultes_pct"}.get(marge)
        return (cand.get(key) or {}).get(modalite) if key else None

    out = []
    for label, kind, modalite, publie in PROTOCOLE_2_1:
        ref = cible(kind, modalite)
        out.append({"ligne": label, "valeur_publiee_protocole": publie,
                    "reference": None if ref is None else round(float(ref), 1),
                    "ecart_pt": None if ref is None else round(float(ref) - publie, 1),
                    "source_reference": sources[kind],
                    "statut": ("aucune référence" if ref is None
                               else "concordant" if round(abs(float(ref) - publie), 1) <= 0.5
                               else "ÉCART — à consigner (Annexe F)")})
    return out


def synthese(rapports: dict[str, RapportMarge], counters: Counter,
             menages: Optional[dict] = None) -> list[dict]:
    rows = []
    if menages:
        if menages["part_membres_presents_pct"] is not None and menages["part_membres_presents_pct"] < 90:
            rows.append({"ecart": "ménages fragmentés (sélection par personne)",
                         "amplitude": f"{menages['menages_complets_taille_declaree']}/{menages['n_menages']} "
                                      f"ménages complets, {menages['part_membres_presents_pct']} % des membres déclarés présents",
                         "nature": "sélection", "verdict": A_PUBLIER,
                         "refermable_au_scellement": "oui — sélection par ménage (v3)"})
        ref = menages["reference_enquete"]
        part_sc = menages.get("part_scolaires_avec_etudes_pct")
        if part_sc is not None and part_sc < SCOLAIRES_ETUDES_SEUIL_PCT:
            lo, hi = SCOLAIRES_ETUDES_REFERENCE_PCT
            rows.append({"ecart": "scolaires sans activité d'études",
                         "amplitude": f"{menages['scolaires_avec_activite_etudes']}/{menages['scolaires_mobiles']} "
                                      f"scolaires (6-17 ans) mobiles avec une activité d'études = {part_sc} % "
                                      f"contre {lo:.0f} à {hi:.0f} % dans l'enquête (seuil {SCOLAIRES_ETUDES_SEUIL_PCT:.0f} %)",
                         "nature": "journées donneuses ENTD 2008 et appariement eqasim (jours de classe, ticket 031 § 1.2)",
                         "verdict": A_PUBLIER,
                         "refermable_au_scellement": "non — appariement HTS (levier eqasim)"})
        if menages["deplacements_par_persona"] < ref["deplacements_par_personne"] - 0.3:
            rows.append({"ecart": "mobilité quotidienne",
                         "amplitude": f"{menages['deplacements_par_persona']:.2f} déplacements par persona "
                                      f"contre {ref['deplacements_par_personne']} dans l'enquête ; "
                                      f"{menages['part_immobiles_pct']} % d'immobiles contre {ref['part_immobiles_pct']} %",
                         "nature": "chaînes d'activités (ENTD 2008 appariée par eqasim)",
                         "verdict": A_PUBLIER,
                         "refermable_au_scellement": "non — enquête d'appariement (levier eqasim)"})
    if counters.get("hors_perimetre"):
        rows.append({"ecart": "domiciles hors des 453 communes",
                     "amplitude": f"{counters['hors_perimetre']} personas "
                                  f"({100.0 * counters['hors_perimetre'] / counters['total']:.1f} %)",
                     "nature": "population", "verdict": A_CORRIGER,
                     "refermable_au_scellement": "oui — exclusion"})
    if counters.get("sans_couronne"):
        rows.append({"ecart": "trait `residence_zone` absent",
                     "amplitude": f"{counters['sans_couronne']} personas",
                     "nature": "post-traitement (étage D non joué)", "verdict": A_CORRIGER,
                     "refermable_au_scellement": "oui — `make residence-zone`"})
    for r in rapports.values():
        if r.verdict in (A_CORRIGER, A_PUBLIER):
            rows.append({"ecart": r.marge, "amplitude": r.raison or f"écart max {r.ecart_max_pt:.1f} pt",
                         "nature": NATURE.get(r.marge, "—"), "verdict": r.verdict,
                         "refermable_au_scellement": "oui — allocation stratifiée"
                         if r.marge in REFERMABLE_AU_SCELLEMENT else "non — à déclarer"})
        elif r.verdict == NON_MESURABLE and r.raison and "aucune cible" in r.raison:
            rows.append({"ecart": r.marge, "amplitude": "—", "nature": NATURE.get(r.marge, "—"),
                         "verdict": NON_MESURABLE, "refermable_au_scellement": "non — cible absente du rapport"})
    return rows


# ── Rendering ─────────────────────────────────────────────────────────────────

def _fmt(v, nd=1, suffix=""):
    return "—" if v is None else f"{v:.{nd}f}{suffix}"


def _scolaires_line(m: dict) -> str:
    """The « schoolchildren with a study activity » line, or its absence, stated."""
    lo, hi = m["reference_enquete"].get("part_scolaires_avec_etudes_pct", SCOLAIRES_ETUDES_REFERENCE_PCT)
    if m.get("scolaires_mobiles"):
        return (f"Scolaires (6-17 ans) avec activité d'études : {m['scolaires_avec_activite_etudes']}/"
                f"{m['scolaires_mobiles']} mobiles = {m['part_scolaires_avec_etudes_pct']} % (enquête "
                f"{lo:.0f} à {hi:.0f} %, seuil {SCOLAIRES_ETUDES_SEUIL_PCT:.0f} %) · scolaires {m['scolaires_6_17']}")
    return (f"Scolaires (6-17 ans) avec activité d'études : non mesurable — {m.get('scolaires_6_17', 0)} "
            f"scolaire(s), aucun mobile")


def render_text(report: dict) -> str:
    L: list[str] = []
    L.append("═" * 78)
    L.append(f"Contrôle de population — {report['population']['path']}")
    L.append(f"{report['population']['n']} personas · sha256 {report['population']['sha256'][:16]}… · "
             f"borne TOST ± {report['parametres']['borne_pt']} pt · n_min {report['parametres']['n_min']}")
    L.append("═" * 78)
    c = report["compteurs"]
    L.append("Cas écartés / comptés : " + ", ".join(f"{k} {v}" for k, v in c.items() if k != "total"))
    m = report.get("menages_et_mobilite")
    if m:
        L.append(f"Ménages : {m['n_menages']} (household.id) · complets au sens de la taille déclarée : "
                 f"{m['menages_complets_taille_declaree']} ({_fmt(m['part_menages_complets_pct'])} %) · "
                 f"membres présents / déclarés : {m['membres_presents']}/{m['membres_declares']} "
                 f"({_fmt(m['part_membres_presents_pct'])} %) · sans household.id : {m['sans_household_id']}")
        ref = m["reference_enquete"]
        L.append(f"Mobilité : {m['deplacements_par_persona']:.2f} déplacements par persona (enquête "
                 f"{ref['deplacements_par_personne']}) · {_fmt(m['deplacements_par_persona_mobile'], 2)} par "
                 f"persona mobile (enquête {ref['deplacements_par_personne_mobile']}) · immobiles "
                 f"{m['part_immobiles_pct']} % (enquête {ref['part_immobiles_pct']} %)")
        L.append(_scolaires_line(m))
    for r in report["marges"]:
        L.append("─" * 78)
        L.append(f"{r['marge']}   [{r['verdict'].upper()}]   base {r['unite']} · {r['echelle']} · n={r['n']}"
                 + (f" · n_eff={r['n_eff']:.0f} (DEFF {r['deff']:.2f})" if r['unite'] == 'menage' else ""))
        L.append(f"   cible : {r['source_cible']}")
        if r["chi2"] is not None:
            L.append(f"   χ² = {r['chi2']:.1f} (ddl {r['ddl']}, p = {r['p_value']:.3g}) · V de Cramér {r['cramer_v']:.3f} · "
                     f"{r['divergence_type']} = {r['divergence']:.3f} · écart max {r['ecart_max_pt']:.2f} pt")
        if r["raison"]:
            L.append(f"   {r['raison']}")
        L.append(f"   {'modalité':30s} {'n':>6s} {'observé':>8s} {'IC95':>17s} {'cible':>7s} {'écart':>7s}  TOST / verdict")
        for k in r["constats"]:
            ic = f"[{k['ic95'][0]:5.1f}, {k['ic95'][1]:5.1f}]" if k["ic95"] else "—"
            L.append(f"   {k['modalite']:30s} {k['n']:6d} {_fmt(k['observe_pct'], 1, ' %'):>8s} {ic:>17s} "
                     f"{_fmt(k['cible_pct'], 1, ' %'):>7s} {_fmt(k['ecart_pt'], 1):>7s}  "
                     f"{(k['tost'] or '—'):13s} {k['verdict']}" + (f"  ({k['raison']})" if k.get("raison") else ""))
    ind = report["independance"]
    L.append("─" * 78)
    L.append("Croisement couronne × motorisation — joint observé contre produit des marges")
    if ind.get("chi2") is not None:
        L.append(f"   χ² = {ind['chi2']:.1f} (ddl {ind['ddl']}, p = {ind['p_value']:.3g}) · V de Cramér {ind['cramer_v']:.3f} · {ind['lecture']}")
        thin = [x["cellule"] for x in ind["cellules"] if not x["mesurable"]]
        if thin:
            L.append(f"   cellules sous l'effectif minimal ({report['parametres']['n_min_cellule']}) : {', '.join(thin)}")
    L.append("─" * 78)
    L.append("Journal de recoupement — tableau §2.1 du protocole contre la référence")
    for row in report["recoupement"]:
        L.append(f"   {row['ligne']:32s} publié {row['valeur_publiee_protocole']:5.1f} % · référence "
                 f"{_fmt(row['reference'], 1, ' %'):>7s} · écart {_fmt(row['ecart_pt'], 1):>5s} pt · {row['statut']}")
        L.append(f"   {'':32s} ↳ {row['source_reference']}")
    L.append("═" * 78)
    L.append("SYNTHÈSE DES ÉCARTS")
    if not report["synthese"]:
        L.append("   aucun écart : toutes les marges mesurées sont conformes")
    for row in report["synthese"]:
        L.append(f"   [{row['verdict'].upper():13s}] {row['ecart']:34s} {row['amplitude']}")
        L.append(f"   {'':15s} nature : {row['nature']} · scellement : {row['refermable_au_scellement']}")
    v = report["verdicts"]
    L.append("═" * 78)
    L.append(f"Verdicts : conforme {v[CONFORME]} · à corriger {v[A_CORRIGER]} · à publier {v[A_PUBLIER]} · non mesurable {v[NON_MESURABLE]}")
    L.append("⚠ Le χ² est donné parce que le gabarit le demande ; il se lit avec son V de Cramér et l'effectif, "
             "jamais seul : sur 1 000 individus il ne tranche pas 0,4 pt, sur 13 000 il rejette tout.")
    return "\n".join(L)


def render_markdown(report: dict) -> str:
    L: list[str] = []
    p = report["population"]
    L.append(f"# Contrôle de population — `{Path(p['path']).name}`")
    L.append("")
    L.append(f"- **Effectif** : {p['n']} personas — sha256 `{p['sha256']}`")
    L.append(f"- **Date** : {report['date']}")
    L.append(f"- **Borne TOST** : ± {report['parametres']['borne_pt']} pt · n_min {report['parametres']['n_min']} · "
             f"n_min cellule {report['parametres']['n_min_cellule']}")
    L.append(f"- **Verdicts** : " + " · ".join(f"{k} {v}" for k, v in report["verdicts"].items()))
    m = report.get("menages_et_mobilite")
    if m:
        ref = m["reference_enquete"]
        L.append(f"- **Ménages** : {m['n_menages']} ; complets (taille déclarée) {m['menages_complets_taille_declaree']} "
                 f"({_fmt(m['part_menages_complets_pct'])} %) ; membres présents / déclarés "
                 f"{m['membres_presents']}/{m['membres_declares']} ({_fmt(m['part_membres_presents_pct'])} %)")
        L.append(f"- **Mobilité** : {m['deplacements_par_persona']:.2f} déplacements par persona (enquête "
                 f"{ref['deplacements_par_personne']}) ; immobiles {m['part_immobiles_pct']} % "
                 f"(enquête {ref['part_immobiles_pct']} %)")
        L.append("- **" + _scolaires_line(m).replace(" : ", "** : ", 1))
    L.append("")
    L.append("## Marges")
    for r in report["marges"]:
        L.append("")
        L.append(f"### `{r['marge']}` — **{r['verdict']}**")
        L.append("")
        L.append(f"Base {r['unite']}, {r['echelle']}, n = {r['n']}. Cible : {r['source_cible']}.")
        if r["chi2"] is not None:
            L.append(f"χ² = {r['chi2']:.1f} (ddl {r['ddl']}, p = {r['p_value']:.3g}), V de Cramér {r['cramer_v']:.3f}, "
                     f"{r['divergence_type']} = {r['divergence']:.3f}, écart max {r['ecart_max_pt']:.2f} pt.")
        if r["raison"]:
            L.append("")
            L.append(f"> {r['raison']}")
        L.append("")
        L.append("| Modalité | n | Observé | IC95 | Cible | Écart | TOST | Verdict |")
        L.append("|---|---:|---:|---|---:|---:|---|---|")
        for k in r["constats"]:
            ic = f"[{k['ic95'][0]:.1f}, {k['ic95'][1]:.1f}]" if k["ic95"] else "—"
            L.append(f"| {k['modalite']} | {k['n']} | {_fmt(k['observe_pct'], 1, ' %')} | {ic} | "
                     f"{_fmt(k['cible_pct'], 1, ' %')} | {_fmt(k['ecart_pt'], 1)} | {k['tost'] or '—'} | "
                     f"{k['verdict']}" + (f" ({k['raison']})" if k.get('raison') else "") + " |")
    L.append("")
    L.append("## Journal de recoupement (protocole §2.1)")
    L.append("")
    L.append("| Ligne | Publié | Référence | Écart | Statut | Source |")
    L.append("|---|---:|---:|---:|---|---|")
    for row in report["recoupement"]:
        L.append(f"| {row['ligne']} | {row['valeur_publiee_protocole']:.1f} % | {_fmt(row['reference'], 1, ' %')} | "
                 f"{_fmt(row['ecart_pt'], 1)} | {row['statut']} | {row['source_reference']} |")
    L.append("")
    L.append("## Synthèse des écarts")
    L.append("")
    L.append("| Écart | Amplitude | Nature | Verdict | Refermable au scellement |")
    L.append("|---|---|---|---|---|")
    for row in report["synthese"]:
        L.append(f"| {row['ecart']} | {row['amplitude']} | {row['nature']} | {row['verdict']} | {row['refermable_au_scellement']} |")
    if not report["synthese"]:
        L.append("| — | — | — | conforme | — |")
    return "\n".join(L) + "\n"


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_control(population_path: Path, borne: float, n_min: int, n_min_cellule: int) -> dict:
    t0 = time.monotonic()
    logger.info("check — start: %s", population_path)
    records = load_population(population_path)
    digest = sha256_of(population_path)
    logger.info("population read: %d records, sha256 %s…", len(records), digest[:16])

    zones = None
    try:
        from mobility_core.residence_zone import CommunalZones
        zones = CommunalZones.load()
    except Exception as exc:  # missing resource: the trait is authoritative, geometry is a backup
        logger.warning("géométrie des couronnes indisponible (%s) — le trait `residence_zone` seul fait foi", exc)

    personas, counters = normalize(records, zones)
    logger.info("normalisation: %s", dict(counters))
    if counters.get("age_sous_5_ans"):
        logger.error("[ALARME] %d persona(s) under %d years old — outside the surveyed population",
                     counters["age_sous_5_ans"], MIN_AGE)

    emd_1d, jsd = _metric_functions()
    items = marges(JOINT_TARGET)
    import yaml
    joint_doc = yaml.safe_load(JOINT_TARGET.read_text(encoding="utf-8"))

    rapports: dict[str, RapportMarge] = {}
    for m in items:
        t1 = time.monotonic()
        rapports[m.nom] = measure_marge(m, personas, borne, n_min, emd_1d, jsd)
        r = rapports[m.nom]
        logger.info("margin %-24s %-13s n=%d max gap %s in %.2fs", m.nom, r.verdict, r.n,
                    _fmt(r.ecart_max_pt, 2, " pt"), time.monotonic() - t1)
        if r.verdict == A_CORRIGER:
            logger.error("[ALARME] margin %s to be corrected — %s", m.nom, r.raison)

    ind = independence_check(personas, n_min_cellule)
    recoup = recoupement(rapports, personas, joint_doc)
    gaps = [row for row in recoup if row["statut"].startswith("ÉCART")]
    if gaps:
        worst = max(gaps, key=lambda r: abs(r["ecart_pt"]))
        logger.error("[ALARME] cross-check: %d row(s) of the protocol's table §2.1 deviate from "
                     "the reference — the worst, « %s », published %.1f %% against %.1f %% (%+.1f pt). "
                     "To be recorded in Appendix F, never corrected silently.",
                     len(gaps), worst["ligne"], worst["valeur_publiee_protocole"],
                     worst["reference"], worst["ecart_pt"])
    menages = households_and_mobility(personas)
    logger.info("households: %d, complete %s; mobility %.2f trips/persona, %.1f %% immobile",
                menages["n_menages"], menages["menages_complets_taille_declaree"],
                menages["deplacements_par_persona"], menages["part_immobiles_pct"])
    synth = synthese(rapports, counters, menages)
    verdicts = Counter(r.verdict for r in rapports.values())
    for k in (CONFORME, A_CORRIGER, A_PUBLIER, NON_MESURABLE):
        verdicts.setdefault(k, 0)

    report = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "population": {"path": str(population_path), "n": len(records), "sha256": digest},
        "parametres": {"borne_pt": borne, "n_min": n_min, "n_min_cellule": n_min_cellule,
                       "ic": "Clopper–Pearson, personas indépendants à poids 1 ; base ménage : n_eff de Kish",
                       "cible_jointe": {"fichier": str(JOINT_TARGET), "version": joint_doc.get("version"),
                                        "sha256": sha256_of(JOINT_TARGET)},
                       "cibles_marges": {"fichier": str(MARGES_TARGET),
                                         "version": cibles_marges(MARGES_TARGET).get("version"),
                                         "sha256": sha256_of(MARGES_TARGET)}},
        "compteurs": dict(counters),
        "menages_et_mobilite": menages,
        "marges": [{**asdict(r), "constats": [asdict(c) for c in r.constats]} for r in rapports.values()],
        "independance": ind,
        "recoupement": recoup,
        "synthese": synth,
        "verdicts": dict(verdicts),
        "duree_s": round(time.monotonic() - t0, 2),
    }
    logger.info("check — end after %.1fs: %s", report["duree_s"], dict(verdicts))
    return report


def write_trace(report: dict, trace_dir: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (trace_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (trace_dir / "empreintes.txt").write_text(
        f"population  {report['population']['sha256']}  {report['population']['path']}\n"
        f"cible_jointe  {report['parametres']['cible_jointe']['sha256']}  {report['parametres']['cible_jointe']['fichier']}\n",
        encoding="utf-8")
    readme = trace_dir / "README.md"
    if not readme.exists():
        v = report["verdicts"]
        readme.write_text(
            f"# Contrôle de population — {Path(report['population']['path']).name}\n\n"
            f"Produit le {report['date']} par `scripts/panel/control_population.py`.\n\n"
            f"- `report.md` — le rapport lisible (marges, recoupement, synthèse des écarts)\n"
            f"- `report.json` — le même, structuré\n"
            f"- `empreintes.txt` — sha256 de la population et de la cible jointe\n\n"
            f"Verdicts : conforme {v[CONFORME]} · à corriger {v[A_CORRIGER]} · à publier {v[A_PUBLIER]} · "
            f"non mesurable {v[NON_MESURABLE]}.\n", encoding="utf-8")
    logger.info("trace archived → %s", trace_dir)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("population", type=Path)
    parser.add_argument("--borne", type=float, default=1.0,
                        help="indifference bound of the TOST, in %% points (default 1.0)")
    parser.add_argument("--n-min", type=int, default=30,
                        help="minimum count of a category for it to be measured (default 30)")
    parser.add_argument("--n-min-cellule", type=int, default=50,
                        help="minimum count of a cross-tabulation cell (default 50)")
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--trace", type=Path, default=None,
                        help="trace directory (report.json, report.md, empreintes.txt)")
    parser.add_argument("--trace-auto", action="store_true",
                        help="trace in docs/traces/<YYYY-MM-DD_HH-MM>_controle_<population>/ — "
                             "timestamped, so two passes on the same day never overwrite each other")
    parser.add_argument("--quiet", action="store_true", help="no text report on stdout")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    if not args.population.exists():
        logger.error("[ALARME] population introuvable : %s", args.population)
        return 2
    try:
        report = run_control(args.population, args.borne, args.n_min, args.n_min_cellule)
    except (ReferenceError, ValueError, OSError) as exc:
        logger.error("[ALARME] check impossible: %s", exc)
        return 2

    if not args.quiet:
        print(render_text(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    trace_dir = args.trace
    if trace_dir is None and args.trace_auto:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        trace_dir = REPO_ROOT / "docs" / "traces" / f"{stamp}_controle_{args.population.stem}"
    if trace_dir:
        write_trace(report, trace_dir)
    return 1 if report["verdicts"][A_CORRIGER] else 0


if __name__ == "__main__":
    raise SystemExit(main())
