"""seal_population.py — Select exactly 1,000 personas, by HOUSEHOLDS, from a pool; then seal.

    # 1. stratified selection in the generated and pre-imputed pool (before the notebook routing)
    services/llm-agents/.venv/bin/python -m scripts.panel.seal_population select \\
        --pool scripts/data/population/Temp/4_zone_enriched/toulouse_population_10000.json \\
        --n 1000 --out scripts/data/population/Temp/4_zone_enriched/toulouse_population_1000_PANEL.json

    # 2. sealing of the final file, AFTER post-processing and check
    services/llm-agents/.venv/bin/python -m scripts.panel.seal_population seal \\
        --population data/population/toulouse_population_1000_PANEL.json \\
        --out-dir data/population/population_1000_PANEL_v5

WHY A SELECTION. The eqasim service draws `population_size × 1.15` persons and names
the file after the REQUESTED size: `toulouse_population_1000.json` holds 1,021. A round
count therefore cannot be set at generation. And a random selection wastes
precision: the sizing note (§ 4.3.1) asks for a STRATIFIED draw on the very strata
that will serve for validation — « 1 000 agents stratifiés valent ≈ 2 000 tirés au hasard ».

THE v5 RULE (`panel_seal_v5`, ticket 031). Same mechanics as v4, but the allocation works
on ONE MORE DIMENSION — the sub-cell (cell × present size × declared size) instead
of the cell alone. The hash salt stays `panel_seal_v4:`: only the stratum changes, not
the order in which households come up. The `cj1` / `cm1` targets do not change: they
are computed on the 453 communes. The SCOPE log (453 communes of the EMC² 2023,
commune polygon, table `commune_couronne.json` cc1; residence departments of the kept persons
read from `household.commune_id`) is that of v4. In four steps:

1. **The unit is the household** (`household.id`, at the root of the records since the widened
   export). v2 selected persons: 1,000 kept persons came from 865 households of which 308
   complete. A household has ONE ring and ONE car ownership (household attributes: 0 mixed
   household measured out of 2,791), hence one cell; its members aged 5 and over are all in the
   pool since the export keeps the immobile — the only ones missing are children under
   5, outside the surveyed population. A population without `household.id` (older than the
   widened export) is handled as one-person households, and the log says so.

2. **Cell counts**: the 12 ring × car ownership cells of the joint target on a
   person basis (`cible_jointe_couronne_motorisation.yaml`), counts in persons by largest
   remainder. A cell the pool does not fill is a DEFICIT: carried over first within the
   same ring, then to the whole pool, logged, alarmed, and exit code 1.

3. **Counts per SUB-CELL, by integer program** (HiGHS via `scipy.optimize.milp`). The
   twelve cell counts in persons are equalities; household size on a person
   basis and car ownership on a HOUSEHOLD basis (weight `1/size`, the one of the check) are bounded
   in maximal gap, at the finest tolerance the pool allows (bisection); among the
   allocations that hold it, the objective keeps the one that moves the fewest households
   relative to the pool composition. v4 allocated by cell only: car ownership on a
   household basis was its only « à publier » gap (22.8 % of households without a car versus 19.2 %
   in the report p. 21) because nothing targeted it — and putting it in the loss of the
   descent alone is not enough, this is measured. A solver failure is an explicit error; there
   is no fallback to the previous allocation. Households then enter in the order of
   `sha256("panel_seal_v4:" + household_id)` WITHIN their sub-cell.

4. **Descent on multiple margins**: as long as a swap of two households of the SAME SUB-CELL
   — one kept, the other not — reduces the loss, it is applied. The loss is the sum, over
   all checked margins (occupation and six age classes published p. 11; five-year
   age, gender, household size, licence, PT pass, housing, immobile: frozen
   recomputations `cm1`), of the absolute gaps in points between observed share and target. Order of
   traversal and of candidacy = hash: deterministic, replayable. A swap at constant
   sub-cell moves neither the twelve cell counts nor the two allocated margins: the
   descent only works on what the allocation does not fix. Imputed traits must
   be set ON THE POOL before selection (step 3ter-a of the notebook), otherwise the margin is
   empty and the descent ignores it — and says so.

Since the kept composition matches the targets, the cohort stays SELF-WEIGHTED: each persona
keeps its weight 1, no design weighting to propagate into the score.

SEALING REFUSES. `seal` replays the check (`control_population.py`) on the final
file; an `à corriger` verdict forbids sealing — nothing is written, the candidate
file stays in place, the report says what. A sealed folder holds the file, its
sha256, that of the pool, the selection rule, the deficits, the descent log, the
check report and the git revision of the repository. It is never modified: any correction
produces a NEW folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mobility_core.population_reference import COURONNES, MIN_AGE, OUT_OF_PERIMETER  # noqa: E402
from scripts.panel import control_population as ctl  # noqa: E402
from scripts.panel.reference_marges import (  # noqa: E402
    JOINT_TARGET, MARGES_PERSONNE, MARGES_TARGET, MOTORISATION, Marge, ReferenceError,
    cible_jointe, marges, motorisation_class)

logger = logging.getLogger("panel.seal")

# The hash salt of the HOUSEHOLDS stays that of v4, DELIBERATELY: the v5 rule only changes
# what is targeted (the allocation sub-cell, the loss function of the descent), not
# the order in which households come up. Keeping the same order makes the effect of the
# new allocation measurable in isolation — a new salt would mix two causes, and the compliance
# table would no longer say which one acted. v3 → v4 had changed the salt because the sampling
# frame changed; here it does not.
SELECTION_NAMESPACE = "panel_seal_v4"   # hash salt of the HOUSEHOLDS — unchanged, see above
SELECTION_RULE = "panel_seal_v5"
# A selection on a TRIMMED pool (ticket 073, axis 2 — spec `cohortes-disjointes-axe2`) does not
# carry the same rule name as a selection on the whole pool, even if the mechanics are identical
# to the character: two cohorts drawn from different pools are not drawn in the
# same way, and a shared label would suggest they are. The SALT does not change — it is
# the exclusion that produces a different cohort, and that is enough; changing both would make
# it impossible to say which one produced the observed gap.
SELECTION_RULE_DISJOINT = "panel_seal_v5_disjoint"
SEAL_VERSION = "sceau1"
# The CURRENT cohort. The selection rule stays `panel_seal_v5`: v6 is the
# same cohort in another language (ticket 074), not a new way of drawing it —
# bumping the rule would have suggested that the draw had changed.
DEFAULT_SEAL_DIR = REPO_ROOT / "data" / "population" / "population_1000_PANEL_v6"

# Scope of the population (ticket 031, option A): the 453 communes of the EMC² 2023, six
# departments, delimited by the POLYGON OF THE COMMUNES (table `commune_couronne.json`), not by
# a radius. The selection excludes homes outside these communes; the log says how many
# kept persons come from each department, so that a trimmed sampling frame (Haute-Garonne
# alone, ticket 026) shows in the seal instead of hiding in it.
PERIMETRE = {
    "definition": "453 communes de l'enquête EMC² Toulouse 2023, six départements "
                  "(31, 32, 81, 82, 09, 11), polygone communal — pas de rayon",
    "table_communes": "packages/mobility_core/src/mobility_core/data/commune_couronne.json",
    "departements_attendus": {"31": 346, "32": 38, "81": 27, "82": 22, "09": 10, "11": 10},
}

# Margins of the descent: occupation and the SIX age classes published by the report (p. 11),
# plus the frozen person margins (cm1). The six published classes AND the fifteen five-year ones:
# the 15-19 class straddles the 17/18 boundary, and holding the fifteen does not hold the share of
# 5-17 year-olds (measured on v3: +1.2 pt on the 5-17 with 57 % of 15-17 in the 15-19 versus
# 45 % in the survey). The reference data of the article is the AUAT report: its classes are
# held first. The ring × car ownership cell is not among them: it is held
# exactly by the allocation.
# Car ownership on a HOUSEHOLD BASIS is in the descent margins since v5, but it is
# the ALLOCATION that holds it (see ALLOCATION_MARGES): under the v5 swap operator it
# is CONSTANT, and its presence here serves the log and the invariant checked after the descent.
DESCENTE_MARGES: tuple[str, ...] = ("occupation", "classe_age", "motorisation_menage",
                                    *MARGES_PERSONNE)

# Margins counted on a HOUSEHOLD basis: each persona weighs `1 / declared size of its household`,
# as in the check (`mobility_core.population_reference.household_weight`). Counting these
# margins at weight 1 would compare a population of persons with a target of households — the
# basis error that the check page explicitly forbids.
DESCENTE_MARGES_MENAGE: frozenset[str] = frozenset({"motorisation_menage"})

# Candidates examined per kept household and per pass (hash order). Bounds the cost without
# changing determinism; 150 is ample on a pool of 10,000.
DESCENTE_CANDIDATS = 150
DESCENTE_PASSES_MAX = 40

# ── The two margins the ALLOCATION holds (v5) ─────────────────────────────────
#
# v4 left an « à publier » gap: households without a car 22.8 % versus 19.2 % (report
# p. 21). Nothing targeted it — the allocation held car ownership on a PERSON basis (exact,
# 13.6 %) and the two bases do not say the same thing: on a household basis a persona weighs
# the inverse of the DECLARED size of its household, a single person without a car weighs 1 and a
# member of a household of four weighs 0.25.
#
# Putting the margin in the loss of the descent is not enough, and this is measured (2026-09-04):
# the operator pairs households by PRESENT size, so the only lever is a
# household whose declared size differs from the present size — the absent children under 5,
# 52 members out of 1,052. Weighting the margin from 1 to 50 in the loss, the gap only
# drops from 3.4 to 2.2 pt, and household size worsens from 0.9 to 3.0 pt.
#
# v5 therefore allocates on ONE MORE DIMENSION: the SUB-CELL is the triplet
# (cell, present size, declared size), and not the cell alone. This triplet determines
# exactly the two margins of household basis / size:
#   * persons of a size class = Σ n(c,S,T) × S over the T of the class;
#   * household weight of a car ownership category = Σ n(c,S,T) × S/T over the cells of that
#     category (the weight `1/size` summed over the S present members).
# The swap operator now pairs at constant SUB-CELL: the two margins — and the
# twelve cell counts in persons — are preserved BY CONSTRUCTION by the descent,
# which keeps all its freedom on the others (occupation, age, gender, licence, pass,
# housing, immobile).
ALLOCATION_MARGES: tuple[str, ...] = ("taille_menage_personne", "motorisation_menage")

# The allocation seeks the smallest MAXIMAL gap (∞ norm, the very criterion of the check)
# reachable on these two margins, by bisection on the tolerance: one feasibility integer
# program per trial, feasibility being increasing with tolerance. The 0.01 pt step
# is one hundredth of the indifference bound of the check. The upper bound is always feasible
# (no share gap exceeds 100 pt) as soon as the twelve cell equalities are: the
# bisection therefore always returns an allocation, and the log says at which tolerance.
ALLOCATION_TOLERANCE_PAS_PT = 0.01
ALLOCATION_TOLERANCE_MAX_PT = 100.0
# Beyond this tolerance, the pool cannot hold the margin: it is no longer a
# setting, it is a gap to declare. It is alarmed and we exit with code 1 — never a silent rounding.
ALLOCATION_TOLERANCE_ALARME_PT = 1.0
ALLOCATION_SOLVEUR_TIMEOUT_S = 300.0


# ── Utilities ─────────────────────────────────────────────────────────────────

def _rank(key: str) -> str:
    return hashlib.sha256(f"{SELECTION_NAMESPACE}:{key}".encode("utf-8")).hexdigest()


def largest_remainder(shares_pct: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder rounding: the counts sum EXACTLY to `n`."""
    total = sum(shares_pct.values())
    exact = {k: n * v / total for k, v in shares_pct.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(floors.values())
    for k in sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)[:remainder]:
        floors[k] += 1
    return floors


def _traits(rec: dict) -> dict:
    return (rec.get("identity") or {}).get("traits_json") or {}


def ensure_residence_zone(records: list[dict]) -> Counter:
    """Sets `residence_zone` on the personas that do not have it yet (stage D, ticket 021)."""
    missing = [r for r in records
               if _traits(r).get("residence_zone") not in (*COURONNES, OUT_OF_PERIMETER)]
    counts: Counter = Counter(deja_pose=len(records) - len(missing))
    if not missing:
        return counts
    from mobility_core.residence_zone import CouronneTable
    from mobility_core.zone_resolver import ZoneResolver
    from scripts.data.population.enrich_residence_zone import enrich

    feature_spec = REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"
    table = CouronneTable.load()
    resolver = ZoneResolver.load(None, feature_spec if feature_spec.exists() else None)
    t0 = time.monotonic()
    posed = enrich(missing, table, resolver)
    counts.update({f"pose_{k}": v for k, v in posed.items()})
    logger.info("residence_zone set on %d personas in %.1fs: %s", len(missing),
                time.monotonic() - t0, posed)
    return counts


# ── Households ────────────────────────────────────────────────────────────────

@dataclass
class Menage:
    id: str
    cellule: str
    membres: list[dict]
    taille_declaree: int
    rank: str = ""

    @property
    def size(self) -> int:
        return len(self.membres)


def group_households(records: list[dict]) -> tuple[list[Menage], Counter]:
    """Groups the eligible personas by household. Discarded cases are counted, never hidden."""
    excluded: Counter = Counter()
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        tr = _traits(rec)
        try:
            age = int(float(tr.get("age")))
        except (TypeError, ValueError):
            excluded["sans_age"] += 1
            continue
        if age < MIN_AGE:
            excluded["moins_de_5_ans"] += 1
            continue
        couronne = tr.get("residence_zone")
        if couronne == OUT_OF_PERIMETER:
            excluded["hors_perimetre"] += 1
            continue
        if couronne not in COURONNES:
            excluded["sans_couronne"] += 1
            continue
        if motorisation_class(tr.get("number_of_cars")) is None:
            excluded["sans_motorisation"] += 1
            continue
        hid = (rec.get("household") or {}).get("id")
        if not hid:
            excluded["sans_household_id_menage_d_une_personne"] += 1
            hid = f"p:{rec.get('person_id')}"
        groups[str(hid)].append(rec)

    menages: list[Menage] = []
    for hid, membres in groups.items():
        cells = {f"{_traits(m)['residence_zone']} × {motorisation_class(_traits(m)['number_of_cars'])}"
                 for m in membres}
        if len(cells) != 1:
            excluded["menage_mixte"] += len(membres)
            continue
        try:
            declared = int(float(_traits(membres[0]).get("household_size")))
        except (TypeError, ValueError):
            declared = len(membres)
        menages.append(Menage(hid, cells.pop(), membres, declared, _rank(hid)))
    menages.sort(key=lambda m: m.rank)
    return menages, excluded


def _commune_of(rec: dict) -> Optional[str]:
    """INSEE commune of the home: `household.commune_id` (eqasim export), else the trait."""
    hh = rec.get("household") or {}
    code = hh.get("commune_id")
    if code is None or str(code) in ("", "undefined", "None"):
        code = _traits(rec).get("residence_insee")
    return str(code).zfill(5) if code not in (None, "") else None


def count_removed_out_of_perimeter(records: list[dict]) -> dict:
    """Activities outside the polygon removed at step 2 of the notebook (`perimetre` at the root).

    `controle: False` when no record carries the key: the population was produced
    before the safeguard, and "0" would then be an invention."""
    total, touches, controles = 0, 0, 0
    for rec in records:
        per = rec.get("perimetre") or {}
        if "activites_hors_perimetre_supprimees" not in per:
            continue
        controles += 1
        k = int(per["activites_hors_perimetre_supprimees"] or 0)
        total += k
        touches += 1 if k else 0
    return {"controle": controles == len(records) and bool(records),
            "personas_controles": controles, "activites_hors_perimetre_supprimees": total,
            "personas_touches": touches}


def perimeter_journal(retenus: list[dict]) -> dict:
    """The declared scope, the residence departments of the kept persons (`household.commune_id`)
    and the activities outside the polygon removed from their chains."""
    by_dep: Counter = Counter()
    sans_commune = 0
    for rec in retenus:
        code = _commune_of(rec)
        if code is None:
            sans_commune += 1
            continue
        by_dep[code[:2]] += 1
    return {
        **PERIMETRE,
        "retenus_par_departement": dict(sorted(by_dep.items())),
        "departements_representes": len(by_dep),
        "retenus_sans_commune": sans_commune,
        "communes_distinctes": len({c for c in (_commune_of(r) for r in retenus) if c}),
        "activites_hors_perimetre": count_removed_out_of_perimeter(retenus),
    }


# ── Allocation by sub-cells (cell × present size × declared size) ─────────────

SousCellule = tuple[str, int, int]


class AllocationError(ValueError):
    """The allocation has no integer solution. Never a fallback to the previous allocation."""


def sous_cellule(m: Menage) -> SousCellule:
    """The v5 allocation stratum: cell, PRESENT size, DECLARED size.

    It is also the pairing key of the swap operator: what the allocation fixes, the
    descent preserves by construction.
    """
    return (m.cellule, m.size, m.taille_declaree)


def _libelle(key: SousCellule) -> str:
    return f"{key[0]} | présents {key[1]} | déclarés {key[2]}"


def inventaire(menages: list[Menage]) -> dict[SousCellule, int]:
    """Pool households by sub-cell — the inventory the allocation cannot go beyond."""
    inv: Counter = Counter(sous_cellule(m) for m in menages)
    return dict(sorted(inv.items()))


def cibles_allocation(joint_path: Path = JOINT_TARGET,
                      marges_path: Path = MARGES_TARGET) -> dict[str, dict[str, float]]:
    """The targets of the two margins held by the allocation, in %, renormalised to 100.

    They are READ from the frozen targets (`cj1`, `cm1`) like those of the descent — no
    literal copied here, otherwise a target that moves would not move the selection.
    """
    out: dict[str, dict[str, float]] = {}
    for m in marges(joint_path, marges_path):
        if m.nom not in ALLOCATION_MARGES:
            continue
        if not m.mesurable:
            raise AllocationError(
                f"the margin « {m.nom} » is allocated by v5 but has no target: "
                f"{m.source_cible}. No allocation on a missing target.")
        total = sum(m.cible_pct.values())
        out[m.nom] = {k: 100.0 * v / total for k, v in m.cible_pct.items()}
    manquantes = [nom for nom in ALLOCATION_MARGES if nom not in out]
    if manquantes:
        raise AllocationError(f"allocation margins missing from the references: {manquantes}")
    return out


@dataclass
class _Contexte:
    """The arrays of the integer program, in the sorted order of the inventory (determinism)."""
    keys: list[SousCellule]
    dispo: "np.ndarray"          # households available per sub-cell
    presents: "np.ndarray"       # present size S
    declares: "np.ndarray"       # declared size T
    cellule: list[str]
    motorisation: list[str]
    classe_taille: list[Optional[str]]
    effectif: dict[str, int]     # target count in PERSONS per cell (after carry-overs)
    n: int
    cibles: dict[str, dict[str, float]]
    reference: "np.ndarray"      # allocation proportional to the pool, rounded


def _contexte(inv: dict[SousCellule, int], effectif: dict[str, int], n: int,
              cibles: dict[str, dict[str, float]]) -> _Contexte:
    from scripts.panel.reference_marges import taille_menage_class
    keys = list(inv)
    dispo = np.array([inv[k] for k in keys], dtype=float)
    presents = np.array([k[1] for k in keys], dtype=float)
    declares = np.array([k[2] for k in keys], dtype=float)
    cellule = [k[0] for k in keys]
    motorisation = [k[0].split(" × ")[1] for k in keys]
    classe_taille = [taille_menage_class(k[2]) for k in keys]
    capacite: Counter = Counter()
    for j, k in enumerate(keys):
        capacite[k[0]] += inv[k] * k[1]
    reference = np.array([round(dispo[j] * effectif[cellule[j]] / capacite[cellule[j]])
                          if capacite[cellule[j]] else 0.0 for j in range(len(keys))])
    return _Contexte(keys, dispo, presents, declares, cellule, motorisation, classe_taille,
                     effectif, n, cibles, reference)


def _programme(ctx: _Contexte, tol: float, avec_objectif: bool):
    """The integer program at tolerance `tol` (maximal gap, in points, of the allocated margins).

    Variables: the number of households kept per sub-cell (integer, bounded by the inventory),
    and — when `avec_objectif` — the absolute gap to the allocation proportional to the pool.
    Constraints: the twelve cell counts in PERSONS (equalities), household size on a
    person basis and car ownership on a household basis within ± `tol` point. The car ownership
    constraint is written `|100·W_m − cible_m·W| ≤ tol·W`: the denominator W (sum of weights
    `S/T`) is itself a variable, and the form stays LINEAR — which avoids estimating W in advance.
    """
    nk = len(ctx.keys)
    nvar = 2 * nk if avec_objectif else nk
    cellules = sorted(ctx.effectif)
    cel = np.array(ctx.cellule)
    mot = np.array(ctx.motorisation)
    cls = np.array([c or "" for c in ctx.classe_taille])
    poids = ctx.presents / ctx.declares          # household weight of a sub-cell
    rows, lo, hi = [], [], []

    def _contrainte(coefficients: "np.ndarray", borne_basse: float, borne_haute: float) -> None:
        rows.append(coefficients)
        lo.append(borne_basse)
        hi.append(borne_haute)

    for c in cellules:                            # 1) twelve cell counts, EXACT
        r = np.zeros(nvar)
        r[:nk] = np.where(cel == c, ctx.presents, 0.0)
        _contrainte(r, float(ctx.effectif[c]), float(ctx.effectif[c]))
    for modalite, cible in ctx.cibles["taille_menage_personne"].items():
        r = np.zeros(nvar)                        # 2) household size, PERSON basis
        r[:nk] = np.where(cls == modalite, 100.0 * ctx.presents / ctx.n, 0.0)
        _contrainte(r, cible - tol, cible + tol)
    for modalite, cible in ctx.cibles["motorisation_menage"].items():
        base = (np.where(mot == modalite, 100.0, 0.0) - cible) * poids
        for signe in (+1.0, -1.0):                # 3) car ownership, HOUSEHOLD basis
            r = np.zeros(nvar)
            r[:nk] = signe * base - tol * poids
            _contrainte(r, -np.inf, 0.0)
    if avec_objectif:
        for j in range(nk):                       # 4) gap to the proportional allocation
            for signe in (+1.0, -1.0):
                r = np.zeros(nvar)
                r[j] = signe
                r[nk + j] = -1.0
                _contrainte(r, -np.inf, signe * ctx.reference[j])
    cons = LinearConstraint(np.array(rows), np.array(lo), np.array(hi))
    borne_haute = (np.concatenate([ctx.dispo, np.full(nk, np.inf)]) if avec_objectif else ctx.dispo)
    bounds = Bounds(np.zeros(nvar), borne_haute)
    integralite = (np.concatenate([np.ones(nk), np.zeros(nk)]) if avec_objectif else np.ones(nk))
    objectif = np.zeros(nvar)
    if avec_objectif:
        objectif[nk:] = 1.0
    return milp(c=objectif, constraints=cons, integrality=integralite, bounds=bounds,
                options={"presolve": True, "mip_rel_gap": 0.0,
                         "time_limit": ALLOCATION_SOLVEUR_TIMEOUT_S})


def _parts_obtenues(ctx: _Contexte, n_par_sous_cellule: "np.ndarray") -> dict[str, dict[str, float]]:
    """Shares of the two allocated margins, recomputed on the kept counts."""
    cls = np.array([c or "" for c in ctx.classe_taille])
    mot = np.array(ctx.motorisation)
    personnes = float(np.sum(n_par_sous_cellule * ctx.presents)) or 1.0
    poids = n_par_sous_cellule * ctx.presents / ctx.declares
    total_poids = float(np.sum(poids)) or 1.0
    return {
        "taille_menage_personne": {
            mod: round(100.0 * float(np.sum(n_par_sous_cellule[cls == mod]
                                            * ctx.presents[cls == mod])) / personnes, 4)
            for mod in ctx.cibles["taille_menage_personne"]},
        "motorisation_menage": {
            mod: round(100.0 * float(np.sum(poids[mot == mod])) / total_poids, 4)
            for mod in ctx.cibles["motorisation_menage"]},
    }


def allouer_sous_cellules(inv: dict[SousCellule, int], effectif: dict[str, int], n: int,
                          cibles: dict[str, dict[str, float]]) -> tuple[dict[SousCellule, int], dict]:
    """Target counts per sub-cell: the finest tolerance the pool allows.

    Bisection on the maximal gap admitted on the two allocated margins (feasibility increasing
    with tolerance), then a last program which, at that tolerance, minimises the number of
    households moved relative to the pool composition — it is the draw that distorts least
    what the pool already carries, and it makes the solution reproducible rather than arbitrary.
    """
    t0 = time.monotonic()
    ctx = _contexte(inv, effectif, n, cibles)
    essais: list[dict] = []

    def faisable(tol: float) -> bool:
        res = _programme(ctx, tol, avec_objectif=False)
        essais.append({"tolerance_pt": round(tol, 4), "faisable": bool(res.success),
                       "statut": int(res.status), "message": str(res.message)[:120]})
        return bool(res.success)

    if not faisable(ALLOCATION_TOLERANCE_MAX_PT):
        raise AllocationError(
            "no integer allocation holds the twelve cell counts in persons on "
            f"this inventory ({len(inv)} sub-cells, {sum(inv.values())} households) — even at "
            f"tolerance {ALLOCATION_TOLERANCE_MAX_PT} pt. The pool does not carry the household "
            "sizes needed to compose these counts exactly.")
    bas, haut = 0.0, ALLOCATION_TOLERANCE_MAX_PT
    while haut - bas > ALLOCATION_TOLERANCE_PAS_PT:
        milieu = round((bas + haut) / 2, 6)
        if faisable(milieu):
            haut = milieu
        else:
            bas = milieu
    tol = round(haut, 4)

    res = _programme(ctx, tol, avec_objectif=True)
    if not res.success:
        raise AllocationError(
            f"the allocation program is feasible at {tol} pt but its optimisation failed "
            f"(status {res.status}: {res.message}). No fallback: the selection stops.")
    n_par = np.rint(res.x[:len(ctx.keys)]).astype(int)
    if np.any(n_par < 0) or np.any(n_par > ctx.dispo + 1e-6):
        raise AllocationError("allocation solution outside the inventory of the pool")

    parts = _parts_obtenues(ctx, n_par.astype(float))
    ecarts = {nom: {mod: round(parts[nom][mod] - cible, 4) for mod, cible in cibles[nom].items()}
              for nom in cibles}
    ecart_max = max((abs(v) for row in ecarts.values() for v in row.values()), default=0.0)
    depassee = ecart_max > ALLOCATION_TOLERANCE_ALARME_PT
    cibles_sc = {ctx.keys[j]: int(n_par[j]) for j in range(len(ctx.keys)) if n_par[j]}
    personnes = int(np.sum(n_par * ctx.presents))
    journal = {
        "methode": ("programme entier (HiGHS via scipy.optimize.milp) sur les sous-cellules "
                    "(cellule × effectif présent × taille déclarée) : les douze effectifs de "
                    "cellule en personnes sont des égalités, les deux marges allouées sont "
                    "bornées en écart maximal — tolérance trouvée par bissection —, et "
                    "l'objectif minimise le nombre de ménages déplacés par rapport à la "
                    "composition du vivier"),
        "marges_allouees": list(ALLOCATION_MARGES),
        "cibles_pct": cibles,
        "parts_obtenues_pct": parts,
        "ecarts_pt": ecarts,
        "ecart_max_pt": round(ecart_max, 4),
        "tolerance": {"pas_pt": ALLOCATION_TOLERANCE_PAS_PT, "retenue_pt": tol,
                      "borne_alarme_pt": ALLOCATION_TOLERANCE_ALARME_PT, "depassee": depassee,
                      "essais": essais, "programmes_resolus": len(essais) + 1},
        "sous_cellules": {
            "vivier": len(inv), "servies": len(cibles_sc),
            "cibles_menages": {_libelle(k): v for k, v in sorted(cibles_sc.items())},
            "vivier_menages": {_libelle(k): v for k, v in inv.items()}},
        "menages": int(n_par.sum()),
        "personnes": personnes,
        "menages_deplaces_vs_vivier": int(round(float(res.fun))),
        "duree_s": round(time.monotonic() - t0, 2),
    }
    if personnes != sum(effectif.values()):
        raise AllocationError(f"the allocation composes {personnes} persons for "
                              f"{sum(effectif.values())} requested — program inconsistency")
    if depassee:
        logger.error("[ALARME] allocation : le vivier ne tient pas les marges allouées — écart "
                     "maximal %.2f pt (> %.2f pt) sur %s ; écarts %s", ecart_max,
                     ALLOCATION_TOLERANCE_ALARME_PT, list(ALLOCATION_MARGES), ecarts)
    else:
        logger.info("allocation: %d sub-cells served out of %d, %d households / %d persons, "
                    "maximal gap %.3f pt (tolerance %.2f pt, %d programs, %.1fs)",
                    len(cibles_sc), len(inv), int(n_par.sum()), personnes, ecart_max, tol,
                    len(essais) + 1, journal["duree_s"])
    return cibles_sc, journal


def effectifs_par_cellule(menages: list[Menage],
                          targets: dict[str, int]) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """Cell counts the pool can serve, and the carry-overs of what it cannot.

    A cell the pool does not fill is a DEFICIT: it is carried over first within the
    SAME RING (the spatial margin moves the modal targets by 30 points), then to the
    whole pool, within the limit of what the other cells can absorb. Each carry-over
    is logged and alarmed; nothing is filled silently.
    """
    capacite: Counter = Counter()
    for m in menages:
        capacite[m.cellule] += m.size
    effectif = {c: min(targets[c], capacite.get(c, 0)) for c in targets}
    deficits = {c: targets[c] - effectif[c] for c in targets if targets[c] > effectif[c]}
    reports: list[dict] = []
    for cell, manque in deficits.items():
        couronne = cell.split(" × ")[0]
        reste = manque
        soeurs = [c for c in targets if c.startswith(couronne + " × ") and c != cell]
        autres = [c for c in targets if c != cell and c not in soeurs]
        for portee, cells in (("même couronne", soeurs), ("vivier entier", autres)):
            for other in sorted(cells, key=lambda c: -targets[c]):
                marge = capacite.get(other, 0) - effectif[other]
                if marge <= 0 or reste == 0:
                    continue
                pris = min(marge, reste)
                effectif[other] += pris
                reste -= pris
                reports.append({"deficit": cell, "vers": other, "n": pris, "portee": portee})
            if reste == 0:
                break
        if reste:
            raise ValueError(f"cannot complete {cell}: {reste} persona(s) missing "
                             "in the whole pool")
    return effectif, deficits, reports


def allocate(menages: list[Menage], targets: dict[str, int],
             n: int) -> tuple[dict[str, Menage], dict, dict, list, dict]:
    """Keeps the households: cell counts, then counts per sub-cell, then hash.

    Households enter in the order of their `sha256` WITHIN each sub-cell —
    the determinism of v4, with one more dimension.
    """
    effectif, deficits, reports = effectifs_par_cellule(menages, targets)
    inv = inventaire(menages)
    cibles_sc, alloc = allouer_sous_cellules(inv, effectif, n, cibles_allocation())

    par_sous_cellule: dict[SousCellule, list[Menage]] = defaultdict(list)
    for m in menages:
        par_sous_cellule[sous_cellule(m)].append(m)     # `menages` is already sorted by rank
    chosen: dict[str, Menage] = {}
    taken: dict[str, int] = {c: 0 for c in targets}
    manques: list[dict] = []
    for key in sorted(cibles_sc):
        veut = cibles_sc[key]
        pris = 0
        for m in par_sous_cellule.get(key, []):
            if pris >= veut:
                break
            chosen[m.id] = m
            taken[key[0]] += m.size
            pris += 1
        if pris < veut:
            manques.append({"sous_cellule": _libelle(key), "manque_menages": veut - pris,
                            "manque_personnes": (veut - pris) * key[1]})

    # Safeguard: the inventory bounds the program, so an under-filled sub-cell is
    # impossible by construction. If it happens anyway, it is filled WITHIN THE SAME
    # CELL (the twelve counts in persons are the constraint not to let go), then within
    # the same ring, then in the pool — logged, alarmed, exit code 1.
    if manques:
        logger.error("[ALARME] allocation : %d sous-cellule(s) sous-remplie(s) — %s ; report en "
                     "cours, la composition allouée n'est plus exacte", len(manques), manques)
        for manque in manques:
            reste = manque["manque_personnes"]
            cell = manque["sous_cellule"].split(" | ")[0]
            couronne = cell.split(" × ")[0]
            for portee, cibles_report in (
                    ("même cellule (autre sous-cellule)", [cell]),
                    ("même couronne", [c for c in targets
                                       if c.startswith(couronne + " × ") and c != cell]),
                    ("vivier entier", [c for c in targets if c != cell
                                       and not c.startswith(couronne + " × ")])):
                for other in sorted(cibles_report, key=lambda c: -targets[c]):
                    for m in menages:
                        if reste == 0:
                            break
                        if m.cellule != other or m.id in chosen or m.size > reste:
                            continue
                        chosen[m.id] = m
                        taken[other] += m.size
                        reste -= m.size
                        reports.append({"deficit": manque["sous_cellule"], "vers": other,
                                        "n": m.size, "portee": portee})
                    if reste == 0:
                        break
                if reste == 0:
                    break
            if reste:
                raise AllocationError(
                    f"sub-cell {manque['sous_cellule']} under-filled by {reste} persona(s) "
                    "and nothing to carry over in the whole pool")
            deficits[cell] = deficits.get(cell, 0) + manque["manque_personnes"]
    alloc["manques"] = manques
    alloc["retenus_par_sous_cellule"] = {
        _libelle(k): v for k, v in sorted(Counter(sous_cellule(m) for m in chosen.values()).items())}
    return chosen, taken, deficits, reports, alloc


# ── Descent on multiple margins ───────────────────────────────────────────────

class _Etat:
    """Counts per margin of the current selection; loss in points, incremental update."""

    def __init__(self, marges_defs: list[tuple[str, Callable, dict[str, float]]]):
        self.defs = marges_defs
        self.counts: dict[str, Counter] = {nom: Counter() for nom, _, _ in marges_defs}
        self.fields: dict[str, float] = {nom: 0.0 for nom, _, _ in marges_defs}

    def add(self, persona, sign: int = 1) -> None:
        for nom, fn, _ in self.defs:
            mod = fn(persona)
            if mod is None:
                continue
            # Household basis: the persona weighs the inverse of its household's declared size.
            # A zero weight (missing size) removes the person from the margin's field without
            # removing it from the others — that is the rule of `household_weight`, not a fallback.
            poids = persona.poids_menage if nom in DESCENTE_MARGES_MENAGE else 1.0
            if not poids:
                continue
            self.counts[nom][mod] += sign * poids
            self.fields[nom] += sign * poids

    def loss(self) -> float:
        total = 0.0
        for nom, _, target in self.defs:
            f = self.fields[nom]
            if f <= 1e-9:
                continue
            c = self.counts[nom]
            for mod, cible in target.items():
                total += abs(100.0 * c[mod] / f - cible)
        return total

    def snapshot(self) -> dict[str, dict[str, float]]:
        out = {}
        for nom, _, target in self.defs:
            f = self.fields[nom]
            out[nom] = {mod: (round(100.0 * self.counts[nom][mod] / f, 2) if f > 1e-9 else None)
                        for mod in target}
        return out


def descend(chosen: dict[str, Menage], menages: list[Menage], personas: dict[str, ctl.Persona],
            marges_defs: list[tuple[str, Callable, dict[str, float]]]) -> dict:
    """Swaps of households of the SAME SUB-CELL that reduce the multi-margin loss.

    The sub-cell is the triplet (cell, present size, declared size) — the very stratum
    of the allocation. A swap at constant sub-cell therefore leaves intact, by construction,
    the twelve cell counts in persons, household size on a person basis and
    car ownership on a household basis: the descent only works on what the allocation does
    not fix. v4 paired on (cell, present size): the declared size could
    move, and with it the weight `1/size` — which made household car ownership
    unreachable by the allocation alone.
    """
    t0 = time.monotonic()
    etat = _Etat(marges_defs)
    for m in chosen.values():
        for rec in m.membres:
            etat.add(personas[str(rec["person_id"])])
    avant = etat.snapshot()
    perte0 = etat.loss()
    # Candidates per sub-cell, hash order.
    rest: dict[SousCellule, list[Menage]] = defaultdict(list)
    for m in menages:
        if m.id not in chosen:
            rest[sous_cellule(m)].append(m)

    swaps = passes = 0
    perte = perte0
    while passes < DESCENTE_PASSES_MAX:
        passes += 1
        improved = False
        for hid in sorted(chosen, key=lambda h: chosen[h].rank):
            h = chosen[hid]
            candidates = rest.get(sous_cellule(h), [])
            if not candidates:
                continue
            for rec in h.membres:
                etat.add(personas[str(rec["person_id"])], -1)
            best, best_loss = None, perte
            for x in candidates[:DESCENTE_CANDIDATS]:
                for rec in x.membres:
                    etat.add(personas[str(rec["person_id"])], +1)
                l = etat.loss()
                for rec in x.membres:
                    etat.add(personas[str(rec["person_id"])], -1)
                if l < best_loss - 1e-9:
                    best, best_loss = x, l
                    break   # first improvement, in hash order: deterministic
            if best is None:
                for rec in h.membres:
                    etat.add(personas[str(rec["person_id"])], +1)
                continue
            for rec in best.membres:
                etat.add(personas[str(rec["person_id"])], +1)
            del chosen[hid]
            chosen[best.id] = best
            candidates.remove(best)
            candidates.append(h)
            candidates.sort(key=lambda m: m.rank)
            perte = best_loss
            swaps += 1
            improved = True
        if not improved:
            break
    apres = etat.snapshot()
    marges_journal = {}
    for nom, _, target in marges_defs:
        ecart_avant = max((abs((avant[nom][k] or 0) - v) for k, v in target.items()), default=0.0)
        ecart_apres = max((abs((apres[nom][k] or 0) - v) for k, v in target.items()), default=0.0)
        marges_journal[nom] = {"cible_pct": target, "avant_pct": avant[nom], "apres_pct": apres[nom],
                               "ecart_max_avant_pt": round(ecart_avant, 2),
                               "ecart_max_apres_pt": round(ecart_apres, 2),
                               "champ": round(etat.fields[nom], 2),
                               "base": "menage" if nom in DESCENTE_MARGES_MENAGE else "personne",
                               "mesuree": etat.fields[nom] > 1e-9}
    non_mesurees = [nom for nom, j in marges_journal.items() if not j["mesuree"]]
    if non_mesurees:
        logger.warning("descent: margins without any value on the pool, ignored — %s "
                       "(traits not imputed before the selection?)", non_mesurees)
    logger.info("descent: %d swaps in %d pass(es), loss %.1f → %.1f pt, %.1fs",
                swaps, passes, perte0, perte, time.monotonic() - t0)
    return {"marges": marges_journal, "echanges": swaps, "passes": passes,
            "perte_avant_pt": round(perte0, 2), "perte_apres_pt": round(perte, 2),
            "candidats_par_menage": DESCENTE_CANDIDATS, "marges_non_mesurees": non_mesurees,
            "duree_s": round(time.monotonic() - t0, 2)}


def _marges_defs(personas: dict[str, ctl.Persona]) -> list[tuple[str, Callable, dict[str, float]]]:
    defs = []
    for m in marges(JOINT_TARGET, MARGES_TARGET):
        if m.nom not in DESCENTE_MARGES or not m.mesurable:
            continue
        total = sum(m.cible_pct.values())
        target = {k: 100.0 * v / total for k, v in m.cible_pct.items()}
        defs.append((m.nom, (lambda p, nom=m.nom: ctl.modalite_of(p, nom)), target))
    return defs


# ── Selection ─────────────────────────────────────────────────────────────────

def charger_exclusions(dossiers: list[Path]) -> tuple[set[str], list[dict]]:
    """Households kept by already sealed cohorts, to be removed from the pool.

    Reads the `household.id` values in the `population.json` of each cohort, not in its
    `selection.json`: it is the sealed file that says what the cohort ACTUALLY contains, and
    it is the one whose sha256 circulates. Excluding the household id excludes all its members at
    grouping time, including those the cohort had not kept — the disjunction
    applies to the household, not to the person (R2).

    The sha256 of each cohort is RE-CHECKED against its manifest: a cohort whose file
    has moved since its sealing is no longer the one we think we exclude, and there is no
    knowing in which direction. It is an error, never a warning (R9).
    """
    exclus: set[str] = set()
    journal: list[dict] = []
    for dossier in dossiers:
        dossier = Path(dossier)
        manifest_path, pop_path = dossier / "MANIFEST.yaml", dossier / "population.json"
        for chemin in (manifest_path, pop_path):
            if not chemin.exists():
                raise ValueError(f"cohort to exclude is incomplete: {chemin} is missing")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        attendu = ((manifest.get("population") or {}).get("sha256") or "")
        lu = ctl.sha256_of(pop_path)
        if attendu != lu:
            raise ValueError(
                f"cohort {dossier.name} modified since its sealing: the manifest announces "
                f"sha256 {attendu or '(missing)'}, the file is {lu} — nothing is excluded"
            )
        records = ctl.load_population(pop_path)
        ids = {str((r.get("household") or {}).get("id")) for r in records}
        ids.discard("None")
        avant = len(exclus)
        exclus |= ids
        journal.append({
            "nom": manifest.get("nom") or dossier.name,
            "dossier": str(dossier),
            "sha256": lu,
            "personnes": len(records),
            "menages": len(ids),
            "menages_nouveaux": len(exclus) - avant,
        })
        logger.info("exclusion: %s — %d households (%d persons), sha256 %s…",
                    journal[-1]["nom"], len(ids), len(records), lu[:16])
    return exclus, journal


def verifier_disjonction(retenus: set[str], exclus: set[str]) -> None:
    """The disjunction is checked on the RESULT, not only at the input (R10).

    A filter that used the wrong key — `id` versus `household.id`, an integer versus a string —
    would let an overlapping cohort through without any message saying so, and axis 2 would be
    published on sand. An error, not an assertion: it must hold under `python -O`.
    """
    if not exclus:
        return
    fuite = sorted(retenus & exclus)
    if fuite:
        raise ValueError(
            f"disjointness broken: {len(fuite)} household(s) already kept by an earlier "
            f"cohort appear in the selection — {fuite[:5]}"
        )


def select(records: list[dict], n: int, joint_path: Path = JOINT_TARGET,
           exclure_menages: Optional[set[str]] = None) -> tuple[list[dict], dict]:
    """Stratified selection by households of `n` personas. Returns `(retenus, journal)`.

    `exclure_menages` removes from the pool the households kept by earlier cohorts, to
    produce a disjoint cohort (ticket 073, axis 2). Whole pool by default: with an empty
    exclusion, this function returns exactly what it returned before the parameter existed.
    """
    t0 = time.monotonic()
    exclure_menages = set(exclure_menages or ())
    joint = cible_jointe(joint_path)
    cells_pct = {f"{c} × {m}": float(joint["cible_pct"][c][m])
                 for c in COURONNES for m in MOTORISATION}
    targets = largest_remainder(cells_pct, n)

    menages, excluded = group_households(records)

    # Trimming of the pool (R1). It happens AFTER grouping, so excluding a
    # `household.id` takes all members of the household without having to list them (R2). The
    # capacities per cell are recorded BEFORE and AFTER so that the log tells a deficit
    # caused by the exclusion from a deficit of the pool itself (R7): both end in
    # carry-overs to other cells, but not for the same reason and not with the same remedy.
    capacite_avant = Counter()
    for m in menages:
        capacite_avant[m.cellule] += m.size
    if exclure_menages:
        retires = [m for m in menages if m.id in exclure_menages]
        menages = [m for m in menages if m.id not in exclure_menages]
        excluded["exclus_cohorte_anterieure"] = sum(m.size for m in retires)
        logger.info("trimmed pool: %d households removed (%d persons) — %d households left",
                    len(retires), sum(m.size for m in retires), len(menages))

    eligible = sum(m.size for m in menages)
    if eligible < n:
        cause = ("après exclusion des cohortes antérieures" if exclure_menages
                 else "sur le vivier entier")
        raise ValueError(f"pool too small {cause}: {eligible} eligible personas for {n} "
                         f"requested (excluded: {dict(excluded)})")
    chosen, taken, deficits, reports, allocation = allocate(menages, targets, n)
    assert sum(m.size for m in chosen.values()) == n, (sum(m.size for m in chosen.values()), n)
    sous_cellules_allouees = Counter(sous_cellule(m) for m in chosen.values())

    personas_list, _counters = ctl.normalize(records)
    personas = {p.id: p for p in personas_list}
    descente = descend(chosen, menages, personas, _marges_defs(personas))

    # Internal check: the descent moved neither a cell count nor the total count…
    cell_counts = Counter()
    for m in chosen.values():
        cell_counts[m.cellule] += m.size
    assert cell_counts == Counter({c: t for c, t in taken.items() if t}), "the descent moved a cell"
    # … nor any SUB-CELL: this is what guarantees that the two allocated margins (household
    # size on a person basis, car ownership on a household basis) come out of the descent intact.
    assert Counter(sous_cellule(m) for m in chosen.values()) == sous_cellules_allouees, \
        "the descent moved a sub-cell (cell × present size × declared size)"
    retenus = [rec for m in chosen.values() for rec in m.membres]
    assert len(retenus) == n, (len(retenus), n)
    retenus.sort(key=lambda r: int(str(r.get("person_id"))) if str(r.get("person_id")).isdigit()
                 else str(r.get("person_id")))

    verifier_disjonction(set(chosen), exclure_menages)

    # Deficits ATTRIBUTABLE to the trimming: the cell served the target before exclusion, not after.
    deficits_exclusion = {}
    if exclure_menages and deficits:
        deficits_exclusion = {c: manque for c, manque in deficits.items()
                              if capacite_avant.get(c, 0) >= targets[c]}
        if deficits_exclusion:
            logger.error("[ALARME] %d cell(s) in deficit solely because of the exclusion — %s; the "
                         "whole pool filled them. It is one cohort too many, not a pool "
                         "too small", len(deficits_exclusion), deficits_exclusion)

    if deficits:
        logger.error("[ALARME] selection: %d cell(s) in deficit — %s — %d carry-over(s); the pool "
                     "is too small for the joint target", len(deficits), dict(deficits),
                     sum(r["n"] for r in reports))
    by_cell_n = Counter()
    for m in menages:
        by_cell_n[m.cellule] += m.size
    for cell in targets:
        logger.info("cell %-36s target %4d · pool %5d · kept %4d", cell, targets[cell],
                    by_cell_n.get(cell, 0), taken[cell])

    sizes = Counter(m.size for m in chosen.values())
    perimetre = perimeter_journal(retenus)
    if perimetre["departements_representes"] < len(PERIMETRE["departements_attendus"]):
        logger.warning("scope: %d department(s) represented out of %d expected — restricted "
                       "sampling frame (%s); the 3rd ring is cut off from its outer "
                       "communes", perimetre["departements_representes"],
                       len(PERIMETRE["departements_attendus"]), perimetre["retenus_par_departement"])
    journal = {
        "version": SELECTION_RULE_DISJOINT if exclure_menages else SELECTION_RULE,
        "perimetre": perimetre,
        "regle": ("unité = ménage (household.id) ; effectifs de cellule proportionnels à la cible "
                  "jointe couronne × motorisation (base personne) par plus fort reste, puis "
                  "effectif cible par SOUS-CELLULE (cellule × effectif présent × taille déclarée) "
                  "par programme entier tenant " + " et ".join(ALLOCATION_MARGES) + " ; les ménages "
                  f"entrent dans l'ordre sha256('{SELECTION_NAMESPACE}:' + household_id) à "
                  "l'intérieur de leur sous-cellule ; exclus : hors périmètre, moins de 5 ans, "
                  "sans motorisation ; puis descente par échanges de ménages de MÊME SOUS-CELLULE "
                  "minimisant la somme des écarts absolus (en points) aux marges : "
                  + ", ".join(DESCENTE_MARGES)),
        "cible_jointe": {"fichier": str(joint_path), "version": joint.get("version"),
                         "sha256": ctl.sha256_of(joint_path)},
        "cibles_marges": {"fichier": str(MARGES_TARGET), "sha256": ctl.sha256_of(MARGES_TARGET)},
        "n_demande": n,
        "n_retenu": len(retenus),
        "vivier": {"n": len(records), "eligibles": eligible, "exclus": dict(excluded),
                   "menages": len(menages), "par_cellule": dict(by_cell_n),
                   "menages_exclus": len(exclure_menages)},
        "deficits_imputables_a_l_exclusion": deficits_exclusion,
        "menages_retenus": {"n": len(chosen), "par_taille": {str(k): v for k, v in sorted(sizes.items())},
                            "membres_declares": sum(m.taille_declaree for m in chosen.values()),
                            "membres_presents": n},
        "cibles": targets,
        "retenus_par_cellule": taken,
        "deficits": deficits,
        "reports": reports,
        "allocation": allocation,
        "descente": descente,
        "person_ids": [str(r.get("person_id")) for r in retenus],
        "household_ids": sorted(chosen),
        "duree_s": round(time.monotonic() - t0, 2),
    }
    logger.info("selection done in %.1fs: %d kept (%d households) out of %d eligible (%d excluded), "
                "%d sub-cell(s) served, maximal gap of the allocated margins %.3f pt, "
                "%d deficit(s), %d under-filled sub-cell(s), %d swap(s)",
                journal["duree_s"], len(retenus), len(chosen), eligible, sum(excluded.values()),
                allocation["sous_cellules"]["servies"], allocation["ecart_max_pt"], len(deficits),
                len(allocation["manques"]), descente["echanges"])
    return retenus, journal


def cmd_select(args) -> int:
    if not args.pool.exists():
        logger.error("[ALARME] pool not found: %s", args.pool)
        return 2
    records = ctl.load_population(args.pool)
    pool_digest = ctl.sha256_of(args.pool)
    logger.info("pool: %s — %d personas, sha256 %s…", args.pool, len(records), pool_digest[:16])
    posed = ensure_residence_zone(records)
    try:
        exclus, exclusions = charger_exclusions(args.exclure or [])
        chosen, journal = select(records, args.n, exclure_menages=exclus)
    except (ReferenceError, ValueError) as exc:
        logger.error("[ALARME] selection failed: %s", exc)
        return 2
    journal["disjonction"] = ({"regle": journal["version"], "cohortes": exclusions,
                               "menages_exclus": len(exclus)} if exclusions else None)
    journal["vivier"]["fichier"] = str(args.pool)
    journal["vivier"]["sha256"] = pool_digest
    journal["vivier"]["residence_zone"] = dict(posed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    journal["sortie"] = {"fichier": str(args.out), "sha256": ctl.sha256_of(args.out)}
    sel_path = args.selection_json or args.out.with_name(args.out.stem + "_selection.json")
    sel_path.write_text(json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("written: %s (%d personas) and %s", args.out, len(chosen), sel_path)
    d = journal["descente"]
    print(f"{len(chosen)} personas kept ({journal['menages_retenus']['n']} households) out of "
          f"{journal['vivier']['eligibles']} eligible ({journal['vivier']['n']} in the pool) → {args.out}")
    per = journal["perimetre"]
    print(f"scope: {per['definition']}; kept per department: {per['retenus_par_departement']} "
          f"({per['departements_representes']}/{len(PERIMETRE['departements_attendus'])} departments, "
          f"{per['communes_distinctes']} communes"
          + (f", {per['retenus_sans_commune']} without commune" if per['retenus_sans_commune'] else "") + ")")
    a = journal["allocation"]
    print(f"allocation: {a['sous_cellules']['servies']}/{a['sous_cellules']['vivier']} sub-cells "
          f"served, {a['menages']} households, tolerance kept {a['tolerance']['retenue_pt']} pt "
          f"({a['tolerance']['programmes_resolus']} integer programs, {a['duree_s']}s); "
          f"{a['menages_deplaces_vs_vivier']} household(s) moved relative to the pool")
    for nom in a["marges_allouees"]:
        pire = max(a["ecarts_pt"][nom], key=lambda m: abs(a["ecarts_pt"][nom][m]))
        print(f"   {nom:24s} max gap {a['ecarts_pt'][nom][pire]:+5.2f} pt on « {pire} » (allocated)")
    print(f"descent: {d['echanges']} swap(s) in {d['passes']} pass(es), loss {d['perte_avant_pt']} → "
          f"{d['perte_apres_pt']} pt" + (f"; margins not measured: {d['marges_non_mesurees']}"
                                        if d["marges_non_mesurees"] else ""))
    for nom, j in d["marges"].items():
        if j["mesuree"]:
            print(f"   {nom:24s} max gap {j['ecart_max_avant_pt']:5.2f} → {j['ecart_max_apres_pt']:5.2f} pt")
    code = 0
    if a["tolerance"]["depassee"]:
        print(f"⚠ allocated margins not held: maximal gap {a['ecart_max_pt']} pt "
              f"(> {a['tolerance']['borne_alarme_pt']} pt) — the pool does not carry the household "
              "composition that would be needed; a gap to declare, not to round off")
        code = 1
    if a["manques"]:
        print(f"⚠ {len(a['manques'])} under-filled sub-cell(s), carried over: "
              + ", ".join(f"{m['sous_cellule']} −{m['manque_menages']} ménage(s)" for m in a["manques"]))
        code = 1
    if journal["deficits"]:
        print(f"⚠ {len(journal['deficits'])} cell(s) in deficit, {sum(r['n'] for r in journal['reports'])} carry-over(s): "
              + ", ".join(f"{k} −{v}" for k, v in journal["deficits"].items()))
        code = 1
    return code


# ── Sealing ───────────────────────────────────────────────────────────────────

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "inconnu"


def _resume_allocation(allocation: Optional[dict]) -> Optional[dict]:
    """The allocation summary carried by the MANIFEST: what commits, without the 130 sub-cells.

    The detail (targets and kept per sub-cell, bisection trials) stays in
    `selection.json`, sealed alongside.
    """
    if not allocation:
        return None
    tol = allocation.get("tolerance") or {}
    return {
        "methode": allocation.get("methode"),
        "marges_allouees": allocation.get("marges_allouees"),
        "ecarts_pt": allocation.get("ecarts_pt"),
        "ecart_max_pt": allocation.get("ecart_max_pt"),
        "tolerance_retenue_pt": tol.get("retenue_pt"),
        "tolerance_borne_alarme_pt": tol.get("borne_alarme_pt"),
        "tolerance_depassee": tol.get("depassee"),
        "programmes_entiers_resolus": tol.get("programmes_resolus"),
        "sous_cellules_servies": (allocation.get("sous_cellules") or {}).get("servies"),
        "sous_cellules_vivier": (allocation.get("sous_cellules") or {}).get("vivier"),
        "menages": allocation.get("menages"),
        "menages_deplaces_vs_vivier": allocation.get("menages_deplaces_vs_vivier"),
        "sous_cellules_sous_remplies": len(allocation.get("manques") or []),
    }


def cmd_seal(args) -> int:
    if not args.population.exists():
        logger.error("[ALARME] population not found: %s", args.population)
        return 2
    out_dir: Path = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.error("[ALARME] %s already exists and is not empty: a sealed folder is never "
                     "rewritten. Choose another name (--out-dir).", out_dir)
        return 2

    t0 = time.monotonic()
    try:
        report = ctl.run_control(args.population, args.borne, args.n_min, args.n_min_cellule)
    except (ReferenceError, ValueError, OSError) as exc:
        logger.error("[ALARME] check failed, nothing is sealed: %s", exc)
        return 2
    verdicts = report["verdicts"]
    n = report["population"]["n"]
    if args.n and n != args.n:
        logger.error("[ALARME] count %d ≠ %d expected — nothing is sealed", n, args.n)
        print(ctl.render_text(report))
        return 1
    if verdicts[ctl.A_CORRIGER]:
        logger.error("[ALARME] %d margin(s) « à corriger » — sealing is REFUSED, the candidate "
                     "file stays in place: %s", verdicts[ctl.A_CORRIGER], args.population)
        print(ctl.render_text(report))
        return 1

    selection = None
    if args.selection_json and args.selection_json.exists():
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
    records = ctl.load_population(args.population)
    hors_perimetre = count_removed_out_of_perimeter(records)
    perimetre_manifest = {**PERIMETRE, **((selection or {}).get("perimetre") or {}),
                          "activites_hors_perimetre": hors_perimetre}
    if hors_perimetre["activites_hors_perimetre_supprimees"]:
        logger.warning("scope: %d activity(ies) outside the polygon removed for %d persona(s) — "
                       "assumed hypothesis, declared in the MANIFEST",
                       hors_perimetre["activites_hors_perimetre_supprimees"], hors_perimetre["personas_touches"])
    if not hors_perimetre["controle"]:
        logger.warning("scope: the activities outside the polygon were NOT checked on this "
                       "population (key `perimetre` missing: produced before the step 2 safeguard)")

    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "population.json"
    shutil.copy2(args.population, target)
    digest = ctl.sha256_of(target)
    (out_dir / "CONTROLE.md").write_text(ctl.render_markdown(report), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if selection is not None:
        shutil.copy2(args.selection_json, out_dir / "selection.json")

    manifest = {
        "version": SEAL_VERSION,
        "nom": out_dir.name,
        "scelle_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "population": {"fichier": "population.json", "sha256": digest, "n": n,
                       "source": str(args.population), "source_sha256": report["population"]["sha256"]},
        "perimetre": perimetre_manifest,
        # What this cohort is disjoint from, by name and by sha256 (R4). Without this block, the
        # disjunction cannot be read back from anything: `population.json` carries no trace of
        # what was removed from the pool before drawing from it. `null` for a whole-pool cohort —
        # a missing key would suggest a manifest from an earlier version.
        "disjonction": (selection or {}).get("disjonction"),
        "selection": ({"fichier": "selection.json", "version": selection.get("version"),
                       "regle": selection.get("regle"),
                       "vivier": {k: v for k, v in selection.get("vivier", {}).items() if k != "par_cellule"},
                       "menages_retenus": selection.get("menages_retenus"),
                       "deficits": selection.get("deficits"), "reports": len(selection.get("reports", [])),
                       "allocation": _resume_allocation(selection.get("allocation")),
                       "descente": {k: v for k, v in (selection.get("descente") or {}).items() if k != "marges"}}
                      if selection else "aucune (population fournie telle quelle)"),
        "controle": {"rapport": "CONTROLE.md", "verdicts": verdicts,
                     "borne_tost_pt": args.borne, "n_min": args.n_min, "n_min_cellule": args.n_min_cellule,
                     "cible_jointe": report["parametres"]["cible_jointe"],
                     "cibles_marges": report["parametres"].get("cibles_marges"),
                     "menages_et_mobilite": report.get("menages_et_mobilite"),
                     "synthese_des_ecarts": report["synthese"]},
        "depot": {"git_head": _git("rev-parse", "HEAD"), "branche": _git("rev-parse", "--abbrev-ref", "HEAD"),
                  "arbre_propre": _git("status", "--porcelain") == ""},
        "regle": ("Ce dossier ne se modifie pas. Toute correction de la population produit un "
                  "nouveau dossier scellé ; les jeux gelés et les runs qui citent celui-ci "
                  "citent son sha256."),
        "note": args.note or "",
    }
    (out_dir / "MANIFEST.yaml").write_text(
        "# Population scellée pour l'article — ne pas modifier (cf. `regle`).\n"
        + yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("sealed in %.1fs → %s (sha256 %s…)", time.monotonic() - t0, out_dir, digest[:16])
    print(ctl.render_text(report))
    print(f"\n✅ Sealed: {out_dir} — {n} personas, sha256 {digest}")
    if verdicts[ctl.A_PUBLIER]:
        print(f"   {verdicts[ctl.A_PUBLIER]} margin(s) « à publier » — see the summary of gaps in CONTROLE.md")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="stratified selection by households of N personas from a pool")
    s.add_argument("--pool", type=Path, required=True)
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--exclure", type=Path, action="append", metavar="SEALED_FOLDER",
                   help="folder of an already sealed cohort whose households are removed from the "
                        "pool; repeatable. The cohort produced is then disjoint from these, "
                        "and its selection rule declares it.")
    s.add_argument("--selection-json", type=Path, default=None,
                   help="selection log (default: <out>_selection.json)")
    s.set_defaults(func=cmd_select)
    z = sub.add_parser("seal", help="check then seal a final population")
    z.add_argument("--population", type=Path, required=True)
    z.add_argument("--out-dir", type=Path, default=DEFAULT_SEAL_DIR)
    z.add_argument("--n", type=int, default=1000, help="required count (0 = do not check)")
    z.add_argument("--selection-json", type=Path, default=None)
    z.add_argument("--borne", type=float, default=1.0)
    z.add_argument("--n-min", type=int, default=30)
    z.add_argument("--n-min-cellule", type=int, default=50)
    z.add_argument("--note", type=str, default=None, help="free note (generation parameters…)")
    z.set_defaults(func=cmd_seal)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
