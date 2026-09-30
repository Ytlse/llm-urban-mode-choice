"""export_mode_hierarchy.py — The survey's mode hierarchy, sourced then verified.

    services/llm-agents/.venv/bin/python -m scripts.progedo_logit.export_mode_hierarchy
    services/llm-agents/.venv/bin/python -m scripts.progedo_logit.export_mode_hierarchy --check

WHAT IT IS FOR. A trip that mixes several modes gets **one** main mode. The repository
carried four tables and three answers for the same trip (ticket 022, M1). This script
freezes **one** hierarchy in `mobility_core/data/mode_hierarchy_emc2.json`, which
`mobility_core.mode_hierarchy` serves to the rest of the repository. The code then no
longer needs the restricted-access microdata to run.

## TWO SOURCES, AND THEIR ORDER

**(1) The order is PUBLISHED, and it is already frozen elsewhere.** The AUAT/CEREMA report
on the 2023 mobility survey of the Toulouse catchment area gives in its appendix, page 53
(« Hiérarchie des modes »), the full order of the **36 surveyed modes**, « défini au niveau
national » (p. 12). This table is transcribed in
[`scripts/panel/hierarchie_modes_emc2.yaml`](../panel/hierarchie_modes_emc2.yaml) (version
`hm1`) — **this script READS it, it does not copy it**. A second transcription would be the
fifth mode table of the repository, and ticket 022 exists to remove them.

What lives *here* and nowhere else is the **mapping** between the report's labels and the
ProGEDO codebook (`CODES_PAR_ORDRE`): without it, the table cannot be checked against the
microdata. It is information of a different nature from the order — a dictionary, not a
hierarchy.

**(2) The measurement CHECKS it.** A list copied from a PDF is a literal, and a literal
never fails when production changes (that asymmetry let the Téléo and rail defects slip
through, cf. `scripts/tests/test_parite_modes.py`). So the order is checked **on the
microdata**: for each pair of modes present together in one trip, we look at which mode
the survey kept as `MODP`. A pair is *informative* when the observed `MODP` is one of the
two modes of the pair; otherwise a third mode won and the pair says nothing.

The check is that of axis A7 of ticket 020, generalised to all pairs. A7 had counted 770
trips mixing car and public transport, of which 760 coded « TC » and 10 « voiture ». This
script **replays exactly that figure** (`--check`), and documents its convention: A7
counted the car in the broad `MODE_GROUP` sense (motorised two-wheelers included) and a PT
list **without the cable car or employer transport**. With the full list, the same
measurement gives 773 / 763 / 10.

## WHAT THE MEASUREMENT SETTLED, AND WHAT WAS NOT ASSUMED

The urban bus comes **before** the train. On trips mixing a bus leg (or an interurban
coach leg) and a train leg, the survey codes the bus **34 times out of 35**. The only
exception observed is a `Flixbus + TER` — and the report anticipates it: long-distance
coaches are at rank 12, *below* trains (ranks 8 to 11). The current cascade of
`move_logger._plan_transport_mode`, which tests `_BUS_MODES` before `_RAIL_MODES`, is thus
**compliant**; it is `mode_choice` (train before PT) and `task_worker` (train first) that
diverge.

What the measurement also refutes: the car is tested **first** by
`move_logger._plan_transport_mode`, although it is at rank 19, below all public transport.

## THE CHECKS THAT ALLOW PUBLISHING

- **Anti-vacuity.** Without detailed legs, or without `MODP`, the measurement would find no
  exception and the agreement would be « perfect »: the script requires a minimum count of
  informative pairs and **fails** rather than publish an empty agreement.
- **Residual walking.** Walking is at rank 36 (« Marche à pied UNIQUEMENT ») and never
  appears as a leg: `T3` carries no code `01`. So we check that trips *without* a detailed
  leg are **all** coded `MODP = 01`, and that no trip *with* a leg is. That is what makes
  walking a measured rank and not an assumed one.
- **Published / measured consistency.** Every informative pair whose winner contradicts the
  published order is listed in `exceptions`, with its count. A single structural exception
  is expected (the Flixbus), and it is *consistent* with the published order.

⚠ This script publishes **no modal share**: the hierarchy is a coding rule, not a
population quantity. Counts are therefore **unweighted** — a `COEP` would not make a rule
truer, it would weight its opportunities of observation. The A7 check, for its part, is
reproduced unweighted as in ticket 020.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV" / "fichiers_standards"
OUT = ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "mode_hierarchy_emc2.json"

logger = logging.getLogger("progedo.mode_hierarchy")

VERSION = "mh1"
RAPPORT = ("AUAT/CEREMA, Rapport final « Enquête mobilité 2023 — bassin de vie "
           "toulousain » (68 p., mai 2024), annexe « Hiérarchie des modes », p. 53")
RAPPORT_URL = ("https://www.aua-toulouse.org/wp-content/uploads/2024/05/"
               "Rapport-final-68-pages-Enquete-mobilite-2023-Bassin-de-vie-toulousain.pdf")

# ── (1) THE SOURCE: the published table, READ and not copied ──────────────────────
HIERARCHIE_PUBLIEE = ROOT / "scripts" / "panel" / "hierarchie_modes_emc2.yaml"
VERSION_PUBLIEE_ATTENDUE = "hm1"

# Published order → `T3` codes (mode of a leg) and `MODP` codes (main mode of a trip) of
# the ProGEDO microdata. This is the MAPPING between the report's labels and the survey
# codebook, and it exists only here: without it, the published table cannot be checked
# against the microdata. An order without a code is a label the coding does not
# distinguish (regional DRT shares code 37 with Tisséo DRT; « autre TER » shares 52 with
# TER liO; rental bikes have no code of their own).
CODES_PAR_ORDRE: dict[int, tuple[str, ...]] = {
    1: ("33",),            # metro passenger (Tisséo)
    2: ("32",),            # tram passenger (Tisséo)
    3: ("34",),            # Téléo passenger (Tisséo cable car)
    4: ("31",),            # bus, shuttle passenger (Tisséo)
    5: ("37",),            # DRT (Tisséo) — code 37 covers « U ou IU »
    6: ("41", "43"),       # liO interurban coaches and other coaches (school)
    7: (),                 # regional DRT — no code distinct from 37
    8: ("52",),            # liO regional train (TER)
    9: ("51",),            # TGV
    10: (),                # other TER — no code distinct from 52
    11: ("53", "54"),      # other trains (Intercités, TET), unspecified train
    12: ("42",),           # long-distance coaches (Flixbus…)
    13: ("71",),           # employer transport
    14: ("61",),           # taxi
    15: ("62",),           # ride-hailing (VTC)
    16: ("81",),           # van/light truck driver
    17: ("82",),           # van/light truck passenger
    18: ("95",),           # other modes (tractor, quad…)
    19: ("21",),           # private car driver
    20: ("22",),           # private car passenger
    21: ("13", "15", "19"),   # motorised 2/3-wheeler driver
    22: ("14", "16", "20"),   # motorised 2/3-wheeler passenger
    23: ("18",),           # self-service e-bike
    24: ("10",),           # VélôToulouse bike share
    25: (),                # rental bike — no code of its own
    26: (),                # rental bike passenger — same
    27: ("17",),           # e-bike
    28: ("11",),           # bicycle rider
    29: ("12",),           # bicycle passenger
    30: ("96",),           # motorised personal mobility devices
    31: ("93", "97"),      # rollerblades, skateboard, scooter
    32: ("94",),           # wheelchair
    33: ("38", "39"),      # passenger of another urban network
    34: ("91",),           # river or sea transport
    35: ("92",),           # plane
    36: ("01",),           # walking ONLY
}

# ── The SIMULATION vocabulary, attached to the published orders ───────────────────
# `ordres` = the published orders this family covers. Its RANK is the smallest of
# them — and it is checked against `correspondance_simulation` of the published table.
# `jambes` = the `leg.mode` values produced by OTP (`trip_helper/otp.py`,
# Transmodel v3), OSMnx (`trip_helper/osmnx_direct.py`) and the synthetic school bus
# (`trip_helper/school_bus.py`), plus the historical aliases still present in the
# caches and labels (`subway`, `bike`, `walk`, `__car__`).
# `libelle_journal` = column « Mode de transport Choisi » of `moves.csv`.
# `mode_canonique` = vocabulary of `mobility_llm.mode_choice.CANONICAL_MODES`.
#
# No speculative mode: `trip_helper/otp.py` **asserts** that every returned leg is in
# `SUPPORTED_MODES`, so a mode missing from that list cannot occur, and listing it would
# create a parity obligation (`test_parite_modes`) for an impossible case.
FAMILLES: tuple[dict, ...] = (
    {"famille": "metro", "ordres": (1,), "jambes": ("metro", "subway"),
     "libelle_journal": "Transports_collectifs", "mode_canonique": "public_transport"},
    {"famille": "tram", "ordres": (2,), "jambes": ("tram", "tramway"),
     "libelle_journal": "Transports_collectifs", "mode_canonique": "public_transport"},
    {"famille": "cableway", "ordres": (3,), "jambes": ("cableway", "gondola", "funicular"),
     "libelle_journal": "Transports_collectifs", "mode_canonique": "public_transport"},
    # `school_bus` is the synthetic option of ticket 030: a school pick-up coach, hence
    # order 6 (« autres autocars — scolaires »), in the same family as the bus. liO
    # coaches come out of OTP as `bus` (route_type=3): no `coach` leg to expect, and
    # Tisséo (4) and liO (6) differ only by operator.
    {"famille": "bus", "ordres": (4, 5, 6, 7), "jambes": ("bus", "school_bus"),
     "libelle_journal": "Transports_collectifs", "mode_canonique": "public_transport"},
    {"famille": "rail", "ordres": (8, 9, 10, 11), "jambes": ("rail",),
     "libelle_journal": "Train", "mode_canonique": "train"},
    # Taxi (14), VTC (15) and van (16-17) are SMALLER orders than the private car (19)
    # and all map to the same leg mode `car`: the family therefore takes rank 14. The
    # project models neither taxi nor VTC — no leg reaches them — but listing them makes
    # the aggregation complete and the pair measurement interpretable.
    {"famille": "car", "ordres": (14, 15, 16, 17, 19, 20), "jambes": ("car", "__car__"),
     "libelle_journal": "Voiture Privée", "mode_canonique": "car"},
    # Missing from `correspondance_simulation`: no motorised two-wheeler leg is
    # produced. The family exists anyway, because `mode_choice.CANONICAL_MODES` and the
    # column `P(Deux-roues motorisé) %` carry it: without a rank, an exotic label would
    # drop it into the catch-all without anyone knowing.
    {"famille": "motorbike", "ordres": (21, 22), "jambes": (),
     "libelle_journal": "Deux-roues motorisé", "mode_canonique": "motorbike"},
    {"famille": "bicycle", "ordres": (23, 24, 25, 26, 27, 28, 29), "jambes": ("bicycle", "bike"),
     "libelle_journal": "Vélo", "mode_canonique": "cycling"},
    {"famille": "foot", "ordres": (36,), "jambes": ("foot", "walk"),
     "libelle_journal": "Marche", "mode_canonique": "walking"},
)

# Minimum count of informative pairs below which publishing is refused: without it, a
# broken read (empty join, unreadable `MODP`) would return « zero exceptions », that is,
# perfect agreement through absence of measurement.
MIN_PAIRES_INFORMATIVES = 500
# Minimum informative count for a pair to decide on its own. Three concordant
# observations distinguish a rule from a coding accident (1 chance in 4 under H0 for
# 3 draws, 1 in 32 for 5); below it, the pair is declared undecided by the
# measurement — and the published order decides it, which is stated rank by rank.
MIN_INFORMATIF_DECISIF = 3

# A7 check (ticket 020). Two readings, because the ticket's is incomplete and that must
# be sayable: A7 put motorised two-wheelers in « voiture » (the `MODE_GROUP` convention
# of `build_mode_choice_dataset.py`) and its PT list carried neither the cable car (34)
# nor employer transport (71).
A7_TC_TICKET = ("31", "32", "33", "37", "38", "39", "41", "42", "43", "51", "52", "53", "54")
A7_TC_COMPLET = A7_TC_TICKET + ("34", "71")
A7_VOITURE_LARGE = ("21", "22", "61", "62", "81", "82", "13", "14", "15", "16", "19", "20")
A7_VOITURE_STRICT = ("21", "22", "61", "62", "81", "82")
A7_VELO = ("10", "11", "12", "17", "18")
A7_ATTENDU = {"n": 770, "collectif": 760, "voiture": 10}
A7_VELO_ATTENDU = {"n": 58, "collectif": 58, "voiture": 0}


# ─────────────────────────────────────────────────────────────────────────────────
# Reading the microdata
# ─────────────────────────────────────────────────────────────────────────────────

def charger_table_publiee(chemin: Path = HIERARCHIE_PUBLIEE) -> dict:
    """The published table (report p. 53), read from its frozen transcription.

    It is NOT copied here: a second transcription would be a fifth mode table, and
    ticket 022 exists to remove them. It is validated though — 36 orders, from 1 to 36,
    no gap — because a truncated table would classify without error.
    """
    if not chemin.exists():
        raise SystemExit(
            f"Published table missing: {chemin}. It is the transcription of the appendix "
            "« Hiérarchie des modes » of the AUAT/CEREMA report; without it the order "
            "would be assumed.")
    doc = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    version = str(doc.get("version") or "")
    if version != VERSION_PUBLIEE_ATTENDUE:
        raise SystemExit(f"[ALARME] {chemin.name} at version {version!r}, expected "
                         f"{VERSION_PUBLIEE_ATTENDUE!r}: reread the table before exporting.")
    ordres = sorted(int(m["ordre"]) for m in doc["modes"])
    if ordres != list(range(1, 37)):
        raise SystemExit(f"[ALARME] {chemin.name} does not carry the 36 orders 1 to 36 "
                         f"({len(ordres)} found): the hierarchy would be truncated.")
    return doc


def rang_par_code() -> dict[str, int]:
    """`T3`/`MODP` code → published order. A code missing from the mapping has no rank."""
    return {code: ordre for ordre, codes in CODES_PAR_ORDRE.items() for code in codes}


def famille_par_code() -> dict[str, str]:
    """`T3`/`MODP` code → simulation family (codes without a counterpart excluded)."""
    par_ordre = {ordre: f["famille"] for f in FAMILLES for ordre in f["ordres"]}
    return {code: par_ordre[ordre]
            for ordre, codes in CODES_PAR_ORDRE.items() if ordre in par_ordre
            for code in codes}


def verifier_raccord(table_publiee: dict, depl: pd.DataFrame) -> dict:
    """Is the labels ↔ codes mapping complete and consistent? Three checks.

    Without them, a forgotten code would have no rank, its pairs would be silent, and the
    « 53 out of 53 » agreement would read as a confirmation while it would be a silence.
    """
    codes_plats = [c for codes in CODES_PAR_ORDRE.values() for c in codes]
    doublons = sorted({c for c in codes_plats if codes_plats.count(c) > 1})
    manquants_dans_ordre = sorted(set(CODES_PAR_ORDRE) - set(range(1, 37)))
    observes = set(depl["MODP"]) | {c for jeu in depl["jambes"].dropna() for c in jeu}
    sans_rang = sorted(observes - set(codes_plats))
    # `correspondance_simulation` of the published table: each family's rank must be
    # the same on both sides. This is the check that keeps the two files from drifting.
    correspondance = table_publiee.get("correspondance_simulation") or {}
    ordres_par_famille = {f["famille"]: set(f["ordres"]) for f in FAMILLES}
    desaccords, conventions = {}, {}
    for mode, node in correspondance.items():
        ordre_publie = int(node["ordre"])
        couverts = ordres_par_famille.get(mode)
        if couverts is None:
            desaccords[mode] = "mode de `correspondance_simulation` absent des FAMILLES"
        elif ordre_publie not in couverts:
            desaccords[mode] = (f"la table publiée le met à l'ordre {ordre_publie}, que la "
                                f"famille {mode} ne couvre pas ({sorted(couverts)})")
        elif ordre_publie != min(couverts):
            # Not a disagreement: a family's rank is the SMALLEST of its orders, whereas
            # the published table names the order of the TYPICAL case (car driver 19,
            # personal bicycle 28). The family also covers smaller orders — taxi 14,
            # self-service e-bike 23 — that share the same leg mode. The gap is
            # declared so that it does not read as an error.
            conventions[mode] = (f"ordre typique publié {ordre_publie} ; rang de la "
                                 f"famille {min(couverts)} (le plus petit de "
                                 f"{sorted(couverts)}, même mode de jambe)")
    if doublons or manquants_dans_ordre or sans_rang:
        raise SystemExit(
            f"[ALARME] Inconsistent labels ↔ ProGEDO codes mapping — doublons={doublons}, "
            f"ordres hors 1-36={manquants_dans_ordre}, codes observés sans rang={sans_rang}. "
            "A code without a rank makes its pairs silent and the agreement would read as a "
            "confirmation.")
    if desaccords:
        raise SystemExit(
            f"[ALARME] This script and {HIERARCHIE_PUBLIEE.name} disagree on "
            f"the rank of some modes: {desaccords}. The two files have drifted — "
            "exactly what ticket 022 removes.")
    return {"codes_raccordes": len(codes_plats),
            "codes_observes_dans_les_microdonnees": len(observes),
            "modes_confrontes_a_correspondance_simulation": sorted(correspondance),
            "ecarts_de_convention": conventions}


def charger(data: Path = DATA) -> pd.DataFrame:
    """One trip per row, with the set of codes of its legs.

    Trip key: `(ZFD, ECH, PER, NDEP)` on the trips side, `(ZFT, ECH, PER, NDEP)` on the
    legs side — `ECH` alone is not unique from one fine zone to another, the same remark
    as `export_bike_ownership` and `export_terminal_time`.
    """
    depl = pd.read_csv(data / "Toulouse_2023_std_depl.csv", dtype=str, low_memory=False)
    traj = pd.read_csv(data / "Toulouse_2023_std_traj.csv", dtype=str, low_memory=False)
    for frame in (depl, traj):
        for colonne in frame.columns:
            frame[colonne] = frame[colonne].astype(str).str.strip()
    depl["cle"] = depl.ZFD + "|" + depl.ECH + "|" + depl.PER + "|" + depl.NDEP
    traj["cle"] = traj.ZFT + "|" + traj.ECH + "|" + traj.PER + "|" + traj.NDEP
    if depl["cle"].duplicated().any():
        raise ValueError("Non-unique trip key — the join of the legs would be wrong.")
    jambes = traj.groupby("cle")["T3"].apply(frozenset)
    depl = depl.set_index("cle")
    depl["jambes"] = jambes
    logger.info("Trips: %d, of which %d with detailed legs; legs: %d",
                len(depl), int(depl["jambes"].notna().sum()), len(traj))
    return depl


# ─────────────────────────────────────────────────────────────────────────────────
# The checks
# ─────────────────────────────────────────────────────────────────────────────────

def controle_marche_residuelle(depl: pd.DataFrame) -> dict:
    """Is walking the *measured* rank 36, or only the copied rank 36?

    Two facts must hold together: no leg is coded « marche à pied » (`T3` carries no
    `01` — walking access is a `T2`/`T6` duration, not a leg), and `MODP = 01`
    designates exactly the trips with no mechanised leg.
    """
    sans = depl[depl["jambes"].isna()]
    avec = depl[depl["jambes"].notna()]
    trajets_marche = sum(1 for jeu in avec["jambes"] if "01" in jeu)
    return {
        "question": "La marche à pied est-elle le résidu, c'est-à-dire le dernier rang ?",
        "deplacements_sans_trajet": int(len(sans)),
        "sans_trajet_dont_modp_marche": int((sans["MODP"] == "01").sum()),
        "deplacements_avec_trajet": int(len(avec)),
        "avec_trajet_dont_modp_marche": int((avec["MODP"] == "01").sum()),
        "trajets_codes_marche": int(trajets_marche),
        "verdict": ("rang mesuré : MODP=01 ⇔ aucun trajet mécanisé"
                    if int((sans["MODP"] == "01").sum()) == len(sans)
                    and int((avec["MODP"] == "01").sum()) == 0
                    else "SUSPECT — la marche n'est pas le résidu, la lecture est fausse"),
    }


def controle_a7(depl: pd.DataFrame, avec: tuple[str, ...], collectif: tuple[str, ...],
                autre: tuple[str, ...]) -> dict:
    """Replay of axis A7: trips mixing `avec` and `collectif`, and their `MODP`."""
    detail = depl[depl["jambes"].notna()]
    mixtes = detail[[bool(jeu & set(avec)) and bool(jeu & set(collectif))
                     for jeu in detail["jambes"]]]
    modp = Counter(mixtes["MODP"])
    n_collectif = sum(v for k, v in modp.items() if k in collectif)
    n_autre = sum(v for k, v in modp.items() if k in autre)
    # A7 counted only two outcomes: « transports collectifs » or « voiture ». Its PT
    # column is thus « everything that is not car », including a MODP outside its own
    # PT list (cable car, employer transport). Both readings are published: the gap
    # between them explains the ticket's 760 against 759 measured under strict
    # membership.
    return {"n": int(len(mixtes)), "collectif": int(n_collectif), "autre": int(n_autre),
            "collectif_au_sens_non_autre": int(len(mixtes) - n_autre),
            "modp": dict(sorted(modp.items()))}


# ─────────────────────────────────────────────────────────────────────────────────
# The measurement: dominance per pair of codes
# ─────────────────────────────────────────────────────────────────────────────────

def dominance(depl: pd.DataFrame) -> list[dict]:
    """For each pair of co-present codes: which one the survey keeps as `MODP`.

    An observation is *informative* for the pair `(a, b)` when the trip carries both
    and its `MODP` is `a` or `b`. Otherwise a third mode won: the observation informs the
    pairs of that third mode, not this one.
    """
    detail = depl[depl["jambes"].notna()]
    codes = sorted({code for jeu in detail["jambes"] for code in jeu})
    jeux = list(detail["jambes"])
    modes_principaux = list(detail["MODP"])
    resultats = []
    for a, b in itertools.combinations(codes, 2):
        contient = informatif = gagne_a = gagne_b = 0
        for jeu, modp in zip(jeux, modes_principaux):
            if a not in jeu or b not in jeu:
                continue
            contient += 1
            if modp == a:
                informatif += 1
                gagne_a += 1
            elif modp == b:
                informatif += 1
                gagne_b += 1
        if not contient:
            continue
        resultats.append({"a": a, "b": b, "contient": contient, "informatif": informatif,
                          "gagne_a": gagne_a, "gagne_b": gagne_b})
    return resultats


def confronter(paires: list[dict], rangs: dict[str, int]) -> dict:
    """Does the measurement confirm the published order? Lists pairs and exceptions."""
    testees = conformes = 0
    exceptions, non_tranchees = [], []
    for paire in paires:
        a, b, ga, gb = paire["a"], paire["b"], paire["gagne_a"], paire["gagne_b"]
        if a not in rangs or b not in rangs:
            continue
        if paire["informatif"] < MIN_INFORMATIF_DECISIF:
            non_tranchees.append({**paire, "rang_a": rangs[a], "rang_b": rangs[b]})
            continue
        testees += 1
        # Smallest rank = wins. The published order thus expects `attendu` wins on side a.
        attendu_a = rangs[a] < rangs[b]
        observe_a = ga > gb
        contre = gb if attendu_a else ga
        if attendu_a == observe_a:
            conformes += 1
        if contre:
            exceptions.append({**paire, "rang_a": rangs[a], "rang_b": rangs[b],
                               "contre_l_ordre_publie": int(contre)})
    return {
        "paires_testees": testees,
        "paires_conformes": conformes,
        "paires_non_tranchees_faute_d_effectif": len(non_tranchees),
        "seuil_informatif_decisif": MIN_INFORMATIF_DECISIF,
        "observations_informatives": sum(p["informatif"] for p in paires),
        "exceptions": sorted(exceptions, key=lambda e: -e["contre_l_ordre_publie"]),
        "non_tranchees": sorted(non_tranchees, key=lambda p: -p["contient"])[:40],
    }


def dominance_familles(depl: pd.DataFrame, familles: dict[str, str]) -> list[dict]:
    """The same measurement, aggregated to simulation families — the matrix to publish."""
    detail = depl[depl["jambes"].notna()]
    jeux = [frozenset(familles[c] for c in jeu if c in familles) for jeu in detail["jambes"]]
    principaux = [familles.get(m) for m in detail["MODP"]]
    noms = sorted({f for jeu in jeux for f in jeu})
    resultats = []
    for a, b in itertools.combinations(noms, 2):
        contient = gagne_a = gagne_b = 0
        for jeu, principal in zip(jeux, principaux):
            if a not in jeu or b not in jeu:
                continue
            contient += 1
            if principal == a:
                gagne_a += 1
            elif principal == b:
                gagne_b += 1
        if not contient:
            continue
        resultats.append({"a": a, "b": b, "contient": contient,
                          "informatif": gagne_a + gagne_b,
                          "gagne_a": gagne_a, "gagne_b": gagne_b})
    return sorted(resultats, key=lambda r: -r["informatif"])


# ─────────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────────

def empreintes(data: Path = DATA) -> dict[str, str]:
    fichiers = ("Toulouse_2023_std_depl.csv", "Toulouse_2023_std_traj.csv")
    table = {nom: hashlib.sha256((data / nom).read_bytes()).hexdigest() for nom in fichiers}
    table[HIERARCHIE_PUBLIEE.name] = hashlib.sha256(
        HIERARCHIE_PUBLIEE.read_bytes()).hexdigest()
    return table


def construire(depl: pd.DataFrame) -> dict:
    table_publiee = charger_table_publiee()
    raccord = verifier_raccord(table_publiee, depl)
    rangs = rang_par_code()
    familles = famille_par_code()

    paires = dominance(depl)
    accord = confronter(paires, rangs)
    if accord["observations_informatives"] < MIN_PAIRES_INFORMATIVES:
        raise SystemExit(
            f"[ALARME] only {accord['observations_informatives']} informative observations "
            f"(threshold {MIN_PAIRES_INFORMATIVES}): the published order would be declared "
            "compliant for lack of measurement. Check the legs ↔ trips join.")

    marche = controle_marche_residuelle(depl)
    if marche["verdict"].startswith("SUSPECT"):
        raise SystemExit(f"[ALARME] {marche['verdict']}")

    # Rank of a family = the SMALLEST published order it covers. Two families cannot
    # share an order: that would be a silent ambiguity.
    ordre_publie_de_famille = {f["famille"]: min(f["ordres"]) for f in FAMILLES}
    if len(set(ordre_publie_de_famille.values())) != len(FAMILLES):
        raise SystemExit(f"[ALARME] Two families share a rank: "
                         f"{ordre_publie_de_famille}")
    ordre = sorted(ordre_publie_de_famille, key=ordre_publie_de_famille.get)

    jambes: dict[str, int] = {}
    for index, nom in enumerate(ordre, start=1):
        for jambe in next(f for f in FAMILLES if f["famille"] == nom)["jambes"]:
            jambes[jambe] = index

    a7_ticket = controle_a7(depl, A7_VOITURE_LARGE, A7_TC_TICKET, A7_VOITURE_LARGE)
    a7_complet = controle_a7(depl, A7_VOITURE_STRICT, A7_TC_COMPLET, A7_VOITURE_STRICT)
    a7_velo = controle_a7(depl, A7_VELO, A7_TC_COMPLET, A7_VELO)
    a7_velo_ticket = controle_a7(depl, A7_VELO, A7_TC_TICKET, A7_VELO)

    detail = depl[depl["jambes"].notna()]
    n_familles = [len({familles[c] for c in jeu if c in familles}) for jeu in detail["jambes"]]

    return {
        "version": VERSION,
        "titre": "Hiérarchie des modes — mode principal d'un déplacement, EMC² Toulouse 2023",
        "avertissement": (
            "GELÉ : produit par scripts/progedo_logit/export_mode_hierarchy.py. Ne pas "
            "éditer à la main. L'ordre vient du rapport publié (p. 53) ; les effectifs "
            "viennent des microdonnées ProGEDO lil-1750 (accès restreint)."),
        # The machine contract: family → rank, leg → rank, and the two label tables.
        "ordre_familles": ordre,
        "rang_famille": {nom: index for index, nom in enumerate(ordre, start=1)},
        "rang_jambe": dict(sorted(jambes.items(), key=lambda kv: (kv[1], kv[0]))),
        "libelle_journal": {f["famille"]: f["libelle_journal"] for f in FAMILLES},
        "mode_canonique": {f["famille"]: f["mode_canonique"] for f in FAMILLES},
        "jambes_par_famille": {f["famille"]: list(f["jambes"]) for f in FAMILLES},
        "codes_emc2_par_famille": {
            nom: sorted(c for c, f in familles.items() if f == nom) for nom in ordre},
        "source_publiee": {
            "rapport": RAPPORT, "url": RAPPORT_URL, "page": 53,
            "renvoi": "« un seul des modes est pris en compte […] : c'est le "
                      "« mode principal », qui découle d'une hiérarchisation des modes "
                      "définie au niveau national » (p. 12)",
            "transcription": {
                "fichier": str(HIERARCHIE_PUBLIEE.relative_to(ROOT)),
                "version": table_publiee["version"],
                "regle": table_publiee.get("regle"),
            },
            # The 36 orders, as the transcription carries them, with their mapping to the
            # ProGEDO codebook and the simulated family that covers them.
            "ordre": [
                {"rang": int(m["ordre"]), "libelle": m["libelle"],
                 "categorie_publiee": m.get("categorie"),
                 "codes": list(CODES_PAR_ORDRE.get(int(m["ordre"]), ())),
                 "famille": next((f["famille"] for f in FAMILLES
                                  if int(m["ordre"]) in f["ordres"]), None)}
                for m in sorted(table_publiee["modes"], key=lambda m: int(m["ordre"]))],
            "rangs_sans_contrepartie_simulee": [
                {"rang": int(m["ordre"]), "libelle": m["libelle"],
                 "codes": list(CODES_PAR_ORDRE.get(int(m["ordre"]), ()))}
                for m in sorted(table_publiee["modes"], key=lambda m: int(m["ordre"]))
                if not any(int(m["ordre"]) in f["ordres"] for f in FAMILLES)],
            "raccord_codebook_progedo": raccord,
        },
        "mesure": {
            "unite": "déplacement de l'enquête, comptage NON pondéré (règle de codage, "
                     "pas grandeur de population)",
            "n_deplacements": int(len(depl)),
            "n_deplacements_detailles": int(len(detail)),
            "n_familles_par_deplacement": {str(k): int(v) for k, v in
                                           sorted(Counter(n_familles).items())},
            "accord_avec_l_ordre_publie": accord,
            "matrice_familles": dominance_familles(depl, familles),
            "paires_de_codes": sorted(paires, key=lambda p: -p["informatif"]),
        },
        "controles": {
            "marche_residuelle": marche,
            "a7_voiture_tc_convention_ticket_020": {
                **a7_ticket, "attendu": A7_ATTENDU,
                "convention": "voiture au sens MODE_GROUP (deux-roues motorisés inclus) ; "
                              "liste TC sans le téléphérique (34) ni le transport "
                              "d'employeur (71)",
                "verdict": ("rejeu exact"
                            if (a7_ticket["n"] == A7_ATTENDU["n"]
                                and a7_ticket["autre"] == A7_ATTENDU["voiture"]
                                and a7_ticket["collectif_au_sens_non_autre"]
                                == A7_ATTENDU["collectif"])
                            else "ÉCART — la convention d'A7 n'est pas reproduite")},
            "a7_voiture_tc_listes_completes": {
                **a7_complet,
                "convention": "voiture stricte (deux-roues motorisés à part) ; liste TC "
                              "complète, téléphérique et transport d'employeur inclus"},
            # ⚠ A7's two figures were NOT computed with the same PT list: the 770 car +
            # PT trips are obtained without the cable car or employer transport, the
            # 58 bicycle + PT include them. Both readings are published, the only way to
            # cross-check the ticket without copying it.
            "a7_velo_tc": {**a7_velo, "attendu": A7_VELO_ATTENDU,
                           "convention": "liste TC complète",
                           "verdict": ("rejeu exact"
                                       if a7_velo["n"] == A7_VELO_ATTENDU["n"]
                                       else "ÉCART — convention non reproduite")},
            "a7_velo_tc_convention_ticket_020": {
                **a7_velo_ticket,
                "convention": "liste TC sans téléphérique ni transport d'employeur — "
                              "celle qui rejoue les 770 de voiture + TC"},
        },
        "provenance": {
            "source": "ordre publié (rapport AUAT/CEREMA p. 53, transcrit dans scripts/panel/hierarchie_modes_emc2.yaml) contrôlé sur les "
                      "microdonnées EMC² 2023 (ProGEDO lil-1750), fichiers déplacements "
                      "× trajets, comptage non pondéré",
            "fichiers": empreintes(),
            "gele_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "par": "scripts/progedo_logit/export_mode_hierarchy.py",
        },
    }


def resumer(doc: dict) -> None:
    accord = doc["mesure"]["accord_avec_l_ordre_publie"]
    print(f"\nHierarchy {doc['version']} — {len(doc['ordre_familles'])} simulated families:")
    for index, nom in enumerate(doc["ordre_familles"], start=1):
        codes = doc["codes_emc2_par_famille"][nom]
        print(f"  {index}. {nom:10s} → « {doc['libelle_journal'][nom]:22s} » "
              f"EMC² codes {codes or '—'}")
    print(f"\nMeasurement ↔ published order agreement: {accord['paires_conformes']}/"
          f"{accord['paires_testees']} compliant code pairs, "
          f"{accord['observations_informatives']} informative observations, "
          f"{len(accord['exceptions'])} exception(s), "
          f"{accord['paires_non_tranchees_faute_d_effectif']} undecided pair(s) "
          f"(< {accord['seuil_informatif_decisif']} obs.)")
    for exception in accord["exceptions"][:8]:
        print(f"    exception {exception['a']}+{exception['b']} "
              f"(ranks {exception['rang_a']}/{exception['rang_b']}): "
              f"{exception['contre_l_ordre_publie']} trip(s) against the published order")
    print("\nMatrix of mixed trips, by simulated family "
          "(informative count ≥ 1):")
    print(f"    {'paire':26s} {'contient':>8s} {'inform.':>8s} {'gagnant':>10s}")
    for ligne in doc["mesure"]["matrice_familles"]:
        gagnant = (ligne["a"] if ligne["gagne_a"] > ligne["gagne_b"]
                   else ligne["b"] if ligne["gagne_b"] > ligne["gagne_a"] else "—")
        print(f"    {ligne['a'] + ' / ' + ligne['b']:26s} {ligne['contient']:8d} "
              f"{ligne['informatif']:8d} {gagnant:>10s} "
              f"({ligne['gagne_a']}–{ligne['gagne_b']})")
    a7 = doc["controles"]["a7_voiture_tc_convention_ticket_020"]
    print(f"\nA7 check (ticket 020 convention): n={a7['n']} "
          f"collectif={a7['collectif_au_sens_non_autre']} voiture={a7['autre']} "
          f"— expected {a7['attendu']} → {a7['verdict']}")
    complet = doc["controles"]["a7_voiture_tc_listes_completes"]
    print(f"A7 check (full lists)           : n={complet['n']} "
          f"collectif={complet['collectif']} voiture={complet['autre']}")
    velo = doc["controles"]["a7_velo_tc"]
    print(f"A7 check bike + PT (full list)  : n={velo['n']} "
          f"collectif={velo['collectif_au_sens_non_autre']} vélo={velo['autre']} "
          f"— expected {velo['attendu']} → {velo['verdict']}")
    print(f"Residual walking: {doc['controles']['marche_residuelle']['verdict']}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true",
                        help="measure and compare with the frozen resource, without writing it")
    args = parser.parse_args(argv)

    if not DATA.exists():
        raise SystemExit(
            f"Microdata missing: {DATA}. They are restricted-access (ProGEDO/ADISP "
            "lil-1750). The frozen resource is enough to run the repository; this script "
            "only serves to reproduce it.")

    doc = construire(charger())
    resumer(doc)

    if args.check:
        if not args.out.exists():
            raise SystemExit(f"[ALARME] Frozen resource missing: {args.out}")
        gele = json.loads(args.out.read_text(encoding="utf-8"))
        ecarts = [cle for cle in ("ordre_familles", "rang_famille", "rang_jambe",
                                  "libelle_journal", "mode_canonique")
                  if gele.get(cle) != doc[cle]]
        # The SUBSTANCE of the published table, not its hash: a comment edited in the
        # YAML must not require a re-export (which would need the restricted-access
        # microdata), but a moved order must force it.
        for cle in ("transcription", "ordre"):
            if (gele.get("source_publiee") or {}).get(cle) != doc["source_publiee"][cle]:
                ecarts.append(f"source_publiee.{cle}")
        if ecarts:
            raise SystemExit(f"[ALARME] The frozen resource diverges from the measurement "
                             f"on {ecarts} — re-export it.")
        print(f"\n--check: the frozen resource {args.out.name} matches the measurement.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\nWrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
