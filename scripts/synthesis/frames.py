"""Data adapters: each strand produces the same "decision frame".

The principle of the whole page fits in one sentence: the three strands are made
comparable by bringing them back to the **same table** — one row per (decision, mode
considered), carrying a probability mass — then by applying to them the **same
scoring formula** (``formule_score.metrics``, the calibration engine's, repatriated).

Frame columns: ``agent_id``, ``mode_cat``, ``weight``, ``genre``,
``age_cat``, ``occupation``, ``motif``, ``dist_cat`` (+ ``lieu_residence``,
outside the composite score).
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Optional

import yaml

from mobility_core.housing_type import (
    MODALITY_KEYS as HOUSING_MODALITY_KEYS,
    REFERENCE_KEYS as HOUSING_REFERENCE_KEYS,
    key_for as housing_key_for,
)
from mobility_core.population_reference import (
    COURONNE_DEPUIS_FR,
    COURONNE_VERS_CEREMA,
    OUT_OF_PERIMETER,
    OUT_OF_PERIMETER_FR,
)

from .sources import REPO_ROOT, Manifest, probe

logger = logging.getLogger("synthesis.frames")

# ── Mapping conventions (moves.csv → EMC² categories) ────────────────────────

# Probability column → EMC² category. Train is filed with public transport
# (it is a collective mode); motorised two-wheelers and « autres » leave
# the scored scope, as in the EMC² reference where they form the
# « autres » residual. The mass thus set aside is measured and shown.
PROBA_COLUMNS = {
    "P(Marche) %": "marche",
    "P(Vélo) %": "velo",
    "P(Voiture Privée) %": "voiture",
    "P(Transports_collectifs) %": "transports_collectifs",
    "P(Train) %": "transports_collectifs",
    "P(Deux-roues motorisé) %": "autres",
    "P(Autres modes) %": "autres",
}

CHOSEN_MODE_MAP = {
    "Marche": "marche",
    "Vélo": "velo",
    "Voiture Privée": "voiture",
    "Transports_collectifs": "transports_collectifs",
    "Train": "transports_collectifs",
    "Deux-roues motorisé": "autres",
    "Autres modes": "autres",
}


def parse_offered_modes(value: str) -> list[str]:
    """« Modes proposés au LLM » → EMC² categories, deduplicated, in order.

    The column lists **one label per OTP itinerary**, separated by ` | `: several
    options often share a mode (six itineraries, four of them by public
    transport). It is the set of modes that matters here, not their multiplicity.

    A label outside the table (unexpected mode) is ignored rather than filed in
    « autres »: « autres » is a category of the EMC² reference, not a catch-all
    for values that cannot be read.
    """
    out: list[str] = []
    for label in (value or "").split("|"):
        cat = CHOSEN_MODE_MAP.get(label.strip())
        if cat is not None and cat not in out:
            out.append(cat)
    return out


def departure_hour(value: str) -> Optional[int]:
    """Departure hour (0-23) from the log's « Heure de départ ».

    The log writes a `YYYY-MM-DD HH:MM:SS` timestamp rendered from the **simulated**
    clock: the hour read is indeed the local hour of the trip, the one the
    survey declares in D4.
    """
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(text.split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        return None

# `Motifs de déplacement` mixes translated and raw labels depending on the source;
# home / leisure / other have no EMC² equivalent and leave the dimension.
MOTIF_MAP = {
    "Travail": "travail", "work": "travail",
    "Etude": "etudes", "Étude": "etudes", "education": "etudes",
    "Achats": "achats", "shop": "achats",
    "Accompagnement": "accompagnement", "escort": "accompagnement",
}

# Displayed dimensions. `scored`: enters the comparable composite score.
DIMENSIONS = [
    {"key": "age", "column": "age_cat", "cerema": "Age", "label": "Âge",
     "kind": "ordinal", "scored": True},
    {"key": "distance", "column": "dist_cat", "cerema": "distance", "label": "Distance",
     "kind": "ordinal", "scored": True},
    {"key": "genre", "column": "genre", "cerema": "genre", "label": "Genre",
     "kind": "nominal", "scored": True},
    {"key": "occupation", "column": "occupation", "cerema": "occupation",
     "label": "Occupation", "kind": "nominal", "scored": True},
    {"key": "motif", "column": "motif", "cerema": "motif_deplacement",
     "label": "Motif de déplacement", "kind": "nominal", "scored": True},
    {"key": "lieu_residence", "column": "lieu_residence", "cerema": "lieu_residence",
     "label": "Lieu de résidence", "kind": "nominal", "scored": False},
    {"key": "type_logement", "column": "type_logement", "cerema": "type_logement",
     "label": "Type de logement", "kind": "nominal", "scored": False},
]

MODES = ["marche", "voiture", "velo", "transports_collectifs"]
MODE_LABELS = {"marche": "Marche", "voiture": "Voiture", "velo": "Vélo",
               "transports_collectifs": "Transports collectifs"}
MODE_COLORS = {"marche": "#00CCCC", "voiture": "#EE4444",
               "velo": "#8844BB", "transports_collectifs": "#22AA44"}

_AGE_BUCKETS = [(9, "5-9"), (14, "10-14"), (19, "15-19"), (24, "20-24"),
                (29, "25-29"), (34, "30-34"), (39, "35-39"), (44, "40-44"),
                (49, "45-49"), (54, "50-54"), (59, "55-59"), (64, "60-64"),
                (69, "65-69"), (74, "70-74")]
_DIST_BUCKETS = [(1, "0-1km"), (2, "1-2km"), (5, "2-5km"),
                 (10, "5-10km"), (20, "10-20km"), (50, "20-50km")]

# Persona `main_occupation` label → identifier of `cerema_values.yaml`.
#
# The KEYS carry both vocabularies since ticket 074: the v6 cohort sets English
# labels (served to the model), the earlier cohorts and their archived runs
# carry the French ones. Knowing only one vocabulary would silently lose the whole
# « occupation » dimension on the other: the rows would not be missing, they would fall
# into « hors référentiel ».
#
# The VALUES do not move: they are the survey identifiers, which `prompt_calibration/`
# also reads. They cite the source, they do not describe it.
OCCUPATION_MAP = {
    # v6 and later
    "Pupil (up to Baccalaureate)": "scolaire",
    "Student": "etudiant",
    "Full-time worker": "actif_temps_plein",
    "Part-time worker": "actif_temps_partiel",
    "Unemployed / job seeker": "chomeur_recherche_emploi",
    "Homemaker": "personne_au_foyer",
    "Retired": "Retraité",
    # v5 and earlier
    "Scolaire (jusqu'au Bac)": "scolaire",
    "Étudiant": "etudiant",
    "Travail à plein temps": "actif_temps_plein",
    "Travail à temps partiel": "actif_temps_partiel",
    "Chômeur/recherche d'emploi": "chomeur_recherche_emploi",
    "Personne au foyer": "personne_au_foyer",
    "Retraité": "Retraité",
}


# Key of the out-of-scope modality. ASCII and unaccented, like the keys of
# `cerema_values.yaml` (`1ere_couronne`): the raw output of `normalize_place`
# would give `hors_périmètre`, which would join nothing and vanish without a word.
OUT_OF_PERIMETER_KEY = "hors_perimetre"

# Log `Lieu de résidence` label → identifier of `cerema_values.yaml`.
#
# WHY THIS TABLE EXISTS (ticket 082). Until then the translation came down to a
# `replace(" ", "_")`, which was enough as long as the log wrote « 1ere couronne ».
# Since the switch to English (ticket 074) it writes « 1st ring », which this replacement
# turns into `1st_ring` — a key the reference does not break down. The three rings
# outside Toulouse thus left the score pages of EVERY v6 run **without a
# single row missing**: they were all counted, all filed under « hors
# référentiel », and the dimension published only one stratum out of four.
#
# The table is DERIVED from `mobility_core.population_reference`, never copied: that module
# is the single declaration of the ring modalities, shared with population
# generation and the log. A fourth copy would end up diverging with nothing
# signalling it — which is exactly what `normalize_housing` avoids by going through
# `mobility_core.housing_type`.
#
# The VALUES stay French: they are the survey identifiers, which
# `prompt_calibration/` also reads, and which the model never sees.
PLACE_MAP: dict[str, str] = {
    # v6 and later (canonical modality, served to the model in the persona narrative)
    **COURONNE_VERS_CEREMA,
    # v5 and earlier, by composition: « 1ere couronne » → « 1st ring » → `1ere_couronne`
    **{ancien: COURONNE_VERS_CEREMA[canonique]
       for ancien, canonique in COURONNE_DEPUIS_FR.items()},
    # Re-reading an already normalised log (page regenerated from a rewritten trace):
    # the key is its own antecedent, as in `normalize_housing`.
    **{cle: cle for cle in COURONNE_VERS_CEREMA.values()},
}

# The two spellings of the out-of-scope modality. It is NOT a ring: it has
# no per-zone target (ticket 021), and the second member returned by `normalize_place`
# says so. It is therefore counted, not joined.
PLACE_OUT_OF_PERIMETER_LABELS = frozenset(
    {OUT_OF_PERIMETER, OUT_OF_PERIMETER_FR, OUT_OF_PERIMETER_KEY}
)

# The keys the old `replace(" ", "_")` produced from an English label: `1st_ring`
# and its two siblings. They denote nothing in the reference, and a `scores.json` that
# carries one was computed BEFORE ticket 082 — its « lieu de résidence » dimension is
# missing three strata. It is the signature that lets `experiences.score` declare
# these files stale and force their full recomputation, once.
#
# Derived, not enumerated: if a label enters `PLACE_MAP` tomorrow, its ghost form
# enters here the same day. The v5 labels exclude themselves — « 1ere couronne »
# with underscores gives `1ere_couronne`, which IS the survey key, and those scores are right.
PLACE_CLES_FANTOMES = frozenset(
    libelle.replace(" ", "_") for libelle in PLACE_MAP
    if libelle.replace(" ", "_") not in set(PLACE_MAP.values())
)

# Residence and housing labels that no table knows, counted per label.
# They no longer go through: an unknown label producing a ghost key would bring back
# the ticket 082 failure, silent and invisible on the pages.
PLACES_INCONNUES: Counter = Counter()
LOGEMENTS_INCONNUS: Counter = Counter()
OCCUPATIONS_INCONNUES: Counter = Counter()

# Purposes without an EMC² equivalent, set aside from the dimension and not reported: they
# are normal log values, not labels to translate. Anything neither in
# `MOTIF_MAP` nor here is, on the other hand, unexpected, and is reported.
MOTIFS_HORS_ENQUETE = frozenset({"home", "leisure", "other"})
MOTIFS_INCONNUS: Counter = Counter()


def _signaler_libelle_inconnu(compteur: Counter, colonne: str, libelle: str,
                              consequence: str) -> None:
    """Counts an unreadable label, and reports it as ERROR on its FIRST occurrence.

    On rising edge: a 3,000-line log carries the same unknown label 3,000
    times, and three thousand identical alarm lines would drown the log instead of
    alerting it. The counter, for its part, keeps the volume.
    """
    premiere = compteur[libelle] == 0
    compteur[libelle] += 1
    if premiere:
        logger.error(
            f"[ALARME] Unreadable label in column « {colonne} »: {libelle!r} — "
            f"{consequence} Mapping table to complete in "
            f"`scripts/synthesis/frames.py` (ticket 082)."
        )


def reinitialiser_compteurs_libelles() -> None:
    """Resets the unknown-label counters (rising edge of the alarms).

    Called between two independent reads — the tests, and the global rescoring that
    chains dozens of runs: without this reset, the second
    run carrying the same unknown label would no longer say anything.
    """
    for compteur in (PLACES_INCONNUES, LOGEMENTS_INCONNUS,
                     OCCUPATIONS_INCONNUES, MOTIFS_INCONNUS):
        compteur.clear()

# Name of the row carrying a dimension's mass outside the reference data. It has neither
# target nor L1: it exists so that "excluded from the targets" is never confused with
# "non-existent". `global_view` does the same with its mass outside scored modes.
OFF_REFERENCE_ROW = "— hors référentiel —"

# The keys of cerema_values.yaml are identifiers, not labels: they are made
# readable for display without ever touching the keys themselves.
CAT_LABELS = {
    "scolaire": "Scolaire", "etudiant": "Étudiant",
    "actif_temps_plein": "Actif à temps plein",
    "actif_temps_partiel": "Actif à temps partiel",
    "chomeur_recherche_emploi": "Chômeur / recherche d'emploi",
    "personne_au_foyer": "Personne au foyer", "Retraité": "Retraité",
    "travail": "Travail", "etudes": "Études", "achats": "Achats",
    "accompagnement": "Accompagnement",
    "Toulouse": "Toulouse", "1ere_couronne": "1re couronne",
    "2eme_couronne": "2e couronne", "3eme_couronne": "3e couronne",
    "individuel_isole": "Individuel isolé", "individuel_accole": "Individuel accolé",
    "petit_habitat_collectif": "Petit habitat collectif",
    "grand_habitat_collectif": "Grand habitat collectif",
    "plus_50km": "plus de 50 km", "75-130": "75 ans et plus",
    # Copy of the canonical modality (`population_reference.OUT_OF_PERIMETER`), not a
    # rewording: it is the same string the log writes and the trace archives.
    OUT_OF_PERIMETER_KEY: OUT_OF_PERIMETER,
}


def pretty_cat(cat: str) -> str:
    return CAT_LABELS.get(cat, str(cat))


def age_to_cat(age: float) -> Optional[str]:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return None
    for bp, label in _AGE_BUCKETS:
        if a <= bp:
            return label
    return "75-130"


def distance_to_cat(km: float) -> Optional[str]:
    try:
        d = float(km)
    except (TypeError, ValueError):
        return None
    for bp, label in _DIST_BUCKETS:
        if d < bp:
            return label
    return "plus_50km"



def normalize_place(value: str) -> tuple[Optional[str], bool]:
    """Log « Lieu de résidence » → EMC² key, and whether it is in the reference.

    The translation goes through `PLACE_MAP`, which knows both vocabularies: the English
    served to the model since ticket 074 (« 1st ring ») and the French of the earlier
    cohorts (« 1ere couronne »). Both give `1ere_couronne`, the survey
    key, and an archived run reads back like a run of the day.

    Since ticket 021 the column also carries `hors périmètre`: a known home,
    located outside the 453 municipalities of the survey. It is not a ring — filing it as
    3rd made us publish a stratum where 76 % of the residents were not in the survey —,
    so it has **no target** per zone, and the second member of the pair says it will
    join no reference row. That is what lets it be COUNTED rather than
    seen vanishing, exactly as `normalize_housing` does with the
    « Autres » modality.

    **An unknown label no longer goes through** (ticket 082). The old version returned
    `text.replace(" ", "_")` and its second member `True`: any label produced a
    valid-looking key, which the reference did not break down and the page filed
    silently under « hors référentiel ». It now returns `(None, False)`, it is
    counted per label, and the first occurrence raises an `[ALARME]`.
    """
    text = (value or "").strip()
    if not text:
        return None, False
    if text in PLACE_OUT_OF_PERIMETER_LABELS:
        return OUT_OF_PERIMETER_KEY, False
    key = PLACE_MAP.get(text)
    if key is None:
        _signaler_libelle_inconnu(
            PLACES_INCONNUES, "Lieu de résidence", text,
            "la décision sort de la dimension « lieu de résidence » au lieu d'être "
            "comparée à la cible de sa couronne.",
        )
        return None, False
    return key, True


def normalize_housing(value: str) -> tuple[Optional[str], bool]:
    """Log « Type de logement » → EMC² key, and whether it is in the reference.

    The log writes the survey **label** (« Petit habitat collectif »); the
    reference indexes it by key (`petit_habitat_collectif`). The mapping comes from
    `mobility_core.housing_type`, single declaration of the modalities, shared with
    population generation and the log — three independent copies would end up
    diverging with nothing signalling it.

    Returns `(key, referenced)`. `autres` exists in the survey but not in the
    published EMC² breakdown: the key is returned anyway, and the second member says
    it will join no reference row — that is what lets it be counted
    rather than seen vanishing silently.
    """
    text = (value or "").strip()
    if not text:
        return None, False
    key = housing_key_for(text)
    if key is None:
        # Already a key (re-reading a log written differently), otherwise unknown.
        key = text if text in HOUSING_MODALITY_KEYS else None
    if key is None:
        # Ticket 082 — the audit of translated columns. `housing_key_for` knows both
        # vocabularies, so this branch does not occur on the known cohorts:
        # it exists so that the day a label changes, we learn it the same day
        # and not by re-reading a three-stratum table six months later.
        _signaler_libelle_inconnu(
            LOGEMENTS_INCONNUS, "Type de logement", text,
            "la décision sort de la dimension « type de logement » au lieu d'être "
            "comparée à la cible de son habitat.",
        )
        return None, False
    return key, key in HOUSING_REFERENCE_KEYS


# ── EMC² reference ───────────────────────────────────────────────────────────

def load_cerema(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def reference_shares(cerema: dict, dim_key: str, cat: Optional[str] = None) -> dict:
    """Target modal shares, renormalised over the 4 scored modes (in %)."""
    parts = cerema.get("parts_modales_2023", {})
    node = parts.get("global", {}) if dim_key == "global" else \
        (parts.get(dim_key) or {}).get(cat, {})
    kept = {m: float(node.get(m, 0.0)) for m in MODES}
    total = sum(kept.values())
    if total <= 0:
        return {}
    return {m: v * 100.0 / total for m, v in kept.items()}


# ── Strand 1: simulation ─────────────────────────────────────────────────────

def resolve_run(manifest: Manifest) -> dict:
    """Locates the run serving as common set, and its files."""
    run_cfg = manifest.get("common_set.run", "experiments/current")
    run_dir = Path(run_cfg)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    info: dict[str, Any] = {"configured": str(run_cfg), "exists": run_dir.exists()}
    if not run_dir.exists():
        return info
    resolved = run_dir.resolve()
    try:
        info["run_id"] = resolved.name
        info["path"] = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        info["run_id"] = resolved.name
        info["path"] = str(resolved)

    moves = resolved / manifest.get("common_set.moves", "moves.csv")
    info["moves"] = manifest.track(
        "common_set.moves", moves, "Décisions modales du run (volet simulation)").to_dict()

    pop_cfg = manifest.get("common_set.population")
    if pop_cfg:
        pop = Path(pop_cfg) if Path(pop_cfg).is_absolute() else REPO_ROOT / pop_cfg
    else:
        candidates = sorted(p for p in resolved.glob("population_*.json")
                            if "checkpoint" not in p.name)
        pop = candidates[0] if candidates else resolved / "population_unknown.json"
    info["population"] = manifest.track(
        "common_set.population", pop,
        "Personas du run (traits, géolocalisation) — socle du volet modèle").to_dict()
    return info


def simulated_day(value: str) -> Optional[str]:
    """Simulated day (``YYYY-MM-DD``) of a row, from « Temps simulé ».

    Same convention as the ``sim_day`` field of ``llm_exchanges.jsonl`` (UTC, cf.
    ``packages/llm_gateway/src/llm_gateway/telemetry/logger.py``): that is what lets strands 1/3 and
    strand 2 cut the run on the same day boundary.
    """
    text = (value or "").strip()
    if not text:
        return None
    try:
        ts = int(float(text))
    except ValueError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def first_simulated_day(path: Path) -> Optional[str]:
    """Smallest simulated day present in a moves.csv.

    Determined by reading, never hard-coded: the reference run can start on
    any day, and a hard-coded date would silently make a whole run
    look empty.
    """
    days = set()
    with Path(path).open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            day = simulated_day(raw.get("Temps simulé") or "")
            if day:
                days.add(day)
    return min(days) if days else None


def attempt_stamp(raw: dict) -> str:
    """« Heure de calcul » of a row: it is what identifies the attempt.

    Two rows of the same (person, activity, simulated day) pair that differ
    only by this timestamp are two attempts of the SAME decision, not two
    decisions.
    """
    return (raw.get("Heure de calcul") or "").strip()


def latest_attempts(raws: list[dict]) -> tuple[list[dict], dict]:
    """Keeps only the most recent attempt of each decision.

    A hot-resumed run (``make run OFFLINE=1 CONT=1``) replays the simulated day
    from t0 in the SAME experiment directory: ``moves.csv`` then carries the same
    (person, activity) pairs twice, one row per attempt, both
    dated the same simulated day. The cut at the first simulated day does not
    separate them — it keeps both — and the score counts them twice. On
    the run resumed on 2026-08-19, 1,469 duplicate rows moved the
    composite score from 24.09 to 24.43, i.e. the order of magnitude of the gains
    calibration seeks to measure.

    The key carries the **simulated day** on top of the (person, activity) pair, and this
    is not a detail: without it, the day-1 decision and its day-2
    repetition — which the sliding planning horizon produces for 442 pairs on this
    run — would pass for two attempts of the same decision. We would then keep
    the day-2 one, which the cut at the first simulated day then drops: the
    decision would vanish from the score instead of entering it once.
    ``model_compare.latest_attempts`` applies the same rule, on the same trap.

    A row without a person or activity identifier cannot be paired with
    any other: it is kept as is, otherwise a log that does not
    carry these columns would collapse onto a single row.
    """
    best: dict[tuple[str, str, Optional[str]], dict] = {}
    unpaired: set[int] = set()
    for raw in raws:
        person = (raw.get("ID Personne") or "").strip()
        activity = (raw.get("ID Activité") or "").strip()
        if not person or not activity:
            unpaired.add(id(raw))
            continue
        key = (person, activity, simulated_day(raw.get("Temps simulé") or ""))
        held = best.get(key)
        # Log without « Heure de calcul » (earlier runs): all timestamps
        # are empty, the comparison is false everywhere and the first row
        # stays. Without a dated attempt, there is no less arbitrary choice.
        if held is None or attempt_stamp(raw) > attempt_stamp(held):
            best[key] = raw
    kept_ids = {id(raw) for raw in best.values()} | unpaired
    kept = [raw for raw in raws if id(raw) in kept_ids]
    stamps = sorted({attempt_stamp(raw)[:10] for raw in raws if attempt_stamp(raw)})
    return kept, {
        "n_dropped": len(raws) - len(kept),
        # Two computation days for a single simulated day: the run was resumed.
        "reprise": len(stamps) > 1,
        "jours_de_calcul": stamps,
    }


def couples_repetes(raws: list[dict]) -> int:
    """How many (person, activity) pairs appear on MORE than one simulated day.

    It is the only thing the first-day cut is there to remove (ticket 057, R2).
    A log that carries none has nothing to cut: cutting anyway removes unique
    trips from it, and that is what emptied the scope of its morning departures.

    A row without a person or activity identifier pairs with nothing: it does not count
    as a repetition, exactly as in ``latest_attempts``.
    """
    jours: dict[tuple[str, str], set] = {}
    for raw in raws:
        person = (raw.get("ID Personne") or "").strip()
        activity = (raw.get("ID Activité") or "").strip()
        if not person or not activity:
            continue
        jours.setdefault((person, activity), set()).add(
            simulated_day(raw.get("Temps simulé") or "")
        )

    return sum(1 for days in jours.values() if len(days) > 1)


def _une_occurrence_par_deplacement(raws: list[dict]) -> tuple[list[dict], int]:
    """R1/R4 — a (person, activity) pair comes out only once, at the smallest simulated day.

    Safety net, not main rule: when R2 has decided to cut, only one occurrence
    is left already and this pass removes nothing. It exists so that the invariant
    "a trip counts once" holds even if the cut criterion is wrong one day.
    """
    garde: dict[tuple[str, str], dict] = {}
    non_appariables: set[int] = set()
    for raw in raws:
        person = (raw.get("ID Personne") or "").strip()
        activity = (raw.get("ID Activité") or "").strip()
        if not person or not activity:
            non_appariables.add(id(raw))
            continue
        cle = (person, activity)
        tenu = garde.get(cle)
        if tenu is None or simulated_day(raw.get("Temps simulé") or "") < simulated_day(
            tenu.get("Temps simulé") or ""
        ):
            garde[cle] = raw
    gardes = {id(raw) for raw in garde.values()} | non_appariables
    retenus = [raw for raw in raws if id(raw) in gardes]
    return retenus, len(raws) - len(retenus)


def read_moves(path: Path, exclude_methods: list[str],
               first_day_only: bool | str = "auto",
               horizon_jours: int | None = None) -> tuple[list[dict], dict]:
    """Reads moves.csv and annotates each trip with its EMC² categories.

    **A trip counts once, and only once** (ticket 057, R1): that is the invariant,
    and everything that follows only serves to choose *which* occurrence to keep.

    ``first_day_only`` limits the read to the run's **first simulated day**. The cut is there
    for repetitions: the bootstrap and the sliding planning horizon spill over
    beyond 24 h, and on the reference run 2,538 (person, activity) pairs
    reappeared one day later, with the same mode in 57.8 % of cases. These repetitions
    are not extra decisions, they only weigh twice in the modal
    shares. Strand 2 applies the same cut on ``sim_day``
    (``common_set_eval.build_sample``): that is what guarantees the three strands a single
    scope.

    ``"auto"`` (default) cuts only when there is something to cut — ``horizon_jours > 1``, or
    at least one repeated pair (R2). **Without this, the cut removes unique trips**:
    on the simulator-less runs of 2026-09-14, it dropped 866 decisions out of 3,299
    for zero duplicates, 797 of them morning departures, which ``jeu.py:deplacements_attendus()`` dates
    to the next day because the originating « home » activity spans midnight. The scored scope
    then rose to 56.9 % of returns home, against 43.8 % over the whole day and
    39.0 % in the survey. ``True`` and ``False`` remain accepted and force the decision (R6).

    This cut is not enough on a **hot-resumed** run: the resume replays
    the simulated day in the same experiment directory, and both attempts
    carry the same simulated day. ``latest_attempts`` keeps only the most
    recent, upstream of the cut; the number of rows thus dropped comes out in
    ``exclues_reprise``.
    """
    # Rising edge per run: the global rescoring chains dozens of
    # logs, and without this reset only the first would report its unreadable label.
    reinitialiser_compteurs_libelles()
    rows: list[dict] = []
    stats = Counter()
    with Path(path).open(encoding="utf-8") as fh:
        raws = list(csv.DictReader(fh))
    stats["total"] = len(raws)
    raws, reprise = latest_attempts(raws)
    stats["exclues_reprise"] = reprise["n_dropped"]
    if reprise["reprise"]:
        stats["reprise"] = True
        stats["jours_de_calcul"] = reprise["jours_de_calcul"]

    repetes = couples_repetes(raws)
    stats["couples_repetes"] = repetes
    if first_day_only == "auto":
        if horizon_jours is not None and horizon_jours > 1:
            motif = "horizon"
        elif repetes:
            motif = "repetitions"
        else:
            motif = "aucune"
    else:
        motif = "forcee" if first_day_only else "aucune"
    stats["coupe"] = motif
    kept_day = first_simulated_day(path) if motif != "aucune" else None
    if kept_day:
        stats["jour_retenu"] = kept_day

    # R1/R4 — AFTER the cut: if it applied, only one occurrence is left and nothing is
    # removed here; if it did not apply, this net still guarantees uniqueness.
    raws, doublons = _une_occurrence_par_deplacement(
        [r for r in raws if not kept_day or simulated_day(r.get("Temps simulé") or "") == kept_day]
        if kept_day
        else raws
    )
    if kept_day:
        stats["exclues_jour"] = stats["total"] - stats["exclues_reprise"] - len(raws) - doublons
    stats["exclues_doublon"] = doublons
    for raw in raws:
        if raw.get("Méthode de sélection") in exclude_methods:
            stats["exclues_methode"] += 1
            continue
        chosen = CHOSEN_MODE_MAP.get((raw.get("Mode de transport Choisi") or "").strip())
        if chosen is None:
            stats["sans_mode"] += 1
            continue
        occupation_brute = (raw.get("Occupation principale") or "").strip()
        occupation = OCCUPATION_MAP.get(occupation_brute)
        if occupation is None:
            stats["occupation_inconnue"] += 1
            if occupation_brute:
                _signaler_libelle_inconnu(
                    OCCUPATIONS_INCONNUES, "Occupation principale", occupation_brute,
                    "la décision sort de la dimension « occupation », qui pèse dans le "
                    "composite.",
                )
        motif_brut = (raw.get("Motifs de déplacement") or "").strip()
        motif = MOTIF_MAP.get(motif_brut)
        if motif is None and motif_brut and motif_brut not in MOTIFS_HORS_ENQUETE:
            # `home`, `leisure` and `other` are normal values without an EMC²
            # equivalent: they leave the dimension without reporting anything. Everything
            # else is a label that cannot be read, and gets reported.
            stats["motif_inconnu"] += 1
            _signaler_libelle_inconnu(
                MOTIFS_INCONNUS, "Motifs de déplacement", motif_brut,
                "la décision sort de la dimension « motif », qui pèse dans le composite.",
            )
        probas = {}
        for col, mode in PROBA_COLUMNS.items():
            value = (raw.get(col) or "").strip()
            if value == "":
                continue
            try:
                probas[mode] = probas.get(mode, 0.0) + float(value)
            except ValueError:
                continue
        if probas:
            stats["avec_distribution"] += 1
        else:
            stats["sans_distribution"] += 1
        logement_brut = (raw.get("Type de logement") or "").strip()
        logement, logement_reference = normalize_housing(logement_brut)
        if logement is None:
            # Empty and unreadable are not the same failure: the empty column is a
            # NORMAL case (population enriched before ticket 019, outside the fine-zone
            # layer), the unreadable label is a translation bug. Mixing them up is
            # what let ticket 082 slip through for six weeks.
            stats["type_logement_vide" if not logement_brut
                  else "type_logement_inconnu"] += 1
        elif not logement_reference:
            # Modality known to the survey but absent from the published breakdown
            # (« Autres »): it will join no reference row. It is
            # counted here, otherwise it would vanish from the tally.
            stats["type_logement_hors_referentiel"] += 1
        lieu_brut = (raw.get("Lieu de résidence") or "").strip()
        lieu_residence, lieu_reference = normalize_place(lieu_brut)
        if lieu_residence is None:
            # Same distinction as for housing: a population enriched before
            # ticket 021 writes the column EMPTY, which is expected; a label the
            # table does not know is a defect, and it carries its own counter.
            stats["lieu_residence_vide" if not lieu_brut
                  else "lieu_residence_inconnu"] += 1
        elif not lieu_reference:
            # `hors périmètre` (ticket 021): known home, outside the 453 municipalities of
            # the survey. No per-zone target, so excluded from the strata — but counted
            # here, otherwise it would dilute without leaving a trace.
            stats["lieu_residence_hors_perimetre"] += 1
        offered = parse_offered_modes(raw.get("Modes proposés au LLM") or "")
        if not offered:
            stats["sans_offre"] += 1
        # Vehicle chain constraint (column written since ticket 008,
        # A4; empty on earlier runs). It EXPLAINS a decision, it does not
        # disqualify it: these rows stay in the scoring, and the page only
        # publishes their breakdown.
        contrainte = (raw.get("Contrainte de chaîne") or "").strip()
        stats["contrainte::" + (contrainte or "aucune")] += 1
        rows.append({
            "contrainte": contrainte or None,
            "agent_id": (raw.get("ID Personne") or "").strip(),
            "activity_id": (raw.get("ID Activité") or "").strip(),
            "chosen": chosen,
            "probas": probas,
            # Choice set actually submitted to the decision: it is what bounds
            # strand 3 (renormalisation over the OTP offer), and it alone tells
            # "mode rejected" from "mode never offered".
            "offered": offered,
            "departure_hour": departure_hour(raw.get("Heure de départ") or ""),
            "genre": (raw.get("Genre") or "").strip() or None,
            "age_cat": age_to_cat(raw.get("Âge")),
            "occupation": occupation,
            "motif": motif,
            "dist_cat": distance_to_cat(raw.get("Distance parcourue")),
            "lieu_residence": lieu_residence,
            "type_logement": logement,
        })
    return rows, dict(stats)


def simulation_frames(rows: list[dict]) -> dict[str, list[dict]]:
    """Two readings of the same run: probability mass, and mode actually drawn.

    ``attendu`` is the quantity calibration optimises (it depends on
    no draw); ``tire`` is what the simulation actually played. The gap
    between the two measures the sampling noise introduced by the draw.
    """
    attrs = ("genre", "age_cat", "occupation", "motif", "dist_cat",
             "lieu_residence", "type_logement")
    expected: list[dict] = []
    drawn: list[dict] = []
    for row in rows:
        meta = {k: row[k] for k in attrs}
        meta["agent_id"] = row["agent_id"]
        total = sum(row["probas"].values())
        if total > 0:
            for mode, mass in row["probas"].items():
                if mass <= 0:
                    continue
                expected.append({**meta, "mode_cat": mode, "weight": mass / total})
        else:
            expected.append({**meta, "mode_cat": row["chosen"], "weight": 1.0})
        drawn.append({**meta, "mode_cat": row["chosen"], "weight": 1.0})
    return {"attendu": expected, "tire": drawn}


# ── Strand 2: prompt calibration ─────────────────────────────────────────────

def load_dataset_metadata(dataset_dir: Path) -> dict[str, dict]:
    """``agent_id → attributes``, rebuilt from the frozen sets."""
    cols = ("age_cat", "occupation", "genre", "motif", "dist_cat")
    meta: dict[str, dict] = {}
    for split in ("train", "val", "test", "screen"):
        path = Path(dataset_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            meta[str(rec["agent_id"])] = {c: rec.get(c) for c in cols}
    return meta


def eval_regime(params_key: str, eval_model: str) -> dict:
    """Measurement regime of an eval: what must be identical to compare.

    The model is not enough to identify a regime. The switch to weighted
    counts changed the **decision policy**: the model returned one elected
    mode per persona, it now returns the full distribution, counted as
    probability mass. Two evals of the same model under these two policies do not
    measure the same thing — the raw decisions themselves differ, so
    no loss recomputation makes them comparable.

    It is the engine's ``eval_params_key`` that carries the information (``policy=weighted``
    since the switch, ``samples=N`` before); a readable label is derived from it.
    """
    key = params_key or ""
    if key == "legacy_import":
        policy = "import hérité"
    elif "policy=weighted" in key:
        policy = "masse de probabilité"
    else:
        policy = "mode élu"
    model = eval_model or "modèle non renseigné"
    return {"key": key or f"{model}?", "model": model, "policy": policy,
            "label": f"{model} · {policy}"}


def read_store_history(db_path: Path, keep_verdicts: list[str]) -> dict:
    """Trajectory of a store's non-rejected prompts, with raw decisions.

    Grouping by ``params_key`` is not cosmetic: two nodes evaluated
    by different models are not comparable, even after recomputing the
    score — it is the *decisions* that change, not just the loss.

    ``edges`` carries the edges actually walked (``node_to`` → ``node_from``
    of the mutations). They complete the nodes' ``parent`` column, empty as soon
    as a prompt has been **deduplicated**: nodes being content-addressed, a
    text already produced on another branch is reused with the parent of its
    first creation. Without these edges, a rebuilt lineage loses its seed.

    Set names **qualified by version** (``test@v2``) are kept on the same
    footing as bare names: since two versions of frozen sets coexist, the
    store tells ``test`` from ``test@v2`` — otherwise a v1 measurement would be
    served again for a v2 request. The filter must follow, otherwise the paid measurement
    stays invisible to the page. ``screen`` stays excluded: it is a strict subset
    of train, it carries no generalisation score.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        verdict_by_node: dict[str, str] = {}
        for row in conn.execute("SELECT node_to, verdict FROM mutations"):
            if row["node_to"]:
                verdict_by_node[row["node_to"]] = row["verdict"]
        nodes = []
        query = """
            SELECT n.hash, n.branch, n.iteration, n.created_at, n.parent,
                   e.dataset, e.params_key, e.scores_json, e.decisions,
                   e.eval_model, e.created_at AS eval_at
            FROM nodes n JOIN evals e ON e.node_hash = n.hash
            WHERE e.dataset IN ('train', 'val', 'test')
               OR e.dataset LIKE 'train@%'
               OR e.dataset LIKE 'val@%'
               OR e.dataset LIKE 'test@%'
            ORDER BY n.created_at, e.created_at
        """
        for row in conn.execute(query):
            verdict = verdict_by_node.get(row["hash"], "seed")
            if keep_verdicts and verdict not in keep_verdicts and verdict != "seed":
                continue
            try:
                scores = json.loads(row["scores_json"])
                decisions = json.loads(row["decisions"])
            except (TypeError, ValueError):
                continue
            nodes.append({
                "hash": row["hash"], "short": row["hash"][:8],
                "branch": row["branch"], "iteration": row["iteration"],
                "created_at": row["created_at"], "eval_at": row["eval_at"],
                "dataset": row["dataset"], "params_key": row["params_key"],
                "eval_model": row["eval_model"], "verdict": verdict,
                "stored_scores": scores, "decisions": decisions,
                "parent": row["parent"],
            })
        counts = {r["verdict"]: r["n"] for r in conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM mutations GROUP BY verdict")}
        totals = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("nodes", "mutations", "evals")}
        edges = {}
        for row in conn.execute(
                "SELECT node_to, node_from FROM mutations WHERE node_to IS NOT NULL "
                "AND node_from IS NOT NULL ORDER BY iteration, id"):
            edges.setdefault(row["node_to"], row["node_from"])
    finally:
        conn.close()
    return {"nodes": nodes, "verdict_counts": counts, "totals": totals,
            "edges": edges}


def lineage_chain(leaf: str, parents: dict[str, Optional[str]],
                  edges: dict[str, str]) -> list[str]:
    """Seed → ``leaf`` chain, falling back on the mutation edges.

    ``parents`` comes from the nodes' ``parent`` column, ``edges`` from the
    mutations table (cf. ``read_store_history``). The guard on already-seen nodes
    protects against a cycle that a hand-repaired store might carry.
    """
    chain: list[str] = []
    seen: set[str] = set()
    cur: Optional[str] = leaf
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = parents.get(cur) or edges.get(cur)
    return list(reversed(chain))


def decisions_frame(decisions: list, metadata: dict[str, dict],
                    categorize) -> list[dict]:
    """Stored decisions → scoring frame (join on ``agent_id``)."""
    out = []
    for item in decisions:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        agent_id, mode = str(item[0]), item[1]
        weight = float(item[2]) if len(item) > 2 else 1.0
        meta = metadata.get(agent_id, {})
        out.append({"agent_id": agent_id, "mode_cat": categorize(mode),
                    "weight": weight, **meta})
    return out


def load_common_set_eval(path: Path) -> list[dict]:
    """Prompts re-evaluated on the common set (action A3) → scoring frames.

    The file is produced by ``scripts/synthesis/common_set_eval.py``: one line
    per measured prompt, carrying its decisions in compact form (``columns`` +
    ``decisions``) and the description of the sample.

    Two differences from the decisions read from the store (``decisions_frame``), and
    they go the same way — being more exact, not less:

    - the strata are carried **per decision** and not re-joined on ``agent_id``.
      A person making three trips keeps their three purposes and three
      distances, where a per-agent join would keep only one;
    - ``mode_cat`` is already categorised by the engine at eval time, hence
      identical to what served to compute the stored composite score.

    A missing or unreadable file returns an empty list: the page then falls back
    on its « Données manquantes » card, it does not fail.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        columns = entry.get("columns") or []
        rows = []
        for raw in entry.get("decisions") or ():
            row = dict(zip(columns, raw))
            if row.get("mode_cat") is None:
                continue
            row["weight"] = float(row.get("weight") or 0.0)
            rows.append(row)
        if not rows:
            continue
        entry["rows"] = rows
        out.append(entry)
    # Seed first, leaf next: that is the reading direction of the trajectory.
    order = {"seed": 0, "leaf": 1}
    out.sort(key=lambda e: (order.get(e.get("role"), 9), e.get("short", "")))
    return out


# ── Strand 3: PROGEDO model ──────────────────────────────────────────────────

def load_model_predictions(path: Path) -> Optional[dict]:
    """Model predictions on the common set (action A8) → scoring frames.

    The parquet is produced by ``scripts/synthesis/model_on_common_set.py``: one row
    per decision of the strand 1 scope, carrying the probabilities **before** and
    **after** renormalisation over the OTP offer, plus the scoring strata
    copied from the log. It can therefore be scored alone, without re-reading ``moves.csv`` — same
    principle as the action A3 jsonl, and for the same reason: the strata follow
    the decision, not the agent.

    Two readings are produced, as for strand 1:

    - ``attendu`` — one row per offered mode, weighted by its renormalised
      probability. It is the quantity the model calibrates best;
    - ``elu`` — one row per decision, on the most probable mode. The model
      almost never elects bike (recall 0.128 in training) while it
      calibrates it well in mass: showing both is the only honest reading.

    ``brut`` completes the picture: the same probability mass **before**
    renormalisation, so that the effect of the OTP correction is measurable and not
    merely asserted.

    File missing, unreadable, or pyarrow not installed → ``None``: the page falls back
    on its « Données manquantes » card, it does not fail.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(p)
    except Exception:  # truncated parquet, pyarrow missing…
        return None
    raw_meta = (table.schema.metadata or {}).get(b"progedo_on_common_set")
    try:
        meta = json.loads(raw_meta) if raw_meta else {}
    except ValueError:
        meta = {}
    records = table.to_pylist()

    attrs = ("genre", "age_cat", "occupation", "motif", "dist_cat",
             "lieu_residence", "type_logement")
    expected: list[dict] = []
    raw_expected: list[dict] = []
    elected: list[dict] = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        base = {k: rec.get(k) for k in attrs}
        base["agent_id"] = rec.get("agent_id")
        for mode in MODES:
            weight = rec.get(f"p_{mode}")
            if weight:
                expected.append({**base, "mode_cat": mode, "weight": float(weight)})
            raw_weight = rec.get(f"p_raw_{mode}")
            if raw_weight:
                raw_expected.append({**base, "mode_cat": mode,
                                     "weight": float(raw_weight)})
        if rec.get("argmax"):
            elected.append({**base, "mode_cat": rec["argmax"], "weight": 1.0})
    if not expected:
        return None
    return {"meta": meta,
            "variants": {"attendu": expected, "elu": elected, "brut": raw_expected},
            "n_rows": len(records)}


def read_prompt_variants(path: Path) -> dict:
    """Historical variants of prompts.yaml and their archived scores."""
    with Path(path).open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    active = (doc.get("active") or {}).get("itinary_multi_agent")
    variants = []
    for name, node in (doc.get("prompts") or {}).items():
        calib = (node or {}).get("_calibration") or {}
        content = (node or {}).get("content") or ""
        variants.append({
            "name": name, "active": name == active,
            "words": len(content.split()),
            "seed": calib.get("seed"), "date": calib.get("date"),
            "iterations": calib.get("iterations"),
            "sample_size": calib.get("sample_size"),
            "score_initial": calib.get("score_initial"),
            "score_final": calib.get("score_final"),
            "has_archived_score": bool(calib.get("score_final")),
        })
    variants.sort(key=lambda v: (v["date"] or "", v["name"]))
    return {"active": active, "variants": variants}


# ── Common scoring ───────────────────────────────────────────────────────────

def dimension_detail(rows: list[dict], cerema: dict, dim: dict) -> list[dict]:
    """Per category: count, observed shares, target shares, L1 gap."""
    parts = cerema.get("parts_modales_2023", {}).get(dim["cerema"]) or {}
    column = dim["column"]
    by_cat: dict[str, dict] = defaultdict(lambda: {"mass": Counter(), "agents": set()})
    for row in rows:
        cat = row.get(column)
        if cat is None or row.get("mode_cat") not in MODES:
            continue
        bucket = by_cat[cat]
        bucket["mass"][row["mode_cat"]] += float(row.get("weight", 1.0))
        bucket["agents"].add(row.get("agent_id"))

    # What the loop below will not see: the categories the reference does not
    # break down (`hors périmètre` for the zone, « Autres » for housing). They
    # have no target and so leave the strata — but their mass is published, instead
    # of vanishing into a denominator.
    off_reference = {cat: {"mass": sum(bucket["mass"].values()),
                           "n": len(bucket["agents"])}
                     for cat, bucket in by_cat.items() if cat not in parts}

    out = []
    for cat in parts.keys():
        target = reference_shares(cerema, dim["cerema"], cat)
        bucket = by_cat.get(cat)
        if not bucket or sum(bucket["mass"].values()) <= 0:
            out.append({"cat": cat, "n": 0, "actual": {}, "target": target,
                        "l1": None, "covered": False})
            continue
        total = sum(bucket["mass"].values())
        actual = {m: bucket["mass"].get(m, 0.0) * 100.0 / total for m in MODES}
        l1 = sum(abs(actual.get(m, 0.0) - target.get(m, 0.0)) for m in MODES)
        n = len(bucket["agents"])
        out.append({"cat": cat, "n": n, "actual": actual, "target": target,
                    "l1": l1, "covered": n >= 5})
    if off_reference:
        excluded_mass = sum(row["mass"] for row in off_reference.values())
        out.append({"cat": OFF_REFERENCE_ROW, "n": sum(row["n"] for row
                                                       in off_reference.values()),
                    "actual": {}, "target": {}, "l1": None, "covered": False,
                    "excluded_mass": excluded_mass, "categories": off_reference})
    return out


def global_view(rows: list[dict], cerema: dict) -> dict:
    """Observed global modal shares vs EMC², plus the out-of-scope mass."""
    mass = Counter()
    excluded = 0.0
    agents = set()
    for row in rows:
        mode = row.get("mode_cat")
        weight = float(row.get("weight", 1.0))
        if mode in MODES:
            mass[mode] += weight
            agents.add(row.get("agent_id"))
        else:
            excluded += weight
    total = sum(mass.values())
    target = reference_shares(cerema, "global")
    actual = {m: (mass.get(m, 0.0) * 100.0 / total if total else 0.0) for m in MODES}
    return {
        "actual": actual, "target": target,
        "gaps": {m: actual[m] - target.get(m, 0.0) for m in MODES},
        "l1": sum(abs(actual[m] - target.get(m, 0.0)) for m in MODES),
        "mass": total, "excluded_mass": excluded, "n_agents": len(agents),
    }


def worst_strata(details: dict[str, list[dict]], top_k: int = 8) -> list[dict]:
    """Worst dimension × category × mode crossings, weighted by count."""
    out = []
    for dim_key, rows in details.items():
        for entry in rows:
            if not entry.get("covered"):
                continue
            for mode in MODES:
                actual = entry["actual"].get(mode)
                target = entry["target"].get(mode)
                if actual is None or target is None:
                    continue
                diff = actual - target
                out.append({"dim": dim_key, "cat": entry["cat"], "mode": mode,
                            "actual": actual, "target": target, "diff": diff,
                            "n": entry["n"], "impact": abs(diff) * entry["n"]})
    out.sort(key=lambda r: r["impact"], reverse=True)
    return out[:top_k]


class Scorer:
    """Applies the scoring formula to a decision frame.

    ``calibration_module`` is ``formule_score`` (``sources.import_formule_score``) or, for the
    private calibration tools, the engine's ``calibration`` package: both expose ``metrics``.
    """

    def __init__(self, calibration_module, weights: dict, primary: str,
                 secondary: str):
        m = calibration_module.metrics
        self._m = m
        self._pd = __import__("pandas")
        self.weights = dict(weights)
        self.categorize = m.categorize_mode
        self.primary = self._build(primary)
        self.secondary = self._build(secondary) if secondary else None

    def _build(self, name: str):
        m = self._m
        cls = m.EMDJSDComposite if name in ("emd_jsd", "emd", "jsd") else m.L1Composite
        # length_penalty at 0 in the weights: the composite score stays defined for a
        # strand without prompt, and the three strands are on the same scale.
        return cls(weights=self.weights)

    def score(self, rows: list[dict], cerema: dict) -> dict:
        if not rows:
            return {}
        df = self._pd.DataFrame(rows)
        out: dict[str, Any] = {}
        primary = self.primary.compute(df, cerema, "")
        out[self.primary.name] = primary.model_dump(by_alias=True)
        if self.secondary is not None:
            secondary = self.secondary.compute(df, cerema, "")
            out[self.secondary.name] = secondary.model_dump(by_alias=True)
        return out


def coverage_matrix(rows: list[dict], cerema: dict) -> dict:
    """Count (in persons) per dimension × category, for the n ≥ 5 threshold."""
    out: dict[str, dict] = {}
    for dim in DIMENSIONS:
        parts = cerema.get("parts_modales_2023", {}).get(dim["cerema"]) or {}
        seen: dict[str, set] = defaultdict(set)
        for row in rows:
            cat = row.get(dim["column"])
            if cat is not None:
                seen[cat].add(row.get("agent_id"))
        out[dim["key"]] = {cat: len(seen.get(cat, ())) for cat in parts.keys()}
    return out
