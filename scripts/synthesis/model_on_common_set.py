"""Applies the PROGEDO policy to the common set, on the OTP offer (action A8).

    python -m scripts.synthesis.model_on_common_set [--dry-run]

**The problem.** Strand 3 of the synthesis page has a trained model
(action A6) and a fine-zone resolver (action A7), but it had been applied to
no decision of the pinned run: its column of the comparison matrix is empty. Yet a
model that has not been confronted with the same substrate as the other two parts says
nothing comparable.

**What this script does.** It replays each decision of the pinned run — the same scope
as strand 1, built by the same ``frames.read_moves`` — rebuilds the 21
variables of the feature contract, predicts ``P(mode)`` over the 4 policy classes,
then **restricts and renormalises the prediction to the modes actually offered by
OTP** for that trip. The result is written to the parquet declared in the manifest
(``arms.model.predictions``), with the probabilities **before and after** renormalisation:
the effect of the correction must stay auditable.

**Why renormalise.** The policy predicts over 4 classes without knowing what was
offered. The simulation, on the other hand, only chooses among the itineraries OTP offered.
Comparing the two without correction would amount to blaming the LLM for not choosing
a mode it was never offered, or crediting the model with a non-existent option.
This is the IIA assumption of ticket 005 §4 (decision E10): the relative preference between
two offered modes does not depend on the presence of a third.

**What it does not impute.** Three situations fall outside the scored scope, and they are
counted rather than silently repaired:

- **outside the zone layer** — the A7 resolver returns "no zone" for ~5 % of
  locations, at a median of 22 km from the survey scope. ``od_km`` is by far the
  first variable of the model and training required it (``CRITICAL``): predicting
  with a missing ``od_km`` would be an out-of-domain extrapolation, not a prediction;
- **offer with no predictable mode** — a trip none of whose offered modes belongs
  to the 4 classes (motorised two-wheeler only, for example) has no distribution to
  renormalise;
- **persona not found** — a decision that cannot be linked to its traits.

These rows are **written anyway** to the parquet, with their ``status`` and without
probability: the excluded mass can be recounted from the file, it is not a figure
to be taken on trust.

Deterministic: no random draw, no network call, no API key. Two runs
produce the same parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import frames
from .sources import REPO_ROOT, load_manifest

SCHEMA = "progedo_on_common_set/v1"

# ── Mode mapping — THE single point ──────────────────────────────────────────
#
# Four vocabularies meet here, and an approximate mapping raises
# no exception: it produces plausible and wrong probabilities.
#
#   policy classes            bike · car · transit · walk
#   sim canonical modes       walking · cycling · car · public_transport · train ·
#                             motorbike · other        (mobility_llm.mode_choice)
#   moves.csv labels          Marche · Vélo · Voiture Privée · Transports_collectifs ·
#                             Train · Deux-roues motorisé · Autres modes
#   page categories           marche · velo · voiture · transports_collectifs · autres
#
# The page is the pivot: `frames.CHOSEN_MODE_MAP` already maps the CSV labels to
# categories, and strand 1 is scored in categories. So only one bridge needs defining
# here, the one for the policy classes.
POLICY_CLASS_TO_CAT = {
    "bike": "velo",
    "car": "voiture",
    "transit": "transports_collectifs",
    "walk": "marche",
}
CAT_TO_POLICY_CLASS = {v: k for k, v in POLICY_CLASS_TO_CAT.items()}

# Simulator canonical modes → page categories. Copied from the table of the
# trip log (`move_logger._CANONICAL_FR` + `frames.CHOSEN_MODE_MAP`), and
# checked by a test: it is the only way to know that the two do not diverge.
CANONICAL_TO_CAT = {
    "walking": "marche",
    "cycling": "velo",
    "car": "voiture",
    "public_transport": "transports_collectifs",
    "train": "transports_collectifs",
    "motorbike": "autres",
    "other": "autres",
}

# Two asymmetric merges, to keep in mind when reading the figures:
#
# - `train`: the policy puts it in `transit`, the page in
#   `transports_collectifs`. Both say the same thing, the bridge holds.
# - `motorbike`: the policy puts it in `car`, the page in "autres" — outside the
#   4 scored modes. A "motorised two-wheeler" offer is therefore **removed from the offer**
#   rather than counted as a car offer: strand 3 must renormalise over
#   the same scope of modes as the one strand 1 is scored on, otherwise
#   the columns stop being comparable. The pinned run offers none.
PREDICTABLE_CATS = tuple(POLICY_CLASS_TO_CAT.values())

# Decision statuses. Only `ok` enters the score; the others are written with
# their reason, so that the excluded mass can be recounted from the parquet.
STATUS_OK = "ok"
STATUS_NO_ZONE = "hors_couche_zones"
STATUS_NO_OFFER = "offre_sans_mode_predictible"
STATUS_NO_PERSONA = "persona_introuvable"

# Metadata columns copied from strand 1: the parquet must be scorable on its own,
# without re-reading moves.csv — same principle as the action A3 jsonl.
META_COLUMNS = ("genre", "age_cat", "occupation", "motif", "dist_cat",
                "lieu_residence", "type_logement")


# ── Persona traits → contract variables ──────────────────────────────────────

def has_bike(personal_bike: Any) -> Optional[bool]:
    """Persona ``personal_bike`` → spec boolean ``has_bike``.

    In training, ``has_bike`` means "the household declares at least one bike"
    (``M21 > 0``). The persona carries a three-valued string, whose only negative
    is "Pas de vélo"; an e-bike counts as a bike, since the survey does not allow
    isolating it (M22 not filled in).
    """
    if personal_bike is None:
        return None
    text = str(personal_bike).strip().lower()
    if not text:
        return None
    # BOTH vocabularies: "No bike" since v6 (ticket 074), "Pas de vélo" before.
    # Knowing only one would make the other return `True` — a persona without a bike counted
    # as having one, over a whole cohort, without a single row missing.
    return text not in _SANS_VELO


#: The two ways of writing "this person has no bike".
_SANS_VELO = frozenset({"pas de vélo", "no bike"})


# Persona `main_occupation` label → `feature_spec.json` category.
#
# The spec and the trained artefacts encode the SURVEY categories, in French
# (`main_occupation=Travail à plein temps` is a column of the design matrix). The v6
# cohort carries English labels, served to the model in its prompt.
#
# The translation happens HERE, at the boundary, and above all NOT by retraining: the four
# control models are the reference against which the LLMs are scored, and refitting their
# coefficients would move published figures for a language reason.
#
# Without this table, every v6 persona would fall into `__missing__`, whose contract says
# "zero coefficient: an absent value behaves like the reference category". No
# exception, no log: a variable set to zero on 100 % of the rows.
_OCCUPATION_VERS_SPEC = {
    "Pupil (up to Baccalaureate)": "Scolaire (jusqu'au Bac)",
    "Student": "Étudiant",
    "Full-time worker": "Travail à plein temps",
    "Part-time worker": "Travail à temps partiel",
    "Unemployed / job seeker": "Chômeur/recherche d'emploi",
    "Homemaker": "Personne au foyer",
    "Retired": "Retraité",
    # Added on 2026-09-16 for the unit audit (ticket 058). Survey personas carry
    # "Other", the translation of the survey category "Autre", which the v6 cohort does not
    # expose. Without this entry, `occupation_du_spec` returned "Other", outside the spec
    # categories, hence `__missing__` — zero coefficient, reference-category behaviour,
    # without a word in the log. 26 respondents out of 2,930 were affected.
    "Other": "Autre",
}


def occupation_du_spec(valeur: Any) -> Optional[str]:
    """Occupation category as the spec expects it, whatever the persona's language."""
    if valeur is None:
        return None
    texte = str(valeur).strip()
    return _OCCUPATION_VERS_SPEC.get(texte, texte or None)


def persona_features(traits: dict) -> dict:
    """The 12 ``source: persona`` variables of the spec, read from ``traits_json``.

    No value is invented: an absent key stays absent, and
    ``encode_features`` will route it as missing. An out-of-spec category (the
    ``socioprofessional_class = "Retired"`` of the synthetic population, which the
    survey recoding never produces) also becomes missing — that is the
    explicit contract of the spec, "unexpected category" not being "most
    frequent category". The information is not lost for all that: ``main_occupation``
    carries "Retraité", which is in the spec.
    """
    return {
        "age": traits.get("age"),
        "gender": traits.get("gender"),
        "household_size": traits.get("household_size"),
        "has_driving_license": traits.get("has_driving_license"),
        "has_pt_subscription": traits.get("has_pt_subscription"),
        "number_of_cars": traits.get("number_of_cars"),
        "car_availability": traits.get("car_availability"),
        "has_bike": has_bike(traits.get("personal_bike")),
        "socioprofessional_class": traits.get("socioprofessional_class"),
        "main_occupation": occupation_du_spec(traits.get("main_occupation")),
        "employed": traits.get("employed"),
        "studies": traits.get("studies"),
    }


def activity_index(population: list[dict]) -> dict[str, dict]:
    """``(person, activity) → context of the trip leading to it``.

    The origin of a trip is the **previous** activity in the persona's chain.
    The chain is cyclic — ``activities[i-1].end_time == activities[i]
    .scheduled_start_time`` for every ``i``, including ``i = 0`` — so the origin of the
    first trip is the last activity, and not "nothing". Checked on the pinned
    run: 3,562 links, none broken.
    """
    index: dict[str, dict] = {}
    for person in population:
        identity = person.get("identity") or {}
        traits = identity.get("traits_json") or {}
        activities = identity.get("activities") or []
        traits_features = persona_features(traits)
        for i, activity in enumerate(activities):
            origin = activities[i - 1]
            index[f'{person.get("person_id")}/{activity.get("id")}'] = {
                "traits": traits_features,
                "purpose": activity.get("purpose"),
                "purpose_origin": origin.get("purpose"),
                "origin": _lat_lon(origin.get("location")),
                "destination": _lat_lon(activity.get("location")),
            }
    return index


def _lat_lon(location: Optional[dict]) -> Optional[tuple[float, float]]:
    if not location:
        return None
    try:
        return float(location["lat"]), float(location["lon"])
    except (KeyError, TypeError, ValueError):
        return None


# ── Renormalisation over the offer ───────────────────────────────────────────

def renormalize(probabilities: dict[str, float],
                offered: list[str]) -> Optional[dict[str, float]]:
    """Restricts ``probabilities`` to the offered modes, then renormalises to 100 %.

    ``probabilities`` and ``offered`` are in **page categories**. Returns ``None``
    when the intersection is empty (nothing to renormalise) or when the offered mass is
    zero — a model that gives exactly no chance to everything offered
    does not provide a distribution, it provides a division by zero.

    A single offered mode gives ``{mode: 1.0}``: this is not a prediction, it is the
    observation that there was no choice. The simulation is in the same case, and
    strand 1 counts these trips — discarding them on one side only would unbalance the
    comparison.
    """
    kept = {m: probabilities.get(m, 0.0) for m in offered
            if m in probabilities}
    total = sum(kept.values())
    if not kept or total <= 0:
        return None
    return {m: v / total for m, v in kept.items()}


def offered_mass(probabilities: dict[str, float], offered: list[str]) -> float:
    """Raw probability mass falling on the offered modes, in [0, 1].

    It is the renormalisation factor, and the direct measure of the effect of the
    correction: 1.0 means OTP offered everything the model considered.
    """
    return sum(probabilities.get(m, 0.0) for m in offered if m in probabilities)


# ── Model loading ────────────────────────────────────────────────────────────

#: Accepted artefact formats — one per oracle. The first is the supervised booster
#: ("high reference"), the second the strict-parity multinomial logit, which serves
#: as a behavioural arbiter and not as a fidelity target, the third the kernel logistic
#: regression (ticket 043), second arbiter of the same block.
#:
#: The random forest enters at ticket 088 § 3.3. It was absent because ticket
#: 043 was modifying this line in parallel, and its launcher worked around the wait by replacing
#: `load_policy` in the decision-maker's namespace. Rule R7 of ticket 044 — a control does
#: not become an arbiter — does not rest on the absence of a format in this tuple, but on the
#: arms of the `sources.yaml` manifest: the composite score (`bi_oracle.py`) only reads `model`,
#: `model_mnl` and `model_klr`, and none of them designates the forest
#: (`test_R7_le_temoin_reste_hors_du_composite`).
POLICY_FORMATS = ("lightgbm_mode_choice_policy", "mnl_mode_choice_policy",
                  "klr_mode_choice_policy", "rf_mode_choice_policy")


def load_policy(path: Path, spec: dict) -> tuple[Any, dict]:
    """Reloads an oracle from its artefact, after checking the contract.

    **Three formats, a single prediction path.** The LightGBM booster is reloaded from
    ``model_text`` (native format, which restores it identically); the multinomial logit is
    reloaded as ``LogitPredictor`` and the kernel logistic regression as ``KLRPredictor``,
    two pure-numpy evaluators built on the coefficients — direct for the logit,
    dual on the support points for the KLR. All three expose ``predict`` and
    ``feature_name``, so everything that follows
    — encoding, renormalisation over the OTP offer, writing the parquet — is the **same code**
    for all three. That is the condition for their parquets to be comparable: two
    columns measured by two paths cannot be compared (rule R7 of
    `specs/score_composite_deux_oracles.md`, K9 of
    `specs/ticket_043/klr-troisieme-famille.md`).

    The ``dump_model`` form of the LightGBM artefact had been planned for a pure
    Python evaluator meant for the ``controller`` container; that evaluator was never written and
    decision E9 was revised on 2026-09-08 — ``lightgbm`` and ``libgomp1`` are now
    in the ``controller`` image, which therefore calls this function like everyone else.

    Three refusals, all silent if they are not set: an unknown artefact format,
    a ``spec_version`` that differs from the one of the spec read, and an order of variables or
    classes that would not be the spec's — a one-column shift gives
    perfectly plausible probabilities.
    """
    artefact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artefact.get("format") not in POLICY_FORMATS:
        raise ValueError(f"Format d'artefact inattendu : {artefact.get('format')!r}.")

    # The forest is not reloaded, it is REFITTED: no tree is serialised in its
    # artefact, which carries its hyperparameters and the SHA of its training matrix.
    # `charger_rf` applies the same three refusals as the rest of this function, plus a check
    # that the other three families do not need — does the refitted forest reproduce
    # the published metrics? — and it returns the same (predictor, artefact) pair. Lazy
    # import: no lightgbm, mnl or klr run pays the cost of scikit-learn.
    if artefact["format"] == "rf_mode_choice_policy":
        from scripts.progedo_logit.mode_choice_rf import charger_rf
        return charger_rf(Path(path), spec)
    if artefact.get("spec_version") != spec.get("spec_version"):
        raise ValueError(
            f"The model was trained under the feature contract v"
            f"{artefact.get('spec_version')}, the spec read is v{spec.get('spec_version')}. "
            "Retrain the policy (make policy) rather than predict under a "
            "contract that has changed.")

    names = [f["name"] for f in spec["features"]]
    if [f["name"] for f in artefact.get("features") or ()] != names:
        raise ValueError("The variable order of the artefact differs from that of the spec.")
    classes = list(spec["target"]["classes"])
    if list((artefact.get("target") or {}).get("classes") or ()) != classes:
        raise ValueError("The class order of the artefact differs from that of the spec.")
    unknown = sorted(set(classes) - set(POLICY_CLASS_TO_CAT))
    if unknown:
        raise ValueError(f"Classes without a mode mapping: {unknown}.")

    if artefact["format"] == "mnl_mode_choice_policy":
        from scripts.progedo_logit.mode_choice_logit import LogitPredictor
        model: Any = LogitPredictor(artefact)
    elif artefact["format"] == "klr_mode_choice_policy":
        from scripts.progedo_logit.mode_choice_klr import KLRPredictor
        model = KLRPredictor(artefact)
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_str=artefact["booster"]["model_text"])
    if list(model.feature_name()) != names:
        raise ValueError("The reloaded model does not expect the spec variables.")
    return model, artefact


# ── Building the prediction set ──────────────────────────────────────────────

def build_rows(moves: list[dict], index: dict[str, dict], resolver) -> list[dict]:
    """One row per decision of the strand-1 scope, enriched with its context.

    ``resolver`` may be ``None``: the six geographic variables are then
    absent everywhere and every decision comes out as ``hors_couche_zones``. This is
    intentional — a missing layer must produce an honest file empty of
    predictions, not predictions without geography.
    """
    rows: list[dict] = []
    pending: list[int] = []
    origins: list[tuple[float, float]] = []
    destinations: list[tuple[float, float]] = []

    for move in moves:
        key = f'{move["agent_id"]}/{move["activity_id"]}'
        context = index.get(key)
        offered = [m for m in move["offered"] if m in PREDICTABLE_CATS]
        row: dict[str, Any] = {
            "agent_id": move["agent_id"],
            "activity_id": move["activity_id"],
            "offered": "|".join(move["offered"]),
            "offered_predictable": "|".join(offered),
            "n_offered": len(offered),
            "sim_chosen": move["chosen"],
            "departure_hour": move["departure_hour"],
            **{c: move.get(c) for c in META_COLUMNS},
        }
        if context is None:
            row["status"] = STATUS_NO_PERSONA
        elif not offered:
            row["status"] = STATUS_NO_OFFER
        else:
            row["status"] = STATUS_OK
            row.update(context["traits"])
            row["purpose"] = context["purpose"]
            row["purpose_origin"] = context["purpose_origin"]
            if resolver is not None and context["origin"] and context["destination"]:
                pending.append(len(rows))
                origins.append(context["origin"])
                destinations.append(context["destination"])
            else:
                row["status"] = STATUS_NO_ZONE
        rows.append(row)

    if pending:
        geo = resolver.geo_features_many(origins, destinations)
        for position, features in zip(pending, geo):
            if features is None:
                rows[position]["status"] = STATUS_NO_ZONE
            else:
                rows[position].update(features.as_dict())
    return rows


def predict(rows: list[dict], booster, spec: dict) -> dict[str, int]:
    """Adds ``p_raw_*`` then ``p_*`` (renormalised) to the predictable rows.

    The encoding is the training script's — ``encode_features`` imported as
    is, never rewritten: rewriting it is the best way to introduce a silent
    shift between training and prediction.

    Returns, per variable, the number of missing values **after encoding**. A
    category that the persona carries but the spec does not know shows up there: it is
    the only place where the vocabulary gap between synthetic population and survey
    becomes visible, since encoding makes it missing without raising anything.
    """
    import pandas as pd

    from scripts.progedo_logit.fit_mode_choice_policy import encode_features

    classes = list(spec["target"]["classes"])
    predictable = [r for r in rows if r["status"] == STATUS_OK]
    if not predictable:
        return {}

    matrix = encode_features(pd.DataFrame(predictable), spec)
    missing = {name: int(matrix[name].isna().sum()) for name in matrix.columns}
    proba = booster.predict(matrix)
    for row, distribution in zip(predictable, proba):
        raw = {POLICY_CLASS_TO_CAT[c]: float(p) for c, p in zip(classes, distribution)}
        offered = [m for m in row["offered_predictable"].split("|") if m]
        row["p_offered_mass"] = offered_mass(raw, offered)
        renormalized = renormalize(raw, offered)
        for cat in PREDICTABLE_CATS:
            row[f"p_raw_{cat}"] = raw[cat]
            row[f"p_{cat}"] = (renormalized or {}).get(cat, 0.0)
        if renormalized is None:
            # The model gives no mass to what was offered: there is no
            # distribution to write. Never met on the pinned run, but it
            # must leave the score rather than enter it as zeros.
            row["status"] = STATUS_NO_OFFER
            for cat in PREDICTABLE_CATS:
                row[f"p_{cat}"] = None
            continue
        row["argmax_raw"] = max(raw, key=raw.get)
        row["argmax"] = max(renormalized, key=renormalized.get)
    return {k: v for k, v in missing.items() if v}


# ── Writing ──────────────────────────────────────────────────────────────────

# Parquet column order: identifiers, offer, status, probabilities before
# then after renormalisation, context, scoring strata.
#
# The six geographic variables are **deliberately absent**. They are computed
# from `zf_zones.gpkg`, kept out of the repository like its PROGEDO source (restricted
# access lil-1750): rewriting them row by row here would republish, for all the
# zones the run crosses, the densities and distances to the centre of that resource.
# They serve neither the score nor the join with strand 1; anyone wanting to audit the
# prediction inputs reruns the script with the layer in place.
COLUMNS = (
    ["agent_id", "activity_id", "status", "offered", "offered_predictable",
     "n_offered", "sim_chosen", "argmax_raw", "argmax", "p_offered_mass"]
    + [f"p_raw_{c}" for c in PREDICTABLE_CATS]
    + [f"p_{c}" for c in PREDICTABLE_CATS]
    + ["departure_hour"]
    + list(META_COLUMNS)
)


def _digest(path: Optional[Path]) -> Optional[str]:
    """sha256 of a file, or ``None`` if it is unreadable. Serves as a model fingerprint."""
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet(rows: list[dict], path: Path, meta: dict) -> None:
    """Writes the parquet and attaches its descriptor (schema, run, model, exclusions).

    The descriptor travels **inside** the file rather than next to it: the manifest only
    declares a path, and a parquet that cannot be linked to its run proves nothing.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = None
    table = pa.Table.from_pandas(df[COLUMNS], preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"progedo_on_common_set": json.dumps(meta, ensure_ascii=False).encode(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def summarize(rows: list[dict]) -> dict:
    """Run counts: statuses, choice set sizes, effect of the renormalisation."""
    statuses = Counter(r["status"] for r in rows)
    scored = [r for r in rows if r["status"] == STATUS_OK]
    masses = sorted(r["p_offered_mass"] for r in scored) if scored else []
    single = sum(1 for r in scored if r["n_offered"] == 1)
    shifted = sum(1 for r in scored if r.get("argmax_raw") != r.get("argmax"))
    return {
        "n_moves": len(rows),
        "n_scored": len(scored),
        "n_agents": len({r["agent_id"] for r in rows}),
        "n_agents_scored": len({r["agent_id"] for r in scored}),
        "status_counts": dict(statuses),
        "excluded_pct": 100.0 * (len(rows) - len(scored)) / max(1, len(rows)),
        "offer_sizes": dict(Counter(r["n_offered"] for r in scored)),
        "n_single_offer": single,
        "n_argmax_shifted": shifted,
        "offered_mass_mean": (sum(masses) / len(masses)) if masses else None,
        "offered_mass_min": masses[0] if masses else None,
        "offered_mass_p10": masses[len(masses) // 10] if masses else None,
        "offered_mass_median": masses[len(masses) // 2] if masses else None,
    }


def shares(rows: list[dict], key: str) -> dict[str, float]:
    """Modal shares in % over the scored rows, as mass (`p_`) or raw (`p_raw_`)."""
    scored = [r for r in rows if r["status"] == STATUS_OK]
    total = sum(sum(r[f"{key}{c}"] for c in PREDICTABLE_CATS) for r in scored)
    if total <= 0:
        return {}
    return {c: 100.0 * sum(r[f"{key}{c}"] for r in scored) / total
            for c in PREDICTABLE_CATS}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="sources manifest (default: sources.yaml)")
    parser.add_argument("--out", help="output parquet (default: the manifest's)")
    parser.add_argument("--policy", help=(
        "oracle artefact (default: arms.model.policy — the LightGBM booster). "
        "For the second oracle: --policy scripts/progedo_logit/mnl_model.json "
        "--out scripts/synthesis/data/mnl_on_common_set.parquet"))
    parser.add_argument("--dry-run", action="store_true",
                        help="shows the scope and the counts, without writing anything")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    spec_path = manifest.path_of("arms.model.feature_spec")
    policy_path = (Path(args.policy) if args.policy
                   else manifest.path_of("arms.model.policy"))
    if args.policy and not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    zones_path = manifest.path_of("arms.model.zones")
    out_path = Path(args.out or manifest.get("arms.model.predictions"))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    for label, path in (("feature_spec", spec_path), ("policy", policy_path)):
        if path is None or not path.exists():
            print(f"[erreur] {label} not found: {path}", file=sys.stderr)
            return 2

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    booster, artefact = load_policy(policy_path, spec)

    # ── Scope: that of strand 1, built by the same code ──────────────────────
    run = frames.resolve_run(manifest)
    if not run.get("exists") or not run.get("moves", {}).get("exists"):
        print(f"[erreur] Run not found or without moves.csv: "
              f"{manifest.get('common_set.run')}", file=sys.stderr)
        return 2
    moves_path = REPO_ROOT / run["moves"]["path"]
    moves, stats = frames.read_moves(
        moves_path, manifest.get("common_set.exclude_selection_methods", []))
    population_path = REPO_ROOT / run["population"]["path"]
    if not population_path.exists():
        print(f"[erreur] Population du run introuvable : {population_path}",
              file=sys.stderr)
        return 2
    population = json.loads(population_path.read_text(encoding="utf-8"))

    print(f"Pinned run: {run['path']}")
    print(f"Strand 1 scope: {len(moves)} decisions, "
          f"{len({m['agent_id'] for m in moves})} persons "
          f"(out of {stats.get('total')} log rows)")
    if artefact["format"] == "mnl_mode_choice_policy":
        fitted = (f"logit multinomial, {artefact['training']['n_design_columns']} "
                  f"colonnes de dessin, C = {artefact['training']['C']}")
    else:
        fitted = f"booster, {artefact['booster']['best_iteration']} itérations"
    print(f"Oracle: {artefact['format']} | spec v{spec['spec_version']}, "
          f"{len(spec['features'])} variables, "
          f"{len(spec['target']['classes'])} classes | {fitted}")

    # ── Fine-zone resolver ───────────────────────────────────────────────────
    resolver = None
    resolver_error = ""
    if zones_path is not None and zones_path.exists():
        try:
            from mobility_core.zone_resolver import ZoneResolver
            resolver = ZoneResolver.load(zones_path, feature_spec=spec_path)
        except Exception as exc:  # unreadable resource, geopandas missing…
            resolver_error = str(exc)
            print(f"⚠ Résolveur de zone fine indisponible : {exc}", file=sys.stderr)
    else:
        resolver_error = f"couche de zones absente ({zones_path}) — `make zones`"
        print(f"⚠ {resolver_error}", file=sys.stderr)

    rows = build_rows(moves, activity_index(population), resolver)
    if args.dry_run:
        print(f"\n--dry-run : {Counter(r['status'] for r in rows)}")
        return 0
    missing = predict(rows, booster, spec)

    summary = summarize(rows)
    if missing:
        print(f"\n⚠ Missing variables after encoding (out of "
              f"{summary['n_scored']} scored decisions): {missing}")
    print(f"\nStatuses: {summary['status_counts']}")
    print(f"Scored: {summary['n_scored']}/{summary['n_moves']} decisions "
          f"({100 - summary['excluded_pct']:.1f} %), "
          f"{summary['n_agents_scored']}/{summary['n_agents']} persons")
    print(f"Choice set size (offered modes): {summary['offer_sizes']}")
    if summary["offered_mass_mean"] is not None:
        print(f"Offered mass before renormalisation: mean "
              f"{summary['offered_mass_mean']:.3f}, median "
              f"{summary['offered_mass_median']:.3f}, p10 "
              f"{summary['offered_mass_p10']:.3f}, min "
              f"{summary['offered_mass_min']:.3f}")
    print(f"Most likely mode moved by the renormalisation: "
          f"{summary['n_argmax_shifted']} decision(s)")

    before, after = shares(rows, "p_raw_"), shares(rows, "p_")
    print("\nPredicted modal shares (probability mass):")
    print(f"  {'mode':22s} {'avant':>8s} {'après':>8s} {'écart':>8s}")
    for cat in PREDICTABLE_CATS:
        b, a = before.get(cat, 0.0), after.get(cat, 0.0)
        print(f"  {cat:22s} {b:7.1f}% {a:7.1f}% {a - b:+7.1f}")

    meta = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run.get("path"),
        "moves_sha256": (run.get("moves") or {}).get("sha256"),
        "spec_version": spec["spec_version"],
        "policy_format": artefact.get("format"),
        "policy_path": str(policy_path.relative_to(REPO_ROOT)
                           if policy_path.is_relative_to(REPO_ROOT) else policy_path),
        "policy_generated_at": artefact.get("generated_at"),
        # Fingerprint of the POLICY, and not only its date. `spec_version` only moves
        # if the variable contract changes: a retraining with identical
        # variables — more iterations, corrected training set — leaves it at its
        # value, and the page then served a stale parquet as current, silently.
        # This is the symmetric flaw of the one closed on 2026-08-25 for the run, on the
        # model axis. Measured on 2026-08-27: fixing the granularity of the zone codes
        # took the training set from 27,886 to 52,248 trips without touching the
        # spec — exactly the case this fingerprint makes visible.
        "policy_sha256": _digest(policy_path),
        "classes": list(spec["target"]["classes"]),
        "class_to_cat": POLICY_CLASS_TO_CAT,
        "exclude_selection_methods":
            manifest.get("common_set.exclude_selection_methods", []),
        "zones": (manifest.get("arms.model.zones") if resolver is not None else None),
        "zones_error": resolver_error or None,
        "zone_coverage": resolver.coverage() if resolver is not None else None,
        "summary": summary,
        "feature_missing": missing,
        "shares_before": before,
        "shares_after": after,
    }
    write_parquet(rows, out_path, meta)
    rel = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
    print(f"\nWritten: {rel} ({len(rows)} rows)")
    print("Regenerate the page: make synthesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
