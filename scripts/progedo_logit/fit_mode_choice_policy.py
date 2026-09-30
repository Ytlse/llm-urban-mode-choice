"""fit_mode_choice_policy.py — Training of the mode-choice policy.

Trains the multiclass LightGBM booster announced by ticket 005 (phase 2) on the
set prepared by `build_mode_choice_dataset.py`, and serialises it into a
self-contained artefact: `mode_choice_policy.json`.

**What "self-contained" means here.** The model's consumer (action A8, then
the run evaluator of phase 3b) must have nothing to guess or to re-read:
the artefact embeds the exact order of the variables, the encoding table of each
categorical category, the order of the output classes, the version of the features
contract, and the booster itself in two forms — the JSON `dump_model()` intended for
a pure-Python evaluator (decision E9, revised on 2026-09-08: `lightgbm` and `libgomp1`
are now in the `controller` image, the evaluator was never written; the form
is still exported, it costs nothing) and LightGBM's native text format, which allows
reloading the booster identically wherever the library is present.
The parquet is needed only here.

**Three safeguards, before anything else.** They keep the model on its rails:

1. `distance_km`, `crow_km` and `duration_min` are contaminated (ticket 005 §1: for
   walking, distance is an affine function of the declared duration) and the spec
   lists them under `diagnostic_only`. The script explicitly checks that none enters
   the training matrix. A walking PR-AUC of 0.985 is the symptom of the leak,
   not of a good model.
2. The train/test split is **read** from the parquet's `split` column, never
   redone: it is sealed by household (`hh_id`), and a locally redrawn split
   would mix the trips of one household between the two sides.
3. `sample_weight` (the survey's `COEP` weighting coefficient) weights
   training **and** all metrics. The goal is to reproduce modal
   shares: unweighted, they are not representative.

Early stopping uses a validation share re-split **within the train** and
by household: using the test to stop would amount to choosing it, and the reported
figure would no longer be a generalisation figure.

Usage:
    python -m scripts.progedo_logit.fit_mode_choice_policy [--out-dir DIR]

Does **not** require the raw PROGEDO data (restricted access lil-1750): the parquet and
the spec are versioned in the repository.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripts.progedo_logit.mode_choice_eval import evaluate_proba, format_shares

# --- Artefact format version ------------------------------------------------
# Describes the *structure* of mode_choice_policy.json, independently of
# `spec_version`, which describes the features contract. A consumer must refuse a
# format it does not know rather than interpreting its keys at random.
POLICY_FORMAT = "lightgbm_mode_choice_policy"
POLICY_FORMAT_VERSION = 1

# --- Training settings ------------------------------------------------------
# Deliberately sober: 21 features, ~21 k training rows, and an artefact
# that must stay diffable in git. `deterministic` + `force_row_wise` fix the order
# of floating-point reductions, otherwise multithreading makes the result non
# reproducible from one run to the next.
SEED = 0
VALID_FRACTION = 0.2          # share of the train reserved for early stopping
# The cap must stay well above the actual early stop (~1,500 rounds
# since the tuning of 2026-08-30). A configuration that *reaches* the cap has not
# converged: its figure is truncated, and nothing in the log says so unless it is
# checked. `best_iteration` is reported at the end of training, compare it.
MAX_ROUNDS = 4000
EARLY_STOPPING_ROUNDS = 50

PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    # Settings from the `tune_mode_choice_policy.py` bench (2026-08-30): random
    # search then refinement, 96 distinct configurations, 5-fold cross-validation by
    # household **inside the train**, the test split never having been read.
    #
    # The result fits in one sentence: the previous model was over-capacity, and
    # the bike paid for it. Going from 31 to 5 leaves, with a step three times
    # shorter and three times more rounds, improves *simultaneously* the global
    # log-loss, the bike likelihood, its PR-AUC, the calibration (ECE) and the L1 of
    # modal shares. Many shallow trees are better here than few deep
    # trees: a class at 4.3 % does not populate the leaves of a 31-leaf tree
    # enough for its probability to be estimated there on anything but noise.
    "learning_rate": 0.015,
    "num_leaves": 5,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,            # without `freq`, `bagging_fraction` is silently ignored
    "lambda_l1": 0.5,
    "lambda_l2": 10.0,
    # Smoothing of rare categories towards the global mean: `purpose = education` and
    # `socioprofessional_class = Farmer` do not have enough bike observations to
    # deserve a branch of their own.
    "cat_smooth": 50.0,
    "min_data_per_group": 50,
    "max_cat_threshold": 16,
    "path_smoothing": 5.0,        # regularises sparsely populated leaves — the minority ones
    # E7: neither `is_unbalance` nor `class_weight`. Rebalancing the classes destroys
    # calibration (bike precision 0.14 in the original notebook), yet it is the
    # probabilities, not the accuracy, that serve to produce modal shares. The gain
    # on under-represented modes came from *better-placed capacity*,
    # not from reweighting — and the bench checks it: every configuration that
    # degraded the modal-share L1 by more than 0.005 was ruled out outright.
    "verbosity": -1,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
    "seed": SEED,
    "data_random_seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Repository root, located by the spec rather than by the raw data.

    `build_mode_choice_dataset.find_project_root` looks for `data/PROGEDO 2023/`, which is
    restricted-access. Training, for its part, only reads versioned files: it must
    run on a bare clone.
    """
    root = Path(__file__).resolve()
    marker = Path("scripts") / "progedo_logit" / "feature_spec.json"
    while not (root / marker).exists() and root != root.parent:
        root = root.parent
    if not (root / marker).exists():
        raise SystemExit(f"Project root not found (marker: {marker}).")
    return root


# ---------------------------------------------------------------------------
# Features contract
# ---------------------------------------------------------------------------

def check_spec(spec: dict, df: pd.DataFrame) -> None:
    """Refuses to train on a set that does not match the contract.

    Each check corresponds to a known way of producing a wrong model without
    any exception being raised.
    """
    names = feature_names(spec)
    diagnostic = set(spec.get("diagnostic_only") or [])

    leaked = sorted(diagnostic & set(names))
    if leaked:
        raise SystemExit(
            f"`diagnostic_only` variables present in the features: {leaked}. "
            "They are contaminated (ticket 005 §1) and must never enter "
            "the model."
        )

    missing = [n for n in names if n not in df.columns]
    if missing:
        raise SystemExit(f"Columns missing from the dataset: {missing}")

    for col in (spec["target"]["name"], spec["sample_weight"], "split", "hh_id"):
        if col not in df.columns:
            raise SystemExit(f"Mandatory column missing from the dataset: {col}")

    observed = set(df[spec["target"]["name"]].dropna().unique())
    declared = set(spec["target"]["classes"])
    if observed - declared:
        raise SystemExit(
            f"Target modalities outside the spec: {sorted(observed - declared)}"
        )

    for feature in spec["features"]:
        if feature["kind"] != "categorical":
            continue
        seen = set(df[feature["name"]].dropna().astype(str).unique())
        unknown = sorted(seen - set(feature["categories"]))
        if unknown:
            raise SystemExit(
                f"Modalities outside the spec for {feature['name']}: {unknown}. "
                "The spec freezes the closed list: regenerate it with "
                "build_mode_choice_dataset.py."
            )


def feature_names(spec: dict) -> list[str]:
    """Order of the matrix columns — the spec imposes it, not the parquet."""
    return [f["name"] for f in spec["features"]]


def categorical_encoding(spec: dict) -> dict[str, dict[str, int]]:
    """Category → integer code, in the spec order.

    Freezing the codes here, and republishing them in the artefact, is what guarantees that a
    category will be encoded identically at training and at runtime. Inferring them
    from the values present in a prediction batch would give an encoding that drifts
    from one call to the next.
    """
    return {
        f["name"]: {cat: i for i, cat in enumerate(f["categories"])}
        for f in spec["features"] if f["kind"] == "categorical"
    }


def encode_features(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Numeric matrix ready for LightGBM, in the spec order.

    - categorical → integer code of the spec, `NaN` for the unknown (never a fallback
      code: "unexpected category" is not "most frequent category");
    - boolean     → 0/1, `NaN` kept when the value is missing;
    - numeric     → float, `NaN` kept.

    Missing values are **not** imputed: LightGBM routes them natively,
    and density is legitimately absent for the 81 zones with no surveyed household (a
    0 there would assert "empty zone", which is false).
    """
    encoding = categorical_encoding(spec)
    out = pd.DataFrame(index=df.index)
    for feature in spec["features"]:
        name, kind = feature["name"], feature["kind"]
        col = df[name]
        if kind == "categorical":
            out[name] = col.astype("object").map(encoding[name]).astype("float64")
            _alarme_si_hors_spec(name, col, out[name])
        elif kind == "bool":
            # A direct `astype(float)` fails on an object column containing NaN.
            out[name] = pd.to_numeric(col.astype("object").map(
                {True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}), errors="coerce")
        else:
            out[name] = pd.to_numeric(col, errors="coerce").astype("float64")
    return out[feature_names(spec)]


#: Share of out-of-spec values beyond which a categorical no longer measures anything.
#
# 40 % is not a fine tuning: below it, a cohort can legitimately contain
# categories the survey does not produce (`socioprofessional_class = "Retired"`, a
# documented case of the spec). Beyond it, it is no longer an exception, it is a vocabulary
# mismatch — and it is paid for in silence.
SEUIL_HORS_SPEC = 0.40

_hors_spec_alarme: set = set()


def _alarme_si_hors_spec(nom: str, brut, encode) -> None:
    """Alarm when a categorical switches massively to `__missing__`.

    The artefact's contract says: "zero coefficient: an absent value behaves like
    the reference category". That is a reasonable fallback for a few rows. Applied to
    ALL of them, it removes the variable from the model without raising any exception or
    writing a single log line: the prediction stays plausible, the metrics stay computable,
    and nothing says that the model no longer sees anybody's occupation.

    This case nearly happened at ticket 074: the v6 cohort carries `main_occupation` in
    English where the spec encodes it in French. Translation happens at the boundary
    (`model_on_common_set.occupation_du_spec`), but a guard is worth more than a table one
    believes complete — the next category added on one side and not the other will land here.

    Emitted once per variable: an alarm repeated at every batch would drown the log.
    """
    import numpy as np

    renseignes = brut.notna().sum()
    if not renseignes:
        return
    perdus = int((encode.isna() & brut.notna()).sum())
    part = perdus / renseignes
    if part < SEUIL_HORS_SPEC or nom in _hors_spec_alarme:
        return
    _hors_spec_alarme.add(nom)
    exemples = sorted({str(v) for v in brut[encode.isna() & brut.notna()].head(200)})[:4]
    print(
        f"[ALARME] {nom}: {perdus}/{renseignes} values ({100 * part:.1f} %) outside the "
        f"feature_spec — they become `__missing__`, with a zero coefficient, so the "
        f"variable no longer weighs anything for these rows. Examples: {exemples}. "
        f"Cohort and spec vocabularies in disagreement?",
        file=__import__("sys").stderr,
    )


def categorical_indices(spec: dict) -> list[int]:
    """Positions of the categoricals — LightGBM wants them by index, not by name."""
    return [i for i, f in enumerate(spec["features"]) if f["kind"] == "categorical"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def split_valid(train: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    """Validation share carved out **within the train**, by household.

    By household for the same reason as the main split: the trips of a
    household share its car equipment. Within the train, because stopping on the
    test amounts to selecting it.
    """
    fit_idx, valid_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        .split(train, groups=train["hh_id"])
    )
    is_valid = pd.Series(False, index=train.index)
    is_valid.iloc[valid_idx] = True
    return is_valid


def train_booster(X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
                  is_valid: np.ndarray, spec: dict,
                  params: Optional[dict] = None) -> tuple[lgb.Booster, dict]:
    """Fits the booster, early stopping on the validation share."""
    params = dict(params or PARAMS)
    params["num_class"] = len(spec["target"]["classes"])

    cat = categorical_indices(spec)
    fit_set = lgb.Dataset(X[~is_valid], label=y[~is_valid], weight=w[~is_valid],
                          categorical_feature=cat, free_raw_data=False)
    valid_set = lgb.Dataset(X[is_valid], label=y[is_valid], weight=w[is_valid],
                            categorical_feature=cat, reference=fit_set,
                            free_raw_data=False)

    history: dict = {}
    booster = lgb.train(
        params, fit_set, num_boost_round=MAX_ROUNDS, valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.record_evaluation(history),
            lgb.log_evaluation(period=100),
        ],
    )
    training = {
        "params": {k: v for k, v in params.items() if k != "verbosity"},
        "max_rounds": MAX_ROUNDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(booster.best_iteration),
        "valid_multi_logloss": float(history["valid"]["multi_logloss"][booster.best_iteration - 1]),
        "n_fit": int((~is_valid).sum()),
        "n_valid": int(is_valid.sum()),
    }
    return booster, training


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(booster: lgb.Booster, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
             classes: list[str]) -> dict:
    """Test split metrics — computed by the shared module, never here.

    `mode_choice_eval.evaluate_proba` judges a **probability matrix**, so the
    booster and the multinomial logit of the second oracle are scored by the same code: otherwise
    the published comparison would mix the gap between two models with the gap between two
    metric implementations.
    """
    proba = booster.predict(X, num_iteration=booster.best_iteration)
    return evaluate_proba(proba, y, w, classes)


def importances(booster: lgb.Booster, names: list[str]) -> list[dict]:
    """Gain importances, descending — used as a leakage diagnostic."""
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    total = float(gains.sum()) or 1.0
    rows = [{"name": n, "gain_share": float(g / total), "splits": int(s)}
            for n, g, s in zip(names, gains, splits)]
    return sorted(rows, key=lambda r: -r["gain_share"])


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def build_policy(booster: lgb.Booster, spec: dict, spec_path: Path,
                 dataset_path: Path, training: dict, metrics: dict) -> dict:
    """Self-contained artefact: everything needed to predict, and nothing from the parquet."""
    return {
        "format": POLICY_FORMAT,
        "format_version": POLICY_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": spec.get("source"),
        # The runtime compares this version with that of the spec it reads: a model
        # trained under another features contract must be refused, not reinterpreted.
        "spec_version": spec["spec_version"],
        "spec_file": spec_path.name,
        "dataset_file": dataset_path.name,
        "target": {"name": spec["target"]["name"], "classes": spec["target"]["classes"]},
        # Exact column order expected by the booster.
        "features": [
            {"name": f["name"], "kind": f["kind"], "source": f["source"],
             **({"categories": f["categories"]} if f["kind"] == "categorical" else {})}
            for f in spec["features"]
        ],
        "encoding": {
            "categorical": categorical_encoding(spec),
            "bool": {"false": 0, "true": 1},
            "missing": "null / NaN, routé nativement par le booster — jamais imputé",
            "unknown_category": "null / NaN (aucun code de repli)",
        },
        # Copied so that the consumer can check it computes od_km and
        # dist_center_* from the same reference (safeguard of ZoneResolver.load).
        "geo_reference": spec.get("geo_reference"),
        "domain": spec.get("domain"),
        "notes": spec.get("notes"),
        "training": training,
        "metrics": metrics,
        "booster": {
            "library": "lightgbm",
            "version": lgb.__version__,
            "best_iteration": int(booster.best_iteration),
            # Two forms, two consumers (see the module docstring).
            "dump_model": booster.dump_model(num_iteration=booster.best_iteration),
            "model_text": booster.model_to_string(num_iteration=booster.best_iteration),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Training parquet (default: progedo_mode_choice_v2.parquet)")
    parser.add_argument("--spec", type=Path, default=None,
                        help="Features contract (default: feature_spec.json)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: scripts/progedo_logit/)")
    args = parser.parse_args()

    root = find_project_root()
    here = root / "scripts" / "progedo_logit"
    dataset_path = args.dataset or (here / "progedo_mode_choice_v2.parquet")
    spec_path = args.spec or (here / "feature_spec.json")
    out_dir = args.out_dir or here
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(dataset_path)
    check_spec(spec, df)

    names = feature_names(spec)
    classes = spec["target"]["classes"]
    print(f"Set: {len(df)} rows | spec v{spec['spec_version']} | "
          f"{len(names)} features | {len(classes)} classes")
    print(f"Excluded from the model (diagnostic): {spec.get('diagnostic_only')}")

    X = encode_features(df, spec)
    y = df[spec["target"]["name"]].map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = df[spec["sample_weight"]].to_numpy(dtype=float)

    is_train = (df["split"] == "train").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    print(f"Split read from the parquet: train={is_train.sum()} test={is_test.sum()} "
          f"| households={df['hh_id'].nunique()}")

    train = df[is_train]
    is_valid_train = split_valid(train, VALID_FRACTION, SEED).to_numpy()

    booster, training = train_booster(
        X[is_train].reset_index(drop=True), y[is_train], w[is_train],
        is_valid_train, spec)
    print(f"\nEarly stop at iteration {training['best_iteration']} "
          f"(valid multi_logloss = {training['valid_multi_logloss']:.4f})")

    metrics = evaluate(booster, X[is_test].reset_index(drop=True),
                       y[is_test], w[is_test], classes)
    metrics["feature_importances"] = importances(booster, names)

    print(f"\nTest ({metrics['n_rows']} rows, COEP-weighted):"
          f"\n  log-loss  = {metrics['log_loss_weighted']:.4f}"
          f"\n  accuracy  = {metrics['accuracy_weighted']:.4f}")
    shares = metrics["mode_shares"]
    print("\nModal shares (test, weighted):")
    print(format_shares(classes, shares["observed"],
                        shares["predicted_probability_mass"], shares["predicted_argmax"]))
    print(f"  L1 probability mass = {shares['l1_probability_mass']:.4f}"
          f" | L1 chosen mode = {shares['l1_argmax']:.4f}")

    print("\nImportances (gain, top 8):")
    for row in metrics["feature_importances"][:8]:
        print(f"  {row['name']:22s} {row['gain_share']:6.1%}")

    policy = build_policy(booster, spec, spec_path, dataset_path, training, metrics)
    policy_path = out_dir / "mode_choice_policy.json"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")

    # Metrics kept apart: the model artefact embeds them too, but a dedicated
    # file reads and diffs without wading through several MB of booster.
    report = {
        "generated_at": policy["generated_at"],
        "spec_version": spec["spec_version"],
        "dataset": dataset_path.name,
        "split": spec.get("split"),
        "training": training,
        "test": metrics,
    }
    metrics_path = out_dir / "mode_choice_policy_metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    size_mb = policy_path.stat().st_size / 1e6
    print(f"\nWritten:\n - {policy_path} ({size_mb:.1f} MB, format "
          f"{POLICY_FORMAT} v{POLICY_FORMAT_VERSION})\n - {metrics_path}")


if __name__ == "__main__":
    main()
