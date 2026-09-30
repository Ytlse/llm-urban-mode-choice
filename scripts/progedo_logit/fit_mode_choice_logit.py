"""fit_mode_choice_logit.py — Estimation of the second oracle: multinomial logit.

Estimates the **multinomial logit** announced by the paper (contribution C1, level 3 of
the ablation, `exp_03a_multinomial_logit`) on the same survey microdata as the LightGBM
booster, and serialises it to `mnl_model.json`.

**Why a second oracle, and not only the booster.** The comparative literature does not
name the same winner depending on the family of indicators: gradient-boosted trees
beat the logit on *disaggregate* accuracy (2.2 to 5.6 points on three real datasets), and
the logit regains the edge on **aggregate modal shares** and behavioural
indicators (Martín-Baos et al. 2023; "behaviorally unreasonable" caveat of Zhao et
al. 2020 on the elasticities of tree models). Our hypothesis H0 is decided on an L1 of
modal shares: the reference ceiling may therefore be the logit, and asserting it without having
estimated it was not verifiable.

**Strict parity — route 1, decided on 2026-09-10.** The 21 variables of `feature_spec.json`,
not one more. This contract holds **no level-of-service variable per mode**
(time or cost of each alternative): this model is therefore a multinomial logistic
regression on individual, purpose and geography characteristics — **not** a
random utility model in McFadden's sense. Consequence to declare, not to work around:
neither value of time nor willingness to pay can be computed here. Route 2 (per-alternative
utilities fed by `mode_skims.parquet`) would break information parity with the
booster and with the agent, and is the subject of a separate ticket.

**What makes parity true by construction**, rather than promised:

1. the **same set** (`progedo_mode_choice_v2.parquet`) and the **same `split` column**, sealed
   by household — never a local re-split, which would mix the trips of a household
   between the two sides (sampling trap documented by Hillel 2021);
2. the **same `sample_weight`** (COEP survey weighting) at estimation and in all
   metrics;
3. the **same encoding** — `encode_features` from the booster script, imported as is;
4. the **same metrics**, produced by `mode_choice_eval.evaluate_proba`, a module shared
   by both oracles.

**Tuning never reads the test.** The regularisation strength is chosen by cross-validation
**grouped by household inside the train**, like the booster's hyperparameters
(`make policy-tune`). Choosing on the test would amount to selecting it, and the published figure
would no longer be a generalisation figure.

Usage:
    python -m scripts.progedo_logit.fit_mode_choice_logit [--out-dir DIR] [--C 1.0]

Does **not** require the raw PROGEDO data (restricted access lil-1750): the parquet and the
spec are versioned in the repository.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from scripts.progedo_logit.fit_mode_choice_policy import (
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
)
from scripts.progedo_logit.mode_choice_eval import (
    evaluate_proba,
    format_shares,
)
from scripts.progedo_logit.mode_choice_logit import (
    LOGIT_FORMAT,
    LOGIT_FORMAT_VERSION,
    MISSING_CATEGORY,
    LogitPredictor,
    design_columns,
    design_matrix,
)

# --- Estimation settings ----------------------------------------------------
# `lbfgs` on the multinomial likelihood: it is the multinomial logit estimator, not
# a stack of binary logits. The iteration cap is well above the
# observed convergence (~150 rounds) — an estimation that reaches it has not converged, and the
# script says so instead of publishing truncated coefficients.
SOLVER = "lbfgs"
MAX_ITER = 2000
TOLERANCE = 1e-6

# Regularisation grid. A reference logit aims to be as close as possible to unpenalised,
# but the design matrix carries rare categories (`purpose = education`,
# `socioprofessional_class = Farmer`) whose coefficients diverge without a minimum of
# ridge. Cross-validation decides, and the grid is published with its score.
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def missing_columns(encoded: pd.DataFrame, spec: dict) -> list[str]:
    """Variables carrying at least one missing value on the rows provided.

    Computed on the **train only**: a missing indicator created from the test
    would bring test information into the model's structure. A variable complete in the
    train and with holes in the test will thus have its missing values imputed with no
    indicator — the fact is counted at prediction (`feature_missing` of the parquet) rather
    than quietly corrected.
    """
    return [name for name in feature_names(spec) if bool(encoded[name].isna().any())]


def standardization(encoded: pd.DataFrame, spec: dict, weights: np.ndarray) -> dict:
    """**Weighted** mean and standard deviation of the numerics, on the train's non-missing.

    Weighted because everything else is: centring on the unweighted mean
    would amount to defining the coefficients' reference point on a population that
    is not the one the model must reproduce.
    """
    stats: dict = {}
    for feature in spec["features"]:
        if feature["kind"] != "numeric":
            continue
        name = feature["name"]
        values = encoded[name].to_numpy(dtype="float64")
        present = ~np.isnan(values)
        w = weights[present]
        v = values[present]
        mean = float(np.average(v, weights=w))
        variance = float(np.average((v - mean) ** 2, weights=w))
        std = float(np.sqrt(variance)) or 1.0
        stats[name] = {"mean": mean, "std": std, "n_present": int(present.sum())}
    return stats


def missing_category_support(encoded_train: pd.DataFrame, spec: dict) -> dict:
    """Training count of each `__missing__` category, per categorical.

    **What this counter prevents believing.** The design matrix always carries a
    `<var>=__missing__` column, but its coefficient is estimated only if the train contains
    such rows. At zero rows, the column is constant and regularisation leaves its
    coefficient at zero: a value absent at prediction then behaves like the
    **reference category**, which is an assumption and not a measurement. Measured on the pinned
    run, `socioprofessional_class` is absent from 498 decisions because the synthetic
    population carries categories that the survey spec does not know — the fact must be
    readable in the artefact, not inferred afterwards.
    """
    out = {}
    for feature in spec["features"]:
        if feature["kind"] != "categorical":
            continue
        n = int(encoded_train[feature["name"]].isna().sum())
        out[feature["name"]] = {
            "n_train": n,
            "identifie": n > 0,
            "sinon": ("coefficient nul : une valeur absente se comporte comme la "
                      f"modalité de référence « {feature['categories'][0]} »"),
        }
    return out


def build_contract(spec: dict, encoded_train: pd.DataFrame,
                   weights_train: np.ndarray) -> dict:
    """`logit` block of the artefact, excluding coefficients: everything that builds `Z`."""
    indicators = missing_columns(encoded_train, spec)
    return {
        "missing_category_support": missing_category_support(encoded_train, spec),
        "design_columns": design_columns(spec, indicators),
        "missing_indicators": indicators,
        "standardization": standardization(encoded_train, spec, weights_train),
        "reference_category": {
            f["name"]: f["categories"][0]
            for f in spec["features"] if f["kind"] == "categorical"
        },
        "missing_rule": {
            "categorical": f"modalité {MISSING_CATEGORY} (jamais la modalité modale)",
            "bool": "0, plus l'indicatrice <nom>__missing à 1",
            "numeric": ("moyenne pondérée du train, soit 0 après centrage, plus "
                        "l'indicatrice <nom>__missing à 1"),
            "note": ("les indicatrices distinguent « absent » de « moyen » ; sans elles "
                     "l'imputation affirmerait une valeur que la donnée ne porte pas"),
        },
        "link": "softmax(Z·coef^T + intercept), classes dans l'ordre du spec",
    }


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def normalized_weights(w: np.ndarray) -> np.ndarray:
    """Survey weights brought to a mean of 1, **for estimation only**.

    The COEP weighting averages 85.5 on this set (min 9.4, max 503.6). Passed as
    is to a penalised estimator, it multiplies the likelihood by ~85 without touching the
    penalty: `C` then means nothing, and the optimiser does not reach its tolerance
    within the iteration cap — as measured, estimation hit the 2,000 rounds. The
    normalisation changes **neither** the relative weights between observations **nor** the
    metrics, which are still computed with the raw COEP; it only makes the scale of
    the regularisation readable.
    """
    return w * (len(w) / w.sum())


def fit_logit(Z: np.ndarray, y: np.ndarray, w: np.ndarray, C: float) -> LogisticRegression:
    """Fits the weighted multinomial logit. `Z` is already standardised and expanded."""
    model = LogisticRegression(
        C=C, solver=SOLVER, max_iter=MAX_ITER, tol=TOLERANCE, fit_intercept=True)
    model.fit(Z, y, sample_weight=normalized_weights(w))
    return model


def select_C(Z: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
             classes: list[str], grid=C_GRID, folds: int = CV_FOLDS) -> dict:
    """Chooses `C` by cross-validation grouped by household, **within the train**.

    The criterion is the weighted negative log-likelihood: it is the quantity the
    probabilities depend on, hence the modal shares. Accuracy does not act as arbiter — a
    model that gains one point of accuracy by crushing the bike degrades the shares.
    """
    splitter = GroupKFold(n_splits=folds)
    labels = list(range(len(classes)))
    scores: list[dict] = []
    for C in grid:
        losses, sizes = [], []
        for fit_idx, valid_idx in splitter.split(Z, y, groups=groups):
            model = fit_logit(Z[fit_idx], y[fit_idx], w[fit_idx], C)
            proba = model.predict_proba(Z[valid_idx])
            losses.append(float(log_loss(y[valid_idx], proba, labels=labels,
                                         sample_weight=w[valid_idx])))
            sizes.append(int(len(valid_idx)))
        scores.append({
            "C": C,
            "cv_log_loss_weighted": float(np.mean(losses)),
            "per_fold": losses,
            "fold_sizes": sizes,
        })
        print(f"  C = {C:<8g} log-loss CV = {scores[-1]['cv_log_loss_weighted']:.5f}")
    best = min(scores, key=lambda s: s["cv_log_loss_weighted"])
    return {
        "grid": [s["C"] for s in scores],
        "folds": folds,
        "grouped_by": "hh_id",
        "scored_on": "train uniquement — le split test n'est jamais lu pour régler",
        "criterion": "log_loss_weighted",
        "results": scores,
        "C": best["C"],
        "cv_log_loss_weighted": best["cv_log_loss_weighted"],
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def build_artefact(model: LogisticRegression, spec: dict, spec_path: Path,
                   dataset_path: Path, contract: dict, training: dict,
                   metrics: dict) -> dict:
    """Self-contained artefact: enough to predict without scikit-learn or parquet."""
    return {
        "format": LOGIT_FORMAT,
        "format_version": LOGIT_FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": spec.get("source"),
        # The consumer compares this version with that of the spec it reads: a model
        # estimated under another variable contract must be refused, not reinterpreted.
        "spec_version": spec["spec_version"],
        "spec_file": spec_path.name,
        "dataset_file": dataset_path.name,
        "target": {"name": spec["target"]["name"], "classes": spec["target"]["classes"]},
        "features": [
            {"name": f["name"], "kind": f["kind"], "source": f["source"],
             **({"categories": f["categories"]} if f["kind"] == "categorical" else {})}
            for f in spec["features"]
        ],
        # Input encoding: the booster's, identical. The expansion into
        # indicators specific to the logit is described in the `logit` block.
        "encoding": {
            "shared_with": "fit_mode_choice_policy.encode_features",
            "missing": "NaN en entrée, traité par logit.missing_rule",
            "unknown_category": f"NaN à l'encodage, puis modalité {MISSING_CATEGORY}",
        },
        "geo_reference": spec.get("geo_reference"),
        "domain": spec.get("domain"),
        "notes": spec.get("notes"),
        "training": training,
        "metrics": metrics,
        "logit": {
            **contract,
            "estimator": {
                "library": "scikit-learn",
                "class": "LogisticRegression",
                "solver": SOLVER,
                "multinomial": True,
                "max_iter": MAX_ITER,
                "tol": TOLERANCE,
            },
            "coef": [[float(v) for v in row] for row in model.coef_],
            "intercept": [float(v) for v in model.intercept_],
        },
    }


def top_coefficients(artefact: dict, k: int = 8) -> list[dict]:
    """Largest-magnitude coefficients, per class — first-level reading.

    A logit coefficient reads on the log-odds scale, and the numerics are
    standardised: "+0.8" reads "one more standard deviation multiplies the odds by e^0.8".
    Published as a diagnostic, like the booster's gain importances.
    """
    logit = artefact["logit"]
    columns = logit["design_columns"]
    rows: list[dict] = []
    for class_index, name in enumerate(artefact["target"]["classes"]):
        coefficients = logit["coef"][class_index]
        ranked = sorted(zip(columns, coefficients), key=lambda p: -abs(p[1]))[:k]
        rows.append({"class": name,
                     "top": [{"column": c, "coef": float(v)} for c, v in ranked]})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Training parquet (default: progedo_mode_choice_v2.parquet)")
    parser.add_argument("--spec", type=Path, default=None,
                        help="Variable contract (default: feature_spec.json)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: scripts/progedo_logit/)")
    parser.add_argument("--C", type=float, default=None,
                        help="Imposed regularisation strength (default: cross-validation)")
    args = parser.parse_args(argv)

    root = find_project_root()
    here = root / "scripts" / "progedo_logit"
    dataset_path = args.dataset or (here / "progedo_mode_choice_v2.parquet")
    spec_path = args.spec or (here / "feature_spec.json")
    out_dir = args.out_dir or here
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(dataset_path)
    check_spec(spec, df)          # same leakage guard as the booster

    names = feature_names(spec)
    classes = spec["target"]["classes"]
    print(f"Set: {len(df)} rows | spec v{spec['spec_version']} | "
          f"{len(names)} variables | {len(classes)} classes")
    print(f"Excluded from the model (diagnostic): {spec.get('diagnostic_only')}")

    encoded = encode_features(df, spec)
    y = df[spec["target"]["name"]].map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = df[spec["sample_weight"]].to_numpy(dtype=float)
    groups = df["hh_id"].to_numpy()

    is_train = (df["split"] == "train").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    print(f"Split read from the parquet: train={is_train.sum()} test={is_test.sum()} "
          f"| households={df['hh_id'].nunique()}")

    contract = build_contract(spec, encoded[is_train], w[is_train])
    print(f"Design matrix: {len(contract['design_columns'])} columns "
          f"({len(contract['missing_indicators'])} missing indicators: "
          f"{contract['missing_indicators']})")

    # The partial artefact serves as the contract for `design_matrix`: the estimation matrix is
    # built by the **same** code as the prediction one, never by a parallel path.
    partial = {"features": spec["features"], "logit": contract}
    Z = design_matrix(encoded, partial)

    if args.C is not None:
        tuning = {"C": args.C, "imposed": True,
                  "scored_on": "aucun réglage — valeur imposée en ligne de commande"}
        print(f"\nImposed regularisation: C = {args.C}")
    else:
        print(f"\n{CV_FOLDS}-fold cross-validation, grouped by household, within the train:")
        tuning = select_C(Z[is_train], y[is_train], w[is_train], groups[is_train], classes)
        print(f"  retenu : C = {tuning['C']}")

    model = fit_logit(Z[is_train], y[is_train], w[is_train], tuning["C"])
    iterations = int(np.max(np.atleast_1d(model.n_iter_)))
    if iterations >= MAX_ITER:
        raise SystemExit(
            f"[ALARME] Estimation reached the cap of {MAX_ITER} iterations: the "
            "coefficients have not converged and must not be published.")

    training = {
        "estimator": "multinomial logistic regression (lbfgs, pondérée COEP)",
        "C": tuning["C"],
        "tuning": tuning,
        "n_iter": iterations,
        "n_fit": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "n_design_columns": int(Z.shape[1]),
        "sample_weight": spec["sample_weight"],
        "weight_normalization": ("poids ramenés à une moyenne de 1 pour l'estimation ; "
                                 "métriques calculées avec le COEP brut"),
        "split": spec.get("split"),
    }
    print(f"Converged in {iterations} iterations (cap {MAX_ITER})")

    proba_test = model.predict_proba(Z[is_test])
    metrics = evaluate_proba(proba_test, y[is_test], w[is_test], classes)

    print(f"\nTest ({metrics['n_rows']} rows, COEP-weighted):"
          f"\n  CEL (log-loss) = {metrics['cel_weighted']:.4f}"
          f"\n  GMPCA          = {metrics['gmpca_weighted']:.4f}"
          f"\n  accuracy       = {metrics['accuracy_weighted']:.4f}")
    shares = metrics["mode_shares"]
    print("\nModal shares (test, weighted):")
    print(format_shares(classes, shares["observed"],
                        shares["predicted_probability_mass"], shares["predicted_argmax"]))
    print(f"  L1 probability mass = {shares['l1_probability_mass']:.4f}"
          f" | L1 chosen mode = {shares['l1_argmax']:.4f}")

    artefact = build_artefact(model, spec, spec_path, dataset_path, contract,
                              training, metrics)
    artefact["metrics"]["top_coefficients"] = top_coefficients(artefact)

    # Self-containment check, **before** writing: the pure-numpy evaluator must
    # reproduce scikit-learn's probabilities. An artefact that does not read back as it
    # was estimated is a wrong artefact, and nothing would say so on reading.
    replayed = LogitPredictor(artefact).predict(encoded[is_test])
    gap = float(np.abs(replayed - proba_test).max())
    if gap > 1e-9:
        raise SystemExit(
            f"[ALARME] The artefact does not reproduce the estimation: max gap {gap:.2e} "
            "on the test probabilities. Artefact not written.")
    print(f"Self-containment checked: max gap artefact / estimator = {gap:.1e}")

    artefact_path = out_dir / "mnl_model.json"
    artefact_path.write_text(json.dumps(artefact, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")

    report = {
        "generated_at": artefact["generated_at"],
        "spec_version": spec["spec_version"],
        "dataset": dataset_path.name,
        "split": spec.get("split"),
        "training": training,
        "test": metrics,
    }
    metrics_path = out_dir / "mnl_model_metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    size_kb = artefact_path.stat().st_size / 1e3
    print(f"\nWritten:\n - {artefact_path} ({size_kb:.0f} kB, format "
          f"{LOGIT_FORMAT} v{LOGIT_FORMAT_VERSION})\n - {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
