"""tune_mode_choice_policy.py — Hyperparameter tuning of the mode-choice booster.

`fit_mode_choice_policy.py` trains with settings set by hand and owned up to as
"deliberately sober": they were never searched for. This script searches for them, with
a constraint that structures everything else — **decision E7 holds**. Neither
`is_unbalance` nor `class_weight` is used: reweighting the classes raises bike recall and
destroys calibration, yet it is the probabilities, not the hard labels, that
produce modal shares.

**What can be done instead for under-represented modes.** The bike weighs
1,581 training rows out of 39,203 (4.3 % in weighted mass). Its probability
mass is already right; what it lacks is **discriminating power** —
separating the 4 % of bike trips from the rest. That is gained on the
model's capacity (`num_leaves`, `min_data_in_leaf`, number of rounds) and on the handling of
rare categoricals (`cat_smooth`, `min_data_per_group`), not on reweighting.
The difference is verifiable: a discrimination gain raises the bike PR-AUC
**without** degrading the modal-share L1; a reweighting does the opposite.

**Three methodological safeguards.**

1. **The test split is never read.** All selection is done by cross-validation
   *inside* the train. A hyperparameter chosen on the test turns the generalisation
   figure into a training figure, silently.
2. **The folds are by household** (`hh_id`), like the main split: the trips
   of one household share its car equipment. Folds by trip
   would give an optimistic CV.
3. **Early stopping is internal to the fold.** In each fold, a validation share is
   re-split by household within the 4/5 of training. Stopping on the held-out fold
   would amount to choosing it, and out-of-sample predictions would no longer be
   out-of-sample.

**Selection criteria** (all weighted by the `COEP` weighting):

- `logloss` — primary criterion, the one E7 designates;
- `nll_bike` — negative log-likelihood *on the bike rows only*, i.e.
  "what probability does the model give the bike when the bike was chosen". The
  honest minority criterion: it does not reward shouting bike everywhere,
  since the global `logloss` would penalise it;
- `ap_bike` / `ap_transit` — one-vs-rest PR-AUC, a discrimination measure
  independent of any threshold (argmax recall, for its part, collapses mechanically on a
  4 % class);
- `l1_mass` — total absolute gap between predicted (in mass) and observed modal shares.
  **Calibration safeguard**: a configuration that degrades the L1 beyond the
  tolerance is ruled out whatever its gain on the bike.

Usage:
    python -m scripts.progedo_logit.tune_mode_choice_policy [--trials N] [--folds K]

Writes `mode_choice_tuning.json`: all the configurations tried with their
metrics, sorted. This file is the trace of the experiment — it modifies no
model. Carrying the winner over into `PARAMS` of `fit_mode_choice_policy.py` remains a human
step, followed by a `make policy`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from scripts.progedo_logit.fit_mode_choice_policy import (
    EARLY_STOPPING_ROUNDS,
    MAX_ROUNDS,
    PARAMS,
    SEED,
    VALID_FRACTION,
    categorical_indices,
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
)

# Calibration tolerance: a configuration that degrades the modal-share L1 by
# more than this amount (in probability points summed over the 4 classes) is ruled out,
# even if it wins on the bike. It is the wall that stops the search from rediscovering
# class reweighting through a back door.
L1_TOLERANCE = 0.005

# Search space. Each axis is there for a reason tied to the rare classes:
#   num_leaves / min_data_in_leaf : the capacity to isolate a 4 % pocket;
#   learning_rate + rounds         : a shorter step leaves time to model it;
#   cat_smooth / min_data_per_group: the smoothing of rare categories (`purpose`,
#                                    `socioprofessional_class`) towards the global mean;
#   path_smoothing                 : regularises sparsely populated leaves, exactly
#                                    those where the minority modes live;
#   objective multiclassova        : one-vs-rest, each class has its own budget
#                                    of trees instead of sharing a softmax.
SEARCH_SPACE = {
    "objective": ["multiclass", "multiclass", "multiclass", "multiclassova"],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [10, 20, 30, 50, 80],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "lambda_l1": [0.0, 0.5, 2.0],
    "lambda_l2": [0.0, 1.0, 5.0, 20.0],
    "cat_smooth": [1.0, 10.0, 50.0],
    "min_data_per_group": [20, 50, 100],
    "max_cat_threshold": [16, 32],
    "path_smoothing": [0.0, 1.0, 10.0],
}

# Refinement space (`--refine`). The first pass brought out a clear signal:
# all the configurations of the three podiums sit at `num_leaves = 15`, the **lower**
# bound of the grid. An optimum on a grid edge is not an optimum: it only says
# that the search stopped too early. This space goes lower in
# capacity and tightens the step, because that is exactly what a 4 % class
# needs — fewer leaves, hence more populated leaves, hence bike
# probabilities estimated on enough people to be worth something.
REFINE_SPACE = {
    "objective": ["multiclass"],
    "learning_rate": [0.015, 0.02, 0.03, 0.05],
    "num_leaves": [7, 10, 12, 15, 20, 24],
    "min_data_in_leaf": [5, 10, 20, 30, 50],
    "feature_fraction": [0.5, 0.6, 0.7, 0.8],
    "bagging_fraction": [0.8, 0.9, 1.0],
    "lambda_l1": [0.0, 0.5, 2.0],
    "lambda_l2": [2.0, 5.0, 10.0, 20.0],
    "cat_smooth": [10.0, 25.0, 50.0, 100.0],
    "min_data_per_group": [30, 50, 100],
    "max_cat_threshold": [16, 32],
    "path_smoothing": [0.0, 1.0, 5.0, 10.0],
}

FIXED = {
    "metric": "multi_logloss",
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
# Metrics
# ---------------------------------------------------------------------------

def per_class_metrics(y: np.ndarray, proba: np.ndarray, w: np.ndarray,
                      classes: list[str]) -> dict:
    """Class-by-class discrimination and calibration, all weighted.

    `nll` is the negative log-likelihood restricted to the rows of the class: it
    answers "when this mode was chosen, what mass did the model give it".
    `ap` (one-vs-rest PR-AUC) is the discrimination measure that stays readable on
    a 4 % class, where argmax recall only measures prevalence.
    """
    eps = 1e-15
    out = {}
    for k, name in enumerate(classes):
        mask = y == k
        target = mask.astype(int)
        out[name] = {
            "support_share": float(w[mask].sum() / w.sum()),
            "nll": float(-(w[mask] * np.log(np.clip(proba[mask, k], eps, 1))).sum()
                         / w[mask].sum()),
            "ap": float(average_precision_score(target, proba[:, k], sample_weight=w)),
            "auc": float(roc_auc_score(target, proba[:, k], sample_weight=w)),
            # Calibration of the class: predicted mass / observed mass. 1.0 = right.
            "mass_ratio": float((w * proba[:, k]).sum() / w[mask].sum()),
        }
    return out


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, w: np.ndarray,
                               bins: int = 15) -> float:
    """ECE on the confidence (max probability), COEP-weighted."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = w.sum()
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        ece += wb / total * abs(
            (w[m] * correct[m]).sum() / wb - (w[m] * conf[m]).sum() / wb)
    return float(ece)


def score_oof(y: np.ndarray, proba: np.ndarray, w: np.ndarray,
              classes: list[str]) -> dict:
    """Full dashboard of a set of out-of-sample predictions."""
    k = len(classes)
    observed = np.array([w[y == c].sum() / w.sum() for c in range(k)])
    mass = (proba * w[:, None]).sum(axis=0) / w.sum()
    hard = proba.argmax(axis=1)
    argmax_share = np.array([w[hard == c].sum() / w.sum() for c in range(k)])
    per_class = per_class_metrics(y, proba, w, classes)
    return {
        "logloss": float(log_loss(y, proba, labels=list(range(k)), sample_weight=w)),
        "accuracy": float((w * (hard == y)).sum() / w.sum()),
        "ece": expected_calibration_error(y, proba, w),
        "l1_mass": float(np.abs(mass - observed).sum()),
        "l1_argmax": float(np.abs(argmax_share - observed).sum()),
        "mode_shares": {"observed": observed.tolist(),
                        "predicted_mass": mass.tolist(),
                        "predicted_argmax": argmax_share.tolist()},
        "per_class": per_class,
        # Shortcuts for sorting and display.
        "nll_bike": per_class["bike"]["nll"],
        "ap_bike": per_class["bike"]["ap"],
        "ap_transit": per_class["transit"]["ap"],
        # Unweighted mean of the per-class NLLs: treats the 4 modes equally,
        # unlike the global log-loss, which the car dominates at 57 %.
        "macro_nll": float(np.mean([per_class[c]["nll"] for c in classes])),
    }


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------

def cross_val_oof(params: dict, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
                  groups: np.ndarray, spec: dict, folds: int,
                  max_rounds: int = MAX_ROUNDS,
                  ) -> tuple[np.ndarray, list[int], float]:
    """Out-of-sample predictions over the whole train, by household folds.

    Each fold redoes an early stop *within* its 4/5, on a validation share
    itself carved out by household. The held-out fold only comes in at
    prediction: this is what makes the aggregated probabilities truly
    out-of-sample.
    """
    cat = categorical_indices(spec)
    n_classes = len(spec["target"]["classes"])
    oof = np.zeros((len(y), n_classes))
    iterations: list[int] = []
    started = time.perf_counter()

    full = dict(FIXED)
    full.update(params)
    full["num_class"] = n_classes

    for fit_idx, held_idx in GroupKFold(n_splits=folds).split(X, y, groups=groups):
        inner, valid = next(
            GroupShuffleSplit(n_splits=1, test_size=VALID_FRACTION,
                              random_state=SEED).split(fit_idx, groups=groups[fit_idx]))
        inner_idx, valid_idx = fit_idx[inner], fit_idx[valid]

        fit_set = lgb.Dataset(X.iloc[inner_idx], label=y[inner_idx],
                              weight=w[inner_idx], categorical_feature=cat,
                              free_raw_data=False)
        valid_set = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx],
                                weight=w[valid_idx], categorical_feature=cat,
                                reference=fit_set, free_raw_data=False)
        booster = lgb.train(
            full, fit_set, num_boost_round=max_rounds, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        iterations.append(int(booster.best_iteration))
        oof[held_idx] = booster.predict(X.iloc[held_idx],
                                        num_iteration=booster.best_iteration)

    # `multiclassova` produces one-vs-rest sigmoids whose sum is not 1.
    # We renormalise here rather than letting each metric do it on its own:
    # modal shares in probability mass would make no sense otherwise.
    oof /= oof.sum(axis=1, keepdims=True)
    return oof, iterations, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def sample_config(rng: np.random.Generator, space: dict) -> dict:
    """One draw from the search space, normalised for LightGBM."""
    cfg = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
    # `bagging_fraction` without `bagging_freq` is silently ignored — the classic
    # trap: one thinks one is regularising, nothing happens.
    cfg["bagging_freq"] = 0 if cfg["bagging_fraction"] >= 1.0 else 1
    # Native types: `json.dumps` refuses numpy integers.
    return {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                else float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in cfg.items()}


def baseline_config() -> dict:
    """The settings currently in production, evaluated on the same bench."""
    return {k: v for k, v in PARAMS.items()
            if k not in FIXED and k not in ("num_class",)}


def describe(cfg: dict) -> str:
    keys = ["objective", "learning_rate", "num_leaves", "min_data_in_leaf",
            "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2",
            "cat_smooth", "min_data_per_group", "path_smoothing"]
    return " ".join(f"{k.split('_')[0][:4]}{'_'.join(k.split('_')[1:])[:3]}={cfg[k]}"
                    for k in keys if k in cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=48,
                        help="Randomly drawn configurations (excluding the reference)")
    parser.add_argument("--folds", type=int, default=5, help="Folds of the grouped CV")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                        help="Round cap. To watch: a configuration that "
                             "reaches it has not converged, its figure is truncated and "
                             "not comparable to the others.")
    parser.add_argument("--refine", action="store_true",
                        help="Draw from REFINE_SPACE (neighbourhood of the winner of the "
                             "first pass) rather than from SEARCH_SPACE")
    parser.add_argument("--seed-configs", type=Path, default=None,
                        help="JSON {name: params} evaluated on top of the draw — used to "
                             "replay the winners of a previous pass on the "
                             "same bench")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = find_project_root()
    here = root / "scripts" / "progedo_logit"
    dataset_path = args.dataset or (here / "progedo_mode_choice_v2.parquet")
    spec_path = args.spec or (here / "feature_spec.json")
    out_path = args.out or (here / "mode_choice_tuning.json")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(dataset_path)
    check_spec(spec, df)

    classes = spec["target"]["classes"]
    is_train = (df["split"] == "train").to_numpy()
    train = df[is_train].reset_index(drop=True)

    X = encode_features(train, spec)
    y = train[spec["target"]["name"]].map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = train[spec["sample_weight"]].to_numpy(dtype=float)
    groups = train["hh_id"].to_numpy()

    print(f"Tuning bench — {len(train)} train rows, {len(np.unique(groups))} "
          f"households, {args.folds} grouped folds. The test split is not read.")
    print(f"Weighted support: " + "  ".join(
        f"{c}={w[y == i].sum() / w.sum():.1%}" for i, c in enumerate(classes)))

    space = REFINE_SPACE if args.refine else SEARCH_SPACE
    print(f"Space: {'REFINE_SPACE' if args.refine else 'SEARCH_SPACE'}")
    rng = np.random.default_rng(SEED + (1 if args.refine else 0))
    configs = [("référence", baseline_config())]
    seen = {json.dumps(configs[0][1], sort_keys=True)}
    if args.seed_configs:
        for name, cfg in json.loads(args.seed_configs.read_text()).items():
            cfg.setdefault("bagging_freq", 0 if cfg.get("bagging_fraction", 1.0) >= 1.0 else 1)
            configs.append((name, cfg))
            seen.add(json.dumps(cfg, sort_keys=True))
    n_target = args.trials + len(configs)
    while len(configs) < n_target:
        cfg = sample_config(rng, space)
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        configs.append((f"t{len(configs):03d}", cfg))

    results = []
    for name, cfg in configs:
        oof, iterations, elapsed = cross_val_oof(
            cfg, X, y, w, groups, spec, args.folds, args.max_rounds)
        metrics = score_oof(y, oof, w, classes)
        capped = max(iterations) >= args.max_rounds
        results.append({"name": name, "params": cfg, "metrics": metrics,
                        "best_iterations": iterations, "hit_round_cap": capped,
                        "seconds": round(elapsed, 1)})
        print(f"  {name:10s}{' [PLAFOND]' if capped else ''} "
              f"logloss={metrics['logloss']:.4f} "
              f"nll_bike={metrics['nll_bike']:.4f} ap_bike={metrics['ap_bike']:.4f} "
              f"l1={metrics['l1_mass']:.4f} ece={metrics['ece']:.4f} "
              f"iters~{int(np.median(iterations))} ({elapsed:.0f}s)")

    baseline = results[0]
    l1_ceiling = baseline["metrics"]["l1_mass"] + L1_TOLERANCE
    eligible = [r for r in results if r["metrics"]["l1_mass"] <= l1_ceiling]
    rejected = [r["name"] for r in results if r not in eligible]

    by_logloss = sorted(eligible, key=lambda r: r["metrics"]["logloss"])
    by_bike = sorted(eligible, key=lambda r: r["metrics"]["nll_bike"])
    by_macro = sorted(eligible, key=lambda r: r["metrics"]["macro_nll"])

    print(f"\n{len(eligible)}/{len(results)} configurations pass the calibration "
          f"safeguard (L1 ≤ {l1_ceiling:.4f}).")
    if rejected:
        print(f"Ruled out for modal-share drift: {', '.join(rejected)}")

    def podium(title: str, ranked: list[dict], key: str) -> None:
        print(f"\n{title}")
        for r in ranked[:5]:
            m = r["metrics"]
            print(f"  {r['name']:10s} {key}={m[key]:.4f}  logloss={m['logloss']:.4f}  "
                  f"nll_bike={m['nll_bike']:.4f}  ap_bike={m['ap_bike']:.4f}  "
                  f"l1={m['l1_mass']:.4f}\n             {describe(r['params'])}")

    podium("Meilleures sur le log-loss global (critère E7) :", by_logloss, "logloss")
    podium("Meilleures sur le vélo (NLL des lignes vélo) :", by_bike, "nll_bike")
    podium("Meilleures sur la moyenne macro des 4 classes :", by_macro, "macro_nll")

    b = baseline["metrics"]
    print("\nRéférence actuelle, par classe (CV hors-échantillon) :")
    for c in classes:
        p = b["per_class"][c]
        print(f"  {c:9s} support={p['support_share']:6.1%}  nll={p['nll']:.4f}  "
              f"ap={p['ap']:.4f}  auc={p['auc']:.4f}  masse/observé={p['mass_ratio']:.3f}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_version": spec["spec_version"],
        "dataset": dataset_path.name,
        "protocol": {
            "folds": args.folds,
            "grouped_by": "hh_id",
            "early_stopping": "interne au pli, part de validation par ménage",
            "test_split_used": False,
            "l1_tolerance": L1_TOLERANCE,
            "seed": SEED,
            "max_rounds": args.max_rounds,
        },
        "n_train_rows": int(len(train)),
        "baseline": baseline,
        "l1_ceiling": l1_ceiling,
        "rejected_for_calibration": rejected,
        "best_by_logloss": by_logloss[0]["name"] if by_logloss else None,
        "best_by_nll_bike": by_bike[0]["name"] if by_bike else None,
        "best_by_macro_nll": by_macro[0]["name"] if by_macro else None,
        "trials": sorted(results, key=lambda r: r["metrics"]["logloss"]),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
