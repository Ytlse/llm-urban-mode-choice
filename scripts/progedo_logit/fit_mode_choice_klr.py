"""fit_mode_choice_klr.py — Estimation of the third family: kernel logistic regression (KLR).

Estimates a **Nyström-approximated RBF-kernel logistic regression** on the same
survey microdata as the LightGBM booster and the multinomial logit, and serialises it to
`klr_model.json` (format `klr_mode_choice_policy` v1).

**Why a third family.** The repository held the two ends of one axis — an
exact, non-smooth booster (0.785 test accuracy, erratic elasticities by
construction), a smooth, less exact logit (0.766) — and nothing in between. This is
exactly the trade-off Martín-Baos et al. (2023) study, and their conclusion names
KLR as the best balance between predictive accuracy and behavioural
plausibility. Dual use, the second perhaps worth more: a third figure
at level 3 of the ablation, and a **second arbiter** for block C of the two-oracle score —
a single arbiter cannot be refuted.

The declared target is to land **between** the two on accuracy while staying smooth in
elasticities. Hoping to beat the booster would be contradicted by Wang et al. (2024), who
conclude over hundreds of models that the data context weighs more than the
family chosen.

**What makes parity true by construction** (rules K1 to K3), rather than promised:

1. the **same set** and the **same `split` column**, sealed by household;
2. the **same `sample_weight`** (COEP) at estimation and in all metrics;
3. the **same `encode_features`**, and above all the **same design matrix** as the logit,
   built by the same function (`design_matrix_from_contract`): an RBF kernel only makes
   sense on standardised variables, and redoing the encoding would introduce a
   shift that nothing would flag;
4. the **same metrics**, produced by `mode_choice_eval.evaluate_proba`.

**Tuning never reads the test** (K5). `γ`, `λ` and `m` are chosen by 5-fold cross-validation
grouped by household **within the train**, and the landmarks are drawn from the
fitting part **of each fold**: a landmark drawn once and for all on the whole train
could fall into the validation fold, which would then enter the model's structure.

**The bench rules out drift before ranking** (K6). It is the probabilities, not
accuracy, that produce the modal shares: a configuration that wins on the
log-likelihood by shifting the shares is a bad model for this repository. Any
configuration whose out-of-sample share L1 exceeds `reference + 0.005` is therefore
removed from the ranking and counted. The reference is the **out-of-sample L1 of the logit on the
same folds**: measured rather than chosen, external to the family, and consistent with what
one expects from a KLR — hold the aggregate like the logit (0.0006 measured on 2026-09-11, against
0.0117 for the booster).

Usage:
    python -m scripts.progedo_logit.fit_mode_choice_klr [--out-dir DIR]
    python -m scripts.progedo_logit.fit_mode_choice_klr --gamma 0.04 --C 1 --m 1000

Does **not** require the raw PROGEDO data (restricted access lil-1750): the parquet and the spec
are in the repository. The artefact, for its part, carries `m` real survey rows as landmarks
and stays gitignored, like the other two.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from scripts.progedo_logit.fit_mode_choice_logit import (
    build_contract,
    fit_logit,
    normalized_weights,
)
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
from scripts.progedo_logit.mode_choice_klr import (
    EIGENVALUE_FLOOR,
    KLR_FORMAT,
    KLR_FORMAT_VERSION,
    KLRPredictor,
    nystrom_basis,
    rbf_kernel,
)
from scripts.progedo_logit.mode_choice_logit import design_matrix_from_contract

# --- Estimation settings ----------------------------------------------------
# Same estimator as the logit, on the Nyström map instead of the design matrix:
# KLR *is* a multinomial logit, in a variable space that the kernel builds.
SOLVER = "lbfgs"
MAX_ITER = 2000
TOLERANCE = 1e-6

#: Single seed of the script. The landmark draw depends on it, hence the artefact too: two
#: runs on the same day produce the same model, and the artefact's SHA remains a
#: version identifier (R12 of the experiment platform).
SEED = 20260911

#: Grid of `γ`, in multiples of the median heuristic (see `median_sq_distance`). Anchoring
#: on a measured quantity rather than on absolute values makes the grid readable: the
#: multiplier 1 is "the kernel width the geometry of the set suggests".
GAMMA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Ridge grid, expressed as `C = 1/λ` as in scikit-learn and as for the logit.
C_GRID = (0.1, 1.0, 10.0, 100.0)

#: Grid of the number of landmarks. The ticket bounds it at 500 to 2,000: below,
#: the approximation no longer describes the space, above, the memory of `n × m` decides.
M_GRID = (500, 1000, 2000)

#: `m` of stage A. The (γ, C) grid is swept at the cheapest `m`, then only `m` varies
#: on the chosen pair: a full sweep of the three axes would cost 300 fits for
#: information that the two stages give in 110.
STAGE_A_M = 500

CV_FOLDS = 5

#: Calibration tolerance, taken as is from the booster's bench: a configuration
#: that degrades the modal-share L1 beyond this threshold is ruled out whatever its
#: gain on the primary criterion.
L1_TOLERANCE = 0.005

#: Below this number of eligible configurations, the bench **says so**: a ceiling that
#: leaves almost nothing no longer selects, it endures. Vacuity ≠ perfection.
MIN_ELIGIBLE = 3

#: Memory cap of the Nyström map, in gibibytes. `n × m × 8` bytes, and
#: scikit-learn keeps a copy of it: the refusal comes **before** allocation, with the value
#: of `m` to reduce, rather than a bare `MemoryError` in the middle of a 20-minute bench.
MEMORY_LIMIT_GB = 3.0

#: Number of pairs drawn for the median heuristic. 1,000 × 1,000 is enough to stabilise a
#: median and costs 8 MB.
MEDIAN_SAMPLE = 1000


# ---------------------------------------------------------------------------
# Kernel geometry
# ---------------------------------------------------------------------------

def median_sq_distance(Z: np.ndarray, rng: np.random.Generator,
                       sample: int = MEDIAN_SAMPLE) -> float:
    """Median of `‖z − z′‖²` over a subsample of pairs — the median heuristic.

    It is the classic anchoring of an RBF kernel's width: `γ = 1/median` places
    the exponential at `e^{-1}` for two "typically distant" points. Without anchoring, a
    grid of `γ` in absolute values would make no sense from one set to another — it depends
    on the number of design columns and on their standardisation.
    """
    n = len(Z)
    size = min(sample, n // 2)
    picks = rng.choice(n, size=2 * size, replace=False)
    left, right = Z[picks[:size]], Z[picks[size:]]
    a2 = np.einsum("ij,ij->i", left, left)[:, None]
    b2 = np.einsum("ij,ij->i", right, right)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (left @ right.T), 0.0)
    return float(np.median(d2))


def draw_landmarks(households: np.ndarray, m: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices of `m` landmarks, **one per household**, and the households drawn (K4).

    Drawing `m` trips at random would concentrate the landmarks on large households — up to 38
    trips for a single household of the train — and the variable space would be described where
    a few households live. We therefore draw `m` distinct households, then one trip in
    each.
    """
    unique = np.unique(households)
    if m > len(unique):
        raise ValueError(
            f"{m} landmarks requested for {len(unique)} available households: "
            "one landmark per household is rule K4, reduce m.")
    chosen = rng.permutation(unique)[:m]
    indices = np.array([rng.choice(np.flatnonzero(households == hh)) for hh in chosen])
    return indices, chosen


def memory_guard(n_rows: int, m: int, limit_gb: float = MEMORY_LIMIT_GB) -> float:
    """Refuses a Nyström map that is too large **before** allocating it. Returns the size in GB.

    Twice `n × m × 8` bytes: the map, plus the copy scikit-learn makes when
    validating its inputs. At `m = 2,000` over 39,203 rows, that makes 1.25 GiB.
    """
    gib = 2.0 * n_rows * m * 8 / 2**30
    if gib > limit_gb:
        raise SystemExit(
            f"[ALARME] Carte de Nyström de {gib:.2f} Gio pour n={n_rows}, m={m} — au-delà du "
            f"plafond de {limit_gb:.2f} Gio. Réduisez m (--m-grid) ou relevez "
            "--memory-limit-gb en connaissance de cause.")
    return gib


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def nystrom_map(landmarks: np.ndarray, gamma: float) -> tuple[np.ndarray, dict]:
    """Projection basis `W` and its diagnostic, from the landmarks and `γ` alone.

    Depends only on the landmarks: this is what allows building it once per fold and
    fitting all the grid's ridges on it.
    """
    return nystrom_basis(rbf_kernel(landmarks, landmarks, gamma), EIGENVALUE_FLOOR)


def features(Z: np.ndarray, landmarks: np.ndarray, gamma: float,
             basis: np.ndarray) -> np.ndarray:
    """Nyström map `Φ = K(Z, L)·W`, the space in which the logit is fitted."""
    return rbf_kernel(Z, landmarks, gamma) @ basis


def fit_klr(Phi: np.ndarray, y: np.ndarray, w: np.ndarray, C: float) -> LogisticRegression:
    """Fits the weighted multinomial logit on the Nyström map.

    The weights are normalised to a mean of 1 by the **same** function as the logit: the
    COEP weighting averages 85.5, and passed as is it multiplies the likelihood
    by ~85 without touching the penalty — `C` would no longer mean anything.
    """
    model = LogisticRegression(
        C=C, solver=SOLVER, max_iter=MAX_ITER, tol=TOLERANCE, fit_intercept=True)
    model.fit(Phi, y, sample_weight=normalized_weights(w))
    return model


def dual_coefficients(basis: np.ndarray, model: LogisticRegression) -> np.ndarray:
    """`B = W·βᵀ`, of shape `m × classes` — what the artefact carries (K7).

    Folding `W` into the coefficients makes the artefact 20 times smaller and prediction
    independent of the eigendecomposition. The equality `Φβᵀ = K·B` is exact; it is
    checked at 1e-9 before writing, and measured at 3e-14.
    """
    return basis @ model.coef_.T


# ---------------------------------------------------------------------------
# Bench: cross-validation grouped by household, within the train
# ---------------------------------------------------------------------------

def l1_mass(proba: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int) -> float:
    """L1 between predicted (in probability mass) and observed modal shares, weighted."""
    observed = np.array([w[y == k].sum() / w.sum() for k in range(n_classes)])
    mass = (proba * w[:, None]).sum(axis=0) / w.sum()
    return float(np.abs(mass - observed).sum())


def score_oof(proba: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int) -> dict:
    """The bench's two quantities: primary criterion and calibration safeguard."""
    return {
        "cv_log_loss_weighted": float(log_loss(
            y, proba, labels=list(range(n_classes)), sample_weight=w)),
        "l1_mass_oof": l1_mass(proba, y, w, n_classes),
    }


def logit_reference(Z: np.ndarray, y: np.ndarray, w: np.ndarray, folds: list,
                    n_classes: int, C: float) -> dict:
    """Reference of the calibration ceiling: the **logit on the same folds** (K6).

    The booster's bench compares itself with its own outgoing configuration; KLR has none,
    being a new family. The reference is therefore external and measured: the
    out-of-sample L1 of the logit, on the same design matrix, the same folds, the same
    weights. It says what is asked of KLR — hold the aggregate like the logit.
    """
    oof = np.zeros((len(y), n_classes))
    for fit_idx, valid_idx in folds:
        model = fit_logit(Z[fit_idx], y[fit_idx], w[fit_idx], C)
        oof[valid_idx] = model.predict_proba(Z[valid_idx])
    scores = score_oof(oof, y, w, n_classes)
    return {"family": "logit multinomial", "C": C,
            "measured_on": "mêmes plis, même matrice de dessin, mêmes poids", **scores}


def cross_val_klr(Z: np.ndarray, y: np.ndarray, w: np.ndarray, households: np.ndarray,
                  folds: list, n_classes: int, gamma: float, m: int,
                  c_values: tuple[float, ...],
                  seed: int) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Out-of-sample predictions for one `(γ, m)` and **each** `C`, in a single pass.

    A fold's Nyström map does not depend on `C`: building it once and fitting
    the four ridges on it saves four fifths of the kernel work. The landmarks are drawn
    **within the fitting part of the fold** (K5), with a seed derived from the fold — hence
    reproducible, and never the same as the final model's.

    Also returns, per `C`, the **number of folds that reached the iteration cap**. A
    non-converged fold does not return the requested ridge's solution: its score is not that
    of the configuration, and letting such a row compete would amount to ranking a model
    that was not estimated. The caller rules it out, it does not correct it.
    """
    oof = {C: np.zeros((len(y), n_classes)) for C in c_values}
    capped = {C: 0 for C in c_values}
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        rng = np.random.default_rng(seed + 1000 * fold)
        local = draw_landmarks(households[fit_idx], m, rng)[0]
        landmarks = Z[fit_idx][local]
        basis, _ = nystrom_map(landmarks, gamma)
        Phi_fit = features(Z[fit_idx], landmarks, gamma, basis)
        Phi_valid = features(Z[valid_idx], landmarks, gamma, basis)
        for C in c_values:
            model = fit_klr(Phi_fit, y[fit_idx], w[fit_idx], C)
            if int(np.max(np.atleast_1d(model.n_iter_))) >= MAX_ITER:
                capped[C] += 1
            oof[C][valid_idx] = model.predict_proba(Phi_valid)
    return oof, capped


def run_bench(Z: np.ndarray, y: np.ndarray, w: np.ndarray, households: np.ndarray,
              folds: list, n_classes: int, gamma_base: float, gamma_multipliers: tuple,
              c_grid: tuple, stage_a_m: int, seed: int, memory_limit: float) -> dict:
    """Stage A of the bench: the (γ, C) grid at the cheapest `m`.

    Returns the measured configurations, each with its two metrics, its count of
    non-converged folds and its duration. The safeguard (`apply_guard`) and the ranking
    (`best_of`) are applied by the caller: the bench **measures**, it does not decide.
    """
    results: list[dict] = []

    def measure(gamma_mult: float, m: int, c_values: tuple) -> None:
        gamma = gamma_mult * gamma_base
        memory_guard(len(y), m, memory_limit)
        start = time.perf_counter()
        oof, capped = cross_val_klr(Z, y, w, households, folds, n_classes, gamma, m,
                                    c_values, seed)
        elapsed = time.perf_counter() - start
        for C in c_values:
            scores = score_oof(oof[C], y, w, n_classes)
            results.append({"gamma_multiplier": gamma_mult, "gamma": gamma, "C": C, "m": m,
                            **scores, "n_folds_not_converged": capped[C],
                            "seconds": round(elapsed / len(c_values), 1)})
            print(f"  γ×{gamma_mult:<5g} C={C:<7g} m={m:<5d} "
                  f"log-loss CV = {scores['cv_log_loss_weighted']:.5f} "
                  f"L1 = {scores['l1_mass_oof']:.4f}"
                  + (f" [{capped[C]} fold(s) NOT CONVERGED]" if capped[C] else "")
                  + f" ({elapsed / len(c_values):.0f} s)", flush=True)

    print(f"\nStage A — (γ, C) grid at m = {stage_a_m}, "
          f"{len(gamma_multipliers) * len(c_grid)} configurations:")
    stage_a_start = time.perf_counter()
    for multiplier in gamma_multipliers:
        measure(multiplier, stage_a_m, c_grid)
    print(f"Étage A terminé en {time.perf_counter() - stage_a_start:.0f} s")

    return {"results": results}


def apply_guard(results: list[dict], reference_l1: float,
                tolerance: float = L1_TOLERANCE) -> tuple[list[dict], list[dict], float]:
    """Separates the eligible configurations from those ruled out, **with their reason**.

    Two grounds for ruling out, and neither is an adjustment of the score:

    - **modal-share drift** (K6), beyond `reference + tolerance`;
    - **non-converged fold**: the requested ridge was not estimated, so the row does not measure
      the configuration it names. Catching it up by raising the iteration
      cap would be another bench, not a correction of this one.
    """
    ceiling = reference_l1 + tolerance
    eligible, rejected = [], []
    for result in results:
        if result.get("n_folds_not_converged"):
            reason = (f"{result['n_folds_not_converged']} pli(s) au plafond de "
                      f"{MAX_ITER} itérations")
        elif result["l1_mass_oof"] > ceiling:
            reason = (f"dérive des parts modales : L1 {result['l1_mass_oof']:.4f} > "
                      f"{ceiling:.4f}")
        else:
            eligible.append(result)
            continue
        rejected.append({**result, "ecartee_pour": reason})
    return eligible, rejected, ceiling


def grid_edges(best: dict, gamma_multipliers: tuple, c_grid: tuple,
               m_grid: tuple) -> list[str]:
    """Axes on which the chosen value is at the **edge** of the explored grid.

    An optimum at the edge does not say "this is the optimum", it says "the optimum may be
    outside". The booster's bench publishes the same information ("no grid edge
    touched"): without it, a grid that is too narrow reads as a result.
    """
    edges = []
    if best["gamma_multiplier"] in (min(gamma_multipliers), max(gamma_multipliers)):
        edges.append(f"γ×{best['gamma_multiplier']:g} (grille "
                     f"{min(gamma_multipliers):g}..{max(gamma_multipliers):g})")
    if best["C"] in (min(c_grid), max(c_grid)):
        edges.append(f"C={best['C']:g} (grille {min(c_grid):g}..{max(c_grid):g})")
    if len(m_grid) > 1 and best["m"] in (min(m_grid), max(m_grid)):
        edges.append(f"m={best['m']} (grille {min(m_grid)}..{max(m_grid)})")
    return edges


def best_of(results: list[dict]) -> dict:
    """Best configuration on the primary criterion: the CV negative log-likelihood.

    Accuracy does not act as arbiter — a model that gains one point of accuracy by
    crushing the bike (4 % of trips) degrades the modal shares, which are what the
    pipeline consumes.
    """
    return min(results, key=lambda r: r["cv_log_loss_weighted"])


# ---------------------------------------------------------------------------
# Smoothness diagnostic
# ---------------------------------------------------------------------------

def profile_by_decile(proba: np.ndarray, values: np.ndarray, w: np.ndarray,
                      classes: list[str], bins: int = 10) -> list[dict]:
    """Predicted modal shares by decile of a continuous variable — an elasticity reading.

    The ticket's target is not only to land between the two families on accuracy,
    it is to **stay smooth**. A profile of shares along the deciles of `od_km` reads at a
    glance: a smooth family draws monotone curves there, a booster draws
    steps. Published as a diagnostic, like the booster's gain importances and the logit's
    largest coefficients.
    """
    finite = np.isfinite(values)
    edges = np.quantile(values[finite], np.linspace(0, 1, bins + 1))
    out: list[dict] = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = finite & (values >= low) & ((values <= high) if i == bins - 1
                                          else (values < high))
        if not mask.any():
            continue
        weights = w[mask]
        mass = (proba[mask] * weights[:, None]).sum(axis=0) / weights.sum()
        out.append({"decile": i + 1, "od_km_min": float(low), "od_km_max": float(high),
                    "n": int(mask.sum()),
                    "shares": {c: float(v) for c, v in zip(classes, mass)}})
    return out


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def build_artefact(spec: dict, spec_path: Path, dataset_path: Path, contract: dict,
                   kernel: dict, nystrom: dict, landmarks: np.ndarray,
                   dual_coef: np.ndarray, intercept: np.ndarray, training: dict,
                   metrics: dict) -> dict:
    """Self-contained artefact: enough to predict without scikit-learn or parquet (K7)."""
    return {
        "format": KLR_FORMAT,
        "format_version": KLR_FORMAT_VERSION,
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
        "encoding": {
            "shared_with": "fit_mode_choice_policy.encode_features",
            "design_shared_with": "mode_choice_logit.design_matrix_from_contract",
            "missing": "NaN en entrée, traité par klr.design.missing_rule",
        },
        "geo_reference": spec.get("geo_reference"),
        "domain": spec.get("domain"),
        "notes": spec.get("notes"),
        "training": training,
        "metrics": metrics,
        "klr": {
            # Matrix contract: the logit's, identical. The key is `design` and
            # not `logit` — it is the same contract, it is not the same family.
            "design": contract,
            "kernel": kernel,
            "nystrom": nystrom,
            "landmarks": [[float(v) for v in row] for row in landmarks],
            "dual_coef": [[float(v) for v in row] for row in dual_coef],
            "intercept": [float(v) for v in intercept],
            "link": ("softmax(K(Z,L)·dual_coef + intercept), K = exp(−γ‖z−l‖²), "
                     "classes dans l'ordre du spec"),
            "landmark_privacy": ("les appuis sont m lignes réelles d'enquête "
                                 "(centrées-réduites) : artefact gitignoré, source PROGEDO "
                                 "d'accès restreint lil-1750"),
        },
    }


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
    parser.add_argument("--gamma", type=float, default=None,
                        help="Imposed γ (default: chosen by cross-validation)")
    parser.add_argument("--C", type=float, default=None,
                        help="Imposed C = 1/λ (default: cross-validation)")
    parser.add_argument("--m", type=int, default=None,
                        help="Imposed number of landmarks (default: cross-validation)")
    parser.add_argument("--m-grid", type=int, nargs="+", default=list(M_GRID),
                        help=f"Grid of m for stage B (default: {list(M_GRID)})")
    parser.add_argument("--gamma-multipliers", type=float, nargs="+",
                        default=list(GAMMA_MULTIPLIERS),
                        help=("Grid of γ in multiples of the median heuristic (default: "
                              f"{list(GAMMA_MULTIPLIERS)}) — to widen when the bench "
                              "reports a grid edge touched"))
    parser.add_argument("--folds", type=int, default=CV_FOLDS,
                        help=f"Folds of the cross-validation (default: {CV_FOLDS})")
    parser.add_argument("--memory-limit-gb", type=float, default=MEMORY_LIMIT_GB,
                        help=f"Cap of the Nyström map (default: {MEMORY_LIMIT_GB} GiB)")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    root = find_project_root()
    here = root / "scripts" / "progedo_logit"
    dataset_path = args.dataset or (here / "progedo_mode_choice_v2.parquet")
    spec_path = args.spec or (here / "feature_spec.json")
    out_dir = args.out_dir or here
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(dataset_path)
    check_spec(spec, df)          # same leakage guard as the other two families

    names = feature_names(spec)
    classes = spec["target"]["classes"]
    print(f"Set: {len(df)} rows | spec v{spec['spec_version']} | "
          f"{len(names)} variables | {len(classes)} classes")

    encoded = encode_features(df, spec)
    y = df[spec["target"]["name"]].map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = df[spec["sample_weight"]].to_numpy(dtype=float)
    households = df["hh_id"].to_numpy()

    is_train = (df["split"] == "train").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    print(f"Split read from the parquet: train={is_train.sum()} test={is_test.sum()} "
          f"| households={df['hh_id'].nunique()}")

    # Design contract: built on the train alone, by the logit's code.
    contract = build_contract(spec, encoded[is_train], w[is_train])
    Z = design_matrix_from_contract(encoded, spec["features"], contract)
    print(f"Design matrix (shared with the logit): {Z.shape[1]} columns "
          f"({len(contract['missing_indicators'])} missing indicators)")

    Ztr, ytr, wtr, hhtr = Z[is_train], y[is_train], w[is_train], households[is_train]
    rng = np.random.default_rng(SEED)
    gamma_base = 1.0 / median_sq_distance(Ztr, rng)
    print(f"Median heuristic: median ‖z−z′‖² = {1.0 / gamma_base:.3f} "
          f"→ γ_med = {gamma_base:.5f}")

    imposed = all(v is not None for v in (args.gamma, args.C, args.m))
    if imposed:
        tuning = {"gamma": args.gamma, "C": args.C, "m": args.m, "imposed": True,
                  "scored_on": "aucun réglage — valeurs imposées en ligne de commande",
                  "gamma_median_heuristic": gamma_base}
        print(f"\nImposed tuning: γ = {args.gamma}, C = {args.C}, m = {args.m}")
    else:
        if any(v is not None for v in (args.gamma, args.C, args.m)):
            raise SystemExit(
                "--gamma, --C and --m are imposed together or not at all: a partially "
                "imposed tuning would make the bench trace unreadable.")
        folds = list(GroupKFold(n_splits=args.folds).split(Ztr, ytr, groups=hhtr))
        print(f"\n{args.folds}-fold cross-validation, grouped by household, WITHIN the train "
              f"(folds of {[len(v) for _, v in folds]} rows)")

        print("\nRéférence du plafond de calibration — le logit sur les mêmes plis :")
        t0 = time.perf_counter()
        reference = logit_reference(Ztr, ytr, wtr, folds, len(classes), C=1.0)
        print(f"  log-loss CV = {reference['cv_log_loss_weighted']:.5f} | "
              f"L1 out-of-sample = {reference['l1_mass_oof']:.4f} "
              f"({time.perf_counter() - t0:.0f} s)")
        ceiling_value = reference["l1_mass_oof"] + L1_TOLERANCE
        print(f"  ceiling = {reference['l1_mass_oof']:.4f} + {L1_TOLERANCE} "
              f"= {ceiling_value:.4f}: beyond it, a configuration is removed from the ranking")

        bench = run_bench(Ztr, ytr, wtr, hhtr, folds, len(classes), gamma_base,
                          tuple(args.gamma_multipliers), C_GRID, STAGE_A_M, SEED,
                          args.memory_limit_gb)
        results = bench["results"]

        eligible, rejected, ceiling = apply_guard(results, reference["l1_mass_oof"])
        print(f"\n{len(eligible)}/{len(results)} configurations pass the calibration "
              f"safeguard (L1 ≤ {ceiling:.4f}).")
        for r in rejected:
            print(f"  ruled out: γ×{r['gamma_multiplier']:g} C={r['C']:g} m={r['m']} — "
                  f"{r['ecartee_pour']}")
        if not eligible:
            raise SystemExit(
                "[ALARME] No configuration passes the calibration safeguard: the "
                "bench does not select, it would merely endure. Nothing is published. Widen the "
                "grid or revisit the ceiling reference (K6).")
        if len(eligible) < MIN_ELIGIBLE:
            print(f"⚠ [ALARME] Seulement {len(eligible)} configuration(s) éligible(s) sur "
                  f"{len(results)} : le plafond de calibration décide plus que le critère "
                  "primaire. Le triplet retenu est publié avec cette réserve.")

        stage_a_best = best_of(eligible)
        print(f"\nÉtage A retenu : γ×{stage_a_best['gamma_multiplier']:g} "
              f"(γ = {stage_a_best['gamma']:.5f}), C = {stage_a_best['C']:g}")

        extra_m = [m for m in args.m_grid if m != STAGE_A_M]
        if extra_m:
            print(f"\nStage B — number of landmarks, on the chosen pair: {extra_m}")
            stage_b_start = time.perf_counter()
            for m in extra_m:
                memory_guard(len(ytr), m, args.memory_limit_gb)
                t0 = time.perf_counter()
                oof, capped = cross_val_klr(Ztr, ytr, wtr, hhtr, folds, len(classes),
                                            stage_a_best["gamma"], m,
                                            (stage_a_best["C"],), SEED)
                scores = score_oof(oof[stage_a_best["C"]], ytr, wtr, len(classes))
                elapsed = time.perf_counter() - t0
                results.append({"gamma_multiplier": stage_a_best["gamma_multiplier"],
                                "gamma": stage_a_best["gamma"], "C": stage_a_best["C"],
                                "m": m, **scores,
                                "n_folds_not_converged": capped[stage_a_best["C"]],
                                "seconds": round(elapsed, 1)})
                print(f"  m={m:<5d} log-loss CV = {scores['cv_log_loss_weighted']:.5f} "
                      f"L1 = {scores['l1_mass_oof']:.4f}"
                      + (f" [{capped[stage_a_best['C']]} fold(s) NOT CONVERGED]"
                         if capped[stage_a_best["C"]] else "")
                      + f" ({elapsed:.0f} s)", flush=True)
            print(f"Étage B terminé en {time.perf_counter() - stage_b_start:.0f} s")

        eligible, rejected, ceiling = apply_guard(results, reference["l1_mass_oof"])
        best = best_of(eligible)
        edges = grid_edges(best, tuple(args.gamma_multipliers), C_GRID,
                           tuple(args.m_grid))
        tuning = {
            "grid": {"gamma_multipliers": list(args.gamma_multipliers),
                     "gamma_median_heuristic": gamma_base,
                     "C": list(C_GRID), "m": list(args.m_grid), "stage_a_m": STAGE_A_M},
            "folds": args.folds,
            "grouped_by": "hh_id",
            "scored_on": "train uniquement — le split test n'est jamais lu pour régler",
            "criterion": "cv_log_loss_weighted, sous garde-fou de calibration",
            "calibration_guard": {
                "tolerance": L1_TOLERANCE,
                "reference": reference,
                "ceiling": ceiling,
                "n_eligible": len(eligible),
                "n_rejected": len(rejected),
                "rejected": [{k: r[k] for k in ("gamma", "C", "m", "l1_mass_oof",
                                                "n_folds_not_converged", "ecartee_pour")}
                             for r in rejected],
                "note": ("ce sont les probabilités, pas l'exactitude, qui produisent les "
                         "parts modales : une configuration qui dérive est écartée avant "
                         "tout classement, comme une configuration dont un pli n'a pas "
                         "convergé"),
            },
            "landmark_draw": ("un trajet tiré par ménage, m ménages distincts, appuis tirés "
                              "dans la partie d'ajustement de CHAQUE pli"),
            "results": results,
            "gamma": best["gamma"],
            "gamma_multiplier": best["gamma_multiplier"],
            "C": best["C"],
            "m": best["m"],
            "cv_log_loss_weighted": best["cv_log_loss_weighted"],
            "l1_mass_oof": best["l1_mass_oof"],
            "grid_edges_touched": edges,
        }
        if edges:
            print("⚠ Grid edge touched: " + " ; ".join(edges)
                  + " — the optimum may be outside the explored grid. "
                    "Widen: --gamma-multipliers / --m-grid.")
        print(f"\nChosen: γ = {best['gamma']:.5f} (×{best['gamma_multiplier']:g}), "
              f"C = {best['C']:g}, m = {best['m']} — log-loss CV "
              f"{best['cv_log_loss_weighted']:.5f}, L1 {best['l1_mass_oof']:.4f}")

    # ── Final model: landmarks drawn on the whole train ──────────────────────
    gamma, C, m = tuning["gamma"], tuning["C"], tuning["m"]
    size_gib = memory_guard(int(is_train.sum()), m, args.memory_limit_gb)
    print(f"\nFinal fit (Nyström map {int(is_train.sum())} × {m}, "
          f"{size_gib:.2f} GiB with the scikit-learn copy):")
    t0 = time.perf_counter()
    final_rng = np.random.default_rng(SEED)
    local_idx, landmark_households = draw_landmarks(hhtr, m, final_rng)
    landmarks = Ztr[local_idx]
    basis, nystrom = nystrom_map(landmarks, gamma)
    if nystrom["dropped_eigencomponents"]:
        print(f"⚠ {nystrom['dropped_eigencomponents']} eigencomponent(s) below the "
              f"floor dropped: effective rank {nystrom['rank']}/{m}")
    Phi_train = features(Ztr, landmarks, gamma, basis)
    model = fit_klr(Phi_train, ytr, wtr, C)
    iterations = int(np.max(np.atleast_1d(model.n_iter_)))
    if iterations >= MAX_ITER:
        raise SystemExit(
            f"[ALARME] The fit reached the cap of {MAX_ITER} iterations: the "
            "coefficients have not converged and must not be published.")
    print(f"Converged in {iterations} iterations (cap {MAX_ITER}), "
          f"{time.perf_counter() - t0:.0f} s")

    dual = dual_coefficients(basis, model)
    nystrom = {**nystrom, "landmark_draw": tuning.get("landmark_draw", "un trajet par ménage"),
               "landmark_households": [str(h) for h in landmark_households],
               "seed": SEED}
    kernel = {"type": "rbf", "gamma": float(gamma),
              "gamma_median_heuristic": float(gamma_base),
              "median_sq_distance": float(1.0 / gamma_base),
              "heuristic": f"médiane des ‖z−z′‖² sur {MEDIAN_SAMPLE}² paires du train"}

    training = {
        "estimator": ("kernel logistic regression (RBF + Nyström, lbfgs multinomial, "
                      "pondérée COEP)"),
        "gamma": float(gamma), "C": float(C), "m": int(m),
        "rank": nystrom["rank"],
        "tuning": tuning,
        "n_iter": iterations,
        "n_fit": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "n_design_columns": int(Z.shape[1]),
        "sample_weight": spec["sample_weight"],
        "weight_normalization": ("poids ramenés à une moyenne de 1 pour l'estimation ; "
                                 "métriques calculées avec le COEP brut"),
        "split": spec.get("split"),
        "seed": SEED,
    }

    # ── Metrics of the sealed test, by the shared module ────────────────────
    proba_test = model.predict_proba(features(Z[is_test], landmarks, gamma, basis))
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

    artefact = build_artefact(spec, spec_path, dataset_path, contract, kernel, nystrom,
                              landmarks, dual, model.intercept_, training, metrics)

    # ── Self-containment, BEFORE writing ────────────────────────────────────
    # The pure-numpy evaluator must reproduce scikit-learn: it is the check that makes
    # the folding `B = W·βᵀ` trustworthy. An artefact that does not read back as it
    # was estimated is a wrong artefact, and nothing would say so on reading.
    replayed = KLRPredictor(artefact).predict(encoded[is_test])
    gap = float(np.abs(replayed - proba_test).max())
    if gap > 1e-9:
        raise SystemExit(
            f"[ALARME] The artefact does not reproduce the estimation: max gap {gap:.2e} on "
            "the test probabilities. Artefact not written.")
    print(f"Self-containment checked: max gap artefact / estimator = {gap:.1e}")

    artefact["metrics"]["mode_profile_by_od_decile"] = profile_by_decile(
        proba_test, df.loc[is_test, "od_km"].to_numpy(dtype="float64"),
        w[is_test], classes)

    artefact_path = out_dir / "klr_model.json"
    artefact_path.write_text(json.dumps(artefact, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")

    report = {
        "generated_at": artefact["generated_at"],
        "spec_version": spec["spec_version"],
        "dataset": dataset_path.name,
        "split": spec.get("split"),
        "training": training,
        "kernel": kernel,
        "nystrom": {k: v for k, v in nystrom.items() if k != "landmark_households"},
        "test": artefact["metrics"],
    }
    metrics_path = out_dir / "klr_model_metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    size_kb = artefact_path.stat().st_size / 1e3
    print(f"\nWritten:\n - {artefact_path} ({size_kb:.0f} kB, format "
          f"{KLR_FORMAT} v{KLR_FORMAT_VERSION})\n - {metrics_path}")
    print(f"Finished in {time.perf_counter() - started:.0f} s — "
          f"γ = {gamma:.5f}, C = {C:g}, m = {m}, rang {nystrom['rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
