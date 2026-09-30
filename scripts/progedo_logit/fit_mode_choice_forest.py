"""fit_mode_choice_forest.py — Random forest control: where does the booster's edge come from?

The LightGBM booster beats the multinomial logit by 1.9 accuracy points and 0.0017 of L1
on modal shares. **Does this edge come from the trees, or from boosting?** The sentence
the article is about to write is not refutable as long as both causes produce it
equally:

- the **trees** — non-linearities and interactions that a form linear in log-odds cannot
  represent; the lever would then be the variable contract;
- **boosting** — aggregation by gradient descent, which puts capacity back where the
  likelihood falls short; a third tree family would then bring nothing.

A random forest separates the two: it is made of trees like the booster, but aggregated by
**bagging**. This script fits it under strict parity and publishes **where it falls between
the two**.

**It is a control, not a candidate.** Nobody proposes replacing the oracle with an RF.
Consequence settled at the start (ticket 044): *no model is serialised*, only a file of
measurements. A self-contained RF would be 400 to 1,200 trees of several thousand
nodes — on the order of 80 to 800 MB of unreadable JSON — whereas the booster already
weighs 18.9 MB, and without a single consumer to reread it. Replaying the control requires
rerunning `make forest`: it is deterministic (`random_state` fixed), offline, and it costs
minutes.

**The four conditions of comparability**, true by construction and not promised:

1. the **same set** (`progedo_mode_choice_v2.parquet`) and the **same `split` column**,
   sealed by household, read and never re-split;
2. the **same `sample_weight`** (COEP survey weights) at fitting and in all
   metrics;
3. the **same `encode_features`**, imported from the booster's script;
4. the **same metrics**, produced by `mode_choice_eval.evaluate_proba`, a module shared
   by the three models.

**The encoding trap, and why the main path is not the literal path.** scikit-learn's
trees have **no support for categoricals**: fed the spec's integer codes, they would read
`purpose`, `socioprofessional_class` and `main_occupation` as **ordinals** on an arbitrary
order, whereas LightGBM partitions sets of categories. A lagging RF would then no longer
say « it is boosting » — it might say « it is the encoding », and the control would lose
the only property that makes it useful. The main path therefore reuses the **logit's
declared design matrix** (`mode_choice_logit.design_matrix`: one indicator per category,
a `__missing__` category, weighted train mean plus a missing indicator). The
standardisation it applies is neutral for a tree — a monotone transformation changes no
split.

The literal path is not abandoned though: `--encodage natif` fits the **same** RF on the
21 raw columns, `NaN` routed natively by scikit-learn, and the gap between the two **puts
a figure on the cost of encoding**. That is what makes the choice above refutable instead
of asserted.

**Tuning never reads the test**, and the verdict has its **threshold declared in
advance**: the share of the booster–logit gap closed by the RF, `≥ 0.70` → the trees,
`≤ 0.30` → boosting, in between → undecided and the sentence remains to be written.
Choosing the threshold after seeing the figure would amount to choosing the conclusion.

Usage:
    python -m scripts.progedo_logit.fit_mode_choice_forest [--encodage les-deux] [--rapide]

Does **not** require the raw PROGEDO data (restricted access lil-1750): the parquet and
the spec are versioned in the repository.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from scripts.progedo_logit.fit_mode_choice_logit import build_contract
from scripts.progedo_logit.fit_mode_choice_policy import (
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
)
from scripts.progedo_logit.mode_choice_eval import evaluate_proba, format_shares
from scripts.progedo_logit.mode_choice_logit import design_matrix

# --- Settings ---------------------------------------------------------------
SEED = 0
CV_FOLDS = 5

#: Tuning grid. Explicit rather than random: a one-hour control does not need the
#: booster's 96-configuration bench, and a published grid can be reread.
DEPTH_GRID = (None, 8, 14, 20)
LEAF_GRID = (1, 5, 20, 50)
FEATURES_GRID = ("sqrt", 0.3, 0.6)

#: Number of trees during cross-validation, then for the plateau check. An RF's
#: validation log-loss decreases **monotonically** with the number of trees — bagging
#: reduces variance, it does not overfit in `n_estimators`. Tuning depth at 200 trees
#: then checking at 400 and 1,200 costs six times less than a full grid, and the
#: plateau is **shown** (the three figures are published) instead of asserted.
N_ESTIMATORS_CV = 200
N_ESTIMATORS_PLATEAU = (400, 1200)

#: Verdict thresholds, fixed **before** launching (rule R8). They have no statistical
#: justification: it is a reading convention, and it is published as such.
SEUIL_ARBRES = 0.70
SEUIL_BOOSTING = 0.30

#: Below this, the booster–logit gap is too small for a share of it to be readable: the
#: ratio would diverge instead of measuring. The comparison is then « non mesuré », no figure.
ECART_MINIMAL = 1e-6

ENCODAGES = ("dessin", "natif")


# ---------------------------------------------------------------------------
# Input matrix
# ---------------------------------------------------------------------------

def build_matrix(encoded: pd.DataFrame, spec: dict, weights: np.ndarray,
                 is_train: np.ndarray, encodage: str) -> tuple[np.ndarray, dict]:
    """RF input matrix, and the description of the rule applied (rule R6).

    ``dessin``: the logit's declared design matrix — one indicator per category, a
    `__missing__` category, weighted train mean plus a missing indicator. The contract is
    built **on the train alone**: an indicator derived from the test would let test
    information in.

    ``natif``: the 21 columns of `encode_features`, integer codes for categoricals,
    `NaN` routed by scikit-learn. Sensitivity path: it measures the cost of reading a
    nominal as an ordinal.
    """
    if encodage == "dessin":
        contract = build_contract(spec, encoded[is_train], weights[is_train])
        partial = {"features": spec["features"], "logit": contract}
        matrix = design_matrix(encoded, partial)
        description = {
            "voie": "dessin",
            "colonnes": len(contract["design_columns"]),
            "noms_colonnes": contract["design_columns"],
            # Republished in full: this is what the control's artefact embeds so that
            # `RFPredictor` rebuilds the SAME matrix, without rereading this script.
            "contrat": contract,
            "indicatrices_de_manquant": contract["missing_indicators"],
            "regle_des_manquants": contract["missing_rule"],
            "partage_avec": "mode_choice_logit.design_matrix (second oracle)",
            "pourquoi": ("les arbres de scikit-learn n'ont aucun support des catégorielles ; "
                         "les codes entiers du spec y seraient lus comme des ordinales"),
        }
        return matrix, description

    if encodage == "natif":
        matrix = encoded.to_numpy(dtype="float64")
        n_nan = int(np.isnan(matrix).sum())
        description = {
            "voie": "natif",
            "colonnes": matrix.shape[1],
            "noms_colonnes": feature_names(spec),
            "indicatrices_de_manquant": [],
            "regle_des_manquants": {
                "toutes": ("NaN laissés tels quels, routés par les arbres de scikit-learn "
                           "(support ajouté en 1.4, splitter « best », matrice dense)"),
                "note": ("les catégorielles sont lues comme des ORDINALES sur l'ordre du "
                         "spec — c'est le biais que cette voie sert à mesurer"),
            },
            "n_valeurs_manquantes": n_nan,
            "partage_avec": "fit_mode_choice_policy.encode_features (booster), tel quel",
            "pourquoi": "voie de sensibilité — elle chiffre le coût de l'encodage",
        }
        return matrix, description

    raise SystemExit(f"Encodage inconnu : {encodage!r} (attendu : {ENCODAGES})")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def forest(n_estimators: int, max_depth, min_samples_leaf: int,
           max_features) -> RandomForestClassifier:
    """The control's RF. `class_weight=None` is not a default: it is rule R4.

    As for the booster (decision E7), no class reweighting. Rebalancing inflates the
    recall of minority classes while destroying calibration — yet it is the probabilities,
    not accuracy, that produce modal shares.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=None,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )


def proba_complete(model: RandomForestClassifier, matrix: np.ndarray,
                   n_classes: int) -> np.ndarray:
    """Probabilities over **all four** classes, even if a fold did not see one.

    `predict_proba` only returns the columns of classes seen at fitting. On a fold
    poor in bicycles (4 % of trips), a three-column matrix would silently mix
    with the others and shift the whole class order.
    """
    partial = model.predict_proba(matrix)
    full = np.zeros((len(matrix), n_classes), dtype="float64")
    for position, label in enumerate(model.classes_):
        full[:, int(label)] = partial[:, position]
    return full


def cv_log_loss(matrix: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
                n_classes: int, n_estimators: int, max_depth, min_samples_leaf: int,
                max_features, folds: int = CV_FOLDS) -> tuple[float, list[float]]:
    """Weighted negative log-likelihood in cross-validation grouped by household.

    Grouped because the trips of a household share its car ownership: randomly drawn
    folds would put the same household on both sides and flatter all configurations in
    the same wrong way (sampling trap, Hillel 2021).
    """
    splitter = GroupKFold(n_splits=folds)
    labels = list(range(n_classes))
    losses: list[float] = []
    for fit_idx, valid_idx in splitter.split(matrix, y, groups=groups):
        model = forest(n_estimators, max_depth, min_samples_leaf, max_features)
        model.fit(matrix[fit_idx], y[fit_idx], sample_weight=w[fit_idx])
        proba = proba_complete(model, matrix[valid_idx], n_classes)
        losses.append(float(log_loss(y[valid_idx], proba, labels=labels,
                                     sample_weight=w[valid_idx])))
    return float(np.mean(losses)), losses


def grids(rapide: bool) -> tuple[tuple, tuple, tuple]:
    """Tuning grid. `--rapide` shrinks it for a smoke pass, never for publishing."""
    if rapide:
        return (None, 14), (5, 50), ("sqrt",)
    return DEPTH_GRID, LEAF_GRID, FEATURES_GRID


def select_params(matrix: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
                  n_classes: int, rapide: bool = False) -> dict:
    """Chooses the configuration by cross-validation, **within the train** (rule R3).

    The criterion is the weighted negative log-likelihood, not accuracy: a model that
    gains an accuracy point by crushing bicycles degrades modal shares, and the shares
    are what the article compares.
    """
    depth_grid, leaf_grid, features_grid = grids(rapide)
    combinations = list(itertools.product(depth_grid, leaf_grid, features_grid))
    results: list[dict] = []
    started = time.monotonic()
    print(f"  {len(combinations)} configurations × {CV_FOLDS} folds, "
          f"{N_ESTIMATORS_CV} trees:")
    for index, (max_depth, min_samples_leaf, max_features) in enumerate(combinations, 1):
        step = time.monotonic()
        mean, per_fold = cv_log_loss(matrix, y, w, groups, n_classes, N_ESTIMATORS_CV,
                                     max_depth, min_samples_leaf, max_features)
        results.append({
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_estimators": N_ESTIMATORS_CV,
            "cv_log_loss_weighted": mean,
            "per_fold": per_fold,
            "seconds": round(time.monotonic() - step, 1),
        })
        print(f"   [{index:>2}/{len(combinations)}] depth={str(max_depth):<4} "
              f"leaf={min_samples_leaf:<3} feat={str(max_features):<5} "
              f"log-loss = {mean:.5f}  ({results[-1]['seconds']:.0f} s)")
    best = min(results, key=lambda r: r["cv_log_loss_weighted"])

    # A configuration retained on a grid edge signals that the optimum may lie
    # beyond: the published figure would then be the best of what was looked at, not the
    # best of the model. We say so in the output rather than leave it to be guessed.
    edges = []
    # `max_depth = None` is the maximum-capacity edge, but it has nothing « beyond »:
    # an unlimited depth cannot be under-explored. Only the finite bound counts.
    if best["max_depth"] is not None and best["max_depth"] == max(
            d for d in depth_grid if d is not None):
        edges.append("max_depth")
    if best["min_samples_leaf"] in (leaf_grid[0], leaf_grid[-1]):
        edges.append("min_samples_leaf")
    if len(features_grid) > 1 and best["max_features"] in (features_grid[0],
                                                           features_grid[-1]):
        edges.append("max_features")
    if edges:
        print(f"  [ALARME] Configuration retained on a grid edge: {edges}. "
              "The optimum may lie beyond what was explored.")

    duration = round(time.monotonic() - started, 1)
    print(f"  Tuning done in {duration:.0f} s — retained: depth={best['max_depth']} "
          f"leaf={best['min_samples_leaf']} feat={best['max_features']} "
          f"(log-loss CV {best['cv_log_loss_weighted']:.5f})")
    return {
        "grille": {"max_depth": list(depth_grid), "min_samples_leaf": list(leaf_grid),
                   "max_features": list(features_grid)},
        "folds": CV_FOLDS,
        "grouped_by": "hh_id",
        "scored_on": "train uniquement — le split test n'est jamais lu pour régler",
        "criterion": "log_loss_weighted",
        "n_estimators_cv": N_ESTIMATORS_CV,
        "results": results,
        "retenu": {k: best[k] for k in
                   ("max_depth", "min_samples_leaf", "max_features", "cv_log_loss_weighted")},
        "bords_de_grille": edges,
        "seconds": duration,
    }


def plateau(matrix: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
            n_classes: int, retenu: dict) -> dict:
    """Plateau check: the same configuration, at 200, 400 and 1,200 trees.

    Published because tuning done at 200 trees only holds for 1,200 if the log-loss has
    tightened. If it keeps going down markedly, the control is undersized and the
    reader must know it.
    """
    points = [{"n_estimators": N_ESTIMATORS_CV,
               "cv_log_loss_weighted": retenu["cv_log_loss_weighted"]}]
    for n_estimators in N_ESTIMATORS_PLATEAU:
        mean, _ = cv_log_loss(matrix, y, w, groups, n_classes, n_estimators,
                              retenu["max_depth"], retenu["min_samples_leaf"],
                              retenu["max_features"])
        points.append({"n_estimators": n_estimators, "cv_log_loss_weighted": mean})
        print(f"   {n_estimators:>5} trees: log-loss CV = {mean:.5f}")
    gains = [points[i - 1]["cv_log_loss_weighted"] - points[i]["cv_log_loss_weighted"]
             for i in range(1, len(points))]
    return {
        "points": points,
        "gains_successifs": gains,
        "n_estimators_retenu": points[-1]["n_estimators"],
        "lecture": ("le gain du dernier palier mesure ce qu'il resterait à prendre en "
                    "ajoutant des arbres"),
    }


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def part_de_l_ecart(rf: float, booster: float, logit: float) -> Optional[float]:
    """Share of the booster–logit gap closed by the RF: 0 = the logit, 1 = the booster.

    The ratio is **insensitive to the direction** of the metric: whether « better » is
    higher (accuracy) or lower (L1), `(rf − logit) / (booster − logit)` is 0 at the
    logit's level and 1 at the booster's. It can leave [0, 1] — an RF better than the
    booster gives more than 1, an RF worse than the logit gives less than 0 — and that is
    information, not an anomaly to clamp.
    """
    ecart = booster - logit
    if abs(ecart) < ECART_MINIMAL:
        return None
    return float((rf - logit) / ecart)


def verdict(parts: list[Optional[float]]) -> dict:
    """The verdict, from the measured shares and the thresholds declared in advance (rule R8)."""
    mesurees = [p for p in parts if p is not None]
    if not mesurees:
        return {"conclusion": "non mesuré",
                "pourquoi": "aucun écart booster–logit exploitable",
                "seuils": {"arbres": SEUIL_ARBRES, "boosting": SEUIL_BOOSTING}}
    if min(mesurees) >= SEUIL_ARBRES:
        conclusion = "les arbres"
        lecture = ("le RF rejoint le booster : l'avantage tient aux arbres — non-linéarités "
                   "et interactions — et le levier est le contrat de variables")
    elif max(mesurees) <= SEUIL_BOOSTING:
        conclusion = "le boosting"
        lecture = ("le RF reste près du logit : l'avantage tient à l'agrégation par descente "
                   "de gradient, et une troisième famille d'arbres n'apporterait rien")
    else:
        conclusion = "indécis"
        lecture = ("le RF se place entre les deux seuils : la phrase de l'article reste à "
                   "écrire, et aucune des deux causes ne peut être affirmée seule")
    return {
        "conclusion": conclusion,
        "lecture": lecture,
        "seuils": {"arbres": SEUIL_ARBRES, "boosting": SEUIL_BOOSTING,
                   "declares": "avant le lancement (règle R8) — jamais ajustés après coup"},
    }


def lire_metriques(path: Path) -> Optional[dict]:
    """`test` block of an oracle metrics file, or `None` if it was not produced."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("test")


def cel(metrics: dict) -> float:
    """Weighted CEL, under either of its two names.

    `mode_choice_eval` publishes the **same quantity in nats** under `log_loss_weighted` (the
    repository's historical name) and `cel_weighted` (that of the comparative literature
    GMPCA derives from). The booster's file dates from 2026-08-30, before the second name
    existed: reading it under its name of the time is not a fallback, it is the same number.
    A real fallback — recomputing a CEL by another path, or publishing a zero — is not done.
    """
    if "cel_weighted" in metrics:
        return float(metrics["cel_weighted"])
    return float(metrics["log_loss_weighted"])


def alias_manquant(path: Path, metrics: dict) -> Optional[str]:
    """Flags a reference file older than the CEL/GMPCA aliases — readable, not silent."""
    if "cel_weighted" in metrics:
        return None
    return (f"{path.name} : produit avant les alias CEL/GMPCA (ticket 042) ; "
            "`log_loss_weighted` lu à leur place — même grandeur, mêmes nats")


def comparer(rf_metrics: dict, booster_path: Path, logit_path: Path) -> dict:
    """Places the RF between the two oracles, by **recomputing** from their files.

    Nothing is copied from a table: the three figures come from the three files produced
    by the **same** evaluation module. If one of the two is missing, the comparison is
    « non mesuré » and not `0.0` — an absence of measurement that output a zero would read
    « no gap », exactly the opposite of what it says (rule R9).
    """
    booster = lire_metriques(booster_path)
    logit = lire_metriques(logit_path)
    manquants = [p.name for p, m in ((booster_path, booster), (logit_path, logit)) if m is None]
    if manquants:
        return {
            "statut": "non mesuré",
            "references_manquantes": manquants,
            "comment_les_produire": "make policy (booster) et make logit (second oracle)",
            "verdict": verdict([]),
        }

    axes = [
        ("accuracy_weighted", "exactitude pondérée", lambda m: m["accuracy_weighted"]),
        ("l1_probability_mass", "L1 masse de probabilité",
         lambda m: m["mode_shares"]["l1_probability_mass"]),
        ("cel_weighted", "CEL", cel),
        ("l1_argmax", "L1 mode élu", lambda m: m["mode_shares"]["l1_argmax"]),
        ("recall_bike", "rappel vélo", lambda m: m["per_class"]["bike"]["recall"]),
    ]
    lignes = {}
    for key, label, read in axes:
        valeurs = {"random_forest": read(rf_metrics), "lightgbm": read(booster),
                   "logit": read(logit)}
        lignes[key] = {
            "libelle": label,
            **valeurs,
            "part_de_l_ecart_comblee": part_de_l_ecart(
                valeurs["random_forest"], valeurs["lightgbm"], valeurs["logit"]),
        }

    # The verdict bears only on the two axes of the question asked: disaggregate
    # accuracy and the L1 of aggregate shares. The other three are published for reading.
    parts = [lignes["accuracy_weighted"]["part_de_l_ecart_comblee"],
             lignes["l1_probability_mass"]["part_de_l_ecart_comblee"]]
    notes = [n for n in (alias_manquant(booster_path, booster),
                         alias_manquant(logit_path, logit)) if n]
    return {
        "statut": "mesuré",
        "sur": {"split": "test scellé", "n_rows": rf_metrics["n_rows"],
                "pondere_par": "COEP"},
        "references": {"lightgbm": booster_path.name, "logit": logit_path.name},
        "compatibilite": notes,
        "axes": lignes,
        "axes_du_verdict": ["accuracy_weighted", "l1_probability_mass"],
        "formule": "(RF − logit) / (booster − logit) : 0 = le logit, 1 = le booster",
        "verdict": verdict(parts),
    }


# ---------------------------------------------------------------------------
# One full pass, for one encoding
# ---------------------------------------------------------------------------

def passe(encodage: str, encoded: pd.DataFrame, spec: dict, y: np.ndarray, w: np.ndarray,
          groups: np.ndarray, is_train: np.ndarray, is_test: np.ndarray,
          classes: list[str], rapide: bool) -> dict:
    """Tunes, fits and evaluates the RF for a given encoding."""
    print(f"\n=== Encoding « {encodage} » " + "=" * 46)
    matrix, description = build_matrix(encoded, spec, w, is_train, encodage)
    print(f"Matrix: {matrix.shape[0]} rows × {matrix.shape[1]} columns "
          f"({len(description['indicatrices_de_manquant'])} missing indicators)")

    print(f"\nCross-validation, {CV_FOLDS} folds, grouped by household, within the train:")
    tuning = select_params(matrix[is_train], y[is_train], w[is_train], groups[is_train],
                           len(classes), rapide)
    retenu = tuning["retenu"]

    print("\nPlateau check (same configuration, more trees):")
    courbe = plateau(matrix[is_train], y[is_train], w[is_train], groups[is_train],
                     len(classes), retenu)
    n_estimators = courbe["n_estimators_retenu"]

    print(f"\nFinal fit: {n_estimators} trees on {int(is_train.sum())} rows…")
    started = time.monotonic()
    model = forest(n_estimators, retenu["max_depth"], retenu["min_samples_leaf"],
                   retenu["max_features"])
    model.fit(matrix[is_train], y[is_train], sample_weight=w[is_train])
    seconds = round(time.monotonic() - started, 1)

    # R4 checked on the fitted estimator, not on intent: a class reweighting
    # introduced by mistake must stop the script, not come out in a published figure.
    if model.class_weight is not None:
        raise SystemExit("[ALARME] non-null `class_weight`: rebalancing destroys the "
                         "calibration of probabilities (rule R4). Nothing is written.")
    print(f"Fitted in {seconds:.0f} s — {model.n_estimators} trees, "
          f"max depth observed {max(e.get_depth() for e in model.estimators_)}")

    proba_test = proba_complete(model, matrix[is_test], len(classes))
    metrics = evaluate_proba(proba_test, y[is_test], w[is_test], classes)

    print(f"\nTest ({metrics['n_rows']} rows, COEP-weighted):"
          f"\n  CEL (log-loss) = {metrics['cel_weighted']:.4f}"
          f"\n  GMPCA          = {metrics['gmpca_weighted']:.4f}"
          f"\n  accuracy       = {metrics['accuracy_weighted']:.4f}"
          f"\n  bike recall    = {metrics['per_class']['bike']['recall']:.3f}")
    shares = metrics["mode_shares"]
    print("\nModal shares (test, weighted):")
    print(format_shares(classes, shares["observed"],
                        shares["predicted_probability_mass"], shares["predicted_argmax"]))
    print(f"  L1 probability mass = {shares['l1_probability_mass']:.4f}"
          f" | L1 elected mode = {shares['l1_argmax']:.4f}")

    importances = sorted(
        ({"colonne": nom, "importance": float(v)} for nom, v in zip(
            description["noms_colonnes"], model.feature_importances_)),
        key=lambda r: -r["importance"])
    return {
        "encodage": description,
        "training": {
            "estimator": "RandomForestClassifier (scikit-learn, pondéré COEP)",
            "n_estimators": int(model.n_estimators),
            "max_depth": retenu["max_depth"],
            "min_samples_leaf": retenu["min_samples_leaf"],
            "max_features": retenu["max_features"],
            "class_weight": None,
            "bootstrap": True,
            "random_state": SEED,
            "n_fit": int(is_train.sum()),
            "n_test": int(is_test.sum()),
            "n_colonnes": int(matrix.shape[1]),
            "sample_weight": spec["sample_weight"],
            "fit_seconds": seconds,
            "tuning": tuning,
            "plateau": courbe,
        },
        "test": metrics,
        "importances": importances[:15],
    }


# ---------------------------------------------------------------------------
# Replay artefact — figures and a contract, never a tree
# ---------------------------------------------------------------------------

def build_artefact(passe_principale: dict, spec: dict, spec_path: Path,
                   dataset_path: Path, trainset_path: Path, genere_le: str) -> dict:
    """The control's artefact: enough to **rebuild** the forest, not to carry it.

    No tree is in it. The 3,740,500 nodes of the retained forest would weigh ~150 MB as
    JSON or 165 MB as compressed `joblib`; `RFPredictor` rebuilds them in about ten
    seconds from the parquet, with a fixed seed, hence identically. The artefact freezes
    the two conditions of that identity — parquet hash, scikit-learn version — and the
    predictor refuses to run if either diverges.
    """
    from sklearn import __version__ as sklearn_version

    from scripts.progedo_logit.mode_choice_rf import RF_FORMAT, RF_FORMAT_VERSION, sha256_fichier

    training = passe_principale["training"]
    return {
        "format": RF_FORMAT,
        "format_version": RF_FORMAT_VERSION,
        "generated_at": genere_le,
        "source": spec.get("source"),
        "spec_version": spec["spec_version"],
        "spec_file": spec_path.name,
        "dataset_file": dataset_path.name,
        # Provenance only: the parquet is not versioned and, above all, the `controller`
        # container where experiments run cannot read it (neither pyarrow nor
        # fastparquet). Replay therefore goes through the matrix below, which numpy alone
        # can reread, and whose hash attests that the refitted forest is the published one.
        "dataset_sha256": sha256_fichier(dataset_path),
        "trainset_file": trainset_path.name,
        "trainset_sha256": sha256_fichier(trainset_path),
        "target": {"name": spec["target"]["name"], "classes": spec["target"]["classes"]},
        "features": [
            {"name": f["name"], "kind": f["kind"], "source": f["source"],
             **({"categories": f["categories"]} if f["kind"] == "categorical" else {})}
            for f in spec["features"]
        ],
        "encoding": {
            "shared_with": "fit_mode_choice_policy.encode_features",
            "missing": "NaN en entrée, traité par rf.contrat.missing_rule",
            "design": "dilatée par mode_choice_logit.design_matrix, comme le second oracle",
        },
        "geo_reference": spec.get("geo_reference"),
        "domain": spec.get("domain"),
        "notes": spec.get("notes"),
        "training": {k: v for k, v in training.items() if k not in ("tuning", "plateau")},
        "metrics": passe_principale["test"],
        "rf": {
            "estimator": {
                "library": "scikit-learn",
                "class": "RandomForestClassifier",
                "version": sklearn_version,
            },
            "hyperparameters": {
                "n_estimators": training["n_estimators"],
                "max_depth": training["max_depth"],
                "min_samples_leaf": training["min_samples_leaf"],
                "max_features": training["max_features"],
                "class_weight": None,
                "bootstrap": True,
                "random_state": SEED,
            },
            "ajustement": ("réajusté au chargement depuis le parquet (~10 s) — aucun arbre "
                           "n'est sérialisé ; graine fixée, donc forêt identique"),
            "pourquoi": ("1 200 arbres et 3 740 500 nœuds pèsent ~150 Mo en JSON et 165 Mo "
                         "en joblib compressé ; un pickle scikit-learn se périme en outre à "
                         "la version suivante de la bibliothèque"),
            "contrat": passe_principale["encodage"]["contrat"],
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
    parser.add_argument("--encodage", choices=("dessin", "natif", "les-deux"),
                        default="les-deux",
                        help="Main path only, native path only, or both (default)")
    parser.add_argument("--rapide", action="store_true",
                        help="Reduced grid — smoke pass, never for publishing")
    parser.add_argument("--artefact", action="store_true",
                        help="Also writes rf_mode_choice_policy.json — tree-free replay "
                             "contract, for the experiment platform")
    parser.add_argument("--artefact-seul", action="store_true",
                        help="Regenerates the artefact from rf_mode_choice_metrics.json, "
                             "without replaying tuning (published measurements unchanged)")
    args = parser.parse_args(argv)

    # Tuning takes about twenty minutes. Without this line, `make forest > journal.log`
    # writes nothing until the end: impossible to tell « it is progressing » from « it no
    # longer runs », although per-configuration progress is already printed.
    sys.stdout.reconfigure(line_buffering=True)

    root = find_project_root()
    here = root / "scripts" / "progedo_logit"
    dataset_path = args.dataset or (here / "progedo_mode_choice_v2.parquet")
    spec_path = args.spec or (here / "feature_spec.json")
    out_dir = args.out_dir or here
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(dataset_path)
    check_spec(spec, df)          # same leakage safeguard as the two oracles

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
          f"| ménages={df['hh_id'].nunique()}")
    if args.rapide:
        print("[ALARME] Reduced grid (--rapide): smoke pass, figures not publishable.")

    if args.artefact_seul:
        # Regenerating the artefact must not cost the grid: the published measurements are
        # already written, and the artefact is only a formatting of them. This path
        # RE-MEASURES nothing — it even refuses to run if the measurements file does not
        # exist, rather than invent a pass.
        mesures_path = out_dir / "rf_mode_choice_metrics.json"
        if not mesures_path.exists():
            raise SystemExit(
                f"Measurements missing: {mesures_path}. `--artefact-seul` formats an "
                "existing measurement, it does not produce one. Run `make forest` first.")
        mesures = json.loads(mesures_path.read_text(encoding="utf-8"))
        principal = mesures["principal"]
        if "contrat" not in principal["encodage"]:
            raise SystemExit(
                "The measurements file predates the replay contract (key "
                "`encodage.contrat` missing). Rerun `make forest FOREST_ARGS=--artefact`.")
        matrix, _ = build_matrix(encoded, spec, w, is_train, "dessin")
        trainset_path = out_dir / "rf_mode_choice_trainset.npz"
        np.savez_compressed(trainset_path, X=matrix, y=y.astype("int8"), w=w,
                            is_train=is_train, is_test=is_test)
        artefact = build_artefact(principal, spec, spec_path, dataset_path, trainset_path,
                                  mesures["generated_at"])
        artefact_path = out_dir / "rf_mode_choice_policy.json"
        artefact_path.write_text(json.dumps(artefact, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
        print(f"Artefact regenerated from {mesures_path.name} (no re-measurement):"
              f"\n - {artefact_path} ({artefact_path.stat().st_size / 1e3:.0f} KB)"
              f"\n - {trainset_path} ({trainset_path.stat().st_size / 1e6:.1f} MB, "
              f"{matrix.shape[1]} columns)")
        return 0

    voies = ENCODAGES if args.encodage == "les-deux" else (args.encodage,)
    started = time.monotonic()
    passes = {voie: passe(voie, encoded, spec, y, w, groups, is_train, is_test,
                          classes, args.rapide) for voie in voies}
    duration = round(time.monotonic() - started, 1)

    principal = passes.get("dessin") or passes[voies[0]]
    comparaison = comparer(principal["test"],
                           here / "mode_choice_policy_metrics.json",
                           here / "mnl_model_metrics.json")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temoin": {
            "question": ("l'avantage du booster sur le logit vient-il des ARBRES ou du "
                         "BOOSTING ?"),
            "ticket": "docs/tickets/ticket_044_temoin_random_forest.md",
            "spec": "specs/ticket_044/temoin_random_forest.md",
            "nature": ("témoin, pas candidat — aucun modèle n'est sérialisé, aucun "
                       "consommateur ; rejouer demande `make forest`"),
        },
        "spec_version": spec["spec_version"],
        "dataset": dataset_path.name,
        "split": spec.get("split"),
        "grille_reduite": bool(args.rapide),
        "seconds": duration,
        "principal": principal,
        "comparaison": comparaison,
    }
    if "natif" in passes and passes["natif"] is not principal:
        report["sensibilite_encodage"] = {
            **passes["natif"],
            "ecart_au_principal": {
                "accuracy_weighted": float(passes["natif"]["test"]["accuracy_weighted"]
                                           - principal["test"]["accuracy_weighted"]),
                "cel_weighted": float(passes["natif"]["test"]["cel_weighted"]
                                      - principal["test"]["cel_weighted"]),
                "l1_probability_mass": float(
                    passes["natif"]["test"]["mode_shares"]["l1_probability_mass"]
                    - principal["test"]["mode_shares"]["l1_probability_mass"]),
                "lecture": ("écart de la voie native à la voie principale : ce que coûte de "
                            "lire les catégorielles comme des ordinales"),
            },
        }

    print("\n" + "=" * 70)
    if comparaison["statut"] == "non mesuré":
        print(f"Comparison: not measured — references missing "
              f"({comparaison['references_manquantes']}). "
              f"{comparaison['comment_les_produire']}")
    else:
        print(f"{'axe':26s} {'RF':>9s} {'LightGBM':>9s} {'logit':>9s} {'part':>7s}")
        for key in comparaison["axes"]:
            row = comparaison["axes"][key]
            part = row["part_de_l_ecart_comblee"]
            part_txt = f"{part:7.2f}" if part is not None else "     n/m"
            print(f"{row['libelle']:26s} {row['random_forest']:9.4f} "
                  f"{row['lightgbm']:9.4f} {row['logit']:9.4f} {part_txt}")
        v = comparaison["verdict"]
        print(f"\nVerdict (declared thresholds {SEUIL_BOOSTING} / {SEUIL_ARBRES}): "
              f"**{v['conclusion']}**\n  {v['lecture']}")
    if "sensibilite_encodage" in report:
        ecart = report["sensibilite_encodage"]["ecart_au_principal"]
        print(f"\nCost of the native encoding (ordinals): "
              f"accuracy {ecart['accuracy_weighted']:+.4f}, "
              f"CEL {ecart['cel_weighted']:+.4f}, "
              f"L1 mass {ecart['l1_probability_mass']:+.4f}")

    if args.artefact:
        if "dessin" not in passes:
            raise SystemExit(
                "[ALARME] `--artefact` requires the main path (« dessin »): the control does "
                "not publish a replay contract built on the sensitivity path.")
        # The design matrix ships WITH the artefact: 1.4 MB compressed (it is mostly
        # 0s and 1s), against 165 MB for the forest. That is what lets replay require
        # only numpy — the experiments container has no parquet engine.
        matrix, _ = build_matrix(encoded, spec, w, is_train, "dessin")
        trainset_path = out_dir / "rf_mode_choice_trainset.npz"
        np.savez_compressed(trainset_path, X=matrix, y=y.astype("int8"), w=w,
                            is_train=is_train, is_test=is_test)
        print(f"\nTraining matrix written: {trainset_path} "
              f"({trainset_path.stat().st_size / 1e6:.1f} MB, {matrix.shape[1]} columns)")
        artefact = build_artefact(passes["dessin"], spec, spec_path, dataset_path,
                                  trainset_path, report["generated_at"])
        artefact_path = out_dir / "rf_mode_choice_policy.json"
        artefact_path.write_text(json.dumps(artefact, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
        report["artefact"] = {
            "fichier": artefact_path.name,
            "matrice": trainset_path.name,
            "contient_des_arbres": False,
            "rejeu": ("RFPredictor réajuste la forêt depuis le parquet en ~10 s ; il refuse "
                      "si le SHA du parquet ou la version de scikit-learn diverge"),
        }
        print(f"\nReplay artefact written: {artefact_path} "
              f"({artefact_path.stat().st_size / 1e3:.0f} KB, no tree)")

    out_path = out_dir / "rf_mode_choice_metrics.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\nWritten in {duration:.0f} s:\n - {out_path} "
          f"({out_path.stat().st_size / 1e3:.0f} KB — measurements only, no model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
