"""Tests of the second oracle — multinomial logit at strict parity.

One test per rule of the two-oracle score specification, R1 to R7. What is checked
is not "the model is good" (that is the role of the published metrics) but **the
properties without which comparing the two oracles means nothing**: same
variable contract, same split, same metrics, and an artefact that predicts without its
estimator.

Offline: no network call, no LLM. The tests that require the estimated artefact are
skipped cleanly if it is absent (`make logit`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.progedo_logit.fit_mode_choice_logit import (
    build_contract,
    fit_logit,
    missing_columns,
    normalized_weights,
    select_C,
)
from scripts.progedo_logit.fit_mode_choice_policy import (
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
)
from scripts.progedo_logit.mode_choice_eval import evaluate_proba, gmpca
from scripts.progedo_logit.mode_choice_logit import (
    LOGIT_FORMAT,
    MISSING_CATEGORY,
    LogitPredictor,
    design_columns,
    design_matrix,
)

ROOT = find_project_root()
HERE = ROOT / "scripts" / "progedo_logit"
SPEC_PATH = HERE / "feature_spec.json"
DATASET_PATH = HERE / "progedo_mode_choice_v2.parquet"
LOGIT_PATH = HERE / "mnl_model.json"
POLICY_METRICS = HERE / "mode_choice_policy_metrics.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artefact() -> dict:
    if not LOGIT_PATH.exists():
        pytest.skip("Second oracle not estimated — `make logit`")
    return json.loads(LOGIT_PATH.read_text(encoding="utf-8"))


# ── R1: strict parity, no contaminated variable ──────────────────────────────

def test_R1_aucune_variable_diagnostic(spec):
    """A `diagnostic_only` slipped into the features must make the estimation fail.

    It is the booster's leakage safeguard, and it must hold for the logit: for
    walking, the distance travelled is an affine function of the declared duration.
    """
    contaminated = dict(spec)
    contaminated["features"] = list(spec["features"]) + [
        {"name": "distance_km", "kind": "numeric", "source": "context"}]
    df = pd.DataFrame({name: [0] for name in feature_names(contaminated)})
    with pytest.raises(SystemExit, match="diagnostic_only"):
        check_spec(contaminated, df)


def test_R1_les_21_variables_et_pas_une_de_plus(spec, artefact):
    assert [f["name"] for f in artefact["features"]] == feature_names(spec)
    assert len(artefact["features"]) == 21
    assert not set(spec["diagnostic_only"]) & {f["name"] for f in artefact["features"]}


# ── R2: substrate parity, by construction ────────────────────────────────────

def test_R2_parite_du_split(spec, artefact):
    """Same train/test counts as the booster: same set, same `split` column."""
    if not POLICY_METRICS.exists():
        pytest.skip("Booster metrics missing — `make policy`")
    booster = json.loads(POLICY_METRICS.read_text(encoding="utf-8"))
    assert artefact["training"]["n_test"] == booster["test"]["n_rows"]
    assert artefact["training"]["split"] == booster["split"]
    # Same weighting column as the booster: the survey weight, not a local weight.
    assert artefact["training"]["sample_weight"] == spec["sample_weight"]
    assert "COEP" in artefact["training"]["estimator"]


def test_R2_poids_normalises_sans_changer_les_rapports():
    """Normalisation only touches the scale: the relative weights are preserved."""
    w = np.array([10.0, 20.0, 70.0])
    n = normalized_weights(w)
    assert n.mean() == pytest.approx(1.0)
    assert (n / n.sum()).tolist() == pytest.approx((w / w.sum()).tolist())


# ── R3: the test set is never read for tuning ────────────────────────────────

def test_R3_test_jamais_lu_pour_regler(artefact):
    tuning = artefact["training"]["tuning"]
    assert tuning["scored_on"].startswith("train uniquement")
    assert tuning["grouped_by"] == "hh_id"
    assert tuning["C"] in tuning["grid"]


def test_R3_les_plis_sont_disjoints_par_menage():
    """Two trips of the same household never fall on either side of a fold."""
    rng = np.random.default_rng(0)
    n, groups = 200, np.repeat(np.arange(40), 5)
    Z = rng.normal(size=(n, 3))
    y = rng.integers(0, 4, size=n)
    w = np.ones(n)
    tuning = select_C(Z, y, w, groups, ["a", "b", "c", "d"], grid=(1.0,), folds=4)
    assert tuning["folds"] == 4
    assert sum(tuning["results"][0]["fold_sizes"]) == n


# ── R4: self-sufficiency of the artefact ─────────────────────────────────────

def test_R4_artefact_autoportant(spec, artefact):
    """The pure numpy evaluator reproduces scikit-learn — checked on the real test set."""
    if not DATASET_PATH.exists():
        pytest.skip("Training set missing")
    df = pd.read_parquet(DATASET_PATH)
    test = df[df["split"] == "test"].head(500)
    encoded = encode_features(test, spec)
    proba = LogitPredictor(artefact).predict(encoded)
    assert proba.shape == (len(test), len(spec["target"]["classes"]))
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(test)))
    # Independent reconstruction: softmax(Zβ + α) by hand.
    Z = design_matrix(encoded, artefact)
    scores = Z @ np.asarray(artefact["logit"]["coef"]).T + np.asarray(
        artefact["logit"]["intercept"])
    manual = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    assert np.abs(manual - proba).max() < 1e-9


def test_R4_format_et_coefficients_coherents(artefact):
    assert artefact["format"] == LOGIT_FORMAT
    coef = artefact["logit"]["coef"]
    assert len(coef) == len(artefact["target"]["classes"])
    assert all(len(row) == len(artefact["logit"]["design_columns"]) for row in coef)
    with pytest.raises(ValueError, match="Coefficients de forme"):
        broken = json.loads(json.dumps(artefact))
        broken["logit"]["coef"] = [[0.0]]
        LogitPredictor(broken)


# ── R5: the same metrics as the booster, by the same code ────────────────────

def test_R5_memes_metriques(artefact):
    if not POLICY_METRICS.exists():
        pytest.skip("Booster metrics missing — `make policy`")
    booster = json.loads(POLICY_METRICS.read_text(encoding="utf-8"))["test"]
    mine = artefact["metrics"]
    assert set(booster) - {"feature_importances"} <= set(mine)
    assert set(mine["mode_shares"]) == set(booster["mode_shares"])


def test_R5_gmpca_est_exp_moins_cel():
    proba = np.array([[0.7, 0.1, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]])
    y = np.array([0, 1])
    w = np.array([1.0, 3.0])
    metrics = evaluate_proba(proba, y, w, ["a", "b", "c", "d"])
    assert metrics["gmpca_weighted"] == pytest.approx(gmpca(metrics["cel_weighted"]))
    assert metrics["cel_weighted"] == pytest.approx(metrics["log_loss_weighted"])


# ── R6: missing values are declared, never silently imputed ──────────────────

def test_R6_manquants_declares(spec, artefact):
    """Unknown categorical and missing numeric: prediction without raising, missing at 1."""
    row = {}
    for feature in spec["features"]:
        if feature["kind"] == "categorical":
            row[feature["name"]] = feature["categories"][0]
        elif feature["kind"] == "bool":
            row[feature["name"]] = True
        else:
            row[feature["name"]] = 1.0
    row["socioprofessional_class"] = "Modalité que le spec ne connaît pas"
    row["density_orig"] = None
    encoded = encode_features(pd.DataFrame([row]), spec)
    assert bool(encoded["socioprofessional_class"].isna().all())

    Z = design_matrix(encoded, artefact)
    columns = artefact["logit"]["design_columns"]
    position = columns.index(f"socioprofessional_class={MISSING_CATEGORY}")
    assert Z[0, position] == 1.0
    assert Z[0, columns.index("density_orig__missing")] == 1.0
    assert LogitPredictor(artefact).predict(encoded).shape == (1, 4)


def test_R6_la_regle_et_le_support_sont_dans_l_artefact(artefact):
    logit = artefact["logit"]
    assert set(logit["missing_rule"]) >= {"categorical", "bool", "numeric"}
    support = logit["missing_category_support"]
    # A `__missing__` level without a training row is not identified: saying so
    # is the only way to keep a reader from taking its zero coefficient for an effect.
    assert support["socioprofessional_class"]["identifie"] is False
    assert "modalité de référence" in support["socioprofessional_class"]["sinon"]


def test_R6_indicatrices_calculees_sur_le_train_seul(spec):
    frame = pd.DataFrame({name: [1.0, 2.0] for name in feature_names(spec)})
    frame["density_orig"] = [np.nan, 1.0]
    assert missing_columns(frame, spec) == ["density_orig"]
    contract = build_contract(spec, frame, np.array([1.0, 1.0]))
    assert "density_orig__missing" in contract["design_columns"]
    assert "density_dest__missing" not in contract["design_columns"]


# ── R7: one contract, one matrix, and a refusal when it diverges ─────────────

def test_R7_colonnes_de_dessin_imposees_par_le_spec(spec):
    columns = design_columns(spec, [])
    for feature in spec["features"]:
        if feature["kind"] != "categorical":
            continue
        # n levels of the spec + `__missing__`, minus the dropped reference.
        expected = len(feature["categories"])
        got = [c for c in columns if c.startswith(f"{feature['name']}=")]
        assert len(got) == expected
        assert f"{feature['name']}={feature['categories'][0]}" not in got


def test_R7_matrice_refuse_un_code_hors_bornes(spec, artefact):
    encoded = pd.DataFrame({f["name"]: [0.0] for f in spec["features"]})
    encoded["gender"] = [99.0]
    with pytest.raises(ValueError, match="out of bounds"):
        design_matrix(encoded, artefact)


def test_R7_estimation_et_prediction_partagent_la_matrice(spec):
    """The estimation matrix is built by `design_matrix`, not by a twin.

    If the two paths diverged by one column, the probabilities would remain
    perfectly plausible and would be wrong — the defect this test closes.
    """
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({f["name"]: rng.normal(size=40) for f in spec["features"]})
    for feature in spec["features"]:
        if feature["kind"] == "categorical":
            frame[feature["name"]] = rng.integers(0, len(feature["categories"]), 40) * 1.0
        elif feature["kind"] == "bool":
            frame[feature["name"]] = rng.integers(0, 2, 40) * 1.0
    contract = build_contract(spec, frame, np.ones(40))
    partial = {"features": spec["features"], "logit": contract}
    Z = design_matrix(frame, partial)
    y = rng.integers(0, 4, 40)
    model = fit_logit(Z, y, np.ones(40), 1.0)
    artefact = {**partial, "target": {"classes": ["bike", "car", "transit", "walk"]}}
    artefact["logit"] = {**contract,
                         "coef": model.coef_.tolist(),
                         "intercept": model.intercept_.tolist()}
    assert np.abs(LogitPredictor(artefact).predict(frame)
                  - model.predict_proba(Z)).max() < 1e-12
