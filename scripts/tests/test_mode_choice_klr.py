"""Tests of the third family — kernel logistic regression (KLR + Nyström).

One test per rule of the third-family specification, K1 to K13. What is
checked is not "the model is good" (that is the role of the published metrics) but **the
properties without which comparing the three families means nothing**: same
contract, same design matrix, same split, same metrics, and an artefact that predicts without
its estimator.

Offline: no network call, no LLM. The tests that require the fitted artefact
skip cleanly if it is absent (`make klr`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.progedo_logit.fit_mode_choice_klr import (
    L1_TOLERANCE,
    apply_guard,
    grid_edges,
    draw_landmarks,
    dual_coefficients,
    features,
    fit_klr,
    l1_mass,
    median_sq_distance,
    memory_guard,
    nystrom_map,
    profile_by_decile,
)
from scripts.progedo_logit.fit_mode_choice_logit import build_contract
from scripts.progedo_logit.fit_mode_choice_policy import (
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
)
from scripts.progedo_logit.mode_choice_eval import evaluate_proba, gmpca
from scripts.progedo_logit.mode_choice_klr import (
    KLR_FORMAT,
    KLRPredictor,
    nystrom_basis,
    rbf_kernel,
    squared_distances,
)
from scripts.progedo_logit.mode_choice_logit import (
    design_matrix,
    design_matrix_from_contract,
)

ROOT = find_project_root()
HERE = ROOT / "scripts" / "progedo_logit"
SPEC_PATH = HERE / "feature_spec.json"
DATASET_PATH = HERE / "progedo_mode_choice_v2.parquet"
KLR_PATH = HERE / "klr_model.json"
LOGIT_PATH = HERE / "mnl_model.json"
POLICY_METRICS = HERE / "mode_choice_policy_metrics.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artefact() -> dict:
    if not KLR_PATH.exists():
        pytest.skip("Third family not fitted — `make klr`")
    return json.loads(KLR_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def toy(spec):
    """A small synthetic set: enough to fit a KLR in one second.

    The rule tests must not depend on the survey parquet — they must run
    on a machine that does not have it, and fail for a rule reason, not an access one.
    """
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame({f["name"]: rng.normal(size=n) for f in spec["features"]})
    for feature in spec["features"]:
        if feature["kind"] == "categorical":
            frame[feature["name"]] = rng.integers(
                0, len(feature["categories"]), n) * 1.0
        elif feature["kind"] == "bool":
            frame[feature["name"]] = rng.integers(0, 2, n) * 1.0
    weights = np.ones(n)
    contract = build_contract(spec, frame, weights)
    Z = design_matrix_from_contract(frame, spec["features"], contract)
    y = rng.integers(0, 4, n)
    households = np.repeat(np.arange(n // 3), 3)
    return {"frame": frame, "Z": Z, "y": y, "w": weights, "hh": households,
            "contract": contract, "spec": spec}


# ── K1: strict parity, no contaminated variable ──────────────────────────────

def test_K1_aucune_variable_diagnostic(spec):
    """The booster's leakage safeguard also holds for the third family."""
    contaminated = dict(spec)
    contaminated["features"] = list(spec["features"]) + [
        {"name": "distance_km", "kind": "numeric", "source": "context"}]
    df = pd.DataFrame({name: [0] for name in feature_names(contaminated)})
    with pytest.raises(SystemExit, match="diagnostic_only"):
        check_spec(contaminated, df)


def test_K1_les_21_variables_et_pas_une_de_plus(spec, artefact):
    assert [f["name"] for f in artefact["features"]] == feature_names(spec)
    assert len(artefact["features"]) == 21
    assert not set(spec["diagnostic_only"]) & {f["name"] for f in artefact["features"]}


# ── K2: substrate parity ─────────────────────────────────────────────────────

def test_K2_parite_du_substrat(spec, artefact):
    """Same train/test counts and same weighting as the other two families."""
    if not POLICY_METRICS.exists():
        pytest.skip("Booster metrics missing — `make policy`")
    booster = json.loads(POLICY_METRICS.read_text(encoding="utf-8"))
    assert artefact["training"]["n_test"] == booster["test"]["n_rows"]
    assert artefact["training"]["split"] == booster["split"]
    assert artefact["training"]["sample_weight"] == spec["sample_weight"]
    assert "COEP" in artefact["training"]["estimator"]


def test_K2_memes_effectifs_que_le_logit(artefact):
    if not LOGIT_PATH.exists():
        pytest.skip("Second oracle missing — `make logit`")
    logit = json.loads(LOGIT_PATH.read_text(encoding="utf-8"))
    assert artefact["training"]["n_fit"] == logit["training"]["n_fit"]
    assert artefact["training"]["n_test"] == logit["training"]["n_test"]
    assert artefact["training"]["n_design_columns"] == logit["training"]["n_design_columns"]


# ── K3: the logit design matrix, reused as is ────────────────────────────────

def test_K3_meme_matrice_que_le_logit(toy, spec):
    """Both families build `Z` with the same function, hence strictly the same.

    An encoding redone for the kernel would give perfectly plausible and
    shifted probabilities: that is the defect this test closes.
    """
    logit_artefact = {"features": spec["features"], "logit": toy["contract"]}
    via_logit = design_matrix(toy["frame"], logit_artefact)
    via_contrat = design_matrix_from_contract(
        toy["frame"], spec["features"], toy["contract"])
    assert np.array_equal(via_logit, via_contrat)


def test_K3_le_contrat_de_l_artefact_est_celui_du_logit(artefact):
    design = artefact["klr"]["design"]
    assert set(design) >= {"design_columns", "missing_indicators", "standardization",
                           "reference_category", "missing_rule"}
    if LOGIT_PATH.exists():
        logit = json.loads(LOGIT_PATH.read_text(encoding="utf-8"))["logit"]
        assert design["design_columns"] == logit["design_columns"]
        assert design["standardization"] == logit["standardization"]


def test_K3_manquants_declares(spec, artefact):
    """Unknown categorical and missing numeric: prediction without raising, missing set to 1."""
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
    design = artefact["klr"]["design"]
    Z = design_matrix_from_contract(encoded, artefact["features"], design)
    columns = design["design_columns"]
    assert Z[0, columns.index("socioprofessional_class=__missing__")] == 1.0
    assert Z[0, columns.index("density_orig__missing")] == 1.0
    assert KLRPredictor(artefact).predict(encoded).shape == (1, 4)


# ── K4: landmarks are drawn by household ─────────────────────────────────────

def test_K4_appuis_tires_par_menage(toy):
    households = toy["hh"]
    indices, menages = draw_landmarks(households, 40, np.random.default_rng(0))
    assert len(indices) == len(set(indices.tolist())) == 40
    assert len(set(menages.tolist())) == 40
    # A household never carries two landmarks: that is the whole rule.
    assert sorted(households[indices].tolist()) == sorted(menages.tolist())


def test_K4_refus_si_pas_assez_de_menages(toy):
    with pytest.raises(ValueError, match="available households"):
        draw_landmarks(toy["hh"], 10_000, np.random.default_rng(0))


def test_K4_tirage_reproductible(toy):
    a, _ = draw_landmarks(toy["hh"], 20, np.random.default_rng(3))
    b, _ = draw_landmarks(toy["hh"], 20, np.random.default_rng(3))
    assert np.array_equal(a, b)


# ── K5: tuning never reads the test set ──────────────────────────────────────

def test_K5_reglage_dans_le_train(artefact):
    tuning = artefact["training"]["tuning"]
    if tuning.get("imposed"):
        pytest.skip("Artefact fitted with imposed hyperparameters")
    assert tuning["scored_on"].startswith("train uniquement")
    assert tuning["grouped_by"] == "hh_id"
    assert tuning["C"] in tuning["grid"]["C"]
    assert tuning["m"] in tuning["grid"]["m"]
    assert tuning["gamma_multiplier"] in tuning["grid"]["gamma_multipliers"]
    assert "CHAQUE pli" in tuning["landmark_draw"]


def test_K5_les_appuis_sortent_de_la_partie_d_ajustement(toy):
    """A landmark drawn in the validation fold would bring that fold into the model."""
    from sklearn.model_selection import GroupKFold

    folds = list(GroupKFold(n_splits=4).split(toy["Z"], toy["y"], groups=toy["hh"]))
    for fold, (fit_idx, valid_idx) in enumerate(folds):
        indices, _ = draw_landmarks(toy["hh"][fit_idx], 10,
                                    np.random.default_rng(1000 * fold))
        globaux = set(np.asarray(fit_idx)[indices].tolist())
        assert not globaux & set(np.asarray(valid_idx).tolist())


# ── K6: the bench rules out share drift before ranking ───────────────────────

def test_K6_banc_ecarte_la_derive():
    """The best log-likelihood is not enough: a share drift rules out."""
    reference = 0.0006
    results = [
        {"gamma": 1.0, "C": 1.0, "m": 500, "cv_log_loss_weighted": 0.52,
         "l1_mass_oof": 0.0030, "n_folds_not_converged": 0},
        {"gamma": 4.0, "C": 100.0, "m": 500, "cv_log_loss_weighted": 0.48,
         "l1_mass_oof": 0.0200, "n_folds_not_converged": 0},
    ]
    eligible, rejected, ceiling = apply_guard(results, reference)
    assert ceiling == pytest.approx(reference + L1_TOLERANCE)
    assert [r["C"] for r in eligible] == [1.0]
    assert len(rejected) == 1 and "dérive des parts" in rejected[0]["ecartee_pour"]


def test_K6_banc_ecarte_un_pli_non_converge():
    """A ridge that was not fitted does not compete: the row does not measure its config."""
    results = [{"gamma": 1.0, "C": 100.0, "m": 500, "cv_log_loss_weighted": 0.40,
                "l1_mass_oof": 0.0010, "n_folds_not_converged": 2}]
    eligible, rejected, _ = apply_guard(results, 0.0006)
    assert eligible == []
    assert "itérations" in rejected[0]["ecartee_pour"]


def test_K6_la_reference_et_les_ecartees_sont_dans_l_artefact(artefact):
    tuning = artefact["training"]["tuning"]
    if tuning.get("imposed"):
        pytest.skip("Artefact fitted with imposed hyperparameters")
    guard = tuning["calibration_guard"]
    assert guard["tolerance"] == L1_TOLERANCE
    assert guard["reference"]["family"].startswith("logit")
    assert guard["ceiling"] == pytest.approx(
        guard["reference"]["l1_mass_oof"] + L1_TOLERANCE)
    assert guard["n_eligible"] + guard["n_rejected"] == len(tuning["results"])
    assert all("ecartee_pour" in r for r in guard["rejected"])


def test_K6_un_optimum_au_bord_de_la_grille_est_signale():
    """An optimum on the edge does not say "this is the optimum", but "it may be outside".

    The booster bench publishes the same information; without it, a grid that is too narrow
    reads as a result.
    """
    au_bord = {"gamma_multiplier": 0.25, "C": 10.0, "m": 1000}
    edges = grid_edges(au_bord, (0.25, 0.5, 1.0, 2.0, 4.0), (0.1, 1.0, 10.0, 100.0),
                       (500, 1000, 2000))
    assert any(e.startswith("γ×0.25") for e in edges)
    assert not any(e.startswith("C=") for e in edges)     # C = 10 is interior
    assert not any(e.startswith("m=") for e in edges)     # m = 1000 is interior

    interieur = {"gamma_multiplier": 0.5, "C": 10.0, "m": 1000}
    assert grid_edges(interieur, (0.25, 0.5, 1.0), (0.1, 10.0, 100.0),
                      (500, 1000, 2000)) == []


def test_K6_l1_masse_est_bien_une_L1(toy):
    proba = np.tile([0.25, 0.25, 0.25, 0.25], (len(toy["y"]), 1))
    value = l1_mass(proba, toy["y"], toy["w"], 4)
    observed = np.array([(toy["y"] == k).mean() for k in range(4)])
    assert value == pytest.approx(float(np.abs(0.25 - observed).sum()))


# ── K7: artefact self-containment ────────────────────────────────────────────

def test_K7_coefficients_duaux_reproduisent_sklearn(toy):
    """`softmax(Φβᵀ + α) == softmax(K·(Wβᵀ) + α)`: the folding is exact."""
    rng = np.random.default_rng(11)
    Z, y, w, hh = toy["Z"], toy["y"], toy["w"], toy["hh"]
    gamma = 1.0 / median_sq_distance(Z, rng)
    indices, _ = draw_landmarks(hh, 30, rng)
    landmarks = Z[indices]
    basis, diagnostic = nystrom_map(landmarks, gamma)
    model = fit_klr(features(Z, landmarks, gamma, basis), y, w, 1.0)

    artefact = {
        "features": toy["spec"]["features"],
        "target": {"classes": ["bike", "car", "transit", "walk"]},
        "klr": {"design": toy["contract"],
                "kernel": {"gamma": gamma},
                "nystrom": diagnostic,
                "landmarks": landmarks.tolist(),
                "dual_coef": dual_coefficients(basis, model).tolist(),
                "intercept": model.intercept_.tolist()},
    }
    replayed = KLRPredictor(artefact).predict(toy["frame"])
    direct = model.predict_proba(features(Z, landmarks, gamma, basis))
    assert np.abs(replayed - direct).max() < 1e-9


def test_K7_artefact_autoportant_sur_le_vrai_jeu(spec, artefact):
    if not DATASET_PATH.exists():
        pytest.skip("Training set missing")
    df = pd.read_parquet(DATASET_PATH)
    test = df[df["split"] == "test"].head(500)
    proba = KLRPredictor(artefact).predict(encode_features(test, spec))
    assert proba.shape == (len(test), len(spec["target"]["classes"]))
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(test)))


def test_K7_formats_et_formes_coherents(artefact):
    assert artefact["format"] == KLR_FORMAT
    klr = artefact["klr"]
    m = artefact["training"]["m"]
    assert len(klr["landmarks"]) == m
    assert len(klr["dual_coef"]) == m
    assert all(len(row) == len(artefact["target"]["classes"]) for row in klr["dual_coef"])
    assert all(len(row) == len(klr["design"]["design_columns"]) for row in klr["landmarks"])
    with pytest.raises(ValueError, match="Coefficients duaux de forme"):
        broken = json.loads(json.dumps(artefact))
        broken["klr"]["dual_coef"] = [[0.0, 0.0, 0.0, 0.0]]
        KLRPredictor(broken)
    with pytest.raises(ValueError, match="Landmarks of shape"):
        broken = json.loads(json.dumps(artefact))
        broken["klr"]["landmarks"] = [[0.0, 0.0]] * m
        KLRPredictor(broken)


def test_K7_le_noyau_par_blocs_egale_le_noyau_plein():
    """Chunking bounds the memory; it must not change any value."""
    rng = np.random.default_rng(5)
    Z, landmarks = rng.normal(size=(9000, 6)), rng.normal(size=(25, 6))
    assert np.array_equal(rbf_kernel(Z, landmarks, 0.3, chunk=1024),
                          np.exp(-0.3 * squared_distances(Z, landmarks)))


def test_K7_noyau_rbf_vaut_1_sur_la_diagonale():
    rng = np.random.default_rng(6)
    landmarks = rng.normal(size=(12, 4))
    kernel = rbf_kernel(landmarks, landmarks, 0.5)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.allclose(kernel, kernel.T)
    assert kernel.min() >= 0.0


# ── K8: the same metrics as the other two families ───────────────────────────

def test_K8_memes_metriques(artefact):
    if not POLICY_METRICS.exists():
        pytest.skip("Booster metrics missing — `make policy`")
    booster = json.loads(POLICY_METRICS.read_text(encoding="utf-8"))["test"]
    mine = artefact["metrics"]
    assert set(booster) - {"feature_importances"} <= set(mine)
    assert set(mine["mode_shares"]) == set(booster["mode_shares"])


def test_K8_gmpca_est_exp_moins_cel():
    proba = np.array([[0.7, 0.1, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]])
    metrics = evaluate_proba(proba, np.array([0, 1]), np.array([1.0, 3.0]),
                             ["a", "b", "c", "d"])
    assert metrics["gmpca_weighted"] == pytest.approx(gmpca(metrics["cel_weighted"]))


# ── K10: the effective rank is measured and published ────────────────────────

def test_K10_rang_publie(artefact):
    nystrom = artefact["klr"]["nystrom"]
    assert nystrom["rank"] <= nystrom["m"] == artefact["training"]["m"]
    assert nystrom["rank"] + nystrom["dropped_eigencomponents"] == nystrom["m"]
    assert nystrom["condition_number"] > 0


def test_K10_composante_sous_le_plancher_ecartee_et_comptee():
    """Two coincident landmarks: the null component is dropped, it is not inverted."""
    duplicated = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
    gram = rbf_kernel(duplicated, duplicated, 0.5)
    basis, diagnostic = nystrom_basis(gram)
    assert diagnostic["dropped_eigencomponents"] == 1
    assert diagnostic["rank"] == 2
    assert basis.shape == (3, 2)
    assert np.isfinite(basis).all()


def test_K10_noyau_degenere_refuse():
    with pytest.raises(ValueError, match=r"\[ALARME\]"):
        nystrom_basis(np.zeros((3, 3)))


# ── K13: vacuity ≠ perfection, and memory refuses before allocating ──────────

def test_K13_plafond_memoire_refuse_avant_d_allouer():
    with pytest.raises(SystemExit, match="Carte de Nyström"):
        memory_guard(39_203, 20_000, limit_gb=1.5)
    assert memory_guard(39_203, 500, limit_gb=1.5) < 1.5


def test_K13_banc_sans_configuration_eligible_ne_selectionne_pas():
    """All configurations ruled out: `apply_guard` returns an empty list, not a choice."""
    results = [{"gamma": 1.0, "C": 1.0, "m": 500, "cv_log_loss_weighted": 0.5,
                "l1_mass_oof": 0.9, "n_folds_not_converged": 0}]
    eligible, rejected, _ = apply_guard(results, 0.0006)
    assert eligible == [] and len(rejected) == 1


def test_K13_profil_par_decile_ne_fabrique_pas_de_strate_vide():
    proba = np.tile([0.25, 0.25, 0.25, 0.25], (50, 1))
    values = np.arange(50, dtype="float64")
    rows = profile_by_decile(proba, values, np.ones(50), ["a", "b", "c", "d"], bins=5)
    assert all(row["n"] > 0 for row in rows)
    assert sum(row["n"] for row in rows) == 50
