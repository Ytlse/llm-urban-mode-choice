"""Tests of the random forest control — ticket 044.

One test per rule of `specs/ticket_044/temoin_random_forest.md`, R1 to R10. What is
checked is not "the random forest is good" — it does not have to be, it is a control — but
**the properties without which its position between the two oracles means nothing**:
same contract, same substrate, a tuning that does not read the test, no class
reweighting, the same metrics, and a verdict whose threshold was written before the figure.

Two tests bear on what the control **does not do**: it serialises no model
(R7) and it does not publish a zero when the reference is missing (R9). These are the two ways
in which this ticket could produce a wrong conclusion without raising a single exception.

Offline: no network call, no LLM. The tests that require the measurement skip
cleanly if it is missing (`make forest`).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from scripts.progedo_logit.fit_mode_choice_forest import (
    ECART_MINIMAL,
    SEUIL_ARBRES,
    SEUIL_BOOSTING,
    build_matrix,
    cel,
    comparer,
    cv_log_loss,
    forest,
    part_de_l_ecart,
    proba_complete,
    verdict,
)
from scripts.progedo_logit.fit_mode_choice_policy import (
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
)
from scripts.progedo_logit.mode_choice_eval import evaluate_proba

ROOT = find_project_root()
HERE = ROOT / "scripts" / "progedo_logit"
SPEC_PATH = HERE / "feature_spec.json"
DATASET_PATH = HERE / "progedo_mode_choice_v2.parquet"
FOREST_PATH = HERE / "rf_mode_choice_metrics.json"
POLICY_METRICS = HERE / "mode_choice_policy_metrics.json"
LOGIT_METRICS = HERE / "mnl_model_metrics.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rapport() -> dict:
    if not FOREST_PATH.exists():
        pytest.skip("Control not measured — `make forest`")
    return json.loads(FOREST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def echantillon(spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An extract of the real set: the matrix tests must see real missing values."""
    if not DATASET_PATH.exists():
        pytest.skip("Training set missing")
    df = pd.read_parquet(DATASET_PATH).head(3000)
    return df, encode_features(df, spec)


# ── R1: contract parity ──────────────────────────────────────────────────────

def test_R1_aucune_variable_diagnostic(spec):
    """A `diagnostic_only` slipped into the features makes it fail before any fitting.

    The safeguard is not copied: it is the booster's `check_spec`, imported as is.
    A copy would diverge at the first addition of a contaminated variable.
    """
    contamine = dict(spec)
    contamine["features"] = list(spec["features"]) + [
        {"name": "distance_km", "kind": "numeric", "source": "context"}]
    df = pd.DataFrame({name: [0] for name in feature_names(contamine)})
    with pytest.raises(SystemExit, match="diagnostic_only"):
        check_spec(contamine, df)


def test_R1_les_21_variables_et_pas_une_de_plus(spec, rapport):
    natif = rapport.get("sensibilite_encodage")
    if natif is None:
        pytest.skip("Native path not measured")
    assert natif["encodage"]["noms_colonnes"] == feature_names(spec)
    assert len(feature_names(spec)) == 21


# ── R2: substrate parity ─────────────────────────────────────────────────────

def test_R2_parite_du_substrat(spec, rapport):
    """Same train/test counts as the booster, and the survey weight, not a local weight."""
    if not POLICY_METRICS.exists():
        pytest.skip("Booster metrics missing — `make policy`")
    booster = json.loads(POLICY_METRICS.read_text(encoding="utf-8"))
    training = rapport["principal"]["training"]
    assert training["n_test"] == booster["test"]["n_rows"]
    assert rapport["split"] == booster["split"]
    assert training["sample_weight"] == spec["sample_weight"]
    assert "COEP" in training["estimator"]


def test_R2_le_split_est_lu_jamais_retire(rapport):
    """`n_fit + n_test` equals the whole set: no line lost, no re-split."""
    training = rapport["principal"]["training"]
    assert training["n_fit"] + training["n_test"] == 52248


# ── R3: the test is never read for tuning ────────────────────────────────────

def test_R3_test_jamais_lu_pour_regler(rapport):
    tuning = rapport["principal"]["training"]["tuning"]
    assert tuning["scored_on"].startswith("train uniquement")
    assert tuning["grouped_by"] == "hh_id"
    assert tuning["criterion"] == "log_loss_weighted"
    retenu = tuning["retenu"]
    assert retenu["max_depth"] in tuning["grille"]["max_depth"]
    assert retenu["min_samples_leaf"] in tuning["grille"]["min_samples_leaf"]


def test_R3_les_plis_sont_disjoints_par_menage():
    """Two trips of the same household never fall on either side of a fold.

    Checked by measuring on a toy set: `cv_log_loss` must return a finite number with
    groups of 5 lines, which is only possible if the split respects the groups.
    """
    rng = np.random.default_rng(0)
    n = 200
    groups = np.repeat(np.arange(40), 5)
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, 4, size=n)
    w = np.ones(n)
    moyenne, par_pli = cv_log_loss(X, y, w, groups, 4, 20, None, 5, "sqrt", folds=4)
    assert len(par_pli) == 4
    assert np.isfinite(moyenne)


# ── R4: neither class_weight nor rebalancing ─────────────────────────────────

def test_R4_ni_class_weight_ni_reequilibrage(rapport):
    """Checked on the published estimator, not on the intention.

    Reweighting the classes inflates the bike recall while destroying the calibration — yet it is
    the probabilities, not the accuracy, that produce the modal shares (decision E7).
    """
    training = rapport["principal"]["training"]
    assert training["class_weight"] is None
    serialise = json.dumps(rapport, ensure_ascii=False)
    for interdit in ("balanced", "is_unbalance", "scale_pos_weight"):
        assert interdit not in serialise


def test_R4_le_constructeur_ne_repondere_pas():
    modele = forest(10, None, 5, "sqrt")
    assert modele.class_weight is None
    assert modele.bootstrap is True


# ── R5: same metrics, same code ──────────────────────────────────────────────

def test_R5_memes_metriques_que_les_deux_oracles(rapport):
    """The keys of the control are exactly those of the two oracles — shared module."""
    if not LOGIT_METRICS.exists():
        pytest.skip("Logit metrics missing — `make logit`")
    logit = json.loads(LOGIT_METRICS.read_text(encoding="utf-8"))["test"]
    attendues = set(logit) - {"top_coefficients"}
    assert attendues <= set(rapport["principal"]["test"])


def test_R5_les_metriques_sortent_du_module_partage():
    """The control has no table of its own: `evaluate_proba` produces the same keys for it."""
    proba = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1]])
    metrics = evaluate_proba(proba, np.array([0, 1]), np.array([1.0, 2.0]),
                             ["bike", "car", "transit", "walk"])
    assert {"cel_weighted", "gmpca_weighted", "accuracy_weighted", "mode_shares",
            "per_class"} <= set(metrics)


def test_R5_cel_lue_sous_ses_deux_noms():
    """A file predating the aliases is read under `log_loss_weighted` — same quantity."""
    assert cel({"cel_weighted": 0.54, "log_loss_weighted": 0.54}) == 0.54
    assert cel({"log_loss_weighted": 0.5402}) == pytest.approx(0.5402)


# ── R6: missing values by the declared rules ─────────────────────────────────

def test_R6_manquants_declares(spec, echantillon):
    """An out-of-spec categorical and a missing numeric go through without raising.

    The design matrix materialises them: `__missing__` level on one side, weighted
    train mean plus indicator on the other. Without the indicator, the imputation
    would assert a value that the data does not carry.
    """
    df, encoded = echantillon
    encoded = encoded.copy()
    encoded.iloc[0, encoded.columns.get_loc("socioprofessional_class")] = np.nan
    encoded.iloc[0, encoded.columns.get_loc("density_orig")] = np.nan
    poids = df[spec["sample_weight"]].to_numpy(dtype=float)
    est_train = np.ones(len(df), dtype=bool)

    matrix, description = build_matrix(encoded, spec, poids, est_train, "dessin")
    assert np.isfinite(matrix).all()
    colonnes = description["noms_colonnes"]
    assert matrix[0, colonnes.index("socioprofessional_class=__missing__")] == 1.0
    # `indicatrices_de_manquant` lists the VARIABLES concerned; the corresponding design
    # column carries the suffix.
    assert "density_orig" in description["indicatrices_de_manquant"]
    assert matrix[0, colonnes.index("density_orig__missing")] == 1.0


def test_R6_la_regle_est_ecrite_dans_la_sortie(rapport):
    encodage = rapport["principal"]["encodage"]
    assert encodage["voie"] == "dessin"
    regle = encodage["regle_des_manquants"]
    assert "__missing__" in regle["categorical"]
    assert "indicatrice" in regle["numeric"]


def test_R6_les_indicatrices_viennent_du_train_seul(spec, echantillon):
    """An indicator derived from the test would let test information into it."""
    df, encoded = echantillon
    encoded = encoded.copy()
    poids = df[spec["sample_weight"]].to_numpy(dtype=float)
    est_train = np.zeros(len(df), dtype=bool)
    est_train[: len(df) // 2] = True
    # A missing value introduced on the test side only must create no column.
    encoded.iloc[-1, encoded.columns.get_loc("age")] = np.nan
    _, description = build_matrix(encoded, spec, poids, est_train, "dessin")
    assert "age__missing" not in description["indicatrices_de_manquant"]


# ── R7: the control produces no model ────────────────────────────────────────

def test_R7_aucun_artefact_predictif(rapport):
    """No trees, no `dump_model`, no coefficients: measurements, and nothing else.

    Settled at the opening of the ticket: a self-contained RF is 400 to 1,200 trees of
    several thousand nodes for a file with no consumer.
    """
    serialise = json.dumps(rapport, ensure_ascii=False)
    # Signatures of a serialised model — never a hyperparameter name: `estimators_` would have
    # matched `n_estimators_cv`, and the test would have failed on its own marker.
    for interdit in ("dump_model", "model_text", "tree_structure", "\"coef\"",
                     "children_left", "children_right", "\"threshold\"", "node_count"):
        assert interdit not in serialise
    assert FOREST_PATH.stat().st_size < 1_000_000
    assert "aucun modèle n'est sérialisé" in rapport["temoin"]["nature"]


def test_R7_le_temoin_reste_hors_du_composite():
    """The control enters neither the composite score nor `common-set-predict`.

    The "no consumer" clause of the morning version is revised: the control is played
    as an experiment since the question asked has been settled. What remains forbidden is
    that it becomes an **arbiter**.

    *Revised on 2026-09-28, after ticket 088 § 3.3.* This test required that
    `model_on_common_set.py` ignore the forest format. Since 2026-09-16, the format appears
    in `POLICY_FORMATS`, which says how to reload an artefact and makes nobody an
    arbiter. What lets an oracle into the composite score is the manifest arms that
    `bi_oracle.py` reads; `common-set-predict` without `POLICY` also takes its policy from
    the manifest. It is this channel that is checked, and the old one did not cover it: an arm
    `model_rf` read by `bi_oracle.py` would have left it green.
    """
    from scripts.progedo_logit.mode_choice_rf import RF_FORMAT
    from scripts.synthesis.sources import load_manifest

    manifeste = load_manifest()
    source = (ROOT / "scripts" / "synthesis" / "bi_oracle.py").read_text(encoding="utf-8")

    # The arms the composite score reads: three oracles. A fourth oracle is a decision, not an
    # addition that goes unnoticed. (Since 2026-09-29 the scoring formula lives in this
    # repository: `bi_oracle.py` no longer reads `arms.calibration.repo` to find it.)
    bras_lus = set(re.findall(r'"arms\.(\w+)', source))
    assert bras_lus == {"model", "model_mnl", "model_klr"}, sorted(bras_lus)

    # No arm of the manifest designates the forest. Checked on the format that each policy
    # declares at the top of the file, not only on its name: a renamed artefact would stay rf.
    for nom in manifeste.get("arms") or {}:
        politique = manifeste.path_of(f"arms.{nom}.policy")
        if politique is None:
            continue
        assert politique.resolve() != ARTEFACT_PATH.resolve(), nom
        if politique.is_file():
            with politique.open(encoding="utf-8") as fh:
                declare = re.search(r'"format"\s*:\s*"([^"]+)"', fh.read(400))
            assert declare, f"arms.{nom}: format unreadable at the top of {politique.name}"
            assert declare.group(1) != RF_FORMAT, f"arms.{nom} designates the forest"

    # The control code stays out of the two score modules. `model_on_common_set.py` knows
    # the format (it reloads the forest for the decision-maker) but does not import the estimation.
    for nom in ("fit_mode_choice_forest", "mode_choice_rf", RF_FORMAT):
        assert nom not in source, nom
    texte = (ROOT / "scripts" / "synthesis" / "model_on_common_set.py").read_text(
        encoding="utf-8")
    assert "fit_mode_choice_forest" not in texte


def test_R7_le_composite_publie_ne_porte_aucun_oracle_rf():
    """Measured on the output, not on the intention: `bi_oracle.json` records each oracle read."""
    from scripts.progedo_logit.mode_choice_rf import RF_FORMAT

    publie = ROOT / "scripts" / "synthesis" / "data" / "bi_oracle.json"
    if not publie.is_file():
        pytest.skip("Composite score not computed — `make bi-oracle`")
    oracles = json.loads(publie.read_text(encoding="utf-8"))["substrat"]["oracles"]
    # An empty dictionary would let the rest pass without having measured anything.
    assert oracles, "no oracle recorded in the published composite score"
    artefact = ARTEFACT_PATH.relative_to(ROOT).as_posix()
    for nom, oracle in oracles.items():
        assert oracle.get("format") != RF_FORMAT, nom
        assert oracle.get("chemin") != artefact, nom


def test_R7_seuls_le_predicteur_et_le_lanceur_consomment_le_temoin():
    """The importers are known and limited: the control, its predictor, its launcher."""
    attendus = {"fit_mode_choice_forest.py", "mode_choice_rf.py",
                "lancer_experience_rf.py", "test_mode_choice_forest.py"}
    importeurs = {
        chemin.name for chemin in (ROOT / "scripts").rglob("*.py")
        if "fit_mode_choice_forest" in chemin.read_text(encoding="utf-8")
    }
    assert importeurs <= attendus


# ── R8: the verdict, and its threshold declared in advance ───────────────────

def test_R8_part_de_l_ecart_sur_un_triplet_calcule_a_la_main():
    """0 at the logit level, 1 at the booster's, and the direction of the metric plays no part."""
    # Accuracy: "better" is higher.
    assert part_de_l_ecart(0.766, 0.785, 0.766) == pytest.approx(0.0)
    assert part_de_l_ecart(0.785, 0.785, 0.766) == pytest.approx(1.0)
    assert part_de_l_ecart(0.7755, 0.785, 0.766) == pytest.approx(0.5)
    # L1: "better" is lower — the formula is unchanged.
    assert part_de_l_ecart(0.0269, 0.0269, 0.0286) == pytest.approx(1.0)
    assert part_de_l_ecart(0.02775, 0.0269, 0.0286) == pytest.approx(0.5)
    # Outside [0, 1]: information, not an anomaly to clamp.
    assert part_de_l_ecart(0.80, 0.785, 0.766) > 1.0
    assert part_de_l_ecart(0.70, 0.785, 0.766) < 0.0


def test_R8_ecart_trop_petit_non_mesure():
    """Two indistinguishable oracles do not give a share, they give "not measured"."""
    assert part_de_l_ecart(0.5, 0.7, 0.7 - ECART_MINIMAL / 2) is None


def test_R8_les_seuils_sont_des_constantes_du_module():
    """Declared in advance and readable: choosing the threshold after the figure chooses the conclusion."""
    assert (SEUIL_BOOSTING, SEUIL_ARBRES) == (0.30, 0.70)
    assert verdict([0.85, 0.92])["conclusion"] == "les arbres"
    assert verdict([0.10, 0.25])["conclusion"] == "le boosting"
    assert verdict([0.50, 0.90])["conclusion"] == "indécis"
    assert verdict([0.85, 0.20])["conclusion"] == "indécis"
    for rendu in (verdict([0.85, 0.92]), verdict([])):
        assert rendu["seuils"]["arbres"] == SEUIL_ARBRES


# ── R9: vacuity ≠ result ─────────────────────────────────────────────────────

def test_R9_comparaison_absente_non_publiee(tmp_path):
    """Missing reference → "not measured", never `0.0`.

    A zero would read "no gap between the three models", exactly the opposite of
    what an absence of measurement says.
    """
    rf = {"n_rows": 10, "accuracy_weighted": 0.77, "cel_weighted": 0.57,
          "mode_shares": {"l1_probability_mass": 0.02, "l1_argmax": 0.09},
          "per_class": {"bike": {"recall": 0.05}}}
    sortie = comparer(rf, tmp_path / "absent_booster.json", tmp_path / "absent_logit.json")
    assert sortie["statut"] == "non mesuré"
    assert sortie["verdict"]["conclusion"] == "non mesuré"
    assert "axes" not in sortie
    assert 0.0 not in list(sortie.values())


def test_R9_le_verdict_sans_mesure_reste_sans_conclusion():
    rendu = verdict([None, None])
    assert rendu["conclusion"] == "non mesuré"


# ── R10: the sensitivity to the encoding is measured ─────────────────────────

def test_R10_les_deux_encodages_existent(spec, echantillon):
    """The two paths produce matrices of different widths on the same set."""
    df, encoded = echantillon
    poids = df[spec["sample_weight"]].to_numpy(dtype=float)
    est_train = np.ones(len(df), dtype=bool)

    dessin, description_dessin = build_matrix(encoded, spec, poids, est_train, "dessin")
    natif, description_natif = build_matrix(encoded, spec, poids, est_train, "natif")

    assert natif.shape[1] == 21
    assert dessin.shape[1] > natif.shape[1]      # expansion into indicators
    assert description_dessin["indicatrices_de_manquant"]
    assert description_natif["indicatrices_de_manquant"] == []
    assert "ORDINALES" in description_natif["regle_des_manquants"]["note"]


def test_R10_encodage_inconnu_refuse(spec, echantillon):
    df, encoded = echantillon
    poids = df[spec["sample_weight"]].to_numpy(dtype=float)
    with pytest.raises(SystemExit, match="Encodage inconnu"):
        build_matrix(encoded, spec, poids, np.ones(len(df), dtype=bool), "one-hot")


def test_R10_ecart_publie(rapport):
    if "sensibilite_encodage" not in rapport:
        pytest.skip("Native path not measured")
    ecart = rapport["sensibilite_encodage"]["ecart_au_principal"]
    assert {"accuracy_weighted", "cel_weighted", "l1_probability_mass"} <= set(ecart)


# ── Cross-cutting safeguard: the four classes, always ────────────────────────

def test_proba_complete_reconstruit_une_classe_absente():
    """A fold without bike must not silently shift the order of the classes."""
    class ModeleJouet:
        classes_ = np.array([1, 2, 3])

        def predict_proba(self, matrix):
            return np.tile([0.5, 0.3, 0.2], (len(matrix), 1))

    plein = proba_complete(ModeleJouet(), np.zeros((3, 2)), 4)
    assert plein.shape == (3, 4)
    assert (plein[:, 0] == 0.0).all()
    assert plein[0].tolist() == [0.0, 0.5, 0.3, 0.2]


# ── R11/R12/R13: the control played as an experiment ─────────────────────────

ARTEFACT_PATH = HERE / "rf_mode_choice_policy.json"
EXPERIENCE_RF = ROOT / "data" / "experiences" / "exp_rf_jtir_nosim" / "experience.yaml"


@pytest.fixture(scope="module")
def artefact_rf() -> dict:
    if not ARTEFACT_PATH.exists():
        pytest.skip("Replay artefact missing — `make forest FOREST_ARGS=--artefact`")
    return json.loads(ARTEFACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predicteur(artefact_rf):
    """Refits the forest once for the whole module (~10 s)."""
    from scripts.progedo_logit.mode_choice_rf import RFPredictor
    return RFPredictor(artefact_rf)


def test_R11_artefact_sans_arbres(artefact_rf):
    """A contract, not a forest: the 3,740,500 nodes are not there."""
    serialise = json.dumps(artefact_rf, ensure_ascii=False)
    # Serialised tree signatures only — a bare "estimators" would match
    # `n_estimators`, which is a hyperparameter and not a forest.
    for interdit in ("children_left", "children_right", "node_count", "estimators_",
                     "tree_structure", "dump_model", '"threshold"'):
        assert interdit not in serialise
    assert ARTEFACT_PATH.stat().st_size < 100_000
    bloc = artefact_rf["rf"]
    assert set(bloc["hyperparameters"]) >= {"n_estimators", "max_depth", "min_samples_leaf",
                                            "max_features", "random_state"}
    assert bloc["hyperparameters"]["class_weight"] is None
    assert len(artefact_rf["dataset_sha256"]) == 64
    assert len(artefact_rf["trainset_sha256"]) == 64
    # The matrix goes alongside: 1.4 MB against 165 MB for the forest.
    assert (HERE / artefact_rf["trainset_file"]).stat().st_size < 5_000_000
    assert bloc["estimator"]["version"]
    assert bloc["contrat"]["design_columns"]


def test_R11_le_contrat_reconstruit_la_meme_matrice(spec, echantillon, artefact_rf):
    """The matrix built from the artefact is the one built from the estimation script."""
    from scripts.progedo_logit.mode_choice_logit import design_matrix

    df, encoded = echantillon
    poids = df[spec["sample_weight"]].to_numpy(dtype=float)
    depuis_script, _ = build_matrix(encoded, spec, poids,
                                    np.ones(len(df), dtype=bool), "dessin")
    depuis_artefact = design_matrix(encoded, {"features": artefact_rf["features"],
                                              "logit": artefact_rf["rf"]["contrat"]})
    # The artefact contract was built on the full TRAIN, the sample on its first 3,000
    # lines: the columns must coincide, the values may differ by a
    # centring. It is the structure that must be identical.
    assert depuis_artefact.shape[1] == len(artefact_rf["rf"]["contrat"]["design_columns"])
    assert depuis_script.shape[0] == depuis_artefact.shape[0]


def test_R12_la_foret_reajustee_reproduit_les_metriques(predicteur):
    """The central safeguard: we measure that it is the forest of the table, we do not presume it."""
    controle = predicteur.controle
    assert controle["reproduit"] is True
    assert max(controle["ecarts"].values()) <= controle["tolerance"]


def test_R12_des_metriques_alterees_font_refuser(predicteur, artefact_rf):
    """A forest that no longer reproduces the published figures is refused, with the gap.

    The forest is already fitted (module fixture): only the check is replayed, with
    falsified published metrics. No refitting, hence not a second lost.
    """
    with np.load(HERE / artefact_rf["trainset_file"]) as jeu:
        matrix, y, w = jeu["X"], jeu["y"].astype("int64"), jeu["w"]

    publie = predicteur.artefact["metrics"]
    predicteur.artefact["metrics"] = {**publie,
                                      "accuracy_weighted": publie["accuracy_weighted"] + 0.01}
    try:
        with pytest.raises(ValueError, match="does not reproduce the published metrics"):
            predicteur._verifier_reproduction(matrix, y, w)
    finally:
        predicteur.artefact["metrics"] = publie


def test_R12_une_matrice_dun_autre_sha_est_refusee(artefact_rf):
    """Flat refusal, and BEFORE any fitting: without the same matrix, nothing to refit."""
    from scripts.progedo_logit.mode_choice_rf import RFPredictor

    faux = dict(artefact_rf)
    faux["trainset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="training matrix has changed"):
        RFPredictor(faux)


def test_R12_une_matrice_absente_est_refusee(artefact_rf, tmp_path):
    from scripts.progedo_logit.mode_choice_rf import RFPredictor

    with pytest.raises(FileNotFoundError, match="Training matrix missing"):
        RFPredictor(artefact_rf, trainset_path=tmp_path / "nexiste_pas.npz")


def test_R12_le_rejeu_ne_demande_pas_de_moteur_parquet():
    """The replay goes through numpy alone: the experiments container has no parquet engine.

    Checked on the code rather than on the environment — a `read_parquet` reintroduced in this
    path would break the replay where it must run, without breaking anything here.
    """
    texte = (HERE / "mode_choice_rf.py").read_text(encoding="utf-8")
    assert "read_parquet" not in texte


@pytest.fixture
def tables_de_famille(monkeypatch):
    """The two tables where the `rf` family is declared since ticket 088 § 3.3."""
    monkeypatch.syspath_prepend(str(ROOT / "services" / "llm-agents"))
    from experiences import decideur_modele

    from scripts.synthesis import model_on_common_set
    return decideur_modele, model_on_common_set


def test_R13_libelle_famille_derive_du_format(tables_de_famille):
    """The family is `rf` — never "lightgbm". A wrong label goes unnoticed.

    *Revised on 2026-09-28, after ticket 088 § 3.3.* The launcher registered the family in
    memory and replaced `load_policy` in the decision-maker's namespace; the family is
    now declared, and the launcher checks the declaration. What the test still fixes:
    the label comes from the table indexed by the format, and the launcher touches neither this
    table nor the `load_policy` that serves the other formats.
    """
    decideur_modele, model_on_common_set = tables_de_famille
    from scripts.progedo_logit.lancer_experience_rf import verifier_famille_rf
    from scripts.progedo_logit.mode_choice_rf import RF_FORMAT

    familles_avant = dict(decideur_modele.FAMILLES)
    verifier_famille_rf()
    # Nothing registered, nothing replaced: the decision-maker loads through the official path.
    assert decideur_modele.FAMILLES == familles_avant
    assert decideur_modele.load_policy is model_on_common_set.load_policy
    assert decideur_modele.FAMILLES[RF_FORMAT] == "rf"
    assert decideur_modele.FAMILLES["lightgbm_mode_choice_policy"] == "lightgbm"
    assert decideur_modele.FAMILLES["mnl_mode_choice_policy"] == "mnl"


@pytest.mark.parametrize("table", ["FAMILLES", "POLICY_FORMATS"])
def test_R13_le_lanceur_refuse_une_famille_non_declaree(tables_de_famille, monkeypatch,
                                                        table):
    """A removed declaration makes the launch be refused; the launcher does not restore it.

    This is what replaces the injection: without this refusal, a line deleted in one of the two
    tables would only show at loading, on « format d'artefact inattendu ».
    """
    decideur_modele, model_on_common_set = tables_de_famille
    from scripts.progedo_logit.lancer_experience_rf import verifier_famille_rf
    from scripts.progedo_logit.mode_choice_rf import RF_FORMAT

    if table == "FAMILLES":
        monkeypatch.delitem(decideur_modele.FAMILLES, RF_FORMAT)
    else:
        monkeypatch.setattr(model_on_common_set, "POLICY_FORMATS", tuple(
            f for f in model_on_common_set.POLICY_FORMATS if f != RF_FORMAT))
    with pytest.raises(SystemExit, match=table):
        verifier_famille_rf()
    # Still missing after the refusal: the launcher reports, it does not fill in.
    apres = (decideur_modele.FAMILLES if table == "FAMILLES"
             else model_on_common_set.POLICY_FORMATS)
    assert RF_FORMAT not in apres


def test_R13_experience_declare_non_rejouable():
    """An experiment that cannot be relaunched by the normal path must say so.

    The warning lives **next to** the `experience.yaml` and not in it: the platform
    schema refuses any key outside the contract, and that is a good thing — an experiment must
    not carry a free field that nobody validates.
    """
    if not EXPERIENCE_RF.exists():
        pytest.skip("Control experiment not launched — lancer_experience_rf.py")
    lisez_moi = EXPERIENCE_RF.parent / "LISEZ-MOI.md"
    assert lisez_moi.exists()
    contenu = lisez_moi.read_text(encoding="utf-8")
    assert "NON REJOUABLE" in contenu
    assert "lancer_experience_rf" in contenu
    # The experience.yaml, for its part, stays compliant with the schema: no key added.
    assert "note:" not in EXPERIENCE_RF.read_text(encoding="utf-8")


def test_R13_meme_couverture_que_les_autres_familles():
    """Without the same coverage, the three composite scores cannot be compared.

    This is the property that makes the control's figure readable next to those of the booster and the
    logit — not that the control is good, but that it decided on the same trips.
    """
    base = ROOT / "data" / "experiences"
    couvertures = {}
    for nom in ("exp_lgbm_jtir_nosim", "exp_mnl_jtir_nosim", "exp_rf_jtir_nosim"):
        executions = base / nom / "executions"
        if not executions.exists():
            pytest.skip(f"{nom} not run")
        derniere = sorted(executions.iterdir())[-1]
        scores = json.loads((derniere / "scores.json").read_text(encoding="utf-8"))
        couvertures[nom] = (scores["couverture"]["decides"],
                            scores["couverture"]["attendus"])
    assert len(set(couvertures.values())) == 1, couvertures
