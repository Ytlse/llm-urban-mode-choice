"""Tests of the PROGEDO mode choice policy
(`scripts/progedo_logit/fit_mode_choice_policy.py`).

What is locked here are the invariants whose violation raises no
exception but produces a silently wrong model:

- no `diagnostic_only` variable enters the model;
- the order of variables and classes of the serialised artefact is **exactly** that
  of `feature_spec.json` — a one-column shift gives plausible
  and wrong probabilities;
- the published categorical encoding is the spec's, and an unknown category becomes
  missing rather than being folded onto a neighbouring code;
- the artefact reloads and predicts a distribution over the 4 classes **without rereading the
  parquet** — that is the contract action A8 consumes.

No retraining of the real model: the tests that need it skip
cleanly if it is absent (it is produced by `make policy`), and a toy model of
a few trees covers the rest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.progedo_logit.fit_mode_choice_policy import (
    POLICY_FORMAT,
    POLICY_FORMAT_VERSION,
    build_policy,
    categorical_encoding,
    categorical_indices,
    check_spec,
    encode_features,
    feature_names,
    find_project_root,
    train_booster,
)

ROOT = find_project_root()
HERE = ROOT / "scripts" / "progedo_logit"
SPEC_PATH = HERE / "feature_spec.json"
POLICY_PATH = HERE / "mode_choice_policy.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def policy() -> dict:
    if not POLICY_PATH.exists():
        pytest.skip(f"Model not trained ({POLICY_PATH.name}) — `make policy`")
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def synthetic_rows(spec: dict, n: int = 8) -> pd.DataFrame:
    """Rows built from the spec alone — never from the parquet.

    That is the point of the test: if predicting required rereading the microdata, the
    self-contained artefact contract would not hold.
    """
    rng = np.random.default_rng(0)
    data = {}
    for feature in spec["features"]:
        name, kind = feature["name"], feature["kind"]
        if kind == "categorical":
            data[name] = list(rng.choice(feature["categories"], size=n))
        elif kind == "bool":
            data[name] = list(rng.random(n) > 0.5)
        else:
            data[name] = list(rng.uniform(0, 20, size=n))
    return pd.DataFrame(data)


# ── Leak: the diagnostic variables ───────────────────────────────────────────

class TestDiagnosticOnly:

    def test_spec_ne_declare_aucune_variable_contaminee_en_feature(self, spec):
        """distance_km / crow_km / duration_min are outside the model, by construction."""
        assert set(spec["diagnostic_only"]) == {"distance_km", "crow_km", "duration_min"}
        assert not set(spec["diagnostic_only"]) & set(feature_names(spec))

    def test_check_spec_refuse_une_variable_contaminee(self, spec):
        polluted = dict(spec)
        polluted["features"] = spec["features"] + [
            {"name": "distance_km", "kind": "numeric", "source": "context"}]
        df = synthetic_rows(spec)
        df["distance_km"] = 1.0
        df["mode"] = "car"
        df["sample_weight"] = 1.0
        df["split"] = "train"
        df["hh_id"] = "x"
        with pytest.raises(SystemExit, match="diagnostic_only"):
            check_spec(polluted, df)

    @pytest.mark.parametrize("banned", ["distance_km", "crow_km", "duration_min"])
    def test_modele_serialise_ignore_les_variables_contaminees(self, policy, banned):
        assert banned not in [f["name"] for f in policy["features"]]


# ── Encoding ─────────────────────────────────────────────────────────────────

class TestEncodage:

    def test_ordre_des_colonnes_impose_par_le_spec(self, spec):
        df = synthetic_rows(spec)
        # Columns deliberately shuffled: the spec must decide.
        shuffled = df[list(reversed(df.columns))]
        assert list(encode_features(shuffled, spec).columns) == feature_names(spec)

    def test_categorielle_encodee_selon_le_spec(self, spec):
        cat = next(f for f in spec["features"] if f["kind"] == "categorical")
        df = synthetic_rows(spec)
        df[cat["name"]] = cat["categories"][-1]
        encoded = encode_features(df, spec)[cat["name"]]
        assert (encoded == len(cat["categories"]) - 1).all()

    def test_modalite_inconnue_devient_manquante(self, spec):
        """Never a fallback code: "unexpected" is not "most frequent"."""
        cat = next(f for f in spec["features"] if f["kind"] == "categorical")
        df = synthetic_rows(spec)
        df[cat["name"]] = "modalité qui n'existe pas"
        assert encode_features(df, spec)[cat["name"]].isna().all()

    def test_booleen_en_zero_un_et_manquant_preserve(self, spec):
        bools = [f["name"] for f in spec["features"] if f["kind"] == "bool"]
        df = synthetic_rows(spec)
        # dtype object: a native bool column does not accept the missing value,
        # whereas the common dataset will produce it (trait absent from a persona).
        df[bools[0]] = pd.Series([True, False] * (len(df) // 2), dtype="object")
        df.loc[0, bools[0]] = None
        encoded = encode_features(df, spec)[bools[0]]
        assert pd.isna(encoded.iloc[0])
        assert set(encoded.dropna().unique()) <= {0.0, 1.0}

    def test_indices_categoriels_pointent_les_bonnes_colonnes(self, spec):
        names = feature_names(spec)
        expected = {f["name"] for f in spec["features"] if f["kind"] == "categorical"}
        assert {names[i] for i in categorical_indices(spec)} == expected


# ── Contract of the serialised artefact ──────────────────────────────────────

class TestArtefact:

    def test_format_et_versions_declares(self, policy, spec):
        assert policy["format"] == POLICY_FORMAT
        assert policy["format_version"] == POLICY_FORMAT_VERSION
        # The runtime refuses a model trained under another feature contract.
        assert policy["spec_version"] == spec["spec_version"]

    def test_ordre_des_variables_identique_au_spec(self, policy, spec):
        assert [f["name"] for f in policy["features"]] == feature_names(spec)
        assert [f["kind"] for f in policy["features"]] == \
               [f["kind"] for f in spec["features"]]

    def test_ordre_des_classes_identique_au_spec(self, policy, spec):
        assert policy["target"]["classes"] == spec["target"]["classes"]

    def test_table_d_encodage_complete_et_conforme(self, policy, spec):
        assert policy["encoding"]["categorical"] == categorical_encoding(spec)
        for feature in policy["features"]:
            if feature["kind"] == "categorical":
                assert feature["categories"] == \
                       [f for f in spec["features"] if f["name"] == feature["name"]][0]["categories"]

    def test_booster_embarque_sous_les_deux_formes(self, policy):
        """dump_model for the pure Python evaluator (E9), text for an exact reload."""
        assert policy["booster"]["dump_model"]["num_class"] == \
               len(policy["target"]["classes"])
        assert policy["booster"]["model_text"].startswith("tree")

    def test_reference_geographique_recopiee(self, policy, spec):
        """Without it, impossible to check that od_km is computed at the same centre."""
        assert policy["geo_reference"] == spec["geo_reference"]


class TestPredictionSansParquet:

    def test_rechargement_et_probabilites_sommant_a_un(self, policy, spec):
        lgb = pytest.importorskip("lightgbm")
        booster = lgb.Booster(model_str=policy["booster"]["model_text"])
        assert booster.feature_name() == [f["name"] for f in policy["features"]]

        rows = synthetic_rows(spec, n=32)
        proba = booster.predict(encode_features(rows, spec))
        assert proba.shape == (32, len(policy["target"]["classes"]))
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()

    def test_valeurs_manquantes_acceptees(self, policy, spec):
        """Density is missing for 81 zones out of 785: predicting must remain possible."""
        lgb = pytest.importorskip("lightgbm")
        booster = lgb.Booster(model_str=policy["booster"]["model_text"])
        rows = synthetic_rows(spec, n=4)
        rows["density_orig"] = np.nan
        rows["density_dest"] = np.nan
        proba = booster.predict(encode_features(rows, spec))
        assert np.allclose(proba.sum(axis=1), 1.0)


# ── Toy model: the contract holds without the real artefact ──────────────────

class TestModeleJouet:
    """Same pipeline, on a few trees and synthetic data.

    Covers `train_booster` / `build_policy` even when the real model has not been
    trained, without ever paying for a full training.
    """

    @pytest.fixture(scope="class")
    def jouet(self, spec):
        pytest.importorskip("lightgbm")
        rows = synthetic_rows(spec, n=200)
        X = encode_features(rows, spec)
        rng = np.random.default_rng(1)
        y = rng.integers(0, len(spec["target"]["classes"]), size=len(rows))
        w = np.ones(len(rows))
        is_valid = np.zeros(len(rows), dtype=bool)
        is_valid[150:] = True
        params = {"objective": "multiclass", "metric": "multi_logloss",
                  "num_leaves": 4, "min_data_in_leaf": 5, "verbosity": -1,
                  "num_threads": 1, "deterministic": True, "force_row_wise": True,
                  "seed": 0}
        booster, training = train_booster(X, y, w, is_valid, spec, params)
        return booster, training

    def test_policy_construite_est_serialisable_et_relisible(self, jouet, spec, tmp_path):
        booster, training = jouet
        policy = build_policy(booster, spec, SPEC_PATH, HERE / "dataset.parquet",
                              training, metrics={})
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")

        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert [f["name"] for f in reloaded["features"]] == feature_names(spec)
        assert reloaded["target"]["classes"] == spec["target"]["classes"]

        lgb = pytest.importorskip("lightgbm")
        proba = lgb.Booster(model_str=reloaded["booster"]["model_text"]).predict(
            encode_features(synthetic_rows(spec, n=5), spec))
        assert proba.shape == (5, len(spec["target"]["classes"]))
        assert np.allclose(proba.sum(axis=1), 1.0)


# ── Safeguards of check_spec ─────────────────────────────────────────────────

class TestCheckSpec:

    @pytest.fixture
    def frame(self, spec):
        df = synthetic_rows(spec, n=4)
        df["mode"] = "car"
        df["sample_weight"] = 1.0
        df["split"] = "train"
        df["hh_id"] = ["a", "a", "b", "b"]
        return df

    def test_jeu_conforme_accepte(self, spec, frame):
        check_spec(spec, frame)  # does not raise

    def test_colonne_manquante_refusee(self, spec, frame):
        with pytest.raises(SystemExit, match="Columns missing"):
            check_spec(spec, frame.drop(columns=["od_km"]))

    def test_ponderation_manquante_refusee(self, spec, frame):
        with pytest.raises(SystemExit, match="sample_weight"):
            check_spec(spec, frame.drop(columns=["sample_weight"]))

    def test_modalite_hors_spec_refusee(self, spec, frame):
        frame["purpose"] = "téléportation"
        with pytest.raises(SystemExit, match="Modalities outside the spec"):
            check_spec(spec, frame)

    def test_classe_cible_hors_spec_refusee(self, spec, frame):
        frame["mode"] = "hélicoptère"
        with pytest.raises(SystemExit, match="Target modalities outside the spec"):
            check_spec(spec, frame)
