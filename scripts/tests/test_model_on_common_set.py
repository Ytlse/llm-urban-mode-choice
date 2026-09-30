"""Tests of applying the PROGEDO model to the common set (action A8).

What is locked here are the invariants whose violation raises no
exception but produces a plausible and wrong "Modèle" column:

- **the mode mapping**. Four vocabularies cross (policy classes,
  canonical simulator modes, journal labels, page
  categories) and two merges are asymmetric: `train` goes into `transit` on the model side and
  into public transport on the page side — the two agree; `motorbike` goes into
  `car` on the model side and into "autres" on the page side — the two diverge, and the
  page's scope decides;
- **the renormalisation over the OTP offer**, edge cases included: empty offer, offer with
  a single mode, zero mass on what is offered;
- **the order of variables and classes**, compared with the spec at load time — a shift
  of one column gives perfectly plausible probabilities;
- **page generation when the parquet is missing**: it must go on, with
  its "Données manquantes" card.

Offline: no LLM call, no survey data, no network. The real model
is not replayed — the tests that need it skip cleanly if it is missing.
"""

from __future__ import annotations

import json

import pytest

from scripts.synthesis import build, frames
from scripts.synthesis.model_on_common_set import (
    CANONICAL_TO_CAT,
    CAT_TO_POLICY_CLASS,
    COLUMNS,
    POLICY_CLASS_TO_CAT,
    PREDICTABLE_CATS,
    STATUS_NO_OFFER,
    STATUS_NO_PERSONA,
    STATUS_NO_ZONE,
    STATUS_OK,
    activity_index,
    build_rows,
    has_bike,
    load_policy,
    offered_mass,
    persona_features,
    renormalize,
    summarize,
    write_parquet,
)
from scripts.synthesis.sources import REPO_ROOT, import_calibration, load_manifest

CALIBRATION, _ENGINE_ERROR = import_calibration()
needs_engine = pytest.mark.skipif(
    CALIBRATION is None, reason=f"Moteur de calibration indisponible : {_ENGINE_ERROR}")

SPEC_PATH = REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"
POLICY_PATH = REPO_ROOT / "scripts" / "progedo_logit" / "mode_choice_policy.json"


# ── Mode mapping ─────────────────────────────────────────────────────────────

def test_les_quatre_classes_du_spec_ont_toutes_un_mode():
    """A class without a mapping would silently drop out of the score."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert set(spec["target"]["classes"]) == set(POLICY_CLASS_TO_CAT)


def test_les_modes_du_modele_sont_ceux_que_la_page_score():
    """The 4 classes must cover exactly the page's 4 scored modes."""
    assert set(PREDICTABLE_CATS) == set(frames.MODES)
    assert set(CAT_TO_POLICY_CLASS) == set(frames.MODES)


def test_train_est_range_avec_les_transports_collectifs_des_deux_cotes():
    """The model's `train → transit` merge agrees with the page's."""
    assert CANONICAL_TO_CAT["train"] == "transports_collectifs"
    assert frames.CHOSEN_MODE_MAP["Train"] == "transports_collectifs"
    assert CAT_TO_POLICY_CLASS["transports_collectifs"] == "transit"


def test_deux_roues_motorise_sort_du_perimetre_plutot_que_de_compter_pour_voiture():
    """The model's `motorbike → car` merge must NOT be imposed on the page.

    The model files motorised two-wheelers with the car; the page files them in
    "autres", outside the four modes renormalised to 100 %. Counting a two-wheeler
    offer as a car offer would inflate the car share of part 3 alone.
    """
    assert CANONICAL_TO_CAT["motorbike"] == "autres"
    assert frames.CHOSEN_MODE_MAP["Deux-roues motorisé"] == "autres"
    assert "autres" not in PREDICTABLE_CATS


def _canonical_fr() -> dict[str, str]:
    """`_CANONICAL_FR` of the trip journal, rebuilt WITHOUT importing the application.

    `urban_mobility_agents` pulls in `settings` and `models`, hence the whole simulator
    environment: importing it would turn this test into an integration test. Now,
    `move_logger._CANONICAL_FR` is no longer a literal: its labels come from the
    frozen hierarchy (`mobility_core/data/mode_hierarchy_emc2.json`), plus the catch-all
    `other`, which is not a survey family. We rebuild it from the same
    source — it is indeed production that we read, not a copy —, and the column ORDER
    is read back from the `move_logger` source since it only lives there now.
    """
    import ast
    from mobility_core.mode_hierarchy import hierarchy

    source = (REPO_ROOT / "services" / "llm-agents" / "urban_mobility_agents" / "utils"
              / "move_logger.py").read_text(encoding="utf-8")
    ordre = None
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "_COLUMN_ORDER" for t in node.targets)):
            ordre = list(ast.literal_eval(node.value))
    assert ordre, "_COLUMN_ORDER not found in move_logger.py"
    h = hierarchy()
    libelle = {h.canonical_mode[f]: h.journal_label[f] for f in h.families}
    table = {canonique: libelle[canonique] for canonique in ordre}
    table["other"] = "Autres modes"
    return table


def test_les_modes_canoniques_du_simulateur_sont_tous_couverts():
    """The table must stay aligned with `move_logger._CANONICAL_FR`.

    Three vocabularies must commute: canonical mode → CSV label (journal),
    CSV label → page category (`frames`), and canonical mode → category
    (local table). A mode added to the simulator without being filed here would leave the
    scope of part 3 with nothing saying so.
    """
    canonical = _canonical_fr()
    assert set(CANONICAL_TO_CAT) == set(canonical)
    for mode, label in canonical.items():
        assert CANONICAL_TO_CAT[mode] == frames.CHOSEN_MODE_MAP[label]


def test_parse_offered_dedoublonne_et_conserve_l_ordre():
    """Several itineraries often share a mode: it is the set that counts."""
    value = ("Voiture Privée | Transports_collectifs | Marche | "
             "Transports_collectifs | Transports_collectifs")
    assert frames.parse_offered_modes(value) == [
        "voiture", "transports_collectifs", "marche"]


def test_parse_offered_ignore_un_libelle_inconnu():
    """"autres" is an EMC² category, not a catch-all for the unreadable."""
    assert frames.parse_offered_modes("Marche | Téléportation") == ["marche"]
    assert frames.parse_offered_modes("") == []
    assert frames.parse_offered_modes(None) == []


def test_heure_de_depart_est_lue_sur_l_horloge_simulee():
    assert frames.departure_hour("2026-03-16 17:30:23") == 17
    assert frames.departure_hour("2026-03-16 00:04:00") == 0
    assert frames.departure_hour("") is None
    assert frames.departure_hour("pas une date") is None


# ── Renormalisation ──────────────────────────────────────────────────────────

RAW = {"velo": 0.10, "voiture": 0.60, "transports_collectifs": 0.20, "marche": 0.10}


def test_renormalisation_redistribue_la_masse_des_modes_non_offerts():
    out = renormalize(RAW, ["voiture", "marche"])
    assert out == pytest.approx({"voiture": 0.6 / 0.7, "marche": 0.1 / 0.7})
    assert sum(out.values()) == pytest.approx(1.0)


def test_renormalisation_preserve_les_rapports_entre_modes_offerts():
    """IIA assumption: removing a mode does not change the preference between two others."""
    out = renormalize(RAW, ["voiture", "marche"])
    assert out["voiture"] / out["marche"] == pytest.approx(RAW["voiture"] / RAW["marche"])


def test_mode_unique_offert_donne_une_probabilite_de_un():
    """Not a prediction: the observation that there was no choice."""
    assert renormalize(RAW, ["marche"]) == {"marche": 1.0}


def test_offre_vide_ne_produit_pas_de_distribution():
    assert renormalize(RAW, []) is None


def test_offre_sans_mode_predictible_ne_produit_pas_de_distribution():
    assert renormalize(RAW, ["autres"]) is None


def test_masse_nulle_sur_l_offre_ne_divise_pas_par_zero():
    """A model giving no chance to what is offered has nothing to renormalise."""
    assert renormalize({"velo": 0.0, "voiture": 1.0}, ["velo"]) is None


def test_offre_complete_laisse_la_distribution_inchangee():
    out = renormalize(RAW, list(RAW))
    assert out == pytest.approx(RAW)
    assert offered_mass(RAW, list(RAW)) == pytest.approx(1.0)


def test_masse_offerte_mesure_le_facteur_de_renormalisation():
    assert offered_mass(RAW, ["voiture", "marche"]) == pytest.approx(0.70)
    assert offered_mass(RAW, ["autres"]) == pytest.approx(0.0)


# ── Persona traits ───────────────────────────────────────────────────────────

def test_has_bike_suit_la_definition_de_l_entrainement():
    """`M21 > 0` in training: the e-bike (VAE) counts as a bicycle."""
    assert has_bike("vélo normal") is True
    assert has_bike("VAE") is True
    assert has_bike("Pas de vélo") is False
    assert has_bike(None) is None
    assert has_bike("") is None


def test_persona_features_expose_exactement_les_variables_persona_du_spec():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = {f["name"] for f in spec["features"] if f["source"] == "persona"}
    assert set(persona_features({})) == expected


def test_persona_features_ne_remplace_pas_une_modalite_inconnue():
    """"unexpected modality" is not "most frequent modality" (spec)."""
    out = persona_features({"socioprofessional_class": "Retired", "age": 70})
    assert out["socioprofessional_class"] == "Retired"  # left as is…
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    known = next(f for f in spec["features"]
                 if f["name"] == "socioprofessional_class")["categories"]
    assert "Retired" not in known  # … and the encoding will make it missing


# ── Activity chain ───────────────────────────────────────────────────────────

def _person(pid: str, purposes: list[str]) -> dict:
    return {
        "person_id": pid,
        "identity": {
            "traits_json": {"age": 40, "gender": "Male", "personal_bike": "VAE"},
            "activities": [
                {"id": f"{pid}-{i}", "purpose": p,
                 "location": {"lat": 43.6 + i / 100, "lon": 1.44 + i / 100}}
                for i, p in enumerate(purposes)],
        },
    }


def test_l_origine_d_un_deplacement_est_l_activite_precedente():
    index = activity_index([_person("p", ["home", "work", "shop"])])
    assert index["p/p-1"]["purpose"] == "work"
    assert index["p/p-1"]["purpose_origin"] == "home"


def test_la_chaine_d_activites_est_cyclique():
    """The origin of the first trip is the LAST activity, not "nothing".

    The persona's chain loops (`activities[-1].end_time == activities[0]
    .scheduled_start_time`): treating the first trip as having no origin
    would lose one decision per person, and not a random one — the early-morning one.
    """
    index = activity_index([_person("p", ["home", "work", "shop"])])
    assert index["p/p-0"]["purpose_origin"] == "shop"


# ── Row building and statuses ────────────────────────────────────────────────

class _FakeResolver:
    """Toy resolver: returns geo features, or `None` for refused pairs."""

    def __init__(self, refuse: set[int] = frozenset()):
        self.refuse = set(refuse)

    def geo_features_many(self, origins, destinations):
        from mobility_core.zone_resolver import GeoFeatures
        return [None if i in self.refuse else
                GeoFeatures(od_km=2.0, same_zone=False, dist_center_orig_km=3.0,
                            dist_center_dest_km=4.0, density_orig=100.0,
                            density_dest=200.0)
                for i in range(len(origins))]


def _move(agent="p", activity="p-1", offered=("voiture", "marche")) -> dict:
    return {"agent_id": agent, "activity_id": activity, "chosen": "voiture",
            "offered": list(offered), "departure_hour": 8, "genre": "Homme",
            "age_cat": "40-44", "occupation": "actif_temps_plein", "motif": "travail",
            "dist_cat": "2-5km", "lieu_residence": "Toulouse", "type_logement": None}


def test_une_decision_sans_persona_est_comptee_et_non_devinee():
    rows = build_rows([_move(agent="inconnu")], {}, _FakeResolver())
    assert rows[0]["status"] == STATUS_NO_PERSONA


def test_une_offre_sans_mode_predictible_est_comptee():
    index = activity_index([_person("p", ["home", "work"])])
    rows = build_rows([_move(offered=["autres"])], index, _FakeResolver())
    assert rows[0]["status"] == STATUS_NO_OFFER
    assert rows[0]["n_offered"] == 0


def test_une_paire_hors_couche_est_comptee_et_non_extrapolee():
    """`od_km` is the model's first variable: guessing it would be out of domain."""
    index = activity_index([_person("p", ["home", "work"])])
    rows = build_rows([_move()], index, _FakeResolver(refuse={0}))
    assert rows[0]["status"] == STATUS_NO_ZONE
    assert "od_km" not in rows[0]


def test_sans_couche_de_zones_aucune_decision_n_est_predite():
    """A missing layer must produce an honest file, not an invented geography."""
    index = activity_index([_person("p", ["home", "work"])])
    rows = build_rows([_move()], index, None)
    assert rows[0]["status"] == STATUS_NO_ZONE


def test_une_decision_complete_porte_les_six_variables_geo():
    index = activity_index([_person("p", ["home", "work"])])
    rows = build_rows([_move()], index, _FakeResolver())
    assert rows[0]["status"] == STATUS_OK
    for name in ("od_km", "same_zone", "dist_center_orig_km", "dist_center_dest_km",
                 "density_orig", "density_dest"):
        assert name in rows[0]


def test_l_offre_brute_reste_lisible_a_cote_de_l_offre_predictible():
    """Auditability: we must see what OTP offered, not only what we keep of it."""
    index = activity_index([_person("p", ["home", "work"])])
    rows = build_rows([_move(offered=["voiture", "autres"])], index, _FakeResolver())
    assert rows[0]["offered"] == "voiture|autres"
    assert rows[0]["offered_predictable"] == "voiture"


def test_le_resume_compte_les_exclusions():
    rows = [{"status": STATUS_OK, "agent_id": "a", "n_offered": 1,
             "p_offered_mass": 0.5, "argmax_raw": "voiture", "argmax": "voiture"},
            {"status": STATUS_NO_ZONE, "agent_id": "b"}]
    out = summarize(rows)
    assert out["n_scored"] == 1
    assert out["excluded_pct"] == pytest.approx(50.0)
    assert out["status_counts"] == {STATUS_OK: 1, STATUS_NO_ZONE: 1}
    assert out["n_single_offer"] == 1


# ── Contract of the serialised model ─────────────────────────────────────────

@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_le_chargement_refuse_un_contrat_de_features_divergent(spec, tmp_path):
    """Predicting under a spec that has changed gives plausible and wrong probabilities."""
    if not POLICY_PATH.exists():
        pytest.skip("Model not trained — `make policy`")
    artefact = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    artefact["spec_version"] = spec["spec_version"] + 1
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artefact), encoding="utf-8")
    with pytest.raises(ValueError, match="feature contract"):
        load_policy(path, spec)


def test_le_chargement_refuse_un_ordre_de_variables_different(spec, tmp_path):
    if not POLICY_PATH.exists():
        pytest.skip("Model not trained — `make policy`")
    artefact = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    artefact["features"] = list(reversed(artefact["features"]))
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artefact), encoding="utf-8")
    with pytest.raises(ValueError, match="variable order"):
        load_policy(path, spec)


def test_le_chargement_refuse_un_ordre_de_classes_different(spec, tmp_path):
    if not POLICY_PATH.exists():
        pytest.skip("Model not trained — `make policy`")
    artefact = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    artefact["target"]["classes"] = list(reversed(artefact["target"]["classes"]))
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artefact), encoding="utf-8")
    with pytest.raises(ValueError, match="class order"):
        load_policy(path, spec)


def test_le_booster_recharge_attend_les_variables_du_spec(spec):
    """End-to-end contract: the artefact predicts without reading the parquet back."""
    if not POLICY_PATH.exists():
        pytest.skip("Model not trained — `make policy`")
    booster, artefact = load_policy(POLICY_PATH, spec)
    assert list(booster.feature_name()) == [f["name"] for f in spec["features"]]
    assert artefact["target"]["classes"] == spec["target"]["classes"]


# ── K9: three formats, a single prediction path ─────────────────

KLR_PATH = REPO_ROOT / "scripts" / "progedo_logit" / "klr_model.json"


def test_K9_les_trois_familles_passent_par_le_meme_chargeur(spec):
    """The KLR reloads like the other two, and expects the same variables.

    This is the condition for the three parquets to be comparable: two columns
    measured through two paths cannot be compared.
    """
    if not KLR_PATH.exists():
        pytest.skip("Third family not estimated — `make klr`")
    model, artefact = load_policy(KLR_PATH, spec)
    assert artefact["format"] == "klr_mode_choice_policy"
    assert list(model.feature_name()) == [f["name"] for f in spec["features"]]
    assert artefact["target"]["classes"] == spec["target"]["classes"]
    assert hasattr(model, "predict")


def test_K9_le_chargeur_refuse_un_klr_sous_un_autre_contrat(spec, tmp_path):
    if not KLR_PATH.exists():
        pytest.skip("Third family not estimated — `make klr`")
    artefact = json.loads(KLR_PATH.read_text(encoding="utf-8"))
    artefact["spec_version"] = spec["spec_version"] + 1
    path = tmp_path / "klr.json"
    path.write_text(json.dumps(artefact), encoding="utf-8")
    with pytest.raises(ValueError, match="feature contract"):
        load_policy(path, spec)


def test_K9_un_format_inconnu_reste_refuse(spec, tmp_path):
    """Widening the list of formats must not open it to anything at all."""
    path = tmp_path / "inconnu.json"
    path.write_text(json.dumps({"format": "arbre_magique_v9",
                                "spec_version": spec["spec_version"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Format d'artefact inattendu"):
        load_policy(path, spec)


# ── Page reading of the parquet ──────────────────────────────────────────────

def _prediction_row(agent="a", status=STATUS_OK, **over) -> dict:
    row = {c: None for c in COLUMNS}
    row.update({
        "agent_id": agent, "activity_id": f"{agent}-1", "status": status,
        "offered": "voiture|marche", "offered_predictable": "voiture|marche",
        "n_offered": 2, "sim_chosen": "voiture", "argmax_raw": "voiture",
        "argmax": "voiture", "p_offered_mass": 0.8,
        "p_raw_velo": 0.1, "p_raw_voiture": 0.5, "p_raw_transports_collectifs": 0.1,
        "p_raw_marche": 0.3,
        "p_velo": 0.0, "p_voiture": 0.625, "p_transports_collectifs": 0.0,
        "p_marche": 0.375,
        "departure_hour": 8, "genre": "Homme", "age_cat": "40-44",
        "occupation": "actif_temps_plein", "motif": "travail", "dist_cat": "2-5km",
        "lieu_residence": "Toulouse",
    })
    row.update(over)
    return row


def _write(tmp_path, rows, meta=None):
    path = tmp_path / "progedo_on_common_set.parquet"
    write_parquet(rows, path, meta or {"schema": "progedo_on_common_set/v1"})
    return path


def test_parquet_absent_donne_none(tmp_path):
    assert frames.load_model_predictions(tmp_path / "nulle-part.parquet") is None


def test_fichier_illisible_ne_casse_pas_la_page(tmp_path):
    """A truncated parquet must not prevent the page from being generated."""
    path = tmp_path / "tronque.parquet"
    path.write_bytes(b"PAR1 ceci n'est pas un parquet")
    assert frames.load_model_predictions(path) is None


def test_lecture_produit_les_trois_variantes(tmp_path):
    path = _write(tmp_path, [_prediction_row("a"), _prediction_row("b")])
    loaded = frames.load_model_predictions(path)
    assert set(loaded["variants"]) == {"attendu", "elu", "brut"}
    # `attendu`: one row per mode with non-zero mass, after renormalisation.
    assert len(loaded["variants"]["attendu"]) == 4
    # `elu`: one row per decision.
    assert len(loaded["variants"]["elu"]) == 2
    assert {r["mode_cat"] for r in loaded["variants"]["elu"]} == {"voiture"}
    # `brut`: the mass BEFORE renormalisation, over the 4 modes.
    assert len(loaded["variants"]["brut"]) == 8


def test_lecture_ecarte_les_decisions_non_scorees(tmp_path):
    path = _write(tmp_path, [_prediction_row("a"),
                             _prediction_row("b", status=STATUS_NO_ZONE)])
    loaded = frames.load_model_predictions(path)
    assert {r["agent_id"] for r in loaded["variants"]["elu"]} == {"a"}


def test_lecture_conserve_les_strates_par_decision(tmp_path):
    """A person with two purposes keeps both purposes, as for part 2."""
    path = _write(tmp_path, [_prediction_row("a", motif="travail"),
                             _prediction_row("a", motif="achats")])
    loaded = frames.load_model_predictions(path)
    assert {r["motif"] for r in loaded["variants"]["elu"]} == {"travail", "achats"}


def test_le_parquet_ne_republie_pas_les_valeurs_de_la_couche_de_zones(tmp_path):
    """Densities and distances to the centre come from a restricted-access resource.

    `zf_zones.gpkg` is kept out of the repository just like its PROGEDO source. Rewriting
    them row by row into a versionable parquet would republish them for all
    the zones crossed by the run. They serve neither the score nor the join.
    """
    path = _write(tmp_path, [_prediction_row("a")])
    import pyarrow.parquet as pq
    columns = set(pq.read_schema(path).names)
    for name in ("od_km", "same_zone", "dist_center_orig_km", "dist_center_dest_km",
                 "density_orig", "density_dest"):
        assert name not in columns


def test_le_descriptif_voyage_dans_le_fichier(tmp_path):
    meta = {"schema": "progedo_on_common_set/v1", "run": "experiments/archive/x",
            "summary": {"n_scored": 1, "n_moves": 2}}
    path = _write(tmp_path, [_prediction_row("a")], meta)
    assert frames.load_model_predictions(path)["meta"]["run"] == "experiments/archive/x"


# ── Scoring and feeding the matrix ───────────────────────────────────────────

@pytest.fixture(scope="module")
def cerema() -> dict:
    manifest = load_manifest()
    path = manifest.path_of("cerema")
    if path is None or not path.exists():
        pytest.skip("Référence EMC² absente")
    return frames.load_cerema(path)


@pytest.fixture(scope="module")
def scorer(cerema):
    if CALIBRATION is None:
        pytest.skip(_ENGINE_ERROR)
    weights = load_manifest().get("score.weights", {})
    return frames.Scorer(CALIBRATION, weights, "emd_jsd", "l1_composite")


def _source(path):
    from scripts.synthesis.sources import probe
    return probe("model.predictions", path, "test")


@needs_engine
def test_scores_sortent_de_la_loss_du_moteur(tmp_path, cerema, scorer):
    """The displayed composite score must be the engine's, not a loss rewritten here."""
    rows = [_prediction_row(str(i)) for i in range(10)]
    path = _write(tmp_path, rows)
    out = build.build_model_predictions(_source(path), cerema, scorer)
    assert out["available"] is True
    frame = frames.load_model_predictions(path)["variants"]["attendu"]
    direct = scorer.score(frame, cerema)
    assert out["variants"]["attendu"]["composite"] == pytest.approx(
        direct[scorer.primary.name]["composite"])


@needs_engine
def test_penalite_de_longueur_reste_neutralisee(tmp_path, cerema, scorer):
    """Part 3 has no prompt: the length term must be 0."""
    path = _write(tmp_path, [_prediction_row(str(i)) for i in range(6)])
    out = build.build_model_predictions(_source(path), cerema, scorer)
    assert out["variants"]["attendu"]["dims"].get("length_penalty") == pytest.approx(0.0)


@needs_engine
def test_source_absente_ne_casse_rien(tmp_path, cerema, scorer):
    out = build.build_model_predictions(_source(tmp_path / "absent.parquet"),
                                        cerema, scorer)
    assert out == {"available": False}


@needs_engine
def test_la_matrice_porte_les_deux_lectures_du_modele(tmp_path, cerema, scorer):
    rows = [_prediction_row(str(i)) for i in range(8)]
    preds = build.build_model_predictions(_source(_write(tmp_path, rows)), cerema, scorer)
    payload = {
        "score_def": {"primary": "emd_jsd"},
        "arms": {"simulation": {"status": "missing"},
                 "calibration": {"status": "missing", "stores": []},
                 "model": {"predictions": preds}},
    }
    syn = build.build_synthesis(payload)
    labels = [a["label"] for a in syn["arms"]]
    assert "Modèle" in labels and "Modèle (élu)" in labels
    assert syn["model_available"] is True
    column = next(a for a in syn["arms"] if a["label"] == "Modèle")
    assert column["basis"].startswith("jeu commun")
    assert column["cells"][-1]["value"] == pytest.approx(
        preds["variants"]["attendu"]["composite"])


@needs_engine
def test_la_matrice_reste_vide_sans_predictions(tmp_path, cerema, scorer):
    """Behaviour before action A8: a "Modèle" column entirely n/a."""
    payload = {
        "score_def": {"primary": "emd_jsd"},
        "arms": {"simulation": {"status": "missing"},
                 "calibration": {"status": "missing", "stores": []},
                 "model": {"predictions": {"available": False}}},
    }
    syn = build.build_synthesis(payload)
    labels = [a["label"] for a in syn["arms"]]
    assert "Modèle (élu)" not in labels
    column = next(a for a in syn["arms"] if a["label"] == "Modèle")
    assert all(c["value"] is None for c in column["cells"])
    assert syn["model_available"] is False


@needs_engine
def test_le_substrat_annonce_la_part_reellement_predite(tmp_path, cerema, scorer):
    """A truncated population must not present itself as "the common set"."""
    rows = [_prediction_row(str(i)) for i in range(8)]
    rows.append(_prediction_row("hors", status=STATUS_NO_ZONE))
    meta = {"schema": "progedo_on_common_set/v1",
            "summary": {"n_scored": 8, "n_moves": 9, "excluded_pct": 100 / 9}}
    preds = build.build_model_predictions(
        _source(_write(tmp_path, rows, meta)), cerema, scorer)
    payload = {"score_def": {"primary": "emd_jsd"},
               "arms": {"simulation": {"status": "missing"},
                        "calibration": {"status": "missing", "stores": []},
                        "model": {"predictions": preds}}}
    syn = build.build_synthesis(payload)
    column = next(a for a in syn["arms"] if a["label"] == "Modèle")
    assert "8/9" in column["basis"]
