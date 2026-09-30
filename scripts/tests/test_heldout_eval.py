"""Tests of the held-out set evaluation and of its sample-size control (action A4).

Three independent halves, tested separately:

- **node selection** (`heldout_eval.select_nodes`): which prompts of a
  lineage are measured, and under which roles. A lineage with no ends to contrast
  must raise, not produce a degraded measurement that nobody will notice
  contrasts nothing;
- **the description of frozen sets** (`heldout_eval.dataset_profile`): it is what
  establishes *on the evidence* whether the split is by person or by trip.
  The whole scope of the word "generalization" depends on it, and a page that got
  this wrong would publish the strong claim instead of the weak one;
- **the sample-size control and its use** (`build.resample_composite`,
  `build.resample_gain`, `build.build_generalization`, `render`): without it,
  the train → test gap reads as overfitting while it largely comes
  from the number of persons observed.

Offline: no LLM call, no API key, no real store. The decisions
are built by hand.
"""

from __future__ import annotations

import json

import pytest

from scripts.synthesis import build, frames, heldout_eval, render
from scripts.synthesis.heldout_eval import (
    dataset_profile,
    select_nodes,
    split_rule,
)
from scripts.synthesis.sources import import_calibration, load_manifest

CALIBRATION, _ENGINE_ERROR = import_calibration()
needs_engine = pytest.mark.skipif(
    CALIBRATION is None, reason=f"Moteur de calibration indisponible : {_ENGINE_ERROR}")

CHAIN = ["a" * 16, "b" * 16, "c" * 16, "d" * 16]


# ── Selection of the lineage nodes ───────────────────────────────────────────

def test_selection_par_defaut_prend_les_deux_extremites():
    """`ends` = the pair the page contrasts everywhere else, and nothing more."""
    picked = select_nodes(CHAIN, "ends")
    assert [p["node"] for p in picked] == [CHAIN[0], CHAIN[-1]]
    assert [p["role"] for p in picked] == ["seed", "leaf"]


def test_selection_complete_garde_l_ordre_et_nomme_les_etapes():
    picked = select_nodes(CHAIN, "all")
    assert [p["node"] for p in picked] == CHAIN
    assert [p["role"] for p in picked] == ["seed", "step", "step", "leaf"]
    assert [p["label"] for p in picked][1:3] == ["Étape 1", "Étape 2"]


def test_selection_expose_le_rang_et_la_taille_de_la_lignee():
    """The page must be able to say "2 nodes out of 6" without recounting itself."""
    picked = select_nodes(CHAIN, "ends")
    assert {p["n_nodes_in_lineage"] for p in picked} == {len(CHAIN)}
    assert [p["rank"] for p in picked] == [0, len(CHAIN) - 1]


def test_lignee_d_un_seul_noeud_est_refusee():
    """Measuring a seed alone says nothing about a calibration generalization."""
    with pytest.raises(ValueError, match="no seed"):
        select_nodes(["a" * 16], "ends")


def test_selection_inconnue_est_refusee():
    with pytest.raises(ValueError, match="Unknown selection"):
        select_nodes(CHAIN, "les-deux-premiers")


# ── Description of frozen sets: by person or by trip? ────────────────────────

def _write_split(dir_path, split: str, records: list[dict]):
    (dir_path / f"{split}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")


def test_profil_detecte_un_decoupage_par_personne(tmp_path):
    """No shared person → the generalization is about individuals."""
    _write_split(tmp_path, "train", [{"agent_id": "A", "section": "x"},
                                     {"agent_id": "A", "section": "y"},
                                     {"agent_id": "B", "section": "z"}])
    _write_split(tmp_path, "test", [{"agent_id": "C", "section": "w"}])
    profile = dataset_profile(tmp_path, splits=("train", "test"))
    assert profile["train"]["n_records"] == 3
    assert profile["train"]["n_agents"] == 2
    assert profile["test"]["agents_shared_with_train"] == 0


def test_profil_detecte_un_decoupage_par_deplacement(tmp_path):
    """Shared persons → the strong claim would be false, and it shows.

    This is the case the page must above all not confuse with the previous one:
    different trips of the *same* individuals do not demonstrate the same thing.
    """
    _write_split(tmp_path, "train", [{"agent_id": "A", "section": "x"}])
    _write_split(tmp_path, "test", [{"agent_id": "A", "section": "y"},
                                    {"agent_id": "B", "section": "z"}])
    profile = dataset_profile(tmp_path, splits=("train", "test"))
    assert profile["test"]["agents_shared_with_train"] == 1


def test_profil_mesure_la_presence_de_la_section_historique(tmp_path):
    """The engine removes memory from held-out sets: the input shape differs.

    This is not a population difference but a *prompt* difference, and confusing
    it with the former would attribute to the change of persons a gap that
    comes from a missing section.
    """
    _write_split(tmp_path, "train", [{"agent_id": "A", "section": "**Historique :** …"},
                                     {"agent_id": "B", "section": "sans mémoire"}])
    _write_split(tmp_path, "test", [{"agent_id": "C", "section": "sans mémoire"}])
    profile = dataset_profile(tmp_path, splits=("train", "test"))
    assert profile["train"]["with_memory"] == 1
    assert profile["train"]["memory_share"] == pytest.approx(0.5)
    assert profile["test"]["memory_share"] == pytest.approx(0.0)


def test_profil_ignore_un_split_absent(tmp_path):
    """A clone without a screening set must not make the page fail."""
    _write_split(tmp_path, "train", [{"agent_id": "A", "section": "x"}])
    profile = dataset_profile(tmp_path, splits=("train", "test", "screen"))
    assert set(profile) == {"train"}


def test_regle_de_decoupage_absente_ne_leve_pas(tmp_path):
    assert split_rule(tmp_path) is None


def test_regle_de_decoupage_est_relue_du_manifeste(tmp_path):
    (tmp_path / "manifest.yaml").write_text("split_rule: par personne\n",
                                            encoding="utf-8")
    assert split_rule(tmp_path) == "par personne"


# ── Sample-size control ──────────────────────────────────────────────────────

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


def _frame(n_agents: int, mode: str = "voiture", start: int = 0) -> list[dict]:
    """Scoring frame: two decisions per person, varied strata."""
    ages = ["20-24", "30-34", "45-49", "60-64", "75-130"]
    rows = []
    for i in range(start, start + n_agents):
        for k in range(2):
            rows.append({"agent_id": f"P{i}", "mode_cat": mode, "weight": 1.0,
                         "genre": "Femme" if i % 2 else "Homme",
                         "age_cat": ages[i % len(ages)],
                         "occupation": "actif_temps_plein",
                         "motif": "travail" if k == 0 else "achats",
                         "dist_cat": "2-5km"})
    return rows


@needs_engine
def test_temoin_tire_par_personne_et_garde_tous_ses_trajets(cerema, scorer,
                                                            monkeypatch):
    """The draw is over whole persons: never an isolated trip.

    It is the number of persons per stratum that biases JSD and EMD, not the number
    of rows; a draw per decision would therefore not neutralize the targeted effect,
    while looking as if it did.
    """
    seen: list[list[dict]] = []
    original = scorer.score

    def spy(rows, ref):
        seen.append(rows)
        return original(rows, ref)

    monkeypatch.setattr(scorer, "score", spy)
    build.resample_composite(_frame(20), cerema, scorer, n_agents=5, n_draws=3)
    assert len(seen) == 3
    for rows in seen:
        agents = {r["agent_id"] for r in rows}
        assert len(agents) == 5
        # Two decisions per person in the test frame: none is lost.
        assert len(rows) == 10


@needs_engine
def test_temoin_est_reproductible_a_graine_fixee(cerema, scorer):
    """The page regenerates identically: a control that moves invalidates the displayed Δ."""
    rows = _frame(30)
    a = build.resample_composite(rows, cerema, scorer, n_agents=8, n_draws=20)
    b = build.resample_composite(rows, cerema, scorer, n_agents=8, n_draws=20)
    assert a == b


@needs_engine
def test_temoin_encadre_sa_moyenne(cerema, scorer):
    rows = _frame(30)
    out = build.resample_composite(rows, cerema, scorer, n_agents=8, n_draws=40)
    assert out["p05"] <= out["median"] <= out["p95"]
    assert out["p05"] <= out["mean"] <= out["p95"]
    assert out["n_agents"] == 8 and out["n_agents_source"] == 30


@needs_engine
def test_temoin_refuse_un_echantillon_plus_grand_que_la_source(cerema, scorer):
    """Drawing 66 persons out of 30 makes no sense: better nothing than a number."""
    assert build.resample_composite(_frame(10), cerema, scorer, n_agents=50) is None
    assert build.resample_composite([], cerema, scorer, n_agents=5) is None


@needs_engine
def test_temoin_du_gain_est_apparie(cerema, scorer):
    """Both prompts are scored on THE SAME drawn persons.

    That is the whole point of this second control: sampling noise
    cancels out between the two prompts, whereas it dominates the per-node control. Two
    identical frames must therefore give a strictly zero gain on every
    draw — which would not be the case if the draws were independent.
    """
    rows = _frame(30)
    out = build.resample_gain(rows, list(rows), cerema, scorer,
                              n_agents=8, n_draws=15)
    assert out["p05"] == pytest.approx(0.0)
    assert out["p95"] == pytest.approx(0.0)
    assert out["n_agents_paired"] == 30


@needs_engine
def test_temoin_du_gain_n_apparie_que_les_personnes_communes(cerema, scorer):
    """Two partial evals must not produce an unpaired draw."""
    out = build.resample_gain(_frame(30), _frame(20), cerema, scorer,
                              n_agents=8, n_draws=5)
    assert out["n_agents_paired"] == 20


# ── Assembly: the generalization block ───────────────────────────────────────

REGIME = "modele-test · masse de probabilité"


def _row(short: str, dataset: str, composite: float) -> dict:
    return {"hash": short + "0" * 8, "short": short, "branch": "essai",
            "created_at": "2026-07-31", "verdict": "accepted",
            "eval_model": "modele-test", "store": "local", "dataset": dataset,
            "regime": REGIME, "regime_key": "k", "recomputed": composite,
            "stored": composite, "dims": {"composite": composite}}


def _profile(train_agents=30, held_agents=8, shared=0, memory=(0.9, 0.0)) -> dict:
    return {
        "train": {"n_records": train_agents * 2, "n_agents": train_agents,
                  "with_memory": 0, "memory_share": memory[0],
                  "agents_shared_with_train": None},
        "test": {"n_records": held_agents * 2, "n_agents": held_agents,
                 "with_memory": 0, "memory_share": memory[1],
                 "agents_shared_with_train": shared},
    }


def _generalization(cerema, scorer, *, held_seed=40.0, held_leaf=30.0, shared=0,
                    n_draws=25):
    """Full assembly, fewer draws.

    ``n_draws`` is cut to 25: the page does 200, but each draw costs a
    full score and what is checked here is the *wiring*, not the precision of the
    control — which is tested directly on ``resample_composite``.
    """
    chain = ["1" * 16, "2" * 16]
    by_key = {
        ("11111111", REGIME, "train"): _row("11111111", "train", 25.0),
        ("22222222", REGIME, "train"): _row("22222222", "train", 22.0),
        ("11111111", REGIME, "test"): _row("11111111", "test", held_seed),
        ("22222222", REGIME, "test"): _row("22222222", "test", held_leaf),
    }
    frames_by_key = {
        ("11111111", REGIME, "train"): _frame(30, "voiture"),
        ("22222222", REGIME, "train"): _frame(30, "marche"),
    }
    return build.build_generalization(
        chain, by_key, frames_by_key, cerema, scorer,
        _profile(shared=shared), REGIME, "règle de test", "test",
        n_draws=n_draws)


@pytest.fixture(scope="module")
def generalization(cerema, scorer):
    """Reference assembly, computed only once for the whole class."""
    if CALIBRATION is None:
        pytest.skip(_ENGINE_ERROR)
    return _generalization(cerema, scorer)


@needs_engine
def test_bloc_oppose_le_train_au_jeu_de_retenue(generalization):
    gen = generalization
    assert gen["available"] is True
    assert gen["seed"]["train"] == 25.0 and gen["seed"]["held"] == 40.0
    assert gen["leaf"]["train"] == 22.0 and gen["leaf"]["held"] == 30.0
    assert gen["gain_train"] == pytest.approx(3.0)
    assert gen["gain_held"] == pytest.approx(10.0)


@needs_engine
def test_ecart_corrige_se_lit_contre_le_temoin_et_non_contre_le_train(generalization):
    """The publishable gap is the difference to the control, not to the training score.

    Without this correction, a smaller held-out set would show
    overfitting even with strictly unchanged decisions.
    """
    leaf = generalization["leaf"]
    assert leaf["gap_raw"] == pytest.approx(leaf["held"] - leaf["train"])
    assert leaf["gap_controlled"] == pytest.approx(
        leaf["held"] - leaf["control"]["mean"])
    assert leaf["gap_controlled"] != pytest.approx(leaf["gap_raw"])


@needs_engine
def test_bloc_declare_un_decoupage_par_personne(generalization):
    assert generalization["by_person"] is True


@needs_engine
def test_bloc_declare_un_decoupage_par_deplacement(cerema, scorer):
    """One shared person is enough to withdraw the strong claim."""
    assert _generalization(cerema, scorer, shared=1, n_draws=3)["by_person"] is False


@needs_engine
def test_bloc_absent_quand_aucun_noeud_n_est_mesure_sur_la_retenue(cerema, scorer):
    """No half-measure: the page shows its `Données manquantes` card."""
    by_key = {("11111111", REGIME, "train"): _row("11111111", "train", 25.0)}
    out = build.build_generalization(
        ["1" * 16, "2" * 16], by_key, {}, cerema, scorer, _profile(), REGIME,
        None, "test")
    assert out is None


@needs_engine
def test_bloc_absent_sans_regime_epingle(cerema, scorer):
    """Without a regime, we would be comparing two measuring instruments — so nothing."""
    assert build.build_generalization(
        ["1" * 16, "2" * 16], {}, {}, cerema, scorer, _profile(), None, None,
        "test") is None


# ── Rendering ────────────────────────────────────────────────────────────────

@needs_engine
def test_rendu_affiche_les_deux_temoins_et_la_nature_du_decoupage(generalization):
    html = render._generalization_block({"generalization": generalization})
    assert "Généralisation" in html
    assert "par personne" in html
    assert "témoin" in html.lower()
    assert "tirages appariés" in html
    # The measurement regime is cited: two instruments cannot be subtracted.
    assert REGIME in html


@needs_engine
def test_rendu_signale_la_section_historique_absente_de_la_retenue(generalization):
    """The residual confound must be written down, not left to the reader."""
    html = render._generalization_block({"generalization": generalization})
    assert "Historique" in html


def test_rendu_sans_mesure_donne_une_carte_manquante():
    html = render._generalization_block({"generalization": {
        "available": False, "dataset": "test", "regime": REGIME,
        "reason": "Aucun nœud évalué.", "action": "make heldout-eval"}})
    assert "Données manquantes" in html
    assert "make heldout-eval" in html


def test_matrice_renvoie_vers_le_bloc_sans_y_ajouter_de_colonne():
    """The held-out set is a THIRD substrate: it does not go into the matrix.

    Sticking it there would put a column of 66 persons next to columns of
    881 — exactly the confusion that action A3 fixed.
    """
    payload = {
        "arms": {
            "simulation": {"status": "missing"},
            "calibration": {"status": "ok", "stores": [],
                            "common_set": {"available": False},
                            "generalization": {"available": True, "dataset": "test"}},
            "model": {"predictions": {"available": False}},
        },
        "score_def": {"primary": "emd_jsd"},
    }
    syn = build.build_synthesis(payload)
    assert syn["generalization_available"] is True
    assert syn["generalization_dataset"] == "test"
    assert not any("test" in (a["label"] or "").lower() for a in syn["arms"])


def test_matrice_sans_mesure_de_retenue_ne_revendique_rien():
    payload = {
        "arms": {
            "simulation": {"status": "missing"},
            "calibration": {"status": "ok", "stores": [],
                            "common_set": {"available": False},
                            "generalization": {"available": False}},
            "model": {"predictions": {"available": False}},
        },
        "score_def": {"primary": "emd_jsd"},
    }
    assert build.build_synthesis(payload)["generalization_available"] is False


# ── Action list: what the title promises must stay true ──────────────────────
#
# A4 was the last action "to start". The page's counter is computed,
# but its title — "what is left to DO" — was not: it announced an
# open work item even when everything else was waiting on an external condition.

def _actions_html(actions: list[dict]) -> str:
    return render.section_provenance({"sources": [], "actions": actions})


def test_liste_d_actions_ne_promet_pas_un_travail_qui_n_existe_plus():
    """Everything else is "started" → there is nothing to start, and the page says so."""
    html = _actions_html([
        {"id": "A1", "title": "Faite", "detail": "d", "cost": "5 min",
         "unlocks": "u", "done": "ok"},
        {"id": "A2", "title": "Entamée", "detail": "d", "cost": "1 j", "unlocks": "u",
         "progress": {"acquis": "a", "reste": "un nouveau run"}},
    ])
    assert "1 en\nattente" in html or "1 en attente" in html
    assert "Plus aucune action n'attend qu'on l'engage" in html


def test_liste_d_actions_reste_franche_quand_du_travail_attend():
    html = _actions_html([
        {"id": "A1", "title": "Ouverte", "detail": "d", "cost": "1 j", "unlocks": "u"},
        {"id": "A2", "title": "Entamée", "detail": "d", "cost": "1 j", "unlocks": "u",
         "progress": {"acquis": "a", "reste": "r"}},
    ])
    assert "Plus aucune action n'attend qu'on l'engage" not in html


def test_liste_d_actions_tout_faite():
    html = _actions_html([{"id": "A1", "title": "Faite", "detail": "d", "cost": "5 min",
                           "unlocks": "u", "done": "ok"}])
    assert "Toutes les actions listées sont faites" in html


def test_action_a4_est_marquee_faite_avec_son_resultat():
    """The list is the source of truth between the doc and the rendering (cf. build.ACTIONS).

    This test does not check that work was done — it checks that the entry says
    what the measurement gave, including what it does not demonstrate. A `done`
    entry that merely announced "measured" would suggest a
    result that the 66 persons of the test set do not support.
    """
    a4 = next(a for a in build.ACTIONS if a["id"] == "A4")
    assert a4.get("done"), "A4 must be marked done"
    assert "PAR PERSONNE" in a4["done"], "the nature of the split must be published"
    assert "témoin" in a4["done"], "the sample-size control must be published"


# ── Consistency with the real repository ─────────────────────────────────────

def test_jeux_geles_du_depot_sont_bien_decoupes_par_personne():
    """The fact on which the page's whole vocabulary rests, checked here.

    If the sets were one day regenerated with a split by trip, the
    page would keep working — but would say `par déplacement`, and this
    test would remind that the published claim has changed in nature.
    """
    dataset_dir = load_manifest().path_of("arms.calibration.datasets")
    if dataset_dir is None or not (dataset_dir / "test.jsonl").exists():
        pytest.skip("Frozen sets missing from this clone")
    profile = dataset_profile(dataset_dir)
    assert profile["test"]["agents_shared_with_train"] == 0
    assert profile["val"]["agents_shared_with_train"] == 0
    # `screen` is on the contrary a STRICT subset of train — this is intended, and
    # it is why it cannot serve as a held-out set.
    assert profile["screen"]["agents_shared_with_train"] == profile["screen"]["n_agents"]
