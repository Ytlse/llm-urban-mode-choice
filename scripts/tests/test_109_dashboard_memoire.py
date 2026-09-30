"""Unit and integration tests for Ticket 109 — Memory experiments control page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_CHEMIN_SERVICES = REPO_ROOT / "services" / "llm-agents"
if str(_CHEMIN_SERVICES) not in sys.path:
    sys.path.insert(0, str(_CHEMIN_SERVICES))

from experiences import memoire
from scripts.dashboard import app as APP
from scripts.experiment import orchestrateur_memoire


@pytest.fixture
def tmp_memoire_env(tmp_path, monkeypatch):
    """Isolated environment for memory experiment data."""
    dossier_exp = tmp_path / "data" / "experiences_memoire"
    dossier_exp.mkdir(parents=True)
    fichier_form = tmp_path / "formulaire_memoire.yaml"

    monkeypatch.setattr(memoire, "DOSSIER_MEMOIRE", dossier_exp)
    monkeypatch.setattr(memoire, "ETAT_FORMULAIRE_MEMOIRE", fichier_form)
    return {"dossier": dossier_exp, "form": fichier_form}


# ── 1. Canonical naming (D4) ───────────────────────────────────────────────────
def test_nom_canonique_format():
    nom = memoire.nom_canonique(
        canal="vecu",
        evenement="c6_voiture_suspecte",
        modele_decision="gemini-3.8-flash",
        population="899549",
        horizon_jours=42,
    )
    assert nom.startswith("exp_mem_choc_")
    assert "c6" in nom
    assert "gem38f" in nom
    assert "899549" in nom
    assert "42j" in nom

    nom_presse = memoire.nom_canonique(
        canal="lu",
        evenement="a13_punaises_metro",
        modele_decision="gemini-3.1-flash-lite",
        population="pop20",
        horizon_jours=14,
    )
    assert nom_presse.startswith("exp_mem_presse_")
    assert "a13" in nom_presse
    assert "gem31flite" in nom_presse


def test_attribuer_nom_avec_collision(tmp_memoire_env):
    dossier = tmp_memoire_env["dossier"]
    nom1 = memoire.attribuer_nom("vecu", "c6_voiture_suspecte", "gemini-3.8-flash", "899549", 42)
    (dossier / nom1).mkdir()
    nom2 = memoire.attribuer_nom("vecu", "c6_voiture_suspecte", "gemini-3.8-flash", "899549", 42)
    assert nom2 == f"{nom1}_2"


# ── 2. Event catalogue and channel filtering (D5) ──────────────────────────────
def test_catalogue_evenements_filtrage():
    chocs = memoire.catalogue_evenements("vecu")
    assert len(chocs) >= 6
    assert all(c["canal"] == "vecu" for c in chocs)
    ids_chocs = {c["id"] for c in chocs}
    assert "c6_voiture_suspecte" in ids_chocs
    assert "c1_bouchon_rocade" in ids_chocs

    presse = memoire.catalogue_evenements("lu")
    assert len(presse) >= 4
    assert all(p["canal"] == "lu" for p in presse)
    ids_presse = {p["id"] for p in presse}
    assert "a13_punaises_metro" in ids_presse


# ── 3. Form persistence (D2) ───────────────────────────────────────────────────
def test_persistance_etat_formulaire(tmp_memoire_env):
    defauts = memoire.defauts()
    assert "modeles" in defauts
    # Six functions since ticket 111: `evenement_relais`, the reader's relay to its household.
    assert set(defauts["modeles"]) == set(memoire.CATEGORIES_COGNITIVES)
    assert len(defauts["modeles"]) == 6

    personnalise = dict(defauts)
    personnalise["population"] = "1250941"
    personnalise["horizon_jours"] = 30
    personnalise["modeles"]["itinary_multi_agent"] = "gpt-5.6-luna"

    memoire.sauver_etat_formulaire(personnalise)
    recharge = memoire.charger_etat_formulaire()

    assert recharge["population"] == "1250941"
    assert recharge["horizon_jours"] == 30
    assert recharge["modeles"]["itinary_multi_agent"] == "gpt-5.6-luna"


# ── 4. Resolution of the instances_admises routing table (D2) ──────────────────
def test_resoudre_instances_admises():
    modeles_test = {
        "itinary_multi_agent": "gemini-3.8-flash",
        "evenement_jugement": "aucun",
        "stm_reflection": "gemini-3.8-flash",
        "ltm_self_reflection": "gemini-3.8-flash",
        "enquete_affinite": "gemini-3.8-flash",
    }
    routage = memoire.resoudre_instances_admises(modeles_test)
    assert "itinary_multi_agent" in routage
    assert "stm_reflection" in routage
    assert "evenement_jugement" in routage
    assert routage["evenement_jugement"] == []  # ablation
    assert "defaut" in routage
    assert len(routage["itinary_multi_agent"]) >= 1


def test_adaptateurs_disponibles_et_modeles_servis():
    adaptateurs = memoire.adaptateurs_disponibles()
    assert isinstance(adaptateurs, list)
    assert "google" in adaptateurs
    assert "groq" in adaptateurs

    modeles_google = memoire.modeles_servis("google")
    assert isinstance(modeles_google, set)
    assert "gemini-3.8-flash" in modeles_google

    modeles_groq = memoire.modeles_servis("groq")
    assert isinstance(modeles_groq, set)
    assert len(modeles_groq) >= 1


def test_resoudre_instances_admises_avec_adaptateur():
    modeles_groq = {
        "itinary_multi_agent": "openai/gpt-oss-120b",
        "evenement_jugement": "qwen/qwen3.8-27b",
        "stm_reflection": "openai/gpt-oss-20b",
        "ltm_self_reflection": "openai/gpt-oss-20b",
        "enquete_affinite": "openai/gpt-oss-20b",
    }
    routage_groq = memoire.resoudre_instances_admises(modeles_groq, adaptateur="groq")
    assert "itinary_multi_agent" in routage_groq
    assert all("groq" in inst for inst in routage_groq["itinary_multi_agent"])
    assert all("groq" in inst for inst in routage_groq["evenement_jugement"])
    assert all("groq" in inst for inst in routage_groq["stm_reflection"])
    assert routage_groq["defaut"] == routage_groq["itinary_multi_agent"]


# ── 5. Recording and viewing an experiment (D3 & D4) ───────────────────────────
def test_enregistrer_et_lister_experience(tmp_memoire_env):
    cfg = memoire.defauts()
    cfg["nom"] = "exp_mem_choc_test_gem38f_899549_42j"
    exp_dir, modifie = memoire.enregistrer_experience(cfg)
    assert modifie is True
    assert (exp_dir / "experience_memoire.yaml").is_file()
    assert (exp_dir / "etat.json").is_file()

    liste = memoire.lister_experiences()
    assert len(liste) == 1
    assert liste[0]["nom"] == "exp_mem_choc_test_gem38f_899549_42j"
    assert liste[0]["etat"] == "en_attente"


# ── 6. Estimation and counterfactual A/B orchestration (D3) ───────────────────
def test_estimer_cout_ab():
    cfg = memoire.defauts()
    cfg["horizon_jours"] = 10
    cfg["population"] = "899549"

    bilan = orchestrateur_memoire.estimer_cout(cfg)
    assert bilan["nb_personas"] == 1
    assert bilan["horizon_jours"] == 10
    assert bilan["total_requetes_experience_ab"] == bilan["total_requetes_par_bras"] * 2


def test_orchestration_ab_dry_run(tmp_memoire_env, monkeypatch):
    cfg = memoire.defauts()
    cfg["nom"] = "exp_mem_choc_dry_gem38f_899549_42j"
    exp_dir, _ = memoire.enregistrer_experience(cfg)

    # Run of the two consecutive arms, where `main()` stores a dry run
    workdir_traite = exp_dir / "essai_a_blanc" / "traite"
    workdir_temoin = exp_dir / "essai_a_blanc" / "temoin"

    ret_a = orchestrateur_memoire.executer_bras(cfg["nom"], "treated", cfg, workdir_traite, dry_run=True)
    assert ret_a == 0
    assert (workdir_traite / "moves.csv").is_file()
    assert (workdir_traite / "evenements.jsonl").is_file()

    ret_b = orchestrateur_memoire.executer_bras(cfg["nom"], "control", cfg, workdir_temoin, dry_run=True)
    assert ret_b == 0
    assert (workdir_temoin / "moves.csv").is_file()

    # Ticket 108 reconciliation
    rappr = orchestrateur_memoire.rapprochement_injections(cfg, workdir_traite)
    assert "declarees" in rappr
    assert "produites" in rappr

    # Synthetic traces do not make a completed experiment: the registry only reads
    # `traite/` and `temoin/`, which the dry run leaves empty (c6 incident of 2026-09-24).
    liste = memoire.lister_experiences()
    assert len(liste) == 1
    assert liste[0]["statut_traite"] != "ok"
    assert liste[0]["statut_temoin"] != "ok"
    assert liste[0]["etat"] != "terminee"


def test_un_essai_a_blanc_ne_touche_ni_a_l_etat_ni_aux_bras(tmp_memoire_env, monkeypatch):
    """On 2026-09-24, a dry run had marked c6 "terminee" in 50 ms: a real
    relaunch would have skipped both arms, and an archiving would have carried off fake traces."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_blanc_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    etat_avant = (exp_dir / "etat.json").read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"], "--dry-run"])
    orchestrateur_memoire.main()
    assert (exp_dir / "etat.json").read_text(encoding="utf-8") == etat_avant
    assert not (exp_dir / "traite" / "moves.csv").exists()
    assert not (exp_dir / "temoin" / "moves.csv").exists()
    assert (exp_dir / "essai_a_blanc" / "traite" / "moves.csv").is_file()
    assert memoire.lister_experiences()[0]["etat"] != "terminee"


# ── 6 bis. Household population and in-household sharing (ticket 100, lot 4) ───────
def test_estimer_cout_lit_l_effectif_reel():
    """A set of twenty households counted as one persona: the estimate was twenty times too low."""
    cfg = memoire.defauts()
    cfg["population"] = "population_20_foyers_059"
    assert orchestrateur_memoire.estimer_cout(cfg)["nb_personas"] == 20


def test_decrire_population_compte_les_foyers_partages():
    assert memoire.decrire_population("population_20_foyers_059") == {"agents": 20, "foyers_partages": 10}
    assert memoire.decrire_population("cohorte_10")["foyers_partages"] == 0


def test_le_partage_du_foyer_entre_dans_le_nom():
    args = ("lu", "a09_vent_autan", "openai/gpt-oss-120b", "population_20_foyers_059", 15)
    sans, avec = memoire.nom_canonique(*args), memoire.nom_canonique(*args, partage_foyer=True)
    assert avec == sans + "_foyer"


def test_le_bras_recoit_horizon_et_partage(tmp_path, monkeypatch):
    """The horizon and the flag reach the launcher; the flag is set even when false, so that the
    launcher checks it in `identite_run.json`."""
    vus = {}

    class _Fini:
        returncode = 0

    def _faux_run(cmd, cwd=None, env=None):
        vus["cmd"], vus["env"] = cmd, env
        return _Fini()

    monkeypatch.setattr(orchestrateur_memoire.subprocess, "run", _faux_run)
    cfg = {**memoire.defauts(), "population": "population_20_foyers_059",
           "horizon_jours": 15, "partage_foyer": True, "evenement": "a09_vent_autan"}
    assert orchestrateur_memoire.executer_bras("exp_t", "treated", cfg, tmp_path / "t") == 0
    i = vus["cmd"].index("--horizon-jours")
    assert vus["cmd"][i + 1] == "15"
    assert vus["env"]["MEMOIRE__PARTAGE_FOYER_ENABLED"] == "true"
    # Micro-batching: 20 agents → 20 tasks in flight, one batch per wave of departures.
    assert vus["env"]["WORLD__WORKER_CONCURRENCY"] == "20"

    cfg["partage_foyer"] = False
    orchestrateur_memoire.executer_bras("exp_t", "treated", cfg, tmp_path / "t")
    assert vus["env"]["MEMOIRE__PARTAGE_FOYER_ENABLED"] == "false"


def test_le_partage_sans_foyer_est_refuse(tmp_memoire_env, monkeypatch):
    """On a single persona, sharing would run empty and would read as "nothing is passed on"."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_c6_gem38f_899549_42j_foyer",
           "population": "899549", "partage_foyer": True}
    memoire.enregistrer_experience(cfg)
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"], "--dry-run"])
    with pytest.raises(SystemExit) as fin:
        orchestrateur_memoire.main()
    assert fin.value.code == 2


def test_un_bras_seul_laisse_l_experience_partielle(tmp_memoire_env, monkeypatch, ab_neutre):
    """D3: a treated arm played alone (debug) never passes for a completed A/B measurement."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_dry_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    # A simulated arm, not a dry run: the latter no longer writes any state.
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({}, []))
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"], "--branche", "treated"])
    orchestrateur_memoire.main()
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["etat"] == "traite_ok"


def _bras_simule(codes: dict[str, int], joues: list[str]):
    """Replaces `executer_bras`: returns the planned code for each arm and records those played."""
    def _executer(nom, branche, config, cible, dry_run=False, strict_avant_ts=None):
        joues.append(branche)
        return codes.get(branche, 0)
    return _executer


@pytest.fixture
def ab_neutre(monkeypatch):
    """Since 2026-09-28, every memory experiment plays with a common prefix: the bound, the
    store and the replay check are neutralised when the test is about something else."""
    monkeypatch.setattr(orchestrateur_memoire, "borne_prefixe_commun", lambda racine: 1774597500)
    monkeypatch.setattr(orchestrateur_memoire, "mettre_de_cote_magasin", lambda nom: None)
    monkeypatch.setattr(
        orchestrateur_memoire, "controler_rejeu",
        lambda nom, racine: {"payes_avant_evenement": 0, "prefixe_deplacements": {"conforme": True}},
    )


@pytest.mark.parametrize("surcharge", [
    {"prefixe_commun": False},
    {"prefixe_commun": None},
    {"rejeu_ab": False},
])
def test_une_experience_memoire_sans_prefixe_commun_est_refusee(tmp_memoire_env, monkeypatch,
                                                                 surcharge):
    """2026-09-28: every memory experiment plays with a common prefix. A declaration that does not
    say so is refused before any arm starts, never reinterpreted."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_sanspref_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    cfg_path = exp_dir / "experience_memoire.yaml"
    declaration = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for cle, valeur in surcharge.items():
        if valeur is None:
            declaration.pop(cle, None)
        else:
            declaration[cle] = valeur
    cfg_path.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    joues: list[str] = []
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({}, joues))
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"]])
    with pytest.raises(SystemExit) as fin:
        orchestrateur_memoire.main()
    assert fin.value.code == 2
    assert joues == []


def test_une_experience_neuve_porte_le_prefixe_commun():
    assert memoire.defauts()["prefixe_commun"] is True
    assert memoire.defauts()["rejeu_ab"] is True


def test_un_bras_suspendu_suspend_l_experience_sans_lancer_le_temoin(tmp_memoire_env, monkeypatch,
                                                                     ab_neutre):
    """Option A: an exhausted quota suspends treated; control does not start on an empty quota."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_susp_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    joues: list[str] = []
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({"treated": 7}, joues))
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"]])
    with pytest.raises(SystemExit) as fin:
        orchestrateur_memoire.main()
    assert fin.value.code == orchestrateur_memoire.CODE_BRAS_SUSPENDU
    assert joues == ["treated"]
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["etat"] == "suspendue"
    assert etat["traite"]["etat"] == "suspendu"
    assert etat["temoin"]["etat"] == "en_attente"
    # The registry shows it suspended, not "treated OK".
    assert memoire.lister_experiences()[0]["etat"] == "suspendue"


def test_la_relance_reprend_le_bras_suspendu_puis_joue_le_temoin(tmp_memoire_env, monkeypatch,
                                                                  ab_neutre):
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_rep_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    (exp_dir / "etat.json").write_text(json.dumps({
        "nom": cfg["nom"], "etat": "suspendue", "debut": "2026-09-24T10:00:00+00:00",
        "traite": {"etat": "suspendu", "suspendu_le": "2026-09-24T12:00:00+00:00"},
        "temoin": {"etat": "en_attente"},
    }), encoding="utf-8")
    joues: list[str] = []
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({}, joues))
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"]])
    orchestrateur_memoire.main()
    assert joues == ["treated", "control"]
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["etat"] == "terminee"
    assert etat["debut"] == "2026-09-24T10:00:00+00:00"   # the trace of the first start remains


def test_un_bras_abouti_n_est_pas_rejoue(tmp_memoire_env, monkeypatch, ab_neutre):
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_ok_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    (exp_dir / "etat.json").write_text(json.dumps({
        "nom": cfg["nom"], "etat": "traite_ok",
        "traite": {"etat": "ok", "fin": "2026-09-24T12:00:00+00:00"}, "temoin": {"etat": "en_attente"},
    }), encoding="utf-8")
    joues: list[str] = []
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({}, joues))
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"], "--branche", "control"])
    orchestrateur_memoire.main()
    assert joues == ["control"]
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["etat"] == "terminee"   # control alone after a completed treated arm: the A/B is complete


def test_un_rejeu_incomplet_marque_la_comparaison_non_conforme(tmp_memoire_env, monkeypatch,
                                                               ab_neutre):
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_prefixe_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({}, []))
    monkeypatch.setattr(
        orchestrateur_memoire,
        "controler_rejeu",
        lambda nom, racine: {"payes_avant_evenement": 1},
    )
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"]])
    with pytest.raises(SystemExit) as fin:
        orchestrateur_memoire.main()
    # Not 0: the night chain wrote it as "✅ TERMINÉE" (2026-09-28).
    assert fin.value.code == orchestrateur_memoire.CODE_NON_CONFORME
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["traite"]["etat"] == etat["temoin"]["etat"] == "ok"
    assert etat["etat"] == "non_conforme"


def test_en_prefixe_commun_un_temoin_suspendu_se_reprend(tmp_memoire_env, monkeypatch):
    """2026-09-28: resumption is no longer refused outright; the strict replay will judge."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_prefrep_gem38f_899549_42j",
           "prefixe_commun": True}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    (exp_dir / "etat.json").write_text(json.dumps({
        "nom": cfg["nom"], "etat": "suspendue", "debut": "2026-09-27T21:00:00+00:00",
        "traite": {"etat": "ok", "fin": "2026-09-27T23:40:00+00:00"},
        "temoin": {"etat": "suspendu", "suspendu_le": "2026-09-28T01:00:00+00:00"},
    }), encoding="utf-8")
    vus: list[tuple[str, int | None]] = []

    def _executer(nom, branche, config, cible, dry_run=False, strict_avant_ts=None):
        vus.append((branche, strict_avant_ts))
        return 0

    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _executer)
    monkeypatch.setattr(orchestrateur_memoire, "borne_prefixe_commun", lambda racine: 1774597500)
    monkeypatch.setattr(
        orchestrateur_memoire, "mettre_de_cote_magasin",
        lambda nom: pytest.fail("a resumption does not set the store aside"),
    )
    monkeypatch.setattr(
        orchestrateur_memoire, "controler_rejeu",
        lambda nom, racine: {"payes_avant_evenement": 0, "prefixe_deplacements": {"conforme": True}},
    )
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"]])
    orchestrateur_memoire.main()
    assert vus == [("control", 1774597500)], "the resumed control keeps its strict bound"
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["etat"] == "terminee"


def test_rapprochement_compte_un_lecteur_par_foyer(tmp_path, monkeypatch):
    """a09 declares six households, one reader each: six readings expected, not one."""
    (tmp_path / "evenements.jsonl").write_text(
        "".join(json.dumps({"person_id": str(i)}) + "\n" for i in range(5)), encoding="utf-8"
    )
    r = orchestrateur_memoire.rapprochement_injections({"evenement": "a09_vent_autan"}, tmp_path)
    assert r["declarees"] == 6
    assert r["produites"] == 5
    assert r["conforme"] is False


# ── 7. Integration of the tab in app.py (D1) ───────────────────────────────────
def test_integration_onglet_app():
    slugs = [slug for slug, _ in APP.ONGLETS]
    assert "memoire" in slugs
    libelles = [libelle for _, libelle in APP.ONGLETS]
    assert any("Expériences Mémoire" in l for l in libelles)


def test_le_registre_montre_non_conforme_meme_si_les_deux_bras_ont_tourne(tmp_memoire_env):
    """2026-09-28: two moves.csv files made an invalid measurement display "Terminée (A+B)"."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_choc_nc_gem38f_899549_42j"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    for bras in ("traite", "temoin"):
        (exp_dir / bras).mkdir(parents=True, exist_ok=True)
        (exp_dir / bras / "moves.csv").write_text("x\n", encoding="utf-8")
    (exp_dir / "etat.json").write_text(json.dumps({
        "nom": cfg["nom"], "etat": "non_conforme",
        "traite": {"etat": "ok"}, "temoin": {"etat": "ok"},
    }), encoding="utf-8")
    assert memoire.lister_experiences()[0]["etat"] == "non_conforme"


def test_un_temoin_sans_borne_sure_n_est_pas_lance_et_reste_reprenable(
    tmp_memoire_env, monkeypatch, caplog
):
    """a13 v6, 2026-09-29: without `premiers_services.jsonl` in `traite/`, the control arm started
    on the injection bound and stopped three times on the same 409. It is now refused BEFORE it
    starts — code 2, which the night chain does not retry — and the experiment stays resumable."""
    cfg = {**memoire.defauts(), "nom": "exp_mem_presse_borne_gem31flite_899549_25j", "canal": "lu"}
    exp_dir, _ = memoire.enregistrer_experience(cfg)
    (exp_dir / "traite").mkdir(parents=True, exist_ok=True)
    (exp_dir / "traite" / "evenements.jsonl").write_text(
        json.dumps({"horodatage_simule": "2026-03-27T00:00:00"}) + "\n", encoding="utf-8"
    )
    (exp_dir / "etat.json").write_text(json.dumps({
        "nom": cfg["nom"], "etat": "suspendue",
        "traite": {"etat": "ok"}, "temoin": {"etat": "suspendu", "suspendu_le": "2026-09-29T20:11"},
    }), encoding="utf-8")
    joues: list[str] = []
    monkeypatch.setattr(orchestrateur_memoire, "executer_bras", _bras_simule({}, joues))
    monkeypatch.setattr(orchestrateur_memoire, "mettre_de_cote_magasin", lambda nom: None)
    monkeypatch.setattr(sys, "argv", ["orch", "--experience", cfg["nom"]])
    with caplog.at_level("ERROR", logger="orchestrateur_memoire"), pytest.raises(SystemExit) as fin:
        orchestrateur_memoire.main()
    assert fin.value.code == 2
    assert any("control arm NOT launched" in r.getMessage() for r in caplog.records), (
        "code 2 must come from the bound refusal, not from another check"
    )
    assert joues == [], "no arm is played on an uncertain bound"
    etat = json.loads((exp_dir / "etat.json").read_text(encoding="utf-8"))
    assert etat["etat"] == "suspendue", "neither « echec » nor « en_cours »: it resumes once repaired"
    assert etat["temoin"]["etat"] == "suspendu"
    assert etat["traite"]["etat"] == "ok"
