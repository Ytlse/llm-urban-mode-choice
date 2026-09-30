"""The cohort orchestrator can play an event OTHER than c6.

Until now `CHOC_REFERENCE = "c6_voiture_suspecte"` was hard-coded: the cohort could only play
the suspicious-car case study. The attribution campaign of `c3_panne_reseau` therefore required
either editing the script or giving up.

The trickiest point is not the parameter but the EXPOSURE RULE: the per-persona
derivation wrote `exposition.agents` without touching `regle`. On c6, already `regle: agents`,
that worked. On c3 (`regle: mode`) the list would have been written, accepted and ignored — and on
a one-inhabitant population the result would have been right BY ACCIDENT. That is what these tests
close off.
"""

import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE / "scripts" / "experiment"))

import run_sequential_cohort as cohorte  # noqa: E402


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Redirects the two declaration directories: no test writes into the repository."""
    evenements, chocs = tmp_path / "evenements", tmp_path / "chocs"
    evenements.mkdir()
    chocs.mkdir()
    monkeypatch.setattr(cohorte, "EVENEMENTS_DIR", evenements)
    monkeypatch.setattr(cohorte, "CHOCS_DIR", chocs)
    return evenements, chocs


def _ecrire(dossier: Path, nom: str, exposition: dict) -> None:
    (dossier / f"{nom}.yaml").write_text(
        yaml.safe_dump(
            {"evenement": nom, "canal": "vecu", "moment": "arrivee", "jugement": "a_l_injection",
             "libelle": "T", "source": "test", "exposition": exposition,
             "jours": [{"jour": 12, "retard_min": 28, "vecu": "Something happened."}]},
            allow_unicode=True, sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_la_regle_mode_devient_agents_et_les_modes_survivent(dirs):
    """The c3 case: `regle: mode` + `modes` → `regle: agents` + `agents` + the SAME modes.

    Keeping the modes is not cosmetic: the `agents` branch reads them and restricts the designated
    agent to those trips. Without them, 861500 would read "the metro stopped" at the wheel of their
    car — exactly the defect c6 had paid for on Corinne.
    """
    evenements, _ = dirs
    _ecrire(evenements, "c3_panne_reseau", {"regle": "mode", "modes": ["public_transport"]})

    nom = cohorte.choc_pour_persona("861500", "c3_panne_reseau")

    assert nom == "c3_panne_reseau__861500"
    produit = yaml.safe_load((evenements / f"{nom}.yaml").read_text(encoding="utf-8"))
    assert produit["exposition"]["regle"] == "agents"
    assert produit["exposition"]["agents"] == ["861500"]
    assert produit["exposition"]["modes"] == ["public_transport"]


def test_les_parametres_du_tirage_ne_survivent_pas(dirs):
    """The c2 case: `part` and `graine` make no sense under `regle: agents`.

    A parameter accepted with no effect is worse than a refused parameter: nothing flags it.
    """
    evenements, _ = dirs
    _ecrire(evenements, "c2_crevaison", {"regle": "tirage", "part": 0.30, "graine": 79})

    nom = cohorte.choc_pour_persona("861500", "c2_crevaison")

    expo = yaml.safe_load((evenements / f"{nom}.yaml").read_text(encoding="utf-8"))["exposition"]
    assert expo == {"regle": "agents", "agents": ["861500"]}


def test_le_defaut_reste_c6_et_le_comportement_d_hier_ne_bouge_pas(dirs):
    """Without `--evenement`, the historical command must give the same run as before."""
    evenements, _ = dirs
    _ecrire(evenements, "c6_voiture_suspecte", {"regle": "agents", "modes": ["car"]})

    assert cohorte.CHOC_REFERENCE == "c6_voiture_suspecte"
    assert cohorte.nom_choc_derive("899549") == "c6_voiture_suspecte__899549"
    assert cohorte.choc_pour_persona("899549") == "c6_voiture_suspecte__899549"


def test_le_repertoire_herite_est_lu_en_second(dirs):
    """The five historical cases still live in config/chocs/: they remain playable."""
    evenements, chocs = dirs
    _ecrire(chocs, "c4_train_supprime", {"regle": "mode", "modes": ["train"]})

    assert cohorte.source_evenement("c4_train_supprime") == chocs / "c4_train_supprime.yaml"
    # …but the derived one goes to the canonical directory, the one the Makefile prefers.
    cohorte.choc_pour_persona("861500", "c4_train_supprime")
    assert (evenements / "c4_train_supprime__861500.yaml").is_file()
    assert not (chocs / "c4_train_supprime__861500.yaml").exists()


def test_le_canonique_prime_sur_l_herite(dirs):
    """The same name on both sides: config/evenements/ decides."""
    evenements, chocs = dirs
    _ecrire(evenements, "c3_panne_reseau", {"regle": "mode", "modes": ["public_transport"]})
    _ecrire(chocs, "c3_panne_reseau", {"regle": "mode", "modes": ["train"]})

    assert cohorte.source_evenement("c3_panne_reseau") == evenements / "c3_panne_reseau.yaml"


def test_un_evenement_inconnu_dit_ce_qui_existe(dirs):
    """Six hours of run are not lost to a typo: the error lists the cases."""
    evenements, _ = dirs
    _ecrire(evenements, "c3_panne_reseau", {"regle": "mode", "modes": ["public_transport"]})

    with pytest.raises(FileNotFoundError) as exc:
        cohorte.source_evenement("c3_panne_resau")

    assert "c3_panne_reseau" in str(exc.value), "the error must show the correct name"


# ══ The progress heartbeat (2026-09-22, during the c3 campaign) ══════════════════════════════


def test_le_battement_distingue_l_amorcage_d_un_blocage(tmp_path):
    """`?` reads as a failure. Over six hours of run, it is the only question that matters.

    `[sync] END` only arrives once the loop has started; bootstrapping — populating, initial
    itineraries, all the longer when the OSMnx cache is cold — produces none.
    """
    journal = tmp_path / "app.log"

    journal.write_text(
        "INFO | simulation_controller - [bootstrap] pre-computing 1 act[N+3] itineraries (wave 3)...\n",
        encoding="utf-8",
    )
    assert cohorte.derniere_journee_simulee(tmp_path).startswith("amorçage — pre-computing")

    # As soon as a cycle has completed, the simulated day takes precedence over bootstrapping.
    with open(journal, "a", encoding="utf-8") as f:
        f.write("INFO | [sync] END sim_time=20 March 2026, 05:00 state_update_duration=0.003s\n")
    assert cohorte.derniere_journee_simulee(tmp_path) == "20 March 2026, 05:00"


def test_le_battement_nomme_l_absence_de_journal(tmp_path):
    """A directory without app.log is not a stuck run: it is a run that has not started."""
    assert cohorte.derniere_journee_simulee(tmp_path) == "journal pas encore écrit"
    assert cohorte.derniere_journee_simulee(None) == "?"


# ══ Who produced this arm (2026-09-22) ═══════════════════════════════════════════════════════


@pytest.fixture
def journal():
    """Captures loguru output — pytest's `caplog` does not see it, loguru not being logging."""
    from loguru import logger

    lignes: list[str] = []
    jeton = logger.add(lignes.append, level="DEBUG", format="{level} {message}")
    yield lignes
    logger.remove(jeton)


def test_le_lanceur_dit_quels_modeles_servent_le_bras(tmp_path, journal):
    """An arm whose producer is unknown compares to nothing.

    The instance restriction is declared per category in docker-compose and appeared
    nowhere in the launcher output. On 2026-09-22 it led to a false announcement — "the
    full pool" — while the run was working on two keys of a single model.
    """
    import json as _json

    (tmp_path / "identite_run.json").write_text(
        _json.dumps({
            "modeles_admis": ["gemini-3.1-flash-lite"],
            "routage_instances": {"itinary_multi_agent": ["google_gemini31_key1"]},
        }),
        encoding="utf-8",
    )
    cohorte.journaliser_qui_sert("861500", "treated", tmp_path)

    sortie = "".join(journal)
    assert "gemini-3.1-flash-lite" in sortie
    assert "itinary_multi_agent" in sortie and "google_gemini31_key1" in sortie


def test_une_identite_illisible_se_signale_au_lieu_de_se_taire(tmp_path, journal):
    """Silence reads as "no restriction", which is the opposite of the dangerous case."""
    cohorte.journaliser_qui_sert("861500", "treated", tmp_path)
    assert "identite_run.json absent" in "".join(journal)

    journal.clear()
    (tmp_path / "identite_run.json").write_text("{ pas du json", encoding="utf-8")
    cohorte.journaliser_qui_sert("861500", "treated", tmp_path)
    assert "illisible" in "".join(journal)


# ══ Sealed population resolution and failure detection ═════════════════════════════════════


def test_resoudre_population_unitaire_vs_scellee():
    """A persona 861500 points to population_1_861500 (n=1), a 20-household population points to its directory (n=20)."""
    pop_path, pop_size, is_dataset, ids = cohorte.resoudre_population_et_taille("861500")
    assert pop_path == "/data/eqasim-output/population_1_861500/population.json"
    assert pop_size == 1
    assert is_dataset is False
    assert ids == ["861500"]

    pop_path_20, pop_size_20, is_dataset_20, ids_20 = cohorte.resoudre_population_et_taille("population_20_foyers_059")
    assert pop_path_20 == "/data/eqasim-output/population_20_foyers_059/population.json"
    assert pop_size_20 == 20
    assert is_dataset_20 is True
    assert len(ids_20) == 20


def test_choc_conserve_regle_foyers_pour_population_scellee(dirs):
    """For a sealed population with a 'foyers'-rule event, the 'foyers' rule is not overwritten with 'agents'."""
    evenements, _ = dirs
    _ecrire(
        evenements,
        "a07_greve_eboueurs",
        {
            "regle": "foyers",
            "foyers": ["605813", "234839"],
            "lecteurs_par_foyer": 1,
            "graine": 59,
        },
    )

    nom = cohorte.choc_pour_persona("population_20_foyers_059", "a07_greve_eboueurs")
    assert nom == "a07_greve_eboueurs__population_20_foyers_059"

    produit = yaml.safe_load((evenements / f"{nom}.yaml").read_text(encoding="utf-8"))
    assert produit["exposition"]["regle"] == "foyers"
    assert produit["exposition"]["foyers"] == ["605813", "234839"]
    assert "agents" not in produit["exposition"]


def test_le_battement_detecte_echec_initialisation(tmp_path):
    """If app.log contains a sealed-population-not-found alarm, the heartbeat does not stay on 'amorçage'."""
    journal = tmp_path / "app.log"
    journal.write_text(
        "INFO | handle.application - INITIALISATION 1/5 Préparation de la population — sim_time=16 March 2026, 05:00\n"
        "ERROR | handle.application - [ALARME] [population] Population scellée introuvable : /data/eqasim-output/foo.json\n",
        encoding="utf-8",
    )
    resultat = cohorte.derniere_journee_simulee(tmp_path)
    assert "échec initialisation" in resultat



# ── Sealed population: resolution depends neither on the current directory nor on luck ──


def test_la_resolution_ne_depend_pas_du_repertoire_courant(tmp_path, monkeypatch):
    """Launched from another directory, the first version no longer found the set, fell back on
    the single-inhabitant case, and the arm exposed an "agent" named like the population: nobody."""
    monkeypatch.chdir(tmp_path)
    _, taille, est_un_jeu, ids = cohorte.resoudre_population_et_taille("population_20_foyers_059")
    assert (taille, est_un_jeu, len(ids)) == (20, True, 20)


def test_une_population_nommee_introuvable_est_refusee(tmp_path, monkeypatch):
    """A non-numeric name without a directory has no fallback: there is nobody to expose there."""
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", tmp_path)
    with pytest.raises(cohorte.PopulationIntrouvable):
        cohorte.resoudre_population_et_taille("population_20_foyers_inexistante")


def test_un_identifiant_numerique_sans_dossier_garde_le_repli_unitaire(tmp_path, monkeypatch):
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", tmp_path)
    chemin, taille, est_un_jeu, ids = cohorte.resoudre_population_et_taille("123456")
    assert chemin == "/data/eqasim-output/population_1_123456/population.json"
    assert (taille, est_un_jeu, ids) == (1, False, ["123456"])


def test_un_fichier_de_population_illisible_est_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", tmp_path)
    (tmp_path / "population_casse").mkdir()
    (tmp_path / "population_casse" / "population.json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(cohorte.PopulationIntrouvable):
        cohorte.resoudre_population_et_taille("population_casse")


def test_sur_un_jeu_la_regle_mode_est_refusee(dirs):
    """Rewritten as `agents`, it would designate the population name as an inhabitant."""
    evenements, _ = dirs
    _ecrire(evenements, "c3_panne_reseau", {"regle": "mode", "modes": ["public_transport"]})
    with pytest.raises(ValueError, match="regle: mode"):
        cohorte.choc_pour_persona("population_20_foyers_059", "c3_panne_reseau")


def test_sur_un_jeu_des_agents_tous_absents_sont_refuses(dirs):
    evenements, _ = dirs
    _ecrire(evenements, "c6_voiture_suspecte", {"regle": "agents", "agents": ["999999999"]})
    with pytest.raises(ValueError, match="AUCUN"):
        cohorte.choc_pour_persona("population_20_foyers_059", "c6_voiture_suspecte")


def test_le_delai_de_garde_suit_le_volume():
    """Six hours for one inhabitant; twenty inhabitants over fifteen days require more."""
    assert cohorte.delai_de_garde(1, 42) == cohorte.ATTENTE_RUN_S
    assert cohorte.delai_de_garde(20, 15) == 20 * 15 * cohorte.DELAI_PAR_AGENT_JOUR_S
    assert cohorte.delai_de_garde(20, 15) > cohorte.ATTENTE_RUN_S


@pytest.mark.parametrize("brut, attendu", [("true", True), ("1", True), ("false", False)])
def test_le_partage_du_foyer_est_verifie_dans_l_identite(brut, attendu):
    """Set by the orchestrator, it must show up in `identite_run.json` — otherwise the
    container did not receive it and the arm stops instead of running without sharing."""
    attendus = cohorte.reglages_attendus({"MEMOIRE__PARTAGE_FOYER_ENABLED": brut})
    assert attendus["partage_foyer"] is attendu


# ── Option A (2026-09-24): an arm suspended by the safeguard is resumed by its name ─────────
def test_un_marqueur_anterieur_au_bras_ne_le_dit_pas_suspendu(tmp_path):
    """A resumed arm replays in the directory that had stopped: the old marker is there."""
    import json as _json
    import os
    import time as _time

    marqueur = tmp_path / "en_attente_quota.json"
    marqueur.write_text(_json.dumps({"motif": "quota_journalier"}), encoding="utf-8")
    vieux = _time.time() - 3600
    os.utime(marqueur, (vieux, vieux))
    assert cohorte.suspension_du_run(tmp_path, depuis=_time.time()) is None
    assert cohorte.suspension_du_run(tmp_path, depuis=vieux)["motif"] == "quota_journalier"
    assert cohorte.suspension_du_run(None) is None


def test_reprise_en_attente_lit_le_nom_du_run(tmp_path):
    assert cohorte.reprise_en_attente(tmp_path) is None
    (tmp_path / cohorte.FICHIER_REPRISE).write_text('{"archive": "2026-09-24_11_14"}', encoding="utf-8")
    assert cohorte.reprise_en_attente(tmp_path)["archive"] == "2026-09-24_11_14"
    (tmp_path / cohorte.FICHIER_REPRISE).write_text("pas du json", encoding="utf-8")
    assert cohorte.reprise_en_attente(tmp_path) is None


def test_attendre_identite_en_reprise_accepte_l_identite_du_premier_depart(tmp_path, monkeypatch):
    """The resumed run's identity PREDATES the controller — the name is authoritative."""
    import os
    import time as _time

    archive = tmp_path / "2026-09-24_11_14"
    archive.mkdir()
    ident = archive / "identite_run.json"
    ident.write_text("{}", encoding="utf-8")
    vieux = _time.time() - 7200
    os.utime(ident, (vieux, vieux))
    monkeypatch.setattr(cohorte, "archive_du_run", lambda: archive)
    monkeypatch.setattr(cohorte.time, "sleep", lambda s: None)
    assert cohorte.attendre_identite(_time.time(), timeout_s=1, reprise="2026-09-24_11_14") == archive
    assert cohorte.attendre_identite(_time.time(), timeout_s=0.05) is None
    assert cohorte.attendre_identite(_time.time(), timeout_s=0.05, reprise="autre_run") is None


# ── 2026-09-24: a household population carries its exposed households in its MANIFEST ──────
def _population_de_foyers(racine: Path, nom: str, foyers: list[str], exposes: list[str] | None):
    import json as _json

    d = racine / nom
    d.mkdir(parents=True)
    agents = [
        {"person_id": f"{f}{i}", "household": {"id": f}} for f in foyers for i in range(2)
    ]
    (d / "population.json").write_text(_json.dumps(agents), encoding="utf-8")
    if exposes is not None:
        (d / "MANIFEST.yaml").write_text(
            yaml.safe_dump({"groupes": {"expose": [{"household_id": f} for f in exposes]}}),
            encoding="utf-8",
        )


def test_des_foyers_declares_absents_se_prennent_au_manifeste(dirs, tmp_path, monkeypatch):
    """a09 declares the households of population_20; on another population, nobody would read."""
    evenements, _ = dirs
    pops = tmp_path / "population"
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", pops)
    _population_de_foyers(pops, "population_4_foyer_x", ["133048"], ["133048"])
    _ecrire(evenements, "a09_t", {"regle": "foyers", "foyers": ["605813"], "lecteurs_par_foyer": 1})

    nom = cohorte.choc_pour_persona("population_4_foyer_x", "a09_t")
    derive = yaml.safe_load((evenements / f"{nom}.yaml").read_text(encoding="utf-8"))
    assert derive["exposition"]["foyers"] == ["133048"]
    assert derive["exposition"]["lecteurs_par_foyer"] == 1


def test_des_foyers_declares_presents_restent_ceux_de_la_declaration(dirs, tmp_path, monkeypatch):
    evenements, _ = dirs
    pops = tmp_path / "population"
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", pops)
    _population_de_foyers(pops, "population_4_foyer_y", ["1", "2"], ["2"])
    _ecrire(evenements, "a09_t", {"regle": "foyers", "foyers": ["1"], "lecteurs_par_foyer": 1})
    nom = cohorte.choc_pour_persona("population_4_foyer_y", "a09_t")
    derive = yaml.safe_load((evenements / f"{nom}.yaml").read_text(encoding="utf-8"))
    assert derive["exposition"]["foyers"] == ["1"]


def test_sans_foyer_present_ni_manifeste_le_bras_est_refuse(dirs, tmp_path, monkeypatch):
    evenements, _ = dirs
    pops = tmp_path / "population"
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", pops)
    _population_de_foyers(pops, "population_4_foyer_z", ["9"], None)
    _ecrire(evenements, "a09_t", {"regle": "foyers", "foyers": ["605813"]})
    with pytest.raises(ValueError, match="sans lecteur"):
        cohorte.choc_pour_persona("population_4_foyer_z", "a09_t")


# ── 2026-09-25: the MANIFEST also designates the readers ────────────────────────────────────
def test_les_lecteurs_du_manifeste_passent_dans_l_evenement_joue(dirs, tmp_path, monkeypatch):
    evenements, _ = dirs
    pops = tmp_path / "population"
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", pops)
    _population_de_foyers(pops, "population_4_foyers_l", ["7", "8"], ["7", "8"])
    manifeste = pops / "population_4_foyers_l" / "MANIFEST.yaml"
    manifeste.write_text(yaml.safe_dump({"groupes": {"expose": [
        {"household_id": "7", "lecteurs": ["71"]}, {"household_id": "8", "lecteurs": ["80"]},
    ]}}), encoding="utf-8")
    _ecrire(evenements, "a13_t", {"regle": "foyers", "foyers": ["605813"], "lecteurs_par_foyer": 1})
    nom = cohorte.choc_pour_persona("population_4_foyers_l", "a13_t")
    derive = yaml.safe_load((evenements / f"{nom}.yaml").read_text(encoding="utf-8"))
    assert derive["exposition"]["foyers"] == ["7", "8"]
    assert derive["exposition"]["lecteurs"] == ["71", "80"]


def test_des_lecteurs_declares_d_une_autre_population_sont_refuses(dirs, tmp_path, monkeypatch):
    evenements, _ = dirs
    pops = tmp_path / "population"
    monkeypatch.setattr(cohorte, "POPULATIONS_DIR", pops)
    _population_de_foyers(pops, "population_2_foyer_m", ["7"], ["7"])
    _ecrire(evenements, "a13_t", {"regle": "foyers", "foyers": ["7"], "lecteurs": ["1320713"]})
    with pytest.raises(ValueError, match="sans lecteur"):
        cohorte.choc_pour_persona("population_2_foyer_m", "a13_t")
