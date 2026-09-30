"""Stop what is running before launching (spec `arreter-avant-de-lancer.md`).

The thread of these tests: a launch must not start in competition with a survivor. A
run can outlive whatever launched it, its `docker compose exec` killed on the host not killing
the process inside the container. Stopping stays cooperative — a `STOP` file, honoured within
a few seconds — because a signal would leave the state at `en cours` forever.
"""

import functools
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402
from scripts.tests.test_dashboard_activites import FauxSt  # noqa: E402

VRAI_LISTER = experiences.lister


@dataclass
class FauxJob:
    """The bare minimum of `runner.Job` for these tests."""

    id: str
    label: str
    running: bool = True


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    texte = json.dumps(contenu) if chemin.suffix == ".json" else yaml.safe_dump(contenu, allow_unicode=True)
    chemin.write_text(texte, encoding="utf-8")


def _execution(dossier: Path, experience: str, nom: str, etat: str) -> Path:
    _ecrire(dossier / experience / "experience.yaml",
            {"nom": experience, "jeu": {"nom": "j"}, "decideur": {"type": "passerelle", "modele": "m"}})
    d = dossier / experience / "executions" / nom
    _ecrire(d / "etat.json", {"etat": etat})
    _ecrire(d / "execution.yaml", {"cree_le": "2026-09-07T10:00:00+00:00"})
    return d


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    """A running run, a finished one, a paused one, and a set under construction."""
    exps, jeux = tmp_path / "experiences", tmp_path / "jeux"
    _execution(exps, "autre_experience", "2026-09-07_10_00_00", "en_cours")
    _execution(exps, "finie", "2026-09-06_09_00_00", "terminee")
    _execution(exps, "suspendue", "2026-09-06_12_00_00", "en_pause")
    _ecrire(jeux / "jeu_en_construction" / "MANIFEST.yaml",
            {"nom": "jeu_en_construction", "clos": False, "population": {"nom": "pop"}, "jour_simule": "2026-03-16"})
    monkeypatch.setattr(experiences, "DOSSIER", exps)
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)
    monkeypatch.setattr(experiences, "lister", functools.partial(VRAI_LISTER, dossier=exps))
    return tmp_path


def test_R1_les_concurrents_sont_les_executions_en_cours_et_les_jobs_de_lancement(plateforme):
    jobs = [FauxJob("1", "root:experience-lancer"), FauxJob("2", "root:run-offline"),
            FauxJob("3", "root:up"), FauxJob("4", "root:experience-lancer", running=False)]
    conc = experiences.concurrents(lambda: jobs)

    assert [e["experience"] for e in conc["executions"]] == ["autre_experience"], \
        "a running run of ANOTHER experiment counts too"
    labels = [j.label for j in conc["jobs"]]
    assert labels == ["root:experience-lancer", "root:run-offline"], labels
    assert "root:up" not in labels, "a job of another kind is not a competitor"


def test_R2_la_construction_d_un_jeu_n_est_jamais_concurrente(plateforme):
    conc = experiences.concurrents(lambda: [FauxJob("1", "root:jeu")])
    assert conc["jobs"] == [], "building a set consumes no quota and its product is awaited"
    assert experiences.jeux_en_preparation(), "the set under construction does exist"

    arretes = experiences.arreter_concurrents(conc)
    assert all("jeu" not in a or "exécution" in a for a in arretes)
    assert not (experiences.DOSSIER_JEUX / "jeu_en_construction" / "STOP").exists()


def test_R3_la_case_annonce_ce_qu_elle_arretera(plateforme):
    conc = experiences.concurrents(lambda: [FauxJob("1", "root:run")])
    libelle = experiences.libelle_concurrents(conc)
    assert "autre_experience / 2026-09-07_10_00_00" in libelle
    assert "`root:run`" in libelle


def test_R4_l_arret_d_une_execution_est_cooperatif(plateforme):
    conc = experiences.concurrents()
    arretes = experiences.arreter_concurrents(conc)

    dossier = Path(conc["executions"][0]["dossier"])
    assert (dossier / "STOP").is_file(), "stopping goes through a STOP file, never a signal"
    assert not (dossier / "PAUSE").exists()
    assert any("effectif en quelques secondes" in a for a in arretes), arretes

    source = Path(experiences.__file__).read_text(encoding="utf-8")
    assert "pkill" not in source, "no process is killed inside the container (non-goal)"
    assert "SIGKILL" not in source


def test_R4_la_demande_est_visible_avant_d_etre_effective(plateforme):
    """The dropped file IS the immediate feedback: the bar says "pause demandée" without waiting
    for the runner to rewrite its state. Without it, a click on Pause showed nothing for
    the whole grace period, and the message promised "the next safe point" — up to 2 min.
    """
    dossier = Path(experiences.concurrents()["executions"][0]["dossier"])
    avant = experiences.activites_en_cours()["executions"]
    assert [e["pause_demandee"] for e in avant] == [None]

    experiences.signaler(dossier, "PAUSE")
    apres = experiences.activites_en_cours()["executions"]
    assert isinstance(apres[0]["pause_demandee"], float) and apres[0]["pause_demandee"] < 5
    assert apres[0]["arret_demande"] is None, "a pause is not a stop"


def test_R5_les_jobs_passent_par_le_registre(plateforme):
    demandes = []
    conc = experiences.concurrents(lambda: [FauxJob("job-7", "root:experience-lancer")])
    arretes = experiences.arreter_concurrents(conc, lambda ident: demandes.append(ident) or True)

    assert demandes == ["job-7"], "the registry must be called with the job identifier"
    assert any("root:experience-lancer" in a for a in arretes)


def test_R5_un_registre_qui_refuse_n_est_pas_annonce_comme_arrete(plateforme):
    conc = experiences.concurrents(lambda: [FauxJob("job-8", "root:run")])
    arretes = experiences.arreter_concurrents(conc, lambda _ident: False)
    assert not any("root:run" in a for a in arretes), "a stop that fails must not be announced"


def test_R6_l_attente_est_bornee_et_nomme_ce_qui_reste(plateforme):
    executions = experiences.concurrents()["executions"]

    debut = time.time()
    restantes = experiences.attendre_arret(executions, delai_s=1, pas_s=0.05)
    assert restantes == ["autre_experience / 2026-09-07_10_00_00"]
    assert time.time() - debut < 3, "the wait must be bounded"


def test_R6_une_execution_qui_s_arrete_pendant_l_attente_laisse_partir_le_lancement(plateforme):
    executions = experiences.concurrents()["executions"]
    etat = Path(executions[0]["dossier"]) / "etat.json"

    def arreter_bientot():
        time.sleep(0.2)
        etat.write_text(json.dumps({"etat": "arretee"}), encoding="utf-8")

    fil = threading.Thread(target=arreter_bientot)
    fil.start()
    try:
        assert experiences.attendre_arret(executions, delai_s=5, pas_s=0.05) == []
    finally:
        fil.join()


def test_R9_ni_les_terminees_ni_les_suspendues_ne_sont_touchees(plateforme):
    conc = experiences.concurrents()
    assert [e["experience"] for e in conc["executions"]] == ["autre_experience"]

    experiences.arreter_concurrents(conc)
    for experience, nom in (("finie", "2026-09-06_09_00_00"), ("suspendue", "2026-09-06_12_00_00")):
        dossier = experiences.DOSSIER / experience / "executions" / nom
        assert not (dossier / "STOP").exists(), f"{experience} was not to be touched"


def test_R8_sans_rien_en_cours_il_n_y_a_aucun_concurrent(tmp_path, monkeypatch):
    vide = tmp_path / "experiences"
    vide.mkdir()
    monkeypatch.setattr(experiences, "DOSSIER", vide)
    monkeypatch.setattr(experiences, "lister", functools.partial(VRAI_LISTER, dossier=vide))

    conc = experiences.concurrents(lambda: [FauxJob("1", "root:up")])
    assert conc == {"executions": [], "jobs": []}
    assert experiences.libelle_concurrents(conc) == ""
    assert experiences.arreter_concurrents(conc) == []


def _vieillir(dossier: Path, minutes: float) -> None:
    """Writes a dated progress file, to simulate a run that nothing feeds any more."""
    from datetime import datetime, timedelta, timezone

    quand = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    _ecrire(dossier / "progression.json", {"faits": 209, "attendus": 2693, "maj": quand.isoformat()})


def test_R11_une_execution_morte_n_est_pas_concurrente(plateforme):
    """The defect of 2026-09-07: a STOP written into a dead run made it non-resumable."""
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"

    _vieillir(dossier, 20)
    assert experiences.execution_vivante(dossier) is False
    assert experiences.concurrents()["executions"] == [], "a dead run competes with nothing"
    experiences.arreter_concurrents(experiences.concurrents())
    assert not (dossier / "STOP").exists(), "no STOP must be written into a dead run"

    _vieillir(dossier, 0.1)
    assert experiences.execution_vivante(dossier) is True
    assert len(experiences.concurrents()["executions"]) == 1, "the one still writing is a competitor"


def test_R11_une_execution_qui_vient_de_naitre_est_vivante(plateforme):
    """No progress yet: it is `etat.json` that dates the birth."""
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    assert not (dossier / "progression.json").exists()
    assert experiences.execution_vivante(dossier) is True

    import os
    vieux = time.time() - 3600
    os.utime(dossier / "etat.json", (vieux, vieux))
    assert experiences.execution_vivante(dossier) is False


def test_R12_les_trois_cas_reprenables(plateforme):
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    _vieillir(dossier, 20)
    par_nom = {l["experience"]: l for l in experiences.lister() if l.get("execution")}

    assert experiences.est_reprenable(par_nom["autre_experience"]) is True, "running, abandoned"
    assert experiences.est_reprenable(par_nom["suspendue"]) is True, "paused"
    assert experiences.est_reprenable(par_nom["finie"]) is False, "finished"

    _vieillir(dossier, 0.1)
    vivante = {l["experience"]: l for l in experiences.lister() if l.get("execution")}
    assert experiences.est_reprenable(vivante["autre_experience"]) is False, \
        "a run that is really running is not to be resumed, it is a competitor"

    for etat in ("epuisee", "arretee"):
        _ecrire(dossier / "etat.json", {"etat": etat})
        ligne = next(l for l in experiences.lister() if l["experience"] == "autre_experience")
        assert experiences.est_reprenable(ligne) is (etat == "epuisee"), etat


def test_R13_le_nombre_de_decisions_deja_archivees_est_lu(plateforme):
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    _ecrire(dossier / "etat.json", {"etat": "en_pause", "decisions_archivees": 209})
    assert experiences.decisions_archivees(dossier) == 209

    # Failing the counter, the lines of the decisions log are authoritative
    _ecrire(dossier / "etat.json", {"etat": "en_pause"})
    (dossier / "decisions.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    assert experiences.decisions_archivees(dossier) == 3
    assert experiences.decisions_archivees(experiences.DOSSIER / "inexistant") == 0


def test_R15_une_archive_cloturee_n_est_jamais_reprenable(plateforme):
    """Closure lives in `execution.yaml`, not in `etat.json`: a closed archive is
    immutable (E19) and resuming always fails on it. Offering it would be a loop — that is the
    trap met on 2026-09-07, where `etat.json` reset to `en pause` changed nothing."""
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    _vieillir(dossier, 20)

    ligne = next(l for l in experiences.lister() if l["experience"] == "autre_experience")
    assert experiences.archive_close(dossier) is False
    assert experiences.est_reprenable(ligne) is True

    _ecrire(dossier / "execution.yaml", {
        "cree_le": "2026-09-07T10:00:00+00:00",
        "cloture": {"le": "2026-09-07T12:54:46+00:00", "etat": "arretee",
                    "sha256": {"decisions.jsonl": "2b7b4c"}},
    })
    ligne = next(l for l in experiences.lister() if l["experience"] == "autre_experience")
    assert experiences.archive_close(dossier) is True
    assert experiences.est_reprenable(ligne) is False, "a closed archive is not resumable"

    for etat in ("en_pause", "epuisee"):
        _ecrire(dossier / "etat.json", {"etat": etat})
        ligne = next(l for l in experiences.lister() if l["experience"] == "autre_experience")
        assert experiences.est_reprenable(ligne) is False, \
            f"{etat} on a closed archive stays non-resumable"


def test_supprimer_efface_le_dossier_d_une_experience_inactive(plateforme):
    """`finie` has no live run: deleting erases its whole folder."""
    dossier = experiences.DOSSIER / "finie"
    assert dossier.is_dir()
    rendu = experiences.supprimer_experience("finie")
    assert rendu == dossier
    assert not dossier.exists(), "the definition and the archived runs are erased"
    assert not any(l["experience"] == "finie" for l in experiences.lister())


def test_supprimer_refuse_une_experience_encore_active(plateforme):
    """A run still alive forbids deletion: nothing is destroyed from under its feet."""
    with pytest.raises(ValueError, match="still active"):
        experiences.supprimer_experience("autre_experience")
    assert (experiences.DOSSIER / "autre_experience").is_dir(), "nothing was deleted"


def test_supprimer_un_nom_inconnu_leve_une_erreur(plateforme):
    with pytest.raises(ValueError, match="no experiment of this name"):
        experiences.supprimer_experience("nom_qui_n_existe_pas")


# ── removing a row from the table without erasing anything ───────────────────
# The registry 🗑 erased `data/experiences/<nom>/` in two clicks; an experiment and its
# already-paid decisions thus vanished beyond recovery on 2026-09-07. It now removes
# only the clicked row, and the disk is not touched.


def test_masquer_retire_la_ligne_du_tableau_et_laisse_le_disque_intact(plateforme):
    dossier = experiences.DOSSIER / "finie" / "executions" / "2026-09-06_09_00_00"
    assert any(l["experience"] == "finie" for l in experiences.lister())

    experiences.masquer("finie", "2026-09-06_09_00_00")

    assert not any(l["experience"] == "finie" for l in experiences.lister()), "the row has left the table"
    assert dossier.is_dir(), "the run archive is intact"
    assert (experiences.DOSSIER / "finie" / "experience.yaml").is_file(), "the definition is intact"


def test_masquer_ne_retire_que_la_ligne_cliquee(plateforme):
    """Scope: one row. The other runs of the same experiment stay in the table."""
    _execution(experiences.DOSSIER, "finie", "2026-09-06_18_00_00", "terminee")
    assert len([l for l in experiences.lister() if l["experience"] == "finie"]) == 2

    experiences.masquer("finie", "2026-09-06_09_00_00")

    restantes = [l["execution"] for l in experiences.lister() if l["experience"] == "finie"]
    assert restantes == ["2026-09-06_18_00_00"], restantes


def test_masquer_refuse_une_execution_encore_vivante(plateforme):
    """Removed from the table, a run still writing would have no one left to stop it."""
    with pytest.raises(ValueError, match="still running"):
        experiences.masquer("autre_experience", "2026-09-07_10_00_00")
    assert any(l["experience"] == "autre_experience" for l in experiences.lister())


def test_masquer_deux_fois_la_meme_ligne_n_ecrit_qu_une_entree(plateforme):
    experiences.masquer("finie", "2026-09-06_09_00_00")
    experiences.masquer("finie", "2026-09-06_09_00_00")
    assert len(experiences.masques()) == 1


def test_demasquer_tout_rend_les_lignes_au_tableau(plateforme):
    experiences.masquer("finie", "2026-09-06_09_00_00")
    experiences.masquer("suspendue", "2026-09-06_12_00_00")

    assert experiences.demasquer_tout() == 2
    assert experiences.masques() == []
    rendues = {l["experience"] for l in experiences.lister()}
    assert {"finie", "suspendue"} <= rendues


def test_le_fichier_des_masques_ne_devient_jamais_une_experience(plateforme):
    """`.masques.json` lives in the experiments folder: it must not become one of them."""
    experiences.masquer("finie", "2026-09-06_09_00_00")
    assert (experiences.DOSSIER / ".masques.json").is_file()
    assert "masques" not in {l["experience"] for l in experiences.lister()}
    assert not (experiences.DOSSIER / ".masques.json.tmp").exists(), "the write is atomic"


def test_une_selection_perimee_ne_fait_plus_tomber_la_page():
    """Streamlit keeps the selection by INDEX: it survives the table shrinking.

    Without a bound, `df.iloc[3]` on a table back down to 2 rows raised "single positional
    indexer is out-of-bounds" and took the whole page down (2026-09-07, after a removal).
    """
    event = {"selection": {"rows": [3]}}
    assert experiences._lignes_selectionnees(event) == [3], "unbounded, the index is returned as is"
    assert experiences._lignes_selectionnees(event, 2) == [], "bounded: the stale index is dropped"
    assert experiences._lignes_selectionnees(event, 4) == [3], "bounded: a valid index passes"
    assert experiences._lignes_selectionnees({"selection": {"rows": []}}, 2) == []


def test_une_ligne_retiree_qui_se_remet_a_tourner_revient_au_tableau(plateforme):
    """Relaunched from a console, a removed run would write without anyone seeing it."""
    experiences.masquer("finie", "2026-09-06_09_00_00")
    assert not any(l["experience"] == "finie" for l in experiences.lister())

    _ecrire(experiences.DOSSIER / "finie" / "executions" / "2026-09-06_09_00_00" / "etat.json",
            {"etat": "en_cours"})

    ligne = next((l for l in experiences.lister() if l["experience"] == "finie"), None)
    assert ligne is not None, "what is running stays visible, even removed from the table"
    assert experiences.masques(), "the removal is kept: it resumes once the run has gone quiet"


# ── ⏸/⏹ are only offered on a run that is still writing ──────────────────────
# `etat.json` says `en cours` forever when the runner was killed. The two buttons
# were therefore still offered on a corpse, and the dropped sentinel waited for the next
# resume to sabotage it: PAUSE put it back on pause at once, STOP sealed the archive.


def _ligne_arret(dossier):
    """The minimal shape `_boutons_arret` expects from a run row."""
    return {"experience": "autre_experience", "execution": "2026-09-07_10_00_00",
            "dossier": str(dossier), "pause_demandee": None, "arret_demande": None}


def test_R11_pause_et_arret_disparaissent_sur_une_execution_morte(plateforme):
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    _vieillir(dossier, 20)
    assert experiences.execution_vivante(dossier) is False

    st = FauxSt()
    experiences._boutons_arret(st, _ligne_arret(dossier), 0)

    assert st.boutons == [], f"no interruption button on a killed runner: {st.boutons}"
    assert st.cases == [], "nor the confirmation box of the final stop"
    assert any("Reprendre" in l for l in st.legendes),         f"the page must name the clean path: {st.legendes}"


def test_R11_pause_et_arret_restent_offerts_sur_une_execution_vivante(plateforme):
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    _vieillir(dossier, 0.1)
    assert experiences.execution_vivante(dossier) is True

    st = FauxSt()
    experiences._boutons_arret(st, _ligne_arret(dossier), 0)

    assert any("Pause" in b for b in st.boutons), st.boutons
    assert any("Arrêter" in b for b in st.boutons), st.boutons


def test_R11_aucune_sentinelle_n_est_deposee_par_le_rendu(plateforme):
    """Rendering alone writes nothing: only clicks do (the fake buttons return False)."""
    dossier = experiences.DOSSIER / "autre_experience" / "executions" / "2026-09-07_10_00_00"
    _vieillir(dossier, 20)
    experiences._boutons_arret(FauxSt(), _ligne_arret(dossier), 0)
    assert not (dossier / "PAUSE").exists() and not (dossier / "STOP").exists()
