"""Experiment parallelisation: keys, reservation, queue.

One test per rule, named by its number. No container, no network: the reservation
registry and the queue live in a temporary `EXPERIENCES_DIR`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from experiences import reservations as R
from experiences.archive import ETAT_EN_COURS, ETAT_INTERROMPUE, Execution
from experiences.cles import identite_cle, jeu_de_cles

PROVIDERS = {
    "mistral": {"default_model": "mistral-small-latest"},
    "google_35": {"adapter": "google", "default_model": "gemini-3.5-flash"},
    "google_36": {"adapter": "google", "default_model": "gemini-3.6-flash"},
    "cerebras_x": {"adapter": "cerebras", "default_model": "gpt-oss-120b"},
    "sans_cle": {"adapter": "orphelin", "default_model": "modele-sans-cle"},
}


@pytest.fixture
def experiences_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    return tmp_path


def _execution(experiences_dir, nom_exp="Exp"):
    """Creates a minimal but valid run archive (execution.yaml + etat.json)."""
    dossier_exp = experiences_dir / nom_exp
    ex = Execution.creer(
        dossier_exp,
        experience={"nom": nom_exp},
        empreintes={},
        regime_demande={},
        sources_alea={},
    )
    return ex


# ── R4: derivation model → key identity ───────────────────────────────────────


def test_R4_meme_fournisseur_meme_cle():
    assert jeu_de_cles("gemini-3.5-flash", PROVIDERS) == {"google"}
    assert jeu_de_cles("gemini-3.6-flash", PROVIDERS) == {"google"}
    # Two DIFFERENT models sharing the `google` adapter → keys NOT disjoint (conflict).
    assert jeu_de_cles("gemini-3.5-flash", PROVIDERS) & jeu_de_cles(
        "gemini-3.6-flash", PROVIDERS
    )


def test_R4_fournisseurs_distincts_cles_disjointes():
    assert jeu_de_cles("mistral-small-latest", PROVIDERS) == {"mistral"}
    assert jeu_de_cles("gpt-oss-120b", PROVIDERS) == {"cerebras"}
    assert not (
        jeu_de_cles("mistral-small-latest", PROVIDERS)
        & jeu_de_cles("gpt-oss-120b", PROVIDERS)
    )


def test_R4_identite_par_adapter_sinon_nom():
    assert identite_cle("google_35", PROVIDERS["google_35"]) == "google"
    assert identite_cle("mistral", PROVIDERS["mistral"]) == "mistral"


# ── R8 / R9: empty key set ────────────────────────────────────────────────────


def test_R8_modele_sans_instance_jeu_vide():
    assert jeu_de_cles("modele-inexistant", PROVIDERS) == set()


def test_R9_instances_indisponibles_exclues():
    # No admitted instance (all exhausted / without key) → empty set.
    assert jeu_de_cles("mistral-small-latest", PROVIDERS, instances_admises=[]) == set()


# ── R1 / R3: disjoint keys → parallel ─────────────────────────────────────────


def test_R1_R3_cles_disjointes_parallele(experiences_dir):
    a = _execution(experiences_dir, "A")
    b = _execution(experiences_dir, "B")
    assert R.reserver({"mistral"}, a.dossier, "A") is True
    assert R.reserver({"cerebras"}, b.dossier, "B") is True
    assert R.cles_reservees() == {"mistral", "cerebras"}
    assert {e["exp"] for e in R.actives()} == {"A", "B"}


# ── R2 / R6: shared key → queue, atomic admission ─────────────────────────────


def test_R2_cle_commune_met_en_file(experiences_dir):
    a = _execution(experiences_dir, "A")
    b = _execution(experiences_dir, "B")
    assert R.admettre({"google"}, a.dossier, "A") == "lance"
    assert R.admettre({"google"}, b.dossier, "B") == "file"
    assert [e["exp"] for e in R.lister_file()] == ["B"]
    assert R.reserver({"google"}, b.dossier, "B") is False


def test_R6_admission_atomique_concurrente(experiences_dir):
    execs = [_execution(experiences_dir, f"E{i}") for i in range(12)]

    def tenter(ex):
        return R.reserver({"google"}, ex.dossier, ex.dossier.name)

    with ThreadPoolExecutor(max_workers=12) as pool:
        gagnants = list(pool.map(tenter, execs))
    assert gagnants.count(True) == 1  # exactly one reserves the shared key


def test_R2b_promotion_fifo_a_la_liberation(experiences_dir):
    a = _execution(experiences_dir, "A")
    b = _execution(experiences_dir, "B")
    c = _execution(experiences_dir, "C")
    assert R.admettre({"google"}, a.dossier, "A") == "lance"
    assert R.admettre({"google"}, b.dossier, "B") == "file"
    assert R.admettre({"google"}, c.dossier, "C") == "file"
    # Nothing is ready while A holds the key.
    assert R.promouvoir_pretes() == []
    R.liberer(a.dossier)
    promues = R.promouvoir_pretes()
    assert [e["exp"] for e in promues] == ["B"]  # FIFO: B before C
    assert [e["exp"] for e in R.lister_file()] == ["C"]


def test_R2e_retrait_de_la_file(experiences_dir):
    a = _execution(experiences_dir, "A")
    b = _execution(experiences_dir, "B")
    R.admettre({"google"}, a.dossier, "A")
    R.admettre({"google"}, b.dossier, "B")
    assert R.retirer_file("B") is True
    assert R.lister_file() == []
    R.liberer(a.dossier)
    assert R.promouvoir_pretes() == []  # B removed: never promoted


# ── R7: reconciliation of a ghost (dead pid) ──────────────────────────────────


def test_R7_pid_mort_reconcilie_et_libere(experiences_dir):
    a = _execution(experiences_dir, "A")
    a.changer_etat(ETAT_EN_COURS)
    pid_mort = 2**31 - 1  # certainly absent
    assert R.reserver({"google"}, a.dossier, "A", pid=pid_mort) is True
    liberees = R.reconcilier()
    assert liberees == ["google"]
    assert R.cles_reservees() == set()
    assert Execution.ouvrir(a.dossier).etat()["etat"] == ETAT_INTERROMPUE


# ── R8: empty key set always admitted ─────────────────────────────────────────


def test_R8_jeu_vide_toujours_lance(experiences_dir):
    a = _execution(experiences_dir, "A")
    b = _execution(experiences_dir, "B")
    R.reserver({"google"}, a.dossier, "A")
    assert (
        R.admettre(set(), b.dossier, "B") == "lance"
    )  # no key → never in conflict


# ── R5: stopping services conditioned on no active experiment ──────────────────


def test_R5_actives_est_vide_code_retour(experiences_dir):
    import argparse

    from experiences.cli import cmd_actives

    # Nothing active → code 0 (the shared services can be stopped).
    assert cmd_actives(argparse.Namespace(est_vide=True)) == 0
    a = _execution(experiences_dir, "A")
    R.reserver({"google"}, a.dossier, "A")
    # An experiment holds a key → code 1 (leave the services up).
    assert cmd_actives(argparse.Namespace(est_vide=True)) == 1


# ── R11: no secret value in the registry ──────────────────────────────────────


def test_R11_registre_sans_secret(experiences_dir):
    a = _execution(experiences_dir, "A")
    R.reserver({"google", "mistral"}, a.dossier, "A")
    for entree in R.actives():
        # Only the key identity (adapter) and non-sensitive metadata circulate.
        assert set(entree) == {"execution", "exp", "cles"}
        assert entree["cles"] == ["google", "mistral"]


# ── Queue: a stale request is not relaunched (2026-09-29) ─────────────────────


def _demande(nom: str, soumis: str, reprendre: bool = False) -> None:
    from experiences import file as F

    e = F.entree(nom, {"google"}, {"reprendre": reprendre})
    e["soumis"] = soumis
    F.sauver([*F.charger(), e])


def _etat(ex, etat: str, maj: str) -> None:
    import json

    (ex.dossier / "etat.json").write_text(json.dumps({"etat": etat, "maj": maj}),
                                           encoding="utf-8")


def test_une_demande_d_une_experience_terminee_depuis_n_est_pas_promue(experiences_dir):
    """The duplicate of the night of 29/09: go123 queued on 26/09 behind its own
    run, finished on 27/09, then relaunched afresh on 29/09 at 01:38 — two hours of quota
    for a measure already made."""
    b = _execution(experiences_dir, "B")
    _demande("B", "2026-09-26T21:15:48+00:00")
    _etat(b, "terminee", "2026-09-27T18:54:27+00:00")
    assert R.promouvoir_pretes() == []
    assert R.lister_file() == [], "the stale request must leave the queue, not stay in it"


def test_une_reprise_demandee_sur_une_execution_close_n_est_pas_promue(experiences_dir):
    b = _execution(experiences_dir, "B")
    _etat(b, "terminee", "2026-09-27T18:54:27+00:00")
    _demande("B", "2026-09-28T23:51:30+00:00", reprendre=True)
    assert R.promouvoir_pretes() == []
    assert R.lister_file() == []


def test_contre_epreuve_une_demande_legitime_est_promue(experiences_dir):
    """"Replay" requested AFTER the end of the last run, and resume of an exhausted one."""
    b = _execution(experiences_dir, "B")
    _etat(b, "terminee", "2026-09-27T18:54:27+00:00")
    _demande("B", "2026-09-28T09:00:00+00:00")
    c = _execution(experiences_dir, "C")
    _etat(c, "epuisee", "2026-09-28T05:16:50+00:00")
    from experiences import file as F

    e = F.entree("C", {"cerebras"}, {"reprendre": True})
    e["soumis"] = "2026-09-28T23:51:30+00:00"
    F.sauver([*F.charger(), e])
    assert [e["exp"] for e in R.promouvoir_pretes()] == ["B", "C"]


def test_une_experience_qui_obtient_ses_cles_retire_sa_propre_demande(experiences_dir):
    """Launched by another route, it left its old request behind."""
    a = _execution(experiences_dir, "A")
    _demande("A", "2026-09-26T21:19:55+00:00")
    assert R.admettre({"google"}, a.dossier, "A") == "lance"
    assert R.lister_file() == []
