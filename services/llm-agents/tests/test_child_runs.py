"""The common base, as child runs.

The first fourteen days of the paired campaign are identical in all three arms:
44 decisions, 0 difference, paid three times. With eleven arms planned, that is about a third of
the cost of each additional arm thrown away.

A child inherits its parent's MEMORY. If the population, the model or the seeds
differ, that memory describes another experiment than the one it is about to play — and nothing
in the outputs would say so. Hence refusal by default, and free fields named one by one.
"""

import json
import sys
from pathlib import Path

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urban_mobility_agents.utils import filiation, identite_run, reprise


IDENTITE_SOCLE = {
    "modeles_admis": ["gemini-3.1-flash-lite"],
    "instances_admises": ["google_gemini31_key1", "google_gemini31_key2"],
    "routage_instances": {},
    "variante_prompt": "factual_neutral",
    "population": "pop_1.json",
    "memoire_longue": True,
    "auto_reflexion": True,
    "choc": "aucun",
    "graine_tirage": 42,
    "graine_ordre": 7,
    "graine_meteo": 3,
    "cache_decisions": False,
    "chaine_vehicules": True,
    "verrou_retour_domicile": True,
    "seuil_troncature": 0.0,
    "seuil_choc": 0.7,
    "retard_saturation": "asymptote",
    "retard_gravite_max": 0.70,
    "retard_ref_s": 1800,
    "fenetre_changements_jours": 14,
    "changements_max": 3,
    "mode_fenetre_changements": "derivee",
    "seuil_service_changement": 0.35,
    "plancher_changement_jours": 2.0,
    "plafond_changement_jours": 30.0,
    "reflexion_stm_min_entrees": 10,
    "meteo_par_agent": False,
    "partage_foyer": False, "taches_en_vol": 8,
    "run_parent": "",
    "champs_libres": [],
}


@pytest.fixture(autouse=True)
def _sans_env(monkeypatch):
    monkeypatch.delenv(filiation.ENV_PARENT, raising=False)
    monkeypatch.delenv(filiation.ENV_CHAMPS_LIBRES, raising=False)


@pytest.fixture
def journal_logs():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(
        lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO"
    )
    yield lignes
    logger.remove(sink)


def _parent(racine: Path, identite=None, avec_point=True, jour=14) -> Path:
    dossier = racine / "parent"
    (dossier / "long_term_memory").mkdir(parents=True, exist_ok=True)
    (dossier / "long_term_memory" / "index.bin").write_text("mémoire du socle", encoding="utf-8")
    identite_run.ecrire(dossier, identite or dict(IDENTITE_SOCLE), ecraser=True)
    if avec_point:
        point = dossier / reprise.POINTS / f"jour_{jour:03d}"
        (point / "long_term_memory").mkdir(parents=True)
        (point / "long_term_memory" / "index.bin").write_text("mémoire du socle", encoding="utf-8")
        (point / reprise.DESCRIPTION).write_text(
            json.dumps({
                "jour_simule": jour,
                "timestamp_simule": 1773637200,
                "horodatage_simule": "2026-03-29T03:00:00",
                "ancre_run": 1773637200,
                "identite": identite or dict(IDENTITE_SOCLE),
            }),
            encoding="utf-8",
        )
    return dossier


def _decisions(dossier: Path, lignes: list[dict]) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "decisions_rejeu.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lignes), encoding="utf-8"
    )


def _decision(activite="a1", instant=1.0, code="__DIRECT_CAR__^^", personne="899549"):
    return {
        "personne": personne, "activite": activite, "instant": instant,
        "code_plan": code, "raison": "", "fournisseur": "google_gemini31_key1",
    }


# ══════════════════════ I — the lineage ═════════════════════════════════════════


def test_I1_l_identite_d_un_enfant_porte_sa_filiation(monkeypatch):
    """I1 — which parent it inherits from, and what it allows itself to vary."""
    from settings import settings

    monkeypatch.setenv(filiation.ENV_CHAMPS_LIBRES, "choc,mode_fenetre_changements")
    ident = identite_run.composer(
        settings, empreinte_choc="abc",
        run_parent="2026-09-21_socle", champs_libres=filiation.champs_libres(),
    )
    assert ident["run_parent"] == "2026-09-21_socle"
    assert ident["champs_libres"] == ["choc", "mode_fenetre_changements"]


def test_I1b_un_champ_libre_inconnu_est_refuse(monkeypatch):
    """I1b — `fenetre_changement` instead of `fenetre_changements_jours` would let through a
    difference we believed we had declared."""
    monkeypatch.setenv(filiation.ENV_CHAMPS_LIBRES, "choc,fenetre_changement")
    with pytest.raises(filiation.FiliationRefusee) as err:
        filiation.champs_libres()
    assert "fenetre_changement" in str(err.value)


def test_I2_un_ecart_non_declare_fait_refuser(tmp_path, monkeypatch):
    """I2 — and the faulty field is named: fixing blindly costs one launch per round."""
    parent = _parent(tmp_path)
    courante = dict(IDENTITE_SOCLE) | {"population": "pop_autre.json"}
    monkeypatch.setenv(filiation.ENV_CHAMPS_LIBRES, "choc")
    with pytest.raises(filiation.FiliationRefusee) as err:
        filiation.amorcer(tmp_path / "enfant", parent, courante)
    assert "population" in str(err.value)


def test_I3_un_ecart_declare_laisse_demarrer(tmp_path, monkeypatch):
    """I3 — this is exactly what an arm does: it changes its shock, and nothing else."""
    parent = _parent(tmp_path)
    courante = dict(IDENTITE_SOCLE) | {"choc": "empreinte_c6"}
    monkeypatch.setenv(filiation.ENV_CHAMPS_LIBRES, "choc")
    meta = filiation.amorcer(tmp_path / "enfant", parent, courante)
    assert meta["jour_simule"] == 14


def test_I4_un_parent_sans_point_de_reprise_fait_refuser(tmp_path):
    """I4 — there is nothing to inherit: a parent is played, then frozen."""
    parent = _parent(tmp_path, avec_point=False)
    with pytest.raises(filiation.FiliationRefusee) as err:
        filiation.amorcer(tmp_path / "enfant", parent, dict(IDENTITE_SOCLE))
    assert "resume point" in str(err.value)


def test_I4b_un_parent_sans_identite_fait_refuser(tmp_path):
    """I4b — impossible to say which experiment its memory comes from."""
    parent = tmp_path / "parent"
    (parent / reprise.POINTS).mkdir(parents=True)
    with pytest.raises(filiation.FiliationRefusee) as err:
        filiation.amorcer(tmp_path / "enfant", parent, dict(IDENTITE_SOCLE))
    assert "identity" in str(err.value)


def test_I5_l_amorcage_copie_le_point_du_parent(tmp_path):
    """I5 — the child starts on the common base's memory, restored by the hot-resume mechanism."""
    parent = _parent(tmp_path)
    enfant = tmp_path / "enfant"
    filiation.amorcer(enfant, parent, dict(IDENTITE_SOCLE))
    trouve = reprise.dernier_point(enfant)
    assert trouve is not None
    chemin, meta = trouve
    assert meta["jour_simule"] == 14
    assert (chemin / "long_term_memory" / "index.bin").read_text(encoding="utf-8") == (
        "mémoire du socle"
    )


def test_I6_le_rejeu_de_l_enfant_reproduit_le_parent(tmp_path):
    """I6 — decision by decision, over the common days."""
    parent, enfant = tmp_path / "parent", tmp_path / "enfant"
    lignes = [_decision("a1", 1.0), _decision("a2", 2.0, "__DIRECT_BIKE__^^")]
    _decisions(parent, lignes)
    _decisions(enfant, lignes + [_decision("a3", 99.0)])  # the child lives its own sequel
    assert filiation.ecarts_de_reproduction(parent, enfant, jusqu_a=50.0) == []


def test_I7_un_ecart_sur_un_jour_commun_est_refuse_et_chiffre(tmp_path, journal_logs):
    """I7 — never absorbed: without this acceptance check, two arms would believe they share
    a baseline they no longer share."""
    parent, enfant = tmp_path / "parent", tmp_path / "enfant"
    _decisions(parent, [_decision("a1", 1.0), _decision("a2", 2.0)])
    _decisions(enfant, [_decision("a1", 1.0), _decision("a2", 2.0, "__DIRECT_BIKE__^^")])
    ecarts = filiation.ecarts_de_reproduction(parent, enfant, jusqu_a=50.0)
    assert len(ecarts) == 1 and "a2" in ecarts[0]
    filiation.verifier_reproduction(parent, enfant, jusqu_a=50.0)
    assert [m for n, m in journal_logs if n == "ERROR" and "1/2 decision" in m]


def test_I7b_une_decision_manquante_compte_comme_un_ecart(tmp_path):
    """I7b — a shortened common base is not a reproduced common base."""
    parent, enfant = tmp_path / "parent", tmp_path / "enfant"
    _decisions(parent, [_decision("a1", 1.0), _decision("a2", 2.0)])
    _decisions(enfant, [_decision("a1", 1.0)])
    ecarts = filiation.ecarts_de_reproduction(parent, enfant, jusqu_a=50.0)
    assert len(ecarts) == 1 and "absente chez l'enfant" in ecarts[0]


def test_I7c_la_recette_conforme_se_journalise_aussi(tmp_path, journal_logs):
    """I7c — SUCCESS explicitly: a program that is silent when all is well does not let us
    tell "it works" from "it no longer runs"."""
    parent, enfant = tmp_path / "parent", tmp_path / "enfant"
    lignes = [_decision("a1", 1.0), _decision("a2", 2.0)]
    _decisions(parent, lignes)
    _decisions(enfant, lignes)
    filiation.verifier_reproduction(parent, enfant, jusqu_a=50.0)
    assert [m for n, m in journal_logs if n == "INFO" and "replay conforms" in m]


def test_I8_un_enfant_n_ecrit_jamais_chez_son_parent(tmp_path):
    """I8 — otherwise the common base would become the residue of the last arm played."""
    parent = _parent(tmp_path)
    avant = sorted(p.relative_to(parent).as_posix() for p in parent.rglob("*"))
    filiation.amorcer(tmp_path / "enfant", parent, dict(IDENTITE_SOCLE))
    apres = sorted(p.relative_to(parent).as_posix() for p in parent.rglob("*"))
    assert avant == apres


def test_I9_sans_parent_declare_rien_ne_change(tmp_path):
    """I9 — lineage is a mode, not a new mandatory path."""
    assert filiation.parent_declare() == ""
    assert filiation.champs_libres() == ()
