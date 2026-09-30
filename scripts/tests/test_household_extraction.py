"""Population extraction works on HOUSEHOLDS, not on individuals.

Extracting one member without the other would remove the quantity
measured — the date at which a reader's co-resident changes behaviour.

Everything is PURE: no simulator, no model. The cases about the sealed cohort
are skipped if it is not present (it is not versioned).

Run:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_household_extraction.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.data.population.extraire_foyers import (  # noqa: E402
    TAILLE_FOYER,
    _rang,
    choisir,
    foyers_eligibles,
)

COHORTE = RACINE / "data" / "population" / "population_1000_PANEL_v6" / "population.json"
SORTIE = RACINE / "data" / "population" / "population_20_foyers_059"

besoin_cohorte = pytest.mark.skipif(not COHORTE.is_file(), reason="v6 cohort missing (not versioned)")


def _personne(pid: str, menage: str, *, mobile: bool = True, tc: bool = False) -> dict:
    return {
        "person_id": pid,
        "immobile": not mobile,
        "household": {"id": menage},
        "identity": {"traits_json": {"name": f"P{pid}", "has_pt_subscription": tc}},
    }


# ── What eligibility keeps, and what it discards ─────────────────────────────────────────


def test_seuls_les_foyers_de_taille_deux_sont_retenus():
    pop = [
        _personne("1", "A", tc=True), _personne("2", "A"),
        _personne("3", "B", tc=True),
        _personne("4", "C", tc=True), _personne("5", "C"), _personne("6", "C"),
    ]
    assert list(foyers_eligibles(pop, abonnement_tc=False)) == ["A"]


def test_un_foyer_dont_un_membre_est_immobile_est_ecarte():
    """An immobile agent decides nothing: it hears nothing and tells nothing."""
    pop = [_personne("1", "A", tc=True), _personne("2", "A", mobile=False)]
    assert foyers_eligibles(pop, abonnement_tc=False) == {}


def test_le_critere_tc_ecarte_les_foyers_sans_aucun_abonne():
    """The article played first targets PT: a household without a user has nothing to shift."""
    pop = [
        _personne("1", "A", tc=True), _personne("2", "A"),
        _personne("3", "B"), _personne("4", "B"),
    ]
    assert list(foyers_eligibles(pop, abonnement_tc=True)) == ["A"]
    assert list(foyers_eligibles(pop, abonnement_tc=False)) == ["A", "B"]


# ── Determinism ─────────────────────────────────────────────────────────────────────────


def test_l_affectation_ne_depend_pas_de_l_ordre_de_la_population():
    """Two extractions of the same seal give the same groups, whatever the input order."""
    pop = [x for i in range(8) for x in (_personne(f"{2*i}", f"M{i}", tc=True), _personne(f"{2*i+1}", f"M{i}"))]
    a = choisir(pop, exposes=3, temoins=2, graine=59, abonnement_tc=True)[:2]
    b = choisir(list(reversed(pop)), exposes=3, temoins=2, graine=59, abonnement_tc=True)[:2]
    assert a == b


def test_changer_la_graine_change_les_groupes():
    pop = [x for i in range(12) for x in (_personne(f"{2*i}", f"M{i}", tc=True), _personne(f"{2*i+1}", f"M{i}"))]
    a = choisir(pop, exposes=4, temoins=3, graine=59, abonnement_tc=True)[0]
    b = choisir(pop, exposes=4, temoins=3, graine=60, abonnement_tc=True)[0]
    assert a != b, "a seed without effect would make the manifest misleading"


def test_le_rang_est_borne_et_stable():
    assert 0.0 <= _rang(59, "5177") < 1.0
    assert _rang(59, "5177") == _rang(59, "5177")


def test_exposes_et_temoins_sont_disjoints():
    pop = [x for i in range(10) for x in (_personne(f"{2*i}", f"M{i}", tc=True), _personne(f"{2*i+1}", f"M{i}"))]
    exposes, temoins, _ = choisir(pop, exposes=4, temoins=3, graine=59, abonnement_tc=True)
    assert not set(exposes) & set(temoins)
    assert len(exposes) == 4 and len(temoins) == 3


def test_refus_franc_quand_le_vivier_est_trop_petit():
    """An incomplete group would make one arm smaller than the other without anything saying so."""
    pop = [_personne("1", "A", tc=True), _personne("2", "A")]
    with pytest.raises(SystemExit, match="eligible households"):
        choisir(pop, exposes=6, temoins=4, graine=59, abonnement_tc=True)


# ── The delivered population ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not (SORTIE / "MANIFEST.yaml").is_file(), reason="population not extracted")
def test_la_population_livree_tient_ses_promesses():
    manifeste = yaml.safe_load((SORTIE / "MANIFEST.yaml").read_text(encoding="utf-8"))
    agents = json.loads((SORTIE / "population.json").read_text(encoding="utf-8"))

    assert manifeste["population"]["n"] == len(agents)
    exposes = manifeste["groupes"]["expose"]
    temoins = manifeste["groupes"]["temoin"]

    # The controls are in the SAME file: the measured noise floor is not zero.
    assert exposes and temoins
    assert all(len(f["membres"]) == TAILLE_FOYER for f in exposes + temoins)

    menages = {str(a["household"]["id"]) for a in agents}
    assert menages == {f["household_id"] for f in exposes + temoins}
    # No household is both exposed and control.
    assert not {f["household_id"] for f in exposes} & {f["household_id"] for f in temoins}


@pytest.mark.skipif(not (SORTIE / "MANIFEST.yaml").is_file(), reason="population not extracted")
def test_le_manifeste_dit_que_ce_n_est_pas_un_sceau():
    """Twenty agents observe a mechanism; they measure no modal share."""
    tete = (SORTIE / "MANIFEST.yaml").read_text(encoding="utf-8")[:600]
    assert "PAS un sceau" in tete
    assert "population_1000_PANEL_v6" in tete


@besoin_cohorte
def test_le_vivier_de_la_cohorte_est_suffisant_pour_les_paliers_P0_et_P1():
    """P0 asks for 2 households, P1 asks for 10. The pool must never be the limiting factor."""
    pop = json.loads(COHORTE.read_text(encoding="utf-8"))
    assert len(foyers_eligibles(pop, abonnement_tc=True)) >= 10
    assert len(foyers_eligibles(pop, abonnement_tc=False)) >= 10


# ── 2026-09-25: mixed sizes and designated readers ──────────────────────────────────────


def _adulte_ou_enfant(pid: str, menage: str, age: int) -> dict:
    p = _personne(pid, menage)
    p["identity"]["traits_json"]["age"] = age
    return p


def test_plusieurs_tailles_se_melent():
    pop = [
        _adulte_ou_enfant("1", "A", 30), _adulte_ou_enfant("2", "A", 30),
        _adulte_ou_enfant("3", "B", 40), _adulte_ou_enfant("4", "B", 38),
        _adulte_ou_enfant("5", "B", 8), _adulte_ou_enfant("6", "B", 5),
        _adulte_ou_enfant("7", "C", 40), _adulte_ou_enfant("8", "C", 40), _adulte_ou_enfant("9", "C", 9),
    ]
    assert list(foyers_eligibles(pop, abonnement_tc=False, taille=[2, 4], adultes=2)) == ["A", "B"]


def test_un_lecteur_designe_est_un_adulte_d_un_foyer_expose():
    from scripts.data.population.extraire_foyers import lecteurs_par_menage

    pop = [
        _adulte_ou_enfant("1", "A", 30), _adulte_ou_enfant("2", "A", 30),
        _adulte_ou_enfant("3", "B", 40), _adulte_ou_enfant("4", "B", 38),
        _adulte_ou_enfant("5", "B", 8), _adulte_ou_enfant("6", "B", 5),
    ]
    eligibles = foyers_eligibles(pop, abonnement_tc=False, taille=[2, 4], adultes=2)
    assert lecteurs_par_menage(["2", "4"], ["A", "B"], eligibles) == {"A": ["2"], "B": ["4"]}
    with pytest.raises(SystemExit, match="mineur"):
        lecteurs_par_menage(["5"], ["A", "B"], eligibles)
    with pytest.raises(SystemExit, match="hors des foyers exposés"):
        lecteurs_par_menage(["3"], ["A"], eligibles)
