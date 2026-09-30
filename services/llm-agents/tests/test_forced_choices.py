"""What the decider decided, and what the situation forced on it.

R10: a summary publishes the modal shares **both ways** — all decisions, and excluding
      single-itinerary decisions. Both carry their count.
R11: the count of forced choices is published next to the shares, passed up and not recomputed.
R12: the term-by-term comparison between arms is made on the **intersection** of the trips
      that all actually decided, and the scope retained is declared with the figure.

The fact, with an example. Jean drives to work in the morning. In the evening, his car is at the
office: the only itinerary that exists to get home is the car. **Nobody decides anything** —
neither the language model, nor the oracle, nor the random draw. Yet this trip counts in the
published modal shares, as if it had been chosen.

And their number **depends on the arm**, since it follows from what the arm decided earlier:
taking the car in the morning means committing to a car return in the evening. Measured on the 23
complete runs of cohort v1, with identical population and set: 393 for
`duree_minimale`, 381 for `majoritaire_voiture`, 300 to 359 for the LLM arms, 286 for
`aleatoire`. That is 11 to 15 % of the day identical and undecided, which mechanically compresses
the gaps we are trying to measure.

**Author's ruling: the constraint is not the defect, it is the frame.** A trip
imposed by an earlier choice remains a fact of the simulated day, and a real day
contains some. So the chain is not neutralized. We publish **both figures**, because a single one
mixes what the decision-maker decided and what the situation imposed on it, and the second
quantity varies from one arm to another. The "all decisions" shares carry the conclusion; the
shares excluding single itineraries say what the decision-maker actually did.

For the comparison between arms, neither suffices: "excluding single itinerary" removes
**different** trips depending on the arm, so it does not make the comparison fairer, it
shifts it. Hence R12: the intersection of the situations that **all** decided, declared with its
count.
"""

from __future__ import annotations

import pytest

from experiences import registre as R
from experiences.archive import Execution
from experiences.decision import METHODE_CHOIX_UNIQUE
from experiences.registre import perimetre_commun


def _trace(person_id, activity_id, mode, methode="llm"):
    """A minimal but COMPLETE decision trace: the archive refuses partial traces."""
    return {
        "person_id": person_id,
        "activity_id": activity_id,
        "methode": methode,
        "retenue": {"mode": mode},
        "presentees": [{"mode": mode}],
        "ecartees": [],
        "distribution": {mode: 1.0},
        "reponse_brute": None,
        "sources": {},
    }


@pytest.fixture
def synthese_banc(tmp_path):
    """Six decisions, of which TWO are single-itinerary, both by car.

    This is the real case alert A5 describes: an agent who left by car in the morning has only
    one itinerary left to get home in the evening. Forced choices therefore pull the car share
    up without anyone having decided them.
    """
    ex = Execution.creer(
        tmp_path / "exp_banc",
        {"nom": "exp_banc"},
        {},
        {"parallelisme": 1},
        {"graine_tirage": 42},
    )
    decisions = [
        _trace("p1", "a1", "car"),
        _trace("p1", "a2", "car", METHODE_CHOIX_UNIQUE),
        _trace("p2", "b1", "walk"),
        _trace("p2", "b2", "bike"),
        _trace("p3", "c1", "transit"),
        _trace("p3", "c2", "car", METHODE_CHOIX_UNIQUE),
    ]
    for t in decisions:
        ex.ajouter_decision(t)
    ex.ecrire_compteurs(
        {
            "attendus": 6,
            "attendus_exploitables": 6,
            "decides": 6,
            "choix_unique": 2,
            "couverture": {"attendus": 6, "attendus_bruts": 6, "decides": 6},
        }
    )
    ex.fermer()
    return R.synthese(ex.dossier)


# ── R10 / R11: the summary publishes both readings ──────────────────────────


def test_r10_les_deux_lectures_sont_publiees(synthese_banc):
    s = synthese_banc
    assert "parts_modales" in s
    assert "parts_modales_hors_choix_unique" in s


def test_r10_les_deux_lectures_portent_leur_effectif(synthese_banc):
    """A percentage without its count cannot be read back: 100 % of two decisions is nothing."""
    s = synthese_banc
    assert s["parts_modales"]["n"] > s["parts_modales_hors_choix_unique"]["n"]
    for bloc in (s["parts_modales"], s["parts_modales_hors_choix_unique"]):
        assert bloc["n"] == sum(bloc["effectifs"].values())


def test_r10_la_lecture_hors_choix_unique_retire_exactement_les_choix_forces(
    synthese_banc,
):
    s = synthese_banc
    ecart = s["parts_modales"]["n"] - s["parts_modales_hors_choix_unique"]["n"]
    assert ecart == s["choix_forces"]["n"]


def test_r10_les_deux_lectures_donnent_des_parts_differentes(synthese_banc):
    """If they coincided, publishing both would teach nothing.

    On the bench, the forced choices are all by car — the real case, an agent who left by
    car returns by car: removing them must lower the car share.
    """
    toutes = synthese_banc["parts_modales"]["pourcent"]["car"]
    hors = synthese_banc["parts_modales_hors_choix_unique"]["pourcent"]["car"]
    assert hors < toutes


def test_r11_le_compte_de_choix_forces_est_remonte_pas_recalcule(synthese_banc):
    """`compteurs.choix_unique` already exists: the summary relays it, it does not invent."""
    bloc = synthese_banc["choix_forces"]
    assert bloc["n"] == 2
    assert bloc["part"] == 2 / synthese_banc["parts_modales"]["n"]
    assert "dépend du bras" in bloc["lecture"]


# ── R12: the common scope ───────────────────────────────────────────────────


def test_r12_le_perimetre_commun_est_lintersection_des_decisions_reelles():
    """Only the trips that ALL arms actually decided get in."""
    a = [
        _trace("p1", "a1", "car"),
        _trace("p1", "a2", "walk"),
        _trace("p2", "b1", "bike"),
    ]
    b = [
        _trace("p1", "a1", "walk"),
        _trace("p1", "a2", "car", METHODE_CHOIX_UNIQUE),  # forced in b, not in a
        _trace("p2", "b1", "car"),
    ]
    commun = perimetre_commun({"a": a, "b": b})
    assert commun["cles"] == {("p1", "a1"), ("p2", "b1")}
    assert commun["n"] == 2


def test_r12_le_perimetre_est_declare_avec_son_effectif_et_ce_quil_retire():
    """A restricted scope without its reason reads as missing data."""
    a = [_trace("p1", "a1", "car"), _trace("p1", "a2", "walk")]
    b = [_trace("p1", "a1", "walk"), _trace("p1", "a2", "car", METHODE_CHOIX_UNIQUE)]
    commun = perimetre_commun({"a": a, "b": b})
    assert commun["n"] == 1
    assert commun["retires"]["a"] == 1
    assert commun["retires"]["b"] == 1
    assert commun["motif"]


def test_r12_un_deplacement_absent_dun_bras_sort_du_perimetre():
    """Two columns whose denominator differs cannot be compared term by term."""
    a = [_trace("p1", "a1", "car"), _trace("p2", "b1", "walk")]
    b = [_trace("p1", "a1", "walk")]
    commun = perimetre_commun({"a": a, "b": b})
    assert commun["cles"] == {("p1", "a1")}


def test_r12_trois_bras_donnent_un_perimetre_plus_etroit_que_deux():
    """Each additional arm can only narrow the intersection, never widen it."""
    a = [
        _trace("p1", "a1", "car"),
        _trace("p2", "b1", "walk"),
        _trace("p3", "c1", "bike"),
    ]
    b = [
        _trace("p1", "a1", "walk"),
        _trace("p2", "b1", "car"),
        _trace("p3", "c1", "car"),
    ]
    c = [
        _trace("p1", "a1", "bike"),
        _trace("p2", "b1", "car", METHODE_CHOIX_UNIQUE),
        _trace("p3", "c1", "walk"),
    ]
    deux = perimetre_commun({"a": a, "b": b})["n"]
    trois = perimetre_commun({"a": a, "b": b, "c": c})["n"]
    assert trois < deux


def test_r12_un_seul_bras_rend_ses_propres_decisions():
    a = [_trace("p1", "a1", "car"), _trace("p1", "a2", "walk", METHODE_CHOIX_UNIQUE)]
    commun = perimetre_commun({"a": a})
    assert commun["n"] == 1 and commun["retires"]["a"] == 1
