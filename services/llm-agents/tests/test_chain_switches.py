"""The vehicle-chain switches live in the definition, never implicitly.

R13: `vehicule_chaine` and `verrou_retour` are fields of `experience.yaml`, enter the
      `reglages_herites` of a run, and enter the name as soon as they depart from the
      reference, which is **active**.

Why this is blocking for the 2×2 design on the vehicle chain. The mechanism already exists — the flags
`vehicle_chain_enabled` and `vehicle_return_home_lock` are driven by the environment — but
`reglages_herites` only carried `agenda_anticipation_enabled` and `max_trip_candidates`. Two
runs differing only by the chain would thus have carried the **same definition, the same
signature and the same name**, and nothing in their trace would have said which is which.

This is exactly the defect audited elsewhere — a setting that changes the measurement without
leaving a trace in its identity — at another place of the same system. Fixing it before launching,
rather than discovering it afterwards, is the whole point of the exercise.

What these tests do not cover, and this is intended: the LLM arms are all played **with the chain
active** (R14, author's decision). The switch exists here only for the seven free
arms of the 2×2.
"""

from __future__ import annotations

import pytest

from experiences import nommage as N
from experiences.experience import Experience


def _definition(**surcharges) -> dict:
    base = {
        "nom": "exp_test",
        "population": {"chemin": "data/population/population_1000_PANEL_v5"},
        "jeu": {"nom": "population_1000_PANEL_v5_20260316"},
        "gabarit": {"categorie": "itinary_multi_agent", "variante": None},
        "decideur": {"type": "duree_minimale"},
        "mode": "sans_simulateur",
        "calendrier": {"politique": "commune", "date": "2026-03-16", "graine": 42},
        "horizon_jours": 1,
        "memoire": False,
        "evenements": [],
        "graine_ordre": 42,
        "graine_tirage": 42,
        "regroupement": {"parallelisme": 8},
        "tolerances_horaires": N.TOLERANCES_REFERENCE,
        "max_candidats": 6,
        "attente_max_s": 120,
    }
    base.update(surcharges)
    return base


# ── The field exists, and its reference is "active" ─────────────────────────


def test_r13_la_chaine_est_active_par_defaut():
    """A definition that says nothing describes the nominal behaviour: chain active.

    The reference must be the actual behaviour of the simulation, otherwise a definition
    older than this field would silently change meaning.
    """
    exp = Experience.model_validate(_definition())
    assert exp.vehicule_chaine is True
    assert exp.verrou_retour is True


def test_r13_les_deux_interrupteurs_se_posent_independamment():
    """Turning off the return lock without turning off the vehicle position must stay possible."""
    exp = Experience.model_validate(_definition(verrou_retour=False))
    assert exp.vehicule_chaine is True
    assert exp.verrou_retour is False


def test_r13_les_interrupteurs_ne_touchent_ni_la_possession_ni_le_permis():
    """Safeguard of the 2×2 design: only the vehicle POSITION and the LOCK can be turned off.

    Ownership, licence and age are attributes of the person, present in the
    21 variables of the contract, which both decision-makers legitimately see. Turning them
    off would amount to giving everyone a car, which no longer measures anything. No
    field of the definition may allow doing so by accident.
    """
    # EXACT check, not by substring: "age" is found in `graine_tirage`, and a
    # test that cries wolf on an innocent word ends up being ignored.
    interrupteurs = {c for c in Experience.model_fields if c in (
        "vehicule_chaine", "verrou_retour", "possession_vehicule", "permis",
        "age_minimum", "equipement_menage", "eligibilite_modes",
    )}
    assert interrupteurs == {"vehicule_chaine", "verrou_retour"}, (
        f"the definition must carry ONLY the two switches of lot 1 of ticket 040; "
        f"found {sorted(interrupteurs)}"
    )


# ── The identity: name and signature ────────────────────────────────────────


def test_r13_la_chaine_coupee_entre_dans_le_nom():
    nominal = N.nom_canonique(_definition())
    coupee = N.nom_canonique(_definition(vehicule_chaine=False))
    assert nominal != coupee
    assert "chaine" in coupee or "nochn" in coupee, coupee


def test_r13_le_verrou_coupe_entre_dans_le_nom():
    nominal = N.nom_canonique(_definition())
    coupe = N.nom_canonique(_definition(verrou_retour=False))
    assert nominal != coupe


def test_r13_la_chaine_active_reste_muette():
    """The reference value is not written (N7): existing names do not change."""
    assert N.nom_canonique(_definition()) == N.nom_canonique(
        _definition(vehicule_chaine=True, verrou_retour=True)
    )


def test_r13_deux_conditions_du_2x2_ont_deux_signatures():
    """The core of the rule: two different measurements do not share an identity.

    Without it, the trace of a run would not say under which condition it ran —
    and this is precisely the defect fixed elsewhere for the population.
    """
    a = N.signature(_definition())
    b = N.signature(_definition(vehicule_chaine=False, verrou_retour=False))
    assert a != b


# ── The run trace ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("actif", [True, False])
def test_r13_les_reglages_herites_portent_la_chaine(actif):
    """A run must be able to say, on its own, under which condition it ran."""
    from experiences.cli import reglages_herites_de

    exp = Experience.model_validate(
        _definition(vehicule_chaine=actif, verrou_retour=actif)
    )
    herites = reglages_herites_de(exp)
    assert herites["vehicule_chaine"] is actif
    assert herites["verrou_retour"] is actif
    # The two settings already traced do not disappear.
    assert "agenda_anticipation_enabled" in herites
    assert "max_trip_candidates" in herites
