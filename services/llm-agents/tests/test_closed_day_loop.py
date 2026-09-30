"""The simulated day closes its loop (last activity back to the first).

R1: a single function enumerates the chain, and it closes the cycle like the controller.
R2: a closing trip whose origin and destination are the same place is unusable.
R3: on v5, the enumeration yields 3,299 expected trips.
R4: the departure time of the closing trip follows the controller's rule.

The defect fixed: `deplacements_attendus` enumerated **consecutive pairs** of activities,
i.e. n − 1 trips per person, while the simulation controller closes the cycle with
`(i + 1) % n` and plays n. The platform therefore never decided the return home — 27 %
of the day on v1 — and the omission was not random: it removed exactly the
trip where the vehicle chain constraint bites hardest.

What the data say, which makes the fix simple (measured on 2026-09-11 on both
sealed cohorts, 1,894 persons, zero difference): `scheduled_start_time` of activity `(i+1) % n`
equals `end_time` of activity `i`, **wrap-around included**. The platform's convention (scheduled
time of the destination) and the controller's (end of the origin activity) therefore give
the same number, for the closing trip as for the rest. There are not two rules to reconcile,
only one pair no longer to forget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from chaine_activites import activite_suivante, paires_de_la_journee
from experiences import jeu as J
from models import Person

HOME = {"lon": 1.4400, "lat": 43.6000, "public_transport": True, "zone": "centre"}
WORK = {"lon": 1.4500, "lat": 43.6100, "public_transport": True, "zone": "nord"}
GYM = {"lon": 1.4600, "lat": 43.6200, "public_transport": False, "zone": "est"}

RACINE = Path(__file__).resolve().parents[3]
POP_V5 = RACINE / "data" / "population" / "population_1000_PANEL_v5" / "population.json"
# v1 has been ARCHIVED: it is only read on a justified exemption, and this
# test is one — it keeps the figure of the fix (2,693 → 3,693).
POP_V1 = (
    RACINE / "data" / "population" / "archive" / "population_1000_PANEL" / "population.json"
)


def _act(aid, purpose, start, end, loc, scheduled=None):
    return {
        "id": aid,
        "scheduled_start_time": scheduled,
        "start_time": float(start),
        "end_time": float(end),
        "purpose": purpose,
        "location": loc,
    }


def _entry(pid, activities):
    traits = {
        "age": 35,
        "gender": "Female",
        "main_occupation": "actif",
        "household_size": 1,
        "number_of_cars": 1,
        "has_driving_license": True,
        "personal_bike": "vélo normal",
        "residence_zone": "Toulouse",
    }
    return {
        "person_id": pid,
        "identity": {"traits_json": traits, "home": HOME, "activities": activities},
        "state": {"last_activity_index": 0},
        "is_llm_based": True,
    }


def _personne(pid, activities) -> Person:
    from inputs.population.eqasim_loader import EqasimJSONPopulationLoader

    return EqasimJSONPopulationLoader().load_population_from_data(
        [_entry(pid, activities)], max_size=1, bbox=None
    )[0]


def _trois_activites() -> Person:
    """home → work → gym → (back home). Three activities, hence THREE trips."""
    return _personne(
        "p3",
        [
            _act("a0", "home", 0, 8 * 3600, HOME, 20 * 3600),
            _act("a1", "work", 9 * 3600, 17 * 3600, WORK, 8 * 3600),
            _act("a2", "leisure", 18 * 3600, 20 * 3600, GYM, 17 * 3600),
        ],
    )


# ── R1: a single enumeration, and it closes the loop ─────────────────────────


def test_r1_trois_activites_donnent_trois_deplacements_pas_deux():
    """The heart of alert A1: n activities mean n trips, not n − 1."""
    personne = _trois_activites()
    paires = paires_de_la_journee(personne.identity.activities)
    assert len(paires) == 3
    assert [(o.id, d.id) for o, d in paires] == [
        ("a0", "a1"),
        ("a1", "a2"),
        ("a2", "a0"),
    ]


def test_r1_la_derniere_paire_est_le_retour_au_domicile():
    """The closing trip goes from the last activity to the first, which is home."""
    personne = _trois_activites()
    origine, destination = paires_de_la_journee(personne.identity.activities)[-1]
    assert (
        origine.purpose.value if hasattr(origine.purpose, "value") else origine.purpose
    )
    assert origine.id == "a2"
    assert destination.id == "a0"


def test_r1_activite_suivante_boucle_comme_le_controleur():
    """`activite_suivante` applies `(i + 1) % n`, the rule of the controller's three sites."""
    acts = _trois_activites().identity.activities
    assert activite_suivante(acts, acts[0]).id == "a1"
    assert activite_suivante(acts, acts[1]).id == "a2"
    assert activite_suivante(acts, acts[2]).id == "a0"


def test_r1_une_personne_a_une_seule_activite_ne_se_deplace_pas():
    """The controller's `len <= 1` guard: no trip from an activity to itself."""
    personne = _personne("p1", [_act("c0", "home", 0, 86400, HOME, 86400)])
    acts = personne.identity.activities
    assert activite_suivante(acts, acts[0]) is None
    assert paires_de_la_journee(acts) == []


def test_r1_deplacements_attendus_utilise_la_chaine_partagee():
    """The platform does not reimplement the enumeration: same pairs on both sides."""
    personne = _trois_activites()
    attendus = J.deplacements_attendus([personne], "2026-03-16")
    paires = paires_de_la_journee(personne.identity.activities)
    assert [(d.origine_activity_id, d.activity_id) for d in attendus] == [
        (o.id, dst.id) for o, dst in paires
    ]


# ── R2: a closing trip on the spot is not a usable trip ──────────────────────


def test_r2_fermeture_origine_egale_destination_est_inexploitable():
    """A person whose last activity is already at home closes on the spot.

    The trip exists (it counts in the raw expected trips), but no proposal is
    possible: it is the `origine_egale_destination` reason the set already knows.
    On v5, 77 mobile persons close their day this way. (The set counts 138
    trips with origin equal to destination: these 77 closing trips, plus 61 intermediate
    trips between two activities located at the same place.)
    """
    personne = _personne(
        "pferme",
        [
            _act("d0", "home", 0, 8 * 3600, HOME, 18 * 3600),
            _act("d1", "work", 9 * 3600, 17 * 3600, WORK, 8 * 3600),
            _act("d2", "home", 18 * 3600, 20 * 3600, HOME, 17 * 3600),
        ],
    )
    attendus = J.deplacements_attendus([personne], "2026-03-16")
    assert len(attendus) == 3
    fermeture = attendus[-1]
    assert (fermeture.origine_activity_id, fermeture.activity_id) == ("d2", "d0")
    assert (fermeture.origine.lat, fermeture.origine.lon) == (
        fermeture.destination.lat,
        fermeture.destination.lon,
    )


# ── R3: the figures of the sealed cohorts ────────────────────────────────────


@pytest.mark.skipif(not POP_V5.is_file(), reason="v5 cohort missing from disk")
def test_r3_la_cohorte_v5_rend_3299_deplacements_attendus():
    from experiences.population import charger_population

    personnes, _ = charger_population(POP_V5.parent)
    assert len(J.deplacements_attendus(personnes, "2026-03-16")) == 3299


@pytest.mark.skipif(not POP_V1.is_file(), reason="v1 cohort missing from disk")
def test_r3_la_cohorte_v1_rend_3693_deplacements_attendus():
    """Kept as a witness of the fix's figure: 2,693 → 3,693, i.e. 27 % of the day
    never decided before this fix."""
    from experiences.population import charger_population

    personnes, _ = charger_population(
        POP_V1.parent, archivee_confirmee="témoin du ticket 045"
    )
    assert len(J.deplacements_attendus(personnes, "2026-03-16")) == 3693


# ── R4: the departure time of the closing trip ───────────────────────────────


def test_r4_le_depart_de_la_fermeture_suit_la_fin_de_la_derniere_activite():
    """Both conventions coincide: scheduled time of the destination == end of the origin.

    This is not an implementation coincidence but a property of the sealed
    populations, checked on the 1,894 persons of both cohorts without a single difference.
    """
    personne = _trois_activites()
    fermeture = J.deplacements_attendus([personne], "2026-03-16")[-1]
    acts = {a.id: a for a in personne.identity.activities}
    assert fermeture.depart_24h == int(acts["a2"].end_time)
    assert fermeture.depart_24h == int(acts["a0"].scheduled_start_time)


@pytest.mark.skipif(not POP_V5.is_file(), reason="v5 cohort missing from disk")
def test_r4_le_bouclage_horaire_est_une_propriete_de_la_cohorte():
    """`scheduled_start_time` of the first activity == `end_time` of the last, for everyone."""
    pop = json.loads(POP_V5.read_text(encoding="utf-8"))
    ecarts = [
        p.get("person_id")
        for p in pop
        if len(p["identity"]["activities"] or []) >= 2
        and int(p["identity"]["activities"][-1]["end_time"])
        != int(p["identity"]["activities"][0]["scheduled_start_time"])
    ]
    assert ecarts == []
