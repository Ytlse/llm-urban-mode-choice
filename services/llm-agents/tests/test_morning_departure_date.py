"""The first trip of the day is dated on the simulated day, not the next one.

The cohort's activity chains are cyclic: the first "home" activity starts
the evening before and closes the next day. Its `start_time` (72 449 s, i.e. 20:07, for
person 609 of v6) therefore belongs to the PREVIOUS day. Anchored on `base`, it put the
cursor at 20:07 of the simulated day, and the morning departure — earlier — rolled over to
the next day through the carry-over line.

Effect measured before the fix, on the eight runs of 2026-09-14: **797 of the 894 personas**
lost their first trip, which the scorer's cut then discarded (866 decisions out of
3 299), and the set recorded for them an itinerary computed on the wrong day — as did the weather
served to the prompt of the LLM arms, 12 °C on 16 March versus 15 °C on the 17th.
"""

import json
from pathlib import Path

import pytest
from experiences.jeu import deplacements_attendus
from models import Person

REPO = Path(__file__).resolve().parents[3]
JOUR = "2026-03-16"
BASE = 1773619200  # midnight of 16 March 2026, GAMA timestamp
COHORTE = REPO / "data" / "population" / "population_1000_PANEL_v6" / "population.json"


def _personne(activites: list[dict]) -> Person:
    return Person.model_validate(
        {
            "person_id": "test",
            "identity": {
                "name": "Test",
                "home": {"lat": 43.6, "lon": 1.44},
                "traits_json": {"age": 40, "has_driving_license": True, "number_of_cars": 1},
                "activities": activites,
            },
        }
    )


def _activite(aid: str, purpose: str, debut: int, fin: int, prevu: int) -> dict:
    return {
        "id": aid,
        "purpose": purpose,
        "start_time": debut,
        "end_time": fin,
        "scheduled_start_time": prevu,
        "location": {"lat": 43.6 + int(aid), "lon": 1.44},
    }


def test_une_activite_qui_enjambe_minuit_ne_repousse_pas_le_depart_du_matin():
    """The cohort case: "home" starts at 20:07 and closes at 08:59 the next day.

    The times are those of person 609 of v6, loop included — the
    `scheduled_start_time` of the first activity equals the `end_time` of the last one, as
    the sealed populations encode it. Without this structure, the closing pair does not fall
    in the right place and the test no longer measures what it believes.
    """
    personne = _personne(
        [
            _activite("0", "home", 72449, 32384, 71316),  # 20:07 → 08:59, spans midnight
            _activite("1", "work", 33516, 46416, 32384),
            _activite("2", "home", 47549, 71316, 46416),
        ]
    )
    deps = deplacements_attendus([personne], JOUR)
    assert deps, "no trip derived"
    premier = next(d for d in deps if d.ordinal == 0)
    assert premier.depart_ts == BASE + 32384, (
        f"the morning departure should be at 08:59 of the simulated day, it is {premier.depart_ts}"
    )
    assert all(BASE <= d.depart_ts < BASE + 86400 for d in deps), (
        "a trip falls outside the simulated day"
    )


def test_le_report_au_lendemain_joue_encore_quand_il_doit():
    """The fix does not disarm the carry-over rule: a target earlier than the arrival at
    the origin activity still rolls over to the next day. Without this test, removing the
    carry-over line entirely would pass too."""
    personne = _personne(
        [
            _activite("0", "home", 25200, 32384, 21600),  # 07:00 → 08:59, ordinary
            _activite("1", "work", 33516, 46416, 21600),  # target 06:00, BEFORE the arrival
            _activite("2", "home", 47549, 21600, 46416),
        ]
    )
    deps = deplacements_attendus([personne], JOUR)
    premier = next(d for d in deps if d.ordinal == 0)
    assert premier.depart_ts == BASE + 86400 + 21600, (
        "a target earlier than the arrival must roll over to the next day"
    )


@pytest.mark.skipif(not COHORTE.exists(), reason="v6 cohort missing from the repository")
def test_la_cohorte_v6_tient_entiere_dans_son_jour_simule():
    """The check that matters: over the 894 mobile personas, no trip overflows.

    Before the fix, 866 of the 3 299 trips fell on 17 March, of which 797 of rank 0.
    """
    brut = json.loads(COHORTE.read_text(encoding="utf-8"))
    gens = brut if isinstance(brut, list) else (brut.get("people") or list(brut.values())[0])
    personnes = [Person.model_validate(p) for p in gens]
    deps = deplacements_attendus(personnes, JOUR)
    hors = [d for d in deps if not (BASE <= d.depart_ts < BASE + 86400)]
    assert not hors, f"{len(hors)} trip(s) outside the simulated day, of which {sum(1 for d in hors if d.ordinal == 0)} of rank 0"
    assert len(deps) == 3299
