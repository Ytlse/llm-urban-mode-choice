"""A denominator that does not depend on how the run unfolded.

R9: the set of unusable trips is DERIVED FROM THE SET when the
     run opens, not accumulated as decisions go.

And three consequences of the closed day (block A), which must be settled here because they
touch the same count:

- `Jeu.est_complet` became unreachable: `couverture()` counted the home →
  home closing trip among the expected ones but never among the covered ones, whereas it CANNOT be
  covered. A perfectly prepared set declared itself incomplete.
- Progress showed 80 % on a finished run: the numerator excluded the
  unusable trips, the denominator did not. Every complete run would have stayed under 100 % in
  the dashboard.
- The "trips without proposal" alarm fired on the slightest set preparation:
  on the v5 cohort, 138 of the 3,299 trips have their origin as destination, i.e. 4.2 %
  for a 5 % threshold. Yet an origin equal to its destination is not a failure:
  there is no itinerary to compute between a point and itself. Counting it in an engine
  health alarm drowns the signal the alarm is there to carry.

The common thread: `origine_egale_destination` and `aucune_proposition` are two things. The
first is a known property of the day, the second is a failure. Adding them up
made three counters lie at once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
import yaml

from experiences import jeu as J
from experiences.jeu import (
    MOTIF_AUCUNE_PROPOSITION,
    MOTIF_ORIGINE_EGALE_DESTINATION,
    est_defaillance_moteur,
)
from experiences.population import charger_population
from models import Transit, TransitLocation, TravelPlan

HOME = {"lon": 1.4400, "lat": 43.6000, "public_transport": True, "zone": "centre"}
WORK = {"lon": 1.4500, "lat": 43.6100, "public_transport": True, "zone": "nord"}


def _act(aid, purpose, start, end, loc, scheduled):
    return {
        "id": aid,
        "scheduled_start_time": float(scheduled),
        "start_time": float(start),
        "end_time": float(end),
        "purpose": purpose,
        "location": loc,
    }


def _population():
    """One person, day looping back home: home → work → home.

    Three activities, hence three trips. The last one (a2 → a0) goes from home to
    home: it is unusable, and it is exactly the case this file exercises.
    The times respect the looping of sealed populations
    (`scheduled_start_time` of the first == `end_time` of the last).
    """
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
    acts = [
        _act("a0", "home", 0, 8 * 3600, HOME, 20 * 3600),
        _act("a1", "work", 9 * 3600, 17 * 3600, WORK, 8 * 3600),
        _act("a2", "home", 18 * 3600, 20 * 3600, HOME, 17 * 3600),
    ]
    return [
        {
            "person_id": "p1",
            "identity": {"traits_json": traits, "home": HOME, "activities": acts},
            "state": {"last_activity_index": 0},
            "is_llm_based": True,
        }
    ]


def _plan(mode, origin, dest, dep_ts, duration=600):
    loc_o = TransitLocation(stop="", lat=origin.lat, lon=origin.lon)
    loc_d = TransitLocation(stop="", lat=dest.lat, lon=dest.lon)
    leg = Transit(
        start_time=dep_ts * 1000,
        end_time=(dep_ts + duration) * 1000,
        duration=duration,
        distance=1500.0,
        mode=mode,
        start_location=loc_o,
        end_location=loc_d,
        transit_route=f"__DIRECT_{mode.upper()}__",
    )
    return TravelPlan(
        id=f"{mode}-{dep_ts}",
        start_location=origin,
        end_location=dest,
        start_time=dep_ts * 1000,
        end_time=(dep_ts + duration) * 1000,
        duration=duration,
        legs=[leg],
    )


class MoteurFactice:
    """Always returns walk, bike and car — no failure, to isolate the subject."""

    async def get_itineraries(self, origin, destination, departure_time, **_):
        return [
            _plan("foot", origin, destination, departure_time, 1800),
            _plan("bicycle", origin, destination, departure_time, 900),
            _plan("car", origin, destination, departure_time, 600),
        ]


@pytest.fixture
def banc_jeu(tmp_path):
    """A closed set, prepared on a day that loops back home."""
    pop_dir = tmp_path / "pop"
    pop_dir.mkdir()
    fichier = pop_dir / "population.json"
    fichier.write_text(json.dumps(_population()), encoding="utf-8")
    sha = hashlib.sha256(fichier.read_bytes()).hexdigest()
    (pop_dir / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "nom": "pop_test",
                "population": {"fichier": "population.json", "sha256": sha},
            }
        ),
        encoding="utf-8",
    )
    personnes, info = charger_population(pop_dir)
    prep = J.JeuEnPreparation.ouvrir(
        tmp_path / "jeu",
        "jeu_test",
        info,
        "2026-03-16",
        dependances={"commit": "abc", "gtfs": {"calendar.txt": "c1"}},
    )
    asyncio.run(
        J.preparer(
            prep,
            personnes,
            MoteurFactice(),
            fabrique_locale=lambda **_: None,
            progression_s=100,
        )
    )
    attendus = J.deplacements_attendus(personnes, "2026-03-16")
    prep.clore(attendus, len(personnes))
    return {"jeu": J.Jeu.charger(tmp_path / "jeu"), "attendus": attendus}


# ── The split of reasons, from which everything follows ─────────────────────


def test_une_fermeture_sur_place_nest_pas_une_defaillance_de_moteur():
    """Between a point and itself there is no itinerary to find: nothing failed."""
    assert est_defaillance_moteur(MOTIF_ORIGINE_EGALE_DESTINATION) is False


def test_aucune_proposition_est_une_defaillance_de_moteur():
    """Here, the engines were queried and returned nothing: it is a failure, it raises an alarm."""
    assert est_defaillance_moteur(MOTIF_AUCUNE_PROPOSITION) is True


def test_un_deplacement_servi_na_pas_de_motif_et_nest_pas_une_defaillance():
    assert est_defaillance_moteur(None) is False


# ── R9: the denominator is read from the set, not from the unfolding ────────


def test_r9_les_inexploitables_se_derivent_du_jeu(banc_jeu):
    """A trip without proposal in the set is unusable, before any decision."""
    jeu, attendus = banc_jeu["jeu"], banc_jeu["attendus"]
    inexploitables = jeu.inexploitables(attendus)
    attendus_sans_props = {
        (d.person_id, d.activity_id)
        for d in attendus
        if (l := jeu.ligne(d.person_id, d.activity_id)) is not None
        and not l.propositions
    }
    assert inexploitables == attendus_sans_props
    assert inexploitables, "the bench must contain at least one closing trip in place"


def test_r9_un_deplacement_non_couvert_nest_pas_un_inexploitable(banc_jeu):
    """Absent from the set and present-but-empty are two things.

    The first is a preparation gap (`non_couvert`), the second a property of the
    trip. Confusing them would make the denominator vary with the set's completeness.
    """
    jeu, attendus = banc_jeu["jeu"], banc_jeu["attendus"]
    inexploitables = jeu.inexploitables(attendus)
    for person_id, activity_id in inexploitables:
        assert jeu.ligne(person_id, activity_id) is not None


def test_r9_le_denominateur_est_le_meme_quel_que_soit_le_sort_de_lexecution(banc_jeu):
    """The core of A3: two runs on the same (population, set) pair count the same.

    We derive twice, running nothing in between: the number cannot depend
    on what did not happen.
    """
    jeu, attendus = banc_jeu["jeu"], banc_jeu["attendus"]
    assert jeu.inexploitables(attendus) == jeu.inexploitables(attendus)
    assert len(attendus) - len(jeu.inexploitables(attendus)) > 0


# ── The completeness of a set ───────────────────────────────────────────────


def test_un_jeu_dont_toutes_les_fermetures_sont_sur_place_est_complet(banc_jeu):
    """What cannot be covered must not prevent a set from being complete.

    Otherwise no set would ever be so again, for any population whose days
    come back home — that is, all of them.
    """
    jeu = banc_jeu["jeu"]
    couv = jeu.couverture()
    assert couv["deplacements_exploitables"] < couv["deplacements_attendus"]
    assert couv["deplacements_couverts"] == couv["deplacements_exploitables"]
    assert jeu.est_complet is True


def test_la_couverture_publie_les_deux_denominateurs(banc_jeu):
    """Raw and usable, both: a single figure would mix two questions."""
    couv = banc_jeu["jeu"].couverture()
    assert couv["deplacements_attendus"] >= couv["deplacements_exploitables"]
    assert couv["taux"] == pytest.approx(
        couv["deplacements_couverts"] / couv["deplacements_exploitables"]
    )
