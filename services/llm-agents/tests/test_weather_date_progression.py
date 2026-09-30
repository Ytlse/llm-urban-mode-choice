"""The drawn weather date is a START that advances one day per simulated day.
"""

import datetime as dt

import pytest

from sim_clock import gama_timestamp, wall_clock
from urban_mobility_agents.utils import ancre_run
from urban_mobility_agents.utils.weather_draw import (
    avancer_date,
    date_meteo,
    jours_eligibles,
    timestamp_meteo,
)

GRAINE = 42
# The "year" window as `_weather_eligible_days` describes it: 2024, hence a leap year.
JOURS_ANNEE = jours_eligibles("2024-01-01", "2024-12-31", None)


def _ts(annee, mois, jour, heure=8, minute=0):
    # Naive ON PURPOSE: the GAMA timestamp is a WALL-CLOCK time, and `gama_timestamp` expects
    # exactly that (cf. sim_clock). A time zone here would shift the bulletin read time.
    return gama_timestamp(dt.datetime(annee, mois, jour, heure, minute))  # noqa: DTZ001


@pytest.fixture(autouse=True)
def _sans_ancre():
    """Each case starts with no anchor: module state must not leak from one test to another."""
    ancre_run.reinitialiser()
    yield
    ancre_run.reinitialiser()


class TestProgression:
    def test_A1_jour_un_rend_exactement_le_tirage_historique(self):
        """A single-day experiment, already measured and sealed, replays the same weather."""
        ts = _ts(2026, 3, 16)
        attendu = date_meteo("609", GRAINE, JOURS_ANNEE)
        obtenu = wall_clock(
            timestamp_meteo(ts, "609", GRAINE, JOURS_ANNEE, jours_ecoules=0)
        )
        assert (obtenu.month, obtenu.day) == attendu

    def test_A2_trois_jours_ecoules_avancent_de_trois_jours(self):
        assert avancer_date(10, 3, 3) == (10, 6)

    def test_A2bis_la_progression_passe_par_le_timestamp(self):
        ts = _ts(2026, 3, 19)
        mois, jour = date_meteo("609", GRAINE, JOURS_ANNEE)
        lu = wall_clock(
            timestamp_meteo(ts, "609", GRAINE, JOURS_ANNEE, jours_ecoules=3)
        )
        assert (lu.month, lu.day) == avancer_date(mois, jour, 3)

    def test_A3_ecart_entre_deux_agents_constant_sur_tout_le_run(self):
        """Two agents start from different dates and advance at the same pace."""
        ecarts = set()
        for n in range(60):
            a = wall_clock(
                timestamp_meteo(
                    _ts(2026, 3, 16), "609", GRAINE, JOURS_ANNEE, jours_ecoules=n
                )
            )
            b = wall_clock(
                timestamp_meteo(
                    _ts(2026, 3, 16), "11195", GRAINE, JOURS_ANNEE, jours_ecoules=n
                )
            )
            # Gap counted on the RANK in the year and modulo 365: both agents wrap around
            # on 31 December, and a gap read in dates would flip sign at that point although
            # the step has not moved.
            rang_a = dt.date(2023, a.month, a.day).timetuple().tm_yday
            rang_b = dt.date(2023, b.month, b.day).timetuple().tm_yday
            ecarts.add((rang_a - rang_b) % 365)
        assert len(ecarts) == 1, (
            f"the gap between the two agents varies: {sorted(ecarts)}"
        )

    def test_A4_le_31_decembre_est_suivi_du_1er_janvier(self):
        assert avancer_date(12, 31, 1) == (1, 1)

    def test_A5_le_29_fevrier_en_depart_est_ramene_au_1er_mars(self):
        assert avancer_date(2, 29, 0) == (3, 1)

    def test_A5bis_la_progression_ne_produit_jamais_un_29_fevrier(self):
        """The source holds 365 days: reading a 29 February would return an empty bulletin."""
        produits = {
            avancer_date(mois, jour, n) for mois, jour in JOURS_ANNEE for n in range(61)
        }
        assert (2, 29) not in produits

    def test_A6_l_heure_murale_du_depart_est_conservee(self):
        ts = _ts(2026, 3, 20, heure=17, minute=42)
        lu = wall_clock(
            timestamp_meteo(ts, "609", GRAINE, JOURS_ANNEE, jours_ecoules=4)
        )
        assert (lu.hour, lu.minute) == (17, 42)


class TestAncre:
    def test_A7_sans_ancre_on_se_comporte_comme_au_premier_jour(self):
        assert ancre_run.jours_ecoules(_ts(2026, 5, 1)) == 0

    def test_A7bis_l_ancre_compte_des_jours_de_calendrier(self):
        ancre_run.ancrer(_ts(2026, 3, 16, heure=5))
        # 5 a.m. the next day: the day has changed, even though 24 h have not elapsed.
        assert ancre_run.jours_ecoules(_ts(2026, 3, 17, heure=4)) == 1
        assert ancre_run.jours_ecoules(_ts(2026, 3, 16, heure=23)) == 0
        assert ancre_run.jours_ecoules(_ts(2026, 5, 14, heure=5)) == 59

    def test_A7ter_un_timestamp_anterieur_ne_rend_jamais_de_jour_negatif(self):
        ancre_run.ancrer(_ts(2026, 3, 16))
        assert ancre_run.jours_ecoules(_ts(2026, 3, 10)) == 0

    def test_A8_l_ancre_restauree_n_est_pas_ecrasee_par_le_rejeu(self):
        """On resume, GAMA replays from its t0: re-anchoring would rewind the weather."""
        ancre_run.ancrer(_ts(2026, 4, 1), origine="point de reprise")
        ancre_run.ancrer(_ts(2026, 3, 16))  # what the replay makes us observe
        assert ancre_run.ancre() == _ts(2026, 4, 1)
