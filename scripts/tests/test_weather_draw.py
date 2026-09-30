"""
Unit tests of drawing a weather date per agent
(`services/llm-agents/urban_mobility_agents/utils/weather_draw.py`).

The mechanism exists because, on a single simulated day, the 1,000 agents
share a single weather: the regressor has zero variance, and "no effect
measured" then means nothing. Each test locks a
property without which the mechanism would be wrong rather than absent:

  - **Determinism.** Same seed, same agent → same day. Without it, two identical
    runs no longer reproduce and an archived trace cannot be replayed.
  - **The departure time is kept.** The bulletin is read in 3 h slots:
    a departure at 08:00 must keep reading the 06:00 record, whatever
    the drawn date. Substituting the whole timestamp would break the slot.
  - **The window is respected**, weekends excluded when the survey requires it — it
    counts only working days.
  - **A window straddling New Year works**: the one of the
    EMC² survey runs from 20 September to 18 February.
  - **Coverage**: 1,000 agents must reach all the eligible
    days, otherwise the mechanism loses the variance it is meant to bring.
  - **Non-regression**: flag false → behaviour strictly identical to
    the existing one.

No network access, no LLM.
"""

from __future__ import annotations

import calendar
import collections
import datetime as dt
import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "llm-agents"))

from sim_clock import wall_clock  # noqa: E402
from urban_mobility_agents.utils.weather_draw import (  # noqa: E402
    date_meteo,
    indice_agent,
    jours_eligibles,
    timestamp_meteo,
)


def mur(annee, mois, jour, heure=0, minute=0, seconde=0) -> int:
    """GAMA timestamp of a WALL-CLOCK time — the inverse of `sim_clock.wall_clock`.

    ⚠ Not `datetime(...).timestamp()`: that one reads the time in the time zone of the
    PROCESS, and a test built on it passes under `TZ=UTC` while failing under
    `TZ=Europe/Paris` (or the reverse depending on the season).
    """
    return calendar.timegm(dt.datetime(annee, mois, jour, heure, minute, seconde).timetuple())

FENETRE_ENQUETE = ("2022-09-20", "2023-02-18")
JOURS_OUVRES = (1, 2, 3, 4, 5)


class TestJoursEligibles(unittest.TestCase):
    def test_fenetre_a_cheval_sur_le_nouvel_an(self):
        jours = jours_eligibles(*FENETRE_ENQUETE, JOURS_OUVRES)
        self.assertEqual(len(jours), 109)
        # September and February must both be present: a window
        # treated as an integer interval would lose one of its two ends.
        mois = {m for m, _ in jours}
        self.assertEqual(mois, {9, 10, 11, 12, 1, 2})

    def test_week_ends_exclus(self):
        jours = jours_eligibles(*FENETRE_ENQUETE, JOURS_OUVRES)
        # Re-check on the actual dates of the window, not on the pairs.
        premier = dt.date.fromisoformat(FENETRE_ENQUETE[0])
        dernier = dt.date.fromisoformat(FENETRE_ENQUETE[1])
        attendus = set()
        jour = premier
        while jour <= dernier:
            if jour.isoweekday() in JOURS_OUVRES:
                attendus.add((jour.month, jour.day))
            jour += dt.timedelta(days=1)
        self.assertEqual(set(jours), attendus)

    def test_sans_filtre_toute_lannee(self):
        # Leap pivot year: 29 February must be drawable.
        jours = jours_eligibles("2024-01-01", "2024-12-31", None)
        self.assertEqual(len(jours), 366)
        self.assertIn((2, 29), jours)

    def test_fenetre_vide_refusee(self):
        with self.assertRaises(ValueError):
            jours_eligibles("2026-03-10", "2026-03-01", None)

    def test_fenetre_sans_jour_eligible_refusee(self):
        # A weekend alone, filtered on working days: no day remains.
        with self.assertRaises(ValueError):
            jours_eligibles("2026-03-21", "2026-03-22", JOURS_OUVRES)

    def test_pas_de_doublon(self):
        jours = jours_eligibles("2024-01-01", "2024-12-31", None)
        self.assertEqual(len(jours), len(set(jours)))


class TestTirage(unittest.TestCase):
    def setUp(self) -> None:
        self.jours = jours_eligibles(*FENETRE_ENQUETE, JOURS_OUVRES)

    def test_deterministe(self):
        for person_id in ("p1", "agent-42", "toulouse_000123"):
            self.assertEqual(
                date_meteo(person_id, 42, self.jours),
                date_meteo(person_id, 42, self.jours),
            )

    def test_graine_differente_tirage_different(self):
        # Not a per-agent guarantee, but the set of assignments must move.
        a = [date_meteo(f"p{i}", 42, self.jours) for i in range(200)]
        b = [date_meteo(f"p{i}", 43, self.jours) for i in range(200)]
        self.assertNotEqual(a, b)

    def test_toujours_dans_la_fenetre(self):
        autorises = set(self.jours)
        for i in range(2000):
            self.assertIn(date_meteo(f"p{i}", 42, self.jours), autorises)

    def test_mille_agents_couvrent_toutes_les_journees(self):
        tirages = collections.Counter(date_meteo(f"p{i}", 42, self.jours) for i in range(1000))
        self.assertEqual(
            len(tirages), len(self.jours),
            "toutes les journées éligibles doivent être atteintes, sinon la variance promise n'est pas là",
        )
        self.assertGreaterEqual(min(tirages.values()), 1)

    def test_indice_borne(self):
        for cardinal in (1, 2, 109, 366):
            for i in range(50):
                self.assertTrue(0 <= indice_agent(42, f"p{i}", cardinal) < cardinal)
        with self.assertRaises(ValueError):
            indice_agent(42, "p0", 0)


class TestTimestamp(unittest.TestCase):
    """⚠ These tests speak in WALL-CLOCK time, the only convention of GAMA's clock.

    The input timestamps are built by :func:`mur` (`calendar.timegm`) and
    read back by :func:`sim_clock.wall_clock`, never by `datetime(...).timestamp()` /
    `datetime.fromtimestamp(...)`: that pair goes through the PROCESS time zone and
    cancelled out by luck in winter, not in summer — this is what left the offset
    invisible until 2026-09-04.
    """

    def setUp(self) -> None:
        self.jours = jours_eligibles(*FENETRE_ENQUETE, JOURS_OUVRES)

    def test_heure_du_depart_conservee(self):
        """The bulletin is read in 3 h slots: the time must not move."""
        for heure in (5, 8, 12, 17, 21, 23):
            depart = mur(2026, 3, 16, heure, 37, 12)
            for person_id in ("p1", "p2", "p3", "p400"):
                obtenu = wall_clock(timestamp_meteo(depart, person_id, 42, self.jours))
                self.assertEqual((obtenu.hour, obtenu.minute, obtenu.second), (heure, 37, 12))

    def test_seule_la_date_change(self):
        depart = mur(2026, 3, 16, 8, 0, 0)
        attendu = date_meteo("p7", 42, self.jours)
        obtenu = wall_clock(timestamp_meteo(depart, "p7", 42, self.jours))
        self.assertEqual((obtenu.month, obtenu.day), attendu)

    def test_le_meme_agent_garde_sa_journee_quelle_que_soit_lheure(self):
        """An agent's weather must not depend on its departure time:
        otherwise its morning and evening trips would live two different days."""
        matin = mur(2026, 3, 16, 8, 0, 0)
        soir = mur(2026, 3, 16, 18, 30, 0)
        a = wall_clock(timestamp_meteo(matin, "p9", 42, self.jours))
        b = wall_clock(timestamp_meteo(soir, "p9", 42, self.jours))
        self.assertEqual((a.month, a.day), (b.month, b.day))

    def test_heure_conservee_au_franchissement_ete_hiver(self):
        """The drawn date may be from another season than the simulated day: the
        WALL-CLOCK time must be kept to the second in both directions.

        The survey window (20/09 → 18/02) crosses the summer/winter time switch.
        The old version went through instants (`fromtimestamp(tz=...)` then
        `.timestamp()`), and therefore had to reason on UTC offsets. GAMA's clock
        has no switch: in wall-clock fields preservation is exact, and there
        is no longer any offset to recompute.
        """
        for depart, jours_cibles in (
            (mur(2026, 9, 22, 8, 0, 0), [(12, 8)]),    # summer day → winter weather
            (mur(2026, 12, 8, 8, 0, 0), [(9, 22)]),    # winter day → summer weather
            (mur(2026, 3, 29, 2, 30, 0), [(12, 8)]),   # wall-clock time NONEXISTENT in France
        ):
            obtenu = wall_clock(timestamp_meteo(depart, "p-dst", 42, jours_cibles))
            attendu_mois, attendu_jour = jours_cibles[0]
            depart_mur = wall_clock(depart)
            self.assertEqual(
                (obtenu.month, obtenu.day, obtenu.hour, obtenu.minute, obtenu.second),
                (attendu_mois, attendu_jour, depart_mur.hour, depart_mur.minute,
                 depart_mur.second))

    def test_independant_du_fuseau_du_processus(self):
        """The same departure must return the same bulletin under any `TZ`.

        The `controller` runs in `TZ=Europe/Paris` and the `osmnx` replicas in
        `TZ=UTC`: a weather that depends on the process time zone is not reproducible
        from one container to another, and an archived trace can no longer be replayed.
        """
        depart = mur(2026, 3, 16, 8, 37, 12)
        obtenus = {}
        initial = os.environ.get("TZ")
        try:
            for tz in ("UTC", "Europe/Paris", "Pacific/Kiritimati", "America/Los_Angeles"):
                os.environ["TZ"] = tz
                time.tzset()
                obtenus[tz] = [timestamp_meteo(depart, f"p{i}", 42, self.jours)
                               for i in range(50)]
        finally:
            if initial is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = initial
            time.tzset()

        distincts = {tuple(v) for v in obtenus.values()}
        self.assertEqual(len(distincts), 1,
                         "le tirage météo dépend du fuseau du processus")


class TestVarianceLiberee(unittest.TestCase):
    """The test that says why the mechanism exists.

    On the simulated day alone, the weather is a constant. The draw must
    produce a real spread, otherwise it is useless.
    """

    def setUp(self) -> None:
        try:
            from urban_mobility_agents.utils.weather_loader import get_weather
        except Exception as err:  # pragma: no cover
            self.skipTest(f"chargeur météo indisponible : {err}")
        self.get_weather = get_weather
        self.jours = jours_eligibles(*FENETRE_ENQUETE, JOURS_OUVRES)
        self.depart = mur(2026, 3, 16, 8, 0, 0)
        if self.get_weather(self.depart) is None:
            self.skipTest("données météo absentes du dépôt")

    def test_variance_nulle_sans_dispositif(self):
        temperatures = {self.get_weather(self.depart)["temperature"] for _ in range(100)}
        self.assertEqual(len(temperatures), 1, "sans dispositif, une seule météo pour tous")

    def test_dispersion_avec_dispositif(self):
        temperatures, precipitants = [], 0
        for i in range(500):
            meteo = self.get_weather(timestamp_meteo(self.depart, f"p{i}", 42, self.jours))
            if meteo is None:
                continue
            temperatures.append(meteo["temperature"])
            precipitants += meteo["precip_mm"] > 0
        self.assertGreater(len(set(temperatures)), 20)
        self.assertGreater(max(temperatures) - min(temperatures), 15)
        self.assertGreater(precipitants, 50, "une part substantielle doit voir de la pluie")


class TestNonRegression(unittest.TestCase):
    """Flag false → the simulated clock, unchanged.

    ⚠ **"By default" means two things, and confusing them made these
    tests fail for a day** (2026-09-04). The CODE DEFAULT is
    `Settings.weather_per_agent_dates = False`: nothing moves unless someone
    asks for it. The REPO CONFIGURATION (`services/llm-agents/config/config.yaml`) no
    longer carries the key since 2026-09-19: as long as it was there, it prevailed over
    the environment, and an arm could not get the per-agent weather. The mechanism
    is now enabled by `WEATHER_PER_AGENT_DATES`, run by run. Each assertion is
    checked here at its source.
    """

    def test_defaut_du_code_desactive(self):
        """The CODE default: nothing moves without explicit intent."""
        from settings import settings

        champ = type(settings.agent).model_fields["weather_per_agent_dates"]
        self.assertFalse(
            champ.default,
            "le défaut du code doit rester faux : rien ne bouge sans intention",
        )

    def test_la_configuration_du_depot_ne_fige_pas_le_dispositif(self):
        """`config.yaml` does not set `weather_per_agent_dates`: the environment decides.

        Deliberate removal of 2026-09-19 (documented in `config.yaml`): the YAML passed at
        initialisation takes precedence over the environment, so a key set here would prevent an arm
        from enabling per-agent weather with `WEATHER_PER_AGENT_DATES`. If this test fails, the key
        has come back into the repo file.
        """
        import yaml

        chemin = REPO_ROOT / "services" / "llm-agents" / "config" / "config.yaml"
        brut = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
        agent = brut.get("agent") or {}
        self.assertNotIn(
            "weather_per_agent_dates",
            agent,
            f"{chemin} ne doit pas poser weather_per_agent_dates : il masquerait "
            "WEATHER_PER_AGENT_DATES",
        )

    def test_agent_retombe_sur_lhorloge_simulee(self):
        """Flag false: `_weather_timestamp` returns the simulated clock, as is.

        The flag is forced here, as the symmetric test does: what is under
        test is the BRANCHING, not the value the run configuration carries.

        A context stand-in rather than a full `Person`: what is under test
        is the branching, not the validation of the person model — and a test
        that breaks when a field of `PersonalIdentity` moves says nothing useful
        about the weather.
        """
        from types import SimpleNamespace

        from settings import settings
        from urban_mobility_agents.agents.llm_agent import LlmAgent

        contexte = SimpleNamespace(
            timestamp=1773648000, person=SimpleNamespace(person_id="p1")
        )
        # Instance without constructor: `_weather_timestamp` delegates to `self._weather_info`,
        # a `self` set to None is no longer enough.
        agent = LlmAgent.__new__(LlmAgent)
        initial = settings.agent.weather_per_agent_dates
        settings.agent.weather_per_agent_dates = False
        try:
            self.assertEqual(agent._weather_timestamp(contexte), 1773648000)
        finally:
            settings.agent.weather_per_agent_dates = initial

    def test_agent_tire_une_date_quand_le_dispositif_est_actif(self):
        """Flag true: the date changes, the WALL-CLOCK departure time does not."""
        from types import SimpleNamespace

        from settings import settings
        from urban_mobility_agents.agents.llm_agent import LlmAgent, _weather_eligible_days

        depart = mur(2026, 3, 16, 8, 0, 0)
        contexte = SimpleNamespace(timestamp=depart, person=SimpleNamespace(person_id="p1"))
        agent = LlmAgent.__new__(LlmAgent)
        initial = settings.agent.weather_per_agent_dates
        settings.agent.weather_per_agent_dates = True
        try:
            obtenu = agent._weather_timestamp(contexte)
        finally:
            settings.agent.weather_per_agent_dates = initial

        tire = wall_clock(obtenu)
        reference = wall_clock(depart)
        self.assertEqual((tire.hour, tire.minute), (reference.hour, reference.minute))
        # The window is the one the agent actually reads (`config.yaml`: the whole year,
        # weekends included), not the survey window of the pure tests above.
        self.assertIn((tire.month, tire.day), set(_weather_eligible_days()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
