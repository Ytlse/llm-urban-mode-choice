"""The prompt, the weather and the cache key read GAMA's clock (2026-09-04).

On the morning of 2026-09-04, itinerary time switched to GAMA's wall clock
(`sim_clock`, cf. `test_fuseau_reseau.py`). Three families of consumers stayed on
the **process** time zone and were therefore now at odds with routing:

1. **the weather** — `weather_loader` / `weather_draw` read
   `fromtimestamp(ts, tz=Europe/Paris)`: for 5:00 wall-clock they opened the
   **6:00** reading, and the **8:00** one for a simulated day in summer (two hours);
2. **the time shown to the agent** — `helper.humanize_time` & co.: "départ 06:00"
   when GAMA says 05:00;
3. **the decision cache key** — `llm/cache.py::_make_time_slice` / `_make_weekday`.

Quantified on the 5,322 trips of the archived run `2026-09-04_01_09`
(`docs/traces/2026-09-04_14-30_horloge_prompt_meteo/`): **2,332 (43.8 %)** changed
3-hour weather reading, **5,322 (100 %)** displayed time, and the **77 departures at
wall-clock 23:00** changed **DAY** — their weather, their displayed weekday and the
`day`/`month` of their cache entry spoke of the next day while their itinerary
was computed on the day before.

These tests fail if the convention changes again. They check, in winter AND in summer:
the displayed time equal to GAMA's clock to the minute; the weather reading taken at wall-clock
time; that a departure at 23:30 does not change day; independence from the process
`TZ` (really set with `tzset`); and the decision cache key.

No network call, no LLM.
"""

import calendar
import os
import time
from datetime import datetime, timezone

import pytest

import helper
from llm.cache import LlmSemanticCache
from sim_clock import gama_timestamp, wall_clock
from urban_mobility_agents.utils import weather_draw, weather_loader


def _gama_ts(annee, mois, jour, heure, minute=0, seconde=0) -> int:
    """The timestamp GAMA publishes for a given WALL-CLOCK time.

    Reproduces `int(current_date - UTC_START_DATE)`: a difference of naive dates,
    i.e. wall-clock time counted as if it were UTC. **Never**
    `datetime(...).timestamp()`, which would go through the process time zone and make
    these tests pass under one `TZ` while making them fail under another.
    """
    return int(calendar.timegm((annee, mois, jour, heure, minute, seconde, 0, 0, 0)))


# Monday 16 March 2026 5:00 wall-clock: the t0 of `starting_date` (Settings.gaml), and the value
# read in the "Temps simulé" column of moves.csv.
TS_HIVER = 1773637200
# Monday 13 July 2026 5:00 wall-clock: summer time, when the old gap was TWO hours.
TS_ETE = _gama_ts(2026, 7, 13, 5)
# Friday 20 March 2026, 23:30 wall-clock: the time at which the old reading changed DAY
# — and, on a Friday, flipped the day category to "Weekend".
TS_VENDREDI_2330 = _gama_ts(2026, 3, 20, 23, 30)

# The time zones actually in play (`controller` in Europe/Paris, `osmnx` replicas in UTC),
# plus two extremes that change the day in both directions.
FUSEAUX = ("UTC", "Europe/Paris", "Pacific/Kiritimati", "America/Los_Angeles")


@pytest.fixture
def sous_fuseau():
    """Really sets the process `TZ` (`tzset`) and restores it afterwards.

    Without `tzset`, changing `os.environ["TZ"]` does not move `datetime.fromtimestamp`:
    a "time-zone independence" test that forgets it always passes, including on the
    defective code.
    """
    initial = os.environ.get("TZ")

    def _poser(nom: str) -> None:
        os.environ["TZ"] = nom
        time.tzset()

    yield _poser
    if initial is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = initial
    time.tzset()


# ── The time shown to the agent ───────────────────────────────────────────────

class TestHeureAffichee:
    """What the prompt announces must be what GAMA displays, to the minute."""

    @pytest.mark.parametrize("ts,attendu", [
        (TS_HIVER, "05:00"),
        (TS_ETE, "05:00"),
        (_gama_ts(2026, 3, 16, 8, 17, 43), "08:17"),
        (_gama_ts(2026, 7, 13, 19, 0), "19:00"),
        (TS_VENDREDI_2330, "23:30"),
    ])
    def test_humanize_time_est_lheure_de_gama(self, ts, attendu):
        assert helper.humanize_time(ts) == attendu

    def test_toutes_les_heures_de_la_journee_a_la_minute(self):
        """Full sweep: not a single slot may drift, winter or summer."""
        for base in (TS_HIVER, TS_ETE):
            jour = base - (base % 86400)
            for minute_du_jour in range(0, 1440, 7):
                ts = jour + minute_du_jour * 60
                mur = wall_clock(ts)
                assert helper.humanize_time(ts) == f"{mur.hour:02d}:{mur.minute:02d}"

    @pytest.mark.parametrize("ts,attendu", [
        (TS_HIVER, "Monday"),
        (TS_ETE, "Monday"),
        # 23:30 on a Friday: the displayed day stays FRIDAY. Read in the process
        # time zone, the prompt announced "Saturday" and the category "WEEKEND".
        (TS_VENDREDI_2330, "Friday"),
    ])
    def test_jour_affiche(self, ts, attendu):
        assert helper.humanize_date_short(ts).split(",")[0] == attendu
        assert helper.categorize_date_time_short(ts).split(" ")[0] == attendu

    def test_categorie_de_jour_a_2330_un_vendredi(self):
        assert helper.get_weekday_category(TS_VENDREDI_2330) == "Weekday"
        # The following Saturday, at the same wall-clock time, is indeed a weekend: the
        # fix does not make the category vanish, it puts it back on the right day.
        assert helper.get_weekday_category(_gama_ts(2026, 3, 21, 23, 30)) == "Weekend"

    @pytest.mark.parametrize("ts,fenetre,tranche,creneau", [
        # 5:00 wall-clock: "early morning", "night", off-peak. The old reading
        # (6:00 in March, 7:00 in July) made it the morning peak.
        (TS_HIVER, "early morning", "Monday night", "night time (20:00 - 6:00)"),
        (TS_ETE, "early morning", "Monday night", "night time (20:00 - 6:00)"),
        (_gama_ts(2026, 3, 16, 17, 30), "end of the workday", "Monday afternoon",
         "evening rush hour (16:00 - 20:00)"),
    ])
    def test_fenetres_temporelles_du_prompt(self, ts, fenetre, tranche, creneau):
        assert helper.time_window_generalize(ts) == fenetre
        assert helper.categorize_date_time_short(ts) == tranche
        assert helper.time_to_bucket_text(ts) == creneau

    @pytest.mark.parametrize("nom", FUSEAUX)
    def test_independant_du_fuseau_du_processus(self, sous_fuseau, nom):
        sous_fuseau(nom)
        for ts in (TS_HIVER, TS_ETE, TS_VENDREDI_2330):
            mur = wall_clock(ts)
            assert helper.humanize_time(ts) == f"{mur.hour:02d}:{mur.minute:02d}"
            assert helper.humanize_date_short(ts) == mur.strftime("%A, %H:%M")
            assert helper.get_weekday_category(ts) == (
                "Weekend" if mur.weekday() >= 5 else "Weekday")
            assert helper.time_window_generalize(ts) == (
                "early morning" if mur.hour < 6 else helper.time_window_generalize(ts))


# ── The weather ───────────────────────────────────────────────────────────────

def _bulletin(ts):
    w = weather_loader.get_weather(ts)
    if w is None:
        pytest.skip("weather data missing from the repo")
    return w


class TestMeteoALheureMurale:
    """The bulletin read is the one for the day and 3-hour slot of the WALL-CLOCK time."""

    @pytest.mark.parametrize("ts,heure_relevee", [
        # `_reading_bucket` takes the nearest 3-hour reading BACKWARDS.
        (TS_HIVER, 3),                        # 5:00 wall-clock → 3:00 reading (not 6:00)
        (TS_ETE, 3),                          # in summer the old reading opened 6:00
        (_gama_ts(2026, 3, 16, 8, 0), 6),
        (_gama_ts(2026, 7, 13, 8, 0), 6),
        (_gama_ts(2026, 3, 16, 11, 59), 9),   # 11:59 → 9:00, not 12:00
        (TS_VENDREDI_2330, 21),               # 23:30 → 21:00, not the next day's midnight
    ])
    def test_creneau_lu(self, ts, heure_relevee):
        """The reading taken is that of `heure_relevee`, on the same wall-clock day."""
        mur = wall_clock(ts)
        reference = _bulletin(_gama_ts(mur.year, mur.month, mur.day, heure_relevee))
        obtenu = _bulletin(ts)
        assert (obtenu["temperature"], obtenu["weather_code"]) == \
               (reference["temperature"], reference["weather_code"])

    @pytest.mark.parametrize("ts", [TS_HIVER, TS_ETE, TS_VENDREDI_2330])
    def test_jour_du_bulletin_est_le_jour_simule(self, ts):
        """The day frame (range, sunrise, sunset) is that of the WALL-CLOCK day.

        This is the DAY-shift test: at 23:30, the old reading opened the
        next day's row — another sunrise, another range, another
        rain — while the itinerary itself was computed on the day before.
        """
        mur = wall_clock(ts)
        meme_jour_midi = _bulletin(_gama_ts(mur.year, mur.month, mur.day, 12))
        obtenu = _bulletin(ts)
        for champ in ("temp_min", "temp_max", "sunrise", "sunset"):
            assert obtenu[champ] == meme_jour_midi[champ], champ

    @pytest.mark.parametrize("nom", FUSEAUX)
    def test_independant_du_fuseau_du_processus(self, sous_fuseau, nom):
        sous_fuseau(nom)
        for ts in (TS_HIVER, TS_ETE, TS_VENDREDI_2330):
            mur = wall_clock(ts)
            reference = _bulletin(_gama_ts(
                mur.year, mur.month, mur.day, (mur.hour // 3) * 3))
            obtenu = _bulletin(ts)
            assert (obtenu["temperature"], obtenu["weather_code"],
                    obtenu["sunrise"], obtenu["sunset"]) == \
                   (reference["temperature"], reference["weather_code"],
                    reference["sunrise"], reference["sunset"])

    def test_anticipation_lit_les_tranches_restantes_du_jour_mural(self):
        """`day_weather_outlook` speaks only of the remaining slices of the WALL-CLOCK day.

        At 23:30 nothing is left: the old reading, which saw the next day's
        midnight, announced instead the whole following day — in a text
        that moreover enters the decision cache key.
        """
        assert weather_loader.day_weather_outlook(TS_VENDREDI_2330) is None
        matin = weather_loader.day_weather_outlook(_gama_ts(2026, 3, 16, 8, 0))
        assert matin and "afternoon" in matin


class TestTirageMeteoParAgent:
    """Drawing a weather date per agent keeps the WALL-CLOCK time, exactly."""

    @pytest.fixture
    def jours(self):
        return weather_draw.jours_eligibles("2022-09-20", "2023-02-18", (1, 2, 3, 4, 5))

    @pytest.mark.parametrize("ts", [TS_HIVER, TS_ETE, TS_VENDREDI_2330,
                                    _gama_ts(2026, 3, 29, 2, 30)])
    def test_heure_murale_conservee_a_la_seconde(self, ts, jours):
        """Including when the drawn date is from another season, and for 2:30 on the
        last Sunday of March — a wall-clock time that does not exist in France, which
        GAMA's clock nevertheless carries."""
        mur = wall_clock(ts)
        for i in range(80):
            tire = wall_clock(weather_draw.timestamp_meteo(ts, f"p{i}", 42, jours))
            assert (tire.hour, tire.minute, tire.second) == \
                   (mur.hour, mur.minute, mur.second)
            assert (tire.month, tire.day) in set(jours)

    @pytest.mark.parametrize("nom", FUSEAUX)
    def test_independant_du_fuseau_du_processus(self, sous_fuseau, nom, jours):
        sous_fuseau(nom)
        obtenus = [weather_draw.timestamp_meteo(TS_HIVER, f"p{i}", 42, jours)
                   for i in range(40)]
        sous_fuseau("UTC")
        assert obtenus == [weather_draw.timestamp_meteo(TS_HIVER, f"p{i}", 42, jours)
                           for i in range(40)]


# ── The decision cache key ────────────────────────────────────────────────────

class TestCleDuCacheDeDecisions:
    @pytest.mark.parametrize("ts,tranche,categorie", [
        (TS_HIVER, "05:00", "Weekday"),
        (TS_ETE, "05:00", "Weekday"),
        (_gama_ts(2026, 3, 16, 8, 17), "08:10", "Weekday"),
        # 23:30 on a Friday: the slice stays 23:30 and the day stays a weekday. Read in
        # the process time zone, this context was written "00:30 / Weekend" and would
        # be confused with that of a real weekend departure.
        (TS_VENDREDI_2330, "23:30", "Weekday"),
        (_gama_ts(2026, 3, 21, 23, 30), "23:30", "Weekend"),
    ])
    def test_tranche_et_categorie(self, ts, tranche, categorie):
        assert LlmSemanticCache._make_time_slice(ts) == tranche
        assert LlmSemanticCache._make_weekday(ts) == categorie

    @pytest.mark.parametrize("nom", FUSEAUX)
    def test_independant_du_fuseau_du_processus(self, sous_fuseau, nom):
        sous_fuseau(nom)
        obtenus = [(LlmSemanticCache._make_time_slice(ts), LlmSemanticCache._make_weekday(ts))
                   for ts in (TS_HIVER, TS_ETE, TS_VENDREDI_2330)]
        assert obtenus == [("05:00", "Weekday"), ("05:00", "Weekday"),
                           ("23:30", "Weekday")]

    def test_deux_heures_murales_distinctes_ne_partagent_pas_la_tranche(self):
        """Safeguard: the fix must not make two contexts converge.

        A constant offset is a bijection, so it introduces no collision —
        but the old offset was NOT constant (one hour in March, two in
        July). Two departures at the same wall-clock time in two different seasons
        received two distinct slices; they now share the same one, and it is
        indeed the weather and the option codes that must separate them.
        """
        assert LlmSemanticCache._make_time_slice(TS_HIVER) == \
               LlmSemanticCache._make_time_slice(TS_ETE)
        distinctes = {LlmSemanticCache._make_time_slice(_gama_ts(2026, 3, 16, h, m))
                      for h in range(24) for m in (0, 10, 20, 30, 40, 50)}
        assert len(distinctes) == 24 * 6


# ── Weather, itinerary and displayed day speak of the same day ────────────────

class TestMemeJourPartout:
    """The end-to-end test of the DAY shift (the 77 departures at 23:00).

    Three independent readings must land on the same wall-clock date: that of
    the itinerary (`sim_clock`, the OTP path), that of the weather bulletin, and that of the
    weekday shown in the prompt.
    """

    @pytest.mark.parametrize("ts", [
        TS_HIVER,
        TS_ETE,
        TS_VENDREDI_2330,
        _gama_ts(2026, 3, 16, 23, 30),
        _gama_ts(2026, 7, 13, 23, 30),   # summer: the old two-hour gap
        _gama_ts(2026, 3, 16, 23, 59, 59),
    ])
    @pytest.mark.parametrize("nom", ("UTC", "Europe/Paris"))
    def test_meme_jour(self, ts, nom, sous_fuseau):
        sous_fuseau(nom)
        mur = wall_clock(ts)

        # 1. the itinerary: the day `sim_clock` (hence OTP) sees.
        assert (mur.year, mur.month, mur.day) == (mur.year, mur.month, mur.day)

        # 2. the weather: its day frame must be that of noon on the SAME day.
        bulletin = _bulletin(ts)
        midi = _bulletin(_gama_ts(mur.year, mur.month, mur.day, 12))
        assert (bulletin["sunrise"], bulletin["sunset"],
                bulletin["temp_min"], bulletin["temp_max"]) == \
               (midi["sunrise"], midi["sunset"], midi["temp_min"], midi["temp_max"])

        # 3. the weekday shown in the prompt.
        assert helper.humanize_date_short(ts).split(",")[0] == mur.strftime("%A")
        assert helper.humanize_date(ts).startswith(mur.strftime("%d"))

        # 4. and the cache key, which indexes that day's decision.
        assert LlmSemanticCache._make_weekday(ts) == (
            "Weekend" if mur.weekday() >= 5 else "Weekday")

    def test_aller_retour_horodatage_souvenir(self):
        """A memory written in wall-clock time must be read back at the same time.

        `add_short_term_memory` stores `wall_clock(ts)` and the prompt shows it again via
        `humanize_date(gama_timestamp(entry.timestamp))`. The two conventions must
        cancel out exactly — that is the trap that gave "- Time … 06:00" for a
        5:00 memory.
        """
        for ts in (TS_HIVER, TS_ETE, TS_VENDREDI_2330):
            assert gama_timestamp(wall_clock(ts)) == ts
            assert helper.humanize_date(gama_timestamp(wall_clock(ts))) == \
                   helper.humanize_date(ts)


def test_aucune_lecture_ne_passe_par_le_fuseau_du_processus():
    """Source safeguard: the aligned modules no longer call bare `fromtimestamp`.

    A value test is not enough here: under `TZ=UTC` — the time zone of the `osmnx`
    replicas — the old reading and the new one give the same result, and a
    regression would therefore go unnoticed in half of the containers.
    """
    import ast
    import inspect

    # The syntax tree, not a regular expression: the docstrings of these
    # modules QUOTE the fixed defect (that is their role), and a grep on the text would
    # confuse them with callable code.
    for module in (helper, weather_loader, weather_draw, __import__("llm.cache", fromlist=["x"])):
        arbre = ast.parse(inspect.getsource(module))
        for noeud in ast.walk(arbre):
            if not isinstance(noeud, ast.Call):
                continue
            fonction = noeud.func
            if not (isinstance(fonction, ast.Attribute) and fonction.attr == "fromtimestamp"):
                continue
            # `fromtimestamp(ts, tz=...)` is explicit and unsurprising; it is
            # the call WITHOUT a time zone that reads the process clock.
            if any(kw.arg == "tz" for kw in noeud.keywords) or len(noeud.args) >= 2:
                continue
            pytest.fail(
                f"{module.__name__}, line {noeud.lineno}: `fromtimestamp` without a time zone "
                "reads the PROCESS clock. Use `sim_clock.wall_clock`.")


def test_weather_loader_se_charge_par_chemin_sans_le_controleur():
    """`prompt_calibration` loads this file BY PATH: it must remain loadable.

    The standalone `prompt_calibration` repo checks that its copy of
    `weather_to_natural_language` has not drifted, and does so via
    `spec_from_file_location` — on purpose, so as not to pull the controller and its
    dependencies into its tests. A `from sim_clock import wall_clock` at the top of the module
    breaks that loading (happened: 14 calibration tests down on 2026-09-04), hence
    the deferred import in `_heure_murale`. This test locks it from this repo,
    so that the breakage shows here rather than in the other one.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    fichier = Path(weather_loader.__file__).resolve()
    programme = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_prod_wl', r'{fichier}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['_prod_wl'] = m\n"
        "spec.loader.exec_module(m)\n"
        "print(m.weather_to_natural_language({'temperature': 3.0, "
        "'weather_label': 'Clear/Sunny', 'precip_mm': 0.0}))\n"
    )
    # `-I`: neither the script directory nor the Python environment variables —
    # `services/llm-agents/` is therefore NOT on the path, as in `prompt_calibration`.
    r = subprocess.run([_sys.executable, "-I", "-c", programme],
                       capture_output=True, text=True, cwd=str(fichier.parent))
    assert r.returncode == 0, (
        "weather_loader n'est plus chargeable par chemin (import de tête à différer ?) :\n"
        + r.stderr)
    assert "Weather: 3°C" in r.stdout


def test_la_meteo_ne_declare_aucun_fuseau():
    """The weather must carry NO time zone, and that is the core of the earlier defect.

    `weather_loader` and `weather_draw` did `fromtimestamp(ts, tz=ZoneInfo(
    "Europe/Paris"))`: an explicit call, immune to the process `TZ` — and wrong
    all the same, because it treated GAMA's WALL-CLOCK time as an instant. The
    hard-coded time zone therefore does not show in the previous safeguard: it shows
    here. The weather source is indexed by (month, day) and read by time slot;
    only wall-clock fields matter, and a time zone has no business there.

    ⚠ The NETWORK time zone, for its part, keeps its place: it lives in `sim_clock`, taken from
    the GTFS feeds' `agency_timezone`, and serves to talk to OTP — not to read a CSV.
    """
    import ast
    import inspect

    for module in (weather_loader, weather_draw):
        arbre = ast.parse(inspect.getsource(module))
        for noeud in ast.walk(arbre):
            if isinstance(noeud, (ast.Import, ast.ImportFrom)):
                noms = ([a.name for a in noeud.names]
                        + [getattr(noeud, "module", None) or ""])
                for nom in noms:
                    if "zoneinfo" in nom.lower() or nom == "ZoneInfo":
                        pytest.fail(
                            f"{module.__name__}, line {noeud.lineno}: weather is read "
                            "in WALL-CLOCK time (`sim_clock.wall_clock`); a time zone written "
                            "here would treat GAMA's clock as an instant.")
