"""
One weather date per agent — the instrument that makes the weather effect measurable.

WHY
---
Ticket 023 measured the enriched weather forecast "at full mass" and concluded
there was no effect, its grid noting "no conclusion on rain, the Δ changes sign
between substrates". The cause is instrumental, not substantive: **over a
single simulated day, the 1,000 agents share a single weather.** The regressor
has zero variance — no effect is detectable, whatever the truth.

This module draws, for each agent, a day of the year within a declared window,
and substitutes only the DATE of the forecast: the time of day is kept
because the forecast is read in 3-hour slots (`weather_loader._reading_bucket`).
Everything else in the simulation — GTFS timetables, vehicles, itineraries,
agendas — stays on the simulated day. It is a *ceteris paribus* device: only the
weather moves.

WHAT THIS FREES UP
------------------
Over the 365 days of `data/weather/meteo_toulouse_12_mois.csv`: morning temperature
from −4 to +23 °C (standard deviation 5.1) and 155 days with precipitation, 16 of them
above 5 mm. Over the survey's collection window alone (`2022-09-20 → 2023-02-18`,
working days): 109 usable days.

THE YEAR IS IGNORED, AND THAT IS INTENDED
-----------------------------------------
`weather_loader.get_weather` indexes by (month, day): only the day of the year
counts. The targeted SEASON is thus matched to the weather available, not to the
historical days the respondents lived — a correct seasonal distribution, not a
reconstruction.

DETERMINISM
-----------
The draw is a pure function of `(graine, person_id)`. Two identical runs produce
exactly the same weathers, like the mode draw seed. It uses neither global
`random` nor a clock.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from loguru import logger

from sim_clock import gama_timestamp, wall_clock

# Reference year for day arithmetic: a leap year, so that 29 February can be
# drawn when the window contains it.
_ANNEE_PIVOT = 2024


def _jour_de_lannee(jour: dt.date) -> tuple[int, int]:
    return jour.month, jour.day


@lru_cache(maxsize=8)
def jours_eligibles(
    debut: str,
    fin: str,
    jours_semaine: tuple[int, ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    """(month, day) days of the window, optionally filtered by weekday.

    `debut` and `fin` are inclusive ISO dates; the window may straddle New Year
    (the EMC² survey runs from 20 September to 18 February). The weekday filter
    applies to the ACTUAL dates of the window — that is where a "working day"
    makes sense —, and only the (month, day) pair is kept, since that is what the
    weather loader indexes.
    """
    premier = dt.date.fromisoformat(debut)
    dernier = dt.date.fromisoformat(fin)
    if dernier < premier:
        raise ValueError(f"empty weather window: {debut} → {fin}")

    autorises = set(jours_semaine) if jours_semaine else None
    sortie: list[tuple[int, int]] = []
    vus: set[tuple[int, int]] = set()
    jour = premier
    while jour <= dernier:
        if autorises is None or jour.isoweekday() in autorises:
            cle = _jour_de_lannee(jour)
            if cle not in vus:
                vus.add(cle)
                sortie.append(cle)
        jour += dt.timedelta(days=1)
    if not sortie:
        raise ValueError(
            f"no eligible day in {debut} → {fin} with weekdays {jours_semaine}"
        )
    return tuple(sortie)


def indice_agent(graine: int, person_id: str, cardinal: int) -> int:
    """Deterministic index in `[0, cardinal)`, drawn from (seed, agent).

    A hash rather than `random.Random(...).randrange`: the result depends neither
    on the Python version nor on the call order, so an archived trace stays
    replayable.
    """
    if cardinal <= 0:
        raise ValueError("zero cardinal")
    empreinte = hashlib.sha256(f"{graine}|{person_id}".encode()).digest()
    return int.from_bytes(empreinte[:8], "big") % cardinal


def date_meteo(
    person_id: str,
    graine: int,
    jours: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """The STARTING (month, day) assigned to this agent."""
    return tuple(jours[indice_agent(graine, person_id, len(jours))])


# ── Progression: the drawn date is a START, not an assignment (ticket 075) ──────────────
#
# The draw alone was enough as long as a run fitted in one simulated day. Over sixty days, it
# made the same agent re-read the same forecast sixty times: no persistence of rainy spells,
# no season, and an episodic memory built on a motionless weather.
# The drawn date therefore becomes the FIRST day, advanced by one calendar day per simulated day.
#
# ⚠ The arithmetic is done on a NON-leap year, and this is not a detail: the source
# (`data/weather/meteo_toulouse_12_mois.csv`) carries 365 days and **no 29 February**. Counting
# on a leap year would make one day in 366 fall into a hole — `get_weather` would return
# `None` and the forecast would vanish from the prompt without any line saying so.
_ANNEE_ARITHMETIQUE = 2023

# 29 February is reported only once: it can come out of the starting draw (the "annee"
# window is described on 2024, a leap year), and the alarm must be visible without flooding the log.
_29_FEVRIER_SIGNALE = False


def avancer_date(mois: int, jour: int, jours_ecoules: int) -> tuple[int, int]:
    """The (month, day) located `jours_ecoules` days after `(mois, jour)`.

    The year is ignored by the weather loader: 31 December is thus followed by 1 January,
    with no break or end of window. A 29 February as input — possible when the draw window
    is described on a leap year — is moved to 1 March, because the source does not carry
    it.
    """
    global _29_FEVRIER_SIGNALE
    if (mois, jour) == (2, 29):
        if not _29_FEVRIER_SIGNALE:
            _29_FEVRIER_SIGNALE = True
            logger.info(
                "[météo] 29 February drawn as starting day: moved to 1 March — the "
                "weather source carries 365 days and does not contain this date."
            )
        mois, jour = 3, 1
    # Advancing is done on the RANK in the year, not by adding days to a date: adding would
    # leave the arithmetic year as soon as the sum passes 31 December, and the arrival date
    # would fall in the following year — a leap year one time in four.
    # Defect found by test A5bis: `31 December + 60 days` returned 29 February, which the
    # source does not carry. The rank, for its part, loops over 365 by construction.
    rang = dt.date(_ANNEE_ARITHMETIQUE, mois, jour).timetuple().tm_yday
    rang = (rang - 1 + int(jours_ecoules)) % 365
    arrivee = dt.date(_ANNEE_ARITHMETIQUE, 1, 1) + dt.timedelta(days=rang)
    return arrivee.month, arrivee.day


@lru_cache(maxsize=4)
def dates_declarees(fichier: str) -> dict[str, tuple[int, int]]:
    """`person_id → (month, day)` of the described day, read once and kept.

    The file lives next to the population and carries `YYYY-MM-DD` dates. Only the month and
    the day count: `weather_loader.get_weather` indexes by (month, day), and the year of our
    readings is not that of the survey. The RIGHT CALENDAR DAY is thus matched in the year
    available, which remains a seasonal match, not the weather actually lived.
    """
    import json

    brut = json.loads(Path(fichier).read_text(encoding="utf-8"))
    table: dict[str, tuple[int, int]] = {}
    for person_id, texte in brut.items():
        try:
            date = dt.date.fromisoformat(str(texte))
        except ValueError:
            continue
        table[str(person_id)] = (date.month, date.day)
    logger.info(
        f"[météo] declared dates loaded: {len(table)} person(s) from {fichier}"
    )
    if len(table) < len(brut):
        logger.warning(
            f"[météo] {len(brut) - len(table)} unreadable date(s) in {fichier}: "
            "these persons fall back on the seeded draw"
        )
    return table


def date_declaree(person_id: object, fichier: object) -> tuple[int, int] | None:
    """The (month, day) described by this person, if the table carries one.

    Defensive by contract: this access path must NEVER bring down the "one weather per
    agent" device. A setting that is missing, of an unexpected type or pointing to an
    unreadable file returns `None`, and the seeded draw takes over — which is the behaviour
    from before ticket 058, not a silent degradation of something else.
    """
    if not isinstance(fichier, (str, os.PathLike)) or not str(fichier).strip():
        return None
    chemin = Path(fichier)
    if not chemin.is_file():
        logger.warning(
            f"[météo] declared dates table not found ({chemin}): seeded draw"
        )
        return None
    try:
        return dates_declarees(str(chemin)).get(str(person_id))
    except Exception as err:  # pragma: no cover — garde-fou
        logger.error(
            f"[ALARME] declared dates table unreadable ({chemin}: {err}) — seeded "
            "draw; the forecasts are NOT those of the survey days"
        )
        return None


def timestamp_meteo(
    timestamp_simule: int,
    person_id: str,
    graine: int,
    jours: Sequence[tuple[int, int]],
    jours_ecoules: int = 0,
    date_imposee: tuple[int, int] | None = None,
) -> int:
    """Timestamp to pass to `get_weather`: the agent's date, the departure time.

    `jours_ecoules` is the number of simulated days elapsed since the start of the run. At
    zero — a one-day run, or the first day of a long run — the returned date is **exactly**
    the drawn one: an experiment already measured and sealed replays the same weather as
    before ticket 075. Beyond that, the agent advances one calendar day per simulated day.

    The hour, minute and second of the simulated day are kept: the forecast is read
    in 3-hour slots, and an 08:00 departure must keep reading the 06:00 reading
    whatever the drawn date. The year used is an arbitrary pivot, since
    `get_weather` ignores it.

    ⚠ **Everything happens in WALL-CLOCK time** (`sim_clock`), and that is what makes
    the substitution exact. The version before 2026-09-04 re-read GAMA's timestamp
    in `Europe/Paris` then rebuilt an instant with `.timestamp()`: the reading
    time was already off by one hour (two in summer), and the question of the
    summer/winter time switch — which the 20/09 → 18/02 window crosses — only
    existed because instants were used. GAMA's clock ignores the switches: in
    wall-clock fields, keeping the time is exact by construction, over the whole
    window.
    """
    mois, jour = date_imposee or date_meteo(person_id, graine, jours)
    if jours_ecoules:
        mois, jour = avancer_date(mois, jour, jours_ecoules)
    reference = wall_clock(timestamp_simule)
    substitue = reference.replace(year=_ANNEE_PIVOT, month=mois, day=jour)
    return gama_timestamp(substitue)


class JourMeteo(str):
    """`MM-DD` representation of a drawn or declared weather day (ticket 107, A1).

    Inherits from `str` (serialises directly as `"MM-DD"` in the JSON traces).
    Equals a `(month, day)` tuple or list or an `"MM-DD"` string,
    so that `jour_meteo_tire == date_meteo(...)` holds in the sense of acceptance criterion 1.
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, list)) and len(other) == 2:
            try:
                m, j = [int(x) for x in self.split("-")]
                return (m, j) == tuple(other)
            except Exception:
                pass
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


def fenetre_meteo_effective(reglages: Any = None) -> dict:
    """Effective weather window (bounds, weekdays, number of eligible days) for the run identity (ticket 107, A3)."""
    try:
        from settings import settings

        agent_cfg = (
            getattr(reglages, "agent", None)
            if reglages is not None
            else getattr(settings, "agent", None)
        )
        fenetre = (
            getattr(agent_cfg, "weather_window", "enquete")
            if agent_cfg is not None
            else "enquete"
        )
        weekdays_only = (
            getattr(agent_cfg, "weather_weekdays_only", True)
            if agent_cfg is not None
            else True
        )

        jours_semaine = None
        if fenetre == "enquete":
            from mobility_core.population_reference import survey_window, surveyed_weekdays

            debut, fin = survey_window()
            if weekdays_only:
                jours_semaine = tuple(surveyed_weekdays())
        elif fenetre == "annee":
            debut, fin = "2024-01-01", "2024-12-31"
            if weekdays_only:
                jours_semaine = (1, 2, 3, 4, 5)
        else:
            debut, fin = fenetre
            if weekdays_only:
                jours_semaine = (1, 2, 3, 4, 5)

        jours = jours_eligibles(debut, fin, jours_semaine)
        return {
            "debut": str(debut),
            "fin": str(fin),
            "jours_semaine": list(jours_semaine) if jours_semaine else None,
            "jours_eligibles": len(jours),
            "nombre_jours_eligibles": len(jours),
        }
    except Exception as err:
        logger.warning(f"[météo] cannot compute the effective weather window: {err}")
        return {
            "debut": None,
            "fin": None,
            "jours_semaine": None,
            "jours_eligibles": 0,
            "nombre_jours_eligibles": 0,
        }


def resoudre_meteo_decision(
    person_id: str,
    timestamp: int,
    graine: int | None = None,
    jours: Sequence[tuple[int, int]] | None = None,
    dates_file: str | None = None,
    weather_per_agent: bool | None = None,
    jours_ecoules_count: int | None = None,
    date_imposee: tuple[int, int] | str | None = None,
) -> dict:
    """Resolves, purely and reproducibly, the weather information of a decision (ticket 107).

    Returns a dictionary:
        "source_date_meteo": "tiree" | "declaree" | "horloge",
        "jour_meteo_tire": JourMeteo("MM-JJ") | None,
        "jour_meteo_lu": str (ISO du CSV) | None,
        "weather_timestamp": int,
    """
    from settings import settings
    from urban_mobility_agents.utils.weather_loader import date_csv_meteo

    if weather_per_agent is None:
        weather_per_agent = bool(getattr(settings.agent, "weather_per_agent_dates", True))

    if not weather_per_agent:
        return {
            "source_date_meteo": "horloge",
            "jour_meteo_tire": None,
            "jour_meteo_lu": date_csv_meteo(timestamp),
            "weather_timestamp": timestamp,
        }

    try:
        from urban_mobility_agents.utils.ancre_run import jours_ecoules as _jours_ecoules

        if jours is None:
            from urban_mobility_agents.agents.llm_agent import _weather_eligible_days

            jours = _weather_eligible_days()
        if graine is None:
            graine = int(getattr(settings.agent, "weather_draw_seed", 42))
        if dates_file is None:
            dates_file = getattr(settings.agent, "weather_dates_file", None)
        if jours_ecoules_count is None:
            jours_ecoules_count = _jours_ecoules(timestamp)

        dec = None
        if date_imposee is not None:
            if isinstance(date_imposee, str) and "-" in date_imposee:
                p = date_imposee.split("-")
                dec = (int(p[0]), int(p[1]))
            elif isinstance(date_imposee, (tuple, list)) and len(date_imposee) == 2:
                dec = (int(date_imposee[0]), int(date_imposee[1]))
        if dec is None:
            dec = date_declaree(person_id, dates_file)
        if dec is not None:
            mois, jour = dec
            jour_tire = JourMeteo(f"{mois:02d}-{jour:02d}")
            ts = timestamp_meteo(
                timestamp,
                person_id,
                graine,
                jours,
                date_imposee=dec,
                jours_ecoules=jours_ecoules_count,
            )
            return {
                "source_date_meteo": "declaree",
                "jour_meteo_tire": jour_tire,
                "jour_meteo_lu": date_csv_meteo(ts),
                "weather_timestamp": ts,
            }
        else:
            mois, jour = date_meteo(person_id, graine, jours)
            jour_tire = JourMeteo(f"{mois:02d}-{jour:02d}")
            ts = timestamp_meteo(
                timestamp,
                person_id,
                graine,
                jours,
                jours_ecoules=jours_ecoules_count,
            )
            return {
                "source_date_meteo": "tiree",
                "jour_meteo_tire": jour_tire,
                "jour_meteo_lu": date_csv_meteo(ts),
                "weather_timestamp": ts,
            }
    except Exception as err:
        logger.error(
            f"[ALARME] weather date draw impossible ({err}) — falling back on "
            f"the simulated clock, the \"one weather per agent\" device is INACTIVE"
        )
        return {
            "source_date_meteo": "horloge",
            "jour_meteo_tire": None,
            "jour_meteo_lu": date_csv_meteo(timestamp),
            "weather_timestamp": timestamp,
        }

