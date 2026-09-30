"""Formatting of the simulated clock, and nothing else.

⚠ **Every timestamp arriving here comes from GAMA: it is a WALL-CLOCK time**, not an
instant. These functions used to be read with `datetime.fromtimestamp(ts)` — without a time zone,
hence in the **process's** own: in the `controller` container (`TZ=Europe/Paris`),
5 a.m. wall-clock was displayed **06:00** in the prompt, and 07:00 for a simulated day
in summer. The 5,322 departures of the archived run `2026-09-04_01_09` were ALL affected
(`docs/traces/2026-09-04_14-30_horloge_prompt_meteo/`).

They now go through :func:`sim_clock.wall_clock`, the only translator in the repository:
the time displayed to the agent is that of GAMA's clock, to the minute, whatever the
process's `TZ`. `wall_clock` opens no file and reads no time zone — only the
wall-clock FIELDS matter here, never an instant.

Only deliberate exception: `format_sim_timing`, whose `real_time` field is the process's
REAL time — that is what it announces and what serves to measure the duration of a run.
"""

import sys
from typing import Tuple
# from decorator import decorator
import datetime
from loguru import logger
import humanize
from settings import Settings
from sim_clock import wall_clock



def to_24h_timestamp(timestamp: int) -> int:
    """
    Convert a timestamp to a 24-hour format.
    :param timestamp: The timestamp to convert.
    :return: The converted timestamp in 24-hour format.
    """
    return timestamp % (24 * 60 * 60)  # Assuming timestamp is in seconds


def to_timestamp_based_on_day(target_24h_timestamp: int, based_on: int) -> int:
    """
    Convert a target 24-hour timestamp to a timestamp based on a given day.
    :param target_24h_timestamp: The target 24-hour timestamp to convert.
    :param based_on: The base timestamp to use for conversion.
    :return: The converted timestamp based on the given day.
    """
    return (based_on // (24 * 60 * 60)) * (24 * 60 * 60) + target_24h_timestamp


def to_24h_timestamp_full(timestamp: int) -> Tuple[int, int]:
    """ :return: The converted timestamp in 24-hour format as a tuple of (day_of_week, total_seconds_in_day). """
    d_ = wall_clock(timestamp)
    day_of_week = d_.weekday()  # Monday is 0 and Sunday is 6
    total_seconds_in_day = timestamp % (24 * 60 * 60)
    return day_of_week, total_seconds_in_day


def ensure_timestamp_in_seconds(timestamp: int) -> int:
    """
    Ensure the timestamp is in seconds.
    :param timestamp: The timestamp to check.
    :return: The timestamp in seconds.
    """
    if timestamp > 1_000_000_0000:
        return timestamp // 1000
    return timestamp


def shift_weekend_departure_to_monday(timestamp: int) -> int:
    """
    If the departure timestamp falls on a Saturday or a Sunday, postpone it to the
    following Monday at the same time (Saturday -> +2 days, Sunday -> +1 day).
    Weekday departures are unchanged.
    :param timestamp: The Unix timestamp (seconds) of the departure.
    :return: The timestamp postponed to Monday if weekend, otherwise unchanged.
    """
    weekday = wall_clock(timestamp).weekday()  # 0=Monday .. 6=Sunday
    if weekday == 5:      # Saturday
        return timestamp + 2 * 24 * 60 * 60
    if weekday == 6:      # Sunday
        return timestamp + 1 * 24 * 60 * 60
    return timestamp


def get_weekday_category(timestamp: int) -> int:
    """
    Get the weekday category based on the timestamp.
    :param timestamp: The timestamp to check.
    :return: The category of the day (0: Monday, 1: Tuesday, 2: Wednesday, 3: Thursday, 4: Friday, 5: Saturday, 6: Sunday).
    """
    timestamp = ensure_timestamp_in_seconds(timestamp)
    weekday = wall_clock(timestamp).weekday()
    return "Weekend" if weekday >= 5 else "Weekday"


def categorize_date_time_short(timestamp: int) -> int:
    """
    Categorize the time of day based on the timestamp.
    :param timestamp: The timestamp to categorize.
    :return: The category of the time of day (0: night, 1: morning, 2: afternoon, 3: evening).
    """
    timestamp = ensure_timestamp_in_seconds(timestamp)

    def _get_day_time():
        hour = wall_clock(timestamp).hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"
        
    return wall_clock(timestamp).strftime('%A') + f" {_get_day_time()}"


def humanize_date(timestamp: int) -> str:
    """
    Convert a timestamp to a human-readable date string.
    :param timestamp: The timestamp to convert.
    :return: The converted date string in the format "YYYY-MM-DD HH:MM:SS".
    """
    timestamp = ensure_timestamp_in_seconds(timestamp)
    return wall_clock(timestamp).strftime('%d %B %Y, %H:%M')


def humanize_date_short(timestamp: int) -> str:
    """
    Convert a timestamp to a human-readable date string.
    :param timestamp: The timestamp to convert.
    :return: The converted date string in the format "YYYY-MM-DD HH:MM:SS".
    """
    timestamp = ensure_timestamp_in_seconds(timestamp)
    return wall_clock(timestamp).strftime('%A, %H:%M')


def format_route_id(route_id: str) -> str:
    if ":" in route_id:
        return route_id.replace("line:", "").replace(":", " ")
    return route_id


def duration_to_bucket_text(seconds) -> str:
    if seconds < 60:
        return "very short (under 1 minute)"
    elif seconds < 5*60:
        return "short (under 5 minutes)"
    elif seconds < 10*60:
        return "moderate (under 10 minutes)"
    elif seconds < 20*60:
        return "long (under 20 minutes)"
    else:
        return "very long (20 minutes or more)"


def time_to_bucket_text(timestamp: int) -> str:
    hour = wall_clock(timestamp).hour
    if 6 <= hour <= 10:
        return "morning rush hour (6:00 - 10:00)"
    if 10 < hour <= 16:
        return "daytime (10:00 - 16:00)"
    if 16 < hour <= 20:
        return "evening rush hour (16:00 - 20:00)"
    return "night time (20:00 - 6:00)"


def humanize_time(timestamp: int) -> str:
    """
    Convert a timestamp to a human-readable hour string.
    :param timestamp: The timestamp to convert.
    :return: The converted hour string in the format "HH:MM".
    """
    timestamp = ensure_timestamp_in_seconds(timestamp)
    return wall_clock(timestamp).strftime('%H:%M')


def humanize_duration(seconds: int) -> str:
    """
    Convert a duration in seconds to a human-readable string.
    :param duration: The duration in seconds to convert.
    :return: The converted duration string in the format "X hours Y minutes".
    """
    duration = datetime.timedelta(seconds=seconds)
    # The duration shoule be like 1 hour, 15 minutes and 16 seconds, we drop the seconds
    return humanize.precisedelta(duration).split("and")[0].strip()


SIM_TIMING_TAG = "[SIM_TIMING]"


def format_sim_timing(event: str, **fields) -> str:
    """Unified log line for the time tracking of the simulation.

    All lines share the ``[SIM_TIMING]`` tag and an ``event=...`` field
    to make searching easier (e.g. ``grep '\\[SIM_TIMING\\]'`` or
    ``grep 'event=SIM_DAY'``). The ``real_time`` field is the real time (wall
    clock) at the moment of the log; the other fields are passed as ``key=value``.

    Emitted events:
      - ``SIM_START``  : reception of /init (simulation launch)
      - ``INIT_DONE``  : end of the init phase (bootstrap finished)
      - ``SIM_DAY``    : crossing of each 24h slice of simulated time
    """
    real_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    parts = [f"{SIM_TIMING_TAG} event={event}", f"real_time={real_time}"]
    parts += [f"{k}={v}" for k, v in fields.items()]
    return " ".join(parts)


def time_window_generalize(timestamp: int) -> str:
    timestamp = ensure_timestamp_in_seconds(timestamp)
    hour = wall_clock(timestamp).hour
    if hour < 6:
        return "early morning"
    elif hour < 9:
        return "morning rush hour"
    elif hour < 12:
        return "morning"
    elif hour < 16:
        return "afternoon"
    elif hour < 18:
        return "end of the workday"
    else:
        return "evening"
    

def lower_first_char(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]


def setup_logging(settings: Settings = None):
    """
    Configure loguru: stdout + file in workdir.
    Replaces create_json_logger() and the former setup_logging().
    """
    from settings import settings as _default_settings
    s = settings or _default_settings

    logger.remove()  # Remove the default handler

    log_level = s.app.log_level

    # Console (stdout)
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )

    # File in workdir with rotation
    if s.app.log_file:
        logger.add(
            s.app.log_file,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        )
        #  logs into gama_results to be used by analysis tools
        from pathlib import Path as _Path
        gama_results_log = _Path(s.app.log_file).parent / "gama_results" / "controller.log"
        gama_results_log.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(gama_results_log),
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        )
