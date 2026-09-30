"""Prompt weather — the day's forecast at the departure slot (ticket 023, lot 4).

The weather line only carried the slot's temperature, the condition, and a precipitation
total. It said neither **when** it rains during the day, nor **what temperature range**
awaits the agent, nor **whether it will be dark** on the way back — three things a human
checks before taking out a bike. The enriched forecast adds them.

⚠ **There is no quantified "chance of rain", and there cannot be.** The source carries no
precipitation probability column; a percentage would be made up. What can be announced is
factual: the slots whose **weather code** is a precipitation one.

⚠ **The enriched form ADDS, it never removes.** 25 days out of 365 carry millimetres
without any slot being coded as precipitation; they keep the original wording rather than
announcing "no precipitation" on a day that carries some. Measured in
`docs/traces/2026-08-25_premesure_meteo_v9/`.

⚠ **The reading time is GAMA's WALL-CLOCK time** (:func:`sim_clock.wall_clock`), not an
instant re-read in a time zone. Until 2026-09-04 the forecast was read with
`datetime.fromtimestamp(ts, tz=ZoneInfo("Europe/Paris"))`: GAMA's integer worth 5 a.m.
wall-clock meant 6 a.m. — hence the **6 a.m.** reading instead of the **3 a.m.** one, and
7 a.m. (6 a.m. reading) for a simulated summer day. Measured on the 5,322 trips of the
archived run `2026-09-04_01_09`: **2,332 departures (43.8%)** changed 3-hour reading, and
the **77 departures at 11 p.m. wall-clock** changed weather **DAY** — they read Tuesday's
forecast while their itinerary was computed on Monday
(`docs/traces/2026-09-04_14-30_horloge_prompt_meteo/`).

There is no time zone here any more, and it is not an oversight: the source is indexed by
(month, day) and read by time slot, so only the wall-clock FIELDS count. A time zone would
only have shifted the reading time.
"""

import csv
import os
import re
from datetime import datetime
from typing import Optional

_base_dir = os.path.dirname(os.path.abspath(__file__))


def _racine_donnees() -> str:
    """Ancestor carrying `data/weather`, seen from the host as from the container.

    A fixed number of `..` cannot fit both sides: this module lives under
    `<repo>/services/llm-agents/urban_mobility_agents/utils/` on the host but under
    `/app/urban_mobility_agents/utils/` in the container, where `./data/weather` is mounted
    on `/app/data/weather`. `../../../` designated the root before ticket 039; it
    designates `services/` since then, and the forecasts could no longer be found — a
    SILENT defect, `get_weather()` returning `None` and the tests merely skipping themselves.
    """
    # Look for the FILE, not the folder: on the host `services/llm-agents/data/weather`
    # exists but is empty — it is the container's mount point. Stopping there would return
    # a plausible root and forecasts that cannot be found.
    ici = os.path.abspath(_base_dir)
    while True:
        if os.path.isfile(os.path.join(ici, "data", "weather", "meteo_toulouse_12_mois.csv")):
            return ici
        parent = os.path.dirname(ici)
        if parent == ici:
            # Nothing found: return the old computation so that the error names a path.
            return os.path.normpath(os.path.join(_base_dir, "../../../"))
        ici = parent


_REPO_ROOT = _racine_donnees()
_WEATHER_CSV = os.path.join(_REPO_ROOT, "data", "weather", "meteo_toulouse_12_mois.csv")
_CODES_CSV = os.path.join(_REPO_ROOT, "data", "weather", "meteo_toulouse_codes.csv")

# (month, day) → row dict from the CSV
_weather_index: dict[tuple[int, int], dict] = {}
# code int → English condition label (ticket 074, B-6)
_code_labels: dict[int, str] = {}
_weather_first_date: Optional[str] = None
_weather_last_date: Optional[str] = None
_weather_csv_sha256: Optional[str] = None
_loaded = False
_load_error: Optional[str] = None


def _load():
    global _loaded, _load_error, _weather_first_date, _weather_last_date, _weather_csv_sha256
    if _loaded or _load_error:
        return

    try:
        # `Condition_EN` is the column served since the switch to English (ticket 074, B-6);
        # `Condition` stays intact because the histories of `data/weather/*.csv` refer to it
        # through `CodeMétéo`, and overwriting it would make the archives unreadable.
        #
        # The fallback to `Condition` is LOUD, on purpose. Silent, it would give French labels
        # to the model without anything reporting it — exactly the defect this ticket fixes,
        # and the worst of both worlds: neither the announced English, nor the error saying so.
        sans_traduction: list[int] = []
        with open(_CODES_CSV, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                code = int(row["CodeMétéo"])
                anglais = (row.get("Condition_EN") or "").strip()
                if not anglais:
                    sans_traduction.append(code)
                _code_labels[code] = anglais or row["Condition"].strip()
        if sans_traduction:
            import logging
            logging.getLogger(__name__).error(
                f"[ALARME] [weather_loader] {len(sans_traduction)} weather code(s) without "
                f"`Condition_EN` in {_CODES_CSV}: {sorted(sans_traduction)}. Their FRENCH "
                f"label goes into the prompt — add the translation (ticket 074, B-6)."
            )

        import hashlib
        with open(_WEATHER_CSV, "rb") as f_bin:
            _weather_csv_sha256 = hashlib.sha256(f_bin.read()).hexdigest()

        with open(_WEATHER_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                date_str = str(row["DATE"]).strip()
                if _weather_first_date is None:
                    _weather_first_date = date_str
                _weather_last_date = date_str
                date = datetime.strptime(date_str, "%Y-%m-%d")
                _weather_index[(date.month, date.day)] = row

        _loaded = True
    except FileNotFoundError as e:
        _load_error = str(e)
        import logging
        logging.getLogger(__name__).warning(f"[weather_loader] Weather files not found, get_weather() will return None. ({e})")


def _time_bucket(hour: int) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "noon"
    return "evening"


_TEMP_COLS = {
    "night": "TEMPERATURE_NIGHT_C_3H",
    "morning": "TEMPERATURE_MORNING_C_6H",
    "noon": "TEMPERATURE_NOON_C_12H",
    "evening": "TEMPERATURE_EVENING_C_18H",
}
_CODE_COLS = {
    "night": "WEATHER_CODE_NIGHT_3H",
    "morning": "WEATHER_CODE_MORNING_6H",
    "noon": "WEATHER_CODE_NOON_12H",
    "evening": "WEATHER_CODE_EVENING_18H",
}

# ── 3-hour resolution (2026-08-26) ────────────────────────────────────────────
# The source carries EIGHT readings (0, 3, 6, 9, 12, 15, 18, 21 h); the code only read
# four, so that an 11 a.m. departure got the 6 a.m. weather and a 5 p.m. departure
# the noon one. The weather code differs between noon and 3 p.m. on 159 days out of 365.
#
# ⚠ Two distinct roles, and they must NOT be confused:
#   * `_reading_bucket` (8 slots) — the reading at the MOMENT of departure;
#   * `_BUCKET_ORDER` (4 slices) — the "Weather later" anticipation and the day's
#     frame (range, precipitation slots), deliberately left coarse so that measuring
#     the resolution does not drag along a longer agenda line.
#     The `v10c` trap (ticket 023) was exactly this bundle of two changes.
_FINE_TEMP_COLS = {
    "midnight": "TEMPERATURE_MIDNIGHT_0H",
    "night":    "TEMPERATURE_NIGHT_C_3H",
    "morning":  "TEMPERATURE_MORNING_C_6H",
    "forenoon": "TEMPERATURE_9H",
    "noon":     "TEMPERATURE_NOON_C_12H",
    "afternoon": "TEMPERATURE_15H",
    "evening":  "TEMPERATURE_EVENING_C_18H",
    "dusk":     "TEMPERATURE_21H",
}
_FINE_CODE_COLS = {
    "midnight": "WEATHER_CODE_MIDNIGHT_0H",
    "night":    "WEATHER_CODE_NIGHT_3H",
    "morning":  "WEATHER_CODE_MORNING_6H",
    "forenoon": "WEATHER_CODE_9H",
    "noon":     "WEATHER_CODE_NOON_12H",
    "afternoon": "WEATHER_CODE_15H",
    "evening":  "WEATHER_CODE_EVENING_18H",
    "dusk":     "WEATHER_CODE_21H",
}
# Order of the eight readings, indexed by `hour // 3`.
_FINE_ORDER = ("midnight", "night", "morning", "forenoon",
               "noon", "afternoon", "evening", "dusk")

# Hazard thresholds carried in the forecast (2026-08-26). Taken from the `v10c` arm of
# ticket 023, where they annotated each leg — an unsuitable place for wind, which is a
# DAILY MAXIMUM (`WINDSPEED_MAX_KMH`) and was thus repeated identically everywhere.
# 30 km/h = fresh breeze (Beaufort 5); 3 °C = black ice threshold.
VENT_FORT_KMH = 30
VERGLAS_C = 3

def _reading_bucket(hour: int) -> str:
    """The closest 3-hour reading BEFORE `hour` (cf. `_FINE_ORDER`)."""
    return _FINE_ORDER[max(0, min(7, int(hour) // 3))]


def _heure_murale(timestamp: int) -> datetime:
    """GAMA's WALL-CLOCK time for this timestamp — the repository's only translator.

    ⚠ **The import is deferred on purpose, and moving it to the top of the module breaks
    the `prompt_calibration` test suite.** That repository is standalone: it loads this file
    **by path** (`calibration/tests/test_weather.py::_load_production_formatter`)
    to check that its copy of `weather_to_natural_language` has not drifted, and it does so
    precisely so as NOT to bring the controller and its dependencies into its tests. A
    top-of-module import makes this loading impossible
    (`ModuleNotFoundError: sim_clock`), and 14 calibration tests fail.

    A local fallback — re-reading the timestamp without a time zone if `sim_clock` is
    missing — would be worse than the error: it would return a plausible time drawn from an
    absent convention, that is exactly the defect fixed on 2026-09-04.
    """
    from sim_clock import wall_clock

    return wall_clock(timestamp)


def get_weather(timestamp: int) -> Optional[dict]:
    """Forecast of the day and of the slot of GAMA's WALL-CLOCK time (`timestamp`).

    Matched on (month, day) — the year is ignored — and on the closest 3-hour reading
    before the wall-clock time. Returns `None` when the data is missing.
    """
    _load()
    if _load_error:
        return None
    dt = _heure_murale(timestamp)
    row = _weather_index.get((dt.month, dt.day))
    if row is None:
        return None

    bucket = _reading_bucket(dt.hour)
    try:
        temp = float(row[_FINE_TEMP_COLS[bucket]])
        code = int(float(row[_FINE_CODE_COLS[bucket]]))
        precip = float(row["PRECIP_TOTAL_DAY_MM"])
    except (ValueError, KeyError):
        return None

    label = _code_labels.get(code, str(code))
    return {
        "date_csv": str(row.get("DATE", "")),
        "temperature": temp,
        "weather_code": code,
        "weather_label": label,
        "precip_mm": precip,
        **day_frame(row),
    }


def date_csv_meteo(timestamp: int) -> Optional[str]:
    """ISO date (YYYY-MM-DD) of the weather CSV row matching the timestamp's wall-clock time."""
    _load()
    if _load_error:
        return None
    dt = _heure_murale(timestamp)
    row = _weather_index.get((dt.month, dt.day))
    return str(row["DATE"]).strip() if row is not None and "DATE" in row else None


def metadonnees_csv_meteo() -> dict:
    """sha256 fingerprint, start date and end date of the weather CSV file (ticket 107, A3)."""
    _load()
    return {
        "fichier": os.path.basename(_WEATHER_CSV),
        "sha256": _weather_csv_sha256,
        "debut": _weather_first_date,
        "fin": _weather_last_date,
    }


# ── The day's frame: temperature range, sun, precipitation slots ──────────────

# Snow wins over rain: "Light snow showers" contains "shower" but is not rain. The
# order of the two tests therefore carries meaning.
#
# ⚠ These expressions follow the LANGUAGE OF THE LABELS, and that is the only thing linking
# them to the CSV. Left in French after the switch (ticket 074, B-6), they would no longer
# have recognised anything: `precip_slots` would always be empty, and the forecast would say
# "No precipitation expected" every day of the year, even in a thunderstorm. No exception, no
# log — the "absence of measurement passes for a healthy case" pattern, worse, because here
# it produces a false STATEMENT.
_SNOW_RE = re.compile(r"snow|sleet|ice pellets|blizzard", re.I)
_RAIN_RE = re.compile(r"rain|drizzle|shower|thunder", re.I)

# Adverbial phrase, and not the bare label of `_BUCKET_LABEL`: one writes "Rain expected
# in the morning", not "Rain expected morning".
_BUCKET_WHEN = {"night": "at night", "morning": "in the morning",
                "noon": "in the afternoon", "evening": "in the evening"}


def _precip_family(label: str) -> Optional[str]:
    """Precipitation family of a condition label, `None` if dry.

    The returned value is SERVED to the model, capitalised, by `_precipitation_phrase`
    ("Rain expected in the morning"): it switched to English with the rest of the forecast
    (ticket 074, B-6). Leaving it in French would have produced "Pluie expected in the
    morning" — a sentence that neither the rendering test nor the detection test would have
    caught, each looking only at its own side.
    """
    if _SNOW_RE.search(label):
        return "snow"
    if _RAIN_RE.search(label):
        return "rain"
    return None


def _hhmm(value: Optional[str]) -> Optional[str]:
    """`"20:57:00"` → `"20:57"`. `None` if the source does not carry the time."""
    text = (value or "").strip()
    return text[:5] if len(text) >= 5 else None


def day_frame(row: dict) -> dict:
    """The day's frame: temperature range, sun, precipitation slots.

    ⚠ **The bounds are widened to the slots actually read.** 30 slots out of 1,460 fall
    outside `[MIN_TEMPERATURE_C, MAX_TEMPERATURE_C]` in the source, by up to 3 °C, all at
    night. Without this widening, the prompt would contradict itself: "Weather: 11°C …
    Today 13°C to 20°C". The source is not modified — only the sentence is made
    consistent with what it announces elsewhere.
    """
    try:
        slots = [int(float(row[_TEMP_COLS[b]])) for b in _BUCKET_ORDER]
        temp_min = min(int(float(row["MIN_TEMPERATURE_C"])), *slots)
        temp_max = max(int(float(row["MAX_TEMPERATURE_C"])), *slots)
    except (ValueError, KeyError, TypeError):
        temp_min = temp_max = None

    precip_slots = []
    for bucket in _BUCKET_ORDER:
        try:
            code = int(float(row[_CODE_COLS[bucket]]))
        except (ValueError, KeyError, TypeError):
            continue
        family = _precip_family(_code_labels.get(code, ""))
        if family:
            precip_slots.append((bucket, family))

    # Wind: DAILY MAXIMUM, hence its place in the day's frame and not per leg.
    try:
        wind_max = int(float(row["WINDSPEED_MAX_KMH"]))
    except (ValueError, KeyError, TypeError):
        wind_max = None

    return {
        "temp_min": temp_min,
        "temp_max": temp_max,
        "sunrise": _hhmm(row.get("SUNRISE")),
        "sunset": _hhmm(row.get("SUNSET")),
        "precip_slots": precip_slots,
        "wind_max_kmh": wind_max,
    }


def _enumerate_en(items: list[str]) -> str:
    """`["in the morning", "in the afternoon"]` → `"in the morning and in the afternoon"`."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _precipitation_phrase(w: dict) -> str:
    """Precipitation sentence — slots if known, total otherwise.

    Three cases, in this order, and the second is the one that matters:

    1. some slots are coded as precipitation → say **when**, with the total in
       brackets when it is non-zero;
    2. no precipitation slot **but** a non-zero total → keep the original wording
       **word for word**. 25 days out of 365 are in this case, up to 2.5 mm. Announcing
       "no precipitation" there would be a loss of information;
    3. nothing at all → "No precipitation expected."
    """
    precip = w.get("precip_mm") or 0.0
    # ENGLISH decimal separator since the switch: the French comma in the middle of an
    # English sentence ("0,2 mm") reads as a thousands separator.
    precip_str = f"{precip:.1f}"
    slots = w.get("precip_slots") or []

    if slots:
        by_family: dict[str, list[str]] = {}
        for bucket, family in slots:
            by_family.setdefault(family, []).append(_BUCKET_WHEN[bucket])
        parts = [f"{family.capitalize()} expected {_enumerate_en(quand)}"
                 for family, quand in by_family.items()]
        phrase = "; ".join(parts)
        return (f"{phrase} ({precip_str} mm over the day)." if precip > 0
                else f"{phrase}.")
    if precip > 0:
        return f"Precipitation expected during the day: {precip_str} mm."
    return "No precipitation expected."


# Chronological order of the intra-day weather slices and labels for the
# "Weather later" line (ticket 014 — anticipation).
_BUCKET_ORDER = ("night", "morning", "noon", "evening")
_BUCKET_LABEL = {"night": "night", "morning": "morning",
                 "noon": "afternoon", "evening": "evening"}


def day_weather_outlook(timestamp: int) -> Optional[str]:
    """Weather of the REMAINING slices of the day (after the timestamp's one).

    Ticket 014: when choosing a mode, the agent must see the weather to come
    (taking the bike in the morning when it will rain in the evening). Returns e.g.
    "afternoon 12°C, Clear/Sunny · evening 13°C, Light rain" — or None when no slice
    remains (evening departure) or the data is missing.
    Deterministic (a function of day and time): the string is part of the decision
    cache key via the anticipation signature.
    """
    _load()
    if _load_error:
        return None
    dt = _heure_murale(timestamp)
    row = _weather_index.get((dt.month, dt.day))
    if row is None:
        return None

    current = _time_bucket(dt.hour)
    remaining = _BUCKET_ORDER[_BUCKET_ORDER.index(current) + 1:]
    parts = []
    for bucket in remaining:
        try:
            temp = int(float(row[_TEMP_COLS[bucket]]))
            code = int(float(row[_CODE_COLS[bucket]]))
        except (ValueError, KeyError):
            continue
        label = _code_labels.get(code, str(code))
        parts.append(f"{_BUCKET_LABEL[bucket]} {temp}°C, {label}")
    return " · ".join(parts) if parts else None


def weather_to_natural_language(w: Optional[dict]) -> Optional[str]:
    """The day's forecast at the departure slot, for the prompt.

        Weather: 2°C, Partly cloudy. Today 2°C to 7°C, sunrise 07:55,
        sunset 17:25. Rain expected in the evening (0.2 mm over the day).

    The day's frame covers the **whole day**, slots already past included: it answers
    "what kind of day is it", a question distinct from the one handled by the
    "Weather later" line, which only carries the remaining slots.

    A dictionary without the frame fields (`temp_min`, `sunrise`…) returns the original
    sentence. This is not a convenience tolerance: the frozen sets predating ticket 023
    carry weathers without a frame, and they must keep being re-read as they are —
    otherwise their re-evaluation would no longer be about what was measured.
    """
    if w is None:
        return None
    temp = int(w["temperature"])
    label = w["weather_label"]
    tail = _precipitation_phrase(w)

    frame = []
    if w.get("temp_min") is not None and w.get("temp_max") is not None:
        frame.append(f"Today {int(w['temp_min'])}°C to {int(w['temp_max'])}°C")
    if w.get("sunrise"):
        frame.append(f"sunrise {w['sunrise']}")
    if w.get("sunset"):
        frame.append(f"sunset {w['sunset']}")
    # Hazards (2026-08-26): added ONLY when the threshold is crossed, so that ordinary
    # days keep the original sentence word for word — and so that earlier frozen sets,
    # lacking these fields, are re-read identically.
    wind = w.get("wind_max_kmh")
    if wind is not None and wind >= VENT_FORT_KMH:
        frame.append(f"gusts up to {int(wind)} km/h")
    if w.get("temp_min") is not None and int(w["temp_min"]) < VERGLAS_C:
        frame.append("risk of black ice")
    if not frame:
        return f"Weather: {temp}°C, {label}. {tail}"
    return f"Weather: {temp}°C, {label}. {', '.join(frame)}. {tail}"
