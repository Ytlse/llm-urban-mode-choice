"""Utility functions for the population generation and enrichment pipeline.

Canonical location: services/llm-agents/population_utils.py
The scripts/data/population/population_utils.py shim re-exports this module.
"""
import copy
import logging
import uuid as _uuid

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

import numpy as np

_logger = logging.getLogger(__name__)

# ── Domain constants ──────────────────────────────────────────────────────────
TRIP_MODES         = ["foot", "bicycle", "car"]
MAX_DAY_SECONDS    = 86_400.0   # 24 h in seconds
MIN_ACTIVITY_DELAY = 15 * 60    # 15 min
_SCHED_MARGIN      = 10 * 60    # 10 min
_RIGID_SCH         = ['education', 'work', 'leisure', 'other', 'shop', 'home']

is_verbose = False


# ── Activity pair utilities ───────────────────────────────────────────────────

def _activity_index_pairs(activities: list, has_car: bool) -> list[tuple]:
    """Return all (prev_i, curr_i, modes) tuples, including the cyclic last→first."""
    n = len(activities)
    modes = TRIP_MODES if has_car else ["foot", "bicycle"]
    pairs = [(i - 1, i, modes) for i in range(1, n)]
    if n >= 2:
        pairs.append((n - 1, 0, modes))  # cyclic pair: return home (last → first)
    return pairs


def scheduling_mode(person: dict) -> str:
    """Return the scheduling mode for a person: 'car' if they own one, 'bicycle' otherwise."""
    has_car = person.get("identity", {}).get("traits_json", {}).get("number_of_cars", 0) > 0
    return "car" if has_car else "bicycle"


def collect_scheduling_pairs(data: list) -> set:
    """Return (lat1, lon1, lat2, lon2, mode, hour|None) for the scheduling mode of each person.

    mode = 'car' for car owners (hour = actual departure hour),
           'bicycle' for others (hour = None, time-independent).
    Coordinates are rounded to 5 decimal places to match OsmnxPersistentCache precision.
    """
    pairs: set = set()
    for entry in data:
        identity   = entry.get("identity", {})
        activities = identity.get("activities", [])
        has_car    = identity.get("traits_json", {}).get("number_of_cars", 0) > 0
        mode       = "car" if has_car else "bicycle"

        for prev_i, curr_i, _ in _activity_index_pairs(activities, has_car):
            prev_loc = activities[prev_i].get("location")
            curr_loc = activities[curr_i].get("location")
            if not prev_loc or not curr_loc:
                continue
            if prev_loc.get("lat") is None or curr_loc.get("lat") is None:
                continue
            departure_hour = int((activities[prev_i].get("end_time") or 0) / 3600) % 24
            pairs.add((
                round(prev_loc["lat"], 5), round(prev_loc["lon"], 5),
                round(curr_loc["lat"], 5), round(curr_loc["lon"], 5),
                mode,
                departure_hour if mode == "car" else None,
            ))
    return pairs


def collect_warmup_pairs(data: list) -> set:
    """Return all (lat1, lon1, lat2, lon2, mode, hour|None) for a full SQLite warm-up.

    - foot + bicycle: 1 entry each per unique O-D pair (time-independent, hour=None).
    - car: 24 hourly entries per O-D pair where at least one person owns a car.
    Coordinates rounded to 5 decimal places (matches OsmnxPersistentCache.make_key).
    """
    od_all: set = set()
    od_car: set = set()

    for entry in data:
        identity   = entry.get("identity", {})
        activities = identity.get("activities", [])
        has_car    = identity.get("traits_json", {}).get("number_of_cars", 0) > 0

        for prev_i, curr_i, _ in _activity_index_pairs(activities, True):
            prev_loc = activities[prev_i].get("location")
            curr_loc = activities[curr_i].get("location")
            if not prev_loc or not curr_loc:
                continue
            if prev_loc.get("lat") is None or curr_loc.get("lat") is None:
                continue
            pair = (
                round(prev_loc["lat"], 5), round(prev_loc["lon"], 5),
                round(curr_loc["lat"], 5), round(curr_loc["lon"], 5),
            )
            od_all.add(pair)
            if has_car:
                od_car.add(pair)

    result: set = set()
    for lat1, lon1, lat2, lon2 in od_all:
        result.add((lat1, lon1, lat2, lon2, "foot",    None))
        result.add((lat1, lon1, lat2, lon2, "bicycle", None))
    for lat1, lon1, lat2, lon2 in od_car:
        for h in range(24):
            result.add((lat1, lon1, lat2, lon2, "car", h))
    return result


def build_travel_times(data: list, route_cache: dict) -> list[dict]:
    """Build per-person travel-time dicts from a pre-computed route cache.

    Returns list[i] = {(prev_i, curr_i): duration_s} for person data[i].
    route_cache key: (lat1, lon1, lat2, lon2, mode, hour|None) — 5-decimal precision.
    Missing or None routes default to 0 (no travel time).
    """
    result = []
    for entry in data:
        identity   = entry.get("identity", {})
        activities = identity.get("activities", [])
        has_car    = identity.get("traits_json", {}).get("number_of_cars", 0) > 0
        mode       = "car" if has_car else "bicycle"
        person_tt: dict[tuple, int] = {}

        for prev_i, curr_i, _ in _activity_index_pairs(activities, has_car):
            prev_loc = activities[prev_i].get("location")
            curr_loc = activities[curr_i].get("location")
            if not prev_loc or not curr_loc:
                continue
            if prev_loc.get("lat") is None or curr_loc.get("lat") is None:
                continue
            departure_hour = int((activities[prev_i].get("end_time") or 0) / 3600) % 24
            key = (
                round(prev_loc["lat"], 5), round(prev_loc["lon"], 5),
                round(curr_loc["lat"], 5), round(curr_loc["lon"], 5),
                mode,
                departure_hour if mode == "car" else None,
            )
            route = route_cache.get(key)
            if route and route.get("duration_s"):
                person_tt[(prev_i, curr_i)] = int(route["duration_s"])
        result.append(person_tt)
    return result


# ── Activity validation ───────────────────────────────────────────────────────

def _make_home_activity(home_loc: dict, start_time: float = 0.0, end_time: float = 86400.0) -> dict:
    return {
        "id": str(_uuid.uuid4()),
        "scheduled_start_time": None,
        "start_time": start_time,
        "end_time": end_time,
        "purpose": "home",
        "location": home_loc,
    }


def check_activities(person: dict) -> list[str]:
    acts = person.get("identity", {}).get("activities", [])
    issues: list[str] = []

    if not acts:
        return ["No activities"]

    n = len(acts)

    # R2 — all times must be in [0, 86400]
    for i, act in enumerate(acts):
        for key in ("start_time", "end_time"):
            t = act.get(key)
            if t is not None and t > MAX_DAY_SECONDS:
                issues.append(
                    f"[R2] Activity[{i}] '{act.get('purpose', '?')}':"
                    f" {key}={t} exceeds 86400"
                )

    # R3 — non-zero duration: start == end is always invalid
    #       start > end is valid (activity crosses midnight)
    for i, act in enumerate(acts):
        st = act.get("start_time")
        et = act.get("end_time")
        p  = act.get("purpose", "?")
        if st is not None and et is not None and st == et:
            issues.append(f"[R3] Activity[{i}] '{p}': zero duration (start == end == {st})")

    if n == 1:
        return issues

    # R4 — strict chronological order for i = 1 ... n-1
    #       skipped when the predecessor crosses midnight (start > end)
    for i in range(1, n):
        prev    = acts[i - 1]
        prev_st = prev.get("start_time")
        prev_et = prev.get("end_time")
        curr_st = acts[i].get("start_time")
        if prev_st is None or prev_et is None or curr_st is None:
            continue
        if prev_st > prev_et:
            continue  # predecessor crosses midnight
        if curr_st <= prev_st:
            issues.append(
                f"[R4] Activity[{i}] '{acts[i].get('purpose')}':"
                f" start_time={curr_st} <= previous {prev_st}"
            )

    # R1 — no overlap: each activity must start at or after the previous ends
    #       a positive gap (travel time from ENTD) is valid and preserved
    for i in range(1, n):
        curr_st = acts[i].get("start_time")
        prev_et = acts[i - 1].get("end_time")
        if curr_st is not None and prev_et is not None and curr_st < prev_et:
            issues.append(
                f"[R1] Activity[{i}] '{acts[i].get('purpose')}':"
                f" overlap: start_time={curr_st} < prev end_time={prev_et}"
            )

    return issues


def fix_activities(person: dict) -> tuple[dict, list[str]]:
    """
    Fix order:
      0. Normalize times to [0, 86400) with modulo 24h
      1. Sort by start_time
      2. Fix zero-duration or inverted activities
      3. Circular merge: fold first+last if same purpose
      4. Fix overlaps only: if start_time[k+1] < end_time[k], set start_time[k+1] = end_time[k].
         Positive gaps (travel time from ENTD) are preserved as-is.
    """
    person   = copy.deepcopy(person)
    identity = person.get("identity", {})
    acts: list[dict] = identity.get("activities", [])
    home_loc: dict   = identity.get("home", {})
    fixes: list[str] = []

    if not acts:
        acts = [_make_home_activity(home_loc)]
        fixes.append("Created single home activity (list was empty)")
        identity["activities"] = acts
        return person, fixes

    # Pre-0 — eqasim cyclic wrap: in the 27-30h model the last activity's end_time
    # is the true start of activity[0]. Without this patch, modulo normalization
    # followed by sort-by-start_time can place activity[0] at the wrong position.
    if len(acts) >= 2:
        last_end = acts[-1].get("end_time")
        if last_end is not None and last_end > MAX_DAY_SECONDS:
            old_st = acts[0].get("start_time", 0) or 0
            acts[0]["start_time"] = last_end
            fixes.append(
                f"Activity[0] '{acts[0].get('purpose')}': start_time {old_st:.0f}"
                f" -> {last_end:.0f} (eqasim 27-30h cyclic wrap, pre-normalization)"
            )

    # 0 — normalize times to [0, 86400) using modulo 24h
    for i, act in enumerate(acts):
        for key in ("start_time", "end_time"):
            t = act.get(key)
            if t is not None and t > MAX_DAY_SECONDS:
                new_t = t % MAX_DAY_SECONDS
                fixes.append(
                    f"Activity[{i}] '{act.get('purpose')}':"
                    f" {key} {t:.0f} -> {new_t:.0f} (modulo 24h)"
                )
                act[key] = new_t

    # 1 — sort by start_time
    original_order = [a.get("start_time") for a in acts]
    acts.sort(key=lambda a: a.get("start_time") or 0.0)
    if [a.get("start_time") for a in acts] != original_order:
        fixes.append("Reordered activities by start_time")

    # 2 — fix zero-duration or inverted activities
    for i, act in enumerate(acts):
        st = act.get("start_time")
        et = act.get("end_time")
        if st is None or et is None:
            continue
        if st > et:
            act["start_time"], act["end_time"] = et, st
            fixes.append(
                f"Activity[{i}] '{act.get('purpose')}':"
                f" swapped start_time/end_time ({st:.0f} <-> {et:.0f})"
            )
        elif st == et:
            act["end_time"] = st + 600.0
            fixes.append(
                f"Activity[{i}] '{act.get('purpose')}':"
                f" zero duration, added 10 min gap -> end_time={act['end_time']:.0f}"
            )

    # 3 — circular merge: fold first+last if same purpose
    if len(acts) >= 2 and acts[0].get("purpose") == acts[-1].get("purpose"):
        last = acts.pop()
        acts[0]["start_time"] = last["start_time"]
        fixes.append(
            f"Circular merge: folded first+last '{acts[0].get('purpose')}'"
            f" (start_time {last['start_time']:.0f} -> {acts[0]['end_time']:.0f}, crosses midnight)"
        )

    # 4 — fix overlaps: if start_time[k+1] < end_time[k], collapse gap to 0.
    #     Positive gaps representing ENTD travel time are preserved untouched.
    n = len(acts)
    if n >= 2:
        for i in range(1, n):
            prev_et = acts[i - 1].get("end_time")
            curr_st = acts[i].get("start_time")
            if prev_et is not None and curr_st is not None and curr_st < prev_et:
                fixes.append(
                    f"Activity[{i}] '{acts[i].get('purpose')}':"
                    f" overlap fixed: start_time {curr_st:.0f} -> {prev_et:.0f}"
                )
                acts[i]["start_time"] = prev_et

    identity["activities"] = acts
    return person, fixes


# ── Activities outside the scope of the 453 communes (ticket 031, decision of 2026-09-03) ────
# Deliberate assumption: an activity located OUTSIDE the polygon of the 453 communes of the survey is
# REMOVED from the person's chain (never the home, which defines the scope). It is
# counted on the record (`perimetre.activites_hors_perimetre_supprimees`, at the root),
# logged by the notebook and alarmed if > 0, and reported in the seal's MANIFEST. The routing
# graph and the survey targets do not cover these places: keeping them meant routing as
# the crow flies towards a point without reference. Measured on 2026-09-03: 0 on the Haute-
# Garonne pools; the safeguard exists for the day eqasim places a school or a job outside.
PERIMETRE_ROOT_KEY = "perimetre"
OUT_OF_PERIMETER_LABEL = "hors périmètre"


def drop_out_of_perimeter_activities(person: dict, classify) -> int:
    """Remove a person's out-of-scope activities; return the number removed.

    `classify(lat, lon)` returns the ring, or `OUT_OF_PERIMETER_LABEL` outside, or an empty string
    if the point is unknown (then nothing is decided). The home is never removed. The previous
    activity absorbs the removed time slot (the person stays where they were); if the first
    activity of the day is removed, the next one starts at its start time. The count is
    set at the root of the record, even when it is 0: a missing key would mean
    "not checked", not "nothing to remove".
    """
    acts = person.get("identity", {}).get("activities", []) or []
    kept, removed = [], 0
    for act in acts:
        loc = act.get("location") or {}
        lat, lon = loc.get("lat"), loc.get("lon")
        if act.get("purpose") != "home" and lat is not None and lon is not None \
                and classify(lat, lon) == OUT_OF_PERIMETER_LABEL:
            removed += 1
            if kept:
                kept[-1]["end_time"] = act.get("end_time", kept[-1].get("end_time"))
                kept[-1]["scheduled_start_time"] = None
            continue
        if removed and not kept and act.get("start_time") is not None:
            # One or more activities removed at the head: the first one kept opens the day.
            act = dict(act); act["start_time"] = 0.0; act["scheduled_start_time"] = None
        kept.append(act)
    if removed:
        person.setdefault("identity", {})["activities"] = kept
    person[PERIMETRE_ROOT_KEY] = {**(person.get(PERIMETRE_ROOT_KEY) or {}),
                                  "activites_hors_perimetre_supprimees": removed}
    return removed


def _same_loc(a: dict, b: dict) -> bool:
    la, lb = a.get("location"), b.get("location")
    if la is None or lb is None:
        return la is lb
    return la.get("lon") == lb.get("lon") and la.get("lat") == lb.get("lat")


def merge_consecutive_activities(data: list) -> int:
    """Merge consecutive (and circular first/last) activities that share the same
    purpose AND same location. Returns total number of merges performed.

    Should be called after fix_activities (which already handles the circular
    same-purpose case without a location check).
    """
    total = 0
    for person in data:
        acts = person.get("identity", {}).get("activities", [])

        # Consecutive merge
        i = 0
        while i < len(acts) - 1:
            if acts[i].get("purpose") == acts[i + 1].get("purpose") and _same_loc(acts[i], acts[i + 1]):
                acts[i]["end_time"] = acts[i + 1]["end_time"]
                acts[i]["scheduled_start_time"] = None
                acts.pop(i + 1)
                total += 1
            else:
                i += 1

        # Circular merge: first and last
        if (
            len(acts) >= 2
            and acts[0].get("purpose") == acts[-1].get("purpose")
            and _same_loc(acts[0], acts[-1])
        ):
            last = acts.pop()
            acts[0]["start_time"] = last["start_time"]
            acts[0]["scheduled_start_time"] = None
            total += 1

    return total


# ── Public transport enrichment ───────────────────────────────────────────────

def nearest_stop_dist_m(lat: float, lon: float, stop_lats, stop_lons) -> float:
    """Vectorised Euclidean approximation for nearest stop distance in metres."""
    dlat = (stop_lats - lat) * 111_320.0
    dlon = (stop_lons - lon) * 111_320.0 * np.cos(np.radians(lat))
    return float(np.hypot(dlat, dlon).min())


def enrich_public_transport(data: list, stop_lats, stop_lons, max_pt_dist_m: float = 1_500.0) -> int:
    """Add / overwrite public_transport field on every location. Returns count of locations updated."""
    updated = 0
    for entry in data:
        identity = entry.get("identity", {})
        home = identity.get("home")
        if home and home.get("lat") is not None:
            pt = nearest_stop_dist_m(home["lat"], home["lon"], stop_lats, stop_lons) <= max_pt_dist_m
            if home.get("public_transport") != pt:
                home["public_transport"] = pt
                updated += 1

        for act in identity.get("activities", []):
            loc = act.get("location")
            if loc and loc.get("lat") is not None:
                pt = nearest_stop_dist_m(loc["lat"], loc["lon"], stop_lats, stop_lons) <= max_pt_dist_m
                if loc.get("public_transport") != pt:
                    loc["public_transport"] = pt
                    updated += 1
    return updated


# ── Scheduling ────────────────────────────────────────────────────────────────

def calculer_duree(start: int, end: int) -> int:
    """Compute the absolute duration of a time slot on a 24h cycle."""
    return (end - start + 86400) % 86400


def get_priority(act) -> int:
    """Return the priority (0 = highest priority)."""
    purpose = act.get('purpose')
    if purpose not in _RIGID_SCH:
        _logger.warning("Unknown purpose '%s' not in priority list %s", purpose, _RIGID_SCH)
        return len(_RIGID_SCH)
    return _RIGID_SCH.index(purpose)


def verifier_topologie_initiale(person_id, activities) -> bool:
    """Check that the end of an activity exactly matches the scheduled start of the next one."""
    erreurs = []
    n = len(activities)
    for i in range(n):
        next_i = (i + 1) % n
        end_actuel = activities[i].get("end_time")
        start_suivant = activities[next_i].get("scheduled_start_time")

        # Fallback if scheduled_start_time is not initialised yet
        if start_suivant is None:
            start_suivant = activities[next_i].get("start_time")

        if end_actuel != start_suivant:
            erreurs.append(
                f"Rupture Act[{i}] -> Act[{next_i}] | end_time: {end_actuel} != scheduled_start: {start_suivant}"
            )
        activities[i]["end_time"] = start_suivant  # Force the initial continuity (corrected later if needed)

    if erreurs:
        _logger.warning("Structural inconsistency detected for Person ID %s :", person_id)
        for err in erreurs:
            _logger.warning("  - %s", err)
        return False

    return True


def get_activity_constraints(travel_s_raw: int, scheduled_start_time: int, end_time: int, index: int) -> tuple[int, int, int, int, int]:
    """Compute scheduling constraints for one activity.

    travel_s_raw: pre-computed travel time in seconds (0 if unknown).
    Returns (travel_s, espace_dispo, ajustement_duree_min, conflit_local, duree).
    """
    travel_s = travel_s_raw + _SCHED_MARGIN
    duree = calculer_duree(scheduled_start_time, end_time)

    espace_dispo = max(0, duree - travel_s - MIN_ACTIVITY_DELAY)
    ajustement_duree_min = max(0, MIN_ACTIVITY_DELAY - (duree))
    conflit_local = travel_s + ajustement_duree_min

    if is_verbose:
        print(f"\n[TRACE]  [Constraints Act {index}] travel={travel_s}s, duree_brute={duree}s | dispo={espace_dispo}s, adj={ajustement_duree_min}s -> conflit={conflit_local}s")

    return travel_s, espace_dispo, ajustement_duree_min, conflit_local, duree


def check_and_print_summary(fname, person_id, population, ajustements, nb_act, raise_error=False):
    if is_verbose:
        print("\n[TRACE] FINAL STATE AND INTEGRITY VALIDATION")
    errors = []
    table_data = []
    nb_ecarts = 0

    for i in range(nb_act):
        act = population[i]
        prev = population[(i - 1) % nb_act]
        sched = act["scheduled_start_time"]
        start = act["start_time"]
        end   = act["end_time"]

        duree_nette    = calculer_duree(start, end)
        marge_restante = duree_nette - MIN_ACTIVITY_DELAY

        # 1. Minimum duration
        if duree_nette < MIN_ACTIVITY_DELAY:
            errors.append(f"[{i}] Durée invalide : {duree_nette}s < minimum {MIN_ACTIVITY_DELAY}s")

        # 2. Cyclic continuity: end_time[i-1] == scheduled_start_time[i]
        end_prev = prev["end_time"]
        if abs(end_prev - sched) > 1:  # 1s tolerance for
            errors.append(f"[{i}] Rupture de continuité : end[{(i-1)%nb_act}]={end_prev:.0f} != sched[{i}]={sched:.0f}")

        # 3. Trip margin: gap(sched, start) >= _SCHED_MARGIN
        ecart_trajet = calculer_duree(sched, start)
        if ecart_trajet < _SCHED_MARGIN:
            errors.append(f"[{i}] Marge trajet insuffisante : {ecart_trajet}s < {_SCHED_MARGIN}s")

        # 4. Time order: sched[i] >= sched[i-1] (on a linear day, excluding the Act[0] wrap)
        if population[(i-1)%nb_act]["scheduled_start_time"] > population[i]["scheduled_start_time"]:
            nb_ecarts += 1

        table_data.append([
            i,
            act.get('purpose', 'N/A'),
            f"{sched:.0f}",
            f"{start:.0f}",
            f"{end:.0f}",
            f"{duree_nette}s",
            f"{marge_restante}s",
            f"{ajustements[i]['travel_s']}s",
            f"{ajustements[i]['shift_gauche']}s",
            f"{ajustements[i]['shift_droite']}s",
            f"{ajustements[i]['shift_gauche_2']}s",
            f"{ajustements[i]['net_shift']}s",
            f"{ajustements[i]['comp']}s"
        ])

    if nb_ecarts > 1:
        errors.append(f"Ordre temporel invalide : {nb_ecarts} écarts décroissants détectés (max 1 toléré pour passage minuit)")

    headers = [
        "Index", "Purpose", "Scheduled", "Start", "End",
        "Durée", "Marge", "Travel",
        "Shift G", "Shift D", "Shift G(2)", "Net Shift", "Compression"
    ]
    if is_verbose or errors:
        if _HAS_TABULATE:
            print(_tabulate(table_data, headers=headers, tablefmt="grid"))
            _logger.debug(_tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            _logger.debug("Schedule table (install tabulate for grid view): %s", table_data)

    if errors:
        for e in errors:
            _logger.error("[population] in %s for Person ID %s : %s", fname, person_id, e)
        if raise_error:
            raise ValueError(f"{fname} : person_id {person_id} — {len(errors)} error(s) detected => {errors[0]}")


def ajuster_planning(
    fname,
    person_id,
    activities: list,
    travel_times: "dict[tuple[int,int], int] | None" = None,
    raise_error: bool = True,
) -> list:
    """Adjust scheduled_start_time for each activity to absorb real travel durations.

    travel_times: {(prev_i, curr_i): duration_s} from collect_scheduling_pairs /
                  build_travel_times.  Defaults to {} (all trips treated as instant).
    """
    n = len(activities)
    # A single-activity day — an IMMOBILE agent, home 0 → 86,400 s — has
    # nothing to plan: no trip, no schedule to readjust. Before this guard, the
    # adjustment computation assumed two activities and raised `KeyError: 1` (measured on
    # 2026-09-03 on a single-activity persona) — which prevented keeping the
    # immobile ones in the population, whereas the survey counts 10.6% of them.
    if n < 2:
        return activities

    if travel_times is None:
        travel_times = {}

    population = copy.deepcopy(activities)
    _tt = lambda idx: travel_times.get(((idx - 1) % n, idx), 0)

    # ==========================================
    # 0. SANITIZATION & INITIAL CONTINUITY
    # ==========================================
    for i in range(n):
        if population[i]["scheduled_start_time"] is None:
            population[i]["scheduled_start_time"] = population[i]["start_time"]

    for i in range(n):
        population[i]["end_time"] = population[(i + 1) % n]["scheduled_start_time"]

    if is_verbose:
        print(f"\n\n\n\n\t\t\t\t\t===========================================\n\t\t\t\t\t[INIT] CONTINUOUS PLANNING PERSON ID: {person_id}\n\t\t\t\t\t===========================================")
        for i, act in enumerate(population):
            print(f"  Act[{i}] '{act.get('purpose')}': sched={act.get('scheduled_start_time')}, start={act.get('start_time')}, end={act.get('end_time')}")

    ajustements = {
        i: {"shift_gauche": 0, "shift_droite": 0, "shift_gauche_2": 0, "comp": 0, "travel_s": 0, "duration_adjustment": 0, "espace_dispo": 0, "estimate_duration": 0, "is_computed": False} for i in range(n)
    }

    # ==========================================
    # 1. PIVOT ANCHORING
    # ==========================================
    pivot_index = min(range(n), key=lambda i: get_priority(population[i]))
    act_pivot = population[pivot_index]

    if is_verbose: print(f"\n[TRACE] PIVOT LOCK -> Act[{pivot_index}] '{act_pivot.get('purpose')}'")
    travel_s, espace_dispo, ajustement_duree_min, conflit_local, duree = get_activity_constraints(
        _tt(pivot_index), act_pivot["scheduled_start_time"], act_pivot["end_time"], pivot_index
    )

    ajustements[pivot_index].update({
        "travel_s": travel_s,
        "espace_dispo": espace_dispo,
        "duration_adjustment": ajustement_duree_min,
        "comp": 0,
        "shift_gauche": -conflit_local,
        "shift_droite": 0,
        "shift_gauche_2": 0,
        "is_computed": True
    })

    if pivot_index > 0:
        ajustements[pivot_index - 1]["shift_gauche"] = -conflit_local
    else:
        ajustements[(pivot_index + 1) % n]["shift_droite"] = conflit_local

    if is_verbose: print("\n[TRACE] APPLICATION VECTORS Step 1 (pivot)")
    for i in range(n):
        if is_verbose:
            print(f"  Act[{i}] '{population[i].get('purpose'):<7}': travel_s={ajustements[i]['travel_s']:>8}s, comp={ajustements[i]['comp']:>8}s, shift_gauche={ajustements[i]['shift_gauche']:>8}s, shift_droite={ajustements[i]['shift_droite']:>8}s")

    # ==========================================
    # 2. LEFT PROPAGATION (PAST)
    # ==========================================
    if is_verbose: print("\n[TRACE] 2. LEFT PROPAGATION") ; print("===========================================")
    for i in range(pivot_index - 1, -1, -1):
        act_courante = population[i]
        temps_requis = -ajustements[i]["shift_gauche"]

        travel_s, espace_dispo, ajustement_duree_min, conflit_local, duree = get_activity_constraints(
            _tt(i), act_courante["scheduled_start_time"], act_courante["end_time"], i
        )

        total_besoin = temps_requis + conflit_local
        compression = min(total_besoin, espace_dispo)
        conflit_local = total_besoin - compression

        ajustements[i].update({
            "travel_s": travel_s,
            "espace_dispo": espace_dispo - compression,
            "duration_adjustment": ajustement_duree_min,
            "comp": compression,
            "shift_gauche": -(total_besoin-compression),
            "is_computed": True
        })

        ajustements[i]["estimate_duration"] = espace_dispo-compression

        if is_verbose:
            print(f"  Act[{i}] receives shift_gauche {-temps_requis}s | comp={compression}s | passes on {conflit_local}s | remaining available  {espace_dispo-compression}s")

        if (i - 1) >= 0:
            ajustements[i - 1]["shift_gauche"] = -conflit_local
        else:
            ajustements[i + 1]["shift_droite"] = conflit_local
            if is_verbose: print(f"  !! Bounce on 0. Conflict {conflit_local}s transferred as shift_droite onto Act[{i+1}]")

    if is_verbose: print("\n[TRACE] APPLICATION VECTORS Step 2 (after left propagation)")
    for i in range(n):
        if is_verbose:
            print(f"  Act[{i}] '{str(population[i].get('purpose')):<7}': travel_s={str(ajustements[i]['travel_s']):>8}s, comp={str(ajustements[i]['comp']):>8}s, shift_gauche={str(ajustements[i]['shift_gauche']):>8}s, shift_droite={str(ajustements[i]['shift_droite']):>8}s, shift_gauche_2={str(ajustements[i].get('shift_gauche_2', 0)):>8}s, duree_restante={str(ajustements[i]['estimate_duration']):>8}s")

    # ==========================================
    # 3. RIGHT PROPAGATION (FUTURE)
    # ==========================================
    if is_verbose: print("\n[TRACE] 3. RIGHT PROPAGATION") ; print("=============================")
    # Resume beyond the pivot or index 1 if a bounce occurred
    start_droite = min(1 if ajustements[1]["shift_droite"] > 0 else n, pivot_index + 1)

    for i in range(start_droite, n):
        act_courante = population[i]
        temps_requis = ajustements[i]["shift_droite"]

        if not ajustements[i]["is_computed"]:
            travel_s, espace_dispo, ajustement_duree_min, conflit_local, duree = get_activity_constraints(
                _tt(i), act_courante["scheduled_start_time"], act_courante["end_time"], i
            )
            total_besoin = temps_requis + conflit_local
        else:
            travel_s = ajustements[i]["travel_s"]
            espace_dispo = ajustements[i]["espace_dispo"]
            ajustement_duree_min = ajustements[i]["duration_adjustment"]
            total_besoin = temps_requis

        new_compression = min(total_besoin, espace_dispo)
        conflit_local = total_besoin - new_compression

        ajustements[i].update({
            "travel_s": travel_s,
            "espace_dispo": espace_dispo - new_compression,
            "duration_adjustment": ajustement_duree_min,
            "comp": ajustements[i]["comp"] + new_compression,
            "shift_droite": temps_requis,
            "is_computed": True
        })

        ajustements[i]["estimate_duration"] = espace_dispo-new_compression

        if is_verbose:
            print(f"  Act[{i}] receives shift_droite {temps_requis}s | comp={ajustements[i]['comp']}s | passes on {(conflit_local)}s | remaining available  {espace_dispo-new_compression}s")

        if i < (n - 1):
            ajustements[i + 1]["shift_droite"] = conflit_local
        else:
            shift = conflit_local-temps_requis  # (=> conflit_local - compression)
            if shift > 0:
                ajustements[i]["shift_gauche_2"] = - shift
                ajustements[i]["shift_droite"] = 0
            else:
                ajustements[i]["shift_droite"] = new_compression

            ajustements[i - 1]["shift_gauche_2"] = -conflit_local
            if is_verbose: print(f"  !! Bounce on {n-1}. Conflict {conflit_local}s transferred as shift_gauche onto Act[{i-1}]")

    if is_verbose: print("\n[TRACE] APPLICATION VECTORS Step 3 (after right propagation)")
    for i in range(n):
        if is_verbose:
            print(f"  Act[{i}] '{str(population[i].get('purpose')):<7}': travel_s={str(ajustements[i]['travel_s']):>8}s, comp={str(ajustements[i]['comp']):>8}s, shift_gauche={str(ajustements[i]['shift_gauche']):>8}s, shift_droite={str(ajustements[i]['shift_droite']):>8}s, shift_gauche_2={str(ajustements[i].get('shift_gauche_2', 0)):>8}s, duree_restante={str(ajustements[i]['estimate_duration']):>8}s")

    # ==========================================
    # 4. LEFT BACK-PROPAGATION (PAST)
    # ==========================================
    if is_verbose: print("\n[TRACE] 4. LEFT BACK-PROPAGATION") ; print("=============================")
    for i in range(n - 2, 0, -1):

        temps_requis = -ajustements[i]["shift_gauche_2"]
        travel_s = ajustements[i]["travel_s"]
        espace_dispo = ajustements[i]["espace_dispo"]
        ajustement_duree_min = 0  # Already accounted for
        total_besoin = temps_requis

        new_compression = min(total_besoin, espace_dispo)
        conflit_local = total_besoin - new_compression

        ajustements[i].update({
            "espace_dispo": espace_dispo - new_compression,
            "comp": ajustements[i]["comp"] + new_compression,
            "shift_gauche_2": -(total_besoin - new_compression)
        })

        ajustements[i]["estimate_duration"] = espace_dispo-new_compression

        if is_verbose:
            print(f"  Act[{i}] receives retro-shift {-temps_requis}s | comp={new_compression}s | passes on {conflit_local}s | remaining available  {espace_dispo-new_compression}s")

        if (i - 1) > 0:
            ajustements[i - 1]["shift_gauche_2"] -= conflit_local
        else:
            if is_verbose and conflit_local > 0:
                print(f"  !! Unsettled conflict hits the pivot: {conflit_local}s")

    if is_verbose: print("\n[TRACE] APPLICATION VECTORS Step 4 (after left back-propagation)")
    for i in range(n):
        if is_verbose:
            print(f"  Act[{i}] '{str(population[i].get('purpose')):<7}': travel_s={str(ajustements[i]['travel_s']):>8}s, comp={str(ajustements[i]['comp']):>8}s, shift_gauche={str(ajustements[i]['shift_gauche']):>8}s, shift_droite={str(ajustements[i]['shift_droite']):>8}s, shift_gauche_2={str(ajustements[i].get('shift_gauche_2', 0)):>8}s, duree_restante={str(ajustements[i]['estimate_duration']):>8}s")

    # ==========================================
    # 5. MATRIX MODIFICATION
    # ==========================================
    if is_verbose: print("\n[TRACE] APPLICATION VECTORS") ; print("=============================")
    for i in range(n):
        net_shift    = ajustements[i]["shift_gauche"] + ajustements[i]["shift_droite"] + ajustements[i]["shift_gauche_2"]
        travel_s     = ajustements[i]["travel_s"]
        ajustements[i]["net_shift"] = net_shift

        population[i]["scheduled_start_time"] = (population[i]["scheduled_start_time"] + net_shift + 86400) % 86400
        population[i]["start_time"] = (population[i]["scheduled_start_time"] + travel_s) % 86400

        if is_verbose:
            print(f"  Act[{i}] '{population[i].get('purpose')}': shift_G={ajustements[i]['shift_gauche']}s, shift_D={ajustements[i]['shift_droite']}s, shift_G_2={ajustements[i]['shift_gauche_2']}s, adj={ajustements[i]['duration_adjustment']}s -> net_shift={net_shift}s")
            print(f"    -> sched={population[i]['scheduled_start_time']}, start={population[i]['start_time']}")

    # ==========================================
    # 6. CYCLIC CONTINUITY
    # ==========================================
    for i in range(n):
        population[i]["end_time"] = population[(i + 1) % n]["scheduled_start_time"]

    # ==========================================
    # 7. INTEGRITY VALIDATION AND FINAL STATE
    # ==========================================
    check_and_print_summary(fname, raise_error=raise_error, person_id=person_id, population=population, ajustements=ajustements, nb_act=n)

    return population


def tester_logique_rebond():
    """Test set for observing the bounce and squeeze mechanics"""

    test_cases = {
        "Test 1: Marge suffisante (Cas standard)": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 10000, "travel_time": 1000},
            {"purpose": "work", "start_time": 11000, "scheduled_start_time": 10000, "end_time": 40000, "travel_time": 500},
            {"purpose": "home", "start_time": 40500, "scheduled_start_time": 40000, "end_time": 86400, "travel_time": 0}
        ],

        "Test 2: Rebond sur 0 par sous-capacité Act[0]": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 500, "travel_time": 0},
            {"purpose": "other", "start_time": 500, "scheduled_start_time": 500, "end_time": 1000, "travel_time": 0},
            {"purpose": "work", "start_time": 1000, "scheduled_start_time": 1000, "end_time": 86400, "travel_time": 0}
        ],

        "Test 3: Pivot en fin de journée (Index N-1) avec compression gauche": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 40000, "travel_time": 1000},
            {"purpose": "shop", "start_time": 41000, "scheduled_start_time": 40000, "end_time": 41500, "travel_time": 1000},
            {"purpose": "work", "start_time": 42500, "scheduled_start_time": 41500, "end_time": 86400, "travel_time": 1000}
        ],

        "Test 4: Pivot en début de journée (Index 0) avec propagation droite": [
            {"purpose": "work", "start_time": 0, "scheduled_start_time": 0, "end_time": 40000, "travel_time": 1000},
            {"purpose": "other", "start_time": 41000, "scheduled_start_time": 40000, "end_time": 41500, "travel_time": 1000},
            {"purpose": "home", "start_time": 42500, "scheduled_start_time": 41500, "end_time": 86400, "travel_time": 1000}
        ],

        "Test 5: Minimum d'activités (N=2) avec conflit nocturne": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 500, "travel_time": 500},
            {"purpose": "work", "start_time": 1000, "scheduled_start_time": 500, "end_time": 86400, "travel_time": 500}
        ],

        "Test 6: Écrasement aux deux bornes (Double sous-capacité)": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 200, "travel_time": 0},
            {"purpose": "work", "start_time": 200, "scheduled_start_time": 200, "end_time": 86000, "travel_time": 0},
            {"purpose": "shop", "start_time": 86000, "scheduled_start_time": 86000, "end_time": 86400, "travel_time": 0}
        ],

        "Test 7: Effet Domino (Cascade de shifts)": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 30000, "travel_time": 0},
            {"purpose": "other", "start_time": 30000, "scheduled_start_time": 30000, "end_time": 30500, "travel_time": 0},
            {"purpose": "other", "start_time": 30500, "scheduled_start_time": 30500, "end_time": 31000, "travel_time": 0},
            {"purpose": "other", "start_time": 31000, "scheduled_start_time": 31000, "end_time": 31500, "travel_time": 0},
            {"purpose": "work", "start_time": 31500, "scheduled_start_time": 31500, "end_time": 86400, "travel_time": 0}
        ],

        "Test 8: Journée très fragmentée (Max N=12)": [
            {"purpose": "home", "start_time": 0, "scheduled_start_time": 0, "end_time": 7200, "travel_time": 600},
            {"purpose": "other", "start_time": 7800, "scheduled_start_time": 7200, "end_time": 14400, "travel_time": 600},
            {"purpose": "shop", "start_time": 15000, "scheduled_start_time": 14400, "end_time": 14900, "travel_time": 600},
            {"purpose": "home", "start_time": 15500, "scheduled_start_time": 14900, "end_time": 21600, "travel_time": 600},
            {"purpose": "work", "start_time": 22200, "scheduled_start_time": 21600, "end_time": 43200, "travel_time": 600},
            {"purpose": "shop", "start_time": 43800, "scheduled_start_time": 43200, "end_time": 43600, "travel_time": 600},
            {"purpose": "leisure", "start_time": 44200, "scheduled_start_time": 43600, "end_time": 50400, "travel_time": 600},
            {"purpose": "home", "start_time": 51000, "scheduled_start_time": 50400, "end_time": 57600, "travel_time": 600},
            {"purpose": "other", "start_time": 58200, "scheduled_start_time": 57600, "end_time": 58600, "travel_time": 600},
            {"purpose": "shop", "start_time": 59200, "scheduled_start_time": 58600, "end_time": 72000, "travel_time": 600},
            {"purpose": "leisure", "start_time": 72600, "scheduled_start_time": 72000, "end_time": 79200, "travel_time": 600},
            {"purpose": "home", "start_time": 79800, "scheduled_start_time": 79200, "end_time": 86400, "travel_time": 600}
        ]
    }

    for name, activities in test_cases.items():
        print(f"\n\n{'#'*60}\nRunning {name}\n{'#'*60}")
        try:
            n = len(activities)
            tt = {((i - 1) % n, i): activities[i].get("travel_time", 0) for i in range(n)}
            ajuster_planning("TEST_USER", "test", activities, travel_times=tt)
            print("\n>> Test passed: the planning converged.")
        except ValueError as e:
            print(f"\n>> Validation failure: {e}")


if __name__ == "__main__":
    tester_logique_rebond()
