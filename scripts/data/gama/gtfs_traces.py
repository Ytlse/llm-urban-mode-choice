"""Paths of a GTFS transit system that publishes no `shapes.txt` — the SNCF TER.

WHY THIS MODULE EXISTS
----------------------
`routes.shp` (the geometry GAMA draws AND along which its `public_vehicle`
agents run) and `trip_info.json` (the trips) are produced by two
recipes. They must name the paths **exactly** the same way:
`PublicTransport.gaml` does `route first_with (each.shape_id = shape_id)` when
creating each vehicle, and a trip `shape_id` missing from the layer yields
a nil `route` — hence a vehicle without geometry, without colour and without distances.
This module is the single place where these `shape_id` are made, so that the
two recipes cannot silently diverge. That is the mismatch that lasted
five months (layers with three transit systems, trips with only one).

ONE PATH PER DISTINCT STOP SEQUENCE, NOT ONE PER (ROUTE, DIRECTION)
-------------------------------------------------------------------
A normal GTFS publishes one `shape_id` per **service pattern**: Tisséo
publishes 395 for 124 routes. The TER, for its part, publishes no geometry (its
`shapes.txt` has only a header) and leaves `trips.shape_id` empty.

Reconstruction by (route, direction) — one path, that of the most-served
trip — is enough for display but **fabricates movement** as soon as it
is used to run vehicles. `build_trips`
(`services/llm-agents/inputs/gtfs/gama.py`) forces the last segment of a trip up to the
last point of the path:

    shape_segments[-1] = len(shape_dist_traveled_list) - 1

A Toulouse → Tarbes trip placed on the Toulouse → Pau path would thus run
all the way to Pau, within the Tarbes travel time. Measured on the export in service:
out of 1 137 TER trips, 168 (14.8 %) are not even a subsequence of the path of
their (route, direction) pair, and 6 have no `direction_id` — they were
**silently missing** from the layer, their `shape_id` being `NaN`.

Hence the rule adopted: **one distinct stop sequence = one path**, as
a GTFS that publishes its geometries does. Each trip then runs exactly on
the polyline of its own stops, forcing the last segment is a
no-op, and no trip is discarded.

The `shape_id` is `<route_id>:<sens>:<empreinte>` where the fingerprint is that of the
stop sequence: it does not move when the operator publishes new trips,
whereas numbering by rank would shift all following identifiers.

WHAT THIS MODULE DOES NOT DO
----------------------------
It knows neither geometry nor projection: it returns **sequences of `stop_id`**.
The polyline (for `routes.shp`) and the cumulative distances (for
`trip_info.json`) are computed by the callers, from the feed's
coordinates. Thus both recipes start from the same stops in the same order.
"""

from __future__ import annotations

import collections
import hashlib

# `direction_id` missing: the TER publishes 6 such trips. An explicit marker is
# better than falling back to "0", which would mix them with a real direction, and
# better than the `NaN` that made them disappear.
SENS_ABSENT = "-"

LONGUEUR_EMPREINTE = 8


def empreinte_suite(suite: list[str]) -> str:
    """Stable fingerprint of a stop sequence — the identity of the service pattern."""
    brut = "".join(suite).encode("utf-8")
    return hashlib.sha1(brut).hexdigest()[:LONGUEUR_EMPREINTE]


def sens_normalise(direction_id) -> str:
    """`direction_id` as a string, `SENS_ABSENT` when there is none.

    The same field arrives here as `""` (csv reader), as `None` or as `float('nan')`
    (pandas, `dtype=str`, empty cell). Confusing them with a real direction — or
    letting them become the string `"nan"` — is exactly what made
    6 TER trips disappear from the layer.
    """
    if direction_id is None:
        return SENS_ABSENT
    if isinstance(direction_id, float) and direction_id != direction_id:  # NaN
        return SENS_ABSENT
    texte = str(direction_id).strip()
    return SENS_ABSENT if texte in ("", "nan", "NaN", "None") else texte


def shape_id_synthetique(route_id: str, direction_id, suite: list[str]) -> str:
    """The `shape_id` that the layer AND the trips will carry for this pattern."""
    return f"{route_id}:{sens_normalise(direction_id)}:{empreinte_suite(suite)}"


def traces_par_suite_d_arrets(
    trips: list[dict], suites_par_course: dict[str, list[str]], journal=print
) -> tuple[dict[str, list[str]], dict[str, str], dict]:
    """Makes one path per distinct stop sequence.

    Args:
        trips: rows of `trips.txt` (dicts with `route_id`, `trip_id`, `direction_id`)
        suites_par_course: `trip_id` → sequence of `stop_id` ordered by `stop_sequence`
        journal: where to write the report

    Returns:
        (traces, course_vers_trace, mesures) where
          * `traces`: `shape_id` → sequence of `stop_id` (the polyline to build)
          * `course_vers_trace`: `trip_id` → `shape_id`
          * `mesures`: counters, including discarded trips and why
    """
    traces: dict[str, list[str]] = {}
    course_vers_trace: dict[str, str] = {}
    ecartees: collections.Counter = collections.Counter()
    sans_sens = 0

    for ligne in trips:
        trip_id = ligne["trip_id"]
        suite = suites_par_course.get(trip_id, [])
        # A polyline needs two points; a repeated stop would make a null segment,
        # and `build_trips` would look for a vertex strictly further than exists.
        if len(suite) < 2:
            ecartees["moins_de_deux_arrets"] += 1
            continue
        if len(set(suite)) != len(suite):
            ecartees["arret_repete"] += 1
            continue
        direction = ligne.get("direction_id")
        if sens_normalise(direction) == SENS_ABSENT:
            sans_sens += 1
        shape_id = shape_id_synthetique(ligne["route_id"], direction, suite)
        connue = traces.get(shape_id)
        if connue is not None and connue != suite:
            # Fingerprint collision: impossible in practice (truncated sha1 over
            # stop sequences of the same route), but an overwritten path would run
            # a trip on another's geometry. It is reported rather than arbitrated.
            journal(
                f"[ALARME] collision d'empreinte sur {shape_id} : deux suites d'arrêts "
                f"différentes ({len(connue)} et {len(suite)} arrêts) — course {trip_id} écartée"
            )
            ecartees["collision_empreinte"] += 1
            continue
        traces[shape_id] = suite
        course_vers_trace[trip_id] = shape_id

    mesures = {
        "courses": len(trips),
        "courses_tracees": len(course_vers_trace),
        "traces": len(traces),
        "courses_sans_direction_id": sans_sens,
        "courses_ecartees": dict(ecartees),
    }
    total_ecartees = sum(ecartees.values())
    if total_ecartees:
        journal(
            f"[ALARME] {total_ecartees} course(s) sans tracé reconstructible : {dict(ecartees)} "
            f"— elles n'auront aucun véhicule dans GAMA"
        )
    journal(
        f"    tracés reconstruits depuis les arrêts : {len(traces)} motif(s) de desserte "
        f"pour {len(course_vers_trace)}/{len(trips)} course(s)"
        + (f", dont {sans_sens} sans direction_id" if sans_sens else "")
    )
    return traces, course_vers_trace, mesures


def suites_depuis_stop_times(lignes_stop_times) -> dict[str, list[str]]:
    """`trip_id` → sequence of `stop_id`, ordered by `stop_sequence`.

    `lignes_stop_times` is any iterable of dicts carrying `trip_id`,
    `stop_id` and `stop_sequence` — a feed's csv reader as well as a DataFrame
    converted to records.
    """
    par_course: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for ligne in lignes_stop_times:
        par_course[ligne["trip_id"]].append((int(ligne["stop_sequence"]), ligne["stop_id"]))
    return {tid: [s for _, s in sorted(v)] for tid, v in par_course.items()}
