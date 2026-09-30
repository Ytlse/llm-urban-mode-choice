"""Chain consistency of personal vehicles — the rules, and nothing else.

A bike and a car are **places**: they stay where their owner parked them.
This module carries the three rules (departure lock, parking, return lock), the
passenger mode and the orphan count, as documented in
`docs/arch/vehicle-chain.md`. It is extracted from `simulation_controller.py` (ticket 035,
spec 02, D1) so that the **same** implementation serves the GAMA simulation and the
simulator-free run (`experiences/decision.py`): no rule is rewritten elsewhere, and the
controller re-exports these names for its historical callers (tests included).

No clock reading, no network access: pure functions on `Person`,
`Location`, `TravelPlan`, plus a Prometheus metric and a rising-edge alarm.
"""

import math
from typing import Optional, Tuple

from loguru import logger
from prometheus_client import Counter

from llm_gateway.telemetry.alarms import fire_alarme
from models import Location, Person, TravelPlan
from settings import settings

# Chain consistency of personal vehicles (bike, car). event ∈
#   unavailable   : mode dropped from the options — vehicle parked elsewhere than the departure point
#   no_driver     : car dropped for lack of a driver (minor, or no licence) — a cause
#                   distinct from `unavailable`, which stays reserved for the vehicle position
#   passenger     : car trip chosen for a non-driver — an adult of the household
#                   drives, the car is not parked at the destination
#   short_return  : return lock not applied, trip below the distance threshold
#   forced_return : return-home trip restricted to this mode (the agent brings its vehicle back)
#   return_failed : return lock not applicable (no itinerary in this mode) → options restored
#   orphaned      : agent back home, vehicle left elsewhere (residual case of the model)
#   reset_home    : orphan vehicle brought back home by the end-of-loop catch-up
VEHICLE_CHAIN = Counter(
    'agent_vehicle_chain_total',
    'Chain consistency events of personal vehicles, by mode and type',
    ['mode', 'event'],
)


def _road_distance_km(origin, destination) -> Optional[float]:
    """Estimated road distance (km): as the crow flies × 1.3.

    The only estimate available **before** calling OTP — `plan.distance` only exists
    once an itinerary is chosen. The 1.3 factor is the historical convention of
    `_estimate_fallback_duration`; factoring it here prevents the return lock (A3)
    and the duration estimate from diverging one day.
    """
    if origin is None or destination is None:
        return None
    lat1, lon1 = math.radians(origin.lat), math.radians(origin.lon)
    lat2, lon2 = math.radians(destination.lat), math.radians(destination.lon)
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    distance_m = 2 * 6_371_000 * math.asin(math.sqrt(a))
    return distance_m * 1.3 / 1000.0



def _vehicle_mode(plan: TravelPlan) -> str:
    """Does the plan use a personal vehicle, and which one?

    A **chain** question (ticket 008), not a main-mode one: a vehicle is a place, it
    stays where its owner parked it. A plan with a car leg commits the car, even if the
    survey would classify the trip as public transport — that is precisely the
    park-and-ride case. This reading is therefore deliberately distinct from
    `_primary_mode`, and its vocabulary is that of `_VEHICLE_MODES`.
    """
    modes = {(leg.mode or "").lower() for leg in plan.legs if not leg.is_transfer}
    if modes & {"car"}:
        return "car"
    if modes & {"bicycle", "bike"}:
        return "bike"
    if not modes - {"foot", "walk", ""}:
        return "walk"
    return "transit"


# ── Chain consistency of personal vehicles ────────────────────────────────────
# A vehicle is a PLACE: it stays where its owner parked it. Three rules, applied
# identically to the bike and the car (cf. docs/arch/vehicle-chain.md):
#   1. departure lock — the mode is only offered if the vehicle is at the departure point;
#   2. parking        — after the choice, the vehicle follows the agent if used, otherwise it stays;
#   3. return lock    — on a trip home, if a vehicle is parked at the departure
#                       point, the agent brings it back (candidates restricted to that mode).
# Keys = outputs of `_vehicle_mode`, to compare directly with a plan's mode.
_VEHICLE_MODES: Tuple[str, ...] = ("bike", "car")

# Threshold below which the return lock does not apply (A3). On the reference run,
# 59.5% of home returns under 1 km were made by car and 7.6% on foot, against ~76%
# walking expected by EMC²: the lock forced the agent to take its car again for two
# hundred metres. Below the threshold, all modes stay offered — at the cost of a few
# more orphan vehicles, caught up by `_settle_vehicles_at_home`. Hard-coded and not
# exposed in `settings.py`: it is a documented modelling convention, not a setting.
RETURN_LOCK_MIN_DISTANCE_KM = 1.0

# Legal driving age. The corrected population (ticket 008, A1) no longer carries
# minors' licences, but the controller does not rely on it: this is where the lock
# is hard.
DRIVING_AGE = 18


def _can_drive(traits: dict) -> bool:
    """Can the agent **drive** a car?

    Licence AND legal age. Both, because neither is enough on its own: a
    population generated before the A1 safeguards hands out licences to
    nine-year-olds, and a missing `has_driving_license` is not an authorisation.
    """
    return bool(traits.get("has_driving_license", False)) and (traits.get("age", 0) or 0) >= DRIVING_AGE


def _is_car_passenger(person: Person) -> bool:
    """Can the agent **ride** in the household car without driving it?

    Three conditions: not being able to drive, the household having a car, and
    someone else being there to drive it (`household_size > 1`). An adult without a
    licence living alone is therefore not a passenger — nobody drives them.

    This is the "child driven to school" mode (v1, decision D5): without modelling
    the household, the parent's escort trip is not generated; the car is simply made
    accessible to the child. EMC² counts the passenger under "car", so the modal
    share stays comparable; what changes is that the car is no longer parked at the
    school and the child does not have to bring it back.
    """
    traits = person.identity.traits_json
    return (
        not _can_drive(traits)
        and _owns_car(traits)
        and (traits.get("household_size", 0) or 0) > 1
    )


# "Missing `personal_bike` field" alarm: it must fire only once per process,
# otherwise it is emitted at every decision of every agent and floods
# `make error`. The Prometheus counter, for its part, counts every case.
_bike_trait_alarm_on = False


def _owns_bike(traits: dict) -> bool:
    """Does the agent own a bike? Missing field ⇒ **no**, and the alarm sounds.

    The default used to be the reverse — a missing field meant "vélo normal" — for the
    sake of backward compatibility with populations generated before the trait existed.
    That is the scenario ticket 015 closes: a population without `personal_bike` thus
    put **100% of the agents on bikes, silently**, which is the worse of the two possible
    errors. The bike is the mode whose modal share is the most scrutinised in the
    project; a 100% ownership floor shifts it by several points without leaving a single
    log line.

    The fallback is therefore "no bike", which deprives the agent of a mode rather than
    offering one it does not have — and it is **loud**: the seven populations concerned
    were moved out of the loader's reach (`data/population/old/`), so this path should
    never be taken again. If it is, it is a regression of the generation chain, not a
    normal case to absorb.
    """
    global _bike_trait_alarm_on
    label = traits.get("personal_bike")
    if label is None:
        if not _bike_trait_alarm_on:
            _bike_trait_alarm_on = True
            logger.error(
                "[ALARME] Trait `personal_bike` missing from traits_json — the agents "
                "concerned are treated WITHOUT a bike. A population without this trait is "
                "not usable: regenerate it, or enrich it with "
                "`python -m scripts.data.population.enrich_personal_bike <fichier>`. "
                "(Alarm emitted only once; the counter alarme_total{source="
                '"personal_bike_absent"} counts all the agents concerned.)'
            )
        fire_alarme("personal_bike_absent")
        return False
    # BOTH vocabularies: "No bike" since the v6 cohort (ticket 074), "Pas de vélo"
    # for earlier cohorts and their archived runs. Knowing only one would make the
    # other return `True` — a persona without a bike would get a bike option across
    # a whole cohort, with no exception or log.
    from mobility_core.bike_ownership import NO_BIKE
    return str(label).strip().lower() not in {NO_BIKE.lower(), "pas de vélo"}


def _owns_car(traits: dict) -> bool:
    """Does the agent have a car in their household?"""
    return (traits.get("number_of_cars", 0) or 0) > 0


def _owns_vehicle(traits: dict, mode: str) -> bool:
    return _owns_bike(traits) if mode == "bike" else _owns_car(traits)


def _same_place(a: Optional[Location], b: Optional[Location]) -> bool:
    """Do two points designate the same parking spot?

    Tolerance in degrees rather than metres: activity places all come from the same
    dataset (eqasim) and compare identically; the margin absorbs serialisation
    rounding, not an actual walking distance.
    """
    if a is None or b is None:
        return False
    return abs(a.lat - b.lat) < 1e-6 and abs(a.lon - b.lon) < 1e-6


def _vehicle_position(person: Person, mode: str) -> Optional[Location]:
    """Where is the `mode` vehicle parked? Missing key ⇒ at home (initial state)."""
    parked = person.state.planning_vehicle_at.get(mode)
    return parked if parked is not None else person.identity.home


def _vehicle_available(person: Person, mode: str, from_location: Optional[Location]) -> bool:
    """Can the vehicle mode be offered for a trip leaving from `from_location`?

    Two conditions, not just ownership: the agent must also have its vehicle **where it
    is**. Without the second, an agent who went to work by bus found its bike again to
    leave — on a reference run, 352 of the 1086 bike trips (5.9 points of modal share)
    relied on this ghost bike. The car, for its part, had no position constraint at
    all.

    The car adds a **driver** condition (A2): an agent who cannot drive is only offered
    the car if it can ride as a passenger — and in that case the vehicle's position does
    not matter, since it is not the one who parked it. Otherwise the mode is refused
    outright: this is the hard lock guaranteeing that no minor or unlicensed person
    drives.
    """
    return _vehicle_unavailable_reason(person, mode, from_location) is None


# Drop reasons of the departure lock (ticket 035, spec 02 D2) — the vocabulary of the
# decision trace. `retour_force` and `plafond` are produced further down the chain
# (experiences/decision.py); they appear here so that the list is complete in one place.
MOTIF_NON_POSSEDE = "non_possede"
MOTIF_PAS_DE_CONDUCTEUR = "pas_de_conducteur"
MOTIF_VEHICULE_AILLEURS = "vehicule_ailleurs"
MOTIF_RETOUR_FORCE = "retour_force"
MOTIF_PLAFOND = "plafond"
MOTIFS_ECART = (MOTIF_NON_POSSEDE, MOTIF_PAS_DE_CONDUCTEUR, MOTIF_VEHICULE_AILLEURS,
                MOTIF_RETOUR_FORCE, MOTIF_PLAFOND)


def _vehicle_unavailable_reason(person: Person, mode: str, from_location: Optional[Location]) -> Optional[str]:
    """Why the vehicle mode can NOT be offered from `from_location` — `None` if it can.

    This is the only implementation of the departure lock: `_vehicle_available` is only its
    boolean reading, and the decision trace (spec 02, D6) takes its reason as is.
    Order of the tests, that of the architecture document: ownership, driver/passenger,
    position.
    """
    if not _owns_vehicle(person.identity.traits_json, mode):
        return MOTIF_NON_POSSEDE
    if mode == "car" and not _can_drive(person.identity.traits_json):
        return None if _is_car_passenger(person) else MOTIF_PAS_DE_CONDUCTEUR
    if not settings.agent.vehicle_chain_enabled:
        return None
    parked_at = _vehicle_position(person, mode)
    if parked_at is None:
        # Population without a known home (the eqasim loader drops them as soon as a bbox
        # is set): we do not know where the vehicle is. Degrading to the old behaviour is
        # better than depriving the agent of any vehicle mode for the whole simulation.
        VEHICLE_CHAIN.labels(mode=mode, event="no_home").inc()
        return None
    return None if _same_place(parked_at, from_location) else MOTIF_VEHICULE_AILLEURS


def _vehicles_parked_at(person: Person, location: Optional[Location]) -> set[str]:
    """Owned vehicles parked at `location`, **outside home**.

    Used by the return lock: these are the vehicles the agent must bring back when it
    goes home. A vehicle already at home has nothing to bring back.
    """
    if _same_place(location, person.identity.home):
        return set()
    return {
        mode for mode in _VEHICLE_MODES
        if _owns_vehicle(person.identity.traits_json, mode)
        and _same_place(_vehicle_position(person, mode), location)
    }


def _park_vehicles(
    person: Person,
    plan: Optional[TravelPlan],
    from_location: Optional[Location],
    destination: Optional[Location],
) -> None:
    """Updates the vehicles' position after a plan is chosen.

    The vehicle used follows the agent to the destination; the others stay parked
    where they were. No implicit return home: that was the last teleportation relic
    of the boolean version (a bike left at the office was deemed found at home in
    the evening).
    """
    if plan is None or destination is None:
        return
    mode = _vehicle_mode(plan)
    if mode not in _VEHICLE_MODES:
        return
    if mode == "car" and _is_car_passenger(person):
        # It is not its car: an adult of the household drove it and leaves with it.
        # Parking nothing at the destination is what prevents, further on, the return
        # lock from forcing a twelve-year-old to bring the car back from school.
        return
    # Defensive: the departure lock has already dropped the plans whose vehicle is
    # elsewhere — a vehicle is only moved from its actual position.
    if not _same_place(_vehicle_position(person, mode), from_location):
        return
    if _same_place(destination, person.identity.home):
        # Return home: the key is removed rather than storing the home, to preserve
        # the invariant "missing key ⇒ vehicle at home".
        person.state.planning_vehicle_at.pop(mode, None)
    else:
        person.state.planning_vehicle_at[mode] = destination


def _orphaned_vehicles(person: Person) -> set[str]:
    """Vehicles parked elsewhere than at home although the agent went back home.

    Accepted residual case of the simple model: home → work by car, work → sport on
    foot, sport → home by bus. The return lock only applies to vehicles parked at the
    DEPARTURE point of the return trip; this one stayed at work. Measured
    (agent_vehicle_chain_total{event="orphaned"}) and, by default, caught up at home so
    as not to deprive the agent of its car on the following days.
    """
    return {
        mode for mode in _VEHICLE_MODES
        if _owns_vehicle(person.identity.traits_json, mode)
        and not _same_place(_vehicle_position(person, mode), person.identity.home)
    }



def _chain_stake_modes(person: Person) -> list[str]:
    """Vehicle modes whose position commits the chain of THIS persona.

    The car only counts for a driver (a passenger has no positional car
    — cf. `_vehicle_available`, ticket 008 A2); the bike, as soon as it is
    owned, passenger included (their bike follows the normal rules).
    """
    traits = person.identity.traits_json
    modes = []
    if _owns_car(traits) and _can_drive(traits):
        modes.append("car")
    if _owns_bike(traits):
        modes.append("bike")
    return modes
