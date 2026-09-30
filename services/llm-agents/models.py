from collections import deque
from typing import List, Optional, TypeAlias
from enum import Enum
from pydantic import BaseModel, Field

""" Base models
"""
class LocationType(str, Enum):
    HOME = "home"
    WORK = "work"
    EDUCATION = "education"
    OTHER = "other"


class Location(BaseModel):
    lon: float
    lat: float
    public_transport: Optional[bool] = None
    zone: Optional[str] = None


class BBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


""" Schedule
"""
ActivityPurpose: TypeAlias = str

class Activity(BaseModel):
    """Activity represents a scheduled task or stay in a person’s daily plan.

    Attributes:
        id: unique activity identifier.
        scheduled_start_time: optional planned start time (across the day).
        start_time: actual or current start time.
        end_time: end time for the activity.
        purpose: type of activity (work, education, leisure, etc.).
        location: optional coordinates of where the activity occurs.
    """

    id: str
    # scheduled start time over the day
    scheduled_start_time: Optional[float] = None
    start_time: float
    end_time: float
    purpose: ActivityPurpose
    location: Optional[Location] = None


""" Travel plan
"""
RouteShape: TypeAlias = str

class TransitLocation(Location):
    """TransitLocation extends Location with a transit stop identifier.

    Attributes:
        stop: stop ID or name.
        lat: latitude (redeclared for transit-specific semantics).
        lon: longitude (redeclared for transit-specific semantics).
    """

    stop: str
    stop_id: Optional[str] = None
    lat: float
    lon: float

class Transit(BaseModel):
    """Transit describes one portion of a trip plan, including transfers.

    Attributes:
        start_time: departure timestamp.
        end_time: arrival timestamp.
        start_location: transit stop/location at start.
        end_location: transit stop/location at end.
        is_transfer: whether this segment is a transfer/wait segment.
        transit_route: optional route ID or name.
        shape_id: optional geometry IDs describing the path.
        transit_agency: optional operator/agency name.
        duration: optional duration (seconds) of the segment.
        distance: optional distance (meters/kilometers) of the segment.
        mode: optional travel mode (walk, bus, subway, etc.).
        step_label: optional ready-to-render label of the step (see below).
    """

    start_time: int
    end_time: int
    start_location: TransitLocation
    end_location: TransitLocation
    # route_shape: Optional[RouteShape] = None
    is_transfer: bool = False
    transit_route: Optional[str] = None
    shape_id: Optional[List[str]] = None
    transit_agency: Optional[str] = None
    duration: Optional[int] = None
    distance: Optional[float] = None
    mode: Optional[str] = None

    # Label of the sub-step, set at scenario CONSTRUCTION and not guessed
    # at rendering (ticket 013, decision T3). Carried by the access /
    # driving / egress legs of direct plans; `None` for legs coming from
    # OTP, which the template already describes from the GTFS.
    step_label: Optional[str] = None

    def get_duration(self) -> int:
        return self.duration or int((self.end_time - self.start_time) // 1000)

    def get_distance(self) -> float:
        return self.distance or 100.0

    def get_code(self) -> str:
        return "^".join([self.transit_route, self.start_location.stop, self.end_location.stop])

    @property
    def is_terminal(self) -> bool:
        """Access or egress leg of a vehicle trip (ticket 013).

        Recognised by the route marker, not by the mode: a terminal leg has no
        mode of its own, precisely so as not to pollute the mode label of
        the option (cf. :meth:`TravelPlan.mode_label`).
        """
        return bool(self.transit_route) and self.transit_route.startswith("__TERMINAL_")

class TravelPlan(BaseModel):
    """TravelPlan represents a complete transportation itinerary for a person.

    Attributes:
        id: unique identifier for the travel plan (UUID, hash, or business code).
        start_location: departure point of the journey (latitude/longitude).
        end_location: destination point of the journey (latitude/longitude).
        start_time: timestamp when the journey starts (usually milliseconds since epoch).
        end_time: timestamp when the journey ends (arrival time).
        start_in: optional countdown before start (seconds from now).
        purpose: optional reason for trip such as work, education or shopping.
        duration: optional total estimated duration of trip (often end_time - start_time).
        distance: optional total estimated distance of trip (meters/kilometers depending on context).
        legs: list of Transit segments composing the trip, including transfers.

    A Transit segment (in legs) contains start/end point/time, route info,
    and optional mode, agency, duration, distance, etc.
    """

    # unique identifier of the travel plan (UUID, hash, logical code)
    id: str

    # starting point of the trip (latitude/longitude)
    start_location: Location

    # destination of the trip (latitude/longitude)
    end_location: Location

    # start timestamp of the trip (usually in ms, consistent with Transit)
    start_time: int

    # arrival/end timestamp of the trip
    end_time: int

    # delay before the plan starts, in seconds from now
    start_in: Optional[int] = 0  # seconds from now

    # objective / purpose of the trip (work, education, shopping, etc.)
    purpose: Optional[str] = None

    # desired or computed total duration of the trip (possibly end_time-start_time)
    duration: Optional[int] = None

    # estimated total distance of the trip (metric, km, unit to be specified depending on use)
    distance: Optional[float] = None

    # sequence of Transit segments (mode, duration, itinerary, transfers)
    legs: List[Transit]

    def get_code(self) -> str:
        """
        Generate a code for the travel plan based on its attributes.
        This can be used to identify the plan in logs or messages.

        ⚠ Terminal legs (access / egress, ticket 013) carry
        ``is_transfer=True`` and are therefore excluded from here. This is an INVARIANT, not
        a side effect: this code is the key of the LLM decision cache and of the
        itinerary deduplication — splitting the display of an option must not
        make it pass for a different option.
        """
        return "+".join([
            leg.get_code() for leg in self.legs if not leg.is_transfer
        ])

    def mode_label(self) -> str:
        """Mode label of the option (``"car"``, ``"foot,bus,foot"``…).

        Terminal legs are excluded: without that, a car option
        would announce itself as ``"None,car,None"`` and the whole measurement chain would follow —
        ``parse_option_modes`` reads this label in the prompt text, and
        it feeds ``categorize_mode``, hence the calibration loss and the
        modal shares of ``moves.csv``.
        """
        legs = [leg for leg in self.legs if not leg.is_terminal]
        return ",".join(str(leg.mode) for leg in legs) if legs else ""


""" Agent & Simulation
"""
class PersonMove(BaseModel):
    """PersonMove represents an in-progress or planned move for an agent.

    Attributes:
        id: unique identifier used for tracking and updating this move.
        person_id: associated person’s ID.
        current_time: current simulation timestamp.
        expected_arrive_at: expected arrival time at target.
        prepare_before_seconds: optional prep time before departure.
        purpose: optional reason for move.
        target_location: optional location to reach.
        for_activity: optional Activity associated with this move.
        plan: optional TravelPlan containing route legs and timing.
    """

    # the id for quickly identifying and updating the move
    id: str
    person_id: str
    current_time: int
    expected_arrive_at: int
    prepare_before_seconds: Optional[int] = 0
    purpose: Optional[str] = None
    target_location: Optional[Location] = None
    for_activity: Optional[Activity] = None
    plan: Optional[TravelPlan] = None


""" Personal Identity
"""
PersonId: TypeAlias = str

class PersonalIdentity(BaseModel):
    name: str
    traits_json: dict
    home: Optional[Location] = None
    activities: Optional[List[Activity]] = None


class PersonState(BaseModel):
    last_location: Optional[Location] = None
    last_activity_index: Optional[int] = 0
    cache_current_activity: Optional[Activity] = None  # current activity
    heading_to: Optional[str] = None  # purpose of the next activity
    scheduling_in_progress: bool = False  # itinerary computation in flight
    scheduling_started_at: Optional[int] = None  # sim 24h-timestamp when scheduling was flagged
    # Async pre-computation: non-None = PLANNED state (trip computed, ready to send to GAMA)
    # None = IDLE state (not yet computed, or previous cycle finished via arrival feedback)
    next_planned_move: Optional["PersonMove"] = None
    # Queue of pre-computed moves for future activities (N+2, N+3, ...).
    # Initialised at bootstrap (full 24h), then maintained as a sliding horizon:
    # each popleft() triggers the computation of the next activity to top the queue up.
    precomputed_moves: deque["PersonMove"] = Field(default_factory=deque)
    # Horizon of the sliding refill: last activity/timestamp already present in precomputed_moves.
    # Lets _precompute_one know where to resume from without walking through the queue.
    precompute_in_progress: bool = False
    precomputed_horizon_act: Optional["Activity"] = None
    precomputed_horizon_ts: Optional[int] = None
    # Cockpit: sim timestamp of the last successful planning (non-empty plan/queue).
    # Used to detect stuck agents (no plan for > threshold of simulated hours).
    last_successful_plan_sim_ts: Optional[int] = None
    # Arrival watchdog: expected_arrive_at of the move pushed to GAMA (None outside a trip).
    # If sim time exceeds this deadline by more than world.arrival_watchdog_hours,
    # the arrival observation is considered lost (move never received by GAMA) and the
    # fallback scan forces the agent's cycle to resume.
    heading_expected_arrive_at: Optional[int] = None
    # Chain consistency of personal vehicles: where are the agent's bike and car
    # parked? Keys = modes of `_primary_mode` ("bike", "car"), value = parking
    # place. **Missing key ⇒ vehicle at home**: the day starts at
    # home, where the agent parks the vehicles they own, so the empty dict is the initial
    # state. This field says nothing about ownership, tested upstream: an agent without a
    # car does not have a car parked at home, they have no car at all.
    # A vehicle mode is only offered if its vehicle is where the agent is, and
    # it only moves if the agent uses it (cf. _vehicle_available / _park_vehicles).
    # This is a **planning** state: the plan runs ahead of execution, this field follows
    # the planned chain, not the agent's real position in GAMA.
    planning_vehicle_at: dict[str, Location] = Field(default_factory=dict)


class Person(BaseModel):
    person_id: PersonId
    identity: PersonalIdentity
    state: PersonState = PersonState()
    # hybrid technique
    is_llm_based: bool = True
    # Household of membership (ticket 100, lot 1). The population JSON has carried it since the
    # seal, under `household.id` — but at the ROOT of the entry, and since `Person` ignores unknown
    # keys, it never reached the runtime. The only reader in the repository
    # (`inputs/population/perimeter.py`) works on the raw dict, before validation.
    #
    # It is the ONLY social group of the simulation that carries a stable identifier, and two
    # mechanisms depend on it: the `foyers` exposure rule (lot 2) and circulation within
    # the household (lot 4). Without it, twenty declared households would produce a whole run without
    # a single reader, and without the slightest symptom.
    #
    # `None` = population generated without `household`, or loaded through a path that does not set
    # it. An [ALARME] is raised at startup if a mechanism that needs it is active.
    household_id: Optional[str] = None
