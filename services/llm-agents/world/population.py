import json
import os
from typing import Dict, List, Optional
from models import Activity, BBox, Location, Person, PersonId
from settings import settings
from inputs.population import PopulationLoader
from inputs.population.eqasim_loader import perimeter_verdict
from errors import PersonNotFoundException
from helper import to_24h_timestamp_full
from loguru import logger
import random


class PersonScheduler:
    DEFAULT_PRE_SCHEDULE_DURATION = 0

    def __init__(self, person: Person):
        self.person = person

    def start_on_activity(self, activity: Activity):
        state = self.person.state
        state.heading_to = activity.purpose
        state.last_activity_index = self.person.identity.activities.index(activity)
        state.cache_current_activity = activity

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        return next((activity for activity in self.person.identity.activities if activity.id == activity_id), None)

    def finish_activity(self):
        state = self.person.state
        state.cache_current_activity = None
        state.heading_to = None

    MAX_RESCHEDULE_PER_STEP = 30 * 60  # 30 minutes max adjustment per arrival

    def reschedule_activity(self, activity: Activity, delta: int):
        delta = min(delta, self.MAX_RESCHEDULE_PER_STEP)
        logger.debug(f"Adjusting activity <{activity.purpose}> of person {self.person.person_id} scheduled start time based on arrival duration: {delta}")
        activities = self.person.identity.activities
        n = len(activities)
        curr_idx = activities.index(activity)
        # Cyclic indexing: act[0]'s previous is the last activity, not -1
        prev_idx = (curr_idx - 1) % n
        next_idx = (curr_idx + 1) % n
        current_target = activity.scheduled_start_time if activity.scheduled_start_time is not None else activity.start_time

        prev_act = activities[prev_idx]
        # Floor = when the previous activity ends (= departure time toward current activity).
        # scheduled_start_time[k] == prev_act.end_time, so the person cannot leave earlier.
        min_time = prev_act.end_time % 86400

        next_act = activities[next_idx]
        next_target = (next_act.scheduled_start_time or next_act.start_time) % 86400
        # Next activity wraps around midnight (its 24h time is earlier than current's)
        max_time = (86400 - 30 * 60) if next_target <= current_target else (next_target - 1)

        _new_time = max(min_time, current_target - delta)
        _new_time = min(_new_time, max_time)
        activity.scheduled_start_time = _new_time

    def next_activity(self, timestamp: int) -> Optional[Activity]:
        """
        Determines the next activity of the person based on the current time.
        Uses scheduled_start_time as the target arrival time when set, otherwise start_time.
        """
        _, current_day_time = to_24h_timestamp_full(timestamp)
        day24h_seconds = 86400

        moved_activities = [act for act in self.person.identity.activities if act.start_time is not None]

        if not moved_activities:
            logger.warning(f"No valid activities for person {self.person.person_id}")
            return None

        def _target(act: Activity) -> float:
            return act.scheduled_start_time if act.scheduled_start_time is not None else act.start_time

        ordered_activities = sorted(moved_activities, key=lambda act: (_target(act) - current_day_time) % day24h_seconds)
        return ordered_activities[0] if ordered_activities else None


    
    def next_upcoming_activity(self, timestamp: int) -> Optional[Activity]:
        """Return the nearest upcoming activity (circular 24h look-ahead).
        Uses modular arithmetic so activities after midnight are found correctly
        even when their 24h timestamp is numerically smaller than the current time."""
        _, time24h = to_24h_timestamp_full(timestamp)
        activities = self.person.identity.activities
        moved = [a for a in activities if a.start_time is not None]
        if not moved:
            return None
        return min(moved, key=lambda a: (int(a.scheduled_start_time or a.start_time) - time24h) % 86400)

    def next_wakeup_ts(self, timestamp: int) -> Optional[int]:
        """SIM timestamp of the next "wake-up": next occurrence of the
        first scheduled activity of the day (smallest 24h time of the
        planning). In the evening, it is the next day's getting up; just after midnight,
        it is the same day's getting up.

        Serves as the EDF deadline of STM reflections (ticket 010, D2): the reflection
        must be written into LTM before the agent's first decision of the next
        day — it is its only natural deadline. Returns None if the agent
        has no timestamped activity (fallback left to the caller)."""
        moved = [a for a in (self.person.identity.activities or []) if a.start_time is not None]
        if not moved:
            return None
        wake24h = min(int(a.scheduled_start_time or a.start_time) % 86400 for a in moved)
        _, time24h = to_24h_timestamp_full(timestamp)
        # Triggered EXACTLY at wake-up time, the modulo difference is 0 and
        # the deadline would be "now" — hence passed at the next sync, alarm
        # included. The next wake-up is then the next day's: it is indeed the
        # first decision this reflection must precede.
        delta = (wake24h - time24h) % 86400
        return timestamp + (delta or 86400)

    def get_current_activity(self, timestamp: int) -> Optional[Activity]:
        """Return the activity in progress at the given time, handling the midnight wrap.
        If start_time > end_time, the activity spans two days (crosses midnight)."""
        _, time24h = to_24h_timestamp_full(timestamp)
        for act in self.person.identity.activities:
            if act.start_time is None or act.end_time is None or act.location is None:
                continue
            start = act.start_time % 86400
            end = act.end_time % 86400
            if start <= end:
                if start <= time24h < end:
                    return act
            else:  # midnight wrap
                if time24h >= start or time24h < end:
                    return act
        return None

    def get_home_location(self) -> Optional[Location]:
        if self.person.identity.home:
            return self.person.identity.home
        return None
 
class WorldPopulation:
    def __init__(self, population_loader: PopulationLoader):
        self.population_loader = population_loader
        self.people: Dict[PersonId, Person] = {}

    def init(self, world_bbox: BBox) -> "WorldPopulation":
        self.load_population(world_bbox)
        self.load_population_state()
        if settings.data.debug_people_ids:
            logger.info(f"Debugging with people: {settings.data.debug_people_ids}")
            self.people = {
                person_id: person
                for person_id, person in self.people.items()
                if person_id in settings.data.debug_people_ids
            }
        return self
    
    def dump_population_snapshot(self, path: str) -> None:
        """Write the full population (with current scheduled_start_time) to path atomically."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([p.model_dump(mode="json") for p in self.get_people_list()], f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        logger.info(f"[checkpoint] Population snapshot written → {path}")

    def dump_population_state(self):
        file_path = settings.data.state_file
        d = [
            {"person_id": p.person_id, "activity_id": act.id, "scheduled_start_time": act.scheduled_start_time}
            for p in self.get_people_list()
            for act in p.identity.activities
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)

    def load_population_state(self):
        file_path = settings.data.state_file
        if not os.path.isfile(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            m = {}
            for item in data:
                activity_id = item.get("activity_id")
                scheduled_start_time = item.get("scheduled_start_time")
                m[activity_id] = (scheduled_start_time, )

        for p in self.get_people_list():
            for act in p.identity.activities:
                if act.id in m:
                    (scheduled_start_time, ) = m[act.id]
                    if scheduled_start_time is not None and scheduled_start_time < 0:
                        scheduled_start_time = None  # stale sentinel from old format, force reinit
                    act.scheduled_start_time = scheduled_start_time
                    #logger.debug(f"Loaded activity {act.id} of person {p.person_id} with scheduled_start_time {scheduled_start_time}")

    @staticmethod
    def _is_within_bbox(person: Person, bbox: BBox) -> bool:
        """Admission to the survey SCOPE, with fallback on the bbox (ticket 026).

        The name stays so as not to break the callers, but the criterion has changed: it is
        the persona's `residence_zone` trait that decides, and the bbox of the transit network is
        only used as a fallback for a population generated before ticket 021.
        """
        admis, _ = perimeter_verdict(person, bbox)
        return admis

    @staticmethod
    def _avec_foyer(entry: dict) -> dict:
        """Copy `household.id` to where `Person` can read it (ticket 100, lot 1).

        Pydantic ignores unknown keys: `household` was present in the JSON since the
        seal and lost at validation, without any log saying so. The copy is made on
        a COPY of the entry — a loader does not modify its caller's dictionary.
        """
        if entry.get("household_id") is not None:
            return entry
        foyer = entry.get("household") or {}
        if not foyer.get("id"):
            return entry
        copie = dict(entry)
        copie["household_id"] = str(foyer["id"])
        return copie

    def load_population(self, world_bbox: BBox):
        file_name = f"{settings.data.population_cache_prefix}{settings.data.population_size}.json"
        if os.path.exists(file_name):
            logger.info(f"Loading population from {file_name}")
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and "name" not in data[0].get("identity", {}):
                # Eqasim format written by _prepare_population
                from inputs.population.eqasim_loader import EqasimJSONPopulationLoader
                all_people = EqasimJSONPopulationLoader().load_population_from_data(
                    data, max_size=settings.data.population_size, bbox=world_bbox,
                )
            else:
                # Legacy Pydantic format
                all_people = [Person.model_validate(self._avec_foyer(p)) for p in data]
                all_people = [p for p in all_people if self._is_within_bbox(p, world_bbox)]
                excluded = len(data) - len(all_people)
                if excluded > 0:
                    logger.warning(f"{excluded} agent(s) excluded from cache: home outside world bbox")
            self.people = {p.person_id: p for p in all_people}
            return
        
        people = self.population_loader.load_population(
            max_size=settings.data.population_size,
            bbox=world_bbox,
        )

        n_llm_based = min(len(people), settings.data.number_of_llm_based_agents)
        if n_llm_based > 0:
            logger.info(f"Random {n_llm_based} out of {len(people)}")
            llm_based_persons = random.sample(
                list(people),
                n_llm_based,
            )
            for person in llm_based_persons:
                person.is_llm_based = True

        self.people = {person.person_id: person for person in people}
        # cache to the file
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump([
                person.model_dump() for person in self.people.values()
            ], f, ensure_ascii=False, indent=4)

    def get_people_list(self) -> List[Person]:
        return list(self.people.values())
    
    def get_llm_based_people_list(self) -> List[Person]:
        return [
            person for person in self.people.values() 
            if person.is_llm_based
        ]
    
    def get_person(self, person_id: PersonId) -> Person:
        return self.people.get(person_id)
    
    def get_person_home_location(self, person_id: PersonId) -> Location:
        person = self.get_person(person_id)
        if person is None:
            raise PersonNotFoundException(f"Person {person_id} not found")
        return PersonScheduler(person).get_home_location()
    
    @classmethod
    def get_person_default_scheduler(cls, person: Person) -> PersonScheduler:
        return PersonScheduler(person)
