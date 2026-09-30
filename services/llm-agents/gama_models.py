from enum import Enum
from pydantic import BaseModel
from models import Location, PersonId
from typing import Any, Generic, Optional, TypeVar


class MessageType(str, Enum):
    # Agent to GAMA
    AG_WORLD_INIT = "ag_world_init"
    AG_PEOPLE_NEXT_MOVE = "ag_people_next_move"
    AG_PEOPLE_BATCH_NEXT_MOVE = "ag_people_batch_next_move"
    AG_ACK = "ag_ack"
    AG_SYNC = "ag_sync"
    UNKNOWN = "unknown"
    # GAMA to Agent
    # GA_PEOPLE_OBSERVATION_UPDATE = "ga_people_observation_update"
    # GA_PEOPLE_ASK_MOVE = "ga_people_ask_move"


T = TypeVar("T")

class MessageResponse(BaseModel, Generic[T]):
    success: bool = True
    error: Optional[str] = None
    error_code: Optional[str] = None
    message_type: Optional[MessageType] = MessageType.UNKNOWN
    data: Optional[T] = None


class BaseRequest(BaseModel):
    timestamp: int

""" World Initialization
"""
class WorldInitRequest(BaseRequest):
    population_size: Optional[int] = None
    part_of_llm_based_agents: Optional[float] = None
    long_term_memory_enabled: Optional[bool] = None
    long_term_self_reflect_enabled: Optional[bool] = None
    # Stop horizon on the GAMA side, in simulated days (0 = unlimited). The controller does
    # not use it to drive the simulation — GAMA is the one that halts — but it
    # records it in scenario_params.yaml: without it, the time scope of an archived
    # run can no longer be reconstructed (ticket 008, A5).
    simulation_max_days: Optional[int] = None
    # The GAMA synchronisation lock must match the controller from /init onwards.
    prefixe_commun: Optional[bool] = None
    # Accident switch (ticket 070). Absent from earlier runs: `None` means
    # "GAMA did not send it", and the controller then keeps its configuration value.
    # Never confuse with `False`, which is an explicit decision of the experimenter.
    accidents_enabled: Optional[bool] = None

class WorldSyncRequest(BaseRequest):
    ready_count: Optional[int] = None
    active_count: Optional[int] = None
    inactive_count: Optional[int] = None

class GamaPersonData(BaseModel):
    person_id: PersonId
    name: str
    location: Location
    is_llm_based: bool = True


class WorldInitResponse(BaseModel):
    people: list[GamaPersonData]
    num_people: int
    timestamp: int


""" People Next Move
"""
class PeopleNextMoveRequest(BaseRequest):
    person_id: PersonId
    from_purpose: Optional[str] = None
    from_location: Optional[Location] = None

""" People Next Move Batch
"""
class PeopleBatchNextMoveRequest(BaseRequest):
    people: list[PeopleNextMoveRequest]

""" Observation Update
"""
class ObservationUpdateRequest(BaseRequest):
    person_id: PersonId
    type: str
    data: Any

class ObservationBatchUpdateRequest(BaseRequest):
    observations: list[ObservationUpdateRequest]

""" Daily cron
"""
class DailyCronRequest(BaseRequest):
    pass
