from typing import Optional
from pydantic import BaseModel
from models import Location

class Action(BaseModel):
    person_id: str
    action: dict

class Observation(BaseModel):
    person_id: str
    activity_id: Optional[str] = None
    timestamp: int
    location: Location
    env_ob_code: str
    data: dict