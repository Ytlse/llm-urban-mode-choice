from typing import Any
import text_helper.models as hm
from text_helper.type import EnvOb, EnvObCode

REGISTERED_MODELS: dict[EnvObCode, Any] = {
    "transfer": hm.EnvObTransfer,
    "transit": hm.EnvObTransit,
    # Ticket 077, lot B2 — a personal vehicle is not public transport.
    "vehicle": hm.EnvObVehicle,
    "arrival": hm.EnvObArrival,
    "travel_plan": hm.TravelPlanWrapper,
    "travel_plan_query": hm.TravelPlanLiteWrapper,
    "wait_in_stop": hm.EnvObWaitInStop,
    "tc_timeout": hm.EnvObTcTimeout,
}

def env_ob_to_text(code: EnvObCode, ob: dict, purpose: str = None, weather=None) -> str:
    if code not in REGISTERED_MODELS:
        raise ValueError(f"Unknown EnvOb type: {code}")

    instance = REGISTERED_MODELS[code](**ob)
    if weather is not None and code in (
        "transit",
        "transfer",
        "vehicle",
        "wait_in_stop",
        "arrival",
    ):
        return instance.describe(weather=weather).strip()
    return instance.describe().strip()

def parse_ob(code: EnvObCode, ob: dict) -> EnvOb:
    if code not in REGISTERED_MODELS:
        raise ValueError(f"Unknown EnvOb type: {code}")
    return REGISTERED_MODELS[code](**ob)
