"""Observation of a segment travelled by a PERSONAL VEHICLE — ticket 077, lot B2.

Before this model, the `__DIRECT_CAR__` and `__DIRECT_BIKE__` segments went through
`submit_ob_transit` on the GAMA side, hence through the public transport template, whose filters
query the GTFS and obviously find nothing there. The model then read, **253 times over
the thirty-day run of ticket 075**:

    [ PUBLIC TRANSPORT ] Trip by Unknown Unknown; From: ''; To: ''; Actual duration: 9 minutes.

An agent who drives every day therefore re-read themselves every evening as a user of an unnamed
public transport, between two unnamed stops. This is not a display defect: this text
is the input of the reflection, and hence the material of the concepts.
"""

from typing import Literal

from text_helper.templates.repository import tpl_describe_the_ob_vehicle
from text_helper.type import EnvOb


class EnvObVehicle(EnvOb):
    """Segment travelled with a personal vehicle (car or bike).

    A single model for both: they only differ by the rendered word, and two twin
    models would diverge. `mode` is CONSTRAINED — an unexpected value must break at parsing
    rather than produce a plausible and false sentence.
    """

    mode: Literal["car", "bike"]
    distance: float
    duration: float

    def describe(self, weather=None) -> str:
        return tpl_describe_the_ob_vehicle.render(ob=self, weather=weather)
