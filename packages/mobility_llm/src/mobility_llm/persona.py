"""persona.py — the item of the mobility categories: a persona and its decision context.

It was ``AgentSpec`` in ``llm_module.core.models``. The gateway now only knows
:class:`llm_gateway.core.models.AgentItem` (an ``agent_id`` and free fields); it is this
module that says what a persona must carry for the mobility templates to render it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    """A persona in a decision batch (itinerary, perception, reflection).

    ``extra="ignore"``: an unknown field is silently lost, as before the split.
    Declaring the field here is the only way to make it travel to the template.
    """

    model_config = ConfigDict(extra="ignore")

    agent_id: str
    perception: str
    destination: str | None = None
    destination_zone: str | None = None
    departure_time: str | None = None
    departure_timestamp: float | None = None  # Unix, serves as the batch priority score
    current_time: str | None = None
    context: str | None = None
    history: list[str] = Field(default_factory=list)
    trajectories: list[dict[str, Any]] = Field(default_factory=list)
    goal: str | None = None
    constraints: str | None = None
    feeling: str | None = None
    # Mode asked about by the evening survey (ticket 095, lot B). ONE prompt per mode, the mode
    # named and the others never cited; when absent, it is the PRIORITIES prompt, which names
    # no mode. Declared here because `extra="ignore"` silently drops any unknown field:
    # without this line, the template would receive an empty mode and would ask the six
    # questions about nothing, with no error saying so.
    mode_interroge: str | None = None
    # The text of the judged event (ticket 100, lot 3) — the article read or the shock
    # suffered. SAME LESSON as `mode_interroge` above, and it was paid again on 2026-09-22:
    # without this line, `extra="ignore"` silently dropped the text and the template asked the
    # agent to judge a blank page. The model answered `negligible` — which was the right answer
    # to the question it was really being asked. Fifteen calls out of fifteen, without an error.
    evenement: str | None = None
    # Ticket 111 — the relay to the household. Same lesson, declared in advance this time: the
    # article read and the card of each other household member. Without these two lines, the
    # reader would receive a blank page and an empty household, and would answer that it has
    # nothing to say to anyone.
    article: str | None = None
    membres: list[dict[str, Any]] = Field(default_factory=list)
    # Anticipation of the day's trip chain (ticket 014).
    day_outlook: str | None = None            # weather of the remaining slots of the day
    agenda: list[str] = Field(default_factory=list)  # remaining trips (sliding agenda)


def departure_priority(items: Sequence[BaseModel]) -> float | None:
    """Priority score of a batch = smallest ``departure_timestamp``; None if none."""
    stamps = [
        ts for ts in (getattr(a, "departure_timestamp", None) for a in items) if ts is not None
    ]
    return min(stamps) if stamps else None


__all__ = ["AgentSpec", "departure_priority"]
