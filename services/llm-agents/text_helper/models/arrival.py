from typing import Optional
from pydantic import BaseModel
from text_helper.templates.repository import tpl_describe_the_ob_trip_feedback
from text_helper.type import EnvOb

# Half a day. CALENDAR postponements move a departure by a whole day (D+1
# loop, `departure_time += 86400`) or up to the following Monday (two to three days); the
# ORDINARY slippage of a departure — an activity that overruns — counts in minutes and never approaches
# this threshold. This is what makes it possible to tell them apart without a flag carried from GAMA.
SEUIL_REPORT_CALENDAIRE_S = 43_200


def retard_d_arrivee(
    arrive_at: int,
    expected_arrive_at: int,
    schedule_at: Optional[int] = None,
    started_at: Optional[int] = None,
) -> int:
    """The delay actually SUFFERED, with the calendar postponement removed.

    `expected_arrive_at` is computed by GAMA from the ORIGINAL `schedule_at`, and it is not
    recomputed when the departure is postponed — neither by the D+1 loop, nor by the
    `agent.no_weekend_departures` rule. A Friday evening trip played on Monday therefore counted
    seventy-two hours of delay, whereas it had gone perfectly well: left 19:15,
    arrived 19:27, twelve minutes against fifteen planned (run of 2026-09-23, day 5).

    What it cost: a severity of 0.70 and a "severe" memory in the baseline,
    indistinguishable from a shock declared at 0.75 — and, in the agent's prompt, a
    "Late by: 23 hours" it never lived through.

    The postponement is DEDUCED: `started_at - schedule_at` beyond half a day can only be
    a calendar postponement. The delay is then measured against the expected arrival REBASED on
    the actual departure, that is `expected_arrive_at + report`.

    ⚠ What still counts: a departure that slips by twenty minutes because the previous
    activity overran. That is a LIVED delay and it must stay so — only the postponement decided by
    the calendar is removed. `departure_delay_s` of `gama_arrivals.csv` carries the ordinary
    slippage for whoever wants to read it separately.
    """
    report = 0
    if schedule_at is not None and started_at is not None:
        glissement = int(started_at) - int(schedule_at)
        if glissement >= SEUIL_REPORT_CALENDAIRE_S:
            report = glissement
    # An early arrival counts as zero and not as a negative delay, which would offset a
    # real incident in the same entry.
    return max(0, int(arrive_at) - int(expected_arrive_at) - report)


class EnvObArrival(EnvOb):
    expected_arrive_at: int
    prepare_before_seconds: Optional[int] = 0
    arrive_at: int
    purpose: str
    duration: Optional[float]
    plan_duration: Optional[float]
    moving_id: Optional[str] = None
    # Carried by the GAMA observation since ticket 071; declared here since 2026-09-23
    # so that `late` can remove a calendar postponement. When absent, the delay is computed
    # as before — no existing caller changes behaviour.
    schedule_at: Optional[int] = None
    started_at: Optional[int] = None

    @property
    def late(self) -> int:
        return retard_d_arrivee(
            self.arrive_at, self.expected_arrive_at, self.schedule_at, self.started_at
        )

    @property
    def is_late(self) -> bool:
        return self.late > 0

    def describe(self, weather=None) -> str:
        return tpl_describe_the_ob_trip_feedback.render(ob=self, weather=weather)


if __name__ == "__main__":
    # Example usage
    event = EnvObArrival(
        type="arrival",
        timestamp=1234567890,
        moving_id="123",
        expected_arrive_at=1234567890,
        arrive_at=1234567890,
        purpose="work",
        duration=3600.0,
        plan_duration=2600.0,
    )
    print(event.describe())

