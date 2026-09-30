from collections import Counter
from text_helper.templates.repository import \
    get_transit_route_type, \
    tpl_describe_the_travel_plan, \
    tpl_describe_the_travel_plan_lite
from models import TravelPlan

class TravelPlanWrapper(TravelPlan):
    def describe(self) -> str:
        # Describe the trip feedback observation in a human-readable format
        return tpl_describe_the_travel_plan.render(
            plan=self,
        )

    # ⚠ `is_transfer` AND `not is_terminal`: since ticket 013, the access
    # and egress legs carry `is_transfer=True` (that is what keeps
    # `get_code()` unchanged) but are NOT walking — looking for a parking
    # space is not walking. Without the exclusion, a car plan declared here
    # 10 min of "walking" and 200 m of walking distance, these 200 m coming from the
    # `distance or 100.0` fallback of `Transit.get_distance()` applied to two legs
    # without network distance. The v2 template does not render these two properties for a
    # plan with terminal legs (the `has_terminal_legs` branch comes first), but
    # nothing must depend on that order.
    @property
    def walking_time(self) -> int:
        return sum(leg.get_duration() for leg in self.legs
                   if leg.is_transfer and not leg.is_terminal)

    @property
    def walking_distance(self) -> float:
        return sum(leg.get_distance() for leg in self.legs
                   if leg.is_transfer and not leg.is_terminal)

    # ── Terminal time (ticket 013) ───────────────────────────────────────────
    # The template stays deliberately poor: it renders what these properties
    # give it (decision T3). All the logic — which legs, which
    # labels, which durations — comes from the scenario construction and from
    # `config/terminal_time.yaml`, never from the template.

    @property
    def has_terminal_legs(self) -> bool:
        """Does the plan carry a named access / egress time?

        True for car and bike plans, false for walking (door to door)
        and public transport (walking legs already routed by OTP).
        """
        return any(leg.is_terminal for leg in self.legs)

    @property
    def terminal_time(self) -> int:
        """Sum of the durations of the terminal legs, in seconds."""
        return sum(leg.get_duration() for leg in self.legs if leg.is_terminal)

    @property
    def direct_leg(self):
        """Routed leg of a direct plan (walk / bike / car), or ``None``.

        Recognised by the ``__DIRECT`` marker and not by the length of ``legs``: since
        ticket 013, a car or bike plan counts three of them (access, trip,
        egress) and a test on ``legs | length == 1`` would miss it.
        """
        for leg in self.legs:
            if leg.transit_route and '__DIRECT' in leg.transit_route:
                return leg
        return None

    @property
    def total_seconds(self) -> int:
        """Total duration of the plan in seconds, normalised bounds (ms or s)."""
        from helper import ensure_timestamp_in_seconds

        return (ensure_timestamp_in_seconds(self.end_time)
                - ensure_timestamp_in_seconds(self.start_time))

    @property
    def _terminal_profile(self):
        """Terminal profile of the plan's main mode, or ``None``.

        The mode is read on the NON-terminal leg: the access and
        egress legs carry none, precisely so as not to pollute the mode label of
        the option.
        """
        from trip_helper.terminal_time import terminal_profile

        for leg in self.legs:
            if leg.is_terminal or leg.mode is None:
                continue
            profile = terminal_profile(str(leg.mode))
            if profile is not None:
                return profile
        return None

    @property
    def terminal_label(self) -> str:
        """Qualifier of the header's "of which …", specific to the mode.

        "of walking" would be wrong for a car: looking for a space is not
        walking. The label therefore comes from the mode's profile
        (``of access and parking`` / ``of access and locking``) and not from a
        single formula applied to everything.
        """
        profile = self._terminal_profile
        return profile.labels["terminal"] if profile is not None else "of access"

    @property
    def described_steps(self) -> list[tuple[str, int]]:
        """Sub-steps ``(label, duration in seconds)`` of a plan with named legs.

        The label of the egress leg carries a ``{destination}``: ``purpose``
        is only set on the plan after routing, so the interpolation can only be
        done here. Without a known destination, we fall back on the profile's wording
        that names none — not on an invented name.
        """
        profile = self._terminal_profile
        steps: list[tuple[str, int]] = []
        for leg in self.legs:
            label = leg.step_label
            if not label:
                continue
            if "{destination}" in label:
                label = (profile.egress_label(self.purpose) if profile is not None
                         else label.format(destination=self.purpose or ""))
            steps.append((label, leg.get_duration()))
        return steps

    def summary(self) -> str:
        n_transits = len([leg for leg in self.legs if not leg.is_transfer])
        n_transfers = len(self.legs) - n_transits
        transit_types = [get_transit_route_type(leg.transit_route) for leg in self.legs
                         if leg.transit_route and not leg.is_terminal]
        counter = Counter(transit_types)
        return f"{n_transits} transits, {n_transfers} transfers, including {', '.join([f'{v} {k}' for k, v in counter.items()])}"


class TravelPlanLiteWrapper(TravelPlanWrapper):
    def describe(self) -> str:
        # Describe the trip feedback observation in a human-readable format
        return tpl_describe_the_travel_plan_lite.render(
            plan=self
        )
