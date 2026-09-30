"""Ranking of itineraries and capping of the candidates presented to the decision-maker.

Two readings of a plan, deliberately distinct (ticket 022): `_primary_mode` returns the
survey's aggregated category (walk / bike / car / public transport) for the metrics;
`_selection_group` derives from it the grouping key of the options offered to the agent
(rail apart), and `_select_candidates` caps the list, keeping the fastest of each group.
Extracted from `simulation_controller.py` (ticket 035, spec 02) so that the GAMA
simulation and the simulator-free run cap **identically**; the controller
re-exports these names.
"""

from loguru import logger

from mobility_core.mode_hierarchy import hierarchy as _mode_hierarchy
from llm_gateway.telemetry.alarms import fire_alarme
from models import TravelPlan

# Survey hierarchy family → metric category. These are the FOUR aggregated categories of
# the survey (walk / bike / private vehicle / public transport, cf. report p. 53: ranks 1
# to 13 form "public transport" and the train is part of it). Merging `rail` into
# `transit` is therefore CORRECT here, and Grafana 07 maps both sides to the same four
# categories.
MODE_HIERARCHY = _mode_hierarchy()

_METRIC_MODE = {
    "metro": "transit", "tram": "transit", "cableway": "transit", "bus": "transit",
    "rail": "transit", "car": "car", "motorbike": "other", "bicycle": "bike",
    "foot": "walk",
}
_unknown_metric_modes: set = set()


def _primary_mode(plan: TravelPlan) -> str:
    """Main mode of a plan, in the survey's four aggregated categories.

    The order comes from `mobility_core.mode_hierarchy` (appendix p. 53 of the report,
    checked against the microdata) and no longer from a cascade written here, which tested
    the **car first** although it is at rank 19, below all public transport (ticket 022).

    ⚠ Is NOT used to answer "does this plan use the car?" — that is
    `_vehicle_mode`. A main mode and a vehicle mode are two different
    quantities; confusing them is the defect this ticket separates.
    """
    modes = {(leg.mode or "").lower() for leg in plan.legs if not leg.is_transfer}
    if not modes - {""}:
        # No motorised leg = a walking trip. This is not a fallback: it is rank 36 of
        # the survey, "Walking ONLY", and it is measured — the 14,842 trips without a
        # detailed route in the microdata are ALL coded 01.
        return "walk"
    family = MODE_HIERARCHY.primary_family(modes)
    if family is None:
        # `transit` was the cascade's DEFAULT: any unknown mode landed there without a
        # single log line, thus inflating the PT share without leaving a trace.
        cle = ",".join(sorted(modes))
        if cle not in _unknown_metric_modes:
            _unknown_metric_modes.add(cle)
            logger.error(
                f"[ALARME] Modes outside the hierarchy in a plan: {{{cle}}} — counted "
                "as \"other\" in trip_mode_by_purpose, no longer as \"transit\". "
                "Complete mobility_core/data/mode_hierarchy_emc2.json.")
            fire_alarme("mode_hierarchie")
        return "other"
    return _METRIC_MODE[family]


# ── Anticipating the day's chain (ticket 014) ─────────────────────────────────
# The choice stays trip by trip, but the persona block of the prompt is enriched:
# weather of the remaining slots (all agents), rolling agenda of the remaining
# trips and position of the vehicles (agents that have something to chain).
# The chain locks remain the safety net — this block informs, it does not
# constrain. The deterministic signature of these texts goes into the key of the
# decision cache (extra_key): two different anticipations cannot serve each
# other a decision.


# The train forms its own option group, distinct from bus and coach. Decision of the
# repository author (2026-09-04), based on measurement: of the 440 points where a **direct
# rail** itinerary exists, 122 (27.7%) lost it to a faster bus + train, because both shared
# the "transit" group. The agent then never saw the train as a choice. The survey's
# hierarchy distinguishes them, by the way: bus and coach are at rank 4, the regional
# train at rank 8 (appendix p. 53).
GROUPE_RAIL = "transit:rail"


def _selection_group(plan: TravelPlan) -> str:
    """Grouping key of the options offered to the agent.

    It is the `_primary_mode` category, **except** public transport, which splits in two:
    rail on one side, the rest of public transport on the other.

    Why only the train, and not one key per family (metro, tram, cable car, bus,
    train). Because the option cap is 6, kept at 6 by decision of 2026-09-04: with
    eight possible groups, the priority pass — which takes the fastest of each new group,
    by increasing duration — could fill the six slots with public-transport variants and
    **drop the car**, the bike or walking. With five groups (walk, bike, car, public
    transport, rail), all five fit and one filler slot remains.

    ⚠ Is NOT used for the metrics: `trip_mode_by_purpose` and the modal shares stay on the
    survey's four categories, where the train is public transport. This key only decides
    what the agent has before its eyes.
    """
    categorie = _primary_mode(plan)
    if categorie != "transit":
        return categorie
    modes = {(leg.mode or "").lower() for leg in plan.legs if not leg.is_transfer}
    return GROUPE_RAIL if MODE_HIERARCHY.primary_family(modes) == "rail" else categorie


def _select_candidates(itineraries: list[TravelPlan], max_n: int) -> list[TravelPlan]:
    """Cap to max_n itineraries, keeping the fastest plan per mode group first."""
    by_duration = sorted(itineraries, key=lambda p: p.duration or float("inf"))
    seen, priority, rest = set(), [], []
    for plan in by_duration:
        mode = _selection_group(plan)
        if mode not in seen:
            seen.add(mode)
            priority.append(plan)
        else:
            rest.append(plan)
    selected = priority[:max_n]
    for plan in rest:
        if len(selected) >= max_n:
            break
        selected.append(plan)
    return selected
