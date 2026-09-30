"""categories/itinary_multi_agent/observe.py — what the worker observes of an itinerary decision.

The LLM returns a distribution over the proposed options, not a choice: the actual draw
happens on the simulation side (``mobility_llm.mode_choice.draw_index`` in the controller). The
worker, for its part, feeds diagnostic counters: most probable mode, probability mass
per mode, distance bracket, disagreements between the mode announced by the LLM and
that of the option. This module is called by the gateway via ``CategorySpec.observe``; it does
not modify the response.

The counter names are those that ``llm_gateway.api.metrics.WorkerMetricsCollector``
reads back from Redis (``transport_mode_chosen:*``, ``mode_by_provider:*``…): changing them
breaks Grafana dashboards 04 and 07.
"""
from __future__ import annotations

from llm_gateway.ports.category import ObserveContext
from loguru import logger
from mobility_core.mode_hierarchy import hierarchy

from mobility_llm.mode_choice import (
    argmax_index,
    canonical_mode,
    mode_distribution,
    normalize_option_probabilities,
)

# Above this rate of mislabelled options over the observed window, an alarm is
# raised: it is no longer noise, the model no longer reads the list it is sent.
MODE_MISMATCH_ALARM_RATIO = 0.05
MODE_MISMATCH_MIN_SAMPLE = 200

# Family of the survey hierarchy → label of the worker counters. The
# vocabulary of these counters is FINE (metro, tram and bus distinct, unlike the
# four categories of the EMC² scoring): it is a diagnostic counter, not a modal
# share. The existing names are kept so that the Redis series stay readable
# from one run to the next.
WORKER_MODE_LABEL = {
    "metro": "metro", "tram": "tram", "cableway": "cableway", "bus": "bus",
    "rail": "train", "car": "car", "motorbike": "motorbike",
    "bicycle": "cycling", "foot": "walking",
}
# Spelled-out synonyms that the hierarchy does not know (it carries the LEG modes of
# OTP/OSMnx, not the labels the LLM may write).
WORKER_MODE_ALIASES = {
    "train": "rail", "ter": "rail", "intercités": "rail", "intercites": "rail",
    "tc": "bus", "transports en commun": "bus", "transport en commun": "bus",
    "voiture": "car", "driving": "car", "conducteur": "car",
    "vélo": "bicycle", "velo": "bicycle", "cycling": "bicycle",
    "marche": "foot", "walking": "foot", "à pied": "foot", "a pied": "foot",
    "pied": "foot",
}


def extract_primary_mode(mode: str) -> str:
    """Reduces a composite mode chain ("foot,bus,foot") to the main mode.

    The order comes from ``mobility_core.mode_hierarchy``, hence from the appendix "Hiérarchie
    des modes" of the survey report: metro > tram > cable car > bus > train > car >
    motorised two-wheeler > bicycle > walking. A chain "bus,train" counts ``bus`` (ticket 022).
    """
    if not mode or mode == "unknown":
        return "unknown"
    parts = {m.strip().lower() for m in mode.split(",")}
    jambes = {WORKER_MODE_ALIASES.get(p, p) for p in parts}
    family = hierarchy().primary_family(jambes)
    if family is None:
        logger.error(f"Unknown or non-standard transport mode in the LLM response | mode={mode}")
        return "other"
    return WORKER_MODE_LABEL[family]


def distance_bracket(distance_m: float) -> str:
    """Classifies a distance in metres into a predefined bracket."""
    if distance_m < 1_000:
        return "0-1km"
    if distance_m < 2_000:
        return "1-2km"
    if distance_m < 5_000:
        return "2-5km"
    if distance_m < 10_000:
        return "5-10km"
    if distance_m < 20_000:
        return "10-20km"
    if distance_m < 50_000:
        return "20-50km"
    return ">50km"


def count_mode_mismatches(metrics, agent_resp, traj_modes: list, provider_name: str) -> None:
    """Compares the mode copied by the LLM with that of the option, index by index.

    This field is redundant on the production side (the real mode comes from the
    trajectories), but that is precisely what makes it a witness: a disagreement signals that
    **the model read another option than the one it thinks it is rating**, and its
    probabilities are then assigned to the wrong indices without anything else showing it.
    """
    total = mismatched = 0
    for entry in agent_resp.probabilities or ():
        announced = getattr(entry, "mode", None)
        if not announced:
            continue
        try:
            i = int(getattr(entry, "index", None))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(traj_modes):
            continue
        total += 1
        if canonical_mode(announced) != canonical_mode(traj_modes[i]):
            mismatched += 1
            logger.warning(
                f"Mode announced by the LLM ≠ mode of the option | index={i} "
                f"annoncé={announced!r} réel={traj_modes[i]!r} "
                f"agent={agent_resp.agent_id} provider={provider_name}"
            )

    if not total:
        return
    metrics.incr("mode_label_checked", amount=total)
    if mismatched:
        metrics.incr("mode_label_mismatch", amount=mismatched)

    checked = metrics.get("mode_label_checked") or 0
    bad = metrics.get("mode_label_mismatch") or 0
    if checked >= MODE_MISMATCH_MIN_SAMPLE and bad / checked > MODE_MISMATCH_ALARM_RATIO:
        # Rising edge: the alarm fires only once. The worker exposes no /metrics:
        # it goes through Redis and comes out as alarme_total{source} on the API side.
        if not metrics.get("alarme:mode_label_mismatch"):
            metrics.incr("alarme:mode_label_mismatch")
            logger.error(
                f"[ALARME] Inconsistent mode labels: {bad}/{checked} "
                f"({100 * bad // checked}%) of the rated options carry a mode that is "
                f"not that of the option. The model confuses the options: the probabilities "
                f"are assigned to the wrong indices and the modal split is wrong."
            )


def observe_itinary(ctx: ObserveContext) -> None:
    """Diagnostic counters of a successful itinerary batch (hook ``CategorySpec.observe``)."""
    metrics = ctx.metrics
    provider_name = ctx.provider
    specs_by_id = {str(getattr(a, "agent_id", "")): a for a in ctx.items}
    for agent_resp in ctx.output.agents:
        spec = specs_by_id.get(str(agent_resp.agent_id))
        trajectories = list(getattr(spec, "trajectories", []) or []) if spec else []
        idx = agent_resp.chosen_index  # fallback: response in the old format

        if agent_resp.probabilities and trajectories:
            # The modes come from the trajectories sent (source of truth), not from those
            # copied by the LLM. They also serve to realign out-of-range indices.
            traj_modes = [t.get("mode") for t in trajectories]
            weights = normalize_option_probabilities(
                agent_resp.probabilities, len(trajectories),
                modes=traj_modes,
                context=f"agent={agent_resp.agent_id} provider={provider_name}",
            )
            for mode, mass in mode_distribution(weights, traj_modes).items():
                metrics.incr(f"mode_probability_pct:{mode}", amount=round(mass * 100))
            idx = argmax_index(weights)
            count_mode_mismatches(metrics, agent_resp, traj_modes, provider_name)

        in_range = idx is not None and trajectories and 0 <= idx < len(trajectories)
        raw_mode = trajectories[idx].get("mode") if in_range else agent_resp.mode
        primary_mode = extract_primary_mode(raw_mode or "unknown")
        metrics.incr(f"transport_mode_chosen:{primary_mode}")
        metrics.incr(f"mode_by_provider:{primary_mode}:{provider_name}")

        if in_range:
            dist_m = float(trajectories[idx].get("total_distance_m") or 0)
            bracket = distance_bracket(dist_m)
            metrics.incr(f"trip_distance_bracket:{bracket}")
            metrics.incr(f"mode_by_distance:{primary_mode}:{bracket}")
            metrics.incr(f"chosen_index:{idx}")


__all__ = [
    "MODE_MISMATCH_ALARM_RATIO",
    "MODE_MISMATCH_MIN_SAMPLE",
    "WORKER_MODE_ALIASES",
    "WORKER_MODE_LABEL",
    "count_mode_mismatches",
    "distance_bracket",
    "extract_primary_mode",
    "observe_itinary",
]
