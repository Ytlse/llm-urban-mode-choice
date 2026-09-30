"""mobility_llm — the LLM categories of the mobility simulation.

This package is the glue between the generic gateway (``llm_gateway``) and the survey
domain (``mobility_core``): it provides the persona model, the templates, the output
schemas, the system prompt variants, the draw of the mode from the distribution returned
by the LLM and the worker's domain metrics.

The gateway finds it via the entry point ``llm_gateway.categories`` (``mobility_llm:bundle``).
Python consumers (controller, experiments, notebooks) go through
:func:`prompt_manager` for the checksum of the active prompt and the list of variants.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from llm_gateway.ports.category import CategoryBundle, CategorySpec, MetricFamilySpec
from llm_gateway.prompts.engine import PromptManager

from mobility_llm.categories.itinary_multi_agent import observe_itinary
from mobility_llm.persona import AgentSpec, departure_priority
from mobility_llm.prompts import CATEGORIES_DIR, PROMPTS_FILE

__version__ = "0.2.0"

BUNDLE_NAME = "mobility"

def _cat(name: str, **kw) -> CategorySpec:
    """A category stored in `categories/<nom>/`: template and schema next to its code."""
    return CategorySpec(
        name=name,
        item_model=AgentSpec,
        priority=departure_priority,
        template_name=f"{name}/template.md.j2",
        schema_path=CATEGORIES_DIR / name / "output_schema.json",
        **kw,
    )


CATEGORIES: dict[str, CategorySpec] = {
    "itinary_multi_agent": _cat("itinary_multi_agent", observe=observe_itinary),
    "perception_filter": _cat("perception_filter"),
    "stm_reflection": _cat("stm_reflection"),
    "ltm_self_reflection": _cat("ltm_self_reflection"),
    "enquete_affinite": _cat("enquete_affinite"),
    # Ticket 100, lot 3 — the agent judges what it has just lived or read (decision D4). One
    # call per EXPOSURE, and exposures are rare by construction: the run of
    # 2026-09-19 counted four over its whole duration.
    "evenement_jugement": _cat("evenement_jugement"),
    # Ticket 111 — the reader passes on to its household: one call per exposed HOUSEHOLD, one
    # message per member (or nothing), and for a minor the parents' decision.
    "evenement_relais": _cat("evenement_relais"),
    # Ticket 120 — TEST ONLY, called by no simulation code: after reading a press article,
    # will the persona change how it travels over the next 7 days? A separate call from
    # `evenement_jugement`, whose template forbids planning.
    "evenement_intention": _cat("evenement_intention"),
}


# Counters fed by categories/itinary_multi_agent.observe_itinary, exposed by the API
# under these names: they are the ones Grafana dashboards 04 and 07 read. Do not rename them
# without updating the dashboards.
METRIC_FAMILIES: tuple[MetricFamilySpec, ...] = (
    MetricFamilySpec("llm_transport_mode_chosen_total", "Main transport modes chosen by the LLM", "transport_mode_chosen", ("mode",)),
    MetricFamilySpec("llm_mode_probability_pct_total", "Sum of the probabilities (in %) the LLM assigns to each mode", "mode_probability_pct", ("mode",)),
    MetricFamilySpec("llm_mode_label_checked_total", "Options scored by the LLM: mode labels checked", "mode_label_checked"),
    MetricFamilySpec("llm_mode_label_mismatch_total", "Options scored by the LLM: mode labels that disagree with the option", "mode_label_mismatch"),
    MetricFamilySpec("llm_trip_distance_bracket_total", "Number of trips per distance bracket", "trip_distance_bracket", ("bracket",)),
    MetricFamilySpec("llm_mode_by_distance_total", "Transport modes per distance bracket", "mode_by_distance", ("mode", "bracket")),
    MetricFamilySpec("llm_mode_by_provider_total", "Transport modes chosen per LLM provider", "mode_by_provider", ("mode", "provider")),
    MetricFamilySpec("llm_chosen_index_total", "Distribution of the trajectory indices chosen by the LLM (0 = first proposed choice)", "chosen_index", ("index",)),
)


def bundle() -> CategoryBundle:
    """The bundle the gateway loads through the entry point."""
    return CategoryBundle(
        name=BUNDLE_NAME,
        templates_dir=CATEGORIES_DIR,
        prompts_file=PROMPTS_FILE,
        categories=dict(CATEGORIES),
        metric_families=METRIC_FAMILIES,
    )


def build_prompt_manager(prompts_file: Path | None = PROMPTS_FILE) -> PromptManager:
    """An engine loaded with the mobility content; `prompts_file` allows other variants (tests)."""
    return PromptManager(
        templates_dir=CATEGORIES_DIR,
        prompts_file=prompts_file,
        template_names={n: sp.template_name for n, sp in CATEGORIES.items() if sp.template_name},
        schema_paths={n: sp.schema_path for n, sp in CATEGORIES.items() if sp.schema_path},
    )


@lru_cache(maxsize=1)
def prompt_manager() -> PromptManager:
    """Prompt engine shared within the process, for consumers outside the gateway
    (active prompt checksum = key of the controller's LLM cache; variants of an experiment)."""
    return build_prompt_manager()


__all__ = [
    "__version__",
    "AgentSpec",
    "BUNDLE_NAME",
    "CATEGORIES",
    "METRIC_FAMILIES",
    "build_prompt_manager",
    "bundle",
    "departure_priority",
    "prompt_manager",
]
