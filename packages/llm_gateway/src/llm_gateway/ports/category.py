"""ports/category.py — the contract between the gateway and a category "bundle".

The gateway knows no business domain. What it can do: receive batches of items per
category, render a prompt, call a provider, validate the output against a schema,
demultiplex by ``agent_id``. Everything specific to a domain (the model of an item, the
priority of a batch, the business metrics to draw from a response) comes from a bundle,
discovered through the ``llm_gateway.categories`` entry point::

    [project.entry-points."llm_gateway.categories"]
    mobility = "mobility_llm:bundle"

where ``bundle()`` returns a :class:`CategoryBundle`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_gateway.core.models import AgentItem, LLMOutput
from llm_gateway.ports.metrics import MetricsSink


@dataclass(frozen=True)
class ObserveContext:
    """What the worker passes to the bundle once the response is validated.

    ``items`` are the items validated by :attr:`CategorySpec.item_model` (in the order of
    the batch), ``output`` the response already realigned on the expected ``agent_id``. The
    bundle modifies neither of them: it observes and counts into ``metrics``.
    """

    category: str
    provider: str
    items: Sequence[BaseModel]
    output: LLMOutput
    metrics: MetricsSink


@dataclass(frozen=True)
class CategorySpec:
    """A prompt category: its item model and its optional hooks."""

    name: str
    # Pydantic model that validates each payload item. None = generic AgentItem.
    item_model: type[BaseModel] = AgentItem
    # Priority score of a batch (lower = more urgent). None = no priority.
    priority: Callable[[Sequence[BaseModel]], float | None] | None = None
    # Business metrics drawn from a successful response.
    observe: Callable[[ObserveContext], None] | None = None
    # Per-category layout (optional): template name relative to `templates_dir` (default
    # `<category>.md.j2`) and JSON file of the output schema (overrides `schemas_file`).
    template_name: str | None = None
    schema_path: Path | None = None

    def validate_items(self, raw_items: Sequence[Any]) -> list[BaseModel]:
        """Validates the payload items with the category's model."""
        model = self.item_model
        out: list[BaseModel] = []
        for raw in raw_items:
            data = raw.model_dump() if isinstance(raw, BaseModel) else raw
            out.append(model.model_validate(data))
        return out


@dataclass(frozen=True)
class MetricFamilySpec:
    """A Prometheus family the bundle wants exposed from the worker's counters.

    The worker exposes no /metrics: its counters live in the metrics sink (Redis
    hash) under keys ``<prefix>:<label1>:<label2>…``. The API reads them back and serves
    them under the name declared here. Without a label, the key is exactly ``redis_prefix``.
    """

    name: str            # Prometheus name, e.g. llm_transport_mode_chosen_total
    help: str
    redis_prefix: str    # e.g. transport_mode_chosen (without the trailing colon)
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryBundle:
    """A set of categories shipped with its templates, schemas and prompt variants.

    ``templates_dir`` contains one ``<category>.md.j2`` per category, ``schemas_file`` a
    JSON object ``{category: output schema}``, ``prompts_file`` (optional) the
    ``active:`` / ``prompts:`` file of the system prompt variants.
    """

    name: str
    templates_dir: Path
    schemas_file: Path | None = None
    categories: dict[str, CategorySpec] = field(default_factory=dict)
    prompts_file: Path | None = None
    # Business counters that ``observe`` feeds and the API must expose (cf. api/metrics.py).
    metric_families: tuple[MetricFamilySpec, ...] = ()

    def spec(self, category: str) -> CategorySpec:
        return self.categories[category]


__all__ = ["CategoryBundle", "CategorySpec", "MetricFamilySpec", "ObserveContext"]
