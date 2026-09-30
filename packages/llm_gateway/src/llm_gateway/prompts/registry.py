"""prompts/registry.py — the category registry, validated at startup.

A badly declared category (template missing, schema missing) is refused when the registry
is built, not at the first request. The registry is built explicitly by the
composition (``build_deps``, ``build_worker_runtime``) from the bundles discovered by
entry point, or passed by hand in the tests.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any

from pydantic import BaseModel

from llm_gateway.core.models import _FALLBACK_PRIORITY_SCORE, InternalMessage
from llm_gateway.ports.category import CategoryBundle, CategorySpec, ObserveContext
from llm_gateway.ports.metrics import MetricsSink
from llm_gateway.prompts.engine import PromptManager
from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "llm_gateway.categories"


class UnknownCategoryError(KeyError):
    """Category missing from the registry: no bundle declares it."""


@dataclass(frozen=True)
class CategoryHandle:
    """What the gateway handles for a category: engine + bundle spec."""

    category: str
    bundle: CategoryBundle
    spec: CategorySpec
    prompt_manager: PromptManager

    def validate_items(self, raw_items: Sequence[Any]) -> list[BaseModel]:
        return self.spec.validate_items(raw_items)

    def priority_score(self, items: Sequence[BaseModel]) -> float:
        """Priority score of the batch; the fallback constant if the category defines none."""
        if self.spec.priority is None:
            return _FALLBACK_PRIORITY_SCORE
        score = self.spec.priority(items)
        return _FALLBACK_PRIORITY_SCORE if score is None else float(score)

    def render(self, items: Sequence[BaseModel], parameters: dict[str, Any]) -> list[InternalMessage]:
        return self.prompt_manager.render(self.category, list(items), parameters)

    @property
    def output_schema(self) -> dict[str, Any]:
        return self.prompt_manager.get_output_schema(self.category)

    def observe(self, provider: str, items: Sequence[BaseModel], output: Any, metrics: MetricsSink) -> None:
        if self.spec.observe is None:
            return
        self.spec.observe(ObserveContext(
            category=self.category, provider=provider, items=items, output=output, metrics=metrics,
        ))


class CategoryRegistry:
    def __init__(self, bundles: Iterable[CategoryBundle]) -> None:
        self._handles: dict[str, CategoryHandle] = {}
        self._managers: dict[str, PromptManager] = {}
        self._bundles: list[CategoryBundle] = []
        for bundle in bundles:
            self._register(bundle)
            self._bundles.append(bundle)

    def bundles(self) -> list[CategoryBundle]:
        """The registered bundles, in discovery order."""
        return list(self._bundles)

    def _register(self, bundle: CategoryBundle) -> None:
        manager = PromptManager(
            templates_dir=bundle.templates_dir,
            schemas_file=bundle.schemas_file,
            prompts_file=bundle.prompts_file,
            template_names={n: sp.template_name for n, sp in bundle.categories.items() if sp.template_name},
            schema_paths={n: sp.schema_path for n, sp in bundle.categories.items() if sp.schema_path},
        )
        self._managers[bundle.name] = manager
        for name, spec in bundle.categories.items():
            if name in self._handles:
                other = self._handles[name].bundle.name
                raise ValueError(
                    f"Category {name!r} declared by two bundles ({other!r} and {bundle.name!r})."
                )
            manager.check_category(name)   # template + schema present, else ValueError
            self._handles[name] = CategoryHandle(
                category=name, bundle=bundle, spec=spec, prompt_manager=manager,
            )
        logger.info(
            f"Category bundle registered | bundle={bundle.name} "
            f"categories={sorted(bundle.categories)}"
        )

    @classmethod
    def from_entry_points(cls, group: str = ENTRY_POINT_GROUP) -> CategoryRegistry:
        bundles: list[CategoryBundle] = []
        for ep in entry_points(group=group):
            factory = ep.load()
            bundle = factory() if callable(factory) else factory
            if not isinstance(bundle, CategoryBundle):
                raise TypeError(
                    f"Entry point {ep.name!r} ({ep.value}) must provide a CategoryBundle, "
                    f"got {type(bundle).__name__}."
                )
            bundles.append(bundle)
        if not bundles:
            logger.warning(
                f"No category bundle found under entry point {group!r}: "
                f"every request will be refused (unknown category)."
            )
        return cls(bundles)

    def get(self, category: str) -> CategoryHandle:
        try:
            return self._handles[category]
        except KeyError:
            raise UnknownCategoryError(
                f"Unknown category {category!r}. Registered categories: {self.categories()}"
            ) from None

    def categories(self) -> list[str]:
        return sorted(self._handles)

    def prompt_manager(self, bundle_name: str) -> PromptManager:
        return self._managers[bundle_name]

    def __contains__(self, category: object) -> bool:
        return category in self._handles


@lru_cache(maxsize=1)
def get_registry() -> CategoryRegistry:
    """Process-wide shared registry, built at the first call from the entry points."""
    return CategoryRegistry.from_entry_points()


__all__ = ["CategoryHandle", "CategoryRegistry", "UnknownCategoryError", "get_registry"]
