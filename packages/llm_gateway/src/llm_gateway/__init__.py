"""llm_gateway — asynchronous multi-provider LLM gateway.

Public facade. Anything not listed in ``__all__`` is internal and may change
without notice; deep paths remain importable but are not a contract.

Client-side usage ::

    from llm_gateway import LLMGatewayClient, LLMRequest

    client = LLMGatewayClient(base_url="http://localhost:8000")
    result = await client.execute(LLMRequest(category="...", agents=[...]))

Service-side usage: ``create_app`` (FastAPI API) and ``create_celery_app`` (worker). Prompt
categories are provided by bundles registered under the entry point
``llm_gateway.categories`` (cf. :mod:`llm_gateway.ports.category`).
"""
from __future__ import annotations

__version__ = "1.3.0"

from typing import TYPE_CHECKING, Any  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - pour les IDE et mypy uniquement
    from llm_gateway.core.models import (  # noqa: F401
        AgentItem,
        AgentResponse,
        LLMOutput,
        LLMRequest,
        Task,
        TaskStatus,
        TaskStatusResponse,
    )
    from llm_gateway.ports.category import (  # noqa: F401
        CategoryBundle,
        CategorySpec,
        ObserveContext,
    )
    from llm_gateway.sdk import LLMGatewayClient, TaskResult, TaskTiming  # noqa: F401

# Each public name → the module that holds it. Resolved on first access (PEP 562): importing
# `llm_gateway` loads neither httpx nor prometheus_client nor FastAPI; a process only embeds
# what it uses (the API does not have the client metrics, the worker does not have FastAPI).
_LAZY: dict[str, str] = {
    "AgentItem": "llm_gateway.core.models",
    "AgentResponse": "llm_gateway.core.models",
    "LLMOutput": "llm_gateway.core.models",
    "LLMRequest": "llm_gateway.core.models",
    "Task": "llm_gateway.core.models",
    "TaskStatus": "llm_gateway.core.models",
    "TaskStatusResponse": "llm_gateway.core.models",
    "CategoryBundle": "llm_gateway.ports.category",
    "CategorySpec": "llm_gateway.ports.category",
    "ObserveContext": "llm_gateway.ports.category",
    "LLMGatewayClient": "llm_gateway.sdk",
    "TaskResult": "llm_gateway.sdk",
    "TaskTiming": "llm_gateway.sdk",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'llm_gateway' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cached: a single import
    return value


def create_app(*args: Any, **kwargs: Any) -> Any:
    """FastAPI application factory (lazy import: FastAPI is only loaded here)."""
    from llm_gateway.api.app import create_app as _create_app

    return _create_app(*args, **kwargs)


def create_celery_app(*args: Any, **kwargs: Any) -> Any:
    """Celery application factory (lazy import: Celery is only loaded here)."""
    from llm_gateway.worker.app import create_celery_app as _create

    return _create(*args, **kwargs)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    *sorted(_LAZY),
    "create_app",
    "create_celery_app",
]
