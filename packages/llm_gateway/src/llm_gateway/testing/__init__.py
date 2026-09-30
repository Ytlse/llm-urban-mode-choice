"""testing — what a consumer or a test needs to plug in without Redis or network.

* the in-memory implementations of the ports (re-exported from ``infra.memory``);
* :func:`echo_bundle`: a bundle with one ``echo`` category that returns each item, to
  test the gateway without any domain logic;
* :class:`FakeAdapter`: a deterministic LLM adapter that answers the requested schema;
* :func:`build_registry`: a registry built by hand, without entry point.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_gateway.adapters.base import BaseAdapter
from llm_gateway.core.models import AgentResponse, InternalRequest, LLMOutput
from llm_gateway.infra.memory import (
    FileLearnedLimits,
    InMemoryBatchQueue,
    InMemoryLearnedLimits,
    InMemoryMetricsSink,
    InMemoryRateLimiter,
    InMemoryTaskStore,
)
from llm_gateway.ports.category import CategoryBundle, CategorySpec
from llm_gateway.prompts.registry import CategoryRegistry

_HERE = Path(__file__).resolve().parent
ECHO_TEMPLATES_DIR = _HERE / "templates"
ECHO_SCHEMAS_FILE = _HERE / "echo_schemas.json"


def echo_bundle() -> CategoryBundle:
    """A minimal bundle: the ``echo`` category asks the model to repeat each item."""
    return CategoryBundle(
        name="echo",
        templates_dir=ECHO_TEMPLATES_DIR,
        schemas_file=ECHO_SCHEMAS_FILE,
        categories={"echo": CategorySpec(name="echo")},
    )


def build_registry(*bundles: CategoryBundle) -> CategoryRegistry:
    """Registry built without entry point; by default only the ``echo`` bundle."""
    return CategoryRegistry(bundles or (echo_bundle(),))


class FakeAdapter(BaseAdapter):
    """Network-free adapter: returns a valid response for each ``agent_id`` of the prompt.

    ``responder`` lets a test build the response (one ``dict`` per agent); by default
    each agent receives ``{"agent_id": …, "summary": "echo"}``. The token counters are
    the length of the messages, so that the metrics move.
    """

    provider_name = "fake"

    def __init__(self, responder: Callable[[str], dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._responder = responder or (lambda aid: {"agent_id": aid, "summary": "echo"})
        self.calls: list[InternalRequest] = []

    def call(self, request: InternalRequest) -> tuple[LLMOutput, int, int]:
        self.calls.append(request)
        text = "\n".join(m.content or "" for m in request.messages)
        agent_ids = _agent_ids_in(text)
        agents = [AgentResponse.model_validate(self._responder(aid)) for aid in agent_ids]
        tokens_in = max(1, len(text) // 4)
        return LLMOutput(agents=agents), tokens_in, max(1, 8 * len(agents))

    def close(self) -> None:  # no HTTP client to close
        return None


_AGENT_ID_RE = re.compile(r"agent_id=([^\s|,;]+)")


def _agent_ids_in(text: str) -> list[str]:
    """The ``agent_id=…`` written in the prompt (echo template at the head of the block, mobility
    in the header of each persona), in order, without duplicates."""
    seen: dict[str, None] = {}
    for m in _AGENT_ID_RE.finditer(text):
        seen.setdefault(m.group(1).strip(), None)
    return list(seen)


def load_echo_schema() -> dict[str, Any]:
    return json.loads(ECHO_SCHEMAS_FILE.read_text(encoding="utf-8"))["echo"]


__all__ = [
    "ECHO_SCHEMAS_FILE",
    "ECHO_TEMPLATES_DIR",
    "FakeAdapter",
    "FileLearnedLimits",
    "InMemoryBatchQueue",
    "InMemoryLearnedLimits",
    "InMemoryMetricsSink",
    "InMemoryRateLimiter",
    "InMemoryTaskStore",
    "build_registry",
    "echo_bundle",
    "load_echo_schema",
]
