"""The mobility bundle, from payload to metrics hook, via the `llm_gateway.testing` tooling.

It is the proof of adoption of `llm_gateway.testing` by a consumer: registry built by
hand, FakeAdapter without network, in-memory metrics sink.
"""
from __future__ import annotations

import pytest
from llm_gateway.core.models import LLMOutput
from llm_gateway.testing import FakeAdapter, InMemoryMetricsSink, build_registry
from pydantic import ValidationError

from mobility_llm import BUNDLE_NAME, CATEGORIES, bundle
from mobility_llm.persona import AgentSpec

PERSONA = {
    "agent_id": "ag_418",
    "perception": "Woman, 34, manager, no car, public transport pass holder.",
    "destination": "Compans-Caffarelli",
    "departure_time": "08:10",
    "departure_timestamp": 1_700_000_000.0,
    "trajectories": [
        {"index": 0, "mode": "foot,bus,foot", "total_distance_m": 4200, "description": "bus L1"},
        {"index": 1, "mode": "bicycle", "total_distance_m": 3900, "description": "vélo"},
    ],
}


@pytest.fixture(scope="module")
def registry():
    return build_registry(bundle())


def test_le_bundle_declare_les_quatre_categories(registry):
    assert registry.categories() == sorted(CATEGORIES)
    assert registry.get("itinary_multi_agent").bundle.name == BUNDLE_NAME


def test_un_persona_sans_perception_est_refuse(registry):
    with pytest.raises(ValidationError):
        registry.get("itinary_multi_agent").validate_items([{"agent_id": "x"}])


def test_priorite_du_lot_est_le_depart_le_plus_tot(registry):
    handle = registry.get("itinary_multi_agent")
    items = handle.validate_items([PERSONA, {**PERSONA, "agent_id": "ag_2", "departure_timestamp": 1_600.0}])
    assert isinstance(items[0], AgentSpec)
    assert handle.priority_score(items) == 1_600.0


def test_rendu_puis_reponse_puis_metriques(registry):
    handle = registry.get("itinary_multi_agent")
    items = handle.validate_items([PERSONA])
    messages = handle.render(items, {})
    assert messages[0].role == "system" and messages[-1].role == "user"
    assert "ag_418" in (messages[-1].content or "")

    # The fake model rates both options: the first one wins.
    fake = FakeAdapter(responder=lambda aid: {
        "agent_id": aid,
        "probabilities": [
            {"index": 0, "mode": "bus", "probability": 70},
            {"index": 1, "mode": "vélo", "probability": 30},
        ],
    })
    from llm_gateway.core.models import InternalRequest
    output, tokens_in, tokens_out = fake.call(
        InternalRequest(provider="fake", messages=messages, response_schema=handle.output_schema)
    )
    assert isinstance(output, LLMOutput) and tokens_in > 0 and tokens_out > 0

    metrics = InMemoryMetricsSink()
    handle.observe("fake", items, output, metrics)
    assert metrics.get("transport_mode_chosen:bus") == 1
    assert metrics.get("mode_by_provider:bus:fake") == 1
    assert metrics.get("trip_distance_bracket:2-5km") == 1
    assert metrics.get("chosen_index:0") == 1
    assert metrics.get("mode_probability_pct:public_transport") == 70, "canonical vocabulary"
    assert metrics.get("mode_probability_pct:cycling") == 30
    assert metrics.get("mode_label_checked") == 2 and metrics.get("mode_label_mismatch") == 0
