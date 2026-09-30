"""
tests/unit/test_models.py — Unit tests for core/models.py.

Covers: AgentItem, LLMRequest, Task, AgentResponse, LLMOutput, TaskStatus.
(AgentItem, the mobility persona, is tested in mobility_llm/tests/unit/test_persona.py.)
"""

import pytest
from pydantic import ValidationError

from llm_gateway.core.models import (
    _FALLBACK_PRIORITY_SCORE,
    AgentItem,
    AgentResponse,
    LLMOutput,
    LLMRequest,
    Task,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------

class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.SUCCESS == "success"
        assert TaskStatus.FAILED  == "failed"


# ---------------------------------------------------------------------------
# AgentItem — the generic item: an agent_id, the rest passes through as is
# ---------------------------------------------------------------------------

class TestAgentItem:
    def test_agent_id_obligatoire(self):
        with pytest.raises(ValidationError):
            AgentItem()

    def test_les_champs_inconnus_sont_conserves(self):
        a = AgentItem(agent_id="ag1", perception="desc", trajectories=[{"mode": "bus"}])
        dumped = a.model_dump()
        assert dumped["perception"] == "desc"
        assert dumped["trajectories"] == [{"mode": "bus"}]

    def test_agent_id_entier_devient_chaine(self):
        assert AgentItem(agent_id=42).agent_id == "42"


# ---------------------------------------------------------------------------
# LLMRequest
# ---------------------------------------------------------------------------

class TestLLMRequest:
    def _agent(self, agent_id="a1"):
        return {"agent_id": agent_id, "perception": "desc"}

    def test_minimal_valid(self):
        req = LLMRequest(category="itinary_multi_agent", agents=[self._agent()])
        assert req.category == "itinary_multi_agent"
        assert req.parameters == {}
        assert req.force_provider is None

    def test_empty_agents_raises(self):
        with pytest.raises(ValidationError):
            LLMRequest(category="itinary_multi_agent", agents=[])

    def test_multiple_agents(self):
        req = LLMRequest(
            category="perception_filter",
            agents=[self._agent("a1"), self._agent("a2")],
            force_provider="groq",
        )
        assert len(req.agents) == 2
        assert req.force_provider == "groq"

    def test_context_field(self):
        req = LLMRequest(
            category="itinary_multi_agent",
            agents=[self._agent()],
            context="traffic jam on highway",
        )
        assert req.context == "traffic jam on highway"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class TestTask:
    def _request(self):
        return LLMRequest(
            category="itinary_multi_agent",
            agents=[{"agent_id": "a1", "perception": "desc"}],
        )

    def test_default_status_is_pending(self):
        t = Task(request=self._request())
        assert t.status == TaskStatus.PENDING

    def test_task_id_is_uuid(self):
        t = Task(request=self._request())
        assert len(t.task_id) == 36
        assert t.task_id.count("-") == 4

    def test_two_tasks_have_different_ids(self):
        t1 = Task(request=self._request())
        t2 = Task(request=self._request())
        assert t1.task_id != t2.task_id

    def test_fallback_priority_score(self):
        t = Task(request=self._request())
        assert t.priority_score == _FALLBACK_PRIORITY_SCORE

    def test_priority_score_override(self):
        t = Task(request=self._request(), priority_score=1234.5)
        assert t.priority_score == 1234.5

    def test_result_and_error_none_by_default(self):
        t = Task(request=self._request())
        assert t.result is None
        assert t.error is None


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

class TestAgentResponse:
    def test_agent_id_cast_to_str(self):
        a = AgentResponse(agent_id=42)
        assert a.agent_id == "42"
        assert isinstance(a.agent_id, str)

    def test_string_agent_id_unchanged(self):
        a = AgentResponse(agent_id="ag_marc")
        assert a.agent_id == "ag_marc"

    def test_optional_fields_default_none(self):
        a = AgentResponse(agent_id="a1")
        assert a.chosen_index is None
        assert a.mode is None
        assert a.reason is None
        assert a.summary is None

    def test_full_itinary_response(self):
        a = AgentResponse(agent_id="a1", chosen_index=2, mode="bus", reason="cheaper")
        assert a.chosen_index == 2
        assert a.mode == "bus"

    def test_extra_fields_allowed(self):
        a = AgentResponse(agent_id="a1", reflection="some text")
        assert a.reflection == "some text"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LLMOutput
# ---------------------------------------------------------------------------

class TestLLMOutput:
    def test_valid_agents_list(self):
        out = LLMOutput(agents=[AgentResponse(agent_id="a1")])
        assert len(out.agents) == 1

    def test_empty_agents_valid(self):
        out = LLMOutput(agents=[])
        assert out.agents == []
