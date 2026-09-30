"""
tests/unit/test_prompt_manager.py — llm_gateway.prompts.engine loaded with the mobility content.

Covers: PromptManager._split_sections, get_output_schema, render.
No need for Redis or an LLM — tests only the templating logic.
"""

import pytest
from llm_gateway.core.models import InternalMessage
from llm_gateway.prompts.engine import _SECTION_SYSTEM, _SECTION_USER

from mobility_llm import build_prompt_manager
from mobility_llm.persona import AgentSpec


# Fixture: isolated instance (the singleton is avoided for tests)
@pytest.fixture
def manager():
    return build_prompt_manager()


# ---------------------------------------------------------------------------
# get_output_schema
# ---------------------------------------------------------------------------

class TestGetOutputSchema:
    KNOWN_CATEGORIES = [
        "itinary_multi_agent",
        "perception_filter",
        "stm_reflection",
        "ltm_self_reflection",
    ]

    def test_known_categories_return_dicts(self, manager):
        for cat in self.KNOWN_CATEGORIES:
            schema = manager.get_output_schema(cat)
            assert isinstance(schema, dict)
            assert "type" in schema

    def test_unknown_category_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown schema"):
            manager.get_output_schema("non_existent_category")


# ---------------------------------------------------------------------------
# _split_sections
# ---------------------------------------------------------------------------

class TestSplitSections:
    def test_no_markers_returns_single_user_message(self, manager):
        text = "Hello world, this is the prompt."
        messages = manager._split_sections(text)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == text

    def test_system_marker_only(self, manager):
        text = f"{_SECTION_SYSTEM}\nThis is the system instruction."
        messages = manager._split_sections(text)
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert "system instruction" in messages[0].content

    def test_user_marker_only(self, manager):
        text = f"{_SECTION_USER}\nThis is the user message."
        messages = manager._split_sections(text)
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_system_then_user(self, manager):
        text = (
            f"{_SECTION_SYSTEM}\nSystem part.\n"
            f"{_SECTION_USER}\nUser part."
        )
        messages = manager._split_sections(text)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "System part" in messages[0].content
        assert "User part" in messages[1].content

    def test_empty_section_is_skipped(self, manager):
        # USER section without content → must not generate an empty message
        text = f"{_SECTION_SYSTEM}\nSystem content.\n{_SECTION_USER}\n   "
        messages = manager._split_sections(text)
        assert len(messages) == 1
        assert messages[0].role == "system"

    def test_no_empty_content_in_result(self, manager):
        text = f"{_SECTION_SYSTEM}\nHello.\n{_SECTION_USER}\nWorld."
        messages = manager._split_sections(text)
        for m in messages:
            assert m.content
            assert m.content.strip()

    def test_returns_internal_message_objects(self, manager):
        text = f"{_SECTION_USER}\nSome content."
        messages = manager._split_sections(text)
        assert all(isinstance(m, InternalMessage) for m in messages)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

class TestRender:
    def _make_agent(self, agent_id="ag1"):
        return AgentSpec(agent_id=agent_id, perception="Test perception")

    def test_render_perception_filter_returns_messages(self, manager):
        agents = [self._make_agent("a1"), self._make_agent("a2")]
        messages = manager.render("perception_filter", agents, {})
        assert len(messages) >= 1
        # At least one message must contain the agent_ids
        all_content = " ".join(m.content for m in messages if m.content)
        assert "a1" in all_content or "a2" in all_content

    def test_render_itinary_multi_agent_has_system_message(self, manager):
        agent = AgentSpec(
            agent_id="a1",
            perception="38y, engineer",
            trajectories=[{"mode": "bus", "total_distance_m": 5000}],
        )
        messages = manager.render("itinary_multi_agent", [agent], {})
        roles = [m.role for m in messages]
        assert "system" in roles

    def test_render_unknown_category_raises(self, manager):
        with pytest.raises(ValueError, match="Template error"):
            manager.render("unknown_category", [self._make_agent()], {})

    def test_render_includes_schema_in_output(self, manager):
        agent = self._make_agent()
        messages = manager.render("perception_filter", [agent], {})
        all_content = " ".join(m.content for m in messages if m.content)
        # The JSON schema is injected into the template → "agents" must appear
        assert "agents" in all_content

    def test_render_context_injected_when_provided(self, manager):
        agent = AgentSpec(
            agent_id="a1",
            perception="student",
            trajectories=[{"mode": "bus", "total_distance_m": 2000}],
        )
        messages = manager.render(
            "itinary_multi_agent",
            [agent],
            {"context": "heavy_rain_today"},
        )
        all_content = " ".join(m.content for m in messages if m.content)
        assert "heavy_rain_today" in all_content


# ---------------------------------------------------------------------------
# get_system_prompt — single source of the system prompts (prompts.yaml)
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_active_category_returns_prompt(self, manager):
        # itinary_multi_agent is mapped in active: → a non-empty prompt
        sp = manager.get_system_prompt("itinary_multi_agent")
        assert sp and len(sp) > 50

    def test_schema_block_stripped(self, manager):
        """The block "Expected JSON schema: {...}" is removed from the system prompt.

        BOTH spellings are checked, and it is not fussiness:
        the English switch translated this title in prompts.yaml, but the
        archived variants carry the French form and are still re-read to recompute
        their fingerprints. A test that knew only one form would let through
        the regression on the other — and what would come out is a schema SERVED TWICE.
        """
        sp = manager.get_system_prompt("itinary_multi_agent")
        assert "Expected JSON schema" not in sp
        assert "Schéma JSON attendu" not in sp

    def test_inactive_category_returns_none(self, manager):
        # Category absent from active: → the template keeps its hard-coded SYSTEM
        assert manager.get_system_prompt("perception_filter") is None

    def test_render_injects_active_system_prompt(self, manager):
        agent = AgentSpec(
            agent_id="a1",
            perception="38y, engineer",
            trajectories=[{"mode": "bus", "total_distance_m": 5000}],
        )
        messages = manager.render("itinary_multi_agent", [agent], {})
        sys_msg = next(m for m in messages if m.role == "system")
        # The schema appears only ONCE, injected by the template — never duplicated by
        # a block left in the variant's text. Both spellings are counted
        # together: what matters is the number of schema titles, not their language.
        titres = sum(sys_msg.content.count(t)
                     for t in ("Expected JSON schema", "Schéma JSON attendu"))
        assert titres == 1, f"{titres} schema title(s) in the served system prompt"
        assert '"agent_id"' in sys_msg.content  # schema properly injected


# ---------------------------------------------------------------------------
# active_prompt_checksum — isolation of the LLM cache per prompt version
# ---------------------------------------------------------------------------

class TestPromptChecksum:
    def test_checksum_is_stable(self, manager):
        assert manager.active_prompt_checksum() == manager.active_prompt_checksum()

    def test_checksum_length(self, manager):
        assert len(manager.active_prompt_checksum(length=8)) == 8

    def test_checksum_changes_when_prompt_changes(self, manager, tmp_path):
        import yaml

        base = manager.active_prompt_checksum("itinary_multi_agent")
        # Alternative store with a different prompt content → different checksum
        alt = tmp_path / "prompts.yaml"
        alt.write_text(yaml.safe_dump({
            "active": {"itinary_multi_agent": "v"},
            "prompts": {"v": {"content": "A completely different system prompt."}},
        }), encoding="utf-8")
        other = build_prompt_manager(prompts_file=alt).active_prompt_checksum("itinary_multi_agent")
        assert base != other

    def test_checksum_empty_when_no_active(self, tmp_path):
        import yaml

        empty = tmp_path / "prompts.yaml"
        empty.write_text(yaml.safe_dump({"active": {}, "prompts": {}}), encoding="utf-8")
        # No active category → fingerprint of the empty set, but deterministic and non-empty
        cs = build_prompt_manager(prompts_file=empty).active_prompt_checksum()
        assert isinstance(cs, str) and len(cs) == 12
