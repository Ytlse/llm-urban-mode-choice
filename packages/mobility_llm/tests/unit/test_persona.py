"""Tests of the mobility persona (formerly AgentSpec of llm_module.core.models)."""

import pytest
from pydantic import ValidationError

from mobility_llm.persona import AgentSpec, departure_priority


class TestAgentSpec:
    def test_minimal_valid(self):
        a = AgentSpec(agent_id="ag1", perception="desc")
        assert a.agent_id == "ag1"
        assert a.history == []
        assert a.trajectories == []

    def test_full(self):
        a = AgentSpec(
            agent_id="ag2",
            perception="desc",
            destination="city center",
            departure_time="08:00",
            departure_timestamp=1_700_000_000.0,
            current_time="07:55",
            context="rush hour",
            history=["event1"],
            trajectories=[{"mode": "bus"}],
            goal="save time",
            constraints="no car",
            feeling="positive",
        )
        assert a.destination == "city center"
        assert a.departure_timestamp == 1_700_000_000.0
        assert len(a.history) == 1

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            AgentSpec(agent_id="ag1")  # perception missing

    def test_missing_agent_id_raises(self):
        with pytest.raises(ValidationError):
            AgentSpec(perception="desc")  # agent_id missing




class TestDeparturePriority:
    def test_min_des_departs(self):
        items = [AgentSpec(agent_id="a", perception="p", departure_timestamp=200.0),
                 AgentSpec(agent_id="b", perception="p", departure_timestamp=100.0)]
        assert departure_priority(items) == 100.0

    def test_aucun_depart_rend_none(self):
        assert departure_priority([AgentSpec(agent_id="a", perception="p")]) is None


# ---------------------------------------------------------------------------
# The field a template reads must exist on the item model
# ---------------------------------------------------------------------------

def test_tout_champ_lu_par_un_gabarit_est_declare_sur_AgentSpec():
    """`extra="ignore"` drops silently: a template may read a field that never arrives.

    The failure happened twice. `mode_interroge` asked six questions
    about nothing. `evenement` asked the agent to judge a blank
    page: fifteen calls out of fifteen answered `negligible`, without an error anywhere,
    and the defect was first read as "the model does not use the scale".

    This test looks at what the templates READ and confronts it with what the model DECLARES.
    It cannot prove that a declared field is filled — but it makes a third occurrence of
    that pattern impossible.
    """
    import re

    from mobility_llm.persona import AgentSpec
    from mobility_llm.prompts import CATEGORIES_DIR

    declares = set(AgentSpec.model_fields)
    manquants: dict[str, set[str]] = {}
    for gabarit in sorted(CATEGORIES_DIR.glob("*/template.md.j2")):
        lus = set(re.findall(r"\bagent\.([a-z_][a-z0-9_]*)", gabarit.read_text("utf-8")))
        if absents := lus - declares:
            manquants[gabarit.parent.name] = absents
    assert not manquants, (
        f"templates read fields that `AgentSpec` does not declare: {manquants}. "
        f"`extra=\"ignore\"` will drop them silently and the prompt will go out truncated."
    )
