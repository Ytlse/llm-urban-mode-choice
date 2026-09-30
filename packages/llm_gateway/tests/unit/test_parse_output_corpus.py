"""Corpus of LLM outputs (real or reconstructed) against the tolerant parser `_parse_output`.

Adding a case = adding an entry in tests/data/llm_outputs.json: `expect` is the number
of expected agents, or "error" if the response must be refused (ProviderParseError).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_gateway.adapters.base import ProviderParseError
from llm_gateway.testing import FakeAdapter

CORPUS = json.loads((Path(__file__).resolve().parents[1] / "data" / "llm_outputs.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CORPUS, ids=[c["name"] for c in CORPUS])
def test_corpus(case):
    adapter = FakeAdapter()
    if case["expect"] == "error":
        with pytest.raises(ProviderParseError):
            adapter._parse_output(case["raw"])
        return
    output = adapter._parse_output(case["raw"])
    assert len(output.agents) == case["expect"]
    assert all(isinstance(a.agent_id, str) for a in output.agents)
