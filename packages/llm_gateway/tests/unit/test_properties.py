"""Properties (hypothesis) of the pure domain: SWRR sequence and batch key."""
from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_gateway.core.batching import compute_batch_key
from llm_gateway.core.models import AgentItem, LLMRequest
from llm_gateway.core.selection import build_swrr_sequence

names = st.text(alphabet="abcdefghij", min_size=1, max_size=4)
weights_strategy = st.dictionaries(names, st.floats(min_value=0.01, max_value=50), min_size=1, max_size=6)


def _effective(weights: dict[str, float]) -> dict[str, int]:
    total = sum(weights.values())
    return {n: max(1, round((w / total) * 100)) for n, w in weights.items()}


@settings(max_examples=200)
@given(weights_strategy)
def test_la_sequence_swrr_respecte_exactement_les_poids(weights):
    seq = build_swrr_sequence(weights)
    eff = _effective(weights)
    assert len(seq) == sum(eff.values())
    assert Counter(seq) == eff, "each provider appears as many times as its effective weight"


@settings(max_examples=100)
@given(weights_strategy)
def test_la_sequence_swrr_est_deterministe(weights):
    assert build_swrr_sequence(weights) == build_swrr_sequence(dict(weights))


@settings(max_examples=100)
@given(weights_strategy)
def test_le_plus_lourd_ouvre_la_sequence(weights):
    seq = build_swrr_sequence(weights)
    eff = _effective(weights)
    assert eff[seq[0]] == max(eff.values())


agents_strategy = st.lists(
    st.builds(AgentItem, agent_id=st.text(min_size=1, max_size=8)), min_size=1, max_size=5,
)
params_strategy = st.dictionaries(st.text(min_size=1, max_size=6), st.integers(), max_size=4)


@settings(max_examples=100)
@given(cat=st.sampled_from(["a", "b"]), params=params_strategy, a1=agents_strategy, a2=agents_strategy)
def test_la_cle_de_lot_ignore_les_agents_et_prefixe_la_categorie(cat, params, a1, a2):
    r1 = LLMRequest(category=cat, agents=a1, parameters=params)
    r2 = LLMRequest(category=cat, agents=a2, parameters=params)
    assert compute_batch_key(r1) == compute_batch_key(r2)
    assert compute_batch_key(r1).startswith(f"{cat}:")


@settings(max_examples=50)
@given(params=params_strategy, a=agents_strategy)
def test_forcer_un_provider_change_la_cle(params, a):
    base = LLMRequest(category="c", agents=a, parameters=params)
    forced = LLMRequest(category="c", agents=a, parameters=params, force_provider="p")
    assert compute_batch_key(base) != compute_batch_key(forced)
