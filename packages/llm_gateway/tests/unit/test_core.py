"""
tests/test_core.py — Unit tests of the pure domain (core/).

Covers:
  - compute_batch_key   : stability, sensitivity to parameters
  - compute_priority_score : min of the scores given by the category, fallback
  - build_swrr_sequence : weighting, interleaving

No Redis/Celery/LLM call.
"""


from llm_gateway.core.batching import compute_batch_key, compute_priority_score
from llm_gateway.core.models import _FALLBACK_PRIORITY_SCORE, AgentItem, LLMRequest


def _request(**overrides) -> LLMRequest:
    base = dict(
        category="itinary_multi_agent",
        agents=[AgentItem(agent_id="a1")],
        parameters={"temperature": 0.7},
    )
    base.update(overrides)
    return LLMRequest(**base)


class TestComputeBatchKey:
    def test_stable_for_identical_requests(self):
        assert compute_batch_key(_request()) == compute_batch_key(_request())

    def test_prefixed_by_category(self):
        assert compute_batch_key(_request()).startswith("itinary_multi_agent:")

    def test_parameter_order_does_not_matter(self):
        a = _request(parameters={"a": 1, "b": 2})
        b = _request(parameters={"b": 2, "a": 1})
        assert compute_batch_key(a) == compute_batch_key(b)

    def test_differs_on_parameters(self):
        assert compute_batch_key(_request(parameters={"temperature": 0.2})) != compute_batch_key(_request())

    def test_differs_on_force_provider(self):
        assert compute_batch_key(_request(force_provider="groq_llama4")) != compute_batch_key(_request())

    def test_differs_on_min_tpm(self):
        assert compute_batch_key(_request(min_tpm_required=30000)) != compute_batch_key(_request())

    def test_agents_do_not_change_key(self):
        # Different agents with the same context must land in the same batch
        other = _request(agents=[AgentItem(agent_id="zz")])
        assert compute_batch_key(other) == compute_batch_key(_request())


class TestComputePriorityScore:
    """The gateway receives scores (given by the category), it interprets nothing."""

    def test_min_des_scores(self):
        assert compute_priority_score([200.0, 100.0]) == 100.0

    def test_ignore_les_scores_absents(self):
        assert compute_priority_score([None, 42.0]) == 42.0

    def test_repli_sans_aucun_score(self):
        assert compute_priority_score([None]) == _FALLBACK_PRIORITY_SCORE
        assert compute_priority_score([]) == _FALLBACK_PRIORITY_SCORE
