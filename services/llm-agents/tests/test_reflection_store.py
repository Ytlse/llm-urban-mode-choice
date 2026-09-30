"""Tests for the exact memoisation of STM/LTM reflections.

Contract: a strictly identical reflection prompt (agent, identity, experience,
guidelines, timestamp, LLM parameters) is served from the store without an LLM call;
the slightest differing byte is a miss. No cross-agent matching,
no persisted fallback (an empty reflection is not replayed — D3).

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_reflection_store.py
"""

import pytest

from llm.reflection_store import ReflectionMemoStore

KEY_ARGS = dict(
    person_id="42",
    category="stm_reflection",
    identity="Jeanne, 34 ans, cadre, vélo normal",
    context_text='[{"purpose": "work", "observations": ["trajet fluide"]}]',
    guidelines="",
    departure_timestamp=1773637201.0,
    llm_params={"temperature": 0.7},
)


@pytest.fixture
def store(tmp_path):
    return ReflectionMemoStore(cache_dir=str(tmp_path))


class TestCle:
    def test_deterministe(self):
        assert ReflectionMemoStore.make_key(**KEY_ARGS) == ReflectionMemoStore.make_key(**KEY_ARGS)

    @pytest.mark.parametrize("champ,valeur", [
        ("person_id", "43"),                      # never any cross-agent reuse
        ("category", "ltm_self_reflection"),      # STM and LTM do not mix
        ("identity", "Jeanne, 35 ans, cadre, vélo normal"),
        ("context_text", '[{"purpose": "work", "observations": ["trajet lent"]}]'),
        ("guidelines", "sois bref"),
        ("departure_timestamp", 1773637202.0),
        ("llm_params", {"temperature": 0.8}),     # temperature changes the writing
    ])
    def test_le_moindre_octet_change_la_cle(self, champ, valeur):
        assert ReflectionMemoStore.make_key(**{**KEY_ARGS, champ: valeur}) \
            != ReflectionMemoStore.make_key(**KEY_ARGS)


class TestStore:
    def test_miss_puis_hit(self, store):
        key = ReflectionMemoStore.make_key(**KEY_ARGS)
        assert store.lookup(key, "stm_reflection") is None

        assert store.store(key, "42", "stm_reflection",
                           "Journée fluide, le vélo reste mon meilleur choix.",
                           [["vélo fiable", "mobilité"]], provider="mistral")
        hit = store.lookup(key, "stm_reflection")
        assert hit == {
            "reflection": "Journée fluide, le vélo reste mon meilleur choix.",
            "concepts": [["vélo fiable", "mobilité"]],
            "provider": "mistral",
        }

    def test_pas_de_cross_agent(self, store):
        """The same experience for two agents yields two keys — never the same hit."""
        key_a = ReflectionMemoStore.make_key(**KEY_ARGS)
        key_b = ReflectionMemoStore.make_key(**{**KEY_ARGS, "person_id": "43"})
        store.store(key_a, "42", "stm_reflection", "réflexion de 42", None)
        assert store.lookup(key_b, "stm_reflection") is None

    def test_d3_refuse_le_vide(self, store):
        """An empty reflection with no concept is a generation failure: not persisted."""
        key = ReflectionMemoStore.make_key(**KEY_ARGS)
        assert store.store(key, "42", "stm_reflection", "   ", None) is False
        assert store.lookup(key, "stm_reflection") is None

    def test_vide_mais_concepts_est_persiste(self, store):
        """Concepts without reflection text remain a real output of the model."""
        key = ReflectionMemoStore.make_key(**KEY_ARGS)
        assert store.store(key, "42", "stm_reflection", "", [["ponctualité"]])
        assert store.lookup(key, "stm_reflection")["concepts"] == [["ponctualité"]]

    def test_persistance_entre_instances(self, tmp_path):
        """The store survives a controller restart (cross-process re-runs)."""
        key = ReflectionMemoStore.make_key(**KEY_ARGS)
        ReflectionMemoStore(cache_dir=str(tmp_path)).store(
            key, "42", "stm_reflection", "réflexion persistée", None)
        relu = ReflectionMemoStore(cache_dir=str(tmp_path)).lookup(key, "stm_reflection")
        assert relu["reflection"] == "réflexion persistée"

    def test_stats(self, store):
        key = ReflectionMemoStore.make_key(**KEY_ARGS)
        store.store(key, "42", "stm_reflection", "r", None)
        assert store.stats() == {"total": 1, "by_category": {"stm_reflection": 1}}
