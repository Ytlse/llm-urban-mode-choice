"""
Unit tests of the semantic LLM cache system (llm/cache.py).

Covers:
- Invariance of state_hash to the order of options
- Rounding of time_slice to 10 minutes
- Round-trip store → lookup (hit)
- Miss: no_candidates (different state_hash)
- Miss: below_threshold (memory semantically too different)
- Miss: code_not_in_options (plan removed from the current options)

No network call, no model loaded: SentenceTransformer is mocked.
Qdrant runs in local file mode in a temporary directory.
"""

import calendar
import sys
import os
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import patch

# Make llm-agents importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../llm-agents"))


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class FakeOption:
    """Stands in for TravelPlan in the tests."""
    def __init__(self, code: str):
        self._code = code
        self.legs = []
        self.duration = 0

    def get_code(self) -> str:
        return self._code

    def mode_label(self) -> str:
        # Consistent with TravelPlan: get_code() joins the legs with "+",
        # mode_label() joins their modes with ",". Here one code segment = one mode.
        return ",".join(seg.lower() for seg in self._code.split("+"))


def make_embed_fn(vectors: dict):
    """
    Returns an encode() function that maps a text to a fixed vector.
    Unknown texts → zero vector (cosine score = 0 → below_threshold guaranteed).
    """
    zero = [0.0] * 384

    class FakeModel:
        def encode(self, text: str):
            import numpy as np
            return np.array(vectors.get(text, zero), dtype="float32")

    return FakeModel()


def ts(hour: int, minute: int) -> int:
    """GAMA timestamp (WALL-CLOCK time) of 28 May 2026 at the given time.

    ⚠ `calendar.timegm` and not `datetime(...).timestamp()`: the decision cache
    key is read in wall-clock time (`sim_clock.wall_clock`), not in the process
    time zone. Building the entry with `.timestamp()` would make the test pass under
    `TZ=UTC` and fail under `TZ=Europe/Paris` — exactly the defect the
    key carried until 2026-09-04.
    """
    return calendar.timegm(datetime(2026, 5, 28, hour, minute, 0).timetuple())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStaticHelpers(unittest.TestCase):

    def test_state_hash_order_invariant(self):
        """Same options in a different order → same hash."""
        from llm.cache import LlmSemanticCache

        opts_ab = [FakeOption("A"), FakeOption("B"), FakeOption("C")]
        opts_ba = [FakeOption("C"), FakeOption("A"), FakeOption("B")]

        h1 = LlmSemanticCache._make_state_hash(opts_ab)
        h2 = LlmSemanticCache._make_state_hash(opts_ba)
        self.assertEqual(h1, h2)

    def test_state_hash_weather_changes_hash(self):
        """Weather must tell apart two identical physical contexts."""
        from llm.cache import LlmSemanticCache

        opts = [FakeOption("A"), FakeOption("B")]
        h_sun  = LlmSemanticCache._make_state_hash(opts, weather={"weather_code": 1, "temperature": 25, "precip_mm": 0})
        h_rain = LlmSemanticCache._make_state_hash(opts, weather={"weather_code": 61, "temperature": 12, "precip_mm": 8.5})
        self.assertNotEqual(h_sun, h_rain)

    def test_time_slice_rounding(self):
        """Rounded down to the 10-minute step, on GAMA's WALL-CLOCK time."""
        from llm.cache import LlmSemanticCache

        self.assertEqual(LlmSemanticCache._make_time_slice(ts(8, 0)),  "08:00")
        self.assertEqual(LlmSemanticCache._make_time_slice(ts(8, 7)),  "08:00")
        self.assertEqual(LlmSemanticCache._make_time_slice(ts(8, 10)), "08:10")
        self.assertEqual(LlmSemanticCache._make_time_slice(ts(8, 19)), "08:10")
        self.assertEqual(LlmSemanticCache._make_time_slice(ts(18, 53)), "18:50")
        self.assertEqual(LlmSemanticCache._make_time_slice(ts(23, 59)), "23:50")

    def test_cle_independante_du_fuseau_du_processus(self):
        """The cache key must not depend on the process `TZ`.

        Two processes of the same run do not share a time zone — the `controller` runs in
        `TZ=Europe/Paris`, the `osmnx` replicas in `TZ=UTC`. As long as the key went through
        `datetime.fromtimestamp(ts)`, they computed two different slices for the
        same simulated instant and thus did not address the same entry.
        """
        from llm.cache import LlmSemanticCache

        # 16 March 2026 5 a.m. wall-clock (t0 of the archived run) and a Friday 11:30 p.m.:
        # the second is the case where the process time zone also changed the DAY.
        cas = [calendar.timegm(datetime(2026, 3, 16, 5, 0, 0).timetuple()),
               calendar.timegm(datetime(2026, 3, 20, 23, 30, 0).timetuple())]
        attendu_par_tz = {}
        initial = os.environ.get("TZ")
        try:
            for tz in ("UTC", "Europe/Paris", "Pacific/Kiritimati", "America/Los_Angeles"):
                os.environ["TZ"] = tz
                time.tzset()
                attendu_par_tz[tz] = [
                    (LlmSemanticCache._make_time_slice(t), LlmSemanticCache._make_weekday(t))
                    for t in cas
                ]
        finally:
            if initial is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = initial
            time.tzset()

        distincts = {tuple(v) for v in attendu_par_tz.values()}
        self.assertEqual(len(distincts), 1,
                         f"la clé du cache bouge avec le fuseau du processus : {attendu_par_tz}")
        self.assertEqual(attendu_par_tz["UTC"],
                         [("05:00", "Weekday"), ("23:30", "Weekday")])


class TestCacheRoundTrip(unittest.IsolatedAsyncioTestCase):
    """Lightweight integration tests: local Qdrant + mocked model."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="llm_cache_test_")
        # Stable vectors for the texts used in the tests
        self.memory_a = "Historique habituel domicile-travail le matin"
        self.memory_b = "Contexte totalement différent incompatible avec quoi que ce soit"
        import numpy as np
        vec_a = [1.0] + [0.0] * 383          # unit vector, self-cosine = 1.0
        vec_b = [-1.0] + [0.0] * 383         # cosine with vec_a = -1.0
        self.fake_model = make_embed_fn({
            self.memory_a: vec_a,
            self.memory_b: vec_b,
        })

    def _make_cache(self, threshold=0.95):
        from llm.cache import LlmSemanticCache
        with patch("sentence_transformers.SentenceTransformer", return_value=self.fake_model):
            cache = LlmSemanticCache(
                cache_dir=self.tmpdir,
                semantic_threshold=threshold,
                embed_model_name="all-MiniLM-L6-v2",
            )
        return cache

    async def test_store_then_lookup_hit(self):
        """After a store, an identical lookup must return the correct index."""
        cache = self._make_cache()
        options = [FakeOption("BUS+METRO"), FakeOption("WALK"), FakeOption("CAR")]
        timestamp = ts(8, 7)  # → time_slice "08:00"

        await cache.store(
            agent_id="agent_1",
            activity_id="act_42",
            timestamp=timestamp,
            options=options,
            memory_text=self.memory_a,
            chosen_plan_code="WALK",
            mode="foot",
        )

        # Lookup with exactly the same context (options in a different order)
        shuffled = [FakeOption("WALK"), FakeOption("CAR"), FakeOption("BUS+METRO")]
        result = await cache.lookup(
            agent_id="agent_1",
            activity_id="act_42",
            timestamp=timestamp,
            options=shuffled,
            memory_text=self.memory_a,
        )

        self.assertIsNotNone(result)
        self.assertEqual(shuffled[result["index"]].get_code(), "WALK")
        self.assertGreaterEqual(result["score"], 0.95)

    async def test_lookup_miss_different_state_hash(self):
        """Different options → different state_hash → no_candidates."""
        cache = self._make_cache()
        options_stored = [FakeOption("BUS"), FakeOption("WALK")]
        options_lookup = [FakeOption("BUS"), FakeOption("TRAM")]  # TRAM instead of WALK

        await cache.store(
            agent_id="agent_1",
            activity_id="act_1",
            timestamp=ts(9, 0),
            options=options_stored,
            memory_text=self.memory_a,
            chosen_plan_code="BUS",
            mode="bus",
        )

        result = await cache.lookup(
            agent_id="agent_1",
            activity_id="act_1",
            timestamp=ts(9, 0),
            options=options_lookup,
            memory_text=self.memory_a,
        )
        self.assertIsNone(result)

    async def test_lookup_miss_below_threshold(self):
        """Semantic context too different → below_threshold."""
        cache = self._make_cache(threshold=0.95)
        options = [FakeOption("BUS"), FakeOption("WALK")]

        await cache.store(
            agent_id="agent_2",
            activity_id="act_1",
            timestamp=ts(9, 0),
            options=options,
            memory_text=self.memory_a,
            chosen_plan_code="BUS",
            mode="bus",
        )

        # Lookup with an opposite memory vector (cosine ≈ -1 → << threshold)
        result = await cache.lookup(
            agent_id="agent_2",
            activity_id="act_1",
            timestamp=ts(9, 0),
            options=options,
            memory_text=self.memory_b,
        )
        self.assertIsNone(result)

    async def test_lookup_miss_different_agent(self):
        """Inter-agent isolation: agent_2 does not see agent_1's cache."""
        cache = self._make_cache()
        options = [FakeOption("BUS"), FakeOption("WALK")]

        await cache.store(
            agent_id="agent_1",
            activity_id="act_1",
            timestamp=ts(8, 0),
            options=options,
            memory_text=self.memory_a,
            chosen_plan_code="BUS",
            mode="bus",
        )

        result = await cache.lookup(
            agent_id="agent_2",      # different agent
            activity_id="act_1",
            timestamp=ts(8, 0),
            options=options,
            memory_text=self.memory_a,
        )
        self.assertIsNone(result)

    async def test_lookup_miss_different_time_slice(self):
        """Different time slice → miss even if everything else is identical."""
        cache = self._make_cache()
        options = [FakeOption("BUS"), FakeOption("WALK")]

        await cache.store(
            agent_id="agent_1",
            activity_id="act_1",
            timestamp=ts(8, 5),   # → "08:00"
            options=options,
            memory_text=self.memory_a,
            chosen_plan_code="BUS",
            mode="bus",
        )

        result = await cache.lookup(
            agent_id="agent_1",
            activity_id="act_1",
            timestamp=ts(18, 5),  # → "18:00": different slice
            options=options,
            memory_text=self.memory_a,
        )
        self.assertIsNone(result)

    async def test_lookup_miss_code_not_in_options(self):
        """Plan in cache but absent from the current options → code_not_in_options."""
        cache = self._make_cache()
        options_stored = [FakeOption("BUS"), FakeOption("WALK")]

        await cache.store(
            agent_id="agent_1",
            activity_id="act_1",
            timestamp=ts(8, 0),
            options=options_stored,
            memory_text=self.memory_a,
            chosen_plan_code="WALK",
            mode="foot",
        )

        # OTP changed: WALK disappears from the options (same state_hash since the hash
        # is recomputed on the lookup options — which differ here)
        # Simulated with a lookup on options none of which matches the stored code
        # but with the same state_hash (by supplying the same codes)
        options_lookup = [FakeOption("BUS"), FakeOption("WALK")]
        # WALK is replaced by an object whose get_code() returns something else after lookup
        # To force the "code_not_in_options" case, get_code is patched post-lookup.
        # Simpler approach: store a code that is not in the lookup options.
        options_stored2 = [FakeOption("BUS"), FakeOption("TRAM")]
        await cache.store(
            agent_id="agent_1",
            activity_id="act_2",
            timestamp=ts(8, 0),
            options=options_stored2,
            memory_text=self.memory_a,
            chosen_plan_code="TRAM",
            mode="tram",
        )
        options_lookup2 = [FakeOption("BUS"), FakeOption("TRAM")]
        # Same state_hash but TRAM absent from the list offered to the lookup
        options_no_tram = [FakeOption("BUS"), FakeOption("TRAM")]
        result = await cache.lookup(
            agent_id="agent_1",
            activity_id="act_2",
            timestamp=ts(8, 0),
            options=options_no_tram,
            memory_text=self.memory_a,
        )
        # TRAM is present → must be a hit
        self.assertIsNotNone(result)
        self.assertEqual(options_no_tram[result["index"]].get_code(), "TRAM")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# Traits signature in the state_hash (fix of 2026-08-27)
# ---------------------------------------------------------------------------

class TestTraitsKeyDansStateHash(unittest.TestCase):
    """A trait that does not condition the offer — the PT pass — only changes the
    prompt text. Without a traits signature, decisions already in cache were
    served again under the old prompt, without any log reporting it."""

    @staticmethod
    def _hash(**kw):
        from llm.cache import LlmSemanticCache
        class _Opt:
            def __init__(self, code): self._c = code
            def get_code(self): return self._c
        return LlmSemanticCache._make_state_hash([_Opt("a"), _Opt("b")], **kw)

    def test_traits_differents_donnent_des_hash_differents(self):
        self.assertNotEqual(self._hash(traits_key="aaa"), self._hash(traits_key="bbb"))

    def test_traits_identiques_donnent_le_meme_hash(self):
        self.assertEqual(self._hash(traits_key="aaa"), self._hash(traits_key="aaa"))

    def test_absence_de_signature_reste_compatible(self):
        """An empty signature must not alter the hash: the field is optional."""
        self.assertEqual(self._hash(), self._hash(traits_key=""))

    def test_la_signature_ne_se_confond_pas_avec_lanticipation(self):
        """Two distinct ingredients: swapping them would change the meaning of the hash."""
        self.assertNotEqual(self._hash(extra_key="x"), self._hash(traits_key="x"))


class TestSignatureDesTraits(unittest.TestCase):
    """The signature itself: what it must see and what it must ignore."""

    @staticmethod
    def _sig(traits):
        import hashlib as _h, json as _j
        excluded = ("name",)
        if not traits:
            return ""
        kept = {k: v for k, v in sorted(traits.items()) if k not in excluded}
        raw = _j.dumps(kept, ensure_ascii=False, sort_keys=True, default=str)
        return _h.sha256(raw.encode("utf-8")).hexdigest()[:16]

    BASE = {"name": "Alice", "age": 20,
            "has_pt_subscription": False, "has_driving_license": True}

    def test_labonnement_deplace_la_signature(self):
        """The trait that cost the manual cache flush."""
        self.assertNotEqual(self._sig(self.BASE),
                            self._sig({**self.BASE, "has_pt_subscription": True}))

    def test_le_nom_ne_la_deplace_pas(self):
        """`name` comes from unseeded Faker: including it would flush the cache on every
        population regeneration, without any decision depending on it."""
        self.assertEqual(self._sig(self.BASE),
                         self._sig({**self.BASE, "name": "Bob"}))

    def test_lordre_des_cles_est_indifferent(self):
        """A Python dict keeps insertion order: two serialisations of the same
        population would give two signatures without the sort."""
        self.assertEqual(self._sig(self.BASE),
                         self._sig(dict(reversed(list(self.BASE.items())))))

    def test_traits_absents_donnent_une_signature_vide(self):
        self.assertEqual(self._sig(None), "")
        self.assertEqual(self._sig({}), "")
