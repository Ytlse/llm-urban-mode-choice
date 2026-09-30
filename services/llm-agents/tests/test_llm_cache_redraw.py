"""Redraw of a decision served by the LLM semantic cache.

A hit no longer returns a frozen decision: the cache keeps the distribution produced
by the LLM and redraws from it at every read (cf. `LlmSemanticCache._redraw_from_cached`).
"""

from dataclasses import dataclass, field

from llm.cache import LlmSemanticCache


@dataclass
class _Leg:
    mode: str
    is_terminal: bool = False


@dataclass
class _Plan:
    """Minimal stand-in for TravelPlan: the cache only uses get_code() and mode_label()."""
    code: str
    legs: list = field(default_factory=list)

    def get_code(self) -> str:
        return self.code

    def mode_label(self) -> str:
        """Reproduces the model's contract: terminal legs are excluded.

        Since the terminal time, a car plan carries three legs (access, driving,
        egress) and only the middle one has a mode. The cache reads this
        label, never ``legs`` directly.
        """
        legs = [leg for leg in self.legs if not leg.is_terminal]
        return ",".join(str(leg.mode) for leg in legs) if legs else ""


def _plan(code: str, *modes: str) -> _Plan:
    return _Plan(code=code, legs=[_Leg(mode=m) for m in modes])


CACHED = [
    {"code": "A", "mode": "CAR", "p": 0.7},
    {"code": "B", "mode": "WALK", "p": 0.3},
]
OPTIONS = [_plan("A", "CAR"), _plan("B", "WALK")]


def _redraw(options=OPTIONS, cached=CACHED, seed=(42, "agent-1", "act", "2026-07-29")):
    return LlmSemanticCache._redraw_from_cached(cached, options, seed)


class TestRedrawFromCached:

    def test_meme_contexte_meme_tirage(self):
        assert {_redraw()["index"] for _ in range(20)} == {_redraw()["index"]}

    def test_le_jour_change_le_tirage(self):
        """The same trip, the next day, can give another mode — without an LLM call."""
        indices = {
            _redraw(seed=(42, "agent-1", "act", f"2026-07-{d:02d}"))["index"]
            for d in range(1, 31)
        }
        assert indices == {0, 1}

    def test_mode_et_repartition_rendus(self):
        out = _redraw()
        assert out["mode"] in ("CAR", "WALK")
        assert out["distribution"]["car"] == 0.7
        assert out["distribution"]["walking"] == 0.3
        # A mode never offered stays present at 0 %.
        assert out["distribution"]["cycling"] == 0.0

    def test_option_disparue_ecartee_et_masse_renormalisee(self):
        """Itinerary B no longer exists: A is drawn for sure, despite its stored 0.7."""
        options = [_plan("A", "CAR"), _plan("C", "BICYCLE")]
        for day in range(1, 15):
            out = _redraw(options=options, seed=(42, "a", "act", f"2026-07-{day:02d}"))
            assert out["index"] == 0
            assert out["distribution"]["car"] == 1.0

    def test_plus_aucune_option_connue_donne_un_miss(self):
        options = [_plan("X", "CAR"), _plan("Y", "WALK")]
        assert _redraw(options=options) is None

    def test_probabilites_toutes_nulles_donnent_un_miss(self):
        cached = [{"code": "A", "mode": "CAR", "p": 0}, {"code": "B", "mode": "WALK", "p": 0}]
        assert _redraw(cached=cached) is None

    def test_entrees_corrompues_ignorees(self):
        cached = [{"mode": "CAR", "p": 0.9},            # no code
                  {"code": "A", "mode": "CAR", "p": "?"},  # unreadable probability
                  {"code": "B", "mode": "WALK", "p": 0.3}]
        out = _redraw(cached=cached)
        assert out["index"] == 1 and out["mode"] == "WALK"
