"""Thinking depth on the Google side — hygiene spec, "thinking budget" item.

The central trap: thinking is drawn from the OUTPUT budget. A thinking budget
requested without a reserve truncates the response (MAX_TOKENS) and the call is lost.
"""

from __future__ import annotations

import pytest

from llm_gateway.adapters.google_adapter import RESERVE_REFLEXION, GoogleAdapter
from llm_gateway.core.models import InternalMessage, InternalRequest


def _req(**kw) -> InternalRequest:
    base = dict(provider="google", messages=[InternalMessage(role="user", content="x")],
                response_schema={"type": "object"}, temperature=0.0, max_tokens=4096)
    base.update(kw)
    return InternalRequest(**base)


def _config(request) -> dict:
    """The `generationConfig` the adapter ACTUALLY builds, without a network call.

    ⚠ This function used to copy its construction instead of calling it. It therefore checked
    its own copy, and the LEVEL setting — which the adapter placed next to
    `maxOutputTokens`, where the API refuses it — did not even appear in it. The defect went to
    production without a test seeing it: every call of the gemini-3.5-flash-lite arm
    came back as 400 on 2026-09-11, and the whole experiment fell over.

    `_generation_config` was extracted from `call` to be callable here as is. Two
    implementations of the same payload always end up diverging; there is now only
    one.
    """
    a = GoogleAdapter()
    a._instance_name = "google_test"
    # The only check that queries the instance configuration: the level tests
    # below exercise it separately, on the real method.
    a._refuser_niveau_non_supporte = lambda _niveau: None  # type: ignore[method-assign]
    return a._generation_config(request)


def test_google_declare_appliquer_la_reflexion():
    assert GoogleAdapter.applique_reflexion is True


def test_rien_n_est_envoye_sans_budget():
    """Behaviour from before 2026-09-10, kept: the provider keeps its default."""
    cfg = _config(_req())
    assert "thinkingConfig" not in cfg
    assert cfg["maxOutputTokens"] == 4096


def test_budget_zero_desactive_sans_reserver():
    cfg = _config(_req(thinking_budget=0))
    assert cfg["thinkingConfig"]["thinkingBudget"] == 0
    assert cfg["maxOutputTokens"] == 4096, "thinking off: no reserve to take"


def test_budget_positif_est_ajoute_au_plafond_de_sortie():
    """Without this reserve, thinking eats the response and the call ends in MAX_TOKENS."""
    cfg = _config(_req(thinking_budget=1024))
    assert cfg["thinkingConfig"]["thinkingBudget"] == 1024
    assert cfg["maxOutputTokens"] == 4096 + 1024


def test_budget_dynamique_reserve_un_forfait():
    """-1 lets the model decide: we do not know how much, we reserve a flat amount."""
    cfg = _config(_req(thinking_budget=-1))
    assert cfg["thinkingConfig"]["thinkingBudget"] == -1
    assert cfg["maxOutputTokens"] == 4096 + RESERVE_REFLEXION


def test_les_pensees_ne_sont_pas_rapatriees():
    """`includeThoughts: False` — the trace has no use for the text of the thinking."""
    assert _config(_req(thinking_budget=512))["thinkingConfig"]["includeThoughts"] is False


@pytest.mark.parametrize("adapter_mod,classe", [
    ("llm_gateway.adapters.openai_compatible", "OpenAICompatibleAdapter"),
    ("llm_gateway.adapters.mistral_adapter", "MistralAdapter"),
    ("llm_gateway.adapters.groq_adapter", "GroqAdapter"),
    ("llm_gateway.adapters.cerebras_adapter", "CerebrasAdapter"),
])
def test_les_autres_adapters_ne_pretendent_pas_l_appliquer(adapter_mod, classe):
    """A setting sealed in the fingerprint and never applied is the defect found on `t0`."""
    import importlib

    cls = getattr(importlib.import_module(adapter_mod), classe)
    assert cls.applique_reflexion is False


def test_l_avertissement_ne_part_qu_une_fois(caplog):
    from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter

    a = OpenAICompatibleAdapter()
    a._instance_name = "lmstudio_test"
    a._signaler_reflexion_ignoree(1024)
    assert a._reflexion_signalee is True
    a._signaler_reflexion_ignoree(1024)  # must not raise the flag a second time


def test_aucun_avertissement_sans_budget():
    from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter

    a = OpenAICompatibleAdapter()
    a._instance_name = "lmstudio_test"
    a._signaler_reflexion_ignoree(None)
    assert a._reflexion_signalee is False


# ── Thinking level — the current API setting (noted on 2026-09-10) ──


def test_niveau_est_emis_tel_quel(monkeypatch):
    from llm_gateway.adapters import google_adapter as G

    a = G.GoogleAdapter()
    a._instance_name = "google_test"
    monkeypatch.setattr(G, "get_settings", lambda: type("S", (), {"providers": {}})())
    a._refuser_niveau_non_supporte("high")  # no declaration: lets it through


def test_niveau_non_declare_ne_bloque_pas(monkeypatch):
    """Without `thinking_levels`, we do not block on a guessed list."""
    from llm_gateway.adapters import google_adapter as G

    a = G.GoogleAdapter()
    a._instance_name = "i"
    cfg = type("C", (), {"thinking_levels": None, "default_model": "m"})()
    monkeypatch.setattr(G, "get_settings", lambda: type("S", (), {"providers": {"i": cfg}})())
    a._refuser_niveau_non_supporte("minimal")


def test_niveau_hors_liste_refuse(monkeypatch):
    """`minimal` on a model that does not declare it: 400 on every call, better say it here."""
    import pytest

    from llm_gateway.adapters import google_adapter as G
    from llm_gateway.adapters.base import ProviderClientError

    a = G.GoogleAdapter()
    a._instance_name = "i"
    cfg = type("C", (), {"thinking_levels": ["low", "medium", "high"], "default_model": "gemini-3.8-flash"})()
    monkeypatch.setattr(G, "get_settings", lambda: type("S", (), {"providers": {"i": cfg}})())
    with pytest.raises(ProviderClientError, match="not accepted"):
        a._refuser_niveau_non_supporte("minimal")


def test_niveau_et_budget_ensemble_refuses_par_la_cascade():
    """The API returns 400 if both coexist: the cascade refuses it before the call."""
    from types import SimpleNamespace

    import pytest

    from llm_gateway.core.inference import ReglagesReflexionIncompatibles, resolve_inference

    d = SimpleNamespace(temperature=0.7, top_p=None, max_tokens=4096)
    with pytest.raises(ReglagesReflexionIncompatibles, match="400"):
        resolve_inference({"thinking_level": "high", "thinking_budget": 1024}, None, d)


def test_niveau_seul_passe():
    from types import SimpleNamespace

    from llm_gateway.core.inference import resolve_inference

    d = SimpleNamespace(temperature=0.7, top_p=None, max_tokens=4096)
    p = resolve_inference({"thinking_level": "high"}, None, d)
    assert p.thinking_level == "high" and p.thinking_budget is None


# ── Silent model substitution on the Google side ─────────────────────────────


def _adapter():
    a = GoogleAdapter()
    a._instance_name = "google_test"
    return a


def test_modele_identique_passe():
    _adapter()._refuser_substitution_de_modele("gemini-3.5-flash", {"modelVersion": "gemini-3.5-flash"})


def test_identifiant_versionne_est_le_meme_modele():
    """The API commonly answers `-001`: refusing on that would break working instances."""
    _adapter()._refuser_substitution_de_modele(
        "gemini-3.5-flash", {"modelVersion": "gemini-3.5-flash-001"}
    )


def test_alias_retire_servi_par_son_successeur_refuse():
    """The real case: `-preview` shut down on 2026-05-25, served by the model without a suffix."""
    import pytest

    from llm_gateway.adapters.base import ProviderClientError

    with pytest.raises(ProviderClientError, match="model substitution"):
        _adapter()._refuser_substitution_de_modele(
            "gemini-3.1-flash-lite-preview", {"modelVersion": "gemini-3.1-flash-lite"}
        )


def test_reponse_sans_model_version_passe():
    _adapter()._refuser_substitution_de_modele("gemini-3.5-flash", {})
    _adapter()._refuser_substitution_de_modele("gemini-3.5-flash", {"modelVersion": ""})


# ── The thinking LEVEL: what the tests did not cover ─────────────────────────
#
# This whole block was missing. The level was built by the adapter, never by the test
# helper, hence never checked. The forms below were tried against the Google API on
# 2026-09-11: only `thinkingConfig.thinkingLevel` is accepted.


def test_le_niveau_va_DANS_thinkingConfig():
    """`generationConfig.thinking_level` → 400 "Unknown name". This is the fixed defect."""
    cfg = _config(_req(thinking_level="low"))
    assert cfg["thinkingConfig"]["thinkingLevel"] == "low"
    assert "thinking_level" not in cfg, "the API refuses this field at generationConfig level"
    assert "thinkingLevel" not in cfg, "it also refuses it in camelCase at this level"


def test_le_niveau_ne_rapatrie_pas_les_pensees():
    assert _config(_req(thinking_level="high"))["thinkingConfig"]["includeThoughts"] is False


def test_un_niveau_non_minimal_reserve_un_forfait_de_sortie():
    """Thinking is drawn from the output budget: without a reserve, the response is truncated."""
    cfg = _config(_req(thinking_level="high"))
    assert cfg["maxOutputTokens"] == 4096 + RESERVE_REFLEXION


def test_le_niveau_minimal_ne_reserve_rien():
    """`minimal` hardly thinks: reserving budget for nothing would waste it."""
    cfg = _config(_req(thinking_level="minimal"))
    assert cfg["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert cfg["maxOutputTokens"] == 4096


def test_sans_niveau_ni_budget_aucun_thinkingConfig():
    """Behaviour from before 2026-09-10: the provider keeps its default."""
    assert "thinkingConfig" not in _config(_req())


def test_le_budget_reste_un_budget_quand_il_est_seul():
    """The level fix must not move the budget, which already worked."""
    cfg = _config(_req(thinking_budget=1024))
    assert cfg["thinkingConfig"] == {"thinkingBudget": 1024, "includeThoughts": False}
    assert "thinkingLevel" not in cfg["thinkingConfig"]


def test_niveau_et_budget_ne_coexistent_jamais_dans_le_payload():
    """The API: "You can only set only one of thinking budget and thinking level" (400).

    The `resolve_inference` cascade already refuses them together; this is the second barrier,
    in case a caller builds the request by hand.
    """
    cfg = _config(_req(thinking_level="low", thinking_budget=512))
    tc = cfg["thinkingConfig"]
    assert ("thinkingLevel" in tc) != ("thinkingBudget" in tc), tc
