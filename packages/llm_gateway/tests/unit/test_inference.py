"""The inference parameter cascade: request > provider > defaults (core/inference.py)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_gateway.core.inference import InferenceParams, resolve_inference

DEFAULTS = SimpleNamespace(temperature=0.7, top_p=None, max_tokens=4096)


def test_defauts_seuls():
    p = resolve_inference({}, None, DEFAULTS)
    assert p == InferenceParams(temperature=0.7, top_p=None, max_tokens=4096)


def test_le_fournisseur_surcharge_les_defauts():
    prov = SimpleNamespace(temperature=0.2, top_p=0.9, max_tokens=None)
    p = resolve_inference({}, prov, DEFAULTS)
    assert (p.temperature, p.top_p, p.max_tokens) == (0.2, 0.9, 4096), "no max_tokens → default"


def test_la_requete_surcharge_le_fournisseur():
    prov = SimpleNamespace(temperature=0.2, top_p=0.9, max_tokens=1000)
    p = resolve_inference({"temperature": 1.0, "max_tokens": 512}, prov, DEFAULTS)
    assert (p.temperature, p.top_p, p.max_tokens) == (1.0, 0.9, 512)


def test_valeurs_de_requete_en_chaine_sont_coercees():
    p = resolve_inference({"temperature": "0.3", "max_tokens": "256", "top_p": "0.5"}, None, DEFAULTS)
    assert p == InferenceParams(temperature=0.3, top_p=0.5, max_tokens=256)


def test_valeur_illisible_ignoree_pas_d_exception():
    p = resolve_inference({"temperature": "hot", "max_tokens": None}, None, DEFAULTS)
    assert p.temperature == 0.7 and p.max_tokens == 4096


def test_top_p_reste_none_si_personne_ne_le_donne():
    assert resolve_inference({"temperature": 0.1}, SimpleNamespace(temperature=None, top_p=None, max_tokens=None), DEFAULTS).top_p is None


@pytest.mark.parametrize("params", [None, {}])
def test_parametres_absents(params):
    assert resolve_inference(params, None, DEFAULTS).max_tokens == 4096


# ── Thinking depth (2026-09-10) ──────────────────────────────────────────────


def test_reflexion_absente_reste_none():
    """`None` ≠ `0`: nothing is requested, the provider applies ITS thinking default."""
    p = resolve_inference(None, None, DEFAULTS)
    assert p.thinking_budget is None


def test_reflexion_depuis_la_requete():
    p = resolve_inference({"thinking_budget": 1024}, None, DEFAULTS)
    assert p.thinking_budget == 1024


def test_reflexion_zero_est_une_valeur_pas_une_absence():
    """0 explicitly disables thinking — it must not fall back to the default."""
    prov = SimpleNamespace(temperature=None, top_p=None, max_tokens=None, thinking_budget=512)
    p = resolve_inference({"thinking_budget": 0}, prov, DEFAULTS)
    assert p.thinking_budget == 0


def test_reflexion_surcharge_fournisseur():
    prov = SimpleNamespace(temperature=None, top_p=None, max_tokens=None, thinking_budget=256)
    assert resolve_inference(None, prov, DEFAULTS).thinking_budget == 256
    assert resolve_inference({"thinking_budget": 64}, prov, DEFAULTS).thinking_budget == 64


def test_reflexion_coercee_depuis_une_chaine():
    p = resolve_inference({"thinking_budget": "2048"}, None, DEFAULTS)
    assert p.thinking_budget == 2048


def test_reflexion_illisible_retombe_sans_exception():
    """An unreadable inference parameter never raises: it falls down one level."""
    p = resolve_inference({"thinking_budget": "deep"}, None, DEFAULTS)
    assert p.thinking_budget is None


def test_defauts_sans_la_cle_ne_font_pas_tomber():
    """This test's DEFAULTS does not declare `thinking_budget`: the cascade must hold."""
    assert not hasattr(DEFAULTS, "thinking_budget")
    assert resolve_inference({"temperature": 0.1}, None, DEFAULTS).thinking_budget is None
