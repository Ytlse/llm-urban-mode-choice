"""OpenAI-compatible adapter — checks on the provider response."""



# ── Silent model substitution (observed on 2026-09-10 on LM Studio) ──


def _adapter_pour(monkeypatch, instance="lmstudio_test", modele="qwen/qwen3.8-27b"):
    from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter

    a = OpenAICompatibleAdapter()
    a._instance_name = instance
    monkeypatch.setattr(a, "_resolve_model", lambda request: modele)
    return a


def test_refuse_un_modele_servi_different(monkeypatch):
    """An unknown identifier produces no error on the LM Studio side: it serves another model."""
    import pytest

    from llm_gateway.adapters.base import ProviderClientError

    a = _adapter_pour(monkeypatch)
    with pytest.raises(ProviderClientError, match="model substitution"):
        a._refuser_substitution_de_modele(object(), {"model": "qwen3-vl-8b-instruct-mlx"})


def test_accepte_le_modele_demande(monkeypatch):
    a = _adapter_pour(monkeypatch)
    a._refuser_substitution_de_modele(object(), {"model": "qwen/qwen3.8-27b"})


def test_reponse_sans_champ_modele_passe(monkeypatch):
    """Refusing on a missing field would block compliant providers."""
    a = _adapter_pour(monkeypatch)
    a._refuser_substitution_de_modele(object(), {})
    a._refuser_substitution_de_modele(object(), {"model": ""})


def test_message_nomme_les_deux_modeles(monkeypatch):
    import pytest

    from llm_gateway.adapters.base import ProviderClientError

    a = _adapter_pour(monkeypatch)
    with pytest.raises(ProviderClientError) as exc:
        a._refuser_substitution_de_modele(object(), {"model": "other-model"})
    assert "qwen/qwen3.8-27b" in str(exc.value) and "other-model" in str(exc.value)
    assert "default_model" in str(exc.value), "the message must say where to fix it"
