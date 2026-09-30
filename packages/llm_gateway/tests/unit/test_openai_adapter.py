"""Native OpenAI adapter — parameter conversion test for reasoning models."""
from llm_gateway.adapters.openai_adapter import OpenAIAdapter
from llm_gateway.core.models import InternalMessage, InternalRequest


def _requete(modele: str, temperature: float = 0.0, max_tokens: int = 4096) -> InternalRequest:
    return InternalRequest(
        provider="openai_key1",
        model=modele,
        messages=[InternalMessage(role="user", content="hello")],
        response_schema={"type": "object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_modele_standard_garde_max_tokens_et_temperature(monkeypatch):
    a = OpenAIAdapter()
    monkeypatch.setattr(a, "_resolve_model", lambda r: "gpt-4o-mini")
    payload = a.build_payload(_requete("gpt-4o-mini", temperature=0.0, max_tokens=1000))
    assert payload["max_tokens"] == 1000
    assert "max_completion_tokens" not in payload
    assert payload["temperature"] == 0.0


def test_modele_raisonnement_utilise_max_completion_tokens_et_omet_temp_nulle(monkeypatch):
    a = OpenAIAdapter()
    for nom_modele in ("gpt-5.6-luna", "o1", "o3-mini", "gpt-5-mini"):
        monkeypatch.setattr(a, "_resolve_model", lambda r, m=nom_modele: m)
        payload = a.build_payload(_requete(nom_modele, temperature=0.0, max_tokens=2048))
        assert payload["max_completion_tokens"] == 2048
        assert "max_tokens" not in payload
        assert "temperature" not in payload


def test_modele_raisonnement_garde_temperature_si_egale_a_un(monkeypatch):
    a = OpenAIAdapter()
    monkeypatch.setattr(a, "_resolve_model", lambda r: "gpt-5.6-luna")
    payload = a.build_payload(_requete("gpt-5.6-luna", temperature=1.0, max_tokens=2048))
    assert payload["max_completion_tokens"] == 2048
    assert payload["temperature"] == 1.0
