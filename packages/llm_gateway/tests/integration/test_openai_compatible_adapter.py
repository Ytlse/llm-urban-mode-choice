"""The OpenAI-compatible translator and its four settings (OpenAI, Groq, Cerebras, Mistral).

No HTTP call: the adapter's httpx client is replaced by a double. What these tests
pin down: the shape of the body sent depending on `structured_output` and `schema_in_system`,
token reading, truncation, and the classification of network errors as retryable errors.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import SecretStr

from llm_gateway.adapters.base import ProviderClientError, ProviderServerError, get_adapter
from llm_gateway.adapters.cerebras_adapter import CerebrasAdapter
from llm_gateway.adapters.groq_adapter import GroqAdapter
from llm_gateway.adapters.mistral_adapter import MistralAdapter
from llm_gateway.adapters.openai_adapter import OpenAIAdapter
from llm_gateway.adapters.openai_compatible import OpenAICompatibleAdapter
from llm_gateway.core.models import InternalMessage, InternalRequest

SCHEMA = {"type": "object", "properties": {"agents": {"type": "array"}}}
CONTENT = json.dumps({"agents": [{"agent_id": "a1", "summary": "ok"}]})


def _response(content: str = CONTENT, finish: str = "stop", status: int = 200, usage=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.headers = {}
    resp.text = content
    resp.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": usage if usage is not None else {"prompt_tokens": 120, "completion_tokens": 30},
    }
    return resp


def _adapter(cls, monkeypatch, *, instance_cfg=None):
    inst = cls()
    inst._instance_name = "inst"
    monkeypatch.setattr(inst, "_get_api_key", lambda: SecretStr("key"))
    monkeypatch.setattr(inst, "_get_base_url", lambda: "https://example.invalid/v1")
    monkeypatch.setattr(inst, "_resolve_model", lambda _r: "m-test")
    monkeypatch.setattr(inst, "_instance_config", lambda: instance_cfg)
    return inst


def _wire(adapter, monkeypatch, response=None, exc=None):
    client = MagicMock()
    if exc is not None:
        client.post.side_effect = exc
    else:
        client.post.return_value = response
    monkeypatch.setattr(adapter, "_http", lambda: client)
    return client


def _request(top_p=None):
    return InternalRequest(
        provider="inst",
        messages=[InternalMessage(role="system", content="Instruction."), InternalMessage(role="user", content="agent_id=a1")],
        response_schema=SCHEMA, temperature=0.4, top_p=top_p, max_tokens=512,
    )


# ── Body shape depending on the subclass ─────────────────────────────────────

@pytest.mark.parametrize("cls, mode, injected", [
    (OpenAIAdapter, "json_schema", False),
    (GroqAdapter, "json_object", False),
    (CerebrasAdapter, "json_object", True),
    (MistralAdapter, "json_object", True),
])
def test_chaque_dialecte_est_un_reglage(cls, mode, injected, monkeypatch):
    a = _adapter(cls, monkeypatch)
    payload = a.build_payload(_request())
    assert payload["model"] == "m-test" and payload["temperature"] == 0.4 and payload["max_tokens"] == 512
    assert "top_p" not in payload, "top_p absent when nobody sets it"
    if mode == "json_schema":
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    else:
        assert payload["response_format"] == {"type": "json_object"}
    system = payload["messages"][0]["content"]
    assert ("schéma" in system) is injected, "the schema is copied into system only if the subclass asks for it"


def test_top_p_envoye_quand_defini(monkeypatch):
    a = _adapter(GroqAdapter, monkeypatch)
    assert a.build_payload(_request(top_p=0.9))["top_p"] == 0.9


def test_sans_message_system_le_schema_en_cree_un(monkeypatch):
    a = _adapter(MistralAdapter, monkeypatch)
    req = InternalRequest(provider="inst", messages=[InternalMessage(role="user", content="x")], response_schema=SCHEMA)
    messages = a.build_payload(req)["messages"]
    assert messages[0]["role"] == "system" and "schéma" in messages[0]["content"]


def test_l_instance_surcharge_les_defauts_de_la_classe(monkeypatch):
    """`adapter: openai_compatible` + `structured_output` / `schema_in_system` in the file."""
    cfg = MagicMock(structured_output="none", schema_in_system=True)
    a = _adapter(OpenAICompatibleAdapter, monkeypatch, instance_cfg=cfg)
    payload = a.build_payload(_request())
    assert "response_format" not in payload
    assert "schéma" in payload["messages"][0]["content"]


def test_en_tete_d_authentification(monkeypatch):
    a = _adapter(OpenAIAdapter, monkeypatch)
    assert a._headers()["Authorization"] == "Bearer key"


# ── Full call ────────────────────────────────────────────────────────────────

def test_appel_rend_sortie_et_tokens(monkeypatch):
    a = _adapter(GroqAdapter, monkeypatch)
    client = _wire(a, monkeypatch, _response())
    output, tokens_in, tokens_out = a.call(_request())
    assert output.agents[0].agent_id == "a1" and (tokens_in, tokens_out) == (120, 30)
    url = client.post.call_args.args[0]
    assert url == "https://example.invalid/v1/chat/completions"


def test_troncature_finish_reason_length_est_reessayable(monkeypatch):
    a = _adapter(OpenAIAdapter, monkeypatch)
    _wire(a, monkeypatch, _response(content='{"agents": [', finish="length"))
    with pytest.raises(ProviderServerError) as exc:
        a.call(_request())
    assert exc.value.error_type == "max_tokens_truncation"


def test_4xx_reste_une_erreur_client(monkeypatch):
    a = _adapter(GroqAdapter, monkeypatch)
    _wire(a, monkeypatch, _response(content='{"error": {"message": "bad", "type": "invalid_request_error"}}', status=400))
    with pytest.raises(ProviderClientError):
        a.call(_request())


@pytest.mark.parametrize("exc, error_type, status", [
    (httpx.ReadTimeout("slow"), "network_timeout", 504),
    (httpx.ConnectError("refused"), "network_connect", 503),
    (httpx.RemoteProtocolError("cut"), "network_protocol", 502),
])
def test_erreurs_reseau_deviennent_des_erreurs_serveur_reessayables(exc, error_type, status, monkeypatch):
    a = _adapter(MistralAdapter, monkeypatch)
    _wire(a, monkeypatch, exc=exc)
    with pytest.raises(ProviderServerError) as e:
        a.call(_request())
    assert e.value.error_type == error_type and e.value.status_code == status
    assert e.value.provider == "inst", "the cooldown is indexed on the instance name"


# ── Registry and discovery ───────────────────────────────────────────────────

def test_les_cinq_noms_historiques_restent_enregistres(monkeypatch):
    import llm_gateway.adapters.base as base
    base._load_adapters()
    for name in ("openai", "groq", "cerebras", "mistral", "google", "openai_compatible"):
        assert name in base._REGISTRY, name


def test_un_adapter_externe_arrive_par_entry_point(monkeypatch):
    import llm_gateway.adapters.base as base

    class ExternAdapter(OpenAICompatibleAdapter):
        provider_name = ""   # will be named by the entry point

    ep = MagicMock()
    ep.name = "externe"
    ep.value = "pkg:ExternAdapter"
    ep.load = lambda: ExternAdapter
    monkeypatch.setattr(base, "entry_points", lambda group: [ep] if group == base.ADAPTERS_ENTRY_POINT else [], raising=False)
    import importlib.metadata as md
    monkeypatch.setattr(md, "entry_points", lambda group=None, **kw: [ep] if group == base.ADAPTERS_ENTRY_POINT else [])
    base._REGISTRY.pop("externe", None)
    base._load_adapters()
    assert base._REGISTRY["externe"] is ExternAdapter and ExternAdapter.provider_name == "externe"
    base._REGISTRY.pop("externe", None)


def test_get_adapter_resout_par_champ_adapter(monkeypatch):
    import llm_gateway.adapters.base as base
    cfg = MagicMock(adapter="openai_compatible")
    monkeypatch.setattr(base, "get_settings", lambda: MagicMock(providers={"mon_ollama": cfg}))
    base._INSTANCES.pop("mon_ollama", None)
    inst = get_adapter("mon_ollama")
    assert isinstance(inst, OpenAICompatibleAdapter) and inst._instance_name == "mon_ollama"
    base._INSTANCES.pop("mon_ollama", None)
