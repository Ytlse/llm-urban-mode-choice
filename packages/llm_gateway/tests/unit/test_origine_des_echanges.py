"""The origin of a request signs the exchange in the worker log (2026-09-24).

The exchange log is written by the WORKER, shared by all clients: an arm of 20
agents read in its `llm_exchanges.jsonl` 97 exchanges of a population of 1,000 agents served
while it was running. Each client now signs its requests; a batch never mixes
two of them.
"""

import asyncio
import json

import httpx

from llm_gateway.core.batching import compute_batch_key
from llm_gateway.core.models import LLMRequest
from llm_gateway.sdk import LLMGatewayClient
from llm_gateway.telemetry.exchanges import ExchangeRecord


def _requete(**extra) -> LLMRequest:
    return LLMRequest(category="itinary_multi_agent", agents=[{"agent_id": "a"}], **extra)


def test_deux_origines_ne_partagent_pas_un_lot():
    assert compute_batch_key(_requete(origine="run_a")) != compute_batch_key(_requete(origine="run_b"))
    assert compute_batch_key(_requete(origine="run_a")) == compute_batch_key(_requete(origine="run_a"))


def test_l_enregistrement_porte_son_origine():
    rec = ExchangeRecord(
        task_id="b", provider="p", category="c", tokens_in=1, tokens_out=1,
        messages=[], response=[], origine="2026-09-24_11_14",
    )
    assert rec.to_json_dict()["origine"] == "2026-09-24_11_14"
    sans = ExchangeRecord(task_id="b", provider="p", category="c", tokens_in=1, tokens_out=1,
                          messages=[], response=[])
    assert sans.to_json_dict()["origine"] is None


def _soumis(payload: dict, **client_kw) -> dict:
    vus: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks" and request.method == "POST":
            vus.append(json.loads(request.content))
            return httpx.Response(202, json={"task_id": "t-1", "status": "pending"})
        if request.url.path == "/tasks/t-1/wait":
            return httpx.Response(200, json={
                "task_id": "t-1", "status": "success",
                "created_at": "2026-09-24T10:00:00Z", "updated_at": "2026-09-24T10:00:01Z",
                "result": [{"agent_id": "a", "chosen_index": 0, "mode": "car", "reason": "r"}],
                "provider_used": "p",
            })
        return httpx.Response(404)

    client = LLMGatewayClient(transport=httpx.MockTransport(handler), **client_kw)

    async def run():
        try:
            await client.execute(payload)
        finally:
            await client.aclose()

    asyncio.run(run())
    return vus[0]


def test_le_client_signe_chaque_appel():
    payload = {"category": "stm_reflection", "agents": [{"agent_id": "a"}]}
    assert _soumis(payload, origine="2026-09-24_11_14")["origine"] == "2026-09-24_11_14"


def test_un_client_sans_origine_ne_pose_rien():
    payload = {"category": "stm_reflection", "agents": [{"agent_id": "a"}]}
    assert "origine" not in _soumis(payload)


def test_rejeu_strict_avant_exposition_seulement():
    payload = {"category": "stm_reflection", "agents": [{"agent_id": "a"}]}
    commun = {"espace_rejeu": "ab", "rejeu_strict_avant_ts": 100}
    avant = _soumis(payload, **commun, horloge_simulee=lambda: 99)
    assert avant["rejeu_obligatoire"] is True and avant["espace_rejeu"] == "ab"
    apres = _soumis(payload, **commun, horloge_simulee=lambda: 100)
    assert "rejeu_obligatoire" not in apres


def test_apres_la_borne_le_temoin_ne_lit_plus_le_magasin():
    """2026-09-28: a response of the treated arm, written in a batch next to a reader of the
    article, must not be served again to the control after the article."""
    payload = {"category": "stm_reflection", "agents": [{"agent_id": "a"}]}
    apres = _soumis(payload, espace_rejeu="ab", rejeu_strict_avant_ts=100,
                    horloge_simulee=lambda: 100)
    assert "espace_rejeu" not in apres


def test_le_traite_garde_son_magasin_apres_l_evenement():
    """Without a strict boundary (treated arm), the space stays set: a resumed day re-reads its
    own responses."""
    payload = {"category": "stm_reflection", "agents": [{"agent_id": "a"}]}
    assert _soumis(payload, espace_rejeu="ab")["espace_rejeu"] == "ab"


def test_rejeu_strict_signale_une_indisponibilite_sans_repli():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tasks"
        return httpx.Response(503, json={"detail": "Magasin de rejeu indisponible"})

    client = LLMGatewayClient(
        transport=httpx.MockTransport(handler), espace_rejeu="ab",
        rejeu_strict_avant_ts=100, horloge_simulee=lambda: 99,
    )

    async def run():
        try:
            resultat = await client.execute({"category": "stm_reflection", "agents": [{"agent_id": "a"}]})
            assert not resultat.ok
            assert "HTTP 503" in (client.strict_replay_failure or "")
        finally:
            await client.aclose()

    asyncio.run(run())
