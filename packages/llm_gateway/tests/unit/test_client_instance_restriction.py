"""The instance restriction is a property of the CLIENT, applied to every call.

The restriction first set `instances_admises` on the decision payload only. The two other
calls of the run (STM reflection, LTM self-reflection) went out without restriction and were thus
served in cascade — measured on 2026-09-16: decisions on gemini 3.1, memory consolidation
on mistral. For an experiment whose subject IS memory, the measured object was
produced by an undeclared model, without any line reporting it.
"""

import asyncio

import httpx

from llm_gateway.sdk import LLMGatewayClient

ADMISES = ["google_gemini31_key1", "google_gemini31_key2"]


def _transport_qui_retient(vus: list[dict]) -> httpx.MockTransport:
    """Answers the submit/wait cycle while keeping each submitted payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks" and request.method == "POST":
            vus.append(httpx.Response(200, content=request.content).json())
            return httpx.Response(202, json={"task_id": "t-1", "status": "pending"})
        if request.url.path == "/tasks/t-1/wait":
            return httpx.Response(200, json={
                "task_id": "t-1",
                "status": "success",
                "created_at": "2026-09-16T10:00:00Z",
                "updated_at": "2026-09-16T10:00:01Z",
                "result": [{"agent_id": "a", "chosen_index": 0, "mode": "car", "reason": "r"}],
                "provider_used": "google_gemini31_key1",
            })
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _payload(category: str) -> dict:
    return {
        "category": category,
        "agents": [{"agent_id": "899549", "perception": "p"}],
    }


def _soumettre(client: LLMGatewayClient, payload: dict) -> None:
    async def run():
        try:
            await client.execute(payload)
        finally:
            await client.aclose()

    asyncio.run(run())


def _soumis(payload: dict, *, admises=ADMISES) -> dict:
    vus: list[dict] = []
    client = LLMGatewayClient(transport=_transport_qui_retient(vus), instances_admises=admises)
    _soumettre(client, payload)
    assert vus, "no payload was submitted"
    return vus[0]


class TestLaRestrictionCouvreLesTroisAppels:
    def test_A1_decision(self):
        assert _soumis(_payload("itinary_multi_agent"))["instances_admises"] == ADMISES

    def test_A2_reflexion_stm(self):
        """The case measured in production: THIS is the one that went to mistral."""
        assert _soumis(_payload("stm_reflection"))["instances_admises"] == ADMISES

    def test_A3_auto_reflexion_ltm(self):
        assert _soumis(_payload("ltm_self_reflection"))["instances_admises"] == ADMISES


class TestCeQuiNeDoitPasChanger:
    def test_A4_un_payload_qui_porte_sa_restriction_n_est_pas_ecrase(self):
        """The caller that knows what it wants stays in control: the client only fills in."""
        propre = ["google_gemini35_key1"]
        soumis = _soumis({**_payload("itinary_multi_agent"), "instances_admises": propre})
        assert soumis["instances_admises"] == propre

    def test_A5_sans_restriction_aucune_cle_ajoutee(self):
        """Behaviour strictly identical to that of a client without restriction: the key does not exist."""
        assert "instances_admises" not in _soumis(_payload("itinary_multi_agent"), admises=None)

    def test_A6_liste_vide_vaut_aucune_restriction(self):
        assert "instances_admises" not in _soumis(_payload("itinary_multi_agent"), admises=[])


class TestTracabilite:
    def test_A7_la_restriction_est_annoncee_au_demarrage(self):
        """A restriction that does not show in the log cannot be checked afterwards."""
        from loguru import logger

        lignes: list[str] = []
        puits = logger.add(lambda m: lignes.append(str(m)), level="INFO")
        try:
            LLMGatewayClient(instances_admises=ADMISES)
        finally:
            logger.remove(puits)
        trace = "\n".join(lignes)
        assert "google_gemini31_key1" in trace and "google_gemini31_key2" in trace
