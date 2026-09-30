"""Exact-prompt replay: the control arm receives the treated arm's responses as long as its
prompts are the same, without a provider call.

On 2026-09-25, the two arms of the a09 V2 A/B diverged from the first evening: gemini-3.5
answered differently to identical prompts. These tests check that a prompt already served in
a space is served again identically, and that anything other than that prompt goes to the provider.
"""
from __future__ import annotations

import json

import httpx
import pytest

import llm_gateway.worker.task_worker as tw
from llm_gateway.api.app import create_app
from llm_gateway.config import Settings
from llm_gateway.core.models import LLMRequest, Task, TaskStatus
from llm_gateway.core.rejeu_ab import cle_rejeu, espace_valide
from llm_gateway.testing import FakeAdapter

ESPACE = "exp_mem_presse_a13_test"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        telemetry={"workdir": str(tmp_path), "exchanges_enabled": True},
        learned_limits="none",
        rejeu={"dir": str(tmp_path / "rejeu_ab")},
    )


class _Dispatch:
    def __init__(self):
        self.appels = 0

    def delay(self, *_a):
        self.appels += 1

    def apply_async(self, **_k):
        self.appels += 1


@pytest.fixture
def dispatch(monkeypatch) -> _Dispatch:
    stub = _Dispatch()
    monkeypatch.setattr(tw, "process_batch_task", stub)
    return stub


@pytest.fixture
def fake_adapter(monkeypatch) -> FakeAdapter:
    fake = FakeAdapter()
    monkeypatch.setattr(tw, "get_adapter", lambda name: fake)
    return fake


@pytest.fixture
async def client(memory_deps, dispatch):
    app = create_app(memory_deps.settings, deps=memory_deps)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c


def _requete(aid: str, texte: str, origine: str, espace: str | None = ESPACE, **parametres) -> dict:
    return {
        "category": "echo",
        "agents": [{"agent_id": aid, "texte": texte}],
        "parameters": parametres,
        "instances_admises": ["fake"],
        "origine": origine,
        **({"espace_rejeu": espace} if espace else {}),
    }


def _bras_traite(memory_runtime, *couples):
    """The treated arm: a merged batch, served by the provider, recorded task by task."""
    tasks = [Task(request=LLMRequest(**_requete(a, t, f"{ESPACE}_treated"))) for a, t in couples]
    for t in tasks:
        memory_runtime.store.save_sync(t)
    tw._execute_batch(memory_runtime, tasks, "batch_traite", "fake")
    return tasks


async def test_le_temoin_recoit_la_reponse_du_traite_sans_appel(
    client, memory_runtime, fake_adapter, dispatch, memory_deps, tmp_path
):
    _bras_traite(memory_runtime, ("a1", "morning"), ("a2", "evening"))
    assert len(fake_adapter.calls) == 1
    assert memory_runtime.metrics.get("rejeu_ab_consigne_total:echo") == 2, "one per task, not per batch"

    r = await client.post("/tasks", json=_requete("a1", "morning", f"{ESPACE}_control"))
    assert r.status_code == 202
    corps = r.json()
    assert corps["status"] == TaskStatus.SUCCESS, "served before even entering the queue"
    assert dispatch.appels == 0, "no batch, no slot, no call"
    assert len(fake_adapter.calls) == 1

    rendu = (await client.get(f"/tasks/{corps['task_id']}/wait", params={"timeout": 1})).json()
    assert rendu["rejeu"] == ESPACE
    assert rendu["provider_used"] == "fake", "the original provider: the client refuses substitutions"
    assert [a["agent_id"] for a in rendu["result"]] == ["a1"]
    assert memory_deps.metrics.get("rejeu_ab_servi_total:echo") == 1

    journal = (tmp_path / "llm_exchanges.jsonl").read_text(encoding="utf-8")
    assert '"provider": "rejeu_ab:fake"' in journal, "the log also records the replayed response"
    assert f'"origine": "{ESPACE}_control"' in journal


async def test_un_prompt_qui_change_part_au_fournisseur(client, memory_runtime, fake_adapter, dispatch, memory_deps):
    _bras_traite(memory_runtime, ("a1", "morning"))
    r = await client.post("/tasks", json=_requete("a1", "morning — I read in the newspaper", f"{ESPACE}_control"))
    assert r.json()["status"] == TaskStatus.PENDING
    assert dispatch.appels == 1
    assert memory_deps.metrics.get("rejeu_ab_absent_total:echo") == 1


async def test_prefixe_commun_refuse_un_prompt_absent_sans_appel(client, dispatch):
    requete = {**_requete("julie", "extra reflection", "temoin"),
               "rejeu_obligatoire": True}
    r = await client.post("/tasks", json=requete)
    assert r.status_code == 409
    assert r.json()["detail"]["erreur"] == "rejeu_obligatoire_absent"
    assert dispatch.appels == 0


async def test_prefixe_commun_sert_une_reponse_identique(client, memory_runtime, fake_adapter, dispatch):
    _bras_traite(memory_runtime, ("julie", "morning reflection"))
    requete = {**_requete("julie", "morning reflection", "temoin"),
               "rejeu_obligatoire": True}
    r = await client.post("/tasks", json=requete)
    assert r.status_code == 202
    assert r.json()["status"] == TaskStatus.SUCCESS
    assert dispatch.appels == 0


async def test_un_parametre_qui_change_part_au_fournisseur(client, memory_runtime, fake_adapter, dispatch):
    _bras_traite(memory_runtime, ("a1", "morning"))
    r = await client.post("/tasks", json=_requete("a1", "morning", f"{ESPACE}_control", temperature=0.7))
    assert r.json()["status"] == TaskStatus.PENDING


async def test_un_autre_espace_ne_voit_rien(client, memory_runtime, fake_adapter, dispatch):
    _bras_traite(memory_runtime, ("a1", "morning"))
    r = await client.post("/tasks", json=_requete("a1", "morning", "autre_control", espace="autre_exp"))
    assert r.json()["status"] == TaskStatus.PENDING


async def test_sans_espace_rien_ne_change(client, memory_runtime, fake_adapter, dispatch, memory_deps):
    _bras_traite(memory_runtime, ("a1", "morning"))
    r = await client.post("/tasks", json=_requete("a1", "morning", "x", espace=None))
    assert r.json()["status"] == TaskStatus.PENDING
    assert memory_deps.metrics.get("rejeu_ab_absent_total:echo") == 0, "no lookup took place"


async def test_un_espace_hors_charset_est_refuse(client, dispatch):
    assert espace_valide("../../etc") is None
    assert espace_valide("exp.a_b-1") == "exp.a_b-1"
    r = await client.post("/tasks", json=_requete("a1", "morning", "x", espace="../../etc"))
    assert r.json()["status"] == TaskStatus.PENDING


def test_l_origine_n_entre_pas_dans_la_cle():
    """It names the arm, which differs by construction: were it included, nothing would replay."""
    traite = LLMRequest(**_requete("a1", "morning", f"{ESPACE}_treated"))
    temoin = LLMRequest(**_requete("a1", "morning", f"{ESPACE}_control"))
    messages = [{"role": "user", "content": "morning"}]
    assert cle_rejeu(traite, messages) == cle_rejeu(temoin, messages)
    autre = LLMRequest(**{**_requete("a1", "morning", "x"), "instances_admises": ["autre"]})
    assert cle_rejeu(autre, messages) != cle_rejeu(traite, messages), "the model, however, counts"


def test_une_reponse_incomplete_n_est_pas_consignee(memory_runtime, monkeypatch, tmp_path):
    """Replaying a response where an agent is missing would make it permanent."""
    fake = FakeAdapter(responder=lambda aid: {"agent_id": "unknown", "summary": "s"})
    monkeypatch.setattr(tw, "get_adapter", lambda name: fake)
    _bras_traite(memory_runtime, ("a1", "morning"))
    assert not list((tmp_path / "rejeu_ab").rglob("*.json"))


def test_la_premiere_reponse_consignee_fait_foi(memory_runtime, monkeypatch, tmp_path):
    reponses = iter(["first", "second"])
    fake = FakeAdapter(responder=lambda aid: {"agent_id": aid, "summary": next(reponses)})
    monkeypatch.setattr(tw, "get_adapter", lambda name: fake)
    _bras_traite(memory_runtime, ("a1", "morning"))
    _bras_traite(memory_runtime, ("a1", "morning"))
    (fichier,) = (tmp_path / "rejeu_ab" / ESPACE).glob("*.json")
    assert json.loads(fichier.read_text())["agents"][0]["summary"] == "first"
