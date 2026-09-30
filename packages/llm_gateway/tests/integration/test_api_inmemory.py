"""The API composed on in-memory ports: validation by the category, 422, 202, /health."""
from __future__ import annotations

import httpx
import pytest

from llm_gateway.api.app import create_app
from llm_gateway.core.models import _FALLBACK_PRIORITY_SCORE, TaskStatus


class _DispatchStub:
    """Replaces the Celery task: records the dispatches, talks to no broker."""

    def __init__(self) -> None:
        self.delay_calls: list[tuple] = []
        self.apply_async_calls: list[dict] = []

    def delay(self, *args):
        self.delay_calls.append(args)

    def apply_async(self, **kwargs):
        self.apply_async_calls.append(kwargs)


@pytest.fixture
def dispatch(monkeypatch) -> _DispatchStub:
    import llm_gateway.worker.task_worker as tw
    stub = _DispatchStub()
    monkeypatch.setattr(tw, "process_batch_task", stub)
    return stub


@pytest.fixture
async def client(memory_deps, dispatch):
    app = create_app(memory_deps.settings, deps=memory_deps)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
        yield c


async def test_health_repond_avec_les_providers(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "providers" in r.json()


async def test_categorie_inconnue_est_refusee_en_422(client):
    r = await client.post("/tasks", json={"category": "undeclared", "agents": [{"agent_id": "a1"}]})
    assert r.status_code == 422
    assert "undeclared" in r.json()["detail"] and "echo" in r.json()["detail"]


async def test_item_sans_agent_id_est_refuse_en_422(client):
    r = await client.post("/tasks", json={"category": "echo", "agents": [{"texte": "x"}]})
    assert r.status_code == 422


async def test_tache_acceptee_stockee_et_dispatch_planifie(client, memory_deps, dispatch):
    r = await client.post("/tasks", json={"category": "echo", "agents": [{"agent_id": "a1", "texte": "hello"}]})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == TaskStatus.PENDING
    task = await memory_deps.store.get(body["task_id"])
    assert task is not None
    assert task.request.agents[0].model_dump()["texte"] == "hello", "free fields travel along"
    assert task.priority_score == _FALLBACK_PRIORITY_SCORE, "echo defines no priority"
    assert dispatch.delay_calls or dispatch.apply_async_calls, "a dispatch was scheduled"

    r2 = await client.get(f"/tasks/{body['task_id']}")
    assert r2.status_code == 200 and r2.json()["status"] == "pending"


async def test_tache_inconnue_404(client):
    r = await client.get("/tasks/nope")
    assert r.status_code == 404


async def test_config_expose_les_reglages_sans_secrets(client, memory_deps):
    r = await client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "batching" in body and "telemetry" in body
    for key in memory_deps.settings.provider_keys.values():
        assert key.get_secret_value() not in r.text
    r2 = await client.get("/config/providers")
    assert r2.status_code == 200 and set(r2.json()) == {"declared", "active"}
    assert all("api_key" not in cfg for cfg in r2.json()["active"].values())


def test_les_familles_declarees_par_un_bundle_sont_exposees(memory_deps):
    """The collector renders the worker counters under the names the bundle declares."""
    import dataclasses

    from llm_gateway.api.metrics import WorkerMetricsCollector
    from llm_gateway.ports.category import MetricFamilySpec
    from llm_gateway.testing import build_registry, echo_bundle

    bundle = dataclasses.replace(echo_bundle(), metric_families=(
        MetricFamilySpec("echo_par_provider_total", "test", "echo_by_provider", ("provider",)),
        MetricFamilySpec("echo_total", "test", "echo_total"),
    ))
    memory_deps.registry = build_registry(bundle)
    memory_deps.metrics.incr("echo_by_provider:fake", amount=3)
    memory_deps.metrics.incr("echo_total", amount=7)
    # prometheus_client strips the _total suffix from a counter family's name;
    # the samples, however, carry it.
    families = {f.name: f for f in WorkerMetricsCollector(memory_deps).collect()}
    par_provider = families["echo_par_provider"].samples[0]
    assert par_provider.name == "echo_par_provider_total"
    assert par_provider.labels == {"provider": "fake"} and par_provider.value == 3
    assert families["echo"].samples[0].value == 7
