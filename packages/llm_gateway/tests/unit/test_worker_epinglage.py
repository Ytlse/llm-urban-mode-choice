"""A pinned decision-maker is NEVER served by another model.

The gateway can switch to another provider when the one it chose fails
(parse error, 4xx, prompt too large): that is the right answer for a GAMA run, which wants a
decision. It is the wrong one for an experiment, which measures ONE model: receiving the
answer of another invalidates the measurement.

Reference outage (2026-09-07): `cerebras_gpt-oss-120b` answers HTTP 402 "payment
required", the gateway replays the batch without pinning, `mistral` answers, the client
refuses the substitution — 8 requests, 8 refusals, 0 decision archived, and a WARNING drowned
in the worker log as the only signal.

No Celery/Redis/LLM call: the helpers are called directly.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from llm_gateway.adapters.base import ProviderClientError
from llm_gateway.core.models import TaskStatus
from llm_gateway.worker.task_worker import _credits_epuises, _switch_provider_or_fail


class _Limiter:
    def __init__(self, disabled=()):
        self.disabled = set(disabled)
        self.cooldowns: dict[str, int] = {}
        self.disables: dict[str, int] = {}

    def cooldown(self, provider, seconds):
        self.cooldowns[provider] = seconds

    def disable(self, provider, seconds):
        self.disables[provider] = seconds
        self.disabled.add(provider)

    def is_disabled(self, provider):
        return provider in self.disabled


class _Store:
    def __init__(self):
        self.sauvees = []

    def save_sync(self, task):
        self.sauvees.append(task)

    def publish_done_sync(self, task):
        pass


class _Queue:
    def __init__(self):
        self.requeues = []

    def requeue(self, batch_key, tasks):
        self.requeues.append((batch_key, list(tasks)))


class _Metrics:
    def __init__(self):
        self.compteurs: dict[str, int] = {}

    def incr(self, cle, *_a, **_k):
        self.compteurs[cle] = self.compteurs.get(cle, 0) + 1


class _CeleryTask:
    """The bare minimum of the Celery API used by the helpers."""

    max_retries = 10

    def __init__(self, retries=0):
        self.request = SimpleNamespace(retries=retries)
        self.retries_demandes = []

    def retry(self, **kwargs):
        self.retries_demandes.append(kwargs)
        return RuntimeError("retry requested")  # raised by the caller


def _runtime(providers=("mistral", "cerebras_gpt-oss-120b"), disabled=()):
    cfgs = {n: SimpleNamespace(disable_timeout=180) for n in providers}
    return SimpleNamespace(
        settings=SimpleNamespace(
            providers=cfgs,
            resilience=SimpleNamespace(provider_switch_cooldown_seconds=30),
        ),
        limiter=_Limiter(disabled),
        store=_Store(),
        queue=_Queue(),
        metrics=_Metrics(),
    )


def _taches(n=2):
    return [
        SimpleNamespace(task_id=f"t{i}", status=TaskStatus.PENDING, error=None,
                        updated_at=datetime.now(UTC))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _switch_provider_or_fail — the switch stops at pinning
# ---------------------------------------------------------------------------

class TestBasculeRefuseeSiEpingle:
    def test_epingle_echoue_sans_rejouer_le_lot(self):
        rt, celery, tasks = _runtime(), _CeleryTask(), _taches()
        exc = ProviderClientError("cerebras_gpt-oss-120b", 400, "boom")

        _switch_provider_or_fail(
            celery, rt, tasks, "bk", exc, "cerebras_gpt-oss-120b",
            reason="Erreur 4xx non récupérable",
            min_tpm_required=None, min_output_required=None,
            force_provider="cerebras_gpt-oss-120b",
        )

        assert celery.retries_demandes == [], "no replay: it would be a substitution"
        assert rt.queue.requeues == [], "the batch is not requeued for another model"
        assert all(t.status == TaskStatus.FAILED for t in tasks)
        assert all("aucune bascule" in t.error for t in tasks), tasks[0].error
        assert "cerebras_gpt-oss-120b" in tasks[0].error, "the reason names the pinned instance"
        assert rt.metrics.compteurs.get("alarme:bascule_refusee") == 1

    def test_sans_epinglage_la_bascule_reste_le_comportement(self):
        """GAMA does not measure a model: it wants a decision. Resilience is kept."""
        rt, celery, tasks = _runtime(), _CeleryTask(), _taches()
        exc = ProviderClientError("cerebras_gpt-oss-120b", 400, "boom")

        with pytest.raises(RuntimeError):
            _switch_provider_or_fail(
                celery, rt, tasks, "bk", exc, "cerebras_gpt-oss-120b",
                reason="Erreur 4xx non récupérable",
                min_tpm_required=None, min_output_required=None,
                force_provider=None,
            )

        assert celery.retries_demandes, "the batch is replayed"
        assert celery.retries_demandes[0]["kwargs"]["force_provider"] is None
        assert rt.queue.requeues, "the batch goes back to the queue for another model"
        assert all(t.status != TaskStatus.FAILED for t in tasks)


# ---------------------------------------------------------------------------
# _credits_epuises — 402
# ---------------------------------------------------------------------------

class TestCreditsEpuises:
    def _appel(self, rt, celery, tasks, force_provider):
        exc = ProviderClientError(
            "cerebras_gpt-oss-120b", 402,
            '{"message":"Payment required to access this resource."}',
        )
        return _credits_epuises(
            celery, rt, tasks, "bk", exc,
            force_provider=force_provider,
            min_tpm_required=None, min_output_required=None,
        )

    def test_desactive_l_instance_au_lieu_d_un_cooldown_court(self):
        """Credits come back after billing, not after 30 seconds."""
        rt, celery, tasks = _runtime(), _CeleryTask(), _taches()
        self._appel(rt, celery, tasks, force_provider="cerebras_gpt-oss-120b")
        assert rt.limiter.disables == {"cerebras_gpt-oss-120b": 180}
        assert rt.limiter.cooldowns == {}, "a 402 is not handled with a short cooldown"

    def test_epingle_erreur_franche_sans_substitution(self):
        rt, celery, tasks = _runtime(), _CeleryTask(), _taches()
        self._appel(rt, celery, tasks, force_provider="cerebras_gpt-oss-120b")
        assert celery.retries_demandes == []
        assert all(t.status == TaskStatus.FAILED for t in tasks)
        assert all("402" in t.error and "épinglée" in t.error for t in tasks), tasks[0].error

    def test_alarme_sur_front_montant_uniquement(self):
        """One line at the first 402; not one per batch while the instance stays down."""
        rt, celery = _runtime(), _CeleryTask()
        self._appel(rt, celery, _taches(), force_provider="cerebras_gpt-oss-120b")
        assert rt.metrics.compteurs.get("alarme:credits_epuises") == 1
        self._appel(rt, celery, _taches(), force_provider="cerebras_gpt-oss-120b")
        assert rt.metrics.compteurs.get("alarme:credits_epuises") == 1, "rising edge"

    def test_sans_epinglage_la_bascule_reste_possible(self):
        rt, celery, tasks = _runtime(), _CeleryTask(), _taches()
        with pytest.raises(RuntimeError):
            self._appel(rt, celery, tasks, force_provider=None)
        assert celery.retries_demandes, "GAMA switches to another model"
        assert rt.limiter.disables == {"cerebras_gpt-oss-120b": 180}, "the instance stays disabled"
