"""A deterministic refusal is reported once, immediately, and never replayed.

The question these tests answer: **does a configuration contradiction cost one second and
a correct message, or two minutes and a wrong message?**

Before this lot, `RestrictionInstances` — a `ValueError` — went through the wait loop, which
only catches `RuntimeError`. It left the task BEFORE `rt.queue.pop`: the batch stayed queued,
every following dispatch picked it up again and failed identically, the client timed out at 120 s
on a "Timeout expiré" that said nothing, and the circuit breaker opened at the tenth failure.
Three hours on 2026-09-16, for one line of YAML.

No Celery/Redis/LLM call: the runtime is doubled and the task called directly.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from llm_gateway.balancer.router import RestrictionInstances
from llm_gateway.core.models import TaskStatus
from llm_gateway.worker import task_worker
from llm_gateway.worker.task_worker import _vider_file, process_batch_task


class _Queue:
    """Batch queue whose `pop` is bounded — like the real one, which counts AGENTS, not tasks."""

    def __init__(self, taches, par_pop=2):
        self.taches = list(taches)
        self.par_pop = par_pop
        self.scheduled_cleared: list[str] = []
        self.requeues: list[tuple] = []
        self.pops = 0

    def clear_scheduled(self, batch_key):
        self.scheduled_cleared.append(batch_key)

    def pop(self, batch_key, max_agents):
        self.pops += 1
        n = min(self.par_pop, max_agents, len(self.taches))
        sortis, self.taches = self.taches[:n], self.taches[n:]
        return sortis

    def requeue(self, batch_key, tasks):
        self.requeues.append((batch_key, list(tasks)))
        self.taches.extend(tasks)

    def size(self, batch_key):
        return len(self.taches)


class _Store:
    def __init__(self):
        self.publiees = []

    def save_sync(self, task):
        pass

    def publish_done_sync(self, task):
        self.publiees.append(task)


class _Metrics:
    def __init__(self):
        self.compteurs: dict[str, int] = {}

    def incr(self, cle, *_a, **_k):
        self.compteurs[cle] = self.compteurs.get(cle, 0) + 1


class _Limiter:
    def __init__(self):
        self.actifs: list[str] = []
        self.rendus: list[str] = []

    def incr_active(self, p):
        self.actifs.append(p)

    def decr_active(self, p):
        pass

    def release_slot(self, p):
        self.rendus.append(p)


class _Balancer:
    """Raises what it is asked to on selection, and counts the calls."""

    def __init__(self, leve):
        self.leve = leve
        self.appels = 0

    def select_provider(self, **_k):
        self.appels += 1
        raise self.leve

    def get_status(self):
        return {}


def _taches(n=5):
    return [
        SimpleNamespace(task_id=f"t{i}", status=TaskStatus.PENDING, error=None,
                        error_kind=None, resume_at=None, updated_at=datetime.now(UTC))
        for i in range(n)
    ]


def _runtime(leve, taches, par_pop=2):
    return SimpleNamespace(
        settings=SimpleNamespace(
            providers={},
            resilience=SimpleNamespace(
                provider_wait_seconds=0.0,
                saturation_poll_seconds=0.0,
                saturation_retries=0,
                saturation_retry_seconds=1.0,
                abandon_when_busy=True,
                client_wait_seconds=120.0,
                client_wait_margin_seconds=10.0,
            ),
        ),
        balancer=_Balancer(leve),
        queue=_Queue(taches, par_pop=par_pop),
        store=_Store(),
        metrics=_Metrics(),
        limiter=_Limiter(),
    )


@pytest.fixture
def lot(monkeypatch):
    """A batch of 5 tasks whose selection is refused by a configuration contradiction."""
    exc = RestrictionInstances(
        "forced provider 'google_gemini35_key1' outside the admitted instances "
        "['google_gemini31_key1', 'google_gemini31_key2'] — contradictory constraints, "
        "none is settled silently."
    )
    taches = _taches(5)
    rt = _runtime(exc, taches, par_pop=2)
    monkeypatch.setattr(task_worker, "get_worker_runtime", lambda: rt)
    process_batch_task.run(
        "bk",
        force_provider="google_gemini35_key1",
        min_tpm_required=None,
        min_output_required=None,
        instances_admises=["google_gemini31_key1", "google_gemini31_key2"],
    )
    return rt, taches


# ── A1. The batch leaves the queue, and does not re-arm ──────────────────────────────────


def test_A1_les_taches_sortent_de_la_file_et_echouent(lot):
    rt, taches = lot
    assert rt.queue.taches == [], "the batch queue is EMPTY: nothing can re-arm on the next dispatch"
    assert all(t.status == TaskStatus.FAILED for t in taches)
    assert len(rt.store.publiees) == 5, "every task is published: the client receives its reason"


def test_A1_aucun_rejeu_nest_planifie(lot):
    rt, _ = lot
    assert rt.queue.requeues == [], "a replay could only reproduce the contradiction"
    assert rt.balancer.appels == 1, "a single selection: no wait loop on a deterministic refusal"


def test_A1_la_file_est_drainee_meme_au_dela_dun_seul_pop():
    """`pop` is bounded by a number of AGENTS: a single call would leave the batch tail behind."""
    rt = _runtime(RestrictionInstances("boum"), _taches(7), par_pop=2)
    assert len(_vider_file(rt, "bk")) == 7
    assert rt.queue.taches == []


def test_A1_aucun_slot_nest_restitue(lot):
    rt, _ = lot
    assert rt.limiter.rendus == [], "selection failed BEFORE any reservation: nothing to give back"


# ── A2. The reason names BOTH constraints ────────────────────────────────────────────────


def test_A2_le_motif_nomme_la_liste_admise_et_le_fournisseur_epingle(lot):
    _, taches = lot
    motif = taches[0].error
    assert "google_gemini31_key1" in motif and "google_gemini31_key2" in motif, motif
    assert "google_gemini35_key1" in motif, motif
    assert "admises" in motif and "épinglé" in motif, (
        "both constraints must be READABLE as such, not merely present"
    )


def test_A2_le_motif_dit_quil_ne_sera_pas_rejoue(lot):
    _, taches = lot
    assert "aucun rejeu" in taches[0].error, taches[0].error


# ── A3. The dispatch flag ────────────────────────────────────────────────────────────────


def test_A3_le_drapeau_de_dispatch_est_leve(lot):
    rt, _ = lot
    assert rt.queue.scheduled_cleared == ["bk"], (
        "without clear_scheduled, the next batch waits for the flag TTL for nothing"
    )


# ── A4. The nature of the failure ────────────────────────────────────────────────────────


def test_A4_lechec_porte_sa_propre_nature(lot):
    _, taches = lot
    assert all(t.error_kind == "restriction_instances" for t in taches)
    assert all(t.error_kind != "quota_journalier" for t in taches), (
        "it is not a quota: looking for quota is what cost the three hours"
    )


def test_A4_le_motif_nevoque_ni_saturation_ni_quota(lot):
    """The text is the client's fallback when `error_kind` is missing: it must not mislead it.

    `decideurs.py` classifies by regular expressions — `satur|indisponible|timeout|occup` to
    "gateway busy", `429|quota|rate.limit` to "exhausted". The reason of a configuration
    contradiction must fall into neither bucket.
    """
    import re

    _, taches = lot
    motif = taches[0].error
    assert not re.search(r"(satur|indisponible|timeout|occup|unavailable)", motif, re.I), motif
    assert not re.search(r"\b(429|quota|rate.?limit)", motif, re.I), motif


def test_A4_une_alarme_est_levee(lot):
    rt, _ = lot
    assert rt.metrics.compteurs.get("alarme:restriction_instances") == 1


# ── A5. Ordinary saturation is left untouched ────────────────────────────────────────────


def test_A5_une_saturation_ordinaire_garde_son_chemin(monkeypatch):
    """`RuntimeError` = full window or outage: wait, give up after retries, historical reason.

    This lot moves nothing on this path. If it did, it would trade a rare incident
    for a regression on the gateway's most frequent case.
    """
    taches = _taches(3)
    rt = _runtime(RuntimeError("All LLM providers saturated"), taches, par_pop=100)
    monkeypatch.setattr(task_worker, "get_worker_runtime", lambda: rt)

    process_batch_task.run("bk", force_provider=None, min_tpm_required=None,
                           min_output_required=None, instances_admises=None)

    assert all(t.status == TaskStatus.FAILED for t in taches)
    assert all("Providers saturés ou indisponibles" in t.error for t in taches), taches[0].error
    assert all(t.error_kind is None for t in taches), (
        "the saturation path carries no failure nature, and does not gain one here"
    )
    assert rt.metrics.compteurs.get("alarme:providers_satures") == 1
    assert "alarme:restriction_instances" not in rt.metrics.compteurs
    assert rt.queue.scheduled_cleared == [], (
        "the saturation path did not touch the dispatch flag: it still does not"
    )
