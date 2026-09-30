"""The worker's give-up rule: busy is not down."""
from llm_gateway.worker.task_worker import _providers_merely_busy


def _st(**kw):
    base = {"disabled": False, "cooldown": False, "quota_exhausted": False, "available": False}
    base.update(kw)
    return base


def test_fenetre_pleine_est_occupe_pas_en_panne():
    statuses = {"g": _st(available=False)}   # rpm full, nothing else
    assert _providers_merely_busy(statuses, None) is True
    assert _providers_merely_busy(statuses, "g") is True


def test_cooldown_disable_quota_sont_des_pannes():
    assert _providers_merely_busy({"g": _st(cooldown=True)}, None) is False
    assert _providers_merely_busy({"g": _st(disabled=True)}, None) is False
    assert _providers_merely_busy({"g": _st(quota_exhausted=True)}, None) is False


def test_le_provider_force_seul_compte():
    statuses = {"g": _st(cooldown=True), "autre": _st()}
    assert _providers_merely_busy(statuses, "g") is False, "the other provider does not count if g is forced"
    assert _providers_merely_busy(statuses, None) is True


# ── 2026-09-25: only ADMITTED instances count ───────────────────────────────────────────
#
# On 2026-09-23, the two admitted gemini31 keys were cooling down after HTTP 503s;
# an excluded model stayed free, and the rule concluded "busy": the batch waited, the client
# timed out at 120 s on a kind-less "Timeout expiré", and the simulation recorded a fallback.

import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from celery.exceptions import Retry

from llm_gateway.adapters.base import ProviderServerError
from llm_gateway.core.models import TaskStatus
from llm_gateway.worker import task_worker
from llm_gateway.worker.task_worker import (
    _attente_client_restante,
    _eligibles,
    _verdict_saturation,
    process_batch_task,
)

ADMISES = ["k1", "k2"]


def _incident():
    """Both admitted keys cooling down, one excluded model free."""
    return {"k1": _st(cooldown=True), "k2": _st(cooldown=True), "exclu": _st()}


def test_une_instance_exclue_ne_rend_pas_le_lot_occupe():
    statuses = _incident()
    assert _providers_merely_busy(statuses, None, ADMISES) is False, (
        "the free instance is EXCLUDED: it will never serve this batch"
    )
    assert _providers_merely_busy(statuses, None) is True, "without restriction, it counts"


def test_les_eligibles_sont_l_epinglee_sinon_les_admises_sinon_toutes():
    statuses = _incident()
    assert _eligibles(statuses, "k2", ADMISES) == ["k2"]
    assert _eligibles(statuses, None, ADMISES) == ["k1", "k2"]
    assert _eligibles(statuses, None, None) == ["k1", "k2", "exclu"]


def _verdict(statuses, ttl, restant, **kw):
    kw.setdefault("attendre_si_occupe", True)
    kw.setdefault("peut_rejouer", True)
    return _verdict_saturation(statuses, _eligibles(statuses, None, ADMISES), ttl, restant, **kw)


def test_un_refroidissement_qui_rouvre_a_temps_s_attend():
    v = _verdict(_incident(), {"k1": 40, "k2": 55}, restant=80)
    assert v.attendre and v.motif == "refroidissement" and v.reprise_dans_s == 40


def test_un_refroidissement_qui_rouvre_trop_tard_rend_le_lot_qualifie():
    v = _verdict(_incident(), {"k1": 40, "k2": 55}, restant=20)
    assert not v.attendre
    assert v.genre == "surcharge_fournisseur", "the client must know this is not an ordinary fallback"
    assert v.reprise_dans_s == 40, "the nearest known reopening"


def test_une_fenetre_pleine_s_attend_tant_que_le_client_attend():
    statuses = {"k1": _st(), "k2": _st(cooldown=True)}
    assert _verdict(statuses, {"k2": 55}, restant=30).motif == "occupe"
    v = _verdict(statuses, {"k2": 55}, restant=-1)
    assert not v.attendre and v.genre == "surcharge_fournisseur", (
        "beyond the client's wait, waiting longer serves no one"
    )


def test_le_quota_du_jour_sur_toutes_les_admises_est_un_quota():
    statuses = {"k1": _st(quota_exhausted=True), "k2": _st(quota_exhausted=True), "exclu": _st()}
    v = _verdict(statuses, {}, restant=80)
    assert not v.attendre and v.genre == "quota_journalier"


def test_sans_instance_eligible_connue_rien_n_est_qualifie():
    v = _verdict_saturation({}, [], {}, 80, attendre_si_occupe=True, peut_rejouer=True)
    assert not v.attendre and v.genre is None


def _reglages(wait_timeout=None):
    return SimpleNamespace(
        providers={"k1": SimpleNamespace(wait_timeout=wait_timeout, batch_max_agents=10,
                                         quota_reset_tz=None)},
        resilience=SimpleNamespace(
            client_wait_seconds=120.0,
            client_wait_margin_seconds=10.0,
            provider_wait_seconds=0.0,
            saturation_poll_seconds=0.0,
            saturation_retries=0,
            saturation_retry_seconds=12.0,
            abandon_when_busy=False,
            backoff_base_seconds=1.0,
        ),
    )


def test_l_attente_du_client_est_celle_de_l_instance_epinglee_si_declaree():
    t0 = time.time()
    assert _attente_client_restante(_reglages(), None, t0) == pytest.approx(110, abs=1)
    assert _attente_client_restante(_reglages(600), "k1", t0) == pytest.approx(590, abs=1)
    assert _attente_client_restante(_reglages(600), None, t0) == pytest.approx(110, abs=1), (
        "not pinned, the client keeps its default"
    )


# ── The full task path, runtime doubled ──────────────────────────────────────────────────


class _Queue:
    def __init__(self, taches):
        self.taches = list(taches)
        self.requeues = []

    def clear_scheduled(self, _k):
        pass

    def pop(self, _k, n):
        sortis, self.taches = self.taches[:n], self.taches[n:]
        return sortis

    def requeue(self, _k, tasks):
        self.requeues.append(list(tasks))
        self.taches.extend(tasks)

    def size(self, _k):
        return len(self.taches)


class _Store:
    def save_sync(self, _t):
        pass

    def publish_done_sync(self, _t):
        pass


class _Metrics:
    def __init__(self):
        self.compteurs = {}

    def incr(self, cle, *_a, **_k):
        self.compteurs[cle] = self.compteurs.get(cle, 0) + 1


class _Limiter:
    def __init__(self, ttl=55):
        self.ttl = ttl
        self.cooldowns = []

    def incr_active(self, _p):
        pass

    def decr_active(self, _p):
        pass

    def release_slot(self, _p):
        pass

    def cooldown(self, p, seconds):
        self.cooldowns.append((p, seconds))

    def cooldown_ttl(self, _p):
        return self.ttl


class _Balancer:
    def __init__(self, choisi=None, statuses=None):
        self.choisi = choisi
        self.statuses = statuses or {}

    def select_provider(self, **_k):
        if self.choisi is None:
            raise RuntimeError("All LLM providers saturated")
        return self.choisi

    def get_status(self):
        return self.statuses


def _taches(n=2):
    return [
        SimpleNamespace(task_id=f"t{i}", status=TaskStatus.PENDING, error=None, error_kind=None,
                        resume_at=None, updated_at=datetime.now(UTC), created_at=datetime.now(UTC))
        for i in range(n)
    ]


def _rt(balancer, taches, ttl=55):
    return SimpleNamespace(settings=_reglages(), balancer=balancer, queue=_Queue(taches),
                           store=_Store(), metrics=_Metrics(), limiter=_Limiter(ttl))


def _lancer(monkeypatch, rt, debut):
    monkeypatch.setattr(task_worker, "get_worker_runtime", lambda: rt)
    process_batch_task.run("bk", force_provider=None, min_tpm_required=None,
                           min_output_required=None, instances_admises=ADMISES,
                           debut_attente=debut)


def test_incident_du_23_09_le_lot_est_rendu_qualifie_avant_l_abandon_du_client(monkeypatch):
    """Both admitted keys cooling down, the client no longer waits: batch returned, kind set."""
    taches = _taches()
    rt = _rt(_Balancer(statuses=_incident()), taches, ttl=55)
    _lancer(monkeypatch, rt, debut=time.time() - 100)
    assert all(t.status == TaskStatus.FAILED for t in taches)
    assert all(t.error_kind == "surcharge_fournisseur" for t in taches)
    assert all(t.resume_at is not None for t in taches), "the estimated reopening travels"


def test_incident_du_23_09_le_lot_attend_une_cle_qui_rouvre_a_temps(monkeypatch):
    taches = _taches()
    rt = _rt(_Balancer(statuses=_incident()), taches, ttl=30)
    with pytest.raises(Retry):
        _lancer(monkeypatch, rt, debut=time.time())
    assert all(t.status == TaskStatus.PENDING for t in taches), "nothing is returned: we wait"


def test_un_5xx_rendu_avant_l_abandon_du_client(monkeypatch):
    taches = _taches()
    rt = _rt(_Balancer(choisi="k1"), taches)

    def _boum(*_a, **_k):
        raise ProviderServerError("k1", 503, "This model is currently experiencing high demand.")

    monkeypatch.setattr(task_worker, "_execute_batch", _boum)
    _lancer(monkeypatch, rt, debut=time.time() - 115)
    assert rt.limiter.cooldowns == [("k1", 60)]
    assert rt.queue.requeues == [], "no replay the client would not wait for"
    assert all(t.error_kind == "surcharge_fournisseur" for t in taches)
    assert rt.metrics.compteurs.get("alarme:surcharge_fournisseur") == 1


def test_un_5xx_se_rejoue_tant_que_le_client_attend(monkeypatch):
    taches = _taches()
    rt = _rt(_Balancer(choisi="k1"), taches)

    def _boum(*_a, **_k):
        raise ProviderServerError("k1", 503, "high demand")

    monkeypatch.setattr(task_worker, "_execute_batch", _boum)
    # Called directly (outside a worker), `self.retry(exc=e)` re-raises `e` itself.
    with pytest.raises((Retry, ProviderServerError)):
        _lancer(monkeypatch, rt, debut=time.time())
    assert len(rt.queue.requeues) == 1
