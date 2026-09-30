"""Port contracts: the same suite for each implementation (cf. conftest.BACKENDS).

What these tests pin down is the CONTRACT: a gateway consumer can replace Redis
with memory (tests, embedded mode) without any change in observable behavior.
"""
from __future__ import annotations

import time

from llm_gateway.core.models import AgentItem, LLMRequest, Task, TaskStatus


def _task(agent_id: str = "a1", category: str = "cat") -> Task:
    return Task(request=LLMRequest(category=category, agents=[AgentItem(agent_id=agent_id)]))


# ── TaskStore ────────────────────────────────────────────────────────────────

class TestTaskStoreContract:
    async def test_save_puis_get_rend_la_meme_tache(self, ports):
        t = _task()
        await ports.store.save(t)
        got = await ports.store.get(t.task_id)
        assert got is not None and got.task_id == t.task_id and got.status == TaskStatus.PENDING

    async def test_get_inconnu_rend_none(self, ports):
        assert await ports.store.get("does-not-exist") is None

    def test_save_sync_get_sync(self, ports):
        t = _task()
        t.status = TaskStatus.RUNNING
        ports.store.save_sync(t)
        got = ports.store.get_sync(t.task_id)
        assert got is not None and got.status == TaskStatus.RUNNING

    async def test_wait_done_rend_immediatement_une_tache_terminee(self, ports):
        t = _task()
        t.status = TaskStatus.SUCCESS
        ports.store.save_sync(t)
        ports.store.publish_done_sync(t)
        got = await ports.store.wait_done(t.task_id, timeout=1.0)
        assert got is not None and got.status == TaskStatus.SUCCESS


# ── BatchQueue ───────────────────────────────────────────────────────────────

class TestBatchQueueContract:
    async def test_add_size_pop_dans_l_ordre_des_scores(self, ports, unique_key):
        tasks = [_task(f"a{i}") for i in range(3)]
        for t in tasks:
            await ports.store.save(t)
        assert await ports.queue.add(unique_key, tasks[0].task_id, 30.0) == 1
        assert await ports.queue.add(unique_key, tasks[1].task_id, 10.0) == 2
        assert await ports.queue.add(unique_key, tasks[2].task_id, 20.0) == 3
        assert ports.queue.size(unique_key) == 3
        popped = ports.queue.pop(unique_key, 2)
        assert [t.task_id for t in popped] == [tasks[1].task_id, tasks[2].task_id]  # lowest first
        assert ports.queue.size(unique_key) == 1

    async def test_requeue_remet_les_taches(self, ports, unique_key):
        t = _task()
        await ports.store.save(t)
        await ports.queue.add(unique_key, t.task_id, 1.0)
        popped = ports.queue.pop(unique_key, 10)
        assert ports.queue.size(unique_key) == 0
        ports.queue.requeue(unique_key, popped)
        assert ports.queue.size(unique_key) == 1
        assert f"batch:{unique_key}" in ports.queue.depths(), "depths() prefixes keys with batch:"

    async def test_un_seul_dispatch_differe_par_cycle(self, ports, unique_key):
        assert await ports.queue.try_mark_scheduled(unique_key, ttl=30) is True
        assert await ports.queue.try_mark_scheduled(unique_key, ttl=30) is False
        ports.queue.clear_scheduled(unique_key)
        assert await ports.queue.try_mark_scheduled(unique_key, ttl=30) is True


# ── RateLimiter ──────────────────────────────────────────────────────────────

class TestRateLimiterContract:
    def test_reservation_lissee_puis_liberee(self, ports):
        """rpm_limit=2: one reservation per smoothing interval (60 s / rpm = 30 s)."""
        lim = ports.limiter
        assert lim.try_reserve("p1") is True
        assert lim.try_reserve("p1") is False, "the second one falls within the smoothing interval"
        assert lim.current_rpm("p1") == 1
        lim.release_slot("p1")
        assert lim.current_rpm("p1") == 0, "the returned slot no longer counts in the window"

    def test_provider_inconnu_refuse(self, ports):
        assert ports.limiter.try_reserve("unknown") is False

    def test_cooldown_et_disable(self, ports):
        lim = ports.limiter
        lim.cooldown("p2", seconds=60)
        assert lim.is_in_cooldown("p2") and lim.cooldown_ttl("p2") > 0
        assert lim.try_reserve("p2") is False
        lim.disable("p1", seconds=60)
        assert lim.is_disabled("p1") and lim.disabled_ttl("p1") > 0

    def test_echecs_consecutifs_et_succes(self, ports):
        lim = ports.limiter
        assert lim.record_failure("p1") == 1
        assert lim.record_failure("p1") == 2
        lim.record_success("p1")
        assert lim.record_failure("p1") == 1

    def test_workers_actifs(self, ports):
        lim = ports.limiter
        assert lim.incr_active("p1") == 1
        assert lim.incr_active("p1") == 2
        assert lim.active_workers("p1") == 2
        assert lim.decr_active("p1") == 1

    def test_quotas_journaliers(self, ports):
        lim = ports.limiter
        lim.record_tokens("p3", 1234)
        assert lim.daily_tokens_local_seulement("p3") == 1234
        for _ in range(3):
            time.sleep(0.002)   # beyond the smoothing interval of p3 (60 µs)
            assert lim.try_reserve("p3") is True
        assert lim.daily_requests_local_seulement("p3") == 3
        # rpd_limit=3 is declarative/informative: reserving and measuring go on beyond it
        time.sleep(0.002)
        assert lim.try_reserve("p3") is True
        assert lim.daily_requests_local_seulement("p3") == 4
        assert lim.is_quota_exhausted("p3") is False
        # Only an actual 429 (via mark_quota_exhausted_until) excludes the provider
        lim.mark_quota_exhausted_until("p3")
        assert lim.is_quota_exhausted("p3") is True
        assert lim.try_reserve("p3") is False

    def test_reset_windows_remet_les_compteurs_rpm(self, ports):
        lim = ports.limiter
        lim.try_reserve("p1")
        lim.reset_windows()
        assert lim.current_rpm("p1") == 0


# ── MetricsSink ──────────────────────────────────────────────────────────────

class TestMetricsSinkContract:
    def test_incr_get_items(self, ports):
        m = ports.metrics
        m.incr("x:a")
        m.incr("x:a", amount=4)
        m.incr("y")
        assert m.get("x:a") == 5
        assert m.get("absent") == 0
        assert m.items().get("x:a") == 5 and m.items().get("y") == 1

    def test_ring_buffer_des_erreurs(self, ports):
        m = ports.metrics
        for i in range(3):
            m.push_error({"provider": "p1", "message": f"e{i}"})
        recent = m.recent_errors(2)
        assert len(recent) == 2
        assert recent[0]["message"] == "e2", "the most recent first"


# ── LearnedLimits ────────────────────────────────────────────────────────────

class TestLearnedLimitsContract:
    def test_absent_puis_appris_puis_liste(self, ports):
        st = ports.learned
        assert st.get_max_output_tokens("p1") is None
        st.set_max_output_tokens("p1", 8192)
        st.set_max_output_tokens("p2", 4096)
        assert st.get_max_output_tokens("p1") == 8192
        assert st.all_max_output_tokens() == {"p1": 8192, "p2": 4096}

    def test_la_derniere_valeur_gagne(self, ports):
        st = ports.learned
        st.set_max_output_tokens("p1", 8192)
        st.set_max_output_tokens("p1", 2048)
        assert st.get_max_output_tokens("p1") == 2048
