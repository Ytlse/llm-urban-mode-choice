"""
tests/test_memory_infra.py — Tests of the InMemory* implementations of the ports.

Validates the contractual behavior of the ports (TaskStore, BatchQueue,
RateLimiter, MetricsSink) in pure Python — the same contract as the
Redis implementations, without a Redis daemon.
"""

import asyncio

import pytest
from pydantic import SecretStr

from llm_gateway.config import ProviderConfig
from llm_gateway.core.models import AgentItem, LLMRequest, Task, TaskStatus
from llm_gateway.infra.memory import (
    InMemoryBatchQueue,
    InMemoryMetricsSink,
    InMemoryRateLimiter,
    InMemoryTaskStore,
)


def _task(agents_count: int = 1, priority: float = 100.0) -> Task:
    request = LLMRequest(
        category="itinary_multi_agent",
        agents=[AgentItem(agent_id=f"a{i}") for i in range(agents_count)],
    )
    return Task(request=request, priority_score=priority)


def _provider(
    rpm: int = 100,
    concurrency: int = 2,
    tpm: int | None = None,
    tpm_estimate: int | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        api_key=SecretStr("k"),
        rpm_limit=rpm,
        tpm_limit=tpm,
        tpm_estimate_per_request=tpm_estimate,
        base_url="http://x",
        default_model="m",
        concurrency_limit=concurrency,
    )


# ---------------------------------------------------------------------------
# InMemoryTaskStore
# ---------------------------------------------------------------------------

class TestInMemoryTaskStore:
    def test_sync_roundtrip(self):
        store = InMemoryTaskStore()
        t = _task()
        store.save_sync(t)
        assert store.get_sync(t.task_id).task_id == t.task_id

    def test_get_missing_returns_none(self):
        assert InMemoryTaskStore().get_sync("nope") is None

    def test_async_roundtrip(self):
        async def run():
            store = InMemoryTaskStore()
            t = _task()
            await store.save(t)
            return await store.get(t.task_id)
        got = asyncio.run(run())
        assert got is not None

    def test_wait_done_returns_when_terminal(self):
        async def run():
            store = InMemoryTaskStore()
            t = _task()
            await store.save(t)

            async def finish():
                await asyncio.sleep(0.02)
                t.status = TaskStatus.SUCCESS
                store.publish_done_sync(t)

            asyncio.ensure_future(finish())
            return await store.wait_done(t.task_id, timeout=1.0)

        result = asyncio.run(run())
        assert result.status == TaskStatus.SUCCESS

    def test_wait_done_timeout_returns_last_state(self):
        async def run():
            store = InMemoryTaskStore()
            t = _task()
            await store.save(t)
            return await store.wait_done(t.task_id, timeout=0.05)
        result = asyncio.run(run())
        assert result.status == TaskStatus.PENDING

    def test_wait_done_unknown_task_returns_none(self):
        async def run():
            return await InMemoryTaskStore().wait_done("nope", timeout=0.01)
        assert asyncio.run(run()) is None


# ---------------------------------------------------------------------------
# InMemoryBatchQueue
# ---------------------------------------------------------------------------

class TestInMemoryBatchQueue:
    def _queue(self):
        store = InMemoryTaskStore()
        return store, InMemoryBatchQueue(store)

    def test_pop_orders_by_priority(self):
        store, queue = self._queue()
        late, early = _task(priority=200.0), _task(priority=50.0)
        for t in (late, early):
            store.save_sync(t)
            asyncio.run(queue.add("k", t.task_id, t.priority_score))
        popped = queue.pop("k", max_agents=10)
        assert [t.task_id for t in popped] == [early.task_id, late.task_id]

    def test_pop_respects_agent_limit_and_requeues(self):
        store, queue = self._queue()
        t1, t2 = _task(agents_count=3, priority=1.0), _task(agents_count=3, priority=2.0)
        for t in (t1, t2):
            store.save_sync(t)
            asyncio.run(queue.add("k", t.task_id, t.priority_score))
        popped = queue.pop("k", max_agents=4)
        assert [t.task_id for t in popped] == [t1.task_id]
        assert queue.size("k") == 1  # t2 put back in the queue with its score

    def test_pop_skips_expired_tasks(self):
        store, queue = self._queue()
        asyncio.run(queue.add("k", "ghost-task", 1.0))
        assert queue.pop("k", max_agents=10) == []

    def test_requeue_restores(self):
        store, queue = self._queue()
        t = _task()
        store.save_sync(t)
        queue.requeue("k", [t])
        assert queue.size("k") == 1

    def test_scheduled_flag_setnx_semantics(self):
        async def run():
            _, queue = self._queue()
            first = await queue.try_mark_scheduled("k", ttl=30)
            second = await queue.try_mark_scheduled("k", ttl=30)
            queue.clear_scheduled("k")
            third = await queue.try_mark_scheduled("k", ttl=30)
            return first, second, third
        first, second, third = asyncio.run(run())
        assert (first, second, third) == (True, False, True)

    def test_depths(self):
        store, queue = self._queue()
        t = _task()
        store.save_sync(t)
        asyncio.run(queue.add("k", t.task_id, 1.0))
        assert queue.depths() == {"batch:k": 1}


# ---------------------------------------------------------------------------
# InMemoryRateLimiter
# ---------------------------------------------------------------------------

class TestInMemoryRateLimiter:
    def test_reserve_within_limit(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100)})
        assert limiter.try_reserve("p") is True
        assert limiter.current_rpm("p") == 1

    def test_unknown_provider_refused(self):
        limiter = InMemoryRateLimiter({})
        assert limiter.try_reserve("ghost") is False

    def test_smoothing_blocks_burst(self):
        # rpm=2 → min_interval of 30s: the 2nd immediate reservation is refused
        limiter = InMemoryRateLimiter({"p": _provider(rpm=2)})
        assert limiter.try_reserve("p") is True
        assert limiter.try_reserve("p") is False

    def test_cooldown_blocks_then_expires(self):
        limiter = InMemoryRateLimiter({"p": _provider()})
        limiter.cooldown("p", seconds=60)
        assert limiter.is_in_cooldown("p") is True
        assert limiter.try_reserve("p") is False

    def test_cooldown_reports_ttl(self):
        limiter = InMemoryRateLimiter({"p": _provider()})
        assert limiter.cooldown_ttl("p") == 0
        limiter.cooldown("p", seconds=60)
        assert 0 < limiter.cooldown_ttl("p") <= 60

    def test_disable_blocks_and_reports_ttl(self):
        limiter = InMemoryRateLimiter({"p": _provider()})
        limiter.disable("p", seconds=120)
        assert limiter.is_disabled("p") is True
        assert 0 < limiter.disabled_ttl("p") <= 120
        assert limiter.try_reserve("p") is False

    def test_concurrency_limit(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=1000, concurrency=1)})
        limiter.incr_active("p")
        assert limiter.try_reserve("p") is False
        limiter.decr_active("p")
        assert limiter.try_reserve("p") is True

    def test_consecutive_errors_counter(self):
        limiter = InMemoryRateLimiter({"p": _provider()})
        assert limiter.record_failure("p") == 1
        assert limiter.record_failure("p") == 2
        limiter.record_success("p")
        assert limiter.record_failure("p") == 1

    def test_release_slot(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100)})
        limiter.try_reserve("p")
        limiter.release_slot("p")
        assert limiter.current_rpm("p") == 0

    def test_reset_windows(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100)})
        limiter.try_reserve("p")
        limiter.reset_windows()
        assert limiter.current_rpm("p") == 0

    # ── TPM reservation at the real size ────────────────────────────────

    def test_reserve_with_explicit_est_tokens(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100, tpm=10_000, tpm_estimate=3_000)})
        assert limiter.try_reserve("p", est_tokens=8_000) is True
        assert limiter.current_tpm("p") == 8_000
        # The window can no longer take the 2nd request of 8,000 tokens.
        limiter._last_req["p"] = 0.0  # neutralizes smoothing
        assert limiter.try_reserve("p", est_tokens=8_000) is False

    def test_reserve_falls_back_to_static_estimate(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100, tpm=10_000, tpm_estimate=3_000)})
        limiter.try_reserve("p")
        assert limiter.current_tpm("p") == 3_000

    def test_adjust_tokens_up_and_down(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100, tpm=10_000, tpm_estimate=3_000)})
        limiter.try_reserve("p")
        limiter.adjust_tokens("p", 2_500)   # real request larger than the flat estimate
        assert limiter.current_tpm("p") == 5_500
        limiter.adjust_tokens("p", -4_000)  # smaller real consumption
        assert limiter.current_tpm("p") == 1_500

    def test_adjust_tokens_clamps_at_zero(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100, tpm=10_000, tpm_estimate=3_000)})
        limiter.try_reserve("p")
        limiter.adjust_tokens("p", -9_999)
        assert limiter.current_tpm("p") == 0

    def test_adjust_tokens_noop_without_tpm_limit(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100)})
        limiter.adjust_tokens("p", 5_000)
        assert limiter.current_tpm("p") == 0

    def test_release_slot_with_explicit_est_tokens(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=100, tpm=10_000, tpm_estimate=3_000)})
        limiter.try_reserve("p")
        limiter.adjust_tokens("p", 2_000)          # reservation adjusted to 5,000
        limiter.release_slot("p", est_tokens=5_000)
        assert limiter.current_rpm("p") == 0
        assert limiter.current_tpm("p") == 0

    # ── Daily quotas (RPD / TPD) ────────────────────────────────────────

    @staticmethod
    def _reserve_no_smoothing(limiter: InMemoryRateLimiter, provider: str) -> bool:
        # Neutralizes time smoothing (min_interval) to test the quota alone,
        # without depending on a sleep: in prod the SWRR cadence spaces out requests.
        limiter._last_req[provider] = 0.0
        return limiter.try_reserve(provider)

    def test_no_daily_quota_never_exhausts(self):
        limiter = InMemoryRateLimiter({"p": _provider(rpm=1_000_000)})
        for _ in range(50):
            self._reserve_no_smoothing(limiter, "p")
        assert limiter.is_quota_exhausted("p") is False

    def test_rpd_limit_is_informative_and_does_not_block_locally(self):
        cfg = ProviderConfig(
            api_key=SecretStr("k"), rpm_limit=1_000_000, rpd_limit=3,
            base_url="http://x", default_model="m", concurrency_limit=100,
        )
        limiter = InMemoryRateLimiter({"p": cfg})
        granted = sum(1 for _ in range(10) if self._reserve_no_smoothing(limiter, "p"))
        # RPD limit is declarative and informative: all requests are admitted for measurement
        assert granted == 10
        assert limiter.is_quota_exhausted("p") is False
        assert limiter.daily_requests_local_seulement("p") == 10
        # Only a 429 signal excludes the instance
        limiter.mark_quota_exhausted_until("p")
        assert limiter.is_quota_exhausted("p") is True
        assert self._reserve_no_smoothing(limiter, "p") is False

    def test_tpd_limit_is_informative_and_does_not_block_locally(self):
        cfg = ProviderConfig(
            api_key=SecretStr("k"), rpm_limit=1_000_000, tpd_limit=100,
            base_url="http://x", default_model="m", concurrency_limit=100,
        )
        limiter = InMemoryRateLimiter({"p": cfg})
        assert self._reserve_no_smoothing(limiter, "p") is True
        limiter.record_tokens("p", 150)  # exceeds the declared tokens/day quota
        assert limiter.daily_tokens_local_seulement("p") == 150
        # Does not block locally
        assert self._reserve_no_smoothing(limiter, "p") is True
        assert limiter.is_quota_exhausted("p") is False
        # Only a 429 excludes the instance
        limiter.mark_quota_exhausted_until("p")
        assert limiter.is_quota_exhausted("p") is True
        assert self._reserve_no_smoothing(limiter, "p") is False


# ---------------------------------------------------------------------------
# InMemoryMetricsSink
# ---------------------------------------------------------------------------

class TestInMemoryMetricsSink:
    def test_incr_and_get(self):
        sink = InMemoryMetricsSink()
        sink.incr("llm_calls_ok_total:groq", 2)
        sink.incr("llm_calls_ok_total:groq")
        assert sink.get("llm_calls_ok_total:groq") == 3

    def test_get_missing_is_zero(self):
        assert InMemoryMetricsSink().get("nope") == 0

    def test_items_snapshot(self):
        sink = InMemoryMetricsSink()
        sink.incr("a", 1)
        sink.incr("b", 5)
        assert sink.items() == {"a": 1, "b": 5}


# ---------------------------------------------------------------------------
# LoadBalancer on the in-memory limiter (ports/infra composition)
# ---------------------------------------------------------------------------

class TestLoadBalancerWithMemoryLimiter:
    def _balancer(self, providers):
        from llm_gateway.balancer.router import LoadBalancer
        limiter = InMemoryRateLimiter(providers)
        return LoadBalancer(providers, limiter), limiter

    def test_select_reserves_slot(self):
        providers = {"p": _provider(rpm=1000)}
        balancer, limiter = self._balancer(providers)
        assert balancer.select_provider() == "p"
        assert limiter.current_rpm("p") == 1

    def test_force_unavailable_raises(self):
        providers = {"p": _provider()}
        balancer, limiter = self._balancer(providers)
        limiter.disable("p", seconds=60)
        with pytest.raises(RuntimeError):
            balancer.select_provider(force="p")

    def test_min_tpm_excludes_small_providers(self):
        small = _provider(rpm=1000)
        small.tpm_limit = 6000
        balancer, _ = self._balancer({"small": small})
        with pytest.raises(RuntimeError):
            balancer.select_provider(min_tpm=30000)

    def test_min_output_excludes_limited_providers(self):
        limited = _provider(rpm=1000)
        limited.max_output_tokens = 8192
        balancer, _ = self._balancer({"limited": limited})
        with pytest.raises(RuntimeError):
            balancer.select_provider(min_output=16384)

    def test_min_output_keeps_unlimited_and_capable_providers(self):
        limited = _provider(rpm=1000)
        limited.max_output_tokens = 8192  # < min_output → excluded
        unknown = _provider(rpm=1000)     # max_output_tokens=None → eligible
        balancer, _ = self._balancer({"limited": limited, "unknown": unknown})
        assert balancer.select_provider(min_output=16384) == "unknown"

    def test_all_saturated_raises(self):
        providers = {"p": _provider(rpm=1000, concurrency=1)}
        balancer, limiter = self._balancer(providers)
        limiter.incr_active("p")
        with pytest.raises(RuntimeError):
            balancer.select_provider()

    def test_status_snapshot(self):
        providers = {"p": _provider(rpm=10)}
        balancer, limiter = self._balancer(providers)
        limiter.try_reserve("p")
        status = balancer.get_status()
        assert status["p"]["current_rpm"] == 1
        assert status["p"]["rpm_limit"] == 10
        assert status["p"]["available"] is True
