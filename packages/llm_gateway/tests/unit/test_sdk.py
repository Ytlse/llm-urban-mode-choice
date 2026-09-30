"""
tests/test_sdk.py — Tests of the typed client SDK (LLMGatewayClient / TaskResult).

The gateway is simulated by an httpx.MockTransport: no server required.
"""

import asyncio

import httpx

from llm_gateway.core.models import TaskStatus
from llm_gateway.sdk import LLMGatewayClient, TaskResult, TaskTiming


def _gateway_mock(wait_response: dict, status_code: int = 200):
    """Transport that answers POST /tasks then GET /tasks/{id}/wait."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tasks" and request.method == "POST":
            return httpx.Response(202, json={"task_id": "t-123", "status": "pending"})
        if request.url.path == "/tasks/t-123/wait":
            return httpx.Response(status_code, json=wait_response)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _execute(client: LLMGatewayClient) -> TaskResult:
    async def run():
        try:
            return await client.execute({"category": "itinary_multi_agent", "agents": [{"agent_id": "a", "perception": "p"}]})
        finally:
            await client.aclose()
    return asyncio.run(run())


class TestExecute:
    def test_success_result_is_typed(self, tmp_path):
        transport = _gateway_mock({
            "task_id": "t-123",
            "status": "success",
            "created_at": "2026-07-07T10:00:00Z",
            "updated_at": "2026-07-07T10:00:01Z",
            "result": [{"agent_id": "a", "chosen_index": 2, "mode": "bus", "reason": "fast"}],
            "provider_used": "groq_llama4",
            "timing_p5": {"P5_4_ms": 850.0},
        })
        client = LLMGatewayClient(transport=transport, dialogue_log_file=str(tmp_path / "dlg.log"))
        result = _execute(client)

        assert result.ok is True
        assert result.status is TaskStatus.SUCCESS
        assert result.task_id == "t-123"
        assert result.provider_used == "groq_llama4"
        assert result.agents[0].chosen_index == 2
        assert result.agents[0].mode == "bus"
        assert result.timing.timing_p5 == {"P5_4_ms": 850.0}
        assert result.timing.post_ms >= 0
        assert result.timing.wait_ms >= 0

    def test_extra_fields_accessible(self, tmp_path):
        # The reflection categories return fields outside the schema (extra=allow)
        transport = _gateway_mock({
            "status": "success",
            "created_at": "2026-07-07T10:00:00Z",
            "updated_at": "2026-07-07T10:00:01Z",
            "result": [{"agent_id": "a", "reflection": "good day", "concepts": [["c1"]]}],
        })
        client = LLMGatewayClient(transport=transport, dialogue_log_file=str(tmp_path / "dlg.log"))
        result = _execute(client)
        assert result.agents[0].reflection == "good day"
        assert result.agents[0].concepts == [["c1"]]

    def test_failed_task(self, tmp_path):
        transport = _gateway_mock({
            "status": "failed",
            "created_at": "2026-07-07T10:00:00Z",
            "updated_at": "2026-07-07T10:00:01Z",
            "result": None,
            "error": "Max retries exceeded",
        })
        client = LLMGatewayClient(transport=transport, dialogue_log_file=str(tmp_path / "dlg.log"))
        result = _execute(client)
        assert result.ok is False
        assert result.status is TaskStatus.FAILED
        assert "Max retries" in result.error

    def test_non_terminal_status_maps_to_failed_timeout(self, tmp_path):
        # /wait budget exhausted: the gateway returns the current state (pending)
        transport = _gateway_mock({
            "status": "pending",
            "created_at": "2026-07-07T10:00:00Z",
            "updated_at": "2026-07-07T10:00:00Z",
        })
        client = LLMGatewayClient(transport=transport, dialogue_log_file=str(tmp_path / "dlg.log"))
        result = _execute(client)
        assert result.ok is False
        assert result.error == "Timeout expiré"

    def test_gateway_5xx_on_wait(self, tmp_path):
        transport = _gateway_mock({}, status_code=503)
        client = LLMGatewayClient(transport=transport, dialogue_log_file=str(tmp_path / "dlg.log"))
        result = _execute(client)
        assert result.ok is False
        assert "Gateway error 503" in result.error

    def test_dialogue_log_written(self, tmp_path):
        log_file = tmp_path / "dlg.log"
        transport = _gateway_mock({
            "status": "success",
            "created_at": "2026-07-07T10:00:00Z",
            "updated_at": "2026-07-07T10:00:01Z",
            "result": [{"agent_id": "a", "mode": "bus"}],
        })
        client = LLMGatewayClient(transport=transport, dialogue_log_file=str(log_file))
        _execute(client)
        content = log_file.read_text(encoding="utf-8")
        assert "REQUEST (Payload)" in content
        assert "bus" in content


class TestTaskResultModel:
    def test_ok_requires_success_and_agents(self):
        assert TaskResult(status=TaskStatus.SUCCESS).ok is False
        assert TaskResult(status=TaskStatus.FAILED).ok is False

    def test_default_timing(self):
        t = TaskTiming()
        assert t.post_ms == 0.0 and t.wait_ms == 0.0 and t.timing_p5 is None


class TestBackpressure:
    def test_alarm_arms_and_success_disarms(self):
        client = LLMGatewayClient(backpressure_max_inflight=20)
        # Below the alarm threshold: nothing armed
        for _ in range(client._FAILURE_ALARM_THRESHOLD - 1):
            client._observe(TaskResult(status=TaskStatus.FAILED, error="boom"),
                            {"category": "c"}, "c", 0.0, 0.0)
        assert client._backpressure_active is False
        # At the threshold: backpressure armed
        client._observe(TaskResult(status=TaskStatus.FAILED, error="boom"),
                        {"category": "c"}, "c", 0.0, 0.0)
        assert client._backpressure_active is True
        # A successful response disarms
        client._observe(TaskResult(status=TaskStatus.SUCCESS, agents=[]),
                        {"category": "c"}, "c", 0.0, 0.0)
        # agents=[] → not ok; force a real success
        from llm_gateway.core.models import AgentResponse
        client._observe(TaskResult(status=TaskStatus.SUCCESS, agents=[AgentResponse(agent_id="a")]),
                        {"category": "c"}, "c", 0.0, 0.0)
        assert client._backpressure_active is False

    def test_drain_waits_until_below_threshold(self):
        async def run():
            client = LLMGatewayClient(backpressure_max_inflight=20, backpressure_release_ratio=0.2)  # threshold=4
            client._backpressure_active = True
            client._inflight = 20
            released = {"done": False}

            async def waiter():
                await client._await_backpressure_drain()
                released["done"] = True

            t = asyncio.create_task(waiter())
            await asyncio.sleep(0.6)
            assert released["done"] is False  # stack full → still suspended
            client._inflight = 4               # drains below the threshold
            await asyncio.sleep(0.7)
            assert released["done"] is True
            assert client._backpressure_active is False
            await t

        asyncio.run(run())


class TestCircuitBreaker:
    """Circuit breaker: after N consecutive failures, submissions are SUSPENDED
    (no degraded decision — we wait for recovery); one of the suspended callers
    periodically becomes the probe that closes the circuit at the first success."""

    @staticmethod
    def _failing_transport(counter: dict):
        def handler(request: httpx.Request) -> httpx.Response:
            counter["requests"] += 1
            if request.url.path == "/tasks" and request.method == "POST":
                return httpx.Response(202, json={"task_id": "t-123", "status": "pending"})
            return httpx.Response(200, json={"status": "failed", "error": "Providers saturés"})
        return httpx.MockTransport(handler)

    @staticmethod
    def _success_transport(counter: dict):
        def handler(request: httpx.Request) -> httpx.Response:
            counter["requests"] += 1
            if request.url.path == "/tasks" and request.method == "POST":
                return httpx.Response(202, json={"task_id": "t-123", "status": "pending"})
            return httpx.Response(200, json={
                "status": "success",
                "result": [{"agent_id": "a", "chosen_index": 0}],
            })
        return httpx.MockTransport(handler)

    def test_opens_after_threshold_and_suspends_submissions(self, tmp_path):
        async def run():
            counter = {"requests": 0}
            client = LLMGatewayClient(
                transport=self._failing_transport(counter),
                dialogue_log_file=None,
                circuit_failure_threshold=3,
                circuit_probe_interval=3600.0,
            )
            for _ in range(3):
                result = await client.execute({"category": "c", "agents": []})
                assert result.ok is False
            assert client.circuit_open is True
            requests_before = counter["requests"]

            # Circuit open, probe not yet due: the submission is SUSPENDED
            # (it waits for recovery) — no HTTP request, no failure returned.
            waiter = asyncio.create_task(client.execute({"category": "c", "agents": []}))
            await asyncio.sleep(0.5)
            assert waiter.done() is False
            assert counter["requests"] == requests_before
            assert client._circuit_waiters == 1

            # Recovery (as if another call's probe had succeeded):
            # the suspended submission resumes on the nominal path.
            client._transport = self._success_transport(counter)
            await client.aclose()  # forces the HTTP client to be rebuilt
            client._circuit_open_since = None
            result = await waiter
            assert result.ok is True
            assert client._circuit_waiters == 0
            await client.aclose()

        asyncio.run(run())

    def test_probe_closes_circuit_on_success(self, tmp_path):
        async def run():
            counter = {"requests": 0}
            client = LLMGatewayClient(
                transport=self._failing_transport(counter),
                dialogue_log_file=None,
                circuit_failure_threshold=2,
                circuit_probe_interval=0.0,  # probe allowed immediately
            )
            for _ in range(2):
                await client.execute({"category": "c", "agents": []})
            assert client.circuit_open is True

            # The gateway has recovered: the next call becomes the probe, goes through,
            # gets its REAL LLM decision and closes the circuit.
            await client.aclose()  # forces the HTTP client to be rebuilt at the next call
            client._transport = self._success_transport(counter)
            result = await client.execute({"category": "c", "agents": []})
            assert result.ok is True
            assert client.circuit_open is False
            await client.aclose()

        asyncio.run(run())

    def test_failed_probe_keeps_circuit_open_and_reschedules(self):
        async def run():
            counter = {"requests": 0}
            client = LLMGatewayClient(
                transport=self._failing_transport(counter),
                dialogue_log_file=None,
                circuit_failure_threshold=2,
                circuit_probe_interval=3600.0,
            )
            for _ in range(2):
                await client.execute({"category": "c", "agents": []})
            assert client.circuit_open is True

            # Forced probe (as if the interval had elapsed): it fails →
            # the circuit stays open and the next probe is postponed. The probing
            # call returns its failure (it will be suspended again at its next submission).
            client._circuit_next_probe_at = 0.0
            requests_before = counter["requests"]
            result = await client.execute({"category": "c", "agents": []})
            assert result.ok is False
            assert counter["requests"] > requests_before  # the probe did go through
            assert client.circuit_open is True
            import time as _time
            assert client._circuit_next_probe_at > _time.monotonic()
            await client.aclose()

        asyncio.run(run())

    def test_network_error_counts_as_failure(self):
        # Gateway unreachable (ConnectError): without counting, neither the alarm nor the
        # circuit breaker fired on a clean cut.
        async def run():
            def handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("connection refused")

            client = LLMGatewayClient(
                transport=httpx.MockTransport(handler),
                dialogue_log_file=None,
                circuit_failure_threshold=2,
                circuit_probe_interval=3600.0,
            )
            for _ in range(2):
                result = await client.execute({"category": "c", "agents": []})
                assert result.ok is False
                assert "injoignable" in result.error
            assert client.circuit_open is True
            await client.aclose()

        asyncio.run(run())

    def test_disabled_when_threshold_zero(self):
        async def run():
            counter = {"requests": 0}
            client = LLMGatewayClient(
                transport=self._failing_transport(counter),
                dialogue_log_file=None,
                circuit_failure_threshold=0,
            )
            for _ in range(15):
                await client.execute({"category": "c", "agents": []})
            assert client.circuit_open is False
            await client.aclose()

        asyncio.run(run())


class TestAttenteParAppel:
    """The wait of a task can be overridden per call (`execute(wait_timeout=…)`).

    The client default is set for a remote provider. A local model serves only one
    call at a time: the wait of a task is not its generation time but the time spent in the
    QUEUE. On 2026-09-08, on Muse Glimmer (28B, 40 to 120 s per generation, a single
    concurrent call), every task after the first one expired at 120 s and the circuit breaker
    opened. The caller that pins an instance therefore passes it its wait, read from the
    providers file (`wait_timeout`).
    """

    @staticmethod
    def _client_espion(vus: list):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/tasks" and request.method == "POST":
                return httpx.Response(202, json={"task_id": "t-123", "status": "pending"})
            if request.url.path == "/tasks/t-123/wait":
                vus.append(request.url.params.get("timeout"))
                return httpx.Response(200, json={"task_id": "t-123", "status": "success",
                                                 "result": [{"agent_id": "a", "chosen_index": 0}]})
            return httpx.Response(404)

        return LLMGatewayClient(transport=httpx.MockTransport(handler), wait_timeout=120.0)

    def test_sans_surcharge_le_defaut_du_client_est_transmis(self):
        vus: list = []
        client = self._client_espion(vus)
        asyncio.run(client.execute({"category": "c", "agents": [{"agent_id": "a"}]}))
        assert vus == ["120.0"]

    def test_la_surcharge_par_appel_est_transmise_au_long_poll(self):
        vus: list = []
        client = self._client_espion(vus)
        asyncio.run(client.execute({"category": "c", "agents": [{"agent_id": "a"}]}, wait_timeout=600))
        assert vus == ["600.0"], "the pinned instance imposes its wait, not the client default"

    def test_une_surcharge_plus_longue_ne_se_fait_pas_couper_par_le_client_http(self):
        """The shared httpx client is built on the DEFAULT wait (+30 s): without a timeout
        set per request, it would cut a 600 s long-poll well before its end."""
        vus: list = []
        client = self._client_espion(vus)

        async def scenario():
            await client.execute({"category": "c", "agents": [{"agent_id": "a"}]}, wait_timeout=600)
            http = await client._http()
            return http.timeout

        timeout_partage = asyncio.run(scenario())
        # The shared client keeps the default timeout; the request carried its own.
        assert timeout_partage.read == 150.0
        assert vus == ["600.0"]
