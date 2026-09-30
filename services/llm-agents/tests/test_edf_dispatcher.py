"""Tests of the EDF dispatcher (ticket 003): the priority queue serves planning
tasks by increasing deadline (simulated departure time), and not in
arrival order.

The heap + consumers are tested on a minimal instance (SimulationLoopV1
built via __new__ to avoid the world_model / trip_helper dependencies), populating
only the attributes touched by _dispatch / _edf_consumer. The tests drive
the asyncio loop via asyncio.run (no dependency on pytest-asyncio).

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_edf_dispatcher.py
"""

import asyncio

import settings as settings_module
from urban_mobility_agents.simulation_controller import SimulationLoopV1


def _make_loop(edf_enabled: bool = True) -> SimulationLoopV1:
    settings_module.settings.world.edf_enabled = edf_enabled
    loop = SimulationLoopV1.__new__(SimulationLoopV1)
    loop._edf_heap = []
    loop._edf_seq = 0
    loop._edf_event = asyncio.Event()
    loop._edf_consumers = []
    loop._inflight_tasks = set()
    loop._edf_active_jobs = 0
    loop._prefixe_job_error = None
    # Registry of pending departure decisions (2026-09-25), populated by _dispatch.
    loop._decisions_depart = {}
    loop._decisions_depart_seq = 0
    loop._current_sim_timestamp = 0
    return loop


async def _drain(loop: SimulationLoopV1, n_consumers: int = 1, timeout: float = 2.0) -> None:
    """Starts n_consumers consumers, waits for the queue to be empty, then cancels them."""
    consumers = [asyncio.create_task(loop._edf_consumer(i)) for i in range(n_consumers)]
    loop._edf_consumers = consumers

    async def _wait_empty():
        while loop._edf_heap:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)  # let the last coroutine finish

    await asyncio.wait_for(_wait_empty(), timeout=timeout)
    for c in consumers:
        c.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)


def test_ordre_croissant_des_deadlines_avec_un_consommateur():
    async def _scenario():
        loop = _make_loop()
        executed: list[float] = []

        def _job(d):
            async def _run():
                executed.append(d)
            return _run

        for d in (500.0, 100.0, 900.0, 300.0, 700.0):
            loop._dispatch(deadline_sim=d, kind="plan", make_coro=_job(d), person_id="p")

        await _drain(loop, n_consumers=1)
        return executed

    assert asyncio.run(_scenario()) == [100.0, 300.0, 500.0, 700.0, 900.0]


def test_refill_lointain_ne_passe_pas_devant_un_plan_proche():
    async def _scenario():
        loop = _make_loop()
        executed: list[str] = []

        def _job(tag):
            async def _run():
                executed.append(tag)
            return _run

        # A refill at D+1 submitted BEFORE an urgent plan
        loop._dispatch(deadline_sim=1_000_000.0, kind="refill", make_coro=_job("refill_lointain"), person_id="p")
        loop._dispatch(deadline_sim=10.0, kind="plan", make_coro=_job("plan_proche"), person_id="p")

        await _drain(loop, n_consumers=1)
        return executed

    assert asyncio.run(_scenario()) == ["plan_proche", "refill_lointain"]


def test_push_passe_devant_tout():
    async def _scenario():
        loop = _make_loop()
        executed: list[str] = []

        def _job(tag):
            async def _run():
                executed.append(tag)
            return _run

        loop._dispatch(deadline_sim=5.0, kind="plan", make_coro=_job("plan"), person_id="p")
        loop._dispatch(deadline_sim=0.0, kind="push", make_coro=_job("push"), person_id="p")

        await _drain(loop, n_consumers=1)
        return executed

    assert asyncio.run(_scenario())[0] == "push"


def test_snapshot_exclut_les_push():
    async def _scenario():
        loop = _make_loop()

        async def _noop():
            return None

        loop._dispatch(deadline_sim=300.0, kind="plan", make_coro=_noop, person_id="p")
        loop._dispatch(deadline_sim=100.0, kind="refill", make_coro=_noop, person_id="p")
        loop._dispatch(deadline_sim=0.0, kind="push", make_coro=_noop, person_id="p")
        return loop.edf_snapshot_deadlines(), loop.edf_queue_depth

    deadlines, depth = asyncio.run(_scenario())
    assert deadlines == [100.0, 300.0]
    assert depth == 3


def test_snapshot_inclut_les_reflexions():
    async def _scenario():
        loop = _make_loop()

        async def _noop():
            return None

        loop._dispatch(deadline_sim=300.0, kind="plan", make_coro=_noop, person_id="p")
        loop._dispatch(deadline_sim=43_200.0, kind="reflect", make_coro=_noop, person_id="p")
        loop._dispatch(deadline_sim=0.0, kind="push", make_coro=_noop, person_id="p")
        return loop.edf_snapshot_deadlines()

    # STM reflections count in the predictive feasibility test (it is what
    # guarantees their sim deadline), pushes stay excluded.
    assert asyncio.run(_scenario()) == [300.0, 43_200.0]


def test_reflexion_lointaine_passe_apres_les_plans_proches():
    async def _scenario():
        loop = _make_loop()
        executed: list[str] = []

        def _job(tag):
            async def _run():
                executed.append(tag)
            return _run

        loop._dispatch(deadline_sim=43_200.0, kind="reflect", make_coro=_job("reflect"), person_id="p")
        loop._dispatch(deadline_sim=600.0, kind="plan", make_coro=_job("plan"), person_id="p")
        loop._dispatch(deadline_sim=86_400.0, kind="refill", make_coro=_job("refill_lointain"), person_id="p")

        await _drain(loop, n_consumers=1)
        return executed

    # Pure EDF: the reflection gives way to urgent plans but goes ahead of
    # deadlines further away than its own.
    assert asyncio.run(_scenario()) == ["plan", "reflect", "refill_lointain"]


def test_edf_desactive_utilise_le_spawn_direct():
    async def _scenario():
        loop = _make_loop(edf_enabled=False)
        executed: list[str] = []

        def _job(tag):
            async def _run():
                executed.append(tag)
            return _run

        # Without EDF, _dispatch does a direct _spawn (fire-and-forget), nothing queued.
        loop._dispatch(deadline_sim=5.0, kind="plan", make_coro=_job("a"), person_id="p")
        heap_empty = (loop._edf_heap == [])
        await asyncio.gather(*list(loop._inflight_tasks), return_exceptions=True)
        return heap_empty, executed

    heap_empty, executed = asyncio.run(_scenario())
    assert heap_empty
    assert executed == ["a"]
