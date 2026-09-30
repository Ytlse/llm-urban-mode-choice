"""Tests for the overnight draining of STM reflections (ticket 010).

A1 — the EDF deadline of a reflection is its agent's WAKE-UP (next
occurrence of the first planned activity of the day), not a fixed delay:
evening decisions go first, the stock drains overnight in the order
of wake-ups. A2 — the composition of the stack (decisions vs reflections, missed
deadlines) is exposed by the scenario for the backlog alarm.

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_stm_reflection_drainage.py
"""

import asyncio
import time
from types import SimpleNamespace

import settings as settings_module
from models import Activity, Location, Person, PersonalIdentity
from urban_mobility_agents.simulation_controller import SimulationLoopV1
from world.population import PersonScheduler

# SIM day aligned on a Unix 24h boundary (to_24h_timestamp_full does % 86400)
DAY0 = 20_000 * 86400
H = 3600

_LOC = Location(lat=43.6, lon=1.44)


def _person(activities, person_id="p1") -> Person:
    return Person(
        person_id=person_id,
        identity=PersonalIdentity(name="Test", traits_json={}, activities=activities),
    )


def _activity(act_id, start_h, end_h, purpose="work", scheduled_start_h=None) -> Activity:
    return Activity(
        id=act_id,
        start_time=start_h * H,
        end_time=end_h * H,
        scheduled_start_time=scheduled_start_h * H if scheduled_start_h is not None else None,
        purpose=purpose,
        location=_LOC,
    )


class TestNextWakeupTs:
    """Wake-up = next occurrence of the smallest 24h time of the schedule."""

    def test_le_soir_le_reveil_est_le_lendemain_matin(self):
        # Agent back home in the evening: reflection triggered at 21:15, getting up at 8.
        p = _person([_activity("a1", 8, 17, "work"), _activity("a2", 18, 7, "home")])
        ts_2115 = DAY0 + 21 * H + 15 * 60
        assert PersonScheduler(p).next_wakeup_ts(ts_2115) == DAY0 + 86400 + 8 * H

    def test_apres_minuit_le_reveil_est_le_matin_meme(self):
        # Triggered at 0:30: the deadline is the same day's wake-up (8:00),
        # not the next day's — the LTM must be ready before the first
        # decision following the current night.
        p = _person([_activity("a1", 8, 17, "work"), _activity("a2", 18, 7, "home")])
        ts_0030 = DAY0 + 30 * 60
        assert PersonScheduler(p).next_wakeup_ts(ts_0030) == DAY0 + 8 * H

    def test_scheduled_start_time_prime_sur_start_time(self):
        # A departure rescheduled earlier brings the wake-up forward.
        p = _person([_activity("a1", 8, 17, "work", scheduled_start_h=7),
                     _activity("a2", 18, 7, "home")])
        ts_2100 = DAY0 + 21 * H
        assert PersonScheduler(p).next_wakeup_ts(ts_2100) == DAY0 + 86400 + 7 * H

    def test_sans_activite_horodatee_retourne_none(self):
        # Fallback (stm_reflection_deadline_sim_s) is up to the caller.
        assert PersonScheduler(_person(None)).next_wakeup_ts(DAY0) is None
        assert PersonScheduler(_person([])).next_wakeup_ts(DAY0) is None

    def test_les_leve_tot_ont_l_echeance_la_plus_proche(self):
        # D2: the stock drains in the order of wake-ups — EDF first serves
        # the agent who gets up at 6:00, then the one at 9:00.
        early = _person([_activity("a1", 6, 14, "work"), _activity("a2", 15, 5, "home")], "early")
        late = _person([_activity("a1", 9, 18, "work"), _activity("a2", 19, 8, "home")], "late")
        ts_2200 = DAY0 + 22 * H
        due_early = PersonScheduler(early).next_wakeup_ts(ts_2200)
        due_late = PersonScheduler(late).next_wakeup_ts(ts_2200)
        assert due_early < due_late


def _make_loop() -> SimulationLoopV1:
    """Minimal instance (cf. test_edf_dispatcher): only the attributes read by
    the composition counters are populated."""
    settings_module.settings.world.edf_enabled = True
    loop = SimulationLoopV1.__new__(SimulationLoopV1)
    loop._edf_heap = []
    loop._edf_seq = 0
    loop._edf_event = asyncio.Event()
    loop._edf_consumers = []
    loop._inflight_tasks = set()
    loop._edf_active_jobs = 0
    loop._prefixe_job_error = None
    loop._prefixe_sync_applique_ts = None
    loop._prefixe_sync_confirme_ts = None
    loop._prefixe_attente_depuis = None
    loop._worker_in_progress = 0
    loop.agent = SimpleNamespace(llm_client=SimpleNamespace(strict_replay_failure=None))
    # Register of pending departure decisions (2026-09-25), populated by _dispatch.
    loop._decisions_depart = {}
    loop._decisions_depart_seq = 0
    loop._current_sim_timestamp = 0
    loop._stm_reflecting = set()
    return loop


async def _noop():
    pass


class TestCompositionDeLaPile:
    """A2: the scenario tells late decisions from pending reflections."""

    def test_overdue_ne_compte_que_les_decisions_a_echeance_depassee(self):
        loop = _make_loop()
        now = 1_000.0
        loop._dispatch(deadline_sim=500.0, kind="plan", make_coro=_noop, person_id="p1")     # late
        loop._dispatch(deadline_sim=999.0, kind="refill", make_coro=_noop, person_id="p2")   # late
        loop._dispatch(deadline_sim=2_000.0, kind="plan", make_coro=_noop, person_id="p3")   # on time
        loop._dispatch(deadline_sim=0.0, kind="push", make_coro=_noop, person_id="p4")       # sentinel, excluded
        loop._dispatch(deadline_sim=100.0, kind="reflect", make_coro=_noop, person_id="p5")  # reflection, excluded
        assert loop.overdue_decision_count(now) == 2

    def test_pile_de_reflexions_nocturne_zero_overdue(self):
        # 2026-08-03 profile: reflections queued, deadlines at wake-up (future),
        # no decision overdue → nothing to report as ERROR.
        loop = _make_loop()
        now = float(DAY0 + 22 * H)
        for i in range(50):
            pid = f"p{i}"
            loop._stm_reflecting.add(pid)
            loop._dispatch(deadline_sim=now + 10 * H, kind="reflect", make_coro=_noop, person_id=pid)
        assert loop.overdue_decision_count(now) == 0
        assert loop.pending_reflections_count == 50

    def test_pending_reflections_compte_file_et_en_vol(self):
        # _stm_reflecting is populated before _dispatch and emptied in finally: it covers
        # reflections in the EDF queue AND those being run.
        loop = _make_loop()
        loop._stm_reflecting.update({"en_file", "en_vol"})
        assert loop.pending_reflections_count == 2


def test_prefixe_commun_attend_la_reflexion_avant_le_pas_suivant():
    """A cached response can no longer change the STM eligibility of the next step."""
    async def scenario():
        loop = _make_loop()
        loop._stm_reflecting.add("julie")
        attente = asyncio.create_task(loop.attendre_prefixe_quiescent(time.monotonic() + 1, DAY0))
        await asyncio.sleep(0.01)
        assert not attente.done()
        loop._stm_reflecting.clear()
        await asyncio.wait_for(attente, timeout=1)

    asyncio.run(scenario())


def test_prefixe_pending_reessaie_le_meme_pas_sans_redispatcher():
    async def scenario():
        loop = _make_loop()
        calculs = []
        attentes = iter((False, True, False, True))

        async def attendre(_deadline, _timestamp):
            return next(attentes)

        async def calculer(timestamp, **_kwargs):
            calculs.append(timestamp)

        loop.attendre_prefixe_quiescent = attendre
        loop.sync = calculer
        assert await loop.synchroniser_prefixe(DAY0, time.monotonic() + 1) is False
        assert calculs == []  # previous observation still in progress
        assert await loop.synchroniser_prefixe(DAY0, time.monotonic() + 1) is False
        assert calculs == [DAY0]  # computation started, but reflection not finished
        assert await loop.synchroniser_prefixe(DAY0, time.monotonic() + 1) is True
        assert calculs == [DAY0]  # final ack without a second STM trigger
        assert loop._prefixe_sync_confirme_ts == DAY0

    asyncio.run(scenario())
