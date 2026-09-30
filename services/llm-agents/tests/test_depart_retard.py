"""Departures served late (2026-09-25) — replay of the treated-arm incident 2026-09-24_17_50.

The incident: the decision for agent 286921's 17:01 departure (population_4_foyer_133048,
four agents), requested at simulated 14:30, stayed in flight ~70 real s (HTTP 503 "high
demand"). The EDF queue no longer saw it (popped by a consumer), the /sync brake did not
see it (1 out of 4), GAMA ran on until 18:15. The trip left at 18:22, the return
home was dated the next day, five trips are missing against the control — and `late_since_last_sync`
stayed at 0 throughout.

These tests pin down, on a minimal instance (SimulationLoopV1 built via __new__, cf.
test_edf_dispatcher):
  - the registry of pending departure decisions, which also sees the popped task;
  - the lateness check against the trip's departure time, no longer against the
    planning base;
  - stopping the experiment (EXPERIMENT_STOP_ON_FALLBACK lock) rather than a trip served
    after its time — and its absence in a classic run;
  - the rising-edge alarm of the classic run;
  - the chaining anchor: a late push no longer shifts the day by one day.

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_depart_retard.py
"""

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

import settings as settings_module
from backpressure import departures_at_risk
from models import Activity, Location, PersonMove
from urban_mobility_agents import simulation_controller as sc
from urban_mobility_agents.simulation_controller import (
    DepartServiEnRetard,
    SimulationLoopV1,
    _depart_du_move,
    _depart_du_trajet,
    depart_prevu,
)

H = 3600
J0 = 1_774_310_400  # Tuesday 2026-03-24 00:00, simulation wall-clock time
DEPART_1701 = J0 + 17 * H + 60  # departure towards "other"
DEMANDE_1430 = J0 + 14 * H + 30 * 60
SYNC_1645 = J0 + 16 * H + 45 * 60
SERVI_1815 = J0 + 18 * H + 15 * 60

MAISON = Location(lat=43.60, lon=1.44)
AILLEURS = Location(lat=43.61, lon=1.45)


def _act(id_, depart_24h, debut_24h, fin_24h, purpose, loc):
    """`scheduled_start_time` = departure time of the trip THAT LEADS to the activity."""
    return Activity(
        id=id_,
        scheduled_start_time=depart_24h,
        start_time=debut_24h,
        end_time=fin_24h,
        purpose=purpose,
        location=loc,
    )


# The day of 286921 reduced to the two links of the incident.
OTHER = _act("152926ea", 17 * H + 60, 17 * H + 48 * 60, 19 * H + 7 * 60, "other", AILLEURS)
HOME = _act("home", 19 * H + 7 * 60, 19 * H + 30 * 60, 17 * H, "home", MAISON)


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
    loop._decisions_depart = {}
    loop._decisions_depart_seq = 0
    loop._current_sim_timestamp = DEMANDE_1430
    loop._departs_en_retard = []
    loop._departs_en_retard_total = 0
    loop._alarme_retard_active = False
    loop._dernier_retard_sim = None
    loop._episode_retards = 0
    loop._episode_retard_max_s = 0
    loop._retenues_depart = 0
    loop._retenues_depart_s = 0.0
    return loop


def _move(depart: int, activity: Activity = OTHER) -> PersonMove:
    """Trip without an OTP itinerary: its departure is `expected_arrive_at` (cf. the controller)."""
    return PersonMove(
        id="m",
        person_id="286921",
        current_time=DEMANDE_1430,
        expected_arrive_at=depart,
        purpose=activity.purpose,
        for_activity=activity,
    )


PERSONNE = SimpleNamespace(person_id="286921")


@pytest.fixture
def journal():
    """The controller logs through loguru: `caplog` does not see it."""
    lignes: list[tuple[str, str]] = []
    puits = logger.add(lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO")
    yield lignes
    logger.remove(puits)


def _erreurs_alarme(journal) -> list[str]:
    return [msg for niveau, msg in journal if niveau == "ERROR" and "[ALARME]" in msg]


@pytest.fixture
def run_classique(monkeypatch):
    for nom in sc._VERROUS_ARRET_EXPERIENCE:
        monkeypatch.delenv(nom, raising=False)


@pytest.fixture
def experience(monkeypatch):
    for nom in sc._VERROUS_ARRET_EXPERIENCE:
        monkeypatch.delenv(nom, raising=False)
    monkeypatch.setenv("EXPERIMENT_STOP_ON_FALLBACK", "1")


# ── The registry of pending departure decisions ──────────────────────────────────────────


class TestRegistreDesDecisions:
    def test_incident_la_decision_depilee_reste_visible(self):
        """The core of the incident: once popped, the decision escaped the EDF queue."""

        async def _scenario():
            loop = _make_loop()
            rendue = asyncio.Event()

            async def _decision_bloquee():
                await rendue.wait()  # the model call that never returns

            loop._dispatch(
                deadline_sim=DEPART_1701,
                kind="plan",
                make_coro=_decision_bloquee,
                person_id="286921",
                activity=OTHER,
                departure_sim=DEPART_1701,
            )
            consommateur = asyncio.create_task(loop._edf_consumer(0))
            while loop._edf_heap:
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.01)

            vu = {
                "file_edf": loop.edf_snapshot_deadlines(),
                "en_retard_1815": loop.overdue_decision_count(SERVI_1815),
                "a_risque_1645": departures_at_risk(loop.pending_departures(), SYNC_1645, H),
                "a_risque_1430": departures_at_risk(loop.pending_departures(), DEMANDE_1430, H),
            }
            rendue.set()
            await asyncio.sleep(0.02)
            vu["apres"] = loop.pending_departures()
            consommateur.cancel()
            await asyncio.gather(consommateur, return_exceptions=True)
            return vu

        vu = asyncio.run(_scenario())
        assert vu["file_edf"] == [], "the EDF queue no longer sees the popped task"
        assert vu["en_retard_1815"] == 1
        assert [p.person_id for p in vu["a_risque_1645"]] == ["286921"]
        assert vu["a_risque_1645"][0].activity_id == "152926ea"
        assert vu["a_risque_1645"][0].requested_sim == DEMANDE_1430
        assert vu["a_risque_1430"] == [], "2 h 31 before departure, GAMA advances freely"
        assert vu["apres"] == [], "the returned decision leaves the registry"

    def test_une_decision_en_echec_sort_du_registre(self):
        async def _scenario():
            loop = _make_loop()

            async def _echec():
                raise RuntimeError("503 high demand")

            loop._dispatch(
                deadline_sim=DEPART_1701, kind="refill", make_coro=_echec,
                person_id="286921", activity=OTHER, departure_sim=DEPART_1701,
            )
            assert len(loop.pending_departures()) == 1
            consommateur = asyncio.create_task(loop._edf_consumer(0))
            await asyncio.sleep(0.05)
            consommateur.cancel()
            await asyncio.gather(consommateur, return_exceptions=True)
            return loop.pending_departures()

        assert asyncio.run(_scenario()) == []

    def test_push_et_reflexion_hors_registre(self):
        loop = _make_loop()

        async def _noop():
            pass

        loop._dispatch(deadline_sim=0.0, kind="push", make_coro=_noop, person_id="p")
        loop._dispatch(deadline_sim=DEPART_1701, kind="reflect", make_coro=_noop, person_id="p")
        assert loop.pending_departures() == []

    def test_sans_heure_de_depart_l_echeance_edf_sert_de_depart(self):
        loop = _make_loop()

        async def _noop():
            pass

        loop._dispatch(deadline_sim=DEPART_1701, kind="plan", make_coro=_noop, person_id="p")
        assert loop.pending_departures()[0].departure_sim == DEPART_1701

    def test_edf_desactive_la_decision_est_aussi_suivie(self):
        async def _scenario():
            loop = _make_loop(edf_enabled=False)
            rendue = asyncio.Event()

            async def _decision():
                await rendue.wait()

            loop._dispatch(
                deadline_sim=DEPART_1701, kind="plan", make_coro=_decision,
                person_id="286921", activity=OTHER, departure_sim=DEPART_1701,
            )
            await asyncio.sleep(0.01)
            pendant = len(loop.pending_departures())
            rendue.set()
            await asyncio.sleep(0.01)
            return pendant, len(loop.pending_departures())

        try:
            assert asyncio.run(_scenario()) == (1, 0)
        finally:
            settings_module.settings.world.edf_enabled = True

    def test_stop_worker_vide_le_registre(self):
        loop = _make_loop()
        loop._worker_loop_task = None
        loop._stm_reflecting = set()
        loop._stm_reflect_due = {}
        loop._stm_floor_day = {}

        async def _noop():
            pass

        loop._dispatch(deadline_sim=DEPART_1701, kind="plan", make_coro=_noop, person_id="p")
        loop.stop_worker()
        assert loop.pending_departures() == [], "a job destroyed in the queue does not go through its finally"


# ── The departure time: a single computation ─────────────────────────────────────────────


class TestHeureDeDepart:
    def test_depart_prevu_est_le_depart_de_la_decision(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings.agent, "no_weekend_departures", False)
        assert depart_prevu(OTHER, DEMANDE_1430) == DEPART_1701
        assert _depart_du_trajet(OTHER, DEMANDE_1430) == (DEPART_1701, 0)

    def test_bouclage_au_lendemain(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings.agent, "no_weekend_departures", False)
        depart, retard = _depart_du_trajet(OTHER, SERVI_1815)
        assert depart == DEPART_1701 + 86400
        assert retard == SERVI_1815 - DEPART_1701

    def test_report_du_week_end(self, monkeypatch):
        samedi = J0 + 4 * 86400  # Saturday 2026-03-28
        monkeypatch.setattr(settings_module.settings.agent, "no_weekend_departures", True)
        assert depart_prevu(OTHER, samedi) == DEPART_1701 + 6 * 86400  # Monday 30
        monkeypatch.setattr(settings_module.settings.agent, "no_weekend_departures", False)
        assert depart_prevu(OTHER, samedi) == DEPART_1701 + 4 * 86400

    def test_la_decision_utilise_le_meme_calcul(self):
        source = Path(sc.__file__).read_text(encoding="utf-8")
        corps = source.split("async def _compute_move_for_activity(")[1][:6000]
        assert "_depart_du_trajet(next_activity, timestamp)" in corps

    def test_depart_du_move_prefere_l_itineraire(self):
        avec_plan = SimpleNamespace(
            plan=SimpleNamespace(start_time=(DEPART_1701 + 37) * 1000),
            expected_arrive_at=DEPART_1701 + 47 * 60,
        )
        assert _depart_du_move(avec_plan) == DEPART_1701 + 37
        assert _depart_du_move(_move(DEPART_1701)) == DEPART_1701


# ── The lateness check ───────────────────────────────────────────────────────────────────


class TestConstatDuRetard:
    def test_incident_servi_a_18h15_compte_en_retard(self, journal):
        loop = _make_loop()
        loop._current_sim_timestamp = SERVI_1815
        retard = loop._constater_depart_en_retard(PERSONNE, OTHER, _move(DEPART_1701), "plan")
        assert isinstance(retard, DepartServiEnRetard)
        assert retard.retard_s == 74 * 60
        assert (retard.person_id, retard.activity_id, retard.purpose) == ("286921", "152926ea", "other")
        assert loop.late_since_last_sync == 1
        assert loop._departs_en_retard_total == 1
        avertis = [m for n, m in journal if n == "WARNING" and "LATE" in m]
        assert len(avertis) == 1
        for attendu in ("286921", "152926ea", "retard≥74 min", "(plan)"):
            assert attendu in avertis[0], attendu

    def test_servi_avant_le_depart_rien(self):
        loop = _make_loop()
        loop._current_sim_timestamp = J0 + 17 * H
        assert loop._constater_depart_en_retard(PERSONNE, OTHER, _move(DEPART_1701), "plan") is None
        assert loop.late_since_last_sync == 0

    def test_servi_pile_a_l_heure_rien(self):
        loop = _make_loop()
        loop._current_sim_timestamp = DEPART_1701
        assert loop._constater_depart_en_retard(PERSONNE, OTHER, _move(DEPART_1701), "plan") is None


# ── Stopping the experiment, never in a classic run ──────────────────────────────────────


def _loop_de_refill(depart: int, servi: int):
    """Instance ready for `_precompute_one`, with a decision that returns at `servi`."""
    loop = _make_loop()
    loop._current_sim_timestamp = servi
    arrets: list = []

    @contextlib.asynccontextmanager
    async def _garde():
        yield

    async def _calcul(person, to_act, base, from_location_override=None):
        return _move(depart, to_act), None

    async def _arreter(decision, maintenant_sim=None):
        arrets.append(decision)

    loop._worker_concurrency_guard = _garde
    loop._compute_move_for_activity = _calcul
    loop._mark_completion = lambda: None
    loop.arreter_pour_decision_en_retard = _arreter
    personne = SimpleNamespace(
        person_id="286921",
        state=SimpleNamespace(
            precomputed_moves=[], precomputed_horizon_act=None,
            precomputed_horizon_ts=None, precompute_in_progress=True,
        ),
    )
    return loop, personne, arrets


class TestArretDeLExperience:
    def test_experience_un_refill_en_retard_arrete_sans_stocker(self, experience):
        loop, personne, arrets = _loop_de_refill(DEPART_1701, SERVI_1815)
        asyncio.run(loop._precompute_one(personne, HOME, DEMANDE_1430, OTHER))
        assert len(arrets) == 1 and arrets[0].retard_s == 74 * 60
        assert personne.state.precomputed_moves == [], "the late trip does not enter the queue"
        assert personne.state.precompute_in_progress is False

    def test_run_classique_le_refill_en_retard_est_servi_et_compte(self, run_classique):
        loop, personne, arrets = _loop_de_refill(DEPART_1701, SERVI_1815)
        asyncio.run(loop._precompute_one(personne, HOME, DEMANDE_1430, OTHER))
        assert arrets == [], "the stop is not armed by default"
        assert len(personne.state.precomputed_moves) == 1
        assert loop.late_since_last_sync == 1

    def test_experience_un_refill_a_l_heure_ne_s_arrete_pas(self, experience):
        loop, personne, arrets = _loop_de_refill(DEPART_1701, J0 + 16 * H)
        asyncio.run(loop._precompute_one(personne, HOME, DEMANDE_1430, OTHER))
        assert arrets == []
        assert len(personne.state.precomputed_moves) == 1

    def test_le_plan_s_arrete_avant_tout_stockage(self):
        """`_compute_and_store_planned`: the stop precedes the counter, the storage and the push."""
        source = Path(sc.__file__).read_text(encoding="utf-8")
        corps = source.split("async def _compute_and_store_planned(")[1].split("\n    async def ")[0]
        constat = corps.index('self._constater_depart_en_retard(person, activity, move, "plan")')
        suite = corps[constat:]
        arret = suite.index("await self.arreter_pour_decision_en_retard(_retard)")
        assert suite[arret:].split("\n", 2)[1].strip() == "return"
        for apres in ("_itinerary_success_count += 1", "next_planned_move ="):
            assert suite.index(apres) > arret, apres

    def test_le_verrou_est_celui_des_campagnes(self, run_classique, monkeypatch):
        loop = _make_loop()
        assert loop.arret_experience_arme is False
        monkeypatch.setenv("EXPERIMENT_HIBERNATE_ON_QUOTA", "1")
        assert loop.arret_experience_arme is True


class TestMarqueurDArret:
    def _armer(self, monkeypatch, tmp_path):
        signaux: list = []
        monkeypatch.setattr(sc, "settings", SimpleNamespace(workdir=tmp_path))
        monkeypatch.setattr(sc.os, "kill", lambda pid, sig: signaux.append(sig))
        loop = _make_loop()
        loop._current_sim_timestamp = SYNC_1645
        loop._replis_consecutifs = 0
        loop._hibernation_declenchee = False
        loop._hibernations_ignorees = 0
        loop.agent = None
        loop.model = SimpleNamespace(population=SimpleNamespace(get_people_list=list))

        async def _point(_ts):
            return None

        loop._ecrire_point_de_reprise = _point
        return loop, signaux

    def test_decision_toujours_attendue_le_marqueur_dit_quel_depart(self, monkeypatch, tmp_path, journal):
        loop, signaux = self._armer(monkeypatch, tmp_path)
        decision = sc.PendingDeparture(
            key=1, person_id="286921", activity_id="152926ea", purpose="other",
            departure_sim=DEPART_1701, kind="plan",
            requested_sim=DEMANDE_1430, requested_wall=0.0,
        )
        asyncio.run(loop.arreter_pour_decision_en_retard(decision, SYNC_1645 + 15 * 60))
        marqueur = json.loads((tmp_path / "en_attente_quota.json").read_text(encoding="utf-8"))
        assert marqueur["motif"] == "decision_en_retard"
        assert marqueur["resume_at"] is None, "a saturation announces no reopening time"
        assert marqueur["person_id"] == "286921"
        detail = marqueur["detail"]
        assert detail["etat"] == "en_attente"
        assert detail["activity_id"] == "152926ea"
        assert detail["depart_ts"] == DEPART_1701
        assert detail["demandee_ts"] == DEMANDE_1430
        assert len(signaux) == 1
        alarmes = _erreurs_alarme(journal)
        assert len(alarmes) == 1 and "Late departure avoided for 286921" in alarmes[0]
        assert "toujours attendue" in alarmes[0]

    def test_decision_rendue_en_retard_le_marqueur_porte_le_retard(self, monkeypatch, tmp_path):
        loop, signaux = self._armer(monkeypatch, tmp_path)
        retard = DepartServiEnRetard("286921", "152926ea", "other", DEPART_1701, SERVI_1815, "plan")
        asyncio.run(loop.arreter_pour_decision_en_retard(retard))
        detail = json.loads((tmp_path / "en_attente_quota.json").read_text(encoding="utf-8"))["detail"]
        assert detail["etat"] == "rendue_en_retard"
        assert detail["retard_s"] == 74 * 60
        assert len(signaux) == 1


# ── Alarm of the classic run ─────────────────────────────────────────────────────────────


class TestAlarmeRunClassique:
    def _retard(self, loop, servi):
        loop._current_sim_timestamp = servi
        loop._constater_depart_en_retard(PERSONNE, OTHER, _move(DEPART_1701), "plan")

    def test_une_alarme_par_episode_puis_levee(self, run_classique, journal):
        loop = _make_loop()
        self._retard(loop, SERVI_1815)
        loop._evaluer_alarme_depart_en_retard(SERVI_1815)
        loop._departs_en_retard = []
        premieres = _erreurs_alarme(journal)
        assert len(premieres) == 1
        for attendu in ("286921", "152926ea", "(other)", "17:01", "18:15", "74 min"):
            assert attendu in premieres[0], attendu

        self._retard(loop, SERVI_1815 + 15 * 60)  # new delay in the episode: 89 min
        loop._evaluer_alarme_depart_en_retard(SERVI_1815 + 15 * 60)
        loop._departs_en_retard = []
        loop._evaluer_alarme_depart_en_retard(SERVI_1815 + 30 * 60)  # 15 min without: nothing
        assert len(_erreurs_alarme(journal)) == 1, "rising edge: no second ERROR"
        assert loop._alarme_retard_active is True

        loop._evaluer_alarme_depart_en_retard(SERVI_1815 + 75 * 60)  # 1 h without delay
        assert loop._alarme_retard_active is False
        levees = [m for _, m in journal if "[ALARME levée]" in m]
        assert len(levees) == 1
        assert "2 late departure(s)" in levees[0] and "89 min" in levees[0]

    def test_experience_pas_d_alarme_ici(self, experience, journal):
        loop = _make_loop()
        self._retard(loop, SERVI_1815)
        loop._evaluer_alarme_depart_en_retard(SERVI_1815)
        assert _erreurs_alarme(journal) == []


# ── The chaining anchor ──────────────────────────────────────────────────────────────────


class TestAncreDuChainage:
    def _chainer(self, base):
        loop = _make_loop()
        loop._worker_in_progress = 0
        vus: list = []
        loop._dispatch = lambda **kw: vus.append(kw)
        personne = SimpleNamespace(
            person_id="286921",
            state=SimpleNamespace(
                scheduling_in_progress=False, next_planned_move=None, scheduling_started_at=None,
            ),
            identity=SimpleNamespace(activities=[HOME, OTHER]),
        )
        assert loop._try_schedule_next_after(personne, OTHER, base)
        return vus[0]

    def test_push_tardif_la_journee_ne_glisse_plus(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings.agent, "no_weekend_departures", False)
        loop = _make_loop()
        loop._current_sim_timestamp = SERVI_1815
        base = loop._base_du_chainage(_move(DEPART_1701))
        assert base == DEPART_1701
        retour = self._chainer(base)
        assert retour["deadline_sim"] == J0 + 19 * H + 7 * 60, "return home on the 24th at 19:07"
        assert retour["departure_sim"] == J0 + 19 * H + 7 * 60

    def test_l_ancienne_ancre_datait_le_retour_du_lendemain(self, monkeypatch):
        """The incident behaviour, for the record: anchor = current time (18:15)."""
        monkeypatch.setattr(settings_module.settings.agent, "no_weekend_departures", False)
        assert self._chainer(SERVI_1815)["deadline_sim"] == J0 + 86400 + 19 * H + 7 * 60

    def test_push_en_avance_ancre_inchangee(self):
        loop = _make_loop()
        loop._current_sim_timestamp = J0 + 16 * H + 50 * 60
        assert loop._base_du_chainage(_move(DEPART_1701)) == J0 + 16 * H + 50 * 60


# ── Wiring of the hold into /sync ────────────────────────────────────────────────────────


class TestBranchementDeLaRetenue:
    """`_sync_impl` is too big to be replayed here: its shape is checked (cf. test_077)."""

    SOURCE = (Path(sc.__file__).resolve().parents[1] / "handle" / "application.py").read_text(
        encoding="utf-8"
    )

    def _bloc(self) -> str:
        debut = self.SOURCE.index("# --- Hold on imminent departure (2026-09-25)")
        fin = self.SOURCE.index("_predictive_hold_s = 0.0", debut)
        return self.SOURCE[debut:fin]

    def test_muette_hors_experience(self):
        bloc = self._bloc()
        garde = bloc.index("if loop_container.scenario.arret_experience_arme:")
        assert bloc.index("await hold_while(") > garde
        assert bloc.index("arreter_pour_decision_en_retard(") > garde

    def test_bornee_par_le_read_timeout_de_gama(self):
        assert "budget_s=_cap" in self._bloc()

    def test_les_retenues_suivantes_n_ont_que_le_reste(self):
        assert "_budget_retenue = max(0.0, _cap - _departure_hold_s)" in self._bloc()
        assert "(time.time() - _drain_start) < _budget_retenue" in self.SOURCE
        assert "(time.time() - _hold_start) < _budget_retenue" in self.SOURCE

    def test_l_arret_ne_rend_pas_la_main_a_gama(self):
        bloc = self._bloc()
        arret = bloc.index("arreter_pour_decision_en_retard(")
        assert 'return MessageResponse(data="arret_experience"' in bloc[arret:]
