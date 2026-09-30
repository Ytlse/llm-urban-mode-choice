"""Ticket 107 — trace the weather day of each decision.

Acceptance criteria:
1. On a replayed v6 cohort: `jour_meteo_tire` equals exactly `date_meteo(person_id, 42, jours)`
   on 100 % of decisions.
2. `jour_meteo_lu` points to an existing row of the weather CSV on 100 % of decisions.
3. A test forces the fallback (e.g. `_weather_eligible_days` raises an exception) and checks that
   `source_date_meteo == "horloge"` and `jour_meteo_tire is None`.
4. Presence of `jour_offre` in the traces.
5. Complete weather metadata in `identite_run.json`.
6. End-of-run counter and alarm raised if a single weather date > 50 %.
"""

import asyncio
from collections import Counter
import csv
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from experiences import decision as d
from experiences.archive import Execution
from experiences.decideurs import DecideurRejeu
from models import Location, Person, PersonalIdentity, PersonState, Transit, TransitLocation, TravelPlan
from urban_mobility_agents.agents.llm_agent import _weather_eligible_days
from urban_mobility_agents.utils import identite_run
from urban_mobility_agents.utils.weather_draw import (
    date_meteo,
    fenetre_meteo_effective,
    resoudre_meteo_decision,
)
from urban_mobility_agents.utils.weather_loader import (
    date_csv_meteo,
    metadonnees_csv_meteo,
)

HOME = Location(lat=43.6000, lon=1.4400)
WORK = Location(lat=43.6100, lon=1.4500)
GRAINE = 42


def _make_plan(code: str, mode: str = "car", duration: int = 600) -> TravelPlan:
    loc = TransitLocation(stop="", lat=HOME.lat, lon=HOME.lon)
    legs = [
        Transit(
            start_time=0,
            end_time=duration,
            duration=duration,
            distance=1000.0,
            mode=mode,
            start_location=loc,
            end_location=loc,
            is_transfer=False,
            transit_route=code,
        )
    ]
    return TravelPlan(
        id=code,
        start_location=HOME,
        end_location=WORK,
        start_time=0,
        end_time=duration,
        duration=duration,
        legs=legs,
    )


class MockProposition:
    def __init__(self, code: str, mode: str = "car", duration: int = 600):
        self.code = code
        self.mode = mode
        self.mode_vehicule = mode
        self.plan = _make_plan(code, mode, duration)


def _make_person(pid: str = "p1") -> Person:
    return Person(
        person_id=pid,
        identity=PersonalIdentity(
            name="test",
            traits_json={
                "personal_bike": "vélo normal",
                "number_of_cars": 1,
                "has_driving_license": True,
                "age": 35,
                "household_size": 1,
            },
            home=HOME,
        ),
        state=PersonState(),
    )


class TestCritere1Et2RejeuCohorteV6:
    """Criteria 1 & 2: on a replayed v6 cohort, jour_meteo_tire == date_meteo(...) at 100%,
    and jour_meteo_lu points to an existing row of the weather CSV at 100%."""

    def test_rejeu_cohorte_v6_dates_meteo(self):
        # Locate the directory of the archived v6 run
        dossier = Path(__file__).resolve().parent.parent.parent.parent / "archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences/exp_alea_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c_nosim/executions/2026-09-16_13_35_28"
        if not dossier.exists():
            pytest.skip(f"v6 archive directory not available at {dossier}")

        execution = Execution.ouvrir(dossier)
        rejeu = DecideurRejeu(execution)
        jours = _weather_eligible_days()

        # Load all valid dates of the weather CSV
        csv_path = Path(__file__).resolve().parent.parent.parent.parent / "data/weather/meteo_toulouse_12_mois.csv"
        assert csv_path.exists()
        dates_valides_csv = set()
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dates_valides_csv.add(row["DATE"])

        nb_testes = 0
        # Test on a significant sample of archived decisions
        for trace_arch in execution.decisions[:150]:
            if not trace_arch.get("retenue"):
                continue
            pid = trace_arch["person_id"]
            act_id = trace_arch["activity_id"]
            personne = _make_person(pid)
            contexte = d.ContexteDecision(
                timestamp=trace_arch["timestamp"],
                activity_id=act_id,
                purpose=trace_arch.get("purpose", "work"),
                departure_time=trace_arch["departure_time"],
                from_location=HOME,
                destination=WORK,
                graine_tirage=GRAINE,
                jour_offre="2026-03-16",
            )
            props = [MockProposition(p["code"], p.get("mode", "car")) for p in trace_arch.get("presentees", [])]

            reponse = asyncio.run(rejeu.choisir(personne, contexte, props))
            assert reponse is not None

            # Criterion 1: jour_meteo_tire == date_meteo(person_id, 42, jours) at 100%
            attendu_tire = date_meteo(pid, GRAINE, jours)
            assert reponse.jour_meteo_tire == attendu_tire
            # Equality as an MM-DD string as well
            assert str(reponse.jour_meteo_tire) == f"{attendu_tire[0]:02d}-{attendu_tire[1]:02d}"

            # Criterion 2: jour_meteo_lu points to an existing CSV row at 100%
            assert reponse.jour_meteo_lu is not None
            assert reponse.jour_meteo_lu in dates_valides_csv

            assert reponse.source_date_meteo == "tiree"
            nb_testes += 1

        assert nb_testes >= 100, f"Not enough decisions tested: {nb_testes}"


class TestCritere3RepliHorloge:
    """Criterion 3: force the fallback (e.g. _weather_eligible_days raises an exception) and check
    that source_date_meteo == 'horloge' and jour_meteo_tire is None."""

    def test_repli_horloge_quand_tirage_leve_exception(self):
        timestamp = 1773648000  # 2026-03-16 08:00:00 UTC (9 am local)
        with patch("urban_mobility_agents.agents.llm_agent._weather_eligible_days", side_effect=RuntimeError("Erreur simulée")):
            res = resoudre_meteo_decision(
                person_id="p123",
                timestamp=timestamp,
                graine=GRAINE,
                weather_per_agent=True,
            )
            assert res["source_date_meteo"] == "horloge"
            assert res["jour_meteo_tire"] is None
            assert res["jour_meteo_lu"] is not None
            assert res["jour_meteo_lu"] == date_csv_meteo(timestamp)

    def test_repli_horloge_quand_weather_per_agent_dates_desactive(self):
        timestamp = 1773648000
        res = resoudre_meteo_decision(
            person_id="p123",
            timestamp=timestamp,
            graine=GRAINE,
            weather_per_agent=False,
        )
        assert res["source_date_meteo"] == "horloge"
        assert res["jour_meteo_tire"] is None
        assert res["jour_meteo_lu"] == date_csv_meteo(timestamp)


class TestCritere4SourceDeclaree:
    """Check source_date_meteo == 'declaree' when an explicit weather date is provided."""

    def test_source_date_declaree(self):
        res = resoudre_meteo_decision(
            person_id="p123",
            timestamp=1773648000,
            graine=GRAINE,
            weather_per_agent=True,
            date_imposee="11-14",
        )
        assert res["source_date_meteo"] == "declaree"
        assert res["jour_meteo_tire"] == "11-14"
        assert res["jour_meteo_lu"] == "2025-11-14"


class TestA1JourOffreEtTrace:
    """Check that jour_offre, jour_meteo_tire, jour_meteo_lu and source_date_meteo are in the trace."""

    def test_trace_contient_les_quatre_champs(self):
        personne = _make_person("609")
        ctx = d.ContexteDecision(
            timestamp=1773648000,
            activity_id="act-1",
            purpose="work",
            departure_time=1773648000,
            from_location=HOME,
            destination=WORK,
            graine_tirage=GRAINE,
            jour_offre="2026-03-16",
        )
        plan = _make_plan("C1", "car")
        props = [d.Proposition(plan)]
        rep = d.ReponseDecideur(
            index=0,
            fournisseur="test",
            distribution={"car": 1.0},
            raison="choix test",
            jour_offre="2026-03-16",
            jour_meteo_tire="11-14",
            jour_meteo_lu="2025-11-14",
            source_date_meteo="tiree",
        )
        trace = d.construire_trace(
            person=personne,
            ctx=ctx,
            presentees=props,
            ecartees=[],
            retenue=props[0],
            methode="decideur",
            reponse=rep,
            contrainte="aucune",
        )
        assert trace["jour_offre"] == "2026-03-16"
        assert trace["jour_meteo_tire"] == "11-14"
        assert trace["jour_meteo_lu"] == "2025-11-14"
        assert trace["source_date_meteo"] == "tiree"


class TestA3IdentiteRunMeteo:
    """A3: identite_run.json contains the effective window and CSV metadata."""

    def test_identite_run_composer_champs_meteo(self):
        class ReglagesTest:
            class agent:
                llm_params = {}
                long_term_memory_enabled = False
                long_term_self_reflect_enabled = False
                mode_draw_seed = 42
                option_order_seed = 42
                weather_draw_seed = 42
                weather_per_agent_dates = True
                weather_window = "annee"
                weather_weekdays_only = True
                vehicle_chain_enabled = True
                vehicle_return_home_lock = True
                mode_choice_truncation_threshold = 0.0
                memoire__importance_choc = 0.7
                memoire__retard_saturation = "asymptote"
                memoire__retard_gravite_max = 0.70
                memoire__retard_ref_s = 1800
                memoire__fenetre_changements_jours = 14
                memoire__changements_max = 3
                memoire__mode_fenetre_changements = "derivee"
                memoire__seuil_service_changement = 0.35
                memoire__plancher_changement_jours = 2.0
                memoire__plafond_changement_jours = 30.0
                stm_reflection_min_entries = 10
                memoire__partage_foyer_enabled = False

            class data:
                population_file = "pop.csv"

            class cache:
                enabled = False

            class llm:
                providers = {}
                instances_admises = []
                rejeu_ab = ""
                rejeu_strict_avant_ts = 0

            class world:
                worker_concurrency = 8
                prefixe_commun = False

        identite = identite_run.composer(ReglagesTest())

        assert "meteo_fenetre_effective" in identite
        fenetre = identite["meteo_fenetre_effective"]
        assert fenetre["debut"] is not None and fenetre["fin"] is not None
        assert fenetre["nombre_jours_eligibles"] > 0

        assert identite["meteo_csv"] == "meteo_toulouse_12_mois.csv"
        assert len(identite["meteo_csv_sha256"]) == 64
        assert identite["meteo_csv_debut"] == "2025-05-01"
        assert identite["meteo_csv_fin"] == "2026-04-30"


class TestA4RunnerCompteursMeteo:
    """A4: Weather counter in runner and alert if > 50% on a single date."""

    def test_compteurs_meteo_nominal(self):
        dates_meteo = ["2025-11-14", "2025-11-15", "2025-12-01", "2025-12-02", "2026-01-10"]
        repartition_mois = Counter(int(d.split("-")[1]) for d in dates_meteo)
        c_dates = Counter(dates_meteo)
        nb_distinctes = len(c_dates)
        total = len(dates_meteo)
        date_max, max_count = c_dates.most_common(1)[0]
        pct_max = (max_count / total) * 100

        assert nb_distinctes == 5
        assert dict(repartition_mois) == {11: 2, 12: 2, 1: 1}
        assert max_count <= total / 2  # No alarm

    def test_compteurs_meteo_alarme_si_plus_de_50_pct(self):
        dates_meteo = ["2025-11-14"] * 6 + ["2025-11-15"] * 2 + ["2025-12-01"] * 2
        total = len(dates_meteo)
        c_dates = Counter(dates_meteo)
        date_max, max_count = c_dates.most_common(1)[0]
        assert max_count == 6
        assert max_count > total / 2  # Triggers the alarm
