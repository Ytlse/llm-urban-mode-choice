"""A milestone already passed is not asked again during the frozen-memory replay (2026-09-28).

At the resume of the a13 v5 control, GAMA replayed from 16/03 with the restored memory of
day 11: the survey of day 1 went again with that memory, its prompt was not that of the
treated arm, and the strict replay of the common prefix refused it (409). Outside the common prefix, nothing
would have stopped it: an answer the agent could never have given that day would have been added to the
CSV, paid for.
"""

import asyncio
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import noyau as noyau_module
from urban_mobility_agents import enquetes
from urban_mobility_agents.utils import reprise

# 16/03/2026 21:00 UTC — the evening of day 1.
SOIR_J1 = 1773694800
# 26/03/2026 12:45 UTC — the resume point of the v5 control.
POINT = 1774529100
PIDS = ["101", "102"]


class _Client:
    def __init__(self):
        self.appels = 0

    async def execute(self, payload):
        self.appels += 1
        item = payload["agents"][0]
        return SimpleNamespace(
            agents=[
                SimpleNamespace(
                    agent_id=item["agent_id"],
                    scores=dict.fromkeys(enquetes.CRITERES, 5),
                    justification="j",
                )
            ],
            provider_used="google_gemini31_key1",
        )


class _Memoire:
    def __init__(self):
        self.user_metadata: dict = {}

    def journal_trajets(self, person_id):
        return {}


@pytest.fixture(autouse=True)
def _propre(monkeypatch):
    enquetes.reinitialiser()
    noyau_module.reinitialiser()
    reprise.reinitialiser()
    monkeypatch.delenv("EXPERIMENT_SURVEY_MODES", raising=False)
    monkeypatch.setenv("EXPERIMENT_TARGET_PERSONAS", ",".join(PIDS))
    yield
    enquetes.reinitialiser()
    noyau_module.reinitialiser()
    reprise.reinitialiser()


@pytest.fixture
def journal():
    lignes: list[str] = []
    sink = logger.add(lambda m: lignes.append(str(m)), level="INFO")
    yield lignes
    logger.remove(sink)


def _jouer(tmp_path, jour=1, timestamp=SOIR_J1):
    client = _Client()
    agent = SimpleNamespace(
        llm_client=client,
        long_term_memory=_Memoire(),
        get_person_identity_description=lambda p: "récit",
    )
    personnes = [
        SimpleNamespace(person_id=p, identity=SimpleNamespace(name=p)) for p in PIDS
    ]
    asyncio.run(
        enquetes.executer_enquetes_jalon(jour, timestamp, personnes, agent, tmp_path)
    )
    return client


def _lignes_csv(tmp_path):
    with (tmp_path / "affinites_declarees.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_hors_reprise_l_enquete_est_posee(tmp_path):
    client = _jouer(tmp_path)
    attendus = len(PIDS) * (len(enquetes.modes_interroges()) + 1)
    assert client.appels == attendus
    assert {r["jour_simule"] for r in _lignes_csv(tmp_path)} == {"1"}


def test_pendant_le_gel_un_jalon_deja_mene_n_est_pas_repose(tmp_path, journal):
    _jouer(tmp_path)
    avant = _lignes_csv(tmp_path)
    reprise.geler_jusqu_a(POINT, 11)

    client = _jouer(tmp_path)

    assert client.appels == 0
    assert _lignes_csv(tmp_path) == avant
    assert any("jalon J1" in m and "déjà mené" in m for m in journal)
    assert not any("[ALARME]" in m for m in journal)


def test_pendant_le_gel_un_jalon_absent_du_csv_leve_une_alarme(tmp_path, journal):
    reprise.geler_jusqu_a(POINT, 11)

    client = _jouer(tmp_path)

    assert client.appels == 0
    assert not (tmp_path / "affinites_declarees.csv").exists()
    alarmes = [m for m in journal if "[ALARME]" in m]
    assert (
        len(alarmes) == 1
        and "jalon J1" in alarmes[0]
        and "impossible à reposer" in alarmes[0]
    )


def test_un_jalon_posterieur_au_point_de_reprise_est_pose(tmp_path):
    """The milestone of the evening of day 11 (after 12:45) was not held at the first pass."""
    reprise.geler_jusqu_a(POINT, 11)
    client = _jouer(tmp_path, jour=11, timestamp=POINT + 8 * 3600)
    assert client.appels == len(PIDS) * (len(enquetes.modes_interroges()) + 1)
