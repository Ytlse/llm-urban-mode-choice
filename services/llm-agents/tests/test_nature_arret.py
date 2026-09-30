"""The nature of an experiment stop: only an overload is retried (2026-09-29).

The control arm of a13 v6 stopped three times at the same instant (26/03 07:30, decision of
1127260) on a 409 `rejeu_obligatoire_absent`. The controller had filed it as `decision_absente`,
with no kind, and the night chain retried it hour after hour. These tests call the REAL controller
code (stop, consolidation, survey) with minimal doubles, and read the marker it writes.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urban_mobility_agents import enquetes
from urban_mobility_agents import simulation_controller as sc
from urban_mobility_agents.agents.llm_agent import ConsolidationMemoryUnavailable
from urban_mobility_agents.utils.nature_arret import (
    DEFAUT,
    PASSAGER,
    QUOTA,
    nature_arret,
    nature_du_marqueur,
)

T0 = 1774510200  # 26/03/2026 07:30, the instant of the 409 of the night of 29/09


# ── The rule ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("motif", "genre", "attendu"),
    [
        # Overload: the reason says so, or the kind the gateway qualified.
        ("surcharge_fournisseur", None, PASSAGER),
        ("decision_en_retard", None, PASSAGER),
        ("decision_absente", "surcharge_fournisseur", PASSAGER),
        (
            "consolidation_memoire",
            "surcharge_fournisseur",
            PASSAGER,
        ),  # 504 of 29/09 14:37
        ("enquete_incomplete", "surcharge_fournisseur", PASSAGER),
        ("enquete_incomplete", "panne_simulee", PASSAGER),  # make banc-reprise bench
        # Daily quota.
        ("quota_journalier", None, QUOTA),
        ("consolidation_memoire", "quota_journalier", QUOTA),
        # Defect: the common prefix, a kind that is not the overload, a missing cause.
        ("prefixe_commun", "rejeu_obligatoire_absent", DEFAUT),  # the 409 of that night
        ("prefixe_commun", "prefixe_bloque", DEFAUT),
        ("prefixe_commun", "surcharge_fournisseur", DEFAUT),  # the prefix prevails
        ("decision_absente", None, DEFAUT),
        ("decision_absente", "restriction_instances", DEFAUT),
        ("consolidation_memoire", None, DEFAUT),
        ("enquete_incomplete", None, DEFAUT),
        ("replis_consecutifs", None, DEFAUT),
        ("un_motif_inconnu", "surcharge_fournisseur", DEFAUT),
        (None, None, DEFAUT),
    ],
)
def test_la_nature_de_chaque_arret(motif, genre, attendu):
    assert nature_arret(motif, genre) == attendu


def test_un_marqueur_porte_sa_nature_ou_elle_se_recalcule():
    assert (
        nature_du_marqueur({"motif": "decision_absente", "nature": "passager"})
        == PASSAGER
    )
    # The real marker of the v6 control arm (archive 2026-09-29_17_48), written before the fix.
    assert (
        nature_du_marqueur(
            {
                "motif": "decision_absente",
                "resume_at": None,
                "replis_consecutifs": 1,
                "person_id": "1127260",
                "timestamp": 1774510516,
                "jour_simule": 11,
            }
        )
        == DEFAUT
    )
    assert nature_du_marqueur({"motif": "surcharge_fournisseur"}) == PASSAGER
    assert (
        nature_du_marqueur(
            {"motif": "enquete_incomplete", "cause": "surcharge_fournisseur"}
        )
        == PASSAGER
    )
    assert (
        nature_du_marqueur(
            {"motif": "surcharge_fournisseur", "nature": "n_importe_quoi"}
        )
        == PASSAGER
    )
    assert nature_du_marqueur({}) == DEFAUT
    assert nature_du_marqueur(None) == DEFAUT


# ── The controller writes the nature into the marker ───────────────────────────────────────


def _boucle(client=None) -> SimpleNamespace:
    """The loop reduced to what the stop and its callers read."""

    async def _point(_ts):
        return None

    return SimpleNamespace(
        agent=SimpleNamespace(
            llm_client=client or SimpleNamespace(strict_replay_failure=None)
        ),
        _current_sim_timestamp=T0,
        _replis_consecutifs=1,
        _hibernation_declenchee=False,
        _hibernations_ignorees=0,
        _ecrire_point_de_reprise=_point,
    )


@pytest.fixture
def arret(monkeypatch, tmp_path):
    """The run workdir, SIGTERM intercepted, experiment lock armed; returns the marker written."""
    signaux: list = []
    monkeypatch.setattr(sc, "settings", SimpleNamespace(workdir=tmp_path))
    monkeypatch.setattr(sc.os, "kill", lambda pid, sig: signaux.append(sig))
    monkeypatch.setattr(sc, "_arret_sur_repli_arme", lambda: True)

    def _marqueur() -> dict:
        assert signaux, "the controller did not stop"
        return json.loads(
            (tmp_path / "en_attente_quota.json").read_text(encoding="utf-8")
        )

    return _marqueur


def _hiberner(boucle, **kw):
    asyncio.run(
        sc.SimulationLoopV1._declencher_hibernation_propre(
            boucle, None, "1127260", **kw
        )
    )


def test_le_409_de_la_nuit_s_ecrit_defaut(arret):
    _hiberner(
        _boucle(),
        motif="prefixe_commun",
        genre="rejeu_obligatoire_absent",
        details={"cause": "rejeu strict itinary_multi_agent refusé (HTTP 409)"},
    )
    m = arret()
    assert (m["motif"], m["nature"], m["cause"]) == (
        "prefixe_commun",
        DEFAUT,
        "rejeu_obligatoire_absent",
    )
    assert m["timestamp"] == T0 and m["person_id"] == "1127260"


def test_une_surcharge_s_ecrit_passager(arret):
    _hiberner(_boucle(), motif="surcharge_fournisseur", genre="surcharge_fournisseur")
    assert arret()["nature"] == PASSAGER


def test_une_decision_absente_sans_genre_s_ecrit_defaut(arret):
    _hiberner(_boucle(), motif="decision_absente")
    m = arret()
    assert (m["nature"], m["cause"]) == (DEFAUT, None)


def test_l_arret_dit_sa_nature_en_alarme(arret):
    from loguru import logger

    lignes: list[str] = []
    puits = logger.add(
        lambda msg: lignes.append(msg.record["message"]), level="WARNING"
    )
    try:
        _hiberner(_boucle(), motif="decision_absente")
    finally:
        logger.remove(puits)
    arret()
    assert any(
        "[hibernation] stop of nature « defaut »" in l and "no new attempt" in l
        for l in lignes
    )
    assert sum("[ALARME] [hibernation]" in l for l in lignes) == 1, (
        "a single alarm per stop"
    )


# ── Consolidation: the gateway's kind travels through the exception ──────────────────────


def _consolider(boucle, erreur: Exception):
    async def _echoue():
        raise erreur

    asyncio.run(
        sc.SimulationLoopV1._consolidation_ou_arret(
            boucle, _echoue(), "1320713", "stm_reflection"
        )
    )


def test_la_consolidation_du_29_09_14h37_est_passagere(arret):
    """504 on both keys: the gateway returns `surcharge_fournisseur`, the exception carries it."""
    boucle = _boucle()
    boucle._declencher_hibernation_propre = lambda *a, **k: (
        sc.SimulationLoopV1._declencher_hibernation_propre(boucle, *a, **k)
    )
    _consolider(
        boucle,
        ConsolidationMemoryUnavailable(
            "STM without a result for 1320713", genre="surcharge_fournisseur"
        ),
    )
    m = arret()
    assert (m["motif"], m["nature"], m["cause"]) == (
        "consolidation_memoire",
        PASSAGER,
        "surcharge_fournisseur",
    )


def test_une_exception_du_code_en_consolidation_est_un_defaut(arret):
    boucle = _boucle()
    boucle._declencher_hibernation_propre = lambda *a, **k: (
        sc.SimulationLoopV1._declencher_hibernation_propre(boucle, *a, **k)
    )
    _consolider(boucle, KeyError("concepts"))
    m = arret()
    assert (m["motif"], m["nature"]) == ("consolidation_memoire", DEFAUT)


def test_un_rejeu_refuse_en_consolidation_est_un_prefixe_casse(arret):
    client = SimpleNamespace(
        strict_replay_failure="rejeu strict stm_reflection refusé (HTTP 409)"
    )
    boucle = _boucle(client)
    boucle._declencher_hibernation_propre = lambda *a, **k: (
        sc.SimulationLoopV1._declencher_hibernation_propre(boucle, *a, **k)
    )
    _consolider(boucle, RuntimeError("Client error '409 Conflict'"))
    m = arret()
    assert (m["motif"], m["nature"], m["cause"]) == (
        "prefixe_commun",
        DEFAUT,
        "rejeu_obligatoire_absent",
    )


def test_l_exception_de_consolidation_porte_le_genre_du_dernier_essai():
    """The real STM path: three attempts without a result, the last one qualified as an overload."""
    import inspect

    from urban_mobility_agents.agents import llm_agent

    source = inspect.getsource(llm_agent)
    assert source.count('genre=getattr(llm_result, "error_kind", None)') == 2, (
        "STM et LTM"
    )


# ── Survey: the kind of the missed question goes up to the stop ────────────────────


class _Client:
    """Answers the questions, except those it is told to miss, with a given kind."""

    def __init__(self, rate: dict[tuple, str | None | Exception]):
        self.rate = rate

    async def execute(self, payload):
        a = payload["agents"][0]
        cle = (a["agent_id"], a.get("mode_interroge"))
        if cle in self.rate:
            genre = self.rate[cle]
            if isinstance(genre, Exception):
                raise genre
            return SimpleNamespace(
                agents=[], provider_used="google_gemini31_key1", error_kind=genre
            )
        return SimpleNamespace(
            agents=[
                SimpleNamespace(
                    agent_id=a["agent_id"],
                    scores=dict.fromkeys(enquetes.CRITERES, 5),
                    justification="ok",
                )
            ],
            provider_used="google_gemini31_key1",
        )


@pytest.fixture
def jalon(monkeypatch, tmp_path):
    from urban_mobility_agents.utils import reprise

    enquetes.reinitialiser()
    reprise.reinitialiser()
    monkeypatch.setenv("EXPERIMENT_TARGET_PERSONAS", "11,12")
    monkeypatch.setenv("EXPERIMENT_SURVEY_RETRIES_S", "0")
    monkeypatch.delenv("EXPERIMENT_SURVEY_MODES", raising=False)
    monkeypatch.delenv("EXPERIMENT_SURVEY_PANNE", raising=False)

    def _jouer(client) -> list[tuple]:
        vus: list[tuple] = []

        async def _saturation(jour, pid, genre=None):
            vus.append((jour, pid, genre))

        agent = SimpleNamespace(
            llm_client=client,
            long_term_memory=SimpleNamespace(
                user_metadata={}, journal_trajets=lambda _p: {}
            ),
            get_person_identity_description=lambda p: f"Persona {p.person_id}.",
        )
        personnes = [
            SimpleNamespace(person_id=p, identity=SimpleNamespace(name=p))
            for p in ("11", "12")
        ]
        asyncio.run(
            enquetes.executer_enquetes_jalon(
                11, T0, personnes, agent, tmp_path, en_cas_de_saturation=_saturation
            )
        )
        return vus

    yield _jouer
    enquetes.reinitialiser()
    reprise.reinitialiser()


def test_une_question_manquee_par_surcharge_arrete_en_passager(jalon):
    vus = jalon(_Client({("12", "walking"): "surcharge_fournisseur"}))
    assert vus == [(11, "12", "surcharge_fournisseur")]
    assert nature_arret("enquete_incomplete", vus[0][2]) == PASSAGER


def test_une_question_manquee_sans_genre_arrete_en_defaut(jalon):
    vus = jalon(_Client({("12", "walking"): None}))
    assert nature_arret("enquete_incomplete", vus[0][2]) == DEFAUT


def test_une_exception_d_appel_en_enquete_arrete_en_defaut(jalon):
    vus = jalon(
        _Client({("12", "walking"): RuntimeError("Client error '400 Bad Request'")})
    )
    assert nature_arret("enquete_incomplete", vus[0][2]) == DEFAUT


def test_une_seule_question_hors_surcharge_suffit_a_faire_un_defaut(jalon):
    vus = jalon(
        _Client(
            {("11", "walking"): "surcharge_fournisseur", ("12", "the bicycle"): None}
        )
    )
    assert len(vus) == 1
    assert nature_arret("enquete_incomplete", vus[0][2]) == DEFAUT


def test_la_panne_simulee_du_banc_reste_passagere(jalon, monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_SURVEY_PANNE", "11:600")
    vus = jalon(_Client({}))
    assert vus and nature_arret("enquete_incomplete", vus[0][2]) == PASSAGER


def test_le_controleur_transmet_le_genre_de_l_enquete_a_l_arret():
    """Wiring: the controller callback accepts the kind and passes it to the stop."""
    corps = (
        Path(sc.__file__)
        .read_text(encoding="utf-8")
        .split("async def _saturation_enquete(")[1][:1500]
    )
    assert corps.startswith("jour: int, person_id: str, genre: str | None = None)")
    assert "genre=genre," in corps


def test_le_chemin_de_decision_reconnait_le_rejeu_refuse_avant_decision_absente():
    """Wiring of the 409 on the decision path (the one of the night of 29/09)."""
    source = Path(sc.__file__).read_text(encoding="utf-8")
    bloc = source.split('selection_method = "LLM"')[1].split("plan: TravelPlan")[0]
    assert bloc.index('"strict_replay_failure"') < bloc.index('motif="prefixe_commun"')
    assert bloc.index('motif="prefixe_commun"') < bloc.index(
        'motif="decision_absente", genre=_genre'
    )
    assert 'genre="rejeu_obligatoire_absent"' in bloc
