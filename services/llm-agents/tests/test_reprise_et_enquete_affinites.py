"""Ticket 118 — the defects of the night of 28 to 29/09 (a13 v5 control).

O1 — During the frozen-memory replay, the controller still wrote its daily resume
points: a « jour_001 » carrying the memory of day 11 and an empty household state (28/09
15:23) overwrote the good point of the first pass. The restored point must remain the last valid one.

O2 — An affinity survey no longer loses any answer: a milestone is complete or is not
written, and an unanswered question cleanly stops the simulation to resume later.
"""

import asyncio
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urban_mobility_agents import enquetes
from urban_mobility_agents import simulation_controller as sc
from urban_mobility_agents.utils import mesures_jour, reprise

T0 = 1773637200  # 16 mars 2026, 05:00 en heure murale de simulation


class _PopulationDouble:
    def get_people_list(self):
        return []


def _boucle(timestamp: int = T0) -> SimpleNamespace:
    return SimpleNamespace(
        agent=None,
        population=_PopulationDouble(),
        _current_sim_timestamp=timestamp,
        _replis_consecutifs=0,
        _hibernation_declenchee=False,
        _hibernations_ignorees=0,
    )


@pytest.fixture
def points(monkeypatch, tmp_path):
    """The calls to `ecrire_point` and to the measures, without writing anything."""
    vus = {"points": [], "mesures": 0}

    def _point(_workdir, jour, *_a, **_k):
        vus["points"].append(jour)

    def _mesures(_workdir):
        vus["mesures"] += 1
        return 1

    monkeypatch.setattr(sc, "ecrire_point", _point)
    monkeypatch.setattr(sc, "settings", SimpleNamespace(workdir=tmp_path))
    monkeypatch.setattr(mesures_jour, "actif", lambda: True)
    monkeypatch.setattr(mesures_jour, "ecrire_mesures_du_jour", _mesures)
    reprise.reinitialiser()
    yield vus
    reprise.reinitialiser()


# ══════════════════════════ O1 — aucun point pendant le gel ══════════════════════════


def test_O1_aucun_point_de_reprise_pendant_le_rejeu_gele(points):
    """Counter-check: without the guard, the point of day 1 is written over the real one."""
    reprise.geler_jusqu_a(T0 + 10 * 86400, 11)
    asyncio.run(sc.SimulationLoopV1._ecrire_point_de_reprise(_boucle(), T0))
    assert points["points"] == [], "un point écrit pendant le gel écrase le point restauré"
    assert points["mesures"] == 1, "les mesures du jour, elles, continuent pendant le rejeu"


def test_O1_le_point_reprend_des_le_degel(points):
    reprise.geler_jusqu_a(T0 + 86400, 2)
    assert reprise.degeler_si_depasse(T0 + 86400)
    asyncio.run(sc.SimulationLoopV1._ecrire_point_de_reprise(_boucle(), T0 + 2 * 86400))
    assert len(points["points"]) == 1, "le dégel rend au contrôleur ses points quotidiens"


def test_O1_hors_reprise_le_point_secrit_comme_avant(points):
    asyncio.run(sc.SimulationLoopV1._ecrire_point_de_reprise(_boucle(), T0))
    assert len(points["points"]) == 1


# ══════════════════════════ O2 — aucune réponse d'enquête perdue ═════════════════════
import json  # noqa: E402

from loguru import logger  # noqa: E402


class _Client:
    """Returns six scores, except for the questions it is told to fail (a number of times, or always)."""

    def __init__(self, rate=None):
        self.rate = dict(rate or {})  # (pid, mode_interroge) → nombre d'échecs à produire
        self.appels: list[tuple[str, str | None, str]] = []

    async def execute(self, payload):
        a = payload["agents"][0]
        cle = (a["agent_id"], a.get("mode_interroge"))
        self.appels.append((*cle, a["perception"]))
        if self.rate.get(cle, 0):
            self.rate[cle] -= 1
            return SimpleNamespace(agents=[], provider_used="google_gemini31_key1")
        return SimpleNamespace(
            agents=[SimpleNamespace(agent_id=a["agent_id"], scores=dict.fromkeys(enquetes.CRITERES, 5),
                                    justification="ok")],
            provider_used="google_gemini31_key1",
        )


class _Memoire:
    user_metadata: dict = {}

    def journal_trajets(self, _pid):
        return {}


def _agent(client):
    return SimpleNamespace(llm_client=client, long_term_memory=_Memoire(),
                           get_person_identity_description=lambda p: f"Persona {p.person_id}.")


def _personnes(*pids):
    return [SimpleNamespace(person_id=p, identity=SimpleNamespace(name=p)) for p in pids]


def _csv(dossier: Path) -> list[dict]:
    f = dossier / "affinites_declarees.csv"
    return list(csv.DictReader(f.open(encoding="utf-8"))) if f.is_file() else []


@pytest.fixture
def enquete(monkeypatch):
    enquetes.reinitialiser()
    reprise.reinitialiser()
    monkeypatch.setenv("EXPERIMENT_TARGET_PERSONAS", "11,12")
    monkeypatch.setenv("EXPERIMENT_SURVEY_RETRIES_S", "0,0,0")
    monkeypatch.delenv("EXPERIMENT_SURVEY_MODES", raising=False)
    yield
    enquetes.reinitialiser()
    reprise.reinitialiser()


@pytest.fixture
def journal():
    lignes: list[tuple[str, str]] = []
    puits = logger.add(lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO")
    yield lignes
    logger.remove(puits)


def _jalon(client, dossier, saturation=None, jour=11):
    asyncio.run(enquetes.executer_enquetes_jalon(
        jour, T0, _personnes("11", "12"), _agent(client), dossier, en_cas_de_saturation=saturation
    ))


def test_O2_une_question_qui_echoue_puis_repond_ne_perd_rien(enquete, tmp_path):
    """Three failures then an answer: the retries absorb the overload (D11 of 28/09)."""
    client = _Client(rate={("12", "walking"): 3})
    _jalon(client, tmp_path)
    lignes = _csv(tmp_path)
    assert len(lignes) == 2 * 5 * 6, "2 personas × 5 questions × 6 critères, rien de perdu"
    assert not list(tmp_path.glob("enquete_en_attente_J*.json"))


def test_O2_un_jalon_incomplet_nest_pas_ecrit_et_arrete_le_run(enquete, tmp_path, journal):
    """Counter-check: the old code wrote the 54 other lines and lost the question for good."""
    arrets: list[tuple[int, str]] = []

    async def _saturation(jour, pid, genre=None):
        arrets.append((jour, pid))

    _jalon(_Client(rate={("12", "walking"): 99}), tmp_path, _saturation)
    assert _csv(tmp_path) == [], "un jalon incomplet n'entre pas dans le CSV"
    attente = json.loads((tmp_path / "enquete_en_attente_J011.json").read_text("utf-8"))
    assert attente["manquantes"] == [["12", "marche"]]
    assert len(attente["lignes"]) == 9 * 6, "les réponses obtenues sont gardées, pas repayées"
    assert arrets == [(11, "12")], "le contrôleur est prévenu une fois"
    assert [m for n, m in journal if n == "ERROR" and "INCOMPLET : 1/10" in m]


def test_O2_la_question_se_repose_sur_la_meme_perception_et_le_jalon_se_complete(
    enquete, tmp_path, journal
):
    _jalon(_Client(rate={("12", "walking"): 99}), tmp_path)
    perception_du_soir = json.loads(
        (tmp_path / "enquete_en_attente_J011.json").read_text("utf-8")
    )["perceptions"]["12"]
    # The resume: another client, another time — the memory may have changed, not the question.
    client = _Client()
    ecrit = asyncio.run(enquetes.completer_en_attente(_agent(client), tmp_path))
    assert ecrit is True
    assert [(p, m) for p, m, _ in client.appels] == [("12", "walking")], "seule la manquante"
    assert client.appels[0][2] == perception_du_soir
    lignes = _csv(tmp_path)
    assert len(lignes) == 60
    assert [l["persona_id"] for l in lignes[:30]] == ["11"] * 30, "l'ordre du CSV est conservé"
    assert {l["jour_simule"] for l in lignes} == {"11"}
    assert not list(tmp_path.glob("enquete_en_attente_J*.json"))
    assert [m for n, m in journal if "jalon J11 COMPLET" in m and "1 reposée(s)" in m]


def test_O2_un_jalon_en_attente_nest_pas_juge_perdu_pendant_le_rejeu(enquete, tmp_path, journal):
    _jalon(_Client(rate={("12", "walking"): 99}), tmp_path)
    reprise.geler_jusqu_a(T0 + 3600, 11)
    assert enquetes.deja_mene_avant_reprise(11, T0, tmp_path / "affinites_declarees.csv")
    assert not [m for n, m in journal if "impossible à reposer" in m]


def test_O2_le_controleur_passe_la_saturation_et_complete_les_jalons_en_attente():
    source = Path(sc.__file__).read_text("utf-8")
    assert "en_cas_de_saturation=_saturation_enquete" in source
    assert "enquetes_module.completer_en_attente(" in source
    assert 'motif="enquete_incomplete"' in source


# ══════════════════════════ O2 — the pause is visible and lasts one hour ══════════════════


RACINE_DEPOT = Path(__file__).resolve().parents[3]


def test_O2_la_chaine_de_nuit_attend_une_heure_et_declare_sa_pause():
    script = (RACINE_DEPOT / "scripts/experiment/enchainer_experiences_memoire.sh").read_text("utf-8")
    assert "ATTENTE_S=${ATTENTE_S:-3600}" in script
    assert ".pause.json" in script and 'rm -f "$PAUSE"' in script
    make = (RACINE_DEPOT / "make/memoire.mk").read_text("utf-8")
    assert "$(or $(ATTENTE_S),3600)" in make, "la cible make ne doit pas ramener l'attente à 30 min"


def test_O2_linterface_lit_la_pause(tmp_path):
    from experiences import memoire

    journal = tmp_path / "enchainement_nuit_20260929_2200.log"
    journal.write_text("x\n", encoding="utf-8")
    journal.with_suffix(".pause.json").write_text(json.dumps({
        "experience": "exp_a13", "motif": "enquete_incomplete", "jour_simule": "11",
        "prochain_essai": "2026-09-29T23:10:00+02:00", "essai": 2, "essais_max": 6,
    }), encoding="utf-8")
    pause = memoire._pause_de(journal)
    assert pause["libelle_motif"] == "enquête incomplète, fournisseur saturé"
    assert pause["prochain_essai"][11:16] == "23:10"


def test_O2_un_jalon_complete_a_lheure_du_point_nest_pas_repose_au_degel(enquete, tmp_path, journal):
    """Bench of 29/09: stop during the 21:00 survey, point written at 21:00, milestone completed
    during the replay, thaw at 21:00 — the evening milestone was asked a second time (180 duplicates).
    Counter-check: with the old guard (strictly before the point), the second pass
    questioned both personas again."""
    _jalon(_Client(rate={("12", "walking"): 99}), tmp_path)
    asyncio.run(enquetes.completer_en_attente(_agent(_Client()), tmp_path))
    avant = _csv(tmp_path)
    assert len(avant) == 60
    reprise.geler_jusqu_a(T0, 11)  # the resume point is the very time of the milestone
    reprise.degeler_si_depasse(T0)
    client = _Client()
    _jalon(client, tmp_path)
    assert client.appels == [], "le jalon déjà écrit ne se repose pas"
    assert _csv(tmp_path) == avant
    assert [m for n, m in journal if "jalon J11" in m and "déjà mené" in m]
