"""Ticket 111 — reading precedes the decision, and it is passed on to the household.

Contract: `specs/ticket_111/tests.md`, T1 to T12. T13 (post-run check) lives in
`scripts/tests/test_111_controle_lecture.py`, T14 (category) in `packages/mobility_llm`.

No test calls a real model: the LLM client is a fake that returns a response written here.

Common calendar: the run is anchored on Monday 16 March 2026, 5 a.m. Household H1 reads on day 4
of the run, Thursday 19 March. Its five service days are thus Thursday 19, Friday 20, Monday 23,
Tuesday 24 and Wednesday 25 — weekend excluded.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import evenements as ev
from llm import foyer as foyer_module
from llm import noyau as noyau_module
from llm.evenements import RefusDEvenement, RegistreEvenements, charger
from llm.evenements import calendrier
from llm.evenements import relais as relais_module
from llm.evenements.injection import PREFIXE_FOYER, PREFIXE_LU, ligne_de_lecture
from llm.memory import MemoryEntry, MemoryType
from llm.noyau import memoire_noyau
from models import Activity, Person, PersonalIdentity
from settings import settings
from sim_clock import gama_timestamp, wall_clock
from urban_mobility_agents.utils import ancre_run

RACINE = Path(__file__).resolve().parents[1]
T0 = 1773637200  # Monday 16 March 2026, 05:00 wall-clock

ARTICLE = (
    "Toulouse: gusts above 80 km/h expected this Thursday evening. The city council is closing "
    "its parks and gardens from 6 p.m., and they will reopen once the warning is lifted."
)

LECTEUR, CONJOINT, ENFANT, SILENCIEUX = "101", "102", "103", "104"


def ts(jour: int, heure: int = 5, minute: int = 57) -> int:
    """GAMA timestamp of day `jour` of the run (1 = Monday 16 March), at the given hour."""
    d = wall_clock(T0).date() + timedelta(days=jour - 1)
    return gama_timestamp(datetime(d.year, d.month, d.day, heure, minute))


# ── Population ───────────────────────────────────────────────────────────────────────────────
def _personne(pid: str, nom: str, age: int | None, foyer: str = "H1", occupation="Full-Time Worker"):
    traits = {"name": nom, "professional_activity": occupation, "household_size": 4}
    if age is not None:
        traits["age"] = age
    return Person(
        person_id=pid,
        household_id=foyer,
        identity=PersonalIdentity(
            name=nom,
            traits_json=traits,
            activities=[
                Activity(id="a0", start_time=0, end_time=27000, purpose="home"),
                Activity(id="a1", start_time=28800, end_time=61200, purpose="work"),
                Activity(id="a2", start_time=62000, end_time=86000, purpose="home"),
            ],
        ),
    )


def _population() -> list:
    # Two adults, so as to have an informed adult; the reader is the one drawn by the seed, and
    # the fixture reads it back rather than assuming it.
    return [
        _personne(LECTEUR, "Arthur Martin", 40),
        _personne(CONJOINT, "Claire Martin", 38),
        _personne(ENFANT, "Josette Martin", 9, occupation="Pupil"),
        _personne(SILENCIEUX, "Paul Martin", 6, occupation="Pupil"),
        _personne("201", "Témoin Durand", 45, foyer="H2"),
    ]


# ── Declaration ──────────────────────────────────────────────────────────────────────────────
def _texte(tmp_path: Path) -> tuple[Path, str]:
    dossier = tmp_path / "articles_txt" / "a99_test"
    dossier.mkdir(parents=True, exist_ok=True)
    f = dossier / "brut.txt"
    f.write_text(ARTICLE, encoding="utf-8")
    return f, hashlib.sha256(f.read_bytes()).hexdigest()


def _declaration(tmp_path: Path, **surcharges) -> dict:
    f, sha = _texte(tmp_path)
    base = {
        "evenement": "a99_test",
        "libelle": "Article de test",
        "source": "test",
        "canal": "lu",
        "moment": "reveil",
        "jugement": "a_l_injection",
        "texte": {"fichier": str(f), "sha256": sha},
        "calendrier": {"jours": [4]},
        "exposition": {"regle": "foyers", "foyers": ["H1"], "lecteurs_par_foyer": 1, "graine": 59},
        "cadence": "jour",
        "service": {"jours_de_deplacement": 5},
        "relais": {"mode": "par_destinataire"},
    }
    base.update(surcharges)
    return base


def _charger(tmp_path: Path, **surcharges):
    p = tmp_path / "evenement.yaml"
    p.write_text(yaml.safe_dump(_declaration(tmp_path, **surcharges), allow_unicode=True), "utf-8")
    return charger(p)


# ── The fake model ───────────────────────────────────────────────────────────────────────────
class FauxClient:
    """Returns a relay and judgements written here. Counts the calls per category."""

    def __init__(self, relais=None, severite="noticeable", refuser_jugement_de=()):
        self.appels: list[dict] = []
        self._relais = relais
        self._severite = severite
        self._refus = set(refuser_jugement_de)

    def par_categorie(self, categorie: str) -> list[dict]:
        return [a for a in self.appels if a["category"] == categorie]

    async def execute(self, payload):
        self.appels.append(payload)
        agent = payload["agents"][0]
        if payload["category"] == "evenement_relais":
            await asyncio.sleep(0.01)  # lets the concurrent requests meet
            reponse = self._relais(agent) if callable(self._relais) else self._relais
            return reponse
        if payload["category"] == "evenement_jugement":
            if agent["agent_id"] in self._refus:
                return SimpleNamespace(agents=[], error="refus de test")
            return SimpleNamespace(
                agents=[SimpleNamespace(
                    agent_id=agent["agent_id"], severity=self._severite,
                    valence="negative", modes=["walking"],
                )],
                provider_used="faux",
            )
        raise AssertionError(f"unexpected category {payload['category']}")


def relais_standard(lecteur_id=LECTEUR, conjoint=CONJOINT):
    """Informed adult, informed minor, one member the reader tells nothing."""
    return SimpleNamespace(
        agents=[SimpleNamespace(agent_id=lecteur_id, recipients=[
            {"agent_id": conjoint, "speaks": True,
             "message": "The parks close at six tonight because of the wind."},
            {"agent_id": ENFANT, "speaks": True,
             "message": "We decided you'll go to school by bus today, not through the park."},
            {"agent_id": SILENCIEUX, "speaks": False, "message": ""},
        ])],
        provider_used="faux",
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _propre(monkeypatch):
    ev.reinitialiser()
    foyer_module.reinitialiser()
    noyau_module.reinitialiser()
    ancre_run.reinitialiser()
    ancre_run.ancrer(T0, origine="test 111")
    monkeypatch.setattr(settings.agent, "no_weekend_departures", True)
    yield
    monkeypatch.setattr(ev, "_registre", None)
    ev.reinitialiser()
    foyer_module.reinitialiser()
    ancre_run.reinitialiser()


@pytest.fixture
def journal_logs():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(lambda m: lignes.append((m.record["level"].name, m.record["message"])),
                      level="INFO")
    yield lignes
    logger.remove(sink)


@pytest.fixture
def dispositif(tmp_path, monkeypatch):
    """Armed registry, indexed population, readers drawn AT LOAD TIME (like the controller)."""
    e = _charger(tmp_path)
    reg = RegistreEvenements(e, journal=tmp_path / "evenements.jsonl")
    monkeypatch.setattr(ev, "_registre", reg)
    pop = _population()
    foyer_module.initialiser(pop)
    lecteurs = reg.lecteurs(pop)
    assert list(lecteurs) in ([LECTEUR], [CONJOINT])
    return SimpleNamespace(registre=reg, population=pop, lecteur=next(iter(lecteurs)),
                           dossier=tmp_path)


def _autre_adulte(lecteur: str) -> str:
    return CONJOINT if lecteur == LECTEUR else LECTEUR


def _brancher(d, client) -> None:
    """The producer as the controller plugs it in, on a minimal loop."""
    boucle = _boucle(d, client)
    d.registre.brancher_producteur_relais(boucle._produire_relais)
    d.boucle = boucle


def _boucle(d, client):
    from urban_mobility_agents.simulation_controller import SimulationLoopV1

    boucle = SimulationLoopV1.__new__(SimulationLoopV1)
    people = {p.person_id: p for p in d.population}
    boucle.model = SimpleNamespace(population=SimpleNamespace(
        people=people, get_people_list=lambda: list(people.values()),
    ))
    boucle.agent = FauxAgentMemoire(client)
    boucle._spawn = lambda coro: asyncio.ensure_future(coro)
    return boucle


class FauxAgentMemoire:
    """What the injection touches in `LlmAgent`: identity, short-term and long-term memory."""

    def __init__(self, client):
        self.llm_client = client
        self.long_term_memory = None
        self.courte: list[dict] = []
        self.longue: list[MemoryEntry] = []

    def get_person_identity_description(self, person):
        t = person.identity.traits_json
        return f"{t['name'].split()[0]}, {t.get('age')}, {t['professional_activity']}"

    def add_short_term_memory(self, context, msg, timestamp=None, importance=0.0, axes=None,
                              valence="neutre", origine=None):
        self.courte.append({"pid": context.person.person_id, "msg": msg,
                            "importance": importance, "origine": origine})

    async def aadd_long_term_memory(self, context, msg):
        await asyncio.sleep(0)
        self.longue.append(msg)


def lignes(pid: str, t: int, **kw) -> list[str]:
    return asyncio.run(ev.lignes_du_jour(pid, t, **kw))


# ═════════════ T1 — the first decision of the reading day carries the article ═════════════
class _Rappel:
    """Long-term memory in RAM, simulated vector recall: three "everything went fine" reviews."""

    def __init__(self, pid: str, entrees: list[MemoryEntry]):
        self.user_metadata = {pid: {"entries": entrees}}
        self._pid = pid

    def journal_trajets(self, pid):
        return {}

    def has_memories(self, pid):
        return True

    async def aquery_user_memories(self, **kw):
        return [
            SimpleNamespace(
                content=f"Today my trip by car went smoothly (#{i}).",
                metadata={
                    "timestamp": (wall_clock(ts(3)) - timedelta(days=i)).isoformat(),
                    "memory_type": MemoryType.REFLECTION.value,
                },
            )
            for i in range(3)
        ]


class _Option:
    purpose = "work"
    end_location = None
    distance = 1000

    def __init__(self, code="car"):
        self._code = code

    def get_code(self):
        return self._code

    def mode_label(self):
        return self._code

    def model_dump(self):
        return {"code": self._code}


def _vrai_agent(monkeypatch, pid: str, entrees: list[MemoryEntry]):
    """The REAL memory builder of `LlmAgent`, on a simulated long-term memory."""
    from urban_mobility_agents.agents import llm_agent as la
    from urban_mobility_agents.agents.prompt_manager import PromptManager

    monkeypatch.setattr(la, "env_ob_to_text", lambda *a, **k: "option")
    monkeypatch.setattr(la, "get_weather", lambda *a, **k: None)
    agent = la.LlmAgent.__new__(la.LlmAgent)
    agent.prompt_manager = PromptManager(os.path.join(os.path.dirname(la.__file__), "prompts"))
    agent.long_term_memory = _Rappel(pid, entrees)
    agent.llm_cache = None
    return agent


def _contexte(personne, t: int):
    from urban_mobility_agents.agents.llm_agent import Context

    return Context(person=personne, activity_id="a1", timestamp=t, data={})



def test_T1_la_decision_calculee_la_veille_porte_l_article(dispositif, monkeypatch, journal_logs):
    d = dispositif
    lecteur = next(p for p in d.population if p.person_id == d.lecteur)
    # The decision of Thursday 19 at 05:57 is built on Wednesday 18 at 07:00 — before the injection.
    d.registre.noter_instant(ts(3, 7, 0))
    servies = lignes(d.lecteur, ts(4))
    assert servies == [ligne_de_lecture(ARTICLE)]

    agent = _vrai_agent(monkeypatch, d.lecteur, entrees=[])
    history = asyncio.run(
        agent.query_past_experiences_for_travel(_contexte(lecteur, ts(4)), [_Option()], servies)
    )
    rendu = "\n".join(history)
    assert f"{PREFIXE_LU} I read in the paper:" in rendu
    # The log says the decision was computed before the injection, and by how much.
    assert any("calculée 17 h avant l'injection" in m for _, m in journal_logs)
    assert d.registre._compteurs.servies_avant_injection == 1


def test_T1_le_premier_service_consigne_l_instant_du_prompt(dispositif):
    """2026-09-28 — the control's bound is read here: the instant of the /sync where the line
    entered a prompt, not the injection timestamp (a13 v5: 16 h 30 earlier)."""
    from experiences import rejeu_ab
    from llm.evenements.registre import FICHIER_PREMIERS_SERVICES

    d = dispositif
    d.registre.noter_instant(ts(3, 7, 0))
    lignes(d.lecteur, ts(4))
    lignes(d.lecteur, ts(4, 18, 0))  # same service day: no second line
    f = d.dossier / FICHIER_PREMIERS_SERVICES
    consignes = [json.loads(ligne) for ligne in f.read_text(encoding="utf-8").splitlines()]
    assert len(consignes) == 1
    c = consignes[0]
    assert (c["person_id"], c["genre"], c["instant_source"]) == (d.lecteur, "lectures_servies", "sync")
    assert c["instant_ts"] == ts(3, 7, 0) and c["decision_ts"] == ts(4)
    assert c["instant_simule"] == wall_clock(ts(3, 7, 0)).isoformat()
    assert c["avant_injection"] is True
    assert FICHIER_PREMIERS_SERVICES == rejeu_ab.FICHIER_PREMIERS_SERVICES
    assert rejeu_ab.instant_debut_traitement(d.dossier / "evenements.jsonl") == (
        c["instant_simule"], "service"
    )


@pytest.mark.parametrize("gravite", [0.30, 0.80])
def test_T1_apres_l_injection_la_ligne_n_apparait_qu_une_fois(dispositif, monkeypatch, gravite):
    """Judged below the shock threshold (0.30) or above it (0.80): exactly once."""
    d = dispositif
    lecteur = next(p for p in d.population if p.person_id == d.lecteur)
    entree = MemoryEntry(
        content=ligne_de_lecture(ARTICLE),
        timestamp=wall_clock(ts(4, 0, 0)),
        memory_type=MemoryType.CONVERSATION,
        person_id=d.lecteur,
        importance=gravite,
        origine="lu",
    )
    servies = lignes(d.lecteur, ts(5))
    agent = _vrai_agent(monkeypatch, d.lecteur, entrees=[entree])
    history = asyncio.run(
        agent.query_past_experiences_for_travel(_contexte(lecteur, ts(5)), [_Option()], servies)
    )
    assert "\n".join(history).count("I read in the paper:") == 1


# ═════════════ T2 — five trip days, weekend excluded ══════════════════════════════════════
def test_T2_cinq_jours_ouvres_depuis_un_jeudi(dispositif):
    d = dispositif
    attendus = {4: True, 5: True, 6: False, 7: False, 8: True, 9: True, 10: True, 11: False}
    for jour, servi in attendus.items():
        assert bool(lignes(d.lecteur, ts(jour))) is servi, f"day {jour}"


def test_T2_sans_regle_de_week_end_cinq_jours_calendaires(dispositif, monkeypatch):
    monkeypatch.setattr(settings.agent, "no_weekend_departures", False)
    d = dispositif
    for jour in range(4, 9):
        assert lignes(d.lecteur, ts(jour)), f"day {jour}"
    assert not lignes(d.lecteur, ts(9))


def test_T2_calendrier_pur():
    from datetime import date

    jeudi = date(2026, 3, 19)
    assert calendrier.jours_de_service(jeudi, 5, True) == (
        date(2026, 3, 19), date(2026, 3, 20), date(2026, 3, 23), date(2026, 3, 24),
        date(2026, 3, 25),
    )
    samedi = date(2026, 3, 21)
    assert calendrier.jours_de_service(samedi, 1, True) == (date(2026, 3, 23),)


# ═════════════ T3 — the relay, per recipient ══════════════════════════════════════════════
def test_T3_chaque_membre_voit_ce_qui_lui_a_ete_dit(dispositif):
    d = dispositif
    adulte = _autre_adulte(d.lecteur)
    client = FauxClient(relais=relais_standard(d.lecteur, adulte))
    _brancher(d, client)
    prenom = "Arthur" if d.lecteur == LECTEUR else "Claire"

    assert lignes(adulte, ts(4)) == [
        f"{PREFIXE_FOYER} {prenom} told me: "
        f"« The parks close at six tonight because of the wind. »"
    ]
    assert lignes(ENFANT, ts(4)) == [
        f"{PREFIXE_FOYER} My parents decided: "
        f"« We decided you'll go to school by bus today, not through the park. »"
    ]
    assert lignes(SILENCIEUX, ts(4)) == []
    assert lignes("201", ts(4)) == []  # control household


def test_T3_trois_demandes_simultanees_un_seul_appel(dispositif):
    d = dispositif
    client = FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur)))
    _brancher(d, client)

    async def trois():
        return await asyncio.gather(
            ev.lignes_du_jour(_autre_adulte(d.lecteur), ts(4)),
            ev.lignes_du_jour(ENFANT, ts(4)),
            ev.lignes_du_jour(SILENCIEUX, ts(4)),
        )

    asyncio.run(trois())
    assert len(client.par_categorie("evenement_relais")) == 1


def test_T3_le_gabarit_recoit_mineur_pose_selon_l_age(dispositif):
    d = dispositif
    client = FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur)))
    _brancher(d, client)
    lignes(ENFANT, ts(4))
    membres = client.par_categorie("evenement_relais")[0]["agents"][0]["membres"]
    mineurs = {m["agent_id"]: m["mineur"] for m in membres}
    assert mineurs == {_autre_adulte(d.lecteur): False, ENFANT: True, SILENCIEUX: True}
    assert all(m["agent_id"] != d.lecteur for m in membres)
    assert membres[0]["trajets_du_jour"] == ["08:00 work", "17:13 home"]


# ═════════════ T4 — an invalid relay produces nothing ═════════════════════════════════════
def _reponse(recipients, agent_id=None):
    return lambda agent: SimpleNamespace(
        agents=[SimpleNamespace(agent_id=agent_id or agent["agent_id"], recipients=recipients)]
    )


@pytest.mark.parametrize("reponse,raison", [
    (lambda agent: SimpleNamespace(agents=[], error="vide"), "réponse vide"),
    (_reponse([], agent_id="999"), "agent_id"),
    (_reponse([{"agent_id": ENFANT, "speaks": False, "message": ""}]), "sans réponse"),
    ("parle_vide", "message vide"),
])
def test_T4_un_relais_invalide_ne_produit_rien(dispositif, journal_logs, reponse, raison):
    d = dispositif
    adulte = _autre_adulte(d.lecteur)
    if reponse == "parle_vide":
        reponse = _reponse([
            {"agent_id": adulte, "speaks": True, "message": "  "},
            {"agent_id": ENFANT, "speaks": False, "message": ""},
            {"agent_id": SILENCIEUX, "speaks": False, "message": ""},
        ])
    client = FauxClient(relais=reponse)
    _brancher(d, client)
    for pid in (adulte, ENFANT, SILENCIEUX):
        assert lignes(pid, ts(4)) == []
    alarmes = [m for n, m in journal_logs if n == "ERROR" and "[ALARME]" in m and "H1" in m]
    assert alarmes and raison in alarmes[0]
    assert "aucun texte de repli" in alarmes[0]
    # Refused once, never retried — and the trace keeps it for the resume.
    assert len(client.par_categorie("evenement_relais")) == 1
    trace = [json.loads(l) for l in (d.dossier / "relais_foyer.jsonl").read_text().splitlines()]
    assert trace[0]["refus"]


# ═════════════ T5 — the injection at 00:00 ════════════════════════════════════════════════
def _injecter(d, client, jour=4):
    _brancher(d, client)
    asyncio.run(d.boucle._injecter_evenements_du_reveil(ts(jour, 0, 0)))
    return d.boucle.agent


def test_T5_lecteur_et_informes_juges_chacun_avec_son_identite(dispositif):
    d = dispositif
    adulte = _autre_adulte(d.lecteur)
    client = FauxClient(relais=relais_standard(d.lecteur, adulte))
    agent = _injecter(d, client)

    juges = client.par_categorie("evenement_jugement")
    assert {j["agents"][0]["agent_id"] for j in juges} == {d.lecteur, adulte, ENFANT}
    for j in juges:
        pid = j["agents"][0]["agent_id"]
        personne = next(p for p in d.population if p.person_id == pid)
        assert j["agents"][0]["perception"].startswith(
            personne.identity.traits_json["name"].split()[0]
        )

    # Short-term AND long-term memory for each informed member, `entendu`, prefix [ FOYER ].
    for pid in (adulte, ENFANT):
        courtes = [c for c in agent.courte if c["pid"] == pid]
        longues = [m for m in agent.longue if m.person_id == pid]
        assert len(courtes) == 1 and len(longues) == 1
        assert courtes[0]["origine"] == "entendu" and longues[0].origine == "entendu"
        assert longues[0].content.startswith(PREFIXE_FOYER)
    # The long-term write is AWAITED: readable as soon as the injection returns.
    assert any(m.person_id == d.lecteur and m.origine == "lu" for m in agent.longue)
    assert not any(c["pid"] == SILENCIEUX for c in agent.courte)

    evs = [json.loads(l) for l in (d.dossier / "evenements.jsonl").read_text().splitlines()]
    assert [e["person_id"] for e in evs] == [d.lecteur, adulte, ENFANT]
    enfant = evs[2]
    assert enfant["raison_exposition"] == f"relais:{d.lecteur}"
    assert enfant["mineur"] is True and enfant["directif"] is True
    assert enfant["origine"] == "entendu" and enfant["message"].startswith("We decided")

    relais = [json.loads(l) for l in (d.dossier / "relais_foyer.jsonl").read_text().splitlines()]
    assert len(relais) == 1
    assert {m["destinataire_id"] for m in relais[0]["messages"]} == {adulte, ENFANT, SILENCIEUX}
    assert relais[0]["informes"] == [adulte, ENFANT]


def test_T5_injection_hors_jour_de_lecture_ne_fait_rien(dispositif):
    d = dispositif
    client = FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur)))
    agent = _injecter(d, client, jour=3)
    assert client.appels == [] and agent.courte == []


# ═════════════ T6 — what is heard is not passed on ═══════════════════════════════════════
def test_T6_un_concept_entendu_ne_traverse_jamais(dispositif, monkeypatch):
    d = dispositif
    monkeypatch.setattr(settings.agent, "memoire__partage_foyer_enabled", True)
    adulte = _autre_adulte(d.lecteur)
    concept = MemoryEntry(
        content=json.dumps(["The parks close when it is windy."]),
        timestamp=wall_clock(ts(4)),
        memory_type=MemoryType.CONCEPT,
        person_id=adulte,
        importance=0.3,
        origine="entendu",
        observations=3,
        axe_objet="walking",
    )
    ltm = SimpleNamespace(user_metadata={adulte: {"entries": [concept]}})
    receveur = next(p for p in d.population if p.person_id == ENFANT)
    montres, refus = foyer_module.croyances_partagees(ltm, receveur, wall_clock(ts(5)))
    assert montres == [] and refus["G3"] == 1


# ═════════════ T7 — exposure voided after a first service ═════════════════════════════════
def test_T7_non_avenue_apres_service_leve_une_alarme_unique(dispositif, journal_logs):
    d = dispositif
    client = FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur)),
                        refuser_jugement_de={d.lecteur})
    d.registre.noter_instant(ts(3, 7, 0))
    assert lignes(d.lecteur, ts(4))
    assert lignes(d.lecteur, ts(4, 17, 0))
    _injecter(d, client)
    alarmes = [m for n, m in journal_logs if n == "ERROR" and "NON AVENUE" in m]
    assert len(alarmes) == 1 and "2 décision(s)" in alarmes[0] and d.lecteur in alarmes[0]
    assert lignes(d.lecteur, ts(5)) == []
    # H7 — the transmission falls with the reading: the members no longer see anything.
    assert lignes(ENFANT, ts(5)) == []
    d.registre.declarer_non_avenue(d.lecteur, "deuxième fois")
    assert len([m for n, m in journal_logs if n == "ERROR" and "NON AVENUE" in m]) == 1


# ═════════════ T8 — the cache never serves a decision that must carry a line ════════════
class _Stop(Exception):
    pass


class _CacheEspion:
    def __init__(self):
        self.lookups = 0

    async def lookup(self, **kw):
        self.lookups += 1
        return None


def _decider(agent, personne, t):
    async def _payload(*a, **k):
        raise _Stop

    agent.build_travel_plan_payload = _payload
    with pytest.raises(_Stop):
        asyncio.run(agent.evaluate_and_choose_travel_plan(
            _contexte(personne, t), [_Option("car"), _Option("walk")], "work", departure_time=t,
        ))


def test_T8_jour_de_service_hors_fenetre_le_cache_n_est_pas_consulte(dispositif, monkeypatch):
    d = dispositif
    lecteur = next(p for p in d.population if p.person_id == d.lecteur)
    agent = _vrai_agent(monkeypatch, d.lecteur, entrees=[])
    agent.long_term_memory.has_memories = lambda pid: False
    agent.llm_cache = _CacheEspion()
    # Friday 20: service day, OUTSIDE the draw window (day 4 only).
    assert not ev.cache_coupe(ts(5))
    _decider(agent, lecteur, ts(5))
    assert agent.llm_cache.lookups == 0
    assert d.registre._compteurs.cache_contourne_ligne == 1
    # Counter-check: the control, for its part, consults the cache.
    temoin = next(p for p in d.population if p.person_id == "201")
    _decider(agent, temoin, ts(5))
    assert agent.llm_cache.lookups == 1


# ═════════════ T9 — the resume re-reads the relay ═════════════════════════════════════════
def test_T9_la_reprise_relit_sans_regenerer(dispositif, monkeypatch):
    d = dispositif
    client = FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur)))
    _brancher(d, client)
    avant = lignes(ENFANT, ts(4))
    assert len(client.appels) == 1

    reprise = RegistreEvenements(d.registre.evenement, journal=d.dossier / "evenements.jsonl")
    monkeypatch.setattr(ev, "_registre", reprise)
    reprise.lecteurs(d.population)
    client2 = FauxClient(relais=lambda a: pytest.fail("relay regenerated on resume"))
    _brancher(SimpleNamespace(registre=reprise, population=d.population), client2)
    assert lignes(ENFANT, ts(4)) == avant
    assert client2.appels == []


# ═════════════ T10 — nothing changes outside the setup ════════════════════════════════════
def _entrees_types():
    base = wall_clock(ts(5))
    return [
        MemoryEntry(content="choc grave", timestamp=base - timedelta(days=1),
                    memory_type=MemoryType.REFLECTION, person_id="1", importance=0.8),
        MemoryEntry(content=json.dumps(["Buses are late."]), timestamp=base - timedelta(days=2),
                    memory_type=MemoryType.CONCEPT, person_id="1", importance=0.4,
                    observations=2),
    ]


def test_T10_sans_lignes_le_bloc_est_identique_octet_pour_octet():
    entrees = _entrees_types()
    maintenant = wall_clock(ts(5))
    avant = memoire_noyau({}, entrees, maintenant, "1")
    assert memoire_noyau({}, entrees, maintenant, "1", ()) == avant
    assert memoire_noyau({}, entrees, maintenant, "1", []) == avant


def test_T10_lignes_en_tete_jamais_evincees(monkeypatch):
    monkeypatch.setattr(settings.agent, "memoire__changements_max", 1)
    bloc = memoire_noyau({}, _entrees_types(), wall_clock(ts(5)), "1", ["[ PRESSE ] x"])
    i = bloc.index("What changed recently")
    assert bloc[i + 1] == "- [ PRESSE ] x" and bloc[i + 2] == "- choc grave"


def test_T10_sans_evenement_ou_pour_un_choc_vecu_aucune_ligne(tmp_path, monkeypatch):
    assert lignes("101", ts(4)) == []  # no registry
    choc = charger(RACINE / "config" / "evenements" / "c6_voiture_suspecte.yaml")
    assert choc.service is None and choc.relais is None
    reg = RegistreEvenements(choc, journal=None)
    monkeypatch.setattr(ev, "_registre", reg)
    assert lignes("101", ts(4)) == []


def test_T10_foyer_temoin_rien(dispositif):
    assert lignes("201", ts(4)) == []


# ═════════════ T11 — the declaration ══════════════════════════════════════════════════════
@pytest.mark.parametrize("surcharges,motif", [
    ({"relais": {"mode": "par_destinataire"}, "canal": "vecu", "moment": "reveil"}, None),
    ({"exposition": {"regle": "agents", "agents": ["101"]}}, "foyers"),
    ({"relais": {"mode": "diner"}}, "unknown"),
    ({"service": {"jours_de_deplacement": 0}}, "≥ 1"),
    ({"service": {"jours_de_deplacement": 2.5}}, "≥ 1"),
    ({"service": {"jours_de_deplacement": True}}, "≥ 1"),
])
def test_T11_refus(tmp_path, surcharges, motif):
    with pytest.raises(RefusDEvenement) as err:
        _charger(tmp_path, **surcharges)
    if motif:
        assert motif in str(err.value)


def test_T11_relais_sur_canal_vecu(tmp_path):
    d = {
        "evenement": "e_test", "canal": "vecu", "moment": "arrivee", "jugement": "aucun",
        "exposition": {"regle": "agents", "agents": ["609"]},
        "jours": [{"jour": 3, "retard_min": 30, "texte": "I could not start my car."}],
        "relais": {"mode": "par_destinataire"},
    }
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(d), "utf-8")
    with pytest.raises(RefusDEvenement, match="canal: vecu"):
        charger(p)


def test_T11_cles_absentes_comportement_d_avant_et_journalise(tmp_path, journal_logs):
    d = _declaration(tmp_path)
    d.pop("service")
    d.pop("relais")
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), "utf-8")
    e = charger(p)
    assert e.service is None and e.relais is None
    assert any("comportement d'avant le ticket 111" in m for _, m in journal_logs)


@pytest.mark.parametrize("article", [
    "a07_greve_eboueurs", "a09_vent_autan", "a13_punaises_metro", "a18_la_machine",
    "a25_velotoulouse",
])
def test_T11_les_cinq_articles_portent_leurs_deux_cles(article):
    e = charger(RACINE / "config" / "evenements" / f"{article}.yaml")
    assert e.service is not None and e.service.jours_de_deplacement == 5
    assert e.relais is not None and e.relais.mode == "par_destinataire"


# ═════════════ T12 — the affinity survey sees the same line ═══════════════════════════════
def test_T12_l_enquete_voit_la_ligne_de_la_decision(dispositif, tmp_path, monkeypatch):
    from urban_mobility_agents import enquetes

    d = dispositif
    lecteur = next(p for p in d.population if p.person_id == d.lecteur)
    monkeypatch.setenv("EXPERIMENT_TARGET_PERSONAS", d.lecteur)
    monkeypatch.setenv("EXPERIMENT_SURVEY_MODES", "car")
    enquetes.reinitialiser()
    vus: list[dict] = []

    class _Client:
        async def execute(self, payload):
            vus.append(payload)
            return SimpleNamespace(agents=[SimpleNamespace(
                agent_id=payload["agents"][0]["agent_id"],
                scores=dict.fromkeys(enquetes.CRITERES, 5), justification="ok",
            )], provider_used="faux")

    agent = SimpleNamespace(
        llm_client=_Client(),
        long_term_memory=SimpleNamespace(user_metadata={}, journal_trajets=lambda pid: {}),
        get_person_identity_description=lambda p: "Arthur, 40",
    )
    asyncio.run(enquetes.executer_enquetes_jalon(12, ts(5, 20, 0), [lecteur], agent, tmp_path))
    assert vus and all(ligne_de_lecture(ARTICLE) in v["agents"][0]["perception"] for v in vus)
    # A survey is not a decision: it does not count among the served lines.
    assert d.registre._compteurs.lectures_servies == 0


# ═════════════ Relay: guards traced, never enforced ═══════════════════════════════════════
def test_les_gardes_tracent_sans_refuser():
    membres = [{"agent_id": "1", "mineur": False}, {"agent_id": "2", "mineur": True}]
    reponse = SimpleNamespace(agents=[SimpleNamespace(agent_id="9", recipients=[
        {"agent_id": "1", "speaks": True, "message": "You should avoid the park tonight."},
        {"agent_id": "2", "speaks": True, "message": "We will walk together."},
    ])])
    messages = relais_module.valider(reponse, "9", membres, "H", "e")
    assert messages[0].directif and "adresse" in messages[0].familles
    assert messages[1].directif  # parental decision: directive by nature


def test_relais_obligatoire_refuse_un_destinataire_non_informe():
    membres = [{"agent_id": "1", "mineur": False}]
    reponse = SimpleNamespace(agents=[SimpleNamespace(agent_id="9", recipients=[
        {"agent_id": "1", "speaks": False, "message": ""},
    ])])
    with pytest.raises(relais_module.RelaisRefuse) as exc:
        relais_module.valider(
            reponse, "9", membres, "H", "e", parole_obligatoire=True
        )
    assert exc.value.technique is True


def test_a13_force_la_parole_a_tout_le_foyer():
    e = charger(RACINE / "config" / "evenements" / "a13_punaises_metro.yaml")
    assert e.relais is not None and e.relais.parole_obligatoire is True


# ═════════════ T1 / T8 on the REAL decision path (spec-critic review of 2026-09-25) ═════════
class _CacheComplet(_CacheEspion):
    def __init__(self):
        super().__init__()
        self.stores = 0

    async def store(self, **kw):
        self.stores += 1


class _ClientDecision:
    """Answers "car 100 %" and keeps the received payload, as it leaves for the gateway."""

    def __init__(self):
        self.payloads = []

    async def execute(self, payload, wait_timeout=None):
        self.payloads.append(payload)
        n = len(payload["agents"][0]["trajectories"])
        probas = [SimpleNamespace(index=i, probability=100 if i == 0 else 0, reason="r")
                  for i in range(n)]
        return SimpleNamespace(
            ok=True, provider_used="faux", task_id="t", timing=None,
            agents=[SimpleNamespace(probabilities=probas, chosen_index=None, reason="r",
                                    model_dump=lambda: {})],
        )


def _decision_complete(monkeypatch, agent, personne, t, memoire: bool):
    import urban_mobility_agents.agents.llm_agent as la

    monkeypatch.setattr(la, "create_background_task", lambda coro: asyncio.ensure_future(coro))
    agent.long_term_memory.has_memories = lambda pid: memoire
    agent.llm_client = _ClientDecision()
    agent.add_short_term_memory = lambda *a, **k: None

    async def _jouer():
        resultat = await agent.evaluate_and_choose_travel_plan(
            _contexte(personne, t), [_Option("car"), _Option("walk")], "work", departure_time=t,
        )
        await asyncio.sleep(0)  # lets a possible background `store` go out
        return resultat

    return asyncio.run(_jouer())


@pytest.mark.parametrize("memoire", [True, False])
def test_T1_la_ligne_part_dans_le_payload_de_la_vraie_decision(dispositif, monkeypatch, memoire):
    """Both branches that build the payload (memory present or empty) carry the line,
    and the decision does not enter the cache (H6) — the control, for its part, does."""
    d = dispositif
    lecteur = next(p for p in d.population if p.person_id == d.lecteur)
    agent = _vrai_agent(monkeypatch, d.lecteur, entrees=[])
    agent.llm_cache = _CacheComplet()
    _decision_complete(monkeypatch, agent, lecteur, ts(4, 7, 30), memoire)
    history = agent.llm_client.payloads[0]["agents"][0]["history"]
    assert any(f"{PREFIXE_LU} I read in the paper:" in h for h in history)
    assert agent.llm_cache.lookups == 0 and agent.llm_cache.stores == 0
    assert d.registre._compteurs.decisions_sans_ligne == 0

    temoin = next(p for p in d.population if p.person_id == "201")
    agent_t = _vrai_agent(monkeypatch, "201", entrees=[])
    agent_t.llm_cache = _CacheComplet()
    _decision_complete(monkeypatch, agent_t, temoin, ts(4, 7, 30), memoire)
    history_t = agent_t.llm_client.payloads[0]["agents"][0]["history"]
    assert not any(PREFIXE_LU in h for h in history_t)
    assert agent_t.llm_cache.stores == 1


def test_une_decision_rendue_sans_sa_ligne_leve_l_alarme(dispositif, journal_logs):
    d = dispositif
    servies = lignes(d.lecteur, ts(4))
    ev.noter_rendu(d.lecteur, ts(4), servies, ["Today went smoothly."])
    assert d.registre._compteurs.decisions_sans_ligne == 1
    assert any(n == "ERROR" and "[ALARME]" in m and d.lecteur in m for n, m in journal_logs)
    ev.noter_rendu(d.lecteur, ts(4), servies, list(servies))
    assert d.registre._compteurs.decisions_sans_ligne == 1


# ═════════════ Resume: who has read is re-read from the log (spec-critic review, § 3.2) ═════
def _reprendre(d, monkeypatch):
    reprise = RegistreEvenements(d.registre.evenement, journal=d.dossier / "evenements.jsonl")
    monkeypatch.setattr(ev, "_registre", reprise)
    reprise.lecteurs(d.population)
    return reprise


def test_la_reprise_relit_qui_a_lu_et_ne_reinjecte_pas(dispositif, monkeypatch, journal_logs):
    """A replay that goes through 00:00 of the reading day again does not make the reading due a
    second time — otherwise the replay freeze would declare it void."""
    d = dispositif
    adulte = _autre_adulte(d.lecteur)
    _injecter(d, FauxClient(relais=relais_standard(d.lecteur, adulte)))
    reprise = _reprendre(d, monkeypatch)
    assert reprise._ont_lu == {d.lecteur}
    assert reprise._entendus == {adulte, ENFANT}
    assert reprise.dus_au_reveil(ts(4, 0, 0), d.population) == []
    assert any("reprise" in m and "1 lecture(s) et 2 membre(s)" in m for _, m in journal_logs)
    # The next day, the reading is still served: it did take place.
    reprise.noter_instant(ts(5, 0, 0))
    assert lignes(d.lecteur, ts(5)) == [ligne_de_lecture(ARTICLE)]


def test_un_lecteur_jamais_injecte_ne_sert_plus_apres_son_jour(dispositif, monkeypatch,
                                                               journal_logs):
    """Resume after a voided exposure: the new registry ignores it, but the log does not carry
    the reading. Past the reading day, neither the reader nor their household is served."""
    d = dispositif
    _brancher(d, FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur))))
    d.registre.noter_instant(ts(3, 7, 0))
    assert lignes(d.lecteur, ts(4)) == [ligne_de_lecture(ARTICLE)]   # the day before: served
    d.registre.noter_instant(ts(4, 12, 0))
    assert lignes(d.lecteur, ts(4, 18, 0)) != []                     # the same day: served
    d.registre.noter_instant(ts(5, 0, 0))
    assert lignes(d.lecteur, ts(5)) == []
    assert lignes(ENFANT, ts(5)) == []
    alarmes = [m for n, m in journal_logs if n == "ERROR" and "jamais été injecté" in m]
    assert len(alarmes) == 1 and d.lecteur in alarmes[0]


# ═════════════ Single-member household, secondary reader (spec-critic review § 3.3, § 3.6) ═
def test_un_lecteur_seul_n_a_rien_a_transmettre_et_ce_n_est_pas_un_refus(dispositif,
                                                                         monkeypatch,
                                                                         journal_logs):
    d = dispositif
    client = FauxClient(relais=lambda a: pytest.fail("no call for a lone reader"))
    _brancher(d, client)
    monkeypatch.setattr(foyer_module, "autres_membres", lambda pid: [])
    assert lignes(ENFANT, ts(4)) == []
    c = d.registre._compteurs
    assert (c.relais_sans_membre, c.relais_refuses, c.relais_produits) == (1, 0, 0)
    assert not any("REFUSÉ" in m for n, m in journal_logs if n == "ERROR")
    trace = (d.dossier / "relais_foyer.jsonl").read_text().splitlines()
    assert len(trace) == 1 and json.loads(trace[0])["refus"] == ""


def test_un_lecteur_secondaire_ne_relaie_pas_une_seconde_fois(dispositif):
    d = dispositif
    adulte = _autre_adulte(d.lecteur)
    client = FauxClient(relais=relais_standard(d.lecteur, adulte))
    agent = _injecter(d, client)
    avant = len(agent.longue)
    hh = d.registre._lecteurs[d.lecteur][0]
    d.registre._lecteurs[adulte] = (hh, "second lecteur")
    applique = SimpleNamespace(evenement_id=d.registre.evenement.evenement_id)
    asyncio.run(d.boucle._injecter_le_relais(d.registre, adulte, applique, ts(4, 0, 0)))
    asyncio.run(d.boucle._injecter_le_relais(d.registre, d.lecteur, applique, ts(4, 0, 0)))
    assert len(agent.longue) == avant


# ═════════════ A technical failure is retried on resume, twice at most (author, 2026-09-25)
class _ClientEnPanne(FauxClient):
    async def execute(self, payload, **kw):
        if payload.get("category") == "evenement_relais":
            self.appels.append(payload)
            return SimpleNamespace(agents=[], error="503 high demand", provider_used="")
        return await super().execute(payload, **kw)


def test_un_echec_technique_est_retente_a_la_reprise_deux_fois_au_plus(dispositif, monkeypatch,
                                                                        journal_logs):
    d = dispositif
    for tentative in (1, 2, 3):
        client = _ClientEnPanne()
        _brancher(d, client)
        assert lignes(ENFANT, ts(4)) == []
        assert lignes(ENFANT, ts(4)) == []           # no new attempt within the same run
        assert len(client.par_categorie("evenement_relais")) == 1, tentative
        d.registre = _reprendre(d, monkeypatch)
    # Three attempts made: the fourth resume no longer asks for anything.
    client = FauxClient(relais=lambda a: pytest.fail("fourth attempt"))
    _brancher(d, client)
    assert lignes(ENFANT, ts(4)) == []
    trace = [json.loads(l) for l in (d.dossier / "relais_foyer.jsonl").read_text().splitlines()]
    assert [(r["technique"], r["tentative"]) for r in trace] == [(True, 1), (True, 2), (True, 3)]
    assert any("ÉCHEC TECHNIQUE" in m for n, m in journal_logs if n == "ERROR")
    assert any("Plus aucun essai" in m for n, m in journal_logs if n == "ERROR")


def test_une_reprise_apres_un_echec_technique_informe_le_foyer(dispositif, monkeypatch):
    d = dispositif
    _brancher(d, _ClientEnPanne())
    assert lignes(ENFANT, ts(4)) == []
    d.registre = _reprendre(d, monkeypatch)
    _brancher(d, FauxClient(relais=relais_standard(d.lecteur, _autre_adulte(d.lecteur))))
    assert lignes(ENFANT, ts(4)) != []


def test_un_refus_sur_le_contenu_n_est_jamais_retente(dispositif, monkeypatch):
    d = dispositif
    _brancher(d, FauxClient(relais=_reponse([], agent_id="999")))
    assert lignes(ENFANT, ts(4)) == []
    d.registre = _reprendre(d, monkeypatch)
    _brancher(d, FauxClient(relais=lambda a: pytest.fail("content refusal retried")))
    assert lignes(ENFANT, ts(4)) == []
