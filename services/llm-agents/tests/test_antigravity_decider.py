"""Antigravity decider.

Covers:
- DecideurSpec and validation (modele required)
- Decision-maker fingerprint with modele_verifie: false
- Canonical naming (segment_decideur agy-..., full name, NommageImpossible)
- Byte-exact identity of the presented text (PromptEngine vs decideur_antigravity)
- IPC contract: atomic write, off-topic rejection, modele_declare rejection, D10 fallback, timeout
- Waiting states (ETAT_EN_ATTENTE_AGENT)
- Sealed trace and archive validation
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from experiences.archive import (
    ETAT_DEFINIE,
    ETAT_EN_ATTENTE_AGENT,
    ETAT_EN_COURS,
    Execution,
    valider_archive,
)
from experiences.decideur_antigravity import DecideurAntigravity
from experiences.decideurs import construire_decideur
from experiences.decision import (
    ContexteDecision,
    Proposition,
    ReponseDecideur,
    construire_trace,
    decider,
)
from experiences.experience import (
    DecideurSpec,
    _empreinte_decideur,
    empreinte_gabarit,
)
from experiences.nommage import NommageImpossible, nom_canonique, segment_decideur
from mobility_llm import CATEGORIES, prompt_manager
from models import (
    Location,
    Person,
    PersonalIdentity,
    PersonState,
    Transit,
    TransitLocation,
    TravelPlan,
)

HOME = Location(lat=43.6000, lon=1.4400)
WORK = Location(lat=43.6100, lon=1.4500)


def _person(pid="p1") -> Person:
    return Person(
        person_id=pid,
        identity=PersonalIdentity(name="test", traits_json={"age": 30}, home=HOME),
        state=PersonState(),
    )


def _plan(code: str, *modes: str, start=HOME, end=WORK, duration=600) -> TravelPlan:
    if not modes:
        modes = ("car",)
    loc = TransitLocation(stop="", lat=start.lat, lon=start.lon)
    legs = [
        Transit(
            start_time=0,
            end_time=duration,
            duration=duration,
            distance=1000.0,
            mode=m,
            start_location=loc,
            end_location=loc,
            is_transfer=(m == "foot" and len(modes) > 1),
            transit_route=code if (m not in ("foot",) or len(modes) == 1) else None,
        )
        for i, m in enumerate(modes)
    ]
    return TravelPlan(
        id=code,
        start_location=start,
        end_location=end,
        start_time=0,
        end_time=duration,
        duration=duration,
        legs=legs,
    )



def _ctx(activity_id="act_test", purpose="work", **kw) -> ContexteDecision:
    return ContexteDecision(
        timestamp=kw.pop("timestamp", 1773733200),
        activity_id=activity_id,
        purpose=purpose,
        departure_time=kw.pop("departure_time", 1773733200),
        from_location=kw.pop("from_location", HOME),
        destination=kw.pop("destination", WORK),
        **kw,
    )


# ── 1. DecideurSpec & Fingerprint ────────────────────────────────────────────


def test_decideur_spec_antigravity_valide():
    spec = DecideurSpec(type="antigravity", modele="gemini-3.8-flash")
    assert spec.valider() == []
    assert spec.type == "antigravity"
    assert spec.modele == "gemini-3.8-flash"


def test_decideur_spec_antigravity_modele_obligatoire():
    spec = DecideurSpec(type="antigravity", modele=None)
    errs = spec.valider()
    assert len(errs) == 1
    assert "decideur.modele est obligatoire" in errs[0]

    spec_vide = DecideurSpec(type="antigravity", modele="")
    assert len(spec_vide.valider()) == 1


def test_empreinte_decideur_modele_verifie_false():
    spec = DecideurSpec(type="antigravity", modele="gemini-3.8-flash")
    emp = _empreinte_decideur(spec)
    assert emp["type"] == "antigravity"
    assert emp["modele"] == "gemini-3.8-flash"
    assert emp["modele_verifie"] is False
    assert "sha256" in emp


# ── 2. Canonical naming ──────────────────────────────────────────────────────


def test_segment_decideur_antigravity():
    assert (
        segment_decideur({"type": "antigravity", "modele": "gemini-3.8-flash"})
        == "agy-gemini-38-f"
    )


def test_segment_decideur_antigravity_modele_vide():
    with pytest.raises(NommageImpossible, match="decideur.modele is empty"):
        segment_decideur({"type": "antigravity", "modele": ""})


def test_nom_canonique_antigravity():
    exp_def = {
        "population": {"chemin": "population_1000_PANEL.json"},
        "jeu": {"nom": "population_1000_PANEL_20260316"},
        "gabarit": {"categorie": "itinary_multi_agent", "variante": "minimal_persona"},
        "decideur": {
            "type": "antigravity",
            "modele": "gemini-3.8-flash",
            "parametres": {"temperature": 0.0},
        },
        "calendrier": {"politique": "aleatoire", "graine": 42},
        "mode": "sans_simulateur",
    }
    nom = nom_canonique(exp_def)
    assert nom == "exp_agy-gemini-38-f_minper_jtir_pop-1000_PANEL_t0_nosim"
    assert len(nom) <= 64

    # Naming queries no prompt (`minimal_persona` has been invalidated since
    # 2026-09-10 but remains nameable: past runs keep their identity).
    # Its replacement carries its own segment, which tells the two apart at a glance.
    exp_def["gabarit"]["variante"] = "prompt_minimal"
    assert (
        nom_canonique(exp_def)
        == "exp_agy-gemini-38-f_promin_jtir_pop-1000_PANEL_t0_nosim"
    )


# ── 3. Byte-exact identity of the presented text (§8.1) ──────────────────────


@pytest.mark.asyncio
async def test_identite_texte_presente_octet_par_octet(tmp_path):
    """Checks that DecideurAntigravity produces exactly the messages of PromptEngine.render."""
    person = _person("4242")
    ctx = _ctx(
        activity_id="act_test",
        purpose="work",
        timestamp=1773733200,
        departure_time=1773733200,
        anticipation=None,
    )
    plan1 = _plan("car_opt", "car", duration=1800)
    plan2 = _plan("bike_opt", "bicycle", duration=2400)
    presentees = [
        Proposition(plan=plan1, source="enregistree"),
        Proposition(plan=plan2, source="enregistree"),
    ]

    mock_agent = MagicMock()
    simulated_payload = {
        "category": "itinary_multi_agent",
        "agents": [
            {
                "agent_id": "4242",
                "perception": "Test Persona",
                "destination": "work",
                "destination_zone": None,
                "departure_time": "08:00",
                "departure_timestamp": 1773733200.0,
                "current_time": "08:00",
                "context": "Beau temps",
                "day_outlook": None,
                "agenda": [],
                "history": [],
                "trajectories": [
                    {
                        "index": 0,
                        "mode": "car",
                        "description": "Voiture 30 min",
                        "total_distance_m": 5000,
                    },
                    {
                        "index": 1,
                        "mode": "bicycle",
                        "description": "Vélo 40 min",
                        "total_distance_m": 6000,
                    },
                ],
            }
        ],
        "parameters": {"prompt_variant": "prompt_minimal"},
    }
    mock_agent.build_travel_plan_payload = AsyncMock(return_value=simulated_payload)

    # Direct PromptEngine reference
    cat = CATEGORIES["itinary_multi_agent"]
    items = [cat.item_model(**a) for a in simulated_payload["agents"]]
    messages_ref = prompt_manager().render(
        "itinary_multi_agent", items, simulated_payload["parameters"]
    )
    ref_dict = [{"role": m.role, "content": m.content} for m in messages_ref]

    decideur = DecideurAntigravity(
        agent=mock_agent,
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=1,
    )

    # Launch choisir and immediately inject the IPC response
    async def injecter_reponse():
        await asyncio.sleep(0.1)
        demande_fichier = tmp_path / "echanges" / "demandes" / "4242__act_test.json"
        assert demande_fichier.is_file()
        demande_data = json.loads(demande_fichier.read_text(encoding="utf-8"))

        # Byte-level check / verbatim structure
        assert demande_data["messages"] == ref_dict
        assert demande_data["person_id"] == "4242"
        assert demande_data["activity_id"] == "act_test"
        assert demande_data["modele_attendu"] == "gemini-3.8-flash"
        assert demande_data["n_options"] == 2

        # Write response
        reponse_fichier = tmp_path / "echanges" / "reponses" / "4242__act_test.json"
        reponse_tmp = tmp_path / "echanges" / "reponses" / "4242__act_test.json.tmp"
        rep_content = {
            "version": 1,
            "person_id": "4242",
            "activity_id": "act_test",
            "modele_declare": "gemini-3.8-flash",
            "reponse_brute": "raw text",
            "agents": [
                {
                    "agent_id": "4242",
                    "probabilities": [
                        {"index": 0, "mode": "car", "probability": 80.0, "reason": "rapide"},
                        {"index": 1, "mode": "bicycle", "probability": 20.0, "reason": "pluie"},
                    ],
                }
            ],
        }
        reponse_tmp.write_text(json.dumps(rep_content), encoding="utf-8")
        import os

        os.replace(reponse_tmp, reponse_fichier)

    task = asyncio.create_task(injecter_reponse())
    rep = await decideur.choisir(person, ctx, presentees)
    await task

    assert rep.index is not None
    assert rep.modele_verifie is False
    assert rep.presente["messages"] == ref_dict
    assert not rep.repli_uniforme


# ── 4. IPC contract (Rejections, Timeout, States) ────────────────────────────


@pytest.mark.asyncio
async def test_ipc_reponse_hors_sujet_puis_bonne_reponse(tmp_path):
    person = _person("101")
    ctx = _ctx(activity_id="act1", purpose="work", timestamp=100, departure_time=100)
    presentees = [
        Proposition(plan=_plan("c", "car", duration=100), source="enregistree"),
        Proposition(plan=_plan("w", "walk", duration=200), source="enregistree"),
    ]

    mock_agent = MagicMock()
    mock_agent.build_travel_plan_payload = AsyncMock(
        return_value={
            "category": "itinary_multi_agent",
            "agents": [
                {
                    "agent_id": "101",
                    "perception": "test",
                    "destination": "work",
                    "trajectories": [{"mode": "car"}, {"mode": "walk"}],
                }
            ],
            "parameters": {},
        }
    )

    decideur = DecideurAntigravity(
        agent=mock_agent,
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=3,
    )

    async def scenario():
        await asyncio.sleep(0.1)
        rep_file = tmp_path / "echanges" / "reponses" / "101__act1.json"
        # 1. Wrong person_id
        rep_file.write_text(
            json.dumps({"person_id": "999", "activity_id": "act1", "modele_declare": "gemini-3.8-flash"})
        )
        await asyncio.sleep(0.6)
        # Off-topic file deleted by the decider
        assert not rep_file.exists()
        # 2. Correct response
        rep_file.write_text(
            json.dumps(
                {
                    "person_id": "101",
                    "activity_id": "act1",
                    "modele_declare": "gemini-3.8-flash",
                    "agents": [
                        {
                            "agent_id": "101",
                            "probabilities": [
                                {"index": 0, "mode": "car", "probability": 100},
                                {"index": 1, "mode": "walk", "probability": 0},
                            ],
                        }
                    ],
                }
            )
        )

    task = asyncio.create_task(scenario())
    rep = await decideur.choisir(person, ctx, presentees)
    await task

    assert rep.index == 0
    assert decideur.compteurs["rejets"] == 1
    assert decideur.compteurs["servies"] == 1


@pytest.mark.asyncio
async def test_ipc_modele_declare_inattendu(tmp_path):
    person = _person("102")
    ctx = _ctx(activity_id="act2", purpose="work", timestamp=100, departure_time=100)
    presentees = [
        Proposition(plan=_plan("c", "car", duration=100), source="enregistree"),
        Proposition(plan=_plan("w", "walk", duration=200), source="enregistree"),
    ]

    mock_agent = MagicMock()
    mock_agent.build_travel_plan_payload = AsyncMock(
        return_value={
            "category": "itinary_multi_agent",
            "agents": [
                {
                    "agent_id": "102",
                    "perception": "test",
                    "destination": "work",
                    "trajectories": [{"mode": "car"}, {"mode": "walk"}],
                }
            ],
            "parameters": {},
        }
    )

    decideur = DecideurAntigravity(
        agent=mock_agent,
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=2,
    )

    async def scenario():
        await asyncio.sleep(0.1)
        rep_file = tmp_path / "echanges" / "reponses" / "102__act2.json"
        rep_file.write_text(
            json.dumps(
                {
                    "person_id": "102",
                    "activity_id": "act2",
                    "modele_declare": "autre-modele-inattendu",
                }
            )
        )

    task = asyncio.create_task(scenario())
    rep = await decideur.choisir(person, ctx, presentees)
    await task

    assert rep.index is None
    assert "modèle déclaré inattendu" in rep.erreur
    assert decideur.compteurs["rejets"] == 1


@pytest.mark.asyncio
async def test_ipc_timeout_non_reponse(tmp_path):
    person = _person("103")
    ctx = _ctx(activity_id="act3", purpose="work", timestamp=100, departure_time=100)
    presentees = [
        Proposition(plan=_plan("c", "car", duration=100), source="enregistree"),
        Proposition(plan=_plan("w", "walk", duration=200), source="enregistree"),
    ]

    mock_agent = MagicMock()
    mock_agent.build_travel_plan_payload = AsyncMock(
        return_value={
            "category": "itinary_multi_agent",
            "agents": [
                {
                    "agent_id": "103",
                    "perception": "test",
                    "destination": "work",
                    "trajectories": [{"mode": "car"}, {"mode": "walk"}],
                }
            ],
            "parameters": {},
        }
    )

    decideur = DecideurAntigravity(
        agent=mock_agent,
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=1,
    )

    rep = await decideur.choisir(person, ctx, presentees)
    assert rep.index is None
    assert "pas de réponse en 1s" in rep.erreur
    assert decideur.compteurs["timeouts"] == 1


@pytest.mark.asyncio
async def test_changement_etat_en_attente_agent(tmp_path):
    person = _person("104")
    ctx = _ctx(activity_id="act4", purpose="work", timestamp=100, departure_time=100)
    presentees = [
        Proposition(plan=_plan("c", "car", duration=100), source="enregistree"),
        Proposition(plan=_plan("w", "walk", duration=200), source="enregistree"),
    ]

    mock_agent = MagicMock()
    mock_agent.build_travel_plan_payload = AsyncMock(
        return_value={
            "category": "itinary_multi_agent",
            "agents": [
                {
                    "agent_id": "104",
                    "perception": "test",
                    "destination": "work",
                    "trajectories": [{"mode": "car"}, {"mode": "walk"}],
                }
            ],
            "parameters": {},
        }
    )

    # Create a fake run to follow the state
    mock_exec = MagicMock()
    etat_dict = {"etat": ETAT_EN_COURS}

    def get_etat():
        return etat_dict

    def changer_etat(nouvel_etat, raison=None):
        etat_dict["etat"] = nouvel_etat

    mock_exec.etat = get_etat
    mock_exec.changer_etat = changer_etat

    decideur = DecideurAntigravity(
        agent=mock_agent,
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=2,  # threshold = 0.5s
        execution=mock_exec,
    )

    async def repondre_apres_seuil():
        await asyncio.sleep(0.7)  # > 0.5s -> switches to ETAT_EN_ATTENTE_AGENT
        assert etat_dict["etat"] == ETAT_EN_ATTENTE_AGENT
        rep_file = tmp_path / "echanges" / "reponses" / "104__act4.json"
        rep_file.write_text(
            json.dumps(
                {
                    "person_id": "104",
                    "activity_id": "act4",
                    "modele_declare": "gemini-3.8-flash",
                    "agents": [
                        {
                            "agent_id": "104",
                            "probabilities": [
                                {"index": 0, "mode": "car", "probability": 100},
                                {"index": 1, "mode": "walk", "probability": 0},
                            ],
                        }
                    ],
                }
            )
        )

    task = asyncio.create_task(repondre_apres_seuil())
    rep = await decideur.choisir(person, ctx, presentees)
    await task

    assert rep.index == 0
    # After a valid response, the state returns to ETAT_EN_COURS
    assert etat_dict["etat"] == ETAT_EN_COURS


# ── 5. Sealed trace and archive validation ───────────────────────────────────


def test_construire_trace_modele_verifie():
    person = _person("200")
    ctx = _ctx(activity_id="act_tr", purpose="work", timestamp=100, departure_time=100)
    prop = Proposition(plan=_plan("car", "car", duration=100), source="enregistree")

    reponse = ReponseDecideur(
        index=0,
        fournisseur="antigravity:gemini-3.8-flash",
        modele_verifie=False,
        presente={"test": True},
    )

    trace = construire_trace(
        person,
        ctx,
        [prop],
        [],
        prop,
        "decideur",
        reponse,
        "",
    )
    assert trace["modele_verifie"] is False
    assert trace["presente"] == {"test": True}


def test_construire_decideur_antigravity_factory(tmp_path):
    spec = DecideurSpec(type="antigravity", modele="gemini-3.8-flash")
    mock_agent = MagicMock()
    d = construire_decideur(
        spec,
        agent=mock_agent,
        dossier_echanges=tmp_path / "echanges",
        attente_max_s=60,
    )
    assert isinstance(d, DecideurAntigravity)
    assert d.modele == "gemini-3.8-flash"
    assert d.sans_quota is True
    assert d.modele_verifie is False


# ── P8 — literal model output (hygiene spec §8) ──────────────────────────────


def test_trace_porte_la_sortie_litterale():
    """The model output travels in the trace, AS EMITTED."""
    person = _person("300")
    ctx = _ctx(activity_id="act_lit", purpose="work", timestamp=100, departure_time=100)
    prop = Proposition(plan=_plan("car", "car", duration=100), source="enregistree")
    litterale = '```json\n{"agents": [{"agent_id": "300"}]}\n```'

    trace = construire_trace(
        person, ctx, [prop], [], prop, "decideur",
        ReponseDecideur(
            index=0,
            fournisseur="antigravity:m",
            reponse_brute='{"agents": [{"agent_id": "300"}]}',
            sortie_litterale=litterale,
            presente={"x": 1},
        ),
        "",
    )
    assert trace["sortie_litterale"] == litterale
    assert trace["sortie_litterale"] != trace["reponse_brute"], (
        "the field is only of interest if it can differ from the normalised version"
    )


def test_trace_declare_un_trou_plutot_qu_une_copie():
    """Without literal output, the trace carries None — never a fallback to reponse_brute."""
    person = _person("301")
    ctx = _ctx(activity_id="act_trou", purpose="work", timestamp=100, departure_time=100)
    prop = Proposition(plan=_plan("car", "car", duration=100), source="enregistree")

    trace = construire_trace(
        person, ctx, [prop], [], prop, "decideur",
        ReponseDecideur(
            index=0, fournisseur="antigravity:m", reponse_brute='{"agents": []}', presente={"x": 1}
        ),
        "",
    )
    assert trace["sortie_litterale"] is None
    assert trace["reponse_brute"] == '{"agents": []}'


@pytest.mark.asyncio
async def test_decideur_transmet_la_sortie_litterale(tmp_path):
    """End to end: the IPC file field lands in the decider's response."""
    person = _person("4243")
    ctx = _ctx(activity_id="act_p8", purpose="work", timestamp=1773733200, departure_time=1773733200, anticipation=None)
    presentees = [
        Proposition(plan=_plan("car_opt", "car", duration=1800), source="enregistree"),
        Proposition(plan=_plan("bike_opt", "bicycle", duration=2400), source="enregistree"),
    ]
    mock_agent = MagicMock()
    mock_agent.build_travel_plan_payload = AsyncMock(return_value={
        "category": "itinary_multi_agent",
        "agents": [{
            "agent_id": "4243", "perception": "P", "destination": "work",
            "destination_zone": None, "departure_time": "08:00",
            "departure_timestamp": 1773733200.0, "current_time": "08:00",
            "context": "Beau temps", "day_outlook": None, "agenda": [], "history": [],
            "trajectories": [
                {"index": 0, "mode": "car", "description": "d", "total_distance_m": 5000},
                {"index": 1, "mode": "bicycle", "description": "d", "total_distance_m": 6000},
            ],
        }],
        "parameters": {"prompt_variant": "prompt_minimal"},
    })
    d = DecideurAntigravity(agent=mock_agent, modele="m", echanges=tmp_path / "e", attente_max_s=5)

    litterale = "Voici ma réponse :\n```json\n{\"agents\": [...]}\n```"

    async def repondre():
        cible = tmp_path / "e" / "reponses" / f"4243__{ctx.activity_id}.json"
        for _ in range(40):
            if (tmp_path / "e" / "demandes" / f"4243__{ctx.activity_id}.json").is_file():
                cible.write_text(json.dumps({
                    "version": 1, "person_id": "4243", "activity_id": ctx.activity_id,
                    "modele_declare": "m",
                    "sortie_litterale": litterale,
                    "reponse_brute": '{"agents": [{"agent_id": "4243"}]}',
                    "agents": [{"agent_id": "4243", "reason": "r", "probabilities": [
                        {"index": 0, "mode": "car", "probability": 70},
                        {"index": 1, "mode": "bicycle", "probability": 30},
                    ]}],
                }, ensure_ascii=False), encoding="utf-8")
                return
            await asyncio.sleep(0.05)

    _, rep = await asyncio.gather(repondre(), d.choisir(person, ctx, presentees))
    assert rep.sortie_litterale == litterale
    assert d.compteurs["sans_sortie_litterale"] == 0


@pytest.mark.asyncio
async def test_decideur_compte_les_reponses_sans_sortie_litterale(tmp_path):
    person = _person("4244")
    ctx = _ctx(activity_id="act_p8b", purpose="work", timestamp=1773733200, departure_time=1773733200, anticipation=None)
    presentees = [Proposition(plan=_plan("car_opt", "car", duration=1800), source="enregistree"),
                  Proposition(plan=_plan("bike_opt", "bicycle", duration=2400), source="enregistree")]
    mock_agent = MagicMock()
    mock_agent.build_travel_plan_payload = AsyncMock(return_value={
        "category": "itinary_multi_agent",
        "agents": [{
            "agent_id": "4244", "perception": "P", "destination": "work",
            "destination_zone": None, "departure_time": "08:00",
            "departure_timestamp": 1773733200.0, "current_time": "08:00",
            "context": "c", "day_outlook": None, "agenda": [], "history": [],
            "trajectories": [
                {"index": 0, "mode": "car", "description": "d", "total_distance_m": 1},
                {"index": 1, "mode": "bicycle", "description": "d", "total_distance_m": 2},
            ],
        }],
        "parameters": {"prompt_variant": "prompt_minimal"},
    })
    d = DecideurAntigravity(agent=mock_agent, modele="m", echanges=tmp_path / "e", attente_max_s=5)

    async def repondre():
        cible = tmp_path / "e" / "reponses" / f"4244__{ctx.activity_id}.json"
        for _ in range(40):
            if (tmp_path / "e" / "demandes" / f"4244__{ctx.activity_id}.json").is_file():
                cible.write_text(json.dumps({
                    "version": 1, "person_id": "4244", "activity_id": ctx.activity_id,
                    "modele_declare": "m",
                    "reponse_brute": '{"agents": []}',
                    "agents": [{"agent_id": "4244", "reason": "r", "probabilities": [
                        {"index": 0, "mode": "car", "probability": 50},
                        {"index": 1, "mode": "bicycle", "probability": 50},
                    ]}],
                }, ensure_ascii=False), encoding="utf-8")
                return
            await asyncio.sleep(0.05)

    _, rep = await asyncio.gather(repondre(), d.choisir(person, ctx, presentees))
    assert rep.sortie_litterale is None
    assert d.compteurs["sans_sortie_litterale"] == 1


# ── Sampling parameters and startup safeguard (fixes of 2026-09-15) ─────────────────────


def _mock_agent(agent_id: str):
    agent = MagicMock()
    agent.build_travel_plan_payload = AsyncMock(
        return_value={
            "category": "itinary_multi_agent",
            "agents": [
                {
                    "agent_id": agent_id,
                    "perception": "test",
                    "destination": "work",
                    "trajectories": [{"mode": "car"}, {"mode": "walk"}],
                }
            ],
            "parameters": {},
        }
    )
    return agent


def _deux_options():
    return [
        Proposition(plan=_plan("c", "car", duration=100), source="enregistree"),
        Proposition(plan=_plan("w", "walk", duration=200), source="enregistree"),
    ]


@pytest.mark.asyncio
async def test_demande_porte_les_parametres_demandes(tmp_path):
    """The temperature must REACH the sub-agent: it was stored without being sent."""
    person = _person("201")
    ctx = _ctx(activity_id="p1", purpose="work", timestamp=100, departure_time=100)

    decideur = DecideurAntigravity(
        agent=_mock_agent("201"),
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=1,
        parametres={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
    )

    await decideur.choisir(person, ctx, _deux_options())

    ecrites = list((tmp_path / "echanges" / "demandes").glob("*.json"))
    assert len(ecrites) == 1, "the request must stay on disk after a timeout"
    corps = json.loads(ecrites[0].read_text(encoding="utf-8"))
    assert corps["parametres"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
    }


@pytest.mark.asyncio
async def test_parametres_appliques_remontent_dans_la_reponse(tmp_path):
    """What the sub-agent declares it applied is archived as is."""
    person = _person("202")
    ctx = _ctx(activity_id="p2", purpose="work", timestamp=100, departure_time=100)
    echanges = tmp_path / "echanges"

    decideur = DecideurAntigravity(
        agent=_mock_agent("202"),
        modele="gemini-3.8-flash",
        echanges=echanges,
        attente_max_s=5,
        parametres={"temperature": 0.0},
    )

    async def repondre():
        reponses = echanges / "reponses"
        for _ in range(100):
            demandes = list((echanges / "demandes").glob("*.json"))
            if demandes:
                (reponses / demandes[0].name).write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "person_id": "202",
                            "activity_id": "p2",
                            "modele_declare": "gemini-3.8-flash",
                            "parametres_appliques": {"temperature": 0.0},
                            "sortie_litterale": '{"agents": []}',
                            "agents": [
                                {
                                    "agent_id": "202",
                                    "probabilities": [
                                        {"index": 0, "mode": "car", "probability": 60},
                                        {"index": 1, "mode": "walk", "probability": 40},
                                    ],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return
            await asyncio.sleep(0.05)

    rep, _ = await asyncio.gather(decideur.choisir(person, ctx, _deux_options()), repondre())
    assert rep.index is not None
    assert rep.parametres_appliques == {"temperature": 0.0}
    assert decideur.compteurs["sans_parametres_appliques"] == 0


@pytest.mark.asyncio
async def test_reponse_muette_sur_les_parametres_est_comptee(tmp_path):
    """A response without `parametres_appliques` is a GAP, never "temperature 0"."""
    person = _person("203")
    ctx = _ctx(activity_id="p3", purpose="work", timestamp=100, departure_time=100)
    echanges = tmp_path / "echanges"

    decideur = DecideurAntigravity(
        agent=_mock_agent("203"),
        modele="gemini-3.8-flash",
        echanges=echanges,
        attente_max_s=5,
        parametres={"temperature": 0.0},
    )

    async def repondre():
        reponses = echanges / "reponses"
        for _ in range(100):
            demandes = list((echanges / "demandes").glob("*.json"))
            if demandes:
                (reponses / demandes[0].name).write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "person_id": "203",
                            "activity_id": "p3",
                            "modele_declare": "gemini-3.8-flash",
                            "agents": [
                                {
                                    "agent_id": "203",
                                    "probabilities": [
                                        {"index": 0, "mode": "car", "probability": 60},
                                        {"index": 1, "mode": "walk", "probability": 40},
                                    ],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return
            await asyncio.sleep(0.05)

    rep, _ = await asyncio.gather(decideur.choisir(person, ctx, _deux_options()), repondre())
    assert rep.index is not None
    assert rep.parametres_appliques is None, "a declared gap, not an invented value"
    assert decideur.compteurs["sans_parametres_appliques"] == 1


@pytest.mark.asyncio
async def test_demarrage_sans_sous_agent_arrete_l_execution(tmp_path):
    """Without a single response, the run stops instead of sleeping through the night."""
    person = _person("204")
    ctx = _ctx(activity_id="p4", purpose="work", timestamp=100, departure_time=100)
    dossier_exec = tmp_path / "execution"
    dossier_exec.mkdir()

    mock_exec = MagicMock()
    mock_exec.dossier = dossier_exec
    mock_exec.etat.return_value = {"etat": ETAT_EN_COURS}

    decideur = DecideurAntigravity(
        agent=_mock_agent("204"),
        modele="gemini-3.8-flash",
        echanges=tmp_path / "echanges",
        attente_max_s=1,
        execution=mock_exec,
        demarrage_max_s=0.0,
    )

    rep = await decideur.choisir(person, ctx, _deux_options())

    assert rep.index is None
    assert (dossier_exec / "STOP").is_file(), (
        "the safeguard must place the STOP file: it is what closes the run as `arretee`, "
        "a FINAL state, so the campaign moves on"
    )


@pytest.mark.asyncio
async def test_garde_fou_desarme_apres_la_premiere_reponse(tmp_path):
    """A slow but live channel must never be stopped by the safeguard."""
    person = _person("205")
    ctx = _ctx(activity_id="p5", purpose="work", timestamp=100, departure_time=100)
    echanges = tmp_path / "echanges"
    dossier_exec = tmp_path / "execution"
    dossier_exec.mkdir()

    mock_exec = MagicMock()
    mock_exec.dossier = dossier_exec
    mock_exec.etat.return_value = {"etat": ETAT_EN_COURS}

    decideur = DecideurAntigravity(
        agent=_mock_agent("205"),
        modele="gemini-3.8-flash",
        echanges=echanges,
        attente_max_s=5,
        execution=mock_exec,
        demarrage_max_s=0.0,
    )
    # The channel has already served: the safeguard is disarmed for the rest of the run.
    decideur._premiere_reponse = True

    async def repondre():
        reponses = echanges / "reponses"
        for _ in range(100):
            demandes = list((echanges / "demandes").glob("*.json"))
            if demandes:
                (reponses / demandes[0].name).write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "person_id": "205",
                            "activity_id": "p5",
                            "modele_declare": "gemini-3.8-flash",
                            "parametres_appliques": {},
                            "agents": [
                                {
                                    "agent_id": "205",
                                    "probabilities": [
                                        {"index": 0, "mode": "car", "probability": 60},
                                        {"index": 1, "mode": "walk", "probability": 40},
                                    ],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return
            await asyncio.sleep(0.05)

    rep, _ = await asyncio.gather(decideur.choisir(person, ctx, _deux_options()), repondre())
    assert rep.index is not None
    assert not (dossier_exec / "STOP").exists()
    assert rep.parametres_appliques == {}, "\"no applicable parameter\" is a response"


def test_trace_porte_les_parametres_appliques():
    """`parametres_appliques` must reach decisions.jsonl, like `modele_verifie`."""
    person = _person("206")
    ctx = _ctx(activity_id="p6", purpose="work", timestamp=100, departure_time=100)
    prop = Proposition(plan=_plan("car", "car", duration=100), source="enregistree")

    reponse = ReponseDecideur(
        index=0,
        fournisseur="antigravity:gemini-3.8-flash",
        modele_verifie=False,
        parametres_appliques={"temperature": "non applicable"},
    )
    trace = construire_trace(person, ctx, [prop], [], prop, "decideur", reponse, "")
    assert trace["parametres_appliques"] == {"temperature": "non applicable"}

    # And without a declaration, the key does NOT exist: a silent trace must not read
    # as "parameters applied = {}".
    muette = ReponseDecideur(index=0, fournisseur="antigravity:x", modele_verifie=False)
    trace_muette = construire_trace(person, ctx, [prop], [], prop, "decideur", muette, "")
    assert "parametres_appliques" not in trace_muette
