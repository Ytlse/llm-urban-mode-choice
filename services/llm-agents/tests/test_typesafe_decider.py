"""The Jev (TypeSafe) decider.

One test per contract rule, written before the code. No network call: the client is injected.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from experiences.decideur_typesafe import (
    TOLERANCE_SOMME,
    DecideurTypesafe,
    cle_option,
    instructions_servies,
    sha_instructions,
)
from experiences.decideurs import construire_decideur
from experiences.decision import ContexteDecision, Proposition
from experiences.experience import DecideurSpec, _empreinte_decideur
from experiences.nommage import NommageImpossible, segment_decideur, segments
from models import (
    Location,
    Person,
    PersonalIdentity,
    PersonState,
    Transit,
    TransitLocation,
    TravelPlan,
)

pytest.importorskip("typesafe_sdk")

HOME = Location(lat=43.6000, lon=1.4400)
WORK = Location(lat=43.6100, lon=1.4500)
MODELE = "jev-1.13.0"


# ── test doubles ────────────────────────────────────────────────────────────────
def _person(pid="p1") -> Person:
    return Person(
        person_id=pid,
        identity=PersonalIdentity(name="test", traits_json={"age": 30}, home=HOME),
        state=PersonState(),
    )


def _plan(code: str, mode: str = "car", duration: int = 600) -> TravelPlan:
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


def _prop(code: str, mode: str = "car", duration: int = 600) -> Proposition:
    return Proposition(plan=_plan(code, mode, duration), source="enregistree")


def _ctx(**kw) -> ContexteDecision:
    return ContexteDecision(
        timestamp=kw.pop("timestamp", 1773733200),
        activity_id=kw.pop("activity_id", "act_test"),
        purpose=kw.pop("purpose", "shop"),
        departure_time=kw.pop("departure_time", 1773733200),
        from_location=HOME,
        destination=WORK,
        **kw,
    )


PAYLOAD_TYPE = {
    "category": "itinary_multi_agent",
    "agents": [
        {
            "agent_id": "p1",
            "perception": "Renée, 29, Full-Time Worker (household of 3, high income).",
            "destination": "shop",
            "destination_zone": "1st ring",
            "departure_time": "09:28",
            "context": "Weather: 10°C, Clear/Sunny.",
            "day_outlook": "afternoon 12°C",
            "agenda": [],
            "history": [],
            "trajectories": [
                {"index": 0, "mode": "car", "description": "Estimated duration: 19 minutes."},
                {"index": 1, "mode": "foot,bus,foot", "description": "Travel time: 2 hours.\n- Walk to 'X'."},
                {"index": 2, "mode": "foot,bus,foot", "description": "Travel time: 3 hours."},
            ],
        }
    ],
}


class AgentEspion:
    """Does NOT return a reinvented payload: it returns the one it is given, and records the call."""

    def __init__(self, payload=None):
        self.payload = json.loads(json.dumps(payload or PAYLOAD_TYPE))
        self.appels: list[dict] = []

    async def build_travel_plan_payload(self, **kw):
        self.appels.append(kw)
        return self.payload


class ClientDouble:
    def __init__(self, probabilities=None, *, leve=None, confidence=0.6, modele=MODELE):
        self.probabilities = probabilities
        self.leve = leve
        self.confidence = confidence
        self.modele = modele
        self.appels: list[dict] = []

    async def system_one(self, *, model, state, questions):
        self.appels.append({"model": model, "state": state, "questions": questions})
        if self.leve is not None:
            raise self.leve
        rep = SimpleNamespace(
            type="choice",
            choice=max(self.probabilities, key=self.probabilities.get),
            probabilities=dict(self.probabilities),
            confidence=self.confidence,
        )
        return SimpleNamespace(
            model=self.modele,
            usage=SimpleNamespace(input_tokens=1099, output_tokens=0),
            answers={"mode": rep},
        )


def _decideur(probabilities=None, *, leve=None, agent=None, variante="prompt_expert_16"):
    return DecideurTypesafe(
        agent=agent or AgentEspion(),
        modele=MODELE,
        variante=variante,
        client=ClientDouble(probabilities, leve=leve),
    )


def _choisir(dec, presentees, ctx=None):
    return asyncio.run(dec.choisir(_person(), ctx or _ctx(), presentees))


TROIS = [_prop("c", "car"), _prop("b1", "foot,bus,foot"), _prop("b2", "foot,bus,foot")]
P_OK = {"option_0": 0.7, "option_1": 0.2, "option_2": 0.1}


# ── S — the specification ─────────────────────────────────────────────────────
def test_S1_version_figee_acceptee():
    assert DecideurSpec(type="typesafe", modele=MODELE).valider() == []


def test_S2_modele_obligatoire():
    erreurs = DecideurSpec(type="typesafe").valider()
    assert any("decideur.modele" in e for e in erreurs)


@pytest.mark.parametrize("alias", ["jev-latest", "jev-preview", "jev-1.13", "jev", "JEV-1.13.0"])
def test_S3_alias_refuse(alias):
    assert DecideurSpec(type="typesafe", modele=alias).valider(), f"{alias} should have been refused"


def test_S4_le_refus_dit_pourquoi_et_donne_la_forme():
    (erreur,) = DecideurSpec(type="typesafe", modele="jev-latest").valider()
    assert "alias" in erreur.lower()
    assert "jev-1.13.0" in erreur


def test_S5_portee_refusee_hors_passerelle():
    erreurs = DecideurSpec(type="typesafe", modele=MODELE, portee="local").valider()
    assert any("portee" in e for e in erreurs)


# ── N — the naming ───────────────────────────────────────────────────────────
def test_N1_segment_nomme_la_version():
    assert segment_decideur({"type": "typesafe", "modele": MODELE}) == "jev-1130"


def test_N2_deux_versions_deux_segments():
    a = segment_decideur({"type": "typesafe", "modele": "jev-1.13.0"})
    b = segment_decideur({"type": "typesafe", "modele": "jev-1.14.0"})
    assert a != b


def test_N2b_modele_vide_refuse_le_nom():
    with pytest.raises(NommageImpossible):
        segment_decideur({"type": "typesafe", "modele": ""})


def _exp(variante="prompt_expert_16"):
    return {
        "decideur": {"type": "typesafe", "modele": MODELE},
        "gabarit": {"categorie": "itinary_multi_agent", "variante": variante},
        "population": {"chemin": "data/population_1000_PANEL_v6.json"},
        "calendrier": {"politique": "commune", "date": "2026-03-16", "graine": 42},
        "mode": "sans_simulateur",
    }


def test_N3_la_variante_de_prompt_nomme_l_experience():
    assert "proexp16" in segments(_exp())
    assert "promin02" in segments(_exp("prompt_minimal_02"))


def test_N4_aucun_segment_de_temperature():
    # Jev has no temperature: writing one would seal a setting that does not exist.
    assert not [s for s in segments(_exp()) if s.startswith("t") and s[1:].isdigit()]


# ── C — the fingerprint ──────────────────────────────────────────────────────────
GABARIT = SimpleNamespace(categorie="itinary_multi_agent", variante="prompt_expert_16")


def test_C1_empreinte_porte_type_et_version():
    emp = _empreinte_decideur(DecideurSpec(type="typesafe", modele=MODELE), GABARIT)
    assert emp["type"] == "typesafe" and emp["modele"] == MODELE


def test_C2_empreinte_porte_le_sha_des_instructions():
    emp = _empreinte_decideur(DecideurSpec(type="typesafe", modele=MODELE), GABARIT)
    assert emp["instructions_sha256"]


def test_C3_le_sha_est_celui_du_texte_ampute_pas_du_content_brut():
    import hashlib

    from mobility_llm import prompt_manager as get_prompt_manager

    brut = get_prompt_manager().get_system_prompt(
        "itinary_multi_agent", "prompt_expert_16", verifier_validite=False
    )
    servi = instructions_servies(brut)
    assert "[Output instructions]" in brut and "[Output instructions]" not in servi
    attendu = hashlib.sha256(servi.encode("utf-8")).hexdigest()
    assert sha_instructions("itinary_multi_agent", "prompt_expert_16") == attendu
    assert attendu != hashlib.sha256(brut.encode("utf-8")).hexdigest()


def test_C4_deux_variantes_deux_empreintes():
    a = sha_instructions("itinary_multi_agent", "prompt_expert_16")
    b = sha_instructions("itinary_multi_agent", "prompt_minimal_02")
    assert a and b and a != b


# ── R — the response conversion ──────────────────────────────────────────
def test_R1_poids_dans_l_ordre_de_presentation():
    # The dictionary comes back out of order: the PRESENTATION ORDER is what governs.
    dec = _decideur({"option_2": 0.1, "option_0": 0.7, "option_1": 0.2})
    r = _choisir(dec, TROIS)
    assert r.poids == pytest.approx([0.7, 0.2, 0.1])


def test_R2_arrondi_a_deux_decimales_accepte_et_renormalise():
    dec = _decideur({"option_0": 0.7, "option_1": 0.2, "option_2": 0.11})  # sum 1.01
    r = _choisir(dec, TROIS)
    assert r.non_imputable is False
    assert sum(r.poids) == pytest.approx(1.0)


def test_R3_somme_hors_tolerance_est_une_non_decision():
    dec = _decideur({"option_0": 0.4, "option_1": 0.2, "option_2": 0.1})  # sum 0.70
    r = _choisir(dec, TROIS)
    assert r.non_imputable and "somme_hors_tolerance" in r.raison
    assert r.index is None


def test_R3b_la_tolerance_est_bien_celle_qui_est_declaree():
    assert TOLERANCE_SOMME == 0.02


def test_R4_cle_manquante_est_une_non_decision():
    dec = _decideur({"option_0": 0.7, "option_1": 0.3})  # option_2 missing
    r = _choisir(dec, TROIS)
    assert r.non_imputable and "cles_manquantes" in r.raison


def test_R5_poids_tous_nuls_est_une_non_decision():
    dec = _decideur({"option_0": 0.0, "option_1": 0.0, "option_2": 0.0})
    r = _choisir(dec, TROIS)
    assert r.non_imputable


def test_R6_le_tirage_est_celui_des_autres_decideurs():
    from experiences.decideur_modele import DecideurModele
    from experiences.decision import graine_ordre

    # The two seeds are DIFFERENT on purpose: with the default values (42 and 42),
    # confusing `graine_tirage` and `graine_ordre` would go unnoticed — and that is exactly
    # the error this rule must catch.
    ctx = _ctx(graine_ordre=7, graine_tirage=99)
    poids = [0.7, 0.2, 0.1]
    attendu = DecideurModele._tirer(poids, graine_ordre(ctx.graine_tirage, "p1", ctx.activity_id))
    autre = DecideurModele._tirer(poids, graine_ordre(ctx.graine_ordre, "p1", ctx.activity_id))
    r = _choisir(_decideur(P_OK), TROIS, ctx)
    assert r.index == attendu
    if attendu != autre:
        assert r.index != autre


def test_R7_la_raison_reste_vide():
    assert _choisir(_decideur(P_OK), TROIS).raison == ""


def test_R8_la_reponse_entiere_est_archivee():
    brut = json.loads(_choisir(_decideur(P_OK), TROIS).reponse_brute)
    assert brut["probabilities"] == P_OK
    assert brut["confidence"] == 0.6
    assert brut["input_tokens"] == 1099
    assert brut["modele"] == MODELE


def test_R9_distribution_agregee_par_mode_canonique():
    d = _choisir(_decideur(P_OK), TROIS).distribution
    assert d["car"] == pytest.approx(0.7)
    assert d["public_transport"] == pytest.approx(0.3)  # the two bus options add up


def test_R10_deux_options_de_meme_mode_ne_s_ecrasent_pas():
    r = _choisir(_decideur(P_OK), TROIS)
    assert len(r.poids) == 3
    assert r.poids[1] != r.poids[2]


# ── E — the failures ───────────────────────────────────────────────────────────
class TypeSafeRateLimitError(Exception): ...
class TypeSafeAPITimeoutError(Exception): ...
class TypeSafeInternalServerError(Exception): ...
class TypeSafeAuthenticationError(Exception): ...
class TypeSafeUnprocessableEntityError(Exception): ...


@pytest.mark.parametrize(
    "exc", [TypeSafeRateLimitError("429"), TypeSafeAPITimeoutError("t"), TypeSafeInternalServerError("529")]
)
def test_E1_echec_transitoire_est_une_erreur_reessayable(exc):
    r = _choisir(_decideur(leve=exc), TROIS)
    assert r.erreur.startswith("passerelle_occupee:")
    assert not r.non_imputable


@pytest.mark.parametrize("exc", [TypeSafeAuthenticationError("401"), TypeSafeUnprocessableEntityError("422")])
def test_E2_erreur_de_configuration_se_dit_comme_telle(exc):
    r = _choisir(_decideur(leve=exc), TROIS)
    assert r.erreur.startswith("configuration:")


def test_E3_un_echec_de_transport_n_est_jamais_une_non_decision():
    r = _choisir(_decideur(leve=TypeSafeRateLimitError("429")), TROIS)
    assert r.non_imputable is False and r.index is None


def test_E4_une_reponse_200_inexploitable_est_toujours_une_non_decision():
    r = _choisir(_decideur({"option_0": 0.4, "option_1": 0.2, "option_2": 0.1}), TROIS)
    assert r.non_imputable is True


def test_E5_sans_quota_aucune_cle_reservee():
    assert DecideurTypesafe.sans_quota is True


# ── A — asynchrony (added on 2026-09-21) ────────────────────────
def test_A1_le_client_construit_est_asynchrone(monkeypatch):
    """A SYNCHRONOUS client would return the same result — while blocking the runner loop and
    cancelling `regroupement.parallelisme`. Measured: 40 decisions/minute instead of ~400."""
    from typesafe_sdk import AsyncTypeSafeClient

    monkeypatch.setenv("PROVIDER_KEYS__typesafeAI", "cle-de-test")
    dec = DecideurTypesafe(agent=AgentEspion(), modele=MODELE)
    assert isinstance(dec.client(), AsyncTypeSafeClient)


def test_A2_l_appel_est_attendu():
    """If the call were not `await`ed, `reponse` would be a coroutine and reading
    `answers` would fail — so the rule is checked with a strict asynchronous double."""
    import inspect

    dec = _decideur(P_OK)
    assert inspect.iscoroutinefunction(dec._client.system_one)
    r = _choisir(dec, TROIS)
    assert r.index is not None and r.erreur is None


def test_A3_la_cle_manquante_refuse_de_demarrer(monkeypatch):
    monkeypatch.delenv("PROVIDER_KEYS__typesafeAI", raising=False)
    dec = DecideurTypesafe(agent=AgentEspion(), modele=MODELE)
    with pytest.raises(RuntimeError, match="PROVIDER_KEYS__typesafeAI"):
        dec.client()


# ── T — the presented text ────────────────────────────────────────────────────
def test_T1_l_etat_vient_du_payload_des_bras_llm():
    agent = AgentEspion()
    dec = _decideur(P_OK, agent=agent)
    _choisir(dec, TROIS)
    assert agent.appels, "build_travel_plan_payload was not called: the presentation was reinvented"
    etat = dec._client.appels[0]["state"]
    for attendu in ("Renée, 29", "shop", "09:28", "Clear/Sunny", "afternoon 12°C"):
        assert attendu in etat


def test_T2_les_options_ne_sont_pas_dans_l_etat():
    dec = _decideur(P_OK)
    _choisir(dec, TROIS)
    etat = dec._client.appels[0]["state"]
    assert "Estimated duration: 19 minutes" not in etat


def test_T3_une_cle_par_option_dans_l_ordre_de_presentation():
    dec = _decideur(P_OK)
    _choisir(dec, TROIS)
    criteres = dec._client.appels[0]["questions"]["mode"].criteria
    assert list(criteres) == [cle_option(i) for i in range(3)]
    assert "Mode car" in criteres["option_0"]


def test_T4_les_instructions_n_emportent_ni_consigne_de_sortie_ni_schema():
    dec = _decideur(P_OK)
    _choisir(dec, TROIS)
    instr = dec._client.appels[0]["questions"]["mode"].instructions
    assert "[Output instructions]" not in instr
    assert "Expected JSON schema" not in instr and '"probabilities"' not in instr


def test_T5_le_modele_envoye_est_celui_de_la_spec():
    dec = _decideur(P_OK)
    _choisir(dec, TROIS)
    assert dec._client.appels[0]["model"] == MODELE


# ── the factory ──────────────────────────────────────────────────────────────
def test_fabrique_exige_un_agent():
    with pytest.raises(ValueError, match="LlmAgent"):
        construire_decideur(DecideurSpec(type="typesafe", modele=MODELE), agent=None)


def test_fabrique_transmet_la_variante_du_gabarit():
    dec = construire_decideur(
        DecideurSpec(type="typesafe", modele=MODELE), agent=AgentEspion(), gabarit=GABARIT
    )
    assert isinstance(dec, DecideurTypesafe)
    assert dec.variante == "prompt_expert_16"


def test_le_constructeur_refuse_aussi_un_alias():
    with pytest.raises(ValueError, match="pinned version"):
        DecideurTypesafe(agent=AgentEspion(), modele="jev-latest")


# ── V — validating a definition (defect of 2026-09-21) ────────────────
#
# The initial integration wired `typesafe` into NAMING (N3) and launching, but not into the
# guard that rejects a variant unknown to `prompts.yaml`. Consequence measured that day:
# `make experience-definir` silently filed a Jev experiment naming
# `prompt_expert_21`, a variant that did not exist yet. The failure only surfaced at the first
# decision, in the middle of a run — that is, at the worst place and the worst time.


class _JeuFactice:
    """The bare minimum `refuser_si_impossible` reads from a closed, up-to-date set."""

    nom = "jeu_t"
    clos = True
    jour_simule = "2026-03-16"
    manifest: ClassVar[dict] = {"dependances": {}}

    def verifier_population(self, _population):
        return None

    def jours_equivalents(self):
        return []


def _definition(tmp_path, decideur: dict, variante: str):
    import yaml
    from experiences import nommage as N
    from experiences.experience import charger_experience

    p = tmp_path / "exp.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "nom": "exp_t",
                "population": {"chemin": "/data/eqasim-output/population_1000_PANEL"},
                "jeu": {"nom": "jeu_t"},
                "gabarit": {"categorie": "itinary_multi_agent", "variante": variante},
                "decideur": decideur,
                "mode": "sans_simulateur",
                "calendrier": {"politique": "commune", "date": "2026-03-16", "graine": 42},
                "horizon_jours": 1,
                "memoire": False,
                "evenements": [],
                "graine_ordre": 42,
                "graine_tirage": 42,
                "regroupement": {"parallelisme": 4},
                "tolerances_horaires": dict(N.TOLERANCES_REFERENCE),
                "max_candidats": 6,
                "attente_max_s": 5,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return charger_experience(p)


def _refus(tmp_path, decideur: dict, variante: str) -> list[str]:
    from experiences.experience import refuser_si_impossible

    refus, _ = refuser_si_impossible(
        _definition(tmp_path, decideur, variante),
        _JeuFactice(),
        object(),
        dependances={},
        periodes={},
    )
    return refus


DECIDEUR_JEV = {"type": "typesafe", "modele": "jev-1.13.0", "parametres": {}}


def test_V1_une_variante_inconnue_est_refusee_pour_jev(tmp_path):
    """Jev reads a prompt: a variant missing from `prompts.yaml` must be refused BEFORE the run."""
    refus = _refus(tmp_path, DECIDEUR_JEV, "prompt_qui_nexiste_pas_0000")
    vises = [r for r in refus if "prompt" in r.lower()]
    assert vises, f"no refusal targets the prompt: {refus}"
    # The message must make it possible to ACT, as for a gateway arm: say what to do.
    assert "prompts.yaml" in vises[0] or "l'une de" in vises[0], vises[0]


def test_V2_une_variante_connue_passe(tmp_path):
    """Symmetric check: the guard must not refuse a variant that really does exist."""
    refus = _refus(tmp_path, DECIDEUR_JEV, "prompt_expert_16")
    assert not [r for r in refus if "prompt" in r.lower()], refus
