"""Tabular model decider.

The model is a decision-maker under the same contract as
the gateway: we check that it decides like the others, seals its version, and returns an
explicit non-decision out of domain — not a silent fallback.
"""

import asyncio
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

lgb = pytest.importorskip("lightgbm")
gpd = pytest.importorskip("geopandas")

from experiences import decision as D
from experiences.decision import ContexteDecision

POLICY = REPO / "scripts/progedo_logit/mode_choice_policy.json"
pytestmark = pytest.mark.skipif(not POLICY.exists(), reason="LightGBM artefact missing")


# ── Light doubles (the decision contract, not the full chain) ────────
class FakeLoc:
    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon
        self.public_transport = True


class FakeAct:
    def __init__(self, id, purpose, loc):
        self.id, self.purpose, self.location = id, purpose, loc


class FakeIdentity:
    def __init__(self, traits, activities):
        self.traits_json, self.activities = traits, activities


class FakePerson:
    def __init__(self, pid, identity):
        self.person_id, self.identity = pid, identity


class FakePlan:
    def __init__(self, duration):
        self.duration = duration


class FakeProp:
    def __init__(self, mode, duree, code, source="enregistree"):
        self.mode, self.code, self.source = mode, code, source
        self.plan = FakePlan(duree)


TRAITS = {
    "age": 35,
    "gender": "Homme",
    "household_size": 3,
    "has_driving_license": True,
    "has_pt_subscription": False,
    "number_of_cars": 1,
    "car_availability": "always",
    "personal_bike": "Un vélo",
    "socioprofessional_class": "Employé",
    "main_occupation": "actif_temps_plein",
    "employed": True,
    "studies": False,
}
# Two points in the Toulouse urban area (inside the zone layer).
CENTRE = FakeLoc(43.6045, 1.4440)
UPS = FakeLoc(43.5610, 1.4650)


def _person():
    home = FakeAct("a0", "home", CENTRE)
    work = FakeAct("a1", "work", UPS)
    return FakePerson("p1", FakeIdentity(TRAITS, [home, work]))


def _ctx(dest_activity="a1"):
    return ContexteDecision(
        timestamp=1773731722,
        activity_id=dest_activity,
        purpose="work",
        departure_time=1773731722,
        from_location=CENTRE,
        destination=UPS,
        graine_ordre=42,
        graine_tirage=42,
        max_candidats=6,
    )


def _presentees():
    return [
        FakeProp("car", 1200, "__DIRECT_CAR__^^"),
        FakeProp("foot,bus,foot,metro,foot", 2400, "line:1^A^B"),
        FakeProp("foot", 3600, "walk^^"),
    ]


@pytest.fixture(scope="module")
def decideur():
    from experiences.decideur_modele import DecideurModele

    return DecideurModele()


def test_R11_decideur_modele_meme_trace(decideur):
    # The model response has the SAME shape as an LLM decision-maker's, and produces a
    # valid trace through the same `construire_trace`. (We call `choisir` directly:
    # upstream eligibility is tested elsewhere and requires the full vehicle chain.)
    person, ctx, props = _person(), _ctx(), _presentees()
    rep = asyncio.run(decideur.choisir(person, ctx, props))
    assert rep.index is not None and 0 <= rep.index < len(props)
    assert abs(sum(rep.poids) - 1.0) < 1e-6
    assert rep.distribution
    trace = D.construire_trace(
        person,
        ctx,
        props,
        [],
        props[rep.index],
        D.METHODE_DECIDEUR,
        rep,
        D.CONTRAINTE_AUCUNE,
    )
    assert D.valider_trace(trace) == []


def test_R11_decider_mappe_non_imputable(decideur, monkeypatch):
    # `decider()` turns a non_imputable response into a TERMINAL decision, counted,
    # with nothing selected, never retried. We bypass eligibility (tested elsewhere).
    props = _presentees()
    monkeypatch.setattr(
        D,
        "eligibilite",
        lambda *a, **k: D.ResultatFiltre(
            eligibles=props, ecartees=[], evenements=[], contrainte=D.CONTRAINTE_AUCUNE
        ),
    )
    monkeypatch.setattr(D, "plafonner", lambda elig, n: (list(elig), []))
    monkeypatch.setattr(
        D, "ordre_presentation", lambda retenues, *a, **k: list(retenues)
    )

    class StubNonImputable:
        sans_quota = True
        nom = "stub"

        async def choisir(self, person, ctx, presentees):
            return D.ReponseDecideur(
                index=None,
                fournisseur=self.nom,
                non_imputable=True,
                raison="modele_non_imputable:test",
            )

    decision = asyncio.run(D.decider(_person(), _ctx(), props, StubNonImputable()))
    assert decision.methode == D.METHODE_MODELE_NON_IMPUTABLE
    assert decision.est_decision and decision.retenue is None


def test_R12_sha_modele_scelle():
    from experiences.experience import DecideurSpec, _empreinte_decideur

    emp = _empreinte_decideur(DecideurSpec(type="modele"))
    assert emp["type"] == "modele"
    assert emp["artefact_sha256"] and len(emp["artefact_sha256"]) == 64
    # Another artefact → another file SHA.
    autre = (
        REPO
        / "docs/traces/2026-08-31_second_modele_19_features/mode_choice_policy_ref_distance.json"
    )
    if autre.exists():
        emp2 = _empreinte_decideur(
            DecideurSpec(type="modele", artefact=str(autre.relative_to(REPO)))
        )
        assert emp2["artefact_sha256"] != emp["artefact_sha256"]


def test_R13_non_decision_hors_domaine(decideur):
    # OD outside the zone layer → TERMINAL non-decision, counted, not retried,
    # with no selected mode (hence excluded from the shares).
    person = _person()
    person.identity.activities[0].location = FakeLoc(0.0, 0.0)  # Gulf of Guinea
    ctx = ContexteDecision(
        timestamp=1773731722,
        activity_id="a1",
        purpose="work",
        departure_time=1773731722,
        from_location=FakeLoc(0.0, 0.0),
        destination=FakeLoc(0.01, 0.01),
        graine_ordre=42,
        graine_tirage=42,
        max_candidats=6,
    )
    rep = asyncio.run(decideur.choisir(person, ctx, _presentees()))
    assert rep.non_imputable and rep.index is None
    assert "od_hors_couche_zones" in rep.raison


def test_R13_persona_sans_traits(decideur):
    person = FakePerson(
        "p2",
        FakeIdentity({}, [FakeAct("a0", "home", CENTRE), FakeAct("a1", "work", UPS)]),
    )
    rep = asyncio.run(decideur.choisir(person, _ctx(), _presentees()))
    assert rep.non_imputable and "persona_sans_traits" in rep.raison


def test_R14_refus_sans_dependance(tmp_path):
    from experiences.decideur_modele import DecideurModele

    with pytest.raises(FileNotFoundError, match="policy"):
        DecideurModele(artefact=tmp_path / "absent.json")


def test_R15_routage_volet_modele(decideur):
    # A run whose decision-maker is of model type is reported in part 3.
    from experiences import score as S

    synthese = {"empreintes": {"decideur": {"type": "modele", "sha256": "abc"}}}
    assert S.volet_pour(synthese) == "3"
    synthese["empreintes"]["decideur"]["type"] = "passerelle"
    assert S.volet_pour(synthese) == "1"


# ── Two families, a single decision path ───────────────────

LOGIT = REPO / "scripts/progedo_logit/mnl_model.json"


def test_le_libelle_de_famille_vient_du_format_de_l_artefact():
    """A logit run must not announce itself as LightGBM.

    The artefact SHA was enough to seal the VERSION (R12), not the FAMILY: as long as the
    decision-maker name, the fingerprint and the raw response carried a hard-coded "lightgbm",
    every comparison read from the traces pointed at the wrong model. A wrong label goes
    unnoticed, unlike a missing one.
    """
    from experiences.decideur_modele import DecideurModele

    booster = DecideurModele()
    assert booster.famille == "lightgbm"
    assert booster.nom.startswith("modele:lightgbm@")
    assert booster.empreinte()["modele"] == "lightgbm_mode_choice_policy"

    if not LOGIT.exists():
        pytest.skip("Second oracle not estimated — `make logit`")
    logit = DecideurModele(artefact="scripts/progedo_logit/mnl_model.json")
    assert logit.famille == "mnl"
    assert logit.nom.startswith("modele:mnl@")
    empreinte = logit.empreinte()
    assert empreinte["modele"] == "mnl_mode_choice_policy"
    assert empreinte["famille"] == "mnl"
    # R12: two families, two artefacts, two SHAs.
    assert empreinte["sha256"] != booster.empreinte()["sha256"]


def test_un_chemin_relatif_se_resout_depuis_la_racine_du_depot():
    """The same `experience.yaml` must point to the same model on the host and in a container."""
    from experiences.decideur_modele import DecideurModele

    if not LOGIT.exists():
        pytest.skip("Second oracle not estimated — `make logit`")
    relatif = DecideurModele(artefact="scripts/progedo_logit/mnl_model.json")
    absolu = DecideurModele(artefact=LOGIT)
    assert relatif.policy_path == absolu.policy_path == LOGIT
    # `empreinte()` needs a path relative to the root: a path relative to the current
    # directory raised a ValueError there.
    assert relatif.empreinte()["artefact"] == "scripts/progedo_logit/mnl_model.json"


def test_un_format_d_artefact_inconnu_est_refuse(tmp_path):
    """Refusal at startup rather than a decision-maker named "unknown" (R14)."""
    from experiences.decideur_modele import DecideurModele

    faux = tmp_path / "modele.json"
    faux.write_text('{"format": "arbre_magique_v9"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Format d'artefact"):
        DecideurModele(artefact=faux)


# ── Three families, still a single decision path ─────────

KLR = REPO / "scripts/progedo_logit/klr_model.json"


def test_la_troisieme_famille_se_nomme_klr():
    """KLR enters the family table without touching the decision path.

    The label stays DERIVED from the artefact format: this is what allows adding a
    family without a run announcing itself under another one's name.
    """
    from experiences.decideur_modele import DecideurModele

    if not KLR.exists():
        pytest.skip("Third family not estimated — `make klr`")
    klr = DecideurModele(artefact="scripts/progedo_logit/klr_model.json")
    assert klr.famille == "klr"
    assert klr.nom.startswith("modele:klr@")
    empreinte = klr.empreinte()
    assert empreinte["modele"] == "klr_mode_choice_policy"
    assert empreinte["famille"] == "klr"
    assert empreinte["artefact"] == "scripts/progedo_logit/klr_model.json"
    # R12: three families, three artefacts, three SHAs.
    assert empreinte["sha256"] != DecideurModele().empreinte()["sha256"]
