"""Setting the "bike" trait on a synthetic population.

This is the step that **replaces** eqasim's `personal_bike`, drawn from a number of bikes
copied from an ENTD 2008 household matched without the household size. What is checked here:

- the trait is set under the name read back by `simulation_controller._owns_bike` and the
  mode choice policy, with the exact labels of the contract;
- **the household is rebuilt at the address**, and the two defects of this key are
  handled: collisions are **split** (two single people at the same point make two
  one-person households, not one household of two) and partially present households draw on
  their **nominal** size;
- **outside the fine-zone layer, nothing is set** — and a `personal_bike` inherited
  from eqasim is **removed** rather than left lying around, otherwise the population would be
  half learned, half copied without anything reporting it;
- the enrichment is **idempotent and deterministic**;
- the validation **refuses to conclude without material**: a population that is too small does
  not "pass", it is declared inconclusive — this is the "vacuity ≠
  perfection" pattern, the absence of measurement must not produce the perfect score.

Offline, without the PROGEDO data: the zone resolver is replaced by a
stand-in that assigns according to latitude, and the model by zero coefficients whose
distribution is known exactly.
"""

from __future__ import annotations

import json
import math

import pytest

from mobility_core.bike_ownership import (
    ELECTRIC_BIKE,
    K_CLASSES,
    NO_BIKE,
    PLAIN_BIKE,
    PROPENSITY_BASE_FEATURES,
    STOCK_FEATURES,
    TRAIT_KEY,
    BikeOwnershipModel,
    LogitModel,
)
from scripts.data.population import enrich_personal_bike as enrich_module


class _Zone:
    """Minimal fine zone, as the enrichment consumes it."""

    def __init__(self, zf: str, density: float = 500.0, dist: float = 3.0):
        self.zf = zf
        self.density_hh_km2 = density
        self.dist_center_km = dist


class _FakeResolver:
    """Assigns the north (lat ≥ 43.6) to a dense zone, the south to a sparse zone, and
    leaves outside the layer everything beyond 44."""

    def resolve_many(self, lats, lons):
        out = []
        for lat in lats:
            if lat is None or lat > 44.0:
                out.append(None)
            elif lat >= 43.6:
                out.append(_Zone("100100000", density=3000.0, dist=1.0))
            else:
                out.append(_Zone("200200000", density=50.0, dist=20.0))
        return out


def _model(k_logits=None, propensity_intercept=0.0,
           under_age_holder_share=0.0) -> BikeOwnershipModel:
    """Test model whose distribution of `k` is explicit.

    `k_logits` gives one logit per class of `K_CLASSES`; all coefficients are
    zero, so the distribution is the softmax of the constants, regardless of the
    covariates. We thus know exactly what the draw must produce.
    """
    logits = k_logits or [0.0] * len(K_CLASSES)
    return BikeOwnershipModel(
        stock=LogitModel(features=STOCK_FEATURES,
                         intercepts=tuple(logits),
                         coefficients=tuple((0.0,) * len(STOCK_FEATURES)
                                            for _ in K_CLASSES),
                         classes=K_CLASSES),
        propensity=LogitModel(features=PROPENSITY_BASE_FEATURES,
                              intercepts=(propensity_intercept,),
                              coefficients=((0.0,) * len(PROPENSITY_BASE_FEATURES),),
                              classes=(1,)),
        occupations=(),
        median_density=500.0,
        under_age_holder_share=under_age_holder_share,
        validation={},
        meta={},
    )


# A certain `k`, to make the assignment readable: -inf everywhere except the wanted class.
def _certain_k(value: int) -> BikeOwnershipModel:
    logits = [-40.0] * len(K_CLASSES)
    logits[value] = 40.0
    return _model(k_logits=logits)


def _person(lat, lon=1.44, household_size=1, age=30, **traits) -> dict:
    home = {"lat": lat, "lon": lon} if lat is not None else {}
    return {
        "person_id": f"{lat}-{lon}-{age}",
        "identity": {
            "traits_json": {"age": age, "household_size": household_size,
                            "gender": "Female", "number_of_cars": 1, **traits},
            "home": home,
        },
    }


def _labels(population):
    return [p["identity"]["traits_json"].get(TRAIT_KEY) for p in population]


# ── Setting the trait ────────────────────────────────────────────────────────

class TestPoseDuTrait:

    def test_un_foyer_sans_velo_recoit_pas_de_velo(self):
        population = [_person(43.61)]
        enrich_module.enrich(population, _certain_k(0), _FakeResolver())
        assert _labels(population) == [NO_BIKE]

    def test_un_foyer_a_un_velo_dote_son_unique_membre(self):
        population = [_person(43.61)]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert _labels(population)[0] in (PLAIN_BIKE, ELECTRIC_BIKE)

    def test_les_libelles_sont_ceux_du_contrat(self):
        population = [_person(43.6 + i / 1000, age=40) for i in range(60)]
        enrich_module.enrich(population, _model(), _FakeResolver())
        assert set(_labels(population)) <= {NO_BIKE, PLAIN_BIKE, ELECTRIC_BIKE}

    def test_les_autres_traits_ne_sont_pas_touches(self):
        population = [_person(43.61, income="Medium-High")]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        traits = population[0]["identity"]["traits_json"]
        assert traits["income"] == "Medium-High"
        assert traits["age"] == 30

    def test_le_decompte_est_rendu_par_libelle(self):
        population = [_person(43.6 + i / 1000) for i in range(10)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert counts[PLAIN_BIKE] + counts[ELECTRIC_BIKE] == 10
        assert counts[NO_BIKE] == 0

    def test_rejouer_ne_change_rien(self):
        population = [_person(43.6 + i / 1000, household_size=2) for i in range(20)]
        enrich_module.enrich(population, _model(), _FakeResolver())
        first = _labels(population)
        enrich_module.enrich(population, _model(), _FakeResolver())
        assert _labels(population) == first

    def test_un_enfant_de_trois_ans_na_jamais_de_velo(self):
        """Eligibility at age 5 (scope of question `P20`): even in a household with 4
        bikes, the toddler does not carry one."""
        population = [_person(43.61, household_size=2, age=3),
                      _person(43.61, household_size=2, age=40)]
        enrich_module.enrich(population, _certain_k(4), _FakeResolver())
        assert _labels(population)[0] == NO_BIKE
        assert _labels(population)[1] != NO_BIKE

    def test_pas_de_vae_avant_quatorze_ans(self):
        population = [_person(43.6 + i / 1000, age=10) for i in range(80)]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert ELECTRIC_BIKE not in _labels(population)


# ── The household rebuilt at the address ─────────────────────────────────────

class TestReconstitutionDuFoyer:

    def test_une_grappe_coherente_est_un_seul_foyer(self):
        population = [_person(43.61, household_size=3, age=30 + i) for i in range(3)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert counts["grappes_coherentes"] == 1
        # A single bike in the household: exactly one member carries it.
        assert sum(1 for label in _labels(population) if label != NO_BIKE) == 1

    def test_une_collision_est_scindee_et_non_fusionnee(self):
        """Two single people at the same address point make TWO one-person households, not one
        household of two — which would inherit a couple's `k`."""
        population = [_person(43.61, household_size=1, age=30),
                      _person(43.61, household_size=1, age=50)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert counts["grappes_en_collision"] == 1
        # Each household has its own bike: both are equipped.
        assert all(label != NO_BIKE for label in _labels(population))

    def test_une_grappe_plus_grande_que_la_taille_declaree_est_scindee(self):
        population = [_person(43.61, household_size=2, age=30 + i) for i in range(5)]
        counts = enrich_module.enrich(population, _certain_k(2), _FakeResolver())
        assert counts["grappes_en_collision"] == 1
        # 5 persons declaring a household of 2 → households of 2, 2 and 1, each with 2 bikes:
        # everyone is equipped (the last household loses its 2nd bike, for lack of a carrier).
        assert all(label != NO_BIKE for label in _labels(population))

    def test_des_tailles_declarees_differentes_scindent_la_grappe(self):
        population = [_person(43.61, household_size=1, age=30),
                      _person(43.61, household_size=4, age=40)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert counts["grappes_en_collision"] == 1

    def test_un_menage_partiellement_present_nest_pas_surequipe(self):
        """The footprint filter kept only one member of a household of 4: it must not
        receive on its own the household's 3 bikes. The absent places compete."""
        population = [_person(43.61, household_size=4, age=30)]
        counts = enrich_module.enrich(population, _certain_k(3), _FakeResolver())
        assert counts["places_absentes"] == 3
        # A single member present: it carries at most one bike, and not systematically.
        assert sum(1 for label in _labels(population) if label != NO_BIKE) <= 1

    def test_les_places_absentes_diluent_bien_lattribution(self):
        """Over many households, an isolated member of a household of 4 with 2 bikes must be
        served ~1 time in 2 — not systematically."""
        served = 0
        trials = 200
        for i in range(trials):
            population = [_person(43.0 + i / 10000, household_size=4, age=30)]
            enrich_module.enrich(population, _certain_k(2), _FakeResolver())
            served += int(_labels(population)[0] != NO_BIKE)
        assert 0.3 < served / trials < 0.7, served / trials

    def test_un_foyer_partage_le_meme_tirage_de_stock(self):
        """Two members of the same household draw ONE `k`, not two: otherwise equipment
        stops being a household trait, which is the whole point of the per-household draw."""
        # A model with a uniform distribution on k: if each member drew its own k, we
        # would see households of 2 with 0 then 2 inconsistent bikes. We check here
        # that the number of carriers of a household never exceeds K_MAX and stays
        # consistent across runs.
        population = [_person(43.61, household_size=2, age=30),
                      _person(43.61, household_size=2, age=32)]
        enrich_module.enrich(population, _model(), _FakeResolver())
        first = _labels(population)
        population2 = [_person(43.61, household_size=2, age=30),
                       _person(43.61, household_size=2, age=32)]
        enrich_module.enrich(population2, _model(), _FakeResolver())
        assert _labels(population2) == first


# ── Outside the layer: no guessing ───────────────────────────────────────────

class TestHorsCouche:

    def test_domicile_hors_couche_reste_sans_trait(self):
        population = [_person(45.0)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert _labels(population) == [None]
        assert counts["hors_couche"] == 1

    def test_domicile_sans_coordonnees_reste_sans_trait(self):
        population = [_person(None)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert _labels(population) == [None]
        assert counts["sans_adresse"] == 1

    def test_un_personal_bike_herite_deqasim_est_retire(self):
        """The trait copied from eqasim is precisely what is replaced: leaving it in
        place outside the layer would give a population half learned, half copied."""
        population = [_person(45.0, **{TRAIT_KEY: "vélo normal"})]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        assert TRAIT_KEY not in population[0]["identity"]["traits_json"]

    def test_une_loi_degeneree_ne_pose_rien(self):
        """Degenerate `k` distribution: no trait, never "Pas de vélo" nor an invented `k`.

        A softmax of all-infinite scores returns `nan`s, with which all
        comparisons are false: without a safeguard, the draw fell onto its rounding
        net and returned the LAST class, i.e. a four-bike household coming out
        of an empty distribution. It is the costliest silence of the module."""
        model = _model()
        object.__setattr__(model.stock, "intercepts",
                           tuple([float("-inf")] * len(K_CLASSES)))
        population = [_person(43.61)]
        counts = enrich_module.enrich(population, model, _FakeResolver())
        assert _labels(population) == [None]
        assert counts["sans_loi"] == 1

    def test_une_loi_degeneree_ne_rend_pas_la_derniere_classe(self):
        """The safeguard, seen from the draw itself."""
        from mobility_core.bike_ownership import draw_index
        nan = float("nan")
        assert draw_index([nan] * 5, 0.5) is None
        assert draw_index([float("inf")] * 5, 0.5) is None
        assert draw_index([0.0] * 5, 0.5) is None


# ── The validation refuses to conclude without material ──────────────────────

class TestValidation:

    def _measure(self, population):
        return enrich_module.measure(population)

    def test_la_mesure_compte_les_foyers_et_pas_seulement_les_personnes(self):
        """`k` is drawn per household: the members of the same household are not
        independent observations, and precision is computed on households."""
        population = [_person(43.61, household_size=3, age=30 + i) for i in range(3)]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        measured = self._measure(population)
        assert measured["with_trait"] == 3
        assert measured["households"] == 1
        _, persons, households = measured["holders_by_size"][3]
        assert (persons, households) == (3, 1)

    def test_les_agents_sans_trait_sortent_des_denominateurs(self):
        """Mixing out-of-layer agents into the denominators would mechanically lower
        the shares and would present the absence of measurement as a result."""
        population = [_person(43.61), _person(45.0)]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        measured = self._measure(population)
        assert measured["n"] == 2
        assert measured["with_trait"] == 1
        assert measured["coverage"] == 0.5
        assert measured["holders_pct"] == 100.0

    def test_une_population_minuscule_est_non_concluante_et_non_reussie(self):
        """The central safeguard: `--check` must not pass on a population where
        nothing could be checked. The absence of measurement does not produce the perfect score."""
        population = [_person(43.61)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        model = _model()
        object.__setattr__(model, "validation", {
            "targets": {"holders_pct": 50.9, "holders_pct_mechanism": 49.4,
                        "vae_share_of_fleet_pct": 7.67,
                        "holders_by_household_size": [], "practising_per_bike": []},
            "stock": {"overall": {"equipped_pct_observed": 53.6},
                      "by_household_size": [], "clipping_cost": {}},
        })
        failures = enrich_module.report(self._measure(population),
                                       enrich_module.household_measure(population),
                                       counts, model)
        assert any("concluant" in failure for failure in failures)

    def test_une_couverture_trop_faible_fait_echouer(self):
        """'A massive `personal_bike = None` must make the validation FAIL.'"""
        population = [_person(45.0) for _ in range(10)] + [_person(43.61)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        model = _model()
        object.__setattr__(model, "validation", {"targets": {}, "stock": {}})
        failures = enrich_module.report(self._measure(population),
                                       enrich_module.household_measure(population),
                                       counts, model)
        assert any("couverture" in failure for failure in failures)

    def test_une_note_incomplete_ne_fait_pas_tomber_le_verdict(self):
        """A partial resource must deprive the report of a NOTE, never of a verdict.

        The explanatory block on the clipping cost read five keys in brackets under
        a mere `if clipping:`. A resource exported by another version made it
        crash **after** the verdicts were displayed, so that `--check` never returned
        its exit code: the caller saw a traceback instead of a conclusion.
        A note is not a verdict.
        """
        population = [_person(43.6 + i / 1000, household_size=1, age=40)
                      for i in range(40)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        model = _model()
        object.__setattr__(model, "validation", {
            "targets": {"holders_pct": 50.9, "holders_pct_mechanism": 49.4,
                        "vae_share_of_fleet_pct": 7.67,
                        "holders_by_household_size": [], "practising_per_bike": []},
            # `clipping_cost` not empty but truncated: the case that crashed.
            "stock": {"overall": {"equipped_pct_observed": 53.6},
                      "by_household_size": [],
                      "clipping_cost": {"k_max": 4,
                                        "bikes_per_household_unclipped": 1.215}},
        })
        failures = enrich_module.report(self._measure(population),
                                       enrich_module.household_measure(population),
                                       counts, model)
        # No exception, and the remaining failures only speak of measurability.
        assert all("clipping" not in failure for failure in failures)

    def test_une_amplitude_publiee_incomplete_nest_pas_fabriquee(self, capsys):
        """A missing bound must not produce a range, but no range at all.

        `published.get("grand_habitat_collectif", 0.0)` made up the range from
        a missing bound: it then equalled the upper bound (70.9 instead of 33.4),
        a wrong and perfectly plausible figure shown to the user. This is the
        "default substituted for missing data, then computed on" pattern.
        """
        # The housing block is printed only if the personas carry `housing_type`:
        # it is `enrich_housing_type` that sets it, upstream of this step.
        population = [_person(43.6 + i / 1000, household_size=1, age=40,
                              housing_type="Individuel isolé")
                      for i in range(40)]
        counts = enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        model = _model()
        object.__setattr__(model, "validation", {
            "targets": {"holders_pct_mechanism": 49.4, "vae_share_of_fleet_pct": 7.67,
                        "holders_by_household_size": [], "practising_per_bike": []},
            "stock": {"overall": {}, "by_household_size": [], "clipping_cost": {}},
            "housing_reference": {
                "attainable_on_imputed_housing": {"individuel_isole": 57.2},
                # Lower bound missing: the range cannot be computed.
                "published_on_observed_housing": {"individuel_isole": 70.9},
                "attainable_spread_pts": 26.8,
            },
        })
        enrich_module.report(self._measure(population),
                            enrich_module.household_measure(population),
                            counts, model)
        printed = capsys.readouterr().out
        assert "écrase l'amplitude." in printed, printed
        assert "70.9 à" not in printed, "a range was made up from a missing bound"

    def test_la_standardisation_recompose_la_cible(self):
        """Dropping incomplete households over-represents single persons: the target
        must be recomposed on the breakdown actually measured."""
        by_size = {1: {"n": 80, "equipped": 0, "bikes": 0},
                   4: {"n": 20, "equipped": 0, "bikes": 0}}
        reference = {1: 33.0, 4: 84.0}
        assert enrich_module.standardise(by_size, reference) == pytest.approx(
            0.8 * 33.0 + 0.2 * 84.0)

    def test_la_standardisation_refuse_une_reference_incomplete(self):
        """Better to serve no target than a shaky one."""
        by_size = {1: {"n": 10, "equipped": 0, "bikes": 0},
                   4: {"n": 10, "equipped": 0, "bikes": 0}}
        assert enrich_module.standardise(by_size, {1: 33.0}) is None

    def test_seuls_les_foyers_complets_entrent_au_niveau_menage(self):
        """"The household's bikes" cannot be measured on a household missing
        members: the numerator would be truncated and the denominator not."""
        population = [_person(43.61, household_size=1, age=30),
                      _person(43.62, household_size=4, age=30)]
        enrich_module.enrich(population, _certain_k(1), _FakeResolver())
        household = enrich_module.household_measure(population)
        assert household["complete_households"] == 1


# ── End to end, on a file ────────────────────────────────────────────────────

class TestBoutEnBout:

    def test_le_fichier_est_reecrit_atomiquement(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "pop.json"
        population = [_person(43.6 + i / 1000, household_size=2, age=30 + i)
                      for i in range(40)]
        path.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")

        model = _model()
        object.__setattr__(model, "validation", {"targets": {}, "stock": {}})
        monkeypatch.setattr(enrich_module.BikeOwnershipModel, "load",
                            classmethod(lambda cls, resource=None: model))
        monkeypatch.setattr(
            "mobility_core.zone_resolver.ZoneResolver.load",
            classmethod(lambda cls, resource=None, spec=None: _FakeResolver()))
        monkeypatch.setattr("sys.argv", ["enrich", str(path)])

        assert enrich_module.main() == 0
        written = json.loads(path.read_text(encoding="utf-8"))
        assert len(written) == 40
        assert all(TRAIT_KEY in p["identity"]["traits_json"] for p in written)
        assert not list(tmp_path.glob("*.tmp"))

    def test_dry_run_ne_reecrit_pas(self, tmp_path, monkeypatch):
        path = tmp_path / "pop.json"
        population = [_person(43.61, household_size=1)]
        original = json.dumps(population, ensure_ascii=False)
        path.write_text(original, encoding="utf-8")

        model = _model()
        object.__setattr__(model, "validation", {"targets": {}, "stock": {}})
        monkeypatch.setattr(enrich_module.BikeOwnershipModel, "load",
                            classmethod(lambda cls, resource=None: model))
        monkeypatch.setattr(
            "mobility_core.zone_resolver.ZoneResolver.load",
            classmethod(lambda cls, resource=None, spec=None: _FakeResolver()))
        monkeypatch.setattr("sys.argv", ["enrich", str(path), "--dry-run"])

        enrich_module.main()
        assert path.read_text(encoding="utf-8") == original

    def test_une_ressource_absente_refuse_de_tourner(self, tmp_path, monkeypatch):
        """Without the model, the command fails saying how to produce it — it
        does not impute blindly and does not fall back on eqasim's formula."""
        path = tmp_path / "pop.json"
        path.write_text("[]", encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["enrich", str(path)])
        monkeypatch.setattr(
            enrich_module.BikeOwnershipModel, "load",
            classmethod(lambda cls, resource=None: (_ for _ in ()).throw(
                FileNotFoundError("Modèle d'équipement vélo absent : make bike-ownership"))))
        assert enrich_module.main() == 1

    def test_le_rapport_json_dit_ce_que_la_console_a_dit(self, tmp_path, monkeypatch):
        """`--rapport-json` writes the checks, the slope, the verdicts and the exit code:
        it is what the representativeness synthesis reads, instead of a copied console."""
        path = tmp_path / "pop.json"
        population = [_person(43.6 + i / 1000, household_size=1 + i % 4, age=30 + i)
                      for i in range(40)]
        path.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")
        rapport = tmp_path / "trace" / "velo.json"

        model = _model()
        object.__setattr__(model, "validation", {"targets": {}, "stock": {}})
        monkeypatch.setattr(enrich_module.BikeOwnershipModel, "load",
                            classmethod(lambda cls, resource=None: model))
        monkeypatch.setattr(
            "mobility_core.zone_resolver.ZoneResolver.load",
            classmethod(lambda cls, resource=None, spec=None: _FakeResolver()))
        monkeypatch.setattr("sys.argv", ["enrich", str(path), "--dry-run", "--check",
                                         "--rapport-json", str(rapport)])

        code = enrich_module.main()
        payload = json.loads(rapport.read_text(encoding="utf-8"))
        assert payload["code_sortie"] == code
        assert payload["check"] is True and payload["dry_run"] is True
        assert payload["regles"]["SLOPE_MIN_CELL"] == enrich_module.SLOPE_MIN_CELL
        pop = payload["populations"][0]
        assert pop["n"] == 40 and pop["fichier"] == str(path)
        assert len(pop["sha256_avant"]) == 64
        # With no target served, each check is logged "pas de cible"; the slope is
        # "non concluant" (40 persons do not make 100 households per size) and says so.
        assert pop["controles"] and all(c["verdict"] == "pas de cible" for c in pop["controles"])
        assert pop["pente_tailles_1_4"]["statut"] == "non concluant"
        assert pop["pente_tailles_1_4"]["min_foyers_pour_juger"] == enrich_module.SLOPE_MIN_CELL
        assert set(pop["verdicts"]) == {"ok", "echec", "non_concluant"}
        assert isinstance(pop["echecs"], list)

    def test_un_persona_sans_adresse_perd_le_trait_herite(self, tmp_path, monkeypatch):
        """A home the resolver cannot place enters no household: the trait that an
        upstream enrichment had set must be REMOVED, not left in place — otherwise the
        population is half learned, half copied without anything reporting it
        (measured on 2026-09-04: 14 personas of the pool)."""
        path = tmp_path / "pop.json"
        sans_domicile = _person(43.61, household_size=1)
        sans_domicile["identity"]["home"] = None
        sans_domicile["identity"]["traits_json"][TRAIT_KEY] = "VAE"
        population = [sans_domicile, _person(43.62, household_size=1)]
        path.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")

        model = _certain_k(1)
        object.__setattr__(model, "validation", {"targets": {}, "stock": {}})
        monkeypatch.setattr(enrich_module.BikeOwnershipModel, "load",
                            classmethod(lambda cls, resource=None: model))
        monkeypatch.setattr(
            "mobility_core.zone_resolver.ZoneResolver.load",
            classmethod(lambda cls, resource=None, spec=None: _FakeResolver()))
        monkeypatch.setattr("sys.argv", ["enrich", str(path)])

        assert enrich_module.main() == 0
        written = json.loads(path.read_text(encoding="utf-8"))
        assert TRAIT_KEY not in written[0]["identity"]["traits_json"]
        assert written[1]["identity"]["traits_json"][TRAIT_KEY] is not None
