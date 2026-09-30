"""Tests of the "bike" trait (core/bike_ownership.py).

The trait is **learned then drawn**: no downstream check can tell, by
looking at a row, a correct assignment from a biased one. What is
locked here are therefore the properties the mechanism must have in bulk, and
the boundaries it must never cross:

- **the stock sets the level, the propensity only sets the order** — `assign` assigns
  exactly `min(k, eligible)` bikes, never more because a member is a keen
  cyclist, never fewer because none is. It is the inversion to
  forbid, and it is the property easiest to break by accident;
- **the propensity biases in the right direction** — a keener cyclist is served more
  often, otherwise stage 2 is useless;
- **there is no deterministic order** — no "always the eldest", no sorting artefact
  on ties: at equal propensities, the members must be served roughly
  as often as one another;
- **the draw is deterministic** — not from an RNG, not from `hash()` (randomised per
  process): two runs, two machines, two moments give the same fleet;
- **dormant bikes exist** — a well-equipped household serves low-propensity
  members, and it is intended: a bike in the garage is a bike;
- **nothing is guessed** — an empty law, a missing resource or diverging features raise
  instead of producing « Pas de vélo », which is a plausible value and therefore undetectable;
- **the e-bike is a share of the fleet, not of holders** — and the age filter is renormalised,
  otherwise the target is missed from below.

Offline, without the PROGEDO data. The parity tests with the real resource
skip themselves when it has not been exported.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from mobility_core.bike_ownership import (
    DEFAULT_RESOURCE,
    ELECTRIC_BIKE,
    K_CLASSES,
    K_MAX,
    LABELS,
    LABELS_FR,
    MIN_AGE_ELECTRIC,
    MIN_AGE_ELIGIBLE,
    NO_BIKE,
    PLAIN_BIKE,
    PROPENSITY_BASE_FEATURES,
    RESOURCE_VERSION,
    STOCK_FEATURES,
    TRAIT_KEY,
    VAE_SHARE,
    BikeOwnershipModel,
    LogitModel,
    Member,
    address_key,
    assign,
    bike_label,
    draw_index,
    electric_probability,
    propensity_design,
    stock_design,
    uniform,
)


def _members(propensities, eligible=None, present=None) -> list[Member]:
    return [
        Member(index=i,
               propensity=p,
               eligible=True if eligible is None else eligible[i],
               present=True if present is None else present[i])
        for i, p in enumerate(propensities)
    ]


# ── Stage 2: the assignment, the core of the trait ───────────────────────────

class TestAttribution:
    """'`k` decides how many, the propensity only decides who.'"""

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_le_nombre_attribue_est_exactement_k(self, k):
        members = _members([0.9, 0.5, 0.2, 0.05])
        assert len(assign(members, k, "foyer")) == k

    def test_une_propension_ecrasante_ne_cree_pas_de_velo(self):
        """The keenest cyclist in the world does not make a 2nd bike appear."""
        members = _members([0.999999, 0.000001, 0.000001])
        assert len(assign(members, 1, "foyer")) == 1

    def test_des_propensions_nulles_nempechent_pas_de_servir(self):
        """`k` sets the level: if the household has 2 bikes, 2 members hold them, even if
        none rides. These are the dormant bikes, and it is right to represent them."""
        members = _members([0.0, 0.0, 0.0])
        assert len(assign(members, 2, "foyer")) == 2

    def test_le_surplus_de_velos_nest_porte_par_personne(self):
        """`k > eligible`: a bike without a holder does not appear in the JSON."""
        members = _members([0.5, 0.5])
        assert len(assign(members, 4, "foyer")) == 2

    def test_les_ineligibles_ne_recoivent_jamais_de_velo(self):
        """Scope of question `P20`: forbidden to assign the household bike to a
        three-year-old, even when the household has more bikes than eligible members."""
        members = _members([0.9, 0.9, 0.9], eligible=[True, False, False])
        assert assign(members, 3, "foyer") == {0}

    def test_aucun_velo_si_personne_nest_eligible(self):
        members = _members([0.9, 0.9], eligible=[False, False])
        assert assign(members, 2, "foyer") == set()

    def test_la_propension_biaise_le_service(self):
        """Over many 1-bike households, the cycling member must be served
        clearly more often — otherwise stage 2 is useless."""
        served = Counter()
        for foyer in range(2000):
            members = _members([0.8, 0.1])
            for index in assign(members, 1, f"foyer-{foyer}"):
                served[index] += 1
        assert served[0] > served[1] * 2

    def test_a_propension_egale_aucun_ordre_deterministe(self):
        """No "always the eldest": at equal propensities, the two members must
        be served roughly as often. A stable sort on the index would give 2000/0."""
        served = Counter()
        for foyer in range(2000):
            members = _members([0.4, 0.4])
            for index in assign(members, 1, f"foyer-{foyer}"):
                served[index] += 1
        assert 800 < served[0] < 1200
        assert served[0] + served[1] == 2000

    def test_les_places_absentes_peuvent_emporter_un_velo(self):
        """A nominal household of 4 with a single member present must not receive
        the household's 3 bikes: the absent seats take part in the draw."""
        present_only = _members([0.5])
        assert len(assign(present_only, 3, "foyer")) == 1

        with_absent = [Member(index=0, propensity=0.5, eligible=True, present=True)] + [
            Member(index=-1 - j, propensity=0.5, eligible=True, present=False)
            for j in range(3)
        ]
        chosen = assign(with_absent, 3, "foyer")
        assert len(chosen) == 3
        # The present member is not served systematically: it competes with the
        # three absent ones for 3 seats out of 4.
        served = sum(1 for f in range(400)
                     if 0 in assign(
                         [Member(index=0, propensity=0.5, eligible=True)] +
                         [Member(index=-1 - j, propensity=0.5, eligible=True)
                          for j in range(3)], 3, f"foyer-{f}"))
        assert 250 < served < 390

    def test_le_tirage_est_deterministe(self):
        members = _members([0.6, 0.3, 0.1])
        assert assign(members, 2, "foyer-x") == assign(members, 2, "foyer-x")

    def test_deux_foyers_ne_tirent_pas_la_meme_chose(self):
        """The household key enters the hash: otherwise all households of the file
        serve the same member rank."""
        members = _members([0.5, 0.5, 0.5])
        results = {frozenset(assign(members, 1, f"foyer-{i}")) for i in range(50)}
        assert len(results) > 1


# ── Stage 3: the bike type ───────────────────────────────────────────────────

class TestTypeDeVelo:

    def test_pas_de_vae_sous_lage_minimum(self):
        for age in (0, 5, MIN_AGE_ELECTRIC - 1):
            labels = {bike_label(f"foyer-{i}", 0, age) for i in range(200)}
            assert labels == {PLAIN_BIKE}

    def test_le_vae_existe_au_dela_de_lage_minimum(self):
        labels = {bike_label(f"foyer-{i}", 0, MIN_AGE_ELECTRIC) for i in range(500)}
        assert labels == {PLAIN_BIKE, ELECTRIC_BIKE}

    def test_la_part_de_vae_est_celle_du_parc(self):
        """7.7% of the fleet — and not 14.8%, which is the share of *equipped households*
        with an e-bike and which the old imputation applied (1.7× too many e-bikes)."""
        electric = sum(bike_label(f"foyer-{i}", 0, 40) == ELECTRIC_BIKE
                       for i in range(20000))
        assert abs(electric / 20000 - VAE_SHARE) < 0.01

    def test_le_filtre_dage_est_renormalise(self):
        """Applying `VAE_SHARE` to eligible holders only would bring the fleet BELOW the
        target, in proportion to the bikes held by children."""
        assert electric_probability(0.0) == pytest.approx(VAE_SHARE)
        assert electric_probability(0.2) == pytest.approx(VAE_SHARE / 0.8)
        # A fleet where 20% of holders are too young does reach the target.
        p = electric_probability(0.2)
        assert pytest.approx(0.8 * p, abs=1e-9) == VAE_SHARE

    def test_la_renormalisation_est_bornee(self):
        """Beyond 50% ineligible holders, the probability is not multiplied indefinitely."""
        assert electric_probability(0.99) == pytest.approx(VAE_SHARE / 0.5)
        assert electric_probability(-1.0) == pytest.approx(VAE_SHARE)

    def test_le_type_est_decorrele_du_rang_dattribution(self):
        """Salt distinct from the assignment's: otherwise e-bikes would
        systematically go to high propensities."""
        keys = [uniform(f"bike-holder:foyer:{i}") for i in range(200)]
        kinds = [uniform(f"bike-kind:foyer:{i}") for i in range(200)]
        assert keys != kinds


# ── Determinism and keys ─────────────────────────────────────────────────────

class TestDeterminisme:

    def test_la_clé_dadresse_est_stable_et_arrondie(self):
        assert address_key(43.6047, 1.4442) == address_key(43.60470004, 1.44420004)
        assert address_key(43.6047, 1.4442) != address_key(43.6048, 1.4442)

    def test_luniforme_est_dans_lintervalle_et_reproductible(self):
        for key in ("a", "foyer:12", "43.6,1.4"):
            value = uniform(key)
            assert 0.0 <= value < 1.0
            assert value == uniform(key)

    def test_luniforme_ne_depend_pas_du_hash_de_python(self):
        """Frozen value: if it moves, the whole fleet has moved, and it must be a
        deliberate act (change of `DRAW_SALT`), not a side effect."""
        assert uniform("foyer-temoin") == pytest.approx(0.2768345234349732, abs=1e-12)


# ── Draws and laws ───────────────────────────────────────────────────────────

class TestTirage:

    def test_une_loi_vide_ne_rend_rien(self):
        """Drawing from nothing would return 0 — "no bike" — hence undetectable."""
        assert draw_index([], 0.5) is None
        assert draw_index([0.0, 0.0, 0.0], 0.5) is None

    def test_le_tirage_suit_la_loi(self):
        law = [0.5, 0.3, 0.2]
        counts = Counter(draw_index(law, i / 10000) for i in range(10000))
        assert abs(counts[0] / 10000 - 0.5) < 0.01
        assert abs(counts[1] / 10000 - 0.3) < 0.01
        assert abs(counts[2] / 10000 - 0.2) < 0.01

    def test_le_tirage_ne_sort_jamais_de_lintervalle(self):
        assert draw_index([0.5, 0.5], 0.0) == 0
        assert draw_index([0.5, 0.5], 0.999999) == 1


# ── Design vectors: the training / application contract ──────────────────────

class TestDesign:

    def test_les_features_du_stock_sont_celles_du_contrat(self):
        design = stock_design(2, 1, 500.0, 3.0)
        assert set(design) == set(STOCK_FEATURES)

    def test_les_features_de_propension_sont_celles_du_contrat(self):
        design = propensity_design(2, 3, 40, "Female", "Retraité", 500.0, 3.0,
                                   ("Retraité", "Étudiant"))
        assert set(design) == set(PROPENSITY_BASE_FEATURES) | {
            "occ_Retraité", "occ_Étudiant"}

    def test_la_taille_est_ecretee(self):
        assert stock_design(4, 0, 1.0, 1.0) == stock_design(9, 0, 1.0, 1.0)

    def test_le_stock_est_ecrete_dans_la_propension(self):
        assert (propensity_design(4, 2, 30, "Male", None, 1.0, 1.0, ())
                == propensity_design(9, 2, 30, "Male", None, 1.0, 1.0, ()))

    def test_une_taille_absurde_ne_leve_pas(self):
        """A malformed population must not bring the enrichment down: the
        size is bounded from below, not rejected."""
        assert stock_design(0, None, None, 0.0)["size2"] == 0.0

    def test_un_age_absent_neutralise_les_termes_dage(self):
        design = propensity_design(1, 1, None, None, None, 1.0, 1.0, ())
        assert design["age"] == 0.0 and design["age2"] == 0.0

    def test_le_genre_est_lu_au_libelle_du_persona(self):
        assert propensity_design(1, 1, 30, "Female", None, 1.0, 1.0, ())["female"] == 1.0
        assert propensity_design(1, 1, 30, "Male", None, 1.0, 1.0, ())["female"] == 0.0
        assert propensity_design(1, 1, 30, None, None, 1.0, 1.0, ())["female"] == 0.0


# ── The served logit ─────────────────────────────────────────────────────────

class TestLogit:

    def test_un_logit_binaire_rend_une_probabilite(self):
        model = LogitModel(features=("x",), intercepts=(0.0,),
                           coefficients=((0.0,),), classes=(1,))
        assert model.probability({"x": 0.0}) == pytest.approx(0.5)
        assert model.probability({"x": 100.0}) == pytest.approx(0.5)

    def test_un_logit_binaire_ne_deborde_pas(self):
        model = LogitModel(features=("x",), intercepts=(0.0,),
                           coefficients=((1.0,),), classes=(1,))
        assert model.probability({"x": -10000.0}) == pytest.approx(0.0)
        assert model.probability({"x": 10000.0}) == pytest.approx(1.0)

    def test_un_multinomial_somme_a_un(self):
        model = LogitModel(features=("x",), intercepts=(0.0, 1.0, -1.0),
                           coefficients=((0.5,), (0.0,), (-0.5,)), classes=(0, 1, 2))
        probabilities = model.probabilities({"x": 2.0})
        assert sum(probabilities) == pytest.approx(1.0)
        assert all(p >= 0 for p in probabilities)

    def test_un_multinomial_ne_deborde_pas(self):
        model = LogitModel(features=("x",), intercepts=(0.0, 0.0),
                           coefficients=((1.0,), (-1.0,)), classes=(0, 1))
        probabilities = model.probabilities({"x": 5000.0})
        assert sum(probabilities) == pytest.approx(1.0)

    def test_une_feature_absente_vaut_zero(self):
        """Meaning of an inactive indicator — and the only case where it happens."""
        model = LogitModel(features=("a", "b"), intercepts=(0.0,),
                           coefficients=((1.0, 1.0),), classes=(1,))
        assert model.probability({"a": 1.0}) == model.probability({"a": 1.0, "b": 0.0})

    def test_des_coefficients_mal_dimensionnes_levent(self):
        with pytest.raises(ValueError):
            LogitModel(features=("a", "b"), intercepts=(0.0,),
                       coefficients=((1.0,),), classes=(1,))
        with pytest.raises(ValueError):
            LogitModel(features=("a",), intercepts=(0.0, 0.0),
                       coefficients=((1.0,),), classes=(1,))

    def test_probability_refuse_un_multinomial(self):
        model = LogitModel(features=("x",), intercepts=(0.0, 0.0),
                           coefficients=((1.0,), (0.0,)), classes=(0, 1))
        with pytest.raises(ValueError):
            model.probability({"x": 1.0})


# ── Loading: no silent fallback ──────────────────────────────────────────────

def _minimal_doc(**overrides) -> dict:
    doc = {
        "version": 1,
        "stock": {
            "features": list(STOCK_FEATURES),
            "classes": list(K_CLASSES),
            "intercepts": [0.0] * len(K_CLASSES),
            "coefficients": [[0.0] * len(STOCK_FEATURES) for _ in K_CLASSES],
        },
        "propensity": {
            "features": list(PROPENSITY_BASE_FEATURES),
            "classes": [1],
            "intercepts": [0.0],
            "coefficients": [[0.0] * len(PROPENSITY_BASE_FEATURES)],
        },
        "occupations": [],
        "median_density": 500.0,
        "under_age_holder_share": 0.16,
    }
    doc.update(overrides)
    return doc


class TestChargement:

    def test_une_ressource_absente_leve(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make bike-ownership"):
            BikeOwnershipModel.load(tmp_path / "absent.json")

    def test_une_ressource_minimale_se_charge(self, tmp_path):
        path = tmp_path / "bike.json"
        path.write_text(json.dumps(_minimal_doc()), encoding="utf-8")
        model = BikeOwnershipModel.load(path)
        assert model.stock.classes == K_CLASSES
        assert model.electric_p == pytest.approx(electric_probability(0.16))

    def test_des_features_de_stock_divergentes_levent(self, tmp_path):
        doc = _minimal_doc()
        doc["stock"]["features"] = ["autre_chose"] + list(STOCK_FEATURES[1:])
        path = tmp_path / "bike.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="stage 1"):
            BikeOwnershipModel.load(path)

    def test_des_occupations_manquantes_levent(self, tmp_path):
        """The resource declares occupations but lacks the columns: coefficients
        aligned on the wrong columns raise no error, they
        produce a wrong fleet."""
        doc = _minimal_doc(occupations=["Retraité"])
        path = tmp_path / "bike.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="stage 2"):
            BikeOwnershipModel.load(path)

    def test_des_classes_de_k_divergentes_levent(self, tmp_path):
        doc = _minimal_doc()
        doc["stock"]["classes"] = [0, 1, 2]
        doc["stock"]["intercepts"] = [0.0] * 3
        doc["stock"]["coefficients"] = [[0.0] * len(STOCK_FEATURES) for _ in range(3)]
        path = tmp_path / "bike.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="classes"):
            BikeOwnershipModel.load(path)

    def test_une_ressource_d_une_autre_version_est_refusee(self, tmp_path):
        """Same safeguard as `residence_zone.RESOURCE_VERSION`: a resource
        written for another schema must not load with misaligned
        coefficients for lack of a version check."""
        doc = _minimal_doc(version=RESOURCE_VERSION + 1)
        path = tmp_path / "bike.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            BikeOwnershipModel.load(path)


class TestContratDeSortie:

    def test_les_trois_libelles_sont_ceux_du_persona(self):
        """`traits_json` carries these exact strings, and `simulation_controller._owns_bike`
        reads them back: a one-character divergence silently deprives agents of a bike."""
        assert LABELS == ("No bike", "regular bike", "e-bike")
        # The pre-v6 labels stay declared: archived cohorts carry them.
        assert LABELS_FR == ("Pas de vélo", "vélo normal", "VAE")
        assert TRAIT_KEY == "personal_bike"

    def test_seul_pas_de_velo_est_negatif(self):
        assert NO_BIKE.lower() == "no bike"
        assert PLAIN_BIKE.lower() != "pas de vélo"
        assert ELECTRIC_BIKE.lower() != "pas de vélo"

    def test_les_bornes_dage_sont_celles_de_lenquete(self):
        assert MIN_AGE_ELIGIBLE == 5      # scope of question P20
        assert MIN_AGE_ELECTRIC == 14     # safeguard A1.a (no e-bike before age 14)
        assert K_MAX == 4


# ── Parity with the real resource (skipped if it is not exported) ────────────

@pytest.mark.skipif(not DEFAULT_RESOURCE.exists(),
                    reason="resource not exported (make bike-ownership)")
class TestRessourceReelle:

    @pytest.fixture(scope="class")
    def model(self) -> BikeOwnershipModel:
        return BikeOwnershipModel.load()

    def test_elle_se_charge_et_expose_les_deux_etages(self, model):
        assert model.stock.classes == K_CLASSES
        assert len(model.propensity.coefficients) == 1

    def test_la_loi_de_k_somme_a_un(self, model):
        for size in (1, 2, 3, 4, 6):
            probabilities = model.stock_probabilities(size, 1, 500.0, 3.0)
            assert sum(probabilities) == pytest.approx(1.0)

    def test_lequipement_croit_avec_la_taille_du_menage(self, model):
        """The gradient this model exists to correct: it was INVERTED."""
        equipped = [1.0 - model.stock_probabilities(size, 1, 500.0, 5.0)[0]
                    for size in (1, 2, 3, 4)]
        assert equipped == sorted(equipped), equipped

    def test_la_propension_est_une_probabilite(self, model):
        for age in (6, 20, 45, 80):
            p = model.propensity_of(2, 3, age, "Female", "Retraité", 500.0, 3.0)
            assert 0.0 <= p <= 1.0

    def test_la_densite_manquante_ne_casse_rien(self, model):
        """81 fine zones out of 785 have no density: the scope median is
        substituted, not zero, which would describe a desert."""
        with_none = model.stock_probabilities(2, 1, None, 3.0)
        with_median = model.stock_probabilities(2, 1, model.median_density, 3.0)
        assert with_none == pytest.approx(with_median)

    def test_la_ressource_publie_ses_cibles_et_ses_effectifs(self, model):
        """'Cell counts published with each table' — acceptance criterion."""
        targets = model.validation["targets"]
        assert 45.0 < targets["holders_pct"] < 56.0
        assert 7.0 < targets["vae_share_of_fleet_pct"] < 8.5
        for row in targets["holders_by_household_size"]:
            assert {"size", "n", "weighted_n", "thin"} <= set(row)
        for cell in model.validation["practice"]["by_k_and_size"]:
            assert {"k", "size", "n", "weighted_n", "thin"} <= set(cell)

    def test_le_gradient_publie_est_croissant_sur_les_tailles_1_a_4(self, model):
        curve = {row["size"]: row["holders_pct"]
                 for row in model.validation["targets"]["holders_by_household_size"]}
        ordered = [curve[size] for size in (1, 2, 3, 4)]
        assert ordered == sorted(ordered), ordered

    def test_la_cible_habitat_est_en_part_de_personnes_pas_de_menages(self, model):
        """The unit of the housing target, and it is a trap that has bitten.

        `personal_bike` is an **individual** trait: the target opposed to the population
        must be a share of equipped PERSONS. The published curve, for its part, is a share of
        equipped HOUSEHOLDS (« Ménages équipés, individuel isolé : 70,9 % »). Confusing the
        two produces a negative bias on ALL categories, all the stronger as
        the housing is family housing — a household of four with one bike is "equipped" but only one
        of its members holds it. The symptom is misleading: it looks like an imputation
        defect whereas it is a unit error.
        """
        reference = model.validation.get("housing_reference") or {}
        if not reference.get("attainable_on_imputed_housing"):
            pytest.skip("housing type table missing at export")
        holders = reference["attainable_on_imputed_housing"]
        households = reference.get("attainable_households_equipped_pct")
        assert households, "the resource must serve BOTH units"
        # In family housing, the share of persons is clearly below the share of
        # households. That is what distinguishes the two units; if equality set in,
        # one of the two would have been recomputed with the other's formula.
        assert holders["individuel_isole"] < households["individuel_isole"] - 3.0
        # The gap must be larger in houses (large households) than in apartments.
        gap_house = households["individuel_isole"] - holders["individuel_isole"]
        gap_flat = (households["grand_habitat_collectif"]
                    - holders["grand_habitat_collectif"])
        assert gap_house > gap_flat, (gap_house, gap_flat)
        assert reference.get("unit"), "the unit served must be documented in the resource"

    def test_la_cible_habitat_diluee_est_publiee_et_plus_plate_que_la_publiee(self, model):
        """The "71% → 38%" criterion is unreachable by construction: the persona's
        housing is itself imputed. The resource must serve the attainable target."""
        reference = model.validation.get("housing_reference") or {}
        if not reference.get("attainable_on_imputed_housing"):
            pytest.skip("housing type table missing at export")
        # `attainable_spread_pts` is an amplitude in share of HOUSEHOLDS, hence comparable
        # with the published curve, which is one too.
        attainable = reference["attainable_households_equipped_pct"]
        published = reference["published_on_observed_housing"]
        published_spread = (published["individuel_isole"]
                            - published["grand_habitat_collectif"])
        assert reference["attainable_spread_pts"] < published_spread
        assert attainable["individuel_isole"] > attainable["grand_habitat_collectif"]
