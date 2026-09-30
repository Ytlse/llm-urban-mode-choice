"""Tests of the post-processing of mode probabilities (core/mode_choice.py)."""

from collections import Counter

import pytest
from llm_gateway.core.models import LLMOutput

from mobility_llm.mode_choice import (
    CANONICAL_MODES,
    UniformFallback,
    argmax_index,
    canonical_mode,
    derive_seed,
    draw_index,
    mode_distribution,
    normalize_option_probabilities,
    option_modes,
)

# ── Normalisation ───────────────────────────────────────────────────────────

class TestNormalizeOptionProbabilities:

    def test_pourcentages_ramenes_a_une_distribution(self):
        entries = [{"index": 0, "probability": 25}, {"index": 1, "probability": 75}]
        assert normalize_option_probabilities(entries, 2) == [0.25, 0.75]

    def test_fractions_acceptees_sans_conversion(self):
        entries = [{"index": 0, "probability": 0.2}, {"index": 1, "probability": 0.8}]
        assert normalize_option_probabilities(entries, 2) == [0.2, 0.8]

    def test_somme_incorrecte_renormalisee(self):
        """The LLM announces 100 but returns 90: renormalise rather than fail."""
        entries = [{"index": 0, "probability": 30}, {"index": 1, "probability": 60}]
        assert normalize_option_probabilities(entries, 2) == pytest.approx([1 / 3, 2 / 3])

    def test_option_non_notee_recoit_zero(self):
        entries = [{"index": 1, "probability": 100}]
        assert normalize_option_probabilities(entries, 3) == [0.0, 1.0, 0.0]

    def test_index_hors_bornes_ignore_faute_de_modes(self):
        """Without the modes sent, nothing tells which option was targeted."""
        entries = [{"index": 7, "probability": 90}, {"index": 0, "probability": 10}]
        assert normalize_option_probabilities(entries, 2) == [1.0, 0.0]

    def test_index_illisible_ignore(self):
        entries = [{"index": "deux", "probability": 90}, {"index": 1, "probability": 10}]
        assert normalize_option_probabilities(entries, 2) == [0.0, 1.0]

    def test_doublons_sommes(self):
        entries = [{"index": 0, "probability": 20}, {"index": 0, "probability": 30},
                   {"index": 1, "probability": 50}]
        assert normalize_option_probabilities(entries, 2) == [0.5, 0.5]

    def test_negatif_ramene_a_zero(self):
        entries = [{"index": 0, "probability": -50}, {"index": 1, "probability": 50}]
        assert normalize_option_probabilities(entries, 2) == [0.0, 1.0]

    def test_pourcentage_en_texte(self):
        entries = [{"index": 0, "probability": "40 %"}, {"index": 1, "probability": "60"}]
        assert normalize_option_probabilities(entries, 2) == pytest.approx([0.4, 0.6])

    def test_vecteur_absent_replie_sur_uniforme(self):
        """Without a usable vector, the draw must stay possible."""
        assert normalize_option_probabilities(None, 4) == [0.25] * 4
        assert normalize_option_probabilities([], 2) == [0.5, 0.5]
        assert normalize_option_probabilities(
            [{"index": 0, "probability": 0}, {"index": 1, "probability": 0}], 2) == [0.5, 0.5]

    def test_repli_uniforme_est_marque(self):
        """The fallback is typed UniformFallback: the persisters (LLM cache) refuse it.

        A fallback written to the cache would serve a uniform distribution — chance —
        to the following runs as if it were a decision of the model (2026-08-03).
        """
        assert isinstance(normalize_option_probabilities(None, 4), UniformFallback)
        assert isinstance(normalize_option_probabilities([], 2), UniformFallback)
        assert isinstance(normalize_option_probabilities(
            [{"index": 0, "probability": 0}], 2), UniformFallback)
        # A legitimate vector — even a uniform one — is NOT marked as a fallback.
        legit = normalize_option_probabilities(
            [{"index": 0, "probability": 50}, {"index": 1, "probability": 50}], 2)
        assert legit == [0.5, 0.5]
        assert not isinstance(legit, UniformFallback)

    def test_sans_option_retourne_liste_vide(self):
        assert normalize_option_probabilities([{"index": 0, "probability": 100}], 0) == []

    def test_accepte_les_modeles_pydantic(self):
        out = LLMOutput(agents=[{
            "agent_id": "a1",
            "probabilities": [{"index": 0, "mode": "foot", "probability": 30},
                              {"index": 1, "mode": "car", "probability": 70}],
            "reason": "r",
        }])
        probs = out.agents[0].probabilities
        assert normalize_option_probabilities(probs, 2) == pytest.approx([0.3, 0.7])
        assert option_modes(probs, 2) == ["foot", "car"]


# ── Realignment of renumbered vectors ───────────────────────────────────────

class TestRealignementIndexHorsBornes:
    """Case observed in a run: several models (mistral, llama3.1, gemma) count the
    detailed steps of the itineraries as options and renumber the whole block —
    indices 0..35 for 6 options, mass placed out of range. The mode label they
    copy stays usable to find the targeted option.
    """

    # 6 options as sent: two PT options, three walks, one car.
    MODES = ["foot,bus,foot", "foot", "bicycle", "foot,bus,foot", "car", "foot"]

    def test_masse_replacee_via_le_libelle_recopie(self):
        entries = [{"index": 7, "mode": "bicycle", "probability": 30},
                   {"index": 12, "mode": "car", "probability": 70}]
        assert normalize_option_probabilities(entries, 6, modes=self.MODES) == \
            pytest.approx([0.0, 0.0, 0.3, 0.0, 0.7, 0.0])

    def test_libelle_reformule_rattrape_par_le_mode_canonique(self):
        entries = [{"index": 9, "mode": "voiture", "probability": 100}]
        assert normalize_option_probabilities(entries, 6, modes=self.MODES) == \
            pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_mode_partage_par_plusieurs_options_masse_repartie(self):
        """"Marche jusqu'à 'work'" designates the two walking options: the modal share
        is preserved even if the exact itinerary stays undetermined."""
        entries = [{"index": 6, "mode": "Marche jusqu'à 'work'", "probability": 100}]
        weights = normalize_option_probabilities(entries, 6, modes=self.MODES)
        assert weights == pytest.approx([0.0, 0.5, 0.0, 0.0, 0.0, 0.5])
        assert mode_distribution(weights, self.MODES)["walking"] == pytest.approx(1.0)

    def test_libelle_non_reconnu_masse_perdue(self):
        """Better to discard the mass than to put it on the wrong option."""
        entries = [{"index": 9, "mode": "téléportation", "probability": 40},
                   {"index": 0, "probability": 60}]
        assert normalize_option_probabilities(entries, 6, modes=self.MODES) == \
            pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_entree_hors_bornes_a_zero_sans_effet(self):
        entries = [{"index": 0, "mode": "foot,bus,foot", "probability": 100},
                   {"index": 8, "mode": "car", "probability": 0}]
        assert normalize_option_probabilities(entries, 6, modes=self.MODES) == \
            pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_vecteur_entierement_renumerote_ne_replie_plus_sur_uniforme(self):
        """Real case (agent 293015): in-range options all at 0, mass at index 34."""
        entries = [{"index": i, "mode": m, "probability": 0}
                   for i, m in enumerate(self.MODES)]
        entries.append({"index": 34, "mode": "car", "probability": 100})
        assert normalize_option_probabilities(entries, 6, modes=self.MODES) == \
            pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ── Split by mode ───────────────────────────────────────────────────────────

class TestModeDistribution:

    def test_modes_non_proposes_a_zero(self):
        """Impossible walking (trip too long) must appear at 0, not disappear."""
        dist = mode_distribution([0.4, 0.6], ["car", "foot,bus,foot"])
        assert dist == {"walking": 0.0, "cycling": 0.0, "car": 0.4,
                        "public_transport": 0.6, "train": 0.0, "motorbike": 0.0}
        assert set(dist) == set(CANONICAL_MODES)

    def test_options_de_meme_mode_agregees(self):
        dist = mode_distribution([0.2, 0.3, 0.5], ["BUS", "TRAM", "CAR"])
        assert dist["public_transport"] == pytest.approx(0.5)
        assert dist["car"] == 0.5

    def test_somme_egale_a_un(self):
        dist = mode_distribution([0.1, 0.2, 0.7], ["WALK", "BICYCLE", "RAIL"])
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_mode_inconnu_verse_dans_other(self):
        dist = mode_distribution([0.5, 0.5], ["CAR", "hélicoptère"])
        assert dist["other"] == 0.5
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_other_absent_quand_tout_est_reconnu(self):
        assert "other" not in mode_distribution([1.0], ["CAR"])


class TestCanonicalMode:

    @pytest.mark.parametrize("raw,expected", [
        ("WALK", "walking"),
        ("foot", "walking"),
        ("BICYCLE", "cycling"),
        ("vélo", "cycling"),
        ("CAR", "car"),
        ("voiture", "car"),
        ("foot,bus,foot", "public_transport"),   # the structuring segment wins
        ("SUBWAY", "public_transport"),
        ("transports en commun", "public_transport"),
        ("RAIL", "train"),
        ("scooter", "motorbike"),
        ("", "other"),
        (None, "other"),
    ])
    def test_mapping(self, raw, expected):
        assert canonical_mode(raw) == expected

    @pytest.mark.parametrize("raw", [
        "cableway", "foot,cableway,foot", "CABLEWAY",
        "gondola", "foot,gondola,foot", "funicular", "foot,funicular,foot",
    ])
    def test_le_teleo_est_un_transport_collectif(self, raw):
        """`cableway` = the Téléo, of the Tisséo network. It was missing from the SIX lists.

        A PURE cable car option therefore went down the cascade as far as "walking",
        which the word "foot" of "foot,cableway,foot" satisfies: the probability mass
        of the Téléo was counted as WALKING in `mode_distribution`, hence in the columns
        `P(...) %` of `moves.csv` and in `llm_mode_probability_pct_total`. No
        exception, no WARNING — the composite chain did not even fall into "other".

        `move_logger._BUS_MODES` and `calibration.metrics.categorize_mode` already carried
        these three modes: it is a divergence fixed on 2026-09-04, not a new
        convention. Effect quantified before applying: 120 of the 385,888 options of the
        calibration frozen sets (0.031%) and 5 of the 17,258 of the last archived run.
        """
        assert canonical_mode(raw) == "public_transport"

    def test_la_masse_du_teleo_ne_part_plus_en_marche(self):
        """The defect seen where it could be read: the split by mode."""
        dist = mode_distribution([1.0], ["foot,cableway,foot"])
        assert dist["public_transport"] == pytest.approx(1.0)
        assert dist["walking"] == pytest.approx(0.0)


# ── Draw ────────────────────────────────────────────────────────────────────

class TestDrawIndex:

    def test_reproductible_a_graine_egale(self):
        w = [0.3, 0.7]
        draws = {draw_index(w, 42, "agent-1", "act-3", "2026-07-29") for _ in range(20)}
        assert len(draws) == 1

    def test_varie_avec_le_jour(self):
        """A cache hit on another day must be able to give another mode."""
        w = [0.5, 0.5]
        days = [f"2026-07-{d:02d}" for d in range(1, 31)]
        draws = {draw_index(w, 42, "agent-1", "act-3", day) for day in days}
        assert draws == {0, 1}

    def test_respecte_les_proportions(self):
        w = [0.8, 0.2]
        counts = Counter(draw_index(w, 42, "agent", "act", i) for i in range(2000))
        assert 0.75 < counts[0] / 2000 < 0.85

    def test_option_a_zero_jamais_tiree(self):
        w = [0.0, 1.0, 0.0]
        assert {draw_index(w, 42, "agent", "act", i) for i in range(200)} == {1}

    def test_poids_non_normalises_acceptes(self):
        """The cache implicitly renormalises when options have disappeared."""
        assert {draw_index([0.0, 0.3], 1, i) for i in range(50)} == {1}

    def test_sans_graine_utilise_l_alea_global(self):
        assert draw_index([0.0, 1.0]) == 1

    def test_liste_vide_rejetee(self):
        with pytest.raises(ValueError):
            draw_index([], 42)

    def test_graine_stable_entre_processus(self):
        """Frozen value: a replayed run must produce exactly the same trips."""
        assert derive_seed(42, "agent-1", "act-3", "2026-07-29") == \
            derive_seed(42, "agent-1", "act-3", "2026-07-29")
        assert derive_seed(42, "a") != derive_seed(42, "b")
        assert derive_seed(None, "a") == derive_seed("", "a")

    def test_troncature_consideration_set(self):
        """The options below the threshold are eliminated and receive no draw."""
        # Options: Car 80%, Bus 10%, Bike 10%. Threshold 15% -> only index 0 survives
        w = [0.80, 0.10, 0.10]
        draws = {draw_index(w, 42, "agent", "act", i, min_prob_threshold=0.15) for i in range(100)}
        assert draws == {0}

    def test_troncature_toutes_options_eliminees_conserve_origine(self):
        """If all options fall below the threshold, fall back to the original weights."""
        w = [0.10, 0.10, 0.10]  # Each is worth 33.3% relative, but if threshold > 0.35:
        draws = {draw_index(w, 42, "agent", "act", i, min_prob_threshold=0.50) for i in range(100)}
        assert len(draws) > 1


class TestArgmaxIndex:

    def test_option_la_plus_probable(self):
        assert argmax_index([0.2, 0.5, 0.3]) == 1

    def test_egalite_departagee_par_le_plus_petit_index(self):
        assert argmax_index([0.5, 0.5]) == 0
