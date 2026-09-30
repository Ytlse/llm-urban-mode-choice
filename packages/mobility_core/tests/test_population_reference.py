"""The framing of the surveyed population is read, complete and consistent.

This test file IS the missing reader. Previously,
`population_emc2_2023.yaml` was only mentioned in a table of the installation
docs: it looked like data, it fed no check.
The tests below fail if a framing value becomes inconsistent — and
`test_couronnes_identiques_a_cerema_values` fails if the ring categories
diverge from those the modal shares are compared with, which is the error
found in production.
"""

from pathlib import Path

import pytest
import yaml

from mobility_core.population_reference import (
    COURONNE_VERS_CEREMA,
    COURONNES,
    MIN_AGE,
    OUT_OF_PERIMETER,
    PopulationReferenceError,
    couronne_commune_counts,
    couronne_population_shares,
    find_reference,
    household_targets,
    household_weight,
    population_reference,
    survey_window,
    surveyed_weekdays,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CEREMA_VALUES = REPO_ROOT / "scripts" / "data" / "population" / "cerema_values.yaml"


@pytest.fixture(scope="module")
def reference() -> dict:
    return population_reference()


def test_le_cadrage_est_present_et_lisible():
    assert find_reference() is not None, (
        "population_emc2_2023.yaml is VERSIONED in the repository: its absence is an "
        "anomaly, not a normal case as for feature_spec.json.")


def test_plus_aucun_bloc_de_cadrage_ne_dort_en_commentaire(reference):
    """The four structuring sections are ACTIVE, not commented out."""
    for section in ("enquete", "territoire", "population", "menages_equipement_voiture"):
        assert section in reference, f"section '{section}' missing from the active framing"
    assert "methodologie" in reference["enquete"]
    assert "echantillon" in reference["enquete"]
    assert "repartition_par_territoire" in reference["population"]


def test_couronnes_identiques_a_cerema_values(reference):
    """The classification and the targets must designate the SAME territories.

    It is the central acceptance criterion: if the categories diverge,
    modal shares by zone are compared with targets that do not talk about the same
    communes, and nobody sees it.
    """
    cerema = yaml.safe_load(CEREMA_VALUES.read_text(encoding="utf-8"))
    cibles = set(cerema["parts_modales_2023"]["lieu_residence"])
    # Since the English switch, the two vocabularies DIFFER, and it is deliberate: the framing is
    # in English because it is served to the model ("Lives in: 3rd ring"), the targets stay
    # French because they cross the boundary of `prompt_calibration/`.
    #
    # What this test guards has not changed for all that: there must be a TOTAL
    # and BIJECTIVE correspondence between the two. It is the central criterion — if a framing
    # category designates no target, modal shares by zone are compared with targets that
    # do not talk about the same communes, and nobody sees it.
    cadrage = {COURONNE_VERS_CEREMA[z] for z in COURONNES}
    assert cadrage == cibles, (
        f"diverging categories: framing {sorted(cadrage)} against targets {sorted(cibles)}")
    assert len(COURONNE_VERS_CEREMA) == len(COURONNES) == len(cibles), (
        "the framing ↔ targets correspondence is no longer bijective")


def test_le_hors_perimetre_n_est_pas_une_couronne():
    assert OUT_OF_PERIMETER not in COURONNES, (
        "\"out of scope\" is a fifth category, not a ring: a home "
        "100 km from the Capitole has no EMC² target to be compared with.")


def test_communes_par_couronne(reference):
    counts = couronne_commune_counts()
    assert list(counts) == list(COURONNES)
    assert sum(counts.values()) == reference["territoire"]["perimetre_2023"]["communes"]
    assert counts["Toulouse"] == 1


def test_concentration_spatiale_cible(reference):
    shares = couronne_population_shares()
    assert set(shares) == set(COURONNES)
    assert abs(sum(shares.values()) - 100.0) < 0.01
    coeur = shares["Toulouse"] + shares["1st ring"]
    publiee = 100.0 * reference["population"]["concentration"][
        "coeur_agglomeration_toulouse_plus_1ere_couronne"]
    assert abs(coeur - publiee) < 1.0, (
        f"the breakdown by ring gives {coeur:.1f}% in the urban core, the "
        f"framing publishes {publiee:.1f}%")


def test_fenetre_d_enquete_est_automne_hiver():
    debut, fin = survey_window()
    assert debut == "2022-09-20" and fin == "2023-02-18"
    assert debut[5:] > fin[5:], (
        "the window crosses 1 January: any seasonal filter working in "
        "month-day must test \">= start OR <= end\", never a simple interval.")


def test_l_enquete_ne_compte_aucun_week_end():
    jours = surveyed_weekdays()
    assert jours == (1, 2, 3, 4, 5)
    assert max(jours) <= 5


def test_age_minimum(reference):
    assert MIN_AGE == 5
    assert reference["enquete"]["methodologie"]["age_minimum_enquete"] == MIN_AGE


def test_cibles_menage_sont_annoncees_comme_telles():
    targets = household_targets()
    assert targets["taille_moyenne_menage"] == pytest.approx(2.08, abs=0.01)
    assert targets["voitures_par_menage"] == pytest.approx(1.25, abs=0.01)
    total = (targets["sans_voiture_pct"] + targets["une_voiture_pct"]
             + targets["deux_voitures_plus_pct"])
    assert total == pytest.approx(100.0, abs=1.0)


def test_poids_menage_debiaise_la_taille():
    """Three households of sizes 1, 2 and 4: the raw mean lies, the weighted one does not."""
    menage_de_chaque = [1] + [2] * 2 + [4] * 4          # 7 persons, 3 households
    brut = sum(menage_de_chaque) / len(menage_de_chaque)
    poids = [household_weight(s) for s in menage_de_chaque]
    pondere = sum(w * s for w, s in zip(poids, menage_de_chaque)) / sum(poids)
    assert brut == pytest.approx(3.0)                    # size bias
    assert pondere == pytest.approx(7 / 3, abs=1e-9)     # true mean per household
    assert sum(poids) == pytest.approx(3.0)              # a weight of 1 per household


def test_poids_menage_sans_taille_ne_compte_pas():
    """Without household size, the person has no household base — no fallback to 1."""
    assert household_weight(None) == 0.0
    assert household_weight(0) == 0.0
    assert household_weight("") == 0.0


# ── The validator refuses, it does not fall back ──────────────────────────────

def _valide(reference: dict) -> dict:
    import copy
    return copy.deepcopy(reference)


def test_validate_refuse_des_couronnes_incompletes(reference):
    casse = _valide(reference)
    casse["territoire"]["decoupage_concentrique"][1]["communes"] = 1
    with pytest.raises(PopulationReferenceError, match="communes"):
        validate(casse)


def test_validate_refuse_des_couronnes_renommees(reference):
    casse = _valide(reference)
    casse["territoire"]["decoupage_concentrique"][0]["nom"] = "Centre"
    with pytest.raises(PopulationReferenceError, match="rings"):
        validate(casse)


def test_validate_refuse_une_repartition_qui_ne_somme_pas(reference):
    casse = _valide(reference)
    casse["population"]["repartition_par_classe_age"]["5-17_ans"] = 40
    with pytest.raises(PopulationReferenceError, match="sums"):
        validate(casse)


def test_validate_refuse_une_taille_de_menage_qui_ne_tombe_pas(reference):
    casse = _valide(reference)
    casse["population"]["totaux_perimetre_2023"]["taille_moyenne_menage"] = 3.5
    with pytest.raises(PopulationReferenceError, match="household size"):
        validate(casse)


def test_validate_refuse_un_cadrage_ampute(reference):
    casse = _valide(reference)
    del casse["population"]["totaux_perimetre_2023"]
    with pytest.raises(PopulationReferenceError, match="incomplete"):
        validate(casse)
