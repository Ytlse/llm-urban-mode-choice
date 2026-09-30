"""Fitted laws speak the survey's vocabulary while the persona speaks English.

What these tests protect fits in one sentence: **a category that a law does not recognise
is missing nowhere**. Its indicators stay at zero, the persona falls into the reference
category, the law returns a perfectly plausible probability, and nothing — not a counter, not
a log line — says that the covariate has disappeared.

It happened. `main_occupation` went English in the language switch; the three frozen artefacts
(`driving_license.json`, `pt_subscription.json`, `bike_ownership.json`) name their variables
`occ_Travail à plein temps`, because it is on those categories that their coefficients were
fitted. Result measured on the v6 pool: the gap of `permis_adultes` to its target went from
2.52 to 5.23 points and that of `abonnement_tc` from 2.65 to 8.45, **without a single error**.

The artefacts do not move — they are frozen measurements. It is the reading that translates.
"""

from __future__ import annotations

import logging

import pytest

from mobility_core.bike_ownership import propensity_design
from mobility_core.equipment_propensity import design_vector
from mobility_core.population_reference import occupation_enquete

# The categories as the fitted resources name them (taken from the three JSON files).
OCCUPATIONS_ENQUETE = (
    "Autre",
    "Chômeur/recherche d'emploi",
    "Personne au foyer",
    "Retraité",
    "Scolaire (jusqu'au Bac)",
    "Travail à plein temps",
    "Travail à temps partiel",
    "Étudiant",
)

# The v6 persona → the survey category. The seven pairs the population produces.
COUPLES = [
    ("Pupil (up to Baccalaureate)", "Scolaire (jusqu'au Bac)"),
    ("Student", "Étudiant"),
    ("Full-time worker", "Travail à plein temps"),
    ("Part-time worker", "Travail à temps partiel"),
    ("Unemployed / job seeker", "Chômeur/recherche d'emploi"),
    ("Homemaker", "Personne au foyer"),
    ("Retired", "Retraité"),
]


@pytest.mark.parametrize("anglais,francais", COUPLES)
def test_les_deux_vocabulaires_donnent_la_meme_modalite(anglais, francais):
    assert occupation_enquete(anglais) == francais
    # French stays recognised as is: archived cohorts read back directly.
    assert occupation_enquete(francais) == francais


@pytest.mark.parametrize("anglais,francais", COUPLES)
def test_le_vecteur_de_design_est_identique_dans_les_deux_langues(anglais, francais):
    """The covariate must weigh THE SAME, not only be recognised."""
    features = ("age10", "female") + tuple(f"occ_{o}" for o in OCCUPATIONS_ENQUETE)
    args = (42.0, "Female", None, 1.0, 500.0, 5.0, OCCUPATIONS_ENQUETE, features, 500.0)
    en = design_vector(*args[:2], anglais, *args[3:])
    fr = design_vector(*args[:2], francais, *args[3:])
    assert en == fr
    # And the expected indicator is indeed 1: an identical but ALL-ZERO vector on
    # both sides would pass this test without proving anything — that is exactly the trap.
    assert en[features.index(f"occ_{francais}")] == 1.0
    assert sum(en[2:]) == 1.0


@pytest.mark.parametrize("anglais,francais", COUPLES)
def test_la_loi_velo_lit_aussi_les_deux_vocabulaires(anglais, francais):
    en = propensity_design(1, 2, 42.0, "Female", anglais, 500.0, 5.0, OCCUPATIONS_ENQUETE)
    fr = propensity_design(1, 2, 42.0, "Female", francais, 500.0, 5.0, OCCUPATIONS_ENQUETE)
    assert en == fr
    assert en[f"occ_{francais}"] == 1.0


def test_une_modalite_inconnue_tombe_en_reference_ET_LE_DIT(caplog):
    """The fallback remains the reference category — but it stops being silent."""
    features = ("age10",) + tuple(f"occ_{o}" for o in OCCUPATIONS_ENQUETE)
    with caplog.at_level(logging.ERROR):
        vecteur = design_vector(42.0, "Female", "Conducteur de dirigeable", 1.0, 500.0, 5.0,
                                OCCUPATIONS_ENQUETE, features, 500.0)
    assert sum(vecteur[1:]) == 0.0
    assert any("[ALARME]" in r.getMessage() and "Conducteur de dirigeable" in r.getMessage()
               for r in caplog.records)


def test_une_occupation_absente_ne_leve_aucune_alarme(caplog):
    """Absence ≠ unknown category: a persona without an occupation is not an anomaly."""
    features = ("age10",) + tuple(f"occ_{o}" for o in OCCUPATIONS_ENQUETE)
    with caplog.at_level(logging.ERROR):
        design_vector(42.0, "Female", None, 1.0, 500.0, 5.0,
                      OCCUPATIONS_ENQUETE, features, 500.0)
        design_vector(42.0, "Female", "", 1.0, 500.0, 5.0,
                      OCCUPATIONS_ENQUETE, features, 500.0)
    assert not [r for r in caplog.records if "[ALARME]" in r.getMessage()]


def test_les_ressources_gelees_gardent_leur_vocabulaire():
    """Safeguard: if one day someone "translates" the artefacts, this test fails.

    Translating them is not an improvement: the variable names are part of
    the fit, and the already sealed cohorts cite the file's `sha256`.
    """
    import json

    from mobility_core.resources import data_path

    for nom in ("driving_license.json", "pt_subscription.json", "bike_ownership.json"):
        chemin = data_path(nom)
        if not chemin.exists():  # restricted-access resource missing on this machine
            pytest.skip(f"{nom} missing")
        texte = json.dumps(json.loads(chemin.read_text(encoding="utf-8")), ensure_ascii=False)
        assert "occ_Travail à plein temps" in texte, (
            f"{nom} no longer names its categories in the survey language: the "
            f"coefficients were fitted on those.")
