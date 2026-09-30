"""An experiment name is never silent about its population.

The root cause lies in two mechanisms that reinforce each other. The form offered
the first cohort in alphabetical order, hence v1 (fixed by R16). And
`DEFAUTS_NOMMAGE["population"]` was `population_1000_PANEL`, which made this
population **silent** in the name (rule N7): none of the 46 experiment names mentioned a
substrate, which read as "nothing to report" and meant "all on v1".

A substrate must never be implicit in the name of a measurement. The fix removes
`population` from the table of silent values: every experiment now carries its cohort in
its name, whatever it is.

An accepted consequence, and the right one: the names proposed from today on differ from
yesterday's. The table renames nothing that exists (N12) — the `nom` written in
`experience.yaml` stays authoritative — so archived definitions keep theirs.
"""

from __future__ import annotations

from experiences.nommage import DEFAUTS_NOMMAGE, nom_canonique


def _definition(chemin_population: str) -> dict:
    return {
        "population": {"chemin": chemin_population},
        "jeu": {"nom": "population_1000_PANEL_v5_20260316"},
        "decideur": {"type": "local", "nom": "aleatoire", "graine": 42},
        "calendrier": {"politique": "commune", "date": "2026-03-16", "graine": 42},
        "mode": "sans_simulateur",
        "horizon_jours": 1,
        "memoire": False,
    }


def test_r18_la_population_nest_plus_une_valeur_muette():
    """The defaults table must no longer contain a population: none is "the normal one"."""
    assert "population" not in DEFAUTS_NOMMAGE


def test_r18_le_nom_mentionne_la_cohorte_v5():
    nom = nom_canonique(_definition("data/population/population_1000_PANEL_v5"))
    assert "pop-" in nom, nom


def test_r18_le_nom_mentionne_aussi_lancienne_cohorte():
    """The central point: it is precisely that one which was invisible."""
    nom = nom_canonique(_definition("data/population/population_1000_PANEL"))
    assert "pop-" in nom, nom


def test_r18_deux_cohortes_donnent_deux_noms_differents():
    """The missing check: two substrates can no longer carry the same name.

    This is exactly what happened — 46 definitions, three spellings of the same path,
    and not one name to say which read what.
    """
    v1 = nom_canonique(_definition("data/population/population_1000_PANEL"))
    v5 = nom_canonique(_definition("data/population/population_1000_PANEL_v5"))
    assert v1 != v5, (v1, v5)
