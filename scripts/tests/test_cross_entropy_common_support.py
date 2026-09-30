"""Cross-entropy is computed on a support common to all deciders.

The defect fixed on 2026-09-21: each decision-maker was scored on the decisions where its own
distribution left a non-zero mass on the declared mode, i.e. on a set it
chose itself by cutting more or less sharply. These tests pin the expected behaviour
so that it does not come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "scripts" / "progedo_logit"))

from audit_unitaire_058 import support_commun

# Four decisions, four classes (bike, car, transit, walk). The declared mode is `car`,
# index 1, everywhere except in d4 where it is `walk`, index 3.
PRUDENT = {  # always leaves some mass on the declared mode
    "d1": (1, [0.1, 0.6, 0.2, 0.1], 1.0),
    "d2": (1, [0.2, 0.5, 0.2, 0.1], 1.0),
    "d3": (1, [0.3, 0.4, 0.2, 0.1], 1.0),
    "d4": (3, [0.2, 0.3, 0.2, 0.3], 1.0),
}
TRANCHANT = {  # puts zero on the declared mode in d3 and d4: its two worst decisions
    "d1": (1, [0.0, 1.0, 0.0, 0.0], 1.0),
    "d2": (1, [0.1, 0.9, 0.0, 0.0], 1.0),
    "d3": (1, [1.0, 0.0, 0.0, 0.0], 1.0),
    "d4": (3, [0.0, 1.0, 0.0, 0.0], 1.0),
}
PARTIEL = {"d1": (1, [0.25, 0.25, 0.25, 0.25], 1.0)}  # does not cover the support


def resultats_vides(*noms: str) -> dict[str, dict]:
    return {nom: {"etat_execution": "terminee"} for nom in noms}


def test_le_support_est_l_intersection_pas_le_sous_ensemble_de_chacun():
    """The decisions the sharp one sets to zero drop out for EVERYONE, not for it alone."""
    resultats = resultats_vides("prudent", "tranchant")
    support = support_commun({"prudent": PRUDENT, "tranchant": TRANCHANT}, resultats)

    assert support == {"d1", "d2"}
    assert resultats["prudent"]["support_commun_notes"] == 2
    assert resultats["tranchant"]["support_commun_notes"] == 2


def test_les_deux_decideurs_sont_notes_sur_le_meme_nombre_de_decisions():
    """The original defect: 5,923 decisions vs 6,588, compared as if nothing were wrong."""
    resultats = resultats_vides("prudent", "tranchant")
    support_commun({"prudent": PRUDENT, "tranchant": TRANCHANT}, resultats)

    assert (
        resultats["prudent"]["support_commun_notes"]
        == resultats["tranchant"]["support_commun_notes"]
    )
    assert resultats["prudent"]["cel_weighted_support_commun"] is not None
    assert resultats["tranchant"]["cel_weighted_support_commun"] is not None


def test_un_plancher_ne_definit_pas_le_support_mais_y_est_note():
    """`alea` would leave thousands of zeros: it would shave the intersection for everyone."""
    notables = {"prudent": PRUDENT, "alea": TRANCHANT}
    resultats = resultats_vides("prudent", "alea")
    support = support_commun(notables, resultats)

    assert support == {"d1", "d2", "d3", "d4"}  # only `prudent` defines it
    assert resultats["alea"]["cel_weighted_support_commun"] is not None


def test_un_decideur_qui_ne_couvre_pas_le_support_ne_recoit_pas_de_valeur():
    """Scoring it on what it covers would put it back on a third subset.

    A defining one always covers the support, the intersection being taken over its own
    decisions. Only a non-defining one can miss part of it: this is the case of uniform chance on
    the real set, which lacks 74 decisions of the common support.
    """
    notables = {"prudent": PRUDENT, "tranchant": TRANCHANT, "alea": PARTIEL}
    resultats = resultats_vides("prudent", "tranchant", "alea")
    support = support_commun(notables, resultats)

    assert support == {"d1", "d2"}
    assert resultats["alea"]["cel_weighted_support_commun"] is None
    assert "support_commun_motif" in resultats["alea"]


def test_une_execution_non_terminee_ne_definit_pas_le_support():
    """A partial run covers only the start of a sample: it bounds nobody."""
    notables = {"prudent": PRUDENT, "tranchant": TRANCHANT}
    resultats = resultats_vides("prudent")
    resultats["tranchant"] = {"etat_execution": "en_cours"}
    support = support_commun(notables, resultats)

    assert support == {"d1", "d2", "d3", "d4"}
