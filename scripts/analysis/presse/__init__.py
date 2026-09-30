"""Local press corpus and sign grid — ticket 059, lots 1 and 2.

Three modules, and the split is the one of the protocol:

- `grille` — the twenty expected signs, frozen before the first request;
- `lexique` — the mobility words, which make condition C3 checkable instead of declared;
- `corpus` — the texts of conditions C2, C3 and C4, and their manifest;
- `scoring` — the paired gap, the sign agreement rate and its binomial, the weighted kappa.

None of these modules calls a model. They load, check and refuse.
"""

from scripts.analysis.presse.corpus import Corpus, RefusDeCorpus, charger_corpus
from scripts.analysis.presse.grille import Grille, RefusDeGrille, charger_grille
from scripts.analysis.presse.lexique import mots_de_mobilite_trouves
from scripts.analysis.presse.scoring import (
    NON_CONCLUANT,
    AccordDeSigne,
    EcartModal,
    accord_de_signe,
    ecart_apparie,
    kappa_pondere,
    signes_observes,
)

__all__ = [
    "NON_CONCLUANT",
    "AccordDeSigne",
    "Corpus",
    "EcartModal",
    "Grille",
    "RefusDeCorpus",
    "RefusDeGrille",
    "charger_corpus",
    "accord_de_signe",
    "charger_grille",
    "ecart_apparie",
    "kappa_pondere",
    "mots_de_mobilite_trouves",
    "signes_observes",
]
