"""Score container of the scoring formula: `Scores` and `UNMEASURED_COMPOSITE`.

Extracted unchanged from the engine the formula was developed in (see the package docstring).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


#: Composite d'un score **non mesuré** — échec PESSIMISTE.
#:
#: Le composite est une loss : ``0.0`` en est l'OPTIMUM. Tant que ``Scores()`` valait
#: ``composite=0.0``, tout code construisant un score nu — repli d'erreur, ligne
#: importée sans scores, branche « on n'a rien pu mesurer » — fabriquait le score
#: PARFAIT et gagnait tous les argmin de la campagne : la vacuité battait la mesure.
#: :meth:`~calibration.metrics.Metric._undefined_scores` avait déjà bouché ce trou
#: pour le chemin d'éval dégénérée (A3) ; il restait ouvert partout ailleurs.
#:
#: Valeur finie (JSON/SQLite-sérialisable, ``json_extract`` et ``ORDER BY`` la
#: trient normalement) et volontairement hors d'échelle : le pire composite réel
#: possible vaut ``L1_MAX_DIM × Σ poids`` ≈ 620. Voir ce nombre dans un rapport, ce
#: n'est pas lire un mauvais score, c'est lire « rien n'a été mesuré ici ».
UNMEASURED_COMPOSITE = 1e6


class Scores(BaseModel):
    """Décomposition du score composite par dimension (↓ = meilleur).

    ⚠ ``composite`` n'a PAS ``0.0`` pour défaut (cf. :data:`UNMEASURED_COMPOSITE`) :
    un score qu'on n'a pas mesuré doit perdre, pas gagner.
    """
    composite: float = UNMEASURED_COMPOSITE
    global_: float = Field(0.0, alias="global")
    absent_penalty: float = 0.0
    age: float = 0.0
    occupation: float = 0.0
    genre: float = 0.0
    motif: float = 0.0
    distance: float = 0.0
    length_penalty: float = 0.0

    model_config = {"populate_by_name": True}

    def to_json(self) -> str:
        return json.dumps(self.model_dump(by_alias=True), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Scores":
        return cls.model_validate(d)
