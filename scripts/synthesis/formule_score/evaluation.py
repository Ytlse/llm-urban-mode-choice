"""Reading of the options offered in a decision: `{index: mode}` from the persona section.

Extracted unchanged from the engine the formula was developed in (see the package docstring).
"""

from __future__ import annotations

import re

_OPTION_RE = re.compile(r'^- \[(\d+)\]\s+([^:]+):', re.M)


def parse_option_modes(section: str) -> dict[int, str]:
    """Extrait ``{index: mode}`` de la liste d'options d'une section persona.

    Le jeu gelé ne contient que du texte pré-rendu : c'est la seule source **locale**
    du mode de chaque option. La lire ici rend la mesure indépendante de l'étiquette
    que le LLM recopie dans sa réponse — étiquette qui, en calibration, *serait* la
    mesure (elle alimente ``categorize_mode`` donc la loss), alors qu'en production
    elle est ignorée au profit des tronçons réels de l'itinéraire.

    Renvoie ``{}`` si la section ne suit pas le format attendu ; l'appelant retombe
    alors sur l'étiquette du LLM (cf. ``decisions_from_agents``).
    """
    return {int(i): m.strip() for i, m in _OPTION_RE.findall(section or "")}


def option_modes_for_record(rec: dict) -> dict[int, str]:
    """``{index: mode}`` d'un record — pré-calculé au chargement, sinon parsé à la volée."""
    cached = rec.get("option_modes")
    if cached:
        # Un aller-retour JSON transforme les clés entières en chaînes.
        return {int(k): v for k, v in cached.items()}
    return parse_option_modes(rec.get("section", ""))
