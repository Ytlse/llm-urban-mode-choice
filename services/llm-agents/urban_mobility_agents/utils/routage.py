"""Which gateway instances serve which FUNCTION — ticket 095, lot C.

WHAT ALREADY EXISTED
--------------------
The plumbing is complete: `LLMRequest` carries `force_provider` and `instances_admises`, both
going through `api/routes.py`, `config/settings.py` and `worker/task_worker.py`, and the SDK
accepts `instances_admises` **per call** in the payload. What was missing fit in one place:
the caller read a single global list, and the 751 requests of the ticket 077 campaign
therefore all went to the same model family.

THE RULE
--------
`settings.llm.instances_admises` accepts a `category → instances` table in addition to a flat
list. The `defaut` key is the fallback for any unnamed category.

**NEVER `force_provider`.** `task_worker.py` only retries if `force_provider is None`: a hard
pin removes the fallback. The fourteen `HTTP 503 high demand` of the campaign went through
precisely because a second key remained. Hence the minimum of TWO instances per category,
and an alarm below it — a one-entry allowlist is a hard pin that does not say its name.
"""

from __future__ import annotations

from loguru import logger
from settings import settings

# Fallback key in the table. Without it, an unnamed category would be restricted by nothing.
CLE_DEFAUT = "defaut"
# Below this, a 503 is a hard failure: there is no second key to fail over to.
INSTANCES_MIN = 2

_ALARMES_DITES: set[str] = set()


def reinitialiser() -> None:
    """Forgets the alarms already raised. Reserved for tests."""
    _ALARMES_DITES.clear()


def table() -> dict[str, list[str]]:
    """The declared routing, as a table. Empty when a flat list is declared."""
    brut = getattr(settings.llm, "instances_admises", None)
    if isinstance(brut, dict):
        return {str(k): [str(v) for v in (vs or [])] for k, vs in brut.items()}
    return {}


def liste_plate() -> list[str]:
    """The declared flat list, or the table's `defaut`. Empty = no restriction."""
    brut = getattr(settings.llm, "instances_admises", None)
    if isinstance(brut, dict):
        return [str(v) for v in (brut.get(CLE_DEFAUT) or [])]
    return [str(v) for v in (brut or [])]


def toutes_les_instances() -> list[str]:
    """The union of everything allowed, whatever the format. Used for the run identity."""
    vues: set[str] = set(liste_plate())
    for instances in table().values():
        vues.update(instances)
    return sorted(vues)


def instances_pour(categorie: str | None) -> list[str]:
    """The instances allowed to serve this category. Empty list = no restriction.

    A category missing from the table falls back to `defaut`, never to nothing: a fallback to
    the empty list would lift the restriction at the very moment it is believed tightened.
    """
    routes = table()
    if routes:
        admises = routes.get(str(categorie)) if categorie else None
        if admises is None:
            admises = routes.get(CLE_DEFAUT) or []
    else:
        admises = liste_plate()
    admises = list(dict.fromkeys(str(a) for a in admises if a))
    if admises and len(admises) < INSTANCES_MIN:
        _alarme(
            f"min:{categorie}",
            f"[ALARME] [routage] la catégorie {categorie!r} n'a qu'UNE instance admise "
            f"({admises}) — un HTTP 503 devient alors un échec sec, sans repli. Déclarer au "
            f"moins {INSTANCES_MIN} instances, ou n'en restreindre aucune.",
        )
    return admises


def _alarme(cle: str, message: str) -> None:
    if cle in _ALARMES_DITES:
        return
    _ALARMES_DITES.add(cle)
    logger.error(message)


def journal_du_routage() -> str:
    """A readable line of the routing in force, to log at startup."""
    routes = table()
    if not routes:
        plate = liste_plate()
        return (
            f"liste plate pour toutes les catégories : {plate}" if plate
            else "aucune restriction — la cascade choisit librement"
        )
    return " · ".join(f"{cat}={instances}" for cat, instances in sorted(routes.items()))
