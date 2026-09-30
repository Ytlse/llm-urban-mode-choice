"""Refuse an itinerary computation missing from the caches of the control prefix."""

from settings import settings

_ECHEC_ROUTE_STRICT: str | None = None


def echec_route_strict() -> str | None:
    return _ECHEC_ROUTE_STRICT


def exiger_route_en_cache(departure_time: int, source: str, cle: str | None = None) -> None:
    global _ECHEC_ROUTE_STRICT
    borne = settings.llm.rejeu_strict_avant_ts or 0
    if borne and int(departure_time) < borne:
        _ECHEC_ROUTE_STRICT = (
            f"[prefixe_commun] cache {source} absent avant traitement : "
            f"départ={departure_time} borne={borne} clé={str(cle or '?')[:24]}"
        )
        raise RuntimeError(_ECHEC_ROUTE_STRICT)
