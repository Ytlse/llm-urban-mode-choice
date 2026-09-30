"""Which MODEL produced a response — ticket 095, lot E.

`TaskResult.provider_used` carries the name of a gateway **instance**
(`google_gemini31_key1`), not of a model. A decision read in `app.log` therefore did not say
which model had produced it, and `llm_exchanges.jsonl` had to be cross-checked to find out —
that is, opening a second file to answer a question of prime importance when reconstructing
a run after the fact.

The model is DERIVED from the instance: `providers.yaml` declares a `default_model` per
instance, and the gateway does not override it per request. This is already how `identite_run`
builds the allowed models of a run. No new field crosses the gateway.

⚠ An unknown instance returns `"?"` and NEVER raises: logging must not bring down a decision
that succeeded.
"""

from __future__ import annotations

from settings import settings

INCONNU = "?"


def modele_de_instance(instance: str | None) -> str:
    """The model served by this gateway instance, or `?`."""
    if not instance:
        return INCONNU
    try:
        fournisseurs = getattr(settings.llm, "providers", {}) or {}
        cfg = fournisseurs.get(str(instance))
        modele = getattr(cfg, "default_model", None) if cfg is not None else None
        return str(modele) if modele else INCONNU
    except Exception:  # noqa: BLE001 — jamais au prix de l'appel qu'on journalise
        return INCONNU


def origine(instance: str | None) -> str:
    """`instance/model`, ready to paste into a log line."""
    return f"{instance or INCONNU}/{modele_de_instance(instance)}"
