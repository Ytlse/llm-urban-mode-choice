"""Identity of a run — ticket 091.

WHY
---
Nothing linked a stored memory to the run that had produced it. Actual content of a resume
point of 2026-09-16: simulated day, timestamp, anchor, counters — no model, no prompt, no
population, no shock. And the resume followed the `experiments/current` link, which twice in
the same day pointed to a run other than the one believed. A `make run CONT=1` could therefore
restore the memory of ANOTHER experiment without a single line reporting it.

THE PRINCIPLE
-------------
By default nothing is reused. A resume is requested by NAMING the run, and only happens if the
experiment is the same. A differing identity makes the launch refuse, naming the faulty
field — never a silent fallback.

⚠ **No escape hatch is provided.** Debugging cases are handled by hand, outside the code: a
workaround shipped in the product would end up being used for measurement, and that is
precisely what this is meant to make impossible.

WHAT THE IDENTITY KEEPS
-----------------------
What makes two measurements comparable, and nothing else. The write time and the counters are
not part of it: they always differ and would make every resume refuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

FICHIER = "identite_run.json"

# Readable labels: a refusal must name what differs in the user's words,
# not in the code's.
LIBELLES = {
    "modeles_admis": "modèle(s)",
    "instances_admises": "instances admises",
    "routage_instances": "routage des instances par fonction",
    "variante_prompt": "variante de prompt",
    "population": "population",
    "memoire_longue": "mémoire longue",
    "auto_reflexion": "auto-réflexion",
    "choc": "choc",
    "graine_tirage": "graine de tirage",
    "graine_ordre": "graine d'ordre des options",
    "graine_meteo": "graine de météo",
    "cache_decisions": "cache de décisions",
    # EXPERIMENT settings (ticket 077, lot K). They do not live in `config.yaml` — they
    # belong to the run, which records them here. Without them, three arms with opposite
    # settings carried the same identity, and nothing made it possible to tell, afterwards,
    # under which settings a run had been played.
    "chaine_vehicules": "chaînage des véhicules",
    "verrou_retour_domicile": "verrou de retour au domicile",
    "seuil_troncature": "seuil de troncature du tirage",
    "seuil_choc": "seuil de choc en mémoire",
    # Ticket 095 — the SEVERITY MODEL belongs to the experiment. Two arms that do not give the
    # same severity to the same delay do not write the same memory.
    "retard_saturation": "forme de la composante de retard",
    "retard_gravite_max": "maximum de la composante de retard",
    "retard_ref_s": "retard de référence (s)",
    "fenetre_changements_jours": "fenêtre « ce qui a changé récemment » (jours)",
    # Ticket 095, lot A — the duration of a shock memory derives from its severity. The MODE in
    # force belongs to the identity: two arms that do not serve memories under the same rule
    # do not measure the same thing, even with an identical declared window.
    "mode_fenetre_changements": "mode de la fenêtre de changements",
    "seuil_service_changement": "seuil de service d'un souvenir de choc",
    "plancher_changement_jours": "plancher de durée d'un souvenir de choc (jours)",
    "plafond_changement_jours": "plafond de durée d'un souvenir de choc (jours)",
    "changements_max": "lignes du bloc « ce qui a changé récemment »",
    "reflexion_stm_min_entrees": "plancher d'entrées avant réflexion",
    "meteo_par_agent": "météo tirée par agent",
    # Ticket 100, lot 4 — two arms of which one lets an inhabitant's experience reach their
    # co-resident and the other does not, do not write the same memory.
    "partage_foyer": "partage de la mémoire dans le foyer",
    # 2026-09-24 — in-flight tasks bound the size of micro-batches: two arms that do not
    # merge as many agents per prompt do not ask the model the same questions.
    "taches_en_vol": "tâches de planification en vol",
    # ── Parentage (ticket 095, lot D) ──
    # A child inherits its parent's MEMORY: which parent, and what it allows itself to vary,
    # belong to its identity. Resumed under another parent, it is no longer the same
    # arm.
    "run_parent": "run parent",
    "champs_libres": "champs déclarés libres vis-à-vis du parent",
}

# Sentinel: the field did not exist in the written identity. It is not "set to None",
# it is "the run predates the field" — and the refusal message must say so, otherwise one
# looks for a configuration difference where there is only an age difference.
ABSENT = object()


class IdentiteIncompatible(RuntimeError):
    """The requested resume is not about the same experiment."""


def _empreinte_du_choc(reglages: Any) -> str:
    """The fingerprint of the DECLARED shock, read from the file and not from the registry.

    ⚠ The shock registry is only initialised at GAMA's `/init`, hence AFTER the identity is
    written at controller startup. Querying it here returned "aucun" even when a shock was
    declared — a field that never discriminates, exactly the defect this module fixes
    elsewhere for the model. The fingerprint is therefore recomputed the way `chocs.charger`
    computes it: sha256 of the file's bytes.
    """
    import hashlib

    # TWO KEYS FOR ONE VERSION, and the order is that of `_declaration_demandee()`:
    # `evenements` (ticket 100) first, `chocs` (ticket 079) second. This function read ONLY
    # the second. On 2026-09-22, the c3 campaign wrote the new key in `config.yaml` — which
    # is the canonical form — and the identity of the TREATED arm recorded "aucun": the same
    # word as its control. Two different experiments passed for one, which the docstring
    # above gives precisely as the thing not to do.
    cfg = None
    for cle in ("evenements", "chocs"):
        bloc = getattr(reglages, cle, None)
        if bloc is not None and getattr(bloc, "enabled", False) and getattr(bloc, "fichier", None):
            cfg = bloc
            break
    if cfg is None:
        return "aucun"
    chemin = getattr(cfg, "fichier", None)
    if not chemin:
        return "aucun"
    p = Path(str(chemin))
    if not p.is_file():
        # A shock declared but not found will make loading fail further on; here the point is
        # above all to refuse returning "aucun", which would pass two different experiments for one.
        return f"declare-introuvable:{p.name}"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _modeles_des_instances(reglages: Any, admises: list[str]) -> list[str]:
    """The models served by the allowed instances. Empty when nothing is restricted."""
    fournisseurs = getattr(reglages.llm, "providers", {}) or {}
    modeles = set()
    for nom in admises:
        cfg = fournisseurs.get(nom)
        modele = getattr(cfg, "default_model", None) if cfg is not None else None
        if modele:
            modeles.add(str(modele))
    return sorted(modeles)


def _routage(reglages: Any) -> tuple[list[str], dict[str, list[str]]]:
    """(allowed instances, routing per category) — ticket 095, lot C.

    `instances_admises` accepts a flat list or a `category → instances` table. A `sorted()`
    applied as is to a table would return its KEYS, that is category names presented as
    instance names: an identity that compares without ever failing, hence a check that is
    no longer one.
    """
    brut = getattr(reglages.llm, "instances_admises", None)
    if isinstance(brut, dict):
        routes = {str(k): sorted(str(v) for v in (vs or [])) for k, vs in brut.items()}
        union = sorted({i for instances in routes.values() for i in instances})
        return union, routes
    return sorted(str(v) for v in (brut or [])), {}


def _meteo_fenetre(reglages: Any) -> dict:
    try:
        from urban_mobility_agents.utils.weather_draw import fenetre_meteo_effective

        return fenetre_meteo_effective(reglages)
    except Exception as e:
        logger.warning(f"[identite] cannot compute the weather window ({e})")
        return {}


def _meteo_csv_meta() -> dict:
    try:
        from urban_mobility_agents.utils.weather_loader import metadonnees_csv_meteo

        return metadonnees_csv_meteo()
    except Exception as e:
        logger.warning(f"[identite] cannot read the weather CSV metadata ({e})")
        return {}


def composer(
    reglages: Any,
    empreinte_choc: str | None = None,
    *,
    run_parent: str = "",
    champs_libres: tuple[str, ...] = (),
) -> dict:
    """The identity of the experiment as the current configuration describes it."""
    params = dict(getattr(reglages.agent, "llm_params", {}) or {})
    admises, routes = _routage(reglages)
    return {
        # The model is not a separate setting: it is carried by the allowed instances.
        # Deriving it rather than reading a field that does not exist avoids an always-empty
        # key — a field that never discriminates gives the illusion of a check.
        "modeles_admis": _modeles_des_instances(reglages, admises),
        "instances_admises": admises,
        # Ticket 095, lot C — the function → model binding belongs to the experiment. Two arms
        # that do not route reflections to the same model do not write the same memory,
        # hence do not take the same decisions: the measured difference would stop being
        # attributable to the shock. Empty = flat list, hence no routing.
        "routage_instances": routes,
        "variante_prompt": str(params.get("prompt_variant", "")),
        "population": str(getattr(reglages.data, "population_file", "")),
        "memoire_longue": bool(getattr(reglages.agent, "long_term_memory_enabled", False)),
        "auto_reflexion": bool(
            getattr(reglages.agent, "long_term_self_reflect_enabled", False)
        ),
        "choc": empreinte_choc if empreinte_choc else _empreinte_du_choc(reglages),
        "graine_tirage": int(getattr(reglages.agent, "mode_draw_seed", 0)),
        "graine_ordre": int(getattr(reglages.agent, "option_order_seed", 0)),
        "graine_meteo": int(getattr(reglages.agent, "weather_draw_seed", 0)),
        "cache_decisions": bool(getattr(reglages.cache, "enabled", False)),
        # ── Experiment settings (lot K) ──
        "chaine_vehicules": bool(getattr(reglages.agent, "vehicle_chain_enabled", True)),
        "verrou_retour_domicile": bool(
            getattr(reglages.agent, "vehicle_return_home_lock", True)
        ),
        "seuil_troncature": float(
            getattr(reglages.agent, "mode_choice_truncation_threshold", 0.0)
        ),
        "seuil_choc": float(getattr(reglages.agent, "memoire__importance_choc", 0.7)),
        "retard_saturation": str(
            getattr(reglages.agent, "memoire__retard_saturation", "asymptote")
        ),
        "retard_gravite_max": float(
            getattr(reglages.agent, "memoire__retard_gravite_max", 0.70)
        ),
        "retard_ref_s": int(getattr(reglages.agent, "memoire__retard_ref_s", 1800)),
        "fenetre_changements_jours": int(
            getattr(reglages.agent, "memoire__fenetre_changements_jours", 14)
        ),
        "changements_max": int(getattr(reglages.agent, "memoire__changements_max", 3)),
        "mode_fenetre_changements": str(
            getattr(reglages.agent, "memoire__mode_fenetre_changements", "derivee")
        ),
        "seuil_service_changement": float(
            getattr(reglages.agent, "memoire__seuil_service_changement", 0.35)
        ),
        "plancher_changement_jours": float(
            getattr(reglages.agent, "memoire__plancher_changement_jours", 2.0)
        ),
        "plafond_changement_jours": float(
            getattr(reglages.agent, "memoire__plafond_changement_jours", 30.0)
        ),
        "reflexion_stm_min_entrees": int(
            getattr(reglages.agent, "stm_reflection_min_entries", 10)
        ),
        "meteo_par_agent": bool(getattr(reglages.agent, "weather_per_agent_dates", True)),
        "partage_foyer": bool(getattr(reglages.agent, "memoire__partage_foyer_enabled", False)),
        "taches_en_vol": int(getattr(getattr(reglages, "world", None), "worker_concurrency", 8)),
        # 2026-09-25 — exact-prompt replay space. Recorded, and checked at launch by the
        # cohort, but OUTSIDE `LIBELLES`: both arms of an A/B share it, and a run predating
        # the field must stay resumable. Replaying does not change what an arm measures,
        # only who pays for the response.
        "rejeu_ab": str(getattr(getattr(reglages, "llm", None), "rejeu_ab", "") or ""),
        "prefixe_commun": bool(getattr(getattr(reglages, "world", None), "prefixe_commun", False)),
        "rejeu_strict_avant_ts": int(getattr(getattr(reglages, "llm", None), "rejeu_strict_avant_ts", 0) or 0),
        # ── Weather (ticket 107) ──
        "meteo_fenetre_effective": _meteo_fenetre(reglages),
        "meteo_csv": _meteo_csv_meta().get("fichier"),
        "meteo_csv_sha256": _meteo_csv_meta().get("sha256"),
        "meteo_csv_debut": _meteo_csv_meta().get("debut"),
        "meteo_csv_fin": _meteo_csv_meta().get("fin"),
        # ── Parentage (lot D) ──
        "run_parent": str(run_parent or ""),
        "champs_libres": sorted(champs_libres or ()),
    }


def chemin(workdir: str | Path) -> Path:
    return Path(workdir) / FICHIER


def ecrire(workdir: str | Path, identite: dict, *, ecraser: bool = False) -> bool:
    """Writes the identity. Without `ecraser`, an identity already set is LEFT IN PLACE.

    This is the ticket's rule: the identity of a resumed run is the reference against which one
    compares, not a log rewritten at every restart. Rewriting it would make the very difference
    one is trying to detect disappear.
    """
    cible = chemin(workdir)
    if cible.exists() and not ecraser:
        # Silent until 2026-09-19, and that silence cost a campaign arm: the run directory
        # name has a MINUTE granularity, two controllers started in the same minute share it,
        # and it is the FIRST one's identity that stays. On 19 September at 16:57, that of a
        # controller launched without the experiment settings or the shock.
        logger.info(
            f"[identite] {cible.name} already set — left in place. Expected on the RESUME "
            f"of a named run; otherwise, another controller opened this directory before this one "
            f"and ITS identity is authoritative."
        )
        return False
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(
        json.dumps(identite, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    logger.info(f"[identite] identité du run écrite dans {cible.name}")
    return True


def lire(workdir: str | Path) -> dict | None:
    """The identity set, or None if missing or unreadable (both make the resume refuse)."""
    cible = chemin(workdir)
    if not cible.is_file():
        return None
    try:
        charge = json.loads(cible.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error(f"[identite] {cible.name} unreadable ({e})")
        return None
    return charge if isinstance(charge, dict) else None


def differences(attendue: dict, actuelle: dict) -> list[str]:
    """The fields that differ, in plain words. Empty list = same experiment.

    ALL differences are returned, not only the first: fixing one field only to hit the next
    one at the following launch would lose three quarters of an hour per round.
    """
    ecarts = []
    for champ, libelle in LIBELLES.items():
        a = attendue.get(champ, ABSENT)
        b = actuelle.get(champ, ABSENT)
        if a is ABSENT:
            # The resumed run predates this field: we do NOT KNOW under which value it
            # ran. Saying so, rather than "None in the resumed run", points to the right
            # cause — the run's age, not a different setting.
            ecarts.append(f"{libelle} : absent du run repris, {b!r} maintenant")
            continue
        if a != b:
            ecarts.append(f"{libelle} : {a!r} au run repris, {b!r} maintenant")
    return ecarts


def verifier(workdir: str | Path, actuelle: dict) -> None:
    """Raises `IdentiteIncompatible` if the resume is not about the same experiment."""
    attendue = lire(workdir)
    if attendue is None:
        raise IdentiteIncompatible(
            f"run {Path(workdir).name} carries no identity file "
            f"({FICHIER}): impossible to check that it is the same experiment. "
            f"A run predating ticket 091 cannot be resumed."
        )
    ecarts = differences(attendue, actuelle)
    if ecarts:
        raise IdentiteIncompatible(
            f"run {Path(workdir).name} is not the same experiment as the one requested — "
            + " ; ".join(ecarts)
        )
