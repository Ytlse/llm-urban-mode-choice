"""Data model, canonical naming and persistence for the memory experiments — Ticket 109.

AUTHOR'S DECISIONS (2026-09-24):
  (D1) Dedicated tab in the Streamlit dashboard: scripts/dashboard/memoire.py.
  (D2) 6 configurable cognitive functions (itinerary, judgement, STM, LTM, surveys, household
  relay — ticket 111)
       with persistence of the form choices from one session to the next.
  (D3) Consecutive A/B counterfactual orchestration: treated arm (with event)
       then matched control arm (without event). The experiment is only finished once both
       have been played.
  (D4) Canonical name computed from the parameters (rule N1) without a free field.
  (D5) Strict selection from the existing catalogue in config/evenements/.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Root of the repository (this file lives in services/llm-agents/experiences/)
REPO_ROOT = Path(__file__).resolve().parents[3]
DOSSIER_MEMOIRE = (
    Path(os.getenv("EXPERIENCES_MEMOIRE_DIR"))
    if os.getenv("EXPERIENCES_MEMOIRE_DIR")
    else REPO_ROOT / "data" / "experiences" / "evenements_non_tabules"
)
DOSSIER_EVENEMENTS = REPO_ROOT / "services" / "llm-agents" / "config" / "evenements"
DOSSIER_CHOCS = REPO_ROOT / "services" / "llm-agents" / "config" / "chocs"
ETAT_FORMULAIRE_MEMOIRE = (
    REPO_ROOT / "experiments" / ".dashboard" / "formulaire_memoire.yaml"
)
PROVIDERS_YAML = REPO_ROOT / "config" / "llm_gateway" / "providers.yaml"

# The exposed cognitive functions (D2); the sixth, `evenement_relais`, comes from ticket 111.
CATEGORIES_COGNITIVES = (
    "itinary_multi_agent",
    "evenement_jugement",
    "stm_reflection",
    "ltm_self_reflection",
    "enquete_affinite",
    "evenement_relais",
)

LIBELLES_CATEGORIES: dict[str, str] = {
    "itinary_multi_agent": "1. Choix modal / Itinéraire",
    "evenement_jugement": "2. Jugement d'événement (injection)",
    "stm_reflection": "3. Mémoire court terme / Soir (STM)",
    "ltm_self_reflection": "4. Auto-réflexion long terme (LTM)",
    "enquete_affinite": "5. Enquêtes d'affinité / Perception",
    "evenement_relais": "6. Transmission au foyer (relais du lecteur)",
}

DESCRIPTIONS_CATEGORIES: dict[str, str] = {
    "itinary_multi_agent": "Décision de transport quotidienne à chaque déplacement.",
    "evenement_jugement": "Évalue sévérité et valence au moment où l'agent vit ou lit l'événement.",
    "stm_reflection": "Consolidation nocturne, formation des croyances et récit du soir au foyer.",
    "ltm_self_reflection": "Synthèse périodique (par défaut tous les 3 jours, sur les 5 derniers jours) des croyances et concepts en mémoire longue.",
    "enquete_affinite": "Réponse aux questionnaires d'affinité modale quotidiens ou périodiques.",
    "evenement_relais": "Le lecteur écrit ce qu'il dit de l'article à chaque membre de son foyer (un appel par foyer exposé, ticket 111).",
}

# Cohorte de référence historique (Ticket 077/095)
COHORTE_10_PERSONAS = [
    "899549",  # Corinne
    "1250941",  # Victoire
    "1016834",
    "1090400",
    "1165279",
    "1172810",
    "1254308",
    "1261543",
    "1298811",
    "1314647",
]

ETATS_EXPERIENCE = (
    "en_cours",
    "traite_ok",
    "partielle",
    "terminee",
    "suspendue",
    "echec",
    "arretee",
)


def slugify(texte: str) -> str:
    """Normalises a string for the canonical name: lower case, dashes, no spaces."""
    s = str(texte).strip().lower()
    s = re.sub(r"[^\w.\-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def raccourcir_modele(modele: str) -> str:
    """Abbreviates frequent model names to keep the canonical name compact."""
    m = slugify(modele)
    m = m.replace("gemini-3.8-flash", "gem38f")
    m = m.replace("gemini-3.1-flash-lite", "gem31flite")
    m = m.replace("gemini-3.5-flash", "gem35f")
    m = m.replace("gpt-5.6-luna", "gpt56luna")
    m = m.replace("qwen-3.8-27b", "qwen38-27b")
    return m


def raccourcir_evenement(evenement: str) -> str:
    """Abbreviates the event identifier for the canonical name."""
    e = slugify(evenement)
    for pfx in (
        "c6-voiture-suspecte",
        "c3-panne-reseau",
        "c1-bouchon-rocade",
        "c2-crevaison",
        "c4-train-supprime",
        "c5-orage-grele",
    ):
        if e.startswith(pfx):
            return pfx[:2] + e[len(pfx) :]
    for pfx in (
        "a13-punaises-metro",
        "a07-greve-eboueurs",
        "a09-vent-autan",
        "a18-la-machine",
        "a25-velotoulouse",
    ):
        if e.startswith(pfx):
            return pfx[:3] + e[len(pfx) :]
    return e


def nom_canonique(
    canal: str,
    evenement: str,
    modele_decision: str,
    population: str,
    horizon_jours: int,
    indice: int = 1,
    partage_foyer: bool = False,
) -> str:
    """Computes the canonical name of a memory experiment (Rule N1 & D4).

    Format: exp_mem_<canal>_<evt>_<modele>_<pop>_<horizon>j[_foyer][_<indice>]

    `_foyer` marks the sharing of memory within the household (ticket 100, batch 4). Without it, two
    experiments that only differ by this setting would only be told apart by their index.
    """
    tag_canal = "choc" if canal == "vecu" else "presse"
    tag_evt = raccourcir_evenement(evenement) or "sans-evt"
    tag_mod = raccourcir_modele(modele_decision) or "mod"
    tag_pop = slugify(population).replace(".json", "")
    if len(tag_pop) > 16:
        tag_pop = tag_pop[:16]
    base = f"exp_mem_{tag_canal}_{tag_evt}_{tag_mod}_{tag_pop}_{horizon_jours}j"
    if partage_foyer:
        base += "_foyer"
    if indice > 1:
        return f"{base}_{indice}"
    return base


def trouver_dossier_experience(nom: str) -> Path | None:
    """Finds the folder of a memory experiment (direct or in a chocs/presse subfolder)."""
    direct = DOSSIER_MEMOIRE / nom
    if direct.is_dir():
        return direct
    if DOSSIER_MEMOIRE.is_dir():
        for f in DOSSIER_MEMOIRE.rglob("experience_memoire.yaml"):
            if "archive" in f.parts or ".system_generated" in f.parts:
                continue
            if f.parent.name == nom:
                return f.parent
    legacy = REPO_ROOT / "data" / "experiences_memoire"
    if legacy.is_dir() and legacy != DOSSIER_MEMOIRE:
        if (legacy / nom).is_dir():
            return legacy / nom
        for f in legacy.rglob("experience_memoire.yaml"):
            if "archive" in f.parts or ".system_generated" in f.parts:
                continue
            if f.parent.name == nom:
                return f.parent
    return None


def sous_dossier_categorie(config: dict[str, Any]) -> Path:
    """Determines the target path according to the type of memory event."""
    nom = str(config.get("nom", ""))
    canal = str(config.get("canal", ""))
    evt = str(config.get("evenement", ""))

    if "choc" in nom.lower() or canal == "vecu" or evt.lower().startswith("c"):
        cat = "chocs_reseau"
        if "c1" in nom.lower() or evt.lower() == "c1":
            evt_dir = "C1_bouchon_rocade_42j"
        elif "c2" in nom.lower() or evt.lower() == "c2":
            evt_dir = "C2_crevaison_42j"
        elif "c6" in nom.lower() or evt.lower() == "c6":
            evt_dir = "C6_voiture_suspecte_42j"
        else:
            evt_dir = evt or "autres_chocs"
    else:
        cat = "articles_presse"
        if "a09" in nom.lower() or "vent" in nom.lower() or evt.lower() == "a09":
            evt_dir = "A09_vent_autan_15j"
        elif "a13" in nom.lower() or "punaises" in nom.lower() or evt.lower() == "a13":
            evt_dir = "A13_punaises_metro_25j"
        else:
            evt_dir = evt or "autres_articles"
    return DOSSIER_MEMOIRE / cat / evt_dir


def nom_disponible(
    canal: str,
    evenement: str,
    modele_decision: str,
    population: str,
    horizon_jours: int,
    nom_existant: str | None = None,
    partage_foyer: bool = False,
) -> str:
    """Finds an available canonical name in data/experiences_memoire/."""
    base = nom_canonique(
        canal,
        evenement,
        modele_decision,
        population,
        horizon_jours,
        partage_foyer=partage_foyer,
    )
    if nom_existant and nom_existant.startswith(base):
        return nom_existant
    if not trouver_dossier_experience(base):
        return base
    idx = 2
    while trouver_dossier_experience(f"{base}_{idx}"):
        idx += 1
    return f"{base}_{idx}"


attribuer_nom = nom_disponible


def decrire_population(population: str) -> dict[str, int]:
    """Size and number of households of at least two members, read from `data/population/`.

    Used to cost an experiment and to refuse household sharing where it has no purpose.
    `cohorte_10` is the list of ten single-person personas of the sequential replay: ten agents, no
    shared household. An identifier without a folder counts as one agent.
    """
    if population == "cohorte_10":
        return {"agents": 10, "foyers_partages": 0}
    racine = REPO_ROOT / "data" / "population"
    for dossier in (racine / population, racine / f"population_1_{population}"):
        fichier = dossier / "population.json"
        if fichier.is_file():
            data = json.loads(fichier.read_text(encoding="utf-8"))
            membres: dict[str, int] = {}
            for p in data:
                foyer = (
                    (p.get("household") or {}).get("id")
                    if isinstance(p, dict)
                    else None
                )
                if foyer is not None:
                    membres[str(foyer)] = membres.get(str(foyer), 0) + 1
            return {
                "agents": len(data),
                "foyers_partages": sum(1 for n in membres.values() if n >= 2),
            }
    return {"agents": 1, "foyers_partages": 0}


def catalogue_evenements(canal: str | None = None) -> list[dict[str, Any]]:
    """Lists all the events of the catalogue (config/evenements/ and chocs).

    Filters by channel ('vecu' or 'lu') if requested (D5).
    """
    evts: list[dict[str, Any]] = []
    dossiers = [DOSSIER_EVENEMENTS]
    if DOSSIER_CHOCS.is_dir():
        dossiers.append(DOSSIER_CHOCS)

    vus: set[str] = set()
    for dossier in dossiers:
        if not dossier.is_dir():
            continue
        for f in sorted(dossier.glob("*.yaml")):
            stem = f.stem
            if stem in vus or "__banc" in stem:  # variantes de banc (__banc, __banc_juge, __banc_reprise…)
                continue
            vus.add(stem)
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            c = (
                str(data.get("canal") or ("vecu" if stem.startswith("c") else "lu"))
                .strip()
                .lower()
            )
            if canal and c != canal:
                continue
            libelle = str(data.get("libelle") or stem)
            evts.append(
                {
                    "id": stem,
                    "canal": c,
                    "moment": str(
                        data.get("moment") or ("arrivee" if c == "vecu" else "reveil")
                    ),
                    "libelle": libelle,
                    "source": str(data.get("source") or ""),
                    "fichier": str(f.relative_to(REPO_ROOT)),
                    "jugement": str(data.get("jugement") or "aucun"),
                }
            )
    return evts


def defauts() -> dict[str, Any]:
    """Default values of the memory experiment form."""
    return {
        "canal": "vecu",
        "evenement": "c6_voiture_suspecte",
        # 3.1 / 3.5 split of 2026-09-24, the same as the cohort default: the STM
        # (49% of the requests measured on 861500) alone on 3.5, everything else on 3.1
        # (decision 43%, surveys 5%, LTM 3%). `gemini-3.8-flash` only has 20 requests per
        # day: a default no experiment could keep up with.
        "modeles": {
            "itinary_multi_agent": "gemini-3.1-flash-lite",
            "evenement_jugement": "gemini-3.1-flash-lite",
            "stm_reflection": "gemini-3.5-flash-lite",
            "ltm_self_reflection": "gemini-3.1-flash-lite",
            "enquete_affinite": "gemini-3.1-flash-lite",
            # Ticket 111 — same model as the default judgement: both calls bear on
            # the same article, at the same instant.
            "evenement_relais": "gemini-3.1-flash-lite",
        },
        "temperature_decision": 0.0,
        "variante_prompt": "prompt_expert_05",
        "population": "899549",
        "horizon_jours": 42,
        "arret_sur_extinction": False,
        "jours_apres_extinction": 5,
        "graine_calendrier": 42,
        "graine_tirage": 42,
        "memoire_importance_choc": 0.70,
        "stm_reflection_min_entries": 5,
        "contrefactuel_ab": True,
        "partage_foyer": False,
        # 2026-09-25 — exact-prompt replay between the two arms (llm_gateway/core/rejeu_ab.py).
        # True for a NEW experiment; a declaration without the key runs without replay.
        "rejeu_ab": True,
        # A/B common prefix lock. Since 2026-09-28, every memory experiment carries it:
        # the orchestrator refuses a declaration that does not state it.
        "prefixe_commun": True,
        "adaptateur": None,
    }


def charger_etat_formulaire() -> dict[str, Any]:
    """Recharge la dernière configuration enregistrée ou jouée (D2)."""
    base = defauts()
    if ETAT_FORMULAIRE_MEMOIRE.is_file():
        try:
            d = yaml.safe_load(ETAT_FORMULAIRE_MEMOIRE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                merged = {**base, **d}
                if "modeles" in d and isinstance(d["modeles"], dict):
                    merged["modeles"] = {**base.get("modeles", {}), **d["modeles"]}
                return merged
        except Exception as e:
            logger.warning(
                f"formulaire_memoire.yaml illisible ({e}) — retour aux défauts"
            )
    return base


def sauver_etat_formulaire(valeurs: dict[str, Any]) -> bool:
    """Persists the current choices for the next session (D2). True if changed."""
    ETAT_FORMULAIRE_MEMOIRE.parent.mkdir(parents=True, exist_ok=True)
    try:
        contenu = yaml.safe_dump(valeurs, allow_unicode=True, sort_keys=False)
        if (
            ETAT_FORMULAIRE_MEMOIRE.is_file()
            and ETAT_FORMULAIRE_MEMOIRE.read_text(encoding="utf-8") == contenu
        ):
            return False
        ETAT_FORMULAIRE_MEMOIRE.write_text(contenu, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"Impossible de sauver formulaire_memoire.yaml : {e}")
        return False


def adaptateurs_disponibles(providers_yaml_path: Path | None = None) -> list[str]:
    """The providers (`adapter`) declared in providers.yaml, sorted."""
    f = providers_yaml_path or PROVIDERS_YAML
    try:
        p_data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Erreur lecture providers.yaml : {e}")
        return []
    providers = p_data.get("providers", p_data) if isinstance(p_data, dict) else {}
    return sorted(
        {
            str(cfg["adapter"])
            for cfg in providers.values()
            if isinstance(cfg, dict) and cfg.get("adapter") and cfg.get("default_model")
        }
    )


def modeles_servis(
    adaptateur: str, providers_yaml_path: Path | None = None
) -> set[str]:
    """The models that at least one instance of this provider serves."""
    f = providers_yaml_path or PROVIDERS_YAML
    try:
        p_data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    providers = p_data.get("providers", p_data) if isinstance(p_data, dict) else {}
    return {
        str(cfg["default_model"]).strip()
        for cfg in providers.values()
        if isinstance(cfg, dict)
        and cfg.get("default_model")
        and str(cfg.get("adapter", "")).strip() == adaptateur
    }


def resoudre_instances_admises(
    modeles_dict: dict[str, str],
    providers_yaml_path: Path | None = None,
    adaptateur: str | None = None,
) -> dict[str, list[str]]:
    """Converts a category -> model dictionary into an instances_admises routing table.

    Uses the declarations of providers.yaml to find the key instances.
    Each category receives at least the instances serving its model.

    `adaptateur` (e.g. `groq`) only keeps the instances of this provider. The same model name
    can be served by two providers — `qwen/qwen3.8-27b` is served by Groq AND by LM Studio —
    and without this filter an experiment declared « all Groq » would send part of its calls
    elsewhere. A category that the filter leaves empty is returned empty: it is up to the caller to
    refuse it, not to this function to fill it on its own authority.
    """
    f = providers_yaml_path or PROVIDERS_YAML
    instances_par_modele: dict[str, list[str]] = {}
    if f.is_file():
        try:
            p_data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            providers = (
                p_data.get("providers", p_data) if isinstance(p_data, dict) else {}
            )
            for inst_nom, cfg in providers.items():
                if isinstance(cfg, dict) and cfg.get("default_model"):
                    if adaptateur and str(cfg.get("adapter", "")).strip() != adaptateur:
                        continue
                    mod = str(cfg["default_model"]).strip()
                    instances_par_modele.setdefault(mod, []).append(inst_nom)
        except Exception as e:
            logger.error(f"Erreur lecture providers.yaml : {e}")

    routage: dict[str, list[str]] = {}
    for cat in CATEGORIES_COGNITIVES:
        mod = modeles_dict.get(cat, "")
        if mod == "aucun" or not mod:
            routage[cat] = []
            continue
        insts = instances_par_modele.get(mod, [])
        routage[cat] = sorted(insts)

    # Repli 'defaut' = instances du modèle de décision
    mod_dec = modeles_dict.get("itinary_multi_agent", "")
    routage["defaut"] = sorted(instances_par_modele.get(mod_dec, []))
    return routage


def enregistrer_experience(config: dict[str, Any]) -> tuple[Path, bool]:
    """Enregistre l'expérience mémoire dans data/experiences_memoire/<categorie>/<evenement>/<nom>/."""
    nom = config.get("nom")
    if not nom or not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{2,127}$", nom):
        raise ValueError(f"Invalid experiment name: {nom!r}")

    existant = trouver_dossier_experience(nom)
    if existant:
        exp_dir = existant
    else:
        exp_dir = sous_dossier_categorie(config) / nom
    exp_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = exp_dir / "experience_memoire.yaml"

    contenu = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    modifie = True
    if yaml_path.is_file() and yaml_path.read_text(encoding="utf-8") == contenu:
        modifie = False
    else:
        yaml_path.write_text(contenu, encoding="utf-8")

    # Initialise etat.json s'il n'existe pas
    etat_path = exp_dir / "etat.json"
    if not etat_path.is_file():
        etat_initial = {
            "nom": nom,
            "etat": "en_attente",
            "cree_le": datetime.now(timezone.utc).isoformat(),
            "traite": {"etat": "en_attente"},
            "temoin": {"etat": "en_attente"},
        }
        etat_path.write_text(json.dumps(etat_initial, indent=2), encoding="utf-8")

    sauver_etat_formulaire(config)
    return exp_dir, modifie


# States written in `etat.json` that take precedence over the state deduced from the arms' files (2026-09-28).
ETATS_DECLARES_PRIMANT = ("non_conforme", "interrompue", "arretee", "echec")


def lister_experiences() -> list[dict[str, Any]]:
    """Scans data/experiences_memoire/ recursively and returns the ordered list of experiments."""
    if not DOSSIER_MEMOIRE.is_dir():
        return []

    exp_dirs = []
    for cfg_file in DOSSIER_MEMOIRE.rglob("experience_memoire.yaml"):
        if "archive" in cfg_file.parts or ".system_generated" in cfg_file.parts:
            continue
        exp_dirs.append(cfg_file.parent)

    resultats: list[dict[str, Any]] = []
    for exp_dir in sorted(exp_dirs, key=lambda p: p.name, reverse=True):
        cfg_file = exp_dir / "experience_memoire.yaml"
        if not cfg_file.is_file():
            continue
        try:
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}

        etat_file = exp_dir / "etat.json"
        etat_data: dict[str, Any] = {}
        if etat_file.is_file():
            try:
                etat_data = json.loads(etat_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Check of the traite and temoin subfolders
        dir_traite = exp_dir / "traite"
        dir_temoin = exp_dir / "temoin"

        def _statut_bras(cle: str, dossier: Path) -> str:
            # The declared state takes precedence when it says « suspended »: a moves.csv left from a previous
            # attempt does not make an arm interrupted by the quota a completed arm.
            declare = etat_data.get(cle, {}).get("etat", "en_attente")
            if declare == "suspendu":
                return declare
            return "ok" if (dossier / "moves.csv").is_file() else declare

        statut_traite = _statut_bras("traite", dir_traite)
        statut_temoin = _statut_bras("temoin", dir_temoin)

        # A declared verdict takes precedence over the files present: two moves.csv say that the arms
        # ran, not that the measurement holds. Until 2026-09-28, a non-compliant experiment
        # (A/B prefix not demonstrated) was displayed as « Terminée (A+B) ».
        verdict_declare = etat_data.get("etat")
        if verdict_declare in ETATS_DECLARES_PRIMANT:
            global_statut = verdict_declare
        elif "suspendu" in (statut_traite, statut_temoin):
            global_statut = "suspendue"
        elif statut_traite == "ok" and statut_temoin == "ok":
            global_statut = "terminee"
        elif statut_traite == "ok":
            global_statut = "traite_ok"
        else:
            global_statut = etat_data.get("etat") or "en_attente"

        # Témoin Ticket 106
        temoin_106 = "—"
        temoin_file = dir_traite / "temoin_souvenir.jsonl"
        if temoin_file.is_file():
            try:
                lignes = [
                    json.loads(l)
                    for l in temoin_file.read_text().splitlines()
                    if l.strip()
                ]
                if any(x.get("retrouve") for x in lignes):
                    temoin_106 = "✅ Retrouvé"
                else:
                    temoin_106 = "❌ Non retrouvé"
            except Exception:
                temoin_106 = "⚠️ Erreur"

        resultats.append(
            {
                "nom": exp_dir.name,
                "dossier": exp_dir,
                "canal": cfg.get("canal", "—"),
                "evenement": cfg.get("evenement", "—"),
                "modele_decision": cfg.get("modeles", {}).get(
                    "itinary_multi_agent", "—"
                ),
                "modele_jugement": cfg.get("modeles", {}).get(
                    "evenement_jugement", "—"
                ),
                "modele_stm": cfg.get("modeles", {}).get("stm_reflection", "—"),
                "population": cfg.get("population", "—"),
                "horizon_jours": cfg.get("horizon_jours", "—"),
                "etat": global_statut,
                "statut_traite": statut_traite,
                "statut_temoin": statut_temoin,
                "temoin_106": temoin_106,
                "modifie_le": datetime.fromtimestamp(cfg_file.stat().st_mtime),
                "config": cfg,
                "etat_data": etat_data,
            }
        )
    return resultats


def execution_vivante(nom_exp: str, exp_dir: Optional[Path] = None) -> bool:
    """Determines whether the memory experiment is actually running on the machine."""
    import subprocess
    import time

    # 1. Processus orchestrateur ou runner actif
    try:
        res = subprocess.run(
            ["pgrep", "-f", f"orchestrateur_memoire.*{nom_exp}|run_sequential_cohort.*{nom_exp}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            return True
    except Exception:
        pass

    # 2. Check etat.json: if it is not marked en_cours, it is not running without a process
    dossier = exp_dir or (DOSSIER_MEMOIRE / nom_exp)
    etat_file = dossier / "etat.json"
    if not etat_file.is_file():
        return False
    try:
        data = json.loads(etat_file.read_text(encoding="utf-8"))
        if data.get("etat") != "en_cours":
            return False
    except Exception:
        return False

    # 3. Si etat == "en_cours", vérifier si experiments/current ou etat.json a été écrit récemment
    lien_current = REPO_ROOT / "experiments" / "current"
    if lien_current.is_symlink() or lien_current.exists():
        try:
            cible = lien_current.resolve()
            app_log = cible / "app.log"
            if app_log.is_file() and (time.time() - app_log.stat().st_mtime) < 180:
                runs_dir = REPO_ROOT / "experiments" / "runs"
                if any(p.name.startswith(nom_exp) for p in runs_dir.iterdir() if p.is_dir()):
                    return True
        except Exception:
            pass

    return (time.time() - etat_file.stat().st_mtime) < 180



def enchainement_nuit_en_cours() -> Optional[dict[str, Any]]:
    """Detects whether a night campaign (enchainer_experiences_memoire.sh) is running."""
    import subprocess

    dossier_exp = REPO_ROOT / "experiments"
    if not dossier_exp.is_dir():
        return None
    journaux = sorted(
        dossier_exp.glob("enchainement_nuit_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not journaux:
        return None
    plus_recent = journaux[0]

    vivant = False
    try:
        res = subprocess.run(
            ["pgrep", "-f", "enchainer_experiences_memoire.sh"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        vivant = res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        pass

    derniere = ""
    try:
        lignes = plus_recent.read_text(encoding="utf-8", errors="ignore").splitlines()
        derniere = lignes[-1] if lignes else ""
    except Exception:
        pass

    detail = plus_recent.with_suffix(".detail.txt")
    return {
        "fichier": plus_recent,
        "detail": detail if detail.is_file() else plus_recent,
        "derniere_ligne": derniere,
        "vivant": vivant,
        "nom": plus_recent.stem,
        "maj": plus_recent.stat().st_mtime,
        # Ticket 118 — the pause in progress (provider overload, incomplete survey…) and
        # the time of the next attempt, written by the chain while it waits.
        "pause": _pause_de(plus_recent) if vivant else None,
    }


# What each stop reason means for whoever is in charge (marker `en_attente_quota.json`).
LIBELLES_MOTIF_PAUSE = {
    "surcharge_fournisseur": "fournisseur surchargé (HTTP 5xx ou 429)",
    "enquete_incomplete": "enquête incomplète, fournisseur saturé",
    "replis_consecutifs": "le modèle ne décide plus (replis consécutifs)",
    "decision_en_retard": "décisions en retard sur leur départ",
    "decision_absente": "décision absente après les retentatives",
    "consolidation_memoire": "consolidation de la mémoire absente",
    "prefixe_commun": "préfixe commun interrompu",
}


def _pause_de(journal: Path) -> Optional[dict[str, Any]]:
    """The pause that the night chain has declared (`<journal>.pause.json`), or None."""
    chemin = journal.with_suffix(".pause.json")
    if not chemin.is_file():
        return None
    try:
        pause = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    motif = str(pause.get("motif") or "inconnu")
    pause["libelle_motif"] = LIBELLES_MOTIF_PAUSE.get(motif, motif)
    return pause


def progression_memoire(nom_exp: str, cfg: dict[str, Any], etat_data: dict[str, Any]) -> dict[str, Any]:
    """Calcule l'état d'avancement d'une expérience mémoire (jours simulés, branche, etc.)."""
    import os
    import time

    # Starting point of Settings.gaml: arms A and B replay the same calendar.
    debut_simulation = datetime(2026, 3, 16, 5)
    mois = {
        nom: numero for numero, nom in enumerate(
            ("", "January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December")
        ) if nom
    }

    def jours_du_journal(journal: Path) -> tuple[int | None, str]:
        """Reads the last finished cycle without loading the whole journal into memory."""
        if not journal.is_file():
            return None, ""
        try:
            with journal.open("rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 65536))
                queue = f.read().decode("utf-8", errors="ignore")
            for ligne in reversed(queue.splitlines()):
                if "[sync] END sim_time=" not in ligne:
                    continue
                libelle = ligne.split("sim_time=", 1)[1].split(" state_update", 1)[0].strip()
                match = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4}), (\d{1,2}):(\d{2})", libelle)
                if not match or match[2] not in mois:
                    return None, libelle
                instant = datetime(int(match[3]), mois[match[2]], int(match[1]),
                                   int(match[4]), int(match[5]))
                return max(0, int((instant - debut_simulation).total_seconds() // 86400)), libelle
        except (OSError, ValueError):
            pass
        return None, ""

    try:
        horizon = int(cfg.get("horizon_jours", 42))
    except (ValueError, TypeError):
        horizon = 42

    branche = etat_data.get("branche_active")
    if not branche:
        if etat_data.get("traite", {}).get("etat") == "en_cours":
            branche = "treated"
        elif etat_data.get("temoin", {}).get("etat") == "en_cours":
            branche = "control"
        else:
            branche = (
                "treated"
                if etat_data.get("traite", {}).get("etat") != "ok"
                else "control"
            )

    derniere_journee = "Amorçage…"
    jours_bras = 0
    current = REPO_ROOT / "experiments" / "current"
    if current.exists():
        try:
            cible = current.resolve()
            identite = cible / "identite_run.json"
            autre_experience = False
            if identite.is_file():
                identite_data = json.loads(identite.read_text(encoding="utf-8"))
                rejeu = identite_data.get("rejeu_ab")
                autre_experience = bool(rejeu and rejeu != nom_exp)
            debut_bras = etat_data.get(
                "traite" if branche in ("treated", "traite") else "temoin", {}
            ).get("debut")
            journal = cible / "app.log"
            if debut_bras and journal.is_file():
                # Pendant le passage A → B, current désigne encore A quelques secondes.
                demarrage = datetime.fromisoformat(debut_bras.replace("Z", "+00:00")).timestamp()
                autre_experience |= journal.stat().st_mtime < demarrage
            if not autre_experience:
                nombre, libelle = jours_du_journal(journal)
                if libelle:
                    derniere_journee = libelle
                if nombre is not None:
                    jours_bras = min(horizon, nombre)
                else:
                    # Fallback during start-up or if the journal is not readable yet.
                    jours_bras = min(horizon, len(list(cible.glob("population_*_checkpoint_*.json"))))
                    if jours_bras == 0:
                        derniere_journee = "Amorçage simulation…"
        except Exception:
            pass

    # The finished arm counts in the progress even when current already points to the next one.
    traite_fait = horizon if etat_data.get("traite", {}).get("etat") == "ok" else 0
    temoin_fait = horizon if etat_data.get("temoin", {}).get("etat") == "ok" else 0
    if branche in ("treated", "traite") and not traite_fait:
        traite_fait = jours_bras
    elif branche in ("control", "temoin") and not temoin_fait:
        temoin_fait = jours_bras
    jours_faits = traite_fait + temoin_fait
    total_jours = 2 * horizon
    pourcent = min(100.0, jours_faits / total_jours * 100.0) if total_jours > 0 else 0.0

    debut_iso = (
        etat_data.get("traite", {}).get("debut")
        if branche == "treated"
        else (
            etat_data.get("temoin", {}).get("debut")
            or etat_data.get("debut")
        )
    ) or etat_data.get("debut")
    ecoule_s = None
    if debut_iso:
        try:
            t0 = datetime.fromisoformat(
                debut_iso.replace("Z", "+00:00")
            ).timestamp()
            ecoule_s = max(0.0, time.time() - t0)
        except Exception:
            pass

    return {
        "branche": branche,
        "branche_label": (
            "A (Traité)" if branche in ("treated", "traite") else "B (Témoin)"
        ),
        "jours_faits": jours_faits,
        "jours_bras": jours_bras,
        "horizon_jours": horizon,
        "total_jours": total_jours,
        "pourcent": pourcent,
        "derniere_journee": derniere_journee,
        "ecoule_s": ecoule_s,
    }


def tail_log(nom_exp: str, nb_lignes: int = 40) -> tuple[str, str]:
    """Extracts the tail of the most relevant journal for a memory experiment.
    Returns (contenu, source_path_str).
    """
    import os

    dash_dir = REPO_ROOT / "experiments" / ".dashboard"
    if dash_dir.is_dir():
        candidats = sorted(
            dash_dir.glob("*-root-experience-memoire-lancer*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for c in candidats:
            try:
                entete = c.open("r", encoding="utf-8", errors="ignore").read(500)
                if nom_exp in entete:
                    lignes = c.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    return (
                        "\n".join(lignes[-nb_lignes:])
                        if lignes
                        else "(journal vide)",
                        str(c.relative_to(REPO_ROOT)),
                    )
            except Exception:
                pass

    runs_dir = REPO_ROOT / "experiments" / "runs"
    if runs_dir.is_dir():
        candidats = sorted(
            runs_dir.glob(f"{nom_exp}_*/**/run_orchestrateur.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidats:
            try:
                lignes = candidats[0].read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                return (
                    "\n".join(lignes[-nb_lignes:])
                    if lignes
                    else "(journal vide)",
                    str(candidats[0].relative_to(REPO_ROOT)),
                )
            except Exception:
                pass

    app_log = REPO_ROOT / "experiments" / "current" / "app.log"
    if app_log.is_file():
        try:
            with open(app_log, "rb") as f:
                f.seek(0, os.SEEK_END)
                taille = f.tell()
                f.seek(max(0, taille - 24576))
                bloc = f.read().decode("utf-8", errors="replace")
            lignes = bloc.splitlines()
            return (
                "\n".join(lignes[-nb_lignes:]) if lignes else "(journal vide)",
                "experiments/current/app.log",
            )
        except Exception:
            pass

    dossier_exp = REPO_ROOT / "experiments"
    if dossier_exp.is_dir():
        nuit_logs = sorted(
            dossier_exp.glob("enchainement_nuit_*.detail.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if nuit_logs:
            try:
                with open(nuit_logs[0], "rb") as f:
                    f.seek(0, os.SEEK_END)
                    taille = f.tell()
                    f.seek(max(0, taille - 24576))
                    bloc = f.read().decode("utf-8", errors="replace")
                lignes = bloc.splitlines()
                return (
                    "\n".join(lignes[-nb_lignes:])
                    if lignes
                    else "(journal vide)",
                    str(nuit_logs[0].relative_to(REPO_ROOT)),
                )
            except Exception:
                pass

    local_log = DOSSIER_MEMOIRE / nom_exp / "traite" / "app.log"
    if local_log.is_file():
        try:
            lignes = local_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            return (
                "\n".join(lignes[-nb_lignes:]) if lignes else "(journal vide)",
                f"data/experiences_memoire/{nom_exp}/traite/app.log",
            )
        except Exception:
            pass

    return ("Aucun journal disponible pour le moment.", "—")


def _lire_fin(chemin: Path, max_octets: int = 2_000_000) -> str:
    """Reads a bounded window at the end of the file, without loading a large run into memory."""
    if not chemin.is_file():
        return ""
    try:
        with chemin.open("rb") as flux:
            flux.seek(0, os.SEEK_END)
            taille = flux.tell()
            flux.seek(max(0, taille - max_octets))
            texte = flux.read().decode("utf-8", errors="replace")
        # A read started in the middle of a line must not produce a false event.
        return texte.split("\n", 1)[-1] if taille > max_octets else texte
    except OSError:
        return ""


def _dossier_run_pour_journal(
    nom_exp: str, etat_data: dict[str, Any]
) -> Path | None:
    """Finds the live run of this experiment, without borrowing the `current` of another one."""
    current = REPO_ROOT / "experiments" / "current"
    if current.exists() or current.is_symlink():
        try:
            cible = current.resolve()
            identite = cible / "identite_run.json"
            if identite.is_file():
                data = json.loads(identite.read_text(encoding="utf-8"))
                rejeu = str(data.get("rejeu_ab") or "")
                if rejeu and rejeu != nom_exp:
                    raise ValueError("the current run belongs to another experiment")
            branche = etat_data.get("branche_active", "treated")
            cle = "traite" if branche in ("treated", "traite") else "temoin"
            debut = (etat_data.get(cle) or {}).get("debut")
            if debut and (cible / "app.log").is_file():
                demarrage = datetime.fromisoformat(str(debut).replace("Z", "+00:00")).timestamp()
                if (cible / "app.log").stat().st_mtime < demarrage:
                    raise ValueError("the current run predates the active arm")
            return cible
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    branche = etat_data.get("branche_active", "treated")
    local = DOSSIER_MEMOIRE / nom_exp / (
        "traite" if branche in ("treated", "traite") else "temoin"
    )
    if local.is_dir():
        return local
    # The catalogue files the experiments by family and protocol before their canonical name.
    nom_bras = "traite" if branche in ("treated", "traite") else "temoin"
    try:
        return next(p for p in DOSSIER_MEMOIRE.glob(f"**/{nom_exp}/{nom_bras}") if p.is_dir())
    except StopIteration:
        return None


def _instant_log(ligne: str) -> tuple[str, datetime | None]:
    match = re.match(r"^(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)", ligne)
    if not match:
        return "--:--:--", None
    try:
        return match[2], datetime.fromisoformat(f"{match[1]}T{match[2]}")
    except ValueError:
        return match[2], None


def _resume_ligne_journal(ligne: str) -> tuple[str, str] | None:
    """Turns a technical line into a readable event: (phase, text)."""
    heure, _ = _instant_log(ligne)
    message = ligne.rsplit(" - ", 1)[-1].strip()
    if "[ALARME]" in ligne or "| ERROR" in ligne:
        return "Alerte", f"{heure} · 🚨 {message[:260]}"
    if "[prefixe_commun]" in ligne:
        return "Synchronisation A/B", f"{heure} · ⏳ Synchronisation A/B · {message.split('] ', 1)[-1][:220]}"
    if "STM reflection for" in ligne:
        detail = message.split("] ", 1)[-1]
        return "Réflexion STM", f"{heure} · 🧠 STM · {detail[:230]}"
    if "ltm_self_reflection" in ligne and ("termin" in ligne.lower() or "reflection" in ligne.lower()):
        return "Réflexion LTM", f"{heure} · 🧠 LTM · {message[:230]}"
    if "[enquete]" in ligne and (
        "persona(s) interrogé(s)" in ligne or "ligne(s) enregistrée(s)" in ligne
    ):
        return "Enquête", f"{heure} · 📋 Enquête · {message.split('] ', 1)[-1][:230]}"
    if "[evenements] jour " in ligne and (
        "du run" in ligne or "service garanti" in ligne
    ):
        return "Événement", f"{heure} · ⚡ Événement · {message.split('] ', 1)[-1][:240]}"
    if "[sync] END sim_time=" in ligne:
        simule = ligne.split("sim_time=", 1)[1].split(" state_update", 1)[0].strip()
        return "Simulation", f"{heure} · ⏱ Simulation · pas terminé à {simule}"
    if "journaux isolés" in ligne or "Génération du rapport" in ligne:
        return "Archivage", f"{heure} · 📦 Archivage · {message[:230]}"
    if "Démarrage de la simulation" in ligne or ">>> Démarrage du Bras" in ligne:
        return "Initialisation", f"{heure} · 🚀 Initialisation · {message[:230]}"
    if "Bras terminé" in ligne:
        return "Fin du bras", f"{heure} · ✅ {message[:230]}"
    return None


def journal_semantique(
    nom_exp: str,
    cfg: dict[str, Any],
    etat_data: dict[str, Any],
    progression: dict[str, Any],
) -> dict[str, Any]:
    """Usable summary of the live run for the dashboard.

    The protocol files remain the sources of truth. This function produces no
    trace and does not change the run; it only aggregates a bounded window of its journals.
    """
    import time
    from collections import Counter

    dossier = _dossier_run_pour_journal(nom_exp, etat_data)
    if dossier is None:
        return {
            "etat": "attente",
            "icone": "⚪",
            "phase": "Amorçage",
            "resume": "En attente des premières traces du bras.",
            "derniere_activite_s": None,
            "alertes": [],
            "compteurs": {},
            "evenement": {},
            "chronologie": [],
            "categories_llm": {},
            "operations_ltm": {},
            "integrite_ab": {},
        }

    app_log = dossier / "app.log"
    app = _lire_fin(app_log, 2_000_000)
    lignes = app.splitlines()
    chronologie: list[str] = []
    phases: list[str] = []
    derniere_sync_texte = ""
    syncs: list[tuple[str, datetime | None]] = []
    dernier_bilan_evenement = ""
    lecteurs_attendus: int | None = None
    fenetre_evenement = ""

    for ligne in lignes:
        resume = _resume_ligne_journal(ligne)
        if resume:
            phase, texte = resume
            phases.append(phase)
            # Simulation steps are too many: only their last line is added.
            # An instant synchronisation is noise; a wait, on the other hand, explains a freeze.
            duree_synchro = re.search(r"drainé en ([0-9.]+)s", texte)
            synchro_banale = (
                phase == "Synchronisation A/B"
                and duree_synchro is not None
                and float(duree_synchro[1]) < 5.0
            )
            if phase != "Simulation" and not synchro_banale:
                chronologie.append(texte)
        if "[sync] END sim_time=" in ligne:
            derniere_sync_texte = ligne.split("sim_time=", 1)[1].split(" state_update", 1)[0].strip()
            _, instant = _instant_log(ligne)
            syncs.append((derniere_sync_texte, instant))
        if "lecteur(s) retenu(s)" in ligne:
            match = re.search(r"(\d+) lecteur\(s\) retenu\(s\)", ligne)
            lecteurs_attendus = int(match[1]) if match else lecteurs_attendus
        if "jours " in ligne and "→" in ligne and "[evenements]" in ligne:
            match = re.search(r"jours\s+(\d+→\d+)", ligne)
            fenetre_evenement = match[1] if match else fenetre_evenement
        if "[evenements] jour " in ligne and "du run" in ligne:
            dernier_bilan_evenement = ligne.rsplit(" - ", 1)[-1].split("] ", 1)[-1]

    if derniere_sync_texte:
        chronologie.append(f"--:--:-- · ⏱ Simulation · dernier pas terminé à {derniere_sync_texte}")
    chronologie = chronologie[-14:][::-1]

    fichiers_activite = [
        dossier / nom for nom in (
            "app.log", "agent_memory_events.jsonl", "trace_rappel.jsonl",
            "llm_exchanges.jsonl", "llm_errors.jsonl", "operations_concept.jsonl",
        )
    ]
    dates_maj = [p.stat().st_mtime for p in fichiers_activite if p.is_file()]
    age_s = max(0.0, time.time() - max(dates_maj)) if dates_maj else None

    memoire_texte = _lire_fin(dossier / "agent_memory_events.jsonl", 8_000_000)
    stm = memoire_texte.count('"context": "shortterm_memory"')
    ltm = memoire_texte.count('"context": "longterm_memory"')

    rappels_texte = _lire_fin(dossier / "trace_rappel.jsonl", 8_000_000)
    rappels = 0
    souvenirs_servis = 0
    rappels_vides = 0
    for ligne in rappels_texte.splitlines():
        try:
            entree = json.loads(ligne)
        except (ValueError, TypeError):
            continue
        rappels += 1
        servis = entree.get("servis") or []
        souvenirs_servis += len(servis)
        rappels_vides += not bool(servis)

    echanges = _lire_fin(dossier / "llm_exchanges.jsonl", 16_000_000)
    categories = Counter(re.findall(r'^\s*"category":\s*"([^"]+)"', echanges, re.MULTILINE))
    appels_llm = sum(categories.values())
    jetons_entree = sum(int(v) for v in re.findall(r'^\s*"tokens_in":\s*(\d+)', echanges, re.MULTILINE))
    jetons_sortie = sum(int(v) for v in re.findall(r'^\s*"tokens_out":\s*(\d+)', echanges, re.MULTILINE))
    erreurs_llm = 0
    derniere_erreur_llm: datetime | None = None
    for ligne in _lire_fin(dossier / "llm_errors.jsonl", 4_000_000).splitlines():
        if not ligne.lstrip().startswith("{"):
            continue
        erreurs_llm += 1
        try:
            instant = datetime.fromisoformat(str(json.loads(ligne).get("time", "")))
        except (ValueError, TypeError):
            instant = None
        if instant and (derniere_erreur_llm is None or instant > derniere_erreur_llm):
            derniere_erreur_llm = instant

    operations = Counter()
    for ligne in _lire_fin(dossier / "operations_concept.jsonl", 4_000_000).splitlines():
        try:
            operation = json.loads(ligne).get("operation")
        except (ValueError, TypeError):
            continue
        if operation:
            operations[str(operation)] += 1

    evenements_produits = sum(
        1 for ligne in _lire_fin(dossier / "evenements.jsonl", 4_000_000).splitlines()
        if ligne.lstrip().startswith("{")
    )
    alertes: list[str] = []
    if age_s is not None and age_s > 600:
        alertes.append(f"Aucune trace mise à jour depuis {int(age_s // 60)} min.")

    # Detects a simulated clock repeated for at least ten wall-clock minutes.
    if syncs:
        valeur = syncs[-1][0]
        identiques = []
        for simule, instant in reversed(syncs):
            if simule != valeur:
                break
            if instant:
                identiques.append(instant)
        if len(identiques) >= 2:
            duree_figee = (max(identiques) - min(identiques)).total_seconds()
            if duree_figee >= 600:
                alertes.append(
                    f"Progression simulée figée à {valeur} depuis {int(duree_figee // 60)} min."
                )
    erreur_recente = erreurs_llm > 0 and derniere_erreur_llm is None
    if derniere_erreur_llm is not None:
        maintenant = datetime.now(derniere_erreur_llm.tzinfo)
        erreur_recente = (maintenant - derniere_erreur_llm).total_seconds() <= 600
    if erreur_recente:
        alertes.append(f"{erreurs_llm} erreur(s) LLM, dont une dans les dix dernières minutes.")

    rejeu_bilan = etat_data.get("rejeu_ab") or {}
    rapprochement = etat_data.get("rapprochement_ticket108") or {}
    payes_avant = rejeu_bilan.get("payes_avant_evenement")
    if isinstance(payes_avant, int) and payes_avant > 0:
        alertes.append(
            f"Intégrité A/B : {payes_avant} appel(s) payé(s) avant l'événement."
        )
    if rapprochement and not rapprochement.get("conforme", False):
        alertes.append(
            "Injection incomplète : "
            f"{rapprochement.get('produites', 0)}/{rapprochement.get('declarees', 0)} produite(s)."
        )

    phase = phases[-1] if phases else "Simulation"
    if alertes:
        etat, icone = "attention", "🟠"
    elif age_s is not None and age_s <= 180:
        etat, icone = "actif", "🟢"
    else:
        etat, icone = "attente", "⚪"
    age_txt = "activité inconnue" if age_s is None else (
        "activité à l'instant" if age_s < 10 else f"dernière activité il y a {int(age_s)} s"
    )

    return {
        "etat": etat,
        "icone": icone,
        "phase": phase,
        "resume": (
            f"{progression.get('branche_label', 'Bras')} · "
            f"{derniere_sync_texte or progression.get('derniere_journee', 'amorçage')} · {age_txt}"
        ),
        "derniere_activite_s": age_s,
        "alertes": alertes,
        "compteurs": {
            "Appels LLM": appels_llm,
            "Erreurs LLM": erreurs_llm,
            "Entrées STM": stm,
            "Entrées LTM": ltm,
            "Rappels": rappels,
            "Souvenirs servis": souvenirs_servis,
            "Rappels vides": rappels_vides,
            "Jetons": jetons_entree + jetons_sortie,
        },
        "evenement": {
            "id": cfg.get("evenement", "—"),
            "fenetre": fenetre_evenement,
            "lecteurs_attendus": lecteurs_attendus,
            "traces_produites": evenements_produits,
            "dernier_bilan": dernier_bilan_evenement,
        },
        "chronologie": chronologie,
        "categories_llm": dict(categories),
        "operations_ltm": dict(operations),
        "integrite_ab": {
            "rejeu_actif": bool(cfg.get("rejeu_ab")),
            "prefixe_commun": bool(cfg.get("prefixe_commun")),
            "traite": (etat_data.get("traite") or {}).get("etat", "en_attente"),
            "temoin": (etat_data.get("temoin") or {}).get("etat", "en_attente"),
            "rejeu_servis": rejeu_bilan.get("servis"),
            "rejeu_payes": rejeu_bilan.get("payes"),
            "payes_avant_evenement": payes_avant,
            "injections_conformes": rapprochement.get("conforme") if rapprochement else None,
        },
    }


def activites_en_cours() -> list[dict[str, Any]]:
    """Returns the list of the memory experiments being run."""
    exps = lister_experiences()
    en_cours = []
    for e in exps:
        nom = e["nom"]
        etat = e["etat"]
        vivante = execution_vivante(nom, e["dossier"])
        if etat == "en_cours" or (vivante and etat not in ("terminee", "arretee")):
            prog = progression_memoire(nom, e["config"], e["etat_data"])
            log_tail, log_src = tail_log(nom, nb_lignes=35)
            journal = journal_semantique(nom, e["config"], e["etat_data"], prog)
            en_cours.append(
                {
                    **e,
                    "vivante": vivante,
                    "progression": prog,
                    "log_tail": log_tail,
                    "log_src": log_src,
                    "journal_semantique": journal,
                }
            )
    return en_cours


def interrompues() -> list[dict[str, Any]]:
    """Returns the suspended, stopped or failed memory experiments."""
    exps = lister_experiences()
    arretees = []
    for e in exps:
        etat = e["etat"]
        vivante = execution_vivante(e["nom"], e["dossier"])
        if etat in ("suspendue", "arretee", "echec") or (
            etat == "en_cours" and not vivante
        ):
            etat_data = e["etat_data"]
            note = etat_data.get("note", "")
            traite_etat = etat_data.get("traite", {}).get("etat", "")
            temoin_etat = etat_data.get("temoin", {}).get("etat", "")
            cause_libelle = "Arrêtée"
            cause_icone = "⏹"
            if etat == "suspendue" or "suspendu" in (traite_etat, temoin_etat):
                cause_libelle = "Suspendue (quota / 503)"
                cause_icone = "⏸"
            elif etat == "echec":
                cause_libelle = "Échec d'exécution"
                cause_icone = "🔴"
            elif etat == "en_cours" and not vivante:
                cause_libelle = "Processus interrompu"
                cause_icone = "⏹"

            detail = note or (
                f"Bras traité : {traite_etat} · Bras témoin : {temoin_etat}"
            )
            arretees.append(
                {
                    **e,
                    "cause_icone": cause_icone,
                    "cause_libelle": cause_libelle,
                    "cause_detail": detail,
                }
            )
    return arretees


def terminees() -> list[dict[str, Any]]:
    """Retourne les expériences mémoire terminées."""
    exps = lister_experiences()
    return [e for e in exps if e["etat"] == "terminee"]
