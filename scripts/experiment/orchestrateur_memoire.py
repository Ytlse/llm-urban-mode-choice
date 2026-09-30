#!/usr/bin/env python3
"""Counterfactual A/B orchestrator for memory experiments — Ticket 109.

DECISION D3: The two runs (treated, then matched control) are executed consecutively.
The experiment is only marked « terminee » if both were played successfully.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "services" / "llm-agents"))
from experiences import memoire, rejeu_ab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrateur_memoire")


# Code returned by `run_sequential_cohort.py` for an arm suspended by the safeguard (ticket 105).
CODE_BRAS_SUSPENDU = 7
# Exit code of an experiment whose two arms ran but whose A/B prefix is
# not demonstrated. Until 2026-09-28 it exited with 0: the night chain wrote it as
# « ✅ TERMINÉE ». Do not relaunch: both arms are played, it is the measurement that is wrong.
CODE_NON_CONFORME = 8


def _foyers_avec_relais(evenement: str) -> int:
    """Number of exposed households of a declaration that carries a relay (ticket 111), else 0."""
    if not evenement or evenement == "0":
        return 0
    chemin = memoire.DOSSIER_EVENEMENTS / f"{evenement}.yaml"
    if not chemin.is_file():
        return 0
    try:
        import yaml

        data = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — an estimate does not bring down the launch
        return 0
    if not data.get("relais"):
        return 0
    return len((data.get("exposition") or {}).get("foyers") or [])


def estimer_cout(config: dict[str, Any]) -> dict[str, Any]:
    """Estimates the volume of LLM calls for the two consecutive arms."""
    horizon = int(config.get("horizon_jours", 42))
    pop = str(config.get("population", "899549"))
    # The headcount READ, not guessed: a set of twenty households used to count as one persona, and
    # the estimate underestimated the campaign by a factor of twenty.
    nb_personas = memoire.decrire_population(pop)["agents"]

    appels_decision = nb_personas * horizon * 4
    appels_stm = nb_personas * horizon * 1
    appels_ltm = nb_personas * (horizon // 1)
    appels_enquetes = nb_personas * horizon * 5
    appels_jugement = nb_personas * 2  # ~2 exposures max
    # Ticket 111 — one relay call per exposed household, and one judgement per informed member
    # (at most the other household members; counted generously, one per non-reader agent).
    foyers_relais = _foyers_avec_relais(str(config.get("evenement", "")))
    appels_relais = foyers_relais
    if foyers_relais:
        appels_jugement += nb_personas

    par_bras = {
        "itinary_multi_agent": appels_decision,
        "stm_reflection": appels_stm,
        "ltm_self_reflection": appels_ltm,
        "enquete_affinite": appels_enquetes,
        "evenement_jugement": appels_jugement,
        "evenement_relais": appels_relais,
    }
    total_bras = sum(par_bras.values())
    total_ab = total_bras * 2  # Factor 2 for A/B (treated + control)

    return {
        "nb_personas": nb_personas,
        "horizon_jours": horizon,
        "appels_par_bras": par_bras,
        "total_requetes_par_bras": total_bras,
        "total_requetes_experience_ab": total_ab,
        "estimation_jetons": total_ab * 1500,  # ~1,500 tokens per call on average
    }


RACINE_REJEU = REPO_ROOT / "experiments" / "rejeu_ab"


def espace_rejeu(nom_exp: str, config: dict[str, Any]) -> str | None:
    """The experiment's replay space: its name, if it declares `rejeu_ab: true`."""
    return nom_exp if config.get("rejeu_ab") else None


class BorneIncertaine(ValueError):
    """The treated arm left no trace of its first prompt carrying the event: no safe bound."""


def borne_prefixe_commun(racine_bras: Path) -> int:
    """First prompt of the treated arm that carries the event, on GAMA's UTC wall clock.

    Until 2026-09-28, the bound was the injection timestamp. But a decision for the
    next day is computed the day before: in a13 v5, the article entered a reader's decision on
    26/03 at 07:30, 16.5 h before the injection. The control arm requested in strict replay a
    prompt the treated arm had never issued, and suspended itself.
    """
    instant, source = rejeu_ab.instant_debut_traitement(racine_bras / "traite" / "evenements.jsonl")
    if not instant:
        raise ValueError("Common prefix: first treated injection not found")
    borne = int(datetime.fromisoformat(instant).replace(tzinfo=timezone.utc).timestamp())
    lu = _canal_de(racine_bras) == "lu"
    sans_service = rejeu_ab.instant_premier_service(racine_bras / "traite" / "evenements.jsonl") is None
    if sans_service and lu:
        # 2026-09-29 — a WARNING until then, and a wrong bound: `run_sequential_cohort` did not
        # bring `premiers_services.jsonl` back into `traite/`, the bound fell back to the injection
        # (27/03 00:00 instead of 26/03 07:30 for a13 v6), and the control arm demanded in strict
        # replay the reader's 07:30 decision, which the treated arm had taken WITH the article:
        # a 409 at the same instant on each of three relaunches. Refused before the arm starts.
        raise BorneIncertaine(
            f"Common prefix: no first service recorded in "
            f"{racine_bras / 'traite' / rejeu_ab.FICHIER_PREMIERS_SERVICES} for a read event "
            f"(canal « lu »). The injection timestamp ({instant}) is NOT a safe bound: a decision "
            f"computed the day before may have carried the article earlier. Copy the file from "
            f"the treated arm's run directory, then relaunch."
        )
    else:
        logger.info(
            f"[prefixe_commun] strict bound of the control arm: {instant} ({borne}), "
            f"{'premier prompt porteur de l’événement' if source == 'service' else 'injection'}"
        )
    return borne


def _canal_de(racine_bras: Path) -> str | None:
    cfg = racine_bras / "experience_memoire.yaml"
    try:
        return (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("canal")
    except (OSError, yaml.YAMLError):
        return None


def mettre_de_cote_magasin(nom_exp: str) -> Path | None:
    """A treated arm that restarts from scratch must not replay a previous attempt.

    The store of an aborted attempt is RENAMED, never deleted: its answers were paid for
    and say what the model answered. Nothing to do if it is absent or empty.
    """
    magasin = RACINE_REJEU / nom_exp
    if not magasin.is_dir() or not any(magasin.glob("*.json")):
        return None
    cible = RACINE_REJEU / f"{nom_exp}.mis_de_cote_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    magasin.rename(cible)
    logger.warning(
        f"[{nom_exp}] replay store of a previous attempt set aside → {cible.name}: "
        f"the treated arm restarts from scratch, it does not replay the old one."
    )
    return cible


def archive_du_bras(nom_exp: str, branche: str) -> Path | None:
    """The archive directory the arm produced, according to the cohort manifest."""
    manifeste = REPO_ROOT / "experiments" / "runs" / f"{nom_exp}_{branche}" / "manifeste.json"
    try:
        entrees = json.loads(manifeste.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for e in reversed(entrees):
        if e.get("branche") == branche and e.get("archive"):
            return Path(e["archive"])
    return None


def verifier_prefixe_deplacements(racine_bras: Path, date_evt: str | None) -> dict[str, Any]:
    """Checks directly that the preprocessing outputs of the two arms are identical."""
    chemins = {b: racine_bras / b / "moves.csv" for b in ("traite", "temoin")}
    if date_evt is None or any(not p.is_file() for p in chemins.values()):
        return {"conforme": False, "raison": "date événement ou moves.csv manquant"}

    champs = (
        "Heure de départ",
        "Mode de transport Choisi",
        "Modes proposés au LLM",
        "Options présentées",
        "P(Marche) %",
        "P(Vélo) %",
        "P(Voiture Privée) %",
        "P(Transports_collectifs) %",
        "P(Train) %",
        "P(Deux-roues motorisé) %",
        "P(Autres modes) %",
        "Mémoire à court terme",
        "Mémoire à long terme",
        "Filtre de perception",
    )

    # Old callers pass a date; new ones pass the exact instant
    # of the first injection to also include the trips of the publication morning.
    limite_exacte = datetime.fromisoformat(date_evt) if len(date_evt) > 10 else None

    def lire(path: Path) -> dict[tuple[str, str, str], list[tuple[str, ...]]]:
        groupes: dict[tuple[str, str, str], list[tuple[str, ...]]] = {}
        with path.open(encoding="utf-8", newline="") as flux:
            for ligne in csv.DictReader(flux):
                depart = ligne.get("Heure de départ", "")
                try:
                    date_depart = datetime.fromisoformat(depart)
                    jour_vecu = (date_depart - timedelta(hours=3)).date().isoformat()
                except ValueError:
                    continue
                if limite_exacte is not None:
                    if date_depart >= limite_exacte:
                        continue
                elif jour_vecu >= date_evt:
                    continue
                cle = (ligne.get("ID Personne", ""), ligne.get("ID Activité", ""), jour_vecu)
                groupes.setdefault(cle, []).append(tuple(ligne.get(f, "") for f in champs))
        for valeurs in groupes.values():
            valeurs.sort()
        return groupes

    traite, temoin = lire(chemins["traite"]), lire(chemins["temoin"])
    cles = traite.keys() | temoin.keys()
    manquantes = sum(cle not in traite or cle not in temoin for cle in cles)
    divergentes = sum(
        cle in traite and cle in temoin and traite[cle] != temoin[cle] for cle in cles
    )
    return {
        "conforme": manquantes == 0 and divergentes == 0,
        "lignes_traite": sum(map(len, traite.values())),
        "lignes_temoin": sum(map(len, temoin.values())),
        "cles_manquantes": manquantes,
        "cles_divergentes": divergentes,
    }


def controler_rejeu(nom_exp: str, racine_bras: Path) -> dict[str, Any] | None:
    """After the control arm: each call before the event must have been served by replay."""
    archive = archive_du_bras(nom_exp, "control")
    journal = archive / "llm_exchanges.jsonl" if archive else None
    if journal is None or not journal.is_file():
        logger.error(
            f"[ALARME] [{nom_exp}] replay: exchange log of the control arm not found "
            f"({journal or 'aucune archive au manifeste'}) — the replay cannot be checked."
        )
        return None
    # The same instant as the strict bound: the first prompt carrying the event.
    instant_evt, _ = rejeu_ab.instant_debut_traitement(racine_bras / "traite" / "evenements.jsonl")
    date_evt = instant_evt[:10] if instant_evt else None
    origine = archive.name
    resultat = rejeu_ab.bilan(rejeu_ab.lire_echanges(journal, origine=origine), date_evt)
    resultat["instant_evenement"] = instant_evt
    resultat["prefixe_deplacements"] = verifier_prefixe_deplacements(racine_bras, instant_evt)
    resultat["consignes"] = len(list((RACINE_REJEU / nom_exp).glob("*.json")))
    resultat["archive_temoin"] = str(archive)
    if resultat["payes_avant_evenement"] or not resultat["prefixe_deplacements"]["conforme"]:
        logger.error(
            f"[ALARME] [{nom_exp}] replay: {resultat['payes_avant_evenement']} call(s) of the control "
            f"PAID before the event ({date_evt}) — the arms diverged for a reason other "
            f"than it. First ones: {resultat['premiers_payes_avant'][:3]}. "
            f"Moves prefix: {resultat['prefixe_deplacements']}"
        )
    else:
        logger.info(
            f"[{nom_exp}] replay compliant: control served at {resultat['servis']}/"
            f"{resultat['servis'] + resultat['payes']} by replay, no call paid before "
            f"the event ({date_evt}); {resultat['consignes']} answers recorded."
        )
    return resultat


def _ecrire_echanges(chemin: Path, echanges: list[dict[str, Any]]) -> None:
    chemin.write_text(
        "".join(json.dumps(e, ensure_ascii=False, default=str, indent=2) + "\n" for e in echanges),
        encoding="utf-8",
    )


def isoler_journaux(archive: Path, cible: Path, origine: str) -> dict[str, int]:
    """Copies into the arm's folder only its LLM exchanges and errors.

    The worker remains shared, so its raw files may contain several clients.
    The experiment's outputs, for their part, become self-contained and unambiguous. Old
    errors without an ``origine`` field are discarded: they cannot be attributed honestly.
    """
    cible.mkdir(parents=True, exist_ok=True)
    bilan = {"echanges": 0, "erreurs": 0, "erreurs_sans_origine": 0}
    echanges_path = archive / "llm_exchanges.jsonl"
    if echanges_path.is_file():
        echanges = rejeu_ab.lire_echanges(echanges_path, origine=origine)
        _ecrire_echanges(cible / "llm_exchanges.jsonl", echanges)
        bilan["echanges"] = len(echanges)
    erreurs_path = archive / "llm_errors.jsonl"
    if erreurs_path.is_file():
        erreurs = []
        for ligne in erreurs_path.read_text(encoding="utf-8").splitlines():
            try:
                erreur = json.loads(ligne)
            except ValueError:
                continue
            if erreur.get("origine") is None:
                bilan["erreurs_sans_origine"] += 1
            elif erreur.get("origine") == origine:
                erreurs.append(erreur)
        (cible / "llm_errors.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in erreurs),
            encoding="utf-8",
        )
        bilan["erreurs"] = len(erreurs)
    return bilan


def executer_bras(
    nom_exp: str,
    branche: str,
    config: dict[str, Any],
    workdir_cible: Path,
    dry_run: bool = False,
    strict_avant_ts: int | None = None,
) -> int:
    """Runs one arm (treated or control) via run_sequential_cohort.py."""
    logger.info(f"[{nom_exp}] >>> Démarrage du Bras : {branche.upper()}")
    workdir_cible.mkdir(parents=True, exist_ok=True)

    pop = str(config.get("population", "899549"))
    personas_arg = ["all"] if pop == "cohorte_10" else [pop]
    evenement_arg = str(config.get("evenement", "c6_voiture_suspecte")) if branche == "treated" else "0"

    # Building the instances_admises routing table
    routage = memoire.resoudre_instances_admises(
        config.get("modeles", {}), adaptateur=config.get("adaptateur") or None
    )
    # A declared model that no instance serves — or that the provider filter discarded —
    # would fall back on the container's default, i.e. on another model, without a word.
    orphelins = [
        f"{cat} → {mod}" for cat, mod in (config.get("modeles") or {}).items()
        if mod and mod != "aucun" and not routage.get(cat)
    ]
    if orphelins:
        logger.error(
            f"[ALARME] [{nom_exp}_{branche}] no instance "
            f"{'« ' + config['adaptateur'] + ' » ' if config.get('adaptateur') else ''}serves: "
            f"{', '.join(orphelins)} — arm refused."
        )
        return 2
    logger.info(f"[{nom_exp}_{branche}] instance routing: {json.dumps(routage)}")

    env = dict(os.environ)
    if routage:
        env["INSTANCES_ADMISES"] = json.dumps(routage)
    # Exact-prompt replay: the SAME space for both arms, otherwise the control sees nothing.
    espace = espace_rejeu(nom_exp, config)
    env["REJEU_AB"] = espace or ""
    env["WORLD__PREFIXE_COMMUN"] = "true" if config.get("prefixe_commun") else "false"
    env["REJEU_STRICT_AVANT_TS"] = str(strict_avant_ts or 0)
    if config.get("prefixe_commun"):
        if not espace or (branche == "control" and strict_avant_ts is None and not dry_run):
            logger.error("[ALARME] Common prefix: cache or treatment bound missing")
            return 2
        logger.info(
            "[%s_%s] common prefix active; replay mandatory before %s",
            nom_exp, branche, strict_avant_ts if strict_avant_ts is not None else "sans objet",
        )
    if espace:
        logger.info(f"[{nom_exp}_{branche}] exact-prompt replay: space {espace}")
    if "memoire_importance_choc" in config:
        env["MEMOIRE__IMPORTANCE_CHOC"] = str(config["memoire_importance_choc"])
    if "stm_reflection_min_entries" in config:
        env["STM_REFLECTION_MIN_ENTRIES"] = str(config["stm_reflection_min_entries"])
    if "graine_tirage" in config:
        env["AGENT__MODE_CHOICE_SEED"] = str(config["graine_tirage"])
    # Always set, true or false: the launcher then compares it with `identite_run.json`, and a
    # container that did not receive it stops the arm instead of running without sharing.
    env["MEMOIRE__PARTAGE_FOYER_ENABLED"] = "true" if config.get("partage_foyer") else "false"
    # Micro-batching at the maximum: as many tasks in flight as agents, so that a wave of
    # departures leaves in ONE batch instead of batches of 8. Floor 8 = the historical default.
    env["WORLD__WORKER_CONCURRENCY"] = str(
        max(8, int(memoire.decrire_population(str(config.get("population", "899549")))["agents"] or 0))
    )
    # Ticket V5 — Daily affinity surveys for all memory experiments
    if config.get("enquetes_quotidiennes", True):
        env["EXPERIMENT_SURVEY_DAYS"] = "daily"
        env["EXPERIMENT_SURVEY_ENABLED"] = "1"
    env["GOOGLE_ADAPTER_REQUEST_TIMEOUT"] = str(config.get("google_request_timeout", 20.0))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "experiment" / "run_sequential_cohort.py"),
        "--personas",
        *personas_arg,
        "--branch",
        branche,
        "--evenement",
        evenement_arg,
        "--experiment-id",
        f"{nom_exp}_{branche}",
        "--horizon-jours",
        str(int(config.get("horizon_jours", 42))),
    ]

    if config.get("arret_sur_extinction"):
        cmd.append("--arret-sur-extinction")
        cmd.extend(["--jours-apres-extinction", str(config.get("jours_apres_extinction", 5))])

    if dry_run:
        cmd.append("--dry-run")
        logger.info(f"[{nom_exp}_{branche}] [DRY-RUN] Command: {' '.join(cmd)}")
        # Creation of synthetic traces to validate the pipeline
        (workdir_cible / "moves.csv").write_text("jour,mode,person_id\n1,car,899549\n", encoding="utf-8")
        (workdir_cible / "evenements.jsonl").write_text(
            json.dumps({"evenement_id": evenement_arg, "jour_run": 15, "person_id": pop}) + "\n"
            if branche == "treated" else ""
        )
        (workdir_cible / "agent_memory_events.jsonl").write_text('{"event": "test"}\n')
        (workdir_cible / "identite_run.json").write_text(json.dumps({"branche": branche, "dry_run": True}))
        (workdir_cible / "temoin_souvenir.jsonl").write_text(
            json.dumps({"evenement_id": evenement_arg, "retrouve": True, "branche": branche}) + "\n"
            if branche == "treated" else ""
        )
        return 0

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        logger.error(f"[{nom_exp}_{branche}] run_sequential_cohort failed (code {proc.returncode})")
        return proc.returncode

    # Bringing the data back from experiments/runs/ to the target workdir
    dossier_source = REPO_ROOT / "experiments" / "runs" / f"{nom_exp}_{branche}"
    if dossier_source.is_dir():
        for sous in dossier_source.iterdir():
            if sous.is_dir():
                for f in ["moves.csv", "evenements.jsonl", rejeu_ab.FICHIER_PREMIERS_SERVICES, "agent_memory_events.jsonl", "identite_run.json", "temoin_souvenir.jsonl", "app.log"]:
                    source_f = sous / f
                    if source_f.is_file():
                        shutil.copy2(source_f, workdir_cible / f)
                if (sous / "reports").is_dir():
                    shutil.copytree(sous / "reports", workdir_cible / "reports", dirs_exist_ok=True)
    archive = archive_du_bras(nom_exp, branche)
    if archive is not None:
        isolement = isoler_journaux(archive, workdir_cible, archive.name)
        logger.info(
            f"[{nom_exp}_{branche}] journaux isolés : {isolement['echanges']} échange(s), "
            f"{isolement['erreurs']} erreur(s) ; "
            f"{isolement['erreurs_sans_origine']} ancienne(s) erreur(s) non attribuable(s) écartée(s)."
        )
    return 0


def rapprochement_injections(config: dict[str, Any], workdir_traite: Path) -> dict[str, Any]:
    """Ticket 108 check: reconciliation of declared vs produced injections."""
    import importlib.util

    evenement_id = config.get("evenement", "")
    dossier_evt = REPO_ROOT / "services" / "llm-agents" / "config" / "evenements"
    derive = dossier_evt / f"{evenement_id}__{config.get('population', '')}.yaml"
    fichier_evt = derive if derive.is_file() else dossier_evt / f"{evenement_id}.yaml"
    if not fichier_evt.is_file():
        fichier_evt = REPO_ROOT / "services" / "llm-agents" / "config" / "chocs" / f"{evenement_id}.yaml"

    chemin_mod = REPO_ROOT / "scripts" / "analysis" / "rapprochement_injections.py"
    spec = importlib.util.spec_from_file_location("rapprochement_injections", chemin_mod)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        r = mod.rapprocher(workdir_traite, fichier_declaration=fichier_evt if fichier_evt.is_file() else None)
        return {
            "evenement_id": r.evenement_id or evenement_id,
            "declarees": r.declarees,
            "produites": r.produites,
            "manquantes": r.manquantes,
            "conforme": r.conforme,
            "details": [
                {
                    "person_id": l.person_id,
                    "evenement_id": l.evenement_id,
                    "jour_declare": l.jour_declare,
                    "date_simulee": l.date_simulee.isoformat() if l.date_simulee else None,
                    "statut": l.statut,
                    "detail": l.detail,
                    "raison": l.raison,
                }
                for l in r.lignes
            ],
        }

    return {
        "evenement_id": evenement_id,
        "declarees": 0,
        "produites": 0,
        "manquantes": 0,
        "conforme": False,
        "details": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B memory experiment orchestrator (Ticket 109).")
    parser.add_argument("--experience", required=True, help="Name of the memory experiment")
    parser.add_argument("--estimer", action="store_true", help="Estimates the cost without launching anything")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without a GAMA simulation")
    parser.add_argument("--branche", choices=["both", "treated", "control"], default="both", help="Branches to play")

    args = parser.parse_args()
    nom_exp = args.experience
    exp_dir = memoire.trouver_dossier_experience(nom_exp) or (memoire.DOSSIER_MEMOIRE / nom_exp)
    cfg_path = exp_dir / "experience_memoire.yaml"

    if not cfg_path.is_file():
        logger.error(f"Configuration file not found: {cfg_path}")
        sys.exit(1)

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # Sharing within the household only makes sense if there are multi-member households. Enabled
    # on a single persona, it would run without sharing anything and the run would read as
    # "nothing is passed on" — an absence of measurement that would pass for a result.
    pop = str(config.get("population", ""))
    if config.get("partage_foyer") and memoire.decrire_population(pop)["foyers_partages"] == 0:
        logger.error(
            f"[ALARME] [{nom_exp}] partage_foyer enabled on « {pop} », which has no household "
            f"of at least two members — experiment refused."
        )
        sys.exit(2)

    if args.estimer:
        bilan = estimer_cout(config)
        print("\n" + "=" * 60)
        print(f"🧮 COST ESTIMATE — MEMORY EXPERIMENT {nom_exp}")
        print("=" * 60)
        print(f"Population: {bilan['nb_personas']} persona(s) | Horizon: {bilan['horizon_jours']} days")
        print("\nBreakdown by cognitive function (per arm):")
        for cat, count in bilan["appels_par_bras"].items():
            print(f"  - {memoire.LIBELLES_CATEGORIES.get(cat, cat):<35} : {count:>5} requests")
        print("-" * 60)
        print(f"Total per arm                : {bilan['total_requetes_par_bras']:>6} requests")
        print(f"Total A/B campaign (x2)      : {bilan['total_requetes_experience_ab']:>6} requests")
        print(f"Estimated tokens (order of magnitude) : ~{bilan['estimation_jetons']:,} tokens")
        print("=" * 60 + "\n")
        return

    # 2026-09-28 — every memory experiment plays in common prefix: it is what keeps the two
    # arms in the same world until the event. A declaration that does not say so is refused,
    # never reinterpreted: the old ones ran without it, and their results read as
    # they were produced.
    if not (config.get("prefixe_commun") and config.get("rejeu_ab")):
        logger.error(
            f"[{nom_exp}] experiment refused: every memory experiment plays in common prefix, "
            f"with the A/B replay it depends on. The declaration carries rejeu_ab="
            f"{config.get('rejeu_ab')!r}, prefixe_commun={config.get('prefixe_commun')!r} — "
            f"add « rejeu_ab: true » and « prefixe_commun: true » to {cfg_path}."
        )
        sys.exit(2)

    # State: we RESTART from the existing one. A relaunch after a suspension must neither replay
    # an arm already completed, nor erase the trace of what happened.
    etat_path = exp_dir / "etat.json"
    try:
        precedent = json.loads(etat_path.read_text(encoding="utf-8")) if etat_path.is_file() else {}
    except (OSError, ValueError):
        precedent = {}
    maintenant = datetime.now(timezone.utc).isoformat()
    etat_data = {
        **precedent,
        "nom": nom_exp,
        "etat": "en_cours",
        "debut": precedent.get("debut") or maintenant,
        "traite": precedent.get("traite") or {"etat": "en_attente"},
        "temoin": precedent.get("temoin") or {"etat": "en_attente"},
    }

    # A dry run touches neither the state nor the arm folders. On 2026-09-24, the one for
    # c6 had marked the experiment « terminee » in 50 ms and dropped a fake `moves.csv` into
    # `traite/` and `temoin/`: the registry showed it completed, a real relaunch would have skipped
    # both arms, and an archiving for the paper would have carried off synthetic traces.
    def ecrire_etat() -> None:
        if not args.dry_run:
            etat_path.write_text(json.dumps(etat_data, indent=2), encoding="utf-8")

    racine_bras = exp_dir / "essai_a_blanc" if args.dry_run else exp_dir
    ecrire_etat()

    branches = ["treated", "control"] if args.branche == "both" else [args.branche]

    for br in branches:
        cle = "traite" if br == "treated" else "temoin"
        cible = racine_bras / cle
        if etat_data[cle].get("etat") == "ok" and not args.dry_run:
            logger.info(f"[{nom_exp}] arm {br} already completed ({etat_data[cle].get('fin')}) — not replayed.")
            continue
        # Common prefix included (2026-09-28): the resume replays the day from its start, the
        # treated arm rereads its own store, and the control arm, before the bound, stops dead on any
        # call missing from the store. A resume that does not reproduce the prefix therefore shows —
        # in strict replay, then in the trip check — instead of being refused outright.
        etat_avant = etat_data[cle].get("etat")
        reprise = etat_avant == "suspendu"
        etat_data["branche_active"] = br
        etat_data[cle]["etat"] = "en_cours"
        etat_data[cle].setdefault("debut", datetime.now(timezone.utc).isoformat())
        ecrire_etat()
        if reprise:
            logger.info(
                f"[{nom_exp}] resuming arm {br}, suspended on {etat_data[cle].get('suspendu_le')}"
                + (" — common prefix: the strict replay and the trip check will judge."
                   if config.get("prefixe_commun") else ".")
            )
        elif br == "treated" and espace_rejeu(nom_exp, config) and not args.dry_run:
            mettre_de_cote_magasin(nom_exp)

        strict_avant_ts = None
        if br == "control" and config.get("prefixe_commun") and not args.dry_run:
            try:
                strict_avant_ts = borne_prefixe_commun(racine_bras)
            except BorneIncertaine as exc:
                # Nothing has run: the arm keeps its state and the experiment stays resumable.
                logger.error("[ALARME] %s — control arm NOT launched.", exc)
                etat_data[cle]["etat"] = etat_avant or "en_attente"
                etat_data["etat"] = "suspendue"
                ecrire_etat()
                sys.exit(2)
            except ValueError as exc:
                logger.error("[ALARME] %s", exc)
                etat_data[cle]["etat"] = "echec"
                etat_data["etat"] = "echec"
                ecrire_etat()
                sys.exit(2)
        options_bras: dict[str, Any] = {"dry_run": args.dry_run}
        if config.get("prefixe_commun"):
            options_bras["strict_avant_ts"] = strict_avant_ts
        ret = executer_bras(nom_exp, br, config, cible, **options_bras)
        if ret == CODE_BRAS_SUSPENDU:
            # Option A (2026-09-24): neither success nor failure — the arm is resumed at relaunch, and
            # the next one is NOT launched on an exhausted quota.
            etat_data["etat"] = "suspendue"
            etat_data[cle]["etat"] = "suspendu"
            etat_data[cle]["suspendu_le"] = datetime.now(timezone.utc).isoformat()
            ecrire_etat()
            logger.error(
                f"[ALARME] [{nom_exp}] arm {br} SUSPENDED by the safeguard (quota or fallbacks) — "
                f"the next arm is not launched. Relaunching the same experiment will resume it."
            )
            sys.exit(ret)
        if ret != 0:
            logger.error(f"[ALARME] Failure on arm {br} (code {ret}) — stopping the campaign.")
            etat_data["etat"] = "echec"
            etat_data["traite" if br == "treated" else "temoin"]["etat"] = "echec"
            ecrire_etat()
            sys.exit(ret)

        etat_data["traite" if br == "treated" else "temoin"]["etat"] = "ok"
        etat_data["traite" if br == "treated" else "temoin"]["fin"] = datetime.now(timezone.utc).isoformat()
        ecrire_etat()

    # Post-run synthesis and quality checks
    rappr = rapprochement_injections(config, racine_bras / "traite")
    if "treated" in branches and not args.dry_run and not rappr["conforme"]:
        logger.error(
            f"[ALARME] [108] [{nom_exp}] injections produced {rappr['produites']} / declared "
            f"{rappr['declarees']} — the treated arm did not receive the event it declares: "
            f"do not read it as a measurement of its effect."
        )
    # D3: completed only if BOTH arms ran. An arm played alone (debug, resume)
    # leaves the experiment « partielle », so that it never passes for an A/B measurement.
    # We read the arms' state, not this invocation's list: a control arm played alone after a
    # treated arm completed the day before does complete the experiment.
    traite_ok = etat_data["traite"].get("etat") == "ok"
    temoin_ok = etat_data["temoin"].get("etat") == "ok"
    if traite_ok and temoin_ok:
        etat_data["etat"] = "terminee"
    else:
        etat_data["etat"] = "traite_ok" if traite_ok else "partielle"
    etat_data["fin"] = datetime.now(timezone.utc).isoformat()
    etat_data["rapprochement_ticket108"] = rappr
    if traite_ok and temoin_ok and espace_rejeu(nom_exp, config) and not args.dry_run:
        etat_data["rejeu_ab"] = controler_rejeu(nom_exp, racine_bras)
        controle = etat_data["rejeu_ab"]
        if (
            controle is None
            or controle.get("payes_avant_evenement", 0) > 0
            or not controle.get("prefixe_deplacements", {}).get("conforme", False)
        ):
            etat_data["etat"] = "non_conforme"
            logger.error(
                f"[ALARME] [{nom_exp}] A/B comparison NOT COMPLIANT: the common prefix "
                f"is not demonstrated. Both arms are finished, but their results must "
                f"not be interpreted as a causal effect."
            )
    ecrire_etat()

    logger.info("=" * 60)
    if args.dry_run:
        logger.info(
            f"✅ DRY RUN of {nom_exp} succeeded ({', '.join(branches)}) — no simulation, "
            f"no call; experiment state UNCHANGED, synthetic traces in {racine_bras}."
        )
    elif etat_data["etat"] == "non_conforme":
        logger.error(
            f"⛔ CAMPAIGN {nom_exp} FINISHED BUT NOT COMPLIANT: A/B prefix check failed."
        )
    elif len(branches) == 2:
        logger.info(f"✅ MEMORY CAMPAIGN {nom_exp} COMPLETED SUCCESSFULLY (BOTH ARMS RAN).")
    else:
        logger.info(
            f"✅ Arm {branches[0]} of {nom_exp} finished — PARTIAL experiment "
            f"(the other arm was not played)."
        )
    logger.info(f"Injection reconciliation (Ticket 108): {rappr['produites']}/{rappr['declarees']} produced.")
    logger.info("=" * 60)
    if not args.dry_run and etat_data["etat"] == "non_conforme":
        sys.exit(CODE_NON_CONFORME)


if __name__ == "__main__":
    main()
