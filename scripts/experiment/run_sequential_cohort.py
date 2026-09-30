#!/usr/bin/env python3
"""Sequential cohort orchestrator with A/B counterfactual cloning and guaranteed hot resume.

Ticket 077 — High-Fidelity Replay (Shock c6_voiture_suspecte).

Architecture principles:
1. Sequential execution of N runs of 1 inhabitant (avoids any TPM/RPD saturation).
2. Counterfactual cloning by simulation fork:
   - Branch A (Treated): with an engine shock on D8-9.
   - Branch B (Control): without shock, strictly identical conditions.
3. Guaranteed hot resume per persona (CONT=1 sealed per workdir).
4. Absolute memory isolation (surveys pollute neither STM nor LTM).
5. Neutrality of the main mode (1000 inhabitants not impacted).

⚠ THIS SCRIPT IS THE ONLY LAUNCHER OF THE CAMPAIGN (since 2026-09-19).

The experiment settings — vehicle chaining, return lock, draw truncation
threshold, reflection floor — are no longer in `config/config.yaml`: set there, they had
become the repository default and appeared in no run identity. They are set here, through
the environment, and `identite_run.json` records them (ticket 077, lot K).

Consequence not to be missed: a `make run OFFLINE=1 CHOC=…` launched BY HAND now runs
with the repository defaults — chaining on, no truncation, reflection at 10 entries.
This is intended, and it is the reverse of the previous trap: a manual run no longer silently
carries the settings of an experiment. To reproduce an arm, go through this script.

Survey milestones are declared through `EXPERIMENT_SURVEY_DAYS`; their default follows the
protocol in force (D12, D17, D29, D40).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# The extinction detector lives in its own module — it is pure, and its tests exercise it
# without a run. Two import paths because this file runs as a script (sys.path[0] is then
# `scripts/experiment/`) AND is imported as a module by the tests.
try:
    from arret_sur_extinction import analyser as analyser_extinction
except ImportError:  # pragma: no cover — depends on the invocation mode, not on the code
    from scripts.experiment.arret_sur_extinction import analyser as analyser_extinction

COHORTE_10_PERSONAS = [
    "899549",   # Corinne (historical subject of Ticket 077)
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

# The reference shock, VERSIONED and tested (`specs/ticket_077/tests.md`, section H): lived
# text without verdict or intention, `cadence: jour`. The cohort DERIVES one file per persona
# from it, where only the `agents:` line changes — it never redefines the text.
#
# ⚠ Until 2026-09-19, this module carried its own copy of the shock, and wrote it into the
# workdir, which the controller does not read. Two consequences: fixes made to the reference
# file did not reach the campaign, and the `make run` command carried NO
# `CHOC=` — so the control branch ran with the shock declared in `config.yaml`. The control
# arm was not one.
# The experiment arm settings, in ONE table: what we set, and what the run identity
# must carry back. Two separate lists would have diverged — that is the very nature of the defect
# this lot fixes.
#
# ⚠ BARE NAME, without the `AGENT__` prefix. The sub-configurations of `settings.py` are
# `BaseSettings` without prefix: `VEHICLE_CHAIN_ENABLED` is read, `AGENT__VEHICLE_CHAIN_ENABLED`
# is read by nobody. This script set the second form for weeks, with no effect, and
# the run of 2026-09-19 16:45 ran with the repository defaults.
#
# ⚠ These variables must also be declared as pass-through in `infra/docker-compose.yml`:
# compose only passes to the container what it declares.
REGLAGES_CAMPAGNE: dict[str, tuple[str, str, Any]] = {
    # environment variable: (value set, identite_run.json field, expected value)
    # ⚠ RE-ENABLED on 2026-09-21, by decision of the author. Switched off, they produced
    # physically impossible chains — driving home from a trip made by bus —
    # and the modal shares that came out of them were not defensible.
    "VEHICLE_CHAIN_ENABLED": ("true", "chaine_vehicules", True),
    "VEHICLE_RETURN_HOME_LOCK": ("true", "verrou_retour_domicile", True),
    "MODE_CHOICE_TRUNCATION_THRESHOLD": ("0.15", "seuil_troncature", 0.15),
    "STM_REFLECTION_MIN_ENTRIES": ("5", "reflexion_stm_min_entrees", 5),
    "WEATHER_PER_AGENT_DATES": ("false", "meteo_par_agent", False),
}

# How long we accept to wait for the run identity (it is written in ~20 s), and the
# run itself (measured: 2 h 50 for 31 mobility days).
ATTENTE_IDENTITE_S = 180
ATTENTE_RUN_S = 6 * 3600
# Beyond one persona, the guard delay follows the volume: 90 s per agent and per simulated day.
# Six hours covered one inhabitant over 42 days; twenty inhabitants over fifteen days on a
# provider at ~4 requests/min need more, and an arm cut at the sixth hour
# would be counted as failed while it was progressing.
DELAI_PAR_AGENT_JOUR_S = 90
HORIZON_JOURS_DEFAUT = 42


def delai_de_garde(taille: int, horizon_jours: int) -> int:
    """The delay beyond which an arm is declared stuck. Never below `ATTENTE_RUN_S`."""
    return max(ATTENTE_RUN_S, int(taille) * int(horizon_jours) * DELAI_PAR_AGENT_JOUR_S)

PAS_DE_SONDAGE_S = 30
PAS_DE_JOURNAL_S = 300
# LIVED days observed after the extinction of the declared memory before cutting the run. Seven:
# one full simulated week, enough to see whether behaviour returns to what it was before, and
# it is the duration already used by `arret_sur_extinction.py` on the command line.
JOURS_APRES_EXTINCTION = 7

# The event played by default: the ticket 077 one, so that yesterday's command gives the same
# run as yesterday. `--evenement` designates another (campaign of 2026-09-22: `c3_panne_reseau`).
CHOC_REFERENCE = "c6_voiture_suspecte"
# The CANONICAL directory since ticket 100; `config/chocs/` is still read second, the five cases
# of 079 still living there. The derived file always goes to the canonical one: the Makefile
# prefers it when the name exists on both sides.
EVENEMENTS_DIR = Path("services/llm-agents/config/evenements")
CHOCS_DIR = Path("services/llm-agents/config/chocs")
# Populations are resolved from the repository ROOT, never from the current directory:
# it is this folder that the container mounts under `/data/eqasim-output`.
REPO_ROOT = Path(__file__).resolve().parents[2]
POPULATIONS_DIR = REPO_ROOT / "data" / "population"


def source_evenement(evenement: str) -> Path:
    """The declaration file of this event, canonical first, inherited next."""
    canonique = EVENEMENTS_DIR / f"{evenement}.yaml"
    if canonique.is_file():
        return canonique
    herite = CHOCS_DIR / f"{evenement}.yaml"
    if herite.is_file():
        return herite
    livres = sorted(p.stem for p in EVENEMENTS_DIR.glob("*.yaml"))
    raise FileNotFoundError(
        f"event not found: {evenement} — neither in {EVENEMENTS_DIR} nor in {CHOCS_DIR}. "
        f"Delivered cases: {', '.join(livres)}"
    )


def nom_choc_derive(persona_id: str, evenement: str = CHOC_REFERENCE) -> str:
    """The name of the derived shock for this persona. Pure: touches nothing."""
    return f"{evenement}__{persona_id}"


class PopulationIntrouvable(RuntimeError):
    """The designated population does not exist, or cannot be read: no run must start."""


def resoudre_population_et_taille(persona_or_pop_id: str) -> tuple[str, int, bool, list[str]]:
    """The CONTAINER path of the population JSON, its size, whether it is a set, its identifiers.

    Three accepted forms, in this order:
    1. a sealed set by its folder name — `population_20_foyers_059`;
    2. a single persona by its identifier — `861500` → `population_1_861500`;
    3. a path to a `population.json` under `data/population/`.

    A set is a population of SEVERAL agents. `population_1_861500` designated by its folder
    name stays single: what decides is the count read, not the form of the name.

    ⚠ PATHS ARE ANCHORED ON THE REPOSITORY ROOT. The first version of this function read
    `data/population` relative to the current directory: launched from elsewhere, it found
    nothing, fell back to the single case, and the arm exposed an "agent" named
    `population_20_foyers_059` — that is to say nobody, without a word. Hence the rule: a
    NUMERIC identifier without a folder stays the historical fallback (the file may exist only
    in the container), any other name not found is an error, and so is an unreadable file.
    """
    nom = str(persona_or_pop_id)
    direct_dir = POPULATIONS_DIR / nom
    unitaire_dir = POPULATIONS_DIR / f"population_1_{nom}"
    comme_chemin = Path(nom) if Path(nom).is_absolute() else REPO_ROOT / nom

    if (direct_dir / "population.json").is_file():
        cible = direct_dir / "population.json"
    elif (unitaire_dir / "population.json").is_file():
        cible = unitaire_dir / "population.json"
    elif comme_chemin.is_file():
        cible = comme_chemin
    elif nom.isdigit():
        logger.warning(
            f"[population] {POPULATIONS_DIR / f'population_1_{nom}'} missing on the host — single "
            f"fallback on the container path, with no possible check of the count."
        )
        return f"/data/eqasim-output/population_1_{nom}/population.json", 1, False, [nom]
    else:
        message = (
            f"[ALARME] [population] « {nom} » introuvable : ni {direct_dir}/population.json, ni "
            f"{unitaire_dir}/population.json, ni un fichier. Aucun bras ne part sur une "
            f"population qu'on ne sait pas lire."
        )
        logger.error(message)
        raise PopulationIntrouvable(message)

    try:
        relatif = cible.resolve().relative_to(POPULATIONS_DIR.resolve())
    except ValueError as exc:
        raise PopulationIntrouvable(
            f"[ALARME] [population] {cible} is outside {POPULATIONS_DIR}: the container does not "
            f"see it (only this folder is mounted under /data/eqasim-output)."
        ) from exc
    try:
        data = json.loads(cible.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        message = f"[ALARME] [population] {cible} illisible ({exc})"
        logger.error(message)
        raise PopulationIntrouvable(message) from exc

    ids = [
        str(item.get("person_id") or (item.get("identity") or {}).get("person_id"))
        for item in data
        if isinstance(item, dict)
        and (item.get("person_id") or (item.get("identity") or {}).get("person_id"))
    ]
    if len(ids) != len(data):
        message = (
            f"[ALARME] [population] {cible} : {len(data)} entrées mais {len(ids)} `person_id` "
            f"lisibles — on ne sait pas qui cibler."
        )
        logger.error(message)
        raise PopulationIntrouvable(message)
    return f"/data/eqasim-output/{relatif.as_posix()}", len(ids), len(ids) > 1, ids


def foyers_de_la_population(persona_or_pop_id: str) -> tuple[set[str], list[str]]:
    """(households present in the population, "expose" households of its MANIFEST).

    The manifest is the source of the exposed households of a household population (ticket 059,
    lot 6): it is the one that chose them, with their reason. Empty if there is no manifest.
    """
    import yaml

    direct = POPULATIONS_DIR / str(persona_or_pop_id)
    fichier = direct / "population.json"
    presents: set[str] = set()
    if fichier.is_file():
        for p in json.loads(fichier.read_text(encoding="utf-8")):
            foyer = (p.get("household") or {}).get("id") if isinstance(p, dict) else None
            if foyer:
                presents.add(str(foyer))
    exposes: list[str] = []
    manifeste = direct / "MANIFEST.yaml"
    if manifeste.is_file():
        groupes = (yaml.safe_load(manifeste.read_text(encoding="utf-8")) or {}).get("groupes") or {}
        exposes = [str(g["household_id"]) for g in (groupes.get("expose") or []) if g.get("household_id")]
    return presents, exposes


def lecteurs_du_manifeste(persona_or_pop_id: str, foyers: list[str]) -> list[str]:
    """The readers the MANIFEST designates for these exposed households (`expose[].lecteurs`).

    Empty if the manifest designates none: the reader is then drawn among the adults.
    """
    import yaml

    manifeste = POPULATIONS_DIR / str(persona_or_pop_id) / "MANIFEST.yaml"
    if not manifeste.is_file():
        return []
    groupes = (yaml.safe_load(manifeste.read_text(encoding="utf-8")) or {}).get("groupes") or {}
    voulus = {str(f) for f in foyers}
    return [
        str(pid)
        for g in (groupes.get("expose") or [])
        if str(g.get("household_id")) in voulus
        for pid in (g.get("lecteurs") or [])
    ]


def choc_pour_persona(persona_id: str, evenement: str = CHOC_REFERENCE) -> str:
    """Writes the reference event restricted to this run, and returns its name for `make run`.

    The derived file carries the same text, the same cadence and the same days as the reference.
    Only the exposure changes, and only for a SINGLE run: it carries only one inhabitant, and
    designating the others would only add inert identifiers to the declaration.

    ⚠ SINGLE RUN — THE RULE IS FORCED TO `agents`, not merely completed. Until
    2026-09-22 this function set `exposition.agents` leaving `regle` intact: on
    `c3_panne_reseau` (`regle: mode`) the agent list would have been written, accepted, and IGNORED.
    On a population of a single inhabitant the result is the same by accident, which is the
    worst of situations: the declaration says something other than what it does.

    ⚠ SET OF SEVERAL AGENTS — THE DECLARED EXPOSURE IS KEPT AS IS, and only two
    rules are allowed. `foyers` (the exposed households AND their control co-residents are in the
    declaration, taken from the population manifest) and `agents` (at least one of which must be
    in the population). Any other rule — `mode`, a draw — is REFUSED: rewriting it as
    `agents` would designate the population name as an inhabitant, and the run would unfold without
    anybody being exposed.

    `modes` is KEPT if declared. The `agents` branch reads it and restricts the designated
    agent to its trips in these modes — this is what keeps 861500 exposed to the metro outage
    on its public transport trips only, and not at the wheel of its car.
    """
    import yaml

    source = source_evenement(evenement)
    declaration = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    exposition = declaration.setdefault("exposition", {})
    regle_origine = exposition.get("regle")

    _, taille, est_un_jeu, ids = resoudre_population_et_taille(persona_id)
    if est_un_jeu:
        if regle_origine not in ("foyers", "agents"):
            message = (
                f"[ALARME] [{persona_id}] « {evenement} » déclare `regle: {regle_origine}` : sur "
                f"un jeu de {taille} agents, seules `foyers` et `agents` désignent quelqu'un. "
                f"Déclarer l'exposition dans l'événement, pas la laisser deviner."
            )
            logger.error(message)
            raise ValueError(message)
        if regle_origine == "foyers":
            foyers_pop, exposes_manifeste = foyers_de_la_population(persona_id)
            declares = [str(f) for f in exposition.get("foyers") or []]
            if foyers_pop and not set(declares) & foyers_pop:
                # The declared households are those of ANOTHER population (a09 carries those of
                # population_20_foyers_059): on this one, nobody would read. The population
                # manifest designates its own; failing that, the arm is refused.
                if not exposes_manifeste:
                    message = (
                        f"[ALARME] [{persona_id}] « {evenement} » expose les foyers {declares}, "
                        f"AUCUN n'est dans la population, et son MANIFEST n'en désigne pas — le run "
                        f"se déroulerait sans lecteur."
                    )
                    logger.error(message)
                    raise ValueError(message)
                logger.warning(
                    f"[{persona_id}] « {evenement} »: declared households {declares} absent from the "
                    f"population — exposure taken from the MANIFEST: {exposes_manifeste}."
                )
                exposition["foyers"] = exposes_manifeste
            # The designated readers come from the same manifest as the households. Those the
            # event declares, if they are not in the population, are those of another one.
            du_manifeste = lecteurs_du_manifeste(persona_id, exposition.get("foyers") or [])
            declares_l = [str(x) for x in exposition.get("lecteurs") or []]
            if declares_l and not set(declares_l) & set(ids):
                if not du_manifeste:
                    message = (
                        f"[ALARME] [{persona_id}] « {evenement} » désigne les lecteurs "
                        f"{declares_l}, AUCUN n'est dans la population, et son MANIFEST n'en "
                        f"désigne pas — le run se déroulerait sans lecteur."
                    )
                    logger.error(message)
                    raise ValueError(message)
                declares_l = []
            if du_manifeste and not declares_l:
                exposition["lecteurs"] = du_manifeste
                logger.info(
                    f"[{persona_id}] « {evenement} »: readers designated by the MANIFEST: "
                    f"{du_manifeste} (instead of a draw among the adults)."
                )
        if regle_origine == "agents":
            presents = set(map(str, exposition.get("agents") or [])) & set(ids)
            if not presents:
                message = (
                    f"[ALARME] [{persona_id}] « {evenement} » désigne "
                    f"{exposition.get('agents')} et AUCUN n'est dans la population — le run "
                    f"se déroulerait sans exposé."
                )
                logger.error(message)
                raise ValueError(message)
        portee = f"l'exposition déclarée (`{regle_origine}`) est conservée sur {taille} agents"
    else:
        exposition["regle"] = "agents"
        exposition["agents"] = [ids[0]]
        # `part` and `graine` only make sense for a draw: leaving them would declare a
        # parameter without effect, which the load guard already holds against the texts.
        for inerte in ("part", "graine"):
            exposition.pop(inerte, None)
        portee = (
            f"l'exposition est restreinte au persona {ids[0]} ; la règle passe de "
            f"`{regle_origine}` à `agents`, les modes déclarés sont conservés"
        )

    nom = nom_choc_derive(persona_id, evenement)
    cible = EVENEMENTS_DIR / f"{nom}.yaml"
    cible.write_text(
        f"# ENGENDRÉ par scripts/experiment/run_sequential_cohort.py — ne pas éditer.\n"
        f"# Source : {source.name}. {portee[0].upper() + portee[1:]}.\n"
        + yaml.safe_dump(declaration, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info(
        f"[{persona_id}] event derived from {source.name} → {cible.name} "
        f"(rule {regle_origine} → {exposition.get('regle')}, modes {exposition.get('modes') or 'tous'})"
    )
    return nom


def verifier_reprise_possible(workdir: Path) -> bool:
    """Does this workdir carry a SUSPENDED arm to resume?

    Only `reprise.json`, written at suspension, says so: it names the run to resume, and a
    resume is named (ticket 091). The previous version concluded to a resume as soon as a
    `moves.csv` and a `decisions_rejeu.jsonl` were there — that is, after a FINISHED arm,
    whose deliverables are copied here — then launched `CONT=1` without a name, which the controller refuses.
    """
    if reprise_en_attente(workdir) is not None:
        return True
    if (workdir / "moves.csv").is_file():
        logger.warning(
            f"[reprise] {workdir} carries the deliverables of an already FINISHED arm, without "
            f"`{FICHIER_REPRISE}`: it is replayed from scratch, not resumed."
        )
    return False


def preparer_workdir(
    workdir: Path,
    persona_id: str,
    branch: str,
    force_fresh: bool = False,
    horizon_jours: int = HORIZON_JOURS_DEFAUT,
) -> bool:
    """Initialises the isolated run folder for this persona and this branch."""
    workdir.mkdir(parents=True, exist_ok=True)
    # The shock is declared at `make run` (CHOC=…), not by putting a file in the workdir: the
    # controller writes ITS copy of the declaration there at the end of the run, it never reads it there.

    pop_file_container, pop_size, is_dataset, _ = resoudre_population_et_taille(persona_id)

    # 1. Configure sim_params.yaml for population_size
    sim_params_path = Path("services/GAMA/CityTransport/config/sim_params.yaml")
    if sim_params_path.is_file():
        import re
        text = sim_params_path.read_text(encoding="utf-8")
        text = re.sub(r"^population_size:\s*\d+", f"population_size: {pop_size}", text, flags=re.MULTILINE)
        text = re.sub(
            r"^simulation_max_days:\s*\d+",
            f"simulation_max_days: {int(horizon_jours)}",
            text,
            flags=re.MULTILINE,
        )
        # The GAMA lock must be active in both arms. This option lives in
        # sim_params.yaml (model parameter), not in the Python container alone.
        prefixe = os.environ.get("WORLD__PREFIXE_COMMUN", "false").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if re.search(r"^prefixe_commun:", text, flags=re.MULTILINE):
            text = re.sub(
                r"^prefixe_commun:\s*(?:true|false)",
                f"prefixe_commun: {str(prefixe).lower()}",
                text, flags=re.MULTILINE,
            )
        else:
            text += f"\nprefixe_commun: {str(prefixe).lower()}\n"
        sim_params_path.write_text(text, encoding="utf-8")

    # 2. Configure config.yaml to point to the resolved population file
    config_yaml_path = Path("services/llm-agents/config/config.yaml")
    if config_yaml_path.is_file():
        import re
        c_text = config_yaml_path.read_text(encoding="utf-8")
        c_text = re.sub(
            r"population_file:\s*/data/eqasim-output/.*",
            f"population_file: {pop_file_container}",
            c_text,
        )
        config_yaml_path.write_text(c_text, encoding="utf-8")

    is_resume = verifier_reprise_possible(workdir)
    if is_resume and not force_fresh:
        logger.info(f"[{persona_id}_{branch}] Hot resume detected in {workdir} (CONT=1 can be enabled).")
        return True
    elif force_fresh and workdir.exists():
        logger.info(f"[{persona_id}_{branch}] --force-fresh mode on: starting from scratch.")
        return False
    return False


def executer_run_persona(
    persona_id: str,
    branch: str,
    workdir: Path,
    is_resume: bool,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
    arret_extinction: bool = False,
    jours_apres_extinction: int = JOURS_APRES_EXTINCTION,
    evenement: str = CHOC_REFERENCE,
    horizon_jours: int = HORIZON_JOURS_DEFAUT,
) -> int:
    """Launches a GAMA/Python run of `horizon_jours` calendar days (42 by default)."""
    env = os.environ.copy()
    env["WORKDIR"] = str(workdir.resolve())
    env["EXPERIMENT_SURVEY_ENABLED"] = "1"
    # Ticket 105 — arms both stop reasons: daily quota AND consecutive fallbacks.
    # Every campaign is thus protected by default, with no lever to remember at launch.
    env["EXPERIMENT_STOP_ON_FALLBACK"] = "1"
    pop_file_container, pop_size, is_dataset, agent_ids = resoudre_population_et_taille(persona_id)
    if is_dataset and agent_ids:
        env["EXPERIMENT_TARGET_PERSONAS"] = ",".join(agent_ids)
    else:
        env["EXPERIMENT_TARGET_PERSONAS"] = str(persona_id)
    for nom, (valeur, _champ, _attendu) in REGLAGES_CAMPAGNE.items():
        env[nom] = valeur
    env["NO_WEEKEND_DEPARTURES"] = "true"
    # ⚠ `MEMOIRE__IMPORTANCE_CHOC=0.50` REMOVED on 2026-09-19. The D15 shock is worth exactly
    # 0.70 and the comparison is `>=`: it crosses the original threshold without lowering it.
    # Lowering it only widened pool C, that is to say out-of-context recall.
    # Ticket 095, lot C — ONE MODEL PER FUNCTION, adopted on 2026-09-21.
    #
    # Until now, a single flat list: the 751 requests of the ticket 077 campaign all
    # went to the same family, and the two biggest items — the decision and the
    # evening reflection, 46 % of input tokens each — competed for the same key.
    #
    # The DECISION is the measured variable of the paper: it keeps `gemini-3.1`, unchanged, to
    # stay comparable with previous campaigns. Self-reflection follows it — rare (13 per run)
    # and high-stakes, it rereads the whole long-term memory. The evening reflection and the survey
    # move to `gemini-3.5`.
    #
    # ⚠ THE NAME IS `INSTANCES_ADMISES`, the BARE name of the field. This module set
    # `LLM__INSTANCES_ADMISES` from 2026-09-08 to 2026-09-21: wrong name, absent from the compose
    # pass-through, and `config.yaml` prevailed anyway. The restriction of the September
    # campaigns thus came from `config.yaml`, not from here — no arm ever declared its
    # own. Checked on `identite_run.json` of 2026-09-21 09:23.
    #
    # ⚠ TWO INSTANCES MINIMUM PER CATEGORY, and never `force_provider`: the worker only retries
    # if no instance is pinned. The fourteen `HTTP 503 high demand` of the campaign
    # of 19 September got through precisely because a second key was left.
    #
    # ⚠ THIS IS NOT AN INFRASTRUCTURE SETTING. Changing the model of the reflections changes the
    # content of the memory, hence the decisions. The binding goes into `identite_run.json` and
    # must stay IDENTICAL in all arms of a campaign — otherwise the measured gap is no longer
    # attributable to the shock.
    if "INSTANCES_ADMISES" in os.environ and os.environ["INSTANCES_ADMISES"]:
        env["INSTANCES_ADMISES"] = os.environ["INSTANCES_ADMISES"]
    else:
        env["INSTANCES_ADMISES"] = json.dumps(
            {
                "defaut": ["google_gemini31_key1", "google_gemini31_key2"],
                "itinary_multi_agent": ["google_gemini31_key1", "google_gemini31_key2"],
                "ltm_self_reflection": ["google_gemini31_key1", "google_gemini31_key2"],
                "stm_reflection": ["google_gemini35_key1", "google_gemini35_key2"],
                # 2026-09-24: the surveys join 3.1. Measured on 861500: STM weighs
                # 49 % of requests and the decision 43 % — 3.5 alone carried STM + surveys.
                "enquete_affinite": ["google_gemini31_key1", "google_gemini31_key2"],
            }
        )
    env["DATA__POPULATION_SIZE"] = str(pop_size)

    # Ticket 091 — a resume is NAMED. `CONT=1` alone makes the controller refuse to start
    # (`CONTINUE_RUN sans REPRISE`): it is the name of the suspended run, written in `reprise.json`,
    # that makes the resume.
    reprise = reprise_en_attente(workdir) if is_resume else None
    if is_resume and reprise is None:
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] resume requested without `{FICHIER_REPRISE}` in "
            f"{workdir} — we do not know which run to resume. Arm REFUSED."
        )
        return 8

    # The window ablation arm goes through here:
    #   extra_env={"MEMOIRE__FENETRE_CHANGEMENTS_JOURS": "7"}
    # Everything set in `env` ends up in `identite_run.json` (ticket 077, lot K),
    # so an arm can no longer be mistaken for another afterwards.
    if extra_env is None:
        extra_env = {}
        for var in [
            "MEMOIRE__FENETRE_CHANGEMENTS_JOURS",
            "MEMOIRE__CHANGEMENTS_MAX",
            "MEMOIRE__IMPORTANCE_CHOC",
            # Ticket 095, lot A — the "derived window" arm and the "fixed window" control
            # arm are declared through MEMOIRE__MODE_FENETRE_CHANGEMENTS.
            "MEMOIRE__MODE_FENETRE_CHANGEMENTS",
            "MEMOIRE__SEUIL_SERVICE_CHANGEMENT",
            "MEMOIRE__PLANCHER_CHANGEMENT_JOURS",
            "MEMOIRE__PLAFOND_CHANGEMENT_JOURS",
            # The severity model: it decides what a long delay is worth.
            "MEMOIRE__RETARD_SATURATION",
            "MEMOIRE__RETARD_GRAVITE_MAX",
            "MEMOIRE__RETARD_REF_S",
            # Ticket 100, lot 4 — sharing within the household belongs to the experiment.
            "MEMOIRE__PARTAGE_FOYER_ENABLED",
            # Tasks in flight: they set the size of micro-batches, hence the prompts served.
            "WORLD__WORKER_CONCURRENCY",
            # 2026-09-25 — exact-prompt replay space, shared by both arms.
            "REJEU_AB",
            "REJEU_STRICT_AVANT_TS",
            "WORLD__PREFIXE_COMMUN",
            # Ticket V5 — Daily surveys and adapter timeouts
            "EXPERIMENT_SURVEY_DAYS",
            "EXPERIMENT_SURVEY_ENABLED",
            "GOOGLE_ADAPTER_REQUEST_TIMEOUT",
        ]:
            if var in os.environ:
                extra_env[var] = os.environ[var]

    if extra_env:
        env.update(extra_env)

    # CHOC: this is WHERE the two branches split. Without it, `make run` leaves the
    # declaration of `config.yaml` in place and the control undergoes the treated arm's shock.
    # CACHE=0: the semantic cache key carries no duration, a decision taken before the
    # shock could be served again during it.
    choc = nom_choc_derive(persona_id, evenement) if branch == "treated" else "0"
    cmd = ["make", "run", "OFFLINE=1", "CACHE=0", f"CHOC={choc}"]
    if reprise:
        cmd.append(f"REPRISE={reprise['archive']}")

    delai = delai_de_garde(pop_size, horizon_jours)
    logger.info(
        f"[{persona_id}_{branch}] Démarrage de la simulation ({horizon_jours} jours calendaires, "
        f"{pop_size} agent(s), délai de garde {delai / 3600:.1f} h)..."
    )
    logger.info(f"[{persona_id}_{branch}] Workdir: {workdir}")
    logger.info(
        f"[{persona_id}_{branch}] Hot resume: "
        f"{('REPRISE=' + reprise['archive']) if reprise else 'non'}"
    )
    logger.info(f"[{persona_id}_{branch}] Command: {' '.join(cmd)}")

    if dry_run:
        logger.info(f"[dry-run] Nothing is launched for {persona_id}_{branch}.")
        return 0

    if lanceur_en_cours():
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] a GAMA launcher is already running — arm REFUSED. "
            f"`make run` would merely write « Lancement ignoré » and return 0, and this "
            f"arm would watch someone else's run. Stop first: make stop-run"
        )
        return 3

    if branch == "treated":
        choc_pour_persona(persona_id, evenement)


    # Process execution. The launch instant dates the markers: only a waiting
    # marker written AFTER it suspends this arm.
    lance_a = time.time()
    gama_depuis = (
        taille_journal_gama(Path("experiments/archive") / reprise["archive"]) if reprise else 0
    )
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_path = workdir / "run_orchestrateur.log"
    with open(log_path, "a", encoding="utf-8") as lf:
        if process.stdout:
            for line in process.stdout:
                lf.write(line)
                if any(k in line for k in ["[enquete]", "[hibernation]", "[choc]", "ERROR", "WARNING", "✅", "⏳", "🚀", "🛑", "♻️", "Services prêts", "Controller", "API prête", "Lancement"]):
                    sys.stdout.write(f"[{persona_id}_{branch}] {line}")
                    sys.stdout.flush()

    process.wait()
    retcode = process.returncode
    if retcode != 0:
        logger.error(f"[ALARME] [{persona_id}_{branch}] `make run` returned {retcode} — arm abandoned.")
        return retcode

    # ⚠ `make run` has returned, NOT the run: it launches GAMA in the background. Everything that follows
    # happens while the simulation is running.

    # 1. Check the settings BEFORE paying for three hours. The identity is written in ~20 s.
    archive = attendre_identite(
        controller_demarre_a(), reprise=reprise["archive"] if reprise else None
    )
    if archive is None:
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] no run identity LATER than the "
            f"controller start after {ATTENTE_IDENTITE_S} s — without it we can assert nothing "
            f"about this arm. Stopping. (A controller launched before `make run`, for example "
            f"by `make up`, may have left a stale identity in the same directory.)"
        )
        subprocess.run(["make", "stop-run"], check=False)
        return 4

    attendus = reglages_attendus(extra_env)
    ecarts = ecarts_de_reglages(archive, attendus)
    if ecarts:
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] the arm is not running with its settings — "
            f"immediate stop:\n  - " + "\n  - ".join(ecarts)
            + f"\n  ({archive / 'identite_run.json'})\n"
            f"  Check that these variables are declared as pass-through in "
            f"infra/docker-compose.yml: compose only passes on what it declares."
        )
        subprocess.run(["make", "stop-run"], check=False)
        return 5
    logger.info(
        f"[{persona_id}_{branch}] settings compliant ({len(attendus)} checked) — "
        f"{archive.name}"
    )
    journaliser_qui_sert(persona_id, branch, archive)

    # 2. Wait for the REAL end of the run.
    if not attendre_fin_du_run(
        persona_id,
        branch,
        archive,
        timeout_s=delai,
        arret_extinction=arret_extinction,
        jours_apres_extinction=jours_apres_extinction,
        lance_a=lance_a,
        gama_depuis=gama_depuis,
    ):
        return 6

    # The launcher has gone — which ALSO happens when the ticket 105 safeguard has stopped the
    # controller. On 2026-09-24, this case was counted as a success: the orchestrator launched the
    # control on the exhausted quota, then marked the experiment "finished" after one simulated day.
    suspension = suspension_du_run(archive, depuis=lance_a)
    if suspension is not None:
        (workdir / FICHIER_REPRISE).write_text(
            json.dumps(
                {"archive": archive.name, **suspension,
                 "suspendu_le": datetime.now().isoformat(timespec="seconds")},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] arm SUSPENDED by the safeguard "
            f"(reason {suspension.get('motif')}, simulated day {suspension.get('jour_simule')}, "
            f"reopening {suspension.get('resume_at') or 'non annoncée'}) — run {archive.name}. "
            f"Relaunching the same experiment will resume it (REPRISE={archive.name})."
        )
        return CODE_BRAS_SUSPENDU

    reprise_faite = workdir / FICHIER_REPRISE
    if reprise_faite.is_file():
        # The resumed arm went to the end: we keep the trace of the suspension, under a name that
        # will no longer trigger a resume.
        reprise_faite.rename(workdir / f"reprise_faite_{datetime.now():%Y%m%d_%H%M%S}.json")
    logger.info(f"[{persona_id}_{branch}] Bras terminé.")
    return 0


# Return code of an arm that the ticket 105 safeguard SUSPENDED (quota or fallbacks): neither a
# success — the run did not go to the end —, nor a final failure — it can be resumed.
CODE_BRAS_SUSPENDU = 7
FICHIER_REPRISE = "reprise.json"


def suspension_du_run(archive: Path | None, depuis: float | None = None) -> dict | None:
    """The waiting marker the controller set DURING this arm, or None.

    ⚠ A resumed arm replays in the SAME directory as the one that had stopped: the marker of
    the previous stop is still there. Only a marker written since the launch of this arm says that
    THIS arm was suspended.
    """
    if archive is None:
        return None
    marqueur = archive / "en_attente_quota.json"
    if not marqueur.is_file():
        return None
    if depuis is not None and marqueur.stat().st_mtime < depuis - 5:
        return None
    try:
        return json.loads(marqueur.read_text(encoding="utf-8")) or {"motif": "inconnu"}
    except (OSError, ValueError):
        return {"motif": "illisible"}


def reprise_en_attente(workdir: Path) -> dict | None:
    """The suspended arm this workdir must resume (`reprise.json`), or None."""
    fichier = workdir / FICHIER_REPRISE
    if not fichier.is_file():
        return None
    try:
        donnees = json.loads(fichier.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return donnees if donnees.get("archive") else None


def lanceur_en_cours() -> bool:
    """Is a headless GAMA launcher already running?

    `make run` merely writes « Lancement ignoré » on its standard output and returns 0:
    without this guard, an arm would believe it is running while it watches someone else's run.
    """
    return (
        subprocess.run(
            ["pgrep", "-f", "launch_headless.py"], capture_output=True, text=True
        ).returncode
        == 0
    )


def derniere_journee_simulee(archive: Path | None) -> str:
    """The last simulated day reached, to log progress and not silence.

    ⚠ `[sync] END` only appears once the simulation loop has started. During bootstrap —
    population, initial itineraries, which last all the longer as the OSMnx cache is cold — there
    is nothing to read, and the heartbeat showed "?". On a six-hour run, "?" does not
    tell "it is bootstrapping" from "it is stuck", which is precisely the question this
    heartbeat exists to settle. So it now reports bootstrap when that is where we are.
    """
    if archive is None:
        return "?"
    log = archive / "app.log"
    if not log.is_file():
        return "journal pas encore écrit"
    try:
        lignes = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "?"
    for ligne in reversed(lignes[-4000:]):
        if "[sync] END sim_time=" in ligne:
            return ligne.split("sim_time=", 1)[1].split(" state_update", 1)[0].strip()
    # Early detection of a fatal error at initialisation (avoids running idle)
    for ligne in reversed(lignes[-4000:]):
        if any(err in ligne for err in ["[ALARME] [population]", "RuntimeError: [population]", "Error in ASGI Framework"]):
            return f"échec initialisation — {ligne.strip()[:70]}"
    # Not a single cycle yet: say where bootstrap is rather than returning a question
    # mark, which reads as a breakdown.
    for ligne in reversed(lignes[-4000:]):
        if "[bootstrap]" in ligne:
            return f"amorçage — {ligne.split('[bootstrap]', 1)[1].strip()[:70]}"
        if "INITIALISATION" in ligne:
            return f"amorçage — {ligne.split('INITIALISATION', 1)[1].strip()[:70]}"
    return "aucun cycle, aucun amorçage journalisé"


def date_du_souvenir_declare(archive: Path | None) -> str | None:
    """The simulated date of the LAST application of the declared event, or None.

    It is what the early stop watches — not "plus aucun souvenir de choc ne pèse",
    which only comes at the very end of the run since the agent makes its own shock-severity
    memories (the control arm, which undergoes nothing, produces three).

    The LAST and not the first: a two-day event (c6: D15 then D16) is only extinct
    when the second memory has left. Watching the first would stop the run while
    the agent still carries the second in its context.

    Returns None as long as the trace is empty — a run whose event has not yet been applied
    cannot have died out, and the wait goes on.
    """
    if archive is None:
        return None
    trace = archive / "evenements.jsonl"
    if not trace.is_file():
        return None
    dates: list[str] = []
    try:
        with open(trace, encoding="utf-8", errors="replace") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    horodatage = json.loads(ligne).get("horodatage_simule")
                except json.JSONDecodeError:
                    continue  # truncated line: the file is written while we read it
                if isinstance(horodatage, str) and len(horodatage) >= 10:
                    dates.append(horodatage[:10])
    except OSError:
        return None
    return max(dates) if dates else None


def journaliser_qui_sert(persona_id: str, branch: str, archive: Path | None) -> None:
    """Says WHICH MODELS will produce this arm, at launch and not afterwards.

    The instance restriction is declared per category in `infra/docker-compose.yml` and
    appeared nowhere in the launcher output: one had to enter the container or
    open `identite_run.json` to learn that the decisions went through two Gemini 3.1
    flash-lite keys and nothing else. On 2026-09-22, this led to a false announcement — "the full
    pool" — while the run was running on two keys of a single model, which started
    returning 503.

    An arm whose producer is unknown compares with nothing. The line is therefore in the
    log, at the moment the arm starts.
    """
    if archive is None:
        return
    identite = archive / "identite_run.json"
    if not identite.is_file():
        logger.warning(
            f"[{persona_id}_{branch}] identite_run.json absent : impossible de dire quels "
            f"modèles servent ce bras."
        )
        return
    try:
        d = json.loads(identite.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[{persona_id}_{branch}] identite_run.json illisible ({exc}).")
        return
    modeles = d.get("modeles_admis") or ["(non déclaré)"]
    routage = d.get("routage_instances") or {}
    logger.info(f"[{persona_id}_{branch}] admitted models: {', '.join(modeles)}")
    for categorie in ("itinary_multi_agent", "stm_reflection", "ltm_self_reflection"):
        instances = routage.get(categorie)
        if instances:
            logger.info(f"[{persona_id}_{branch}]   {categorie} ← {', '.join(instances)}")
    if not routage:
        logger.info(
            f"[{persona_id}_{branch}]   no routing restriction: all declared "
            f"instances may serve."
        )


def taille_journal_gama(archive: Path | None) -> int:
    """Bytes already written to `gama_headless.log` BEFORE the launch of an arm (2026-09-28).

    A resumed arm writes into the log of the attempt it resumes: its end still carries
    the prefix alarm of the previous suspension ("simulation à invalider", written by
    GAMA when the controller stops). Reread at restart, it stopped the resumed arm
    before its first step — a13 v5 control, 19:55, code 6 — and a night retry
    failed instead of resuming. So we note where the log is, and read
    only what THIS arm adds to it.
    """
    if archive is None:
        return 0
    try:
        return (archive / "gama_headless.log").stat().st_size
    except OSError:
        return 0


def lire_journal_gama(gama_log: Path, depuis: int, fin: int) -> str:
    """The last `fin` characters of what GAMA wrote to `gama_log` since byte `depuis`.

    A log shorter than `depuis` has been replaced (set aside then recreated): everything it
    contains then belongs to this arm.
    """
    with gama_log.open("rb") as f:
        taille = f.seek(0, os.SEEK_END)
        if taille < depuis:
            depuis = 0
        # 4 bytes per character at most in UTF-8: enough to return `fin` whole characters.
        f.seek(max(depuis, taille - 4 * fin))
        return f.read().decode("utf-8", errors="replace")[-fin:]


def attendre_fin_du_run(
    persona_id: str,
    branch: str,
    archive: Path | None,
    timeout_s: int = ATTENTE_RUN_S,
    arret_extinction: bool = False,
    jours_apres_extinction: int = JOURS_APRES_EXTINCTION,
    lance_a: float | None = None,
    gama_depuis: int = 0,
) -> bool:
    """Waits until the GAMA launcher has gone. Returns False if the guard delay is exceeded.

    `gama_depuis`: size of `gama_headless.log` at the arm's launch (`taille_journal_gama`).
    Clean end and synchronisation failure are only read in what has been written since.

    `lance_a`: instant of the arm's launch. An older waiting marker — the one of the
    suspension a resumed arm has come to resume, left in the same directory — does not make
    a dead launcher pass for a clean end (2026-09-28).

    ⚠ `make run OFFLINE=1` DOES NOT BLOCK: its recipe launches `launch_headless.py` in the background
    (`&`) and returns within twenty seconds or so. On 2026-09-19, the orchestrator took
    this for the end of the run, moved on to the next arm, whose `make stop-run` killed the first
    thirty-four seconds after it started.

    The launcher, for its part, lives exactly as long as the run: GAMA Server kills the experiment as soon as its
    WebSocket client disconnects. That is the clear signal.

    ⚠ `arret_extinction` SHORTENS THE ARM, and two arms of different lengths do not
    compare day by day. The control undergoes no declared event: its trace is empty,
    no date is found, and it will therefore run to its full horizon while the treated
    arm stops earlier. This is intended — the days paid after extinction teach
    nothing — but the analysis MUST truncate both arms at the same simulated day. The file
    `arret_sur_extinction.json` written in the run directory carries that day; it is what we
    read, not the length of the log.
    """
    debut = time.monotonic()
    if gama_depuis:
        logger.info(
            f"[{persona_id}_{branch}] gama_headless.log : {gama_depuis} octet(s) hérités d'une "
            f"tentative précédente — ignorés par la détection de fin et d'échec"
        )
    dernier_journal = 0.0
    souvenir_du: str | None = None
    extinction_annoncee = False
    dernier_examen = -PAS_DE_JOURNAL_S  # first check on the first round, without waiting 5 min
    while lanceur_en_cours():
        ecoule = time.monotonic() - debut
        if ecoule > timeout_s:
            logger.error(
                f"[ALARME] [{persona_id}_{branch}] guard delay exceeded "
                f"({timeout_s // 3600} h) and the launcher is still running. The run is NOT killed: "
                f"a slow run is not a dead run, it is up to the author to decide."
            )
            return False
        if ecoule - dernier_journal >= PAS_DE_JOURNAL_S:
            dernier_journal = ecoule
            derniere = derniere_journee_simulee(archive)
            logger.info(
                f"[{persona_id}_{branch}] running for {int(ecoule // 60)} min — "
                f"simulated day: {derniere}"
            )
            if "échec initialisation" in derniere:
                logger.error(
                    f"[ALARME] [{persona_id}_{branch}] Échec d'initialisation détecté dans app.log — arrêt du run."
                )
                subprocess.run(["make", "stop-run"], check=False)
                return False
        # Early stop: the declared memory has left the block N LIVED days ago.
        # Checked at the log step (5 min) and not at the polling step (30 s): the detection rereads
        # the whole app.log, which reaches tens of megabytes on a fifty-day
        # campaign. Extinction is an event at the scale of the simulated day — five minutes of
        # latency cost nothing, sixty useless rereads per half-hour do.
        if arret_extinction and archive and (ecoule - dernier_examen) >= PAS_DE_JOURNAL_S:
            dernier_examen = ecoule
            if souvenir_du is None:
                souvenir_du = date_du_souvenir_declare(archive)
                if souvenir_du:
                    logger.info(
                        f"[{persona_id}_{branch}] stop on extinction armed — declared memory "
                        f"of {souvenir_du}, observing {jours_apres_extinction} lived days "
                        f"after it leaves the \"What changed recently\" block."
                    )
            journal = archive / "app.log"
            if souvenir_du and journal.is_file():
                try:
                    with open(journal, encoding="utf-8", errors="replace") as f:
                        vue, ecoules = analyser_extinction(f, souvenir_du)
                except OSError:
                    vue, ecoules = False, 0
                if vue and not extinction_annoncee:
                    extinction_annoncee = True
                    logger.info(
                        f"[{persona_id}_{branch}] extinction of the memory of {souvenir_du} "
                        f"detected — {jours_apres_extinction} more lived days to observe."
                    )
                if vue and ecoules >= jours_apres_extinction:
                    dernier_jour = derniere_journee_simulee(archive)
                    logger.info(
                        f"[{persona_id}_{branch}] {ecoules} lived days since the extinction of "
                        f"{souvenir_du} (last simulated day: {dernier_jour}) — early stop. "
                        f"Truncate the control arm at this day to compare."
                    )
                    try:
                        (archive / "arret_sur_extinction.json").write_text(
                            json.dumps(
                                {
                                    "souvenir_du": souvenir_du,
                                    "jours_apres_extinction": ecoules,
                                    "seuil_jours": jours_apres_extinction,
                                    "dernier_jour_simule": dernier_jour,
                                    "persona": persona_id,
                                    "branche": branch,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        logger.error(
                            f"[ALARME] [{persona_id}_{branch}] early stop decided but the "
                            f"proof could not be written ({exc}): the arm is truncated and "
                            f"nothing says at which day. The paired analysis is compromised."
                        )
                    time.sleep(3)
                    subprocess.run(["make", "stop-run"], check=False)
                    break

        # End-of-simulation detection through gama_headless.log (anti-stall safety)
        if archive:
            gama_log = archive / "gama_headless.log"
            if gama_log.is_file():
                try:
                    tail_txt = lire_journal_gama(gama_log, gama_depuis, 3000)
                    if "Simulation stopped after" in tail_txt:
                        logger.info(
                            f"[{persona_id}_{branch}] 'Simulation stopped after' détecté dans {gama_log.name} — "
                            f"arrêt propre du run."
                        )
                        time.sleep(3)
                        subprocess.run(["make", "stop-run"], check=False)
                        break
                    if any(err in tail_txt for err in ["Préfixe commun interrompu", "simulation à invalider"]):
                        logger.error(
                            f"[ALARME] [{persona_id}_{branch}] Échec de synchronisation détecté dans {gama_log.name} — "
                            f"arrêt immédiat du run."
                        )
                        subprocess.run(["make", "stop-run"], check=False)
                        return False
                except OSError:
                    pass
        time.sleep(PAS_DE_SONDAGE_S)
    logger.info(
        f"[{persona_id}_{branch}] launcher finished after "
        f"{int((time.monotonic() - debut) // 60)} min."
    )

    # Check that the run ended on a valid criterion and not on a premature stop
    termine_proprement = False
    if arret_extinction and archive and (archive / "arret_sur_extinction.json").is_file():
        termine_proprement = True
    elif suspension_du_run(archive, depuis=lance_a) is not None:
        termine_proprement = True
    elif archive and (archive / "gama_headless.log").is_file():
        try:
            log_txt = lire_journal_gama(archive / "gama_headless.log", gama_depuis, 5000)
            if "Simulation stopped after" in log_txt:
                termine_proprement = True
        except OSError:
            pass

    if not termine_proprement:
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] The launcher disappeared prematurely without reaching "
            f"the end of simulation ('Simulation stopped after' missing). Arm failed."
        )
        return False

    return True


COMPOSE = [
    "docker", "compose", "-f", "infra/docker-compose.yml", "--project-directory", ".",
]


def controller_demarre_a() -> float | None:
    """Start instant of the controller container, in epoch seconds. None if unavailable."""
    try:
        cid = subprocess.run(
            [*COMPOSE, "ps", "-q", "controller"], capture_output=True, text=True, timeout=30
        ).stdout.strip().splitlines()
        if not cid:
            return None
        brut = subprocess.run(
            ["docker", "inspect", cid[0], "--format", "{{.State.StartedAt}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not brut:
        return None
    # RFC3339 to the nanosecond: `datetime.fromisoformat` stops at the microsecond.
    brut = brut.replace("Z", "+00:00")
    if "." in brut:
        tete, reste = brut.split(".", 1)
        frac, _, fuseau = reste.partition("+")
        brut = f"{tete}.{frac[:6]}+{fuseau}"
    try:
        return datetime.fromisoformat(brut).timestamp()
    except ValueError:
        return None


def attendre_identite(
    depuis: float | None = None,
    timeout_s: int = ATTENTE_IDENTITE_S,
    reprise: str | None = None,
) -> Path | None:
    """Waits for a run identity written AFTER `depuis`, and returns the directory that carries it.

    ⚠ Freshness is not a luxury. The run directory name has a granularity of one
    MINUTE: two controllers started within the same minute share it, and `identite_run.ecrire`
    leaves the identity already set in place — that is the ticket 091 rule, and it is a good one.
    Consequence measured on 2026-09-19 at 16:57: a controller launched by `make up` (without the
    experiment settings, without the shock) wrote the identity at 16:57:12, the run's controller
    replaced it at 16:57:35 — and the arm was judged on the identity of the first, which announced
    « choc : aucun » and the default values.
    """
    debut = time.monotonic()
    prevenu = False
    while time.monotonic() - debut < timeout_s:
        archive = archive_du_run()
        fichier = (archive / "identite_run.json") if archive else None
        if reprise is not None:
            # NAMED RESUME: the identity is that of the resumed run, set at its first start and
            # left in place by the ticket 091 rule — it is therefore EARLIER than the controller,
            # by construction. What is authoritative here is the NAME of the directory served.
            if archive is not None and archive.name == reprise and fichier and fichier.is_file():
                return archive
            time.sleep(2)
            continue
        if fichier and fichier.is_file():
            if depuis is None or fichier.stat().st_mtime >= depuis - 5:
                return archive
            if not prevenu:
                prevenu = True
                logger.info(
                    f"[identite] {fichier} is older than the controller start — "
                    f"identity of a previous controller, waiting for the right one."
                )
        time.sleep(2)
    return None


def ecarts_de_reglages(archive: Path, attendus: dict[str, Any]) -> list[str]:
    """The run settings that are not the requested ones, named one by one."""
    fichier = archive / "identite_run.json"
    try:
        identite = json.loads(fichier.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"identite_run.json illisible ({exc})"]
    ecarts = []
    for champ, attendu in attendus.items():
        obtenu = identite.get(champ, "<absent>")
        if isinstance(attendu, float) and isinstance(obtenu, (int, float)):
            identique = abs(float(obtenu) - attendu) < 1e-9
        else:
            identique = obtenu == attendu
        if not identique:
            ecarts.append(f"{champ} : attendu {attendu!r}, obtenu {obtenu!r}")
    return ecarts


def reglages_attendus(extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """What the run identity must carry, deduced from what we set — never hard-coded."""
    attendus = {champ: valeur for _, (_, champ, valeur) in REGLAGES_CAMPAGNE.items()}
    for nom, brut in (extra_env or {}).items():
        if nom == "MEMOIRE__FENETRE_CHANGEMENTS_JOURS":
            attendus["fenetre_changements_jours"] = int(brut)
        elif nom == "MEMOIRE__CHANGEMENTS_MAX":
            attendus["changements_max"] = int(brut)
        elif nom == "MEMOIRE__IMPORTANCE_CHOC":
            attendus["seuil_choc"] = float(brut)
        elif nom == "MEMOIRE__MODE_FENETRE_CHANGEMENTS":
            attendus["mode_fenetre_changements"] = brut
        elif nom == "MEMOIRE__SEUIL_SERVICE_CHANGEMENT":
            attendus["seuil_service_changement"] = float(brut)
        elif nom == "MEMOIRE__PLANCHER_CHANGEMENT_JOURS":
            attendus["plancher_changement_jours"] = float(brut)
        elif nom == "MEMOIRE__PLAFOND_CHANGEMENT_JOURS":
            attendus["plafond_changement_jours"] = float(brut)
        elif nom == "MEMOIRE__RETARD_SATURATION":
            attendus["retard_saturation"] = brut
        elif nom == "MEMOIRE__RETARD_GRAVITE_MAX":
            attendus["retard_gravite_max"] = float(brut)
        elif nom == "MEMOIRE__RETARD_REF_S":
            attendus["retard_ref_s"] = int(brut)
        elif nom == "MEMOIRE__PARTAGE_FOYER_ENABLED":
            attendus["partage_foyer"] = str(brut).strip().lower() in ("1", "true", "yes", "on")
        elif nom == "WORLD__WORKER_CONCURRENCY":
            attendus["taches_en_vol"] = int(brut)
        elif nom == "REJEU_AB":
            # A container that did not receive it would pay the whole control without saying so.
            attendus["rejeu_ab"] = brut
        elif nom == "WORLD__PREFIXE_COMMUN":
            attendus["prefixe_commun"] = brut.strip().lower() in ("1", "true", "yes", "on")
        elif nom == "REJEU_STRICT_AVANT_TS":
            attendus["rejeu_strict_avant_ts"] = int(brut)
    return attendus


def archive_du_run() -> Path | None:
    """The directory where the run has ACTUALLY just written.

    ⚠ `WORKDIR` is read by nobody: the controller always creates
    `experiments/archive/<AAAA-MM-JJ>_<HH_MM>` and makes `experiments/current` point to it. The
    per-branch `workdir` of this script is therefore a bookkeeping convenience, not a
    destination. § 10.7 of ticket 077 puts it differently: « `experiments/current` ne prouve
    rien » — we resolve it here, right after the run, while it still points to the right
    place, and we write it into the campaign manifest.
    """
    lien = Path("experiments/current")
    if not lien.exists():
        return None
    try:
        return lien.resolve()
    except OSError:
        return None


def consigner_au_manifeste(base_dir: Path, persona_id: str, branch: str, archive: Path | None) -> None:
    """The campaign manifest: which arm produced which archive directory."""
    manifeste = base_dir / "manifeste.json"
    entrees = []
    if manifeste.is_file():
        try:
            entrees = json.loads(manifeste.read_text(encoding="utf-8"))
        except ValueError:
            entrees = []
    entrees.append(
        {
            "persona": str(persona_id),
            "branche": branch,
            "archive": str(archive) if archive else None,
            "identite": str(archive / "identite_run.json") if archive else None,
            "horodatage": datetime.now().isoformat(timespec="seconds"),
        }
    )
    manifeste.parent.mkdir(parents=True, exist_ok=True)
    manifeste.write_text(json.dumps(entrees, ensure_ascii=False, indent=2), encoding="utf-8")
    if archive is None:
        logger.error(
            f"[ALARME] [{persona_id}_{branch}] archive directory not found after the run: "
            f"the arm cannot be paired with its identity."
        )
    else:
        logger.info(f"[{persona_id}_{branch}] arm archive: {archive}")


def generer_analyses_post_run(workdir: Path, persona_id: str, branch: str) -> None:
    """Generates the charts and the dynamic report after the end of the run."""
    moves_csv = workdir / "moves.csv"
    if not moves_csv.is_file():
        logger.warning(f"[{persona_id}_{branch}] moves.csv not found in {workdir} — analysis skipped.")
        return

    reports_dir = workdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/analysis/modal_variation_rate.py",
        str(moves_csv),
        "-o",
        str(reports_dir),
    ]
    logger.info(f"[{persona_id}_{branch}] Génération du rapport de transition modale dans {reports_dir}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        logger.info(f"[{persona_id}_{branch}] Charts and SVG report generated successfully.")
    else:
        logger.error(f"[{persona_id}_{branch}] Error during the analysis: {res.stderr}")

    plot_script = Path("scripts/analysis/plot_agent_itineraries.py")
    if plot_script.is_file():
        cmd_html = [
            sys.executable,
            str(plot_script),
            "--moves-csv",
            str(moves_csv),
            "--html-report",
            str(reports_dir / "itineraires.html"),
            "--export-svg",
            "--output-dir",
            str(reports_dir / "svg"),
        ]
        logger.info(f"[{persona_id}_{branch}] Generating the interactive HTML viewer and the SVGs...")
        subprocess.run(cmd_html, capture_output=True, text=True)

    # Cognitive memory report
    cmd_mem = [
        sys.executable,
        "-m",
        "scripts.analysis.memoire.rapport",
        str(workdir if (workdir / "agent_memory_events.csv").is_file() else archive_du_run() or workdir),
        "-o",
        str(reports_dir / "rapport_memoire.html"),
    ]
    subprocess.run(cmd_mem, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential A/B cohort orchestrator for the high-fidelity replay (Ticket 077)."
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=["899549"],
        help="Identifiers of the personas to test (default: 899549 Corinne; or 'all' for the 10)",
    )
    parser.add_argument(
        "--branch",
        choices=["both", "treated", "control"],
        default="both",
        help="Branch to run: treated (shock), control (control), or both (A/B fork)",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Identifier of the campaign (default: exp_cohort_YYYY-MM-DD_HH_MM)",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="Forces a restart from scratch, ignoring existing resume points",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prints the execution plan without launching the simulations",
    )
    parser.add_argument(
        "--evenement",
        default=CHOC_REFERENCE,
        help=f"Event played by the treated arm, by its file name without extension "
             f"(default: {CHOC_REFERENCE}). Looked up in config/evenements/ then config/chocs/. "
             f"The control arm receives none, whatever this setting.",
    )
    parser.add_argument(
        "--horizon-jours",
        type=int,
        default=HORIZON_JOURS_DEFAUT,
        help=f"Simulated calendar days (default: {HORIZON_JOURS_DEFAUT}). Written to "
             f"sim_params.yaml; the arm's guard delay depends on it.",
    )
    parser.add_argument(
        "--arret-sur-extinction",
        action="store_true",
        help="Cuts the arm once the DECLARED memory has left the \"What changed "
             "recently\" block for --jours-apres-extinction lived days. The arm is then "
             "shorter than its control: the analysis must truncate both at the day written in "
             "arret_sur_extinction.json. Without this flag, the run goes to its horizon.",
    )
    parser.add_argument(
        "--jours-apres-extinction",
        type=int,
        default=JOURS_APRES_EXTINCTION,
        help=f"LIVED days observed after extinction before cutting "
             f"(default: {JOURS_APRES_EXTINCTION})",
    )

    args = parser.parse_args()

    # Persona resolution
    if args.personas == ["all"]:
        personas = COHORTE_10_PERSONAS
    else:
        personas = args.personas

    exp_id = args.experiment_id or f"exp_cohort_{datetime.now().strftime('%Y-%m-%d_%H_%M')}"
    base_dir = Path("experiments/runs") / exp_id

    branches = ["treated", "control"] if args.branch == "both" else [args.branch]

    print("=" * 75)
    print(f"SEQUENTIAL COHORT ORCHESTRATOR — TICKET 077 REPLAY")
    print("=" * 75)
    print(f"Campaign        : {exp_id}")
    print(f"Personas ({len(personas)}) : {', '.join(personas)}")
    print(f"Branches        : {', '.join(branches)}")
    print(f"Root folder     : {base_dir}")
    print(f"Event           : {args.evenement} (treated arm only)")
    print(f"Horizon         : {args.horizon_jours} calendar days")
    _jalons = os.getenv("EXPERIMENT_SURVEY_DAYS") or "12,17,29,40 (défaut)"
    print(f"Survey days     : {_jalons} — 21:00, absolute memory isolation")
    if args.arret_sur_extinction:
        print(f"Early stop      : ARMED — {args.jours_apres_extinction} lived days after "
              f"the extinction of the declared memory (arm shorter than its control)")
    else:
        print("Early stop      : disarmed — each arm goes to its horizon")
    print("=" * 75)

    for i, pid in enumerate(personas, 1):
        for br in branches:
            print(f"\n>>> [{i}/{len(personas)}] Processing persona {pid} — Branch: {br.upper()}")
            workdir = base_dir / f"{pid}_{br}"
            is_resume = preparer_workdir(
                workdir, pid, br, force_fresh=args.force_fresh, horizon_jours=args.horizon_jours
            )

            ret = executer_run_persona(
                persona_id=pid,
                branch=br,
                workdir=workdir,
                is_resume=is_resume,
                dry_run=args.dry_run,
                arret_extinction=args.arret_sur_extinction,
                jours_apres_extinction=args.jours_apres_extinction,
                evenement=args.evenement,
                horizon_jours=args.horizon_jours,
            )

            if not args.dry_run and ret == 0:
                archive = archive_du_run()
                consigner_au_manifeste(base_dir, pid, br, archive)
                generer_analyses_post_run(archive or workdir, pid, br)
                if archive and archive.is_dir() and workdir.resolve() != archive.resolve():
                    # Everything the memory orchestrator rereads: without `evenements.jsonl`, its
                    # ticket 108 reconciliation counted zero injections produced on an arm
                    # where the article had indeed been read.
                    # `premiers_services.jsonl` (2026-09-29): the first prompt carrying the event,
                    # from which the orchestrator draws the control arm's strict bound. Missing
                    # here, it never reached `traite/`, and the bound fell back to the injection.
                    for fname in [
                        "moves.csv", "decisions_rejeu.jsonl", "identite_run.json",
                        "evenements.jsonl", "premiers_services.jsonl", "agent_memory_events.jsonl",
                        "temoin_souvenir.jsonl", "affinites_declarees.csv", "app.log",
                    ]:
                        if (archive / fname).is_file():
                            shutil.copy2(archive / fname, workdir / fname)
                    if (archive / "reports").is_dir():
                        shutil.copytree(archive / "reports", workdir / "reports", dirs_exist_ok=True)
                    logger.info(f"[{pid}_{br}] Deliverables copied from {archive.name} to {workdir}")
            elif ret != 0:
                # We STOP. Chaining to the next arm after a failure is what
                # produced on 2026-09-19 a "control" that had killed the treated arm.
                logger.error(
                    f"[ALARME] Arm {pid}_{br} failed (code {ret}) — CAMPAIGN INTERRUPTED. "
                    f"The following arms are not launched: a failing arm leaves the stack "
                    f"in a state the next one would inherit."
                )
                sys.exit(ret)

    print("\n" + "=" * 75)
    print("ALL COHORT ITERATIONS ARE FINISHED.")
    print("=" * 75)


if __name__ == "__main__":
    main()
