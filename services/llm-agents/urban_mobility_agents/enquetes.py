"""Declared modal affinity survey — tickets 077 (probe) and 095, batch B (instrument).

TWO DISTINCT REQUIREMENTS, AND THE MODULE ONLY CARRIED ONE.

**On OUTPUT, absolute sealing.** The answers go into `affinites_declarees.csv` and nowhere
else: neither STM, nor LTM, nor ChromaDB, nor night reflection gateway. The survey
contaminates no later deliberation. That is what this module already promised, and that does not
change.

**On INPUT, full fidelity.** The prompt serves EXACTLY the same memory block as the decision
prompt — `memoire_noyau(journal, entrées, maintenant, person_id)`, that is « Mes habitudes »,
« Ce que je sais » and « Ce qui a changé récemment » — plus the full identity narrative of the
persona. Until 2026-09-21, the perception served fitted in four fields: name, age, gender,
occupation. **As written, the survey measured the prior of the base model about a
53-year-old part-time woman** — identical on day 12 and on day 29, identical in the arm with window
7 and in the one with window 14. It would have detected nothing, and its silence would have been taken for
an absence of effect.

⚠ **The survey READS the memory, it does not RECALL it.** `memoire_noyau` touches neither `force` nor
`dernier_rappel`. The vector recall path, for its part, strengthens the force of each memory served
(`llm_agent.py`, `force_apres_rappel`): taking it would make the probe an extender of the
lifetime of what it observes. It does not take it.

THE INSTRUMENT (ticket 095, batch B)
------------------------------------
Six criteria of Adam & Gaudou (2025) — speed, practicality, comfort, safety, financial
accessibility, ecology — on a 0-10 Likert scale, and **five prompts per milestone**:

- four perception prompts, **one per mode**, the mode named and the three others never mentioned.
  The 24 scores asked for at once produced a flat grid;
- a PRIORITIES prompt, which names no mode. With the previous ones, it makes computable
  `score(mode) = Σ val(mode, criterion) × prio(criterion)` — the mode that the symbolic model
  of Adam & Gaudou would predict from our agent's statements, to be set against the mode
  it chooses the next day.

ECOLOGY IS THE CONTROL QUESTION. None of the six shocks declared in the repository targets it. If the
ecology score of the bike drops after a puncture, the agent has not rated a criterion: it has expressed an
overall mood, and the whole instrument is invalid. A question that must stay flat is a
guard, not an expense.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from llm.noyau import TITRE_CHANGEMENTS, memoire_noyau
from sim_clock import wall_clock
from urban_mobility_agents.utils.ancre_run import jours_ecoules
from urban_mobility_agents.utils.modeles import modele_de_instance
from urban_mobility_agents.utils.reprise import point_de_reprise_timestamp
from urban_mobility_agents.utils.routage import instances_pour

# Survey milestones, in CALENDAR days of the run (day 1 is the first simulated day).
#
# ⚠ The milestones are DECLARED (`EXPERIMENT_SURVEY_DAYS="12,17,29,40"`). The first version
# hard-coded them — 5, 9 and 19 — with the labels « pré-choc », « péri-choc », « post-choc ».
# These labels dated from the protocol where the shock fell on day 8. When the shock moved to
# days 15 and 16, day 9 became pre-shock, day 19 ended up four days from the shock, and
# no milestone fell any more after the exit from the memory window. Three wrong labels, and
# a survey paid for nothing.
#
# Default aligned on the protocol in force (shock on days 15 and 16, anchor Monday 16 March):
#   D12 — Friday 27 March: last working day of the baseline, the habit is formed
#   D17 — Wednesday 1 April: day after the second shock day
#   D29 — Monday 13 April: day on which the shock memory leaves the window of the core block
#   D40 — Friday 24 April: late, once the shock is out of reach
JOURS_JALONS_DEFAUT: tuple[int, ...] = (12, 17, 29, 40)
HEURE_ENQUETE_H = 21  # 21h00

# ── The instrument: six criteria, four modes, one priority vector ───────────
# The six criteria of Adam & Gaudou (2025). Canonical keys in FRENCH — it is the language in
# which the scale was defined and discussed, and they are what comes out in the CSV; the
# template addresses the model in English (ticket 074).
#
# ⚠ Removing SAFETY would make the instrument blind to four declared shocks out of six (C2, C3,
# C5, C6). ECOLOGY, for its part, is the CONTROL question: none of the six shocks targets it, and it
# must stay flat. If it moves, the agent has expressed an overall mood instead of rating a
# criterion, and the rest of the measurement can no longer be interpreted.
CRITERES: tuple[str, ...] = (
    "rapidite", "praticite", "confort", "securite", "cout", "ecologie"
)

# Modes questioned by default: the four of Adam & Gaudou. The train and the powered two-wheeler
# are added when the shock of the run targets them (C4, C5) — DECLARED list, never hard-coded.
MODES_DEFAUT: tuple[str, ...] = ("voiture", "transports_collectifs", "velo", "marche")

# How the mode is NAMED to the model. One prompt per mode, the others never mentioned.
LIBELLE_MODE_EN = {
    "voiture": "the car",
    "transports_collectifs": "public transport",
    "velo": "the bicycle",
    "marche": "walking",
    "train": "the train",
    "deux_roues": "the motor scooter",
}

# Value of the `mode` column for the six lines of the priority vector. The same file carries
# both vectors: a second file would force bringing them together, and the score formula
# consumes them together.
MODE_PRIORITES = "critere_priorite"

SCORE_MIN, SCORE_MAX = 0, 10

COLONNES_CSV = (
    "sim_timestamp", "date_simulee", "jour_simule", "persona_id",
    "mode", "critere", "score", "justification", "provider", "model",
)

_jalons: tuple[int, ...] | None = None
_modes: tuple[str, ...] | None = None
_jalons_verifies = False


def reinitialiser() -> None:
    """Forgets the milestones and the modes read, and the warning already given. Reserved for tests."""
    global _jalons, _jalons_verifies, _modes, _completion_en_cours, _prochaine_completion
    _jalons = None
    _modes = None
    _jalons_verifies = False
    _completion_en_cours = False
    _prochaine_completion = 0.0


def jours_jalons() -> tuple[int, ...]:
    """The milestones in force, read once then memorised."""
    global _jalons
    if _jalons is not None:
        return _jalons

    brut = (os.getenv("EXPERIMENT_SURVEY_DAYS") or "").strip()
    if not brut:
        _jalons = JOURS_JALONS_DEFAUT
        logger.info(f"[enquete] survey milestones not declared — default value {list(_jalons)}")
        return _jalons

    if brut.lower() in ("daily", "quotidien", "tous", "all"):
        _jalons = tuple(range(1, 101))
        logger.info("[enquete] survey milestones declared in daily mode (every day)")
        return _jalons

    try:
        lus = tuple(int(p.strip()) for p in brut.split(",") if p.strip())
        if not lus or any(j < 1 for j in lus):
            raise ValueError(f"jalons vides ou hors domaine : {lus}")
    except ValueError as exc:
        # A survey switched off silently cannot be told from a survey that found nothing.
        logger.error(
            f"[ALARME] [enquete] EXPERIMENT_SURVEY_DAYS illisible (« {brut} » — {exc}) : "
            f"repli sur {list(JOURS_JALONS_DEFAUT)}. Les réponses attendues aux jours déclarés "
            f"n'existeront pas."
        )
        _jalons = JOURS_JALONS_DEFAUT
        return _jalons

    _jalons = lus
    logger.info(f"[enquete] jalons déclarés : {list(_jalons)}")
    return _jalons


def verifier_jalons_atteignables(jour_courant: int, date_du_jour_courant: str) -> None:
    """Flags ONCE the milestones that fall on a weekend, hence will never trigger.

    The simulation skips Saturday and Sunday departures (`agent.no_weekend_departures`). A
    milestone set on one of these days stays mute, and its absence reads as a failure of the survey —
    that is exactly what happened to the shocks set on days 6 and 7 (ticket 077, § 10.2).
    """
    global _jalons_verifies
    if _jalons_verifies:
        return
    _jalons_verifies = True
    try:
        ancre = datetime.fromisoformat(str(date_du_jour_courant)[:10]).date() - timedelta(
            days=int(jour_courant) - 1
        )
    except (TypeError, ValueError) as exc:
        logger.warning(f"[enquete] jalons non vérifiables ({exc!r})")
        return
    brut = (os.getenv("EXPERIMENT_SURVEY_DAYS") or "").strip().lower()
    if brut in ("daily", "quotidien", "tous", "all"):
        return
    for jalon in jours_jalons():
        jour = ancre + timedelta(days=jalon - 1)
        if jour.weekday() >= 5:
            logger.warning(
                f"[enquete] milestone J{jalon} never reachable: it falls on a "
                f"{'Saturday' if jour.weekday() == 5 else 'Sunday'} ({jour.isoformat()}), "
                f"and the simulation skips weekends. This survey will never fire."
            )


def is_enquete_due(timestamp: int, enquetes_menees: set[int]) -> int | None:
    """Checks whether a declared affinity survey must be triggered.

    Returns the number of the milestone day, or None.
    """
    # Check of the activation flag (default False for the main mode)
    if os.getenv("EXPERIMENT_SURVEY_ENABLED") != "1":
        return None

    # No survey at the weekend: the agents do not travel
    if wall_clock(timestamp).weekday() >= 5:
        return None

    jour = jours_ecoules(timestamp) + 1
    verifier_jalons_atteignables(jour, wall_clock(timestamp).strftime("%Y-%m-%d"))
    if jour in jours_jalons() and jour not in enquetes_menees:
        heure = wall_clock(timestamp).hour
        if heure >= HEURE_ENQUETE_H:
            return jour
    return None


def modes_interroges() -> tuple[str, ...]:
    """The modes to question, DECLARED by `EXPERIMENT_SURVEY_MODES`, read once.

    Hard-coded, they would make the instrument blind to the shocks that target the train (C4) or the
    two-wheeler (C5): four prompts on four modes, none of which is the one the shock
    struck.
    """
    global _modes
    if _modes is not None:
        return _modes
    brut = (os.getenv("EXPERIMENT_SURVEY_MODES") or "").strip()
    if not brut:
        _modes = MODES_DEFAUT
        logger.info(f"[enquete] modes non déclarés — valeur par défaut {list(_modes)}")
        return _modes
    lus = tuple(m.strip() for m in brut.split(",") if m.strip())
    inconnus = [m for m in lus if m not in LIBELLE_MODE_EN]
    if inconnus:
        # An unknown mode has no label: the prompt would ask six questions about nothing.
        logger.error(
            f"[ALARME] [enquete] mode(s) inconnu(s) déclaré(s) : {inconnus} — ils sont écartés. "
            f"Modes connus : {sorted(LIBELLE_MODE_EN)}."
        )
        lus = tuple(m for m in lus if m in LIBELLE_MODE_EN)
    if not lus:
        logger.error(
            f"[ALARME] [enquete] EXPERIMENT_SURVEY_MODES (« {brut} ») ne laisse aucun mode "
            f"connu — repli sur {list(MODES_DEFAUT)}."
        )
        lus = MODES_DEFAUT
    _modes = lus
    logger.info(f"[enquete] modes déclarés : {list(_modes)}")
    return _modes


def perception_de(
    agent: Any, person: Any, timestamp: int, lignes: tuple[str, ...] | list[str] = ()
) -> str:
    """What the agent KNOWS about itself at the time of the survey — identity and core memory.

    This is where input fidelity is at stake. The memory block is built by the SAME call
    as the decision prompt (`memoire_noyau`), and not by a « lighter » variant that
    would diverge silently: a probe that does not see what the agent sees does not measure the agent.

    ⚠ No vector recall: `memoire_noyau` reads, it strengthens nothing. Taking the recall
    path would extend the lifetime of the memories being observed.

    A block that fails to build does not lose the survey, but it does not disappear
    silently either: without it, the answer measures the base model and not the agent, and that is
    exactly the defect this batch fixes.

    `lignes` (ticket 111, D6) — what is guaranteed to the decision prompt that day. A survey
    held on a service day sees the SAME line as the decision: without that, it would question
    an agent that does not know what it read that very morning.
    """
    pid = str(person.person_id)
    morceaux: list[str] = []
    try:
        recit = agent.get_person_identity_description(person)
        if recit:
            morceaux.append(str(recit).strip())
    except Exception as err:  # noqa: BLE001
        logger.warning(f"[enquete] récit d'identité indisponible pour {pid} ({err})")

    try:
        entrees = (
            agent.long_term_memory.user_metadata.get(pid, {}).get("entries", [])
            if getattr(agent, "long_term_memory", None) is not None
            else []
        )
        journal = (
            agent.long_term_memory.journal_trajets(pid)
            if getattr(agent, "long_term_memory", None) is not None
            else {}
        )
        bloc = memoire_noyau(journal, entrees, wall_clock(timestamp), pid, lignes)
    except Exception as err:  # noqa: BLE001
        logger.error(
            f"[ALARME] [enquete] mémoire noyau non construite pour {pid} ({err}) — la réponse "
            f"mesurerait le modèle de base et non l'agent."
        )
        bloc = [TITRE_CHANGEMENTS, *(f"- {l}" for l in lignes)] if lignes else []
    if bloc:
        morceaux.append("\n".join(bloc))
    return "\n\n".join(morceaux)


def _scores_valides(pid: str, jour: int, mode: str, scores: dict) -> dict[str, Any]:
    """The six scores, as returned. An out-of-range value is FLAGGED, not clipped.

    Clipping silently would fabricate a measurement: a 14 brought back to 10 reads as a maximal opinion
    whereas it says that the model did not respect the scale.
    """
    rendus: dict[str, Any] = {}
    for critere in CRITERES:
        brut = scores.get(critere)
        if brut is None:
            logger.error(
                f"[ALARME] [enquete] critère « {critere} » absent de la réponse | "
                f"persona={pid} jour=J{jour} mode={mode}"
            )
            continue
        try:
            valeur = int(brut)
        except (TypeError, ValueError):
            logger.error(
                f"[ALARME] [enquete] score illisible ({brut!r}) | persona={pid} jour=J{jour} "
                f"mode={mode} critere={critere}"
            )
            rendus[critere] = brut
            continue
        if not (SCORE_MIN <= valeur <= SCORE_MAX):
            logger.error(
                f"[ALARME] [enquete] score hors du domaine {SCORE_MIN}-{SCORE_MAX} ({valeur}) | "
                f"persona={pid} jour=J{jour} mode={mode} critere={critere} — écrit tel quel, "
                f"jamais raboté."
            )
        rendus[critere] = valeur
    return rendus


class ReponseAbsente(RuntimeError):
    """A question the gateway left unanswered, with the error kind it qualified."""

    def __init__(self, genre: str | None) -> None:
        super().__init__(f"empty answer ({genre or 'kind not qualified'})")
        self.genre = genre


async def _interroger(
    llm_client: Any, pid: str, perception: str, mode: str | None
) -> tuple[dict, str, str] | None:
    """One prompt, one mode (or the priorities if `mode` is None). Returns (scores, provider, model)."""
    payload = {
        "category": "enquete_affinite",
        # Ticket 095, batch C — the probe does not have to fight the measured variable for its key.
        "instances_admises": instances_pour("enquete_affinite"),
        "agents": [
            {
                "agent_id": pid,
                "perception": perception,
                "mode_interroge": LIBELLE_MODE_EN[mode] if mode else None,
            }
        ],
        "parameters": {"temperature": 0.2, "max_tokens": 512},
    }
    res = await llm_client.execute(payload)
    if not res or not res.agents:
        # The kind the gateway qualified (overload, quota…): it says whether the stop that follows
        # is retried (`utils/nature_arret.py`). None: the cause is not the provider.
        raise ReponseAbsente(getattr(res, "error_kind", None) if res else None)
    rep = res.agents[0]
    scores = getattr(rep, "scores", {}) or {}
    justification = (getattr(rep, "justification", "") or "").strip()
    provider = res.provider_used or "unknown"
    return (
        {"scores": scores, "justification": justification},
        provider,
        modele_de_instance(provider),
    )


def deja_mene_avant_reprise(jour: int, timestamp: int, csv_file: Path) -> bool:
    """A milestone already held, or pending, is never asked again (2026-09-28, widened on 2026-09-29).

    At the resume (ticket 075), the memory is restored to the LAST resume point, then GAMA
    replays from t0. Asking the day-1 survey again with the memory of day 11 would return an answer
    the agent could never have given that day: one more wrong line in the CSV, paid for, and,
    in the common prefix, a prompt unknown to the treated arm's store, which the strict replay refuses (409 —
    a13 v5 control, resume of 2026-09-28 15:23). The answers of the first pass remain the
    only source.

    The CSV is authoritative, frozen or not: a milestone is complete or absent in it (ticket 118). Until
    2026-09-29 the guard only applied STRICTLY before the resume point. Yet a stop during
    the survey writes its point at the very time of the milestone (21:00): at the thaw, the evening milestone,
    completed in the meantime, was asked a second time — 180 duplicate lines on the
    `make banc-reprise` bench of 29/09.

    Returns True when the milestone is to be skipped. The controller marks the milestone as held before the call:
    the message only goes out once per milestone.
    """
    quand = wall_clock(timestamp).strftime("%Y-%m-%d %H:%M")
    attente = lire_en_attente(csv_file.parent, jour)
    if attente:
        logger.info(
            f"[enquete] jalon J{jour} ({quand}) EN ATTENTE depuis une surcharge : "
            f"{len(attente.get('manquantes') or [])} question(s) seront reposées sur la "
            f"perception photographiée ce soir-là, pas ici."
        )
        return True
    lignes = 0
    if csv_file.is_file():
        with csv_file.open(encoding="utf-8", newline="") as f:
            lignes = sum(1 for r in csv.DictReader(f) if str(r.get("jour_simule")) == str(jour))
    if lignes:
        logger.info(
            f"[enquete] jalon J{jour} ({quand}) déjà mené ({lignes} ligne(s) dans "
            f"{csv_file.name}) — pas réinterrogé."
        )
        return True
    point = point_de_reprise_timestamp()
    if point is None or int(timestamp) >= point:
        return False
    gel = wall_clock(point).strftime("%Y-%m-%d %H:%M")
    logger.error(
        f"[ALARME] [enquete] jalon J{jour} ({quand}) absent de {csv_file.name} et impossible "
        f"à reposer : la mémoire de ce jour-là n'existe plus (rejeu gelé jusqu'au {gel}). Ce "
        f"jalon manquera à la mesure."
    )
    return True


async def executer_enquetes_jalon(
    jour: int,
    timestamp: int,
    people: list[Any],
    agent: Any,
    output_dir: Path,
    en_cas_de_saturation: Callable[[int, str, str | None], Awaitable[None]] | None = None,
) -> None:
    """The five prompts of the milestone, for each target persona.

    Saves into `affinites_declarees.csv` in LONG format — one line per (day, persona,
    mode, criterion, score) — WITHOUT EVER touching the memory of the agents. The wide format with 24
    columns forced rewriting the header as soon as a mode was added; the long format absorbs
    the train and the two-wheeler without changing schema.

    `agent` is the full LLM agent, and not only its client: it is from it that the identity
    narrative and the core memory come, without which the probe measures the base model.
    """
    csv_file = output_dir / "affinites_declarees.csv"
    if deja_mene_avant_reprise(jour, timestamp, csv_file):
        return
    llm_client = getattr(agent, "llm_client", None) or agent

    target_env = os.getenv("EXPERIMENT_TARGET_PERSONAS")
    if target_env:
        cibles_ids = {pid.strip() for pid in target_env.split(",") if pid.strip()}
        cibles = [p for p in people if str(p.person_id) in cibles_ids]
    elif len(people) == 1:
        # In sequential mode with 1 inhabitant, the single persona is the target
        cibles = people
    else:
        logger.warning(
            "[enquete] Plusieurs personas présents sans EXPERIMENT_TARGET_PERSONAS défini — "
            "sondage ignoré."
        )
        return

    modes = modes_interroges()
    attendus = len(modes) + 1
    logger.info(
        f"[enquete] jalon J{jour} — {len(cibles)} persona(s), {attendus} prompt(s) chacun "
        f"({len(modes)} mode(s) + priorités)."
    )

    # The perception is PHOTOGRAPHED here, once: it is it, and it alone, that the question
    # and all its retries receive, including those after a resume.
    from llm import evenements as _evenements

    perceptions = {
        str(p.person_id): perception_de(
            agent,
            p,
            timestamp,
            await _evenements.lignes_du_jour(str(p.person_id), timestamp, compter=False),
        )
        for p in cibles
    }
    questions = [(pid, mode) for mode in (*modes, None) for pid in perceptions]
    genres: dict[tuple, str | None] = {}
    obtenues, manquantes = await _poser_questions(
        llm_client, jour, timestamp, perceptions, questions, output_dir, genres
    )
    await _clore_jalon(
        output_dir, jour, timestamp, perceptions, obtenues, manquantes, en_cas_de_saturation,
        attendues=len(questions), genres=genres,
    )


# ── No answer lost (ticket 118, O2) ──────────────────────────────────────────────────
# On 28/09 at 21:00, the a13 v5 control lost 78 answers out of 180 at milestone D11: Gemini 3.1
# overloaded from 20:59 to 21:06, each unanswered question noted as [ALARME], and the milestone written
# incomplete. These answers could no longer be asked again. Author's rule (29/09): no
# answer lost, not a single one.
#
# 1. Each question is retried on the spot (`EXPERIMENT_SURVEY_RETRIES_S`, 1, 2 then 4 min):
#    the survey has no departure time to keep, and the overload of 28/09 lasted 7 minutes.
# 2. A milestone is COMPLETE OR IS NOT WRITTEN. If answers are still missing, the perception
#    photographed at the milestone and the answers obtained go into `enquete_en_attente.json`,
#    and the controller is notified (orderly stop, resume later).
# 3. The missing questions are then asked again ON THE SAME PERCEPTION: the answer is the one
#    the persona would have given that evening, whatever the time at which the run resumes.
#    The perception only depends on the memory at the milestone, and the survey never writes to memory.
FICHIER_EN_ATTENTE = "enquete_en_attente_J{jour:03d}.json"
RETENTATIVES_DEFAUT_S: tuple[int, ...] = (60, 120, 240)
_completion_en_cours = False
_prochaine_completion = 0.0
PAUSE_ENTRE_COMPLETIONS_S = 600


def retentatives_s() -> tuple[int, ...]:
    """Delays between two attempts of the same question (`EXPERIMENT_SURVEY_RETRIES_S`)."""
    brut = os.getenv("EXPERIMENT_SURVEY_RETRIES_S")
    if not brut:
        return RETENTATIVES_DEFAUT_S
    try:
        return tuple(max(0, int(x)) for x in brut.split(",") if x.strip())
    except ValueError:
        logger.error(
            f"[ALARME] [enquete] EXPERIMENT_SURVEY_RETRIES_S illisible (« {brut} ») : délais par "
            f"défaut {RETENTATIVES_DEFAUT_S}."
        )
        return RETENTATIVES_DEFAUT_S


# ── Simulated outage — functional bench of ticket 118 (2026-09-29) ────────────────────────────
# `EXPERIMENT_SURVEY_PANNE=<jour>:<durée_s>`: from milestone <jour> on, NO survey question
# gets an answer for <durée_s> seconds of wall-clock time — the Gemini
# overload of 28/09, where nothing moved for hours. No call is made during the
# outage. Its start is written in the run folder (`banc_panne_enquete.json`): it survives
# stops and resumes, as a real outage survives a restart of the controller.
# Never in a measurement: a run that carries it says so at each refused question.
FICHIER_PANNE = "banc_panne_enquete.json"


def _panne_simulee(output_dir: Path | None, jour: int) -> bool:
    """Does the question of this milestone fall within the simulated outage of the bench?"""
    brut = (os.getenv("EXPERIMENT_SURVEY_PANNE") or "").strip()
    if not brut or output_dir is None:
        return False
    try:
        jour_panne, duree_s = (int(x) for x in brut.split(":"))
    except ValueError:
        logger.error(
            f"[ALARME] [enquete] EXPERIMENT_SURVEY_PANNE illisible (« {brut} », attendu "
            f"<jour>:<durée_s>) : aucune panne simulée."
        )
        return False
    if int(jour) < jour_panne:
        return False
    chemin = Path(output_dir) / FICHIER_PANNE
    try:
        etat = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        etat = {}
    if "debut" not in etat:
        etat = {"jour": jour_panne, "duree_s": duree_s, "debut": time.time()}
        chemin.write_text(json.dumps(etat), encoding="utf-8")
        fin = datetime.fromtimestamp(etat["debut"] + duree_s).astimezone()
        logger.warning(
            f"[banc] PANNE SIMULÉE des enquêtes à partir du jalon J{jour} : aucune réponse pendant "
            f"{duree_s} s, jusqu'à {fin:%H:%M:%S}. Ce run est un banc, pas une mesure."
        )
    return time.time() < float(etat["debut"]) + float(etat["duree_s"])


def _ordre(question: tuple[str, str | None]) -> tuple:
    """Persona, then mode in the order of the questions, priorities last: the order of the CSV."""
    pid, mode = question
    modes = modes_interroges()
    return (pid, modes.index(mode) if mode in modes else len(modes))


async def _poser_une(
    llm_client: Any,
    jour: int,
    timestamp: int,
    pid: str,
    perception: str,
    mode: str | None,
    output_dir: Path | None = None,
    genres: dict[tuple, str | None] | None = None,
) -> list[dict[str, Any]] | None:
    """One question and its retries. Returns its CSV lines, or None if it stays unanswered.

    Unanswered, `genres[(pid, mode)]` receives the error kind of the last attempt (2026-09-29):
    an overload is retried, an unreadable answer or an exception of the code is not.

    An answer is only kept if the six criteria are in it: a missing line is an
    answer lost just like a failed call. An out-of-range value, for its part, is written
    as is and flagged by `_scores_valides` (never clipped).
    """
    etiquette = mode or MODE_PRIORITES
    delais = retentatives_s()
    horodatage = wall_clock(timestamp).strftime("%Y-%m-%d %H:%M")
    genre: str | None = None
    for essai in range(len(delais) + 1):
        genre = None
        if _panne_simulee(output_dir, jour):
            res, cause, genre = None, "panne simulée (banc, EXPERIMENT_SURVEY_PANNE)", "panne_simulee"
        else:
            try:
                res = await _interroger(llm_client, pid, perception, mode)
                cause = None
            except ReponseAbsente as err:
                res, cause, genre = None, str(err), err.genre
            except Exception as err:  # noqa: BLE001
                res, cause = None, f"appel en échec : {err}"
        if res is not None:
            contenu, provider, modele = res
            scores = _scores_valides(pid, jour, etiquette, contenu["scores"])
            if len(scores) == len(CRITERES):
                logger.info(
                    f"[enquete] J{jour} {pid} · {etiquette} : "
                    + ", ".join(f"{c}={scores.get(c)}" for c in CRITERES)
                    + f" | {provider}/{modele} | essai {essai + 1} | étanchéité mémoire : RESPECTÉE."
                )
                return [
                    {
                        "sim_timestamp": timestamp,
                        "date_simulee": horodatage,
                        "jour_simule": jour,
                        "persona_id": pid,
                        "mode": etiquette,
                        "critere": critere,
                        "score": scores[critere],
                        "justification": contenu["justification"],
                        "provider": provider,
                        "model": modele,
                    }
                    for critere in CRITERES
                ]
            cause = f"{len(CRITERES) - len(scores)} critère(s) absent(s)"
        if essai < len(delais):
            logger.warning(
                f"[enquete] J{jour} {pid} · {etiquette} : {cause} — essai {essai + 2}/"
                f"{len(delais) + 1} dans {delais[essai]} s."
            )
            await asyncio.sleep(delais[essai])
        else:
            logger.error(
                f"[ALARME] [enquete] J{jour} {pid} · {etiquette} : {cause} après "
                f"{len(delais) + 1} essai(s) — la question sera reposée plus tard, sur la même "
                f"perception."
            )
    if genres is not None:
        genres[(pid, mode)] = genre
    return None


async def _poser_questions(
    llm_client: Any,
    jour: int,
    timestamp: int,
    perceptions: dict[str, str],
    questions: list[tuple[str, str | None]],
    output_dir: Path | None = None,
    genres: dict[tuple, str | None] | None = None,
) -> tuple[dict[tuple, list[dict]], list[tuple[str, str | None]]]:
    """Asks the questions, the personas TOGETHER and the modes one after the other.

    2026-09-24 — the gateway's micro-batching merges a wave into a single multi-agent prompt;
    the modes stay sequential, so that a batch only gathers questions about the same mode.
    """
    obtenues: dict[tuple, list[dict]] = {}
    manquantes: list[tuple[str, str | None]] = []
    modes_vus: list[str | None] = []
    for _pid, mode in questions:
        if mode not in modes_vus:
            modes_vus.append(mode)
    for mode in modes_vus:
        vague = [(pid, m) for pid, m in questions if m == mode]
        debut_mode = time.monotonic()
        reponses = await asyncio.gather(
            *(
                _poser_une(llm_client, jour, timestamp, pid, perceptions[pid], m, output_dir, genres)
                for pid, m in vague
            )
        )
        for question, lignes in zip(vague, reponses):
            if lignes:
                obtenues[question] = lignes
            else:
                manquantes.append(question)
        logger.info(
            f"[enquete] J{jour} · {mode or MODE_PRIORITES} : {len(vague)} persona(s) interrogé(s) "
            f"ensemble en {time.monotonic() - debut_mode:.1f}s."
        )
    return obtenues, manquantes


def _chemin_attente(output_dir: Path, jour: int) -> Path:
    return Path(output_dir) / FICHIER_EN_ATTENTE.format(jour=int(jour))


def lire_en_attente(output_dir: Path, jour: int | None = None) -> dict | None:
    """The pending milestone (that of `jour`, or the oldest one), or None.

    One file per milestone: two pending milestones do not overwrite each other. An unreadable file
    raises an alarm and stays in place for examination.
    """
    if jour is not None:
        chemin = _chemin_attente(output_dir, jour)
        if not chemin.is_file():
            return None
    else:
        candidats = sorted(Path(output_dir).glob("enquete_en_attente_J*.json"))
        if not candidats:
            return None
        chemin = candidats[0]
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        logger.error(
            f"[ALARME] [enquete] {chemin} illisible ({err}) — le jalon en attente ne peut pas "
            f"être complété ; le fichier est laissé en place pour examen."
        )
        return None


def _ecrire_lignes(output_dir: Path, lignes: list[dict]) -> None:
    csv_file = Path(output_dir) / "affinites_declarees.csv"
    existe = csv_file.is_file()
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLONNES_CSV))
        if not existe:
            writer.writeheader()
        writer.writerows(lignes)
    logger.info(f"[enquete] {len(lignes)} ligne(s) enregistrée(s) dans {csv_file}")


async def _clore_jalon(
    output_dir: Path,
    jour: int,
    timestamp: int,
    perceptions: dict[str, str],
    obtenues: dict[tuple, list[dict]],
    manquantes: list[tuple[str, str | None]],
    en_cas_de_saturation: Callable[[int, str, str | None], Awaitable[None]] | None,
    *,
    attendues: int,
    reprises: int = 0,
    genres: dict[tuple, str | None] | None = None,
) -> bool:
    """Writes the milestone if it is complete, otherwise puts it on hold. Returns True if it is written."""
    if not manquantes:
        lignes = [l for q in sorted(obtenues, key=_ordre) for l in obtenues[q]]
        _ecrire_lignes(output_dir, lignes)
        chemin = _chemin_attente(output_dir, jour)
        if chemin.is_file():
            chemin.unlink()
        logger.info(
            f"[enquete] jalon J{jour} COMPLET : {len(obtenues)}/{attendues} question(s) "
            f"répondue(s), {len(lignes)} ligne(s)"
            + (f", dont {reprises} reposée(s) après une surcharge." if reprises else ".")
        )
        return True
    attente = {
        "jour": jour,
        "timestamp": timestamp,
        "perceptions": perceptions,
        "manquantes": [list(q) for q in sorted(manquantes, key=_ordre)],
        "lignes": [l for q in sorted(obtenues, key=_ordre) for l in obtenues[q]],
        "attendues": attendues,
        "mis_en_attente_le": datetime.now().astimezone().isoformat(),
    }
    chemin = _chemin_attente(output_dir, jour)
    chemin.write_text(json.dumps(attente, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.error(
        f"[ALARME] [enquete] jalon J{jour} INCOMPLET : {len(manquantes)}/{attendues} question(s) "
        f"sans réponse ({', '.join(f'{p}·{m or MODE_PRIORITES}' for p, m in manquantes[:6])}"
        f"{'…' if len(manquantes) > 6 else ''}). RIEN n'est écrit dans le CSV : le jalon attend "
        f"dans {chemin.name}, et ses questions seront reposées sur la même perception."
    )
    if en_cas_de_saturation is not None:
        await en_cas_de_saturation(jour, manquantes[0][0], genre_du_jalon(manquantes, genres))
    return False


def genre_du_jalon(
    manquantes: list[tuple[str, str | None]], genres: dict[tuple, str | None] | None
) -> str | None:
    """The kind that qualifies the stop: a single question missed outside an overload makes it a defect.

    If every missed question was missed through an overload, it is the overload; otherwise, the
    kind of the first one that was not (None included: cause not qualified).
    """
    from urban_mobility_agents.utils.nature_arret import GENRES_QUOTA, GENRES_SURCHARGE

    vus = [(genres or {}).get(q) for q in manquantes]
    if not vus:
        return None
    hors_surcharge = [g for g in vus if g not in GENRES_SURCHARGE and g not in GENRES_QUOTA]
    if hors_surcharge:
        return hors_surcharge[0]
    return next((g for g in vus if g in GENRES_QUOTA), vus[0])


async def completer_en_attente(
    agent: Any,
    output_dir: Path,
    en_cas_de_saturation: Callable[[int, str, str | None], Awaitable[None]] | None = None,
) -> bool | None:
    """Asks again the questions of a pending milestone. Returns None if there is none, otherwise whether it is written.

    Called by the controller at each synchronisation: only one at a time, and at most one attempt
    every `PAUSE_ENTRE_COMPLETIONS_S` after a failure, so as not to hammer a saturated
    provider when the orderly stop is not armed (run outside an experiment).
    """
    global _completion_en_cours, _prochaine_completion
    attente = lire_en_attente(output_dir)
    if not attente or _completion_en_cours or time.monotonic() < _prochaine_completion:
        return None
    _completion_en_cours = True
    try:
        llm_client = getattr(agent, "llm_client", None) or agent
        jour, timestamp = int(attente["jour"]), int(attente["timestamp"])
        perceptions = dict(attente["perceptions"])
        manquantes = [(str(p), m) for p, m in attente["manquantes"]]
        obtenues: dict[tuple, list[dict]] = {}
        for ligne in attente.get("lignes") or []:
            mode = None if ligne["mode"] == MODE_PRIORITES else ligne["mode"]
            obtenues.setdefault((str(ligne["persona_id"]), mode), []).append(ligne)
        logger.info(
            f"[enquete] jalon J{jour} en attente : {len(manquantes)} question(s) reposée(s) sur la "
            f"perception du {wall_clock(timestamp).strftime('%Y-%m-%d %H:%M')}."
        )
        genres: dict[tuple, str | None] = {}
        nouvelles, encore = await _poser_questions(
            llm_client, jour, timestamp, perceptions, manquantes, output_dir, genres
        )
        obtenues.update(nouvelles)
        ecrit = await _clore_jalon(
            output_dir, jour, timestamp, perceptions, obtenues, encore, en_cas_de_saturation,
            attendues=int(attente.get("attendues") or len(obtenues) + len(encore)),
            reprises=len(nouvelles), genres=genres,
        )
        if not ecrit:
            _prochaine_completion = time.monotonic() + PAUSE_ENTRE_COMPLETIONS_S
        return ecrit
    finally:
        _completion_en_cours = False
