"""Ticket 074, lot D — a CAMPAIGN: a named batch of experiments, chained one by one.

This module **assembles**, it reinvents nothing. Everything that runs an experiment already
exists: `lancer` reserves its keys or queues itself, `reservations` holds the FIFO queue,
`ordonnanceur.tour()` reconciles the ghosts and promotes. What was missing is what happens
ABOVE: experiments to carry out in an order that makes sense, without a human being
there to launch the next one at three in the morning.

TWO MODES (2026-09-29)
----------------------

**Chaining — the default.** The experiments start one by one, in file order.
An experiment stopped on an exhausted quota is NOT relaunched: it is marked "not finished"
and the campaign moves to the next one. Keys held by another experiment leave it
"not played", and its request leaves the queue so that nothing starts it later outside the
campaign. Only a launch that really fails (crash, new resumable state) is relaunched,
`TENTATIVES_MAX` times at most. No sleep, no deferral, no retry pass: at the end of the list the
campaign ends, says what is left, and hands back control — the author wants to test afterwards.
Nor does the campaign start the others' queue: it only releases the keys of
dead runs. It was the campaign itself that, on 29/09 at 01:38, had relaunched from scratch a
stale request (go123, already finished on 27/09) and deprived its own experiments of their keys.

**Across quotas — on request** (`a_travers_les_quotas: true` in the file). The
historical behaviour, described below: batch sleep, deferral, retry pass, promotion of
the queue.

WHAT THE CAMPAIGN BRINGS, AND NOTHING MORE
------------------------------------------

1. **Phases.** A phase only starts if the previous one is closed. It is the only
   dependency the ticket needs, and it serves a precise purpose: the fourteen
   deterministic controls consume no quota, so they go first. If the
   substrate is broken, we learn it for free, before spending a single LLM call.

2. **A sleep at BATCH level.** `--attendre-fenetre` makes ONE run sleep until its
   window. But when everything in flight is asleep, the campaign has nothing to do either:
   it SAYS so, notes the wake-up time, and resumes at the current experiment — never
   at the start. That is the difference between "the machine is stuck" and "the campaign awaits
   07:00 UTC, 4 h 12 left".

3. **A state that survives a restart.** `campagnes/<nom>/etat.json` is rewritten at every
   transition, atomically. Relaunching the campaign after a `reboot` resumes where it was.

WHERE IT RUNS. On the **host**, like the scheduler and for the same reason: the launch goes
through `docker compose exec`, which makes no sense from inside the container. The campaign
therefore calls `ordonnanceur.tour()` at every round and becomes self-sufficient — no need to
run the scheduler alongside, even if doing so does no harm (the round is idempotent).

WHAT IT DOES NOT DO. It does not judge a result: a `terminee` experiment is done,
full stop. The score, compliance and comparability have their own tools (`score`,
`registre.comparer`, rule P6), and mixing them with scheduling would make both unreadable.

    make campagne-lancer NOM=bascule_anglaise_v6
    make campagne-etat   NOM=bascule_anglaise_v6
    make campagne-arreter NOM=bascule_anglaise_v6
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from experiences import refus as REFUS
from experiences.archive import (
    ETAT_ARRETEE,
    ETAT_EN_ATTENTE_QUOTA,
    ETAT_EN_PAUSE,
    ETAT_EPUISEE,
    ETAT_INTERROMPUE,
    ETAT_TERMINEE,
    F_ETAT,
    F_EXECUTION,
)
from experiences.chemins import racine_depot
from experiences.experience import dossier_experience, dossier_experiences, dossier_lancements
from loguru import logger

VERSION_CAMPAGNE = "campagne1"

#: Interval between two control rounds. Five seconds would be noise: an experiment
#: lasts minutes to hours, and each round reads a handful of files.
INTERVALLE_S = 30.0

#: Margin after the announced reopening time. Waking up to the second lands back on a
#: 429 and goes back to sleep at once — for nothing, having consumed a call.
MARGE_REVEIL_S = 60

#: Beyond this, it is no longer a daily quota holding us back (a window lasts 24 h).
#: Rising edge: the alarm fires once per sleep, not at every round.
SOMMEIL_SUSPECT_S = 26 * 3600

#: Time given to a freshly launched run to write its first `etat.json`.
#: Beyond this, the launch failed before opening its folder — we say so and we
#: relaunch. Two minutes: `lancer` replays the checks and loads the population.
DELAI_ECRITURE_ETAT_S = 120.0

#: Number of consecutive failures on the SAME experiment before declaring it failed and
#: moving to the next one. Two, because a single failure is often a service restarting.
TENTATIVES_MAX = 2

# Number of RETRY passes at the end of the campaign. A pass takes back everything that was
# deferred (quota exhausted while something else could run) and everything that failed, with
# attempt counters reset to zero. Two passes suffice: the first catches the quota
# of a provider renewed in the meantime, the second covers a second exhaustion. Beyond that, it
# is no longer a quota, it is a failure — and a campaign looping endlessly goes unnoticed.
REPECHAGES_MAX = 2

#: States from which an experiment can be resumed as is.
ETATS_REPRENABLES = (ETAT_EPUISEE, ETAT_ARRETEE, ETAT_INTERROMPUE)

#: Prefix of the interruption `raison` set by the inactivity watchdog
#: (`runner`, `inactivite:<N>s`). It is the STRUCTURED trace of the suffered pause: the message
#: in `etat.json` is written for humans and may be reworded without notice.
RAISON_PAUSE_SUBIE = "inactivite:"

#: Beyond this, an experiment the campaign believes "in flight" has shown no sign of life
#: for too long for it to be a slow computation: its state has not moved by a single
#: byte. Rising edge, one alarm per experiment — the campaign kills nothing, it SAYS so.
#: Thirty minutes: a live run rewrites its `etat.json` at every archiving, and the
#: runner's watchdog pauses after 420 s without progress.
EN_VOL_FIGE_S = 1800.0

F_STOP = "STOP"
F_ETAT_CAMPAGNE = "etat.json"

#: What must be found in a pid's command line to recognise it as a
#: running campaign, and not as a pid recycled by the system.
SIGNATURE_LANCEMENT = "campagne-lancer"

#: Exit code of a launch refused because the campaign is already running. Distinct from 1
#: (experiments failed) and from 130 (stop requested): here nothing was attempted.
CODE_DEJA_EN_VOL = 3


class CampagneInvalide(ValueError):
    """Campaign file missing, unreadable, or describing something impossible."""


# ── Model ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Phase:
    nom: str
    experiences: tuple[str, ...]
    raison: str = ""


@dataclass(frozen=True)
class Campagne:
    nom: str
    phases: tuple[Phase, ...]
    note: str = ""
    substrat: dict = field(default_factory=dict)
    #: experiment → dedicated launch command, when `experiences lancer` does not fit.
    #: The known case is the random forest control: rule R7 of ticket 044 keeps it OUT of
    #: `decideur_modele.FAMILLES` so that it does not become an arbiter, and its dedicated
    #: launcher registers its family for the lifetime of its own process. Without this way
    #: out, the campaign relaunched it endlessly on a "Format d'artefact inattendu".
    lanceurs: dict = field(default_factory=dict)
    #: False by default: the campaign chains and ends (see the module header). True:
    #: it sleeps on quotas, defers, retries and promotes the queue, as before 29/09.
    a_travers_les_quotas: bool = False

    @property
    def toutes(self) -> tuple[str, ...]:
        return tuple(e for p in self.phases for e in p.experiences)

    def phase_de(self, exp: str) -> str | None:
        for p in self.phases:
            if exp in p.experiences:
                return p.nom
        return None


def dossier_campagnes() -> Path:
    return Path(os.getenv("CAMPAGNES_DIR") or (racine_depot() / "campagnes"))


def chemin_definition(nom: str) -> Path:
    return dossier_campagnes() / f"{nom}.yaml"


def dossier_etat(nom: str) -> Path:
    return dossier_campagnes() / nom


def charger(nom: str) -> Campagne:
    """Reads `campagnes/<nom>.yaml` and VALIDATES it.

    Refuses rather than corrects: a campaign that names a non-existent experiment
    would stop halfway, after spending the quota of the previous ones.
    """
    chemin = chemin_definition(nom)
    if not chemin.is_file():
        connues = (
            sorted(p.stem for p in dossier_campagnes().glob("*.yaml"))
            if dossier_campagnes().is_dir()
            else []
        )
        raise CampagneInvalide(
            f"campaign not found: {chemin}. "
            + (
                f"Known: {', '.join(connues)}."
                if connues
                else "No campaign defined in campagnes/."
            )
        )
    try:
        doc = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise CampagneInvalide(f"{chemin.name} unreadable: {e}") from e
    if doc.get("version") != VERSION_CAMPAGNE:
        raise CampagneInvalide(
            f"{chemin.name} at version {doc.get('version')!r}, expected {VERSION_CAMPAGNE!r}."
        )

    phases: list[Phase] = []
    vues: set[str] = set()
    for i, brut in enumerate(doc.get("phases") or []):
        exps = tuple(str(e) for e in (brut.get("experiences") or []))
        if not exps:
            raise CampagneInvalide(
                f"{chemin.name}: phase {brut.get('nom') or i} carries no experiment. "
                "An empty phase would block the next one without saying anything."
            )
        doublons = sorted(set(exps) & vues)
        if doublons:
            raise CampagneInvalide(
                f"{chemin.name}: {doublons} appear in two phases. An experiment "
                "run twice in the same campaign would overwrite its own execution."
            )
        vues |= set(exps)
        phases.append(
            Phase(
                nom=str(brut.get("nom") or f"phase{i + 1}"),
                experiences=exps,
                raison=str(brut.get("raison") or ""),
            )
        )
    if not phases:
        raise CampagneInvalide(f"{chemin.name} carries no phase.")

    a_travers = doc.get("a_travers_les_quotas", False)
    if not isinstance(a_travers, bool):
        raise CampagneInvalide(
            f"{chemin.name}: `a_travers_les_quotas` is {a_travers!r}, expected true or false.")
    campagne = Campagne(
        nom=str(doc.get("nom") or nom),
        phases=tuple(phases),
        note=str(doc.get("note") or ""),
        substrat=doc.get("substrat") or {},
        lanceurs={str(k): list(v) for k, v in (doc.get("lanceurs") or {}).items()},
        a_travers_les_quotas=a_travers,
    )
    hors_campagne = sorted(set(campagne.lanceurs) - set(campagne.toutes))
    if hors_campagne:
        raise CampagneInvalide(
            f"{chemin.name}: `lanceurs` names {hors_campagne}, which are in no "
            "phase. A launcher for an absent experiment will never be used and hides a "
            "typo.")
    from experiences.experience import trouver_dossier_experience

    manquantes = [
        e
        for e in campagne.toutes
        if not ((dossier_experiences() / e / "experience.yaml").is_file() or (trouver_dossier_experience(e) and (trouver_dossier_experience(e) / "experience.yaml").is_file()))
    ]
    if manquantes:
        raise CampagneInvalide(
            f"{chemin.name} names {len(manquantes)} experiment(s) that do not exist in "
            f"{dossier_experiences()}: {manquantes[:5]}"
            + (" …" if len(manquantes) > 5 else "")
            + "\nDefine them first (`make experience-definir`): a campaign that "
            "stops halfway has already spent the quota of what comes before."
        )
    return campagne


# ── Reading progress, on disk ────────────────────────────────────────────────


def _iso(moment: datetime | None = None) -> str:
    return (moment or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def derniere_execution(exp: str) -> Path | None:
    """Folder of an experiment's latest run, or None.

    Run folders are timestamped (`2026-09-14_22_43_19`): lexical order IS
    chronological order, and staying so is a property of the format, not luck.
    """
    executions = dossier_experience(exp) / "executions"
    if not executions.is_dir():
        return None
    candidats = sorted(p for p in executions.iterdir() if p.is_dir())
    return candidats[-1] if candidats else None


def etat_experience(exp: str) -> dict:
    """State of the latest run: `{etat, raison, depuis, dossier}`.

    `etat` is `"definie"` when the experiment has never run — it is a real state of the
    platform's vocabulary, not a fallback invented here.
    """
    dossier = derniere_execution(exp)
    if dossier is None:
        return {"etat": "definie", "raison": None, "depuis": None, "dossier": None}
    fichier = dossier / F_ETAT
    try:
        brut = json.loads(fichier.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A run that has just been born has not written its state yet. It is not an
        # anomaly: we call it "en_cours" rather than count it as done or failed.
        return {
            "etat": "en_cours",
            "raison": "état non encore écrit",
            "depuis": None,
            "dossier": str(dossier),
        }
    return {
        "etat": brut.get("etat", "?"),
        "raison": brut.get("raison"),
        "depuis": brut.get("maj"),
        "dossier": str(dossier),
    }


def _fige_depuis(infos: dict) -> float | None:
    """Seconds elapsed since the state was last written, or None if undatable.

    A live run rewrites its `etat.json` as it progresses. A state that no longer moves
    is therefore a run that no longer progresses — without having to test a pid's life,
    which the campaign cannot do from the host: the runs' pids belong to the
    container's namespace.
    """
    depuis = infos.get("depuis")
    if not depuis:
        return None
    try:
        quand = datetime.fromisoformat(str(depuis))
    except ValueError:
        return None
    if quand.tzinfo is None:
        quand = quand.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - quand).total_seconds()


# ── Persistent campaign state ────────────────────────────────────────────────


def etat_par_defaut(campagne: Campagne) -> dict:
    return {
        "campagne": campagne.nom,
        "version": VERSION_CAMPAGNE,
        "demarree_le": _iso(),
        "maj": _iso(),
        "phase_courante": campagne.phases[0].nom,
        "courante": None,
        "faites": [],
        "restantes": list(campagne.toutes),
        "echouees": {},
        # Deferred: set aside because THEIR quota was exhausted while other
        # experiments could still run. It is not a failure — they come back at the
        # retry pass. Sleeping 24 h in front of a full queue was the real flaw.
        "reportees": {},
        "repechages": 0,
        "sommeils": [],
        "pid": os.getpid(),
        "terminee_le": None,
    }


def lire_etat(nom: str) -> dict | None:
    chemin = dossier_etat(nom) / F_ETAT_CAMPAGNE
    if not chemin.is_file():
        return None
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[campagne] unreadable state ({chemin}): {e} — starting from scratch")
        return None


def ecrire_etat(nom: str, etat: dict) -> None:
    """ATOMIC write. An `etat.json` truncated by a power cut would restart
    the campaign from the beginning, that is, spend again all the quota already consumed."""
    base = dossier_etat(nom)
    base.mkdir(parents=True, exist_ok=True)
    etat["maj"] = _iso()
    temp = base / f".{F_ETAT_CAMPAGNE}.tmp"
    temp.write_text(json.dumps(etat, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, base / F_ETAT_CAMPAGNE)


def demande_arret(nom: str) -> bool:
    return (dossier_etat(nom) / F_STOP).exists()


def arreter(nom: str) -> bool:
    """Sets the stop flag. The run in flight finishes; no other one is launched."""
    base = dossier_etat(nom)
    base.mkdir(parents=True, exist_ok=True)
    (base / F_STOP).write_text(_iso() + "\n", encoding="utf-8")
    logger.info(f"[campagne] {nom}: stop requested — the run in progress finishes")
    return True


def _lever_arret(nom: str) -> None:
    (dossier_etat(nom) / F_STOP).unlink(missing_ok=True)


# ── A single launch at a time ────────────────────────────────────────────────


def _ligne_de_commande(pid: int) -> str | None:
    """The command line of process `pid`, or None — either it no longer exists, or
    we could not ask for it. Both cases return None ON PURPOSE: the safeguard
    below fails OPEN. A doubt about a pid's state must never prevent a
    campaign from starting; there is only one thing worse than two campaigns in parallel,
    and it is zero campaigns because a `ps` hiccuped."""
    try:
        vu = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(
            f"[campagne] pid {pid} cannot be queried ({e}) — the double-launch "
            "safeguard lets it through; check yourself that no campaign is already running.")
        return None
    return vu.stdout.strip() or None


def _est_lancement_de(ligne: str, nom: str) -> bool:
    """Is this command line a `campagne-lancer` of campaign `nom`?

    The name is compared on the TOKEN that follows `--nom`, not as a substring: a campaign
    named `c` would recognise itself in almost anything, starting with the names of
    all the others.
    """
    if SIGNATURE_LANCEMENT not in ligne:
        return False
    jetons = ligne.replace("--nom=", "--nom ").split()
    return any(a == "--nom" and b == nom for a, b in zip(jetons, jetons[1:]))


def campagne_en_vol(nom: str) -> int | None:
    """The pid of a `campagne-lancer {nom}` ALREADY running on this machine, otherwise None.

    Why: two processes launched on the same campaign write the SAME `etat.json`, in
    turn, without seeing each other. On 2026-09-16, two launches twenty minutes apart
    fought over the same experiments for half a day, marked two of them
    "arrêt demandé — 0/3299 archivées", then declared the campaign FINISHED while one
    arm had never run. Nothing in the log said so.

    The pid alone is not enough to conclude: the system recycles them. We confirm on the command
    line, which must carry `campagne-lancer` AND the campaign name.
    """
    etat = lire_etat(nom)
    if not etat or etat.get("terminee_le"):
        return None
    pid = etat.get("pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return None
    ligne = _ligne_de_commande(pid)
    if not ligne or not _est_lancement_de(ligne, nom):
        return None
    return pid


# ── Quota window ─────────────────────────────────────────────────────────────


def prochain_reveil(maintenant: datetime | None = None) -> tuple[str, int]:
    """Next quota reopening, in ISO UTC, and the number of seconds until then.

    We take the MINIMUM over the time zones involved: sleeping until the latest would lose
    the hours during which another provider has already reopened. The Pacific default
    is that of the Gemini free tier (`ressources.FUSEAU_QUOTA_DEFAUT`); UTC is added because
    it is the gateway's fallback when a provider declares no time zone.
    """
    from experiences.ressources import FUSEAU_QUOTA_DEFAUT
    from llm_gateway.core.quota import next_quota_reset

    now = maintenant or datetime.now(timezone.utc)
    candidats = [next_quota_reset(f, now) for f in (FUSEAU_QUOTA_DEFAUT, None)]
    tot = min(candidats)
    return tot.isoformat(timespec="seconds"), max(0, int((tot - now).total_seconds()))


# ── The control loop ─────────────────────────────────────────────────────────


def journal_lancement(exp: str) -> Path:
    """Where the output of a detached launch goes.

    It went to `/dev/null`, and that is how the first launch of each
    experiment failed WITHOUT A WORD on 2026-09-15: `lancer --reprendre` refuses when there is
    nothing to resume, it said so on its standard output, and nobody read it. A
    detached launch whose output is thrown away is a launch whose fate is unknown.
    """
    base = dossier_lancements(exp, creer=True)
    return base / f"{datetime.now(timezone.utc):%Y-%m-%d_%H_%M_%S}.log"


def _lancer_experience(
    exp: str, lanceurs: dict | None = None, attendre_fenetre: bool = False
) -> None:
    """Starts an experiment, detached, by the same path as the scheduler.

    The launch is not reimplemented: `lancer` knows how to reserve its keys or queue
    itself. Waiting for the quota window is stated EXPLICITLY in both directions: the
    parser's default is "do not wait", whereas its doc announced the opposite until
    2026-09-29. Depending on no default means no longer being fooled by it.

    `--reprendre` IS ONLY PASSED IF THERE IS SOMETHING TO RESUME. The CLI refuses the flag on
    an experiment that has never run ("rien à reprendre : aucune exécution pour cette
    expérience"), and passing it by default made the FIRST launch of each of the
    twenty fail — hence the whole campaign, on its very first action.
    """
    from experiences import ordonnanceur as O

    dedie = (lanceurs or {}).get(exp)
    if dedie:
        argv = [str(m).replace("{exp}", exp) for m in dedie]
        logger.info(f"[campagne] DEDICATED launch of {exp}: {' '.join(argv)}")
    else:
        reprendre = derniere_execution(exp) is not None
        argv = O._argv_lancer(
            {"exp": exp, "args": {"reprendre": reprendre, "attendre_fenetre": attendre_fenetre}})
    sortie = journal_lancement(exp)
    racine = dossier_experiences()
    lieu = sortie.relative_to(racine) if sortie.is_relative_to(racine) else sortie
    if not dedie:
        logger.info(f"[campagne] lancement de {exp}"
                    f"{' (reprise)' if reprendre else ' (première exécution)'} — "
                    f"sortie dans {lieu}")
    with sortie.open("w", encoding="utf-8") as flux:
        subprocess.Popen(
            argv, cwd=str(racine_depot()), stdout=flux, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"},
        )


def _tour_ordonnanceur() -> None:
    """Reconciles the ghosts and promotes the queue.

    The campaign calls the round itself to be self-sufficient: without it, an
    experiment queued by a key conflict would stay there until a human
    runs `make experience-ordonnancer`. The round is idempotent, so running
    the scheduler alongside does no harm.
    """
    from experiences import ordonnanceur as O

    try:
        O.tour()
    except Exception as e:  # noqa: BLE001 — a failed round must not kill the campaign
        logger.warning(f"[campagne] scheduler round failed: {e}")


def _reconcilier_cles() -> None:
    """Releases the keys of finished or dead runs, WITHOUT promoting the queue.

    Chaining mode starts nothing but its own experiments: promoting the
    queue means launching what someone else left there — on 29/09, a two-day-old stale
    request, relaunched from scratch, held the Google keys for two hours.
    """
    from experiences import ordonnanceur as O

    try:
        O._reconcilier_conteneur()
    except Exception as e:  # noqa: BLE001 — a failed reconciliation must not kill the campaign
        logger.warning(f"[campagne] key reconciliation failed: {e}")


def _instantane(exp: str) -> tuple:
    """What distinguishes a NEW state from the previous one: folder, write date, state.

    We compare with the snapshot taken at launch rather than with a time: the two clocks (host
    and container) do not have to agree for a rewritten state to show.
    """
    infos = etat_experience(exp)
    return (infos["dossier"], infos["depuis"], infos["etat"])


def _demande_en_file(exp: str) -> dict | None:
    """The request of `exp` in the key queue, or None. Fails open: an unreadable queue
    means "not queued", and the grace period will eventually decide."""
    from experiences import reservations as R

    try:
        return next((e for e in R.lister_file() if e.get("exp") == exp), None)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[campagne] key queue unreadable ({e}) — {exp} assumed not queued")
        return None


def _retirer_demande(exp: str) -> bool:
    from experiences import reservations as R

    try:
        return R.retirer_file(exp)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[campagne] cannot remove {exp} from the queue ({e})")
        return False


def _detenteurs(cles: list[str] | None = None) -> str:
    """`clé(s) google_key1←exp_x, google_key2←exp_x tenue(s)` — to say WHO is blocking."""
    from experiences import reservations as R

    try:
        actives = R.actives()
    except Exception as e:  # noqa: BLE001
        return f"clés tenues par une autre exécution (registre illisible : {e})"
    tenues = sorted(
        f"{c}←{a.get('exp')}"
        for a in actives
        for c in a.get("cles") or []
        if not cles or c in cles
    )
    if not tenues:
        return "clés tenues par une autre exécution (détenteur déjà parti)"
    return f"clé(s) {', '.join(tenues)} tenue(s)"


def _signaler_cles_au_demarrage(campagne: Campagne) -> None:
    """At startup, say who already holds keys and what is waiting in the queue.

    On the night of 29/09, nothing said so: the keys were taken before the first launch,
    and the campaign found out at its own expense, experiment by experiment.
    """
    from experiences import reservations as R

    try:
        actives = R.actives()
        file = R.lister_file()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[campagne] keys and queue unreadable at startup ({e})")
        return
    if actives:
        suite = (
            "les expériences qui en ont besoin attendront en file."
            if campagne.a_travers_les_quotas
            else "une expérience de la campagne qui en a besoin sera notée « non jouée »."
        )
        logger.warning(
            "[campagne] au démarrage, des clés sont déjà tenues : "
            + " · ".join(f"{a.get('exp')} ({', '.join(a.get('cles') or [])})" for a in actives)
            + f" — {suite}")
    else:
        logger.info("[campagne] at startup: no key held")
    if file:
        suite = (
            "l'ordonnanceur les démarrera quand leurs clés se libèrent."
            if campagne.a_travers_les_quotas
            else "la campagne ne les démarre pas."
        )
        logger.warning(
            f"[campagne] au démarrage, {len(file)} demande(s) en file : "
            + " · ".join(f"{e.get('exp')} (soumise {e.get('soumis')})" for e in file)
            + f" — {suite}")


@dataclass
class _Vue:
    """What the campaign sees at a time t, once the states are read from disk."""

    faites: list[str] = field(default_factory=list)
    en_vol: list[str] = field(default_factory=list)
    dorment: list[str] = field(default_factory=list)   # en_vol AND waiting for quota
    a_reprendre: list[str] = field(default_factory=list)
    jamais_lancees: list[str] = field(default_factory=list)


def _refus_du_dernier_lancement(exp: str) -> tuple[str | None, list[str]]:
    """What the last refused launch left behind, or `(None, [])` if it said nothing.

    ONLY the most recent marker is read, and only if it is later than the last
    launch log: yesterday's marker says nothing about today's launch.
    """
    base = dossier_lancements(exp)
    if not base.is_dir():
        return None, []
    marqueurs = sorted(base.glob(f"*{REFUS.SUFFIXE_MARQUEUR}"))
    if not marqueurs:
        return None, []
    journaux = sorted(p for p in base.glob("*.log"))
    if journaux and journaux[-1].stem > marqueurs[-1].name.split(".")[0]:
        return None, []
    try:
        contenu = json.loads(marqueurs[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, []
    return contenu.get("classe"), list(contenu.get("motifs") or [])


def _reporter_experience(exp: str) -> bool:
    """Stops an experiment asleep on its quota, to hand over to the next one.

    Goes through the STOP file of its run folder — the existing stop mechanism, which
    the runner already watches. The run closes as `arretee`, hence resumable: that is what
    lets the retry pass relaunch it where it had stopped, without losing anything.
    """
    dossier = derniere_execution(exp)
    if dossier is None:
        return False
    try:
        (dossier / "STOP").write_text(
            "campagne: quota épuisé, expérience reportée à la passe de repêchage\n",
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - dépend du système de fichiers
        logger.error(f"[campagne] cannot defer {exp} ({exc})")
        return False
    return True


def _reste_ailleurs(
    campagne: Campagne, etat: dict, phase: Phase, en_vol: list[str]
) -> list[str]:
    """What could run NOW if we dropped what is asleep on its quota.

    The current phase first, then all the following ones: a quota exhausted at one provider
    says nothing about the others, and a multi-provider campaign has no reason to wait for
    the window of one to launch the arms of the other.
    """
    index = campagne.phases.index(phase)
    a_venir = [phase, *campagne.phases[index + 1:]]
    return [
        e
        for p in a_venir
        for e in p.experiences
        if e not in etat["faites"]
        and e not in etat["echouees"]
        and e not in etat["reportees"]
        # What is ALREADY in flight is not "something else to do": without this exclusion, the
        # sleeping experiments would count themselves as an alternative to
        # themselves, and the campaign would never sleep.
        and e not in en_vol
    ]


def _pause_est_subie(dossier: str | Path | None) -> bool:
    """Is this paused run waiting to be resumed, or to be left alone?

    `en_pause` covers three situations that the runner distinguishes, but that the state alone
    conflates:

    * the **watchdog** cut it after 420 s without progress — the process is dead, nobody
      will resume it any more;
    * the run stopped **incomplete** without anyone asking for a pause — same
      thing, its own message says "reprendre";
    * a **human** asked for the pause — it waits for a human decision, not for the campaign.

    The first two are *suffered*: the campaign resumes them. The third is not resumed
    behind the back of whoever asked for it. The discriminant is the structured trace
    `interruptions[].raison`, not the message of `etat.json`: the latter is addressed to
    humans and may be reworded without anything visibly breaking.

    Measured on 2026-09-16: without this distinction, a watchdog pause froze the
    campaign for good — counted "in flight", it was neither resumed nor declared failed,
    and the experiments behind it were never launched. Six hours of silence.

    An unreadable or missing `execution.yaml` returns `True`. A silent, unbounded block
    is worse than one resumption too many: resumption is capped by `TENTATIVES_MAX`, after
    which the experiment is declared failed and the campaign goes on. The block has
    no cap at all.
    """
    if dossier is None:
        return True
    try:
        config = yaml.safe_load((Path(dossier) / F_EXECUTION).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return True
    if not isinstance(config, dict):
        return True
    pauses = [
        i
        for i in (config.get("interruptions") or [])
        if isinstance(i, dict) and i.get("cause") == "pause"
    ]
    if not pauses:
        # No pause requested, and yet the run is paused: it is the runner's
        # "incomplète — reprendre" case.
        return True
    return str(pauses[-1].get("raison") or "").startswith(RAISON_PAUSE_SUBIE)


def _observer(experiences: list[str], echouees: dict, reportees: dict | None = None) -> _Vue:
    vue = _Vue()
    reportees = reportees or {}
    for exp in experiences:
        if exp in echouees or exp in reportees:
            continue
        infos = etat_experience(exp)
        etat = infos["etat"]
        if etat == ETAT_TERMINEE:
            vue.faites.append(exp)
        elif etat == "definie":
            vue.jamais_lancees.append(exp)
        elif etat in ETATS_REPRENABLES or (
            etat == ETAT_EN_PAUSE and _pause_est_subie(infos["dossier"])
        ):
            vue.a_reprendre.append(exp)
        else:  # en_cours, requested pause, en_attente_quota, en_attente_agent
            vue.en_vol.append(exp)
            if etat == ETAT_EN_ATTENTE_QUOTA:
                vue.dorment.append(exp)
    return vue


def lancer(
    nom: str,
    *,
    reprendre: bool = True,
    intervalle_s: float = INTERVALLE_S,
    max_tours: int | None = None,
    dormir: Callable[[float], None] = time.sleep,
) -> int:
    """Carries the campaign through to the end. Returns 0 on success, 1 if experiments failed or
    remained unfinished, 130 if a stop was requested.

    `max_tours` and `dormir` exist for the tests: the logic is the same, only time
    is injected. A loop that can only be tested by really waiting thirty seconds is not
    tested.
    """
    campagne = charger(nom)

    # BEFORE touching anything — in particular before clearing the stop flag,
    # which a second launch would erase from under the first one's feet.
    deja = campagne_en_vol(nom)
    if deja is not None:
        logger.error(
            f"[ALARME] [campagne] {nom} TOURNE DÉJÀ (pid {deja}) — ce lancement s'arrête "
            "sans rien toucher. Deux processus sur la même campagne écrivent le même "
            "etat.json sans se voir : ils se disputent les expériences et finissent par en "
            f"déclarer perdues qui n'ont jamais tourné. Suivre celle qui tourne : "
            f"make campagne-etat NOM={nom} — l'arrêter : make campagne-arreter NOM={nom}")
        return CODE_DEJA_EN_VOL

    _lever_arret(nom)

    etat = (lire_etat(nom) if reprendre else None) or etat_par_defaut(campagne)
    # A state written before deferral existed does not carry these keys: setting them here
    # avoids making resumption depend on a file migration.
    etat.setdefault("reportees", {})
    etat.setdefault("repechages", 0)
    etat["pid"] = os.getpid()
    if reprendre and etat.get("terminee_le"):
        logger.info(f"[campagne] {nom} was already finished on {etat['terminee_le']} — "
                    "relaunch with --recommencer to replay it.")
        return 0
    ecrire_etat(nom, etat)

    total = len(campagne.toutes)
    logger.info(
        f"[campagne] {nom} started — {total} experiment(s) in {len(campagne.phases)} phase(s): "
        + " · ".join(f"{p.nom} ({len(p.experiences)})" for p in campagne.phases)
    )
    if campagne.note:
        logger.info(f"[campagne] {campagne.note}")
    logger.info(
        "[campagne] mode: "
        + ("across quotas — batch sleep, deferral, retry pass, queue promotion"
           if campagne.a_travers_les_quotas
           else "chaining — one experiment at a time; stopped on quota, it is marked not "
                "finished; deprived of its keys, not played; the campaign moves to the next, "
                "no sleep nor retry pass"))
    _signaler_cles_au_demarrage(campagne)

    tours = 0
    tentatives: dict[str, int] = {}
    depuis: dict[str, float] = {}
    alarme_sommeil = False
    # Rising edge of the "in flight but frozen" alarm, one entry per experiment: without it
    # the message would go out at every round, i.e. twice a minute, and drown the log
    # it is meant to make readable.
    alarme_en_vol: set[str] = set()
    # An experiment just launched has not written its `etat.json` yet: while
    # `lancer` starts, reserves its keys and opens its run folder, it still reads
    # as "definie". Without this grace period, the campaign relaunched it at every round —
    # and an experiment launched twice overwrites its own run.
    lancees: dict[str, float] = {}
    # The last state TAKEN INTO ACCOUNT for each experiment this process launched or saw in
    # flight. A resumable state that differs from it results from a launch; an identical one
    # is the one from BEFORE, and says nothing about this launch. On 29/09, the previous day's
    # `epuisee` state was counted at every round: three "failures" in a minute, without a run.
    vus: dict[str, tuple] = {}
    en_file_annoncees: set[str] = set()
    # Experiment → why it is relaunched, for the log of the launch that follows.
    relances: dict[str, str] = {}

    def _clore_en_echec(exp: str, fiche: dict, message: str) -> None:
        etat["echouees"][exp] = {**fiche, "le": _iso()}
        etat["restantes"] = [e for e in etat["restantes"] if e != exp]
        if etat.get("courante") == exp:
            etat["courante"] = None
        ecrire_etat(nom, etat)
        logger.error(message)

    def _demarrer(exp: str, libelle: str) -> None:
        vus[exp] = _instantane(exp)
        if _retirer_demande(exp):
            logger.warning(f"[campagne] {exp} had an old request in the key queue "
                           "— removed: this launch replaces it.")
        depuis.setdefault(exp, time.time())
        etat["courante"] = exp
        ecrire_etat(nom, etat)
        logger.info(libelle)
        lancees[exp] = time.time()
        _lancer_experience(exp, campagne.lanceurs,
                           attendre_fenetre=campagne.a_travers_les_quotas)

    while True:
        if max_tours is not None and tours >= max_tours:
            logger.info(f"[campagne] {nom}: {max_tours} rounds reached (bounded mode) — exit")
            return 0
        tours += 1

        if demande_arret(nom):
            logger.info(f"[campagne] {nom} STOPPED on request — {len(etat['faites'])}/{total} "
                        "done; the run in progress finishes on its own.")
            ecrire_etat(nom, etat)
            return 130

        if campagne.a_travers_les_quotas:
            _tour_ordonnanceur()
        else:
            _reconcilier_cles()

        # ── Where are we? ─────────────────────────────────────────────────────
        phase = next((p for p in campagne.phases if p.nom == etat["phase_courante"]), None)
        if phase is None:  # phase gone from the file between two launches
            raise CampagneInvalide(
                f"the state of {nom} names phase {etat['phase_courante']!r}, missing from the "
                "campaign file. Fix the file, or relaunch with --recommencer.")
        vue = _observer(list(phase.experiences), etat["echouees"], etat["reportees"])
        for exp in vue.en_vol:
            # Seen in flight without this process launching it (campaign restarted, launch by
            # hand): its current state becomes the reference, its end will read as new.
            vus.setdefault(exp, _instantane(exp))

        # ── In flight, really? ────────────────────────────────────────────────
        # The grace period below only watches the launches of THIS process
        # (`lancees`): a run stuck before a campaign restart escaped
        # all monitoring. Here nothing is assumed about the pid — we look at whether the state
        # moves. The campaign kills nothing and relaunches nothing: it SAYS so, and that is
        # already what was missing on 2026-09-16, when six hours went by without a word.
        #
        # What is ASLEEP on its quota is excluded: its state does not move either, but it is
        # an understood wait, already recorded by the batch sleep and its 26 h alarm.
        # Raising an alarm here would duplicate the message and pass a normal wait off as an
        # anomaly — exactly what an alarm must avoid.
        # What is BEING LAUNCHED too: a resumption keeps the previous day's state as long as its
        # launcher has rewritten nothing, which would make it look frozen for hours.
        for exp in [e for e in vue.en_vol if e not in vue.dorment and e not in lancees]:
            fige = _fige_depuis(etat_experience(exp))
            if fige is None or fige < EN_VOL_FIGE_S:
                alarme_en_vol.discard(exp)
                continue
            if exp in alarme_en_vol:
                continue
            alarme_en_vol.add(exp)
            infos = etat_experience(exp)
            logger.error(
                f"[ALARME] [campagne] {exp} est comptée EN VOL, mais son état n'a pas bougé "
                f"depuis {fige / 60:.0f} min — état {infos['etat']!r}, motif : "
                f"{infos.get('raison')!r}, dossier {infos['dossier']}. La campagne ne la "
                f"touche pas et attend derrière elle. Reprendre : "
                f"make experience-reprendre EXP={exp} — arrêter : "
                f"make experience-arreter EXP={exp}"
            )

        # ── Being launched: a new state, a queueing, or silence ──────────────
        # A launched experiment keeps the state from BEFORE until its launcher writes something:
        # `definie` for a first run, the previous day's state for a resumption. Neither
        # says anything about this launch; it therefore counts as in flight, until
        # a new state, a request in the key queue, or the end of the grace period.
        for exp, quand in list(lancees.items()):
            neuf = _instantane(exp)
            # `definie` WITH a folder: `lancer` has just created the run and has not yet
            # reserved its keys — it may still end up queued, and its folder disappear.
            if neuf != vus.get(exp) and neuf[2] != "definie":
                lancees.pop(exp, None)
                en_file_annoncees.discard(exp)
                continue
            for liste in (vue.a_reprendre, vue.jamais_lancees, vue.en_vol, vue.dorment):
                if exp in liste:
                    liste.remove(exp)
            demande = _demande_en_file(exp)
            if demande is not None:
                tenues = _detenteurs(demande.get("cles"))
                if campagne.a_travers_les_quotas:
                    if exp not in en_file_annoncees:
                        en_file_annoncees.add(exp)
                        logger.info(
                            f"[campagne] ⏳ {exp} QUEUED — {tenues}. It will start when the "
                            "keys are released; no attempt counted.")
                    vue.en_vol.append(exp)
                    continue
                # Chaining: waiting for keys that another experiment holds is sleeping
                # without saying so. The request leaves the queue — if left, the scheduler
                # would start it later, outside the campaign — and the campaign moves on.
                lancees.pop(exp, None)
                _retirer_demande(exp)
                _clore_en_echec(
                    exp,
                    {"motif": f"non jouée — {tenues}", "classe": "cles_tenues",
                     "tentatives": tentatives.get(exp, 0)},
                    f"[ALARME] [campagne] {exp} NON JOUÉE — {tenues}. Sa demande est retirée "
                    "de la file (rien ne la démarrera hors campagne) ; la campagne passe à la "
                    f"suivante. La relancer quand les clés sont libres : "
                    f"make experience-reprendre EXP={exp}")
                continue
            if time.time() - quand <= DELAI_ECRITURE_ETAT_S:
                vue.en_vol.append(exp)
                continue
            lancees.pop(exp, None)
            # A launch that never writes a state is a launch that FAILED, not a
            # slow launch. It therefore counts as an attempt — otherwise the campaign
            # relaunches it endlessly: measured on 2026-09-15, 178 relaunches in six hours on a
            # control whose artefact was refused, the phase stuck and the LLM arms never
            # reached. A silent block is worth less than a declared failure.
            # …EXCEPT when the launcher said why it refused: it then leaves a marker
            # next to its log. On 2026-09-16, four arms were declared broken for
            # the sole reason that nobody read what the refusal said.
            classe, motifs = _refus_du_dernier_lancement(exp)
            if classe is not None and REFUS.est_reportable(classe):
                if campagne.a_travers_les_quotas:
                    # A shortage of today's quota is DEFERRED without consuming an attempt;
                    # the retry pass will take it back.
                    etat["reportees"][exp] = {
                        "motif": "; ".join(motifs)[:300] or classe, "classe": classe,
                        "le": _iso()}
                    if etat.get("courante") == exp:
                        etat["courante"] = None
                    ecrire_etat(nom, etat)
                    logger.info(
                        f"[campagne] ⏭ {exp} DEFERRED at launch — {classe}: the launcher "
                        f"refused to create the run for lack of today's quota. No attempt "
                        f"counted; it will come back at the retry pass.")
                    continue
                _clore_en_echec(
                    exp,
                    {"motif": "non jouée — quota épuisé au lancement : "
                              + ("; ".join(motifs)[:300] or classe),
                     "classe": classe, "tentatives": tentatives.get(exp, 0)},
                    f"[ALARME] [campagne] {exp} NON JOUÉE — le lanceur a refusé faute de quota "
                    f"du jour ({'; '.join(motifs)[:200]}). Ni report ni repêchage : la campagne "
                    "passe à la suivante.")
                continue
            tentatives[exp] = tentatives.get(exp, 0) + 1
            journal = dossier_lancements(exp)
            if classe is not None:
                # The refusal has a known reason: state it, rather than "sans jamais écrire
                # d'état", which sends people looking for a non-existent failure.
                _clore_en_echec(
                    exp,
                    {"motif": f"lancement refusé ({classe}) : " + "; ".join(motifs)[:300],
                     "classe": classe, "tentatives": tentatives[exp]},
                    f"[ALARME] [campagne] {exp} : lancement refusé — {classe}. "
                    + "; ".join(motifs)[:200])
                continue
            if tentatives[exp] > TENTATIVES_MAX:
                _clore_en_echec(
                    exp,
                    {"motif": f"lancée {tentatives[exp]} fois sans jamais écrire d'état "
                              f"— voir {journal}",
                     "tentatives": tentatives[exp]},
                    f"[ALARME] [campagne] {exp} lancée {tentatives[exp]} fois sans jamais "
                    f"écrire d'état — déclarée en échec, la campagne passe à la suivante. "
                    f"Le refus est dans {journal}.")
            else:
                relances[exp] = (f"lancée il y a plus de {DELAI_ECRITURE_ETAT_S:.0f} s sans "
                                 f"avoir écrit d'état — relance {tentatives[exp]}/"
                                 f"{TENTATIVES_MAX}")
                logger.warning(
                    f"[campagne] {exp} {relances[exp]}. Any refusal is in {journal}.")

        for exp in vue.faites:
            if exp not in etat["faites"]:
                duree = time.time() - depuis.pop(exp, time.time())
                etat["faites"].append(exp)
                etat["restantes"] = [e for e in etat["restantes"] if e != exp]
                if etat.get("courante") == exp:
                    etat["courante"] = None
                logger.info(
                    f"[campagne] ✅ {exp} finished in {duree:.0f} s — "
                    f"{len(etat['faites'])}/{total} done, "
                    f"{len(etat['restantes'])} remaining, {len(etat['echouees'])} failed")
                ecrire_etat(nom, etat)

        # ── Phase closed? ─────────────────────────────────────────────────────
        reste = [e for e in phase.experiences
                 if e not in etat["faites"]
                 and e not in etat["echouees"]
                 and e not in etat["reportees"]]
        if not reste:
            suivantes = [p for p in campagne.phases
                         if campagne.phases.index(p) > campagne.phases.index(phase)]
            if not suivantes:
                # ── Retry pass: nothing is abandoned without a second chance ───────
                # Everything that was deferred (quota) or declared failed comes back, attempt
                # counters reset to zero. Without this pass, a quota exhausted mid-campaign
                # cost the experiment for good: it left observation and
                # never came back, even when relaunching the campaign.
                # "Across quotas" mode only: in chaining, once the list has been played
                # it is over — the author takes back control of what is left.
                a_repecher = list(etat["reportees"]) + list(etat["echouees"])
                if (campagne.a_travers_les_quotas and a_repecher
                        and etat["repechages"] < REPECHAGES_MAX):
                    etat["repechages"] += 1
                    reportees, echouees = dict(etat["reportees"]), dict(etat["echouees"])
                    etat["reportees"], etat["echouees"] = {}, {}
                    for exp in reportees:
                        # A deferral is not a failed attempt: counter reset.
                        tentatives.pop(exp, None)
                    for exp in echouees:
                        # A real failure is only entitled to ONE relaunch per retry pass, not
                        # to a new counter: otherwise a broken experiment would be relaunched
                        # TENTATIVES_MAX times at every pass, and the cap would mean nothing.
                        tentatives[exp] = TENTATIVES_MAX - 1
                    premiere = next(
                        (p for p in campagne.phases
                         if any(e in a_repecher for e in p.experiences)),
                        campagne.phases[0],
                    )
                    etat["phase_courante"] = premiere.nom
                    etat["courante"] = None
                    ecrire_etat(nom, etat)
                    logger.info(
                        f"[campagne] ↺ RETRY PASS {etat['repechages']}/{REPECHAGES_MAX} — "
                        f"{len(reportees)} deferred and {len(echouees)} failed come back, "
                        f"from phase {premiere.nom!r}: "
                        + " · ".join(a_repecher)
                    )
                    continue

                etat["terminee_le"] = _iso()
                etat["courante"] = None
                ecrire_etat(nom, etat)
                echecs = len(etat["echouees"])
                reports = len(etat["reportees"])
                logger.info(
                    f"[campagne] {nom} FINISHED — {len(etat['faites'])}/{total} done, "
                    f"{echecs} failed, {reports} still deferred, "
                    f"{etat['repechages']} retry pass(es), "
                    f"{len(etat['sommeils'])} sleep(s), {tours} round(s).")
                for exp, det in etat["reportees"].items():
                    logger.error(
                        f"[ALARME] [campagne] {exp} reste REPORTÉE après "
                        f"{etat['repechages']} repêchage(s) — {det.get('motif')}. Son quota ne "
                        "s'est pas rouvert dans la campagne : relancez-la seule plus tard.")
                if echecs or reports:
                    for exp, det in etat["echouees"].items():
                        logger.error(f"[campagne] open failure: {exp} — {det.get('motif')}")
                    return 1
                return 0
            etat["phase_courante"] = suivantes[0].nom
            ecrire_etat(nom, etat)
            logger.info(f"[campagne] phase {phase.nom!r} closed — moving to "
                        f"{suivantes[0].nom!r} ({len(suivantes[0].experiences)} experiments)"
                        + (f" : {suivantes[0].raison}" if suivantes[0].raison else ""))
            continue

        # ── Resumptions and failures ──────────────────────────────────────────
        # Only a NEW state — written after a launch by this process — is judged. The state
        # from before (the previous day's, or a previous campaign's) says nothing about any
        # launch: the experiment is simply relaunched, by the "nothing in flight" block below.
        for exp in vue.a_reprendre:
            instant = _instantane(exp)
            if exp not in vus or instant == vus[exp]:
                continue
            vus[exp] = instant
            infos = etat_experience(exp)
            motif = infos.get("raison") or infos["etat"]
            if not campagne.a_travers_les_quotas and infos["etat"] == ETAT_EPUISEE:
                _clore_en_echec(
                    exp,
                    {"motif": f"non terminée — quota épuisé : {motif}",
                     "classe": REFUS.QUOTA_EPUISE, "tentatives": tentatives.get(exp, 0)},
                    f"[ALARME] [campagne] {exp} NON TERMINÉE — quota épuisé ({motif}). Pas de "
                    "sommeil ni de relance : la campagne passe à la suivante. La reprendre là "
                    f"où elle s'est arrêtée : make experience-reprendre EXP={exp}")
                continue
            tentatives[exp] = tentatives.get(exp, 0) + 1
            if tentatives[exp] > TENTATIVES_MAX:
                _clore_en_echec(
                    exp,
                    {"motif": str(motif), "tentatives": tentatives[exp]},
                    f"[ALARME] [campagne] {exp} échoue pour la {tentatives[exp]}ᵉ fois "
                    f"({motif}) — déclarée en échec, la campagne passe à la suivante. "
                    "Les expériences restantes ne sont PAS annulées.")
            else:
                relances[exp] = (f"tentative {tentatives[exp]}/{TENTATIVES_MAX}, "
                                 f"motif : {motif}")

        # ── Quota exhausted: defer rather than sleep in front of a full queue ─
        # Sleeping is right when there is NOTHING else to do, and wrong as soon as another
        # experiment could run — a quota exhausted at one provider says nothing about the
        # others. Measured on 2026-09-15: a three-provider campaign slept up to 24 h
        # on the first one's quota, the arms of the other two stopped behind it.
        # "Across quotas" mode only: chaining launches without waiting for the
        # window, and therefore never sleeps.
        if (campagne.a_travers_les_quotas and vue.en_vol and vue.dorment
                and len(vue.dorment) == len(vue.en_vol)):
            ailleurs = _reste_ailleurs(campagne, etat, phase, vue.en_vol)
            if ailleurs:
                for exp in vue.dorment:
                    motif = etat_experience(exp).get("raison") or ETAT_EN_ATTENTE_QUOTA
                    if _reporter_experience(exp):
                        etat["reportees"][exp] = {"motif": str(motif), "le": _iso()}
                        if etat.get("courante") == exp:
                            etat["courante"] = None
                        logger.info(
                            f"[campagne] ⏭ {exp} DEFERRED (quota: {motif}) — "
                            f"{len(ailleurs)} experiment(s) can run without it. "
                            "It will come back at the retry pass, resumed where it "
                            "stopped."
                        )
                ecrire_etat(nom, etat)
                dormir(intervalle_s)
                continue
            quand, secondes = prochain_reveil()
            if secondes > SOMMEIL_SUSPECT_S and not alarme_sommeil:
                alarme_sommeil = True
                logger.error(
                    f"[ALARME] [campagne] {nom} would sleep {secondes / 3600:.1f} h — a "
                    "quota window lasts 24. It is probably not a quota: "
                    "check the machine clock and the time zones of the providers.")
            etat["sommeils"].append({"depuis": _iso(), "jusqu": quand, "duree_s": secondes,
                                     "motif": f"quota épuisé sur {len(vue.dorment)} "
                                              f"exécution(s) en vol"})
            ecrire_etat(nom, etat)
            logger.info(
                f"[campagne] 💤 {nom} asleep: the {len(vue.dorment)} run(s) in flight "
                f"all wait for the quota window. Wake-up at {quand} "
                f"(in {secondes / 3600:.1f} h). Resumption will happen on the current "
                "experiment, not at the start.")
            dormir(min(secondes + MARGE_REVEIL_S, 3600))
            continue
        alarme_sommeil = False

        # ── Nothing in flight? Launch the next one ────────────────────────────
        # One at a time, in file order: a resumption does not jump the queue.
        # `reste` predates the resumptions: what they have just closed is no longer part of it.
        if not vue.en_vol:
            suivante = next((e for e in phase.experiences
                             if e in reste and e not in etat["echouees"]
                             and e not in etat["reportees"]), None)
            if suivante is not None:
                rang = f"phase {phase.nom}, {len(etat['faites']) + 1}/{total}"
                if suivante in relances:
                    libelle = f"[campagne] ↻ {suivante} — {relances.pop(suivante)} ({rang})"
                elif derniere_execution(suivante) is not None:
                    libelle = f"[campagne] ↻ reprise de {suivante} ({rang})"
                else:
                    libelle = f"[campagne] ▶ {suivante} ({rang})"
                _demarrer(suivante, libelle)

        if tours % 20 == 0:
            logger.info(
                f"[campagne] {nom} alive — {len(etat['faites'])}/{total} done, "
                f"in flight: {vue.en_vol or '—'}, {tours} rounds")
        dormir(intervalle_s)


# ── Reading for display (CLI and dashboard) ──────────────────────────────────


def etat_lisible(nom: str) -> dict:
    """Everything needed to display a campaign, without deciding anything.

    Does not raise when the campaign has never run: `etat` is then None and the caller
    says so — a campaign defined but never launched is not an error.
    """
    campagne = charger(nom)
    etat = lire_etat(nom)
    quand, secondes = prochain_reveil()
    par_experience = {e: etat_experience(e) for e in campagne.toutes}
    faites = [e for e, s in par_experience.items() if s["etat"] == ETAT_TERMINEE]
    return {
        "nom": campagne.nom,
        "note": campagne.note,
        "substrat": campagne.substrat,
        "phases": [{"nom": p.nom, "raison": p.raison,
                    "experiences": list(p.experiences),
                    "faites": [e for e in p.experiences if e in faites]}
                   for p in campagne.phases],
        "total": len(campagne.toutes),
        "faites": faites,
        "reportees": (etat or {}).get("reportees", {}),
        "repechages": (etat or {}).get("repechages", 0),
        "etat": etat,
        "arret_demande": demande_arret(nom),
        "par_experience": par_experience,
        "prochain_reveil": quand,
        "secondes_avant_reveil": secondes,
    }


__all__ = [
    "CODE_DEJA_EN_VOL",
    "INTERVALLE_S",
    "VERSION_CAMPAGNE",
    "Campagne",
    "CampagneInvalide",
    "Phase",
    "arreter",
    "campagne_en_vol",
    "etat_lisible",
    "lancer",
    "charger",
    "chemin_definition",
    "derniere_execution",
    "dossier_campagnes",
    "dossier_etat",
    "ecrire_etat",
    "etat_experience",
    "etat_par_defaut",
    "journal_lancement",
    "lire_etat",
    "prochain_reveil",
]
