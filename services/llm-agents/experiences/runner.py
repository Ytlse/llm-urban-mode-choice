"""Run without the simulator (ticket 035, spec 03) with resources and resume (spec 05).

For each person, their trips of the day **in order**, calling the single decision
(spec 02) with the proposals of the recorded set (spec 01). No routing engine. Persons
advance in parallel (`regroupement.parallelisme`), never two trips of the same
person (S4). Everything decided is archived at once (Q9); on resume, the archived
decisions are served again and their effect on the vehicle chain replayed without any request (Q5).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from experiences import decision as D
from experiences import journal
from experiences.archive import (
    ETAT_ARRETEE,
    ETAT_EN_ATTENTE_QUOTA,
    ETAT_EN_COURS,
    ETAT_EN_PAUSE,
    ETAT_EPUISEE,
    ETAT_TERMINEE,
    METHODE_INEXPLOITABLE,
    METHODE_NON_COUVERT,
    Execution,
)
from experiences.experience import Experience
from experiences.jeu import Deplacement, Jeu, deplacements_attendus
from experiences.ressources import MoniteurRessources, prochaine_fenetre_quota
from loguru import logger
from models import Person

FICHIER_PAUSE = "PAUSE"
FICHIER_STOP = "STOP"


def _env_flottant(nom: str, defaut: float) -> float:
    """Operating setting read from the environment. An unreadable value breaks nothing:
    it is reported and the default applies."""
    brut = os.environ.get(nom)
    if brut is None or not brut.strip():
        return defaut
    try:
        return float(brut)
    except ValueError:
        logger.warning(
            f"[execution] {nom}={brut!r} unreadable — default {defaut} kept"
        )
        return defaut


# Time left to a request ALREADY IN FLIGHT when a pause / stop is requested. Past this
# delay, the network call is abandoned (never an archive write): the trip stays not
# archived, hence requested again as is on resume (Q5/Q6), and nothing is skipped or made up.
# Without this grace, the pause waited for the call to return — up to `remote_llm_poll_timeout`, 120 s.
DELAI_GRACE_PAUSE_S = _env_flottant("EXP_PAUSE_GRACE_S", 15.0)

# AUTOMATIC pause after this delay without a single trip settled (0 or negative =
# disabled). A run that no longer advances reports itself and pauses instead of staying
# "in progress" forever; it remains resumable.
INACTIVITE_PAUSE_S = _env_flottant("EXP_INACTIVITE_PAUSE_S", 420.0)

# Maximum age of the quota snapshot during a run (0 or negative = disabled). Without this
# refresh, `moniteur.etat` stayed the one from STARTUP: a key that ran out along the
# way stayed "available", the decision-maker kept pinning it (`force_provider`
# is strict on the gateway side, which refuses rather than substitutes) and the next key was
# never touched. On 2026-09-08, a run stopped at 70.5 % with 500 untouched requests in
# reserve on the second key.
RAFRAICHIR_QUOTAS_S = _env_flottant("EXP_RAFRAICHIR_QUOTAS_S", 30.0)


@dataclass
class Controle:
    """Pause / stop requested by a file in the run directory, or by a signal (Q7, Q8).

    Also carries the **grace delay**: a pause requested while a request is in
    flight no longer waits for its return indefinitely. Past `grace_s`, the call is abandoned — the
    trip, not archived, will be requested again on resume (decision of 2026-09-08, revising
    that of 2026-09-07: the pause took effect up to 2 min after the click).
    """

    dossier: Path
    pause: bool = False
    arret: bool = False
    installes: bool = field(default=False, repr=False)
    grace_s: float = field(default_factory=lambda: DELAI_GRACE_PAUSE_S)
    # Why the pause: "manuelle", "inactivite:<n>s", "signal:SIGINT"… Carried over into
    # the run state and into its interruption history.
    raison: str = ""
    demande_a: float | None = field(default=None, repr=False)
    _en_vol: set[asyncio.Task] = field(default_factory=set, repr=False)
    _abandonnees: set[asyncio.Task] = field(default_factory=set, repr=False)

    def demande_pause(self) -> bool:
        return self.pause or (self.dossier / FICHIER_PAUSE).exists()

    def demande_arret(self) -> bool:
        return self.arret or (self.dossier / FICHIER_STOP).exists()

    def interrompu(self) -> bool:
        if self.demande_pause() or self.demande_arret():
            if self.demande_a is None:
                self.demande_a = time.monotonic()  # start of the grace
            return True
        return False

    def grace_echue(self) -> bool:
        """Has the grace left to in-flight requests elapsed?"""
        return (
            self.demande_a is not None
            and (time.monotonic() - self.demande_a) >= self.grace_s
        )

    # ── in-flight requests ──
    def suivre(self, tache: asyncio.Task) -> None:
        self._en_vol.add(tache)

    def oublier(self, tache: asyncio.Task) -> None:
        self._en_vol.discard(tache)
        self._abandonnees.discard(tache)

    def abandonnee(self, tache: asyncio.Task) -> bool:
        """Was this call cancelled BY US? Tells a deliberate abandonment from a
        cancellation coming from outside, which must be let through."""
        return tache in self._abandonnees

    def abandonner_en_vol(self) -> int:
        """Cancels the decision-maker calls still in flight. Returns their number (0 if none)."""
        a_annuler = [t for t in self._en_vol if not t.done()]
        for t in a_annuler:
            self._abandonnees.add(t)
            t.cancel()
        return len(a_annuler)

    def nettoyer(self) -> list[str]:
        """Removes the interruption sentinels. Returns those that existed (empty if none)."""
        retirees: list[str] = []
        for f in (FICHIER_PAUSE, FICHIER_STOP):
            try:
                (self.dossier / f).unlink()
            except FileNotFoundError:
                pass
            else:
                retirees.append(f)
        return retirees

    def installer_signaux(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, self._sur_signal, "SIGINT")
            loop.add_signal_handler(signal.SIGTERM, self._sur_signal, "SIGTERM")
            self.installes = True
        except (NotImplementedError, RuntimeError, ValueError):
            self.installes = False

    def _sur_signal(self, nom: str) -> None:
        logger.warning(
            f"[execution] {nom} received: pausing — in-flight requests abandoned "
            f"within {self.grace_s:.0f} s at most"
        )
        self.raison = self.raison or f"signal:{nom}"
        self.pause = True


def _decalage_calendrier(exp: Experience, jeu: Jeu) -> int:
    """Policy `commune`: the experiment date shifts timestamps by whole days;
    the transport offer stays that of the set (question 11). Other policies: the per-person
    date is drawn by the weather settings (`weather_per_agent_dates`), not here."""
    if exp.calendrier.politique != "commune":
        return 0
    return (
        date.fromisoformat(exp.calendrier.date) - date.fromisoformat(jeu.jour_simule)
    ).days * 86400


def _anticipation(person: Person, activity_id: str, departure_time: int) -> dict | None:
    """Same anticipation block as the simulation (ticket 014) — or nothing if unavailable."""
    from settings import settings

    if not settings.agent.agenda_anticipation_enabled:
        return None
    act = next(
        (a for a in (person.identity.activities or []) if a.id == activity_id), None
    )
    if act is None:
        return None
    try:
        from urban_mobility_agents.simulation_controller import _build_anticipation

        return _build_anticipation(person, act, departure_time)
    except Exception as e:  # noqa: BLE001 — anticipation is a context, not a rule: we say so
        logger.warning(
            f"[execution] anticipation indisponible pour {person.person_id}: {e}"
        )
        return None


def _methode_moves(decision: D.Decision, decideur) -> str:
    """Kept for legacy callers; the rule lives in `journal.methode_moves`.

    Ticket 081 — the label is now computed from the METHOD alone, not from a
    `Decision` object, so that regenerating a log from the archive produces exactly
    the same labels as live writing.
    """
    return journal.methode_moves(decision.methode, decideur)


FICHIER_PROGRESSION = (
    "progression.json"  # read by the dashboard (progress bar)
)


class Progression:
    """Visible progress (S6): log every `periode_s` seconds AND the directory's `progression.json`."""

    def __init__(
        self,
        attendus: int,
        personnes: int,
        deja: int,
        periode_s: float = 5.0,
        dossier: Path | None = None,
    ):
        self.attendus, self.personnes, self.deja, self.periode_s = (
            attendus,
            personnes,
            deja,
            periode_s,
        )
        self.dossier = dossier
        self.debut = time.monotonic()
        self.compteurs: Counter = Counter()
        self.personnes_terminees = 0
        # Two counters, two meanings (decision of 2026-09-07). R1 NEVER skips a
        # trip: a failed attempt is retried until a decision. Counting it
        # as an "error" showed 128 errors on a run that had none, and
        # hid the real signal — the provider's throughput.
        #   attentes_par_type : attempts that failed THEN were retried (the nominal case)
        #   erreurs_par_type  : final failure, trip closed without a decision. Stays empty
        #                       as long as R1 holds: a non-zero value EXPOSES a violation.
        self.attentes_par_type: Counter = Counter()
        self.erreurs_par_type: Counter = Counter()
        # Trips waiting for a transient resource (R3): (person, activity) → (t0, type).
        self._attentes: dict[tuple[str, str], tuple[float, str]] = {}
        self._attente_quota_jusqu: str | None = (
            None  # current quota window (R4), ISO UTC or None
        )
        # Watchdog idleness clock: time of the last settled trip.
        self._avancement_vu = 0
        self._avancement_t = self.debut

    def avancement(self) -> int:
        """Number of trips SETTLED since the start, whatever the outcome — decided,
        served again from the archive, not covered, unusable, without solution. This is the counter
        that moves when the run advances; if it no longer moves, the run no longer advances."""
        return (
            self.traites()
            + int(self.compteurs["resservies"])
            + int(self.compteurs["inexploitables"])
        )

    def immobile_depuis(self) -> float | None:
        """Seconds elapsed without a single settled trip, or `None` when idleness is
        INTENDED — waiting for the quota window (R4, `--attendre-fenetre`): there, doing nothing
        is the requested behaviour, the watchdog stays quiet."""
        if self._attente_quota_jusqu is not None:
            return None
        vu = self.avancement()
        if vu != self._avancement_vu:
            self._avancement_vu, self._avancement_t = vu, time.monotonic()
            return 0.0
        return time.monotonic() - self._avancement_t

    def entrer_attente(self, dep, type_err: str) -> None:
        """A trip starts (or continues) a transient wait — the clock starts at the 1st failure."""
        self._attentes.setdefault(
            (dep.person_id, dep.activity_id), (time.monotonic(), type_err)
        )

    def sortir_attente(self, dep) -> None:
        self._attentes.pop((dep.person_id, dep.activity_id), None)

    def entrer_attente_quota(self, reprise_a: str) -> None:
        self._attente_quota_jusqu = reprise_a

    def sortir_attente_quota(self) -> None:
        self._attente_quota_jusqu = None

    def _attente_la_plus_ancienne(self) -> dict | None:
        now = time.monotonic()
        candidates = [
            (now - t0, pid, aid, typ)
            for (pid, aid), (t0, typ) in self._attentes.items()
        ]
        if not candidates:
            return None
        depuis, pid, aid, typ = max(candidates)
        return {
            "person_id": pid,
            "activity_id": aid,
            "depuis_s": round(depuis, 1),
            "type": typ,
        }

    def etat(self) -> dict:
        faits = self.deja + self.traites()
        ecoule = time.monotonic() - self.debut
        debit = self.traites() / ecoule if ecoule > 0 else 0.0
        reste = (self.attendus - faits) / debit if debit > 0 else None
        en_attente = self._attente_la_plus_ancienne()
        return {
            "faits": faits,
            "attendus": self.attendus,
            "pourcent": round(100 * faits / self.attendus, 1)
            if self.attendus
            else None,
            "personnes_terminees": self.personnes_terminees,
            "personnes": self.personnes,
            "ecoule_s": round(ecoule, 1),
            "reste_s": (round(reste) if reste is not None else None),
            "sollicitations": int(self.compteurs["sollicitations"]),
            "attentes": int(sum(self.attentes_par_type.values())),
            "attentes_par_type": dict(self.attentes_par_type),
            "erreurs": int(sum(self.erreurs_par_type.values())),
            "erreurs_par_type": dict(self.erreurs_par_type),
            "en_attente_depuis_s": (en_attente["depuis_s"] if en_attente else 0.0),
            "en_attente": en_attente,
            "en_attente_quota_jusqu": self._attente_quota_jusqu,
            # Read by the dashboard: "idle for N min" before the watchdog
            # pauses. None during a quota window (intended idleness).
            "immobile_depuis_s": (
                round(immobile, 1)
                if (immobile := self.immobile_depuis()) is not None
                else None
            ),
            "maj": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def ecrire(self) -> None:
        if self.dossier is None:
            return
        try:
            tmp = self.dossier / (FICHIER_PROGRESSION + ".tmp")
            tmp.write_text(
                json.dumps(self.etat(), ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self.dossier / FICHIER_PROGRESSION)
        except OSError as e:
            logger.warning(f"[execution] progression.json not written: {e}")

    def traites(self) -> int:
        return (
            self.compteurs["decides"]
            + self.compteurs["non_couverts"]
            + self.compteurs["sans_solution"]
        )

    def ligne(self) -> str:
        faits = self.deja + self.traites()
        ecoule = time.monotonic() - self.debut
        debit = self.traites() / ecoule if ecoule > 0 else 0.0
        reste = (self.attendus - faits) / debit if debit > 0 else float("nan")
        return (
            f"[execution] {faits}/{self.attendus} déplacements ({100 * faits / max(1, self.attendus):.1f} %) · "
            f"{self.personnes_terminees}/{self.personnes} personnes · {ecoule:.0f} s écoulées · reste ≈ {reste:.0f} s · "
            f"sollicitations {self.compteurs['sollicitations']} · attentes {sum(self.attentes_par_type.values())} "
            f"{dict(self.attentes_par_type) if self.attentes_par_type else ''}"
            + (
                f" · ERREURS {sum(self.erreurs_par_type.values())} {dict(self.erreurs_par_type)}"
                if self.erreurs_par_type
                else ""
            )
        )

    async def boucle(self) -> None:
        while True:
            await asyncio.sleep(self.periode_s)
            logger.info(self.ligne())
            self.ecrire()


PAS_ATTENTE_S = (
    0.5  # granularity of an interruptible sleep: hands back quickly to PAUSE/STOP
)
PAUSE_BASE_S = (
    5.0  # pause of the 1st attempt; grows (× attempt) up to the ceiling (R1)
)
PAUSE_PLAFOND_S = 60.0  # ceiling of the growing pause between two attempts (R1)


async def _dormir_interruptible(
    duree_s: float, controle: Controle, epuise: asyncio.Event
) -> None:
    """Sleeps `duree_s`, but hands back as soon as an interruption (PAUSE/STOP/SIGINT) or an
    exhaustion is requested — this is what makes a trip's wait interruptible (R1)."""
    reste = duree_s
    while reste > 0:
        if controle.interrompu() or epuise.is_set():
            return
        await asyncio.sleep(min(PAS_ATTENTE_S, reste))
        reste -= PAS_ATTENTE_S


def _confirme_epuise(
    moniteur: MoniteurRessources | None, *, annonce_par_fournisseur: bool = False
) -> bool:
    """Is the exhaustion REAL?

    If a monitor is present, the experiment stops ONLY if all the instances serving
    this model are exhausted (e.g. failure on both Google keys). A 429 received on one key
    exhausts only that instance: as long as another instance remains available, the experiment
    switches to it and is not considered exhausted.

    Without a monitor, a 429/402 (or annonce_par_fournisseur) is authoritative.
    """
    if moniteur is None:
        return True
    moniteur.rafraichir()
    return moniteur.epuise()


def _instantane_requetes(moniteur) -> dict[str, int] | None:
    """{instance: requests of the day} as the gateway publishes it at this moment.

    Read at opening AND at closing, it gives by difference the number of requests
    ACTUALLY consumed by this run — the only honest measure of the batching factor
    (see `experiences/lots.py`). The counter alone, read at closing, aggregates everything the
    instance served that day: the other runs, the reflections, the retries.
    """
    if moniteur is None:
        return None
    etat = getattr(moniteur, "etat", None) or {}
    instantane = {}
    for i in getattr(moniteur, "instances", []) or []:
        v = (etat.get(i) or {}).get("daily_requests")
        if v is not None:
            instantane[str(i)] = int(v)
    return instantane or None


def _bilan_requetes(
    moniteur, ouverture: dict[str, int] | None, debut_iso: str | None
) -> dict:
    """Provider requests consumed by THIS run, or why it cannot be told.

    `fiable` is false as soon as there is a doubt: missing opening snapshot, instance
    that appeared along the way, negative delta (the daily counter was reset during
    the run — midnight in the provider's time zone). A figure without this flag would be worse than
    no figure: the estimate would use it as a measurement.
    """
    cloture = _instantane_requetes(moniteur)
    if not ouverture or not cloture:
        return {
            "ouverture": ouverture,
            "cloture": cloture,
            "delta": None,
            "fiable": False,
            "motif": "instantané manquant à l'ouverture ou à la clôture",
        }
    manquantes = [i for i in cloture if i not in ouverture]
    deltas = {i: cloture[i] - ouverture.get(i, 0) for i in cloture}
    negatifs = [i for i, d in deltas.items() if d < 0]
    fiable = not manquantes and not negatifs
    motif = None
    if negatifs:
        motif = (
            f"compteur journalier remis à zéro pendant le run sur {', '.join(sorted(negatifs))} "
            f"(fenêtre de quota traversée) : delta inexploitable"
        )
    elif manquantes:
        motif = f"instance(s) apparue(s) en cours de run : {', '.join(sorted(manquantes))}"
    bilan = {
        "ouverture": ouverture,
        "cloture": cloture,
        "par_instance": deltas,
        "delta": sum(d for d in deltas.values() if d >= 0),
        "fiable": fiable,
        "depuis": debut_iso,
    }
    if motif:
        bilan["motif"] = motif
    return bilan


async def executer(
    exp: Experience,
    jeu: Jeu,
    personnes: Sequence[Person],
    execution: Execution,
    decideur,
    *,
    moniteur: MoniteurRessources | None = None,
    controle: Controle | None = None,
    ecrire_moves: bool = True,
    # By default we wait for the window (2026-09-08): the reopening is now known and
    # bounded (time announced by the provider, or midnight in its time zone), so an
    # experiment launched in the evening finishes on its own instead of dying at 10 % on a daily
    # quota. The wait stays visible (state `en_attente_quota`, `en_attente_quota_jusqu` in
    # progression.json) and interruptible (PAUSE/STOP/SIGINT).
    attendre_fenetre: bool = True,
    periode_progression_s: float = 5.0,
) -> dict:
    """Runs (or resumes) the run; returns the counters (S8).

    No trip is ever skipped (R1): a transient error is retried indefinitely,
    growing pause capped at 60 s, interruptible by PAUSE/STOP/SIGINT; the trip stays
    un-archived hence resumed as is. `attente_max_s` no longer bounds the wait — it is an
    `[ALARME]` threshold (decision a). Only a CONFIRMED quota exhaustion stops (or, if `attendre_fenetre`,
    sleeps until the resume window — R4)."""
    debut = time.monotonic()
    controle = controle or Controle(execution.dossier)
    # A leftover PAUSE/STOP is purged BEFORE starting, not only at closing: a
    # runner killed outright (container stopped) leaves its sentinel on disk, and without this
    # cleanup the resume paused itself again within seconds — the run looked
    # unresumable. The interruption request only applies to the run that received it.
    residus = controle.nettoyer()
    if residus:
        logger.info(
            f"[execution] leftover interruption sentinel(s) purged before resume: "
            f"{', '.join(residus)} — they came from a previous run, not this one"
        )
    controle.installer_signaux()
    attendus = deplacements_attendus(personnes, jeu.jour_simule)
    par_personne: dict[str, list[Deplacement]] = {}
    for d in attendus:
        par_personne.setdefault(d.person_id, []).append(d)
    for lst in par_personne.values():
        lst.sort(key=lambda d: d.ordinal)
    deja = len(execution.decisions)
    reprise = deja > 0
    decalage = _decalage_calendrier(exp, jeu)
    # R9 (alert A3): the denominator is DERIVED FROM THE SET at opening, it does not accumulate
    # as decisions come. Without this, an interrupted run does not report the same number
    # of unusable trips as a complete run on the same (population, set) pair, and two
    # columns stop being comparable term by term.
    inexploitables_connus = jeu.inexploitables(attendus)
    attendus_exploitables_ouverture = len(attendus) - len(inexploitables_connus)
    logger.info(
        f"[execution] denominator fixed at opening: {attendus_exploitables_ouverture} "
        f"usable trips out of {len(attendus)} expected "
        f"({len(inexploitables_connus)} unusable derived from the set {jeu.nom!r})"
    )
    prog = Progression(
        attendus_exploitables_ouverture,
        len(par_personne),
        deja,
        periode_progression_s,
        dossier=execution.dossier,
    )
    prog.ecrire()
    moves = execution.journal_moves() if ecrire_moves else None

    if reprise:
        interruptions = execution.config.get("interruptions") or []
        derniere = interruptions[-1]["instant"] if interruptions else None
        # Run left `en_cours` = process killed without closing (failure, kill): we record it
        # before resuming, otherwise the abrupt stop leaves no trace (R3).
        if execution.etat().get("etat") == ETAT_EN_COURS:
            execution.ajouter_interruption("arret_force", depuis=derniere)
            logger.warning(
                f"[execution] {execution.nom} reopened while it was `en_cours` — "
                f"forced stop recorded (process interrupted without closing)"
            )
        execution.ajouter_interruption("reprise", depuis=derniere)
        if moves is not None:
            # Ticket 081 — the log is rebuilt BEFORE the first new decision.
            # Decisions served again do not go through line writing (they
            # request nothing): without this rebuild, `moves.csv` would stay as
            # the interrupted process left it, and the score would cover a
            # fraction of the work without anything saying so. Regenerating rather than completing
            # makes the log independent of the interruption history.
            try:
                await journal.regenerer(execution, jeu, personnes, decideur)
            except Exception as exc:  # noqa: BLE001
                # Fail-open: the resume must succeed even if the rebuild fails. The
                # log stays incomplete, but it will not be scored silently — the
                # safeguard of `score.calculer` refuses a truncated log (ticket 081, lot A).
                logger.error(
                    f"[ALARME] Log regeneration failed on resuming "
                    f"{execution.nom}: {type(exc).__name__}: {exc}; the run "
                    f"continues, but its `moves.csv` stays incomplete and its scoring "
                    f"will be refused at closing"
                )
    execution.changer_etat(ETAT_EN_COURS)
    # Gateway counters BEFORE the first request: without this starting point, only the
    # closing value remains — a daily counter, which also counts what the other
    # runs of the day consumed. This is what made the archives unusable for measuring
    # batching (observed ratios from 0.26 to 14 agents/request on the same template).
    requetes_ouverture = _instantane_requetes(moniteur)
    debut_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if requetes_ouverture:
        logger.info(
            f"[execution] requests already served today by the targeted instances: "
            f"{requetes_ouverture} — the delta at closing will give the real cost of this run"
        )
    execution.mettre_a_jour_regime(
        parallelisme=exp.regroupement.parallelisme,
        unite_sollicitation="deplacement",
        decideur=getattr(decideur, "nom", "?"),
        reprise=reprise,
    )
    logger.info(
        f"[execution] Start {execution.nom} ({exp.nom}): {len(attendus)} trips expected, "
        f"{len(par_personne)} persons, {deja} decisions already archived, decision-maker {getattr(decideur, 'nom', '?')}"
    )

    epuise = asyncio.Event()
    raison_epuisement: dict = {}
    sem = asyncio.Semaphore(max(1, exp.regroupement.parallelisme))
    quota_lock = (
        asyncio.Lock()
    )  # R4: a single sleeper waits for the window, the others re-check

    def _par_code(props: list[D.Proposition], code: str | None) -> D.Proposition | None:
        return next((p for p in props if p.code == code), None) if code else None

    async def _attendre_fenetre_quota(
        raison: str, reprise_annoncee: str | None = None
    ) -> None:
        """R4: sleeps in-process until the quota window reopens, then refreshes the
        monitor and restarts. Interruptible (PAUSE/STOP/SIGINT). A single coroutine sleeps at a time;
        the others re-test the exhaustion and restart at once if the quota has reopened. No
        substitution, no default decision: we WAIT (RG-2, Q3).

        `reprise_annoncee` is the time the PROVIDER gave (429 "per day"): it
        prevails over the local computation. Aiming at a hard-coded UTC midnight woke the run at 02:00
        Paris time for a Gemini quota that only reopens at 09:00 — seven hours of refusals.
        """
        async with quota_lock:
            if controle.interrompu() or not _confirme_epuise(
                moniteur, annonce_par_fournisseur=bool(reprise_annoncee)
            ):
                return
            reprise_a = reprise_annoncee or prochaine_fenetre_quota()
            execution.changer_etat(ETAT_EN_ATTENTE_QUOTA, raison, reprise_a)
            prog.entrer_attente_quota(reprise_a)
            logger.error(
                f"[ALARME] [execution] {execution.nom} en attente de la fenêtre quota "
                f"jusqu'à {reprise_a} — {raison}"
            )
            cible = datetime.fromisoformat(reprise_a)
            while datetime.now(timezone.utc) < cible:
                if controle.interrompu() or epuise.is_set():
                    break
                restant = (cible - datetime.now(timezone.utc)).total_seconds()
                await asyncio.sleep(min(PAS_ATTENTE_S * 10, max(0.0, restant)))
            prog.sortir_attente_quota()
            if moniteur is not None:
                moniteur.rafraichir()
            if not controle.interrompu() and not epuise.is_set():
                execution.changer_etat(ETAT_EN_COURS)
                logger.info(
                    f"[execution] {execution.nom} : fenêtre quota atteinte, reprise"
                )

    async def traiter_personne(personne: Person, deps: list[Deplacement]) -> None:
        async with sem:
            personne.state.planning_vehicle_at = {}  # the day starts at home (D4)
            for dep in deps:
                if controle.interrompu() or epuise.is_set():
                    return
                props = jeu.propositions(dep.person_id, dep.activity_id)
                archivee = execution.decision(dep.person_id, dep.activity_id)
                if archivee is not None:
                    prog.compteurs["resservies"] += 1
                    retenue = _par_code(
                        props or [], (archivee.get("retenue") or {}).get("code")
                    )
                    if retenue is not None:
                        D.avancer_chaine(
                            personne,
                            retenue.plan,
                            dep.origine,
                            dep.destination,
                            dep.purpose,
                        )
                    continue
                if not props:
                    # props is None : the set does not cover this trip (not covered, in the expected ones).
                    # props == []  : covered, but the engines have NO proposal — a trip
                    # unusable by any decision-maker whatsoever: excluded from the expected ones, reported
                    # as a warning (author's decision, 2026-09-06, question 3).
                    methode = (
                        METHODE_NON_COUVERT if props is None else METHODE_INEXPLOITABLE
                    )
                    trace = D.construire_trace(
                        personne,
                        D.ContexteDecision(
                            timestamp=dep.depart_ts + decalage,
                            activity_id=dep.activity_id,
                            purpose=dep.purpose,
                            departure_time=dep.depart_ts + decalage,
                            from_location=dep.origine,
                            destination=dep.destination,
                            graine_ordre=exp.graine_ordre,
                            graine_tirage=exp.graine_tirage,
                            max_candidats=exp.max_candidats,
                            jour_offre=jeu.jour_simule,
                        ),
                        [],
                        [],
                        None,
                        methode,
                        None,
                        "",
                    )
                    if props is not None:
                        ligne = jeu.ligne(dep.person_id, dep.activity_id)
                        trace["motif_absence"] = (
                            ligne.motif_absence if ligne is not None else None
                        )
                    execution.ajouter_decision(trace)
                    prog.compteurs[
                        "non_couverts" if props is None else "inexploitables"
                    ] += 1
                    continue
                ctx = D.ContexteDecision(
                    timestamp=dep.depart_ts + decalage,
                    activity_id=dep.activity_id,
                    purpose=dep.purpose,
                    departure_time=dep.depart_ts + decalage,
                    from_location=dep.origine,
                    destination=dep.destination,
                    anticipation=_anticipation(
                        personne, dep.activity_id, dep.depart_ts + decalage
                    ),
                    graine_ordre=exp.graine_ordre,
                    graine_tirage=exp.graine_tirage,
                    max_candidats=exp.max_candidats,
                    jour_offre=jeu.jour_simule,
                )
                # R1 — retry until a decision is obtained: NEVER a skip. The trip
                # is archived only once decided; interrupted, it stays un-archived hence resumed as is.
                decision: D.Decision | None = None
                attente_totale = 0.0
                tentative = 0
                alarme_levee = False
                while True:
                    if controle.interrompu() or epuise.is_set():
                        return  # safe point: nothing partial archived (Q7/Q8)
                    tentative += 1
                    # The request is a TRACKED task: past the grace delay, a
                    # pause/stop abandons it (`abandonner_en_vol`). The cancellation only affects
                    # the network call — no archive write is ever interrupted.
                    appel = asyncio.ensure_future(
                        D.decider(personne, ctx, props, decideur)
                    )
                    controle.suivre(appel)
                    try:
                        decision = await appel
                    except asyncio.CancelledError:
                        if controle.abandonnee(appel):
                            return  # safe point: the trip stays un-archived (Q7/Q8)
                        raise  # cancellation from elsewhere: it must propagate
                    finally:
                        controle.oublier(appel)
                    if decision.sollicite:
                        prog.compteurs["sollicitations"] += 1
                    if decision.est_decision:
                        break
                    erreur = str(decision.trace.get("erreur") or "")
                    type_err = erreur.split(":")[0].strip() or "erreur"
                    # Retried attempt: a WAIT, not an error. The trip is
                    # not closed, it will go through this loop again until it gets its decision (R1).
                    prog.attentes_par_type[type_err] += 1
                    execution.ajouter_erreur(
                        {
                            "horodatage": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                            "person_id": dep.person_id,
                            "activity_id": dep.activity_id,
                            "tentative": tentative,
                            "type": type_err,
                            "message": erreur[:500],
                            "fournisseur": (
                                decision.reponse.fournisseur if decision.reponse else ""
                            )
                            or "",
                            "attente_s": round(attente_totale, 1),
                        }
                    )
                    # CONFIRMED quota exhaustion: either the provider announced it (429
                    # "per day", which carries its resume time), or 429/402 + monitor exhausted.
                    reprise_annoncee = (
                        decision.reponse.reprise_a if decision.reponse else None
                    )
                    if type_err == "epuise":
                        if _confirme_epuise(
                            moniteur, annonce_par_fournisseur=bool(reprise_annoncee)
                        ):
                            if attendre_fenetre:
                                await _attendre_fenetre_quota(
                                    erreur, reprise_annoncee
                                )  # R4: sleeps until the window, then retries
                                continue
                            raison_epuisement.update(
                                {
                                    "raison": erreur,
                                    "reprise": reprise_annoncee
                                    or prochaine_fenetre_quota(),
                                }
                            )
                            epuise.set()
                            return
                        # Partial exhaustion (only one key failed, another remains available):
                        # switch immediately to the next available instance without waiting.
                        continue
                    if type_err == "substitution_refusee":
                        continue  # another instance, right away (Q3)
                    # Transient error (busy gateway, timeout, unconfirmed 429…): we WAIT,
                    # growing capped pause (R1); `attente_max_s` no longer bounds, it raises an ALARM.
                    pause = min(PAUSE_BASE_S * tentative, PAUSE_PLAFOND_S)
                    attente_totale += pause
                    prog.entrer_attente(dep, type_err)
                    if attente_totale > exp.attente_max_s and not alarme_levee:
                        logger.error(
                            f"[ALARME] [execution] attente {attente_totale:.0f}s > seuil "
                            f"{exp.attente_max_s}s pour {dep.person_id}/{dep.activity_id} "
                            f"({type_err}) — on continue d'attendre, aucun déplacement n'est sauté"
                        )
                        alarme_levee = True
                    await _dormir_interruptible(pause, controle, epuise)
                prog.sortir_attente(dep)
                execution.ajouter_decision(decision.trace)
                prog.compteurs["decides"] += 1
                prog.compteurs[decision.methode] += 1
                if decision.retenue is not None:
                    D.avancer_chaine(
                        personne,
                        decision.retenue.plan,
                        dep.origine,
                        dep.destination,
                        dep.purpose,
                    )
                if moves is not None:
                    # Ticket 081 — ONE single function writes a log line, here as
                    # in the regeneration on resume. The trace just archived is enough
                    # to produce it: this is what guarantees that a log rebuilt from
                    # the archive is identical to the one the live path would have written.
                    await journal.ecrire_ligne(
                        moves,
                        personne=personne,
                        trace=decision.trace,
                        props=props,
                        decideur=decideur,
                    )
            prog.personnes_terminees += 1

    async def surveiller() -> None:
        """The only place that watches interruption requests and inactivity.

        Two duties, every `PAS_ATTENTE_S`:

        1. **watchdog** — beyond `INACTIVITE_PAUSE_S` without a single trip
           settled, the run pauses itself and says so at ERROR, rather than
           staying "in progress" indefinitely. It remains resumable;
        2. **grace** — once the request is made and `controle.grace_s` has elapsed, the
           requests still in flight are abandoned: the pause takes effect within
           a few seconds instead of waiting for a call to return (up to 120 s).
        """
        silence_annonce = False
        while True:
            await asyncio.sleep(PAS_ATTENTE_S)
            demande = controle.interrompu()
            if not demande and INACTIVITE_PAUSE_S > 0:
                immobile = prog.immobile_depuis()
                if immobile is not None and immobile >= INACTIVITE_PAUSE_S:
                    attente = prog._attente_la_plus_ancienne()
                    detail = (
                        f"plus vieille attente {attente['person_id']}/{attente['activity_id']} "
                        f"({attente['type']}) depuis {attente['depuis_s']:.0f} s"
                        if attente
                        else "aucune attente déclarée — décideur muet ou bloqué"
                    )
                    logger.error(
                        f"[ALARME] [execution] {execution.nom} : {immobile:.0f} s sans avancée "
                        f"(seuil {INACTIVITE_PAUSE_S:.0f} s) — mise en pause automatique ; "
                        f"{prog.deja + prog.traites()}/{prog.attendus} archivées, {detail}"
                    )
                    controle.raison = f"inactivite:{int(immobile)}s"
                    controle.pause = True
                    demande = True
            if demande and controle.grace_echue():
                nb = controle.abandonner_en_vol()
                if nb:
                    logger.warning(
                        f"[execution] {nb} in-flight request(s) abandoned after "
                        f"{controle.grace_s:.0f} s of grace — trips not archived, "
                        f"requested again as is on resume"
                    )
                elif not silence_annonce:
                    silence_annonce = True
                    logger.info(
                        f"[execution] {execution.nom}: interruption taken, no "
                        f"in-flight request to abandon"
                    )

    async def veiller_quotas() -> None:
        """Keeps the quota snapshot fresh, so that the decision-maker stops pinning an
        exhausted key and starts on the next one (incident of 2026-09-08).

        A task separate from `surveiller`: that one runs every 0.5 s for the pause
        grace and the watchdog, whereas a read of `/health` costs up to 5 s. The
        `to_thread` is essential — `lire_etat_passerelle` is a blocking `httpx.get`,
        which would freeze the event loop, and the pause and the watchdog with it.

        Fail-open end to end: an unreachable gateway leaves the previous snapshot
        in place and must never interrupt the run, nor this watch.
        """
        if moniteur is None or RAFRAICHIR_QUOTAS_S <= 0:
            return
        from experiences.decideurs import rang_en_mots

        precedentes = list(moniteur.instances_disponibles())
        alarme_epuisement = False
        while True:
            await asyncio.sleep(min(RAFRAICHIR_QUOTAS_S, 5.0))
            try:
                lu = await asyncio.to_thread(
                    moniteur.rafraichir_si_perime, RAFRAICHIR_QUOTAS_S
                )
            except Exception as e:  # noqa: BLE001 — the watch never brings down a run
                logger.warning(
                    f"[execution] lecture des quotas impossible ({type(e).__name__}: {e}) — "
                    f"instantané précédent conservé, nouvelle tentative dans "
                    f"{RAFRAICHIR_QUOTAS_S:.0f} s"
                )
                continue
            if not lu:
                continue
            courantes = list(moniteur.instances_disponibles())
            if courantes == precedentes:
                continue  # nothing to say: only changes are logged
            if courantes:
                # The RANK in the declared order, never the name: it designates an account.
                rang = moniteur.instances.index(courantes[0]) + 1
                logger.warning(
                    f"[execution] {execution.nom} : quotas relus — passage sur la "
                    f"{rang_en_mots(rang)} clé ({rang}/{len(moniteur.instances)}), "
                    f"{len(courantes)} clé(s) encore disponible(s) ; la précédente a épuisé "
                    f"son quota du jour"
                )
                alarme_epuisement = False
            elif not alarme_epuisement:
                # Rising edge: a single alarm, not one per watch round.
                alarme_epuisement = True
                logger.error(
                    f"[ALARME] [execution] {execution.nom}: no key available any more — "
                    f"{moniteur.raison_epuisement()}"
                )
            precedentes = courantes

    suivi = asyncio.create_task(prog.boucle())
    garde = asyncio.create_task(surveiller())
    quotas = asyncio.create_task(veiller_quotas())
    try:
        await asyncio.gather(
            *(
                traiter_personne(p, par_personne[p.person_id])
                for p in personnes
                if p.person_id in par_personne
            )
        )
    finally:
        suivi.cancel()
        garde.cancel()
        quotas.cancel()
        prog.ecrire()

    # ── summary ──
    duree = time.monotonic() - debut
    hors_decision = (
        METHODE_NON_COUVERT,
        METHODE_INEXPLOITABLE,
        D.METHODE_SANS_SOLUTION,
    )
    decides_total = sum(
        1 for t in execution.decisions if t.get("methode") not in hors_decision
    )
    non_couverts_total = sum(
        1 for t in execution.decisions if t.get("methode") == METHODE_NON_COUVERT
    )
    inexploitables = [
        t for t in execution.decisions if t.get("methode") == METHODE_INEXPLOITABLE
    ]
    sans_solution_total = sum(
        1 for t in execution.decisions if t.get("methode") == D.METHODE_SANS_SOLUTION
    )
    attendus_exploitables = len(attendus) - len(inexploitables)
    # The denominator fixed at opening is authoritative (R9). If it diverges from what the run
    # observed, the set changed under the run's feet, or a trip was
    # classified unusable for a reason the set did not carry: in both cases the
    # measurement is not the one we think, and that must show.
    if attendus_exploitables != attendus_exploitables_ouverture:
        logger.error(
            f"[ALARME] inconsistent denominator: {attendus_exploitables_ouverture} usable "
            f"derived from the set at opening, {attendus_exploitables} observed at closing "
            f"({len(inexploitables)} unusable archived versus {len(inexploitables_connus)} "
            f"expected). The published coverage keeps the opening denominator."
        )
    attendus_exploitables = attendus_exploitables_ouverture
    if inexploitables:
        motifs = Counter(str(t.get("motif_absence")) for t in inexploitables)
        logger.warning(
            f"[execution] {len(inexploitables)} déplacement(s) INEXPLOITABLES exclus des attendus — aucune proposition des "
            f"moteurs ({', '.join(f'{m} × {n}' for m, n in motifs.most_common())}) ; "
            f"personnes : {', '.join(sorted({str(t.get('person_id')) for t in inexploitables})[:10])}"
            f"{'…' if len({t.get('person_id') for t in inexploitables}) > 10 else ''}"
        )
    dates_meteo = [
        t.get("jour_meteo_lu")
        for t in execution.decisions
        if t.get("jour_meteo_lu")
    ]
    repartition_mois = Counter(
        int(d.split("-")[1])
        for d in dates_meteo
        if len(d.split("-")) >= 2 and d.split("-")[1].isdigit()
    )
    c_dates = Counter(dates_meteo)
    nb_distinctes = len(c_dates)
    total_meteo = len(dates_meteo)
    if total_meteo > 0:
        date_max, max_count = c_dates.most_common(1)[0]
        pct_max = (max_count / total_meteo) * 100
        if max_count > total_meteo / 2:
            logger.error(
                f"[ALARME] a single weather date ({date_max}) covers {max_count}/{total_meteo} decisions "
                f"({pct_max:.1f} %) — sign that the draw is not rotating"
            )
        else:
            logger.info(
                f"[execution] weather: {nb_distinctes} distinct dates served over {total_meteo} decisions "
                f"· distribution by month: {dict(sorted(repartition_mois.items()))}"
            )
        compte_meteo = {
            "dates_distinctes": nb_distinctes,
            "repartition_par_mois": dict(sorted(repartition_mois.items())),
            "date_max_frequence": date_max,
            "part_date_max": round(pct_max / 100, 3),
        }
    else:
        compte_meteo = {
            "dates_distinctes": 0,
            "repartition_par_mois": {},
            "date_max_frequence": None,
            "part_date_max": 0.0,
        }

    compteurs = {
        "attendus": len(attendus),
        "inexploitables": len(inexploitables),
        "attendus_exploitables": attendus_exploitables,
        "decides": decides_total,
        "non_couverts": non_couverts_total,
        "sans_solution": sans_solution_total,
        "choix_unique": sum(
            1 for t in execution.decisions if t.get("methode") == D.METHODE_CHOIX_UNIQUE
        ),
        "replis_uniformes": sum(
            1
            for t in execution.decisions
            if t.get("methode") == D.METHODE_REPLI_UNIFORME
        ),
        "attentes": int(sum(prog.attentes_par_type.values())),
        "attentes_par_type": dict(prog.attentes_par_type),
        "erreurs": int(sum(prog.erreurs_par_type.values())),
        "erreurs_par_type": dict(prog.erreurs_par_type),
        "sollicitations": int(prog.compteurs["sollicitations"]),
        "resservies": int(prog.compteurs["resservies"]),
        "substitution_refusee": int(moniteur.compteurs["substitution_refusee"])
        if moniteur
        else 0,
        "couverture": {
            "decides": decides_total,
            "attendus": attendus_exploitables,
            "attendus_bruts": len(attendus),
            "inexploitables_exclus": len(inexploitables),
            "taux": (decides_total / attendus_exploitables)
            if attendus_exploitables
            else None,
        },
        "duree_s": round(duree, 3),
        "quota": (
            {"sans_quota": True}
            if getattr(decideur, "sans_quota", False)
            else (moniteur.tableau() if moniteur else {})
        ),
        # Provider requests of THIS run — the other unit. `sollicitations` counts
        # trips; these two numbers are not the same and their ratio IS the batching
        # factor that the estimate of a next arm will reuse.
        "requetes": (
            {"sans_quota": True}
            if getattr(decideur, "sans_quota", False)
            else _bilan_requetes(moniteur, requetes_ouverture, debut_iso)
        ),
        "meteo": compte_meteo,
    }
    execution.ecrire_compteurs(compteurs)
    nb_archivees = len(execution.decisions)
    complet = nb_archivees >= len(attendus)

    if epuise.is_set():
        # Exhausted = resumable: no closing, the archive stays open for the resume (Q4, Q5).
        execution.ajouter_interruption(
            "epuisement", raison=raison_epuisement.get("raison")
        )
        execution.changer_etat(
            ETAT_EPUISEE,
            raison_epuisement.get("raison"),
            raison_epuisement.get("reprise"),
        )
        logger.error(
            f"[ALARME] [execution] {execution.nom} épuisée — {raison_epuisement.get('raison')} ; "
            f"reprise possible à {raison_epuisement.get('reprise')} ; {nb_archivees}/{len(attendus)} archivées"
        )
        etat = ETAT_EPUISEE
    elif controle.demande_arret():
        execution.ajouter_interruption("arret", raison=controle.raison or "manuelle")
        execution.cloturer(
            ETAT_ARRETEE, f"arrêt demandé — {nb_archivees}/{len(attendus)} archivées"
        )
        etat = ETAT_ARRETEE
    elif controle.demande_pause() and not complet:
        # `raison` tells the intended pause from the one decided by the watchdog: without
        # it, a run found "paused" does not say whether someone clicked or whether
        # it stopped advancing.
        raison = controle.raison or "manuelle"
        execution.ajouter_interruption("pause", raison=raison)
        auto = raison.startswith("inactivite:")
        execution.changer_etat(
            ETAT_EN_PAUSE,
            (
                f"pause automatique — {raison.split(':', 1)[1]} sans avancée ; "
                f"{nb_archivees}/{len(attendus)} archivées"
                if auto
                else f"pause — {nb_archivees}/{len(attendus)} archivées"
            ),
        )
        etat = ETAT_EN_PAUSE
    elif complet:
        # No trip is skipped any more (R1): complete ⇒ finished. `erreurs` only counts
        # failed attempts (all followed by a decision), never gaps any more.
        execution.cloturer(ETAT_TERMINEE)
        etat = ETAT_TERMINEE
    else:
        execution.changer_etat(
            ETAT_EN_PAUSE,
            f"incomplète : {nb_archivees}/{len(attendus)} archivées — reprendre",
        )
        etat = ETAT_EN_PAUSE
    controle.nettoyer()
    execution.fermer()
    try:
        from experiences.registre import ecrire_synthese

        ecrire_synthese(execution.dossier)
    except Exception as e:  # noqa: BLE001 — the summary is a rendering, its failure loses no decision
        logger.warning(f"[execution] summary not written for {execution.nom}: {e}")
    if etat == ETAT_TERMINEE:
        # R23 — a closed run carries its composite score without being asked. The computation
        # is offline (no LLM or network call) and fail-open: `scorer_a_la_cloture` does not
        # raise, a failed scoring leaves the run finished without `scores.json`.
        from experiences import score as _score

        _score.scorer_a_la_cloture(execution.dossier)
    niveau = logger.info if etat == ETAT_TERMINEE else logger.warning
    niveau(
        f"[execution] {'Exécution terminée' if etat == ETAT_TERMINEE else 'Exécution ' + etat} {execution.nom} en {duree:.1f} s — "
        f"décidés {compteurs['decides']}, non couverts {compteurs['non_couverts']}, inexploitables exclus {compteurs['inexploitables']}, "
        f"sans solution {compteurs['sans_solution']}, choix unique {compteurs['choix_unique']}, replis {compteurs['replis_uniformes']}, "
        f"attentes {compteurs['attentes']} · erreurs {compteurs['erreurs']} · "
        f"sollicitations {compteurs['sollicitations']}, resservies {compteurs['resservies']}"
    )
    # The measured batching, stated explicitly even when all is well: it is the figure that
    # the estimate of the next arm will reuse, and a log silent on it does not allow telling
    # "the gateway merged" from "the reading did not take place".
    bilan = compteurs.get("requetes") or {}
    if bilan.get("fiable") and bilan.get("delta"):
        logger.info(
            f"[execution] measured batching: {compteurs['sollicitations']} solicitation(s) "
            f"served in {bilan['delta']} provider request(s), i.e. "
            f"{compteurs['sollicitations'] / bilan['delta']:.2f} agent(s)/request "
            f"({bilan.get('par_instance')})"
        )
    elif not bilan.get("sans_quota"):
        logger.warning(
            f"[execution] batching NOT measured on this run: "
            f"{bilan.get('motif') or 'relevé indisponible'} — the estimate of the next arms "
            f"will fall back on the cautious assumption of one request per trip"
        )
    compteurs["etat"] = etat
    return compteurs


__all__ = [
    "DELAI_GRACE_PAUSE_S",
    "FICHIER_PAUSE",
    "FICHIER_PROGRESSION",
    "FICHIER_STOP",
    "INACTIVITE_PAUSE_S",
    "Controle",
    "Progression",
    "executer",
]
