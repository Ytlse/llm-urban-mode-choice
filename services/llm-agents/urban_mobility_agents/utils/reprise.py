"""Hot resume of a long run — ticket 075.

THE PROBLEM, IN ONE SENTENCE
----------------------------
`make run OFFLINE=1 CONT=1` reuses the run's directory — so **finds the memory full** — while
GAMA restarts at its t0 and **replays** the days already lived. On an ordinary run, no
consequence. On a run whose memory IS the object, the replayed days would rewrite memories
already written: duplicated episodes, observation and counter-example counters incremented
twice, concepts set aside on contradictions counted twice. The memory of day 40 would no
longer be the one a continuous run would have produced.

THE ANSWER, IN TWO PIECES
-------------------------
1. **One resume point per simulated day**, written at 3 a.m. — after the nightly draining of
   reflections and after the 10 p.m. consolidation floor, when the short-term memory buffers
   are empty and almost nobody is travelling. It is the only instant of the day when a
   snapshot is consistent without freezing machinery.
2. **Replay with frozen memory**: on restart, the state is restored to the last point, then
   everything that WRITES memory is suspended until the simulated clock passes that point.
   Agents move, decide, but learn nothing of what they have already learnt.

WHAT IS FROZEN, AND WHAT IS NOT
-------------------------------
Frozen: short-term memory writing (hence, mechanically, any consolidation, since an empty
buffer makes no agent eligible), long-term self-reflection, and the memory log.
Not frozen: decisions, itineraries, trips — they must take place for the simulation to get
back to its state. The decision cache serves them without calling the model.

⚠ **Writing a point is ATOMIC**: temporary directory, then rename. A point interrupted while
being written is never considered valid — a half-written point would be worse than no point
at all, because it would be restored silently.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger
from sim_clock import wall_clock
from urban_mobility_agents.utils import identite_run
from urban_mobility_agents.utils.ancre_run import ancre, ancrer

# Name of the resume points directory, in the run's workdir.
POINTS = "checkpoints_memoire"
# Description file of a point. Its presence makes the point valid: it is written last,
# after the data has been copied.
DESCRIPTION = "reprise.json"

# What is copied into a point, relative to the workdir. Long-term memory (vector index and
# per-agent metadata) and the memory log.
_A_COPIER = ("long_term_memory", "memoires")

# Freeze state, at process level. The controller sets it on restore and lifts it when the
# simulated clock passes the point; the agent consults it on the write path.
_gel_jusqu_a: int | None = None
_gel_jour: int | None = None
_degel_signale = False


# ── Freeze ───────────────────────────────────────────────────────────────────────────────


def geler_jusqu_a(timestamp_simule: int, jour: int) -> None:
    global _gel_jusqu_a, _gel_jour, _degel_signale
    _gel_jusqu_a = int(timestamp_simule)
    _gel_jour = int(jour)
    _degel_signale = False
    logger.warning(
        f"[reprise] MEMORY FROZEN until {wall_clock(_gel_jusqu_a)} (simulated day {jour}): "
        f"the days already lived are replayed without being relearnt. Decisions still happen."
    )


def gel_actif() -> bool:
    """Is the replay in progress? Consulted on the memory write path."""
    return _gel_jusqu_a is not None


def point_de_reprise_timestamp() -> int | None:
    """The simulated instant up to which the replay runs, or None outside a resume.

    Ticket 090: bounds the decision trace. Beyond this instant the run is live again,
    and serving a decision again there would be a cache, not a replay.
    """
    return _gel_jusqu_a


def degeler_si_depasse(timestamp_simule: int) -> bool:
    """Lifts the freeze as soon as the simulated clock passes the resume point. Returns `True` on thaw."""
    global _gel_jusqu_a, _gel_jour, _degel_signale
    if _gel_jusqu_a is None or int(timestamp_simule) < _gel_jusqu_a:
        return False
    logger.info(
        f"[reprise] THAW at {wall_clock(int(timestamp_simule))} — the replay of day "
        f"{_gel_jour} is over, memory learns again from here."
    )
    _gel_jusqu_a = None
    _gel_jour = None
    _degel_signale = True
    return True


def reinitialiser() -> None:
    """Forgets the freeze state. Reserved for tests."""
    global _gel_jusqu_a, _gel_jour, _degel_signale
    _gel_jusqu_a = None
    _gel_jour = None
    _degel_signale = False


# ── Resume points ────────────────────────────────────────────────────────────────────────


def _repertoire_points(workdir: Path) -> Path:
    return Path(workdir) / POINTS


def ecrire_point(
    workdir: Path,
    jour: int,
    timestamp_simule: int,
    *,
    compteurs: dict | None = None,
    foyer: dict | None = None,
) -> Path | None:
    """Writes a complete resume point. Returns its path, or `None` if nothing could be written.

    A write error does not propagate: losing a resume point costs at worst the replay of one
    day, bringing down the run costs all sixty.
    """
    workdir = Path(workdir)
    cible = _repertoire_points(workdir) / f"jour_{int(jour):03d}"
    provisoire = cible.with_suffix(".en_cours")
    try:
        if provisoire.exists():
            shutil.rmtree(provisoire)
        provisoire.mkdir(parents=True)
        for nom in _A_COPIER:
            source = workdir / nom
            if source.is_dir():
                shutil.copytree(source, provisoire / nom, dirs_exist_ok=True)
        # The description is written LAST: it is what makes the point valid.
        (provisoire / DESCRIPTION).write_text(
            json.dumps(
                {
                    "jour_simule": int(jour),
                    "timestamp_simule": int(timestamp_simule),
                    "horodatage_simule": wall_clock(int(timestamp_simule)).isoformat(),
                    "ancre_run": ancre(),
                    "ecrit_le": datetime.now().astimezone().isoformat(),
                    "compteurs": compteurs or {},
                    # Ticket 100, lot 4 — the household reading marker, per receiver, and the
                    # beliefs already shown. Without it, a hot resume would make the whole
                    # household HEAR AGAIN several nights already heard: the defect this
                    # ticket fixed for memory, not to be reintroduced through the
                    # household door.
                    "foyer": foyer or {},
                    # Ticket 091 — the run identity travels WITH the point. Without it, a
                    # restored point does not say which experiment it comes from, and nothing
                    # prevents giving one run the memory of another.
                    "identite": identite_run.lire(workdir) or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if cible.exists():
            shutil.rmtree(cible)
        os.replace(provisoire, cible)
        logger.info(
            f"[reprise] resume point written: {cible} (simulated day {jour}, "
            f"{wall_clock(int(timestamp_simule))})"
        )
        return cible
    except OSError as err:
        logger.error(
            f"[ALARME] [reprise] resume point of day {jour} NOT written ({err}) — an "
            f"interruption would restart from the previous point, or from the start if none."
        )
        shutil.rmtree(provisoire, ignore_errors=True)
        return None


def dernier_point(workdir: Path) -> tuple[Path, dict] | None:
    """The most recent valid point, or `None`. A point without a description is not valid."""
    repertoire = _repertoire_points(Path(workdir))
    if not repertoire.is_dir():
        return None
    points: list[tuple[int, Path, dict]] = []
    for chemin in repertoire.iterdir():
        description = chemin / DESCRIPTION
        if not description.is_file():
            continue
        try:
            meta = json.loads(description.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            logger.warning(f"[reprise] unreadable point ignored: {chemin} ({err})")
            continue
        points.append((int(meta.get("jour_simule", 0)), chemin, meta))
    if not points:
        return None
    _, chemin, meta = max(points, key=lambda t: t[0])
    return chemin, meta


# Per-trip outputs that a replay REWRITES without warning. These are not logs: they are
# the measurements. A replayed line there is indistinguishable from an original one.
_SORTIES_PAR_TRAJET = ("moves.csv",)


def ecarter_les_sorties_du_rejeu(workdir: Path) -> list[Path]:
    """Sets aside the per-trip outputs before a replay — ticket 077, lot E6.

    A resume without a valid resume point makes GAMA restart from t0: the first days are
    replayed and RE-DECIDED. On the ticket 075 run, **84 trips** from 16 to 21 March thus
    ended up a second time in `moves.csv`, with no column telling them apart from the
    originals. Any modal share computed on that file counts those days twice.

    The file is not deleted — it carries measurements — but timestamped and set aside, in the
    manner of `archive_log.py`. The replay then writes into a new file, and the two series
    stay readable separately.
    """
    workdir = Path(workdir)
    ecartes: list[Path] = []
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    for nom in _SORTIES_PAR_TRAJET:
        source = workdir / nom
        if not source.is_file():
            continue
        destination = source.with_name(f"{source.name}.{horodatage}.avant_rejeu")
        try:
            os.rename(source, destination)
            ecartes.append(destination)
            logger.warning(
                f"[reprise] {nom} set aside as {destination.name} before the replay — "
                f"the re-decided trips would have been indistinguishable from the originals"
            )
        except OSError as err:
            logger.error(
                f"[ALARME] [reprise] {nom} could NOT be set aside ({err}) — the replay "
                f"will add duplicated trips to it, and any modal share computed on it "
                f"will count the replayed days twice"
            )
    return ecartes


def restaurer_si_demande(
    workdir: Path, *, reprise_demandee: bool, alarme_si_absent: bool = False
) -> dict | None:
    """Restores the last point and sets the freeze. To be called BEFORE any opening of memory.

    The order is not negotiable: the vector index is opened when the agent is built, and
    replacing its files under it would give an open database on bytes that are no longer there.

    ⚠ **The right moment is GAMA's `/init`, not the controller's startup.** Defect found while
    testing the resume on 2026-09-14: `make run OFFLINE=1 CONT=1` restarts ONLY GAMA — the
    `controller` container keeps running, its `startup_event` is not replayed, and the
    restore therefore never happened. What restarts, in a resume, is the SIMULATION; and a
    simulation that restarts sends an `/init`.

    `alarme_si_absent` distinguishes the two callers: an explicitly requested resume
    (`CONTINUE_RUN`) without a resume point deserves an alarm; an `/init` of a new run does not.
    """
    if not reprise_demandee:
        return None
    trouve = dernier_point(workdir)
    if trouve is None:
        if alarme_si_absent:
            logger.error(
                "[ALARME] [reprise] resume requested but NO valid resume point in "
                f"{_repertoire_points(Path(workdir))} — the memory already present would be "
                "rewritten by the replay. The run therefore restarts without freeze: check that "
                "this is intended."
            )
            # Ticket 077, lot E6 — the replay will re-decide the first days. Its trips must
            # not be added to the originals in the same measurement file.
            ecarter_les_sorties_du_rejeu(Path(workdir))
        else:
            logger.info(
                f"[reprise] no resume point in {_repertoire_points(Path(workdir))}: "
                f"new run, nothing to restore."
            )
        return None
    chemin, meta = trouve
    workdir = Path(workdir)
    try:
        for nom in _A_COPIER:
            source = chemin / nom
            if not source.is_dir():
                continue
            destination = workdir / nom
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
    except OSError as err:
        logger.error(
            f"[ALARME] [reprise] restore IMPOSSIBLE from {chemin} ({err}) — the run "
            f"restarts on the current state, which may be later than the point."
        )
        return None

    # Ticket 105 — BOTH BRANCHES set aside the per-trip outputs, not only the one without a
    # point. A valid point exempts from nothing: GAMA still restarts from its t0 and replays the
    # days already lived, and `move_logger` has NO knowledge of the freeze — replayed trips
    # were therefore appended to `moves.csv`, indistinguishable from the originals. This is
    # exactly the defect that had duplicated 84 trips on the ticket 075 run; the function
    # written to prevent it was only called when no point was found.
    ecarter_les_sorties_du_rejeu(workdir)

    if meta.get("ancre_run"):
        # The anchor BEFORE anything: it must be set before the replay makes its first
        # timestamp observed, otherwise every agent's weather progression rewinds.
        ancrer(int(meta["ancre_run"]), origine="point de reprise")
    geler_jusqu_a(int(meta["timestamp_simule"]), int(meta.get("jour_simule", 0)))
    logger.info(
        f"[reprise] state restored from {chemin} — simulated day {meta.get('jour_simule')}, "
        f"{meta.get('horodatage_simule')}"
    )
    return meta
