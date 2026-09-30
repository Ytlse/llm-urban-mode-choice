"""Queue scheduler (ticket 035, spec parallelisation_experiences, R2b/R7/R12).

Runs on the **host** side (like the dashboard), because it relaunches experiments via
`docker compose exec`. Each round:

1. **reconciles** the ghosts INSIDE the container (`reconcilier` tests whether pids are alive,
   which only makes sense in the namespace where the runs live);
2. **promotes** in FIFO order the queued experiments whose keys are now all free
   (`reservations.promouvoir_pretes`), without ever preempting an active run (R2c);
3. **relaunches** each promoted one, in the background, via `docker compose exec … lancer` —
   which will reserve its keys at startup and replay the launch checks (R2d).

The logic is isolated in `tour()` with its effects injected (`reconcilieur`, `lanceur`), so as
to be testable without Docker.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from experiences import reservations as R
from experiences.chemins import racine_depot

INTERVALLE_S = 5.0

_RACINE_DEPOT = racine_depot()

# The compose file lives in infra/ (ticket 039): it has to be named, and
# `--project-directory` keeps the repository root as the base of the relative paths
# it contains. Without this second flag, compose re-anchors everything on infra/.
_EXEC_BASE = [
    "docker",
    "compose",
    "-f",
    str(_RACINE_DEPOT / "infra" / "docker-compose.yml"),
    "--project-directory",
    str(_RACINE_DEPOT),
    # Call base inside the container, aligned with the Makefile's `EXPERIENCES_PY`.
    "exec",
    "-T",
    "-e",
    "EXPERIENCES_DIR=/app/data/experiences",
    "-e",
    "JEUX_DIR=/app/data/jeux",
    "-e",
    "REFERENTIEL_ENQUETE=/app/scripts/data/population/cerema_values.yaml",
    "controller",
    "python",
    "-m",
    "experiences",
]


def _racine_depot() -> Path:
    return _RACINE_DEPOT


def _argv_lancer(entree: dict) -> list[str]:
    args = entree.get("args") or {}
    argv = [*_EXEC_BASE, "lancer", "--experience", entree["exp"]]
    if args.get("reprendre"):
        argv.append("--reprendre")
    if args.get("accepter_perime"):
        argv.append("--accepter-perime")
    # Passed in both directions: the parser's default is "do not wait", whereas this
    # comment claimed the opposite until 2026-09-29 — and a request that wanted to
    # wait therefore lost that choice on promotion. Absent: the parser's default.
    if args.get("attendre_fenetre") is True:
        argv.append("--attendre-fenetre")
    elif args.get("attendre_fenetre") is False:
        argv.append("--ne-pas-attendre-fenetre")
    return argv


def _reconcilier_conteneur() -> None:
    """Reconciles the ghosts inside the container. Fail-open: a failure (container down) does not
    block the round; the promotions will simply not get the freshly released keys."""
    try:
        subprocess.run(
            [*_EXEC_BASE, "reconcilier"],
            cwd=str(_racine_depot()),
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"[ordonnanceur] container reconciliation impossible: {e}")


def _lancer_detache(entree: dict) -> None:
    """Relaunches the promoted experiment in the background (does not block the loop).

    Its output goes to the experiment's launch log, like that of a campaign
    launch. It used to go to /dev/null: the go123 relaunch promoted on the night of 29/09
    left no trace of what it had done nor why.
    """
    from experiences.campagne import journal_lancement

    argv = _argv_lancer(entree)
    sortie = journal_lancement(entree["exp"])
    logger.info(f"[ordonnanceur] relaunch {entree['exp']}: {' '.join(argv)} — output in {sortie}")
    with sortie.open("w", encoding="utf-8") as flux:
        subprocess.Popen(
            argv,
            cwd=str(_racine_depot()),
            stdout=flux,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"},
        )


def tour(
    reconcilieur: Callable[[], None] = _reconcilier_conteneur,
    lanceur: Callable[[dict], None] = _lancer_detache,
) -> list[str]:
    """One round: reconcile, promote FIFO, relaunch. Returns the promoted experiments."""
    reconcilieur()
    promues = R.promouvoir_pretes()
    for entree in promues:
        lanceur(entree)
    return [e["exp"] for e in promues]


def boucle(intervalle_s: float = INTERVALLE_S) -> int:
    """Loops until interrupted. Logs the start, each promotion, and success (silence = alive)."""
    logger.info(f"[ordonnanceur] started — one round every {intervalle_s:.0f} s")
    tours = 0
    try:
        while True:
            promues = tour()
            tours += 1
            if promues:
                logger.info(f"[ordonnanceur] promoted: {promues}")
            elif tours % 60 == 0:
                logger.info(f"[ordonnanceur] alive — {tours} rounds, queue processed")
            time.sleep(intervalle_s)
    except KeyboardInterrupt:
        logger.info(f"[ordonnanceur] stopped after {tours} rounds")
        return 0


__all__ = ["boucle", "tour"]
