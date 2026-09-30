"""API key reservation registry (ticket 035, spec parallelisation_experiences).

LLM throughput is capped per API key: two active experiments can run in parallel only
if their key sets are **disjoint** (R1). This registry is the shared, atomic source of
truth for "which key is held by which run". It lives on `data/experiences/`, a
bind-mount visible both in the `controller` container (where the runs execute) and on
the host (where the scheduler runs).

Atomicity (R6): every read → decide → write sequence runs under a **global lock** taken by
exclusive directory creation (`mkdir`, atomic on the same file system, valid
host ↔ container over the bind-mount). A lock older than `VERROU_TTL_S` is deemed abandoned
and stolen (no process holds the lock that long: the critical section is a
read-modify-write of a few milliseconds).

Reservation ↔ process: a key is held as long as the run has no **terminal status**
(R7). The normal case (end, stop, exception, SIGTERM) releases in the `finally` of `cmd_lancer`.
A process **killed hard** (SIGKILL, OOM) leaves an orphan reservation: it is
`reconcilier()` that detects it — by testing whether the pid is alive, which only makes sense
**in the namespace of the container** where the process lives. `reconcilier()` must therefore
only be called from the container (`python -m experiences reconcilier`), never from the host.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

from loguru import logger

from experiences.archive import ETAT_INTERROMPUE, ETAT_TERMINEE, ETATS_FINAUX, F_ETAT
from experiences.experience import dossier_experience, dossier_experiences

VERROU_TTL_S = 30.0
_ATTENTE_VERROU_S = 5.0
_PAS_ATTENTE_S = 0.05


def _base() -> Path:
    return dossier_experiences()


def _chemin_registre() -> Path:
    return _base() / ".reservations.json"


def _chemin_verrou() -> Path:
    return _base() / ".reservations.lock"


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Verrou:
    """Global inter-process lock via exclusive `mkdir`, stealing a stale lock."""

    def __enter__(self) -> Self:
        verrou = _chemin_verrou()
        verrou.parent.mkdir(parents=True, exist_ok=True)
        debut = time.monotonic()
        while True:
            try:
                verrou.mkdir()
                return self
            except FileExistsError:
                try:
                    age = time.time() - verrou.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > VERROU_TTL_S:
                    logger.warning(
                        f"[reservations] stale lock ({age:.0f}s > {VERROU_TTL_S:.0f}s) — stolen"
                    )
                    try:
                        verrou.rmdir()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() - debut > _ATTENTE_VERROU_S:
                    # Fail-safe (R12): better to make the caller wait than risk a concurrent write.
                    raise TimeoutError(
                        f"verrou de réservation indisponible après {_ATTENTE_VERROU_S:.0f}s"
                    )
                time.sleep(_PAS_ATTENTE_S)

    def __exit__(self, *exc) -> None:
        try:
            _chemin_verrou().rmdir()
        except FileNotFoundError:
            pass


def _lire() -> dict[str, dict]:
    p = _chemin_registre()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError) as e:
        logger.error(
            f"[reservations] unreadable registry ({p}): {e} — reset to empty"
        )
        return {}


def _ecrire(registre: dict[str, dict]) -> None:
    p = _chemin_registre()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registre, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _etat_execution(dossier: str) -> str | None:
    """Current status of a run, read directly from `etat.json` (light, without opening the archive)."""
    p = Path(dossier) / F_ETAT
    if not p.is_file():
        return None
    try:
        return (json.loads(p.read_text(encoding="utf-8")) or {}).get("etat")
    except (json.JSONDecodeError, OSError):
        return None


def _pid_vivant(pid: int) -> bool:
    """Does the process exist IN THIS namespace? (only meaningful in the container, see module.)"""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive but not ours
    return True


def _interrompre(dossier: str, raison: str) -> None:
    """Moves a ghost run to the terminal status `interrompue` to release its keys (R7)."""
    from experiences.archive import Execution

    try:
        Execution.ouvrir(dossier).changer_etat(ETAT_INTERROMPUE, raison=raison)
    except Exception as e:  # noqa: BLE001 — the reconciliation must never crash the scheduler
        logger.error(f"[reservations] reconciliation failed for {dossier}: {e}")


def _reconcilier(registre: dict[str, dict], *, tester_pid: bool) -> dict[str, dict]:
    """Removes the keys whose run is finished; reconciles ghosts (dead pid)."""
    garde: dict[str, dict] = {}
    interrompus: set[str] = set()
    for cle, e in registre.items():
        dossier = e.get("execution", "")
        etat = _etat_execution(dossier)
        if etat in ETATS_FINAUX:
            continue  # key released: terminal status
        if tester_pid and not _pid_vivant(int(e.get("pid") or 0)):
            if dossier not in interrompus:
                _interrompre(dossier, f"processus {e.get('pid')} disparu (clé {cle})")
                interrompus.add(dossier)
            continue
        garde[cle] = e
    return garde


# ── API ───────────────────────────────────────────────────────────────────────


def reconcilier() -> list[str]:
    """Releases the keys of finished runs AND of ghosts (dead pid). To be called ONLY
    in the container (the pid test depends on the namespace). Returns the released keys."""
    with _Verrou():
        avant = _lire()
        apres = _reconcilier(avant, tester_pid=True)
        if apres != avant:
            _ecrire(apres)
        return sorted(set(avant) - set(apres))


def _conflits(registre: dict[str, dict], cles: set[str], dossier: str) -> set[str]:
    """Keys already held by ANOTHER run (a reservation by oneself is not a conflict)."""
    return {
        c for c in cles if c in registre and registre[c].get("execution") != dossier
    }


def _poser(
    registre: dict[str, dict], cles: set[str], dossier: str, exp_nom: str, pid: int
) -> None:
    for c in cles:
        registre[c] = {
            "execution": dossier,
            "exp": exp_nom,
            "pid": pid,
            "depuis": _iso(),
        }


def reserver(
    cles: set[str], dossier_execution: str | Path, exp_nom: str, pid: int | None = None
) -> bool:
    """Atomically reserves ALL of `cles` for this run, or nothing (R1/R6).

    Returns True if reserved (no key held by another run), False otherwise. An empty key
    set is always granted (R8: experiment without gateway calls)."""
    dossier = str(Path(dossier_execution))
    pid = int(pid if pid is not None else os.getpid())
    with _Verrou():
        registre = _reconcilier(_lire(), tester_pid=True)
        conflits = _conflits(registre, cles, dossier)
        if conflits:
            retenues = sorted(f"{c}←{registre[c].get('exp', '?')}" for c in conflits)
            logger.info(
                f"[reservations] {exp_nom} conflicts on {len(conflits)} key(s): {', '.join(retenues)}"
            )
            return False
        _poser(registre, cles, dossier, exp_nom, pid)
        _ecrire(registre)
        if cles:
            logger.info(f"[reservations] {exp_nom} reserves {sorted(cles)}")
        return True


def liberer(dossier_execution: str | Path) -> list[str]:
    """Releases all the keys held by this run. Returns the released keys."""
    dossier = str(Path(dossier_execution))
    with _Verrou():
        registre = _lire()
        liberees = [c for c, e in registre.items() if e.get("execution") == dossier]
        if liberees:
            _ecrire({c: e for c, e in registre.items() if c not in liberees})
            logger.info(f"[reservations] releases {sorted(liberees)} (run closed)")
        return sorted(liberees)


def cles_reservees(*, reconcilier_pid: bool = False) -> set[str]:
    """Keys currently held. `reconcilier_pid` (container only) purges ghosts first."""
    with _Verrou():
        registre = _reconcilier(_lire(), tester_pid=reconcilier_pid)
        _ecrire(registre)
        return set(registre)


def actives(*, reconcilier_pid: bool = False) -> list[dict]:
    """Distinct runs holding at least one key (one entry per run, its keys listed)."""
    with _Verrou():
        registre = _reconcilier(_lire(), tester_pid=reconcilier_pid)
        _ecrire(registre)
        par_exec: dict[str, dict] = {}
        for cle, e in registre.items():
            ref = par_exec.setdefault(
                e.get("execution", ""),
                {"execution": e.get("execution"), "exp": e.get("exp"), "cles": []},
            )
            ref["cles"].append(cle)
        for ref in par_exec.values():
            ref["cles"].sort()
        return sorted(
            par_exec.values(),
            key=lambda r: (r.get("exp") or "", r.get("execution") or ""),
        )


# ── Queue: admission and promotion (they share the registry lock) ──


def admettre(
    cles: set[str],
    dossier_execution: str | Path,
    exp_nom: str,
    args: dict | None = None,
    pid: int | None = None,
) -> str:
    """Atomic admission of a launch (R2/R3/R6). Under ONE lock:

    - keys disjoint from the current reservations → **reserves** and returns ``"lance"``;
    - at least one key held by another experiment → **queues** (FIFO, deduplicated by
      experiment name) and returns ``"file"``.

    Called in the container (pid reconciliation). An empty key set always passes (R8)."""
    from experiences import file as F

    dossier = str(Path(dossier_execution))
    pid = int(pid if pid is not None else os.getpid())
    with _Verrou():
        registre = _reconcilier(_lire(), tester_pid=True)
        conflits = _conflits(registre, cles, dossier)
        if not conflits:
            _poser(registre, cles, dossier, exp_nom, pid)
            _ecrire(registre)
            if cles:
                logger.info(f"[reservations] {exp_nom} reserves {sorted(cles)}")
            # Launched by a route other than promotion (campaign, dashboard, by
            # hand), it left its old request in the queue — which the scheduler then promoted
            # for a second run. Night of 29/09: go123 restarted from scratch this way.
            attente = F.charger()
            siennes = [e for e in attente if e.get("exp") == exp_nom]
            if siennes:
                F.sauver([e for e in attente if e.get("exp") != exp_nom])
                logger.warning(
                    f"[reservations] {exp_nom} gets its keys: its queued request "
                    f"(submitted {siennes[0].get('soumis')}) is withdrawn, this launch replaces it")
            return "lance"
        attente = [e for e in F.charger() if e.get("exp") != exp_nom]
        attente.append(F.entree(exp_nom, cles, args or {}))
        F.sauver(attente)
        retenues = sorted(f"{c}←{registre[c].get('exp', '?')}" for c in conflits)
        logger.info(
            f"[reservations] {exp_nom} queued — key(s) held: {', '.join(retenues)}"
        )
        return "file"


def lister_file() -> list[dict]:
    """Waiting entries, FIFO order (read-only, for display)."""
    from experiences import file as F

    with _Verrou():
        return F.charger()


def retirer_file(exp_nom: str) -> bool:
    """Removes an experiment from the queue before its promotion (R2e). True if an entry is removed."""
    from experiences import file as F

    with _Verrou():
        avant = F.charger()
        apres = [e for e in avant if e.get("exp") != exp_nom]
        if len(apres) != len(avant):
            F.sauver(apres)
            logger.info(f"[reservations] {exp_nom} removed from the queue")
            return True
        return False


def _instant(iso: str | None) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _demande_perimee(entree: dict) -> str | None:
    """Why this queued request must no longer be promoted, or None if it is still valid.

    A request sometimes waits for days; meanwhile, the experiment may have been played by
    another route. Promoting it then reruns a measurement already made: night of 29/09, go123,
    queued on 26/09 behind its own run finished on 27/09, restarted from scratch at 01:38 and
    two hours of Google quota spent for nothing. A request is stale when the latest
    run is `terminee` AND it asked for a resume (there is nothing left to resume) or
    it was submitted BEFORE that end. A request submitted after the end is a deliberate
    "replay": it stays. Fails open — an unreadable date or state leaves the request valid.
    """
    try:
        executions = dossier_experience(str(entree.get("exp"))) / "executions"
        dossiers = sorted(p for p in executions.iterdir() if p.is_dir()) if executions.is_dir() else []
        if not dossiers:
            return None
        brut = json.loads((dossiers[-1] / F_ETAT).read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return None
    if brut.get("etat") != ETAT_TERMINEE:
        return None
    fin = _instant(brut.get("maj"))
    if (entree.get("args") or {}).get("reprendre"):
        return f"reprise demandée, mais sa dernière exécution est terminée ({brut.get('maj')})"
    soumis = _instant(entree.get("soumis"))
    if fin is not None and soumis is not None and fin > soumis:
        return (f"soumise le {entree.get('soumis')}, et l'expérience s'est terminée depuis "
                f"({brut.get('maj')})")
    return None


def promouvoir_pretes() -> list[dict]:
    """Takes out of the queue, in FIFO order, the experiments whose keys are ALL free
    (R2b), never choosing two entries with overlapping keys in the same round. Does NOT
    reserve: the relaunch of `lancer` will reserve at its start. Called on the host side AFTER a
    reconciliation in the container; therefore does not test pids.

    Returns the entries to relaunch, in order. An active experiment is never preempted
    (R2c): only keys that are already free are consumed."""
    from experiences import file as F

    with _Verrou():
        pris = set(_lire())
        choisies, restantes, perimees = [], [], []
        for e in F.charger():
            motif = _demande_perimee(e)
            if motif is not None:
                perimees.append(e)
                logger.warning(
                    f"[reservations] request of {e.get('exp')} removed from the queue without "
                    f"being promoted — {motif}")
                continue
            cles = set(e.get("cles") or [])
            if cles & pris:
                restantes.append(e)
            else:
                choisies.append(e)
                pris |= cles
        if choisies or perimees:
            F.sauver(restantes)
            for e in choisies:
                logger.info(
                    f"[reservations] promoting {e.get('exp')} (keys {e.get('cles')})"
                )
        return choisies


__all__ = [
    "actives",
    "admettre",
    "cles_reservees",
    "liberer",
    "lister_file",
    "promouvoir_pretes",
    "reconcilier",
    "reserver",
    "retirer_file",
]
