"""Launching and following `make` targets from the dashboard.

Each launch is a detached subprocess (`start_new_session`) whose
output is written to `experiments/.dashboard/<job>.log`. The registry survives
Streamlit reruns (`st.cache_resource`), which makes it possible to follow a job
while it runs and to stop it cleanly (SIGTERM to the process
group, then SIGKILL after a grace period).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "experiments" / ".dashboard"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


@dataclass
class Job:
    id: str
    label: str
    argv: list[str]
    cwd: Path
    log_path: Path
    started_at: float
    proc: subprocess.Popen | None = None
    returncode: int | None = None
    finished_at: float | None = None
    stopped_by_user: bool = False
    error: str | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.returncode is None

    @property
    def duration(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @property
    def state(self) -> str:
        if self.error:
            return "erreur"
        if self.running:
            return "en cours"
        if self.stopped_by_user:
            return "arrêté"
        return "ok" if self.returncode == 0 else "échec"

    @property
    def command_line(self) -> str:
        return " ".join(self.argv)


_RE_INDEX_JOB = re.compile(r"^(\d{3,})-")


def dernier_index(dossier: Path) -> int:
    """The largest job number already written in this log directory (0 if it is empty).

    The counter restarts from there, instead of 1: the registry lives in `st.cache_resource`, so
    a restart of the Streamlit server resets it, whereas the jobs it launched
    are DETACHED and keep writing. On 2026-09-08, a `001-root-experience-lancer` job
    reopened for writing the log of a `001-root-experience-lancer` from a previous
    session that was still alive: both handles wrote to the same file, each at
    its own offset, and one job's console showed the other's output.
    """
    plus_grand = 0
    try:
        noms = [p.name for p in dossier.glob("*.log")]
    except OSError:
        return 0
    for nom in noms:
        m = _RE_INDEX_JOB.match(nom)
        if m:
            plus_grand = max(plus_grand, int(m.group(1)))
    return plus_grand


class Registry:
    """Job registry, shared by all the reruns of the Streamlit server."""

    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._counter = dernier_index(LOG_DIR)

    # ── reading ───────────────────────────────────────────────────────────────
    def jobs(self) -> list[Job]:
        with self._lock:
            self._reap()
            return list(reversed(self._jobs))

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._reap()
            return next((j for j in self._jobs if j.id == job_id), None)

    def running_count(self) -> int:
        return sum(1 for j in self.jobs() if j.running)

    def _reap(self) -> None:
        for job in self._jobs:
            if job.proc is not None and job.returncode is None:
                code = job.proc.poll()
                if code is not None:
                    job.returncode = code
                    job.finished_at = time.time()

    # ── writing ───────────────────────────────────────────────────────────────
    def launch(self, label: str, argv: list[str], cwd: Path, flags: tuple[str, ...] = ()) -> Job:
        with self._lock:
            slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()[:48]
            # Never another job's log, even from a previous session: a file that
            # already exists may be held open by a detached process still alive.
            while True:
                self._counter += 1
                job_id = f"{self._counter:03d}-{slug}"
                log_path = LOG_DIR / f"{job_id}.log"
                if not log_path.exists():
                    break
            job = Job(
                id=job_id,
                label=label,
                argv=argv,
                cwd=cwd,
                log_path=log_path,
                started_at=time.time(),
                flags=flags,
            )
            try:
                handle = log_path.open("w", encoding="utf-8", errors="replace")
                handle.write(f"$ cd {cwd}\n$ {' '.join(argv)}\n\n")
                handle.flush()
                job.proc = subprocess.Popen(  # noqa: S603 — argv construit depuis le Makefile
                    argv,
                    cwd=str(cwd),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    env={**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"},
                )
            except OSError as exc:
                job.error = str(exc)
                job.finished_at = time.time()
                job.returncode = -1
            self._jobs.append(job)
            return job

    def stop(self, job_id: str, grace: float = 5.0) -> bool:
        """SIGTERM to the process group, SIGKILL if still alive after `grace`."""
        job = self.get(job_id)
        if job is None or job.proc is None or not job.running:
            return False
        job.stopped_by_user = True
        try:
            pgid = os.getpgid(job.proc.pid)
        except ProcessLookupError:
            return False
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 1.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                job.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        job.returncode = job.proc.poll()
        job.finished_at = time.time()
        return True

    def stop_all(self) -> int:
        return sum(1 for job in self.jobs() if job.running and self.stop(job.id))

    def clear_finished(self) -> None:
        with self._lock:
            self._reap()
            self._jobs = [j for j in self._jobs if j.running]


def par_priorite(jobs: list[Job]) -> list[Job]:
    """What is running first, the most recent first within each group.

    `jobs()` returns reverse chronological order: a job finished a minute ago came
    above a job still in progress, in a panel named « Activités en cours ».
    """
    return sorted(jobs, key=lambda job: not job.running)


def tail(job: Job, max_lines: int = 400) -> str:
    """Last lines of a job's log, stripped of ANSI codes."""
    try:
        raw = job.log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log indisponible)"
    lines = _ANSI_RE.sub("", raw).replace("\r", "\n").splitlines()
    clipped = lines[-max_lines:]
    prefix = f"… {len(lines) - len(clipped)} lignes antérieures dans {job.log_path.name}\n" if len(lines) > len(clipped) else ""
    return prefix + "\n".join(clipped)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
