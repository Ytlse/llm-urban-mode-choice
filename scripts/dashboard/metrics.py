"""Collection of the metrics shown by the dashboard.

Local sources, all read in read-only mode and without network calls (the HTTP
probes live in `live.py`):
  * `docker compose ps`           → state of the services
  * `experiments/**`              → health of the runs (logs, moves.csv, agents,
                                    LLM errors, semantic cache)
  * `config/llm_gateway/providers.yaml` → declared quotas of the providers
  * `docs/synthesis/data.json`    → scores of the synthesis page
"""

from __future__ import annotations

import json
import shutil
import subprocess
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The compose file lives in infra/ (ticket 039): `-f` points to it and
# `--project-directory` keeps the root as the base of its relative paths.
COMPOSE_CMD = ["docker", "compose", "-f", str(REPO_ROOT / "infra" / "docker-compose.yml"),
               "--project-directory", str(REPO_ROOT)]

EXPERIMENTS = REPO_ROOT / "experiments"
SYNTHESIS_DATA = REPO_ROOT / "docs" / "synthesis" / "data.json"

# Expected services (docker-compose.yml) — used to spot the missing ones.
EXPECTED_SERVICES = ("api", "worker", "controller", "redis", "otp", "grafana", "prometheus")


# ── Docker ────────────────────────────────────────────────────────────────────
@dataclass
class Service:
    name: str
    state: str
    status: str
    health: str = ""

    @property
    def kind(self) -> str:
        if self.health in ("unhealthy", "starting") or self.state in ("restarting", "paused"):
            return "warning"
        if self.state == "running":
            return "good"
        if self.state in ("exited", "dead"):
            return "critical"
        return "muted"


@dataclass
class DockerStatus:
    available: bool
    services: list[Service] = field(default_factory=list)
    error: str = ""

    @property
    def running(self) -> int:
        return sum(1 for s in self.services if s.state == "running")

    @property
    def missing(self) -> list[str]:
        present = {s.name for s in self.services}
        return [name for name in EXPECTED_SERVICES if not any(name in p for p in present)]


def docker_status(timeout: float = 8.0) -> DockerStatus:
    if shutil.which("docker") is None:
        return DockerStatus(False, error="binaire `docker` introuvable")
    try:
        proc = subprocess.run(  # noqa: S603 — commande fixe
            [*COMPOSE_CMD, "ps", "--format", "json", "--all"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return DockerStatus(False, error=f"docker compose ps a échoué : {exc}")
    if proc.returncode != 0:
        return DockerStatus(False, error=(proc.stderr or "").strip()[:300])

    raw = proc.stdout.strip()
    records: list[dict] = []
    if raw.startswith("["):
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            records = []
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    services = [
        Service(
            name=r.get("Service") or r.get("Name", "?"),
            state=(r.get("State") or "").lower(),
            status=r.get("Status", ""),
            health=(r.get("Health") or "").lower(),
        )
        for r in records
    ]
    return DockerStatus(True, sorted(services, key=lambda s: s.name))


# ── Runs ──────────────────────────────────────────────────────────────────────
@dataclass
class RunInfo:
    path: Path
    label: str
    is_current: bool
    modified: datetime
    log_size: int
    errors: int = 0
    warnings: int = 0
    alarms: int = 0
    log_span: tuple[str, str] | None = None
    has_moves: bool = False

    @property
    def rel_path(self) -> str:
        return str(self.path.relative_to(REPO_ROOT))


@dataclass
class MovesStats:
    trips: int
    persons: int
    modal_split: list[tuple[str, int]]
    selection: list[tuple[str, int]]
    with_distribution: int
    sim_start: datetime | None
    sim_end: datetime | None
    delay_mean: float | None
    delay_p95: float | None

    @property
    def sim_hours(self) -> float | None:
        if self.sim_start and self.sim_end:
            return (self.sim_end - self.sim_start).total_seconds() / 3600
        return None

    @property
    def llm_share(self) -> float | None:
        total = sum(n for _, n in self.selection)
        if not total:
            return None
        llm = sum(n for k, n in self.selection if k == "LLM")
        return 100 * llm / total

    @property
    def llm_error_share(self) -> float | None:
        total = sum(n for _, n in self.selection)
        if not total:
            return None
        errs = sum(n for k, n in self.selection if "Error" in k or "error" in k)
        return 100 * errs / total


def log_counts(log_path: Path, max_bytes: int = 64 * 1024 * 1024) -> tuple[int, int, int, tuple[str, str] | None]:
    """Counts ERROR / WARNING / [ALARME] and the time bounds of the log."""
    errors = warnings = alarms = 0
    first = last = ""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            if log_path.stat().st_size > max_bytes:  # safeguard: we only read the end
                fh.seek(log_path.stat().st_size - max_bytes)
                fh.readline()
            for line in fh:
                if "| ERROR" in line:
                    errors += 1
                elif "| WARNING" in line:
                    warnings += 1
                if "[ALARME]" in line:
                    alarms += 1
                if len(line) > 19 and line[4] == "-" and line[13] == ":":
                    if not first:
                        first = line[:19]
                    last = line[:19]
    except OSError:
        return 0, 0, 0, None
    return errors, warnings, alarms, ((first, last) if first else None)


def list_runs(limit: int = 12) -> list[RunInfo]:
    """Most recent runs, without going through the logs (see `log_counts`).

    `experiments/current` is a symbolic link to the archive of the run in progress:
    we deduplicate on the resolved path so as not to list the same run twice,
    and we mark the archive pointed to as « en cours ».
    """
    current = EXPERIMENTS / "current"
    current_target = current.resolve() if current.exists() else None

    candidates: list[Path] = []
    archive = EXPERIMENTS / "archive"
    if archive.is_dir():
        candidates += [p for p in archive.iterdir() if p.is_dir()]
    # `current` is only added if it does not already point into the listed archive.
    if current.is_dir() and current_target not in {p.resolve() for p in candidates}:
        candidates.append(current)

    runs: list[RunInfo] = []
    for path in candidates:
        log = path / "app.log"
        stat_target = log if log.is_file() else path
        try:
            mtime = datetime.fromtimestamp(stat_target.stat().st_mtime)
        except OSError:
            continue
        is_current = current_target is not None and path.resolve() == current_target
        name = "current" if path.name == "current" else path.name
        runs.append(
            RunInfo(
                path=path,
                label=f"{name} (en cours)" if is_current and path.name != "current" else name,
                is_current=is_current,
                modified=mtime,
                log_size=log.stat().st_size if log.is_file() else 0,
                has_moves=(path / "moves.csv").is_file(),
            )
        )

    runs.sort(key=lambda r: (r.is_current, r.modified), reverse=True)
    return runs[:limit]


def moves_stats(run_path: Path) -> MovesStats | None:
    """Statistics of a run's moves.csv (pandas required)."""
    csv = run_path / "moves.csv"
    if not csv.is_file():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None

    header = pd.read_csv(csv, nrows=0).columns.tolist()
    wanted = [
        "Mode de transport Choisi",
        "ID Personne",
        "Temps simulé",
        "Méthode de sélection",
        "P(Marche) %",
        "Retard planification (s)",
    ]
    usecols = [c for c in wanted if c in header]
    try:
        df = pd.read_csv(csv, usecols=usecols)
    except (ValueError, OSError):
        return None

    def counts(col: str) -> list[tuple[str, int]]:
        if col not in df:
            return []
        vc = df[col].value_counts()
        return [(str(k), int(v)) for k, v in vc.items()]

    sim_start = sim_end = None
    if "Temps simulé" in df:
        times = pd.to_numeric(df["Temps simulé"], errors="coerce").dropna()
        if not times.empty:
            sim_start = datetime.fromtimestamp(float(times.min()))
            sim_end = datetime.fromtimestamp(float(times.max()))

    delay_mean = delay_p95 = None
    if "Retard planification (s)" in df:
        delays = pd.to_numeric(df["Retard planification (s)"], errors="coerce").dropna()
        if not delays.empty:
            delay_mean = float(delays.mean())
            delay_p95 = float(delays.quantile(0.95))

    with_distribution = 0
    if "P(Marche) %" in df:
        with_distribution = int(df["P(Marche) %"].notna().sum())

    return MovesStats(
        trips=len(df),
        persons=int(df["ID Personne"].nunique()) if "ID Personne" in df else 0,
        modal_split=counts("Mode de transport Choisi"),
        selection=counts("Méthode de sélection"),
        with_distribution=with_distribution,
        sim_start=sim_start,
        sim_end=sim_end,
        delay_mean=delay_mean,
        delay_p95=delay_p95,
    )


# ── Score synthesis ───────────────────────────────────────────────────────────
@dataclass
class SynthesisSummary:
    available: bool
    generated_at: str = ""
    run_id: str = ""
    run_pinned: bool = False
    n_trips: int = 0
    n_persons: int = 0
    pct_distribution: float = 0.0
    primary: str = ""
    dims: list[str] = field(default_factory=list)
    arms: list[dict] = field(default_factory=list)
    arm_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def synthesis_summary() -> SynthesisSummary:
    if not SYNTHESIS_DATA.is_file():
        return SynthesisSummary(False, error="docs/synthesis/data.json absent — lancez `make synthesis`")
    try:
        data = json.loads(SYNTHESIS_DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SynthesisSummary(False, error=f"data.json illisible : {exc}")

    common = data.get("common_set", {}) or {}
    synth = data.get("synthesis", {}) or {}
    dims = [d.get("label", d.get("key", "?")) for d in synth.get("dims", [])]
    arms = []
    for arm in synth.get("arms", []):
        cells = [c.get("value") for c in arm.get("cells", [])]
        arms.append({"label": arm.get("label", "?"), "cells": cells, "basis": arm.get("basis", "")})

    return SynthesisSummary(
        available=True,
        generated_at=str(data.get("generated_at", "")),
        run_id=str(common.get("run_id", "")),
        run_pinned=bool(common.get("run_pinned")),
        n_trips=int(common.get("n_trips") or 0),
        n_persons=int(common.get("n_persons") or 0),
        pct_distribution=float(common.get("pct_distribution") or 0.0),
        primary=str((data.get("score_def") or {}).get("primary", "")),
        dims=dims,
        arms=arms,
        arm_status={k: str((v or {}).get("status", "?")) for k, v in (data.get("arms") or {}).items()},
        warnings=[str(w) for w in (common.get("warnings") or [])],
    )


# ── GAMA run: progress and health ────────────────────────────────────────────
def agent_states(run_path: Path):
    """Inactive/ready/active curve of the run — `gama_results/agent_states.csv`,
    one row per controller `/sync`. None if absent or unreadable."""
    csv = run_path / "gama_results" / "agent_states.csv"
    if not csv.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(csv)
    except (ImportError, ValueError, OSError):
        return None
    needed = {"step", "sim_time", "inactive", "ready", "active", "total"}
    return df if needed.issubset(df.columns) else None


_LOG_PREFIX_RE = None


def top_log_messages(
    log_path: Path, level: str = "ERROR", limit: int = 8, max_bytes: int = 64 * 1024 * 1024
) -> list[tuple[int, str]]:
    """Most frequent ERROR/WARNING messages, normalised (numbers → N,
    hex identifiers → #). Returns [(occurrences, example)] sorted descending."""
    global _LOG_PREFIX_RE
    import re

    if _LOG_PREFIX_RE is None:
        _LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]?\d*\s*\|\s*\w+\s*\|\s*")
    marker = f"| {level}"
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            if log_path.stat().st_size > max_bytes:
                fh.seek(log_path.stat().st_size - max_bytes)
                fh.readline()
            for line in fh:
                if marker not in line:
                    continue
                message = _LOG_PREFIX_RE.sub("", line).strip()
                key = re.sub(r"0x[0-9a-fA-F]+|[0-9a-f]{8,}", "#", message)
                key = re.sub(r"\d+(?:\.\d+)?", "N", key)[:200]
                counts[key] = counts.get(key, 0) + 1
                examples.setdefault(key, message[:300])
    except OSError:
        return []
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [(n, examples[key]) for key, n in ranked]


@dataclass
class LlmErrorStats:
    total: int = 0
    by_provider: list[tuple[str, int]] = field(default_factory=list)
    n_429: int = 0
    by_provider_429: list[tuple[str, int]] = field(default_factory=list)
    last_time: str = ""


def llm_errors_stats(run_path: Path) -> LlmErrorStats:
    """Breakdown of `llm_errors.jsonl` (one JSON line per LLM error)."""
    path = run_path / "llm_errors.jsonl"
    stats = LlmErrorStats()
    if not path.is_file():
        return stats
    by_provider: dict[str, int] = {}
    by_provider_429: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats.total += 1
                provider = str(rec.get("provider") or "?")
                by_provider[provider] = by_provider.get(provider, 0) + 1
                if rec.get("http_status") == 429:
                    stats.n_429 += 1
                    by_provider_429[provider] = by_provider_429.get(provider, 0) + 1
                stats.last_time = str(rec.get("time") or stats.last_time)
    except OSError:
        return stats
    stats.by_provider = sorted(by_provider.items(), key=lambda kv: kv[1], reverse=True)
    stats.by_provider_429 = sorted(by_provider_429.items(), key=lambda kv: kv[1], reverse=True)
    return stats


def llm_cache_hit_rate(run_path: Path) -> tuple[float, int, int] | None:
    """Hit rate of the LLM semantic cache = hits / (hits + real calls).

    `llm_cache_hits.jsonl` is real JSONL (1 line = 1 hit);
    `llm_exchanges.jsonl` is a concatenation of indented JSON objects — we
    count the lines reduced to `{`, which open each object (same computation as
    scripts/debug/run_report.py)."""
    hits_path = run_path / "llm_cache_hits.jsonl"
    exch_path = run_path / "llm_exchanges.jsonl"
    if not hits_path.is_file() and not exch_path.is_file():
        return None

    def count_lines(path: Path, predicate) -> int:
        if not path.is_file():
            return 0
        n = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if predicate(line):
                        n += 1
        except OSError:
            return 0
        return n

    hits = count_lines(hits_path, lambda l: l.strip().startswith("{"))
    exchanges = count_lines(exch_path, lambda l: l.rstrip("\n") == "{")
    total = hits + exchanges
    if not total:
        return None
    return 100 * hits / total, hits, exchanges


# ── Providers (lecture statique de providers.yaml) ───────────────────────────
PROVIDERS_YAML = REPO_ROOT / "config" / "llm_gateway" / "providers.yaml"


@dataclass
class ProvidersStatic:
    available: bool
    providers: list[dict] = field(default_factory=list)
    refreshed_at: datetime | None = None
    error: str = ""


def providers_static() -> ProvidersStatic:
    """Providers declared in `providers.yaml` (the commented-out blocks — removed
    models — are invisible to yaml.safe_load, this is intended). The file's mtime
    dates the last `make providers`."""
    if not PROVIDERS_YAML.is_file():
        return ProvidersStatic(False, error="config/llm_gateway/providers.yaml absent")
    try:
        import yaml

        data = yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 — yaml.YAMLError + OSError
        return ProvidersStatic(False, error=f"providers.yaml illisible : {exc}")

    providers = []
    for name, cfg in (data.get("providers") or {}).items():
        cfg = cfg or {}
        providers.append(
            {
                "name": name,
                "adapter": cfg.get("adapter", name),
                "model": cfg.get("default_model", "?"),
                "rpm_limit": cfg.get("rpm_limit"),
                "tpm_limit": cfg.get("tpm_limit"),
                "rpd_limit": cfg.get("rpd_limit"),
                "tpd_limit": cfg.get("tpd_limit"),
                "weight": cfg.get("weight", 1.0),
            }
        )
    providers.sort(key=lambda p: p["name"])
    return ProvidersStatic(
        True, providers, refreshed_at=datetime.fromtimestamp(PROVIDERS_YAML.stat().st_mtime)
    )


# ── Miscellaneous ─────────────────────────────────────────────────────────────
# ── Container probe (scripts/debug/watch_containers.py) ──────────────────────
DOSSIER_SONDE = REPO_ROOT / "experiments" / ".dashboard" / "conteneurs"


@dataclass
class SondeConteneurs:
    """What the last probe recorded. File reading only, never network."""

    dossier: Path | None = None
    debut: str | None = None
    mesures: int = 0
    chutes: list[str] = field(default_factory=list)
    appelants: list[dict] = field(default_factory=list)
    pics: dict[str, int] = field(default_factory=dict)
    alarmes: list[str] = field(default_factory=list)

    @property
    def presente(self) -> bool:
        return self.dossier is not None


def _lignes_appelant(chemin: Path) -> dict:
    """An `appelant-*.txt` file: the service, the action, and the captured commands.

    We only keep the lines that look like a docker or make command: they are the ones
    that name the culprit. The rest of the snapshot is environment noise.
    """
    nom = chemin.stem  # appelant-<conteneur>-<action>-<HH_MM_SS>
    morceaux = nom.split("-")
    action = morceaux[-2] if len(morceaux) >= 3 else "?"
    heure = morceaux[-1].replace("_", ":") if len(morceaux) >= 2 else "?"
    service = "-".join(morceaux[1:-2]).replace("llm-agents-gama-", "")
    commandes = []
    for ligne in chemin.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ligne.startswith("#") or not ligne.strip():
            continue
        # « pid ppid Day Month DD HH:MM:SS YYYY command… »
        morceaux_l = ligne.split()
        commande = " ".join(morceaux_l[7:]) if len(morceaux_l) > 7 else ligne
        if re.search(r"docker\s+(compose\s+)?(stop|kill|down|restart)|make\s+\S*(down|stop|restart)", commande):
            commandes.append(f"pid {morceaux_l[0]} (parent {morceaux_l[1]}) : {commande[:160]}")
    return {"service": service, "action": action, "heure": heure, "commandes": commandes,
            "fichier": chemin}


def sonde_conteneurs(dossier: Path | None = None) -> SondeConteneurs:
    """The probe's last campaign: measurements, crashes, callers captured."""
    racine = Path(dossier) if dossier is not None else DOSSIER_SONDE
    if not racine.is_dir():
        return SondeConteneurs()
    campagnes = sorted((p for p in racine.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not campagnes:
        return SondeConteneurs()
    d = campagnes[-1]

    mesures, pics = 0, {}
    csv_chemin = d / "memoire.csv"
    if csv_chemin.is_file():
        try:
            with csv_chemin.open(encoding="utf-8") as f:
                for ligne in csv.DictReader(f):
                    mesures += 1
                    if ligne.get("octets"):
                        nom = (ligne["service"] or "").replace("llm-agents-gama-", "")
                        pics[nom] = max(pics.get(nom, 0), int(ligne["octets"]))
        except (OSError, ValueError):
            pass

    alarmes = []
    log = d / "sonde.log"
    debut = None
    if log.is_file():
        for ligne in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "[ALARME]" in ligne:
                alarmes.append(ligne)
            elif debut is None and "début de la sonde" in ligne:
                debut = ligne.split("|")[0].strip()

    return SondeConteneurs(
        dossier=d, debut=debut, mesures=mesures, pics=pics, alarmes=alarmes[-6:],
        chutes=sorted(p.stem.replace("chute-", "").replace("llm-agents-gama-", "")
                      for p in d.glob("chute-*.txt")),
        appelants=[_lignes_appelant(p) for p in sorted(d.glob("appelant-*.txt"))[-6:]],
    )


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            proc = subprocess.run(  # noqa: S603 — commandes git fixes
                ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    dirty = run("status", "--porcelain")
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "?",
        "head": run("log", "-1", "--pretty=%h %s"),
        "head_date": run("log", "-1", "--pretty=%cd", "--date=short"),
        "dirty": str(len([line for line in dirty.splitlines() if line.strip()])),
    }


def human_size(num: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if abs(num) < 1024:
            return f"{num:.0f} {unit}" if unit == "o" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} To"
