"""“Real-time” state: the only dashboard probes that go beyond the disk.

`metrics.py` stays strictly off the network; everything that queries a live
service is isolated here, with short timeouts and a silent fallback — a
stopped service is a normal state, not an error.

Three probes:
  * GET http://localhost:8000/health   → provider quotas and availability
  * GET http://localhost:8002/metrics  → controller gauges (cycle, agents…)
  * pgrep                              → headless launcher / GAMA GUI
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field

API_HEALTH_URL = "http://localhost:8000/health"
CONTROLLER_METRICS_URL = "http://localhost:8002/metrics"
GAMA_GUI_PATTERN = "/Applications/GAMA.app/Contents/MacOS/GAMA"

# Controller gauges kept for steering (handle/application.py).
_CONTROLLER_GAUGES = (
    "gama_sim_step_count",
    "gama_sim_logical_time_seconds",
    "gama_sim_real_elapsed_seconds",
    "gama_sim_agents_total",
    "gama_agent_states",
    "controller_backlog_fill_ratio",
    "controller_agents_stuck",
    "controller_throughput_tasks_per_min",
    "controller_init_stage",
    "controller_init_progress_ratio",
)
_METRIC_RE = re.compile(r'^([a-z_]+)(?:\{([^}]*)\})?\s+([0-9eE.+-]+)$')


@dataclass
class ProviderLive:
    name: str
    current_rpm: int
    rpm_limit: int | None
    active_tasks: int
    usage_pct: float
    cooldown: bool
    daily_requests: int
    rpd_limit: int | None
    daily_tokens: int
    tpd_limit: int | None
    quota_exhausted: bool
    available: bool


@dataclass
class ApiHealth:
    available: bool
    providers: list[ProviderLive] = field(default_factory=list)
    error: str = ""


def _get(url: str, timeout: float) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost
            return resp.read()
    except OSError:
        return None


def api_health(timeout: float = 2.0) -> ApiHealth:
    """Provider quotas and availability, as seen by the load balancer."""
    raw = _get(API_HEALTH_URL, timeout)
    if raw is None:
        return ApiHealth(False, error="API (port 8000) injoignable — pile arrêtée ?")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ApiHealth(False, error=f"/health illisible : {exc}")

    providers = [
        ProviderLive(
            name=name,
            current_rpm=int(p.get("current_rpm") or 0),
            rpm_limit=p.get("rpm_limit"),
            active_tasks=int(p.get("active_tasks") or 0),
            usage_pct=float(p.get("usage_pct") or 0.0),
            cooldown=bool(p.get("cooldown")),
            daily_requests=int(p.get("daily_requests") or 0),
            rpd_limit=p.get("rpd_limit"),
            daily_tokens=int(p.get("daily_tokens") or 0),
            tpd_limit=p.get("tpd_limit"),
            quota_exhausted=bool(p.get("quota_exhausted")),
            available=bool(p.get("available")),
        )
        for name, p in (data.get("providers") or {}).items()
    ]
    providers.sort(key=lambda p: p.name)
    return ApiHealth(True, providers)


def controller_stats(timeout: float = 2.0) -> dict[str, float]:
    """Controller gauges. `gama_agent_states` is split by state
    (`agents_inactive` / `agents_ready` / `agents_active`). Empty dict if the
    controller does not answer."""
    raw = _get(CONTROLLER_METRICS_URL, timeout)
    if raw is None:
        return {}
    out: dict[str, float] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        match = _METRIC_RE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name not in _CONTROLLER_GAUGES:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if name == "gama_agent_states" and labels:
            state = re.search(r'state="([^"]+)"', labels)
            if state:
                out[f"agents_{state.group(1)}"] = number
            continue
        out[name] = number
    return out


@dataclass
class RunProcess:
    active: bool
    mode: str = ""  # "offline" | "ihm"
    pid: int | None = None


def _pgrep(pattern: str) -> int | None:
    try:
        proc = subprocess.run(  # noqa: S603 — motif fixe
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    first = proc.stdout.strip().splitlines()
    return int(first[0]) if first else None


def run_process() -> RunProcess:
    """Detection of the GAMA run — the same guards as `make run` (Makefile)."""
    pid = _pgrep("launch_headless.py")
    if pid is not None:
        return RunProcess(True, "offline", pid)
    pid = _pgrep(GAMA_GUI_PATTERN)
    if pid is not None:
        return RunProcess(True, "ihm", pid)
    return RunProcess(False)
