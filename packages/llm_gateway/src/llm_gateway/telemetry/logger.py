"""
telemetry/logger.py — Unified logging via loguru.

Handler configuration is an explicit action of the entrypoints
(create_app, Celery worker) via configure_logging() — never again an import
side effect of this module.

log_llm_exchange() writes a JSONL to the file designated by the environment
variable LLM_EXCHANGES_FILE, or failing that APP_WORKDIR/llm_exchanges.jsonl:
  - time, task_id, provider, tokens_in, tokens_out, messages (input), response (output)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

_configured = False


_handler_ids: list[int] = []
_LOGURU_DEFAULT_HANDLER_ID = 0


def configure_logging(telemetry: Any = None, level: str | None = None, *, replace_default_handler: bool = True) -> None:
    """
    Install the gateway handler (level, text or JSON format, per-service file sink).
    Idempotent — called by the entry points (create_app, worker, CLI), never by an import.

    Non-intrusive: only loguru's **default** handler (id 0, DEBUG on stderr) is removed,
    and only if `replace_default_handler`; the sinks a host installed itself stay in
    place. `reset_logging()` undoes what this function installed (tests, embedded mode).

    `telemetry`: the `GatewaySettings.telemetry` settings. Without settings, falls back to the old
    environment variables (LOG_LEVEL, SERVICE_NAME, APP_WORKDIR), deprecated.
    """
    global _configured
    if _configured:
        return
    _configured = True
    lvl = level or _attr(telemetry, "log_level") or os.environ.get("LOG_LEVEL", "INFO")
    if replace_default_handler:
        try:
            logger.remove(_LOGURU_DEFAULT_HANDLER_ID)
        except ValueError:
            pass   # already removed by the host
    if _attr(telemetry, "log_format") == "json":
        _handler_ids.append(logger.add(sys.stderr, level=lvl, serialize=True))
    else:
        _handler_ids.append(logger.add(
            sys.stderr,
            level=lvl,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        ))
    _add_service_file_sink(lvl, telemetry)


def reset_logging() -> None:
    """Remove the handlers installed by `configure_logging` (loguru's default is not restored)."""
    global _configured
    for hid in _handler_ids:
        try:
            logger.remove(hid)
        except ValueError:
            pass
    _handler_ids.clear()
    _configured = False


def _attr(telemetry: Any, name: str) -> Any:
    """Tolerant read of a telemetry setting (object or None)."""
    return getattr(telemetry, name, None) if telemetry is not None else None


def _add_service_file_sink(level: str, telemetry: Any = None) -> None:
    """Per-service file sink in the workdir, enabled if a service name is defined.

    Format identical to the controller logs (`app.log`) so that the same aggregation
    regex works. Tolerant: any error (missing workdir, permissions) is
    silent — console logging stays operational.
    """
    service = _attr(telemetry, "service_name") or os.environ.get("SERVICE_NAME")
    if not service:
        return
    try:
        workdir = _workdir(telemetry)
        if not workdir.exists():
            return
        _handler_ids.append(logger.add(
            str(workdir / f"{service}.log"),
            level=level,
            rotation="10 MB",
            retention="7 days",
            enqueue=True,  # thread/process-safe writes (multi-threaded worker)
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        ))
    except OSError:
        pass


def get_logger(name: str):
    """Return the loguru logger. The name parameter is kept for API compatibility."""
    return logger


# ---------------------------------------------------------------------------
# Helper to log the metrics of a completed LLM call
# ---------------------------------------------------------------------------

def log_llm_call(
    task_id: str,
    provider: str,
    status: str,        # "success" | "failed"
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    http_status: int,
    error: str | None = None,
) -> None:
    msg = (
        f"llm_call_completed | task_id={task_id} provider={provider} status={status} "
        f"latency_ms={latency_ms:.1f} tokens_in={tokens_in} tokens_out={tokens_out} "
        f"http_status={http_status}"
    )
    if error:
        msg += f" error={error}"

    if status == "success":
        logger.info(msg)
    else:
        logger.error(msg)


# ---------------------------------------------------------------------------
# LLM exchange log (prompt sent + response + tokens)
# ---------------------------------------------------------------------------

def _workdir(telemetry: Any = None) -> Path:
    """Run folder: `telemetry.workdir`, else APP_WORKDIR (deprecated), else the current directory."""
    configured = _attr(telemetry, "workdir")
    if configured is not None:
        return Path(configured)
    return Path(os.environ.get("APP_WORKDIR", "."))


def log_llm_error(
    task_id: str,
    provider: str,
    error_type: str,
    error_message: str,
    http_status: int | None = None,
    ratelimit_reset: str | None = None,
    telemetry: Any = None,
    origine: str | None = None,
) -> None:
    """
    Record an LLM error in <workdir>/llm_errors.jsonl.
    """
    log_file = _workdir(telemetry) / "llm_errors.jsonl"

    entry = {
        "time": datetime.now(UTC).isoformat(),
        "task_id": task_id,
        "provider": provider,
        "error_type": error_type,
        "error_message": error_message,
        "http_status": http_status,
        # The worker is shared by several simulations. Without the origin, an archive
        # attributed to the current arm the errors of another run served at the same time.
        "origine": origine,
    }
    if ratelimit_reset is not None:
        entry["ratelimit_reset"] = ratelimit_reset

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        logger.warning(f"Cannot write to {log_file}: {e}")


def log_llm_exchange(
    task_id: str,
    provider: str,
    messages: list[dict],
    response: Any,
    tokens_in: int,
    tokens_out: int,
    category: str = "",
    sim_ts: float | None = None,
    telemetry: Any = None,
    origine: str | None = None,
) -> None:
    """
    Record a complete exchange with the LLM in the exchange log (cf. telemetry/exchanges).

    With settings: nothing is written while `telemetry.exchanges_enabled` is false; the
    path is `telemetry.exchanges_file` else <workdir>/llm_exchanges.jsonl; the redactor
    `telemetry.redactor` transforms the record before writing; rotation by size.
    Without settings (legacy call): falls back to LLM_EXCHANGES_FILE / APP_WORKDIR, deprecated.
    """
    from llm_gateway.telemetry.exchanges import ExchangeJournal, ExchangeRecord, load_redactor

    if telemetry is not None and not _attr(telemetry, "exchanges_enabled"):
        return
    configured = _attr(telemetry, "exchanges_file")
    override = os.environ.get("LLM_EXCHANGES_FILE")
    if configured is not None:
        log_file = Path(configured)
    elif override:
        log_file = Path(override)
    else:
        log_file = _workdir(telemetry) / "llm_exchanges.jsonl"
    try:
        redactor = load_redactor(_attr(telemetry, "redactor"))
    except Exception as e:  # a misconfigured redactor must neither write in clear nor crash the batch
        logger.error(f"[ALARME] Exchange log redactor cannot be loaded, log skipped | error={e!r}")
        return
    journal = ExchangeJournal(
        log_file, redactor=redactor, max_bytes=int(_attr(telemetry, "exchanges_max_bytes") or 200_000_000),
    )
    journal.write(ExchangeRecord(
        task_id=task_id, provider=provider, category=category, tokens_in=tokens_in,
        tokens_out=tokens_out, messages=messages, response=response, sim_ts=sim_ts,
        origine=origine,
    ))
