"""telemetry/exchanges.py — the LLM exchange log, disabled by default, redacted on request.

Each successful call can be recorded as JSON: full prompt, full response, tokens,
simulated timestamp. It is useful for debugging, calibration and cost auditing; it is also
a file of **potential personal data** when the items describe people.
Hence three rules: nothing is written without `telemetry.exchanges_enabled`, a redactor can
transform each record before writing, and the file rotates by size.

The format is the one read by the repository tools (`make report`, dashboard): one indented
JSON object per record, separated by a blank line. Do not change it without them.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ExchangeRecord:
    task_id: str
    provider: str
    category: str
    tokens_in: int
    tokens_out: int
    messages: list[dict[str, Any]]
    response: Any
    sim_ts: float | None = None
    origine: str | None = None
    time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def sim_day(self) -> str | None:
        # `tz=UTC` on a GAMA clock timestamp gives the simulation's WALL-CLOCK day,
        # regardless of the process `TZ` (definition of `services/llm-agents/sim_clock.wall_clock`).
        return datetime.fromtimestamp(self.sim_ts, tz=UTC).strftime("%Y-%m-%d") if self.sim_ts else None

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {
            "time": d["time"], "sim_ts": d["sim_ts"], "sim_day": self.sim_day, "origine": d["origine"],
            "task_id": d["task_id"], "provider": d["provider"], "category": d["category"],
            "tokens_in": d["tokens_in"], "tokens_out": d["tokens_out"],
            "messages": d["messages"], "response": d["response"],
        }


Redactor = Callable[[ExchangeRecord], ExchangeRecord]


def identity(record: ExchangeRecord) -> ExchangeRecord:
    return record


class FieldRedactor:
    """Masks (`mode="mask"`) or hashes (`mode="hash"`) fields in the messages and the response.

    `fields`: names of JSON keys to redact wherever they appear (recursive), for example
    ``("perception", "history", "agent_id")``. The text content of the messages is redacted
    entirely if ``"content"`` is in `fields`.
    """

    def __init__(self, fields: Iterable[str], mode: str = "mask") -> None:
        self._fields = set(fields)
        if mode not in ("mask", "hash"):
            raise ValueError(f"unknown redaction mode: {mode!r} (mask or hash)")
        self._mode = mode

    def _redact_value(self, value: Any) -> Any:
        if self._mode == "hash":
            return "sha256:" + hashlib.sha256(json.dumps(value, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
        return "***"

    def _walk(self, node: Any) -> Any:
        if isinstance(node, dict):
            return {k: (self._redact_value(v) if k in self._fields else self._walk(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [self._walk(x) for x in node]
        return node

    def __call__(self, record: ExchangeRecord) -> ExchangeRecord:
        record.messages = self._walk(record.messages)
        record.response = self._walk(record.response)
        return record


class TruncateRedactor:
    """Cuts the message text and the serialized response beyond `max_chars`."""

    def __init__(self, max_chars: int = 2000) -> None:
        self._max = int(max_chars)

    def __call__(self, record: ExchangeRecord) -> ExchangeRecord:
        for m in record.messages:
            content = m.get("content")
            if isinstance(content, str) and len(content) > self._max:
                m["content"] = content[: self._max] + f"… [{len(content) - self._max} caractères coupés]"
        serialized = json.dumps(record.response, ensure_ascii=False, default=str)
        if len(serialized) > self._max:
            record.response = {"_tronque": serialized[: self._max], "_caracteres_coupes": len(serialized) - self._max}
        return record


def load_redactor(spec: str | None) -> Redactor:
    """Resolves a redactor from a dotted path (`package.module:attribute` or `package.module.attribute`).

    The attribute can be a redactor (callable) or a redactor class without arguments.
    `None` or empty = identity.
    """
    if not spec:
        return identity
    module_name, _, attr = spec.replace(":", ".").rpartition(".")
    if not module_name:
        raise ValueError(f"redactor {spec!r}: expected 'module.attribute' or 'module:attribute'")
    target = getattr(importlib.import_module(module_name), attr)
    if isinstance(target, type):
        target = target()
    if not callable(target):
        raise TypeError(f"redactor {spec!r}: {attr} is not callable")
    return target


class ExchangeJournal:
    """Writes the redacted records to `path`, with rotation by size."""

    def __init__(self, path: Path, *, redactor: Redactor = identity, max_bytes: int = 200_000_000) -> None:
        self.path = Path(path)
        self._redactor = redactor
        self._max_bytes = int(max_bytes)

    def _rotate_if_needed(self) -> None:
        try:
            if self._max_bytes > 0 and self.path.is_file() and self.path.stat().st_size >= self._max_bytes:
                rotated = self.path.with_name(self.path.name + ".1")
                os.replace(self.path, rotated)
                logger.info(f"Exchange log rotated ({self._max_bytes} bytes reached) → {rotated}")
        except OSError as e:
            logger.warning(f"Exchange log rotation failed: {e}")

    def write(self, record: ExchangeRecord) -> None:
        record = self._redactor(record)
        self._rotate_if_needed()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_json_dict(), ensure_ascii=False, default=str, indent=2) + "\n")
        except OSError as e:
            logger.warning(f"Cannot write to {self.path}: {e}")


__all__ = [
    "ExchangeJournal",
    "ExchangeRecord",
    "FieldRedactor",
    "Redactor",
    "TruncateRedactor",
    "identity",
    "load_redactor",
]
