"""infra/memory/learned_limits.py — learned limits in memory, or in a JSON file."""
from __future__ import annotations

import json
import os
from pathlib import Path


class InMemoryLearnedLimits:
    def __init__(self) -> None:
        self._limits: dict[str, int] = {}

    def get_max_output_tokens(self, provider: str) -> int | None:
        return self._limits.get(provider)

    def set_max_output_tokens(self, provider: str, limit: int) -> None:
        self._limits[provider] = int(limit)

    def all_max_output_tokens(self) -> dict[str, int]:
        return dict(self._limits)


class FileLearnedLimits:
    """A JSON file `{"max_output_tokens": {provider: limit}}`, atomic write.

    For the Redis-less mode (tests, embedded executor). A missing file = nothing learned.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def _read(self) -> dict[str, int]:
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        raw = data.get("max_output_tokens", {}) if isinstance(data, dict) else {}
        return {str(k): int(v) for k, v in raw.items()}

    def get_max_output_tokens(self, provider: str) -> int | None:
        return self._read().get(provider)

    def set_max_output_tokens(self, provider: str, limit: int) -> None:
        limits = self._read()
        limits[provider] = int(limit)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps({"max_output_tokens": limits}, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)

    def all_max_output_tokens(self) -> dict[str, int]:
        return self._read()


__all__ = ["FileLearnedLimits", "InMemoryLearnedLimits"]
