"""Resolution and probing of the synthesis page sources.

A source is described in ``sources.yaml`` then probed: present or not,
fingerprint, date. Nothing fails here — a missing source becomes a
« Données manquantes » card in the page.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fingerprint: capped, a 6,000-line moves.csv does not need to be read in
# full to be identified stably between two regenerations.
_HASH_MAX_BYTES = 4 * 1024 * 1024


@dataclass
class Source:
    """One manifest entry, probed on disk."""

    id: str
    path: Path
    role: str = ""
    exists: bool = False
    is_dir: bool = False
    size: Optional[int] = None
    mtime: Optional[str] = None
    sha256: Optional[str] = None
    resolved: Optional[str] = None
    note: str = ""

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "path": self.rel, "role": self.role,
            "exists": self.exists, "is_dir": self.is_dir, "size": self.size,
            "mtime": self.mtime, "sha256": self.sha256,
            "resolved": self.resolved, "note": self.note,
        }


def probe(source_id: str, path: Path | str, role: str = "", note: str = "") -> Source:
    """Probes a path: existence, size, date, partial fingerprint."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    src = Source(id=source_id, path=p, role=role, note=note)
    if not p.exists():
        return src
    src.exists = True
    src.is_dir = p.is_dir()
    resolved = p.resolve()
    if resolved != p:
        try:
            src.resolved = str(resolved.relative_to(REPO_ROOT))
        except ValueError:
            src.resolved = str(resolved)
    stat = p.stat()
    src.mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds")
    if not src.is_dir:
        src.size = stat.st_size
        src.sha256 = _digest(p)
    return src


def _digest(path: Path) -> Optional[str]:
    h = hashlib.sha256()
    read = 0
    try:
        with path.open("rb") as fh:
            while read < _HASH_MAX_BYTES:
                chunk = fh.read(min(1 << 20, _HASH_MAX_BYTES - read))
                if not chunk:
                    break
                h.update(chunk)
                read += len(chunk)
    except OSError:
        return None
    return h.hexdigest()


@dataclass
class Manifest:
    """The loaded manifest, plus the register of probed sources."""

    raw: dict
    sources: dict[str, Source] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def track(self, source_id: str, path: Path | str, role: str = "",
              note: str = "") -> Source:
        src = probe(source_id, path, role, note)
        self.sources[source_id] = src
        return src

    def path_of(self, dotted: str) -> Optional[Path]:
        value = self.get(dotted)
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else REPO_ROOT / p


def load_manifest(path: Path | str | None = None) -> Manifest:
    """Loads ``sources.yaml`` (or an alternative manifest)."""
    p = Path(path) if path else Path(__file__).with_name("sources.yaml")
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open(encoding="utf-8") as fh:
        return Manifest(raw=yaml.safe_load(fh) or {})


# ── Access to the scoring formula, and to the calibration engine ─────────────
#
# The scoring formula lives in this repository (`scripts/synthesis/formule_score`,
# repatriated on 2026-09-29): every score goes through `import_formule_score`.
# `import_calibration` remains for the private calibration tools only (store, datasets,
# lineage), which need the whole engine of the standalone prompt_calibration/ repository.
# `scripts/tests/test_formule_score_alignee.py` keeps both formulas identical.

def import_formule_score() -> tuple[Any | None, str]:
    """Imports the scoring formula (``formule_score``). Returns ``(module, error message)``.

    The module exposes ``metrics``, ``models`` and ``evaluation``, like the ``calibration``
    package it was extracted from: ``frames.Scorer`` accepts either.
    """
    try:
        from scripts.synthesis import formule_score
    except Exception as exc:  # noqa: BLE001 — dependency missing from the interpreter: a message, not a crash
        return None, (f"Import de la formule de score impossible : {exc}. "
                      "Utilisez un interpréteur disposant de pandas/numpy/pydantic "
                      "(par ex. services/llm-agents/.venv/bin/python).")
    return formule_score, ""


def import_calibration(repo: Path | str = "prompt_calibration") -> tuple[Optional[Any], str]:
    """Imports the ``calibration`` package. Returns ``(module, error message)``."""
    root = Path(repo)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not (root / "calibration" / "metrics.py").exists():
        return None, (f"Le dépôt de calibration est introuvable ({root}). "
                      "Le score ne peut pas être calculé : clonez-le à cet emplacement.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import calibration  # noqa: F401
        from calibration import metrics  # noqa: F401
    except Exception as exc:  # dependency missing from the current interpreter
        return None, (f"Import du moteur de calibration impossible : {exc}. "
                      "Utilisez un interpréteur disposant de pandas/numpy/pydantic "
                      "(par ex. services/llm-agents/.venv/bin/python).")
    return calibration, ""
