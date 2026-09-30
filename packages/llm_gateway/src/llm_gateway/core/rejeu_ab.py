"""Exact-prompt replay — a response already obtained is served again until the prompt changes.

The two arms of a memory experiment (the treated one, then the control) ask the same questions
to the same model until the event. At temperature 0, gemini-3.5-flash-lite still returns
different responses to identical prompts. The arms diverged from the first evening (a09 V2,
2026-09-25), and the measured gap mixed the effect of the event with the model's noise.

Each task served by a provider is recorded under the key of ITS exact prompt: category,
messages rendered for this task alone, parameters, routing constraints. Recording happens in
a space named by the caller (`LLMRequest.espace_rejeu`). A task with the same key in the
same space receives the recorded response, without a call. A request without a space keeps
exactly the previous behaviour.

The key is the task's, not the batch's: micro-batching merges tasks by their order of
arrival, which differs from one arm to the other.

Storage: one JSON file per key, `<racine>/<espace>/<clé>.json`, written by atomic rename.
The API reads, the worker writes, on the same mount: no lock is needed, and a space
can be inspected or archived like a directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)

_ESPACE_VALIDE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def espace_valide(espace: str | None) -> str | None:
    """The space as is if it can name a directory without escaping it, None otherwise."""
    if espace and _ESPACE_VALIDE.match(espace) and ".." not in espace:
        return espace
    return None


def cle_rejeu(request: Any, messages: list[Any]) -> str:
    """Fingerprint of the exact prompt of ONE task.

    Included: the category, the rendered messages (everything the adapter sends), the
    parameters (temperature, budget, variant…), the pinned provider, the allowed instances
    (sorted: it is a set) and the context. Not included: the origin (it names the arm,
    which differs by construction), the task identifier, the space itself.
    """
    corps = {
        "category": request.category,
        "messages": [
            m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in messages
        ],
        "parameters": request.parameters,
        "force_provider": request.force_provider,
        "instances_admises": sorted(set(request.instances_admises or [])),
        "context": request.context,
    }
    brut = json.dumps(corps, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(brut.encode("utf-8")).hexdigest()


class MagasinRejeu:
    """Recorded responses, one file per key and per space."""

    def __init__(self, racine: Path):
        self.racine = Path(racine)

    def _chemin(self, espace: str, cle: str) -> Path:
        return self.racine / espace / f"{cle}.json"

    def lire(self, espace: str, cle: str) -> dict[str, Any] | None:
        chemin = self._chemin(espace, cle)
        try:
            enregistrement = json.loads(chemin.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            # Unreadable = missing: the task goes to the provider, nothing is served wrongly.
            logger.error(
                f"[ALARME] Replay: unreadable record, task sent to the provider | "
                f"espace={espace} cle={cle[:12]} erreur={exc!r}"
            )
            return None
        if not isinstance(enregistrement, dict):
            # Valid JSON but not a record (a list, a number…): same rule as unreadable.
            logger.error(
                f"[ALARME] Replay: record is not a JSON object, task sent to the provider | "
                f"espace={espace} cle={cle[:12]} type={type(enregistrement).__name__}"
            )
            return None
        return enregistrement

    def ecrire(self, espace: str, cle: str, enregistrement: dict[str, Any]) -> bool:
        """Records a response. True if it is new; the first one recorded is authoritative."""
        chemin = self._chemin(espace, cle)
        if chemin.exists():
            return False
        chemin.parent.mkdir(parents=True, exist_ok=True)
        provisoire = chemin.with_suffix(f".{os.getpid()}.{id(enregistrement)}.tmp")
        provisoire.write_text(
            json.dumps(enregistrement, ensure_ascii=False, default=str), encoding="utf-8"
        )
        os.replace(provisoire, chemin)
        return True


@lru_cache(maxsize=4)
def _magasin(racine: str) -> MagasinRejeu:
    return MagasinRejeu(Path(racine))


def magasin_rejeu(settings: Any) -> MagasinRejeu | None:
    """The store declared by `rejeu.dir`, or None if replay is not configured."""
    rejeu = getattr(settings, "rejeu", None)
    racine = getattr(rejeu, "dir", None) if rejeu is not None else None
    return _magasin(str(racine)) if racine else None


def enregistrement(
    *, cle: str, espace: str, request: Any, provider: str, agents: list[Any],
    tokens_in: int, tokens_out: int, task_id: str,
) -> dict[str, Any]:
    """What is recorded for a served task: enough to serve it again and know where it came from."""
    return {
        "cle": cle,
        "espace": espace,
        "category": request.category,
        "provider": provider,
        "agents": [a.model_dump() if hasattr(a, "model_dump") else a for a in agents],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "task_id": task_id,
        "origine": request.origine,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


__all__ = ["MagasinRejeu", "cle_rejeu", "enregistrement", "espace_valide", "magasin_rejeu"]
