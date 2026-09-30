"""Exact memoisation of STM/LTM reflections (ticket 012).

A reflection LLM call is a pure function of its effective prompt: same
agent, same identity, same experience, same guidelines ⇒ same introspection. On
deterministic re-runs (decisions served by the cache, seeded draws, replayed
weather), these prompts are byte-identical from one run to the next — paying for them again
wastes quota. This store memoises them by exact SHA-256 fingerprint.

What this store IS NOT: a similarity cache. No semantic
branch, no threshold, no reuse across agents or across different
experiences — the slightest differing byte is a miss. Serving one agent
another's introspection would be a scientific degradation (doctrine
"no degraded mode", docs/arch/cache-memory.md).

Location: `reflections.sqlite` in the SAME directory as the decision
cache (`<cache_dir>/<prompt_checksum>/<population>/`) — invalidation on a
system prompt change is inherited from the directory checksum, as for
decisions.

The model is NOT part of the key (amendment to D2, ticket 012): the
multi-provider cascade routes dynamically, the model is not known at
lookup. The provider that actually produced the reflection is kept in the
VALUE (audit). Replaying the stored reflection is in fact more deterministic
than asking live again, where the provider roulette would change the pen.

Concurrency: SQLite in WAL + process lock. Methods are synchronous —
call them via `asyncio.to_thread` from the event loop (same convention as
`LlmSemanticCache`).
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from prometheus_client import Counter

REFLECTION_MEMO = Counter(
    "agent_reflection_memo_total",
    "Memoisation of STM/LTM reflections: hits, misses and stores, per category",
    ["category", "event"],  # event ∈ hit | miss | store | store_refused
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    key         TEXT PRIMARY KEY,
    person_id   TEXT NOT NULL,
    category    TEXT NOT NULL,
    reflection  TEXT NOT NULL,
    concepts    TEXT NOT NULL,   -- JSON (liste), '[]' si aucune
    provider    TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_reflections_person ON reflections(person_id);
"""


class ReflectionMemoStore:
    """Exact memoisation store of reflections (key = SHA-256 of the effective prompt)."""

    def __init__(self, cache_dir: str):
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path / "reflections.sqlite")
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info(f"ReflectionMemoStore initialised — {self._db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------ key

    @staticmethod
    def make_key(
        person_id: str,
        category: str,
        identity: str,
        context_text: str,
        guidelines: str = "",
        departure_timestamp: float = 0.0,
        llm_params: Optional[dict] = None,
        schema_version: int = 1,
    ) -> str:
        """Exact fingerprint of the effective reflection prompt.

        Everything that reaches the LLM goes into the key: identity, experience, guidelines,
        departure timestamp (interpolated in the rendering — deterministic on re-run) and
        generation parameters (temperature changes the pen). The version of the
        system prompt is not in it: it already isolates the store's DIRECTORY
        (checksum, cf. llm_agent.py).

        `schema_version` (ticket 071, lot 1) also goes into the key. The output schema of
        the reflection changed — each concept now carries a severity level and a
        valence — so a response memoised under the old schema does not contain what
        the code now expects. Serving it would produce zero-severity concepts, without
        any error showing: a zero severity is exactly that of a perfect
        trip. The version therefore makes the cache MISS rather than serve wrongly. The
        cost is accepted: the reflection cache accumulated under the old schema is lost.
        """
        material = json.dumps(
            {
                "person_id": str(person_id),
                "category": category,
                "identity": identity,
                "context": context_text,
                "guidelines": guidelines,
                "departure_timestamp": departure_timestamp,
                "llm_params": llm_params or {},
                "schema_version": int(schema_version),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------- read

    def lookup(self, key: str, category: str) -> Optional[dict]:
        """Returns {reflection, concepts, provider} if the exact prompt was already paid for."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT reflection, concepts, provider FROM reflections WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            REFLECTION_MEMO.labels(category=category, event="miss").inc()
            return None
        REFLECTION_MEMO.labels(category=category, event="hit").inc()
        return {
            "reflection": row[0],
            "concepts": json.loads(row[1]),
            "provider": row[2],
        }

    # -------------------------------------------------------------- write

    def store(
        self,
        key: str,
        person_id: str,
        category: str,
        reflection: str,
        concepts: Optional[list] = None,
        provider: str = "",
    ) -> bool:
        """Persists a reflection actually produced. Refuses the empty one (D3).

        A reflection that is empty AND without concept is a generation failure, not an
        introspection: persisting it would serve nothingness to re-runs (same principle
        as refusing uniform fallbacks in the decision cache).
        """
        reflection = (reflection or "").strip()
        concepts = concepts or []
        if not reflection and not concepts:
            REFLECTION_MEMO.labels(category=category, event="store_refused").inc()
            logger.info(
                f"[reflection-memo] store refused — empty reflection not persisted | "
                f"person={person_id} category={category}"
            )
            return False
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reflections "
                "(key, person_id, category, reflection, concepts, provider) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, str(person_id), category, reflection,
                 json.dumps(concepts, ensure_ascii=False), provider),
            )
            conn.commit()
        REFLECTION_MEMO.labels(category=category, event="store").inc()
        return True

    # ------------------------------------------------------------ diagnostic

    def stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
            by_cat = dict(conn.execute(
                "SELECT category, COUNT(*) FROM reflections GROUP BY category"
            ).fetchall())
        return {"total": total, "by_category": by_cat}
