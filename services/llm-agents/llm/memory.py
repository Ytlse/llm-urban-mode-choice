from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Dict, Optional

# LLM imports
from enum import Enum

class MemoryType(Enum):
    CONVERSATION = "conversation"
    REFLECTION = "reflection"
    CONCEPT = "concept"
    SUMMARY = "summary"

    def __str__(self):
        return str(self.value)

# Declarative memory registers (Tulving, 1972; Sumers et al., 2024 for language agents).
# The two do not share the same erasure dynamics: episodic memory fades with TIME,
# semantic memory with CONTRADICTION (lot 3). That a line is saturated on rainy days does not
# become false because ten days have passed.
TYPES_EPISODIQUES = (MemoryType.CONVERSATION, MemoryType.REFLECTION)
TYPES_SEMANTIQUES = (MemoryType.CONCEPT, MemoryType.SUMMARY)


@dataclass
class MemoryEntry:
    """Represents a memory entry with metadata"""
    content: str
    timestamp: datetime
    memory_type: MemoryType
    person_id: str
    activity_id: Optional[str] = None
    tags: Optional[str] = ""
    # Identifier of the document in the vector index. Set at write time by
    # `MultiUserLongTermMemory.aadd_memory`, it makes the entry ADDRESSABLE for
    # deletion: without it, an entry removed from the metadata would stay in
    # the index and keep being served again to the model (ticket 071, defect B).
    # `None` for entries written before this ticket, which cannot be deleted.
    doc_id: Optional[str] = None

    # ── Memory qualification (ticket 071, lot 1) ─────────────────────────────────
    # All these fields have a DEFAULT, and not for convenience: an entry written
    # before lot 1 must reload without an exception, and an entry written afterwards must
    # stay readable by code that ignores these fields. The reversibility of the ticket
    # depends on it.

    # Severity on [0, 1]. Decides the lifetime, and the weight at recall (lot 2).
    importance: float = 0.0
    # Separates a happy salient memory from a suffered salient memory.
    valence: str = "neutre"
    # ── Provenance (ticket 100, D2) ──────────────────────────────────────────────
    # Where what the entry tells comes from: `vecu` (the agent did or suffered it), `lu` (it
    # read it, press channel), `entendu` (another member of its household told it).
    #
    # It carries decision D2 — **a single hop**. What is heard never travels on: a
    # belief born of hearsay is indistinguishable from a belief born of a trip without this
    # field, and there is no shortcut. Ticket 078 § 4.1 refused this genealogy;
    # it becomes necessary as soon as circulation is limited to one hop.
    #
    # `None` = entry written BEFORE this ticket. It is READ as experienced (`origine_effective`)
    # for lack of anything better, but it does not DECLARE it: confusing the two would pass off
    # as a measurement what is only a default, and this repository has already paid for that.
    origine: Optional[str] = None
    # The four axes, NORMALISED AT WRITE TIME and never at read time (lot 2): without this,
    # two spellings of the same bus line never meet. An unresolved axis is
    # `None`, treated as an absence of match and not as a match
    # with everything.
    axe_objet: Optional[str] = None
    axe_lieu: Optional[str] = None
    axe_creneau: Optional[str] = None
    axe_motif: Optional[str] = None
    # Fifth axis, added in lot 2: the day's weather, in five values (`sec`, `pluie`,
    # `neige`, `canicule`, `froid`). It is stored nowhere else — it is
    # recomputed on the fly when the prompt is built — and the ranking needs it to pair
    # a rain memory with a decision taken in the rain.
    axe_meteo: Optional[str] = None
    # Forgetting time constant, in DAYS. `None` = entry never qualified: recall
    # then falls back on the default constant rather than giving it a zero weight.
    force: Optional[float] = None
    rappels: int = 0
    # Last moment the memory was SERVED to the model, in simulated time. Distinct from
    # `timestamp`, which remains the time of the EVENT: the latter is shown in the prompt and
    # is the left side of the per-day and per-age filters. Sliding it at recall
    # would rewrite the agent's history — it would believe its bike fall happened yesterday.
    # `None` = never recalled: decay then starts from the write.
    dernier_rappel: Optional[datetime] = None

    # ── Concepts only (lot 3) ────────────────────────────────────────────────────
    # Carried by entries of type `concept`. Null elsewhere.
    panier: Optional[str] = None
    observations: int = 0
    contre_exemples: int = 0
    # Date on which an outdated concept was SET ASIDE, in simulated time. Never a deletion:
    # this dated setting-aside IS the readable trace of a habit change, and it is
    # the observable the hysteresis experiment looks for.
    depasse_le: Optional[str] = None
    # Last time this concept was CONFIRMED or REFINED, in simulated time.
    # ⚠ A field distinct from `timestamp`, and not a duplicate: the specification asked
    # to "refresh the timestamp" at each confirmation, which lot 1 forbids — the
    # `timestamp` is the time of the EVENT, it is shown in the prompt, and sliding it
    # would rewrite the agent's history. This one only breaks ties between two concepts of
    # equal confidence, and touches neither the prompt text nor the age filters.
    derniere_observation: Optional[datetime] = None

    # Version of the entry schema. Used to tell apart, at reload, an entry
    # never qualified (severity truly unknown) from an entry qualified at zero
    # (a trip that went well). Both carry `importance = 0.0`.
    schema_version: int = 1

    @property
    def origine_effective(self) -> str:
        """The provenance, with its default reading for entries prior to ticket 100."""
        return self.origine or "vecu"

    @property
    def est_episodique(self) -> bool:
        """Time-based forgetting concerns ONLY episodic memory (lot 1); the rest is lot 3."""
        return self.memory_type in TYPES_EPISODIQUES

    @property
    def confiance(self) -> float:
        """Laplace's rule of succession: `(obs + 1) / (obs + contre_ex + 2)`.

        For a concept, it REPLACES the clock decay in the ranking (lot 3). A
        never-contradicted concept keeps its weight; a contradicted concept loses it, whatever
        its age.
        """
        from llm.concepts import confiance as _confiance

        return _confiance(self.observations, self.contre_exemples)

    @property
    def est_servi(self) -> bool:
        """Can this memory be served to the model?

        False for a concept contradicted more often than confirmed. It stays in the
        metadata: it is not deleted, it is set aside.
        """
        if self.est_episodique:
            return True
        from llm.concepts import est_hors_service

        return not est_hors_service(self.observations, self.contre_exemples)

    @property
    def est_depasse(self) -> bool:
        """At least three contradictions AND more contradictions than confirmations."""
        if self.est_episodique:
            return False
        from llm.concepts import est_depasse as _est_depasse

        return _est_depasse(self.observations, self.contre_exemples)

    @property
    def horodatage_de_reference(self) -> datetime:
        """Where decay starts from: the last recall, failing that the write."""
        return self.dernier_rappel or self.timestamp

    def to_dict(self) -> Dict:
        # JSON-safe: timestamps in ISO and memory_type as a str value (round-trip with from_dict)
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'memory_type': str(self.memory_type),
            'dernier_rappel': self.dernier_rappel.isoformat() if self.dernier_rappel else None,
            'derniere_observation': (
                self.derniere_observation.isoformat() if self.derniere_observation else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        # Defensive copy: `from_dict` must not modify the caller's dictionary.
        data = dict(data)
        try:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        except Exception as e:
            print(f"Error parsing timestamp: {e}, data: {data}")
            raise e
        if data.get('derniere_observation'):
            try:
                data['derniere_observation'] = datetime.fromisoformat(
                    data['derniere_observation']
                )
            except (TypeError, ValueError):
                data['derniere_observation'] = None
        if data.get('dernier_rappel'):
            try:
                data['dernier_rappel'] = datetime.fromisoformat(data['dernier_rappel'])
            except (TypeError, ValueError):
                # An unreadable last recall must not cost the memory: we
                # fall back on the write, which only makes the entry OLDER — never
                # the reverse, which would wrongly make it younger.
                data['dernier_rappel'] = None
        if isinstance(data.get('memory_type'), str):
            data['memory_type'] = MemoryType(data['memory_type'])
        # Ticket 071 (lot 1) — UNKNOWN keys are ignored instead of blowing up
        # `cls(**data)`. Without this, rolling back this ticket would make unreadable
        # all the metadata written in the meantime: a field removed from the code would be enough
        # to lose the memory of a thousand agents. It is a reversibility guarantee.
        connus = {f.name for f in fields(cls)}
        inconnus = set(data) - connus
        if inconnus:
            from loguru import logger
            logger.debug(
                f"[memory] unknown fields ignored at reload: {sorted(inconnus)} "
                f"(entry written by another version of the schema)"
            )
        return cls(**{k: v for k, v in data.items() if k in connus})

    def __str__(self) -> str:
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"[{timestamp_str}]: {self.content}"
