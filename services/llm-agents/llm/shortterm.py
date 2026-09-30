from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from llama_index.core.llms import ChatMessage

# Embedding imports
from loguru import logger

from llm.memory import MemoryEntry, MemoryType


class UserShortTermMemory:
    """Manages short-term conversational memory for a specific user"""
    
    def __init__(self, person_id: str):
        self.person_id = person_id
        self.recent_entries: List[MemoryEntry] = []
        self.max_entries = 100
        self.last_activity = datetime.now()
    
    def add_message(
        self,
        msg: str,
        timestamp: Optional[datetime] = None,
        activity_id: Optional[str] = None,
        importance: float = 0.0,
        axes: Optional[dict] = None,
        valence: str = "neutre",
        origine: Optional[str] = None,
    ):
        """Add a chat message to short-term memory.

        `importance` (ticket 071, lot 1) is the DETERMINISTIC severity of the entry, computed
        from what the simulation measured: delay suffered, missed connection, constrained
        mode. It costs no model call and it is objective.

        It serves two purposes, both downstream: deciding whether the agent's day
        is a BREAK that triggers an immediate consolidation, and providing the
        FLOOR that the model's judgement cannot go below during reflection.
        """
        self.last_activity = datetime.now()

        #logger.info(f"User {self.person_id} added message at {self.last_activity} for activity: {activity_id}")

        # Create memory entry
        entry = MemoryEntry(
            content=msg,
            timestamp=timestamp or datetime.now(),
            memory_type=MemoryType.CONVERSATION,
            person_id=self.person_id,
            activity_id=activity_id,
            importance=float(importance or 0.0),
            # Ticket 100 — valence and provenance travel with the entry. No parameter
            # carried them: the `MemoryEntry` declared them, and nobody filled them in
            # on this path.
            valence=str(valence or "neutre"),
            origine=origine,
            # Axes are normalised AT WRITE TIME (ticket 071, lot 2): this is where we know
            # which mode was chosen, under which weather and for which purpose. Recomputing
            # them at recall would redo the same work at every decision, on the critical path —
            # and two spellings of the same line would never meet.
            **{k: v for k, v in (axes or {}).items() if k.startswith("axe_")},
        )
        
        self.recent_entries.append(entry)
        
        # Keep only recent entries
        if len(self.recent_entries) > self.max_entries:
            self.recent_entries = self.recent_entries[-self.max_entries:]
    
    def get_all(self) -> List[ChatMessage]:
        """Get all messages from short-term memory"""
        return [entry.content for entry in self.recent_entries]

    def get_all_messages(self) -> List[MemoryEntry]:
        """Get all memory entries"""
        return self.recent_entries

    def get_all_message_and_group(self) -> Tuple[List[List[MemoryEntry]], List[MemoryEntry]]:
        all_entries = self.get_all_messages()
        # for loop to reserve the order
        results = []
        buffer = []
        for entry in all_entries:
            if buffer and entry.activity_id != buffer[-1].activity_id:
                results.append(buffer)
                buffer = []
            if entry.activity_id:
                buffer.append(entry)
            else:
                results.append([entry])
        if buffer:
            results.append(buffer)
        return results, all_entries

    def get_recent_entries(self, hours: int = 24) -> List[MemoryEntry]:
        """Get recent memory entries within specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [entry for entry in self.recent_entries if entry.timestamp > cutoff]
    
    def gravite_cumulee(self) -> float:
        """Sum of the deterministic severities in the buffer (ticket 071, lot 1).

        This is the trigger mechanism of Park et al. (2023, § 4.2), which replaces a
        count of entries with a sum of importances. A difference worth stating: for them the
        threshold is crossed two or three times a day, a routine regime; here it is
        EXCEPTIONAL by construction of the threshold, the base consolidation remaining the
        daily floor at 22:00.
        """
        return float(sum(float(e.importance or 0.0) for e in self.recent_entries))

    def gravite_maximale(self) -> float:
        """The highest severity in the buffer — the floor of the maximum rule."""
        return max((float(e.importance or 0.0) for e in self.recent_entries), default=0.0)

    def clear(self):
        """Clear short-term memory"""
        self.recent_entries.clear()

    def remove_batch(self, entries: List[MemoryEntry]):
        """Remove a batch of entries from short-term memory"""
        self.recent_entries = [entry for entry in self.recent_entries if entry not in entries]
