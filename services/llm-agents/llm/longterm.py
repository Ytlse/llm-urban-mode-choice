# scalable_memory.py - Scalable long-term memory system optimized for 1000+ users
import asyncio
import hashlib
import json
import string
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from prometheus_client import Histogram
from settings import settings
from sim_clock import gama_timestamp, wall_clock

# Default embedding model, inherited from the implementation of Vu et al. (2025). It is a
# Sentence-Transformers model (Reimers & Gurevych, 2019) trained on an English corpus
# according to its model card — consistent with the corpus since the English switch of ticket 074.
MODELE_PLONGEMENT_DEFAUT = "all-MiniLM-L6-v2"

LTM_QUERY_DURATION = Histogram(
    "ltm_query_duration_seconds",
    "Duration of aquery_user_memories calls (ChromaDB)",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
)


@dataclass
class MemorySearchResult:
    content: str
    metadata: dict
    score: float = 0.0


from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llm.axes import affinite_axes, affinite_meteo
from llm.trace_rappel import tracer_rappel
from llm.gravite import (
    est_purgeable,
    force_apres_rappel,
    force_initiale,
    poids_temporel,
)
from llm.journal_memoire import journal
from llm.memory import MemoryEntry


class VectorStoreFactory:
    """Factory for creating optimized vector stores"""

    @staticmethod
    def create_chroma_store(storage_dir: Path) -> BasePydanticVectorStore | None:
        """Create ChromaDB vector store with optimizations"""
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore

            chroma_client = chromadb.PersistentClient(
                path=str(storage_dir / "chroma_db"),
            )

            chroma_collection = chroma_client.get_or_create_collection(
                "memory_collection", metadata={"hnsw:space": "cosine"}
            )

            return ChromaVectorStore(chroma_collection=chroma_collection)

        except ImportError:
            logger.warning("ChromaDB not available, falling back to simple storage")
            return None

    # @staticmethod
    # def create_qdrant_store(storage_dir: Path) -> Optional[BasePydanticVectorStore]:
    #     """Create Qdrant vector store with optimizations"""
    #     try:
    #         import qdrant_client
    #         from llama_index.vector_stores.qdrant import QdrantVectorStore

    #         client = qdrant_client.QdrantClient(
    #             path=str(storage_dir / "qdrant_db"),
    #             grpc_port=6334,
    #             prefer_grpc=True
    #         )

    #         return QdrantVectorStore(
    #             client=client,
    #             collection_name="memory_collection",
    #             parallel=4
    #         )

    #     except ImportError:
    #         print("Qdrant not available, falling back to simple storage")
    #         return None

    # @staticmethod
    # def create_pinecone_store(config: Dict[str, Any]) -> Optional[BasePydanticVectorStore]:
    #     """Create Pinecone vector store"""
    #     try:
    #         import pinecone
    #         from llama_index.vector_stores.pinecone import PineconeVectorStore

    #         api_key = config.get("api_key")
    #         environment = config.get("environment")
    #         index_name = config.get("index_name", "memory-index")

    #         if api_key and environment:
    #             pinecone.init(api_key=api_key, environment=environment)
    #             return PineconeVectorStore(
    #                 pinecone_index=pinecone.Index(index_name)
    #             )

    #     except ImportError:
    #         print("Pinecone not available, falling back to simple storage")
    #         return None


class MultiUserLongTermMemory:
    def __init__(
        self,
        storage_dir: str = "/tmp/memory_storage",
        vector_store_type: str = "chroma",
        vector_store_config: dict = None,
        max_loaded_metadata: int = 2000,
        use_async: bool = False,
        long_term_memory_filter_by_datetime: bool = True,
    ):

        self.use_async = use_async
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.vector_store_type = vector_store_type
        self.vector_store_config = vector_store_config or {}
        self.max_loaded_metadata = max_loaded_metadata
        self.long_term_memory_filter_by_datetime = long_term_memory_filter_by_datetime

        # Shared vector store - KEY OPTIMIZATION
        self.vector_store = self._create_vector_store()
        self.shared_index = None

        # LRU cache for user metadata
        self.user_metadata: dict[str, dict[str, Any]] = {}
        self.metadata_access_times: dict[str, datetime] = {}

        # Deferred write of the metadata: modified agents are marked dirty
        # and flushed in bursts (debounce) instead of one disk rewrite per entry.
        self._dirty: set = set()
        self._flush_task: asyncio.Task | None = None

        # Performance metrics
        self.metrics = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_cleanups": 0,
        }

        self._init_shared_index(use_async=self.use_async)
        logger.info(
            f"Initialized scalable memory with {vector_store_type} vector store"
        )

    def _create_vector_store(self) -> BasePydanticVectorStore | None:
        """Create vector store based on type"""
        if self.vector_store_type == "chroma":
            return VectorStoreFactory.create_chroma_store(self.storage_dir)
        # elif self.vector_store_type == "qdrant":
        #     return VectorStoreFactory.create_qdrant_store(self.storage_dir)
        # elif self.vector_store_type == "pinecone":
        #     return VectorStoreFactory.create_pinecone_store(self.vector_store_config)
        else:
            return None  # Simple storage

    def _init_shared_index(self, use_async: bool = False):
        """Initialize single shared vector store index"""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Ticket 071, lot 2 — the embedding model comes from the PARAMETER. It was hard-coded
        # here although `settings.agent.embedding_model` existed: a parameter that controls
        # nothing is a configuration lie, and it prevented comparing two models without
        # touching the code.
        #
        # ⚠ The model does NOT CHANGE: the default remains the one in service. The move to a
        # French-language model, once planned, is abandoned — the setup switches to English
        # (ticket 074), and the model inherited from Vu et al. fits the corpus again.
        # Any later change would require a COMPLETE REBUILD of the index, the
        # vectors not being comparable from one model to another.
        modele = (
            settings.agent.embedding_model or ""
        ).strip() or MODELE_PLONGEMENT_DEFAUT
        if modele != MODELE_PLONGEMENT_DEFAUT:
            logger.warning(
                f"[ltm] NON-STANDARD embedding model: « {modele} » instead of "
                f"« {MODELE_PLONGEMENT_DEFAUT} ». The index must have been rebuilt with this "
                f"model, otherwise the similarities are meaningless."
            )
        else:
            logger.info(f"[ltm] embedding model: {modele}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=modele)
        Settings.llm = None

        if self.vector_store:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            try:
                self.shared_index = load_index_from_storage(
                    storage_context, use_async=use_async
                )
                logger.info("Loaded existing shared vector index")
            except:
                self.shared_index = VectorStoreIndex.from_documents(
                    [], storage_context=storage_context, use_async=use_async
                )
                logger.info("Created new shared vector index")
        else:
            # Fallback to simple index
            index_path = self.storage_dir / "shared_index"
            if index_path.exists():
                try:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(index_path)
                    )
                    self.shared_index = load_index_from_storage(
                        storage_context, use_async=use_async
                    )
                    logger.info("Loaded simple vector index")
                    return
                except Exception as e:
                    logger.warning(f"Simple vector index unreadable ({e}) — recreating")
            # No existing index (or unreadable index): start again from a fresh StorageContext
            storage_context = StorageContext.from_defaults()
            self.shared_index = VectorStoreIndex.from_documents(
                [], storage_context=storage_context, use_async=use_async
            )
            self._persist_shared_index()
            logger.info("Created new simple vector index")

    def _persist_shared_index(self):
        """Persist shared index (only for simple storage)"""
        if not self.vector_store:
            index_path = self.storage_dir / "shared_index"
            self.shared_index.storage_context.persist(persist_dir=str(index_path))

    def _get_user_metadata_path(self, person_id: str) -> Path:
        """Get metadata file path with sharding"""
        # Use sharding to avoid too many files in one directory
        # shard = abs(hash(person_id)) % 100
        id_bytes = str(person_id).encode("utf-8")
        hash_obj = hashlib.md5(id_bytes)
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        shard = hash_int % 100
        shard_dir = self.storage_dir / "user_metadata" / f"shard_{shard:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{person_id}.json"

    def _load_user_metadata(self, person_id: str) -> dict[str, Any]:
        """Load user metadata from disk"""
        metadata_path = self._get_user_metadata_path(person_id)

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    # Ignore non-dict entries (files written by the old buggy format
                    # that serialised MemoryEntry objects as strings via default=str)
                    metadata["entries"] = [
                        MemoryEntry.from_dict(entry)
                        for entry in metadata.get("entries", [])
                        if isinstance(entry, dict)
                    ]
                    self.metadata_access_times[person_id] = datetime.now()
                    self.metrics["cache_misses"] += 1
                    return metadata
            except Exception as e:
                logger.error(f"Error loading metadata for user {person_id}: {e}")

        # Default metadata for new user
        metadata = {
            "entries": [],
            "last_cleanup": None,
            "last_reflection": None,
            "person_id": person_id,
            "created_at": datetime.now().isoformat(),
            "memory_usage_mb": 0,
            "total_entries": 0,
        }
        self.metadata_access_times[person_id] = datetime.now()
        self.metrics["cache_misses"] += 1
        return metadata

    def _save_user_metadata(self, person_id: str):
        """Save user metadata to disk"""
        if person_id not in self.user_metadata:
            return

        metadata_path = self._get_user_metadata_path(person_id)

        try:
            metadata = self.user_metadata[person_id]
            metadata["total_entries"] = len(metadata["entries"])
            # Entries are MemoryEntry objects: explicit serialisation via to_dict().
            # (json.dumps(default=str) would write them as "[date]: content" strings,
            # unrecoverable by from_dict at reload → memory lost at restart.)
            serializable = {
                **metadata,
                "entries": [entry.to_dict() for entry in metadata["entries"]],
            }
            # Single serialisation: the memory_usage_mb written in the file is the one
            # from the previous save (a purely indicative value, one save behind).
            metadata_json = json.dumps(
                serializable, indent=2, default=str, ensure_ascii=False
            )
            metadata["memory_usage_mb"] = len(metadata_json.encode("utf-8")) / (
                1024 * 1024
            )

            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata_json)
            self.metadata_access_times[person_id] = datetime.now()

        except Exception as e:
            logger.error(f"Error saving metadata for user {person_id}: {e}")

    _FLUSH_DELAY_S = 30.0

    def _schedule_flush(self) -> None:
        """Schedules a deferred flush of the dirty metadata (a single task in flight)."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_delay())

    async def _flush_after_delay(self) -> None:
        await asyncio.sleep(self._FLUSH_DELAY_S)
        await self.aflush_dirty()

    async def aflush_dirty(self) -> None:
        """Writes to disk the metadata of all agents marked dirty (off the event loop)."""
        dirty = list(self._dirty)
        self._dirty.clear()
        for person_id in dirty:
            if person_id in self.user_metadata:
                await asyncio.to_thread(self._save_user_metadata, person_id)

    def _cleanup_metadata_cache(self):
        """LRU eviction for metadata cache"""
        if len(self.user_metadata) <= self.max_loaded_metadata:
            return

        # Sort by access time and remove oldest
        sorted_users = sorted(self.metadata_access_times.items(), key=lambda x: x[1])

        users_to_remove = len(self.user_metadata) - self.max_loaded_metadata
        removed_count = 0

        for person_id, _ in sorted_users[:users_to_remove]:
            if person_id in self.user_metadata:
                # Save before removing from cache (flush guaranteed even if a debounce was pending)
                self._save_user_metadata(person_id)
                self._dirty.discard(person_id)
                del self.user_metadata[person_id]
                if person_id in self.metadata_access_times:
                    del self.metadata_access_times[person_id]
                removed_count += 1

        # No gc.collect() here: called from the event loop via
        # ensure_user_initialized(), it froze the loop ~110 ms per eviction.
        self.metrics["memory_cleanups"] += 1

        if removed_count > 0:
            logger.info(
                f"Cleaned up metadata cache: removed {removed_count} users from memory"
            )

    def ensure_user_initialized(self, person_id: str):
        """Ensure user metadata is loaded with cache management"""
        if person_id not in self.user_metadata:
            self.user_metadata[person_id] = self._load_user_metadata(person_id)
            self._cleanup_metadata_cache()
        else:
            # Update access time for LRU
            self.metadata_access_times[person_id] = datetime.now()
            self.metrics["cache_hits"] += 1

    def has_memories(self, person_id: str) -> bool:
        """Does the agent have at least one long-term memory?

        Reads the LRU cache of the metadata (in RAM) — no query to the vector store.
        Lets the semantic LLM cache choose between the exact branch (empty memory)
        and the similarity branch (filled memory).
        """
        self.ensure_user_initialized(person_id)
        return bool(self.user_metadata[person_id]["entries"])

    def get_last_user_memories(
        self, person_id: str, from_date: datetime
    ) -> list[MemoryEntry]:
        """Get last user memories from a specific date"""
        self.ensure_user_initialized(person_id)
        # logger.debug(f"Retrieving memories for user {person_id} since {from_date}, data: {self.user_metadata[person_id]['entries'][::-1]}")
        return [
            entry
            for entry in self.user_metadata[person_id]["entries"]
            if entry.timestamp >= from_date
        ]

    async def aadd_memory(self, entry: MemoryEntry):
        """Add memory to shared vector store with user namespace"""
        person_id = entry.person_id
        self.ensure_user_initialized(person_id)

        # Create document with namespace for user isolation
        # Ticket 071 (defect B) — the identifier derived from the LENGTH of the list. After a
        # cleanup the list gets shorter, and the following identifiers collided
        # with those already indexed. The counter is now monotonic and persisted.
        doc_index = self.user_metadata[person_id].get("next_doc_index")
        if doc_index is None:  # metadata from before ticket 071
            doc_index = len(self.user_metadata[person_id]["entries"])
        self.user_metadata[person_id]["next_doc_index"] = doc_index + 1
        doc_id = f"{person_id}_{doc_index}"

        # Ticket 071, lot 1 — the lifetime is fixed AT WRITE TIME, from severity.
        # `force = min(S0 × (1 + k × I), FORCE_MAX)`, ceiling included: no entry starts
        # beyond it, even when S0 triples. An already qualified entry (replayed reflection, run
        # resume) keeps its own.
        if entry.force is None:
            entry.force = force_initiale(entry.importance)

        doc = Document(
            # `id_` makes the document addressable for deletion: without it, an entry
            # removed from the metadata would stay in the vector index indefinitely.
            id_=doc_id,
            text=str(entry.content),
            metadata={
                "person_id": person_id,
                "timestamp": entry.timestamp.isoformat(),
                "memory_type": entry.memory_type,
                "namespace": f"user_{person_id}",  # Key for isolation
                "doc_id": doc_id,
                "tags": entry.tags,
                # Memory qualification (lot 1). It is copied here so that pools
                # B and C of lot 2 can filter without embedding. ⚠ It is a SNAPSHOT frozen at
                # write time: `force` and `rappels` evolve afterwards, and the authority on
                # these two fields remains the agent's metadata, never this copy. The ranking
                # therefore reads the entry, not this dictionary (cf. `rank_nodes`).
                "importance": float(entry.importance or 0.0),
                "axe_objet": entry.axe_objet or "",
                "axe_lieu": entry.axe_lieu or "",
                "axe_creneau": entry.axe_creneau or "",
                "axe_motif": entry.axe_motif or "",
                "valence": entry.valence or "neutre",
            },
        )

        # Add to shared index
        await self.shared_index.ainsert(doc)

        # Update user metadata
        entry.doc_id = doc_id
        self.user_metadata[person_id]["entries"].append(entry)
        # Ticket 075 — readable trace of the write. `journal()` returns `None` when the journal
        # is off: neither a file nor a disk call on the path of a decision.
        _journal = journal()
        if _journal is not None:
            _journal.ecriture(person_id, entry.timestamp, entry)
        # logger.debug(f"Add memory entry for user {person_id}: {entry.to_dict()}")

        # Memory limits per user
        if len(self.user_metadata[person_id]["entries"]) > 10000:
            logger.warning(f"User {person_id} exceeds memory limit, triggering cleanup")
            self.cleanup_user_memories(person_id, days_threshold=7)

        # Deferred write: reflections arrive in bursts (several entries per
        # agent) — the debounce groups each burst into a single disk write,
        # run off the event loop. Flush also guaranteed at LRU eviction.
        self._dirty.add(person_id)
        self._schedule_flush()

        # Periodic persistence for simple storage
        if (
            not self.vector_store
            and len(self.user_metadata[person_id]["entries"]) % 10 == 0
        ):
            self._persist_shared_index()

    def _filter_memory_by_working_day(
        self, message_datetime: datetime, search_datetime: datetime
    ) -> bool:
        # Filter by working day first
        # Convert search_time (seconds since epoch) to day of week
        search_day_of_week = search_datetime.weekday()  # 0=Monday, 6=Sunday
        entry_day_of_week = message_datetime.weekday()
        if (search_day_of_week < 5 and entry_day_of_week >= 5) or (
            search_day_of_week >= 5 and entry_day_of_week < 5
        ):
            return False

        return True

    def _filter_memory_by_peak_time(
        self, message_datetime: datetime, search_datetime: datetime
    ) -> bool:
        # TODO: Because we do reflection in batch, so the time could be wrong
        # If we do reflection for every entry (arrival), we can filter by peak time

        # Filter by peak time in day
        # search_time_label = time_to_bucket_text(search_datetime.timestamp())
        # entry_time_label = time_to_bucket_text(message_datetime.timestamp())
        # return search_time_label == entry_time_label
        # TODO: For now, we assume all memories are valid
        # We just need a lot of memories to make the model faster converge
        return True

    def _filter_memory_by_past_days(
        self, message_datetime: datetime, search_datetime: datetime, max_past_days: int
    ) -> bool:
        # Filter by past days
        if max_past_days < 0:
            return True

        delta_days = (search_datetime - message_datetime).days
        return delta_days <= max_past_days

    async def aquery_user_memories(
        self,
        person_id: str,
        query: str,
        top_k: int = 8,
        max_past_days: int = 30,
        query_at: int | None = None,
        modes_offerts: list[str] | None = None,
        contexte: dict[str, Any] | None = None,
    ) -> list[MemorySearchResult]:
        """Query memories with namespace filtering"""
        _t0 = time.monotonic()
        self.ensure_user_initialized(person_id)
        self.metrics["queries"] += 1
        # GAMA's WALL-CLOCK time: the right side of the filters must be compared with the
        # `datetime` of the memories, which also carry wall-clock fields
        # (`llm_agent.add_short_term_memory`). Read in the process time zone, it
        # shifted by one hour the weekday and the age of a late-evening
        # memory.
        query_at_datetime = wall_clock(query_at) if query_at else None

        logger.debug(
            f"Querying user long term memories for person {person_id}, at {query_at}"
        )

        def filter_message(metadata: dict) -> bool:
            # Ticket 071 (defect C) — the two filters ADD UP. Before, the filter by working
            # day and time slot returned immediately, so that the age window
            # (`max_past_days`) was never applied when the option was active: a
            # memory three years old in simulated time got through, provided it fell on the same
            # weekday as the query.
            msg_datetime = datetime.fromisoformat(metadata["timestamp"])
            if max_past_days >= 0 and not self._filter_memory_by_past_days(
                msg_datetime, query_at_datetime, max_past_days
            ):
                return False
            if self.long_term_memory_filter_by_datetime and query_at_datetime:
                return self._filter_memory_by_working_day(
                    msg_datetime, query_at_datetime
                ) and self._filter_memory_by_peak_time(msg_datetime, query_at_datetime)
            return True

        from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

        try:
            # Filtering by person_id delegated to the vector store (a Chroma `where`):
            # avoids fetching up to 500 global nodes only to keep a few of them.
            # The ×5 margin leaves room for re-ranking (time decay, keywords).
            retriever = self.shared_index.as_retriever(
                similarity_top_k=min(max(top_k * 5, 32), 100),
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="person_id", value=person_id)]
                ),
            )

            nodes = await retriever.aretrieve(query)
            logger.debug(f"Retrieved {len(nodes)} raw nodes for user {person_id}")

            # Ranking and filters read the agent's metadata, not the copy
            # frozen in the index: `force`, `rappels` and concept counters evolve,
            # and the index is not rewritten for all that. The entries are already in RAM here —
            # `ensure_user_initialized` reloaded them from disk if the agent had been evicted
            # from the LRU cache. Built BEFORE the pools: the filter of out-of-service concepts
            # needs it.
            entrees_par_doc = {
                e.doc_id: e
                for e in self.user_metadata.get(person_id, {}).get("entries", [])
                if getattr(e, "doc_id", None)
            }

            # ── Pool A: semantic. Defence in depth on person_id on the Python side.
            user_results = []
            vus = set()
            for node in nodes:
                if node.metadata.get("person_id") == person_id and filter_message(
                    node.metadata
                ):
                    _meta = dict(node.metadata)
                    _meta.setdefault("vivier", "A")
                    _doc = _meta.get("doc_id")
                    if _doc and _doc in vus:
                        continue
                    if _doc:
                        vus.add(_doc)
                    user_results.append(
                        MemorySearchResult(
                            content=node.text,
                            metadata=_meta,
                            score=getattr(node, "score", 0.0),
                        )
                    )

            # ── Pools B and C: structured, read in RAM, without embedding (ticket 071, lot 2).
            # They ADD to the semantic pool and are deduplicated by document
            # identifier. The age window is applied to them as to the others: it is the only
            # filter that remains, together with the agent's identity.
            for res in self.viviers_structures(person_id, modes_offerts):
                _doc = res.metadata.get("doc_id")
                if _doc in vus or not filter_message(res.metadata):
                    continue
                vus.add(_doc)
                user_results.append(res)

            # Ticket 071, lot 3 — concepts TAKEN OUT OF SERVICE are excluded from recall. They
            # stay in the metadata: they are not deleted, their dated setting-aside
            # is the observable the hysteresis experiment looks for.
            #
            # ⚠ This is the THIRD exception to the non-exclusion rule of lot 2, together with
            # the agent's identity and the age window. It is written as an exception
            # and not merged into the rule: "nothing filters" carries over to the paper, and
            # would become false there without mention.
            _avant = len(user_results)
            user_results = [
                r
                for r in user_results
                if (
                    entrees_par_doc.get((r.metadata or {}).get("doc_id")) is None
                    or entrees_par_doc[(r.metadata or {}).get("doc_id")].est_servi
                )
            ]
            _ecartes = _avant - len(user_results)
            if _ecartes:
                logger.debug(
                    f"[concepts] {_ecartes} concept(s) hors service écarté(s) du rappel pour "
                    f"{person_id} — contredits plus souvent que confirmés, conservés en mémoire"
                )
            # Re-rank the results
            scores = self.rank_nodes(
                query, query_at, user_results, entrees_par_doc, contexte
            )
            # get topk user_results by scores
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            result = [user_results[i] for i in top_k_indices]

            self._compter_viviers(person_id, user_results, result)

            # Ticket 077, lots E1 and D2 — what was served, with its score and its pool,
            # and the concentration of recalls. It is here, and only here, that the top-K, the
            # scores that produced it and the pools of origin are known together.
            tracer_rappel(
                person_id,
                query_at,
                result,
                {
                    str((user_results[i].metadata or {}).get("doc_id") or ""): float(
                        scores[i]
                    )
                    for i in top_k_indices
                },
                len(user_results),
            )

            # Recall REINFORCES, and only what was actually served to the model: the
            # candidates excluded from the top-K were not recalled. This is the mechanism of
            # MemoryBank, and the corollary of Park et al. whose freshness decays since the
            # last recall and not since creation.
            self._renforcer_les_servis(
                person_id, result, entrees_par_doc, query_at_datetime
            )

            LTM_QUERY_DURATION.observe(time.monotonic() - _t0)
            return result

        except Exception as e:
            LTM_QUERY_DURATION.observe(time.monotonic() - _t0)
            logger.exception(f"Error querying memories for user {person_id}: {e}")
            return []

    # Observation window of the pools, in number of decisions. The alarms are read over
    # a window and not a single shot: an empty pool B on ONE decision is commonplace — the agent
    # has no memory of the offered mode yet. Empty on a third of a window, it is a
    # failing axis normalisation.
    _FENETRE_VIVIERS = 200

    def _compter_viviers(
        self,
        person_id: str,
        candidats: list[MemorySearchResult],
        servis: list[MemorySearchResult],
    ) -> None:
        """Share of the top-K from each pool, and the two alarms of lot 2.

        This is the DIRECT measurement of the usefulness of the structured pools. Without it, one
        could not tell "pools B and C bring up memories that A missed" from "they
        are useless and the design must be revisited".
        """
        if not hasattr(self, "_viviers_fenetre"):
            self._viviers_fenetre: list = []
            self._alarme_vivier_b = False
            self._alarme_vivier_a = False

        parts = {"A": 0, "B": 0, "C": 0}
        for r in servis:
            parts[(r.metadata or {}).get("vivier", "A")] = (
                parts.get((r.metadata or {}).get("vivier", "A"), 0) + 1
            )
        b_propose = any((c.metadata or {}).get("vivier") == "B" for c in candidats)
        self._viviers_fenetre.append((parts, b_propose))
        if len(self._viviers_fenetre) > self._FENETRE_VIVIERS:
            self._viviers_fenetre = self._viviers_fenetre[-self._FENETRE_VIVIERS :]

        n = len(self._viviers_fenetre)
        if n < self._FENETRE_VIVIERS:
            return  # an incomplete window triggers nothing: too little to conclude

        sans_b = sum(1 for p, propose in self._viviers_fenetre if not propose)
        total_servis = sum(sum(p.values()) for p, _ in self._viviers_fenetre) or 1
        part_a = sum(p.get("A", 0) for p, _ in self._viviers_fenetre) / total_servis

        logger.info(
            f"[viviers] window of {n} decisions — share of the top-K: "
            f"A {part_a:.0%}, B {sum(p.get('B', 0) for p, _ in self._viviers_fenetre) / total_servis:.0%}, "
            f"C {sum(p.get('C', 0) for p, _ in self._viviers_fenetre) / total_servis:.0%} "
            f"| pool B empty on {sans_b / n:.0%} of decisions"
        )

        if sans_b / n > 1 / 3 and not self._alarme_vivier_b:
            self._alarme_vivier_b = True
            logger.error(
                f"[ALARME] pool B empty on {sans_b / n:.0%} of the last {n} decisions "
                f"(threshold: one third) — axis normalisation is probably failing, "
                f"the memories do not carry the mode of the offered options"
            )
        elif sans_b / n <= 1 / 6 and self._alarme_vivier_b:
            self._alarme_vivier_b = False
            logger.info("[viviers] pool B is fed again")

        if part_a > 0.95 and not self._alarme_vivier_a:
            self._alarme_vivier_a = True
            logger.error(
                f"[ALARME] {part_a:.0%} du top-K vient du SEUL vivier sémantique sur les {n} "
                f"dernières décisions (seuil : 95 %) — les viviers structurés n'apportent rien "
                f"et la conception du lot 2 est à revoir"
            )
        elif part_a <= 0.90 and self._alarme_vivier_a:
            self._alarme_vivier_a = False
            logger.info(
                "[viviers] the structured pools contribute to the top-K again"
            )

    def journal_trajets(self, person_id: str) -> dict:
        """The agent's trip journal — the counter its habits come from (lot 4)."""
        self.ensure_user_initialized(person_id)
        return self.user_metadata[person_id].setdefault("journal", {})

    def noter_trajet(
        self,
        person_id: str,
        motif: str | None,
        creneau: str | None,
        mode: str | None,
        retard_s: float = 0.0,
    ) -> None:
        """Records a completed trip in the agent's journal.

        The journal is PERSISTED with the metadata: it reuses the deferred write already in
        place. Without persistence, a resumed run would restart without habits, and the habits
        block would lie by omission throughout the first day.
        """
        from llm.noyau import noter_trajet as _noter

        _noter(self.journal_trajets(person_id), motif, creneau, mode, retard_s)
        self._dirty.add(person_id)
        self._schedule_flush()

    def viviers_structures(
        self,
        person_id: str,
        modes_offerts: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Pools B (per object) and C (shocks), read from the agent's metadata.

        No embedding, no query to the vector store: the agent's memories are
        already in RAM here — `ensure_user_initialized` reloads them from disk if the agent had been
        evicted from the LRU cache. The cost is reading a list of a few hundred items.

        **B — per object.** For each mode offered in the options of the decision, the
        memories carrying this mode, the most severe and the most recent first. It is what
        brings a morning bike fall up on an evening decision: neither the place, nor the
        time slot, nor the purpose coincide, but the object links them, and the object is enough.

        **C — shocks.** The memories above the severity threshold, **without any condition**
        of place, time or purpose. It guarantees that a severe memory is never lost by
        a ranking accident.
        """
        self.ensure_user_initialized(person_id)
        entrees = self.user_metadata.get(person_id, {}).get("entries", [])
        if not entrees:
            return []

        par_mode = int(settings.agent.memoire__vivier_b_par_mode)
        taille_c = int(settings.agent.memoire__vivier_c_taille)
        seuil_choc = float(settings.agent.memoire__importance_choc)

        def _cle(e: MemoryEntry):
            # Severity first, recency next: at equal severity, the freshest comes first.
            return (float(e.importance or 0.0), e.horodatage_de_reference)

        retenus: dict[str, MemoryEntry] = {}
        origines: dict[str, str] = {}

        for mode in {m for m in (modes_offerts or []) if m}:
            candidats = [e for e in entrees if e.axe_objet == mode and e.doc_id]
            for e in sorted(candidats, key=_cle, reverse=True)[:par_mode]:
                retenus.setdefault(e.doc_id, e)
                origines.setdefault(e.doc_id, "B")

        chocs = [
            e for e in entrees if e.doc_id and float(e.importance or 0.0) >= seuil_choc
        ]
        for e in sorted(chocs, key=_cle, reverse=True)[:taille_c]:
            retenus.setdefault(e.doc_id, e)
            origines.setdefault(e.doc_id, "C")

        return [
            MemorySearchResult(
                content=entree.content,
                metadata={
                    "person_id": person_id,
                    "timestamp": entree.timestamp.isoformat(),
                    "memory_type": str(entree.memory_type),
                    "doc_id": doc_id,
                    "tags": entree.tags,
                    "vivier": origines[doc_id],
                },
                # No semantic similarity was computed for these candidates: they
                # were not found through the text. Zero is the EXACT value of their
                # measured similarity, not a default — and the four other components
                # rank them.
                score=0.0,
            )
            for doc_id, entree in retenus.items()
        ]

    def rank_nodes(
        self,
        query: str,
        query_at: int | None,
        nodes: list[MemorySearchResult],
        entrees_par_doc: dict[str, MemoryEntry] | None = None,
        contexte: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Rank nodes based on their relevance to the query.

        `entrees_par_doc` (ticket 071, lot 1) gives access to the AUTHORITATIVE entry behind
        each node: its own lifetime and the date of its last recall. When absent, the
        ranking falls back on the index timestamp and the default time constant,
        that is exactly the behaviour before lot 1.
        """
        if not nodes:
            return np.array([])

        sim_w = settings.agent.long_term_retrieval__sim_weight
        cat_w = settings.agent.long_term_retrieval__keyword_weight
        temps_w = settings.agent.long_term_retrieval__time_weight
        grav_w = settings.agent.long_term_retrieval__importance_weight
        axes_w = settings.agent.long_term_retrieval__affinite_weight

        entrees = entrees_par_doc or {}
        ctx = contexte or {}

        def _entree(n):
            return entrees.get((n.metadata or {}).get("doc_id"))

        # 1. Semantic similarity. Bounded: the vector store may go outside [0, 1], and the
        # candidates of the structured pools have no measured similarity — it is zero,
        # which is exact, and their four other components rank them.
        _sim = np.clip(np.array([n.score for n in nodes]), 0.0, 1.0)

        # 2. Categorical affinity — WEATHER only (decision of 2026-09-14, issue A). It
        # replaces the so-called "BLEU-2" score, which was an asymmetric lexical recall rate on
        # the labels and not the BLEU of Papineni et al. Reduced to weather because its
        # three other attributes — mode, time slot, purpose — ARE already the axes of the next
        # component: counting them twice would make the score uninterpretable.
        _cat = np.array(
            [
                affinite_meteo(
                    getattr(_entree(n), "axe_meteo", None), ctx.get("axe_meteo")
                )
                for n in nodes
            ]
        )

        # 3. Time weight: exp(-Δt / force), Δt since the last recall.
        _temps = np.array(
            [
                self._time_decay_score(
                    n.metadata.get("timestamp"), query_at, entree=_entree(n)
                )
                for n in nodes
            ]
        )

        # 4. Severity of the memory. Component restored from Park et al. (2023), which Vu et al.
        # had set aside.
        _grav = np.array(
            [float(getattr(_entree(n), "importance", 0.0) or 0.0) for n in nodes]
        )

        # 5. Axis affinity, as a BONUS and never a veto: a mismatching axis contributes zero,
        # it subtracts nothing. Apart from the agent's identity and the age window, NOTHING filters.
        _axes = np.array(
            [
                affinite_axes(
                    getattr(_entree(n), "axe_objet", None),
                    getattr(_entree(n), "axe_lieu", None),
                    getattr(_entree(n), "axe_creneau", None),
                    getattr(_entree(n), "axe_motif", None),
                    objet_courant=ctx.get("axe_objet"),
                    lieu_courant=ctx.get("axe_lieu"),
                    creneau_courant=ctx.get("axe_creneau"),
                    motif_courant=ctx.get("axe_motif"),
                )
                for n in nodes
            ]
        )

        # Ticket 048 — per-COMPONENT min-max normalisation remains abandoned. It mechanically
        # brought the best candidate of the batch to 1 and the worst to 0, whatever the real
        # gap: two decisions were not comparable, and the order by age was
        # INVARIANT to the time constant. The five components enter as ABSOLUTE values.
        combined_score = (
            _sim * sim_w
            + _cat * cat_w
            + _temps * temps_w
            + _grav * grav_w
            + _axes * axes_w
        )

        return combined_score

    def _time_decay_score(
        self,
        timestamp_str: str,
        query_at: int | None,
        entree: MemoryEntry | None = None,
    ) -> float:
        """Time weight of a memory, on [0, 1] and as an ABSOLUTE value.

        Ticket 071, lot 1 — two changes compared with ticket 048:

        - the time constant is no longer shared, it is that of the memory, a function of its
          severity: nearly twenty days for a `marquant` memory against less than three for
          an ordinary trip;
        - Δt is counted from the **last recall** and not from the write, as in
          Park et al. (2023, § 4.1) and MemoryBank. A memory often recalled stays fresh;
          a memory never recalled ages from its write, as before.

        `gama_timestamp` and not `.timestamp()`: `query_at` is a GAMA timestamp (wall-clock
        time) and the `datetime` of the memories carry wall-clock fields — subtracting them
        after a trip through the process time zone would add one hour of fictitious age.

        Without an authoritative entry (entry written before lot 1, or a test that provides only
        nodes), we fall back on the index timestamp and the default constant.
        """
        if query_at is None:
            return 0.0

        if entree is not None:
            # Ticket 071, lot 3 — TWO REGIMES, and it is the strongest point of the ticket.
            # For a concept or a summary, the time component is no longer a
            # clock decay but the CONFIDENCE. That a line is saturated on rainy days
            # between 8:00 and 8:30 does not become false because ten days have passed — yet under
            # the uniform regime this concept fell to 2.8 % of its weight in ten days and
            # left the top-K without any observation having disproved it.
            if not entree.est_episodique:
                return float(entree.confiance)
            reference = entree.horodatage_de_reference
            force = entree.force
        else:
            if not timestamp_str:
                return 0.0
            try:
                reference = datetime.fromisoformat(timestamp_str)
            except ValueError:
                return 0.0
            force = None

        time_diff = max(0, (query_at - gama_timestamp(reference)) / (24 * 3600))
        return poids_temporel(time_diff, force)

    def _renforcer_les_servis(
        self,
        person_id: str,
        servis: list[MemorySearchResult],
        entrees_par_doc: dict[str, MemoryEntry],
        quand: datetime | None,
    ) -> int:
        """`force ← min(force + δ, FORCE_MAX)` and `rappels += 1` on the SERVED entries.

        Three precautions that are not details:

        - only the top-K entries are reinforced, not the candidates: a memory excluded
          from the ranking was not recalled;
        - `timestamp` is NEVER touched — it is shown in the prompt and is the left side
          of the per-day and per-age filters. Sliding it would rewrite the history of the
          agent, who would believe its bike fall happened yesterday. Only `dernier_rappel` moves;
        - the disk write goes through the existing `dirty` mechanism: no synchronous
          disk access on the path of a decision.

        Without a simulated clock (`quand is None`), the lifetime is still reinforced but
        the recall is not dated: never a fallback on the machine clock.
        """
        if not servis or not entrees_par_doc:
            return 0
        renforces = 0
        renforcees: list[MemoryEntry] = []
        for resultat in servis:
            entree = entrees_par_doc.get((resultat.metadata or {}).get("doc_id"))
            if entree is None:
                continue
            entree.force = force_apres_rappel(entree.force)
            entree.rappels = int(entree.rappels or 0) + 1
            if quand is not None:
                entree.dernier_rappel = quand
            renforces += 1
            renforcees.append(entree)
        if renforces:
            self._dirty.add(person_id)
            self._schedule_flush()
            # Ticket 075 — recall MODIFIES memory (lifetime, counter, date): it has its
            # line in the journal. Without a simulated clock, nothing is written rather than dated
            # with the machine clock.
            _journal = journal()
            if _journal is not None and quand is not None:
                _journal.rappel(person_id, quand, renforcees)
        return renforces

    def _bleu_score(self, query: str, keyword: str) -> float:
        if not keyword or not query:
            return 0.0

        # Tokenize keywords and query
        kw_tokens = [
            token.strip(string.punctuation)
            for token in keyword.lower().split()
            if token.strip(string.punctuation)
        ]
        query_tokens = [
            token.strip(string.punctuation)
            for token in query.lower().split()
            if token.strip(string.punctuation)
        ]

        # Calculate unigram (1-gram) overlap
        kw_unigrams = set(kw_tokens)
        query_unigrams = set(query_tokens)
        unigram_overlap = len(kw_unigrams.intersection(query_unigrams))
        unigram_score = unigram_overlap / len(kw_unigrams) if kw_unigrams else 0.0

        # Calculate bigram (2-gram) overlap
        kw_bigrams = (
            set(zip(kw_tokens[:-1], kw_tokens[1:])) if len(kw_tokens) > 1 else set()
        )
        query_bigrams = (
            set(zip(query_tokens[:-1], query_tokens[1:]))
            if len(query_tokens) > 1
            else set()
        )
        bigram_overlap = len(kw_bigrams.intersection(query_bigrams))
        bigram_score = bigram_overlap / len(kw_bigrams) if kw_bigrams else 0.0

        # Combine unigram and bigram scores with weights
        # Weight unigrams more heavily as they're more likely to match
        # Ticket 071 (defect D) — a one-word label has NO bigram: applying
        # the 0.3 weight to a zero score capped it at 0.70 even for a perfect match,
        # against 1.00 for a two-word label. Without bigrams, the weight goes back
        # entirely to unigrams.
        if kw_bigrams:
            combined_score = 0.7 * unigram_score + 0.3 * bigram_score
        else:
            combined_score = unigram_score

        return combined_score

    def _sim_now(self, person_id: str) -> datetime | None:
        """SIMULATED wall-clock reference time for this agent, or None if it cannot be determined.

        The agent's most recent memory carries GAMA's wall-clock time: it is the only
        "now" that makes sense here. The host machine's clock is not one.
        """
        entries = self.user_metadata.get(person_id, {}).get("entries") or []
        stamps = [
            e.timestamp for e in entries if getattr(e, "timestamp", None) is not None
        ]
        return max(stamps) if stamps else None

    def cleanup_user_memories(
        self, person_id: str, days_threshold: int = 30, now: datetime | None = None
    ):
        """Cleanup old memories for specific user.

        Ticket 071 (defect A) — the threshold was computed on `datetime.now()`, the
        MACHINE clock, whereas `entry.timestamp` carries GAMA's wall-clock time. As soon as a run
        replayed a date earlier than the threshold, the retention condition was
        false for ALL entries and the cleanup emptied concepts and conversations.

        Rule adopted: a cleanup that cannot establish the simulated time cleans NOTHING.
        Never a fallback on the machine clock — a deletion must not depend on
        the date the run was started.
        """
        self.ensure_user_initialized(person_id)

        if person_id not in self.user_metadata:
            return

        sim_now = now or self._sim_now(person_id)
        if sim_now is None:
            logger.warning(
                f"[cleanup] Simulated time undeterminable for {person_id} — no timestamped "
                f"memory: cleanup ABANDONED (never a fallback on the machine clock)"
            )
            return

        original_count = len(self.user_metadata[person_id]["entries"])

        # Ticket 071, lot 1 — retention stops being blind to the memory's content.
        #
        # BEFORE: a common age threshold, plus an exemption by TYPE (reflections and
        # summaries never left). Two unfortunate consequences: a severe memory
        # thirty-one days old fell with the ordinary trips, and reflections
        # piled up without end.
        #
        # NOW, two regimes, as Tulving's taxonomy (1972) requires:
        #   - SEMANTIC (concepts, summaries): never purged by the clock. A concept does not
        #     become false because ten days have passed; it is lost through CONTRADICTION,
        #     and its dated setting-aside is the observable the experiment looks for (lot 3);
        #   - EPISODIC (raw entries, reflections): purged when its time weight drops
        #     below the threshold, i.e. ~4.6 time constants. Thirteen days for an ordinary trip
        #     never recalled, ninety-one for a `marquant` memory — hence never within
        #     a run. Severity and recalls decide, no longer the calendar alone.
        #
        # `days_threshold` remains a safety FLOOR: nothing younger is purged,
        # whatever its lifetime. It can therefore only delay a purge, never
        # cause one.
        filtered_entries = []
        supprimes = []
        for entry in self.user_metadata[person_id]["entries"]:
            try:
                if not entry.est_episodique:
                    filtered_entries.append(entry)
                    continue
                age_jours = (
                    sim_now - entry.horodatage_de_reference
                ).total_seconds() / 86400.0
                if age_jours <= days_threshold:
                    filtered_entries.append(entry)
                elif est_purgeable(age_jours, entry.force):
                    supprimes.append(entry)
                else:
                    filtered_entries.append(entry)

            except (TypeError, AttributeError):
                # Keep malformed entries to be safe
                filtered_entries.append(entry)

        # Ticket 071 (defect B) — also remove from the vector index. Without this, an entry
        # absent from the metadata kept being returned by ChromaDB and reinjected into
        # the decision prompts. A failed deletion must not cost the update of the
        # metadata: it is logged, not propagated.
        self._delete_from_index(supprimes, person_id)

        # Ticket 075 — what is forgotten can be read in the agent's journal, with the age and the
        # force that made it fall: a memory that disappears without trace is indistinguishable
        # from a memory that was never written.
        _journal = journal()
        if _journal is not None and supprimes:
            _journal.purge(person_id, sim_now, supprimes)

        # Update metadata
        self.user_metadata[person_id]["entries"] = filtered_entries
        self.user_metadata[person_id]["last_cleanup"] = sim_now.isoformat()
        self._save_user_metadata(person_id)

        removed_count = original_count - len(filtered_entries)
        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} old memories for user {person_id} "
                f"(threshold {days_threshold} d before {sim_now.isoformat()}, simulated time)"
            )

    def _delete_from_index(self, entries: list[MemoryEntry], person_id: str) -> int:
        """Removes the given entries' documents from the vector store. Fail-open, logged."""
        if not entries or self.shared_index is None:
            return 0
        supprimes = 0
        sans_id = 0
        for entry in entries:
            doc_id = getattr(entry, "doc_id", None)
            if not doc_id:
                sans_id += 1  # entry written before ticket 071: not addressable
                continue
            try:
                self.shared_index.delete_ref_doc(doc_id, delete_from_docstore=True)
                supprimes += 1
            except Exception as exc:  # noqa: BLE001 — a failed deletion must not break anything
                logger.warning(
                    f"[cleanup] Index deletion failed for {doc_id}: {exc}"
                )
        if sans_id:
            logger.warning(
                f"[cleanup] {sans_id} memory(ies) of {person_id} without a document identifier "
                f"(written before ticket 071): removed from the metadata, KEPT in the index"
            )
        return supprimes

    def batch_cleanup_users(self, user_ids: list[str], days_threshold: int = 30):
        """Batch cleanup for multiple users"""
        cleaned_count = 0
        for person_id in user_ids:
            try:
                self.cleanup_user_memories(person_id, days_threshold)
                cleaned_count += 1

                # Periodic cache cleanup during batch
                if cleaned_count % 50 == 0:
                    self._cleanup_metadata_cache()

            except Exception as e:
                logger.error(f"Error cleaning up user {person_id}: {e}")

        logger.info(f"Batch cleanup completed for {cleaned_count} users")

    def get_all_users(self) -> list[str]:
        """Get all users efficiently by scanning shard directories"""
        users = set(self.user_metadata.keys())

        # Scan shard directories
        metadata_dir = self.storage_dir / "user_metadata"
        if metadata_dir.exists():
            for shard_dir in metadata_dir.glob("shard_*"):
                if shard_dir.is_dir():
                    for metadata_file in shard_dir.glob("*.json"):
                        users.add(metadata_file.stem)

        return list(users)

    def get_user_stats(self, person_id: str) -> dict[str, Any]:
        """Get statistics for specific user"""
        self.ensure_user_initialized(person_id)

        if person_id not in self.user_metadata:
            return {"person_id": person_id, "error": "User not found"}

        metadata = self.user_metadata[person_id]

        # Calculate recent entries
        recent_24h = 0
        recent_7d = 0
        now = datetime.now()

        for entry in metadata["entries"]:
            try:
                if now - entry.timestamp < timedelta(hours=24):
                    recent_24h += 1
                if now - entry.timestamp < timedelta(days=7):
                    recent_7d += 1
            except (TypeError, AttributeError):
                continue

        return {
            "person_id": person_id,
            "total_entries": len(metadata["entries"]),
            "recent_24h": recent_24h,
            "recent_7d": recent_7d,
            "last_cleanup": metadata.get("last_cleanup"),
            "last_reflection": metadata.get("last_reflection"),
            "created_at": metadata.get("created_at"),
            "memory_usage_mb": metadata.get("memory_usage_mb", 0),
            "in_memory_cache": True,
        }

    def get_system_stats(self) -> dict[str, Any]:
        """Get system-wide statistics"""
        return {
            "total_users": len(self.get_all_users()),
            "loaded_users_in_cache": len(self.user_metadata),
            "max_loaded_metadata": self.max_loaded_metadata,
            "vector_store_type": self.vector_store_type,
            "storage_dir": str(self.storage_dir),
            "cache_hit_ratio": self.metrics["cache_hits"]
            / max(self.metrics["cache_hits"] + self.metrics["cache_misses"], 1),
            "total_queries": self.metrics["queries"],
            "memory_cleanups": self.metrics["memory_cleanups"],
            "memory_optimized": True,
            "using_shared_index": True,
        }

    def force_cleanup_all_users(self, days_threshold: int = 30):
        """Force cleanup for all users (maintenance operation)"""
        all_users = self.get_all_users()
        logger.info(f"Starting cleanup for {len(all_users)} users...")

        # Process in batches to manage memory
        batch_size = 50
        for i in range(0, len(all_users), batch_size):
            batch = all_users[i : i + batch_size]
            self.batch_cleanup_users(batch, days_threshold)

            # Progress update
            logger.info(
                f"Cleanup progress: {min(i + batch_size, len(all_users))}/{len(all_users)} users"
            )

        # Final cleanup
        self._cleanup_metadata_cache()
        if not self.vector_store:
            self._persist_shared_index()

        logger.info("Force cleanup completed for all users")

    def get_memory_usage_breakdown(self) -> dict[str, Any]:
        """Get detailed memory usage breakdown"""
        total_entries = 0
        total_size_mb = 0
        user_count = len(self.user_metadata)

        for metadata in self.user_metadata.values():
            total_entries += len(metadata.get("entries", []))
            total_size_mb += metadata.get("memory_usage_mb", 0)

        return {
            "loaded_users": user_count,
            "total_entries_in_cache": total_entries,
            "total_cache_size_mb": total_size_mb,
            "avg_entries_per_user": total_entries / max(user_count, 1),
            "avg_size_per_user_mb": total_size_mb / max(user_count, 1),
            "cache_efficiency": f"{user_count}/{self.max_loaded_metadata}",
        }

    def get_user_all_memories(self, person_id: str) -> list[MemoryEntry]:
        """Get all memories for a specific user"""
        self.ensure_user_initialized(person_id)

        if person_id not in self.user_metadata:
            return []

        # Entries are already MemoryEntry objects (converted at load time)
        return list(self.user_metadata[person_id]["entries"])

    async def aexport_user_data(self, person_id: str) -> dict[str, Any]:
        """Export all data for a specific user"""
        self.ensure_user_initialized(person_id)

        if person_id not in self.user_metadata:
            return {"error": "User not found"}

        # Get user metadata
        user_data = {
            "person_id": person_id,
            "metadata": self.user_metadata[person_id].copy(),
            "stats": self.get_user_stats(person_id),
        }

        # Query all memories for this user
        try:
            all_memories = await self.aquery_user_memories(person_id, "", top_k=1000)
            user_data["memories"] = all_memories
        except Exception as e:
            user_data["memories"] = []
            user_data["export_error"] = str(e)

        return user_data

    def __str__(self) -> str:
        return f"ScalableLongTermMemory({self.vector_store_type}, {len(self.user_metadata)} users cached)"
