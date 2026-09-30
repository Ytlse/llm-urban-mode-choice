import asyncio
import hashlib
import json
import threading
import time
import uuid
from typing import Optional

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

from mobility_llm.mode_choice import draw_index, mode_distribution
from llm_gateway.telemetry.alarms import fire_alarme
# `settings` was read in `_redraw` without being imported (ticket 077, 2026-09-18): drawing
# from a cache entry raised `NameError`. The defect stayed invisible because the
# cache was off on every run of the ticket — it would have broken the first cache-on run.
from settings import settings
from sim_clock import wall_clock

COLLECTION_NAME = "llm_decisions"
VECTOR_SIZE = 384

LLM_CACHE_HITS = Counter(
    "llm_cache_hits_total",
    "Nombre de requêtes servies depuis le cache LLM sémantique",
    ["activity_purpose"],
)
LLM_CACHE_MISSES = Counter(
    "llm_cache_misses_total",
    "Number of LLM cache misses",
    ["reason"],
)
LLM_CACHE_LOOKUP_SECONDS = Histogram(
    "llm_cache_lookup_seconds",
    "Latence totale de la recherche dans le cache LLM (embed + Qdrant)",
)
LLM_CACHE_EMBED_SECONDS = Histogram(
    "llm_cache_embed_seconds",
    "Latence de l'inférence sentence-transformer pour le cache LLM",
)
LLM_CACHE_INSERT_SECONDS = Histogram(
    "llm_cache_insert_seconds",
    "Latence d'écriture Qdrant locale pour le cache LLM",
)

# Coverage of the persistent cache, read at startup — answers at a
# glance "is the cache populated enough for init?" (cf. dashboards 02_init_bootstrap
# and 06_cache_llm / init_report).
LLM_CACHE_POINTS_TOTAL = Gauge(
    "llm_cache_points_total",
    "Nombre total de points dans la collection Qdrant du cache LLM au démarrage",
)
LLM_CACHE_POINTS_EXACT = Gauge(
    "llm_cache_points_exact",
    "« Empty memory » points (exact match) — path used by the bootstrap at /init",
)
LLM_CACHE_POINTS_STALE = Gauge(
    "llm_cache_points_stale",
    "Points unusable by the current filter (obsolete schema: weekday missing)",
)
LLM_CACHE_AGENTS_COVERED = Gauge(
    "llm_cache_agents_covered",
    "Number of distinct agents with at least one « empty memory » decision in cache",
)

# Process-wide hits/lookups counters for reporting the LLM cache rate in the logs.
_LLM_CACHE_HITS = 0
_LLM_CACHE_LOOKUPS = 0
# Breakdown of misses by reason (no_candidates, code_not_in_options, lookup_error)
# to diagnose the failure rate without depending on Prometheus.
_LLM_MISS_REASONS: dict[str, int] = {}


def _record_llm_miss(reason: str) -> None:
    _LLM_MISS_REASONS[reason] = _LLM_MISS_REASONS.get(reason, 0) + 1


def get_llm_cache_stats() -> tuple[int, int]:
    """Returns the cumulative (hits, lookups) of the semantic LLM cache since startup."""
    return _LLM_CACHE_HITS, _LLM_CACHE_LOOKUPS


def get_llm_miss_breakdown() -> dict[str, int]:
    """Returns the cumulative breakdown of LLM cache misses by reason."""
    return dict(_LLM_MISS_REASONS)


_instances: dict[str, "LlmSemanticCache"] = {}
_instances_lock = threading.Lock()


class LlmSemanticCache:
    def __new__(cls, cache_dir: str, semantic_threshold: float, embed_model_name: str):
        with _instances_lock:
            if cache_dir not in _instances:
                instance = super().__new__(cls)
                instance._initialized = False
                _instances[cache_dir] = instance
            return _instances[cache_dir]

    def __init__(self, cache_dir: str, semantic_threshold: float, embed_model_name: str):
        """Loads the embedding model and opens the local Qdrant database in cache_dir."""
        if self._initialized:
            return
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self._threshold = semantic_threshold
        logger.info(f"Chargement du modèle d'embedding LLM cache : {embed_model_name}")
        self._model = SentenceTransformer(embed_model_name)
        self._embed_lock = threading.Lock()
        # The embedded-mode QdrantClient (path=...) is NOT thread-safe:
        # concurrent query_points/upsert (started via asyncio.to_thread)
        # corrupt the index ("operands could not be broadcast", SQLite errors).
        # This lock serialises all operations on the database.
        self._db_lock = threading.Lock()
        self._consecutive_errors = 0
        self._empty_vector: Optional[list] = None
        # Agents with at least one « empty memory » decision in cache — filled in by
        # log_coverage(). Used to classify a no_candidates miss: agent absent (coverage
        # gap) vs agent present but different key (weather/time slot/state_hash).
        self._exact_agents: set[str] = set()
        self._miss_diag_budget = 30  # number of no_candidates misses detailed in the log (bounded)
        self._client = QdrantClient(path=cache_dir)
        self._ensure_collection()
        self._initialized = True
        logger.info(f"LlmSemanticCache initialisé — répertoire: {cache_dir}, seuil: {semantic_threshold}")
        self.log_coverage()

    def log_coverage(self) -> None:
        """Reads the coverage of the persistent cache at startup and exposes it (log + metrics).

        Answers "is the cache populated enough to serve init at 100%?" without having to
        replay a run: how many points, how many usable by the bootstrap (« empty
        memory » path), how many agents covered, and how many points inherited from an
        obsolete schema (weekday missing), hence never served by the current filter.
        """
        try:
            total = 0
            exact = 0            # memory_empty=True → bootstrap path
            stale = 0            # weekday missing → never matched by the current filter
            agents: set[str] = set()
            offset = None
            with self._db_lock:
                while True:
                    pts, offset = self._client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=4000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for p in pts:
                        pl = p.payload or {}
                        total += 1
                        if pl.get("weekday") is None:
                            stale += 1
                        if pl.get("memory_empty"):
                            exact += 1
                            aid = pl.get("agent_id")
                            if aid:
                                agents.add(str(aid))
                    if offset is None:
                        break
            self._exact_agents = agents
            LLM_CACHE_POINTS_TOTAL.set(total)
            LLM_CACHE_POINTS_EXACT.set(exact)
            LLM_CACHE_POINTS_STALE.set(stale)
            LLM_CACHE_AGENTS_COVERED.set(len(agents))
            logger.info(
                f"[cache] LLM coverage at startup: {total} points "
                f"({exact} exact/bootstrap, {len(agents)} agents covered, {stale} obsolete weekday=None)"
            )
            if total and stale / total > 0.3:
                fire_alarme("cache_llm_stale")
                logger.warning(
                    f"[ALARME] LLM cache: {stale}/{total} points ({100*stale//total}%) are inherited from an "
                    f"obsolete schema (weekday=None) and will NEVER be served by the current filter. "
                    f"They bloat the database without improving the hit rate — consider purging/repopulating the cache."
                )
        except Exception as e:
            logger.warning(f"[cache] log_coverage failed (non-blocking): {e}")

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not exist yet."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    _ERROR_ALARM_THRESHOLD = 5

    def _record_db_error(self, operation: str, error: Exception) -> None:
        """Traces a Qdrant error and raises an alarm after N consecutive errors (corrupted database?)."""
        self._consecutive_errors += 1
        if self._consecutive_errors == self._ERROR_ALARM_THRESHOLD:
            fire_alarme("cache_llm_qdrant")
            logger.error(
                f"[ALARME] LLM cache: {self._consecutive_errors} consecutive errors "
                f"(last: {operation} → {error}) — Qdrant database probably corrupted, "
                f"the cache no longer serves any decision. Delete the cache directory and restart."
            )

    def _record_db_ok(self) -> None:
        self._consecutive_errors = 0

    @staticmethod
    def _make_state_hash(options: list, weather: Optional[dict] = None, extra_key: str = "",
                         traits_key: str = "") -> str:
        """SHA-256 hash of the sorted option codes + weather + data version.

        ⚠ The itinerary data version (``terminal_time.data_version()``)
        is ESSENTIAL here, and it is the subtlest trap of ticket 013:
        ``get_code()`` is made of routes and stops, hence **insensitive to
        durations** by construction. Without a version, a run would quietly replay
        decisions taken on options where the car was faster
        than it is — without any log reporting it, since from its point
        of view the decision context is identical.

        ``traits_key`` (2026-08-27): signature of the persona's traits. **Third
        occurrence of the same trap**, and the one that cost a manual cache flush. The
        traits that do not condition the offer — the PT subscription first and foremost —
        appear NEITHER in the option codes NOR in the weather: they only change
        the prompt text (cf. ``_pt_subscription_note``). Without this signature, fixing
        the subscription of 352 agents let their already cached decisions be served again
        under the old prompt, without any log reporting it. The licence, for its part, goes through
        ``_can_drive`` and therefore moves the option codes: it already self-invalidated.

        ``extra_key`` (ticket 014): signature of the anticipation context injected
        into the prompt (day's weather, remaining agenda, vehicle positions).
        Same trap as above: this context appears neither in the option
        codes nor in the current weather — without it, two different agendas
        would silently serve each other their decisions. Empty when
        anticipation is disabled: the hash then carries only the data
        version. (It is not "the old one" for all that: adding
        ``data_version`` already made unreachable the points prior to
        ticket 013 — which is precisely the intended effect.)
        """
        from trip_helper.terminal_time import data_version

        codes = sorted(opt.get_code() for opt in options)
        weather_key = ""
        if weather:
            weather_key = f"{weather.get('weather_code','')}|{round(weather.get('temperature', 0))}|{round(weather.get('precip_mm', 0), 1)}"
        raw = (f"{data_version()}|" + json.dumps(codes, ensure_ascii=False)
               + weather_key)
        if extra_key:
            raw += f"|anticipation:{extra_key}"
        if traits_key:
            raw += f"|traits:{traits_key}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _make_time_slice(timestamp: int) -> str:
        """10-minute slice of GAMA's WALL-CLOCK time (e.g. "08:40").

        ⚠ `datetime.fromtimestamp(timestamp)` read this timestamp in the PROCESS
        time zone: the cache key carried 06:00 for 5:00 wall-clock in the `controller`
        (`TZ=Europe/Paris`) and 05:00 in a replica in `TZ=UTC` — two processes of the same
        run therefore did not address the same entry. Fixed on 2026-09-04.
        """
        dt = wall_clock(timestamp)
        minutes = (dt.minute // 10) * 10
        return f"{dt.hour:02d}:{minutes:02d}"

    @staticmethod
    def _make_weekday(timestamp: int) -> str:
        """Day category ("Weekday"/"Weekend") of GAMA's WALL-CLOCK time.

        ⚠ The shift was not only in the hour: read in the process time zone,
        a departure at **23:00 wall-clock on Friday** tipped over to Saturday, hence to
        "Weekend" — its context key got confused with that of a real
        weekend departure. 77 of the 5,322 trips of the archived run
        `2026-09-04_01_09` leave within the 23:00 wall-clock hour.
        """
        return "Weekend" if wall_clock(timestamp).weekday() >= 5 else "Weekday"

    def _make_filter(
        self,
        agent_id: str,
        activity_id: Optional[str],
        timestamp: int,
        options: list,
        weather: Optional[dict],
        memory_empty: bool,
        extra_key: str = "",
        traits_key: str = "",
    ):
        """Deterministic filter of the factual conditions, shared by both cache branches."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(key="agent_id", match=MatchValue(value=str(agent_id))),
                FieldCondition(key="activity_id", match=MatchValue(value=str(activity_id or ""))),
                FieldCondition(key="weekday", match=MatchValue(value=self._make_weekday(timestamp))),
                FieldCondition(key="time_slice", match=MatchValue(value=self._make_time_slice(timestamp))),
                FieldCondition(key="state_hash", match=MatchValue(value=self._make_state_hash(options, weather, extra_key, traits_key))),
                FieldCondition(key="memory_empty", match=MatchValue(value=memory_empty)),
            ]
        )

    def _embed(self, text: str) -> list:
        """Encodes the text as a vector via the sentence-transformer and records the latency."""
        t0 = time.perf_counter()
        with self._embed_lock:
            vec = self._model.encode(text).tolist()
        LLM_CACHE_EMBED_SECONDS.observe(time.perf_counter() - t0)
        return vec

    # Number of duplicates fetched to break ties by `stored_at`: the same context
    # may have been written several times (upsert by UUID) over the runs.
    _LOOKUP_SCROLL_LIMIT = 8

    def _lookup_exact_sync(
        self,
        agent_id: str,
        activity_id: Optional[str],
        timestamp: int,
        options: list,
        weather: Optional[dict],
        extra_key: str = "",
        traits_key: str = "",
    ) -> tuple[Optional[dict], Optional[str]]:
        """« Empty memory » branch: exact match on the factual conditions.

        Without memories, two decisions taken under the same conditions are identical:
        the deterministic filter is enough. Simple key-value `scroll` — no embedding.
        """
        filt = self._make_filter(agent_id, activity_id, timestamp, options, weather, memory_empty=True, extra_key=extra_key, traits_key=traits_key)
        with self._db_lock:
            candidates, _ = self._client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=filt,
                limit=self._LOOKUP_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=False,
            )

        if not candidates:
            return None, "no_candidates"

        # `scroll` does not guarantee order: we keep the most recent decision.
        best = max(candidates, key=lambda p: (p.payload or {}).get("stored_at") or 0)
        payload = best.payload or {}
        return {
            "chosen_plan_code": payload.get("chosen_plan_code"),
            "probabilities": payload.get("probabilities"),
            "mode": payload.get("mode", ""),
            "score": None,
        }, None

    def _lookup_semantic_sync(
        self,
        agent_id: str,
        activity_id: Optional[str],
        timestamp: int,
        options: list,
        memory_text: str,
        weather: Optional[dict],
        extra_key: str = "",
        traits_key: str = "",
    ) -> tuple[Optional[dict], Optional[str]]:
        """« Filled memory » branch: semantic similarity on the LTM, under equal conditions.

        The agent's experience influences its decision: the cached decision is accepted only
        if the current LTM is close to the one that produced it (similarity ≥ threshold).
        """
        filt = self._make_filter(agent_id, activity_id, timestamp, options, weather, memory_empty=False, extra_key=extra_key, traits_key=traits_key)
        query_vector = self._embed(memory_text)
        with self._db_lock:
            candidates = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=filt,
                limit=1,
            ).points

        if not candidates:
            return None, "no_candidates"

        best = candidates[0]
        if best.score < self._threshold:
            return None, "below_threshold"

        payload = best.payload or {}
        return {
            "chosen_plan_code": payload.get("chosen_plan_code"),
            "probabilities": payload.get("probabilities"),
            "mode": payload.get("mode", ""),
            "score": best.score,
        }, None

    def _empty_memory_vector(self) -> list:
        """Neutral vector of the « empty memory » points: these points are read by `scroll`
        (filter only), never by similarity — but Qdrant requires a vector on insertion."""
        if self._empty_vector is None:
            self._empty_vector = [0.0] * (VECTOR_SIZE - 1) + [1.0]
        return self._empty_vector

    def _store_sync(
        self,
        agent_id: str,
        activity_id: Optional[str],
        timestamp: int,
        options: list,
        memory_text: Optional[str],
        chosen_plan_code: str,
        mode: str,
        weather: Optional[dict],
        probabilities: Optional[list[float]] = None,
        extra_key: str = "",
        traits_key: str = "",
    ):
        """Inserts (upserts) a Qdrant point with the metadata of the decision.

        `memory_text=None` signals a decision taken without memories: the point is marked
        `memory_empty=True` and carries a neutral vector (it will only be re-read by filter).

        `probabilities` (aligned with `options`) is the distribution produced by the LLM:
        it is the one replayed at each hit — the cached decision is not frozen,
        it is drawn again. `chosen_plan_code`/`mode` keep the original draw
        (diagnostics, and re-reading by readers prior to the switch).
        """
        from qdrant_client.models import PointStruct

        state_hash = self._make_state_hash(options, weather, extra_key, traits_key)
        time_slice = self._make_time_slice(timestamp)
        # `day`/`month` of the payload: the SIMULATED DAY, the one whose weather was read.
        # `_dt.fromtimestamp(timestamp)` took them in the process time zone, and thus
        # wrote the next day for a departure at 23:00 wall-clock.
        dt = wall_clock(timestamp)

        trajectories = [
            {"code": opt.get_code(), "mode": opt.mode_label(), "duration_ms": opt.duration or 0}
            for opt in options
        ]

        # Distribution persisted per plan code: on re-reading, the options may
        # have changed (itinerary gone) — matching is done on the code, not
        # on the position.
        probability_payload = None
        if probabilities is not None:
            probability_payload = [
                {"code": traj["code"], "mode": traj["mode"], "p": float(p)}
                for traj, p in zip(trajectories, probabilities)
            ]

        memory_empty = memory_text is None
        vector = self._empty_memory_vector() if memory_empty else self._embed(memory_text)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "agent_id": str(agent_id),
                "activity_id": str(activity_id or ""),
                "weekday": self._make_weekday(timestamp),
                "time_slice": time_slice,
                "state_hash": state_hash,
                "memory_empty": memory_empty,
                "day": dt.day,
                "month": dt.month,
                "temperature": weather.get("temperature") if weather else None,
                "weather_code": weather.get("weather_code") if weather else None,
                "weather_label": weather.get("weather_label") if weather else None,
                "precip_mm": weather.get("precip_mm") if weather else None,
                "trajectories": trajectories,
                "probabilities": probability_payload,
                "chosen_plan_code": chosen_plan_code,
                "mode": mode,
                "stored_at": time.time(),
            },
        )
        with self._db_lock:
            self._client.upsert(collection_name=COLLECTION_NAME, points=[point])

    @staticmethod
    def _redraw_from_cached(
        cached_probabilities: list,
        options: list,
        seed_parts: tuple,
    ) -> Optional[dict]:
        """Replays a draw on the cached distribution, restricted to the current options.

        A hit does not return a frozen decision: the distribution is drawn again
        every time, with a seed derived from the context (agent, activity, day) — the
        same agent can thus take its car one day and the bus the next without
        an LLM call, while keeping a run replayable identically.

        Options gone since the write are dropped and the remaining mass is
        renormalised by the draw. Returns None if nothing drawable remains (the
        cache is then treated as a miss → new LLM call).
        """
        p_by_code = {}
        for entry in cached_probabilities or ():
            code = (entry or {}).get("code")
            if code is None:
                continue
            try:
                p_by_code[code] = max(0.0, float(entry.get("p") or 0.0))
            except (TypeError, ValueError):
                continue

        weights = [p_by_code.get(opt.get_code(), 0.0) for opt in options]
        if sum(weights) <= 0:
            return None

        index = draw_index(
            weights,
            *seed_parts,
            min_prob_threshold=settings.agent.mode_choice_truncation_threshold,
        )
        modes = [opt.mode_label() for opt in options]
        return {
            "index": index,
            "mode": modes[index],
            "weights": weights,
            "distribution": mode_distribution(
                [w / sum(weights) for w in weights], modes
            ),
        }

    async def lookup(
        self,
        agent_id: str,
        activity_id: Optional[str],
        timestamp: int,
        options: list,
        memory_text: Optional[str] = None,
        weather: Optional[dict] = None,
        activity_purpose: str = "",
        seed_parts: tuple = (),
        extra_key: str = "",
        traits_key: str = "",
    ) -> Optional[dict]:
        """
        Hybrid lookup in the cache. Returns a dict {index, mode, score} on hit,
        None on miss. The index is the position in `options` as provided.

        `memory_text=None` (empty long-term memory) → exact match on the
        factual conditions, without embedding. Otherwise → semantic similarity search
        on the LTM, under equal factual conditions, with rejection below the confidence threshold.

        On hit, a point carrying a probability distribution is **drawn
        again** (cf. `_redraw_from_cached`), `seed_parts` providing the seed of the draw.
        Points prior to the switch (frozen decision) return their original option.
        """
        global _LLM_CACHE_HITS, _LLM_CACHE_LOOKUPS
        _LLM_CACHE_LOOKUPS += 1
        t0 = time.perf_counter()
        try:
            if memory_text is None:
                result, miss_reason = await asyncio.to_thread(
                    self._lookup_exact_sync, agent_id, activity_id, timestamp, options, weather, extra_key,
                    traits_key
                )
            else:
                result, miss_reason = await asyncio.to_thread(
                    self._lookup_semantic_sync, agent_id, activity_id, timestamp, options, memory_text, weather, extra_key,
                    traits_key
                )
        except Exception as e:
            logger.warning(f"LLM cache lookup error: {e}")
            LLM_CACHE_MISSES.labels(reason="lookup_error").inc()
            _record_llm_miss("lookup_error")
            self._record_db_error("lookup", e)
            return None
        finally:
            LLM_CACHE_LOOKUP_SECONDS.observe(time.perf_counter() - t0)
        self._record_db_ok()

        if result is None:
            LLM_CACHE_MISSES.labels(reason=miss_reason).inc()
            _record_llm_miss(miss_reason or "unknown")
            # Bounded diagnostics of « exact » misses (bootstrap path): tells a coverage
            # gap (agent never stored) from a key that differs (weather/time slot/state_hash).
            if miss_reason == "no_candidates" and memory_text is None and self._miss_diag_budget > 0:
                self._miss_diag_budget -= 1
                agent_known = str(agent_id) in self._exact_agents
                cause = "clé différente (météo/créneau/state_hash) — agent présent en cache" if agent_known \
                    else "agent ABSENT du cache — trou de couverture (jamais stocké)"
                logger.info(
                    f"[cache] miss exact no_candidates — agent={agent_id} act={activity_id} "
                    f"weekday={self._make_weekday(timestamp)} slice={self._make_time_slice(timestamp)} "
                    f"state_hash={self._make_state_hash(options, weather, extra_key, traits_key)[:12]} → {cause}"
                )
            return None

        # Point carrying a distribution: new draw at each hit.
        if result.get("probabilities"):
            drawn = self._redraw_from_cached(result["probabilities"], options, seed_parts)
            if drawn is not None:
                LLM_CACHE_HITS.labels(activity_purpose=activity_purpose).inc()
                _LLM_CACHE_HITS += 1
                return {**drawn, "score": result.get("score")}
            # No cached option survives in the current list any more.
            LLM_CACHE_MISSES.labels(reason="code_not_in_options").inc()
            _record_llm_miss("code_not_in_options")
            return None

        chosen_code = result.get("chosen_plan_code")
        if not chosen_code:
            LLM_CACHE_MISSES.labels(reason="no_candidates").inc()
            _record_llm_miss("no_candidates")
            return None

        # Inherited point (frozen decision): we return the original option, without a draw.
        for i, opt in enumerate(options):
            if opt.get_code() == chosen_code:
                LLM_CACHE_HITS.labels(activity_purpose=activity_purpose).inc()
                _LLM_CACHE_HITS += 1
                return {"index": i, "mode": result["mode"], "score": result.get("score")}

        # The code was in cache but no longer exists in the current options (itinerary changed)
        LLM_CACHE_MISSES.labels(reason="code_not_in_options").inc()
        _record_llm_miss("code_not_in_options")
        return None

    async def store(
        self,
        agent_id: str,
        activity_id: Optional[str],
        timestamp: int,
        options: list,
        memory_text: Optional[str],
        chosen_plan_code: str,
        mode: str,
        weather: Optional[dict] = None,
        probabilities: Optional[list[float]] = None,
        extra_key: str = "",
        traits_key: str = "",
    ):
        """Async wrapper of _store_sync: persists the decision in Qdrant and records the write latency.

        `memory_text=None` → decision taken without memories (point `memory_empty=True`).
        `probabilities` (aligned with `options`) → distribution replayed at each hit.
        `extra_key` → anticipation signature, included in the `state_hash` (ticket 014).
        """
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(
                self._store_sync,
                agent_id,
                activity_id,
                timestamp,
                options,
                memory_text,
                chosen_plan_code,
                mode,
                weather,
                probabilities,
                extra_key,
                traits_key,
            )
        except Exception as e:
            logger.warning(f"LLM cache store error: {e}")
            self._record_db_error("store", e)
        else:
            self._record_db_ok()
            # Keep the set of covered agents (exact path) up to date for miss classification.
            if memory_text is None:
                self._exact_agents.add(str(agent_id))
        finally:
            LLM_CACHE_INSERT_SECONDS.observe(time.perf_counter() - t0)
