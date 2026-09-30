"""
pipeline_logger.py — CSV timing log for the transport scenario pipeline.

One row per LLM agent per scheduling cycle. Each checkpoint is a Unix
wall-clock timestamp (time.time(), seconds as float). Durations and
inter-segment gaps are computed in the analysis notebook via subtraction.

P5 Celery-worker fields stay as durations (separate process — no shared
clock reference).

Enable via settings.app.pipeline_log_enabled = true.

Checkpoint timeline (sequential for one agent):
  T0          /sync handler start (before body read)
  T_parse     after orjson.loads + Pydantic validation
  T_flag      after eligible-agent scan (begin() called)
  T_otp_start before trip_helper.get_itineraries()
  T_otp_end   after trip_helper.get_itineraries()
  T_ltm_start before ChromaDB query (None when long_term_memory disabled)
  T_ltm_end   after ChromaDB query
  T_llm_start before execute_async() call
  T_llm_sent  after POST submitted (T_llm_start + _post_ms/1000)
  T_llm_result after long-poll returns
  T_extract_end after plan-index extraction
  T_enqueue   after PersonMove appended to queue
  T_fin       after WebSocket send to GAMA
"""

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PipelineRecord:
    agent_id: str = ""
    sim_time: Optional[int] = None        # simulation clock sent by GAMA (seconds)

    # Wall-clock checkpoints (Unix epoch, float seconds)
    T0: Optional[float] = None            # /sync handler start
    T_parse: Optional[float] = None       # after body parse + model validation
    T_flag: Optional[float] = None        # after eligible-agent scan
    T_otp_start: Optional[float] = None    # before OTP / trip-helper call
    T_transit_sem: Optional[float] = None  # OTP semaphore acquired (end of queue wait)
    T_transit_end: Optional[float] = None  # OTP HTTP request done (parallel with osmnx)
    T_osmnx_sem: Optional[float] = None    # last OSMnx semaphore acquired, HTTP mode only
    T_osmnx_end: Optional[float] = None    # OSMnx processing done (parallel with transit)
    T_otp_end: Optional[float] = None      # after gather = max(transit_end, osmnx_end)
    T_ltm_start: Optional[float] = None   # before ChromaDB (None if disabled)
    T_ltm_end: Optional[float] = None     # after ChromaDB
    T_llm_start: Optional[float] = None   # before execute_async POST
    T_llm_sent: Optional[float] = None    # after POST submitted
    T_llm_result: Optional[float] = None  # after long-poll returns
    T_extract_end: Optional[float] = None # after plan-index extraction
    T_enqueue: Optional[float] = None     # after PersonMove enqueued
    T_fin: Optional[float] = None         # after WebSocket send

    # P5 Celery-worker timing (durations in ms — separate process)
    P4_4_ms: Optional[float] = None       # micro-batch wait
    P5_1_ms: Optional[float] = None       # provider availability wait
    P5_3_ms: Optional[float] = None       # prompt build (merge + render)
    P5_4_ms: Optional[float] = None       # LLM provider call (retries included)
    P5_5_ms: Optional[float] = None       # demux + Redis persist + Pub/Sub publish
    P5_llm_provider: str = ""
    P5_llm_retries: int = 0
    P5_tokens_in: int = 0
    P5_tokens_out: int = 0

    plan_selected_index: Optional[int] = None
    selection_method: str = ""


class PipelineLogger:
    """Singleton that collects per-agent timing and writes one CSV row on completion."""

    _instance: Optional["PipelineLogger"] = None

    def __init__(self, csv_path: Path) -> None:
        self._path = csv_path
        self._records: dict[str, PipelineRecord] = {}
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def init(cls, csv_path: Path) -> "PipelineLogger":
        cls._instance = cls(csv_path)
        return cls._instance

    @classmethod
    def get(cls) -> Optional["PipelineLogger"]:
        return cls._instance

    # ------------------------------------------------------------------
    # Per-agent record management
    # ------------------------------------------------------------------

    def begin(
        self,
        agent_id: str,
        sim_time: int,
        t0: float | None = None,
        t_parse: float | None = None,
        t_flag: float | None = None,
    ) -> PipelineRecord:
        """Create a new record for an agent. Call once per scheduling cycle."""
        now = time.time()
        rec = PipelineRecord(
            agent_id=agent_id,
            sim_time=sim_time,
            T0=t0 or now,
            T_parse=t_parse,
            T_flag=t_flag or now,
        )
        self._records[agent_id] = rec
        return rec

    def get_record(self, agent_id: str) -> Optional[PipelineRecord]:
        return self._records.get(agent_id)

    def mark_enqueued(self, agent_id: str) -> None:
        """Record T_enqueue timestamp. Call when Action is appended to the queue."""
        rec = self._records.get(agent_id)
        if rec is not None:
            rec.T_enqueue = time.time()

    def complete(self, agent_id: str) -> None:
        """Finalise the record and write a CSV row. Call after WebSocket send."""
        rec = self._records.pop(agent_id, None)
        if rec is None:
            return
        rec.T_fin = time.time()
        self._write_row(rec)

    def apply_timing_p5(self, agent_id: str, timing_p5: dict) -> None:
        """Merge P5 worker timing (from task result) into the agent's record."""
        rec = self._records.get(agent_id)
        if rec is None:
            return
        rec.P4_4_ms = timing_p5.get("P4_4_ms")
        rec.P5_1_ms = timing_p5.get("P5_1_ms")
        rec.P5_3_ms = timing_p5.get("P5_3_ms")
        rec.P5_4_ms = timing_p5.get("P5_4_ms")
        rec.P5_5_ms = timing_p5.get("P5_5_ms")
        rec.P5_llm_provider = timing_p5.get("provider", "")
        rec.P5_llm_retries = timing_p5.get("retries", 0)
        rec.P5_tokens_in = timing_p5.get("tokens_in", 0)
        rec.P5_tokens_out = timing_p5.get("tokens_out", 0)

    # ------------------------------------------------------------------
    # CSV writing
    # ------------------------------------------------------------------

    def _write_row(self, rec: PipelineRecord) -> None:
        row = asdict(rec)
        if self._writer is None:
            self._file = open(self._path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
