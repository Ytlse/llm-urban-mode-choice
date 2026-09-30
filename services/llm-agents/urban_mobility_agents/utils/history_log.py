import csv
import json
from functools import partial
from pathlib import Path
from typing import Optional
from settings import settings
from sim_clock import wall_clock


_CSV_COLUMNS = ["context", "timestamp", "datetime", "person_id", "activity_id", "message"]


class HistoryStreamLog:
    """
    Logs agent memory events (STM + LTM) in two parallel formats:
    - JSONL (agent_memory_events.jsonl): full payload including data dict
    - CSV (agent_memory_events.csv): flat columns for easy analysis
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, file_path: str = None):
        self.file_path = file_path

        self.log_shortterm_memory = partial(self.log, context="shortterm_memory")
        self.log_longterm_memory = partial(self.log, context="longterm_memory")

    def log(self, context: str, timestamp: int, person_id: str, message: str, activity_id: Optional[str] = None, data: Optional[dict] = None):
        log_entry = {
            "context": context,
            "timestamp": timestamp,
            "person_id": person_id,
            "message": message,
            "activity_id": activity_id,
            "data": data or {}
        }

        jsonl_path = self.file_path or settings.app.agent_memory_events_jsonl
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        csv_path = settings.app.agent_memory_events_csv
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_exists = Path(csv_path).exists() and Path(csv_path).stat().st_size > 0
        # GAMA WALL-CLOCK time (`sim_clock`): the column duplicates `timestamp`, which is
        # the simulated timestamp — both must show the same time.
        dt_str = wall_clock(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else ""
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(_CSV_COLUMNS)
            writer.writerow([context, timestamp, dt_str, person_id, activity_id or "", message])
