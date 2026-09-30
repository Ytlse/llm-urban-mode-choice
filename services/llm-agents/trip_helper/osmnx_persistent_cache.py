import asyncio
import hashlib
import json
import os
import sqlite3
import time as _time
from datetime import datetime
from typing import NamedTuple, Optional


class OsmnxCacheEntry(NamedTuple):
    found: bool
    result: Optional[dict]  # None when found=True means impossible route


class OsmnxPersistentCache:
    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "osmnx_cache.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS osmnx_cache (
                    key         TEXT PRIMARY KEY,
                    date        TEXT,        -- NULL for foot/bicycle (time-independent)
                    day_of_week INTEGER,     -- NULL for foot/bicycle
                    time_bucket TEXT,        -- NULL for foot/bicycle; HH:00 (1h) for car
                    mode        TEXT NOT NULL,
                    lat_from    REAL NOT NULL,
                    lon_from    REAL NOT NULL,
                    lat_to      REAL NOT NULL,
                    lon_to      REAL NOT NULL,
                    result_json TEXT,
                    stored_at   INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_osmnx_coords_mode_date
                ON osmnx_cache(lat_from, lon_from, lat_to, lon_to, mode, date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_osmnx_day_bucket_mode
                ON osmnx_cache(day_of_week, time_bucket, mode)
            """)
            conn.commit()

    @staticmethod
    def make_key(
        congestion_dt: datetime,
        mode: str,
        lat_from: float,
        lon_from: float,
        lat_to: float,
        lon_to: float,
    ) -> tuple[str, Optional[str], Optional[int], Optional[str]]:
        """Returns (key, date_str, day_of_week, time_bucket).

        car   → time-aware key: day_of_week + 1h bucket (matches _congestion_factor
                granularity, which depends only on weekday + hour, not the absolute date).
                The absolute date is still stored in the `date` column for reference but is
                intentionally excluded from the key so runs on different calendar dates reuse
                the same cached routes.
        foot/bicycle → time-independent key: coordinates + mode only.

        ⚠ The key is prefixed by ``terminal_time.routing_version()`` — the version of
        ROUTING, not that of terminal time. Without a version, this cache would serve again
        durations computed under another definition of network time: it is addressed by
        (mode, coordinates, slot) and has no way of knowing that the semantics of
        ``duration_s`` changed the day parking was taken out of it.

        But the version must be the RIGHT one: this cache only stores network time,
        independent of terminal time. Indexing it on ``data_version()`` would make
        thousands of routes be recomputed cold at every parking adjustment
        — measured at ~2 h for 930 personas — for an identical result. The caches that
        store PLANS (OTP, LLM decisions) stay, for their part, on ``data_version()``.
        """
        from trip_helper.terminal_time import routing_version

        # `routing_version`, NOT `data_version`: this cache stores pure network time.
        version = routing_version()
        if mode == "car":
            date_str = congestion_dt.strftime('%Y-%m-%d')
            day_of_week = congestion_dt.weekday()
            time_bucket = f"{congestion_dt.hour:02d}:00"
            raw = (
                f"{version}|{day_of_week}|{time_bucket}|{mode}"
                f"|{lat_from:.5f}|{lon_from:.5f}|{lat_to:.5f}|{lon_to:.5f}"
            )
        else:
            date_str = None
            day_of_week = None
            time_bucket = None
            raw = (f"{version}|{mode}"
                   f"|{lat_from:.5f}|{lon_from:.5f}|{lat_to:.5f}|{lon_to:.5f}")
        key = hashlib.sha256(raw.encode()).hexdigest()
        return key, date_str, day_of_week, time_bucket

    def lookup(self, key: str) -> OsmnxCacheEntry:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT result_json FROM osmnx_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return OsmnxCacheEntry(found=False, result=None)
        result = json.loads(row[0]) if row[0] is not None else None
        return OsmnxCacheEntry(found=True, result=result)

    def store(
        self,
        key: str,
        date_str: Optional[str],
        day_of_week: Optional[int],
        time_bucket: Optional[str],
        mode: str,
        lat_from: float,
        lon_from: float,
        lat_to: float,
        lon_to: float,
        result: Optional[dict],
    ):
        result_json = json.dumps(result) if result is not None else None
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO osmnx_cache
                   (key, date, day_of_week, time_bucket, mode,
                    lat_from, lon_from, lat_to, lon_to, result_json, stored_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (key, date_str, day_of_week, time_bucket, mode,
                 lat_from, lon_from, lat_to, lon_to, result_json, int(_time.time()))
            )
            conn.commit()

    def count(self) -> int:
        """Number of routes in the database — logged at startup.

        It is the only figure that distinguishes "warmed cache found" from "starting from
        scratch". Its absence let 196 runs recompute cold a cache of 83,478 routes
        that existed on disk, in a file the runtime did not read.
        """
        with self._get_conn() as conn:
            return conn.execute("SELECT count(*) FROM osmnx_cache").fetchone()[0]

    async def lookup_async(self, key: str) -> OsmnxCacheEntry:
        return await asyncio.to_thread(self.lookup, key)

    async def store_async(
        self,
        key: str,
        date_str: Optional[str],
        day_of_week: Optional[int],
        time_bucket: Optional[str],
        mode: str,
        lat_from: float,
        lon_from: float,
        lat_to: float,
        lon_to: float,
        result: Optional[dict],
    ):
        await asyncio.to_thread(
            self.store, key, date_str, day_of_week, time_bucket,
            mode, lat_from, lon_from, lat_to, lon_to, result,
        )
