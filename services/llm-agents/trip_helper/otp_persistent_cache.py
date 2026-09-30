import asyncio
import hashlib
import json
import os
import sqlite3
import time as _time
from typing import Optional

from models import Location, TravelPlan


class OtpPersistentCache:
    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "otp_cache.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS otp_cache (
                    key TEXT PRIMARY KEY,
                    plans_json TEXT NOT NULL,
                    departure_time INTEGER NOT NULL,
                    stored_at INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS otp_blacklist (
                    key TEXT PRIMARY KEY,
                    stored_at INTEGER NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def make_key(departure_time: int, origin: Location, destination: Location, include_car: bool, arrive_by: bool, include_bike: bool = True) -> str:
        # include_bike is part of the key: without it, a result computed for an
        # agent without a bike (no bike option) would be served again to an agent with a bike.
        #
        # ⚠ The version of the itinerary data (`terminal_time.data_version()`) is
        # in the key, and it is here that it matters most of the three caches: this cache
        # does not store durations, it stores **serialised TravelPlans**, car and bike
        # options included. Without a version, a warm cache would serve again plans
        # with A SINGLE leg, carrying the former parking time melted into the duration — that
        # is, the whole defect of ticket 013, resurrected after its fix, and
        # without any log reporting it. Bumping the version in
        # config/terminal_time.yaml makes the old rows unreachable without
        # destroying them.
        #
        # NOTE fixed_day: the key includes the absolute date (YYYY-MM-DD) of the simulated
        # departure_time, computed BEFORE the fixed_day remapping done in OTPTripHelper
        # (otp.py). When gtfs.fixed_day is active, two different simulated dates
        # nevertheless produce the same OTP request (same GTFS schedules), but different
        # cache keys → the cache warmed on day D is entirely missed
        # on day D+1.
        # TODO: when fixed_day is active, replace date_str with the fixed date (or the
        # weekday) to share the cache across simulated dates, as
        # OsmnxPersistentCache already does (weekday key, not absolute date).
        #
        # ⚠ The INSTANT, offset included, and no longer a bare date and time
        # (2026-09-04). `datetime.fromtimestamp(departure_time)` read GAMA's wall
        # clock in the PROCESS's time zone: 5 a.m. wall-clock became 6 a.m., and
        # it was at 6 a.m. that OTP was queried. This cache does not store durations but
        # serialised TravelPlans, and `lookup` shifts them by
        # `departure_time - stored_departure_time`: an entry filed under the label
        # "06:00" whereas it answered a 5 a.m. wall-clock departure would be
        # served again, then shifted by one more hour, to a 6 a.m. wall-clock departure.
        # No version protected against that — `data_version()` does not move and
        # `routing_version` does not enter here. Writing the full instant
        # (`2026-03-16T05:00:00+01:00`) changes the FORM of the key: no entry
        # of the old convention is reachable any more, without manual purge, and the
        # offset carried by the key finally distinguishes winter time from summer time
        # — the same wall-clock time is not the same instant depending on the season.
        from sim_clock import to_network_datetime
        from trip_helper.terminal_time import data_version

        dt = to_network_datetime(departure_time)
        bucket = dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)
        raw = (f"{data_version()}|{bucket.isoformat()}|{origin.lat:.5f}|{origin.lon:.5f}"
               f"|{destination.lat:.5f}|{destination.lon:.5f}"
               f"|{int(include_car)}|{int(arrive_by)}|{int(include_bike)}")
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def make_blacklist_key(origin: Location, destination: Location,
                           departure_time: Optional[int] = None) -> str:
        """Blacklist key: the two points AND the time slot.

        ⚠ THE TIME HAS BEEN PART OF THE KEY since 2026-09-04, and the comment it
        replaces said exactly the opposite: "the blacklist says 'OTP does not connect
        these two points', a fact of network topology that depends on no time".
        The intention was right, the trigger was not. `cached_triphelper`
        blacklists a pair as soon as the result is **empty**, without looking at why — and
        `noTransitConnectionInSearchWindow` counts as empty. Yet this reason is
        eminently time-dependent: measured on 2026-09-04, it amounts to 29 points without an itinerary at
        6 a.m. and **341 at 5 a.m.** on the same 2,580 points. A pair blacklisted because
        no coach passes at 5 in the morning therefore returned "no public transport"
        at 5 p.m., without calling OTP and without a single log line. The
        pre-planning waves query precisely the same pair at successive
        times: the defect triggered **within a single run**.
        The 62 pairs already blacklisted had been so at the wrong time; they were
        archived and purged the same day.

        The slot is the same as that of `make_key` — the full network instant,
        offset included, rounded to ten minutes — so that both caches speak of the
        same time. Without `departure_time` the key stays that of topology alone,
        for callers that have no time to give; it is then marked
        as such, and cannot collide with a time-based key.
        """
        if departure_time is None:
            raw = (f"sans-heure|{origin.lat:.5f}|{origin.lon:.5f}"
                   f"|{destination.lat:.5f}|{destination.lon:.5f}")
            return hashlib.sha256(raw.encode()).hexdigest()

        from sim_clock import to_network_datetime

        dt = to_network_datetime(departure_time)
        bucket = dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)
        raw = (f"{bucket.isoformat()}|{origin.lat:.5f}|{origin.lon:.5f}"
               f"|{destination.lat:.5f}|{destination.lon:.5f}")
        return hashlib.sha256(raw.encode()).hexdigest()

    def lookup(self, key: str) -> Optional[tuple[list[TravelPlan], int]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT plans_json, departure_time FROM otp_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        plans = [TravelPlan.model_validate(p) for p in json.loads(row[0])]
        return plans, row[1]

    def store(self, key: str, itineraries: list[TravelPlan], departure_time: int):
        plans_json = json.dumps([p.model_dump() for p in itineraries])
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO otp_cache (key, plans_json, departure_time, stored_at) VALUES (?, ?, ?, ?)",
                (key, plans_json, departure_time, int(_time.time()))
            )
            conn.commit()

    def is_blacklisted(self, bl_key: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute("SELECT 1 FROM otp_blacklist WHERE key = ?", (bl_key,)).fetchone()
        return row is not None

    def blacklist_add(self, bl_key: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO otp_blacklist (key, stored_at) VALUES (?, ?)",
                (bl_key, int(_time.time()))
            )
            conn.commit()

    async def lookup_async(self, key: str) -> Optional[tuple[list[TravelPlan], int]]:
        return await asyncio.to_thread(self.lookup, key)

    async def store_async(self, key: str, itineraries: list[TravelPlan], departure_time: int):
        await asyncio.to_thread(self.store, key, itineraries, departure_time)

    async def blacklist_add_async(self, bl_key: str):
        await asyncio.to_thread(self.blacklist_add, bl_key)

    async def is_blacklisted_async(self, bl_key: str) -> bool:
        return await asyncio.to_thread(self.is_blacklisted, bl_key)
