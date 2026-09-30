"""
CachedTripHelper — Cache decorator for TripHelper.

Wraps an underlying TripHelper (e.g. OTP) by adding:
- A persistent on-disk cache (OtpPersistentCache) to avoid repeated calls
  to the routing engine for the same origin/destination/time triple.
  Cached results are reused at other departure times by
  simple time shift (delta_ms).
- A blacklist of O/D pairs without a known itinerary, in order to
  short-circuit immediately the requests doomed to fail.
- Two search strategies selectable through the config:
    v1 (recursion_search_depth > 0): recursive search that re-queries OTP
      from the transfer points to explore alternative combinations.
    v2 (default): parallel requests per access/egress mode (foot, bike, …)
      on a single time slot, with final deduplication.
- A Prometheus indicator (trip_cache_hit_ratio) measuring the cache hit rate.
"""

from trip_helper import TripHelper
from trip_helper.otp_persistent_cache import OtpPersistentCache
from models import Location, TravelPlan
from world import WorldModel
from utils import create_background_task, random_uuid
from settings import settings
from trip_helper.strict_replay import exiger_route_en_cache
from loguru import logger
import asyncio
from prometheus_client import Gauge

TRIP_CACHE_HIT_RATIO = Gauge(
    'trip_cache_hit_ratio',
    'Cache hits / total ratio of the CachedTripHelper (0-1)',
)

# Process-wide hits/lookups counters for reporting the OTP cache rate in the logs.
# A hit = itinerary served from the cache OR blacklisted O/D pair (no OTP call).
_OTP_CACHE_HITS = 0
_OTP_CACHE_LOOKUPS = 0


def get_otp_cache_stats() -> tuple[int, int]:
    """Return the cumulative (hits, lookups) of the OTP cache since startup."""
    return _OTP_CACHE_HITS, _OTP_CACHE_LOOKUPS


# Process-wide singleton of the persistent OTP cache, used by OtpCachedTripHelper in OTP
# mode. Initialised per population (init_otp_persistent_cache), like the OSMnx cache.
_otp_persistent_cache: "OtpPersistentCache | None" = None


def init_otp_persistent_cache(cache_dir: str) -> None:
    """Initialise the shared persistent OTP cache (one subfolder per population)."""
    global _otp_persistent_cache
    _otp_persistent_cache = OtpPersistentCache(cache_dir)
    logger.info(f"[otp-cache] Persistent cache enabled at {cache_dir}")


class CachedTripHelper(TripHelper):
    def __init__(self,
                 world_model: WorldModel,
                 trip_helper: TripHelper):
        super().__init__()
        self.trip_helper = trip_helper
        self._world_model = None
        if world_model is not None:
            self.world_model = world_model
        self.recursion_search_depth = settings.gtfs.recursion_search_depth
        self.max_transfers = 5
        self.otp_cache_enabled = settings.gtfs.otp_cache_enabled
        self._stats_cache_hit = (0, 0)

        if self.otp_cache_enabled:
            self.persistent_cache = OtpPersistentCache(settings.gtfs.otp_persistent_cache_dir)
            logger.info(f"[CachedTripHelper]: Persistent cache enabled at {settings.gtfs.otp_persistent_cache_dir}")
        else:
            self.persistent_cache = None
            logger.warning("[CachedTripHelper]: Cache is disabled, all requests will go to the trip_helper directly.")

        if settings.gtfs.recursion_search_depth > 0:
            logger.warning(f"[CachedTripHelper]: Using recursive search strategy with depth {settings.gtfs.recursion_search_depth}")
            self.do_get_iteraries = self.do_get_iteraries_v1
        else:
            logger.warning(f"[CachedTripHelper]: Using time-range expanded based search strategy")
            self.do_get_iteraries = self.do_get_iteraries_v2

    @property
    def world_model(self):
        return self._world_model

    @world_model.setter
    def world_model(self, value: WorldModel):
        self._world_model = value
        if value is not None:
            self.world_grid = value.world_grid
            self.time_grid = value.time_grid

    def get_unique_itineraries(self, itineraries: list[TravelPlan]) -> list[TravelPlan]:
        """
        Get unique itineraries by comparing the start and end locations, and the legs of the itinerary.
        """
        unique_itineraries = {}
        for it in itineraries:
            transits = [leg for leg in it.legs if not leg.is_transfer]
            key = (tuple((leg.transit_route, leg.start_location.stop, leg.end_location.stop) for leg in transits))
            if key not in unique_itineraries:
                unique_itineraries[key] = it
        return list(unique_itineraries.values())
    
    def is_circular_route(self, itinerary: TravelPlan) -> bool:
        """
        Check if the itinerary is circular, meaning the start and end locations are the same.
        """
        all_transits = [leg for leg in itinerary.legs if not leg.is_transfer]
        keys = [
            (leg.start_location.stop, leg.end_location.stop, leg.transit_route)
            for leg in all_transits
        ]
        return len(set(keys)) < len(keys)  # if there are duplicates, it's circular
    
    async def do_get_iteraries_v2(self, origin: Location, destination: Location, departure_time: int, include_car: bool = False, include_bike: bool = True, arrive_by: bool = False, _pipeline_rec=None) -> list[TravelPlan]:
        import time as _time
        max_transfers = self.max_transfers
        time_step = settings.world.time_step
        departure_time = departure_time // time_step * time_step  # round down to the nearest time step

        access_egress_modes = settings.gtfs.transit_access_egress_modes
        otp_sinks = [{} for _ in access_egress_modes] if _pipeline_rec is not None else [None] * len(access_egress_modes)
        direct_sink: dict | None = {} if _pipeline_rec is not None else None

        tasks = []
        # One OTP call per access/egress mode; all queries use the same departure_time T.
        # arrive_by is forwarded so OTP searches backwards from the target arrival time.
        for idx, mode in enumerate(access_egress_modes):
            tasks.append(
                self.trip_helper.get_itineraries(
                    origin=origin,
                    destination=destination,
                    departure_time=departure_time,
                    include_car=include_car,
                    max_transfers=max_transfers,
                    include_direct=False,
                    arrive_by=arrive_by,
                    access_egress_mode=mode,
                    _timing_sink=otp_sinks[idx],
                )
            )
        # Direct routes once only: foot/bicycle are time-independent, car congestion
        # barely differs across ±15 min so one query at T is sufficient.
        # Direct routes are always computed departure-based; when arrive_by=True the
        # caller is responsible for shifting their times after the fact.
        tasks.append(
            self.trip_helper.get_itineraries(
                origin=origin,
                destination=destination,
                departure_time=departure_time,
                include_car=include_car,
                include_bike=include_bike,
                max_transfers=max_transfers,
                include_transit=False,
                arrive_by=False,
                _timing_sink=direct_sink,
            )
        )
        results = await asyncio.gather(*tasks)

        # Map OTP sink results to pipeline record (positional: mode index 0/1/2 → P3A-2/4/6, P3A-3/5/7)
        if _pipeline_rec is not None:
            _otp_fields = [
                ("P3A_2_ms", "P3A_3_ms"),
                ("P3A_4_ms", "P3A_5_ms"),
                ("P3A_6_ms", "P3A_7_ms"),
            ]
            for idx, sink in enumerate(otp_sinks):
                if sink and idx < len(_otp_fields):
                    sem_f, req_f = _otp_fields[idx]
                    setattr(_pipeline_rec, sem_f, sink.get("sem_ms"))
                    setattr(_pipeline_rec, req_f, sink.get("req_ms"))
            if direct_sink:
                _pipeline_rec.P3B_2_ms = direct_sink.get("osmnx_ms")

        # When arrive_by=True, shift direct plans (last task result) so their end_time
        # equals the target arrival time instead of departure_time + duration.
        if arrive_by and results:
            direct_plans = results[-1]
            target_ms = departure_time * 1000
            for plan in direct_plans:
                duration_ms = plan.end_time - plan.start_time
                plan.end_time = target_ms
                plan.start_time = target_ms - duration_ms
                for leg in plan.legs:
                    leg.end_time = leg.end_time - duration_ms
                    leg.start_time = leg.start_time - duration_ms

        # Deduplication across all results
        _t_dedup = _time.monotonic()
        bl = set()
        rs = []
        for plan in [plan for plans in results for plan in plans]:
            code = plan.get_code()
            if code in bl:
                continue
            bl.add(code)
            rs.append(plan)
        if _pipeline_rec is not None:
            _pipeline_rec.P3B_3_ms = (_time.monotonic() - _t_dedup) * 1000

        itineraries = rs
        return itineraries

    async def do_get_iteraries_v1(self,
                               origin: Location,
                               destination: Location,
                               departure_time: int,
                               include_car: bool = False,
                               include_bike: bool = True,
                               arrive_by: bool = False,
                               _pipeline_rec=None) -> list[TravelPlan]:
        max_transfers = self.max_transfers
        recursion_search_depth = self.recursion_search_depth
        itineraries: list[TravelPlan] = await self.trip_helper.get_itineraries(origin, destination, departure_time, include_car=include_car, include_bike=include_bike, max_transfers=max_transfers, arrive_by=arrive_by)
        if not itineraries:
            return []
        
        _stats_number_of_calls = 1
        
        first_index = -1
        while recursion_search_depth > 0 and max_transfers > 2:
            # Itinerary includes transfers and transits. For example if the itinerary has 3 transits, for each depth
            # we will try to keep the previous transits and find new ways from this transit to the destination, then 
            # combine them together.
            new_itineraries = []
            max_transfers -= 1
            recursion_search_depth -= 1
            first_index += 1

            tasks = []
            for it in itineraries:
                # find the first transit in the itinerary
                all_transits = [leg for leg in it.legs if not leg.is_transfer]
                if len(all_transits) <= first_index:
                    continue
                first_transit = all_transits[first_index]

                # find the index of the first transit
                first_transit_index = it.legs.index(first_transit)
                # prepare the coroutine for concurrent execution
                tasks.append((
                    it,
                    first_transit_index,
                    self.trip_helper.get_itineraries(
                        origin=first_transit.end_location,
                        destination=destination,
                        departure_time=first_transit.end_time // 1000,  # convert to seconds
                        max_transfers=max_transfers
                    )
                ))

            # Run all get_itineraries concurrently
            results = await asyncio.gather(*(task[2] for task in tasks))
            _stats_number_of_calls += len(results)

            for (it, first_transit_index, _), new_ways in zip(tasks, results):
                if not new_ways:
                    continue
                for new_way in new_ways:
                    # combine the first part of the itinerary with the new way
                    combined_itinerary = TravelPlan(
                        id=random_uuid(),
                        start_location=it.start_location,
                        end_location=new_way.end_location,
                        start_time=it.start_time,
                        end_time=new_way.end_time,
                        legs=it.legs[:first_transit_index] + new_way.legs
                    )
                    if not self.is_circular_route(combined_itinerary):
                        new_itineraries.append(combined_itinerary)
                
            # merge all itineraries
            itineraries = self.get_unique_itineraries(new_itineraries + itineraries)

        return itineraries

    async def get_itineraries(self,
                              origin: Location,
                              destination: Location,
                              departure_time: int,
                              include_car: bool = False,
                              include_bike: bool = True,
                              arrive_by: bool = False,
                              _pipeline_rec=None,
                              ) -> list[TravelPlan]:
        import time as _time
        _t_cache = _time.monotonic()

        if self.otp_cache_enabled:
            key = OtpPersistentCache.make_key(departure_time, origin, destination, include_car, arrive_by, include_bike)
            bl_key = OtpPersistentCache.make_blacklist_key(origin, destination, departure_time)
            is_blacklisted = await self.persistent_cache.is_blacklisted_async(bl_key)
            cached = None if is_blacklisted else await self.persistent_cache.lookup_async(key)
        else:
            key = bl_key = None
            is_blacklisted = False
            cached = None

        cache_hit = cached is not None or is_blacklisted

        if self.otp_cache_enabled:
            global _OTP_CACHE_HITS, _OTP_CACHE_LOOKUPS
            _OTP_CACHE_LOOKUPS += 1
            if cache_hit:
                _OTP_CACHE_HITS += 1

        if _pipeline_rec is not None:
            _pipeline_rec.P3_1_ms = (_time.monotonic() - _t_cache) * 1000
            _pipeline_rec.P3_cache_hit = cache_hit

        if is_blacklisted:
            return []

        if cached is not None:
            self._stats_cache_hit = (self._stats_cache_hit[0] + 1, self._stats_cache_hit[1] + 1)
            itineraries, stored_departure_time = cached
            # shift all times to match the actual requested departure_time
            delta_ms = (departure_time - stored_departure_time) * 1000
            for itinerary in itineraries:
                itinerary.start_location = origin
                itinerary.end_location = destination
                itinerary.start_time += delta_ms
                itinerary.end_time += delta_ms
                for leg in itinerary.legs:
                    leg.start_time += delta_ms
                    leg.end_time += delta_ms
        else:
            exiger_route_en_cache(departure_time, "OTP/Solari", key)
            itineraries = await self.do_get_iteraries(origin, destination, departure_time, include_car=include_car, include_bike=include_bike, arrive_by=arrive_by, _pipeline_rec=_pipeline_rec)
            self._stats_cache_hit = (self._stats_cache_hit[0], self._stats_cache_hit[1] + 1)
            if itineraries:
                for it in itineraries:
                    it.id = random_uuid()
                if self.otp_cache_enabled:
                    ecriture = self.persistent_cache.store_async(key, itineraries, departure_time)
                    if settings.world.prefixe_commun:
                        await ecriture
                    else:
                        create_background_task(ecriture)
            else:
                if self.otp_cache_enabled:
                    ecriture = self.persistent_cache.blacklist_add_async(bl_key)
                    if settings.world.prefixe_commun:
                        await ecriture
                    else:
                        create_background_task(ecriture)

        _hits, _total = self._stats_cache_hit
        if _total > 0:
            TRIP_CACHE_HIT_RATIO.set(_hits / _total)

        return itineraries


class OtpCachedTripHelper(TripHelper):
    """THIN persistent cache decorator for the OTP mode (main mode).

    Unlike CachedTripHelper (SOLARI mode), it **does not change** the search
    strategy: on a miss it delegates the call verbatim to the underlying TripHelper
    (OTPTripHelper), then stores the result. The cache only sits at the
    caller → helper boundary (controller._compute_move_for_activity), where the requests
    always use the default parameters — the internal OTP sub-calls stay
    uncached and unchanged.

    Key/blacklist/time shift: A stored itinerary is reused at a nearby departure time by simple shift of the
    timestamps (delta_ms), an approximation already accepted by the historical cache.
    """

    def __init__(self, trip_helper: TripHelper):
        super().__init__()
        self.trip_helper = trip_helper

    async def get_itineraries(self,
                              origin: Location,
                              destination: Location,
                              departure_time: int,
                              include_car: bool = False,
                              include_bike: bool = True,
                              arrive_by: bool = False,
                              _timing_sink: dict | None = None,
                              **kwargs) -> list[TravelPlan]:
        global _OTP_CACHE_HITS, _OTP_CACHE_LOOKUPS
        cache = _otp_persistent_cache

        # Cache not initialised yet (e.g. before population setup) → pass-through.
        if cache is None:
            exiger_route_en_cache(departure_time, "OTP non initialisé")
            return await self.trip_helper.get_itineraries(
                origin, destination, departure_time,
                include_car=include_car, include_bike=include_bike,
                arrive_by=arrive_by, _timing_sink=_timing_sink, **kwargs)

        key = OtpPersistentCache.make_key(departure_time, origin, destination, include_car, arrive_by, include_bike)
        bl_key = OtpPersistentCache.make_blacklist_key(origin, destination, departure_time)
        is_blacklisted = await cache.is_blacklisted_async(bl_key)
        cached = None if is_blacklisted else await cache.lookup_async(key)

        _OTP_CACHE_LOOKUPS += 1
        if cached is not None or is_blacklisted:
            _OTP_CACHE_HITS += 1
        TRIP_CACHE_HIT_RATIO.set(_OTP_CACHE_HITS / _OTP_CACHE_LOOKUPS)

        if is_blacklisted:
            return []

        if cached is not None:
            itineraries, stored_departure_time = cached
            delta_ms = (departure_time - stored_departure_time) * 1000
            for itinerary in itineraries:
                itinerary.start_location = origin
                itinerary.end_location = destination
                itinerary.start_time = int(itinerary.start_time + delta_ms)
                itinerary.end_time = int(itinerary.end_time + delta_ms)
                for leg in itinerary.legs:
                    leg.start_time = int(leg.start_time + delta_ms)
                    leg.end_time = int(leg.end_time + delta_ms)
            return itineraries

        # Miss: verbatim OTP call (no change to the search strategy).
        exiger_route_en_cache(departure_time, "OTP", key)
        itineraries = await self.trip_helper.get_itineraries(
            origin, destination, departure_time,
            include_car=include_car, include_bike=include_bike,
            arrive_by=arrive_by, _timing_sink=_timing_sink, **kwargs)
        if itineraries:
            for it in itineraries:
                it.id = random_uuid()
            ecriture = cache.store_async(key, itineraries, departure_time)
        else:
            ecriture = cache.blacklist_add_async(bl_key)
        if settings.world.prefixe_commun:
            await ecriture
        else:
            create_background_task(ecriture)
        return itineraries
