import os
import time
import yaml
import httpx
from settings import settings
from inputs.gtfs.reader import GTFSData
from world import *
from trip_helper.base import TripHelper
from trip_helper.cached_triphelper import CachedTripHelper, OtpCachedTripHelper
from trip_helper.otp import OTPTripHelper
from inputs.population import EqasimJSONPopulationLoader, PersonCloseToTheStopFilter
from trip_helper import SolariTripHelper
from urban_mobility_agents.simulation_controller import SimulationLoopV1
from urban_mobility_agents.core.scenario import BaseScenario
from urban_mobility_agents.agents.llm_agent import LlmAgent
from loguru import logger
from dataclasses import dataclass
from typing import Optional


def _otp_endpoints_to_wait() -> list[str]:
    """Return the list of OTP transmodel endpoints that will actually be used."""
    env = os.getenv("OTP_ENDPOINTS", "")
    if env:
        return [e.strip() for e in env.split(",") if e.strip()]
    return [settings.gtfs.otp_endpoint]


def wait_for_otp(endpoint: str, timeout: int = 300, interval: int = 5) -> bool:
    """Poll a single OTP endpoint until it responds, logging progress."""
    health_url = endpoint.replace("/otp/transmodel/v3", "/otp")
    logger.info(f"Waiting for OTP at {health_url} ...")
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = httpx.get(health_url, timeout=3.0)
            if resp.status_code < 500:
                logger.info(f"✅ OTP reachable (HTTP {resp.status_code}) after {attempt} attempt(s) — {health_url}")
                return True
        except Exception as e:
            logger.debug(f"OTP not ready yet (attempt {attempt}): {e} — {health_url}")
        time.sleep(interval)
    logger.error(f"❌ OTP not reachable after {timeout}s — {health_url}")
    return False


def wait_for_all_otp(timeout: int = 300, interval: int = 5) -> bool:
    """Wait for every OTP endpoint listed in OTP_ENDPOINTS (or the single otp_endpoint)."""
    endpoints = _otp_endpoints_to_wait()
    results = [wait_for_otp(ep, timeout=timeout, interval=interval) for ep in endpoints]
    return all(results)


@dataclass
class StaticWorldData:
    gtfs_data: GTFSData
    trip_helper: TripHelper


def init_static_data() -> StaticWorldData:
    """Initialises the heavy shared data (GTFS, routers) at server startup."""
    logger.info("Initialising the static world data...")
    gtfs_data = GTFSData.DEFAULT()
    
    trip_helper = None
    if settings.gtfs.mode == "OTP":
        logger.info("Using OTP trip helper")
        wait_for_all_otp()
        trip_helper = OTPTripHelper(gtfs_data=gtfs_data)
        # Persistent OTP cache in main mode: a THIN decorator that does not change the
        # search strategy (delegates verbatim on miss). The cache itself is
        # initialised per population in handle.application (init_otp_persistent_cache).
        if settings.gtfs.otp_cache_enabled:
            trip_helper = OtpCachedTripHelper(trip_helper)
            logger.info("[cache] OTP itinerary cache wired (OtpCachedTripHelper)")
        else:
            logger.warning("[cache] OTP itinerary cache disabled (gtfs.otp_cache_enabled=false)")
    else:
        logger.info("Using Solari trip helper")
        trip_helper = CachedTripHelper(
            world_model=None, # Injected later if needed, or CachedTripHelper is changed not to depend on it
            trip_helper=SolariTripHelper(
                endpoint=settings.gtfs.solari_endpoint,
                gtfs_data=gtfs_data,
            ),
        )
    return StaticWorldData(gtfs_data=gtfs_data, trip_helper=trip_helper)


def _save_scenario_params(
    population_size: int,
    llm_agents: int,
    long_term_memory_enabled: bool,
    long_term_self_reflect_enabled: bool,
    simulation_max_days: Optional[int] = None,
    accidents_enabled: bool = False,
) -> None:
    """Persists the effective GAMA scenario parameters in the experiment directory."""
    workdir = getattr(settings, 'workdir', None)
    if workdir is None:
        return
    params = {
        'population_size': population_size,
        'number_of_llm_based_agents': llm_agents,
        'long_term_memory_enabled': long_term_memory_enabled,
        'long_term_self_reflect_enabled': long_term_self_reflect_enabled,
        # Always written, including as `False` (ticket 070, R5). Unlike
        # `simulation_max_days`, whose absence signals a run predating its existence,
        # an unrecorded accident regime would cast doubt on ALL runs: the reader of an
        # archive could no longer tell whether silence means "disabled" or "the
        # question did not arise yet".
        'accidents_enabled': accidents_enabled,
    }
    # Stopping horizon: recorded as soon as GAMA sends it (ticket 008, A5). Absent
    # from earlier runs — do not write a default value, which would suggest a known
    # scope.
    if simulation_max_days is not None:
        params['simulation_max_days'] = simulation_max_days
    scenario_file = workdir / 'scenario_params.yaml'
    with open(scenario_file, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Scenario parameters saved to {scenario_file}")


def init_dynamic_scenario(
    static_data: StaticWorldData,
    sim_base_timestamp: int = 0,
    population_size: int = None,
    part_of_llm_agents: float = None,
    long_term_memory_enabled: bool = None,
    long_term_self_reflect_enabled: bool = None,
    simulation_max_days: Optional[int] = None,
    accidents_enabled: Optional[bool] = None,
) -> BaseScenario:
    """Initialises a new simulation run with its dynamic agents."""
    logger.info("Creating a new dynamic scenario...")
    gtfs_data = static_data.gtfs_data

    # Override parameters if provided by GAMA, otherwise keep the current settings values
    if population_size is not None:
        settings.data.population_size = population_size
    if part_of_llm_agents is not None:
        settings.data.number_of_llm_based_agents = int(settings.data.population_size * part_of_llm_agents)
    if long_term_memory_enabled is not None:
        settings.agent.long_term_memory_enabled = long_term_memory_enabled
    if long_term_self_reflect_enabled is not None:
        settings.agent.long_term_self_reflect_enabled = long_term_self_reflect_enabled
    if accidents_enabled is not None:
        settings.accidents.enabled = accidents_enabled

    # Accident registry of the run (ticket 070). Open or closed depending on the switch, and it
    # logs so in both cases: a run without accidents must say so, otherwise nothing
    # distinguishes "disabled" from "the feature is broken".
    from trip_helper import accidents as accidents_module

    accidents_module.reinitialiser()
    accidents_module.initialiser()

    # Declared event of the run (ticket 100; ticket 079 for shocks). Same rule as the
    # accidents: open or closed, it logs so. A loading refusal STOPS here — a clean failure
    # at startup is better than a sixty-day run that does nothing and for which nobody
    # will know why.
    from llm import evenements as evenements_module

    evenements_module.reinitialiser()
    evenements_module.initialiser(workdir=settings.workdir)

    _save_scenario_params(
        population_size=settings.data.population_size,
        llm_agents=settings.data.number_of_llm_based_agents,
        long_term_memory_enabled=settings.agent.long_term_memory_enabled,
        long_term_self_reflect_enabled=settings.agent.long_term_self_reflect_enabled,
        simulation_max_days=simulation_max_days,
        accidents_enabled=settings.accidents.enabled,
    )

    # World extent (ticket 031, part 2): the POLYGON of the survey's 453 municipalities, joined
    # with the extent of the GTFS stops ± 0.05° (~5 km). Before that day, the world was this
    # single rectangle of Tisséo stops — 221 municipalities out of 453 — and `WorldGrid` asserted
    # that no location left it: a 3rd-ring home made it fail. The admission filter does not
    # read this extent: it reads the `residence_zone` trait (eqasim_loader.perimeter_verdict).
    from inputs.population.perimeter import world_extent

    min_lon, min_lat, max_lon, max_lat = gtfs_data.get_bounding_box()
    buffer = 0.05  # degrees ~ 5km
    stops_bbox = BBox(
        min_lon=min_lon - buffer,
        min_lat=min_lat - buffer,
        max_lon=max_lon + buffer,
        max_lat=max_lat + buffer,
    )
    world_bbox = world_extent(stops_bbox)

    world_grid = WorldGrid(world_bbox)
    time_grid = TimeGrid()

    stop_filter = PersonCloseToTheStopFilter(
        max_distance=5000,  # 5000 meters — TODO: remove when car mode is added
        stop_locations=gtfs_data.all_stop_locations(),
    )
    population = WorldPopulation(
        #EqasimJSONPopulationLoader(filters=[stop_filter])
        EqasimJSONPopulationLoader(filters=[])
    ).init(world_bbox=world_bbox)

    # Ticket 100, lot 4 — household index, as soon as the population is loaded. The household
    # is the ONLY social group of the simulation that carries a stable identifier, and two
    # mechanisms depend on it: the `foyers` exposure rule and circulation within the
    # household. Indexed open or closed, and logged in both cases: without `household_id`,
    # an [ALARME] says so rather than letting a whole run unfold with an empty channel
    # and without the slightest symptom.
    from llm import foyer as foyer_module

    foyer_module.reinitialiser()
    foyer_module.initialiser(population.get_people_list())

    for person in population.get_people_list():
        scheduler = PersonScheduler(person)
        current_act = scheduler.get_current_activity(sim_base_timestamp) if sim_base_timestamp else None
        person.state.last_location = current_act.location if current_act else scheduler.get_home_location()

    world_model = WorldModel(
        world_grid=world_grid,
        time_grid=time_grid,
        gtfs_data=gtfs_data,
        bbox=world_bbox,
        population=population,
    )

    # If Solari is used, update its world_model reference dynamically if needed
    if hasattr(static_data.trip_helper, 'world_model'):
        static_data.trip_helper.world_model = world_model

    loop = SimulationLoopV1(
        world_model=world_model,
        trip_helper=static_data.trip_helper,
        agent=LlmAgent(),
    )

    return loop
