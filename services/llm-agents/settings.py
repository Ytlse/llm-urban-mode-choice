import os
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import yaml
from loguru import logger
from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

base_dir = os.path.dirname(os.path.abspath(__file__))


def _racine_partagee(*repere: str) -> Path:
    """Ancestor of `base_dir` under which `repere` exists, seen from the host as from the container.

    Same reason as `_candidats_providers_yaml`: `base_dir` is `<repo>/services/llm-agents`
    on the host but `/app` in the container, where the repository folders are mounted next to
    the code. A fixed `..` cannot land right on both sides — the number of levels differs.
    So the marker is searched for instead of counted (ticket 039: `llm-agents` moved down
    one level on the host, not in the container).

    Returns the first ancestor that contains `repere`, and failing that the immediate parent —
    the old behaviour, so that a missing marker (partial repository, test) does not raise here
    but fails later where the file is actually read.
    """
    depart = Path(base_dir).resolve()
    for ancetre in [depart, *depart.parents[:4]]:
        if ancetre.joinpath(*repere).exists():
            return ancetre
    return depart.parent


def _resolve_experiments_dir(package_dir: Path) -> Path:
    """`experiments/` folder of the repository, seen from the host as from a container.

    The container mounts `./experiments` on `/app/experiments`, next to the code
    (`/app`): `package_dir / "experiments"` is then the right answer. On the
    host, the code lives in `<repo>/llm-agents/` and the experiments in
    `<repo>/experiments/` — one level up. Taking `package_dir` without
    telling the two cases apart created a parallel `services/llm-agents/experiments/`,
    whose paths did not resolve from `services/GAMA/CityTransport/`.

    `APP_EXPERIMENTS_DIR` bypasses the detection if needed.
    """
    override = os.environ.get("APP_EXPERIMENTS_DIR", "").strip()
    if override:
        return Path(override)
    parent_candidate = package_dir.parent / "experiments"
    if parent_candidate.is_dir():
        return parent_candidate
    return package_dir / "experiments"


def _run_artifacts_disabled() -> bool:
    """True when importing this module must create no run artefact.

    A test suite that imported `settings` created a run folder and
    repointed the `services/GAMA/CityTransport/results` symlink — stealing its output from the
    running simulation. A test import observes the configuration, it
    does not open a run.
    """
    if os.environ.get("APP_NO_RUN_ARTIFACTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    import sys

    if "pytest" in sys.modules:
        return True
    # `python -m unittest`: detected from the command line, not from
    # sys.modules — any dependency may import `unittest`.
    argv0 = os.path.basename(sys.argv[0] or "")
    return argv0 in ("pytest", "unittest") or argv0.startswith("pytest")


def merge_configs(*config_paths: str) -> dict[str, Any]:
    """Merge multiple YAML files, with later files overriding earlier ones."""
    merged_config = {}

    for path in config_paths:
        if Path(path).exists():
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                if config:
                    merged_config = deep_merge(merged_config, config)

    return merged_config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class WorkdirPathResolutionMixin:
    """Mixin to handle path resolution in nested models."""

    # Define path fields at class level
    _in_workdir_path_fields: ClassVar[list[str]] = []

    def resolve_paths(self, workdir: Path):
        """Resolve relative paths to absolute paths."""
        for field_name in self._in_workdir_path_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if value is not None and not Path(value).is_absolute():
                    resolved_path = workdir / value
                    setattr(self, field_name, str(resolved_path))


def _candidats_providers_yaml() -> list[Path]:
    """Locations probed for providers.yaml, in order of priority.

    DEPLOYMENT configuration of the gateway (outside the package since ticket 037, iteration 2):
    the variable the gateway reads itself, then the repository root, then the container mount.
    The ancestors of `base_dir` are walked up instead of a fixed `..`: `base_dir` is
    `<repo>/llm-agents` on the host (root ONE level up) but `/app` in the container, where
    `/app/../config` = `/config` does not exist and where the file is mounted under `/app/config`.
    """
    candidates: list[Path] = []
    from_env = os.environ.get("LLM_GATEWAY_PROVIDERS_FILE")
    if from_env:
        candidates.append(Path(from_env))
    racine = Path(base_dir).resolve()
    for ancetre in [racine, *racine.parents[:3]]:
        candidates.append(ancetre / "config" / "llm_gateway" / "providers.yaml")
    candidates.append(Path("/app/config/llm_gateway/providers.yaml"))
    vus: set = set()
    return [c for c in candidates if not (str(c) in vus or vus.add(str(c)))]


def _find_providers_yaml() -> Path | None:
    """Looks for providers.yaml in the standard locations and returns the first one found."""
    candidates = _candidats_providers_yaml()
    for p in candidates:
        if p.exists():
            return p.resolve()
    # Without this file, `settings.llm.providers` stays EMPTY and every gateway call fails:
    # saying so here saves looking for a quota outage (outage of 2026-09-07, container not recreated).
    logger.error(
        "[ALARME] providers.yaml introuvable — LLM_GATEWAY_PROVIDERS_FILE={} ; sondés : {}",
        os.environ.get("LLM_GATEWAY_PROVIDERS_FILE") or "non définie",
        ", ".join(str(c) for c in candidates),
    )
    return None


class ProviderConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    rpm_limit: int
    base_url: str
    default_model: str
    weight: float = 1.0
    batch_max_agents: int = 5
    concurrency_limit: int = 2
    disable_timeout: int = 180
    # Client wait for a task served by this instance (s); None = client default.
    wait_timeout: float | None = None
    adapter: str = ""


class LlmConfig(BaseSettings, WorkdirPathResolutionMixin):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    circuit_breaker_threshold: float = 0.95
    max_retries: int = 50
    backoff_base_seconds: float = 1.0
    # Ticket 084 — gateway instances allowed to serve the decisions of THIS run.
    # Empty: no restriction, the cascade chooses freely. Filled in, the gateway refuses
    # to serve from another instance, AT SELECTION and not after the fact.
    #
    # Used to run an experiment on one family of models while the gateway serves
    # others to another job: the providers file is global to the stack, this setting
    # is not.
    #
    # ── Ticket 095, batch C — ONE MODEL PER FUNCTION ─────────────────────────
    # Accepts, besides a flat list, a TABLE `category → instances`:
    #
    #   {"defaut": ["a", "b"], "stm_reflection": ["c", "d"]}
    #
    # The `defaut` key is the fallback of any category not named — without it, a category
    # missing from the table would be restricted by nothing, which is the opposite of what
    # an allow-list is asked to do.
    #
    # ⚠ This is NOT an infrastructure setting. Changing the model of the STM reflections changes
    # the content of the memory, hence the decisions: the binding is frozen before the campaign,
    # goes into `identite_run.json`, and stays identical in all arms — otherwise the measured gap
    # can no longer be attributed to the shock. Adopting it requires redoing the noise floor (E1).
    instances_admises: list[str] | dict[str, list[str]] = []

    # 2026-09-25 — exact-prompt replay space (llm_gateway/core/rejeu_ab.py), set by the
    # memory orchestrator (`rejeu_ab: true`) and IDENTICAL in both arms of an A/B. A
    # question already asked word for word in this space receives the recorded answer, without
    # a call: the control replays the treated arm as long as nothing separates them. Empty: no replay.
    # BARE name (`REJEU_AB`), like `INSTANCES_ADMISES`: that is what compose passes on.
    rejeu_ab: str = ""
    # The control forces a cache hit before the first day of exposure.
    rejeu_strict_avant_ts: int | None = None

    # Short cooldown of the faulty provider during a failover (parse error / 4xx):
    # forces the rotation to pick another model on retry (cf. worker/task_worker).
    provider_switch_cooldown_seconds: int = 30
    batch_max_agents: int = 5
    batch_delay_seconds: float = (
        3.0  # miroir de llm_gateway.config (fenêtre d'accumulation du micro-batching)
    )

    # Clés API lues depuis l'env : PROVIDER_KEYS__groq=gsk-...
    provider_keys: dict[str, SecretStr] = {}

    # Construit après validation depuis providers.yaml + provider_keys
    providers: dict[str, ProviderConfig] = {}

    @model_validator(mode="after")
    def build_providers(self) -> "LlmConfig":
        """Merges the providers.yaml entries with the API keys from the env to build the ProviderConfig."""
        yaml_path = _find_providers_yaml()
        if yaml_path is None:
            return self
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        defaults = data.get("providers", {})
        result = {}
        for name, entry in defaults.items():
            adapter_name = entry.get("adapter", name)
            key = self.provider_keys.get(name) or self.provider_keys.get(
                adapter_name, SecretStr("")
            )
            result[name] = ProviderConfig(api_key=key, **entry)
        self.providers = result
        return self


class ServerConfig(BaseSettings, WorkdirPathResolutionMixin):
    # HTTP settings
    http_host: str = "localhost"
    http_port: int = 8002

    # GAMA websocket settings
    gama_ws_url: str = "ws://localhost:3001"


class WorldConfig(BaseSettings, WorkdirPathResolutionMixin):
    # General settings
    geo_crs: str = "EPSG:4326"
    geo_projection: str = "EPSG:3857"
    # Grid settings
    grid_size: int = 1000  # 1km
    time_step: int = 900  # 15 minutes
    # A/B opt-in: no asynchronous computation of the current step spills over onto the next.
    prefixe_commun: bool = False
    prefixe_commun_sync_timeout_s: float = 24.0
    prefixe_commun_stall_timeout_s: float = 1800.0
    # Dynamic throttling: min_interval = cap * min(1, n / population)^k
    # where n = itineraries in progress (backlog). Threshold relative to the
    # population so the cap is always reachable (n can never exceed population).
    # k   : convexity exponent (>1 — higher = later and steeper braking).
    #       With k=1.5, cap=30: ~1s at 10%, ~2.7s at 20%, ~5s at 30%, ~7.6s at 40%,
    #       ~10.6s at 50%, ~21.5s at 80% (where drain mode takes over and holds at
    #       cap until the backlog is back below drain_release_ratio), 30s at 100%.
    # cap : hard ceiling in seconds (keeps delay below GAMA's HTTP read timeout).
    min_internal_coeff_k: float = 1.5
    min_internal_coeff_cap: float = 30.0
    # Drain mode (hysteresis on top of the progressive brake): once the backlog
    # reaches drain_trigger_ratio of the population, every /sync response is held
    # up to `cap` seconds (GAMA's HTTP read timeout is the hard limit per response)
    # until the backlog falls back below drain_release_ratio — i.e. the pile must
    # be ~80% drained (release=0.2) before control returns to GAMA at full speed.
    # The backlog alarm ([ALARME]) fires/clears on the same thresholds.
    # drain_trigger_ratio <= 0 disables the mechanism.
    drain_trigger_ratio: float = 0.8
    drain_release_ratio: float = 0.2
    # Backlog re-sampling period (seconds) while holding a /sync response in drain mode.
    drain_poll_interval: float = 1.0
    # Number of concurrent Worker coroutines consuming the activity queue.
    # Each worker independently picks items from the same asyncio.Queue, so up to
    # worker_concurrency LLM+OTP computations run in parallel (matching the old
    # fire-and-forget asyncio.create_task approach).
    worker_concurrency: int = 8

    # Bootstrap concurrency (/init): at startup, ~N agents launch their first
    # itinerary almost simultaneously. Without a cap, this burst saturates the providers'
    # RPM/TPM quotas and triggers a cascade of 429/5xx → massive fallbacks.
    # This semaphore bounds the in-flight OTP+LLM pipelines and spreads the load in waves.
    bootstrap_concurrency: int = 30

    # --- Ticket 003: EDF scheduling and predictive backpressure ---
    # EDF (Earliest Deadline First) dispatcher: planning tasks are
    # served by increasing deadline (simulated departure time) through a priority
    # queue consumed by worker_concurrency tasks, instead of the FIFO semaphore.
    # false = historical behaviour (direct spawn + semaphore, arrival order).
    edf_enabled: bool = True
    # Predictive backpressure: hold the /sync only if the estimated time to
    # drain the queue threatens a deadline (EDF feasibility test), instead
    # of the progressive brake cap·ratio^k. false = historical progressive brake.
    # The hysteresis drain mode (drain_*) remains the last safety net.
    predictive_backpressure_enabled: bool = True
    # Time constant (s) of the EWMA of the completion rate D (tasks/s).
    # Short enough to react to a collapse of a provider quota (per minute).
    throughput_ewma_tau_s: float = 90.0
    # Floor of D (tasks/s): avoids an estimated T = ∞ when there is no recent completion.
    throughput_floor_per_s: float = 0.05
    # Marge multiplicative du test de faisabilité EDF : rétention si T_k·marge > slack_k.
    predictive_margin: float = 1.4
    # Cumulative predictive hold (s) on a /sync beyond which GAMA is
    # notified of the degraded regime (topic system/throttle, rising edge).
    throttle_notify_threshold_s: float = 5.0
    # Refresh period (s) of the system/throttle message while the degraded
    # regime persists (avoids spam at every /sync).
    throttle_notify_refresh_s: float = 30.0

    # Hold on imminent departure (2026-09-25) — EXPERIMENTS ONLY: armed by the same
    # lock as the stop on fallback (`EXPERIMENT_STOP_ON_FALLBACK=1`, set by
    # `run_sequential_cohort.py`), silent in an ordinary run. As long as a departure
    # decision has not been returned and that departure falls within this horizon (SIMULATED
    # seconds), each /sync response is held down to `min_internal_coeff_cap`. If releasing GAMA
    # would make it pass the departure time, the run stops (hibernation, reason
    # `decision_en_retard`) rather than serving the trip late. 3600 s = 4 holds of
    # 30 s, i.e. ~2 real minutes left to the decision before the stop.
    departure_hold_lookahead_s: float = 3600.0
    # Run classique : l'alarme « départ servi en retard » (front montant) se réarme après ce
    # nombre de secondes SIMULÉES sans nouveau retard.
    late_departure_alarm_rearm_s: float = 3600.0

    # Cockpit: an agent is "stuck" if it has had no successful planning
    # for more than this number of hours of SIMULATED time (metric controller_agents_stuck).
    stuck_agent_threshold_hours: float = 20.0

    # Lost-arrival watchdog: an agent "travelling" whose expected arrival
    # (expected_arrive_at of the pushed move) is exceeded by more than this margin (hours
    # of SIMULATED time) is considered lost (move never received by GAMA, e.g. WebSocket
    # cut): [ALARME] + forced restart of the cycle by the fallback scan.
    # 0 disables the watchdog. The margin must stay above the legitimate delays
    # in GAMA (waiting for a PT vehicle, congestion).
    arrival_watchdog_hours: float = 1.0


class GTFSConfig(BaseSettings, WorkdirPathResolutionMixin):
    _in_workdir_path_fields: ClassVar[list[str]] = ["solari_cache_file"]

    mode: str = "SOLARI"  # SOLARI or OTP

    # GTFS settings
    gtfs_file: str = (
        str(_racine_partagee("data", "gtfs") / "data" / "gtfs" / "tisseo_gtfs") + os.sep
    )
    # Table of route shapes published by `scripts/data/gama/export_trip_info.py`:
    # `route_id → {shape_id → {stop_id: stop_sequence}}` for the THREE networks, plus
    # the catalogue of the stops they serve. Without it, `get_shape_id_from_route_info`
    # only knows the primary feed above and returns `[]` for 80 of the 199 lines that
    # carry trips — no agent can board a TER or a liO coach.
    #
    # ⚠ INVARIANT: the file lives in `services/GAMA/CityTransport/includes/`, next to
    # `routes.shp` and `trip_info.json`, whose fingerprint it records to prove it comes
    # from the same generation. The `controller` container mounts `./services/GAMA` on
    # `/services/GAMA`; on the host the same folder is at `<repo>/services/GAMA`. Both
    # are found by searching for the marker, not by counting levels (cf. `_racine_partagee`).
    # Moving it out of this folder would break the freshness check, which looks for its
    # witnesses in ITS own folder.
    shape_lookup_file: str = str(
        _racine_partagee("services", "GAMA", "CityTransport")
        / "services"
        / "GAMA"
        / "CityTransport"
        / "includes"
        / "shape_lookup.json"
    )
    # GTFS `route_type` → name of the mode served to the agent in its prompt (« Trajet en
    # Train 12 »). A missing key shows as « Unknown »: the TER (route_type=2)
    # comes in here on 2026-09-04 with the `rail` mode requested from OTP (ticket 031, q. 16).
    gtfs_modality_name_map: dict[str, str] = {
        "0": "T1/Tram",
        "1": "Metro",
        "2": "Train",
        "3": "Bus",
        "6": "Teleo",
    }

    # RAPTOR provider settings
    solari_endpoint: str = "http://localhost:8000/v1/plan"
    solari_cache_file: str = "raptor_cache.pickle"

    # OTP provider settings
    otp_endpoint: str = "http://localhost:8080/otp/transmodel/v3"
    otp_max_concurrent: int = 30  # max simultaneous get_itineraries calls

    # OSMnx direct routing cache (walk/bike/car graphs, persisted across restarts)
    osmnx_cache_dir: str = "/app/osmnx_cache"
    # Key of the OSMnx graph served at run time (`graphs_<key>.pkl` / `boundary_<key>.pkl` in
    # `osmnx_cache_dir`). Empty → the graph of the polygon of the 453 communes
    # (`geography.PERIMETER_CACHE_KEY`, ticket 031 part 2). The key of the historical 30 km
    # disc (`geography.PRODUCTION_CACHE_KEY_30KM`) is still accepted for an audit; any other
    # key must have its pickle in cache — nothing is downloaded in its place.
    osmnx_graph_key: str | None = None
    # Enables the persistent on-disk cache of the OSMnx graphs across restarts
    # (avoids re-downloading/rebuilding the city+distance graphs at every start)
    osmnx_cache_enabled: bool = True
    # Root folder of the ROUTE cache (SQLite); one subfolder per population is created in it.
    #
    # ⚠ INVARIANT: this container path must be a MOUNT point to `./data/cache/osmnx`
    # (docker-compose, `controller` service) — it is the folder where the bulk populator
    # (step 6 of the `generate_population.ipynb` notebook) writes its warmed routes. Without this
    # mount, `/app` being the bind of `./llm-agents`, the runtime writes into a
    # `services/llm-agents/data/…` invisible to the populator: that is exactly what happened from
    # 2026-06-02 to 2026-09-04, when 196 runs recomputed from cold routes already in cache.
    # The name follows `otp_persistent_cache_dir` below, which never had the defect.
    osmnx_persistent_cache_dir: str = "/app/data/cache/osmnx"

    # number of cached itineraries per grid cell
    n_trip_in_grid: int = 5
    otp_cache_enabled: bool = True
    otp_persistent_cache_dir: str = "/app/data/cache/otp"
    recursion_search_depth: int = (
        0  # 0 means no recursion, 1 means one level of recursion
    )
    search_window_m: int = 30
    transit_access_egress_modes: list[str] = ["foot"]
    max_trip_candidates: int = 6  # maximum number of trip candidates to be selected
    fixed_day: str | None = None

    # Time zone of the simulated NETWORK, to read the wall clock that GAMA publishes
    # (`sim_clock`). Empty (default) = read from the `agency_timezone` of the GTFS feeds in
    # service — the same source OTP uses to interpret its timetables, hence
    # the one that cannot drift from it. To be set explicitly in two cases
    # only: the feeds do not declare the same time zone, or they are not
    # mounted in the service (both then raise an [ALARME] and refuse, rather
    # than guessing a time).
    #
    # ⚠ This is NOT the time zone of the process. `TZ=Europe/Paris` in the
    # `controller` container decided the time at which the agents saw the network:
    # 5 a.m. wall-clock was requested from OTP as 6 a.m. local (235 points without
    # an itinerary instead of 605 on the sealed population v4). A misconfigured
    # container must not move the itineraries.
    network_timezone: str | None = None


class DataConfig(BaseSettings, WorkdirPathResolutionMixin):
    _in_workdir_path_fields: ClassVar[list[str]] = [
        "population_cache_prefix",
        "state_file",
    ]

    # Agent settings
    population_size: int | None = 1
    population_cache_prefix: str = "./population_"
    # Deterministic seed for the random sampling of agents from the eqasim
    # output: guarantees the same subset of agents (hence the same trips) from one run
    # to the next → the OSMnx cache can be reused on replay.
    population_sample_seed: int = 42
    # SEALED population (paper, docs/arch/controle-population-jeu-de-test.md):
    # path of a population file to use as is, instead of the lookup of
    # `{eqasim_output_dir}/{prefix}population_{population_size}.json` and of any call to
    # eqasim. The file is taken WHOLE: if it does not count exactly
    # `population_size` agents (after an optional bbox filter), loading REFUSES rather
    # than resampling — a seal is not trimmed silently.
    population_file: str | None = None
    # ── Recorded trip set (ticket 035, spec 04) ─────────────────────────────────
    # Folder of a set prepared by `python -m experiences preparer-jeu` (data/jeux/<name>).
    # When designated, the simulation SERVES its proposals instead of calling the engines (G3),
    # after checking that the set carries the fingerprint of the loaded population (G1: refused otherwise).
    # Absent: historical behaviour unchanged, computed on the fly (G2). Set by `make run JEU=<name>`.
    jeu_enregistre: str | None = None
    # Time tolerance per mode group (walk, bike, car, transit, rail): "insensible",
    # "heure" (recomputed if the full hour changes) or {pas_min: N}. REQUIRED as soon as a set is
    # designated — no default in the code (E1, question 8); the proposed values are in
    # config/config.yaml, commented.
    jeu_tolerances_horaires: dict[str, Any] | None = None
    # Share of time recomputations that return the already recorded proposals, beyond
    # which the « tolerance too sensitive » ALARM is raised (G7, rising edge).
    jeu_seuil_recalcul_sans_effet: float = 0.3
    state_file: str = "./state.json"
    number_of_llm_based_agents: int | None = 0

    # Eqasim settings
    synthetic_file_prefix: str = "toulouse_"
    eqasim_output_dir: str = "/data/eqasim-output"
    generate_personality_traits: bool = False

    # Debug
    debug_people_ids: list[str] | None = None


class AgentConfig(BaseSettings, WorkdirPathResolutionMixin):
    _in_workdir_path_fields: ClassVar[list[str]] = [
        "long_term_memory_storage_dir",
        "chat_log_dir",
        "journal_memoire_dir",
    ]

    # ── Memory journal, one Markdown per agent (ticket 075) ─────────────────────
    # OFF by default, and not out of token caution: the journal writes the COMPLETE state
    # of the memory after each consolidation. With five agents it is what you come to
    # read; with a thousand, it is hundreds of megabytes that nobody will ever open.
    journal_memoire_enabled: bool = False
    journal_memoire_dir: str = "memoires"

    # ── Trace of the memories served (ticket 077, batch E1) ─────────────────────
    # OFF by default, for the same reason as the journal: one line per decision and per
    # memory served. The concentration MEASURE of batch D2, on the other hand, always runs — it
    # only costs a bounded counter in RAM, and it is the one that carries the alarm.
    trace_rappel_enabled: bool = False

    # ── Trace of the concept operations and per-day measures (ticket 093) ───────
    # OFF by default, like the two above. The trace writes one line per concept
    # operation — a few dozen per day and per agent at most. The measures reread the workdir
    # once per simulated day, at the checkpoint time: with ten agents it is immediate,
    # with a thousand you have to mean it.
    trace_concepts_enabled: bool = False
    mesures_jour_enabled: bool = False

    embedding_model: str | None = None
    chat_log_dir: str = "chat_logs"
    long_term_memory_storage_dir: str = "long_term_memory"
    long_term_memory_filter_by_datetime: bool = False
    long_term_memory_enabled: bool = True  # surchargé par la valeur GAMA
    long_term_max_entries_query: int = 10
    # Age window at recall, IN DAYS. Raised from 30 to 60 in ticket 071 (§ 2.10):
    # nothing must filter by age WITHIN a run, the temporal decay
    # is enough to silence an old memory. At 30 days, a sixty-day run
    # lost its second month at once, without a single log line saying so.
    # Rule: window = horizon of the experiment, capped at `memoire__fenetre_age_max_jours`.
    # `experiences/cli.py` applies it from `horizon_jours`.
    long_term_max_days_query: int = 60
    # Cap of the LRU cache of the LTM metadata. Must stay above the number
    # of agents: below it, every decision causes an eviction (reread +
    # disk rewrite) since the agents are visited round-robin.
    # The metadata weigh ~3 KB/agent, so 2,000 agents ≈ 6 MB in memory.
    long_term_max_loaded_metadata: int = 2000
    long_term_reflect_interval: int = (
        6 * 3600
    )  # 6 hours (legacy — non utilisé si stm_reflection_min_entries > 0)
    stm_reflection_min_entries: int = (
        10  # déclenche la réflexion STM dès que N entrées accumulées
    )
    # Ticket 105 — number of CONSECUTIVE fallbacks beyond which an experiment run stops.
    # A fallback (`plan_index = 0`) is not a missing measurement: the trip happens, the agent
    # remembers it, and the mode taken goes into the habit/break statistic. The threshold is
    # not 1 because an isolated 503 is caught by the retries and produces no fallback: what is
    # caught here is a REGIME, not an incident. Measured on 2026-09-23: 4 fallbacks in 40 minutes
    # brought the measurement window to 10.3% before anyone noticed by hand.
    # Only takes effect if EXPERIMENT_STOP_ON_FALLBACK (or its historical alias) is "1".
    replis_consecutifs_max: int = 3
    # Ticket 106 — number of DISTINCTIVE WORDS of the injected text that must be found in the
    # consolidation that consumes it to consider that the memory has reached long-term memory.
    # Calibrated on 2026-09-23 on the 13 archived runs where a consolidation follows the injection:
    # the runs that keep the shock find 3 to 81 of them, those that lose it 0 and 1. The
    # separation is clear-cut, the threshold is not a fine trade-off.
    temoin_souvenir_mots_min: int = 2
    # FALLBACK deadline (SIMULATED time) of an STM reflection, used only if
    # the agent has no timestamped activity. Since ticket 010, the normal EDF
    # deadline is the agent's WAKE-UP (first planned activity of the next
    # day): the evening decisions go first and the backlog drains
    # during the simulated night. The deadline is kept across retries
    # (gateway failure → resubmission at the next sync, same deadline).
    stm_reflection_deadline_sim_s: int = 12 * 3600
    # Daily consolidation floor (ticket 048). The threshold above is
    # VOLUMETRIC: the moment an agent consolidates is therefore a function of the number
    # of trips it makes that day — an agent with a single trip produces
    # ~6 entries and does not consolidate, an agent with four consolidates twice. The
    # position of a shock relative to the consolidation becomes an uncontrolled
    # variable of the experiment, and since the decision reads ONLY long-term memory,
    # a memory not consolidated does not exist for it. The floor makes the schedule
    # deterministic: at the set time, every agent whose buffer is not empty submits
    # a reflection, at most once per simulated day, whatever its fill level.
    # 10 p.m. leaves the simulated night to the EDF drain, in line with ticket 010.
    stm_reflection_daily_floor_enabled: bool = True
    stm_reflection_daily_floor_hour: int = 22

    # ── The five components of the recall score (ticket 071, batch 2) ────────────
    # ✅ All FIVE are read by `rank_nodes` since batch 2 (2026-09-14), and they sum to
    # 1.00. They switched TOGETHER, and that was the condition: the components enter as
    # ABSOLUTE values since ticket 048 — the min-max normalisation was removed precisely
    # to make two decisions comparable — so a sum other than 1 would run the device
    # under a score regime nobody specified. A test of
    # batch 0 locks the invariant on the weights ACTUALLY read by the ranking.
    #
    # Starting values of ticket 071 § 2.10, TO BE CALIBRATED on frozen sets: they are proposed
    # weights, not a result.
    long_term_retrieval__sim_weight: float = 0.30
    # CATEGORICAL AFFINITY since batch 2. The name is kept so as not to break the
    # existing experiment overrides; what it weighs has changed twice.
    # It weighed a lexical overlap on the labels, called « BLEU-2 » although it
    # is not one — it is an asymmetric recall rate, whereas the BLEU of Papineni et al.
    # (2002) is an n-gram precision with a brevity penalty.
    # It now weighs the match of the WEATHER, and of the weather alone: the three other
    # attributes considered (mode, time slot, purpose) ARE the axes of `affinite_weight`, and
    # counting them twice would make the score uninterpretable (decision of 2026-09-14, issue A).
    long_term_retrieval__keyword_weight: float = 0.10
    long_term_retrieval__time_weight: float = 0.20
    # Severity of the memory — a component of Park et al. (2023) that Vu et al. had left out.
    long_term_retrieval__importance_weight: float = 0.20
    # Axis affinity (object, place, time slot, purpose), as a BONUS and never as a veto.
    long_term_retrieval__affinite_weight: float = 0.20
    long_term_retrieval__default_reflection_importance_score: float = 0.2
    # Time constant of forgetting, IN DAYS: the temporal score of a memory is
    # exp(-Δt / force_base_jours). Replaces the old per-day exponential base
    # (`long_term_retrieval__time_decay = 0.7`), which could not be read: 2.8 days
    # reproduces EXACTLY this decay (0.7^Δt = exp(-Δt/2.8034), half-life
    # 1.94 days). A parameter in days can be discussed — « an ordinary memory lasts
    # three times longer » — whereas an exponential base cannot.
    # This is the parameter the sensitivity arms of the experiments must target:
    # conversion from a declared λ, force = 1 / λ. Ticket 048.
    long_term_retrieval__force_base_jours: float = 2.8

    # ══ Constants of the target memory (ticket 071, § 2.10) ════════════════════
    # Set and PUBLISHED on 2026-09-14. Rule: each constant carries a design rule
    # tied to the PHENOMENON and not to the horizon — the same set holds
    # for five days as for sixty. What makes a value defensible is not
    # a citation but three things: the rule, the date, and a sensitivity point.
    # The sensitivity point of the device is `force_base_jours = 8.3` days, the value
    # published by Park et al. (2023), against 2.8 days inherited from the implementation of Vu et al.
    #
    # ⚠ DELIVERY STATE: this block is batch 0. NONE of these constants is read by
    # code yet — they become so batch by batch. Declared as one block so that the
    # specification and the configuration say the same thing from now on.

    # Lengthening of the lifetime by severity: force = S0 × (1 + k × importance).
    # Rule: a `memorable` memory — « I will remember it in a month » — keeps
    # half of its weight at two weeks and a fifth at thirty days. At k = 3 it
    # fell to 7% in a month, which contradicted the sentence that defines it. Batch 1.
    memoire__force_k_importance: float = 6.0
    # Reinforcement at recall: force ← min(force + δ, FORCE_MAX). ADDITIVE and not
    # multiplicative — a × 1.15 factor saturated in seventeen recalls everything that is
    # recalled often. Rule: each recall adds one day of lifetime; an ordinary
    # trip reaches the cap in twenty-eight recalls, a `memorable` memory in eleven. Batch 1.
    memoire__force_delta_rappel_jours: float = 1.0
    # Lifetime cap, applied FROM WRITING and not only at reinforcement:
    # no lifetime exceeds it, including when S0 triples. Rule: no memory
    # becomes eternal — at the cap, two months without recall bring it down to an eighth,
    # exp(-60/30) = 0.14. Batch 1.
    memoire__force_max_jours: float = 30.0
    # Purge of an EPISODIC entry when its temporal weight falls below this threshold, i.e.
    # 4.6 time constants: 13 days for an ordinary trip never recalled, 90 for a
    # `memorable` memory — so never within a run. Replaces the per-type thresholds of the
    # in-service cleaning. Concepts are NEVER purged, only marked
    # outdated: their dated setting-aside is the observable the experiment is after. Batch 1.
    memoire__purge_seuil_poids: float = 0.01
    # REFERENCE delay of the deterministic severity component, in seconds. It is the
    # anchor point of the calibration: at this delay, the component is exactly `POIDS_RETARD`
    # (0.50). The whole shock catalogue is built around this point.
    # ⚠ Its design rule remains to be written: to be tied to the distribution of the trip
    # durations of the cohort, measurable without a run. The value is the ticket's. Batch 1.
    memoire__retard_ref_s: int = 1800
    # ── What the component becomes BEYOND the reference delay (ticket 095) ──
    #   `asymptote` — DEFAULT since 2026-09-21. The component grows without ever reaching
    #                 its maximum: `max × (1 - exp(-t/τ))`, τ set so that the reference
    #                 delay returns exactly 0.50. **No delay is confused with
    #                 another**, and the component stays bounded.
    #   `palier`    — the former behaviour: sharp cut at the reference delay. Beyond it,
    #                 declaring 45, 60 or 90 minutes gives strictly the same severity, and a
    #                 decreasing profile that stayed above 30 minutes is INVISIBLE to the
    #                 mechanism. Kept to reproduce earlier runs.
    #
    # ⚠ Simply removing the step without changing the shape does not remove the wall, it
    # MOVES it: an unbounded linear component would reach 1.0 at 60 minutes, and 60, 90 and
    # 120 minutes would become indistinguishable again. Only an asymptotic shape removes it.
    memoire__retard_saturation: str = "asymptote"
    # Maximum that the delay component approaches without ever reaching it, in `asymptote` mode.
    # Must stay STRICTLY above POIDS_RETARD (0.50), otherwise the time constant
    # would be undefined. At 0.70, a one-hour delay is worth 0.64 and a delay of an hour and a half
    # 0.68: the gap narrows, it never vanishes.
    memoire__retard_gravite_max: float = 0.70
    # Shock threshold, i.e. the `grave` level and above. Opens pool C of batch 2,
    # which serves these memories WITHOUT any condition of place, time or purpose. Batches 1 and 2.
    memoire__importance_choc: float = 0.7
    # Θ — triggering of the DAYTIME reflection on cumulative severity. Equal to the shock
    # threshold: a single serious memory triggers it, or a day whose cumulative delays are
    # worth one. Base regime = the daily 10 p.m. floor (ticket 048); this one
    # is the EXCEPTION, reserved for breaks. The line A outage of the hysteresis
    # experiments is worth 0.8 in deterministic severity: at Θ = 1.0 the shock studied would
    # not have triggered before the evening.
    # ⚠ TO BE CHECKED BY MEASUREMENT: less than one trigger per agent and per week, otherwise
    # Θ goes up. The measurement needs a GAMA run (ticket 071, rank 2). Batch 1.
    memoire__theta_gravite_cumulee: float = 0.7
    # Below this threshold, a concept stops being SERVED to the model — without being deleted.
    # Exact reading under Laplace smoothing: the concept has been contradicted more often
    # than it has been confirmed, contre_exemples > observations. Batch 3.
    memoire__confiance_seuil_service: float = 0.5

    # ── Sharing within the household (ticket 100, batch 4; design: ticket 078) ──────────
    # FALSE BY DEFAULT, and not as a matter of style: everything measured
    # before this batch stays comparable, and the « no sharing » arm is a switch, not a
    # rebuild.
    memoire__partage_foyer_enabled: bool = False
    # R1, the anchoring: one only tells the household what one has checked oneself. A concept
    # created starts at zero observations — only a day where the agent LIVED the thing again
    # increments it. That is what makes an amplification cycle without contact with the world
    # impossible, and it costs nothing: the rule rereads a counter that already exists.
    # `0` is the « hearsay » arm, for robustness: it turns the finding « hearsay does not
    # circulate » into a measurement.
    memoire__partage_foyer_observations_min: int = 1
    # R5 — a safeguard, not a policy. With R1 and R2 the expected volume is a few statements
    # per night; the bound is there so that a runaway does not go unnoticed in a prompt
    # that ticket 077 measured as already stalling around 2,100 tokens.
    memoire__partage_foyer_max_bloc: int = 12
    # The account is given IN THE EVENING (analysis of 2026-09-25): at the receiver's first
    # consolidation from this simulated time on, or before 3 a.m. (the night belongs to the previous
    # day), and only once per simulated day. Before this setting, the block was served at each
    # consolidation — three per agent and per day at the median — and the paper says « in the
    # evening ». On the treated arm of 2026-09-24_17_50, 11 lived days out of 12 had a
    # consolidation at 6 p.m. or later; the one that has none postpones its account to the next day.
    memoire__recit_soir_heure: int = 18
    # ALERT threshold of the evening account, in pending summaries for one member — not a bound.
    # The account is NEVER truncated (author, 2026-09-29): everything is quoted. A member writes
    # three summaries a day at the median, six after a missed evening; beyond this threshold, a
    # rising-edge [ALARME] flags an upstream defect (markers lost at a resume, receiver that no
    # longer consolidates in the evening). Replaces `memoire__recit_soir_max` and `…_max_par_membre`.
    memoire__recit_soir_alerte_par_membre: int = 8

    # ── D7, 2026-09-22: the severity is the agent's, alone ─────────────────────────────
    # The floor `max(estimated, measured)` is lifted. Nothing corrects an aberrant judgement
    # any more, and that is accepted — but something must SAY it. An [ALARME] is raised when the
    # agent underestimates the measured fact by more than this value.
    #
    # 0.30 = more than one step. The five steps are worth 0.10 / 0.30 / 0.50 / 0.75 / 1.00,
    # hence steps of 0.20 to 0.25: a gap of 0.30 cannot come from a hesitation
    # between two neighbouring levels. Alarming lower would flood the journal, higher would let
    # through the case that matters — « negligible » on a thirty-minute breakdown is worth −0.60.
    memoire__ecart_jugement_alarme: float = 0.30
    # A concept is marked OUTDATED at this number of counter-examples AND confidence < 0.5.
    # Both conditions, not just one: neither three contradictions against twenty
    # confirmations, nor a majority of contradictions on two observations. Batch 3.
    memoire__contre_exemples_seuil: int = 3
    # Pool B of batch 2: memories drawn PER MODE considered, sorted by severity then by
    # recency. No embedding. Independent of the horizon.
    memoire__vivier_b_par_mode: int = 8
    # Pool C of batch 2: serious memories drawn without any context condition. It is the one that
    # makes a morning bike fall weigh on an evening decision. Independent of the horizon.
    memoire__vivier_c_taille: int = 5
    # Cap of the age window at recall, in days. The effective window is
    # the horizon of the experiment, capped here (cf. `long_term_max_days_query`).
    memoire__fenetre_age_max_jours: int = 60
    # EPISODIC entries served next to the core memory (batch 4). A parameter DISTINCT from
    # `long_term_max_entries_query`, which remains the top-K of the recall: reusing the same name for
    # two different things would make any sensitivity measurement ambiguous.
    memoire__episodiques_avec_noyau: int = 3
    # ── Block « what changed recently » (batch 4 of 071, made tunable by 077) ──
    # Age window, in simulated days, of a memory of shock severity served in this block.
    # SHARP CUT: beyond it, it is no longer served at all — this is not a decay.
    #
    # ⚠ This is the parameter that governs the duration of a shock effect, and it was hard-coded.
    # On `experiments/archive/2026-09-19_07_31`, the car disappears the day after the shock and
    # comes back FOURTEEN DAYS later, to the day, at the first prompt that no longer carries the
    # account: P(car) = 6.9% while it is there (n = 18), 55.2% once it is gone (n = 33). The
    # run report attributed this return to the decay of the memory, whose computed lifetime
    # was ~14.6 days: the two explanations predict the same date, and nothing
    # allowed telling them apart. Declared here, the parameter becomes ablatable.
    #
    # 0 = no shock episode in the block. It is the value of the ablation arm, not an
    # accidental switch-off. A negative value is brought back to 0 and logged.
    memoire__fenetre_changements_jours: int = 14
    # Maximum number of lines of the block. Beyond it, it costs tokens without teaching anything.
    memoire__changements_max: int = 3
    # ── The duration of a shock memory, DERIVED from its severity (ticket 095, batch A) ──
    # Serving mode of the block « what changed recently »:
    #   `derivee` — DEFAULT. A memory is served as long as its decay weight
    #               `exp(-Δt / force)` exceeds `memoire__seuil_service_changement`, i.e. a
    #               duration `force × ln(1/threshold)` bounded by the floor and the cap below.
    #               The duration becomes a CONSEQUENCE of the event, and not a set integer.
    #   `fixe`     — the behaviour before ticket 095: sharp cut at
    #               `memoire__fenetre_changements_jours`, whatever the severity. Kept
    #               for the methodological control arms, which must be able to reproduce
    #               the campaigns of 19 and 20 September 2026.
    #
    # ⚠ The default CHANGES the behaviour of any run that declares nothing (author's decision,
    # 2026-09-21). An archived run compares with a new run only by declaring `fixe`.
    memoire__mode_fenetre_changements: str = "derivee"
    # Weight threshold below which a shock memory leaves the block. Rule: at 0.35, a memory
    # has lost two thirds of its weight — it still weighs, but it no longer dominates. Open
    # domain ]0, 1[: at 1 the duration would be zero (silent ablation), at 0 it would be infinite.
    memoire__seuil_service_changement: float = 0.35
    # Floor of the served duration, in days. A safety bound: at S0 = 2.8 days and threshold 0.35,
    # the duration of a memory of ZERO severity is already 2.94 days, so this floor does not bite.
    # It is not an observed minimum duration, and must not be quoted as such.
    memoire__plancher_changement_jours: float = 2.0
    # Cap of the served duration, in days. Raised from 30 to 50 on 2026-09-22, by the author's
    # decision: **it never bites any more, and that is the point.**
    #
    # What really bounds the duration is `memoire__force_max_jours`, because
    # `duration = force × ln(1/threshold) = force × 1.0498`. With force capped at 30, the duration
    # cannot exceed 31.49 days, whatever the number of recalls. The duration cap could
    # therefore only bite in the band `force ∈ ]28.58; 30]`, where it trimmed AT MOST 1.49 days —
    # after 9 recalls for a `memorable` memory, 25 for a `negligible` one.
    #
    # ⚠ The previous comment claimed that without this cap « a memory recalled often
    # would push back its own deadline indefinitely ». That was FALSE: `force_apres_rappel`
    # saturates at `memoire__force_max_jours`, and the protection against the archive comes from there.
    #
    # Why keep it rather than remove it: a declared bound can be read, is logged
    # when it bites (`DureeService.borne`), and documents the intent. A bound that never bites
    # is a witness — if a log line names it one day, the law or the force
    # cap has moved without anyone noticing.
    #
    # ⚠ DO NOT QUOTE « between 2 and 30 days » as observed durations: both bounds are
    # safety guards that nothing has ever reached (floor 2 days against 2.94 days at zero
    # severity; cap out of reach by construction).
    memoire__plafond_changement_jours: float = 50.0
    # ════════════════════════════════════════════════════════════════════════════

    long_term_self_reflect_enabled: bool = True  # surchargé par la valeur GAMA
    long_term_self_reflect_interval_days: int = 3
    long_term_self_reflect_window_days: int = 5

    # The LLM no longer chooses an itinerary: it gives a probability to each
    # option, and the actual mode is drawn from this distribution (including
    # at each reread of the semantic cache). The seed of the draw is derived from
    # (this value, agent, activity, simulated day): with the same seed, a replayed run
    # reproduces exactly the same trips; changing it explores another draw
    # without calling the LLM again.
    mode_draw_seed: int = 42

    # Order in which the options are presented to the decision-maker (ticket 035, spec 02 D7 / EF-23):
    # DETERMINISTIC shuffle derived from (this seed, agent, activity), identical in the
    # two execution modes — and no longer an unseeded `random.shuffle`. The final draw does
    # not depend on this order (the weights are realigned on the canonical order by code).
    # Named as in config/experience_plan/experiments.yaml (`option_order_seed`).
    option_order_seed: int = 42

    # ── One weather date per agent (ticket 023, continued) ─────────────────────
    # Over a single simulated day, all agents share a single weather:
    # the regressor has zero variance, and « no measured effect » then means
    # nothing. When enabled, each agent reads the bulletin of a day of the year drawn
    # deterministically from its identifier — only the DATE of the bulletin changes,
    # the departure time is kept, and the transport offer remains that of the
    # simulated day. Ceteris paribus device: the seed of the mode draw
    # is not touched.
    # DISABLED BY DEFAULT: nothing moves without explicit intent.
    weather_per_agent_dates: bool = False
    # "enquete" reads the EMC² collection window and its surveyed days from
    # mobility_core.population_reference (no copied bounds); "annee"
    # takes the 365 days; otherwise a pair ["YYYY-MM-DD", "YYYY-MM-DD"].
    weather_window: Any = "enquete"
    # The survey only covers working days; no effect if the window is
    # "annee" and this flag is false.
    weather_weekdays_only: bool = True
    weather_draw_seed: int = 42
    # Table `person_id → "YYYY-MM-DD"` of the day ACTUALLY described, when it is known — the case
    # of respondent populations, where no date needs drawing since the survey carries it
    # (ticket 058). The runner fills it in by convention: a `dates_meteo.json` placed next to
    # the population. Absent, the seeded draw resumes, unchanged.
    weather_dates_file: str | None = None

    llm_params: dict[str, Any] = {
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 4096,
        # `thinking_budget` is NOT listed here: absent from the parameters, the gateway leaves
        # the provider its default reasoning. An experiment that wants to control it sets it
        # in `decideur.parametres`, which seals it into its fingerprint.
    }
    llm_retry_count: int = 3
    llm_retry_delay: int = 5  # seconds

    # Scheduler settings
    reschedule_activity__version: int = 2
    reschedule_activity_departure_time: bool = True
    reschedule_transition_ratio: float = 0.75
    reschedule_activity_v2__k: float = 0.02
    max_reschedule_amount: int = 3600  # 1 hour
    pre_schedule_duration: int = 0

    # No trip starts at the weekend: a departure falling on a Saturday or a
    # Sunday is postponed to the following Monday at the same time.
    no_weekend_departures: bool = True

    # --- Anticipation of the day's chain (ticket 014) ---
    # The persona block of the mode choice prompt is enriched with three elements:
    # the weather of the remaining slots of the day (all agents), the rolling
    # agenda of the remaining trips and the position of the personal vehicles
    # (only the agents that have something to chain: drivers with a
    # car, bike owners — never passengers). The choice stays trip
    # by trip; the chain locks remain the safety net.
    # False restores the short-sighted prompt, for the A/B against a reference run.
    agenda_anticipation_enabled: bool = True

    # --- Chain consistency of personal vehicles (bike, car) ---
    # A vehicle is a place: it stays parked where the agent left it, is offered as a
    # mode only from this position, and is brought back home at the end of the loop.
    # False restores the historical behaviour (ownership = available everywhere),
    # useful to measure the effect of the fix on the modal shares at equal population.
    vehicle_chain_enabled: bool = True
    # Return lock: a trip home from a place where a vehicle is
    # parked is restricted to the itineraries of that mode (the agent brings back its bike / car).
    vehicle_return_home_lock: bool = True
    # Recovery of vehicles left at an intermediate stop: brought back home
    # when the agent returns there. Without it, an agent loses its car for all following days.
    vehicle_orphan_reset_at_home: bool = True
    # [ALARME] alarm if the share of returns home leaving a vehicle orphaned
    # exceeds this ratio (over at least N observed returns, to avoid start-up noise).
    vehicle_orphan_alarm_ratio: float = 0.05
    vehicle_orphan_alarm_min_returns: int = 200

    # --- Truncation threshold of the Consideration Set (ticket 077, 30-day replay) ---
    # The options whose relative share is strictly below this threshold
    # are removed (weight set to zero) before the categorical draw. The remaining
    # mass is renormalised. Value 0.0 = no truncation (historical
    # behaviour). Recommended value for an individual agent: 0.15 (15%).
    # Reference: Consideration Set (Hauser & Wernerfelt 1990, Ben-Akiva &
    # Boccara 1995) — a traveller does not integrate marginal alternatives.
    mode_choice_truncation_threshold: float = 0.0

    quantify_time_window: bool = True
    reflection_custom_guidelines: str | None = None

    # Remote LLM settings
    # 120 s gives the worker time to absorb a 5xx cooldown (120 s) + backoff and to
    # fail over to another provider BEFORE the client gives up (otherwise fallback).
    remote_llm_poll_timeout: float = 120.0  # timeout (secondes) d'une tâche LLM
    stm_reflection_min_tpm: int | None = (
        30000  # excludes the providers below this TPM threshold for the STM reflection
    )
    # Backpressure: when the SDK client raises the alarm (N consecutive failures), it
    # blocks new submissions until the in-flight stack falls back
    # below this ratio of worker_concurrency (0.2 = 20%). 0 disables the backpressure.
    remote_llm_backpressure_ratio: float = 0.2
    # Circuit breaker of the gateway client: after N consecutive failures (token shortage,
    # gateway/network down), LLM submissions are SUSPENDED: the simulation waits
    # quietly for recovery (quota renewal, service back)
    # instead of burning attempts doomed to fail (120 s of timeout each) and
    # degrading the decisions to default indexes. No decision is taken outside
    # the nominal path (exact cache or LLM) during the wait. A probe retests the
    # gateway periodically; the first success closes the breaker and everything restarts
    # automatically. 0 disables the breaker (historical behaviour: each
    # decision fails after its timeout and falls back to the default index).
    remote_llm_circuit_failure_threshold: int = 10
    remote_llm_circuit_probe_interval: float = 60.0  # secondes entre deux sondes

    def fenetre_age_pour_horizon(self, horizon_jours: int) -> int:
        """Recall age window, in days, for an experiment of a given horizon.

        Ticket 071 § 2.10: the window is the HORIZON of the experiment, capped by
        `memoire__fenetre_age_max_jours`. Nothing must filter by age within
        a run — the temporal decay is enough to silence an old memory, and
        a sharp cut at 30 days took its second month away from a sixty-day run without
        a single log line saying so.

        A zero or negative horizon makes no sense here: it would make the window empty,
        that is a mute memory that no symptom would distinguish from a memory
        cut off. It is brought back to one day.
        """
        return max(1, min(int(horizon_jours), int(self.memoire__fenetre_age_max_jours)))


class CacheConfig(BaseSettings, WorkdirPathResolutionMixin):
    _in_workdir_path_fields: ClassVar[list[str]] = []

    enabled: bool = True
    cache_dir: str = "/app/data/llm_cache"
    semantic_threshold: float = 0.95
    embed_model_name: str = "all-MiniLM-L6-v2"
    # Exact memoisation of the STM/LTM reflections (ticket 012): serves the reflection
    # already paid for when the effective prompt is byte-identical (deterministic re-runs).
    # Exact match only — never any cross-agent matching.
    reflection_memo_enabled: bool = True


class AccidentsConfig(BaseSettings, WorkdirPathResolutionMixin):
    """Accidents drawn at random on the roads (ticket 070, first slice).

    This slice ONLY produces the existence of the accidents: they are drawn, placed on an
    edge of the road graph and logged. **No itinerary duration is changed** —
    the delay suffered, the cache guards and the memory come in a later slice.
    That is intended: an accident that slows nobody down cannot poison a cache
    addressed without the date, nor make a decision taken without the delay be served again.

    THE DRAWING LAW IS MEASURED, and it does not live here: the coefficients estimated on
    BAAC/ONISR 2019-2024 are in ``config/accidents_baac.yaml``, with the reason for each
    choice. This block only carries the execution settings. One variable of the law remains
    NOT ESTABLISHED, the weather factor, and the coefficients file says why its computation
    was rejected rather than adjusted.
    """

    _in_workdir_path_fields: ClassVar[list[str]] = []

    # Overridden by the GAMA value at /init. TRUE BY DEFAULT since 2026-09-15: the realistic
    # regime becomes the ordinary one, and it is its absence that must be asked for. Accepted
    # consequence — an `/init` that does not carry the key (GAMA before ticket 070) enables accidents.
    # ⚠ Harmless as long as no duration is changed; to be re-examined the day the delay suffered
    # is wired in, since every run will then undergo its effect without having asked for it.
    enabled: bool = True

    # OVERRIDE of the base rate, in accidents per simulated day. `None` — the default — reads
    # the rate measured in `config/accidents_baac.yaml` (1.56/day within the extent of the
    # graph). Fill in this field only to make the mechanism observable in
    # development: a value here REPLACES the measurement, and any run that carries it stops
    # being representative. The weekday and weather factors apply on top.
    taux_journalier: float | None = None

    # Multiplicative factor applied to the travel time of an edge while an accident
    # is active on it. ⚠ DECLARED ASSUMPTION, NOT A MEASUREMENT: no source gives the size of the
    # slowdown today — the DATEX feed of the DIRs, the only one to carry real durations,
    # is not archived yet. The value is of the same order as what a partial blockage
    # produces on the scale of the TomTom table (whose congestion factors cap
    # around 1.8), with no more justification than that. Any effect measured on the agents will
    # be the effect of THIS number: it is published with the result, never silently.
    facteur_ralentissement: float = 3.0

    # Duration of an accident, in minutes. NO SOURCE gives it today: the DATEX feed
    # of the DIRs, the only source of real durations, is not archived yet. These bounds are
    # DECLARED ASSUMPTIONS, and for now they only serve to give the event an end —
    # nothing depends on their accuracy as long as the delay is not wired in.
    duree_min_minutes: int = 20
    duree_max_minutes: int = 90

    # Seed of the draw. Fixed so that a replayed run draws the same accidents: without it,
    # two runs of the same scenario would no longer be comparable, and the gap would be put
    # down to the agents.
    graine: int = 70

    # Safeguard: beyond this, drawing is refused rather than filling the state of the world. A rate
    # typed as 1730 instead of 1.73 is a typo, not an intention.
    taux_journalier_max: float = 500.0


class ChocsConfig(BaseSettings, WorkdirPathResolutionMixin):
    """Declared shock, suffered by the agents (ticket 079).

    A shock is a QUANTIFIED DELAY plus a LIVED SENTENCE, placed on designated agents on
    designated days. It cuts no line and degrades no offer: it is the **suffered
    regime**, where the agent decides seeing the nominal offer and then takes the hit. The day
    of the shock therefore measures no choice, and the whole effect of the following days can be
    put down to the memory — which is exactly what Stage 3a seeks to establish.

    The declaration lives in its own file, not here: `config/chocs/*.yaml`. This block only
    carries what designates it, just as `AccidentsConfig` does not carry the BAAC law.
    """

    _in_workdir_path_fields: ClassVar[list[str]] = []

    # False by default: a run that asks for nothing behaves exactly as before this ticket.
    enabled: bool = False

    # Path of the declaration. `None` — the default — means « no shock ». A designated and
    # invalid file makes loading FAIL: a clear refusal at startup is better than a
    # sixty-day run that does nothing and nobody knows why.
    fichier: str | None = None


class EvenementsConfig(BaseSettings, WorkdirPathResolutionMixin):
    """Declared event, lived or read (ticket 100).

    An event is a TEXT placed in the memory of designated agents on designated days,
    with at most one MEASURED FACT. Two entry points: `arrivee`, after the decision — the agent
    chose seeing the nominal offer then takes the hit, the suffered regime of ticket 079; and
    `reveil`, before the first decision — the agent knows before choosing, the article of ticket 059.

    This block replaces `ChocsConfig`, which is still read for one more version. The declaration
    lives in its own file, not here: `config/evenements/*.yaml`, just as `AccidentsConfig` does
    not carry the BAAC law.
    """

    _in_workdir_path_fields: ClassVar[list[str]] = []

    # False by default: a run that asks for nothing behaves exactly as before.
    enabled: bool = False

    # Path of the declaration. `None` — the default — means « no event ». A designated
    # and invalid file makes loading FAIL: a clear refusal at startup is better
    # than a sixty-day run that does nothing and nobody knows why.
    fichier: str | None = None


class AppConfig(BaseSettings, WorkdirPathResolutionMixin):
    _in_workdir_path_fields: ClassVar[list[str]] = [
        "agent_memory_events_jsonl",
        "agent_memory_events_csv",
        "log_file",
        "llm_exchanges_file",
        "llm_cache_hits_file",
        "trace_rappel_file",
        "trace_concepts_file",
        "temoin_souvenir_file",
        "mesures_dir",
        "pipeline_log_file",
    ]

    # Agent memory events log (STM + LTM observations, reflections, concepts)
    agent_memory_events_jsonl: str = "agent_memory_events.jsonl"
    agent_memory_events_csv: str = "agent_memory_events.csv"

    # LLM exchange log (service, prompt, response, tokens)
    llm_exchanges_file: str = "llm_exchanges.jsonl"

    # LLM cache hit log (decisions served from the semantic cache → no LLM call,
    # hence absent from llm_exchanges.jsonl; needed to measure the token savings)
    llm_cache_hits_file: str = "llm_cache_hits.jsonl"
    trace_rappel_file: str = "trace_rappel.jsonl"
    # Concept operations (ticket 093): one line per created / confirmed / refined / contradicted.
    trace_concepts_file: str = "operations_concept.jsonl"
    # Witness of the injected memory (ticket 106): one line per consolidation that follows an
    # injection, whether it found the memory or not. Both verdicts are traced —
    # a false alarm rate cannot be counted on the failures alone.
    temoin_souvenir_file: str = "temoin_souvenir.jsonl"
    # Folder of the per-simulated-day measurement CSVs. Created at the first write, never at
    # import.
    mesures_dir: str = "mesures"

    # Application log
    log_file: str = "app.log"
    log_level: str = "INFO"

    # Pipeline timing CSV log (T0 → Fin, LLM agents only)
    pipeline_log_enabled: bool = True
    pipeline_log_file: str = "pipeline_timing.csv"


class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    server: ServerConfig = ServerConfig()
    data: DataConfig = DataConfig()
    world: WorldConfig = WorldConfig()
    gtfs: GTFSConfig = GTFSConfig()
    agent: AgentConfig = AgentConfig()
    llm: LlmConfig = LlmConfig()
    cache: CacheConfig = CacheConfig()
    accidents: AccidentsConfig = AccidentsConfig()
    chocs: ChocsConfig = ChocsConfig()
    # Ticket 100 — the single channel. `chocs` is still read for one more version, with a
    # warning: a campaign launched under the old key must keep running.
    evenements: EvenementsConfig = EvenementsConfig()

    # Directory settings
    workdir: Path = Path.cwd()

    # @field_validator('workdir', mode='before')
    # @classmethod
    # def resolve_workdir(cls, v):
    #     """Ensure workdir is an absolute Path."""
    #     return Path(v).resolve()

    @model_validator(mode="after")
    def resolve_all_paths(self):
        """Resolves all relative paths of the sub-configs against workdir after Pydantic validation."""
        # This will only be triggered if you instantiate Settings via pydantic's validation process,
        # e.g., Settings(**data), not when you subclass or access attributes directly.
        # If you use Settings.from_yaml_files or FactorySettings, it will be triggered.
        # If you instantiate Settings without validation, it won't.
        for field_value in self.__dict__.values():
            if isinstance(field_value, WorkdirPathResolutionMixin):
                field_value.resolve_paths(self.workdir)
        return self

    def _resolve_nested_paths(self, model_instance: BaseModel, path_fields: list):
        """Resolve paths in a nested model instance."""
        for path_field in path_fields:
            if hasattr(model_instance, path_field):
                current_value = getattr(model_instance, path_field)
                if current_value is not None and not Path(current_value).is_absolute():
                    resolved_path = self.workdir / current_value
                    setattr(model_instance, path_field, str(resolved_path))

    @classmethod
    def from_yaml_files(
        cls, *yaml_paths: str, workdir: str | None = None
    ) -> "Settings":
        """Load and merge multiple YAML files."""
        merged_data = merge_configs(*yaml_paths)
        # Remove workdir from YAML data — it is now derived from the config file name
        merged_data.pop("workdir", None)
        if workdir:
            merged_data["workdir"] = Path(workdir).resolve()
        return cls(**merged_data)


class FactorySettings:
    _instance: Settings | None = None
    _creation_time: datetime | None = None

    @classmethod
    def get(cls) -> Settings:
        """Returns the Settings singleton, creating it from the YAML files on the first call."""
        if cls._instance is not None:
            return cls._instance

        # A single run configuration, always loaded from this fixed path —
        # no more selection through APP_CONFIG_PATH/CONFIG=...: to change the config,
        # edit services/llm-agents/config/config.yaml directly.
        base_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config/config.yaml",
        )
        yaml_files = [base_config_path]

        # The experiment workdir (experiments/archive/<YYYY-MM-DD>_<HH_MM>) is
        # created and archived unconditionally at each start.
        now = datetime.now()  # noqa: DTZ005 — local time INTENDED: names the experiment folder, read by a human
        cls._creation_time = now
        experiments_dir = _resolve_experiments_dir(
            Path(base_config_path).resolve().parent.parent
        )
        # Hot resume (`make run … REPRISE=<name>` → REPRISE_RUN): the workdir of the
        # NAMED run is reused instead of creating a new one. The logs are APPENDED to it (moves.csv
        # keeps its header, app.log continues), state.json and the population checkpoints are
        # found there through the _in_workdir_path_fields paths. The GAMA simulation, for its part,
        # restarts at t0 of its calendar and replays the days already lived (no state freeze on the
        # GAMA side, cf. ticket 002); the memory is frozen during this replay (ticket 075) and the
        # decisions are served again from the run's trace (ticket 090) instead of being paid again.
        _resume = os.environ.get("CONTINUE_RUN", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        # Ticket 091 — a resume is NAMED. The `experiments/current` link is not authoritative: it
        # pointed twice on 2026-09-16 to a run other than the one we thought we were resuming, and
        # following a moving link to find a memory amounts to reusing at random.
        _nomme = os.environ.get("REPRISE_RUN", "").strip()
        if _nomme:
            _repris = experiments_dir / "archive" / _nomme
            if not _repris.is_dir():
                raise RuntimeError(
                    f"REPRISE={_nomme}: no run of this name under {experiments_dir / 'archive'} "
                    f"— a resume is named, and the name must exist."
                )
            workdir = str(_repris)
        elif _resume:
            raise RuntimeError(
                "CONTINUE_RUN without REPRISE: a resume is named. Relaunch with "
                "`make run … REPRISE=<run name>` — otherwise nothing guarantees that the memory "
                "recovered is that of this experiment (ticket 091)."
            )
        else:
            exp_name = f"{now.strftime('%Y-%m-%d')}_{now.strftime('%H_%M')}"
            workdir = str(experiments_dir / "archive" / exp_name)

        cls._instance = Settings.from_yaml_files(*yaml_files, workdir=workdir)
        # WorldConfig receives a YAML table: the WORLD__* fields are not injected
        # automatically by BaseSettings into this sub-model. Read this opt-in flag
        # explicitly, otherwise the manifest would say « lockstep » without enabling it.
        prefixe_env = os.environ.get("WORLD__PREFIXE_COMMUN")
        if prefixe_env is not None:
            valeur = prefixe_env.strip().lower()
            if valeur not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
                raise ValueError(f"WORLD__PREFIXE_COMMUN invalide : {prefixe_env!r}")
            cls._instance.world.prefixe_commun = valeur in ("1", "true", "yes", "on")
        concurrence_env = os.environ.get("WORLD__WORKER_CONCURRENCY")
        if concurrence_env is not None:
            concurrence = int(concurrence_env)
            if concurrence <= 0:
                raise ValueError("WORLD__WORKER_CONCURRENCY must be positive")
            cls._instance.world.worker_concurrency = concurrence
        if cls._instance.world.prefixe_commun and not (
            cls._instance.gtfs.otp_cache_enabled and cls._instance.gtfs.osmnx_cache_enabled
        ):
            raise RuntimeError(
                "Common prefix: OTP and OSMnx caches required to replay the itineraries"
            )

        if _run_artifacts_disabled():
            logger.info(
                "Import sous test : aucun répertoire de run créé, symlinks "
                "experiments/current et services/GAMA/CityTransport/results laissés en place."
            )
            return cls._instance

        # An import READS the configuration, it does not open a run: no folder created, no
        # configuration frozen on disk, no symlink moved. All of that belongs to
        # `claim_run()`, which only the owning process calls.
        logger.debug(
            f"Configuration lue ; workdir de ce processus s'il ouvrait un run : "
            f"{cls._instance.workdir.name} (aucun artefact créé, cf. claim_run())"
        )

        # logger.info(f"Settings loaded from: {yaml_files}")
        # logger.info(f"All settings: {cls._instance.model_dump_json(indent=2)}")
        return cls._instance

    @classmethod
    def save_static_config(cls) -> None:
        """
        Saves the current state of the configuration to the static_config.yaml file.
        To be called after the GAMA simulation has overridden the parameters.
        """
        # The folder is created here rather than required: since 2026-09-04, importing this
        # module no longer creates anything (cf. `claim_run`), so requiring it to exist would
        # silence this method instead of making it work.
        if cls._instance and cls._instance.workdir:
            cls._instance.workdir.mkdir(parents=True, exist_ok=True)
            static_config_path = cls._instance.workdir / "static_config.yaml"
            with open(static_config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    cls._instance.model_dump(mode="json"),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            logger.info(f"Configuration statique mise à jour : {static_config_path}")

    def __getattribute__(self, name):
        """Delegates all attribute accesses to the underlying Settings singleton."""
        # Handle special methods and private attributes directly
        if name.startswith("_") or name in (
            "get",
            "force_reload",
            "force_reload_paths",
            "save_static_config",
            "claim_run",
        ):
            return super().__getattribute__(name)

        # Delegate all other attributes to the Settings instance
        return getattr(self.get(), name)

    @classmethod
    def claim_run(cls) -> Settings:
        """Declares that THIS process opens the run: moves the symlinks to its workdir.

        **To be called only by the process that owns the run** — the controller, through
        `handle/__init__.py`. Everything else (analysis script, configuration check,
        shell in a container, notebook, routing worker) imports this module to READ the
        configuration, and must not touch the links.

        Why this is an explicit call and not a side effect of the import: until 2026-09-04,
        `get()` repointed `experiments/current` at import. An analysis script launched
        during a run therefore switched the link to an empty folder, and everything that
        resolves the link at EACH write — the log of exchanges with the model — started
        writing next to the run. Seen four times between the evening of 2026-09-03 and the
        morning of 2026-09-04, the first one diverting 1,037 exchanges in a minute. The
        run folder, for its part, is still prepared at import: creating it steals nothing from
        anyone, moving the link away from it does.

        Idempotent: calling this method again on the same workdir rewrites the same links.
        """
        cls.get()
        # The workdir is `<experiments>/archive/<name>`: going up two levels gives back the
        # experiments folder exactly as `get()` resolved it, without redoing the
        # detection (which depends on `base_config_path`, local to `get()`).
        experiments_dir = cls._instance.workdir.parent.parent
        if _run_artifacts_disabled():
            logger.info(
                "claim_run() sous test : ni répertoire de run créé, ni symlinks "
                "experiments/current et services/GAMA/CityTransport/results déplacés."
            )
            return cls._instance

        cls._instance.workdir.mkdir(parents=True, exist_ok=True)
        cls.save_static_config()
        current_link = experiments_dir / "current"
        # Several processes import this module in the same second (routing workers of the
        # notebook, hypercorn workers): unlink then symlink is not atomic, and the second
        # to arrive hit FileExistsError — three workers spawned together, two dead at
        # initialisation, BrokenProcessPool (2026-09-03). A temporary link specific to this
        # process, then os.replace: atomic, and the last writer wins.
        _tmp_link = experiments_dir / f".current.{os.getpid()}"
        _tmp_link.unlink(missing_ok=True)
        _tmp_link.symlink_to(Path("archive") / cls._instance.workdir.name)
        os.replace(_tmp_link, current_link)

        # Redirect GAMA results into this experiment's workdir.
        gama_results_dir = cls._instance.workdir / "gama_results"
        gama_results_dir.mkdir(parents=True, exist_ok=True)
        gama_results_link = (
            _racine_partagee("services", "GAMA", "CityTransport")
            / "services"
            / "GAMA"
            / "CityTransport"
            / "results"
        )
        if gama_results_link.parent.exists():
            # The link is written here but READ elsewhere — by GAMA on the host, or by the
            # `gama` container (which mounts ./services/GAMA on /services/GAMA and
            # ./experiments on /experiments). Its target must therefore be expressed in the
            # layout of the repository, not in that of the controller: from
            # services/GAMA/CityTransport, THREE levels up give the root,
            # then experiments/. It is so that this count is the same on both sides that
            # the mount carries the `services/` prefix (ticket 039). A computation by
            # `relpath` from the controller's workdir (/app/experiments/…) would give
            # a link correct in this container only, and dangling everywhere else.
            within_experiments = os.path.relpath(gama_results_dir, experiments_dir)
            relative_target = Path("../../../experiments") / within_experiments
            # ⚠ Writing the link is CONDITIONAL since ticket 075 (see below): it
            # was done here, before any check, and a misplaced process
            # took its output away from the running simulation. Several hypercorn workers
            # importing this module in parallel, unlink/symlink still tolerate another
            # worker having passed first.
            # The link does not resolve from this process (the controller does not see
            # /experiments): what can be checked is the invariant that
            # had broken it — the workdir must live under a folder named
            # `experiments` at the root of the repository. Otherwise the link dangles, and GAMA fails
            # on `save` with an I/O error that does not name the cause.
            # ⚠ The folder name is NOT enough to validate the invariant, and that is what
            # broke the run of 2026-09-14 (ticket 075): a process launched from
            # `services/llm-agents/` resolves its `experiments_dir` to
            # `services/llm-agents/experiments`, whose NAME is indeed « experiments » and whose
            # relative path starts with no `..`. The link was therefore written, it
            # pointed to a folder that only exists under `services/llm-agents/`, and
            # GAMA — which reads the link from the root — hit an I/O error in the middle
            # of the run, with nothing naming the cause. What must be checked is that the
            # experiments folder is indeed the one of THE ROOT, the same one where
            # `services/GAMA/CityTransport` lives.
            #
            # The check can only be made where it MEANS something. In the
            # `controller` container, `services/GAMA` is mounted on `/services/GAMA` and the
            # experiments on `/app/experiments`: the root seen from here is `/`, where `experiments`
            # does not exist, and no local comparison can validate anything — the link
            # is then written on the strength of the mount layout, as before. On
            # the host, on the other hand, `<root>/experiments` EXISTS: if the experiments
            # folder of this process is not that one, the link it would write
            # would dangle, and it is refused.
            _racine = _racine_partagee("services", "GAMA", "CityTransport")
            _experiences_de_la_racine = _racine / "experiments"
            _incoherent = (
                _experiences_de_la_racine.is_dir()
                and experiments_dir.resolve() != _experiences_de_la_racine.resolve()
            )
            if (
                not _incoherent
                and experiments_dir.name == "experiments"
                and not within_experiments.startswith("..")
            ):
                if gama_results_link.is_symlink():
                    gama_results_link.unlink(missing_ok=True)
                elif gama_results_link.exists():
                    gama_results_link.rename(
                        gama_results_link.parent / "results_legacy"
                    )
                try:
                    gama_results_link.symlink_to(relative_target)
                except FileExistsError:
                    pass
                logger.info(
                    f"Sorties GAMA redirigées : {gama_results_link} → {relative_target}"
                )
            else:
                # REFUSAL to write, and no longer an alarm after the fact: the link in place may
                # belong to a running simulation, and replacing it with a dangling link would
                # make it fail on `save`. A process that does not own the run
                # must not be able to take its output away from it.
                logger.error(
                    f"[ALARME] Redirection des sorties GAMA REFUSÉE : le répertoire "
                    f"d'expériences de ce processus ({experiments_dir}) n'est pas celui de la "
                    f"racine ({_experiences_de_la_racine}). Le lien {gama_results_link} est "
                    f"laissé TEL QUEL — il appartient peut-être à une simulation en cours, et "
                    f"le réécrire la ferait échouer sur `save`. Posez APP_EXPERIMENTS_DIR si "
                    f"ce processus doit vraiment ouvrir un run."
                )

        logger.info(
            f"Run ouvert par ce processus : experiments/current → "
            f"archive/{cls._instance.workdir.name}"
        )
        return cls._instance

    @classmethod
    def force_reload(cls) -> Settings:
        """Force reload the settings."""
        cls._instance = None
        return cls.get()

    @classmethod
    def force_reload_paths(cls) -> Settings:
        """Forces a full reload of the settings (alias of force_reload)."""
        cls._instance = None
        return cls.get()


settings = FactorySettings()
