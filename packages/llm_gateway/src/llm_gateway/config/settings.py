"""
config/settings.py — the gateway settings, layered and grouped.

The configuration is NEVER built as a side effect of an import: the entry points
(create_app, Celery worker, CLI, tests) explicitly call `GatewaySettings()` or
`get_settings()`. Building the settings reads the environment and the providers file;
no Redis or network access.

Sources, from strongest to weakest (see `config/sources.py`): constructor
arguments, `LLM_GATEWAY_*` environment (`__` separates the levels:
`LLM_GATEWAY_BATCHING__DELAY_SECONDS`), old unprefixed names (deprecated), YAML file
`LLM_GATEWAY_CONFIG`, profile `LLM_GATEWAY_PROFILE`, defaults below.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from llm_gateway.config.providers import (
    DEFAULT_EXAMPLE_FILE,
    InferenceOverrides,
    ProviderEntry,
    ProvidersFile,
    load_providers_file,
)
from llm_gateway.config.sources import ENV_PREFIX, LegacyEnvSource, yaml_sources
from llm_gateway.core.batching import compute_batch_max_agents
from llm_gateway.core.quota import DEFAUT_FUSEAU_QUOTA
from llm_gateway.telemetry.logger import get_logger

logger = get_logger(__name__)

# Instances named <model>_key<N>: their key is dedicated (no adapter fallback).
_INSTANCE_KEY_SUFFIX = re.compile(r"_key\d+$")


# ---------------------------------------------------------------------------
# A resolved provider: the file entry + the key + the computed values
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """Effective configuration of a provider instance, as the gateway uses it."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr = SecretStr("")
    rpm_limit: int
    tpm_limit: int | None = None
    rpd_limit: int | None = None
    tpd_limit: int | None = None
    quota_reset_tz: str = DEFAUT_FUSEAU_QUOTA   # time zone of the daily reset (see core.quota)
    max_tokens_per_request: int | None = None
    max_output_tokens: int | None = None
    # Thinking cap of the served model (see `ProviderEntry.thinking_budget_max` for the
    # reason). Read by the Google adapter, which refuses a budget above it.
    thinking_budget_max: int | None = None
    # Thinking levels accepted by the served model (`minimal`, `low`, `medium`, `high`).
    # Read from the provider's documentation: they vary from one model to another, and
    # requesting a missing level returns 400. Declared → the form only offers those and an
    # unknown level is refused upfront; absent → no level is offered.
    thinking_levels: list[str] | None = None
    base_url: str
    default_model: str
    weight: float = 1.0
    batch_max_agents: int = 1            # computed: min(tpm/tokens_per_agent, request capacity, rpm, cap)
    tpm_estimate_per_request: int | None = None  # computed: TPM reservation of a full request
    concurrency_limit: int = 2
    # CALLER setting, copied as is from the providers file: the gateway does not read it,
    # it publishes it so that the client knows how long to wait for a task served
    # by this instance (see `sdk.client.execute(wait_timeout=…)`).
    wait_timeout: float | None = None
    disable_timeout: int = 180
    adapter: str = ""
    structured_output: Literal["json_schema", "json_object", "none"] | None = None
    schema_in_system: bool | None = None
    inference: InferenceOverrides | None = None

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(rpm_limit={self.rpm_limit}, model='{self.default_model}', "
            f"base_url='{self.base_url}', weight={self.weight}, "
            f"api_key_length={len(self.api_key.get_secret_value())})"
        )


# ---------------------------------------------------------------------------
# The settings groups
# ---------------------------------------------------------------------------

class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"


class ExecutorSettings(BaseModel):
    kind: Literal["celery"] = "celery"   # "inprocess" will come with the execution port
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


class InferenceSettings(BaseModel):
    """Global inference defaults; overridden per provider (file `inference:`) then per request."""

    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int = 4096   # output budget PER TASK; a batch multiplies it by its agent count
    # Default thinking depth. `None`: nothing is requested, each provider applies its
    # own — this is the behavior that prevailed before 2026-09-10, kept as the
    # default so as not to shift all existing measurements at once.
    thinking_budget: int | None = None
    # Default thinking level. `None`: nothing is sent, the model applies its own
    # (`high` for Flash and Pro, `minimal` for Flash-Lite according to the 2026-09-10 docs).
    thinking_level: str | None = None


class BatchingSettings(BaseModel):
    """Micro-batching. The values are those measured on the July 2026 runs: to be recalibrated
    for other prompts or other providers (see the quota guide)."""

    max_agents: int = 5           # fallback if no provider is configured
    # Accumulation window: a request below the threshold waits this delay to be merged.
    # Set on the measured inter-arrival time of prompts (run 2026-07-10: p50 = 1.4 s).
    delay_seconds: float = 3.0
    # Queue size triggering an immediate dispatch (see get_dispatch_threshold).
    target_agents: int = 10
    assumed_prompt_tokens: int = 2200   # max tokens_in per agent (history +10% margin)
    assumed_output_tokens: int = 800    # estimated tokens_out per agent (sliding TPM reservation)
    # tokens ≈ characters / ratio; measured p50 = 3.24, p10 = 3.05 on FR + JSON prompts.
    token_chars_ratio: float = 3.0
    max_batch_agents: int = 20          # absolute cap of the batch_max_agents computation
    min_output_tokens: int = 512        # minimum acceptable output budget per request
    max_output_tokens: int = 16384      # cap of the max_tokens sent (gpt-4o-mini limit)


class RoutingSettings(BaseModel):
    """How the balancer distributes requests among providers.

    `swrr` spreads the load over all providers in rotation (maximum throughput).
    `cascade` exhausts them in order: the first as long as it accepts, the next only
    when it refuses — daily quota reached, cooldown, or per-minute rate saturated. This is
    what we want when two keys serve the same model and we prefer to consume the first
    entirely before touching the second (request of 2026-09-07).
    """

    policy: Literal["swrr", "cascade"] = "swrr"


class ResilienceSettings(BaseModel):
    max_retries: int = 50
    backoff_base_seconds: float = 1.0
    # Short cooldown of the faulty provider on a switch (parse error, non-recoverable 4xx).
    provider_switch_cooldown_seconds: int = 30
    # Beyond this, the provider is disabled for `disable_timeout` seconds.
    disable_after_consecutive_errors: int = 30
    # Saturation: a batch waits `provider_wait_seconds` for a slot (polled every
    # `saturation_poll_seconds`), then `saturation_retries` retries spaced by
    # `saturation_retry_seconds`. After that, if the eligible providers are only
    # BUSY (RPM/TPM window full, smoothing, concurrency), the batch keeps waiting until
    # `max_retries`; it is abandoned only if they are truly unavailable (cooldown,
    # disabling, daily quota) or if `abandon_when_busy` is true.
    provider_wait_seconds: float = 8.0
    saturation_poll_seconds: float = 2.0
    saturation_retries: int = 2
    saturation_retry_seconds: float = 12.0
    abandon_when_busy: bool = False
    # CLIENT wait (seconds). Beyond it, the client gives up on "Timeout expiré", with no
    # failure kind, and the simulation records a fallback: the batch still waiting in the worker
    # no longer serves anyone. The worker therefore stops waiting `client_wait_margin_seconds`
    # BEFORE, and says so (`error_kind = "surcharge_fournisseur"`, `resume_at`). Default: that of
    # the SDK and the simulation (120 s). The `wait_timeout` of a pinned instance overrides it.
    client_wait_seconds: float = 120.0
    client_wait_margin_seconds: float = 10.0


class RejeuSettings(BaseModel):
    # Root of the responses recorded for exact-prompt replay (see core/rejeu_ab.py).
    # None = replay disabled: a request that names a space is served like the others.
    # The API reads there, the worker writes there: the directory must be mounted in both.
    dir: Path | None = None


class ApiSettings(BaseModel):
    cors_origins: list[str] = Field(default_factory=list)   # empty = no CORS middleware
    max_request_bytes: int = 2_000_000
    auth_tokens: list[SecretStr] = Field(default_factory=list)   # ticket 036: not enforced yet


class TelemetrySettings(BaseModel):
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    service_name: str | None = None      # adds a file sink <workdir>/<service>.log
    workdir: Path = Path(".")
    # Exchange log (full prompts and responses: potential personal data).
    # DISABLED by default; the simulation enables it in its compose.
    exchanges_enabled: bool = False
    exchanges_file: Path | None = None    # default: <workdir>/llm_exchanges.jsonl
    exchanges_max_bytes: int = 200_000_000  # rotated to .1 beyond this; 0 = never
    redactor: str | None = None           # dotted path of a redactor (telemetry.exchanges)


# ---------------------------------------------------------------------------
# The gateway settings
# ---------------------------------------------------------------------------

class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_nested_delimiter="__", extra="ignore", case_sensitive=False,
    )

    redis: RedisSettings = Field(default_factory=RedisSettings)
    executor: ExecutorSettings = Field(default_factory=ExecutorSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    rejeu: RejeuSettings = Field(default_factory=RejeuSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    # Providers file (deployment configuration). None = shipped example + warning.
    providers_file: Path | None = None
    # Where the learned limits live (max_output_tokens revealed by HTTP 400).
    learned_limits: Literal["redis", "file", "none"] = "redis"
    learned_limits_file: Path | None = None   # default: <telemetry.workdir>/learned_limits.json

    # API keys: LLM_GATEWAY_PROVIDER_KEYS__<name> or PROVIDER_KEYS__<name>.
    provider_keys: dict[str, SecretStr] = Field(default_factory=dict)

    # Built after validation — not read from the environment.
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    declared_providers: list[str] = Field(default_factory=list)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, LegacyEnvSource(settings_cls), *yaml_sources(settings_cls))

    # ── Building the providers ───────────────────────────────────────────────

    def resolved_providers_file(self) -> Path:
        return Path(self.providers_file) if self.providers_file else DEFAULT_EXAMPLE_FILE

    @model_validator(mode="after")
    def build_providers(self) -> GatewaySettings:
        if self.providers:   # already built (copy, tests): do not re-read the file
            return self
        path = self.resolved_providers_file()
        if self.providers_file is None:
            logger.warning(
                f"No providers file designated ({ENV_PREFIX}PROVIDERS_FILE): shipped example "
                f"loaded ({path.name}). Without an API key, no provider will be active."
            )
        declared = load_providers_file(path)
        self.declared_providers = sorted(declared.providers)
        self.providers = {
            name: self._resolve(name, entry) for name, entry in declared.providers.items()
        }
        return self

    def _resolve_key(self, name: str, adapter_name: str) -> SecretStr:
        """API key of an instance: PROVIDER_KEYS__<instance>, otherwise the adapter's.

        The fallback to the adapter is REFUSED for an instance suffixed `_key<N>`: its name
        announces a dedicated key, and falling back would hit another project's key without
        a word — this is the `google2_36` incident, where the naming convention sent
        the calls to the wrong key. Without its own key, the instance comes out with
        an empty key and `filter_providers_without_api_key` removes it from the rotation.

        A variable that is present but EMPTY (`PROVIDER_KEYS__x=""`, see `make run NO_GOOGLE=1`)
        is a deliberate blanking: no fallback, no alarm.
        """
        propre = self.provider_keys.get(name)
        if propre is not None:
            return propre
        if _INSTANCE_KEY_SUFFIX.search(name):
            if self.provider_keys.get(adapter_name):
                # Inconsistent config: before the guard, this instance worked "by
                # accident" on the adapter's key. It is now excluded — say it
                # loudly, otherwise the disappearance of an instance shows up nowhere.
                logger.error(
                    f"[ALARME] Instance '{name}' without PROVIDER_KEYS__{name}, yet the key of "
                    f"adapter '{adapter_name}' exists: instance EXCLUDED, no fallback (the name "
                    f"announces a dedicated key). Declare PROVIDER_KEYS__{name} to serve it."
                )
            else:
                logger.warning(
                    f"Instance '{name}' without PROVIDER_KEYS__{name}: excluded from the rotation."
                )
            return SecretStr("")
        return self.provider_keys.get(adapter_name, SecretStr(""))

    def _resolve(self, name: str, entry: ProviderEntry) -> ProviderConfig:
        b = self.batching
        adapter_name = entry.adapter or name
        key = self._resolve_key(name, adapter_name)
        # Token cost (in+out) of one agent in a batch — sizes batch_max_agents.
        tokens_per_agent = b.assumed_prompt_tokens + b.assumed_output_tokens
        batch_max = compute_batch_max_agents(
            tpm_limit=entry.tpm_limit,
            rpm_limit=entry.rpm_limit,
            max_tokens_per_request=entry.max_tokens_per_request,
            tokens_per_agent=tokens_per_agent,
            plafond=b.max_batch_agents,
        )
        if entry.batch_max_agents is not None:
            logger.warning(
                f"Provider '{name}': batch_max_agents={entry.batch_max_agents} written in the file "
                f"is ignored, the gateway computes it ({batch_max})."
            )
        tpm_estimate = batch_max * tokens_per_agent if entry.tpm_limit else None
        logger.info(
            f"Provider '{name}' — batch_max_agents={batch_max} tpm_estimate_per_request={tpm_estimate} "
            f"(tpm={entry.tpm_limit}, rpm={entry.rpm_limit}, cap={b.max_batch_agents})"
        )
        data = entry.model_dump(exclude={"batch_max_agents"})
        return ProviderConfig(
            api_key=key, batch_max_agents=batch_max, tpm_estimate_per_request=tpm_estimate, **data,
        )

    # ── Reading the capacities ──────────────────────────────────────────────

    def _providers_retenus(self, instances_admises: list[str] | None = None) -> dict:
        """The providers on which to size a batch — ticket 084.

        Without a restriction, all of them. With one, only the admitted instances: sizing on
        the full set while only a handful can serve would give a batch size and a dispatch
        threshold set on unreachable capacities. A restriction that designates no known
        provider is ignored HERE — the selection will refuse it with a message that names
        the culprit, and it is not up to the sizing to raise.
        """
        if not instances_admises:
            return self.providers
        retenus = {k: v for k, v in self.providers.items() if k in set(instances_admises)}
        return retenus or self.providers

    def get_batch_max_agents(
        self, force_provider: str | None = None, instances_admises: list[str] | None = None
    ) -> int:
        """Worker batch limit: that of the forced provider, else the min of retained providers."""
        if force_provider:
            cfg = self.providers.get(force_provider)
            if cfg:
                return cfg.batch_max_agents
        retenus = self._providers_retenus(instances_admises)
        if retenus:
            return min(p.batch_max_agents for p in retenus.values())
        return self.batching.max_agents

    def get_dispatch_threshold(
        self, force_provider: str | None = None, instances_admises: list[str] | None = None
    ) -> int:
        """Queue size triggering an immediate dispatch on the API side.

        Forced provider: its capacity. Otherwise the batch target, bounded by the largest
        RETAINED provider — above all NOT the min of the providers, which is 1 because of the
        small TPMs and would make the accumulation window ineffective.
        """
        if force_provider:
            cfg = self.providers.get(force_provider)
            if cfg:
                return cfg.batch_max_agents
        retenus = self._providers_retenus(instances_admises)
        if retenus:
            return min(self.batching.target_agents, max(p.batch_max_agents for p in retenus.values()))
        return self.batching.target_agents

    # ── Flat aliases (compatibility for one version; prefer the groups) ────

    @property
    def redis_url(self) -> str: return self.redis.url
    @property
    def celery_broker_url(self) -> str: return self.executor.celery_broker_url
    @property
    def celery_result_backend(self) -> str: return self.executor.celery_result_backend
    @property
    def max_retries(self) -> int: return self.resilience.max_retries
    @property
    def backoff_base_seconds(self) -> float: return self.resilience.backoff_base_seconds
    @property
    def provider_switch_cooldown_seconds(self) -> int: return self.resilience.provider_switch_cooldown_seconds
    @property
    def batch_max_agents(self) -> int: return self.batching.max_agents
    @property
    def batch_delay_seconds(self) -> float: return self.batching.delay_seconds
    @property
    def batch_target_agents(self) -> int: return self.batching.target_agents
    @property
    def assumed_prompt_tokens(self) -> int: return self.batching.assumed_prompt_tokens
    @property
    def assumed_output_tokens(self) -> int: return self.batching.assumed_output_tokens
    @property
    def token_chars_ratio(self) -> float: return self.batching.token_chars_ratio
    @property
    def max_batch_agents(self) -> int: return self.batching.max_batch_agents
    @property
    def min_output_tokens(self) -> int: return self.batching.min_output_tokens
    @property
    def max_output_tokens(self) -> int: return self.batching.max_output_tokens


# Historical name, kept for existing imports.
Settings = GatewaySettings


# ---------------------------------------------------------------------------
# Filtering, masked exposure, shared access
# ---------------------------------------------------------------------------

def filter_providers_without_api_key(settings: GatewaySettings) -> dict[str, ProviderConfig]:
    valid: dict[str, ProviderConfig] = {}
    b = settings.batching
    for name, provider in settings.providers.items():
        if not provider.api_key.get_secret_value():
            logger.warning(f"Provider '{name}' excluded: missing API key.")
            continue
        if provider.max_tokens_per_request is not None:
            min_needed = provider.batch_max_agents * b.assumed_prompt_tokens + b.min_output_tokens
            if provider.max_tokens_per_request < min_needed:
                logger.warning(
                    f"Provider '{name}' excluded: not enough capacity "
                    f"(max_tokens_per_request={provider.max_tokens_per_request} < "
                    f"batch_max_agents={provider.batch_max_agents} × assumed_prompt_tokens={b.assumed_prompt_tokens} "
                    f"+ min_output_tokens={b.min_output_tokens} = {min_needed})."
                )
                continue
        valid[name] = provider
        logger.info(f"Provider '{name}' included: {provider}")
    return valid


def redacted_dump(settings: GatewaySettings) -> dict[str, Any]:
    """The effective configuration, secrets masked: what `/config` and `config show` publish."""
    data = settings.model_dump(mode="json")
    data["provider_keys"] = {k: "***" for k in settings.provider_keys}
    data["api"]["auth_tokens"] = ["***" for _ in settings.api.auth_tokens]
    for name, cfg in data.get("providers", {}).items():
        cfg["api_key"] = "***" if settings.providers[name].api_key.get_secret_value() else ""
    data["providers_file"] = str(settings.resolved_providers_file())
    return data


def load_provider_defaults(path: Path | None = None) -> dict[str, dict]:
    """The RAW entries of the providers file (with or without a key) — compatibility.

    Consumed by prompt_calibration to know the declared instances. Without a path, that of the
    current settings.
    """
    target = Path(path) if path else get_settings().resolved_providers_file()
    declared: ProvidersFile = load_providers_file(target)
    return {name: entry.model_dump(exclude_none=True) for name, entry in declared.providers.items()}


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    """Shared settings of the process — built on the first call, not at import.

    Providers without a key (or without capacity) are removed here; `declared_providers` keeps
    the full list from the file.
    """
    settings = GatewaySettings()
    settings.providers = filter_providers_without_api_key(settings)
    return settings


__all__ = [
    "ApiSettings",
    "BatchingSettings",
    "ExecutorSettings",
    "GatewaySettings",
    "InferenceSettings",
    "ProviderConfig",
    "RedisSettings",
    "ResilienceSettings",
    "Settings",
    "TelemetrySettings",
    "filter_providers_without_api_key",
    "get_settings",
    "load_provider_defaults",
    "redacted_dump",
]
