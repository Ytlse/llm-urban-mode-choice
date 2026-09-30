"""Layered settings (config/settings.py, config/sources.py, config/providers.py).

No Redis or network access. Each test builds its settings with a temporary providers
file: the repository defaults never come into play.
"""
from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from llm_gateway.config import (
    GatewaySettings,
    ProviderConfig,
    ProvidersConfigError,
    Settings,
    filter_providers_without_api_key,
    load_providers_file,
    redacted_dump,
)
from llm_gateway.config.sources import LegacyEnvSource

PROVIDERS_YAML = """
providers:
  big:
    rpm_limit: 90
    tpm_limit: 300000
    base_url: https://big.example/v1
    default_model: m-big
  tiny:
    rpm_limit: 2
    tpm_limit: 6000
    max_tokens_per_request: 6000
    base_url: https://tiny.example/v1
    default_model: m-tiny
    adapter: groq
    inference:
      temperature: 0.2
  nokey:
    rpm_limit: 10
    base_url: https://nokey.example/v1
    default_model: m
"""


@pytest.fixture
def providers_file(tmp_path):
    p = tmp_path / "providers.yaml"
    p.write_text(PROVIDERS_YAML, encoding="utf-8")
    return p


@pytest.fixture
def clean_env(monkeypatch):
    """No gateway variable or old name in the test environment."""
    import os
    for name in list(os.environ):
        if name.startswith(("LLM_GATEWAY_", "PROVIDER_KEYS__")) or name in (
            "REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "LOG_LEVEL",
            "SERVICE_NAME", "APP_WORKDIR", "LLM_EXCHANGES_FILE",
        ):
            monkeypatch.delenv(name, raising=False)


def _settings(providers_file, **kwargs) -> GatewaySettings:
    return GatewaySettings(providers_file=providers_file, learned_limits="none", **kwargs)


# ── Providers file ───────────────────────────────────────────────────────────

class TestProvidersFile:
    def test_charge_et_calcule_les_capacites(self, clean_env, providers_file):
        s = _settings(providers_file, provider_keys={"big": "k", "groq": "k"})
        assert s.declared_providers == ["big", "nokey", "tiny"]
        big, tiny = s.providers["big"], s.providers["tiny"]
        # tokens per agent = 2200 + 800 = 3000; big: min(300000/3000=100, cap 20, rpm 90) = 20
        assert big.batch_max_agents == 20 and big.tpm_estimate_per_request == 60_000
        # tiny: min(6000/3000=2, 6000/3000=2, rpm 2, 20) = 2
        assert tiny.batch_max_agents == 2
        assert tiny.inference is not None and tiny.inference.temperature == 0.2
        assert tiny.api_key.get_secret_value() == "k", "key inherited from the adapter name"

    def test_cle_inconnue_nomme_le_provider_et_la_cle(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text("providers:\n  x:\n    rpm_limit: 1\n    base_url: u\n    default_model: m\n    tpm_limt: 5\n")
        with pytest.raises(ProvidersConfigError) as exc:
            load_providers_file(p)
        assert "'x'" in str(exc.value) and "tpm_limt" in str(exc.value)

    def test_fichier_absent(self, tmp_path):
        with pytest.raises(ProvidersConfigError, match="not found"):
            load_providers_file(tmp_path / "absent.yaml")

    def test_champ_obligatoire_manquant(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text("providers:\n  x:\n    base_url: u\n    default_model: m\n")
        with pytest.raises(ProvidersConfigError, match="rpm_limit"):
            load_providers_file(p)

    def test_sans_fichier_designe_l_exemple_est_charge(self, clean_env):
        s = GatewaySettings(learned_limits="none")
        assert s.providers_file is None and s.declared_providers, "the shipped example declares providers"
        assert filter_providers_without_api_key(s) == {}, "without a key, none is active"


# ── Filtering and capacities ─────────────────────────────────────────────────

class TestCapacites:
    def test_filtre_sans_cle(self, clean_env, providers_file):
        s = _settings(providers_file, provider_keys={"big": "k"})
        assert set(filter_providers_without_api_key(s)) == {"big"}

    def test_capacite_insuffisante_exclue(self, clean_env, tmp_path):
        # Capacity of 2,500 tokens per request: batch_max_agents drops to 1 and it takes
        # 1 × 2200 + 512 = 2,712 tokens → the provider is excluded even with a key.
        p = tmp_path / "p.yaml"
        p.write_text(PROVIDERS_YAML.replace("max_tokens_per_request: 6000", "max_tokens_per_request: 2500"))
        s = _settings(p, provider_keys={"groq": "k"})
        assert s.providers["tiny"].batch_max_agents == 1
        assert "tiny" not in filter_providers_without_api_key(s)

    def test_seuil_de_dispatch_et_limite_worker(self, clean_env, providers_file):
        s = _settings(providers_file, provider_keys={"big": "k", "groq": "k"})
        s.providers = filter_providers_without_api_key(s)  # big (20) and tiny (2)
        assert s.get_dispatch_threshold() == min(s.batching.target_agents, 20), "bounded by the largest"
        assert s.get_dispatch_threshold("big") == 20
        assert s.get_batch_max_agents() == 2, "the worker stays conservative: min of providers"
        s.providers = {}
        assert s.get_dispatch_threshold() == s.batching.target_agents
        assert s.get_batch_max_agents() == s.batching.max_agents


# ── Layers ───────────────────────────────────────────────────────────────────

class TestCouches:
    def test_constructeur_puis_env_prefixe_puis_defaut(self, clean_env, monkeypatch, providers_file):
        monkeypatch.setenv("LLM_GATEWAY_BATCHING__DELAY_SECONDS", "7.5")
        monkeypatch.setenv("LLM_GATEWAY_REDIS__URL", "redis://env:6379/0")
        s = _settings(providers_file, redis={"url": "redis://init:6379/0"})
        assert s.redis.url == "redis://init:6379/0", "the constructor wins"
        assert s.batching.delay_seconds == 7.5, "the prefixed environment wins over the default"
        assert s.batching.target_agents == 10, "code default"

    def test_fichier_yaml_puis_profil(self, clean_env, monkeypatch, providers_file, tmp_path):
        cfg = tmp_path / "gateway.yaml"
        cfg.write_text("batching:\n  target_agents: 33\nresilience:\n  max_retries: 3\n")
        monkeypatch.setenv("LLM_GATEWAY_CONFIG", str(cfg))
        monkeypatch.setenv("LLM_GATEWAY_PROFILE", "paid")   # paid: delay 1.0, target 20, max_retries 10
        s = _settings(providers_file)
        assert s.batching.target_agents == 33, "the file wins over the profile"
        assert s.resilience.max_retries == 3
        assert s.batching.delay_seconds == 1.0, "the profile wins over the default"

    def test_profil_inconnu_refuse(self, clean_env, monkeypatch, providers_file):
        monkeypatch.setenv("LLM_GATEWAY_PROFILE", "n-existe-pas")
        with pytest.raises(FileNotFoundError, match="unknown profile"):
            _settings(providers_file)

    def test_anciens_noms_lus_mais_le_prefixe_gagne(self, clean_env, monkeypatch, providers_file):
        monkeypatch.setenv("REDIS_URL", "redis://legacy:6379/0")
        monkeypatch.setenv("APP_WORKDIR", "/tmp/legacy")
        monkeypatch.setenv("LLM_GATEWAY_TELEMETRY__WORKDIR", "/tmp/new")
        s = _settings(providers_file)
        assert s.redis.url == "redis://legacy:6379/0"
        assert str(s.telemetry.workdir) == "/tmp/new"
        assert s.redis_url == s.redis.url, "flat alias kept for one version"

    def test_provider_keys_ancien_nom_canonique(self, clean_env, monkeypatch, providers_file):
        monkeypatch.setenv("PROVIDER_KEYS__big", "legacy-key")
        monkeypatch.setenv("LLM_GATEWAY_PROVIDER_KEYS__groq", "new-key")
        s = _settings(providers_file)
        assert s.provider_keys["big"].get_secret_value() == "legacy-key"
        assert s.provider_keys["groq"].get_secret_value() == "new-key"

    def test_source_heritee_ignore_les_variables_absentes(self, clean_env):
        assert LegacyEnvSource(GatewaySettings)() == {}

    def test_alias_settings(self):
        assert Settings is GatewaySettings


# ── Exposure ─────────────────────────────────────────────────────────────────

class TestRedaction:
    def test_dump_masque_les_secrets(self, clean_env, providers_file):
        s = _settings(providers_file, provider_keys={"big": "very-secret"}, api={"auth_tokens": ["t0k"]})
        d = redacted_dump(s)
        assert d["provider_keys"] == {"big": "***"}
        assert d["providers"]["big"]["api_key"] == "***" and d["providers"]["nokey"]["api_key"] == ""
        assert d["api"]["auth_tokens"] == ["***"]
        assert "very-secret" not in str(d) and "t0k" not in str(d)


class TestProviderConfig:
    def test_cle_inconnue_refusee(self):
        with pytest.raises(ValidationError):
            ProviderConfig(rpm_limit=1, base_url="u", default_model="m", weigth=2)

    def test_repr_ne_montre_pas_la_cle(self):
        cfg = ProviderConfig(api_key=SecretStr("abc"), rpm_limit=1, base_url="u", default_model="m")
        assert "abc" not in repr(cfg) and "api_key_length=3" in repr(cfg)


# ── The batch cap is a pure function, reusable outside the gateway ───────────
#
# The cost estimate of an experiment (`experiences/lots.py`) must convert
# trips into requests. Copying the formula there would have made it diverge silently
# the day this one changes: it is extracted, and these tests hold the equivalence.

class TestPlafondDeLotPartage:
    def test_la_fonction_pure_rend_ce_que_resolve_calcule(self, clean_env, providers_file):
        from llm_gateway.core.batching import compute_batch_max_agents

        s = _settings(providers_file, provider_keys={"big": "k", "groq": "k"})
        b = s.batching
        for nom, cfg in s.providers.items():
            assert cfg.batch_max_agents == compute_batch_max_agents(
                tpm_limit=cfg.tpm_limit,
                rpm_limit=cfg.rpm_limit,
                max_tokens_per_request=cfg.max_tokens_per_request,
                tokens_per_agent=b.assumed_prompt_tokens + b.assumed_output_tokens,
                plafond=b.max_batch_agents,
            ), nom

    def test_sans_tpm_le_rpm_borne(self):
        from llm_gateway.core.batching import compute_batch_max_agents

        assert compute_batch_max_agents(
            tpm_limit=None, rpm_limit=3, max_tokens_per_request=None,
            tokens_per_agent=3000, plafond=20,
        ) == 3

    def test_jamais_moins_dun_agent(self):
        from llm_gateway.core.batching import compute_batch_max_agents

        assert compute_batch_max_agents(
            tpm_limit=100, rpm_limit=1, max_tokens_per_request=100,
            tokens_per_agent=3000, plafond=20,
        ) == 1


class TestTailleDeLot:
    """The size of a batch is read from the identifier the worker forges.

    `llm_exchanges.jsonl` records the tokens of the BATCH, one line per request. Without this
    reading, these tokens pass for those of one agent and any estimate built on them is
    multiplied by the grouping factor (observed: 4,607 tokens for 8 agents).
    """

    def test_lit_le_nombre_dagents(self):
        from llm_gateway.core.batching import taille_de_lot

        assert taille_de_lot("batch_7cc7ed2c_8") == 8
        assert taille_de_lot("batch_7cc7ed2c_15") == 15

    def test_forme_inconnue_rend_none_plutot_que_un(self):
        from llm_gateway.core.batching import taille_de_lot

        for mauvais in (None, "", "batch_zz_8", "7cc7ed2c_8", "batch_7cc7ed2c", "batch_7cc7ed2c_0"):
            assert taille_de_lot(mauvais) is None, mauvais


class TestSanteExposeLaCapacite:
    def test_health_publie_batch_max_agents(self, clean_env, providers_file):
        from llm_gateway.balancer.router import LoadBalancer

        s = _settings(providers_file, provider_keys={"big": "k", "groq": "k"})
        actifs = filter_providers_without_api_key(s)

        class _LimiterFactice:
            def current_rpm(self, n): return 0
            def is_quota_exhausted(self, n): return False
            def is_disabled(self, n): return False
            def is_in_cooldown(self, n): return False
            def active_workers(self, n): return 0
            def daily_requests_local_seulement(self, n): return 0
            def daily_tokens_local_seulement(self, n): return 0

        statut = LoadBalancer(actifs, _LimiterFactice()).get_status()
        assert statut["big"]["batch_max_agents"] == actifs["big"].batch_max_agents
        assert statut["big"]["tpm_estimate_per_request"] == actifs["big"].tpm_estimate_per_request
