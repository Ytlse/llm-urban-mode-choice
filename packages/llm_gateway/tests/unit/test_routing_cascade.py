"""Cascade routing: use up one key before starting on the next.

Two keys serve the same model, each with its bucket of 500 requests per day. SWRR routing
draws on them in parallel; the cascade consumes the first one entirely and only switches
when it REFUSES — quota reached, cooldown, or per-minute rate saturated (request of
2026-09-07).
"""

import pytest

from llm_gateway.balancer.router import LoadBalancer
from llm_gateway.config.settings import ProviderConfig, RoutingSettings


def _provider(rpm: int = 15, weight: float = 1.0) -> ProviderConfig:
    return ProviderConfig(
        rpm_limit=rpm, base_url="https://example.test", default_model="model-x", weight=weight,
        api_key="test-key",
    )


class LimiteurFactice:
    """A limiter that accepts, refuses, or accepts only a finite number of reservations."""

    def __init__(self, capacites: dict[str, int]):
        self.capacites = dict(capacites)
        self.reservations: list[str] = []

    def try_reserve(self, provider: str, *_a, **_k) -> bool:
        """The only method `_try_reserve` calls: no catch-all net, which would hide a
        refusal and make these tests always green."""
        if self.capacites.get(provider, 0) <= 0:
            return False
        self.capacites[provider] -= 1
        self.reservations.append(provider)
        return True


@pytest.fixture
def fournisseurs() -> dict[str, ProviderConfig]:
    return {"cle1": _provider(), "cle2": _provider(), "cle3": _provider()}


def test_la_cascade_epuise_le_premier_avant_de_basculer(fournisseurs):
    limiteur = LimiteurFactice({"cle1": 3, "cle2": 2, "cle3": 99})
    balancer = LoadBalancer(fournisseurs, limiteur, policy="cascade")

    choisis = [balancer.select_provider() for _ in range(7)]
    assert choisis == ["cle1", "cle1", "cle1", "cle2", "cle2", "cle3", "cle3"], choisis


def test_le_swrr_etale_la_charge(fournisseurs):
    limiteur = LimiteurFactice({"cle1": 99, "cle2": 99, "cle3": 99})
    balancer = LoadBalancer(fournisseurs, limiteur, policy="swrr")

    choisis = {balancer.select_provider() for _ in range(6)}
    assert len(choisis) > 1, "SWRR routing must reach several providers"


def test_la_cascade_epuisee_le_dit_avec_l_ordre_suivi(fournisseurs):
    limiteur = LimiteurFactice({})
    balancer = LoadBalancer(fournisseurs, limiteur, policy="cascade")

    with pytest.raises(RuntimeError) as refus:
        balancer.select_provider()
    message = str(refus.value)
    assert "Cascade exhausted" in message
    for nom in ("cle1", "cle2", "cle3"):
        assert nom in message, message


def test_la_politique_par_defaut_reste_le_swrr():
    assert RoutingSettings().policy == "swrr"
    assert RoutingSettings(policy="cascade").policy == "cascade"
    with pytest.raises(ValueError):
        RoutingSettings(policy="random")


def test_le_depot_demande_la_cascade_par_l_environnement():
    """The repo runs in cascade mode. The providers file accepts ONLY providers
    (it refuses any other key): the policy therefore goes through the compose environment."""
    from pathlib import Path

    racine = Path(__file__).resolve().parents[4]
    compose = (racine / "infra" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "LLM_GATEWAY_ROUTING__POLICY: ${LLM_GATEWAY_ROUTING__POLICY:-cascade}" in compose

    import yaml
    conf = yaml.safe_load((racine / "config" / "llm_gateway" / "providers.yaml").read_text(encoding="utf-8"))
    assert "routing" not in conf, "this file only accepts providers: it would refuse the key"
