"""Routing restricted to a list of admitted instances.

The question these tests answer: **is the restriction honoured at SELECTION, before a call
is paid for, and does it survive the whole journey — API validation, batch key, passage
through Celery, switchover after an incident?** A mechanism that only held for the first
call would be worse than none: it would give a false guarantee.
"""

import pytest

from llm_gateway.balancer.router import LoadBalancer, RestrictionInstances
from llm_gateway.config.settings import ProviderConfig
from llm_gateway.core.batching import compute_batch_key
from llm_gateway.core.models import LLMRequest


def _provider(rpm: int = 15, weight: float = 1.0, batch_max: int = 5) -> ProviderConfig:
    return ProviderConfig(
        rpm_limit=rpm, base_url="https://example.test", default_model="model-x",
        weight=weight, api_key="test-key", batch_max_agents=batch_max,
    )


class LimiteurFactice:
    """Accepts, refuses, or accepts only a finite number of reservations — per instance."""

    def __init__(self, capacites: dict[str, int]):
        self.capacites = dict(capacites)
        self.reservations: list[str] = []

    def try_reserve(self, provider: str, *_a, **_k) -> bool:
        if self.capacites.get(provider, 0) <= 0:
            return False
        self.capacites[provider] -= 1
        self.reservations.append(provider)
        return True


CINQ = ("alpha", "beta", "gamma", "delta", "epsilon")


@pytest.fixture
def fournisseurs() -> dict[str, ProviderConfig]:
    return {nom: _provider() for nom in CINQ}


def _balancer(fournisseurs, capacites=None, policy="swrr") -> LoadBalancer:
    caps = capacites if capacites is not None else {nom: 99 for nom in fournisseurs}
    return LoadBalancer(fournisseurs, LimiteurFactice(caps), policy=policy)


# ── A. Filtering at selection ───────────────────────────────────────────────────────────


def test_A1_rotation_ne_sert_que_les_admises(fournisseurs):
    lb = _balancer(fournisseurs)
    admises = ["beta", "delta"]
    servis = {lb.select_provider(admises=admises) for _ in range(20)}
    assert servis == {"beta", "delta"}, (
        f"the rotation served outside the allowed set: {servis - set(admises)}"
    )


def test_A2_cascade_conserve_son_ordre_sous_restriction(fournisseurs):
    """The declared order remains the priority; only the set shrinks."""
    lb = _balancer(fournisseurs, capacites={n: 1 for n in fournisseurs}, policy="cascade")
    premier = lb.select_provider(admises=["gamma", "beta"])
    second = lb.select_provider(admises=["gamma", "beta"])
    # `beta` precedes `gamma` in the configuration: the cascade must start on it first.
    assert (premier, second) == ("beta", "gamma")


@pytest.mark.parametrize("politique", ["swrr", "cascade"])
@pytest.mark.parametrize("restriction", [None, []])
def test_A3_sans_restriction_le_comportement_est_inchange(fournisseurs, politique, restriction):
    """`None` and an empty list both mean "no constraint"."""
    lb = _balancer(fournisseurs, policy=politique)
    servis = {lb.select_provider(admises=restriction) for _ in range(30)}
    attendu = {"alpha"} if politique == "cascade" else set(CINQ)
    assert servis == attendu


def test_A4_la_bascule_reste_dans_l_ensemble_admis(fournisseurs):
    """The first allowed one refuses: we move to the next ALLOWED one, never beyond."""
    lb = _balancer(fournisseurs, capacites={"beta": 0, "delta": 5, "alpha": 99})
    servis = {lb.select_provider(admises=["beta", "delta"]) for _ in range(5)}
    assert servis == {"delta"}, "an instance outside the list served as fallback"


def test_A5_toutes_les_admises_saturees_leve_en_les_nommant(fournisseurs):
    lb = _balancer(fournisseurs, capacites={"beta": 0, "delta": 0, "alpha": 99})
    with pytest.raises(RuntimeError) as err:
        lb.select_provider(admises=["beta", "delta"])
    message = str(err.value)
    assert "beta" in message and "delta" in message
    assert "alpha" not in message, (
        "the message offers an instance outside the set as fallback"
    )


def test_A6_une_liste_sans_aucune_instance_connue_est_refusee(fournisseurs):
    lb = _balancer(fournisseurs)
    with pytest.raises(RestrictionInstances):
        lb.select_provider(admises=["nonexistent"])


def test_A7_une_liste_a_moitie_fausse_est_refusee(fournisseurs):
    """A typo must not turn into a silent partial restriction."""
    lb = _balancer(fournisseurs)
    with pytest.raises(RestrictionInstances) as err:
        lb.select_provider(admises=["beta", "betaa"])
    assert "betaa" in str(err.value)


def test_A6bis_une_liste_fausse_ne_vaut_JAMAIS_aucune_restriction(fournisseurs):
    """The heart of the batch: the error must not degenerate into "help yourself everywhere"."""
    lb = _balancer(fournisseurs)
    try:
        lb.select_provider(admises=["nonexistent"])
    except RestrictionInstances:
        pass
    else:
        pytest.fail("a wrong restriction was accepted")
    assert lb._limiter.reservations == [], "a call was reserved despite the wrong restriction"


def test_A8_force_hors_des_admises_est_refuse(fournisseurs):
    lb = _balancer(fournisseurs)
    with pytest.raises(RestrictionInstances) as err:
        lb.select_provider(force="alpha", admises=["beta"])
    assert "alpha" in str(err.value) and "beta" in str(err.value)


def test_A9_force_dans_les_admises_est_servi(fournisseurs):
    lb = _balancer(fournisseurs)
    assert lb.select_provider(force="beta", admises=["beta", "delta"]) == "beta"


# ── B. The restriction survives the journey ─────────────────────────────────────────────


def _requete(**extra) -> LLMRequest:
    base = {"category": "itinary_multi_agent", "agents": [{"agent_id": "1"}]}
    return LLMRequest(**{**base, **extra})


def test_B1_le_champ_survit_a_la_validation():
    """⚠ `LLMRequest` does not forbid extra fields.

    A field set by the client without being declared in the model would be SILENTLY ignored
    by validation — no exception, no log, and a measurement taken under a restriction that
    never existed. This test guards that property.
    """
    req = _requete(instances_admises=["beta", "delta"])
    assert req.instances_admises == ["beta", "delta"]


def test_B2_sans_restriction_le_champ_vaut_none():
    assert _requete().instances_admises is None


def test_B3_la_restriction_est_serialisable():
    """It crosses Celery and Redis: it must be plain JSON, not a set."""
    import json

    charge = _requete(instances_admises=["beta"]).model_dump()
    assert json.loads(json.dumps(charge))["instances_admises"] == ["beta"]


def test_B4_le_parametre_est_bien_cable_du_worker_au_selecteur():
    """The Celery task accepts the restriction and passes it on to `select_provider`."""
    import inspect

    from llm_gateway.worker import task_worker

    signature = inspect.signature(task_worker.process_batch_task.__wrapped__)
    assert "instances_admises" in signature.parameters
    source = inspect.getsource(task_worker.process_batch_task.__wrapped__)
    assert "admises=instances_admises" in source


# ── C. The batch key ─────────────────────────────────────────────────────────────────────


def test_C1_restrictions_differentes_donnent_des_lots_differents():
    """A batch is served by ONE instance: mixing two restrictions would serve one outside
    its set, without any trace."""
    a = compute_batch_key(_requete(instances_admises=["beta"]))
    b = compute_batch_key(_requete(instances_admises=["delta"]))
    assert a != b


def test_C2_meme_restriction_meme_lot():
    a = compute_batch_key(_requete(instances_admises=["beta", "delta"]))
    b = compute_batch_key(_requete(instances_admises=["beta", "delta"]))
    assert a == b


def test_C3_l_ordre_de_declaration_ne_compte_pas():
    """It is a set, not a sequence."""
    a = compute_batch_key(_requete(instances_admises=["beta", "delta"]))
    b = compute_batch_key(_requete(instances_admises=["delta", "beta"]))
    assert a == b


def test_C4_restreinte_et_non_restreinte_ne_se_melangent_pas():
    assert compute_batch_key(_requete(instances_admises=["beta"])) != compute_batch_key(_requete())


# ── D. Switchover after an incident ─────────────────────────────────────────────────────


def test_D1_la_bascule_reporte_la_restriction():
    """Without this, the restriction vanished at the FIRST incident and the retry went back
    to free rotation — the defect this batch removes."""
    import inspect

    from llm_gateway.worker import task_worker

    source = inspect.getsource(task_worker._switch_provider_or_fail)
    assert '"instances_admises": instances_admises' in source, (
        "the retry after switchover does not carry the restriction over"
    )


def test_D3_le_chemin_402_transmet_la_restriction():
    import inspect

    from llm_gateway.worker import task_worker

    assert "instances_admises" in inspect.signature(task_worker._credits_epuises).parameters
    assert "instances_admises=instances_admises" in inspect.getsource(task_worker._credits_epuises)


def test_D_tous_les_appels_de_bascule_la_transmettent():
    """A single call site forgetting it would be enough to lose the guarantee."""
    import inspect

    from llm_gateway.worker import task_worker

    source = inspect.getsource(task_worker.process_batch_task.__wrapped__)
    appels = source.count("_switch_provider_or_fail(") + source.count("_credits_epuises(")
    portes = source.count("instances_admises=instances_admises")
    assert portes >= appels, (
        f"{appels} switchover call(s) for {portes} transmission(s) of the restriction"
    )


# ── E. Batch sizing ──────────────────────────────────────────────────────────────────────


def _settings_avec(fournisseurs):
    from llm_gateway.config.settings import Settings

    s = Settings()
    s.providers = fournisseurs
    return s


def test_E1_taille_de_lot_calculee_sur_les_admises(fournisseurs):
    fournisseurs["beta"] = _provider(batch_max=2)
    fournisseurs["delta"] = _provider(batch_max=7)
    s = _settings_avec(fournisseurs)
    assert s.get_batch_max_agents(instances_admises=["delta"]) == 7
    assert s.get_batch_max_agents(instances_admises=["beta", "delta"]) == 2


def test_E2_seuil_de_dispatch_calcule_sur_les_admises(fournisseurs):
    fournisseurs["delta"] = _provider(batch_max=7)
    s = _settings_avec(fournisseurs)
    restreint = s.get_dispatch_threshold(instances_admises=["delta"])
    assert restreint == min(s.batching.target_agents, 7)


def test_E3_sans_restriction_les_valeurs_sont_inchangees(fournisseurs):
    s = _settings_avec(fournisseurs)
    assert s.get_batch_max_agents() == s.get_batch_max_agents(instances_admises=[])
    assert s.get_dispatch_threshold() == s.get_dispatch_threshold(instances_admises=None)
