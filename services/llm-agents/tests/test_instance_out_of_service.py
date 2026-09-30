"""An out-of-service gateway instance is not "available"; a busy one is.

`/health` publishes distinct signals: `quota_exhausted` (the day's bucket is empty), `disabled`
(disabled after consecutive errors — 402 credits, 5xx…), `cooldown`, and `available`
("can take a request NOW": neither disabled, nor in cooldown, nor exhausted, nor saturated
with simultaneous calls).

Two failures fixed this contract:
- 2026-09-07: `cerebras_gpt-oss-120b` in HTTP 402, disabled by the gateway but
  `quota_exhausted: false` (day's bucket intact). The go/no-go only read the quota: experiment
  admitted on a dead instance, discovered by burning its requests.
- 2026-09-08: `lmstudio_muse_glimmer_28b_key1`, a local model with one call at a time. The go/no-go
  then read `available`, false during each generation: two decisions after its resumption,
  the experiment was declared exhausted until 07:00 the next day, with neither quota nor failure.

Pattern to hunt down in this project: a signal read for what it is not.
"""

from experiences.ressources import MoniteurRessources, hors_service, occupee

_PROVIDERS = {
    "cerebras_gpt-oss-120b": {"default_model": "gpt-oss-120b", "rpd_limit": 1000},
    "mistral": {"default_model": "mistral-small-latest"},
    "lmstudio_muse_glimmer_28b_key1": {"default_model": "meta/muse-glimmer", "concurrency_limit": 1},
}


def _moniteur(etat: dict, instances: list[str] | None = None) -> MoniteurRessources:
    m = MoniteurRessources(
        instances or list(_PROVIDERS), _PROVIDERS, base_url="http://x", lecteur=lambda _u: etat
    )
    m.rafraichir()
    return m


def test_desactivee_rend_l_instance_indisponible():
    """The 402 of 2026-09-07: the gateway disables, the day's bucket stays intact."""
    m = _moniteur({
        "cerebras_gpt-oss-120b": {"available": False, "disabled": True, "cooldown": False,
                                  "quota_exhausted": False, "daily_requests": 9},
        "mistral": {"available": True, "disabled": False, "cooldown": False, "quota_exhausted": False,
                    "daily_requests": 280},
    }, ["cerebras_gpt-oss-120b", "mistral"])
    assert not m.disponible("cerebras_gpt-oss-120b"), "disabled, hence not servable"
    assert m.disponible("mistral")
    assert m.instances_disponibles() == ["mistral"]


def test_en_cooldown_rend_l_instance_indisponible():
    m = _moniteur({"mistral": {"available": False, "disabled": False, "cooldown": True}}, ["mistral"])
    assert not m.disponible("mistral")
    assert hors_service({"disabled": False, "cooldown": True})


def test_occupee_n_est_pas_hors_service():
    """On 2026-09-08: a local model with one call at a time is `available: false` during each
    generation. It is servable — the next request will wait its turn, that is all."""
    etat = {"lmstudio_muse_glimmer_28b_key1": {"available": False, "disabled": False, "cooldown": False,
                                               "quota_exhausted": False, "active_tasks": 1, "daily_requests": 11}}
    m = _moniteur(etat, ["lmstudio_muse_glimmer_28b_key1"])
    assert m.disponible("lmstudio_muse_glimmer_28b_key1"), "occupée ≠ hors service"
    assert not m.epuise(), "no quota, no failure: nothing is exhausted"
    assert not hors_service(etat["lmstudio_muse_glimmer_28b_key1"])
    assert occupee(etat["lmstudio_muse_glimmer_28b_key1"])
    ligne = m.tableau()[0]
    assert ligne["disponible"] and ligne["occupee"] and not ligne["epuisee"]


def test_une_experience_sur_la_seule_instance_hors_service_est_epuisee():
    """The refusal must come at admission, not after nine burnt requests."""
    m = _moniteur(
        {"cerebras_gpt-oss-120b": {"available": False, "disabled": True, "cooldown": False, "daily_requests": 9}},
        ["cerebras_gpt-oss-120b"],
    )
    assert m.epuise(), "no servable instance"
    assert "désactivée côté passerelle" in m.raison_epuisement(), m.raison_epuisement()


def test_sans_disabled_ni_cooldown_available_fait_foi():
    """Old gateway, which does not publish the two fields: `available` remains the only signal."""
    m = _moniteur({"mistral": {"available": False, "daily_requests": 9}}, ["mistral"])
    assert not m.disponible("mistral")
    m = _moniteur({"mistral": {"available": True, "daily_requests": 9}}, ["mistral"])
    assert m.disponible("mistral")


def test_available_absent_reste_permissif():
    """A gateway that publishes none of these fields must block no experiment."""
    m = _moniteur({"mistral": {"daily_requests": 3}})
    assert m.disponible("mistral")
    assert not hors_service({}) and not occupee({})


def test_le_quota_du_jour_declaratif_ne_bloque_pas_sans_429():
    """The local ceiling is declarative and informative: zero margin does not block if quota_exhausted is False."""
    m = _moniteur({
        "cerebras_gpt-oss-120b": {"available": True, "disabled": False, "cooldown": False,
                                  "quota_exhausted": False, "daily_requests": 1000},
    })
    assert m.disponible("cerebras_gpt-oss-120b"), "informative ceiling: servable as long as no 429 is set"
    assert not occupee({"available": True}), "disponible n'est pas occupée"

    # On the other hand, the quota_exhausted flag set after a 429 makes the instance non-servable
    m_epuise = _moniteur({
        "cerebras_gpt-oss-120b": {"available": False, "disabled": False, "cooldown": False,
                                  "quota_exhausted": True, "daily_requests": 1000},
    })
    assert not m_epuise.disponible("cerebras_gpt-oss-120b")
