"""API key set of an experiment (ticket 035, spec parallelisation_experiences, R4/R8/R9).

LLM throughput is capped **per API key** on the provider side: two experiments sharing a
key compete for the same quota. From a pinned **model**, this module derives the set of
**key identities** the experiment may call on — the basis of the parallel/serial
safeguard (`reservations.py`).

Key identity of an instance (gateway resolution, `llm_gateway/config/settings.py`):
`PROVIDER_KEYS[nom_instance]` if it exists, otherwise `PROVIDER_KEYS[adapter]` (adapter = the
`adapter` field, or else the instance name). On the experiments side, the gateway's
`PROVIDER_KEYS` entries are unknown; the **adapter** (or else the instance name) is therefore
taken as identity. This is CONSERVATIVE: two instances of the same adapter are deemed to share
a key even if the gateway gave them distinct keys — we may serialise wrongly, never
parallelise wrongly (fail-safe, consistent with R12). A per-instance override visible in the
environment (`(LLM_GATEWAY_)PROVIDER_KEYS__<instance>`) is honoured when present.
"""

from __future__ import annotations

import os
import re

from experiences.ressources import instances_pour_modele

# Key rank in an instance name: `google_gemini35_key2` → "2".
_RANG_CLE = re.compile(r"_key(\d+)$")


def _override_instance_present(nom_instance: str) -> bool:
    """Is a key specific to this instance declared in the environment?"""
    cible = nom_instance.lower()
    for var in os.environ:
        bas = var.lower()
        for prefixe in ("provider_keys__", "llm_gateway_provider_keys__"):
            if bas.startswith(prefixe) and bas[len(prefixe) :] == cible:
                return True
    return False


def identite_cle(nom_instance: str, cfg: dict) -> str:
    """Identity of the API key served by an instance (see the module docstring)."""
    adapter = cfg.get("adapter") if isinstance(cfg, dict) else None
    # Convention <model>_key<N> (2026-09-08): the name carries the key's RANK, so the physical
    # key reads `<adapter>_key<N>`. That is what must be kept, NOT the instance name:
    # `google_gemini31_key1` and `google_gemini35_key1` are two models on ONE key — treating
    # them as two identities would run them in parallel on the same quota, the mistake this
    # module forbids itself. Symmetrically, `…_key1` and `…_key2` are indeed two keys, where
    # the old per-adapter rule merged them.
    m = _RANG_CLE.search(nom_instance)
    if m and adapter:
        return f"{adapter}_key{m.group(1)}"
    if _override_instance_present(nom_instance):
        return nom_instance
    return str(adapter or nom_instance)


def jeu_de_cles(
    modele: str,
    providers: dict[str, dict],
    instances_admises: list[str] | None = None,
    portee: str | None = None,
) -> set[str]:
    """Set of key identities an experiment on `modele` may call on.

    `instances_admises`, when given, restricts to the instances actually available
    (key present, quota not exhausted): instances without an available key do not count (R9).
    A model without an instance (replay, local decision-maker, empty offer) gives an **empty** set (R8/R9).

    `portee` restricts to the side pinned by the experiment (`local` / `distant`): a model served
    on both sides must not reserve the remote quota when the experiment runs locally.
    Note: all LM Studio instances share the `openai_compatible` adapter, hence ONE
    key identity — two local experiments are serialised, which is what we want on one machine.
    """
    instances = instances_pour_modele(modele, providers, portee)
    if instances_admises is not None:
        garde = set(instances_admises)
        instances = [i for i in instances if i in garde]
    return {identite_cle(i, providers.get(i, {})) for i in instances}


__all__ = ["identite_cle", "jeu_de_cles"]
