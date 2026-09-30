"""Scope of a gateway decider: `local` or `distant`.

Why this rule exists: `qwen/qwen3.8-27b` is the `default_model` of a Groq instance AND
of an LM Studio instance. Since pinning works by model equality, the experiment
`exp_qwen38-27b_minper_jtir_pop-1000_PANEL_t0_nosim` started on Groq and would have continued on the local
4-bit MLX once the quota ran out — two quantizations under a single name. The scope names the
side; both uses remain open, it is silence that no longer is.

One test per rule. No access to the repo's providers.yaml except P4, which checks precisely that it
does not put the two implementations of the local rule in disagreement.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from experiences import experience as E
from experiences import nommage as N
from experiences.cles import jeu_de_cles
from experiences.ressources import (
    est_instance_locale,
    instances_pour_modele,
    portees_pour_modele,
)

RACINE = Path(__file__).resolve().parents[3]

# Two instances serving the SAME model identifier on both sides: the actual
# configuration of 2026-09-11, reduced to what matters.
PROVIDERS = {
    "groq_qwen_qwen3_8_27b_key1": {
        "adapter": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "qwen/qwen3.8-27b",
        "rpm_limit": 30,
    },
    "lmstudio_qwen3_8_27b_key1": {
        "adapter": "openai_compatible",
        "base_url": "http://host.docker.internal:1234/v1",
        "default_model": "qwen/qwen3.8-27b",
        "rpm_limit": 30,
    },
    # Two KEYS of the same provider: a single side, so no ambiguity to settle.
    "google_gemini31_key1": {
        "adapter": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-3.1-flash-lite",
        "rpm_limit": 15,
    },
    "google_gemini31_key2": {
        "adapter": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-3.1-flash-lite",
        "rpm_limit": 15,
    },
}


@pytest.fixture
def providers_factices(tmp_path, monkeypatch):
    """`charger_providers()` reads THIS file: `candidats_providers` puts the env first."""
    p = tmp_path / "providers.yaml"
    p.write_text(yaml.safe_dump({"providers": PROVIDERS}), encoding="utf-8")
    monkeypatch.setenv("LLM_GATEWAY_PROVIDERS_FILE", str(p))
    return p


# ── P1–P4: the scope of an instance ──────────────────────────────────────────


def test_P1_sans_portee_les_deux_bords_repondent():
    """Compat: a definition predating the field keeps EXACTLY its behaviour."""
    assert instances_pour_modele("qwen/qwen3.8-27b", PROVIDERS) == [
        "groq_qwen_qwen3_8_27b_key1",
        "lmstudio_qwen3_8_27b_key1",
    ]


def test_P2_la_portee_restreint_au_bord_demande():
    assert instances_pour_modele("qwen/qwen3.8-27b", PROVIDERS, "local") == [
        "lmstudio_qwen3_8_27b_key1"
    ]
    assert instances_pour_modele("qwen/qwen3.8-27b", PROVIDERS, "distant") == [
        "groq_qwen_qwen3_8_27b_key1"
    ]


@pytest.mark.parametrize(
    "url, locale",
    [
        ("http://host.docker.internal:1234/v1", True),
        ("http://localhost:1234/v1", True),
        ("http://127.0.0.1:1234/v1", True),
        ("https://api.groq.com/openai/v1", False),
        ("http://host.docker.internal:8000/v1", False),  # the host, but not LM Studio
        ("", False),
    ],
)
def test_P3_une_instance_est_locale_si_elle_vise_LM_Studio_sur_cette_machine(
    url, locale
):
    assert est_instance_locale({"base_url": url}) is locale


def test_P4_la_regle_locale_ne_diverge_pas_de_celle_du_tableau_de_bord():
    """The rule is written twice — container and host — so it is pinned here.

    `experiences.ressources` runs in the `controller` container, without the dashboard
    package; `scripts.dashboard.lmstudio` runs on the host, without the container's `loguru`
    or `yaml`. They cannot import each other: this test is the only safeguard
    against their drift, and it works on the ACTUAL providers.yaml.
    """
    if str(RACINE) not in sys.path:
        sys.path.insert(0, str(RACINE))
    from scripts.dashboard import lmstudio

    reel = yaml.safe_load(
        (RACINE / "config" / "llm_gateway" / "providers.yaml").read_text(
            encoding="utf-8"
        )
    )
    instances = (reel or {}).get("providers") or {}
    assert instances, "actual providers.yaml not found or empty"
    desaccords = [
        nom
        for nom, cfg in instances.items()
        if isinstance(cfg, dict)
        and est_instance_locale(cfg) != lmstudio.est_instance_lmstudio(cfg)
    ]
    assert not desaccords, f"the two rules do not say the same thing: {desaccords}"


# ── P5–P7: the refusal, which targets SILENCE and not use ────────────────────


class _JeuFactice:
    """The bare minimum `refuser_si_impossible` reads on a closed, up-to-date set."""

    nom = "jeu_t"
    clos = True
    jour_simule = "2026-03-16"
    manifest: ClassVar[dict] = {"dependances": {}}

    def verifier_population(self, _population):
        return None

    def jours_equivalents(self):
        return []


def _exp(tmp_path, decideur: dict) -> E.Experience:
    p = tmp_path / "exp.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "nom": "exp_t",
                "population": {"chemin": "/data/eqasim-output/population_1000_PANEL"},
                "jeu": {"nom": "jeu_t"},
                "gabarit": {"categorie": "itinary_multi_agent"},
                "decideur": decideur,
                "mode": "sans_simulateur",
                "calendrier": {
                    "politique": "commune",
                    "date": "2026-03-16",
                    "graine": 42,
                },
                "horizon_jours": 1,
                "memoire": False,
                "evenements": [],
                "graine_ordre": 42,
                "graine_tirage": 42,
                "regroupement": {"parallelisme": 4},
                "tolerances_horaires": dict(N.TOLERANCES_REFERENCE),
                "max_candidats": 6,
                "attente_max_s": 5,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return E.charger_experience(p)


def _refus(tmp_path, decideur: dict) -> list[str]:
    refus, _ = E.refuser_si_impossible(
        _exp(tmp_path, decideur),
        _JeuFactice(),
        object(),
        dependances={},
        periodes={},
    )
    return refus


def test_P5_un_modele_servi_des_deux_cotes_sans_portee_est_refuse(
    tmp_path, providers_factices
):
    refus = _refus(tmp_path, {"type": "passerelle", "modele": "qwen/qwen3.8-27b"})
    ambigus = [r for r in refus if "des deux côtés" in r]
    assert len(ambigus) == 1, refus
    # The message must make it possible to ACT: both instances named, and the way out stated.
    assert "groq_qwen_qwen3_8_27b_key1" in ambigus[0]
    assert "lmstudio_qwen3_8_27b_key1" in ambigus[0]
    assert "decideur.portee" in ambigus[0]


@pytest.mark.parametrize("portee", ["local", "distant"])
def test_P6_la_portee_posee_rouvre_les_deux_usages(
    tmp_path, providers_factices, portee
):
    """The goal is not to forbid a side: it is to know which one is measured."""
    refus = _refus(
        tmp_path,
        {"type": "passerelle", "modele": "qwen/qwen3.8-27b", "portee": portee},
    )
    assert not [r for r in refus if "des deux côtés" in r], refus


def test_P7_deux_cles_d_un_meme_fournisseur_ne_sont_pas_une_ambiguite(
    tmp_path, providers_factices
):
    """Two interchangeable quota buckets: nothing to decide, nothing to ask."""
    refus = _refus(tmp_path, {"type": "passerelle", "modele": "gemini-3.1-flash-lite"})
    assert not [r for r in refus if "des deux côtés" in r], refus
    assert portees_pour_modele("gemini-3.1-flash-lite", PROVIDERS) == {
        "distant": ["google_gemini31_key1", "google_gemini31_key2"]
    }


def test_P8_la_portee_ne_vaut_que_pour_la_passerelle():
    erreurs = E.DecideurSpec(type="aleatoire", graine=1, portee="local").valider()
    assert any("ne vaut que pour un décideur passerelle" in e for e in erreurs)


# ── P9–P10: the fingerprint, which seals the archives ────────────────────────


def test_P9_une_definition_sans_portee_garde_son_empreinte():
    """Archive non-regression: already sealed runs must remain verifiable."""
    dec = E.DecideurSpec(type="passerelle", modele="qwen/qwen3.8-27b")
    # The formula from BEFORE the field, rebuilt here: the test fails if the scope sneaks in.
    brut = f"passerelle|qwen/qwen3.8-27b|{sorted({}.items())}||None|"
    assert dec.empreinte() == hashlib.sha256(brut.encode("utf-8")).hexdigest()


def test_P10_deux_portees_sont_deux_decideurs():
    commun = {"type": "passerelle", "modele": "qwen/qwen3.8-27b"}
    local = E.DecideurSpec(**commun, portee="local").empreinte()
    distant = E.DecideurSpec(**commun, portee="distant").empreinte()
    muet = E.DecideurSpec(**commun).empreinte()
    assert local != distant, "two quantizations do not return the same decisions"
    assert muet not in (local, distant)


# ── P11–P12: the name, and the quota ─────────────────────────────────────────


def _def_nommage(portee):
    dec = {
        "type": "passerelle",
        "modele": "qwen/qwen3.8-27b",
        "parametres": {"temperature": 0.0},
    }
    if portee:
        dec["portee"] = portee
    return {
        "population": {"chemin": "/data/eqasim-output/population_1000_PANEL"},
        "jeu": {"nom": "population_1000_PANEL_20260316"},
        "gabarit": {"variante": "prompt_minimal_01"},
        "decideur": dec,
        "calendrier": {"politique": "aleatoire", "date": "2026-03-16", "graine": 42},
        "mode": "sans_simulateur",
    }


def test_P11_seul_le_local_se_dit_dans_le_nom():
    """`distant` is the reference (N7): no existing name changes."""
    muet = N.nom_canonique(_def_nommage(None))
    assert muet == "exp_qwen38-27b_promin01_jtir_pop-1000_PANEL_t0_nosim"
    assert N.nom_canonique(_def_nommage("distant")) == muet
    assert (
        N.nom_canonique(_def_nommage("local"))
        == "exp_qwen38-27b_local_promin01_jtir_pop-1000_PANEL_t0_nosim"
    )


def test_P12_le_quota_reserve_suit_le_bord_epingle():
    """A local experiment must not reserve the Groq quota, and vice versa."""
    assert jeu_de_cles("qwen/qwen3.8-27b", PROVIDERS, portee="distant") == {"groq_key1"}
    assert jeu_de_cles("qwen/qwen3.8-27b", PROVIDERS, portee="local") == {
        "openai_compatible_key1"
    }
    # Without a scope, both sides are reserved: conservative, as before the field.
    assert jeu_de_cles("qwen/qwen3.8-27b", PROVIDERS) == {
        "groq_key1",
        "openai_compatible_key1",
    }
