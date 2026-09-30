"""No instance requests a model through an alias that the provider has retired.

On 2026-09-10, `google_gemini31_key1` requested `gemini-3.1-flash-lite-preview`: Google
served `gemini-3.1-flash-lite` WITHOUT an error, and the sealed fingerprint of each measurement
therefore carried a model name that had not answered. The run guard
(`_refuser_substitution_de_modele`) detects it, but after consuming quota and in the
middle of a campaign. This test detects it before the commit.

`-latest` is NOT refused here: `mistral-small-latest` is a live alias, maintained by
its provider, and choosing it remains a usage decision. What is refused is an alias
known BY NAME to be retired.
"""

from pathlib import Path

import yaml

PROVIDERS_YAML = Path(__file__).resolve().parents[4] / "config" / "llm_gateway" / "providers.yaml"

# Aliases OBSERVED to be served by another model. Add an entry
# here for each observed substitution — this is the only registry in the repository.
ALIAS_RETIRES = {
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
}


def test_le_fichier_de_providers_est_bien_la():
    """Without this guard, a path error would make the next test pass vacuously."""
    assert PROVIDERS_YAML.is_file(), PROVIDERS_YAML


def test_aucune_instance_ne_demande_un_alias_retire():
    providers = yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8"))["providers"]
    assert providers, "providers.yaml read empty: the test would check nothing"
    fautives = {
        nom: cfg["default_model"]
        for nom, cfg in providers.items()
        if (cfg or {}).get("default_model") in ALIAS_RETIRES
    }
    assert not fautives, (
        f"instances requesting a retired alias: {fautives}. The provider serves the "
        f"successor silently and the measurement carries a wrong name. Replace with "
        f"{ {k: v for k, v in ALIAS_RETIRES.items()} }.")
