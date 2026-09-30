"""The repatriated scoring formula stays the calibration engine's (2026-09-29).

`scripts/synthesis/formule_score` is a copy of the measurement part of the engine
(`calibration/metrics.py`, `Scores`, the reading of the options). Every score of this
repository goes through the copy; the private calibration engine keeps its own. A formula
changed on one side only would make the two disagree without a word: these tests fail
instead, and force a conscious choice (re-copy, or keep the divergence and say so).

The engine is looked for in the submodule `prompt_calibration/`, then in a clone next to
the repository. Absent (public copy, fresh clone): the comparison is skipped, by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COPIE = REPO_ROOT / "scripts" / "synthesis" / "formule_score"
CANDIDATS = [REPO_ROOT / "prompt_calibration" / "calibration",
             REPO_ROOT.parent / "prompt_calibration" / "calibration"]
MOTEUR = next((c for c in CANDIDATS if (c / "metrics.py").is_file()), None)
needs_engine = pytest.mark.skipif(
    MOTEUR is None, reason="calibration engine absent (submodule empty, no neighbouring clone): "
                           "nothing to compare the repatriated formula with")


def tranche(texte: str, debut: str, fin: str) -> str:
    i = texte.index(debut)
    return texte[i:texte.index(fin, i)].rstrip() + "\n"


@needs_engine
def test_the_formula_is_the_engines_byte_for_byte():
    assert (COPIE / "metrics.py").read_bytes() == (MOTEUR / "metrics.py").read_bytes()


@needs_engine
def test_the_score_container_is_the_engines():
    modeles = (MOTEUR / "models.py").read_text(encoding="utf-8")
    debut = modeles.rindex("\n\n", 0, modeles.index("UNMEASURED_COMPOSITE = ")) + 2
    attendu = modeles[debut:modeles.index("# ── Configuration du run")].rstrip() + "\n"
    assert (COPIE / "models.py").read_text(encoding="utf-8").endswith(attendu)


@needs_engine
def test_the_reading_of_options_is_the_engines():
    evaluation = (MOTEUR / "evaluation.py").read_text(encoding="utf-8")
    copie = (COPIE / "evaluation.py").read_text(encoding="utf-8")
    for debut, fin in (("_OPTION_RE = ", "\n\ndef persona_agent_id"),
                       ("def parse_option_modes(", "\n\ndef inject_context"),
                       ("def option_modes_for_record(", "\n\ndef decisions_from_agents")):
        assert tranche(evaluation, debut, fin) in copie


def test_the_formula_imports_without_the_engine():
    from scripts.synthesis.sources import import_formule_score

    module, erreur = import_formule_score()
    assert erreur == ""
    assert module.metrics.categorize_mode("Bus") == "transports_collectifs"
    assert module.models.UNMEASURED_COMPOSITE > 1
    assert module.evaluation.parse_option_modes("- [1] walk: 12 min") == {1: "walk"}
    assert "calibration" not in module.metrics.__name__
