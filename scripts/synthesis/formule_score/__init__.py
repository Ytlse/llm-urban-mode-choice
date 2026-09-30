"""Scoring formula of the mode-choice simulation: gap of the simulated modal shares to the
EMC² 2023 survey (EMD–JSD composite, L1 composite and their dimensions).

Repatriated on 2026-09-29 into this repository from the engine it was developed in (commit
8be8f26 of that repository), so that scoring no longer depends on it. Only the measurement came:
``metrics`` (the formula, byte for byte), the score container (``models.Scores``,
``UNMEASURED_COMPOSITE``) and the reading of the options offered in a decision
(``evaluation``). Nothing that tunes a prompt. ``scripts/tests/test_formule_score_alignee.py``
fails if the engine's formula diverges from this copy.

Import it through ``scripts.synthesis.sources.import_formule_score()``.
"""

from . import evaluation, metrics, models  # noqa: F401
