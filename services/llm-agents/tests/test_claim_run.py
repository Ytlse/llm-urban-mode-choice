"""Importing `settings` reads the configuration; opening a run is an explicit act.

Until 2026-09-04, importing `settings` created a run directory and pointed
`experiments/current` at it. An analysis script started during a run therefore switched the
link to an empty directory, and the log of exchanges with the model — which follows the link on
every write — started writing beside the run. Four occurrences in one night, the first
having diverted 1,037 exchanges. These tests hold the boundary.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings as settings_module  # noqa: E402
from settings import FactorySettings  # noqa: E402


@pytest.fixture
def experiences(tmp_path, monkeypatch):
    """A fresh experiments folder, with a `current` pointing at an existing run."""
    exp = tmp_path / "experiments"
    (exp / "archive" / "run_en_cours").mkdir(parents=True)
    lien = exp / "current"
    lien.symlink_to(Path("archive") / "run_en_cours")
    monkeypatch.setenv("APP_EXPERIMENTS_DIR", str(exp))
    # The test safeguards neutralise precisely what we want to observe: we lift them,
    # and work in a temporary folder so that nothing real moves.
    monkeypatch.setattr(settings_module, "_run_artifacts_disabled", lambda: False)
    monkeypatch.setattr(FactorySettings, "_instance", None, raising=False)
    yield exp, lien
    FactorySettings._instance = None


def test_un_import_ne_cree_rien_et_ne_deplace_pas_le_lien(experiences):
    exp, lien = experiences
    avant = sorted(p.name for p in (exp / "archive").iterdir())

    s = FactorySettings.force_reload()

    assert not s.workdir.exists(), "an import must not create any run directory"
    assert sorted(p.name for p in (exp / "archive").iterdir()) == avant
    assert os.readlink(lien) == str(Path("archive") / "run_en_cours")


def test_claim_run_ouvre_le_run_et_deplace_le_lien(experiences):
    exp, lien = experiences
    s = FactorySettings.force_reload()
    assert not s.workdir.exists()

    FactorySettings.claim_run()

    assert s.workdir.is_dir(), "claim_run doit créer le répertoire du run"
    assert (s.workdir / "static_config.yaml").exists(), "the configuration must be frozen there"
    assert os.readlink(lien) == str(Path("archive") / s.workdir.name)


def test_claim_run_est_idempotente(experiences):
    exp, lien = experiences
    s = FactorySettings.force_reload()
    FactorySettings.claim_run()
    cible = os.readlink(lien)

    FactorySettings.claim_run()

    assert os.readlink(lien) == cible
    assert s.workdir.is_dir()


def test_sous_test_claim_run_ne_touche_a_rien(tmp_path, monkeypatch):
    """The existing safeguard still holds: a test suite that calls claim_run()
    — by mistake or through a transitive import — does not steal a running run's output."""
    exp = tmp_path / "experiments"
    (exp / "archive" / "run_en_cours").mkdir(parents=True)
    lien = exp / "current"
    lien.symlink_to(Path("archive") / "run_en_cours")
    monkeypatch.setenv("APP_EXPERIMENTS_DIR", str(exp))
    monkeypatch.setenv("APP_NO_RUN_ARTIFACTS", "1")
    FactorySettings._instance = None

    FactorySettings.claim_run()

    assert os.readlink(lien) == str(Path("archive") / "run_en_cours")
    FactorySettings._instance = None
