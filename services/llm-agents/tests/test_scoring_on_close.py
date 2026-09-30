"""Automatic scoring when an execution closes.

Substrate: a real terminated run from the
repository, copied to tmp so that writes do not pollute `data/`.

The contract held here is fail-open: closing-time scoring is an output, never
a reason to fail a run that produced all of its decisions.
"""

import json
import shutil
from pathlib import Path

import pytest
from experiences import score as S
from loguru import logger

from tests.test_composite_scoring import EXEC_REELLE

pytestmark = pytest.mark.skipif(
    EXEC_REELLE is None,
    reason="no terminated run in data/experiences (substrate missing)",
)


@pytest.fixture
def exec_tmp(tmp_path):
    dst = tmp_path / EXEC_REELLE.name
    shutil.copytree(EXEC_REELLE, dst)
    for f in ("scores.json", "synthese_scores.html"):
        (dst / f).unlink(missing_ok=True)
    return dst


def test_R23_execution_terminee_scoree_a_la_cloture(exec_tmp):
    # Without being asked: no more `score --toutes` to run by hand to see a composite score.
    chemin = S.scorer_a_la_cloture(exec_tmp)
    assert chemin is not None and chemin.exists()
    contenu = json.loads(chemin.read_text(encoding="utf-8"))
    assert contenu["composite"]["emd_jsd"] is not None
    assert (exec_tmp / "synthese_scores.html").exists(), "the detail page must follow"


@pytest.mark.parametrize("etat", ["arretee", "en_pause", "en_cours", "epuisee"])
def test_R21_seule_une_execution_terminee_est_scoree(exec_tmp, etat):
    # Settled: a stopped run is NOT taken into account, even if largely filled.
    synthese = json.loads((exec_tmp / "synthese.json").read_text(encoding="utf-8"))
    synthese["etat"]["etat"] = etat
    (exec_tmp / "synthese.json").write_text(
        json.dumps(synthese, ensure_ascii=False), encoding="utf-8"
    )
    assert S.scorer_a_la_cloture(exec_tmp) is None
    assert not (exec_tmp / "scores.json").exists()


def test_R23_fail_open_moteur_absent(exec_tmp, monkeypatch):
    # Scoring formula unavailable: no exception, no scores.json, a readable ALARM.
    monkeypatch.setattr(
        S.sources, "import_formule_score", lambda *a, **k: (None, "moteur absent (test)")
    )
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        assert S.scorer_a_la_cloture(exec_tmp) is None  # does not raise
    finally:
        logger.remove(sink)
    assert not (exec_tmp / "scores.json").exists()
    assert any("[ALARME]" in m and exec_tmp.name in m for m in messages), messages


def test_R23_fail_open_erreur_inattendue(exec_tmp, monkeypatch):
    # Any failure, not only the missing engine: the run stays terminated.
    def boum(*a, **k):
        raise RuntimeError("disque plein")

    monkeypatch.setattr(S, "score_execution", boum)
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        assert S.scorer_a_la_cloture(exec_tmp) is None
    finally:
        logger.remove(sink)
    etat = json.loads((exec_tmp / "synthese.json").read_text(encoding="utf-8"))["etat"]
    assert etat["etat"] == "terminee", "scoring never changes the state of the run"
    assert any("[ALARME]" in m and "disque plein" in m for m in messages), messages
