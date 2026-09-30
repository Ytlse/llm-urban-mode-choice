"""A stale waiting marker does not pass off a dead launcher as a clean finish (2026-09-28).

A resumed arm replays in the SAME directory as the one that had been suspended: the marker
`en_attente_quota.json` from the previous suspension is still there. `attendre_fin_du_run`
took it for a clean finish, and a resumed arm whose launcher died along the way was counted
as completed. Only a marker written since THIS arm was launched counts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment import run_sequential_cohort as C  # noqa: E402


def _archive(tmp_path: Path, age_marqueur_s: float) -> Path:
    archive = tmp_path / "2026-09-28_01_00"
    archive.mkdir()
    marqueur = archive / "en_attente_quota.json"
    marqueur.write_text(json.dumps({"motif": "surcharge_fournisseur", "jour_simule": 14}))
    passe = time.time() - age_marqueur_s
    os.utime(marqueur, (passe, passe))
    return archive


def test_un_marqueur_de_la_suspension_precedente_ne_vaut_pas_fin_propre(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "lanceur_en_cours", lambda: False)
    archive = _archive(tmp_path, age_marqueur_s=3600)
    assert C.attendre_fin_du_run("p", "control", archive, lance_a=time.time()) is False


def test_un_marqueur_ecrit_pendant_ce_bras_vaut_fin_propre(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "lanceur_en_cours", lambda: False)
    lance_a = time.time() - 600
    archive = _archive(tmp_path, age_marqueur_s=60)
    assert C.attendre_fin_du_run("p", "control", archive, lance_a=lance_a) is True
