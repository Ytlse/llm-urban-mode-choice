"""`make run` aligns the GAMA common-prefix lock on the controller it recreates (2026-09-28).

A manual run launched after a memory A/B found `prefixe_commun: true` in sim_params.yaml,
facing a controller at `false`: /init refused it (409). The A/B arms also go through
`make run` (run_sequential_cohort): the value is therefore not forced to `false`, it follows
WORLD__PREFIXE_COMMUN, which the orchestrator passes on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make missing")


def _commandes(tmp_path: Path, env_prefixe: str | None, *args: str) -> list[str]:
    """The lines of `make -n run` that touch the lock, for a throwaway sim_params.yaml."""
    env = {k: v for k, v in os.environ.items() if k != "WORLD__PREFIXE_COMMUN"}
    if env_prefixe is not None:
        env["WORLD__PREFIXE_COMMUN"] = env_prefixe
    sortie = subprocess.run(
        ["make", "-n", "run", f"SIM_PARAMS={tmp_path / 'sim_params.yaml'}", *args],
        cwd=RACINE, env=env, capture_output=True, text=True, check=False,
    ).stdout
    return [l for l in sortie.splitlines() if l.startswith("perl") and "prefixe_commun" in l]


def _appliquer(tmp_path: Path, contenu: str, env_prefixe: str | None, *args: str) -> str:
    fichier = tmp_path / "sim_params.yaml"
    fichier.write_text(contenu, encoding="utf-8")
    for commande in _commandes(tmp_path, env_prefixe, *args):
        subprocess.run(["bash", "-c", commande], check=True)
    return fichier.read_text(encoding="utf-8")


def test_un_run_a_la_main_remet_le_verrou_a_false(tmp_path):
    avant = "population_size: 6\nprefixe_commun: true\nsimulation_max_days: 25\n"
    apres = _appliquer(tmp_path, avant, None)
    assert apres == "population_size: 6\nprefixe_commun: false\nsimulation_max_days: 25\n"


def test_un_bras_ab_garde_le_verrou_que_l_orchestrateur_transmet(tmp_path):
    avant = "population_size: 6\nprefixe_commun: false\n"
    assert "prefixe_commun: true\n" in _appliquer(tmp_path, avant, "true")


def test_un_fichier_sans_la_cle_la_recoit(tmp_path):
    apres = _appliquer(tmp_path, "population_size: 6\n", None)
    assert apres.count("prefixe_commun:") == 1
    assert "prefixe_commun: false" in apres


def test_la_reprise_a_chaud_n_y_touche_pas(tmp_path):
    """With CONT=1 the controller is not recreated: it keeps the value of the resumed run."""
    assert _commandes(tmp_path, None, "CONT=1") == []
    avant = "prefixe_commun: true\n"
    assert _appliquer(tmp_path, avant, None, "CONT=1") == avant
