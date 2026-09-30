"""`make run` starts only ONE controller when config.yaml has changed (2026-09-24).

The comparison with `.config.yaml.applique` happened AFTER `make up`: a fresh controller
started, then was recreated within the same minute. Both opened the same run
directory, the second kept the identity of the first, and the cohort refused the arm.
"""

import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]


def _recette_run() -> list[str]:
    sortie = subprocess.run(
        ["make", "-n", "run", "OFFLINE=1", "CACHE=0", "CHOC=0"],
        cwd=RACINE, capture_output=True, text=True, check=False,
    ).stdout
    return sortie.splitlines()


def _indice(lignes, motif):
    return next(i for i, l in enumerate(lignes) if motif in l)


def test_la_config_se_compare_avant_de_demarrer_les_services():
    lignes = _recette_run()
    comparaison = _indice(lignes, "cmp -s")
    demarrage = next(i for i, l in enumerate(lignes) if l.rstrip().endswith("make up"))
    assert comparaison < demarrage


def test_plus_aucune_recreation_apres_le_demarrage():
    assert not any("--force-recreate" in l and "controller" in l for l in _recette_run())
