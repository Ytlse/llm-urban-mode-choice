"""The night chain recognises a suspended arm and retries it (2026-09-25).

On 25/09 at 18:52, the treated arm of a13 got suspended on an HTTP 503 episode (code 7). The
chain classified it as "failed": it went through `make`, which returns 2 for any failed
recipe, and code 7 never reached it. These tests play the real script with a
fake orchestrator, which suspends first and then finishes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "experiment" / "enchainer_experiences_memoire.sh"
EXP = "exp_mem_test_suspension"

FAUX_ORCHESTRATEUR = """
import json, sys
from pathlib import Path
racine = Path.cwd()
compteur = racine / "appels.txt"
n = int(compteur.read_text()) + 1 if compteur.exists() else 1
compteur.write_text(str(n))
codes = [int(c) for c in (racine / "codes.txt").read_text().split()]
code = codes[min(n, len(codes)) - 1]
if code == 7:
    courant = racine / "experiments" / "current"
    courant.mkdir(parents=True, exist_ok=True)
    (courant / "en_attente_quota.json").write_text(json.dumps(
        {"motif": (racine / "motif.txt").read_text().strip(), "resume_at": "2026-09-25T16:52:47",
         "jour_simule": 4}))
print("argv", sys.argv[1:])
sys.exit(code)
"""


def _jouer(tmp_path: Path, codes: list[int], motif: str = "surcharge_fournisseur",
           file: list[str] | None = None) -> str:
    (tmp_path / "data" / "experiences_memoire" / EXP).mkdir(parents=True)
    (tmp_path / "data" / "experiences_memoire" / EXP / "experience_memoire.yaml").write_text("nom: x\n")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "codes.txt").write_text(" ".join(map(str, codes)))
    (tmp_path / "motif.txt").write_text(motif)
    faux = tmp_path / "faux_orchestrateur.py"
    faux.write_text(FAUX_ORCHESTRATEUR)
    # A fake `pgrep`: the script refuses to start if a real campaign runs on the machine,
    # and these tests must neither hinder it nor depend on it (they were skipped during a run).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pgrep").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "pgrep").chmod(0o755)
    env = {**os.environ, "RACINE": str(tmp_path), "PYTHON": sys.executable,
           "ORCHESTRATEUR": str(faux), "ATTENTE_S": "0", "ESSAIS_MAX": "3",
           "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run(["bash", str(SCRIPT), *([EXP] if file is None else file)], env=env, check=True,
                   capture_output=True, timeout=60)
    (journal,) = (tmp_path / "experiments").glob("enchainement_nuit_*.log")
    return journal.read_text(encoding="utf-8")


def test_un_bras_suspendu_par_un_503_est_reessaye_puis_termine(tmp_path):
    journal = _jouer(tmp_path, [7, 0])
    assert "suspendue" in journal and "motif=surcharge_fournisseur" in journal
    assert "ÉCHEC" not in journal
    assert "TERMINÉE" in journal
    assert (tmp_path / "appels.txt").read_text() == "2", "a single new attempt"


def test_l_orchestrateur_recoit_le_nom_de_l_experience(tmp_path):
    _jouer(tmp_path, [0])
    detail = next((tmp_path / "experiments").glob("enchainement_nuit_*.detail.txt"))
    assert f"'--experience', '{EXP}'" in detail.read_text(encoding="utf-8")


def test_un_quota_epuise_passe_a_la_suite_sans_reessayer(tmp_path):
    journal = _jouer(tmp_path, [7, 0], motif="quota_journalier")
    assert "quota du jour épuisé" in journal
    assert (tmp_path / "appels.txt").read_text() == "1"


def test_un_vrai_echec_reste_un_echec(tmp_path):
    journal = _jouer(tmp_path, [1])
    assert "ÉCHEC (code 1)" in journal


def test_le_script_n_appelle_plus_make():
    """`make` returns 2 for any failed recipe: code 7 would get lost there."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "make experience-memoire-lancer EXP" not in source.split("set -u", 1)[1]


def test_une_experience_non_conforme_n_est_ni_terminee_ni_reessayee(tmp_path):
    """Code 8 (2026-09-28): both arms ran, the A/B prefix is not demonstrated."""
    journal = _jouer(tmp_path, [8, 0])
    assert "NON CONFORME" in journal
    assert "TERMINÉE" not in journal
    assert (tmp_path / "appels.txt").read_text() == "1"


def test_la_file_par_defaut_ecarte_les_experiences_jugees_ou_abandonnees(tmp_path):
    """Rerunning v3, v3r1 or v4 would replay their treated arm from scratch, on paid calls."""
    racine = tmp_path / "data" / "experiences" / "evenements_non_tabules" / "articles_presse" / "A13"
    for nom, etat in (("exp_jugee", "non_conforme"), ("exp_abandonnee", "interrompue"),
                      ("exp_ratee", "echec"), ("exp_suspendue", "suspendue")):
        (racine / nom).mkdir(parents=True)
        (racine / nom / "experience_memoire.yaml").write_text("nom: x\n")
        (racine / nom / "etat.json").write_text(json.dumps({"etat": etat}))
    journal = _jouer(tmp_path, [0], file=[])
    detail = next((tmp_path / "experiments").glob("enchainement_nuit_*.detail.txt")).read_text()
    assert "'--experience', 'exp_suspendue'" in detail
    for nom in ("exp_jugee", "exp_abandonnee", "exp_ratee"):
        assert f"'{nom}'" not in detail
        assert nom in journal, "the excluded one is named in the log"
