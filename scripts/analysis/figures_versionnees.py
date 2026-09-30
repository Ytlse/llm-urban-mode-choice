"""Warns when a regenerated figure overwrites a file TRACKED BY GIT.

Ticket 039, step 6. The paper's figures live in `figures/` of the papers repository
(`PAPER_DIR`, ticket 115; it was `docs/paper/figures/` before), which is versioned — on purpose: a frozen chapter must be able to show the figure it cites.
But `plot_experiences.py` and `plot_familles.py` write there BY DEFAULT. Each
committed regeneration thus adds one more blob to the pack, permanently, while
only one version is ever of interest: the latest.

The repository measured that the problem is NOT the installed weight — the 34 MB tracked
under `docs/paper/` each have exactly ONE version in the history, which git stores
optimally. The problem is future growth, and it has already started (the four
`familles_composite_l1_*` date from 2026-09-14, the two `comparaison_experiences_*`
from 2026-09-09). Hence: no LFS migration, which would save nothing and would cost a
dependency on every clone — but a regeneration that ANNOUNCES what it has just created.

This module forbids nothing and never fails: it makes visible, at the moment it
happens, a cost that is otherwise only seen in `git status`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("figures")

RACINE = Path(__file__).resolve().parents[2]


def _suivi_par_git(chemin: Path) -> bool:
    """Is the file versioned? Any git failure answers "no" — we only raise an alarm when certain."""
    # git runs in the figure's folder: since ticket 115, the store lives in the papers
    # repository, and the working repository would wrongly answer "not tracked".
    dossier = chemin.parent if chemin.parent.is_dir() else RACINE
    try:
        # fixed argv, path passed after `--`: no path can be read as an option.
        # check=False: the return code IS the answer (1 = not tracked), not an error.
        issue = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(chemin)],
            cwd=dossier, capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return issue.returncode == 0


def signaler(ecrits: list[Path]) -> None:
    """Logs the written figures, and singles out those that will cost one more blob."""
    versionnees = [p for p in ecrits if _suivi_par_git(p)]
    for chemin in ecrits:
        try:
            relatif = chemin.relative_to(RACINE)
        except ValueError:
            relatif = chemin
        taille = chemin.stat().st_size / 1024 if chemin.is_file() else 0
        logger.info("Figure écrite : %s (%.0f Ko)", relatif, taille)

    if not versionnees:
        return
    poids = sum(p.stat().st_size for p in versionnees if p.is_file()) / 1024
    logger.warning(
        "%d figure(s) suivie(s) par git viennent d'être écrasées (%.0f Ko). "
        "Un `git commit` les ajoutera au pack DÉFINITIVEMENT, en plus des versions "
        "précédentes. Ne committez une figure que si un chapitre la cite dans son état "
        "gelé ; pour une exploration, écrivez ailleurs : --sortie docs/synthesis/<nom>",
        len(versionnees), poids,
    )
