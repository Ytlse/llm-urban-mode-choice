"""Where paper writing lives: a separate private repository, cloned next to the working repo.

Ticket 115. The article, the GAMA Days abstract, the presentations and their figures have
left `docs/paper/`. Scripts that produce a figure or a page for the papers write into that
repository, designated by the `PAPER_DIR` variable, and failing that by the
`llm-agents-gama-papiers` folder next to the working repository.

The neighbour folder is computed from the MAIN repository, not from the current tree: in a
worktree (`.claude/worktrees/<name>/`), the tree's neighbour would be `.claude/worktrees/`.

Three functions, for two moments:

- `chemin_papier(*parties)` never raises: it serves module constants and argparse default
  values, which a test imports without having the papers repository;
- `sortie_papier(*parties)` does the same, but falls back to `outputs/figures/` when the papers
  repository is missing and `PAPER_DIR` is not set (public copy, ticket 113);
- `exiger_depot_papiers(*chemins)` is called just before writing. If one of the paths targets the
  papers repository and that repository is absent, it logs an [ALARME] and stops the script,
  instead of letting a `mkdir(parents=True)` build a fake repository that nobody will read.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("depot_papiers")

RACINE = Path(__file__).resolve().parents[1]
NOM_PAR_DEFAUT = "llm-agents-gama-papiers"
VARIABLE = "PAPER_DIR"
#: Output fallback when the papers repository is missing (public copy): see `sortie_papier`.
SORTIE_LOCALE = RACINE / "outputs" / "figures"


def racine_depot_principal(racine: Path = RACINE) -> Path:
    """The root of the main repository, even when `racine` is a worktree.

    In a worktree, `.git` is a FILE "gitdir: <main>/.git/worktrees/<name>".
    Any failed read returns `racine` unchanged: we do not guess further.
    """
    marque = racine / ".git"
    if not marque.is_file():
        return racine
    try:
        ligne = marque.read_text(encoding="utf-8").strip()
    except OSError:
        return racine
    if not ligne.startswith("gitdir:"):
        return racine
    gitdir = Path(ligne.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (racine / gitdir).resolve()
    # <main>/.git/worktrees/<name> → <main>
    if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
        return gitdir.parent.parent.parent
    return racine


def paper_dir() -> Path:
    """The papers repository: `PAPER_DIR` if set, else the neighbour of the main repository."""
    brut = os.environ.get(VARIABLE, "").strip()
    if brut:
        return Path(brut).expanduser().resolve()
    return (racine_depot_principal().parent / NOM_PAR_DEFAUT).resolve()


def chemin_papier(*parties: str) -> Path:
    """A path in the papers repository. Checks nothing: see `exiger_depot_papiers`."""
    return paper_dir().joinpath(*parties)


def sortie_papier(*parties: str) -> Path:
    """Where to write an output meant for the papers: the papers repository, or `outputs/figures/`.

    Ticket 113. In the public copy there is no papers repository: a figure is regenerated
    under `outputs/figures/<file name>` (or `outputs/figures/` for a folder), created by the
    script when it writes. The fallback applies only if `PAPER_DIR` is NOT set and the
    default neighbour is missing: a variable set to an absent folder keeps the [ALARME] of
    `exiger_depot_papiers`. Creates nothing (the function serves module constants).
    """
    if os.environ.get(VARIABLE, "").strip() or paper_dir().is_dir():
        return chemin_papier(*parties)
    nom = parties[-1] if parties and Path(parties[-1]).suffix else None
    local = SORTIE_LOCALE / nom if nom else SORTIE_LOCALE
    logger.warning("dépôt papiers absent (%s) : sortie repliée sur %s", paper_dir(), local)
    return local


def exiger_depot_papiers(*chemins: Path) -> None:
    """Stops the script if a path targets the papers repository and that repository is absent.

    A path outside the papers repository (`--sortie docs/synthesis/…`) always passes.
    """
    base = paper_dir()
    vises = [Path(c).resolve() for c in chemins if _sous(Path(c).resolve(), base)]
    if not vises:
        return
    if base.is_dir():
        logger.info("Dépôt papiers trouvé : %s (%d sortie(s) y écrivent)", base, len(vises))
        return
    defini = os.environ.get(VARIABLE)
    logger.error(
        "[ALARME] dépôt papiers introuvable : %s (%s=%s). Sorties visées : %s. Clonez le dépôt "
        "papiers à côté du dépôt de travail, définissez %s, ou passez une sortie explicite.",
        base, VARIABLE, defini if defini else "<non défini>",
        ", ".join(str(v) for v in vises), VARIABLE,
    )
    raise SystemExit(2)


def _sous(chemin: Path, base: Path) -> bool:
    try:
        chemin.relative_to(base)
    except ValueError:
        return False
    return True
