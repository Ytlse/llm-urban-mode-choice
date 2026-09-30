"""The cold-archive guard — a single place that says what "archived" means (ticket 074, A-4).

    Cold = restorable and auditable, never used nor referenced.

Not to be confused with the platform's `archivee` status (`experiences.statut`), which only
changes **visibility**: the files stay in place and remain readable. Here, the
path itself is out of service.

The guard bears on the **LOCATION**, never on a text match in the name: a
cohort that would be called "population_archivistes" stays perfectly readable. It is enough
for one segment of the path to be named `archive` for the read to be refused.

Why a refusal in the code and not a sentence in a README: the platform's 36 runs
all read the v1 cohort when the article's reference was v5, and nothing
stood in the way (ticket 045). A safeguard that only exists in prose never fires.

The override requires a **REASON**, not a boolean: `confirme=True` is ticked without thinking,
`confirme="garde de comparabilité D-7, ticket 074"` is written, logged and read back.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

# Path segment that marks content withdrawn from service.
SEGMENT_ARCHIVE = "archive"


class ContenuArchive(ValueError):
    """Read refused: the content has been withdrawn from service (cold archive)."""


def sous_archive(chemin: str | Path) -> bool:
    """Does the path go through an `archive` folder?"""
    return SEGMENT_ARCHIVE in Path(chemin).resolve().parts


def verifier(
    chemin: str | Path,
    confirme: str | None,
    *,
    quoi: str,
    comment_lever: str,
    repli: str = "",
) -> None:
    """Refuse `chemin` if under archive, unless a reason is given explicitly — and log the override.

    - `quoi` names the nature of the content ("a population", "a sealed set"), article included:
      the sentence is built around it, with no agreement to guess;
    - `comment_lever` says CONCRETELY how to bypass it — the field to write, not "provide a
      reason". A refusal that does not say what to do gets worked around by guesswork, or endured;
    - `repli` names the current reference, when there is one.
    """
    if not sous_archive(chemin):
        return
    motif = confirme.strip() if isinstance(confirme, str) else ""
    if not motif:
        raise ContenuArchive(
            f"read refused — {quoi} in cold archive: {chemin}\n"
            f"  Content under `{SEGMENT_ARCHIVE}/` can be restored and audited, but it is not "
            f"replayed and not referenced."
            + (f"\n  The current reference is `{repli}`." if repli else "")
            + f"\n  To read it anyway: {comment_lever}. The reason is logged and must "
            "be recorded in the ticket that requests it — a boolean is ticked without thinking, "
            "a reason is written and reread."
        )
    logger.warning(
        f"[froid] OVERRIDE: cold-archive read of {chemin} — reason: {motif!r}. "
        f"This read grounds no measurement comparable to those of the current reference."
    )


__all__ = ["SEGMENT_ARCHIVE", "ContenuArchive", "sous_archive", "verifier"]
