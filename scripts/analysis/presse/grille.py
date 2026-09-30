"""The grid of twenty signs, frozen before the first request — ticket 059, lot 2.

WHAT THIS MODULE GUARANTEES
---------------------------
A pre-registered prediction is worth something only if one can prove it preceded the
measurement. Three things prove it, and this module carries all three:

1. **Twenty cells, no more, no less.** Five articles by four modes. A missing cell
   is a prediction one did not have to write; an extra cell is a
   prediction added after the fact.
2. **One sign per cell, including no effect.** `signe: '0'` is a PREDICTION —
   "no net shift expected" — and a significant shift refutes it just as
   an inverted sign does. The equivalence is strict: `signe == '0'` if and only if
   `intensite == 0`. Without it, a cell could fall back on "no opinion" after the
   measurement, and the denominator of the sign rate would become negotiable.
3. **A fingerprint.** `Grille.empreinte` is copied at the head of every tally report.
   It is the only point that makes "pre-registered" checkable afterwards: a grid
   modified after a campaign shows.

The refusal is BLUNT. An invalid grid stops the load rather than letting a
tally run on nineteen cells and return a rate nobody will know
covered something other than what is announced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

SIGNES_ADMIS: tuple[str, ...] = ("+", "-", "0")
INTENSITES_ADMISES: tuple[int, ...] = (0, 1, 2, 3)

# Chapter 7 announces "twenty signed predictions" and sets the binomial bar at fifteen
# agreements. The count is therefore not a format convenience: it is quoted in the paper,
# and changing it requires recomputing that bar BEFORE the campaign.
CELLULES_ATTENDUES = 20


class RefusDeGrille(ValueError):
    """The grid does not load. The message gives the reason AND the action."""


@dataclass(frozen=True)
class Cellule:
    article: str
    mode: str
    signe: str
    intensite: int
    motif: str

    @property
    def sans_effet_attendu(self) -> bool:
        return self.signe == "0"

    def concorde(self, observe: str) -> bool:
        """Does the observed sign agree with the predicted sign?

        `observe` comes from the tally and is '+', '-' or '0'. The comparison is direct,
        including for '0': predicting no effect and observing it is an agreement, not
        an abstention.
        """
        if observe not in SIGNES_ADMIS:
            raise ValueError(f"observed sign out of domain: {observe!r} ({SIGNES_ADMIS})")
        return observe == self.signe


@dataclass(frozen=True)
class Grille:
    version: int
    gele_le: str
    source: str
    modes: tuple[str, ...]
    cellules: tuple[Cellule, ...]
    empreinte: str
    chemin: Path

    def de(self, article: str, mode: str) -> Cellule:
        for c in self.cellules:
            if c.article == article and c.mode == mode:
                return c
        raise KeyError(f"no cell for ({article}, {mode})")

    @property
    def articles(self) -> tuple[str, ...]:
        vus: list[str] = []
        for c in self.cellules:
            if c.article not in vus:
                vus.append(c.article)
        return tuple(vus)


def charger_grille(chemin: str | Path) -> Grille:
    chemin = Path(chemin)
    if not chemin.is_file():
        raise RefusDeGrille(
            f"sign grid not found: {chemin} → check the path, or freeze the grid "
            f"before any campaign (ticket 059, lot 2)"
        )
    brut = chemin.read_bytes()
    try:
        d = yaml.safe_load(brut.decode("utf-8")) or {}
    except yaml.YAMLError as err:
        raise RefusDeGrille(f"unreadable grid ({chemin}): {err}") from err

    modes = tuple(d.get("modes") or ())
    if not modes:
        raise RefusDeGrille(f"{chemin}: `modes` is empty → declare the modes of the grid")

    articles = d.get("articles") or {}
    cellules: list[Cellule] = []
    for nom_article, contenu in articles.items():
        declarees = (contenu or {}).get("cellules") or {}
        manquants = [m for m in modes if m not in declarees]
        if manquants:
            raise RefusDeGrille(
                f"{chemin}: article {nom_article!r} has no cell for {manquants} → "
                f"a missing cell is a prediction one did not have to write, add it "
                f"(sign '0' is a legitimate choice, absence is not)"
            )
        en_trop = [m for m in declarees if m not in modes]
        if en_trop:
            raise RefusDeGrille(
                f"{chemin}: article {nom_article!r} declares modes outside the grid {en_trop} → "
                f"remove them or add them to `modes`"
            )
        for mode in modes:
            c = declarees[mode] or {}
            signe = str(c.get("signe", ""))
            if signe not in SIGNES_ADMIS:
                raise RefusDeGrille(
                    f"{chemin}: ({nom_article}, {mode}) carries the sign {signe!r} → "
                    f"expected {SIGNES_ADMIS}"
                )
            intensite = c.get("intensite")
            if intensite not in INTENSITES_ADMISES:
                raise RefusDeGrille(
                    f"{chemin}: ({nom_article}, {mode}) carries intensity {intensite!r} → "
                    f"expected {INTENSITES_ADMISES}"
                )
            if (signe == "0") != (intensite == 0):
                raise RefusDeGrille(
                    f"{chemin}: ({nom_article}, {mode}) carries signe={signe!r} and "
                    f"intensite={intensite} → the equivalence is strict, « no expected effect » "
                    f"is written sign '0' AND intensity 0. A cell that declares a sign without "
                    f"intensity could fall back on « no opinion » after the measurement"
                )
            motif = str(c.get("motif") or "").strip()
            if not motif:
                raise RefusDeGrille(
                    f"{chemin}: ({nom_article}, {mode}) has no rationale → a prediction without "
                    f"a reason cannot be discussed"
                )
            cellules.append(Cellule(nom_article, mode, signe, int(intensite), motif))

    if len(cellules) != CELLULES_ATTENDUES:
        raise RefusDeGrille(
            f"{chemin}: {len(cellules)} cells instead of {CELLULES_ATTENDUES} → chapter 7 "
            f"announces twenty signed predictions and sets the binomial bar at fifteen "
            f"agreements. Changing this count requires recomputing that bar BEFORE the campaign, "
            f"never after"
        )

    return Grille(
        version=int(d.get("version", 0)),
        gele_le=str(d.get("gele_le", "")),
        source=str(d.get("source", "")),
        modes=modes,
        cellules=tuple(cellules),
        empreinte=hashlib.sha256(brut).hexdigest(),
        chemin=chemin,
    )
