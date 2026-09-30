"""Composite scoring formulas — versioned definition, canonical fingerprint.

Spec: `specs/scoring_composite_experiences.md` (R2, R3, R4, R7, R19, R20).

A formula weights **7 dimensions** (`DIMENSIONS`). `length_penalty` is not one
of them: it stays outside the composite score (R2), and the engine's ``Scorer``
sees it absent, hence with zero weight. The `formule_sha256` fingerprint is
**canonical** (R3): it depends neither on key order nor on how floats are
written, only on the values of the 7 dimensions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

# Canonical order of the weightable dimensions of the composite score (R2). Fixed: it
# defines the serialisation order of the fingerprint, so it must never change.
DIMENSIONS: tuple[str, ...] = (
    "global",
    "absent_penalty",
    "age",
    "occupation",
    "genre",
    "motif",
    "distance",
)

_REGISTRE_DEFAUT = Path(__file__).parent / "formules" / "reference.yaml"


def _poids_canoniques(poids: dict) -> dict:
    """Fills in the 7 dimensions (missing → 0.0) and forces the float type."""
    return {d: float(poids.get(d, 0.0)) for d in DIMENSIONS}


def empreinte(poids: dict) -> str:
    """Canonical SHA-256 of the weights (R3).

    Serialises the 7 dimensions in the fixed order of ``DIMENSIONS``, each weight
    normalised to a float via ``.12g``: ``0.5`` and ``0.50`` give the same
    string, the original key order plays no part.
    """
    canon = _poids_canoniques(poids)
    texte = ";".join(f"{d}={format(canon[d], '.12g')}" for d in DIMENSIONS)
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def _valider(nom: str, poids: dict) -> None:
    """Rejects an invalid formula (R19) — never evaluated as code."""
    inconnues = set(poids) - set(DIMENSIONS)
    if inconnues:
        raise ValueError(
            f"formula {nom!r}: dimension(s) outside the 7 admitted: {sorted(inconnues)}"
        )
    for dim, valeur in poids.items():
        try:
            f = float(valeur)
        except (TypeError, ValueError):
            raise ValueError(
                f"formula {nom!r}: non-numeric weight for {dim!r}: {valeur!r}"
            )
        if f < 0:
            raise ValueError(f"formula {nom!r}: negative weight for {dim!r}: {f}")


@dataclass(frozen=True)
class Formule:
    """A named formula: its 7 weights and its canonical fingerprint."""

    nom: str
    poids: dict
    sha256: str

    def poids_scorer(self) -> dict:
        """Weights to pass to the ``Scorer`` (defensive copy)."""
        return dict(self.poids)


class RegistreFormules:
    """The loaded registry: lookup by name, by SHA (R20), and reference formula."""

    def __init__(self, formules: list[Formule], reference_nom: str):
        self._par_nom = {f.nom: f for f in formules}
        # Last one wins on an identical SHA (two formulas with the same weights):
        # irrelevant, they carry the same definition.
        self._par_sha = {f.sha256: f for f in formules}
        self._reference_nom = reference_nom

    @property
    def reference(self) -> Formule:
        """The current reference formula (R4, R7)."""
        return self._par_nom[self._reference_nom]

    def par_nom(self, nom: str) -> Formule | None:
        return self._par_nom.get(nom)

    def resoudre(self, sha256: str) -> Formule | None:
        """Finds a formula by its fingerprint, including an old one (R20)."""
        return self._par_sha.get(sha256)

    def toutes(self) -> list[Formule]:
        return list(self._par_nom.values())


def charger(chemin: Path | str | None = None) -> RegistreFormules:
    """Loads the registry, validates it, and enforces a single reference (R4, R19).

    Loud refusal (exception) if: missing or duplicate name, unknown dimension,
    non-numeric or negative weight, or a number of ``reference: true`` other than 1.
    """
    chemin = Path(chemin) if chemin else _REGISTRE_DEFAUT
    data = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    entrees = data.get("formules") or []
    formules: list[Formule] = []
    references: list[str] = []
    noms: set[str] = set()
    for entree in entrees:
        nom = entree.get("nom")
        if not nom:
            raise ValueError(f"formula without a name in {chemin}")
        if nom in noms:
            raise ValueError(f"duplicate formula in {chemin}: {nom!r}")
        noms.add(nom)
        poids = entree.get("poids") or {}
        _valider(nom, poids)
        canon = _poids_canoniques(poids)
        formules.append(Formule(nom=nom, poids=canon, sha256=empreinte(canon)))
        if entree.get("reference"):
            references.append(nom)
    if len(references) != 1:
        raise ValueError(
            f"{chemin} must declare exactly one formula reference:true, "
            f"{len(references)} found: {references or '—'}"
        )
    return RegistreFormules(formules, references[0])


__all__ = ["DIMENSIONS", "Formule", "RegistreFormules", "charger", "empreinte"]
