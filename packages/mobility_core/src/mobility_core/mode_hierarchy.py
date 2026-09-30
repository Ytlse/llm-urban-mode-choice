"""
core/mode_hierarchy.py — The main mode of a trip, once for the whole repository.

A trip that mixes several modes receives **one** main mode. The repository carried
four tables and three answers for the same trip — a liO coach + TER was
`Transports_collectifs` in `moves.csv`, `train` in the split, `train` in the
worker counters and `transit` in the Prometheus metric (ticket 022, M1). This module
is **the only place** where the order is written, and it does not write it: it reads it from
`mobility_core/data/mode_hierarchy_emc2.json`, frozen by
`scripts/progedo_logit/export_mode_hierarchy.py`.

## WHERE THE ORDER COMES FROM

From the survey's published report: AUAT/CEREMA, « Enquête mobilité 2023 — bassin de vie
toulousain », appendix « Hiérarchie des modes », **p. 53**, which gives the 36 surveyed
modes in order, « défini au niveau national » (p. 12). It is **checked against the
microdata** — 53 code pairs settled by 2,607 observations, a single exception,
and it matches the appendix (a Flixbus, rank 12, loses to a TER, rank 8).

Mapped to the leg vocabulary produced by OTP, OSMnx and the school coach:

    1. metro   2. tram   3. cableway   4. bus   5. rail
    6. car     7. motorbike   8. bicycle   9. foot

## THE TWO SURPRISING NOTCHES, AND WHAT THEY CHANGE

**The bus comes before the train.** A "liO coach + TER" itinerary is a *bus* trip
for the survey: out of 35 settled mixed bus/coach ↔ train trips, 34 are coded
bus. The `move_logger` cascade was therefore right on this point; it was `mode_choice`
(train first) and `task_worker` (train first) that diverged.

**The car comes after all public transport** (rank 19, below ranks 1 to 13). Out of 770
trips mixing car and public transport, the survey codes 760 as public transport and
10 as car. `move_logger` tested the car **first**: that was the inversion.

## WHAT THIS MODULE DOES NOT DO

It does not answer the question "does this plan use the car?". A main mode and
a vehicle mode are two different quantities: the vehicle chain of ticket 008
asks "where is the car", not "what is the main mode". Confusing them produced
half of ticket 022; `simulation_controller._vehicle_mode` therefore keeps its own
reading, and it does not go through here.

Nor does it know the EMC² score categories (`marche` / `voiture` / `velo` /
`transports_collectifs`): that is an aggregation, not a hierarchy, and it lives in
`scripts/synthesis/frames.py` and `prompt_calibration/calibration/metrics.py`. This module
only gives them the means to check that they do not contradict the survey order.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_RESOURCE = Path(__file__).resolve().parent / "data" / "mode_hierarchy_emc2.json"

# Required resource version. An older (or hand-made) resource does not carry the
# microdata check: we refuse rather than serve an order that was not cross-checked.
REQUIRED_VERSION = "mh1"

# Families expected in the resource. One family missing and the mode it carries
# would silently fall into "unknown" — the vacuity pattern the repository hunts down.
REQUIRED_FAMILIES = ("metro", "tram", "cableway", "bus", "rail", "car", "motorbike",
                     "bicycle", "foot")


@dataclass(frozen=True)
class ModeHierarchy:
    """The loaded hierarchy: ranks, labels, and the two vocabularies it serves.

    `families` is the published order, from highest to lowest priority.
    `leg_rank` maps a leg mode (`"bus"`, `"rail"`, `"__car__"`…) to its rank.
    `journal_label` and `canonical_mode` translate a family into the two vocabularies
    of the repository: the « Mode de transport Choisi » column of `moves.csv` and
    `mode_choice.CANONICAL_MODES`.
    """

    families: tuple[str, ...]
    family_rank: Mapping[str, int]
    leg_rank: Mapping[str, int]
    journal_label: Mapping[str, str]
    canonical_mode: Mapping[str, str]
    legs_by_family: Mapping[str, frozenset[str]]
    meta: dict = field(default_factory=dict)

    # ── Loading ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, resource: Path | None = None) -> ModeHierarchy:
        """Loads the frozen resource (the module's only I/O point).

        Missing or unexpected version = explicit error. A fallback to an order hard-coded
        here would bring back the literal this module exists to remove.
        """
        path = Path(resource) if resource else DEFAULT_RESOURCE
        if not path.exists():
            raise FileNotFoundError(
                f"Mode hierarchy missing: {path}. Produce it with "
                "`python -m scripts.progedo_logit.export_mode_hierarchy` (ProGEDO "
                "lil-1750 microdata, restricted access) — the resource is normally "
                "versioned in the repository.")
        doc = json.loads(path.read_text(encoding="utf-8"))
        version = str(doc.get("version") or "")
        if version != REQUIRED_VERSION:
            raise ValueError(
                f"Hierarchy {path} is at version {version!r}, but the module requires "
                f"{REQUIRED_VERSION!r}. Re-export it.")
        families = tuple(doc.get("ordre_familles") or ())
        manquantes = [f for f in REQUIRED_FAMILIES if f not in families]
        if manquantes:
            raise ValueError(
                f"Hierarchy {path} lacks the families {manquantes}: the modes they "
                "carry would be classed as \"unknown\" without anything reporting it.")
        leg_rank = {str(k): int(v) for k, v in (doc.get("rang_jambe") or {}).items()}
        if not leg_rank:
            raise ValueError(f"Hierarchy {path} has no leg mode at all: it would "
                             "classify nothing.")
        return cls(
            families=families,
            family_rank={str(k): int(v) for k, v in (doc.get("rang_famille") or {}).items()},
            leg_rank=leg_rank,
            journal_label=dict(doc.get("libelle_journal") or {}),
            canonical_mode=dict(doc.get("mode_canonique") or {}),
            legs_by_family={k: frozenset(v)
                            for k, v in (doc.get("jambes_par_famille") or {}).items()},
            meta={"version": version, "titre": doc.get("titre"),
                  "source_publiee": {k: v for k, v in
                                     (doc.get("source_publiee") or {}).items()
                                     if k in ("rapport", "url", "page")},
                  "provenance": doc.get("provenance") or {}},
        )

    # ── Reading ──────────────────────────────────────────────────────────────────

    def family_of(self, leg_mode: object) -> str | None:
        """Family of a leg mode. `None` for a mode the hierarchy does not know.

        `None` is an answer, not a default: an unknown mode must be **counted** and
        reported by the caller, never filed by default in the catch-all next door.
        """
        mode = str(leg_mode or "").strip().lower()
        if not mode:
            return None
        rank = self.leg_rank.get(mode)
        if rank is None:
            return None
        return self.families[rank - 1]

    def primary_family(self, leg_modes: Iterable[object]) -> str | None:
        """Family of the **main mode** of a set of leg modes.

        It is the best-ranked family present. `None` when no mode is recognised
        (empty set, or all modes unknown) — up to the caller to count it.
        """
        best: int | None = None
        for leg_mode in leg_modes:
            rank = self.leg_rank.get(str(leg_mode or "").strip().lower())
            if rank is not None and (best is None or rank < best):
                best = rank
        return None if best is None else self.families[best - 1]

    def primary_label(self, leg_modes: Iterable[object]) -> str | None:
        """« Mode de transport Choisi » label of the main mode, or `None`."""
        family = self.primary_family(leg_modes)
        return None if family is None else self.journal_label.get(family)

    def primary_canonical(self, leg_modes: Iterable[object]) -> str | None:
        """Canonical mode (`mode_choice.CANONICAL_MODES`) of the main mode, or `None`."""
        family = self.primary_family(leg_modes)
        return None if family is None else self.canonical_mode.get(family)

    def canonical_order(self) -> tuple[str, ...]:
        """Canonical modes in hierarchy order, without duplicates.

        It is the order the `mode_choice._MODE_KEYWORDS` cascade must follow:
        `public_transport` (metro, rank 1) before `train` (rank 5) before `car` (rank 6)…
        """
        ordre: list[str] = []
        for family in self.families:
            canonical = self.canonical_mode.get(family)
            if canonical and canonical not in ordre:
                ordre.append(canonical)
        return tuple(ordre)

    def label_order(self) -> tuple[str, ...]:
        """Log labels in hierarchy order, without duplicates."""
        ordre: list[str] = []
        for family in self.families:
            label = self.journal_label.get(family)
            if label and label not in ordre:
                ordre.append(label)
        return tuple(ordre)


@lru_cache(maxsize=1)
def hierarchy() -> ModeHierarchy:
    """The repository hierarchy, loaded once. Entry point for all callers."""
    return ModeHierarchy.load()


# ── Shortcuts, so that callers do not have to carry the object ───────────────────

def leg_family(leg_mode: object) -> str | None:
    return hierarchy().family_of(leg_mode)


def primary_family(leg_modes: Iterable[object]) -> str | None:
    return hierarchy().primary_family(leg_modes)


def primary_label(leg_modes: Iterable[object]) -> str | None:
    return hierarchy().primary_label(leg_modes)


def primary_canonical(leg_modes: Iterable[object]) -> str | None:
    return hierarchy().primary_canonical(leg_modes)


def family_rank(family: str) -> int | None:
    return hierarchy().family_rank.get(family)
