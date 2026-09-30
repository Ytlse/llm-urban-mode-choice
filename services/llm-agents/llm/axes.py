"""Axes of a memory — ticket 071, lot 2.

A memory does not carry text alone: it carries **four typed axes** — the object, the place, the
time slot, the purpose — plus the day's weather. These axes serve two purposes:

- the **per-object pool** of recall: for each mode offered in the options of a decision,
  fetch directly the memories that carry this mode, without going through the embedding;
- the **affinity** in the ranking, as a BONUS and never as a veto.

**Normalisation happens at WRITE time, never at read time.** Without it, two spellings of the
same line never meet; and normalising at recall would redo the same work at every
decision, on the critical path.

An unresolved axis is `None`, and `None` matches **nothing** — not even another
`None`. Two unknowns do not make a match: this is what prevents the absence of measurement
from producing resemblance.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger


def _hierarchie():
    """Mode hierarchy, loaded on FIRST use and not at import.

    ⚠ This is not a matter of style. Loaded at import, it freezes the repository
    configuration at the moment this module enters the import tree — which happened before
    the tests had set theirs, and made seven scope and ring tests fail
    with no relation to memory. A utility module must freeze nothing on
    arrival.
    """
    global _HIERARCHIE
    if _HIERARCHIE is None:
        from mobility_core.mode_hierarchy import hierarchy

        _HIERARCHIE = hierarchy()
    return _HIERARCHIE


_HIERARCHIE = None

# Counter of the modes the hierarchy ignores. An unknown mode must be COUNTED and reported,
# never filed by default into the neighbouring catch-all — this is the rule of `family_of`, and a
# wrong modal share is more costly than a missing modal share (ticket 022).
MODES_INCONNUS: dict[str, int] = {}


# Value of the `mode` field by which the `stm_reflection` schema says "this concept is NOT
# about a mode". It is not an unknown mode: counting it as such would drown the counter
# of real anomalies under a perfectly legitimate value.
_SANS_MODE = frozenset({"any"})


def _modes_canoniques() -> frozenset[str]:
    """The canonical modes the hierarchy can produce, DERIVED from the frozen resource.

    Never hard-coded here: `canonical_order()` gets them from the same JSON as the ranks, and a
    list copied into this module would diverge from the hierarchy the day it moves, without
    anything saying so.
    """
    return frozenset(_hierarchie().canonical_order())


def mode_canonique(mode_label: str | None) -> str | None:
    """Canonical mode of the MAIN MODE of a trip, from its leg label.

    "foot,bus,foot" returns `public_transport`, not `walking`: it is the best-ranked family
    present that designates the trip, according to the AUAT/CEREMA hierarchy that is authoritative
    in the repository. A cascade written here would duplicate a sixth one, and an incomplete list
    returns a plausible and wrong figure.

    **Two vocabularies come in here, and it is deliberate** (ticket 077, lot A). Itinerary
    decisions carry LEG labels (`foot`, `bus`, `bicycle`); the JSON schema of
    `stm_reflection` imposes the CANONICAL modes on the model (`walking`, `cycling`,
    `public_transport`…). Until 2026-09-15 only the first family got through: of the seven
    values the schema allows, `car` was the only one that was also a leg label, and
    the other six returned `None`. Consequence measured on the thirty-day run of 075 —
    `axe_objet` empty on 211 concepts out of 231, degenerate baskets, `known_beliefs` empty in 68 %
    of reflection calls, and the concept correction mechanism of 071 (lot 3) never
    executed: 0 refinements and 1 contradiction for 231 creations.

    A mode that is already canonical is therefore **its own canonical form**. Recognition uses
    the set derived from the frozen resource, not a list written here.

    ⚠ This function READS the mode label, it does not replace it. `parse_option_modes`
    rereads these labels in the text of the prompt: changing them would break the calibration and the
    modal shares. The present addition changes none of them — it ACCEPTS one more vocabulary.
    """
    if not mode_label:
        return None
    jambes = [j.strip() for j in str(mode_label).split(",") if j.strip()]
    if not jambes:
        return None

    # Single label: it may be an already canonical mode. Tested BEFORE the hierarchy, which
    # only knows legs and would return `None` on « walking » as on a made-up word.
    if len(jambes) == 1:
        seul = jambes[0].lower()
        if seul in _SANS_MODE:
            return None
        if seul in _modes_canoniques():
            return seul

    canonique = _hierarchie().primary_canonical(jambes)
    if canonique is None:
        clef = ",".join(jambes)
        MODES_INCONNUS[clef] = MODES_INCONNUS.get(clef, 0) + 1
        if MODES_INCONNUS[clef] == 1:
            logger.warning(
                f"[axes] mode unknown to the hierarchy: « {clef} » — axe_objet left empty "
                f"rather than filed into a catch-all"
            )
    return canonique


def normaliser_lieu(texte: str | None) -> str | None:
    """Stop, line or neighbourhood, reduced to a single form.

    "Line: 401", "line:401" and "LINE : 401" designate the same line: without this
    normalisation, three memories of the same line never meet.
    """
    if texte is None:
        return None
    brut = str(texte).strip().lower()
    for prefixe in ("line:", "line :", "ligne:", "ligne :", "stop:", "stop :"):
        if brut.startswith(prefixe):
            brut = brut[len(prefixe):]
            break
    brut = " ".join(brut.replace(":", " ").replace("_", " ").split())
    return brut or None


def creneau_de(quand: datetime | None) -> str | None:
    """Four time slots, on the hour bands ALREADY in use in the repository.

    Same bounds as `categorize_date_time_short`: 6-12, 12-18, 18-24, the rest being night.
    Choosing others would be enough to make two setups incomparable.
    """
    if quand is None:
        return None
    heure = quand.hour
    if 6 <= heure < 12:
        return "matin"
    if 12 <= heure < 18:
        return "midi"
    if 18 <= heure < 24:
        return "soir"
    return "nuit"


def normaliser_motif(purpose: str | None) -> str | None:
    """Trip purpose, in the vocabulary of activities."""
    if not purpose:
        return None
    return str(purpose).strip().lower() or None


# Weather cascade, in this order: what changes a mode decision the most comes first.
# Snow beats rain, rain beats temperature. A hot AND rainy day is classified
# `pluie`: it is the rain that makes one give up the bike, not the degrees.
SEUIL_PLUIE_MM = 0.2
SEUIL_CANICULE_C = 30.0
SEUIL_FROID_C = 0.0
_CODES_NEIGE = frozenset({71, 73, 75, 77, 85, 86})


def meteo_de(w: dict | None) -> str | None:
    """Weather axis of a memory, in five values: snow, rain, heatwave, cold, dry.

    Five values and no more: the axis serves to MATCH two days, not to describe the
    weather. A finer vocabulary would mean that two similar days never meet,
    which is exactly the defect the axes correct.
    """
    if not w:
        return None
    try:
        code = int(w.get("weather_code") or -1)
        precip = float(w.get("precip_mm") or 0.0)
        temp = float(w.get("temperature"))
    except (TypeError, ValueError):
        return None
    if code in _CODES_NEIGE:
        return "neige"
    if precip >= SEUIL_PLUIE_MM:
        return "pluie"
    if temp >= SEUIL_CANICULE_C:
        return "canicule"
    if temp <= SEUIL_FROID_C:
        return "froid"
    return "sec"


# ── Affinities ───────────────────────────────────────────────────────────────────

POIDS_AXE_OBJET = 0.50
POIDS_AXE_LIEU = 0.20
POIDS_AXE_CRENEAU = 0.15
POIDS_AXE_MOTIF = 0.15


def _concorde(a: str | None, b) -> bool:
    """Does axis `a` of the memory match what the current context carries?

    `None` matches nothing, not even `None`. Matching two unknowns
    would produce resemblance out of an absence of measurement — a recurring pattern to track
    in this project, where the absence of measurement readily produces the perfect score.

    The current side may be a single value or a SET. An itinerary decision offers
    several modes: a bike memory is relevant as soon as the bike is among the options,
    and requiring equality with a single "current" mode would make no sense — there is not one.
    """
    if a is None or b is None:
        return False
    if isinstance(b, (set, frozenset, list, tuple)):
        return a in b
    return a == b


def affinite_axes(
    objet: str | None = None,
    lieu: str | None = None,
    creneau: str | None = None,
    motif: str | None = None,
    *,
    objet_courant: str | None = None,
    lieu_courant: str | None = None,
    creneau_courant: str | None = None,
    motif_courant: str | None = None,
) -> float:
    """Affinity of a memory with the current context, on [0, 1].

    **BONUS, never veto.** A mismatching axis contributes zero: it subtracts nothing, and a
    memory with no matching axis remains rankable on its four other components.
    This is what allows a morning bike fall to reach the evening decision, even though
    none of their contexts coincide. It is also the partial-matching rule
    of ACT-R: a mismatching attribute applies a penalty, it does not exclude.
    """
    total = 0.0
    if _concorde(objet, objet_courant):
        total += POIDS_AXE_OBJET
    if _concorde(lieu, lieu_courant):
        total += POIDS_AXE_LIEU
    if _concorde(creneau, creneau_courant):
        total += POIDS_AXE_CRENEAU
    if _concorde(motif, motif_courant):
        total += POIDS_AXE_MOTIF
    return total


def affinite_meteo(meteo: str | None, meteo_courante: str | None) -> float:
    """Discrete attribute matching on the WEATHER, on [0, 1].

    ⚠ **Decision of 2026-09-14, issue A** (`specs/ticket_071/tests_lot2.md` § 8.1). The ticket
    planned here a "categorical affinity" matching mode, time slot, purpose and weather — but
    three of these four attributes are ALREADY the axes of `affinite_axes`. They would have been
    counted twice, with different weights, without anything saying so: a score two of whose terms
    measure the same thing is no longer interpretable, and calibrating the five weights would have
    borne on components correlated by construction.

    The component is therefore reduced to the only attribute it brings, the weather — and it makes
    sense: a rain memory sheds light on a decision taken in the rain. The five components
    become disjoint again, and their number does not change.
    """
    return 1.0 if _concorde(meteo, meteo_courante) else 0.0
