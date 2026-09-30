"""
core/mode_choice.py — From the probability vector returned by the LLM to the drawn mode.

The LLM no longer chooses an option: it assigns to **each** proposed option the
probability that the persona picks it (sum = 100). The actual choice is a
random draw from this distribution, which restores at the aggregate level the
variability of individual behaviours instead of a deterministic argmax.

Three responsibilities, all **pure** (no I/O, no provider dependency):

1. ``normalize_option_probabilities`` — cleans the raw LLM vector (out-of-range
   indices, duplicates, negatives, sum ≠ 100) and brings it back to a distribution
   index-aligned on the options actually proposed;
2. ``mode_distribution`` — projects this distribution onto the **canonical** list
   of modes: a mode absent from the options (e.g. walking when the trip is too
   long) explicitly receives 0%, which makes the splits comparable from one
   agent to another and from one run to another;
3. ``draw_index`` — draws an option according to its probabilities, with a seed
   derived deterministically from the decision context: two identical runs
   replay the same draw, but the same agent draws differently from one day to
   the next (and hence at each cache hit on a new simulated day).

Shared between the worker (metrics), the simulation agents (decision + semantic
cache) and the prompt calibration module, so that all apply
exactly the same decision policy.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Canonical modes
# ---------------------------------------------------------------------------

# Closed list of modes over which every split is expressed. A mode
# not proposed in the options stays present with 0% (cf. mode_distribution).
# Aligned on the project's reference palette (cf. .claude/CLAUDE.md).
CANONICAL_MODES: tuple[str, ...] = (
    "walking",
    "cycling",
    "car",
    "public_transport",
    "train",
    "motorbike",
)

# Overflow bucket: unrecognised modes. Outside CANONICAL_MODES so as not to
# pollute the comparisons, but kept in the split so that the sum
# stays 1 whatever happens.
OTHER_MODE = "other"

# Keywords → canonical mode. **THE ORDER IS THE SURVEY HIERARCHY**, plus a
# reading convention: a composite chain ("foot,bus,foot") also contains "foot",
# and it is the best-ranked mode that qualifies the trip.
#
# The order changed on 2026-09-04 (ticket 022): `train` was tested FIRST, so an
# option "autocar liO + TER" ("foot,bus,rail,foot") was counted as `train`. The appendix
# "Hiérarchie des modes" of the AUAT/CEREMA report (p. 53) puts the bus at rank 4 and the TER
# at rank 8, and the microdata confirm it: 34 of the 35 resolved mixed bus ↔ train trips
# are coded bus. `public_transport` therefore moves ahead of `train`, and `car` ahead of
# `motorbike` (ranks 19-20 versus 21-22). The check `_verifier_ordre()` below
# compares this order with the frozen resource: it can no longer drift silently.
_MODE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("public_transport", ("metro", "métro", "subway", "tram", "tramway", "bus",
                          "school_bus",  # synthetic school bus (ticket 030)
                          # Téléo (route_type=6) and cousin aerial lifts. ADDED on
                          # 2026-09-04: they were missing from the SIX lists, so that a
                          # PURE cable car option ("foot,cableway,foot")
                          # went down the cascade as far as "walking", which the word
                          # "foot" satisfies — the probability mass of the Téléo was
                          # counted as WALKING in `mode_distribution`, hence in the
                          # P(...) columns of moves.csv and in Grafana 07.
                          # `move_logger._BUS_MODES` and `categorize_mode` already
                          # carried them (fix of 2026-08-26): divergence fixed, not a
                          # new convention. Effect measured before applying:
                          # 120 of the 385,888 options of the frozen sets (0.031%), 5 of
                          # the 17,258 of the last archived run, a single label affected.
                          "cableway", "gondola", "funicular",
                          "transit", "public_transport", "transports en commun",
                          "transport en commun", "transports_collectifs", "tc")),
    ("train", ("rail", "train", "ter", "intercités", "intercites")),
    ("car", ("car", "driving", "voiture", "conducteur", "taxi")),
    ("motorbike", ("scooter", "moto", "motorbike", "motorcycle", "deux-roues",
                   "deux_roues")),
    ("cycling", ("bicycle", "bike", "cycling", "vélo", "velo")),
    ("walking", ("foot", "walk", "walking", "marche", "à pied", "a pied", "pied")),
)


def _verifier_ordre() -> None:
    """Does the cascade follow the frozen survey hierarchy?

    Checked at import, not in a test: a test that copies the expected order only fails
    if the instrument changes, never if production changes — it is this asymmetry that
    let the Téléo and rail defects through (cf. `scripts/tests/test_parite_modes`).
    """
    from mobility_core.mode_hierarchy import hierarchy

    attendu = hierarchy().canonical_order()
    observe = tuple(mode for mode, _ in _MODE_KEYWORDS)
    if observe != attendu:
        raise RuntimeError(
            "The order of `_MODE_KEYWORDS` no longer follows the survey hierarchy.\n"
            f"  cascade   : {observe}\n  hierarchy : {attendu}\n"
            "Fix the cascade, or re-export the hierarchy "
            "(scripts/progedo_logit/export_mode_hierarchy.py) if the survey has changed.")
    manquants = [mode for mode in attendu if mode not in CANONICAL_MODES]
    if manquants:
        raise RuntimeError(
            f"The hierarchy carries canonical modes absent from CANONICAL_MODES: "
            f"{manquants} — their probability mass would be lost.")


_verifier_ordre()


def canonical_mode(raw: str | None) -> str:
    """Reduces a mode chain ("foot,bus,foot", "BICYCLE") to a canonical mode.

    Returns ``OTHER_MODE`` if no known keyword is recognised — the case is logged
    as WARNING because it signals either a new itinerary mode or a label
    invented by the LLM.
    """
    if not raw:
        return OTHER_MODE
    text = str(raw).lower()
    parts = {p.strip() for p in text.replace("+", ",").split(",") if p.strip()}
    for mode, keywords in _MODE_KEYWORDS:
        for kw in keywords:
            # Equality on a segment first ("bus" in "foot,bus,foot"), then
            # substring for spelled-out labels ("transports en commun").
            # Short words are excluded from the substring search: "tc" or
            # "car" occur in too many words to be reliable that way.
            if kw in parts or (len(kw) >= 5 and kw in text):
                return mode
    logger.warning(f"Mode de transport non reconnu, classé 'other' | mode={raw!r}")
    return OTHER_MODE


# ---------------------------------------------------------------------------
# Normalisation of the raw vector
# ---------------------------------------------------------------------------

class UniformFallback(list):
    """Uniform vector produced by a fallback (LLM vector absent or with a zero sum).

    Subclass of ``list``: every caller treats it as a normal vector
    (draw, metrics, equality with a plain list). But it is NOT a
    decision of the model — callers that PERSIST a distribution (LLM
    cache) must test it (``isinstance``) and refuse the write, otherwise the
    fallback of one run pollutes the draws of all following runs (observed on
    2026-08-03: cerebras_zai-glm-4.7 truncations → uniform distributions
    written to the cache as legitimate decisions).
    """


def _entry_field(entry: Any, name: str) -> Any:
    """Reads a field of an entry, whether it is a dict or a Pydantic model."""
    if isinstance(entry, Mapping):
        return entry.get(name)
    return getattr(entry, name, None)


def _as_float(value: Any) -> float | None:
    """Coerces a probability ("40", "40 %", 0.4) to float, None if unreadable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _options_matching_mode(raw_mode: str | None, modes: Sequence[str | None]) -> list[int]:
    """Options designated by the mode label of an entry whose index is out of range.

    Two passes, from the safest to the most tolerant:
    1. string equality with the mode sent — the model copies it as is, it is
       an unambiguous designation;
    2. canonical mode equality ("Marche jusqu'à 'work'" → ``walking``), which catches
       the entries where it rephrased the label.

    Returns the list of candidate options (empty if none): several options may
    share a mode, it is up to the caller to decide what to do with them.
    """
    if not raw_mode:
        return []
    text = " ".join(str(raw_mode).strip().lower().split())
    exact = [i for i, m in enumerate(modes)
             if m and " ".join(str(m).strip().lower().split()) == text]
    if exact:
        return exact
    target = canonical_mode(raw_mode)
    if target == OTHER_MODE:
        return []
    return [i for i, m in enumerate(modes) if m and canonical_mode(m) == target]


def normalize_option_probabilities(
    entries: Iterable[Any] | None,
    n_options: int,
    *,
    modes: Sequence[str | None] | None = None,
    context: str = "",
) -> list[float]:
    """Converts the raw LLM vector into a distribution index-aligned on the options.

    ``entries``: iterable of ``{index, probability}`` (dicts or ``OptionProbability``).
    ``modes``: mode **sent** for each option (source of truth); when provided, it allows
    catching renumbered vectors (cf. below). Returns a list of
    ``n_options`` floats summing to 1.

    Tolerances — the LLM makes mistakes, the simulation must not stop for all that:
    - probabilities are accepted in % (sum 100) as well as in fractions (sum 1):
      the final normalisation makes the scale irrelevant;
    - an entry with an unreadable index is ignored (logged);
    - several entries on the same index are summed;
    - a negative probability is brought back to 0;
    - an option never rated receives 0;
    - if the total is zero or the vector absent, fallback to uniform (the draw
      stays possible, and the anomaly is logged as ERROR: the model's decision is
      then lost, the run's modal split bears the trace of it).

    Out-of-range indices — some models renumber the options (they count the
    detailed steps of an itinerary, or continue the numbering from one persona to
    the next in a batch) and put the mass on non-existent indices. With ``modes``,
    the entry is realigned on the option its label designates; if several options
    share this mode, the mass is split between them — arbitrary as to the
    exact itinerary, but faithful to the **modal share**, which is the measure. Without
    ``modes`` (or with an unrecognised label), the entry is discarded: better to lose the
    mass than to put it on the wrong option.
    """
    if n_options <= 0:
        return []

    weights = [0.0] * n_options
    seen = False

    for i, entry in enumerate(entries or ()):
        if isinstance(entry, (int, float)):
            idx = i
            prob = float(entry)
        else:
            raw_index = _entry_field(entry, "index")
            try:
                idx = int(raw_index)
            except (TypeError, ValueError):
                logger.warning(f"Probability ignored: unreadable index {raw_index!r} | {context}")
                continue
            prob = _as_float(_entry_field(entry, "probability"))
            if prob is None:
                logger.warning(f"Probability ignored: unreadable value for index {idx} | {context}")
                continue
        prob = max(0.0, prob)

        if not 0 <= idx < n_options:
            if prob <= 0:
                # An option rated 0 on a non-existent index costs nothing: it is the
                # noise of the renumbering, no need to flood the logs with it.
                logger.debug(
                    f"Out-of-range entry with zero probability ignored: index {idx} "
                    f"(0..{n_options - 1}) | {context}"
                )
                continue
            raw_mode = _entry_field(entry, "mode")
            targets = _options_matching_mode(raw_mode, modes or ())
            if not targets:
                logger.warning(
                    f"Probability lost: index {idx} out of range (0..{n_options - 1}), "
                    f"mode {raw_mode!r} not realignable — {prob:g} point(s) dropped | {context}"
                )
                continue
            where = f"option {targets[0]}" if len(targets) == 1 \
                else f"options {targets} (mass split equally)"
            logger.warning(
                f"Index {idx} out of range (0..{n_options - 1}) realigned on {where} "
                f"via its mode {raw_mode!r} — {prob:g} point(s) kept | {context}"
            )
            for target in targets:
                weights[target] += prob / len(targets)
            seen = True
            continue

        weights[idx] += prob
        seen = True

    total = sum(weights)
    if not seen or total <= 0:
        logger.error(
            f"[ALARME] Vecteur de probabilités inexploitable "
            f"({'absent' if not seen else 'somme nulle'}) — fallback to a uniform "
            f"distribution over {n_options} option(s), the model's decision is lost | {context}"
        )
        return UniformFallback(1.0 / n_options for _ in range(n_options))

    return [w / total for w in weights]


def option_modes(entries: Iterable[Any] | None, n_options: int) -> list[str | None]:
    """Extracts the mode label announced by the LLM for each option index."""
    modes: list[str | None] = [None] * n_options
    for entry in entries or ():
        try:
            idx = int(_entry_field(entry, "index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_options:
            mode = _entry_field(entry, "mode")
            if mode:
                modes[idx] = str(mode)
    return modes


# ---------------------------------------------------------------------------
# Split by mode
# ---------------------------------------------------------------------------

def mode_distribution(
    weights: Sequence[float],
    modes: Sequence[str | None],
) -> dict[str, float]:
    """Aggregates a distribution over the options into a split by canonical mode.

    All the keys of ``CANONICAL_MODES`` are present: a mode that no option
    proposes (walking excluded because the trip is too long, no train on
    the corridor…) is explicitly 0.0 — this is what makes two splits comparable
    without asking whether the mode was offered. ``other`` appears only if non-zero.

    The weights are assumed normalised (cf. ``normalize_option_probabilities``);
    the sum of the values is therefore 1 (up to floating-point rounding).
    """
    dist: dict[str, float] = {m: 0.0 for m in CANONICAL_MODES}
    for i, w in enumerate(weights):
        mode = canonical_mode(modes[i] if i < len(modes) else None)
        if mode == OTHER_MODE:
            dist[OTHER_MODE] = dist.get(OTHER_MODE, 0.0) + w
        else:
            dist[mode] += w
    if dist.get(OTHER_MODE) == 0.0:
        dist.pop(OTHER_MODE, None)
    return dist


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def derive_seed(*parts: Any) -> int:
    """64-bit seed derived from the decision context (agent, activity, day, run).

    Deterministic and stable across processes (unlike ``hash()``), which
    makes a run replayable identically while varying the draw from one
    context to another.
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big")


def draw_index(
    weights: Sequence[float],
    *seed_parts: Any,
    min_prob_threshold: float = 0.0,
) -> int:
    """Draws the index of an option in proportion to its probability.

    Without ``seed_parts``, the draw uses a fresh ``Random`` instance, not
    replayable. With them, the seed is derived from the context: same context → same draw.

    If ``min_prob_threshold > 0.0``, the options whose relative share
    is strictly below the threshold are eliminated (weight set to zero)
    and the remaining mass is drawn (Consideration Set theory).
    If all options fall below the threshold (edge case), the original weights are kept.
    """
    if not weights:
        raise ValueError("draw_index: no option to draw")
    rng = random.Random(derive_seed(*seed_parts)) if seed_parts else random.Random()
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))

    effective_weights = list(weights)
    if min_prob_threshold > 0.0:
        filtered = [
            w if (w / total) >= min_prob_threshold else 0.0
            for w in weights
        ]
        if sum(filtered) > 0:
            effective_weights = filtered

    # random.choices tolerates non-normalised weights — no need to re-normalise.
    return rng.choices(range(len(effective_weights)), weights=effective_weights, k=1)[0]


def argmax_index(weights: Sequence[float]) -> int:
    """Index of the most probable option (tie-break: the smallest index).

    Used for metrics and for the deterministic fallback, never for the agent's decision.
    """
    if not weights:
        raise ValueError("argmax_index: no option")
    return max(range(len(weights)), key=lambda i: weights[i])
