"""
core/housing_type.py — The persona's housing type, imputed and not invented.

The EMC² reference breaks modal shares down along eight axes, including **housing
type** (variable `M1` of the survey's household file). On the simulation side, this
trait exists nowhere: neither eqasim nor the census used by the generation
chain carries it, and the « Type de logement » column of the trip
log was written empty (action A2).

**This module imputes, and says so.** An imputed axis does not have the status of an observed
axis, and anything that publishes it must remind the reader. Four safeguards frame
the imputation:

1. **Conditioned on geography.** A draw independent of location would produce
   large apartment blocks in the rural outskirts and would distort precisely the axis we
   are trying to measure. The law drawn is that of the **households of the fine zone** (weighted
   by the survey's adjustment coefficients), falling back on its sampling
   sector then on the whole scope when the zone is too thin.
2. **Conditioned on household size** (ticket 019). Within the same zone,
   families live in houses and single persons in apartments: the
   zone law alone mixed the two and **flattened the gradient**. The zone
   law therefore receives a **size lever** estimated at scope level, then it is
   renormalised:

       P(M1 = m | zone, size) ∝ P(M1 = m | zone) × [ P(M1 = m | size) / P(M1 = m) ]

   This odds-ratio transfer (one-dimensional *raking*) divides by four
   the error of the mechanism measured inside EMC², without moving geography
   (3.00 → 0.75 point; § "What the lever does" below).
3. **Deterministic.** The draw is a hash function of the home address,
   not of a random generator: two runs, two machines and two moments
   give the same trait for the same dwelling. The key is the **address** and not the
   person, so that two personas of the same household do not end up one in a
   detached house and the other in a tower block.
4. **Outside the layer, no guessing.** A home that belongs to no fine
   zone (outside the survey scope) has no housing type: the function returns
   `None`, and the log column stays empty — "not filled in" is a piece of
   information, not a category. A persona **without household size** returns
   `None` in the same way: serving the zone law alone would be exactly the silent
   fallback, and the flattened gradient, that ticket 019 removes.

What the lever does — measured inside EMC², each surveyed household receiving the
law of its zone corrected for its size, then compared with its actual `M1` (share of detached
houses, and mean absolute error over the 20 cells 5 categories × 4 sizes):

    size         observed  zone only (COEP, before)   zone COE0 + lever (after)
    1            15.7%               26.4%                     14.0%
    2            46.5%               41.6%                     45.6%
    3            45.5%               45.3%                     47.1%
    4 and +      53.9%               47.8%                     55.2%
    mean abs. error                  3.00 pt                   0.75 pt

The overall marginal does not move: 34.7 / 12.9 / 28.2 / 23.6 / 0.6% observed against
34.1 / 13.1 / 28.2 / 23.9 / 0.7% raked. The lever shifts sizes relative to
one another, it does not shift geography.

Two lessons not to lose:

- **Weighting is not the issue.** Switching from person weights (`COEP`) to household
  weights (`COE0`) *without* the lever **degrades** the result (3.00 → 3.76 pt): the
  person weighting partly compensated for the missing size, by coincidence.
  Once size is conditioned, household weighting is the right one — a household draws
  once — and the person marginal rebuilds itself. Never touch
  one without the other.
- **The residual is real and small.** The raked law overshoots by 1.2 to 1.6 points at sizes 3
  and 4+: the transfer assumption (the size lever is the same in all
  zones) is not exact, it is good to 0.75 point. A lever estimated per sampling
  sector is the next improvement, outside the scope of ticket 019.

**Trap**: the size to use is the persona's **nominal** `household_size`, not the
number of members present in the population file. 118 of the 498 address clusters
of `toulouse_population_1000.json` are partial (bbox filtering); drawing on the
number present would put families of four into single-person laws.
Same rule as stage 2 of ticket 015.

The resource (`mobility_core/data/zf_housing_type.json`) is produced by
`scripts/progedo_logit/export_housing_type.py` (`make housing-type`) from the
restricted-access microdata. Like the fine-zone layer, it is **outside the
repository**: its absence is a normal case, handled by an explicit error when
`load` is called, never by a silent fallback draw. A `version` 1 resource (without
a levers block) is **rejected** at load time: imputing without a lever would mean going back to
the mechanism that ticket 019 corrects, silently.

Inputs/outputs are confined to `HousingTypeTable.load`; the rest of the module is
pure, in line with the `mobility_core` architecture contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mobility_core.resources import restricted_data_path, restricted_resource_hint

# Categories, in the order that fixes the draw's cumulative distribution. The keys are
# those of `scripts/data/population/cerema_values.yaml` (`parts_modales_2023.
# type_logement`): it is the join key of the synthesis page, and it must
# not deviate from it. The labels are the survey's (`M1`), and they are what
# `traits_json` carries, then the « Type de logement » column of the log.
#
# `autres` is NOT in the reference: the survey knows it (0.4% of persons),
# the published EMC² breakdown ignores it. It is imputed anyway — squashing it onto one
# of the four other categories would redistribute 0.4% of the population without a
# source — and the page counts it outside the reference data, as it already does for
# unknown occupations.
# (key, label), and the two do NOT follow the same rule since ticket 074.
#
# The KEY indexes `cerema_values.yaml`, which `prompt_calibration/` also reads: it stays
# French, as the identifier it is. The LABEL is carried by the persona and read by
# `move_logger`; it switched to English with everything else that goes out to the model.
MODALITIES: tuple[tuple[str, str], ...] = (
    ("individuel_isole", "Detached house"),
    ("individuel_accole", "Terraced house"),
    ("petit_habitat_collectif", "Small apartment building"),
    ("grand_habitat_collectif", "Large apartment building"),
    ("autres", "Other"),
)

#: Pre-v6 labels. `key_for` still accepts them: archived cohorts carry them,
#: and `move_logger._housing_type` would otherwise return an empty column over all of v5 — an
#: absence that would read as "trait not imputed", which is false.
MODALITIES_FR: tuple[tuple[str, str], ...] = (
    ("individuel_isole", "Individuel isolé"),
    ("individuel_accole", "Individuel accolé"),
    ("petit_habitat_collectif", "Petit habitat collectif"),
    ("grand_habitat_collectif", "Grand habitat collectif"),
    ("autres", "Autres"),
)

MODALITY_KEYS: tuple[str, ...] = tuple(key for key, _ in MODALITIES)
LABEL_BY_KEY: dict[str, str] = dict(MODALITIES)
KEY_BY_LABEL: dict[str, str] = {label: key for key, label in MODALITIES}

#: Pre-v6 label → key. SEPARATE from `KEY_BY_LABEL`, which must remain a bijection:
#: a test checks it, and it is what guarantees that no category has two keys.
KEY_BY_LABEL_FR: dict[str, str] = {label: key for key, label in MODALITIES_FR}

# The four categories actually broken down by the EMC² reference.
REFERENCE_KEYS: tuple[str, ...] = MODALITY_KEYS[:4]

# Trait key in `traits_json`. The persona carries the LABEL, not the key: it is what
# the trip log reads, and what a human reads back in the JSON.
TRAIT_KEY = "housing_type"

# Persona trait carrying the NOMINAL household size (cf. the trap in the docstring).
SIZE_TRAIT_KEY = "household_size"

# Capping of household size. Four classes: 1, 2, 3, 4 and more. Beyond that,
# the survey counts 1,271 households of 4+ for 4,778 single persons: a finer split
# would give levers estimated on a few dozen households.
SIZE_MAX = 4

# Draw salt. Versioned: changing it reshuffles all imputations, which must be
# a deliberate and dated act, not a side effect. `v2` = conditioning on household
# size (ticket 019), recorded in the changelog of 2026-08-21.
DRAW_SALT = "housing_type_v2"

# Minimum resource version. A v1 does not carry the size levers: it
# is rejected rather than served without conditioning.
MIN_RESOURCE_VERSION = 2

# Sampling-sector prefix in the fine-zone code (`101101000` → `1011`), the
# same split that `build_mode_choice_dataset.build_geo` uses for the hypercentre.
SECTOR_PREFIX_LEN = 4

# RESTRICTED-ACCESS resource: it publishes the count per fine zone (median 12 households,
# 195 zones under 5), so it goes neither into the repository nor into the package (ticket 038).
RESOURCE_NAME = "zf_housing_type.json"

# Evaluated at import, for callers that test its existence; `load()` resolves again.
DEFAULT_RESOURCE = restricted_data_path(RESOURCE_NAME)


def label_for(key: str) -> str | None:
    """Category key → EMC² label. `None` if the key is unknown."""
    return LABEL_BY_KEY.get(key)


def key_for(label: str) -> str | None:
    """EMC² label → category key, in BOTH vocabularies. `None` if unknown."""
    texte = (label or "").strip()
    return KEY_BY_LABEL.get(texte) or KEY_BY_LABEL_FR.get(texte)


def size_bucket(household_size: float | str | None) -> int | None:
    """Nominal household size → lever class (1, 2, 3, 4 = "4 and more").

    `None` when the size is missing or unusable: the caller must then
    give up the trait, not fall back on the zone law alone.
    """
    if household_size is None:
        return None
    try:
        size = int(household_size)
    except (TypeError, ValueError):
        return None
    if size < 1:
        return None
    return min(size, SIZE_MAX)


def address_key(lat: float, lon: float) -> str:
    """Stable identifier of an address, to 10⁻⁶ degree (~0.1 m).

    The draw is on the address and not on the person: in the synthetic
    population, 930 personas share 498 homes (up to 6 per address). Making
    them draw separately would put flatmates in two different dwellings.
    """
    return f"{float(lat):.6f},{float(lon):.6f}"


def uniform(key: str) -> float:
    """Deterministic uniform on [0, 1), derived from a stable hash.

    Python's `hash()` is randomised per process: it would give different traits
    on every run. SHA-256 depends neither on the version, nor on the
    platform, nor on `PYTHONHASHSEED`.
    """
    digest = hashlib.sha256(f"{DRAW_SALT}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def rake(shares: Sequence[float], leverage: Sequence[float] | None) -> tuple[float, ...]:
    """Zone law × size lever, renormalised. Missing lever → law unchanged.

    It is the odds-ratio transfer of ticket 019: geography sets the
    level, household size shifts the categories relative to one another.
    A category with zero mass in the zone stays so — the lever does not create
    dwellings that the survey did not see there.
    """
    # `len` and not the truth value: the law may arrive as a numpy array from
    # the exporter, and `not array` raises on more than one element.
    if shares is None or len(shares) == 0:
        return ()
    if leverage is None:
        return tuple(float(v) for v in shares)
    if len(leverage) != len(shares):
        raise ValueError(
            f"Size lever of length {len(leverage)} applied to a law of "
            f"length {len(shares)}: the resource and the module have diverged.")
    tilted = [float(s) * float(lev) for s, lev in zip(shares, leverage)]
    total = sum(tilted)
    if total <= 0:
        return ()
    return tuple(v / total for v in tilted)


def draw(shares: Sequence[float], u: float) -> str | None:
    """Inverse of the cumulative distribution function over `MODALITY_KEYS`.

    `shares` follows the order of `MODALITY_KEYS` and sums to 1. An empty or
    degenerate law (zero sum) returns `None` rather than a default category: imputing
    from nothing would be exactly the invention this module refuses.
    """
    if shares is None or len(shares) == 0:
        return None
    total = sum(shares)
    if total <= 0:
        return None
    threshold = u * total
    cumulated = 0.0
    for key, share in zip(MODALITY_KEYS, shares):
        cumulated += share
        if threshold < cumulated:
            return key
    # Numerical net: u < 1 guarantees we only exit here on floating-point rounding.
    return MODALITY_KEYS[len(shares) - 1]


@dataclass(frozen=True)
class HousingTypeTable:
    """Law of housing type by fine zone and household size, as the survey
    gives it.

    `zones` carries an already smoothed law for each zone known to the survey;
    `sectors` and `global_shares` are the fallbacks, in that order. Smoothing is done
    at export, not here: the table served is the one that can be reread and checked.

    `size_leverage` carries the ratios `P(M1 | size) / P(M1)` per size class
    (1, 2, 3, 4+), estimated at **scope** level: the (zone, size) cell
    counts 3 households at the median, serving its raw ratio would pass sampling
    noise off as geography.
    """

    zones: dict[str, tuple[float, ...]]
    sectors: dict[str, tuple[float, ...]]
    global_shares: tuple[float, ...]
    size_leverage: dict[int, tuple[float, ...]]
    meta: dict
    # The internal EMC² test published by the export: it carries the shares
    # observed by household size, the only enforceable targets for an enriched
    # population. Empty on a hand-built table (tests).
    validation: dict = field(default_factory=dict)

    @classmethod
    def load(cls, resource: Path | None = None) -> HousingTypeTable:
        """Loads the resource (the module's only I/O point).

        Missing = explicit error. A silent fallback to the overall law
        would produce a trait decorrelated from geography, i.e. precisely the
        bias the imputation exists to avoid. Resource older than ticket 019
        (`version` 1, without levers) = explicit error for the same reason: it
        would impute without household size, with nothing reporting it.
        """
        path = Path(resource) if resource else restricted_data_path(RESOURCE_NAME)
        if not path.exists():
            raise FileNotFoundError(
                "Housing type table missing.\n" + restricted_resource_hint(RESOURCE_NAME)
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        modalities = tuple(doc.get("modalities") or ())
        if modalities != MODALITY_KEYS:
            raise ValueError(
                f"Table {path} written for categories other than the module's.\n"
                f"  table  : {modalities}\n  module : {MODALITY_KEYS}\n"
                "Re-export the table (make housing-type)."
            )
        version = int(doc.get("version") or 0)
        if version < MIN_RESOURCE_VERSION:
            raise ValueError(
                f"Table {path} is at version {version}, but the module requires "
                f"{MIN_RESOURCE_VERSION}: it does not carry the household size "
                "levers (ticket 019) and would impute housing on the fine zone alone, "
                "flattening the size gradient. Re-export it "
                "(make housing-type)."
            )
        leverage = {}
        for size, node in (doc.get("size_leverage") or {}).items():
            # JSON keys are strings; `size_bucket` converts them.
            bucket = size_bucket(size)
            if bucket is None:
                continue
            leverage[bucket] = tuple(float(v) for v in node["leverage"])
        missing = [size for size in range(1, SIZE_MAX + 1) if size not in leverage]
        if missing:
            raise ValueError(
                f"Table {path} has no lever for household sizes {missing}: "
                "the imputation would be conditioned for some households and not for "
                "others. Re-export it (make housing-type)."
            )
        return cls(
            zones={str(zf): tuple(float(v) for v in node["shares"])
                   for zf, node in (doc.get("zones") or {}).items()},
            sectors={str(sec): tuple(float(v) for v in node["shares"])
                     for sec, node in (doc.get("sectors") or {}).items()},
            global_shares=tuple(float(v) for v in (doc.get("global") or ())),
            size_leverage=leverage,
            meta=doc.get("meta") or {},
            validation=doc.get("validation") or {},
        )

    def observed_isolated_share_by_size(self) -> dict[int, float]:
        """Share of detached houses **observed** in the survey, by household size.

        Target of acceptance criterion no. 2 of ticket 019 (15.7 / 46.4 / 45.5 / 53.9%).
        Empty if the table carries no validation block.

        An incomplete row is **ignored**, not filled with a default: it is a
        verdict path, and a target made up from a missing key would have
        the population judged against nothing. The downstream check can say "target not
        served"; it could not make up for an invented target.
        """
        rows = ((self.validation.get("delivered") or {}).get("by_size") or [])
        out: dict[int, float] = {}
        for row in rows:
            size = size_bucket(row.get("size"))
            observed = row.get("individuel_isole_observed_pct")
            if size is not None and observed is not None:
                out[size] = float(observed)
        return out

    def level_for(self, zf: str | None) -> str | None:
        """Fallback level served for a zone: `zone`, `secteur` or `perimetre`.

        Published at each enrichment: this count says how many personas
        received the law of their zone, and how many that of a wider aggregate.
        """
        if zf is None:
            return None
        zf = str(zf)
        if zf in self.zones:
            return "zone"
        if zf[:SECTOR_PREFIX_LEN] in self.sectors:
            return "secteur"
        return "perimetre"

    def zone_shares(self, zf: str | None) -> tuple[float, ...]:
        """Geographic law of a fine zone, before the size lever: its own, that
        of its sector, or that of the whole scope. Unknown zone → empty law."""
        if zf is None:
            return ()
        zf = str(zf)
        if zf in self.zones:
            return self.zones[zf]
        sector = zf[:SECTOR_PREFIX_LEN]
        if sector in self.sectors:
            return self.sectors[sector]
        return self.global_shares

    def shares_for(self, zf: str | None,
                   household_size: float | None) -> tuple[float, ...]:
        """Law applicable to a home: that of its zone, raked on its size.

        `household_size` is the **nominal** household size. When missing, the law is
        empty: the trait will be `None`, and a counted `None` is better than a draw
        from the zone law alone, which was measured to flatten the gradient.
        """
        shares = self.zone_shares(zf)
        if not shares:
            return ()
        bucket = size_bucket(household_size)
        if bucket is None:
            return ()
        return rake(shares, self.size_leverage.get(bucket))

    def housing_type(self, zf: str | None, lat: float, lon: float,
                     household_size: float | None) -> str | None:
        """EMC² label of a home's dwelling, or `None` if it is outside the layer or
        without household size."""
        key = draw(self.shares_for(zf, household_size),
                   uniform(address_key(lat, lon)))
        return label_for(key) if key else None
