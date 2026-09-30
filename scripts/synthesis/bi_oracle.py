"""bi_oracle.py — Two-oracle composite score, the multinomial logit as arbiter.

The fidelity composite score measures only one thing: the gap of modal shares to the
EMC² 2023 survey. It is necessary and insufficient. Martín-Baos et al. (2023) show that the
ranking of mode choice models **flips depending on the family of indicators** — the
gradient-boosted trees beat the multinomial logit on disaggregate accuracy, the
logit regains the lead on aggregate shares and on behavioural indicators. A
single scalar therefore cannot rank, and this module produces the two terms that were missing
from the published figure.

## The three blocks, and what each one answers

| Block | Question | Oracle |
|---|---|---|
| **A** — fidelity | are the modal shares those of the survey? | the survey (unchanged) |
| **B** — disaggregate agreement | decision by decision, does the prompt answer like a behavioural model? | the MNL, relative to the booster |
| **B'** — the same thing, against the third family | *(published column, same computation)* | the KLR, relative to the booster |
| **C** — behavioural consistency | do the shares move in the **direction** a behavioural model predicts? | the MNL **and** the KLR |

## Two arbiters for block C (ticket 043)

A single arbiter cannot be refuted: when the prompt contradicts the logit's direction of
change, nothing says whether the defect lies in the prompt or in the arbiter. Since ticket
043, kernel logistic regression — non-linear like the booster, smooth like the logit — is
the **second arbiter** of block C1. A transition enters the score only if both arbiters
carry a clear sign and the **same** one; if they diverge, it is **set aside and counted**,
and the test stays silent rather than blaming the prompt for a disagreement the models
have not settled between themselves. Without a KLR parquet (`make klr && make klr-predict`),
the single-arbiter regime is unchanged, and the output says which of the two produced the
figure.

The weights do not move for all that (K12): a second arbiter changes what the block measures,
it does not change the rule that forbids selecting a prompt against a model.

All terms are oriented "loss, smaller is better", and the composite score stays
**linear**:

    S₂ = Σ_dim w_dim·s_dim  +  w_B·s_B  +  w_C·s_C

`w_B` and `w_C` are **0** (manifest `score.bi_oracle`, settled on 2026-09-10): both
terms are computed, logged and published, but choose no prompt. A prompt
selected under a term measured against a model would fit that model, not the survey.
Linearity makes later promotion exact and retroactive: it suffices to add
`w·s` to the already stored composite score, without paying for a single LLM call again.

## Block B: why a relative divergence, and not an error

Per decision, we measure `KL(p_MNL ‖ p_LLM)` in bits. Two precautions make all the
difference between a readable figure and a decorative one:

1. **we do not publish the raw CEL against a soft target.** The LLM's cross-entropy against
   the MNL probabilities has an irreducible floor equal to the MNL's own entropy (measured
   on the spec example: 1.0623 nat of which 0.9999 is floor). What carries the signal is
   the **excess**, i.e. exactly the Kullback-Leibler divergence;
2. **the denominator is the divergence between the two oracles** on the same decisions.
   Nobody has to judge whether 0.09 bit "is a lot": `s_B = 29` reads "the prompt is
   at 29 % of the distance separating the two oracles from each other". It is a measured scale,
   not a chosen constant.

The MNL is **not** individual truth: no output of this module calls `s_B`
an error. It is an agreement, and it is published with its denominator.

## Block C: the direction of change, not the level

`C1` compares, along the ordinal distance axis, the **sign** of the modal share
changes of the prompt and of the MNL. The logit is monotonic by construction in its continuous
variables: that is what qualifies it as arbiter, and it is also what makes the caveat of
Zhao et al. (2020) — elasticities of tree models « behaviorally unreasonable » —
measurable here. A transition where the arbiter itself has no clear sign is
**set aside and counted**, never used as reference.

`C2` (arc elasticities on an already paid A/B pair) is measured only if such a pair is
provided: this module requests **no** decision again from a language model.

## Vacuity ≠ perfection

A block with no measured count returns `null` and the mention « non mesuré », **never**
`0.0` — which would be the perfect score. The inter-oracle denominator is checked on the
same substrate as the numerator (same run, same `moves.csv` fingerprint): otherwise `s_B`
is not published.

Usage:
    python -m scripts.synthesis.bi_oracle [--config sources.yaml] [--out data/bi_oracle.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import frames
from .model_on_common_set import (
    CANONICAL_TO_CAT,
    PREDICTABLE_CATS,
    STATUS_OK,
    renormalize,
)
from .sources import REPO_ROOT, import_formule_score, load_manifest

#: Probability floor. A zero probability from the prompt on a mode the oracle
#: loads would make the divergence infinite: a single decision would then carry the whole
#: score. The floor is **declared in the output** and the number of decisions concerned
#: is counted — it is a measurement convention, not a discreet smoothing.
EPSILON = 1e-4

#: Ordinal axis of block C1, derived from the same bounds as the decision binning: an axis
#: copied by hand gets misaligned at the first moved threshold.
DIST_ORDER = [label for _, label in frames._DIST_BUCKETS] + ["plus_50km"]

#: Minimal count of a distance class for it to carry a transition. Below it,
#: the modal share of a 4 % mode is only a count of a few decisions.
MIN_STRATUM = 30

#: Amplitude below which the arbiter is deemed **signless** on a transition
#: (in percentage points of modal share). Comparing a sign to a 0.1 point noise
#: would amount to flipping a coin, and grading the prompt on it.
MIN_ARBITER_SHIFT = 0.5


# ── Divergences ──────────────────────────────────────────────────────────────

def kl_bits(target: dict[str, float], candidate: dict[str, float],
            epsilon: float = EPSILON) -> tuple[float, bool]:
    """`KL(target ‖ candidate)` in bits, and the "floor applied" flag.

    Oriented from the oracle: we ask "how much information does the candidate lack to
    account for the oracle", which is the question of block B. The reverse divergence
    would penalise a cautious prompt (spread mass) in favour of a categorical prompt, whereas
    the opposite is what we want to detect.
    """
    total = 0.0
    floored = False
    for mode, p in target.items():
        if p <= 0:
            continue
        q = candidate.get(mode, 0.0)
        if q < epsilon:
            q = epsilon
            floored = True
        total += p * math.log2(p / q)
    return total, floored


def entropy_bits(distribution: dict[str, float]) -> float:
    """Entropy of the distribution, in bits — the floor of any CEL against it."""
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)


def jsd_bits(left: dict[str, float], right: dict[str, float]) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]. **No constant required.**

    It is the headline quantity of block B, and the reason is measured: 1,118 of the 2,593
    decisions of the pinned run carry a near-degenerate distribution (one mode above
    0.999), and 1,923 trigger the Kullback-Leibler divergence floor. The
    KL ratio thus becomes a function of the floor `ε` as much as of the decisions —
    in other words a convention, not a measurement. The JSD is defined on zeros, bounded
    by 1 bit, and symmetric; it neither rewards nor punishes a categorical prompt through a
    scale artefact. The KL ratio remains published second, with the count of
    decisions where the floor kicked in: it is the one comparable to the literature.
    """
    total = 0.0
    for mode in set(left) | set(right):
        p, q = left.get(mode, 0.0), right.get(mode, 0.0)
        mid = 0.5 * (p + q)
        if p > 0:
            total += 0.5 * p * math.log2(p / mid)
        if q > 0:
            total += 0.5 * q * math.log2(q / mid)
    return total


# ── Reading the three decision-makers on the same scope ──────────────────────

def llm_distributions(moves: list[dict]) -> tuple[dict[tuple[str, str], dict], dict]:
    """Prompt decisions, restricted to the predictable offer and renormalised.

    The support must be **that of the oracles**: the modes actually offered by OTP,
    excluding powered two-wheelers and "other modes" (outside the four classes of the
    policy). Comparing three distributions on three different supports would produce
    divergences that only measure the disagreement of scopes.
    """
    out: dict[tuple[str, str], dict] = {}
    skipped = Counter()
    for move in moves:
        offered = [m for m in move["offered"] if m in PREDICTABLE_CATS]
        if not offered:
            skipped["offre_sans_mode_predictible"] += 1
            continue
        if len(offered) < 2:
            # **A single-mode offer is not a comparable decision.** The prompt
            # was not queried (method « Un seul itinéraire disponible »: 656
            # decisions of the pinned run, none carries a distribution), and the three
            # decision-makers would be forced onto the same mode — i.e. a free perfect
            # agreement for everyone. Including it would lower the score without any agreement
            # having been measured: that is vacuity taken for perfection.
            skipped["offre_unique"] += 1
            continue
        distribution = renormalize({m: v for m, v in move["probas"].items()}, offered)
        if distribution is None:
            # Multi-mode offer but no mass on it: to watch, not to
            # fill in. Zero on the pinned run.
            skipped["sans_distribution"] += 1
            continue
        out[(move["agent_id"], move["activity_id"])] = {
            "p": distribution,
            "offered": sorted(offered),
            "dist_cat": move.get("dist_cat"),
            "degenere": max(distribution.values()) > 0.999,
        }
    return out, dict(skipped)


def oracle_distributions(path: Path) -> tuple[dict[tuple[str, str], dict], dict]:
    """Predictions of an oracle → per-decision distributions, plus the parquet description.

    The description travels in the parquet (run, `moves.csv` fingerprint, artefact sha):
    it is what allows refusing two columns measured on two substrates.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}, {"error": "pyarrow indisponible"}
    if not Path(path).exists():
        return {}, {"error": f"parquet absent : {path}"}
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(b"progedo_on_common_set")
    try:
        meta = json.loads(raw) if raw else {}
    except ValueError:
        meta = {}
    out: dict[tuple[str, str], dict] = {}
    for record in table.to_pylist():
        if record.get("status") != STATUS_OK:
            continue
        distribution = {c: float(record[f"p_{c}"]) for c in PREDICTABLE_CATS
                        if record.get(f"p_{c}")}
        if not distribution:
            continue
        offered = [CANONICAL_TO_CAT.get(m, m)
                   for m in (record.get("offered_predictable") or "").split("|") if m]
        if len(set(offered)) < 2:
            continue                  # same rule as the prompt: cf. llm_distributions
        out[(record.get("agent_id"), record.get("activity_id"))] = {
            "p": distribution,
            "offered": sorted(set(offered)),
            "dist_cat": record.get("dist_cat"),
            "degenere": max(distribution.values()) > 0.999,
        }
    return out, meta


# ── Block B: disaggregate agreement with the second oracle ───────────────────

def block_b(llm: dict, mnl: dict, lgb: dict, *, n_perimeter: Optional[int] = None,
            llm_skipped: Optional[dict] = None, epsilon: float = EPSILON,
            arbitre: str = "mnl") -> dict:
    """`s_B` and its breakdown: paired divergences, coverage, counted exclusions.

    Two ratios, and the order between them is deliberate (cf. :func:`jsd_bits`): the JSD first
    because it depends on no constant, the KL ratio second because it is
    the quantity of the literature — published with the count of decisions where its floor
    kicked in, otherwise one would read a convention as a measurement.

    ``arbitre`` names the oracle passed in second position, and **only** that: the published
    keys carry its name. The third family (ticket 043) thus gets its column by
    calling the same function again with ``arbitre="klr"``, without a single figure computed
    through a second path — and without an output measured against the KLR labelled « mnl ».
    A wrong label goes unnoticed; that is what makes it worse than a missing label.
    """
    keys = sorted(set(llm) & set(mnl) & set(lgb))
    excluded = {
        f"sans_{arbitre}": len(set(llm) - set(mnl)),
        "sans_booster": len(set(llm) - set(lgb)),
        "sans_prompt": len(set(mnl) - set(llm)),
        "offre_divergente": 0,
        **{f"prompt::{k}": v for k, v in (llm_skipped or {}).items()},
    }
    jsd_llm = jsd_lgb = kl_llm = kl_lgb = 0.0
    floored = degenerate = 0
    per_decision: list[dict] = []
    for key in keys:
        if not (llm[key]["offered"] == mnl[key]["offered"] == lgb[key]["offered"]):
            # Same decision, three different offers: the divergence would measure the
            # disagreement of scopes, not that of the decision-makers.
            excluded["offre_divergente"] += 1
            continue
        target = mnl[key]["p"]
        d_jsd_llm = jsd_bits(target, llm[key]["p"])
        d_jsd_lgb = jsd_bits(target, lgb[key]["p"])
        d_kl_llm, floor_llm = kl_bits(target, llm[key]["p"], epsilon)
        d_kl_lgb, floor_lgb = kl_bits(target, lgb[key]["p"], epsilon)
        jsd_llm += d_jsd_llm
        jsd_lgb += d_jsd_lgb
        kl_llm += d_kl_llm
        kl_lgb += d_kl_lgb
        floored += int(floor_llm or floor_lgb)
        degenerate += int(llm[key]["degenere"])
        per_decision.append({"jsd_llm": d_jsd_llm, "jsd_lgb": d_jsd_lgb,
                             "kl_llm": d_kl_llm, "kl_lgb": d_kl_lgb,
                             "entropy_arbitre": entropy_bits(target),
                             "dist_cat": mnl[key]["dist_cat"]})

    n = len(per_decision)
    base = n_perimeter if n_perimeter else max(len(llm), 1)
    if n == 0 or jsd_lgb <= 0:
        return {
            "s_B": None,
            "mesure": "non mesuré",
            "raison": ("aucune décision appariée sur les trois décideurs"
                       if n == 0 else
                       "les deux oracles sont exactement d'accord : pas d'échelle"),
            "n_decisions": n,
            "exclusions": excluded,
            "epsilon": epsilon,
        }
    return {
        # The ratio, as a percentage of the inter-oracle distance. An `s_B` of 100
        # means "the prompt is as far from the MNL as the booster is".
        "s_B": 100.0 * jsd_llm / jsd_lgb,
        "mesure": (f"accord à l'oracle {arbitre} (JSD), rapporté à la distance "
                   "inter-oracles mesurée sur les mêmes décisions"),
        "arbitre": arbitre,
        f"jsd_prompt_{arbitre}_bits_mean": jsd_llm / n,
        f"jsd_booster_{arbitre}_bits_mean": jsd_lgb / n,
        # Kullback-Leibler variant, floor-dependent: to read with
        # `n_plancher_applique`, never alone.
        "s_B_kl": (100.0 * kl_llm / kl_lgb) if kl_lgb > 0 else None,
        f"kl_prompt_{arbitre}_bits_mean": kl_llm / n,
        f"kl_booster_{arbitre}_bits_mean": kl_lgb / n,
        "n_plancher_applique": floored,
        "n_prompt_degenere": degenerate,
        f"entropie_{arbitre}_bits_mean": sum(d["entropy_arbitre"] for d in per_decision) / n,
        "n_decisions": n,
        "couverture": n / base,
        "base_couverture": base,
        "exclusions": excluded,
        "epsilon": epsilon,
        "par_distance": _by_distance(per_decision, arbitre),
    }


def _by_distance(per_decision: list[dict], arbitre: str = "mnl") -> list[dict]:
    """Mean divergences per distance class — where agreement is won or lost."""
    grouped: dict[Optional[str], list[dict]] = defaultdict(list)
    for row in per_decision:
        grouped[row["dist_cat"]].append(row)
    out = []
    for cat in DIST_ORDER:
        rows = grouped.get(cat)
        if not rows:
            continue
        num = sum(r["jsd_llm"] for r in rows)
        den = sum(r["jsd_lgb"] for r in rows)
        out.append({
            "dist_cat": cat,
            "n": len(rows),
            f"jsd_prompt_{arbitre}_bits_mean": num / len(rows),
            f"jsd_booster_{arbitre}_bits_mean": den / len(rows),
            # `null` rather than 0: without a local scale, there is no ratio to read.
            "s_B": (100.0 * num / den) if den > 0 else None,
        })
    return out


# ── Block C1: direction of change along the distance axis ────────────────────

def mass_by_stratum(distributions: dict, column: str = "dist_cat") -> dict:
    """Modal shares in probability mass, per axis class, in percent."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for entry in distributions.values():
        cat = entry.get(column)
        if cat is None:
            continue
        counts[cat] += 1
        for mode, mass in entry["p"].items():
            totals[cat][mode] += mass
    out = {}
    for cat, masses in totals.items():
        n = counts[cat]
        out[cat] = {"n": n,
                    "shares": {m: 100.0 * masses.get(m, 0.0) / n for m in PREDICTABLE_CATS}}
    return out


def block_c1(llm: dict, mnl: dict, klr: Optional[dict] = None, *,
             min_stratum: int = MIN_STRATUM,
             min_shift: float = MIN_ARBITER_SHIFT) -> dict:
    """`s_C1`: share of (mode, transition) pairs whose sign contradicts the arbiter.

    **Two arbiters rather than one (K11, ticket 043).** A single arbiter cannot be refuted:
    when the prompt contradicts the logit's direction of change, nothing says whether the
    defect is in the prompt or in the arbiter. With kernel logistic regression as second
    arbiter, the rule becomes explicit:

    - both arbiters carry a clear sign and **the same** one → the transition enters the
      score, and a disagreement of the prompt is a defect of the prompt;
    - the two arbiters **diverge**, or the second has no clear sign → the transition
      **leaves the score and is counted**. The test says nothing, and stays silent.

    Without ``klr``, the single-arbiter regime is unchanged — it is what an output
    produced before the third family existed reads. The output **says** which of the two
    regimes produced the figure: two `s_C1` values that do not rest on the same rule are not
    comparable from one run to another.
    """
    llm_strata = mass_by_stratum(llm)
    mnl_strata = mass_by_stratum(mnl)
    klr_strata = mass_by_stratum(klr) if klr else None
    transitions: list[dict] = []
    skipped = {"effectif_insuffisant": 0, "arbitre_sans_signe": 0}
    if klr_strata is not None:
        skipped["second_arbitre_sans_signe"] = 0
        skipped["arbitres_en_desaccord"] = 0

    axis = [c for c in DIST_ORDER if c in llm_strata and c in mnl_strata
            and (klr_strata is None or c in klr_strata)]
    for left, right in zip(axis, axis[1:]):
        counts = [llm_strata[left]["n"], llm_strata[right]["n"],
                  mnl_strata[left]["n"], mnl_strata[right]["n"]]
        if klr_strata is not None:
            counts += [klr_strata[left]["n"], klr_strata[right]["n"]]
        thin = min(counts) < min_stratum
        for mode in PREDICTABLE_CATS:
            if thin:
                skipped["effectif_insuffisant"] += 1
                continue
            d_mnl = (mnl_strata[right]["shares"][mode] - mnl_strata[left]["shares"][mode])
            if abs(d_mnl) < min_shift:
                # R20: the arbiter has no sign to give on this transition. Counting
                # it as an agreement would flatter the prompt, counting it as a
                # disagreement would condemn it — it leaves the score and is counted.
                skipped["arbitre_sans_signe"] += 1
                continue
            d_klr = None
            if klr_strata is not None:
                d_klr = (klr_strata[right]["shares"][mode]
                         - klr_strata[left]["shares"][mode])
                if abs(d_klr) < min_shift:
                    skipped["second_arbitre_sans_signe"] += 1
                    continue
                if (d_klr > 0) != (d_mnl > 0):
                    # The two behavioural models do not say the same thing: the
                    # transition cannot serve as sign reference, and the prompt's
                    # disagreement there is not interpretable.
                    skipped["arbitres_en_desaccord"] += 1
                    continue
            d_llm = (llm_strata[right]["shares"][mode] - llm_strata[left]["shares"][mode])
            transitions.append({
                "mode": mode, "de": left, "vers": right,
                "delta_llm_pt": d_llm, "delta_mnl_pt": d_mnl,
                **({"delta_klr_pt": d_klr} if d_klr is not None else {}),
                "accord": (d_llm > 0) == (d_mnl > 0),
            })

    regime = ("deux arbitres (logit + KLR, transition écartée s'ils divergent)"
              if klr_strata is not None else "un seul arbitre (logit)")
    if not transitions:
        return {"s_C1": None, "mesure": "non mesuré", "regime": regime,
                "raison": ("aucune transition ne porte à la fois un effectif et un signe "
                           "partagé par les deux arbitres" if klr_strata is not None else
                           "aucune transition ne porte à la fois un effectif et un signe"),
                "ecartees": skipped}
    disagreements = [t for t in transitions if not t["accord"]]
    return {
        "s_C1": 100.0 * len(disagreements) / len(transitions),
        "mesure": "part des couples (mode, transition) dont le signe contredit l'arbitre",
        "regime": regime,
        "n_transitions": len(transitions),
        "n_desaccords": len(disagreements),
        "ecartees": skipped,
        "seuils": {"effectif_min": min_stratum, "amplitude_min_pt": min_shift},
        "desaccords": disagreements,
        "transitions": transitions,
    }


# ── Block C2: arc elasticities on an already paid A/B pair ───────────────────

def arc_elasticities(before: dict, after: dict) -> dict[str, float]:
    """Relative change of each modal share between two states of the same scope.

    Both states must carry **the same decisions**: that is what makes the gap
    a treatment effect, and not a composition effect.
    """
    keys = sorted(set(before) & set(after))
    if not keys:
        return {}
    out = {}
    for mode in PREDICTABLE_CATS:
        p0 = sum(before[k]["p"].get(mode, 0.0) for k in keys) / len(keys)
        p1 = sum(after[k]["p"].get(mode, 0.0) for k in keys) / len(keys)
        out[mode] = ((p1 - p0) / p0) if p0 > 0 else float("nan")
    return out


def block_c2(llm_before: Optional[dict], llm_after: Optional[dict],
             mnl_before: Optional[dict], mnl_after: Optional[dict],
             variable: Optional[str]) -> dict:
    """`s_C2`: share of modes whose elasticity sign contradicts the arbiter.

    Not measured by default, and that is the point: measuring a prompt elasticity requires
    replaying decisions, hence LLM calls. This module makes none — it uses an
    **already paid** A/B pair when one is designated to it.
    """
    if not all((llm_before, llm_after, mnl_before, mnl_after, variable)):
        return {
            "s_C2": None, "mesure": "non mesuré",
            "raison": ("aucune paire A/B fournie — une élasticité du prompt exige des "
                       "décisions rejouées, que ce module ne redemande jamais"),
            "comment": ("--ab-llm-before/--ab-llm-after (moves.csv des deux bras), "
                        "--ab-mnl-before/--ab-mnl-after (parquets du MNL), "
                        "--ab-variable <une des 21 variables du contrat>"),
        }
    e_llm = arc_elasticities(llm_before, llm_after)
    e_mnl = arc_elasticities(mnl_before, mnl_after)
    rows, skipped = [], 0
    for mode in PREDICTABLE_CATS:
        a, b = e_llm.get(mode), e_mnl.get(mode)
        if a is None or b is None or math.isnan(a) or math.isnan(b) or abs(b) < 1e-6:
            skipped += 1
            continue
        rows.append({"mode": mode, "elasticite_llm": a, "elasticite_mnl": b,
                     "accord": (a > 0) == (b > 0)})
    if not rows:
        return {"s_C2": None, "mesure": "non mesuré",
                "raison": "aucun mode ne porte une élasticité de signe franc chez l'arbitre",
                "ecartes": skipped}
    disagreements = [r for r in rows if not r["accord"]]
    return {
        "s_C2": 100.0 * len(disagreements) / len(rows),
        "mesure": "part des modes dont le signe d'élasticité contredit l'arbitre",
        "variable": variable,
        "n_modes": len(rows),
        "ecartes": skipped,
        "modes": rows,
    }


# ── Composition ──────────────────────────────────────────────────────────────

def compose(fidelity: Optional[float], s_b: Optional[float], s_c: Optional[float],
            weights: dict) -> dict:
    """Linear `S₂`, and the list of terms it could not include.

    An unmeasured term does not enter the sum **and says so**. Above all it is not worth 0:
    in this project, the absence of measurement yields the perfect score, and that is the
    trap this function is there to close.
    """
    w_b = float(weights.get("accord_mnl", 0.0) or 0.0)
    w_c = float(weights.get("coherence_mnl", 0.0) or 0.0)
    terms = [{"terme": "fidelite", "poids": 1.0, "score": fidelity},
             {"terme": "accord_mnl", "poids": w_b, "score": s_b},
             {"terme": "coherence_mnl", "poids": w_c, "score": s_c}]
    missing = [t["terme"] for t in terms if t["score"] is None and t["poids"] > 0]
    total = None
    if fidelity is not None and not missing:
        total = fidelity + w_b * (s_b or 0.0) + w_c * (s_c or 0.0)
    return {
        "S2": total,
        "termes": terms,
        "poids": {"accord_mnl": w_b, "coherence_mnl": w_c},
        "non_mesures_bloquants": missing,
        "note": ("composite linéaire : promouvoir un poids se rétro-applique par simple "
                 "addition de w·s au composite stocké, sans rejouer une décision"),
    }


# ── Substrate: two oracles, a single run ─────────────────────────────────────

def _substrate_problems(label: str, meta: dict, run_path: Optional[str],
                        moves_sha: Optional[str]) -> list[str]:
    """Substrate mismatches of an oracle: missing file, other run, other `moves.csv`."""
    if meta.get("error"):
        return [f"{label} : {meta['error']}"]
    problems = []
    if run_path and meta.get("run") != run_path:
        problems.append(f"{label} : mesuré sur {meta.get('run')}, run épinglé {run_path}")
    if moves_sha and meta.get("moves_sha256") != moves_sha:
        problems.append(f"{label} : empreinte moves.csv divergente "
                        f"({str(meta.get('moves_sha256'))[:12]}… vs {moves_sha[:12]}…)")
    return problems


def check_substrate(mnl_meta: dict, lgb_meta: dict, run_path: Optional[str],
                    moves_sha: Optional[str], klr_meta: Optional[dict] = None) -> dict:
    """Refuses a numerator and a denominator measured on two substrates (R9).

    The fingerprint is not redundant with the run name: a hot resume rewrites
    `moves.csv` **in the same folder**, hence under the same name. Only the fingerprint
    tells these two states apart.

    The **third oracle is checked separately** (`klr_ok`): a KLR missing or measured on another
    run must deprive block C of its second arbiter, not deprive block B of its
    figure. A safeguard that cancels more than it protects ends up being bypassed.
    """
    problems = []
    for label, meta in (("mnl", mnl_meta), ("booster", lgb_meta)):
        problems += _substrate_problems(label, meta, run_path, moves_sha)
    if (mnl_meta.get("spec_version") is not None
            and mnl_meta.get("spec_version") != lgb_meta.get("spec_version")):
        problems.append("les deux oracles n'ont pas le même contrat de variables")

    klr_problems: list[str] = []
    if klr_meta is not None:
        klr_problems = _substrate_problems("klr", klr_meta, run_path, moves_sha)
        if (klr_meta.get("spec_version") is not None
                and klr_meta.get("spec_version") != mnl_meta.get("spec_version")):
            klr_problems.append("le troisième oracle n'a pas le contrat de variables des "
                                "deux autres")
    return {
        "ok": not problems,
        "problemes": problems,
        "klr_ok": klr_meta is not None and not klr_problems,
        "problemes_klr": klr_problems,
        "run": run_path,
        "moves_sha256": moves_sha,
        "oracles": {
            "mnl": {"format": mnl_meta.get("policy_format"),
                    "sha256": mnl_meta.get("policy_sha256"),
                    "genere_le": mnl_meta.get("policy_generated_at"),
                    "chemin": mnl_meta.get("policy_path")},
            "booster": {"format": lgb_meta.get("policy_format"),
                        "sha256": lgb_meta.get("policy_sha256"),
                        "genere_le": lgb_meta.get("policy_generated_at"),
                        "chemin": lgb_meta.get("policy_path")},
            **({"klr": {"format": (klr_meta or {}).get("policy_format"),
                        "sha256": (klr_meta or {}).get("policy_sha256"),
                        "genere_le": (klr_meta or {}).get("policy_generated_at"),
                        "chemin": (klr_meta or {}).get("policy_path")}}
               if klr_meta is not None else {}),
        },
    }


def fidelity_composite(moves: list[dict], manifest, metric: str) -> tuple[Optional[float], str]:
    """Fidelity composite score of strand 1, computed by the scoring formula.

    Never reimplemented here: the published figure must be exactly the formula's
    (`formule_score`). Formula unavailable → `None` and the reason, not a zero.
    """
    module, error = import_formule_score()
    if module is None:
        return None, f"formule de score indisponible ({error})"
    cerema_path = manifest.path_of("cerema")
    if cerema_path is None or not cerema_path.exists():
        return None, "référence EMC² introuvable"
    cerema = frames.load_cerema(cerema_path)
    scorer = frames.Scorer(module, manifest.get("score.weights", {}), metric,
                           manifest.get("score.secondary"))
    rows = frames.simulation_frames(moves)["attendu"]
    scores = scorer.score(rows, cerema)
    composite = (scores.get(metric) or {}).get("composite")
    return (float(composite) if composite is not None else None), ""


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="sources manifest (default: sources.yaml)")
    parser.add_argument("--out", help="output JSON (default: data/bi_oracle.json)")
    parser.add_argument("--ab-variable", help="perturbed variable of the A/B pair (block C2)")
    parser.add_argument("--ab-llm-before", help="moves.csv of the before arm")
    parser.add_argument("--ab-llm-after", help="moves.csv of the after arm")
    parser.add_argument("--ab-mnl-before", help="MNL parquet of the before arm")
    parser.add_argument("--ab-mnl-after", help="MNL parquet of the after arm")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    exclude = manifest.get("common_set.exclude_selection_methods", [])
    run = frames.resolve_run(manifest)
    if not run.get("exists") or not run.get("moves", {}).get("exists"):
        print(f"[erreur] Run not found or without moves.csv: "
              f"{manifest.get('common_set.run')}", file=sys.stderr)
        return 2
    moves_path = REPO_ROOT / run["moves"]["path"]
    moves, stats = frames.read_moves(moves_path, exclude)

    llm, llm_skipped = llm_distributions(moves)
    mnl_path = manifest.path_of("arms.model_mnl.predictions")
    lgb_path = manifest.path_of("arms.model.predictions")
    klr_path = manifest.path_of("arms.model_klr.predictions")
    mnl, mnl_meta = oracle_distributions(mnl_path) if mnl_path else ({}, {"error": "chemin absent du manifeste"})
    lgb, lgb_meta = oracle_distributions(lgb_path) if lgb_path else ({}, {"error": "chemin absent du manifeste"})
    klr, klr_meta = (oracle_distributions(klr_path) if klr_path
                     else ({}, {"error": "chemin absent du manifeste"}))

    print(f"Pinned run: {run['path']}")
    print(f"Comparable decisions (offer of 2 modes or more): prompt {len(llm)} | "
          f"MNL {len(mnl)} | booster {len(lgb)} | KLR {len(klr)} — set aside, prompt side: "
          f"{llm_skipped} (out of {stats.get('total')} log lines)")

    substrate = check_substrate(mnl_meta, lgb_meta, run.get("path"),
                                (run.get("moves") or {}).get("sha256"), klr_meta)
    if not substrate["ok"]:
        # Loud and non-blocking: the output is written with its « non mesuré » blocks,
        # because a page without a figure is more useful than a figure without a substrate.
        for problem in substrate["problemes"]:
            print(f"[ALARME] Divergent substrate — {problem}", file=sys.stderr)

    n_perimeter = len(moves)
    b = (block_b(llm, mnl, lgb, n_perimeter=n_perimeter, llm_skipped=llm_skipped)
         if substrate["ok"]
         else {"s_B": None, "mesure": "non mesuré",
               "raison": "substrat divergent entre les deux oracles et le run épinglé",
               "problemes": substrate["problemes"]})
    # Third column: the SAME computation, the prompt's agreement measured against the KLR.
    # Published next to block B, never in its place — the two arbiters are not
    # interchangeable.
    b_klr = (block_b(llm, klr, lgb, n_perimeter=n_perimeter, llm_skipped=llm_skipped,
                     arbitre="klr")
             if substrate["ok"] and substrate["klr_ok"] and klr
             else {"s_B": None, "mesure": "non mesuré", "arbitre": "klr",
                   "raison": ("troisième oracle absent du manifeste ou de son substrat : "
                              + "; ".join(substrate["problemes_klr"] or ["parquet vide"]))})

    # Second arbiter of block C, only if its substrate is the same: an arbiter measured
    # on another run would settle signs that are not those of these decisions.
    second_arbiter = klr if (substrate["ok"] and substrate["klr_ok"] and klr) else None
    if substrate["ok"] and not second_arbiter:
        print("⚠ Bloc C à un seul arbitre — le second est indisponible : "
              + "; ".join(substrate["problemes_klr"] or ["parquet KLR vide"])
              + " (make klr && make klr-predict)", file=sys.stderr)
    c1 = (block_c1(llm, mnl, second_arbiter) if substrate["ok"]
          else {"s_C1": None, "mesure": "non mesuré",
                "raison": "substrat divergent entre l'arbitre et le run épinglé"})

    ab = {}
    if args.ab_variable:
        before, _ = (oracle_distributions(Path(args.ab_mnl_before))
                     if args.ab_mnl_before else ({}, {}))
        after, _ = (oracle_distributions(Path(args.ab_mnl_after))
                    if args.ab_mnl_after else ({}, {}))
        llm_before = llm_after = None
        if args.ab_llm_before and args.ab_llm_after:
            rows_before, _ = frames.read_moves(Path(args.ab_llm_before), exclude)
            rows_after, _ = frames.read_moves(Path(args.ab_llm_after), exclude)
            llm_before = llm_distributions(rows_before)
            llm_after = llm_distributions(rows_after)
        ab = {"llm_before": llm_before, "llm_after": llm_after,
              "mnl_before": before or None, "mnl_after": after or None}
    c2 = block_c2(ab.get("llm_before"), ab.get("llm_after"), ab.get("mnl_before"),
                  ab.get("mnl_after"), args.ab_variable)

    metric = manifest.get("score.metric", "emd_jsd")
    fidelity, fidelity_error = fidelity_composite(moves, manifest, metric)
    if fidelity is None:
        print(f"⚠ Composite de fidélité indisponible : {fidelity_error}", file=sys.stderr)

    # `s_C` aggregates the two measured sub-blocks — simple mean, both being already
    # disagreement shares in percent. Neither fills in for the absence of the other.
    measured = [v for v in (c1.get("s_C1"), c2.get("s_C2")) if v is not None]
    s_c = sum(measured) / len(measured) if measured else None

    composition = compose(fidelity, b.get("s_B"), s_c, manifest.get("score.bi_oracle", {}))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "substrat": substrate,
        "perimetre": {
            "n_decisions_prompt": len(llm),
            "n_decisions_mnl": len(mnl),
            "n_decisions_booster": len(lgb),
            "n_decisions_klr": len(klr),
            "n_decisions_perimetre": n_perimeter,
            "ecartees_prompt": llm_skipped,
            "exclude_selection_methods": exclude,
            "lecture_moves": stats.get("total"),
        },
        "bloc_A_fidelite": {"metric": metric, "composite": fidelity,
                            "raison": fidelity_error or None},
        "bloc_B_accord_mnl": b,
        "bloc_B_accord_klr": b_klr,
        "bloc_C_coherence": {"s_C": s_c, "C1_axe_distance": c1, "C2_elasticites_ab": c2},
        "composition": composition,
        "avertissement": ("le MNL et la KLR sont des arbitres comportementaux, pas des "
                          "cibles de fidélité : s_B et s_C mesurent un ACCORD, jamais une "
                          "erreur ; les poids restent à 0 (K12)"),
    }

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "scripts" / "synthesis" / "data" / "bi_oracle.json")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"\nBlock A — fidelity ({metric}): "
          + ("not measured" if fidelity is None else f"{fidelity:.2f}"))
    if b.get("s_B") is None:
        print(f"Block B — agreement with the MNL: not measured ({b.get('raison')})")
    else:
        print(f"Block B — agreement with the MNL: {b['s_B']:.1f} % of the inter-oracle "
              f"distance (JSD prompt {b['jsd_prompt_mnl_bits_mean']:.4f} bit, booster "
              f"{b['jsd_booster_mnl_bits_mean']:.4f} bit, on {b['n_decisions']} "
              f"decisions, coverage {100 * b['couverture']:.0f} %)")
        print(f"        KL variant (depends on the floor ε = {b['epsilon']:g}): "
              f"{b['s_B_kl']:.0f} % — floor applied on {b['n_plancher_applique']} "
              f"decisions, prompt near-degenerate on {b['n_prompt_degenere']}")
    if b_klr.get("s_B") is None:
        print(f"Block B' — agreement with the KLR: not measured ({b_klr.get('raison')})")
    else:
        print(f"Block B' — agreement with the KLR: {b_klr['s_B']:.1f} % of the inter-oracle "
              f"distance (JSD prompt {b_klr['jsd_prompt_klr_bits_mean']:.4f} bit, "
              f"booster {b_klr['jsd_booster_klr_bits_mean']:.4f} bit, on "
              f"{b_klr['n_decisions']} decisions)")
    if c1.get("s_C1") is None:
        print(f"Block C1 — direction of change: not measured ({c1.get('raison')})")
    else:
        print(f"Block C1 — direction of change: {c1['s_C1']:.1f} % disagreement "
              f"({c1['n_desaccords']}/{c1['n_transitions']} transitions, "
              f"{c1['ecartees']} set aside) — {c1['regime']}")
    s_c2 = c2.get("s_C2")
    print("Block C2 — A/B elasticities: "
          + ("not measured" if s_c2 is None else f"{s_c2:.1f} % disagreement"))
    total = composition["S2"]
    print(f"\nS₂ (weights {composition['poids']}): "
          + ("not measured" if total is None else f"{total:.2f}"))
    rel = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
    print(f"Written: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
