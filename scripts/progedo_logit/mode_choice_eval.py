"""mode_choice_eval.py — Mode choice metrics, shared by the two oracles.

This module exists for a single reason: **the LightGBM booster and the multinomial logit
must be judged by the same code**. Two copied metric tables diverge at the first added
key, and the published comparison then becomes a comparison of two implementations as
much as of two models — the flaw Hillel (2021) points out in the comparative
literature.

The metrics take a **probability matrix**, never a model: that is what lets them apply
to a booster, to a logit, and tomorrow to any decision-maker able to produce a
distribution over the four classes.

**The two families of indicators, and why both.** Martín-Baos et al. (2023) show that
the ranking of models flips depending on the family:

1. *disaggregate* — accuracy, CEL (cross-entropy on the observed label) and
   GMPCA = exp(−CEL), the « geometric mean of the probability of correct assignment ».
   GMPCA reads as a probability, which log-loss does not;
2. *aggregate* — observed versus predicted modal shares, as probability mass and as
   elected mode, and their L1. This is the axis on which our H0 plays out, and the one
   where the logit holds its own against the trees.

Publishing only the first would flatter the booster; publishing only the second would say
nothing about the individual quality of the probabilities.

All metrics are weighted by the survey weights (COEP): unweighted, modal shares are
not representative.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def mode_shares(labels: np.ndarray, weights: np.ndarray, n_classes: int) -> list[float]:
    """Weighted modal shares, from hard labels."""
    total = weights.sum()
    return [float(weights[labels == k].sum() / total) for k in range(n_classes)]


def probability_mass_shares(proba: np.ndarray, weights: np.ndarray) -> list[float]:
    """Weighted modal shares as **probability mass** — what the pipeline consumes."""
    return list((proba * weights[:, None]).sum(axis=0) / weights.sum())


def gmpca(cel: float) -> float:
    """GMPCA = exp(−CEL), geometric mean of the probability of correct assignment.

    Defined on the **observed** label: it is a likelihood measure, not an agreement
    between two models. Against a soft target (another model's probabilities), CEL
    has an irreducible floor equal to that target's entropy, and the corresponding GMPCA
    no longer reads as a probability — cf. :func:`scripts.synthesis.bi_oracle.kl_bits`.
    """
    return float(np.exp(-cel))


def evaluate_proba(proba: np.ndarray, y: np.ndarray, w: np.ndarray,
                   classes: list[str]) -> dict:
    """Test split metrics, for any producer of probabilities.

    Modal shares are reported in **two** ways, because both have a use: as
    probability mass (what the pipeline actually consumes, cf. ticket 005 §4) and
    as elected mode (what an argmax would produce). The second is systematically
    more contrasted — a well-calibrated classifier exaggerates the shares when
    hardened.
    """
    hard = proba.argmax(axis=1)
    k = len(classes)

    observed = mode_shares(y, w, k)
    predicted_hard = mode_shares(hard, w, k)
    predicted_mass = probability_mass_shares(proba, w)

    cm = confusion_matrix(y, hard, labels=list(range(k)), sample_weight=w)
    cm_counts = confusion_matrix(y, hard, labels=list(range(k)))

    # Weighted recall/precision per class. Bicycle (4 % of trips) is the class where
    # calibration plays out: it is the one any reweighting breaks.
    per_class = {}
    for i, name in enumerate(classes):
        tp = cm[i, i]
        support = cm[i, :].sum()
        predicted = cm[:, i].sum()
        per_class[name] = {
            "support_share": float(support / cm.sum()),
            "recall": float(tp / support) if support else None,
            "precision": float(tp / predicted) if predicted else None,
        }

    labels = list(range(k))
    cel_weighted = float(log_loss(y, proba, labels=labels, sample_weight=w))
    cel_unweighted = float(log_loss(y, proba, labels=labels))

    return {
        "n_rows": int(len(y)),
        # `log_loss_*` and `cel_*` are the same quantity, in nats. Both names are
        # published: the first is the repository's historical one, the second that of the
        # comparative literature GMPCA derives from.
        "log_loss_weighted": cel_weighted,
        "log_loss_unweighted": cel_unweighted,
        "cel_weighted": cel_weighted,
        "cel_unweighted": cel_unweighted,
        "gmpca_weighted": gmpca(cel_weighted),
        "gmpca_unweighted": gmpca(cel_unweighted),
        "accuracy_weighted": float(accuracy_score(y, hard, sample_weight=w)),
        "accuracy_unweighted": float(accuracy_score(y, hard)),
        "classes": classes,
        "mode_shares": {
            "observed": observed,
            "predicted_probability_mass": predicted_mass,
            "predicted_argmax": predicted_hard,
            "l1_probability_mass": float(
                np.abs(np.array(predicted_mass) - np.array(observed)).sum()),
            "l1_argmax": float(
                np.abs(np.array(predicted_hard) - np.array(observed)).sum()),
        },
        "per_class": per_class,
        # Rows = true, columns = predicted, in the order of `classes`.
        "confusion_matrix_weighted": [[float(v) for v in row] for row in cm],
        "confusion_matrix_counts": [[int(v) for v in row] for row in cm_counts],
    }


def format_shares(classes: list[str], observed: list[float],
                  mass: list[float], hard: list[float]) -> str:
    lines = [f"  {'mode':10s} {'observé':>9s} {'masse p.':>9s} {'mode élu':>9s}"]
    for i, name in enumerate(classes):
        lines.append(f"  {name:10s} {observed[i]:8.1%} {mass[i]:8.1%} {hard[i]:8.1%}")
    return "\n".join(lines)
