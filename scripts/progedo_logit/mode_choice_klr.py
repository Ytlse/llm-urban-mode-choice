"""mode_choice_klr.py — Contract and evaluator of the third family (RBF-kernel KLR).

Same division of roles as for the logit: this module cannot *estimate*, it can
**read back and predict**. Estimation needs scikit-learn, prediction needs only
numpy, and prediction is what runs in the consumers (common set, experiment
decision-maker, `controller` container).

**What this family brings, and why it was missing.** The repository held the two
ends of an axis: an exact, non-smooth booster, and a smooth, less exact logit. Kernel
logistic regression is non-linear like the first and has a smooth response like the
second — it is the family Martín-Baos et al. (2023) name as the best
trade-off between predictive accuracy and behavioural plausibility. It therefore serves
twice: a third figure at level 3 of the ablation, and a **second arbiter** for block
C of the two-oracle score, where a single arbiter cannot be refuted.

**Nyström, and why there is no choice.** An exact kernel would require the Gram matrix
of the train set: 39,203², i.e. 1.5 billion entries. We therefore project onto `m` landmarks
(500 to 2,000), drawn **per household** — `m` distinct households, one trip drawn in each. A
per-trip draw would concentrate the landmarks on large households, which carry up to 38
trips, and the variable space would be described where a few households live.

**The dual coefficients, and why the artefact is small.** Estimation fits a multinomial
logit on the Nyström map `Φ = K(Z,L)·W`, where `W = U Λ^{-1/2}` is the inverse
square root of the landmarks' Gram matrix. Predicting does not need `W`: the two matrices
fold together, and

    softmax(Φ·βᵀ + α) = softmax(K(Z,L)·(W·βᵀ) + α) = softmax(K(Z,L)·B + α)

with `B` of shape `m × 4`. The artefact thus carries the landmarks, `B`, `γ` and the constants —
`m × 54` floats, i.e. 1.2 MB at `m = 1,000`, where carrying `W` would have made 20. The
folding is **exact** (checked at 1e-9 before writing, measured at 3e-14), and `B` is
exactly what are called dual coefficients: one weight per landmark and per
alternative.

**The design matrix is not rebuilt.** It comes from `mode_choice_logit`, through the same
function as the logit: an RBF kernel only makes sense on standardised variables, and
missing values are materialised there by the same declared indicators. Redoing the encoding
here would have produced a silent shift — the probabilities would have stayed plausible.

**Confidentiality.** The landmarks are `m` **real survey rows**, centred and
scaled. The artefact therefore stays under `scripts/progedo_logit/*.json`, gitignored like the
other two: the PROGEDO source is restricted-access (lil-1750) and is not republished by
ricochet in a model file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.progedo_logit.mode_choice_logit import (
    design_matrix_from_contract,
    softmax,
)

# Format of the third family's artefact. Describes the *structure* of the JSON,
# independently of `spec_version`, which describes the variable contract. A consumer
# refuses a format it does not know rather than interpreting its keys at random.
KLR_FORMAT = "klr_mode_choice_policy"
KLR_FORMAT_VERSION = 1

#: Size of the row blocks for kernel evaluation. The `n × m` product is never
#: allocated in full: at 2,000 landmarks, predicting 500,000 rows at once would need 8 GB. The
#: chunking changes no value (checked by test), it bounds the decision-maker's memory.
KERNEL_CHUNK = 4096

#: Relative floor of the spectrum of the landmarks' Gram matrix. Two almost
#: coincident landmarks make `K_LL` numerically singular; the corresponding component is
#: **dropped and counted** rather than inverted, which would produce huge coefficients and
#: a model that blows up outside the train set.
EIGENVALUE_FLOOR = 1e-10


def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared distances between each row of `A` and each row of `B`.

    Expanded form `‖a‖² + ‖b‖² − 2a·b`: an explicit loop over 39,203 × 2,000 pairs
    would cost minutes where the matrix product costs a fraction of a second. The
    values are clamped at zero from below — catastrophic cancellation can yield
    −1e-13 on the diagonal, and a negative square root does not exist.
    """
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def rbf_kernel(Z: np.ndarray, landmarks: np.ndarray, gamma: float,
               chunk: int = KERNEL_CHUNK) -> np.ndarray:
    """`K[i, j] = exp(−γ‖z_i − l_j‖²)`, by row blocks."""
    Z = np.asarray(Z, dtype="float64")
    landmarks = np.asarray(landmarks, dtype="float64")
    out = np.empty((len(Z), len(landmarks)), dtype="float64")
    for start in range(0, len(Z), chunk):
        stop = min(start + chunk, len(Z))
        out[start:stop] = np.exp(-gamma * squared_distances(Z[start:stop], landmarks))
    return out


def nystrom_basis(gram: np.ndarray,
                  floor: float = EIGENVALUE_FLOOR) -> tuple[np.ndarray, dict]:
    """Inverse square root `W = U Λ^{-1/2}` of the landmarks' Gram matrix, and its diagnostic.

    `eigh` rather than `svd`: `K_LL` is symmetric positive definite by construction, and
    `eigh` knows it. The diagnostic (effective rank, dropped components, condition number)
    goes up into the artefact: a rank below the announced `m` describes a poorer model
    than its label, and that is the kind of fact nobody notices afterwards.
    """
    values, vectors = np.linalg.eigh(np.asarray(gram, dtype="float64"))
    largest = float(values.max())
    kept = values > floor * largest
    rank = int(kept.sum())
    if rank == 0:
        raise ValueError(
            "[ALARME] Degenerate landmark Gram matrix: no eigenvalue above "
            f"the floor {floor:g} × {largest:.3e}. Kernel unusable, nothing is published.")
    basis = vectors[:, kept] / np.sqrt(values[kept])
    diagnostic = {
        "m": int(len(values)),
        "rank": rank,
        "dropped_eigencomponents": int(len(values) - rank),
        "eigenvalue_floor": floor,
        "eigenvalue_max": largest,
        "eigenvalue_min_kept": float(values[kept].min()),
        "condition_number": float(largest / values[kept].min()),
    }
    return basis, diagnostic


class KLRPredictor:
    """KLR-Nyström evaluator, without scikit-learn.

    Deliberately exposes the same surface as the LightGBM booster and the logit
    (``predict``, ``feature_name``): the three families then take **the same prediction
    path** on the common set, which is the condition for their parquets
    to be comparable (rule K9 of `specs/ticket_043/klr-troisieme-famille.md`).
    """

    def __init__(self, artefact: dict):
        self.artefact = artefact
        klr = artefact["klr"]
        self.contract = klr["design"]
        self.gamma = float(klr["kernel"]["gamma"])
        self.landmarks = np.asarray(klr["landmarks"], dtype="float64")
        self.dual_coef = np.asarray(klr["dual_coef"], dtype="float64")
        self.intercept = np.asarray(klr["intercept"], dtype="float64")
        self.classes = list(artefact["target"]["classes"])
        self.names = [f["name"] for f in artefact["features"]]

        n_columns = len(self.contract["design_columns"])
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != n_columns:
            raise ValueError(
                f"Landmarks of shape {self.landmarks.shape}, expected (m, {n_columns}) "
                "— a landmark described in another space than the design matrix.")
        if self.dual_coef.shape != (len(self.landmarks), len(self.classes)):
            raise ValueError(
                f"Coefficients duaux de forme {self.dual_coef.shape}, attendu "
                f"({len(self.landmarks)}, {len(self.classes)}).")
        if self.intercept.shape != (len(self.classes),):
            raise ValueError(
                f"Intercepts of shape {self.intercept.shape}, expected ({len(self.classes)},).")
        if not self.gamma > 0:
            raise ValueError(f"Invalid kernel width: γ = {self.gamma}.")

    # ``num_iteration`` has no meaning here: accepted and ignored, so that the caller does
    # not have to know which family it holds.
    def predict(self, matrix: Any, num_iteration: Any = None) -> np.ndarray:
        """Probabilities over the four classes, in the spec order."""
        design = design_matrix_from_contract(
            matrix, self.artefact["features"], self.contract)
        scores = np.empty((len(design), len(self.classes)), dtype="float64")
        for start in range(0, len(design), KERNEL_CHUNK):
            stop = min(start + KERNEL_CHUNK, len(design))
            kernel = np.exp(-self.gamma * squared_distances(
                design[start:stop], self.landmarks))
            scores[start:stop] = kernel @ self.dual_coef + self.intercept
        return softmax(scores)

    def feature_name(self) -> list[str]:
        """Expected variables, in the spec order — same contract as the other two."""
        return list(self.names)
