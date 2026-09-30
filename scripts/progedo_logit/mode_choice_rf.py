"""mode_choice_rf.py — Contract and evaluator of the random forest control (ticket 044).

This module does not know how to *choose* a forest: it knows how to **rebuild it and make
it predict**, from an artefact that contains no tree.

**Why no tree in the artefact.** The retained forest has 1,200 trees and 3,740,500
nodes. Serialised, it weighs ~150 MB as compact JSON or 165 MB as compressed `joblib` —
against 18.9 MB for the booster, already at the limit of what can be diffed, and 21 KB
for the logit. Such a file cannot be reread or diffed, and a scikit-learn `pickle`
goes stale at the library's next version (which is why the second oracle serialises
coefficients and not a model).

**What the artefact carries instead.** Enough to rebuild *exactly* the same forest: the
retained hyperparameters, the seed, the design-matrix contract, the parquet hash
and the scikit-learn version. Refitting costs **about ten seconds** and returns a
bit-identical model — `random_state` is fixed, `RandomForestClassifier` is
deterministic for a given seed. The model fingerprint (rule R12 of the experiment
platform) becomes the SHA of this small JSON, which fully determines the forest: it is
a more honest provenance than the SHA of a 165 MB pickle, which nobody can check
matches what the artefact declares.

**The two conditions, refused and not bypassed.** Refitting depends on two things that
the artefact freezes and that the predictor **checks before fitting**:

1. the **training matrix**, shipped next to the artefact in a 1.4 MB `.npz` and
   compared by SHA-256. It replaces the parquet for a measured reason: the
   `controller` container, where experiments run, has **neither `pyarrow` nor
   `fastparquet`** and cannot read a parquet. Replay thus only needs **numpy**. The file
   is small because the design matrix is mostly 0s and 1s: 52,248 × 50 values compress to
   1.4 MB, against 165 MB for the forest itself;
2. the **reproduction of the published metrics**. This is where the first version of this
   module picked the wrong safeguard: it compared the *scikit-learn version*, and refused
   to run as soon as it differed. Yet the host has 1.8.0 and the `controller` container,
   where experiments run, 1.9.0 — the control could never have been replayed there, for a
   reason that was only a number. A version number does not say whether the forest has
   changed: it says it *might* have changed. The predictor therefore refits, **predicts on
   the test split and compares accuracy and CEL with those the artefact publishes**.
   Identical → it is the forest of the table, whatever the library. Different → refusal,
   with the gap.

A loud refusal is better than a forest silently different from the one in the table — but
the refusal must be on what changed, not on what might have changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

#: Format of the control's artefact. Describes the *structure* of the JSON, independently
#: of `spec_version` which describes the variable contract.
RF_FORMAT = "rf_mode_choice_policy"
RF_FORMAT_VERSION = 1

#: Maximum tolerated gap between the refitted forest's metrics and the published ones.
#: Tight on purpose: two identical forests give exactly the same figures, and this
#: margin only covers the order of floating-point reductions.
TOLERANCE_REPRODUCTION = 1e-9


def sha256_fichier(chemin: Path) -> str:
    """SHA-256 of a file, read in blocks — the parquet weighs several tens of MB."""
    digest = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            digest.update(bloc)
    return digest.hexdigest()


class RFPredictor:
    """Evaluator of the random forest control — refits, then predicts.

    Deliberately exposes the same surface as the LightGBM booster and ``LogitPredictor``
    (``predict``, ``feature_name``): consumers then call **the same prediction path**
    for all families, which is the condition for their outputs to be
    comparable.
    """

    def __init__(self, artefact: dict, trainset_path: Path | None = None,
                 verifier_metriques: bool = True):
        from scripts.progedo_logit.fit_mode_choice_forest import forest

        self.artefact = artefact
        bloc = artefact["rf"]
        self.classes = list(artefact["target"]["classes"])
        self.names = [f["name"] for f in artefact["features"]]

        chemin = Path(trainset_path) if trainset_path else (
            Path(__file__).resolve().parent / artefact["trainset_file"])
        if not chemin.exists():
            raise FileNotFoundError(
                f"Training matrix missing: {chemin}. The baseline is refitted at "
                "loading (no tree is serialised); it needs the matrix that was used "
                "to estimate it. It is regenerated with "
                "`make forest FOREST_ARGS=--artefact`.")
        empreinte = sha256_fichier(chemin)
        if empreinte != artefact["trainset_sha256"]:
            raise ValueError(
                f"The training matrix has changed since the estimation (SHA "
                f"{empreinte[:12]} versus {artefact['trainset_sha256'][:12]}). The refitted "
                "forest would not be the one whose metrics are published. Re-estimate "
                "with `make forest FOREST_ARGS=--artefact`.")

        with np.load(chemin) as jeu:
            matrix = jeu["X"]
            y = jeu["y"].astype("int64")
            w = jeu["w"]
            est_train = jeu["is_train"]
            self.est_test = jeu["is_test"]
        if matrix.shape[1] != len(bloc["contrat"]["design_columns"]):
            raise ValueError(
                f"Matrix with {matrix.shape[1]} columns, contract with "
                f"{len(bloc['contrat']['design_columns'])}: the artefact and its matrix do "
                "not come from the same estimation.")

        h = bloc["hyperparameters"]
        self.model = forest(h["n_estimators"], h["max_depth"], h["min_samples_leaf"],
                            h["max_features"])
        self.model.fit(matrix[est_train], y[est_train], sample_weight=w[est_train])
        self.n_fit = int(est_train.sum())
        self.sklearn_estimation = bloc["estimator"]["version"]
        self.sklearn_ici = sklearn.__version__
        if verifier_metriques:
            self.controle = self._verifier_reproduction(matrix, y, w)

    def _verifier_reproduction(self, matrix: np.ndarray, y: np.ndarray,
                               w: np.ndarray) -> dict:
        """Does the refitted forest reproduce the published metrics?

        The only safeguard that measures instead of presuming. It costs one prediction on
        the 13,045 test rows, and it holds for any scikit-learn version: what we want to
        know is not « has the library changed » but « has the forest
        changed ».
        """
        from scripts.progedo_logit.fit_mode_choice_forest import proba_complete
        from scripts.progedo_logit.mode_choice_eval import evaluate_proba

        proba = proba_complete(self.model, matrix[self.est_test], len(self.classes))
        obtenu = evaluate_proba(proba, y[self.est_test], w[self.est_test], self.classes)
        publie = self.artefact["metrics"]
        ecarts = {
            "accuracy_weighted": abs(obtenu["accuracy_weighted"]
                                     - publie["accuracy_weighted"]),
            "cel_weighted": abs(obtenu["cel_weighted"] - publie["cel_weighted"]),
        }
        if max(ecarts.values()) > TOLERANCE_REPRODUCTION:
            raise ValueError(
                f"The refitted forest does not reproduce the published metrics: gaps "
                f"{ecarts} (tolerance {TOLERANCE_REPRODUCTION:g}). scikit-learn "
                f"{self.sklearn_ici} here versus {self.sklearn_estimation} at estimation. "
                "It is no longer the forest of the table — re-estimate with `make forest --artefact` "
                "under this version, or predict under the estimation one.")
        return {
            "reproduit": True,
            "ecarts": ecarts,
            "tolerance": TOLERANCE_REPRODUCTION,
            "sklearn_estimation": self.sklearn_estimation,
            "sklearn_execution": self.sklearn_ici,
        }

    def predict(self, matrix: Any, num_iteration: Any = None) -> np.ndarray:
        """Probabilities over the four classes, in the spec's order.

        ``matrix`` is the matrix **already encoded** by ``encode_features`` — the same input
        as ``LogitPredictor``. ``num_iteration`` makes no sense for a forest: accepted and
        ignored, so that the caller need not know which family it holds.
        """
        from scripts.progedo_logit.fit_mode_choice_forest import proba_complete
        from scripts.progedo_logit.mode_choice_logit import design_matrix

        design = design_matrix(matrix, {"features": self.artefact["features"],
                                        "logit": self.artefact["rf"]["contrat"]})
        return proba_complete(self.model, design, len(self.classes))

    def feature_name(self) -> list[str]:
        """Expected variables, in the spec's order — same contract as the other families."""
        return list(self.names)


def charger_rf(path: Path, spec: dict) -> tuple[RFPredictor, dict]:
    """Loads the control from its artefact — signature of ``load_policy``.

    The same three refusals as ``load_policy``: an unknown format, a ``spec_version`` that
    diverges, a variable or class order that is not the spec's. A one-column shift
    gives perfectly plausible probabilities.
    """
    artefact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artefact.get("format") != RF_FORMAT:
        raise ValueError(f"Format d'artefact inattendu : {artefact.get('format')!r}.")
    if artefact.get("spec_version") != spec.get("spec_version"):
        raise ValueError(
            f"Control estimated under variable contract v{artefact.get('spec_version')}, "
            f"the spec read is v{spec.get('spec_version')}. Re-estimate (`make forest`).")
    names = [f["name"] for f in spec["features"]]
    if [f["name"] for f in artefact.get("features") or ()] != names:
        raise ValueError("The variable order of the artefact differs from that of the spec.")
    classes = list(spec["target"]["classes"])
    if list((artefact.get("target") or {}).get("classes") or ()) != classes:
        raise ValueError("The class order of the artefact differs from that of the spec.")
    return RFPredictor(artefact), artefact
