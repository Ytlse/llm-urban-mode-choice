"""Figure G.5 — confusion matrices of three decision-makers on the trip-level audit.

Appendix G of the short paper. No LLM call and no simulation: the script reads the
measured confusion counts (`matrice_effectifs`) of `audit_12_decideurs.json`, normalises each row
(reported mode) so that the diagonal reads as recall, and prints the count in every cell.

    python3 scripts/annexes/fig_G_confusion.py [--input <audit json>] [--sortie <png>]

Input
    `--input`, by default the cited trace `audit_12_decideurs.json` (shipped with the cited traces)
Output
    `--sortie`, by default `$PAPER_DIR/article-court/appendices/images/G5_confusion_matrices.png`,
    or `outputs/figures/G5_confusion_matrices.png` without the papers repository (ticket 113)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402

SOURCE = REPO / "docs/traces/2026-09-22_lot0_entropie_support_unique/audit_12_decideurs.json"
# In the papers repository (PAPER_DIR, ticket 115).
SORTIE = sortie_papier("article-court", "appendices", "images", "G5_confusion_matrices.png")

# Same three decision-makers as Figure 4 of the main paper, same order as Tables G.9.
DECIDEURS = [
    ("lgbm", "Gradient boosting"),
    ("gemini-35-fl_proexp05", "gemini-3.5, expert prompt"),
    ("jev-1130_proexp32", "Typed classifier, its expert prompt"),
]
# Row and column order of `matrice_effectifs`, checked against the keys of `par_mode`.
MODES = ["bike", "car", "transit", "walk"]
LIBELLES = ["Bike", "Car", "Public\ntransport", "Walking"]

log = logging.getLogger("fig_G_confusion")


def main(argv: list[str] | None = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parseur.add_argument("--input", type=Path, default=SOURCE, help="audit JSON (audit_12_decideurs.json)")
    parseur.add_argument("--sortie", type=Path, default=SORTIE, help="PNG to write")
    args = parseur.parse_args(argv)
    source, sortie = args.input, args.sortie
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exiger_depot_papiers(sortie)
    depart = time.monotonic()
    if not source.is_file():
        log.error("[ALARME] input not found: %s (pass --input)", source)
        return 1
    donnees = json.loads(source.read_text(encoding="utf-8"))
    log.info("Start: %d decision-makers read from %s", len(donnees), source)

    fig, axes = plt.subplots(1, len(DECIDEURS), figsize=(13, 4.6), constrained_layout=True)
    for ax, (cle, titre) in zip(axes, DECIDEURS):
        bloc = donnees.get(cle)
        if bloc is None:
            log.error("[ALARME] decision-maker %s missing from %s", cle, source.name)
            return 1
        if list(bloc["par_mode"]) != MODES:
            log.error("[ALARME] %s: mode order %s differs from %s", cle, list(bloc["par_mode"]), MODES)
            return 1
        effectifs = np.array(bloc["matrice_effectifs"], dtype=float)
        parts = effectifs / effectifs.sum(axis=1, keepdims=True)
        for i, mode in enumerate(MODES):
            rappel = bloc["par_mode"][mode]["rappel"]
            if abs(parts[i, i] - rappel) > 1e-9:
                log.error("[ALARME] %s %s: diagonal %.4f != recall %.4f", cle, mode, parts[i, i], rappel)
                return 1
        ax.imshow(parts, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(MODES)):
            for j in range(len(MODES)):
                couleur = "white" if parts[i, j] > 0.55 else "#1f2933"
                ax.text(j, i, f"{parts[i, j] * 100:.1f}%\n({int(effectifs[i, j]):,})",
                        ha="center", va="center", fontsize=9, color=couleur)
        ax.set_xticks(range(len(MODES)), LIBELLES, fontsize=9)
        ax.set_yticks(range(len(MODES)), LIBELLES, fontsize=9)
        ax.set_xlabel("Retained mode")
        ax.set_title(f"{titre}\n{int(effectifs.sum()):,} scored trips", fontsize=10, loc="left")
        log.info("%s: %d trips, recalls %s", cle, effectifs.sum(),
                 ", ".join(f"{m} {parts[i, i]:.3f}" for i, m in enumerate(MODES)))
    axes[0].set_ylabel("Reported mode")

    sortie.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sortie, dpi=200, facecolor="white")
    plt.close(fig)
    log.info("Success: %s written in %.1f s", sortie, time.monotonic() - depart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
