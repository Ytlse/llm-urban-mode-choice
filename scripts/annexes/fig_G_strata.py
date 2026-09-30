"""Figures G.1 to G.4 — mode shares by stratum on the sealed cohort c1, without in-image titles.

Appendix G of the short paper. No LLM call and no simulation: the script re-reads
the `scores.json` of the executions published in Table 1 of the main paper, and draws them
with the plotting functions of `scripts/analysis/plot_decision_makers.py` (panels `ch99_*`), so the
figures keep the layout already reviewed. Two things change:

* every score is read from a PINNED execution (the one of Table 1), never "the latest";
* the figure carries no title — the caption lives in the Markdown, as the form rules require.

The reference curve of each panel is the tabular method with the lowest person-weighted
stratum L1 on that dimension (`_meilleure_reference` of plot_decision_makers).

Run with the service venv (plot_decision_makers imports the `experiences` package):

    services/llm-agents/.venv/bin/python scripts/annexes/fig_G_strata.py

Outputs
    docs/paper/article-court/appendices/images/G1_modes_distance.png
    docs/paper/article-court/appendices/images/G2_modes_purpose.png
    docs/paper/article-court/appendices/images/G3_modes_occupation.png
    docs/paper/article-court/appendices/images/G4_car_share_six_dimensions.png
    scripts/annexes/data/fig_G_strata.json   (every plotted share, per stratum)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services" / "llm-agents"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import scripts.analysis.plot_decision_makers as P  # noqa: E402
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402

log = logging.getLogger("fig_G_strata")

EXPERIENCES = REPO / "archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences"
# In the papers repository (PAPER_DIR, ticket 115).
IMAGES = sortie_papier("article-court", "appendices", "images")
DONNEES = REPO / "scripts/annexes/data/fig_G_strata.json"
S = P.SUFFIXE + "_c"

# The executions of Table 1 of the main paper (thirteen decision-makers drawn by plot_decision_makers).
PINNED = {
    f"exp_alea{S}_nosim": "2026-09-16_13_35_28",
    f"exp_majvoiture{S}_nosim": "2026-09-16_13_36_34",
    f"exp_durmin{S}_nosim": "2026-09-16_13_35_59",
    f"exp_mistral-l-25_promin02{S}_t0_nosim": "2026-09-17_01_13_43",
    f"exp_gemini-31-fl_promin02{S}_t0_nosim": "2026-09-17_00_04_49",
    f"exp_gemini-35-fl_promin02{S}_t0_nosim": "2026-09-16_19_05_45",
    f"exp_gemini-31-fl_proexp05{S}_t0_nosim": "2026-09-17_00_51_26",
    f"exp_mistral-l-25_proexp05{S}_t0_nosim": "2026-09-17_02_47_58",
    f"exp_gemini-35-fl_proexp05{S}_t0_nosim": "2026-09-16_22_20_25",
    f"exp_rf{S}_nosim": "2026-09-16_15_52_59",
    f"exp_mnl{S}_nosim": "2026-09-16_13_43_16",
    f"exp_klr{S}_nosim": "2026-09-16_13_38_33",
    f"exp_lgbm{S}_nosim": "2026-09-16_13_40_53",
}

FIGURES = [
    # (output name, kind, dimension(s))
    ("G1_modes_distance", "modes", "distance"),
    ("G2_modes_purpose", "modes", "motif"),
    ("G3_modes_occupation", "modes", "occupation"),
    ("G4_car_share_six_dimensions", "car",
     ["distance", "motif", "age", "occupation", "lieu_residence", "type_logement"]),
]


def score_epingle(experience: str) -> dict | None:
    """The `scores.json` of the pinned execution; None (logged as ERROR) if absent."""
    execution = PINNED.get(experience)
    if execution is None:
        log.error("[ALARME] No pinned execution for %s: decision-maker left out", experience)
        return None
    fichier = EXPERIENCES / experience / "executions" / execution / "scores.json"
    if not fichier.is_file():
        log.error("[ALARME] Pinned scores.json missing: %s", fichier)
        return None
    donnees = json.loads(fichier.read_text(encoding="utf-8"))
    donnees["_execution"] = execution
    return donnees


def ecrire_png(figure, sortie: Path, nom: str) -> list[Path]:
    """PNG only, into the appendix image folder."""
    sortie.mkdir(parents=True, exist_ok=True)
    # A single panel legend (G.1 to G.3) sits inside the first axis and can hide a curve:
    # move it under the panels. Several legends (G.4) differ by panel and stay where they are.
    avec_legende = [ax for ax in figure.axes if ax.get_legend() is not None]
    if len(avec_legende) == 1:
        ax = avec_legende[0]
        poignees, libelles = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        figure.legend(poignees, libelles, loc="upper center", ncol=len(libelles),
                      bbox_to_anchor=(0.5, 0.0), frameon=False)
    chemin = sortie / f"{nom}.png"
    figure.savefig(chemin, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [chemin]


def exporter(decideurs: list[dict]) -> dict:
    """Every share the figures draw, so each number of a caption can be checked."""
    dims = sorted({d for _, _, x in FIGURES for d in ([x] if isinstance(x, str) else x)})
    sortie: dict = {"executions": PINNED, "decideurs": {}}
    for decideur in decideurs:
        detail = decideur["score"]["detail"]
        sortie["decideurs"][decideur["cle"]] = {
            dim: [
                {"cat": s["cat"], "n": s["n"], "covered": s.get("covered"),
                 "actual": s.get("actual"), "target": s.get("target"), "l1": s.get("l1")}
                for s in detail.get(dim, {}).get("strates", [])
            ]
            for dim in dims
        }
    sortie["meilleure_reference"] = {
        dim: (P._meilleure_reference(decideurs, dim) or {}).get("label") for dim in dims
    }
    return sortie


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exiger_depot_papiers(IMAGES)
    depart = time.monotonic()
    log.info("Start: %d pinned executions under %s", len(PINNED), EXPERIENCES)

    P.dernier_score = score_epingle          # pinned executions, never "the latest"
    P.ecrire = ecrire_png                    # PNG into the appendix folder
    Figure.suptitle = lambda self, *a, **k: None  # no in-image title

    decideurs = P.lire_decideurs()
    if len(decideurs) != len(P.DECIDEURS):
        log.error("[ALARME] %d decision-makers read out of %d: figures not written",
                  len(decideurs), len(P.DECIDEURS))
        return 1

    ecrits: list[Path] = []
    for nom, genre, dimension in FIGURES:
        if genre == "modes":
            ecrits += P._planche_modes(decideurs, IMAGES, nom, dimension, titre="")
        else:
            ecrits += P._planche_parts(decideurs, IMAGES, nom, dimension,
                                       avec_minimal=True, titre="")
    if len(ecrits) != len(FIGURES):
        log.error("[ALARME] %d figures written out of %d", len(ecrits), len(FIGURES))
        return 1

    DONNEES.parent.mkdir(parents=True, exist_ok=True)
    DONNEES.write_text(json.dumps(exporter(decideurs), indent=1, ensure_ascii=False),
                       encoding="utf-8")
    for chemin in ecrits:
        log.info("Written: %s (%.0f kB)", chemin, chemin.stat().st_size / 1024)
    log.info("Success: %d figures and %s in %.1f s", len(ecrits),
             DONNEES.relative_to(REPO), time.monotonic() - depart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
