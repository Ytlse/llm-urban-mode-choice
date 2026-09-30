#!/usr/bin/env python3
"""Figure 5 of the short paper, in English.

WHY THIS FILE EXISTS. `shock_figures.py` composes its figures in French: it serves
the long paper, whose masters are French. The short paper is composed in English, and the
rule is asymmetric — an English figure in the French version bothers no one, a French
figure in the English version shows. Rather than switching the language of the shared
script, which would break the long paper, this module reuses its data reading and only
rewrites the labels.

Output: article-court/overleaf{,-fr}/images/ch7_propension_quotidienne.png, in the papers
repository (`PAPER_DIR`, ticket 115). Both projects receive the same English figure, by the
rule above.
"""
import datetime
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "scripts/analysis"))
from shock_figures import CHOC, COULEUR, SORTIE, TEMOIN, TRAITE, propension_par_jour
from scripts.depot_papiers import chemin_papier, exiger_depot_papiers, paper_dir  # noqa: E402


def figure_propension_en(sorties: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for run, nom in ((TEMOIN, "control"), (TRAITE, "exposed")):
        d = propension_par_jour(run)
        jours = [j for j in sorted(d) if d[j]]
        ax.plot(jours, [statistics.mean(d[j]) for j in jours], marker="o", ms=3.5, lw=1.6,
                color=COULEUR["témoin" if nom == "control" else "traité"], label=nom)
    ax.axvline(CHOC, color="#CE3B4B", ls="--", lw=1.2)
    ax.annotate("suspicious noise in car engine", (CHOC, 103), color="#CE3B4B", fontsize=9, ha="center")
    ax.axvline(SORTIE, color="#333", ls=":", lw=1.2)
    ax.annotate("the account leaves the context", (SORTIE, 103), color="#333", fontsize=9,
                ha="center")
    ax.set_ylim(-5, 115)
    ax.set_ylabel("Stated P(car), daily mean (%)")
    ax.set_title("Daily propensity to the car, stated at every decision")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    for chemin in sorties:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, format=chemin.suffix.lstrip("."), dpi=200, bbox_inches="tight")
        print(f"written: {chemin.relative_to(paper_dir())}")
    plt.close(fig)


if __name__ == "__main__":
    base = chemin_papier("article-court")
    sorties = [
        base / "overleaf/images/ch7_propension_quotidienne.png",
        base / "overleaf-fr/images/ch7_propension_quotidienne.png",
        chemin_papier("figures", "ch7_propension_quotidienne_en.png"),
    ]
    exiger_depot_papiers(*sorties)
    figure_propension_en(sorties)
