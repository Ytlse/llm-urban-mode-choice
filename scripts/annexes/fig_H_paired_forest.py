"""Figure H.1: the twenty paired differences, with their 95 % interval.

Reads docs/traces/2026-09-21_ticket096_lot2/paired_complet_B2000.json (produced by
docs/traces/2026-09-21_ticket096_lot2/scripts/paired_avec_jev.py: 868 common persons,
2 000 replicates, seed 2026) and draws two panels side by side, EMD-JSD composite and L1 on
the overall shares. No value is written by hand.

Convention: each row is A - B; a negative value means that A has the lower
error. Filled dot: the interval excludes zero. Hollow dot: it contains zero.

Also accepts, through `--input`, the output of `python -m scripts.analysis.paired_intervals
--preset chapter6 --out <dir>` (`<dir>/paired_chapter6_B2000.json`): same pairs, stored
under the key `pairs` (ticket 113).

Usage:
    services/llm-agents/.venv/bin/python scripts/annexes/fig_H_paired_forest.py \
        [--input <json>] [--sortie <png>]
Output: `$PAPER_DIR/article-court/appendices/images/H1_paired_differences.png`, or
`outputs/figures/H1_paired_differences.png` without the papers repository (ticket 113).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402

SOURCE = RACINE / "docs/traces/2026-09-21_ticket096_lot2/paired_complet_B2000.json"
# In the papers repository (`PAPER_DIR`, ticket 115).
SORTIE = sortie_papier("article-court", "appendices", "images", "H1_paired_differences.png")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fig_H1")

# Order and labels of Table H.2 (same numbering).
GROUPES = [
    ("Expert minus minimal prompt, same model", [
        ("g35_exp-g35_min", "1  gemini-3.5"),
        ("g31_exp-g31_min", "2  gemini-3.1"),
        ("mis_exp-mis_min", "3  mistral-large"),
        ("jev_exp-jev_min", "4  typed classifier"),
    ]),
    ("Expert-prompt condition minus reference model", [
        ("g35_exp-lgbm", "5  gemini-3.5 − gradient boosting"),
        ("g35_exp-klr", "6  gemini-3.5 − kernel logistic regr."),
        ("g35_exp-rf", "7  gemini-3.5 − random forest"),
        ("g35_exp-mnl", "8  gemini-3.5 − multinomial logit"),
        ("jev_exp-lgbm", "9  typed classifier − gradient boosting"),
        ("jev_exp-klr", "10  typed classifier − kernel logistic regr."),
        ("jev_exp-rf", "11  typed classifier − random forest"),
        ("jev_exp-mnl", "12  typed classifier − multinomial logit"),
        ("g31_exp-lgbm", "13  gemini-3.1 − gradient boosting"),
        ("mis_exp-lgbm", "14  mistral-large − gradient boosting"),
    ]),
    ("Between decision-makers, same prompt", [
        ("g31_exp-g35_exp", "15  gemini-3.1 − gemini-3.5, expert"),
        ("mis_exp-g35_exp", "16  mistral-large − gemini-3.5, expert"),
        ("g31_exp-mis_exp", "17  gemini-3.1 − mistral-large, expert"),
        ("jev_exp-g35_exp", "18  typed classifier − gemini-3.5, expert"),
        ("g31_min-g35_min", "19  gemini-3.1 − gemini-3.5, minimal"),
        ("g31_min-mis_min", "20  gemini-3.1 − mistral-large, minimal"),
    ]),
]
PANNEAUX = [("comp", "Composite difference (points)"),
            ("l1", "L1 difference on overall shares (percentage points)")]

ENCRE = "#2a78d6"
TEXTE = "#0b0b0b"
TEXTE_2 = "#52514e"
GRILLE = "#d9d8d4"


def main(argv: list[str] | None = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parseur.add_argument("--input", type=Path, default=SOURCE,
                         help="JSON of the paired differences (cited trace, or paired_intervals output)")
    parseur.add_argument("--sortie", type=Path, default=SORTIE, help="PNG to write")
    args = parseur.parse_args(argv)
    source, sortie = args.input, args.sortie
    exiger_depot_papiers(sortie)
    depart = time.time()
    try:
        donnees = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.error("[ALARME] cannot read %s: %s (pass --input)", source, exc)
        return 1
    if isinstance(donnees.get("pairs"), dict):
        # Output of `paired_intervals`: the pairs are under `pairs`.
        log.info("paired_intervals format (preset %s, B=%s)", donnees.get("preset"), donnees.get("replicates"))
        donnees = donnees["pairs"]

    attendues = [cle for _, lignes in GROUPES for cle, _ in lignes]
    manquantes = [cle for cle in attendues if cle not in donnees]
    if manquantes:
        log.error("[ALARME] pairs missing from JSON %s: %s", source.name, manquantes)
        return 1
    log.info("%d pairs read from %s", len(attendues), source)

    # Vertical positions: one row per pair, one empty row + title per group.
    y, positions, titres = 0.0, {}, []
    for titre, lignes in GROUPES:
        titres.append((titre, y))
        y += 1.0
        for cle, libelle in lignes:
            positions[cle] = (y, libelle)
            y += 1.0
        y += 0.5
    hauteur = y

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.2), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    exclut_zero = 0
    for ax, (champ, titre_axe) in zip(axes, PANNEAUX):
        ax.axvline(0, color=TEXTE_2, lw=1.0, zorder=1)
        for cle, (yy, _) in positions.items():
            r = donnees[cle][champ]
            plein = r["lo"] > 0 or r["hi"] < 0
            exclut_zero += plein
            ax.plot([r["lo"], r["hi"]], [yy, yy], color=ENCRE, lw=2, solid_capstyle="round",
                    zorder=2)
            ax.plot(r["moy"], yy, "o", ms=7, mec=ENCRE, mew=1.6,
                    mfc=ENCRE if plein else "white", zorder=3)
        for _, yt in titres[1:]:
            ax.axhline(yt - 0.75, color=GRILLE, lw=0.8, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(5 if champ == "comp" else 10))
        ax.grid(axis="x", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(titre_axe, color=TEXTE, fontsize=10)
        ax.tick_params(colors=TEXTE_2, labelsize=9)
        for cote in ("top", "right", "left"):
            ax.spines[cote].set_visible(False)
        ax.spines["bottom"].set_color(TEXTE_2)

    axes[0].set_yticks([p[0] for p in positions.values()])
    axes[0].set_yticklabels([p[1] for p in positions.values()], fontsize=9, color=TEXTE)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    for titre, yt in titres:
        axes[0].text(-0.02, yt, titre, transform=axes[0].get_yaxis_transform(),
                     ha="right", va="center", fontsize=9.5, fontweight="bold", color=TEXTE)
    axes[0].set_ylim(hauteur - 0.5, -0.8)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=7, mec=ENCRE, mfc=ENCRE, mew=1.6,
                   label="95% interval excludes zero"),
        plt.Line2D([], [], marker="o", ls="", ms=7, mec=ENCRE, mfc="white", mew=1.6,
                   label="95% interval contains zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.67, -0.01))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.98, bottom=0.12)

    sortie.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sortie, dpi=200, facecolor="white")
    log.info("wrote %s: %d intervals out of %d exclude zero, in %.1f s",
             sortie, exclut_zero, 2 * len(attendues), time.time() - depart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
