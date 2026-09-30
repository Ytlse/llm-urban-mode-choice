#!/usr/bin/env python3
"""The two figures of the ticket 058 unit audit, read from its JSON output.

The audit confronts nine decision-makers with the declared mode, trip by trip, on the
days actually described by the respondents. Two of its statements cannot be read in a
single number, and § 6.4 carries them as figures (6.5 and 6.6); the full tables,
nine decision-makers and confusion matrix, live in appendix I:

* ``ch6_audit_distance`` — accuracy by distance band. It establishes that the whole
  gap between decision-makers plays out under 2 km, and that beyond 10 km the "always
  the car" floor joins the tabular ceiling;
* ``ch6_audit_modes`` — recall and precision by mode. They say what the accuracy gap
  is paid with: the tiers recall cycling better than any tabular method
  and lose walking.

**The figures are in English**, like all those of the manuscript.

No figure is written by hand. The input JSON is the one from
`scripts/progedo_logit/audit_unitaire_058.py --sortie`; if it is missing, this script reruns
the audit, which rereads the archived runs without a single model call.

Usage:
    services/llm-agents/.venv/bin/python scripts/analysis/plot_audit_unitaire.py
    … [--sortie <dossier>] [--copie <dossier>]

Default output: `figures/` of the papers repository (`PAPER_DIR`, ticket 115). The copy to
`article/images/` went away with that folder, renamed `article_old_do_not_maintain/`:
`--copie` now only copies when asked to.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from scripts.analysis.figures_versionnees import signaler
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier

logger = logging.getLogger("audit058")

AUDIT = RACINE / "scripts" / "progedo_logit" / "audit_unitaire_058.py"
PYTHON = RACINE / "services" / "llm-agents" / ".venv" / "bin" / "python"
JSON_DEFAUT = (RACINE / "docs" / "traces" / "2026-09-21_08-52_ticket058_audit_unitaire"
               / "audit_unitaire_058.json")
SORTIE_DEFAUT = sortie_papier("figures")

# Author's decision of 2026-09-22: the plate now only carries the tabular ceiling and the
# expert prompt of each audited backbone. The minimal prompts and the all-car floor leave
# it — they remain in the tables of appendix I, and the plate that used them, the unit
# agreement by distance band, left the chapter the same day.
DECIDEURS = [
    ("lgbm", "Gradient boosting (LightGBM)", "#1baf7a", "-", "o"),
    ("gemini-35-fl_proexp05", "Gemini 3.5 Flash-Lite, expert prompt", "#2a78d6", "-", "s"),
]

# Ticket 096 — added to DECIDEURS by `--avec-jev` only. Without the flag, the two
# plates remain those of the reference chapter, which only cites four of them.
DECIDEURS_JEV = [
    # prompt_expert_32 is Jev's expert prompt (author's decision, 2026-09-22), and it is
    # the same arm as in § 6.1. prompt_expert_05 is still measured, in appendices H.6 and I.
    ("jev-1130_proexp32", "Jev 1.13 (TypeSafe), expert prompt", "#9b51e0", "-", "D"),
]
BANDES = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km"]
LIBELLES_BANDES = ["0–1 km", "1–2 km", "2–5 km", "5–10 km", "10–20 km", "20–50 km"]
MODES = ["bike", "car", "transit", "walk"]
LIBELLES_MODES = ["Bike", "Car", "Public transport", "Walking"]


def charger(chemin: Path) -> dict:
    """The audit JSON; reruns the audit if it is missing, without a single model call."""
    if not chemin.is_file():
        logger.info("Audit output missing (%s) — rerunning the audit", chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        depart = time.monotonic()
        issue = subprocess.run(
            [str(PYTHON), str(AUDIT), "--sortie", str(chemin)],
            cwd=RACINE, check=False,
        )
        if issue.returncode != 0 or not chemin.is_file():
            raise SystemExit(f"[ALARME] The audit failed (code {issue.returncode}): {chemin}")
        logger.info("Audit replayed in %.1f s", time.monotonic() - depart)
    resultats = json.loads(chemin.read_text(encoding="utf-8"))
    manquants = [c for c, *_ in DECIDEURS if c not in resultats]
    if manquants:
        raise SystemExit(f"[ALARME] Decision-maker(s) missing from the audit: {', '.join(manquants)}")
    logger.info("Audit loaded: %d decision-makers, %d plotted", len(resultats), len(DECIDEURS))
    return resultats


def figure_distance(resultats: dict, sortie: Path) -> list[Path]:
    figure, axe = plt.subplots(figsize=(7.4, 4.3))
    effectifs = [resultats["lgbm"]["par_bande"][b]["n"] for b in BANDES]
    for cle, libelle, couleur, style, marque in DECIDEURS:
        valeurs = [resultats[cle]["par_bande"][b]["exactitude"] * 100 for b in BANDES]
        axe.plot(range(len(BANDES)), valeurs, style, color=couleur, marker=marque,
                 markersize=4.0, linewidth=1.3, label=libelle, zorder=3)
    axe.set_xticks(range(len(BANDES)))
    axe.set_xticklabels([f"{lib}\nn = {n:,}".replace(",", " ")
                         for lib, n in zip(LIBELLES_BANDES, effectifs)], fontsize=8.5)
    axe.set_ylim(30, 95)
    axe.set_ylabel("Agreement with the declared mode (%)", fontsize=9.5)
    axe.set_title("Unit agreement by trip distance", fontsize=10.5, loc="left")
    axe.grid(axis="y", color="#d8d6d1", linewidth=0.7, zorder=1)
    axe.set_axisbelow(True)
    for bord in ("top", "right"):
        axe.spines[bord].set_visible(False)
    axe.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2)
    return ecrire(figure, sortie, "ch6_audit_distance")


def figure_modes(resultats: dict, sortie: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    traces = [d for d in DECIDEURS if d[0] != "majvoiture"]
    largeur = 0.14
    for axe, grandeur, titre in ((axes[0], "rappel", "Recall"), (axes[1], "precision", "Precision")):
        for rang, (cle, libelle, couleur, _, _) in enumerate(traces):
            valeurs = [resultats[cle]["par_mode"][m][grandeur] * 100 for m in MODES]
            positions = [i + (rang - 1) * largeur for i in range(len(MODES))]
            axe.bar(positions, valeurs, largeur, color=couleur, label=libelle, zorder=3)
        axe.set_xticks(range(len(MODES)))
        axe.set_xticklabels(LIBELLES_MODES, fontsize=8.5)
        axe.set_title(titre, fontsize=10, loc="left", pad=8)
        axe.grid(axis="y", color="#d8d6d1", linewidth=0.7, zorder=1)
        axe.set_axisbelow(True)
        for bord in ("top", "right"):
            axe.spines[bord].set_visible(False)
    axes[0].set_ylabel("%", fontsize=9.5)
    axes[0].set_ylim(0, 90)
    figure.suptitle("Recall and precision by transport mode",
                    fontsize=10.5, x=0.02, y=1.02, ha="left")
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left",
                   bbox_to_anchor=(0.0, -0.13), ncol=3)
    return ecrire(figure, sortie, "ch6_audit_modes")


def ecrire(figure, sortie: Path, nom: str) -> list[Path]:
    """Writes the figure as PNG and SVG, and returns the written paths."""
    sortie.mkdir(parents=True, exist_ok=True)
    ecrits = []
    for extension in ("png", "svg"):
        chemin = sortie / f"{nom}.{extension}"
        figure.savefig(chemin, dpi=200, bbox_inches="tight", facecolor="white")
        ecrits.append(chemin)
    plt.close(figure)
    return ecrits


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--audit", type=Path, default=JSON_DEFAUT)
    analyseur.add_argument("--sortie", type=Path, default=SORTIE_DEFAUT)
    analyseur.add_argument("--copie", type=Path, default=None)
    analyseur.add_argument("--sans-copie", action="store_true")
    analyseur.add_argument(
        "--avec-jev", action="store_true",
        help="adds the expert prompt of Jev 1.13 (ticket 096) to the plates",
    )
    arguments = analyseur.parse_args()
    if arguments.avec_jev:
        DECIDEURS.extend(DECIDEURS_JEV)
        logger.info("Jev arm added: %d decision-makers plotted", len(DECIDEURS))
    exiger_depot_papiers(arguments.sortie, *([arguments.copie] if arguments.copie else []))

    depart = time.monotonic()
    resultats = charger(arguments.audit)
    ecrits = figure_distance(resultats, arguments.sortie)
    ecrits += figure_modes(resultats, arguments.sortie)
    signaler(ecrits)

    copies = 0
    if arguments.copie and not arguments.sans_copie:
        arguments.copie.mkdir(parents=True, exist_ok=True)
        for chemin in ecrits:
            (arguments.copie / chemin.name).write_bytes(chemin.read_bytes())
            copies += 1
    logger.info("Done in %.1f s: %d file(s) written, %d copied",
                time.monotonic() - depart, len(ecrits), copies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
