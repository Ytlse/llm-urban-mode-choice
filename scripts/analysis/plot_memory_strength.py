#!/usr/bin/env python3
"""The chapter 3 figure: what is forgotten, and what is not.

§ 3.4 states that two registers coexist under two clocks — episodic traces
erode, concepts do not — and gives three numbers to say so:
a time constant of 2.8 days, a lifetime ceiling at thirty, a
half-life of 1.9 days. Putting them on the same axis shows the gap that the
sentence states without making it visible.

The figure is **in English**, like all figures of the submitted manuscript. Its caption
follows the paper's style rules: it defines what is plotted, without an "X, not Y"
phrasing, without defending a choice, and without saying anything about how it is made —
traceability lives in the repository and in the chapter's HTML comments.

``ch3_oubli`` carries, on the y axis, the TEMPORAL COMPONENT of the retrieval score.
A single axis, and it is not a presentation trick: for a concept,
`_time_decay_score` literally returns its confidence instead of a
decay (`llm/longterm.py`, two regimes of ticket 071 lot 3). The two
registers really are the same component, read in two ways.

**No constant is written by hand.** The script imports the repository's
functions — `poids_temporel`, `force_initiale`, `confiance` — and calls them. The
values come from `settings.agent`. The figure therefore cannot diverge from the
code, and the figure footnote prints the constants that produced it.
If the import fails, nothing is plotted: a mechanism figure drawn with
guessed values is worse than no figure.

Usage:
    services/llm-agents/.venv/bin/python scripts/analysis/plot_memory_strength.py
    … [--sortie <dossier>] [--copie <dossier>]

Default output: `figures/` of the papers repository (`PAPER_DIR`, ticket 115). The copy to
`article/images/` went away with that folder: `--copie` only copies on request.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from scripts.analysis.figures_versionnees import signaler
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier

logger = logging.getLogger("ch3")

SERVICE = RACINE / "services" / "llm-agents"
SORTIE_DEFAUT = sortie_papier("figures")

# The sensitivity point of the setup: the time constant published by Park et al.
# (2023), against the one inherited from Vu et al. that the repository uses by default. It is
# not a repository setting but a value from the literature — hence the only literal
# constant of this file, and the comment tying it to its source.
S0_PARK_JOURS = 8.3

# The concept shown is the one of the "What I know" block served to the model: twelve
# observations, no counter-example. The second is the same after five counter-examples.
CONCEPT_OBSERVATIONS = 12
CONCEPT_CONTRE_EXEMPLES = 5

HORIZON_JOURS = 30

# The set-aside threshold applies ONLY to concepts. Drawn across the whole width, it
# read as a cut-off on the episodic curves, which have none: they go
# below 0.5 and keep being served, less strongly.
DEBUT_SEUIL_CONCEPTS = 15.5

EPISODIQUE = "#c0562c"
CONCEPT = "#008572"
REFERENCE = "#6f6c66"
ENCRE = "#2b2a28"
ENCRE_SECONDE = "#55534f"
GRILLE = "#e2e0dc"


class Dispositif:
    """The repository's functions and constants, read once and carried together."""

    def __init__(self, poids_temporel, force_initiale, confiance, agent) -> None:
        self.poids_temporel = poids_temporel
        self.force_banale = force_initiale(0.0)
        self.poids_temps = float(agent.long_term_retrieval__time_weight)
        self.delta_rappel = float(agent.memoire__force_delta_rappel_jours)
        self.force_marquante = force_initiale(1.0)
        self.confiance_confirmee = confiance(CONCEPT_OBSERVATIONS, 0)
        self.confiance_contredite = confiance(CONCEPT_OBSERVATIONS, CONCEPT_CONTRE_EXEMPLES)
        self.seuil_service = float(agent.memoire__confiance_seuil_service)
        self.seuil_purge = float(agent.memoire__purge_seuil_poids)
        self.force_base = float(agent.long_term_retrieval__force_base_jours)
        self.force_max = float(agent.memoire__force_max_jours)
        self.k_importance = float(agent.memoire__force_k_importance)

    @property
    def demi_vie(self) -> float:
        return math.log(2) * self.force_banale

    @property
    def age_de_purge(self) -> float:
        """The age at which an ordinary trace never recalled falls below the threshold."""
        return -self.force_banale * math.log(self.seuil_purge)


def charger_le_dispositif() -> Dispositif | None:
    """Imports the service's functions. Any import failure forbids the figure."""
    if str(SERVICE) not in sys.path:
        sys.path.insert(0, str(SERVICE))
    try:
        from llm.concepts import confiance
        from llm.gravite import force_initiale, poids_temporel
        from settings import settings
    except ImportError as erreur:
        logger.error(
            "[ALARME] The memory functions cannot be found from %s: %s. "
            "Run the script with services/llm-agents/.venv/bin/python; "
            "no figure is plotted with guessed values.",
            SERVICE.relative_to(RACINE), erreur,
        )
        return None
    dispositif = Dispositif(poids_temporel, force_initiale, confiance, settings.agent)
    logger.info(
        "Constants read from the repository: S0 = %.1f d, k = %.0f, ceiling = %.0f d, "
        "ordinary strength = %.1f d, serious strength = %.1f d, half-life = %.2f d, "
        "component weight = %.2f",
        dispositif.force_base, dispositif.k_importance, dispositif.force_max,
        dispositif.force_banale, dispositif.force_marquante, dispositif.demi_vie,
        dispositif.poids_temps,
    )
    return dispositif


def figure_oubli(d: Dispositif, sortie: Path) -> list[Path]:
    """The two clocks on the same axis: what erodes, what only moves when observed."""
    jours = np.linspace(0, HORIZON_JOURS, 601)
    banale = np.array([d.poids_temporel(t, d.force_banale) for t in jours])
    marquante = np.array([d.poids_temporel(t, d.force_marquante) for t in jours])
    sensibilite = np.array([d.poids_temporel(t, S0_PARK_JOURS) for t in jours])

    figure, axes = plt.subplots(figsize=(9.0, 5.4))

    # What severity buys: the filled area between the ordinary and the serious trace.
    axes.fill_between(jours, banale, marquante, color=EPISODIQUE, alpha=0.10, zorder=1)

    axes.plot(jours, marquante, color=EPISODIQUE, linewidth=2.0, linestyle="--", zorder=3,
              label=f"serious episodic trace — lifetime {d.force_marquante:.1f} d")
    axes.plot(jours, banale, color=EPISODIQUE, linewidth=2.2, zorder=4,
              label=f"ordinary episodic trace — lifetime {d.force_banale:.1f} d")
    axes.plot(jours, sensibilite, color=REFERENCE, linewidth=1.5, linestyle=":", zorder=3,
              label=f"sensitivity arm — S₀ = {S0_PARK_JOURS} d (Park et al., 2023)")

    axes.axhline(d.confiance_confirmee, color=CONCEPT, linewidth=2.2, zorder=5,
                 label=f"concept — {CONCEPT_OBSERVATIONS} observations, no counter-example")
    axes.axhline(d.confiance_contredite, color=CONCEPT, linewidth=2.0, linestyle=(0, (6, 3)),
                 alpha=0.85, zorder=5,
                 label=f"same concept after {CONCEPT_CONTRE_EXEMPLES} counter-examples")

    # The threshold carries the concepts' colour and stops with them: it applies to them
    # alone. The episodic curves cross it without anything happening to them.
    axes.plot([DEBUT_SEUIL_CONCEPTS, HORIZON_JOURS], [d.seuil_service, d.seuil_service],
              color=CONCEPT, linewidth=1.2, linestyle="-.", alpha=0.9, zorder=5)

    # The arrow fits in the free band between the two lines, the legend having left
    # the plot area. It measures the drop, it does not comment on it.
    milieu = (d.confiance_confirmee + d.confiance_contredite) / 2
    axes.annotate(
        "", xy=(19.0, d.confiance_contredite + 0.012), xytext=(19.0, d.confiance_confirmee - 0.012),
        arrowprops={"arrowstyle": "-|>", "color": CONCEPT, "linewidth": 1.4},
    )
    axes.text(19.6, milieu, f"{CONCEPT_CONTRE_EXEMPLES} counter-examples,\n"
              f"confidence {d.confiance_confirmee:.2f} → {d.confiance_contredite:.2f}",
              fontsize=8.5, color=CONCEPT, va="center", linespacing=1.4)
    axes.text(DEBUT_SEUIL_CONCEPTS, d.seuil_service + 0.032,
              f"concepts set aside below {d.seuil_service:g}",
              fontsize=8.5, color=ENCRE_SECONDE)

    # The half-life and the purge age: two markers placed in the empty triangle under the
    # ordinary curve, the only place no curve goes through.
    axes.plot([d.demi_vie, d.demi_vie], [0, 0.5], color=EPISODIQUE, linewidth=0.9,
              linestyle=":", alpha=0.9, zorder=2)
    axes.annotate(
        f"half-life {d.demi_vie:.1f} d", xy=(d.demi_vie, 0.5), xytext=(6.4, 0.205),
        fontsize=8.5, color=ENCRE_SECONDE,
        arrowprops={"arrowstyle": "-", "color": "#a8a6a1", "linewidth": 0.9},
    )
    axes.plot([d.age_de_purge], [d.seuil_purge], marker="o", markersize=5,
              color=EPISODIQUE, zorder=5)
    axes.annotate(
        f"purged at {d.age_de_purge:.0f} d", xy=(d.age_de_purge, d.seuil_purge),
        xytext=(d.age_de_purge + 1.1, 0.062), fontsize=8.5, color=ENCRE_SECONDE,
        arrowprops={"arrowstyle": "-", "color": "#a8a6a1", "linewidth": 0.9},
    )

    axes.set_xlim(0, HORIZON_JOURS)
    axes.set_ylim(0, 1.04)
    axes.set_xlabel("age since last recall (days)", fontsize=10, color=ENCRE)
    axes.set_ylabel("temporal component of the retrieval score", fontsize=10, color=ENCRE)
    axes.tick_params(labelsize=9, colors=ENCRE_SECONDE)
    axes.grid(color=GRILLE, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for bord in ("top", "right"):
        axes.spines[bord].set_visible(False)
    for bord in ("left", "bottom"):
        axes.spines[bord].set_color(GRILLE)
    # The legend leaves the plot area: inside, it collided with the concepts' arrow,
    # the only place where the latter is readable.
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5,
                frameon=False, labelcolor=ENCRE_SECONDE, handlelength=2.6,
                columnspacing=2.4, borderaxespad=0.0)

    # The note fits on two lines: in one piece, `bbox_inches="tight"` widened
    # the canvas to the text's length and squashed the figure.
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.text(
        0.012, 0.014,
        f"Lifetime = min(S₀ × (1 + {d.k_importance:.0f} × gravity), {d.force_max:.0f} d), with "
        f"S₀ = {d.force_base} d. Each recall resets the age to zero and adds "
        f"{d.delta_rappel:.0f} day of lifetime.\n"
        f"Concept confidence follows Laplace's rule of succession. A set-aside concept stays in "
        "the index, with the date it was set aside.\n"
        f"An episodic trace is purged below {d.seuil_purge:g} of weight. This component carries "
        f"weight {d.poids_temps:.2f} in the retrieval score.",
        fontsize=7.5, color=ENCRE_SECONDE, linespacing=1.6,
    )
    return ecrire(figure, sortie, "ch3_oubli")


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
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--sortie", type=Path, default=SORTIE_DEFAUT)
    analyseur.add_argument("--copie", type=Path, default=None,
                           help="folder to copy the figures to (no copy by default)")
    analyseur.add_argument("--sans-copie", action="store_true")
    arguments = analyseur.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")
    exiger_depot_papiers(arguments.sortie, *([arguments.copie] if arguments.copie else []))
    depart = time.monotonic()
    logger.info("Chapter 3 figure: reading the memory constants")

    dispositif = charger_le_dispositif()
    if dispositif is None:
        return 1

    ecrits = figure_oubli(dispositif, arguments.sortie)
    signaler(ecrits)

    copies = 0
    recopie = arguments.copie if arguments.copie and not arguments.sans_copie else None
    if recopie:
        recopie.mkdir(parents=True, exist_ok=True)
        for chemin in ecrits:
            shutil.copy2(chemin, recopie / chemin.name)
            copies += 1

    logger.info(
        "Done in %.1f s: %d file(s) written to %s, %d copied to %s",
        time.monotonic() - depart, len(ecrits), arguments.sortie, copies, recopie or "(aucun)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
