#!/usr/bin/env python3
"""Scatter plot comparing the mode choice experiments.

Each experiment of ``data/experiences/`` is placed on two composite
metrics, both "lower = better":

* on the x axis, the EMD/JSD composite score;
* on the y axis, the L1 composite score.

Colour encodes the decision model, shape encodes the prompt template. The
references without a prompt (LightGBM, heuristics) are in neutral grey: they do not
take part in the prompt comparison.

Usage:
    python scripts/analysis/plot_experiences.py exp_a exp_b ...
    python scripts/analysis/plot_experiences.py --sortie "$PAPER_DIR/figures/comparaison"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

RACINE = Path(__file__).resolve().parents[2]
# The root in sys.path: the sibling module is imported as `scripts.analysis.*`, not by
# its bare name — a bare import only works when running the file, not with `python -m`.
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))
# The `experiences` package's locator, imported by its path as the dashboard does,
# with no flat fallback: an experiment filed in a family would otherwise report "folder missing".
_CHEMIN_PAQUET = RACINE / "services" / "llm-agents"
if str(_CHEMIN_PAQUET) not in sys.path:
    sys.path.insert(0, str(_CHEMIN_PAQUET))

from experiences.experience import trouver_dossier_experience

from scripts.analysis.figures_versionnees import signaler
from scripts.depot_papiers import chemin_papier, exiger_depot_papiers

DOSSIER_EXPERIENCES = RACINE / "data" / "experiences"

# exp_gemini-31-fl_expcham5_jtir_t0_nosim is deliberately left out: its
# run of 2026-09-08 was not finished. Add it here once scored.
EXPERIENCES_PAR_DEFAUT = [
    "exp_gemini-35-fl_expcham5_jtir_t0_nosim_2",
    "exp_gemini-31-fl_expcha_jtir_t0_nosim",
    "exp_gemini-35-fl_expcha_jtir_t0_nosim",
    "exp_gemini-35-fl_minper_jtir_t0_nosim_2",
    "exp_lgbm_jtir_nosim",
    "exp_mistral-s_minper_jtir_t0_nosim",
    "exp_durmin_nosim",
    "exp_alea_nosim",
    "exp_gemini-31-fl_minper_jtir_t0_nosim",
    "exp_majvoiture_nosim",
]

# Validated categorical palette (slots 1-3, "all pairs" check in light
# mode: worst ΔE CVD 9.2, worst ΔE normal vision 21.8). Beyond three
# hues, the scatter would no longer meet the colour-blindness thresholds: the
# references are therefore in neutral grey, told apart by shape and label.
COULEUR_NEUTRE = "#52514e"
COULEURS_MODELES = {
    "gemini-3.5-flash-lite": "#2a78d6",
    "gemini-3.1-flash-lite": "#eb6834",
    # Faulty name kept: runs archived before 2026-09-10 carry it,
    # and without this entry their points would fall to neutral grey.
    "gemini-3.1-flash-lite-preview": "#eb6834",
    "mistral-small-latest": "#1baf7a",
    "gemini-3.8-flash": "#9c27b0",
}

# Shapes by prompt template, then by decision-maker type for the references.
FORMES_PROMPTS = {
    "minimal_persona": ("o", "Persona minimal", "persona"),
    # `expert_chaine` was renamed `expert_m4` on 2026-09-11. The historical key stays: it is
    # the ARCHIVED experiments that carry it in their `experience.yaml`, and stripping
    # them of their shape would leave the chart silent on half of the points.
    "expert_chaine": ("s", "Expert chaîne", "chaîne"),
    "expert_m4": ("s", "Expert chaîne", "chaîne"),
    "expert_chaine_m5": ("^", "Expert chaîne (m5)", "chaîne m5"),
}
FORMES_REFERENCES = {
    "modele": ("D", "LightGBM (référence ML)"),
    "duree_minimale": ("P", "Durée minimale"),
    "majoritaire_voiture": ("X", "Tout voiture"),
    "aleatoire": ("*", "Aléatoire"),
    "antigravity": ("v", "Antigravity (sous-agent)"),
}

ETIQUETTES_MODELES = {
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",   # archives
    "mistral-small-latest": "Mistral Small",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
}

# Short labels drawn next to the points: the LLM cluster is tight,
# long labels overlap there.
ETIQUETTES_COURTES = {
    "gemini-3.5-flash-lite": "G3.5",
    "gemini-3.1-flash-lite": "G3.1",
    "gemini-3.1-flash-lite-preview": "G3.1",   # archives
    "mistral-small-latest": "Mistral S",
    "gemini-3.8-flash": "G3.8",
}

# Label offset (in typographic points) per experiment, tuned by
# hand: the scatter is too dense for automatic placement.
DECALAGES = {
    "exp_lgbm_jtir_nosim": (16, 2),
    "exp_alea_nosim": (-14, 4),
    "exp_majvoiture_nosim": (14, 2),
    "exp_durmin_nosim": (14, -12),
    "exp_mistral-s_minper_jtir_t0_nosim": (-14, 2),
    "exp_gemini-31-fl_minper_jtir_t0_nosim": (14, 0),
    "exp_gemini-35-fl_minper_jtir_t0_nosim_2": (14, 0),
    "exp_gemini-31-fl_expcha_jtir_t0_nosim": (14, -10),
    "exp_gemini-31-fl_expcham5_jtir_t0_nosim": (14, 10),
    "exp_gemini-35-fl_expcha_jtir_t0_nosim": (-14, 2),
    "exp_gemini-35-fl_expcham5_jtir_t0_nosim_2": (-14, -12),
}

INK_PRIMAIRE = "#0b0b0b"
INK_SECONDAIRE = "#52514e"
INK_DISCRET = "#8a8880"
SURFACE = "#fcfcfb"

logger = logging.getLogger("plot_experiences")


class ExperienceIllisible(RuntimeError):
    """A requested experiment has no usable result."""


def derniere_execution_scoree(dossier: Path) -> Path:
    """Returns the latest run that has a ``scores.json``."""
    executions = sorted((dossier / "executions").glob("*"))
    ignorees = 0
    for execution in reversed(executions):
        if (execution / "scores.json").exists():
            if ignorees:
                logger.info(
                    "%s: %d more recent run(s) skipped, without scores.json",
                    dossier.name,
                    ignorees,
                )
            return execution
        ignorees += 1
    raise ExperienceIllisible(
        f"{dossier.name}: none of the {len(executions)} run(s) has a scores.json"
    )


def lire_experience(nom: str) -> dict:
    """Assembles an experiment's configuration and scores, read in its family or else flat."""
    dossier = trouver_dossier_experience(nom, racine=DOSSIER_EXPERIENCES)
    if dossier is None:
        raise ExperienceIllisible(
            f"{nom}: not found under {DOSSIER_EXPERIENCES}, neither flat nor in a family "
            "(the cold archive is not read)"
        )

    config = yaml.safe_load((dossier / "experience.yaml").read_text(encoding="utf-8"))
    execution = derniere_execution_scoree(dossier)
    scores = json.loads((execution / "scores.json").read_text(encoding="utf-8"))

    decideur = config.get("decideur") or {}
    gabarit = config.get("gabarit") or {}
    modele = decideur.get("modele")
    type_decideur = decideur.get("type")
    est_reference = modele is None

    return {
        "nom": nom,
        "execution": execution.name,
        "modele": modele,
        "type_decideur": type_decideur,
        "variante": gabarit.get("variante"),
        "est_reference": est_reference,
        "emd_jsd": scores["composite"]["emd_jsd"],
        "l1": scores["composite"]["l1"],
        "couverture": scores["couverture"]["taux"],
        "n_agents": scores["global"]["n_agents"],
        "jour": (scores.get("lecture") or {}).get("jour_retenu"),
        "genere_le": scores.get("genere_le"),
    }


def style_du_point(point: dict) -> tuple[str, str, str]:
    """Returns (colour, shape, shape label) for a point."""
    if point["est_reference"]:
        forme, libelle = FORMES_REFERENCES.get(point["type_decideur"], ("o", point["type_decideur"]))
        return COULEUR_NEUTRE, forme, libelle
    couleur = COULEURS_MODELES.get(point["modele"])
    if couleur is None:
        raise ExperienceIllisible(
            f"{point['nom']}: model \"{point['modele']}\" missing from the palette "
            f"(available slots: {', '.join(COULEURS_MODELES)})"
        )
    forme, libelle, _ = FORMES_PROMPTS.get(point["variante"], ("o", str(point["variante"]), str(point["variante"])))
    return couleur, forme, libelle


def etiquette_point(point: dict) -> str:
    """Short label, on a single line, drawn next to the point."""
    if point["est_reference"]:
        return FORMES_REFERENCES.get(point["type_decideur"], ("", point["type_decideur"]))[1]
    modele = ETIQUETTES_COURTES.get(point["modele"], point["modele"])
    prompt = FORMES_PROMPTS.get(point["variante"], ("", "", point["variante"]))[2]
    return f"{modele} · {prompt}"


def dessiner(points: list[dict], sortie: Path) -> list[Path]:
    figure, axes = plt.subplots(figsize=(10.0, 7.0), dpi=300)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    for point in points:
        couleur, forme, _ = style_du_point(point)
        taille = 320 if forme == "*" else 150
        axes.scatter(
            point["emd_jsd"],
            point["l1"],
            s=taille,
            c=couleur,
            marker=forme,
            edgecolors=SURFACE,
            linewidths=2.0,
            zorder=3,
        )

    # Direct labels: the contrast rule requires visible labels,
    # the palette's aqua falling below 3:1 contrast on a light background.
    for point in points:
        dx, dy = DECALAGES.get(point["nom"], (14, 2))
        axes.annotate(
            etiquette_point(point),
            xy=(point["emd_jsd"], point["l1"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=INK_SECONDAIRE,
            ha="left" if dx >= 0 else "right",
            va="center" if abs(dy) < 6 else ("bottom" if dy > 0 else "top"),
            linespacing=1.25,
            zorder=4,
        )

    axes.set_xlabel("Composite EMD/JSD  —  lower = better", fontsize=10, color=INK_PRIMAIRE)
    axes.set_ylabel("Composite L1  —  lower = better", fontsize=10, color=INK_PRIMAIRE)
    axes.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for cote in ("top", "right"):
        axes.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        axes.spines[cote].set_color("#d8d7d1")
    axes.tick_params(colors=INK_SECONDAIRE, labelsize=9)

    marge_x = (max(p["emd_jsd"] for p in points) - min(p["emd_jsd"] for p in points)) * 0.14
    marge_y = (max(p["l1"] for p in points) - min(p["l1"] for p in points)) * 0.14
    axes.set_xlim(min(p["emd_jsd"] for p in points) - marge_x, max(p["emd_jsd"] for p in points) + marge_x)
    axes.set_ylim(min(p["l1"] for p in points) - marge_y, max(p["l1"] for p in points) + marge_y)

    axes.annotate(
        "",
        xy=(0.030, 0.560),
        xytext=(0.105, 0.635),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": INK_DISCRET, "linewidth": 1.4},
        zorder=2,
    )
    axes.text(
        0.115, 0.638, "meilleur", transform=axes.transAxes,
        fontsize=9, color=INK_DISCRET, style="italic", va="bottom",
    )

    modeles_presents = [m for m in COULEURS_MODELES if any(p["modele"] == m for p in points)]
    legende_couleurs = [
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=COULEURS_MODELES[m],
               markeredgecolor=SURFACE, label=ETIQUETTES_MODELES.get(m, m))
        for m in modeles_presents
    ]
    if any(p["est_reference"] for p in points):
        legende_couleurs.append(
            Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=COULEUR_NEUTRE,
                   markeredgecolor=SURFACE, label="Références (sans prompt)")
        )

    prompts_presents = [v for v in FORMES_PROMPTS if any(p["variante"] == v and not p["est_reference"] for p in points)]
    legende_formes = [
        Line2D([], [], marker=FORMES_PROMPTS[v][0], linestyle="", markersize=9, markerfacecolor=INK_SECONDAIRE,
               markeredgecolor=SURFACE, label=FORMES_PROMPTS[v][1])
        for v in prompts_presents
    ]
    legende_formes += [
        Line2D([], [], marker=FORMES_REFERENCES[t][0], linestyle="", markersize=9,
               markerfacecolor=COULEUR_NEUTRE, markeredgecolor=SURFACE, label=FORMES_REFERENCES[t][1])
        for t in FORMES_REFERENCES
        if any(p["type_decideur"] == t and p["est_reference"] for p in points)
    ]

    premiere = axes.legend(
        handles=legende_couleurs, title="Model (colour)", loc="upper left",
        frameon=False, fontsize=9, title_fontsize=9, labelcolor=INK_SECONDAIRE,
    )
    premiere.get_title().set_color(INK_PRIMAIRE)
    axes.add_artist(premiere)
    seconde = axes.legend(
        handles=legende_formes, title="Prompt / method (shape)", loc="lower right",
        frameon=False, fontsize=9, title_fontsize=9, labelcolor=INK_SECONDAIRE,
    )
    seconde.get_title().set_color(INK_PRIMAIRE)

    jours = sorted({p["jour"] for p in points if p["jour"]})
    executions = sorted(p["execution"][:10] for p in points)
    n_agents = sorted({p["n_agents"] for p in points})
    figure.suptitle(
        "Modal share fidelity by model and by prompt",
        fontsize=14, color=INK_PRIMAIRE, x=0.055, ha="left", y=0.965,
    )
    axes.set_title(
        f"{len(points)} experiments · simulated day {', '.join(jours)} · "
        f"{n_agents[0] if len(n_agents) == 1 else f'{n_agents[0]}–{n_agents[-1]}'} agents · "
        f"runs from {executions[0]} to {executions[-1]}",
        fontsize=9.5, color=INK_SECONDAIRE, loc="left", pad=10,
    )
    figure.text(
        0.055, 0.018,
        f"Référentiel CEREMA · figure générée le {date.today().isoformat()}",
        fontsize=8, color=INK_DISCRET, ha="left",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.925))

    ecrits = []
    for extension in ("png", "svg"):
        chemin = sortie.with_suffix(f".{extension}")
        chemin.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(chemin, facecolor=SURFACE)
        ecrits.append(chemin)
    plt.close(figure)
    # Ticket 039, step 6: the DEFAULT output is versioned. A regeneration must
    # say what it has just cost, otherwise the growth only shows in the pack.
    signaler(ecrits)
    return ecrits


def main(argv: list[str] | None = None) -> int:
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("experiences", nargs="*", default=None, help="experiment names (default: the main batch)")
    analyseur.add_argument(
        "--sortie", type=Path,
        default=chemin_papier("figures", "comparaison_experiences"),
        help="output path without extension (PNG and SVG written; default: papers repository)",
    )
    options = analyseur.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    exiger_depot_papiers(options.sortie)

    noms = options.experiences or EXPERIENCES_PAR_DEFAUT
    debut = time.monotonic()
    logger.info("Reading %d experiment(s) under %s", len(noms), DOSSIER_EXPERIENCES)

    points, echecs = [], []
    for nom in noms:
        try:
            point = lire_experience(nom)
        except (ExperienceIllisible, OSError, KeyError, ValueError) as erreur:
            echecs.append((nom, str(erreur)))
            logger.error("[ALARME] unreadable experiment: %s", erreur)
            continue
        points.append(point)
        logger.info(
            "%s → run %s · EMD/JSD %.2f · L1 %.1f · coverage %.2f%%",
            nom, point["execution"], point["emd_jsd"], point["l1"], point["couverture"] * 100,
        )

    if not points:
        logger.error("[ALARME] no usable experiment, figure not produced")
        return 1
    if echecs:
        logger.error("[ALARME] %d/%d experiment(s) discarded: %s", len(echecs), len(noms),
                     ", ".join(nom for nom, _ in echecs))

    ecrits = dessiner(points, options.sortie)
    logger.info(
        "Figure produced: %d point(s) plotted in %.1f s → %s",
        len(points), time.monotonic() - debut, ", ".join(str(c) for c in ecrits),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
