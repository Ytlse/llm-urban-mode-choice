#!/usr/bin/env python3
"""EMD/JSD composite score and L1 composite score, by decision-maker FAMILY.

Three families are set side by side, each with its colour and its shape:

* the classical models (booster, logit, random forest, kernel logistic);
* the LLMs driven by a minimal prompt;
* the LLMs driven by an expert prompt.

The naive baselines (random, shortest duration, all-car) are drawn in neutral
grey, outside the families: they give the scale of what is gained, they do not
take part in the comparison.

Two figures are produced, which say the same thing in two ways:

* ``*_barres`` — two panels (EMD/JSD composite, L1 composite), one row per
  run, SAME order in both panels. Ranks can thus be read, and
  above all their disagreements: a row ranked better on the left than on the right says
  that the two metrics do not rank alike.
* ``*_nuage`` — EMD composite on the x axis, L1 composite on the y axis, hull
  coloured by family. Shows the separation of the families at a glance.

Reading goes through ``scripts.dashboard.experiences.lister()`` — the same source
as the dashboard, on purpose: the family of a classical model is derived
from the FORMAT of its artefact (``modele:klr``, ``modele:lightgbm``), and deriving
it again here would have made it diverge from the dashboard at the first change.

Usage:
    python scripts/analysis/plot_familles.py
    python scripts/analysis/plot_familles.py --sans-naifs --chaine toutes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from math import log10
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RACINE = Path(__file__).resolve().parents[2]
# Both entries are needed: `scripts/` to import the `dashboard` package,
# the root because `dashboard.experiences` imports itself as `scripts.dashboard.*`.
for _chemin in (RACINE, RACINE / "scripts"):
    if str(_chemin) not in sys.path:
        sys.path.insert(0, str(_chemin))

from scripts.analysis.figures_versionnees import signaler
from scripts.dashboard import experiences as registre  # noqa: E402
from scripts.depot_papiers import chemin_papier, exiger_depot_papiers  # noqa: E402

logger = logging.getLogger("plot_familles")

# ── Families ─────────────────────────────────────────────────────────────────
# Validated categorical palette (the same three slots as `plot_experiences.py`,
# "all pairs" check in light mode). Beyond three hues the scatter would no
# longer meet the colour-blindness thresholds: the baselines are therefore neutral grey.
# Colour never carries identity alone — each family ALSO has its shape,
# and the run's name is written on the bar axis.
CLASSIQUES, MINIMAL, EXPERT, NAIFS = "classiques", "llm_minimal", "llm_expert", "naifs"
FAMILLES = {
    CLASSIQUES: ("Modèles classiques", "#2a78d6", "D"),
    MINIMAL: ("LLM · prompt minimal", "#eb6834", "o"),
    EXPERT: ("LLM · prompt expert", "#1baf7a", "s"),
    NAIFS: ("Repères naïfs (hors familles)", "#52514e", "X"),
}
ORDRE_FAMILLES = (CLASSIQUES, MINIMAL, EXPERT, NAIFS)

# Decision-makers that consult neither a fitted model nor an LLM: they set the scale.
DECIDEURS_NAIFS = {"aleatoire", "duree_minimale", "majoritaire_voiture"}

ETIQUETTES_DECIDEURS = {
    "modele:lightgbm": "LightGBM",
    "modele:klr": "Logistique à noyau",
    "modele:mnl": "Logit multinomial",
    "modele:rf": "Forêt aléatoire",
    "aleatoire": "Aléatoire",
    "duree_minimale": "Durée minimale",
    "majoritaire_voiture": "Tout voiture",
    "gemini-3.5-flash-lite": "Gemini 3.5 FL",
    "gemini-3.1-flash-lite": "Gemini 3.1 FL",
    "antigravity:gemini-3.8-flash": "Gemini 3.8 F (agent)",
}
ETIQUETTES_PROMPTS = {
    "prompt_minimal": "minimal",
    "expert_gem_3.8_v2": "expert v2",
    "expert_gem_3.8_v2_neutre_justif": "expert v2 neutre",
    "expert_gem_3.8_v3": "expert v3",
    "prompt_optimise_v4": "optimisé v4",
}

INK_PRIMAIRE = "#0b0b0b"
INK_SECONDAIRE = "#52514e"
INK_DISCRET = "#8a8880"
SURFACE = "#fcfcfb"
GRILLE = "#e6e5e0"


class RienATracer(RuntimeError):
    """No scored run satisfies the requested scope."""


def famille_de(ligne: dict) -> str | None:
    """The family of a registry row, or None if it fits none.

    The `prompt_optimise_v4` prompt is filed with the experts: it is a long
    prompt, carrying modal priors, and contrasting it with the minimal one is exactly
    what the figure compares. Singling it out would need a fourth hue, which
    the palette cannot hold without losing the colour-blind separation.
    """
    decideur = ligne.get("decideur") or ""
    if decideur.startswith("modele:") or decideur == "modele":
        return CLASSIQUES
    if decideur in DECIDEURS_NAIFS:
        return NAIFS
    prompt = ligne.get("prompt") or ""
    if "minimal" in prompt:
        return MINIMAL
    if prompt.startswith(("expert", "prompt_optimise")):
        return EXPERT
    return None


def etiquette(ligne: dict) -> str:
    """Readable name of a run: the decision-maker, and the prompt if there is one."""
    decideur = ligne.get("decideur") or "?"
    libelle = ETIQUETTES_DECIDEURS.get(decideur, decideur)
    prompt = ligne.get("prompt") or "—"
    if prompt == "—":
        return libelle
    return f"{libelle} · {ETIQUETTES_PROMPTS.get(prompt, prompt)}"


def charger(*, chaine: str, avec_naifs: bool) -> list[dict]:
    """The plottable runs, with their family — and the count of what is discarded.

    Only the LATEST finished and scored runs of an active experiment are
    kept. An older run of the same experiment carries
    an obsolete score: plotting it would show two points for a single
    condition, with nothing on the figure saying which one is authoritative.
    """
    lignes = registre.lister()
    logger.info("Registry read: %d row(s) under %s", len(lignes), registre.DOSSIER)

    ecartes: dict[str, int] = {}

    def ecarter(motif: str) -> None:
        ecartes[motif] = ecartes.get(motif, 0) + 1

    points = []
    for ligne in lignes:
        if ligne.get("statut") != "actif":
            ecarter("expérience archivée ou invalidée")
            continue
        if ligne.get("execution") is None:
            ecarter("expérience définie, jamais exécutée")
            continue
        if not ligne.get("derniere"):
            ecarter("exécution remplacée par une plus récente")
            continue
        if ligne.get("etat") != registre.ETAT_TERMINEE:
            ecarter(f"exécution non terminée ({ligne.get('etat')})")
            continue
        if ligne.get("composite_emd") is None or ligne.get("composite_l1") is None:
            ecarter("exécution sans score composite")
            continue
        if chaine != "toutes" and ligne.get("chaine") != chaine:
            ecarter(f"chaîne des véhicules ≠ « {chaine} »")
            continue
        famille = famille_de(ligne)
        if famille is None:
            ecarter("décideur hors des familles comparées")
            continue
        if famille == NAIFS and not avec_naifs:
            ecarter("repère naïf, écarté sur demande")
            continue
        points.append({
            "nom": ligne["experience"],
            "execution": ligne["execution"],
            "famille": famille,
            "etiquette": etiquette(ligne),
            "emd": float(ligne["composite_emd"]),
            "l1": float(ligne["composite_l1"]),
            "couverture": ligne.get("couverture"),
            "chaine": ligne.get("chaine"),
            "jeu": ligne.get("jeu"),
            "formule": ligne.get("formule"),
            "formule_perimee": bool(ligne.get("formule_perimee")),
        })

    for motif, n in sorted(ecartes.items(), key=lambda kv: -kv[1]):
        logger.info("Discarded — %s: %d row(s)", motif, n)

    perimees = [p["nom"] for p in points if p["formule_perimee"]]
    if perimees:
        logger.error(
            "[ALARME] %d run(s) scored with an outdated scoring formula, comparison "
            "skewed: %s", len(perimees), ", ".join(perimees),
        )
    formules = {p["formule"] for p in points}
    if len(formules) > 1:
        logger.error(
            "[ALARME] %d different scoring formulas on the same figure (%s): "
            "the composite scores are not comparable with each other",
            len(formules), ", ".join(sorted(str(f) for f in formules)),
        )

    if not points:
        raise RienATracer(
            f"no execution kept out of {len(lignes)} row(s) "
            f"(chain « {chaine} », naive baselines {'included' if avec_naifs else 'excluded'})"
        )
    for famille in ORDRE_FAMILLES:
        n = sum(1 for p in points if p["famille"] == famille)
        logger.info("Family \"%s\": %d run(s)", FAMILLES[famille][0], n)
    return points


# ── Common styling ───────────────────────────────────────────────────────────
def _nettoyer(axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRILLE, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for cote in ("top", "right"):
        axes.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        axes.spines[cote].set_color("#d8d7d1")
    axes.tick_params(colors=INK_SECONDAIRE, labelsize=9)


def _pied(figure, points: list[dict], resume: str) -> None:
    jeux = sorted({p["jeu"] for p in points if p["jeu"]})
    chaines = sorted({p["chaine"] for p in points if p["chaine"]})
    figure.text(
        0.012, 0.015,
        f"{resume} · jeu {', '.join(jeux) or '?'} · chaîne des véhicules : "
        f"{', '.join(chaines) or '?'} · référentiel CEREMA · "
        f"figure générée le {date.today().isoformat()}",
        fontsize=7.5, color=INK_DISCRET, ha="left",
    )


def _legende_handles(points: list[dict]) -> list[Line2D]:
    return [
        Line2D([], [], marker=FAMILLES[f][2], linestyle="", markersize=9,
               markerfacecolor=FAMILLES[f][1], markeredgecolor=SURFACE, label=FAMILLES[f][0])
        for f in ORDRE_FAMILLES if any(p["famille"] == f for p in points)
    ]


# ── Figure 1: two bar panels ─────────────────────────────────────────────────
def dessiner_barres(points: list[dict], sortie: Path) -> list[Path]:
    """Two panels, same row order: rank disagreement becomes visible."""
    # Sorted on L1 (the most readable: percentage points), best at the top.
    ordonnes = sorted(points, key=lambda p: p["l1"])
    y = range(len(ordonnes))
    hauteur = max(4.2, 0.42 * len(ordonnes) + 2.0)

    figure, (gauche, droite) = plt.subplots(
        1, 2, figsize=(12.4, hauteur), dpi=300, sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    figure.patch.set_facecolor(SURFACE)

    for axes, cle, titre in ((gauche, "emd", "Composite EMD/JSD"), (droite, "l1", "Composite L1")):
        _nettoyer(axes)
        valeurs = [p[cle] for p in ordonnes]
        axes.barh(list(y), valeurs, height=0.66, zorder=3,
                  color=[FAMILLES[p["famille"]][1] for p in ordonnes],
                  edgecolor=SURFACE, linewidth=0.8)
        marge = max(valeurs) * 0.16
        axes.set_xlim(0, max(valeurs) + marge)
        for i, (p, v) in enumerate(zip(ordonnes, valeurs)):
            axes.text(v + max(valeurs) * 0.015, i, f"{v:.1f}", va="center", ha="left",
                      fontsize=8.5, color=INK_SECONDAIRE, zorder=4)
        axes.set_title(f"{titre}  —  lower = better", fontsize=10.5,
                       color=INK_PRIMAIRE, loc="left", pad=8)

    # A single inversion: the axes share their y axis, inverting
    # two would put the best run back at the bottom.
    gauche.invert_yaxis()
    gauche.set_yticks(list(y))
    gauche.set_yticklabels([p["etiquette"] for p in ordonnes], fontsize=9, color=INK_PRIMAIRE)

    # Margins set in INCHES, not fractions: the figure's height follows the
    # number of rows, and fixed fractions would grow the title band
    # with it — until it squashed the bars on a large batch.
    figure.subplots_adjust(left=0.185, right=0.985,
                           top=1 - 1.05 / hauteur, bottom=0.95 / hauteur)

    figure.suptitle(
        "Modal share fidelity: classical models, minimal prompt, expert prompt",
        fontsize=13.5, color=INK_PRIMAIRE, x=0.012, ha="left", y=1 - 0.30 / hauteur,
    )
    figure.text(
        0.012, 1 - 0.56 / hauteur,
        "Lignes ordonnées sur le composite L1 (panneau de droite) ; le panneau de gauche "
        "garde le même ordre, ses barres non décroissantes signalent un désaccord de rang "
        "entre les deux métriques.",
        fontsize=9, color=INK_SECONDAIRE, ha="left", va="top",
    )
    figure.legend(handles=_legende_handles(points), loc="lower right",
                  bbox_to_anchor=(0.985, 0.06 / hauteur), frameon=False, fontsize=9,
                  ncols=2, labelcolor=INK_SECONDAIRE)
    _pied(figure, points, f"{len(ordonnes)} exécutions")
    return _ecrire(figure, sortie)


# ── Figure 2: EMD × L1 scatter ───────────────────────────────────────────────
def _enveloppe(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """Convex hull (Andrew's monotone chain), without external dependency."""
    pts = sorted(set(zip(xs, ys)))
    if len(pts) <= 2:
        return pts

    def demi(sequence):
        pile: list[tuple[float, float]] = []
        for p in sequence:
            while len(pile) >= 2:
                (x1, y1), (x2, y2) = pile[-2], pile[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                pile.pop()
            pile.append(p)
        return pile

    return demi(pts)[:-1] + demi(reversed(pts))[:-1]


def dessiner_nuage(points: list[dict], sortie: Path) -> list[Path]:
    """EMD × L1 scatter, families circled.

    Both axes are logarithmic: from the best model (4.5) to the random
    draw (56.0), the EMD composite spans a factor of 12, and on a linear
    scale the naive baselines squashed the three compared families into the
    lower left corner — the figure only showed what was already known,
    namely that drawing at random is bad.
    """
    figure, axes = plt.subplots(figsize=(10.4, 7.0), dpi=300)
    figure.patch.set_facecolor(SURFACE)
    _nettoyer(axes)
    axes.set_xscale("log")
    axes.set_yscale("log")

    # Hulls: only for the three compared families. The naive baselines
    # get none — circling them would suggest a coherent group while they
    # only share consulting nothing. The computation is done in
    # log space, the one where the axes are straight: a convex hull
    # computed on the raw values would be concave on screen.
    for famille in (CLASSIQUES, MINIMAL, EXPERT):
        groupe = [p for p in points if p["famille"] == famille]
        if len(groupe) < 3:
            continue
        sommets = _enveloppe([log10(p["emd"]) for p in groupe], [log10(p["l1"]) for p in groupe])
        if len(sommets) >= 3:
            axes.add_patch(Polygon([(10 ** x, 10 ** y) for x, y in sommets], closed=True,
                                   facecolor=FAMILLES[famille][1], edgecolor=FAMILLES[famille][1],
                                   alpha=0.12, linewidth=1.2, zorder=1))

    for famille in ORDRE_FAMILLES:
        groupe = [p for p in points if p["famille"] == famille]
        if not groupe:
            continue
        libelle, couleur, forme = FAMILLES[famille]
        axes.scatter([p["emd"] for p in groupe], [p["l1"] for p in groupe],
                     s=150, c=couleur, marker=forme, edgecolors=SURFACE,
                     linewidths=2.0, zorder=3, label=libelle)

    axes.set_xlabel("Composite EMD/JSD  —  lower = better  (log scale)",
                    fontsize=10, color=INK_PRIMAIRE)
    axes.set_ylabel("Composite L1  —  lower = better  (log scale)",
                    fontsize=10, color=INK_PRIMAIRE)

    # Ticks written out: on a logarithmic axis, matplotlib's automatic
    # ticks are powers of ten, and there would be only
    # one of them in the useful interval.
    xs, ys = [p["emd"] for p in points], [p["l1"] for p in points]
    axes.set_xlim(min(xs) / 1.35, max(xs) * 1.35)
    axes.set_ylim(min(ys) / 1.35, max(ys) * 1.35)
    for axe, valeurs in ((axes.xaxis, (4, 6, 10, 15, 25, 40, 60)),
                         (axes.yaxis, (40, 60, 100, 150, 200, 300))):
        axe.set_major_locator(FixedLocator(valeurs))
        axe.set_minor_locator(NullLocator())
        axe.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    # The reading direction, placed at the top left: the figure's free corridor,
    # the points lining up on the rising diagonal.
    axes.annotate("", xy=(0.045, 0.855), xytext=(0.130, 0.940),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops={"arrowstyle": "-|>", "color": INK_DISCRET, "linewidth": 1.4}, zorder=2)
    axes.text(0.140, 0.944, "meilleur", transform=axes.transAxes, fontsize=9,
              color=INK_DISCRET, style="italic", va="bottom")

    legende = axes.legend(handles=_legende_handles(points), title="Decision-maker family",
                          loc="lower right", frameon=False, fontsize=9.5,
                          title_fontsize=9.5, labelcolor=INK_SECONDAIRE)
    legende.get_title().set_color(INK_PRIMAIRE)

    figure.suptitle("Modal share fidelity by decision-maker family",
                    fontsize=13.5, color=INK_PRIMAIRE, x=0.055, ha="left", y=0.975)
    axes.set_title(
        f"{len(points)} runs · both composite scores are gaps to the reference modal "
        f"shares, lower = more faithful",
        fontsize=9.5, color=INK_SECONDAIRE, loc="left", pad=10,
    )
    _pied(figure, points, f"{len(points)} exécutions")
    figure.tight_layout(rect=(0.0, 0.030, 1.0, 0.960))
    return _ecrire(figure, sortie)


def _ecrire(figure, sortie: Path) -> list[Path]:
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
    analyseur = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument(
        "--sortie", type=Path,
        default=chemin_papier("figures", "familles_composite_l1"),
        help="output prefix without extension (suffixed _barres / _nuage, PNG and SVG written)")
    analyseur.add_argument("--figures", choices=("barres", "nuage", "toutes"), default="toutes")
    analyseur.add_argument(
        "--chaine", choices=("active", "coupée", "toutes"), default="active",
        help="vehicle chain state kept (default: active — the only state where the "
             "three families coexist; \"toutes\" mixes non-comparable conditions)")
    analyseur.add_argument("--sans-naifs", action="store_true",
                           help="discards random, shortest duration and all-car")
    options = analyseur.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    exiger_depot_papiers(options.sortie)

    debut = time.monotonic()
    try:
        points = charger(chaine=options.chaine, avec_naifs=not options.sans_naifs)
    except RienATracer as erreur:
        logger.error("[ALARME] figure not produced: %s", erreur)
        return 1

    ecrits: list[Path] = []
    if options.figures in ("barres", "toutes"):
        ecrits += dessiner_barres(points, options.sortie.with_name(options.sortie.name + "_barres"))
    if options.figures in ("nuage", "toutes"):
        ecrits += dessiner_nuage(points, options.sortie.with_name(options.sortie.name + "_nuage"))

    logger.info(
        "Figures produced: %d run(s) plotted in %.1f s → %s",
        len(points), time.monotonic() - debut,
        ", ".join(str(c) for c in ecrits),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
