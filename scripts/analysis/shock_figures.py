#!/usr/bin/env python3
"""PROVISIONAL figures of § 7.2 — shock campaign of 2026-09-21 (ticket 095).

Two figures, from the sealed observations:
  1. the car taken when it is offered, day by day, treated against control;
  2. the six criteria of the car at the four survey milestones.

⚠ PROVISIONAL. The phases are aligned on the DECLARED shock and its COMPUTED window exit, and
not hard-coded — that is the flaw of `modal_variation_rate.py`, whose four phases are
fixed on a shock at days 8-9 which is no longer the protocol's.
"""
from __future__ import annotations

import csv
import collections
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from scripts.depot_papiers import chemin_papier, exiger_depot_papiers  # noqa: E402

ARCHIVE = RACINE / "experiments/archive/simulations_septembre_2026"
TRAITE, TEMOIN = "2026-09-21_15_13", "2026-09-21_11_11"
CHOC = datetime.date(2026, 3, 30)
SORTIE = datetime.date(2026, 4, 14)          # 15.29 d after the shock, derived from severity 0.70
COULEUR = {"traité": "#CE3B4B", "témoin": "#4A6FA5"}


def _ecrire(fig, sortie: Path) -> None:
    """Writes the figure as SVG and PNG: not every Markdown reader displays SVG."""
    for ext in ("svg", "png"):
        chemin = sortie.with_suffix("." + ext)
        fig.savefig(chemin, format=ext, dpi=200, bbox_inches="tight")
        print(f"written: {chemin}")


def part_voiture_par_jour(run: str) -> dict[datetime.date, tuple[int, int]]:
    """(car chosen, car offered) per day, over the model's decisions only."""
    par_jour: dict[datetime.date, list[int]] = collections.defaultdict(lambda: [0, 0])
    with open(ARCHIVE / run / "moves.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Méthode de sélection") != "LLM":
                continue
            j = datetime.datetime.fromtimestamp(float(r["Temps simulé"]), datetime.UTC).date()
            offerts = {x.strip() for x in r["Modes proposés au LLM"].split("|")}
            if "Voiture Privée" in offerts:
                par_jour[j][1] += 1
                if r["Mode de transport Choisi"] == "Voiture Privée":
                    par_jour[j][0] += 1
    return {k: tuple(v) for k, v in par_jour.items()}


def propension_par_jour(run: str) -> dict[datetime.date, list[float]]:
    """P(car) stated by the model at each decision, grouped by lived day.

    This signal is DAILY and already logged: it costs no call. Above all it requires
    no "before / during / after" slice, hence no bound derived from the parameter
    whose effect is being measured.
    """
    par_jour: dict[datetime.date, list[float]] = collections.defaultdict(list)
    with open(ARCHIVE / run / "moves.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Méthode de sélection") != "LLM":
                continue
            if "Voiture Privée" not in r["Modes proposés au LLM"]:
                continue
            j = datetime.datetime.fromtimestamp(float(r["Temps simulé"]), datetime.UTC).date()
            try:
                par_jour[j].append(float(r["P(Voiture Privée) %"]))
            except (TypeError, ValueError):
                pass
    return par_jour


def figure_propension(sortie: Path) -> None:
    import statistics
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for run, nom in ((TEMOIN, "témoin"), (TRAITE, "exposé")):
        d = propension_par_jour(run)
        jours = [j for j in sorted(d) if d[j]]
        ax.plot(jours, [statistics.mean(d[j]) for j in jours], marker="o", ms=3.5, lw=1.6,
                color=COULEUR["témoin" if nom == "témoin" else "traité"], label=nom)
    ax.axvline(CHOC, color="#CE3B4B", ls="--", lw=1.2)
    ax.annotate("breakdown", (CHOC, 103), color="#CE3B4B", fontsize=9, ha="center")
    ax.axvline(SORTIE, color="#333", ls=":", lw=1.2)
    ax.annotate("the account leaves the context", (SORTIE, 103), color="#333", fontsize=9, ha="center")
    ax.set_ylim(-5, 115)
    ax.set_ylabel("Stated P(car), daily mean (%)")
    ax.set_title("Daily propensity to the car, stated at every decision")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _ecrire(fig, sortie)


def figure_comportement(sortie: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for run, nom in ((TEMOIN, "témoin"), (TRAITE, "traité")):
        d = part_voiture_par_jour(run)
        jours = sorted(d)
        # Rolling mean over three lived days: a single day carries two or three decisions,
        # and the raw point would jump from 0 to 100 % without meaning anything.
        xs, ys = [], []
        for i, j in enumerate(jours):
            f = jours[max(0, i - 2): i + 1]
            pris = sum(d[k][0] for k in f)
            off = sum(d[k][1] for k in f)
            if off:
                xs.append(j)
                ys.append(100 * pris / off)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=COULEUR[nom], label=nom)
    ax.axvline(CHOC, color="#CE3B4B", ls="--", lw=1.2)
    ax.annotate("engine breakdown", (CHOC, 103), color="#CE3B4B", fontsize=9, ha="center")
    ax.axvline(SORTIE, color="#333", ls=":", lw=1.2)
    ax.annotate("block exit\n(15.29 d, derived from severity)", (SORTIE, 103),
                color="#333", fontsize=9, ha="center")
    ax.set_ylim(-5, 118)
    ax.set_ylabel("car taken when it is offered (%)")
    ax.set_title("Share of the car among trips where it was offered — mean over three lived days")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    _ecrire(fig, sortie)


def figure_croyances(sortie: Path) -> None:
    crit = ["rapidite", "praticite", "confort", "securite", "cout", "ecologie"]
    lib = {"rapidite": "rapidité", "praticite": "praticité", "confort": "confort",
           "securite": "sécurité", "cout": "coût", "ecologie": "écologie (témoin)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (run, nom) in zip(axes, ((TRAITE, "traité"), (TEMOIN, "témoin"))):
        lignes = list(csv.DictReader(
            open(ARCHIVE / run / "affinites_declarees.csv", encoding="utf-8")))
        jalons = sorted({int(x["jour_simule"]) for x in lignes})
        for c in crit:
            ys = [next(int(x["score"]) for x in lignes
                       if x["mode"] == "voiture" and x["critere"] == c
                       and int(x["jour_simule"]) == j) for j in jalons]
            ax.plot(jalons, ys, marker="o", ms=4,
                    lw=2.2 if c == "ecologie" else 1.4,
                    color="#2E7D32" if c == "ecologie" else None, label=lib[c])
        ax.set_title(f"car — {nom} arm")
        ax.set_xlabel("simulated day")
        ax.set_xticks(jalons)
        ax.grid(alpha=.25)
        ax.set_ylim(0, 10.5)
    axes[0].set_ylabel("declared score (0-10)")
    axes[0].axvspan(15, 16, color="#CE3B4B", alpha=.15)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, fontsize=9)
    fig.tight_layout()
    _ecrire(fig, sortie)


# The six criteria of the evening survey, and the colour that follows them across figures.
# ENGLISH labels: these figures go as they are into the manuscript.
CRITERES = [
    ("rapidite", "speed", "#1F77B4"),
    ("praticite", "convenience", "#7B3294"),
    ("confort", "comfort", "#E08214"),
    ("securite", "safety", "#C51B7D"),
    ("cout", "cost", "#6E7B8B"),
    ("ecologie", "environment", "#2E7D32"),
]
MODES_ENQUETE = [
    ("voiture", "Car"),
    ("transports_collectifs", "Public transport"),
    ("velo", "Bicycle"),
    ("marche", "Walking"),
    ("train", "Train"),
]
JOUR_CHOC, JOUR_SORTIE = 15, 30


def _affinites(run: str) -> dict:
    """(mode, criterion, day) -> declared score, from an arm's evening survey."""
    lignes = csv.DictReader(
        open(ARCHIVE / run / "affinites_declarees.csv", encoding="utf-8")
    )
    return {(x["mode"], x["critere"], int(x["jour_simule"])): int(x["score"]) for x in lignes}


def figure_affinites_tous_modes(sortie: Path) -> None:
    """The six criteria, over the five modes, at the four milestones, treated against control.

    The § 7.2 figure only showed the car. This one answers the question the previous
    one left open: do the opinions of the OTHER modes move when the car breaks down?
    A criterion whose solid line leaves the dotted one has moved for the exposed agent.
    """
    traite, temoin = _affinites(TRAITE), _affinites(TEMOIN)
    jalons = sorted({j for _, _, j in traite})

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True, sharey=True)
    plats = axes.ravel()

    for ax, (cle, libelle) in zip(plats, MODES_ENQUETE):
        ax.axvspan(JOUR_CHOC - 0.6, JOUR_CHOC + 0.6, color="#CE3B4B", alpha=.16, lw=0)
        ax.axvline(JOUR_SORTIE, color="#555", ls=":", lw=1.1)
        for critere, _, couleur in CRITERES:
            ax.plot(jalons, [temoin[(cle, critere, j)] for j in jalons],
                    color=couleur, lw=1.1, ls=(0, (4, 2)), alpha=.45, zorder=2)
            ax.plot(jalons, [traite[(cle, critere, j)] for j in jalons],
                    color=couleur, lw=1.9, marker="o", ms=4, zorder=3)
        ax.set_title(libelle, fontsize=11)
        ax.set_xticks(jalons)
        ax.set_ylim(0, 10.6)
        ax.grid(alpha=.22)
        for cote in ("top", "right"):
            ax.spines[cote].set_visible(False)

    # The sixth cell carries the legend: two lines for the arms, six for the criteria.
    legende = plats[-1]
    legende.axis("off")
    poignees = [
        mlines.Line2D([], [], color="#37474F", lw=1.9, marker="o", ms=4, label="exposed agent"),
        mlines.Line2D([], [], color="#37474F", lw=1.1, ls=(0, (4, 2)), alpha=.6, label="control agent"),
    ] + [
        mlines.Line2D([], [], color=couleur, lw=2.2, label=libelle)
        for _, libelle, couleur in CRITERES
    ]
    legende.legend(handles=poignees, loc="upper left", bbox_to_anchor=(0.02, 1.0),
                   frameon=False, fontsize=9.5, labelspacing=0.62, handlelength=2.2)
    legende.text(0.02, 0.04, "red band: the breakdown\ndotted line: the account leaves the context",
                 transform=legende.transAxes, fontsize=8.5, color="#6B7684", va="bottom")

    for ax in (axes[1][0], axes[1][1], axes[0][2]):
        ax.set_xlabel("simulated day")
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("declared score (0-10)")

    fig.tight_layout()
    _ecrire(fig, sortie)


if __name__ == "__main__":
    # Figure store: `figures/` of the papers repository (`PAPER_DIR`, ticket 115). The copy to
    # `article/images/` went away with that folder, renamed `article_old_do_not_maintain/`.
    stock = chemin_papier("figures")
    exiger_depot_papiers(stock)
    for dst in (stock,):
        dst.mkdir(parents=True, exist_ok=True)
        figure_propension(dst / "tmp_ch7_propension_quotidienne.svg")
        figure_comportement(dst / "tmp_ch7_voiture_par_jour.svg")
        figure_croyances(dst / "tmp_ch7_croyances_voiture.svg")
        figure_affinites_tous_modes(dst / "tmp_ch7_affinites_tous_modes.svg")
