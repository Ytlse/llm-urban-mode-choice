#!/usr/bin/env python3
"""Figure — drop and return, by role, for both regimes (ticket 100, lot 5).

A SINGLE FIGURE FOR BOTH REGIMES, and that is the whole point of the ticket. A shock suffered
on arrival and an article read on waking produce the same curve: the propensity to the target
mode, by day RELATIVE to the event, and by role. Figure 7.2 of the manuscript is its first
instance; § 7.3 will be the second, without one more script.

THREE ROLES, AND NONE CAN BE DEDUCED FROM ANOTHER
--------------------------------------------------------
`expose` — the event reached them. `co_resident` — lives under the same roof as an exposed
agent and received NOTHING: this is where what is passed on can be read, and without it the
diffusion stage has no object. `temoin` — neither, this is the run's baseline.

THE X AXIS IS THE RELATIVE DAY, AND IT EXISTS EVERY DAY
--------------------------------------------------------------
Including before the event and long after. An x axis that only existed on event
days would plot nothing — and that is exactly what `moves.csv` writes for EVERY
decision, including nominal days.

VACUITY GUARD, NOT NEGOTIABLE
---------------------------------
A role whose count falls below the declared minimum comes out **"not conclusive"**, never a
curve. In this repository, the absence of measurement readily produces the perfect value, and
this pattern has lied before. A curve drawn on two decisions looks exactly like a curve drawn
on two hundred.

Standard library only, inline SVG, modelled on `figure_fenetre_choc.py`.
Figure composed in ENGLISH: all of the paper's figures are.

    python scripts/analysis/figure_evenement.py <run> --mode car -o <sortie.svg>
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis import lecture_avant_decision as _lad  # noqa: E402
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402

# Validated categorical palette (dataviz, slots 1-3). Do not replace without revalidating:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light
ROLES = (
    ("expose", "Exposed", "#eb6834", ""),
    ("co_resident", "Co-resident (heard only)", "#2a78d6", ""),
    ("temoin", "Control", "#1baf7a", ""),
)
# Ticket 111 — when the reader passed it on to their household (`relais_foyer.jsonl`), the
# co-resident splits in two. Same hue, distinct stroke: the validated palette has only three
# slots, and the two sub-roles are two halves of the same group.
ROLES_RELAIS = (
    ("expose", "Exposed", "#eb6834", ""),
    ("co_resident_informe", "Co-resident, told", "#2a78d6", ""),
    ("co_resident_non_informe", "Co-resident, not told", "#2a78d6", "6 4"),
    ("temoin", "Control", "#1baf7a", ""),
)

# Canonical mode → probability column of `moves.csv`. EXPLICIT and not guessed: three
# mode vocabularies coexist in this repository (ticket 077, lot A), and a word passing
# "by default" would plot another mode's curve without anything saying so.
COLONNE_PROBA = {
    "walking": "P(Marche) %",
    "cycling": "P(Vélo) %",
    "car": "P(Voiture Privée) %",
    "public_transport": "P(Transports_collectifs) %",
    "train": "P(Train) %",
    "motorbike": "P(Deux-roues motorisé) %",
}
LIBELLE_EN = {
    "walking": "walking", "cycling": "cycling", "car": "car",
    "public_transport": "public transport", "train": "train", "motorbike": "motorbike",
}

# Minimum count per (role, relative day) below which the point is not plotted. Declared here,
# and restated on the figure: a threshold not readable on the image cannot be discussed.
EFFECTIF_MINIMAL = 3

L, R, T, B = 66, 170, 62, 66
W, H = 1180, 500


def series(run: Path, mode: str) -> tuple[dict, dict, str]:
    """Propensity to the target mode by (role, relative day), and the counts behind it.

    Returns `({role: {jour: moyenne}}, {role: {jour: effectif}}, event identifier)`.
    """
    colonne = COLONNE_PROBA[mode]
    informes = _lad.informes_du_run(run)
    roles = ROLES if informes is None else ROLES_RELAIS
    valeurs: dict[str, dict[int, list[float]]] = {r: {} for r, _, _, _ in roles}
    evenement = ""
    chemin = run / "moves.csv"
    if not chemin.is_file():
        raise SystemExit(f"❌ {chemin} not found — this run has no decision log")

    with chemin.open(encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        if colonne not in (lecteur.fieldnames or []):
            raise SystemExit(
                f"❌ column \"{colonne}\" missing from {chemin}. Probability columns "
                f"present: {[c for c in (lecteur.fieldnames or []) if c.startswith('P(')]}"
            )
        if "Rôle" not in (lecteur.fieldnames or []):
            raise SystemExit(
                f"❌ column « Rôle » missing from {chemin}: this run predates ticket 100, "
                f"batch 5. The roles cannot be reconstructed afterwards — one would have to "
                f"know who was exposed, and that is precisely what the column carries."
            )
        for ligne in lecteur:
            role = _lad.sous_role(
                (ligne.get("Rôle") or "").strip(), ligne.get("ID Personne") or "", informes
            )
            relatif = (ligne.get("Jour relatif au choc") or "").strip()
            brut = (ligne.get(colonne) or "").strip()
            # An EMPTY cell is not a zero: it is a decision without a distribution
            # (single choice, inherited cache). Counting it as 0 % would drop the curve for a
            # reason that has nothing to do with the event.
            if not role or not relatif or not brut:
                continue
            evenement = evenement or (ligne.get("Choc") or "").strip()
            try:
                valeurs.setdefault(role, {}).setdefault(int(relatif), []).append(float(brut))
            except ValueError:
                continue

    moyennes = {
        role: {
            j: statistics.mean(v) for j, v in sorted(par.items())
            if len(v) >= EFFECTIF_MINIMAL
        }
        for role, par in valeurs.items()
    }
    effectifs = {role: {j: len(v) for j, v in sorted(par.items())}
                 for role, par in valeurs.items()}
    return moyennes, effectifs, evenement


def _x(j: int, jmin: int, jmax: int) -> float:
    if jmax == jmin:
        return (L + W - R) / 2
    return L + (j - jmin) / (jmax - jmin) * (W - L - R)


def _y(p: float) -> float:
    return T + (100 - p) / 100 * (H - T - B)


def composer(moyennes: dict, effectifs: dict, mode: str, evenement: str) -> str:
    jours = sorted({j for par in moyennes.values() for j in par})
    if not jours:
        raise SystemExit(
            "❌ inconclusive: no point gathers the minimum count of "
            f"{EFFECTIF_MINIMAL} decisions. This is NOT a null result — it is an absence "
            "of measurement, and an empty figure would read as an absence of effect."
        )
    jmin, jmax = jours[0], jours[-1]
    o: list[str] = []
    a = o.append
    libelle = LIBELLE_EN[mode]

    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
      f'height="{H}" font-family="Inter, Helvetica, Arial, sans-serif" role="img" '
      f'aria-label="Propensity to use {libelle} by day relative to the event, for three roles">')
    a("<style>"
      ".surface{fill:#fcfcfb}.ink{fill:#1a1a19}.ink2{fill:#5c5b54}.grid{stroke:#e6e5e0}"
      ".rule{stroke:#c9c8c1}"
      "@media (prefers-color-scheme: dark){"
      ".surface{fill:#1a1a19}.ink{fill:#ffffff}.ink2{fill:#c3c2b7}"
      ".grid{stroke:#333330}.rule{stroke:#4a4944}}"
      "</style>")
    a(f'<rect class="surface" width="{W}" height="{H}"/>')

    # The title names the RESULT, not the axes.
    a(f'<text class="ink" x="{L}" y="26" font-size="17" font-weight="600">'
      f'Who changes, and for how long</text>')
    a(f'<text class="ink2" x="{L}" y="45" font-size="12.5">'
      f'Propensity to choose {libelle}, by day relative to the event. Exposed agents met the '
      f'event; co-residents only heard about it; controls neither.</text>')

    for p in range(0, 101, 25):
        y = _y(p)
        a(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke-width="1"/>')
        a(f'<text class="ink2" x="{L - 10}" y="{y + 4:.1f}" font-size="11.5" '
          f'text-anchor="end">{p}%</text>')

    # Day 0: the one when the event reaches the exposed agent.
    if jmin <= 0 <= jmax:
        x0 = _x(0, jmin, jmax)
        a(f'<line x1="{x0:.1f}" y1="{T - 6}" x2="{x0:.1f}" y2="{H - B}" stroke="#5c5b54" '
          f'stroke-width="1.5" stroke-dasharray="4 4" opacity="0.75"/>')
        a(f'<text class="ink2" x="{x0 + 5:.1f}" y="{T + 6}" font-size="11">'
          f'day 0 · the event reaches the exposed agent</text>')

    a(f'<line class="rule" x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke-width="1"/>')
    for j in jours:
        if j % 5 == 0 or j == 0:
            x = _x(j, jmin, jmax)
            a(f'<text class="ink2" x="{x:.1f}" y="{H - B + 18}" font-size="11" '
              f'text-anchor="middle">{j:+d}</text>')
    a(f'<text class="ink2" x="{(L + W - R) / 2:.0f}" y="{H - B + 38}" font-size="11.5" '
      f'text-anchor="middle">days relative to the event (weekends are not simulated)</text>')
    a(f'<text class="ink2" x="{L - 46}" y="{T - 14}" font-size="11.5">P({libelle})</text>')

    absents = []
    roles = ROLES_RELAIS if any(r.startswith("co_resident_") for r in moyennes) else ROLES
    for role, nom, couleur, tirets in roles:
        points = moyennes.get(role) or {}
        if not points:
            # "Not conclusive" WRITTEN on the figure, not a missing curve: a missing series
            # reads as a null effect to whoever does not know it is missing.
            absents.append(nom)
            continue
        pts = [(_x(j, jmin, jmax), _y(p)) for j, p in sorted(points.items())]
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        dash = f' stroke-dasharray="{tirets}"' if tirets else ""
        a(f'<path d="{d}" fill="none" stroke="{couleur}" stroke-width="2"{dash} '
          f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            a(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{couleur}" '
              f'stroke="#fcfcfb" stroke-width="1.5"/>')
        xf, yf = pts[-1]
        a(f'<text x="{xf + 12:.1f}" y="{yf + 4:.1f}" font-size="12" font-weight="600" '
          f'fill="{couleur}">{nom}</text>')

    note = (
        f"Event: {evenement or 'undeclared'} · points shown only where at least "
        f"{EFFECTIF_MINIMAL} decisions back them"
    )
    if absents:
        note += " · not conclusive for: " + ", ".join(absents)
    a(f'<text class="ink2" x="{L}" y="{H - 14}" font-size="11">{note}</text>')

    a("</svg>")
    return "".join(o)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="run directory (containing moves.csv)")
    p.add_argument("--mode", required=True, choices=sorted(COLONNE_PROBA),
                   help="the mode targeted by the event")
    p.add_argument("-o", "--sortie", type=Path,
                   default=sortie_papier("figures", "figure_evenement.svg"))
    args = p.parse_args()
    exiger_depot_papiers(args.sortie)

    moyennes, effectifs, evenement = series(args.run, args.mode)
    roles = ROLES_RELAIS if any(r.startswith("co_resident_") for r in moyennes) else ROLES
    for role, nom, _, _ in roles:
        total = sum((effectifs.get(role) or {}).values())
        retenus = len(moyennes.get(role) or {})
        print(f"  {nom:28s} {total:5d} decision(s), {retenus} day(s) above the threshold")

    svg = composer(moyennes, effectifs, args.mode, evenement)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(svg, encoding="utf-8")
    try:
        from scripts.analysis.figures_versionnees import signaler

        signaler([args.sortie])
    except Exception:  # noqa: BLE001 — the warning never makes a figure fail
        pass
    print(f"✅ {args.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
