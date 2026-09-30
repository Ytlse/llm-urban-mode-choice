#!/usr/bin/env python3
"""From a campaign run to the frozen grid — ticket 059, lot 5 (bridge).

WHAT THIS MODULE FILLS IN
-------------------------
`scoring.py` holds all the computation — paired gap, signs, agreement, kappa, vacuity guards —
but it was written for **stage 1**, which paired by CONDITION: the same trips
played once without an article (C1) and once with it (C2), without a simulator. That stage no
longer exists.

A campaign does not pair by condition, it pairs by **role** and by **relative day**:

| Stage 1 (no longer exists) | Campaign (this module) |
|---|---|
| same trips, two conditions | same agents, before / after publication |
| C1 → C2 | `avant` → `apres`, per role |
| no run | one run, `moves.csv` + `evenements.jsonl` |

**The counterfactual is no longer a condition, it is a period.** We compare a reader's modal
share before their publication with the one after. The co-resident gives the spread; the control,
when the run has one, gives the common drift.

⚠ **The noise floor has no default, and this module does not invent one.** Since the
control household is no longer required (059 § 6.4, 2026-09-22), it is no longer read from the
run: it is taken from 095 § 7 bis (3.2 % — **one** measurement, **one** persona) or re-established
by an identical replay. It is **declared** on the command line, and copied into the report
header. A gap smaller than it is not an effect.

⚠ **Each agent is its own control, and that has a price.** A model drift over the course of the
run would read as an effect. That is what the `temoin` role is there to rule out when it is
present; without it, the drift is not measured and the report says so in plain words.

    python -m scripts.analysis.presse.campagne <run> --plancher 0.032
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RACINE))

from scripts.analysis.presse.grille import charger_grille  # noqa: E402
from scripts.analysis.presse.scoring import (  # noqa: E402
    NON_CONCLUANT,
    EcartModal,
    accord_de_signe,
    kappa_pondere,
    signes_observes,
)

# Grid mode → canonical mode of the repository. EXPLICIT: three vocabularies coexist, and a
# word passed through "by default" would score the prediction of one mode against another's.
MODE_GRILLE_VERS_CANONIQUE = {
    "voiture": "car",
    "velo": "cycling",
    "marche": "walking",
    "tc": "public_transport",
}

# Column of `moves.csv` holding the chosen mode, and the role column.
COLONNE_MODE = "Mode de transport Choisi"
COLONNE_ROLE = "Rôle"
COLONNE_JOUR = "Jour relatif au choc"
# ⚠ "Référence" holds the RUN identifier, not the agent's — found on 2026-09-22 on
# a real run, after the bench had let it through: its stubs wrote an agent identifier
# into that column, so they were testing their own convention. The agent is under
# "ID Personne".
COLONNE_AGENT = "ID Personne"
COLONNE_EVENEMENT = "Choc"

# Minimum count per (role, period). Lower than stage 1's (30 trips) because a
# six-household campaign will never produce that many per mode and per period — and higher
# than 1, because a single trip that flips moves a share by a hundred points.
EFFECTIF_MIN_PERIODE = 8


def _decisions(run: Path) -> list[dict]:
    chemin = run / "moves.csv"
    if not chemin.is_file():
        raise SystemExit(f"❌ {chemin} introuvable")
    with chemin.open(encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        manquantes = [
            c for c in (COLONNE_MODE, COLONNE_ROLE, COLONNE_JOUR)
            if c not in (lecteur.fieldnames or [])
        ]
        if manquantes:
            raise SystemExit(
                f"❌ columns missing from {chemin}: {manquantes}. A run predating "
                f"ticket 100 batch 5 does not carry the role, and it cannot be reconstructed "
                f"afterwards — one would have to know who was exposed."
            )
        return list(lecteur)


def _evenement_du_run(run: Path) -> str:
    for nom in ("evenements.jsonl", "chocs.jsonl"):
        chemin = run / nom
        if not chemin.is_file():
            continue
        for brut in chemin.read_text("utf-8").splitlines():
            if not brut.strip():
                continue
            try:
                ligne = json.loads(brut)
            except json.JSONDecodeError:
                continue
            return str(ligne.get("evenement_id") or ligne.get("choc_id") or "")
    return ""


def parts_par_role(
    run: Path, *, article: str
) -> tuple[dict[tuple[str, str], EcartModal], dict[str, dict[str, int]]]:
    """Gap before / after publication, per role and per grid mode.

    Returns `({(role, mode_grille): EcartModal}, effectifs)`. The grid mode is translated into
    the canonical vocabulary once, here.
    """
    lignes = _decisions(run)
    from scripts.analysis.lecture_avant_decision import informes_du_run, sous_role

    informes = informes_du_run(run)
    # `avant` = relative day < 0; `apres` = relative day >= 0. Day 0 is the one on which the agent
    # receives the text: it decides AFTER reading it, since intake is at wake-up. Filing it
    # under "avant" would dilute the effect of the first day, which is the one we look for.
    seaux: dict[tuple[str, str], list[str]] = defaultdict(list)
    effectifs: dict[str, dict[str, int]] = defaultdict(lambda: {"avant": 0, "apres": 0})
    for ligne in lignes:
        role = (ligne.get(COLONNE_ROLE) or "").strip()
        mode = (ligne.get(COLONNE_MODE) or "").strip()
        brut_jour = (ligne.get(COLONNE_JOUR) or "").strip()
        if not role or not mode or not brut_jour:
            continue
        try:
            jour = int(brut_jour)
        except ValueError:
            continue
        periode = "apres" if jour >= 0 else "avant"
        # Ticket 111 — the co-resident ALSO counts under its sub-role (informed or not), without
        # leaving `co_resident`: campaigns already scored under that role do not move.
        for r in dict.fromkeys((role, sous_role(role, ligne.get("ID Personne") or "", informes))):
            seaux[(r, periode)].append(mode)
            effectifs[r][periode] += 1

    ecarts: dict[tuple[str, str], EcartModal] = {}
    for role in sorted({r for r, _ in seaux}):
        avant = seaux.get((role, "avant")) or []
        apres = seaux.get((role, "apres")) or []
        for mode_grille, canonique in MODE_GRILLE_VERS_CANONIQUE.items():
            if len(avant) < EFFECTIF_MIN_PERIODE or len(apres) < EFFECTIF_MIN_PERIODE:
                # Vacuity guard: "non concluant", never 0.0. In this repository, a missing
                # measurement readily yields the perfect value, and this pattern has lied before.
                ecarts[(role, mode_grille)] = EcartModal(
                    article, mode_grille, None, None,
                    min(len(avant), len(apres)), verdict=NON_CONCLUANT,
                )
                continue
            ecarts[(role, mode_grille)] = EcartModal(
                article=article,
                mode=mode_grille,
                part_reference=avant.count(canonique) / len(avant),
                part_condition=apres.count(canonique) / len(apres),
                n_apparies=min(len(avant), len(apres)),
            )
    return ecarts, dict(effectifs)


def depouiller(
    runs: list[Path], *, grille_chemin: Path, plancher: float, role: str = "expose"
) -> dict:
    """The full tally: one run per article, scored against the frozen grid."""
    grille = charger_grille(grille_chemin)
    observes: dict[tuple[str, str], str | None] = {}
    intensites: dict[tuple[str, str], int | None] = {}
    detail: list[tuple[str, dict]] = []

    for run in runs:
        article = _evenement_du_run(run)
        if not article:
            raise SystemExit(f"❌ {run} carries no event: nothing to score")
        ecarts, effectifs = parts_par_role(run, article=article)
        du_role = [e for (r, _m), e in ecarts.items() if r == role]
        signes = signes_observes(du_role, plancher_de_bruit=plancher)
        for (_a, mode), signe in signes.items():
            observes[(article, mode)] = signe
            # The observed intensity is NOT derived from the gap: the grid expresses it on an
            # ordinal 0-3 scale that nothing in a modal share can recover. It therefore
            # stays empty here, and the kappa will come out "non concluant". That is correct: this
            # measurement needs the third level — what the agent understood of the text — which
            # is read in `evenements.jsonl` under `intensite_jugee`, not in `moves.csv`.
            intensites[(article, mode)] = None
        detail.append((article, {"effectifs": effectifs, "ecarts": du_role}))

    accord = accord_de_signe(grille, observes)
    return {
        "grille": grille,
        "accord": accord,
        "kappa": kappa_pondere(grille, intensites),
        "observes": observes,
        "detail": detail,
        "plancher": plancher,
        "role": role,
    }


def rendre(resultat: dict) -> str:
    grille = resultat["grille"]
    accord = resultat["accord"]
    lignes = [
        "# Dépouillement de campagne — presse locale",
        "",
        f"- grille gelée : `{grille.empreinte[:16]}…`",
        f"- plancher de bruit déclaré : **{resultat['plancher']:.3f}** "
        f"({resultat['plancher'] * 100:.1f} %)",
        f"- rôle scoré : `{resultat['role']}`",
        "",
        "⚠ Le plancher de bruit est **déclaré**, pas mesuré par ce run. Depuis que le foyer",
        "témoin n'est plus exigé, il se reprend du ticket 095 § 7 bis — une mesure, un persona —",
        "ou se réétablit par un rejeu à l'identique. Un écart plus petit que lui n'est pas un effet.",
        "",
        "## Accord de signe",
        "",
    ]
    if accord.verdict:
        lignes.append(f"**{accord.verdict}** — aucune cellule lisible.")
    else:
        lignes.append(
            f"{accord.concordants} concordances sur {accord.lisibles} cellules lisibles "
            f"({accord.taux:.0%}), {accord.non_lisibles} non lisibles sur les "
            f"{len(grille.cellules)} de la grille."
        )
        if accord.intervalle:
            lignes.append(
                f"Intervalle à 95 %, rééchantillonné **par événement** : "
                f"[{accord.intervalle[0]:.0%} ; {accord.intervalle[1]:.0%}]."
            )
        else:
            lignes.append(
                "Intervalle **non calculable** : moins de deux événements distincts. Un "
                "intervalle tiré sur un seul groupe ne rééchantillonne rien."
            )
        lignes.append("")
        lignes.append(
            "⚠ Pas de test binomial : les cellules d'un même événement ne sont pas "
            "indépendantes — leurs parts somment à un, et un seul comportement en produit "
            "plusieurs. L'incertitude est groupée par article."
        )

    lignes += ["", f"## Kappa pondéré", "", f"{resultat['kappa']}", ""]
    if resultat["kappa"] == NON_CONCLUANT:
        lignes.append(
            "L'intensité ordinale ne se dérive pas d'une part modale. Elle demande le "
            "troisième niveau de mesure — ce que l'agent a compris du texte — qui se lit dans "
            "`evenements.jsonl` sous `intensite_jugee`."
        )

    lignes += ["", "## Effectifs, par article et par rôle", ""]
    for article, contenu in resultat["detail"]:
        lignes.append(f"**{article}**")
        for role, compte in sorted(contenu["effectifs"].items()):
            lignes.append(
                f"- `{role}` : {compte['avant']} décision(s) avant parution, "
                f"{compte['apres']} après"
            )
        lignes.append("")
    return "\n".join(lignes)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+", help="one run per article")
    p.add_argument(
        "--plancher", type=float, required=True,
        help="noise floor, as a SHARE (0.032 for 3.2 %%). Required: a default would get it "
             "forgotten, and a gap smaller than the noise is not an effect",
    )
    p.add_argument(
        "--role", default="expose",
        choices=("expose", "co_resident", "co_resident_informe", "co_resident_non_informe",
                 "temoin"),
        help="co_resident_informe / co_resident_non_informe: ticket 111, read from "
             "relais_foyer.jsonl",
    )
    p.add_argument(
        "--grille", type=Path,
        default=RACINE / "data" / "presse" / "grille_signes.yaml",
    )
    p.add_argument("-o", "--sortie", type=Path)
    args = p.parse_args()

    rendu = rendre(
        depouiller(args.runs, grille_chemin=args.grille,
                   plancher=args.plancher, role=args.role)
    )
    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(rendu, encoding="utf-8")
        print(f"✅ {args.sortie}")
    else:
        print(rendu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
