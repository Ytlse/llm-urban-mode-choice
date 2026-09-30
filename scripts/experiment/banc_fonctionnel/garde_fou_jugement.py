#!/usr/bin/env python3
"""The judgement safeguard — ticket 095, author's decision of 2026-09-22.

WHY
--------
Since decision D7 of ticket 100, the severity of a memory is the one the agent estimates, and
only that one. It sets its lifetime. A model that answered "harmless" everywhere
would therefore run fifty days of campaign to measure nothing — which nearly happened on
2026-09-22, for another reason: the event text did not reach the model, and fifteen
judgements out of fifteen answered `negligible` to a blank page.

This script confronts what the agent answers with what we had in mind when analysing the texts,
BEFORE paying for a campaign. Thirty-two calls, about twelve minutes.

WHAT IT ANSWERS, AND IN THIS ORDER
----------------------------------
1. **The gap to the expected.** A judgement outside its range is not an error in itself — it is
   a signal. Beyond `part_hors_plage_max`, the campaign does not start.
2. **The spread between texts.** It is the criterion that DECIDES. If all texts receive the same
   level, the experiment on duration will measure nothing, however accurate each
   judgement taken in isolation. And if the three incidents do not separate from one another,
   it is the three arms of the experiment that collapse, not just a statistic.
3. **The share of the profile.** The same text submitted to several personas. The gap between
   profiles is EXPECTED — "perhaps, depending on the person's profile, the answer will not
   systematically be the same" — and it is measured instead of being treated as noise.

⚠ WHAT IT CANNOT DO: say whether the agent is RIGHT. It says whether the setup still produces
the spread on which the analysis of the texts relied.

⚠ THE VALUE OF A RANGE DEPENDS ON WHO WROTE IT AND WHEN. The grid carries a `deja_vu` field per
text. A range written after seeing answers proves almost nothing; the report therefore
separates the two populations and never mixes their rates.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
AGENTS = RACINE / "services" / "llm-agents"
for chemin in (str(RACINE), str(AGENTS)):
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

from scripts.experiment.banc_fonctionnel import stubs  # noqa: E402
from scripts.experiment.banc_fonctionnel.clients import (  # noqa: E402
    GROQ,
    BudgetEpuise,
    ClientEpingle,
    Journal,
)

# The grid is the author's prediction, frozen before the measurement: it lives next to its bench.
GRILLE = Path(__file__).resolve().with_name("grille_attendus.yaml")
# Fallback when the grid does not declare its ordering. The TARGET order belongs to the author and
# is read from the grid (key `marche:`): coding it here would amount to the tester deciding
# the hypothesis they test.
BRAS = ("c2_crevaison", "c6_voiture_suspecte", "c3_panne_reseau")


def charger_grille(chemin: Path) -> tuple[dict, dict, tuple[str, ...]]:
    """The grid, its thresholds and its ordering.

    An empty range STOPS: a half-filled grid guards nothing. An arm named in the
    ordering but absent from the grid STOPS too — the order would bear on a text that is not
    measured.
    """
    import yaml

    brut = yaml.safe_load(chemin.read_text("utf-8"))
    seuils = brut.pop("seuils", {}) or {}
    marche = tuple(brut.pop("marche", None) or BRAS)
    vides = [k for k, v in brut.items() if not (v or {}).get("attendu")]
    if vides:
        raise SystemExit(
            f"incomplete grid: {vides} have no expected range. A text without an expectation "
            f"would pass the safeguard whatever the model answers — that is worse than no test."
        )
    if absents := [b for b in marche if b not in brut]:
        raise SystemExit(
            f"the ladder names {absents}, which the grid does not declare. The order would bear on "
            f"a text that is not measured."
        )
    return brut, seuils, marche


def texte_de(nom: str) -> str:
    """The text actually served, read by the loader — never copied by hand."""
    from llm.evenements import charger

    e = charger(AGENTS / "config" / "evenements" / f"{nom}.yaml")
    if e.texte_cite is not None:
        return e.texte_cite.servi
    return e.jours[e.premier_jour].texte


async def mesurer(client: ClientEpingle, grille: dict, personas: int, sortie: Path,
                  marche: tuple[str, ...] = BRAS) -> dict:
    from llm.evenements.jugement import JugementRefuse, juger
    from llm.gravite import NIVEAUX

    population = stubs.population_de_banc()[:personas]
    rangs = {niveau: i for i, niveau in enumerate(NIVEAUX)}
    lignes: list[dict] = []

    for nom, attendu in grille.items():
        texte = texte_de(nom)
        plage = set(attendu["attendu"])
        modes_attendus = set(attendu.get("modes_attendus") or [])
        for agent in population:
            ligne = {
                "evenement": nom, "agent_id": agent.person_id,
                "deja_vu": "oui" if attendu.get("deja_vu") else "non",
                "attendu": "|".join(attendu["attendu"]),
            }
            try:
                j = await juger(
                    client, agent.person_id, stubs.perception_stub(agent), texte,
                    gravite_deterministe=0.0, evenement_id=nom, jour=10,
                )
            except JugementRefuse as err:
                dernier = client.journal.appels[-1]
                ligne |= {"rendu": "SANS RÉPONSE", "dans_la_plage": "",
                          "erreur": (dernier.erreur or str(err))[:140]}
            else:
                hors_modes = sorted(set(j.modes) - modes_attendus) if modes_attendus else []
                ligne |= {
                    "rendu": j.intensite,
                    "gravite": j.importance_retenue,
                    "valence": j.valence,
                    "modes": "|".join(j.modes),
                    "dans_la_plage": "oui" if j.intensite in plage else "NON",
                    # A mode outside the expected is REPORTED; it does not cause a failure. The
                    # expected list is a reading judgement, not a truth.
                    "modes_hors_attendu": "|".join(hors_modes),
                    "rang": rangs[j.intensite],
                    "instance": client.journal.appels[-1].instance,
                }
            lignes.append(ligne)
            _ecrire(sortie / "garde_fou_jugement.csv", lignes)

    return _depouiller(lignes, grille, rangs, marche)


def _depouiller(lignes: list[dict], grille: dict, rangs: dict,
                marche: tuple[str, ...] = BRAS) -> dict:
    rendus = [l for l in lignes if l.get("rendu") and l["rendu"] != "SANS RÉPONSE"]

    def part_hors(population: list[dict]) -> float | None:
        if not population:
            return None
        return round(sum(1 for l in population if l["dans_la_plage"] == "NON") / len(population), 3)

    # ⚠ The two populations NEVER mix. A global rate would add up a blind prediction
    # and a range written after seeing the answers, and would mean nothing.
    aveugles = [l for l in rendus if l["deja_vu"] == "non"]
    vues = [l for l in rendus if l["deja_vu"] == "oui"]

    par_evenement: dict[str, Counter] = defaultdict(Counter)
    for l in rendus:
        par_evenement[l["evenement"]][l["rendu"]] += 1

    # The spread: how many distinct levels, and the MODAL level of each text.
    modal = {e: c.most_common(1)[0][0] for e, c in par_evenement.items()}
    # The share of the profile: a text on which all personas agree says nothing about the profile.
    desaccord = {e: len(c) for e, c in par_evenement.items()}

    # Do the arms separate, and in the target order?
    #
    # ⚠ A STRICT GAP IS ONLY REQUIRED WHERE THE GRID PREDICTS IT. Two arms whose expected
    # ranges overlap were not separated by the author; asking the measurement to
    # separate them would amount to testing a hypothesis nobody stated. Everywhere, on the
    # other hand, the order must not GO DOWN: an arm declared more severe that receives a lower
    # level contradicts the ordering, overlap or not.
    bras = [(b, modal.get(b)) for b in marche if b in modal]
    ordre_tenu, ordre_detail = True, []
    for (gauche, ng), (droite, nd) in zip(bras, bras[1:]):
        if ng is None or nd is None:
            ordre_tenu = False
            continue
        plages_disjointes = not (set(grille[gauche]["attendu"]) & set(grille[droite]["attendu"]))
        if rangs[nd] < rangs[ng]:
            ordre_tenu = False
            ordre_detail.append(f"{droite} ({nd}) est SOUS {gauche} ({ng})")
        elif plages_disjointes and rangs[nd] == rangs[ng]:
            ordre_tenu = False
            ordre_detail.append(
                f"{droite} et {gauche} sont au même échelon ({ng}) alors que leurs plages "
                f"attendues sont disjointes"
            )
        elif not plages_disjointes and rangs[nd] == rangs[ng]:
            ordre_detail.append(
                f"{gauche} et {droite} à égalité ({ng}) — NON départagés par la grille, "
                f"donc non comptés contre la marche"
            )

    return {
        "appels": len(lignes),
        "sans_reponse": len(lignes) - len(rendus),
        "part_hors_plage_aveugle": part_hors(aveugles),
        "part_hors_plage_deja_vue": part_hors(vues),
        "echelons_distincts": len(set(modal.values())),
        "echelon_modal_par_texte": modal,
        "personas_en_desaccord": desaccord,
        "bras_dans_l_ordre_predit": ordre_tenu,
        "marche": list(marche),
        "ordre_detail": ordre_detail,
        "bras": dict(bras),
    }


def _ecrire(chemin: Path, lignes: list[dict]) -> None:
    if not lignes:
        return
    colonnes: list[str] = []
    for l in lignes:
        for k in l:
            if k not in colonnes:
                colonnes.append(k)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        w.writerows(lignes)


def verdict(bilan: dict, seuils: dict) -> tuple[bool, list[str]]:
    """Does the safeguard pass? Each refusal says what to do, not only that it refuses."""
    motifs: list[str] = []
    ok = True

    if bilan["sans_reponse"]:
        ok = False
        motifs.append(
            f"⛔ {bilan['sans_reponse']} appel(s) sans réponse. Une exposition non jugée n'est "
            f"pas une exposition anodine : elle n'a pas eu lieu. Corriger la passerelle avant "
            f"toute campagne."
        )

    minimum = int(seuils.get("echelons_distincts_min", 3))
    if bilan["echelons_distincts"] < minimum:
        ok = False
        motifs.append(
            f"⛔ {bilan['echelons_distincts']} échelon(s) distinct(s) sur les textes, {minimum} "
            f"demandés. Toutes les durées de vie seraient voisines et l'expérience sur la durée "
            f"ne mesurerait rien. Revoir les textes ou le gabarit, pas le seuil."
        )

    if not bilan["bras_dans_l_ordre_predit"]:
        ok = False
        motifs.append(
            f"⛔ la marche n'est pas tenue : {'; '.join(bilan.get('ordre_detail') or []) or bilan['bras']}. "
            f"C'est l'hypothèse même de l'expérience ; la lancer ainsi coûterait une campagne par "
            f"bras pour l'apprendre."
        )
    elif bilan.get("ordre_detail"):
        motifs.append("⚠ " + " ; ".join(bilan["ordre_detail"]))

    plafond = float(seuils.get("part_hors_plage_max", 0.20))
    part = bilan["part_hors_plage_aveugle"]
    if part is not None and part > plafond:
        ok = False
        motifs.append(
            f"⛔ {part:.0%} des jugements AVEUGLES sortent de leur plage, plafond {plafond:.0%}. "
            f"Ce taux-là compte : la plage a été écrite sans avoir vu une réponse."
        )

    vue = bilan["part_hors_plage_deja_vue"]
    if vue is not None and vue > plafond:
        motifs.append(
            f"⚠ {vue:.0%} des jugements sortent de plages écrites APRÈS avoir vu des réponses. "
            f"Signalé, jamais bloquant : une plage ajustée après coup ne prouve rien dans un "
            f"sens comme dans l'autre."
        )
    return ok, motifs


async def _jouer(a) -> int:
    grille, seuils, marche = charger_grille(Path(a.grille))
    personas = int(a.personas or seuils.get("personas", 4))
    sortie = Path(a.sortie)
    journal = Journal()
    client = ClientEpingle(
        instances=tuple(a.instances), base_url=a.passerelle,
        budget_jetons=a.budget, pause=a.pause, journal=journal,
    )
    print(f"Judgement safeguard — {len(grille)} texts × {personas} personas = "
          f"{len(grille) * personas} calls, instances {list(a.instances)}.")
    try:
        bilan = await mesurer(client, grille, personas, sortie, marche)
    except BudgetEpuise as err:
        print(f"⏸ {err}")
        return 2
    finally:
        journal.ecrire(sortie / "garde_fou_passerelle.csv")

    print(json.dumps(bilan, ensure_ascii=False, indent=2))
    ok, motifs = verdict(bilan, seuils)
    for m in motifs:
        print(m)
    print("\n" + ("✅ The safeguard passes — the campaign can be launched."
                  if ok else "⛔ The safeguard REFUSES. The campaign does not start."))
    print(f"{journal.resume()}\nDetail: {sortie / 'garde_fou_jugement.csv'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grille", default=str(GRILLE))
    p.add_argument("--personas", type=int, default=None)
    p.add_argument("--instances", nargs="+", default=list(GROQ))
    p.add_argument("--passerelle", default="http://localhost:8000")
    p.add_argument("--budget", type=int, default=6000)
    p.add_argument("--pause", type=float, default=20.0)
    p.add_argument("--sortie", default="docs/traces/garde_fou_jugement_095")
    return asyncio.run(_jouer(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
