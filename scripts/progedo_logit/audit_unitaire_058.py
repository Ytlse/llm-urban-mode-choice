"""audit_unitaire_058.py — Decision-by-decision agreement, against the declared mode.

The aggregated composite score says whether a population behaves like the survey. It does not say whether
**this** particular trip received the mode the person declared. That is what this
script measures, on the days actually described by the respondents (ticket 058).

## What is computed, and for whom

| Quantity | On which decision-makers |
|---|---|
| Accuracy, weighted and unweighted | all |
| 4 × 4 confusion matrix, counts and weighted | all |
| Recall and precision per mode | all |
| Accuracy per distance band | all |
| Cross-entropy and GMPCA | **only those that return a distribution, on the common support** |
| Zero rate on offered mode | all those that return a distribution |

The cross-entropy of a hard decision is infinite from the first error onwards. Smoothing a
baseline to give it a figure would produce a value that depends on the smoothing and on nothing
else: the « always the car » and « minimum duration » baselines therefore get none,
and the column stays empty for them.

**It is computed on the COMMON SUPPORT: the arbitrated decisions that all the compared
decision-makers score.** An exact zero has no cross-entropy — the term is minus infinity, and
the library bounds it at its truncation threshold, so that the mean would measure the threshold
rather than the decision-maker. Measured on the gradient boosting: **4.21 nats counting the zeros, 0.42
without them**. Publishing the 4.21 alone would be publishing `1e-15`.

Setting these decisions aside decision-maker by decision-maker, on the other hand, makes the values incomparable:
each is then scored on the set it picks for itself by deciding more or less hard, 5,923
decisions for the expert prompt versus 6,588 for the gradient boosting, and the hardest-deciding one
removes the most of its own failures. Hence the intersection, and `--definisseur` to say who
defines it. Fixed on 2026-09-21: the order of decision-makers is changed by it, cf.
`docs/traces/2026-09-21_11-45_ticket058_entropie_support_commun/`.

What the common support throws away is published alongside, as a result: **the rate of decisions where the
declared mode was among the presented options and where the decision-maker gave it zero**. It does not
depend on any probability floor, and it opposes two families of outputs — a softmax does not
produce an exact zero, a verbalised distribution does.

These zeros have two causes, which must be separated because they do not cost the same:

- **the declared mode was not among the presented options** — the chain lock or the
  option cap removed it, and no decision-maker could find it;
- **it was there, and the decision-maker gave it zero** — that is its error, not the rule's.

## Two counters that condition the reading

**Was the declared mode in the offer?** Under the chain constraint, the car is removed
from the offer of part of the trips. When the declared mode is not there, no decision-maker
can find it: the error belongs to the rule, not to it. The count is the ceiling of
the audit, and it is published alongside accuracy.

**How many decisions are arbitrated?** The count is read on the decision, not on the offer of the
set: the vehicle filter and the option cap come in between, and they reduce
a lot. A single-option trip is not a decision — all decision-makers give it
the same answer. Accuracy is therefore published twice: on the whole, and on the
arbitrated decisions only, where decision-makers actually separate.

## The modes

The survey codes four classes: `walk`, `bike`, `car`, `transit`. The platform returns one canonical
mode out of six, reduced from a chain of legs by the EMC² hierarchy (ticket 022,
`mode_hierarchy.primary_canonical`). The mapping is that of the survey recoding, where
motorised two-wheelers fall with the car and the train with public transport.

Usage:
    services/llm-agents/.venv/bin/python scripts/progedo_logit/audit_unitaire_058.py \
        --experience exp_lgbm_jtir_pop-enquete_058_test_jeu-58_test_20260316_nosim
    # without --experience: all experiments played on the audit set
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ICI = Path(__file__).resolve().parent
RACINE = ICI.parents[1]
sys.path.insert(0, str(RACINE / "packages" / "mobility_core" / "src"))
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))
sys.path.insert(0, str(ICI))

from experiences.experience import trouver_dossier_experience  # noqa: E402
from mobility_core.mode_hierarchy import primary_canonical  # noqa: E402
from mode_choice_eval import evaluate_proba  # noqa: E402

JEU_DEFAUT = "enquete_058_test_20260316"
VERITE_DEFAUT = "verite_058_test.csv"

# The four survey classes, in the order of the spec (`TARGET_CLASSES`).
CLASSES = ("bike", "car", "transit", "walk")

# Who defines the common support of the cross-entropy. The baselines are excluded from it:
# `alea` spreads at random and leaves thousands of decisions at zero on the declared mode,
# which it would trim for everyone; `durmin` and `majvoiture` decide hard and do not return
# a distribution. They are SCORED on the support when they cover it, they do not
# define it.
PLANCHERS = ("alea", "durmin", "majvoiture")
INDEX = {c: i for i, c in enumerate(CLASSES)}

# Platform canonical mode → survey class. Motorised two-wheelers fall with
# the car and the train with public transport, as in `MODE_GROUP` of the
# training-set builder: it is the survey recoding that is authoritative, not the
# number of modes the engines can produce.
CANONIQUE_VERS_CLASSE = {
    "walking": "walk",
    "cycling": "bike",
    "car": "car",
    "motorbike": "car",
    "public_transport": "transit",
    "train": "transit",
}

BANDES = ((1, "0-1km"), (2, "1-2km"), (5, "2-5km"), (10, "5-10km"), (20, "10-20km"),
          (50, "20-50km"))
BANDE_LOINTAINE = "plus_50km"


def bande(km: float) -> str:
    for seuil, nom in BANDES:
        if km < seuil:
            return nom
    return BANDE_LOINTAINE


def classe_du_libelle(libelle: str | None) -> str | None:
    """`foot,bus,foot` → `transit`. Reduction by the hierarchy, then survey recoding."""
    if not libelle:
        return None
    canonique = primary_canonical(str(libelle).split(","))
    return CANONIQUE_VERS_CLASSE.get(canonique or "")


def charger_verite(fichier: Path) -> dict[str, dict[str, Any]]:
    """The declared modes, excluding trips whose sequence is broken."""
    with fichier.open(encoding="utf-8") as flux:
        lignes = [r for r in csv.DictReader(flux) if r["chaine_rompue"] == "0"]
    rompus = 0
    with fichier.open(encoding="utf-8") as flux:
        rompus = sum(1 for r in csv.DictReader(flux) if r["chaine_rompue"] == "1")
    logger.info(
        f"Ground truth: {len(lignes)} declared trips "
        f"({rompus} set aside for broken sequence)"
    )
    return {r["activity_id"]: r for r in lignes}


def charger_offre(dossier_jeu: Path) -> dict[str, dict[str, Any]]:
    """Per trip: the offered classes and the number of options."""
    offre: dict[str, dict[str, Any]] = {}
    with (dossier_jeu / "propositions.jsonl").open(encoding="utf-8") as flux:
        for ligne in flux:
            enregistrement = json.loads(ligne)
            propositions = enregistrement.get("propositions") or []
            if not propositions:
                continue
            classes = set()
            for proposition in propositions:
                legs = (proposition.get("plan") or {}).get("legs") or []
                classe = classe_du_libelle(",".join(str(leg.get("mode")) for leg in legs))
                if classe:
                    classes.add(classe)
            offre[enregistrement["activity_id"]] = {
                "classes": classes,
                "n_options": len(propositions),
            }
    logger.info(f"Set offer: {len(offre)} trips with at least one option")
    return offre


def distribution_forcee(decision: dict[str, Any]) -> dict[str, float] | None:
    """The distribution of a forced choice: all the mass on the only presented option.

    This is not smoothing. Rule 3 renormalises over the offered modes; when only one
    remains, the renormalised distribution IS mass 1 on that mode. Reading it this way keeps
    the cross-entropy comparable from one decision-maker to another, instead of refusing it because of
    trips where nobody had a choice.
    """
    presentees = decision.get("presentees") or []
    if decision.get("methode") != "choix_unique" or len(presentees) != 1:
        return None
    canonique = primary_canonical(str(presentees[0].get("mode") or "").split(","))
    return {canonique: 1.0} if canonique else None


def distribution_en_classes(distribution: dict[str, float] | None) -> list[float] | None:
    """The canonical distribution of the decision, folded onto the four classes."""
    if not distribution:
        return None
    masse = [0.0] * len(CLASSES)
    for mode, poids in distribution.items():
        classe = CANONIQUE_VERS_CLASSE.get(mode)
        if classe is None:
            return None  # a mode with no mapping: we do not guess
        masse[INDEX[classe]] += float(poids or 0.0)
    total = sum(masse)
    if total <= 0:
        return None
    return [m / total for m in masse]


def etat_execution(execution: Path) -> str:
    """`terminee`, `en_cours`, `interrompue`… as the run declares it."""
    fichier = execution / "etat.json"
    if not fichier.is_file():
        return "inconnu"
    try:
        return str(json.loads(fichier.read_text(encoding="utf-8")).get("etat") or "inconnu")
    except (OSError, json.JSONDecodeError):
        return "inconnu"


def derniere_execution(dossier_experience: Path) -> Path | None:
    executions = sorted(p for p in (dossier_experience / "executions").glob("*") if p.is_dir())
    return executions[-1] if executions else None


def experiences_du_jeu(dossier_experiences: Path, jeu: str) -> list[str]:
    """The `exp_*` experiments defined on this set, sorted by name.

    They are stored by families (`regime_nominal/<jeu>/<exp>/`): the whole tree is walked,
    excluding `archive/` and `.system_generated/` like the resolver of the `experiences` package."""
    noms = []
    for definition in dossier_experiences.rglob("experience.yaml"):
        parties = definition.relative_to(dossier_experiences).parts
        if "archive" in parties or ".system_generated" in parties:
            continue
        if definition.parent.name.startswith("exp_") and jeu in definition.read_text(
            encoding="utf-8"
        ):
            noms.append(definition.parent.name)
    return sorted(noms)


def derniere_execution_de(nom: str, dossier_experiences: Path) -> Path | None:
    """The last run of an experiment designated by its name, read in its family."""
    dossier = trouver_dossier_experience(nom, racine=dossier_experiences) or (
        dossier_experiences / nom
    )
    return derniere_execution(dossier) if dossier.is_dir() else None


def rappel_precision(matrice: np.ndarray) -> dict[str, dict[str, float]]:
    """Recall and precision per class, read on the confusion matrix (true × predicted)."""
    resultat = {}
    for i, classe in enumerate(CLASSES):
        vrais_positifs = float(matrice[i, i])
        declares = float(matrice[i, :].sum())
        predits = float(matrice[:, i].sum())
        resultat[classe] = {
            "rappel": vrais_positifs / declares if declares else float("nan"),
            "precision": vrais_positifs / predits if predits else float("nan"),
            "declares": declares,
            "predits": predits,
        }
    return resultat


def auditer(execution: Path, verite: dict, offre: dict) -> dict[str, Any]:
    """The metrics of a run, against the declared modes."""
    y, yhat, poids, probas, bandes, arbitrees = [], [], [], [], [], []
    identifiants, offert = [], []
    compte = Counter()
    par_bande: dict[str, list[int]] = defaultdict(list)

    with (execution / "decisions.jsonl").open(encoding="utf-8") as flux:
        for ligne in flux:
            decision = json.loads(ligne)
            reference = verite.get(decision.get("activity_id") or "")
            if reference is None:
                compte["hors_verite"] += 1  # end-of-day closure, or broken sequence
                continue
            retenue = decision.get("retenue") or {}
            classe_choisie = classe_du_libelle(retenue.get("mode"))
            if classe_choisie is None:
                compte["sans_decision"] += 1
                continue
            declare = reference["mode_declare"]
            if declare not in INDEX:
                compte["mode_declare_hors_classes"] += 1
                continue

            info = offre.get(decision["activity_id"]) or {}
            if declare not in (info.get("classes") or set()):
                compte["mode_declare_absent_du_jeu"] += 1
            # What really counts: the options ACTUALLY presented to the decision-maker, after the
            # chain lock and the cap. The set, for its part, carries all modes unfiltered.
            classes_presentees = {
                classe_du_libelle(o.get("mode"))
                for o in (decision.get("presentees") or [])
            }
            if declare not in classes_presentees:
                compte["mode_declare_absent_des_presentees"] += 1
            if info.get("n_options") == 1:
                compte["option_unique_dans_le_jeu"] += 1
            forcee = decision.get("methode") == "choix_unique"
            if forcee:
                compte["decision_forcee"] += 1
            arbitrees.append(0 if forcee else 1)

            identifiants.append(str(decision["activity_id"]))
            offert.append(declare in classes_presentees)
            y.append(INDEX[declare])
            yhat.append(INDEX[classe_choisie])
            poids.append(float(reference["sample_weight"] or 1.0))
            probas.append(distribution_en_classes(decision.get("distribution")))
            b = bande(float(reference["vol_oiseau_declare_km"] or 0.0))
            bandes.append(b)
            par_bande[b].append(1 if classe_choisie == declare else 0)
            compte["notes"] += 1

    if not y:
        logger.error(f"[ALARME] No scorable decision in {execution}: nothing to publish")
        return {}, {}

    y_arr = np.asarray(y)
    yhat_arr = np.asarray(yhat)
    poids_arr = np.asarray(poids)

    matrice = np.zeros((len(CLASSES), len(CLASSES)))
    matrice_ponderee = np.zeros((len(CLASSES), len(CLASSES)))
    for vrai, predit, p in zip(y, yhat, poids, strict=True):
        matrice[vrai, predit] += 1
        matrice_ponderee[vrai, predit] += p

    juste = y_arr == yhat_arr
    arbitrees_arr = np.asarray(arbitrees, dtype=bool)

    resultat: dict[str, Any] = {
        "execution": execution.name,
        "notes": int(compte["notes"]),
        "arbitrees": int(arbitrees_arr.sum()),
        "exactitude_arbitrees": (
            float(juste[arbitrees_arr].mean()) if arbitrees_arr.any() else None
        ),
        "exactitude_forcees": (
            float(juste[~arbitrees_arr].mean()) if (~arbitrees_arr).any() else None
        ),
        "exactitude_non_ponderee": float((y_arr == yhat_arr).mean()),
        "exactitude_ponderee": float(
            poids_arr[y_arr == yhat_arr].sum() / poids_arr.sum()
        ),
        "matrice_effectifs": matrice.astype(int).tolist(),
        "matrice_ponderee": matrice_ponderee.tolist(),
        "par_mode": rappel_precision(matrice),
        "par_bande": {
            b: {"n": len(v), "exactitude": float(np.mean(v))}
            for b, v in sorted(par_bande.items())
        },
        "compteurs": dict(compte),
    }

    # The per-decision detail, returned to the caller so that it builds the common support:
    # activity_id -> (declared mode, distribution, weight), on the arbitrated decisions that
    # carry a distribution, ZEROS INCLUDED. It is the caller that filters.
    notables = {
        identifiants[i]: (int(y[i]), probas[i], float(poids[i]))
        for i in range(len(y))
        if arbitrees[i] and probas[i] is not None
    }

    # Cross-entropy per decision-maker, on ITS own subset: the arbitrated decisions
    # whose distribution leaves a non-zero mass on the declared mode. This reading is not
    # comparable from one decision-maker to another — the harder it decides, the more of its own
    # failures it removes — and it only serves to measure the gap with the common support. It is
    # `cel_weighted_support_commun` that is published.
    retenus = [
        i
        for i, p in enumerate(probas)
        if p is not None and arbitrees[i] and p[y[i]] > 0
    ]
    exclus = [
        i
        for i, p in enumerate(probas)
        if p is not None and arbitrees[i] and p[y[i]] <= 0
    ]
    resultat["mode_declare_a_zero"] = len(exclus)

    # What the common support throws away, and which is nonetheless a result: the decisions where the
    # declared mode WAS among the presented options and where the decision-maker gave it zero. It
    # is not the rule that removed the option, it is the decision-maker that excluded it. The rate compares
    # from one decision-maker to another without depending on any probability floor.
    offerts_arbitres = [
        i for i in range(len(y)) if arbitrees[i] and offert[i] and probas[i] is not None
    ]
    zeros_propres = [i for i in offerts_arbitres if probas[i][y[i]] <= 0]
    resultat["arbitrees_mode_offert"] = len(offerts_arbitres)
    resultat["zeros_mode_offert"] = len(zeros_propres)
    resultat["taux_zero_mode_offert"] = (
        len(zeros_propres) / len(offerts_arbitres) if offerts_arbitres else None
    )
    # A degenerate distribution (all the mass on one mode) is not a distribution: the
    # decision-maker decided hard and wrote it as probabilities. Filtered on the correct
    # decisions, its cross-entropy is 0 by construction — a figure that says nothing.
    degeneree = all(
        max(probas[i]) >= 1.0 - 1e-9 for i in retenus
    ) if retenus else False
    if degeneree:
        resultat["entropie_croisee"] = None
        resultat["entropie_croisee_motif"] = (
            "décision dure : la distribution rendue est dégénérée (masse 1 sur un mode)"
        )
        resultat["distribution_degeneree"] = True
        return resultat, notables
    if retenus and len(retenus) >= int(0.5 * arbitrees_arr.sum()):
        resultat["entropie_croisee_notes"] = len(retenus)
        mesures = evaluate_proba(
            np.asarray([probas[i] for i in retenus]),
            y_arr[retenus],
            poids_arr[retenus],
            list(CLASSES),
        )
        for cle in (
            "cel_weighted", "cel_unweighted", "gmpca_weighted", "gmpca_unweighted"
        ):
            if cle in mesures:
                resultat[cle] = mesures[cle]
    else:
        manquantes = sum(1 for i, p in enumerate(probas) if p is None and arbitrees[i])
        resultat["entropie_croisee"] = None
        resultat["entropie_croisee_motif"] = (
            f"décision dure : {manquantes} décision(s) arbitrée(s) sur "
            f"{int(arbitrees_arr.sum())} sans distribution"
        )
    return resultat, notables


# Two LLM arms differ only by their prompt suffix (`…_promin02`,
# `…_proexp05`): a column too narrow made them identical on display.
LARGEUR_NOM = 22


def support_commun(
    notables: dict[str, dict[str, tuple[int, list[float], float]]],
    resultats: dict[str, dict[str, Any]],
    definisseurs_demandes: list[str] | None = None,
) -> set[str]:
    """The decisions that ALL compared decision-makers score, and the cross-entropy on them.

    Without this, each decision-maker is scored on the decisions where it leaves a non-zero mass on the
    declared mode, that is on a set it chooses itself by deciding more or
    less hard. Two cross-entropies computed on 5,923 and 6,588 decisions do not
    compare. The common support is the intersection, defined by the compared decision-makers — the
    baselines are excluded from it, cf. PLANCHERS — and everyone is scored on it.
    """
    # Who defines the support decides its size: one more decision-maker trims it for
    # everyone. By default, all those that return a non-degenerate distribution; --definisseur
    # restricts to the comparison being published.
    definisseurs = [
        nom
        for nom, detail in notables.items()
        if detail
        and nom not in PLANCHERS
        and (definisseurs_demandes is None or nom in definisseurs_demandes)
        and not resultats.get(nom, {}).get("distribution_degeneree")
        and resultats.get(nom, {}).get("etat_execution") in (None, "terminee")
    ]
    if not definisseurs:
        logger.warning("No decision-maker can define a common support: entropy not published")
        return set()

    support: set[str] | None = None
    for nom in definisseurs:
        avec_masse = {
            cle for cle, (y, proba, _) in notables[nom].items() if proba[y] > 0
        }
        support = avec_masse if support is None else (support & avec_masse)
    support = support or set()

    plancher = min(len(notables[nom]) for nom in definisseurs)
    logger.info(
        f"Common support of the cross-entropy: {len(support)} decisions, "
        f"defined by {len(definisseurs)} decision-makers ({', '.join(sorted(definisseurs))})"
    )
    if plancher and len(support) < 0.60 * plancher:
        logger.error(
            f"[ALARME] Common support too narrow: {len(support)} decisions versus {plancher} "
            f"for the least-covering decision-maker ({len(support) / plancher:.0%}) — the cross-"
            f"entropy would only bear on the easy cases"
        )

    for nom, detail in notables.items():
        resultat = resultats.get(nom)
        if not resultat or resultat.get("distribution_degeneree"):
            continue
        manquantes = support - detail.keys()
        resultat["support_commun_notes"] = len(support) - len(manquantes)
        if manquantes:
            # A decision-maker that does not cover the whole support would be scored on something other than
            # the others: it is exactly what has just been fixed, it is not reintroduced.
            resultat["cel_weighted_support_commun"] = None
            resultat["support_commun_motif"] = (
                f"{len(manquantes)} décision(s) du support commun absente(s) de ce décideur"
            )
            continue
        mesures = evaluate_proba(
            np.asarray([detail[cle][1] for cle in sorted(support)]),
            np.asarray([detail[cle][0] for cle in sorted(support)]),
            np.asarray([detail[cle][2] for cle in sorted(support)]),
            list(CLASSES),
        )
        for cle in ("cel_weighted", "cel_unweighted", "gmpca_weighted", "gmpca_unweighted"):
            if cle in mesures:
                resultat[f"{cle}_support_commun"] = mesures[cle]
    return support


def rendre(resultats: dict[str, dict[str, Any]]) -> None:
    """The table read by a human. Empty values are blanks, not zeros."""
    entete = (
        f"{'décideur':{LARGEUR_NOM}s} {'notés':>7s} {'exact.':>8s} {'pondérée':>9s} "
        f"{'arbitrées':>10s} {'forcées':>8s} {'entropie':>9s} {'support':>8s} "
        f"{'vélo R':>7s} {'TC R':>7s} {'marche R':>9s} {'hors options':>13s} "
        f"{'0/offert':>9s}"
    )
    print("\n" + entete)
    print("-" * len(entete))
    for nom, r in sorted(resultats.items(), key=lambda kv: -kv[1].get("exactitude_ponderee", 0)):
        if not r:
            continue
        if r.get("etat_execution") not in (None, "terminee"):
            nom = f"{nom[: LARGEUR_NOM - 8]} PARTIEL"
        # Published: the cross-entropy of the COMMON SUPPORT. That of the own support stays in
        # the JSON, it compares to nothing.
        ec = r.get("cel_weighted_support_commun")
        taux_zero = r.get("taux_zero_mode_offert")
        par_mode = r.get("par_mode", {})
        print(
            f"{nom[:LARGEUR_NOM]:{LARGEUR_NOM}s} {r['notes']:7d} {r['exactitude_non_ponderee']:8.1%} "
            f"{r['exactitude_ponderee']:9.1%} "
            f"{(f'{r["exactitude_arbitrees"]:.1%}' if r.get("exactitude_arbitrees") is not None else '—'):>10s} "
            f"{(f'{r["exactitude_forcees"]:.1%}' if r.get("exactitude_forcees") is not None else '—'):>8s} "
            f"{(f'{ec:.4f}' if ec is not None else '—'):>9s} "
            f"{r.get('support_commun_notes', 0):8d} "
            f"{par_mode.get('bike', {}).get('rappel', float('nan')):7.1%} "
            f"{par_mode.get('transit', {}).get('rappel', float('nan')):7.1%} "
            f"{par_mode.get('walk', {}).get('rappel', float('nan')):9.1%} "
            f"{r['compteurs'].get('mode_declare_absent_des_presentees', 0):13d} "
            f"{(f'{taux_zero:.1%}' if taux_zero is not None else '—'):>9s}"
        )
    print(f"\nClasses, in the order of the matrices: {', '.join(CLASSES)}")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--jeu", default=JEU_DEFAUT)
    parseur.add_argument("--verite", default=VERITE_DEFAUT)
    parseur.add_argument("--experience", action="append", help="repeatable; default: all")
    parseur.add_argument("--sortie", type=Path, help="JSON file of the detailed results")
    parseur.add_argument(
        "--definisseur",
        action="append",
        help="repeatable; short name of a decision-maker that defines the common support of "
        "the cross-entropy. Default: all those that return a non-degenerate distribution",
    )
    args = parseur.parse_args()

    verite = charger_verite(ICI / args.verite)
    offre = charger_offre(RACINE / "data" / "jeux" / args.jeu)

    dossier_experiences = RACINE / "data" / "experiences"
    if args.experience:
        noms = args.experience
    else:
        noms = experiences_du_jeu(dossier_experiences, args.jeu)
        logger.info(f"Experiments found on set {args.jeu}: {len(noms)}")

    resultats: dict[str, dict[str, Any]] = {}
    notables: dict[str, dict[str, tuple[int, list[float], float]]] = {}
    for nom in noms:
        execution = derniere_execution_de(nom, dossier_experiences)
        if execution is None or not (execution / "decisions.jsonl").exists():
            logger.warning(f"{nom}: no usable run, skipped")
            continue
        court = nom.replace("exp_", "").split("_jtir")[0]
        etat = etat_execution(execution)
        resultats[court], notables[court] = auditer(execution, verite, offre)
        r = resultats[court]
        if r:
            r["etat_execution"] = etat
            # An unfinished run covers only the START of a sample — the first
            # respondents in key order, not a draw. Its rates compare to nothing
            # and the row is marked as such, rather than ranked in the table.
            if etat != "terminee":
                logger.warning(
                    f"[ALARME] {court}: run {etat} ({r['notes']} trips scored out of "
                    f"{len(verite)}) — PARTIAL figures, not comparable to terminated arms"
                )
            logger.info(
                f"{court:12s} accuracy {r['exactitude_ponderee']:.1%} weighted "
                f"on {r['notes']} trips"
            )

    if not resultats:
        raise SystemExit("no result: run the experiments on this set first")

    support_commun(notables, resultats, args.definisseur)

    rendre(resultats)

    if args.sortie:
        args.sortie.write_text(
            json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.success(f"Résultats détaillés écrits : {args.sortie}")


if __name__ == "__main__":
    main()
