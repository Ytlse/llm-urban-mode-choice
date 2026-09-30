"""lancer_experience_rf.py — Plays the random forest control as an experiment, once.

**Why a one-off launcher rather than wiring.** Making the control a full-fledged family
takes two lines: one entry in `FAMILLES` (`experiences/decideur_modele.py`) and
one in `POLICY_FORMATS` / `load_policy` (`scripts/synthesis/model_on_common_set.py`). These
two files are the ones **ticket 043** (kernel logistic regression) is modifying right
now, and it adds exactly one entry to the same tables. Writing there in parallel means
overwriting one of the two pieces of work.

This script works around the wait without faking anything: it **replaces `load_policy` in
the decision-maker's namespace**, at run time, then lets the platform do all the
rest — decision path, renormalisation over the OTP offer, draw, counters, scoring,
`execution.yaml`. The run produced is a real run, computed by the platform's code. Nothing
is copied by hand: a `scores.json` typed on the keyboard would no longer be comparable to
the logit's 6.1734, since one would come from the scoring code and the other from us.

⚠ **What we lose, and accept (decision of 2026-09-11).** The experiment produced is
**not replayable by the CLI**: `python -m experiences lancer --experience exp_rf_jtir_nosim`
will fail on an unknown artefact format until the two lines are in place. It will be
necessary to go through this script again. The fact is written in the produced
`experience.yaml`, so that nobody discovers it by trying.

Usage:
    python -m scripts.progedo_logit.lancer_experience_rf [--vers NOM] [--definir-seulement]

Offline on the model side (no LLM call, no quota); the itinerary offer comes from OTP
as for the other `sans_simulateur` experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE / "services" / "llm-agents") not in sys.path:
    sys.path.insert(0, str(RACINE / "services" / "llm-agents"))

from experiences.experience import trouver_dossier_experience

#: Experiments are stored by families (`regime_nominal/<jeu>/<exp>/`): a name resolves
#: through `trouver_dossier_experience`, never through `DOSSIER_EXPERIENCES / nom`.
DOSSIER_EXPERIENCES = RACINE / "data" / "experiences"

ARTEFACT = "scripts/progedo_logit/rf_mode_choice_policy.json"

#: `exp_mnl_jtir_nosim` designates the population by its path IN THE CONTAINER
#: (`/data/eqasim-output/…` ← bind on `data/population/`). This control must run on
#: the host: the `controller` container carries scikit-learn 1.9.0 while the forest was
#: estimated under 1.8.0, and the two do not give the same forest — measured, not assumed
#: (accuracy +0.000188, CEL +0.0008). Only this field changes; **the population's identity
#: fingerprint is the file's SHA**, not its path, so the comparison with the other families
#: stays exact.
POPULATION_CONTENEUR = "/data/eqasim-output/population_1000_PANEL"
POPULATION_HOTE = "data/population/population_1000_PANEL"
DERIVE_DE = "exp_mnl_jtir_nosim"
NOM_DEFAUT = "exp_rf_jtir_nosim"

#: Note placed NEXT TO the experience.yaml. This experiment has a constraint the three
#: other families do not have — it is launched from the host — and an experiment launched
#: differently from its neighbours must say so itself. Not *inside* the `experience.yaml`:
#: its schema is strict and refuses any key outside the contract, which is a good thing (an
#: experiment must not carry a free field nobody validates). A neighbouring file is
#: seen when opening the folder, and does not cheat the schema.
LISEZ_MOI = """# exp_rf_… — se lance depuis l'HÔTE, pas depuis le conteneur

Forêt aléatoire du [ticket 044](__DOCS__/tickets/ticket_044_temoin_random_forest.md),
quatrième méthode de référence du § 4.4.

## Comment la lancer

Depuis la racine du dépôt, **sur l'hôte** :

```bash
services/llm-agents/.venv/bin/python -m experiences lancer --experience <nom>
```

ou par le lanceur, qui définit l'expérience si elle n'existe pas encore :

```bash
services/llm-agents/.venv/bin/python scripts/progedo_logit/lancer_experience_rf.py --vers <nom>
```

`make experience-lancer` ne convient PAS : cette cible exécute dans le conteneur
`controller`, voir la dernière section.

## Ce qui a changé le 2026-09-16 (ticket 088 § 3.3)

La famille `rf` est déclarée dans les deux tables de la plateforme — `FAMILLES`
(`services/llm-agents/experiences/decideur_modele.py`) et `POLICY_FORMATS` / `load_policy`
(`scripts/synthesis/model_on_common_set.py`). Jusque-là, le lanceur les contournait en
inscrivant la famille en mémoire et en remplaçant `load_policy` dans l'espace de noms du
décideur, si bien que l'expérience n'était pas rejouable par la CLI. Elle l'est.

Ces deux lignes avaient été laissées de côté le 2026-09-11 parce que le **ticket 043**
(régression logistique à noyau) modifiait les mêmes tables au même moment. Il est clos.

## Ce qui n'a PAS été bricolé

L'exécution est une **vraie exécution** : chemin de décision, renormalisation sur l'offre OTP,
tirage, compteurs et scoring sont le code de la plateforme. Aucun `scores.json` n'a été écrit
à la main — un composite tapé au clavier ne serait pas comparable à celui du logit ou du
booster, puisque l'un sortirait du code de scoring et l'autre de nous.

Le décideur **réajuste la forêt** au chargement (~10 s, aucun arbre sérialisé : 1 200 arbres
et 3 740 500 nœuds pèseraient ~150 Mo), puis **vérifie qu'elle reproduit les métriques
publiées** avant de décider quoi que ce soit.

## Pourquoi elle tourne sur l'hôte et non dans le `controller`

Le conteneur porte **scikit-learn 1.9.1** (mesuré le 2026-09-16 ; il portait la 1.9.0 au
2026-09-11), la forêt a été estimée sous **1.8.0**, et deux versions ne donnent pas la même
forêt — mesuré et non supposé : exactitude +0,000188, CEL +0,0008. Le garde-fou de
reproduction, qui réajuste puis compare les métriques publiées à 1e-9 près, refuse donc de
démarrer dans le conteneur. C'est son travail.

Aligner la version dans l'image a été envisagé au ticket 088 § 3.3 et écarté le 2026-09-16 :
cela déplacerait le réajustement de l'hôte (macOS arm64) vers un conteneur Linux, où rien ne
garantit une forêt bit-à-bit identique même à version égale, et ferait donc courir aux
chiffres publiés du § 4.4 un risque que la parité d'exécution ne justifie pas. Le chapitre 4
affirme une parité d'**estimation** — mêmes microdonnées, même découpage, mêmes poids, même
encodage, mêmes métriques — et elle est vraie quel que soit l'endroit où la forêt s'exécute.

Conséquence sur la définition : `population.chemin` désigne le chemin **hôte**
(`data/population/…`) là où `exp_mnl_jtir_nosim` désigne le chemin conteneur
(`/data/eqasim-output/…`). C'est le même fichier des deux côtés du bind, et l'empreinte
d'identité de la population est **son SHA**, pas son chemin : la comparaison avec les autres
familles reste exacte.

(Le conteneur n'a de toute façon ni `pyarrow` ni `fastparquet` ; c'est pourquoi le rejeu passe
par une matrice `.npz` de 1,4 Mo que numpy seul sait relire, et non par le parquet.)
"""


def verifier_famille_rf() -> None:
    """Is the `rf` family properly declared in the platform?

    Until ticket 088 § 3.3, this function REGISTERED the family for the lifetime of the
    process and replaced `load_policy` in the decision-maker's namespace. Both tables now
    carry it (`decideur_modele.FAMILLES`, `model_on_common_set.POLICY_FORMATS`): there is
    nothing left to inject, only to check that this script and the platform agree.
    A check rather than a silence — if someone removes the line, the failure must be readable
    here and not three minutes later on « format d'artefact non reconnu ».
    """
    from experiences import decideur_modele
    from scripts.progedo_logit.mode_choice_rf import RF_FORMAT
    from scripts.synthesis.model_on_common_set import POLICY_FORMATS

    manquantes = []
    if decideur_modele.FAMILLES.get(RF_FORMAT) != "rf":
        manquantes.append("experiences/decideur_modele.py : FAMILLES")
    if RF_FORMAT not in POLICY_FORMATS:
        manquantes.append("scripts/synthesis/model_on_common_set.py : POLICY_FORMATS")
    if manquantes:
        raise SystemExit(
            f"The « rf » family ({RF_FORMAT}) is not declared in: "
            + " ; ".join(manquantes)
            + ". See ticket 088 § 3.3 — it entered there on 2026-09-16.")
    print(f"[lanceur] « rf » family declared in the platform ({RF_FORMAT})")


def _chemin_population_hote(chemin_yaml: Path) -> None:
    """Brings the population path back from the container to the host, if needed.

    A path substitution, nothing more: the `population.json` file is the same on both
    sides of the bind, and **its SHA** serves as identity fingerprint. Done
    explicitly rather than suffered — the alternative was discovering « population
    introuvable » at launch.
    """
    contenu = chemin_yaml.read_text(encoding="utf-8")
    if POPULATION_CONTENEUR not in contenu:
        return
    if Path(POPULATION_CONTENEUR).exists():
        return                      # running in the container: nothing to change
    chemin_yaml.write_text(
        contenu.replace(POPULATION_CONTENEUR, str(RACINE / POPULATION_HOTE)),
        encoding="utf-8")
    print(f"[lanceur] population path brought back to the host ({POPULATION_HOTE}) — "
          "same file, same identity SHA")


def _ecrire_lisez_moi(dossier: Path) -> None:
    """The LISEZ-MOI, its relative links computed from the folder where it is written.

    Hard-coded for `data/experiences/<exp>/`, they were dead inside a family
    (`regime_nominal/<jeu>/<exp>/`), two levels lower."""
    docs = Path(os.path.relpath(RACINE / "docs", dossier)).as_posix()
    (dossier / "LISEZ-MOI.md").write_text(LISEZ_MOI.replace("__DOCS__", docs), encoding="utf-8")


def definir(vers: str) -> int:
    """Creates the experiment by duplicating the logit's — a single parameter changes.

    The experiment is looked up in its family before AND after duplication. Before: a
    stored experiment we did not see would be re-duplicated under the same name, and the copy
    would rewrite its `experience.yaml` with `executions_connues: []`. After: the CLI stores
    the copy in its set's family, not at the root.
    """
    from experiences import cli

    dossier = trouver_dossier_experience(vers, racine=DOSSIER_EXPERIENCES)
    if dossier is not None:
        print(f"[lanceur] {vers} already exists ({dossier.relative_to(DOSSIER_EXPERIENCES)}) — "
              "definition unchanged")
        _chemin_population_hote(dossier / "experience.yaml")
        _ecrire_lisez_moi(dossier)
        return 0
    code = cli.main(["dupliquer", "--de", DERIVE_DE, "--vers", vers,
                     "--artefact", ARTEFACT])
    if code != 0:
        return code
    dossier = trouver_dossier_experience(vers, racine=DOSSIER_EXPERIENCES)
    if dossier is None:
        print(f"[lanceur] ERREUR : {vers} dupliquée depuis {DERIVE_DE} (code 0) mais "
              f"introuvable sous {DOSSIER_EXPERIENCES}, ni à plat ni dans une famille — "
              "rien n'est lancé", file=sys.stderr)
        return 1
    _chemin_population_hote(dossier / "experience.yaml")
    _ecrire_lisez_moi(dossier)
    print(f"[lanceur] {vers} defined from {DERIVE_DE}, artefact {ARTEFACT}")
    print(f"[lanceur] non-replayable warning written: {dossier / 'LISEZ-MOI.md'}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vers", default=NOM_DEFAUT, help=f"experiment name (default: {NOM_DEFAUT})")
    parser.add_argument("--definir-seulement", action="store_true",
                        help="creates the experience.yaml without launching the run")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(line_buffering=True)

    chemin_artefact = RACINE / ARTEFACT
    if not chemin_artefact.exists():
        raise SystemExit(
            f"Control artefact missing: {ARTEFACT}. Produce it with "
            "`make forest FOREST_ARGS=--artefact`.")

    verifier_famille_rf()
    code = definir(args.vers)
    if code != 0 or args.definir_seulement:
        return code

    from experiences import cli
    print(f"[lanceur] launching {args.vers} — the decision-maker refits the forest (~10 s), "
          "then checks that it reproduces the published metrics")
    return cli.main(["lancer", "--experience", args.vers, "--ne-pas-attendre-fenetre"])


if __name__ == "__main__":
    raise SystemExit(main())
