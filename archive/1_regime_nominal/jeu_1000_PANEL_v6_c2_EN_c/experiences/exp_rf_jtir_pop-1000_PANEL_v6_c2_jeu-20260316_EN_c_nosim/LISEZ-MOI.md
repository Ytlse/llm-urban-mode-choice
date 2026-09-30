# exp_rf_… — se lance depuis l'HÔTE, pas depuis le conteneur

Forêt aléatoire du [ticket 044](../../../docs/tickets/ticket_044_temoin_random_forest.md),
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
