# exp_rf_jtir_nosim — NON REJOUABLE par la CLI

Témoin **random forest** du [ticket 044](../../../docs/tickets/ticket_044_temoin_random_forest.md),
lancé par `scripts/progedo_logit/lancer_experience_rf.py`.

## Pourquoi ce fichier existe

Ce script enregistre la famille `rf` **au moment de l'exécution**, sans l'écrire dans les
tables de la plateforme. Conséquence directe :

```
python -m experiences lancer --experience exp_rf_jtir_nosim     # ÉCHOUE
```

L'exécution échoue sur un format d'artefact inconnu (`rf_mode_choice_policy`). **Pour
relancer, repasser par le script :**

```bash
python -m scripts.progedo_logit.lancer_experience_rf
```

## Ce qu'il faudrait pour la rendre rejouable

Deux lignes, pas davantage :

- `rf_mode_choice_policy: "rf"` dans `FAMILLES` (`services/llm-agents/experiences/decideur_modele.py`)
- `rf_mode_choice_policy` dans `POLICY_FORMATS` et son branchement dans `load_policy`
  (`scripts/synthesis/model_on_common_set.py`)

Elles ont été laissées de côté le 2026-09-11 parce que le **ticket 043** (régression
logistique à noyau) modifiait ces deux fichiers au même moment, et y ajoutait exactement une
entrée aux mêmes tables. Écrire en parallèle aurait écrasé l'un des deux travaux.

## Ce qui n'a PAS été bricolé

L'exécution est une **vraie exécution** : chemin de décision, renormalisation sur l'offre OTP,
tirage, compteurs et scoring sont le code de la plateforme. Aucun `scores.json` n'a été écrit
à la main — un composite tapé au clavier ne serait pas comparable à celui du logit ou du
booster, puisque l'un sortirait du code de scoring et l'autre de nous.

Le décideur **réajuste la forêt** au chargement (~10 s, aucun arbre sérialisé : 1 200 arbres
et 3 740 500 nœuds pèseraient ~150 Mo), puis **vérifie qu'elle reproduit les métriques
publiées** avant de décider quoi que ce soit.

## Pourquoi elle tourne sur l'hôte et non dans le `controller`

Le conteneur porte **scikit-learn 1.9.0**, la forêt a été estimée sous **1.8.0**, et les deux
ne donnent pas la même forêt — mesuré et non supposé : exactitude +0,000188, CEL +0,0008. Le
garde-fou de reproduction refuse donc de démarrer dans le conteneur, ce qui est son travail.

Conséquence sur la définition : `population.chemin` désigne le chemin **hôte**
(`data/population/…`) là où `exp_mnl_jtir_nosim` désigne le chemin conteneur
(`/data/eqasim-output/…`). C'est le même fichier des deux côtés du bind, et l'empreinte
d'identité de la population est **son SHA**, pas son chemin : la comparaison avec les autres
familles reste exacte.

(Le conteneur n'a de toute façon ni `pyarrow` ni `fastparquet` ; c'est pourquoi le rejeu passe
par une matrice `.npz` de 1,4 Mo que numpy seul sait relire, et non par le parquet.)
