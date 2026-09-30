# Exécution compromise — NE PAS REPRENDRE, NE PAS SCORER

Deux exécutants ont tourné en parallèle sur cette exécution entre 12:02:12 et 12:13:14
(2026-09-22) : un lancement (`experience-lancer`, PID conteneur 258) et une reprise
(`experience-reprendre`, PID 294) lancés à une minute d'intervalle. La plateforme a bien
détecté la réouverture — « rouverte alors qu'elle était `en_cours` — arrêt forcé consigné » —
mais elle n'a fait que le CONSIGNER : le premier processus n'a pas été tué et a continué
d'écrire.

## Ce que contient `decisions.jsonl`

Au moment de l'arrêt : 945 lignes pour 504 décisions distinctes, soit 448 doublons.
Aucune ligne tronquée (les écritures sont restées atomiques), mais :

- 154 doublons dont les deux copies sont identiques — sans conséquence ;
- **294 doublons dont les copies DIFFÈRENT**, parfois totalement.

L'écart ne vient pas du modèle : à température nulle et à entrée identique, ce décideur est
exactement reproductible (mesuré le 2026-09-21 sur 176 décisions, 0 différence au second
passage). Il vient de l'ÉTAT DE CHAÎNE : l'un des exécutants partait de zéro, l'autre
reprenait à 56 décisions archivées, donc le véhicule était disponible pour l'un et pas pour
l'autre, et les options offertes différaient. Deux réponses justes à deux questions
différentes.

## Pourquoi elle est irrécupérable

Les enregistrements ne portent aucun identifiant d'exécutant. Pour ces 294 déplacements,
rien ne permet de savoir laquelle des deux décisions appartient à la chaîne cohérente avec
la suite du run. Un `scores.json` calculé là-dessus ne serait pas défendable.

Conservée comme trace de l'incident. La mesure de non-régression de `prompt_expert_25` est à
refaire dans une exécution neuve (un lancement SANS `--reprendre` en crée une : `cli.py:635`).
