"""Builds the score synthesis page.

    python -m scripts.synthesis.build [--config …] [--run …] [--out …]

Nothing fails on missing data: each missing source becomes a
« Données manquantes » card along with the action that would produce it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import frames, heldout_eval, render
from .frames import DIMENSIONS, MODES
from .sources import REPO_ROOT, import_formule_score, load_manifest, probe

# Actions to fill the page. Shown as is at the bottom of the page: a
# single source of truth between the doc and the rendering. Identifiers are
# never recycled — a completed action stays in the list, marked `done`, because
# code warnings and tickets refer to it by number.
ACTIONS = [
    {"id": "A1", "title": "Figer le run servant de jeu commun",
     "detail": "Remplacer experiments/current par un chemin d'archive explicite dans "
               "sources.yaml, pour que la page reste reproductible quand le symlink bouge.",
     "cost": "5 min", "unlocks": "Reproductibilité de toute la page",
     "done": "Le manifeste épingle experiments/archive/2026-07-31_15_45 (il a porté "
             "experiments/archive/2026-07-29_18_34 jusqu'au 2026-07-31). La page ne "
             "suit plus le symlink : elle décrit le même run à chaque régénération, "
             "et l'empreinte de moves.csv le vérifie. Changer de run est désormais un "
             "geste sûr de bout en bout : le cache d'évals est indexé sur l'empreinte "
             "de l'échantillon, et la page écarte une mesure du volet 2 faite sur un "
             "autre run que celui qu'elle épingle."},
    {"id": "A2", "title": "Renseigner le type de logement dans moves.csv",
     "detail": "La colonne est écrite vide (move_logger.py) alors que la référence EMC² "
               "porte une ventilation par type de logement. Le trait n'existe pas non "
               "plus dans traits_json : il faut le produire à la génération de population.",
     "cost": "1/2 j", "unlocks": "Dimension type de logement, volet 1",
     "done":
             "Le trait est produit et journalisé. Aucune source de la chaîne ne le "
             "portait — ni eqasim, ni les tables INSEE de l'étape 3bis du notebook : "
             "vérifié, ce n'est pas un branchement mais une création. La seule source "
             "qui le porte pour Toulouse est l'enquête elle-même (variable M1 du "
             "fichier ménages, « Type d'habitat », dont les cinq modalités sont "
             "exactement celles de la ventilation publiée). Le trait est donc IMPUTÉ, "
             "et la page doit le dire : make housing-type exporte la loi du type de "
             "logement par zone fine (pondérée par les coefficients de redressement "
             "des PERSONNES — 41,7 % des personnes en individuel isolé contre 34,7 % "
             "des ménages, les foyers en individuel étant plus grands), lissée zone → "
             "secteur de tirage → périmètre parce qu'une zone fine ne compte que 18 "
             "personnes enquêtées en médiane. L'imputation est conditionnée à la zone "
             "fine du domicile via le résolveur de l'action A7 : un tirage indépendant "
             "de la géographie aurait mis des tours en périphérie rurale et faussé "
             "l'axe même qu'on cherche à mesurer. Elle est déterministe (hachage "
             "SHA-256 de l'adresse, pas d'un RNG) et porte sur l'ADRESSE, pour que deux "
             "personas d'un même foyer — 930 personas pour 498 domiciles sur le run — "
             "ne se retrouvent pas l'un en maison et l'autre en tour. Hors couche, rien "
             "n'est deviné : 4,4 % des personas n'ont pas de trait, et la colonne reste "
             "vide. La distribution obtenue est vérifiable : sur la population de "
             "10 000, elle s'écarte de 2,9 points L1 cumulés de la loi de l'enquête, "
             "l'écart résiduel diminuant avec la taille comme du bruit de tirage. Côté "
             "journal, move_logger.py écrit le libellé porté par le persona et vide "
             "quand il n'y en a pas ; les modalités sont déclarées EN UN SEUL POINT "
             "(packages/mobility_core/src/mobility_core/housing_type.py), partagé par la génération, le journal "
             "et la page. L'AXE EST PEUPLÉ depuis que la page épingle le run du "
             "2026-07-31 : 302 individuel isolé, 219 petit collectif, 211 grand "
             "collectif, 143 individuel accolé. Il a fallu attendre un run, et c'était "
             "inévitable — le changement ne touche que les runs FUTURS, le moves.csv "
             "d'un run déjà écrit ne se corrige pas. L'axe n'a pas été reconstruit à la "
             "volée depuis population_1000.json, et ce n'était pas possible sans "
             "tricher : la population de l'ancien run ne portait pas le trait, le "
             "recalculer à la génération de la page supposerait de rejouer "
             "l'imputation, donc d'exiger deux ressources d'accès restreint (couche de "
             "zones, table du type de logement) au moment de bâtir la page — la page "
             "cesserait d'être reproductible sur un poste sans les données PROGEDO, ce "
             "que l'action A1 a précisément acquis."},
    {"id": "A3", "title": "Ré-évaluer graine et meilleur prompt sur le jeu commun",
     "detail": "Rejouer deux prompts sur un échantillon du run (viser ~400 décisions) "
               "avec le modèle d'évaluation épinglé, et écrire le résultat au format "
               "décisions attendu par la page.",
     "cost": "175 appels LLM", "unlocks": "Volet 2 dans la comparaison finale",
     "progress": {
         "acquis":
             "scripts/synthesis/common_set_eval.py (make common-set-eval) a rejoué la "
             "graine 4c2ea894 et la feuille 0fc427e7 sur 509 décisions du run alors "
             "épinglé (2026-07-29_18_34), "
             "sous le régime épinglé, avec 100 % de couverture (80/80 personnes). "
             "L'échantillon est gelé : tirage PAR PERSONNE sur "
             "sha256(\"common_set_v1:\" + agent_id) % 1000 < 99 — même famille de règle "
             "que les jeux gelés du moteur, mais dans un espace de hachage DISTINCT, "
             "sans quoi l'échantillon aurait été un préfixe de l'intervalle train et "
             "n'aurait contenu que des personas ayant servi à optimiser la lignée. Le "
             "seuil 99 n'est pas rond : c'est le plus petit dont le rapport de "
             "couverture du moteur est propre (à 424 décisions, la tranche 70-74 est "
             "vide et la dimension « âge » ne porterait plus sur le même support que le "
             "volet 1). Le lotissement et le rattrapage sont ceux de l'action A10, non "
             "réécrits : bien leur en a pris, 29 lots sur 128 sont revenus amputés de "
             "personas (jusqu'à 2 rendus sur 8) et ont tous été re-tirés par moitiés — "
             "un découpage maison aurait scoré sur une sous-population sans le dire. "
             "RÉSULTAT : le gain de la lignée se transporte presque à l'identique "
             "(+2,13 points sur le jeu commun contre +2,12 sur les personas gelés), "
             "mais le NIVEAU ne se transporte pas du tout — 38,53 et 36,41 sur le jeu "
             "commun contre 24,35 et 22,24 sur les personas gelés, soit +14,2 points "
             "pour les deux prompts. Une part de ce décalage est un artefact "
             "d'effectif, et elle est désormais chiffrée plutôt que supposée : la "
             "colonne « Sim. (éch. V2) » restreint le volet 1 aux 81 mêmes personnes et "
             "montre que la seule réduction d'effectif coûte +5,02 points (24,37 → "
             "29,39), les divergences par strate étant biaisées vers le haut à petits "
             "effectifs. C'est donc à 29,39 que les colonnes de calibration se "
             "comparent, pas à 24,37 — et le volet 2 reste au-dessus, donc moins fidèle "
             "à l'enquête que la simulation sur son propre substrat. Quota : 175 appels "
             "sur la seconde clé Google (la première était encore épuisée, le seau free "
             "tier se réinitialisant à minuit PACIFIC et non UTC — une sonde de 4 "
             "appels a réussi avant que le compteur ne rattrape, l'application du RPD "
             "n'étant pas exacte à la frontière). La page regroupant les régimes par "
             "modèle · politique et non par clé, le libellé produit reste le régime "
             "épinglé.",
         "reste":
             "La refaire sur le run épinglé depuis le 2026-07-31 (2026-07-31_15_45). La "
             "mesure ci-dessus reste juste, mais elle porte sur un AUTRE substrat : la "
             "page l'écarte donc du volet 2 plutôt que de la faire voisiner avec des "
             "volets 1 et 3 calculés sur le nouveau run. L'échantillon gelé y vaut 383 "
             "décisions pour les mêmes 80 personnes — la règle est inchangée, c'est le "
             "run qui porte moins de décisions LLM (beaucoup de trajets n'ont plus "
             "qu'un itinéraire depuis la cohérence de chaîne des véhicules). Coût "
             "chiffré : 96 appels avant re-tirs, ≈ 111 avec. Bloqué le 2026-07-31 par "
             "le quota — les DEUX clés Google épuisées (RPD 500 chacune) ; reprise "
             "autorisée le 2026-08-01 à 09:00 CEST. Ce report a mis au jour un défaut "
             "corrigé au passage : le cache d'évals du store était indexé sur le seul "
             "nom de jeu, sans le run, si bien qu'un changement de run resservait la "
             "mesure précédente en la réétiquetant — zéro appel, composites inchangés "
             "au centième. La clé porte désormais l'empreinte des records soumis."}},
    {"id": "A4", "title": "Évaluer sur le jeu de test gelé",
     "detail": "Aucun nœud du store n'a d'évaluation sur le split test : seuls train et "
               "screen sont peuplés. Le chiffre publiable de la calibration n'existe pas.",
     "cost": "~2 h de quota", "unlocks": "Score de généralisation, volet 2",
     "done": "Le constat était exact et il a été revérifié avant de payer quoi que ce "
             "soit : zéro éval sur « test » dans les deux stores, et les 3 évals « val » "
             "n'existaient que sous mistral-small — donc inutilisables sous le régime "
             "épinglé. scripts/synthesis/heldout_eval.py (make heldout-eval) a mesuré la "
             "lignée ENTIÈRE — 6 nœuds sur 6, pas seulement ses extrémités — sur les 106 "
             "décisions du jeu test, sous gemini-3.1-flash-lite-preview · masse de "
             "probabilité, en déléguant le lotissement et le rattrapage à l'évaluateur du "
             "moteur (défenses de l'action A10) : 7 lots amputés de personas sur 84, tous "
             "re-tirés par moitiés. 98 appels sur la seconde clé Google. "
             "NATURE DE LA GÉNÉRALISATION, établie sur les fichiers et non sur la règle "
             "déclarée : le découpage est PAR PERSONNE — les 66 personnes du test "
             "n'apparaissent dans aucun des 298 personas du train (intersection vide, "
             "vérifiée ; val de même ; screen au contraire entièrement inclus dans le "
             "train, ce qui lui interdit ce rôle). Ce sont donc des individus jamais vus, "
             "pas d'autres trajets des mêmes individus. "
             "RÉSULTAT. Lu brut, l'écart ressemble à du surapprentissage : la graine passe "
             "de 24,35 à 31,60 et la feuille de 22,24 à 24,06. Il n'en est rien, et le "
             "témoin le montre sans un seul appel LLM — rééchantillonner les décisions "
             "train DÉJÀ STOCKÉES à 66 personnes (200 tirages appariés, par personne, "
             "graine fixée) donne 29,84 pour la graine et 26,90 pour la feuille. La seule "
             "réduction d'effectif coûte donc +5,49 et +4,66 points, du même ordre que les "
             "+5,02 mesurés par l'action A3 sur la simulation. À effectif neutralisé la "
             "feuille est MEILLEURE sur le test que sur le train (-2,84), et les six nœuds "
             "tombent dans la bande du témoin : aucun surapprentissage détectable. "
             "Le gain de la lignée survit : +2,12 sur le train, +7,54 sur le test. "
             "L'amplification, elle, n'est PAS démontrée et la page ne la revendique pas — "
             "le témoin apparié du gain vaut +2,94 sur une bande de -1,84 à +8,24, et 7,54 "
             "y tombe. 66 personnes ne permettent pas de trancher plus finement. "
             "UNE CONFUSION RÉSIDUELLE EST PUBLIÉE PLUTÔT QUE TUE : le moteur retire la "
             "section « Historique » (mémoire STM/LTM, non reproductible) des jeux val et "
             "test et la garde dans le train, où elle couvre 86 % des records. Le prompt de "
             "test n'est donc pas seulement adressé à d'autres personnes, il est aussi plus "
             "court d'une section — les deux effets sont mêlés et rien dans les données ne "
             "les sépare. Ces évals ne rejoignent ni la trajectoire, ni la lignée, ni la "
             "matrice de synthèse : le jeu de retenue est un troisième substrat, et l'y "
             "coller rejouerait la confusion que l'action A3 a corrigée."},
    {"id": "A5", "title": "Rejouer une lignée sous un modèle d'évaluation unique",
     "detail": "L'historique mélange mistral-small, gemini-3.1-flash-lite et des imports "
               "hérités. Les niveaux de score ne sont pas comparables entre eux.",
     "cost": "variable", "unlocks": "Trajectoire lisible bout à bout, volet 2",
     "done": "La page distingue les régimes de mesure (modèle ET politique de décision, la "
             "seconde changeant les décisions elles-mêmes) au lieu du seul modèle, et "
             "affiche la lignée épinglée dans sources.yaml : 6 nœuds, de la graine à "
             "0fc427e7. La chaîne est reconstruite par les arêtes de mutation, sans quoi "
             "elle perdait sa graine — le deuxième nœud, dédoublonné depuis la branche "
             "main, a un parent vide. Le rejeu est produit : calibrate reeval a mesuré les "
             "6 nœuds sur le jeu train sous le régime ÉPINGLÉ "
             "(gemini-3.1-flash-lite-preview · masse de probabilité), une fois l'éval "
             "débloquée par l'action A10 ; le repli sur mistral-small-latest · mode élu a "
             "disparu. La lignée se lit maintenant sous DEUX régimes en regard, et ils "
             "s'accordent : la calibration gagne 7,60 points sous l'ancien instrument "
             "(24,9 % du niveau de la graine) et 2,12 sous celui de la production (8,7 %) "
             "— près de trois fois moins en part, mais dans le même sens. Le gain n'est "
             "donc pas un artefact de l'instrument qui a guidé l'optimisation ; son "
             "ampleur, elle, ne se transporte pas d'un régime à l'autre."},
    {"id": "A6", "title": "Entraîner la politique LightGBM PROGEDO",
     "detail": "Le parquet et feature_spec.json existent ; il manque le notebook "
               "d'entraînement et le modèle sérialisé mode_choice_policy.json.",
     "cost": "1 j", "unlocks": "Volet 3",
     "done": "scripts/progedo_logit/fit_mode_choice_policy.py (make policy) entraîne un "
             "booster LightGBM multiclasse sur les 21 variables du spec — pondéré par "
             "les coefficients de redressement de l'enquête, split train/test lu dans le "
             "parquet donc étanche au ménage, arrêt anticipé sur une validation "
             "redécoupée dans le train et jamais sur le test, graine fixée et résultat "
             "reproductible à l'octet. Les trois variables marquées diagnostic_only "
             "(distance_km, crow_km, duration_min) sont refusées par un contrôle "
             "explicite : ce sont elles qui donnaient une PR-AUC marche de 0,985, "
             "c'est-à-dire une fuite. Sur le split test, pondéré : log-loss 0,5363, "
             "accuracy 79,5 %, et 2,1 points d'écart cumulé sur les parts modales en "
             "masse de probabilité (8,7 points en mode élu). L'artefact "
             "mode_choice_policy.json est autoportant — ordre des variables, encodage "
             "des modalités, ordre des classes, version du contrat, métriques, et le "
             "booster sous deux formes (dump_model pour l'évaluateur pur Python, texte "
             "natif pour un rechargement exact) : un consommateur prédit sans relire le "
             "parquet. Le modèle est depuis appliqué au jeu commun (A8), et les écarts "
             "mesurés ici sur le split test se retrouvent sur le run : la masse de "
             "probabilité reste nettement mieux calibrée que le mode élu."},
    {"id": "A7", "title": "Construire le résolveur de zone fine",
     "detail": "Les variables géographiques du modèle (od_km, densités, distances au "
               "centre) exigent un point → zone fine. La couche ZF existe dans les "
               "données PROGEDO mais n'est pas exploitable à l'exécution.",
     "cost": "1 j", "unlocks": "Variables géo du volet 3",
     "done": "packages/mobility_core/src/mobility_core/zone_resolver.py rattache un point à sa zone par jointure "
             "point-dans-polygone, et en dérive les six variables géo à la formule de "
             "l'entraînement — distance entre centroïdes, imputation intra-zone — et non "
             "à vol d'oiseau, qui donnait un facteur 2 sur les trajets intra-zone. La "
             "couche servie est exportée en réutilisant le build_geo() du jeu "
             "d'entraînement (make zones), donc identique par construction ; le "
             "résolveur refuse de démarrer si elle et feature_spec.json ne décrivent pas "
             "le même hypercentre. Couverture mesurée sur la population de référence : "
             "95,1 % des paires origine-destination, 95,5 % des localisations ; hors "
             "couche, il renvoie « pas de zone » au lieu de deviner. Le run épinglé, "
             "lui, est intégralement couvert : sa population est un autre tirage que "
             "celle sur laquelle les 95 % ont été mesurés (toulouse_population_1000.json, "
             "4,5 % hors couche), et aucune de ses 11 890 localisations n'échappe au "
             "périmètre d'enquête. Le repli reste posé et testé — il n'a simplement pas "
             "eu à servir sur ce run."},
    {"id": "A8", "title": "Prédire sur le jeu commun et renormaliser sur l'offre OTP",
     "detail": "Appliquer le modèle aux personas du run, en renormalisant sur les modes "
               "réellement proposés par OTP, puis écrire les probabilités attendues.",
     "cost": "1/2 j", "unlocks": "Volet 3 dans la comparaison finale",
     "done": "scripts/synthesis/model_on_common_set.py (make common-set-predict) applique "
             "la politique aux 5 945 décisions du run épinglé — le périmètre du volet 1 "
             "lui-même, construit par le même frames.read_moves et les mêmes exclusions, "
             "sans quoi les colonnes ne seraient pas comparables. Aucun appel LLM, aucun "
             "réseau, résultat déterministe. La correspondance des modes est établie en "
             "UN point et testée : train tombe dans les transports collectifs des deux "
             "côtés, mais le deux-roues motorisé diverge (voiture pour la politique, "
             "« autres » pour la page) — il est donc retiré de l'offre plutôt que compté "
             "comme une offre de voiture, faute de quoi le seul volet 3 verrait sa part "
             "voiture gonflée. La renormalisation sur l'offre OTP n'est pas cosmétique : "
             "96,6 % de la masse prédite tombe en moyenne sur des modes réellement "
             "proposés (médiane 99,8 %, minimum 0,8 %), la correction déplace le mode le "
             "plus probable sur 142 décisions, et rapproche les parts modales de la "
             "référence de 17,9 à 14,1 points d'écart cumulé. Aucune décision n'a été "
             "écartée : les trois causes prévues (zone inconnue, offre sans mode "
             "prédictible, persona introuvable) sont codées et testées, mais la "
             "population de ce run tombe entièrement dans la couche de zones. Deux "
             "lectures sont publiées, comme pour le volet 1 — masse de probabilité "
             "(composite 4,66) et mode élu (5,98) — parce que l'écart entre les deux est "
             "structurel : le modèle n'élit presque jamais le vélo alors qu'il le calibre "
             "bien. Le volet 3 écrase les deux autres (24,37 pour la simulation, 22,92 "
             "pour le meilleur prompt) et c'est attendu : entraîné sur l'enquête qui sert "
             "de cible, il borne ce qu'un modèle statistique atteint. La page le dit au "
             "lecteur au-dessus de la matrice, pas trois sections plus loin. Une surprise "
             "reste ouverte : 919 décisions (15,5 %) n'ont pas de "
             "socioprofessional_class, la population synthétique portant « Retired », "
             "modalité que le recodage de l'enquête ne produit jamais. Elle est rendue "
             "manquante par le contrat du spec plutôt que remappée à l'aveugle — "
             "main_occupation = « Retraité » porte la même information."},
    {"id": "A9", "title": "Unifier l'hypercentre",
     "detail": "43.597347/1.444997 dans feature_spec.json contre 43.6047/1.4442 dans "
               "move_logger.py : 820 m d'écart, qui déplacent les couronnes de résidence.",
     "cost": "15 min", "unlocks": "Cohérence lieu de résidence entre volets 1 et 3",
     "done": "move_logger.py ne déclare plus de centre : il lit celui de "
             "feature_spec.json via packages/mobility_core/src/mobility_core/geo_reference.py, unique point de "
             "lecture du bloc geo_reference — le même que celui sur lequel le résolveur "
             "de zone fine (A7) refuse de démarrer en cas de divergence. Le spec étant "
             "produit depuis des données d'accès restreint, son absence est prévue : le "
             "repli est la valeur publiée recopiée en constante, et un test échoue si "
             "les deux se mettent à diverger. Les couronnes de résidence des futurs runs "
             "sont donc mesurées depuis 43.597347/1.444997, comme les dist_center_* du "
             "modèle. C'EST LE CAS DEPUIS LE RUN DU 2026-07-31, et l'effet est "
             "vérifié plutôt que supposé : à population identique (mêmes 901 "
             "personnes), 30 d'entre elles changent de couronne par rapport au run "
             "précédent, dans les DEUX sens — signature d'un déplacement latéral du "
             "centre, et non d'un seuil qu'on aurait déplacé."},
    {"id": "A10", "title": "Débloquer l'éval sous la politique pondérée, puis porter la "
                           "lignée sur le modèle épinglé",
     "detail": "La lignée se lit bout à bout (A5), mais sous mistral-small-latest et la "
               "politique « mode élu » — pas sous le modèle qu'utilise la campagne. "
               "calibrate reeval fait la mesure, mais aucune éval n'aboutit : les lots "
               "dépassent le timeout de 240 s de l'adaptateur Google et sont retentés 5 "
               "fois, sans qu'aucune erreur ne remonte. Réduire le lot de 15 à 8 n'a pas "
               "suffi. À instrumenter avant de corriger : tokens de complétion, "
               "finishReason, nombre de décisions rendues. Bloque aussi la boucle — "
               "aucune campagne n'a encore tourné sous cette politique.",
     "cost": "diagnostic + ~372 appels", "unlocks": "Trajectoire sous le modèle de "
                                                    "production, et reprise de la campagne",
     "done": "Le diagnostic a écarté la cause supposée. Instrumenté sur des lots réels : "
             "3,6 à 8,8 s par appel pour un timeout de 240 s, finishReason=STOP partout, "
             "2 742 tokens de complétion au pire pour un plafond de 4 096 — ni lenteur, ni "
             "troncature. Le défaut est que le modèle rend un JSON valide et conforme mais "
             "AMPUTÉ de personas : 4 lots sur 12 à 15 personas n'ont rendu que 5 à 8 "
             "décisions sur 15, soit 18 % de la population perdue. Aucune défense ne "
             "pouvait le voir — ni erreur HTTP, ni troncature, ni schéma invalide : le lot "
             "passait pour un succès et l'éval était mise en cache sur une sous-population. "
             "Trois défenses posées : comparaison des personas envoyés aux décisions rendues "
             "à chaque requête ; re-tir du lot incomplet par moitiés (redemander à "
             "l'identique en décodage déterministe redonne la même réponse — il faut réduire "
             "la demande) ; refus de mettre en cache une éval sous le plancher de couverture, "
             "la base ne gardant pas le nombre de personas vus. L'échec silencieux proprement "
             "dit est refermé : la boucle de retry rendait une liste vide en s'épuisant, elle "
             "lève désormais avec une [ALARME]. Réduire le lot atténue sans régler — à 8 "
             "personas, 40 lots sur 372 sont encore revenus incomplets, jusqu'à 1 persona "
             "rendu sur 8, tous rattrapés. La lignée est mesurée : 6 nœuds sur 6 sous "
             "gemini-3.1-flash-lite-preview · masse de probabilité, 432 appels (+16 % de "
             "re-tirs), sur la seconde clé Google — le quota journalier de la première étant "
             "épuisé, et la page regroupant les régimes par modèle · politique, pas par clé. "
             "La campagne peut reprendre sous cette politique, ce qu'aucune n'avait fait."},
]

SCORE_DIMS = [
    ("global", "toutes", "JSD inter-modes (×100)", "global",
     "Part modale de la population entière."),
    ("absent_penalty", "toutes", "5 × part cible du mode oublié", "absent_penalty",
     "Sanctionne un mode auquel plus personne n'accorde la moindre chance."),
    ("age", "ordinale", "EMD le long de l'axe des 15 tranches", "age",
     "Déplacer une préférence vers une tranche voisine coûte moins cher que vers une tranche lointaine."),
    ("occupation", "nominale", "JSD pondérée par effectif", "occupation",
     "7 modalités, de scolaire à retraité."),
    ("genre", "nominale", "JSD pondérée par effectif", "genre", "Homme / femme."),
    ("motif", "nominale", "JSD pondérée par effectif", "motif",
     "Travail, études, achats. Accompagnement n'est jamais produit par la simulation."),
    ("distance", "ordinale", "EMD le long des 7 tranches", "distance",
     "De moins d'1 km à plus de 50 km."),
    ("length_penalty", "prompt", "neutralisée ici", "length_penalty",
     "Sans objet hors du volet calibration : poids ramené à 0."),
]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y à %H:%M")


def build_score_def(manifest, weights: dict) -> dict:
    dims = [{"dim": key, "kind": kind, "metric": metric, "weight": weights.get(wkey, 0.0),
             "note": note} for key, kind, metric, wkey, note in SCORE_DIMS]
    return {
        "dimensions": dims,
        "primary": manifest.get("score.metric", "emd_jsd"),
        "secondary": manifest.get("score.secondary", "l1_composite"),
        "engine": "scripts/synthesis/formule_score/metrics.py",
        "cerema_note": (
            "La référence couvre huit axes : global, âge, occupation, genre, motif, "
            "distance, lieu de résidence et type de logement. Les six premiers entrent "
            "dans le composite ; les deux derniers sont affichés hors score. Le type de "
            "logement est produit depuis l'action A2, mais il est IMPUTÉ — aucune "
            "source de la chaîne de génération ne le porte, il est tiré dans la loi que "
            "l'enquête observe dans la zone fine du domicile — et le run épinglé, "
            "antérieur, ne le porte pas encore. Les modes « autres » et « deux-roues "
            "motorisé » sont exclus et les quatre modes restants renormalisés à 100 %."),
    }


def frozen_sets_lineage(manifest, run: dict) -> dict:
    """Do the calibration frozen sets still carry the population in service?

    The page claims that the three strands share a substrate. That is true of the run — the
    guards check it — but strand 2 is *also* scored on frozen sets, and those
    were cut from a past run. Their manifest records the fingerprint of the
    original population; we compare it with that of the pinned run.

    Why this is needed since 2026-08-27: tickets 016 and 017 rewrite
    `has_pt_subscription` and `has_driving_license` in the population in service. The frozen
    sets keep the values copied from the ENTD 2008 donor. A strand 2 score
    thus describes a population the simulation no longer plays — and nothing said so.

    Divergence **declared, not corrected**: rebuilding the frozen sets would break the
    comparability of the whole calibration trajectory already measured. The choice is to
    write it down.
    """
    import yaml

    repo = manifest.get("arms.calibration.repo", "prompt_calibration")
    path = REPO_ROOT / repo / "calibration_datasets" / "v1" / "manifest.yaml"
    out: dict = {"manifest": str(path.relative_to(REPO_ROOT)) if path.exists() else None}
    if not path.exists():
        return out
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return out
    frozen = ((raw.get("sources") or {}).get("population") or {})
    frozen_sha = frozen.get("sha256")
    current = (run.get("population") or {})
    current_sha = current.get("sha256")
    out.update({
        "frozen_population_path": frozen.get("path"),
        "frozen_population_sha256": frozen_sha,
        "run_population_path": current.get("path"),
        "run_population_sha256": current_sha,
        "diverges": bool(frozen_sha and current_sha and frozen_sha != current_sha),
    })
    return out


def build_common_set(manifest, cerema: dict) -> tuple[dict, list[dict]]:
    run = frames.resolve_run(manifest)
    if not run.get("exists") or not run.get("moves", {}).get("exists"):
        return {"available": False,
                "reason": "Le run configuré est introuvable, ou il ne contient pas de "
                          "moves.csv exploitable.",
                "expected": [str(manifest.get("common_set.run")) + "/moves.csv"]}, []

    moves_path = REPO_ROOT / run["moves"]["path"]
    rows, stats = frames.read_moves(
        moves_path, manifest.get("common_set.exclude_selection_methods", []))
    warnings = []
    if stats.get("occupation_inconnue"):
        warnings.append(f"{stats['occupation_inconnue']} trajets portent une occupation "
                        "hors du référentiel EMC² : exclus de la dimension occupation.")
    # The log fills the column since action A2, but the pinned run may predate
    # it: the warning must say which of the two cases is being read, otherwise
    # it would blame the log for a gap that comes from the run's date.
    vides = stats.get("type_logement_vide", 0)
    if vides and vides == len(rows):
        warnings.append("Le type de logement est vide sur la totalité des trajets. Le "
                        "journal de déplacements renseigne désormais cette colonne "
                        "(action A2) et la population porte le trait, mais le run "
                        "épinglé a été écrit avant : l'axe se remplira au prochain run "
                        "épinglé, pas sur celui-ci.")
    elif vides:
        warnings.append(f"{vides} trajets sans type de logement : le domicile tombe "
                        "hors de la couche de zones fines, où le trait n'est pas imputé "
                        "(« non renseigné » n'est pas une modalité).")
    if stats.get("type_logement_hors_referentiel"):
        warnings.append(f"{stats['type_logement_hors_referentiel']} trajets portent un "
                        "type de logement « Autres » : l'enquête connaît cette modalité, "
                        "la ventilation EMC² publiée ne la reprend pas — ces trajets "
                        "sortent de la dimension type de logement.")
    if stats.get("sans_distribution"):
        warnings.append(f"{stats['sans_distribution']} trajets sans distribution de "
                        "probabilité (erreur LLM, itinéraire unique ou entrée de cache "
                        "héritée) : le mode retenu leur sert de masse.")
    if not run.get("population", {}).get("exists"):
        warnings.append("Le fichier de population du run est introuvable : le volet "
                        "modèle ne pourra pas reconstruire ses variables.")

    # ── Scope: what the page excluded, and why (ticket 008, A6) ────
    # These counts are not quality warnings but the very definition
    # of what the page measures. Hiding them would pass off a subset of the
    # log as the whole log.
    excl_methodes = manifest.get("common_set.exclude_selection_methods", [])
    if stats.get("exclues_methode"):
        warnings.append(
            f"{stats['exclues_methode']} lignes écartées du scoring parce qu'elles ne "
            f"portent pas de décision modale ({', '.join(excl_methodes)}). Les replis "
            "d'erreur LLM en font partie : le contrôleur y prend l'itinéraire par "
            "défaut, il n'y a pas de choix à noter.")
    # Hot resume (`make run OFFLINE=1 CONT=1`): the simulated day is replayed from
    # t0 in the SAME experiment folder, and the log carries the same
    # (person, activity) pairs twice, both dated on that simulated day. The cut at the
    # first simulated day therefore does not separate them. Saying so beats hiding it: the
    # reader would otherwise think they read a run played in one go.
    if stats.get("exclues_reprise"):
        jours = ", ".join(stats.get("jours_de_calcul") or [])
        warnings.append(
            f"Run repris à chaud (calculs datés du {jours}) : "
            f"{stats['exclues_reprise']} lignes en doublon écartées, seule la tentative "
            "la plus récente de chaque décision est comptée. Sans cette coupe, les "
            "décisions rejouées pèseraient deux fois dans les parts modales — et le "
            "biais est du même ordre que les gains que la calibration mesure.")
    if stats.get("jour_retenu"):
        warnings.append(
            f"Périmètre borné au premier jour simulé du run ({stats['jour_retenu']}) : "
            f"{stats.get('exclues_jour', 0)} lignes postérieures écartées. Le bootstrap "
            "24 h et l'horizon glissant de planification font déborder le journal "
            "au-delà de la journée mesurée, en répétant les mêmes couples "
            "(personne, activité). Le volet 2 applique la même coupe sur sim_day.")

    n_persons = len({r["agent_id"] for r in rows})
    total = max(1, len(rows))
    # A configured path that resolves elsewhere is a symlink: the page then does
    # not describe a stable run, and saying so beats hiding it (action A1).
    configured = str(run.get("configured", ""))
    resolved = run.get("path", "")
    common = {
        "available": True,
        "run_id": run.get("run_id", "?"),
        "run_path": resolved,
        "run_pinned": bool(resolved) and configured.rstrip("/") == resolved.rstrip("/"),
        "run_date": (run.get("moves", {}).get("mtime") or "")[:10],
        "frozen_sets": frozen_sets_lineage(manifest, run),
        "n_trips": len(rows),
        "n_persons": n_persons,
        "pct_distribution": 100.0 * stats.get("avec_distribution", 0) / total,
        "sim_day": stats.get("jour_retenu"),
        "n_excluded_method": stats.get("exclues_methode", 0),
        "n_excluded_day": stats.get("exclues_jour", 0),
        # Attempts excluded from a resumed run: 0 on a run played in one go.
        "n_excluded_resume": stats.get("exclues_reprise", 0),
        "resumed": bool(stats.get("reprise")),
        "compute_days": stats.get("jours_de_calcul") or [],
        # Breakdown of « Contrainte de chaîne » (ticket 008, A4): what share of the
        # kept decisions was made on an option set already restricted by
        # vehicle consistency. These rows ARE in the score — the breakdown tells
        # the reader how many there are, it does not take them out.
        "chain_constraints": {k.split("::", 1)[1]: v for k, v in stats.items()
                              if isinstance(k, str) and k.startswith("contrainte::")},
        "stats": stats,
        "warnings": warnings,
        "coverage": {},
    }
    return common, rows


def build_simulation(rows: list[dict], cerema: dict, scorer) -> dict:
    if not rows:
        return {"status": "missing",
                "reason": "Aucun trajet exploitable dans le run.",
                "expected": [],
                "action": "Vérifier common_set.run dans sources.yaml"}
    variants = frames.simulation_frames(rows)
    out: dict[str, Any] = {"status": "ok", "variants": {}}
    details: dict[str, list[dict]] = {}
    for name, frame in variants.items():
        gview = frames.global_view(frame, cerema)
        scores = scorer.score(frame, cerema) if scorer else {}
        out["variants"][name] = {"global": gview, "scores": scores,
                                 "n_rows": len(frame)}
    expected = variants["attendu"]
    for dim in DIMENSIONS:
        detail = frames.dimension_detail(expected, cerema, dim)
        if any(d["n"] for d in detail):
            details[dim["key"]] = detail
    out["details"] = details
    out["worst_strata"] = frames.worst_strata(
        {k: v for k, v in details.items()
         if any(d["key"] == k and d["scored"] for d in DIMENSIONS)})
    gv = out["variants"]["attendu"]["global"]
    total_mass = gv["mass"] + gv["excluded_mass"]
    out["excluded_pct"] = 100.0 * gv["excluded_mass"] / total_mass if total_mass else 0.0
    return out


def build_lineage(history: dict, by_regime: dict, pinned: dict) -> Optional[dict]:
    """Trajectory of an entire lineage measured under a single regime (action A5).

    A chronological curve mixes branches and nodes with no kinship: it
    says « the store contains scores », not « the calibration progressed ». A
    **lineage** — the seed → leaf chain of accepted mutations — does say it, on
    condition that all its nodes are measured under the same regime.

    The leaf is **pinned in the manifest**, for the same reason as the run
    of the common set: an automatic reconstruction (« the longest chain
    available ») would change subject with each campaign, without the page
    flagging it. A missing node is not hidden: it appears without a score, and the
    lineage is declared incomplete.
    """
    leaf = (pinned or {}).get("leaf")
    if not leaf:
        return None
    parents = {n["hash"]: n.get("parent") for n in history["nodes"]}
    chain = frames.lineage_chain(leaf, parents, history.get("edges", {}))
    if len(chain) < 2:
        return None

    def measured(bucket: dict) -> dict:
        """Measure of the lineage under a regime: one score per node of the chain."""
        by_hash = {n["hash"]: n for n in bucket["nodes"]}
        steps = [{"short": h[:8], "score": (by_hash.get(h) or {}).get("recomputed"),
                  "branch": (by_hash.get(h) or {}).get("branch", "—")} for h in chain]
        scored = [s for s in steps if s["score"] is not None]
        return {
            "label": bucket["label"], "steps": steps,
            # Provenance: the cache keys actually traversed. Several keys
            # for one regime = several providers (hence several API keys)
            # on the same model — the measurement is the same, the trace says so.
            "params_keys": sorted(bucket.get("keys") or ()),
            "n_scored": len(scored), "complete": len(scored) == len(chain),
            "seed_score": scored[0]["score"] if scored else None,
            "leaf_score": scored[-1]["score"] if scored else None,
            "gain": (scored[0]["score"] - scored[-1]["score"]) if len(scored) >= 2 else None,
        }

    # A regime says something about the lineage only from two measured nodes on.
    regimes = [m for m in (measured(b) for b in by_regime.values()) if m["n_scored"] >= 2]
    if not regimes:
        return None

    # Primary regime: the one requested by the manifest if it covers the lineage,
    # otherwise the best covering one. The others stay shown alongside — the same
    # lineage measured by two instruments tells whether the gain is due to the instrument.
    wanted = (pinned or {}).get("regime")
    primary = next((m for m in regimes if m["label"] == wanted), None)
    if primary is None:
        primary = max(regimes, key=lambda m: m["n_scored"])
    others = [m for m in regimes if m is not primary]
    others.sort(key=lambda m: -m["n_scored"])
    return {
        **primary,
        "leaf": leaf[:8],
        "n_nodes": len(chain),
        "pinned_regime": wanted,
        "is_pinned": bool(wanted) and primary["label"] == wanted,
        "regimes": [primary] + others,
    }


def sample_predicate(sample: dict):
    """Rebuilds the sampling filter described by the measurement file.

    The descriptor (namespace, modulus, threshold) is written INTO the produced file:
    the page thus replays the rule as it was used, and not a copied
    constant that could diverge. Falls back on the producer's constants when
    the descriptor is incomplete (file from an earlier version).
    """
    from .common_set_eval import (SAMPLE_BUCKET_MAX, SAMPLE_MODULUS,
                                  SAMPLE_NAMESPACE)
    # Fall back on `is None` and not on truthiness: a threshold of 0 is a
    # LEGITIMATE value (empty sample), and `0 or default` would silently replace it
    # with the current threshold — the page would then describe another sample than the
    # one that was measured.
    sample = sample or {}
    namespace = sample.get("namespace")
    namespace = SAMPLE_NAMESPACE if namespace is None else str(namespace)
    modulus = sample.get("modulus")
    modulus = SAMPLE_MODULUS if modulus is None else int(modulus)
    bucket_max = sample.get("bucket_max")
    bucket_max = SAMPLE_BUCKET_MAX if bucket_max is None else int(bucket_max)
    if modulus <= 0:
        raise ValueError(f"Invalid sampling modulus: {modulus}")

    def keep(agent_id: str) -> bool:
        digest = hashlib.sha256(f"{namespace}:{agent_id}".encode()).hexdigest()
        return int(digest, 16) % modulus < bucket_max

    return keep


def build_simulation_on_sample(rows: list[dict], cerema: dict, scorer,
                               sample: dict) -> Optional[dict]:
    """Strand 1 restricted to the strand 2 sample — the size control.

    Without it, the matrix sets a column computed on 5,945 decisions against
    columns computed on ~500, and credits the prompt with a gap that largely
    comes from the number of persons observed: per-stratum divergences
    (JSD, EMD) are biased upwards when sample sizes are small, and
    the effect is far from negligible — MEASURED on this run, +5.02 points of
    composite score for the mere reduction from 881 persons to 81, with decisions
    unchanged.

    This row is therefore the honest point of comparison for strand 2: same run,
    same persons, same loss. It does not replace the « Simulation » row,
    which remains the reference measurement on the whole run; it says by how much
    the reading must be corrected before crediting a gap to the prompt.
    """
    if not rows or scorer is None or not sample:
        return None
    keep = sample_predicate(sample)
    subset = [r for r in rows if keep(r["agent_id"])]
    if not subset:
        return None
    frame = frames.simulation_frames(subset)["attendu"]
    scores = scorer.score(frame, cerema)
    dims = scores.get(scorer.primary.name, {}) if scores else {}
    n_persons = len({r["agent_id"] for r in subset})
    # The sampling rule is the same on both sides, but it does not apply to the
    # same pool: strand 2 starts from `llm_exchanges.jsonl` (only decisions
    # that went through a real LLM call are there), this control starts from `moves.csv` (all
    # decisions, cache included). The control may thus be a little larger than
    # the sample it neutralises. The gap is published rather than hidden: this is the
    # column meant to say what the sample size costs, it cannot get the size
    # wrong silently.
    return {
        "dims": dims,
        "composite": dims.get("composite"),
        "n_trips": len(subset),
        "n_persons": n_persons,
        "n_persons_arm2": sample.get("n_agents"),
        "persons_match": sample.get("n_agents") in (None, n_persons),
    }


def build_common_set_eval(source, cerema: dict, scorer,
                          nodes_table: list[dict],
                          pinned_run: Optional[str] = None) -> dict:
    """Prompts re-evaluated on the common set (action A3), scored like the rest.

    This is what makes strand 2 commensurable with strand 1: the same personas, the
    same run, the same loss and the same weights as everywhere else on the page
    (``length_penalty`` at 0). Nothing is recomputed askew here — the score comes
    from the same ``Scorer`` as the simulation.

    Each prompt ALSO carries its score on the frozen personas, under the same
    measurement regime when it exists. These are two different figures, and the page must
    say which one it shows: the first answers « where is the calibration on the
    run », the second « where was it on its own training set ».

    Missing source → ``{"available": False}``: the page shows its « Données
    manquantes » card and builds normally (case of a clone without the data).
    """
    if not source.exists or scorer is None:
        return {"available": False}
    entries = frames.load_common_set_eval(source.path)
    if not entries:
        return {"available": False}

    # Substrate guard: strand 2 only makes sense if it was measured on the run
    # the page pins. Without this check, changing run left the previous
    # measurement in place and the matrix compared two substrates while announcing them
    # as one — the store cache, indexed on the set name alone, produced
    # exactly this situation on 2026-07-31. The measurement is set aside rather than
    # corrected: it is right, but it bears on another run.
    measured_on = {str((e.get("sample") or {}).get("run") or "?") for e in entries}
    if pinned_run and measured_on != {str(pinned_run)}:
        return {"available": False,
                "reason": (f"Mesure faite sur {', '.join(sorted(measured_on))}, "
                           f"alors que la page épingle {pinned_run}."),
                "action": "Reproduire la mesure sur le run épinglé : "
                          "make common-set-eval (chiffrer d'abord : DRY_RUN=1)"}

    # Score of the same node on the frozen personas, under the same regime: this is the
    # point of comparison that tells whether the calibration gain carries over.
    frozen = {(r["short"], r["regime"]): r for r in nodes_table
              if r.get("recomputed") is not None}

    out = []
    for entry in entries:
        scores = scorer.score(entry["rows"], cerema)
        dims = scores.get(scorer.primary.name, {}) if scores else {}
        regime = (entry.get("regime") or {}).get("label")
        ref = frozen.get((entry.get("short"), regime))
        out.append({
            "role": entry.get("role"), "label": entry.get("label"),
            "short": entry.get("short"), "node": entry.get("node"),
            "branch": entry.get("branch"),
            "regime": regime,
            "regime_detail": entry.get("regime") or {},
            "sample": entry.get("sample") or {},
            "coverage": entry.get("coverage"),
            "n_decisions": entry.get("n_decisions"),
            "created_at": entry.get("created_at"),
            "dims": dims,
            "composite": dims.get("composite"),
            "secondary": scores.get(scorer.secondary.name, {}).get("composite")
            if (scorer.secondary and scores) else None,
            "frozen_composite": (ref or {}).get("recomputed"),
            "frozen_regime": regime if ref else None,
        })

    seed = next((e for e in out if e["role"] == "seed"), None)
    leaf = next((e for e in out if e["role"] == "leaf"), None)
    gain = None
    if seed and leaf and seed["composite"] is not None and leaf["composite"] is not None:
        gain = seed["composite"] - leaf["composite"]
    frozen_gain = None
    if seed and leaf and seed["frozen_composite"] is not None \
            and leaf["frozen_composite"] is not None:
        frozen_gain = seed["frozen_composite"] - leaf["frozen_composite"]
    sample = (out[0].get("sample") or {}) if out else {}
    # A measurement made under two different regimes cannot be read as one block: we
    # flag it rather than let a trajectory be assumed.
    regimes = {e["regime"] for e in out if e["regime"]}
    return {
        "available": True,
        "entries": out,
        "seed": seed, "leaf": leaf,
        "gain": gain, "frozen_gain": frozen_gain,
        "sample": sample,
        "regime": sorted(regimes)[0] if len(regimes) == 1 else None,
        "mixed_regimes": len(regimes) > 1,
        "path": source.rel,
    }


def resample_composite(rows: list[dict], cerema: dict, scorer, n_agents: int,
                       n_draws: int = 200, seed: int = 20260731) -> Optional[dict]:
    """Distribution of a frame's composite score brought down to ``n_agents`` persons.

    This is the sample-size control of strand 2, and it costs no LLM call: we
    replay the score of the **already stored** decisions, on randomly drawn
    subsets of persons. The draw is PER PERSON — all the decisions
    of a selected person are kept — because it is the number of
    persons per stratum that biases JSD and EMD, not the number of rows.

    Why it is indispensable here: the ``train`` set holds 298 persons, the
    ``test`` set 66. A3 measured, on the simulation and with decisions unchanged,
    +5.02 points of composite score for the mere reduction from 881 persons to 81. A
    train → test gap read raw would thus confuse overfitting with the sample-size
    effect, and would publish precisely the misleading figure that action A4
    claims to produce.

    The seed is fixed: the page rebuilds identically. ``None`` if the frame
    does not have enough persons for the draw to make sense.
    """
    if not rows or scorer is None or n_agents <= 0:
        return None
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[str(row.get("agent_id"))].append(row)
    agents = sorted(by_agent)
    if len(agents) <= n_agents:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_draws):
        picked = rng.sample(agents, n_agents)
        subset = [r for a in picked for r in by_agent[a]]
        scores = scorer.score(subset, cerema)
        composite = scores.get(scorer.primary.name, {}).get("composite")
        if composite is not None:
            values.append(composite)
    if not values:
        return None
    values.sort()

    def quantile(q: float) -> float:
        return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]

    return {
        "n_agents": n_agents, "n_draws": len(values),
        "n_agents_source": len(agents),
        "mean": sum(values) / len(values),
        "p05": quantile(0.05), "median": quantile(0.50), "p95": quantile(0.95),
        "seed": seed,
    }


def resample_gain(frame_a: list[dict], frame_b: list[dict], cerema: dict, scorer,
                  n_agents: int, n_draws: int = 200,
                  seed: int = 20260731) -> Optional[dict]:
    """Distribution of the seed → leaf **gain** at the size of the held-out set.

    The per-node control (``resample_composite``) is noisy: resampling 66
    persons out of 298 moves the composite score by ±7 points, so that no
    level gap stands out from the band. The *gain*, however, is **paired** — the
    two prompts are scored on the **same** drawn persons — and its noise largely
    cancels out. It is thus the quantity on which a generalisation
    conclusion can actually rest.

    Both frames must carry the same persons (two evals of the same frozen
    set): otherwise the draws would no longer be paired and the comparison
    would lose what makes it worthwhile. We require it rather than hope for it.
    """
    if not frame_a or not frame_b or scorer is None or n_agents <= 0:
        return None
    a_by_agent: dict[str, list[dict]] = defaultdict(list)
    b_by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in frame_a:
        a_by_agent[str(row.get("agent_id"))].append(row)
    for row in frame_b:
        b_by_agent[str(row.get("agent_id"))].append(row)
    agents = sorted(set(a_by_agent) & set(b_by_agent))
    if len(agents) <= n_agents:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_draws):
        picked = rng.sample(agents, n_agents)
        sa = scorer.score([r for p in picked for r in a_by_agent[p]], cerema)
        sb = scorer.score([r for p in picked for r in b_by_agent[p]], cerema)
        ca = sa.get(scorer.primary.name, {}).get("composite")
        cb = sb.get(scorer.primary.name, {}).get("composite")
        if ca is not None and cb is not None:
            values.append(ca - cb)
    if not values:
        return None
    values.sort()

    def quantile(q: float) -> float:
        return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]

    return {
        "n_agents": n_agents, "n_draws": len(values),
        "n_agents_paired": len(agents),
        "mean": sum(values) / len(values),
        "p05": quantile(0.05), "median": quantile(0.50), "p95": quantile(0.95),
        "seed": seed,
    }


def _resolve_train_dataset(by_key: dict, chain: list[str], regime: Optional[str],
                           preferred_version: str) -> str:
    """Set name under which the `train` evals of THIS lineage are stored.

    The store names an eval by split AND version as soon as we leave v1
    (`train`, then `train@v2`…). The page therefore cannot hard-code the name:
    it collects those that exist for the lineage nodes, and prefers the one
    that carries the same version as the held-out set — that is the comparison that
    makes sense. Failing that, the bare name (v1), then any other, in deterministic
    order. No candidate: we return the bare name, and the train column will come out
    empty as before.
    """
    shorts = {h[:8] for h in chain}
    available = {ds for (short, reg, ds) in by_key
                 if short in shorts and reg == regime
                 and (ds == "train" or ds.startswith("train@"))}
    for candidate in (f"train@{preferred_version}", "train"):
        if candidate in available:
            return candidate
    return min(available) if available else "train"


def build_generalization(chain: list[str], by_key: dict, frames_by_key: dict,
                         cerema: dict, scorer, profile: dict,
                         regime: Optional[str], split_rule: Optional[str],
                         heldout_dataset: str = "test",
                         n_draws: int = 200) -> Optional[dict]:
    """The calibration score on a set the loop has never seen (action A4).

    All the rest of strand 2 is measured on ``train`` — the set on which the
    lineage was *optimised*. A training composite score does not tell apart a
    prompt that understood the population from a prompt that memorised its 298
    personas. This block brings the missing figure, and the two safeguards without
    which it would be misleading:

    1. **The sample-size control** (``resample_composite``). The two sets do not have
       the same size: comparing their raw levels would credit the prompt with a
       gap that largely comes from the number of persons observed.
    2. **The nature of the split** (``profile``, established from the files by
       ``heldout_eval.dataset_profile``). A split by person and a
       split by trip do not support the same claim: the
       first says « never-seen individuals », the second only « other
       trips of the same individuals ». The page must say which one it is.

    Returns ``None`` as long as no node is measured on both sides: the page
    then shows its « Données manquantes » card rather than a half-measurement.
    """
    if not chain or not regime:
        return None
    # The frozen-set profile is indexed by SPLIT name (`test`), whereas the
    # store names the eval by split AND version (`test@v2`, so as not to confuse two
    # sets with different weather). Without this stripping, the held-out set size
    # came back as 0 — and with it the size control, which depends on it directly.
    held = profile.get(heldout_dataset) or profile.get(heldout_dataset.split("@", 1)[0]) or {}
    n_agents_held = held.get("n_agents") or 0

    # Frozen-set version on each side of the comparison. The « train » side
    # comes from the CAMPAIGN evals, which ran on the version of the time; the
    # held-out side comes from the re-evaluation, done on the version the page pins.
    # When they differ, the train → held-out gap no longer measures only the effect of
    # the split: it also carries the regime change (in v2, the weather is drawn
    # from the climatic year instead of being that of the source run, uniformly
    # sunny). The page must say so — it is the explicit counterpart of moving to v2.
    held_version = heldout_dataset.split("@", 1)[1] if "@" in heldout_dataset else "v1"
    # The train side is READ from the store, never assumed. Freezing it to « v1 »
    # held as long as the campaign had only measured bare-named splits; the day
    # it stores its evals under `train@v2`, a frozen name does not just mislabel
    # — `by_key[(short, regime, "train")]` no longer finds anything and the whole
    # train column disappears, sample-size control included.
    train_dataset = _resolve_train_dataset(by_key, chain, regime, held_version)
    train_version = (train_dataset.split("@", 1)[1] if "@" in train_dataset else "v1")
    versions_match = held_version == train_version

    steps = []
    for rank, node_hash in enumerate(chain):
        short = node_hash[:8]
        train_row = by_key.get((short, regime, train_dataset))
        held_row = by_key.get((short, regime, heldout_dataset))
        if train_row is None and held_row is None:
            continue
        control = None
        train_frame = frames_by_key.get((short, regime, train_dataset))
        if train_frame and n_agents_held:
            control = resample_composite(train_frame, cerema, scorer,
                                         n_agents_held, n_draws=n_draws)
        if rank == 0:
            role, label = "seed", "Graine"
        elif rank == len(chain) - 1:
            role, label = "leaf", "Meilleur prompt"
        else:
            role, label = "step", f"Étape {rank}"
        train_score = (train_row or {}).get("recomputed")
        held_score = (held_row or {}).get("recomputed")
        steps.append({
            "short": short, "role": role, "label": label, "rank": rank,
            "branch": (train_row or held_row or {}).get("branch", "—"),
            "train": train_score,
            "held": held_score,
            "held_dims": (held_row or {}).get("dims") or {},
            "train_dims": (train_row or {}).get("dims") or {},
            # RAW gap: shown, but never alone — it mixes the sample-size effect
            # and what could be overfitting.
            "gap_raw": (held_score - train_score)
            if (held_score is not None and train_score is not None) else None,
            "control": control,
            # CORRECTED gap: the test compared with the train brought down to the test size.
            # It is the only one of the two that speaks about the prompt.
            "gap_controlled": (held_score - control["mean"])
            if (held_score is not None and control) else None,
            "within_control": (control["p05"] <= held_score <= control["p95"])
            if (held_score is not None and control) else None,
        })
    measured = [s for s in steps if s["held"] is not None]
    if not measured:
        return None

    seed_step = next((s for s in steps if s["role"] == "seed"), None)
    leaf_step = next((s for s in steps if s["role"] == "leaf"), None)

    def gain(field: str) -> Optional[float]:
        if not (seed_step and leaf_step):
            return None
        a, b = seed_step.get(field), leaf_step.get(field)
        return (a - b) if (a is not None and b is not None) else None

    # Paired GAIN control: it is the one that carries the conclusion, the per-node
    # control being too noisy to decide (cf. resample_gain).
    gain_control = None
    if seed_step and leaf_step and n_agents_held:
        gain_control = resample_gain(
            frames_by_key.get((seed_step["short"], regime, train_dataset)) or [],
            frames_by_key.get((leaf_step["short"], regime, train_dataset)) or [],
            cerema, scorer, n_agents_held, n_draws=n_draws)

    # Is the split by person? Established from the files themselves, not
    # on the strength of the rule declared in the frozen-set manifest.
    shared = held.get("agents_shared_with_train")
    by_person = shared == 0
    return {
        "gain_control": gain_control,
        "available": True,
        "dataset": heldout_dataset,
        "regime": regime,
        "split_rule": split_rule,
        "profile": profile,
        "n_records": held.get("n_records"),
        "n_agents": n_agents_held,
        "train_records": (profile.get("train") or {}).get("n_records"),
        "train_agents": (profile.get("train") or {}).get("n_agents"),
        "agents_shared_with_train": shared,
        "by_person": by_person,
        # Difference in input FORM between the two sets, not in population:
        # `calibration.datasets` removes the « Historique » section from val and test
        # (STM/LTM memory of the source run, not reproducible) and keeps it in the
        # train. The test prompt is thus not only addressed to other
        # persons, it is one section shorter.
        "memory_train": (profile.get("train") or {}).get("memory_share"),
        "memory_held": held.get("memory_share"),
        # Frozen-set regime on each side. Different ⇒ the train → held-out gap
        # is not a pure split effect.
        "train_version": train_version,
        "held_version": held_version,
        "versions_match": versions_match,
        "steps": steps,
        "seed": seed_step, "leaf": leaf_step,
        "n_measured": len(measured), "n_nodes": len(chain),
        "complete": len(measured) == len(chain),
        "gain_train": gain("train"),
        "gain_held": gain("held"),
    }


def build_calibration(manifest, cerema: dict, scorer) -> dict:
    repo = manifest.get("arms.calibration.repo", "prompt_calibration")
    dataset_dir = manifest.path_of("arms.calibration.datasets")
    manifest.track("calibration.datasets", dataset_dir or "prompt_calibration/calibration_datasets/v1",
                   "Jeux de personas gelés (train/val/test/screen)")
    metadata = frames.load_dataset_metadata(dataset_dir) if dataset_dir and dataset_dir.exists() else {}
    if not metadata:
        return {"status": "missing",
                "reason": "Les jeux de personas gelés sont introuvables : impossible de "
                          "rattacher les décisions stockées à leurs attributs.",
                "expected": [f"{repo}/calibration_datasets/v1/*.jsonl"],
                "action": "Reconstruire les jeux (calibrate datasets)"}

    keep = manifest.get("arms.calibration.keep_verdicts", ["accepted", "imported"])
    stores_out = []
    nodes_table: list[dict] = []
    eval_models: set[str] = set()
    # Nodes measured on a **held-out** set (val/test), indexed by
    # (node, regime, set). They do not join `nodes_table` — the trajectory and
    # the lineage of strand 2 are read on train, and mixing another set in
    # would overlay two populations in the same curve. They feed the
    # generalisation block, which sets them explicitly against each other.
    by_key: dict[tuple, dict] = {}
    # Matching decision frames: kept OUTSIDE the payload (they weigh
    # thousands of rows) and consumed only by the sample-size control.
    frames_by_key: dict[tuple, list] = {}
    lineage_chain_full: list[str] = []
    # Train evals QUALIFIED by version (`train@v2`…) met but set aside
    # from the main curve: counted so as to be reported, never merged into it.
    train_versionne: Counter = Counter()

    for entry in manifest.get("arms.calibration.stores", []) or []:
        src = manifest.track(f"calibration.store.{entry['id']}", entry["path"],
                             f"Store de calibration — {entry.get('label', entry['id'])}")
        if not src.exists:
            stores_out.append({"id": entry["id"], "label": entry.get("label", entry["id"]),
                               "totals": {"nodes": 0, "mutations": 0, "evals": 0},
                               "kept": 0, "eval_models": [], "series": []})
            continue
        history = frames.read_store_history(src.path, keep)
        by_regime: dict[str, dict] = {}
        scored_nodes: list[dict] = []
        seen: set[tuple] = set()
        for node in history["nodes"]:
            # One node may carry several evals (replays, distinct params_key
            # values): we keep only one row per node × set × regime triple.
            key = (node["short"], node["dataset"], node["eval_model"],
                   node["params_key"])
            if key in seen:
                continue
            seen.add(key)
            frame = frames.decisions_frame(node["decisions"], metadata, scorer.categorize) \
                if scorer else []
            scores = scorer.score(frame, cerema) if (frame and scorer) else {}
            recomputed = scores.get(scorer.primary.name, {}).get("composite") \
                if scorer and scores else None
            regime = frames.eval_regime(node["params_key"], node["eval_model"])
            row = {**{k: node[k] for k in
                      ("hash", "short", "branch", "created_at", "verdict", "eval_model")},
                   "store": entry.get("label", entry["id"]),
                   "dataset": node["dataset"],
                   "regime": regime["label"], "regime_key": regime["key"],
                   "recomputed": recomputed,
                   "stored": node["stored_scores"].get("composite"),
                   "dims": scores.get(scorer.primary.name, {}) if scorer else {}}
            # Indexed for the generalisation block, all sets together. The
            # first measurement wins: stores are walked from the oldest to the
            # richest, and a duplicate brought back from the cloud carries the same
            # decisions.
            by_key.setdefault((node["short"], regime["label"], node["dataset"]), row)
            frames_by_key.setdefault(
                (node["short"], regime["label"], node["dataset"]), frame)
            # The main curve carries ONLY the bare-named train split (v1). A
            # `train@vN` is not merged into it: that would mix two substrates in one
            # curve, exactly the defect that version qualification
            # comes to correct. It is counted to be FLAGGED further down — a measurement
            # paid for then made invisible without a word is what gets the eval
            # rerun a second time.
            if node["dataset"] != "train":
                if node["dataset"].startswith("train@"):
                    train_versionne[node["dataset"]] += 1
                continue
            eval_models.add(regime["label"])
            nodes_table.append(row)
            if recomputed is not None:
                scored_nodes.append(row)
                # Grouping by LABEL (model · policy) and not by raw
                # `params_key`: the latter also carries the provider name, hence
                # the API key used. Two keys on the same model query the
                # same model — it is a single measurement regime, and a replay finished
                # on the second key (first key's quota exhausted) must remain a
                # single curve.
                bucket = by_regime.setdefault(
                    regime["label"], {"label": regime["label"], "points": [],
                                      "nodes": [], "keys": set()})
                bucket["keys"].add(regime["key"])
                bucket["points"].append(
                    {"score": recomputed,
                     "label": f'{node["short"]} · {node["branch"]}'})
                bucket["nodes"].append(row)
        series = [{"label": b["label"], "points": b["points"]}
                  for b in by_regime.values() if b["points"]]

        # Seed and best node are taken WITHIN A SINGLE regime — the richest one.
        # Comparing them across regimes would amount to setting two measuring
        # instruments against each other: this is precisely what action A5 corrects.
        ref = max(by_regime.values(), key=lambda b: len(b["nodes"])) if by_regime else None
        ref_nodes = ref["nodes"] if ref else []
        seeds = [n for n in ref_nodes if n["verdict"] == "seed"]
        best = min(ref_nodes, key=lambda n: n["recomputed"]) if ref_nodes else None
        pinned_lineage = manifest.get("arms.calibration.lineage") or {}
        lineage = build_lineage(history, by_regime, pinned_lineage)
        if pinned_lineage.get("leaf") and not lineage_chain_full:
            candidate = frames.lineage_chain(
                pinned_lineage["leaf"],
                {n["hash"]: n.get("parent") for n in history["nodes"]},
                history.get("edges", {}))
            if len(candidate) > 1:
                lineage_chain_full = candidate
        stores_out.append({
            "id": entry["id"], "label": entry.get("label", entry["id"]),
            "totals": history["totals"],
            "kept": len(scored_nodes),
            "hashes": sorted({n["short"] for n in scored_nodes}),
            "eval_models": sorted({b["label"] for b in by_regime.values()}),
            "reference_regime": ref["label"] if ref else None,
            "series": series,
            "lineage": lineage,
            "seed": min(seeds, key=lambda n: n["created_at"] or "") if seeds else None,
            "best": best,
            "span": ([round(min(n["recomputed"] for n in ref_nodes), 2),
                      round(max(n["recomputed"] for n in ref_nodes), 2)]
                     if ref_nodes else None),
        })

    prompts_path = manifest.path_of("arms.calibration.prompts_yaml")
    src = manifest.track("calibration.prompts", prompts_path or "mobility_llm/prompts/prompts.yaml",
                         "Variantes de prompt livrées à la simulation")
    variants = frames.read_prompt_variants(src.path) if src.exists else {}

    common_eval = manifest.track(
        "calibration.common_set_eval",
        manifest.get("arms.calibration.common_set_eval"),
        "Décisions des prompts ré-évalués sur le jeu commun")
    common_set = build_common_set_eval(common_eval, cerema, scorer, nodes_table,
                                       manifest.get("common_set.run"))

    # ── Generalisation: the set the loop has never seen (action A4) ──────
    # The regime is the one pinned by the manifest. Do not fall back on « the
    # best covering regime »: a test score read under one instrument and a
    # training score read under another cannot be subtracted.
    pinned = manifest.get("arms.calibration.lineage") or {}
    heldout_dataset = manifest.get("arms.calibration.heldout_dataset", "test")
    profile = heldout_eval.dataset_profile(dataset_dir) if dataset_dir else {}
    generalization = build_generalization(
        lineage_chain_full, by_key, frames_by_key, cerema, scorer, profile,
        pinned.get("regime"), heldout_eval.split_rule(dataset_dir) if dataset_dir else None,
        heldout_dataset)
    if generalization is None:
        generalization = {
            "available": False,
            "dataset": heldout_dataset,
            "regime": pinned.get("regime"),
            "profile": profile,
            "reason": f"Aucun nœud de la lignée épinglée n'est évalué sur le jeu "
                      f"« {heldout_dataset} » sous le régime "
                      f"{pinned.get('regime') or 'épinglé'} : la calibration n'a de "
                      f"score que sur le jeu qui a servi à l'optimiser.",
            "action": f"make heldout-eval PROVIDER=google_gemini31_key2 "
                      f"(chiffrer d'abord : make heldout-eval DRY_RUN=1)",
        }

    # The cloud store is regularly brought back into the local store: plotting the
    # two trajectories would give the same curve twice. We only plot the
    # stores that bring their own nodes.
    # The canonical store is the richest: for equal sets of scored nodes,
    # it is the one that carries the most complete history.
    def richness(store: dict) -> tuple:
        return (len(store.get("hashes") or ()), store["totals"]["nodes"], store["id"])

    for store in stores_out:
        own = set(store.get("hashes") or ())
        if not own:
            store["subset_of"] = None
            continue
        store["subset_of"] = next(
            (o["label"] for o in stores_out
             if o is not store and own <= set(o.get("hashes") or ())
             and richness(o) > richness(store)),
            None)
    duplicated = {s["label"] for s in stores_out if s.get("subset_of")}
    table = [r for r in nodes_table if r["store"] not in duplicated]

    return {
        "status": "ok",
        "stores": stores_out,
        "nodes_table": sorted(table, key=lambda r: r["created_at"] or ""),
        "prompt_variants": variants,
        "mixed_models": len(eval_models) > 1,
        # Train evals qualified by version, set aside from the curve: published
        # so that the page can say so instead of making them disappear.
        "train_versioned_skipped": dict(train_versionne),
        "common_set": common_set,
        "common_set_expected": [common_eval.rel],
        "generalization": generalization,
    }


def renormalisation_bias(variants: dict) -> dict:
    """What renormalisation on the OTP offer adds to each mode, measured.

    The `attendu` composite score is the headline figure of strand 3, and it is **biased**:
    ~20 % of the model's mass falls on modes OTP did not offer and is
    redistributed pro rata, to the benefit of modes almost always offered (public
    transport) and to the detriment of those that are not on short
    trips (walking, cycling). Reading consequence, measured on 2026-08-27 on
    four configurations of the same run: `attendu` **penalises** any correction that
    increases PT, even a right one — it went from 6.015 to 6.208 while the gap of the
    15-19 year-olds, the largest on the page, fell from 63.8 to 48.7 in `l1`.

    We thus compute the `attendu − brut` gap per mode, and where both readings stand
    against the target. Computed, not hard-coded: the bias depends on the run's offer,
    and a frozen value would become wrong at the next run without anything saying so.
    """
    raw = (variants.get("brut") or {}).get("global") or {}
    renorm = (variants.get("attendu") or {}).get("global") or {}
    if not raw.get("actual") or not renorm.get("actual"):
        return {}
    target = renorm.get("target") or {}
    modes = {}
    for mode, after in (renorm.get("actual") or {}).items():
        before = (raw.get("actual") or {}).get(mode)
        if before is None:
            continue
        aim = target.get(mode)
        modes[mode] = {
            "raw_pct": round(before, 2),
            "renormalised_pct": round(after, 2),
            "added_pt": round(after - before, 2),
            "target_pct": round(aim, 2) if aim is not None else None,
            "raw_gap_pt": round(before - aim, 2) if aim is not None else None,
            "renormalised_gap_pt": round(after - aim, 2) if aim is not None else None,
        }
    inflated = sorted((m for m, v in modes.items() if v["added_pt"] > 0),
                      key=lambda m: -modes[m]["added_pt"])
    return {
        "modes": modes,
        "most_inflated": inflated[0] if inflated else None,
        "note": ("La renormalisation sur l'offre OTP déplace de la masse entre modes : "
                 "l'écart `attendu − brut` ci-dessus le chiffre. Un mode dont la "
                 "renormalisation gonfle la part et qui dépasse déjà sa cible fait "
                 "empirer le composite `attendu` à chaque correction qui l'augmente, "
                 "même juste. Lire alors `elu` et `brut`, qui n'ont pas ce biais."),
    }


def build_model_predictions(source, cerema: dict, scorer,
                            pinned_run: Optional[str] = None,
                            pinned_digest: Optional[str] = None,
                            pinned_policy: Optional[str] = None) -> dict:
    """Model predictions on the common set (action A8), scored like the rest.

    Same requirement as for strand 2: the score comes from the same ``Scorer`` — so from the
    engine's ``calibration.metrics``, with ``length_penalty`` at 0 — and bears on the
    scope of strand 1. Nothing is recomputed askew here.

    Three readings are produced. ``attendu`` (probability mass, renormalised on
    the OTP offer) and ``elu`` (most likely mode) are the two faces of strand 1;
    ``brut`` is the same mass **before** renormalisation, and has only one use: measuring
    what the OTP correction really changes, rather than asserting it.

    Missing or unreadable source → ``{"available": False}``: the page shows its
    « Données manquantes » card and builds normally.
    """
    if not source.exists or scorer is None:
        return {"available": False}
    loaded = frames.load_model_predictions(source.path)
    if not loaded:
        return {"available": False}

    # Substrate guard, symmetric to that of strand 2. It was missing here, and the
    # defect was not theoretical: on 2026-08-25, pinning a new run set aside
    # the strand 2 measurement and left the strand 3 one in place, so that the
    # matrix compared a simulation read on one run with a model read on another,
    # while announcing them as a single substrate.
    #
    # The run name is not enough: a hot resume (make run CONT=1) rewrites
    # moves.csv IN THE SAME folder, hence under the same name. The log fingerprint
    # is the only check that tells these two states apart, and the parquet carries it.
    meta_guard = loaded.get("meta") or {}
    measured_run = str(meta_guard.get("run") or "?")
    if pinned_run and measured_run != str(pinned_run):
        return {"available": False,
                "reason": (f"Prédictions faites sur {measured_run}, alors que la page "
                           f"épingle {pinned_run}."),
                "action": "Reproduire la mesure sur le run épinglé : "
                          "make common-set-predict (chiffrer d'abord : DRY_RUN=1)"}
    # Same guard, on the other axis: the POLICY. A retraining with an unchanged
    # variable contract does not move `spec_version`, so nothing flagged that a
    # parquet measured under the old model was served as current.
    measured_policy = meta_guard.get("policy_sha256")
    if measured_policy and pinned_policy and measured_policy != pinned_policy:
        return {"available": False,
                "reason": (f"Prédictions faites sous une autre politique — empreinte "
                           f"{str(measured_policy)[:12]} contre "
                           f"{str(pinned_policy)[:12]} sur le disque. Un "
                           f"ré-entraînement ne change pas `spec_version` : seule "
                           f"l'empreinte distingue les deux modèles."),
                "action": "Reproduire la mesure sous la politique courante : "
                          "make common-set-predict (chiffrer d'abord : DRY_RUN=1)"}
    measured_digest = meta_guard.get("moves_sha256")
    if pinned_digest and measured_digest and measured_digest != pinned_digest:
        return {"available": False,
                "reason": (f"Prédictions faites sur une autre version du journal de "
                           f"{measured_run} — empreinte {str(measured_digest)[:12]} "
                           f"contre {str(pinned_digest)[:12]} sur le disque. Une reprise "
                           f"à chaud réécrit moves.csv sans changer de nom de run."),
                "action": "Reproduire la mesure sur le journal courant : "
                          "make common-set-predict (chiffrer d'abord : DRY_RUN=1)"}

    variants: dict[str, Any] = {}
    for name, frame in loaded["variants"].items():
        if not frame:
            continue
        gview = frames.global_view(frame, cerema)
        scores = scorer.score(frame, cerema)
        dims = scores.get(scorer.primary.name, {})
        variants[name] = {
            "global": gview, "scores": scores, "dims": dims,
            "composite": dims.get("composite"),
            "secondary": scores.get(scorer.secondary.name, {}).get("composite")
            if scorer.secondary else None,
            "n_rows": len(frame),
        }

    details: dict[str, list[dict]] = {}
    expected = loaded["variants"]["attendu"]
    for dim in DIMENSIONS:
        detail = frames.dimension_detail(expected, cerema, dim)
        if any(d["n"] for d in detail):
            details[dim["key"]] = detail

    meta = loaded.get("meta") or {}
    summary = meta.get("summary") or {}
    gv = variants.get("attendu", {}).get("global", {})
    total_mass = gv.get("mass", 0.0) + gv.get("excluded_mass", 0.0)
    return {
        "available": True,
        "variants": variants,
        "details": details,
        "meta": meta,
        "summary": summary,
        "renormalisation_bias": renormalisation_bias(variants),
        # Two excluded masses, and they do not say the same thing: the decisions
        # outside the model's scope (unknown zone, offer with no predictable mode)
        # and, within the kept decisions, the probability share falling on modes
        # outside the four scored ones. The second is zero by construction here — the
        # renormalisation leaves only scored modes.
        "excluded_decisions_pct": summary.get("excluded_pct"),
        "excluded_pct": (100.0 * gv.get("excluded_mass", 0.0) / total_mass
                         if total_mass else 0.0),
        "path": source.rel,
    }


def _policy_digest(manifest) -> Optional[str]:
    """Fingerprint of the policy artefact present on disk, or ``None``."""
    path = manifest.path_of("arms.model.policy")
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(manifest, cerema: dict, scorer) -> dict:
    spec_src = manifest.track("model.feature_spec",
                              manifest.get("arms.model.feature_spec"),
                              "Spécification versionnée des variables du modèle")
    manifest.track("model.dataset", manifest.get("arms.model.dataset"),
                   "Déplacements PROGEDO préparés pour l'entraînement")
    policy = manifest.track("model.policy", manifest.get("arms.model.policy"),
                            "Modèle entraîné et sérialisé")
    preds = manifest.track("model.predictions", manifest.get("arms.model.predictions"),
                           "Probabilités prédites sur le jeu commun")
    zones = manifest.track("model.zones", manifest.get("arms.model.zones"),
                           "Couche de zones fines (résolveur point → zone)")

    # Fingerprint of the log ACTUALLY on disk, not the one a manifest
    # would announce: it is the one that must match the one written in the parquet.
    run_info = frames.resolve_run(manifest)
    moves_rel = ((run_info.get("moves") or {}).get("path")) if run_info else None
    pinned_digest = probe("model.pinned_moves", REPO_ROOT / moves_rel).sha256 \
        if moves_rel else None
    predictions = build_model_predictions(preds, cerema, scorer,
                                          manifest.get("common_set.run"), pinned_digest,
                                          _policy_digest(manifest))
    # What the prediction actually found, rather than what was expected of it:
    # an « available » variable may be massively missing on arrival (a
    # category that the synthetic population carries and the spec does not know
    # becomes missing at encoding, without any error saying so).
    measured_missing = ((predictions.get("meta") or {}).get("feature_missing") or {})
    n_scored = ((predictions.get("summary") or {}).get("n_scored") or 0)

    features = []
    if spec_src.exists:
        try:
            spec = json.loads(spec_src.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            spec = {}
        # Where each variable is found in the common set: this is what decides
        # the feasibility of the strand, well before training.
        persona = {"age", "gender", "household_size", "has_driving_license",
                   "has_pt_subscription", "number_of_cars", "car_availability",
                   "has_bike", "socioprofessional_class", "main_occupation",
                   "employed", "studies"}
        context = {"purpose", "purpose_origin", "departure_hour"}
        for name in spec.get("features", []) or []:
            fname = name if isinstance(name, str) else name.get("name", str(name))
            if fname in persona:
                source, status = "population_*.json → traits_json", "disponible"
            elif fname in context:
                source, status = "moves.csv + chaîne d'activités", "disponible"
            else:
                # Geo variables: derivable from the coordinates as soon as the zone
                # layer is there, the resolver replaying the training formula.
                source = "population_*.json (coordonnées) + couche de zones fines"
                status = ("dérivable — résolveur de zone fine" if zones.exists
                          else "couche absente — `make zones`")
            gap = measured_missing.get(fname)
            if predictions.get("available"):
                status = ("renseignée" if not gap else
                          f"manquante sur {gap} décisions "
                          f"({100.0 * gap / max(1, n_scored):.1f} %)")
            features.append({"name": fname, "source": source, "status": status})

    # The trained model carries its own metrics (it is self-contained by
    # design, cf. fit_mode_choice_policy.py): the page reads them back from it rather
    # than opening one more source. Aggregates only — no survey micro-data
    # passes through here.
    trained = None
    if policy.exists:
        try:
            art = json.loads(policy.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            art = {}
        test = (art.get("metrics") or {})
        shares = test.get("mode_shares") or {}
        if test:
            trained = {
                "spec_version": art.get("spec_version"),
                "best_iteration": (art.get("training") or {}).get("best_iteration"),
                "n_test": test.get("n_rows"),
                "log_loss": test.get("log_loss_weighted"),
                "accuracy": test.get("accuracy_weighted"),
                "l1_mass": shares.get("l1_probability_mass"),
                "l1_argmax": shares.get("l1_argmax"),
                "classes": test.get("classes") or [],
                "observed": shares.get("observed") or [],
                "predicted": shares.get("predicted_probability_mass") or [],
                "top_features": [
                    {"name": f["name"], "gain": f["gain_share"]}
                    for f in (test.get("feature_importances") or [])[:6]
                ],
            }

    return {
        # « missing » as long as the strand cannot produce a score on the common
        # set: a model trained but never applied produces none.
        "status": "ok" if predictions.get("available") else "missing",
        "trained": trained,
        "feature_spec": {"features": features},
        "predictions": predictions,
        "expected": [preds.rel] if policy.exists else [policy.rel, preds.rel],
    }


def build_synthesis(payload: dict) -> dict:
    dims = [{"key": d["key"], "label": d["label"]}
            for d in DIMENSIONS if d["scored"]]
    dims.insert(0, {"key": "global", "label": "Global"})
    dims.append({"key": "composite", "label": "Composite comparable"})

    sim = payload["arms"]["simulation"]
    primary = payload["score_def"]["primary"]
    arms_out = []

    def cells_from(scores: dict) -> list[dict]:
        values = scores.get(primary, {}) if scores else {}
        out = []
        for dim in dims:
            key = "global" if dim["key"] == "global" else dim["key"]
            out.append({"value": values.get(key)})
        return out

    def cells_from_dims(values: dict) -> list[dict]:
        return [{"value": values.get(d["key"])} for d in dims]

    if sim.get("status") == "ok":
        arms_out.append({"label": "Simulation",
                         "cells": cells_from(sim["variants"]["attendu"]["scores"])})
        arms_out.append({"label": "Sim. (tirée)",
                         "cells": cells_from(sim["variants"]["tire"]["scores"])})
        # Size control, inserted JUST BEFORE the calibration columns: it is
        # the one they must be compared with, not the whole-run column.
        on_sample = sim.get("on_calibration_sample")
        if on_sample:
            _n2 = on_sample.get("n_persons_arm2")
            if on_sample.get("persons_match"):
                _note = "témoin de taille, mêmes personnes que la calibration"
            else:
                # Do not suggest a population equality that does not exist:
                # the control over-covers strand 2 with agents all of whose decisions
                # came from the LLM cache, hence are absent from the exchange log.
                _note = (f"témoin de taille — même règle de tirage, mais {on_sample['n_persons']} "
                         f"personnes contre {_n2} au volet 2 (celui-ci ne voit que les "
                         f"décisions passées par un appel LLM réel)")
            arms_out.append({
                "label": "Sim. (éch. V2)",
                "basis": f'jeu commun — {on_sample["n_persons"]} personnes tirées par la '
                         f'règle gelée du volet 2',
                "note": _note,
                "cells": cells_from_dims(on_sample.get("dims") or {})})
    else:
        arms_out.append({"label": "Simulation", "cells": [{"value": None}] * len(dims)})

    # Strand 2: seed and best prompt. TWO possible substrates, and the
    # difference is not cosmetic.
    #
    # - the common set (action A3): the personas of the pinned run, the very ones of
    #   strand 1 — the columns are then commensurable;
    # - failing that, the frozen personas of the calibration engine, i.e. a
    #   subset of an EARLIER run. Comparing this figure with that of strand 1
    #   amounts to comparing two measurements made on two populations.
    #
    # We therefore always prefer the common set when it exists, and each column
    # declares its substrate (``basis``) so that the page cannot hide it.
    cal = payload["arms"]["calibration"]
    common = cal.get("common_set") or {}
    store = None
    if cal.get("status") == "ok":
        candidates = [s for s in cal["stores"] if s.get("best")]
        store = max(candidates, key=lambda s: s["kept"]) if candidates else None

    if common.get("available") and common.get("seed") and common.get("leaf"):
        for entry in (common["seed"], common["leaf"]):
            arms_out.append({
                "label": "Calib. graine" if entry["role"] == "seed" else "Calib. meilleur",
                "basis": "jeu commun",
                "note": f'{entry["short"]} · {entry.get("regime") or "régime inconnu"}',
                "cells": cells_from_dims(entry.get("dims") or {})})
    elif store:
        arms_out.append({"label": "Calib. graine", "basis": "personas gelés",
                         "note": f'{(store.get("seed") or {}).get("short", "?")} · '
                                 f'{(store.get("seed") or {}).get("regime", "?")}',
                         "cells": cells_from_dims((store.get("seed") or {}).get("dims", {}))})
        arms_out.append({"label": "Calib. meilleur", "basis": "personas gelés",
                         "note": f'{store["best"]["short"]} · {store["best"]["regime"]}',
                         "cells": cells_from_dims(store["best"]["dims"])})
    else:
        arms_out.append({"label": "Calibration", "basis": None,
                         "cells": [{"value": None}] * len(dims)})
    # Strand 3: two readings, like strand 1. The gap between them is structural and
    # measured at training (L1 of modal shares 0.021 in mass versus 0.087 in chosen
    # mode, bike recall 0.128): showing only the first would flatter the model,
    # showing only the second would condemn it. The declared substrate says on how many
    # decisions of the common set the measurement actually bears.
    model = payload["arms"].get("model") or {}
    preds = model.get("predictions") or {}
    if preds.get("available"):
        summary = preds.get("summary") or {}
        scored = summary.get("n_scored")
        total = summary.get("n_moves")
        basis = "jeu commun"
        if scored is not None and total and scored < total:
            basis = f"jeu commun — {scored}/{total} décisions prédictibles"
        note = "renormalisé sur l'offre OTP"
        for name, label in (("attendu", "Modèle"), ("elu", "Modèle (élu)")):
            variant = (preds.get("variants") or {}).get(name)
            if variant is None:
                continue
            arms_out.append({"label": label, "basis": basis, "note": note,
                             "cells": cells_from_dims(variant.get("dims") or {})})
    else:
        arms_out.append({"label": "Modèle", "basis": None,
                         "cells": [{"value": None}] * len(dims)})
    for arm in arms_out:
        arm.setdefault("basis", "jeu commun")
        arm.setdefault("note", "")
    # The held-out set (action A4) is a THIRD substrate: neither the run, nor the
    # training personas, but frozen personas never seen by the loop. It
    # deliberately does not enter the matrix — sticking it there would put a
    # 66-person column next to 881-person columns, i.e. would replay the
    # confusion that action A3 corrected. The matrix merely flags that
    # the figure exists and where to read it.
    generalization = (cal.get("generalization") or {}) if isinstance(cal, dict) else {}
    return {"dims": dims, "arms": arms_out,
            "calibration_basis": ("jeu commun" if common.get("available")
                                  else "personas gelés"),
            "model_available": bool(preds.get("available")),
            "generalization_available": bool(generalization.get("available")),
            "generalization_dataset": generalization.get("dataset"),
            "commensurable": bool(common.get("available"))}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="source manifest (default: sources.yaml)")
    parser.add_argument("--run", help="run used as common set (overrides the manifest)")
    parser.add_argument("--out", help="output HTML path")
    parser.add_argument("--json", dest="json_out", help="output JSON path")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    if args.run:
        manifest.raw.setdefault("common_set", {})["run"] = args.run

    cerema_src = manifest.track("cerema", manifest.get("cerema"),
                                "Référence EMC² 2023 — parts modales cibles")
    if not cerema_src.exists:
        print(f"[erreur] Référence EMC² introuvable : {cerema_src.rel}", file=sys.stderr)
        return 2
    cerema = frames.load_cerema(cerema_src.path)

    weights = manifest.get("score.weights", {})
    formule_score, engine_error = import_formule_score()
    scorer = None
    if formule_score is not None:
        scorer = frames.Scorer(formule_score, weights,
                               manifest.get("score.metric", "emd_jsd"),
                               manifest.get("score.secondary", "l1_composite"))
        engine_note = "Formule de score : scripts/synthesis/formule_score"
    else:
        engine_note = f"Score indisponible — {engine_error}"
        print(f"[avertissement] {engine_error}", file=sys.stderr)

    common, rows = build_common_set(manifest, cerema)
    if common.get("available"):
        expected = frames.simulation_frames(rows)["attendu"]
        common["coverage"] = frames.coverage_matrix(expected, cerema)

    payload: dict[str, Any] = {
        "generated_at": _now(),
        "engine_note": engine_note,
        "score_def": build_score_def(manifest, weights),
        "common_set": common,
        "arms": {
            "simulation": build_simulation(rows, cerema, scorer) if common.get("available")
            else {"status": "missing", "reason": common.get("reason", ""),
                  "expected": common.get("expected", []),
                  "action": "Vérifier common_set.run dans sources.yaml"},
            "calibration": build_calibration(manifest, cerema, scorer),
            "model": build_model(manifest, cerema, scorer),
        },
        "actions": ACTIONS,
    }
    # Size control: strand 1 restricted to the persons of strand 2. Computed here
    # because it crosses both strands — it needs the run's trips AND the
    # sample descriptor written by the strand 2 measurement.
    cal_arm = payload["arms"]["calibration"]
    common_eval = (cal_arm.get("common_set") or {}) if isinstance(cal_arm, dict) else {}
    if common_eval.get("available"):
        on_sample = build_simulation_on_sample(
            rows, cerema, scorer, common_eval.get("sample") or {})
        if on_sample:
            payload["arms"]["simulation"]["on_calibration_sample"] = on_sample
            # Strand 2 must be able to cite its control without fetching it from
            # another strand: the reading of its figures depends on it.
            full = ((payload["arms"]["simulation"].get("variants") or {})
                    .get("attendu", {}).get("scores", {})
                    .get(payload["score_def"]["primary"], {}) or {})
            common_eval["size_control"] = {
                **on_sample,
                "full_composite": full.get("composite"),
                "penalty": (on_sample["composite"] - full["composite"])
                if (on_sample.get("composite") is not None
                    and full.get("composite") is not None) else None,
            }

    payload["synthesis"] = build_synthesis(payload)
    payload["sources"] = [s.to_dict() for s in manifest.sources.values()]

    html_out = Path(args.out or manifest.get("output.html", "docs/synthesis/index.html"))
    if not html_out.is_absolute():
        html_out = REPO_ROOT / html_out
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(render.render(payload), encoding="utf-8")

    # Dedicated pages: the « Détail par sous-catégorie » sub-chapter of strands 1
    # and 3, extracted next to the full page (which keeps it). Written in the
    # same folder as the main HTML so that relative links hold,
    # archive included.
    detail_out = {}
    for arm_key, spec in render.DETAIL_PAGES.items():
        path = html_out.parent / spec["file"]
        path.write_text(render.render_detail(payload, arm_key), encoding="utf-8")
        detail_out[arm_key] = path

    json_out = Path(args.json_out or manifest.get("output.json", "docs/synthesis/data.json"))
    if not json_out.is_absolute():
        json_out = REPO_ROOT / json_out
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")

    # --out may point outside the repository (archive, comparison): show the absolute
    # path rather than fail on an impossible relative_to().
    def display(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    missing = [s for s in payload["sources"] if not s["exists"]]
    print(f"Page written: {display(html_out)}")
    for arm_key, path in detail_out.items():
        print(f"Detail {arm_key:<10}: {display(path)}")
    print(f"Data        : {display(json_out)}")
    print(f"Sources     : {len(payload['sources']) - len(missing)} present, "
          f"{len(missing)} missing")
    for src in missing:
        print(f"  missing : {src['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
