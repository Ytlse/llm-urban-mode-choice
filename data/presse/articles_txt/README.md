# Corpus textuel des cinq articles — ticket 059, lot 1

Les textes que les conditions C2, C3 et C4 du protocole à cinq conditions servent au décideur.
La grille des signes attendus vit à côté, dans
[`../grille_signes.yaml`](../grille_signes.yaml), et les deux se lisent ensemble : les
identifiants d'article y sont les mêmes, et un test le vérifie.

## Ce que chaque répertoire contient

| Fichier | Condition | Contrainte |
|---|---|---|
| `brut.fr.txt` | C2, source | extrait cité de l'article archivé, jamais réécrit |
| `brut.txt` | C2, servi | sa traduction anglaise |
| `paraphrase.fr.txt` · `paraphrase.txt` | C3 | **aucun mot de mobilité**, vérifié dans les deux langues |
| `temoin.fr.txt` · `temoin.txt` | C4 | aucun mot de mobilité, et longueur à ±15 % de l'article |

## Pourquoi les textes sont traduits

Le dispositif fonctionne en anglais depuis le ticket 074 : prompts, gabarits, couche de rendu.
Une phrase française dans un prompt anglais réintroduit exactement le facteur que la bascule a
supprimé, et que le ticket 072 mesure séparément.

La traduction est la **seule** réécriture admise, et elle se déclare de trois façons : les deux
versions sont au dépôt avec leurs empreintes, le manifeste nomme qui a traduit et quand, et
l'entrée servie à l'agent porte la mention `[Translated from French]`. Un texte traduit servi
comme un original serait une sixième condition que personne n'aurait déclarée.

## Pourquoi la paraphrase se vérifie au lieu de se déclarer

La condition C3 réfute une objection précise : *le modèle obéit à une consigne lexicale, il ne
raisonne pas*. Elle ne tient que si le texte ne nomme **aucun** mode ni aucune voirie. Une
paraphrase qui s'annonce neutre ne prouve rien ; celle-ci se mesure contre
[`scripts/analysis/presse/lexique.py`](../../../scripts/analysis/presse/lexique.py), et un
seul mot suffit à la faire refuser au chargement.

Deux mots se sont fait prendre à la première écriture, et ils disent à quoi sert le contrôle :
« mises **en ligne** » en français, « **underground** » en anglais. Aucun des deux n'était une
mention de transport dans l'intention de celui qui écrivait ; tous deux en étaient une pour un
modèle qui lit.

## Ce que C3 retire, et ce qu'elle garde

La première écriture interdisait **tout** mot de mobilité, sujet de l'article compris. Elle
racontait donc le lancement du vélo partagé sans nommer le vélo — absurde, et sans rapport avec ce
que la condition teste. L'objection à réfuter n'est pas « le texte parle de transport », c'est
**« le texte dit à l'agent quel mode prendre »**.

La règle, arrêtée le 2026-09-21 : **le sujet se nomme, les modes de report disparaissent.**

| Article | Exempté | Toujours interdit |
|---|---|---|
| Punaises dans le métro | métro, rame | voiture, vélo, marche |
| VélôToulouse électrique | vélo, station | voiture, bus, marche |
| Vent d'Autan, éboueurs, La Machine | — | tout le lexique |

Chaque exemption se déclare dans `MANIFEST.yaml` avec sa raison, et deux gardes l'encadrent : un
mot ne s'exempte que s'il **figure dans le texte brut** de l'article — on n'exempte pas par
précaution, on exempte ce que le sujet impose — et un test refuse qu'un article exempte un mode
vers lequel son événement pousserait. « Métro » s'exempte sur l'article des punaises parce qu'il
en est le sujet ; « vélo » ne s'y exempte pas, parce qu'il en est le report attendu.

## Le témoin de C4 : un seul, et il ne vient pas du corpus

Les trente articles ont été réunis article par article **pour** leur lien avec la mobilité ;
aucun ne peut servir de témoin. Celui retenu le 2026-09-21 est une dépêche de sciences
naturelles — une espèce de félin décrite en Bolivie — apparié en longueur aux cinq articles, de
0,5 % à 8 % d'écart, et vide de tout mot de mobilité dans les deux langues.

**Un seul témoin pour les cinq**, et non un par article. Ce que C4 mesure — l'effet d'ajouter un
texte, quel qu'il soit — n'en demande pas cinq, et un témoin unique rend la condition comparable
d'un article à l'autre au lieu de la faire dépendre de cinq choix.

⚠ **Il n'est pas de la presse locale**, alors que le protocole l'annonçait ainsi. Une dépêche
internationale et un fait divers toulousain ne sont pas interchangeables : un agent peut traiter
différemment ce qui se passe chez lui et ce qui se passe à 9 000 km. C'est à écrire dans le texte
plutôt qu'à laisser croire.

`python -m scripts.data.presse.extraire_textes verifier` dit l'état exact à tout moment.

## Refaire le corpus

```bash
services/llm-agents/.venv/bin/python -m scripts.data.presse.extraire_textes extraire
# puis, une fois les traductions, paraphrases et témoins écrits :
services/llm-agents/.venv/bin/python -m scripts.data.presse.extraire_textes sceller --par "…"
```

`sceller` se lance une fois. Ensuite, toute divergence entre un fichier et son empreinte fait
refuser le chargement : c'est ce qui distingue un texte cité d'un texte retouché entre deux
campagnes.
