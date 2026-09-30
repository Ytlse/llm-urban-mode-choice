#!/usr/bin/env python3
"""Extracts the text of the archived articles and prepares the corpus — ticket 059, lot 1.

WHY A SCRIPT RATHER THAN A COPY-PASTE
-------------------------------------
The text of condition C2 is **quoted**, not written. An excerpt copied by hand cannot be
replayed: nobody can check that it really comes from the archive, nor reproduce the
selection. Here, the selection rule is written down — title, standfirst, then the first
paragraphs of the body up to a word cap — and two runs return the same text.

WHAT THE SCRIPT DOES NOT DO
---------------------------
It neither translates nor paraphrases: both operations require a model, they are
declared, and they live in `traduire` and `paraphraser` — two separate passes of which the
manifest keeps track of who did them and when. A script that translated along the way
would make the translation invisible, and the article must say that it took place.

Nor does it choose the control texts of C4: the corpus of thirty articles was
assembled for its link with mobility, none of its members can serve as a neutral
control. See `--verifier`, which flags it rather than hiding it.

USAGE
-----
    services/llm-agents/.venv/bin/python -m scripts.data.presse.extraire_textes extraire
    services/llm-agents/.venv/bin/python -m scripts.data.presse.extraire_textes sceller --par "…"
    services/llm-agents/.venv/bin/python -m scripts.data.presse.extraire_textes verifier
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[3]
SOURCES = RACINE / "data" / "presse"
HTML = SOURCES / "articles_html"
CORPUS = SOURCES / "articles_txt"
MANIFESTE = CORPUS / "MANIFEST.yaml"

# The five selected articles, and the archived file of each. The identifier is the one of the
# sign grid: the two files are read together, and an identifier that diverged
# would score a prediction against another article.
ARTICLES: dict[str, str] = {
    "a09_vent_autan": "09_vent_autan_fermeture_parcs.html",
    "a13_punaises_metro": "13_psychose_punaises_metro.html",
    "a07_greve_eboueurs": "07_greve_eboueurs_hypercentre.html",
    "a18_la_machine": "18_minotaure_la_machine.html",
    "a25_velotoulouse": "25_velotoulouse_electrique.html",
}

# The control of condition C4, common to the five articles. It can NOT come from the corpus of
# thirty: that one was assembled article by article FOR its link with mobility, and a
# control must have none. The one chosen by the author on 2026-09-21 is a natural-science
# news item — a feline species described in Bolivia.
TEMOIN_COMMUN = "_temoin_commun"

# Word cap of the excerpt. A whole article would drown the rest of the context: chapter 6
# measures a prompt of ~2,100 tokens, and pouring a thousand press words into it would change
# the measured quantity instead of illuminating it. The cap cuts at the end of a paragraph; when a
# single paragraph exceeds it — the layout of some newspapers makes only one — it
# cuts at the end of a sentence. The excerpt stays continuous in both cases: it is a quotation,
# not a montage.
MOTS_MAX = 220

# What is not part of the article: photo credits, share buttons, breadcrumbs, ad-sales
# notices. These lines pass through `get_text` extraction and have no place in a quoted
# excerpt. Comparison made without accents or case.
LIGNES_CHROME: tuple[str, ...] = (
    "ajouter aux sources", "ajouter comme source", "sur facebook", "sur whatsapp",
    "copier le lien", "nouvelle fenetre", "temps de lecture", "credit", "article redige par",
    "journaliste", "publie le", "publie :", "mis a jour", "modifie :", "l'essentiel",
    "partager", "abonnez-vous", "s'abonner", "newsletter", "commentaires", "voir les",
    "(c)", "©", "ddm -", "photo",
    # Recommendation boxes: they cut through the body of the article and talk about something
    # else. Left in place, they would make the agent read a headline unrelated to
    # the event, and C2 would measure two texts instead of one.
    "a lire aussi", "lire aussi", "sur le meme sujet", "a voir aussi", "notre dossier",
    "cet article", "en savoir plus",
    # Calls to the reader: comment threads, alerts, polls. They address
    # whoever reads the site, not whoever reads the article, and an agent asked whether it wants
    # to follow a discussion reads something other than the event it is claimed to be served.
    "vous souhaitez", "souhaitez-vous", "voulez-vous", "donnez votre avis", "reagir",
    "recevez", "inscrivez-vous", "connectez-vous", "identifiez-vous",
)


# Class or identifier fragments that designate a zone outside the article. The comparison is
# an inclusion, case-insensitive: sites combine these words with their own prefixes
# (`ldm-comments-list`, `block-related-articles`).
_FRAGMENTS_HORS_ARTICLE: tuple[str, ...] = (
    "comment", "commentaire", "reaction", "forum", "disqus", "related", "recommend",
    "sidebar", "advert", "publicite", "sponsor", "partner", "newsletter", "social",
    "share", "partage", "breadcrumb", "tag-list",
)

# Share of the page text beyond which a block is the BODY, whatever its class.
# La Dépêche wraps its articles in a container named `paywall`: the list above, taken
# literally, removed the whole article and the extraction returned an empty file. A word list
# cannot foresee each site's conventions; a guard on size can.
PART_CORPS_MIN = 0.40


def _ZONES_HORS_ARTICLE(valeur) -> bool:  # noqa: N802 — prédicat passé à BeautifulSoup
    """True if a class or an identifier designates a zone that is not the body."""
    if not valeur:
        return False
    plat = " ".join(valeur) if isinstance(valeur, list) else str(valeur)
    plat = _sans_accents(plat).lower()
    return any(f in plat for f in _FRAGMENTS_HORS_ARTICLE)


def _sans_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _est_chrome(ligne: str) -> bool:
    plat = _sans_accents(ligne).lower().strip()
    # A teaser truncated by the layout — property ad, sponsored article, "read
    # more" — is recognised by its ellipsis in square brackets. It does not belong
    # to the article body, and a flat advert in a press excerpt
    # would add to the agent's context a topic the protocol has not declared.
    if plat.endswith("[...]") or plat.endswith("[…]"):
        return True
    if len(plat) < 25:
        # A short line is almost always a navigation label or a caption.
        # The body of a press article does not have twenty-five-character sentences.
        return True
    return any(marqueur in plat for marqueur in LIGNES_CHROME)


def extraire_texte(chemin_html: Path) -> tuple[str, str]:
    """(title, body) — the title as it appears, and the first paragraphs of the body."""
    from bs4 import BeautifulSoup  # imported here: the `verifier` script does not need it

    soup = BeautifulSoup(chemin_html.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "figure", "form"]):
        t.decompose()

    # The zones that are not the article, removed by their STRUCTURE and not by their words.
    # Press sites put comments, recommendations and ads in the document
    # tree, often under the same tag as the body: `La Dépêche` thus made the
    # vent d'Autan excerpt end on a reader comment recommending a weather site.
    # Filtering on the text does not work — a comment looks like an article sentence.
    total = len(soup.get_text(" ", strip=True)) or 1
    for attribut in ("class", "id"):
        for t in soup.find_all(attrs={attribut: _ZONES_HORS_ARTICLE}):
            if len(t.get_text(" ", strip=True)) / total < PART_CORPS_MIN:
                t.decompose()

    h1 = soup.find("h1")
    titre = re.sub(r"\s+", " ", h1.get_text(" ", strip=True)) if h1 else ""

    zone = soup.find("article") or soup.find("main") or soup.body
    if zone is None:
        return titre, ""

    paragraphes: list[str] = []
    mots = 0
    for p in zone.find_all("p"):
        ligne = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        if not ligne or _est_chrome(ligne) or ligne == titre:
            continue
        if ligne in paragraphes:
            continue
        n = len(ligne.split())
        if mots + n > MOTS_MAX:
            if paragraphes:
                break
            ligne = _couper_a_la_phrase(ligne, MOTS_MAX)
            n = len(ligne.split())
        paragraphes.append(ligne)
        mots += n
    return titre, "\n\n".join(paragraphes)


def _couper_a_la_phrase(texte: str, mots_max: int) -> str:
    """Truncates at the last complete sentence under `mots_max`, or at the first if it exceeds.

    Cutting in the middle of a sentence would produce an unreadable excerpt, and an agent
    served an unfinished sentence is no longer in the conditions we claim to measure.
    """
    phrases = re.split(r"(?<=[.!?…])\s+", texte)
    garde: list[str] = []
    mots = 0
    for phrase in phrases:
        n = len(phrase.split())
        if garde and mots + n > mots_max:
            break
        garde.append(phrase)
        mots += n
    return " ".join(garde)


def _ecrire(chemin: Path, contenu: str) -> str:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    donnees = (contenu.strip() + "\n").encode("utf-8")
    chemin.write_bytes(donnees)
    return hashlib.sha256(donnees).hexdigest()


def commande_extraire() -> int:
    """Writes `brut.fr.txt` for the five articles and initialises the manifest."""
    manifeste: dict = {
        "corpus": "presse locale toulousaine — ticket 059, lot 1",
        "extrait_le": date.today().isoformat(),
        "regle_de_selection": (
            f"titre, puis paragraphes du corps dans l'ordre de parution, chrome retiré, "
            f"coupe à la fin d'un paragraphe sous {MOTS_MAX} mots"
        ),
        "traduction": {"par": None, "le": None},
        "articles": {},
    }
    if MANIFESTE.is_file():
        ancien = yaml.safe_load(MANIFESTE.read_text(encoding="utf-8")) or {}
        manifeste["traduction"] = ancien.get("traduction", manifeste["traduction"])
        manifeste["articles"] = ancien.get("articles") or {}

    for article_id, nom in ARTICLES.items():
        source = HTML / nom
        if not source.is_file():
            print(f"  ABSENT  {article_id} : {source}", file=sys.stderr)
            return 2
        titre, corps = extraire_texte(source)
        if not corps:
            print(f"  VIDE    {article_id} : aucun paragraphe retenu dans {nom}", file=sys.stderr)
            return 2
        if not titre:
            # Not blocking: some archives have no usable <h1>. But the excerpt
            # then loses the headline the reader sees first, and this is made known.
            print(f"  (no title) {article_id}: no <h1> in {nom}", file=sys.stderr)
        texte = f"{titre}\n\n{corps}" if titre else corps
        sha = _ecrire(CORPUS / article_id / "brut.fr.txt", texte)
        # Keys already present are KEPT — first among them
        # `c3_mots_autorises`, which is declared by hand and justified. A re-extraction must
        # not erase an editorial decision on the grounds that it recomputes a text.
        entree = manifeste.setdefault("articles", {}).setdefault(article_id, {})
        entree["source"] = {
            "fichier": f"articles_html/{nom}",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        entree.setdefault("brut", {})["fr"] = {"sha256": sha, "mots": len(texte.split())}
        print(f"  written {article_id}/brut.fr.txt — {len(texte.split())} words")

    MANIFESTE.write_text(
        yaml.safe_dump(manifeste, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"\nmanifest: {MANIFESTE}")
    return 0


def commande_sceller(par: str, le: str) -> int:
    """Recomputes the fingerprint of each present text and records the translation trace.

    To run once the corpus is complete, and never again afterwards: from then on, any
    divergence between a file and its fingerprint makes the loading REFUSE. That is what
    distinguishes a quoted text from a text retouched between two campaigns.
    """
    if not MANIFESTE.is_file():
        print("manifest missing — run `extraire` first", file=sys.stderr)
        return 2
    manifeste = yaml.safe_load(MANIFESTE.read_text(encoding="utf-8")) or {}
    manifeste["traduction"] = {"par": par, "le": le}
    manifeste["scelle_le"] = date.today().isoformat()

    scelles = 0
    for article_id in ARTICLES:
        entree = manifeste.setdefault("articles", {}).setdefault(article_id, {})
        for condition in ("brut", "paraphrase"):
            par_langue = entree.setdefault(condition, {})
            for langue in ("fr", "en"):
                nom = f"{condition}.txt" if langue == "en" else f"{condition}.{langue}.txt"
                chemin = CORPUS / article_id / nom
                if not chemin.is_file():
                    par_langue.pop(langue, None)
                    continue
                donnees = chemin.read_bytes()
                par_langue[langue] = {
                    "sha256": hashlib.sha256(donnees).hexdigest(),
                    "mots": len(donnees.decode("utf-8").split()),
                }
                scelles += 1
            if not par_langue:
                entree.pop(condition, None)

    MANIFESTE.write_text(
        yaml.safe_dump(manifeste, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"{scelles} text(s) sealed — translation by {par}, on {le}")
    return 0


def commande_verifier() -> int:
    """Says what is missing from the corpus, without writing anything.

    The control is COMMON to the five articles since 2026-09-21: it lives in `_temoin_commun/`
    and not in each article folder. What C4 measures — the effect of adding any text
    whatsoever — requires only one, and a single control makes the condition comparable from one
    article to another instead of making it depend on five choices.
    """
    attendus = [(a, c, lg) for a in ARTICLES for c in ("brut", "paraphrase") for lg in ("fr", "en")]
    attendus += [(TEMOIN_COMMUN, "temoin", lg) for lg in ("fr", "en")]
    manquants = [
        (a, c, lg)
        for a, c, lg in attendus
        if not (CORPUS / a / (f"{c}.txt" if lg == "en" else f"{c}.{lg}.txt")).is_file()
    ]
    print(f"{len(attendus) - len(manquants)}/{len(attendus)} files present")
    for a, c, lg in manquants:
        print(f"  missing {a}/{c}.{'' if lg == 'en' else lg + '.'}txt")
    return 1 if manquants else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("commande", choices=("extraire", "sceller", "verifier"))
    p.add_argument("--par", help="who translated — mandatory for `sceller`")
    p.add_argument("--le", default=date.today().isoformat(), help="date of the translation")
    args = p.parse_args(argv)
    if args.commande == "sceller":
        if not args.par:
            p.error("`sceller` requires --par: an anonymous translation cannot be verified")
        return commande_sceller(args.par, args.le)
    return {"extraire": commande_extraire, "verifier": commande_verifier}[args.commande]()


if __name__ == "__main__":
    raise SystemExit(main())
