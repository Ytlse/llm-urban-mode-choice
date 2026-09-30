"""The texts of conditions C2, C3 and C4, and their manifest — ticket 059, lot 1.

THE GUARD SPECIFIC TO THIS CHANNEL
----------------------------------
A shock `vecu` is written by us (ticket 079); a press article is **quoted**. The
difference is not one of style: a rewording "that makes the text more readable for the
model" would turn C2 into an undeclared sixth condition, and nobody would know which one was
played. The served text is therefore checked against its fingerprint at every load.

THE ONLY REWRITE ALLOWED IS TRANSLATION
---------------------------------------
The five articles are published in French; the setup runs in English since ticket
074, and a French sentence in an English prompt reintroduces exactly the factor that the
switch removed. Author's decision of 2026-09-21: **we translate, and we say so**.

- Each text exists as TWO files, `*.fr.txt` and `*.txt`, each with its fingerprint.
- The entry served to the agent carries the mention "Translated from French". It is not a
  repository comment: it is information the agent has, and the setup has no reason to hide
  it.
- The manifest names WHO translated and WHEN. An anonymous translation cannot be checked.

WHAT THE MANIFEST REFUSES
-------------------------
A paraphrase that contains a mobility word — condition C3 would no longer separate anything. A
control text that contains one — the specificity check would measure something else. A control
whose length deviates by more than 15 % from its article — the pairing is lost, and "adding a
text" stops being what sets them apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.analysis.presse.lexique import mots_de_mobilite_trouves

# The three textual conditions. C1 receives nothing and C5 is tabular: neither one
# has a file.
CONDITIONS: tuple[str, ...] = ("brut", "paraphrase", "temoin")

# Length gap allowed between a control text and the article it pairs, in words. Beyond it, what
# sets C4 apart from C2 is no longer the content but the size of the block added to the context.
ECART_LONGUEUR_MAX = 0.15

# The mention carried by the entry served to the agent, at the head of the translated text.
MENTION_TRADUCTION = "Translated from French"


class RefusDeCorpus(ValueError):
    """The corpus does not load. The message gives the reason AND the action."""


@dataclass(frozen=True)
class Texte:
    condition: str
    langue: str
    chemin: Path
    contenu: str
    empreinte: str
    mots: int


@dataclass(frozen=True)
class ArticleDuCorpus:
    article_id: str
    source_html: str
    empreinte_source: str
    textes: dict[tuple[str, str], Texte]  # (condition, langue) -> Texte
    # The mobility words the paraphrase is allowed to keep, per language: those that name
    # the very object of the event. An article on bike-sharing talks about bikes.
    exemptions: dict[str, tuple[str, ...]]

    def texte(self, condition: str, langue: str = "en") -> Texte:
        cle = (condition, langue)
        if cle not in self.textes:
            raise KeyError(f"{self.article_id}: no {condition!r} text in {langue!r}")
        return self.textes[cle]

    def entree_pour_agent(self) -> str:
        """The English text, preceded by the translation mention.

        This is what the `information` channel (lot 4) serves to the agent, and it is on this
        string that the fingerprint guard is checked on the run side.
        """
        return f"[{MENTION_TRADUCTION}] {self.texte('brut', 'en').contenu}"


@dataclass(frozen=True)
class Corpus:
    racine: Path
    articles: dict[str, ArticleDuCorpus]
    traduction_par: str
    traduction_le: str
    empreinte_manifeste: str


def _charger_texte(
    racine: Path, article_id: str, condition: str, langue: str, declare: dict,
    temoin_commun: str | None = None,
) -> Texte:
    nom = f"{condition}.txt" if langue == "en" else f"{condition}.{langue}.txt"
    # A control text SHARED by the five articles lives in its own folder. Author's decision of
    # 2026-09-21: one text rather than five. What C4 measures — the effect of adding a text,
    # whatever it is — does not need one control per article, and a single control makes the
    # condition COMPARABLE from one article to the next instead of depending on five choices.
    dossier = temoin_commun if (condition == "temoin" and temoin_commun) else article_id
    chemin = racine / dossier / nom
    if not chemin.is_file():
        raise RefusDeCorpus(
            f"missing text: {chemin} → each article carries {len(CONDITIONS)} conditions in "
            f"two languages, i.e. six files (ticket 059, lot 1)"
        )
    brut = chemin.read_bytes()
    empreinte = hashlib.sha256(brut).hexdigest()
    attendue = str(declare.get("sha256") or "")
    if attendue and empreinte != attendue:
        raise RefusDeCorpus(
            f"{chemin}: fingerprint {empreinte[:12]} instead of {attendue[:12]} declared in the "
            f"manifest → the text changed after the freeze. An article is QUOTED, never rewritten: "
            f"restore the file, or refreeze the corpus and say so"
        )
    contenu = brut.decode("utf-8").strip()
    if not contenu:
        raise RefusDeCorpus(f"{chemin}: empty text")
    return Texte(condition, langue, chemin, contenu, empreinte, len(contenu.split()))


def charger_corpus(chemin_manifeste: str | Path) -> Corpus:
    chemin_manifeste = Path(chemin_manifeste)
    if not chemin_manifeste.is_file():
        raise RefusDeCorpus(
            f"corpus manifest not found: {chemin_manifeste} → produce it with "
            f"`scripts/data/presse/extraire_textes.py` (ticket 059, lot 1)"
        )
    octets_manifeste = chemin_manifeste.read_bytes()
    d = yaml.safe_load(octets_manifeste.decode("utf-8")) or {}
    racine = chemin_manifeste.parent
    temoin_commun = d.get("temoin_commun")

    traduction = d.get("traduction") or {}
    par, le = str(traduction.get("par") or ""), str(traduction.get("le") or "")
    if not par or not le:
        raise RefusDeCorpus(
            f"{chemin_manifeste}: `traduction.par` and `traduction.le` are mandatory → an "
            f"anonymous translation cannot be checked, and the five texts are translated from French"
        )

    articles: dict[str, ArticleDuCorpus] = {}
    for article_id, contenu in (d.get("articles") or {}).items():
        contenu = contenu or {}
        textes: dict[tuple[str, str], Texte] = {}
        for condition in CONDITIONS:
            declare_cond = contenu.get(condition) or {}
            for langue in ("fr", "en"):
                declare = (declare_cond.get(langue) or {}) if isinstance(declare_cond, dict) else {}
                if condition == "temoin" and temoin_commun:
                    declare = ((d.get("temoin") or {}).get(langue)) or {}
                textes[(condition, langue)] = _charger_texte(
                    racine, article_id, condition, langue, declare, temoin_commun
                )

        # The words the SUBJECT imposes, and the guard that prevents convenience exemptions:
        # a word is exempted only if it is in the raw text. Without it, exempting "voiture"
        # on the bedbug article would be enough for the paraphrase to suggest the shift.
        exemptions: dict[str, tuple[str, ...]] = {}
        declarees = contenu.get("c3_mots_autorises") or {}
        for langue in ("fr", "en"):
            mots = tuple((declarees.get(langue) or {}).get("mots") or ())
            texte_brut = textes[("brut", langue)]
            dans_le_brut = set(mots_de_mobilite_trouves(texte_brut.contenu, langue))
            absents = [mot for mot in mots if mot not in dans_le_brut]
            if absents:
                raise RefusDeCorpus(
                    f"{article_id} ({langue}): {absents} exempted but absent from the raw text → "
                    f"a word is not exempted as a precaution, only the one the subject "
                    f"imposes. Remove them from `c3_mots_autorises`"
                )
            if mots and not str((declarees.get(langue) or {}).get("motif") or "").strip():
                raise RefusDeCorpus(
                    f"{article_id} ({langue}): `c3_mots_autorises` without a reason → an exemption "
                    f"without a reason cannot be argued, and that is how condition C3 empties out"
                )
            exemptions[langue] = mots

        # C3 and C4 contain no mobility word outside exemptions, in BOTH languages.
        # Translating is no opportunity to reintroduce a word the rewrite had removed.
        for condition in ("paraphrase", "temoin"):
            for langue in ("fr", "en"):
                t = textes[(condition, langue)]
                # The control has NO exemption: it does not talk about the event, so nothing
                # in its subject imposes a mobility word.
                admis = exemptions[langue] if condition == "paraphrase" else ()
                trouves = mots_de_mobilite_trouves(t.contenu, langue, admis)
                if trouves:
                    raise RefusDeCorpus(
                        f"{t.chemin}: mobility words present {list(trouves)} → condition "
                        f"{'C3' if condition == 'paraphrase' else 'C4'} separates nothing. "
                        f"Rewrite the passage, or remove the word from "
                        f"`scripts/analysis/presse/lexique.py` saying why"
                    )

        # Length pairing of the control, on the ENGLISH version: that is the one that enters
        # the context, so its size is what counts.
        brut_en, temoin_en = textes[("brut", "en")], textes[("temoin", "en")]
        if brut_en.mots:
            ecart = abs(temoin_en.mots - brut_en.mots) / brut_en.mots
            if ecart > ECART_LONGUEUR_MAX:
                raise RefusDeCorpus(
                    f"{article_id}: the control has {temoin_en.mots} words against "
                    f"{brut_en.mots} for the article, a {ecart:.0%} gap (max "
                    f"{ECART_LONGUEUR_MAX:.0%}) → what sets C4 apart from C2 must not be the "
                    f"size of the block added to the context"
                )

        articles[article_id] = ArticleDuCorpus(
            article_id=article_id,
            source_html=str(contenu.get("source", {}).get("fichier", "")),
            empreinte_source=str(contenu.get("source", {}).get("sha256", "")),
            textes=textes,
            exemptions=exemptions,
        )

    if not articles:
        raise RefusDeCorpus(f"{chemin_manifeste}: no article declared")

    return Corpus(
        racine=racine,
        articles=articles,
        traduction_par=par,
        traduction_le=le,
        empreinte_manifeste=hashlib.sha256(octets_manifeste).hexdigest(),
    )
