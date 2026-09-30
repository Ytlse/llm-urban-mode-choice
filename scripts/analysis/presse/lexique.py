"""The mobility lexicon — what makes condition C3 checkable (ticket 059, lot 1).

WHY THIS FILE EXISTS
--------------------
Condition C3 serves to refute a precise objection: *"the model obeys a lexical
instruction, it does not reason"*. It gives the agent the same fact, rewritten **without any
mention of mode or roadway**. If a single mobility word survives the rewrite, the condition
no longer separates anything — and the objection comes back intact.

A paraphrase that DECLARES itself neutral proves nothing. This one is CHECKED, against a list that
lives in a file rather than in a test: it can be argued, it can grow, and every addition
shows in the history.

The same ban applies to the C4 control text. A control that talked about traffic would
not be a control, and the specificity check would measure something other than what it announces.

WHAT THE PARAPHRASE MUST NOT REMOVE — author's decision, 2026-09-21
-------------------------------------------------------------------
The first version banned EVERY mobility word, including the subject of the article. It
thus told the launch of bike-sharing without naming the bike, which is absurd: the objection
to refute is not "the text talks about transport", it is "the text TELLS the agent which mode
to take". An article on bike-sharing talks about bikes; what it must not do is name
the car one gives up or the bus one abandons.

Each article therefore declares its EXEMPTED words, those that name the very object of the event,
with their reason. The rest of the lexicon — and in particular the modes towards which a shift
would be possible — stays banned: that is where, and only where, lexical following would hide.

A guard comes with it, without which the exemption would be an open door: a word can be exempted
only if it appears in the article's RAW TEXT. A word is not exempted as a precaution, only
the one the subject imposes. The check is in `corpus.py`.

TWO LANGUAGES, ON PURPOSE
-------------------------
The articles are published in French, the setup runs in English (ticket 074). Both
versions of a paraphrase are therefore checked against both lists: translating is no
opportunity to reintroduce a word the rewrite had removed.
"""

from __future__ import annotations

import re
import unicodedata

# Modes, infrastructure, travel actions. A word enters here as soon as it NAMES a means of
# travel or the place where one travels — not as soon as it evokes a trip. "Sortir"
# and "aller" stay allowed: banning them would make any paraphrase impossible.
MOTS_FR: tuple[str, ...] = (
    # modes
    "metro", "rame", "bus", "autobus", "tram", "tramway", "train", "ter", "navette",
    "velo", "bicyclette", "cyclable", "cycliste", "voiture", "auto", "automobile",
    "vehicule", "scooter", "moto", "deux-roues", "taxi", "vtc", "covoiturage",
    "trottinette", "marche", "marcher", "pieton", "pietonne", "pietonnise",
    # infrastructure and roadway
    "trottoir", "chaussee", "rue", "avenue", "boulevard", "route", "rocade", "peripherique",
    "voie", "carrefour", "pont", "passerelle", "tunnel", "station", "arret", "quai", "gare",
    "parking", "stationnement", "piste",
    # traffic and operations
    "circulation", "trafic", "embouteillage", "bouchon", "transport", "transports",
    "deplacement", "trajet", "itineraire", "conduire", "conducteur", "rouler", "pedaler",
    "tisseo", "ligne",
)

# ⚠ "ligne" and "line" are the two most polysemous words of these lists: "ligne A"
# is a mode, "ligne de conduite" is not. They stay in the lexicon — the Toulouse network
# is named by its lines, and removing them would let "la ligne B est
# perturbée" through in a paraphrase meant to say nothing about it. The refusal names them, and a
# legitimately refused paraphrase is rewritten in a minute.

MOTS_EN: tuple[str, ...] = (
    # modes
    "metro", "subway", "underground", "bus", "coach", "tram", "streetcar", "train", "rail",
    "shuttle", "bike", "bicycle", "cycling", "cyclist", "car", "vehicle", "scooter",
    "motorbike", "motorcycle", "taxi", "cab", "rideshare", "carpool", "walk", "walking",
    "pedestrian", "pedestrianise", "pedestrianize", "foot",
    # infrastructure
    "pavement", "sidewalk", "roadway", "street", "avenue", "boulevard", "road", "ring road",
    "bypass", "lane", "junction", "bridge", "footbridge", "tunnel", "station", "stop",
    "platform", "parking", "car park",
    # traffic and operations
    "traffic", "congestion", "gridlock", "jam", "transport", "transit", "commute",
    "commuting", "journey", "trip", "route", "drive", "driver", "driving", "ride", "riding",
    "line",
)


def _sans_accents(texte: str) -> str:
    """The words "chaussée" and "chaussee" are the same for this check."""
    decompose = unicodedata.normalize("NFD", texte)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def mots_de_mobilite_trouves(
    texte: str, langue: str, exemptions: tuple[str, ...] | list[str] = ()
) -> tuple[str, ...]:
    r"""The banned words present in `texte`, in list order.

    The search is on the WHOLE WORD, common inflections included, and nothing more:
    "car" must not fire on "careful", nor "lane" on "planet". An open
    suffix (`\w*`) would produce incomprehensible refusals, and an incomprehensible refusal ends
    up being worked around. Forms that do not follow from a short suffix appear
    spelled out in the list.

    Two-word expressions ("ring road", "car park") are searched as they are;
    the space there accepts several blanks or a hyphen.

    `exemptions` removes words from the search: those naming the subject of the article, which it
    would be absurd to hide. Their legitimacy is checked elsewhere — `corpus.py` requires each
    to appear in the raw text of its article.
    """
    liste = {"fr": MOTS_FR, "en": MOTS_EN}.get(langue)
    if liste is None:
        raise ValueError(f"unknown language for the mobility lexicon: {langue!r} (fr | en)")
    exemptes = {_sans_accents(m).lower() for m in exemptions}
    liste = tuple(m for m in liste if _sans_accents(m).lower() not in exemptes)
    suffixes = {"fr": r"(?:s|es|e|ent|ons|ez)?", "en": r"(?:s|es|ing|ed)?"}[langue]
    plat = _sans_accents(texte).lower()
    trouves: list[str] = []
    for mot in liste:
        parts = [re.escape(p) for p in _sans_accents(mot).lower().split()]
        motif = r"\b" + r"[\s-]+".join(parts) + suffixes + r"\b"
        if re.search(motif, plat):
            trouves.append(mot)
    return tuple(trouves)
