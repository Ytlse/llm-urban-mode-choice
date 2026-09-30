"""What an event text is not allowed to say — ticket 100, lot 1.

Taken over **word for word** from `llm/chocs.py` (ticket 079, rules R7 to R10). Nothing is added,
nothing is relaxed: the lists are calibrated on the repository corpus, the five cases c1 to c5 pass
unchanged, and « I am starting to wonder whether this is worth it » (c1, day 14) stays
accepted — a doubt is not a verdict.

The module lives here, not in the declaration, because the **`lu` channel obeys the same rules**.
A press article that told the agent which mode to take would manufacture the result we claim to
measure, exactly like an experienced shock that concludes in its place. One list for both channels:
two lists would diverge, and the divergence would not show.

The QUOTATION guard — the fingerprint of a quoted text against its manifest — is at the bottom of
this module. It is specific to the `lu` channel: we write the experienced shock, it has no source.

WHAT IS NOT REFUSED, AND MUST NOT BE CONFUSED
---------------------------------------------
**A text may name transport modes.** Metro, bus, bike, car: naming them has never been
refused here, and the author's decision of 2026-09-22 confirms it — an article about a
transport strike that could not say "metro" would be unintelligible, and hiding the word
would prove nothing.

⚠ **Not to be confused with the word list of ticket 059, lot 1.** That one forbids mobility
vocabulary in the **paraphrase** (condition C3) and in the **witness** (C4), so that these two
conditions do not talk about transport at all. It does NOT apply to the raw quoted text (C2),
which is the article itself. Two lists, two objects, two different conditions: the same words
play opposite roles in them.

What this module refuses is something else: a text that tells the AGENT what to do, that
concludes in its place on the reliability of a mode, or that announces what it will do tomorrow.
The boundary is not the vocabulary, it is the DESTINATION of the sentence.
"""

from __future__ import annotations

import re

from loguru import logger

# Markers of an INSTRUCTION disguised as event text. R7.
#
# Why refuse rather than warn: a warning in the middle of a run log alerts nobody (ticket 077
# measured it — two WARNINGs on mode vocabulary drowned in 355,000 lines, and the concept
# mechanism stayed broken for thirty days). And an instruction that gets through does not bias a
# little: it manufactures exactly the result we claim to measure.
MARQUEURS_CONSIGNE: tuple[str, ...] = (
    "you should", "you must", "you'd better", "you ought", "you need to",
    "avoid ", "remember to", "consider ", "try to", "don't ", "do not ",
    "make sure", "next time you", "from now on",
    "tu devrais", "tu dois", "évite ", "evite ", "pense à", "pense a",
    "il faut que tu", "n'oublie pas", "la prochaine fois", "désormais tu",
)

# Markers of a VERDICT — a lasting belief about a mode or a vehicle. R9.
#
# Belief is exactly what the reflection step exists to produce: writing it by hand short-circuits
# the one mechanism the experiment claims to measure. The 19 September run paid for it —
# `vecu`: « I no longer trust this car at all »; concept "learnt" the same evening:
# « My car is unreliable ». The second is only a rewording of the first.
MARQUEURS_VERDICT: tuple[str, ...] = (
    "no longer trust", "cannot trust", "can't trust", "never trust",
    "is unreliable", "are unreliable", "'s unreliable", "was unreliable",
    "is not reliable", "isn't reliable", "not reliable at all",
    "can't rely on", "cannot rely on", "can no longer rely",
    "ne fais plus confiance", "n'ai plus confiance", "plus confiance en",
    "n'est pas fiable", "pas fiable du tout", "ne peux plus compter sur",
)

# Markers of a MODAL INTENTION — what the agent will do tomorrow. R10.
#
# A family distinct from the previous one, and refused by a distinct message: the two are relaxed
# separately the day the author wants to discuss the line. A PAST fact, even a modal one, stays
# accepted — « I had to sort out another way of getting around » (c2, day 13) describes a lived
# day, not a resolution.
MARQUEURS_INTENTION: tuple[str, ...] = (
    "thinking about not using", "thinking of not using", "considering not using",
    "thinking about not taking", "considering not taking",
    "i will not use", "i won't use", "i will stop using", "i'll stop using",
    "i am not going to use", "i'm not going to use", "i will never use",
    "i will never drive", "i will never take", "i will stop driving",
    "from now on i", "i will take the", "i'll take the", "i will use the",
    "alternative transport", "another mode of transport", "switch to the",
    " prendrai", "je vais arrêter", "je vais arreter", "je ne conduirai",
    "désormais je", "desormais je", "un autre mode de transport",
)

# Second-person address, at the head of a sentence or on its own. An experienced shock is told in
# the first person; addressing the agent is talking to it, hence dictating to it.
DEUXIEME_PERSONNE = re.compile(
    r"\b(you|your|yours|tu|toi|ton|ta|tes|vous|votre|vos)\b", re.IGNORECASE
)

# First-person marks. Their ABSENCE is not refused — « Flat tyre on the way, hands covered in
# grease » is a perfectly valid experienced shock without a single « I » — but it is reported.
PREMIERE_PERSONNE = re.compile(
    r"\b(i|i'm|i've|i'd|my|mine|me|je|j'ai|j'|mon|ma|mes|moi)\b", re.IGNORECASE
)


def verifier_texte(
    texte: str,
    evenement_id: str,
    jour: int,
    refus,
    exemptes: dict[str, str] | None = None,
    attendre_premiere_personne: bool = True,
) -> None:
    """R7 to R10: an event text tells, it does not command.

    `refus` is the exception class to raise — passed as a parameter so that this module does
    not depend on `declaration.py`, which depends on it. The import cycle would otherwise be
    unavoidable, and working around it with a local import would hide the real dependency.

    `exemptes` — an instruction marker present in a QUOTED text, lifted with its reason
    (ticket 100, lot 2). The marker list was calibrated on experienced shocks we write in the
    first person; a press article is third-person narrative, and it trips over them for
    reasons that have nothing to do with the reader. Measured on the five articles of
    corpus 059: **a single occurrence**, « City staff must first inspect each site to make
    sure there is no danger », whose subject is the municipal staff.

    The exemption follows the 059 precedent on the forbidden words of the paraphrase: it is
    declared, justified, and it travels with the fingerprint of the text — a modified text
    loses its exemptions along with its validity. It can NEVER lift second-person address:
    it is the only reliable signal that a text talks to the agent, and none of the five
    articles carries it.

    `attendre_premiere_personne` — false for a quoted text. An article is not a first-person
    narrative, and a warning raised at every load stops being read.
    """
    nu = (texte or "").strip()
    if not nu:
        raise refus(
            f"event « {evenement_id} », day {jour}: empty text — an event without "
            f"text leaves no memory, it serves no purpose"
        )
    bas = nu.lower()
    # R9 and R10 BEFORE R7: « from now on I will take the metro » is a first-person intention,
    # not a second-person instruction. The order decides which message is returned, and a message
    # naming the wrong fault sends the author off to fix the wrong word.
    for marqueur in MARQUEURS_VERDICT:
        if marqueur in bas:
            raise refus(
                f"event « {evenement_id} », day {jour}: the text carries a VERDICT on a "
                f"mode — « {marqueur.strip()} ». The belief is what the agent's reflection "
                f"must produce; writing it here amounts to measuring one's own instruction. "
                f"Tell the fact: « the engine stalled twice », never « this car is unreliable »"
            )
    for marqueur in MARQUEURS_INTENTION:
        if marqueur in bas:
            raise refus(
                f"event « {evenement_id} », day {jour}: the text announces a modal "
                f"INTENTION — « {marqueur.strip()} ». What the agent will do tomorrow is the result "
                f"being measured, not an input. A past fact remains allowed: "
                f"« I had to sort out another way of getting around »"
            )
    for marqueur in MARQUEURS_CONSIGNE:
        if marqueur in bas:
            motif = (exemptes or {}).get(marqueur)
            if motif:
                k = bas.find(marqueur)
                logger.info(
                    f"[evenements] « {evenement_id} »: instruction marker "
                    f"« {marqueur.strip()} » EXEMPTED — {motif}. Passage: "
                    f"« …{nu[max(0, k - 60):k + 50].strip()}… »"
                )
                continue
            raise refus(
                f"event « {evenement_id} », day {jour}: the text contains "
                f"« {marqueur.strip()} », which addresses the agent instead of telling what "
                f"happened. A text that dictates a behaviour manufactures the result it "
                f"claims to measure. Tell the fact: « I was stuck for an hour », never "
                f"« avoid the ring road ». If the marker does not address the reader — "
                f"a cited press text may carry it for another reason — exempt it "
                f"by name under `texte.marqueurs_exemptes`, with its reason"
            )
    # Second-person address is NEVER exempted. It is the only reliable signal that a text talks
    # to the agent rather than telling something; none of the five articles of the corpus
    # carries it, and one that did would have to explain itself otherwise.
    if DEUXIEME_PERSONNE.search(nu):
        raise refus(
            f"event « {evenement_id} », day {jour}: the text addresses the agent in the "
            f"second person. A lived experience is told in the first person"
        )
    if attendre_premiere_personne and not PREMIERE_PERSONNE.search(nu):
        logger.warning(
            f"[evenements] « {evenement_id} », day {jour}: the text carries no first-person "
            f"mark — check that it is really an experienced shock and not an outside "
            f"description: « {nu[:70]}… »"
        )


# ── QUOTATION guard (ticket 100, lot 2) ─────────────────────────────────────────────────────
# Specific to the `lu` channel, with no counterpart for the experienced shock: that one is
# written by us, it has no source to betray. A press text is QUOTED, never rewritten — this is
# what makes the claim "real articles from the local press" verifiable.
#
# THREE values, TWO comparisons: the fingerprint declared in the YAML, that of the file on
# disk, and that of the corpus manifest. Comparing the file to the declaration alone would let
# through a correct declaration placed on a file modified together with its manifest;
# comparing to the manifest alone would let through a declaration pointing to another text.

# Roots tried for a relative path, in order. The repository and the container do not see the
# corpus in the same place: `/app` in the latter, the repository root in the former.
def _racines_possibles() -> tuple:
    from pathlib import Path

    ici = Path(__file__).resolve()
    return (
        Path.cwd(),
        ici.parents[4] if len(ici.parents) > 4 else ici.parents[-1],  # repository root
        Path("/app"),
    )


def resoudre_chemin(chemin: str):
    """The designated file, looked up where the repository and the container store it.

    Returns `None` if no candidate exists: the caller turns it into a refusal naming the paths
    tried. A "file not found" error without the list of places looked at sends people
    searching in the wrong place.
    """
    from pathlib import Path

    p = Path(chemin)
    if p.is_absolute():
        return p if p.is_file() else None
    for racine in _racines_possibles():
        candidat = racine / p
        if candidat.is_file():
            return candidat
    return None


def _empreinte_du_manifeste(fichier, evenement_id: str):
    """The fingerprint the corpus manifest gives for this file, or `None`.

    `None` means "this file belongs to no manifest" — a text outside the press corpus, for
    example — and not "the manifest agrees". The distinction matters: without it, a text
    stored elsewhere would pass the guard for want of a contradictor, and the absence of
    verification would look exactly like a successful verification.
    """
    import yaml as _yaml

    manifeste = fichier.parent.parent / "MANIFEST.yaml"
    if not manifeste.is_file():
        return None
    data = _yaml.safe_load(manifeste.read_text("utf-8")) or {}
    article = (data.get("articles") or {}).get(fichier.parent.name)
    if not article:
        return None
    # `brut.txt` = English; `brut.fr.txt` = French. The setup addresses the model in English
    # since ticket 074; the French stays in the repository to prove faithfulness to the
    # source.
    morceaux = fichier.name.split(".")
    variante = morceaux[0]
    langue = morceaux[1] if len(morceaux) > 2 else "en"
    entree = (article.get(variante) or {}).get(langue) or {}
    return entree.get("sha256")


def verifier_empreinte(chemin: str, sha256_declare: str, evenement_id: str, refus):
    """Reads the quoted text and checks its TWO fingerprints. Returns the content.

    Raises `refus` at the slightest disagreement: a text that has moved since the declaration
    makes incomparable two campaigns that believe they played the same article.
    """
    import hashlib

    fichier = resoudre_chemin(chemin)
    if fichier is None:
        essayes = ", ".join(str(r / chemin) for r in _racines_possibles())
        raise refus(
            f"event « {evenement_id} »: text not found — « {chemin} ». Looked in: "
            f"{essayes}. In the `controller` container, the corpus must be mounted "
            f"(infra/docker-compose.yml)"
        )
    brut = fichier.read_bytes()
    empreinte = hashlib.sha256(brut).hexdigest()

    attendue = str(sha256_declare or "").strip().lower()
    if not attendue:
        raise refus(
            f"event « {evenement_id} »: `texte.sha256` missing. A text cited without "
            f"a fingerprint is not cited, it is copied — and two campaigns can no longer "
            f"prove they played the same article. Fingerprint of the current file: "
            f"{empreinte}"
        )
    if empreinte != attendue:
        raise refus(
            f"event « {evenement_id} »: the text has CHANGED since the declaration. "
            f"Declared {attendue[:12]}…, found {empreinte[:12]}… in {fichier}. Either the "
            f"file was edited, or the declaration designates another text; in both "
            f"cases, no campaign played before remains comparable"
        )

    du_manifeste = _empreinte_du_manifeste(fichier, evenement_id)
    if du_manifeste is None:
        logger.warning(
            f"[evenements] « {evenement_id} »: the text « {fichier.name} » belongs to no "
            f"corpus manifest — its fingerprint is checked only against the declaration. "
            f"This is not a successful verification, it is a missing one."
        )
    elif str(du_manifeste).strip().lower() != empreinte:
        raise refus(
            f"event « {evenement_id} »: the text agrees with the declaration but NOT "
            f"with the corpus manifest ({str(du_manifeste)[:12]}… expected, {empreinte[:12]}… "
            f"found). The file and its declaration moved together: that is exactly what "
            f"the second comparison exists to catch"
        )
    else:
        logger.info(
            f"[evenements] « {evenement_id} »: quoted text checked against its declaration AND "
            f"against the corpus manifest — {fichier.name}, fingerprint {empreinte[:12]}…"
        )
    return brut.decode("utf-8")


def familles_directives(texte: str) -> tuple[str, ...]:
    """The guard families a text triggers, WITHOUT refusing — ticket 111.

    Used by the reader's relay to their household. This message is not a stimulus we write: it
    is the reader's cognition, and a parent who says « take the bus today » says what they
    think. Refusing it would rewrite the reader; letting it through silently would make the
    measurement unreadable. The families are therefore TRACED (`directif`), never enforced.

    `adresse` — an instruction marker; `verdict` — a belief about a mode; `intention` —
    what someone will do. Second person alone is not counted: a message addressed to someone
    carries it by nature.
    """
    bas = (texte or "").lower()
    familles = []
    if any(m in bas for m in MARQUEURS_CONSIGNE):
        familles.append("adresse")
    if any(m in bas for m in MARQUEURS_VERDICT):
        familles.append("verdict")
    if any(m in bas for m in MARQUEURS_INTENTION):
        familles.append("intention")
    return tuple(familles)
