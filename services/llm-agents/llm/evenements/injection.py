"""The two injection points — ticket 100, lot 1.

Only one delivered: `arrivee`.

`a_l_arrivee` applies AFTER the decision. The agent chose seeing the nominal offer, then takes
the blow. This is the ENDURED regime, and it is what makes the event day silent on the choice:
all the effect observed on the following days is attributable to the memory, and nothing else.

`au_reveil` will apply BEFORE the first decision of the day. That is lot 2, and the difference
between the two is not an implementation detail: it is the heart of the contrast between the
two regimes that chapter 7 measures.

WHY TWO GESTURES, AND NOT ONE
-----------------------------
The ticket announces "a single function drops the entry". Reading the controller, this is not
what happens and cannot be: at arrival, the text is **attached** to the observation GAMA has
just returned, and the severity carries the SUM of the measured and injected delays. Writing
a separate entry would change the number of entries, their timestamps and the severity of each —
the golden test would fail, and the figures of § 7.2 would no longer hold.

What is common to both injection points, and this is the real gain of the ticket, is therefore
not the writing: it is the **qualification**. Severity — the agent's estimate, and only that since
D7 —, valence, origin, axes, strength, service duration: a single route, whatever the channel.
"""

from __future__ import annotations

from llm.evenements.declaration import EvenementApplique

# Prefix of the text attached to the arrival observation. Unchanged since ticket 079: it appears
# in the memory of every agent of the runs already archived, and moving it would make the two
# corpora incomparable word for word.
PREFIXE_VECU = "[ INCIDENT ]"

# Prefix of the `lu` channel, set here so that both live in the same place. Used in lot 2.
PREFIXE_LU = "[ PRESSE ]"

# Ticket 111 — what a household member HEARD from the reader. Rendered word for word, it is what
# lets the post-run check be exact, without heuristics.
PREFIXE_FOYER = "[ FOYER ]"


def a_l_arrivee(registre, person_id: str, mode: str | None, timestamp: int):
    """The event applicable to this arrival, or `None`.

    Does nothing but consult the registry: the decision to write belongs to the caller, who
    alone knows what the observation contains besides.
    """
    if registre is None:
        return None
    return registre.applique(person_id, mode, timestamp)


def joindre(texte_observation: str, applique: EvenementApplique) -> str:
    """The event text, APPENDED to the observation — never substituted.

    The agent must keep what the simulation measured, and add to it what it experienced.
    Replacing one with the other would lose the time, mode and destination of its own trip.
    """
    prefixe = PREFIXE_VECU if applique.canal == "vecu" else PREFIXE_LU
    return f"{texte_observation}\n{prefixe} {applique.texte}"


def au_reveil(registre, population, timestamp: int) -> list:
    """What agents receive this morning, BEFORE their first decision.

    Returns a list of `(person_id, EvenementApplique)`, empty on other days and empty when
    the declared event does not belong to this moment.

    ⚠ **This is not the injection point of the shock**, and the difference is the heart of the
    regime: a shock applies at arrival, after the decision; an article is known **before**
    deciding. It is the only contrast chapter 7 measures, and it exists only if the two
    injection points stay in their place.
    """
    if registre is None:
        return []
    return registre.dus_au_reveil(timestamp, population)


def entree_de_lecture(applique) -> str:
    """The text as it enters short-term memory, prefixed.

    Unlike the `arrivee` injection point, there is NOTHING to attach the text to: no
    observation has taken place, the agent is still asleep. The entry is therefore
    standalone, and this is the only writing difference between the two injection points.
    """
    prefixe = PREFIXE_LU if applique.canal == "lu" else PREFIXE_VECU
    if applique.canal == "lu":
        return ligne_de_lecture(applique.texte)
    return f"{prefixe} {applique.texte}"


def ligne_de_lecture(texte: str) -> str:
    """The reading entry, built from the text alone — ticket 111.

    The line served to the prompt during the service days and the long-term memory entry are
    the SAME string. This is what lets the block avoid serving it twice when the agent judged
    the article serious, and lets the post-run check find it word for word.
    """
    # Without « this morning » since 2026-09-25: the line is served for five days and stays in
    # long-term memory, and "this morning" became false there from the next day on.
    return f"{PREFIXE_LU} I read in the paper: « {texte} »"


def ligne_de_foyer(message: str, prenom_lecteur: str, mineur: bool) -> str:
    """What an informed member sees, and keeps in memory — ticket 111.

    For a minor, it is the PARENTS' DECISION (D5): in reality, they decide a child's trip,
    and a simplified version of the article would not make it change. The other parent's
    name is not given: the population does not carry parentage.
    """
    if mineur:
        return f"{PREFIXE_FOYER} My parents decided: « {message} »"
    return f"{PREFIXE_FOYER} {prenom_lecteur or 'Someone at home'} told me: « {message} »"
