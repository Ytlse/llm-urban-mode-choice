"""Did the injected memory reach long-term memory? — ticket 106.

WHY THIS CHECK EXISTS
---------------------
On 2026-09-23, the campaign `e_c3_attribution_861500_v4` injected a metro breakdown into the
short-term memory of 861500 on 27 March at 14:02, judged `grave` (0.75, RETAINED 0.75). The
evening consolidation wrote: *« Today went very smoothly overall, with all my trips adhering
closely to schedule »*, importance 0.10. Of the 123 long-term memory documents of the run,
**none** mentions the metro, the lighting, an announcement or a train. The shock never existed
for the agent, and nothing said so — the run went on for another 2 h 22.

Mechanical cause: the event text is ATTACHED to the arrival observation (ticket 100), and
that arrival said `On time.` — the agent arrived 199 s ahead of its expected schedule,
because `retard_injecte_s` only goes up to the `GamaArrivalsLogger`. The consolidation
summarised a day whose dominant line was "everything went well".

WHAT THIS MODULE DOES, AND WHAT IT DOES NOT DO
----------------------------------------------
It **observes and raises the alarm**. It stops nothing. This is deliberate, and the reason is
measured: of the 15 archived runs carrying an injection, **14 keep the shock and 1 loses it**
(7 %). The defect is therefore rare, whereas the witness is heuristic — a legitimate paraphrase
(« the underground » for « metro ») would produce a false positive. Stopping a run on such a
witness means risking killing a good run to catch one case in fifteen. We first learn its false
alarm rate on real runs; wiring the stop afterwards is one line to change, not a project.

It is the gesture of ticket 105, one notch lower, because the certainty is not the same.

WHY WORDS, AND NOT IMPORTANCE
-----------------------------
The obvious witness — "did the consolidation produce an entry of importance comparable to the
retained importance?" — was measured and **rejected**: run `11_10`, which keeps the shock
perfectly, peaks at 0.10 in long-term memory, lower than run `13_54` which loses it (0.15). Of
13 readable runs, 7 have a maximum importance below the retained importance. Short-term memory
importance does not propagate: consolidation is a synthesis reflection, not a copy, and no
identifier links the short-term entry to the long-term reflection. What remains is the words.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from loguru import logger

from llm.evenements.injection import PREFIXE_FOYER, PREFIXE_LU, PREFIXE_VECU

# `[ FOYER ]` (ticket 111): what an informed member heard from the reader. Without it, the
# consolidation of an informed member would escape the memory check.
PREFIXES = (PREFIXE_VECU, PREFIXE_LU, PREFIXE_FOYER)

# Minimum length of a retained word. Below it, these are articles, prepositions and numbers
# written as digits — none of them tells one text from another.
LONGUEUR_MIN = 4

# ── The exclusion list, and why each family is in it ─────────────────────────────────────
#
# A word COMMONPLACE IN THIS DOMAIN found in a reflection proves nothing: every agent of
# every run writes « trip », « minutes », « late ». On 2026-09-23, run `09_31` was first
# counted as HAVING LOST its shock, then as having kept it: the first reading matched
# « incident » with the sentence *« without any unexpected incidents »*, which says exactly
# the opposite. A witness that fires on the negation of what it looks for is worse than no
# witness.
#
# The admission rule for this list: would the word appear in the reflection of an ORDINARY
# day? If so, it is excluded.
BANALITES = frozenset(
    {
        # Everyday mobility vocabulary — present in every reflection, shock or not.
        "trip", "trips", "travel", "travelled", "travelling", "travels", "journey",
        "journeys", "route", "routes", "commute", "commuted", "commuting",
        "arrive", "arrived", "arrives", "arrival", "depart", "departed", "departure",
        "leave", "left", "went", "took", "taken", "take", "taking", "board", "boarded",
        "walk", "walked", "walking", "ride", "rode", "riding",
        # Time and delay: every reflection talks about them, including to say that everything
        # went well. « late », « delay » and « minutes » are the witness's three false friends.
        "time", "times", "late", "later", "delay", "delays", "delayed", "early",
        "minute", "minutes", "hour", "hours", "morning", "afternoon", "evening",
        "today", "day", "days", "week", "month", "schedule", "scheduled", "planned",
        # Modes and places: they NAME the trip, they do not qualify the incident.
        "line", "lines", "station", "stations", "stop", "stops", "stopped", "service",
        "services", "train", "trains", "metro", "bus", "buses", "tram", "car", "bike",
        "home", "work", "city", "centre", "center",
        # General appreciation words. « incident » is here FOR the reason above.
        "incident", "incidents", "problem", "problems", "issue", "issues",
        "usual", "usually", "normal", "again", "than", "that", "this", "with", "from",
        "have", "were", "was", "been", "being", "which", "when", "what", "where",
        "very", "much", "more", "most", "some", "they", "them", "there", "their",
        "then", "over", "into", "also", "just", "made", "make", "about", "after",
        # « between » was added on 2026-09-23 while calibrating: it is the ONLY word that the
        # v4 run found in an unrelated sentence (« between errands »), and it was enough to
        # make it pass for a healthy run. A linking word is never distinctive.
        "between", "among", "along", "around", "through", "while", "during",
        "before", "could", "would", "should", "will", "able", "back", "down", "well",
        "good", "fine", "nice", "only", "other", "same", "each", "both",
    }
)

_MOT = re.compile(r"[a-zA-Z]+")


@dataclass
class Constat:
    """What a consolidation did with the injected memory."""

    person_id: str
    sim_ts: int
    evenement_id: str
    mots_cherches: list[str] = field(default_factory=list)
    mots_retrouves: list[str] = field(default_factory=list)
    seuil: int = 2

    @property
    def retrouve(self) -> bool:
        return len(self.mots_retrouves) >= self.seuil

    @property
    def verdict(self) -> str:
        return "retrouvé" if self.retrouve else "PERDU"


def texte_injecte(contenus: Iterable[str]) -> str | None:
    """The event text among the short-term entries that the consolidation consumes.

    Returns `None` — the case of the vast majority of consolidations — when no entry carries
    an injection prefix. The anchor is the PREFIX and not the registry: the entry already
    carries the text, and bringing it down from the controller would add a path that can diverge.
    """
    for contenu in contenus:
        if not contenu:
            continue
        for prefixe in PREFIXES:
            position = contenu.find(prefixe)
            if position != -1:
                return contenu[position + len(prefixe) :].strip()
    return None


def mots_distinctifs(texte: str) -> list[str]:
    """The words of this text that could not come from an ordinary day.

    The order of appearance is kept: an alarm that names them reads in the order of the
    injected text, which makes the sentence recognisable at a glance.
    """
    vus: list[str] = []
    connus: set[str] = set()
    for brut in _MOT.findall(texte or ""):
        mot = brut.lower()
        if len(mot) < LONGUEUR_MIN or mot in BANALITES or mot in connus:
            continue
        connus.add(mot)
        vus.append(mot)
    return vus


def mots_retrouves(mots: Sequence[str], textes: Iterable[str]) -> list[str]:
    """Those of `mots` that appear in what the consolidation wrote.

    The comparison is made on WHOLE WORDS, never on substrings: « announce » in
    « announcement » is recognised through the shared root below, but « rain » must not be
    recognised in « train ». Substring matching produced exactly this kind of silent false
    positive.
    """
    presents: set[str] = set()
    for texte in textes:
        for brut in _MOT.findall(texte or ""):
            presents.add(brut.lower())
    trouves = []
    for mot in mots:
        if mot in presents:
            trouves.append(mot)
            continue
        # Inflection tolerance: « announcement » covers « announcements » and « announced ».
        # Five characters of common root — enough not to confuse two words of the corpus,
        # few enough to follow a plural or a participle.
        if len(mot) > 5 and any(_meme_racine(mot, p) for p in presents):
            trouves.append(mot)
    return trouves


def _meme_racine(mot: str, autre: str) -> bool:
    """Five characters of common root, both words being longer than five characters."""
    return len(autre) > 5 and (mot[:5] == autre[:5])


def _evenement_declare() -> str:
    """The identifier of the event declared by the run, or `""` if nothing is declared.

    The caller is the evening consolidation, which does not know which event was attached to
    a morning arrival: the identifier lives only in the registry, and the text of the short-term
    entry does not carry it. A run declares only ONE event (`evenement.yaml`, key
    `evenement`), possibly over several days — the reading is therefore exact as long as
    this form holds. The day a run declares two, the call site will have to pass
    `evenement_id` explicitly, and this fallback will become wrong.

    Never raises: without an identifier, the trace stays readable, it just names less.
    """
    try:
        from llm import evenements as evenements_module

        registre = evenements_module.registre()
        if registre is None:
            return ""
        return str(registre.evenement.evenement_id or "")
    except Exception:  # pragma: no cover - the control never breaks its caller
        return ""


def controler(
    *,
    person_id: str,
    sim_ts: int,
    contenus_courts: Iterable[str],
    textes_longs: Iterable[str],
    seuil: int,
    evenement_id: str = "",
) -> Constat | None:
    """The full check. Returns `None` when this consolidation follows no injection.

    NEVER raises: a witness that brought down a consolidation would cost more than the
    defect it watches for.
    """
    try:
        from urban_mobility_agents.utils.reprise import gel_actif

        if gel_actif():
            # A replay decides nothing and writes nothing: the consolidation it goes through was
            # already checked on the first pass. Checking it twice would double the trace and
            # skew the false alarm rate we are precisely trying to measure.
            return None
        texte = texte_injecte(contenus_courts)
        if texte is None:
            return None
        mots = mots_distinctifs(texte)
        if not mots:
            # A text without a single distinctive word cannot be searched for. It is not a
            # memory failure, it is a text this witness cannot follow — and saying so avoids
            # counting an alarm that is not one.
            logger.warning(
                f"[temoin] no distinctive word in the text injected into {person_id} — "
                f"the witness cannot conclude anything about this consolidation."
            )
            return None
        constat = Constat(
            person_id=str(person_id),
            sim_ts=int(sim_ts),
            evenement_id=str(evenement_id or _evenement_declare()),
            mots_cherches=mots,
            mots_retrouves=mots_retrouves(mots, textes_longs),
            seuil=int(seuil),
        )
        _journaliser(constat)
        _tracer(constat)
        return constat
    except Exception as err:  # noqa: BLE001 — jamais vers l'appelant
        logger.warning(f"[temoin] check impossible for {person_id} ({err})")
        return None


def _journaliser(constat: Constat) -> None:
    """Success is ALSO reported: a silent witness cannot tell "it works" from "it no longer runs"."""
    cherches = ", ".join(constat.mots_cherches[:8])
    retrouves = ", ".join(constat.mots_retrouves) or "aucun"
    if constat.retrouve:
        logger.info(
            f"[temoin] injected memory FOUND in long-term memory | "
            f"agent={constat.person_id} evenement={constat.evenement_id or '?'} "
            f"mots={retrouves} (threshold {constat.seuil})"
        )
        return
    logger.error(
        f"[ALARME] [temoin] the injected memory did NOT reach long-term memory | "
        f"agent={constat.person_id} evenement={constat.evenement_id or '?'} "
        f"retrouvés={retrouves} sur seuil={constat.seuil} | cherchés={cherches} — "
        f"the consolidation following the injection keeps no trace of it: the following days "
        f"do NOT measure the effect of a memory, and this arm cannot be used as it is."
    )


def _tracer(constat: Constat) -> None:
    """One JSONL line per check, read by `make report`.

    Written apart from the application log because a false alarm rate is counted over runs,
    not over a `grep` — and this is precisely what this witness must learn before anyone
    considers letting it stop anything.
    """
    try:
        from settings import _run_artifacts_disabled, settings

        if _run_artifacts_disabled():
            return
        entree = {
            "sim_ts": constat.sim_ts,
            "sim_day": datetime.fromtimestamp(constat.sim_ts, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            ),
            "person_id": constat.person_id,
            "evenement_id": constat.evenement_id,
            "retrouve": constat.retrouve,
            "seuil": constat.seuil,
            "mots_cherches": constat.mots_cherches,
            "mots_retrouves": constat.mots_retrouves,
        }
        with open(settings.app.temoin_souvenir_file, "a", encoding="utf-8") as flux:
            flux.write(json.dumps(entree, ensure_ascii=False, default=str) + "\n")
    except Exception as err:  # noqa: BLE001
        logger.warning(f"[temoin] trace not written for {constat.person_id} ({err})")
