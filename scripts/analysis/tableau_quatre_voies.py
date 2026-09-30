#!/usr/bin/env python3
"""Through which PATHWAY the past reaches a decision — ticket 100, lot 5.

THE QUESTION
------------
An agent suffers a breakdown, turns away from it for fifteen days, then comes back. Of the four
pathways through which its past reaches a decision, **which one carried this effect?** § 3.4 of
the manuscript names them, and the setup writes all four of them in the prompt:

| Pathway | In the prompt | What it carries |
|---|---|---|
| `habitudes` | « Mes habitudes » | what the agent does most often |
| `connaissances` | « Ce que je sais » | what it holds to be true — the concepts |
| `changements` | « Ce qui a changé récemment » | the serious memories and the dismissed beliefs |
| `rappel` | three chosen memories | the vector recall, traced in `trace_rappel.jsonl` |

This script says, **per regime and per role**, how many decisions saw the event go through
each one, and how many saw it through none. Only decisions **after** the exposure
count (`sim_ts` of the exchange against `timestamp` of the exposure); earlier and
undated ones are set aside, and counted. Each agent is read in **its** section of the prompt: a
call groups several agents, and a neighbour's blocks are not its own.

RECALL: THE MEMORY OF THE EVENT, SERVED, AND SEEN
-------------------------------------------------
That a memory was served says nothing: it has to be **the one of the event**. It is
found in long-term memory — as is for the `lu`, which deposits `[ PRESSE ] … « texte »` (exact
link); by salient words for the lived one, which reflection rewords (link `~`). The decision is
tied to its trace by the agent, the day and the departure time of its header, and the memory counts
if it is in `servis` **and** in the recalled memories of the prompt: the trace lists the top-K,
the prompt receives only the three most recent, and never a `conversation` entry. A `0`
comes out only for recall, measured: exact link, matched trace, never served. Without a trace, or
without an identified memory, the cell stays empty, and the table footer says why.

WHAT IT CANNOT DO, AND IT SAYS SO
---------------------------------
⚠ **Matching by text fails for the lived event.** Measured on 2026-09-16: the injected text —
« The engine made a grinding noise and the car stalled » — is never copied as is. It
goes into short-term memory, reflection REWORDS it, and it is the rewording that reaches
long-term memory. Searching for the literal text therefore returns nothing.

Hence two searches, and they are not worth the same:

- **exact** — the text of the event, or one of its distinctive fragments, found as is.
  It is reliable when it finds, and silent when it does not.
- **by salient words** — the rare words of the text, found together. It is indicative,
  and the table says so: a cell coming from this search carries the mark `~`.

A decision for which neither of the two concludes comes out **empty**, never `non`. Writing
"the event weighed on no decision" where the truth is "we cannot tell"
would be a lie, and this repository has already produced some.

    python scripts/analysis/tableau_quatre_voies.py <run> [--markdown]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from scripts.analysis.lecture_avant_decision import informes_du_run, sous_role
from scripts.analysis.memoire.sources import (
    EntreeLTM,
    horodatage_iso,
    lire_echanges,
    lire_ltm,
)

# The headers that `llm/noyau.py` puts in the prompt. READ HERE, and they must stay
# aligned: a renamed header would make a pathway come out at zero without any error showing.
# The test `scripts/tests/test_four_way_outputs.py` compares them to those of `noyau.py`.
#
# Each pathway carries its TWO headers: English since 2026-09-25, French in the archives
# from before. The first one is the one `noyau.py` puts today.
EN_TETES = {
    "habitudes": ("My habits", "Mes habitudes"),
    "connaissances": ("What I know", "Ce que je sais"),
    "changements": ("What changed recently", "Ce qui a changé récemment"),
}
VOIES = ("habitudes", "connaissances", "changements", "rappel")
CATEGORIE_DECISION = "itinary_multi_agent"

# The header of each persona in a decision prompt. A prompt groups several: the
# blocks of an agent are searched in ITS section, never in the whole prompt.
_ENTETE_AGENT = re.compile(r"^--- agent_id=(?P<agent>[^\s|]+)(?P<reste>.*)$", re.MULTILINE)
_DEPART = re.compile(r"Departure:\s*(?P<heure>\d{1,2}):(?P<minute>\d{2})")
# The recalled memories, which `llm_agent.py` writes AFTER the three blocks, dated
# (`[Thursday, March 19] …`) or marked `[Concept] …`, and which the template prefixes with « - ».
# Without this bound, the last named block swallowed the recalled memories.
_LIGNE_RAPPEL = re.compile(r"^- \[(?:Concept|[A-Z][a-z]+, [A-Z][a-z]+ \d{1,2})\] ",
                           re.MULTILINE)
# Length of the compared text. Normalised — blanks collapsed — on both sides: a line break
# of the article and the space replacing it in another rendering stay the same text.
_SIGNATURE = 60
_ENONCE = 80

# Words too common to be salient. Deliberately short: the list is there to discard
# noise, not to do linguistics.
_BANALS = frozenset("""
the and for with that this from they have been were was are but not you your his her its
into than then there here when what which while would could should about after before
""".split())
_MOT = re.compile(r"[A-Za-zÀ-ÿ']{4,}")


def mots_saillants(texte: str, combien: int = 6) -> list[str]:
    """The rarest words of the text, in order of appearance. Indicative, never conclusive."""
    vus: list[str] = []
    for mot in _MOT.findall(texte or ""):
        bas = mot.lower()
        if bas in _BANALS or bas in vus:
            continue
        vus.append(bas)
    # Longest first: « grinding » discriminates, « road » does not.
    return sorted(vus, key=len, reverse=True)[:combien]


def _jsonl(chemin: Path) -> list[dict]:
    """A true JSONL, one object per line: `evenements.jsonl`, `chocs.jsonl`, `trace_rappel.jsonl`.

    NOT `llm_exchanges.jsonl`: the gateway writes indented objects there, which a line-by-line
    read does not decode — or decodes wrongly, a list of strings yielding bare strings.
    """
    if not chemin.is_file():
        return []
    lignes = []
    for brut in chemin.read_text("utf-8").splitlines():
        brut = brut.strip()
        if not brut:
            continue
        try:
            lignes.append(json.loads(brut))
        except json.JSONDecodeError:
            # Last line truncated by a hard stop: skipped, never fatal.
            continue
    return lignes


def evenements_du_run(run: Path) -> list[dict]:
    lignes = _jsonl(run / "evenements.jsonl")
    return lignes or _jsonl(run / "chocs.jsonl")


def roles_du_run(run: Path) -> dict[str, str]:
    """`{person_id: role}`, read from `moves.csv`. Empty if the column is not there.

    The agent is in `ID Personne`. `Référence` carries the run name (`experiences/journal.py`):
    read as an identifier, it returned no role, and each row of the table came out `?`.
    """
    chemin = run / "moves.csv"
    if not chemin.is_file():
        return {}
    roles: dict[str, str] = {}
    with chemin.open(encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        if "Rôle" not in (lecteur.fieldnames or []):
            return {}
        for ligne in lecteur:
            pid = (ligne.get("ID Personne") or ligne.get("person_id") or "").strip()
            role = (ligne.get("Rôle") or "").strip()
            if pid and role:
                roles.setdefault(pid, role)
    return roles


def _norm(texte: str) -> str:
    """Blanks collapsed to one space: the form in which two texts are compared."""
    return " ".join((texte or "").split())


def _signature(texte: str) -> str:
    """The start of the event text, normalised: what the exact matching looks for."""
    return _norm(texte)[:_SIGNATURE]


def _instant(valeur) -> float | None:
    """Simulated UTC seconds, or `None` if the value is missing or cannot be read."""
    if valeur is None or valeur == "":
        return None
    try:
        return float(valeur)
    except (TypeError, ValueError):
        return None


def _jour_et_heure(instant: float) -> tuple[str, str]:
    t = datetime.fromtimestamp(instant, tz=timezone.utc)
    return t.strftime("%Y-%m-%d"), t.strftime("%H:%M")


# ── The exposures ────────────────────────────────────────────────────────────────


class Exposition(NamedTuple):
    canal: str
    texte: str
    instant: float | None  # the FIRST known instant, if the agent is exposed several times


def instant_exposition(evenement: dict) -> float | None:
    """`timestamp`, otherwise `horodatage_simule` read as UTC — the convention of `sim_ts`."""
    instant = _instant(evenement.get("timestamp"))
    if instant is not None:
        return instant
    quand = horodatage_iso(evenement.get("horodatage_simule"))
    if quand is None:
        return None
    if quand.tzinfo is None:
        quand = quand.replace(tzinfo=timezone.utc)
    return quand.timestamp()


def expositions_du_run(evenements: list[dict]) -> dict[str, Exposition]:
    """`{person_id: Exposition}`. An agent without text is not exposed: nothing to look for."""
    par_agent: dict[str, Exposition] = {}
    for e in evenements:
        pid = str(e.get("person_id") or "")
        texte = str(e.get("texte") or e.get("vecu") or "")
        if not pid or not texte:
            continue
        instant = instant_exposition(e)
        avant = par_agent.get(pid)
        if avant is not None:
            connus = [t for t in (avant.instant, instant) if t is not None]
            par_agent[pid] = avant._replace(instant=min(connus) if connus else None)
            continue
        par_agent[pid] = Exposition(str(e.get("canal") or ""), texte, instant)
    return par_agent


# ── The prompt: raw content, sections, blocks, recall zone ───────────────────────


def _contenu(echange: dict) -> str:
    """The text sent to the model, as is: the `content` of the messages, end to end.

    Never `json.dumps(messages)`: serialisation escapes line breaks and quotes, and
    the start of an article (« (Translated from French)\\nGusts… ») was never found in it.
    A bare string is taken as is.
    """
    messages = echange.get("messages")
    if isinstance(messages, str):
        return messages
    return "\n".join(
        str(m.get("content") or "") for m in messages or [] if isinstance(m, dict)
    )


class Section(NamedTuple):
    agent: str
    depart: str | None  # « HH:MM », the departure time the header sets
    texte: str


def sections(contenu: str) -> list[Section]:
    """One section per persona, from its `--- agent_id=` header to the next one.

    ⚠ It is the ONLY reliable attribution, and it is essential for the `lu` channel: all
    the readers of a same article share the same text, searching the text alone would not say
    who saw it. The identifier is read whole: `agent_id=286920` is not
    `agent_id=2869201`, which a substring search confused.
    """
    entetes = list(_ENTETE_AGENT.finditer(contenu))
    decoupe: list[Section] = []
    for i, entete in enumerate(entetes):
        fin = entetes[i + 1].start() if i + 1 < len(entetes) else len(contenu)
        depart = _DEPART.search(entete.group("reste"))
        decoupe.append(Section(
            agent=entete.group("agent"),
            depart=f"{int(depart['heure']):02d}:{depart['minute']}" if depart else None,
            texte=contenu[entete.start():fin],
        ))
    return decoupe


def zone_rappel(section: str) -> str:
    """The recalled memories: from the first dated or `[Concept]` line to the end of the section."""
    debut = _LIGNE_RAPPEL.search(section)
    return section[debut.start():] if debut else ""


def _bloc(section: str, voie: str) -> str:
    """The content of the named block, in an agent's section. Empty string if it is not there.

    The block stops at the next known header, and in any case before the first recalled
    memory: without this bound, the last block swallowed the recall memories, and a recalled
    memory carrying the text was attributed to « Ce que je sais ».
    """
    rappel = _LIGNE_RAPPEL.search(section)
    noyau = section[: rappel.start()] if rappel else section
    for en_tete in EN_TETES[voie]:
        debut = noyau.find(en_tete)
        if debut >= 0:
            break
    else:
        return ""
    reste = noyau[debut + len(en_tete):]
    tous = [t for alternatives in EN_TETES.values() for t in alternatives]
    fins = [i for i in (reste.find(t) for t in tous) if i > 0]
    return reste[: min(fins)] if fins else reste


def _voies_textuelles(section: str, texte: str) -> list[tuple[str, str]]:
    """`[(voie, « exact » | « saillant »)]` for the three blocks of the core."""
    signature = _signature(texte)
    saillants = mots_saillants(texte)
    trouvees: list[tuple[str, str]] = []
    for voie in ("habitudes", "connaissances", "changements"):
        bloc = _bloc(section, voie)
        if not bloc:
            continue
        if signature and signature in _norm(bloc):
            trouvees.append((voie, "exact"))
        elif saillants and sum(m in bloc.lower() for m in saillants) >= 2:
            trouvees.append((voie, "saillant"))
    return trouvees


# ── Recall: the memory of the event, the trace, the prompt ───────────────────────


class Souvenir(NamedTuple):
    doc_id: str
    lien: str  # « exact » or « saillant »
    memory_type: str
    enonce: str  # what the prompt shows of it: the text, or the concept statement


def souvenirs_de_l_evenement(entrees: list[EntreeLTM], exposition: Exposition) -> list[Souvenir]:
    """The long-term memory entries born of the event, and the strength of the link.

    - **exact** — the entry contains the text of the event. It is the case of the `lu`, deposited
      as is: `[ PRESSE ] This morning I read in the paper: « … »`.
    - **saillant** — dated on the exposure day or later, it carries at least two salient
      words of the text. It is the case of the lived event, which reflection REWORDS. Indicative.

    No identifier ties an entry to the event: `agent_memory_events.jsonl` marks
    the write (`data.evenement`) without the `doc_id`, and a concept drawn from the memory by
    consolidation is marked nowhere.
    """
    signature = _signature(exposition.texte)
    saillants = mots_saillants(exposition.texte)
    jour = _jour_et_heure(exposition.instant)[0] if exposition.instant is not None else ""
    lies: list[Souvenir] = []
    for e in entrees:
        if not e.doc_id:
            continue
        if signature and signature in _norm(e.contenu):
            lien = "exact"
        elif (jour and e.timestamp[:10] >= jour and saillants
              and sum(m in e.contenu.lower() for m in saillants) >= 2):
            lien = "saillant"
        else:
            continue
        lies.append(Souvenir(e.doc_id, lien, e.memory_type, e.enonce))
    return lies


def traces_par_depart(rappels: list[dict]) -> dict[tuple[str, str, str], list[frozenset[str]]]:
    """`{(agent, jour, « HH:MM »): [doc_ids servis, …]}`, from `trace_rappel.jsonl`.

    The trace is dated at the instant of the request, which is the agent's departure time — and
    not the `sim_ts` of the exchange, which is the batch's. On run 2026-09-24_17_50,
    matching by `sim_ts` missed 43 sections out of 140; matching by departure time
    misses 29, which have no trace at that time.
    """
    index: dict[tuple[str, str, str], list[frozenset[str]]] = defaultdict(list)
    for r in rappels:
        instant = _instant(r.get("sim_ts"))
        if instant is None:
            continue
        jour, heure = _jour_et_heure(instant)
        index[(str(r.get("person_id") or ""), jour, heure)].append(frozenset(
            str(s["doc_id"]) for s in r.get("servis") or []
            if isinstance(s, dict) and s.get("doc_id")
        ))
    return index


def _dans(enonce: str, zone_normalisee: str) -> bool:
    debut = _norm(enonce)[:_ENONCE]
    return bool(debut) and debut in zone_normalisee


def depouiller(run: Path) -> dict:
    evenements = evenements_du_run(run)
    if not evenements:
        raise SystemExit(
            f"❌ neither `evenements.jsonl` nor `chocs.jsonl` in {run}: no event was "
            f"applied. This is not a null result, it is a run without an event."
        )
    # Ticket 111: in a household where the reader spoke, the informed co-resident and the one
    # who was told nothing are not mixed — `relais_foyer.jsonl` is authoritative.
    informes = informes_du_run(run)
    roles = {pid: sous_role(r, pid, informes) for pid, r in roles_du_run(run).items()}
    expositions = expositions_du_run(evenements)

    # The decisions, read from the LLM exchanges. Only those of the decision category
    # count: a nightly reflection is not a decision. The reader is the one of the memory
    # report: the file is not JSONL (cf. `_jsonl`).
    echanges = lire_echanges(run)
    rappels = _jsonl(run / "trace_rappel.jsonl")
    traces = traces_par_depart(rappels)
    memoire, _habitudes = lire_ltm(run)
    souvenirs = {
        pid: souvenirs_de_l_evenement(memoire.get(pid, []), expo)
        for pid, expo in expositions.items()
    }

    compte: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    decisions: Counter = Counter()
    # Per table row: what makes recall measurable, or not. Without these counters, an
    # empty cell would not say whether the trace was missing, or the memory, or it was not served.
    rappel: dict[tuple[str, str], Counter] = defaultdict(Counter)
    souvenirs_par_ligne: dict[tuple[str, str], set[Souvenir]] = defaultdict(set)
    ecartees: Counter = Counter()
    decisions_lues = 0
    lues_exposes = 0
    traces_appariees: set[tuple[str, str, str]] = set()

    for echange in echanges:
        if str(echange.get("category") or "") != CATEGORIE_DECISION:
            continue
        decisions_lues += 1
        instant = _instant(echange.get("sim_ts"))
        for section in sections(_contenu(echange)):
            expo = expositions.get(section.agent)
            if expo is None:
                continue
            lues_exposes += 1
            # A decision sees the event only if it follows it. Waking up precedes the
            # reading: a decision taken AT the instant of the exposure did not see it.
            if instant is None or expo.instant is None:
                ecartees["non_datees"] += 1
                continue
            if instant <= expo.instant:
                ecartees["anterieures"] += 1
                continue
            cle_base = (expo.canal, roles.get(section.agent, ""))
            decisions[cle_base] += 1
            for voie, qualite in _voies_textuelles(section.texte, expo.texte):
                compte[(*cle_base, voie)][qualite] += 1

            # Recall: the memory BORN OF THE EVENT, served for THIS decision, and
            # present in its prompt. Served is not enough: the trace lists the top-K, the prompt
            # receives only the most recent, and never a `conversation` entry.
            lies = souvenirs.get(section.agent) or []
            suivi = rappel[cle_base]
            if not lies:
                suivi["sans_souvenir"] += 1
                continue
            souvenirs_par_ligne[cle_base].update(lies)
            jour = str(echange.get("sim_day") or "") or _jour_et_heure(instant)[0]
            cle_trace = (section.agent, jour, section.depart or "")
            candidates = set(traces.get(cle_trace, ())) if section.depart else set()
            if not candidates:
                suivi["sans_trace"] += 1
                continue
            traces_appariees.add(cle_trace)
            if len(candidates) > 1:
                # Two traces at the same minute that do not serve the same thing: which one
                # belonged to this decision, nothing says.
                suivi["ambigu"] += 1
                continue
            servis = next(iter(candidates))
            suivi["mesurables"] += 1
            if any(s.lien == "exact" for s in lies):
                suivi["mesurables_exact"] += 1
            zone = _norm(zone_rappel(section.texte))
            vus = [s for s in lies if s.doc_id in servis and _dans(s.enonce, zone)]
            if vus:
                meilleur = "exact" if any(s.lien == "exact" for s in vus) else "saillant"
                compte[(*cle_base, "rappel")][meilleur] += 1
            elif any(s.doc_id in servis for s in lies):
                suivi["hors_prompt"] += 1
            if any(s.doc_id not in servis and _dans(s.enonce, zone) for s in lies):
                suivi["hors_trace"] += 1

    # The recalls of the event memory that no read decision claims: the trace
    # exists, the prompt does not — decision served by the cache, or not logged.
    rappels_sans_decision: list[tuple[str, str, str]] = []
    for (pid, jour, heure), listes in sorted(traces.items()):
        docs = {s.doc_id for s in souvenirs.get(pid) or []}
        if not docs or (pid, jour, heure) in traces_appariees:
            continue
        for doc in sorted(docs & frozenset().union(*listes)):
            rappels_sans_decision.append((pid, f"{jour} {heure}", doc))

    return {"compte": compte, "decisions": decisions, "evenements": len(evenements),
            "roles_connus": bool(roles), "echanges": len(echanges),
            "decisions_lues": decisions_lues, "lues_exposes": lues_exposes,
            "ecartees": ecartees, "rappel": rappel, "traces_lues": len(rappels),
            "souvenirs": {k: sorted(v) for k, v in souvenirs_par_ligne.items()},
            "rappels_sans_decision": rappels_sans_decision}


def _lu(resultat: dict) -> str:
    """What was read. A table without its count cannot be told from an empty table."""
    n, d = resultat["echanges"], resultat["decisions_lues"]
    return (f"Lu : {n} échange{'s' * (n > 1)}, dont {d} décision{'s' * (d > 1)} "
            f"(`{CATEGORIE_DECISION}`), pour {resultat['evenements']} exposition(s).")


def _retenues(resultat: dict) -> str:
    """The decisions of the exposed agents, those that count and those that cannot."""
    ecartees = resultat["ecartees"]
    return (
        f"Décisions des agents exposés : {resultat['lues_exposes']} lue(s), "
        f"{sum(resultat['decisions'].values())} postérieure(s) à l'exposition retenue(s), "
        f"{ecartees['anterieures']} antérieure(s) et {ecartees['non_datees']} non datée(s) "
        f"écartée(s)."
    )


def _suivi_rappel(resultat: dict) -> list[str]:
    """What made recall measurable, row by row. An empty cell is explained there."""
    lignes: list[str] = []
    if not resultat["traces_lues"]:
        lignes.append("⚠ `trace_rappel.jsonl` absent ou vide (`agent.trace_rappel_enabled` "
                      "était-il actif ?) : le rappel ne se mesure pas.")
    for cle in sorted(resultat["decisions"]):
        canal, role = cle
        nom = f"{canal or '?'}/{role or '?'}"
        suivi = resultat["rappel"].get(cle) or Counter()
        lies = resultat["souvenirs"].get(cle) or []
        if not lies:
            lignes.append(f"Rappel ({nom}) : aucun souvenir de l'événement identifié en mémoire "
                          f"longue — voie non mesurable.")
            continue
        noms = ", ".join(f"{s.doc_id} ({s.lien}, {s.memory_type or '?'})" for s in lies[:5])
        noms += " …" if len(lies) > 5 else ""
        details = [
            f"souvenir(s) de l'événement : {noms}",
            f"servi au top-K sans atteindre le prompt : {suivi['hors_prompt']}",
            f"sans trace de rappel appariée : {suivi['sans_trace']}",
        ]
        if suivi["ambigu"]:
            details.append(f"trace ambiguë : {suivi['ambigu']}")
        if suivi["sans_souvenir"]:
            details.append(f"agent sans souvenir identifié : {suivi['sans_souvenir']}")
        lignes.append(f"Rappel ({nom}) : mesurable sur {suivi['mesurables']} des "
                      f"{resultat['decisions'][cle]} décision(s) — " + " ; ".join(details) + ".")
        if suivi["hors_trace"]:
            lignes.append(
                f"⚠ [INCOHÉRENCE] {suivi['hors_trace']} décision(s) ({nom}) portent un souvenir "
                f"de l'événement dans la zone de rappel sans que la trace ne le liste : la trace "
                f"ou le découpage du prompt est faux."
            )
    sans = resultat["rappels_sans_decision"]
    if sans:
        detail = ", ".join(f"{pid} le {quand} ({doc})" for pid, quand, doc in sans[:5])
        detail += " …" if len(sans) > 5 else ""
        lignes.append(
            f"⚠ {len(sans)} rappel(s) d'un souvenir de l'événement sans décision lue : {detail}. "
            f"Servi pour un départ dont le prompt n'est pas dans `llm_exchanges.jsonl` "
            f"(décision servie par le cache, ou non journalisée) — non compté."
        )
    return lignes


def rendre(resultat: dict, markdown: bool = False) -> str:
    compte, decisions = resultat["compte"], resultat["decisions"]
    lignes: list[str] = []
    if not resultat["echanges"]:
        # Nothing read is not "nothing found": without a prompt, no decision could be examined.
        return (
            "non concluant — aucun échange lu dans `llm_exchanges.jsonl` (absent, vide ou "
            "illisible : `telemetry.exchanges_enabled` était-il actif ?).\n"
            "Ce n'est PAS « l'événement n'a pesé sur rien » : aucune décision n'a été lue.\n"
            + _lu(resultat)
        )
    if not decisions and resultat["lues_exposes"]:
        # Decisions of the exposed agent, but none that could have seen the event.
        return (
            "non concluant — aucune décision postérieure à l'exposition.\n"
            "Ce n'est PAS « l'événement n'a pesé sur rien » : aucune décision retenue ne suit "
            "l'exposition, ou aucune ne se date.\n"
            + _retenues(resultat) + "\n" + _lu(resultat)
        )
    if not decisions:
        return (
            "non concluant — aucune décision ne porte le texte de l'événement.\n"
            "Ce n'est PAS « l'événement n'a pesé sur rien » : l'appariement par le texte échoue "
            "pour le régime vécu, la réflexion reformule le vécu avant qu'il n'atteigne la "
            "mémoire longue (mesuré le 2026-09-16). Il faudrait un lien explicite posé à la "
            "source, et il n'existe pas.\n" + _lu(resultat)
        )
    sep = "|" if markdown else " "
    lignes.append(f"{'canal':<8}{sep}{'rôle':<14}{sep}{'décisions':>9}{sep}"
                  + sep.join(f"{v:>14}" for v in VOIES))
    if markdown:
        lignes.append("|".join(["---"] * (3 + len(VOIES))))
    for (canal, role), total in sorted(decisions.items()):
        cellules = []
        for voie in VOIES:
            c = compte.get((canal, role, voie)) or Counter()
            exact, saillant = c.get("exact", 0), c.get("saillant", 0)
            if not exact and not saillant:
                mesure = (resultat["rappel"].get((canal, role)) or Counter())["mesurables_exact"]
                # `0` for recall only, and only MEASURED: memory linked exactly, trace
                # matched, never served. Everywhere else, EMPTY, never zero.
                cellules.append(f"{0:>14}" if voie == "rappel" and mesure else f"{'':>14}")
            elif saillant and not exact:
                cellules.append(f"{f'~{saillant}':>14}")
            elif saillant:
                cellules.append(f"{f'{exact} (~{saillant})':>14}")
            else:
                cellules.append(f"{exact:>14}")
        lignes.append(f"{canal or '?':<8}{sep}{role or '?':<14}{sep}{total:>9}{sep}"
                      + sep.join(cellules))
    # The decisions where NO pathway was found. Counted and named, because a row
    # of empty cells reads too easily as "the event weighed on nothing".
    muettes = sum(
        total for (canal, role), total in decisions.items()
        if not any((compte.get((canal, role, v)) or Counter()) for v in VOIES)
    )
    lignes.append("")
    lignes.append("`~` = appariement par mots saillants, INDICATIF. Une cellule vide n'est pas "
                  "un zéro : c'est une absence de mesure. Un `0` ne sort qu'au rappel, mesuré : "
                  "souvenir de l'événement identifié tel quel, trace appariée, jamais servi.")
    if muettes:
        lignes.append(
            f"⚠ {muettes} décision(s) où l'événement n'a été retrouvé par AUCUNE voie. Ce n'est "
            f"PAS « l'événement n'a pesé sur rien » : l'appariement par le texte échoue pour le "
            f"régime vécu, la réflexion reformulant le vécu avant qu'il n'atteigne la mémoire "
            f"longue (mesuré le 2026-09-16). Il faudrait un lien explicite posé à la source, et "
            f"il n'existe pas."
        )
    lignes.extend(_suivi_rappel(resultat))
    if not resultat["roles_connus"]:
        lignes.append("⚠ `moves.csv` ne porte pas la colonne « Rôle » : run antérieur au ticket "
                      "100, lot 5. Les rôles ne se reconstituent pas après coup.")
    lignes.append(_retenues(resultat))
    lignes.append(_lu(resultat))
    return "\n".join(lignes)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    print(rendre(depouiller(args.run), args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
