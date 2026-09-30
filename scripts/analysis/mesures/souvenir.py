"""Is the memory of an event SERVED to the decisions that follow? — analysis of 2026-09-25.

WHAT DID NOT WORK
-----------------
`evenement_par_jour.csv` answered with two columns, `souvenir_choc_servi` and
`decisions_avec_souvenir_choc`, and they were 0 on the treated arm `2026-09-24_17_50` although
six decision prompts carried article a09: 286923 on 27 March at 12:40, 14:21 and
17:36, 286920 on 30 and 31 March at 5:57, 286921 on 30 March at 13:38. Three reasons, which
added up:

1. **The link sought was literal.** The first sixty characters of the injected text, in
   long-term memory. Consolidation REWORDS: "Toulouse closed its parks and gardens from
   6 p.m. due to a yellow storm warning" does not contain the beginning of the article.
2. **"Served" was read in `trace_rappel`**, which counts what the search brought up and not
   what the prompt contains: when the core is present, only the two or three most recent
   episodic memories are kept, and the core itself ("Ce que je sais", "Ce qui a changé
   récemment") does not appear there.
3. **Same day only.** An article read at wake-up weighs, if it weighs, on the following days.

WHAT THIS MODULE MEASURES INSTEAD
---------------------------------
It reads the `**History:**` section of each decision prompt — what the model read, no more
no less — and looks for the event there in two ways:

- **texte**: the article itself, whitespace normalised (the `[ PRESSE ]` entry of memory, the
  guaranteed line of ticket 111, or what a household member said of it word for word);
- **mots**: a rewording — at least `MOTS_MIN` distinct stems (five characters) among
  the article's distinctive words (`llm.evenements.temoin.mots_distinctifs`, the detector that
  consolidation already uses), **stripped of any word the household had written in memory before
  the reading**. Without this filter, "weather" or "warning" would be enough to turn a rainy
  day into a memory of the article. Measured on 2026-09-25: 7 derived documents on the treated
  arm, 0 out of 51 in the control `2026-09-24_23_06`.

The follow-up covers the reader's **whole household**, from the day of the reading (J0) to the end
of the run: the reader (`expose`) and its co-residents, informed by the ticket 111 relay or not.

It also says THROUGH WHERE the memory came in: "Ce que je sais" (a concept), "Ce qui a changé
récemment" (a shock, or the GUARANTEED line of ticket 111 — five days of trips, which is therefore
not a spontaneous memory) or the recalled episodic memories. That is what separates "the mechanism
served the article" from "the agent remembered it".

⚠ **A decision taken from the cache has no prompt.** It counts neither with nor without
memory: `decisions_sans_prompt` shows it, so that a day without a prompt does not read as
a day without memory.

⚠ **Continuity.** Nothing here depends on the final state of memory, except the vocabulary from
before J0, frozen from J0 on: a line of a past day does not move when the run moves on. The list of
derived documents (`souvenirs_derives.csv`), on the other hand, is a SNAPSHOT of long-term memory at
computation time — a concept merged later disappears from it. It therefore does not enter the
continuity check.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from scripts.analysis.memoire.sources import BlocPrompt, blocs_itineraires, lire_echanges, lire_ltm
from scripts.analysis.mesures import calendrier

RACINE = Path(__file__).resolve().parents[3]
if str(RACINE / "services" / "llm-agents") not in sys.path:
    sys.path.insert(0, str(RACINE / "services" / "llm-agents"))

from llm.evenements import temoin  # noqa: E402  (après l'ajout au chemin)

# Distinct stems required to recognise a rewording. ⚠ MORE than the consolidation detector
# (`temoin_souvenir_mots_min`, 2), and it is measured: the detector compares the injected
# text with what consolidation writes RIGHT after; here the whole memory served on
# the following days is swept, and two rare words meet there by chance. On the treated arm of
# 2026-09-24_17_50, threshold 2 took for a memory of the article "a quick shopping trip"
# to Compans (the metro station, namesake of the Compans-Caffarelli garden) that "remained"
# rainy; all real memories there carry 3 stems or more.
MOTS_MIN = 3
LONGUEUR_RACINE = 5
# The translation mention required by the text contract is not the article.
MOTS_DE_LA_MENTION = frozenset({"translated", "french"})
# Day and month names distinguish nothing: the template prefixes EACH recalled episodic memory
# with its date ("[Thursday, March 26]"), and article a09 says "this Thursday". Without
# this filter, every reflection of a Thursday already carried a stem of the article.
CALENDRIER = frozenset({
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
})
_MENTION = re.compile(r"^\s*\((?:translated|traduit)[^)]*\)\s*", re.IGNORECASE)
LONGUEUR_EXTRAIT = 60

# English titles since 2026-09-25 (`llm/noyau.py`), French in the archives from before.
SECTIONS = {
    "My habits": "habitudes",
    "What I know": "connaissances",
    "What changed recently": "changements",
    "Mes habitudes": "habitudes",
    "Ce que je sais": "connaissances",
    "Ce qui a changé récemment": "changements",
}
VOIES = ("connaissances", "changements", "rappel")


@dataclass(frozen=True)
class Signature:
    """What recognises an event in a text: a literal excerpt and words."""

    evenement_id: str
    extrait: str
    mots: tuple[str, ...]

    def reconnait(self, texte: str) -> str | None:
        """`"texte"`, `"mots"` or `None`."""
        normal = _normaliser(texte)
        if self.extrait and self.extrait in normal:
            return "texte"
        if len(racines(temoin.mots_retrouves(self.mots, [texte]))) >= MOTS_MIN:
            return "mots"
        return None


def racines(mots: Sequence[str]) -> set[str]:
    """Five characters, without the plural: "park" and "parks" are ONE word of the article."""
    return {m[:LONGUEUR_RACINE].removesuffix("s") for m in mots}


@dataclass(frozen=True)
class Suivi:
    """A member of an exposed household, followed from the day of the reading."""

    evenement_id: str
    person_id: str
    household_id: str | None
    role: str          # "expose" | "co_resident"
    informe: bool      # ticket 111: an `origine: entendu` line concerns it
    j0: str
    # The instant of the first reading in the household: the "before" vocabulary stops there.
    # ⚠ Not the day: an article read at 00:00 is, with the 3 a.m. boundary, in the day before
    # — and its `[ PRESSE ]` entry would have poured all its words into the earlier
    # vocabulary, emptying the signature.
    instant: str
    signature: Signature


@dataclass(frozen=True)
class LigneSouvenir:
    jour_simule: int
    date_simulee: str
    person_id: str
    evenement_id: str
    role: str
    informe: bool
    jours_depuis_j0: int
    decisions: int
    prompts: int
    decisions_sans_prompt: int
    prompts_avec_souvenir: int
    souvenir_texte: int
    souvenir_mots: int
    via_connaissances: int
    via_changements: int
    via_rappel: int


@dataclass(frozen=True)
class SouvenirDerive:
    evenement_id: str
    person_id: str
    role: str
    doc_id: str
    type_souvenir: str
    ecrit_le: str
    appariement: str
    mots_retrouves: str
    enonce: str


# ── Signature of an event ─────────────────────────────────────────────────────────────


def _normaliser(texte: str) -> str:
    return re.sub(r"\s+", " ", texte or "").strip().lower()


def _sans_mention(texte: str) -> str:
    return _MENTION.sub("", texte or "", count=1)


def _vocabulaire(textes: Sequence[str]) -> set[str]:
    return {m.lower() for t in textes for m in temoin._MOT.findall(t or "")}


def signature(evenement_id: str, texte: str, vocabulaire_anterieur: set[str]) -> Signature:
    """Excerpt and distinctive words of the article, minus the vocabulary the household had."""
    corps = _sans_mention(texte)
    mots = tuple(
        m for m in temoin.mots_distinctifs(corps)
        if m not in MOTS_DE_LA_MENTION and m not in CALENDRIER
        and m not in vocabulaire_anterieur
    )
    return Signature(evenement_id, _normaliser(corps)[:LONGUEUR_EXTRAIT], mots)


# ── Whom to follow, from when ─────────────────────────────────────────────────────────


def suivis(chemin_run: Path, evenements: Sequence[dict],
           journee_de: Any) -> list[Suivi]:
    """The members of exposed households, with J0 and the event signature for that household.

    `journee_de(evenement)` returns the simulated day of an exposure (that of `evenement_par_jour`).
    """
    menages = calendrier.menages(chemin_run)
    informes = {
        (str(e.get("evenement_id") or e.get("choc_id") or ""), str(e.get("person_id")))
        for e in evenements if e.get("origine") == "entendu"
    }
    # (event, household) → J0, readers and text. An agent outside any known population
    # forms its own household: it is followed alone rather than not followed.
    foyers: dict[tuple[str, str], dict[str, Any]] = {}
    for e in evenements:
        if e.get("origine") == "entendu":
            continue
        agent = str(e.get("person_id") or "")
        identifiant = str(e.get("evenement_id") or e.get("choc_id") or "")
        journee = journee_de(e)
        texte = str(e.get("vecu") or "").strip()
        if not (agent and identifiant and journee and texte):
            continue
        foyer = menages.get(agent) or f"seul:{agent}"
        instant = _instant(e.get("horodatage_simule"))
        courant = foyers.setdefault((identifiant, foyer), {"j0": journee, "lecteurs": set(),
                                                           "texte": texte, "instant": instant})
        courant["j0"] = min(courant["j0"], journee)
        courant["instant"] = min(courant["instant"], instant)
        courant["lecteurs"].add(agent)
    if not foyers:
        return []

    entrees, _ = lire_ltm(chemin_run)
    resultat: list[Suivi] = []
    for (identifiant, foyer), info in sorted(foyers.items()):
        membres = (
            sorted(p for p, h in menages.items() if h == foyer)
            if not foyer.startswith("seul:") else [foyer.split(":", 1)[1]]
        )
        anterieur = _vocabulaire([
            x.contenu for p in membres for x in entrees.get(p, [])
            if _instant(x.timestamp) and _instant(x.timestamp) < info["instant"]
        ])
        sig = signature(identifiant, info["texte"], anterieur)
        for p in membres:
            resultat.append(Suivi(
                evenement_id=identifiant, person_id=p,
                household_id=None if foyer.startswith("seul:") else foyer,
                role="expose" if p in info["lecteurs"] else "co_resident",
                informe=(identifiant, p) in informes, j0=info["j0"],
                instant=info["instant"], signature=sig,
            ))
    return resultat


def _instant(horodatage: Any) -> str:
    """Timestamp "YYYY-MM-DDTHH:MM:SS", comparable as a string. Empty if unreadable."""
    texte = str(horodatage or "").strip().replace(" ", "T")[:19]
    return texte if len(texte) >= 10 else ""


def _journee_ltm(horodatage: str) -> str:
    from scripts.analysis.mesures.calcul import journee_du_moment

    return journee_du_moment(horodatage) or ""


# ── What the prompts contained ────────────────────────────────────────────────────────


def elements_d_historique(historique: str) -> list[tuple[str, str]]:
    """(voie, texte) for each element of the History section, continuation included.

    The template renders each entry as "- <entry>"; the core titles are bare entries,
    their lines "- - …" entries. A long entry (the article, over several
    paragraphs) continues on the lines that do not start with "- ".
    """
    elements: list[list[str]] = []
    voie = "rappel"
    for ligne in (historique or "").splitlines():
        if ligne.startswith("- "):
            corps = ligne[2:]
            titre = SECTIONS.get(corps.strip())
            if titre:
                voie = titre
                continue
            if not corps.startswith("- "):
                # A first-level entry outside the core: a recalled episodic memory.
                voie = "rappel"
            elements.append([voie, corps])
        elif elements and ligne.strip():
            elements[-1][1] += "\n" + ligne
    return [(v, t) for v, t in elements]


def _journee_du_bloc(bloc: BlocPrompt) -> str | None:
    from scripts.analysis.mesures.calcul import journee_du_moment

    if not bloc.jour or bloc.depart_minutes is None:
        return None
    heures, minutes = divmod(bloc.depart_minutes, 60)
    return journee_du_moment(f"{bloc.jour} {heures:02d}:{minutes:02d}:00")


def _blocs_uniques(chemin_run: Path) -> list[BlocPrompt] | None:
    """One block per decision. `None` when the run has no exchange log."""
    if not (chemin_run / "llm_exchanges.jsonl").is_file():
        return None
    uniques: dict[tuple, BlocPrompt] = {}
    for bloc in blocs_itineraires(lire_echanges(chemin_run)):
        # The replay of a resume asks for the same decision again: the last request is the one
        # whose answer stays in `moves.csv`.
        uniques[(bloc.agent, bloc.jour, bloc.depart_minutes, bloc.motif)] = bloc
    return list(uniques.values())


# ── Assembly ──────────────────────────────────────────────────────────────────────────


@dataclass
class Souvenirs:
    lignes: list[LigneSouvenir]
    derives: list[SouvenirDerive]
    # (day, agent, event) → matching of the day, for `evenement_par_jour.csv`.
    du_jour: dict[tuple[str, str, str], tuple[int, int, str]]
    journal_present: bool


def mesurer(chemin_run: Path, evenements: Sequence[dict], journees: Sequence[Any],
            decisions_par_jour: dict[tuple[str, str], int], journee_de: Any) -> Souvenirs:
    chemin_run = Path(chemin_run)
    liste = suivis(chemin_run, evenements, journee_de)
    blocs = _blocs_uniques(chemin_run)
    if not liste:
        return Souvenirs([], [], {}, blocs is not None)

    par_agent_jour: dict[tuple[str, str], list[BlocPrompt]] = {}
    for bloc in blocs or []:
        journee = _journee_du_bloc(bloc)
        if journee:
            par_agent_jour.setdefault((bloc.agent, journee), []).append(bloc)

    index = {j.date: j.index for j in journees}
    lignes: list[LigneSouvenir] = []
    du_jour: dict[tuple[str, str, str], tuple[int, int, str]] = {}
    for suivi in liste:
        j0_index = _rang(suivi.j0, journees)
        for journee in journees:
            if journee.date < suivi.j0:
                continue
            prompts = par_agent_jour.get((suivi.person_id, journee.date), [])
            voies: Counter = Counter()
            avec = texte = mots = 0
            for bloc in prompts:
                trouve: dict[str, str] = {}
                for voie, contenu in elements_d_historique(bloc.historique):
                    appariement = suivi.signature.reconnait(contenu)
                    if appariement and (voie not in trouve or appariement == "texte"):
                        trouve[voie] = appariement
                if not trouve:
                    continue
                avec += 1
                texte += "texte" in trouve.values()
                mots += "texte" not in trouve.values()
                for voie in trouve:
                    voies[voie] += 1
            decisions = decisions_par_jour.get((journee.date, suivi.person_id), 0)
            lignes.append(LigneSouvenir(
                jour_simule=journee.index, date_simulee=journee.date,
                person_id=suivi.person_id, evenement_id=suivi.evenement_id,
                role=suivi.role, informe=suivi.informe,
                jours_depuis_j0=journee.index - j0_index if j0_index is not None else None,
                decisions=decisions, prompts=len(prompts),
                decisions_sans_prompt=max(0, decisions - len(prompts)),
                prompts_avec_souvenir=avec, souvenir_texte=texte, souvenir_mots=mots,
                via_connaissances=voies["connaissances"], via_changements=voies["changements"],
                via_rappel=voies["rappel"],
            ))
            du_jour[(journee.date, suivi.person_id, suivi.evenement_id)] = (
                len(prompts), avec, "texte" if texte else ("mots" if mots else ""))
        # The day of the reading may have no trip: it has no lived day, but
        # its matching is still requested by `evenement_par_jour.csv`.
        if suivi.j0 not in index:
            prompts = par_agent_jour.get((suivi.person_id, suivi.j0), [])
            du_jour.setdefault((suivi.j0, suivi.person_id, suivi.evenement_id),
                               (len(prompts), 0, ""))

    return Souvenirs(
        lignes=sorted(lignes, key=lambda l: (l.jour_simule, l.evenement_id,
                                             _cle(l.person_id))),
        derives=_derives(chemin_run, liste),
        du_jour=du_jour,
        journal_present=blocs is not None,
    )


def _rang(date_iso: str, journees: Sequence[Any]) -> int | None:
    from scripts.analysis.mesures.calcul import _indice_calendaire

    return _indice_calendaire(date_iso, journees)


def _derives(chemin_run: Path, liste: Sequence[Suivi]) -> list[SouvenirDerive]:
    """The long-term memory documents that carry the event — a snapshot, for the audit."""
    entrees, _ = lire_ltm(chemin_run)
    derives = []
    for suivi in liste:
        for x in entrees.get(suivi.person_id, []):
            if not _instant(x.timestamp) or _instant(x.timestamp) < suivi.instant:
                continue
            appariement = suivi.signature.reconnait(x.contenu)
            if not appariement:
                continue
            derives.append(SouvenirDerive(
                evenement_id=suivi.evenement_id, person_id=suivi.person_id, role=suivi.role,
                doc_id=x.doc_id, type_souvenir=x.memory_type,
                ecrit_le=_instant(x.timestamp), appariement=appariement,
                mots_retrouves=" ".join(temoin.mots_retrouves(suivi.signature.mots, [x.contenu])),
                enonce=_normaliser(x.enonce or x.contenu)[:160],
            ))
    return sorted(derives, key=lambda d: (d.evenement_id, _cle(d.person_id), d.ecrit_le,
                                          d.doc_id))


def _cle(person_id: str) -> tuple[int, int, str]:
    return (0, int(person_id), "") if person_id.isdigit() else (1, 0, person_id)
