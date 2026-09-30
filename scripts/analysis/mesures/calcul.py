"""What a simulated day says about a persona — ticket 093, lot 2.

Contract: `specs/ticket_093/tests.md`.

WHAT THIS MODULE MEASURES
-------------------------
Four families, and the first one conditions how the others are read:

1. **Modal choice**, with the share of trips ACTUALLY DECIDED. Without it, "car 100 %"
   means either a choice or the absence of an alternative — that is what made two of the
   five personas of run `2026-09-16_15_58` unreadable.
2. **Habit and break**, per ACTIVITY. The home-work trip and the leisure trip have
   no reason to switch together, and aggregating them would hide the change sought.
3. **Memory**: recall pool, concept operations, lifetime of memories.
4. **Shock**: exposed agents, injected minutes, and whether the shock memory is SERVED to the
   decision — a memory written but never recalled changes nothing in a behaviour. The follow-up
   of the memory in the household, day after day, lives in `souvenir.py` (analysis of 2026-09-25).

FLOW AND STATE, AND WHY THE DISTINCTION MATTERS
-----------------------------------------------
A **flow** is read from a dated log: it is recomputed identically, always. A **state** cannot
be reconstructed after the fact — a memory's `force` grows at each recall, so the median
computed today on the memories of day 2 is not the one they had on day 2. States
are therefore read from the **checkpoint** that closes the day, and stay empty as long as it
does not exist. Without this rule, a value already written would move retroactively at each
recomputation, and the continuity of the resume would be wrong where it looks right.

THE SIMULATED DAY
-----------------
It starts at **3 a.m.**, like the checkpoint of ticket 075: at that hour the buffers are
empty and almost nobody is on a trip. A return at 00:30 therefore belongs to the evening of the
day before.

TWO TIME MARKERS, AND THEY DO NOT SAY THE SAME THING
----------------------------------------------------
`moves.csv` carries two instants, and mixing them up skews everything:

- **`Temps simulé`** is the instant the DECISION was made. At bootstrap, the five agents of the
  reference run all carry `2026-03-16 05:00:01` there — the instant of `/init`.
- **`Heure de départ`** is the instant the TRIP leaves.

The gap is not a format detail. Measured on `2026-09-16_15_58`: 30 lines out of 270 set
the two markers more than an hour and a half apart, **19 of them by exactly 48 hours** — decisions
made on Saturday 21 March for departures postponed to Monday 23, the simulation moving
nobody at weekends.

Hence the rule, and it reads both ways:

| Use | Marker | What the other one would break |
|---|---|---|
| **Deduplicate** the replay | `Temps simulé` | two distinct decisions of one activity would merge |
| **Date the lived day** | `Heure de départ` | a "Saturday" day with 19 trips would appear, and Monday would lose 19 out of 35 |

⚠ The `Jour relatif au choc` column of `moves.csv` is **never** read here, on purpose:
it mixes two conventions if the shock configuration moved during the run. Days
relative to the shock are re-derived from the timestamps of `chocs.jsonl`.

And "the day before" is the previous **lived** day. Measured on the reference run: dates go
from 16 to 27 March skipping weekends, whose departures are postponed to Monday. Comparing with
the previous calendar day would empty the previous-day carry-over every Monday — one day in five —
for a weekend in which the agent lived nothing and so could carry nothing over.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from scripts.analysis.memoire.sources import MODES, Trajet, lire_ltm, lire_moves

# Hour at which the simulated day switches. Same boundary as the checkpoint
# (ticket 075): empty buffers, almost nobody on a trip.
FRONTIERE_JOUR_H = 3

# Habit window, in days ON WHICH THE ACTIVITY WAS OBSERVED — not in simulated days. An agent
# who only goes out one day in three thus has the same habit depth as the others, spread
# over more days; a window in simulated days would have told its regularity rather than its
# habit.
FENETRE_HABITUDE = 5

OPERATIONS_CONCEPT = ("créé", "confirmé", "précisé", "contredit")


# ── The simulated calendar ────────────────────────────────────────────────────────────


def journee_du_moment(horodatage: str) -> str | None:
    """Simulated day of an instant "YYYY-MM-DD HH:MM:SS", or of a date already per day.

    ⚠ **A bare date is not shifted.** Defect found while writing the tests: `sim_day` is
    "2026-03-16" in `trace_rappel.jsonl` and in the operations trace — it is ALREADY a
    day, not an instant. Applying the 3 a.m. boundary to it read it as midnight and
    moved it back one day: the whole recall pool and all concept operations
    ended up shifted by one day relative to the trips, with nothing flagging it.

    ⚠ Those days are dated by their producer in calendar UTC, not at the 3 a.m.
    boundary. The gap can only concern a recall occurring between midnight and 3 a.m.; it is
    stated here rather than corrected, for lack of the run's time zone at hand.

    `None` if the value is not a readable date — never a ten-character prefix picked at
    random from an arbitrary string.
    """
    texte = (horodatage or "").strip().replace("T", " ")
    if len(texte) < 10:
        return None
    if len(texte) == 10:
        try:
            return date.fromisoformat(texte).isoformat()
        except ValueError:
            return None
    try:
        quand = datetime.fromisoformat(texte[:19])
    except ValueError:
        try:
            return date.fromisoformat(texte[:10]).isoformat()
        except ValueError:
            return None
    return (quand - timedelta(hours=FRONTIERE_JOUR_H)).date().isoformat()


@dataclass(frozen=True)
class Journee:
    """A LIVED day: an index, a date, and the lived day before it."""

    index: int
    date: str
    veille: str | None


def journees_vecues(trajets: Sequence[Trajet]) -> list[Journee]:
    """The days on which the run produced trips, indexed from the first one.

    The index counts calendar days since the first lived day — it is the same
    `jour_simule` as the checkpoints', so the two cross-check. Days without a
    trip have no line: a line of zeros would read as a motionless day.
    """
    dates = sorted({d for d in (journee_du_moment(t.depart_complet) for t in trajets) if d})
    if not dates:
        return []
    origine = datetime.fromisoformat(dates[0]).date()
    return [
        Journee(
            index=(datetime.fromisoformat(date).date() - origine).days + 1,
            date=date,
            veille=dates[rang - 1] if rang else None,
        )
        for rang, date in enumerate(dates)
    ]


# ── Measurement lines ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LigneChoixModal:
    jour_simule: int
    date_simulee: str
    person_id: str
    trajets: int
    trajets_decides: int
    part_decidee: float
    modes_distincts: int
    parts: dict[str, float]
    trajets_mode_inconnu: int


@dataclass(frozen=True)
class LigneHabitude:
    jour_simule: int
    date_simulee: str
    person_id: str
    activite: str
    motif: str
    mode: str | None
    occurrences: int
    mode_veille: str | None
    reprise_veille: bool | None
    mode_habituel: str | None
    conforme_habitude: bool | None
    observations_fenetre: int


@dataclass(frozen=True)
class LigneMemoire:
    jour_simule: int
    date_simulee: str
    person_id: str
    rappels: int
    vivier_min: int | None
    vivier_median: float | None
    vivier_max: int | None
    souvenirs_servis: int
    operations: dict[str, int | None]
    operations_hors_vocabulaire: int
    etat_lu_dans: str | None
    entrees_ltm: int | None


@dataclass(frozen=True)
class LigneDureeDeVie:
    jour_simule: int
    date_simulee: str
    person_id: str
    type_souvenir: str
    entrees: int
    duree_vie_mediane_jours: float
    etat_lu_dans: str


@dataclass(frozen=True)
class LigneChoc:
    jour_simule: int
    date_simulee: str
    person_id: str
    choc_id: str
    # Ticket 100 — through which CHANNEL the event came in: `vecu` (undergone on arrival, after
    # the decision) or `lu` (learnt at wake-up, before deciding). Both regimes are measured with
    # the same function and read in the same file, which is the whole point of the ticket;
    # but they are not mixed up, and a column separates them. Empty for earlier runs.
    canal: str
    expositions: int
    minutes_injectees: float
    incidents_reseau: int
    correspondances_ratees: int
    souvenir_choc_servi: bool | None
    decisions_avec_souvenir_choc: int | None
    appariement: str


@dataclass
class Mesures:
    chemin_run: Path
    journees: list[Journee] = field(default_factory=list)
    choix_modal: list[LigneChoixModal] = field(default_factory=list)
    habitudes: list[LigneHabitude] = field(default_factory=list)
    memoire: list[LigneMemoire] = field(default_factory=list)
    durees_de_vie: list[LigneDureeDeVie] = field(default_factory=list)
    chocs: list[LigneChoc] = field(default_factory=list)
    souvenirs: list = field(default_factory=list)
    souvenirs_derives: list = field(default_factory=list)
    trajets_rejoues: int = 0
    rappels_rejoues: int = 0
    operations_tracees: bool = False


# ── Reading the sources ───────────────────────────────────────────────────────────────


def _jsonl(chemin: Path) -> list[dict[str, Any]]:
    """Tolerant JSONL: an unreadable line is skipped, never fatal."""
    if not chemin.is_file():
        return []
    lignes = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            objet = json.loads(ligne)
        except json.JSONDecodeError:
            continue
        if isinstance(objet, dict):
            lignes.append(objet)
    return lignes


def rappels_sans_rejeu(lignes: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Recall trace stripped of the lines REWRITTEN during the replay of a resume.

    ⚠ `trace_rappel` is not frozen during the replay, unlike the memory log.
    Measured on `2026-09-16_15_58`: key `(1773637201, 41275)` carries two lines — the first
    with **0 candidates**, the live run's on the first day, the second with **22**, the replay's,
    which sees memory frozen in the restored state. The pattern is clear-cut: 13 duplicates a day
    until 24 March, 1 on the 25th, 0 after — exactly the end of the freeze window.

    Without this deduplication, the growth of the pool — the very object of the measurement — is
    drowned under the frozen state: the pool capped at 30 from the first day.

    Same rule as for `moves.csv`: key `(simulated instant, person)`, the FIRST line
    wins, the following ones are counted. A line without a simulated instant cannot be deduplicated
    and goes through as is: we do not merge what we cannot tell apart.
    """
    vues: set[tuple[Any, str]] = set()
    gardees: list[dict[str, Any]] = []
    rejeu = 0
    for ligne in lignes:
        instant = ligne.get("sim_ts")
        clef = (instant, str(ligne.get("person_id") or ""))
        if instant is None:
            gardees.append(ligne)
            continue
        if clef in vues:
            rejeu += 1
            continue
        vues.add(clef)
        gardees.append(ligne)
    return gardees, rejeu


def _points_de_reprise(chemin_run: Path) -> dict[str, Path]:
    """Day CLOSED by each checkpoint → directory of the checkpoint.

    A checkpoint written on 17 March at 3 a.m. closes the day of the 16th: the closed day is that
    of the instant immediately before the checkpoint.
    """
    racine = Path(chemin_run) / "checkpoints_memoire"
    par_journee: dict[str, Path] = {}
    if not racine.is_dir():
        return par_journee
    for dossier in sorted(racine.glob("jour_*")):
        meta = dossier / "reprise.json"
        if not meta.is_file():
            continue
        try:
            charge = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        horodatage = str(charge.get("horodatage_simule") or "")
        if not horodatage:
            continue
        try:
            instant = datetime.fromisoformat(horodatage.replace("T", " ")[:19])
        except ValueError:
            continue
        journee = journee_du_moment((instant - timedelta(seconds=1)).isoformat())
        if journee:
            par_journee[journee] = dossier
    return par_journee


# ── Modal choice ──────────────────────────────────────────────────────────────────────


def choix_modal(trajets_par_jour: dict[tuple[str, str], list[Trajet]],
                journees: Sequence[Journee]) -> list[LigneChoixModal]:
    index = {j.date: j for j in journees}
    lignes = []
    for (date, person_id), trajets in sorted(trajets_par_jour.items()):
        journee = index.get(date)
        if journee is None:
            continue
        decides = sum(1 for t in trajets if (t.options_presentees or 0) >= 2)
        modes = Counter(t.mode for t in trajets if t.mode)
        lignes.append(LigneChoixModal(
            jour_simule=journee.index,
            date_simulee=date,
            person_id=person_id,
            trajets=len(trajets),
            trajets_decides=decides,
            part_decidee=decides / len(trajets),
            modes_distincts=len(modes),
            parts={mode: modes.get(mode, 0) / len(trajets) for mode in MODES},
            # A mode the vocabulary does not know is not silently poured into
            # "others": it is counted here, so that a modal share that does not sum to 1
            # shows instead of catching up on its own.
            trajets_mode_inconnu=sum(1 for t in trajets if not t.mode),
        ))
    return sorted(lignes, key=lambda l: (l.jour_simule, _cle_agent(l.person_id)))


# ── Habit and break ───────────────────────────────────────────────────────────────────


def habitudes(trajets_par_jour: dict[tuple[str, str], list[Trajet]],
              journees: Sequence[Journee]) -> list[LigneHabitude]:
    """One line per agent, per activity and per lived day.

    Two measures, because they do not detect the same thing: the previous-day carry-over sees
    a clear switch on the day it happens; conformity to habit sees a slow drift
    that a single day does not reveal.
    """
    index = {j.date: j for j in journees}
    # (agent, activity) → [(date, mode)] in the order of lived days
    historique: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    lignes: list[LigneHabitude] = []

    for journee in journees:
        agents_du_jour = sorted(
            (person_id for (date, person_id) in trajets_par_jour if date == journee.date),
            key=_cle_agent,
        )
        for person_id in agents_du_jour:
            par_activite = _par_activite(trajets_par_jour[(journee.date, person_id)])
            for activite, trajets in sorted(par_activite.items()):
                premier = trajets[0]
                clef = (person_id, activite)
                passe = historique.get(clef, [])
                mode_veille = next(
                    (mode for date, mode in reversed(passe) if date == journee.veille), None
                )
                # The window covers the last five OBSERVATIONS of this activity, not
                # the last five simulated days: an agent who only goes out one day in three thus
                # has the same habit depth, simply spread over more days.
                fenetre = [mode for _, mode in passe[-FENETRE_HABITUDE:] if mode]
                habituel = _majoritaire(fenetre)
                lignes.append(LigneHabitude(
                    jour_simule=journee.index,
                    date_simulee=journee.date,
                    person_id=person_id,
                    activite=activite,
                    motif=premier.motif,
                    mode=premier.mode,
                    occurrences=len(trajets),
                    mode_veille=mode_veille,
                    reprise_veille=(premier.mode == mode_veille) if mode_veille else None,
                    mode_habituel=habituel,
                    conforme_habitude=(premier.mode == habituel) if habituel else None,
                    observations_fenetre=len(fenetre),
                ))
                historique.setdefault(clef, []).append((journee.date, premier.mode))
    return sorted(lignes, key=lambda l: (l.jour_simule, _cle_agent(l.person_id), l.activite))


def _par_activite(trajets: Sequence[Trajet]) -> dict[str, list[Trajet]]:
    """Trips of one day grouped by activity, each group sorted by departure time.

    Several trips of one activity on the same day: the mode kept is that of the FIRST
    departure, and the number of occurrences is written (question 4 of the ticket).
    """
    par_activite: dict[str, list[Trajet]] = {}
    for trajet in sorted(trajets, key=lambda t: (t.depart, t.trajet_id)):
        par_activite.setdefault(trajet.activity_id, []).append(trajet)
    return par_activite


def _majoritaire(modes: Sequence[str]) -> str | None:
    """The strict majority mode of the window, or `None`.

    ⚠ A tie returns `None`, never a mode drawn at random nor the first one found: breaking a
    tie would fabricate a habit the agent does not have, and the conformity measured then
    against it would mean nothing.
    """
    if not modes:
        return None
    comptes = Counter(modes).most_common()
    if len(comptes) > 1 and comptes[0][1] == comptes[1][1]:
        return None
    return comptes[0][0]


# ── Memory ────────────────────────────────────────────────────────────────────────────


def memoire(
    chemin_run: Path, journees: Sequence[Journee], agents: Iterable[str],
    points: dict[str, Path],
) -> tuple[list[LigneMemoire], list[LigneDureeDeVie], bool, int]:
    rappels, rappels_rejoues = rappels_sans_rejeu(
        _jsonl(Path(chemin_run) / "trace_rappel.jsonl"))
    operations = _jsonl(Path(chemin_run) / "operations_concept.jsonl")
    tracees = (Path(chemin_run) / "operations_concept.jsonl").is_file()

    par_jour_agent: dict[tuple[str, str], list[dict]] = {}
    for ligne in rappels:
        journee = journee_du_moment(str(ligne.get("sim_day") or ""))
        agent = str(ligne.get("person_id") or "")
        if journee and agent:
            par_jour_agent.setdefault((journee, agent), []).append(ligne)

    ops_par_jour_agent: dict[tuple[str, str], Counter] = {}
    hors_vocabulaire: Counter = Counter()
    for ligne in operations:
        journee = journee_du_moment(str(ligne.get("sim_day") or ""))
        agent = str(ligne.get("person_id") or "")
        if not (journee and agent):
            continue
        operation = str(ligne.get("operation") or "")
        if operation in OPERATIONS_CONCEPT:
            ops_par_jour_agent.setdefault((journee, agent), Counter())[operation] += 1
        else:
            # An operation outside the four is not filed under one of them: it is
            # counted separately, otherwise a vocabulary that changes silently would read as a
            # change of behaviour.
            hors_vocabulaire[(journee, agent)] += 1

    etats = {date: _etat_memoire(dossier) for date, dossier in points.items()}

    lignes: list[LigneMemoire] = []
    durees: list[LigneDureeDeVie] = []
    for journee in journees:
        etat = etats.get(journee.date)
        for agent in sorted(agents, key=_cle_agent):
            traces = par_jour_agent.get((journee.date, agent), [])
            viviers = [int(t.get("candidats") or 0) for t in traces]
            ops = ops_par_jour_agent.get((journee.date, agent))
            lignes.append(LigneMemoire(
                jour_simule=journee.index,
                date_simulee=journee.date,
                person_id=agent,
                rappels=len(traces),
                vivier_min=min(viviers) if viviers else None,
                vivier_median=median(viviers) if viviers else None,
                vivier_max=max(viviers) if viviers else None,
                souvenirs_servis=sum(len(t.get("servis") or []) for t in traces),
                # Trace switched off: the columns are EMPTY, not zero. "No contradiction"
                # and "contradictions are not measured" must not be quoted the same way.
                operations={
                    nom: ((ops or Counter()).get(nom, 0) if tracees else None)
                    for nom in OPERATIONS_CONCEPT
                },
                operations_hors_vocabulaire=hors_vocabulaire.get((journee.date, agent), 0),
                etat_lu_dans=points[journee.date].name if journee.date in points else None,
                entrees_ltm=len(etat.get(agent, [])) if etat is not None else None,
            ))
            if etat is None:
                continue
            par_type: dict[str, list[float]] = {}
            for entree in etat.get(agent, []):
                par_type.setdefault(entree.memory_type or "inconnu", []).append(entree.force)
            for type_souvenir, forces in sorted(par_type.items()):
                durees.append(LigneDureeDeVie(
                    jour_simule=journee.index,
                    date_simulee=journee.date,
                    person_id=agent,
                    type_souvenir=type_souvenir,
                    entrees=len(forces),
                    duree_vie_mediane_jours=round(median(forces), 2),
                    etat_lu_dans=points[journee.date].name,
                ))
    return lignes, durees, tracees, rappels_rejoues


def _etat_memoire(dossier: Path) -> dict[str, list]:
    entrees, _habitudes = lire_ltm(dossier)
    return entrees


# ── Shock ─────────────────────────────────────────────────────────────────────────────


def chocs(chemin_run: Path, journees: Sequence[Journee],
          decisions_par_jour: dict[tuple[str, str], int] | None = None) -> list[LigneChoc]:
    """Exposed agents and injected minutes, per exposure day. See `chocs_et_souvenirs`."""
    return chocs_et_souvenirs(chemin_run, journees, decisions_par_jour)[0]


def chocs_et_souvenirs(
    chemin_run: Path, journees: Sequence[Journee],
    decisions_par_jour: dict[tuple[str, str], int] | None = None,
) -> tuple[list[LigneChoc], Any]:
    """`evenement_par_jour` lines, and the memory follow-up in the household (`souvenir.py`).

    ⚠ **"Served" is read in the prompt, no longer in `trace_rappel`** (2026-09-25). The recall
    trace counts what the search brought up, not what the model read: the core does not appear
    in it, and when it is present only two or three episodic memories survive. And the link sought
    was the literal text of the injection, which consolidation always rewords — the
    memory of article a09 was therefore never found, and six prompts that carried it
    gave 0.

    The two memory columns describe the DAY of the exposure — which keeps a line
    of a past day still when the run moves on. What happens next, day after day and
    for the whole household, is in `souvenir_evenement_par_jour.csv`.

    `appariement` says how the memory was found in the day's prompts:
    `texte` (the article word for word), `mots` (a rewording), `aucune trace` (prompts,
    none carries it: 0 measured), `sans prompt` (no prompt that day, or no exchange
    log: we do not know, and the cell stays empty).
    """
    from scripts.analysis.mesures import souvenir

    # Ticket 100 — `evenements.jsonl` is the new name; `chocs.jsonl` is still read for archived
    # runs, and that is where the published figures of § 7.2 live. The first one that
    # exists is read, never both: in a new run, the second is a link to the first.
    tous = _jsonl(Path(chemin_run) / "evenements.jsonl")
    if not tous:
        tous = _jsonl(Path(chemin_run) / "chocs.jsonl")
    # Ticket 111: the line of an informed member (`origine: entendu`) is not an exposure.
    # What it received is read in `relais_foyer.jsonl` and through the co-resident sub-role.
    evenements = [e for e in tous if e.get("origine") != "entendu"]
    if not evenements:
        return [], None
    index = {j.date: j for j in journees}
    suivi = souvenir.mesurer(
        Path(chemin_run), tous, journees, decisions_par_jour or {},
        lambda e: journee_de_l_exposition(e, journees, bavard=False),
    )

    groupes: dict[tuple[str, str, str], list[dict]] = {}
    for evenement in evenements:
        journee = journee_de_l_exposition(evenement, journees)
        agent = str(evenement.get("person_id") or "")
        if journee and agent:
            # `evenement_id` is the new name, `choc_id` the one from 079: both are written
            # side by side in a new run, and only the second in an archived run.
            identifiant = str(
                evenement.get("evenement_id") or evenement.get("choc_id") or ""
            )
            groupes.setdefault((journee, agent, identifiant), []).append(evenement)

    lignes = []
    for (date, agent, choc_id), liste in sorted(groupes.items()):
        # An article is read at wake-up, whether or not the agent travels that day: the exposure
        # keeps its line even on a day without a trip, indexed from the first lived day.
        indice = index[date].index if date in index else _indice_calendaire(date, journees)
        if indice is None:
            continue
        prompts, avec, trouve = suivi.du_jour.get((date, agent, choc_id), (0, 0, ""))
        if not suivi.journal_present or not prompts:
            decisions, appariement = None, "sans prompt"
        else:
            decisions, appariement = avec, (trouve or "aucune trace")
        lignes.append(LigneChoc(
            jour_simule=indice,
            date_simulee=date,
            person_id=agent,
            choc_id=choc_id,
            # Empty — never `vecu` — for an archived run that did not carry the field: writing
            # a modality that was not measured would turn it into a measurement.
            canal=str(liste[0].get("canal") or ""),
            expositions=len(liste),
            minutes_injectees=round(
                sum(float(e.get("retard_injecte_s") or 0) for e in liste) / 60.0, 2),
            incidents_reseau=sum(1 for e in liste if e.get("incident_reseau")),
            correspondances_ratees=sum(1 for e in liste if e.get("correspondance_ratee")),
            souvenir_choc_servi=(decisions > 0) if decisions is not None else None,
            decisions_avec_souvenir_choc=decisions,
            appariement=appariement,
        ))
    return lignes, suivi


def journee_de_l_exposition(evenement: dict, journees: Sequence[Journee],
                             bavard: bool = True) -> str | None:
    """Simulated day of an exposure, depending on the moment the event comes in.

    ⚠ **An article read at wake-up belongs to the day it is read.** The registry injects it at
    the first step after midnight, timestamped `YYYY-MM-DDT00:00:00`. The 3 a.m. boundary, right
    for a trip, moved it back one day: article a09 of run `2026-09-24_17_50`, read on 26 March
    (`jour_run: 11`), came out on day 10. The calendar day of the timestamp is the
    registry's (`ancre_run.jours_ecoules` counts from midnight); it is cross-checked with
    `jour_run` when the line carries it, and a gap says so instead of settling silently.

    A shock undergone on arrival (`moment: arrivee`, or absent in a 079 run) keeps the
    3 a.m. boundary: it is attached to the trip it accompanies, and must fall on the same day as
    that trip. That is what leaves the archived figures of § 7.2 unchanged.
    """
    horodatage = str(evenement.get("horodatage_simule") or "")
    if evenement.get("moment") != "reveil":
        return journee_du_moment(horodatage)
    date_lue = journee_du_moment(horodatage[:10])
    jour_run = evenement.get("jour_run")
    indice = _indice_calendaire(date_lue, journees) if date_lue else None
    if bavard and isinstance(jour_run, int) and indice is not None and indice != jour_run:
        print(
            f"  ⚠ exposition de {evenement.get('person_id')} datée du {date_lue} (jour {indice} "
            f"des mesures) mais `jour_run: {jour_run}` dans evenements.jsonl — la première "
            f"journée vécue n'est pas le premier jour du run ; la date de l'horodatage est gardée."
        )
    return date_lue


def _indice_calendaire(date_iso: str, journees: Sequence[Journee]) -> int | None:
    """Calendar rank of a date since the first lived day, even without a trip on that day."""
    if not journees:
        return None
    origine = date.fromisoformat(journees[0].date) - timedelta(days=journees[0].index - 1)
    try:
        return (date.fromisoformat(date_iso) - origine).days + 1
    except ValueError:
        return None


# ── Assembly ──────────────────────────────────────────────────────────────────────────


def _cle_agent(person_id: str) -> tuple[int, int, str]:
    return (0, int(person_id), "") if person_id.isdigit() else (1, 0, person_id)


def calculer(chemin_run: Path | str) -> Mesures:
    """All measurements of a run, recomputed from scratch.

    Recomputing in full rather than completing is what makes continuity true by
    construction: a day replayed after a resume cannot double since it starts again from
    the same deduplicated source, and a cut cannot leave a gap in the curve.
    """
    chemin_run = Path(chemin_run)
    trajets, rejeu, _detail = lire_moves(chemin_run)
    journees = journees_vecues(trajets)

    par_jour: dict[tuple[str, str], list[Trajet]] = {}
    for trajet in trajets:
        date = journee_du_moment(trajet.depart_complet)
        if date:
            par_jour.setdefault((date, trajet.person_id), []).append(trajet)

    agents = sorted({t.person_id for t in trajets}, key=_cle_agent)
    points = _points_de_reprise(chemin_run)
    lignes_memoire, durees, tracees, rappels_rejoues = memoire(
        chemin_run, journees, agents, points)
    decisions_par_jour = {
        cle: sum(1 for t in liste if (t.options_presentees or 0) >= 2)
        for cle, liste in par_jour.items()
    }
    lignes_chocs, suivi = chocs_et_souvenirs(chemin_run, journees, decisions_par_jour)
    return Mesures(
        chemin_run=chemin_run,
        journees=journees,
        choix_modal=choix_modal(par_jour, journees),
        habitudes=habitudes(par_jour, journees),
        memoire=lignes_memoire,
        durees_de_vie=durees,
        chocs=lignes_chocs,
        souvenirs=suivi.lignes if suivi else [],
        souvenirs_derives=suivi.derives if suivi else [],
        trajets_rejoues=rejeu,
        rappels_rejoues=rappels_rejoues,
        operations_tracees=tracees,
    )
