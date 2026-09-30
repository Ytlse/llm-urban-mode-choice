"""Reading of the files of a memory run.

This module measures nothing: it returns clean and **sorted** structures, so that
the report is reproducible byte for byte (contract F7).

Three pitfalls of the format, handled here once and for all:

1. ``llm_exchanges.jsonl`` is **not** JSONL despite the extension: it is
   concatenated indented JSON objects. It is read with ``json.JSONDecoder.raw_decode``
   in a loop (contract F2).
2. ``moves.csv`` contains **replayed trips** after a restart: the same
   decision (same person, same activity, same simulated instant) appears twice
   in it. The first one per computation time is kept and the others are counted
   (contract F1).
3. The itinerary options proposed to the agent **exist only in the text of the
   prompts**. They are extracted from it; the matching lives in :mod:`mesures`.
"""
from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Mode vocabulary ───────────────────────────────────────────────────────────
# Official palette of the repository (`.claude/CLAUDE.md`): car red, bike purple,
# public transport green, walking cyan, train purple, motor scooters magenta.
MODE_COULEURS: dict[str, str] = {
    "car": "#EE4444",
    "cycling": "#8844BB",
    "public_transport": "#22AA44",
    "walking": "#00CCCC",
    "train": "#8844BB",
    "motorbike": "magenta",
    "autres": "#9AA0A6",
}

MODE_LIBELLES: dict[str, str] = {
    "car": "Voiture",
    "cycling": "Vélo",
    "public_transport": "Transports collectifs",
    "walking": "Marche",
    "train": "Train",
    "motorbike": "Deux-roues motorisé",
    "autres": "Autres modes",
}

# Stable display order, independent of the data.
MODES = ["car", "public_transport", "walking", "cycling", "train", "motorbike", "autres"]

# French labels of `moves.csv` → canonical modes.
_MODE_DEPUIS_MOVES = {
    "Voiture Privée": "car",
    "Vélo": "cycling",
    "Transports_collectifs": "public_transport",
    "Marche": "walking",
    "Train": "train",
    "Deux-roues motorisé": "motorbike",
    "Autres modes": "autres",
}

# Probability columns of `moves.csv` → canonical modes.
COLONNES_PROBA = {
    "P(Marche) %": "walking",
    "P(Vélo) %": "cycling",
    "P(Voiture Privée) %": "car",
    "P(Transports_collectifs) %": "public_transport",
    "P(Train) %": "train",
    "P(Deux-roues motorisé) %": "motorbike",
    "P(Autres modes) %": "autres",
}

# Purposes: `moves.csv` mixes French and English, the prompts are in English.
_MOTIF_CANON = {
    "travail": "work",
    "work": "work",
    "etude": "education",
    "étude": "education",
    "education": "education",
    "achats": "shop",
    "shop": "shop",
    "shopping": "shop",
    "loisir": "leisure",
    "loisirs": "leisure",
    "leisure": "leisure",
    "home": "home",
    "domicile": "home",
    "escort": "escort",
    "accompagnement": "escort",
}

MOTIF_LIBELLES = {
    "work": "Travail",
    "education": "Études",
    "shop": "Achats",
    "leisure": "Loisirs",
    "home": "Retour au domicile",
    "escort": "Accompagnement",
}

# Leg labels found in the prompt options → canonical mode.
# The first word recognised in the order below wins: a « foot,bus,foot »
# is a public transport trip, not a walk.
_JAMBES_PRIORITAIRES = [
    (("rail", "train"), "train"),
    (("bus", "metro", "tram", "subway", "school_bus", "ferry", "trolleybus"), "public_transport"),
    (("car", "car_park", "taxi"), "car"),
    (("motorbike", "moped", "scooter"), "motorbike"),
    (("bicycle", "bike", "cycling"), "cycling"),
    (("foot", "walk", "walking"), "walking"),
]

_UNITES_DUREE = {
    "second": 1, "seconds": 1,
    "minute": 60, "minutes": 60,
    "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
}


def mode_canonique_depuis_moves(libelle: str) -> str | None:
    """French label of `moves.csv` → canonical mode. Outside the vocabulary: ``None``."""
    return _MODE_DEPUIS_MOVES.get((libelle or "").strip())


def mode_canonique_depuis_option(mode_option: str) -> str | None:
    """Leg sequence of a prompt option (« foot,bus,foot ») → canonical mode.

    A word outside both vocabularies stays ``None``: no guessing, it is declared.
    """
    jambes = [j.strip().lower() for j in (mode_option or "").split(",") if j.strip()]
    if not jambes:
        return None
    for mots, canon in _JAMBES_PRIORITAIRES:
        if any(j in mots for j in jambes):
            return canon
    return None


def motif_canonique(libelle: str) -> str:
    """Trip purpose, brought back to the English vocabulary of the prompts."""
    brut = (libelle or "").strip()
    return _MOTIF_CANON.get(brut.lower(), brut.lower() or "inconnu")


def libelle_motif(motif: str) -> str:
    return MOTIF_LIBELLES.get(motif, motif or "inconnu")


def duree_en_secondes(texte: str) -> int | None:
    """« 1 hour, 56 minutes » → 6960. ``None`` if no duration is readable."""
    couples = re.findall(
        r"(\d+)\s+(seconds?|minutes?|hours?|days?)", texte or "", flags=re.IGNORECASE)
    if not couples:
        return None
    return sum(int(n) * _UNITES_DUREE[u.lower()] for n, u in couples)


# ── Structures ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Trajet:
    person_id: str
    activity_id: str
    trajet_id: str
    mode: str | None
    modes_proposes: tuple[str, ...]
    probabilites: dict[str, float]
    motif: str
    methode: str
    jour: str                 # « AAAA-MM-JJ » of the simulated departure
    depart: str               # « HH:MM:SS »
    depart_minutes: int | None
    heure_calcul: str
    sim_ts: str
    distance_km: float | None
    temperature: float | None
    meteo: str
    raisonnement: str
    fournisseur: str
    # Ticket 093 — number of options ACTUALLY presented. Without it, "car 100 %" does not
    # tell a choice from the absence of an alternative, and a decision that was not one
    # counts as a decision. `None` when the column is missing or empty: a trip
    # whose number of options is unknown does not prove there was a choice, nor does it prove
    # the opposite — zero would be a lie in one direction as in the other.
    options_presentees: int | None = None

    @property
    def depart_complet(self) -> str:
        """« AAAA-MM-JJ HH:MM:SS » — the departure instant, day and time together."""
        return f"{self.jour} {self.depart}".strip()


@dataclass(frozen=True)
class Option:
    index: int
    mode_brut: str
    mode: str | None
    description: str
    etapes: tuple[str, ...]

    @property
    def identite(self) -> tuple[str, tuple[str, ...]]:
        """Identity of an itinerary: its mode and the sequence of its steps."""
        return (self.mode_brut, self.etapes)


@dataclass(frozen=True)
class BlocPrompt:
    agent: str
    jour: str
    motif: str
    depart_minutes: int | None
    options: tuple[Option, ...]
    a_historique: bool
    taille_historique: int
    # The text of the `**History:**` section, as the model read it. It is this text — not the
    # recall trace, wider than what the prompt contains when the core is present —
    # that says whether a memory was SERVED to a decision (`scripts/analysis/mesures/souvenir.py`).
    historique: str = ""


@dataclass(frozen=True)
class EntreeLTM:
    person_id: str
    doc_id: str
    contenu: str
    enonce: str
    memory_type: str
    timestamp: str
    importance: float
    valence: str
    axe_objet: str | None
    axe_motif: str | None
    axe_lieu: str | None
    axe_creneau: str | None
    axe_meteo: str | None
    force: float
    rappels: int
    dernier_rappel: str | None
    observations: int
    contre_exemples: int


@dataclass
class Journal:
    """What the readable log `memoires/<id>.md` tells."""
    person_id: str
    consolidations: int = 0
    declencheurs: dict[str, int] = field(default_factory=dict)
    operations: dict[str, int] = field(default_factory=dict)
    rappels: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)


@dataclass
class Run:
    chemin: Path
    trajets: list[Trajet]
    rejeu: int
    rejeu_detail: list[tuple[str, str, str]]
    arrivees: list[dict[str, Any]]
    evenements: list[dict[str, Any]]
    echanges: list[dict[str, Any]]
    blocs: list[BlocPrompt]
    ltm: dict[str, list[EntreeLTM]]
    habitudes: dict[str, dict[str, Any]]
    journaux: dict[str, Journal]
    personas: dict[str, dict[str, Any]]
    autoreflexions: dict[str, list[tuple[str, str]]]

    @property
    def jours(self) -> list[str]:
        return sorted({t.jour for t in self.trajets})

    @property
    def agents(self) -> list[str]:
        """Agent identifiers, sorted numerically when possible."""
        ids = set(self.personas) | {t.person_id for t in self.trajets} | set(self.ltm)
        return sorted(ids, key=_cle_agent)

    @property
    def fournisseurs(self) -> list[str]:
        return sorted({str(e.get("provider") or "") for e in self.echanges if e.get("provider")})


def _cle_agent(identifiant: str) -> tuple[int, Any]:
    return (0, int(identifiant)) if identifiant.isdigit() else (1, identifiant)


# ── moves.csv ─────────────────────────────────────────────────────────────────

def _flottant(valeur: Any) -> float | None:
    try:
        return float(str(valeur).replace(",", "."))
    except (TypeError, ValueError):
        return None


def lire_moves(chemin_run: Path) -> tuple[list[Trajet], int, list[tuple[str, str, str]]]:
    """Trips of the run, **replay excluded** (contract F1).

    Decision key: ``(ID Personne, ID Activité, Temps simulé)`` — the activity
    identifier alone repeats from one day to the next (same schedule), and
    the simulated instant tells apart two real trips of a same purpose in the day.
    After a restart, the same decision is recomputed: two lines, two
    computation times. The **first one per computation time** is kept; the others
    are replay, counted and declared.
    """
    fichier = Path(chemin_run) / "moves.csv"
    if not fichier.is_file():
        return [], 0, []
    with fichier.open(newline="", encoding="utf-8") as flux:
        lignes = list(csv.DictReader(flux))

    groupes: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = {}
    for rang, ligne in enumerate(lignes):
        instant = (ligne.get("Temps simulé") or "").strip()
        if not instant:
            # Fallback if the column is missing: the day of the simulated departure.
            instant = (ligne.get("Heure de départ") or "")[:10]
        cle = ((ligne.get("ID Personne") or "").strip(),
               (ligne.get("ID Activité") or "").strip(),
               instant)
        groupes.setdefault(cle, []).append((rang, ligne))

    trajets: list[Trajet] = []
    rejeu = 0
    rejeu_detail: list[tuple[str, str, str]] = []
    for cle in sorted(groupes):
        membres = sorted(groupes[cle],
                         key=lambda item: ((item[1].get("Heure de calcul") or ""), item[0]))
        garde = membres[0][1]
        for _, exclu in membres[1:]:
            rejeu += 1
            rejeu_detail.append((cle[0], (exclu.get("Heure de départ") or "")[:10],
                                 exclu.get("Heure de calcul") or ""))
        trajets.append(_trajet_depuis_ligne(garde))

    trajets.sort(key=lambda t: (_cle_agent(t.person_id), t.jour, t.depart, t.trajet_id))
    rejeu_detail.sort()
    return trajets, rejeu, rejeu_detail


def _trajet_depuis_ligne(ligne: dict[str, str]) -> Trajet:
    depart = (ligne.get("Heure de départ") or "").strip()
    jour, _, heure = depart.partition(" ")
    depart_minutes = None
    if len(heure) >= 5 and heure[:2].isdigit() and heure[3:5].isdigit():
        depart_minutes = int(heure[:2]) * 60 + int(heure[3:5])
    probabilites = {}
    for colonne, mode in COLONNES_PROBA.items():
        valeur = _flottant(ligne.get(colonne))
        if valeur:
            probabilites[mode] = valeur
    proposes = tuple(
        m for m in (
            mode_canonique_depuis_moves(x) or x.strip()
            for x in (ligne.get("Modes proposés au LLM") or "").split("|")
        ) if m
    )
    return Trajet(
        person_id=(ligne.get("ID Personne") or "").strip(),
        activity_id=(ligne.get("ID Activité") or "").strip(),
        trajet_id=(ligne.get("ID Trajet") or "").strip(),
        mode=mode_canonique_depuis_moves(ligne.get("Mode de transport Choisi", "")),
        modes_proposes=proposes,
        probabilites=probabilites,
        motif=motif_canonique(ligne.get("Motifs de déplacement", "")),
        methode=(ligne.get("Méthode de sélection") or "").strip(),
        jour=jour,
        depart=heure,
        depart_minutes=depart_minutes,
        heure_calcul=(ligne.get("Heure de calcul") or "").strip(),
        sim_ts=(ligne.get("Temps simulé") or "").strip(),
        distance_km=_flottant(ligne.get("Distance parcourue")),
        temperature=_flottant(ligne.get("Météo Température (°C)")),
        meteo=(ligne.get("Météo Condition") or "").strip(),
        raisonnement=(ligne.get("Raisonnement") or "").strip(),
        fournisseur=(ligne.get("Fournisseur & Modèle") or "").strip(),
        options_presentees=_entier(ligne.get("Options présentées")),
    )


def _entier(valeur: Any) -> int | None:
    try:
        return int(str(valeur).strip())
    except (TypeError, ValueError):
        return None


# ── gama_arrivals.csv ─────────────────────────────────────────────────────────

def lire_arrivees(chemin_run: Path) -> list[dict[str, Any]]:
    fichier = Path(chemin_run) / "gama_results" / "gama_arrivals.csv"
    if not fichier.is_file():
        return []
    with fichier.open(newline="", encoding="utf-8") as flux:
        lignes = list(csv.DictReader(flux))
    arrivees = []
    for ligne in lignes:
        debut, fin = ligne.get("started_at", ""), ligne.get("arrive_at", "")
        if not str(debut).strip().lstrip("-").isdigit() or not str(fin).strip().lstrip("-").isdigit():
            continue
        arrivees.append({
            "move_id": ligne.get("move_id", ""),
            "person_id": (ligne.get("person_id") or "").strip(),
            "started_at": int(debut),
            "arrive_at": int(fin),
            "delay_s": int(_flottant(ligne.get("delay_s")) or 0),
            "departure_delay_s": int(_flottant(ligne.get("departure_delay_s")) or 0),
            "timed_out": str(ligne.get("timed_out", "")).strip().lower() == "true",
        })
    arrivees.sort(key=lambda a: (a["person_id"], a["started_at"], a["move_id"]))
    return arrivees


# ── agent_memory_events.jsonl ────────────────────────────────────────────────

def lire_evenements(chemin_run: Path) -> list[dict[str, Any]]:
    """True JSONL, this one. An unreadable line is skipped, never fatal."""
    fichier = Path(chemin_run) / "agent_memory_events.jsonl"
    if not fichier.is_file():
        return []
    evenements = []
    with fichier.open(encoding="utf-8") as flux:
        for ligne in flux:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                evenements.append(json.loads(ligne))
            except json.JSONDecodeError:
                continue
    evenements.sort(key=lambda e: (str(e.get("person_id", "")), int(e.get("timestamp") or 0),
                                   str(e.get("message", ""))[:80]))
    return evenements


# ── llm_exchanges.jsonl (concatenated JSON, NOT JSONL) ───────────────────────

def lire_echanges(chemin_run: Path) -> list[dict[str, Any]]:
    """Concatenated indented JSON objects (contract F2).

    The extension lies: a line-by-line read fails from the first object.
    Decoding is done in a loop with ``raw_decode``, skipping blanks between objects.
    """
    fichier = Path(chemin_run) / "llm_exchanges.jsonl"
    if not fichier.is_file():
        return []
    # The worker writes this log for ALL its clients: only the exchanges signed by this run
    # (or unsigned, prior to 2026-09-24) belong to it.
    nom_run = Path(chemin_run).resolve().name
    return [
        o for o in decoder_json_concatene(fichier.read_text(encoding="utf-8"))
        if o.get("origine") in (None, nom_run)
    ]


def decoder_json_concatene(texte: str) -> list[dict[str, Any]]:
    decodeur = json.JSONDecoder()
    objets: list[dict[str, Any]] = []
    position, longueur = 0, len(texte)
    while position < longueur:
        while position < longueur and texte[position] in " \t\r\n":
            position += 1
        if position >= longueur:
            break
        try:
            objet, position = decodeur.raw_decode(texte, position)
        except json.JSONDecodeError:
            # Truncated object at the end of the file (interrupted run): stop there.
            break
        if isinstance(objet, dict):
            objets.append(objet)
    return objets


# ── Itinerary options, extracted from the text of the prompts ────────────────

_ENTETE_BLOC = re.compile(
    r"^--- agent_id=(?P<agent>\S+)\s*\|\s*Destination:\s*(?P<motif>\S+)"
    r".*?\|\s*Departure:\s*(?P<depart>\d{1,2}:\d{2})\s*---\s*$")
_LIGNE_OPTION = re.compile(r"^-\s*\[(?P<index>\d+)\]\s*(?P<mode>[^:]+):\s*(?P<desc>.*)$")
_LIGNE_ETAPE = re.compile(r"^\s+·\s*(?P<etape>.+?)\s*$")


def blocs_itineraires(echanges: Iterable[dict[str, Any]]) -> list[BlocPrompt]:
    """One block per agent and per decision prompt (`itinary_multi_agent`).

    The « · » sub-bullets detail the steps of an option; they are NOT
    options. The memory block starts at ``**History:**``.
    """
    blocs: list[BlocPrompt] = []
    for echange in echanges:
        if echange.get("category") != "itinary_multi_agent":
            continue
        jour = str(echange.get("sim_day") or "")
        for message in echange.get("messages") or []:
            if message.get("role") != "user":
                continue
            blocs.extend(_blocs_du_prompt(str(message.get("content") or ""), jour))
    blocs.sort(key=lambda b: (_cle_agent(b.agent), b.jour, b.depart_minutes or 0, b.motif))
    return blocs


def _blocs_du_prompt(texte: str, jour: str) -> list[BlocPrompt]:
    blocs: list[BlocPrompt] = []
    courant: dict[str, Any] | None = None

    def clore() -> None:
        if courant is None:
            return
        blocs.append(BlocPrompt(
            agent=courant["agent"], jour=jour, motif=motif_canonique(courant["motif"]),
            depart_minutes=courant["depart"],
            options=tuple(Option(index=o["index"], mode_brut=o["mode"],
                                 mode=mode_canonique_depuis_option(o["mode"]),
                                 description=o["desc"], etapes=tuple(o["etapes"]))
                          for o in courant["options"]),
            a_historique=courant["historique"] > 0,
            taille_historique=courant["historique"],
            historique="\n".join(courant["lignes_historique"]).strip()))

    for ligne in texte.splitlines():
        entete = _ENTETE_BLOC.match(ligne)
        if entete:
            clore()
            heure, _, minute = entete.group("depart").partition(":")
            courant = {"agent": entete.group("agent"), "motif": entete.group("motif"),
                       "depart": int(heure) * 60 + int(minute),
                       "options": [], "historique": 0, "lignes_historique": [],
                       "dans_historique": False}
            continue
        if courant is None:
            continue
        if ligne.startswith("**History:**"):
            courant["historique"] = max(1, len(ligne) - len("**History:**"))
            courant["dans_historique"] = True
            reste = ligne[len("**History:**"):].strip()
            if reste:
                courant["lignes_historique"].append(reste)
            continue
        if courant["dans_historique"]:
            # The section runs until the next block or until the reply instruction, which
            # closes the prompt: nothing that follows it is memory.
            if ligne.startswith("Reply with"):
                courant["dans_historique"] = False
            else:
                courant["lignes_historique"].append(ligne)
            continue
        etape = _LIGNE_ETAPE.match(ligne)
        if etape and courant["options"]:
            courant["options"][-1]["etapes"].append(etape.group("etape"))
            continue
        option = _LIGNE_OPTION.match(ligne)
        if option:
            courant["options"].append({
                "index": int(option.group("index")),
                "mode": option.group("mode").strip(),
                "desc": option.group("desc").strip(),
                "etapes": []})
    clore()
    return blocs


# ── Reflection blocks (known_beliefs) ────────────────────────────────────────

@dataclass(frozen=True)
class BlocReflexion:
    agent: str
    jour: str
    nb_croyances: int
    nb_observations: int


def blocs_reflexion(echanges: Iterable[dict[str, Any]]) -> list[BlocReflexion]:
    """Payload of `stm_reflection`: per agent, ``today`` and ``known_beliefs``."""
    blocs: list[BlocReflexion] = []
    for echange in echanges:
        if echange.get("category") != "stm_reflection":
            continue
        jour = str(echange.get("sim_day") or "")
        for message in echange.get("messages") or []:
            if message.get("role") != "user":
                continue
            blocs.extend(_blocs_reflexion_du_prompt(str(message.get("content") or ""), jour))
    blocs.sort(key=lambda b: (_cle_agent(b.agent), b.jour, b.nb_croyances))
    return blocs


_ENTETE_AGENT = re.compile(r"^---\s*AGENT\s+(?P<agent>\S+)\s*---\s*$")


def _blocs_reflexion_du_prompt(texte: str, jour: str) -> list[BlocReflexion]:
    depart = texte.find("# INPUT DATA")
    if depart < 0:
        return []
    blocs: list[BlocReflexion] = []
    agent: str | None = None
    tampon: list[str] = []

    def clore() -> None:
        if agent is None:
            return
        charge = "\n".join(tampon)
        debut = charge.find("{")
        if debut < 0:
            return
        objets = decoder_json_concatene(charge[debut:])
        if not objets:
            return
        donnees = objets[0]
        croyances = donnees.get("known_beliefs") or []
        aujourdhui = donnees.get("today") or []
        blocs.append(BlocReflexion(
            agent=agent, jour=jour,
            nb_croyances=len(croyances) if isinstance(croyances, list) else 0,
            nb_observations=len(aujourdhui) if isinstance(aujourdhui, list) else 0))

    for ligne in texte[depart:].splitlines():
        entete = _ENTETE_AGENT.match(ligne)
        if entete:
            clore()
            agent, tampon = entete.group("agent"), []
            continue
        if agent is not None and not ligne.startswith("```"):
            tampon.append(ligne)
    clore()
    return blocs


def autoreflexions(echanges: Iterable[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    """Long-term self-reflections, per agent: ``(jour, texte)`` in chronological order."""
    par_agent: dict[str, list[tuple[str, str]]] = {}
    for echange in echanges:
        if echange.get("category") != "ltm_self_reflection":
            continue
        jour = str(echange.get("sim_day") or "")
        for item in _reponse_en_liste(echange.get("response")):
            agent = str(item.get("agent_id") or "")
            texte = str(item.get("reflection") or item.get("summary") or "").strip()
            if agent and texte:
                par_agent.setdefault(agent, []).append((jour, texte))
    for entrees in par_agent.values():
        entrees.sort()
    return par_agent


def _reponse_en_liste(reponse: Any) -> list[dict[str, Any]]:
    """`response` is sometimes a JSON string, sometimes already decoded."""
    if isinstance(reponse, str):
        try:
            reponse = json.loads(reponse)
        except json.JSONDecodeError:
            return []
    if isinstance(reponse, dict):
        reponse = reponse.get("agents") or [reponse]
    if not isinstance(reponse, list):
        return []
    return [item for item in reponse if isinstance(item, dict)]


# ── LTM metadata ──────────────────────────────────────────────────────────────

def lire_ltm(chemin_run: Path) -> tuple[dict[str, list[EntreeLTM]], dict[str, dict[str, Any]]]:
    """Long-term memory entries and habit log, per agent.

    The traversal of the shards is **sorted**: the file system order must
    never enter the report (contract F7).
    """
    racine = Path(chemin_run) / "long_term_memory" / "user_metadata"
    entrees: dict[str, list[EntreeLTM]] = {}
    habitudes: dict[str, dict[str, Any]] = {}
    if not racine.is_dir():
        return entrees, habitudes
    for fichier in sorted(racine.glob("shard_*/*.json")):
        try:
            charge = json.loads(fichier.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        person_id = str(charge.get("person_id") or fichier.stem)
        liste = [_entree_ltm(person_id, brut) for brut in (charge.get("entries") or [])
                 if isinstance(brut, dict)]
        liste.sort(key=lambda e: (e.timestamp, e.doc_id))
        entrees.setdefault(person_id, []).extend(liste)
        journal = charge.get("journal")
        if isinstance(journal, dict):
            habitudes[person_id] = journal
    return entrees, habitudes


def _entree_ltm(person_id: str, brut: dict[str, Any]) -> EntreeLTM:
    contenu = str(brut.get("content") or "")
    return EntreeLTM(
        person_id=person_id,
        doc_id=str(brut.get("doc_id") or ""),
        contenu=contenu,
        enonce=enonce_du_concept(contenu),
        memory_type=str(brut.get("memory_type") or ""),
        timestamp=str(brut.get("timestamp") or ""),
        importance=float(_flottant(brut.get("importance")) or 0.0),
        valence=str(brut.get("valence") or ""),
        axe_objet=brut.get("axe_objet"),
        axe_motif=brut.get("axe_motif"),
        axe_lieu=brut.get("axe_lieu"),
        axe_creneau=brut.get("axe_creneau"),
        axe_meteo=brut.get("axe_meteo"),
        force=float(_flottant(brut.get("force")) or 0.0),
        rappels=int(_flottant(brut.get("rappels")) or 0),
        dernier_rappel=brut.get("dernier_rappel"),
        observations=int(_flottant(brut.get("observations")) or 0),
        contre_exemples=int(_flottant(brut.get("contre_exemples")) or 0),
    )


def enonce_du_concept(contenu: str) -> str:
    """The content of a concept is a JSON string of a 5-element array.

    The first element is the statement; the next ones are the keywords and the axes.
    A content that is not this array (a narrative reflection) is returned as is.
    """
    texte = (contenu or "").strip()
    if not texte.startswith("["):
        return texte
    try:
        decode = json.loads(texte)
    except json.JSONDecodeError:
        return texte
    if isinstance(decode, list) and decode:
        return str(decode[0])
    return texte


# ── Readable logs memoires/*.md ──────────────────────────────────────────────

_NOM_AGENT = re.compile(r"^\d+$")
_CONSOLIDATION = re.compile(r"^###\s+`(?P<heure>[^`]+)`\s+CONSOLIDATION\s+—\s+déclencheur\s*:\s*"
                            r"\*\*(?P<declencheur>[^*]+)\*\*")
_OPERATION = re.compile(r"opération\s+\*\*(?P<operation>créé|confirmé|précisé|contredit)\*\*")
_RAPPEL = re.compile(r"\*\*rappel\*\*\s+—\s+(?P<nb>\d+)\s+souvenir\(s\) servi\(s\)\s*:\s*(?P<liste>.*)$")


def lire_journaux(chemin_run: Path) -> dict[str, Journal]:
    """Logs `memoires/<person_id>.md`.

    Any file whose name is not a numeric identifier is skipped: the
    reference run carries a `609_FR.md`, a translated duplicate that would double every
    counter if it were read.
    """
    racine = Path(chemin_run) / "memoires"
    journaux: dict[str, Journal] = {}
    if not racine.is_dir():
        return journaux
    for fichier in sorted(racine.glob("*.md")):
        if not _NOM_AGENT.match(fichier.stem):
            continue
        journaux[fichier.stem] = _journal_depuis_md(
            fichier.stem, fichier.read_text(encoding="utf-8"))
    return journaux


def _journal_depuis_md(person_id: str, texte: str) -> Journal:
    journal = Journal(person_id=person_id)
    for ligne in texte.splitlines():
        consolidation = _CONSOLIDATION.match(ligne)
        if consolidation:
            journal.consolidations += 1
            declencheur = consolidation.group("declencheur").strip()
            journal.declencheurs[declencheur] = journal.declencheurs.get(declencheur, 0) + 1
            continue
        operation = _OPERATION.search(ligne)
        if operation:
            nom = operation.group("operation")
            journal.operations[nom] = journal.operations.get(nom, 0) + 1
            continue
        rappel = _RAPPEL.search(ligne)
        if rappel:
            incipits = tuple(
                _incipit(part) for part in rappel.group("liste").split(" · ")
                if part.strip() and not part.strip().startswith("+"))
            journal.rappels.append((rappel.group("nb"), incipits))
    return journal


def _incipit(fragment: str) -> str:
    """The log truncates the served memories: the text before « (force … ) » is kept."""
    texte = fragment.strip()
    coupe = texte.rfind(" (force ")
    if coupe > 0:
        texte = texte[:coupe]
    return texte.strip().rstrip("…").strip()


# ── Population ────────────────────────────────────────────────────────────────

def lire_population(chemin_run: Path) -> dict[str, dict[str, Any]]:
    """Personas of the run, indexed by identifier. The first sorted `population_*.json`."""
    racine = Path(chemin_run)
    candidats = sorted(p for p in racine.glob("population_*.json")
                       if "checkpoint" not in p.name)
    for fichier in candidats:
        try:
            charge = json.loads(fichier.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(charge, dict):
            charge = charge.get("personas") or charge.get("population") or []
        if not isinstance(charge, list):
            continue
        return {str(p.get("person_id")): p for p in charge
                if isinstance(p, dict) and p.get("person_id") is not None}
    return {}


def profil(persona: dict[str, Any]) -> dict[str, Any]:
    """Profile sheet, reduced to what the report displays."""
    identite = persona.get("identity") or {}
    traits = identite.get("traits_json") or {}
    activites = identite.get("activities") or []
    motifs: list[str] = []
    for activite in activites:
        motif = motif_canonique(str(activite.get("purpose") or ""))
        if motif and motif not in motifs:
            motifs.append(motif)
    return {
        "nom": str(identite.get("name") or traits.get("name") or "—"),
        "age": traits.get("age"),
        "occupation": traits.get("main_occupation") or traits.get("professional_activity"),
        "revenu": traits.get("income"),
        "voiture": traits.get("car_availability"),
        "velo": traits.get("personal_bike"),
        "abonnement_tc": traits.get("has_pt_subscription"),
        "commune": traits.get("residence_commune"),
        "zone": traits.get("residence_zone"),
        "motifs": motifs,
        "nb_activites": len(activites),
    }


# ── Full loading ──────────────────────────────────────────────────────────────

def charger(chemin_run: Path | str) -> Run:
    """Reads the whole run at once. Each missing source returns an empty structure."""
    racine = Path(chemin_run)
    if not racine.is_dir():
        raise FileNotFoundError(f"run directory not found: {racine}")
    trajets, rejeu, rejeu_detail = lire_moves(racine)
    echanges = lire_echanges(racine)
    ltm, habitudes = lire_ltm(racine)
    return Run(
        chemin=racine,
        trajets=trajets,
        rejeu=rejeu,
        rejeu_detail=rejeu_detail,
        arrivees=lire_arrivees(racine),
        evenements=lire_evenements(racine),
        echanges=echanges,
        blocs=blocs_itineraires(echanges),
        ltm=ltm,
        habitudes=habitudes,
        journaux=lire_journaux(racine),
        personas=lire_population(racine),
        autoreflexions=autoreflexions(echanges),
    )


def horodatage_iso(valeur: Any) -> datetime | None:
    """Tolerant parse of an ISO timestamp; ``None`` if unreadable."""
    texte = str(valeur or "").strip().replace("Z", "+00:00")
    if not texte:
        return None
    try:
        return datetime.fromisoformat(texte)
    except ValueError:
        return None
