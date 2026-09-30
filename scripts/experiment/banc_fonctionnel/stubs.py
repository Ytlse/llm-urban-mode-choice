"""The bench's stubs — ticket 100, functional tests.

NONE OF THESE OBJECTS SIMULATES INTELLIGENCE. Each one builds an object whose **format** is
the repository's: a population, a long-term memory, a day, a `moves.csv`, a
checkpoint. They are structures, not behaviours — building them by hand costs zero
tokens and makes them reproducible, which a run will never be.

ONE RULE, AND IT IS NOT NEGOTIABLE
-------------------------------------------
A format stub **imports** the repository's definition — `CSV_HEADERS`, `MemoryEntry`, the
JSON schemas — instead of copying it. A stub that copies a list of columns tests its
own copy: it stays green the day the real format changes, and that is precisely what the
bench must catch. It is the lesson of lot A of ticket 077, where two mode vocabularies
coexisted without anything saying so.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
AGENTS = RACINE / "services" / "llm-agents"
for chemin in (str(RACINE), str(AGENTS)):
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

from llm.memory import MemoryEntry, MemoryType  # noqa: E402

# Bench anchor: Monday 16 March 2026, 5 am — that of the repository's runs.
ANCRE = datetime(2026, 3, 16, 5, 0)


# ── The population ──────────────────────────────────────────────────────────────────────────
@dataclass
class IdentiteStub:
    traits_json: dict


@dataclass
class AgentStub:
    """The minimum that `foyer.py`, `exposition.py` and the controller read.

    ⚠ This is NOT a `models.Person`: building a complete Person would require a home,
    activities and a planning state that nothing here needs. The three attributes
    below are exactly those that ticket 100's code reads on an agent — if a
    fourth appeared, this stub would break, and that is what we want.
    """

    person_id: str
    household_id: str | None = None
    immobile: bool = False
    identity: IdentiteStub = field(default_factory=lambda: IdentiteStub({}))


def population_de_banc() -> list[AgentStub]:
    """Twelve agents, six households of two, deliberately contrasted profiles.

    The shape comes from the `MANIFEST.yaml` of `population_20_foyers_059`: households of size 2,
    two mobile members, at least one PT subscriber. The contrasts are those of ticket 078 § 0 —
    licence, bike, subscription — because that is where rule R4 plays out: what one learns about
    the car must never become a belief of the other.
    """
    modele = [
        # (household, (id, name, age, licence, cars, bike, PT subscription))
        ("605813", ("1254859", "Frédérique Lacombe", 20, True, 1, "e-bike", True),
                   ("1254860", "Auguste Guérin", 20, False, 0, "No bike", True)),
        ("234839", ("527098", "Charlotte-Odette Moreau", 62, True, 1, "No bike", False),
                   ("527099", "Rémy de Renaud", 62, False, 0, "No bike", True)),
        ("471076", ("1013072", "Colette Vaillant-Fleury", 44, True, 2, "Own bike", False),
                   ("1013073", "Aimé Vaillant", 12, False, 0, "No bike", True)),
        ("538432", ("880011", "Lucien Barbier", 35, True, 1, "Own bike", True),
                   ("880012", "Marthe Barbier", 33, True, 1, "No bike", True)),
        ("73563",  ("640201", "Solange Perrot", 71, False, 0, "No bike", True),
                   ("640202", "Hubert Perrot", 74, True, 1, "No bike", False)),
        ("625551", ("990301", "Nadia Fontaine", 28, False, 0, "e-bike", True),
                   ("990302", "Oscar Fontaine", 30, True, 1, "Own bike", False)),
    ]
    agents: list[AgentStub] = []
    for foyer, *membres in modele:
        for pid, nom, age, permis, voitures, velo, tc in membres:
            agents.append(AgentStub(
                person_id=pid,
                household_id=foyer,
                identity=IdentiteStub({
                    "name": nom, "age": age,
                    "has_driving_license": permis, "number_of_cars": voitures,
                    "personal_bike": velo, "has_pt_subscription": tc,
                    "main_occupation": "Full-Time Worker" if age < 65 else "Retired",
                    "household_size": 2,
                }),
            ))
    return agents


def perception_stub(agent: AgentStub) -> str:
    """The identity narrative served to the model, in the shape of `get_person_identity_description`.

    Deliberately short: the judgement bears on the event, not on the day. A long persona
    would make one pay input tokens without moving the answer.
    """
    t = agent.identity.traits_json
    morceaux = [f"{t['name']}, {t['age']}, {t.get('main_occupation', 'Worker')}"]
    morceaux.append("has a driving licence" if t.get("has_driving_license")
                    else "has no driving licence")
    morceaux.append(f"{t.get('number_of_cars', 0)} car(s) in the household")
    morceaux.append("owns a bike" if str(t.get("personal_bike", "")).lower() not in
                    ("", "no bike") else "has no bike")
    morceaux.append("holds a public transport pass" if t.get("has_pt_subscription")
                    else "holds no public transport pass")
    return ". ".join(morceaux) + "."


# ── The long-term memory ────────────────────────────────────────────────────────────────────
class MemoireLongueStub:
    """What `foyer.py` and the measures read: `user_metadata[pid]["entries"]`.

    Enough for the whole of lot 4. Carries no vector index — no bench test recalls
    by similarity, and an index would require an embedding model, hence a cost.
    """

    def __init__(self) -> None:
        self.user_metadata: dict[str, dict] = {}

    def ajouter(self, entree: MemoryEntry) -> MemoryEntry:
        self.user_metadata.setdefault(entree.person_id, {"entries": []})["entries"].append(entree)
        return entree

    def has_memories(self, person_id: str) -> bool:
        return bool((self.user_metadata.get(str(person_id)) or {}).get("entries"))


def reflexion_stub(person_id: str, texte: str, jour: int) -> MemoryEntry:
    """A day summary, as the consolidation writes one.

    It is THIS text that the evening narrative cites (D1 + Q3, 2026-09-22): the household writes
    nothing, it repeats what the other wrote. Writing it by hand replaces eight days of simulation.
    """
    return MemoryEntry(
        content=texte,
        timestamp=ANCRE + timedelta(days=jour - 1, hours=17),
        memory_type=MemoryType.REFLECTION,
        person_id=str(person_id),
        importance=0.30,
    )


def concept_stub(
    person_id: str, texte: str, jour: int, *, mode: str = "public_transport",
    observations: int = 2, contre: int = 0, origine: str | None = "vecu",
    motif: str | None = "work",
) -> MemoryEntry:
    """A concept, in the canonical 5-tuple of lot 3 of ticket 071.

    `observations` decides R1, `origine` decides D2, `mode` decides R4: the three rules
    that lot 4 exercises are driven from this signature.
    """
    return MemoryEntry(
        content=json.dumps([texte, "", "", "", ""], ensure_ascii=False),
        timestamp=ANCRE + timedelta(days=jour - 1, hours=17),
        memory_type=MemoryType.CONCEPT,
        person_id=str(person_id),
        observations=observations,
        contre_exemples=contre,
        axe_objet=mode,
        axe_motif=motif,
        origine=origine,
        importance=0.30,
    )


def journee_stub(person_id: str, jour: int, *, mode: str = "car") -> list[MemoryEntry]:
    """The three short-term memory entries a day actually produces.

    Decision, arrival, chain constraint. This is not a simplification: it is what the
    controller writes, and it is the reason why the evening narrative "per raw entry"
    would have tripled the block (Q1).
    """
    base = ANCRE + timedelta(days=jour - 1)
    return [
        MemoryEntry(
            content=f"[ TRAVEL_PLAN ] Plan to head <work>. Chosen mode: {mode}.",
            timestamp=base + timedelta(hours=3), memory_type=MemoryType.CONVERSATION,
            person_id=str(person_id), axe_objet=mode, axe_motif="work", origine="vecu",
        ),
        MemoryEntry(
            content="[ ARRIVAL ] Arrived at work 12 minutes later than planned.",
            timestamp=base + timedelta(hours=3, minutes=50),
            memory_type=MemoryType.CONVERSATION, person_id=str(person_id),
            importance=0.24, axe_objet=mode, axe_motif="work", origine="vecu",
        ),
        MemoryEntry(
            content="[ TRAVEL_PLAN ] No choice was possible: this was the only itinerary.",
            timestamp=base + timedelta(hours=11), memory_type=MemoryType.CONVERSATION,
            person_id=str(person_id), importance=0.10, axe_objet=mode, axe_motif="home",
            origine="vecu",
        ),
    ]


# ── The itineraries ─────────────────────────────────────────────────────────────────────────
def itineraires_stub(modes: tuple[str, ...] = ("car", "public_transport", "cycling")) -> list[dict]:
    """Two to four proposals, in the shape of an option served to the decision-maker.

    No call to OTP: the bench does not test routing, it tests what ticket 100 adds.
    The durations are plausible and above all **distinct**, so that a choice makes sense.
    """
    duree = {"car": 1080, "public_transport": 2040, "cycling": 1500, "walking": 3600}
    return [
        {"index": i, "mode": m, "duration_s": duree.get(m, 1800),
         "distance_km": round(duree.get(m, 1800) / 240, 1)}
        for i, m in enumerate(modes)
    ]


# ── The decision log ────────────────────────────────────────────────────────────────────────
def moves_stub(
    chemin: Path, *, evenement: str, jours: range = range(-5, 16),
    roles: tuple[str, ...] = ("expose", "co_resident", "temoin"),
    part_avant: float = 0.60, creux: dict[str, float] | None = None,
    par_jour: int = 4,
) -> Path:
    """A `moves.csv` with the REAL columns, imported from `move_logger`.

    ⚠ The headers are not copied: `CSV_HEADERS` is imported. A column added to the
    repository therefore appears here without touching anything, and a line that no longer has
    the right number of fields makes the bench fail — it is exactly the defect we want to see
    early, because a column shift only shows on re-reading, when it is too late.
    """
    from urban_mobility_agents.utils.move_logger import CSV_HEADERS

    creux = creux or {"expose": 0.30, "co_resident": 0.08, "temoin": 0.0}
    lignes = []
    for jour in jours:
        for role in roles:
            for i in range(par_jour):
                apres = jour >= 0
                part = part_avant - (creux.get(role, 0.0) if apres and jour <= 10 else 0.0)
                # Deterministic alternation: the target share reads in the frequencies, not in
                # a draw — a bench that draws at random does not replay identically.
                voiture = (i / par_jour) < part
                lignes.append({
                    # ⚠ « Référence » is the RUN identifier; the agent is under « ID
                    # Personne ». The stub wrote the agent in the wrong column, and the
                    # tests passed: they checked the stub's convention, not the
                    # repository's. It is exactly the trap that the rule "import, do not
                    # copy" was meant to avoid — it covered the HEADERS, not the MEANING of
                    # the columns. Found by a real run on 2026-09-22.
                    "Référence": "banc",
                    "ID Personne": f"{role}_{i}",
                    "Heure de départ": (ANCRE + timedelta(days=jour + 10)).strftime(
                        "%Y-%m-%d %H:%M"),
                    "Mode de transport Choisi": "car" if voiture else "public_transport",
                    "P(Voiture Privée) %": round(part * 100, 1),
                    "P(Transports_collectifs) %": round((1 - part) * 100, 1),
                    "Choc": evenement,
                    "Jour relatif au choc": jour,
                    "Rôle": role,
                    "Raison d'exposition": f"foyer:{role}",
                    "Contrainte de chaîne": "",
                })
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        w.writeheader()
        for ligne in lignes:
            w.writerow({c: ligne.get(c, "") for c in CSV_HEADERS})
    return chemin


def point_de_reprise_stub(workdir: Path, jour: int, *, foyer: dict | None = None) -> Path:
    """A valid checkpoint, in the format of `utils/reprise.py`.

    The description is written LAST, as in the real one: it is what makes the checkpoint
    valid, and a half-written checkpoint would be restored silently.
    """
    from urban_mobility_agents.utils import reprise

    cible = workdir / reprise.POINTS / f"jour_{jour:03d}"
    for nom in reprise._A_COPIER:
        (cible / nom).mkdir(parents=True, exist_ok=True)
    (cible / reprise.DESCRIPTION).write_text(
        json.dumps({
            "jour_simule": jour,
            "timestamp_simule": int((ANCRE + timedelta(days=jour - 1)).timestamp()),
            "compteurs": {},
            "foyer": foyer or {},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return cible


def echanges_stub(
    chemin: Path, *, agent: str, texte: str, exposition: int,
    categorie: str = "itinary_multi_agent",
) -> Path:
    """A `llm_exchanges.jsonl` carrying DECISION prompts, in the real shape.

    `exposition` is the event instant, in simulated UTC seconds. Each decision is
    dated after it (`sim_ts`, `sim_day`), like the real exchanges: the table discards an
    undated decision, or one prior to the exposure (2026-09-25).

    The four-channel table reads this file: it looks there for the event text, block by
    block. Without it, it outputs « non concluant », which is correct — but prompts are then
    needed to check that it can also conclude.

    Three prompts, and they cover the three cases that matter:
    - the event found AS IS in « Ce qui a changé récemment »;
    - the event REPHRASED, caught only by the salient words — it is the real case of the
      lived regime, where the evening reflection rephrases before the text reaches the memory;
    - a prompt where it does not appear at all.

    ⚠ The agent is recognised by `agent_id=`, which the decision template puts at the head of each
    persona. It is the only reliable attribution: all readers of the same article share
    the same text.

    ⚠ The words of the rephrasing come from `mots_saillants()`, IMPORTED from the table. The stub
    drew them from its own rule (words of more than four letters), which returned nothing on
    « Bed bugs on line A. »: the rephrased prompt carried no salient word, no `~`
    cell could come out, and A9.5 stayed green on the legend alone (2026-09-25). It takes
    TWO, the table's threshold: a text that does not provide as many is refused, rather than
    silently building a prompt that no matching can catch.
    """
    from scripts.analysis.tableau_quatre_voies import mots_saillants

    saillants = mots_saillants(texte, combien=2)
    if len(saillants) < 2:
        raise ValueError(
            f"echanges_stub: « {texte} » provides only {len(saillants)} salient word(s) "
            f"({saillants}); the table requires two to match a rephrasing."
        )
    prompts = [
        f"--- agent_id={agent} ---\nMes habitudes\n- souvent la voiture\n"
        f"Ce qui a changé récemment\n- {texte}\n",
        f"--- agent_id={agent} ---\nCe que je sais\n"
        f"- I read something about {' and '.join(saillants)} the other morning\n",
        f"--- agent_id={agent} ---\nMes habitudes\n- rien de notable\n",
    ]
    chemin.parent.mkdir(parents=True, exist_ok=True)
    # Indented, like the gateway: written on one line, this stub let through a line-by-line
    # reader that the first real run brought down (2026-09-25).
    def _echange(rang: int, prompt: str) -> dict:
        sim_ts = exposition + 3600 * (rang + 1)
        return {"category": categorie, "sim_ts": sim_ts,
                "sim_day": datetime.fromtimestamp(sim_ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "messages": [{"role": "user", "content": prompt}]}

    chemin.write_text(
        "".join(
            json.dumps(_echange(rang, p), ensure_ascii=False, indent=2) + "\n"
            for rang, p in enumerate(prompts)
        ),
        encoding="utf-8",
    )
    return chemin
