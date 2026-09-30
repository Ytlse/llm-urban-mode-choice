#!/usr/bin/env python3
"""Chooses personas whose decisions are OBSERVABLE (ticket 093, lot 1).

WHY
---
Ticket 075 chose five agents on their TRAITS: a motorised commuter, an urban cyclist,
a school pupil… Contrasting profiles on paper. Measured over ten simulated days, three of the five
learn nothing observable: `609` and `41275` are offered a single itinerary one time out of
two — half of their "decisions" are not decisions — and take the car 100 % of the rest of the
time; `11195` lives only 19 trips against 34-35 for the others. We observe the memory of two
agents and pay the model for five.

A trait does not say whether the agent WILL DECIDE. So this script does not look at who the
agent is, but at what the world actually offered it, over a reference day already played.

⚠ **"Day" means the activity chain of a typical day, not a date.** Measured on the
ticket's reference run: departures spread from 16 March 04:13 to 18 March 06:47, because
a chain started late spills over into the next day. Filtering on the first day's date
would cut off late evenings — exactly the trips where a mode switches. So the criterion applies
to ALL trips of the reference run, and the MANIFEST writes the days covered so that
the reader sees what it applied to.

⚠ **A selection holds only for its reference run.** `41275` passes the criterion on the
thousand-agent run (two modes over four trips) and misses it on the five-persona run of
2026-09-16 (car only, one itinerary one time out of two) — another model, another prompt
variant. This is why the run is named and its fingerprint written: without it, the criterion would
look like a property of the agent whereas it is a measurement of a run.

THE CRITERION
-------------
Over that day: at least four trips, **always** more than one itinerary offered, and at
least two modes actually chosen. The three conditions are measurable BEFORE the run to observe,
hence reproducible and enforceable. Passed through the ticket's reference run, this criterion
keeps 234 candidates out of a thousand agents.

⚠ The criterion applies to the **chosen** modes, never to the offered modes. An agent who is
offered nine options and always takes the car is perfectly visible — and perfectly
useless: its memory will have nothing to tip over.

DETERMINISM
-----------
No random draw. Candidates are ordered by numeric identifier, and coverage of the four
dominant modes is done round-robin in a fixed order. Two runs on the same inputs
return the same identifiers, in the same order.

WHAT IT IS NOT
--------------
The produced file is **not a cohort seal**, and its MANIFEST says so. Ten agents
represent nothing: they serve to observe a mechanism, never to measure a modal share.
`data/population/population_1000_PANEL_v6` remains the article's reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.analysis.memoire.sources import MODE_LIBELLES, Trajet, lire_moves


@dataclass(frozen=True)
class Critere:
    """The three conditions, named. A bare threshold inside a condition cannot be discussed."""

    trajets_min: int = 4
    options_min: int = 2
    modes_min: int = 2


CRITERE = Critere()

# Modes for which we want at least one persona to carry the habit. These are the four dominant modes
# of the conurbation; the train and the powered two-wheeler are too rare for an agent to
# carry them ten days in a row, and requiring their coverage would make the selection fail over a
# matter of scenery.
MODES_CIBLES: tuple[str, ...] = ("car", "walking", "public_transport", "cycling")


@dataclass(frozen=True)
class Mesure:
    """What the reference day says about an agent."""

    person_id: str
    trajets: int
    options_min: int | None
    options_max: int | None
    options_inconnues: int
    modes: Counter = field(default_factory=Counter)
    mode_dominant: str | None = None
    jours: tuple[str, ...] = ()

    @property
    def modes_distincts(self) -> int:
        return len([m for m in self.modes if m])


@dataclass(frozen=True)
class Retenu:
    person_id: str
    persona: dict
    mesure: Mesure | None
    mode_dominant: str | None
    passe_le_critere: bool
    motif: str  # "critère" or "conservé"


# ── Measurement ───────────────────────────────────────────────────────────────────────


def mesurer(chemin_run: Path | str) -> dict[str, Mesure]:
    """One measurement per agent, from the `moves.csv` of a run already played.

    Reading goes through `sources.lire_moves`, which discards the trips REPLAYED after a
    restart. Without this deduplication, a resumed run would count the replayed days twice
    and would inflate each agent's number of trips.
    """
    trajets, _rejeu, _detail = lire_moves(Path(chemin_run))
    par_agent: dict[str, list[Trajet]] = {}
    for trajet in trajets:
        par_agent.setdefault(trajet.person_id, []).append(trajet)
    return {agent: _mesure_agent(agent, liste) for agent, liste in par_agent.items()}


def _mesure_agent(person_id: str, trajets: Sequence[Trajet]) -> Mesure:
    ordonnes = sorted(trajets, key=lambda t: (t.jour, t.depart, t.trajet_id))
    options = [t.options_presentees for t in ordonnes if t.options_presentees is not None]
    modes = Counter(t.mode for t in ordonnes if t.mode)
    return Mesure(
        person_id=person_id,
        trajets=len(ordonnes),
        options_min=min(options) if options else None,
        options_max=max(options) if options else None,
        options_inconnues=sum(1 for t in ordonnes if t.options_presentees is None),
        modes=modes,
        mode_dominant=_mode_dominant(ordonnes, modes),
        jours=tuple(sorted({t.jour for t in ordonnes if t.jour})),
    )


def _mode_dominant(trajets: Sequence[Trajet], modes: Counter) -> str | None:
    """The most chosen mode; on a tie, the one that departs FIRST in the day.

    Breaking ties with a canonical order table would have fabricated a hierarchy of modes where
    the agent expresses none — and would have filed all ties under "voiture", hence all
    coverage under a single mode. The first departure, on the other hand, is a fact of the day.
    """
    if not modes:
        return None
    premier_rang = {}
    for rang, trajet in enumerate(trajets):
        if trajet.mode and trajet.mode not in premier_rang:
            premier_rang[trajet.mode] = rang
    return min(modes, key=lambda mode: (-modes[mode], premier_rang.get(mode, len(trajets))))


def est_candidat(mesure: Mesure, critere: Critere = CRITERE) -> bool:
    """The three conditions, with no fallback or tolerance.

    A trip whose number of options is UNKNOWN disqualifies the agent: it does not prove there
    was a choice, and counting it as if there were one would let into the selection
    exactly the decisions that were not decisions.
    """
    if mesure.options_inconnues:
        return False
    if mesure.trajets < critere.trajets_min:
        return False
    if mesure.options_min is None or mesure.options_min < critere.options_min:
        return False
    return mesure.modes_distincts >= critere.modes_min


# ── Choice ────────────────────────────────────────────────────────────────────────────


def _cle_agent(person_id: str) -> tuple[int, int, str]:
    return (0, int(person_id), "") if person_id.isdigit() else (1, 0, person_id)


def _cle_richesse(mesure: Mesure) -> tuple:
    """Most observable first: modes, then trips, then choice offered.

    ⚠ The criterion is a FLOOR, not a goal. Taking candidates in identifier order
    gave ten agents with four trips and two modes — all compliant, all
    minimal, and `41275` among them: the very agent the ticket discards for learning nothing.
    An agent making fifteen trips over four modes gives memory fifteen opportunities to
    weigh; an agent making four gives it four. At equal richness, the numeric identifier
    decides, and the selection stays reproducible.
    """
    return (
        -mesure.modes_distincts,
        -mesure.trajets,
        -(mesure.options_min or 0),
        _cle_agent(mesure.person_id),
    )


def choisir(
    mesures: dict[str, Mesure],
    population: Sequence[dict],
    conserver: Sequence[str],
    combien: int,
    critere: Critere = CRITERE,
) -> list[Retenu]:
    """The selected agents, kept ones first, then the candidates round-robin by mode.

    Candidates are taken from most observable to least observable (see `_cle_richesse`),
    round-robin over the four dominant modes so that no mode is missing.

    The agents in `conserver` are selected **by right**: they carry the shock and its control,
    and replacing them would change the study's subject. But their measurement is written, and
    the MANIFEST says whether they pass the criterion — a silent retention would be rigged.
    """
    par_id = {str(p.get("person_id")): p for p in population}
    retenus: list[Retenu] = []
    deja: set[str] = set()

    for person_id in conserver:
        if person_id not in par_id:
            raise SystemExit(
                f"The kept agent '{person_id}' is missing from the source population: "
                f"it cannot be written into a population that claims to come from it."
            )
        mesure = mesures.get(person_id)
        retenus.append(Retenu(
            person_id=person_id,
            persona=par_id[person_id],
            mesure=mesure,
            mode_dominant=mesure.mode_dominant if mesure else None,
            passe_le_critere=bool(mesure) and est_candidat(mesure, critere),
            motif="conservé",
        ))
        deja.add(person_id)

    candidats = [
        mesure.person_id
        for mesure in sorted(mesures.values(), key=_cle_richesse)
        if mesure.person_id not in deja
        and mesure.person_id in par_id
        and est_candidat(mesure, critere)
    ]
    if len(retenus) + len(candidats) < combien:
        raise SystemExit(
            f"{len(candidats)} candidate(s) satisfy the criterion and {combien} are requested. "
            f"The selection does NOT fill up outside the criterion: change the reference run, or "
            f"relax the criterion and say so."
        )

    # Round-robin: one dominant mode at a time, until the count is reached. Covering the
    # four modes BEFORE filling up, rather than taking the first ten identifiers,
    # is what prevents a fully motorised selection by an accident of ordering.
    restants = list(candidats)
    while len(retenus) < combien:
        pris_ce_tour = False
        for mode in MODES_CIBLES:
            if len(retenus) >= combien:
                break
            for person_id in restants:
                if mesures[person_id].mode_dominant != mode:
                    continue
                retenus.append(_retenu_par_critere(person_id, par_id, mesures))
                restants.remove(person_id)
                pris_ce_tour = True
                break
        if not pris_ce_tour:
            break

    for person_id in list(restants):
        if len(retenus) >= combien:
            break
        retenus.append(_retenu_par_critere(person_id, par_id, mesures))
        restants.remove(person_id)
    return retenus


def _retenu_par_critere(person_id: str, par_id: dict, mesures: dict) -> Retenu:
    return Retenu(
        person_id=person_id,
        persona=par_id[person_id],
        mesure=mesures[person_id],
        mode_dominant=mesures[person_id].mode_dominant,
        passe_le_critere=True,
        motif="critère",
    )


# ── Writing ───────────────────────────────────────────────────────────────────────────


def sha256(chemin: Path) -> str:
    empreinte = hashlib.sha256()
    with chemin.open("rb") as flux:
        for bloc in iter(lambda: flux.read(1 << 20), b""):
            empreinte.update(bloc)
    return empreinte.hexdigest()


def ecrire(
    chemin_run: Path | str,
    source: Path | str,
    sortie: Path | str,
    conserver: Sequence[str],
    combien: int,
    critere: Critere = CRITERE,
) -> list[Retenu]:
    """Writes `population.json` and `MANIFEST.yaml`, and returns the selected agents."""
    chemin_run, source, sortie = Path(chemin_run), Path(source), Path(sortie)
    population = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(population, dict):
        population = population.get("personas") or population.get("population") or []
    mesures = mesurer(chemin_run)
    retenus = choisir(mesures, population, conserver, combien, critere)

    sortie.mkdir(parents=True, exist_ok=True)
    fichier = sortie / "population.json"
    # Copied AS IS: no normalisation, no field added. A test population that
    # diverges from the seal it comes from no longer proves anything.
    fichier.write_text(
        json.dumps([r.persona for r in retenus], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (sortie / "MANIFEST.yaml").write_text(
        _manifeste(chemin_run, source, fichier, population, mesures, retenus, critere),
        encoding="utf-8",
    )
    return retenus


def _manifeste(
    chemin_run: Path,
    source: Path,
    fichier: Path,
    population: Sequence[dict],
    mesures: dict[str, Mesure],
    retenus: Sequence[Retenu],
    critere: Critere,
) -> str:
    candidats = sum(1 for m in mesures.values() if est_candidat(m, critere))
    moves = chemin_run / "moves.csv"
    lignes = [
        "# Population de TEST — ticket 093. Ce n'est PAS un sceau de cohorte.",
        "#",
        "# Dix agents ne représentent rien : ils servent à LIRE l'évolution d'une mémoire et",
        "# la rupture d'une habitude, jamais à mesurer une part modale. La référence de",
        "# l'article reste data/population/population_1000_PANEL_v6.",
        f"nom: {fichier.parent.name}",
        f"extrait_le: '{datetime.now(timezone.utc).isoformat()}'",
        "source:",
        f"  fichier: {source.as_posix()}",
        f"  sha256: {sha256(source)}",
        f"  n: {len(population)}",
        "population:",
        "  fichier: population.json",
        f"  sha256: {sha256(fichier)}",
        f"  n: {len(retenus)}",
        "run_de_reference:",
        f"  chemin: {chemin_run.as_posix()}",
        f"  moves_sha256: {sha256(moves) if moves.is_file() else 'absent'}",
        f"  agents_mesures: {len(mesures)}",
        f"  candidats: {candidats}",
        # The days COVERED, and not "the" day: a chain started late spills over into
        # the next day, and the reader must see the exact base of the criterion.
        f"  journees_couvertes: {json.dumps(sorted({j for m in mesures.values() for j in m.jours}))}",
        "critere:",
        f"  trajets_min: {critere.trajets_min}",
        f"  options_min: {critere.options_min}",
        f"  modes_min: {critere.modes_min}",
        "  enonce: >-",
        "    Sur la journée de référence : au moins trajets_min trajets, TOUJOURS au moins",
        "    options_min itinéraires proposés, et au moins modes_min modes réellement CHOISIS.",
        "    Le critère porte sur les modes choisis et jamais sur les modes proposés : un agent",
        "    à qui l'on propose tout et qui prend toujours la voiture est visible et inutile.",
        "selection:",
        "  methode: >-",
        "    Aucun tirage. Les agents conservés sont retenus de droit ; les autres sont pris",
        "    parmi les candidats, en tourniquet sur les quatre modes dominants, du plus",
        "    observable au moins observable — modes, puis trajets, puis choix offert — et",
        "    l'identifiant numérique départage les ex æquo. Reproductible par",
        "    scripts/data/population/selectionner_personas_mesurables.py.",
        "  agents:",
    ]
    for retenu in retenus:
        mesure = retenu.mesure
        lignes += [
            f"    - person_id: '{retenu.person_id}'",
            f"      motif: {retenu.motif}",
            f"      passe_le_critere: {str(retenu.passe_le_critere).lower()}",
            f"      mode_dominant: {retenu.mode_dominant or 'inconnu'}",
        ]
        if mesure is None:
            lignes.append("      mesure: absente du run de référence")
            continue
        modes = " · ".join(
            f"{MODE_LIBELLES.get(mode, mode)} {compte}"
            for mode, compte in mesure.modes.most_common()
        )
        lignes += [
            f"      trajets: {mesure.trajets}",
            f"      options: {mesure.options_min}-{mesure.options_max}",
            f"      modes_distincts: {mesure.modes_distincts}",
            f"      modes: {json.dumps(modes, ensure_ascii=False)}",
        ]
    return "\n".join(lignes) + "\n"


def verifier(dossier: Path | str) -> list[str]:
    """Anomalies between a produced population and its MANIFEST. Empty list = intact.

    ⚠ This is NOT a cohort seal, and it does not claim to be one: `scripts.panel.seal_population`
    draws households by strata and checks thirteen margins to ± 1 point, which makes no sense with
    ten agents and is not the goal — these ten serve to observe a mechanism, never to
    represent a population.

    What this check guarantees is more modest and suffices here: the delivered file is indeed
    the one the MANIFEST describes, it comes from the announced source, and the count is right.
    Without it, a population modified by hand would keep claiming a measured criterion.
    """
    dossier = Path(dossier)
    manifeste = dossier / "MANIFEST.yaml"
    fichier = dossier / "population.json"
    anomalies: list[str] = []
    if not manifeste.is_file() or not fichier.is_file():
        return [f"{dossier} ne porte pas population.json ET MANIFEST.yaml"]

    declare: dict[str, str] = {}
    section = ""
    for ligne in manifeste.read_text(encoding="utf-8").splitlines():
        if ligne and not ligne.startswith((" ", "#")) and ligne.rstrip().endswith(":"):
            section = ligne.rstrip()[:-1]
        elif ligne.startswith("  ") and ":" in ligne and not ligne.startswith("    "):
            clef, _, valeur = ligne.strip().partition(":")
            declare[f"{section}.{clef}"] = valeur.strip()

    empreinte = sha256(fichier)
    if declare.get("population.sha256") not in (None, empreinte):
        anomalies.append(
            f"population.json a changé depuis son manifeste : {empreinte[:12]}… au lieu de "
            f"{declare['population.sha256'][:12]}…")
    agents = json.loads(fichier.read_text(encoding="utf-8"))
    attendu = declare.get("population.n")
    if attendu and attendu.isdigit() and int(attendu) != len(agents):
        anomalies.append(f"{len(agents)} agent(s) dans le fichier, {attendu} annoncés")

    source = declare.get("source.fichier")
    if source:
        chemin_source = Path(source)
        if not chemin_source.is_file():
            anomalies.append(f"source introuvable : {source} — la sélection n'est plus rejouable")
        elif declare.get("source.sha256") != sha256(chemin_source):
            anomalies.append(f"la source {source} a changé depuis l'extraction")
    return anomalies


def main(argv: Sequence[str] | None = None) -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--verifier", type=Path, default=None,
                         help="checks an already produced population against its MANIFEST, "
                              "and exits")
    parseur.add_argument("--run", type=Path,
                         help="directory of a run already played, containing moves.csv")
    parseur.add_argument("--source", type=Path,
                         help="sealed population.json the agents are taken from")
    parseur.add_argument("--sortie", type=Path, help="directory to create")
    parseur.add_argument("--conserver", default="",
                         help="identifiers selected by right, comma-separated")
    parseur.add_argument("-n", "--combien", type=int, default=10)
    args = parseur.parse_args(argv)

    if args.verifier:
        anomalies = verifier(args.verifier)
        if not anomalies:
            print(f"✔ {args.verifier} is intact: fingerprint, count and source agree.")
            return 0
        print(f"✘ {args.verifier} — {len(anomalies)} anomaly(ies):")
        for anomalie in anomalies:
            print(f"  · {anomalie}")
        return 1

    manquants = [nom for nom in ("run", "source", "sortie") if getattr(args, nom) is None]
    if manquants:
        parseur.error("required arguments: " + ", ".join(f"--{nom}" for nom in manquants))

    conserver = [x.strip() for x in args.conserver.split(",") if x.strip()]
    retenus = ecrire(args.run, args.source, args.sortie, conserver, args.combien)

    print(f"{len(retenus)} agents written → {args.sortie / 'population.json'}")
    for retenu in retenus:
        mesure = retenu.mesure
        detail = (
            f"{mesure.trajets:>3} trajets · {mesure.modes_distincts} modes · "
            f"{mesure.options_min}-{mesure.options_max} options"
            if mesure else "non mesuré sur le run de référence"
        )
        marque = "" if retenu.passe_le_critere else "  ⚠ NE PASSE PAS LE CRITÈRE"
        print(f"  {retenu.person_id:>8s}  {retenu.motif:9s} {detail}{marque}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
