"""Does the reading precede the decision? — ticket 111, lot 6.

On 2026-09-25, the a09 treated arm `2026-09-24_17_50` showed that an article read on waking
was before the model NEITHER at the reader's decision on the reading day — precomputed the day before —
NOR on the following days, for lack of a severity judged high enough. Nothing said so: the prompts
had to be reread one by one.

This check rereads `llm_exchanges.jsonl` and says, decision by decision, whether the guaranteed line
was there during the service days: `[ PRESSE ]` for the reader, `[ FOYER ]` for an informed
member. It returns the SUCCESSES too: a check whose failures alone are visible does not tell
"all is well" from "it no longer runs".

Exact, with no heuristic: the prefix is rendered word for word in the prompt. Each user
message is split by `--- agent_id=<id> | … ---` and only the block of the person concerned
is tested — a merged prompt carries several agents, and the reader's line does not belong to
their co-resident (ticket 110, § 3).

Usage:
    python scripts/analysis/lecture_avant_decision.py experiments/archive/2026-09-24_17_50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

# « This morning » left the line on 2026-09-25: the archives from before still carry it.
MARQUE_LECTURE = re.compile(r"\[ PRESSE \] (?:This morning )?I read in the paper:")
MARQUE_FOYER = re.compile(re.escape("[ FOYER ]"))
# Ticket 111 default when the archived declaration does not carry `service` yet (runs from before
# the ticket): it is what allows replaying the check on the archive of the defect.
JOURS_PAR_DEFAUT = 5
CATEGORIE_DECISION = "itinary_multi_agent"
_ENTETE = re.compile(r"^--- agent_id=(\S+) \|.*---\s*$", re.M)
_DEPART = re.compile(r"Departure: (\d{1,2}):(\d{2})")


@dataclass
class Jour:
    jour: date
    decisions: int = 0
    avec_ligne: int = 0


@dataclass
class Constat:
    """What a person saw during their service days."""

    person_id: str
    role: str                    # « lecteur », « informé », « non informé »
    date_lecture: date
    jours: list[Jour] = field(default_factory=list)
    message: str = ""
    mineur: bool = False
    modes_choisis: list[str] = field(default_factory=list)
    # Did the FIRST decision of day 1 carry the line? `None` without a decision that day.
    premiere: bool | None = None

    @property
    def attend_ligne(self) -> bool:
        return self.role != "non informé"

    @property
    def privees(self) -> int:
        """Decisions of a service day WITHOUT the line (or WITH it, for a non-informed one)."""
        if self.attend_ligne:
            return sum(j.decisions - j.avec_ligne for j in self.jours)
        return sum(j.avec_ligne for j in self.jours)

    @property
    def verdict(self) -> str:
        if not any(j.decisions for j in self.jours):
            return "⚪"
        if self.attend_ligne and self.premiere is False:
            return "🔴"
        return "🔴" if self.privees else "✅"


# ── Reading the run ─────────────────────────────────────────────────────────────────────────
def iter_json_concat(chemin: Path):
    """Concatenated and indented JSON objects — `llm_exchanges.jsonl` is not JSONL."""
    tampon = ""
    with chemin.open(encoding="utf-8") as f:
        for ligne in f:
            tampon += ligne
            try:
                yield json.loads(tampon)
                tampon = ""
            except json.JSONDecodeError:
                continue


def _jsonl(chemin: Path) -> list[dict]:
    if not chemin.is_file():
        return []
    sortie = []
    for brute in chemin.read_text(encoding="utf-8").splitlines():
        if brute.strip():
            try:
                sortie.append(json.loads(brute))
            except json.JSONDecodeError:
                continue
    return sortie


def _declaration(run: Path) -> dict:
    chemin = run / "evenement.yaml"
    if not chemin.is_file():
        return {}
    return yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}


def _sans_week_end(run: Path) -> bool:
    chemin = run / "static_config.yaml"
    if not chemin.is_file():
        return True
    brut = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
    return bool((brut.get("agent") or {}).get("no_weekend_departures", True))


def jours_de_service(debut: date, n: int, sans_week_end: bool) -> list[date]:
    """Same rule as `llm/evenements/calendrier.jours_de_service`, copied to stay self-contained."""
    jours, courant = [], debut
    while len(jours) < n:
        if not (sans_week_end and courant.weekday() >= 5):
            jours.append(courant)
        courant += timedelta(days=1)
    return jours


def jour_du_trajet(sim_ts: float, entete: str, sans_week_end: bool) -> date:
    """The day of the trip decided in a prompt block.

    ⚠ `sim_ts` is NOT the agent's departure: the worker writes there the smallest departure of the
    BATCH (`rejeu_decisions.py`), and a batch mixes trips of two days. The block header carries
    only the time (`Departure: 05:57`). The departure is therefore the first instant at that time
    not before `sim_ts`, moved to Monday under `no_weekend_departures` as the calendar
    does. Without a readable time, it falls back on the day of `sim_ts`.
    """
    debut = datetime.fromtimestamp(float(sim_ts), timezone.utc)
    m = _DEPART.search(entete)
    if not m:
        return debut.date()
    depart = debut.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    if depart < debut.replace(second=0, microsecond=0):
        depart += timedelta(days=1)
    jour = depart.date()
    while sans_week_end and jour.weekday() >= 5:
        jour += timedelta(days=1)
    return jour


def blocs_par_personne(
    run: Path, sans_week_end: bool = True
) -> dict[str, list[tuple[str, date, str]]]:
    """`{person_id: [(real time, trip day, prompt block), …]}`, in real order."""
    chemin = run / "llm_exchanges.jsonl"
    blocs: dict[str, list[tuple[str, date, str]]] = defaultdict(list)
    if not chemin.is_file():
        return blocs
    for o in iter_json_concat(chemin):
        if o.get("category") != CATEGORIE_DECISION:
            continue
        try:
            sim_ts = float(o["sim_ts"])
        except (KeyError, TypeError, ValueError):
            continue
        for m in o.get("messages") or []:
            if m.get("role") != "user":
                continue
            texte = str(m.get("content") or "")
            entetes = list(_ENTETE.finditer(texte))
            for i, e in enumerate(entetes):
                fin = entetes[i + 1].start() if i + 1 < len(entetes) else len(texte)
                jour = jour_du_trajet(sim_ts, e.group(0), sans_week_end)
                blocs[e.group(1)].append((str(o.get("time") or ""), jour, texte[e.start():fin]))
    for liste in blocs.values():
        liste.sort(key=lambda t: t[0])
    return blocs


def _modes_choisis(run: Path, pid: str, jours: set[date]) -> list[str]:
    chemin = run / "moves.csv"
    if not chemin.is_file():
        return []
    modes = []
    with chemin.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ID Personne") != pid:
                continue
            try:
                jour = datetime.fromtimestamp(float(row["Temps simulé"]), timezone.utc).date()
            except (KeyError, TypeError, ValueError):
                continue
            if jour in jours:
                modes.append(f"{jour:%d/%m} {row.get('Mode de transport Choisi', '?')}")
    return modes


# ── The two sub-roles of the co-resident ────────────────────────────────────────────────────
def informes_du_run(run: Path | str) -> set[str] | None:
    """The members informed by the reader, read from `relais_foyer.jsonl`, which is authoritative.

    `None` when the run has no relay: the co-resident then stays a single role. An empty
    set, in contrast, says that the relay took place and that nobody was informed.
    """
    chemin = Path(run) / "relais_foyer.jsonl"
    if not chemin.is_file():
        return None
    return {
        str(m.get("destinataire_id"))
        for r in _jsonl(chemin) if not r.get("refus")
        for m in r.get("messages") or [] if m.get("parle")
    }


def sous_role(role: str, person_id: str, informes: set[str] | None) -> str:
    """`co_resident` → `co_resident_informe` / `co_resident_non_informe` when a relay took place.

    The `Rôle` column of `moves.csv` does not change during a run (it would break the per-role
    series): the distinction is made here, on reading.
    """
    if role != "co_resident" or informes is None:
        return role
    return "co_resident_informe" if str(person_id) in informes else "co_resident_non_informe"


# ── The check ───────────────────────────────────────────────────────────────────────────────
def controler(run_dir: Path | str) -> list[Constat]:
    """One finding per reader, per informed member and per non-informed member."""
    run = Path(run_dir)
    decl = _declaration(run)
    if str(decl.get("canal") or "") != "lu":
        return []
    n = int(((decl.get("service") or {}).get("jours_de_deplacement")) or JOURS_PAR_DEFAUT)
    sans_we = _sans_week_end(run)
    blocs = blocs_par_personne(run, sans_we)
    lignes = [l for l in _jsonl(run / "evenements.jsonl") if l.get("canal") == "lu"]

    constats: list[Constat] = []
    lecteurs: dict[str, date] = {}
    for l in lignes:
        pid = str(l.get("person_id"))
        quand = datetime.fromisoformat(str(l["horodatage_simule"])).date()
        role = "informé" if l.get("origine") == "entendu" else "lecteur"
        if role == "lecteur":
            lecteurs[pid] = quand
        constats.append(Constat(
            person_id=pid, role=role, date_lecture=quand,
            message=str(l.get("message") or ""), mineur=bool(l.get("mineur")),
        ))
    # The non-informed: reread from `relais_foyer.jsonl`, which is authoritative.
    for r in _jsonl(run / "relais_foyer.jsonl"):
        quand = lecteurs.get(str(r.get("lecteur_id")))
        if quand is None or r.get("refus"):
            continue
        for m in r.get("messages") or []:
            if not m.get("parle"):
                constats.append(Constat(
                    person_id=str(m.get("destinataire_id")), role="non informé",
                    date_lecture=quand, mineur=bool(m.get("mineur")),
                ))

    for c in constats:
        service = jours_de_service(c.date_lecture, n, sans_we)
        marque = MARQUE_LECTURE if c.role == "lecteur" else MARQUE_FOYER
        vus = blocs.get(c.person_id, [])
        for j in service:
            du_jour = [b for _, d, b in vus if d == j]
            c.jours.append(Jour(j, len(du_jour), sum(1 for b in du_jour if marque.search(b))))
            if j == service[0] and du_jour and c.attend_ligne:
                c.premiere = bool(marque.search(du_jour[0]))
        if c.mineur and c.role == "informé":
            c.modes_choisis = _modes_choisis(run, c.person_id, set(service))
    return constats


def rendre(constats: list[Constat]) -> list[str]:
    """The Markdown lines of the section — successes AND defects."""
    out = [
        "| Agent | Rôle | Lecture | Jour 1 : 1re décision | Décisions servies / jours de service | Verdict |",
        "|:--|:--|:--|:--|:--|:--|",
    ]
    for c in constats:
        premiere = {True: "✅ ligne présente", False: "🔴 **sans la ligne**", None: "—"}[
            c.premiere
        ] if c.attend_ligne else "—"
        detail = " · ".join(
            f"{j.jour:%d/%m} {j.avec_ligne}/{j.decisions}" for j in c.jours
        )
        role = c.role + (" (mineur)" if c.mineur else "")
        out.append(
            f"| `{c.person_id}` | {role} | {c.date_lecture:%d/%m} | {premiere} | {detail} | "
            f"{c.verdict} |"
        )
    for c in constats:
        if c.mineur and c.role == "informé":
            out.append(
                f"\n- `{c.person_id}` (mineur) — décision parentale reçue : « {c.message} » ; "
                f"modes choisis pendant ses jours de service : "
                f"{', '.join(c.modes_choisis) or 'aucun déplacement'}. Pas de score de "
                f"conformité : l'enfant décide avec son propre appel."
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    constats = controler(args.run)
    if not constats:
        print(f"{args.run.name}: no `lu` event to check.")
        return 0
    print("\n".join(rendre(constats)))
    return 1 if any(c.verdict == "🔴" for c in constats) else 0


if __name__ == "__main__":
    raise SystemExit(main())
