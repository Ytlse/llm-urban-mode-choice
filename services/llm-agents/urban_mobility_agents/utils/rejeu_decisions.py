"""Decision replay trace — ticket 090.

THE PROBLEM
-----------
On a hot resume, GAMA restarts from its `starting_date` and **replays** the days already lived
to rebuild its state: agent positions, and above all where each one's car is. Memory, for its
part, is frozen (ticket 075) — agents move around without relearning anything. But they
**re-decide**, and with the semantic cache off — the condition of a run logged over its full
scope — each replayed decision is paid to the model again. Measured on 2026-09-16: eight
replayed days, about a hundred decisions, three quarters of an hour of network wait to get
back to an already known state.

WHAT THIS MODULE DOES
---------------------
During normal life, it records each decision under its key `(person, activity, instant)`.
During the freeze window, it serves it again without calling the model.

⚠ **This is NOT the semantic cache**, and the distinction is what makes this lot acceptable
where the cache is not:

| | Semantic cache | Replay trace |
|---|---|---|
| Source | directory shared between runs | the workdir of THIS run |
| Key | state fingerprint, **not indexed by model** | `(person, activity, instant)` |
| Scope | the whole run | only while `reprise.gel_actif()` |
| Nothing matches | silent recomputation | counted, then **alarmed** above a threshold |

⚠ **The window is bounded on the DECISION time, not on the activity time** (2026-09-28).
A decision is often taken the day before: on the resume of the a13 v5 control, 1320712 had
chosen on 25/03 at 19:45 its trip of 26/03 at 18:16, and the resume point fell on 26/03 at
12:45. Bounded on the activity instant, the trace discarded this decision — and six others in
the same case — although the run had indeed taken it before the point. The replay asked the
model again with memory frozen at 26/03, that is a choice the agent could never have made that
evening: paid for and wrong outside the common prefix, refused (409) in the common prefix. Each
decision therefore now carries `decide_a`, the simulated clock of the step where it was asked.

The cache can serve again a decision taken by another model under another prompt: that is
why it is off on measurement runs. Here, only what THIS run itself produced is served again,
and only to replay days it has already lived.

⚠ **`llm_exchanges.jsonl` cannot serve as a source.** It is written on the worker side and its
`sim_ts` is `min(departure_timestamp)` of the BATCH: five agents of the same batch share a
timestamp there. The controller therefore cannot predict, before the call, the key under which
its response will be recorded. The trace is written here, on the controller side, where the
key is known.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

FICHIER = "decisions_rejeu.jsonl"

# Above this share of missed keys, the replay has diverged: it is no longer an identical
# resume, and saying so is better than discovering it on a curve three weeks later.
SEUIL_ALARME_MANQUEES = 0.20

_index: dict[str, dict] = {}
_chemin: Path | None = None
# Simulated clock of the last synchronised step, set by the controller at each sync: it is
# the decision time that `tracer` records in `decide_a`.
_horloge: int | None = None
_servies = 0
_manquees = 0
_alarme_levee = False


def _cle(personne: str, activite: str, instant: float) -> str:
    return f"{personne}|{activite}|{int(instant)}"


def chemin(workdir: str | Path) -> Path:
    return Path(workdir) / FICHIER


def noter_horloge(timestamp_simule: float) -> None:
    """The controller sets it at each sync, before any decision of the step."""
    global _horloge
    _horloge = int(timestamp_simule)


def horloge() -> int | None:
    """The simulated clock of the current step, or None before the first sync."""
    return _horloge


def charger(workdir: str | Path, jusqu_a: float | None = None) -> int:
    """Indexes the run's trace, keeping only the decisions TAKEN before the resume point.

    The criterion is `decide_a`, the decision time — not `instant`, the activity time, which
    can fall after the point for a decision taken the day before. A line without `decide_a`
    (trace written before 2026-09-28) is kept: the trace is only consulted during the freeze,
    and a decision taken after the point is asked again after the thaw, when nothing consults
    it any more. These lines are counted in the log.

    A missing trace is the NORMAL case of the first run: 0 is returned with nothing alarming.
    An unreadable trace must not stop the run — it is ignored, and says so.
    """
    global _index, _chemin
    _index = {}
    _chemin = chemin(workdir)
    if not _chemin.is_file():
        logger.info(f"[rejeu] no decision trace in {workdir} — nothing to serve again")
        return 0
    gardees = 0
    ignorees = 0
    posterieures = 0
    sans_heure = 0
    for ligne in _chemin.read_text(encoding="utf-8", errors="replace").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            enr = json.loads(ligne)
            cle = _cle(enr["personne"], enr["activite"], enr["instant"])
            if jusqu_a is not None:
                if enr.get("decide_a") is None:
                    sans_heure += 1
                elif float(enr["decide_a"]) > float(jusqu_a):
                    posterieures += 1
                    continue
            _index[cle] = enr
            gardees += 1
        except (ValueError, KeyError, TypeError):
            ignorees += 1
    if ignorees:
        logger.warning(
            f"[rejeu] {ignorees} unreadable line(s) ignored in {_chemin.name} — "
            f"the replay will proceed without them"
        )
    logger.info(
        f"[rejeu] {gardees} decision(s) indexed from {_chemin.name}"
        + (
            f", taken before the resume point ({jusqu_a}); {posterieures} taken after, "
            f"discarded; {sans_heure} without a decision time (trace older than "
            f"2026-09-28), kept"
            if jusqu_a is not None
            else ""
        )
    )
    return gardees


def tracer(
    personne: str,
    activite: str,
    instant: float,
    *,
    code_plan: str,
    raison: str,
    fournisseur: str,
    distribution: dict | None,
    decide_a: float | None = None,
) -> None:
    """Records a live decision. No effect if the file cannot be opened.

    `decide_a`: the simulated clock read when the decision was ASKED — the caller reads it
    before waiting for the model. Failing that, the clock of the current step.
    """
    if _chemin is None:
        return
    if decide_a is None:
        decide_a = _horloge
    enr = {
        "personne": str(personne),
        "activite": str(activite),
        "instant": float(instant),
        "decide_a": float(decide_a) if decide_a is not None else None,
        "code_plan": code_plan,
        "raison": raison,
        "fournisseur": fournisseur,
        "distribution": distribution or {},
    }
    try:
        with _chemin.open("a", encoding="utf-8") as f:
            f.write(json.dumps(enr, ensure_ascii=False) + "\n")
    except OSError as e:
        # A failing trace must never break a running run.
        logger.warning(f"[rejeu] trace not written ({e})")


def chercher(personne: str, activite: str, instant: float) -> dict | None:
    """The decision already taken for this key, or None. Counts served and missed."""
    global _servies, _manquees, _alarme_levee
    enr = _index.get(_cle(personne, activite, instant))
    if enr is None:
        _manquees += 1
        total = _servies + _manquees
        if (
            not _alarme_levee
            and total >= 20
            and _manquees / total > SEUIL_ALARME_MANQUEES
        ):
            _alarme_levee = True
            logger.error(
                f"[ALARME] Replay divergent: {_manquees}/{total} replayed decisions were not "
                f"found in the trace ({_manquees / total:.0%}). The replayed run does not "
                f"make the same choices as the first time — the rebuilt state is not the one "
                f"believed to be resumed."
            )
        return None
    _servies += 1
    return enr


def bilan() -> str:
    total = _servies + _manquees
    part = f"{_servies / total:.0%}" if total else "—"
    return (
        f"[rejeu] bilan : {_servies} décision(s) resservie(s) sans appel au modèle, "
        f"{_manquees} manquée(s), {part} de la fenêtre de rejeu épargnée"
    )


def reinitialiser() -> None:
    """For tests: clears the index and the counters."""
    global _index, _chemin, _servies, _manquees, _alarme_levee, _horloge
    _index = {}
    _chemin = None
    _horloge = None
    _servies = 0
    _manquees = 0
    _alarme_levee = False
