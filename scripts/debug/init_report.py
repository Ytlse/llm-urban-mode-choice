#!/usr/bin/env python3
"""
init_report.py — Analysis of the initialisation phase of a run.

Complements `run_report.py` (overall run health) and `llm_capacity.py`
(LLM throughput vs capacity). Here the focus is exclusively on the **startup**
of the simulation, before GAMA plays:

  - ⏱️ Timeline of the 5 INITIALISATION steps (duration of each), from SIM_START to
    INIT_DONE — spots the expensive step (almost always the bootstrap).
  - 💾 Wiring & warm-up of the 3 persistent caches (OTP, OSMnx, semantic LLM):
    enabled? paths? hit rate reached at the end of init (warm-up)?
  - 🔥 Bootstrap: number of pre-computed agents, itinerary waves, rise of the
    cache hit rate across agents (cold → warm), cost per activity.
  - 🚨 Init bugs: event loop stalls (blocking I/O → WS 1006 disconnects),
    LTM metadata cache thrashing (gc.collect in a loop), unreachable ODs.

All measurements are derived from `app.log` (or `controller.log`). Stdlib only,
tolerant to missing files/lines.

Usage:
    python3 scripts/debug/init_report.py [RUN_DIR] [--out FILE]

Default RUN_DIR: experiments/current (resolved from the repo root).
Without --out, the report is written to stdout.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── ALARM thresholds (adjustable) ───────────────────────────────────────────
TH_INIT_DURATION_S = 300.0      # init > 5 min → slow startup (often bootstrap)
TH_STALL_S = 30.0               # single event-loop stall above this → WS disconnects
TH_STALL_TOTAL_S = 60.0         # total init stalls above this → asyncio loop saturated
TH_METADATA_CLEANUPS = 200      # LTM cache evictions during init → thrashing + gc
TH_WARM_HIT_RATE = 0.90         # OTP/OSMnx hit rate at end of init below this → cold cache
TH_UNREACHABLE = 20             # unreachable ODs during the bootstrap

# ── Log formats ─────────────────────────────────────────────────────────────
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+)\s+\| (.+)$")

# Timeline
_RE_SIM_START = re.compile(r"event=SIM_START\b.*?sim_time=(.+?)\s*$")
_RE_INIT_STEP = re.compile(r"INITIALISATION (\d+)/(\d+)\s+(.+?)(?:\s+—|\s+-\s|$)")
_RE_INIT_DONE = re.compile(r"event=INIT_DONE\b.*?agents=(\d+).*?init_duration_s=(\d+)")

# Cache wiring
_RE_OTP_WIRED = re.compile(r"\[cache\] OTP itinerary cache wired")
_RE_OSMNX_CACHE = re.compile(r"\[osmnx-cache\] Persistent cache enabled at (\S+)")
_RE_OTP_CACHE = re.compile(r"\[otp-cache\] Persistent cache enabled at (\S+)")
_RE_LLM_CACHE_INIT = re.compile(r"LlmSemanticCache initialisé — répertoire: (\S+?),?\s*seuil: ([\d.]+)")
_RE_LLM_CACHE_ISO = re.compile(r"LLM cache isolé par prompt — checksum=(\w+)")
_RE_EMB_LOAD = re.compile(r"Chargement du modèle d'embedding LLM cache")

# Periodic combined cache summary
_RE_CACHE_SUMMARY = re.compile(
    r"\[cache\] OTP (\d+)% \((\d+)/(\d+)\) · OSMnx (\d+)% \((\d+)/(\d+)\) · "
    r"LLM (\d+)% \((\d+)/(\d+)\)(?: \[miss: no_candidates=(\d+)\])?"
)

# Bootstrap
_RE_BS_START = re.compile(r"\[bootstrap\].*computing itineraries for (\d+)/(\d+) agents")
_RE_BS_PROGRESS = re.compile(
    r"\[bootstrap\] progress (\d+)/(\d+) \((\d+)%\) — cache hits: (\d+)/(\d+) \((\d+)%\)"
)
_RE_BS_WAVE = re.compile(r"\[bootstrap\] pre-computing (\d+) act\[N\+(\d+)\] itineraries \(wave (\d+)\)")
_RE_BS_DONE = re.compile(
    r"\[bootstrap\] all activities pre-planning tasks done \((\d+) future moves pre-cached "
    r"across (\d+) additional wave"
)
_RE_BS_HITMISS = re.compile(r"\[bootstrap\] cache (hit|miss) — \S+ / (\w+)")

# Bugs
_RE_STALL = re.compile(r"\[ALARME\] Event loop bloquée pendant ~([\d.]+)s")
_RE_WS_1006 = re.compile(r"Connection closed: 1006")
_RE_WS_SENDFAIL = re.compile(r"Send failed: no close frame")
_RE_METADATA_CLEANUP = re.compile(r"Cleaned up metadata cache")
_RE_UNREACHABLE = re.compile(r"Can't get to destination")


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _find_log(run: Path) -> Path | None:
    """Picks the controller log: app.log first, otherwise controller.log."""
    for cand in (run / "app.log", run / "gama_results" / "controller.log"):
        if cand.exists():
            return cand
    logs = sorted(run.glob("*.log"))
    return logs[0] if logs else None


class InitData:
    """State accumulated in a single pass over the log."""

    def __init__(self) -> None:
        self.sim_start_ts: datetime | None = None
        self.sim_start_simtime: str | None = None
        self.steps: list[tuple[int, str, datetime]] = []  # (num, label, ts)
        self.init_done_ts: datetime | None = None
        self.init_done_agents: int | None = None
        self.init_duration_s: int | None = None

        self.otp_wired = False
        self.otp_cache_path: str | None = None
        self.osmnx_cache_path: str | None = None
        self.llm_cache_path: str | None = None
        self.llm_cache_seuil: str | None = None
        self.emb_load_ts: datetime | None = None
        self.llm_cache_ready_ts: datetime | None = None

        # Combined summary: (last before INIT_DONE) and (very last)
        self.summary_at_init: tuple | None = None
        self.summary_last: tuple | None = None

        self.bs_start_ts: datetime | None = None
        self.bs_agents: int | None = None
        self.bs_progress: list[tuple[int, int, int, int]] = []  # (done,total,hits,hits_total)
        self.bs_waves: list[tuple[int, int]] = []  # (wave, count)
        self.bs_moves_precached: int | None = None
        self.bs_hitmiss = Counter()  # hit/miss
        self.bs_by_activity = Counter()

        # Bugs, windowed on the init
        self.stalls: list[float] = []
        self.ws_1006 = 0
        self.ws_sendfail = 0
        self.metadata_cleanups = 0
        self.metadata_cleanups_total = 0
        self.unreachable = 0


def parse(log: Path) -> InitData:
    d = InitData()
    with log.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TS_RE.match(line.rstrip())
            if not m:
                continue
            ts = _parse_ts(m.group(1))
            msg = m.group(3)

            # ── Timeline ──
            mm = _RE_SIM_START.search(msg)
            if mm:
                d.sim_start_ts = ts
                d.sim_start_simtime = mm.group(1).strip()
                continue
            mm = _RE_INIT_STEP.search(msg)
            if mm:
                d.steps.append((int(mm.group(1)), mm.group(3).strip(), ts))
                continue
            mm = _RE_INIT_DONE.search(msg)
            if mm:
                d.init_done_ts = ts
                d.init_done_agents = int(mm.group(1))
                d.init_duration_s = int(mm.group(2))
                continue

            in_init = d.init_done_ts is None  # still in the init phase

            # ── Cache wiring ──
            if _RE_OTP_WIRED.search(msg):
                d.otp_wired = True
                continue
            mm = _RE_OSMNX_CACHE.search(msg)
            if mm:
                d.osmnx_cache_path = mm.group(1)
                continue
            mm = _RE_OTP_CACHE.search(msg)
            if mm:
                d.otp_cache_path = mm.group(1)
                continue
            mm = _RE_LLM_CACHE_INIT.search(msg)
            if mm:
                d.llm_cache_path = mm.group(1)
                d.llm_cache_seuil = mm.group(2)
                d.llm_cache_ready_ts = ts
                continue
            if _RE_EMB_LOAD.search(msg):
                d.emb_load_ts = ts
                continue

            # ── Combined cache summary ──
            mm = _RE_CACHE_SUMMARY.search(msg)
            if mm:
                g = mm.groups()
                summary = {
                    "otp_pct": int(g[0]), "otp_hit": int(g[1]), "otp_tot": int(g[2]),
                    "osmnx_pct": int(g[3]), "osmnx_hit": int(g[4]), "osmnx_tot": int(g[5]),
                    "llm_pct": int(g[6]), "llm_hit": int(g[7]), "llm_tot": int(g[8]),
                    "no_candidates": int(g[9]) if g[9] else 0,
                }
                d.summary_last = summary
                if in_init:
                    d.summary_at_init = summary
                continue

            # ── Bootstrap ──
            mm = _RE_BS_START.search(msg)
            if mm:
                d.bs_start_ts = ts
                d.bs_agents = int(mm.group(2))
                continue
            mm = _RE_BS_PROGRESS.search(msg)
            if mm:
                d.bs_progress.append(
                    (int(mm.group(1)), int(mm.group(2)), int(mm.group(4)), int(mm.group(5)))
                )
                continue
            mm = _RE_BS_WAVE.search(msg)
            if mm:
                d.bs_waves.append((int(mm.group(3)), int(mm.group(1))))
                continue
            mm = _RE_BS_DONE.search(msg)
            if mm:
                d.bs_moves_precached = int(mm.group(1))
                continue
            mm = _RE_BS_HITMISS.search(msg)
            if mm:
                d.bs_hitmiss[mm.group(1)] += 1
                d.bs_by_activity[mm.group(2)] += 1
                continue

            # ── Bugs (windowed on init) ──
            mm = _RE_STALL.search(msg)
            if mm:
                if in_init:
                    d.stalls.append(float(mm.group(1)))
                continue
            if _RE_METADATA_CLEANUP.search(msg):
                d.metadata_cleanups_total += 1
                if in_init:
                    d.metadata_cleanups += 1
                continue
            if in_init and _RE_WS_1006.search(msg):
                d.ws_1006 += 1
                continue
            if in_init and _RE_WS_SENDFAIL.search(msg):
                d.ws_sendfail += 1
                continue
            if in_init and _RE_UNREACHABLE.search(msg):
                d.unreachable += 1
                continue
    return d


# ── Report sections ─────────────────────────────────────────────────────────

def section_timeline(d: InitData, out: list[str], alarms: list[str]) -> None:
    out.append("\n## ⏱️ Timeline de l'initialisation\n")

    # Init bounds: SIM_START → INIT_DONE
    t0 = d.sim_start_ts
    dur = d.init_duration_s
    if dur is None and t0 and d.init_done_ts:
        dur = int((d.init_done_ts - t0).total_seconds())

    if dur is not None:
        line = f"Durée d'init totale : **{_fmt_dur(dur)}**"
        if d.init_done_agents:
            line += f" · **{d.init_done_agents} agents** envoyés à GAMA"
        if d.sim_start_simtime:
            line += f" · sim_time départ `{d.sim_start_simtime}`"
        out.append(line + "\n")
        if dur >= TH_INIT_DURATION_S:
            alarms.append(
                f"🟠 Init lente : {_fmt_dur(dur)} avant le démarrage GAMA "
                f"(> {_fmt_dur(TH_INIT_DURATION_S)}) — voir l'étape la plus coûteuse ci-dessous."
            )
    else:
        out.append("_Bornes d'init incomplètes (SIM_START / INIT_DONE manquant)._\n")

    # Steps: duration = ts(step n+1) - ts(step n); last → INIT_DONE
    marks: list[tuple[str, datetime]] = []
    if t0:
        marks.append(("SIM_START", t0))
    for num, label, ts in d.steps:
        if ts:
            marks.append((f"{num}/5 {label}", ts))
    if d.init_done_ts:
        marks.append(("INIT_DONE", d.init_done_ts))

    if len(marks) >= 2:
        out.append("| Étape | Début | Durée | Part |")
        out.append("|:--|:--|--:|--:|")
        span = (marks[-1][1] - marks[0][1]).total_seconds() or 1.0
        longest = ("", 0.0)
        for i in range(len(marks) - 1):
            label, ts = marks[i]
            step_dur = (marks[i + 1][1] - ts).total_seconds()
            share = step_dur / span
            if step_dur > longest[1]:
                longest = (label, step_dur)
            bar = "█" * int(round(share * 12))
            out.append(
                f"| {label[:44]} | {ts.strftime('%H:%M:%S')} | {_fmt_dur(step_dur)} | "
                f"{share:.0%} {bar} |"
            )
        out.append("")
        if longest[0]:
            out.append(
                f"_Étape dominante : **{longest[0][:60]}** "
                f"({_fmt_dur(longest[1])})._\n"
            )


def section_caches(d: InitData, out: list[str], alarms: list[str]) -> None:
    out.append("\n## 💾 Caches persistants (câblage & réchauffage)\n")

    out.append("| Cache | Câblé | Chemin / détail |")
    out.append("|:--|:--:|:--|")
    otp_ok = "✅" if (d.otp_wired or d.otp_cache_path) else "❌"
    out.append(f"| OTP (itinéraires TC) | {otp_ok} | `{d.otp_cache_path or '—'}` |")
    osmnx_ok = "✅" if d.osmnx_cache_path else "❌"
    out.append(f"| OSMnx (routes) | {osmnx_ok} | `{d.osmnx_cache_path or '—'}` |")
    llm_ok = "✅" if d.llm_cache_path else "❌"
    llm_detail = f"`{d.llm_cache_path}`" if d.llm_cache_path else "—"
    if d.llm_cache_seuil:
        llm_detail += f" · seuil sémantique **{d.llm_cache_seuil}**"
    out.append(f"| LLM (sémantique) | {llm_ok} | {llm_detail} |")
    out.append("")

    if not (d.otp_wired or d.otp_cache_path):
        alarms.append("🔴 Cache OTP non câblé — chaque itinéraire TC recalculé (appels OTP répétés).")
    if not d.osmnx_cache_path:
        alarms.append("🔴 Cache OSMnx non activé — recalcul des routes voiture/vélo à chaque fois.")
    if not d.llm_cache_path:
        alarms.append("🔴 Cache LLM sémantique non initialisé — aucun hit possible, coût LLM plein.")

    # Loading cost of the embedding model (blocking)
    if d.emb_load_ts and d.llm_cache_ready_ts:
        emb = (d.llm_cache_ready_ts - d.emb_load_ts).total_seconds()
        if emb > 0:
            out.append(f"Chargement du modèle d'embedding LLM cache : **{_fmt_dur(emb)}**\n")

    # Hit rate reached at the end of init (warm-up), from the combined summary
    s = d.summary_at_init or d.summary_last
    if s:
        when = "en fin d'init" if d.summary_at_init else "en fin de run (init non bornée)"
        out.append(f"**Réchauffage atteint {when}** (résumé combiné) :\n")
        out.append("| Cache | Hit rate | Hits / lookups |")
        out.append("|:--|--:|--:|")
        out.append(f"| OTP | {s['otp_pct']}% | {s['otp_hit']}/{s['otp_tot']} |")
        out.append(f"| OSMnx | {s['osmnx_pct']}% | {s['osmnx_hit']}/{s['osmnx_tot']} |")
        out.append(f"| LLM | {s['llm_pct']}% | {s['llm_hit']}/{s['llm_tot']} |")
        out.append("")
        if s["no_candidates"]:
            out.append(
                f"_LLM : {s['no_candidates']} miss `no_candidates` "
                f"(aucun voisin sémantique au-dessus du seuil)._\n"
            )
        for name, pct, tot in (
            ("OTP", s["otp_pct"], s["otp_tot"]),
            ("OSMnx", s["osmnx_pct"], s["osmnx_tot"]),
        ):
            if tot > 50 and pct / 100.0 < TH_WARM_HIT_RATE:
                alarms.append(
                    f"🟠 Cache {name} peu réchauffé : {pct}% de hit en fin d'init "
                    f"(< {TH_WARM_HIT_RATE:.0%}) — clés de cache instables ou données absentes ?"
                )
    else:
        out.append("_Pas de ligne de résumé combiné `[cache] OTP … OSMnx … LLM …` trouvée._\n")


def section_bootstrap(d: InitData, out: list[str], alarms: list[str]) -> None:
    if d.bs_agents is None and not d.bs_progress and not d.bs_waves:
        return
    out.append("\n## 🔥 Bootstrap (pré-calcul des premiers itinéraires)\n")

    dur = ""
    if d.bs_start_ts and d.init_done_ts:
        dur = f" en **{_fmt_dur((d.init_done_ts - d.bs_start_ts).total_seconds())}**"
    head = f"Pré-calcul pour **{d.bs_agents or '?'} agents**{dur}"
    if d.bs_moves_precached is not None:
        head += f" · **{d.bs_moves_precached} futurs déplacements** pré-cachés"
    if d.bs_waves:
        head += f" · {len(d.bs_waves)} vague(s) d'anticipation"
    out.append(head + ".\n")

    # Rise of the hit rate (cold → warm)
    if d.bs_progress:
        first = d.bs_progress[0]
        last = d.bs_progress[-1]
        first_rate = first[2] / first[3] if first[3] else 0.0
        last_rate = last[2] / last[3] if last[3] else 0.0
        out.append(
            f"Montée du cache pendant le bootstrap : "
            f"**{first_rate:.0%}** de hit à {first[0]}/{first[1]} agents → "
            f"**{last_rate:.0%}** à {last[0]}/{last[1]}.\n"
        )

    # Hit/miss per activity (the first computation per place is necessarily a miss)
    total_hm = sum(d.bs_hitmiss.values())
    if total_hm:
        hits = d.bs_hitmiss.get("hit", 0)
        misses = d.bs_hitmiss.get("miss", 0)
        out.append(
            f"Lookups tracés au bootstrap : {hits} hits / {misses} miss "
            f"({hits / total_hm:.0%} hit).\n"
        )
        if d.bs_by_activity:
            top = " · ".join(f"{a}:{n}" for a, n in d.bs_by_activity.most_common(6))
            out.append(f"_Par type d'activité : {top}._\n")


def section_bugs(d: InitData, out: list[str], alarms: list[str]) -> None:
    out.append("\n## 🐛 Anomalies de la phase d'init\n")
    found = False

    # 1) Event-loop stalls → WS 1006 disconnects
    if d.stalls:
        found = True
        mx = max(d.stalls)
        tot = sum(d.stalls)
        out.append(
            f"- **Event loop bloquée** : {len(d.stalls)} stall(s), "
            f"max **{mx:.0f}s**, cumul **{_fmt_dur(tot)}**. "
            f"Opération synchrone dans la boucle asyncio (bootstrap CPU / I/O bloquante)."
        )
        if d.ws_1006 or d.ws_sendfail:
            out.append(
                f"  ↳ Conséquence : {d.ws_1006} coupure(s) WS 1006 + "
                f"{d.ws_sendfail} `Send failed` pendant l'init."
            )
        if mx >= TH_STALL_S:
            alarms.append(
                f"🔴 Stall event-loop de {mx:.0f}s pendant l'init (> {TH_STALL_S:.0f}s) — "
                f"le calcul synchrone du bootstrap gèle le WebSocket (coupures 1006). "
                f"Piste : déporter le bootstrap hors de la boucle asyncio / le rendre incrémental."
            )
        elif tot >= TH_STALL_TOTAL_S:
            alarms.append(
                f"🟠 Cumul des stalls event-loop = {_fmt_dur(tot)} pendant l'init — "
                f"boucle asyncio saturée par du travail synchrone."
            )

    # 2) LTM metadata cache thrashing
    if d.metadata_cleanups:
        found = True
        out.append(
            f"- **Cache métadonnées LTM** : {d.metadata_cleanups} éviction(s) pendant l'init "
            f"({d.metadata_cleanups_total} sur tout le run). Chaque éviction relit et réécrit "
            f"les métadonnées sur disque — coûteux si répété."
        )
        if d.metadata_cleanups >= TH_METADATA_CLEANUPS:
            alarms.append(
                f"🟠 Thrashing cache LTM : {d.metadata_cleanups} évictions pendant l'init "
                f"(> {TH_METADATA_CLEANUPS}) — `agent.long_term_max_loaded_metadata` trop bas "
                f"vs. nb d'agents (llm/longterm.py:_cleanup_metadata_cache). "
                f"Piste : augmenter le plafond au-delà de la taille de population."
            )

    # 3) Unreachable ODs at bootstrap
    if d.unreachable:
        found = True
        out.append(
            f"- **OD injoignables** : {d.unreachable} « Can't get to destination » pendant "
            f"le bootstrap (itinéraire sans solution → fallback)."
        )
        if d.unreachable >= TH_UNREACHABLE:
            alarms.append(
                f"🟠 {d.unreachable} OD injoignables au bootstrap — couverture réseau/OTP "
                f"insuffisante ou bbox trop serrée."
            )

    if not found:
        out.append("_Aucune anomalie d'init détectée._\n")


def section_alarms(out: list[str], alarms: list[str]) -> None:
    banner = ["\n## 🚨 ALARMES INIT\n"]
    if not alarms:
        banner.append("_Aucune anomalie d'init franchissant les seuils._\n")
    else:
        for a in alarms:
            banner.append(f"- {a}")
        banner.append("")
    out[1:1] = banner


# ── Entry point ─────────────────────────────────────────────────────────────

def resolve_run(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    repo = Path(__file__).resolve().parents[2]
    return (repo / "experiments" / "current").resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="Run directory (default experiments/current)")
    ap.add_argument("--out", help="Write the report to this file instead of stdout")
    args = ap.parse_args()

    run = resolve_run(args.run_dir)
    if not run.exists():
        print(f"❌ Run directory not found: {run}", file=sys.stderr)
        return 1

    log = _find_log(run)
    if not log:
        print(f"❌ No log (app.log / controller.log) in {run}", file=sys.stderr)
        return 1

    d = parse(log)

    out: list[str] = []
    alarms: list[str] = []
    out.append(f"# 🚀 Rapport d'initialisation — `{run.name}`\n")
    out.append(
        f"_Généré {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"source `{log.relative_to(run) if log.is_relative_to(run) else log.name}`_\n"
    )

    section_timeline(d, out, alarms)
    section_caches(d, out, alarms)
    section_bootstrap(d, out, alarms)
    section_bugs(d, out, alarms)
    section_alarms(out, alarms)  # inserts at the top, must stay last

    report = "\n".join(out).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"✅ Init report written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
