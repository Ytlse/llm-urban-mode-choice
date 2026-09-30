#!/usr/bin/env python3
"""
llm_capacity.py — LLM "throughput vs capacity" analysis of the latest run.

Answers two sizing questions, solely from the artefacts
on disk (no instrumentation added — everything is already logged):

  1. Does the rate of planning requests exceed the current LLM capacity?
  2. Is there enough *simulated* time left to drain the most critical task
     of the stack before its deadline (slack)?

Sources:
  - llm_exchanges.jsonl : each exchange = 1 prompt (after micro-batching);
    its `response` field lists the agents of the batch → demand BEFORE batching.
  - llm_errors.jsonl    : provider errors (429 = rate-limit / saturation).
  - app.log             : predictive backpressure telemetry `[predictive]`
    (D=rate/min, file=EDF stack, T≈drain time, slack_min=simulated time left
    on the critical task) + `[BACKPRESSURE]` episodes + `[ALARME] Gateway`.

Usage:
    python3 scripts/debug/llm_capacity.py [RUN_DIR] [--top N] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

TOP_DEFAULT = 8
TH_429_SHARE = 0.10          # share of active minutes with 429 → saturation
TH_SLACK_RATIO = 1.5         # T (drain) vs slack_min → deadline risk if slack < T*ratio

# [predictive] /sync retenu 4.4s (D=56.9/min, R×100.0, file=578, T≈609s, slack_min=35303s sim) — relâché
_PRED_RE = re.compile(
    r"retenu ([\d.]+)s \(D=([\d.]+)/min, R×([\d.]+), file=(\d+), T≈(\d+)s, slack_min=(-?\d+)s"
)
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def _iter_json_concat(path: Path):
    buf = ""
    with path.open(encoding="utf-8") as f:
        for line in f:
            buf += line
            try:
                yield json.loads(buf)
                buf = ""
            except json.JSONDecodeError:
                continue


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _minute(iso_or_ts: str) -> str | None:
    """Bucket 'YYYY-MM-DD HH:MM' from an ISO-8601 string."""
    try:
        return datetime.fromisoformat(iso_or_ts).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return None


# ── Sources ─────────────────────────────────────────────────────────────────

def load_exchanges(run: Path):
    """Returns (prompts_per_min, agents_per_min, tokens_in_per_min, batch_sizes)."""
    path = run / "llm_exchanges.jsonl"
    prompts = Counter()
    agents = Counter()
    tok_in = Counter()
    batch_sizes: list[int] = []
    if not path.exists():
        return prompts, agents, tok_in, batch_sizes
    run_name = run.resolve().name
    for o in _iter_json_concat(path):
        # Exchange of another client, served by the same worker during this run: out of scope.
        if o.get("origine") not in (None, run_name):
            continue
        m = _minute(o.get("time", ""))
        if not m:
            continue
        prompts[m] += 1
        resp = o.get("response")
        size = len(resp) if isinstance(resp, list) else 1
        batch_sizes.append(size)
        agents[m] += size
        tok_in[m] += int(o.get("tokens_in") or 0)
    return prompts, agents, tok_in, batch_sizes


def load_429(run: Path):
    """429 per minute + per provider."""
    path = run / "llm_errors.jsonl"
    per_min = Counter()
    per_provider = Counter()
    total = 0
    if not path.exists():
        return per_min, per_provider, total
    for o in _iter_jsonl(path):
        total += 1
        if o.get("http_status") == 429:
            m = _minute(o.get("time", ""))
            if m:
                per_min[m] += 1
            per_provider[o.get("provider", "?")] += 1
    return per_min, per_provider, total


def load_app_log(run: Path):
    """Parse [predictive], [BACKPRESSURE] episodes, [ALARME] Gateway."""
    path = run / "app.log"
    pred = []  # dicts: hold, D, file, T, slack
    bp_active = bp_release = gateway_alarm = 0
    suspended_s = 0.0
    active_since: datetime | None = None
    if not path.exists():
        return pred, bp_active, bp_release, gateway_alarm, suspended_s

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _PRED_RE.search(line)
            if m:
                pred.append({
                    "hold": float(m.group(1)), "D": float(m.group(2)),
                    "file": int(m.group(4)), "T": int(m.group(5)),
                    "slack": int(m.group(6)),
                })
            if "[ALARME] Gateway LLM" in line:
                gateway_alarm += 1
            if "Alarme gateway active" in line:
                bp_active += 1
                ts = _TS_RE.match(line)
                if ts and active_since is None:
                    active_since = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
            elif "Pile drainée" in line:
                bp_release += 1
                ts = _TS_RE.match(line)
                if ts and active_since is not None:
                    dt = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
                    suspended_s += max(0.0, (dt - active_since).total_seconds())
                    active_since = None
    return pred, bp_active, bp_release, gateway_alarm, suspended_s


# ── Report ──────────────────────────────────────────────────────────────────

def build(run: Path, top: int) -> str:
    out: list[str] = []
    alarms: list[str] = []

    prompts, agents, tok_in, batch_sizes = load_exchanges(run)
    err429_min, err429_prov, err_total = load_429(run)
    pred, bp_active, bp_release, gateway_alarm, suspended_s = load_app_log(run)

    out.append(f"# 🚦 Capacité LLM vs demande — `{run.name}`\n")

    # ── 1. Throughput vs demand (per wall-clock minute) ─────────────────────
    out.append("## 📈 Débit vs demande (par minute, horloge mur)\n")
    if prompts:
        pv = list(prompts.values())
        av = list(agents.values())
        avg_batch = mean(batch_sizes) if batch_sizes else 0
        out.append(
            f"- **Avant micro-batching** (agents demandant une planif) : "
            f"moy **{mean(av):.0f}/min**, pic **{max(av)}/min** "
            f"({sum(av)} agents au total)\n"
            f"- **Après micro-batching** (prompts LLM envoyés) : "
            f"moy **{mean(pv):.0f}/min**, pic **{max(pv)}/min** "
            f"({sum(pv)} prompts)\n"
            f"- **Taille de batch** : moy {avg_batch:.1f}, max {max(batch_sizes)} "
            f"→ facteur de compression ×{avg_batch:.1f}\n"
        )
        # Busiest minutes (demand before batching), with 429 alongside
        busiest = sorted(agents.items(), key=lambda kv: -kv[1])[:top]
        out.append("\n**Minutes les plus chargées**\n")
        out.append("| Minute | Agents/min | Prompts/min | 429 |")
        out.append("|:--|--:|--:|--:|")
        for m, a in busiest:
            out.append(f"| {m} | {a} | {prompts.get(m, 0)} | {err429_min.get(m, 0)} |")
    else:
        out.append("_llm_exchanges.jsonl absent ou vide._\n")

    # ── 2. EDF predictive backpressure ([predictive]) ───────────────────────
    out.append("\n## 🧮 Contre-pression prédictive (EDF)\n")
    if pred:
        D = [p["D"] for p in pred]
        files = [p["file"] for p in pred]
        T = [p["T"] for p in pred]
        slack = [p["slack"] for p in pred]
        holds = [p["hold"] for p in pred]
        min_slack = min(slack)
        max_T = max(T)
        out.append(
            f"- {len(pred)} points de rétention `[predictive]` · "
            f"rétention cumulée **{sum(holds):.0f}s**\n"
            f"- **Débit LLM D** : min {min(D):.0f}, moy {mean(D):.0f}, max {max(D):.0f} /min\n"
            f"- **Pile EDF (file)** : min {min(files)}, moy {mean(files):.0f}, max {max(files)}\n"
            f"- **T (écoulement estimé)** : moy {mean(T):.0f}s, max **{max_T}s**\n"
            f"- **slack_min (temps simulé restant sur la tâche critique)** : "
            f"min **{min_slack}s**, moy {mean(slack):.0f}s\n"
        )
        # Deadline risk: slack of the critical task close to the drain time
        if min_slack < max_T * TH_SLACK_RATIO:
            alarms.append(
                f"🔴 Risque d'échéance : slack_min={min_slack}s sim proche du temps "
                f"d'écoulement T={max_T}s — la tâche critique peut manquer sa deadline."
            )
    else:
        out.append(
            "_Aucune ligne `[predictive]` (contre-pression prédictive inactive ou "
            "jamais déclenchée sur ce run)._\n"
        )

    # ── 3. Hysteresis backpressure + gateway alarms ─────────────────────────
    out.append("\n## 🛑 Contre-pression (drainage) & alarmes\n")
    out.append(
        f"- **{bp_active} suspensions** de soumission / {bp_release} reprises "
        f"(drainage) · temps suspendu cumulé ≈ **{suspended_s:.0f}s**\n"
        f"- **{gateway_alarm} alarmes `[ALARME] Gateway LLM`** "
        f"(≥10 tâches échouées d'affilée — providers saturés)\n"
    )
    if gateway_alarm:
        alarms.append(
            f"🔴 {gateway_alarm} alarmes gateway (providers saturés/indisponibles) — "
            f"des décisions LLM ne reviennent plus par moments."
        )

    # ── 4. Provider saturation (429) ────────────────────────────────────────
    out.append("\n## ⛔ Saturation providers (429)\n")
    total_429 = sum(err429_prov.values())
    if err_total:
        active_minutes = set(prompts) or set(err429_min)
        min_with_429 = sum(1 for m in active_minutes if err429_min.get(m, 0) > 0)
        share = min_with_429 / len(active_minutes) if active_minutes else 0
        out.append(
            f"- **{total_429} erreurs 429** sur {err_total} erreurs LLM · "
            f"{min_with_429}/{len(active_minutes)} minutes actives touchées ({share:.0%})\n"
        )
        if err429_prov:
            out.append("\n**429 par provider**\n")
            out.append("| Provider | 429 |")
            out.append("|:--|--:|")
            for prov, n in err429_prov.most_common(top):
                out.append(f"| {prov} | {n} |")
        if share >= TH_429_SHARE:
            alarms.append(
                f"🟠 Saturation soutenue : {share:.0%} des minutes actives subissent des "
                f"429 — la demande dépasse régulièrement la capacité des providers."
            )
    else:
        out.append("_llm_errors.jsonl absent — pas de 429 mesurable._\n")

    # ── ALARMS on top ───────────────────────────────────────────────────────
    banner = ["## 🚨 ALARMES\n"]
    banner += ([f"- {a}" for a in alarms] + [""]) if alarms else ["_Capacité LLM tenue : aucun seuil franchi._\n"]
    out[1:1] = banner

    return "\n".join(out).rstrip() + "\n"


def resolve_run(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return (Path(__file__).resolve().parents[2] / "experiments" / "current").resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--top", type=int, default=TOP_DEFAULT)
    ap.add_argument("--out")
    args = ap.parse_args()

    run = resolve_run(args.run_dir)
    if not run.exists():
        print(f"❌ Run directory not found: {run}", file=sys.stderr)
        return 1

    report = build(run, args.top)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"✅ Report written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
