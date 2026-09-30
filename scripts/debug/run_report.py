#!/usr/bin/env python3
"""
run_report.py — "Agent-ready" health report of the latest run.

Aggregates into a single dense markdown artefact the signals essential to debugging
a run (by default `experiments/current/`):

  - errors / warnings from app.log (normalised, top-N)
  - LLM health: errors per provider × http_status, 429 rate, tokens
  - pipeline latency (percentiles) + backlog detection
  - agent activity (inactive over time)
  - decisions: modal distribution, selection methods, fallbacks
  - arrivals: timeouts, departure delays
  - ALARMS: derived anomalies with thresholds

Usage:
    python3 scripts/debug/run_report.py [RUN_DIR] [--top N] [--out FILE]

Default RUN_DIR: experiments/current (resolved from the repo root).
Without --out, the report is written to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# ── ALARM thresholds (adjustable) ───────────────────────────────────────────
TH_PIPELINE_P95_S = 300.0       # pipeline p95 latency above this → backlog
TH_RATE_LIMIT_SHARE = 0.30      # share of 429 in LLM errors → saturation
TH_FALLBACK_SHARE = 0.05        # share of decisions in LLM fallback → degraded quality
TH_INACTIVE_FINAL_SHARE = 0.20  # share of inactive agents at end of run → stuck agents
TH_MISSED_ACTIVITY_SHARE = 0.05 # share of expected activities not run on a given day
SIM_DAY_SECONDS = 86400
TOP_DEFAULT = 12

# ── Message normalisation (taken from scripts/errors.py) ────────────────────
_NORMALIZE = [
    (re.compile(r"\btask_id=[0-9a-f-]{36}\b"), ""),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "{uuid}"),
    (re.compile(r"\[timestamp:[^\]]+\]"), ""),
    (re.compile(r"\blon=-?[\d.]+\s+lat=-?[\d.]+"), ""),
    (re.compile(r"from=\(-?[\d.]+,-?[\d.]+\)\s+to=\(-?[\d.]+,-?[\d.]+\)"), ""),
    (re.compile(r"\(\d+,\)\s+\(\d+,\)"), "({n},) ({m},)"),
    (re.compile(r"\b(waited|timeout)=[\d.]+s\b"), ""),
    (re.compile(r"\bpublic_transport=(True|False)\b"), ""),
    (re.compile(r"\bpour \d+:"), "pour {id}:"),
    (re.compile(r"\bperson \d+\b"), "person {id}"),
    (re.compile(r"\bfor \d+\b"), "for {id}"),
    (re.compile(r"\bto \d+\b"), "to {id}"),
    (re.compile(r"\b\d{4,}\b"), "{n}"),
    (re.compile(r"  +"), " "),
]
_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| (\w+)\s+\| (.+)$")


def _normalize(msg: str) -> str:
    for pattern, replacement in _NORMALIZE:
        msg = pattern.sub(replacement, msg)
    return msg.strip().rstrip("|").strip()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * q))
    return s[idx]


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _iter_jsonl(path: Path):
    """Yield objects from a JSONL file (one line = one object)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_json_concat(path: Path):
    """Yield objects from a file of concatenated/pretty-printed JSON (llm_exchanges)."""
    buf = ""
    with path.open(encoding="utf-8") as f:
        for line in f:
            buf += line
            try:
                yield json.loads(buf)
                buf = ""
            except json.JSONDecodeError:
                continue


# ── Report sections ─────────────────────────────────────────────────────────

def section_meta(run: Path, out: list[str], alarms: list[str]) -> None:
    out.append(f"# 🩺 Rapport de run — `{run.name}`\n")
    out.append(f"_Généré {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · dossier `{run}`_\n")

    params = {}
    for name in ("scenario_params.yaml",):
        p = run / name
        if p.exists():
            for line in p.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    params[k.strip()] = v.strip()

    meta = []
    if params:
        pop = params.get("population_size", "?")
        agents = params.get("number_of_llm_based_agents", "?")
        ltm = params.get("long_term_memory_enabled", "?")
        meta.append(f"- Population : **{pop}** · agents LLM : **{agents}** · LTM : {ltm}")

    # Simulated duration (via moves.csv "Temps simulé") + wall-clock duration (via app.log)
    out.append("\n".join(meta) + "\n" if meta else "")


def _discover_logs(run: Path) -> list[Path]:
    """`*.log` files at the root of the run (one per service: app, api, worker, …).

    Excludes gama_results/ (controller.log there duplicates app.log), .bak archives and
    loguru rotation files (`worker.<timestamp>.log`, whose stem contains a
    dot). app.log (controller) is listed first if it exists.
    """
    logs = sorted(
        p for p in run.glob("*.log")
        if p.is_file() and "." not in p.stem  # excludes rotation artefacts
    )
    logs.sort(key=lambda p: (p.name != "app.log", p.name))
    return logs


def section_logs(run: Path, out: list[str], alarms: list[str], top: int) -> None:
    logs = _discover_logs(run)
    if not logs:
        out.append("## 📜 Logs\n_Aucun fichier .log dans le run._\n")
        return

    # Combined counters (all services) + per-service totals for the header.
    counts = {"ERROR": Counter(), "WARNING": Counter()}
    per_service: dict[str, Counter] = {}
    first_ts = last_ts = None

    for log in logs:
        service = log.stem  # app, api, worker, …
        stotals = per_service.setdefault(service, Counter())
        with log.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_RE.match(line.rstrip())
                if not m:
                    continue
                level, msg = m.group(1), m.group(2)
                stotals[level] += 1
                ts = line[:19]
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
                if level in counts:
                    # [service] tag to trace the origin in the combined top.
                    counts[level][f"[{service}] {_normalize(msg)}"] += 1

    dur = ""
    if first_ts and last_ts:
        try:
            t0 = datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
            if t1 > t0:
                dur = f" · durée mur **{_fmt_dur((t1 - t0).total_seconds())}**"
        except ValueError:
            pass

    services = ", ".join(f"`{s}.log`" for s in per_service)
    out.append(f"## 📜 Logs — {len(logs)} service(s) : {services}{dur}\n")
    out.append("| Service | ERROR | WARNING | INFO |")
    out.append("|:--|--:|--:|--:|")
    for service, st in per_service.items():
        out.append(f"| {service} | {st['ERROR']} | {st['WARNING']} | {st['INFO']} |")

    for level, emoji in (("ERROR", "🔴"), ("WARNING", "🟠")):
        c = counts[level]
        if not c:
            continue
        out.append(f"\n### {emoji} Top {level} (normalisés, tous services)\n")
        out.append(f"| # | Source · {level.title()} |")
        out.append("|--:|:--|")
        for msg, n in c.most_common(top):
            out.append(f"| {n} | {msg[:150]} |")
        out.append(f"\n_{sum(c.values())} occurrences, {len(c)} types distincts._\n")


def section_llm(run: Path, out: list[str], alarms: list[str], top: int) -> None:
    errors_path = run / "llm_errors.jsonl"
    out.append("\n## 🤖 Santé LLM\n")

    by_provider = Counter()
    by_status = Counter()
    by_sig = Counter()
    total_err = 0
    rate_limited = 0
    if errors_path.exists():
        for o in _iter_jsonl(errors_path):
            total_err += 1
            prov = o.get("provider", "?")
            status = o.get("http_status")
            by_provider[prov] += 1
            by_status[str(status)] += 1
            if status == 429:
                rate_limited += 1
            by_sig[(prov, str(status), str(o.get("error_type", ""))[:44])] += 1

        share_429 = rate_limited / total_err if total_err else 0.0
        out.append(
            f"**{total_err} erreurs LLM** · dont **{rate_limited} rate-limit (429)** "
            f"({share_429:.0%})\n"
        )
        if total_err and share_429 >= TH_RATE_LIMIT_SHARE:
            alarms.append(
                f"🔴 Saturation LLM : {share_429:.0%} des erreurs sont des 429 "
                f"(demande > capacité providers)."
            )

        out.append("\n**Erreurs par provider** · **par statut HTTP**\n")
        out.append("| Provider | Err | | HTTP | Err |")
        out.append("|:--|--:|--|:--|--:|")
        prov_rows = by_provider.most_common()
        stat_rows = by_status.most_common()
        for i in range(max(len(prov_rows), len(stat_rows))):
            pl = f"{prov_rows[i][0]} | {prov_rows[i][1]}" if i < len(prov_rows) else " | "
            sl = f"{stat_rows[i][0]} | {stat_rows[i][1]}" if i < len(stat_rows) else " | "
            out.append(f"| {pl} | | {sl} |")

        out.append("\n**Top signatures d'erreur**\n")
        out.append("| # | Provider | HTTP | Type |")
        out.append("|--:|:--|:--|:--|")
        for (prov, status, etype), n in by_sig.most_common(top):
            out.append(f"| {n} | {prov} | {status} | {etype} |")
    else:
        out.append("_llm_errors.jsonl absent._\n")

    # Cache & exchange throughput
    hits_path = run / "llm_cache_hits.jsonl"
    exch_path = run / "llm_exchanges.jsonl"
    n_hits = sum(1 for _ in _iter_jsonl(hits_path)) if hits_path.exists() else 0
    n_exch = 0
    tok_in = tok_out = 0
    prov_calls = Counter()
    exch_times: list[datetime] = []
    # The log is written by the worker, shared by all clients: only the exchanges
    # signed by THIS run are counted (or unsigned, older than 2026-09-24).
    run_name = run.resolve().name
    n_etrangers = 0
    if exch_path.exists():
        for o in _iter_json_concat(exch_path):
            if o.get("origine") not in (None, run_name):
                n_etrangers += 1
                continue
            n_exch += 1
            tok_in += int(o.get("tokens_in") or 0)
            tok_out += int(o.get("tokens_out") or 0)
            prov_calls[o.get("provider", "?")] += 1
            t = o.get("time")
            if t:
                try:
                    exch_times.append(datetime.fromisoformat(t))
                except ValueError:
                    pass

    if n_etrangers:
        out.append(
            f"\n_{n_etrangers} échanges d'un AUTRE client (origine ≠ `{run_name}`) écartés du "
            f"compte : le worker les a servis pendant ce run._\n"
        )
    total_decisions = n_hits + n_exch
    hit_rate = n_hits / total_decisions if total_decisions else 0.0
    out.append(
        f"\n**Cache** : {n_hits} hits / {total_decisions} décisions → "
        f"**{hit_rate:.0%} hit rate**\n"
    )
    if n_exch:
        rate_line = ""
        if len(exch_times) >= 2:
            span_min = (max(exch_times) - min(exch_times)).total_seconds() / 60
            if span_min > 0:
                rate_line = f" · débit moyen **{n_exch / span_min:.0f} req/min** (mur)"
        out.append(
            f"**Appels LLM** : {n_exch} échanges · "
            f"{tok_in:,} tokens_in · {tok_out:,} tokens_out{rate_line}\n"
        )


def section_pipeline(run: Path, out: list[str], alarms: list[str]) -> None:
    path = run / "pipeline_timing.csv"
    if not path.exists():
        return
    lat = []
    retries = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t0 = float(row["T0"])
                tf = float(row["T_fin"])
                if tf > t0:
                    lat.append(tf - t0)
            except (ValueError, KeyError, TypeError):
                pass
            try:
                retries.append(int(row.get("P5_llm_retries") or 0))
            except ValueError:
                pass

    if not lat:
        return
    p50, p95, mx = _pct(lat, 0.50), _pct(lat, 0.95), max(lat)
    out.append("\n## ⏱️ Latence pipeline (bout-en-bout, T0→T_fin)\n")
    out.append(
        f"n={len(lat)} · p50 **{p50:.0f}s** · p95 **{p95:.0f}s** · max **{mx:.0f}s**"
    )
    if retries:
        r_total = sum(1 for r in retries if r > 0)
        out.append(f" · retries LLM sur {r_total} tâches")
    out.append("\n")
    if p95 >= TH_PIPELINE_P95_S:
        alarms.append(
            f"🔴 Backlog pipeline : latence p95 = {p95:.0f}s "
            f"(> {TH_PIPELINE_P95_S:.0f}s) — la file de planif s'accumule."
        )


def section_agents(run: Path, out: list[str], alarms: list[str]) -> None:
    path = run / "gama_results" / "agent_states.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return

    def _i(r, k):
        try:
            return int(r[k])
        except (ValueError, KeyError, TypeError):
            return 0

    first, last = rows[0], rows[-1]
    peak_inactive = max(_i(r, "inactive") for r in rows)
    total = _i(last, "total") or 1
    final_inactive = _i(last, "inactive")
    out.append("\n## 👥 Activité des agents\n")
    out.append(
        f"{len(rows)} steps ({first.get('sim_time', '?')} → {last.get('sim_time', '?')}) · "
        f"total {total} · inactifs : début {_i(first, 'inactive')}, "
        f"pic {peak_inactive}, fin **{final_inactive}** "
        f"({final_inactive / total:.0%})\n"
    )
    if final_inactive / total >= TH_INACTIVE_FINAL_SHARE:
        alarms.append(
            f"🟠 {final_inactive}/{total} agents ({final_inactive / total:.0%}) "
            f"encore inactifs en fin de run — vérifier planifs non abouties."
        )


# Maximum absolute gap tolerated (% points) between the expected share of a mode and its
# share actually drawn, before considering that the draw is drifting.
TH_MODE_DRIFT_PTS = 8.0
# Below this count, the gap is only sampling noise.
TH_MODE_DRIFT_MIN_ROWS = 200


def _decisions_expected_vs_drawn(expected: Counter, expected_rows: int,
                                 drawn: Counter, out: list[str], alarms: list[str]) -> None:
    """Compares the distribution announced by the LLM with the one actually drawn.

    The draw must reproduce the distribution in expectation. A clear and lasting gap
    therefore does not come from the model: it comes from the draw itself, from the cache (options
    gone, inherited points served again without a draw) or from a normalisation bias.

    **A single denominator on both sides**: the rows that carry a distribution.
    Counting the drawn modes over ALL rows of the log — single choice, fallback,
    "Aucun" — mixed in decisions without a draw and produced a gap of more than
    10 points on the most frequent mode, with the unfounded accusation that goes with it
    ("check the cache"). Measured on run 2026-09-04_16_25: -11.9 pt announced
    for public transport, +0.3 pt with a common denominator.
    """
    if not expected_rows:
        return
    out.append(
        f"\n**Répartition attendue vs tirée** ({expected_rows} décisions probabilistes, "
        f"même dénominateur des deux côtés)\n")
    out.append("| Mode | attendu | tiré | écart |")
    out.append("|:--|--:|--:|--:|")
    drawn_total = sum(drawn.values()) or 1
    worst = (0.0, "")
    for mode, mass in expected.most_common():
        exp_pct = 100.0 * mass / expected_rows
        got_pct = 100.0 * drawn.get(mode, 0) / drawn_total
        delta = got_pct - exp_pct
        if exp_pct == 0 and got_pct == 0:
            continue
        out.append(f"| {mode} | {exp_pct:.1f} % | {got_pct:.1f} % | {delta:+.1f} pts |")
        if abs(delta) > worst[0]:
            worst = (abs(delta), mode)
    if expected_rows >= TH_MODE_DRIFT_MIN_ROWS and worst[0] > TH_MODE_DRIFT_PTS:
        alarms.append(
            f"🟠 Tirage modal dérivant : {worst[1]} s'écarte de {worst[0]:.1f} pts de la "
            f"répartition annoncée par le LLM sur {expected_rows} décisions — vérifier le "
            f"cache (points hérités resservis sans tirage) et la normalisation."
        )


def section_decisions(run: Path, out: list[str], alarms: list[str]) -> None:
    path = run / "moves.csv"
    if not path.exists():
        return
    modes = Counter()
    methods = Counter()
    # Sum of the probabilities announced by the LLM, per mode: the distribution it
    # "wanted". The drawn modes must reproduce it in expectation — a persistent
    # gap signals a bias of the draw or of the cache, not of the model.
    expected = Counter()
    # Modes drawn on the ONLY rows that carry a distribution: this is the
    # denominator of `expected`, and the only one against which the gap means something.
    drawn = Counter()
    expected_rows = 0
    total = 0
    fallbacks = 0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prob_cols = [c for c in (reader.fieldnames or [])
                     if c.startswith("P(") and c.endswith(") %")]
        for row in reader:
            total += 1
            modes[row.get("Mode de transport Choisi", "?")] += 1
            method = row.get("Méthode de sélection", "?")
            methods[method] += 1
            if "Error" in method or "Default" in method:
                fallbacks += 1
            # Empty cell = decision without a distribution (single choice, error, inherited cache);
            # 0 = mode explicitly ruled out. Only filled rows count.
            cells = {c: row.get(c, "") for c in prob_cols}
            if any(v not in ("", None) for v in cells.values()):
                expected_rows += 1
                drawn[row.get("Mode de transport Choisi", "?")] += 1
                for col, val in cells.items():
                    try:
                        expected[col[2:-3]] += float(val) / 100.0
                    except (TypeError, ValueError):
                        pass
    if not total:
        return
    # "Final LLM error" ratio: fallbacks relative to the decisions alone
    # that went through the LLM (actual choice + fallback), not single-choice/no-move.
    llm_decisions = methods.get("LLM", 0) + fallbacks
    llm_fallback_share = fallbacks / llm_decisions if llm_decisions else 0.0
    out.append("\n## 🧭 Décisions de mobilité\n")
    out.append(
        f"{total} trajets · fallback LLM : **{fallbacks}** ({fallbacks / total:.1%} des trajets, "
        f"{llm_fallback_share:.1%} des {llm_decisions} décisions LLM)\n"
    )
    out.append("\n**Modes** · **Méthodes de sélection**\n")
    out.append("| Mode | n | | Méthode | n |")
    out.append("|:--|--:|--|:--|--:|")
    ml = modes.most_common()
    sl = methods.most_common()
    for i in range(max(len(ml), len(sl))):
        left = f"{ml[i][0]} | {ml[i][1]}" if i < len(ml) else " | "
        right = f"{sl[i][0][:32]} | {sl[i][1]}" if i < len(sl) else " | "
        out.append(f"| {left} | | {right} |")

    _decisions_expected_vs_drawn(expected, expected_rows, drawn, out, alarms)
    if fallbacks / total >= TH_FALLBACK_SHARE:
        alarms.append(
            f"🟠 {fallbacks}/{total} décisions ({fallbacks / total:.1%}) en fallback "
            f"(LLM KO → index par défaut) — qualité de planif dégradée."
        )
    if llm_decisions and llm_fallback_share >= TH_FALLBACK_SHARE:
        alarms.append(
            f"🟠 {fallbacks}/{llm_decisions} décisions LLM ({llm_fallback_share:.1%}) retombées "
            f"sur l'index par défaut (erreur définitive LLM) — vérifier providers et cache."
        )


def _is_llm_fallback(method: str) -> bool:
    """Decision fallen back to the default index for lack of an LLM response."""
    return "Error" in method or "Default" in method


def _instant_evenement(run: Path) -> tuple[int, str] | None:
    """Simulated time of the FIRST event injection, and its identifier.

    Ticket 105. `evenements.jsonl` is the new name (ticket 100); `chocs.jsonl` is a symbolic
    link to it on migrated runs, and the real file on older ones.
    """
    for nom in ("evenements.jsonl", "chocs.jsonl"):
        chemin = run / nom
        if not chemin.exists():
            continue
        instants = []
        for enr in _iter_jsonl(chemin):
            ts = enr.get("timestamp")
            if isinstance(ts, (int, float)):
                instants.append((int(ts), str(enr.get("evenement_id") or enr.get("choc_id") or "?")))
        if instants:
            return min(instants, key=lambda t: t[0])
    return None


def section_replis_autour_evenement(run: Path, out: list[str], alarms: list[str]) -> None:
    """Fallbacks BEFORE and AFTER the event — ticket 105.

    A global rate says nothing: on 2026-09-23, the baseline was perfectly clean
    (0/48) while the measurement window carried 10.3 % (4/39). That is exactly the gap
    that makes an arm unusable, and it is the one an aggregated figure hides. It had taken
    digging through `moves.csv` by hand to see it.
    """
    chemin = run / "moves.csv"
    evt = _instant_evenement(run)
    if not chemin.exists() or evt is None:
        return
    instant, evenement_id = evt

    avant_llm = avant_repli = apres_llm = apres_repli = 0
    with chemin.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts = float(row.get("Temps simulé") or 0)
            except (TypeError, ValueError):
                continue
            methode = row.get("Méthode de sélection", "?")
            if methode != "LLM" and not _is_llm_fallback(methode):
                continue  # single choice, no-move: not a model decision
            repli = _is_llm_fallback(methode)
            if ts < instant:
                avant_repli += repli
                avant_llm += 1
            else:
                apres_repli += repli
                apres_llm += 1

    if not (avant_llm or apres_llm):
        return

    def part(n: int, d: int) -> str:
        return f"{n}/{d} = {n / d:.1%}" if d else "aucune décision"

    out.append("\n## 🎯 Replis autour de l'événement\n")
    out.append(
        f"Événement `{evenement_id}` injecté à {datetime.fromtimestamp(instant, timezone.utc):%Y-%m-%d %H:%M} "
        f"(temps simulé).\n"
    )
    out.append("| Fenêtre | Décisions du modèle | Replis |")
    out.append("|:--|--:|:--|")
    out.append(f"| Avant (ligne de base) | {avant_llm} | {part(avant_repli, avant_llm)} |")
    out.append(f"| Après (mesure) | {apres_llm} | {part(apres_repli, apres_llm)} |")

    taux_apres = apres_repli / apres_llm if apres_llm else 0.0
    taux_avant = avant_repli / avant_llm if avant_llm else 0.0
    if taux_apres >= TH_FALLBACK_SHARE:
        alarms.append(
            f"🔴 {apres_repli}/{apres_llm} décisions ({taux_apres:.1%}) de la FENÊTRE DE MESURE "
            f"servies par l'index par défaut, contre {taux_avant:.1%} avant l'événement — "
            f"ce ne sont pas des décisions du modèle, la comparaison est entamée."
        )


def section_temoin_souvenir(run: Path, out: list[str], alarms: list[str]) -> None:
    """Did the injected memory reach long-term memory? — ticket 106.

    On 2026-09-23, the v4 run injected a metro outage rated 0.75 and lost it: the
    evening consolidation wrote "Today went very smoothly overall", and none of the 123
    long-term memory documents mentioned the metro. The following days therefore measured
    the effect of no memory at all, and nothing said so.

    The section gives BOTH verdicts. A witness that shows only its failures does not let one
    tell "it never fires" from "it no longer runs".
    """
    chemin = run / "temoin_souvenir.jsonl"
    if not chemin.exists():
        return
    lignes = []
    for brute in chemin.read_text(encoding="utf-8").splitlines():
        if not brute.strip():
            continue
        try:
            lignes.append(json.loads(brute))
        except json.JSONDecodeError:
            continue
    if not lignes:
        return

    perdus = [l for l in lignes if not l.get("retrouve")]
    out.append("\n## 🧠 Témoin du souvenir injecté\n")
    out.append("| Agent | Jour simulé | Verdict | Mots retrouvés |")
    out.append("|:--|:--|:--|:--|")
    for l in lignes:
        mots = ", ".join(l.get("mots_retrouves") or []) or "—"
        verdict = "✅ retrouvé" if l.get("retrouve") else "🔴 **PERDU**"
        out.append(
            f"| `{l.get('person_id', '?')}` | {l.get('sim_day', '?')} | {verdict} | {mots} |"
        )
    out.append(
        f"\n{len(lignes)} consolidation(s) contrôlée(s) après injection, "
        f"{len(perdus)} sans trace du souvenir.\n"
    )

    for l in perdus:
        cherches = ", ".join((l.get("mots_cherches") or [])[:8])
        alarms.append(
            f"🔴 SOUVENIR INJECTÉ PERDU — l'agent `{l.get('person_id', '?')}` a consolidé le "
            f"{l.get('sim_day', '?')} sans garder trace de l'événement "
            f"(cherchés : {cherches}). Les jours suivants ne mesurent l'effet d'AUCUN "
            f"souvenir : ce bras est inexploitable tel quel."
        )


def section_rapprochement_injections(run: Path, out: list[str], alarms: list[str]) -> None:
    """Reconciliation of declared vs produced injections — ticket 108.

    A run can declare injection days in evenement.yaml and produce none of them,
    or fewer than declared, without any alarm or report flagging it. This section
    gives the reconciliation in both directions (successes and missing ones) and identifies the
    plausible cause of each gap.
    """
    import importlib.util

    chemin = Path(__file__).resolve().parents[1] / "analysis" / "rapprochement_injections.py"
    spec = importlib.util.spec_from_file_location("rapprochement_injections", chemin)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    out.append("\n## ⚡ Rapprochement des injections (ticket 108)\n")
    try:
        r = module.rapprocher(run)
    except Exception as err:  # noqa: BLE001
        out.append(f"_Rapprochement impossible : {type(err).__name__}: {err}_\n")
        alarms.append(f"🔴 Rapprochement des injections impossible ({err}).")
        return

    if r.aucun_evenement or (not r.declarees and not r.produites):
        out.append("_Aucun événement déclaré dans ce run : rien à rapprocher._\n")
        return

    out.extend(module.rendre(r))
    for a in r.alarmes:
        alarms.append(a)


def section_lecture_avant_decision(run: Path, out: list[str], alarms: list[str]) -> None:
    """Was the article read in front of the model when the agent decided? — ticket 111.

    On 2026-09-25, arm a09 `2026-09-24_17_50` showed that it was not: the decision of the
    reading day was pre-computed the day before, and the following ones did not see an article
    rated below the shock threshold. The section gives the successes AND the defects, and also
    says when there is nothing to check: a missing section does not tell "nothing to see" from
    "the check no longer runs".
    """
    import importlib.util

    chemin = Path(__file__).resolve().parents[1] / "analysis" / "lecture_avant_decision.py"
    spec = importlib.util.spec_from_file_location("lecture_avant_decision", chemin)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution: the module's dataclasses resolve their annotations
    # in `sys.modules`, and an anonymous module makes them fail.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    out.append("\n## 📰 Lecture avant décision (ticket 111)\n")
    try:
        constats = module.controler(run)
    except Exception as err:  # noqa: BLE001 — a section does not bring down the report
        out.append(f"_Contrôle impossible : {type(err).__name__}: {err}_\n")
        alarms.append(f"🔴 Contrôle « lecture avant décision » impossible ({err}).")
        return
    if not constats:
        out.append(
            "_Aucun événement `lu` dans ce run (ou aucune lecture tracée) : rien à contrôler._\n"
        )
        return
    out.extend(module.rendre(constats))
    rouges = [c for c in constats if c.verdict == "🔴"]
    verts = [c for c in constats if c.verdict == "✅"]
    out.append(
        f"\n{len(constats)} agent(s) contrôlé(s) : {len(verts)} ✅, {len(rouges)} 🔴, "
        f"{len(constats) - len(verts) - len(rouges)} sans décision sur ses jours de service.\n"
    )
    for c in rouges:
        if c.attend_ligne:
            quoi = (
                f"la première décision du {c.date_lecture:%d/%m} ne porte pas la ligne"
                if c.premiere is False
                else f"{c.privees} décision(s) de ses jours de service sans la ligne"
            )
            alarms.append(
                f"🔴 LECTURE ABSENTE DE LA DÉCISION — `{c.person_id}` ({c.role}) : {quoi}. "
                f"L'effet mesuré sur ces jours n'est pas celui d'un agent qui SAIT."
            )
        else:
            alarms.append(
                f"🔴 LIGNE DU FOYER CHEZ UN NON-INFORMÉ — `{c.person_id}` : {c.privees} "
                f"décision(s) portent `[ FOYER ]` alors que le lecteur ne lui a rien dit."
            )


def section_activity_coverage(run: Path, out: list[str], alarms: list[str]) -> None:
    """Activity coverage per simulated day.

    Activities are recurring and undated: we check that EACH activity
    of an agent runs EACH day of its activity range. Two ways to "miss"
    an activity:
      - degraded: decided without an LLM response (fallback → default index);
      - missed  : no execution that day (agent stuck / planning not completed).

    The activity universe of an agent = set of activity IDs seen for it over
    the whole run; "missed" = activity of the universe absent on a day of its range.
    """
    path = run / "moves.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not {"ID Personne", "ID Activité", "Temps simulé"} <= set(cols):
            out.append("\n## 🎯 Couverture des activités\n")
            out.append(
                "_Indisponible : nécessite les colonnes `ID Personne` / `ID Activité` / "
                "`Temps simulé` du move-log (runs récents uniquement)._\n"
            )
            return
        # (person, day) → set of executed activities; (person,day) → fallbacks
        executed: dict[tuple[str, int], set[str]] = {}
        fallback_exec: dict[tuple[str, int], set[str]] = {}
        universe: dict[str, set[str]] = {}
        person_days: dict[str, set[int]] = {}
        per_day_total: Counter = Counter()
        per_day_fallback: Counter = Counter()
        for row in reader:
            person = (row.get("ID Personne") or "").strip()
            activity = (row.get("ID Activité") or "").strip()
            if not person or not activity:
                continue
            try:
                day = int(float(row["Temps simulé"])) // SIM_DAY_SECONDS
            except (ValueError, KeyError, TypeError):
                continue
            method = row.get("Méthode de sélection", "?")
            executed.setdefault((person, day), set()).add(activity)
            universe.setdefault(person, set()).add(activity)
            person_days.setdefault(person, set()).add(day)
            per_day_total[day] += 1
            if _is_llm_fallback(method):
                per_day_fallback[day] += 1
                fallback_exec.setdefault((person, day), set()).add(activity)

    if not per_day_total:
        return

    # Missed activities: for each agent, over its day range [min, max],
    # activities of the universe absent on a given day.
    per_day_expected: Counter = Counter()
    per_day_missed: Counter = Counter()
    for person, acts in universe.items():
        days = person_days[person]
        span = range(min(days), max(days) + 1)
        for day in span:
            per_day_expected[day] += len(acts)
            done = executed.get((person, day), set())
            per_day_missed[day] += len(acts - done)

    tot_expected = sum(per_day_expected.values())
    tot_missed = sum(per_day_missed.values())
    tot_decisions = sum(per_day_total.values())
    tot_fallback = sum(per_day_fallback.values())

    out.append("\n## 🎯 Couverture des activités (par jour simulé)\n")
    out.append(
        f"{tot_decisions} exécutions · dégradées (sans LLM) **{tot_fallback}** "
        f"({tot_fallback / tot_decisions:.1%}) · attendues {tot_expected} · "
        f"manquées **{tot_missed}** ({tot_missed / tot_expected:.1%} des attendues)"
        if tot_expected else f"{tot_decisions} exécutions · dégradées {tot_fallback}"
    )
    out.append("\n| Jour | Exécutées | Sans LLM (fallback) | Attendues | Manquées |")
    out.append("|--:|--:|--:|--:|--:|")
    for day in sorted(per_day_total):
        n = per_day_total[day]
        fb = per_day_fallback[day]
        exp = per_day_expected.get(day, 0)
        miss = per_day_missed.get(day, 0)
        fb_s = f"{fb} ({fb / n:.0%})" if n else str(fb)
        miss_s = f"{miss} ({miss / exp:.0%})" if exp else str(miss)
        out.append(f"| {day} | {n} | {fb_s} | {exp} | {miss_s} |")
    out.append("")

    if tot_decisions and tot_fallback / tot_decisions >= TH_FALLBACK_SHARE:
        alarms.append(
            f"🟠 {tot_fallback}/{tot_decisions} activités ({tot_fallback / tot_decisions:.1%}) "
            f"décidées sans réponse LLM (fallback index par défaut)."
        )
    if tot_expected and tot_missed / tot_expected >= TH_MISSED_ACTIVITY_SHARE:
        alarms.append(
            f"🟠 {tot_missed}/{tot_expected} activités-jours ({tot_missed / tot_expected:.1%}) "
            f"non exécutées (activité récurrente sautée un jour donné)."
        )


def section_arrivals(run: Path, out: list[str], alarms: list[str]) -> None:
    path = run / "gama_results" / "gama_arrivals.csv"
    if not path.exists():
        return
    total = 0
    timed_out = 0
    dep_delays = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if str(row.get("timed_out", "")).strip().lower() == "true":
                timed_out += 1
            try:
                dep_delays.append(int(row["departure_delay_s"]))
            except (ValueError, KeyError, TypeError):
                pass
    if not total:
        return
    out.append("\n## 🚦 Arrivées\n")
    line = f"{total} arrivées · **{timed_out} timed_out** ({timed_out / total:.1%})"
    if dep_delays:
        line += (
            f" · retard départ p95 {_pct([float(d) for d in dep_delays], 0.95):.0f}s "
            f"max {max(dep_delays)}s"
        )
    out.append(line + "\n")
    if timed_out:
        alarms.append(
            f"🟠 {timed_out} trajets timed_out — planif non livrée avant l'échéance simulée."
        )


def section_quotas(run: Path, out: list[str], alarms: list[str]) -> None:
    """Integrates the RPM/TPM quota validation from quota_validator."""
    import subprocess

    quota_script = Path(__file__).resolve().parent / "quota_validator.py"
    if not quota_script.exists():
        return

    try:
        result = subprocess.run(
            ["python3", str(quota_script), str(run)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return

        # Parse the outputs to extract the table and summary
        lines = result.stdout.split("\n")

        # Look for the summary table
        table_start = -1
        for i, line in enumerate(lines):
            if "TABLEAU RÉCAPITULATIF" in line:
                table_start = i + 3  # Skip the header lines
                break

        if table_start < 0:
            return

        out.append("\n## 💾 Validation des Quotas RPM/TPM\n")
        out.append("```")

        # Add the table
        for i in range(table_start, len(lines)):
            line = lines[i]
            if "ANALYSE DU MICRO-BATCHING" in line or "════" in line:
                break
            if line.strip():
                out.append(line)

        out.append("```\n")

        # Add the violation summary
        for i, line in enumerate(lines):
            if "🚨 ALERTE" in line or "✅ SUCCÈS" in line:
                if "ALERTE" in line:
                    out.append("⚠️ **Violations détectées** — voir le tableau ci-dessus\n")
                    alarms.append("Violations de quotas RPM/TPM — vérifier provider_config")
                else:
                    out.append("✅ **Tous les quotas respectés** — aucune violation\n")
                break

    except Exception:
        pass  # Silently ignore quota_validator errors


def section_jeu(run: Path, out: list[str], alarms: list[str]) -> None:
    """Recorded trip set (ticket 035, spec 04, G14): engine calls, sources, recomputations
    with no effect, triggers never fired, coverage. Nothing without `jeu_stats.json`."""
    path = run / "jeu_stats.json"
    if not path.exists():
        return
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        out.append("\n## 📼 Jeu enregistré\n\n`jeu_stats.json` illisible.\n")
        return
    out.append("\n## 📼 Jeu enregistré\n")
    out.append(f"Jeu **{st.get('jeu')}** (empreinte `{str(st.get('empreinte'))[:12]}…`) · population {st.get('population')}\n")
    out.append("| Rubrique | Valeur |")
    out.append("|:--|--:|")
    out.append(f"| Appels au trip helper (moteurs) | {st.get('appels_moteur', 0)} |")
    out.append(f"| — dont OTP / OSMnx (quand mesuré) | {st.get('appels_otp', 0)} / {st.get('appels_osmnx', 0)} |")
    sources = st.get("propositions_par_source") or {}
    for k in sorted(sources):
        out.append(f"| Propositions `{k}` | {sources[k]} |")
    out.append(f"| Déplacements servis du jeu | {st.get('deplacements_servis', 0)} |")
    out.append(f"| Déplacements recalculés (horaire) | {st.get('recalculs_horaire', 0)} |")
    out.append(f"| Recalculs sans effet | {st.get('recalcul_sans_effet', 0)} |")
    out.append(f"| Recalculs illégitimes | {st.get('recalcul_illegitime', 0)} |")
    out.append(f"| Déplacements hors jeu (calcul en vol) | {st.get('hors_jeu', 0)} |")
    jamais = st.get("declencheurs_jamais_declenches") or []
    out.append(f"| Déclencheurs jamais déclenchés | {', '.join(jamais) if jamais else 'aucun'} |")
    couv = st.get("couverture_jeu") or {}
    if couv:
        out.append(f"| Couverture du jeu | {couv.get('deplacements_couverts')} / {couv.get('deplacements_attendus')} |")
    if st.get("recalcul_illegitime"):
        alarms.append(f"🔴 {st['recalcul_illegitime']} appel(s) moteur hors des conditions admises (spec 04, G4).")
    rec = st.get("recalculs_horaire", 0)
    if rec and st.get("recalcul_sans_effet", 0) / rec > 0.3:
        alarms.append(f"🟠 {st['recalcul_sans_effet']}/{rec} recalculs horaires sans effet — tolérance trop sensible (G7).")
    if jamais:
        alarms.append(f"🟡 déclencheur(s) de recalcul jamais déclenché(s) : {', '.join(jamais)} — fonction fantôme ? (EF-44)")


def section_alarms(out: list[str], alarms: list[str]) -> None:
    banner = ["\n## 🚨 ALARMES\n"]
    if not alarms:
        banner.append("_Aucune anomalie franchissant les seuils._\n")
    else:
        for a in alarms:
            banner.append(f"- {a}")
        banner.append("")
    # Insert the alarms right after the title (position 1)
    out[1:1] = banner


# ── Entry point ─────────────────────────────────────────────────────────────

def resolve_run(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    # Repo root = two levels above this file (scripts/debug/)
    repo = Path(__file__).resolve().parents[2]
    return (repo / "experiments" / "current").resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="Run directory (default experiments/current)")
    ap.add_argument("--top", type=int, default=TOP_DEFAULT, help="Number of top-N rows")
    ap.add_argument("--out", help="Write the report to this file instead of stdout")
    args = ap.parse_args()

    run = resolve_run(args.run_dir)
    if not run.exists():
        print(f"❌ Run directory not found: {run}", file=sys.stderr)
        return 1

    out: list[str] = []
    alarms: list[str] = []

    section_meta(run, out, alarms)
    section_logs(run, out, alarms, args.top)
    section_llm(run, out, alarms, args.top)
    section_pipeline(run, out, alarms)
    section_agents(run, out, alarms)
    section_decisions(run, out, alarms)
    section_replis_autour_evenement(run, out, alarms)
    section_rapprochement_injections(run, out, alarms)
    section_temoin_souvenir(run, out, alarms)
    section_lecture_avant_decision(run, out, alarms)
    section_activity_coverage(run, out, alarms)
    section_arrivals(run, out, alarms)
    section_quotas(run, out, alarms)
    section_jeu(run, out, alarms)
    section_alarms(out, alarms)  # must stay last (inserts at the top)

    report = "\n".join(out).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"✅ Report written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
