"""Compares a run with its predecessors and breaks it down by language model.

    python -m scripts.synthesis.model_compare --run experiments/archive/<run> \
        [--baseline experiments/archive/<run>] [--out docs/synthesis/models/<run>]

The main synthesis page (`make synthesis`) answers "where does the
simulation stand against the survey". It does not answer "which model produced this
score", because it aggregates all the decisions of a run without looking at the
"Fournisseur & Modèle" column. As soon as a run uses several models
— which is the case when the gateway spreads the load across providers — the
run mean mixes lineages of decisions that have nothing in common.

This module reimplements NEITHER the loss NOR the log reading: it splits
moves.csv into subsets, writes each one to a temporary CSV, and passes it to the
official reader (`frames.read_moves`) then to the official scorer
(`frames.Scorer`). A score shown here is therefore exactly that of the scoring
formula, measured on the same scope as the main page.

Two precautions specific to the split:

* **Hot resume.** `make run OFFLINE=1 CONT=1` replays the simulated day from
  t0 in the SAME experiment directory: moves.csv then holds the same
  (person, activity) pairs twice, once per attempt. The official reader
  only cuts on the simulated day, not on the attempt: it therefore counts
  the decisions made before the resume twice. This module keeps the
  most recent attempt (`Heure de calcul`) and publishes the gap between the two
  readings, instead of leaving it in the score without saying so.
* **Sample size.** A per-model subset is smaller than the whole run, and
  per-stratum divergences are biased upwards at small sample sizes. A
  permutation test therefore quantifies the gap between two models against the
  splitting noise: without it, any composite score gap would be interpretable, including
  the one a random draw would produce.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import random
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from . import frames
from .frames import DIMENSIONS, MODE_COLORS, MODE_LABELS, MODES
from .sources import REPO_ROOT, import_formule_score, load_manifest

# The permutation test replays the scoring: each draw costs two passes of
# the loss. 60 draws are enough to separate "real gap" from "splitting
# noise" — the question asked is a sign, not a p-value to the thousandth.
PERMUTATIONS = 60

# Per-model subset size below which no score is shown: the survey
# dimensions (15 age bands, 7 occupations) no longer have any
# support, and the composite score would mostly measure the sample size.
MIN_ROWS = 60

PROVIDER_MODELS = {}  # filled from providers.yaml: gateway key → model


# ── Splitting the log ────────────────────────────────────────────────────────

def load_provider_models(path: Path) -> dict[str, str]:
    """``gateway key → model name``, read from providers.yaml.

    A line-by-line scan rather than a YAML load: the file holds
    dozens of commented-out blocks (disabled providers), and we only want
    the active keys with their `default_model`.
    """
    out: dict[str, str] = {}
    current: Optional[str] = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        key = re.match(r"^  ([A-Za-z0-9_]+):\s*$", line)
        if key:
            current = key.group(1)
            continue
        model = re.match(r"^\s+default_model:\s*(\S+)", line)
        if model and current:
            out[current] = model.group(1)
            current = None
    return out


def attempt_key(row: dict) -> str:
    """Computation timestamp, which identifies the attempt (before/after resume)."""
    return (row.get("Heure de calcul") or "").strip()


def read_raw(path: Path) -> tuple[list[dict], list[str]]:
    with Path(path).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def latest_attempts(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keeps only the most recent attempt of each decision.

    The key carries the **simulated day** in addition to the (person, activity) pair, and
    this is not a detail: without it, the day-1 decision and its day-2
    repetition — which the rolling planning horizon produces for 440 pairs
    on this run — would be seen as two attempts at the same decision. We
    would then keep the day-2 one, which the cut at the first simulated day then
    discards: the decision would vanish from the score instead of entering it once.

    Also returns the summary of what was discarded: it is what tells whether the run
    was hot-resumed, and by how many rows the raw reading is wrong.

    A row without a person or activity identifier is kept as is,
    as in ``frames.latest_attempts``: it cannot be paired with any
    other, and grouping them under a shared empty key would collapse onto ONE SINGLE
    row a whole log that lacked these columns. The two
    implementations must stay in agreement — it is the same rule on the same
    trap, and a gap between them would make the main page diverge from the
    per-model page without anything flagging it.
    """
    best: dict[tuple[str, str, Optional[str]], dict] = {}
    unpaired: set[int] = set()
    for row in rows:
        person = (row.get("ID Personne") or "").strip()
        activity = (row.get("ID Activité") or "").strip()
        if not person or not activity:
            unpaired.add(id(row))
            continue
        key = (person, activity, frames.simulated_day(row.get("Temps simulé") or ""))
        keep = best.get(key)
        if keep is None or attempt_key(row) > attempt_key(keep):
            best[key] = row
    kept_ids = {id(r) for r in best.values()} | unpaired
    kept = [r for r in rows if id(r) in kept_ids]
    dropped = [r for r in rows if id(r) not in kept_ids]
    stamps = sorted({attempt_key(r)[:10] for r in rows if attempt_key(r)})
    bilan = {
        "n_total": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "resumed": len(stamps) > 1,
        "attempt_days": stamps,
        "dropped_methods": dict(collections.Counter(
            r.get("Méthode de sélection") or "?" for r in dropped)),
        "kept_by_attempt_day": dict(collections.Counter(
            attempt_key(r)[:10] for r in kept)),
    }
    return kept, bilan


class Measurer:
    """Scores a subset of moves.csv with the official reader and loss."""

    def __init__(self, cerema: dict, scorer, exclude: list[str], tmpdir: Path):
        self.cerema = cerema
        self.scorer = scorer
        self.exclude = exclude
        self.tmpdir = tmpdir
        self._seq = 0

    def measure(self, rows: list[dict], fields: list[str], label: str,
                note: str = "") -> dict:
        self._seq += 1
        path = self.tmpdir / f"subset_{self._seq:04d}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        kept, stats = frames.read_moves(path, self.exclude)
        out: dict[str, Any] = {
            "label": label, "note": note,
            "n_input": len(rows), "n_trips": len(kept),
            "n_persons": len({r["agent_id"] for r in kept}),
            "stats": {k: v for k, v in stats.items()
                      if not str(k).startswith("contrainte::")},
            "methods": dict(collections.Counter(
                r.get("Méthode de sélection") or "?" for r in rows)),
            "scored": len(kept) >= MIN_ROWS,
        }
        if not out["scored"]:
            return out
        variants = frames.simulation_frames(kept)
        out["variants"] = {}
        for name, frame in variants.items():
            out["variants"][name] = {
                "global": frames.global_view(frame, self.cerema),
                "scores": self.scorer.score(frame, self.cerema),
                "n_rows": len(frame),
            }
        out["details"] = {}
        for dim in DIMENSIONS:
            detail = frames.dimension_detail(variants["attendu"], self.cerema, dim)
            if any(d["n"] for d in detail):
                out["details"][dim["key"]] = detail
        out["worst_strata"] = frames.worst_strata(
            {k: v for k, v in out["details"].items()
             if any(d["key"] == k and d["scored"] for d in DIMENSIONS)})
        return out

    def composite(self, rows: list[dict], fields: list[str]) -> Optional[float]:
        """Composite score only — for the permutation test draws."""
        res = self.measure(rows, fields, "permutation")
        if not res.get("scored"):
            return None
        return res["variants"]["attendu"]["scores"]["emd_jsd"]["composite"]


def permutation_test(measurer: Measurer, a: list[dict], b: list[dict],
                     fields: list[str], seed: int = 20260820) -> dict:
    """Does the composite score gap between two models hold against noise?

    The decisions of the two models are reshuffled, re-split at the same sizes,
    and we count how many draws produce a gap as large as the one
    observed. Without this control, a two-point gap would be read as a
    ranking of models whereas a random split sometimes produces it.
    """
    observed_a = measurer.composite(a, fields)
    observed_b = measurer.composite(b, fields)
    if observed_a is None or observed_b is None:
        return {"available": False}
    observed = observed_a - observed_b
    pool = list(a) + list(b)
    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        left = measurer.composite(pool[:len(a)], fields)
        right = measurer.composite(pool[len(a):], fields)
        if left is None or right is None:
            continue
        gaps.append(left - right)
    if not gaps:
        return {"available": False}
    gaps.sort()
    extreme = sum(1 for g in gaps if abs(g) >= abs(observed))
    return {
        "available": True, "observed": observed, "n_draws": len(gaps),
        "median": gaps[len(gaps) // 2], "min": gaps[0], "max": gaps[-1],
        "n_extreme": extreme, "p_value": extreme / len(gaps),
    }


def comparability(a: list[dict], b: list[dict]) -> list[dict]:
    """Did the two models receive comparable decisions?

    A model may look better because it inherited the easy trips.
    Load balancing across providers is not supposed to look at the
    persona, but this is checked rather than assumed.
    """
    rows: list[dict] = []

    def mean(rs: list[dict], col: str) -> Optional[float]:
        vals = []
        for r in rs:
            try:
                vals.append(float(r.get(col) or ""))
            except ValueError:
                continue
        return sum(vals) / len(vals) if vals else None

    for col, label in (("Âge", "Âge moyen"),
                       ("Distance parcourue", "Distance moyenne (km)")):
        rows.append({"trait": label, "kind": "num",
                     "a": mean(a, col), "b": mean(b, col)})
    for col, label in (("Genre", "Genre"), ("Occupation principale", "Occupation"),
                       ("Motifs de déplacement", "Motif")):
        def top(rs: list[dict]) -> str:
            counts = collections.Counter((r.get(col) or "?") for r in rs)
            total = max(1, sum(counts.values()))
            return " · ".join(f"{k[:16]} {100 * v / total:.0f}%"
                              for k, v in counts.most_common(3))
        rows.append({"trait": label, "kind": "cat", "a": top(a), "b": top(b)})

    def offers(rs: list[dict]) -> Optional[float]:
        counts = [len(frames.parse_offered_modes(r.get("Modes proposés au LLM") or ""))
                  for r in rs]
        return sum(counts) / len(counts) if counts else None
    rows.append({"trait": "Modes proposés (moyenne)", "kind": "num",
                 "a": offers(a), "b": offers(b)})
    return rows


# ── Run health ───────────────────────────────────────────────────────────────

def run_health(run_dir: Path) -> dict:
    """What the logs say about the run, next to what the score says.

    A composite score is always read on a scope: if a third of the decisions
    never reached a model, the score covers the other two thirds and
    ignoring it would pass off a shortage of providers as a performance.
    """
    out: dict[str, Any] = {"errors": {}, "n_errors": 0, "alarms": {}, "cycle": None}
    err_path = run_dir / "llm_errors.jsonl"
    if err_path.exists():
        counts: collections.Counter = collections.Counter()
        for line in err_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            out["n_errors"] += 1
            counts[(entry.get("provider") or "?",
                    entry.get("http_status") or "?")] += 1
        out["errors"] = [{"provider": p, "status": s, "n": n}
                         for (p, s), n in counts.most_common()]
    # Last cycle summary published by the controller in the GAMA log:
    # LLM cache, throughput, backlog, active agents.
    gama = run_dir / "gama_headless.log"
    if gama.exists():
        pattern = re.compile(r"cache LLM (\d+)% \((\d+)/(\d+)\).*?débit ([\d.]+) req/min"
                             r".*?backlog (\d+) \((\d+)%\)"
                             r".*?agents (\d+) actifs / (\d+) inactifs / (\d+) total")
        last = None
        for line in gama.read_text(encoding="utf-8", errors="replace").splitlines():
            found = pattern.search(line)
            if found:
                last = found
        if last:
            out["cycle"] = {
                "cache_pct": int(last.group(1)), "cache_hits": int(last.group(2)),
                "cache_total": int(last.group(3)), "throughput": float(last.group(4)),
                "backlog": int(last.group(5)), "backlog_pct": int(last.group(6)),
                "agents_active": int(last.group(7)),
                "agents_idle": int(last.group(8)), "agents_total": int(last.group(9)),
            }
    # Alarms: only the pattern is kept, not the numbered occurrences.
    alarms: collections.Counter = collections.Counter()
    for name in ("app.log",):
        path = run_dir / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[ALARME]" in line:
                motif = line.split("[ALARME]", 1)[1].strip()
                motif = re.sub(r"\d+", "N", motif)[:150]
                alarms[motif] += 1
    out["alarms"] = alarms.most_common(8)
    return out


# ── Rendering ────────────────────────────────────────────────────────────────

CSS = """
:root{--ink:#1b1b1f;--ink2:#54545e;--ink3:#82828e;--bg:#fbfbfa;--card:#ffffff;
--line:#e3e3df;--line2:#cfcfc9;--accent:#4A6FE3;--warn:#C2571A;--warnbg:#fdf4ec;
--ok:#2E7D5B;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--ink:#ececef;--ink2:#a8a8b2;--ink3:#7c7c88;
--bg:#17171a;--card:#1e1e22;--line:#2e2e34;--line2:#3d3d45;--accent:#8AA4F2;
--warn:#E08B4A;--warnbg:#2a2018;--ok:#5FB894}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:400 15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{display:grid;grid-template-columns:220px minmax(0,1fr);gap:0;max-width:1240px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;padding:28px 18px;
border-right:1px solid var(--line)}
nav h1{font-size:14px;font-weight:500;margin:0 0 4px;letter-spacing:-.01em}
nav .sub{font-size:12px;color:var(--ink3);margin-bottom:20px}
nav a{display:block;padding:5px 0;font-size:13px;color:var(--ink2);text-decoration:none;
border-left:2px solid transparent;padding-left:10px;margin-left:-10px}
nav a:hover{color:var(--ink);border-left-color:var(--line2)}
nav .grp{font-size:11px;color:var(--ink3);margin:16px 0 4px;font-weight:500}
main{padding:36px 40px 96px;min-width:0}
section{margin-bottom:52px;scroll-margin-top:20px}
h2{font-size:21px;font-weight:500;margin:0 0 6px;letter-spacing:-.015em}
h3{font-size:16px;font-weight:500;margin:26px 0 8px}
p{margin:0 0 12px;color:var(--ink2);max-width:74ch}
.lede{color:var(--ink2);font-size:15px;margin-bottom:20px;max-width:74ch}
code,.mono{font-family:var(--mono);font-size:12.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{font-size:11.5px;color:var(--ink3);margin-bottom:6px}
.tile .v{font-size:25px;font-weight:500;letter-spacing:-.02em;line-height:1.1}
.tile .u{font-size:12px;color:var(--ink3);margin-top:4px}
.tile.hi{border-color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th{text-align:left;font-weight:500;color:var(--ink3);font-size:11.5px;padding:6px 10px 6px 0;
border-bottom:1px solid var(--line2)}
td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line);color:var(--ink2)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td strong{font-weight:500;color:var(--ink)}
tr.hi td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.up{color:var(--warn)}.down{color:var(--ok)}
.missing{border:1px dashed var(--warn);background:var(--warnbg);border-radius:10px;
padding:16px 18px;margin:14px 0}
.missing .t{font-size:13.5px;font-weight:500;color:var(--warn);margin-bottom:6px}
.missing p{color:var(--ink2);font-size:13.5px;margin:0 0 8px}
.note{border:1px solid var(--line2);border-left:3px solid var(--ok);background:var(--card);
border-radius:10px;padding:14px 18px;margin:14px 0}
.note .t{font-size:13.5px;font-weight:500;color:var(--ink);margin-bottom:6px}
.note p{color:var(--ink2);font-size:13.5px;margin:0 0 8px}
.note ul,.missing ul{margin:0 0 8px 18px;padding:0;color:var(--ink2);font-size:13px}
.badge{display:inline-block;font-size:11px;padding:2px 7px;border-radius:99px;
border:1px solid var(--line2);color:var(--ink3);margin-left:6px;vertical-align:middle}
.badge.ok{color:var(--ok);border-color:var(--ok)}
.badge.warn{color:var(--warn);border-color:var(--warn)}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 2px}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;color:var(--ink2)}
.chip i{width:9px;height:9px;border-radius:2px;display:inline-block}
svg text{font-family:inherit}
.bar-label{font-size:11.5px;fill:var(--ink2)}
.bar-value{font-size:11px;fill:var(--ink);font-variant-numeric:tabular-nums}
.bar-track{fill:var(--line);opacity:.5}
.bar-target{stroke:var(--ink);stroke-width:2}
.formula{font-family:var(--mono);font-size:13.5px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:12px 0;overflow-x:auto}
.scroll{overflow-x:auto}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--ink3);font-size:12.5px}
@media(max-width:880px){.wrap{grid-template-columns:1fr}nav{position:static;height:auto;
border-right:0;border-bottom:1px solid var(--line)}main{padding:24px 20px 64px}}
"""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{int(round(value)):,}".replace(",", " ")


def delta_cell(value: Optional[float], ref: Optional[float]) -> str:
    """Gap to the baseline: an increase is a loss of fidelity, not a gain."""
    if value is None or ref is None:
        return '<td class="num">—</td>'
    d = value - ref
    if abs(d) < 0.005:
        return '<td class="num">=</td>'
    cls = "up" if d > 0 else "down"
    return f'<td class="num {cls}">{"+" if d > 0 else "−"}{fmt(abs(d))}</td>'


def mode_bars(shares: dict, target: dict, width: int = 330) -> str:
    """Simulated modal shares, with the EMC² target as a vertical marker."""
    rows = []
    top, height, gap = 6, 20, 8
    left, right = 118, 52
    span = width - left - right
    scale = max([*shares.values(), *target.values(), 1.0]) * 1.12
    for i, mode in enumerate(MODES):
        y = top + i * (height + gap)
        got = shares.get(mode, 0.0)
        want = target.get(mode, 0.0)
        w = span * got / scale
        tx = left + span * want / scale
        rows.append(
            f'<text class="bar-label" x="{left - 8}" y="{y + 14}" text-anchor="end">'
            f'{esc(MODE_LABELS[mode])}</text>'
            f'<rect class="bar-track" x="{left}" y="{y + 4}" width="{span}" height="{height - 8}" rx="2"/>'
            f'<rect x="{left}" y="{y + 4}" width="{w:.1f}" height="{height - 8}" rx="2" '
            f'fill="{MODE_COLORS[mode]}"/>'
            f'<line class="bar-target" x1="{tx:.1f}" y1="{y + 1}" x2="{tx:.1f}" y2="{y + height - 1}"/>'
            f'<text class="bar-value" x="{width - right + 6}" y="{y + 14}">'
            f'{fmt(got, 1)}&#8201;% <tspan fill="var(--ink3)">/ {fmt(want, 1)}</tspan></text>')
    height_total = top * 2 + len(MODES) * (height + gap)
    return (f'<svg viewBox="0 0 {width} {height_total}" width="100%" '
            f'style="max-width:{width}px" role="img">' + "".join(rows) + "</svg>")


DIM_KEYS = ["global", "absent_penalty", "age", "occupation", "genre", "motif", "distance"]
DIM_LABELS = {"global": "Global", "absent_penalty": "Mode absent", "age": "Âge",
              "occupation": "Occupation", "genre": "Genre",
              "motif": "Motif", "distance": "Distance"}


def score_of(entry: dict, variant: str = "attendu", loss: str = "emd_jsd") -> dict:
    if not entry.get("scored"):
        return {}
    return entry["variants"][variant]["scores"].get(loss, {})


def shares_of(entry: dict, variant: str = "attendu") -> dict:
    if not entry.get("scored"):
        return {}
    return entry["variants"][variant]["global"]["actual"]


def comparison_table(entries: list[dict], ref_key: Optional[str] = None) -> str:
    """Composite score and dimensions, one row per scope, gap to the baseline."""
    ref = None
    for e in entries:
        if ref_key and e.get("key") == ref_key:
            ref = score_of(e).get("composite")
    head = ("<thead><tr><th>Périmètre</th><th class='num'>Décisions</th>"
            "<th class='num'>Pers.</th>"
            + "".join(f"<th class='num'>{DIM_LABELS[k]}</th>" for k in DIM_KEYS)
            + "<th class='num'>Composite</th><th class='num'>Tiré</th>"
            + ("<th class='num'>Δ repère</th>" if ref is not None else "")
            + "</tr></thead>")
    body = []
    for e in entries:
        s = score_of(e)
        t = score_of(e, "tire")
        if not s:
            body.append(f"<tr><td><strong>{esc(e['label'])}</strong></td>"
                        f"<td class='num'>{fmt_int(e['n_trips'])}</td>"
                        f"<td class='num'>{fmt_int(e.get('n_persons'))}</td>"
                        f"<td colspan='{len(DIM_KEYS) + 2}' style='color:var(--ink3)'>"
                        f"effectif sous le seuil de {MIN_ROWS} décisions — non scoré</td>"
                        + ("<td class='num'>—</td>" if ref is not None else "")
                        + "</tr>")
            continue
        cells = "".join(f"<td class='num'>{fmt(s.get(k))}</td>" for k in DIM_KEYS)
        cls = " class='hi'" if e.get("highlight") else ""
        body.append(
            f"<tr{cls}><td><strong>{esc(e['label'])}</strong>"
            + (f"<br><span style='font-size:11.5px;color:var(--ink3)'>{esc(e['note'])}</span>"
               if e.get("note") else "")
            + f"</td><td class='num'>{fmt_int(e['n_trips'])}</td>"
            + f"<td class='num'>{fmt_int(e['n_persons'])}</td>{cells}"
            + f"<td class='num'><strong>{fmt(s.get('composite'))}</strong></td>"
            + f"<td class='num'>{fmt(t.get('composite'))}</td>"
            + (delta_cell(s.get("composite"), ref) if ref is not None else "")
            + "</tr>")
    return f"<div class='scroll'><table>{head}<tbody>{''.join(body)}</tbody></table></div>"


def render(payload: dict) -> str:
    d = payload
    run = d["run"]
    target = d["target_shares"]
    verdict = d["verdict"]

    legend = ('<div class="legend">'
              + "".join(f'<span class="chip"><i style="background:{MODE_COLORS[m]}"></i>'
                        f'{esc(MODE_LABELS[m])}</span>' for m in MODES)
              + '<span class="chip" style="color:var(--ink3)">trait vertical = cible EMC² 2023</span>'
              "</div>")

    # Modal share cards: the run, its baselines, then each model.
    def share_card(entry: dict) -> str:
        if not entry.get("scored"):
            return ""
        return (f'<div class="card"><div style="font-size:13.5px;font-weight:500;'
                f'margin-bottom:4px">{esc(entry["label"])}</div>'
                f'<div style="font-size:11.5px;color:var(--ink3);margin-bottom:8px">'
                f'composite {fmt(score_of(entry).get("composite"))} · '
                f'{fmt_int(entry["n_trips"])} décisions</div>'
                + mode_bars(shares_of(entry), target) + "</div>")

    model_cards = "".join(share_card(e) for e in d["models"] if e.get("scored"))
    run_cards = "".join(share_card(e) for e in d["runs"] if e.get("scored"))

    perm = d.get("permutation") or {}
    perm_html = ""
    if perm.get("available"):
        verdict_txt = ("l'écart ne s'explique pas par le découpage"
                       if perm["p_value"] < 0.05 else
                       "l'écart reste dans le bruit de découpage")
        perm_html = (
            f'<div class="note"><div class="t">Test de permutation — {esc(verdict_txt)}</div>'
            f'<p>Écart observé de composite : <strong>{fmt(perm["observed"])}</strong>. '
            f'En remélangeant les décisions des deux modèles et en recoupant aux mêmes '
            f'effectifs ({perm["n_draws"]} tirages), l\'écart médian est de '
            f'{fmt(perm["median"])} et l\'étendue va de {fmt(perm["min"])} à '
            f'{fmt(perm["max"])}. {perm["n_extreme"]} tirage(s) sur {perm["n_draws"]} '
            f'atteignent l\'écart observé, soit p ≈ {fmt(perm["p_value"], 3)}.</p></div>')

    comp_rows = "".join(
        f"<tr><td><strong>{esc(r['trait'])}</strong></td>"
        + (f"<td class='num'>{fmt(r['a'])}</td><td class='num'>{fmt(r['b'])}</td>"
           if r["kind"] == "num" else
           f"<td class='mono'>{esc(r['a'])}</td><td class='mono'>{esc(r['b'])}</td>")
        + "</tr>" for r in d.get("comparability", []))

    health = d["health"]
    err_rows = "".join(
        f"<tr><td><strong>{esc(e['provider'])}</strong></td>"
        f"<td class='num'>{esc(e['status'])}</td>"
        f"<td class='num'>{fmt_int(e['n'])}</td></tr>"
        for e in health.get("errors", []))
    cycle = health.get("cycle") or {}
    alarm_rows = "".join(
        f"<tr><td class='num'>{fmt_int(n)}</td><td>{esc(m)}</td></tr>"
        for m, n in health.get("alarms", []))

    perim_rows = "".join(
        f"<tr><td><strong>{esc(k)}</strong></td><td class='num'>{fmt_int(v)}</td>"
        f"<td class='num'>{fmt(100.0 * v / max(1, d['perimeter']['n_day_total']), 1)}&#8201;%</td>"
        f"<td>{esc(d['perimeter']['legend'].get(k, ''))}</td></tr>"
        for k, v in d["perimeter"]["methods"].items())

    nav = ('<nav><h1>Ventilation par modèle</h1>'
           f'<div class="sub">Run {esc(run["run_id"])} face à l\'enquête EMC² 2023</div>'
           '<div class="grp">Réponse</div><a href="#verdict">Est-ce mieux&nbsp;?</a>'
           '<a href="#perimetre">Périmètre mesuré</a>'
           '<div class="grp">Mesures</div><a href="#runs">Run contre ses repères</a>'
           '<a href="#modeles">Modèle par modèle</a><a href="#decoupes">Découpes internes</a>'
           '<div class="grp">Lecture</div><a href="#sante">Santé du run</a>'
           '<a href="#limites">Limites</a><a href="#provenance">Provenance</a></nav>')

    verdict_tiles = "".join(
        f'<div class="tile{" hi" if t.get("hi") else ""}"><div class="k">{esc(t["k"])}</div>'
        f'<div class="v">{esc(t["v"])}</div><div class="u">{esc(t["u"])}</div></div>'
        for t in verdict["tiles"])

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run {esc(run["run_id"])} — ventilation par modèle</title>
<style>{CSS}</style></head>
<body><div class="wrap">
{nav}
<main>
<p style="color:var(--ink3);font-size:12.5px;margin-bottom:24px">
Généré le {esc(d["generated_at"])} · {esc(d["engine_note"])} · run
<span class="mono">{esc(run["path"])}</span></p>

<section id="verdict">
<h2>Est-ce mieux&nbsp;?</h2>
<p class="lede">{verdict["lede"]}</p>
<div class="tiles">{verdict_tiles}</div>
{verdict["body"]}
</section>

<section id="perimetre">
<h2>Ce qui a été mesuré</h2>
<p class="lede">Le score porte sur les décisions du <strong>premier jour simulé</strong>
({esc(d["perimeter"]["sim_day"])}) qui portent une décision modale. Tout le reste est
hors périmètre — et c'est cette part-là qui distingue ce run des précédents.</p>
<div class="scroll"><table><thead><tr><th>Méthode de sélection</th>
<th class="num">Décisions</th><th class="num">Part</th><th>Statut</th></tr></thead>
<tbody>{perim_rows}</tbody></table></div>
{d["perimeter"]["html_note"]}
</section>

<section id="runs">
<h2>Le run face à ses repères</h2>
<p class="lede">Même loss, même lecteur, même coupe au premier jour simulé. La colonne
<em>Composite</em> est la masse de probabilité attribuée aux modes ; <em>Tiré</em> est
le mode réellement joué après tirage. Plus bas est meilleur.</p>
{comparison_table(d["runs"], ref_key=d["baseline_key"])}
{legend}
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px">
{run_cards}</div>
</section>

<section id="modeles">
<h2>Modèle par modèle</h2>
<p class="lede">Toutes les décisions <code>LLM</code> du run, ventilées par la colonne
« Fournisseur &amp; Modèle ». Ces sous-ensembles excluent les itinéraires uniques et les
replis d'erreur : ils mesurent le modèle, pas la chaîne.</p>
{comparison_table(d["models"], ref_key=d.get("model_ref_key"))}
{perm_html}
<h3>Les deux modèles ont-ils reçu des décisions comparables&nbsp;?</h3>
<p>La passerelle répartit la charge sans regarder le persona ; le vérifier coûte moins
cher que de le supposer. Si les deux colonnes se ressemblent, l'écart de score porte
sur le modèle et non sur son échantillon.</p>
<div class="scroll"><table><thead><tr><th>Trait</th>
<th>{esc(d["model_a_label"])}</th><th>{esc(d["model_b_label"])}</th></tr></thead>
<tbody>{comp_rows}</tbody></table></div>
{legend}
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px">
{model_cards}</div>
</section>

<section id="decoupes">
<h2>Découpes internes du run</h2>
<p class="lede">Le composite du run entier n'est pas le composite de ses modèles : il
contient aussi les trajets à itinéraire unique, où aucun modèle n'a rien choisi. Les
isoler dit à quel point le score du run mesure la chaîne plutôt que le prompt.</p>
{comparison_table(d["slices"])}
</section>

<section id="sante">
<h2>Santé du run</h2>
<p class="lede">Un composite se lit sur un périmètre. Si les décisions n'atteignent pas
un modèle, le score porte sur celles qui y sont arrivées — et le taire ferait passer une
pénurie de fournisseurs pour une performance.</p>
<div class="tiles">
<div class="tile"><div class="k">Échecs fournisseurs</div>
<div class="v">{fmt_int(health.get("n_errors"))}</div><div class="u">llm_errors.jsonl</div></div>
<div class="tile"><div class="k">Cache LLM</div>
<div class="v">{esc(cycle.get("cache_pct", "—"))}&#8201;%</div>
<div class="u">{fmt_int(cycle.get("cache_hits"))} / {fmt_int(cycle.get("cache_total"))}</div></div>
<div class="tile"><div class="k">Débit</div>
<div class="v">{esc(cycle.get("throughput", "—"))}</div><div class="u">req/min en fin de run</div></div>
<div class="tile"><div class="k">Backlog final</div>
<div class="v">{fmt_int(cycle.get("backlog"))}</div>
<div class="u">{esc(cycle.get("backlog_pct", "—"))}&#8201;% du pipeline</div></div>
<div class="tile"><div class="k">Agents inactifs</div>
<div class="v">{fmt_int(cycle.get("agents_idle"))}</div>
<div class="u">sur {fmt_int(cycle.get("agents_total"))}</div></div>
</div>
<h3>Échecs par fournisseur</h3>
<div class="scroll"><table><thead><tr><th>Fournisseur</th><th class="num">Statut HTTP</th>
<th class="num">Échecs</th></tr></thead><tbody>{err_rows}</tbody></table></div>
<h3>Alarmes du contrôleur</h3>
<div class="scroll"><table><thead><tr><th class="num">n</th><th>Motif</th></tr></thead>
<tbody>{alarm_rows}</tbody></table></div>
</section>

<section id="limites">
<h2>Limites de lecture</h2>
{d["limits"]}
</section>

<section id="provenance">
<h2>Provenance et régénération</h2>
<div class="scroll"><table><thead><tr><th>Source</th><th>Chemin</th>
<th class="num">Empreinte</th></tr></thead><tbody>
{"".join(f"<tr><td><strong>{esc(s['role'])}</strong></td><td class='mono'>{esc(s['path'])}</td>"
         f"<td class='num mono'>{esc((s.get('sha256') or '')[:12])}</td></tr>"
         for s in d["sources"])}
</tbody></table></div>
<div class="formula">make model-compare RUN={esc(run["path"])}</div>
<p>Les scores sont importés de <code>scripts/synthesis/formule_score/metrics.py</code> et
les décisions lues par <code>scripts/synthesis/frames.py</code> : ce sont le lecteur et
la loss de la page principale, sur le même périmètre. Les valeurs de cette page vivent
à côté d'elle dans <code>data.json</code>.</p>
<footer>{esc(d["engine_note"])} — page générée par
<code>scripts/synthesis/model_compare.py</code>.</footer>
</section>
</main></div></body></html>
"""


# ── Assembly ─────────────────────────────────────────────────────────────────

def build(run_path: Path, baseline_paths: list[Path], manifest, cerema: dict,
          scorer, tmpdir: Path) -> dict:
    exclude = manifest.get("common_set.exclude_selection_methods", [])
    measurer = Measurer(cerema, scorer, exclude, tmpdir)
    moves = run_path / "moves.csv"
    raw, fields = read_raw(moves)
    kept, attempts = latest_attempts(raw)

    # ── Scope: what the simulated day contains, most recent attempt ──────────
    day = frames.first_simulated_day(moves)
    day_rows = [r for r in kept
                if frames.simulated_day(r.get("Temps simulé") or "") == day]
    methods = collections.Counter(r.get("Méthode de sélection") or "?"
                                 for r in day_rows)
    legend = {
        "LLM": "décision d'un modèle — dans le score",
        "Un seul itinéraire disponible": "aucun choix à faire — dans le score",
        "LLM Error (Default index)": "repli sur l'itinéraire d'index 0 — hors score, "
                                     "mais joué par la simulation",
        "Pas de déplacement (même localisation)": "pas de trajet — hors score",
        "Pas de solution de déplacement": "aucun itinéraire — hors score",
    }

    runs: list[dict] = []
    full = measurer.measure(raw, fields, f"Run {run_path.name} — lecture brute",
                            "toutes les lignes, comme la page principale les lit")
    full["key"] = "run_raw"
    live = measurer.measure(kept, fields,
                            f"Run {run_path.name} — tentative la plus récente",
                            "lecture retenue ici" if attempts["resumed"] else "")
    live["key"] = "run_live"
    live["highlight"] = True
    if attempts["resumed"]:
        runs.append(full)
    runs.append(live)

    baseline_key = None
    for path in baseline_paths:
        b_raw, b_fields = read_raw(path / "moves.csv")
        entry = measurer.measure(b_raw, b_fields, f"Run {path.name}", "repère")
        entry["key"] = f"base_{path.name}"
        if baseline_key is None:
            baseline_key = entry["key"]
        runs.append(entry)

    # ── Internal slices ──────────────────────────────────────────────────────
    llm_rows = [r for r in kept if r.get("Méthode de sélection") == "LLM"]
    single_rows = [r for r in kept
                   if r.get("Méthode de sélection") == "Un seul itinéraire disponible"]
    slices = [
        {**measurer.measure(llm_rows, fields, "Décisions d'un modèle seules",
                            "colonne « Méthode de sélection » = LLM"),
         "key": "llm", "highlight": True},
        {**measurer.measure(single_rows, fields, "Itinéraires uniques seuls",
                            "aucun choix : la chaîne OTP décide"), "key": "single"},
        {**live, "label": "Les deux réunis (= le run)", "note": "", "key": "both",
         "highlight": False},
    ]

    # ── Model by model ───────────────────────────────────────────────────────
    by_provider = collections.Counter(r.get("Fournisseur & Modèle") or ""
                                     for r in llm_rows)
    models: list[dict] = []
    cache_rows: list[dict] = []
    for provider, n in by_provider.most_common():
        if not provider:
            continue
        if provider.startswith("cache:"):
            cache_rows.extend(r for r in llm_rows
                              if r.get("Fournisseur & Modèle") == provider)
            continue
        rows = [r for r in llm_rows if r.get("Fournisseur & Modèle") == provider]
        label = PROVIDER_MODELS.get(provider, provider)
        entry = measurer.measure(rows, fields, label, f"clé de passerelle {provider}")
        entry["key"] = provider
        entry["provider"] = provider
        entry["rows"] = rows
        models.append(entry)
    if cache_rows:
        entry = measurer.measure(cache_rows, fields, "Tirage au cache",
                                 "décision rejouée depuis le cache de modes")
        entry["key"] = "cache"
        models.append(entry)

    scored_models = [m for m in models if m.get("scored")]
    scored_models.sort(key=lambda m: score_of(m).get("composite", 1e9))
    if scored_models:
        scored_models[0]["highlight"] = True
    ordered = scored_models + [m for m in models if not m.get("scored")]

    perm: dict = {"available": False}
    comp_rows: list[dict] = []
    a_label = b_label = "—"
    if len(scored_models) >= 2 and all("rows" in m for m in scored_models[:2]):
        a, b = scored_models[0], scored_models[1]
        a_label, b_label = a["label"], b["label"]
        perm = permutation_test(measurer, a["rows"], b["rows"], fields)
        comp_rows = comparability(a["rows"], b["rows"])
    for m in models:
        m.pop("rows", None)

    return {
        "run": {"run_id": run_path.name, "path": str(run_path.relative_to(REPO_ROOT))},
        "attempts": attempts,
        "perimeter": {
            "sim_day": day, "methods": dict(methods.most_common()),
            "n_day_total": sum(methods.values()), "legend": legend,
        },
        "runs": runs, "baseline_key": baseline_key, "slices": slices,
        "models": ordered,
        "model_ref_key": scored_models[0]["key"] if scored_models else None,
        "model_a_label": a_label, "model_b_label": b_label,
        "permutation": perm, "comparability": comp_rows,
        "health": run_health(run_path),
        "target_shares": frames.reference_shares(cerema, "global"),
    }


def narrative(payload: dict) -> dict:
    """The page text, derived from the measurements — not written by hand.

    The conclusion must change when the figures change: it is therefore
    computed, including its sign.
    """
    live = next(r for r in payload["runs"] if r.get("key") == "run_live")
    base = next((r for r in payload["runs"]
                 if r.get("key") == payload["baseline_key"]), None)
    live_c = score_of(live).get("composite")
    base_c = score_of(base).get("composite") if base else None
    best_run = min((r for r in payload["runs"] if score_of(r).get("composite")),
                   key=lambda r: score_of(r)["composite"], default=None)
    models = [m for m in payload["models"] if m.get("scored")]
    perim = payload["perimeter"]
    fallbacks = perim["methods"].get("LLM Error (Default index)", 0)
    fallback_pct = 100.0 * fallbacks / max(1, perim["n_day_total"])

    if live_c is None or base_c is None:
        sense, word = 0.0, "incomparable"
    else:
        sense = live_c - base_c
        word = ("meilleur" if sense < -0.5 else
                "moins bon" if sense > 0.5 else "équivalent")

    tiles = [
        {"k": "Composite du run", "v": fmt(live_c), "u": "emd_jsd, masse de probabilité",
         "hi": True},
        {"k": "Repère précédent", "v": fmt(base_c),
         "u": (base["label"] if base else "—")},
        {"k": "Écart", "v": ("=" if word == "équivalent"
                             else f"{'+' if sense > 0 else '−'}{fmt(abs(sense))}"),
         "u": word},
    ]
    if models:
        tiles.append({"k": "Meilleur modèle", "v": fmt(score_of(models[0]).get("composite")),
                      "u": models[0]["label"], "hi": True})
        if len(models) > 1:
            tiles.append({"k": "Le suivant", "v": fmt(score_of(models[1]).get("composite")),
                          "u": models[1]["label"]})
    tiles.append({"k": "Replis d'erreur", "v": f"{fmt(fallback_pct, 1)} %",
                  "u": f"{fmt_int(fallbacks)} décisions hors score"})

    liaison = "à son repère" if word == "équivalent" else "que son repère"
    lede = (f"Sur la formule de score, ce run est <strong>{word}</strong> "
            f"{liaison} : composite {fmt(live_c)} contre {fmt(base_c)}. ")
    if best_run is not None and best_run.get("key") != "run_live":
        lede += (f"Le meilleur run mesuré reste "
                 f"<strong>{esc(best_run['label'])}</strong> à "
                 f"{fmt(score_of(best_run).get('composite'))}. ")
    if models:
        lede += (f"La ventilation par modèle, elle, sépare nettement : "
                 f"<strong>{esc(models[0]['label'])}</strong> à "
                 f"{fmt(score_of(models[0]).get('composite'))}"
                 + (f" contre {fmt(score_of(models[1]).get('composite'))} pour "
                    f"{esc(models[1]['label'])}." if len(models) > 1 else "."))

    perm = payload.get("permutation") or {}
    noise = max(abs(perm.get("min", 0.0)), abs(perm.get("max", 0.0))) if perm.get(
        "available") else None

    body = ["<div class='note'><div class='t'>Ce que disent les chiffres</div><ul>"]
    if word == "équivalent":
        body.append(f"<li>Le composite du run entier ({fmt(live_c)}) ne bouge pas par "
                    f"rapport au repère ({fmt(base_c)})"
                    + (f" : l'écart, {fmt(abs(sense))} point, est très inférieur au bruit "
                       f"de découpage mesuré plus bas ({fmt(noise)} point d'amplitude)."
                       if noise else ".") + "</li>")
    else:
        body.append(f"<li>Le composite du run entier passe de {fmt(base_c)} à "
                    f"{fmt(live_c)}, soit {fmt(abs(sense))} point de "
                    f"{'perte' if sense > 0 else 'gain'} de fidélité"
                    + (f", à comparer au bruit de découpage mesuré plus bas "
                       f"({fmt(noise)} point d'amplitude)." if noise else ".") + "</li>")
    if len(models) >= 2:
        spread = abs(score_of(models[0]).get("composite", 0.0)
                     - score_of(models[-1]).get("composite", 0.0))
        solid = perm.get("available") and perm.get("p_value", 1.0) < 0.05
        body.append(f"<li>La moyenne du run masque un écart "
                    f"{'qui ne vient pas du découpage' if solid else 'que le découpage suffit à produire'} "
                    f"entre modèles : {esc(models[0]['label'])} est à "
                    f"{fmt(score_of(models[0]).get('composite'))}, soit {fmt(spread)} point(s) "
                    f"du dernier du classement.</li>")
    body.append(f"<li>{fmt(fallback_pct, 1)}&#8201;% des décisions du jour simulé sont des "
                f"replis d'erreur : elles sortent du score, mais la simulation les a jouées. "
                f"Le score porte donc sur une journée dont une part n'a pas été décidée par "
                f"un modèle.</li>")
    body.append("</ul></div>")

    limits = ["<div class='missing'><div class='t'>À ne pas conclure de cette page</div><ul>"]
    if payload["attempts"]["resumed"]:
        limits.append(
            f"<li><strong>Reprise à chaud.</strong> Le journal porte "
            f"{fmt_int(payload['attempts']['n_total'])} lignes pour "
            f"{fmt_int(payload['attempts']['n_kept'])} couples (personne, activité) : la "
            f"reprise a rejoué la journée dans le même dossier. La lecture brute — celle de "
            f"<code>make synthesis</code> — compte donc deux fois les décisions d'avant la "
            f"reprise. C'est pourquoi cette page mesure la tentative la plus récente, et "
            f"publie les deux lectures côte à côte.</li>")
    limits.append(
        "<li><strong>Effectif.</strong> Un sous-ensemble par modèle est plus petit que le "
        "run : les divergences par strate sont biaisées vers le haut à petits effectifs. "
        "Le test de permutation dit si l'écart entre deux modèles y survit ; il ne rend "
        "pas comparables un modèle et le run entier.</li>")
    limits.append(
        "<li><strong>Ce n'est pas un classement de modèles en général.</strong> C'est un "
        "classement sur ce prompt, ce jeu de choix OTP, cette population et ce jour "
        "simulé. Un modèle qui répartit mieux la probabilité entre quatre modes sur ce "
        "prompt-là peut se comporter autrement sur un autre.</li>")
    limits.append(
        "<li><strong>Les replis d'erreur ne sont pas neutres.</strong> Ils sortent du "
        "score, pas de la simulation : l'agent a bien pris l'itinéraire d'index 0. Les "
        "trajectoires, les temps de parcours et la charge du réseau du run portent cette "
        "part non décidée.</li>")
    limits.append("</ul></div>")

    return {"tiles": tiles, "lede": lede, "body": "".join(body),
            "limits": "".join(limits)}


def perimeter_note(payload: dict) -> str:
    p = payload["perimeter"]
    fall = p["methods"].get("LLM Error (Default index)", 0)
    llm = p["methods"].get("LLM", 0)
    single = p["methods"].get("Un seul itinéraire disponible", 0)
    total = max(1, p["n_day_total"])
    return (f"<div class='note'><div class='t'>Lecture</div>"
            f"<p>Sur les {fmt_int(total)} décisions du jour simulé, "
            f"{fmt_int(llm + single)} entrent dans le score — dont "
            f"{fmt_int(single)} ({fmt(100.0 * single / total, 1)}&#8201;%) sans aucun "
            f"choix à faire, un seul itinéraire ayant été proposé. "
            f"{fmt_int(fall)} ({fmt(100.0 * fall / total, 1)}&#8201;%) sont des replis "
            f"d'erreur : le modèle n'a pas répondu, le contrôleur a pris l'itinéraire "
            f"d'index 0. Ces lignes sont exclues du score — il n'y a pas de choix à "
            f"noter — mais la simulation les a jouées.</p></div>")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run to measure")
    parser.add_argument("--baseline", action="append", default=[],
                        help="reference run (repeatable; the first one is the baseline)")
    parser.add_argument("--config", help="sources manifest (default: sources.yaml)")
    parser.add_argument("--out", help="output directory (default: "
                                      "docs/synthesis/models/<run_id>)")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    global PROVIDER_MODELS
    PROVIDER_MODELS = load_provider_models(REPO_ROOT / "config/llm_gateway/providers.yaml")

    cerema_src = manifest.track("cerema", manifest.get("cerema"),
                                "Référence EMC² 2023 — parts modales cibles")
    if not cerema_src.exists:
        print(f"[erreur] Référence EMC² introuvable : {cerema_src.rel}", file=sys.stderr)
        return 2
    cerema = frames.load_cerema(cerema_src.path)

    formule_score, engine_error = import_formule_score()
    if formule_score is None:
        print(f"[erreur] {engine_error}", file=sys.stderr)
        return 2
    scorer = frames.Scorer(formule_score, manifest.get("score.weights", {}),
                           manifest.get("score.metric", "emd_jsd"),
                           manifest.get("score.secondary", "l1_composite"))

    def resolve(value: str) -> Path:
        p = Path(value)
        return (p if p.is_absolute() else REPO_ROOT / p).resolve()

    run_path = resolve(args.run)
    if not (run_path / "moves.csv").exists():
        print(f"[erreur] moves.csv not found in {run_path}", file=sys.stderr)
        return 2
    baselines = []
    for value in args.baseline:
        path = resolve(value)
        if (path / "moves.csv").exists():
            baselines.append(path)
        else:
            print(f"[avertissement] baseline skipped (moves.csv missing): {path}",
                  file=sys.stderr)
    if not baselines:
        configured = manifest.get("common_set.run")
        if configured:
            path = resolve(configured)
            if (path / "moves.csv").exists() and path != run_path:
                baselines.append(path)

    tmpdir = Path(tempfile.mkdtemp(prefix="model_compare_"))
    payload = build(run_path, baselines, manifest, cerema, scorer, tmpdir)
    payload["generated_at"] = datetime.now().strftime("%d/%m/%Y à %H:%M")
    payload["engine_note"] = "Formule de score : scripts/synthesis/formule_score"
    payload["verdict"] = narrative(payload)
    payload["limits"] = payload["verdict"].pop("limits")
    payload["perimeter"]["html_note"] = perimeter_note(payload)
    manifest.track("moves", run_path / "moves.csv", "Journal du run mesuré")
    for path in baselines:
        manifest.track(f"moves_{path.name}", path / "moves.csv", f"Repère {path.name}")
    manifest.track("metrics", "scripts/synthesis/formule_score/metrics.py",
                   "Formule de score")
    payload["sources"] = [s.to_dict() for s in manifest.sources.values()]

    out_dir = (resolve(args.out) if args.out
               else REPO_ROOT / "docs/synthesis/models" / run_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    (out_dir / "index.html").write_text(render(payload), encoding="utf-8")
    print(f"[ok] {out_dir / 'index.html'}")
    print(f"[ok] {out_dir / 'data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
