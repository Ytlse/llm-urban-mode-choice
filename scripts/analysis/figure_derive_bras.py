#!/usr/bin/env python3
"""Drift between the treated and the control arm of a memory A/B experiment, day by day.

An A/B only measures the effect of an event if both arms stay the same world until the
injection. This script says from when that is no longer true, and by how much the paired
decisions already diverge before the event:

1. per simulated day, the share of decision prompts (`itinary_multi_agent`) identical byte
   for byte in both arms, and the share of identical answers when the prompt is;
   same count for the evening memories (`stm_reflection`);
2. decision by decision, the total variation distance between the mode distributions
   stated by both arms (0: identical, 100: disjoint), with the mean of the day;
3. a one-sided Fisher test on the share of decisions more than SEUIL points apart,
   after the event against the THREE last days before it — the drift grows over
   time, and comparing it with the whole period before would favour it.

Standalone HTML page (inline SVG, stdlib only, composed in English), summary on the console.

Usage:
    python scripts/analysis/figure_derive_bras.py TRAITE TEMOIN SORTIE.html \\
        --evenement 2026-03-26 [--suivre 286923@12:40] [--titre "…"]

    TRAITE, TEMOIN: archive directories of the two arms (experiments/archive/<run>), which
    hold `moves.csv` and `llm_exchanges.jsonl`. --suivre circles a slot
    (person@HH:MM of departure) to show where the drift started.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
from collections import defaultdict
from datetime import date
from math import comb
from pathlib import Path

SEUIL = 20.0
PCOLS = ["P(Marche) %", "P(Vélo) %", "P(Voiture Privée) %", "P(Transports_collectifs) %",
         "P(Train) %", "P(Deux-roues motorisé) %", "P(Autres modes) %"]
MODE_EN = {"Voiture Privée": "car", "Vélo": "bike", "Marche": "walk",
           "Transports_collectifs": "public transport", "Train": "train",
           "Deux-roues motorisé": "motorbike"}


# --------------------------------------------------------------------------- reading
def lire_echanges(chemin: Path) -> list[dict]:
    """`llm_exchanges.jsonl` is a sequence of indented JSON objects, not one object per line."""
    texte = chemin.read_text(encoding="utf-8")
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        while i < len(texte) and texte[i] in " \n\r\t":
            i += 1
        if i >= len(texte):
            return out
        obj, i = dec.raw_decode(texte, i)
        out.append(obj)


# Provider prefix of an exchange served again by the exact-prompt replay (2026-09-25).
PREFIXE_REJEU = "rejeu_ab:"


def rejoue(e: dict) -> bool:
    return str(e.get("provider") or "").startswith(PREFIXE_REJEU)


def indexer(echanges: list[dict], categorie: str) -> dict:
    """Pairing key of a call: simulated day, simulated instant, agents of the batch.

    Exchanges served again by replay are excluded: they are counted apart (`rejoues_par_jour`),
    because a treated call merged with other agents does not pair with the key of the
    single task that the control replayed.
    """
    index = {}
    for e in echanges:
        if e.get("category") != categorie or rejoue(e):
            continue
        rep = e.get("response")
        agents = tuple(sorted(str(r.get("agent_id", "")) for r in rep)) if isinstance(rep, list) else ()
        index[(e["sim_day"], e["sim_ts"], agents)] = (
            json.dumps(e["messages"], ensure_ascii=False, sort_keys=True),
            json.dumps(rep, ensure_ascii=False, sort_keys=True),
        )
    return index


def rejoues_par_jour(echanges: list[dict], categorie: str) -> dict:
    """Control calls served by replay, per day: identical prompt by construction (the replay
    key IS the prompt's fingerprint), identical answer since it is the treated one's."""
    n = defaultdict(int)
    for e in echanges:
        if e.get("category") == categorie and rejoue(e):
            n[e.get("sim_day")] += 1
    return n


def identite_par_jour(t: dict, c: dict, rejoues: dict | None = None) -> dict:
    """Per day: [paired, identical prompts, identical answers among them]."""
    par_jour = defaultdict(lambda: [0, 0, 0])
    for jour, n in (rejoues or {}).items():
        if jour:
            par_jour[jour][0] += n
            par_jour[jour][1] += n
            par_jour[jour][2] += n
    for k in set(t) & set(c):
        j = par_jour[k[0]]
        j[0] += 1
        if t[k][0] == c[k][0]:
            j[1] += 1
            j[2] += t[k][1] == c[k][1]
    return par_jour


def lire_trajets(run: Path) -> dict:
    with (run / "moves.csv").open(encoding="utf-8") as f:
        return {(r["ID Personne"], r["ID Activité"], r["Heure de départ"][:10]): r
                for r in csv.DictReader(f)}


def distrib(r: dict) -> list[float]:
    out = []
    for col in PCOLS:
        try:
            out.append(float(r.get(col) or 0))
        except ValueError:
            out.append(0.0)
    return out


def variation_totale(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(distrib(a), distrib(b)))


def fisher_unilateral(a: int, n1: int, b: int, n2: int) -> float:
    """P(X >= a) under the null hypothesis, a successes out of n1 against b out of n2."""
    k, n = a + b, n1 + n2
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / comb(n, k)


# --------------------------------------------------------------------------- measure
def mesurer(traite: Path, temoin: Path, evenement: str, suivre: str | None) -> dict:
    ex_t, ex_c = lire_echanges(traite / "llm_exchanges.jsonl"), lire_echanges(temoin / "llm_exchanges.jsonl")
    dec = identite_par_jour(indexer(ex_t, "itinary_multi_agent"), indexer(ex_c, "itinary_multi_agent"),
                            rejoues_par_jour(ex_c, "itinary_multi_agent"))
    stm = identite_par_jour(indexer(ex_t, "stm_reflection"), indexer(ex_c, "stm_reflection"),
                            rejoues_par_jour(ex_c, "stm_reflection"))

    t, c = lire_trajets(traite), lire_trajets(temoin)
    jours = sorted({k[2] for k in set(t) | set(c)})
    suivi_pid, suivi_hm = (suivre.split("@") + [""])[:2] if suivre else (None, None)
    paires = []
    for k in sorted(set(t) & set(c), key=lambda k: t[k]["Heure de départ"]):
        a, b = t[k], c[k]
        if a["Méthode de sélection"] != "LLM" or b["Méthode de sélection"] != "LLM":
            continue  # forced returns and single itineraries: no decision to compare
        hm = a["Heure de départ"][11:16]
        paires.append({
            "person": k[0], "day": k[2], "time": hm,
            "frac": (int(hm[:2]) * 60 + int(hm[3:5])) / 1440,
            "tv": variation_totale(a, b),
            "mode_t": MODE_EN.get(a["Mode de transport Choisi"], a["Mode de transport Choisi"]),
            "mode_c": MODE_EN.get(b["Mode de transport Choisi"], b["Mode de transport Choisi"]),
            "after": k[2] >= evenement,
            "tracked": k[0] == suivi_pid and hm == suivi_hm,
        })

    avant = [p for p in paires if not p["after"]]
    apres = [p for p in paires if p["after"]]
    jours_avant = sorted({p["day"] for p in avant})
    veille = [p for p in avant if p["day"] in jours_avant[-3:]]
    debut = [p for p in avant if p["day"] not in jours_avant[-3:]]

    def part(ps):
        n = sum(p["tv"] > SEUIL for p in ps)
        return n, len(ps)

    s_apres, s_veille, s_avant = part(apres), part(veille), part(avant)
    return {
        "jours": jours, "evenement": evenement, "paires": paires, "dec": dec, "stm": stm,
        "debut": part(debut), "veille": s_veille, "apres": s_apres, "avant": s_avant,
        "p_veille": fisher_unilateral(s_apres[0], s_apres[1], s_veille[0], s_veille[1]),
        "p_avant": fisher_unilateral(s_apres[0], s_apres[1], s_avant[0], s_avant[1]),
        "jours_veille": jours_avant[-3:],
        "moyenne": {j: statistics.mean([p["tv"] for p in paires if p["day"] == j])
                    for j in jours if any(p["day"] == j for p in paires)},
    }


# --------------------------------------------------------------------------- rendering
def etiquette(j: str) -> str:
    d = date.fromisoformat(j)
    return f"{d.strftime('%a')} {d.day}"


def svg_prompts(m: dict, x0: float, w: float) -> str:
    jours, h, top = m["jours"], 150, 18
    pas = w / len(jours)
    y = lambda v: top + h * (1 - v)
    out = [f'<svg viewBox="0 0 {x0 + w + 10} {top + h + 48}" role="img" '
           f'aria-label="Share of decision prompts identical in both runs, by day">']
    for v in (0, .5, 1):
        out.append(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{x0 - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
    for i, j in enumerate(jours):
        n, ident, _ = m["dec"].get(j, [0, 0, 0])
        cx = x0 + pas * (i + .5)
        if n:
            v = ident / n
            bh = max(h * v, 0)
            if bh > 0:
                out.append(f'<path class="bar" d="M{cx - 11:.1f},{y(0):.1f} V{y(v) + 4:.1f} '
                           f'q0,-4 4,-4 h14 q4,0 4,4 V{y(0):.1f} Z"><title>{etiquette(j)}: '
                           f'{ident} of {n} decision prompts identical</title></path>')
            out.append(f'<text class="val" x="{cx:.1f}" y="{y(v) - 6:.1f}" text-anchor="middle">{ident}/{n}</text>')
        out.append(f'<text class="tick" x="{cx:.1f}" y="{top + h + 18}" text-anchor="middle">{etiquette(j)}</text>')
    ie = jours.index(m["evenement"]) if m["evenement"] in jours else None
    if ie is not None:
        xe = x0 + pas * ie
        out.append(f'<line class="event" x1="{xe:.1f}" x2="{xe:.1f}" y1="{top - 6}" y2="{top + h}"/>'
                   f'<text class="ann" x="{xe + 5:.1f}" y="{top + 4}">event</text>')
    out.append("</svg>")
    return "".join(out)


def svg_ecarts(m: dict, x0: float, w: float) -> str:
    jours, h, top = m["jours"], 260, 22
    pas = w / len(jours)
    y = lambda v: top + h * (1 - v / 100)
    xj = lambda j, frac: x0 + pas * (jours.index(j) + .08 + .84 * frac)
    out = [f'<svg viewBox="0 0 {x0 + w + 10} {top + h + 48}" role="img" '
           f'aria-label="Gap between the two runs, decision by decision">']
    out.append(f'<rect class="floor" x="{x0}" y="{y(SEUIL):.1f}" width="{w}" height="{y(0) - y(SEUIL):.1f}"/>')
    for v in (0, 25, 50, 75, 100):
        out.append(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{x0 - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    out.append(f'<text class="lab" transform="translate({x0 - 38},{top + h / 2}) rotate(-90)" '
               f'text-anchor="middle">Total variation (points)</text>')
    for i, j in enumerate(jours):
        out.append(f'<text class="tick" x="{x0 + pas * (i + .5):.1f}" y="{top + h + 18}" '
                   f'text-anchor="middle">{etiquette(j)}</text>')
        if i and (date.fromisoformat(j) - date.fromisoformat(jours[i - 1])).days > 1:
            xw = x0 + pas * i
            out.append(f'<line class="wkend" x1="{xw:.1f}" x2="{xw:.1f}" y1="{top}" y2="{top + h}"/>')
    # mean of the day: one line, 2 px, points at the centres of the days
    pts = [(x0 + pas * (jours.index(j) + .5), y(v)) for j, v in m["moyenne"].items()]
    out.append('<polyline class="mean" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
    for (a, b), (j, v) in zip(pts, m["moyenne"].items()):
        out.append(f'<circle class="mean-pt" cx="{a:.1f}" cy="{b:.1f}" r="4"><title>{etiquette(j)}: '
                   f'mean gap {v:.1f} points</title></circle>')
    for p in m["paires"]:
        cx, cy = xj(p["day"], p["frac"]), y(p["tv"])
        tip = (f'{p["person"]}, {etiquette(p["day"])} {p["time"]} — exposed: {p["mode_t"]}, '
               f'control: {p["mode_c"]}, gap {p["tv"]:.0f} points')
        cls = "pt-post" if p["after"] else "pt-pre"
        out.append(f'<g class="hit" data-tip="{html.escape(tip)}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" '
                   f'fill="transparent"/><circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/>'
                   + (f'<circle class="track" cx="{cx:.1f}" cy="{cy:.1f}" r="7"/>' if p["tracked"] else "")
                   + "</g>")
    ie = jours.index(m["evenement"]) if m["evenement"] in jours else None
    if ie is not None:
        xe = x0 + pas * ie
        out.append(f'<line class="event" x1="{xe:.1f}" x2="{xe:.1f}" y1="{top - 10}" y2="{top + h}"/>'
                   f'<text class="ann" x="{xe + 5:.1f}" y="{top - 2}">event</text>')
    out.append("</svg>")
    return "".join(out)


CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--grid:#e7e6e2;--floor:#ecebe7;--s1:#2a78d6;--s2:#eb6834;--accent:#0b0b0b;
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--surface);color:var(--ink);
padding:24px 16px;max-width:1080px;margin:0 auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8d8c86;--grid:#33332f;--floor:#2a2a27;
--s1:#3987e5;--s2:#d95926;--accent:#fff}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#8d8c86;--grid:#33332f;--floor:#2a2a27;--s1:#3987e5;--s2:#d95926;--accent:#fff}
body{margin:0;background:#fcfcfb}@media (prefers-color-scheme:dark){body{background:#1a1a19}}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 2px}p.sub{color:var(--ink2);margin:0 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:14px 0 6px}
.tile{border:1px solid var(--grid);border-radius:8px;padding:10px 12px}.tile b{display:block;font-size:22px}
.tile span{color:var(--ink2);font-size:12.5px}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.tick{fill:var(--ink2);font-size:11.5px}.lab{fill:var(--ink2);font-size:12px}
.val{fill:var(--ink2);font-size:11px}.ann{fill:var(--ink2);font-size:11.5px}
.bar{fill:var(--s1)}.floor{fill:var(--floor)}
.wkend{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3}
.event{stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 3}
.mean{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round}
.mean-pt{fill:var(--s1);stroke:var(--surface);stroke-width:2}
.pt-pre{fill:var(--muted);stroke:var(--surface);stroke-width:1.5}
.pt-post{fill:var(--ink);stroke:var(--surface);stroke-width:1.5}
.track{fill:none;stroke:var(--s2);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 4px;color:var(--ink2);font-size:12.5px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.cap{color:var(--ink2);font-size:13px;margin:6px 0 0}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--grid);
border-radius:6px;padding:5px 8px;font-size:12.5px;display:none;max-width:320px;z-index:9}
details{margin-top:10px;color:var(--ink2)}table{border-collapse:collapse;font-size:12.5px;margin-top:6px}
td,th{border-bottom:1px solid var(--grid);padding:3px 8px;text-align:left}
@media (max-width:600px){.viz-root{padding:16px}}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('.hit').forEach(g=>{
 g.addEventListener('mousemove',e=>{tip.textContent=g.dataset.tip;tip.style.display='block';
  tip.style.left=Math.min(e.clientX+12,innerWidth-330)+'px';tip.style.top=(e.clientY+12)+'px';});
 g.addEventListener('mouseleave',()=>{tip.style.display='none';});});
"""


def rendre(m: dict, titre: str, sous_titre: str, suivre: str | None) -> str:
    stm1 = m["stm"].get(m["jours"][0], [0, 0, 0])
    dec_ident = sum(v[1] for v in m["dec"].values())
    dec_meme = sum(v[2] for v in m["dec"].values())
    dernier_ident = max((j for j, v in m["dec"].items() if v[1]), default=None)
    (dv, dn), (vv, vn), (av, an) = m["debut"], m["veille"], m["apres"]
    pct = lambda a, n: f"{100 * a / n:.0f} %" if n else "–"
    tiles = [
        (f"{stm1[1] - stm1[2]} of {stm1[1]}", f"memory notes of {etiquette(m['jours'][0])} sent with a prompt "
         f"identical in both runs that came back different (temperature 0)"),
        (f"{dec_meme} of {dec_ident}", "decision prompts identical in both runs that got the identical answer"),
        (etiquette(dernier_ident) if dernier_ident else "–", "last day with a decision prompt identical in both runs"),
        (f"{pct(vv, vn)} → {pct(av, an)}", f"decisions more than {SEUIL:.0f} points apart: the three days before "
         f"the event, then from the event on (one-sided Fisher p = {m['p_veille']:.2f})"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>'
                         for a, b in tiles)
    lignes = "".join(
        f"<tr><td>{p['person']}</td><td>{etiquette(p['day'])} {p['time']}</td><td>{p['mode_t']}</td>"
        f"<td>{p['mode_c']}</td><td>{p['tv']:.0f}</td><td>{'after' if p['after'] else 'before'}</td></tr>"
        for p in m["paires"])
    suivi = ""
    if suivre:
        ecart = next((p for p in m["paires"] if p["tracked"] and p["mode_t"] != p["mode_c"]), None)
        quand = (f" — modes first differ on {etiquette(ecart['day'])} ({ecart['mode_t']} exposed, "
                 f"{ecart['mode_c']} control)") if ecart else " — same mode in both runs throughout"
        suivi = (f'<span><i style="border:2px solid var(--s2);background:transparent"></i>'
                 f'tracked trip {html.escape(suivre)}{html.escape(quand)}</span>')
    x0, w = 56, 1000
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Drift between runs</title>
<style>{CSS}</style></head><body><div class="viz-root">
<h1>{html.escape(titre)}</h1><p class="sub">{html.escape(sous_titre)}</p>
<div class="tiles">{tiles_html}</div>
<h2>1 · Decision prompts identical in both runs</h2>
<p class="cap">Share of paired decision calls whose prompt is byte-identical in the exposed and the control run, by simulated day (labels: identical / paired).</p>
{svg_prompts(m, x0, w)}
<h2>2 · Gap between the two runs, decision by decision</h2>
<div class="legend"><span><i style="background:var(--muted)"></i>before the event</span>
<span><i style="background:var(--ink)"></i>from the event on</span>
<span><i style="background:var(--s1)"></i>mean gap of the day</span>{suivi}
<span><i style="border-radius:2px;background:var(--floor)"></i>below {SEUIL:.0f} points</span></div>
{svg_ecarts(m, x0, w)}
<p class="cap">Total variation distance between the mode distributions stated in the two runs for the same trip,
in points (0: identical, 100: disjoint). Trips with a single itinerary carry no decision and are left out.
Before the event: {pct(dv, dn)} of decisions above {SEUIL:.0f} points on the first days ({dn} decisions),
{pct(vv, vn)} on the last three ({vn}); from the event on, {pct(av, an)} ({an}).
Against the whole period before the event, one-sided Fisher p = {m['p_avant']:.3f}.</p>
<details><summary>Table view: {len(m['paires'])} paired decisions</summary><table>
<tr><th>person</th><th>trip</th><th>exposed</th><th>control</th><th>gap</th><th>period</th></tr>{lignes}</table></details>
<div id="tip"></div></div><script>{JS}</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traite", type=Path)
    ap.add_argument("temoin", type=Path)
    ap.add_argument("sortie", type=Path)
    ap.add_argument("--evenement", required=True, help="first exposed simulated day, YYYY-MM-DD")
    ap.add_argument("--suivre", help="slot to circle, person@HH:MM")
    ap.add_argument("--titre", default="The exposed and the control run drift apart before the event")
    ap.add_argument("--sous-titre", default="")
    a = ap.parse_args()
    for d in (a.traite, a.temoin):
        for f in ("moves.csv", "llm_exchanges.jsonl"):
            if not (d / f).is_file():
                print(f"ERROR: {d / f} not found", file=sys.stderr)
                return 2
    m = mesurer(a.traite, a.temoin, a.evenement, a.suivre)
    a.sortie.parent.mkdir(parents=True, exist_ok=True)
    a.sortie.write_text(rendre(m, a.titre, a.sous_titre, a.suivre), encoding="utf-8")
    (dv, dn), (vv, vn), (av, an) = m["debut"], m["veille"], m["apres"]
    print(f"paired decisions: {len(m['paires'])}; > {SEUIL:.0f} points: start {dv}/{dn}, "
          f"three days before {vv}/{vn} ({', '.join(m['jours_veille'])}), after {av}/{an}")
    print(f"One-sided Fisher, after against three days before: p = {m['p_veille']:.3f}; "
          f"against the whole period before: p = {m['p_avant']:.3f}")
    for nom, table in (("décisions", m["dec"]), ("souvenirs STM", m["stm"])):
        print(f"{nom} — per day (paired, identical prompts, identical answers):",
              {j: tuple(v) for j, v in sorted(table.items())})
    print(f"✅ figure written: {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
