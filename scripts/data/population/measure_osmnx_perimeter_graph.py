"""measure_osmnx_perimeter_graph.py — Measurements O2 and O4 of the scope report (ticket 031 § 1.4).

    services/llm-agents/.venv/bin/python -m scripts.data.population.measure_osmnx_perimeter_graph \\
        --population data/population/population_1000_PANEL_v3/population.json \\
        --trace docs/traces/<date>_mesures_graphe_perimetre

WHAT IT MEASURES, on the activity pairs that a population routes at step 4+5 of the notebook
(`collect_scheduling_pairs`: one mode per person, car if the household has one, bike otherwise):

  * **O2 — same node.** For each pair, origin and destination are snapped to the nearest graph
    node (`ox.distance.nearest_nodes`, in batch). Two points on the SAME node are
    not routed: `_route_sync` serves them the "same node" fallback (crow-fly × 1.3 at the
    speed of the mode) — it is not an itinerary. The share of these pairs is given per origin
    ring, for the production graph (30 km disc) and for the graph of the polygon of the
    453 communes. **Criterion 3 of ticket 031**: the "same node" pairs more than
    500 m apart as the crow flies — those a complete graph would have separated — must be ≈ 0
    (≤ 0.5 %); shorter pairs are genuine short trips, not a graph defect.
  * **O2 — ms per route, None routes.** Each pair is routed with `_route_sync` (the production
    code, congestion of Monday 8 January 2024) on both graphs; median and
    mean duration per route, share of `None`, and distance to the nearest node (median, p95).
  * **O4 — memory of a worker.** A separate process loads the pickle like `route_worker.init_worker`
    (the three graphs) and returns its peak memory and its loading time, for each key:
    this is the bound of `MAX_WORKERS` (machine RAM / RAM per worker).

Nothing is written into the population nor into the SQLite cache. Timestamped trace with `--trace`.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
LLMAGENTS_PATH = REPO_ROOT / "services" / "llm-agents"
for _p in (str(REPO_ROOT), str(LLMAGENTS_PATH), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_osmnx_perimeter_graph import (OSMNX_CACHE_DIR, PERIMETER_CACHE_KEY,  # noqa: E402
                                         PRODUCTION_CACHE_KEY)

logger = logging.getLogger("osmnx.mesures")

GRAPHS = {"disque_30km": PRODUCTION_CACHE_KEY, "polygone_453": PERIMETER_CACHE_KEY}
SAME_NODE_FAR_M = 500.0   # beyond: two points a complete graph would have separated (criterion 3)
_MODE = {"foot": "walk", "bicycle": "bike", "car": "drive"}
SIM_DATE = date(2024, 1, 8)   # Monday, like route_worker


def _load(cache_dir: Path, key: str):
    t0 = time.monotonic()
    with (cache_dir / f"graphs_{key}.pkl").open("rb") as fh:
        graphs = pickle.load(fh)
    with (cache_dir / f"boundary_{key}.pkl").open("rb") as fh:
        boundary = pickle.load(fh)
    # Congestion zones (ticket 031, decision 4): the historical 30 km graph does not carry them
    # until the runtime has loaded it once; we set them in memory, without rewriting
    # the pickle (measurement tool, not a cache).
    from trip_helper.congestion_zones import ensure_zones
    counts = ensure_zones(graphs, boundary, log=logger)
    logger.info("graphs %s loaded in %.1fs%s", key, time.monotonic() - t0,
                " — zones de congestion posées en mémoire" if counts else " — zones lues dans le pickle")
    return graphs, boundary


def _couronne_of(zones, lat: float, lon: float) -> str:
    return zones.classify(lat, lon) or "inconnue"


def measure_pairs(population: list[dict], cache_dir: Path, route_all: bool) -> dict:
    import numpy as np
    import osmnx as ox
    from mobility_core.residence_zone import CommunalZones
    from models import Location
    from population_utils import collect_scheduling_pairs
    from trip_helper.osmnx_direct import _crow_flies_m, _route_sync

    zones = CommunalZones.load()
    pairs = sorted(collect_scheduling_pairs(population))
    logger.info("%d paires de planification (%s)", len(pairs), dict(Counter(p[4] for p in pairs)))
    couronne_o = [_couronne_of(zones, p[0], p[1]) for p in pairs]
    crow_m = [_crow_flies_m(Location(lat=p[0], lon=p[1]), Location(lat=p[2], lon=p[3])) for p in pairs]
    out: dict = {"n_paires": len(pairs), "par_mode": dict(Counter(p[4] for p in pairs)),
                 "par_couronne_origine": dict(Counter(couronne_o)),
                 "seuil_meme_noeud_lointain_m": SAME_NODE_FAR_M, "graphes": {}}

    for label, key in GRAPHS.items():
        graphs, boundary = _load(cache_dir, key)
        res: dict = {"cle": key, "modes": {}}
        same_by_c: Counter = Counter()
        far_same_by_c: Counter = Counter()     # same node AND > SAME_NODE_FAR_M as the crow flies
        n_by_c: Counter = Counter(couronne_o)
        snap_dists: list[float] = []
        durations_ms: list[float] = []
        n_none = 0
        n_routed = 0
        for mode in ("car", "bicycle"):
            idx = [i for i, p in enumerate(pairs) if p[4] == mode]
            if not idx:
                continue
            G = graphs[_MODE[mode]]
            X = [pairs[i][1] for i in idx] + [pairs[i][3] for i in idx]
            Y = [pairs[i][0] for i in idx] + [pairs[i][2] for i in idx]
            t0 = time.monotonic()
            nodes, dists = ox.distance.nearest_nodes(G, X, Y, return_dist=True)
            t_snap = time.monotonic() - t0
            n = len(idx)
            same = [nodes[k] == nodes[n + k] for k in range(n)]
            for k, i in enumerate(idx):
                if same[k]:
                    same_by_c[couronne_o[i]] += 1
                    if crow_m[i] > SAME_NODE_FAR_M:
                        far_same_by_c[couronne_o[i]] += 1
            snap_dists.extend(float(d) for d in dists)
            res["modes"][mode] = {"n": n, "meme_noeud": int(sum(same)),
                                  "part_meme_noeud_pct": round(100.0 * sum(same) / n, 1),
                                  "noeuds": G.number_of_nodes(), "aretes": G.number_of_edges(),
                                  "rabattement_lot_s": round(t_snap, 1)}
            # Actual routing, production code.
            sample = idx if route_all else idx[:: max(1, len(idx) // 400)]
            t0 = time.monotonic()
            for i in sample:
                lat1, lon1, lat2, lon2, m, hour = pairs[i]
                h = int(hour) if hour is not None else 8
                cdt = datetime(SIM_DATE.year, SIM_DATE.month, SIM_DATE.day, min(h, 23), 0)
                t1 = time.monotonic()
                r = _route_sync(G, boundary, Location(lat=lat1, lon=lon1), Location(lat=lat2, lon=lon2),
                                _MODE[m], cdt)
                durations_ms.append(1000.0 * (time.monotonic() - t1))
                n_routed += 1
                if r is None:
                    n_none += 1
            res["modes"][mode]["routes_mesurees"] = len(sample)
            res["modes"][mode]["duree_routage_s"] = round(time.monotonic() - t0, 1)
        res["meme_noeud_par_couronne"] = {
            c: {"n": n_by_c[c], "meme_noeud": same_by_c[c],
                "part_pct": round(100.0 * same_by_c[c] / n_by_c[c], 1) if n_by_c[c] else None,
                "meme_noeud_lointain": far_same_by_c[c],
                "part_lointain_pct": round(100.0 * far_same_by_c[c] / n_by_c[c], 2) if n_by_c[c] else None}
            for c in sorted(n_by_c)}
        res["meme_noeud_total_pct"] = round(100.0 * sum(same_by_c.values()) / max(len(pairs), 1), 1)
        res["meme_noeud_lointain_total_pct"] = round(100.0 * sum(far_same_by_c.values()) / max(len(pairs), 1), 2)
        res["critere_3_ok"] = all((v["part_lointain_pct"] or 0) <= 0.5 for v in res["meme_noeud_par_couronne"].values())
        res["distance_rabattement_m"] = {"mediane": round(statistics.median(snap_dists), 1),
                                         "p95": round(float(np.percentile(snap_dists, 95)), 1),
                                         "max": round(max(snap_dists), 1)}
        res["routage"] = {"routes": n_routed, "none": n_none,
                          "part_none_pct": round(100.0 * n_none / max(n_routed, 1), 2),
                          "ms_par_route_mediane": round(statistics.median(durations_ms), 1),
                          "ms_par_route_moyenne": round(statistics.fmean(durations_ms), 1),
                          "ms_par_route_p95": round(float(np.percentile(durations_ms, 95)), 1)}
        logger.info("%s: same node %.1f %% (%s); distant (> %d m) %.2f %% (%s) — criterion 3 %s; "
                    "%d routes, %d None, %.1f ms/route (median)", label,
                    res["meme_noeud_total_pct"],
                    {c: v["part_pct"] for c, v in res["meme_noeud_par_couronne"].items()},
                    SAME_NODE_FAR_M, res["meme_noeud_lointain_total_pct"],
                    {c: v["part_lointain_pct"] for c, v in res["meme_noeud_par_couronne"].items()},
                    "tenu" if res["critere_3_ok"] else "NON tenu",
                    n_routed, n_none, res["routage"]["ms_par_route_mediane"])
        out["graphes"][label] = res
        del graphs
    return out


_WORKER_PROBE = r"""
import resource, sys, time, json
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import route_worker
t0 = time.monotonic()
route_worker.init_worker(sys.argv[1], sys.argv[3], sys.argv[4])
load_s = time.monotonic() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mo = rss / (1_048_576 if sys.platform == "darwin" else 1024)
g = route_worker._graphs
print(json.dumps({"ram_pointe_mo": round(rss_mo), "chargement_s": round(load_s, 1),
                  "noeuds": {m: G.number_of_nodes() for m, G in g.items()}}))
"""


def measure_worker_memory(cache_dir: Path) -> dict:
    out = {}
    for label, key in GRAPHS.items():
        proc = subprocess.run([sys.executable, "-c", _WORKER_PROBE, str(LLMAGENTS_PATH),
                               str(Path(__file__).resolve().parent), str(cache_dir), key],
                              capture_output=True, text=True, cwd=str(REPO_ROOT))
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not line:
            logger.error("[ALARME] memory probe %s: code %d — %s", label, proc.returncode, proc.stderr[-600:])
            out[label] = {"erreur": proc.stderr[-600:]}
            continue
        out[label] = {"cle": key, **json.loads(line[-1])}
        logger.info("worker %s: %d MB peak, loading %.1fs", label, out[label]["ram_pointe_mo"],
                    out[label]["chargement_s"])
    return out


def write_trace(mesures: dict, trace_dir: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "mesures.json").write_text(json.dumps(mesures, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"# Mesures O2 / O4 — graphe du polygone contre disque de 30 km — {mesures['date']}", "",
         f"Population : `{mesures['population']}` ({mesures['paires']['n_paires']} paires de planification, "
         f"{mesures['paires']['par_mode']}). Script : `scripts/data/population/measure_osmnx_perimeter_graph.py`.", "",
         "## O2 — paires « même nœud » (repli vitesse, pas un itinéraire) par couronne d'origine", "",
         "| Couronne d'origine | Paires | Disque 30 km | dont > 500 m | Polygone 453 | dont > 500 m |", "|---|---:|---:|---:|---:|---:|"]
    g = mesures["paires"]["graphes"]
    for c, v in g["disque_30km"]["meme_noeud_par_couronne"].items():
        w = g["polygone_453"]["meme_noeud_par_couronne"].get(c, {})
        L.append(f"| {c} | {v['n']} | {v['meme_noeud']} ({v['part_pct']} %) | {v['meme_noeud_lointain']} ({v['part_lointain_pct']} %) | "
                 f"{w.get('meme_noeud')} ({w.get('part_pct')} %) | {w.get('meme_noeud_lointain')} ({w.get('part_lointain_pct')} %) |")
    L.append("")
    L.append(f"Critère 3 (paires « même nœud » distantes de plus de 500 m ≤ 0,5 % par couronne) : "
             f"disque {'tenu' if g['disque_30km']['critere_3_ok'] else 'NON tenu'}, "
             f"polygone {'tenu' if g['polygone_453']['critere_3_ok'] else 'NON tenu'}.")
    L += ["", "## O2 — routage effectif (`_route_sync`, code de production)", "",
          "| Graphe | Routes | None | ms/route médiane | moyenne | p95 | rabattement médian (m) | p95 (m) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in g.items():
        L.append(f"| {label} | {r['routage']['routes']} | {r['routage']['none']} ({r['routage']['part_none_pct']} %) | "
                 f"{r['routage']['ms_par_route_mediane']} | {r['routage']['ms_par_route_moyenne']} | "
                 f"{r['routage']['ms_par_route_p95']} | {r['distance_rabattement_m']['mediane']} | "
                 f"{r['distance_rabattement_m']['p95']} |")
    L += ["", "## O4 — mémoire d'un worker (`route_worker.init_worker`, trois graphes chargés)", "",
          "| Graphe | RAM de pointe (Mo) | Chargement (s) |", "|---|---:|---:|"]
    for label, w in mesures["workers"].items():
        L.append(f"| {label} | {w.get('ram_pointe_mo')} | {w.get('chargement_s')} |")
    L.append("")
    (trace_dir / "README.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("trace archived → %s", trace_dir)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=Path,
                        default=REPO_ROOT / "data" / "population" / "population_1000_PANEL_v3" / "population.json")
    parser.add_argument("--cache-dir", type=Path, default=OSMNX_CACHE_DIR)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--route-sample", action="store_true",
                        help="route only a sample (≈ 400 pairs per mode) instead of all of them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    t0 = time.monotonic()
    for key in GRAPHS.values():
        if not (args.cache_dir / f"graphs_{key}.pkl").exists():
            logger.error("[ALARME] graph missing: %s", args.cache_dir / f"graphs_{key}.pkl")
            return 2
    population = json.loads(args.population.read_text(encoding="utf-8"))
    logger.info("measurements — start: %s (%d personas)", args.population, len(population))
    mesures = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "population": str(args.population), "n_personas": len(population),
               "workers": measure_worker_memory(args.cache_dir),
               "paires": measure_pairs(population, args.cache_dir, route_all=not args.route_sample),
               "duree_s": round(time.monotonic() - t0, 1)}
    if args.trace:
        write_trace(mesures, args.trace)
    print(json.dumps({k: v for k, v in mesures.items() if k != "paires"}, ensure_ascii=False, indent=1))
    for label, r in mesures["paires"]["graphes"].items():
        print(label, "same node:", r["meme_noeud_total_pct"], "%", {c: v["part_pct"] for c, v in r["meme_noeud_par_couronne"].items()},
              "| routing:", r["routage"])
    logger.info("measurements — end in %.1fs", mesures["duree_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
