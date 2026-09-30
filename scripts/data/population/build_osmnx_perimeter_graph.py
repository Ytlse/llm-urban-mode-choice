"""build_osmnx_perimeter_graph.py — The three OSMnx graphs of the 453-municipality polygon.

    services/llm-agents/.venv/bin/python -m scripts.data.population.build_osmnx_perimeter_graph
    services/llm-agents/.venv/bin/python -m scripts.data.population.build_osmnx_perimeter_graph --force --trace docs/traces/<date>_graphe_perimetre
    make osmnx-perimeter-graph

WHY (ticket 031, § 1.4; scope report, action O1). The production routing graph
is a 30 km disc around Toulouse downloaded through Overpass
(`ox.graph_from_address("Toulouse, France", dist=30000)`, key `ecb40f20a303`). The study
scope is the polygon of the 453 municipalities of EMC² 2023: 5,428 km², up to 62 km from the Capitole.
98 of the 154 3rd-ring agents of the sealed population v3 live outside the disc; their
trips snap onto the same graph node and get a fallback speed — these are not
itineraries. A 5,428 km² polygon exceeds what Overpass reasonably serves
(> 100 requests with the default subdivision): the graphs are built here from the regional
OSM pbf files already present in the eqasim fork (`osmium extract --polygon`, then
`graph_from_xml`), without download.

WHAT THE SCRIPT GUARANTEES.
  * **Same networks as production.** The `walk` / `bike` / `drive` filters are those
    of OSMnx itself (`osmnx._overpass._get_network_filter`), read at run time and transcribed
    into Python predicates — not a copy that would diverge at the next version. Walking is
    bidirectional as in OSMnx (`settings.bidirectional_network_types`).
  * **Same speeds as production.** `speed_kph` per road type comes from
    `trip_helper.osmnx_direct._SPEEDS` / `_FALLBACKS` (config/osmnx.yaml), and
    `ox.add_edge_travel_times` sets the travel times — the code of `_GraphStore._build_sync`
    identically.
  * **A distinct cache key.** `graphs_<clé>.pkl` / `boundary_<clé>.pkl` in
    `data/cache/osmnx/`, key = md5(`PERIMETER_GRAPH_LABEL`)[:12] — the label states the scope, the
    version of the municipality table and the date of the pbf files. The 30 km disc stays intact.
  * **Congestion zones are set on the nodes** (`zone`: `city` = municipality of Toulouse,
    `agglo` = rings Toulouse + 1st + 2nd outside Toulouse, `outside` = the rest; ticket 031,
    decision 4): `_route_sync` congests each edge according to the zone of its origin node.
    `boundary_<clé>.pkl` is the municipality of Toulouse, copied from the 30 km disc cache
    (no geocoding over the wire). `--zones-only` (re)sets the zones on an existing pickle.
  * **A log that can be read back.** Duration of each step, nodes and edges per mode, pickle
    size, peak memory, written to stderr and, with `--trace`, into a timestamped folder
    (`mesures.json`, `README.md`). A `graphs_<clé>.meta.json` file carries the provenance
    next to the pickle.

WHAT THE SCRIPT DOES NOT DO. It touches neither the runtime (`osmnx_server.py`, `osmnx_direct.py`,
`geography.py`: part 2 of the ticket) nor the SQLite itinerary cache. It downloads nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
LLMAGENTS_PATH = REPO_ROOT / "services" / "llm-agents"
for _p in (str(REPO_ROOT), str(LLMAGENTS_PATH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("osmnx.perimetre")

# ── Graph identity ────────────────────────────────────────────────────────────
# The label states what the graph covers and where it comes from; the key derives from it.
# Changing the pbf (vintage), the municipality table (version `cc1`) or the scope changes the
# key — and hence the cache — instead of serving an old graph again under a new name.
# Defined in `services/llm-agents/geography.py` since part 2 of ticket 031: the runtime
# (`osmnx_server`, `osmnx_direct`) serves that graph, a single definition of the key.
from geography import PERIMETER_CACHE_KEY, PERIMETER_GRAPH_LABEL  # noqa: E402
from geography import PRODUCTION_CACHE_KEY_30KM as PRODUCTION_CACHE_KEY  # noqa: E402  (disque de 30 km : frontière réutilisée)

COURONNE_GEOJSON = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "couronne_perimetre.geojson"
COMMUNE_TABLE = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "commune_couronne.json"
OSM_PBF_SOURCES = [
    REPO_ROOT / "services" / "eqasim-toulouse" / "data" / "osm_toulouse" / "midi-pyrenees-220101.osm.pbf",
    REPO_ROOT / "services" / "eqasim-toulouse" / "data" / "osm_toulouse" / "languedoc-roussillon-220101.osm.pbf",
]
OSMNX_CACHE_DIR = REPO_ROOT / "data" / "cache" / "osmnx"
WORK_DIR = OSMNX_CACHE_DIR / "perimetre_453"      # intermediate extracts (pbf, xml)
OSMIUM = shutil.which("osmium") or "/opt/homebrew/bin/osmium"

OSMNX_MODES = ("walk", "bike", "drive")


# ── OSMnx network filters → Python predicates ─────────────────────────────────

_FILTER_RE = re.compile(r'\["([^"]+)"(?:(!~|~|!=|=)"([^"]*)")?\]')


def parse_overpass_filter(way_filter: str) -> list[tuple[str, Optional[str], Optional[str]]]:
    """`["highway"]["area"!~"yes"]…` → [("highway", None, None), ("area", "!~", "yes"), …].

    Rejects a string it cannot read entirely: a partially understood filter
    would let through roads that production excludes, silently.
    """
    clauses = _FILTER_RE.findall(way_filter)
    rebuilt = "".join(f'["{k}"]' if not op else f'["{k}"{op}"{v}"]' for k, op, v in clauses)
    if rebuilt != way_filter:
        raise ValueError(f"Overpass filter not entirely parsed:\n  {way_filter}\n  {rebuilt}")
    return [(k, op or None, v if op else None) for k, op, v in clauses]


def way_predicate(way_filter: str) -> Callable[[dict], bool]:
    """Predicate "this road passes the Overpass filter" on the tag dict of a way.

    Overpass semantics: `["k"]` → the key exists; `["k"!~"re"]` → the key is absent or its
    value does not contain the pattern (`re.search`, unanchored, as Overpass); `["k"~"re"]` →
    present and the pattern is found; `["k"="v"]` / `["k"!="v"]` → strict equality.
    """
    clauses = parse_overpass_filter(way_filter)
    compiled = [(k, op, re.compile(v) if op in ("~", "!~") else v) for k, op, v in clauses]

    def _ok(tags: dict) -> bool:
        for k, op, v in compiled:
            val = tags.get(k)
            if op is None:
                if val is None:
                    return False
            elif op == "!~":
                if val is not None and v.search(str(val)):
                    return False
            elif op == "~":
                if val is None or not v.search(str(val)):
                    return False
            elif op == "=":
                if val is None or str(val) != v:
                    return False
            elif op == "!=":
                if val is not None and str(val) == v:
                    return False
        return True

    return _ok


def network_filters() -> dict[str, str]:
    """The OSMnx Overpass filters for the three modes, read from the installed version."""
    from osmnx._overpass import _get_network_filter
    return {mode: _get_network_filter(mode) for mode in OSMNX_MODES}


# ── Steps ─────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> float:
    t0 = time.monotonic()
    logger.info("%s — start: %s", label, " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        logger.error("[ALARME] %s failed (code %d) after %.1fs: %s", label, proc.returncode, dt,
                     (proc.stderr or proc.stdout).strip()[-800:])
        raise RuntimeError(f"{label}: code {proc.returncode}")
    logger.info("%s — end in %.1fs", label, dt)
    return dt


def _mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576 if path.exists() else 0.0


def perimeter_polygon(geojson: Path, out: Path) -> dict:
    """Union of the four rings → a single-geometry GeoJSON, for `osmium extract -p`."""
    import geopandas as gpd

    g = gpd.read_file(geojson)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    union = g.union_all() if hasattr(g, "union_all") else g.unary_union
    area_km2 = gpd.GeoSeries([union], crs=4326).to_crs(2154).area.iloc[0] / 1e6
    out.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"name": ["perimetre_453_communes"]}, geometry=[union], crs=4326).to_file(
        out, driver="GeoJSON")
    bounds = [round(b, 5) for b in union.bounds]
    logger.info("scope polygon: %s, %.0f km², extent %s", union.geom_type, area_km2, bounds)
    return {"geom_type": union.geom_type, "area_km2": round(area_km2, 1), "bounds_wgs84": bounds,
            "n_couronnes": int(len(g))}


def extract_highways_xml(polygon: Path, sources: list[Path], work: Path, force: bool) -> tuple[Path, dict]:
    """regional pbf → extract by polygon → merge → `highway` ways as `.osm` XML."""
    work.mkdir(parents=True, exist_ok=True)
    xml_path = work / "perimetre_453_highway.osm"
    journal: dict = {"etapes_s": {}, "tailles_mo": {}}
    if xml_path.exists() and not force:
        logger.info("XML extract already present (%.0f MB): %s — --force to redo it", _mb(xml_path), xml_path)
        journal["reutilise"] = True
        journal["tailles_mo"]["highway.osm"] = round(_mb(xml_path), 1)
        return xml_path, journal
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError(f"OSM pbf missing: {missing}")
    extracts = []
    for src in sources:
        dst = work / f"ext_{src.stem}.pbf"
        journal["etapes_s"][f"extract {src.name}"] = round(
            _run([OSMIUM, "extract", "-p", str(polygon), "-s", "smart", "--overwrite", "-o", str(dst), str(src)],
                 f"osmium extract {src.name}"), 1)
        journal["tailles_mo"][dst.name] = round(_mb(dst), 1)
        extracts.append(dst)
    merged = work / "perimetre_453.osm.pbf"
    journal["etapes_s"]["merge"] = round(
        _run([OSMIUM, "merge", "--overwrite", "-o", str(merged), *map(str, extracts)], "osmium merge"), 1)
    journal["tailles_mo"][merged.name] = round(_mb(merged), 1)
    journal["etapes_s"]["tags-filter highway → xml"] = round(
        _run([OSMIUM, "tags-filter", "--overwrite", "-o", str(xml_path), str(merged), "w/highway"],
             "osmium tags-filter w/highway"), 1)
    journal["tailles_mo"]["highway.osm"] = round(_mb(xml_path), 1)
    return xml_path, journal


def build_graphs(xml_path: Path, speeds: dict, fallbacks: dict) -> tuple[dict, dict]:
    """One graph per mode, with the production filters and speeds."""
    import osmnx as ox
    from osmnx import settings as ox_settings
    from osmnx._osm_xml import _overpass_json_from_xml
    from osmnx.graph import _create_graph

    t0 = time.monotonic()
    response = _overpass_json_from_xml(xml_path, "utf-8")
    elements = response["elements"]
    nodes = [e for e in elements if e["type"] == "node"]
    ways = [e for e in elements if e["type"] == "way"]
    node_by_id = {n["id"]: n for n in nodes}
    journal: dict = {"xml_lecture_s": round(time.monotonic() - t0, 1),
                     "noeuds_xml": len(nodes), "voies_highway_xml": len(ways), "modes": {}}
    logger.info("XML read in %.1fs: %d nodes, %d highway ways", journal["xml_lecture_s"], len(nodes), len(ways))

    filters = network_filters()
    graphs: dict = {}
    for mode in OSMNX_MODES:
        t1 = time.monotonic()
        keep = way_predicate(filters[mode])
        mode_ways = [w for w in ways if keep(w.get("tags", {}))]
        needed = {nid for w in mode_ways for nid in w["nodes"]}
        mode_nodes = [node_by_id[nid] for nid in needed if nid in node_by_id]
        bidirectional = mode in ox_settings.bidirectional_network_types
        G = _create_graph([{"elements": mode_nodes + mode_ways}], bidirectional)
        n_raw, e_raw = G.number_of_nodes(), G.number_of_edges()
        G = ox.truncate.largest_component(G, strongly=False)
        G = ox.simplification.simplify_graph(G)
        # Production speeds, identical to `_GraphStore._build_sync`.
        n_fallback = 0
        for _, _, _, data in G.edges(keys=True, data=True):
            hwy = data.get("highway")
            if isinstance(hwy, list):
                hwy = hwy[0]
            if hwy in speeds[mode]:
                data["speed_kph"] = speeds[mode][hwy]
            else:
                data["speed_kph"] = fallbacks[mode]
                n_fallback += 1
        G = ox.add_edge_travel_times(G)
        graphs[mode] = G
        journal["modes"][mode] = {
            "filtre_overpass": filters[mode], "bidirectionnel": bidirectional,
            "voies_retenues": len(mode_ways), "noeuds_bruts": n_raw, "aretes_brutes": e_raw,
            "noeuds": G.number_of_nodes(), "aretes": G.number_of_edges(),
            "aretes_vitesse_repli": n_fallback,
            "part_aretes_vitesse_repli_pct": round(100.0 * n_fallback / max(G.number_of_edges(), 1), 1),
            "duree_s": round(time.monotonic() - t1, 1),
        }
        logger.info("graph %-5s: %d ways → %d nodes / %d edges (simplified; raw %d / %d), "
                    "%d edges at fallback speed (%.1f %%), %.1fs", mode, len(mode_ways),
                    G.number_of_nodes(), G.number_of_edges(), n_raw, e_raw, n_fallback,
                    journal["modes"][mode]["part_aretes_vitesse_repli_pct"], time.monotonic() - t1)
    journal["construction_s"] = round(time.monotonic() - t0, 1)
    return graphs, journal


def respeed_graphs(graphs: dict, speeds: dict, fallbacks: dict) -> dict:
    """Resets `speed_kph` and `travel_time` of each edge from the current config; log per mode.

    Used when `config/osmnx.yaml` changes (ticket 031 part 2, action O3: bike speeds for
    `track`, `service`, `trunk`, `*_link`…): the pickle carries the speeds of its build, the
    config alone is not enough. Rebuilds nothing else — nodes, edges, zones stay.
    """
    import osmnx as ox
    from collections import Counter

    journal: dict = {}
    for mode, G in graphs.items():
        t1 = time.monotonic()
        n_fallback = n_changed = 0
        en_repli: Counter = Counter()
        for _, _, _, data in G.edges(keys=True, data=True):
            hwy = data.get("highway")
            if isinstance(hwy, list):
                hwy = hwy[0]
            new = speeds[mode].get(hwy)
            if new is None:
                new = fallbacks[mode]
                n_fallback += 1
                en_repli[str(hwy)] += 1
            if data.get("speed_kph") != new:
                n_changed += 1
            data["speed_kph"] = new
        ox.add_edge_travel_times(G)
        journal[mode] = {
            "aretes": G.number_of_edges(), "aretes_modifiees": n_changed,
            "aretes_vitesse_repli": n_fallback,
            "part_aretes_vitesse_repli_pct": round(100.0 * n_fallback / max(G.number_of_edges(), 1), 1),
            "types_en_repli": dict(en_repli.most_common(10)), "duree_s": round(time.monotonic() - t1, 1),
        }
        logger.info("speeds reset on %-5s: %d edges changed / %d, %d at fallback (%.1f %%): %s — %.1fs",
                    mode, n_changed, G.number_of_edges(), n_fallback,
                    journal[mode]["part_aretes_vitesse_repli_pct"], dict(en_repli.most_common(6)),
                    time.monotonic() - t1)
    return journal


def city_geometry(cache_dir: Path):
    """The municipality of Toulouse (geocoded boundary of the production graph)."""
    b_path = cache_dir / f"boundary_{PRODUCTION_CACHE_KEY}.pkl"
    if not b_path.exists():
        raise FileNotFoundError(f"Toulouse boundary missing: {b_path}")
    with b_path.open("rb") as fh:
        return pickle.load(fh).geometry.iloc[0]


def assign_zones(graphs: dict, cache_dir: Path) -> dict:
    """Congestion zones of the nodes (ticket 031, decision 4): city / urban area / outside."""
    from trip_helper.congestion_zones import agglo_polygon, assign_node_zones

    city = city_geometry(cache_dir)
    agglo = agglo_polygon()
    journal = {}
    for mode, G in graphs.items():
        t0 = time.monotonic()
        counts = assign_node_zones(G, city, agglo)
        journal[mode] = {**dict(counts), "duree_s": round(time.monotonic() - t0, 1)}
        logger.info("congestion zones %-5s: %s (%.1fs)", mode, dict(counts), time.monotonic() - t0)
    return journal


def production_speeds() -> tuple[dict, dict]:
    """Production `_SPEEDS` / `_FALLBACKS` — imported, never copied."""
    from trip_helper.osmnx_direct import _FALLBACKS, _SPEEDS
    return _SPEEDS, _FALLBACKS


def write_cache(graphs: dict, cache_dir: Path, key: str, force: bool) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    g_path = cache_dir / f"graphs_{key}.pkl"
    b_path = cache_dir / f"boundary_{key}.pkl"
    prod_boundary = cache_dir / f"boundary_{PRODUCTION_CACHE_KEY}.pkl"
    if g_path.exists() and not force:
        raise FileExistsError(f"{g_path} already exists — --force to rewrite it")
    t0 = time.monotonic()
    with g_path.open("wb") as fh:
        pickle.dump(graphs, fh)
    pickle_s = time.monotonic() - t0
    if not b_path.exists() or force:
        if not prod_boundary.exists():
            raise FileNotFoundError(
                f"Toulouse boundary missing: {prod_boundary}. It comes from the cache of the production "
                "graph (Nominatim geocoding of \"Toulouse, France\"); nothing is downloaded here.")
        shutil.copy2(prod_boundary, b_path)
    logger.info("pickle written: %s (%.0f MB, %.1fs); boundary: %s", g_path.name, _mb(g_path), pickle_s, b_path.name)
    return {"graphs_pkl": str(g_path), "graphs_pkl_mo": round(_mb(g_path), 1), "pickle_s": round(pickle_s, 1),
            "boundary_pkl": str(b_path), "boundary_source": str(prod_boundary)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_trace(mesures: dict, trace_dir: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "mesures.json").write_text(json.dumps(mesures, ensure_ascii=False, indent=1), encoding="utf-8")
    modes = mesures["graphes"]["modes"]
    lines = [f"# Graphes OSMnx du polygone des 453 communes — {mesures['date']}", "",
             f"Produit par `scripts/data/population/build_osmnx_perimeter_graph.py` (ticket 031 § 1.4, action O1).",
             f"Clé de cache `{mesures['cache_key']}` (label `{mesures['label']}`), pickle "
             f"{mesures['cache']['graphs_pkl_mo']} Mo, polygone {mesures['polygone']['area_km2']} km².", "",
             "| Mode | Voies retenues | Nœuds | Arêtes | Arêtes en vitesse de repli | Durée |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode, m in modes.items():
        lines.append(f"| {mode} | {m['voies_retenues']} | {m['noeuds']} | {m['aretes']} | "
                     f"{m['aretes_vitesse_repli']} ({m['part_aretes_vitesse_repli_pct']} %) | {m['duree_s']} s |")
    lines += ["", f"Mémoire de pointe du processus de construction : {mesures['ram_pointe_mo']} Mo ; "
              f"durée totale {mesures['duree_totale_s']} s.", "",
              "Les filtres réseau sont ceux d'OSMnx " + mesures["osmnx_version"] + " (voir `mesures.json`).", ""]
    (trace_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("trace archived → %s", trace_dir)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="redo the extract and rewrite the pickle")
    parser.add_argument("--zones-only", action="store_true",
                        help="only (re)set the congestion zones on the existing pickle, without rebuilding")
    parser.add_argument("--respeed", action="store_true",
                        help="reset the speeds of config/osmnx.yaml (speed_kph, travel_time) on the existing "
                             "pickle, without rebuilding — after any change to `speeds`")
    parser.add_argument("--trace", type=Path, default=None, help="timestamped trace folder (docs/traces/…)")
    parser.add_argument("--cache-dir", type=Path, default=OSMNX_CACHE_DIR)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    import osmnx as ox
    t0 = time.monotonic()
    logger.info("scope graph — start: label %s, key %s, osmnx %s", PERIMETER_GRAPH_LABEL,
                PERIMETER_CACHE_KEY, ox.__version__)
    if not Path(OSMIUM).exists():
        logger.error("[ALARME] osmium introuvable (%s) — `brew install osmium-tool`", OSMIUM)
        return 2
    g_path = args.cache_dir / f"graphs_{PERIMETER_CACHE_KEY}.pkl"
    if args.zones_only:
        if not g_path.exists():
            logger.error("[ALARME] --zones-only: no pickle %s", g_path)
            return 2
        with g_path.open("rb") as fh:
            graphs = pickle.load(fh)
        zones = assign_zones(graphs, args.cache_dir)
        with g_path.open("wb") as fh:
            pickle.dump(graphs, fh)
        meta_path = args.cache_dir / f"graphs_{PERIMETER_CACHE_KEY}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.setdefault("graphes", {})["zones_congestion"] = zones
        meta["zones_posees_le"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("zones set and pickle rewritten (%.0f MB) in %.1fs", _mb(g_path), time.monotonic() - t0)
        return 0
    if args.respeed:
        if not g_path.exists():
            logger.error("[ALARME] --respeed: no pickle %s", g_path)
            return 2
        with g_path.open("rb") as fh:
            graphs = pickle.load(fh)
        speeds, fallbacks = production_speeds()
        journal = respeed_graphs(graphs, speeds, fallbacks)
        tmp = g_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(graphs, fh)
        tmp.replace(g_path)
        meta_path = args.cache_dir / f"graphs_{PERIMETER_CACHE_KEY}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        for mode, j in journal.items():
            meta.setdefault("graphes", {}).setdefault("modes", {}).setdefault(mode, {}).update(
                {"aretes_vitesse_repli": j["aretes_vitesse_repli"],
                 "part_aretes_vitesse_repli_pct": j["part_aretes_vitesse_repli_pct"]})
        meta["vitesses"] = {"source": "trip_helper.osmnx_direct._SPEEDS / _FALLBACKS (config/osmnx.yaml)",
                            "speeds_kph": speeds, "fallbacks_kph": fallbacks, "reposees_le":
                            datetime.now(timezone.utc).isoformat(timespec="seconds"), "journal": journal}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        ram_mo = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1_048_576 if sys.platform == "darwin" else 1024)
        logger.info("speeds reset and pickle rewritten (%.0f MB) in %.1fs; peak RAM %d MB",
                    _mb(g_path), time.monotonic() - t0, round(ram_mo))
        if args.trace:
            args.trace.mkdir(parents=True, exist_ok=True)
            (args.trace / "respeed.json").write_text(json.dumps(
                {"date": meta["vitesses"]["reposees_le"], "cache_key": PERIMETER_CACHE_KEY, "journal": journal,
                 "speeds_kph": speeds, "fallbacks_kph": fallbacks, "duree_s": round(time.monotonic() - t0, 1),
                 "ram_pointe_mo": round(ram_mo)}, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    if g_path.exists() and not args.force:
        logger.info("graphs already cached: %s (%.0f MB) — --force to rebuild", g_path, _mb(g_path))
        return 0

    polygon_path = args.work_dir / "perimetre_453_communes.geojson"
    poly = perimeter_polygon(COURONNE_GEOJSON, polygon_path)
    xml_path, extract_journal = extract_highways_xml(polygon_path, OSM_PBF_SOURCES, args.work_dir, args.force)
    speeds, fallbacks = production_speeds()
    graphs, build_journal = build_graphs(xml_path, speeds, fallbacks)
    build_journal["zones_congestion"] = assign_zones(graphs, args.cache_dir)
    cache_journal = write_cache(graphs, args.cache_dir, PERIMETER_CACHE_KEY, args.force)

    ram_mo = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1_048_576 if sys.platform == "darwin" else 1024)
    mesures = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": PERIMETER_GRAPH_LABEL, "cache_key": PERIMETER_CACHE_KEY, "osmnx_version": ox.__version__,
        "polygone": {**poly, "source": str(COURONNE_GEOJSON), "communes": str(COMMUNE_TABLE)},
        "sources_pbf": [{"fichier": str(p), "mo": round(_mb(p), 1)} for p in OSM_PBF_SOURCES],
        "extraction": extract_journal, "graphes": build_journal, "cache": cache_journal,
        "vitesses": {"source": "trip_helper.osmnx_direct._SPEEDS / _FALLBACKS (config/osmnx.yaml)",
                     "fallbacks_kph": fallbacks},
        "ram_pointe_mo": round(ram_mo), "duree_totale_s": round(time.monotonic() - t0, 1),
    }
    (args.cache_dir / f"graphs_{PERIMETER_CACHE_KEY}.meta.json").write_text(
        json.dumps(mesures, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.trace:
        write_trace(mesures, args.trace)
    logger.info("scope graph — end in %.1fs: %s; peak RAM %d MB", mesures["duree_totale_s"],
                ", ".join(f"{m} {j['noeuds']} nœuds / {j['aretes']} arêtes" for m, j in build_journal["modes"].items()),
                round(ram_mo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
