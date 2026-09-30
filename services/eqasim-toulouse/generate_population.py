"""
Wrapper script for the eqasim Docker service.

Reads population size from APP_CONFIG_PATH (same YAML as the controller) or from the
EQASIM_POPULATION_SIZE env var, computes the synpp sampling_rate, writes a temporary
config, and runs the synpp pipeline.

Cache: if a population JSON already exists in /eqasim-output with enough people
(>= the requested population_size), synpp is skipped entirely.

Expected volumes in docker-compose:
  /eqasim-data   → raw input data (INSEE, OSM, GTFS, BAN, BDTOPO)
  /eqasim-cache  → intermediate pipeline cache (warm on re-runs)
  /eqasim-output → output JSON consumed by the controller
"""

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile

import yaml

# Approximate population of Haute-Garonne (dept 31), used as fallback when no bbox is given.
TOULOUSE_DEPT_POPULATION = 1_400_000

# Départements of the EMC² 2023 survey study area: SIX (ticket 031, option A). It is the default
# of the code; the deployment can restrict it through `EQASIM_DEPARTMENTS` (docker-compose.yml)
# or through the request body. The Haute-Garonne version of ticket 026 passes ["31"], for lack of
# the BD TOPO and BAN data of the five other départements — limit quantified in
# docs/arch/perimetre-population.md (limit no. 6): the 3rd ring caps at 10.6% of the
# population where the survey counts 15.4%.
DEPARTMENTS = ["31", "32", "81", "82", "09", "11"]

# Département data expected by synpp for EACH département requested. The pipeline
# checks them before launching synpp: without it, the assertion of `data/bdtopo/raw.py` falls
# after ten minutes of processing and without saying which file is missing.
BDTOPO_URL = "https://geoservices.ign.fr/bdtopo"
BAN_URL = "https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/adresses-{dep}.csv.gz"

# Safety margin over the effective zone population to absorb IPF rounding
SAMPLING_MARGIN = 1.15

OUTPUT_DIR = "/eqasim-output"
OUTPUT_PREFIX = "toulouse_"

# BASE configuration of the pipeline: `config_toulouse.yml`, mounted in the container. It is the
# only source of the scientific settings (HTS matching: `filter_hts`, `matching_attributes`,
# `matching_minimum_observations`; donor days: `hts_school_days_only`,
# `hts_exclude_wednesday_under_age`). Until 2026-09-03 the wrapper built its own
# config WITHOUT these keys: synpp fell back on its defaults — `filter_hts: True`, that is 308 ENTD
# donors residing in Haute-Garonne for 12,000 people to match, and a degradation that
# dropped the age class (`matching_minimum_observations` 20) — while
# `config_toulouse.yml` (ticket 008, A1.a) said the opposite. The missing file is an error,
# not a return to the defaults.
BASE_CONFIG_PATH = os.environ.get("EQASIM_BASE_CONFIG", "/eqasim/config_toulouse.yml")
# Settings of the base that the wrapper REPLACES (paths and execution parameters of the container).
RUNTIME_OVERRIDDEN_KEYS = (
    "processes", "sampling_rate", "random_seed", "data_path", "output_path", "output_prefix",
    "java_memory", "regions", "departments", "communes", "communes_file", "gtfs_path", "osm_path",
    "ban_path", "bdtopo_path", "generate_personality_traits",
)
# Scientific settings reread from the base and logged at each generation.
SCIENTIFIC_KEYS = (
    "hts", "filter_hts", "matching_attributes", "matching_minimum_observations",
    "matching_age_boundaries", "hts_school_days_only", "hts_exclude_wednesday_under_age",
    "census_undefined_reweighting", "mode_choice",
)


def load_base_config(path: str = BASE_CONFIG_PATH) -> dict:
    """`config:` section of `config_toulouse.yml`. Raises if the file is missing: without it, synpp
    would match on its defaults (308 donors) silently."""
    if not os.path.isfile(path):
        print(f"[eqasim] ERROR [ALARME] configuration de base introuvable : {path} — monter "
              "eqasim-toulouse/config_toulouse.yml dans le conteneur (docker-compose.yml, service "
              "eqasim) ou pointer EQASIM_BASE_CONFIG. Rien n'est généré.")
        raise SystemExit(5)
    with open(path, encoding="utf-8") as f:
        base = (yaml.safe_load(f) or {}).get("config") or {}
    missing = [k for k in SCIENTIFIC_KEYS if k not in base]
    if missing:
        print(f"[eqasim] ERROR [ALARME] {path} ne fixe pas {missing} : ces réglages ne doivent pas "
              "retomber sur les défauts de synpp. Rien n'est généré.")
        raise SystemExit(5)
    return base


def _communes_from_bbox(
    bbox: list[float],
    data_path: str = "/eqasim-data",
) -> tuple[list[str], int]:
    """
    Return (commune_ids, total_population) for communes intersecting the given
    WGS84 bbox [min_lon, min_lat, max_lon, max_lat].

    Uses the IRIS shapefile and INSEE population CSV already present in the
    eqasim data volume — no network call required.
    """
    import geopandas as gpd
    import pandas as pd
    import py7zr
    from pyproj import Transformer
    from shapely.geometry import box

    min_lon, min_lat, max_lon, max_lat = bbox

    # Convert bbox to Lambert-93 (EPSG:2154), the CRS of IRIS shapes.
    t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x_min, y_min = t.transform(min_lon, min_lat)
    x_max, y_max = t.transform(max_lon, max_lat)
    bbox_geom = box(x_min, y_min, x_max, y_max)

    # Extract IRIS shapes from the 7z archive into a temp dir.
    iris_dir = os.path.join(data_path, "iris_2024")
    candidates = sorted(glob.glob(os.path.join(iris_dir, "*.7z")))
    if not candidates:
        print(f"[eqasim] Warning: no IRIS archive in {iris_dir}, skipping bbox filter")
        return [], 0

    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(candidates[0]) as archive:
            names = [n for n in archive.getnames() if "LAMB93" in n]
            archive.extract(tmp, names)

        gpkg_files = [n for n in names if n.endswith(".gpkg")]
        if not gpkg_files:
            print("[eqasim] Warning: no .gpkg inside IRIS archive, skipping bbox filter")
            return [], 0

        df_iris = gpd.read_file(
            os.path.join(tmp, gpkg_files[0]),
            dtype={"code_iris": str, "code_insee": str},
        )[["code_insee", "geometry"]].rename(columns={"code_insee": "commune_id"})

    # Dissolve to commune level then spatial-join with the bbox polygon.
    df_communes = df_iris.dissolve("commune_id").reset_index()
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:2154")
    joined = gpd.sjoin(df_communes, bbox_gdf, how="inner", predicate="intersects")
    communes = sorted(joined["commune_id"].unique().tolist())

    # Sum population for those communes from the INSEE population CSV.
    pop_path = os.path.join(data_path, "rp_2022", "base-ic-evol-struct-pop-2022_csv.zip")
    total_pop = 0
    if os.path.exists(pop_path):
        with zipfile.ZipFile(pop_path) as z:
            with z.open("base-ic-evol-struct-pop-2022.CSV") as f:
                df_pop = pd.read_csv(f, sep=";", usecols=["COM", "P22_POP"], dtype={"COM": str})
        total_pop = int(df_pop[df_pop["COM"].isin(communes)]["P22_POP"].sum())

    print(f"[eqasim] bbox filter: {len(communes)} communes, effective population={total_pop}")
    return communes, total_pop


def _perimeter_communes(departments: list[str] | None) -> list[str]:
    """Sampling frame: the communes of the EMC² survey study area (ticket 026).

    The list comes from `mobility_core/data/commune_couronne.json`, produced by
    `make communes-couronnes` from the GIS layer of the survey — 453 communes over six
    départements. `departments` restricts it: the Haute-Garonne version of ticket 026
    passes `["31"]` and gets 346 communes.

    ⚠ **The frame is not the study area.** Restricting to the communes of 31 is a choice of
    division of the work, quantified and published (perimetre-population.md, limit no. 6): it
    caps the 3rd ring at 10.6% of the population where the survey counts 15.4%.
    The admission filter at loading, for its part, stays on the 453.

    Raises if the list is empty: without this safeguard, a typo would silently fall back
    on the whole département, and one would believe to have a compliant frame.
    """
    # `mobility_core` since the split of ticket 037 (2026-09-07), which broke
    # `llm_module` into three packages. The import had stayed on the old name: the service
    # answered HTTP 500 « No module named 'llm_module' » to any generation requesting a
    # study area — that is, to all of them. The package is mounted in the image (`/eqasim/mobility_core`),
    # only the import name was missing.
    from mobility_core.residence_zone import CommuneTable

    table = CommuneTable.load()
    communes = table.communes(departments)
    counts = table.counts(departments)
    print(f"[eqasim] cadre de tirage : périmètre EMC² restreint à "
          f"{departments or 'tous les départements'} → {len(communes)} communes "
          f"({', '.join(f'{k} {v}' for k, v in counts.items())})")
    return communes


def check_department_data(departments: list[str], data_path: str = "/eqasim-data",
                          bdtopo_path: str = "bdtopo_toulouse", ban_path: str = "ban_toulouse") -> list[str]:
    """Returns the list of MISSING département data (empty = everything is there).

    BD TOPO: a `.7z` archive or a `BDTOPO_*_D0<dep>_*` folder (unpacked IGN delivery)
    in `bdtopo_path`; BAN: `adresses-<dep>.csv.gz` in `ban_path`. No download
    here — the decision to obtain 1 to 2 GB of BD TOPO per département belongs to the author of the
    repository (ticket 031, § 1.0).
    """
    missing: list[str] = []
    bdtopo_dir = os.path.join(data_path, bdtopo_path)
    ban_dir = os.path.join(data_path, ban_path)
    for dep in departments:
        code = str(dep).zfill(2)
        tag = f"D0{code}" if len(code) == 2 else f"D{code}"
        has_bdtopo = any(tag in name for name in os.listdir(bdtopo_dir)) if os.path.isdir(bdtopo_dir) else False
        if not has_bdtopo:
            missing.append(f"BD TOPO {tag} attendue dans {bdtopo_dir} (édition alignée sur D031 "
                           f"2024-09-15 ; {BDTOPO_URL})")
        ban_file = os.path.join(ban_dir, f"adresses-{code}.csv.gz")
        if not os.path.isfile(ban_file):
            missing.append(f"BAN {ban_file} ({BAN_URL.format(dep=code)})")
    return missing


def _population_of(communes: list[str], data_path: str = "/eqasim-data") -> int:
    """RP 2022 population of the communes kept, for the `sampling_rate`."""
    import pandas as pd

    pop_path = os.path.join(data_path, "rp_2022", "base-ic-evol-struct-pop-2022_csv.zip")
    if not os.path.exists(pop_path):
        print(f"[eqasim] Warning: {pop_path} absent — population effective inconnue")
        return 0
    with zipfile.ZipFile(pop_path) as z:
        with z.open("base-ic-evol-struct-pop-2022.CSV") as f:
            df = pd.read_csv(f, sep=";", usecols=["COM", "P22_POP"], dtype={"COM": str})
    return int(df[df["COM"].isin(communes)]["P22_POP"].sum())


def _communes_cache_prefix() -> str:
    """Return the standard output prefix regardless of commune subset."""
    return OUTPUT_PREFIX


def _read_population_size_from_config(config_path: str) -> int | None:
    """Extract population_size from the controller YAML config if present."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        data = cfg.get("data", {})
        return int(data["population_size"]) if "population_size" in data else None
    except Exception:
        return None


def _find_cached_file(population_size: int, prefix: str = OUTPUT_PREFIX) -> str | None:
    """Return the exact population JSON for the requested size, or None."""
    path = os.path.join(OUTPUT_DIR, f"{prefix}population_{population_size}.json")
    return path if os.path.isfile(path) else None


def run(
    population_size: int | None = None,
    generate_personality: bool = False,
    force: bool = False,
    bbox: list[float] | None = None,
    perimeter: bool | None = None,
    departments: list[str] | None = None,
) -> str | None:
    """
    Generate the population JSON.  Returns the output file path on cache-hit or after
    successful generation.  Raises SystemExit on synpp failure (non-zero exit code).

    bbox: optional [min_lon, min_lat, max_lon, max_lat] in WGS84.  When provided,
    synpp is restricted to the communes that intersect the bbox, and the sampling_rate
    is derived from their actual population instead of the full département.

    perimeter: when true (or EQASIM_PERIMETER=true), the sampling frame is the EMC² 2023
    survey perimeter itself — a LIST OF COMMUNES, not a rectangle (ticket 026).  Takes
    precedence over bbox: a rectangle cannot express "the survey perimeter, no more no
    less".  departments restricts that frame; it defaults to EQASIM_DEPARTMENTS or the
    départements listed in the synpp config (today ["31"], the Haute-Garonne version).
    """
    # ── Resolve population size ────────────────────────────────────────────────
    if population_size is None:
        env_size = os.environ.get("EQASIM_POPULATION_SIZE")
        if env_size:
            population_size = int(env_size)

    if population_size is None:
        app_config_path = os.environ.get("APP_CONFIG_PATH")
        if app_config_path:
            population_size = _read_population_size_from_config(app_config_path)

    if population_size is None:
        population_size = 1000
        print(
            f"[eqasim] EQASIM_POPULATION_SIZE not set and APP_CONFIG_PATH not found "
            f"— defaulting to {population_size} agents."
        )

    if not force:
        force = os.environ.get("EQASIM_FORCE_REGENERATE", "false").lower() == "true"

    print(f"[eqasim] population_size={population_size}  generate_personality={generate_personality}  bbox={bbox}")

    # ── Resolve sampling frame ─────────────────────────────────────────────────
    communes: list[str] = []
    effective_population = TOULOUSE_DEPT_POPULATION
    output_prefix = OUTPUT_PREFIX

    if perimeter is None:
        perimeter = os.environ.get("EQASIM_PERIMETER", "false").lower() == "true"
    if departments is None:
        env_dep = os.environ.get("EQASIM_DEPARTMENTS", "")
        departments = [d.strip() for d in env_dep.split(",") if d.strip()] or DEPARTMENTS

    if perimeter:
        # The survey study area is a LIST OF COMMUNES, not a rectangle: it is the
        # only way to say « neither more nor less » (ticket 026). It therefore takes precedence over bbox.
        if bbox is not None:
            print("[eqasim] perimeter=true — la bbox est ignorée : un rectangle ne peut "
                  "pas exprimer le périmètre d'enquête")
        communes = _perimeter_communes(departments)
        effective_population = _population_of(communes)
        print(f"[eqasim] population effective du cadre : {effective_population:,} hab "
              f"(RP 2022)")
        output_prefix = _communes_cache_prefix()
    elif bbox is not None:
        communes, effective_population = _communes_from_bbox(bbox)
        if communes:
            output_prefix = _communes_cache_prefix()
        else:
            print("[eqasim] bbox produced no communes — falling back to full département")

    # ── Cache check ────────────────────────────────────────────────────────────
    cached = _find_cached_file(population_size, prefix=output_prefix)
    if cached and not force:
        print(f"[eqasim] Cache hit — using existing file: {cached}  (skipping synpp)")
        return cached
    if cached and force:
        print(f"[eqasim] EQASIM_FORCE_REGENERATE=true — ignoring cached file: {cached}")

    # ── Département data: all or nothing ─────────────────────────────────
    # A département without BD TOPO or BAN is not « skipped »: the frame would be cut down without
    # the population saying so. We stop BEFORE synpp, with the list of what is missing.
    missing_data = check_department_data(departments)
    if missing_data:
        print(f"[eqasim] ERROR [ALARME] données départementales manquantes pour "
              f"{departments} — génération refusée :")
        for item in missing_data:
            print(f"[eqasim]   - {item}")
        raise SystemExit(3)

    # ── Build synpp config ─────────────────────────────────────────────────────
    if effective_population <= 0:
        effective_population = TOULOUSE_DEPT_POPULATION
    sampling_rate = min(1.0, (population_size * SAMPLING_MARGIN) / effective_population)
    print(
        f"[eqasim] Cache miss — running synpp "
        f"(sampling_rate={sampling_rate:.6f}, effective_population={effective_population})"
    )

    base_config = load_base_config()
    runtime_config = {
        "processes": int(os.environ.get("EQASIM_PROCESSES", "4")),
        "sampling_rate": round(sampling_rate, 8),
        "random_seed": int(os.environ.get("EQASIM_RANDOM_SEED", "1234")),
        "data_path": "/eqasim-data",
        "output_path": OUTPUT_DIR,
        "output_prefix": output_prefix,
        "java_memory": "4G",
        "regions": [],
        "departments": departments,
        "communes": communes,
        "communes_file": "",   # the list is passed in clear above; the host path does not exist here
        "gtfs_path": "gtfs_toulouse",
        "osm_path": "osm_toulouse",
        "ban_path": "ban_toulouse",
        "bdtopo_path": "bdtopo_toulouse",
        "generate_personality_traits": generate_personality,
    }
    assert set(runtime_config) == set(RUNTIME_OVERRIDDEN_KEYS)
    config = {
        "working_directory": "/eqasim-cache",
        "run": [
            "synthesis.population.llm_agents",
        ],
        "config": {**base_config, **runtime_config},
    }
    print("[eqasim] réglages scientifiques (config_toulouse.yml) : "
          + ", ".join(f"{k}={config['config'][k]}" for k in SCIENTIFIC_KEYS))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="eqasim_config_"
    ) as tmp:
        yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
        tmp_path = tmp.name

    # The file written by synpp is recognised by its DATE (written after the launch), not by the
    # novelty of its name: two generations that deliver the same size write the same
    # name, and « name unknown before the run » missed it (found on 2026-09-03: the fresh pool
    # stayed under its actual-size name, the stale target file was returned to the caller).
    pattern = re.compile(rf"^{re.escape(output_prefix)}population_(\d+)\.json$")
    t_start = time.time()

    print(f"[eqasim] Running synpp with config: {tmp_path}")
    result = subprocess.run(
        ["uv", "run", "-m", "synpp", tmp_path],
        cwd="/eqasim",
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    # synpp writes the file with the actual agent count (e.g. population_1021.json).
    # Rename it to the exact requested size so downstream code can find it by name.
    target_path = os.path.join(OUTPUT_DIR, f"{output_prefix}population_{population_size}.json")
    new_files = [
        (os.path.getmtime(os.path.join(OUTPUT_DIR, name)), int(m.group(1)), os.path.join(OUTPUT_DIR, name))
        for name in os.listdir(OUTPUT_DIR)
        if (m := pattern.match(name)) and os.path.getmtime(os.path.join(OUTPUT_DIR, name)) >= t_start - 1
    ]
    if new_files:
        _, n_written, src = max(new_files)
        if src != target_path:
            # The target file may already exist (forced regeneration): it is REPLACED. Before
            # 2026-09-03, it was left in place and returned to the caller — the fresh population
            # stayed under its actual-size name, and the notebook silently reread the old one.
            if os.path.isfile(target_path):
                print(f"[eqasim] {os.path.basename(target_path)} existait (régénération forcée) : remplacé")
            os.replace(src, target_path)
            print(f"[eqasim] Renamed {os.path.basename(src)} ({n_written} personnes) → {os.path.basename(target_path)}")
    elif not os.path.isfile(target_path):
        print(f"[eqasim] ERROR [ALARME] synpp n'a produit aucun fichier {output_prefix}population_*.json "
              f"dans {OUTPUT_DIR}")
        sys.exit(4)

    return _find_cached_file(population_size, prefix=output_prefix)


def main() -> None:
    """CLI entry point: launches run() with the default parameters."""
    run()


if __name__ == "__main__":
    main()
