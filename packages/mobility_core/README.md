# mobility-core

The EMC² Toulouse 2023 survey domain, with no LLM or infrastructure dependency. Residence
rings, fine zones, mode hierarchy, bike equipment, housing type, individual propensities
and framing of the surveyed population. Each module reads a frozen resource in
`mobility_core/data/` and refuses to guess what it does not contain: outside the layer, no
guessing; resource missing, it raises at load time. Version 0.1.0 (ticket 037).

The package imports neither `llm_gateway`, nor `mobility_llm`, nor Redis, Celery, httpx,
FastAPI or Jinja2 (import-linter contract `domaine-sans-gateway`). Its dependencies: PyYAML,
loguru; extra `geo`: geopandas, shapely, pyproj, numpy.

## Modules

| Module | One line |
|---|---|
| `resources.py` | where the files are: `data_path()` for the package resources, `find_repo_file()` for the repository reference files that are not copied |
| `residence_zone.py` | the residence ring, **read** from a list of communes (fine zone code → sector → ring), never computed from distance; outside the 453 communes = "out of scope", not 3rd ring |
| `zone_resolver.py` | from a point to the geographic variables of mode choice (`od_km` between centroids, `same_zone`, densities, distances to the centre) by point-in-polygon join; extra `geo`; alarm if more than 15% of points fall outside the layer |
| `geo_reference.py` | the Toulouse hypercentre, read from `feature_spec.json` and not redeclared; falls back to the copied constant if the spec is missing |
| `mode_hierarchy.py` | the main mode of a multimodal trip, once for the whole repository: metro > tram > cable car > bus > train > car > powered two-wheeler > bike > walking (appendix p. 53 of the report) |
| `bike_ownership.py` | the persona's bike in three stages learned from the survey: how many bikes in the household, who holds them, which type; deterministic by hashing the address |
| `equipment_propensity.py` | individual propensity for an equipment item — PT pass (`P12 == 6`) and driving licence (`P7 == 1`) — same design vector at training and at application |
| `housing_type.py` | the imputed housing type: fine-zone law, household-size lever (raking), deterministic draw by address; `None` outside the layer or without size |
| `population_reference.py` | the framing of the surveyed population (`population_emc2_2023.yaml`), validated and enforceable; `household_weight` to compare a household target with a sample of persons |

## `data/` resources

Three statuses not to be confused: **versioned** (in git), **shipped** (in the sdist and
the wheel), **restricted** (neither — produced locally). What separates them is the
ProGEDO / ADISP `lil-1750` agreement: it allows distributing **results**, not **data**
(ticket 038).

### Shipped with the package — results

| File | Read by | Produced by | Why distributable |
|---|---|---|---|
| `mode_hierarchy_emc2.json` | `mode_hierarchy` | `export_mode_hierarchy.py` | published AUAT/CEREMA report (p. 53) |
| `bike_ownership.json` | `bike_ownership` | `make bike-ownership` | coefficients of a fitted model, no table of observations |
| `pt_subscription.json`, `driving_license.json` | `equipment_propensity` | `make equipment-propensity` | same |
| `terminal_time_emc2.json` | terminal time | `make terminal-time` | aggregates, `min_cell` = 200 |
| `car_availability_emc2.json` | car ownership | `make car-availability` | three categories at ring level, 10,783 households |
| `commune_couronne.json` (453 communes), `couronne_perimetre.geojson` | `residence_zone` | `make communes-couronnes` | published scope; Admin Express geometry (Licence Ouverte). **Status to be confirmed with ADISP** |

### Restricted access — neither in git nor in the package

They reproduce the survey microdata or zoning. To be produced with the PROGEDO data under
`data/PROGEDO 2023/`:

| File | Read by | Command | Why excluded |
|---|---|---|---|
| `zf_couronne.json` | `residence_zone` | `make communes-couronnes` | reproduces the sampling plan: 785 fine zones → sampling sector → commune |
| `zf_housing_type.json` | `housing_type` | `make housing-type` | publishes counts per fine zone — median 12 households, 195 zones under 5 |
| `zf_zones.gpkg`, `zf_zones.meta.json` | `zone_resolver` | `make zones` | the survey's GIS layer itself |

**Where the code looks for them.** `mobility_core.resources.restricted_data_path(nom)` tries,
in order: `$MOBILITY_CORE_EMC2_DATA_DIR/<nom>`, the package's `data/` (repository, editable
install, containers that mount `mobility_core/src/mobility_core` as a volume), then
`mobility_core/src/mobility_core/data/<nom>` under a plausible repository root. None
exists? The `load()` functions raise, with the resource name, the command that produces it,
the environment variable and the locations tried — **never a silent fallback**: a guessed
ring is later read as a modal share, not as a bug.

```bash
export MOBILITY_CORE_EMC2_DATA_DIR=/chemin/vers/mes/ressources_emc2
```

The `RESTRICTED_RESOURCES` table in `resources.py` is the source of truth for this regime;
`tests/test_packaging_licence.py` checks it against `package-data` and `MANIFEST.in`.

`data_path("…")` returns the path even if the file does not exist: the caller decides
whether to make it an error.

### The package settings live in three places — and three are needed

1. `package-data` in `pyproject.toml` — explicit list, never a glob (`data/*.json`
   picks up whatever lies around on the build disk, `.gitignore` cannot prevent it);
2. `include-package-data = false` — at `true`, the default since setuptools 61, it
   **merges** with `package-data` instead of restricting it, and the wheel ships everything;
3. `MANIFEST.in` — whitelist governing the sdist, which `pyproject.toml` does not reach.

`tests/test_packaging_licence.py` builds both archives and checks their content: what
is authoritative is the artifact, not the declaration.

### Licence

The **code** is under Apache-2.0 (`LICENSE`). The code licence does **not** extend to the
resources in `data/`: the `NOTICE` file carries their conditions — research use, citation
of the survey according to the model appended to the `lil-1750` agreement.

## Repository reference files

Two files are not copied into the package, so as not to create a second source of
truth: `scripts/progedo_logit/feature_spec.json` (feature contract and hypercentre) and
`scripts/data/population/population_emc2_2023.yaml` (framing). `find_repo_file` looks for
them in order:

1. the dedicated environment variable, full path of the file: `MODE_CHOICE_FEATURE_SPEC`,
   `POPULATION_EMC2_REFERENCE`;
2. the `MOBILITY_CORE_REPO_ROOT` root;
3. a walk up from the current directory (8 levels);
4. `/app` — the `controller` container mounts `scripts/` under `/app/scripts`.

Nothing found → `None`, and each module says what it does about the absence (`geo_reference`
falls back to its constant and logs it; `population_reference` raises with the list of paths
tried). The tests set `MOBILITY_CORE_REPO_ROOT` to the repository root.

## Installation and tests

```bash
pip install -e ./mobility_core[geo,test]      # from the repository root; without [geo]: zone_resolver and CommunalZones unavailable
cd mobility_core && pytest                    # or `make test-mobility` from the root
```

199 tests collected on 2026-09-07, 1 skipped: the parity tests on the real zone layer
skip themselves when `zf_zones.gpkg` has not been exported; the others run offline on
versioned or synthetic resources. Markers: `unit`, `needs_data`.

History: `CHANGELOG.md`. The header docstrings of each module carry the measurements and
decisions (why the bus comes before the train, why the bike is not conditioned on the
imputed housing): read them before changing a law.
