# Data

This page lists every dataset the project reads, where it is expected on disk, where to get
it and under which licence. It then states what this repository ships, and what it does not
ship and why.

The licence of the code (Apache-2.0, see the [README](../README.md)) does not apply to the
data. Each dataset keeps its own licence, and you must follow it when you download or
redistribute it.

Contents:

1. Open datasets
2. The restricted survey (EMC² Toulouse 2023)
3. Derived resources shipped in the code
4. What this repository ships
5. What this repository does not ship
6. Press articles

---

## 1. Open datasets

### 1.1 Synthetic population (eqasim pipeline)

These inputs are needed only to regenerate a population. The shipped cohorts (Section 4)
are enough to run the simulation and to reproduce the paper. The folders are relative to
`services/eqasim-toulouse/data/`.

| Dataset | Folder | Official page | Licence |
|---|---|---|---|
| Population census 2022, individual records (RP 2022) | `rp_2022/` | https://www.insee.fr/fr/statistiques/8647104 and https://www.insee.fr/fr/statistiques/8647014 | Etalab Open Licence 2.0 |
| Home–work and home–school flows (MOBPRO, MOBSCO 2022) | `rp_2022/` | https://www.insee.fr/fr/statistiques/8589904 and https://www.insee.fr/fr/statistiques/8589945 | Etalab Open Licence 2.0 |
| Income by municipality and IRIS (Filosofi 2021) | `filosofi_2021/` | https://www.insee.fr/fr/statistiques/7756855 | Etalab Open Licence 2.0 |
| Permanent facility database (BPE 2024) | `bpe_2024/` | https://www.insee.fr/fr/statistiques/8217525 | Etalab Open Licence 2.0 |
| IRIS contours 2024 | `iris_2024/` | https://geoservices.ign.fr/contoursiris | Etalab Open Licence 2.0 |
| Municipality and IRIS codes 2024 | `codes_2024/` | https://www.insee.fr/fr/information/7708995 | Etalab Open Licence 2.0 |
| Company register (SIRENE) and its geolocation | `sirene/` | https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/ and https://www.data.gouv.fr/fr/datasets/geolocalisation-des-etablissements-du-repertoire-sirene-pour-les-etudes-statistiques/ | Etalab Open Licence 2.0 |
| Buildings (BD TOPO), departments 09, 11, 31, 32, 81, 82, edition 2025-03-15 | `bdtopo_toulouse/` | https://geoservices.ign.fr/bdtopo | Etalab Open Licence 2.0 |
| National address base (BAN) | `ban_toulouse/` | https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/ | Etalab Open Licence 2.0 |
| National transport and travel survey 2008 (ENTD) | `entd_2008/` | https://www.statistiques.developpement-durable.gouv.fr/enquete-nationale-transports-et-deplacements-entd-2008 | Public data of the French statistical service (open download) |
| OpenStreetMap, regional extracts of 2022-01-01 (`midi-pyrenees-220101.osm.pbf`, `languedoc-roussillon-220101.osm.pbf`) | `osm_toulouse/` | https://download.geofabrik.de/europe/france/midi-pyrenees.html and https://download.geofabrik.de/europe/france/languedoc-roussillon.html | ODbL 1.0, © OpenStreetMap contributors |
| Transit feeds for the pipeline | `gtfs_toulouse/` | see Section 1.2 | see Section 1.2 |

<!-- TODO: confirm the exact licence wording of the ENTD 2008 files on the ministry page
     (open download, but check whether a named licence is attached). -->

The upstream eqasim documentation describes the expected layout of each folder. The fork
keeps it and lists its Toulouse-specific changes at the top of
`services/eqasim-toulouse/CHANGELOG.md`.

Two further INSEE products support population framing on the host side:

| Dataset | Official page | Licence |
|---|---|---|
| Urban areas 2020 (AAV2020) | https://www.insee.fr/fr/information/4802589 | Etalab Open Licence 2.0 |
| 200 m population grid | https://www.insee.fr/fr/statistiques/fichier/6214726/grille200m_gpkg.zip | Etalab Open Licence 2.0 |

<!-- TODO: state the host folder these two files are read from (the working copy keeps them
     in data/insee/, which is not shipped) and which script reads them. -->

### 1.2 Routing

| Dataset | Folder | Official page | Licence |
|---|---|---|---|
| Urban transit network of Toulouse (Tisséo GTFS) | `data/gtfs/tisseo_gtfs/` | https://data.toulouse-metropole.fr/explore/dataset/tisseo-gtfs/ | ODbL, per the dataset page |
| Regional coaches (liO GTFS) | `data/gtfs/lio_gtfs/` | https://transport.data.gouv.fr/datasets/reseau-lio-occitanie | Per the dataset page |
| Regional trains (TER GTFS) | `data/gtfs/ter_gtfs/` | https://transport.data.gouv.fr/datasets/horaires-sncf and https://www.data.gouv.fr/fr/datasets/horaires-des-lignes-ter-sncf/ | Per the dataset page |
| OSM extract of the 453 survey municipalities | `data/gtfs/Toulouse.osm.pbf` | Cut from the two Geofabrik extracts above by `make osmnx-perimeter-graph` | ODbL 1.0, © OpenStreetMap contributors |
| Road congestion by hour and weekday, city and metropolitan area | `services/llm-agents/config/osmnx.yaml` | Aggregate values of the TomTom Traffic Index (tomtom.com/traffic-index) | Aggregate figures, cited |

<!-- TODO: check and state the licence of the liO and TER feeds on their dataset pages
     (transport.data.gouv.fr shows one licence per resource). -->
<!-- TODO: cite the edition (year) of the traffic index the congestion table was read from. -->

The transit feeds in service are annual feeds rebuilt from several partial exports by
`make gtfs-year` (output under `data/gtfs_year/`; `make gtfs-year-dry` plans without
writing). `make gtfs-year-holdout` masks one published month and measures how far the
extrapolated service is from the real one. The frozen trip sets record the SHA-256 of the feed
files they were computed from, so a newer export can be detected with
`make jeu-verifier NOM=<set>`.

### 1.3 Weather

| Dataset | Folder | Source | Licence |
|---|---|---|---|
| Daily weather history of Toulouse, monthly files 2025-01 to 2026-04 | `data/weather/historique-meteo-toulouse-YYYY-MM.csv` | https://www.historique-meteo.net | Free use provided the source is mentioned (header of each file) |
| Twelve-month summary and weather code table | `data/weather/meteo_toulouse_12_mois.csv`, `data/weather/meteo_toulouse_codes.csv`, `data/weather/Codes meteo.csv` | same | same |

Each persona is given a weather date taken from the survey period, and the simulation reads
the weather of the same month and day in these files.

<!-- TODO: the header states free use with attribution, but no named licence. Confirm the
     terms with the provider before redistribution, or replace the files by a download
     recipe. -->

## 2. The restricted survey (EMC² Toulouse 2023)

The 2023 household travel survey of the Toulouse area (EMC², certified methodology) is the
reference the paper scores against. Its record-level data are distributed by PROGEDO under
a restricted-access agreement:

- study page: https://data.progedo.fr/studies/doi/10.13144/lil-1750 (doi:10.13144/lil-1750);
- access: on request to the distributor, for research use, under its conditions;
- **this repository never redistributes the survey records**, in any format.

With access granted, place the delivery under `data/PROGEDO 2023/` (the code reads the CSV
standard files, `lil-1750-Donnees_CSV/fichiers_standards/`, and the fine-zone geometry of the
documentation folder). The following then become runnable:

| Task | Command |
|---|---|
| Recompute the reference shares and control margins | `make reference-marges` |
| Rebuild the fine-zone lookups (rings, housing type, zones) | `make zones`, `make housing-type`, `make communes-couronnes` |
| Refit the equipment and time laws | `make bike-ownership`, `make equipment-propensity`, `make terminal-time`, `make car-availability` |
| Refit the tabular reference models | `make policy`, `make logit`, `make forest`, `make klr` |
| Evaluate them on the frozen set | `make mnl-predict`, `make klr-predict`, `make common-set-predict` |

<!-- TODO: check the full list of PROGEDO-dependent targets and their prerequisites against
     make/choix-modal.mk before release. -->

**Test sample excluded.** A 58-person sample drawn from the survey records (with its frozen
trip set) served for a trip-level audit. It is excluded for the same reason as the survey
itself: it reproduces individual records.

## 3. Derived resources shipped in the code

`packages/mobility_core/src/mobility_core/data/` holds resources derived from the survey.
They are fitted coefficients, aggregate shares and lookup tables, not records:

| File | Content |
|---|---|
| `bike_ownership.json` | Bicycle ownership law (fitted coefficients) |
| `driving_license.json` | Driving licence law |
| `pt_subscription.json` | Transit pass law |
| `car_availability_emc2.json` | Car availability law |
| `terminal_time_emc2.json` | Access and egress time law |
| `mode_hierarchy_emc2.json` | Mode hierarchy used to name the main mode of a multi-leg trip |
| `commune_couronne.json` | Municipality → residential ring |
| `couronne_perimetre.geojson` | Ring geometry, built from IGN Admin Express (Etalab Open Licence 2.0) |

The survey reference shares used by the scorer are aggregates as well:
`scripts/data/population/cerema_values.yaml`.

Four lookups at fine-zone level are **not** shipped and are produced locally from the
survey by the targets of Section 2: `zf_couronne.json`, `zf_housing_type.json`,
`zf_zones.gpkg`, `zf_zones.meta.json`. The training tables and fitted models of
`scripts/progedo_logit/` are record-level and are not shipped either; they are refitted
from the survey.

Fitted coefficients and aggregate shares are summary statistics: they reproduce no survey
record, and the survey's aggregate results are themselves published in its public report.

## 4. What this repository ships

| Item | Path | Approximate size |
|---|---|---|
| Sealed cohort c1: 1,000 personas, 499 households | `data/population/population_1000_PANEL_v6/` | 2.5 MB |
| Sealed cohort c2: 1,000 personas, 501 households, disjoint from c1 | `data/population/population_1000_PANEL_v6_c2/` | 3.4 MB |
| Frozen trip set of c1: 3,299 trips, 3,161 with options | `data/jeux/population_1000_PANEL_v6_20260316_EN_c/` | 38 MB |
| Frozen trip set of c2 | `data/jeux/population_1000_PANEL_v6_c2_20260316_EN_c/` | 38 MB |
| Archived executions on c1: 52 experiments, 53 executions, 46 of them scored | `archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences/` | 70.6 MB compressed |
| Archived executions on c2: 13 experiments, 13 executions, 11 of them scored | `archive/1_regime_nominal/jeu_1000_PANEL_v6_c2_EN_c/experiences/` | 15.0 MB compressed |
| Memory experiment definitions (10 `experience_memoire.yaml`) | `config/experiences_memoire/` | small |
| Populations of the memory experiments, extracted from c1 | `data/population/population_1_*`, `population_4_foyer_*`, `population_6_foyers_*`, `population_20_foyers_*`, `population_5_memoire_*` | small |
| Weather history | `data/weather/` | 0.2 MB |
| Derived survey resources | `packages/mobility_core/src/mobility_core/data/` | small |

Counted on the built copy on 2026-09-28. `make unpack-runs` roughly doubles the size of
`archive/` on disk; the decompressed files are ignored by git.

**Cohorts.** Each cohort folder holds `population.json` (the file the simulation loads),
`MANIFEST.yaml` (its seal, including `population.sha256`), `CONTROLE.md` (the thirteen
control margins and their deviations), `report.json` and `selection.json`. The personas
are synthetic: they are drawn from the eqasim population built on open data, then selected
so that thirteen margins match the survey within one point. They describe no real person.
`make personas-verifier POP=<cohort folder>` checks that a cohort is the one its manifest
describes.

**Frozen trip sets.** Each set holds `MANIFEST.yaml` (the digests of every dependency:
population, GTFS files, OTP graph, OSMnx graph key, routing configuration) and
`propositions.jsonl` (for every trip, the itineraries offered). A set lets any decider be
evaluated on the same options without routers.

**Archived executions.** Each experiment folder holds its `experience.yaml` and one
`executions/<timestamp>/` folder per execution, with `decisions.jsonl.gz`, `moves.csv.gz`,
`scores.json`, `compteurs.json`, `synthese.json`, `execution.yaml`, and when present
`progression.json`, `erreurs.jsonl` and `INCIDENT.md`. The full prompts and answers of the
gateway are not included.

## 5. What this repository does not ship

| Item | Reason | How to obtain it |
|---|---|---|
| EMC² Toulouse 2023 records, and anything record-level derived from them | Restricted access | PROGEDO, see Section 2 |
| The 58-person survey test sample and its trip set | Reproduces survey records | Rebuild from the survey with access |
| Open inputs of the population pipeline (Section 1.1) | Size; each has its own distributor | Official pages above |
| GTFS feeds and OSM extracts | Size; licence of each distributor | Official pages above |
| OpenTripPlanner jar and graph, OSMnx graph pickles | Size; rebuilt deterministically | [INSTALL.md](INSTALL.md), Section 6 |
| GAMA layers (`services/GAMA/CityTransport/includes/`) | Derived from GTFS and OSM | `make gama-includes` |
| Caches (LLM decisions, OTP and OSMnx itineraries) | Regenerated by use | — |
| Full LLM exchanges of each run (`llm_exchanges.jsonl`) | Size, and they repeat the personas verbatim | Rerun |
| Simulator runs with memory and events | Size | Rerun with `make run` (see [EXPERIMENTS.md](EXPERIMENTS.md)) |

## 6. Press articles

Five experiments expose agents to a local press article (event files
`services/llm-agents/config/evenements/a07_*.yaml` to `a25_*.yaml`). The article texts are not
redistributed. `data/presse/sources.yaml` gives, for each article, its URL, title, outlet,
publication date and the sha256 of the texts used in the paper. The repository ships only the
texts written for the experiments: the paraphrases (`data/presse/articles_txt/<id>/paraphrase*.txt`)
and the sign grid (`data/presse/grille_signes.yaml`).

To rebuild the quoted excerpts:

```bash
python -m scripts.data.presse.fetch_articles             # pages → data/presse/articles_html/
python -m scripts.data.presse.extraire_textes extraire   # → articles_txt/<id>/brut.fr.txt
python -m scripts.data.presse.extraire_textes verifier   # compares with the paper's hashes
```

News sites change their pages, so the rebuilt French excerpt may differ from the one used in
the paper. The English excerpt (`brut.txt`) served to the agents was machine-translated. A new
translation has a new sha256, which the event file must declare before it loads: the replay is
then a new condition, not the paper's.
