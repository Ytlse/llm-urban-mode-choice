# Generative mobility agents on a real household travel survey

This repository holds the code, configuration and minimal data behind the GAMA Days
abstract *Plugging Language Models into GAMA Mobility Simulations: Generative Mode Choice
Against a Real Household Travel Survey*. It simulates daily mobility in the Toulouse metropolitan area
(France) with agents whose transport-mode decision is delegated to a large language
model (LLM), and it measures those decisions against a certified household travel survey.

## What the simulation does

1. A synthetic population of residents is drawn from open data with a fork of eqasim,
   then controlled and sealed against the margins of the 2023 Toulouse household travel
   survey (EMC²).
2. Each resident is a persona with a daily activity chain (home, work, school, shops...).
3. The GAMA model moves the agents in space and time on the road and transit networks.
4. For every trip, the controller asks OpenTripPlanner (transit) and OSMnx (walk, bike,
   car) for itineraries, and removes those that use a vehicle parked elsewhere.
5. A decision-maker returns one probability per itinerary: an LLM reached through a
   multi-provider gateway, a tabular reference model fitted on the survey, or a baseline.
6. The executed itinerary is drawn from those probabilities with a fixed seed.
7. Agents keep a short-term and a long-term memory, and can be exposed to declared events
   (a network incident, a local press article).
8. Mode shares are scored against the survey, overall and by stratum.

This work builds on the architecture described by Vu et al. (2025), *Modeling realistic
human behavior using generative agents in a multimodal transport system: Software
architecture and Application to Toulouse*, arXiv:2510.19497.

## Repository map

| Folder | Content |
|---|---|
| `infra/` | `docker-compose.yml`: every service of the stack, including headless GAMA |
| `Makefile`, `make/*.mk` | All commands (`make help` lists the documented targets) |
| `services/GAMA/` | The GAML model (`CityTransport/models/City.gaml`) and its scenario parameters |
| `services/llm-agents/` | The controller: agent lifecycle, routing clients, memory, events, experiment platform (`experiences/`) |
| `services/otp-toulouse/` | OpenTripPlanner build and router configuration |
| `services/eqasim-toulouse/` | Fork of the eqasim synthetic-population pipeline (GPL-2.0) |
| `packages/llm_gateway/` | Generic multi-provider LLM gateway: micro-batching, quota-aware routing, Celery workers |
| `packages/mobility_core/` | Survey domain: residential rings, mode hierarchy, equipment laws, population framing |
| `packages/mobility_llm/` | LLM categories of the mobility model: persona, prompt templates, output schemas, probabilistic mode choice |
| `config/llm_gateway/` | `providers.yaml`: declared model providers and their quotas (no keys) |
| `scripts/` | Data preparation, analysis, figures, reference models (`progedo_logit/`), score synthesis (`synthesis/`), dashboard |
| `data/` | Sealed cohorts, frozen trip sets, weather history (see [docs/DATA.md](docs/DATA.md)) |
| `archive/1_regime_nominal/` | Archived executions of the benchmark, decision by decision |
| `campagnes/` | Declared batches of experiments |
| `docs/` | This documentation |

<!-- TODO: confirm the final public folder list once the copy script has run (tests/, notebooks/). -->

## Quickstart

Full instructions, prerequisites and troubleshooting are in [docs/INSTALL.md](docs/INSTALL.md).

```bash
# 1. Provider keys (variable names only; see docs/INSTALL.md)
cp .env.example .env            # then fill PROVIDER_KEYS__<provider>=...

# 2. Fetch or rebuild the routing and GAMA assets (OTP jar, OSM extract, GTFS, GAMA layers)
make otp-graph                  # builds data/gtfs/graph.obj from data/gtfs/
make gama-includes              # rebuilds services/GAMA/CityTransport/includes/

# 3. Start the stack
make up                         # docker compose up -d, all services

# 4a. Desktop GAMA: open services/GAMA/CityTransport/models/City.gaml, run experiment "e"
# 4b. or headless GAMA in a container
make run OFFLINE=1
```

`make run OFFLINE=1` starts the stack if needed, launches GAMA Server in the `gama`
container and drives the experiment through `scripts/gama/launch_headless.py`. The console
of GAMA goes to `experiments/current/gama_headless.log`. `make stop-run` stops the
simulation and leaves the other services up; `make down` stops everything.

**Security note.** Grafana runs with anonymous access and the Admin role
(`GF_AUTH_ANONYMOUS_ORG_ROLE=Admin` in `infra/docker-compose.yml`). Do not expose port 3000
beyond `localhost`. The same caution applies to the other published ports (Redis 6379,
the gateway 8000, the controller 8002).

## Reproducing the paper's numbers without any LLM call

Most results of the paper are recomputed from archived decisions. No model is called, no
router is queried, and GAMA is not needed.

**What is shipped for this purpose.**

| Item | Path |
|---|---|
| Sealed cohort c1 (1,000 personas, 499 households) | `data/population/population_1000_PANEL_v6/` |
| Sealed cohort c2 (1,000 personas, 501 households, disjoint from c1) | `data/population/population_1000_PANEL_v6_c2/` |
| Frozen trip set of c1 (3,299 trips, 3,161 with options) | `data/jeux/population_1000_PANEL_v6_20260316_EN_c/` |
| Frozen trip set of c2 | `data/jeux/population_1000_PANEL_v6_c2_20260316_EN_c/` |
| Archived executions on c1 | `archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences/` |
| Archived executions on c2 | `archive/1_regime_nominal/jeu_1000_PANEL_v6_c2_EN_c/experiences/` |
| Survey reference shares (aggregates) | `scripts/data/population/cerema_values.yaml` |

Each experiment folder holds its `experience.yaml` and one folder per execution under
`executions/<timestamp>/`, with `decisions.jsonl.gz` (one line per decision: options shown,
options removed and why, probabilities, drawn option), `moves.csv.gz` (the trip log),
`scores.json`, `compteurs.json`, `synthese.json` and `execution.yaml`.

**Check the seals.**

```bash
shasum -a 256 data/population/population_1000_PANEL_v6/population.json
# 412efada802f79e8a72976ba25e0c7db8c9404adaed7c1f5e3e3d6afa3531db6
shasum -a 256 data/jeux/population_1000_PANEL_v6_20260316_EN_c/propositions.jsonl
# e8f9eab14fb4a0600bc04bd85591053dd0fd3bd3571e6e9f98aec87e51bc9f53
make personas-verifier POP=data/population/population_1000_PANEL_v6
```

**Recompute the paired intervals of the results section and Appendix H.**

```bash
services/llm-agents/.venv/bin/python -m scripts.analysis.paired_intervals \
    --preset chapter6 --out <dir>          # add -B 2000 (default) or fewer replicates
```

The preset pins the twelve executions used by the paper (never "the latest"), resamples
persons with replacement (seed 2026) and writes `paired_chapter6_B<replicates>.json`,
with the twenty paired differences and the per-arm half-widths. It reads `moves.csv.gz`
directly. Two other presets exist: `jev_mutations` and `jev32_vs_tabular`.

**Rescore one execution.** From `services/llm-agents/` (or inside the `controller`
container), `python -m experiences score <execution folder>` recomputes the composite and
the per-stratum detail from the decisions.

**Decompress for the platform tools.** The platform commands (`registre`, `comparer`,
`apparier`, `score`) and some figure scripts expect plain `decisions.jsonl` and `moves.csv`:

```bash
make unpack-runs
```

`make unpack-runs` (defined in `make/publication.mk`) decompresses every `decisions.jsonl.gz`
and `moves.csv.gz` under `archive/1_regime_nominal/` next to its archive; `RUNS=<folder>`
restricts it. It is idempotent.

The read-only commands then find the archive on their own: with no `EXPERIENCES_DIR` and
no execution under `data/experiences/`, they read `archive/1_regime_nominal/`
([docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), Section 6). The scoring formula
(EMD, JSD, L1) is in `scripts/synthesis/formule_score/`.

The mapping from each table and figure to its script is in
[docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md). Results that need new decisions (a model
replay), the simulator (Section 6) or the survey microdata (reference-model fitting, the
trip-level audit) are marked there.

## Documentation

| Document | Content |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | Prerequisites, provider keys, startup order, offline mode, assets to fetch, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Services and data flow, LLM gateway, agent lifecycle, memory, vehicle chain, routing and caches, backpressure |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiment platform, frozen sets, memory and news-shock experiments, per-day measurements, sealed population, dashboard |
| [docs/DATA.md](docs/DATA.md) | Every dataset, its source, its licence, and what is not redistributed |
| [docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md) | Paper element → script → command → data |

## License and third-party components

| Component | License |
|---|---|
| This project (all code unless stated otherwise) | Apache License 2.0, see [LICENSE](LICENSE) |
| `packages/mobility_core` | Dual licence: `Apache-2.0 OR GPL-2.0-or-later`, because the GPL-2.0 eqasim fork imports it |
| `services/eqasim-toulouse` | GNU GPL v2.0. Fork of [eqasim-org/eqasim-france](https://github.com/eqasim-org/eqasim-france) at upstream commit `2fade7c1`, with Toulouse-specific changes listed at the top of its `CHANGELOG.md`. Its [LICENSE](services/eqasim-toulouse/LICENSE) and [CITATION.cff](services/eqasim-toulouse/CITATION.cff) are kept unchanged; please cite eqasim as that file asks |
| GAMA platform | Used as an external application (Docker image `gamaplatform/gama:2025.06.4`), not redistributed |
| OpenTripPlanner | Used as an external jar (2.8.1), not redistributed |

<!-- TODO: check that packages/mobility_core/pyproject.toml and its NOTICE carry the dual
     licence expression in the public copy (the working copy still declares Apache-2.0). -->

Data licences differ from the code licence. In short: INSEE, IGN (BD TOPO, IRIS), BAN and
SIRENE data are under the Etalab Open Licence 2.0; OpenStreetMap under the ODbL; transit
feeds under the licence of each dataset page; the ENTD 2008 survey is public; the
2023 Toulouse EMC² microdata are **restricted** (PROGEDO, convention `lil-1750`), available
on request and never redistributed here. The resources in `packages/mobility_core/src/mobility_core/data/`
derived from that survey are fitted coefficients and aggregates, not records. Details and
URLs: [docs/DATA.md](docs/DATA.md).

## Citation

```bibtex
@misc{bru2026pluggingllm,
  title  = {Plugging Language Models into {GAMA} Mobility Simulations: Generative Mode Choice
            Against a Real Household Travel Survey},
  author = {Bru, Y. and Gaudou, B. and Oberoi, K. S.},
  year   = {2026},
  note   = {Abstract, GAMA Days. Code and data: https://github.com/Ytlse/llm-urban-mode-choice}
}
```

Prior work this repository builds on: Vu et al. (2025), *Modeling realistic
human behavior using generative agents in a multimodal transport system: Software
architecture and Application to Toulouse*, arXiv preprint
[arXiv:2510.19497](https://arxiv.org/abs/2510.19497).
