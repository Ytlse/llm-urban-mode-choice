# Installation

This page takes a fresh clone to a running simulation. If you only want to recompute the
paper's scores from the archived decisions, you need neither Docker nor provider keys:
install the host Python environment (Section 2) and follow "Reproducing the paper's
numbers" in the [README](../README.md).

Contents:

1. Prerequisites
2. Host Python environment
3. Provider keys (`.env`)
4. Declaring providers (`providers.yaml`)
5. Checking the gateway configuration
6. Assets to fetch or rebuild
7. Starting the stack
8. Offline (headless) mode
9. Ports
10. Troubleshooting

---

## 1. Prerequisites

| Requirement | Version or size | Needed for |
|---|---|---|
| Docker Engine with the Compose plugin (`docker compose`) | recent | the whole stack |
| Memory available to Docker | see below | the whole stack |
| GAMA platform, desktop | 2025.06.4 | GUI mode only; offline mode uses the Docker image `gamaplatform/gama:2025.06.4` pinned in `infra/docker-compose.yml` |
| Python | 3.12 | host scripts, analysis, figures, `make` targets that run on the host |
| Java runtime | 21 | building the OpenTripPlanner graph on the host (`make otp-graph`) |
| `osmium` (osmium-tool) | any recent | cutting the OSM extract (`make osmnx-perimeter-graph`) |
| GNU make, `curl`, `shasum` | — | Makefile recipes |

**Memory.** The compose file caps the heavy services: `osmnx1` at 8 GB (it loads the walk,
bike and car graphs at once), each of `otp1`, `otp2`, `otp3` at 6 GB (4 GB Java heap),
`eqasim` at 10 GB and the headless `gama` service at 12 GB. Headless runs of 1,000 agents
were validated with a Docker VM of 24 GB. To free memory without removing containers,
stop the services you do not need:

```bash
make stop-services SERVICES="eqasim otp2 otp3"
```

An experiment on a frozen trip set (no simulator) needs far less: `make services-pretes
REQUIS="controller api worker"` starts only the controller, the gateway and their
dependencies (see [EXPERIMENTS.md](EXPERIMENTS.md)).

<!-- TODO: state the disk space needed once the asset list is final (OTP jar ~169 MB,
     OSM extract ~76 MB, GTFS feeds, OSMnx graph pickle ~225 MB, GAMA layers). -->

## 2. Host Python environment

The Makefile runs host-side Python through one interpreter,
`services/llm-agents/.venv/bin/python` (variable `VENV_PYTHON`). Create it at that path:

```bash
python3.12 -m venv services/llm-agents/.venv
services/llm-agents/.venv/bin/python -m pip install -r services/llm-agents/requirements.txt
services/llm-agents/.venv/bin/python -m pip install \
    -e packages/mobility_core -e "packages/llm_gateway[test]" -e packages/mobility_llm
```

Always call tools as `services/llm-agents/.venv/bin/python -m <tool>` rather than through
the scripts of `.venv/bin/`: those carry an absolute shebang that breaks when the folder
moves. Targets that need another interpreter accept an override, for example
`make dashboard DASHBOARD_PYTHON=/path/to/python` or `make policy SYNTHESIS_PYTHON=...`.

Tests of the three packages (they run separately, their `conftest.py` share a module name):

```bash
make test-all        # llm_gateway, mobility_core, mobility_llm, then import contracts
make lint            # ruff on the three packages
make typecheck       # mypy, strict on the public contract of the gateway
```

<!-- TODO: confirm that requirements.txt installs cleanly on a fresh Python 3.12 venv. -->

## 3. Provider keys (`.env`)

Docker Compose reads a `.env` file at the repository root; the `controller`, `api` and
`worker` services load it. Start from the template:

```bash
cp .env.example .env
```


Fill only the keys you have. The canonical names are `PROVIDER_KEYS__<provider>`:

| Variable | Instances it enables in `providers.yaml` |
|---|---|
| `PROVIDER_KEYS__google` | the `google_*_key1` instances (Gemini and Gemma models) |
| `PROVIDER_KEYS__google2` | the `google_*_key2` instances (a second key, hence a second quota bucket) |
| `PROVIDER_KEYS__mistral` | the `mistral_*` instances |
| `PROVIDER_KEYS__openai` | the `openai_*` instances |
| `PROVIDER_KEYS__groq` | the `groq_*` instances |
| `PROVIDER_KEYS__cerebras` | the `cerebras_*` instances |
| `PROVIDER_KEYS__lmstudio` | the `lmstudio_*` instances (local models served by LM Studio on the host, port 1234; any non-empty value switches them on) |
| `PROVIDER_KEYS__typesafeAI` | the `typesafe` decider of the experiment platform (typed classifier, called outside the gateway) |

`infra/docker-compose.yml` maps each of these generic names onto the instance names
(`PROVIDER_KEYS__<instance>`). An instance without a key is excluded from the rotation at
startup, with a warning in the gateway log. `LLM_GATEWAY_PROVIDERS_FILE` points to the
provider file (default `config/llm_gateway/providers.yaml`).

Never commit a filled `.env`. `make run NO_GOOGLE=1` blanks the Google keys for one run,
so that the Google instances leave the rotation.

The gateway writes the full prompts and answers of a run to `llm_exchanges.jsonl` in the
run folder. Treat that file as sensitive if your personas are.

## 4. Declaring providers (`providers.yaml`)

`config/llm_gateway/providers.yaml` is a deployment file: it declares the instances the
gateway may call, never their keys.

```yaml
providers:
  <instance_name>:
    adapter: <openai|google|groq|cerebras|mistral>   # base adapter; required when the name differs
    base_url: https://...                            # required
    default_model: <model id>                        # required
    rpm_limit: 15                                    # requests per minute, required
    tpm_limit: 250000                                # tokens per minute
    rpd_limit: 500                                   # requests per day (UTC window)
    tpd_limit: 2000000                               # tokens per day
    weight: 1.0                                      # share in the weighted rotation; 0 = out of rotation
    concurrency_limit: 3                             # batches in flight
    max_output_tokens: 16384                         # completion cap of the model
    disable_timeout: 180                             # seconds out of rotation after repeated errors
```

Notes:

- **Weights.** The file follows `weight = min(rpm_limit, tpm_limit / 3000) / 15`, so that a
  provider's share of the traffic follows its effective capacity. Recompute it when a limit
  changes.
- **Out of rotation.** `weight: 0` keeps an instance declared but never picked by the
  balancer; an experiment can still pin it by name.
- **Batch size.** The gateway derives `batch_max_agents` per instance at startup from its
  quotas; `GET /health` on port 8000 publishes the value in use.
- **Routing policy.** `LLM_GATEWAY_ROUTING__POLICY` selects `swrr` (smooth weighted round
  robin across all instances) or `cascade` (use the first instance until it refuses, then
  the next). The package default is `swrr`; the compose file sets `cascade` unless you
  override it.
- **Local models.** Any OpenAI-compatible server can be declared with `adapter: openai`
  and its `base_url`. For LM Studio: `make lmstudio-charger MODELE=<id> [CTX=16384]`,
  `make lmstudio-etat`, `make lmstudio-decharger MODELE=<id>`.
- **Refreshing quotas.** `make providers` probes each provider (one minimal request, or the
  provider's quota API) and rewrites the limits in place; `make providers DRY_RUN=1` only
  prints the result. The probe consumes a request per instance.
- After changing prompts or providers, reload the gateway: `make passerelle-recharger`
  (restarts `api` and `worker`).

## 5. Checking the gateway configuration

The package declares the entry point `llm-gateway = "llm_gateway.cli:main"`
(`packages/llm_gateway/pyproject.toml`). From the repository root, with the keys exported
in your shell (for example `set -a; . ./.env; set +a`):

```bash
LLM_GATEWAY_PROVIDERS_FILE=config/llm_gateway/providers.yaml \
    services/llm-agents/.venv/bin/python -m llm_gateway.cli config validate
```

It loads the effective configuration, prints the number of declared instances and of
instances with a key, and exits with code 1 if the configuration is invalid. Two other
sub-commands help before a first run:

```bash
services/llm-agents/.venv/bin/python -m llm_gateway.cli config show     # effective settings, secrets masked
services/llm-agents/.venv/bin/python -m llm_gateway.cli categories      # categories registered by the mobility bundle
```

`categories` should list `itinary_multi_agent`, `perception_filter`, `stm_reflection`,
`ltm_self_reflection` and `evenement_relais`. If `llm-gateway` is on your `PATH`, the same
commands read `llm-gateway config validate`, and so on.

## 6. Assets to fetch or rebuild

Large binary inputs are not in the repository. Each has a recipe or a download page.

<!-- TODO: a fetch script for the OTP jar, the Geofabrik extracts and the GTFS feeds is
     planned; until it exists, fetch them by hand as described below. -->

| Asset | Where it goes | How to obtain it |
|---|---|---|
| OpenTripPlanner 2.8.1 shaded jar | `services/otp-toulouse/bin/otp-shaded-2.8.1.jar` | Release page of OpenTripPlanner 2.8.1 (https://github.com/opentripplanner/OpenTripPlanner/releases/tag/v2.8.1) |
| Regional OSM extracts, edition 2022-01-01 | `services/eqasim-toulouse/data/osm_toulouse/midi-pyrenees-220101.osm.pbf` and `languedoc-roussillon-220101.osm.pbf` | Geofabrik, see [DATA.md](DATA.md) |
| OSM extract of the 453 survey communes | `data/gtfs/Toulouse.osm.pbf` | `make osmnx-perimeter-graph` cuts it with `osmium` into `data/cache/osmnx/perimetre_453/perimetre_453.osm.pbf`; copy that file to `data/gtfs/Toulouse.osm.pbf` |
| OSMnx graphs (walk, bike, car) of the same polygon | `data/cache/osmnx/graphs_444ca7e6a515.pkl` | `make osmnx-perimeter-graph` (no download; the `osmnx1` service refuses to start without it) |
| GTFS feeds | `data/gtfs/tisseo_gtfs/`, `data/gtfs/ter_gtfs/`, `data/gtfs/lio_gtfs/` | Dataset pages listed in [DATA.md](DATA.md). The TER and liO feeds in service are annual feeds rebuilt from partial exports by `make gtfs-year` (`make gtfs-year-dry` to plan without writing) |
| OpenTripPlanner graph | `data/gtfs/graph.obj` | `make otp-graph` (copies the versioned configs of `services/otp-toulouse/toulouse/`, archives the previous graph, then builds with `java -Xmx4G`) |
| GAMA layers | `services/GAMA/CityTransport/includes/` | `make gama-includes` (see below) |
| Synthetic population inputs (census, BD TOPO, BAN, ...) | `services/eqasim-toulouse/data/` | Only to regenerate a population; see [DATA.md](DATA.md) |

**GAMA layers.** The model reads four products from `includes/`: the world boundary
(`perimetre_453.shp`), the transit lines and stops (`routes.shp`, `stops.shp`), the
scheduled runs (`trip_info.json`) and, on the Python side, the route-to-shape table
(`shape_lookup.json`). See `make gama-includes`, which chains the recipes in that order and
stops at the first failure. `make gama-layers` and `make gama-trip-info` rebuild parts of it;
`make test-gama-includes` runs the unit tests of those recipes on synthetic feeds.

<!-- TODO: `make gama-includes` is being finalised (optional layers ANNEXES=1, comparison
     target). Describe its variables here once it is frozen. -->

**Checking the transit assets against the paper.** The manifest of the frozen trip set
records the SHA-256 of the Tisséo feed files, of the OTP graph, the OSMnx graph key and the
routing configuration files. `make jeu-verifier NOM=population_1000_PANEL_v6_20260316_EN_c`
lists every dependency that differs from the files in place. A recent GTFS export will
differ: that only matters if you rebuild trip sets or run the simulator.

## 7. Starting the stack

```bash
make up            # docker compose -f infra/docker-compose.yml --project-directory . up -d
make ps            # container status
make logs          # follow all logs
make down          # stop everything (including the offline gama service)
```

The compose file lives in `infra/`; the Makefile always passes `--project-directory` so that
relative paths resolve from the repository root. If you call Docker Compose directly, pass
the same two options.

**Startup order (GUI mode).**

1. `make up` starts `redis`, `eqasim`, `otp1`–`otp3`, `osmnx1`, `api`, `worker`,
   `controller` and the monitoring services. The controller waits for healthy routers and
   gateway; the first start of the routers takes several minutes (graph loading).
2. Open GAMA 2025.06.4 and the model `services/GAMA/CityTransport/models/City.gaml`.
3. Run the experiment `e`. The controller connects to the model's WebSocket at
   `ws://host.docker.internal:3001` and reconnects indefinitely, so GAMA can be started at
   any time after the stack.

`make run` does all three on a machine where GAMA is installed at the path of `GAMA_BIN`
in the `Makefile` (override it: `make run GAMA_BIN=/path/to/GAMA`). Before starting, it stops
any previous simulation, resets Grafana and Prometheus data (unless `CONT=1`), and recreates
the controller when `services/llm-agents/config/config.yaml` changed since the last launch.
`make wait-ready` polls the gateway, the controller and Grafana before GAMA is launched.

Run options, all written to configuration files before the stack starts:

| Option | Effect |
|---|---|
| `MEM=0` / `MEM=1` | turn long-term memory and self-reflection off or on (written to `services/GAMA/CityTransport/config/sim_params.yaml`, persistent) |
| `CACHE=0` / `CACHE=1` | turn the semantic decision cache off or on (written to `services/llm-agents/config/config.yaml`) |
| `EVENEMENT=<name>` (aliases `CHOC=`, `PRESSE=`) | play a declared event of `services/llm-agents/config/evenements/`; `EVENEMENT=0` removes it |
| `JEU=<name>` | serve itineraries from the frozen trip set `data/jeux/<name>/` instead of querying the routers |
| `CONT=1` | resume the previous run in the same run folder |
| `NO_GOOGLE=1` | run without the Google instances |

Scenario parameters (population, simulated days, start date) are read by GAMA from
`services/GAMA/CityTransport/config/sim_params.yaml`, in both modes.

## 8. Offline (headless) mode

```bash
make run OFFLINE=1       # alias: make run-offline
```

What it does:

1. Activates the compose profile `offline`, which adds the `gama` service
   (`gamaplatform/gama:2025.06.4`, started as GAMA Server on port 6868).
2. Points the controller and the gateway to `ws://gama:3001` instead of
   `ws://host.docker.internal:3001` (variable `GAMA_WS_URL`).
3. Once the services are ready, runs `scripts/gama/launch_headless.py` inside the
   controller container. The launcher sends `load` then `play` to GAMA Server and injects
   the controller address into the experiment.
4. Relays the GAMA console to `experiments/current/gama_headless.log`.

The launcher must keep its WebSocket open for the whole run: GAMA Server stops an
experiment whose client disconnects. Stop a run with `make stop-run` (the rest of the stack
stays up) and check its state with `make status`.

**Resuming.** `make run OFFLINE=1 CONT=1` reuses the run folder that `experiments/current`
points to: logs are appended, the controller state and memory checkpoints are reloaded,
Grafana, Prometheus and Redis data are kept. GAMA restarts at the beginning of the
simulated day and replays it; the caches make the replay fast, and memory stays frozen
until the simulated clock passes the checkpoint. `make run OFFLINE=1 REPRISE=<run name>`
resumes a named run.

Each run writes to `experiments/archive/<YYYY-MM-DD>_<HH_MM>/`, and `experiments/current`
is a symbolic link to the latest one. Without a display, observe a run through Grafana
(port 3000), the population map served by the controller (port 5050) and the reports of
Section 10.

## 9. Ports

| Service | Port | Role |
|---|---|---|
| `controller` | 8002 | Controller (hypercorn); `/sync` endpoint called by GAMA |
| `controller` | 5050 | Population map |
| `api` | 8000 | LLM gateway (`/tasks`, `/health`, `/metrics`) |
| `eqasim` | 8003 | Synthetic population service |
| `otp1`, `otp2`, `otp3` | 8080, 8081, 8082 | OpenTripPlanner |
| `osmnx1` | 8090 | Direct routing (walk, bike, car) |
| `redis` | 6379 | Quotas, task queue, controller state |
| `flower` | 5555 | Celery task monitor |
| `prometheus` | 9090 | Metrics |
| `grafana` | 3000 | Dashboards (**anonymous Admin access: keep it on localhost**) |
| `node_exporter`, `cadvisor` | 9100, 8888 | Host and container metrics |
| `gama` (offline profile) | 6868 | GAMA Server |

**Do not change the controller's HTTP server.** GAMA's Java 21 HTTP client adds an
`Upgrade: h2c` header to every request. uvicorn drops the body of such requests, so every
`/sync` would arrive empty. The controller therefore runs under hypercorn, which supports
HTTP/2 cleartext, and its `/sync` handler reads the raw request body on purpose.

## 10. Troubleshooting

| Symptom | What to check |
|---|---|
| Agents do not move | Were the Docker services up before GAMA? Do the controller logs (`docker compose -f infra/docker-compose.yml --project-directory . logs controller`) show a successful WebSocket connection to `ws://host.docker.internal:3001` (GUI) or `ws://gama:3001` (offline)? In offline mode, read `experiments/current/gama_headless.log` for load or play errors |
| `/sync` requests arrive empty | The controller is not running under hypercorn (see Section 9) |
| `osmnx1` never becomes healthy | The graph pickle `data/cache/osmnx/graphs_444ca7e6a515.pkl` is missing: run `make osmnx-perimeter-graph` |
| OTP instances restart in a loop | `data/gtfs/graph.obj` is missing or was built with another OTP version: run `make otp-graph` |
| GAMA shows no transit vehicles | `includes/` is stale or incomplete: rebuild it with `make gama-includes`; the simulated date must be served by the feeds |
| Every decision falls back to a default | No instance has a key, or all quotas are spent. Check `curl -s localhost:8000/health` and the gateway log; in an experiment, the run stops after three consecutive fallbacks instead of recording them |
| A container dies without a log line | Memory limit. `make watch-containers` samples container memory and records which one falls |

Reports on the latest run (they read `experiments/current` unless `RUN=` is given):

```bash
make error        # ERROR lines and [ALARME] alarms
make warning      # WARNING lines
make report       # run health: errors, LLM saturation, backlog, idle agents, timeouts
make init         # initialisation timeline and cache warm-up
make capacity     # LLM throughput against declared capacity
```

Alarms are logged at ERROR level with the prefix `[ALARME]`, on the rising edge of the
condition, so that `make error` shows each incident once.
