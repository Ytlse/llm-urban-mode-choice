# llm-agents

Urban multi-agent simulation project combining a GAMA simulation backend, cognitive agents driven by LLM, and an asynchronous API server. The goal is to model urban mobility behaviours from synthetic populations, GTFS data and trip scenarios.

---

## General tree

```
llm-agents/
├── api_server.py
├── backup_helper.py
├── errors.py
├── gama_models.py
├── helper.py
├── models.py
├── server.py
├── settings.py
├── utils.py
├── api/
├── config/
├── handle/
├── input/
│   ├── experiments/
│   └── gtfs/
├── llm/
├── scenario/
│   └── scenario_v1/
├── text_helper/
│   ├── models/
│   └── templates/
├── trip_helper/
└── world/
```

---

## Root files

| File | Description |
|---|---|
| `api_server.py` | API server initialisation. Secondary entry point or startup utility. |
| `backup_helper.py` | File rolling mechanism (log rotation or backup). Exact role to be clarified. **TBD** |
| `errors.py` | Centralised definition of the custom exceptions used in the project. |
| `gama_models.py` | Binding classes with GAMA: responses, limits, synchronisation, exchange objects with the simulation engine. |
| `helper.py` | General utility functions hard to classify elsewhere: time conversions (e.g. time slot → "morning" label), date conversion, sentiment utilities, etc. |
| `models.py` | Business domain classes mirroring the GAML model: persons, transitions, states. Python mirror of the GAMA model. |
| `server.py` | Main entry point. Contains the `main` class, loads the configuration and the settings, starts the server through `uvicorn.run`. |
| `settings.py` | Global project configurations: LLM models, server, environment parameters. Groups all the configuration classes (Settings pattern). |
| `utils.py` | Miscellaneous utilities: random generation, distance computations, auxiliary mathematical functions. |

---

## `api/`

Asynchronous API layer handling the exchanges between the server and the GAMA simulation.

| File | Description |
|---|---|
| `application.py` | Declares the FastAPI objects and the simulation loading logger. Short file, bootstrap role for the application. |
| `batch.py` | Asynchronous functions handling trips in batch: `query_move` (trip request for an agent, with semaphore), `batch_move_next` (asynchronous creation of the trip tasks), `batch_ob_update` (update of observations — **not implemented yet, TBD**). |
| `handles.py` | Handling of the initialisation of the simulated environment. The `init` function sends a `world_init` message to GAMA with the list of persons, the number of agents and the start date. |

---

## `config/`

Contains the YAML configuration files of the experiments (baseline and tested variants). These files are probably selected or copied dynamically depending on the scenario launched.

---

## `handle/`

Low-level interface layer with GAMA and WebSocket.

| File | Description |
|---|---|
| `application.py` | Main interface with GAMA. Listening loop for new incoming messages, processing of observations, GAMA initialisation, definition of the FastAPI handlers.
| `websocket.py` | WebSocket client to the GAMA server (port 3001). Handles connection, disconnection, sending and receiving of messages. Indefinite automatic reconnection to tolerate GAMA starting after the Docker services are launched. |

---

## `input/`

Input data of the simulation.

### `experiments/`
Current configuration of the ongoing experiment (probably copied automatically from `config/` at launch).

### `gtfs/`

| File | Description |
|---|---|
| `gamma.py` | GAMA builders for the GTFS objects: loading of the trips, declaration of the `TripInfo` classes, feeding of the GAMA model from the transport data. |
| `reader.py` | Reading and parsing of the GTFS zip. Initialises the routes, stops and schedule information. Creates the corresponding Python objects for injection into GAMA. |

### `population/`

| File | Description |
|---|---|
| `base.py` | Abstract base class for handling the simulated population. |
| `spatial_filter.py` | Filters the population by spatial proximity to transport stops (configurable radius). |
| `synthetic.py` | Loading of a synthetic population from external files. |

---

## `llm/`

LLM integration layer and management of the agents' memory.

| File | Description |
|---|---|
| `longterm.py` | Long-term memory of the agents: storage, retrieval, consolidation logic. Rich module, to be documented in detail. **TBC** |
| `memory.py` | Definition of the `MemoryEntry` class: unit memory entry with all its metadata (timestamp, type, content, etc.). |
| `shortterm.py` | Short-term memory: ordered list of `MemoryEntry`, management of the agent's context window. |
| `vllm_server.py` | Client for an OpenAI-compatible LLM server (vLLM). Provides two message-sending functions: synchronous and asynchronous. |

---

## `scenario/`

### `scenario_v1/`

| File | Description |
|---|---|
| `base.py` | Abstract class defining the contract of a scenario: methods `find_observation`, `assign_message`, `init_population`, etc. |
| `history.py` | Logging of simulation events in JSON format. Makes it possible to trace the full history of a run. |
| `llm_config.py` | Builds the LLM configurations on the fly from `settings.py` according to the active provider (OpenAI or vLLM): API URL, temperature, max number of tokens, etc. |

---

## `text_helper/`

Generation of narrative text describing the agents' trips (LLM feedback, context prompts).

### `models/`

| File | Description |
|---|---|
| `arrival.py` | Describes an agent's arrival at destination: on time, late, associated feedback. |
| `transfer.py` | Describes a walking segment between two points. |
| `transit.py` | Describes a public transport segment: distance, duration, line used. |
| `travel_plan.py` | Aggregates the information of all the segments of a trip: total walking time, total distance, etc. |
| `wait_in_stop.py` | Describes the waiting time at a stop. |

### `templates/`

Subfolder containing Jinja2 templates (`.j2`) for generating textual descriptions of travel plans, transfers and waits at stops. To be explored further. **TBC**

| File | Description |
|---|---|
| `repository.py` | Loading and selection of the `.j2` templates depending on the context. Filtering logic to be clarified. **TBD** |

---

## `trip_helper/`

Itinerary planning interface through external tools (OTP, Solari).

| File | Description |
|---|---|
| `cached_triphelper.py` | Common interface (facade) for retrieving trip scenarios. Delegates to the selected implementation (OTP or Solari) through `get_itinerary`-type methods. |
| `otp.py` | OTP (OpenTripPlanner) client: building of requests, parsing of responses, return of structured itineraries. |
| `solari.py` | Solari client: seems to be an earlier or alternative version to OTP, with a different endpoint. Probably V1 of the planner. **TBC** |

---

## `world/`

Management of the global state of the simulation.

| File | Description |
|---|---|
| `population.py` | `Population` class: handling of individual actions on the agents (start an activity, get the next one, finish). `WorldPopulation` class: initialisation of the GAMA environment with the population, global statistics and current state of the simulation. |
| `world_data.py` | Environmental data of the simulation: time grid, geospatial data, map projections. |

---

## Points to explore further

| Item | Nature | Priority |
|---|---|---|
| `backup_helper.py` | Exact role of the file rolling | **TBD** |
| `api/batch.py` → `batch_ob_update` | Observation feedback, not implemented | **TBD** |
| `llm/longterm.py` | Detail of the long-term memory strategies | **TBC** |
| `text_helper/templates/repository.py` | Template selection logic | **TBD** |
| `trip_helper/solari.py` | Exact relationship with OTP, active/deprecated status | **TBC** |

---

*README generated from a spoken description of the project — some interpretations are approximate and marked TBD (to be defined) or TBC (to be confirmed).*
