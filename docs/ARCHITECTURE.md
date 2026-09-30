# Architecture

This page describes how the simulation is built: the services and the data that flows
between them, the LLM gateway, the life of an agent, its memory, the vehicle chain, routing
and caches, and the backpressure that keeps GAMA and the model providers in step.

Contents:

1. Services and data flow
2. The LLM gateway
3. Agent lifecycle
4. Short-term and long-term memory
5. The vehicle chain
6. Routing and caches
7. Backpressure
8. Where to look in the code

Parameter values quoted here are the defaults of `services/llm-agents/settings.py` and of
the gateway settings (`packages/llm_gateway/src/llm_gateway/config/settings.py`). An
experiment can override them; the overrides it used are recorded in its `execution.yaml`.

---

## 1. Services and data flow

All services are declared in `infra/docker-compose.yml`.

| Service | Image or entry point | Role |
|---|---|---|
| `controller` | hypercorn `handle.application:app`, port 8002 | Owns the agents: activity chains, itinerary requests, LLM decisions, memory, events. Talks to GAMA over HTTP (`/sync`) and WebSocket |
| `api` | uvicorn `llm_gateway.main:app`, port 8000 | Front of the LLM gateway: receives tasks, groups them in batches |
| `worker` | Celery worker of the gateway | Sends batches to the model providers, under quotas |
| `redis` | Redis 7, port 6379 | Database 0: gateway state and quotas, controller state. Database 1: Celery broker. Database 2: Celery results |
| `otp1`, `otp2`, `otp3` | OpenTripPlanner 2.8.1 on Java 21, ports 8080–8082 | Transit itineraries (urban network, regional coaches, regional trains) |
| `osmnx1` | uvicorn `osmnx_server:app` (`services/llm-agents/osmnx_server.py`), port 8090 | Direct itineraries on the road graph: walk, bike, car |
| `eqasim` | port 8003 | Synthetic population generation |
| `gama` | `gamaplatform/gama:2025.06.4`, profile `offline` | Headless GAMA Server (port 6868) |
| `prometheus`, `grafana`, `node_exporter`, `cadvisor`, `flower` | ports 9090, 3000, 9100, 8888, 5555 | Monitoring |

The controller also serves a population map on port 5050 (`services/llm-agents/vizpop.py`).

```
             ┌──────────── GAMA (desktop, or the `gama` container) ────────────┐
             │  moves the agents, runs the clock, displays the city            │
             └───────┬──────────────────────────────────────▲──────────────────┘
          POST /sync │  (agent states, arrivals)             │ WebSocket
                     ▼                                       │ (decisions, notifications)
             ┌──────────────── controller (port 8002) ───────┴─────────────────┐
             │ activity chains · dispatcher · memory · events · backpressure   │
             └──┬──────────────┬──────────────┬───────────────┬────────────────┘
                │ itineraries  │ itineraries  │ POST /tasks   │ state
                ▼              ▼              ▼               ▼
         otp1 / otp2 / otp3   osmnx1      api (8000) ──► redis ◄── worker (Celery)
         (transit)          (walk, bike,                                │
                             car)                                        ▼
                                                              model providers
                                                              (OpenAI-compatible,
                                                               Google, Mistral, ...)
```

A simulated day proceeds as follows:

1. GAMA calls `/sync` at each simulation step with the agents' states and the arrivals.
2. The controller works out which agents must plan their next trip.
3. For each of them it requests itineraries from the routers, removes the options that the
   vehicle chain forbids, and builds a decision task.
4. The task goes to the gateway, which batches it with others and sends it to a provider.
5. The answer carries one probability per option. The controller draws the executed option
   with a fixed seed, records the decision and pushes it to GAMA over the WebSocket.
6. GAMA moves the agent along the chosen itinerary and reports its arrival on a later
   `/sync`.

**Why hypercorn.** GAMA's Java 21 HTTP client adds an `Upgrade: h2c` header to every
request. uvicorn drops the body of such requests, so `/sync` would always arrive empty. The
controller runs under hypercorn, which supports HTTP/2 cleartext, and the `/sync` handler
reads the raw request body on purpose. Keep hypercorn on port 8002.

**WebSocket.** The controller is the client: it connects to GAMA at `GAMA_WS_URL`
(`ws://host.docker.internal:3001` for a desktop GAMA, `ws://gama:3001` offline) and
reconnects indefinitely (`services/llm-agents/handle/websocket.py`, with a keepalive
`ping_timeout` of 60 s). In offline mode, `scripts/gama/launch_headless.py` holds a second
connection to GAMA Server; its ping timeout (`GAMA_PING_TIMEOUT_S`, default 1,200 s)
tolerates long blocking steps, because GAMA Server stops an experiment whose client
disconnects.

## 2. The LLM gateway

The gateway is a separate package (`packages/llm_gateway`) with no knowledge of mobility.
The mobility model plugs into it through a bundle (`packages/mobility_llm`) registered under
the entry point group `llm_gateway.categories` (`mobility = "mobility_llm:bundle"`).

### 2.1 Categories

A category is one kind of LLM call. It brings its prompt template
(`categories/<name>/template.md.j2`), its output schema (`output_schema.json`) and its
priority. The mobility bundle declares, among others:

| Category | Purpose |
|---|---|
| `itinary_multi_agent` | Mode choice: one probability per offered itinerary, for a batch of agents |
| `perception_filter` | Filters what an agent notices of an event or of its trip |
| `stm_reflection` | Consolidates the short-term buffer into long-term memories |
| `ltm_self_reflection` | Periodic self-reflection over recent long-term memories |
| `evenement_relais` | Relays a declared event (press article, network incident) to an agent |

`python -m llm_gateway.cli categories` lists what is registered.

### 2.2 From task to answer

1. The controller posts a task to `POST /tasks` on the `api` service
   (`packages/llm_gateway/src/llm_gateway/api/routes.py`).
2. The task enters a Redis sorted set `batch:{key}`
   (`infra/redis/batch_queue.py`). The key groups tasks that may share a request: same
   category, same parameters, same pinned provider, same admitted instances. The score is
   the priority given by the category; for mode choice it is the earliest simulated
   departure time in the task, so urgent trips are served first.
3. A Celery worker (`worker/task_worker.py`) pops a batch, sized by the capacity of the
   selected provider (`batch_max_agents`, derived from its quotas at startup).
4. The balancer (`balancer/router.py`) chooses an instance. Two policies exist, selected
   by `LLM_GATEWAY_ROUTING__POLICY`: `swrr` (smooth weighted round robin, weights from
   `providers.yaml`) and `cascade` (use the first instance in order until it refuses).
5. The rate limiter (`infra/redis/rate_limiter.py`) reserves the request and its estimated
   tokens atomically, in a Lua script, against the per-minute limits (RPM, TPM) and the
   daily limits (RPD, TPD). No two workers can overrun a limit together.
6. The adapter (`adapters/`) calls the provider and validates the answer against the
   category's output schema.

### 2.3 Failures

| Event | Reaction |
|---|---|
| HTTP 5xx | Instance in cooldown for 60 s; the task is retried with exponential backoff |
| HTTP 429 (per-minute limit) | Cooldown taken from the provider's rate-limit reset headers when present |
| HTTP 429 (daily quota) | Instance paused until the computed reset time |
| HTTP 402 (no credit) | Instance disabled; `[ALARME]` logged once, on the rising edge |
| 30 consecutive errors (`disable_after_consecutive_errors`) | Instance excluded for `disable_timeout` seconds (180 by default) |

Two restrictions make a measurement attributable to one model:

- `force_provider` pins a task to one instance. A pinned task never switches model: if the
  instance cannot serve it, the task waits or fails, it is never rerouted.
- `instances_admises` restricts a task to a list of instances. A list that names nothing
  usable is an error at selection time, never "no restriction".

The client side (`sdk/client.py`) also has a circuit breaker: when the gateway is
saturated, the controller waits rather than degrading the decision.

In an experiment, a run stops rather than record degraded decisions: after three
consecutive fallbacks (`replis_consecutifs_max`), or when quotas are exhausted or the
providers overloaded.

### 2.4 The decision

The mode-choice answer is a probability for each offered option
(`packages/mobility_llm/src/mobility_llm/mode_choice.py`). The executed option is drawn by
`draw_index` from that distribution. Two seeds make the draw reproducible:
`option_order_seed` (order in which options are shown, 42) and `mode_draw_seed` (the draw,
42). A decision is therefore replayable from its recorded probabilities.

The prompt carries at most `max_trip_candidates` options (6): the fastest itinerary of each
mode group first, then the next fastest ones. The prompts themselves are in
`packages/mobility_llm/src/mobility_llm/prompts/prompts.yaml` and the category templates;
they are rendered by `packages/llm_gateway/src/llm_gateway/prompts/engine.py`.

Every exchange of a run (prompt, answer, provider, tokens) is appended to
`llm_exchanges.jsonl` in the run folder.

## 3. Agent lifecycle

### 3.1 Activity chain

Each persona has a daily activity chain (home, work, school, shopping, leisure...) built by
`services/llm-agents/chaine_activites.py`. The chain is cyclic: *n* activities produce *n*
trips, the last one returning home.

### 3.2 From trigger to decision

A trip is planned when one of three triggers fires for the agent:

- the agent is idle and its next departure falls within the planning horizon;
- the agent has just arrived (the arrival is reported on `/sync`);
- a timeout on a pending transit connection.

Planning jobs go to an earliest-deadline-first dispatcher: a priority queue (`_edf_heap`
in `urban_mobility_agents/simulation_controller.py`) ordered by simulated departure time
and consumed by `worker_concurrency` coroutines (8). Each job:

1. asks `CachedTripHelper` (`trip_helper/cached_triphelper.py`) for itineraries: OTP in
   `arriveBy` mode for transit, OSMnx for walk, bike and car;
2. applies the vehicle chain (Section 5) and records every option removed with its reason;
3. calls `LlmAgent.evaluate_and_choose_travel_plan`
   (`urban_mobility_agents/agents/llm_agent.py`), which builds the persona context, the
   memory block and the options, and submits the task to the gateway;
4. draws the executed option and pushes the decision to GAMA. The push happens at once if
   the agent is waiting, or at its next arrival if it is still travelling.

### 3.3 Bootstrap and horizon

At startup (`/init`), the controller precomputes the whole first cycle of every agent,
under a concurrency cap (`bootstrap_concurrency`, 30) so that the burst does not exhaust
the provider quotas. Afterwards a sliding 24-hour horizon keeps the queue filled: each
agent's decisions for the next simulated day are computed ahead of its departures.

### 3.4 Run folder

Each run writes to `experiments/archive/<YYYY-MM-DD>_<HH_MM>/` (link
`experiments/current`): the trip log `moves.csv`, the decisions, the LLM exchanges, the
memory checkpoints, the controller state (for `CONT=1` resumption), the logs, and the
per-day measurements under `mesures/` (see [EXPERIMENTS.md](EXPERIMENTS.md)).

## 4. Short-term and long-term memory

The memory design follows the generative-agent architecture of Park et al. (2023), as
adapted to transport by Vu et al. (2025), with changes listed below. Memory can be turned
off for a run with `make run MEM=0`.

### 4.1 Short-term memory

`UserShortTermMemory` (`services/llm-agents/llm/shortterm.py`) is a FIFO buffer of at most
100 entries per agent: trips made, options seen, perceived events. Each entry carries a
deterministic gravity score, a valence and its provenance.

The buffer is consolidated into long-term memory by an `stm_reflection` call when any of
these holds:

- it holds at least 10 entries;
- the cumulative gravity of its entries reaches 0.7;
- the daily floor is reached (22:00 simulated, `stm_reflection_daily_floor_hour`): any
  non-empty buffer is consolidated once a day, so that the timing of consolidation does not
  depend on how many trips an agent makes.

### 4.2 Long-term memory

`MultiUserLongTermMemory` (`services/llm-agents/llm/longterm.py`) stores consolidated
memories in ChromaDB, embedded with `all-MiniLM-L6-v2`. Before a decision, candidate
memories are gathered from three pools:

- **A**, semantic: nearest neighbours of the current situation;
- **B**, by offered mode: up to 8 memories per mode on offer, so that every option has a
  chance to be remembered;
- **C**, shocks: up to 5 memories tied to declared events.

Candidates are ranked by a score whose five weights sum to 1:

| Component | Weight | Definition |
|---|---|---|
| Semantic similarity | 0.30 | `exp(cos − 1)` of the embeddings |
| Weather match | 0.10 | same weather as now |
| Recency | 0.20 | `exp(−Δt / force)`, with a base force of 2.8 days, lengthened by gravity and by recall |
| Gravity | 0.20 | deterministic gravity of the memory |
| Axis affinity | 0.20 | shared object, place, time slot or purpose (a bonus, never a veto) |

The top 10 feed the decision. The prompt receives a core memory block in three parts
("My habits", "What I know", "What changed recently") and three episodic traces.

An `ltm_self_reflection` call runs every 3 simulated days over the memories of the last 5
days.

<!-- TODO: point to the exact settings that hold the pool sizes (8, 5), the top-K (10) and
     the self-reflection period, once their public names are frozen. -->

## 5. The vehicle chain

`services/llm-agents/urban_mobility_agents/vehicle_chain.py` keeps track of where each
private vehicle is, and removes the options that would use a vehicle the agent does not
have at hand.

- **State.** `planning_vehicle_at` maps each vehicle mode of a person to where the vehicle
  is parked. It is updated when a trip with that vehicle is chosen: the vehicle follows its
  user.
- **Exit lock.** A car or bike option is offered only if the household owns the vehicle and
  the vehicle is at the trip origin. Driving also requires a licence and an age of 18 or
  more. A car trip for a non-driver is kept as a passenger trip when an adult of the
  household drives; the car is then not parked at the destination.
- **Return lock.** On a trip home, if one of the agent's vehicles is parked at the trip
  origin, the options are restricted to that mode, so that the vehicle comes back. The lock
  is not applied to returns under 1 km (`RETURN_LOCK_MIN_DISTANCE_KM`); a vehicle left behind
  is brought home at the end of the cycle.
- **Recorded motives.** Every removed option is logged with its motive: `non_possede` (not
  owned), `pas_de_conducteur` (no driver), `vehicule_ailleurs` (vehicle elsewhere). The trip
  log `moves.csv` carries the column `Contrainte de chaîne`, and the archived decisions carry
  the removed options.

The chain constrains the offer; it does not score compliance. A decision is measured on the
options that remained.

## 6. Routing and caches

### 6.1 Routers

| Modes | Router | Notes |
|---|---|---|
| Transit (urban network, regional trains, regional coaches) | OpenTripPlanner 2.8.1, GraphQL Transmodel API | Three instances behind the controller; graph built by `make otp-graph` from `data/gtfs/` |
| Walk | OSMnx graph | Cut off at 15 km |
| Bike | OSMnx graph | Cut off at 30 km |
| Car | OSMnx graph | Travel time scaled by a congestion factor per hour, weekday and zone of the edge (city, agglomeration, outside), from `services/llm-agents/config/osmnx.yaml` |

The OSMnx graphs cover the polygon of the 453 municipalities of the survey. They are built
once by `make osmnx-perimeter-graph` and stored under the key `444ca7e6a515`
(`data/cache/osmnx/graphs_444ca7e6a515.pkl`). Speeds by road type and penalties are in
`services/llm-agents/config/osmnx.yaml`. Access and egress times for car and bike come from
the survey-fitted law in `packages/mobility_core/src/mobility_core/data/terminal_time_emc2.json`.

### 6.2 Caches

| Cache | Store | Key | Invalidated by |
|---|---|---|---|
| LLM decisions (semantic) | Qdrant, embedded (`services/llm-agents/llm/cache.py`) | embedding of the decision context; stores the probability distribution, not a drawn mode | `make run CACHE=0`; it also switches itself off on event days |
| OTP itineraries | SQLite (`trip_helper/otp_persistent_cache.py`) | origin, destination, time | a new graph |
| OSMnx itineraries | SQLite (`trip_helper/osmnx_persistent_cache.py`) | prefixed by the routing version | a change of routing version |
| Road graphs | pickles under `data/cache/osmnx/` | polygon key | `make osmnx-perimeter-graph` |

Because the semantic cache stores a distribution, a cache hit still goes through the seeded
draw. Experiments on frozen trip sets do not use the routers at all: the options come from
`propositions.jsonl` (see [EXPERIMENTS.md](EXPERIMENTS.md)).

<!-- TODO: confirm the exact key of the OTP cache (rounding of coordinates and time). -->

## 7. Backpressure

GAMA advances faster than the providers can answer. The controller slows it down by
holding the response to `/sync`. No hold ever exceeds 30 s (`min_internal_coeff_cap`),
GAMA's HTTP read timeout. The functions are in `services/llm-agents/backpressure.py`.

1. **Predictive EDF feasibility** (default, `predictive_backpressure_enabled`). The
   controller estimates its completion rate with an exponential moving average
   (`throughput_ewma_tau_s`, 90 s). For each pending job *k* it compares the estimated time
   to reach it, *T_k*, with its slack to the departure time, *slack_k*. It holds `/sync` when
   `T_k × 1.4 > slack_k` (`predictive_margin`).
2. **Progressive brake** (the older mechanism, used when the predictive control is off):
   `min_interval = cap × min(1, backlog / population)^k`, with `cap = 30` s and `k = 1.5`.
   At 10 % backlog this gives about 1 s; at 50 %, about 10.6 s.
3. **Drain mode with hysteresis**, the last safety net: when the backlog reaches 80 % of the
   population (`drain_trigger_ratio`), every `/sync` is held up to the cap until the backlog
   falls below 20 % (`drain_release_ratio`). The backlog alarm fires and clears on the same
   thresholds.
4. **Imminent-departure hold**, in experiments only: while a departure decision is missing
   and the departure falls within 3,600 simulated seconds (`departure_hold_lookahead_s`),
   `/sync` is held. If releasing GAMA would make it pass the departure time, the run stops
   instead of serving a late trip.

When the cumulative hold on a `/sync` passes 5 s, GAMA receives a `system/throttle`
notification (rising edge, refreshed every 30 s while the degraded regime lasts).

Every threshold crossing is logged at ERROR level with the prefix `[ALARME]`, on the rising
edge. `make error` lists them; `make report` and `make capacity` put them in context.

## 8. Where to look in the code

| Concern | Path |
|---|---|
| HTTP endpoints of the controller | `services/llm-agents/handle/application.py` |
| WebSocket client | `services/llm-agents/handle/websocket.py` |
| Dispatcher, triggers, backpressure wiring | `services/llm-agents/urban_mobility_agents/simulation_controller.py` |
| Backpressure functions | `services/llm-agents/backpressure.py` |
| Candidate options | `services/llm-agents/urban_mobility_agents/candidats.py` |
| Vehicle chain | `services/llm-agents/urban_mobility_agents/vehicle_chain.py` |
| Decision call | `services/llm-agents/urban_mobility_agents/agents/llm_agent.py` |
| Routing clients and caches | `services/llm-agents/trip_helper/` |
| Memory | `services/llm-agents/llm/` (`shortterm.py`, `longterm.py`, `noyau.py`, `gravite.py`) |
| Events | `services/llm-agents/llm/evenements/` and `services/llm-agents/config/evenements/` |
| Controller settings | `services/llm-agents/settings.py`, `services/llm-agents/config/config.yaml` |
| Gateway: API, batching, balancer, worker | `packages/llm_gateway/src/llm_gateway/{api,core,balancer,worker}/` |
| Gateway: ports and adapters | `packages/llm_gateway/src/llm_gateway/{ports,infra,adapters}/` |
| Mobility categories and mode choice | `packages/mobility_llm/src/mobility_llm/` |
| Survey domain (rings, mode hierarchy, equipment laws) | `packages/mobility_core/src/mobility_core/` |
| GAMA model | `services/GAMA/CityTransport/models/` |

The three packages keep their dependencies one-way: `llm_gateway` imports neither
`mobility_core` nor `mobility_llm`, and `mobility_core` imports neither the gateway nor any
infrastructure library. The contracts are in `.importlinter`; `make lint-imports` checks
them.
