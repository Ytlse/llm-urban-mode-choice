# ADR 0002 — The gateway discovers its categories through a pluggable hook

- **Date**: 2026-09-06
- **Status**: accepted (ticket 037, iteration 1, point 4 "minimal category hook")
- **Scope**: `llm_gateway.ports.category`, `llm_gateway.prompts.registry`, the API and the worker

## Context

Once the domain was taken out of the gateway ([ADR 0001](0001-trois-paquets.md)), we had to decide
how the gateway finds what it no longer contains: the template and schema of a
category, the model that validates an item, the priority rule of a batch, the domain metrics
the worker used to compute itself (`llm_transport_mode_chosen_total`, mode disagreements,
distance bands).

Constraints: the gateway must import no domain package (contract
`gateway-sans-metier`); the HTTP contract consumed by the GAMA controller must not move;
a badly declared category must fail at startup, not at the first request; the
Grafana dashboards read the existing Redis counters.

## Decision

A **bundle** is a `CategoryBundle` object (name, templates directory, schemas file,
optional `prompts.yaml`, dictionary of `CategorySpec`) provided by a third-party package
under the `llm_gateway.categories` **entry point**:

```toml
[project.entry-points."llm_gateway.categories"]
mobility = "mobility_llm:bundle"
```

A `CategorySpec` carries three hooks, all optional: `item_model` (pydantic model of the
item; default `AgentItem`, an `agent_id` and free fields kept), `priority`
(score of a batch, lower = more urgent; default none), `observe` (called after validation
of the response with an `ObserveContext` — category, provider, validated items, output,
`MetricsSink`).

The **registry** (`CategoryRegistry`) is built explicitly by `build_deps` (API) and
`build_worker_runtime` (worker) from the entry points, or passed by hand
(`llm_gateway.testing.build_registry`). At construction, for each category: template
and schema present, otherwise `ValueError`; the same name in two bundles → `ValueError`.

The **API** resolves the category, validates the items with `item_model` and computes the priority
before persisting the task: unknown category or invalid item → 422. The **worker**
asks the handle to validate, render and provide the schema, calls the adapter, then
`handle.observe(...)`; an exception in `observe` is logged and never fails the
batch. The worker no longer imports the mode choice.

Associated assumptions: the gateway's Prometheus collector keeps the mobility domain
families hard-coded (H11), the `sim_ts` of the exchange log becomes the priority score
of the batch (H9), the response contract stays typed "options" (H5), the `echo` bundle of
`llm_gateway.testing` serves as a minimal bundle and the proof of adoption of the test tooling
is `mobility_llm/tests/unit/test_bundle_end_to_end.py` (H12).

## Consequences

- With no bundle installed, `llm-gateway categories` returns 1 and every request is refused with 422
  with the (empty) list of categories: the gateway no longer has a "default" behaviour
  that would hide an incomplete installation.
- Adding a category does not touch the gateway: a package, an entry point, some files.
- The registry validates at startup: a deployment with a missing template does not
  start.
- `mobility_llm` must be installed in the environment of the API **and** of the worker (the Docker
  image embeds the three packages).
- Known limit: the counters written by `observe` only come out on `/metrics` if the
  API collector knows the family; exposure declared by the bundle will come with
  genericity (next iteration). Same for the output: a single shape (`agents`
  aligned on `agent_id`).

## Alternatives rejected

- **A registry configured by file (YAML listing template paths)**: solves
  neither item validation nor domain metrics, which are code.
- **Worker subclasses per category**: imposes Celery on the domain and one worker per
  bundle.
- **Keeping `AgentSpec` in the gateway with all fields optional**: the gateway
  would keep saying what a persona is; a documentation bundle would have no use for it.
- **Discovery by importing a module named in an environment variable**: less
  standard than the entry point, which `importlib.metadata` serves without configuration and which
  `pip install` is enough to activate.
