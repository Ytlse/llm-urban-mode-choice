# Changelog — mobility-llm

Format: `## [version] - YYYY-MM-DD`, most recent entries first. The tone is that of use. The
files touched are in git.

## [Unreleased]

**Test-only category `evenement_intention`** (ticket 120): after reading a press article, will
the persona change how it travels over the next 7 days (`no_change`, `some_trips`,
`most_trips`, the modes used less and more, one sentence)? No simulation code calls it; it
serves the press gravity test. It is a separate call from `evenement_jugement`, whose template
forbids planning.

**Declared licence** (ticket 038): `Apache-2.0`, with `LICENSE` and `NOTICE` shipped. No
reservation — the prompts and templates cite municipalities and network names, they
reproduce no data from the EMC² survey. The reservations on the survey live in the
`NOTICE` of `mobility-core`, which this package depends on.

## [0.2.0] - 2026-09-07

Each category stores its parts in `categories/<nom>/`: `template.md.j2`,
`output_schema.json`, and `observe.py` for the itinerary. `prompts/schemas.json` and
`prompts/templates/` are removed; `prompts/prompts.yaml` stays in the same place. The bundle
declares its Prometheus families (`METRIC_FAMILIES`) under their current names, and
`build_prompt_manager(prompts_file=…)` builds an engine with another set of variants.
Requires llm-gateway ≥ 1.3.

## [0.1.0] - 2026-09-07

First version: the LLM categories of the mobility simulation, moved out of the gateway
(ticket 037). The package is the glue between `llm_gateway` (generic) and `mobility_core`
(survey domain).

**What the package provides:**

- **A bundle** registered under the entry point `llm_gateway.categories`
  (`mobility = "mobility_llm:bundle"`): four categories — `itinary_multi_agent`,
  `perception_filter`, `stm_reflection`, `ltm_self_reflection` — with their templates
  `.md.j2`, `schemas.json` and the system prompt variants of `prompts.yaml`.
- **`AgentSpec`**, the persona (formerly `llm_module.core.models.AgentSpec`): it is what says
  what an item must carry (`agent_id`, `perception`, options, weather, agenda…). A persona
  without `perception` is rejected by the API with 422; an undeclared field is ignored, as
  before the split.
- **The priority of a batch** = smallest `departure_timestamp` (`departure_priority`).
- **The worker's domain metrics** (`categories.itinary_multi_agent.observe_itinary`):
  most probable mode, probability mass per canonical mode, distance brackets,
  mode label disagreements with an alarm above 5% over 200 options. The Redis counter
  names are unchanged: Grafana dashboards 04 and 07 keep reading them.
- **The mode choice** (`mode_choice`): normalisation of the probability vector, projection
  onto the canonical modes, deterministic draw with a seed derived from the context.
- **`prompt_manager()`** for consumers outside the gateway (controller, experiments):
  checksum of the active prompt, list of variants.

**Before:** the gateway worker imported the mode choice and computed these counters
itself; changing a mobility rule modified the gateway.
**After:** the worker calls `CategorySpec.observe` without knowing what a transport mode
is; an error in the hook is logged and never fails the batch.

Tests: 134 collected on 2026-09-07 (the e2e test is skipped without `LLM_GATEWAY_E2E_URL`).
`tests/unit/test_bundle_end_to_end.py` is the proof of adoption of `llm_gateway.testing`:
registry built by hand, `FakeAdapter`, in-memory metrics sink.
