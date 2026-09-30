# mobility-llm

The LLM categories of the mobility simulation. This package is the glue between the generic
gateway (`llm_gateway`) and the EMC² survey domain (`mobility_core`): it provides the
persona model, the templates, the output schemas, the system prompt variants, the
draw of the mode from the distribution returned by the LLM and the worker's domain metrics.
Version 0.1.0 (ticket 037).

## The bundle

The gateway discovers this package through the entry point declared in `pyproject.toml`:

```toml
[project.entry-points."llm_gateway.categories"]
mobility = "mobility_llm:bundle"
```

`bundle()` returns a `CategoryBundle` named `mobility`: each category lives in
`categories/<nom>/` (`template.md.j2`, `output_schema.json`, and `observe.py` for the
itinerary domain metrics); the prompt variants stay in `prompts/prompts.yaml`. Four
`CategorySpec`:

| Category | Item | Batch priority | `observe` hook |
|---|---|---|---|
| `itinary_multi_agent` | `AgentSpec` | smallest `departure_timestamp` | `observe_itinary`: most probable mode, probability mass per canonical mode, distance bracket, mode label disagreements |
| `perception_filter` | `AgentSpec` | same | — |
| `stm_reflection` | `AgentSpec` | same | — |
| `ltm_self_reflection` | `AgentSpec` | same | — |

The gateway validates each category at startup (template `<catégorie>.md.j2` and entry in
`schemas.json` present). `llm-gateway categories` lists the four with `bundle=mobility,
items=AgentSpec`.

## Modules

| Module | One line |
|---|---|
| `mobility_llm/__init__.py` | the bundle (`bundle()`, `CATEGORIES`, `BUNDLE_NAME`) and `prompt_manager()` for consumers outside the gateway |
| `persona.py` | `AgentSpec`, the item of the mobility categories (formerly `llm_module.core.models.AgentSpec`) and `departure_priority` |
| `categories/itinary_multi_agent.py` | what the worker observes of an itinerary decision: Redis counters `transport_mode_chosen:*`, `mode_probability_pct:*`, `trip_distance_bracket:*`, `mode_by_distance:*`, `mode_by_provider:*`, `chosen_index:*`, `mode_label_checked` / `mode_label_mismatch`; alarm above 5% of disagreements over 200 options |
| `mode_choice.py` | from the probability vector to the drawn mode: `normalize_option_probabilities`, `mode_distribution` over the canonical modes, `draw_index` (deterministic seed derived from the context), `argmax_index`, `canonical_mode` |
| `prompts/__init__.py` | the paths: `TEMPLATES_DIR`, `SCHEMAS_FILE`, `PROMPTS_FILE` |

## `AgentSpec`

A persona in a decision batch. Required: `agent_id`, `perception`. Optional:
`destination`, `destination_zone`, `departure_time`, `departure_timestamp` (Unix, priority
score), `current_time`, `context`, `history`, `trajectories`, `goal`, `constraints`,
`feeling`, `day_outlook`, `agenda`. `extra="ignore"`: an undeclared field is silently
lost, as before the split — declaring the field here is the only way to make it
travel to the template. A persona without `perception` is rejected by the API with 422.

## Prompts, schemas, templates

Everything is flat in `src/mobility_llm/prompts/` (hypothesis H6 of ticket 037:
`prompt_calibration` and the experiments cite `prompts.yaml`):

- `templates/<catégorie>.md.j2` — markers `<!-- SYSTEM -->` / `<!-- USER -->`,
  `{{ schema }}` injected from `schemas.json`. `itinary_multi_agent.md.j2` has **no**
  system text of its own: it renders `{{ system_prompt }}`, hence the active variant.
- `schemas.json` — one output JSON Schema per category (`agents: [{agent_id, …}]`).
- `prompts.yaml` — `active: {itinary_multi_agent: expert_m4}` and `prompts: {clé:
  {content: …}}`; eighteen variants (`expert_m4` and its lineage `expert_chaine_m5`…`m7.1`,
  `expert_m1`, `prompt_minimal`, `b_min`, `expert_gem_3.8_v2`, plus the archived `expert`,
  `b0_pristine`, `persona_v1`…`v5`, `minimal_persona`, `expert_gem_3.8_v1`).
  An entry carrying `_archive: {statut: archive}` is no longer offered by the dashboard
  but **remains servable**: it is a withdrawal from the choice, not a refusal of service. The
  refusal of service comes from `_invalidation` or from a non-compliant or outdated `_neutralite`.

**Adding a prompt variant**: an entry under `prompts:` with `content:`; designate it either
as active (`active.itinary_multi_agent`), or per request
(`parameters.prompt_variant`, which enters the gateway's batch key). The block "Schéma JSON
attendu : {…}" at the end of `content` is removed at rendering. An unknown variant raises
`ValueError`: never a silent substitution. `prompt_manager().variantes()` lists the
keys, `prompt_manager().active_prompt_checksum()` gives the fingerprint of the active prompt
(key of the controller's LLM cache).

## For Python consumers

```python
from mobility_llm import prompt_manager, AgentSpec
from mobility_llm.mode_choice import normalize_option_probabilities, draw_index

pm = prompt_manager()                       # the gateway's PromptManager loaded with the mobility content
pm.active_prompt_checksum("itinary_multi_agent")
weights = normalize_option_probabilities(result.agents[0].probabilities, n_options, modes=modes)
index = draw_index(weights, agent_id, sim_day)   # même contexte → même tirage
```

## Installation and tests

```bash
pip install -e ./mobility_core -e ./llm_gateway -e ./mobility_llm[test]   # depuis la racine du dépôt
cd mobility_llm && pytest -m "not e2e"                                    # ou `make test-mobility` à la racine
LLM_GATEWAY_E2E_URL=http://localhost:8000 pytest -m e2e                   # perception_filter on a real gateway
```

134 tests collected on 2026-09-07 (`unit`: prompt rendering, variants, persona, mode
choice, itinerary metrics; `e2e`: skipped without the variable).
`tests/unit/test_bundle_end_to_end.py` is the usage example of `llm_gateway.testing`:
`build_registry(bundle())`, `FakeAdapter(responder=…)`, `InMemoryMetricsSink`, from payload
to metrics hook without Redis or network.

Architecture contract (import-linter, `.importlinter` at the root): this package does not
import the `llm_module` shell. History: `CHANGELOG.md`.
