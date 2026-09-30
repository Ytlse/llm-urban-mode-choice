# Add a category

A category is a request type: a prompt template, an output schema, an
item model and, if needed, a priority rule and metrics. The gateway contains
none. They are shipped by a **bundle** — a Python package that declares a
`CategoryBundle` under the `llm_gateway.categories` entry point.

Two real examples serve as models: `llm_gateway.testing` (one `echo` category, the
minimum) and `mobility_llm` (four categories, a rich item model, priority, metrics).

## What a bundle provides

```python
from llm_gateway import CategoryBundle, CategorySpec

CategoryBundle(
    name="mon_bundle",                    # unique name; shows in the logs and in `llm-gateway categories`
    templates_dir=Path(".../categories"), # root of the templates (one or more folders)
    prompts_file=Path(".../prompts.yaml"),# facultatif : variantes de prompt système (active: / prompts:)
    categories={
        "ma_categorie": CategorySpec(
            name="ma_categorie",
            template_name="ma_categorie/template.md.j2",              # relatif à templates_dir
            schema_path=Path(".../categories/ma_categorie/output_schema.json"),
        ),
    },
)
```

Two layouts are possible, and coexist: **per folder** (above, what
`mobility_llm` does: `categories/<nom>/template.md.j2`, `output_schema.json` and the hook code
side by side) or **flat** (`templates_dir/<nom>.md.j2` and a JSON `schemas_file`
`{catégorie: schéma}`, what the `echo` category of `llm_gateway.testing` does). Without
`template_name`, the engine looks for `<nom>.md.j2`; without `schema_path`, it reads `schemas_file`.

A `CategorySpec` carries five things, all optional except the name:

| Field | Default | Role |
|---|---|---|
| `item_model` | `AgentItem` (an `agent_id`, free fields kept) | pydantic model that validates each item of the payload; an invalid item → 422 on the API side |
| `priority` | `None` (no priority) | `Callable[[Sequence[BaseModel]], float | None]`: score of the batch, **lower = more urgent** |
| `observe` | `None` | `Callable[[ObserveContext], None]`: called by the worker after validation of the response, to count domain metrics in `ctx.metrics` |
| `template_name` | `<nom>.md.j2` | template name, relative to `templates_dir` |
| `schema_path` | `None` (reads `schemas_file`) | JSON file of this category's output schema |

The counters that `observe` feeds are exposed in Prometheus by declaring them in
`CategoryBundle.metric_families` (a `MetricFamilySpec`: name, help, Redis prefix, labels);
the API collector renders them without knowing anything about the domain.

## Step 1 — the template

`categories/ma_categorie/template.md.j2` (or `templates/ma_categorie.md.j2` flat), Jinja2 (`trim_blocks`, `lstrip_blocks`, no autoescape on
`.md.j2`). Two markers split the rendering into messages; without a marker, everything becomes a
`user` message.

```jinja
<!-- SYSTEM -->
Tu réponds uniquement avec le JSON attendu.

Schéma JSON attendu :
{{ schema }}

<!-- USER -->
{% for agent in agents %}
agent_id={{ agent.agent_id }}
{% for key, value in agent.items() if key != 'agent_id' and value %}{{ key }} : {{ value }}
{% endfor %}
{% endfor %}
```

Available variables: `agents` (the validated items, as dicts), `agent_ids`, `parameters`
(the request dict), `schema` (the serialised JSON Schema, indented), `system_prompt` (the
active variant of `prompts.yaml`, or the one requested by `parameters.prompt_variant`, or
`""`).

Two conventions matter for the worker:

- **write `agent_id=<valeur>` at the head of each item block**: this is what `FakeAdapter`
  spots to make one response per agent in the tests, and what the model copies
  into its response;
- **inject `{{ schema }}`** into the system message: the adapters also send it as *structured
  output* when the API allows it, but the text remains the only instruction for Mistral.

## Step 2 — the output schema

`categories/ma_categorie/output_schema.json` (or an entry of a flat `schemas.json`). The model's response must be an object with an
`agents` key (list), each element carrying `agent_id`; the gateway parser
(`_parse_output`) requires this shape, then each element is validated as `AgentResponse`
(`extra="allow"`: your fields travel all the way to the client).

```json
{
  "ma_categorie": {
    "type": "object",
    "properties": {
      "agents": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {"agent_id": {"type": "string"}, "summary": {"type": "string"}},
          "required": ["agent_id", "summary"]
        }
      }
    },
    "required": ["agents"]
  }
}
```

## Step 3 — `prompts.yaml` (optional)

To vary the system prompt without touching the template:

```yaml
active:
  ma_categorie: v2
prompts:
  v1:
    content: "Première rédaction…"
  v2:
    content: "Rédaction du 2026-09-01… Schéma JSON attendu : {…}"
```

`active[catégorie]` designates the served variant; a request can impose another one with
`parameters.prompt_variant` (unknown → `ValueError`, never a silent substitution).
The block "Schéma JSON attendu : …" at the end of `content` is removed at rendering: it is
`{{ schema }}` that injects it. `PromptManager.active_prompt_checksum()` gives a fingerprint
of the active variants, useful as a cache key on the consumer side.

!!! warning "A template that renders `{{ system_prompt }}` without text of its own"
    If the category disappears from `active:`, `system_prompt` is `""` and the system message
    shrinks to the schema, without error. `itinary_multi_agent.md.j2` is in this case: keep
    `active.itinary_multi_agent` filled in.

## Step 4 — the item model and the hooks

```python
from pydantic import BaseModel, ConfigDict
from llm_gateway import ObserveContext

class MonItem(BaseModel):
    model_config = ConfigDict(extra="ignore")   # ou "allow" : à vous de choisir
    agent_id: str
    texte: str
    urgence: float | None = None

def priorite(items):                             # lower = more urgent; None = no priority
    scores = [i.urgence for i in items if i.urgence is not None]
    return min(scores) if scores else None

def observe(ctx: ObserveContext) -> None:        # ctx.category, ctx.provider, ctx.items, ctx.output, ctx.metrics
    for rep in ctx.output.agents:
        ctx.metrics.incr(f"ma_metrique:{ctx.provider}")
```

`observe` modifies neither the items nor the output. An exception in the hook is logged
as WARNING and **never fails the batch**. The counters written into `ctx.metrics` live in
the Redis hash `wmetrics`; for them to come out on `/metrics`, a family is needed in
`WorkerMetricsCollector` (`api/metrics.py`) — today the mobility domain families
are hard-coded there (assumption H11 of ticket 037; exposure declared by the bundle
will come later).

Real example: `mobility_llm.persona.AgentSpec` (item), `departure_priority` (smallest
`departure_timestamp`) and `categories.itinary_multi_agent.observe_itinary` (most
probable mode, probability mass per mode, distance bands, label disagreements).

## Step 5 — the entry point

In the `pyproject.toml` of your package:

```toml
[project.entry-points."llm_gateway.categories"]
mon_bundle = "mon_paquet:bundle"
```

where `bundle()` returns the `CategoryBundle` (an already built object is accepted too). Do not
forget the data files in `[tool.setuptools.package-data]`:
`prompts/*.yaml`, `categories/*/*.j2`, `categories/*/*.json`. Reinstall (`pip install
-e .`) so that `importlib.metadata` sees the entry point.

## What is checked at startup

`CategoryRegistry` is built by `build_deps` (API) and `build_worker_runtime` (worker),
from all the entry points found. For each declared category:

- missing template → `ValueError: Catégorie 'x' : template x/template.md.j2 absent de […]`;
- missing schema (neither `schema_path` nor an entry in `schemas_file`) → `ValueError: Catégorie 'x' : schéma de sortie absent (…)`;
- same name in two bundles → `ValueError: Catégorie 'x' déclarée par deux bundles (…)`;
- entry point that does not return a `CategoryBundle` → `TypeError`.

The process refuses to start: better that than a failing first request.
`llm-gateway categories` replays exactly this construction.

## Step 6 — test without Redis or network

`llm_gateway.testing` is made for this. The reference test is
`mobility_llm/tests/unit/test_bundle_end_to_end.py`; in short:

```python
from llm_gateway.core.models import InternalRequest
from llm_gateway.testing import FakeAdapter, InMemoryMetricsSink, build_registry
from mon_paquet import bundle

def test_du_payload_aux_metriques():
    registry = build_registry(bundle())                       # builds and VALIDATES the bundle
    handle = registry.get("ma_categorie")
    items = handle.validate_items([{"agent_id": "a1", "texte": "bonjour"}])
    messages = handle.render(items, {})
    assert messages[0].role == "system" and "agent_id=a1" in messages[-1].content

    fake = FakeAdapter(responder=lambda aid: {"agent_id": aid, "summary": "ok"})
    output, tokens_in, tokens_out = fake.call(
        InternalRequest(provider="fake", messages=messages, response_schema=handle.output_schema)
    )
    metrics = InMemoryMetricsSink()
    handle.observe("fake", items, output, metrics)
    assert metrics.get("ma_metrique:fake") == 1
```

`build_registry()` without argument loads only the `echo` bundle. To test the full API in
memory (202, 422, `/health`), reuse the `memory_deps` fixture of
`tests/integration/conftest.py`: `create_app(settings, deps=GatewayDeps(…, registry=build_registry(bundle())))`.

## What the gateway will not do for you

It does not know what your items represent, it does not read your counters back on `/metrics`
without a declared family, and it has no output format other than an `agents` list
aligned on `agent_id` (`AgentResponse` also keeps the mobility fields `probabilities`,
`chosen_index`, `mode`, `reason`, `summary` — assumption H5, made generic in
iteration 2). A malformed returned `agent_id` ("PERSONA 446264") is realigned on its numeric
part; an unknown `agent_id` is lost and logged as ERROR.
