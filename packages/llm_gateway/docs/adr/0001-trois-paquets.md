# ADR 0001 — `llm_module` becomes three packages

- **Date**: 2026-09-06
- **Status**: accepted (decision of the repository author, ticket 037)
- **Scope**: `llm_module/` of the `llm-agents-gama` monorepo

## Context

`llm_module` was halfway to being a library: ports & adapters architecture since
the July 2026 overhaul, `pyproject.toml`, import-linter contracts, 504 green tests. But
it was two libraries in a single package — a generic LLM gateway and a
"Toulouse EMC² mobility" domain — and the second had colonised the first.

Measurements of 2026-09-06: 13,513 Python lines including tests; `core/` had 3,352
lines of which only 254 were generic (models, batching, selection); ≈ 2,250 of the 4,700
test lines were about the mobility domain; of 117 imports from `llm-agents`,
`scripts` and `prompt_calibration`, 95 targeted `core` and **only one** the gateway SDK.

Concrete leaks: the worker imported the mode choice, counted mode disagreements and
binned distances into bands; `AgentSpec` imposed perception, trajectories and
feelings on every category; prompts and schemas were shipped in the package; the "pure"
`core/` climbed two levels up to the repository root and read `/app/scripts/...`.
Component reflexes rather than library ones: empty `__init__.py`, `logger.remove()`
wiping the host's handlers, Prometheus metrics declared at SDK import,
`providers.yaml` rewritten inside the installed package, unprefixed environment variables,
versioned egg-info, shims without deprecation.

## Decision

Three installable packages, sibling directories at the repository root, `src/` layout:

| Package | Content | Depends on |
|---|---|---|
| `llm_gateway` 1.1.0 | api, worker, infra, ports, adapters, balancer, prompt engine **without content**, telemetry, SDK, config | nothing in the repository |
| `mobility_core` 0.1.0 | rings, fine zones, mode hierarchy, bicycle, housing, equipment, population calibration, frozen data | nothing in the repository |
| `mobility_llm` 0.1.0 | persona, templates, schemas, prompt variants, mode choice, domain metrics; registers with the gateway through an entry point | the other two |

`llm_module/` remains a compatibility shell: each old module re-exports its
successor and emits a `DeprecationWarning`; removal at the next major version (2.0). The
imports of `llm-agents` and `scripts` are rewritten; `prompt_calibration` (standalone
repository) stays on the shell.

Assumptions made with the decision (`specs/ticket_037/questions.md`): names `llm_gateway`,
`mobility_core`, `mobility_llm` (H1); three directories in the monorepo, extraction
later (H2); French for code, docstrings and documentation (H3); no `LICENSE` until
the licence is chosen (H4, open question); the response contract `AgentResponse`
stays typed "options" in the gateway so as not to break the HTTP contract (H5); the
templates, schemas and `prompts.yaml` keep their flat layout in
`mobility_llm/prompts/` because `prompt_calibration` and the experiments cite
`prompts.yaml` (H6); the Celery worker stays in `llm_gateway/worker/` (H7);
`providers.yaml` remains data of the `llm_gateway` package (H8).

## Consequences

- A consumer of the gateway installs `llm-gateway` alone and embeds neither geopandas, nor the
  survey resources, nor a prompt. Without a bundle, every category is refused with 422.
- The EMC² domain is usable without an LLM: eqasim and the notebooks import
  `mobility_core` without pulling Redis, Celery, httpx, FastAPI or Jinja2 (import-linter contract).
- Five architecture contracts replace the previous two; `lint-imports` runs in CI.
- Three test suites (233 · 199 · 134 on 2026-09-07), three `CHANGELOG.md`, three
  READMEs; one mkdocs site for the gateway.
- The `api`/`worker` Docker image has the repository root as context and embeds the three
  packages: the worker needs the bundle and its domain.
- Accepted cost: a `git mv` of all the code, rewritten imports in
  `llm-agents` and `scripts`, a shell to maintain until 2.0. `ruff format` is deferred
  (H10) so that git detects the renames.

## Alternatives rejected

- **Two packages** (gateway + mobility): the survey domain would have kept
  depending on the gateway, or the reverse. eqasim does not need an LLM gateway to draw a
  bicycle.
- **Staying as one package with sub-packages and import-linter contracts**: the
  contracts already existed and did not prevent the colonisation; a single `pyproject.toml`
  imposes everyone's dependencies on each one.
- **Extracting the gateway into a separate repository right away**: depends on versioning and
  distribution, put off to a second stage by the author.
- **Authentication, generic adapter, executor without Celery in the same piece of work**:
  postponed (ticket 036 and later iterations of 037) to keep iteration 1 readable.
