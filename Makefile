# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# A single run configuration file, no more choice by variable: to
# change config, edit services/llm-agents/config/config.yaml directly.

GAMA_BIN        = /Applications/GAMA.app/Contents/MacOS/GAMA
# Repository root, deduced from the Makefile's location (no hard-coded absolute path).
# ⚠ `firstword`, NOT `lastword`: MAKEFILE_LIST grows with each `include` below, and
# `lastword` would then designate the last included .mk — hence `make/` instead of the root.
# All path resolution in the repository switches with this variable, silently.
PROJECT_ROOT   := $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
WORKSPACE       = $(PROJECT_ROOT)/services/GAMA/CityTransport

# ── Docker stack (ticket 039) ─────────────────────────────────────────────────
# The compose file lives in infra/; `--project-directory` keeps the ROOT as the
# base for relative paths. Without it, compose re-anchors everything on infra/ and fails at
# `.env` (verified: COMPOSE_FILE and COMPOSE_PROJECT_DIRECTORY are not enough).
COMPOSE_FILE_PATH = infra/docker-compose.yml
COMPOSE           = docker compose -f $(COMPOSE_FILE_PATH) --project-directory $(PROJECT_ROOT)

# The project venv lives in the controller service (ticket 039: services/llm-agents).
# ⚠ ALWAYS invoke it as `$(VENV_PYTHON) -m <tool>`: the scripts in `.venv/bin/`
# carry an absolute shebang that the move made stale.
# ABSOLUTE path: targets that `cd` into a package thus have no `../` to count
# — that counting is what broke when `packages/` was inserted (ticket 039).
VENV_PYTHON       = $(PROJECT_ROOT)/services/llm-agents/.venv/bin/python

MODEL_PATH      = $(WORKSPACE)/models/City.gaml
EXPERIMENT_NAME = e

# ── Offline mode: headless GAMA in a container ────────────────────────────────
# `make run OFFLINE=1` (or the alias `make run-offline`): GAMA runs in the
# compose service `gama` (profile "offline", image gamaplatform/gama) instead of
# the local GUI. The launcher scripts/gama/launch_headless.py drives load/play
# via the GAMA Server protocol (port 6868).
# NB: `make run --offline` is not valid make syntax — use OFFLINE=1.
OFFLINE ?=
ifneq ($(OFFLINE),)
  export COMPOSE_PROFILES = offline
  export GAMA_WS_URL = ws://gama:3001
endif

# ── Hot resume ─────────────────────────────────────────────────────────────────
# `make run OFFLINE=1 CONT=1`: resumes the previous run instead of creating a
# new one — the controller reuses the workdir pointed to by experiments/current
# (logs appended, state.json and checkpoints recovered) and the
# Grafana/Prometheus/Redis data are KEPT. The GAMA simulation restarts at t0 of the
# simulated day (no state freeze on the GAMA side, cf. ticket 002); the caches make
# the replay near-instant. Prior hot stop: `make stop-run`.
CONT ?=
ifneq ($(CONT),)
  export CONTINUE_RUN = 1
endif
# Ticket 091 — a resume is NAMED: `make run … REPRISE=<run name>`. Without that name, nothing is
# reused (neither memory checkpoint nor decision trace), because the experiments/current link
# does not prove which experiment the files found there come from.
ifneq ($(REPRISE),)
  export REPRISE_RUN = $(REPRISE)
  export CONTINUE_RUN = 1
endif

# ── Agent memory ───────────────────────────────────────────────────────────────
# `make run MEM=0`: turns off long-term memory AND self-reflection;
# `make run MEM=1`: turns them back on. Without MEM, the file is not touched.
# ⚠ The lever is services/GAMA/CityTransport/config/sim_params.yaml, NOT GAMA Server
# parameter injection: Settings.gaml (load_sim_config, cycle 1) overwrites the
# injected parameters with the content of this file. The setting is PERSISTENT
# (the file is rewritten at cycle 2): it also applies to subsequent GUI runs.
MEM ?=
# `make run CACHE=0`: turns off the LLM semantic cache — every decision goes through the
# model and therefore ends up in llm_exchanges.jsonl. Prerequisite for a replay ("bare
# prompt" floor, prompt A/B) on the FULL scope. Costs ~4x more calls.
# `make run CACHE=1`: turns it back on. Without CACHE, the file is not touched.
CACHE ?=
# `make run CHOC=<name>`: plays the shock declared in services/llm-agents/config/chocs/<name>.yaml
# a quantified delay and a lived sentence, applied to designated agents on designated days.
# `make run CHOC=0`: removes it. Without CHOC, the file is not touched.
# ⚠ Turn off the cache along with it (`CACHE=0`): its key carries no duration, a decision made
# before the shock can be served again during it. An [ALARME] is raised if you forget.
CHOC ?=
CHOCS_DIR = services/llm-agents/config/chocs
# `make run EVENEMENT=<name>`: plays the event declared in
# services/llm-agents/config/evenements/<name>.yaml — a shock SUFFERED on arrival, or an article
# READ on waking up. `EVENEMENT=0` removes it. Without EVENEMENT, the file is not touched.
# `CHOC=` and `PRESSE=` are aliases: they write the same key, and say which one was used.
# The decision cache now turns itself off ON ITS OWN on event days (ticket 100); it
# stays active for the rest of the run, and the log counts what it serves there.
EVENEMENT ?=
PRESSE ?=
EVENEMENTS_DIR = services/llm-agents/config/evenements
# The three levers write the same key. The order says which one wins if two are set, and the
# name of the lever retained is displayed: two contradictory levers in the same command must
# not be resolved silently.
EVT := $(or $(EVENEMENT),$(PRESSE),$(CHOC))
EVT_LEVIER := $(if $(EVENEMENT),EVENEMENT,$(if $(PRESSE),PRESSE,$(if $(CHOC),CHOC,)))
SIM_PARAMS = services/GAMA/CityTransport/config/sim_params.yaml
APP_CONFIG = services/llm-agents/config/config.yaml

# ── Run without Google models ─────────────────────────────────────────────────
# `make run NO_GOOGLE=1`: blanks both Google keys in the containers;
# the google* instances are excluded from rotation ("missing API key")
# and the cascade continues on mistral/groq/cerebras. No degraded fallback:
# simply less LLM capacity.
NO_GOOGLE ?=
ifneq ($(NO_GOOGLE),)
  export SIM_PROVIDER_KEYS__google =
  export PROVIDER_KEYS__google2 =
endif

# ── Papers repository (ticket 115) ──────────────────────────────────────────
# The writing of the papers (article, GAMA Days abstract, presentations, figures) lives
# in a separate private repository, cloned next to this one. By default: the neighbour of
# the MAIN repository, computed from `git-common-dir` so that a worktree finds the
# same folder. Exported: the Python scripts read the same value
# (scripts/depot_papiers.py).
_GIT_COMMUN := $(shell git -C $(PROJECT_ROOT) rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
PAPER_DIR ?= $(abspath $(if $(_GIT_COMMUN),$(_GIT_COMMUN)/../..,$(PROJECT_ROOT)/..)/llm-agents-gama-papiers)
export PAPER_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Delegation (ticket 039, principle 3)
# ──────────────────────────────────────────────────────────────────────────────
#
# This file carries ONLY the configuration above. The 120 targets live in
# make/*.mk, one section per file. To add a target, edit the .mk of its
# section — nothing needs to be declared here, the wildcard picks them all up.
#
# ⚠ Variables must stay ABOVE the `include`s: a .mk that reads a
# variable not yet defined does not see it (`:=` and `ifneq` are evaluated
# at read time, not when the recipe runs).

MAKE_MODULES := $(sort $(wildcard $(PROJECT_ROOT)/make/*.mk))
include $(MAKE_MODULES)

# Pinned explicitly: without this line, the default target would be the FIRST
# target of the FIRST included .mk — so the alphabetical order of the files would
# decide what a bare `make` does.
.DEFAULT_GOAL := up
