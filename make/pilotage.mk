# ──────────────────────────────────────────────────────────────────────────────
# Pilotage
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: dashboard

DASHBOARD_PYTHON ?= $(VENV_PYTHON)
DASHBOARD_PORT   ?= 8503
# Imposed theme (light|dark): the charts pick their colour steps
# from it. Leaving it empty would make the UI and the text colours diverge.
DASHBOARD_THEME  ?= light

.PHONY: help
## Lists the documented targets: the name of each target and the `##` block preceding it.
## Useful since the dashboard no longer shows the target catalogue.
## Scans ALL the files read (root + make/*.mk): with `firstword`, the split
## of ticket 039 would have reduced this list to the root file's targets alone — zero.
help:
	@awk '\
	  /^## / { if (doc == "") doc = substr($$0, 4); next } \
	  /^\.PHONY/ { next } \
	  /^[a-zA-Z0-9_.-]+:/ { if (doc != "") { split($$0, cible, ":"); printf "  %-24s %s\n", cible[1], doc }; doc = ""; next } \
	  /^[[:space:]]*$$/ { doc = "" } \
	' $(MAKEFILE_LIST)

## Control dashboard: state of the services, experiments in progress, tickets, run metrics.
## Usage: make dashboard [DASHBOARD_THEME=dark] [DASHBOARD_PORT=8503] [DASHBOARD_PYTHON=/path/python]
dashboard:
	@test -x $(DASHBOARD_PYTHON) || { \
	  echo "Interpreter not found: $(DASHBOARD_PYTHON)"; \
	  echo "Override it: make dashboard DASHBOARD_PYTHON=/path/to/python"; \
	  exit 1; }
	$(DASHBOARD_PYTHON) -m streamlit run scripts/dashboard/app.py \
	  --server.port $(DASHBOARD_PORT) --server.headless false \
	  --theme.base $(DASHBOARD_THEME) --theme.primaryColor "#2a78d6"
