## ── Experiment platform (ticket 035) ───────────────────────────────────────────────────
## Everything goes through `python -m experiences` in the controller container (services required to
## prepare a set — OTP/OSMnx — and for a gateway decision-maker; a local decision-maker needs
## nothing). Design: docs/arch/plateforme-experiences.md.
## Repository state measured on the HOST and passed to the container (ticket 045, A4): `git` is not
## installed in the `controller` image, so every run was archived with
## `depot = {commit: null, arbre_propre: null}` — no measurement could be tied to a
## code state. `:=` and not `=`: measured once when the Makefile loads, not at every call.
EXP_DEPOT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
EXP_DEPOT_ARBRE_PROPRE := $(shell test -z "$$(git status --porcelain --untracked-files=no 2>/dev/null)" && echo 1 || echo 0)

# A host path as seen from the container: the repository root is mounted there on /app.
chemin_app = $(if $(patsubst /app/%,,$(1)),/app/$(patsubst ./%,%,$(1)),$(1))

EXPERIENCES_PY = $(COMPOSE) exec -T -e REJEU_AB= -e REJEU_STRICT_AVANT_TS=0 -e EXPERIENCES_DIR=/app/data/experiences -e JEUX_DIR=/app/data/jeux -e REFERENTIEL_ENQUETE=/app/scripts/data/population/cerema_values.yaml -e EXP_DEPOT_COMMIT=$(EXP_DEPOT_COMMIT) -e EXP_DEPOT_ARBRE_PROPRE=$(EXP_DEPOT_ARBRE_PROPRE) controller python -m experiences
APPARIER_PY = $(COMPOSE) exec -T -e EXPERIENCES_DIR=/app/data/experiences -e JEUX_DIR=/app/data/jeux -e REFERENTIEL_ENQUETE=/app/scripts/data/population/cerema_values.yaml -e EXP_DEPOT_COMMIT=$(EXP_DEPOT_COMMIT) -e EXP_DEPOT_ARBRE_PROPRE=$(EXP_DEPOT_ARBRE_PROPRE) controller python /app/scripts/analysis/appariement_executions.py

## Prepares a recorded trip set. Usage: make jeu POP=data/population/population_1000_PANEL_v5 NOM=v5_j1 [JOUR=2026-03-16] [CONCURRENCE=8] [REQUIS="…"]
## The routing engines (default: controller otp1 otp2 otp3 osmnx1) are started and awaited healthy first.
jeu:
	@test -n "$(POP)" -a -n "$(NOM)" || { echo "Usage: make jeu POP=<population folder or file> NOM=<set name> [JOUR=AAAA-MM-JJ]"; exit 1; }
	@mkdir -p data/jeux
	@$(MAKE) --no-print-directory services-pretes REQUIS="$(if $(REQUIS),$(REQUIS),controller otp1 otp2 otp3 osmnx1)"
	$(EXPERIENCES_PY) preparer-jeu --population $(POP:data/population/%=/data/eqasim-output/%) --nom $(NOM) $(if $(JOUR),--jour $(JOUR),) $(if $(CONCURRENCE),--concurrence $(CONCURRENCE),)

## Inspects a set: make jeu-consulter NOM=v5_j1 [PERSONNE=418]
jeu-consulter:
	$(EXPERIENCES_PY) consulter-jeu --nom $(NOM) $(if $(PERSONNE),--personne $(PERSONNE),)

## Checks whether a set is stale: make jeu-verifier NOM=v5_j1
jeu-verifier:
	$(EXPERIENCES_PY) verifier-jeu --nom $(NOM)

## Config values registry (config/empreintes_valeurs.yaml), rebuilt from git on the HOST:
## a set stays valid when only the comments of a config file changed. After committing a
## config change: make config-empreintes — check without writing: make config-empreintes VERIFIER=1
config-empreintes:
	cd services/llm-agents && .venv/bin/python -m experiences config-empreintes $(if $(VERIFIER),--verifier,)

## Is another day's PT offer the same as the set's? METHODE=gtfs (default: do the proposed runs
## exist on that day in the feeds, validity per trip, no service required) or
## METHODE=moteurs (OTP on a sample). DECLARER=1 writes EQUIVALENCES.yaml next to the set.
## make jeu-verifier-jours NOM=v5_j1 JOUR=2026-03-17 [METHODE=gtfs|moteurs] [ECHANTILLON=100] [DECLARER=1]
jeu-verifier-jours:
	$(EXPERIENCES_PY) verifier-jours --nom $(NOM) --jour $(JOUR) $(if $(METHODE),--methode $(METHODE),) $(if $(ECHANTILLON),--echantillon $(ECHANTILLON),) $(if $(DECLARER),--declarer,)

## Reloads the LLM gateway (api + worker) — needed after an addition to packages/mobility_llm/src/mobility_llm/prompts/prompts.yaml
passerelle-recharger:
	$(COMPOSE) restart api worker

## LM Studio (local models, host) — loads a model with a context large enough for the gateway.
## LM Studio loads at 4,096 tokens by default: too short for a batch of 2 agents (~4,400 of prompt + answer).
## The identifier is the one from `lms ls` (e.g. mistralai/mistral-small-3.2) = default_model in providers.yaml.
## IDENTIFIANT=<alias> loads the model under another API name: required when the same identifier exists at
## a remote provider (qwen/qwen3.8-27b is also served by Groq) — the alias is then the local default_model.
## RECHARGER=1 first unloads what is in memory under that name (context too short, wrong identifier).
## Usage: make lmstudio-charger MODELE=mistralai/mistral-small-3.2 [CTX=16384] [RECHARGER=1]
##         make lmstudio-charger MODELE=qwen/qwen3.8-27b IDENTIFIANT=qwen3.8-27b-local
LMS ?= $(shell command -v lms 2>/dev/null || echo $(HOME)/.lmstudio/bin/lms)
CTX ?= 16384
lmstudio-charger:
	@test -n "$(MODELE)" || { echo "❌ MODELE= required (identifier as 'lms ls' gives it, e.g. mistralai/mistral-small-3.2)"; exit 1; }
	@command -v $(LMS) >/dev/null || { echo "❌ CLI '$(LMS)' not found — install it from LM Studio (Developer → lms) or pass LMS=/path/to/lms"; exit 1; }
	@if [ -n "$(RECHARGER)" ]; then \
	  for id in "$(or $(IDENTIFIANT),$(MODELE))" $(if $(IDENTIFIANT),"$(MODELE)",); do \
	    if $(LMS) ps 2>/dev/null | awk '{print $$1}' | grep -q -x -F "$$id"; then echo "⏏️  Unloading $$id…"; $(LMS) unload "$$id"; fi; \
	  done; \
	fi
	@if $(LMS) ps 2>/dev/null | awk '{print $$1}' | grep -q -x -F "$(or $(IDENTIFIANT),$(MODELE))"; then \
	  echo "ℹ️  $(or $(IDENTIFIANT),$(MODELE)) is already loaded — to change its context: rerun with RECHARGER=1"; \
	else \
	  echo "⏳ Loading $(MODELE) under the identifier $(or $(IDENTIFIANT),$(MODELE)) (context $(CTX) tokens)…"; \
	  $(LMS) load "$(MODELE)" -c $(CTX) -y $(if $(IDENTIFIANT),--identifier "$(IDENTIFIANT)",); \
	fi
	@$(LMS) ps

## Unloads a model from LM Studio (frees the memory): make lmstudio-decharger MODELE=<loaded identifier, IDENTIFIER column of `lms ps`>
lmstudio-decharger:
	@test -n "$(MODELE)" || { echo "❌ MODELE= required (loaded identifier, IDENTIFIER column of 'lms ps')"; exit 1; }
	@command -v $(LMS) >/dev/null || { echo "❌ CLI '$(LMS)' not found — install it from LM Studio (Developer → lms) or pass LMS=/path/to/lms"; exit 1; }
	$(LMS) unload "$(MODELE)"
	@$(LMS) ps

## LM Studio state: loaded models (identifier, context) and lmstudio_* instances seen by the gateway
lmstudio-etat:
	@command -v $(LMS) >/dev/null && $(LMS) ps || echo "CLI lms not found"
	@echo "— Gateway (/health), lmstudio_* instances:"
	@curl -sf localhost:8000/health | python3 -c "import sys,json; p=json.load(sys.stdin).get('providers',{}); [print(f'  {k:36s} available={v.get(\"available\")}  rpm={v.get(\"current_rpm\")}') for k,v in sorted(p.items()) if k.startswith('lmstudio_')] or print('  aucune instance lmstudio_* (PROVIDER_KEYS__lmstudio absent de .env ? make passerelle-recharger ?)')" || echo "  gateway unreachable (make up?)"

## Validates and stores an experiment: make experience-definir FICHIER=path/experience.yaml
experience-definir:
	$(EXPERIENCES_PY) definir $(FICHIER)

## Estimates the cost before launch: make experience-estimer EXP=<name>
experience-estimer:
	$(EXPERIENCES_PY) estimer --experience $(EXP)

## Launches (or resumes with REPRENDRE=1) an experiment without simulator: make experience-lancer EXP=<name> [REPRENDRE=1] [REQUIS="controller api worker"] [ACCEPTER_PERIME=1] [ATTENDRE_FENETRE=1]
## The REQUIS services (default: `controller`) are started and awaited healthy before the launch.
## ATTENDRE_FENETRE=1: when the quota runs out, sleeps in-process until midnight UTC then restarts on its own (R4).
experience-lancer:
	@mkdir -p data/experiences
	@$(MAKE) --no-print-directory services-pretes REQUIS="$(if $(REQUIS),$(REQUIS),controller)"
	$(EXPERIENCES_PY) lancer --experience $(EXP) $(if $(REPRENDRE),--reprendre,) $(if $(ACCEPTER_PERIME),--accepter-perime,) $(if $(ATTENDRE_FENETRE),--attendre-fenetre,)

experience-reprendre:
	@$(MAKE) experience-lancer EXP=$(EXP) REPRENDRE=1 REQUIS="$(REQUIS)"

## Per-key queue (spec parallelisation_experiences):
## experience-file            experiments waiting for a key (FIFO)
## experience-actives         experiments holding a key (parallel, in progress)
## experience-defiler EXP=<n> removes an experiment from the queue before its promotion
experience-file:
	$(EXPERIENCES_PY) file
experience-actives:
	$(EXPERIENCES_PY) actives
experience-defiler:
	$(EXPERIENCES_PY) defiler --experience $(EXP)

## Campaign (ticket 074, lot D): a named batch of experiments played one after the other. Runs
## on the HOST. By default (chaining): one at a time, without waiting for the quota window;
## stopped on quota → « non terminée », keys held elsewhere → « non jouée », no relaunch; at
## the end of the list, it hands control back. Do NOT run `experience-ordonnancer` alongside: it
## would start the queue behind its back. `a_travers_les_quotas: true` in the YAML restores the old
## behaviour (batch sleep, postponement, retrieval, promotion of the queue at each round).
##   make campagne-lancer NOM=bascule_anglaise_v6 [ESTIMER=1] [RECOMMENCER=1] [INTERVALLE=30]
##   make campagne-etat   NOM=bascule_anglaise_v6 [JSON=1]
##   make campagne-arreter NOM=bascule_anglaise_v6
## ESTIMER=1 states the budget and exits, without queuing anything. RECOMMENCER=1 ignores the existing state.
campagne-lancer:
	@test -n "$(NOM)" || { echo "Usage: make campagne-lancer NOM=<campaign>"; exit 1; }
	@$(MAKE) --no-print-directory services-pretes REQUIS="$(if $(REQUIS),$(REQUIS),controller)"
	cd services/llm-agents && .venv/bin/python -m experiences campagne-lancer --nom $(NOM) \
	  $(if $(ESTIMER),--estimer,) $(if $(RECOMMENCER),--recommencer,) \
	  $(if $(INTERVALLE),--intervalle $(INTERVALLE),)

campagne-etat:
	@test -n "$(NOM)" || { echo "Usage: make campagne-etat NOM=<campaign>"; exit 1; }
	cd services/llm-agents && .venv/bin/python -m experiences campagne-etat --nom $(NOM) $(if $(JSON),--json,)

campagne-arreter:
	@test -n "$(NOM)" || { echo "Usage: make campagne-arreter NOM=<campaign>"; exit 1; }
	cd services/llm-agents && .venv/bin/python -m experiences campagne-arreter --nom $(NOM)

## Scheduler (HOST): reconciles ghosts and starts the queued experiments as soon as a
## key frees up. Leave it running (the dashboard also supervises it). Ctrl-C to stop.
experience-ordonnancer:
	cd services/llm-agents && .venv/bin/python -m experiences ordonnancer $(if $(INTERVALLE),--intervalle $(INTERVALLE),)

## Pause / clean stop of the current run: make experience-pause EXP=<name> · make experience-arreter EXP=<name>
experience-pause:
	$(EXPERIENCES_PY) pause --experience $(EXP)
experience-arreter:
	$(EXPERIENCES_PY) arreter --experience $(EXP)

## Reconciles decisions.jsonl with the set (no trip skipped?): make experience-erreurs EXP=<name> [EXEC=<folder>]
experience-erreurs:
	$(EXPERIENCES_PY) erreurs $(if $(EXEC),$(EXEC),--experience $(EXP))

## Registry of experiments and runs: make registre [TRIER=couverture] [FILTRER=decideur=gemini] [TOUT=1]
## TOUT=1 shows again the archived or invalidated experiments (hidden by default).
registre:
	$(EXPERIENCES_PY) registre $(if $(TRIER),--trier $(TRIER),) $(if $(FILTRER),--filtrer $(FILTRER),) $(if $(TOUT),--inclure-masquees,)

## Status of all experiments (hidden ones included): make experience-statuts
experience-statuts:
	$(EXPERIENCES_PY) statuts

## Sets an experiment's status WITHOUT DELETING or moving ANYTHING:
##   make experience-statuer EXP=<name> STATUT=archivee|invalide|actif MOTIF="..." [REF=specs/x.md]
## Runs, traces, scores and fingerprints stay on disk; only the visibility changes.
experience-statuer:
	@test -n "$(EXP)" || (echo "ERROR: EXP=<experiment name> missing" >&2; exit 2)
	@test -n "$(STATUT)" || (echo "ERROR: STATUT=archivee|invalide|actif missing" >&2; exit 2)
	$(EXPERIENCES_PY) statuer $(EXP) $(STATUT) $(if $(MOTIF),--motif "$(MOTIF)",) $(if $(REF),--reference $(REF),)

## Compares two runs (refuses if not comparable): make comparer A=<folder> B=<folder> [TOUT=1]
## Also refuses to pair an invalidated or archived experiment; TOUT=1 forces it, recalling the reason.
comparer:
	$(EXPERIENCES_PY) comparer $(A) $(B) $(if $(TOUT),--inclure-invalides,)

## Pairs TWO runs decision by decision — the quantification of axis 0 (ticket 073):
##   make apparier A=<run folder> B=<run folder> [JSON=<file>] [TOUT=1] [SEUIL=0.8]
## A and B are given relative to the repository root (data/experiences/<exp>/executions/<timestamp>).
## Runs in the `controller` container, the only place where the registry's comparability guard
## is available. READ ONLY: nothing is written into the runs.
## `comparer` answers "are these two measurements comparable?"; `apparier` answers
## "by how much did the model move, decision by decision?".
apparier:
	@test -n "$(A)" -a -n "$(B)" || { echo "Usage: make apparier A=<folder> B=<folder> [JSON=<file>] [TOUT=1]"; exit 1; }
	$(APPARIER_PY) $(call chemin_app,$(A)) $(call chemin_app,$(B)) \
	  $(if $(JSON),--json $(call chemin_app,$(JSON)),) $(if $(TOUT),--tout,) $(if $(SEUIL),--seuil $(SEUIL),)

.PHONY: apparier

.PHONY: campagne-lancer campagne-etat campagne-arreter

.PHONY: services-pretes passerelle-recharger jeu jeu-consulter config-empreintes lmstudio-charger lmstudio-decharger lmstudio-etat jeu-verifier jeu-verifier-jours experience-definir experience-estimer experience-lancer experience-reprendre experience-pause experience-arreter experience-erreurs experience-file experience-actives experience-defiler experience-ordonnancer registre comparer experience-statuts experience-statuer

.PHONY: status
## Status of the current GAMA run. Parsable key=value output:
## run=actif|inactif, mode=offline|ihm, pid, current=<target of the experiments/current symlink>
status:
	@if pgrep -f "launch_headless.py" > /dev/null; then \
		echo "run=actif mode=offline pid=$$(pgrep -f launch_headless.py | head -1)"; \
	elif pgrep -f "$(GAMA_BIN)" > /dev/null; then \
		echo "run=actif mode=ihm pid=$$(pgrep -f "$(GAMA_BIN)" | head -1)"; \
	else \
		echo "run=inactif"; \
	fi
	@echo "current=$$(readlink experiments/current 2>/dev/null || echo '-')"

.PHONY: stop-run
## Stops the current GAMA run WITHOUT touching the rest of the stack (api, worker, redis…).
## Offline: kills the launcher in the controller container then stops the gama service
## (GAMA Server kills the experiment whose client disconnected). GUI: SIGTERM to GAMA.
## To stop everything, services included: make down.
stop-run:
	@if $(COMPOSE) ps --status running controller 2>/dev/null | grep -q controller; then \
		$(COMPOSE) exec -T controller python3 -c "import os, psutil, signal; [p.send_signal(signal.SIGTERM) for p in psutil.process_iter(['name', 'cmdline']) if p.pid != os.getpid() and p.info['cmdline'] and any('launch_headless.py' in arg for arg in p.info['cmdline'])]" 2>/dev/null || true; \
	fi
	-@pkill -f "scripts/gama/launch_headless.py" 2>/dev/null || true
	-@$(COMPOSE) --profile offline stop gama 2>/dev/null || true
	-@pkill -f "$(GAMA_BIN)" 2>/dev/null || true
	@echo "✅ Run stopped. The services stay in place (make down to shut everything down)."
