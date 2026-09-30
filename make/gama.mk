# ──────────────────────────────────────────────────────────────────────────────
# GAMA
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: run

## Wait until the API and controller are ready (polls /health)
wait-ready:
	@echo "⏳ Waiting for the API to be ready (max 300s)..."
	@elapsed=0; \
	while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do \
		if [ $$elapsed -ge 300 ]; then \
			echo ""; \
			echo "❌ Timeout: the API (port 8000) did not respond within 300s."; \
			echo "   Check the logs: make logs"; \
			exit 1; \
		fi; \
		printf "\r   API  (port 8000): %ds elapsed..." $$elapsed; \
		sleep 5; elapsed=$$((elapsed + 5)); \
	done
	@echo "\n✅ API prête"
	@echo "⏳ Waiting for the Controller to be ready (max 60s)..."
	@elapsed=0; \
	while ! curl -sf http://localhost:8002/ > /dev/null 2>&1; do \
		if [ $$elapsed -ge 60 ]; then \
			echo ""; \
			echo "⚠️  Controller (port 8002) not ready yet, launching GAMA anyway."; \
			break; \
		fi; \
		printf "\r   Controller (port 8002): %ds elapsed..." $$elapsed; \
		sleep 3; elapsed=$$((elapsed + 3)); \
	done
	@echo "⏳ Waiting for Grafana to be ready (max 60s)..."
	@elapsed=0; \
	while ! curl -sf http://localhost:3000/api/health > /dev/null 2>&1; do \
		if [ $$elapsed -ge 60 ]; then \
			echo ""; \
			echo "⚠️  Grafana (port 3000) not ready yet, launching GAMA anyway."; \
			break; \
		fi; \
		printf "\r   Grafana  (port 3000): %ds elapsed..." $$elapsed; \
		sleep 3; elapsed=$$((elapsed + 3)); \
	done
	@echo "\n✅ Services prêts — lancement GAMA autorisé"

## Start all services then launch the GAMA experiment
## Usage: make run [EXPERIMENT_NAME=e] [OFFLINE=1]
## OFFLINE=1: headless GAMA in a container (service `gama`, compose profile "offline"),
## driven via GAMA Server — no GUI, everything starts with docker compose.
# ⚠ The `mkdir -p` that follows the `rm -rf` is not cosmetic: Prometheus refuses to start
# if its mount point does not exist, and the metrology was then missing for the whole run.
# 2026-09-28 — value of the common-prefix lock that `make run` writes into sim_params.yaml: the one
# the controller receives (infra/docker-compose.yml, WORLD__PREFIXE_COMMUN, `false` by default).
PREFIXE_COMMUN_GAMA = $(if $(filter 1 true True TRUE yes on,$(WORLD__PREFIXE_COMMUN)),true,false)

run:
	@# Ticket 089 — the previous run is stopped BEFORE everything else, and for ALL
	@# launches, hot resume included. Without it the GAMA container stays up and
	@# each launch loads the model into the process already in place: on
	@# 2026-09-16, two City.gaml resident in the same JVM capped at 12 GB got
	@# the container killed by its own limit on day 1. What weighs is the territory
	@# (453 communes, full network), not the number of agents.
	@echo "🛑 Stopping the previous run before launch (otherwise GAMA would keep its experiment in memory)..."
	@$(MAKE) stop-run
ifeq ($(CONT),)
	echo "🗑️  Stopping Grafana and Prometheus..."; \
	$(COMPOSE) stop grafana prometheus 2>/dev/null || true; \
	$(COMPOSE) rm -f grafana prometheus 2>/dev/null || true; \
	echo "🗑️  Deleting Grafana and Prometheus data..."; \
	rm -rf data/grafana_data data/prometheus_data; \
	mkdir -p data/grafana_data data/prometheus_data; \
	echo "🗑️  Purging Redis counters (wmetrics:)..."; \
	$(COMPOSE) exec -T redis redis-cli --scan --pattern "wmetrics:*" | xargs -r $(COMPOSE) exec -T redis redis-cli del 2>/dev/null || true; \
	echo "♻️  Stopping the controller for a fresh start..."; \
	$(COMPOSE) stop controller 2>/dev/null || true; \
	$(COMPOSE) rm -f controller 2>/dev/null || true;
	@# 2026-09-28 — the GAMA lock follows the controller being recreated: `true` when the orchestrator
	@# launches a memory A/B arm (it passes WORLD__PREFIXE_COMMUN), `false` for a run launched by
	@# hand. Without it, a manual run after an A/B found `prefixe_commun: true` in
	@# sim_params.yaml against a controller at `false`, and /init refused it (409). On resume
	@# (CONT=1), the controller is not recreated: it is left untouched.
	@perl -0pi -e 's/^prefixe_commun:.*/prefixe_commun: $(PREFIXE_COMMUN_GAMA)/m or $$_ .= "\nprefixe_commun: $(PREFIXE_COMMUN_GAMA)\n"' $(SIM_PARAMS)
	@echo "🔒 GAMA common prefix: $(PREFIXE_COMMUN_GAMA) (the controller's) — written to $(SIM_PARAMS)"

else
	@echo "♻️  Hot resume: workdir, metrics and counters kept ($(shell readlink experiments/current))"
endif
ifneq ($(MEM),)
	@perl -pi -e 's/^long_term_memory_enabled:.*/long_term_memory_enabled: $(if $(filter 0,$(MEM)),false,true)/; s/^long_term_self_reflect_enabled:.*/long_term_self_reflect_enabled: $(if $(filter 0,$(MEM)),false,true)/' $(SIM_PARAMS)
	@echo "🧠 Agent memory (LTM + self-reflection): $(if $(filter 0,$(MEM)),DÉSACTIVÉE,activée) — written to $(SIM_PARAMS)"
endif
ifneq ($(CACHE),)
	@perl -0pi -e 's/^(cache:\n(?:.*\n)*?\s*enabled:).*/$$1 $(if $(filter 0,$(CACHE)),false,true)/m' $(APP_CONFIG)
	@echo "💾 LLM semantic cache: $(if $(filter 0,$(CACHE)),DÉSACTIVÉ — chaque décision sera journalisée (~4x plus d'appels),activé) — written to $(APP_CONFIG)"
endif
# ── Ticket 100 — a single lever for both regimes ─────────────────────────────────────
# `EVENEMENT=` is the new name; `CHOC=` and `PRESSE=` are aliases, and say which one was used.
# The file lives in config/evenements/, just as the BAAC law lives outside the configuration.
ifneq ($(EVT),)
ifeq ($(EVT),0)
	@perl -0pi -e 's/^chocs:\n(?:[ \t]+.*\n)*//m; s/^evenements:\n(?:[ \t]+.*\n)*//m' $(APP_CONFIG)
	@echo "⚡ Event: NONE — removed from $(APP_CONFIG) (lever $(EVT_LEVIER))"
else
	@# Looked up in config/evenements/ then, for declarations not yet migrated, in
	@# config/chocs/ — which still loads, with a warning naming the file.
	@test -f $(EVENEMENTS_DIR)/$(EVT).yaml || test -f $(CHOCS_DIR)/$(EVT).yaml || { echo "❌ Event not found: $(EVENEMENTS_DIR)/$(EVT).yaml (shipped cases: $$(ls $(EVENEMENTS_DIR)/*.yaml | xargs -n1 basename | sed 's/.yaml//' | tr '\n' ' '))"; exit 1; }
	@perl -0pi -e 's/^chocs:\n(?:[ \t]+.*\n)*//m; s/^evenements:\n(?:[ \t]+.*\n)*//m' $(APP_CONFIG)
	@if [ -f $(EVENEMENTS_DIR)/$(EVT).yaml ]; then \
		printf 'evenements:\n  enabled: true\n  fichier: /app/config/evenements/%s.yaml\n' "$(EVT)" >> $(APP_CONFIG); \
	else \
		printf 'evenements:\n  enabled: true\n  fichier: /app/config/chocs/%s.yaml\n' "$(EVT)" >> $(APP_CONFIG); \
		echo "  ↳ declaration in the ticket 079 format; it loads, and the log says so"; \
	fi
	@echo "⚡ Event: $(EVT) (lever $(EVT_LEVIER)) — written to $(APP_CONFIG). The decision cache turns itself off on event days."
endif
endif
ifneq ($(JEU),)
	@# Ticket 035 (spec 04, G2): the simulation consumes the recorded set data/jeux/$(JEU) —
	@# nominal regime with no engine call. Writes `data.jeu_enregistre` into $(APP_CONFIG); the
	@# time tolerances must be declared there (commented block).
	@test -f data/jeux/$(JEU)/MANIFEST.yaml || { echo "❌ Set not found: data/jeux/$(JEU)/MANIFEST.yaml (prepare it: make jeu POP=… NOM=$(JEU))"; exit 1; }
	@grep -q '^  jeu_tolerances_horaires:' $(APP_CONFIG) || { echo "❌ data.jeu_tolerances_horaires is not declared in $(APP_CONFIG) — uncomment and validate the block (spec 04, G5)"; exit 1; }
	@perl -0pi -e 's/^  #? ?jeu_enregistre:.*\n//m; s/^(data:\n)/$$1  jeu_enregistre: \/app\/data\/jeux\/$(JEU)\n/m' $(APP_CONFIG)
	@echo "📼 Recorded set: data/jeux/$(JEU) — written to $(APP_CONFIG)"
else
	@if grep -q '^  jeu_enregistre:' $(APP_CONFIG); then \
		perl -0pi -e 's/^  jeu_enregistre:.*\n//m' $(APP_CONFIG); \
		echo "📼 No set designated: computed on the fly (data.jeu_enregistre removed from $(APP_CONFIG))"; \
	fi
endif
	@# The settings of $(APP_CONFIG) are read at controller STARTUP, never hot: as long as
	@# the file differs from the copy applied at the last launch (.config.yaml.applique), the
	@# controller is recreated — CACHE=, JEU= and any manual edit thus take effect
	@# (author's decision of 2026-09-06, question 17 of ticket 035).
	@# ⚠ The comparison happens BEFORE `make up` (2026-09-24): done after, it recreated a
	@# controller that `make up` had just started. Both fell within the same minute, hence
	@# in the same run directory; the second kept the identity of the first (ticket 091) and
	@# the cohort refused the arm (« aucune identité POSTÉRIEURE au démarrage du contrôleur »).
	@# So the controller is removed here, and `make up` creates it ONCE, with the right config.
	@if ! cmp -s $(APP_CONFIG) .config.yaml.applique; then \
		echo "♻️  $(APP_CONFIG) changed since the last launch: the controller will restart fresh"; \
		$(COMPOSE) rm -sf controller 2>/dev/null || true; \
	fi
	@cp $(APP_CONFIG) .config.yaml.applique 2>/dev/null || true
	@$(MAKE) up
	@$(MAKE) wait-ready
ifneq ($(OFFLINE),)
	@if pgrep -f "launch_headless.py" > /dev/null; then \
		echo "⚠️  Un launcher GAMA headless tourne déjà. Lancement ignoré."; \
	else \
		echo "🚀 Lancement headless de l'expérience GAMA : $(EXPERIMENT_NAME) (GAMA Server, conteneur gama)..."; \
		mkdir -p experiments/current; \
		$(COMPOSE) exec -T -e GAMA_EXPERIMENT=$(EXPERIMENT_NAME) controller \
			python /app/scripts/gama/launch_headless.py \
			>> experiments/current/gama_headless.log 2>&1 & \
		echo "   GAMA console → experiments/current/gama_headless.log"; \
	fi
else
	@if pgrep -f "$(GAMA_BIN)" > /dev/null; then \
		echo "⚠️  GAMA est déjà en cours d'exécution. Lancement ignoré."; \
	else \
		echo "🚀 Lancement de l'expérience GAMA : $(EXPERIMENT_NAME)..."; \
		$(GAMA_BIN) -p $(WORKSPACE) -o $(MODEL_PATH) -e "$(EXPERIMENT_NAME)" & \
	fi
endif

.PHONY: run-offline
## Alias: make run-offline == make run OFFLINE=1
run-offline:
	@$(MAKE) run OFFLINE=1
