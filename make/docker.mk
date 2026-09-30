# ──────────────────────────────────────────────────────────────────────────────
# Docker Compose
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: up down restart rebuild logs ps clean purge-cache

up:
	$(COMPOSE) up -d

# --profile offline: includes the gama service (headless mode) if it is running;
# no effect when it is not started.
down:
	$(COMPOSE) --profile offline down

restart:
	$(COMPOSE) restart

.PHONY: watch-containers
## Probes container memory and captures the one that goes down. Exists because three runs were
## lost on 2026-09-07 to a `controller` killed (code 137) with no trace in its logs and
## no OOM reported by Docker. Writes memoire.csv, chute-<service>.txt and sonde.log in
## experiments/.dashboard/conteneurs/<timestamp>/. Modifies no container.
## Usage: make watch-containers [INTERVAL=10] [SEUIL=85] [SERVICES=controller,osmnx1] [DUREE=0]
watch-containers:
	$(DASHBOARD_PYTHON) scripts/debug/watch_containers.py \
	  --interval $(or $(INTERVAL),10) --seuil-pct $(or $(SEUIL),85) \
	  $(if $(SERVICES),--services $(SERVICES),) $(if $(DUREE),--duree $(DUREE),)

.PHONY: stop-services
## Stops ONLY the named services, without removing containers or volumes: frees the RAM
## (osmnx1 holds 3.5 GiB, each OTP 1.2 to 1.5 GiB) and a restart stays fast.
## Usage: make stop-services SERVICES="controller api worker otp1 otp2 otp3 osmnx1 eqasim redis"
stop-services:
	@test -n "$(SERVICES)" || { echo "SERVICES= est vide : nommez les services à arrêter"; exit 2; }
	$(COMPOSE) stop $(SERVICES)

.PHONY: experience-lancer-arret
## Launches an experiment THEN stops the services it used. The stop is chained in the
## command: it happens even if the dashboard is closed in the meantime. The experiment's
## return code is kept, so that a failure stays a failure in the job log.
## Usage: make experience-lancer-arret EXP=<name> SERVICES="controller api worker …" [REQUIS="…"]
## SERVICES = what is STOPPED at the end; REQUIS = what is guaranteed BEFORE launching.
experience-lancer-arret:
	@test -n "$(SERVICES)" || { echo "SERVICES= est vide : nommez les services à arrêter"; exit 2; }
	@$(MAKE) --no-print-directory services-pretes REQUIS="$(if $(REQUIS),$(REQUIS),controller)"
	@set +e; \
	$(EXPERIENCES_PY) lancer --experience $(EXP); code=$$?; \
	if $(EXPERIENCES_PY) actives --est-vide; then \
		echo "[arret-fin] (code $$code) no active experiment left — stopping: $(SERVICES)"; \
		$(COMPOSE) stop $(SERVICES); \
	else \
		echo "[arret-fin] (code $$code) other experiments hold a key — services left up (R5):"; \
		$(EXPERIENCES_PY) actives; \
	fi; \
	exit $$code

.PHONY: run-arret
## Same for a launch with GAMA: the `gama` service of the offline profile is stopped too.
## Usage: make run-arret JEU=<jeu> SERVICES="controller api worker …"
run-arret:
	@test -n "$(SERVICES)" || { echo "SERVICES= est vide : nommez les services à arrêter"; exit 2; }
	@set +e; \
	$(MAKE) run OFFLINE=1 JEU=$(JEU); code=$$?; \
	echo "[arret-fin] run finished (code $$code) — stopping: $(SERVICES) gama"; \
	$(COMPOSE) --profile offline stop $(SERVICES) gama; \
	exit $$code

.PHONY: up-services services-pretes
## Starts ONLY the named services, with their compose dependencies: an experiment
## does not need the monitoring (Prometheus, Grafana, cAdvisor, node-exporter, Flower). Does
## not wait for healthy services before returning — for that, see `services-pretes`.
## Usage: make up-services SERVICES="controller api worker"
up-services:
	@test -n "$(SERVICES)" || { echo "SERVICES= est vide : nommez les services à démarrer"; exit 2; }
	$(COMPOSE) up -d $(SERVICES)

## Guarantees that the named services are running AND healthy, then returns. Idempotent:
## on a stack already up, docker compose recreates nothing and the target passes in a second.
## Chained at the head of `experience-lancer` and `jeu`: no more need to remember to start the
## stack before launching an experiment, from the dashboard as from a terminal.
## `--wait` waits for the healthchecks: `controller` depends on healthy `api`, `otp1-3`,
## `eqasim` and `osmnx1`, and loading the OTP/OSMnx graphs takes several minutes when
## cold. ATTENTE bounds this wait so as to fail loudly rather than hang.
## `--no-recreate` is NON-NEGOTIABLE: a bare `up -d` recreates a container whose configuration
## has changed since it started, which WOULD KILL the runner of an experiment running in
## `controller` (launched by `docker compose exec`). This target is chained at every launch,
## including while another experiment is working: it starts what is missing and
## touches nothing that is running. To apply a configuration change, use
## `make run` (which recreates explicitly) or `docker compose up -d --force-recreate` by hand.
## Usage: make services-pretes REQUIS="controller api worker" [ATTENTE=600]
services-pretes:
	@test -n "$(REQUIS)" || { echo "REQUIS= is empty: name the services to guarantee"; exit 2; }
	@echo "🐳 Required services: $(REQUIS) — starting what is missing, without touching what is running (max $(if $(ATTENTE),$(ATTENTE),600)s)"
	$(COMPOSE) up -d --no-recreate --wait --wait-timeout $(if $(ATTENTE),$(ATTENTE),600) $(REQUIS)

## Rebuild all images from scratch and restart
rebuild:
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

## Rebuild and restart api + worker + controller only
api:
	$(COMPOSE) up --build api worker controller

## Rebuild and restart otp + worker only
otp:
	$(COMPOSE) up --build otp worker

.PHONY: otp-graph
## Builds the OTP graph (data/gtfs/graph.obj) starting from the VERSIONED configurations.
## `data/gtfs/` is an unversioned working directory: without this copy, a
## rebuild loses the settings of `services/otp-toulouse/toulouse/*.json` without saying so — it
## happened on 2026-09-04 (embedRouterConfig, boardingLocationTags, staticParkAndRide and
## maxStopToShapeSnapDistance lost, hence instances running on OTP's defaults).
## The old graph is archived, never overwritten.  Usage: make otp-graph
otp-graph:
	@test -f data/gtfs/Toulouse.osm.pbf || { echo "data/gtfs/Toulouse.osm.pbf missing"; exit 1; }
	@cp -v services/otp-toulouse/toulouse/build-config.json services/otp-toulouse/toulouse/router-config.json \
	       services/otp-toulouse/toulouse/otp-config.json data/gtfs/
	@if [ -f data/gtfs/graph.obj ]; then \
	  d=data/gtfs/archives/$$(date +%Y-%m-%d_%H-%M)_pre_build ; mkdir -p $$d ; \
	  mv -v data/gtfs/graph.obj $$d/ ; fi
	java -Xmx4G -jar services/otp-toulouse/bin/otp-shaded-*.jar --build data/gtfs --save
	@ls -l data/gtfs/graph.obj && shasum -a 256 data/gtfs/graph.obj

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

error:
	python3 scripts/errors.py $(if $(LOG),$(LOG),experiments/current/app.log)

warning:
	python3 scripts/warnings.py $(if $(LOG),$(LOG),experiments/current/app.log)

## "Agent-ready" health report of the latest run. Usage: make report [RUN=experiments/archive/<date>] [OUT=rapport.md]
report:
	python3 scripts/debug/run_report.py $(if $(RUN),$(RUN),) $(if $(OUT),--out $(OUT),)

## Throughput vs LLM capacity analysis of the latest run. Usage: make capacity [RUN=… OUT=…]
capacity:
	python3 scripts/debug/llm_capacity.py $(if $(RUN),$(RUN),) $(if $(OUT),--out $(OUT),)

## Init phase analysis: step timeline, cache warm-up (OTP/OSMnx/LLM), startup bugs. Usage: make init [RUN=… OUT=…]
init:
	python3 scripts/debug/init_report.py $(if $(RUN),$(RUN),) $(if $(OUT),--out $(OUT),)

.PHONY: personas-verifier
## Checks that a persona population is indeed the one its MANIFEST describes (ticket 093).
## Usage: make personas-verifier [POP=data/population/population_10_mesurables_093]
# File fingerprint, size, and existence + fingerprint of the source it is drawn from. This
# is NOT a cohort seal (no strata or margins — they make no sense with ten agents):
# it guarantees that a population edited by hand stops claiming a measured criterion.
personas-verifier:
	$(VENV_PYTHON) -m scripts.data.population.selectionner_personas_mesurables \
		--verifier "$(if $(POP),$(POP),data/population/population_10_mesurables_093)"

.PHONY: mesures
## Recomputes a run's per-simulated-day measurement CSVs (ticket 093). Usage: make mesures RUN=experiments/archive/<date> [OUT=…]
# The CSVs live in <run>/mesures/ and survive the containers being stopped. The computation is
# idempotent on FLOWS (everything is rewritten from the deduplicated moves.csv) and keeps the
# STATES already written: a state cannot be reconstructed afterwards, and replaying a resume
# overwrites the checkpoints of days already lived.
mesures:
	@test -n "$(RUN)" || { echo "Usage: make mesures RUN=experiments/archive/<date>"; exit 2; }
	$(VENV_PYTHON) -m scripts.analysis.mesures "$(RUN)" $(if $(OUT),-o "$(OUT)",)

.PHONY: mesures-continuite
## Checks that a cut did not betray the curves (ticket 093, acceptance). Usage: make mesures-continuite AVANT=<copy of the CSVs> APRES=<run>/mesures
# Non-zero exit if a key is duplicated, a working day is missing, or if a value of a day
# PRIOR to the cut has changed. The acceptance procedure is in docs/arch/mesures-personas.md.
mesures-continuite:
	@test -n "$(AVANT)" -a -n "$(APRES)" || { echo "Usage: make mesures-continuite AVANT=<copy> APRES=<run>/mesures"; exit 2; }
	$(VENV_PYTHON) -m scripts.analysis.mesures.continuite "$(AVANT)" "$(APRES)"

.PHONY: personas-mesurables
## Picks personas whose decisions are observable, from a run already played (ticket 093).
## Usage: make personas-mesurables RUN=data/experiences/<exp>/executions/<date> [SOURCE=…] [SORTIE=…] [CONSERVER=899549,616478] [N=10]
# The criterion (≥ 4 trips, ALWAYS more than one itinerary, ≥ 2 modes chosen) is measured BEFORE
# the run to observe: the selection is thus reproducible and verifiable, and the MANIFEST names the
# reference run — the same agent can meet the criterion on one run and miss it on another.
personas-mesurables:
	@test -n "$(RUN)" || { echo "Usage: make personas-mesurables RUN=<directory of a played run>"; exit 2; }
	$(VENV_PYTHON) -m scripts.data.population.selectionner_personas_mesurables \
		--run "$(RUN)" \
		--source "$(if $(SOURCE),$(SOURCE),data/population/population_1000_PANEL_v6/population.json)" \
		--sortie "$(if $(SORTIE),$(SORTIE),data/population/population_10_mesurables_093)" \
		--conserver "$(if $(CONSERVER),$(CONSERVER),899549,616478)" \
		-n $(if $(N),$(N),10)

.PHONY: memoire-rapport
## Per-persona HTML report on a memory run (ticket 077, lot F). Usage: make memoire-rapport RUN=experiments/archive/2026-09-14_23_58
# The output lives under docs/traces/<date_heure>_rapport_memoire/, outside git: it is a
# dated analysis trace, not a versioned deliverable (decision 2026-09-02).
memoire-rapport:
	@test -n "$(RUN)" || { echo "Usage: make memoire-rapport RUN=experiments/archive/<date>"; exit 2; }
	@out="docs/traces/$$(date +%Y-%m-%d_%H-%M)_rapport_memoire"; \
	mkdir -p "$$out"; \
	$(VENV_PYTHON) -m scripts.analysis.memoire.rapport "$(RUN)" -o "$$out/rapport.html"

.PHONY: providers provider
## Updates config/llm_gateway/providers.yaml from the real quotas (x-ratelimit headers + Google Cloud Quotas). Usage: make providers [PROVIDER=groq] [DRY_RUN=1]
providers:
	$(VENV_PYTHON) scripts/providers/refresh.py $(if $(DRY_RUN),--dry-run,) $(if $(PROVIDER),--provider $(PROVIDER),)

provider: providers

## Remove containers, volumes and images
clean:
	@read -rp "Voulez-vous supprimer toutes les images Docker ? (y/N): " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || [ "$$ans" = "yes" ] || [ "$$ans" = "YES" ]; then \
		$(COMPOSE) down -v --rmi all; \
		docker system prune -a --volumes -f; \
	fi

clean_all:
	@read -rp "Voulez-vous supprimer toutes les images Docker ? (y/N): " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ] || [ "$$ans" = "yes" ] || [ "$$ans" = "YES" ]; then \
		docker ps -aq | xargs -r docker rm -f; \
		docker system prune -a -f --volumes; \
	fi

## Purges all application caches (OSMnx, eqasim, RAPTOR) + Docker builder cache
purge_cache:
	@echo "🗑️  Cache Docker builder..."
	docker builder prune -a -f
	@echo "🗑️  Cache OSMnx graphs (data/cache/osmnx)..."
	rm -f data/cache/osmnx/*.pkl
	@echo "🗑️  Cache OSMnx local (services/llm-agents/osmnx_cache)..."
	rm -f services/llm-agents/osmnx_cache/*.pkl
	@echo "🗑️  Cache scripts OSMnx (scripts/general/cache)..."
	rm -f scripts/general/cache/*.pkl
	@echo "🗑️  Cache pipeline eqasim (services/eqasim-toulouse/cache)..."
	rm -rf services/eqasim-toulouse/cache/*.cache
	@echo "🗑️  Cache eqasim (data/cache/eqasim) + generated population (data/population)..."
	rm -rf data/cache/eqasim/*.cache
	rm -f data/population/*.json
	@echo "🗑️  Cache RAPTOR/Solari..."
	rm -f services/llm-agents/raptor_cache.pickle
	@echo "✅ All caches purged."
