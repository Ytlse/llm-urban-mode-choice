# ──────────────────────────────────────────────────────────────────────────────
# Mode choice model (ticket 005)
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: zones housing-type bike-ownership terminal-time car-availability policy policy-tune common-set-predict equipment-propensity
.PHONY: logit mnl-predict klr klr-predict bi-oracle forest
.PHONY: communes-couronnes audit-perimetre audit-couronnes residence-zone

## ──────────────────────────────────────────────────────────────────────────────
## Population scope (ticket 020)
## ──────────────────────────────────────────────────────────────────────────────

## Rebuild the commune → couronne table and the couronne geometry (ticket 020, lot 3).
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
## This is the ticket's MISSING DATA: the survey splits its rings by LIST OF
## COMMUNES (1 / 69 / 108 / 275), whereas `geo_reference.residence_zone` classifies by
## distance to the hypercentre. Produces packages/mobility_core/src/mobility_core/data/commune_couronne.json and
## couronne_perimetre.geojson, both versioned.
communes-couronnes:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_commune_couronne

## Audit of the nine baseline gaps between the surveyed and the simulated population
## (ticket 020, lot 2). Does NOT require the PROGEDO data: it reads the framing
## `population_emc2_2023.yaml` and the versioned resources of `make communes-couronnes`.
##   make audit-perimetre                                  # default population and run
##   make audit-perimetre POP=data/population/x.json RUN=experiments/archive/y
##   make audit-perimetre TRACE=docs/traces/2026-08-24_perimetre_population
## Exit codes: 0 all compliant, 2 at least one axis to correct, 3 at least one axis
## NOT MEASURABLE — an unmeasured axis is an axis that passes, and the script refuses to hide it.
## The two equivalences of ticket 021, lot 0: classifying a home by fine-zone code PREFIX
## versus classifying it by geometric MEMBERSHIP, and "outside the fine-zone
## layer" versus "outside the scope". Seven gates, including an INDEPENDENT cross-check
## against the ticket 020 trace. Modifies nothing.
##   make audit-couronnes
##   make audit-couronnes POP=data/population/x.json TRACE=docs/traces/y
## Exit codes: 0 the gates pass, 2 a gate FAILS (the ticket must be redesigned),
## 3 a gate is NOT MEASURABLE — restricted-access GIS data missing, and an unmeasured
## gate is a gate that passes. After lot 1, the versioned table replaces the GIS.
## Sets the residence ring and the commune on an already generated population
## (ticket 021, lot 2 — stage D). Deterministic, no draw, no LLM call: the trait is
## OBSERVED. The resource itself is (re)produced by `make communes-couronnes`.
##   make residence-zone                                    # default population
##   make residence-zone POP=data/population/x.json CHECK=1
##   make residence-zone POP=experiments/archive/y/population_1000.json OUT=/tmp/z.json
## ⚠ NEVER enrich IN PLACE a population pinned by a frozen-set manifest
## (calibration_datasets/v5..v8 pin the sha256 of the 2026-08-19_14_36 archive): go
## through OUT=. CHECK exit codes: 0 gates passed, 1 missing resource, 2 a gate
## contradicted, 4 gates passed but gap to the framing (axis A9 — the draw, not this trait).
## The target TRANSLATES 4 into success, and says so: make cannot tell an informative
## exit code from an error, and an "Error 4" would teach people to ignore errors.
residence-zone:
	@$(SYNTHESIS_PYTHON) -m scripts.data.population.enrich_residence_zone \
	  $(if $(POP),$(POP),data/population/toulouse_population_1000.json) \
	  $(if $(OUT),--out $(OUT),) $(if $(CHECK),--check,) $(if $(DRY),--dry-run,) ; \
	code=$$? ; \
	if [ $$code -eq 4 ]; then \
	  echo "→ code 4: gap to the framing (axis A9, the draw) — NOT a failure of this trait." ; \
	  exit 0 ; \
	fi ; \
	exit $$code

audit-couronnes:
	$(SYNTHESIS_PYTHON) -m scripts.data.population.audit_couronne_equivalences \
	  $(if $(POP),--population $(POP),) $(if $(TRACE),--trace $(TRACE),)

audit-perimetre:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make audit-perimetre SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.data.population.audit_perimetre \
	  $(if $(POP),--population $(POP),) $(if $(RUN),--run $(RUN),) \
	  $(if $(TRACE),--trace $(TRACE),)

.PHONY: osmnx-perimeter-graph

## OSMnx graphs (walk, bike, car) of the polygon of the 453 communes of the survey scope
## (ticket 031 § 1.4): extracted from the eqasim fork's regional OSM pbf by `osmium extract`, with
## production network filters and speeds, cached in data/cache/osmnx/graphs_<key>.pkl under a key
## distinct from the 30 km disc. No download. The generate_population notebook requires this
## graph for steps 4+5. FORCE=1 rebuilds; TRACE=<folder> archives the measurements.
##   make osmnx-perimeter-graph TRACE=docs/traces/$$(date +%Y-%m-%d_%H-%M)_graphe_osmnx_perimetre_453
osmnx-perimeter-graph:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; exit 1; }
	@command -v osmium >/dev/null || test -x /opt/homebrew/bin/osmium || { \
	  echo "osmium not found: brew install osmium-tool"; exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.data.population.build_osmnx_perimeter_graph \
	  $(if $(FORCE),--force,) $(if $(TRACE),--trace $(TRACE),)

.PHONY: synthese-representativite

## HTML representativeness synthesis of a sealed population (visual identity of synthesis v2,
## figures read from report.json / selection.json / the seal's MANIFEST, the pool check and the audit).
##   make synthese-representativite SCEAU=data/population/population_1000_PANEL_v4 \
##        PRECEDENT=data/population/population_1000_PANEL_v3 VIVIER=docs/traces/<d>_controle_vivier/report.json \
##        AUDIT=docs/traces/<d>_audit/audit_perimetre.json OUT=docs/traces/<d>/synthese_representativite_v3.html \
##        VELO=docs/traces/<d>/velo_cohorte.json VELO_VIVIER=docs/traces/<d>/velo_vivier.json \
##        COPIE=$(PAPER_DIR)/methode/population/synthese_representativite_v3_population_v4_<date>.html
## The two bike reports come from `enrich_personal_bike <population> --dry-run --check --rapport-json <file>`
## on the sealed cohort and on the pre-imputed pool (Temp/4_zone_enriched): the slope is judged on the pool.
synthese-representativite:
	@test -n "$(SCEAU)" || { echo "SCEAU=<sealed folder> required"; exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.panel.synthese_representativite --sceau $(SCEAU) \
	  $(if $(PRECEDENT),--precedent $(PRECEDENT),) $(if $(VIVIER),--vivier $(VIVIER),) \
	  $(if $(AUDIT),--audit $(AUDIT),) $(if $(VELO),--velo $(VELO),) $(if $(VELO_VIVIER),--velo-vivier $(VELO_VIVIER),) \
	  --out $(if $(OUT),$(OUT),$(SCEAU)/synthese_representativite.html) \
	  $(if $(COPIE),--copie $(COPIE),)

.PHONY: synthese-generation-population

## Page "How the test set population is made" (eqasim, notebook, selection, routing,
## traits, check, seal — and the results of each stage), figures read from the same files as the
## representativeness synthesis plus the OSMnx graph metadata and the graph measurements.
##   make synthese-generation-population SCEAU=data/population/population_1000_PANEL_v4 \
##        VIVIER=… AUDIT=… VELO=… VELO_VIVIER=… MESURES_GRAPHE=docs/traces/<d>_mesures_graphe_perimetre_v4/mesures.json \
##        OUT=docs/traces/<d>/fabrication_population.html COPIE=$(PAPER_DIR)/methode/population/fabrication_population_v4_<date>.html
synthese-generation-population:
	@test -n "$(SCEAU)" || { echo "SCEAU=<sealed folder> required"; exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.panel.synthese_generation_population --sceau $(SCEAU) \
	  $(if $(VIVIER),--vivier $(VIVIER),) $(if $(AUDIT),--audit $(AUDIT),) \
	  $(if $(VELO),--velo $(VELO),) $(if $(VELO_VIVIER),--velo-vivier $(VELO_VIVIER),) \
	  $(if $(MESURES_GRAPHE),--mesures-graphe $(MESURES_GRAPHE),) \
	  --out $(if $(OUT),$(OUT),$(SCEAU)/fabrication_population.html) $(if $(COPIE),--copie $(COPIE),)

.PHONY: reference-marges control-population select-population seal-population

## Check of the test set population (article, milestone 0 of the protocol).
## Compares a synthetic population with the EMC² 2023 margins — age classes, occupation,
## car ownership (person basis and household basis), ring, ring × car ownership crossing —
## with CI95, TOST at ± BORNE pt, χ² + Cramér's V, EMD/JSD, the protocol's cross-check
## log and a synthesis of the gaps. Code 1 if an "à corriger" remains.
##   make control-population                                   # default population
##   make control-population POP=data/population/x.json BORNE=1.0 TRACE=docs/traces/y
control-population:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make control-population SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.panel.control_population \
	  $(if $(POP),$(POP),data/population/toulouse_population_1000.json) \
	  $(if $(BORNE),--borne $(BORNE),) $(if $(TRACE),--trace $(TRACE),--trace-auto) $(if $(JSON),--json $(JSON),)

## The reference margins, with their source (report page or frozen recomputation).
## RECOMPUTE=1 re-freezes the joint ring × car ownership target from the ProGEDO microdata.
reference-marges:
	$(SYNTHESIS_PYTHON) -m scripts.panel.reference_marges $(if $(RECOMPUTE),--recompute,)

## Stratified selection of N personas from a pool (before routing — step 3ter of the
## notebook calls it). POOL required; OUT default: <pool folder>/toulouse_population_<N>_PANEL.json
##   make select-population POOL=scripts/data/population/Temp/4_zone_enriched/toulouse_population_5000.json N=1000
select-population:
	@test -n "$(POOL)" || { echo "POOL=<pool.json> required"; exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.panel.seal_population select --pool $(POOL) \
	  --n $(if $(N),$(N),1000) \
	  --out $(if $(OUT),$(OUT),$(dir $(POOL))toulouse_population_$(if $(N),$(N),1000)_PANEL.json)

## Sealing: check then copy into an immutable folder with MANIFEST.yaml and CONTROLE.md.
## REFUSES if a margin is "à corriger". POP required; OUT_DIR default: data/population/population_1000_PANEL_v4
## (v4 selection rule: whole households + six age classes + scope of the 453 communes,
## ticket 031; the v2 and v3 folders remain intact)
##   make seal-population POP=data/population/toulouse_population_1000_PANEL.json \
##        SELECTION=scripts/data/population/Temp/4_zone_enriched/toulouse_population_1000_PANEL_selection.json
seal-population:
	@test -n "$(POP)" || { echo "POP=<population.json> required"; exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.panel.seal_population seal --population $(POP) \
	  $(if $(OUT_DIR),--out-dir $(OUT_DIR),) $(if $(N),--n $(N),) \
	  $(if $(SELECTION),--selection-json $(SELECTION),) $(if $(BORNE),--borne $(BORNE),) \
	  $(if $(NOTE),--note "$(NOTE)",)

.PHONY: gtfs-year gtfs-year-dry gtfs-year-holdout gtfs-window test-gtfs-year

## Rebuilds a GTFS feed covering the whole year from the operator's partial
## exports. Each day carries either the real published offer, or the verbatim
## copy of a real day with the same signature (weekday × zone C school
## period); no timetable is synthesised, and the provenance of each
## day is traced under docs/traces/<date>_gtfs_annee/.
##   make gtfs-year                              # Tisséo + TER, 2026 and 2027
##   make gtfs-year RESEAU=tisseo ANNEES="2026"
## Exit codes: 0 all held, 1 missing resource, 2 invariant contradicted
## (the feed must NOT be published), 4 built but with degraded confidence.
## The target TRANSLATES 4 into success, and says so: a yearly feed built on six
## months of exports necessarily contains days extrapolated from far away, and an
## "Error 4" would teach people to ignore errors.
gtfs-year:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make gtfs-year SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	@$(SYNTHESIS_PYTHON) -m scripts.data.gtfs_year.build_year_feed \
	  $(foreach r,$(RESEAU),--reseau $(r)) \
	  $(foreach a,$(ANNEES),--annee $(a)) \
	  $(if $(SORTIE),--sortie $(SORTIE),) $(if $(TRACE),--trace $(TRACE),) \
	  $(if $(DRY),--dry-run,) $(if $(HOLDOUT),--holdout $(HOLDOUT),) \
	  $(if $(REFRESH),--rafraichir-calendrier,) ; \
	code=$$? ; \
	if [ $$code -eq 4 ]; then \
	  echo "→ code 4: feed built, but some days are extrapolated without a donor of the same kind." ; \
	  echo "  Read docs/traces/*_gtfs_annee/provenance_*.csv before publishing." ; \
	  exit 0 ; \
	fi ; \
	exit $$code

## Plans without writing anything: which days are real, which would be
## copied and from when. To run before any build after receiving exports.
gtfs-year-dry:
	@$(MAKE) gtfs-year DRY=1

## Masks a real month and measures the gap between the extrapolated offer and the offer
## actually published that month. It is the only proof that the extrapolation
## model is worth anything. Reference measurement on May 2026: maximum
## gap 5.3 %, median under 1 %.
##   make gtfs-year-holdout HOLDOUT=202605
gtfs-year-holdout:
	@$(MAKE) gtfs-year RESEAU=tisseo ANNEES=2026 HOLDOUT=$(if $(HOLDOUT),$(HOLDOUT),202605) \
	  SORTIE=/tmp/gtfs_year_holdout

## Extracts from the yearly feed the window that GAMA and the runtime consume. OTP reads
## the whole year, GAMA does not: its calendar is a 64-bit binary mask
## (services/llm-agents/inputs/gtfs/gama.py, PublicTransport.gaml). The window MUST
## contain the simulation date, otherwise no run is scheduled any more.
##   make gtfs-window START=2026-03-16 DAYS=64
gtfs-window:
	@$(SYNTHESIS_PYTHON) -m scripts.data.gtfs_year.window_feed \
	  --source $(if $(SOURCE),$(SOURCE),data/gtfs_year/tisseo_2026) \
	  --debut $(if $(START),$(START),2026-03-16) \
	  --jours $(if $(DAYS),$(DAYS),64) \
	  --sortie $(if $(OUT),$(OUT),data/gtfs_year/fenetre_gama) --zip

## Unit tests of the yearly feed pipeline. Synthetic feeds, no network
## access, under one second. Each test covers a decision that, taken the
## wrong way, produces a plausible but wrong feed.
test-gtfs-year:
	@$(SYNTHESIS_PYTHON) -m pytest scripts/tests/test_gtfs_year.py -q

.PHONY: gama-layers gama-trip-info test-gama-includes

## Rebuilds the LAYERS that GAMA draws — services/GAMA/CityTransport/includes/routes.shp
## and stops.shp — from the three networks of the scope (Tisséo, TER, liO). The
## previous layers are moved into an archives_<date> folder, never
## deleted. `includes/` is not versioned: this recipe is the only trace.
##   make gama-layers
##   make gama-layers FEEDS="tisseo=data/gtfs/tisseo_gtfs ter=data/gtfs/ter_gtfs"
gama-layers:
	@$(SYNTHESIS_PYTHON) scripts/data/gama/export_gtfs_layers.py \
	  $(foreach f,$(FEEDS),--feed $(f)) \
	  $(if $(OUT),--sortie $(OUT),) $(if $(TOUT),--tout,) $(if $(JSON),--json $(JSON),)

## Rebuilds the RUNS that GAMA drives — services/GAMA/CityTransport/includes/trip_info.json —
## from the three networks and the simulated date (read from Settings.gaml), THEN the table
## of shapes that the runtime reads (includes/shape_lookup.json).
##
## The table publishes the route_id -> shape_id -> stop order mapping that the recipe
## ACTUALLY used: it is what lets an agent board a vehicle
## (get_shape_id_from_route_info, then `shape_id_list contains each.shape_id` on the GAMA side).
## The runtime does not recompute it — TER publishes no geometry, its shape_ids are
## made here, and the recipe discards runs that would otherwise have to be discarded
## identically. It records the fingerprint of routes.shp/.dbf and of trip_info.json: rebuilding
## the layers ALONE mismatches it, and the runtime raises an alarm on loading.
##
## The layers are rebuilt FIRST, in the same recipe: `trip_info.json` carries
## vertex indices into the geometry of `routes.shp`, and producing one without the other is
## exactly the defect that lasted five months — layers with three networks, runs with one,
## 34 TER lines drawn where no train ran. `COUCHES=0` skips this step
## when the layers have just been made.
##
## Blocking checks (the file is not written if they fail): the simulated date is
## in the window AND served; the date range fits in GAMA's 64-bit binary
## mask; each run has its shape in routes.shp with the same number of points; no
## route_type of the layer is without a run on the simulated day.
##   make gama-trip-info
##   make gama-trip-info DATE=2026-03-16 DAYS=64 COUCHES=0
##   make gama-trip-info TABLE=/tmp/shape_lookup.json     # table elsewhere than next to the runs
## Exit codes: 0 written, 1 missing resource, 2 invariant contradicted.
gama-trip-info:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make gama-trip-info SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	@if [ "$(COUCHES)" != "0" ]; then $(MAKE) --no-print-directory gama-layers ; fi
	@$(SYNTHESIS_PYTHON) scripts/data/gama/export_trip_info.py \
	  $(foreach f,$(FEEDS),--feed $(f)) \
	  $(if $(DATE),--date-simulee $(DATE),) $(if $(START),--debut $(START),) \
	  $(if $(DAYS),--jours $(DAYS),) $(if $(ROUTES),--routes $(ROUTES),) \
	  $(if $(OUT),--sortie $(OUT),) $(if $(TABLE),--table-traces $(TABLE),) \
	  $(if $(JSON),--json $(JSON),)

## Unit tests of the two recipes above, including the layers/runs consistency
## check and the published shape table: minimal synthetic feeds, no
## large file, no network access.
test-gama-includes:
	@$(SYNTHESIS_PYTHON) -m pytest scripts/tests/test_gama_includes.py -q

## Rebuild the fine-zone resource read by mobility_core.zone_resolver.
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
zones:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_zone_layer

## Rebuild the housing-type law read when enriching a synthetic population (action A2).
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
## Ticket 019: the law is conditioned on the FINE ZONE and the HOUSEHOLD SIZE, and the
## resource produced is in v2 — the module refuses a v1. The export publishes the EMC²
## internal test and FAILS if the mechanism's error exceeds 1 point over the 20 cells.
## Then, to set the trait on a population (no LLM call, deterministic):
##   $(VENV_PYTHON) -m scripts.data.population.enrich_housing_type \
##     data/population/toulouse_population_1000.json --check
## Exit codes of --check: 0 everything within tolerance, 1 missing resource,
## 2 a served target is contradicted, 3 population enriched but too small to decide.
housing-type:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_housing_type

## Rebuild the bike-ownership model read when enriching a synthetic population
## (ticket 015: the three stages k / attribution / e-bike learned on EMC²).
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
## It also reads the housing-type table (make housing-type) to publish the DILUTED
## ownership target by housing — the only one comparable to a synthetic population.
## Then, to set the trait on a population (no LLM call, deterministic):
##   $(VENV_PYTHON) -m scripts.data.population.enrich_personal_bike \
##     data/population/toulouse_population_1000.json --check
bike-ownership:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_bike_ownership

## Rebuild the two equipment-propensity laws read when enriching a population:
## `has_pt_subscription` (ticket 016) and `has_driving_license` (ticket 017).
## Lot 1 shared by both tickets: a single loader, two targets learned on the
## EMC² standard `pers` file (PENQ = 1, COEP weighting), cross-validation
## GROUPED BY HOUSEHOLD. The fare thresholds (under 26, senior eligibility)
## are fitted then DECIDED on the out-of-sample AUC, not decreed.
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
##   make equipment-propensity DRY_RUN=1   # fits and displays the recipe, without writing
equipment-propensity:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_equipment_propensity \
	  $(if $(DRY_RUN),--dry-run,)

## Rebuild the EMC²-measured car terminal time (access + parking search) law.
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
## What it measures: T2 (walk at departure), T6 (walk at arrival) and T11 (parking
## search time) of the trips file, by ring. To compare with the values
## of services/llm-agents/config/terminal_time.yaml, measured 8x to 24x larger.
terminal-time:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_terminal_time

## Rebuild the EMC²-measured household car availability reference (ticket 018).
## Requires the restricted PROGEDO data under 'data/PROGEDO 2023/'.
## What it measures: `car_availability` (all / some / none) recomputed with EQASIM'S
## RULE — all if cars >= licences of adults, some if <, none if cars == 0 —
## from M6 (cars) and P7 (licences). Two weightings: households (COE0) and persons
## (COE1), the latter being the one comparable to an agent population.
## The export FAILS if its positive control does not reproduce the published car ownership
## (1.25 cars/household; 19 / 45 / 35 %): a reading that misses the fleet cannot claim
## to measure its availability.
car-availability:
	@test -d "data/PROGEDO 2023" || { \
	  echo "PROGEDO data missing: data/PROGEDO 2023/ (restricted access lil-1750)"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.export_car_availability

## Retrain the PROGEDO mode-choice policy → scripts/progedo_logit/mode_choice_policy.json
## The training parquet is versioned: unlike `zones`, this target
## does NOT require the raw PROGEDO data. Deterministic result (fixed seed).
policy:
	@test -f scripts/progedo_logit/progedo_mode_choice_v2.parquet || { \
	  echo "Training set missing: scripts/progedo_logit/progedo_mode_choice_v2.parquet"; \
	  echo "It is versioned; if it is missing, regenerate it with build_mode_choice_dataset.py"; \
	  echo "(which, itself, requires the restricted-access PROGEDO data)."; \
	  exit 1; }
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make policy SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.fit_mode_choice_policy

## Tune the mode-choice booster's hyperparameters → scripts/progedo_logit/mode_choice_tuning.json
## Cross-validation grouped BY HOUSEHOLD, entirely INSIDE the train: the test
## split is never read. Writes no model — the winner is copied by hand into
## PARAMS of fit_mode_choice_policy.py, then `make policy`.
##   make policy-tune TUNE_ARGS="--refine --trials 40"   # narrowed space (2nd pass)
policy-tune:
	@test -f scripts/progedo_logit/progedo_mode_choice_v2.parquet || { \
	  echo "Training set missing: scripts/progedo_logit/progedo_mode_choice_v2.parquet"; \
	  exit 1; }
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make policy-tune SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.tune_mode_choice_policy $(TUNE_ARGS)

## Apply the trained policy to the pinned common set, renormalised on the OTP offer
## (action A8) → scripts/synthesis/data/progedo_on_common_set.parquet
## No LLM call, no network, seed irrelevant: the result is deterministic.
## Requires the zone layer (`make zones`) for the six geographic variables.
##   make common-set-predict DRY_RUN=1   # scope and statuses, without writing
common-set-predict:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make common-set-predict SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.synthesis.model_on_common_set \
	  $(if $(DRY_RUN),--dry-run,) $(if $(POLICY),--policy $(POLICY),) $(if $(OUT),--out $(OUT),)

## Estimate the SECOND oracle: multinomial logit at strict parity (21 variables)
## → scripts/progedo_logit/mnl_model.json + mnl_model_metrics.json
## Same set, same split by household, same sample_weight and same metrics as
## `make policy`: comparing the two oracles measures only the two models.
## The regularisation is chosen by grouped cross-validation WITHIN the train.
##   make logit C=1.0        # imposes C instead of choosing it (diagnostic)
logit:
	@test -f scripts/progedo_logit/progedo_mode_choice_v2.parquet || { \
	  echo "Training set missing: scripts/progedo_logit/progedo_mode_choice_v2.parquet"; \
	  echo "It is versioned; if it is missing, regenerate it with build_mode_choice_dataset.py"; \
	  exit 1; }
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make logit SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.fit_mode_choice_logit $(if $(C),--C $(C),)

## Random-forest WITNESS: does the booster's edge come from the trees or the boosting?
## → scripts/progedo_logit/rf_mode_choice_metrics.json (measurements only, NO model)
## Same parquet, split by household, COEP sample_weight and metrics as `make policy` and
## `make logit`. Tuning by grouped cross-validation WITHIN the train; neither class_weight nor
## rebalancing. Verdict at a threshold declared in advance: share of the booster–logit gap closed,
## ≥ 0.70 → the trees, ≤ 0.30 → the boosting. Offline, ~20 min.
## Prerequisite for the verdict: make policy && make logit (otherwise "non mesuré")
##   make forest FOREST_ARGS="--encodage dessin"   # without the sensitivity path
##   make forest FOREST_ARGS="--rapide"            # smoke pass, figures not publishable
forest:
	@test -f scripts/progedo_logit/progedo_mode_choice_v2.parquet || { \
	  echo "Training set missing: scripts/progedo_logit/progedo_mode_choice_v2.parquet"; \
	  echo "It is versioned; if it is missing, regenerate it with build_mode_choice_dataset.py"; \
	  exit 1; }
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make forest SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.fit_mode_choice_forest $(FOREST_ARGS)

## Apply the SECOND oracle to the pinned common set (same code path as the booster)
## → scripts/synthesis/data/mnl_on_common_set.parquet
mnl-predict:
	@$(MAKE) --no-print-directory common-set-predict \
	  POLICY=scripts/progedo_logit/mnl_model.json \
	  OUT=scripts/synthesis/data/mnl_on_common_set.parquet

## Estimate the THIRD family: kernel logistic regression (RBF + Nyström, strict parity)
## → scripts/progedo_logit/klr_model.json + klr_model_metrics.json
## Same set, same split by household, same sample_weight, SAME design matrix as
## `make logit` and same metrics: comparing the three families measures only the
## three models. γ, λ and m are chosen by grouped cross-validation WITHIN the train, and
## a configuration that drifts on the modal shares is discarded before any ranking.
## Offline, deterministic (fixed seed), ~20 min for the full bench.
##   make klr KLR_ARGS="--gamma 0.0418 --C 1 --m 1000"   # imposed setting, no bench
##   make klr KLR_ARGS="--m-grid 500 1000"               # shortened bench
klr:
	@test -f scripts/progedo_logit/progedo_mode_choice_v2.parquet || { \
	  echo "Training set missing: scripts/progedo_logit/progedo_mode_choice_v2.parquet"; \
	  echo "It is versioned; if it is missing, regenerate it with build_mode_choice_dataset.py"; \
	  exit 1; }
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make klr SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.progedo_logit.fit_mode_choice_klr $(KLR_ARGS)

## Apply the THIRD family to the pinned common set (same code path as the two others)
## → scripts/synthesis/data/klr_on_common_set.parquet
klr-predict:
	@$(MAKE) --no-print-directory common-set-predict \
	  POLICY=scripts/progedo_logit/klr_model.json \
	  OUT=scripts/synthesis/data/klr_on_common_set.parquet

## Two-oracle composite score → scripts/synthesis/data/bi_oracle.json
## Block A EMC² fidelity (unchanged), block B disaggregated agreement with the logit relative to the
## inter-oracle distance, block C direction of variation. Weights B and C at 0: the terms are
## published, they select no prompt. No LLM call, no network.
## Block C with TWO arbiters if the KLR parquet is present (make klr && make klr-predict): a
## transition where logit and KLR diverge leaves the score instead of being blamed on the prompt.
## Prerequisites: make logit && make common-set-predict && make mnl-predict
##                (+ make klr && make klr-predict for the second arbiter)
bi-oracle:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make bi-oracle SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.synthesis.bi_oracle
