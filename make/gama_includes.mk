# ──────────────────────────────────────────────────────────────────────────────
# GAMA — regenerate all of services/GAMA/CityTransport/includes/
# ──────────────────────────────────────────────────────────────────────────────
#
# `includes/` is not published (public copy): each layer is rebuilt from
# a versioned recipe. This target chains them, into a chosen folder.
#
#   make gama-includes                                   # writes into includes/ (the replaced
#                                                        # layers go into archives_<date>/)
#   make gama-includes INCLUDES_OUT=/tmp/regen            # elsewhere, to compare
#   make gama-includes INCLUDES_OUT=/tmp/regen ANNEXES=1 BDTOPO_ARCHIVE=…/BDTOPO_…_D031_….7z
#   make gama-includes-compare INCLUDES_OUT=/tmp/regen    # identical / equivalent / different table
#
# Layers READ by the model (Settings.gaml) — always produced, in this order:
#   1. perimetre_453.shp          scripts/data/gama/gama_includes.py perimetre (→ export_perimetre_shapefile.py)
#   2. routes.shp, stops.shp      scripts/data/gama/export_gtfs_layers.py
#   3. trip_info.json             scripts/data/gama/export_trip_info.py (reads routes.shp from step 2)
#      + shape_lookup.json        (same run; read by the Python runtime, not by GAMA)
# ANNEX layers — read by no model, produced only with ANNEXES=1:
#   4. Toulouse_bbox_p95.osm.pbf  gama_includes.py osm-p95   (osmium; Geofabrik source 2022-01-01)
#   5. building.shp               gama_includes.py batiments (BD TOPO IGN D031, BDTOPO_ARCHIVE=…)
#   6. toulouse_map.png           gama_includes.py fond-carte — NOT reproducible as is: the CartoDB
#                                 Positron tiles now require an API key (observed on 2026-09-28).
#                                 Produced only with FOND_CARTE=1 (image to check by eye).
# Same variables as gama-layers / gama-trip-info: FEEDS, DATE, START, DAYS.
# Exit codes: 0 written, 1 missing resource, 2 invariant contradicted — the chain stops at the first failure.

.PHONY: gama-includes gama-includes-compare test-gama-includes-recettes

INCLUDES_OUT ?= services/GAMA/CityTransport/includes
INCLUDES_REF ?= services/GAMA/CityTransport/includes
ANNEXES ?= 0
FOND_CARTE ?= 0
OSM_P95_SOURCE ?= services/eqasim-toulouse/data/osm_toulouse/midi-pyrenees-220101.osm.pbf
BDTOPO_ARCHIVE ?=
GAMA_INCLUDES_PY = scripts/data/gama/gama_includes.py

gama-includes:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make gama-includes SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	@mkdir -p $(INCLUDES_OUT)
	@t0=$$(date +%s); echo "▶ gama-includes → $(INCLUDES_OUT) (annexes: $(ANNEXES)) — start $$(date '+%F %T')"; \
	echo "── 1/3 perimeter" && \
	$(SYNTHESIS_PYTHON) $(GAMA_INCLUDES_PY) perimetre --sortie $(INCLUDES_OUT) && \
	echo "── 2/3 routes/stops layers" && \
	$(SYNTHESIS_PYTHON) scripts/data/gama/export_gtfs_layers.py \
	  $(foreach f,$(FEEDS),--feed $(f)) --sortie $(INCLUDES_OUT) \
	  --json $(INCLUDES_OUT)/.gama_layers.json > /dev/null && \
	echo "   routes.shp / stops.shp written (measurements: $(INCLUDES_OUT)/.gama_layers.json)" && \
	echo "── 3/3 runs + shape table" && \
	$(SYNTHESIS_PYTHON) scripts/data/gama/export_trip_info.py \
	  $(foreach f,$(FEEDS),--feed $(f)) \
	  $(if $(DATE),--date-simulee $(DATE),) $(if $(START),--debut $(START),) \
	  $(if $(DAYS),--jours $(DAYS),) \
	  --routes $(INCLUDES_OUT)/routes.shp --sortie $(INCLUDES_OUT)/trip_info.json \
	  --json $(INCLUDES_OUT)/.gama_trip_info.json > /dev/null && \
	echo "   trip_info.json / shape_lookup.json written (measurements: $(INCLUDES_OUT)/.gama_trip_info.json)" && \
	if [ "$(ANNEXES)" = "1" ]; then \
	  echo "── annex: Toulouse_bbox_p95.osm.pbf" && \
	  $(SYNTHESIS_PYTHON) $(GAMA_INCLUDES_PY) osm-p95 --source $(OSM_P95_SOURCE) --sortie $(INCLUDES_OUT) && \
	  echo "── annex: building.shp" && \
	  $(SYNTHESIS_PYTHON) $(GAMA_INCLUDES_PY) batiments --archive "$(BDTOPO_ARCHIVE)" --sortie $(INCLUDES_OUT) && \
	  if [ "$(FOND_CARTE)" = "1" ]; then \
	    echo "── annex: toulouse_map.png (forced, to check by eye)" && \
	    $(SYNTHESIS_PYTHON) $(GAMA_INCLUDES_PY) fond-carte --forcer --sortie $(INCLUDES_OUT); \
	  else echo "── annex toulouse_map.png skipped: not reproducible (CARTO tiles behind an API key); FOND_CARTE=1 to force"; fi; \
	else echo "── annexes skipped (3 layers read by no model; ANNEXES=1 to produce them)"; fi; \
	code=$$?; duree=$$(( $$(date +%s) - t0 )); \
	if [ $$code -eq 0 ]; then echo "✅ gama-includes finished in $${duree} s → $(INCLUDES_OUT)"; \
	else echo "❌ [ALARME] gama-includes interrupted (code $$code) after $${duree} s → $(INCLUDES_OUT)"; fi; \
	exit $$code

## Compares a regenerated folder (INCLUDES_OUT) with the reference (INCLUDES_REF): features, extent,
## columns, sha256, geometry fingerprints. Full JSON in $(INCLUDES_OUT)/.comparaison.json.
gama-includes-compare:
	@$(SYNTHESIS_PYTHON) $(GAMA_INCLUDES_PY) comparer --reference $(INCLUDES_REF) \
	  --candidat $(INCLUDES_OUT) --json $(INCLUDES_OUT)/.comparaison.json > /dev/null
	@echo "Comparison: $(INCLUDES_OUT)/.comparaison.json"

## Tests of the complementary recipes and of the comparator (synthetic shapefiles, < 3 s, offline).
test-gama-includes-recettes:
	@$(SYNTHESIS_PYTHON) -m pytest scripts/tests/test_gama_includes_recettes.py -q
