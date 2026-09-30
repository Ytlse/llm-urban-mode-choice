# ──────────────────────────────────────────────────────────────────────────────
# Score synthesis
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: synthesis synthesis-open \
        common-set-eval heldout-eval \
        model-compare model-compare-open

# The synthesis imports pandas/numpy and the calibration engine: the system
# python3 is not enough. The project venv is used, and can be overridden.
SYNTHESIS_PYTHON ?= $(VENV_PYTHON)

# Fetch of the cloud campaign store before each synthesis (PULL=0 to
# skip it, e.g. offline). The campaign runs on the VM: without this pull, the
# page's calibration column reflects a stale local snapshot.
SYNTHESIS_PULL_DB := prompt_calibration/calibration_results/calibration_cloud.db
PULL ?= 1


## Regenerate the score synthesis page. Usage: make synthesis [RUN=experiments/archive/2026-07-29_18_34] [PULL=0]
synthesis:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make synthesis SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.synthesis.build $(if $(RUN),--run $(RUN),)

## Regenerate then open the page in the default browser.
synthesis-open: synthesis
	open docs/synthesis/index.html

## Compare a run to its predecessors AND break its score down by LLM model.
## Usage: make model-compare RUN=experiments/archive/<run> [BASELINE="a b"] [OUT=…]
## No LLM call: everything is reread from moves.csv, with the reader and the loss of
## `make synthesis`. To use when a run ran several models — the
## main page, which aggregates the whole run, cannot separate them. Output:
## docs/synthesis/models/<run>/index.html
model-compare:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make model-compare SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	@test -n "$(RUN)" || { \
	  echo "RUN is mandatory: make model-compare RUN=experiments/archive/<run>"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.synthesis.model_compare --run $(RUN) \
	  $(foreach b,$(BASELINE),--baseline $(b)) $(if $(OUT),--out $(OUT),)

## Same, then opens the page.
model-compare-open: model-compare
	open docs/synthesis/models/$(notdir $(patsubst %/,%,$(RUN)))/index.html

## Re-evaluate the pinned prompt lineage's seed and leaf on the common set (action A3).
## CONSUMES LLM QUOTA (~130 Gemini free tier calls). Estimate first:
##   make common-set-eval DRY_RUN=1
## Free resume: evals already paid for are served by the store's cache.
common-set-eval:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make common-set-eval SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.synthesis.common_set_eval \
	  $(if $(DRY_RUN),--dry-run,) $(if $(PROVIDER),--provider $(PROVIDER),) \
	  $(if $(BATCH),--batch $(BATCH),)

## Evaluate the pinned prompt lineage on a HELD-OUT frozen split (action A4).
## It is the only calibration score that does not bear on the set used to
## optimise it. CONSUMES LLM QUOTA (~100 Gemini free tier calls for the 6 nodes
## of the lineage, ~35 for the two ends). Estimate first:
##   make heldout-eval DRY_RUN=1
##   make heldout-eval NODES=all PROVIDER=google2     # the whole lineage
## Free, per-node resume: evals already paid for are served by the store's
## cache. The sample-size witness is computed by `make synthesis` without LLM calls.
heldout-eval:
	@test -x $(SYNTHESIS_PYTHON) || { \
	  echo "Interpreter not found: $(SYNTHESIS_PYTHON)"; \
	  echo "Override it: make heldout-eval SYNTHESIS_PYTHON=/path/to/python"; \
	  exit 1; }
	$(SYNTHESIS_PYTHON) -m scripts.synthesis.heldout_eval \
	  $(if $(DRY_RUN),--dry-run,) $(if $(PROVIDER),--provider $(PROVIDER),) \
	  $(if $(BATCH),--batch $(BATCH),) $(if $(NODES),--nodes $(NODES),) \
	  $(if $(DATASET),--dataset $(DATASET),)
