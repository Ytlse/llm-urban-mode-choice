## ── Archived executions ─────────────────────────────────────────────────────
## The executions behind the paper's tables ship gzip-compressed
## (archive/1_regime_nominal/*/experiences/*/executions/*/{decisions.jsonl,moves.csv}.gz).
## The platform tools read the plain files: this target decompresses them next to the .gz,
## which stay in place. Idempotent.
##
## Usage : make unpack-runs [RUNS=archive/1_regime_nominal]
RUNS ?= archive/1_regime_nominal
.PHONY: unpack-runs
unpack-runs:
	@n=0; for gz in $$(find $(RUNS) -path '*/executions/*' \( -name 'decisions.jsonl.gz' -o -name 'moves.csv.gz' \)); do \
	  out=$${gz%.gz}; \
	  if [ ! -f "$$out" ] || [ "$$gz" -nt "$$out" ]; then gzip -dkf "$$gz" && n=$$((n+1)); fi; \
	done; echo "unpack-runs: $$n file(s) decompressed under $(RUNS)"

## ── Memory experiment definitions ───────────────────────────────────────────
## The definitions of the memory experiments (experience_memoire.yaml) ship under
## config/experiences_memoire/; the platform reads and writes them under
## data/experiences/evenements_non_tabules/ (EXPERIENCES_MEMOIRE_DIR), which git ignores
## because runs write their state there. This target copies each definition in place,
## never overwriting one that already exists. Idempotent.
##
## Usage : make experiences-memoire-installer
MEMOIRE_DEFINITIONS ?= config/experiences_memoire
MEMOIRE_DIR ?= $(or $(EXPERIENCES_MEMOIRE_DIR),data/experiences/evenements_non_tabules)
.PHONY: experiences-memoire-installer
experiences-memoire-installer:
	@test -d $(MEMOIRE_DEFINITIONS) || { echo "experiences-memoire-installer: ERROR $(MEMOIRE_DEFINITIONS) not found" >&2; exit 1; }
	@n=0; k=0; for f in $$(cd $(MEMOIRE_DEFINITIONS) && find . -name experience_memoire.yaml); do \
	  out="$(MEMOIRE_DIR)/$${f#./}"; \
	  if [ -f "$$out" ]; then k=$$((k+1)); else mkdir -p "$$(dirname "$$out")" && cp "$(MEMOIRE_DEFINITIONS)/$${f#./}" "$$out" && n=$$((n+1)); fi; \
	done; echo "experiences-memoire-installer: $$n definition(s) installed under $(MEMOIRE_DIR), $$k already present"
