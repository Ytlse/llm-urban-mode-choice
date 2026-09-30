# ──────────────────────────────────────────────────────────────────────────────
# Memory experiments (Ticket 109)
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: experience-memoire-lancer experience-memoire-estimer experience-memoire-nuit banc-reprise

## Launches an A/B memory experiment (treated then control arm, back to back, Ticket 109).
## Usage: make experience-memoire-lancer EXP=<name> [DRY_RUN=1] [BRANCHE=both|treated|control]
experience-memoire-lancer:
	@test -n "$(EXP)" || { echo "❌ Variable EXP missing (e.g.: make experience-memoire-lancer EXP=exp_mem_...)"; exit 1; }
	$(VENV_PYTHON) scripts/experiment/orchestrateur_memoire.py --experience $(EXP) $(if $(filter 1,$(DRY_RUN)),--dry-run,) $(if $(BRANCHE),--branche $(BRANCHE),)

## Estimates the cost in requests and tokens of a memory experiment.
## Usage: make experience-memoire-estimer EXP=<name>
experience-memoire-estimer:
	@test -n "$(EXP)" || { echo "❌ Variable EXP missing (e.g.: make experience-memoire-estimer EXP=exp_mem_...)"; exit 1; }
	$(VENV_PYTHON) scripts/experiment/orchestrateur_memoire.py --experience $(EXP) --estimer

## Ticket 118 functional bench (≈ 30 min, one arm, 6 personas, 2 simulated days): the real
## night chain, a simulated 4 "hour" survey outage, the pause, the resume. Run it before a
## full memory campaign. Report: docs/traces/<date>_banc_reprise_118/BILAN.md. Exit 1 on failure.
## The multi-attempt outage itself is checked in seconds by scripts/tests/test_chaine_memoire_panne_fournisseur.py.
## Usage: make banc-reprise [BANC_ATTENTE_S=60]
banc-reprise:
	@env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS $(if $(BANC_ATTENTE_S),BANC_ATTENTE_S=$(BANC_ATTENTE_S),) \
		$(VENV_PYTHON) scripts/experiment/banc_reprise_chaine.py

## Chains the memory experiments one by one, in the background, for an unattended night.
## Without EXP: all declared experiments not yet finished, in creation order.
## Transient stop (qualified provider overload): new attempt every ATTENTE_S s, ESSAIS_MAX times
## without progress. Defect (anything else, e.g. a broken common prefix): stopped, no new attempt.
## Quota exhausted: next experiment. Log: experiments/enchainement_nuit_<date>.log
## Usage: make experience-memoire-nuit [EXP="exp_a exp_b"] [ATTENTE_S=3600] [ESSAIS_MAX=6]
experience-memoire-nuit:
	@env -u MAKEFLAGS -u MAKELEVEL -u MFLAGS ATTENTE_S=$(or $(ATTENTE_S),3600) ESSAIS_MAX=$(or $(ESSAIS_MAX),6) \
		nohup $(if $(shell command -v caffeinate),caffeinate -i,) bash scripts/experiment/enchainer_experiences_memoire.sh $(EXP) >/dev/null 2>&1 &
	@sleep 3; journal=$$(ls -t experiments/enchainement_nuit_*.log 2>/dev/null | head -1); \
		cat "$$journal"; echo "🌙 Follow: tail -f $$journal"
