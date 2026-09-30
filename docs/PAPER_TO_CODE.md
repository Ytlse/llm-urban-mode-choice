# From the paper to the code

This page maps each table and figure of the paper and its appendices to the script that
produces it, the command that runs it, and the data it reads. Elements that this repository
cannot regenerate are listed at the end, each with its reason.

The last column gives the **replication level** needed (Appendix M of the paper):

| Level | What it needs |
|---|---|
| 1 | Archived executions and the survey reference shares only; no model call, no router, no GAMA |
| 2 | The controller, the gateway and one provider key (new decisions on the frozen trip set) |
| 3 | The routers (OpenTripPlanner, OSMnx), and GAMA for the simulator results |
| 4 | The EMC² survey microdata, available on request only (see [DATA.md](DATA.md)) |

Conventions:

- Commands are run from the repository root with the project interpreter,
  `services/llm-agents/.venv/bin/python` (written `python` below), after `make unpack-runs`.
  No command of level 1 needs Docker, a provider key or a network access.
- `archive/c1/` stands for `archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences/`
  and `archive/c2/` for `archive/1_regime_nominal/jeu_1000_PANEL_v6_c2_EN_c/experiences/`.
- The read-only tools find the shipped executions on their own: they read `EXPERIENCES_DIR`
  if set, otherwise `data/experiences/` if it holds an execution, otherwise
  `archive/1_regime_nominal/` ([EXPERIMENTS.md](EXPERIMENTS.md), Section 6).
- Figures are written to `--sortie` (or `--out`) when the script has the option, and by
  default to `outputs/figures/`, created on the fly and ignored by git.
- `traces/…` stands for `docs/traces/…`: the folders of measurements the appendices cite.

Every level-1 command below was run on 2026-09-28 in a fresh clone of this repository, with
a new virtual environment installed as in [INSTALL.md](INSTALL.md). The values it printed
match the paper wherever the paper states them.

## Main paper

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Figure 1 (agent architecture) | Drawing, not generated | — | — | — |
| Table 1 (composite and L1 of the fifteen decision-makers) | `services/llm-agents/experiences/score.py`, `services/llm-agents/experiences/registre.py` | `python -m experiences score <execution>` recomputes one `scores.json` (from `services/llm-agents/`); `python -m experiences registre` lists them all (Section 6 of [EXPERIMENTS.md](EXPERIMENTS.md)) | Executions in `archive/c1/`; `scripts/data/population/cerema_values.yaml` | 1 |
| Table 2 (paired gains of the expert prompt) | `scripts/analysis/paired_intervals.py` | `python -m scripts.analysis.paired_intervals --preset chapter6 --out <dir>` (about 50 min for 2,000 replicates) | `moves.csv` of the twelve pinned executions in `archive/c1/` | 1 |
| Figure 2 (decision-makers on the composite axis) | `scripts/analysis/plot_decision_makers.py` (panel `ch6_echelle`) | `python scripts/analysis/plot_decision_makers.py --avec-jev [--sortie <dir>]` | `scores.json` of the pinned executions | 1 |
| Figure 3 (car share by distance band) | `scripts/analysis/plot_decision_makers.py` (panel `ch6_distance`) | same | same | 1 |
| Figure 4 (recall and precision by mode) | `scripts/analysis/plot_audit_unitaire.py` (panel `ch6_audit_modes`) | `python scripts/analysis/plot_audit_unitaire.py --audit docs/traces/2026-09-22_lot0_entropie_support_unique/audit_12_decideurs.json --avec-jev [--sortie <dir>]` | The audit output of twelve decision-makers, shipped with the cited traces | 1 to redraw, 4 to recompute the audit |
| Table 3 (trip-level agreement) | `scripts/progedo_logit/audit_unitaire_058.py`, arms in `scripts/progedo_logit/arms_058/` | Read: the fields `exactitude_ponderee`, `cel_weighted_support_commun` and `par_mode.*.rappel` of the audit JSON above. Recompute: `python scripts/progedo_logit/audit_unitaire_058.py [--jeu enquete_058_test_20260316] [--experience <name>]… [--sortie <json>]` | Survey-day trip set and its reported modes (restricted, not shipped) | 1 to read, 4 to recompute |
| Figure 5 (daily propensity after a shock) | `scripts/analysis/plot_shock_propensity.py` | Not regenerable here, see below | Two simulator runs with memory and a shock | 3 |
| Table 4 (shock runs) | `scripts/analysis/shock_figures.py`, `scripts/analysis/tableau_quatre_voies.py` | Not regenerable here, see below | Same simulator runs | 3 |

`--avec-jev` adds the typed classifier (the third decision-maker family) to Figures 2 to 4.
The paper shows it; without the option the scripts draw the thirteen other decision-makers.

## Appendix B. Cohort

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Tables B.9, B.10 (control margins of the sealed cohort) | `scripts/panel/control_population.py` | `make control-population POP=data/population/population_1000_PANEL_v6/population.json` | Sealed cohort c1 and the frozen reference margins; its `CONTROLE.md` and `report.json` hold the published values | 1 |
| Reference margins of Table B.9 | `scripts/panel/reference_marges.py` | `make reference-marges` | Frozen reference margins | 1 (4 with `RECOMPUTE=1`) |
| Seal of the cohort | `scripts/data/population/selectionner_personas_mesurables.py` | `make personas-verifier POP=data/population/population_1000_PANEL_v6` | `MANIFEST.yaml` of the cohort | 1 |
| Figures B.1, B.2 (perimeter and generation pipeline) | — | Not regenerable here, see below | `report.json` of the cohort | — |

## Appendix C. Variables and reference models

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Tables C.1–C.2 (the 21 variables of the contract) | `scripts/progedo_logit/feature_spec.json`, `scripts/progedo_logit/build_mode_choice_dataset.py` | — | — | — |
| Reference models: policy, logit, forest, kernel logistic regression | `scripts/progedo_logit/fit_mode_choice_policy.py`, `fit_mode_choice_logit.py`, `fit_mode_choice_forest.py`, `fit_mode_choice_klr.py`, `mode_choice_logit.py` | `make policy`, `make logit`, `make forest`, `make klr` | Training table built from the survey (not shipped) | 4 |
| Table C.8 (the four models on the test trips) | same, plus `scripts/progedo_logit/mode_choice_eval.py` | same | same | 4 |
| Reference models on the frozen trip set | `scripts/synthesis/model_on_common_set.py`; decider `services/llm-agents/experiences/decideur_modele.py` | `make common-set-predict`, `make mnl-predict`, `make klr-predict` | Fitted models (refit at level 4), frozen trip set of c1 | 4 to refit, then 1 |
| Executions of the reference models | archived | `python -m experiences registre` | `archive/c1/exp_*` of type `modele` | 1 |

## Appendix D. Metrics and intervals

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Composite score, EMD, JSD, L1 (definitions) | `services/llm-agents/experiences/score.py`, `services/llm-agents/experiences/formules/reference.yaml`, `scripts/synthesis/frames.py`, `scripts/synthesis/formule_score/` | `python -m experiences score <execution>` | `decisions.jsonl`, `moves.csv`, `scripts/data/population/cerema_values.yaml` | 1 |
| Figure D.1 (finite-sample bias) | `scripts/annexes/fig_D_finite_sample_bias.py` | `python scripts/annexes/fig_D_finite_sample_bias.py` (about 5 min; `--redraw` redraws from `scripts/annexes/data/fig_D_finite_sample_bias.json`) | `moves.csv` of three executions in `archive/c1/` | 1 |
| Table D.7 (half-widths per arm) | `scripts/analysis/paired_intervals.py` | `python -m scripts.analysis.paired_intervals --preset chapter6 --out <dir>` (key `arms`, field `comp_half_width`) | as Table 2 | 1 |

## Appendix E. Prompts and trace

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Prompt variants | `packages/mobility_llm/src/mobility_llm/prompts/prompts.yaml` | — | — | — |
| Templates and output schemas | `packages/mobility_llm/src/mobility_llm/categories/*/template.md.j2`, `output_schema.json` | `python -m llm_gateway.cli categories` | — | 1 |
| Prompt rendering | `packages/llm_gateway/src/llm_gateway/prompts/engine.py` | — | — | — |
| Answer parsing and draw | `packages/mobility_llm/src/mobility_llm/mode_choice.py`, `services/llm-agents/experiences/decision.py` | — | — | — |
| Decision trace (Figure E.1, tables) | — | read one line of `decisions.jsonl.gz` | Any execution in `archive/c1/` | 1 |

## Appendix F. Press protocol

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Event declarations | `services/llm-agents/config/evenements/a*.yaml` | `make run EVENEMENT=<name>` | Press texts (not redistributed, see [DATA.md](DATA.md)) | 3 |
| Sign grid scoring | `scripts/analysis/presse/grille.py`, `scripts/analysis/presse/scoring.py` | — | `data/presse/grille_signes.yaml` | — |
| Figure F.2 (sign grid) | `scripts/analysis/presse/figure_grille_signes.py` | `python -m scripts.analysis.presse.figure_grille_signes` | `data/presse/grille_signes.yaml`, frozen before any run | 1 |

## Appendix G. Strata and trips

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Table G.1 and Figures G.1–G.4 (modes by distance, purpose, occupation, age) | `scripts/annexes/fig_G_strata.py` | `python scripts/annexes/fig_G_strata.py` | `scores.json` of the executions of Table 1 in `archive/c1/`; rewrites the cache `scripts/annexes/data/fig_G_strata.json` identically | 1 |
| Figure G.5 (confusion counts) | `scripts/annexes/fig_G_confusion.py` | `python scripts/annexes/fig_G_confusion.py [--input <audit json>] [--sortie <png>]` | By default `traces/2026-09-22_lot0_entropie_support_unique/audit_12_decideurs.json` | 1 to redraw, 4 to recompute the audit |
| Tables G.6 onwards (trip-level audit) | `scripts/progedo_logit/build_enquete_population.py`, `scripts/progedo_logit/audit_unitaire_058.py`, `scripts/progedo_logit/arms_058/*.yaml`, `scripts/progedo_logit/mode_choice_eval.py` | as Table 3 | Survey-day sample (restricted) | 1 to read, 4 to recompute |

## Appendix H. Paired differences

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Tables H.1, H.2 (twenty paired differences) | `scripts/analysis/paired_intervals.py` | `python -m scripts.analysis.paired_intervals --preset chapter6 --out <dir>` writes `<dir>/paired_chapter6_B2000.json` | as Table 2 | 1 |
| Third decider family against its mutations and the tabular models | same | `--preset jev_mutations`, `--preset jev32_vs_tabular` | pinned executions in `archive/c1/` | 1 |
| Figure H.1 (forest of paired differences) | `scripts/annexes/fig_H_paired_forest.py` | `python scripts/annexes/fig_H_paired_forest.py --input <dir>/paired_chapter6_B2000.json [--sortie <png>]`; without `--input` it reads the cited trace `traces/2026-09-21_ticket096_lot2/paired_complet_B2000.json` | Output of `paired_intervals --preset chapter6` | 1 |
| Per-stratum panels | `scripts/analysis/plot_decision_makers.py` (panels `ch99_*`) | `python scripts/analysis/plot_decision_makers.py [--sortie <dir>]` | `scores.json` of the pinned executions | 1 |
| Decision-by-decision pairing | `scripts/analysis/appariement_executions.py` | `PYTHONPATH=services/llm-agents python scripts/analysis/appariement_executions.py <execution A> <execution B>`; `make apparier A=… B=…` runs the same in the `controller` container | Two executions | 1 |

## Appendix I. Second cohort and seeds

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Second cohort, sealed and disjoint | `scripts/data/population/selectionner_personas_mesurables.py` | `make personas-verifier POP=data/population/population_1000_PANEL_v6_c2` | `data/population/population_1000_PANEL_v6_c2/` | 1 |
| Table I.1 (the two cohorts and their trip sets) | `scripts/panel/control_population.py`, `services/llm-agents/experiences/jeu.py` | `make control-population POP=data/population/population_1000_PANEL_v6_c2/population.json`; from `services/llm-agents/`: `EXPERIENCES_DIR=../../archive/1_regime_nominal JEUX_DIR=../../data/jeux python -m experiences verifier-jeu --nom population_1000_PANEL_v6_c2_20260316_EN_c` | Both cohorts, both frozen trip sets | 1 |
| Table I.2 (scores on c1 and c2) | `services/llm-agents/experiences/score.py` | `python -m experiences registre` (Section 6 of [EXPERIMENTS.md](EXPERIMENTS.md)) | `archive/c1/`, `archive/c2/` | 1 |
| Tables I.3, I.4 (seed replays, repeated execution) | `scripts/experiment/lancer_rejeux_graines_c1.sh` | `bash scripts/experiment/lancer_rejeux_graines_c1.sh`; campaign `campagnes/rejeu_inter_graines_c1.yaml`. The archived replays (`*_go123_*`, `*_go789_*`) are read with `python -m experiences registre` | Frozen trip set of c1 | 2 (1 to read the archived replays) |

## Appendix J. Cost

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Requests and tokens per execution | `services/llm-agents/experiences/runner.py` | `make experience-estimer EXP=<name>` before a run; `compteurs.json` after | `compteurs.json` of each execution | 1 |
| Tables J.1, J.3, J.4 (tokens per decision, cost at an equal load) | `traces/2026-09-17_11-26_cout_inference_chapitre8/mesure_tokens.py`, `traces/2026-09-21_13-10_cout_jev_vs_gemini/scripts/cout_phases_corrige.py` | Not regenerable here, see below | Gateway exchange logs of the runs | — |
| Table J.2 (tariffs) | — | Read from the providers' price pages on the dates given | — | — |

## Appendix K. Execution language

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Table K.1 (tokens per language) | `scripts/annexes/mesure_K_jetons.py` | `python scripts/annexes/mesure_K_jetons.py` (reads `PROVIDER_KEYS__google` from the environment or `.env`) | Prompts rendered from the frozen trip set; result cached in `scripts/annexes/data/mesure_K_jetons.json` | 2 (a token-count call per prompt, no generation) |

## Appendix L. Memory and a traced event

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Memory mechanics | `services/llm-agents/llm/shortterm.py`, `services/llm-agents/llm/longterm.py`, `services/llm-agents/llm/noyau.py` | — | — | — |
| Figure L.2 (memory strength over time) | `scripts/analysis/plot_memory_strength.py` | `python scripts/analysis/plot_memory_strength.py [--sortie <dir>]` | None: the curves are computed from the memory functions and constants of `services/llm-agents/llm/` and `settings.py` | 1 |
| Traced recall example | `scripts/annexes/exemple_L_rappel.py` | Not regenerable here, see below; the extract is shipped in `scripts/annexes/data/exemple_L_rappel.json` | A traced simulator run | 3 |
| Shock figures | `scripts/analysis/shock_figures.py` | Not regenerable here, see below | Simulator runs | 3 |
| Per-day measurements | `scripts/analysis/mesures/` | `make mesures RUN=<run>` | A simulator run | 3 |
| Memory experiment definitions | `config/experiences_memoire/` | `make experiences-memoire-installer`, then Section 8 of [EXPERIMENTS.md](EXPERIMENTS.md) | Their populations under `data/population/` | 3 |

## Appendix M. Reproducibility

| Paper element | Script | Command | Data | Level |
|---|---|---|---|---|
| Table M.2 (services) | `infra/docker-compose.yml` | `make up`, `make ps` | — | 3 |
| Figure M.1 (container topology) | `scripts/annexes/fig_M_container_topology.py` | `python scripts/annexes/fig_M_container_topology.py` | `infra/docker-compose.yml` | 1 |
| Table M.3 (holds on `/sync`) | `services/llm-agents/backpressure.py`, `services/llm-agents/settings.py` | — | — | — |
| Tables M.4, M.5 (speeds, intersection penalties) | `services/llm-agents/config/osmnx.yaml` | — | — | — |
| Table M.6 (OpenTripPlanner settings) | `services/otp-toulouse/toulouse/router-config.json`, `build-config.json`, `services/llm-agents/trip_helper/otp.py` | `make otp-graph` | `data/gtfs/` | 3 |
| Table M.8 (seals) | — | `shasum -a 256 <file>`; `make personas-verifier POP=data/population/population_1000_PANEL_v6`; the trip set as in Table I.1 (`verifier-jeu --nom population_1000_PANEL_v6_20260316_EN_c`) | Cohort and frozen trip set | 1 |

## Drawn figures of the compiled appendix volume

The LaTeX rendering of the appendix volume carries nine figures that one script draws:
`scripts/generer_figures_annexes.py`, run as `python scripts/generer_figures_annexes.py`
(level 1, a few seconds). **It reads no data**: each figure draws values written in the script.
Running it reproduces the drawing; it does not recompute a measurement. Where the paper measures the
same quantity from data, the row names that script.

| Figure (caption, abridged) | Function of the script | Measured from data by |
|---|---|---|
| Cumulative-distribution reading of EMD and finite-sample bias | `generer_metrics_explanation` | Figure D.1, `scripts/annexes/fig_D_finite_sample_bias.py` |
| Architectural cost comparison and metropolitan cost | `generer_metropolitan_waterfall` | — |
| Byte-pair fragmentation of French transport vocabulary | `generer_bilingual_tokenization` | Table K.1, `scripts/annexes/mesure_K_jetons.py` |
| Stages of the memory pipeline | `generer_memory_pipeline` | — (a diagram) |
| Retrieval scoring components across three profiles | `generer_retrieval_components` | — |
| Retention trajectories of memories | `generer_ebbinghaus_decay` | Figure L.2, `scripts/analysis/plot_memory_strength.py` |
| Normalised confusion matrices of two decision-makers | `generer_confusion_matrices` | Figure G.5, `scripts/annexes/fig_G_confusion.py` |
| Container topology | `generer_c4_topology` | Figure M.1, `scripts/annexes/fig_M_container_topology.py` |
| Backpressure throttle curve and freeze threshold | `generer_backpressure_curve` | — (the rule of `services/llm-agents/backpressure.py`) |

## Not regenerable from this repository

| Paper element | Why | What ships instead |
|---|---|---|
| Figure 5, Table 4, shock figures of Appendix L | They read two simulator runs with memory and a shock (`plot_shock_propensity.py` and `shock_figures.py` name them in their code). A simulator run holds the full exchange log with the models and the agents' memory stores; it is not published | The scripts, and the definitions of the memory experiments (`config/experiences_memoire/`) that replay such runs at level 3 |
| Traced recall example (Appendix L) | Same: it reads one traced run | The extract used by the paper, `scripts/annexes/data/exemple_L_rappel.json` |
| Tables J.1, J.3, J.4 | Tokens per decision come from pairing each execution with the gateway exchange log (`llm_exchanges.jsonl`) over the same time window. The logs carry every prompt and answer of the campaign and are not published | The two scripts that computed the tables, for reading, in the cited traces; `compteurs.json` of each archived execution |
| Figures B.1, B.2 | Drawn by a script kept with the paper sources, outside this repository | The data they draw: `report.json` of the cohort, and the margins of `make control-population` |
