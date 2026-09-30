# Experiments

This page explains how a measurement is defined, run, archived and compared. It covers the
experiment platform (frozen trip sets and deciders), the memory and news-shock experiments
run on the simulator, the per-day measurements, the sealed populations, and the dashboard.

Contents:

1. Two ways to run
2. The experiment platform
3. Frozen trip sets
4. Defining an experiment
5. Running, pausing, resuming
6. The archive, and comparing executions
7. Campaigns
8. Memory experiments and news shocks
9. Per-day measurements
10. Sealed populations
11. Dashboard

Most commands below are `make` targets. The platform targets run
`python -m experiences` inside the `controller` container, with
`EXPERIENCES_DIR=/app/data/experiences` and `JEUX_DIR=/app/data/jeux`; the repository root
is mounted on `/app`, so `data/experiences/` and `data/jeux/` on the host are what the
platform reads and writes.

---

## 1. Two ways to run

| | Without simulator | With simulator |
|---|---|---|
| What moves the agents | nothing: one decision per trip of the frozen set | GAMA, over one or more simulated days |
| Where the options come from | `propositions.jsonl` of a frozen trip set | the routers, live (or a frozen set with `make run JEU=<name>`) |
| Memory, events | off | available (`MEM=`, `EVENEMENT=`) |
| Services needed | `controller` (and `api`, `worker` for an LLM decider) | the whole stack |
| Used in the paper for | the benchmark of deciders (results section, appendices C, D, G, H, I) | the memory and shock illustrations |
| Entry point | `make experience-lancer EXP=<name>` | `make run` / `make experience-memoire-lancer EXP=<name>` |

The benchmark is run without the simulator so that every decider sees exactly the same
options for exactly the same trips.

## 2. The experiment platform

The platform lives in `services/llm-agents/experiences/` and is called as
`python -m experiences <sub-command>`. The sub-commands and their `make` wrappers:

| Sub-command | `make` target | Purpose |
|---|---|---|
| `preparer-jeu` | `make jeu POP=<cohort> NOM=<set> [JOUR=YYYY-MM-DD]` | Freeze the options of every trip of a cohort |
| `consulter-jeu` | `make jeu-consulter NOM=<set> [PERSONNE=<id>]` | Show a set, or the trips of one person |
| `verifier-jeu` | `make jeu-verifier NOM=<set>` | List the dependencies that changed since the set was frozen |
| `verifier-jours` | `make jeu-verifier-jours NOM=<set> JOUR=YYYY-MM-DD [METHODE=gtfs\|moteurs]` | Check whether another day has the same transit offer |
| `definir` | `make experience-definir FICHIER=<experience.yaml>` | Validate an experiment and file it under its canonical name |
| `estimer` | `make experience-estimer EXP=<name>` | Estimate requests and tokens before launching |
| `lancer` | `make experience-lancer EXP=<name>` | Run an experiment |
| `reprendre` | `make experience-reprendre EXP=<name>` | Resume an interrupted execution |
| `pause`, `arreter` | `make experience-pause EXP=<name>`, `make experience-arreter EXP=<name>` | Pause or stop cleanly |
| `file`, `defiler` | `make experience-file`, `make experience-defiler EXP=<name>` | Queue of experiments waiting for a provider key |
| `erreurs` | `make experience-erreurs EXP=<name> [EXEC=<folder>]` | Check that no trip of the set was skipped |
| `registre` | `make registre [TRIER=...] [FILTRER=...] [TOUT=1]` | Table of experiments and executions |
| `statuer` | `make experience-statuer EXP=<name> STATUT=archivee\|invalide\|actif MOTIF="..."` | Hide or restore an experiment without deleting anything |
| `comparer` | `make comparer A=<folder> B=<folder>` | Compare two executions; refuses if they are not comparable |
| `synthese` | — | Rebuild the summary of an execution |
| `score` | — | Rescore an execution from its decisions |

`make apparier A=<execution> B=<execution>` pairs two executions decision by decision (it
runs `scripts/analysis/appariement_executions.py` in the controller, read-only).
`make experience-ordonnancer` runs the host-side scheduler that starts queued experiments
when a key frees up.

## 3. Frozen trip sets

A frozen trip set records, once, the itineraries offered for every trip of a cohort on a
given day. Every decider is then evaluated on those same options, without routers.

```bash
make jeu POP=data/population/population_1000_PANEL_v6 NOM=<set name> JOUR=2026-03-16
```

Preparing a set needs the routers (`controller`, `otp1`–`otp3`, `osmnx1`; `REQUIS=`
overrides the list). The set is written to `data/jeux/<set name>/`:

| File | Content |
|---|---|
| `MANIFEST.yaml` | The day, the cohort and its digest, and the SHA-256 of every dependency: GTFS files, OTP graph, OSMnx graph key, routing configuration, candidate-selection settings |
| `propositions.jsonl` | One line per trip: person, origin, destination, departure, and the offered itineraries with their legs, durations and modes |

The two sets of the paper are shipped: `data/jeux/population_1000_PANEL_v6_20260316_EN_c/`
(cohort c1: 3,299 trips, 3,161 with at least one option) and
`data/jeux/population_1000_PANEL_v6_c2_20260316_EN_c/` (cohort c2).

**Staleness.** `make jeu-verifier NOM=<set>` compares the manifest with the files in place and
names every dependency that moved. An experiment refuses to run on a stale set unless
`ACCEPTER_PERIME=1` is given, and the execution then records it.

**Other days.** `make jeu-verifier-jours NOM=<set> JOUR=<day>` checks whether the transit
trips offered in the set also run on another day (method `gtfs`, the default, reads the feeds
and needs no service; method `moteurs` queries OTP on a sample). `DECLARER=1` writes the
result to `EQUIVALENCES.yaml` next to the set.

## 4. Defining an experiment

An experiment is one YAML file. Its name is computed from its parameters, never typed:
two definitions with the same parameters get the same name. An archived example:
`archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences/exp_gemini-31-fl_proexp05_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c_t0_nosim/experience.yaml`.

```yaml
population:
  chemin: <cohort folder>
jeu:
  nom: population_1000_PANEL_v6_20260316_EN_c     # the frozen trip set
gabarit:
  categorie: itinary_multi_agent                  # LLM category
  variante: prompt_expert_05                      # prompt variant in prompts.yaml
decideur:
  type: passerelle                                # see below
  modele: gemini-3.1-flash-lite
  portee: distant
  parametres: {temperature: 0.0, top_p: 1.0, max_tokens: 4096}
mode: sans_simulateur
calendrier: {politique: aleatoire, date: '2026-03-16', graine: 42}
horizon_jours: 1
memoire: false
evenements: []
graine_ordre: 42                                  # order in which options are shown
graine_tirage: 42                                 # draw of the executed option
regroupement: {parallelisme: 8}
max_candidats: 6
vehicule_chaine: true                             # apply the vehicle chain
verrou_retour: true                               # apply the return lock
```

Decider types (`decideur.type`, `services/llm-agents/experiences/experience.py`):

| Type | Decision |
|---|---|
| `passerelle` | An LLM through the gateway; `modele` names the model, the task is pinned to it |
| `modele` | A tabular reference model fitted on the survey (`services/llm-agents/experiences/decideur_modele.py`) |
| `typesafe` | A typed classifier reached outside the gateway (`services/llm-agents/experiences/decideur_typesafe.py`, key `PROVIDER_KEYS__typesafeAI`) |
| `duree_minimale` | The fastest option |
| `aleatoire` | Uniform draw, with a seed |
| `majoritaire_voiture` | Car whenever a car option exists |
| `rejeu` | Replays the probabilities of an earlier execution (`rejeu_de`) |

<!-- TODO: describe the `typesafe` decider in one line once its public name is settled. -->

`make experience-definir FICHIER=<file>` validates the definition and files it under
`data/experiences/<canonical name>/experience.yaml`. The name states the parameters: a name that
does not is refused, and the expected one is printed.

## 5. Running, pausing, resuming

```bash
make experience-estimer EXP=<name>          # requests and tokens, before paying anything
make experience-lancer  EXP=<name>          # REQUIS="controller api worker" for an LLM decider
make experience-pause   EXP=<name>
make experience-reprendre EXP=<name>        # continues the same execution
make experience-arreter EXP=<name>
```

Options of `experience-lancer`:

| Option | Effect |
|---|---|
| `REQUIS="..."` | Services started and awaited healthy before launching (default `controller`) |
| `REPRENDRE=1` | Resume the last execution instead of starting a new one |
| `ACCEPTER_PERIME=1` | Run on a stale frozen set (recorded in the execution) |
| `ATTENDRE_FENETRE=1` | When the daily quota is exhausted, sleep until the reset (midnight UTC) and continue |

A local decider (`duree_minimale`, `aleatoire`, `majoritaire_voiture`, `modele`, `rejeu`)
needs no provider and runs in minutes. An LLM decider runs under the provider quotas. When
several experiments want the same provider key, the later ones wait in a queue
(`make experience-file`), and the scheduler starts them when the key frees up.

An execution never records a degraded decision in silence: a decision that cannot be
obtained is logged in `erreurs.jsonl`, and `make experience-erreurs EXP=<name>` checks the
decisions against the set so that no trip goes missing.

## 6. The archive, and comparing executions

Each execution writes to `data/experiences/<name>/executions/<YYYY-MM-DD_HH_MM_SS>/`:

| File | Content |
|---|---|
| `execution.yaml` | Effective parameters, code state, set digest, start and end times |
| `decisions.jsonl` | One line per trip: options shown, options removed by the vehicle chain and why, probabilities, drawn option |
| `moves.csv` | The trip log, in the same format as a simulator run |
| `scores.json` | The composite score against the survey and its components, overall and by stratum |
| `compteurs.json` | Counts: decisions, fallbacks, refusals, cache hits, tokens |
| `synthese.json` | Summary used by the registry and the dashboard |
| `progression.json`, `etat.json` | Progress and resumable state |
| `erreurs.jsonl` | Decisions that could not be obtained, with their cause |

The executions used by the paper are shipped under `archive/1_regime_nominal/`, with
`decisions.jsonl` and `moves.csv` compressed (`.gz`); see the [README](../README.md) for
their use without any LLM call.

`make unpack-runs` decompresses them in place for the tools that read plain files.

**Where the tools read.** The platform writes new experiments under `data/experiences/`
(ignored by git). The read-only tools — the registry, the figure scripts, the paired
intervals — look for executions in this order: `EXPERIENCES_DIR` if set; `data/experiences/`
if it holds at least one execution; otherwise `archive/1_regime_nominal/`. In a fresh clone
they therefore read the archive with no setting. The `make` targets that go through the
`controller` container (`make registre`, `make apparier`, `make jeu-verifier`) need the
running stack; their read-only work also runs directly, without Docker:

```bash
cd services/llm-agents
EXPERIENCES_DIR=../../archive/1_regime_nominal JEUX_DIR=../../data/jeux \
  .venv/bin/python -m experiences registre
EXPERIENCES_DIR=../../archive/1_regime_nominal JEUX_DIR=../../data/jeux \
  .venv/bin/python -m experiences verifier-jeu --nom population_1000_PANEL_v6_20260316_EN_c
cd ../..
PYTHONPATH=services/llm-agents EXPERIENCES_DIR=archive/1_regime_nominal \
  services/llm-agents/.venv/bin/python scripts/analysis/appariement_executions.py <execution A> <execution B>
```

**Scoring.** `python -m experiences score <execution folder>` (from `services/llm-agents/`)
recomputes `scores.json` from the decisions. The reference shares of the survey come from
`scripts/data/population/cerema_values.yaml` (variable `REFERENTIEL_ENQUETE` in the
container); the score formula is declared in
`services/llm-agents/experiences/formules/reference.yaml` and implemented in
`services/llm-agents/experiences/score.py`.

**Comparing.** `make comparer A=<execution> B=<execution>` refuses to compare executions that
differ in anything but the decider (another set, another cohort, another seed) and says why.
`make apparier A=... B=...` then measures how far two deciders are apart, decision by
decision. Paired confidence intervals over persons are computed by
`scripts/analysis/paired_intervals.py` (see [PAPER_TO_CODE.md](PAPER_TO_CODE.md)).

## 7. Campaigns

A campaign is a named batch of experiments, declared in `campagnes/<name>.yaml`, and carried
to completion across quota renewals. It runs on the host and drives the scheduler itself.

```bash
make campagne-lancer  NOM=<campaign> [ESTIMER=1] [RECOMMENCER=1] [INTERVALLE=30]
make campagne-etat    NOM=<campaign> [JSON=1]
make campagne-arreter NOM=<campaign>
```

`ESTIMER=1` prints the budget and exits. Never run two campaigns at the same time: they
compete for the same experiments and the same keys.

A campaign recreates the controller between experiments. Do not launch one while a
simulator run is in progress.

## 8. Memory experiments and news shocks

These run on the simulator, over several simulated days, with long-term memory on.

### 8.1 Events

A declared event is one YAML file in `services/llm-agents/config/evenements/`. Two kinds
exist:

| Kind | Files | `canal` | Injected |
|---|---|---|---|
| Local press articles | `a07_greve_eboueurs.yaml`, `a09_vent_autan.yaml`, `a13_punaises_metro.yaml`, `a18_la_machine.yaml`, `a25_velotoulouse.yaml` | `lu` (the agent has read it) | at wake-up, before the first decision of the day |
| Network shocks | `c1_bouchon_rocade.yaml` to `c7_voiture_bruit_suspect.yaml` | `vecu` (the agent has lived it) | on arrival, after the trip |

Each file declares who is exposed (households or persons), on which day, the text given to
the agent, whether the agent judges it on injection, and whether it is relayed to the other
members of the household (`evenement_relais`). Files whose name carries a suffix
(`__<population>`) are variants for a smaller population. The press articles are cited, not
rewritten: each file records the SHA-256 of the article text and refuses to load if it
differs.

The shocks (`c*.yaml`) replay as shipped. The press events (`a*.yaml`) need the article texts,
which are rebuilt locally ([DATA.md § 6](DATA.md#6-press-articles)).

Play an event in a simulator run:

```bash
make run OFFLINE=1 EVENEMENT=c1_bouchon_rocade     # aliases: CHOC=, PRESSE=
make run OFFLINE=1 EVENEMENT=0                     # remove any event
```

The semantic decision cache switches itself off on event days, so that a cached answer
never hides the event. The run records every injection in `evenements.jsonl` and adds event
columns to `moves.csv`.

### 8.2 Treated and control arms

A memory experiment plays two runs in sequence: the treated arm (with the event) and a
matched control arm (same population, same days, no event). It is complete only when both
arms have run.

```bash
make experience-memoire-estimer EXP=<name>
make experience-memoire-lancer  EXP=<name> [DRY_RUN=1] [BRANCHE=both|treated|control]
make experience-memoire-nuit    [EXP="<name> <name>"] [ATTENTE_S=1800] [ESSAIS_MAX=6]
```

The orchestrator is `scripts/experiment/orchestrateur_memoire.py`. Before the event, the two
arms must be identical: the gateway serves the control arm the answers the treated arm
received, as long as the prompt is the same word for word
(`packages/llm_gateway/src/llm_gateway/core/rejeu_ab.py`). A paid call in the control arm
before the event means the arms diverged for another reason; the orchestrator then marks the
experiment as non-conforming instead of complete. `experience-memoire-nuit` chains
experiments in the background and logs to `experiments/enchainement_nuit_<date>.log`.

**Definitions.** The ten memory experiments of this work ship as definitions only, under
`config/experiences_memoire/<family>/<event>/<name>/experience_memoire.yaml`; their runs do
not ship. The platform reads and writes memory experiments under
`data/experiences/evenements_non_tabules/` (`EXPERIENCES_MEMOIRE_DIR`), where each run adds
its state. Copy the definitions there once, before the first memory experiment:

```bash
make experiences-memoire-installer
```

The target never overwrites a definition already in place. The populations these
definitions name ship under `data/population/`: `population_1_861500` and
`population_1_899549` (one persona each, identical to their record in cohort c1),
`population_4_foyer_133048` and `population_6_foyers_a13`.

### 8.3 Reading the results

| Script | Output |
|---|---|
| Command | Output |
|---|---|
| `python scripts/analysis/figure_evenement.py <run> --mode <mode> [-o <svg>]` | Mode shares of exposed and unexposed agents, day by day, around the event; `<mode>` is the mode the event targets |
| `python scripts/analysis/tableau_quatre_voies.py <run> [--markdown]` | The four-way table: exposed or not, treated arm or control |
| `python scripts/analysis/lecture_avant_decision.py <run>` | Whether the memory of the event was read before each decision; exits 1 if one was not |
| `python scripts/analysis/shock_figures.py` | Figures of the shock runs. No option: it reads two fixed run folders of `experiments/archive/`, which do not ship |
| `make memoire-rapport RUN=<run folder>` | Per-persona memory report (HTML): mode timelines, itineraries offered and chosen, concepts, habits |

`<run>` is a simulator run folder (`experiments/archive/<date>`). Figures go to `-o`, or by
default to `outputs/figures/`.

## 9. Per-day measurements

A simulator run writes one set of CSV files per simulated day under `<run>/mesures/`. The
controller rewrites them every simulated day at 03:00; they can be recomputed at any time:

```bash
make mesures RUN=experiments/archive/<run> [OUT=<folder>]
```

| File | Content |
|---|---|
| `choix_modal_par_jour.csv` | Mode shares per day, and the share of trips actually decided (not served from the cache or a fallback) |
| `habitudes_par_activite.csv` | Habit and break per activity: same mode as the previous day or not |
| `memoire_par_jour.csv` | Memory pool and concept operations per day |
| `duree_de_vie_par_type.csv` | Lifetime of memories by type |
| `evenement_par_jour.csv` | Exposure to the declared event per day |
| `souvenir_evenement_par_jour.csv` | Whether the event reached long-term memory, per day |
| `souvenirs_derives.csv` | Memories derived from the event |

The calculation is idempotent on flows (everything is recomputed from the deduplicated
`moves.csv`) and keeps the states already written, which cannot be rebuilt afterwards.

**Continuity check.** After a stop and resume (`CONT=1`), check that the cut neither
duplicated nor dropped a day and did not change a past value:

```bash
cp -r experiments/current/mesures /tmp/mesures_before      # before stopping
make stop-run
make run OFFLINE=1 CONT=1
# ... once the run has passed the cut:
make mesures-continuite AVANT=/tmp/mesures_before APRES=experiments/current/mesures
```

It exits non-zero if a key is duplicated, a weekday is missing, or a value of a day before
the cut changed.

Columns, as written by `scripts/analysis/mesures/ecriture.py`. Every table starts with
`jour_simule`, `date_simulee` and `person_id`; the key is given in brackets.

| File | Other columns |
|---|---|
| `choix_modal_par_jour.csv` [day, person] | `trajets`, `trajets_decides`, `part_decidee`, `modes_distincts`, `part_car`, `part_public_transport`, `part_walking`, `part_cycling`, `part_train`, `part_motorbike`, `part_autres`, `trajets_mode_inconnu` |
| `habitudes_par_activite.csv` [day, person, activity] | `activite`, `motif`, `mode`, `occurrences`, `mode_veille`, `reprise_veille`, `mode_habituel`, `conforme_habitude`, `observations_fenetre` |
| `memoire_par_jour.csv` [day, person] | `rappels`, `vivier_min`, `vivier_median`, `vivier_max`, `souvenirs_servis`, `concepts_crees`, `concepts_confirmes`, `concepts_precises`, `concepts_contredits`, `operations_hors_vocabulaire`, `etat_lu_dans`\*, `entrees_ltm`\* |
| `duree_de_vie_par_type.csv` [day, person, memory type] | `type_souvenir`, `entrees`\*, `duree_vie_mediane_jours`\*, `etat_lu_dans`\* |
| `evenement_par_jour.csv` [day, person, shock] | `choc_id`, `canal`, `expositions`, `minutes_injectees`, `incidents_reseau`, `correspondances_ratees`, `souvenir_choc_servi`, `decisions_avec_souvenir_choc`, `appariement` |
| `souvenir_evenement_par_jour.csv` [day, person, event] | `evenement_id`, `role`, `informe`, `jours_depuis_j0`, `decisions`, `prompts`, `decisions_sans_prompt`, `prompts_avec_souvenir`, `souvenir_texte`, `souvenir_mots`, `via_connaissances`, `via_changements`, `via_rappel` |
| `souvenirs_derives.csv` (no day column) | `evenement_id`, `person_id`, `role`, `doc_id`, `type_souvenir`, `ecrit_le`, `appariement`, `mots_retrouves`, `enonce` |

\* State columns: read from the memory store on the day itself, never rewritten afterwards.

## 10. Sealed populations

A sealed cohort is a population of personas controlled against the survey margins, then
frozen with a manifest.

| Step | Command |
|---|---|
| Reference margins, with their source | `make reference-marges` (`RECOMPUTE=1` recomputes the joint target from the survey microdata) |
| Stratified selection of N personas from a pool | `make select-population POOL=<pool.json> N=1000` |
| Control against the thirteen margins | `make control-population POP=<population.json> [BORNE=1.0]` |
| Seal: control, then copy into an immutable folder | `make seal-population POP=<population.json> SELECTION=<selection.json> [OUT_DIR=...]` |
| Check a sealed cohort | `make personas-verifier POP=data/population/population_1000_PANEL_v6` |

The control reports age classes, occupation, car ownership (per person and per household),
residential ring, and ring × car ownership, with 95 % intervals, equivalence tests at
± `BORNE` points (1 by default), χ² with Cramér's V, and distances between distributions. The
seal refuses a population with a margin to correct. The sealed folder holds
`population.json`, `MANIFEST.yaml` (with `population.sha256`), `CONTROLE.md`, `report.json`
and `selection.json`. The scripts are in `scripts/panel/`.

`make reference-marges`, `make control-population` and `make personas-verifier` run on the
frozen reference margins alone: they were run on 2026-09-28 in a fresh clone, without the
survey microdata, and reproduce the thirteen margins of the paper (Appendix B). Only
`RECOMPUTE=1` reads the microdata. `make control-population` also writes a copy of its
report under `docs/traces/<date>_controle_population/` (`--trace-auto`).

**Measurable personas.** For memory experiments, `make personas-mesurables RUN=<execution>
[N=10]` selects personas whose decisions are observable (at least four trips, always more
than one itinerary, at least two modes chosen) from a run already played, and writes a small
population with its own manifest. `make personas-verifier POP=<folder>` checks it the same
way.

## 11. Dashboard

```bash
make dashboard [DASHBOARD_PORT=8503] [DASHBOARD_THEME=dark]
```

A Streamlit application (`scripts/dashboard/app.py`) on port 8503: service status,
experiments in progress and queued, campaigns, memory experiments (a form that writes the
treated and control definitions), live run metrics, and the LM Studio models loaded. It
reads the same folders as the platform and drives it through the same `make` targets. Its
labels are in French.

For a simulator run, Grafana (port 3000, localhost only) shows the pipeline metrics
exported by the controller and the gateway to Prometheus, and the population map is served
by the controller on port 5050.
