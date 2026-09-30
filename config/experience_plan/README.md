# Experimental Plan & Unified Experiment Specification

This folder contains the complete, standardised and reproducible specification of all the experiments to be run for the research article:
> **« Limites et perspectives des agents LLM en simulation de mobilité urbaine »** (“Limits and prospects of LLM agents in urban mobility simulation”).

The central file is **[`experiments.yaml`](experiments.yaml)**. It serves as the **single source of truth** for the entire test campaign.


---

## 1. Purpose and Philosophy of the Unified Specification

In experimental research that confronts generative AI models (LLMs), supervised tabular models (LightGBM, Logit) and physical heuristics, the major risk is methodological dispersion: prompts modified on the fly, varying temperatures, heterogeneous population versions, uncontrolled costs.

The `experiments.yaml` file formalises a strict contract:
1. **Sealed common substrate (*Common Set*)**: All benchmark experiments share exactly the same input set ($1{,}000$ people aligned with the EMC² 2023 survey, with their precomputed OTP alternatives).
2. **Strict informational parity**: No model receives privileged information.
3. **Full reproducibility**: Each experiment is self-contained (model, temperature, prompt template, seed, micro-batching parameters).
4. **Cost sizing before execution (*Dry-run*)**: Predictive computation of the number of API requests and token volume to avoid any quota overrun or budget surprise.

---

## 2. The 4 Uses of the YAML File

The `experiments.yaml` file is designed to be consumed by 4 distinct tools of the project:

```
                        ┌─────────────────────────────────┐
                        │        experiments.yaml         │
                        │     (Single source of truth)    │
                        └────────────────┬────────────────┘
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼                           ▼
    1. EXECUTION & RUN          2. SCORING ENGINE           3. HTML DASHBOARD           4. ARTICLE      
   Drives inference             Computes EMD, JSD, L1,      Generates / updates         Generates the LaTeX
   (API micro-batching,         unit Accuracy against       docs/synthesis/index.html   tables and fills the
   concurrency, rate-limits)    EMC² and LightGBM           (gap radar, badges)         manuscript TODOs
```

### 1. Orchestration & Execution
The launcher reads the configuration of the selected experiment, loads the sealed population, groups requests by `batch_size`, calls the provider (Google, Mistral, local vLLM or local Heuristic) and records the raw responses in `experiments/archive/<id>/`.

### 2. Scoring and Evaluation Engine
At the end of the run, the computation engine reuses the `scoring:` keys to produce:
* **Global modal shares**: Compared with the Cerema EMC² 2023 targets (Car, PT, Walking, Bike).
* **Macroscopic gaps by stratum**: EMD (Earth Mover's Distance) for ordinal variables (age, distance) and JSD (Jensen-Shannon) for nominal variables (purpose, gender, ring).
* **Unit metrics**: Accuracy and Log-Loss against the actually observed choice and the LightGBM oracle.

### 3. Visual Reporting & Dashboard
The synthesis web page (`docs/synthesis/index.html`) is regenerated automatically:
* Addition of a column dedicated to the experiment.
* Display of badges (Model, Temperature, Date, Batching).
* Graphical visualisation of modal over/under-estimations (over-attractiveness of the bike, underestimation of walking).
* Operational monitoring box (successful requests, tokens consumed, actual cost).

### 4. Direct Injection into the Manuscript
An export script extracts the metrics to automatically fill the comparative tables of the LaTeX/Markdown paper (Sections 4 and 5), replacing the `⟨xx⟩ %` markers without any error-prone manual entry.

---

## 3. Anatomy of the `experiments.yaml` File

The file is divided into two main sections:

### Section `defaults:` (Common Base)
Defines everything shared by all experiments to avoid any unnecessary duplication:
* `inputs`: Paths to the sealed population (`population.json`), Cerema targets (`cerema_values.yaml`), LightGBM policies and OTP cache.
* `scoring`: List of metrics (EMD, JSD, L1, Accuracy) and of the sociological evaluation strata.
* `outputs`: Root folder for result archiving and path of the synthesis dashboard.

### Section `experiments:` (Test Matrix)
Each experiment is a list entry with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | String | Unique technical identifier (e.g. `exp_02_bare_gemini_flash_lite`) |
| `title` | String | Readable title displayed in the dashboard and the paper |
| `phase` | String | Methodological section (`ablation_baseline`, `calibrated`, `hysteresis_5d`, `news_events`) |
| `description` | String | Context and scientific hypothesis tested |
| `engine` | Object | Decision engine: `type` (`heuristic`, `llm`, `tabular_ml`), `model`, `provider`, `temperature`, `seed`, `prompt_template` |
| `execution` | Object | Execution parameters: `batch_size` (micro-batching), `max_parallel_requests` |
| `estimation` | Object | Token-per-trip assumptions and pricing for the predictive cost computation |

---

## 4. Operational Sizing & Compliance with API Quotas

The evaluation does not rely on theoretical dollar billing, but on compliance with the **actual throughput quotas and daily caps of the SWRR pool** configured in [`config/llm_gateway/providers.yaml`](../llm_gateway/providers.yaml) and summarised in `methode/quotas_summary.html` of the papers repository (`llm-agents-gama-papiers`).

### 4.1 Quotas of the key providers in rotation (Source: `quotas_summary.html`)

| Instance | Actual model | RPM | TPM | RPD (Req/day) | TPD (Tokens/day) | Role in the project |
|---|---|:---:|:---:|:---:|:---:|---|
| **`google_gemini31_key1`** | `gemini-3.1-flash-lite` | **15** | 250,000 | **500** | — | **Reference judge** (`thinking: 0`, strict cache) |
| **`google_gemini35_key1`** | `gemini-3.5-flash-lite` | **15** | 250,000 | **500** | — | Mutator / Variants (+ thinking 1024) |
| **`google_gemini31_key2`** | `gemini-3.1-flash-lite` | **15** | 250,000 | **500** | — | Second Google key (off-campaign measurements) |
| **`mistral`** | `mistral-small-latest` | **60** | 500,000 | — | 100 M (safeguard) | Simulation pillar (1 req/s) |
| **`vllm_local`** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | $\infty$ | $\infty$ | $\infty$ | $\infty$ | Deterministic local GPU inference |

---

### 4.2 The gateway sizing ratios

1. **Per-task allowance (`tokens_per_agent`)**:
   $$\text{tokens\_per\_agent} = 3{,}000\text{ tokens (2,200 in + 800 out)}$$
   Set from the ~1,600 tokens/agent measured in simulation + a safety margin to guarantee the absence of HTTP 413 truncation.
2. **Batch sizing (`batch_target_agents`)**:
   * Gateway grouping target: **$B = 10\text{ agents per batch}$** (default `Settings.batch_target_agents`).
   * Average observed in simulation: **~7 to 10 agents per batch**.
3. **Critical RPD threshold (Why batching is mandatory)**:
   * **Without batching** ($1{,}000$ requests for $1{,}000$ trips): the run immediately saturates the 500 RPD quota of `google_gemini31_key1` at 50% of the dataset and fails.
   * **With micro-batching ($B = 10$)**: $1{,}000$ trips $\to$ **$\approx 100\text{ requests}$**, i.e. only **20% of the daily quota (100 / 500 RPD)**.

---

### 4.3 Operational impact matrix (Sealed cohort of 1,000 people ≈ 2,579 trips, nominal Batch B = 10)

> **Unit of evaluation = the trip.** The sealed cohort is **1,000 people** (894 mobile);
> their activity chains produce **≈ 2,579 trips** (2.58/person), the unit on which modal shares
> and unit metrics are computed. The request estimates below assume
> a micro-batch B=10 on this number of trips.

| Experiment | Gateway model | Req. without batch | Req. micro-batch ($B=10$) | Daily Quota Consumption (RPD) | Rate-Limit Risk (RPM/TPM) |
|---|---|:---:|:---:|:---:|---|
| **`exp_00b` Car prior** | Local heuristic | 0 | 0 | **0%** (0 req) | None (local) |
| **`exp_01a` Gemini Flash-Lite** | `gemini-3.1-flash-lite` | 1,000 | **100** | **20%** (100 / 500 RPD) | None (15 RPM handled by SWRR) |
| **`exp_01b` Mistral Small** | `mistral-small-latest` | 1,000 | **100** | **< 1%** (of 100M TPD) | None (60 RPM, ample) |
| **`exp_01c` Qwen-32B Local** | `Qwen2.5-32B-Instruct` | 1,000 | **100** | **0%** (local GPU) | None (local GPU) |
| **`exp_02a` Calibrated Gemini** | `gemini-3.1-flash-lite` | 1,000 | **100** | **20%** (100 / 500 RPD) | None (optimised cache) |
| **`exp_02b` Gemini Few-Shot** | `gemini-3.1-flash-lite` | 1,000 | **100** | **20%** (100 / 500 RPD) | None |
| **`exp_03b` LightGBM Oracle** | Supervised tabular | 0 | 0 | **0%** (0 req) | None (local) |
| **`exp_04a` Hysteresis (5 days)**| `gemini-3.1-flash-lite` | 5,000 | **500** | **100%** (500 / 500 RPD) | Requires 2 Google keys (`google_gemini31_key1` + `google_gemini31_key2`) |
| **`exp_05a..c` Local Press** | `gemini-3.1-flash-lite` | 1,000 | **100** | **20%** (100 / 500 RPD) | None (per condition; 4 LLM conditions/event) |

---

## 5. Summary of the Defined Experiments

The `experiments.yaml` file covers the entire research plan. The
identifiers below are those of the **paper** (they carry the phase and the order of the sections);
on disk, each experiment lives under the name that the platform **computes** from its parameters,
given by the `nom_runtime` key of `experiments.yaml` (for example `exp_00a_random_otp` →
`exp_alea_nosim`, `exp_01a_bare_gemini_flash_lite` → `exp_gemini-31-fl_minper_jtir_t0_nosim`).
That is the name taken by `make experience-lancer EXP=…` and `data/experiences/<nom>/`
(spec `specs/nommage-canonique-experiences.md`).

1. **Phase 0: Heuristic & Physical Floors**
   * `exp_00a_random_otp`: Uniform random choice among the OTP route options.
   * `exp_00b_majority_car`: Empirical prior (majority mode: Car 100%).
   * `exp_00c_shortest_time`: Fastest-trip heuristic (min OTP duration).
2. **Phase 1: Neutral, factual and detailed prompt (Zero Prompt Engineering, $T=0.0$)**
   * `exp_01a_bare_gemini_flash_lite`: Google Gemini 3.1 Flash-Lite (low-cost remote reference).
   * `exp_01b_bare_mistral_small`: Mistral Small (sovereign European model).
   * `exp_01c_bare_qwen_32b_local`: Qwen-2.5-32B Instruct (deterministic local open-weight vLLM model).
3. **Phase 2: Semantic Optimisation & Calibrated Prompt**
   * `exp_02a_calibrated_prompt`: LLM with optimised prompt from the genetic calibration.
   * `exp_02b_few_shot_emc2`: LLM with injection of $k$ real examples from the EMC² survey.
4. **Phase 3: Supervised Tabular Ceiling (Reference baselines)**
   * `exp_03a_multinomial_logit`: Multinomial Logit (McFadden MNL).
   * `exp_03b_oracle_lightgbm`: Supervised LightGBM oracle sealed on ProGEDO / EMC² 2023.
5. **Phase 4: LLM Added Value — Post-Incident Hysteresis (5-Day Kinetics)**
   * `exp_04a_hysteresis_memory_5d`: Metro breakdown on D2 + restoration D3-D5 with short-term memory $\mathcal{M}_t$ ($\lambda = 0.4$).
   * `exp_04b_hysteresis_amnesic_5d`: Control condition: without memory register.
   * `exp_04c_hysteresis_oracle_lightgbm`: Behaviour of the tabular oracle (structural amnesia).
6. **Phase 5: LLM Added Value — Ecological Evaluation on Local Press**
   * Event 1: La Machine / Minotaure festival (pedestrianised city centre) — **5 conditions** (Nominal, Raw, Paraphrase without modal cue, Placebo, Encoded oracle).
   * Event 2: Heatwave & Ozone Peak (Crit'Air / reduced PT) — **5 conditions** (“worst case” of instruction following: the raw article names the modal response).
   * Event 3: Rocade Empalot infrastructure closure (+1h of traffic jam) — **5 conditions**.

> **Note.** Uncertainty quantification is no longer optional: each metric is published with a
> 95% CI by **cluster bootstrap by agent**; the paired contrasts of Phases 4 and 5 use **McNemar**;
> the macro calibration vs EMC² uses an **equivalence test (TOST, ±1 pt)**. See `defaults.scoring.uncertainty`.
