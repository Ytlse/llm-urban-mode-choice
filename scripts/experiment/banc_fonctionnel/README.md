# Functional test bench — ticket 100

Full specification: [`specs/ticket_100/tests_fonctionnels.md`](../../../specs/ticket_100/tests_fonctionnels.md).

**Goal**: find defects **before** a forty-day campaign consumes the quota.
As many checks as possible for as few calls as possible.

## Running

```bash
python -m scripts.experiment.banc_fonctionnel.famille_a
```

**Zero calls, a few seconds.** Builds a complete run — population, decisions, memory,
event, checkpoints — without a simulator or a model, then runs the real analysis chain
on it. **If it fails, do not launch family B**: it would pay for nothing.

```bash
python -m scripts.experiment.banc_fonctionnel.famille_b --test B2
```

**27 calls in total**, in series, on the declared instances. The default order is
**B2 → B3 → B1**, and it is not arbitrary: B2 is the test whose failure is *silent* in
production.

| | What the model is asked | Calls |
|---|---|---|
| **B2** | does it say `source: heard` when the household talked to it? | 4 |
| **B3** | does the same judgement, repeated, return the same level? | 8 |
| **B1** | is the five-level scale used? | 15 |

## The gateway

**Today: Groq only** (author's decision, 2026-09-22). Long-running tests will move
to other models — hence a **parameter**, not a constant:

```bash
python -m scripts.experiment.banc_fonctionnel.famille_b --instances google_gemini31_key1
```

⚠ **No fallback.** If the quota is exhausted, the bench **waits** or **stops**; it switches to
no other instance. An undeclared instance that would have served makes the call fail and
names the problem: a measurement obtained on a gateway that was not declared is not the
measurement one thinks one is reading.

⚠ **The limit that bites is not the RPM.** Groq was dropped from campaigns on 2026-09-08 for a
measured limit of **1,000 output tokens per minute**, invisible outside the body of the 429s, and
the gateway was not fixed. The 30 req/min and 1,000 req/day will never be reached here;
the OTPM will. Hence the calls in series, a tight `max_tokens`, and an output-token budget
that the bench watches itself (`--budget`, default 6,000) rather than discovering the limit in
a silent 429.

## What the bench writes

In `docs/traces/banc_fonctionnel_100/`, rewritten **after each call** — a quota
interruption must not lose anything that was paid for.

| File | What it is for |
|---|---|
| `jugements.csv` | **Paper figure**: what the agent understood of each text, set against the grid — without waiting for a trip |
| `plancher_jugement.csv` / `.md` | The noise floor of the **judgement**, to be declared with any severity result — as the 3.2 % of 095 are declared with any modal share gap |
| `provenance.csv` | Says whether the diffusion stage is observable **before** paying for a campaign |
| `passerelle.csv` | Serving instance, output tokens, off-schema, per call |

## What the bench does not test

- **The GAMA path.** The short non-regression run on `c6` remains the only test of this
  boundary, and no stub replaces it.
- **A behavioural effect.** No agent moves here.
- **Duration.** One can write an entry thirty days old; not the thirty days of
  recalls that wore it down.

A green bench does not prove an effect. A red bench avoids paying forty days to learn it.

## The judgement safeguard (ticket 095)

```bash
python -m scripts.experiment.banc_fonctionnel.garde_fou_jugement
```

**32 calls, about twelve minutes.** Since the decision of 2026-09-22, the severity of a
memory is the one the agent estimates, and only that one: it sets the lifetime of the
memory. This script checks, **before** a fifty-day campaign starts, that the
agent's judgement still falls within the range we had in mind when analysing the texts.

It reads `grille_attendus.yaml`, next to it, where each text declares a **range** of levels
(not a value: two sensible people do not judge an article about bedbugs identically),
the modes it should bear on, and the reason for the range.

It refuses the campaign in four cases, and each refusal says what to do:

| Refusal | Why it is blocking |
|---|---|
| a call without an answer | an unjudged exposure is not harmless: it did not take place |
| fewer than 3 distinct levels | all lifetimes would be close and the experiment would measure nothing |
| the three incidents out of the predicted order | it is the very hypothesis of the experiment |
| too many **blind** judgements out of range | the range was written without having seen an answer |

⚠ **The `deja_vu` field separates two populations that never mix.** A range written
after seeing answers proves almost nothing: its out-of-range rate is **reported** and never
blocks. Only blind predictions can refuse a campaign. Today, two
of the eight texts are blind — `c2_crevaison` and `c3_panne_reseau`, two of the three arms.

The verdict logic is exercised **without a single call** by
`scripts/tests/test_judgement_safeguard.py`: a safeguard that can only be checked by
running it costs thirty calls per check, and one stops checking it.
