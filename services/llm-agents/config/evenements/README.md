# Declared events — a single channel, two entry points

Ticket [100](../../../docs/tickets/ticket_100_un_seul_canal_d_evenement_pour_le_vecu_et_le_lu.md).
An event is **a text** placed in the memory of designated agents on designated days,
with **at most one measured fact**.

This directory replaces `config/chocs/`, which is still read for one more version with a warning.

## The two entry points

| `moment` | When | What the agent knows when deciding | Case |
|---|---|---|---|
| `arrivee` | after the decision | the nominal service, nothing else | the shock of ticket 079 |
| `reveil` | before the first decision | what they read this morning | the article of ticket 059 — **lot 2** |

The difference is not an implementation detail. On arrival, **the day of the event
measures no choice**: the whole effect of the following days is attributable to the memory, and to nothing
else. At wake-up, the agent decides knowing — and that is the contrast chapter 7 measures.

## Launching

```bash
make run CHOC=c6_voiture_suspecte CACHE=0
```

⚠ **Turn off the cache.** Its exact key does not carry the agent's memory, and its semantic
branch only discards a decision below 0.95 similarity: one memory line
added to a block is not always enough. An `[ALARME]` is raised if you forget it.

## What the declaration refuses

Common to both channels, taken over from 079: instruction ("avoid", "you should"), verdict on a
mode ("this line is unreliable"), intention ("from now on I will…"), address in the second
person, day outside the run, unknown exposure rule, mode outside the hierarchy, two entries for the
same day, negative delay, **a `gravite` field set by hand**.

A text that concludes in the agent's place fabricates the result one claims to measure: the
belief is what the reflection must produce. The run of 19 September paid for it — "I no longer
trust this car at all" in the lived text, and that very evening a concept that was merely its
rewording.

## Delivery status

| Field | Value | Delivered |
|---|---|---|
| `canal` | `vecu` | lot 1 |
| `canal` | `lu` | **lot 2** |
| `moment` | `arrivee` | lot 1 |
| `moment` | `reveil` | **lot 2** |
| `exposition.regle` | `mode`, `tirage`, `agents` | lot 1 |
| `exposition.regle` | `foyers` | **lot 2** |
| `exposition.lecteurs` (designated readers, `foyers` rule) | — | 2026-09-25 |
| `texte` (cited file + fingerprint) | — | **lot 2** |
| `calendrier` (window drawn per household) | — | **lot 2** |
| `jugement` | `aucun` | lot 1 |
| `jugement` | `a_l_injection` | **lot 3** |

What is not delivered is **refused at loading**, with a message that names the lot. A field
accepted and without effect is worse than a refused field: nothing reports it, and the run starts on a
protocol that is not the one that was written.

## Why `jugement: aucun` on the eight migrated cases

It is the state from before ticket 100: the severity of a shock is computed on what the simulation
measured, without the agent saying anything about it. Decision D4 wants the agent to judge in both
regimes; that will be lot 3. `jugement: aucun` will remain there as a **declared ablation** — the arm
that measures what the judgement adds — and it is also the mode under which the migration is
checked.
