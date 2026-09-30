# Declared shocks (ticket 079)

> ⚠ **This directory has a successor: [`../evenements/`](../evenements/README.md)** (ticket 100,
> lot 1, 2026-09-21). The declarations that are here still load — they are
> converted on the fly, with a warning that names the file — but they have all been
> migrated there. What follows remains accurate; what is added to it (article read, injection at wake-up, the
> agent's judgement, circulation within the household) is read in the neighbouring README.

A shock is **a quantified delay** plus **a lived sentence**, placed on designated agents on
designated days. It cuts no line and degrades no service: the agent decides seeing
the nominal service, then takes the hit. This is the **suffered regime**.

## Launching a run with a shock

```bash
make run OFFLINE=1 CACHE=0 CHOC=c3_panne_reseau
```

The cache is deliberately turned off: its key carries no duration, a decision taken before the shock
could be served again during it. An `[ALARME]` is raised if you forget it.

## Why the texts are in English

The whole set-up switched to English (ticket 074): prompts, templates, rendering layer. A
French sentence in an English prompt would reintroduce exactly the factor the switch
removed. The format imposes no language; these examples do.

## What a `vecu` can say, and what it cannot

It **tells** what the agent lived through. It **dictates** nothing to them.

| Accepted | Refused at loading |
|---|---|
| "I was stuck for an hour on the ring road" | "avoid the ring road tomorrow" |
| "Flat tyre, hands covered in grease" | "you should take the metro instead" |

The refusal is outright, not a warning: a warning in the middle of a run log alerts
nobody, and an instruction that gets through does not bias a little — it fabricates exactly the result
one claims to measure.

Since 2026-09-19, it does not **conclude** either. A verdict on a mode ("I no longer trust
this car", "this bus line is unreliable") and an intention for tomorrow ("I am thinking about
not using this car anymore", "from now on I will take the metro") are refused on the same grounds.
The belief is what the agent's reflection must produce; writing it here amounts to measuring one's
own instruction. A doubt remains accepted — "I am starting to wonder whether this is worth it" — and
so does a past fact, even modal: "I had to sort out another way of getting around".

## How many times per day

`cadence: trajet` (default) — every eligible trip suffers the shock: a traffic jam lasts the whole
day. `cadence: jour` — only the first one: a repaired breakdown does not happen again identically
three hours later. The default is logged at loading, an unknown value is refused.

## The values are a declared scenario

Delays and durations are not measured on a source: they are deliberate assumptions, with the
reference alongside when it exists. They can be discussed; they do not present themselves as
facts.

## Known limitation

`c2_crevaison` declares on the following day that the bike is immobilised, but **the bike stays
available** in the eligibility filter: only the memory carries the unavailability. Making a
mode really unavailable touches the decision shared with the platform, and falls outside this lot.
