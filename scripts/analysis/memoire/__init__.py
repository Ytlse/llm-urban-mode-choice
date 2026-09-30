"""Memory analysis report, per persona (ticket 077, lot F).

A generator, not a hand-written report: the next run will raise the same
questions. Standard library only, inline SVG hand-written after the
model of ``scripts/synthesis/charts.py`` — the report must open from a
``file://``, with no network and no installed dependency.

Four modules:

* :mod:`sources` — raw reading of the run (trips outside replay, arrivals, memory
  events, LLM exchanges, LTM metadata, logs, population);
* :mod:`mesures` — redundancy, contamination, decisiveness and entropy, recalls;
* :mod:`graphiques` — mode timeline, itinerary table, curves, bars;
* :mod:`rapport` — self-contained HTML assembly and CLI.

Two runs on the same run produce the same bytes: no generation timestamp
enters the body, no set traversal is left unsorted.
"""

__all__ = ["graphiques", "mesures", "rapport", "sources"]
