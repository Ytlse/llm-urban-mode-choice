#!/usr/bin/env python3
"""Early stop of a run once the shock has died out — ticket 095.

WHY
---
The return to the previous behaviour is reached a few days after the last shock memory
has left the "what changed recently" block. The following days are paid for and
teach nothing. On the 2026-09-21 campaign, they make up about 12 % of the run.

WHAT TRIGGERS
-------------
The line the controller writes at the exact moment of extinction, and nothing else:

    [noyau] <agent> : le souvenir de choc du <date> est sorti du bloc « ce qui a changé
    récemment » (…) — plus aucun souvenir de choc ne pèse sur ses décisions.

⚠ The tail "plus aucun souvenir de choc" is MANDATORY. A memory that leaves while
others are still served ends nothing: the agent still carries a shock in its context.

⚠ NEVER trigger on the stability of a modal share. On that same campaign, the
daily propensity is 5 % on 11 April and 52 % on the 13th: a narrow band would stop the run
at the whim of noise. Extinction is a dated event, not a statistic.

WHAT IS COUNTED NEXT
--------------------
LIVED days, not calendar days: the simulation skips weekends, and counting in
calendar days would remove two days of observation out of seven.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# The tail that tells "one memory has left" from "none is left".
EXTINCTION = re.compile(
    r"est sorti du bloc .*plus aucun souvenir de choc ne pèse sur ses décisions"
)
# `sim_time=27 April 2026, 05:00` — the simulated date, to count lived days.
SIM_TIME = re.compile(r"sim_time=(\d{2}) (\w+) (\d{4})")
MOIS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


def date_simulee(ligne: str):
    """The simulated date carried by a line, or None."""
    m = SIM_TIME.search(ligne)
    if not m:
        return None
    import datetime
    try:
        return datetime.date(int(m.group(3)), MOIS[m.group(2)], int(m.group(1)))
    except (KeyError, ValueError):
        return None


def analyser(lignes, souvenir_du: str | None = None) -> tuple[bool, int]:
    """(extinction seen, LIVED days elapsed since). Pure: this is what the tests exercise.

    ⚠ `souvenir_du` changes what counts as extinction, and the choice is not neutral.

    Without it, the trigger is "plus aucun souvenir de choc ne pèse sur ses décisions".
    Measured on the 2026-09-21 campaign: this line only comes five lived days before the
    end of the run, because the agent MAKES its own shock-severity memories — the control
    arm, which undergoes nothing, produces three. The saving is then zero.

    With it, the trigger is the exit of the DECLARED memory, at its date. That is the event
    the protocol studies, and it falls thirteen days before the end of the same run.
    """
    vue = False
    jours: list = []
    for ligne in lignes:
        if not vue:
            if souvenir_du:
                touche = f"souvenir de choc du {souvenir_du}" in ligne and "est sorti du bloc" in ligne
            else:
                touche = bool(EXTINCTION.search(ligne))
            if not touche:
                continue
            vue = True
            jours = []
        if vue:
            d = date_simulee(ligne)
            if d is not None and d not in jours:
                jours.append(d)
    # The extinction day counts as zero: we want N days AFTER it.
    return vue, max(0, len(jours) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", default="experiments/current/app.log",
                    help="app.log of the run to watch")
    ap.add_argument("--jours", type=int, default=7,
                    help="LIVED days to observe after extinction (default: 7)")
    ap.add_argument("--arreter", action="store_true",
                    help="run `make stop-run` once the delay has elapsed. Without this flag, "
                         "the script only reports and returns.")
    ap.add_argument("--souvenir-du", default=None, metavar="AAAA-MM-JJ",
                    help="SIMULATED date of the declared memory to watch. Recommended: without it, "
                         "the trigger waits until NO shock memory weighs any more, which "
                         "only happens at the very end of the run since the agent makes its own.")
    ap.add_argument("--intervalle", type=float, default=30.0)
    ap.add_argument("--limite-h", type=float, default=6.0,
                    help="give up after this many hours, so as never to run forever")
    a = ap.parse_args()

    journal = Path(a.journal)
    depart = time.monotonic()
    annonce = False
    while time.monotonic() - depart < a.limite_h * 3600:
        if journal.is_file():
            with open(journal, encoding="utf-8", errors="replace") as f:
                vue, ecoules = analyser(f, a.souvenir_du)
            if vue and not annonce:
                annonce = True
                print(f"[arret] extinction detected — observing {a.jours} lived days", flush=True)
            if vue and ecoules >= a.jours:
                print(f"[arret] {ecoules} lived days since extinction: the run can stop.",
                      flush=True)
                if a.arreter:
                    print("[arret] `make stop-run`", flush=True)
                    subprocess.run(["make", "stop-run"], check=False)
                return 0
        time.sleep(a.intervalle)
    print(f"[arret] {a.limite_h} h limit reached without extinction — nothing was stopped.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
