#!/usr/bin/env python3
"""Aggregate ERROR lines from app.log into a frequency table."""

import re
import sys
from collections import Counter

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "experiments/current/app.log"

_NORMALIZE = [
    (re.compile(r"\btask_id=[0-9a-f-]{36}\b"), ""),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "{uuid}"),
    (re.compile(r"\[timestamp:[^\]]+\]"), ""),
    (re.compile(r"\blon=-?[\d.]+\s+lat=-?[\d.]+"), ""),
    (re.compile(r"from=\(-?[\d.]+,-?[\d.]+\)\s+to=\(-?[\d.]+,-?[\d.]+\)"), ""),
    (re.compile(r"\(\d+,\)\s+\(\d+,\)"), "({n},) ({m},)"),
    (re.compile(r"\b(waited|timeout)=[\d.]+s\b"), ""),
    (re.compile(r"\bpublic_transport=(True|False)\b"), ""),
    (re.compile(r"\bpour \d+:"), "pour {id}:"),
    (re.compile(r"\bperson \d+\b"), "person {id}"),
    (re.compile(r"\bfor \d+\b"), "for {id}"),
    (re.compile(r"\bto \d+\b"), "to {id}"),
    (re.compile(r"  +"), " "),
]

ERROR_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ERROR\s+\| (.+)$")


def normalize(msg: str) -> str:
    for pattern, replacement in _NORMALIZE:
        msg = pattern.sub(replacement, msg)
    return msg.strip().rstrip("|").strip()


counts: Counter = Counter()
with open(LOG_FILE) as f:
    for line in f:
        m = ERROR_RE.match(line.rstrip())
        if m:
            counts[normalize(m.group(1))] += 1

if not counts:
    print("No errors found.")
    sys.exit(0)

rows = sorted(counts.items(), key=lambda x: -x[1])
col_w = max(len(e) for e, _ in rows)
col_w = max(col_w, 5)  # min width for "Error"

header = f"| {'Error':<{col_w}} | {'Occurrences':>11} |"
sep    = f"+-{'-' * col_w}-+{'-' * 13}+"
print(sep)
print(header)
print(sep)
for error, count in rows:
    print(f"| {error:<{col_w}} | {count:>11} |")
print(sep)
print(f"  Total: {sum(counts.values())} errors, {len(counts)} distinct types")
