#!/usr/bin/env python3
"""Aggregate WARNING lines from app.log into a frequency table."""

import re
import sys
from collections import Counter

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "experiments/current/app.log"

# Patterns to strip variable data before grouping
_NORMALIZE = [
    # Remove task_id key-value pair
    (re.compile(r"\btask_id=[0-9a-f-]{36}\b"), ""),
    # UUIDs
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "{uuid}"),
    # Simulation timestamps in brackets
    (re.compile(r"\[timestamp:[^\]]+\]"), ""),
    # lon/lat pairs
    (re.compile(r"\blon=-?[\d.]+\s+lat=-?[\d.]+"), ""),
    # OTP from=(x,y) to=(x,y) coordinates
    (re.compile(r"from=\(-?[\d.]+,-?[\d.]+\)\s+to=\(-?[\d.]+,-?[\d.]+\)"), ""),
    # Numpy shape tuples like (1777,) (1778,)
    (re.compile(r"\(\d+,\)\s+\(\d+,\)"), "({n},) ({m},)"),
    # waited=Ns / timeout=Ns
    (re.compile(r"\b(waited|timeout)=[\d.]+s\b"), ""),
    # public_transport flag
    (re.compile(r"\bpublic_transport=(True|False)\b"), ""),
    # Variable person/entity IDs
    (re.compile(r"\bpour \d+:"), "pour {id}:"),
    (re.compile(r"\bperson \d+\b"), "person {id}"),
    (re.compile(r"\bfor \d+\b"), "for {id}"),
    (re.compile(r"\bto \d+\b"), "to {id}"),
    # Collapse multiple spaces
    (re.compile(r"  +"), " "),
]

WARNING_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| WARNING\s+\| (.+)$")


def normalize(msg: str) -> str:
    for pattern, replacement in _NORMALIZE:
        msg = pattern.sub(replacement, msg)
    return msg.strip().rstrip("|").strip()


counts: Counter = Counter()
with open(LOG_FILE) as f:
    for line in f:
        m = WARNING_RE.match(line.rstrip())
        if m:
            counts[normalize(m.group(1))] += 1

if not counts:
    print("No warnings found.")
    sys.exit(0)

rows = sorted(counts.items(), key=lambda x: -x[1])
col_w = max(len(w) for w, _ in rows)
col_w = max(col_w, 7)  # min width for "Warning"

header = f"| {'Warning':<{col_w}} | {'Occurrences':>11} |"
sep    = f"+-{'-' * col_w}-+{'-' * 13}+"
print(sep)
print(header)
print(sep)
for warning, count in rows:
    print(f"| {warning:<{col_w}} | {count:>11} |")
print(sep)
print(f"  Total: {sum(counts.values())} warnings, {len(counts)} distinct types")
