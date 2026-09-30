"""Category `itinary_multi_agent`: template, output schema and observation hook.

The folder carries the three parts of the category: `template.md.j2`, `output_schema.json` and
`observe.py` (domain metrics). The public names are re-exported here so that
`from mobility_llm.categories.itinary_multi_agent import extract_primary_mode` stays valid.
"""
from mobility_llm.categories.itinary_multi_agent.observe import (
    MODE_MISMATCH_ALARM_RATIO,
    MODE_MISMATCH_MIN_SAMPLE,
    WORKER_MODE_ALIASES,
    WORKER_MODE_LABEL,
    count_mode_mismatches,
    distance_bracket,
    extract_primary_mode,
    observe_itinary,
)

__all__ = [
    "MODE_MISMATCH_ALARM_RATIO",
    "MODE_MISMATCH_MIN_SAMPLE",
    "WORKER_MODE_ALIASES",
    "WORKER_MODE_LABEL",
    "count_mode_mismatches",
    "distance_bracket",
    "extract_primary_mode",
    "observe_itinary",
]
