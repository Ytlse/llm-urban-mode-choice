"""prompts — the system prompt variants (`prompts.yaml`, keys `active:` and `prompts:`).

Since iteration 2 of ticket 037, templates and output schemas live with each category,
in `mobility_llm/categories/<nom>/` (`template.md.j2`, `output_schema.json`). This file
stays here because prompt_calibration and the experiments cite it by this path.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = PROMPTS_DIR / "prompts.yaml"
CATEGORIES_DIR = PROMPTS_DIR.parent / "categories"

__all__ = ["CATEGORIES_DIR", "PROMPTS_DIR", "PROMPTS_FILE"]
