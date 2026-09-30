"""Roots resolved the same way on the host and inside the containers.

Two roots, and the right one must be chosen: `racine_depot()` for what lives next to
`services/llm-agents/` (`scripts/`, `data/`, `docs/`), `racine_llm_agents()` for what lives
INSIDE it (`text_helper/`, `config/`, `settings.py`). Mixing them up works on the host, where
one is the parent of the other, and breaks in the container, where `llm-agents` is mounted on
`/app`: `racine_depot()/"llm-agents"/x` gives `/app/llm-agents/x` there, which does not exist.
That is how the option template silently dropped out of the `empreinte_gabarit` hash,
and how the same prompt carried two fingerprints depending on where the experiment was
launched from (observed on 2026-09-11, ticket 045 alert A2).

--- Repository root ---

`Path(__file__).resolve().parents[N]` cannot fit both sides: on the host
this module lives in `<repo>/llm-agents/experiences/`, but in the `controller`
container the `llm-agents` folder is mounted on `/app` — so the module sits in
`/app/experiences/`, one level higher. `parents[2]` returned `/` there, and the model
decision-maker looked for its artefacts in `/scripts/progedo_logit/`: the experiment refused
to start with "policy not found" although the file was right there
(observed on 2026-09-08 on the Light_GBM experiment).

So we walk up to the first ancestor that holds `scripts/synthesis` — the same anchor
as `scripts.synthesis.sources.REPO_ROOT`, verified correct in both environments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ANCRE = Path("scripts") / "synthesis"


@lru_cache(maxsize=1)
def racine_depot() -> Path:
    """First ancestor of this file that contains `scripts/synthesis`.

    Raises rather than return a wrong path: a wrong root does not show, it turns
    into a "file not found" three calls further on.
    """
    ici = Path(__file__).resolve()
    for parent in ici.parents:
        if (parent / _ANCRE).is_dir():
            return parent
    raise RuntimeError(
        f"Repository root not found from {ici}: no ancestor contains "
        f"{_ANCRE}. In a container, check that the repository's `scripts/` folder is "
        "actually mounted (volume `./scripts:/app/scripts`)."
    )


@lru_cache(maxsize=1)
def racine_llm_agents() -> Path:
    """Folder `services/llm-agents/`: `<repo>/llm-agents` on the host, `/app` in the container.

    It is the folder that contains the `experiences` package — hence the parent of this file,
    on both sides, with no anchor to look for. Use it for everything shipped WITH the
    code: `text_helper/` templates, `config/`, top-level modules.
    """
    return Path(__file__).resolve().parents[1]


__all__ = ["racine_depot", "racine_llm_agents"]
