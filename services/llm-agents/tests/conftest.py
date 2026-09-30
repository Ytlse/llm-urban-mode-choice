"""Shared pytest configuration.

Adds to sys.path:
- the **repository root**, for the tests that import the full chain of the
  controller (which imports `mobility_core`, packaged at the root) — as in
  Docker where `mobility_core` is installed on the path;
- the **`services/llm-agents/`** folder itself, where the top-level modules
  `backpressure`, `settings`, `world`, etc. live. Without it, `pytest` launched from the
  repository root fails at collection (`ModuleNotFoundError: backpressure` /
  `settings`) and interrupts the whole run; with it, the tests are collected whatever
  the invocation directory.
"""

import sys

import pytest
from pathlib import Path

# Since ticket 039 `llm-agents` lives under `services/`: the root is THREE levels up,
# plus two. It is searched for by its anchor rather than counted, so that the next
# move does not break the collection (same reason as `experiences.chemins`).
_REPO_ROOT = next(
    (a for a in Path(__file__).resolve().parents if (a / "scripts" / "synthesis").is_dir()),
    Path(__file__).resolve().parents[3],
)
_LLM_AGENTS = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _LLM_AGENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Ticket 118, O2 — an unanswered survey question is retried on the spot (1, 2 then 4 min in
# production). The tests that simulate a failure must not each wait seven minutes.
import os as _os

_os.environ.setdefault("EXPERIMENT_SURVEY_RETRIES_S", "0,0,0")

# ── Public copy: tests that read material this repository does not ship ─────────────────
# Skipped by name with their reason, never silently; `pytest -rs` lists them.
_PUBLIC_SKIPS = {('test_lecture_et_relais_foyer.py', 'test_T11_les_cinq_articles_portent_leurs_deux_cles'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_lecture_et_relais_foyer.py', 'test_a13_force_la_parole_a_tout_le_foyer'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_event_channel_compat.py', 'test_R6quater_lire_une_declaration_et_armer_un_run_sont_deux_choses'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_event_read_channel.py', 'test_R16_les_cinq_articles_du_059_concordent_avec_leur_manifeste'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_event_read_channel.py', 'test_R17_la_mention_de_traduction_est_dans_lentree_servie'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_event_read_channel.py', 'test_les_foyers_declares_existent_dans_la_population_du_059'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_model_family_in_name.py', 'test_chaque_famille_se_nomme_elle_meme'): 'needs a model artefact derived from the restricted PROGEDO survey, not shipped'}


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = _PUBLIC_SKIPS.get((item.path.name, getattr(item, "originalname", item.name)))
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
