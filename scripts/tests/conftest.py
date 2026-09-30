"""Safeguards shared by all tests in `scripts/tests`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHEMIN_SERVICES = REPO_ROOT / "services" / "llm-agents"
if str(_CHEMIN_SERVICES) not in sys.path:
    sys.path.insert(0, str(_CHEMIN_SERVICES))

try:
    from experiences import memoire as _memoire
except Exception:  # noqa: BLE001 — module absent ou cassé : rien à isoler
    _memoire = None


@pytest.fixture(autouse=True)
def _isoler_donnees_memoire(tmp_path_factory, monkeypatch):
    """Redirects the writes of memory experiments to a temporary directory.

    The 🧠 tab (ticket 109) records an experiment as soon as a button is true. Rendered
    with a MagicMock `st`, every `st.button` is true: without this detour, the test creates a
    `exp_mem_…_magicmock-…` directory in the real `data/experiences_memoire/` and rewrites
    `experiments/.dashboard/formulaire_memoire.yaml`.
    """
    if _memoire is None:
        yield None
        return
    racine = tmp_path_factory.mktemp("memoire_isolee")
    dossier = racine / "data" / "experiences" / "evenements_non_tabules"
    dossier.mkdir(parents=True)
    monkeypatch.setattr(_memoire, "DOSSIER_MEMOIRE", dossier)
    monkeypatch.setattr(_memoire, "ETAT_FORMULAIRE_MEMOIRE", racine / "formulaire_memoire.yaml")
    yield dossier

# ── Public copy: tests that read material this repository does not ship ─────────────────
# Skipped by name with their reason, never silently; `pytest -rs` lists them.
_PUBLIC_SKIPS = {('test_press_corpus_and_grid.py', 'test_C1_chaque_article_porte_son_brut_dans_les_deux_langues'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C1_les_cinq_textes_bruts_francais_existent_et_sont_non_vides'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C1_les_empreintes_du_manifeste_correspondent_aux_fichiers'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C2_un_mot_exempte_figure_dans_le_texte_brut'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C4_le_corpus_complet_se_charge'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C4_le_temoin_est_apparie_en_longueur_a_chaque_article'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C4_un_seul_temoin_sert_les_cinq_articles'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_press_corpus_and_grid.py', 'test_C6_l_entree_servie_a_l_agent_porte_la_mention_de_traduction'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_functional_bench.py', 'test_la_famille_A_tourne_entierement_sans_appel'): 'needs the press article texts, which are not redistributed; rebuild them with scripts/data/presse/fetch_articles.py (docs/DATA.md, section 6)', ('test_dashboard_app.py', 'test_R6_un_bouton_contextuel_resout_sa_cible_et_appelle_le_registre'): 'needs docs/synthesis/data.json, which `make synthesis` builds from runs that are not shipped', ('test_synthese_generation_population.py', 'test_la_page_se_construit_depuis_le_sceau_seul'): 'reads a paper-writing artefact, not shipped'}


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = _PUBLIC_SKIPS.get((item.path.name, getattr(item, "originalname", item.name)))
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
