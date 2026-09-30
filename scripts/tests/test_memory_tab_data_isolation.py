"""Rendering the memory dashboard tab under test never writes into the real data."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dashboard import memoire as ONGLET

VRAI_DOSSIER = REPO_ROOT / "data" / "experiences" / "evenements_non_tabules"
VRAI_FORMULAIRE = REPO_ROOT / "experiments" / ".dashboard" / "formulaire_memoire.yaml"


def _st_tout_clique() -> MagicMock:
    """A MagicMock `st`: every `st.button` returns a truthy mock, so every action fires."""
    st = MagicMock()
    st.columns.side_effect = lambda spec, **_: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    st.session_state = {}
    return st


def _empreinte(chemin: Path):
    return chemin.stat().st_mtime_ns if chemin.exists() else None


def test_rendu_mocke_ecrit_sous_tmp_et_pas_dans_data(_isoler_donnees_memoire):
    avant = set(VRAI_DOSSIER.iterdir()) if VRAI_DOSSIER.exists() else set()
    formulaire_avant = _empreinte(VRAI_FORMULAIRE)

    # The folder is created before the YAML, which fails on the MagicMock values: that is
    # how an empty `exp_mem_…_magicmock-…` folder was left behind in `data/`.
    with contextlib.suppress(Exception):
        ONGLET.render(_st_tout_clique(), pd)

    apres = set(VRAI_DOSSIER.iterdir()) if VRAI_DOSSIER.exists() else set()
    assert apres == avant, f"folders created in the real data: {apres - avant}"
    assert _empreinte(VRAI_FORMULAIRE) == formulaire_avant
    # The "Enregistrer" button did fire: the write happened, but under tmp.
    assert any(_isoler_donnees_memoire.iterdir())
