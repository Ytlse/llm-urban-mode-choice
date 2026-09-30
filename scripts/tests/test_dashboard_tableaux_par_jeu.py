"""Tests of partitioning the experiments table by test set.

Checks:
1. Normalisation of set keys (empty values, dashes, etc.).
2. Ordering of sets (current reference, old sets, others alphabetically, no set).
3. Generation of group titles with badges and explanatory captions.
4. Rendering of one table per test set without a redundant `jeu` column.
5. Synchronised handling of the selection across tables.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences as D  # noqa: E402


class RerunDemande(Exception):
    pass


class FauxSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cles_instanciees: set[str] = set()

    def __setitem__(self, key, value):
        if key in self.cles_instanciees:
            raise RuntimeError(f"st.session_state.{key} cannot be modified after the widget with key {key} is instantiated.")
        super().__setitem__(key, value)


class FauxSt:
    def __init__(self, event=None):
        self.session_state = FauxSessionState()
        self.legendes: list[str] = []
        self.textes: list[str] = []
        self.avis: list[str] = []
        self.alertes: list[str] = []
        self.options: dict = {}
        self.tables: list = []
        self.boutons: list[str] = []
        self.event = event

    def columns(self, spec, **_k):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [self] * n

    @contextlib.contextmanager
    def expander(self, label, expanded=False):
        yield self

    @contextlib.contextmanager
    def popover(self, label, **_k):
        yield self

    def empty(self):
        return self

    def fragment(self, run_every=None):
        return lambda f: f

    def rerun(self, scope=None):
        raise RerunDemande()

    def multiselect(self, label, options, default=None, key=None, **_k):
        self.options[key] = list(options)
        return self.session_state.get(key, list(default or []))

    def selectbox(self, label, options, index=0, key=None, **_k):
        options = list(options)
        return self.session_state.setdefault(key, options[index] if options else None)

    def text_input(self, label, value="", key=None, **_k):
        return self.session_state.setdefault(key, value)

    def number_input(self, label, *args, value=None, key=None, **_k):
        return self.session_state.get(key, args[2] if len(args) >= 3 else value)

    def checkbox(self, label, value=False, key=None, **_k):
        return self.session_state.get(key, value)

    def button(self, label, key=None, **_k):
        self.boutons.append(label)
        return False

    def dataframe(self, donnees, key=None, **_k):
        self.tables.append(donnees)
        if key:
            self.session_state.cles_instanciees.add(key)
            if key in self.session_state:
                return self.session_state[key]
        return self.event

    def caption(self, texte, **_k):
        self.legendes.append(texte)

    def markdown(self, texte, **_k):
        self.textes.append(texte)

    def info(self, texte, **_k):
        self.avis.append(texte)

    def warning(self, texte, **_k):
        self.alertes.append(texte)

    def progress(self, valeur, text=""):
        self.textes.append(text)

    def toast(self, *_a, **_k):
        pass


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if chemin.suffix == ".json":
        chemin.write_text(json.dumps(contenu), encoding="utf-8")
    else:
        chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")


# ── Unit tests of the helper functions ───────────────────────────────────────


def test_normaliser_cle_jeu():
    assert D.normaliser_cle_jeu(None) == "sans_jeu"
    assert D.normaliser_cle_jeu("") == "sans_jeu"
    assert D.normaliser_cle_jeu("   ") == "sans_jeu"
    assert D.normaliser_cle_jeu("—") == "sans_jeu"
    assert D.normaliser_cle_jeu("None") == "sans_jeu"
    assert D.normaliser_cle_jeu("jeu_v6") == "jeu_v6"


def test_ordonner_jeux():
    ref = "jeu_ref"
    anciens = ["jeu_ancien_1", "jeu_ancien_2"]
    cles = ["autre_b", "jeu_ancien_2", "sans_jeu", "autre_a", "jeu_ref", "jeu_ancien_1"]

    ordonne = D.ordonner_jeux(cles, ref=ref, anciens=anciens)
    assert ordonne == [
        "jeu_ref",       # Active reference first
        "jeu_ancien_1",  # Old sets in historical order
        "jeu_ancien_2",
        "autre_a",       # Other sets sorted by name
        "autre_b",
        "sans_jeu",      # No set at the end
    ]


def test_titre_groupe_jeu():
    ref = "jeu_ref"
    anciens = ["jeu_ancien"]

    titre_ref, cap_ref = D.titre_groupe_jeu("jeu_ref", 1, ref, anciens)
    assert "🎯" in titre_ref and "référence" in titre_ref and "1 exécution" in titre_ref
    assert "référence actif" in cap_ref

    titre_anc, cap_anc = D.titre_groupe_jeu("jeu_ancien", 5, ref, anciens)
    assert "📦" in titre_anc and "ancien jeu" in titre_anc and "5 exécutions" in titre_anc
    assert "Ancien substrat" in cap_anc

    titre_autre, cap_autre = D.titre_groupe_jeu("autre_jeu", 2, ref, anciens)
    assert "📦" in titre_autre and "2 exécutions" in titre_autre
    assert "Substrat de test spécifique" in cap_autre

    titre_sj, cap_sj = D.titre_groupe_jeu("sans_jeu", 1, ref, anciens)
    assert "⚪" in titre_sj and "Sans jeu" in titre_sj
    assert "sans jeu de test associé" in cap_sj


# ── Integration tests of the per-set rendering ───────────────────────────────


@pytest.fixture
def multi_jeux_plateforme(tmp_path, monkeypatch):
    exps = tmp_path / "experiences"
    ref_yaml = tmp_path / "jeux" / "reference.yaml"
    _ecrire(ref_yaml, {
        "jeu": "jeu_ref",
        "depuis": "2026-09-16",
        "anciens": [{"jeu": "jeu_ancien"}],
    })
    monkeypatch.setattr(D, "JEU_REFERENCE_YAML", ref_yaml)
    monkeypatch.setattr(D, "DOSSIER", exps)
    monkeypatch.setattr(D, "DOSSIER_JEUX", tmp_path / "jeux")
    monkeypatch.setattr(D, "ETAT_VUE_REGISTRE", tmp_path / "vue.yaml")

    # 4 experiments on 3 different substrates + one without a set
    configurations = [
        ("exp_sur_ref", "jeu_ref"),
        ("exp_sur_ancien", "jeu_ancien"),
        ("exp_sur_autre", "jeu_enquete"),
        ("exp_sans_jeu", None),
    ]
    for nom, j in configurations:
        jeu_dict = {"nom": j} if j else {}
        definition = {"nom": nom, "mode": "sans_simulateur", "jeu": jeu_dict,
                      "decideur": {"type": "aleatoire"}}
        _ecrire(exps / nom / "experience.yaml", definition)
        d = exps / nom / "executions" / "2026-09-18_10_00_00"
        _ecrire(d / "etat.json", {"etat": "terminee"})
        _ecrire(d / "execution.yaml", {"experience": definition, "cree_le": "2026-09-18T10:00:00"})

    return exps


def test_rendu_trois_jeux_et_sans_jeu_produit_quatre_tableaux(multi_jeux_plateforme):
    st = FauxSt()
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)

    tables_registre = [t for t in st.tables if "experience" in getattr(t, "data", t).columns]
    assert len(tables_registre) == 4, "4 different substrates (ref, ancien, enquete, sans_jeu) = 4 tables"

    # Check that the 'jeu' column is absent from each table
    for table in tables_registre:
        vue = getattr(table, "data", table)
        assert "jeu" not in vue.columns, "The 'jeu' column must not appear in the tables"

    # Expected order: jeu_ref, jeu_ancien, jeu_enquete, sans_jeu
    assert list(tables_registre[0]["experience"]) == ["exp_sur_ref"]
    assert list(tables_registre[1]["experience"]) == ["exp_sur_ancien"]
    assert list(tables_registre[2]["experience"]) == ["exp_sur_autre"]
    assert list(tables_registre[3]["experience"]) == ["exp_sans_jeu"]

    # Check of the titles
    titres = [t for t in st.textes if t.startswith("####")]
    assert any("🎯 jeu_ref · référence" in t for t in titres)
    assert any("📦 jeu_ancien · ancien jeu de référence" in t for t in titres)
    assert any("📦 jeu_enquete" in t for t in titres)
    assert any("⚪ Sans jeu de test" in t for t in titres)


def test_selection_d_une_ligne_active_les_actions(multi_jeux_plateforme):
    event_sel = {"selection": {"rows": [0]}}
    st = FauxSt(event=event_sel)
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)

    # The action buttons must be shown for exp_sur_ref
    assert any("Actions sur « exp_sur_ref »" in t for t in st.textes)
    assert "🔁 Rejouer" in st.boutons
    assert "📋 Dupliquer" in st.boutons


def test_synchronisation_selection_multi_tables_sans_erreur_session_state(multi_jeux_plateforme):
    """Checks that switching selection between tables never modifies an already instantiated widget."""
    st = FauxSt()

    # Step 1: selection of a row in the first table (jeu_ref)
    st.session_state["exp-table-jeu_ref"] = {"selection": {"rows": [0], "columns": []}}
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)

    assert st.session_state.get("_table_active_cle") == "jeu_ref"
    assert st.session_state.get("_table_active_row") == 0

    # New Streamlit render cycle (widgets are re-instantiated on every run)
    st.session_state.cles_instanciees.clear()
    st.tables.clear()
    st.textes.clear()
    st.boutons.clear()

    # Step 2: user selects a row in another table (jeu_ancien)
    st.session_state["exp-table-jeu_ancien"] = {"selection": {"rows": [0], "columns": []}}
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)

    # Must have switched to jeu_ancien without raising a RuntimeError
    assert st.session_state.get("_table_active_cle") == "jeu_ancien"
    assert st.session_state.get("_table_active_row") == 0
    # The other table must have been cleared before its instantiation
    assert st.session_state.get("exp-table-jeu_ref") == {"selection": {"rows": [], "columns": []}}

