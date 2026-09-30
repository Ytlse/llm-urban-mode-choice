"""`jeu` column and greying of rows outside the reference substrate.

The thread: a composite score is compared only within the same set. The table must therefore
say on WHICH set each run ran — the one frozen in the run, never the one the
definition designates today — and distinguish at a glance what cannot be compared.
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

REF = "jeu_corrige"
ANCIEN = "jeu_ancien"


class RerunDemande(Exception):
    """What `st.rerun` raises: in Streamlit it interrupts the script, here too."""


class FauxSt:
    """The bare minimum to draw the table and read back what comes out of it."""

    def __init__(self, event=None):
        self.session_state: dict = {}
        self.legendes: list[str] = []
        self.textes: list[str] = []
        self.avis: list[str] = []
        self.alertes: list[str] = []
        self.options: dict = {}
        self.tables: list = []          # what Streamlit received, Styler included
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

    @contextlib.contextmanager
    def container(self, **_k):
        yield self

    def empty(self):
        return self

    def fragment(self, run_every=None):
        self.run_every = run_every
        return lambda fonction: fonction

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
        return False

    def dataframe(self, donnees, **_k):
        self.tables.append(donnees)
        return self.event

    def caption(self, texte, **_k):
        self.legendes.append(texte)

    def markdown(self, texte, **_k):
        self.textes.append(texte)

    def info(self, texte, **_k):
        self.avis.append(texte)

    def warning(self, texte, **_k):
        self.alertes.append(texte)

    def error(self, texte, **_k):
        self.alertes.append(texte)

    def success(self, texte, **_k):
        self.avis.append(texte)

    def progress(self, valeur, text=""):
        self.textes.append(text)

    def subheader(self, texte, **_k):
        self.textes.append(texte)

    def divider(self):
        pass

    def toast(self, *_a, **_k):
        pass

    def code(self, *_a, **_k):
        pass

    @property
    def tout(self) -> str:
        return "\n".join(self.legendes + self.textes + self.avis + self.alertes)


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if chemin.suffix == ".json":
        chemin.write_text(json.dumps(contenu), encoding="utf-8")
    else:
        chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")


@pytest.fixture(autouse=True)
def vue_isolee(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ETAT_VUE_REGISTRE", tmp_path / "vue" / "vue.yaml")


@pytest.fixture
def reference(tmp_path, monkeypatch):
    """The versioned file that designates the reference substrate."""
    chemin = tmp_path / "jeux" / "reference.yaml"
    _ecrire(chemin, {"jeu": REF, "depuis": "2026-09-16"})
    monkeypatch.setattr(D, "JEU_REFERENCE_YAML", chemin)
    return chemin


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    """Two experiments whose DEFINITIONS both name the corrected set…

    …but only one of which actually ran on it: `exp_vieille` ran on the old
    substrate, and its run snapshot says so. This is exactly the state left when
    the definitions were re-pointed after the fact.
    """
    exps = tmp_path / "experiences"
    for nom, jeu_execute in (("exp_a_jour", REF), ("exp_vieille", ANCIEN)):
        definition = {"nom": nom, "mode": "sans_simulateur", "jeu": {"nom": REF},
                      "decideur": {"type": "passerelle", "modele": "m1"},
                      "gabarit": {"variante": "b_min"}}
        _ecrire(exps / nom / "experience.yaml", definition)
        d = exps / nom / "executions" / "2026-09-10_08_00_00"
        _ecrire(d / "etat.json", {"etat": "terminee"})
        _ecrire(d / "compteurs.json", {"couverture": {"taux": 0.99, "decides": 99, "attendus": 100}})
        _ecrire(d / "execution.yaml",
                {"cree_le": "2026-09-10T08:00:00", "experience": {**definition, "jeu": {"nom": jeu_execute}}})
    monkeypatch.setattr(D, "DOSSIER", exps)
    monkeypatch.setattr(D, "DOSSIER_JEUX", tmp_path / "jeux_inexistants")
    return exps


def _dessiner(st) -> None:
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)


def _lignes(nom_a_jeu: dict) -> list[dict]:
    return [{"experience": n, "execution": "e", "etat": "terminee", "jeu": j,
             "hors_reference": D.hors_reference(j, REF)} for n, j in nom_a_jeu.items()]


# ── R21 — the column states the set ACTUALLY run ─────────────────────────────


def test_R21_le_jeu_affiche_est_celui_fige_dans_l_execution(plateforme, reference):
    """The definition says "corrected" for both; the archive tells the truth for each."""
    par_exp = {l["experience"]: l for l in D.lister() if l.get("execution")}
    assert par_exp["exp_a_jour"]["jeu"] == REF
    assert par_exp["exp_vieille"]["jeu"] == ANCIEN, "the re-pointed definition masked the archive"


def test_R21_une_experience_jamais_lancee_montre_le_jeu_qu_elle_designe(plateforme, reference):
    """Without a run there is nothing to freeze: the definition is the only honest source."""
    _ecrire(plateforme / "exp_definie" / "experience.yaml",
            {"nom": "exp_definie", "mode": "sans_simulateur", "jeu": {"nom": REF},
             "decideur": {"type": "passerelle", "modele": "m1"}})
    ligne = next(l for l in D.lister() if l["experience"] == "exp_definie")
    assert ligne["jeu"] == REF and ligne["hors_reference"] is False


def test_un_tableau_par_jeu_sans_colonne_jeu(plateforme, reference):
    """Each test set has its own table, and the jeu column is no longer shown in the table."""
    st = FauxSt()
    _dessiner(st)
    tables_jeux = [t for t in st.tables if "experience" in getattr(t, "data", t).columns]
    assert len(tables_jeux) == 2, "Two distinct sets (REF and ANCIEN) must produce two tables"
    for table in tables_jeux:
        vue = getattr(table, "data", table)
        assert "jeu" not in vue.columns, "The set appears in the title, not in the columns"
        assert "experience" in vue.columns

    # The reference table comes first
    vue_ref = getattr(tables_jeux[0], "data", tables_jeux[0])
    vue_ancien = getattr(tables_jeux[1], "data", tables_jeux[1])
    assert "exp_a_jour" in vue_ref["experience"].values
    assert "exp_vieille" in vue_ancien["experience"].values

    # The markdown titles contain the set names and the reference badge
    titres = [t for t in st.textes if t.startswith("####")]
    assert any(REF in t and "référence" in t for t in titres), titres
    assert any(ANCIEN in t for t in titres), titres


def test_un_jeu_absent_cree_un_groupe_dedie_sans_jeu(plateforme, reference):
    """An experiment with no attached set produces a dedicated table titled Sans jeu."""
    _ecrire(plateforme / "exp_sans_jeu" / "experience.yaml",
            {"nom": "exp_sans_jeu", "mode": "sans_simulateur",
             "decideur": {"type": "passerelle", "modele": "m1"}})
    st = FauxSt()
    _dessiner(st)
    titres = [t for t in st.textes if t.startswith("####")]
    assert any("Sans jeu de test" in t for t in titres), titres


def test_sans_reference_designee_les_tableaux_sont_rendus(plateforme, tmp_path, monkeypatch):
    """Without a reference file, the tables are still rendered per set without error."""
    monkeypatch.setattr(D, "JEU_REFERENCE_YAML", tmp_path / "absent.yaml")
    assert D.jeu_reference() is None
    st = FauxSt()
    _dessiner(st)
    tables_jeux = [t for t in st.tables if "experience" in getattr(t, "data", t).columns]
    assert len(tables_jeux) == 2


def test_reference_illisible_ne_bloque_pas(plateforme, tmp_path, monkeypatch):
    """A damaged file must neither raise nor block: jeu_reference() returns None."""
    abime = tmp_path / "reference.yaml"
    abime.write_text("jeu: [pas, une, chaine\n", encoding="utf-8")
    monkeypatch.setattr(D, "JEU_REFERENCE_YAML", abime)
    assert D.jeu_reference() is None
    st = FauxSt()
    _dessiner(st)
    tables_jeux = [t for t in st.tables if "experience" in getattr(t, "data", t).columns]
    assert len(tables_jeux) == 2


def test_hors_reference_helper():
    assert D.hors_reference(ANCIEN, REF) is True
    assert D.hors_reference(REF, REF) is False
    assert D.hors_reference(ANCIEN, None) is False
    assert D.hors_reference(None, REF) is False
