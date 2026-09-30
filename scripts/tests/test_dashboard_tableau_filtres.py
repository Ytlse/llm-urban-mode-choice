"""Displayed columns and per-column filter of the "Mes expériences" table.

One test per rule of the specification, named after it.

The thread of these tests: a filter is a VIEW. It does not touch the disk, it never makes
a row disappear without counting it, and it must not be able to silently hide what was
produced after it was set — all the more since it now survives closing the page.
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


# ── A pocket Streamlit ───────────────────────────────────────────────────────


class RerunDemande(Exception):
    """What `st.rerun` raises: in Streamlit it interrupts the script, here too."""


class FauxSt:
    """The bare minimum of the Streamlit API to draw the table and its filters.

    `columns`, `expander` and `popover` return this very object: everything that is drawn
    ends up in the same lists, whatever box contains it.
    """

    def __init__(self, event=None):
        self.session_state: dict = {}
        self.legendes: list[str] = []
        self.textes: list[str] = []
        self.avis: list[str] = []
        self.alertes: list[str] = []
        self.popovers: list[str] = []
        self.boutons: list[str] = []
        self.options: dict = {}
        self.tables: list = []
        self.clics: set = set()
        self.reruns: list = []
        self.event = event

    # — containers —
    def columns(self, spec, **_k):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [self] * n

    @contextlib.contextmanager
    def expander(self, label, expanded=False):
        yield self

    @contextlib.contextmanager
    def popover(self, label, **_k):
        self.popovers.append(label)
        yield self

    @contextlib.contextmanager
    def container(self, **_k):
        yield self

    @contextlib.contextmanager
    def spinner(self, *_a, **_k):
        yield self

    def empty(self):
        return self

    def fragment(self, run_every=None):
        self.run_every = run_every
        return lambda fonction: fonction

    def rerun(self, scope=None):
        self.reruns.append(scope)
        raise RerunDemande()

    # — widgets —
    def multiselect(self, label, options, default=None, key=None, **_k):
        self.options[key] = list(options)
        return self.session_state.get(key, list(default or []))

    def selectbox(self, label, options, index=0, key=None, **_k):
        options = list(options)
        return self.session_state.setdefault(key, options[index] if options else None)

    def text_input(self, label, value="", key=None, **_k):
        return self.session_state.setdefault(key, value)

    def text_area(self, label, value="", key=None, **_k):
        return self.session_state.setdefault(key, value)

    def number_input(self, label, *args, value=None, key=None, **_k):
        if len(args) >= 3:
            val = args[2]
        elif args:
            val = args[0]
        else:
            val = value
        return self.session_state.get(key, val)

    def checkbox(self, label, value=False, key=None, **_k):
        return self.session_state.get(key, value)

    def radio(self, label, options, index=0, key=None, **_k):
        options = list(options)
        return self.session_state.setdefault(key, options[index] if options else None)

    def date_input(self, label, value=None, key=None, **_k):
        return self.session_state.setdefault(key, value)

    def button(self, label, key=None, **_k):
        self.boutons.append(label)
        return key in self.clics

    def dataframe(self, donnees, **_k):
        # The table greys out its rows outside the reference (R22): what Streamlit receives is then
        # a `Styler`, not a `DataFrame`. We keep the BARE table — it is what the tests
        # query, and the styling is checked elsewhere (`test_dashboard_jeu_reference.py`).
        self.tables.append(getattr(donnees, "data", donnees))
        return self.event

    # — writes —
    def caption(self, texte, **_k):
        self.legendes.append(texte)

    def markdown(self, texte, **_k):
        self.textes.append(texte)

    def info(self, texte, **_k):
        self.avis.append(texte)

    def warning(self, texte, **_k):
        self.alertes.append(texte)

    def success(self, texte, **_k):
        self.avis.append(texte)

    def error(self, texte, **_k):
        self.alertes.append(texte)

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
        return "\n".join(self.legendes + self.textes + self.avis + self.alertes + self.popovers)


class FauxEvent:
    def __init__(self, rows):
        self.selection = {"rows": list(rows)}


# ── Data ─────────────────────────────────────────────────────────────────────

LIGNES = [
    {"experience": "exp_a", "execution": "2026-09-10_08_00_00", "etat": "terminee",
     "fournisseur": "google", "prompt": "minper", "composite_l1": 0.20, "couverture": 0.99},
    {"experience": "exp_b", "execution": "2026-09-10_09_00_00", "etat": "terminee",
     "fournisseur": "mistral", "prompt": "promin", "composite_l1": 0.40, "couverture": 0.98},
    {"experience": "exp_c", "execution": "2026-09-10_10_00_00", "etat": "en_cours",
     "fournisseur": "google", "prompt": None, "composite_l1": None, "couverture": None},
]
COLONNES = ["experience", "execution", "etat", "fournisseur", "prompt", "couverture", "composite_l1"]


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if chemin.suffix == ".json":
        chemin.write_text(json.dumps(contenu), encoding="utf-8")
    else:
        chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")


@pytest.fixture(autouse=True)
def vue_isolee(tmp_path, monkeypatch):
    """The view's memory lives in tmp: no test writes into `experiments/`."""
    monkeypatch.setattr(D, "ETAT_VUE_REGISTRE", tmp_path / "vue" / "vue.yaml")
    return tmp_path / "vue" / "vue.yaml"


@pytest.fixture(autouse=True)
def sans_jeu_de_reference(tmp_path, monkeypatch):
    """No reference set designated: these tests are not about the greying (R22).

    Otherwise they would read the repository's versioned file, and changing the reference
    substrate would break tests that have nothing to do with it.
    """
    monkeypatch.setattr(D, "JEU_REFERENCE_YAML", tmp_path / "absent" / "reference.yaml")


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    """Three experiments: two finished and scored, one running and never scored."""
    exps = tmp_path / "experiences"
    for nom, fournisseur, etat, score in (("exp_a", "google", "terminee", 0.20),
                                          ("exp_b", "mistral", "terminee", 0.40),
                                          ("exp_c", "google", "en_cours", None)):
        _ecrire(exps / nom / "experience.yaml",
                {"nom": nom, "mode": "sans_simulateur", "jeu": {"nom": "j1"},
                 "decideur": {"type": "passerelle", "modele": "m1"},
                 "gabarit": {"variante": "b_min"}})
        d = exps / nom / "executions" / f"2026-09-1{'012'[('exp_a', 'exp_b', 'exp_c').index(nom)]}_08_00_00"
        _ecrire(d / "etat.json", {"etat": etat})
        _ecrire(d / "compteurs.json", {"couverture": {"taux": 0.99, "decides": 99, "attendus": 100}})
        if score is not None:
            _ecrire(d / "scores.json", {"composite": {"l1": score, "emd_jsd": score},
                                        "formule": {"nom": "reference", "sha256": "sha-perimee"}})
    monkeypatch.setattr(D, "DOSSIER", exps)
    monkeypatch.setattr(D, "DOSSIER_JEUX", tmp_path / "jeux")
    return exps


def _dessiner(st) -> None:
    """A full drawing of the table, as the fragment does every 5 s."""
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)


# ── R1 · R2 — the displayed columns ──────────────────────────────────────────


def test_R1_les_colonnes_par_defaut_dans_l_ordre_annonce():
    st = FauxSt()
    presentes = list(D.COLONNES_REGISTRE)
    choix = D._panneau_colonnes_et_filtres(st, LIGNES, presentes, "")
    assert choix["colonnes"] == list(D.COLONNES_REGISTRE_DEFAUT)
    for sortie in ("scores", "jeu", "jeu_etat", "chaine", "formule"):
        assert sortie not in choix["colonnes"], sortie


def test_R21_jeu_n_est_plus_au_defaut_mais_garde_sa_place_canonique():
    """The set name appears in each table's title. Brought back, the column sits between prompt and mode."""
    st = FauxSt()
    presentes = list(D.COLONNES_REGISTRE)
    st.session_state[D._cle_vue(st, "colonnes")] = list(D.COLONNES_REGISTRE_DEFAUT) + ["jeu"]
    colonnes = D._panneau_colonnes_et_filtres(st, LIGNES, presentes, "")["colonnes"]
    assert colonnes.index("prompt") < colonnes.index("jeu") < colonnes.index("mode")


def test_R2_une_colonne_rappelee_reprend_sa_place():
    """Brought back, `jeu_etat` goes back between `prompt` (or `jeu`) and `mode` — never stuck at the row end."""
    st = FauxSt()
    presentes = list(D.COLONNES_REGISTRE)
    D._panneau_colonnes_et_filtres(st, LIGNES, presentes, "")
    st.session_state[D._cle_vue(st, "colonnes")] = list(D.COLONNES_REGISTRE_DEFAUT) + ["jeu", "jeu_etat"]
    colonnes = D._panneau_colonnes_et_filtres(st, LIGNES, presentes, "")["colonnes"]
    assert colonnes.index("prompt") < colonnes.index("jeu") < colonnes.index("jeu_etat") < colonnes.index("mode")


def test_R2_les_cinq_colonnes_restent_proposees_au_selecteur():
    st = FauxSt()
    D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert set(st.options[D._cle_vue(st, "colonnes")]) == set(D.COLONNES_REGISTRE)


# ── R3 — the warning does not go with the column ─────────────────────────────


def test_R3_une_formule_perimee_est_dite_meme_colonne_formule_masquee(plateforme, monkeypatch):
    monkeypatch.setattr(D, "_formule_reference_sha", lambda: "sha-courante")
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    assert "formule" not in D.COLONNES_REGISTRE_DEFAUT
    assert any("PÉRIMÉE" in a for a in st.alertes), st.alertes


def test_R3_sans_ligne_perimee_aucun_avertissement(plateforme, monkeypatch):
    monkeypatch.setattr(D, "_formule_reference_sha", lambda: "sha-perimee")  # same as the scores
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    assert not any("PÉRIMÉE" in a for a in st.alertes), st.alertes


# ── R4 to R6 — the filter by values ──────────────────────────────────────────


def test_R4_decocher_une_valeur_retire_ses_lignes():
    garde = D.appliquer_filtres(LIGNES, COLONNES, retenues={"etat": ["terminee"]})
    assert garde == [True, True, False]


def test_R5_les_colonnes_se_combinent_en_et_les_valeurs_en_ou():
    garde = D.appliquer_filtres(LIGNES, COLONNES,
                                retenues={"etat": ["terminee"], "fournisseur": ["google", "mistral"]})
    assert garde == [True, True, False]
    garde = D.appliquer_filtres(LIGNES, COLONNES,
                                retenues={"etat": ["terminee"], "fournisseur": ["mistral"]})
    assert garde == [False, True, False]


def test_R6_l_absence_de_valeur_se_filtre_sous_son_propre_nom():
    assert D.valeurs_filtrables(LIGNES, "prompt") == ["minper", "promin", D.VALEUR_VIDE]
    garde = D.appliquer_filtres(LIGNES, COLONNES, retenues={"prompt": ["minper", "promin"]})
    assert garde == [True, True, False], "the row without a prompt, and it alone, drops out"


# ── R7 — what is filtered is shown and counted ───────────────────────────────


def test_R7_le_selecteur_d_une_colonne_filtree_porte_son_compte():
    st = FauxSt()
    st.session_state[D._cle_vue(st, "val-fournisseur")] = ["google"]
    st.session_state[D._cle_vue(st, "vues-fournisseur")] = ["google", "mistral"]
    D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert any(p.startswith("🔹 fournisseur") and "1/2" in p for p in st.popovers), st.popovers


def test_R7_le_nombre_de_lignes_affichees_est_dit(plateforme):
    st = FauxSt(event=FauxEvent([]))
    st.session_state[D._cle_vue(st, "val-etat")] = ["terminee"]
    st.session_state[D._cle_vue(st, "vues-etat")] = ["en_cours", "terminee"]
    _dessiner(st)
    assert any("2 ligne(s) affichée(s) sur 3" in c for c in st.legendes), st.legendes


# ── R8 — reset ───────────────────────────────────────────────────────────────


def test_R8_reinitialiser_rend_toutes_les_lignes():
    st = FauxSt()
    st.session_state.update({D._cle_vue(st, "filtre"): "google", D._cle_vue(st, "colonnes"): ["etat"],
                             D._cle_vue(st, "val-etat"): ["terminee"],
                             D._cle_vue(st, "vues-etat"): ["en_cours", "terminee"],
                             D._cle_vue(st, "min-composite_l1"): 0.1,
                             "_vue_registre": {"exclues": {"etat": ["en_cours"]}}})
    D._oublier_filtres(st)
    assert st.session_state == {"_vue_registre": {}, "_vue_version": 1,
                                "_vue_registre_restauree": False}
    choix = D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert D.appliquer_filtres(LIGNES, choix["colonnes"], retenues=choix["retenues"],
                               bornes=choix["bornes"]) == [True, True, True]


def test_R8_les_widgets_changent_de_cle_au_lieu_d_etre_effaces():
    """Streamlit sends a widget's state back from the browser: a key that was merely
    deleted came back filled, and the table stayed filtered while the memory was empty."""
    st = FauxSt()
    avant = D._cle_vue(st, "val-etat")
    st.session_state[avant] = ["terminee"]
    D._oublier_filtres(st)
    assert D._cle_vue(st, "val-etat") != avant, "the generation must change"
    assert avant not in st.session_state


def test_R8_le_bouton_de_reinitialisation_est_offert():
    st = FauxSt()
    D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert any("Réinitialiser" in b for b in st.boutons), st.boutons


# ── R9 — filters survive the fragment's beat ─────────────────────────────────


def test_R9_un_filtre_survit_au_redessin(plateforme):
    st = FauxSt(event=FauxEvent([]))
    st.session_state[D._cle_vue(st, "val-experience")] = ["exp_b"]
    st.session_state[D._cle_vue(st, "vues-experience")] = ["exp_a", "exp_b", "exp_c"]
    _dessiner(st)
    _dessiner(st)  # the fragment beats again five seconds later
    assert st.session_state[D._cle_vue(st, "val-experience")] == ["exp_b"]
    assert any("1 ligne(s) affichée(s) sur 3" in c for c in st.legendes), st.legendes


# ── R10 — an invisible criterion does not filter ─────────────────────────────


def test_R10_un_filtre_sur_une_colonne_masquee_est_ignore():
    garde = D.appliquer_filtres(LIGNES, ["experience", "etat"], retenues={"fournisseur": ["mistral"]})
    assert garde == [True, True, True]


def test_R10_masquer_une_colonne_retire_son_filtre_du_choix():
    st = FauxSt()
    st.session_state[D._cle_vue(st, "val-fournisseur")] = ["google"]
    st.session_state[D._cle_vue(st, "vues-fournisseur")] = ["google", "mistral"]
    st.session_state[D._cle_vue(st, "colonnes")] = ["experience", "etat"]
    choix = D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert "fournisseur" not in choix["retenues"]
    assert D.appliquer_filtres(LIGNES, choix["colonnes"], retenues=choix["retenues"]) == [True] * 3


# ── R11 — the offered values do not depend on the other filters ──────────────


def test_R11_le_selecteur_d_une_colonne_ignore_les_filtres_voisins():
    st = FauxSt()
    st.session_state[D._cle_vue(st, "val-fournisseur")] = ["mistral"]
    st.session_state[D._cle_vue(st, "vues-fournisseur")] = ["google", "mistral"]
    D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert st.options[D._cle_vue(st, "val-etat")] == ["en_cours", "terminee"], \
        "the states of the Google rows stay offered, otherwise they could not be re-checked"


# ── R12 — the registry values move under the filter ──────────────────────────


def test_R12_une_valeur_qui_apparait_arrive_cochee():
    assert D.retenues_a_jour(["google", "mistral", "openai"], ["google"], ["google", "mistral"]) \
        == ["google", "openai"]


def test_R12_une_valeur_disparue_est_oubliee_sans_erreur():
    assert D.retenues_a_jour(["google"], ["google", "mistral"], ["google", "mistral"]) == ["google"]
    assert D.appliquer_filtres(LIGNES, COLONNES, retenues={"fournisseur": ["disparu"]}) == [False] * 3


# ── R13 — actions apply to the displayed row ─────────────────────────────────


def test_R13_les_actions_nomment_la_ligne_cochee_apres_filtrage(plateforme):
    st = FauxSt(event=FauxEvent([1]))
    # Sorted by descending run, the table shows exp_c then exp_a: the second
    # DISPLAYED row is exp_a, whereas index 1 of the unfiltered registry would be exp_b.
    st.session_state[D._cle_vue(st, "val-experience")] = ["exp_a", "exp_c"]
    st.session_state[D._cle_vue(st, "vues-experience")] = ["exp_a", "exp_b", "exp_c"]
    _dessiner(st)
    assert any("Actions sur « exp_a »" in t for t in st.textes), st.textes


# ── R14 — a running run stays controllable ───────────────────────────────────


def test_R14_une_execution_en_cours_filtree_garde_ses_boutons(plateforme):
    st = FauxSt(event=FauxEvent([]))
    st.session_state[D._cle_vue(st, "val-etat")] = ["terminee"]  # exp_c (running) drops out of the table
    st.session_state[D._cle_vue(st, "vues-etat")] = ["en_cours", "terminee"]
    _dessiner(st)
    assert any("exp_c" in t and "en cours" in t for t in st.textes), st.textes


# ── R15 — a filter writes nothing to the data ────────────────────────────────


def test_R15_filtrer_ne_touche_ni_les_experiences_ni_les_masques(plateforme):
    avant = {p: p.read_bytes() for p in plateforme.rglob("*") if p.is_file()}
    st = FauxSt(event=FauxEvent([]))
    st.session_state.update({D._cle_vue(st, "val-etat"): ["terminee"],
                             D._cle_vue(st, "vues-etat"): ["en_cours", "terminee"],
                             D._cle_vue(st, "min-composite_l1"): 0.3,
                             D._cle_vue(st, "filtre"): "exp_"})
    _dessiner(st)
    apres = {p: p.read_bytes() for p in plateforme.rglob("*") if p.is_file()}
    assert apres == avant


# ── R16 — the search is literal ──────────────────────────────────────────────


def test_R16_le_filtre_texte_n_est_pas_une_expression_reguliere():
    lignes = [{"experience": "exp_(alea"}, {"experience": "exp_durmin"}]
    assert D.appliquer_filtres(lignes, ["experience"], texte="exp_(alea") == [True, False]


def test_R16_le_filtre_texte_ignore_la_casse_et_cherche_partout():
    assert D.appliquer_filtres(LIGNES, COLONNES, texte="MISTRAL") == [False, True, False]


# ── R17 — numeric columns are filtered by bounds ─────────────────────────────


def test_R17_les_bornes_sont_incluses_et_les_non_scorees_optionnelles():
    sans = D.appliquer_filtres(LIGNES, COLONNES, bornes={"composite_l1": (0.10, 0.30, False)})
    assert sans == [True, False, False], "the row showing `—` drops out when the box is unchecked"
    avec = D.appliquer_filtres(LIGNES, COLONNES, bornes={"composite_l1": (0.10, 0.30, True)})
    assert avec == [True, False, True]
    bord = D.appliquer_filtres(LIGNES, COLONNES, bornes={"composite_l1": (0.20, 0.20, False)})
    assert bord == [True, False, False], "the bound is included"


def test_R17_une_borne_absente_ne_borne_pas():
    assert D.appliquer_filtres(LIGNES, COLONNES, bornes={"composite_l1": (None, 0.30, False)}) \
        == [True, False, False]


# ── R18 — the view survives closing the dashboard ────────────────────────────


def test_R18_ce_qui_est_ecrit_se_relit(vue_isolee):
    D.sauver_vue_registre(colonnes=["experience", "etat", "chaine"], exclues={"etat": ["en_cours"]},
                          bornes={"composite_l1": (0.1, None, False)}, texte="exp_")
    vue = D.charger_vue_registre()
    assert vue["ajoutees"] == ["chaine"]
    assert "experience" not in vue.get("retirees", []) and "mode" in vue["retirees"]
    assert vue["exclues"] == {"etat": ["en_cours"]}
    assert vue["bornes"] == {"composite_l1": (0.1, None, False)}
    assert vue["texte"] == "exp_"


def test_R18_une_colonne_absente_ce_jour_la_n_est_pas_retenue_comme_masquee(vue_isolee):
    """A registry with no scored run does not expose `composite_l1`: recording it as
    "removed" would lose it forever, although nobody ever unchecked it."""
    offertes = [c for c in D.COLONNES_REGISTRE_DEFAUT if c != "composite_l1"]
    D.sauver_vue_registre(colonnes=offertes, presentes=offertes, exclues={}, bornes={}, texte="")
    assert "composite_l1" not in D.charger_vue_registre().get("retirees", [])
    st = FauxSt()
    colonnes = D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")["colonnes"]
    assert colonnes == list(D.COLONNES_REGISTRE_DEFAUT), "the column comes back as soon as it exists"


def test_R18_une_colonne_rappelee_revient_a_l_ouverture_suivante(vue_isolee):
    st = FauxSt()
    st.session_state[D._cle_vue(st, "colonnes")] = list(D.COLONNES_REGISTRE_DEFAUT) + ["chaine"]
    D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    autre = FauxSt()
    colonnes = D._panneau_colonnes_et_filtres(autre, LIGNES, list(D.COLONNES_REGISTRE), "")["colonnes"]
    assert "chaine" in colonnes


def test_R18_un_etat_illisible_ou_absent_rend_le_tableau_par_defaut(vue_isolee):
    assert D.charger_vue_registre() == {}
    vue_isolee.parent.mkdir(parents=True, exist_ok=True)
    vue_isolee.write_text("retirees: [inconnue]\nexclues: 3\nbornes: [oui]\n", encoding="utf-8")
    assert D.charger_vue_registre() == {}
    st = FauxSt()
    assert D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")["colonnes"] \
        == list(D.COLONNES_REGISTRE_DEFAUT)


def test_R18_un_filtre_pose_revient_a_l_ouverture_suivante():
    st = FauxSt()
    st.session_state[D._cle_vue(st, "val-etat")] = ["terminee"]
    st.session_state[D._cle_vue(st, "vues-etat")] = ["en_cours", "terminee"]
    D._panneau_colonnes_et_filtres(st, LIGNES, list(D.COLONNES_REGISTRE), "")
    autre = FauxSt()  # new session: nothing left in live memory
    choix = D._panneau_colonnes_et_filtres(autre, LIGNES, list(D.COLONNES_REGISTRE), "")
    assert choix["retenues"]["etat"] == ["terminee"]


def test_R18_une_valeur_nee_apres_le_filtre_revient_cochee(vue_isolee):
    """That is why the disk keeps the EXCLUDED values, not the checked ones: a run
    launched tomorrow must not be born invisible under a filter written yesterday."""
    D.sauver_vue_registre(colonnes=D.COLONNES_REGISTRE_DEFAUT, exclues={"fournisseur": ["mistral"]},
                          bornes={}, texte="")
    st = FauxSt()
    choix = D._panneau_colonnes_et_filtres(
        st, LIGNES + [{"fournisseur": "openai"}], list(D.COLONNES_REGISTRE), "")
    assert set(choix["retenues"]["fournisseur"]) == {"google", "openai"}, \
        "mistral stays excluded, openai — unknown when the filter was written — arrives checked"


# ── R19 — restored filters that hide rows say so upfront ─────────────────────


def test_R19_un_filtre_restaure_qui_cache_des_lignes_est_annonce(plateforme, vue_isolee):
    D.sauver_vue_registre(colonnes=D.COLONNES_REGISTRE_DEFAUT,
                          exclues={"etat": ["en_cours"]}, bornes={}, texte="")
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    assert any("dernière session" in a and "etat" in a for a in st.avis), st.avis


def test_R19_sans_ligne_cachee_aucun_bandeau(plateforme, vue_isolee):
    D.sauver_vue_registre(colonnes=D.COLONNES_REGISTRE_DEFAUT,
                          exclues={"etat": ["etat_inexistant"]}, bornes={}, texte="")
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    assert not any("dernière session" in a for a in st.avis), st.avis


def test_R19_le_bandeau_tient_pendant_que_le_fragment_bat(plateforme, vue_isolee):
    """The registry redraws every 5 s while a run is going: a banner shown
    only once would vanish before being read."""
    D.sauver_vue_registre(colonnes=D.COLONNES_REGISTRE_DEFAUT,
                          exclues={"etat": ["en_cours"]}, bornes={}, texte="")
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    _dessiner(st)
    assert sum("dernière session" in a for a in st.avis) == 2, st.avis


def test_R19_le_bandeau_tombe_des_qu_on_reinitialise(plateforme, vue_isolee):
    D.sauver_vue_registre(colonnes=D.COLONNES_REGISTRE_DEFAUT,
                          exclues={"etat": ["en_cours"]}, bornes={}, texte="")
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    D._oublier_filtres(st)
    st.avis.clear()
    _dessiner(st)
    assert not any("dernière session" in a for a in st.avis), st.avis


# ── Swapped block order & forced-choice percentage ───────────────────────────


def test_ordre_des_blocs_mes_experiences_avant_nouvelle_experience(plateforme):
    """`📚 Mes expériences` must come before `🧪 Nouvelle expérience` in the page."""
    st = FauxSt(event=FauxEvent([]))
    with contextlib.suppress(RerunDemande):
        D.render(st, pd)
    subheaders = [t for t in st.textes if "Mes expériences" in t or "Nouvelle expérience" in t]
    assert len(subheaders) >= 2, subheaders
    assert "Mes expériences" in subheaders[0]
    assert "Nouvelle expérience" in subheaders[1]


def test_choix_forces_affiche_pourcentage_trois_decimales(plateforme):
    """The choix_forces column shows the percentage with at least 3 decimals."""
    exp_a_exec = plateforme / "exp_a" / "executions" / "2026-09-10_08_00_00"
    _ecrire(exp_a_exec / "synthese.json", {
        "choix_forces": {"n": 5, "part": 5 / 99},
        "parts_modales": {"n": 99}
    })
    st = FauxSt(event=FauxEvent([]))
    _dessiner(st)
    table = next(t for t in st.tables if "experience" in t.columns)
    assert "choix_forces" in table.columns
    valeurs = list(table["choix_forces"])
    # 5 / 99 = 0.050505... -> 5.051 %
    assert any(v == "5.051 %" for v in valeurs), valeurs
    # A run without a forced-choice summary shows "—"
    assert any(v == "—" for v in valeurs), valeurs

