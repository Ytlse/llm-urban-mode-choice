"""The conditions sheet of an experiment, and the ⏳ / 📅 markers of the registry.

One test per rule of the specification, named after it.

The thread: an experiment's computed name carries its settings in shorthand, unreadable.
The sheet states them plainly, READ FROM THE DEFINITION (frozen in the run when there is one),
never decoded from the name. And the registry tells what is running from what a live
campaign is still going to launch.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences as D  # noqa: E402

VRAI_LISTER = D.lister


# ── A pocket Streamlit ───────────────────────────────────────────────────────


class RerunDemande(Exception):
    """What `st.rerun` raises: in Streamlit it interrupts the script, here too."""


class FauxSt:
    def __init__(self, event=None):
        self.session_state: dict = {}
        self.legendes: list[str] = []
        self.textes: list[str] = []
        self.avis: list[str] = []
        self.alertes: list[str] = []
        self.boutons: list[str] = []
        self.barres: list[tuple[float, str]] = []
        self.tables: list = []
        self.reruns: list = []
        self.event = event

    def columns(self, spec, **_k):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [self] * n

    @contextlib.contextmanager
    def expander(self, label, **_k):
        yield self

    @contextlib.contextmanager
    def popover(self, label, **_k):
        yield self

    @contextlib.contextmanager
    def container(self, **_k):
        yield self

    def fragment(self, run_every=None):
        return lambda fonction: fonction

    def rerun(self, scope=None):
        self.reruns.append(scope)
        raise RerunDemande()

    def multiselect(self, label, options, default=None, key=None, **_k):
        return self.session_state.get(key, list(default or []))

    def selectbox(self, label, options, index=0, key=None, **_k):
        options = list(options)
        return self.session_state.setdefault(key, options[index] if options else None)

    def text_input(self, label, value="", key=None, **_k):
        return self.session_state.setdefault(key, value)

    def number_input(self, label, *args, value=None, key=None, **_k):
        return self.session_state.get(key, args[2] if len(args) >= 3 else (args[0] if args else value))

    def checkbox(self, label, value=False, key=None, **_k):
        return self.session_state.get(key, value)

    def radio(self, label, options, index=0, key=None, **_k):
        options = list(options)
        return self.session_state.setdefault(key, options[index] if options else None)

    def button(self, label, key=None, **_k):
        self.boutons.append(label)
        return False

    def dataframe(self, donnees, **_k):
        # The table greys out rows outside the reference (R22): what Streamlit receives is then
        # a `Styler`, not a `DataFrame`. We keep the BARE table — that is what the tests
        # query, and the style is checked elsewhere (`test_dashboard_jeu_reference.py`).
        self.tables.append(getattr(donnees, "data", donnees))
        return self.event

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
        self.barres.append((valeur, text))

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
        return "\n".join(self.legendes + self.textes + self.avis + self.alertes + [t for _, t in self.barres])


class FauxEvent:
    def __init__(self, rows):
        self.selection = {"rows": list(rows)}


# ── Data ─────────────────────────────────────────────────────────────────────

DEFINITION_LLM = {
    "nom": "exp_llm",
    "population": {"chemin": "/data/eqasim-output/population_1000_PANEL_v6"},
    "jeu": {"nom": "j1"},
    "gabarit": {"categorie": "itinary_multi_agent", "variante": "prompt_expert_04"},
    "decideur": {"type": "passerelle", "modele": "gemini-3.5-flash-lite", "portee": "distant",
                 "parametres": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096,
                                "thinking_level": "high"},
                 "rejeu_de": None, "graine": None, "artefact": None},
    "mode": "sans_simulateur",
    "calendrier": {"politique": "aleatoire", "date": "2026-03-16", "graine": 42},
    "horizon_jours": 1, "memoire": False, "graine_ordre": 42, "graine_tirage": 42,
    "regroupement": {"parallelisme": 8}, "max_candidats": 6, "attente_max_s": 120,
    "vehicule_chaine": True, "verrou_retour": True,
}
EXEC_LLM = "2026-09-15_05_18_34"
EXEC_TEMOIN = "2026-09-14_08_00_00"


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if chemin.suffix == ".json":
        chemin.write_text(json.dumps(contenu), encoding="utf-8")
    else:
        chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")


def _pid_mort() -> int:
    """The pid of a process that has finished: it no longer exists at test time."""
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


@pytest.fixture(autouse=True)
def sans_jeu_de_reference(tmp_path, monkeypatch):
    """No reference set designated: these tests are not about the greying out (R22).

    Without it they would read the repository's versioned file, and changing reference substrate
    would break tests that have nothing to do with it.
    """
    monkeypatch.setattr(D, "JEU_REFERENCE_YAML", tmp_path / "absent" / "reference.yaml")


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    """Three experiments: a running LLM arm, a finished control, a defined one never launched."""
    exps, campagnes = tmp_path / "experiences", tmp_path / "campagnes"

    _ecrire(exps / "exp_llm" / "experience.yaml", DEFINITION_LLM)
    d = exps / "exp_llm" / "executions" / EXEC_LLM
    _ecrire(d / "etat.json", {"etat": "en_cours"})
    _ecrire(d / "execution.yaml", {"cree_le": "2026-09-15T05:18:34", "experience": DEFINITION_LLM})
    _ecrire(d / "progression.json", {"faits": 120, "attendus": 400, "pourcent": 30, "personnes": 50,
                                     "personnes_terminees": 12, "ecoule_s": 60})

    _ecrire(exps / "exp_temoin" / "experience.yaml",
            {"nom": "exp_temoin", "jeu": {"nom": "j1"}, "mode": "sans_simulateur",
             "decideur": {"type": "aleatoire", "graine": 7}, "gabarit": {"variante": "minimal_persona"}})
    _ecrire(exps / "exp_temoin" / "executions" / EXEC_TEMOIN / "etat.json", {"etat": "terminee"})

    _ecrire(exps / "exp_def" / "experience.yaml",
            {"nom": "exp_def", "jeu": {"nom": "j1"}, "decideur": {"type": "duree_minimale"}})

    monkeypatch.setattr(D, "DOSSIER", exps)
    monkeypatch.setattr(D, "DOSSIER_JEUX", tmp_path / "jeux")
    monkeypatch.setattr(D, "DOSSIER_CAMPAGNES", campagnes)
    monkeypatch.setattr(D, "lister", functools.partial(VRAI_LISTER, dossier=exps))
    monkeypatch.setattr(D, "ETAT_VUE_REGISTRE", tmp_path / "vue" / "tableau.yaml")
    # The provider is deduced from today's providers.yaml: we freeze it here.
    monkeypatch.setattr(D, "familles_par_modele", lambda: {"gemini-3.5-flash-lite": ["google"]})
    return tmp_path


def _campagne(tmp_path: Path, nom: str, restantes: list[str], *, pid=None, terminee_le=None, stop=False):
    _ecrire(tmp_path / "campagnes" / nom / "etat.json",
            {"campagne": nom, "restantes": restantes, "pid": os.getpid() if pid is None else pid,
             "terminee_le": terminee_le})
    if stop:
        (tmp_path / "campagnes" / nom / "STOP").write_text("", encoding="utf-8")


def _dessiner(st) -> None:
    with contextlib.suppress(RerunDemande):
        D._suivi_du_registre(st, pd)


def _cellule_etat(st, experience: str) -> str:
    vue = st.tables[-1]
    return str(vue.loc[vue["experience"] == experience, "etat"].iloc[0])


# ── R1 to R5 — the sheet itself ──────────────────────────────────────────────


def test_R1_la_fiche_dit_les_conditions_dans_l_ordre(plateforme):
    fiche = D.fiche_conditions(DEFINITION_LLM)
    libelles = [l for l, _ in fiche]
    assert libelles == ["décideur", "fournisseur", "prompt", "température", "réflexion", "population",
                        "jeu", "mode", "calendrier", "horizon", "mémoire", "chaîne des véhicules",
                        "parallélisme", "candidats max", "attente max", "graines"]
    valeurs = dict(fiche)
    assert valeurs["décideur"] == "gemini-3.5-flash-lite"
    assert valeurs["fournisseur"] == "google"
    assert valeurs["prompt"] == "prompt_expert_04"
    assert valeurs["température"] == "0.0"
    assert valeurs["réflexion"] == "high"
    assert valeurs["population"] == "population_1000_PANEL_v6"
    assert valeurs["mode"] == "sans simulateur"
    assert valeurs["calendrier"] == "aleatoire · 2026-03-16 · graine 42"
    assert valeurs["chaîne des véhicules"] == "active"
    assert valeurs["graines"] == "ordre 42 · tirage 42"


def test_R2_une_valeur_absente_est_une_ligne_absente():
    sans = json.loads(json.dumps(DEFINITION_LLM))
    del sans["decideur"]["parametres"]["thinking_level"]
    del sans["horizon_jours"]
    fiche = D.fiche_conditions(sans)
    libelles = [l for l, _ in fiche]
    assert "réflexion" not in libelles and "horizon" not in libelles
    assert all(v not in ("—", "", "None") for _, v in fiche), "never a `—` nor an invented default"


def test_R2_le_budget_de_reflexion_tient_lieu_de_niveau():
    avec_budget = json.loads(json.dumps(DEFINITION_LLM))
    avec_budget["decideur"]["parametres"] = {"thinking_budget": 600}
    assert dict(D.fiche_conditions(avec_budget))["réflexion"] == "budget 600"


def test_R3_le_prompt_ne_figure_que_si_le_decideur_en_lit_un():
    temoin = {"decideur": {"type": "aleatoire"}, "gabarit": {"variante": "minimal_persona"}}
    assert "prompt" not in dict(D.fiche_conditions(temoin))
    llm = {"decideur": {"type": "passerelle", "modele": "m"}, "gabarit": {"variante": "minimal_persona"}}
    assert dict(D.fiche_conditions(llm))["prompt"] == "minimal_persona"


def test_R3_le_classifieur_typé_lit_un_prompt_et_la_fiche_dit_qu_il_est_tronqué():
    """Jev reads a variant — and not the same one as the LLM arm under the same name.

    The `[Output instructions]` block is removed from it before sending (the `Choice` type
    replaces it), and the text actually served carries its own sha in the fingerprint. Two
    `prompt_expert_05` rows, one Jev the other LLM, therefore designate two texts. The
    registry column keeps the bare name so as not to split its filter; it is here, where there
    is room, that the truncation is stated.
    """
    jev = {"decideur": {"type": "typesafe", "modele": "jev-1.13.0"},
           "gabarit": {"variante": "prompt_expert_05"}}
    prompt = dict(D.fiche_conditions(jev)).get("prompt")
    assert prompt is not None, "Jev reads a prompt: the sheet cannot omit it"
    assert prompt.startswith("prompt_expert_05"), "the variant is named as is"
    assert "bloc de sortie" in prompt, "the truncation is STATED, else the sheet lies by omission"

    # Symmetric check: an LLM arm under the same variant does NOT carry the mention.
    llm = {"decideur": {"type": "passerelle", "modele": "m"},
           "gabarit": {"variante": "prompt_expert_05"}}
    assert dict(D.fiche_conditions(llm))["prompt"] == "prompt_expert_05"


def test_R4_la_fiche_d_une_execution_lit_sa_definition_figee(plateforme):
    d = D.DOSSIER / "exp_llm" / "executions" / EXEC_LLM
    fige = json.loads(json.dumps(DEFINITION_LLM))
    fige["decideur"] = {"type": "modele", "modele": "klr", "artefact": "scripts/x/klr_mode_choice_policy.json"}
    _ecrire(d / "execution.yaml", {"experience": fige})
    ligne_exec = {"experience": "exp_llm", "execution": EXEC_LLM, "dossier": str(d)}
    ligne_def = {"experience": "exp_llm", "execution": None, "dossier": str(D.DOSSIER / "exp_llm")}
    assert dict(D.fiche_de_ligne(ligne_exec))["décideur"] == "modele:klr"
    assert dict(D.fiche_de_ligne(ligne_def))["décideur"] == "gemini-3.5-flash-lite"
    assert "prompt" not in dict(D.fiche_de_ligne(ligne_exec)), "the frozen decider reads no prompt"


def test_R4_le_fournisseur_reellement_sollicite_l_emporte(plateforme):
    d = D.DOSSIER / "exp_llm" / "executions" / EXEC_LLM
    ligne = {"experience": "exp_llm", "execution": EXEC_LLM, "dossier": str(d), "fournisseur": "google · groq"}
    assert dict(D.fiche_de_ligne(ligne))["fournisseur"] == "google · groq"


def test_R5_la_ligne_et_la_table_portent_les_memes_valeurs():
    fiche = D.fiche_conditions(DEFINITION_LLM)
    ligne = D.fiche_en_ligne(fiche)
    st = FauxSt()
    D.rendre_fiche(st, fiche)
    table = st.textes[-1]
    for libelle, valeur in fiche:
        assert f"{libelle} {valeur}" in ligne
        assert f"| {libelle} | {valeur} |" in table
    assert ligne.split(" · ")[0] == "décideur gemini-3.5-flash-lite", "same order as the sheet"


# ── R6 to R8 — where the sheet is displayed ──────────────────────────────────


def test_R6_la_ligne_cliquee_montre_sa_table_de_conditions(plateforme):
    st = FauxSt(event=FauxEvent([0]))  # sorted by descending run: exp_llm first
    _dessiner(st)
    tables = [t for t in st.textes if t.startswith("| condition | valeur |")]
    assert len(tables) == 1, st.textes
    assert "| décideur | gemini-3.5-flash-lite |" in tables[0]
    assert "| prompt | prompt_expert_04 |" in tables[0]
    assert any(t.startswith("**🧾 Conditions de « exp_llm / " + EXEC_LLM) for t in st.textes)


def test_R7_le_bloc_en_cours_porte_la_ligne_des_conditions(plateforme):
    st = FauxSt()
    _dessiner(st)
    assert any(t.startswith("**⏳ exp_llm / ") for t in st.textes)
    legendes = [l for l in st.legendes if l.startswith("🧾 ")]
    assert len(legendes) == 1
    assert "décideur gemini-3.5-flash-lite" in legendes[0] and "prompt prompt_expert_04" in legendes[0]


def test_R8_les_activites_portent_la_ligne_hors_mode_compact(plateforme):
    act = D.activites_en_cours()
    assert [e["experience"] for e in act["executions"]] == ["exp_llm"]
    assert "décideur gemini-3.5-flash-lite" in act["executions"][0]["conditions"]

    st = FauxSt()
    D.rendre_activites(st, act)
    assert any(l.startswith("🧾 ") and "gemini-3.5-flash-lite" in l for l in st.legendes)

    compact = FauxSt()
    D.rendre_activites(compact, act, compact=True)
    assert not any(l.startswith("🧾 ") for l in compact.legendes), "the overview stays one line per run"


# ── R9 to R15 — the registry markers ─────────────────────────────────────────


def test_R9_ce_qui_tourne_porte_le_sablier(plateforme):
    st = FauxSt()
    _dessiner(st)
    assert _cellule_etat(st, "exp_llm") == "⏳ en_cours"
    assert _cellule_etat(st, "exp_temoin") == "terminee"


def test_R10_ce_qu_une_campagne_vivante_va_lancer_porte_le_calendrier(plateforme):
    _campagne(plateforme, "lot_x", ["exp_def"])
    st = FauxSt()
    _dessiner(st)
    assert _cellule_etat(st, "exp_def") == "📅 definie · planifiée (campagne lot_x)"
    assert _cellule_etat(st, "exp_temoin") == "terminee", "outside the remaining ones: nothing"


def test_R10_une_execution_epuisee_que_la_campagne_reprendra_est_planifiee(plateforme):
    _ecrire(D.DOSSIER / "exp_temoin" / "executions" / EXEC_TEMOIN / "etat.json", {"etat": "epuisee"})
    _campagne(plateforme, "lot_x", ["exp_temoin"])
    st = FauxSt()
    _dessiner(st)
    assert _cellule_etat(st, "exp_temoin") == "📅 epuisee · planifiée (campagne lot_x)"


@pytest.mark.parametrize("cas", ["pid_mort", "terminee", "stop"])
def test_R11_une_campagne_morte_ne_planifie_rien(plateforme, cas):
    if cas == "pid_mort":
        _campagne(plateforme, "lot_x", ["exp_def"], pid=_pid_mort())
    elif cas == "terminee":
        _campagne(plateforme, "lot_x", ["exp_def"], terminee_le="2026-09-15T06:00:00+00:00")
    else:
        _campagne(plateforme, "lot_x", ["exp_def"], stop=True)
    assert D.planifiees_par_campagne() == {}
    st = FauxSt()
    _dessiner(st)
    assert _cellule_etat(st, "exp_def") == "definie"


def test_R11_un_pid_illisible_ou_absent_vaut_mort(plateforme):
    _ecrire(plateforme / "campagnes" / "lot_x" / "etat.json", {"restantes": ["exp_def"], "pid": "n/a"})
    _ecrire(plateforme / "campagnes" / "lot_y" / "etat.json", {"restantes": ["exp_def"]})
    assert D.planifiees_par_campagne() == {}


def test_R12_les_marqueurs_ne_touchent_ni_le_filtre_ni_le_tri(plateforme):
    _campagne(plateforme, "lot_x", ["exp_def"])
    lignes = [l for l in D.lister() if not D.masquee(l)]
    avant = [dict(l) for l in lignes]
    planifiees = D.planifiees_par_campagne()
    decorees = [D.decorer_etat(l, planifiees) for l in lignes]
    assert any(c.startswith("⏳") for c in decorees) and any(c.startswith("📅") for c in decorees)
    assert lignes == avant, "decoration does not rewrite the rows"
    garde = D.appliquer_filtres(lignes, ["experience", "etat"], retenues={"etat": ["en_cours"]})
    assert [l["experience"] for l, g in zip(lignes, garde) if g] == ["exp_llm"]


def test_R13_une_definition_illisible_donne_une_fiche_vide(plateforme):
    (D.DOSSIER / "exp_def" / "experience.yaml").write_text("nom: [tronqué", encoding="utf-8")
    ligne = {"experience": "exp_def", "execution": None, "dossier": str(D.DOSSIER / "exp_def")}
    assert D.fiche_de_ligne(ligne) == []
    assert D.fiche_conditions(None) == [] and D.fiche_conditions("n'importe quoi") == []
    st = FauxSt()
    D.rendre_fiche(st, [])
    assert any("Conditions illisibles" in l for l in st.legendes)


def test_R13_une_execution_sans_dossier_retombe_sur_la_definition(plateforme):
    ligne = {"experience": "exp_llm", "execution": "2026-01-01_00_00_00",
             "dossier": str(D.DOSSIER / "exp_llm" / "executions" / "2026-01-01_00_00_00")}
    assert dict(D.fiche_de_ligne(ligne))["décideur"] == "gemini-3.5-flash-lite"


def test_R14_la_premiere_campagne_par_ordre_alphabetique_l_emporte(plateforme):
    _campagne(plateforme, "beta", ["exp_def"])
    _campagne(plateforme, "alpha", ["exp_def"])
    assert D.planifiees_par_campagne() == {"exp_def": "alpha"}


def test_R16_une_execution_terminee_n_est_jamais_planifiee(plateforme):
    """Seen on 2026-09-15: the pilot state still listed two finished controls in its `restantes`."""
    _campagne(plateforme, "lot_x", ["exp_temoin", "exp_def"])
    st = FauxSt()
    _dessiner(st)
    assert _cellule_etat(st, "exp_temoin") == "terminee"
    assert _cellule_etat(st, "exp_def").startswith("📅 ")


def test_R15_ce_qui_tourne_n_est_plus_a_venir(plateforme):
    _campagne(plateforme, "lot_x", ["exp_llm", "exp_def"])
    st = FauxSt()
    _dessiner(st)
    assert _cellule_etat(st, "exp_llm") == "⏳ en_cours"
    assert _cellule_etat(st, "exp_def").startswith("📅 ")


def test_fiche_conditions_troncature_15():
    """Checks the display of the 15% truncation in the conditions sheet."""
    exp_avec = {"troncature_15": True}
    fiche_avec = dict(D.fiche_conditions(exp_avec))
    assert fiche_avec.get("troncature 15 %") == "activée (Consideration Set)"

    exp_sans = {"troncature_15": False}
    fiche_sans = dict(D.fiche_conditions(exp_sans))
    assert fiche_sans.get("troncature 15 %") == "désactivée"

    exp_muet = {}
    fiche_muet = dict(D.fiche_conditions(exp_muet))
    assert "troncature 15 %" not in fiche_muet
