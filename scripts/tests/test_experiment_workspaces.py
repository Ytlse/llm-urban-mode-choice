"""Registry workspaces: a workspace filters a view, it never moves anything.

One test per rule, the rule number in the name. What is locked here:

  * a workspace **filters a view, it tidies nothing away**: no path of this module touches
    `data/experiences/`, and "Toutes les expériences" gives back exactly the list from before;
  * the definition file **fails open**. Absent, truncated, of an unexpected type, it leaves
    the registry standing on "Toutes les expériences". A convenience file that made the
    dashboard inaccessible would be worse than its absence;
  * a name cited **without a folder on disk is not an error**: the workspace fills up before
    the experiments, and that is precisely the intended use;
  * the shipped workspace, "papier version courte", says what the plan announces — otherwise
    the dashboard tells a plan nobody has validated.

    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_experiment_workspaces.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dashboard import espaces as E  # noqa: E402

LIVRE = REPO_ROOT / "scripts" / "dashboard" / "espaces_experiences.yaml"
ESPACE_103 = "papier version courte"


def _ecrire(tmp_path: Path, contenu: object, nom: str = "espaces.yaml") -> Path:
    p = tmp_path / nom
    p.write_text(contenu if isinstance(contenu, str) else yaml.safe_dump(contenu, allow_unicode=True),
                 encoding="utf-8")
    return p


@pytest.fixture()
def deux_espaces(tmp_path: Path) -> Path:
    return _ecrire(tmp_path, {
        "version": "espaces1",
        "espaces": [
            {"nom": "alpha", "entrees": [
                {"experience": "exp_a", "phase": "P1"},
                {"experience": "exp_b", "phase": "P1"},
                {"experience": "exp_c", "phase": "P2", "optionnel": True},
            ]},
            {"nom": "beta", "entrees": [{"experience": "exp_b"}]},
        ],
    })


# ── R1 — a workspace carries a name and an ordered list ──────────────────────────────
def test_R1_deux_espaces_lus_avec_leurs_listes_dans_l_ordre(deux_espaces):
    lus = E.espaces(deux_espaces)
    assert [e["nom"] for e in lus] == ["alpha", "beta"]
    assert [x["experience"] for x in lus[0]["entrees"]] == ["exp_a", "exp_b", "exp_c"]
    assert [x["experience"] for x in lus[1]["entrees"]] == ["exp_b"]


# ── R2 — "Toutes les expériences" at the top of the menu ─────────────────────────────
def test_R2_le_menu_commence_par_toutes_les_experiences(deux_espaces):
    assert E.noms(deux_espaces) == [E.TOUTES, "alpha", "beta"]


# ── R3 — default: nothing is filtered ────────────────────────────────────────────────
def test_R3_toutes_les_experiences_ne_filtre_rien(deux_espaces):
    lignes = [{"experience": f"exp_{c}"} for c in "abcdxyz"]
    assert E.filtrer(lignes, E.TOUTES, deux_espaces) == lignes
    assert E.filtrer(lignes, None, deux_espaces) == lignes


# ── R4 — a named workspace restricts the registry to its list ────────────────────────
def test_R4_un_espace_restreint_le_registre_a_sa_liste(deux_espaces):
    lignes = [{"experience": f"exp_{c}"} for c in "abcdxyz"]
    gardees = E.filtrer(lignes, "alpha", deux_espaces)
    assert [l["experience"] for l in gardees] == ["exp_a", "exp_b", "exp_c"]


# ── R6 — the shipped workspace is the plan's one ───────────────────────────────────
def test_R6_espace_papier_version_courte_couvre_les_quatre_phases():
    noms = E.noms(LIVRE)
    assert ESPACE_103 in noms
    par_phase: dict[str, int] = {}
    for x in E.entrees(ESPACE_103, LIVRE):
        par_phase[x["phase"]] = par_phase.get(x["phase"], 0) + 1
    assert par_phase == {
        "Phase 1 — échelle c2": 7,
        "Phase 2 — Jev sur c2": 3,
        "Phase 3 — Gemini sur c2": 2,
        "Phase 4 — graines Jev (c1)": 4,
        "Phase 5 — rejeux inter-graines (c1)": 6,
        "Optionnel — Gemini différé": 3,
        "Acquis (c1)": 14,
    }


def test_R6_espace_livre_contient_le_pivot_et_la_ligne_qui_rend_le_carre_carre():
    dedans = E.index(ESPACE_103, LIVRE)
    # The out-of-sample pivot result (Q1) and Jev's prompt served to Gemini (Q2).
    assert "exp_jev-1130_proexp32_jtir_pop-1000_PANEL_v6_c2_jeu-20260316_EN_c_nosim" in dedans
    assert "exp_gemini-35-fl_proexp32_jtir_pop-1000_PANEL_v6_c2_jeu-20260316_EN_c_t0_nosim" in dedans


# ── R6a — the phase label lives in the workspace, never in the name ──────────────────
def test_R6a_la_phase_est_une_etiquette_et_non_un_segment_du_nom():
    for x in E.entrees(ESPACE_103, LIVRE):
        assert x["phase"], f"entry without a phase: {x['experience']}"
        # The name is computed from the parameters: it must carry no phase mark.
        assert "phase" not in x["experience"].lower()
        assert "_p1_" not in x["experience"] and "_opt_" not in x["experience"]


# ── R6b — optional entries are marked and visible ────────────────────────────────────
def test_R6b_les_trois_runs_differes_sont_marques_optionnels():
    opts = [x for x in E.entrees(ESPACE_103, LIVRE) if x["optionnel"]]
    assert len(opts) == 3
    assert all("promin02" in x["experience"] for x in opts)
    # Marked, but returned: `entrees()` does not remove them.
    assert all(x in E.entrees(ESPACE_103, LIVRE) for x in opts)


# ── R7 — a workspace is a view, not a storage place ──────────────────────────────────
def test_R7_filtrer_ne_touche_a_aucune_experience_du_disque(deux_espaces):
    avant = sorted(p.name for p in (REPO_ROOT / "data" / "experiences").iterdir())
    lignes = [{"experience": "exp_a"}, {"experience": "exp_z"}]
    E.filtrer(lignes, "alpha", deux_espaces)
    E.filtrer(lignes, E.TOUTES, deux_espaces)
    assert sorted(p.name for p in (REPO_ROOT / "data" / "experiences").iterdir()) == avant
    # And going back to "Toutes" gives back the whole list, without loss.
    assert E.filtrer(lignes, E.TOUTES, deux_espaces) == lignes


# ── R8 — an experiment can belong to several workspaces ──────────────────────────────
def test_R8_une_experience_appartient_a_deux_espaces(deux_espaces):
    assert "exp_b" in E.index("alpha", deux_espaces)
    assert "exp_b" in E.index("beta", deux_espaces)


# ── R9 — a name without a folder is not an error ─────────────────────────────────────
def test_R9_les_noms_sans_dossier_sont_comptes_et_non_fatals(deux_espaces):
    manque = E.manquantes({"exp_a"}, "alpha", deux_espaces)
    assert manque == ["exp_b", "exp_c"]
    # The workspace stays perfectly readable despite the missing ones.
    assert len(E.entrees("alpha", deux_espaces)) == 3


def test_R9_l_espace_livre_peut_citer_des_experiences_pas_encore_lancees():
    presentes = {p.name for p in (REPO_ROOT / "data" / "experiences").iterdir() if p.is_dir()}
    # No completeness requirement: the test checks that the function answers, not that it is empty.
    assert isinstance(E.manquantes(presentes, ESPACE_103, LIVRE), list)


# ── R10 — no experiment disappears from "Toutes" ─────────────────────────────────────
def test_R10_une_experience_citee_par_aucun_espace_reste_dans_toutes(deux_espaces):
    lignes = [{"experience": "exp_orpheline"}]
    assert E.filtrer(lignes, E.TOUTES, deux_espaces) == lignes
    assert E.filtrer(lignes, "alpha", deux_espaces) == []


# ── R11 — an empty workspace is displayed and not confused with "Toutes" ─────────────
def test_R11_un_espace_a_liste_vide_existe_et_ne_filtre_pas_comme_toutes(tmp_path):
    p = _ecrire(tmp_path, {"espaces": [{"nom": "vide", "entrees": []}]})
    assert E.noms(p) == [E.TOUTES, "vide"]
    assert E.entrees("vide", p) == []
    assert E.filtrer([{"experience": "exp_a"}], "vide", p) == []


# ── R12 — the workspace filter composes with hiding by status ────────────────────────
def test_R12_le_filtre_par_espace_conserve_le_champ_statut(deux_espaces):
    lignes = [{"experience": "exp_a", "statut": "actif"},
              {"experience": "exp_b", "statut": "archivee"}]
    gardees = E.filtrer(lignes, "alpha", deux_espaces)
    # The workspace hides nothing itself: it returns the rows as they are, status included,
    # and it is the view that then applies `masquee()` as before.
    assert [l["statut"] for l in gardees] == ["actif", "archivee"]


# ── R13 — the file fails open ────────────────────────────────────────────────────────
def test_R13_fichier_absent(tmp_path):
    assert E.espaces(tmp_path / "rien.yaml") == []
    assert E.noms(tmp_path / "rien.yaml") == [E.TOUTES]


def test_R13_fichier_tronque_en_plein_milieu(tmp_path):
    p = _ecrire(tmp_path, "espaces:\n  - nom: alpha\n    entrees:\n      - experience: 'exp_a\n")
    assert E.espaces(p) == []
    assert E.noms(p) == [E.TOUTES]


def test_R13_contenu_de_type_inattendu(tmp_path):
    assert E.espaces(_ecrire(tmp_path, "- juste\n- une\n- liste\n")) == []
    assert E.espaces(_ecrire(tmp_path, {"espaces": "pas une liste"})) == []
    assert E.espaces(_ecrire(tmp_path, {"espaces": [{"nom": "a", "entrees": "pas une liste"}]})) \
        == [{"nom": "a", "note": None, "entrees": []}]


def test_R13_entrees_invalides_ignorees_une_a_une(tmp_path):
    p = _ecrire(tmp_path, {"espaces": [
        {"nom": "alpha", "entrees": [
            {"experience": "exp_a"},
            {"phase": "sans nom"},
            {"experience": "../../etc/passwd"},
            {"experience": "avec/slash"},
            {"experience": "ctrl\x01"},
            "exp_forme_courte",
        ]},
        {"nom": E.TOUTES, "entrees": []},          # reserved name
        {"nom": "alpha", "entrees": []},           # duplicate
        {"entrees": []},                            # without a name
    ]})
    lus = E.espaces(p)
    assert [e["nom"] for e in lus] == ["alpha"]
    assert [x["experience"] for x in lus[0]["entrees"]] == ["exp_a", "exp_forme_courte"]


# ── R14 — a remembered workspace that disappeared falls back on "Toutes" ─────────────
def test_R14_espace_memorise_disparu_retombe_sur_toutes(deux_espaces):
    assert E.actif_valide("alpha", deux_espaces) == "alpha"
    assert E.actif_valide("gamma", deux_espaces) == E.TOUTES
    assert E.actif_valide(None, deux_espaces) == E.TOUTES
    assert E.actif_valide(E.TOUTES, deux_espaces) == E.TOUTES


# ── The shipped file must stay readable by this module ───────────────────────────────
def test_le_fichier_livre_est_lisible_et_sans_entree_rejetee(caplog):
    with caplog.at_level("WARNING"):
        lus = E.espaces(LIVRE)
    assert len(lus) == 1 and lus[0]["nom"] == ESPACE_103
    assert len(lus[0]["entrees"]) == 39
    assert not [r for r in caplog.records if "[espaces]" in r.getMessage()]


# ── Rules wired into the dashboard ───────────────────────────────────────────────────
# The tests above cover the reading module; these cover its wiring in
# `scripts.dashboard.experiences`, where persistence (R5), the phase column (R6a)
# and the filter's scope (R15, R16) live.

from scripts.dashboard import experiences as X  # noqa: E402


# ── R5 — the active workspace survives a restart ─────────────────────────────────────
def test_R5_l_espace_actif_survit_au_redemarrage(tmp_path, monkeypatch, deux_espaces):
    monkeypatch.setattr(X, "ETAT_ESPACE_ACTIF", tmp_path / "espace_actif.txt")
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    assert X.charger_espace_actif() == E.TOUTES          # nothing remembered yet
    assert X.sauver_espace_actif("alpha") is True
    assert X.charger_espace_actif() == "alpha"           # read back by a fresh process
    assert X.sauver_espace_actif("alpha") is False       # nothing to rewrite


def test_R5_un_fichier_d_espace_actif_illisible_ne_casse_rien(tmp_path, monkeypatch, deux_espaces):
    dossier = tmp_path / "espace_actif.txt"
    dossier.mkdir()                                       # a folder where a file is expected
    monkeypatch.setattr(X, "ETAT_ESPACE_ACTIF", dossier)
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    assert X.charger_espace_actif() == E.TOUTES
    assert X.sauver_espace_actif("alpha") is False        # fails open, no exception


# ── R14 — on restart, a workspace gone from the file falls back on "Toutes" ──────────
def test_R14_espace_retenu_puis_supprime_du_fichier(tmp_path, monkeypatch, deux_espaces):
    monkeypatch.setattr(X, "ETAT_ESPACE_ACTIF", tmp_path / "espace_actif.txt")
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    X.sauver_espace_actif("alpha")
    deux_espaces.write_text(yaml.safe_dump({"espaces": [{"nom": "beta", "entrees": []}]}),
                            encoding="utf-8")
    assert X.charger_espace_actif() == E.TOUTES


# ── R6a — the phase column is annotated on the rows, not on the names ────────────────
def test_R6a_annoter_pose_la_phase_sur_les_lignes(monkeypatch, deux_espaces):
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    lignes = [{"experience": "exp_a"}, {"experience": "exp_c"}]
    X._annoter_espace(lignes, "alpha")
    assert lignes[0]["phase"] == "P1"
    assert lignes[1]["phase"] == "P2 · optionnel"        # R6b — the optional flag is visible


def test_R6a_sans_espace_aucune_colonne_de_phase_n_est_posee(monkeypatch, deux_espaces):
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    lignes = [{"experience": "exp_a"}]
    X._annoter_espace(lignes, E.TOUTES)
    assert "phase" not in lignes[0]


def test_R6a_la_colonne_phase_est_connue_du_registre_et_affichee_par_defaut():
    assert "phase" in X.COLONNES_REGISTRE
    assert "phase" in X.COLONNES_REGISTRE_DEFAUT


# ── R12 — workspace then status: the order of the two filters ───────────────────────
def test_R12_le_filtre_par_espace_precede_le_masquage_par_statut(monkeypatch, deux_espaces):
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    lignes = [{"experience": "exp_a", "statut": "actif"},
              {"experience": "exp_b", "statut": "archivee"},
              {"experience": "exp_hors", "statut": "actif"}]
    dans_espace = E.filtrer(lignes, "alpha", deux_espaces)
    assert [l["experience"] for l in dans_espace] == ["exp_a", "exp_b"]
    visibles = [l for l in dans_espace if not X.masquee(l)]
    assert [l["experience"] for l in visibles] == ["exp_a"]


# ── R15 — "s'inspirer de" follows the active workspace ───────────────────────────────
def test_R15_la_liste_du_formulaire_se_restreint_a_l_espace(monkeypatch, deux_espaces):
    monkeypatch.setattr(X.ESP, "FICHIER", deux_espaces)
    proposables = {"exp_a": {}, "exp_b": {}, "exp_hors": {}}
    dedans = X.ESP.index("alpha")
    restreintes = {n: e for n, e in proposables.items() if n in dedans}
    assert list(restreintes) == ["exp_a", "exp_b"]
    # And under "Toutes", the list stays whole.
    assert list(proposables) == ["exp_a", "exp_b", "exp_hors"]


# ── R16 — what is running stays visible whatever the workspace ───────────────────────
def test_R16_les_vues_d_activite_ne_consultent_jamais_l_espace_actif():
    import inspect
    for fn in (X.activites_en_cours, X.interrompues, X.rendre_activites, X.rendre_reprenables):
        source = inspect.getsource(fn)
        assert "ESP." not in source and "espace_actif" not in source, (
            f"{fn.__name__} filters by workspace: a run in progress would be lost from view "
            "when switching menus")


# ── R6a — the column appears mid-session, when the workspace brings it into being ────
def test_R6a_la_colonne_phase_entre_quand_un_espace_devient_actif():
    """Defect seen on screen on 2026-09-22: filtering worked, the column did not.

    The list of columns is frozen at the FIRST draw, under "Toutes les expériences", where
    `phase` does not exist yet. Without catching up, choosing a workspace never brings it in.
    """
    presentes = ["phase", "experience", "etat"]
    # What the session had remembered before the workspace existed: no `phase`.
    retenues = ["experience", "etat"]
    memoire = {}
    if ("phase" in presentes and "phase" not in retenues
            and "phase" not in (memoire.get("retirees") or ())):
        retenues = ["phase", *retenues]
    assert retenues[0] == "phase"


def test_R6a_une_phase_decochee_par_l_usager_ne_revient_pas():
    """Catching up must not resurrect an explicit choice of the user."""
    presentes = ["phase", "experience"]
    retenues = ["experience"]
    memoire = {"retirees": ["phase"]}
    if ("phase" in presentes and "phase" not in retenues
            and "phase" not in (memoire.get("retirees") or ())):
        retenues = ["phase", *retenues]
    assert "phase" not in retenues


def test_R6a_le_rattrapage_est_bien_pose_dans_le_code():
    """The test above reproduces the rule; this one checks that it is in the module."""
    import inspect
    source = inspect.getsource(X._panneau_colonnes_et_filtres)
    assert '"phase" in presentes' in source and '"phase" not in (memoire.get("retirees")' in source
