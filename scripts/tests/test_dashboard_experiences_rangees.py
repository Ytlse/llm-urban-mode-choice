"""The dashboard reads and writes an experiment where it lives: in its family (2026-09-28).

Experiments are filed by family (`data/experiences/regime_nominal/<jeu>/<exp>/`). Three
places in `scripts/dashboard/experiences.py` still built the flat path
`DOSSIER / <exp>`, which does not exist for a filed experiment:

- `fiche_de_ligne`: the conditions sheet of a `definie` row came out empty;
- `executions_connues`: "These parameters are already those of X (0 run(s))", although X
  had some;
- `enregistrer`: a NEW experiment was written at the root, whereas the CLI files it under
  `regime_nominal/<sous_dossier_jeu>/<nom>`; a filed experiment, saved again, left a
  flat duplicate that `trouver_dossier_experience` found FIRST.

The location is computed by the `experiences` package (`emplacement_experience`), the very rule of
`Experience.dossier()`: a second computation would drift from the first.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from experiences.experience import Experience

from scripts.dashboard import experiences as D

VRAI_LISTER = D.lister
JEU = "pop_sans_jeu_20260316"


def _ecrire(chemin: Path, contenu: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True, sort_keys=False), encoding="utf-8")


@pytest.fixture
def exps(tmp_path, monkeypatch):
    """A pocket repository: the experiments root, empty."""
    racine = tmp_path / "data" / "experiences"
    racine.mkdir(parents=True)
    monkeypatch.setattr(D, "DOSSIER", racine)
    monkeypatch.setattr(D, "DOSSIER_JEUX", tmp_path / "data" / "jeux")
    monkeypatch.setattr(D, "DOSSIER_POP", tmp_path / "data" / "population")
    monkeypatch.setattr(D, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(D, "lister", functools.partial(VRAI_LISTER, dossier=racine))
    return racine


def _valeurs(**surcharges) -> dict:
    """The form values, as `_formulaire` returns them (no name: it is computed)."""
    base = {
        "population": "data/population/pop_sans_jeu", "jeu": "", "sans_jeu": True,
        "variante": "b_min", "decideur_type": "passerelle", "modele": "m1", "temperature": 0.0,
        "graine_decideur": 42, "rejeu_de": "", "mode": "sans_simulateur", "politique": "commune",
        "date": "2026-03-16", "graine_calendrier": 42, "horizon_jours": 1, "memoire": False,
        "graine_ordre": 42, "graine_tirage": 42, "parallelisme": 8, "max_candidats": 6,
        "attente_max_s": 120, "tolerances": dict(D.TOLERANCES_PROPOSEES), "derive_de": None,
    }
    base.update(surcharges)
    return base


def _rangee(racine: Path, exp: dict, famille: str = "jeu_x") -> Path:
    """Puts `exp` in a family — deliberately not the one its set would compute."""
    d = racine / "regime_nominal" / famille / exp["nom"]
    _ecrire(d / "experience.yaml", exp)
    return d


# ── reading ──────────────────────────────────────────────────────────────────

def test_la_fiche_d_une_ligne_definie_lit_la_definition_rangee(exps):
    exp = D.construire_experience(_valeurs())
    d = _rangee(exps, exp)
    ligne = {"experience": exp["nom"], "execution": None, "etat": "definie", "dossier": str(d)}
    fiche = D.fiche_de_ligne(ligne)
    assert fiche, "the definition exists in its family: the sheet cannot be empty"
    assert fiche == D.fiche_conditions(exp)


def test_la_fiche_d_une_archive_manquante_lit_la_definition_rangee(exps):
    """Snapshot gone: the sheet falls back on the current definition, where it lives."""
    exp = D.construire_experience(_valeurs())
    d = _rangee(exps, exp)
    ligne = {"experience": exp["nom"], "execution": "2026-09-01_10_00_00",
             "etat": D.ETAT_ARCHIVE_MANQUANTE, "dossier": str(d / "executions" / "2026-09-01_10_00_00")}
    assert D.fiche_de_ligne(ligne) == D.fiche_conditions(exp)


def test_executions_connues_compte_celles_d_une_experience_rangee(exps):
    exp = D.construire_experience(_valeurs())
    d = _rangee(exps, exp)
    for horodatage in ("2026-09-01_10_00_00", "2026-09-02_11_00_00"):
        (d / "executions" / horodatage).mkdir(parents=True)
    assert D.executions_connues(exp["nom"]) == 2


# ── writing ──────────────────────────────────────────────────────────────────

def test_une_nouvelle_experience_est_rangee_dans_la_famille_de_son_jeu(exps):
    exp = D.construire_experience(_valeurs())
    chemin, change = D.enregistrer(exp)
    assert change is True
    assert chemin == exps / "regime_nominal" / JEU / exp["nom"] / "experience.yaml"
    assert not (exps / exp["nom"]).exists(), "nothing must be written flat"


def test_une_experience_rangee_reenregistree_est_reecrite_en_place(exps):
    exp = D.construire_experience(_valeurs())
    d = _rangee(exps, exp)
    chemin, change = D.enregistrer({**exp, "max_candidats": 9})
    assert change is True
    assert chemin == d / "experience.yaml"
    assert yaml.safe_load(chemin.read_text(encoding="utf-8"))["max_candidats"] == 9
    assert not (exps / exp["nom"]).exists(), "no flat duplicate"
    assert not (exps / "regime_nominal" / JEU).exists(), "no duplicate in the computed family"
    assert len(list(exps.rglob("experience.yaml"))) == 1


def test_une_ancienne_experience_a_plat_reste_a_plat(exps):
    """Saving again moves nothing silently: its runs stay next to it."""
    exp = D.construire_experience(_valeurs())
    _ecrire(exps / exp["nom"] / "experience.yaml", exp)
    chemin, _ = D.enregistrer({**exp, "max_candidats": 9})
    assert chemin == exps / exp["nom"] / "experience.yaml"
    assert not (exps / "regime_nominal").exists()


@pytest.mark.parametrize("nom_jeu, famille", [
    ("population_1000_PANEL_v6_c2_20260316_EN_c", "jeu_1000_PANEL_v6_c2_EN_c"),
    ("population_enquete_058_20260316", "jeu_enquete_058_test"),
    ("population_1000_PANEL_v6_20260316_EN_c", "jeu_1000_PANEL_v6_EN_c"),
    (JEU, JEU),
])
def test_la_famille_est_celle_que_la_cli_choisirait(exps, nom_jeu, famille):
    exp = D.construire_experience(_valeurs(jeu=nom_jeu, sans_jeu=False))
    chemin, _ = D.enregistrer(exp)
    # The CLI method, called as is: it is the authority.
    cli = Experience.sous_dossier_jeu(SimpleNamespace(jeu=SimpleNamespace(nom=nom_jeu)))
    assert cli == famille
    assert chemin.parent == exps / "regime_nominal" / famille / exp["nom"]


def test_sans_le_paquet_l_enregistrement_refuse_au_lieu_d_ecrire_a_plat(exps, monkeypatch):
    """A flat fallback would recreate the duplicate silently: a readable refusal is better."""
    exp = D.construire_experience(_valeurs())
    monkeypatch.setitem(sys.modules, "experiences.experience", None)
    with pytest.raises(ValueError, match="location"):
        D.enregistrer(exp)
    assert not any(exps.rglob("experience.yaml")), "nothing must be written"


def test_le_bouton_enregistrer_affiche_le_refus_au_lieu_d_une_trace(exps, monkeypatch):
    """The refusal shows under the button, like the other messages; no stack trace."""
    exp = D.construire_experience(_valeurs())
    monkeypatch.setitem(sys.modules, "experiences.experience", None)
    message = D._enregistrer_et_dire(exp, inline=None, sans_validation="pas de conteneur")
    assert message.startswith("non enregistrée")
    assert "location" in message
    assert "validation" not in message, "nothing was written: there is nothing to validate"
    assert not any(exps.rglob("experience.yaml"))
