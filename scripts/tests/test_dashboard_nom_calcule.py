"""The name of an experiment is computed in the form (spec `nommage-canonique-experiences`).

Follow-up of the 2026-09-07 outage: one experiment = one name = one directory = one queue entry =
one `arreter` target. As long as the name was typed in, switching model and relaunching under the
same name rewrote `experience.yaml` — three runs of "Prompt_Minimaliste" measured gemini,
gpt-oss and mistral under a single identity. With the name now DERIVED from the parameters, the
whole class of this defect disappears: these tests check that it cannot come back.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    texte = json.dumps(contenu) if chemin.suffix == ".json" else yaml.safe_dump(contenu, allow_unicode=True)
    chemin.write_text(texte, encoding="utf-8")


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    exps, jeux, pops = tmp_path / "experiences", tmp_path / "jeux", tmp_path / "population"
    _ecrire(pops / "population_1000_PANEL" / "MANIFEST.yaml", {"nom": "population_1000_PANEL"})
    _ecrire(jeux / "population_1000_PANEL_20260316" / "MANIFEST.yaml",
            {"nom": "population_1000_PANEL_20260316", "clos": True,
             "population": {"nom": "population_1000_PANEL"}, "jour_simule": "2026-03-16"})
    monkeypatch.setattr(experiences, "DOSSIER", exps)
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)
    monkeypatch.setattr(experiences, "DOSSIER_POP", pops)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)
    exps.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _valeurs(**surcharges) -> dict:
    base = {
        "population": "data/population/population_1000_PANEL",
        "jeu": "population_1000_PANEL_20260316", "sans_jeu": False,
        "variante": "minimal_persona", "decideur_type": "passerelle",
        "modele": "gemini-3.1-flash-lite-preview", "temperature": 0.0, "graine_decideur": 42,
        "rejeu_de": "", "artefact": "", "mode": "sans_simulateur", "politique": "aleatoire",
        "date": "2026-03-16", "graine_calendrier": 42, "horizon_jours": 1, "memoire": False,
        "graine_ordre": 42, "graine_tirage": 42, "parallelisme": 8, "max_candidats": 6,
        "attente_max_s": 120, "tolerances": dict(experiences.TOLERANCES_PROPOSEES),
        "derive_de": None,
    }
    base.update(surcharges)
    return base


def test_le_nom_dit_les_parametres(plateforme):
    exp = experiences.construire_experience(_valeurs())
    assert exp["nom"] == "exp_gemini-31-fl_minper_jtir_pop-1000_PANEL_t0_nosim"


def test_changer_de_modele_change_le_nom(plateforme):
    """The actual gesture of the outage: keep the form, cycle through the model selector."""
    noms = {
        experiences.construire_experience(_valeurs(modele=m))["nom"]
        for m in ("gemini-3.1-flash-lite-preview", "mistral-small-latest", "qwen/qwen3.6-27b")
    }
    assert len(noms) == 3, noms
    assert noms == {
        "exp_gemini-31-fl_minper_jtir_pop-1000_PANEL_t0_nosim",
        "exp_mistral-s_minper_jtir_pop-1000_PANEL_t0_nosim",
        "exp_qwen36-27b_minper_jtir_pop-1000_PANEL_t0_nosim",
    }


def test_changer_de_modele_n_ecrase_pas_l_experience_precedente(plateforme):
    """The core of the outage: two models can no longer share an `experience.yaml`."""
    premier = experiences.construire_experience(_valeurs(modele="mistral-small-latest"))
    experiences.enregistrer(premier)
    second = experiences.construire_experience(_valeurs(modele="qwen/qwen3.6-27b"))
    experiences.enregistrer(second)

    ecrits = {
        p.parent.name: yaml.safe_load(p.read_text(encoding="utf-8"))["decideur"]["modele"]
        for p in experiences.DOSSIER.rglob("experience.yaml")  # stored in their family
    }
    assert ecrits == {premier["nom"]: "mistral-small-latest", second["nom"]: "qwen/qwen3.6-27b"}


def test_la_temperature_et_le_mode_sont_toujours_dans_le_nom(plateforme):
    chaud = experiences.construire_experience(_valeurs(temperature=0.7))["nom"]
    gama = experiences.construire_experience(_valeurs(mode="simulateur"))["nom"]
    assert chaud.endswith("_t07_nosim") and gama.endswith("_t0_sim")


def test_une_definition_identique_est_annoncee_et_non_reecrite(plateforme):
    """N10: same parameters = same experiment. Launching adds a run, it overwrites nothing."""
    exp = experiences.construire_experience(_valeurs())
    experiences.enregistrer(exp)
    attribution, raison = experiences.nommer(exp)
    assert raison is None
    assert attribution.reutilise == exp["nom"]
    assert attribution.indice == 1

    _, change = experiences.enregistrer(experiences.construire_experience(_valeurs()))
    assert change is False, "the file already said this: nothing is rewritten"


def test_une_definition_differente_de_meme_nom_recoit_un_indice(plateforme):
    """Two definitions the grammar abbreviates alike: the second takes `_2`, like a copy."""
    exp = experiences.construire_experience(_valeurs())
    experiences.enregistrer(exp)
    # `top_p` is not part of the name: two distinct definitions, a single canonical name.
    autre = {**exp, "decideur": {**exp["decideur"], "parametres": {"temperature": 0.0, "top_p": 0.5}}}
    attribution, _ = experiences.nommer(autre)
    assert attribution.nom == exp["nom"] + "_2"
    assert attribution.reutilise is None
    assert attribution.voisins == [exp["nom"]]


def test_sans_modele_le_nom_est_impossible_et_la_raison_le_dit(plateforme):
    exp = experiences.construire_experience(_valeurs(modele=""))
    assert exp["nom"] == ""
    _attribution, raison = experiences.nommer(exp)
    assert "decideur.modele" in raison
    motifs = experiences.motifs_indisponibilite(exp, jeu_clos=True, controleur_ok=True, registre=True)
    assert any("decideur.modele" in m for m in motifs["enregistrer"]), motifs["enregistrer"]


def test_le_nom_calcule_est_toujours_un_identifiant_valide(plateforme):
    for surcharges in ({}, {"modele": "acme/foo-bar-9b-latest"}, {"temperature": 1.25},
                       {"parallelisme": 16, "memoire": True, "mode": "simulateur"}):
        nom = experiences.construire_experience(_valeurs(**surcharges))["nom"]
        assert experiences.MOTIF_NOM.match(nom), nom


def test_recopier_une_experience_ne_recopie_pas_son_nom(plateforme):
    """« S'inspirer de » can no longer offer the source's name: it will be recomputed."""
    exp = experiences.construire_experience(_valeurs())
    champs = experiences.depuis_experience({**exp, "nom": "un_nom_historique"})
    assert "nom" not in champs
    assert champs["derive_de"] == "un_nom_historique"
