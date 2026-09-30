"""Tests for copying experiment parameters and preserving the reasoning depth.

Checks that:
1. The 6 key parameters (set, prompt, decision-maker, model, temperature, depth)
   are faithfully extracted and validated during a copy (« S'inspirer de », « Dupliquer »).
2. The reasoning depth (numeric budget or declarative level) is preserved
   on reading, in the base validation, in the form fingerprint and when building
   the experiment (for gateway and antigravity).
3. The secondary parameters (seeds, mode, calendar, parallelism, tolerances)
   are all carried over in full.
"""

from pathlib import Path
import pytest
import yaml

from scripts.dashboard import experiences


@pytest.fixture
def base_exp():
    return {
        "nom": "exp_gemini38_test",
        "population": {"chemin": "data/population/population_1000_PANEL_v5"},
        "jeu": {"nom": "population_1000_PANEL_v5_20260316"},
        "gabarit": {"categorie": "itinary_multi_agent", "variante": "expert_gem_3.8_v3"},
        "decideur": {
            "type": "passerelle",
            "portee": "distant",
            "modele": "gemini-3.5-flash-lite",
            "parametres": {
                "temperature": 0.7,
                "thinking_level": "high",
                "top_p": 1.0,
                "max_tokens": 4096,
            },
        },
        "mode": "sans_simulateur",
        "calendrier": {
            "politique": "commune",
            "date": "2026-03-16",
            "graine": 123,
        },
        "horizon_jours": 1,
        "memoire": False,
        "graine_ordre": 456,
        "graine_tirage": 789,
        "regroupement": {"parallelisme": 16},
        "tolerances_horaires": {"walk": "insensible", "car": "heure"},
        "max_candidats": 8,
        "attente_max_s": 90,
        "derive_de": None,
    }


def test_defauts_reflexion_initiale():
    """defauts() must initialise the reasoning fields to None without breaking the model."""
    d = experiences.defauts()
    assert "reflexion" in d
    assert "niveau_reflexion" in d
    assert d["reflexion"] is None
    assert d["niveau_reflexion"] is None
    assert d["modele"] != "", "a usable remote model must be proposed by default"


def test_depuis_experience_copie_profondeur_niveau(base_exp):
    """depuis_experience() and _valider_base() must extract thinking_level and all the key parameters."""
    recopie = experiences.depuis_experience(base_exp)

    # 6 key parameters
    assert recopie["jeu"] == "population_1000_PANEL_v5_20260316"
    assert recopie["variante"] == "expert_gem_3.8_v3"
    assert recopie["decideur_type"] == "passerelle_distant"
    assert recopie["modele"] == "gemini-3.5-flash-lite"
    assert recopie["temperature"] == 0.7
    assert recopie["niveau_reflexion"] == "high"
    assert recopie["reflexion"] is None

    # Secondary parameters
    assert recopie["politique"] == "commune"
    assert recopie["date"] == "2026-03-16"
    assert recopie["graine_calendrier"] == 123
    assert recopie["graine_ordre"] == 456
    assert recopie["graine_tirage"] == 789
    assert recopie["parallelisme"] == 16
    assert recopie["max_candidats"] == 8
    assert recopie["attente_max_s"] == 90
    assert recopie["tolerances"] == {"walk": "insensible", "car": "heure"}
    assert recopie["derive_de"] == "exp_gemini38_test"

    valide = experiences._valider_base(recopie)
    assert valide["niveau_reflexion"] == "high"
    assert valide["temperature"] == 0.7
    assert valide["modele"] == "gemini-3.5-flash-lite"


def test_depuis_experience_copie_profondeur_budget(base_exp):
    """depuis_experience() and _valider_base() must extract thinking_budget."""
    base_exp["decideur"]["parametres"] = {
        "temperature": 0.3,
        "thinking_budget": 2048,
        "top_p": 1.0,
    }
    recopie = experiences.depuis_experience(base_exp)
    assert recopie["reflexion"] == 2048
    assert recopie["niveau_reflexion"] is None
    assert recopie["temperature"] == 0.3

    valide = experiences._valider_base(recopie)
    assert valide["reflexion"] == 2048
    assert valide["niveau_reflexion"] is None
    assert valide["temperature"] == 0.3


def test_depuis_experience_antigravity(base_exp):
    """For an antigravity decision-maker, the model, temperature and reasoning are preserved."""
    base_exp["decideur"] = {
        "type": "antigravity",
        "modele": "gemini-3.8-flash",
        "parametres": {
            "temperature": 0.5,
            "thinking_level": "medium",
        },
    }
    recopie = experiences.depuis_experience(base_exp)
    assert recopie["decideur_type"] == "antigravity"
    assert recopie["modele"] == "gemini-3.8-flash"
    assert recopie["temperature"] == 0.5
    assert recopie["niveau_reflexion"] == "medium"

    valide = experiences._valider_base(recopie)
    assert valide["decideur_type"] == "antigravity"
    assert valide["modele"] == "gemini-3.8-flash"
    assert valide["temperature"] == 0.5
    assert valide["niveau_reflexion"] == "medium"


def test_construire_experience_reflexion():
    """construire_experience() must write thinking_level or thinking_budget for gateway and antigravity."""
    v_niveau = experiences.defauts()
    v_niveau.update({
        "decideur_type": "passerelle_distant",
        "modele": "gemini-3.8-flash",
        "temperature": 0.4,
        "niveau_reflexion": "high",
        "reflexion": None,
    })
    exp_niveau = experiences.construire_experience(v_niveau)
    params_niveau = exp_niveau["decideur"]["parametres"]
    assert params_niveau["thinking_level"] == "high"
    assert "thinking_budget" not in params_niveau
    assert params_niveau["temperature"] == 0.4

    # Antigravity with budget
    v_budget = experiences.defauts()
    v_budget.update({
        "decideur_type": "antigravity",
        "modele": "gemini-3.8-flash",
        "temperature": 0.0,
        "niveau_reflexion": None,
        "reflexion": 4096,
    })
    exp_budget = experiences.construire_experience(v_budget)
    params_budget = exp_budget["decideur"]["parametres"]
    assert params_budget["thinking_budget"] == 4096
    assert "thinking_level" not in params_budget
    assert exp_budget["decideur"]["type"] == "antigravity"

    # No reasoning requested: neither thinking_level nor thinking_budget must be present
    v_sans = experiences.defauts()
    v_sans.update({
        "decideur_type": "passerelle_distant",
        "modele": "gemini-3.8-flash",
        "niveau_reflexion": None,
        "reflexion": None,
    })
    exp_sans = experiences.construire_experience(v_sans)
    assert "thinking_level" not in exp_sans["decideur"]["parametres"]
    assert "thinking_budget" not in exp_sans["decideur"]["parametres"]


def test_empreinte_formulaire_retient_reflexion():
    """The form fingerprint (draft) must preserve the reasoning keys."""
    v = experiences.defauts()
    v["niveau_reflexion"] = "low"
    v["reflexion"] = None
    v["temperature"] = 0.2

    texte = experiences._empreinte_formulaire(v)
    brut = yaml.safe_load(texte)
    assert brut["niveau_reflexion"] == "low"
    assert brut["temperature"] == 0.2
    assert "reflexion" in brut


def test_aller_retour_experience_reelle_disque():
    """Checks the full cycle on a real on-disk experiment carrying thinking_level."""
    chemin = Path("data/experiences/exp_gemini-35-fl_expgem38v2_jtir_pop-1000_PANEL_v5_t0_nosim/experience.yaml")
    if not chemin.is_file():
        pytest.skip("real experiment missing from disk")
    exp_source = yaml.safe_load(chemin.read_text(encoding="utf-8"))

    formulaire = experiences.depuis_experience(exp_source)
    valide = experiences._valider_base(formulaire)

    assert valide["jeu"] == exp_source["jeu"]["nom"]
    assert valide["variante"] == exp_source["gabarit"]["variante"]
    assert valide["decideur_type"] == "passerelle_distant"
    assert valide["modele"] == exp_source["decideur"]["modele"]
    assert valide["temperature"] == exp_source["decideur"]["parametres"]["temperature"]
    assert valide["niveau_reflexion"] == exp_source["decideur"]["parametres"]["thinking_level"]

    reconstruite = experiences.construire_experience(valide)
    assert reconstruite["decideur"]["parametres"]["thinking_level"] == "high"
    assert reconstruite["decideur"]["modele"] == "gemini-3.5-flash-lite"
    assert reconstruite["decideur"]["parametres"]["temperature"] == 0.0

