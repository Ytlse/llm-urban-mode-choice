"""The model is logged with each decision.

Request of 2026-09-20. `llm_exchanges.jsonl` already carries `provider`; the application log did not.
A decision read in `app.log` did not say which model had produced it, and two files had
to be cross-checked to establish it — on a setup where changing the reflection model
changes the memory content, hence the decisions.

The model is DERIVED from the serving instance: `providers.yaml` declares a `default_model` per
instance and the gateway does not override it per request. No new field crosses the
gateway.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urban_mobility_agents.utils.modeles import INCONNU, modele_de_instance, origine

AGENT = Path(__file__).resolve().parents[1] / "urban_mobility_agents" / "agents" / "llm_agent.py"
ENQUETES = Path(__file__).resolve().parents[1] / "urban_mobility_agents" / "enquetes.py"


def test_G3_une_instance_connue_rend_son_modele():
    """G3 — the derivation is that of `identite_run`, not a parallel table."""
    from settings import settings

    instance = next(iter(settings.llm.providers), None)
    if instance is None:
        pytest.skip("no provider declared in this environment")
    assert modele_de_instance(instance) == settings.llm.providers[instance].default_model


def test_G3b_une_instance_inconnue_rend_un_point_d_interrogation_sans_lever():
    """G3b — logging must NEVER bring down a call that succeeded."""
    assert modele_de_instance("instance_qui_n_existe_pas") == INCONNU
    assert modele_de_instance(None) == INCONNU
    assert modele_de_instance("") == INCONNU
    assert origine(None) == f"{INCONNU}/{INCONNU}"


@pytest.mark.parametrize(
    "marqueur,ce_que_c_est",
    [
        ("[decision]", "la décision d'itinéraire"),
        ("[reflexion-stm]", "la réflexion nocturne"),
        ("[auto-reflexion]", "l'auto-réflexion long terme"),
    ],
)
def test_G1_G2_chaque_appel_journalise_son_modele(marqueur, ce_que_c_est):
    """G1 and G2 — the run's three calls name the model that served them."""
    source = AGENT.read_text(encoding="utf-8")
    lignes = [l for l in source.splitlines() if marqueur in l and "logger" not in l]
    assert lignes, f"no log line for {ce_que_c_est}"
    # The line's block must carry `modele=`.
    debut = source.index(marqueur)
    bloc = source[debut : debut + 600]
    assert "modele=" in bloc, f"{ce_que_c_est} does not log its model"
    assert "origine_modele(" in bloc, (
        f"{ce_que_c_est} does not use the shared deriver: a second model table "
        f"would diverge from the first."
    )


def test_G2b_l_enquete_journalise_aussi_son_modele():
    """G2b — the run's fourth call category."""
    source = ENQUETES.read_text(encoding="utf-8")
    assert "modele_de_instance" in source
    assert '"model"' in source, "the `model` column is missing from the survey CSV"


def test_G4_une_decision_servie_par_le_cache_le_dit():
    """G4 — otherwise the cache would pass itself off as a call, and the count of saved
    calls would be made blind."""
    source = AGENT.read_text(encoding="utf-8")
    debut = source.index("[decision]")
    bloc = source[debut : debut + 600]
    assert "cache" in bloc and "direct" in bloc


def test_G5_le_derivateur_est_le_seul_endroit_ou_le_modele_se_lit():
    """G5 — two derivations written in two places diverge, like two default values."""
    arbre = ast.parse(AGENT.read_text(encoding="utf-8"))
    lectures = [
        n for n in ast.walk(arbre)
        if isinstance(n, ast.Attribute) and n.attr == "default_model"
    ]
    assert not lectures, (
        "llm_agent.py reads `default_model` directly: go through "
        "`utils.modeles.modele_de_instance`."
    )
