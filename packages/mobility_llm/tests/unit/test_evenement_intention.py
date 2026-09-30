"""Ticket 120 — the test-only intention question: does a press article change the next 7 days?"""
import json

from mobility_llm import CATEGORIES, build_prompt_manager
from mobility_llm.persona import AgentSpec


def _rendu() -> str:
    pm = build_prompt_manager()
    agent = AgentSpec(agent_id="858909", perception="Timothée, 30, Full-Time Worker. Lives in: Toulouse",
                      evenement="Téléo en panne : une trentaine de voyageurs bloqués une heure.")
    return "\n".join(str(m.content) for m in pm.render("evenement_intention", [agent], {"temperature": 0.2}))


def test_le_gabarit_montre_le_persona_le_texte_et_l_identifiant():
    rendu = _rendu()
    assert "Timothée, 30" in rendu
    assert "Téléo en panne" in rendu
    assert "--- AGENT 858909 ---" in rendu
    assert "next 7 days" in rendu


def test_le_schema_donne_trois_niveaux_et_les_six_modes_du_jugement():
    schema = json.loads(CATEGORIES["evenement_intention"].schema_path.read_text(encoding="utf-8"))
    item = schema["properties"]["agents"]["items"]
    assert item["properties"]["change"]["enum"] == ["no_change", "some_trips", "most_trips"]
    jugement = json.loads(CATEGORIES["evenement_jugement"].schema_path.read_text(encoding="utf-8"))
    modes = jugement["properties"]["agents"]["items"]["properties"]["modes"]["items"]["enum"]
    for champ in ("use_less", "use_more"):
        assert item["properties"][champ]["items"]["enum"] == modes, champ
    assert set(item["required"]) == {"agent_id", "change", "use_less", "use_more", "reason"}
