"""The prompt engine accepts a template name and a schema per category (per-folder layout)."""
import json

import pytest

from llm_gateway.core.models import AgentItem
from llm_gateway.prompts.engine import PromptManager


@pytest.fixture
def layout(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "template.md.j2").write_text("<!-- SYSTEM -->\nS {{ schema }}\n<!-- USER -->\n{% for agent in agents %}agent_id={{ agent.agent_id }}\n{% endfor %}", encoding="utf-8")
    (tmp_path / "a" / "output_schema.json").write_text(json.dumps({"type": "object", "title": "A"}), encoding="utf-8")
    (tmp_path / "b.md.j2").write_text("plat {{ agent_ids }}", encoding="utf-8")
    (tmp_path / "schemas.json").write_text(json.dumps({"b": {"type": "object", "title": "B"}}), encoding="utf-8")
    return tmp_path


def test_par_dossier_et_a_plat_cohabitent(layout):
    pm = PromptManager(
        templates_dir=layout, schemas_file=layout / "schemas.json",
        template_names={"a": "a/template.md.j2"}, schema_paths={"a": layout / "a" / "output_schema.json"},
    )
    pm.check_category("a")
    pm.check_category("b")
    assert pm.get_output_schema("a")["title"] == "A" and pm.get_output_schema("b")["title"] == "B"
    messages = pm.render("a", [AgentItem(agent_id="x1")], {})
    assert messages[0].role == "system" and "agent_id=x1" in messages[-1].content


def test_sans_schemas_file_le_schema_par_categorie_suffit(layout):
    pm = PromptManager(templates_dir=layout, template_names={"a": "a/template.md.j2"},
                       schema_paths={"a": layout / "a" / "output_schema.json"})
    pm.check_category("a")
    assert pm.categories == ["a"]


def test_template_manquant_nomme_le_fichier_attendu(layout):
    pm = PromptManager(templates_dir=layout, schema_paths={"c": layout / "a" / "output_schema.json"},
                       template_names={"c": "c/template.md.j2"})
    with pytest.raises(ValueError, match="c/template.md.j2"):
        pm.check_category("c")
