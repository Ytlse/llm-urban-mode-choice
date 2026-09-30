"""Each mobility category keeps its template, its schema and, if needed, its hook in its folder."""
import json

from mobility_llm import CATEGORIES, bundle
from mobility_llm.prompts import CATEGORIES_DIR


def test_chaque_categorie_est_rangee_dans_son_dossier():
    for name, spec in CATEGORIES.items():
        assert (CATEGORIES_DIR / name / "template.md.j2").is_file(), name
        schema = json.loads((CATEGORIES_DIR / name / "output_schema.json").read_text(encoding="utf-8"))
        assert schema.get("type") == "object", f"{name}: the schema is a JSON Schema object"
        assert spec.template_name == f"{name}/template.md.j2" and spec.schema_path.name == "output_schema.json"


def test_le_bundle_se_passe_de_schemas_json():
    b = bundle()
    assert b.schemas_file is None and b.templates_dir == CATEGORIES_DIR


def test_toute_categorie_qui_exige_un_agent_id_le_montre_au_modele():
    """An identifier required in the output but never given in the input is INVENTED by the model.

    Found on 2026-09-21 on the first run of the evening survey, after an hour and a half of
    simulation. The `enquete_affinite` template was the only one of the five never to render
    `agent.agent_id`. The model answered perfectly — six consistent scores, the ecology of the
    car at 3 and that of public transport at 8 — but it signed "Capucine", the persona's
    name, the only thing it could read. The gateway sorts the responses by `agent_id`:
    the twelve responses were dropped at demultiplexing, and the survey returned zero rows.

    The decision template already carries the warning — "copy its agent_id exactly as provided
    above (numeric identifier only, without the persona's name)". This test makes it mandatory
    wherever the output schema requires it, instead of relying on vigilance.
    """
    import json

    from mobility_llm import CATEGORIES

    manquants = []
    for nom, spec in CATEGORIES.items():
        if spec.schema_path is None or not spec.schema_path.is_file():
            continue
        schema = json.loads(spec.schema_path.read_text(encoding="utf-8"))
        items = (
            schema.get("properties", {}).get("agents", {}).get("items", {})
        )
        if "agent_id" not in (items.get("required") or []):
            continue
        gabarit = (CATEGORIES_DIR / spec.template_name).read_text(encoding="utf-8")
        if "agent.agent_id" not in gabarit:
            manquants.append(nom)

    assert not manquants, (
        "these categories require `agent_id` in the output without ever showing it to the "
        "model: "
        + ", ".join(sorted(manquants))
        + " — their responses will be dropped at demultiplexing."
    )
