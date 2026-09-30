"""A request may designate its system prompt variant (`parameters.prompt_variant`)."""

import pytest

from mobility_llm import build_prompt_manager


@pytest.fixture(scope="module")
def pm():
    return build_prompt_manager()


def test_variante_designee_remplace_l_active(pm):
    actif = pm.get_system_prompt("itinary_multi_agent")
    minimal = pm.get_system_prompt("itinary_multi_agent", "prompt_expert_02")
    assert actif and minimal and minimal != actif and len(minimal) < len(actif)
    assert "prompt_expert_02" in pm.variantes()


def test_variante_inconnue_refusee(pm):
    with pytest.raises(ValueError, match="introuvable"):
        pm.get_system_prompt("itinary_multi_agent", "n_existe_pas")


def test_render_utilise_la_variante_de_la_requete(pm):
    from mobility_llm.persona import AgentSpec
    agent = AgentSpec(agent_id="a1", perception="p", destination="work", trajectories=[{"index": 0, "mode": "car", "description": "d"}])
    msgs_actif = pm.render("itinary_multi_agent", [agent], {})
    msgs_min = pm.render("itinary_multi_agent", [agent], {"prompt_variant": "prompt_expert_02"})
    systeme = lambda msgs: next(m.content for m in msgs if m.role == "system")
    assert systeme(msgs_min) != systeme(msgs_actif)
    assert pm.get_system_prompt("itinary_multi_agent", "prompt_expert_02").split("\n")[0] in systeme(msgs_min)


def test_prompt_minimal_rend_systeme_et_utilisateur(pm):
    """The variant's system prompt AND the user block (persona, options) go out together."""
    from mobility_llm.persona import AgentSpec
    # Perception and descriptions in English since the English switch: the fixture must look like
    # what the pipeline actually produces.
    agent = AgentSpec(agent_id="418", perception="Claire, 34, Full-Time Worker (lives alone, medium income)", destination="work",
                      trajectories=[{"index": 0, "mode": "foot", "description": "Walk 25 min"}, {"index": 1, "mode": "bus", "description": "Bus L1 12 min"}])
    msgs = pm.render("itinary_multi_agent", [agent], {"prompt_variant": "prompt_minimal_02"})
    roles = [m.role for m in msgs]
    assert roles == ["system", "user"], roles
    systeme, utilisateur = msgs[0].content, msgs[1].content
    assert "taking the persona into account" in systeme and "Do not rule out any option" in systeme
    # BOTH spellings are excluded: the English switch translated this title, but the
    # archived variants carry the French form and are re-read for the fingerprints.
    premiere = systeme.split("\n")[0]
    assert "Expected JSON schema" not in premiere and "Schéma JSON attendu" not in premiere
    assert '"probabilities"' in systeme
    assert "agent_id=418" in utilisateur and "Claire, 34" in utilisateur and "[0] foot" in utilisateur and "[1] bus" in utilisateur


# ---------------------------------------------------------------------------
# Invalidation of a variant


# The 9 `calibrated_*` were removed from the file on 2026-09-10 (temporary artefacts from
# June, no experiment referenced them). The remaining invalidated ones are these.
INVALIDES = ["prompt_minimal_01", "prompt_expert_19", "prompt_expert_03"]


@pytest.mark.parametrize("variante", INVALIDES)
def test_variante_invalidee_refusee_au_service(pm, variante):
    """Serving an invalidated prompt is refused, and the refusal names the rule broken."""
    from llm_gateway.prompts.engine import VariantePromptInvalide

    with pytest.raises(VariantePromptInvalide) as exc:
        pm.get_system_prompt("itinary_multi_agent", variante)
    assert exc.value.regle
    assert variante in str(exc.value)


def test_variante_invalidee_refusee_au_rendu(pm):
    """The refusal also holds through `render`: it is the path a request takes."""
    from llm_gateway.prompts.engine import VariantePromptInvalide

    from mobility_llm.persona import AgentSpec

    agent = AgentSpec(agent_id="a1", perception="p", destination="work",
                      trajectories=[{"index": 0, "mode": "car", "description": "d"}])
    with pytest.raises(VariantePromptInvalide):
        pm.render("itinary_multi_agent", [agent], {"prompt_variant": "prompt_minimal_01"})


def test_refus_est_une_valueerror(pm):
    """Subclass of ValueError: the callers that already filtered the variants see it."""
    from llm_gateway.prompts.engine import VariantePromptInvalide

    assert issubclass(VariantePromptInvalide, ValueError)
    with pytest.raises(ValueError):
        pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_01")


def test_empreinte_reste_calculable_sur_une_variante_invalidee(pm):
    """An invalidation must NOT make the already sealed fingerprints irreproducible."""
    texte = pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_01", verifier_validite=False)
    assert texte and "why walking does not get the highest probability" in texte


def test_variantes_valides_inchangees(pm):
    """The refusal is targeted: the other variants are served as before."""
    for variante in ("prompt_minimal_02", "prompt_expert_02", "prompt_expert_16", "prompt_expert_20"):
        assert pm.get_system_prompt("itinary_multi_agent", variante)
        assert pm.invalidation(variante) is None


def test_prompt_minimal_ne_nomme_aucun_mode(pm):
    """Minimal family: nothing that could lean towards a mode (rule M1)."""
    texte = pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_02").lower()
    for mot in ("marche", "à pied", "vélo", "voiture", "transports collectifs",
                "transports en commun", "bus", "métro", "tram", "train"):
        assert mot not in texte, f"{mot!r} appears in the minimal prompt"


# Minimal family: `prompt_minimal` is the only one IN SERVICE; `minimal_persona` declares the
# family under which it served and was judged (invalidated, hence refused). Everything else is
# expert by `familles.defaut`.
MINIMALES = {"prompt_minimal_02", "prompt_minimal_01"}


def test_familles_declarees(pm):
    """The family is declared, never guessed from the name."""
    for variante in pm.variantes():
        attendue = "minimale" if variante in MINIMALES else "experte"
        assert pm.famille(variante) == attendue, variante


def test_un_seul_minimal_en_service(pm):
    """The only servable minimal prompt is `prompt_minimal`: the other is invalidated."""
    from llm_gateway.prompts.engine import VariantePromptInvalide

    assert pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_02")
    with pytest.raises(VariantePromptInvalide):
        pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_01")


def test_mode_strict_arme_et_toutes_les_variantes_auditees(pm):
    """The safeguard only works when armed, and it is armed only once the corpus is audited."""
    assert pm.exiger_avis_neutralite is True, "strict mode must be armed by default"
    sans_avis = [v for v in pm.variantes() if pm.etat_neutralite(v)[0] == "absent"]
    assert sans_avis == [], f"variants without a neutrality review: {sans_avis}"


def test_aucun_sceau_perime(pm):
    """A review given on a text that has moved since is worth nothing: none must be."""
    perimes = [v for v in pm.variantes() if pm.etat_neutralite(v)[0] == "perime"]
    assert perimes == [], f"outdated reviews (text modified after audit): {perimes}"


def test_le_prompt_actif_est_servable(pm):
    """Cardinal non-regression: the active prompt must start the platform."""
    assert pm.get_system_prompt("itinary_multi_agent")
    pm.check_category("itinary_multi_agent")


def test_seul_prompt_minimal_est_exempt_de_mode(pm):
    """The replacement prompt differs from the old one only by instruction no. 4."""
    ancien = pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_01", verifier_validite=False)
    nouveau = pm.get_system_prompt("itinary_multi_agent", "prompt_minimal_02")
    clause = ", stating where applicable why walking does not get the highest probability"
    assert ancien.replace(clause, "") == nouveau


# ---------------------------------------------------------------------------
# Neutrality review given by the prompt-auditor agent (hygiene spec §4.2)


def _pm_sur(tmp_path, entrees: dict, actif="v"):
    """A PromptManager on a throwaway prompts.yaml, to isolate the effect of `_neutralite`."""
    import json

    import yaml
    from llm_gateway.prompts.engine import PromptManager

    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "cat.md.j2").write_text(
        "<!-- SYSTEM -->\n{{ system_prompt }}\n<!-- USER -->\nu", encoding="utf-8"
    )
    schemas = tmp_path / "s.json"
    schemas.write_text(json.dumps({"cat": {"type": "object"}}), encoding="utf-8")
    py = tmp_path / "prompts.yaml"
    py.write_text(
        yaml.safe_dump({"active": {"cat": actif}, "prompts": entrees}, allow_unicode=True),
        encoding="utf-8",
    )
    # Strict mode disarmed: these throwaway variants have no review, and that is not
    # the purpose of the test. The dedicated test checks that it is armed by default.
    return PromptManager(tpl, schemas_file=schemas, prompts_file=py,
                         exiger_avis_neutralite=False)


def _sha(texte: str) -> str:
    import hashlib

    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def test_verdict_non_conforme_refuse(tmp_path):
    from llm_gateway.prompts.engine import AvisNeutraliteManquant

    texte = "consigne"
    pm = _pm_sur(
        tmp_path,
        {
            "v": {
                "content": texte,
                "_neutralite": {
                    "verdict": "non_conforme",
                    "sha256_texte": _sha(texte),
                    "le": "2026-09-10",
                    "constats": [{"regle": "M1", "passage": "la marche"}],
                },
            }
        },
    )
    with pytest.raises(AvisNeutraliteManquant, match="non_conforme"):
        pm.get_system_prompt("cat", "v")
    assert pm.etat_neutralite("v")[0] == "refus"


def test_avis_perime_par_une_retouche_refuse(tmp_path):
    """The seal counts as much as the verdict: else one text is validated and another served."""
    from llm_gateway.prompts.engine import AvisNeutraliteManquant

    pm = _pm_sur(
        tmp_path,
        {
            "v": {
                "content": "text EDITED after the review",
                "_neutralite": {
                    "verdict": "conforme",
                    "sha256_texte": _sha("the original text"),
                    "le": "2026-09-10",
                },
            }
        },
    )
    with pytest.raises(AvisNeutraliteManquant, match="contenu a changé"):
        pm.get_system_prompt("cat", "v")
    assert pm.etat_neutralite("v")[0] == "perime"


def test_verdict_conforme_passe(tmp_path):
    texte = "consigne neutre"
    pm = _pm_sur(
        tmp_path,
        {"v": {"content": texte, "_neutralite": {"verdict": "conforme", "sha256_texte": _sha(texte)}}},
    )
    assert pm.get_system_prompt("cat", "v") == texte
    assert pm.etat_neutralite("v")[0] == "conforme"


def test_reserve_passe(tmp_path):
    texte = "consigne"
    pm = _pm_sur(
        tmp_path,
        {"v": {"content": texte, "_neutralite": {"verdict": "conforme_avec_reserve", "sha256_texte": _sha(texte)}}},
    )
    assert pm.get_system_prompt("cat", "v") == texte
    assert pm.etat_neutralite("v")[0] == "reserve"


def test_avis_absent_avertit_mais_ne_refuse_pas(tmp_path):
    """Otherwise introducing the rule would have made the 25 existing variants unusable."""
    pm = _pm_sur(tmp_path, {"v": {"content": "consigne"}})
    assert pm.get_system_prompt("cat", "v") == "consigne"
    assert pm.etat_neutralite("v")[0] == "absent"


def test_mode_strict_refuse_un_avis_absent(tmp_path):
    from llm_gateway.prompts.engine import AvisNeutraliteManquant

    pm = _pm_sur(tmp_path, {"v": {"content": "consigne"}})
    pm.exiger_avis_neutralite = True
    with pytest.raises(AvisNeutraliteManquant, match="without a valid neutrality opinion"):
        pm.get_system_prompt("cat", "v")


def test_empreinte_ignore_l_avis(tmp_path):
    """As for invalidation: a fingerprint can always be computed."""
    pm = _pm_sur(
        tmp_path,
        {"v": {"content": "t", "_neutralite": {"verdict": "non_conforme", "sha256_texte": _sha("t")}}},
    )
    assert pm.get_system_prompt("cat", "v", verifier_validite=False) == "t"


def test_les_25_variantes_du_depot_restent_servies(pm):
    """Non-regression: no valid variant of the repository must be blocked by the new rule."""
    from llm_gateway.prompts.engine import VariantePromptInvalide

    bloquees = []
    for v in pm.variantes():
        try:
            pm.get_system_prompt("itinary_multi_agent", v)
        except VariantePromptInvalide:
            pass  # the three invalidated ones, expected
        except Exception as e:  # noqa: BLE001
            bloquees.append((v, type(e).__name__))
    assert bloquees == []
