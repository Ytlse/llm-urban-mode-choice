"""One model per function (decision, STM reflection, LTM reflection).

The 751 requests of the paired campaign all went to the same model
family: the plumbing already accepted a per-call allow-list, but the caller read a single
global list. Decision and STM reflection each consume 46 % of the input tokens, and
competed for the same key.

⚠ This is NOT an infrastructure setting. Changing the reflection model changes the content
of memory, hence the decisions. The binding enters the run identity and is frozen before the
campaign — otherwise the measured gap can no longer be attributed to the shock.
"""

import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import settings
from urban_mobility_agents.utils import identite_run, routage

RACINE = Path(__file__).resolve().parents[1]
AGENT = RACINE / "urban_mobility_agents" / "agents" / "llm_agent.py"


@pytest.fixture(autouse=True)
def _propre():
    routage.reinitialiser()
    yield
    routage.reinitialiser()


@pytest.fixture
def admises(monkeypatch):
    def _poser(valeur):
        monkeypatch.setattr(settings.llm, "instances_admises", valeur, raising=False)
    return _poser


@pytest.fixture
def journal_logs():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(
        lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO"
    )
    yield lignes
    logger.remove(sink)


# ══════════════════════ H — routing ═════════════════════════════════════════════


def test_H1_une_liste_plate_se_comporte_comme_avant(admises):
    """H1 — every category receives it: no existing campaign moves."""
    admises(["a", "b"])
    for categorie in ("itinary_multi_agent", "stm_reflection", "ltm_self_reflection", "autre"):
        assert routage.instances_pour(categorie) == ["a", "b"]


def test_H2_une_table_route_chaque_appel(admises):
    """H2 — the two halves of the run stop competing for the same key."""
    admises({
        "defaut": ["a", "b"],
        "stm_reflection": ["c", "d"],
        "enquete_affinite": ["c", "d"],
    })
    assert routage.instances_pour("itinary_multi_agent") == ["a", "b"]
    assert routage.instances_pour("stm_reflection") == ["c", "d"]
    assert routage.instances_pour("enquete_affinite") == ["c", "d"]


def test_H3_une_categorie_absente_retombe_sur_le_defaut(admises):
    """H3 — never on nothing: a fallback to the empty list would lift the restriction at the
    very moment one believes it tightened."""
    admises({"defaut": ["a", "b"], "stm_reflection": ["c", "d"]})
    assert routage.instances_pour("ltm_self_reflection") == ["a", "b"]
    assert routage.instances_pour(None) == ["a", "b"]


def test_H4_une_seule_instance_leve_une_alarme(admises, journal_logs):
    """H4 — without a second key, a 503 is a plain failure. The fourteen `high demand` of the
    campaign went through because one was left."""
    admises({"defaut": ["a", "b"], "stm_reflection": ["c"]})
    assert routage.instances_pour("stm_reflection") == ["c"]
    assert [m for n, m in journal_logs if n == "ERROR" and "qu'UNE instance" in m]


def test_H4b_l_alarme_ne_se_repete_pas(admises, journal_logs):
    """H4b — repeated at every call, it would drown the log it is meant to get read."""
    admises({"defaut": ["a"], "stm_reflection": ["c"]})
    for _ in range(5):
        routage.instances_pour("stm_reflection")
    assert len([m for n, m in journal_logs if "qu'UNE instance" in m]) == 1


def test_H5_le_routage_ne_produit_jamais_un_epinglage_dur():
    """H5 — `task_worker` retries ONLY if `force_provider is None`: a hard pin
    removes the fallback, that is exactly what the allow-list seeks to preserve.

    ⚠ The `force_provider` PARAMETER of `evaluate_and_choose_travel_plan` stays: it serves the
    experiment platform (`experiences/decideurs.py`), which deliberately pins an instance
    to compare models. What is forbidden is that per-function ROUTING produces one.
    """
    for fichier in (
        RACINE / "urban_mobility_agents" / "utils" / "routage.py",
        RACINE / "urban_mobility_agents" / "enquetes.py",
    ):
        arbre = ast.parse(fichier.read_text(encoding="utf-8"))
        poses = [
            n for n in ast.walk(arbre)
            if (isinstance(n, ast.keyword) and n.arg == "force_provider")
            or (isinstance(n, ast.Constant) and n.value == "force_provider")
        ]
        assert not poses, f"{fichier.name} sets force_provider: the fallback disappears."

    # In the agent, no statement mixes routing and pinning.
    for ligne in AGENT.read_text(encoding="utf-8").splitlines():
        assert not ("force_provider" in ligne and "instances_pour" in ligne), (
            "le routage par catégorie alimente un épinglage dur : " + ligne.strip()
        )


def test_H6_le_binding_entre_dans_l_identite_du_run(admises):
    """H6 — two arms with different bindings do not carry the same identity."""
    admises({"defaut": ["a", "b"], "stm_reflection": ["c", "d"]})
    ident = identite_run.composer(settings, empreinte_choc="aucun")
    assert ident["routage_instances"] == {"defaut": ["a", "b"], "stm_reflection": ["c", "d"]}
    # And the union feeds `instances_admises`, from which the admitted models derive.
    assert ident["instances_admises"] == ["a", "b", "c", "d"]


def test_H6b_une_table_ne_se_compare_pas_par_ses_cles(admises):
    """H6b — a `sorted()` on the table would return CATEGORY NAMES presented as
    instance names: an identity that no longer discriminates anything."""
    admises({"defaut": ["a", "b"]})
    ident = identite_run.composer(settings, empreinte_choc="aucun")
    assert "defaut" not in ident["instances_admises"]


def test_H6c_deux_bindings_differents_font_deux_identites(admises):
    """H6c — it is the resumption refusal that protects the comparability of the arms."""
    admises({"defaut": ["a", "b"], "stm_reflection": ["c", "d"]})
    une = identite_run.composer(settings, empreinte_choc="aucun")
    admises({"defaut": ["a", "b"], "stm_reflection": ["e", "f"]})
    autre = identite_run.composer(settings, empreinte_choc="aucun")
    ecarts = identite_run.differences(une, autre)
    assert any("routage" in e for e in ecarts)


def test_H7_une_table_vide_ne_veut_pas_dire_aucune_instance_admise(admises):
    """H7 — "no restriction" and "nothing is admitted" are not the same: one lets
    the cascade choose, the other would make every call fail."""
    admises({})
    assert routage.instances_pour("itinary_multi_agent") == []
    admises([])
    assert routage.instances_pour("itinary_multi_agent") == []


def test_H8_le_routage_en_vigueur_se_journalise(admises):
    """H8 — a binding read nowhere is a binding discovered after the campaign."""
    admises({"defaut": ["a", "b"], "stm_reflection": ["c", "d"]})
    ligne = routage.journal_du_routage()
    assert "stm_reflection=['c', 'd']" in ligne
    admises([])
    assert "aucune restriction" in routage.journal_du_routage()


# ══════════════════ I — the missing link: does the variable reach the run? ══════════════════


COMPOSE = RACINE.parents[1] / "infra" / "docker-compose.yml"
COHORTE = RACINE.parents[1] / "scripts" / "experiment" / "run_sequential_cohort.py"
CONFIG_YAML = RACINE / "config" / "config.yaml"


def test_I1_le_compose_declare_la_restriction_en_passe_plat():
    """I1 — set on the host and not declared here, the variable reaches nothing.

    Defect found on 2026-09-21 while launching the first arm: the orchestrator had set
    `LLM__INSTANCES_ADMISES` since 2026-09-08, compose did not declare it, and
    `identite_run.json` carried the value of `config.yaml`. An arm's restriction was not
    declared, it was endured.
    """
    texte = COMPOSE.read_text(encoding="utf-8")
    assert re.search(r"^\s*INSTANCES_ADMISES:\s*'\$\{INSTANCES_ADMISES:-", texte, re.M), (
        "infra/docker-compose.yml does not declare INSTANCES_ADMISES as pass-through"
    )


def test_I2_le_nom_est_le_nom_nu_du_champ():
    """I2 — `INSTANCES_ADMISES`, and above all NOT `LLM__INSTANCES_ADMISES`.

    The sub-configurations of `settings.py` are instantiated without prefix: `LLM__…` is read
    by nobody. It is the same mistake as `AGENT__…`, already made and already documented.
    """
    for fichier in (COMPOSE, COHORTE):
        actives = [
            l for l in fichier.read_text(encoding="utf-8").splitlines()
            if "LLM__INSTANCES_ADMISES" in l and not l.strip().startswith("#")
        ]
        assert not actives, (
            f"{fichier.name} pose encore LLM__INSTANCES_ADMISES, qui n'est lu par personne :\n"
            + "\n".join(actives)
        )
    assert 'env["INSTANCES_ADMISES"]' in COHORTE.read_text(encoding="utf-8")


def test_I3_le_yaml_ne_reprend_pas_la_main():
    """I3 — set in `config.yaml`, the setting TAKES PRECEDENCE over the environment.

    This is the rule of experiment settings, and the reason why the key was removed from it: a
    setting that belongs to the run is not redefined in the repo's file.
    """
    import yaml as _yaml

    charge = _yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")) or {}
    llm = charge.get("llm") or {}
    assert "instances_admises" not in llm, (
        "config.yaml sets instances_admises again: no environment variable will be able to "
        "change it any more, and the experiment arms will fall silent again."
    )


def test_I4_le_defaut_du_compose_reproduit_ce_que_le_yaml_posait():
    """I4 — removing the key from the YAML must not silently lift the restriction.

    A `make run` launched by hand must keep running on the same two instances
    as before 2026-09-21.
    """
    texte = COMPOSE.read_text(encoding="utf-8")
    m = re.search(r"^\s*INSTANCES_ADMISES:\s*'\$\{INSTANCES_ADMISES:-(.*?)\}'\s*$", texte, re.M)
    assert m, "INSTANCES_ADMISES default unreadable in the compose file"
    import json as _json

    assert _json.loads(m.group(1)) == ["google_gemini31_key1", "google_gemini31_key2"]
