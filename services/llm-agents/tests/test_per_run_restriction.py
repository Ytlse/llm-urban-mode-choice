"""A run constraint does not live in a stack-wide global file.

The two questions these tests answer:

1. **Does an experiment carry ITS routing restriction, or the shared file's?** On
   2026-09-16, a line of `config/config.yaml` set for a gemini 3.1 run made every decision
   of a gemini 3.5 run be refused — two contradictory constraints, and the router was right
   to refuse. It was a SCOPE defect, not a rule defect.
2. **Does a refusal give the right reason?** "No instance serves this model" announced right after
   listing this model among the served models, while the real reason was "the day's quota
   has not been renewed yet".

No network: the gateway is never reached, the monitor is a test double.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from experiences import archive as A
from experiences import cli as CLI
from experiences import experience as E
from tests.test_offline_execution import (  # noqa: F401 — fixtures pytest réutilisées
    _exp,
    banc,
)

RACINE = Path(__file__).resolve().parents[1]

PROVIDERS = {
    "google_gemini31_key1": {"default_model": "gemini-3.1-flash", "adapter": "google"},
    "google_gemini31_key2": {"default_model": "gemini-3.1-flash", "adapter": "google"},
    "google_gemini35_key1": {"default_model": "gemini-3.5-flash-lite", "adapter": "google"},
    "google_gemini35_key2": {"default_model": "gemini-3.5-flash-lite", "adapter": "google"},
    "lmstudio_qwen": {
        "default_model": "gemini-3.5-flash-lite",
        "adapter": "openai_compatible",
        "base_url": "http://localhost:1234/v1",
    },
}


def _moniteur(providers=None, disponibles=None, joignable=True, raison="") -> SimpleNamespace:
    return SimpleNamespace(
        providers=dict(PROVIDERS if providers is None else providers),
        joignable=joignable,
        instances_disponibles=lambda besoin=1: list(disponibles or []),
        raison_epuisement=lambda: raison,
    )


@pytest.fixture
def reglages(monkeypatch):
    """`settings.llm.instances_admises` restored after each test — it is a singleton."""
    from settings import settings

    monkeypatch.setattr(settings.llm, "instances_admises", [], raising=False)
    return settings


# ══ B. The restriction follows the experiment ════════════════════════════════════════════


def test_B1_la_valeur_du_fichier_natteint_pas_les_requetes(banc, reglages):  # noqa: F811
    """The incident, in one test: gemini 3.5 launched while the file carries the 3.1 keys."""
    reglages.llm.instances_admises = ["google_gemini31_key1", "google_gemini31_key2"]
    exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                               "portee": "distant"})

    retenue = CLI.appliquer_instances_admises(exp, _moniteur())

    assert retenue == ["google_gemini35_key1", "google_gemini35_key2"]
    assert reglages.llm.instances_admises == retenue, (
        "the restriction imposed on the PROCESS is the experiment's, not the file's"
    )
    assert "google_gemini31_key1" not in reglages.llm.instances_admises


def test_B1_le_seul_lecteur_de_la_restriction_lit_bien_les_reglages_du_processus():
    """`llm_agent.py` is the only reader: if it stopped reading the settings, B1 would be empty.

    The restriction is no longer copied onto the decision payload but given to the
    CLIENT at construction, the single gate through which STM and LTM reflections ALSO go out.

    It now comes PER CATEGORY, and the settings reader
    becomes `utils.routage`. The link guarded here is the same — from the process settings
    to the request — but it goes through this module, and the client gets the UNION as a net
    so that a call added later is restricted by default rather than free.
    """
    source = (RACINE / "urban_mobility_agents" / "agents" / "llm_agent.py").read_text(
        encoding="utf-8"
    )
    routage = (RACINE / "urban_mobility_agents" / "utils" / "routage.py").read_text(
        encoding="utf-8"
    )
    assert "settings.llm.instances_admises" in routage, (
        "`utils.routage` is the settings reader: if it stopped reading them, the "
        "restriction would be empty everywhere without any test saying so."
    )
    assert "instances_admises=toutes_les_instances()" in source, (
        "the client keeps a net: set call by call WITHOUT a net, the restriction had "
        "let memory consolidation go to another model (ticket 092)"
    )
    for categorie in ("itinary_multi_agent", "stm_reflection", "ltm_self_reflection"):
        assert f'instances_pour("{categorie}")' in source, (
            f"category {categorie} does not set its allow-list: it would fall back on the "
            f"client's net, hence on the union, hence with no routing."
        )


def test_B2_la_liste_est_celle_des_instances_servant_le_modele_filtree_par_portee(banc, reglages):  # noqa: F811
    exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                               "portee": "distant"})
    assert CLI.appliquer_instances_admises(exp, _moniteur()) == [
        "google_gemini35_key1", "google_gemini35_key2"
    ], "the `distant` scope rules out the LM Studio instance serving the same model identifier"

    exp_local = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                                     "portee": "local"})
    assert CLI.appliquer_instances_admises(exp_local, _moniteur()) == ["lmstudio_qwen"]


def test_B2_une_instance_sans_quota_reste_admise(banc, reglages):  # noqa: F811
    """The restriction is structural, arbitrating the quota belongs to the monitor.

    Deriving the list from `instances_disponibles()` would make it shrink over the run: a key
    momentarily at its cap would leave it, and never come back — the run would end up pinned to
    whatever was left at launch time.
    """
    exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                               "portee": "distant"})
    moniteur = _moniteur(disponibles=["google_gemini35_key1"])  # key2 at its cap
    assert CLI.appliquer_instances_admises(exp, moniteur) == [
        "google_gemini35_key1", "google_gemini35_key2"
    ]


def test_B3_lecrasement_dune_valeur_non_vide_est_journalise(banc, reglages, caplog):  # noqa: F811
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        reglages.llm.instances_admises = ["google_gemini31_key1", "google_gemini31_key2"]
        exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                                   "portee": "distant"})
        CLI.appliquer_instances_admises(exp, _moniteur())
    finally:
        logger.remove(sink)

    trace = "\n".join(messages)
    assert "instances_admises" in trace and "ÉCRASÉE" in trace, trace
    assert "google_gemini31_key1" in trace, "the OLD list is named"
    assert "google_gemini35_key1" in trace, "the NEW list is named"


def test_B3_aucun_journal_quand_le_fichier_est_vide(banc, reglages):  # noqa: F811
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                                   "portee": "distant"})
        CLI.appliquer_instances_admises(exp, _moniteur())
    finally:
        logger.remove(sink)
    assert not [m for m in messages if "instances_admises" in m], (
        "nothing is overwritten: the normal case must not add noise to the log"
    )


def test_B4_la_liste_retenue_entre_dans_execution_yaml(tmp_path):
    dossier = tmp_path / "exp"
    ex = A.Execution.creer(
        dossier, {"nom": "exp"}, {}, regime_demande={}, sources_alea={},
        reglages_herites={"max_trip_candidates": 6},
    )
    ex.noter_reglages_herites(instances_admises=["google_gemini35_key1"])

    sur_disque = yaml.safe_load((ex.dossier / "execution.yaml").read_text(encoding="utf-8"))
    herites = sur_disque["reglages_herites"]
    assert herites["instances_admises"] == ["google_gemini35_key1"]
    assert herites["max_trip_candidates"] == 6, "settings already recorded are not lost"

    rouverte = A.Execution.ouvrir(ex.dossier)
    assert rouverte.config["reglages_herites"]["instances_admises"] == ["google_gemini35_key1"]


def test_B4_la_reprise_reecrit_la_trace(tmp_path):
    """An archived measurement says under which restriction it was taken, not under which it
    started."""
    ex = A.Execution.creer(tmp_path / "exp", {"nom": "exp"}, {}, regime_demande={},
                           sources_alea={}, reglages_herites={})
    ex.noter_reglages_herites(instances_admises=["google_gemini35_key1"])
    A.Execution.ouvrir(ex.dossier).noter_reglages_herites(
        instances_admises=["google_gemini35_key1", "google_gemini35_key2"]
    )
    sur_disque = yaml.safe_load((ex.dossier / "execution.yaml").read_text(encoding="utf-8"))
    assert sur_disque["reglages_herites"]["instances_admises"] == [
        "google_gemini35_key1", "google_gemini35_key2"
    ]


def test_B4_cmd_lancer_consigne_la_restriction_dans_les_deux_branches():
    """Creation and resumption share one point: it is AFTER the `if derniere is not None`."""
    source = inspect.getsource(CLI.cmd_lancer)
    assert "appliquer_instances_admises(exp, moniteur)" in source
    assert "execution.noter_reglages_herites(instances_admises=" in source
    assert source.index("execution.noter_reglages_herites") > source.index(
        "execution = Execution.ouvrir(derniere)"
    ), "the call must be downstream of both branches, otherwise resumption is not recorded"


@pytest.mark.parametrize(
    "decideur",
    [
        {"type": "duree_minimale"},
        {"type": "aleatoire", "graine": 7},
        {"type": "majoritaire_voiture"},
    ],
)
def test_B5_un_decideur_hors_passerelle_nimpose_aucune_restriction(banc, reglages, decideur):  # noqa: F811
    """Including when the file carried one: the restriction follows the experiment BOTH
    WAYS. Otherwise, a local control would silently inherit another run's constraint."""
    reglages.llm.instances_admises = ["google_gemini31_key1"]
    exp = _exp(banc, decideur=decideur)
    assert CLI.appliquer_instances_admises(exp, None) == []
    assert reglages.llm.instances_admises == []


def test_B6_la_restriction_survit_au_deplacement_hors_du_yaml():
    """What is protected: emptying `config.yaml` must not DELETE the restriction.

    The original version of this test required the key to STAY in `config.yaml`, naming its
    own lifting condition: "as long as `make run` has no per-run equivalent". That
    condition has been met since 2026-09-21: `infra/docker-compose.yml`
    declares `INSTANCES_ADMISES` as a pass-through, with a default identical to what the YAML set.

    The move was NECESSARY: set in `config.yaml`, the key took precedence over
    the environment, and no experiment arm could declare its own. The campaign
    orchestrator set a variable with no effect from 2026-09-08 to 2026-09-21 without anything
    flagging it.

    What is guarded here is the invariant, not the location: the restriction exists by default,
    and the field is still read by the configuration model.
    """
    import json as _json
    import re as _re

    compose = (RACINE.parents[1] / "infra" / "docker-compose.yml").read_text(encoding="utf-8")
    m = _re.search(r"^\s*INSTANCES_ADMISES:\s*'\$\{INSTANCES_ADMISES:-(.*?)\}'\s*$", compose, _re.M)
    assert m, (
        "the restriction vanished from the pass-through: `make run` would start again with no "
        "restriction, and nothing would say so"
    )
    assert _json.loads(m.group(1)) == ["google_gemini31_key1", "google_gemini31_key2"], (
        "the default no longer reproduces what config.yaml set before the move"
    )
    from settings import settings

    assert hasattr(settings.llm, "instances_admises"), (
        "the configuration model still reads the field: without it, the pass-through is inert"
    )


def test_B6_appliquer_nécrit_rien_sur_le_disque(banc, reglages):  # noqa: F811
    chemin = RACINE / "config" / "config.yaml"
    avant = chemin.read_bytes()
    exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                               "portee": "distant"})
    CLI.appliquer_instances_admises(exp, _moniteur())
    assert chemin.read_bytes() == avant, (
        "the per-run restriction lives IN MEMORY; rewriting the shared file would reproduce "
        "exactly the defect being fixed"
    )


# ══ C. The refusal gives the right reason ════════════════════════════════════════════════


def _refus(banc, *, servantes, disponibles, raison=""):  # noqa: F811
    exp = _exp(banc, decideur={"type": "passerelle", "modele": "gemini-3.5-flash-lite",
                               "portee": "distant"})
    refus, _ = E.refuser_si_impossible(
        exp, banc["jeu"], banc["info"],
        instances_disponibles=disponibles,
        instances_servantes=servantes,
        detail_epuisement=raison,
        dependances={"commit": "abc"}, periodes={},
    )
    return refus


def test_C1_modele_servi_par_personne_message_inchange(banc):  # noqa: F811
    refus = _refus(banc, servantes=[], disponibles=[])
    assert any("aucune instance de passerelle ne sert le modèle" in r for r in refus), refus


def test_C2_modele_servi_mais_epuise_dit_lepuisement_et_les_quotas(banc):  # noqa: F811
    raison = (
        "google_gemini35_key1 : 1000/1000 requêtes/jour (épuisée) ; "
        "google_gemini35_key2 : 1000/1000 requêtes/jour (épuisée)"
    )
    refus = _refus(
        banc,
        servantes=["google_gemini35_key1", "google_gemini35_key2"],
        disponibles=[],
        raison=raison,
    )
    ligne = next(r for r in refus if "épuisées" in r)
    assert "1000/1000 requêtes/jour" in ligne, "the per-instance count is reported"
    assert "google_gemini35_key1" in ligne and "google_gemini35_key2" in ligne
    assert "renouvellement du quota" in ligne, "the refusal states the ACTION, not only the reason"


def test_C3_les_deux_motifs_sexcluent(banc):  # noqa: F811
    """The word "épuisée" cannot appear when nobody serves the model, and
    vice versa. That is the rule: one variable for two meanings made the message lie."""
    aucun = " ".join(_refus(banc, servantes=[], disponibles=[]))
    assert "épuis" not in aucun, aucun

    epuise = " ".join(
        _refus(banc, servantes=["google_gemini35_key1"], disponibles=[], raison="…")
    )
    assert "aucune instance de passerelle ne sert" not in epuise, epuise


def test_C3_rien_nest_refuse_quand_une_instance_reste_disponible(banc):  # noqa: F811
    refus = _refus(
        banc,
        servantes=["google_gemini35_key1", "google_gemini35_key2"],
        disponibles=["google_gemini35_key1"],
    )
    assert not any("épuis" in r or "ne sert le modèle" in r for r in refus), refus


def test_C4_un_appelant_qui_ne_passe_quune_liste_garde_son_message(banc):  # noqa: F811
    """Separating the two meanings is ADDITIVE: code written before this lot does not change a word.

    Those callers (archive tests, tools) do not distinguish the two sets; their single
    list stands for both at once, as before.
    """
    exp = _exp(banc, decideur={"type": "passerelle", "modele": "inconnu"})
    refus, _ = E.refuser_si_impossible(
        exp, banc["jeu"], banc["info"], instances_disponibles=[],
        dependances={"commit": "abc"}, periodes={},
    )
    assert any("aucune instance" in r for r in refus)
    assert not any("épuis" in r for r in refus)


def test_C_le_preparateur_ne_confond_plus_les_deux_ensembles():
    """The rewritten variable is the root cause: two names, and it can no longer be one."""
    source = inspect.getsource(CLI._preparer_lancement)
    assert "servantes" in source and "disponibles" in source
    assert "instances_servantes=servantes" in source
    assert "instances_disponibles=disponibles" in source


# ══ A client side — the nature of the failure ════════════════════════════════════════════


def test_A6_une_restriction_non_satisfiable_nest_ni_epuisee_ni_occupee():
    """The bucket the failure goes into says where to look for the fault. This one is not one."""
    from experiences import decideurs as D

    motif = (
        "Restriction d'instances non satisfiable : fournisseur forcé 'google_gemini35_key1' "
        "hors des instances admises ['google_gemini31_key1', 'google_gemini31_key2'] "
        "[instances admises déclarées : ['google_gemini31_key1'] ; fournisseur épinglé : "
        "google_gemini35_key1] — refus déterministe, aucun rejeu n'est planifié."
    )
    assert not D._RE_OCCUPEE.search(motif), "this is not an occupied gateway"
    assert not D._RE_QUOTA.search(motif) and not D._RE_CREDIT.search(motif), "this is not a quota"

    source = inspect.getsource(D)
    assert 'genre_erreur") == "restriction_instances"' in source
    assert '"configuration: " + erreur' in source


def test_A7_le_comportement_du_runner_face_a_ce_type_est_inchange():
    """Decision of § 9: this lot makes the reason right and fast, it does not change what the client
    does with it. `configuration` is NOT a terminal type — it falls into the waiting branch, and
    the pause for immobility does the rest, as for any blockage."""
    source = inspect.getsource(
        __import__("experiences.runner", fromlist=["runner"])
    )
    assert 'type_err == "epuise"' in source
    assert "configuration" not in source, (
        "if the runner came to know this type, Q1 of questions.md would have been settled "
        "otherwise — and this test must then be rewritten, not deleted"
    )
