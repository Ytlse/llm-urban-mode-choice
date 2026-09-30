"""The agent judges what it has just experienced or read.

The model is DOUBLED: no network call. What is checked here is the grid, the refusal, the
maximum rule and the number of calls — not the quality of a judgement, which is not testable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.evenements import jugement as jg
from llm.evenements.jugement import JugementRefuse, juger, resoudre_modes, resoudre_valence
from llm.gravite import NIVEAUX

RACINE = Path(__file__).resolve().parents[1]


class _Rendu:
    def __init__(self, severity=None, valence=None, modes=None):
        self.severity = severity
        self.valence = valence
        self.modes = modes


class _Reponse:
    def __init__(self, agents):
        self.agents = agents


class _Client:
    """Doubled client. Counts its calls: that is what R29 checks."""

    def __init__(self, *rendus):
        self._rendus = list(rendus)
        self.appels = 0
        self.charges = []

    async def execute(self, payload):
        self.appels += 1
        self.charges.append(payload)
        rendu = self._rendus.pop(0) if self._rendus else None
        return _Reponse([rendu]) if rendu is not None else None


async def _juger(client, gravite=0.0):
    return await juger(
        client, "609", "Jacques Aubert, 49, works full time.", "The engine stalled twice.",
        gravite_deterministe=gravite, evenement_id="c6", jour=15,
    )


# ── R24. The refusal is clear-cut ────────────────────────────────────────────────────────
@pytest.mark.parametrize("severity", ["catastrophic", "", None, "7", "très grave"])
@pytest.mark.asyncio
async def test_R24_un_echelon_hors_grille_refuse_sans_aucun_repli(severity):
    """No replacement value: a fallback would fabricate an unjudged exposure."""
    with pytest.raises(JugementRefuse):
        await _juger(_Client(_Rendu(severity=severity, valence="negative")))


@pytest.mark.asyncio
async def test_R24bis_une_reponse_vide_est_refusee():
    with pytest.raises(JugementRefuse):
        await _juger(_Client())


# ── R25. The two vocabularies ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_R25_langlais_et_le_francais_donnent_la_meme_gravite():
    anglais = await _juger(_Client(_Rendu("serious", "negative", [])))
    francais = await _juger(_Client(_Rendu("grave", "negative", [])))
    assert anglais.importance_estimee == francais.importance_estimee == NIVEAUX["grave"]
    assert anglais.intensite == francais.intensite == "grave"


# ── R26 bis, R27. The severity is the agent's, ALONE (D7, 2026-09-22) ───────────────────
@pytest.mark.asyncio
async def test_R26bis_la_gravite_retenue_EST_lestimation_quel_que_soit_le_fait():
    """R26 is WITHDRAWN. The floor `max(estimée, mesurée)` no longer exists.

    A thirty-minute breakdown repair judged "anodin" is now worth 0.10, and not 0.70. This is
    not a loosening of the test: it is decision D7, and it is quantified — the memory will live
    2.8 days instead of 15.4.
    """
    rendu = await _juger(_Client(_Rendu("negligible", "neutral", [])), gravite=0.70)
    assert rendu.importance_estimee == pytest.approx(0.10)
    assert rendu.importance_retenue == pytest.approx(0.10)
    assert rendu.ecart_au_fait == pytest.approx(-0.60)


@pytest.mark.asyncio
async def test_un_jugement_fort_vaut_toujours_lui_meme():
    """Valid with or without a floor — but for a different reason than before."""
    rendu = await _juger(_Client(_Rendu("memorable", "negative", [])), gravite=0.70)
    assert rendu.importance_retenue == pytest.approx(1.00)
    assert rendu.ecart_au_fait == pytest.approx(+0.30)


@pytest.mark.asyncio
async def test_une_sous_estimation_de_plus_dun_echelon_leve_une_ALARME():
    """The guard that replaces the floor SAYS, it does not correct."""
    from loguru import logger as _logger

    messages: list[str] = []
    jeton = _logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        rendu = await _juger(_Client(_Rendu("negligible", "neutral", [])), gravite=0.70)
    finally:
        _logger.remove(jeton)
    assert rendu.importance_retenue == pytest.approx(0.10), "the value is not corrected"
    assert any("[ALARME]" in m and "SOUS-ESTIME" in m for m in messages)


@pytest.mark.asyncio
async def test_un_ecart_dun_seul_echelon_nalarme_pas():
    """Raising an alarm on a hesitation between two neighbouring levels would drown the log."""
    from loguru import logger as _logger

    messages: list[str] = []
    jeton = _logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await _juger(_Client(_Rendu("inconvenient", "neutral", [])), gravite=0.70)
    finally:
        _logger.remove(jeton)
    assert not [m for m in messages if "SOUS-ESTIME" in m]


@pytest.mark.asyncio
async def test_sans_fait_mesure_aucune_alarme_decart():
    """An article measures nothing: the gap has no meaning, and has no reason to shout."""
    from loguru import logger as _logger

    messages: list[str] = []
    jeton = _logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        rendu = await _juger(_Client(_Rendu("negligible", "neutral", [])), gravite=0.0)
    finally:
        _logger.remove(jeton)
    assert rendu.importance_retenue == pytest.approx(0.10)
    assert not [m for m in messages if "SOUS-ESTIME" in m]


@pytest.mark.parametrize("niveau,attendu", sorted(NIVEAUX.items()))
@pytest.mark.parametrize("mesure", [0.0, 0.70])
@pytest.mark.asyncio
async def test_R27_le_jugement_decide_seul_avec_ou_sans_fait_mesure(niveau, attendu, mesure):
    """Since D7, the presence of a measured fact no longer changes the retained severity."""
    rendu = await _juger(_Client(_Rendu(niveau, "neutral", [])), gravite=mesure)
    assert rendu.importance_retenue == pytest.approx(attendu)


# ── R29, R31. One call per exposure, and what comes out of it ────────────────────────────
@pytest.mark.asyncio
async def test_R29_un_appel_par_exposition():
    client = _Client(_Rendu("serious", "negative", ["car"]))
    await _juger(client)
    assert client.appels == 1


@pytest.mark.asyncio
async def test_R31_la_valence_et_les_modes_traversent():
    rendu = await _juger(_Client(_Rendu("serious", "positive", ["car", "cycling"])))
    assert rendu.valence == "positive"
    assert rendu.modes == ("car", "cycling")


def test_une_valence_hors_grille_retombe_sur_neutre_sans_rien_fabriquer():
    """Unlike the level: the severity holds without the valence, the default adds nothing."""
    assert resoudre_valence("enthousiaste") == "neutre"
    assert resoudre_valence(None) == "neutre"
    assert resoudre_valence("positive") == "positive"


def test_un_mode_hors_hierarchie_est_ecarte_et_non_laisse_passer():
    """The lesson of the mode vocabulary: two mode vocabularies that coexist silently."""
    assert resoudre_modes(["car", "teleportation", "cycling"]) == ("car", "cycling")
    assert resoudre_modes(["car", "car"]) == ("car",)
    assert resoudre_modes(None) == ()


# ── The anchors have ONE single source ───────────────────────────────────────────────────
def test_les_ancres_viennent_de_gravite_et_non_du_gabarit():
    """The `stm_reflection` template copies them; this one receives them.

    Two sources for the same scale would diverge the day one of them moves, and nobody would
    see it — that is exactly the pattern of the mode vocabulary.
    """
    from llm.gravite import ANCRES, ancres_anglaises

    charge = jg.charge_utile("609", "persona", "texte")
    assert charge["parameters"]["ancres"] == ancres_anglaises()
    # The TEXT of the anchors is that of `gravite.py`, whatever label carries them.
    assert sorted(charge["parameters"]["ancres"].values()) == sorted(ANCRES.values())
    gabarit = (
        RACINE.parents[1] / "packages" / "mobility_llm" / "src" / "mobility_llm"
        / "categories" / "evenement_jugement" / "template.md.j2"
    ).read_text("utf-8")
    assert "parameters.ancres" in gabarit
    for ancre in ANCRES.values():
        assert ancre not in gabarit, "an anchor copied into the template would make two sources"


def test_les_ancres_portent_le_libelle_de_l_enumeration_du_schema():
    """Measured on 2026-09-22: the prompt offered `anodin`, the schema only accepted English.

    The model received two vocabularies for the same scale and had to guess one. It is a
    defect no test saw because the two halves were each correct on their own side.
    """
    import json

    schema = json.loads((
        RACINE.parents[1] / "packages" / "mobility_llm" / "src" / "mobility_llm"
        / "categories" / "evenement_jugement" / "output_schema.json"
    ).read_text("utf-8"))
    enum = schema["properties"]["agents"]["items"]["properties"]["severity"]["enum"]
    charge = jg.charge_utile("609", "persona", "texte")
    assert sorted(charge["parameters"]["ancres"]) == sorted(enum)


def test_le_budget_de_sortie_laisse_la_place_au_raisonnement():
    """256 tokens were not enough: the models' reasoning is paid for out of this budget.

    Measured on Groq on 2026-09-22 — 240 to 330 reasoning tokens BEFORE the first character
    of JSON, and a provider that returns "max completion tokens reached before generating a valid
    document". The threshold is not a comfort setting: below it, the answer is empty or
    degraded, and a degraded answer takes the first value of each enumeration.
    """
    charge = jg.charge_utile("609", "persona", "texte")
    assert charge["parameters"]["max_tokens"] >= 512


# ── R28. The declared ablation ───────────────────────────────────────────────────────────
def test_R28_jugement_aucun_nappelle_pas_le_modele():
    """Checked by reading the source: setting up a controller would require GAMA and a model."""
    ctrl = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text("utf-8")
    assert 'registre.evenement.jugement == "a_l_injection"' in ctrl
    assert '_registre_chocs.evenement.jugement == "a_l_injection"' in ctrl


def test_une_seule_ligne_de_trace_par_exposition():
    """Otherwise every per-exposure count would be wrong.

    When a judgement is expected, the trace goes out WITH it; when it fails, it is written
    anyway, without its judgement columns. The four exit paths of the judgement
    therefore write exactly one line.
    """
    ctrl = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text("utf-8")
    debut = ctrl.index("async def _juger_evenement_subi")
    fin = ctrl.index("async def _injecter_evenements_du_reveil")
    methode = ctrl[debut:fin]
    assert methode.count("registre.tracer(") == 4, (
        "the four exits — refusal, error, late judgement, success — each write a trace"
    )
    assert "not _jugement_attendu" in ctrl, (
        "the immediate trace must be skipped when a judgement is expected"
    )


def test_un_canal_lu_sans_jugement_narme_pas_un_run():
    """An article with severity 0.00 lives 2.8 days: its silence would pass for an absence of effect."""
    from llm.gravite import force_initiale

    assert force_initiale(0.0) == pytest.approx(2.8)
