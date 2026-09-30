"""Severity, lifetime, retention: values and invariants.

The cases carry rule identifiers (A1, B2, C7…).

Everything here is PURE: no simulator, no model, no disk. The end-to-end
behaviour is in `test_severity_chain.py`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.gravite import (
    NIVEAUX,
    borne_0_1,
    ecart_de_rang,
    est_purgeable,
    force_apres_rappel,
    force_initiale,
    gravite_concept,
    gravite_deterministe,
    gravite_jugee,
    journal_des_composantes,
    poids_temporel,
)
from settings import settings

# The value of the delay weight, stated here so that the test keeps it in view.
POIDS_RETARD_ATTENDU = 0.50

# ── Reference scenarios ─────────────────────────────────────────────────────────
# They come from what the simulation produces: an `arrival` observation carries a delay,
# a `tc_timeout` observation reports a missed vehicle, the vehicle chain reports a constrained
# mode. (retard_s, correspondance_ratee, incident_reseau, mode_contraint, expected I_det)
#
# ⚠ TICKET 095 UPDATE (2026-09-21). The delay component no longer PLATEAUS at the
# reference delay. BELOW 30 minutes, nothing changes: the first four scenarios
# keep their values to the thousandth. ABOVE, the plateau is replaced by a rise that
# decelerates, and the line A outage — 45 minutes — goes from 0.80 to 0.94.
SCENARIOS = {
    "nominal": (0, False, False, False, 0.00),
    "petit_retard": (540, False, False, False, 0.15),
    "quart_heure": (900, False, False, False, 0.25),
    "demi_heure": (1800, False, False, False, 0.50),
    "panne_ligne_a": (2700, True, False, True, 0.9426990406),
}


def _det(nom: str) -> float:
    retard, corresp, incident, contraint, _ = SCENARIOS[nom]
    return gravite_deterministe(retard, corresp, incident, contraint)[0]


# ═══════════════════════════ A. Deterministic severity ═════════════════════════


@pytest.mark.parametrize("nom", sorted(SCENARIOS))
def test_A1_les_cinq_scenarios_de_reference(nom):
    assert _det(nom) == pytest.approx(SCENARIOS[nom][4], abs=1e-9)


def test_A1bis_la_panne_de_la_ligne_a_reste_au_dessus_du_seuil_du_choc():
    """The outage of the hysteresis experiments must clearly exceed the shock threshold.

    This is the argument supporting the choice of Θ = 0.7: "at Θ = 1.0 the shock
    studied would not have triggered". The exact value rose from 0.80 to 0.89 with the asymptotic
    component; what the test keeps is the property — it crosses the
    threshold, and with margin.
    """
    assert _det("panne_ligne_a") == pytest.approx(0.9426990406, abs=1e-9)
    assert _det("panne_ligne_a") > settings.agent.memoire__importance_choc


def test_A2_aucun_retard_n_est_confondu_avec_un_autre():
    """The plateau is removed, and that is the point of the change.

    Under the plateau, 45, 60 and 90 minutes gave EXACTLY the same severity: a decreasing shock
    profile that stayed above 30 minutes was invisible to the mechanism and existed only
    in the text. It was the first trap of any new shock declaration.
    """
    valeurs = [gravite_deterministe(m * 60)[0] for m in (30, 45, 60, 90, 120, 240)]
    assert valeurs == sorted(valeurs)
    assert len(set(valeurs)) == len(valeurs), "two different delays return the same severity"


def test_A2bis_la_composante_reste_bornee_par_son_maximum():
    """Removing the plateau does not mean letting it run: it APPROACHES 0.70 without reaching it.

    An unbounded linear component would be 1.0 from 60 minutes, total severity being
    capped at 1 — the wall would be moved, not removed, and the other three components would stop
    weighing anything at all.
    """
    maxi = float(settings.agent.memoire__retard_gravite_max)
    for minutes in (45, 60, 120):
        assert gravite_deterministe(minutes * 60)[0] < maxi
    # Beyond that, the exponential decay falls below float precision and equality
    # becomes reachable. What matters is that it is NEVER exceeded.
    for minutes in (600, 6000):
        assert gravite_deterministe(minutes * 60)[0] <= maxi


def test_A2ter_rien_ne_change_en_dessous_du_retard_de_reference():
    """The guarantee that makes the change safe: three quarters of the catalogue are under 30 min.

    A pure exponential through the anchor point would have been 1.75 times steeper at
    the origin — nine minutes would have gone from 0.15 to 0.22, and all small incidents
    would have become more severe. Nobody asked for that.
    """
    from llm import gravite as g

    for secondes in (0, 300, 540, 900, 1500, 1800):
        assert gravite_deterministe(secondes)[0] == pytest.approx(
            POIDS_RETARD_ATTENDU * min(secondes / 1800, 1.0), abs=1e-12
        ), f"{secondes} s moved below the reference delay"
    assert g.POIDS_RETARD == POIDS_RETARD_ATTENDU


def test_A2quinquies_la_pente_est_continue_au_point_d_ancrage():
    """Without this, one second more than the reference would mean a JUMP in severity."""
    avant = gravite_deterministe(1800)[0] - gravite_deterministe(1799)[0]
    apres = gravite_deterministe(1801)[0] - gravite_deterministe(1800)[0]
    assert apres == pytest.approx(avant, rel=1e-3)


def test_A2quater_le_mode_palier_reste_disponible(monkeypatch):
    """The former behaviour can be declared, to reproduce a run from before 2026-09-21."""
    from llm import gravite as g

    monkeypatch.setattr(
        settings.agent, "memoire__retard_saturation", g.MODE_RETARD_PALIER, raising=False
    )
    assert gravite_deterministe(2700)[0] == pytest.approx(0.50)
    assert gravite_deterministe(36_000)[0] == pytest.approx(0.50)


def test_A3_une_arrivee_en_avance_n_est_pas_un_bonus():
    """A negative delay is zero, never a negative severity.

    Otherwise an early arrival would OFFSET a real incident that occurred in the same entry.
    """
    assert gravite_deterministe(-600)[0] == 0.0
    assert gravite_deterministe(-600, correspondance_ratee=True)[0] == pytest.approx(0.20)


def test_A4_la_somme_est_bornee_a_un():
    valeur, _ = gravite_deterministe(36_000, True, True, True)
    assert valeur == pytest.approx(1.0, abs=1e-9)
    assert valeur <= 1.0


def test_A5_chaque_composante_pese_ce_qu_elle_annonce():
    assert gravite_deterministe(retard_s=1800)[0] == pytest.approx(0.50)
    assert gravite_deterministe(correspondance_ratee=True)[0] == pytest.approx(0.20)
    assert gravite_deterministe(incident_reseau=True)[0] == pytest.approx(0.20)
    assert gravite_deterministe(mode_contraint=True)[0] == pytest.approx(0.10)


def test_A6_le_detail_des_composantes_est_rendu():
    """Instrumentation must be able to say WHICH component played, not only how much."""
    valeur, detail = gravite_deterministe(900, True, False, True)
    assert valeur == pytest.approx(0.25 + 0.20 + 0.10)
    assert detail.retard == pytest.approx(0.25)
    assert detail.correspondance_ratee == pytest.approx(0.20)
    assert detail.incident_reseau == 0.0
    assert detail.mode_contraint == pytest.approx(0.10)
    assert set(detail.composantes_actives()) == {"retard", "correspondance_ratee", "mode_contraint"}


def test_A7_la_composante_sans_source_est_declaree_au_journal():
    """An inactive component is not a zero component, and the setup must say so.

    Zero is exactly the value of a perfect trip: without a declaration, severity would be
    underestimated without any symptom appearing.
    """
    ligne = journal_des_composantes()
    assert "INACTIVES" in ligne
    assert "incident_reseau" in ligne.split("INACTIVES")[1]
    # and the chain is in place: the parameter goes through the function
    assert gravite_deterministe(incident_reseau=True)[1].incident_reseau == pytest.approx(0.20)


def test_A8_les_autres_composantes_ne_sont_pas_repondrees():
    """No component is reweighted to "compensate" for the absence of another.

    ⚠ The sum of the four maxima is no longer 1.00 but 1.20: making the delay component
    discriminating beyond 30 minutes requires it to be able to exceed 0.50, and there is
    no way to do so while keeping the sum at 1 without lowering the other three weights, which
    would downgrade shocks of the catalogue. Accepted consequence: the CAP at 1 now bites for
    the extreme combination — very long delay AND missed connection AND constrained mode. It
    exists in no declared shock, where the maximum reached is 0.99 (C3, network outage).
    """
    assert gravite_deterministe(36_000, True, False, True)[0] == pytest.approx(1.00)
    # And silent reweighting stays forbidden: each component keeps its weight.
    _, detail = gravite_deterministe(0, True, False, True)
    assert detail.correspondance_ratee == pytest.approx(0.20)
    assert detail.mode_contraint == pytest.approx(0.10)


# ═════════════════════ B. Named level and maximum rule ═════════════════════════


def test_B1_les_cinq_echelons_ont_leurs_valeurs():
    """A concept ALONE in its level carries the full value of the step.

    It is the most common case, and it is defect no. 1 fixed on 2026-09-14: the original
    formula penalised it by 0.05.
    """
    attendu = {"anodin": 0.10, "notable": 0.30, "genant": 0.50, "grave": 0.75, "marquant": 1.00}
    for niveau, valeur in attendu.items():
        assert gravite_jugee(niveau) == pytest.approx(valeur), niveau
    assert NIVEAUX == attendu


def test_B2_le_modele_ne_peut_pas_degrader_un_fait_mesure():
    """THE test of the lot. Non-negotiable safety rule.

    A model that rates as `notable` a forty-five-minute outage with a missed connection
    and a constrained mode must not be able to downgrade it: the fact wins.
    """
    i_llm = gravite_jugee("notable")
    assert gravite_concept(i_llm, _det("panne_ligne_a")) == pytest.approx(0.9426990406, abs=1e-9)


def test_B3_quand_le_modele_est_d_accord_son_jugement_passe():
    assert gravite_concept(gravite_jugee("grave"), _det("demi_heure")) == pytest.approx(0.75)


def test_B4_la_surestimation_n_est_bornee_que_par_le_plafond():
    """DOCUMENTED behaviour, not a wished-for one: the maximum rule only protects downwards.

    This test pins the asymmetry so that nobody believes the opposite when reading the rule. The
    day a symmetric ceiling is decided, it must be rewritten knowingly.
    """
    assert gravite_concept(gravite_jugee("marquant"), _det("nominal")) == pytest.approx(1.00)


def test_B5_le_maximum_porte_sur_tout_le_groupe_consomme():
    """The worst of the group, not the last nor the mean."""
    groupe = [_det("petit_retard"), _det("panne_ligne_a"), _det("nominal")]
    assert gravite_concept(gravite_jugee("anodin"), max(groupe)) == pytest.approx(
        0.9426990406, abs=1e-9
    )


def test_B6_un_niveau_inconnu_ne_fait_pas_perdre_la_reflexion():
    """The concept falls back on the deterministic severity, and the fact is logged."""
    assert gravite_jugee("catastrophique") is None
    assert gravite_concept(gravite_jugee("catastrophique"), _det("demi_heure")) == pytest.approx(0.50)


@pytest.mark.parametrize("absent", [None, "", "   "])
def test_B7_un_niveau_absent_ne_fait_pas_perdre_la_reflexion(absent):
    assert gravite_jugee(absent) is None
    assert gravite_concept(gravite_jugee(absent), _det("quart_heure")) == pytest.approx(0.25)


def test_B8_le_departage_ne_deplace_jamais_d_un_niveau():
    """Three `genant` ranked 1, 2, 3: 0.55 / 0.50 / 0.45, and nothing crosses a step."""
    valeurs = [gravite_jugee("genant", n_niveau=3, rang=r) for r in (1, 2, 3)]
    assert valeurs == pytest.approx([0.55, 0.50, 0.45])
    assert all(NIVEAUX["notable"] < v < NIVEAUX["grave"] for v in valeurs)


def test_B9_le_departage_est_nul_quand_le_concept_est_seul():
    """Defect no. 1 fixed: nothing to break a tie on when there is only one candidate."""
    assert ecart_de_rang(1, 1) == 0.0
    assert gravite_jugee("marquant", n_niveau=1, rang=1) == pytest.approx(1.00)
    assert gravite_jugee("genant", n_niveau=1, rang=1) == pytest.approx(0.50)


def test_B10_la_gravite_jugee_ne_depasse_jamais_un():
    """Defect no. 2 fixed: a `marquant` first of three was worth 1.05.

    Its lifetime would have gone from 19.60 to 20.44 days, outside the published table.
    """
    assert gravite_jugee("marquant", n_niveau=3, rang=1) == pytest.approx(1.00)
    assert force_initiale(gravite_jugee("marquant", n_niveau=3, rang=1)) == pytest.approx(19.60)


def test_B11_les_bornes_du_depart_age_sont_plus_un_et_moins_un():
    assert ecart_de_rang(3, 1) == pytest.approx(1.0)
    assert ecart_de_rang(3, 2) == pytest.approx(0.0)
    assert ecart_de_rang(3, 3) == pytest.approx(-1.0)
    # an out-of-range rank is brought back into the interval rather than producing an aberrant gap
    assert ecart_de_rang(3, 99) == pytest.approx(-1.0)
    assert ecart_de_rang(3, 0) == pytest.approx(1.0)


def test_B12_borne_0_1():
    assert borne_0_1(-5) == 0.0
    assert borne_0_1(5) == 1.0
    assert borne_0_1(0.42) == pytest.approx(0.42)


# ════════════════ C. Lifetime, recall and reinforcement ════════════════════════

TABLE_FORCE = {0.00: 2.80, 0.10: 4.48, 0.30: 7.84, 0.50: 11.20, 0.75: 15.40, 1.00: 19.60}


@pytest.mark.parametrize(("gravite", "attendue"), sorted(TABLE_FORCE.items()))
def test_C1_table_des_durees_de_vie_initiales(gravite, attendue):
    assert force_initiale(gravite) == pytest.approx(attendue, abs=0.01)


# (severity, Δt in days, expected weight) — the table published in memory-stm-ltm.md
TABLE_POIDS = [
    (0.00, 7, 0.08), (0.00, 14, 0.007),
    (0.50, 7, 0.54), (0.50, 14, 0.29), (0.50, 30, 0.07), (0.50, 60, 0.005),
    (0.75, 7, 0.63), (0.75, 14, 0.40), (0.75, 30, 0.14), (0.75, 60, 0.02),
    (1.00, 7, 0.70), (1.00, 14, 0.49), (1.00, 30, 0.22), (1.00, 60, 0.05),
]


@pytest.mark.parametrize(("gravite", "jours", "attendu"), TABLE_POIDS)
def test_C2_table_des_poids_dans_le_temps(gravite, jours, attendu):
    """If the code departs from this table, it is the published specification that lies."""
    assert poids_temporel(jours, force_initiale(gravite)) == pytest.approx(attendu, abs=0.01)


def test_C3_le_plafond_s_applique_des_l_ecriture(monkeypatch):
    """At the published sensitivity point (S0 = 8.3 d, Park et al.), a `marquant` would go at 58 d."""
    monkeypatch.setattr(settings.agent, "long_term_retrieval__force_base_jours", 8.3)
    assert 8.3 * 7 == pytest.approx(58.1)  # what the formula would give without a ceiling
    assert force_initiale(1.00) == pytest.approx(30.0)


def test_C4_le_renforcement_est_additif_et_non_multiplicatif():
    """Ten recalls of an ordinary trip: 12.80 d. A × 1.15 factor would give 11.33 d."""
    f = force_initiale(0.0)
    for _ in range(10):
        f = force_apres_rappel(f)
    assert f == pytest.approx(12.80, abs=0.01)
    assert f != pytest.approx(2.8 * 1.15**10, abs=0.01)


def test_C5_nombre_de_rappels_jusqu_au_plafond():
    """Twenty-eight recalls for an ordinary trip, eleven for a `marquant`.

    These are the two figures on which the specification bases the choice of δ = 1 day.
    """
    def _rappels_jusqu_au_plafond(gravite: float) -> int:
        f, n = force_initiale(gravite), 0
        while f < settings.agent.memoire__force_max_jours:
            f, n = force_apres_rappel(f), n + 1
        return n

    assert _rappels_jusqu_au_plafond(0.00) == 28
    assert _rappels_jusqu_au_plafond(1.00) == 11


def test_C6_le_plafond_tient_au_renforcement():
    f = force_initiale(1.00)
    for _ in range(100):
        f = force_apres_rappel(f)
    assert f == pytest.approx(30.0)


def test_C7_une_force_absente_retombe_sur_la_constante_par_defaut():
    """An entry written before severity has no `force`: it must not be worth zero.

    A zero weight would take it out of recall without any rule having decided so.
    """
    assert poids_temporel(7, None) == pytest.approx(poids_temporel(7, 2.8))
    assert force_apres_rappel(None) == pytest.approx(3.8)


# ═══════════════════════════ E. Retention ═══════════════════════════════════════


@pytest.mark.parametrize(
    ("gravite", "jours", "purgeable"),
    [
        (0.00, 12, False),   # weight 0.0138
        (0.00, 13, True),    # weight 0.0096
        (1.00, 30, False),   # weight 0.216 — a month-old `marquant` stays
        (1.00, 90, False),   # weight 0.0101, still above the threshold
        (1.00, 95, True),    # weight 0.0079
    ],
)
def test_E1_la_purge_suit_le_poids_et_non_le_type(gravite, jours, purgeable):
    """Before severity, retention was decided on age and type alone.

    A thirty-one-day-old `marquant` memory fell along with ordinary trips.
    """
    assert est_purgeable(jours, force_initiale(gravite)) is purgeable
