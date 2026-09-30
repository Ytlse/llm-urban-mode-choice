"""Target-memory constants and the age window.

This step changes no recall behaviour: it DECLARES the constants of the
target memory and wires the age window to the experiment horizon. These tests therefore lock
two things, and two only:

1. the published values are indeed those of the specification — a constant drifting silently
   would make any difference measured afterwards unattributable;
2. the recall score IN SERVICE has not moved — the three weights read by `rank_nodes`
   keep their values, the two new ones are read by no one. An unspecified
   intermediate state must not be executable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import settings

# ── 1. The eleven constants of § 2.10, at their published values ─────────────────

@pytest.mark.parametrize(
    ("nom", "attendu"),
    [
        ("memoire__force_k_importance", 6.0),
        ("memoire__force_delta_rappel_jours", 1.0),
        ("memoire__force_max_jours", 30.0),
        ("memoire__purge_seuil_poids", 0.01),
        ("memoire__retard_ref_s", 1800),
        ("memoire__importance_choc", 0.7),
        ("memoire__theta_gravite_cumulee", 0.7),
        ("memoire__confiance_seuil_service", 0.5),
        ("memoire__contre_exemples_seuil", 3),
        ("memoire__vivier_b_par_mode", 8),
        ("memoire__vivier_c_taille", 5),
        ("memoire__fenetre_age_max_jours", 60),
    ],
)
def test_les_constantes_publiees_sont_celles_du_ticket(nom, attendu):
    assert hasattr(settings.agent, nom), f"missing constant: {nom}"
    assert getattr(settings.agent, nom) == pytest.approx(attendu)


def test_le_seuil_de_choc_et_theta_sont_egaux():
    """Θ equals the shock threshold: a single severe memory triggers during the day.

    Decoupling them silently would make the intraday trigger independent of
    what the setup calls "a break".
    """
    assert (
        settings.agent.memoire__theta_gravite_cumulee
        == settings.agent.memoire__importance_choc
    )


def test_la_duree_de_vie_d_un_souvenir_marquant_suit_sa_regle():
    """Rule for `k`: "I will remember it in a month".

    A `marquant` memory (severity 1.0) must keep half of its weight at two
    weeks and a fifth at thirty days. This is the rule that moved k from 3 to 6;
    checking it here prevents lowering k again without noticing.
    """
    import math

    s0 = settings.agent.long_term_retrieval__force_base_jours
    k = settings.agent.memoire__force_k_importance
    force = min(s0 * (1 + k * 1.0), settings.agent.memoire__force_max_jours)

    assert math.exp(-14 / force) == pytest.approx(0.5, abs=0.05)
    assert math.exp(-30 / force) == pytest.approx(0.2, abs=0.05)


def test_le_plafond_de_force_borne_meme_un_s0_triple():
    """`FORCE_MAX` applies FROM WRITING ON, not only on reinforcement.

    At the published sensitivity point (S0 = 8.3 d, Park et al.) as at S0 × 3, the
    lifetime of a `marquant` memory would exceed the cap without this bound.
    """
    plafond = settings.agent.memoire__force_max_jours
    k = settings.agent.memoire__force_k_importance
    for s0 in (8.3, 2.8 * 3):
        assert min(s0 * (1 + k * 1.0), plafond) == pytest.approx(plafond)


# ── 2. The five weights, and the invariant "nothing moves before the five-component score" ─────────

def test_les_deux_poids_nouveaux_existent_a_leur_valeur_cible():
    assert settings.agent.long_term_retrieval__importance_weight == pytest.approx(0.20)
    assert settings.agent.long_term_retrieval__affinite_weight == pytest.approx(0.20)


def test_les_cinq_poids_ont_bascule_ensemble():
    """Since the five-component score, the five weights are read and carry their target values.

    They switched TOGETHER, and that was the condition. Components enter as ABSOLUTE
    values — min-max normalisation was removed precisely to
    make two decisions comparable: switching only part of them would have run the
    setup under a scoring regime nobody specified.

    ⚠ This test replaces `test_le_score_en_service_n_a_pas_bouge`, which locked the
    INTERMEDIATE state where the ranking read only three weights out of five.
    """
    assert settings.agent.long_term_retrieval__sim_weight == pytest.approx(0.30)
    assert settings.agent.long_term_retrieval__keyword_weight == pytest.approx(0.10)
    assert settings.agent.long_term_retrieval__time_weight == pytest.approx(0.20)


def test_les_poids_lus_par_le_classement_somment_a_un():
    """Comparability invariant, whatever the current batch.

    This test reads the weights that `rank_nodes` ACTUALLY reads, by inspecting the code rather
    than copying a list: it stays true now that the five weights
    are read and sum to 1 again.
    """
    import inspect

    from llm.longterm import MultiUserLongTermMemory

    tous = (
        "long_term_retrieval__sim_weight",
        "long_term_retrieval__keyword_weight",
        "long_term_retrieval__time_weight",
        "long_term_retrieval__importance_weight",
        "long_term_retrieval__affinite_weight",
    )
    source = inspect.getsource(MultiUserLongTermMemory.rank_nodes)
    poids_lus = [nom for nom in tous if nom in source]
    assert poids_lus, "no weight read by rank_nodes — the ranking no longer weights anything"
    total = sum(getattr(settings.agent, nom) for nom in poids_lus)
    assert total == pytest.approx(1.0), (
        f"weights read by rank_nodes: {poids_lus} → sum {total:.2f}"
    )


# ── 3. The age window follows the horizon ───────────────────────────────────────

def test_la_fenetre_d_age_vaut_l_horizon_quand_il_est_court():
    assert settings.agent.fenetre_age_pour_horizon(5) == 5


def test_la_fenetre_d_age_est_plafonnee():
    """Beyond the cap, the window no longer follows the horizon."""
    assert settings.agent.fenetre_age_pour_horizon(90) == 60
    assert settings.agent.fenetre_age_pour_horizon(60) == 60


def test_un_horizon_de_soixante_jours_n_est_plus_coupe_a_trente():
    """The historical 30-day default removed its second month from a 60-day run.

    This is the defect the age window fixes; this test fails on the previous behaviour.
    """
    assert settings.agent.fenetre_age_pour_horizon(60) > 30
    assert settings.agent.long_term_max_days_query == 60


def test_un_horizon_absurde_ne_produit_pas_une_memoire_muette():
    """A zero horizon would make the window empty, hence the memory silently inert.

    An empty memory cannot be told apart from a cut-off memory in the outputs: the case
    is brought back to one day rather than left to produce zero.
    """
    assert settings.agent.fenetre_age_pour_horizon(0) == 1
    assert settings.agent.fenetre_age_pour_horizon(-3) == 1
