"""Scoring of the first stage of the press protocol.

Two guards are checked here more than the
rest, because they are the ones that lie when they are missing:

- **vacuity** — an absent mode yields "non concluant", never 0.0;
- **noise** — a gap smaller than the decision-maker's floor is not an effect.

Everything is PURE: no model, no network call, no corpus required.

Run:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_press_scoring.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis.presse.grille import charger_grille  # noqa: E402
from scripts.analysis.presse.scoring import (  # noqa: E402
    EFFECTIF_MIN_PAR_MODE,
    NON_CONCLUANT,
    intervalle_groupe_par_evenement,
    accord_de_signe,
    ecart_apparie,
    kappa_pondere,
    signes_observes,
)

GRILLE = RACINE / "data" / "presse" / "grille_signes.yaml"


def _decisions(n: int, mode: str, part: float) -> dict[str, str]:
    """n trips of which `part` carry `mode`, the others a neutral mode."""
    combien = round(n * part)
    return {f"d{i}": (mode if i < combien else "autre") for i in range(n)}


# ── The paired gap ──────────────────────────────────────────────────────────────────────


def test_S1_l_ecart_se_calcule_sur_l_intersection_des_deux_conditions():
    """A trip carried by only one condition is dropped, never filled in with a default."""
    ref = _decisions(60, "tc", 0.5)
    con = dict(list(_decisions(60, "tc", 0.5).items())[:40])
    e = ecart_apparie(ref, con, article="a13", mode="tc")
    assert e.n_apparies == 40


def test_S4_un_effectif_trop_faible_rend_non_concluant_et_pas_zero():
    """In this repo, no measurement yields the perfect score. This test forbids it here."""
    petit = _decisions(EFFECTIF_MIN_PAR_MODE - 1, "tc", 0.5)
    e = ecart_apparie(petit, petit, article="a13", mode="tc")
    assert e.verdict == NON_CONCLUANT
    assert e.ecart is None, "an unmeasurable gap is not worth 0.0"


def test_l_ecart_a_le_signe_et_l_amplitude_attendus():
    ref = _decisions(100, "tc", 0.80)
    con = _decisions(100, "tc", 0.55)
    e = ecart_apparie(ref, con, article="a13", mode="tc")
    assert e.ecart == pytest.approx(-0.25)


# ── The noise floor ─────────────────────────────────────────────────────────────────────


def test_S_un_ecart_dans_le_bruit_compte_pour_zero():
    """Calling '+' a gap smaller than the noise would amount to scoring chance."""
    ref, con = _decisions(100, "tc", 0.50), _decisions(100, "tc", 0.52)
    e = ecart_apparie(ref, con, article="a13", mode="tc")
    assert signes_observes([e], plancher_de_bruit=0.032)[("a13", "tc")] == "0"
    assert signes_observes([e], plancher_de_bruit=0.005)[("a13", "tc")] == "+"


def test_S_le_plancher_de_bruit_n_a_pas_de_valeur_par_defaut():
    """A default would be forgotten — and the measured floor is not zero."""
    import inspect

    params = inspect.signature(signes_observes).parameters
    assert params["plancher_de_bruit"].default is inspect.Parameter.empty


def test_S_un_ecart_non_lisible_donne_un_signe_absent_et_non_zero():
    petit = _decisions(5, "tc", 0.5)
    e = ecart_apparie(petit, petit, article="a13", mode="tc")
    assert signes_observes([e], plancher_de_bruit=0.032)[("a13", "tc")] is None


# ── The sign agreement rate ─────────────────────────────────────────────────────────────


def test_S3_le_binomial_est_RETIRE_et_lincertitude_se_groupe_par_evenement():
    """The chapter set the bar at fifteen agreements out of twenty. It is withdrawn.

    The binomial assumed twenty independent draws. They are not: the modal shares
    of a single event sum to one, so a single behaviour mechanically produces
    several agreements. Uncertainty goes through a resampling GROUPED by event.
    """
    grille = charger_grille(GRILLE)
    parfait = {(c.article, c.mode): c.signe for c in grille.cellules}
    a = accord_de_signe(grille, parfait)
    assert a.p_binomial is None, "the field stays, empty, so that reports can be read back"
    assert a.intervalle is not None
    assert a.intervalle == (1.0, 1.0), "a perfect agreement on all articles stays perfect"


def test_S3_lintervalle_reechantillonne_les_ARTICLES_et_non_les_cellules():
    """Four cells of a single article travel as a block: that is the whole point of grouping."""
    # Five articles, four cells each. A single article disagreeing on all four.
    concordances = [
        (f"a{i}", i != 0) for i in range(5) for _ in range(4)
    ]
    bas, haut = intervalle_groupe_par_evenement(concordances)
    # Observed rate: 16/20 = 0.80. Grouped by article, the interval is [0.40; 1.00].
    assert (bas, haut) == (0.4, 1.0)

    # ⚠ THE COMPARISON THAT JUSTIFIES EVERYTHING: resampling the CELLS as if they
    # were independent — what the binomial assumed — the same set yields [0.60; 0.95].
    # The interval narrows by half without any extra data coming in. That
    # precision is manufactured by the assumption, not measured.
    import random as _r

    plats = [c for _a, c in concordances]
    alea = _r.Random(59)
    taux = sorted(
        sum(alea.choices(plats, k=len(plats))) / len(plats) for _ in range(2000)
    )
    faux_bas, faux_haut = round(taux[49], 2), round(taux[-50], 2)
    assert (faux_bas, faux_haut) == (0.6, 0.95)
    assert (haut - bas) > (faux_haut - faux_bas), (
        "grouping by event must WIDEN the interval: it gives back the precision manufactured "
        "by the independence assumption"
    )


def test_S3_lintervalle_est_reproductible_et_refuse_un_seul_evenement():
    concordances = [("a1", True), ("a1", False), ("a2", True), ("a2", True)]
    assert intervalle_groupe_par_evenement(concordances) == intervalle_groupe_par_evenement(
        concordances
    ), "fixed seed: two tallies of the same result must coincide"
    assert intervalle_groupe_par_evenement([("a1", True), ("a1", False)]) is None, (
        "an interval drawn on a single group resamples nothing"
    )


def test_S3_l_accord_porte_sur_la_grille_gelee():
    grille = charger_grille(GRILLE)
    parfait = {(c.article, c.mode): c.signe for c in grille.cellules}
    a = accord_de_signe(grille, parfait)
    assert a.concordants == 20 and a.lisibles == 20 and a.non_lisibles == 0
    assert a.taux == 1.0


def test_S3_une_cellule_non_lisible_sort_du_denominateur():
    """Counting it as a disagreement would blame the effect for the sample's fault."""
    grille = charger_grille(GRILLE)
    observes = {(c.article, c.mode): c.signe for c in grille.cellules}
    observes[("a09_vent_autan", "velo")] = None
    a = accord_de_signe(grille, observes)
    assert a.lisibles == 19 and a.non_lisibles == 1
    assert a.concordants == 19


def test_S3_predire_l_absence_d_effet_et_l_observer_est_une_concordance():
    grille = charger_grille(GRILLE)
    observes = {(c.article, c.mode): c.signe for c in grille.cellules}
    assert observes[("a18_la_machine", "marche")] == "0"
    assert accord_de_signe(grille, observes).concordants == 20


def test_S4_aucune_cellule_lisible_rend_non_concluant():
    grille = charger_grille(GRILLE)
    a = accord_de_signe(grille, {(c.article, c.mode): None for c in grille.cellules})
    assert a.verdict == NON_CONCLUANT
    assert a.taux is None, "a rate over zero cells is not worth 0.0"


# ── The weighted kappa ──────────────────────────────────────────────────────────────────


def test_le_kappa_vaut_un_sur_un_accord_parfait():
    grille = charger_grille(GRILLE)
    parfait = {(c.article, c.mode): c.intensite for c in grille.cellules}
    assert kappa_pondere(grille, parfait) == pytest.approx(1.0)


def test_le_kappa_rend_non_concluant_sous_dix_cellules():
    grille = charger_grille(GRILLE)
    observees: dict = {(c.article, c.mode): None for c in grille.cellules}
    for c in list(grille.cellules)[:8]:
        observees[(c.article, c.mode)] = c.intensite
    assert kappa_pondere(grille, observees) == NON_CONCLUANT


def test_le_kappa_rend_non_concluant_sur_une_serie_constante():
    """A kappa on a constant series is 0 by construction, and that zero says nothing."""
    grille = charger_grille(GRILLE)
    plat = {(c.article, c.mode): 2 for c in grille.cellules}
    assert kappa_pondere(grille, plat) == NON_CONCLUANT
