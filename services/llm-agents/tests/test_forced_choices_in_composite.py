"""A single-mode offer is not a decision: forced choices are measured systematically.

Rules R10-R12 had already made `synthese.json` publish the two readings of the MODAL
SHARES. What remained was the quantity that decides the ranking of the arms: the
**composite score**. It counted single-itinerary decisions without saying so, and the count
accompanied neither the figure nor the table that compares the arms.

What these tests lock in is that the measurement happens **on its own**: no option,
no command to remember. Nothing here settles the convention — the main composite keeps
counting these rows, the second reading removes them, and both are published side by side.

Why the measurement is mandatory rather than optional, in figures from 2026-09-12 (16
runs of 11/09, same population, same set): removing the single-itinerary decisions
shifts the EMD composite by **−3.75** (`durmin`) to **+12.22** (`alea`), and **changes the
ranking** — `lgbm` (4.50) is first with all decisions included, but moves to 10.46 and
behind `klr` (9.95) without single itineraries. Chain off, the same gap tops out at 0.32:
these rows come from the vehicle chain, hence from the arm's own earlier choices.
"""

import json
import shutil
from pathlib import Path

import pytest
from experiences import formule as F
from experiences import score as S

REPO = Path(__file__).resolve().parents[3]


def _exec_reelle() -> Path | None:
    """A finished run of the repo that carries forced choices in its scored scope.

    When hard-coded, the reference would disappear with the experiment that carries it. Forced
    choices are required: on a run that has none, the two readings coincide and all
    the tests would pass without measuring anything — vacuity taken for success, a recurring
    pattern in this repo.
    """
    racine = REPO / "data" / "experiences"
    for exp in sorted(racine.iterdir()) if racine.is_dir() else []:
        executions = exp / "executions"
        for d in sorted(executions.iterdir()) if executions.is_dir() else []:
            if not (d / "moves.csv").exists() or not (d / "synthese.json").exists():
                continue
            try:
                synthese = json.loads((d / "synthese.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (synthese.get("etat") or {}).get("etat") != "terminee":
                continue
            if ((synthese.get("choix_forces") or {}).get("n") or 0) > 0:
                return d
    return None


EXEC_REELLE = _exec_reelle()

pytestmark = pytest.mark.skipif(
    EXEC_REELLE is None,
    reason="no finished run WITH forced choices in data/experiences",
)


@pytest.fixture
def exec_tmp(tmp_path):
    dst = tmp_path / EXEC_REELLE.name
    shutil.copytree(EXEC_REELLE, dst)
    for f in ("scores.json", "synthese_scores.html"):
        (dst / f).unlink(missing_ok=True)
    return dst


@pytest.fixture(scope="module")
def registre():
    return F.charger()


@pytest.fixture(scope="module")
def scores(registre):
    """One scoring, shared: `calculer` reads moves.csv twice and scores two frames."""
    if EXEC_REELLE is None:
        pytest.skip("substrate missing")
    return S.calculer(EXEC_REELLE, registre.reference)


# ── The measurement is made, without being asked for ─────────────────────────


def test_le_scoring_publie_les_deux_lectures_sans_option(scores):
    """No flag, no parameter: an ordinary scoring carries both composites."""
    comp = scores["composite"]
    for cle in ("emd_jsd", "l1", "emd_jsd_hors_choix_unique", "l1_hors_choix_unique"):
        assert comp.get(cle) is not None, f"reading missing from scores.json: {cle}"


def test_la_seconde_lecture_egale_un_scoring_direct_sans_ces_lignes(scores, registre):
    """It is not approximated from the first one: it is the SAME Scorer on the filtered frame."""
    scorer, _ = S.scorer_pour(registre.reference)
    # Through `lire_perimetre`, and not by re-specifying the cut here: it is this
    # duplication that had let the test reproduce a scope that was no longer the
    # scorer's, and pass while the two diverged.
    rows, _ = S.lire_perimetre(
        EXEC_REELLE, S.EXCLURE_METHODES + [S.METHODE_CHOIX_UNIQUE_MOVES]
    )
    attendu = S.frames.simulation_frames(rows)["attendu"]
    cerema = S.frames.load_cerema(
        S._resoudre_cerema(json.loads((EXEC_REELLE / "synthese.json").read_text()))
    )
    direct = scorer.score(attendu, cerema)["emd_jsd"]["composite"]
    assert abs(scores["composite"]["emd_jsd_hors_choix_unique"] - direct) < 1e-9


def test_le_composite_principal_compte_toujours_ces_lignes(scores):
    """The convention has not changed: we measure, we remove nothing.

    `EXCLURE_METHODES` stays the list of rows WITHOUT a mode decision; a single-mode offer
    is not one of them — it produced a trip, and the day played it.
    """
    assert S.METHODE_CHOIX_UNIQUE_MOVES not in S.EXCLURE_METHODES
    forces = scores["choix_forces"]
    assert forces["n_scorees"] == forces["n_hors_choix_unique"] + forces["n"]


# ── The count travels with the figure, and names its scope ───────────────────


def test_le_compte_accompagne_le_composite(scores):
    forces = scores["choix_forces"]
    assert forces["n"] > 0
    assert forces["part"] == pytest.approx(forces["n"] / forces["n_scorees"])
    assert forces["perimetre"] and forces["lecture"]


def test_les_deux_perimetres_sont_nommes_et_distincts(scores):
    """The trap closed here: a count placed next to a figure it does not cover.

    `synthese.json` counts over ALL archived decisions; the composite only covers
    the first simulated day and the last attempt. Measured on the 16 runs of 11/09,
    the two counts differ by 14 to 19 rows — never by zero. Publishing one for the other
    would give a forced-choice share wrong by several points (27.0 % versus 20.6 % on
    `exp_lgbm`).
    """
    forces = scores["choix_forces"]
    assert forces["perimetre"] != forces["perimetre_execution"]
    assert forces["n_execution"] is not None, (
        "the run count must be relayed"
    )
    # Relayed, not recomputed: it equals the one `synthese.json` wrote.
    synthese = json.loads((EXEC_REELLE / "synthese.json").read_text())
    assert forces["n_execution"] == synthese["choix_forces"]["n"]


# ── Nothing degrades silently ────────────────────────────────────────────────


def test_un_rejeu_de_formule_recompose_LES_DEUX_composites(exec_tmp, registre):
    """Otherwise the second reading would stay frozen under the old formula.

    Two figures side by side computed under two weightings: their gap would no longer
    measure the forced choices but the formula change, without anything saying so.
    """
    ref = S.calculer(exec_tmp, registre.reference)
    poids_alt = {**registre.reference.poids, "age": 0.9, "distance": 0.1}
    alt = F.Formule("alt", poids_alt, F.empreinte(poids_alt))
    rejoue, frais = S.rejouer(ref, alt), S.calculer(exec_tmp, alt)
    for cle in ("emd_jsd", "emd_jsd_hors_choix_unique", "l1", "l1_hors_choix_unique"):
        assert abs(rejoue["composite"][cle] - frais["composite"][cle]) < 1e-9, cle


def test_sans_decision_restante_la_seconde_lecture_est_non_mesuree_jamais_zero(
    exec_tmp, registre, monkeypatch
):
    """Vacuity ≠ perfection: here 0.0 is the PERFECT score, so never a fallback value.

    We simulate the edge case — all scored decisions are single-itinerary — by
    making the second reading return an empty frame.
    """
    vraie_lecture = S.frames.read_moves

    def lecture(path, exclure, **kw):
        if S.METHODE_CHOIX_UNIQUE_MOVES in exclure:
            return [], {}
        return vraie_lecture(path, exclure, **kw)

    monkeypatch.setattr(S.frames, "read_moves", lecture)
    contenu = S.calculer(exec_tmp, registre.reference)
    assert contenu["composite"]["emd_jsd_hors_choix_unique"] is None
    assert contenu["composite"]["l1_hors_choix_unique"] is None
    assert contenu["choix_forces"]["n_hors_choix_unique"] == 0


def test_l_ecart_entre_lectures_leve_une_alarme(exec_tmp, registre, caplog):
    """A decisive gap is not discovered by comparing two columns by hand.

    The threshold (1.0 EMD point) separates the two regimes measured on 2026-09-12: chain off,
    the gap tops out at 0.32; chain active, it starts at 2.62.
    """
    import logging

    from loguru import logger

    tampon: list[str] = []
    sink = logger.add(lambda m: tampon.append(m), level="ERROR")
    try:
        contenu = S.calculer(exec_tmp, registre.reference)
    finally:
        logger.remove(sink)
    comp = contenu["composite"]
    ecart = abs(comp["emd_jsd_hors_choix_unique"] - comp["emd_jsd"])
    attendue = ecart >= S.ECART_LECTURES_ALARME
    levee = any("[ALARME]" in m and "choix forcés" in m for m in tampon)
    assert levee is attendue, (
        f"gap {ecart:.2f} vs threshold {S.ECART_LECTURES_ALARME} — alarm raised: {levee}"
    )
    assert logging  # the import stays explicit: caplog does not capture loguru


# ── The comparison surfaces carry the count ──────────────────────────────────


def test_le_registre_cli_porte_le_compte_et_la_seconde_lecture():
    """The table where the arms are compared, in the console."""
    from experiences.registre import formater_table

    colonnes_par_defaut = formater_table([]).splitlines()[0]
    for colonne in ("choix_forces", "composite_emd_hors_forces"):
        assert colonne in colonnes_par_defaut, (
            f"`{colonne}` must be displayed BY DEFAULT: a column that has to be "
            "asked for is a column nobody asks for"
        )


def test_le_tableau_de_bord_porte_le_compte_par_defaut():
    """And the one where they are compared on screen — the surface that was blocked."""
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.dashboard.experiences import (
        COLONNES_REGISTRE,
        COLONNES_REGISTRE_DEFAUT,
    )

    for colonne in ("choix_forces", "composite_emd_hors_forces"):
        assert colonne in COLONNES_REGISTRE_DEFAUT
    for colonne in ("part_forces", "choix_forces_score", "composite_l1_hors_forces"):
        assert colonne in COLONNES_REGISTRE, "at least recallable from the selector"
