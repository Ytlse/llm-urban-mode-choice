"""The judgement safeguard, exercised without a single model call.

The safeguard decides whether a fifty-day campaign starts or not. Its verdict logic must
therefore be exercised without a model: a safeguard only tested by launching it costs thirty calls
at every check, and people stop checking it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
for chemin in (str(RACINE), str(RACINE / "services" / "llm-agents")):
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

from scripts.experiment.banc_fonctionnel.garde_fou_jugement import (  # noqa: E402
    BRAS,
    GRILLE,
    charger_grille,
    verdict,
)

SEUILS = {"part_hors_plage_max": 0.20, "echelons_distincts_min": 3}


def _bilan(**kw):
    base = {
        "appels": 32, "sans_reponse": 0,
        "part_hors_plage_aveugle": 0.0, "part_hors_plage_deja_vue": 0.0,
        "echelons_distincts": 3, "bras_dans_l_ordre_predit": True, "ordre_detail": [],
        "bras": {"c2_crevaison": "notable", "c6_voiture_suspecte": "genant",
                 "c3_panne_reseau": "grave"},
    }
    return base | kw


# ── Reading a grid ──────────────────────────────────────────────────────────────────────────
def test_une_marche_nommant_un_texte_absent_arrete_le_test(tmp_path):
    """The order would bear on a text that is not measured."""
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({
        "a07_greve_eboueurs": {"attendu": ["anodin"]},
        "marche": ["a07_greve_eboueurs", "c9_inexistant"],
    }), "utf-8")
    with pytest.raises(SystemExit, match="does not declare"):
        charger_grille(p)


def test_une_grille_incomplete_arrete_le_test(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"a07_greve_eboueurs": {"attendu": []}}), "utf-8")
    with pytest.raises(SystemExit, match="incomplete"):
        charger_grille(p)


# ── The verdict ─────────────────────────────────────────────────────────────────────────────
def test_tout_au_vert_laisse_partir_la_campagne():
    ok, motifs = verdict(_bilan(), SEUILS)
    assert ok and not motifs


def test_une_seule_reponse_manquante_refuse():
    """An unjudged exposure is not a harmless exposure: it did not take place."""
    ok, motifs = verdict(_bilan(sans_reponse=1), SEUILS)
    assert not ok and any("sans réponse" in m for m in motifs)


def test_un_seul_echelon_refuse_meme_si_tout_est_dans_la_plage():
    """The case of 2026-09-22: judgements defensible one by one, and nothing to measure."""
    ok, motifs = verdict(_bilan(echelons_distincts=1), SEUILS)
    assert not ok and any("échelon" in m for m in motifs)


def test_des_bras_dans_le_desordre_refusent():
    ok, motifs = verdict(
        _bilan(bras_dans_l_ordre_predit=False,
               ordre_detail=["c3_panne_reseau (genant) est SOUS c6_voiture_suspecte (grave)"]),
        SEUILS)
    assert not ok and any("marche n'est pas tenue" in m for m in motifs)


def test_une_egalite_non_departagee_se_signale_sans_bloquer():
    """Two arms carrying the same expected range are not a refutation of the ladder."""
    ok, motifs = verdict(
        _bilan(ordre_detail=["c6 et c3 à égalité (grave) — NON départagés par la grille"]),
        SEUILS)
    assert ok, "a tie that nobody had predicted does not block"
    assert any(m.startswith("⚠") for m in motifs)


def test_le_taux_aveugle_bloque_et_le_taux_deja_vu_ne_bloque_pas():
    """The distinction IS the test: mixing the two populations would mean nothing."""
    ok, _ = verdict(_bilan(part_hors_plage_aveugle=0.50), SEUILS)
    assert not ok

    ok, motifs = verdict(_bilan(part_hors_plage_deja_vue=0.90), SEUILS)
    assert ok, "a range written after the fact cannot block a campaign"
    assert any(m.startswith("⚠") for m in motifs), "it must still be flagged"


def test_aucun_jugement_aveugle_ne_fabrique_pas_de_verdict():
    """Grid entirely already seen: the blind rate is None and must not raise."""
    ok, _ = verdict(_bilan(part_hors_plage_aveugle=None), SEUILS)
    assert ok


# ── The delivered grid ──────────────────────────────────────────────────────────────────────
def test_la_grille_du_depot_est_complete_et_lisible():
    grille, seuils, marche = charger_grille(GRILLE)
    assert len(grille) == 9, "five articles, the three arms, and the thunderstorm probe"
    assert set(marche) <= set(grille), "the arms of the ladder must be declared"
    assert len(marche) >= 3, "a ladder of fewer than three arms cannot be tested"
    assert "c5_orage_grele" in grille and "c5_orage_grele" not in marche, (
        "the thunderstorm is measured but is not part of the ordered ladder: it crosses it"
    )
    assert seuils["echelons_distincts_min"] >= 3


def test_chaque_plage_est_une_plage_d_echelons_connus():
    from llm.gravite import NIVEAUX

    grille, _, marche = charger_grille(GRILLE)
    for nom, e in grille.items():
        assert e["attendu"], nom
        assert set(e["attendu"]) <= set(NIVEAUX), f"{nom}: unknown level"
        assert e["motif"].strip(), f"{nom}: a range without a reason cannot be reviewed"
        assert isinstance(e["deja_vu"], bool), f"{nom}: `deja_vu` must be declared"


def test_la_marche_predit_une_progression_et_non_un_plateau():
    """Without a predicted order, the test could refute nothing: it would record what comes out.

    The ladder does NOT have to be strictly increasing: two arms the author cannot
    separate carry the same range, and the measurement is not asked to decide what
    nobody predicted. What is required is that it rises from one end to the other and never
    comes back down.
    """
    from llm.gravite import NIVEAUX

    rangs = {n: i for i, n in enumerate(NIVEAUX)}
    grille, _, marche = charger_grille(GRILLE)
    bornes = [
        (min(rangs[x] for x in grille[b]["attendu"]), max(rangs[x] for x in grille[b]["attendu"]))
        for b in marche
    ]
    for (b1, h1), (b2, h2) in zip(bornes, bornes[1:]):
        assert b2 >= b1 and h2 >= h1, "the ladder must never come back down"
    assert bornes[0][1] < bornes[-1][1], "the last arm must be predicted above the first"


def test_au_moins_deux_bras_sont_des_predictions_aveugles():
    """A grid written entirely after the fact guards nothing — that is the known weak point."""
    grille, _, marche = charger_grille(GRILLE)
    aveugles = [b for b in marche if not grille[b]["deja_vu"]]
    assert len(aveugles) >= 2, f"only {aveugles} are blind"
