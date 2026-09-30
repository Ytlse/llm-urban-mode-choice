"""Bridge from a campaign run to the frozen press grid.

Written on SHAPE runs: a bridge whose script only exists after the campaign is
cut to what it found. No call, no simulator.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis.presse import campagne  # noqa: E402
from scripts.analysis.presse.scoring import NON_CONCLUANT  # noqa: E402

GRILLE = RACINE / "data" / "presse" / "grille_signes.yaml"


def _run(dossier: Path, article: str, lignes: list[dict], avec_role: bool = True) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "evenements.jsonl").write_text(
        json.dumps({"evenement_id": article, "canal": "lu", "person_id": "1"}) + "\n",
        encoding="utf-8",
    )
    entetes = ["Référence", campagne.COLONNE_AGENT, campagne.COLONNE_MODE, campagne.COLONNE_JOUR,
               campagne.COLONNE_EVENEMENT] + ([campagne.COLONNE_ROLE] if avec_role else [])
    with (dossier / "moves.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=entetes, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)
    return dossier


def _decisions(role: str, avant: dict[str, int], apres: dict[str, int]) -> list[dict]:
    """`{canonical mode: how many}` before and after publication."""
    lignes = []
    for periode, compte, jour in (("avant", avant, -2), ("apres", apres, 1)):
        for mode, n in compte.items():
            for i in range(n):
                lignes.append({
                    campagne.COLONNE_AGENT: f"{role}{i}",
                    campagne.COLONNE_MODE: mode,
                    campagne.COLONNE_JOUR: jour,
                    campagne.COLONNE_EVENEMENT: "a13_punaises_metro",
                    campagne.COLONNE_ROLE: role,
                })
    return lignes


# ── Pairing is by period, not by condition ───────────────────────────────────────────────
def test_lecart_compare_AVANT_et_APRES_la_parution(tmp_path):
    """Stage 1 paired two conditions; a campaign pairs two periods."""
    run = _run(tmp_path / "r", "a13_punaises_metro", _decisions(
        "expose",
        avant={"public_transport": 10, "car": 10},
        apres={"public_transport": 4, "car": 16},
    ))
    ecarts, effectifs = campagne.parts_par_role(run, article="a13_punaises_metro")
    tc = ecarts[("expose", "tc")]
    assert tc.part_reference == pytest.approx(0.50)
    assert tc.part_condition == pytest.approx(0.20)
    assert tc.ecart == pytest.approx(-0.30)
    assert effectifs["expose"] == {"avant": 20, "apres": 20}


def test_le_jour_0_est_rangé_APRES_la_parution(tmp_path):
    """The take is on waking: the agent decides after reading, on the same day."""
    lignes = [
        {campagne.COLONNE_AGENT: "a", campagne.COLONNE_MODE: "car",
         campagne.COLONNE_JOUR: 0, campagne.COLONNE_EVENEMENT: "a13",
         campagne.COLONNE_ROLE: "expose"},
    ]
    run = _run(tmp_path / "r", "a13", lignes)
    _e, effectifs = campagne.parts_par_role(run, article="a13")
    assert effectifs["expose"] == {"avant": 0, "apres": 1}


# ── The guards ───────────────────────────────────────────────────────────────────────────
def test_sous_leffectif_minimal_le_verdict_est_non_concluant_jamais_zero(tmp_path):
    run = _run(tmp_path / "r", "a13", _decisions(
        "expose", avant={"car": 2}, apres={"car": 2},
    ))
    ecarts, _ = campagne.parts_par_role(run, article="a13")
    for mode in campagne.MODE_GRILLE_VERS_CANONIQUE:
        e = ecarts[("expose", mode)]
        assert e.verdict == NON_CONCLUANT
        assert e.ecart is None, "above all not 0.0: that is the value of a null effect"


def test_un_run_sans_colonne_role_le_dit_au_lieu_de_deviner(tmp_path):
    run = _run(tmp_path / "r", "a13", [
        {campagne.COLONNE_AGENT: "a", campagne.COLONNE_MODE: "car",
         campagne.COLONNE_JOUR: 1, campagne.COLONNE_EVENEMENT: "a13"},
    ], avec_role=False)
    with pytest.raises(SystemExit, match="cannot be reconstructed"):
        campagne.parts_par_role(run, article="a13")


def test_un_run_sans_evenement_ne_se_score_pas(tmp_path):
    dossier = tmp_path / "r"
    dossier.mkdir()
    (dossier / "moves.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="carries no event"):
        campagne.depouiller([dossier], grille_chemin=GRILLE, plancher=0.032)


def test_le_plancher_de_bruit_na_pas_de_defaut():
    """A default would make it forgotten, and a gap smaller than the noise is not an effect."""
    import inspect

    signature = inspect.signature(campagne.depouiller)
    assert signature.parameters["plancher"].default is inspect.Parameter.empty


def test_la_traduction_des_modes_est_explicite_et_complete():
    """Three mode vocabularies coexist: a guessed word would score the wrong mode."""
    from scripts.analysis.presse.grille import charger_grille

    grille = charger_grille(GRILLE)
    modes_grille = {c.mode for c in grille.cellules}
    assert modes_grille == set(campagne.MODE_GRILLE_VERS_CANONIQUE), (
        "every mode of the grid must have its canonical translation, and conversely"
    )


# ── The report ───────────────────────────────────────────────────────────────────────────
def _campagne_de_forme(tmp_path: Path, n_articles: int = 5) -> list[Path]:
    from scripts.analysis.presse.grille import charger_grille

    articles = sorted({c.article for c in charger_grille(GRILLE).cellules})[:n_articles]
    runs = []
    for i, article in enumerate(articles):
        lignes = _decisions(
            "expose",
            avant={"public_transport": 10, "car": 10},
            apres={"public_transport": 4, "car": 16} if i else
                  {"public_transport": 10, "car": 10},
        )
        lignes = [dict(l, **{campagne.COLONNE_EVENEMENT: article}) for l in lignes]
        runs.append(_run(tmp_path / article, article, lignes))
    return runs


def test_le_rapport_recopie_lempreinte_le_plancher_et_le_role(tmp_path):
    """A score without them does not prove that the prediction preceded the measurement."""
    rendu = campagne.rendre(campagne.depouiller(
        _campagne_de_forme(tmp_path), grille_chemin=GRILLE, plancher=0.032,
    ))
    assert "grille gelée" in rendu and "plancher de bruit déclaré" in rendu
    assert "0.032" in rendu and "`expose`" in rendu
    assert "déclaré**, pas mesuré par ce run" in rendu


def test_le_rapport_dit_quil_ny_a_PAS_de_binomial(tmp_path):
    rendu = campagne.rendre(campagne.depouiller(
        _campagne_de_forme(tmp_path), grille_chemin=GRILLE, plancher=0.032,
    ))
    assert "Pas de test binomial" in rendu
    assert "Intervalle à 95 %" in rendu and "par événement" in rendu


def test_un_seul_article_ne_donne_aucun_intervalle(tmp_path):
    """An interval drawn on a single group resamples nothing."""
    rendu = campagne.rendre(campagne.depouiller(
        _campagne_de_forme(tmp_path, n_articles=1), grille_chemin=GRILLE, plancher=0.032,
    ))
    assert "non calculable" in rendu


def test_le_kappa_sort_non_concluant_et_dit_pourquoi(tmp_path):
    """The ordinal intensity is not derived from a modal share — it requires judgement."""
    rendu = campagne.rendre(campagne.depouiller(
        _campagne_de_forme(tmp_path), grille_chemin=GRILLE, plancher=0.032,
    ))
    assert NON_CONCLUANT in rendu
    assert "intensite_jugee" in rendu
