"""The run report's "expected vs drawn" gap is measured on a common denominator.

The report sums the probabilities announced by the LLM on the rows that carry
one, but counted the drawn modes on ALL rows of `moves.csv` — single-choice,
fallback, `Aucun`. The most frequent mode lost more than ten points there and the
report raised an alarm blaming the cache. These tests pin the denominator.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_report", RACINE / "scripts" / "debug" / "run_report.py")
run_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_report)

COLONNES = ["Mode de transport Choisi", "Méthode de sélection",
            "P(Marche) %", "P(Voiture Privée) %", "P(Transports_collectifs) %"]


def _ecrire_moves(dossier: Path, lignes: list[dict]) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / "moves.csv"
    with chemin.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLONNES)
        w.writeheader()
        for ligne in lignes:
            w.writerow({c: ligne.get(c, "") for c in COLONNES})
    return chemin


def _decision(mode: str, p_marche: float, p_voiture: float, p_tc: float) -> dict:
    return {"Mode de transport Choisi": mode, "Méthode de sélection": "LLM",
            "P(Marche) %": p_marche, "P(Voiture Privée) %": p_voiture,
            "P(Transports_collectifs) %": p_tc}


def _sans_repartition(mode: str, methode: str) -> dict:
    return {"Mode de transport Choisi": mode, "Méthode de sélection": methode}


def _rapport(tmp_path: Path, lignes: list[dict]) -> tuple[str, list[str]]:
    _ecrire_moves(tmp_path, lignes)
    out: list[str] = []
    alarmes: list[str] = []
    run_report.section_decisions(tmp_path, out, alarmes)
    return "\n".join(out), alarmes


def test_tirage_fidele_aucune_alarme(tmp_path):
    """A draw that reproduces the announced distribution raises no alarm."""
    lignes = [_decision("Transports_collectifs", 20, 30, 50) for _ in range(150)]
    lignes += [_decision("Voiture Privée", 20, 30, 50) for _ in range(90)]
    lignes += [_decision("Marche", 20, 30, 50) for _ in range(60)]
    texte, alarmes = _rapport(tmp_path, lignes)
    assert not [a for a in alarmes if "Tirage modal" in a], texte
    assert "Transports_collectifs | 50.0 % | 50.0 %" in texte


def test_lignes_sans_repartition_ne_faussent_pas_l_ecart(tmp_path):
    """Single-choice, fallback and `Aucun` leave BOTH sides of the comparison.

    This is the defect measured on run 2026-09-04_16_25: 1 689 rows without a
    distribution out of 5 257 diluted the drawn share of a mode by 12 points.
    """
    lignes = [_decision("Transports_collectifs", 20, 30, 50) for _ in range(150)]
    lignes += [_decision("Voiture Privée", 20, 30, 50) for _ in range(90)]
    lignes += [_decision("Marche", 20, 30, 50) for _ in range(60)]
    # As many rows without a distribution as probabilistic decisions.
    lignes += [_sans_repartition("Aucun", "Pas de déplacement (même localisation)")
               for _ in range(150)]
    lignes += [_sans_repartition("Voiture Privée", "Un seul itinéraire disponible")
               for _ in range(150)]
    texte, alarmes = _rapport(tmp_path, lignes)
    assert not [a for a in alarmes if "Tirage modal" in a], texte
    assert "(300 décisions probabilistes" in texte
    # The mode table, for its part, does count every row of the log.
    assert "| Voiture Privée | 240 |" in texte


def test_vrai_biais_de_tirage_leve_l_alarme(tmp_path):
    """A draw that really deviates by more than 8 points is still detected."""
    lignes = [_decision("Voiture Privée", 20, 30, 50) for _ in range(300)]
    texte, alarmes = _rapport(tmp_path, lignes)
    modales = [a for a in alarmes if "Tirage modal" in a]
    assert modales, texte
    # The alarm names the most deviating mode: car, drawn 100 % for 30 % announced.
    assert "Voiture Privée" in modales[0]
    assert "70.0 pts" in modales[0]


def test_effectif_trop_faible_reste_muet(tmp_path):
    """Below 200 decisions, the gap is only noise: no alarm."""
    lignes = [_decision("Voiture Privée", 20, 30, 50) for _ in range(50)]
    _, alarmes = _rapport(tmp_path, lignes)
    assert not [a for a in alarmes if "Tirage modal" in a]


def test_journal_absent(tmp_path):
    """Without moves.csv, the section produces nothing and does not break the report."""
    out: list[str] = []
    alarmes: list[str] = []
    run_report.section_decisions(tmp_path, out, alarmes)
    assert out == [] and alarmes == []
