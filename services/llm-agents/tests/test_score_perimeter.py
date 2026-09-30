"""A trip counts once, and the cut applies only when there is something to cut.

What these tests lock, in figures from 2026-09-16: over the eight runs without the
simulator on 14/09, the first-simulated-day cut removed **866 decisions out of 3,299**
for **zero repeated pair** — 797 of them the rank-0 trip of each persona, which
`jeu.py:deplacements_attendus()` dates to the next day because the original "home" activity
spans midnight. The scored scope rose to 56.9% of returns home versus 43.8% over
the whole day and 39.0% in the survey, and the gap between the two composite score readings
went from +2.24 to +5.64 EMD points for `lgbm`.

The tests run on fabricated logs: a run of the repository cannot produce a repeated
pair at will, and a test that checked only the no-repetition case would pass
while measuring nothing.
"""

import csv
from pathlib import Path

import pytest
from scripts.synthesis import frames

EXCLURE: list[str] = []

# Minimal columns `read_moves` requires to produce a usable row.
ENTETE = [
    "ID Personne",
    "ID Activité",
    "Temps simulé",
    "Heure de calcul",
    "Mode de transport Choisi",
    "Méthode de sélection",
    "Occupation principale",
    "Genre",
    "Âge",
    "Motifs de déplacement",
    "Distance parcourue",
    "Lieu de résidence",
    "Type de logement",
    "Modes proposés au LLM",
    "Heure de départ",
]

# 2026-03-16 08:00 UTC and the same instant the next day.
J1 = 1773648000
J2 = J1 + 86400


def _ligne(person: str, activity: str, ts: int, calcul: str = "2026-09-16 10:00:00") -> dict:
    return {
        "ID Personne": person,
        "ID Activité": activity,
        "Temps simulé": str(ts),
        "Heure de calcul": calcul,
        "Mode de transport Choisi": "Voiture Privée",
        "Méthode de sélection": "LLM",
        "Occupation principale": "Full-time worker",
        "Genre": "Homme",
        "Âge": "40",
        "Motifs de déplacement": "Travail",
        "Distance parcourue": "5000",
        "Lieu de résidence": "Toulouse",
        "Type de logement": "individuel isolé",
        "Modes proposés au LLM": "Marche | Voiture Privée",
        "Heure de départ": "2026-03-16 08:00:00",
    }


def _journal(tmp_path: Path, lignes: list[dict]) -> Path:
    chemin = tmp_path / "moves.csv"
    with chemin.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ENTETE)
        writer.writeheader()
        writer.writerows(lignes)
    return chemin


# ── R3 — no repetition, no cut ─────────────────────────────────────────────────

def test_sans_repetition_aucune_ligne_ecartee(tmp_path):
    """The case of runs without the simulator: unique trips, some of them dated
    to the next day. None must leave the scope."""
    journal = _journal(
        tmp_path,
        [
            _ligne("1", "a", J1),
            _ligne("1", "b", J1),
            _ligne("1", "c", J2),  # morning departure, dated the next day
            _ligne("2", "d", J2),
        ],
    )
    rows, stats = frames.read_moves(journal, EXCLURE, horizon_jours=1)
    assert stats["coupe"] == "aucune"
    assert stats["couples_repetes"] == 0
    assert stats.get("exclues_jour", 0) == 0
    assert len(rows) == 4


def test_la_coupe_systematique_retirait_ces_memes_lignes(tmp_path):
    """The former behaviour, kept under `first_day_only=True` (R6): it drops the two
    next-day trips, which nevertheless repeat nothing."""
    journal = _journal(
        tmp_path,
        [_ligne("1", "a", J1), _ligne("1", "c", J2), _ligne("2", "d", J2)],
    )
    rows, stats = frames.read_moves(journal, EXCLURE, first_day_only=True)
    assert stats["coupe"] == "forcee"
    assert stats["exclues_jour"] == 2
    assert len(rows) == 1


# ── R2 — cut when there is something to cut ──────────────────────────────────────────────

def test_couple_repete_declenche_la_coupe(tmp_path):
    """Sliding horizon: the same pair reappears the next day. This is the case the cut
    exists for, and it must apply without any setting announcing it."""
    journal = _journal(
        tmp_path,
        [
            _ligne("1", "a", J1),
            _ligne("1", "a", J2),  # repetition
            _ligne("1", "b", J1),
        ],
    )
    rows, stats = frames.read_moves(journal, EXCLURE, horizon_jours=1)
    assert stats["coupe"] == "repetitions"
    assert stats["couples_repetes"] == 1
    assert len(rows) == 2


def test_horizon_au_dela_dun_jour_declenche_la_coupe(tmp_path):
    """A multi-day run whose days all differ carries no repetition: only the
    declared horizon says that just one must be scored."""
    journal = _journal(tmp_path, [_ligne("1", "a", J1), _ligne("1", "b", J2)])
    rows, stats = frames.read_moves(journal, EXCLURE, horizon_jours=3)
    assert stats["coupe"] == "horizon"
    assert stats["couples_repetes"] == 0
    assert len(rows) == 1


# ── R1/R4 — uniqueness holds even if the criterion is wrong ─────────────────────────────────

def test_un_deplacement_ne_compte_quune_fois(tmp_path):
    """The invariant, checked where the cut criterion is explicitly disarmed: even
    `first_day_only=False` does not let a pair weigh twice."""
    journal = _journal(
        tmp_path,
        [_ligne("1", "a", J1), _ligne("1", "a", J2), _ligne("1", "b", J1)],
    )
    rows, stats = frames.read_moves(journal, EXCLURE, first_day_only=False)
    assert stats["exclues_doublon"] == 1
    assert len(rows) == 2
    couples = [(r["agent_id"], r["activity_id"]) for r in rows]
    assert len(couples) == len(set(couples))


def test_loccurrence_gardee_est_celle_du_premier_jour(tmp_path):
    """When the safety net applies, it keeps the first simulated day's decision — the one the
    cut would have kept. Without this rule, the decision would change day with file order."""
    journal = _journal(
        tmp_path,
        [_ligne("1", "a", J2), _ligne("1", "a", J1)],  # the next day written first
    )
    rows, _ = frames.read_moves(journal, EXCLURE, first_day_only=False)
    assert len(rows) == 1
    assert rows[0]["departure_hour"] is not None


# ── R5 — the applied scope is published ────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("horizon", "attendu"),
    [(1, "aucune"), (5, "horizon")],
)
def test_le_motif_de_coupe_est_publie(tmp_path, horizon, attendu):
    journal = _journal(tmp_path, [_ligne("1", "a", J1), _ligne("1", "b", J2)])
    _, stats = frames.read_moves(journal, EXCLURE, horizon_jours=horizon)
    assert stats["coupe"] == attendu


def test_libelle_de_perimetre_dit_le_motif():
    from experiences.score import _libelle_perimetre

    assert "toutes les décisions" in _libelle_perimetre({"coupe": "aucune"})
    assert "premier jour simulé" in _libelle_perimetre({"coupe": "horizon"})
    texte = _libelle_perimetre({"coupe": "repetitions", "couples_repetes": 3})
    assert "3 couple(s) répété(s)" in texte
