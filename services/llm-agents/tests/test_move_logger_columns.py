"""Header / value alignment of the move-log, and breakdown of probabilities by mode.

A `moves.csv` row is built as a positional list: an offset between
`CSV_HEADERS` and the written row silently shifts ALL the following
columns, and the analyses then read the wrong data without flagging anything.
"""

import asyncio

import pytest

from models import Location, PersonalIdentity, Person
from urban_mobility_agents.utils.move_logger import (
    CHAIN_CONSTRAINTS,
    CSV_HEADERS,
    MODE_PROBABILITY_HEADERS,
    MoveLogger,
    _mode_probability_cells,
)

DISTRIBUTION = {"walking": 0.0, "cycling": 0.1, "car": 0.6,
                "public_transport": 0.3, "train": 0.0, "motorbike": 0.0}


class TestModeProbabilityCells:

    def test_une_cellule_par_mode_dans_l_ordre_des_entetes(self):
        cells = _mode_probability_cells(DISTRIBUTION)
        assert len(cells) == len(MODE_PROBABILITY_HEADERS)
        assert dict(zip(MODE_PROBABILITY_HEADERS, cells)) == {
            "P(Marche) %": 0.0,
            "P(Vélo) %": 10.0,
            "P(Voiture Privée) %": 60.0,
            "P(Transports_collectifs) %": 30.0,
            "P(Train) %": 0.0,
            "P(Deux-roues motorisé) %": 0.0,
            "P(Autres modes) %": 0.0,
        }

    def test_somme_a_cent(self):
        assert sum(_mode_probability_cells(DISTRIBUTION)) == pytest.approx(100.0)

    def test_absence_de_repartition_laisse_les_cellules_vides(self):
        """Empty ≠ 0: "no probabilistic decision" vs "mode explicitly ruled out"."""
        assert _mode_probability_cells(None) == [""] * len(MODE_PROBABILITY_HEADERS)
        assert _mode_probability_cells({}) == [""] * len(MODE_PROBABILITY_HEADERS)


def _person() -> Person:
    return Person(
        person_id="42",
        identity=PersonalIdentity(
            name="Test Persona",
            traits_json={"age": 30, "gender": "Female", "main_occupation": "actif"},
            home=Location(lat=43.6047, lon=1.4442),
        ),
    )


class TestRowAlignment:

    def _row(self, monkeypatch, **kwargs) -> list:
        captured = {}
        logger = MoveLogger()
        monkeypatch.setattr(logger, "_write_row", lambda row: captured.setdefault("row", row))
        asyncio.run(logger.log_move(
            person=_person(), plan=None, purpose="work", selection_method="LLM",
            provider_model="p/m", faster_itinerary=None, reasoning="parce que", **kwargs))
        return captured["row"]

    def test_ligne_alignee_sur_les_entetes(self, monkeypatch):
        assert len(self._row(monkeypatch, mode_probabilities=DISTRIBUTION)) == len(CSV_HEADERS)

    def test_ligne_alignee_sans_repartition(self, monkeypatch):
        assert len(self._row(monkeypatch)) == len(CSV_HEADERS)

    def test_probabilites_ecrites_sous_les_bonnes_colonnes(self, monkeypatch):
        row = dict(zip(CSV_HEADERS, self._row(monkeypatch, mode_probabilities=DISTRIBUTION)))
        assert row["P(Voiture Privée) %"] == 60.0
        assert row["P(Marche) %"] == 0.0
        # The neighbouring columns have not shifted.
        assert row["ID Personne"] == "42"
        assert row["Méthode de sélection"] == "LLM"


class TestContrainteDeChaine:
    """Column "Contrainte de chaîne" (chain constraint).

    It explains a decision, it does not filter it: its only requirement is
    to land in the right column with a value from the closed vocabulary.
    """

    def _row(self, monkeypatch, **kwargs) -> dict:
        captured = {}
        logger = MoveLogger()
        monkeypatch.setattr(logger, "_write_row",
                            lambda row: captured.setdefault("row", row))
        asyncio.run(logger.log_move(
            person=_person(), plan=None, purpose="work", selection_method="LLM",
            provider_model="p/m", faster_itinerary=None, reasoning="parce que",
            **kwargs))
        return dict(zip(CSV_HEADERS, captured["row"]))

    def test_colonne_presente_apres_la_methode_de_selection(self):
        i = CSV_HEADERS.index("Méthode de sélection")
        assert CSV_HEADERS[i + 1] == "Contrainte de chaîne"

    def test_absente_par_defaut(self, monkeypatch):
        """A trip without a constraint leaves the cell empty — no catch-all
        category, as for the probabilities."""
        assert self._row(monkeypatch)["Contrainte de chaîne"] == ""

    @pytest.mark.parametrize("value", [v for v in CHAIN_CONSTRAINTS if v])
    def test_valeurs_du_vocabulaire(self, monkeypatch, value):
        row = self._row(monkeypatch, chain_constraint=value)
        assert row["Contrainte de chaîne"] == value
        assert row["Fournisseur & Modèle"] == "p/m"  # the next column has not shifted

    def test_valeur_hors_vocabulaire_ramenee_a_vide(self, monkeypatch):
        """An unexpected value must not end up in the log: the summary page
        breaks this column down, an exotic category would skew the
        distribution there without being flagged."""
        assert self._row(monkeypatch, chain_constraint="n_importe_quoi"
                         )["Contrainte de chaîne"] == ""
