"""Writing of the log's "Type de logement" (housing type) column (action A2).

The column used to be written empty, which left a whole axis of the EMC² reference
at zero. It now carries the trait imputed at population generation — and the
log only copies it: it draws nothing, it guesses nothing.

Three boundaries are checked here, because none of them raises an exception when it
is crossed:

- the written label is **exactly** the reference one, the only join key of
  the summary page;
- a persona without the trait leaves the cell **empty** — "not filled in" is not a
  category, just as an empty probability cell is not a 0;
- a value outside the reference data is reset to empty rather than logged as is,
  otherwise it would vanish from the page without being counted there.
"""

import asyncio

import pytest

from mobility_core.housing_type import LABEL_BY_KEY, REFERENCE_KEYS, TRAIT_KEY
from models import Location, PersonalIdentity, Person
from urban_mobility_agents.utils.move_logger import CSV_HEADERS, MoveLogger, _housing_type


def _person(**traits) -> Person:
    return Person(
        person_id="42",
        identity=PersonalIdentity(
            name="Test Persona",
            traits_json={"age": 30, "gender": "Female", **traits},
            home=Location(lat=43.6047, lon=1.4442),
        ),
    )


def _row(monkeypatch, person: Person) -> dict:
    captured = {}
    logger = MoveLogger()
    monkeypatch.setattr(logger, "_write_row", lambda row: captured.setdefault("row", row))
    asyncio.run(logger.log_move(
        person=person, plan=None, purpose="work", selection_method="LLM",
        provider_model="p/m", faster_itinerary=None, reasoning="parce que"))
    return dict(zip(CSV_HEADERS, captured["row"]))


class TestValeurEcrite:

    @pytest.mark.parametrize("key", REFERENCE_KEYS)
    def test_chaque_modalite_de_reference_est_journalisee_telle_quelle(self, monkeypatch, key):
        label = LABEL_BY_KEY[key]
        row = _row(monkeypatch, _person(**{TRAIT_KEY: label}))
        assert row["Type de logement"] == label

    def test_autres_est_journalise_aussi(self, monkeypatch):
        """The survey knows this category; it is the page that will count it outside
        the reference data, not the log that will erase it."""
        row = _row(monkeypatch, _person(**{TRAIT_KEY: "Autres"}))
        assert row["Type de logement"] == "Autres"

    def test_la_colonne_ne_deborde_pas_sur_ses_voisines(self, monkeypatch):
        row = _row(monkeypatch, _person(**{TRAIT_KEY: "Grand habitat collectif"}))
        assert row["Occupation principale"] == ""
        assert row["Motifs de déplacement"] == "Travail"


class TestAbsenceEtValeursHorsReferentiel:

    def test_persona_sans_trait_laisse_la_cellule_vide(self, monkeypatch):
        """Population generated before action A2, or home outside the zone layer."""
        assert _row(monkeypatch, _person())["Type de logement"] == ""

    def test_valeur_hors_referentiel_ramenee_a_vide(self):
        assert _housing_type({TRAIT_KEY: "Maison de ville"}) == ""
        assert _housing_type({TRAIT_KEY: "individuel_isole"}) == ""

    def test_valeur_vide_ou_absente(self):
        assert _housing_type({}) == ""
        assert _housing_type({TRAIT_KEY: None}) == ""
        assert _housing_type({TRAIT_KEY: "   "}) == ""

    def test_espaces_autour_du_libelle_tolérés(self):
        assert _housing_type({TRAIT_KEY: " Individuel isolé "}) == "Individuel isolé"
