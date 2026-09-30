"""Setting the "residence ring" trait on a synthetic population.

This is the step that creates the trait: without it, the ring of a home stays guessed from its
distance to the hypercentre, and 24.4 % of personas are compared with the target of another zone.
What is checked here:

- the trait is set under the names that the log and the synthesis read back, with the labels
  of the EMC² reference, **and the municipality comes with it** — it is what makes the classification
  auditable;
- **three situations, three distinct writes**: a ring when the home is in
  the scope, `hors périmètre` when it is known and outside, **no trait** when it has
  no coordinates. Confusing the last two means asserting "outside" about someone
  we know nothing about;
- **the municipality is never invented**: a home outside the layer gets none, and a zone
  matched but absent from the table gets nothing at all;
- the enrichment is **idempotent**: the trait is observed, not drawn, so replaying does not
  change a single byte;
- `--check` checks what the enrichment controls — coverage, agreement between the
  classification by CODE and the classification by geometric MEMBERSHIP, categories, rate outside
  the scope — and **nothing else**: the gap to the population framing is reported with its
  own exit code, because it measures the draw (axis A9) and not this trait;
- **`--out` exists for populations pinned** by a frozen-set manifest: rewriting
  them in place would break four sets at once.

Offline, without the PROGEDO data: the ring table is built by hand and
the zone resolver is replaced by a double that assigns by latitude.
"""

from __future__ import annotations

import json

import pytest

from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
from mobility_core.residence_zone import (
    COMMUNE_TRAIT_KEY,
    INSEE_TRAIT_KEY,
    TRAIT_KEY,
    CouronneTable,
    ResidenceZoneError,
    ZoneCouronne,
)
from scripts.data.population import enrich_residence_zone as enrich_module

# Four fine zones, one per ring, on four distinct sectors.
ZONES = [
    ZoneCouronne("101101000", "101", "Toulouse", "31555", "Toulouse"),
    ZoneCouronne("201101000", "201", "1st ring", "31069", "Blagnac"),
    ZoneCouronne("301101000", "301", "2nd ring", "31088", "Bruguières"),
    ZoneCouronne("401101000", "401", "3rd ring", "31009", "Alan"),
]


class FakeZone:
    """What `ZoneResolver.resolve` returns that is useful for this trait: a fine zone code."""

    def __init__(self, zf: str) -> None:
        self.zf = zf


class FakeResolver:
    """Assigns by latitude band, and returns `None` outside the known bands.

    Latitude carries the information: above 44, we are "outside the layer". This avoids
    shipping the restricted-access GIS layer in a unit test.
    """

    MAPPING = {43: "101101000", 42: "201101000", 41: "301101000", 40: "401101000"}

    def resolve(self, lat, lon):
        code = self.MAPPING.get(int(lat)) if lat is not None else None
        return FakeZone(code) if code else None


class FakeGeometry:
    """Reference classification: here, the truth by construction of the test."""

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    def classify(self, lat, lon):
        if lat is None or lon is None:
            return ""
        return self._mapping.get(int(lat), OUT_OF_PERIMETER)


GEOMETRY_TRUTH = {43: "Toulouse", 42: "1st ring", 41: "2nd ring",
                  40: "3rd ring"}


def table() -> CouronneTable:
    return CouronneTable(ZONES)


def person(lat, lon=1.44, **traits) -> dict:
    return {"identity": {"home": {"lat": lat, "lon": lon}, "traits_json": dict(traits)}}


def traits(person_dict: dict) -> dict:
    return person_dict["identity"]["traits_json"]


# ── Setting the trait ────────────────────────────────────────────────────────

def test_le_trait_porte_la_couronne_et_la_commune():
    people = [person(43.6), person(42.5)]
    counts = enrich_module.enrich(people, table(), FakeResolver())

    assert traits(people[0])[TRAIT_KEY] == "Toulouse"
    assert traits(people[0])[COMMUNE_TRAIT_KEY] == "Toulouse"
    assert traits(people[0])[INSEE_TRAIT_KEY] == "31555"
    assert traits(people[1])[TRAIT_KEY] == "1st ring"
    assert traits(people[1])[COMMUNE_TRAIT_KEY] == "Blagnac"
    assert counts["Toulouse"] == 1 and counts["1st ring"] == 1


def test_hors_couche_recoit_hors_perimetre_et_aucune_commune():
    """`hors périmètre` is a value; the municipality, for its part, is not invented."""
    people = [person(44.9)]
    counts = enrich_module.enrich(people, table(), FakeResolver())

    assert traits(people[0])[TRAIT_KEY] == OUT_OF_PERIMETER
    assert COMMUNE_TRAIT_KEY not in traits(people[0])
    assert INSEE_TRAIT_KEY not in traits(people[0])
    assert counts[OUT_OF_PERIMETER] == 1
    # And it is not a ring: confusing it with the 3rd is gap A4 of the audit.
    assert OUT_OF_PERIMETER not in COURONNES


def test_sans_coordonnees_aucun_trait_et_l_heritage_est_retire():
    """Neither inside nor outside: we do not know. Writing "outside" would be an assertion."""
    people = [person(None, None, **{TRAIT_KEY: "Toulouse",
                                    COMMUNE_TRAIT_KEY: "Toulouse",
                                    INSEE_TRAIT_KEY: "31555"})]
    counts = enrich_module.enrich(people, table(), FakeResolver())

    assert TRAIT_KEY not in traits(people[0])
    assert COMMUNE_TRAIT_KEY not in traits(people[0])
    assert counts["sans_domicile"] == 1


def test_zone_resolue_mais_absente_de_la_table_ne_pose_rien():
    """The layer and the table do not describe the same scope: we do not guess."""
    partielle = CouronneTable([ZONES[0]])
    people = [person(42.5)]  # mapped to 201101000, absent from the partial table
    counts = enrich_module.enrich(people, partielle, FakeResolver())

    assert TRAIT_KEY not in traits(people[0])
    assert counts["zone_hors_table"] == 1


def test_une_valeur_changee_est_comptee():
    people = [person(43.6, **{TRAIT_KEY: "3rd ring"})]
    counts = enrich_module.enrich(people, table(), FakeResolver())

    assert traits(people[0])[TRAIT_KEY] == "Toulouse"
    assert counts["valeur_changee"] == 1


def test_l_enrichissement_est_idempotent():
    """The trait is OBSERVED: two passes cannot differ."""
    people = [person(43.6), person(41.2), person(44.9), person(None, None)]
    enrich_module.enrich(people, table(), FakeResolver())
    premier = json.dumps(people, ensure_ascii=False, sort_keys=True)
    enrich_module.enrich(people, table(), FakeResolver())
    assert json.dumps(people, ensure_ascii=False, sort_keys=True) == premier


def test_la_structure_de_population_est_lue_ou_refusee():
    liste = [person(43.6)]
    assert enrich_module.people_of(liste) is liste
    enveloppe = {"people": liste}
    assert enrich_module.people_of(enveloppe) is liste
    with pytest.raises(ResidenceZoneError, match="population structure"):
        enrich_module.people_of({"agents": liste})


# ── The gates of `--check` ───────────────────────────────────────────────────

def audit_of(people, zones=None):
    return enrich_module.audit(people, table(), zones or FakeGeometry(GEOMETRY_TRUTH))


def test_une_population_conforme_ne_declenche_aucune_porte():
    people = [person(43.6), person(42.5), person(41.2), person(40.1)]
    counts = enrich_module.enrich(people, table(), FakeResolver())
    assert enrich_module.report(counts, audit_of(people)) == []


def test_un_desaccord_avec_la_geometrie_fait_echouer():
    """The gate: the classification by code must equal the one by membership."""
    people = [person(43.6)]
    counts = enrich_module.enrich(people, table(), FakeResolver())
    menteuse = FakeGeometry({43: "2nd ring"})

    failures = enrich_module.report(counts, audit_of(people, menteuse))
    assert failures and "APPARTENANCE" in failures[0]


def test_une_modalite_hors_referentiel_fait_echouer():
    people = [person(43.6, **{TRAIT_KEY: "4th ring"})]
    checks = audit_of(people)  # without going through enrich: the trait is already there, wrong
    failures = enrich_module.report({}, checks)
    assert any("hors référentiel" in f for f in failures)


def test_un_persona_localise_sans_valeur_fait_echouer():
    people = [person(43.6)]  # never enriched
    failures = enrich_module.report({}, audit_of(people))
    assert any("couverture" in f for f in failures)


def test_un_taux_hors_perimetre_massif_fait_echouer():
    """Beyond the alarm threshold, it is no longer a tail: it is another scope."""
    people = [person(44.9) for _ in range(3)] + [person(43.6) for _ in range(7)]
    counts = enrich_module.enrich(people, table(), FakeResolver())
    failures = enrich_module.report(counts, audit_of(people))
    assert any("hors périmètre" in f for f in failures)
    assert enrich_module.MAX_OUT_OF_PERIMETER_RATE == 0.15


def test_le_cadrage_exclut_le_hors_perimetre_du_denominateur():
    """The out-of-scope ones have no target: diluting them would compare two quantities."""
    lignes, _ = enrich_module.framing_gap({"Toulouse": 1, OUT_OF_PERIMETER: 1})
    assert lignes["Toulouse"]["observe"] == pytest.approx(100.0)
    assert set(lignes) == set(COURONNES)


def test_le_cadrage_n_est_pas_une_porte():
    """It measures the draw (A9), not this trait: it cannot make `report` fail."""
    people = [person(43.6) for _ in range(10)]  # 100 % in Toulouse: framing very wrong
    counts = enrich_module.enrich(people, table(), FakeResolver())
    assert enrich_module.report(counts, audit_of(people)) == []
    assert enrich_module.print_framing(audit_of(people)) is True
    assert enrich_module.EXIT_FRAMING_GAP == 4


# ── Writing ──────────────────────────────────────────────────────────────────

def test_destination_ecrit_en_place_par_defaut(tmp_path):
    source = tmp_path / "population_1000.json"
    assert enrich_module.destination(source, None, 1) == source


def test_destination_respecte_out_fichier_et_dossier(tmp_path):
    """`--out` protects a population pinned by a frozen-set manifest."""
    source = tmp_path / "population_1000.json"
    cible = tmp_path / "ailleurs" / "copie.json"
    assert enrich_module.destination(source, cible, 1) == cible

    dossier = tmp_path / "sorties"
    dossier.mkdir()
    assert enrich_module.destination(source, dossier, 1) == dossier / source.name
    # Several inputs: `--out` can only be a directory, otherwise we would overwrite.
    assert enrich_module.destination(source, dossier, 3) == dossier / source.name
