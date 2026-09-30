"""The population admission filter works on the SCOPE, not on a rectangle.

What is locked here:

- **the trait decides, not the geometry**: a 3rd-ring home 60 km from the Capitole
  is admitted, whereas the Tisséo stops rectangle dropped it. Measured:
  that rectangle held only 221 of the survey's 453 communes;
- **`hors périmètre` is an explicit rejection**, not an oversight: the home is known and it
  lies outside the 453 communes, it has no per-zone target;
- **a population without the trait does not pass silently**: the filter falls back to the
  bbox — former behaviour — and an **alarm** says the scope is not guaranteed.
  Without it, one would think the filter follows the survey when it follows the PT network;
- **the filter stays on the 453 communes**, even when the sampling frame is restricted to
  Haute-Garonne: hard-coding the frame limitation into the runtime would force digging it
  up again when the frame widens.
"""

from __future__ import annotations

import pytest

from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
from models import BBox, Location, PersonalIdentity, Person, PersonState

# The historical rectangle: extent of the Tisséo stops ± 0.05°.
TISSEO_BBOX = BBox(min_lon=1.1010, min_lat=43.3464, max_lon=1.7405, max_lat=43.7999)

# A 3rd-ring home south of the scope — in the survey, outside the rectangle.
LOIN = (43.20, 1.30)
# A Toulouse home, inside both.
CENTRE = (43.6045, 1.4440)


def person(lat, lon, zone=None, pid="p1") -> Person:
    traits = {"name": "Test"}
    if zone is not None:
        traits["residence_zone"] = zone
    return Person(
        person_id=pid,
        identity=PersonalIdentity(
            name="Test", traits_json=traits,
            home=Location(lon=lon, lat=lat), activities=[]),
        state=PersonState(last_location=None, last_activity_index=0),
    )


def verdict(p, bbox=TISSEO_BBOX):
    from inputs.population.eqasim_loader import perimeter_verdict

    return perimeter_verdict(p, bbox)


# ── The trait decides ──────────────────────────────────────────────────────────

def test_toutes_les_couronnes_sont_admises_meme_hors_du_rectangle():
    for zone in COURONNES:
        admis, motif = verdict(person(*LOIN, zone=zone))
        assert admis, f"{zone} rejected although it is in the survey ({motif})"


def test_hors_perimetre_est_un_rejet_explicite():
    admis, motif = verdict(person(*CENTRE, zone=OUT_OF_PERIMETER))
    assert not admis
    assert motif == OUT_OF_PERIMETER
    # And it was not the geometry that decided: the point is right in the centre.
    assert TISSEO_BBOX.min_lat <= CENTRE[0] <= TISSEO_BBOX.max_lat


def test_une_valeur_de_zone_inconnue_est_rejetee_en_le_disant():
    admis, motif = verdict(person(*CENTRE, zone="4eme couronne"))
    assert not admis
    assert "zone inconnue" in motif


def test_sans_domicile_pas_d_admission():
    p = person(*CENTRE, zone="Toulouse")
    p.identity.home = None
    admis, motif = verdict(p)
    assert not admis and motif == "sans domicile"


# ── The fallback, and its alarm ──────────────────────────────────────────────────

def test_sans_trait_le_filtre_retombe_sur_la_bbox():
    """Behaviour from before the residence ring, kept for older populations."""
    assert verdict(person(*CENTRE))[0] is True
    admis, motif = verdict(person(*LOIN))
    assert not admis and "trait absent" in motif


def test_sans_trait_ni_bbox_tout_passe():
    """No criterion available: we do not invent a rejection."""
    assert verdict(person(*LOIN), bbox=None)[0] is True


@pytest.fixture
def alarmes():
    """Captures loguru ERRORs — `caplog` does not see them (no propagation)."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="ERROR")
    yield messages
    logger.remove(sink)


def test_une_population_sans_trait_leve_une_alarme(alarmes):
    from inputs.population.eqasim_loader import _apply_perimeter_filter

    people = [person(*CENTRE, pid="a"), person(*CENTRE, zone="Toulouse", pid="b")]
    retenus = _apply_perimeter_filter(people, TISSEO_BBOX, "test")
    assert len(retenus) == 2
    assert any("[ALARME]" in m and "residence_zone" in m for m in alarmes), (
        "une population non enrichie doit alarmer : sinon on croit filtrer sur "
        "l'enquête alors qu'on filtre sur le réseau TC")


def test_une_population_enrichie_n_alarme_pas(alarmes):
    from inputs.population.eqasim_loader import _apply_perimeter_filter

    people = [person(*LOIN, zone="3rd ring", pid="a"),
              person(*CENTRE, zone="Toulouse", pid="b"),
              person(*CENTRE, zone=OUT_OF_PERIMETER, pid="c")]
    retenus = _apply_perimeter_filter(people, TISSEO_BBOX, "test")
    assert [p.person_id for p in retenus] == ["a", "b"]
    assert not [m for m in alarmes if "[ALARME]" in m]


# ── The sampling frame is not the filter ───────────────────────────────────

def test_le_filtre_porte_sur_les_453_pas_sur_le_cadre_restreint():
    """A home in a scope commune outside Haute-Garonne stays admissible.

    The sampling frame of the light version is restricted to the 31, but the
    admission filter is not: otherwise widening the frame would require
    changing the runtime, and the limitation would be hard-coded where no one looks for it.
    """
    from mobility_core.residence_zone import CommuneTable

    table = CommuneTable.load()
    hors_31 = [c for c in table.communes() if not c.startswith("31")]
    assert hors_31, "the scope does cover several départements"
    # The ring of one of them is enough to decide admission.
    zone = table.couronne_of_insee(hors_31[0])
    assert verdict(person(*LOIN, zone=zone))[0] is True
