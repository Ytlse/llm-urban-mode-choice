"""The mode hierarchy: a single source, sourced, and which refuses to guess.

What these tests lock is not a code convention but a **survey
verdict**: the order in which a multimodal trip receives its main mode.
Two notches are surprising and are therefore tested by name — the bus comes before the train, and
the whole public transport family comes before the car.

Three guards against vacuity ("the absence of measurement produces the perfect score"):
a missing resource, a missing family and an unexpected version must raise, not
fall back on a hard-coded order.
"""

from __future__ import annotations

import json

import pytest

from mobility_core.mode_hierarchy import (
    DEFAULT_RESOURCE,
    REQUIRED_VERSION,
    ModeHierarchy,
    hierarchy,
)

# ── The order itself ────────────────────────────────────────────────────────────

def test_l_ordre_est_celui_de_l_annexe_publiee():
    """AUAT/CEREMA report, appendix « Hiérarchie des modes », p. 53, mapped to legs."""
    assert hierarchy().families == (
        "metro", "tram", "cableway", "bus", "rail", "car", "motorbike", "bicycle", "foot")


def test_le_bus_passe_avant_le_train():
    """The hierarchy's arbitration: Tisséo bus at rank 4, TER liO at rank 8.

    Measured on the microdata: 34 of the 35 mixed bus/coach ↔ train trips
    settled by the survey are coded bus. A "liO coach + TER" itinerary is therefore a
    surface public transport trip, and not a train trip.
    """
    h = hierarchy()
    assert h.primary_family(("foot", "bus", "rail", "foot")) == "bus"
    assert h.primary_family(("foot", "rail", "foot")) == "rail"
    assert h.primary_label(("foot", "bus", "rail", "foot")) == "Transports_collectifs"
    assert h.primary_label(("foot", "rail", "foot")) == "Train"


def test_tout_le_collectif_passe_avant_la_voiture():
    """The notch of axis A7: 760 of the 770 mixed trips are coded "TC".

    It is the one `move_logger` had backwards — it tested `_CAR_MODES` first.
    """
    h = hierarchy()
    for collectif in ("metro", "tram", "cableway", "bus", "rail"):
        assert h.primary_family((collectif, "car")) == collectif, collectif
    assert h.primary_label(("car", "bus")) == "Transports_collectifs"


def test_la_marche_est_le_dernier_rang():
    """Rank 36, « Marche à pied UNIQUEMENT » — and it is measured, not assumed.

    In the microdata, `MODP = 01` designates exactly the trips without any
    motorised leg: 14,842 out of 54,585, and none of the 39,743 detailed trips.
    """
    h = hierarchy()
    assert h.family_rank["foot"] == max(h.family_rank.values())
    assert h.primary_family(("foot",)) == "foot"
    for autre in ("bus", "rail", "car", "bicycle", "metro"):
        assert h.primary_family(("foot", autre)) == autre, autre


def test_le_car_scolaire_est_un_autocar():
    """The synthetic "school coach" option has rank 6 (« autres autocars — scolaires »).

    It is therefore public transport, at the same rank as the bus, and **not** a car —
    even if GAMA displays it with the `__DIRECT_CAR__` marker to interpolate it.
    """
    h = hierarchy()
    assert h.primary_family(("school_bus",)) == "bus"
    assert h.primary_label(("school_bus",)) == "Transports_collectifs"
    assert h.primary_canonical(("school_bus",)) == "public_transport"


@pytest.mark.parametrize("alias,famille", [
    ("subway", "metro"), ("tramway", "tram"), ("gondola", "cableway"),
    ("funicular", "cableway"), ("bike", "bicycle"), ("walk", "foot"),
    ("__car__", "car"), ("METRO", "metro"), (" bus ", "bus"),
])
def test_les_alias_historiques_sont_reconnus(alias, famille):
    """Caches and labels still carry these spellings: ignoring them would lose them."""
    assert hierarchy().family_of(alias) == famille


# ── What must NOT be guessed ────────────────────────────────────────────────────

def test_un_mode_inconnu_rend_none_et_non_le_fourre_tout_d_a_cote():
    """`None` is an answer to count, not a default to absorb.

    The Téléo defect (2026-08-26) and the TER one (2026-09-04) are both modes
    that fell into the neighbouring category without a single log line.
    """
    h = hierarchy()
    assert h.family_of("hovercraft") is None
    assert h.family_of("") is None
    assert h.family_of(None) is None
    assert h.primary_family(()) is None
    assert h.primary_family(("hovercraft", "zeppelin")) is None
    assert h.primary_label(("hovercraft",)) is None
    # An unknown mode does not hide a known mode.
    assert h.primary_family(("hovercraft", "bus")) == "bus"


def test_une_ressource_absente_leve():
    with pytest.raises(FileNotFoundError, match="Mode hierarchy missing"):
        ModeHierarchy.load(DEFAULT_RESOURCE.parent / "il_ny_a_pas_de_fichier_ici.json")


def test_une_version_inattendue_leve(tmp_path):
    """A resource of another version has not been checked against the microdata."""
    doc = json.loads(DEFAULT_RESOURCE.read_text(encoding="utf-8"))
    doc["version"] = "mh0"
    chemin = tmp_path / "vieille.json"
    chemin.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=REQUIRED_VERSION):
        ModeHierarchy.load(chemin)


def test_une_famille_manquante_leve(tmp_path):
    """Anti-vacuity guard: without rail, "foot,rail,foot" would become walking."""
    doc = json.loads(DEFAULT_RESOURCE.read_text(encoding="utf-8"))
    doc["ordre_familles"] = [f for f in doc["ordre_familles"] if f != "rail"]
    chemin = tmp_path / "sans_rail.json"
    chemin.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="rail"):
        ModeHierarchy.load(chemin)


def test_une_ressource_sans_mode_de_jambe_leve(tmp_path):
    """It would classify nothing, so it would report no error."""
    doc = json.loads(DEFAULT_RESOURCE.read_text(encoding="utf-8"))
    doc["rang_jambe"] = {}
    chemin = tmp_path / "vide.json"
    chemin.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="no leg mode"):
        ModeHierarchy.load(chemin)


# ── The vocabularies served ─────────────────────────────────────────────────────

def test_l_ordre_canonique_derive_de_l_ordre_des_familles():
    """It is this order that the `mode_choice._MODE_KEYWORDS` cascade must follow."""
    assert hierarchy().canonical_order() == (
        "public_transport", "train", "car", "motorbike", "cycling", "walking")


def test_l_ordre_des_libelles_derive_de_l_ordre_des_familles():
    assert hierarchy().label_order() == (
        "Transports_collectifs", "Train", "Voiture Privée", "Deux-roues motorisé",
        "Vélo", "Marche")


def test_la_ressource_porte_sa_provenance():
    """A frozen target without provenance cannot be cross-checked — same rule as the others."""
    doc = json.loads(DEFAULT_RESOURCE.read_text(encoding="utf-8"))
    provenance = doc["provenance"]
    assert provenance["fichiers"], "microdata fingerprints missing"
    assert all(len(h) == 64 for h in provenance["fichiers"].values()), provenance
    assert provenance["gele_le"] and provenance["par"].endswith("export_mode_hierarchy.py")
    assert doc["source_publiee"]["page"] == 53
