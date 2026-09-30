"""Mode vocabulary.

These tests would have failed before 2026-09-15: the thirty-day run produced
231 concepts, 211 of them without an object axis, because `mode_canonique` only knew the
leg labels whereas the reflection schema imposes the canonical modes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.axes import MODES_INCONNUS, mode_canonique
from mobility_core.mode_hierarchy import hierarchy

# The seven values that the `stm_reflection` JSON schema allows in the `mode` field.
# Copied here ON PURPOSE: if the schema changes, this test must fail and not adapt
# silently — it is the only place that links the two files.
MODES_DU_SCHEMA = (
    "walking",
    "cycling",
    "car",
    "public_transport",
    "train",
    "motorbike",
    "any",
)


@pytest.mark.parametrize(
    "mode",
    [m for m in MODES_DU_SCHEMA if m != "any"],
)
def test_A1_un_mode_canonique_est_son_propre_canonique(mode):
    """A1 — the six schema modes that denote a mode are returned as is."""
    assert mode_canonique(mode) == mode


@pytest.mark.parametrize(
    "jambe, attendu",
    [
        ("foot", "walking"),
        ("bus", "public_transport"),
        ("bicycle", "cycling"),
        ("metro", "public_transport"),
        ("rail", "train"),
        ("school_bus", "public_transport"),
    ],
)
def test_A2_les_etiquettes_de_jambes_sont_inchangees(jambe, attendu):
    """A2 — the reading of legs does not change: it is what carries the modal shares."""
    assert mode_canonique(jambe) == attendu


def test_A3_le_mode_principal_l_emporte_toujours():
    """A3 — "foot,bus,foot" remains a public transport trip, not a walk."""
    assert mode_canonique("foot,bus,foot") == "public_transport"
    assert mode_canonique("foot,metro,foot,bus,foot") == "public_transport"


def test_A4_any_ne_devient_pas_un_axe():
    """A4 — `any` says "this concept is not about a mode": it must produce no axis."""
    assert mode_canonique("any") is None


def test_A4bis_any_n_est_pas_compte_comme_un_mode_inconnu():
    """A4 — and it is not an anomaly: counting it would drown the counter of real ones."""
    MODES_INCONNUS.pop("any", None)
    mode_canonique("any")
    assert "any" not in MODES_INCONNUS


def test_A5_un_mot_hors_des_deux_vocabulaires_est_compte():
    """A5 — the unknown stays `None`, counted, and never filed in a catch-all."""
    MODES_INCONNUS.pop("teleport", None)
    assert mode_canonique("teleport") is None
    assert MODES_INCONNUS.get("teleport") == 1
    mode_canonique("teleport")
    assert MODES_INCONNUS.get("teleport") == 2


@pytest.mark.parametrize("vide", [None, "", "   ", ",", " , "])
def test_A6_les_valeurs_vides_ne_levent_pas(vide):
    """A6 — a missing field must never bring a consolidation down."""
    assert mode_canonique(vide) is None


def test_A7_la_liste_des_canoniques_vient_de_la_ressource_gelee():
    """A7 — no list of modes hard-coded in `axes.py`.

    The test reads the resource and checks that EVERYTHING it declares goes through. A list
    copied into the module would diverge the day the hierarchy changes.
    """
    for canonique in hierarchy().canonical_order():
        assert mode_canonique(canonique) == canonique

    source = (Path(__file__).resolve().parents[1] / "llm" / "axes.py").read_text(
        encoding="utf-8"
    )
    for canonique in hierarchy().canonical_order():
        if canonique == "car":
            continue  # legitimately appears in the module's prose
        assert f'"{canonique}"' not in source, (
            f"'{canonique}' is hard-coded in axes.py: the list must come from the "
            f"frozen resource, not from the module"
        )


@pytest.mark.parametrize(
    "brut, attendu",
    [
        ("  walking ", "walking"),
        ("PUBLIC_TRANSPORT", "public_transport"),
        ("Car", "car"),
        (" FOOT", "walking"),
    ],
)
def test_A8_casse_et_espaces(brut, attendu):
    """A8 — the model guarantees neither the case nor the absence of spaces."""
    assert mode_canonique(brut) == attendu


def test_A9_garde_aucun_concept_sans_axe_objet_quand_le_mode_est_renseigne():
    """A9 — the guard rule, the one that should have failed on 14 September.

    Every mode the schema allows, except `any`, must produce an object axis. That is
    exactly what was missing: six values out of seven returned `None`, and the alarms raised
    by the module warned no one amid 355 000 log lines.
    """
    sans_axe = [
        m for m in MODES_DU_SCHEMA if m != "any" and mode_canonique(m) is None
    ]
    assert not sans_axe, (
        f"these schema modes produce no object axis: {sans_axe} — the concepts "
        f"that carry them would be invisible to the basket and to the correction"
    )
