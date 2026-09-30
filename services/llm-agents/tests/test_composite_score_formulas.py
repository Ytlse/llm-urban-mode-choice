"""Composite score formulas.

Each test carries the number of its rule.
"""

import textwrap
from pathlib import Path

import pytest
from experiences import formule as F


def _registre(tmp_path: Path, contenu: str) -> Path:
    p = tmp_path / "reg.yaml"
    p.write_text(textwrap.dedent(contenu), encoding="utf-8")
    return p


def test_R2_length_penalty_hors_dimensions():
    # length_penalty is not a weightable dimension: the fingerprint ignores it, and
    # declaring it is refused as an unknown dimension.
    assert "length_penalty" not in F.DIMENSIONS
    with pytest.raises(ValueError):
        F._valider("x", {"length_penalty": 0.0})


def test_R3_sha_canonique_ordre_et_flottants():
    a = {"global": 1.0, "age": 0.5, "genre": 0.3}
    b = {"genre": 0.30, "age": 0.50, "global": 1.0}  # reversed order, 0.50 vs 0.5
    assert F.empreinte(a) == F.empreinte(b)


def test_R3_sha_change_si_poids_change():
    base = {"global": 1.0, "age": 0.5}
    autre = {"global": 1.0, "age": 0.6}
    assert F.empreinte(base) != F.empreinte(autre)


def test_R4_une_seule_reference_zero(tmp_path):
    reg = _registre(
        tmp_path,
        """
        formules:
          - nom: a
            poids: {global: 1.0}
    """,
    )
    with pytest.raises(ValueError, match="exactly one"):
        F.charger(reg)


def test_R4_une_seule_reference_deux(tmp_path):
    reg = _registre(
        tmp_path,
        """
        formules:
          - nom: a
            reference: true
            poids: {global: 1.0}
          - nom: b
            reference: true
            poids: {global: 0.5}
    """,
    )
    with pytest.raises(ValueError, match="exactly one"):
        F.charger(reg)


def test_R19_dimension_inconnue_refusee(tmp_path):
    reg = _registre(
        tmp_path,
        """
        formules:
          - nom: a
            reference: true
            poids: {global: 1.0, farfelu: 2.0}
    """,
    )
    with pytest.raises(ValueError, match="outside the 7"):
        F.charger(reg)


def test_R19_poids_non_numerique_refuse(tmp_path):
    reg = _registre(
        tmp_path,
        """
        formules:
          - nom: a
            reference: true
            poids: {global: "beaucoup"}
    """,
    )
    with pytest.raises(ValueError, match="non-numeric"):
        F.charger(reg)


def test_R19_poids_negatif_refuse(tmp_path):
    reg = _registre(
        tmp_path,
        """
        formules:
          - nom: a
            reference: true
            poids: {global: -1.0}
    """,
    )
    with pytest.raises(ValueError, match="negative"):
        F.charger(reg)


def test_R20_sha_ancien_resoluble(tmp_path):
    reg = _registre(
        tmp_path,
        """
        formules:
          - nom: ancienne
            poids: {global: 1.0, age: 0.5}
          - nom: courante
            reference: true
            poids: {global: 1.0, age: 0.7}
    """,
    )
    r = F.charger(reg)
    sha_ancienne = F.empreinte({"global": 1.0, "age": 0.5})
    retrouvee = r.resoudre(sha_ancienne)
    assert retrouvee is not None and retrouvee.nom == "ancienne"
    assert retrouvee.poids["age"] == 0.5


def test_registre_par_defaut_charge_et_reference_unique():
    # The shipped registry (formules/reference.yaml) must be loadable and consistent.
    r = F.charger()
    assert r.reference is not None
    assert r.reference.poids["global"] == 1.0
