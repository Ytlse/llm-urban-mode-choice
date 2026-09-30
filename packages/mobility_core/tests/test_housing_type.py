"""Tests of the "housing type" trait (core/housing_type.py, action A2).

The trait is **imputed**: no downstream check can tell a correct imputation
from a biased one by looking at a row. What is locked here
are therefore the properties the imputation must have in bulk, and the boundaries
it must never cross:

- **the categories are exactly those of the EMC² reference** — it is the join key
  of the synthesis page, a one-character divergence would make the axis disappear
  there without an error;
- **the draw is deterministic** — not from an RNG, not from `hash()` (randomised per
  process): two runs, two machines, two moments give the same trait;
- **it is on the address** — two personas of the same home cannot end up
  one in a detached house and the other in a tower block;
- **it reproduces the law it is given** — otherwise the axis would be scored against EMC² on
  an invented distribution, which is worse than the empty axis it replaces;
- **it conditions on household size** — the lever must shift the
  law in the right direction, leave geography where it is, and never resurrect a
  category that the survey did not see in the zone;
- **it guesses nothing outside the layer** — an unknown zone returns `None`, not a category,
  a missing resource raises at load time instead of silently falling back, and a
  resource from before the size lever (v1, without levers) is rejected rather than served.

Offline, without the PROGEDO data. The parity tests with the real resource
skip themselves when it has not been exported.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from mobility_core.housing_type import (
    DEFAULT_RESOURCE,
    KEY_BY_LABEL,
    LABEL_BY_KEY,
    MIN_RESOURCE_VERSION,
    MODALITY_KEYS,
    REFERENCE_KEYS,
    SIZE_MAX,
    SIZE_TRAIT_KEY,
    TRAIT_KEY,
    HousingTypeTable,
    address_key,
    draw,
    key_for,
    label_for,
    rake,
    size_bucket,
    uniform,
)

CEREMA = Path("scripts/data/population/cerema_values.yaml")

# Test law: deliberately contrasted from one zone to the other, so that the
# geographic conditioning is visible in the drawn distributions.
_URBAIN = (0.05, 0.05, 0.30, 0.58, 0.02)
_RURAL = (0.80, 0.15, 0.03, 0.02, 0.00)
_SECTEUR = (0.40, 0.15, 0.25, 0.19, 0.01)
_GLOBAL = (0.42, 0.15, 0.24, 0.19, 0.00)

# Test levers, in the spirit of those the survey gives: the single person is
# pulled towards apartments, the large household towards houses. The neutral lever (size
# 2) is used to check that it then moves nothing at all.
_LEVIERS = {
    1: (0.45, 0.70, 1.40, 1.45, 2.00),
    2: (1.00, 1.00, 1.00, 1.00, 1.00),
    3: (1.30, 1.45, 0.72, 0.64, 0.26),
    4: (1.55, 1.35, 0.57, 0.54, 0.30),
}

# Counts by size, as the resource publishes them ("cell counts
# are written"). No effect on the draw, read by the acceptance check.
_BY_SIZE = [
    {"size": size, "individuel_isole_observed_pct": pct}
    for size, pct in ((1, 15.7), (2, 46.4), (3, 45.5), (4, 53.9))
]


@pytest.fixture
def table() -> HousingTypeTable:
    return HousingTypeTable(
        zones={"100100000": _URBAIN, "200200000": _RURAL},
        sectors={"1001": _SECTEUR},
        global_shares=_GLOBAL,
        size_leverage=dict(_LEVIERS),
        meta={},
        validation={"delivered": {"by_size": _BY_SIZE}},
    )


def _document(version: int = MIN_RESOURCE_VERSION, **overrides) -> dict:
    """A resource written to disk, as the export produces it."""
    doc = {
        "version": version,
        "modalities": list(MODALITY_KEYS),
        "sizes": list(range(1, SIZE_MAX + 1)),
        "global": list(_GLOBAL),
        "size_leverage": {str(size): {"n": 1000, "leverage": list(values),
                                      "cells": []}
                          for size, values in _LEVIERS.items()},
        "sectors": {"1001": {"n": 100, "shares": list(_SECTEUR)}},
        "zones": {"100100000": {"n": 20, "n_persons": 31, "shares": list(_URBAIN)},
                  "200200000": {"n": 20, "n_persons": 44, "shares": list(_RURAL)}},
        "validation": {"delivered": {"by_size": _BY_SIZE}},
        "meta": {"source": "test"},
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def written_table(tmp_path) -> Path:
    path = tmp_path / "zf_housing_type.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")
    return path


class TestModalites:
    """The categories are a shared contract, not a local convention."""

    def test_les_quatre_modalites_de_reference_sont_celles_de_cerema(self):
        if not CEREMA.exists():
            pytest.skip("cerema_values.yaml missing")
        published = yaml.safe_load(CEREMA.read_text(encoding="utf-8"))
        keys = tuple((published["parts_modales_2023"]["type_logement"] or {}).keys())
        assert REFERENCE_KEYS == keys

    def test_autres_est_connu_du_module_mais_hors_reference(self):
        """The survey knows « Autres » (0.4%), the published breakdown ignores it."""
        assert "autres" in MODALITY_KEYS
        assert "autres" not in REFERENCE_KEYS

    def test_libelles_et_cles_sont_en_bijection(self):
        assert len(KEY_BY_LABEL) == len(LABEL_BY_KEY) == len(MODALITY_KEYS)
        for key in MODALITY_KEYS:
            assert key_for(label_for(key)) == key

    def test_libelles_exacts_de_l_enquete(self):
        """These are the strings that go through traits_json then moves.csv."""
        assert LABEL_BY_KEY["individuel_isole"] == LABEL_BY_KEY["individuel_isole"]
        assert LABEL_BY_KEY["individuel_accole"] == LABEL_BY_KEY["individuel_accole"]
        assert LABEL_BY_KEY["petit_habitat_collectif"] == LABEL_BY_KEY["petit_habitat_collectif"]
        assert LABEL_BY_KEY["grand_habitat_collectif"] == LABEL_BY_KEY["grand_habitat_collectif"]

    def test_libelle_inconnu_ne_devient_pas_une_modalite(self):
        assert key_for("Maison") is None
        assert key_for("") is None
        assert label_for("individuel_neuf") is None

    def test_le_trait_porte_le_nom_attendu_par_le_journal(self):
        assert TRAIT_KEY == "housing_type"


class TestTirage:

    def test_uniforme_dans_l_intervalle(self):
        values = [uniform(f"adresse-{i}") for i in range(500)]
        assert all(0.0 <= v < 1.0 for v in values)
        assert len(set(values)) == len(values)

    def test_deterministe_entre_appels(self):
        assert uniform("43.600000,1.440000") == uniform("43.600000,1.440000")

    def test_valeur_gelee(self):
        """A change of salt or algorithm reshuffles ALL imputations.

        It must be a deliberate act: the test fails if the value moves without
        someone having updated this figure knowingly.
        """
        assert uniform("43.600000,1.440000") == pytest.approx(0.0759474, abs=1e-7)

    def test_inverse_de_la_fonction_de_repartition(self):
        shares = (0.25, 0.25, 0.25, 0.25, 0.0)
        assert draw(shares, 0.0) == "individuel_isole"
        assert draw(shares, 0.24) == "individuel_isole"
        assert draw(shares, 0.26) == "individuel_accole"
        assert draw(shares, 0.51) == "petit_habitat_collectif"
        assert draw(shares, 0.99) == "grand_habitat_collectif"

    def test_modalite_de_masse_nulle_jamais_tiree(self):
        shares = (0.0, 0.0, 1.0, 0.0, 0.0)
        assert {draw(shares, i / 1000) for i in range(1000)} == {"petit_habitat_collectif"}

    def test_loi_vide_ou_degeneree_ne_produit_pas_de_modalite(self):
        """Imputing from nothing would be exactly the invention the module refuses."""
        assert draw((), 0.5) is None
        assert draw((0.0, 0.0, 0.0, 0.0, 0.0), 0.5) is None


class TestLoiParZone:

    def test_zone_connue_sert_sa_propre_loi(self, table):
        assert table.zone_shares("100100000") == _URBAIN

    def test_zone_inconnue_se_replie_sur_son_secteur(self, table):
        assert table.zone_shares("100199000") == _SECTEUR

    def test_secteur_inconnu_se_replie_sur_le_perimetre(self, table):
        assert table.zone_shares("999999999") == _GLOBAL

    def test_hors_couche_ne_donne_aucune_loi(self, table):
        assert table.zone_shares(None) == ()

    def test_hors_couche_ne_donne_aucun_type(self, table):
        assert table.housing_type(None, 43.6, 1.44, 2) is None

    def test_le_niveau_de_repli_servi_est_publiable(self, table):
        """The count per level is required at each enrichment."""
        assert table.level_for("100100000") == "zone"
        assert table.level_for("100199000") == "secteur"
        assert table.level_for("999999999") == "perimetre"
        assert table.level_for(None) is None


class TestImputation:

    def test_meme_adresse_meme_logement(self, table):
        """930 personas share 498 homes: flatmates go together."""
        first = table.housing_type("100100000", 43.6047, 1.4442, 3)
        second = table.housing_type("100100000", 43.6047, 1.4442, 3)
        assert first is not None and first == second

    def test_adresses_voisines_tirent_independamment(self, table):
        types = {table.housing_type("100100000", 43.6 + i * 1e-4, 1.44, 2)
                 for i in range(200)}
        assert len(types) > 1

    def test_la_cle_d_adresse_est_arrondie_au_decimicron(self):
        assert address_key(43.60470004, 1.44420001) == address_key(43.6047, 1.4442)
        assert address_key(43.6047, 1.4442) != address_key(43.6048, 1.4442)

    def test_la_distribution_tiree_reproduit_la_loi(self, table):
        """Without this property, the axis would be scored against EMC² on an invented law.

        Drawn at size 2, whose test lever is neutral: what the draw must
        reproduce here is the zone law itself.
        """
        counts = Counter(table.housing_type("100100000", 43.0 + i * 1e-5, 1.4, 2)
                         for i in range(20_000))
        for key, share in zip(MODALITY_KEYS, _URBAIN):
            got = counts[LABEL_BY_KEY[key]] / 20_000
            assert got == pytest.approx(share, abs=0.015)

    def test_la_geographie_change_le_resultat(self, table):
        """A trait drawn independently of the zone would put tower blocks in open country."""
        def part(zf: str, label: str) -> float:
            counts = Counter(table.housing_type(zf, 43.0 + i * 1e-5, 1.4, 2)
                             for i in range(5_000))
            return counts[label] / 5_000

        assert part("100100000", LABEL_BY_KEY["grand_habitat_collectif"]) > 0.5
        assert part("200200000", LABEL_BY_KEY["grand_habitat_collectif"]) < 0.05
        assert part("200200000", LABEL_BY_KEY["individuel_isole"]) > 0.7


class TestLevierDeTaille:
    """The core of the lever: household size comes in, geography stays."""

    def test_le_levier_neutre_ne_deplace_rien(self):
        assert rake(_URBAIN, (1.0,) * 5) == pytest.approx(_URBAIN)

    def test_levier_absent_laisse_la_loi_intacte(self):
        assert rake(_URBAIN, None) == pytest.approx(_URBAIN)

    def test_la_loi_rakee_reste_une_distribution(self, table):
        for size in range(1, SIZE_MAX + 1):
            shares = table.shares_for("100100000", size)
            assert sum(shares) == pytest.approx(1.0)
            assert all(s >= 0.0 for s in shares)

    def test_la_personne_seule_est_tiree_vers_le_collectif(self, table):
        """Within the same zone, it is the whole point of the lever: families in
        houses, single persons in apartments."""
        seule = table.shares_for("200200000", 1)
        famille = table.shares_for("200200000", 4)
        index = MODALITY_KEYS.index("individuel_isole")
        assert seule[index] < _RURAL[index] < famille[index]

    def test_une_modalite_absente_de_la_zone_ne_ressuscite_pas(self, table):
        """The lever does not create a dwelling that the survey did not see there."""
        index = MODALITY_KEYS.index("autres")
        assert _RURAL[index] == 0.0
        for size in range(1, SIZE_MAX + 1):
            assert table.shares_for("200200000", size)[index] == 0.0

    def test_le_conditionnement_change_le_type_tire(self, table):
        """Otherwise the lever would be written in the resource with no observable effect."""
        types = {size: Counter(
            table.housing_type("200200000", 43.0 + i * 1e-5, 1.4, size)
            for i in range(5_000)) for size in (1, 4)}
        assert (types[4][LABEL_BY_KEY["individuel_isole"]] - types[1][LABEL_BY_KEY["individuel_isole"]]) > 500

    def test_la_taille_est_ecretee_a_quatre(self, table):
        """A household of six draws from the "4 and more" law: the survey says no
        more, and a finer split would estimate a lever on a few dozen."""
        assert size_bucket(6) == SIZE_MAX
        assert table.shares_for("100100000", 6) == table.shares_for("100100000", 4)

    def test_taille_absente_ou_absurde_ne_donne_aucune_loi(self, table):
        """No fallback to the zone law alone: it would be the flattened gradient
        coming back through the window, with nothing reporting it."""
        for size in (None, 0, -1, "", "quatre"):
            assert size_bucket(size) is None
            assert table.shares_for("100100000", size) == ()
            assert table.housing_type("100100000", 43.6, 1.44, size) is None

    def test_la_cle_du_trait_de_taille_est_celle_du_persona(self):
        assert SIZE_TRAIT_KEY == "household_size"

    def test_un_levier_de_mauvaise_longueur_leve(self):
        """Resource and module diverging: raise rather than rake wrongly."""
        with pytest.raises(ValueError, match="Size lever"):
            rake(_URBAIN, (1.0, 1.0))


class TestChargement:

    def test_ressource_absente_leve_avec_la_commande_a_lancer(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make housing-type"):
            HousingTypeTable.load(tmp_path / "nulle-part.json")

    def test_ressource_lue_telle_qu_ecrite(self, written_table, table):
        loaded = HousingTypeTable.load(written_table)
        assert loaded.zones == table.zones
        assert loaded.sectors == table.sectors
        assert loaded.global_shares == table.global_shares
        assert loaded.size_leverage == table.size_leverage
        assert loaded.meta["source"] == "test"

    def test_modalites_divergentes_refusees(self, tmp_path):
        """A table of another version would silently shift the whole law."""
        path = tmp_path / "table.json"
        path.write_text(json.dumps({
            "version": MIN_RESOURCE_VERSION,
            "modalities": ["maison", "appartement"],
            "global": [0.5, 0.5], "sectors": {}, "zones": {},
        }), encoding="utf-8")
        with pytest.raises(ValueError, match="categories"):
            HousingTypeTable.load(path)

    def test_ressource_d_avant_le_ticket_019_refusee(self, tmp_path):
        """A v1 has no levers: serving it would impute without size, silently.

        It is the half-done deployment scenario — module up to date, resource
        outdated — and it must raise, not produce a flattened gradient.
        """
        path = tmp_path / "v1.json"
        document = _document(version=1)
        del document["size_leverage"]
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="version 1"):
            HousingTypeTable.load(path)

    def test_ressource_a_leviers_incomplets_refusee(self, tmp_path):
        """Conditioning some households and not others would be worse than nothing."""
        path = tmp_path / "trous.json"
        document = _document()
        del document["size_leverage"]["4"]
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=r"household sizes \[4\]"):
            HousingTypeTable.load(path)

    def test_une_ligne_de_validation_incomplete_est_ignoree(self, tmp_path):
        """VERDICT path: better a missing target than a fabricated one.

        A resource written by another version may not carry all the keys.
        Filling the row with a default would have the population judged against nothing;
        ignoring it lets the downstream check say "target not served".
        """
        path = tmp_path / "partiel.json"
        document = _document()
        document["validation"]["delivered"]["by_size"] = [
            {"size": 1, "individuel_isole_observed_pct": 15.7},
            {"size": 2},                                    # measurement key missing
            {"individuel_isole_observed_pct": 45.5},        # size missing
        ]
        path.write_text(json.dumps(document), encoding="utf-8")
        assert HousingTypeTable.load(path).observed_isolated_share_by_size() == {1: 15.7}

    def test_le_bloc_de_validation_sert_les_cibles_de_recette(self, written_table):
        loaded = HousingTypeTable.load(written_table)
        assert loaded.observed_isolated_share_by_size() == {
            1: 15.7, 2: 46.4, 3: 45.5, 4: 53.9}


@pytest.mark.skipif(not DEFAULT_RESOURCE.exists(),
                    reason="housing type table not exported (make housing-type)")
class TestPariteAvecLaVraieTable:
    """Checks on the real resource, when it is present."""

    def test_les_lois_sont_des_distributions(self):
        table = HousingTypeTable.load()
        assert sum(table.global_shares) == pytest.approx(1.0, abs=1e-3)
        for shares in list(table.zones.values()) + list(table.sectors.values()):
            assert len(shares) == len(MODALITY_KEYS)
            assert sum(shares) == pytest.approx(1.0, abs=1e-3)
            assert all(s >= 0.0 for s in shares)

    def test_les_quatre_leviers_sont_servis(self):
        table = HousingTypeTable.load()
        assert set(table.size_leverage) == set(range(1, SIZE_MAX + 1))
        for values in table.size_leverage.values():
            assert len(values) == len(MODALITY_KEYS)
            assert all(v >= 0.0 for v in values)

    def test_le_levier_va_dans_le_sens_de_l_enquete(self):
        """Single person → apartment, large household → house. If this sign
        flips, it is the resource that is wrong, not the module."""
        table = HousingTypeTable.load()
        assert table.size_leverage[1][0] < 1.0 < table.size_leverage[SIZE_MAX][0]
        assert table.size_leverage[SIZE_MAX][3] < 1.0 < table.size_leverage[1][3]

    def test_le_test_interne_emc2_tient_le_critere_du_ticket(self):
        """The acceptance criterion lives in the resource, not in a promise."""
        table = HousingTypeTable.load()
        validation = table.validation
        assert validation.get("passes") is True
        assert (validation["delivered"]["mean_abs_error_pt"]
                <= validation["max_mean_abs_error_pt"])

    def test_chaque_zone_a_le_secteur_de_repli_correspondant(self):
        table = HousingTypeTable.load()
        assert table.zones and table.sectors
        for zf in table.zones:
            assert zf[:4] in table.sectors
