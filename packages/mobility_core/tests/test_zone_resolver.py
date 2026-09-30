"""Tests of the fine-zone resolver (core/zone_resolver.py).

Two levels. The unit tests work on a **synthetic** layer of three
squares written into a temporary GPKG: they run offline, without the restricted-access
PROGEDO data, and check the formula and the edge cases. The parity
tests, at the end of the file, skip themselves when the real layer has not been
exported; they check that the mapping and `dist_center_km` hold on the 785
real zones and agree with `feature_spec.json`.
"""

import json
import math
import warnings
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas", reason="the resolver requires the 'geo' extra")
from shapely.geometry import Polygon  # noqa: E402

from mobility_core.zone_resolver import (  # noqa: E402
    DEFAULT_RESOURCE,
    GeoFeatures,
    Zone,
    ZoneResolver,
    geo_features,
    od_km,
)

FEATURE_SPEC = Path(__file__).resolve().parents[3] / "scripts" / "progedo_logit" / "feature_spec.json"

# Lambert 93 around Toulouse: three adjacent squares with 1 km sides, aligned along
# X. B touches A on its west boundary, C is separated from B by 1 km of void.
#
#   A [574000..575000]  B [575000..576000]        C [577000..578000]
_SQUARES = {
    "A": (574000.0, 6278000.0, 1000.0, 500.0),
    "B": (575000.0, 6278000.0, 1000.0, None),   # missing density, like the 81 real zones
    "C": (577000.0, 6278000.0, 1000.0, 2500.0),
}


def _square(x0: float, y0: float, side: float) -> Polygon:
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


@pytest.fixture(scope="module")
def fake_layer(tmp_path_factory) -> Path:
    """Synthetic layer in Lambert 93, in the format expected by the resolver."""
    rows, geoms = [], []
    for zf, (x0, y0, side, density) in _SQUARES.items():
        rows.append({
            "ZF": zf,
            # Centroid of the square, like XL93/YL93 of the real layer.
            "XL93": x0 + side / 2,
            "YL93": y0 + side / 2,
            "SURF_M2": side * side,
            "density_hh_km2": density,
            "dist_center_km": abs(x0 + side / 2 - 574500.0) / 1000,
        })
        geoms.append(_square(x0, y0, side))

    layer = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:2154")
    out_dir = tmp_path_factory.mktemp("zones")
    gpkg = out_dir / "zf_zones.gpkg"
    layer.to_file(gpkg, layer="zf", driver="GPKG")
    (out_dir / "zf_zones.meta.json").write_text(
        json.dumps({"layer": "zf", "geo_reference": {"n_zones": 3}}), encoding="utf-8"
    )
    return gpkg


@pytest.fixture
def resolver(fake_layer) -> ZoneResolver:
    return ZoneResolver.load(fake_layer)


def _wgs84(x_l93: float, y_l93: float) -> tuple[float, float]:
    """WGS84 (lat, lon) of a Lambert 93 point — to drive the tests in L93."""
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform(
        x_l93, y_l93
    )
    return lat, lon


# ── Point → zone mapping ────────────────────────────────────────────────────

class TestResolve:

    def test_point_interieur_rattache_a_sa_zone(self, resolver):
        assert resolver.resolve(*_wgs84(574500, 6278500)).zf == "A"
        assert resolver.resolve(*_wgs84(575500, 6278500)).zf == "B"

    def test_point_hors_couche_renvoie_none(self, resolver):
        # In the void between B and C: outside the layer, no approximate mapping.
        assert resolver.resolve(*_wgs84(576500, 6278500)) is None

    def test_point_loin_de_la_couche_renvoie_none(self, resolver):
        assert resolver.resolve(43.0, 0.5) is None

    def test_frontiere_partagee_departagee_sur_le_code_zf(self, resolver):
        """A point on the A|B boundary belongs to the layer, and always to the same zone."""
        on_edge = _wgs84(575000, 6278500)
        assert resolver.resolve(*on_edge).zf == "A"
        # Two successive resolutions must not alternate between A and B.
        assert resolver.resolve(*on_edge).zf == resolver.resolve(*on_edge).zf

    def test_densite_manquante_reste_none(self, resolver):
        """`None` and not 0: "no surveyed household" is not "empty zone"."""
        assert resolver.resolve(*_wgs84(575500, 6278500)).density_hh_km2 is None
        assert resolver.resolve(*_wgs84(574500, 6278500)).density_hh_km2 == 500.0

    def test_coordonnees_absentes_ne_sont_pas_rattachees(self, resolver):
        assert resolver.resolve_many([float("nan")], [1.44]) == [None]
        assert resolver.resolve_many([43.6], [float("nan")]) == [None]

    def test_lot_vide(self, resolver):
        assert resolver.resolve_many([], []) == []

    def test_toutes_coordonnees_absentes(self, resolver):
        """Non-empty but entirely invalid batch: no point reaches the index."""
        nan = float("nan")
        assert resolver.resolve_many([nan, nan], [nan, nan]) == [None, None]

    def test_resolution_unitaire_sans_avertissement_de_depreciation(self, resolver):
        """`resolve` is the path of a simulation decision: it must stay silent.

        pyproj fell back on its deprecated array→scalar conversion for a batch of
        size 1, at the rate of one warning per decision — and of a hard error in
        a future numpy version.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert resolver.resolve(*_wgs84(574500, 6278500)).zf == "A"

    def test_lot_desaligne_leve(self, resolver):
        with pytest.raises(ValueError, match="latitudes"):
            resolver.resolve_many([43.6, 43.7], [1.44])

    def test_lot_preserve_l_ordre_des_entrees(self, resolver):
        lats, lons = zip(
            _wgs84(577500, 6278500),   # C
            _wgs84(576500, 6278500),   # outside the layer
            _wgs84(574500, 6278500),   # A
        )
        assert [z.zf if z else None for z in resolver.resolve_many(lats, lons)] == \
            ["C", None, "A"]

    def test_zone_par_code(self, resolver):
        assert resolver.zone_by_code("A").surf_m2 == 1_000_000.0
        assert resolver.zone_by_code("Z") is None


# ── od_km: the training formula, identically ────────────────────────────────

class TestOdKm:

    def test_inter_zone_est_la_distance_entre_centroides(self, resolver):
        a, c = resolver.zone_by_code("A"), resolver.zone_by_code("C")
        # Centroids at 574500 and 577500: 3 km, whatever the query points.
        assert od_km(a, c) == pytest.approx(3.0)

    def test_intra_zone_est_la_longueur_caracteristique(self, resolver):
        a = resolver.zone_by_code("A")
        # 0.5 × √1 000 000 / 1000 = 0.5 km, and not 0 as the distance between
        # centroids of a zone with itself would give.
        assert od_km(a, a) == pytest.approx(0.5)
        assert od_km(a, a) != 0.0

    def test_intra_zone_ne_depend_pas_des_points_exacts(self, resolver):
        """The trap: two neighbouring points of the same zone keep od_km."""
        proche = resolver.geo_features(_wgs84(574100, 6278100), _wgs84(574150, 6278150))
        loin = resolver.geo_features(_wgs84(574100, 6278100), _wgs84(574900, 6278900))
        assert proche.od_km == loin.od_km == pytest.approx(0.5)
        assert proche.same_zone is True

    def test_symetrique(self, resolver):
        a, c = resolver.zone_by_code("A"), resolver.zone_by_code("C")
        assert od_km(a, c) == pytest.approx(od_km(c, a))


# ── Assembling the six features ─────────────────────────────────────────────

class TestGeoFeatures:

    def test_features_assemblees_depuis_les_deux_zones(self, resolver):
        feats = resolver.geo_features(_wgs84(574500, 6278500), _wgs84(577500, 6278500))
        assert feats == GeoFeatures(
            od_km=pytest.approx(3.0),
            same_zone=False,
            dist_center_orig_km=pytest.approx(0.0),
            dist_center_dest_km=pytest.approx(3.0),
            density_orig=500.0,
            density_dest=2500.0,
        )

    def test_une_extremite_hors_couche_annule_la_paire(self, resolver):
        dedans = _wgs84(574500, 6278500)
        dehors = _wgs84(576500, 6278500)
        assert resolver.geo_features(dedans, dehors) is None
        assert resolver.geo_features(dehors, dedans) is None

    def test_cles_alignees_sur_le_feature_spec(self, resolver):
        feats = resolver.geo_features(_wgs84(574500, 6278500), _wgs84(577500, 6278500))
        assert set(feats.as_dict()) == {
            "od_km", "same_zone", "dist_center_orig_km", "dist_center_dest_km",
            "density_orig", "density_dest",
        }

    def test_lot_desaligne_leve(self, resolver):
        with pytest.raises(ValueError, match="origins"):
            resolver.geo_features_many([(43.6, 1.44)], [])

    def test_lot_vide(self, resolver):
        assert resolver.geo_features_many([], []) == []

    def test_lot_equivaut_aux_appels_un_par_un(self, resolver):
        origins = [_wgs84(574500, 6278500), _wgs84(576500, 6278500)]
        dests = [_wgs84(577500, 6278500), _wgs84(574500, 6278500)]
        assert resolver.geo_features_many(origins, dests) == [
            resolver.geo_features(origins[0], dests[0]),
            resolver.geo_features(origins[1], dests[1]),
        ]

    def test_fonction_pure_utilisable_sans_resolveur(self):
        """`geo_features` is pure: the caller that already has its zones has nothing to load."""
        o = Zone("A", 0.0, 0.0, 1_000_000.0, 100.0, 1.0)
        d = Zone("B", 3000.0, 0.0, 4_000_000.0, None, 2.0)
        assert geo_features(o, d).as_dict() == {
            "od_km": pytest.approx(3.0),
            "same_zone": False,
            "dist_center_orig_km": 1.0,
            "dist_center_dest_km": 2.0,
            "density_orig": 100.0,
            "density_dest": None,
        }


# ── Loading and safeguards ──────────────────────────────────────────────────

class TestLoad:

    def test_ressource_absente_indique_comment_la_produire(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="make zones"):
            ZoneResolver.load(tmp_path / "absente.gpkg")

    def test_reference_geo_divergente_du_spec_leve(self, fake_layer, tmp_path):
        """Two competing hypercentres would shift dist_center_* without reporting anything."""
        spec = tmp_path / "feature_spec.json"
        spec.write_text(json.dumps({"geo_reference": {"n_zones": 785}}), encoding="utf-8")
        with pytest.raises(ValueError, match="geographic reference"):
            ZoneResolver.load(fake_layer, feature_spec=spec)

    def test_reference_geo_concordante_passe(self, fake_layer, tmp_path):
        spec = tmp_path / "feature_spec.json"
        spec.write_text(json.dumps({"geo_reference": {"n_zones": 3}}), encoding="utf-8")
        assert len(ZoneResolver.load(fake_layer, feature_spec=spec)) == 3

    def test_zones_et_geometries_desalignees_levent(self, resolver):
        with pytest.raises(ValueError, match="inconsistent resource"):
            ZoneResolver([Zone("A", 0.0, 0.0, 1.0, None, 0.0)], [], "EPSG:2154")


# ── Coverage and alarm ──────────────────────────────────────────────────────

class TestCoverage:

    def test_taux_hors_couche_compte(self, resolver):
        dedans = _wgs84(574500, 6278500)
        dehors = _wgs84(576500, 6278500)
        resolver.geo_features_many([dedans, dedans], [dedans, dehors])
        stats = resolver.coverage()
        assert stats["total"] == 4
        assert stats["outside"] == 1
        assert stats["outside_rate"] == pytest.approx(0.25)

    def test_alarme_sous_le_seuil_d_echantillon_ne_part_pas(self, resolver):
        """Too few points to conclude: no alarm, even at 100% outside the layer."""
        dehors = _wgs84(576500, 6278500)
        resolver.resolve_many([dehors[0]] * 10, [dehors[1]] * 10)
        assert resolver.coverage()["outside_rate"] == 1.0
        assert resolver.coverage()["alarm"] is False

    def test_alarme_sur_front_montant_puis_rearmement(self, resolver):
        dedans = _wgs84(574500, 6278500)
        dehors = _wgs84(576500, 6278500)

        resolver.resolve_many([dehors[0]] * 300, [dehors[1]] * 300)
        assert resolver.coverage()["alarm"] is True

        # A long mapped batch brings the overall rate below the low threshold: re-arming.
        resolver.resolve_many([dedans[0]] * 5000, [dedans[1]] * 5000)
        assert resolver.coverage()["outside_rate"] < 0.08
        assert resolver.coverage()["alarm"] is False


# ── Parity with the real layer (skipped if it is not exported) ──────────────

needs_layer = pytest.mark.skipif(
    not DEFAULT_RESOURCE.exists(),
    reason="real layer missing — `make zones` requires the PROGEDO data",
)


@pytest.fixture(scope="module")
def real_resolver() -> ZoneResolver:
    return ZoneResolver.load()


@needs_layer
class TestParitéCoucheRéelle:

    def test_les_785_zones_sont_chargees(self, real_resolver):
        assert len(real_resolver) == 785

    def test_chaque_zone_se_rattache_a_elle_meme(self, real_resolver):
        """A point inside each zone must fall back into that zone.

        A `representative_point` is taken and not the centroid: the real zones are
        concave, their centroid may fall outside.
        """
        layer = gpd.read_file(DEFAULT_RESOURCE, layer="zf")
        inner = layer.geometry.representative_point().to_crs("EPSG:4326")
        resolved = real_resolver.resolve_many(inner.y.values, inner.x.values)

        mismatched = [
            (zf, z.zf if z else None)
            for zf, z in zip(layer["ZF"].astype(str), resolved)
            if z is None or z.zf != zf
        ]
        assert not mismatched, f"{len(mismatched)} zones mismapped: {mismatched[:5]}"

    @pytest.mark.skipif(not FEATURE_SPEC.exists(), reason="feature_spec.json missing")
    def test_dist_center_km_concorde_avec_l_hypercentre_du_spec(self, real_resolver):
        """The layer's `dist_center_km` must be measured from the spec's centre.

        It is the safeguard against the project's two competing hypercentres: if the
        layer had been built with the `move_logger.py` constant, the gap
        would reach ~820 m on all zones.
        """
        hc = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))["geo_reference"]["hypercenter"]
        for zf in ("102105000", "218102000"):
            zone = real_resolver.zone_by_code(zf)
            expected = math.hypot(zone.x_l93 - hc["x_l93"], zone.y_l93 - hc["y_l93"]) / 1000
            assert zone.dist_center_km == pytest.approx(expected, abs=1e-4)

    @pytest.mark.skipif(not FEATURE_SPEC.exists(), reason="feature_spec.json missing")
    def test_le_spec_et_la_couche_decrivent_le_meme_hypercentre(self):
        """Loading with the spec must not raise: it is the A9 contract."""
        assert len(ZoneResolver.load(feature_spec=FEATURE_SPEC)) == 785

    @pytest.mark.skipif(not FEATURE_SPEC.exists(), reason="feature_spec.json missing")
    def test_projection_wgs84_vers_la_couche_retrouve_l_hypercentre(self, real_resolver):
        """WGS84 → layer CRS, to within a metre despite the spec rounding to 6 decimals."""
        hc = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))["geo_reference"]["hypercenter"]
        x, y = real_resolver._to_layer.transform(hc["lon"], hc["lat"])
        assert math.hypot(x - hc["x_l93"], y - hc["y_l93"]) < 1.0
