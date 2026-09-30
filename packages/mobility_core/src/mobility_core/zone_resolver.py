"""
core/zone_resolver.py — From a point to the geographic variables of mode choice.

The six `source: "geo"` features of `feature_spec.json` (`od_km`, `same_zone`,
`dist_center_orig_km`, `dist_center_dest_km`, `density_orig`, `density_dest`)
all derive from the same prerequisite: mapping a point to its **fine zone** of
the EMC² survey. At training this mapping is given (`D3`/`D7` of the trips
file); in simulation there are only coordinates. This module therefore replays the
spatial join, and **only with the training formula**:

- `od_km` is a distance between **zone centroids**, never between the exact
  points. The temptation of an origin→destination haversine gives a factor of 2 on
  intra-zone trips (0.65 km against 1.29 km, ticket 005 §2.1) — and these are the
  short trips, those where walking, bike and car really compete for the
  decision, and `od_km` is by far the model's first feature;
- for an intra-zone trip the distance between centroids is 0: it is replaced
  by the characteristic length of the zone, `0.5 × √surface`. `same_zone`
  comes with the value so that the model knows it is imputed, not measured;
- `density_*` and `dist_center_*` are read as they are from the resource, computed
  once and for all by the training set builder. The hypercentre is not
  redeclared here: it is already baked into `dist_center_km`, and callers that
  need it explicitly (the residence rings of the move-log) read it from
  `core.geo_reference`, which also serves the `load` safeguard below.

**The mapping is a point-in-polygon join, not a nearest neighbour.**
Mapping to the nearest centroid amounts to a Voronoi partition, which does not
resemble elongated and very unequal administrative zones: measured at only
72.9% agreement, with a median gap of 1.19 km when it is wrong.

**Outside the layer, no guessing.** ~5% of the synthetic population's locations
fall outside the survey scope, at a median 22.8 km from the nearest zone:
these are clearly external communes, not edge cases. `resolve` then returns
`None` and so does `geo_features` — up to the caller to switch to its fallback
policy (the LLM), never to invent a mapping.

The resource (`mobility_core/data/zf_zones.gpkg`) is produced by
`scripts/progedo_logit/export_zone_layer.py`. Inputs/outputs are confined to
`ZoneResolver.load`; the rest of the module is pure, in line with the
`mobility_core` architecture contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from mobility_core.geo_reference import read_geo_reference
from mobility_core.resources import restricted_data_path, restricted_resource_hint

try:
    import numpy as np
    from pyproj import Transformer
    from shapely import STRtree
    from shapely import points as shapely_points
except ImportError as exc:  # pragma: no cover - dépend de l'image
    raise ImportError(
        "mobility_core.zone_resolver requires numpy, shapely and pyproj. They come "
        "with geopandas, present in services/llm-agents/requirements.txt (controller container) "
        "but missing from the gateway dependencies: install the 'geo' extra "
        "(pip install -e 'mobility_core[geo]') in the relevant environment."
    ) from exc



# Default resource: mobility_core/data/zf_zones.gpkg, next to the package.
# RESTRICTED-ACCESS resource: the survey's GIS layer itself. Neither in the repository, nor
# in the package (ticket 038). Evaluated at import; `load()` resolves again on each call.
RESOURCE_NAME = "zf_zones.gpkg"
DEFAULT_RESOURCE = restricted_data_path(RESOURCE_NAME)
DEFAULT_META = DEFAULT_RESOURCE.with_suffix(".meta.json")

# CRS of the input coordinates. The simulation, the synthetic population and OTP
# all work in WGS84 lon/lat; the layer itself is in Lambert 93.
INPUT_CRS = "EPSG:4326"

# Coverage alarm thresholds. The expected out-of-layer rate is ~5%: beyond
# 15% on a significant sample, the layer or the population has
# changed scope, and the geo features are massively missing.
_COVERAGE_MIN_SAMPLE = 200
_COVERAGE_ALARM_RATE = 0.15
_COVERAGE_CLEAR_RATE = 0.08


@dataclass(frozen=True)
class Zone:
    """An EMC² fine zone, reduced to what the model needs."""

    zf: str
    x_l93: float
    y_l93: float
    surf_m2: float
    # Missing for the 81 zones (out of 785) with no surveyed household. `None`, never 0:
    # "unknown" and "desert" are not the same information, and the booster routes
    # missing values natively.
    density_hh_km2: float | None
    dist_center_km: float


@dataclass(frozen=True)
class GeoFeatures:
    """The six `source: "geo"` features of the spec, for an origin-destination pair."""

    od_km: float
    same_zone: bool
    dist_center_orig_km: float
    dist_center_dest_km: float
    density_orig: float | None
    density_dest: float | None

    def as_dict(self) -> dict:
        """Key names strictly those of `feature_spec.json`."""
        return {
            "od_km": self.od_km,
            "same_zone": self.same_zone,
            "dist_center_orig_km": self.dist_center_orig_km,
            "dist_center_dest_km": self.dist_center_dest_km,
            "density_orig": self.density_orig,
            "density_dest": self.density_dest,
        }


def od_km(origin: Zone, destination: Zone) -> float:
    """Origin-destination distance **with the training formula**.

    Inter-zone: distance between Lambert 93 centroids. Intra-zone: `0.5 × √surface`,
    the distance between centroids being zero by construction.
    """
    if origin.zf == destination.zf:
        return 0.5 * math.sqrt(origin.surf_m2) / 1000
    return math.hypot(origin.x_l93 - destination.x_l93,
                      origin.y_l93 - destination.y_l93) / 1000


def geo_features(origin: Zone, destination: Zone) -> GeoFeatures:
    """Assembles the six geo features from the two mapped zones."""
    return GeoFeatures(
        od_km=od_km(origin, destination),
        same_zone=origin.zf == destination.zf,
        dist_center_orig_km=origin.dist_center_km,
        dist_center_dest_km=destination.dist_center_km,
        density_orig=origin.density_hh_km2,
        density_dest=destination.density_hh_km2,
    )


class ZoneResolver:
    """Maps points to their fine zones, and derives the geo features from them.

    Building the instance loads the layer and builds a spatial index once;
    `resolve` is then a local call, without I/O. To be instantiated once per
    process (cf. `load`).
    """

    def __init__(self, zones: Sequence[Zone], geometries, layer_crs,
                 geo_reference: dict | None = None) -> None:
        if len(zones) != len(geometries):
            raise ValueError(
                f"{len(zones)} zones for {len(geometries)} geometries: inconsistent resource."
            )
        self._zones = tuple(zones)
        self._by_code = {z.zf: z for z in self._zones}
        self._tree = STRtree(geometries)
        self._to_layer = Transformer.from_crs(INPUT_CRS, layer_crs, always_xy=True)
        self.geo_reference = geo_reference or {}

        self._n_inside = 0
        self._n_outside = 0
        self._coverage_alarm = False

    # -- Loading ------------------------------------------------------------

    @classmethod
    def load(cls, resource: Path | None = None,
             feature_spec: Path | None = None) -> ZoneResolver:
        """Loads the zone resource (the module's only I/O point).

        `feature_spec` is optional but recommended: when it is given, the
        geographic reference of the layer is compared with that of the model spec, and
        any divergence raises an error at load time rather than silently producing
        `dist_center_*` measured from two different centres.
        """
        import geopandas as gpd  # local: only `load` needs geopandas

        resource = Path(resource) if resource else restricted_data_path(RESOURCE_NAME)
        if not resource.exists():
            raise FileNotFoundError(
                "Fine-zone resource missing.\n" + restricted_resource_hint(RESOURCE_NAME)
            )

        meta_path = resource.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        layer = gpd.read_file(resource, layer=meta.get("layer", "zf"))

        geo_reference = meta.get("geo_reference", {})
        if feature_spec is not None:
            expected = read_geo_reference(Path(feature_spec))
            if geo_reference and expected and geo_reference != expected:
                raise ValueError(
                    "The zone layer and feature_spec.json do not describe the same "
                    f"geographic reference.\n  layer : {geo_reference}\n  spec  : {expected}\n"
                    "Re-export the layer from the same run as the spec "
                    "(python -m scripts.progedo_logit.export_zone_layer)."
                )

        zones = [
            Zone(
                zf=str(row.ZF),
                x_l93=float(row.XL93),
                y_l93=float(row.YL93),
                surf_m2=float(row.SURF_M2),
                # NaN from parquet/GPKG → None: the boundary between "missing" and
                # "value" is carried by the type, not by a float sentinel.
                density_hh_km2=(None if row.density_hh_km2 is None
                                or _is_nan(row.density_hh_km2)
                                else float(row.density_hh_km2)),
                dist_center_km=float(row.dist_center_km),
            )
            for row in layer.itertuples(index=False)
        ]
        logger.info(
            f"Fine-zone layer loaded | zones={len(zones)} source={resource.name} "
            f"crs={layer.crs.to_string() if layer.crs else 'inconnu'}"
        )
        return cls(zones, layer.geometry.values, layer.crs, geo_reference)

    # -- Mapping ------------------------------------------------------------

    def resolve(self, lat: float, lon: float) -> Zone | None:
        """Zone containing the WGS84 point, or `None` if it is outside the layer."""
        return self.resolve_many([lat], [lon])[0]

    def zone_by_code(self, zf: str) -> Zone | None:
        """Zone by its ZF code, for the caller that already knows it (survey, tests)."""
        return self._by_code.get(str(zf))

    def resolve_many(self, lats: Iterable[float],
                     lons: Iterable[float]) -> list[Zone | None]:
        """Vectorised version: a single index query for the whole batch."""
        lat_arr = np.asarray(list(lats), dtype=float)
        lon_arr = np.asarray(list(lons), dtype=float)
        if lat_arr.shape != lon_arr.shape:
            raise ValueError(
                f"{lat_arr.size} latitudes for {lon_arr.size} longitudes."
            )

        out: list[Zone | None] = [None] * lat_arr.size
        if lat_arr.size == 0:
            return out

        # Missing coordinates must not reach the index: NaN would produce
        # an empty geometry there, silently unmatched.
        valid = ~(np.isnan(lat_arr) | np.isnan(lon_arr))
        # `.tolist()` and not the numpy array: pyproj falls back on its scalar path
        # for a size-1 array — that of `resolve`, hence of every decision in
        # simulation — and triggers there a deprecated array→scalar conversion, bound
        # to become an error. Lists go through the vectorised path at any size.
        x, y = self._to_layer.transform(lon_arr[valid].tolist(), lat_arr[valid].tolist())
        # `intersects` and not `within`: a point falling exactly on a shared boundary
        # does belong to the layer, it must not switch to the fallback.
        pairs = self._tree.query(shapely_points(x, y), predicate="intersects")

        valid_idx = np.flatnonzero(valid)
        for point_pos, zone_pos in zip(pairs[0], pairs[1]):
            target = valid_idx[point_pos]
            zone = self._zones[zone_pos]
            current = out[target]
            # Shared boundary: several zones intersect. Ties are broken on the
            # ZF code so that two runs map the point to the same zone.
            if current is None or zone.zf < current.zf:
                out[target] = zone

        self._record_coverage(out)
        return out

    # -- Features -----------------------------------------------------------

    def geo_features(self, origin: tuple[float, float],
                     destination: tuple[float, float]) -> GeoFeatures | None:
        """The six geo features for a `(lat, lon)` pair, or `None` outside the layer.

        `None` as soon as either end escapes the layer: a half-mapped
        pair gives no `od_km`, and imputing the missing end
        would amount to the approximate mapping that ticket 005 §2.1 rules out.
        """
        pairs = self.geo_features_many([origin], [destination])
        return pairs[0]

    def geo_features_many(self, origins: Sequence[tuple[float, float]],
                          destinations: Sequence[tuple[float, float]],
                          ) -> list[GeoFeatures | None]:
        """Vectorised version, to apply the model to a whole run."""
        if len(origins) != len(destinations):
            raise ValueError(
                f"{len(origins)} origins for {len(destinations)} destinations."
            )
        if not origins:
            return []

        o_zones = self.resolve_many([p[0] for p in origins], [p[1] for p in origins])
        d_zones = self.resolve_many([p[0] for p in destinations], [p[1] for p in destinations])
        return [
            geo_features(o, d) if o is not None and d is not None else None
            for o, d in zip(o_zones, d_zones)
        ]

    # -- Coverage -----------------------------------------------------------

    def coverage(self) -> dict:
        """Mapping rate since loading — to be attached to run reports."""
        total = self._n_inside + self._n_outside
        return {
            "resolved": self._n_inside,
            "outside": self._n_outside,
            "total": total,
            "outside_rate": (self._n_outside / total) if total else 0.0,
            "alarm": self._coverage_alarm,
        }

    def _record_coverage(self, resolved: Sequence[Zone | None]) -> None:
        """Tracks the out-of-layer rate and raises an alarm if it soars.

        Rising edge only, re-armed below a low threshold: an out-of-scope
        population makes the geo features massively missing, which would degrade the
        model without any error surfacing.
        """
        self._n_outside += sum(1 for z in resolved if z is None)
        self._n_inside += sum(1 for z in resolved if z is not None)

        stats = self.coverage()
        if stats["total"] < _COVERAGE_MIN_SAMPLE:
            return
        rate = stats["outside_rate"]
        if rate > _COVERAGE_ALARM_RATE and not self._coverage_alarm:
            self._coverage_alarm = True
            logger.error(
                f"[ALARME] Fine-zone mapping degraded: {rate:.1%} of points "
                f"outside the layer ({stats['outside']}/{stats['total']}), expected ~5%. "
                "The geographic features of the mode choice model are missing "
                "for these decisions."
            )
        elif rate < _COVERAGE_CLEAR_RATE and self._coverage_alarm:
            self._coverage_alarm = False
            logger.info(f"Fine-zone mapping back to {rate:.1%} outside the layer.")

    def __len__(self) -> int:
        return len(self._zones)


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
