"""export_zone_layer.py — Fine-zone resource embedded for the runtime.

The four geographic variables of the mode-choice model (`od_km`, `density_*`,
`dist_center_*`) weigh heavily in its importances, and all derive from the same
prerequisite: knowing in **which fine zone** a point falls. At training time this
information is given by the survey (`D3`/`D7`); in simulation there are only
coordinates. The spatial join must therefore be replayed at runtime — hence this
resource (ticket 005 §2.1, action A7).

What the script writes into `mobility_core/data/`:

- `zf_zones.gpkg` — the 785 fine-zone polygons, with per zone the Lambert 93
  centroid, the area, the household density and the distance to the city core;
- `zf_zones.meta.json` — provenance (source layer and its fingerprint) and the
  geographic reference, copied as is from `build_geo`.

**Values are not recomputed here.** The script imports `build_geo` from the
training-set builder: density, distance to the centre and centroids are
therefore identical by construction to those seen at training time. Any other
approach (reimplementing density, rereading the shapefile its own way) would create two
competing definitions of the same variable — exactly the defect that
`feature_spec.json` exists to prevent.

Why a derived resource rather than the source shapefile: `data/PROGEDO 2023`
contains the restricted-access microdata (lil-1750), is not versioned and is
not mounted in the `controller` container. The layer exported here carries only
zone-level aggregates, and lives under `mobility_core/`, already mounted wherever the resolver
runs.

Usage:
    python -m scripts.progedo_logit.export_zone_layer [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

from scripts.progedo_logit.build_mode_choice_dataset import (
    build_geo,
    find_project_root,
    load_raw,
)

# Layer name expected by the resolver (packages/mobility_core/src/mobility_core/zone_resolver.py).
LAYER_NAME = "zf"

# Columns of the resource. `ZF` is the key, the next five are all that
# the resolver needs to produce the six geographic features of the spec.
COLUMNS = ["ZF", "XL93", "YL93", "SURF_M2", "density_hh_km2", "dist_center_km"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_layer(sig_zf: Path, men) -> tuple[gpd.GeoDataFrame, dict]:
    """Polygon layer enriched with the attributes of `build_geo`.

    The geometry comes from the shapefile, the attributes from `build_geo`: the
    two are joined on `ZF` rather than recomputed, so that the resource and the training
    set cannot diverge.
    """
    geo, xys, ref = build_geo(sig_zf, men)

    layer = gpd.read_file(sig_zf)[["ZF", "geometry"]].copy()
    layer["ZF"] = layer["ZF"].astype(str).str.strip()

    attrs = xys.join(geo)
    out = layer.merge(attrs, left_on="ZF", right_index=True, how="left")

    missing = out["XL93"].isna().sum()
    if missing:
        raise SystemExit(
            f"{missing} zones without centroid after the join: the ZF key does not match "
            "between the shapefile and build_geo."
        )
    if len(out) != ref["n_zones"]:
        raise SystemExit(
            f"{len(out)} polygons for {ref['n_zones']} expected zones: the layer has changed."
        )

    # Density is legitimately missing for zones with no surveyed household. It is
    # not imputed: the booster routes missing values natively, and a 0
    # would mean « empty zone », which is false.
    n_no_density = int(out["density_hh_km2"].isna().sum())
    print(f"Zones with no surveyed household (density missing, not imputed): {n_no_density}")

    return out[COLUMNS + ["geometry"]], ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: mobility_core/data/)")
    args = parser.parse_args()

    root = find_project_root()
    progedo_dir = root / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV" / "fichiers_standards"
    sig_zf = (root / "data" / "PROGEDO 2023" / "lil-1750-Documentation" / "SIG"
              / "EMC2_Toulouse_2023_ZF_26052023.shp")
    out_dir = args.out_dir or (root / "packages" / "mobility_core" / "src" / "mobility_core" / "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, men, _ = load_raw(progedo_dir)
    layer, ref = build_layer(sig_zf, men)

    gpkg_path = out_dir / "zf_zones.gpkg"
    meta_path = out_dir / "zf_zones.meta.json"

    # Full rewrite: without prior deletion, pyogrio adds a second
    # layer of the same name instead of replacing the first.
    gpkg_path.unlink(missing_ok=True)
    layer.to_file(gpkg_path, layer=LAYER_NAME, driver="GPKG")

    meta = {
        "layer": LAYER_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "shapefile": sig_zf.name,
            "sha256": sha256(sig_zf),
            "survey": "EMC² Toulouse 2023 (ProGEDO / lil-1750)",
        },
        # Copied from build_geo, and therefore comparable one-to-one with the
        # `geo_reference` block of feature_spec.json: the resolver refuses to serve a
        # layer and a model that do not refer to the same city core (cf. A9).
        "geo_reference": ref,
        "columns": COLUMNS,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    print(f"\nWritten:\n - {gpkg_path} ({len(layer)} zones, layer '{LAYER_NAME}')"
          f"\n - {meta_path}")


if __name__ == "__main__":
    main()
