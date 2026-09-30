"""Exports the polygon of the 453-commune scope as a WGS84 shapefile for GAMA (ticket 031, G1).

    services/llm-agents/.venv/bin/python scripts/data/gama/export_perimetre_shapefile.py

Produces `services/GAMA/CityTransport/includes/perimetre_453.shp` (+ .shx/.dbf/.prj/.cpg), a single feature:
the dissolved polygon of the four rings of `mobility_core/data/couronne_perimetre.geojson` (EPSG:4326,
like `routes.shp` and `stops.shp`). `Settings.gaml` makes it the world extent
(`geometry shape <- envelope(perimetre_shape_file)`) instead of the envelope of the Tisséo lines,
which left 163 homes of the population outside (scope report of 2026-09-03).

The `includes/` folder is not versioned: this script is the recipe. The attributes carry the
provenance (label, version of the communes table, export date).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "services" / "llm-agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INCLUDES = REPO_ROOT / "services" / "GAMA" / "CityTransport" / "includes"
OUT = INCLUDES / "perimetre_453.shp"


def main() -> int:
    import geopandas as gpd
    from inputs.population.perimeter import PERIMETER_LABEL, PopulationPerimeter

    perimeter = PopulationPerimeter.load()
    version = (perimeter.communes.meta or {}).get("version", "?")
    gdf = gpd.GeoDataFrame(
        {"label": [PERIMETER_LABEL], "communes": [len(perimeter.communes)], "table_v": [str(version)],
         "exporte_le": [date.today().isoformat()]},
        geometry=[perimeter.polygon], crs="EPSG:4326")
    INCLUDES.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT)
    b = perimeter.bbox
    info = {"fichier": str(OUT), "crs": "EPSG:4326", "geom_type": perimeter.polygon.geom_type,
            "bounds": [b.min_lon, b.min_lat, b.max_lon, b.max_lat], "communes": len(perimeter.communes),
            "table_version": version, "date": date.today().isoformat()}
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
