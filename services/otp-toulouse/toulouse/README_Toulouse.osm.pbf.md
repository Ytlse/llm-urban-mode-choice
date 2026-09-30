# `Toulouse.osm.pbf` — provenance (2026-09-03, ticket 031 part 2, action T1)

| | Before 2026-09-03 | Since 2026-09-03 |
|---|---|---|
| File | `Toulouse_bbox30km_2026-05-06.osm.pbf` (kept here) | `Toulouse.osm.pbf` |
| Footprint | rectangle 1.085/43.336 → 1.815/43.868 (`TOULOUSE_OSM_ROUTES_30K_BBOX`, 73 × 72 km) | exact polygon of the **453 communes** of the EMC² 2023 survey (0.866/43.115 → 1.928/43.954, 86 × 93 km, 5,428 km² of communes) |
| OSM vintage | 2026 (bbbike download of 2026-05-06) | **2022-01-01** (Geofabrik regional pbf files of the eqasim fork) |
| Size / md5 | 88,204,065 B / `cc9520bf0200752d031f65b9f6c3b4ae` | 76,475,294 B / `62d45fe568aa822b794603149d4e492d` |
| Recipe | none in the repository | `scripts/data/population/build_osmnx_perimeter_graph.py` → `data/cache/osmnx/perimetre_453/perimetre_453.osm.pbf` (osmium extract --polygon, no download) |

The folder actually loaded by the OTP instances of `docker-compose.yml` is `data/gtfs/`
(mount `/var/otp/toulouse`, `--load`): the reference copy there is identical, and the former extract
with its `graph.obj` are archived in `data/gtfs/archives/2026-09-03_pre_perimetre_453/`.

Why: three TER stations and the 3rd-ring homes of the v4 sealed population were outside
the rectangle — OTP could not attach them to the street graph ("Couldn't link"). Limitation to
be aware of: the road network goes from 2026 to 2022 (freshness noted in ticket 031 § 1.0); a 2026 extract
of the same polygon requires a regional download (~270 MB), which is not done without agreement.

`dvc` is not installed: the neighbouring `.dvc` is written by hand (md5 and size recomputed).
