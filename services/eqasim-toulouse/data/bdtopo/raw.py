import pyogrio
import pandas as pd
import os
import geopandas as gpd
import py7zr
import glob
import numpy as np

"""
This stage loads the raw data from the French building registry (BD-TOPO).
"""

def configure(context):
    context.config("data_path")
    context.config("bdtopo_path", "bdtopo_idf")

    context.stage("data.spatial.departments")

def get_department_string(department_id):
    department_id = str(department_id)

    if len(department_id) == 2:
        return "0{}".format(department_id)
    elif len(department_id) == 3:
        return department_id
    else:
        raise RuntimeError("Department identifier should have at least two characters")

def execute(context):
    df_departments = context.stage("data.spatial.departments")
    print("Expecting data for {} departments".format(len(df_departments)))

    source_paths = find_bdtopo("{}/{}".format(context.config("data_path"), context.config("bdtopo_path")))

    df_bdtopo = []
    known_ids = set()

    for source_path in source_paths:
        print("Loading {}".format(source_path.split("/")[-1]))

        if source_path.endswith(".shp"):
            df_buildings = pyogrio.read_dataframe(source_path, columns=["ID", "NB_LOGTS"]).to_crs("EPSG:2154")
            df_buildings["building_id"] = df_buildings["ID"].apply(lambda x: int(x[8:]))
            df_buildings["housing"] = df_buildings["NB_LOGTS"].fillna(0).astype(int)
            df_buildings["centroid"] = df_buildings["geometry"].centroid
            df_buildings = df_buildings.set_geometry("centroid")
        else:
            geometry_path = None

            with py7zr.SevenZipFile(source_path) as archive:
                names = archive.getnames()
                internal_path = [path for path in names if path.endswith(".gpkg")]
                # Fork Toulouse (ticket 031): IGN "TOUSTHEMES_SHP" deliveries have no
                # GeoPackage but a BATI/BATIMENT.* shapefile; only these members are extracted.
                shp_members = [path for path in names if path.endswith("/BATI/BATIMENT.shp")]

                if len(internal_path) == 1:
                    print("  Extracting (GeoPackage) ...")
                    archive.extract(context.path(), [internal_path[0]])
                    geometry_path = "{}/{}".format(context.path(), internal_path[0])
                    layer_kwargs = dict(layer = "batiment", columns = ["cleabs", "nombre_de_logements"])
                    id_column, housing_column = "cleabs", "nombre_de_logements"
                elif len(shp_members) == 1:
                    stem = shp_members[0][:-len(".shp")]
                    members = [path for path in names if path.startswith(stem + ".")]
                    print("  Extracting (shapefile BATI/BATIMENT, %d members) ..." % len(members))
                    archive.extract(context.path(), members)
                    geometry_path = "{}/{}".format(context.path(), shp_members[0])
                    layer_kwargs = dict(columns = ["ID", "NB_LOGTS"])
                    id_column, housing_column = "ID", "NB_LOGTS"
                else:
                    print("  Skipping: No unambiguous geometry source found (gpkg: %d, BATIMENT.shp: %d)!" % (
                        len(internal_path), len(shp_members)))
                    continue

            df_buildings = pyogrio.read_dataframe(geometry_path, **layer_kwargs).to_crs("EPSG:2154")
            df_buildings["building_id"] = df_buildings[id_column].apply(lambda x: int(x[8:]))
            df_buildings["housing"] = df_buildings[housing_column].fillna(0).astype(int)
            df_buildings["centroid"] = df_buildings["geometry"].centroid
            df_buildings = df_buildings.set_geometry("centroid")
            for extracted in glob.glob(geometry_path[:-len(".shp")] + ".*") if geometry_path.endswith(".shp") else [geometry_path]:
                os.remove(extracted)

        print("  Filtering ...")

        initial_count = len(df_buildings)
        df_buildings = df_buildings[df_buildings["housing"] > 0]
        final_count = len(df_buildings)
        print("    {}/{} filtered by dwellings".format(initial_count - final_count, initial_count))

        initial_count = len(df_buildings)
        df_buildings = df_buildings[~df_buildings["building_id"].isin(known_ids)]
        final_count = len(df_buildings)
        print("    {}/{} filtered duplicates".format(initial_count - final_count, initial_count))

        initial_count = len(df_buildings)
        df_buildings = gpd.sjoin(df_buildings, df_departments, predicate="within")
        final_count = len(df_buildings)
        print("    {}/{} filtered spatially".format(initial_count - final_count, initial_count))

        if len(df_buildings) == 0:
            continue

        df_buildings["department_id"] = df_buildings["departement_id"]
        df_buildings = df_buildings.set_geometry("geometry")

        df_bdtopo.append(df_buildings[["building_id", "housing", "department_id", "geometry"]])
        known_ids |= set(df_buildings["building_id"].unique())

    if len(df_bdtopo) == 0:
        raise RuntimeError("[ALARME] BD TOPO: no usable archive in %s (expected: .7z with "
                           "a .gpkg or a BATI/BATIMENT.shp, or extracted shapefiles)" % source_paths)
    df_bdtopo = pd.concat(df_bdtopo)

    for department_id in df_departments["departement_id"].values:
        if np.count_nonzero(df_bdtopo["department_id"] == department_id) == 0:
            raise RuntimeError("[ALARME] BD TOPO: no building for department %s — archive "
                               "missing or empty in bdtopo_path" % department_id)

    return df_bdtopo[["building_id", "housing", "geometry"]]

def find_bdtopo(path):
    archives = sorted(glob.glob("{}/*.7z".format(path)))
    shp_files = sorted(glob.glob("{}/**/BATI/BATIMENT.shp".format(path), recursive=True))
    candidates = archives + shp_files

    if len(candidates) == 0:
        raise RuntimeError("BD TOPO data is not available in {}".format(path))

    return candidates

def validate(context):
    paths = find_bdtopo("{}/{}".format(context.config("data_path"), context.config("bdtopo_path")))
    return sum([os.path.getsize(path) for path in paths])
