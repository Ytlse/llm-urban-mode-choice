import pandas as pd
import os
import zipfile
import pyarrow as pa
import polars as pl
"""
This stage loads the raw data from the French population census.
"""

def configure(context):
    context.stage("data.spatial.codes")

    context.config("data_path")
    context.config("census_path", "rp_2022/RP2022_indcvi.parquet")

COLUMNS_DTYPES = {
    "CANTVILLE":"str",
    "NUMMI":"str",
    "AGED":"str",
    "COUPLE":"str",
    "GS":"str",
    "STAT_GSEC":"str",
    "DEPT":"str",
    "ETUD":"str",
    "IPONDI":"str",
    "IRIS":"str",
    "REGION":"str",
    "SEXE":"str",
    "TACT":"str",
    "TP":"str",
    "TRANS":"str",
    "VOIT":"str",
    "DEROU":"str",
    "PCSL":"str",
    "NA17":"str",
}


def execute(context):
    df_codes = context.stage("data.spatial.codes")
    requested_departements = df_codes["departement_id"].unique().tolist()

    path = "{}/{}".format(context.config("data_path"), context.config("census_path"))

    with context.progress(label="Reading census ...") as progress:
        parquet = (
            pl.scan_parquet(path)
            .select(list(COLUMNS_DTYPES.keys()))
            .with_columns([pl.col(c).cast(pl.Utf8) for c in COLUMNS_DTYPES.keys()])
            .filter(pl.col("DEPT").is_in(requested_departements))
            .collect()
        )
        progress.update(len(parquet))

    return parquet.to_pandas()

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("census_path"))):
        raise RuntimeError("RP 2022 data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("census_path")))
