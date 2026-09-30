from tqdm import tqdm
import pandas as pd
import numpy as np
import zipfile

"""
This stage filters out census observations which live or work outside of
Île-de-France.

Fork Toulouse (ticket 031): when the sampling frame is a LIST OF COMMUNES, the census persons
whose commune is "undefined" — they live in a commune of the department without IRIS, which
the RP does not name — cannot be filtered by commune. The stage
`synthesis.population.spatial.home.zones` then draws them a commune without IRIS **from the frame**,
in proportion to population. Keeping them all amounted to pouring the population of all the
department's communes without IRIS into the frame's communes alone: measured on 2026-09-03 on the
six departments of the scope, 17,986 persons for 10,000 requested, 42.5 % in the 3rd ring
(target 15.4 %), 1,682 personas for the ten Aude villages of the frame (2,143 inhabitants). Their weight
is therefore multiplied by the share of the department's non-IRIS population that lives in the frame
(RP 2022: 86.7 % in Haute-Garonne, 9.4 % in Gers, 9.0 % in Tarn, 20.1 % in
Tarn-et-Garonne, 4.0 % in Ariège, 1.0 % in Aude).
"""

def configure(context):
    context.stage("data.census.cleaned")
    context.stage("data.spatial.codes")

    # Explicit setting (config_toulouse.yml): it enters the stage's synpp fingerprint, so the
    # cache is invalidated when it is switched on — a code change alone does not do that.
    context.config("census_undefined_reweighting", True)
    context.config("data_path")
    context.config("population_path", "rp_2022/base-ic-evol-struct-pop-2022_csv.zip")
    context.config("population_csv", "base-ic-evol-struct-pop-2022.CSV")
    context.config("population_year", 22)


def undefined_commune_shares(df_codes, data_path, population_path, population_csv, year):
    """Share, per department, of the population of communes WITHOUT IRIS that is in the frame.

    Reads the RP population aggregated by IRIS (a commune without IRIS has a single IRIS "COM0000").
    Returns ``{departement_id: share}``; 1.0 when the frame contains all communes without IRIS.
    """
    with zipfile.ZipFile("{}/{}".format(data_path, population_path)) as archive:
        with archive.open(population_csv) as f:
            df_pop = pd.read_csv(f, sep = ";", usecols = ["IRIS", "COM", "P%s_POP" % year],
                                 dtype = {"IRIS": str, "COM": str}).rename(columns = {"P%s_POP" % year: "population"})
    df_pop = df_pop[df_pop["IRIS"].str.endswith("0000")]           # communes without IRIS
    df_pop["departement_id"] = df_pop["COM"].str[:2]
    frame_communes = set(df_codes["commune_id"].astype(str))
    frame_departments = set(df_codes["departement_id"].astype(str))
    df_pop = df_pop[df_pop["departement_id"].isin(frame_departments)]
    total = df_pop.groupby("departement_id")["population"].sum()
    in_frame = df_pop[df_pop["COM"].isin(frame_communes)].groupby("departement_id")["population"].sum()
    return {dep: float(in_frame.get(dep, 0.0)) / float(total[dep]) if total.get(dep, 0) > 0 else 1.0
            for dep in frame_departments}


def execute(context):
    df = context.stage("data.census.cleaned")

    # Filter requested codes
    df_codes = context.stage("data.spatial.codes")

    requested_communes = set(df_codes["commune_id"].unique())
    df = df[df["commune_id"].isin(requested_communes) | (df["commune_id"] == "undefined")]

    excess_iris = set(df["iris_id"].unique()) - set(df_codes["iris_id"].unique())
    if not excess_iris == {"undefined"}:
        raise RuntimeError("Found additional IRIS: %s" % excess_iris)

    # Fork Toulouse (ticket 031): persons with an "undefined" commune are weighted by the share
    # of their department's non-IRIS population that lives in the frame (see the header).
    shares = undefined_commune_shares(df_codes, context.config("data_path"), context.config("population_path"),
                                      context.config("population_csv"), str(context.config("population_year")))
    f_undefined = df["commune_id"] == "undefined"
    if not context.config("census_undefined_reweighting"):
        print("Commune frame: reweighting of undefined-commune persons DISABLED by config (census_undefined_reweighting = false)")
    elif f_undefined.any() and any(share < 0.999 for share in shares.values()):
        df = df.copy()
        weight_before = float(df.loc[f_undefined, "weight"].sum())
        factors = df.loc[f_undefined, "departement_id"].astype(str).map(shares).fillna(1.0).astype(float)
        df.loc[f_undefined, "weight"] = df.loc[f_undefined, "weight"] * factors.values
        weight_after = float(df.loc[f_undefined, "weight"].sum())
        print("Commune frame: %d census persons with undefined commune reweighted by the in-frame share of their "
              "departement's non-IRIS population (%s) — summed weight %.0f -> %.0f ; %d persons with a known commune" % (
                  int(f_undefined.sum()), ", ".join("%s %.1f%%" % (dep, 100.0 * share) for dep, share in sorted(shares.items())),
                  weight_before, weight_after, int((~f_undefined).sum())))
    else:
        print("Commune frame: no reweighting of undefined-commune persons (%s)" % (
            "no undefined commune" if not f_undefined.any() else "frame covers all non-IRIS communes"))

    return df
