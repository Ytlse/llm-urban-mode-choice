import os
import json
import pandas as pd
import zipfile

"""
This stages loads a file containing all spatial codes in France and how
they can be translated into each other. These are mainly IRIS, commune,
departement and région.

Fork Toulouse (tickets 026 and 031): the sampling frame can be a LIST OF COMMUNES
(`communes`, or `communes_file` — a `commune_couronne.json` file from the llm-agents-gama repository),
intersected with `departments`. The stage logs the retained frame per department and rejects
a requested commune that the IRIS reference data does not know: a merged or wrongly
coded commune would otherwise silently drop out of the frame.
"""

def configure(context):
    context.config("data_path")

    context.config("regions", [11])
    context.config("departments", [])
    context.config("communes", [])
    # Fork Toulouse: path of a `commune_couronne.json` (key `communes[].insee`) used
    # as the list of communes when `communes` is empty. Empty string = no file.
    context.config("communes_file", "")
    context.config("codes_path", "codes_2024/reference_IRIS_geo2024.zip")
    context.config("codes_xlsx", "reference_IRIS_geo2024.xlsx")


def load_communes_file(path):
    """INSEE codes (5 characters) of a `commune_couronne.json` (mobility_core/data)."""
    with open(path, encoding = "utf-8") as f:
        payload = json.load(f)
    rows = payload.get("communes") or []
    if not rows:
        raise RuntimeError("communes_file %s contains no commune" % path)
    return sorted({str(row["insee"]).zfill(5) for row in rows})


def apply_commune_frame(df_codes, requested_communes, requested_departments):
    """Restricts the reference data to the requested communes and returns (df, journal).

    The journal counts the retained communes per department (expected for the full EMC²
    scope: 31 → 346, 32 → 38, 81 → 27, 82 → 22, 09 → 10, 11 → 10) and lists the
    requested communes missing from the reference data — after the departmental restriction, a
    commune of a non-requested department is not "missing", it is outside the frame.
    """
    requested = sorted({str(c).zfill(5) for c in requested_communes})
    known = set(df_codes["commune_id"].astype(str))
    if requested_departments:
        in_scope = [c for c in requested if c[:2] in set(requested_departments)
                    or c[:3] in set(requested_departments)]
        out_of_departments = len(requested) - len(in_scope)
    else:
        in_scope, out_of_departments = requested, 0
    unknown = sorted(c for c in in_scope if c not in known)
    df_codes = df_codes[df_codes["commune_id"].astype(str).isin(set(in_scope))]
    per_department = (df_codes.drop_duplicates("commune_id")["departement_id"].astype(str)
                      .value_counts().sort_index().to_dict())
    journal = {
        "communes_demandees": len(requested),
        "hors_departements_demandes": out_of_departments,
        "communes_retenues": int(df_codes["commune_id"].nunique()),
        "iris_retenus": int(df_codes["iris_id"].nunique()),
        "par_departement": per_department,
        "inconnues": unknown,
    }
    return df_codes, journal

def execute(context):
    # Load IRIS registry
    with zipfile.ZipFile(
        "{}/{}".format(context.config("data_path"), context.config("codes_path"))) as archive:
        with archive.open(context.config("codes_xlsx")) as f:
            df_codes = pd.read_excel(f,
                skiprows = 5, sheet_name = "Emboitements_IRIS",dtype={"CODE_IRIS":str,"DEPCOM":str}
            )[["CODE_IRIS", "DEPCOM", "DEP", "REG"]].rename(columns = {
                "CODE_IRIS": "iris_id",
                "DEPCOM": "commune_id",
                "DEP": "departement_id",
                "REG": "region_id"
            }).fillna('0')

    df_codes["iris_id"] = df_codes["iris_id"].astype("category")
    df_codes["commune_id"] = df_codes["commune_id"].astype("category")
    df_codes["departement_id"] = df_codes["departement_id"].astype("category")
    df_codes["region_id"] = df_codes["region_id"].astype(int)

    # Filter zones
    requested_regions = list(map(int, context.config("regions")))
    requested_departments = list(map(str, context.config("departments")))

    if len(requested_regions) > 0:
        df_codes = df_codes[df_codes["region_id"].isin(requested_regions)]

    if len(requested_departments) > 0:
        df_codes = df_codes[df_codes["departement_id"].isin(requested_departments)]

    # Fork Toulouse: sampling frame = list of communes (ticket 026), logged per
    # department and checked (ticket 031).
    requested_communes = list(map(str, context.config("communes")))
    communes_file = context.config("communes_file")
    if len(requested_communes) == 0 and communes_file:
        requested_communes = load_communes_file(communes_file)
        print("Sampling frame: %d communes read from %s" % (len(requested_communes), communes_file))

    if len(requested_communes) > 0:
        df_codes, journal = apply_commune_frame(df_codes, requested_communes, requested_departments)
        print("Sampling frame: %d communes retained out of %d requested (%d outside the requested "
              "departments), %d IRIS; per department: %s" % (
                  journal["communes_retenues"], journal["communes_demandees"],
                  journal["hors_departements_demandes"], journal["iris_retenus"],
                  ", ".join("%s %d" % kv for kv in journal["par_departement"].items())))
        if journal["inconnues"]:
            # A requested commune that the reference data ignores is not a detail: it leaves
            # the frame without a trace, and the resulting population believes it is compliant.
            raise RuntimeError(
                "[ALARME] %d requested commune(s) missing from the IRIS reference data %s: %s — "
                "outdated INSEE codes (commune merger?) or reference data to be updated" % (
                    len(journal["inconnues"]), context.config("codes_path"), journal["inconnues"]))
        if journal["communes_retenues"] == 0:
            raise RuntimeError("[ALARME] empty sampling frame: none of the %d requested communes "
                               "is in the departments %s" % (
                                   journal["communes_demandees"], requested_departments))

    df_codes["iris_id"] = df_codes["iris_id"].cat.remove_unused_categories()
    df_codes["commune_id"] = df_codes["commune_id"].cat.remove_unused_categories()
    df_codes["departement_id"] = df_codes["departement_id"].cat.remove_unused_categories()

    return df_codes

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("codes_path"))):
        raise RuntimeError("Spatial reference codes are not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("codes_path")))
