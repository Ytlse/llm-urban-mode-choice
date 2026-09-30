from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import numba

import data.hts.egt.cleaned
import data.hts.entd.cleaned

import multiprocessing as mp

"""
This stage fuses census data with HTS data.
"""

def configure(context):
    context.config("with_motorcycles", False)

    context.stage("synthesis.population.matched")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.income.selected")
    context.config("extra_enriched_attributes", [])
    context.config("random_seed")

    hts = context.config("hts")
    context.stage("data.hts.selected", alias = "hts")

def execute(context):
    # Select population columns
    df_population = context.stage("synthesis.population.sampled")[[
        "person_id", "household_id",
        "census_person_id", "census_household_id",
        "age", "sex", "employed", "studies",
        "number_of_cars", "number_of_motorcycles", "number_of_vehicles", "use_motorcycle",
        "household_size", "consumption_units",
        "socioprofessional_class", "professional_activity",
        "socioprofessional_class_detail", "employment_sector",
        # Cohort sealing (ticket 028 / population check): these three columns
        # exist in the cleaned census and were dropped here. `iris_id` and
        # `commune_id` attach the household to its commune WITHOUT a geometric lookup;
        # `commute_mode` (RP `TRANS`) is the DECLARED commute mode — a per-person ground
        # truth, exported at the root of the record and NEVER in the prompt.
        "commute_mode", "iris_id", "commune_id",
    ]]

    # Attach matching information
    df_matching = context.stage("synthesis.population.matched")
    df_population = pd.merge(df_population, df_matching, on="person_id", how="left")

    initial_size = len(df_population)
    initial_person_ids = len(df_population["person_id"].unique())
    initial_household_ids = len(df_population["household_id"].unique())

    # Attach person and household attributes from HTS
    df_hts_households, df_hts_persons, _ = context.stage("hts")
    df_hts_persons = df_hts_persons.rename(columns = { "person_id": "hts_id", "household_id": "hts_household_id" })
    df_hts_households = df_hts_households.rename(columns = { "household_id": "hts_household_id" })

    columns = ["hts_id", "hts_household_id", "has_license", "has_pt_subscription", "is_passenger"]
    extra_cols = context.config("extra_enriched_attributes")
    assert isinstance(extra_cols, list), "`extra_enriched_attributes` parameter must be a list"
    columns += extra_cols
    df_population = pd.merge(df_population, df_hts_persons[columns], on="hts_id", how="left")

    df_population = pd.merge(df_population, df_hts_households[[
        "hts_household_id", "number_of_bikes"
    ]], on="hts_household_id", how="left")

    # Attach income
    df_income = context.stage("synthesis.population.income.selected")
    df_population = pd.merge(df_population, df_income[[
        "household_id", "household_income"
    ]], on="household_id", how="left")

    # Check consistency
    final_size = len(df_population)
    final_person_ids = len(df_population["person_id"].unique())
    final_household_ids = len(df_population["household_id"].unique())

    assert initial_size == final_size
    assert initial_person_ids == final_person_ids
    assert initial_household_ids == final_household_ids

    # Add car availability
    df_number_of_cars = df_population[["household_id", "number_of_cars"]].drop_duplicates("household_id")
    # Only adults count towards the household's number of licences (ticket 008, A1.a).
    # Licences inherited by a child from an adult donor switched
    # households from "all" to "some": the car there became "to be shared", even though
    # the supposed extra driver is nine years old.
    df_adult_licenses = df_population[["household_id", "has_license", "age"]].copy()
    df_adult_licenses.loc[df_adult_licenses["age"] < 18, "has_license"] = False
    df_number_of_licenses = df_adult_licenses[["household_id", "has_license"]].groupby("household_id").sum().reset_index().rename(columns = { "has_license": "number_of_licenses" })
    df_car_availability = pd.merge(df_number_of_cars, df_number_of_licenses)

    df_car_availability["car_availability"] = None
    df_car_availability.loc[df_car_availability["number_of_cars"] >= df_car_availability["number_of_licenses"], "car_availability"] = "all"
    df_car_availability.loc[df_car_availability["number_of_cars"] < df_car_availability["number_of_licenses"], "car_availability"] = "some"
    df_car_availability.loc[df_car_availability["number_of_cars"] == 0, "car_availability"] = "none"
    df_car_availability["car_availability"] = df_car_availability["car_availability"].astype("category")

    df_population = pd.merge(df_population, df_car_availability[["household_id", "car_availability"]])

    # Handle motorcycle use if needed (remove use_motorcycle)
    if not context.config("with_motorcycles"):
        df_population.drop(columns=["use_motorcycle"])

    # Add bike availability
    # This is done at the household level and then merged with the persons so that not-matched
    # persons have the same bike availability as their household members.
    df_bike_availability = df_population[["household_id", "number_of_bikes", "household_size"]].drop_duplicates("household_id").dropna()

    df_bike_availability["bike_availability"] = "all"
    df_bike_availability.loc[df_bike_availability["number_of_bikes"] < df_bike_availability["household_size"], "bike_availability"] = "some"
    df_bike_availability.loc[df_bike_availability["number_of_bikes"] == 0, "bike_availability"] = "none"
    df_bike_availability["bike_availability"] = df_bike_availability["bike_availability"].astype("category")

    df_population = pd.merge(df_population, df_bike_availability[["household_id", "bike_availability"]])

    # Add age range for education
    df_population["age_range"] = "higher_education"
    df_population.loc[df_population["age"]<=10,"age_range"] = "primary_school"
    df_population.loc[df_population["age"].between(11,14),"age_range"] = "middle_school"
    df_population.loc[df_population["age"].between(15,17),"age_range"] = "high_school"
    df_population["age_range"] = df_population["age_range"].astype("category")

    # ── Bike ownership: learnt on EMC² 2023, no longer copied from ENTD 2008 ──────
    #
    # Ticket 015, lot 4 (the root cause). What was done here:
    #
    #     P(the person has a bike) = min(1, number_of_bikes / household_size)
    #     then 14.8 % of e-bikes among owners
    #
    # Three stacked errors, all measured against the EMC² Toulouse 2023
    # microdata (ProGEDO lil-1750):
    #
    # 1. `number_of_bikes` is **copied** from the ENTD 2008 household matched to the person, yet
    #    the matching uses NEITHER household size NOR housing type. A
    #    single person thus inherits the 3 bikes of a family of five, and a family of
    #    five the zero bikes of an elderly couple. Result: the total came out roughly right
    #    (53.3 % of owners against ~51 % expected) but the household-size gradient
    #    was **inverted** — 76 % of owners among people living alone against 33 %
    #    observed, 36 % in 4-person households against 65 %.
    # 2. The ENTD variable read is `V1_JNBVELOADT`, the **adult** bikes:
    #    `V1_JNBVELOENF` is never loaded, i.e. 25 % of the fleet ignored and 4.2 % of
    #    households classified "no bike" although they only have children's bikes.
    # 3. 14.8 % is the share of **equipped households** owning an e-bike (8 % / 54 %), not
    #    the e-bike share **of the fleet**, which is 7.7 % (`ML21 / M21`). Hence 1.7× too many e-bikes.
    #
    # What replaces it: the three stages of `mobility_core.bike_ownership`, learnt
    # on EMC² 2023 — but **applied by the post-processing, not here** (ticket 034,
    # lot 2; decision of the repository author of 2026-08-24, applied on 2026-09-04).
    # Lot 4 of ticket 015 had carried the law into this stage; since the
    # post-processing `scripts.data.population.enrich_personal_bike` runs anyway
    # (step 8 of the notebook requires it for `housing_type` as well as for the bike),
    # two implementations of the same law could only drift apart — and they had
    # drifted: their draw keys differ (`household_id` + person identifier
    # here, address + file index there), and they contradicted each other for 4.7 % of the
    # pool. So there is only one law left, and this stage writes **no**
    # `personal_bike` column: the JSON export omits the key, the runtime sees it missing and raises
    # its alarm if the post-processing has not run.
    #
    # Measured before the removal (2026-09-04, pool of 11,329 persons): the final trait
    # changed for only **14 personas**, all without home coordinates — the
    # only ones the post-processing cannot serve, and which the selection excludes
    # anyway. The sealed cohort is bit-for-bit identical (same sha256, same
    # 513 households, same 393 descent swaps).
    #
    # `number_of_bikes` is still computed above for `bike_availability`, which MATSim
    # consumes; it no longer determines `personal_bike`.

    return df_population


# Recoding of `professional_activity` → the persona's `main_occupation`. **Must stay
# identical to `_MAIN_OCCUPATION_FR` in `llm_agents.py`**, which writes the trait into the
# JSON: it is the same variable, read by stage 2 here and displayed there.
_MAIN_OCCUPATION_FR = {
    "full_time_worker": "Travail à plein temps",
    "part_time_worker": "Travail à temps partiel",
    "unemployed":       "Chômeur/recherche d'emploi",
    "retired":          "Retraité",
    "homemaker":        "Personne au foyer",
    "other":            "Personne au foyer",
}


def _main_occupation_fr(row):
    activity = str(row.get("professional_activity", ""))
    if activity == "student":
        return "Scolaire (jusqu'au Bac)" if int(row["age"]) < 18 else "Étudiant"
    if activity == "under14":
        return "Scolaire (jusqu'au Bac)"
    return _MAIN_OCCUPATION_FR.get(activity, "")


