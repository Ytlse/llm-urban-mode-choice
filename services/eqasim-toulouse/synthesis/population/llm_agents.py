import hashlib
import json
import math
import os
import uuid
from collections import Counter
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
from faker import Faker
from shapely.geometry import Point, box
from shapely.ops import nearest_points
import shapely.wkt

"""
Export the synthetic population in the JSON format used by the LLM-agents GAMA simulation.
"""

# Fixed namespace for deterministic UUID generation
_UUID_NAMESPACE = uuid.UUID("e4a51200-0000-0000-0000-000000000000")

_fake = Faker("fr_FR")


# ── Ring of residence (ticket 021, batch 5 — stage B) ──────────────────────
#
# The trait is set HERE, and not in `enriched.py`, for a material reason: this
# module SNAPS the locations outside the OTP polygon onto its edge, and `identity.home`
# carries the POST-SNAP coordinates (cf. the update of `home_location` further down).
# `enriched.py` works on `spatial.home.locations`, that is BEFORE the snap: setting
# the ring there would make it diverge from that of the residence the journal uses, for any
# snapped persona. A silent disagreement between two definitions is exactly what
# ticket 021 fixes; reproducing it while fixing it would be ironic.
#
# The resources (`zf_zones.gpkg`, `zf_couronne.json`) come from `mobility_core/data`, mounted
# in the eqasim service since ticket 015. Their absence is NOT silent: the
# stage raises, because a population without a ring makes the per-zone scoring fall back on
# a ring guessed from the distance — the original bug.
_RESIDENCE_ZONES = None


def _residence_resolver():
    """Loads once the fine-zone layer and the ring table."""
    global _RESIDENCE_ZONES
    if _RESIDENCE_ZONES is None:
        from mobility_core.residence_zone import CouronneTable
        from mobility_core.zone_resolver import ZoneResolver
        _RESIDENCE_ZONES = (ZoneResolver.load(), CouronneTable.load())
    return _RESIDENCE_ZONES


def _residence_traits(home_location) -> dict:
    """`residence_zone` / `residence_commune` / `residence_insee` of a residence.

    Three distinct writings, like stage D (`enrich_residence_zone.py`): a
    ring within the study area; `hors périmètre` for a residence known and outside, which
    is NOT a ring and has no EMC² target; nothing at all without coordinates —
    asserting « outside » about someone we know nothing about would be an invention. And the
    commune is never deduced from a sector, which covers several communes.
    """
    from mobility_core.population_reference import OUT_OF_PERIMETER

    if not home_location:
        return {}
    lat, lon = home_location.get("lat"), home_location.get("lon")
    if lat is None or lon is None:
        return {}
    resolver, table = _residence_resolver()
    zone = resolver.resolve(lat, lon)
    if zone is None:
        return {"residence_zone": OUT_OF_PERIMETER}
    couronne = table.couronne_of_zf(zone.zf)
    if couronne is None:
        return {}
    out = {"residence_zone": couronne}
    commune = table.commune_of_zf(zone.zf)
    if commune is not None:
        out["residence_insee"], out["residence_commune"] = commune
    return out


def _flag(value) -> bool:
    """Boolean of an HTS field, NaN included.

    `bool(nan)` is `True`: a simple `bool(row.get(col, False))` therefore turns
    any unmatched person into a licence holder. The missing value is
    « no » (ticket 008, A1.a).
    """
    return bool(pd.notna(value) and value)

# Socioprofessional class (INSEE PCS-2020 compatible, 8-class)
_SPC_LABEL = {
    1: "Farmer",
    2: "Craftsperson or Shop Owner",
    3: "Executive or Higher Intellectual Professional",
    4: "Intermediate Professional",
    5: "Employee",
    6: "Manual Worker",
    7: "Retired",
    8: "Other Inactive",
}

_SPC_OCCUPATION_TITLE = {
    1: "Farmer",
    2: "Shop Owner",
    3: "Executive",
    4: "Technician",
    5: "Office Worker",
    6: "Factory Worker",
    7: "Retired",
    8: "Without Occupation",
}

# Employment sector → organization type
_SECTOR_ORGANIZATION = {
    "agriculture": "Agricultural Business",
    "energy_water_waste_mining": "Utilities Company",
    "food_beverages_tobacco": "Food & Beverage Company",
    "refining": "Petrochemical Company",
    "electrical_electronics_ict_machinery": "Technology Company",
    "transport_equipment": "Automotive Manufacturer",
    "other_manufacturing": "Manufacturing Company",
    "construction": "Construction Company",
    "retail_auto": "Retail Chain",
    "transport_storage": "Logistics Company",
    "accommodation_food_services": "Hospitality Business",
    "information_communication": "Media & Communications Company",
    "finance_insurance": "Financial Institution",
    "real_estate": "Real Estate Agency",
    "scientific_technical_support_services": "Consulting Firm",
    "public_admin_education_health": "Public Institution",
    "arts_recreation_other_services": "Cultural Organization",
    "not_applicable": "",
}

# Household monthly income (€) → text label
_INCOME_THRESHOLDS = [
    (1500,  "Very Low"),
    (2500,  "Low"),
    (3500,  "Medium"),
    (5000,  "Medium-High"),
    (float("inf"), "High"),
]

# Professional activity → readable label
_PROFESSIONAL_ACTIVITY_LABEL = {
    "full_time_worker":  "Full-Time Worker",
    "part_time_worker":  "Part-Time Worker",
    "unemployed":        "Unemployed",
    "retired":           "Retired",
    "student":           "Student",
    "under14":           "Child (under 14)",
    "homemaker":         "Homemaker",
    "other":             "Other Inactive",
}

# Professional activity → French calibration label (7 categories)
# Main occupation, in ENGLISH since ticket 074: this trait is served to the model
# in the persona narrative. The link with the CEREMA survey goes through an explicit
# mapping table (`scripts/synthesis/frames.py:OCCUPATION_MAP`), which translates these
# labels into the identifiers of `cerema_values.yaml` (`actif_temps_plein`…). Those
# identifiers do not move: they cite the source, they do not describe it.
_MAIN_OCCUPATION_LABEL = {
    "full_time_worker": "Full-time worker",
    "part_time_worker": "Part-time worker",
    "unemployed":       "Unemployed / job seeker",
    "retired":          "Retired",
    "homemaker":        "Homemaker",
    "other":            "Homemaker",
}

# Activity purpose → label of `travel_purposes`. READ from `mobility_core` (mounted in
# the image), no longer copied: this table lived in duplicate with that of `align_minor_traits`,
# and the two diverged at the first translation (ticket 074).
from mobility_core.population_reference import PURPOSE_LABEL as _PURPOSE_LABEL


def _stable_hash(value: int) -> int:
    """Deterministic integer hash independent of PYTHONHASHSEED."""
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16)


def _fetch_otp_polygon(otp_endpoint: str):
    """Fetch OTP graph coverage polygon. Returns None if unavailable.

    Tries OTP1 REST API first, then OTP2 GraphQL (stops convex hull).
    """
    # OTP1: /otp/routers/default returns polygon or boundingBox directly.
    try:
        url = f"{otp_endpoint}/otp/routers/default"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        raw = data.get("polygon")
        if isinstance(raw, str):
            return shapely.wkt.loads(raw)
        if isinstance(raw, dict):
            from shapely.geometry import shape
            return shape(raw)
        bb = data.get("boundingBox", {})
        if bb:
            return box(bb["minLon"], bb["minLat"], bb["maxLon"], bb["maxLat"])
    except Exception:
        pass

    # OTP2: query stops via GraphQL and return their convex hull.
    graphql_url = f"{otp_endpoint}/otp/transmodel/v3"
    query = "{ stopPlaces { geometry { type coordinates } } }"
    try:
        req = urllib.request.Request(
            graphql_url,
            data=json.dumps({"query": query}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        stops = (data.get("data") or {}).get("stopPlaces") or []
        coords = []
        for sp in stops:
            geom = sp.get("geometry") or {}
            if geom.get("type") == "Point" and geom.get("coordinates"):
                lon, lat = geom["coordinates"]
                coords.append((lon, lat))
        if len(coords) >= 3:
            from shapely.geometry import MultiPoint
            polygon = MultiPoint(coords).convex_hull.buffer(0.02)
            print(f"[llm_agents] OTP2 polygon built from {len(coords)} stops (convex hull + 0.02° buffer)")
            return polygon
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return box(min(lons) - 0.02, min(lats) - 0.02, max(lons) + 0.02, max(lats) + 0.02)
    except Exception as e:
        print(f"[llm_agents] Warning: could not fetch OTP polygon from {otp_endpoint}: {e}")
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _snap_to_polygon(lon: float, lat: float, polygon) -> tuple[float, float, float]:
    """Return (snapped_lon, snapped_lat, extra_distance_m). Zero distance if inside."""
    pt = Point(lon, lat)
    if polygon.contains(pt):
        return lon, lat, 0.0
    snapped = nearest_points(polygon.exterior, pt)[0]
    dist_m = _haversine_m(lat, lon, snapped.y, snapped.x)
    return snapped.x, snapped.y, dist_m


def _generate_name(person_id: int, sex: str) -> str:
    """Name of the persona, DRAWN DETERMINISTICALLY from its `person_id`.

    `person_id` was already received as a parameter but was not used: the draw came from
    the global state of Faker, hence from the call order. Two generations of the SAME pool
    returned two sets of names — measured on 2026-09-14 by comparing v5 and v6: 11,329
    personas out of 11,329 had changed name, whereas their activity chains, their
    residences and all their demographic margins were identical to the hundredth.

    The first name is served to the model in the persona narrative. It is excluded from the key of the decision
    cache (`_TRAITS_EXCLUDED_FROM_CACHE_KEY`), so it skews no measurement — but
    a cohort that changes names while nothing else moves is not reproducible, and
    two execution traces stop being readable together.

    The seed goes through `_stable_hash`, which does not depend on `PYTHONHASHSEED`.
    """
    from faker import Faker

    tireur = Faker("fr_FR")
    tireur.seed_instance(_stable_hash(person_id) % (2 ** 32))
    if sex == "male":
        return tireur.name_male()
    if sex == "female":
        return tireur.name_female()
    return tireur.name()


def _income_label(income) -> str:
    if income is None or (isinstance(income, float) and np.isnan(income)):
        return ""
    for threshold, label in _INCOME_THRESHOLDS:
        if income < threshold:
            return label
    return "High"


_PERSONALITY_TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "personality_templates.json"
)
_personality_templates: list | None = None


def _load_personality_templates() -> list:
    global _personality_templates
    if _personality_templates is None:
        with open(_PERSONALITY_TEMPLATES_PATH, encoding="utf-8") as f:
            _personality_templates = json.load(f)
    return _personality_templates


def _pick_personality(person_id: int) -> dict:
    """Deterministically pick a personality template for a given person_id."""
    templates = _load_personality_templates()
    return templates[_stable_hash(person_id) % len(templates)]


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    # Ticket 031 § 1.1: the commune and the IRIS of the HOUSEHOLD, as the draw of the residence
    # set them. The census columns `commune_id` / `iris_id` are « undefined » for
    # 36% of households (IRIS of fewer than 200 inhabitants, communes without IRIS): the
    # `home.zones` stage resolved them before drawing the address, it is the one that is authoritative.
    context.stage("synthesis.population.spatial.home.zones")
    context.config("output_path")
    context.config("output_prefix", "ile_de_france_")
    context.config("generate_personality_traits", False)


def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")
    generate_personality_traits = context.config("generate_personality_traits")

    otp_endpoint = os.environ.get("OTP_ENDPOINT", "")
    otp_polygon = _fetch_otp_polygon(otp_endpoint) if otp_endpoint else None
    if otp_polygon is None:
        print("[llm_agents] OTP polygon unavailable — out-of-graph detection skipped")

    # ── Persons ────────────────────────────────────────────────────────────────
    df_persons = context.stage("synthesis.population.enriched")

    # ── Commune and IRIS of the household (ticket 031 § 1.1) ───────────────────────────
    # Before: `household.commune_id` copied the census column, « undefined » for
    # 4,292 of the 11,922 people of the v3 pool (36%). The runtime must filter by commune of
    # residence (part 2, § 2.1): the value now comes from the zone draw, which sets a
    # commune and an IRIS for ALL households. The census column remains the fallback, counted.
    df_zones = context.stage("synthesis.population.spatial.home.zones")[
        ["household_id", "commune_id", "iris_id"]
    ].rename(columns={"commune_id": "zone_commune_id", "iris_id": "zone_iris_id"})
    df_persons = pd.merge(df_persons, df_zones, on="household_id", how="left")
    _n_census_undefined = int((df_persons["commune_id"].astype(str) == "undefined").sum()) \
        if "commune_id" in df_persons else len(df_persons)
    _n_zone_missing = int(df_persons["zone_commune_id"].isna().sum())
    for _col, _zone_col in (("commune_id", "zone_commune_id"), ("iris_id", "zone_iris_id")):
        _zone = df_persons[_zone_col].astype(object)
        if _col in df_persons:
            _census = df_persons[_col].astype(object)
            df_persons[_col] = _zone.where(_zone.notna(), _census)
        else:
            df_persons[_col] = _zone
    print(f"[llm_agents] Commune du ménage : {_n_census_undefined} personnes « undefined » au "
          f"recensement, {len(df_persons) - _n_zone_missing}/{len(df_persons)} résolues par le tirage "
          f"de zone (home.zones)" + (f" — {_n_zone_missing} ménage(s) sans zone : repli recensement"
                                   if _n_zone_missing else ""))
    if _n_zone_missing:
        print(f"[llm_agents] WARNING [ALARME] {_n_zone_missing} personne(s) dont le ménage n'a pas "
              "de zone de domicile — le filtre runtime par commune les écartera")

    # ── Activities ─────────────────────────────────────────────────────────────
    df_activities = context.stage("synthesis.population.activities")[
        ["person_id", "activity_index", "purpose", "start_time", "end_time", "is_first", "is_last"]
    ]

    # ── Spatial locations (EPSG:2154 → WGS84) ─────────────────────────────────
    df_locations = context.stage("synthesis.population.spatial.locations")[
        ["person_id", "activity_index", "geometry"]
    ]
    df_locations = gpd.GeoDataFrame(df_locations).to_crs("EPSG:4326")
    df_locations["lon"] = df_locations.geometry.x
    df_locations["lat"] = df_locations.geometry.y

    # Merge locations into activities
    df_activities = pd.merge(
        df_activities,
        df_locations[["person_id", "activity_index", "lon", "lat"]],
        on=["person_id", "activity_index"],
        how="left",
    )

    # Group activities by person for fast lookup
    activities_by_person = {
        pid: grp.sort_values("activity_index")
        for pid, grp in df_activities.groupby("person_id")
    }

    # ── Build JSON ─────────────────────────────────────────────────────────────
    result = []
    # Sealing fields missing from a row (cf. `_root_field`): counted, reported
    # at the end of the stage. A column dropped upstream must show in the journal.
    _missing_root_fields: Counter = Counter()
    _immobiles = 0                # journées sans activité hors domicile, GARDÉES (ticket 029)
    _immobiles_sans_domicile = 0  # same, without residence coordinates: discarded and counted

    for _, row in df_persons.iterrows():
        pid = int(row["person_id"])
        sex = str(row["sex"])
        name = _generate_name(pid, sex)

        spc = int(row["socioprofessional_class"]) if not pd.isna(row.get("socioprofessional_class")) else 8
        sector = str(row.get("employment_sector", "not_applicable"))
        income = row.get("household_income")

        occ_title = _SPC_OCCUPATION_TITLE.get(spc, "")
        occ_org = _SECTOR_ORGANIZATION.get(sector, "")
        spc_label = _SPC_LABEL.get(spc, "")
        income_text = _income_label(income)
        pro_act = str(row.get("professional_activity", ""))
        pro_act_label = _PROFESSIONAL_ACTIVITY_LABEL.get(pro_act, pro_act)

        # ── Activities list ────────────────────────────────────────────────────
        raw_acts = activities_by_person.get(pid, pd.DataFrame())

        # IMMOBILE person: no activity outside the residence. They were discarded here, which
        # emptied the population of its immobile people — EMC² 2023 counts 10.6% of those aged 5 and over
        # (ticket 029). They now stay, with a « home 0 → 86,400 s » day and the
        # root flag `immobile`: no trip, no LLM call, but they count in the
        # population, its margins and its size.
        immobile = bool(raw_acts.empty or raw_acts[raw_acts["purpose"] != "home"].empty)

        activities_list = []
        home_location = None

        if immobile:
            _home_rows = raw_acts[raw_acts["purpose"] == "home"] if not raw_acts.empty else raw_acts
            if _home_rows.empty or pd.isna(_home_rows.iloc[0]["lon"]) or pd.isna(_home_rows.iloc[0]["lat"]):
                # Without residence coordinates, the person cannot be placed anywhere: discarded,
                # and counted — an immobile person without an address is an upstream anomaly, not a normal case.
                _immobiles_sans_domicile += 1
                continue
            _h = _home_rows.iloc[0]
            home_location = {"lon": float(_h["lon"]), "lat": float(_h["lat"])}
            activities_list.append({
                "id": str(uuid.uuid5(_UUID_NAMESPACE, f"{pid}_0")),
                "scheduled_start_time": None,
                "start_time": 0.0,
                "end_time": 86400.0,
                "purpose": "home",
                "location": {"lon": home_location["lon"], "lat": home_location["lat"]},
            })
            _immobiles += 1
        elif not raw_acts.empty:
            acts = raw_acts.copy().reset_index(drop=True)

            # 1. Resolve home coordinates strictly from existing home activities.
            #    home_location is the single source of truth: it feeds both identity.home
            #    and the coordinates of any synthetic home activity added below.
            home_rows = acts[acts["purpose"] == "home"]
            if not home_rows.empty:
                _h = home_rows.iloc[0]
                home_lon = None if pd.isna(_h["lon"]) else float(_h["lon"])
                home_lat = None if pd.isna(_h["lat"]) else float(_h["lat"])
                home_location = {"lon": home_lon, "lat": home_lat} if home_lon is not None else None
            else:
                home_lon = None
                home_lat = None

            # 2. Ensure first activity is home
            if acts.iloc[0]["purpose"] != "home":
                first_st = acts.iloc[0]["start_time"]
                home_end = float(first_st) if not pd.isna(first_st) else 0.0
                if home_end > 0.0:
                    # There is time before the first activity: prepend a home covering [0, home_end].
                    acts.at[0, "start_time"] = home_end
                    acts.at[0, "is_first"] = False
                    prepend = pd.DataFrame([{
                        "person_id": pid,
                        "activity_index": int(acts["activity_index"].min()) - 1,
                        "purpose": "home",
                        "start_time": np.nan,
                        "end_time": home_end,
                        "is_first": True,
                        "is_last": False,
                        "lon": home_lon,
                        "lat": home_lat,
                    }])
                    acts = pd.concat([prepend, acts], ignore_index=True)
                # else: first non-home activity starts at t=0 (person away since midnight).
                # No room for a home prefix; keep acts[0] as is_first with start=0.0.

            # 3. Ensure last activity is home
            if acts.iloc[-1]["purpose"] != "home":
                last_st = float(acts.iloc[-1]["start_time"]) if not pd.isna(acts.iloc[-1]["start_time"]) else 0.0
                last_et = float(acts.iloc[-1]["end_time"]) if not pd.isna(acts.iloc[-1]["end_time"]) else 86400.0
                if last_st > 86400.0 or last_et >= 86400.0:
                    # Last activity already closes at or after midnight: day is naturally closed.
                    # Adding a home with start=86400 and end=86400 would give zero duration.
                    pass
                else:
                    home_start = last_et
                    acts.at[len(acts) - 1, "end_time"] = home_start
                    acts.at[len(acts) - 1, "is_last"] = False
                    append_df = pd.DataFrame([{
                        "person_id": pid,
                        "activity_index": int(acts["activity_index"].max()) + 1,
                        "purpose": "home",
                        "start_time": home_start,
                        "end_time": np.nan,
                        "is_first": False,
                        "is_last": True,
                        "lon": home_lon,
                        "lat": home_lat,
                    }])
                    acts = pd.concat([acts, append_df], ignore_index=True)

            # 4. Enforce spatial continuity: first and last share home (lat, lon)
            if home_lon is not None:
                acts.at[0, "lon"] = home_lon
                acts.at[0, "lat"] = home_lat
                acts.at[len(acts) - 1, "lon"] = home_lon
                acts.at[len(acts) - 1, "lat"] = home_lat

            # 4.5. Merge consecutive activities with identical purpose and location.
            # Avoids routing requests where origin == destination (e.g. two consecutive
            # work sessions at the same workplace, or duplicate home entries).
            _acts_list = acts.to_dict("records")
            _merged: list[dict] = []
            i = 0
            while i < len(_acts_list):
                cur = dict(_acts_list[i])
                j = i + 1
                while j < len(_acts_list):
                    nxt = _acts_list[j]
                    if nxt["purpose"] != cur["purpose"]:
                        break
                    cur_lon, cur_lat = cur.get("lon"), cur.get("lat")
                    nxt_lon, nxt_lat = nxt.get("lon"), nxt.get("lat")
                    both_null = (cur_lon is None or pd.isna(cur_lon)) and (nxt_lon is None or pd.isna(nxt_lon))
                    both_close = (
                        cur_lon is not None and not pd.isna(cur_lon) and
                        nxt_lon is not None and not pd.isna(nxt_lon) and
                        abs(float(cur_lon) - float(nxt_lon)) < 1e-5 and
                        abs(float(cur_lat) - float(nxt_lat)) < 1e-5
                    )
                    if not (both_null or both_close):
                        break
                    cur["end_time"] = nxt["end_time"]
                    cur["is_last"] = nxt["is_last"]
                    j += 1
                _merged.append(cur)
                i = j
            if len(_merged) < len(_acts_list):
                print(f"[llm_agents] person={pid}: merged {len(_acts_list) - len(_merged)} duplicate activity pair(s) ({len(_acts_list)} → {len(_merged)})")
            acts = pd.DataFrame(_merged).reset_index(drop=True)

            # 5. Build activities list (0.0 for first, 86400.0 for last — set in activities.py)
            for _, act in acts.iterrows():
                st = act["start_time"]
                et = act["end_time"]

                start_time = float(st) if not pd.isna(st) else 0.0
                end_time   = float(et) if not pd.isna(et) else 86400.0

                lon_v = None if pd.isna(act["lon"]) else float(act["lon"])
                lat_v = None if pd.isna(act["lat"]) else float(act["lat"])

                act_entry = {
                    "id": str(uuid.uuid5(_UUID_NAMESPACE, f"{pid}_{int(act['activity_index'])}")),
                    "scheduled_start_time": None,
                    "start_time": start_time,
                    "end_time": end_time,
                    "purpose": str(act["purpose"]),
                    "location": {"lon": lon_v, "lat": lat_v}
                }
                activities_list.append(act_entry)

            # Update home_location from the snapped first home activity so that
            # identity.home reflects the post-snap coordinates (used as last_location at init).
            if otp_polygon is not None and home_location is not None:
                first_home = next(
                    (a for a in activities_list if a["purpose"] == "home" and a["location"]["lon"] is not None),
                    None,
                )
                if first_home:
                    home_location = {"lon": first_home["location"]["lon"], "lat": first_home["location"]["lat"]}

        age_val = int(row["age"])

        if pro_act == "student":
            main_occupation = "Pupil (up to Baccalaureate)" if age_val < 18 else "Student"
        elif pro_act == "under14":
            main_occupation = "Pupil (up to Baccalaureate)"
        else:
            main_occupation = _MAIN_OCCUPATION_LABEL.get(pro_act, "")

        # Order of FIRST APPEARANCE in the chain, and not `list({...})`.
        #
        # A set comprehension converted into a list returns an order that depends on the hashing
        # of strings, randomised in each Python process: the same activity chain
        # produced `['Travail', 'Achats']` or `['Achats', 'Travail']` depending on the run. Measured on
        # 2026-09-14: 1,656 personas out of 11,329 differed between two generations of the same
        # pool, by this order alone. The content of the file therefore changed without any
        # information changing — a `sha256` seal that no longer means anything.
        #
        # The order of appearance is moreover meaningful: it says in what order the day
        # chains its purposes, where an alphabetical sort would only have said the alphabet.
        travel_purposes: list[str] = []
        for act in activities_list:
            libelle = _PURPOSE_LABEL.get(act["purpose"])
            if libelle and libelle not in travel_purposes:
                travel_purposes.append(libelle)

        traits = {
            "name": name,
            "age": age_val,
            "gender": "Male" if sex == "male" else "Female",
            "main_occupation": main_occupation,
            "travel_purposes": travel_purposes,
            "occupation": {
                "title": occ_title,
                "organization": occ_org,
            },
            # Additional demographic fields
            "household_size": int(row["household_size"]),
            "income": income_text,
            "socioprofessional_class": spc_label,
            "professional_activity": pro_act_label,
            "employment_sector": sector if sector != "not_applicable" else "",
            "car_availability": str(row.get("car_availability", "")),
            # `bool(row.get(col, False))` only protected against the missing column,
            # never against the NaN value — and `bool(nan)` is True. Every unmatched
            # person therefore received the licence (ticket 008, A1.a). Two guards:
            # `pd.notna` for the NaN, and the legal age, which makes the anomaly impossible
            # whatever the matched donor.
            "has_driving_license": _flag(row.get("has_license")) and age_val >= 18,
            "has_pt_subscription": _flag(row.get("has_pt_subscription")),
            "number_of_cars": int(row.get("number_of_cars", 0)),
            "employed": bool(row.get("employed", False)),
            "studies": bool(row.get("studies", False)),
        }

        # `personal_bike` is NOT set here: the law lives in the post-processing
        # (`scripts.data.population.enrich_personal_bike`, ticket 034 batch 2). The default
        # « Pas de vélo » that stood at this place asserted that a persona has no
        # bike where the truth is « nobody has decided yet » — and it
        # stifled the runtime alarm, which precisely tells the ABSENT trait
        # (no bike + `[ALARME]`) from the trait that SAYS « Pas de vélo ». The key is
        # therefore only written if the column exists, which is no longer the case of this stage.
        if "personal_bike" in row and pd.notna(row.get("personal_bike")):
            traits["personal_bike"] = str(row["personal_bike"])

        # Ring and commune of residence, read from the survey's commune division.
        # Set AFTER the snap block, hence on the coordinates that `identity.home`
        # will carry — the same that the « Lieu de résidence » column of the journal will read.
        traits.update(_residence_traits(home_location))

        if generate_personality_traits:
            persona = _pick_personality(pid)
            p = persona.get("personality", {})
            traits["style"] = persona.get("style", "")
            traits["personality"] = {
                "traits": p.get("traits", []),
                "big_five": p.get("big_five", {}),
            }

        # ── Sealing fields (population control, ticket 028) ─────────
        # At the ROOT of the record, not in `traits_json`, and that is not a
        # detail: `traits_json` is what the agent knows about itself — it enters the
        # narrative of the prompt and the key of the decision cache. A household
        # identifier is not something a persona perceives, and `commute_mode` is the
        # ANSWER to the problem the agent must solve: showing it to it would empty any
        # unit accuracy of its meaning. These fields serve the control (bootstrap by
        # household, exact household targets) and the validation, never the decision.
        #
        # Missing → `None`, and the count is reported at the end of the stage: a column lost
        # upstream must show, not melt into an « almost » complete population.
        def _root_field(column: str):
            value = row.get(column)
            if value is None or (isinstance(value, float) and pd.isna(value)) \
                    or str(value) == "undefined":
                # « undefined » is the missing value of the census (anonymised IRIS):
                # it is worth no more than a None, and is counted like it.
                _missing_root_fields[column] += 1
                return None
            return str(value)

        entry = {
            "person_id": str(pid),
            # Day without a trip (a single activity, home). Root, not `traits_json`.
            "immobile": immobile,
            "household": {
                "id": _root_field("household_id"),
                "iris_id": _root_field("iris_id"),
                "commune_id": _root_field("commune_id"),
            },
            "provenance": {
                "census_person_id": _root_field("census_person_id"),
                "hts_id": _root_field("hts_id"),
            },
            # ⚠ Individual ground truth (census `TRANS`, declared commuting mode).
            # MUST NOT reach the prompt. Used to compare, agent by agent, the simulated mode
            # of the home-work trip with the declared mode.
            "validation": {
                "commute_mode": _root_field("commute_mode"),
            },
            "identity": {
                "name": name,
                "traits_json": traits,
                "home": home_location,
                "activities": activities_list,
            },
            "state": {
                "last_location": None,
                "last_activity_index": 0,
                "cache_current_activity": None,
                "heading_to": None,
                "scheduling_in_progress": False,
                "scheduling_started_at": None,
            },
            "is_llm_based": True,
        }
        result.append(entry)

    # Fix zero-duration intermediate activities: if start_time == end_time,
    # set end_time = start_time of the next activity (eqasim arrival_time artifact).
    for entry in result:
        acts = entry["identity"]["activities"]
        for i in range(len(acts) - 1):
            act = acts[i]
            if act["start_time"] == act["end_time"]:
                act["end_time"] = acts[i + 1]["start_time"]

    n = len(result)
    if _missing_root_fields:
        print("[llm_agents] WARNING champs de scellement absents : "
              + ", ".join(f"{k} sur {v}/{len(result)} personnes"
                          for k, v in sorted(_missing_root_fields.items()))
              + " — une colonne a été perdue en amont (enriched.py) ou n'existe pas dans le "
                "recensement pour ces lignes ; le contrôle de population le verra.")
    else:
        print(f"[llm_agents] Champs de scellement complets sur {len(result)} personnes "
              "(household, provenance, validation).")
    print(f"[llm_agents] Immobiles gardés : {_immobiles} sur {len(result)} personnes "
          f"({100.0 * _immobiles / max(len(result), 1):.1f} % ; enquête EMC² 2023 : 10,6 %)"
          + (f" — {_immobiles_sans_domicile} immobile(s) sans domicile écarté(s)"
             if _immobiles_sans_domicile else ""))
    output_file = os.path.join(output_path, f"{output_prefix}population_{n}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Exported {n} agents to {output_file}")

    # Sanity check: no two consecutive activities at the same location.
    violations = []
    for entry in result:
        acts = entry["identity"]["activities"]
        for i in range(len(acts) - 1):
            a, b = acts[i], acts[i + 1]
            la, lb = a.get("location") or {}, b.get("location") or {}
            lon_a, lat_a = la.get("lon"), la.get("lat")
            lon_b, lat_b = lb.get("lon"), lb.get("lat")
            if (lon_a is not None and lon_b is not None and
                    abs(lon_a - lon_b) < 1e-5 and abs(lat_a - lat_b) < 1e-5):
                violations.append(
                    f"  person={entry['person_id']} idx={i}→{i+1} "
                    f"{a['purpose']}→{b['purpose']} @ ({lat_a:.5f},{lon_a:.5f})"
                )
    if violations:
        print(f"[llm_agents] WARNING: {len(violations)} consecutive same-location activity pair(s) "
              f"(scheduler handles these via same_location check — no routing issued):")
        for v in violations[:10]:
            print(v)
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
    else:
        print(f"[llm_agents] Sanity check passed: no consecutive same-location activities")

    return n
