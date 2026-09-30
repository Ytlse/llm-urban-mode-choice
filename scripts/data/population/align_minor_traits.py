"""align_minor_traits.py — Align minors' traits with their age (population step 8; ticket 008, A1.b).

**Surface** fix on an already generated population. It exists because the
root cause is in eqasim (`config_toulouse.yml`, `synthesis/population/`) and
rerunning it requires access to the source data, outside the repository. This script
unblocks the run without waiting for that access; the eqasim guards (A1.a) are what
will make the fix useless at the next generation cycle.

**What it fixes.** HTS matching lost `age_class` (donor pool
reduced to department 31 alone, a degradation that drops columns from the end), and
a `bool(nan)` evaluating to `True` handed the licence to every unmatched person.
Result: 131 of the 165 minors of the population carried `has_driving_license`,
and a nine-year-old pupil reached the LLM with the activity chain of a worker.

Five transformations, in order:

1. `has_driving_license → false` under 18;
2. `purpose: "work" → "education"` for pupils and students;
3. recomputation of `travel_purposes` from the corrected activities;
4. recomputation of `car_availability` per household, minors' licences no longer counting;
5. `personal_bike: "VAE" → "vélo normal"` under 14.

**What it does not fix.** The *activity chains* remain those of adult
donors: workers' departure times, workers' destinations. Renaming `work` to
`education` does not bring the school closer to home, so a child may still be
expected at 8 am at the other end of the urban area. This is the accepted limit of the
surface fix, recalled in its exit report at every run.

Idempotent: a second run changes nothing (the rules are target
states, not increments).

Usage:
    python -m scripts.data.population.align_minor_traits data/population/toulouse_population_1000.json
    python -m scripts.data.population.align_minor_traits data/population/*.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mobility_core.bike_ownership import ELECTRIC_BIKE, PLAIN_BIKE
from mobility_core.population_reference import PURPOSE_LABEL

# Driving licence age, and age below which an electric-assist bike makes no
# sense. Hard-coded values: these are legal and physical thresholds, not settings.
DRIVING_AGE = 18
VAE_MIN_AGE = 14

# Activity purpose → `travel_purposes` label. READ from `mobility_core`, no longer copied:
# this table and the generator's promised each other to be identical through a comment,
# and they diverged at the first translation (ticket 074).
PURPOSE_FR = PURPOSE_LABEL

# An agent "in education" in the sense of rule 2. Depending on the version,
# `professional_activity` carries the eqasim code (`student`, `under14`) or its readable
# label (`Student`, `Child (under 14)`): both are accepted.
STUDENT_ACTIVITIES = {"student", "under14", "Student", "Child (under 14)"}

# `main_occupation` switched to English in ticket 074: v6 says "Pupil (up to
# Baccalaureate)" where v5 said "Scolaire (jusqu'au Bac)". So we compare on the
# survey IDENTIFIER (`scolaire`), which `OCCUPATION_MAP` returns from both
# vocabularies. A French literal would have stopped recognising a single pupil of
# v6 — and rule 2 (`work` → `education`) would have applied to nobody, without
# any counter moving, since its counter is the number of CORRECTED cases.
STUDENT_OCCUPATION_KEY = "scolaire"




def _occupation_key(main_occupation) -> str | None:
    """Survey identifier of an occupation, in both vocabularies."""
    from scripts.synthesis.frames import OCCUPATION_MAP

    return OCCUPATION_MAP.get(str(main_occupation or ""))


def _is_student(traits: dict) -> bool:
    return (
        str(traits.get("professional_activity", "")) in STUDENT_ACTIVITIES
        or _occupation_key(traits.get("main_occupation")) == STUDENT_OCCUPATION_KEY
    )


def _home_key(person: dict) -> tuple | None:
    """Household key: the home coordinates, rounded.

    Lacking a `household_id` in the exported population, the home *is* the
    household. Rounding (≈ 1 cm) absorbs serialisation differences without ever
    merging two distinct addresses.
    """
    home = (person.get("identity") or {}).get("home") or {}
    lat, lon = home.get("lat"), home.get("lon")
    if lat is None or lon is None:
        return None
    return (round(float(lat), 7), round(float(lon), 7))


def fix(population: list[dict]) -> Counter:
    """Applies the five rules in place. Returns the count of corrections."""
    counts: Counter = Counter()

    # ── Rules 1, 2, 3, 5 — at person level ────────────────────────────────────
    for person in population:
        identity = person.get("identity") or {}
        traits = identity.get("traits_json")
        if traits is None:
            counts["sans_traits"] += 1
            continue
        counts["personnes"] += 1
        age = traits.get("age")
        age = int(age) if age is not None else None

        # 1. Nobody drives before the legal age.
        if age is not None and age < DRIVING_AGE and traits.get("has_driving_license"):
            traits["has_driving_license"] = False
            counts["permis_retires"] += 1

        # 2. A pupil goes to school, not to work. The `work` purpose of an adult
        #    donor becomes `education` — the place, however, stays the donor's.
        activities = identity.get("activities") or []
        if _is_student(traits):
            counts["scolaires"] += 1
            for activity in activities:
                if activity.get("purpose") == "work":
                    activity["purpose"] = "education"
                    counts["motifs_reclasses"] += 1

        # 3. `travel_purposes` derives from the activities: it is recomputed, not
        #    corrected. Stable order (that of PURPOSE_FR) so that two
        #    runs produce a byte-for-byte identical file.
        purposes = {a.get("purpose") for a in activities}
        recomputed = [label for key, label in PURPOSE_FR.items() if key in purposes]
        if traits.get("travel_purposes") != recomputed:
            traits["travel_purposes"] = recomputed
            counts["travel_purposes_recalcules"] += 1

        # 5. No e-bike before 14 (random draw by eqasim, without age filter).
        # Both vocabularies: "e-bike" (v6) and "VAE" (earlier cohorts). This
        # fix is idempotent and replayed on already enriched populations — including
        # old ones. Knowing only one would leave e-bikes to 10-year-olds without
        # any counter moving: the rule would apply, simply to nothing.
        if (age is not None and age < VAE_MIN_AGE
                and str(traits.get("personal_bike")) in {ELECTRIC_BIKE, "VAE"}):
            traits["personal_bike"] = PLAIN_BIKE
            counts["vae_declasses"] += 1

    # ── Rule 4 — at household level, after licences are removed ──────────────
    #
    # ⚠ MEASURED LIMIT: the household is rebuilt from home coordinates, but
    # the exported population is a SAMPLE — 127 of the 547 "households" of
    # toulouse_population_1000.json have fewer members present than
    # their `household_size` announces (2 people for a household of 3). Licences
    # of absent members are therefore not counted, `cars >= licenses` passes too
    # often, and `car_availability` leans towards "all" — which travels up to the
    # prompt through the persona narrative ("car always available" instead of "to
    # be shared"). This is flagged (`menages_incomplets`) rather than endured: the
    # proper fix requires a `household_id` at generation.
    licenses_by_home: dict[tuple, int] = defaultdict(int)
    cars_by_home: dict[tuple, int] = {}
    members_by_home: dict[tuple, int] = defaultdict(int)
    declared_size_by_home: dict[tuple, int] = {}
    for person in population:
        key = _home_key(person)
        traits = (person.get("identity") or {}).get("traits_json")
        if key is None or traits is None:
            continue
        age = traits.get("age")
        if traits.get("has_driving_license") and (age is None or int(age) >= DRIVING_AGE):
            licenses_by_home[key] += 1
        cars = int(traits.get("number_of_cars", 0) or 0)
        # The field is supposed to be a household property; if it differs between
        # housemates, the maximum is kept and flagged.
        if key in cars_by_home and cars_by_home[key] != cars:
            counts["menages_number_of_cars_incoherent"] += 1
        cars_by_home[key] = max(cars_by_home.get(key, 0), cars)
        members_by_home[key] += 1
        declared_size_by_home[key] = max(
            declared_size_by_home.get(key, 0),
            int(traits.get("household_size", 0) or 0))

    for key, declared in declared_size_by_home.items():
        if declared > members_by_home[key]:
            counts["menages_incomplets"] += 1
        if cars_by_home.get(key, 0) > 0 and licenses_by_home[key] == 0:
            # Car in the household, nobody to drive it among the PRESENT members.
            # Non-drivers there are nonetheless eligible for the passenger mode
            # (`_is_car_passenger` only tests `household_size > 1`): they would be
            # driven by an adult who does not exist in the data.
            counts["menages_voiture_sans_conducteur"] += 1

    for person in population:
        key = _home_key(person)
        traits = (person.get("identity") or {}).get("traits_json")
        if traits is None:
            continue
        if key is None:
            counts["sans_domicile"] += 1
            continue
        cars = cars_by_home[key]
        licenses = licenses_by_home[key]
        if cars == 0:
            availability = "none"
        elif cars >= licenses:
            availability = "all"
        else:
            availability = "some"
        if traits.get("car_availability") != availability:
            traits["car_availability"] = availability
            counts["car_availability_recalcules"] += 1
        counts[f"car_availability::{availability}"] += 1

    counts["menages"] = len(cars_by_home)
    return counts


def report(population: list[dict], counts: Counter) -> None:
    """Corrections applied, residual state, and uncovered limit."""
    traits = [(p.get("identity") or {}).get("traits_json") or {} for p in population]
    minors = [t for t in traits if (t.get("age") or 0) < DRIVING_AGE]
    purposes = Counter(a.get("purpose")
                       for p in population
                       for a in ((p.get("identity") or {}).get("activities") or []))

    print(f"\n  {counts['personnes']} people, {counts['menages']} households "
          f"(grouped by home coordinates)")
    for key, label in (
        ("permis_retires",              "permis retirés (< 18 ans)"),
        ("motifs_reclasses",            "activités work → education"),
        ("travel_purposes_recalcules",  "travel_purposes recalculés"),
        ("car_availability_recalcules", "car_availability recalculés"),
        ("vae_declasses",               "VAE → vélo normal (< 14 ans)"),
    ):
        print(f"  {label:38s} {counts.get(key, 0):5d}")
    for key, label in (
        ("sans_traits",   "personas sans traits_json"),
        ("sans_domicile", "personas sans domicile (ménage indéterminé)"),
        ("menages_number_of_cars_incoherent",
         "number_of_cars divergent dans un ménage"),
        ("menages_incomplets",
         "ménages partiellement exportés"),
        ("menages_voiture_sans_conducteur",
         "ménages : voiture, aucun conducteur"),
    ):
        if counts.get(key):
            print(f"  ⚠ {label:36s} {counts[key]:5d}")

    print("\n  State after correction:")
    print(f"    minors with a licence                "
          f"{sum(1 for t in minors if t.get('has_driving_license')):5d}  (target 0)")
    print(f"    \"education\" activities               {purposes.get('education', 0):5d}  (target > 120)")
    print(f"    \"work\" activities                    {purposes.get('work', 0):5d}")
    print("    car_availability : " + ", ".join(
        f"{k} {counts.get('car_availability::' + k, 0)}" for k in ("all", "some", "none")))

    print("\n  ⚠ Limit not covered by this script (ticket 008, D1): the activity "
          "chains\n    remain those of adult donors — workers' departure times "
          "and destinations.\n    Renaming `work` to `education` does not bring "
          "the school closer to home. Only the\n    eqasim guards (A1.a) and a "
          "population regeneration lift this limit.")

    if counts.get("menages_incomplets") or counts.get("menages_voiture_sans_conducteur"):
        print("\n  ⚠ Household rebuilt from coordinates, on a sample: rule 4 "
              "counts the\n    licences of PRESENT members only. A partially "
              "exported household thus has\n    fewer licences than in reality, and its "
              "`car_availability` leans towards \"all\".\n    Where no driver is "
              "present, non-drivers remain eligible for the\n    passenger mode "
              "(`_is_car_passenger` only tests `household_size > 1`): they are\n    "
              "driven by an adult absent from the data. Proper fix: a "
              "`household_id`\n    at generation, or a trait "
              "`household_has_licensed_driver` set here and read by\n    the controller.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population", type=Path, nargs="+",
                        help="Population JSON files to fix (modified in place)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Computes and reports without rewriting the files")
    args = parser.parse_args()

    for path in args.population:
        if not path.exists():
            print(f"[ERREUR] Population introuvable : {path}", file=sys.stderr)
            return 1
        population = json.loads(path.read_text(encoding="utf-8"))
        counts = fix(population)
        print(f"\n=== {path}")
        report(population, counts)
        if args.dry_run:
            print("\n  [dry-run] file not rewritten")
            continue
        # Atomic write: a crash in the middle of writing must not leave
        # a truncated population behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(f"\n  written → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
