"""
Population loader that reads the JSON output produced by the eqasim pipeline
(synthesis.population.llm_agents stage).

The file is expected at:
  {settings.data.eqasim_output_dir}/{settings.data.synthetic_file_prefix}population_*.json

If several files match, the one with the largest embedded count (highest N in
  toulouse_population_N.json) is used.
"""

import json
import os
import re
from typing import Optional
from loguru import logger

import numpy as np

from inputs.population.base import Filter, PopulationLoader
from models import Activity, BBox, Location, Person, PersonalIdentity, PersonState
from settings import settings
from utils import fake

from mobility_core.population_reference import (
    COURONNES,
    OUT_OF_PERIMETER,
    OUT_OF_PERIMETER_FR,
    couronne_canonique,
)
from mobility_core.residence_zone import TRAIT_KEY as RESIDENCE_TRAIT_KEY

# Population ADMISSION filter (ticket 026, stage 3).
#
# It used to apply to a RECTANGLE — the footprint of the GTFS stops widened by 5 km —, which
# matches no survey definition: measured, this rectangle only contains 221 of the
# 453 communes of the EMC² scope and 51 of its 277 fine zones of the 3rd ring. A
# population conforming to the scope lost half of its territory there at loading.
#
# It now applies to the SCOPE, read on the persona: the `residence_zone` trait
# (ticket 021) says whether the home is in one of the four rings of the survey or
# `hors périmètre`. No geometry at loading, and the definition is that of
# the survey, not that of the transit network.
#
# ⚠ The SAMPLING frame can be narrower than the scope (Haute-Garonne version of
# ticket 026: 346 communes out of 453). This filter stays on the 453: burning the frame's
# limitation into the runtime would force us to dig it up when the frame widens.
_ADMITTED_ZONES = frozenset(COURONNES)


def perimeter_verdict(person, bbox: Optional[BBox]) -> tuple[bool, str]:
    """`(admitted, rejection reason)` — the reason is empty when the person is admitted.

    Three cases, and the third is the only one that falls back on the rectangle:

    - trait present and in a ring → **admitted**, wherever the bbox is;
    - trait present and `hors périmètre` → **rejected**: the home is known and it is
      outside the 453 communes of the survey, it has no target per zone;
    - trait missing (population generated before ticket 021) → fallback on the bbox, and
      the caller raises an alarm: the scope is then NOT guaranteed.
    """
    home = person.identity.home
    if home is None or home.lon is None or home.lat is None:
        return False, "sans domicile"

    zone = (person.identity.traits_json or {}).get(RESIDENCE_TRAIT_KEY)
    if zone:
        # Ticket 081 — the modality is TRANSLATED before being judged. A cohort from before the
        # English switch (ticket 074) carries "1ere couronne" where the canon says "1st
        # ring": read as is, it fell entirely into "zone inconnue" and the filter
        # rejected 637 out of 1,000 of it — measured on 2026-09-15 by reopening the frozen v5 cohort.
        # Loading did not fail: it returned 363 persons, and the caller believed it had
        # all of them. `couronne_canonique` exists exactly for this re-reading.
        canonique = couronne_canonique(zone)
        if canonique in _ADMITTED_ZONES:
            return True, ""
        if canonique in (OUT_OF_PERIMETER, OUT_OF_PERIMETER_FR):
            # Rejected, but NAMED: out of scope is a modality of the survey, not a
            # value we do not understand. Confusing them would mask a badly translated cohort
            # behind a legitimate rejection.
            return False, OUT_OF_PERIMETER
        return False, f"zone inconnue ({zone})"

    if bbox is None:
        return True, ""
    inside = (bbox.min_lon <= home.lon <= bbox.max_lon
              and bbox.min_lat <= home.lat <= bbox.max_lat)
    return (inside, "" if inside else "hors bbox (trait absent)")


def _apply_perimeter_filter(people: list, bbox: Optional[BBox], source: str) -> list:
    """Filter, count, and ALARM if the scope could not be checked."""
    from collections import Counter

    motifs: Counter = Counter()
    sans_trait = 0
    retenus = []
    for person in people:
        if not (person.identity.traits_json or {}).get(RESIDENCE_TRAIT_KEY):
            sans_trait += 1
        admis, motif = perimeter_verdict(person, bbox)
        if admis:
            retenus.append(person)
        else:
            motifs[motif] += 1

    detail = ", ".join(f"{n} {m}" for m, n in motifs.most_common()) or "aucun rejet"
    logger.info(f"[{source}] scope filter: {len(people)} → {len(retenus)} "
                f"({detail})")
    if sans_trait:
        # Rising edge deliberately absent: this case is a state of the population, not
        # a repeated event. A non-enriched population must be visible at every
        # loading, otherwise we believe we filter on the scope while we filter on a
        # rectangle.
        logger.error(
            f"[ALARME] {sans_trait}/{len(people)} persona(s) sans trait "
            f"`{RESIDENCE_TRAIT_KEY}` : le périmètre d'enquête n'est PAS garanti pour "
            f"eux, le filtre retombe sur la bbox du réseau TC. Corrigez la population "
            f"avec `make residence-zone` (ticket 021)."
        )
    return retenus

def _generate_name(gender: str) -> str:
    if gender == "Male":
        return fake.name_male()
    elif gender == "Female":
        return fake.name_female()
    return fake.name()



def _identifiant_de_foyer(entry: dict) -> str | None:
    """`household.id` of the raw entry, or `None` (ticket 100, lot 1).

    The household is the ONLY social group of the simulation that carries a stable identifier. It
    was in the JSON since the seal and lost when building `Person`, without any
    log saying so — the `foyers` exposure rule and circulation within the household both
    depend on it.
    """
    foyer = entry.get("household") or {}
    identifiant = foyer.get("id")
    return str(identifiant) if identifiant not in (None, "") else None

class EqasimJSONPopulationLoader(PopulationLoader):
    def __init__(self, filters: Optional[list[Filter]] = None):
        self.filters = filters or []

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_population_file(output_dir: str, prefix: str) -> str:
        pattern = re.compile(rf"^{re.escape(prefix)}population_(\d+)\.json$")
        candidates = []
        for name in os.listdir(output_dir):
            m = pattern.match(name)
            if m:
                candidates.append((int(m.group(1)), os.path.join(output_dir, name)))
        if not candidates:
            raise FileNotFoundError(
                f"No eqasim population JSON found in {output_dir!r} "
                f"(expected prefix {prefix!r}population_N.json)"
            )
        # Pick the file with the most people
        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def _parse_activity(act: dict) -> Activity:
        loc = act.get("location")
        scheduled_start_time = act.get("scheduled_start_time")
        if act.get("scheduled_start_time") is None:
            scheduled_start_time = act["start_time"] - 15 * 60
        return Activity(
            id=act["id"],
            scheduled_start_time=scheduled_start_time,
            start_time=float(act["start_time"]),
            end_time=float(act["end_time"]),
            purpose=act["purpose"],
            location=Location(lon=loc["lon"], lat=loc["lat"], public_transport=loc.get("public_transport")) if loc and loc.get("lon") is not None else None,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_population(self, max_size: int, bbox: Optional[BBox] = None) -> list[Person]:
        output_dir = settings.data.eqasim_output_dir
        prefix = settings.data.synthetic_file_prefix

        json_file = self._find_population_file(output_dir, prefix)
        print(f"[EqasimJSONPopulationLoader] Loading from {json_file}")

        with open(json_file, encoding="utf-8") as f:
            raw = json.load(f)

        people: list[Person] = []
        for entry in raw:
            identity_data = entry["identity"]

            activities = [
                self._parse_activity(act)
                for act in identity_data.get("activities", [])
            ]

            home_raw = identity_data.get("home")
            home = Location(lon=home_raw["lon"], lat=home_raw["lat"], public_transport=home_raw.get("public_transport")) if home_raw and home_raw.get("lon") is not None else None

            state_raw = entry.get("state", {})
            state = PersonState(
                last_location=None,
                last_activity_index=state_raw.get("last_activity_index", 0),
            )

            traits_json = identity_data["traits_json"]
            name = _generate_name(traits_json.get("gender", ""))
            traits_json["name"] = name

            person = Person(
                person_id=entry["person_id"],
                identity=PersonalIdentity(
                    name=name,
                    traits_json=traits_json,
                    home=home,
                    activities=activities,
                ),
                state=state,
                is_llm_based=entry.get("is_llm_based", True),
                # Ticket 100, lot 1 — the household travels to the runtime. `household` is at the
                # root of the eqasim entry, never in the persona's traits: putting it
                # in the traits would one day bring it into a prompt, and household membership
                # is a simulation property, not a character trait.
                household_id=_identifiant_de_foyer(entry),
            )
            people.append(person)

        total_parsed = len(people)
        print(f"[EqasimJSONPopulationLoader] Parsed {total_parsed} people from JSON")

        # Admission filter: the survey SCOPE, plus the transit network rectangle.
        people = _apply_perimeter_filter(people, bbox, "EqasimJSONPopulationLoader")

        # # Quality filter: at least 3 activities, at least one work/education trip
        # before_quality = len(people)
        # no_work_edu = [
        #     p for p in people
        #     if not any(a.purpose in ("work", "education") for a in (p.identity.activities or []))
        # ]
        # too_few_acts = [
        #     p for p in people
        #     if len(p.identity.activities or []) <= 3
        #     and any(a.purpose in ("work", "education") for a in (p.identity.activities or []))
        # ]
        # people = [
        #     p for p in people
        #     if len(p.identity.activities or []) > 0
        #     and any(a.purpose in ("work", "education") for a in (p.identity.activities or []))
        # ]
        # print(
        #     f"[EqasimJSONPopulationLoader] Quality filter: {before_quality} → {len(people)} "
        #     f"(dropped {len(no_work_edu)} without work/education, {len(too_few_acts)} with ≤3 activities)"
        # )
        # for p in range(len(no_work_edu)):
        #     if (p < 5):  # print up to 5 examples
        #         print(f"  - Person {no_work_edu[p]} dropped: no work/education activities")
        # for p in range(len(too_few_acts)):
        #     if (p < 5):  # print up to 5 examples
        #         print(f"  - Person {too_few_acts[p]} dropped: only {len(too_few_acts[p].identity.activities or [])} activities")

        # Additional caller-supplied filters (e.g. PersonCloseToTheStopFilter)
        for f in self.filters:
            before = len(people)
            people = [p for p in people if f.is_valid(p)]
            print(
                f"[EqasimJSONPopulationLoader] Filter {f.__class__.__name__}: "
                f"{before} → {len(people)}"
            )

        if max_size and max_size < len(people):
            people = list(np.random.choice(people, max_size, replace=False))

        print(f"[EqasimJSONPopulationLoader] Loaded {len(people)} people")
        return people

    def load_population_from_data(self, raw: list, max_size: int, bbox: Optional[BBox] = None) -> list[Person]:
        """Parse pre-loaded eqasim JSON entries into Person objects (same logic as load_population)."""
        people: list[Person] = []
        for entry in raw:
            identity_data = entry["identity"]
            activities = [
                self._parse_activity(act)
                for act in identity_data.get("activities", [])
            ]
            home_raw = identity_data.get("home")
            home = Location(
                lon=home_raw["lon"], lat=home_raw["lat"],
                public_transport=home_raw.get("public_transport"),
            ) if home_raw and home_raw.get("lon") is not None else None
            state_raw = entry.get("state", {})
            state = PersonState(
                last_location=None,
                last_activity_index=state_raw.get("last_activity_index", 0),
            )
            traits_json = identity_data["traits_json"]
            name = traits_json.get("name") or _generate_name(traits_json.get("gender", ""))
            traits_json["name"] = name
            person = Person(
                person_id=entry["person_id"],
                identity=PersonalIdentity(name=name, traits_json=traits_json, home=home, activities=activities),
                state=state,
                is_llm_based=entry.get("is_llm_based", True),
                household_id=_identifiant_de_foyer(entry),
            )
            people.append(person)

        people = _apply_perimeter_filter(people, bbox, "EqasimJSONPopulationLoader/cache")

        if max_size and max_size < len(people):
            people = list(np.random.choice(people, max_size, replace=False))

        print(f"[EqasimJSONPopulationLoader] Loaded {len(people)} people from pre-loaded data")
        return people
