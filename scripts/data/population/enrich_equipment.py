"""enrich_equipment.py — Set the PT pass and the driving licence learned on EMC² (lot 2).

Path 1 of [tickets 016](../../../docs/tickets/ticket_016_abonnement_tc_progedo.md) and
[017](../../../docs/tickets/ticket_017_permis_progedo.md): a population file is reread,
two fields of `traits_json` are rewritten, and it is saved. No regeneration,
applicable to existing populations.

Both traits are handled by **a single script** because the tickets state that
their lots 1 and 2 are shared, and because the order between them and `car_availability`
is a trap that a single entry point closes:

    has_pt_subscription   ← Bernoulli draw from the law pt_subscription.json
    has_driving_license   ← Bernoulli draw from the law driving_license.json
    car_availability      ← DERIVES from the household's number of licences: it is STALE
                            as soon as the previous line runs

⚠ **This script does not recompute `car_availability`, and that is deliberate.** The rule lives
already in `align_minor_traits.py` (rule 4, adults only, ticket 008 A1.a), which is
**idempotent** — copying it here would make two definitions to keep in agreement. The generation
notebook therefore replays `align_minor_traits` **after** this enrichment, and this script
fails loudly under `--check` if the licences it has just set are no longer consistent
with the file's `car_availability`. A `car_availability` computed on the old licences
shows up nowhere: this is the exact warning of ticket 017.

## What is imputed, and what is not

The traits remain **booleans**, as today: they are read by the mode choice
policy, by the persona's narrative and by the summary page, and making them nullable
would break three consumers for nothing. On the other hand the **fallback is explicit and counted**:

| Level | When | What is served |
|---|---|---|
| `zone` | home in the layer, zone with surveyed households | density and distance of the zone |
| `zone_sans_densite` | home in the layer, zone without surveyed household (81 of 785) | distance of the zone, median density of the scope |
| `perimetre` | home outside the layer | distance **computed** from the published hypercentre, median density |

The `perimetre` level **never** sets the distance to zero: that would place the home at
the hypercentre, i.e. impute the law's most discriminating variable with its
value most favourable to public transport. It is computed in Lambert-93 from
`feature_spec.json`, the same hypercentre as part 3 of the summary page.

Below a trait's age range, no propensity is evaluated: the licence is `false`
under 18 by construction (legal floor, not a model parameter) and the count comes out
as `sous_age_champ`.

## Determinism

Bernoulli of the propensity, key `(home address, person identifier, versioned
salt)`. Two runs, two machines, two moments give the same result. The
draw is made on the **person** and not on the address — a pass is personal, two
flatmates have no reason to share theirs.

Usage:
    python -m scripts.data.population.enrich_equipment data/population/toulouse_population_1000.json
    python -m scripts.data.population.enrich_equipment data/population/*.json --dry-run
    python -m scripts.data.population.enrich_equipment data/population/toulouse_population_1000.json --check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from mobility_core.equipment_propensity import (
    DRIVING_LICENSE,
    PT_SUBSCRIPTION,
    PropensityLaw,
    TraitSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

EXIT_OK = 0
EXIT_RESOURCE_MISSING = 1
EXIT_TARGET_MISSED = 2
EXIT_NOT_MEASURABLE = 3
# Code 4 is NOT a failure of the trait, and the distinction is the heart of the acceptance test:
# the overall gates pass, but a STRATUM deviates because the population does not have
# the survey's composition inside that stratum. This is fixed in the
# DRAW, not in the enrichment — same vocabulary as axis A9 of ticket 020, which
# the notebook already knows how to read as non-blocking.
#
# The rule that separates them: an OVERALL gap or a coverage defect blames the
# law or its application (code 2); a STRATUM gap alone, overall held, blames the
# population's composition (code 4). Confusing them would block the generation
# chain on a defect that no enrichment can repair.
EXIT_COMPOSITION_GAP = 4

# Minimum coverage: below it, the population is not usable and no target
# makes sense. A trait missing everywhere does not "pass" the acceptance test, it invalidates it.
MIN_COVERAGE = 0.95

# Acceptance tolerances. Those of the tickets, increased by the stratum's sampling
# noise — a population of 1,000 agents carries ~50 people per occupation category,
# i.e. ~7 points of standard deviation on a 50 % share.
TOLERANCE_OVERALL_PT = 6.0
TOLERANCE_STRATUM_PT = 12.0

# Below this count, a stratum decides nothing: it is displayed, not enforced.
THIN_STRATUM = 20

SPECS = (PT_SUBSCRIPTION, DRIVING_LICENSE)

# Enforceable targets. Those of the tickets, with the pass target for "Étudiant"
# RESTATED from 74.3 to 72.2 %: the ticket measures on `P9 == 4` alone, whereas the repository's
# recoding also files work-study/internship (`P9 == 3`, 146 people, 56.7 % pass holders)
# under "Étudiant". The repository's definition must win, since it is the one
# the persona carries — enforcing 74.3 % would grade the law on a stratum it does not see.
TARGETS = {
    "has_pt_subscription": {
        "_overall": 25.8,
        "Étudiant": 72.2,
        "Scolaire (jusqu'au Bac)": 33.3,
        "Chômeur/recherche d'emploi": 28.8,
        "Personne au foyer": 24.0,
        "Travail à temps partiel": 21.5,
        "Retraité": 17.7,
        "Travail à plein temps": 14.8,
    },
    "has_driving_license": {
        "_overall": 85.9,
        "Travail à plein temps": 94.8,
        "Retraité": 92.6,
        "Travail à temps partiel": 86.5,
        "Chômeur/recherche d'emploi": 69.4,
        "Personne au foyer": 63.9,
        "Étudiant": 59.2,
    },
}

# The gap that is the heart of ticket 016: it holds the SIGN and the AMPLITUDE, and it is the
# most discriminating criterion — the ENTD copy put it at +5.7 pt against +54.5
# observed (72.2 − 17.7), i.e. a tenfold flattening.
STUDENT_MINUS_RETIRED_TARGET = 54.5
STUDENT_MINUS_RETIRED_MIN = 25.0


def _home(person: dict) -> tuple[Optional[float], Optional[float]]:
    home = (person.get("identity") or {}).get("home") or {}
    return home.get("lat"), home.get("lon")


def _hypercenter(spec_path: Path) -> tuple[float, float]:
    """Hypercentre in Lambert-93, read from the model spec.

    The same as that of part 3 of the summary page and of the residential rings:
    three consumers measuring from three different centres would produce
    three distances that would look comparable.
    """
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    center = spec["geo_reference"]["hypercenter"]
    return float(center["x_l93"]), float(center["y_l93"])


def enrich(population: list[dict], laws: dict[str, PropensityLaw], resolver,
           hypercenter: tuple[float, float]) -> Counter:
    """Sets both traits in place. Returns the counts, fallbacks included."""
    from pyproj import Transformer

    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    cx, cy = hypercenter
    counts: Counter = Counter()

    for person in population:
        identity = person.get("identity") or {}
        traits = identity.get("traits_json")
        if not isinstance(traits, dict):
            counts["sans_traits"] += 1
            continue
        lat, lon = _home(person)

        # Geography of the home, with its fallback level.
        density: Optional[float] = None
        dist_center: Optional[float] = None
        level = "perimetre"
        zone = resolver.resolve(float(lat), float(lon)) if (
            lat is not None and lon is not None) else None
        if zone is not None:
            dist_center = zone.dist_center_km
            density = zone.density_hh_km2
            level = "zone" if density is not None else "zone_sans_densite"
        elif lat is not None and lon is not None:
            # Outside the layer: the distance is COMPUTED, it is not set to zero.
            x, y = to_l93.transform(float(lon), float(lat))
            dist_center = math.hypot(x - cx, y - cy) / 1000.0
        counts[f"repli::{level}"] += 1

        for spec in SPECS:
            law = laws[spec.key]
            value, reason = law.value(
                traits.get("age"), traits.get("gender"),
                traits.get("main_occupation"), traits.get("number_of_cars"),
                density, dist_center, lat, lon, person.get("person_id"))
            if traits.get(spec.key) != value:
                counts[f"{spec.key}::modifie"] += 1
            traits[spec.key] = value
            counts[f"{spec.key}::{reason}"] += 1
            if value:
                counts[f"{spec.key}::vrai"] += 1
        counts["personnes"] += 1
    return counts


# ── Measurement ──────────────────────────────────────────────────────────────

def measure(population: list[dict], spec: TraitSpec) -> dict:
    """Share of the trait, overall and by occupation, over the trait's age range."""
    total = 0
    held = 0
    by_occupation: dict[str, list[int]] = {}
    for person in population:
        traits = ((person.get("identity") or {}).get("traits_json") or {})
        age = traits.get("age")
        if age is None or float(age) < spec.min_age:
            continue
        value = traits.get(spec.key)
        if not isinstance(value, bool):
            continue
        total += 1
        held += int(value)
        occupation = traits.get("main_occupation") or "(inconnue)"
        bucket = by_occupation.setdefault(occupation, [0, 0])
        bucket[0] += 1
        bucket[1] += int(value)
    return {
        "n": total,
        "pct": (100.0 * held / total) if total else None,
        "by_occupation": {
            occupation: {"n": n, "pct": 100.0 * k / n}
            for occupation, (n, k) in sorted(by_occupation.items())
        },
    }


def coverage(population: list[dict], spec: TraitSpec) -> float:
    eligible = 0
    typed = 0
    for person in population:
        traits = ((person.get("identity") or {}).get("traits_json") or {})
        age = traits.get("age")
        if age is None or float(age) < spec.min_age:
            continue
        eligible += 1
        typed += int(isinstance(traits.get(spec.key), bool))
    return (typed / eligible) if eligible else 0.0


def car_availability_is_stale(population: list[dict]) -> Optional[int]:
    """Number of households whose `car_availability` no longer derives from the licences set.

    The eqasim rule is applied — `none` if no car, `all` if cars ≥ adults'
    licences, `some` otherwise — and disagreements are counted. This script does not fix:
    the rule belongs to `align_minor_traits`, and two implementations would drift.
    """
    households: dict[tuple, dict] = {}
    for person in population:
        identity = person.get("identity") or {}
        traits = identity.get("traits_json") or {}
        home = identity.get("home") or {}
        lat, lon = home.get("lat"), home.get("lon")
        if lat is None or lon is None:
            continue
        key = (round(float(lat), 7), round(float(lon), 7))
        entry = households.setdefault(key, {"cars": 0, "licences": 0, "labels": set()})
        entry["cars"] = max(entry["cars"], int(traits.get("number_of_cars") or 0))
        if traits.get("has_driving_license") and (traits.get("age") or 0) >= 18:
            entry["licences"] += 1
        if traits.get("car_availability"):
            entry["labels"].add(traits["car_availability"])

    stale = 0
    for entry in households.values():
        expected = ("none" if entry["cars"] == 0
                    else ("all" if entry["cars"] >= entry["licences"] else "some"))
        if entry["labels"] and entry["labels"] != {expected}:
            stale += 1
    return stale


def report(counts: Counter, population: list[dict]) -> list[str]:
    lines = [f"Personnes traitées : {counts['personnes']}"]
    replis = {k.split("::", 1)[1]: v for k, v in counts.items()
              if k.startswith("repli::")}
    lines.append("  Niveau de repli géographique : " + ", ".join(
        f"{k} {v}" for k, v in sorted(replis.items(), key=lambda kv: -kv[1])))
    for spec in SPECS:
        m = measure(population, spec)
        pct = f"{m['pct']:.1f} %" if m["pct"] is not None else "—"
        lines.append(f"  {spec.key} : {pct} sur {m['n']} personnes du champ "
                     f"({counts[f'{spec.key}::modifie']} valeurs modifiées, "
                     f"{counts[f'{spec.key}::sous_age_champ']} sous l'âge de champ)")
        for occupation, stat in m["by_occupation"].items():
            flag = "  ⚠ strate mince" if stat["n"] < THIN_STRATUM else ""
            target = TARGETS[spec.key].get(occupation)
            target_txt = f" | cible {target:5.1f}" if target is not None else ""
            lines.append(f"      {occupation:28s} n={stat['n']:4d} "
                         f"{stat['pct']:5.1f} %{target_txt}{flag}")
    return lines


def check(population: list[dict]) -> tuple[bool, bool, list[str], bool]:
    """Compares with the targets.

    Returns ``(ensemble_ok, strates_ok, lignes, mesurable)`` — two booleans and not
    one, because the two defects are not repaired in the same place.
    """
    lines: list[str] = []
    ok = True          # overall gates: coverage, global share, safeguards
    strata_ok = True   # stratum gates: composition of the population
    measurable = True

    for spec in SPECS:
        cov = coverage(population, spec)
        if cov < MIN_COVERAGE:
            lines.append(f"  ✗ {spec.key} : couverture {100 * cov:.1f} % "
                         f"< {100 * MIN_COVERAGE:.0f} % — trait absent, la recette est "
                         f"INVALIDE, pas réussie")
            ok = False
            continue
        m = measure(population, spec)
        targets = TARGETS[spec.key]
        gap = m["pct"] - targets["_overall"]
        verdict = "✓" if abs(gap) <= TOLERANCE_OVERALL_PT else "✗"
        ok &= abs(gap) <= TOLERANCE_OVERALL_PT
        lines.append(f"  {verdict} {spec.key} ensemble : {m['pct']:.1f} % contre "
                     f"{targets['_overall']} % attendus (écart {gap:+.1f}, "
                     f"tolérance ± {TOLERANCE_OVERALL_PT})")
        for occupation, target in targets.items():
            if occupation == "_overall":
                continue
            stat = m["by_occupation"].get(occupation)
            if stat is None:
                continue
            if stat["n"] < THIN_STRATUM:
                lines.append(f"      ~ {occupation:28s} n={stat['n']:3d} "
                             f"{stat['pct']:5.1f} % — strate mince, non opposée")
                continue
            g = stat["pct"] - target
            v = "✓" if abs(g) <= TOLERANCE_STRATUM_PT else "✗"
            strata_ok &= abs(g) <= TOLERANCE_STRATUM_PT
            lines.append(f"      {v} {occupation:28s} n={stat['n']:3d} "
                         f"{stat['pct']:5.1f} % contre {target:5.1f} (écart {g:+5.1f})")

    # The most discriminating criterion of ticket 016: the student − retiree gap.
    pt = measure(population, PT_SUBSCRIPTION)["by_occupation"]
    student = pt.get("Étudiant")
    retired = pt.get("Retraité")
    if student and retired and min(student["n"], retired["n"]) >= THIN_STRATUM:
        spread = student["pct"] - retired["pct"]
        v = "✓" if spread >= STUDENT_MINUS_RETIRED_MIN else "✗"
        ok &= spread >= STUDENT_MINUS_RETIRED_MIN
        lines.append(f"  {v} écart étudiant − retraité : {spread:+.1f} pt "
                     f"(≥ {STUDENT_MINUS_RETIRED_MIN} exigé, "
                     f"{STUDENT_MINUS_RETIRED_TARGET} observé dans l'enquête) — "
                     f"c'est le critère que la recopie ENTD écrasait d'un facteur 10")
    else:
        lines.append("  ~ écart étudiant − retraité : strates trop minces pour "
                     "trancher")
        measurable = False

    stale = car_availability_is_stale(population)
    if stale:
        lines.append(f"  ✗ car_availability périmé sur {stale} ménage(s) : il dérive du "
                     f"nombre de permis, que ce script vient de réécrire. Rejouez "
                     f"`python -m scripts.data.population.align_minor_traits <fichier>` "
                     f"(idempotent, règle 4).")
        ok = False
    else:
        lines.append("  ✓ car_availability cohérent avec les permis posés")
    return ok, strata_ok, lines, measurable


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population", type=Path, nargs="+",
                        help="population files to enrich in place")
    parser.add_argument("--zones", type=Path, default=None,
                        help="fine-zone layer (default: mobility_core/data/zf_zones.gpkg)")
    parser.add_argument("--spec", type=Path, default=None,
                        help="model spec, for the hypercentre")
    parser.add_argument("--dry-run", action="store_true",
                        help="measures and displays, without rewriting the files")
    parser.add_argument("--check", action="store_true",
                        help="compares with the targets of tickets 016/017 and exits on failure")
    args = parser.parse_args(argv)

    zones = args.zones or REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_zones.gpkg"
    spec_path = args.spec or REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"
    if not zones.exists():
        print(f"[erreur] Zone layer missing: {zones} — `make zones`",
              file=sys.stderr)
        return EXIT_RESOURCE_MISSING
    if not spec_path.exists():
        print(f"[erreur] Model spec missing: {spec_path} — `make policy`",
              file=sys.stderr)
        return EXIT_RESOURCE_MISSING

    try:
        laws = {spec.key: PropensityLaw.load(spec) for spec in SPECS}
    except FileNotFoundError as exc:
        print(f"[erreur] {exc}", file=sys.stderr)
        return EXIT_RESOURCE_MISSING

    from mobility_core.zone_resolver import ZoneResolver
    resolver = ZoneResolver.load(zones, feature_spec=spec_path)
    hypercenter = _hypercenter(spec_path)
    for spec in SPECS:
        law = laws[spec.key]
        print(f"Law {spec.key}: {len(law.features)} variables, steps "
              f"{'retenus' if any(f in law.features for f in ('under_26', 'age_62p', 'age_65p')) else 'retirés'}"
              f", generated on {law.meta.get('generated_at', '?')}")

    status = EXIT_OK
    for path in args.population:
        if not path.exists():
            print(f"[erreur] Population not found: {path}", file=sys.stderr)
            status = max(status, EXIT_RESOURCE_MISSING)
            continue
        print(f"\n── {path} " + "─" * max(0, 50 - len(str(path))))
        raw = json.loads(path.read_text(encoding="utf-8"))
        population = raw if isinstance(raw, list) else raw.get("agents", raw)
        counts = enrich(population, laws, resolver, hypercenter)
        for line in report(counts, population):
            print(line)

        if not args.dry_run:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            print(f"  Written: {path}")
        else:
            print("  --dry-run: file unchanged")

        if args.check:
            ok, strata_ok, lines, measurable = check(population)
            print("  Acceptance test (tickets 016 / 017):")
            for line in lines:
                print(line)
            if not ok:
                status = max(status, EXIT_TARGET_MISSED)
            elif not strata_ok:
                print(f"\n  → code {EXIT_COMPOSITION_GAP}: the overall gates "
                      f"pass, a stratum deviates. The law sets the trait "
                      f"correctly; it is the COMPOSITION of the population in that "
                      f"stratum that differs from the survey. This is fixed in the draw, "
                      f"not here.")
                status = max(status, EXIT_COMPOSITION_GAP)
            elif not measurable:
                print(f"\n  → code {EXIT_NOT_MEASURABLE}: enriched but NOT VALIDATED — "
                      f"population too small to decide")
                status = max(status, EXIT_NOT_MEASURABLE)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
