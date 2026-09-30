"""enrich_residence_zone.py — The residence ring, set on the persona (ticket 021).

Reads back a generated population, resolves each home to a fine zone, and writes into
`traits_json`:

- `residence_zone` — the ring as defined by the EMC² survey (`Toulouse`, `1ere couronne`,
  `2eme couronne`, `3eme couronne`), or `hors périmètre`;
- `residence_commune` and `residence_insee` — the municipality of the home, which makes
  the classification auditable and survives a redrawing of the rings.

**This trait is OBSERVED, not imputed, and that is what sets it apart from its two neighbours.**
`housing_type` is drawn from a distribution, `personal_bike` in three stages; here there is
no draw, no hash, no salt, no distribution. A home is in a municipality or it is not.
Two consequences: the script is idempotent in the strong sense — two runs, two machines,
two moments give the same byte — and its validation is not about a distribution
but about an AGREEMENT (cf. `--check`).

**Why this post-processing exists.** The ring used to be guessed at runtime from the
distance of the home to the hypercentre (`geo_reference.residence_zone`, 8 / 20 / 40 km).
That is not the survey's partition, which works by **list of municipalities**: ticket 020
measured 24.4 % of misclassified personas, 66 "fake Toulousains" living in Blagnac or Balma,
and a "3rd ring" stratum 76 % of whose inhabitants were not even inside the survey
scope. Setting the ring on the persona fixes both gaps **without touching the
runtime**: the metric classification stays for the terminal time, which classifies arbitrary
points and whose distributions are stratified with it.

**Three values, three meanings not to be confused.**

- a ring: the home is inside the scope, and here is its zone;
- `hors périmètre`: the home is known and it is **outside**. It is not a ring
  (confusing it with the 3rd is gap A4), it has no EMC² target, and its mass is counted
  instead of being diluted;
- **trait absent**: the home has no coordinates. Nothing is known — neither inside nor
  outside — and writing `hors périmètre` would be a claim that nothing supports.

⚠ **NEVER enrich in place a population pinned by a frozen-set manifest.**
`prompt_calibration/calibration_datasets/v5` to `v8` pin the sha256 of
`experiments/archive/2026-08-19_14_36/population_1000.json`: rewriting it breaks four sets
at once. Use `--out` (or `--dry-run`) for these populations. Since the trait enters neither
the persona narrative (allow-list of `_build_profile_narrative`) nor the key of the
decision cache (options + weather + agent/activity/slot), adding it moves no
decision — that is the whole point of this path.

`--check` verifies what the enrichment CONTROLS, and nothing else:

1. **coverage** — every persona with coordinates carries a decided value;
2. **agreement with the geometric reference** — the written ring (obtained from the fine
   zone CODE) is recomputed by MEMBERSHIP of the ring polygons, and the two
   must agree at 100 %. This is the gate of ticket 021, replayed on each population;
3. **modalities** — nothing outside `COURONNES ∪ {hors périmètre}`;
4. **out-of-scope rate** — below the alarm threshold of `zone_resolver` (15 %). This check
   does not judge the draw: it catches a population that is not the one of this scope.

The gap to the population **framing** by ring (36.4 / 34.1 / 14.2 / 15.4 %) is
reported but **is not a gate**: it measures the spatial over-concentration of the draw
(axis A9 of ticket 020), declared out of scope of ticket 021 and already measured at 76.0 %
in the urban core against 70.5 %. Making it a failure would set a red gate from the
first day for a cause we decline to handle here. It has its own exit code,
so that a caller can track it without confusing it with a failure.

Exit codes of `--check`:
  0  all gates pass
  1  missing resource (`make zones`, `make communes-couronnes`)
  2  a gate FAILS — the population is not usable for per-zone scoring
  4  the gates pass, but the gap to the framing exceeds the tolerance (A9, informative)

Usage:
    python -m scripts.data.population.enrich_residence_zone data/population/toulouse_population_1000.json --check
    python -m scripts.data.population.enrich_residence_zone data/population/*.json --dry-run
    python -m scripts.data.population.enrich_residence_zone \\
      experiments/archive/2026-08-19_14_36/population_1000.json \\
      --out /tmp/population_1000.residence_zone.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from mobility_core.population_reference import (
    COURONNES, OUT_OF_PERIMETER, couronne_population_shares)
from mobility_core.residence_zone import (
    COMMUNE_TRAIT_KEY,
    INSEE_TRAIT_KEY,
    TRAIT_KEY,
    CommunalZones,
    CouronneTable,
    ResidenceZoneError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

EXIT_OK = 0
EXIT_RESOURCE_MISSING = 1
EXIT_GATE_FAILED = 2
EXIT_FRAMING_GAP = 4

# Alarm threshold of the out-of-scope rate. It is the one of `zone_resolver.coverage()`: above
# 15 %, this is no longer a distribution tail, it is a population that does not describe
# the survey scope.
MAX_OUT_OF_PERIMETER_RATE = 0.15

# Tolerance, in points, on the population share of each ring against the framing. It
# is NOT a gate (cf. the docstring): it only decides code 4.
FRAMING_TOLERANCE_PT = 5.0


def people_of(population) -> list[dict]:
    if isinstance(population, list):
        return population
    for key in ("people", "personas", "persons"):
        if isinstance(population.get(key), list):
            return population[key]
    raise ResidenceZoneError(
        "unknown population structure: neither a list nor a `people`/`personas` key.")


def home_of(person: dict) -> dict:
    return (person.get("identity") or {}).get("home") or {}


def traits_of(person: dict) -> dict:
    identity = person.get("identity")
    if not isinstance(identity, dict):
        raise ResidenceZoneError("persona without an `identity` block: unreadable population.")
    traits = identity.get("traits_json")
    if not isinstance(traits, dict):
        traits = {}
        identity["traits_json"] = traits
    return traits


def enrich(population, table: CouronneTable, resolver) -> dict:
    """Sets the trait on each persona. Returns the counters of the pass.

    Classification goes through the fine zone CODE, never through a geometry: it is the
    path the runtime could follow, and `--check` checks it against the geometry.
    """
    counts: Counter = Counter()
    for person in people_of(population):
        traits = traits_of(person)
        home = home_of(person)
        lat, lon = home.get("lat"), home.get("lon")
        avant = traits.get(TRAIT_KEY)

        if lat is None or lon is None:
            # Neither inside nor outside: unknown. The trait is removed rather than left
            # at an inherited value that could no longer be tied to a home.
            for key in (TRAIT_KEY, COMMUNE_TRAIT_KEY, INSEE_TRAIT_KEY):
                traits.pop(key, None)
            counts["sans_domicile"] += 1
            continue

        zone = resolver.resolve(lat, lon)
        if zone is None:
            traits[TRAIT_KEY] = OUT_OF_PERIMETER
            # The municipality of a home outside the layer is unknown: it is not made up.
            traits.pop(COMMUNE_TRAIT_KEY, None)
            traits.pop(INSEE_TRAIT_KEY, None)
            counts[OUT_OF_PERIMETER] += 1
        else:
            couronne = table.couronne_of_zf(zone.zf)
            if couronne is None:
                # Zone found but absent from the table: the resource and the layer do not
                # describe the same scope. We do not guess.
                for key in (TRAIT_KEY, COMMUNE_TRAIT_KEY, INSEE_TRAIT_KEY):
                    traits.pop(key, None)
                counts["zone_hors_table"] += 1
                continue
            traits[TRAIT_KEY] = couronne
            commune = table.commune_of_zf(zone.zf)
            if commune is not None:
                traits[INSEE_TRAIT_KEY], traits[COMMUNE_TRAIT_KEY] = commune
            counts[couronne] += 1

        if avant is not None and avant != traits.get(TRAIT_KEY):
            counts["valeur_changee"] += 1
    return dict(counts)


def audit(population, table: CouronneTable, zones: Optional[CommunalZones]) -> dict:
    """Recomputes, checks, and returns what is needed to decide — never repairing silently."""
    people = people_of(population)
    written: Counter = Counter()
    desaccords: list[dict] = []
    hors_modalite: list[str] = []
    sans_valeur = 0
    metrique_divergent = 0

    from mobility_core.geo_reference import residence_zone as classement_metrique

    for person in people:
        traits = traits_of(person)
        home = home_of(person)
        lat, lon = home.get("lat"), home.get("lon")
        valeur = traits.get(TRAIT_KEY)

        if lat is None or lon is None:
            continue
        if not valeur:
            sans_valeur += 1
            continue
        written[valeur] += 1
        if valeur not in COURONNES and valeur != OUT_OF_PERIMETER:
            hors_modalite.append(valeur)
            continue
        if zones is not None:
            par_geometrie = zones.classify(lat, lon)
            if par_geometrie != valeur:
                desaccords.append({"lat": lat, "lon": lon, "trait": valeur,
                                   "geometrie": par_geometrie})
        if classement_metrique(lat, lon) != valeur:
            metrique_divergent += 1

    localises = sum(1 for p in people
                    if home_of(p).get("lat") is not None
                    and home_of(p).get("lon") is not None)
    hors = written.get(OUT_OF_PERIMETER, 0)
    return {
        "n_personas": len(people),
        "n_localises": localises,
        "n_sans_valeur": sans_valeur,
        "written": dict(written),
        "hors_perimetre": hors,
        "taux_hors_perimetre": (hors / localises) if localises else 0.0,
        "desaccords_geometrie": desaccords,
        "hors_modalite": sorted(set(hors_modalite)),
        "divergence_metrique": metrique_divergent,
        "taux_divergence_metrique": (metrique_divergent / localises) if localises else 0.0,
    }


def framing_gap(measured: dict) -> tuple[dict, float]:
    """Population shares by ring against the EMC² framing, out-of-scope excluded.

    Out-of-scope homes have no target: keeping them in the denominator would compare a
    share with another quantity. They are counted apart, never diluted.
    """
    cible = couronne_population_shares()
    total = sum(measured.get(z, 0) for z in COURONNES)
    lignes = {}
    for zone in COURONNES:
        part = 100.0 * measured.get(zone, 0) / total if total else 0.0
        lignes[zone] = {"observe": part, "cible": cible[zone],
                        "ecart": part - cible[zone]}
    l1 = sum(abs(row["ecart"]) for row in lignes.values())
    return lignes, l1


def report(counts: dict, checks: dict) -> list[str]:
    """Prints the pass and returns the list of failing gates (empty if all pass)."""
    failures: list[str] = []

    print(f"  personas                : {checks['n_personas']}")
    print(f"  located homes           : {checks['n_localises']}")
    if counts.get("sans_domicile"):
        print(f"  without coordinates     : {counts['sans_domicile']} — trait absent, "
              f"neither inside nor outside")
    if counts.get("zone_hors_table"):
        print(f"  zone not in table       : {counts['zone_hors_table']}")
    if counts.get("valeur_changee"):
        print(f"  changed values          : {counts['valeur_changee']}")

    for zone in (*COURONNES, OUT_OF_PERIMETER):
        n = checks["written"].get(zone, 0)
        part = 100.0 * n / checks["n_localises"] if checks["n_localises"] else 0.0
        print(f"    {zone:16s} {n:5d}  {part:5.1f} %")

    # Gate 1 — coverage.
    if checks["n_sans_valeur"]:
        failures.append(f"{checks['n_sans_valeur']} persona(s) localisé(s) sans valeur : "
                        f"la couverture doit être totale, un domicile connu se classe "
                        f"toujours (couronne ou hors périmètre)")

    # Gate 2 — agreement with the geometric reference.
    desaccords = checks["desaccords_geometrie"]
    if desaccords:
        failures.append(
            f"{len(desaccords)} domicile(s) classé(s) différemment par le CODE de zone "
            f"fine et par l'APPARTENANCE géométrique, ex. {desaccords[:3]} — le chemin "
            f"par code n'est plus légitime, reprendre `make audit-couronnes`")
    else:
        print("  code ↔ geometry agreement: 100 %")

    # Gate 3 — modalities.
    if checks["hors_modalite"]:
        failures.append(f"modalités hors référentiel : {checks['hors_modalite']}")

    # Gate 4 — out-of-scope rate.
    taux = checks["taux_hors_perimetre"]
    print(f"  out of scope            : {taux * 100:.2f} % "
          f"(alarm threshold {MAX_OUT_OF_PERIMETER_RATE * 100:.0f} %)")
    if taux > MAX_OUT_OF_PERIMETER_RATE:
        failures.append(f"{taux * 100:.1f} % de domiciles hors périmètre : ce n'est plus "
                        f"une queue de distribution, c'est une population qui ne décrit "
                        f"pas le périmètre de l'enquête")

    # Information — what the fix changes compared with the metric classification.
    print(f"  divergence / metric     : {checks['divergence_metrique']} personas "
          f"({checks['taux_divergence_metrique'] * 100:.1f} %) would get another ring "
          f"by distance to the hypercentre")
    return failures


def print_framing(checks: dict) -> bool:
    """Reports the gap to the framing. Returns `True` if it exceeds the tolerance."""
    lignes, l1 = framing_gap(checks["written"])
    print("  population framing (informative — axis A9, out of scope of ticket 021):")
    for zone, row in lignes.items():
        print(f"    {zone:16s} {row['observe']:5.1f} % against {row['cible']:5.1f} % "
              f"→ {row['ecart']:+5.1f} pt")
    print(f"    L1 = {l1:.1f} pt, tolerance per ring "
          f"± {FRAMING_TOLERANCE_PT:.0f} pt")
    return any(abs(row["ecart"]) > FRAMING_TOLERANCE_PT for row in lignes.values())


def destination(path: Path, out: Optional[Path], n_inputs: int) -> Path:
    if out is None:
        return path
    if n_inputs > 1 or out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        return out / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("population", type=Path, nargs="+",
                        help="Population JSON files (modified in place unless --out)")
    parser.add_argument("--table", type=Path, default=None,
                        help="Ring table (default: mobility_core/data/)")
    parser.add_argument("--zones", type=Path, default=None,
                        help="Fine zone layer (default: mobility_core/data/)")
    parser.add_argument("--geojson", type=Path, default=None,
                        help="Ring geometry, for the agreement check")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write elsewhere than in place — MANDATORY for a "
                             "population pinned by a frozen-set manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Computes and reports without rewriting anything")
    parser.add_argument("--check", action="store_true",
                        help="Exits with failure if a gate of ticket 021 is contradicted")
    args = parser.parse_args()

    from mobility_core.zone_resolver import ZoneResolver

    feature_spec = REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"
    try:
        table = CouronneTable.load(args.table)
        zones = CommunalZones.load(args.geojson)
        resolver = ZoneResolver.load(args.zones,
                                     feature_spec if feature_spec.exists() else None)
    except (ResidenceZoneError, FileNotFoundError, ValueError) as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return EXIT_RESOURCE_MISSING

    print(f"Ring table: {len(table)} fine zones, {len(table.secteurs)} "
          f"sectors, version {table.meta.get('version', '?')}")

    all_failures: list[tuple[Path, list[str]]] = []
    framing_gaps: list[Path] = []

    for path in args.population:
        if not path.exists():
            print(f"[ERREUR] Population introuvable : {path}", file=sys.stderr)
            return EXIT_RESOURCE_MISSING
        population = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n=== {path}")
        try:
            counts = enrich(population, table, resolver)
            checks = audit(population, table, zones)
        except ResidenceZoneError as exc:
            print(f"[ERREUR] {exc}", file=sys.stderr)
            return EXIT_RESOURCE_MISSING

        failures = report(counts, checks)
        if print_framing(checks):
            framing_gaps.append(path)
        if failures:
            all_failures.append((path, failures))

        target = destination(path, args.out, len(args.population))
        if args.dry_run:
            print(f"  [dry-run] {target} not written")
            continue
        # Atomic write: a crash in the middle of writing must not leave a
        # truncated population behind.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(population, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        print(f"  written → {target}")

    if all_failures:
        print("\n── Contradicted gates ──────────────────────────────────────────────────")
        for path, failures in all_failures:
            for failure in failures:
                print(f"  [{path.name}] {failure}")
        if args.check:
            return EXIT_GATE_FAILED
        print("  (informative: rerun with --check to make it a failure)")
        return EXIT_OK

    if args.check:
        print("\nAll gates pass.")
        if framing_gaps:
            print(f"  → code {EXIT_FRAMING_GAP}: the gap to the framing exceeds the tolerance "
                  f"on {', '.join(p.name for p in framing_gaps)}. This is axis A9 — the "
                  f"spatial over-concentration of the draw —, not a defect of this trait: "
                  f"fixing it requires reworking the draw (another ticket). The code is "
                  f"distinct so that the caller can track it without confusing it with "
                  f"a failure.")
            return EXIT_FRAMING_GAP
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
