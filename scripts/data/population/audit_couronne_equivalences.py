"""audit_couronne_equivalences.py — Ticket 021, lot 0: the two equivalences, measured.

Ticket 021 sets the residence ring on the persona by locating its home in a
fine zone, then reading the ring of the **sampling sector** carried by the first three
digits of the `ZF` code. The rest of the ticket rests on two equivalences that
the first draft asserted without having measured them:

1. classifying a point by **fine-zone code prefix** yields the same ring as
   classifying it by **geometric membership** in the ring polygons;
2. "outside the **fine-zone layer**" (`zone_resolver.resolve()` returns `None`) designates
   exactly the same set as "outside the **four rings**" (`hors périmètre`).

These are not the same objects: the first pits an attachment by code against a
spatial join, the second pits two footprints built separately (the union of the 785
fine zones against the dissolution of the 88 sectors). Nothing forces the two to
coincide a priori — hence this measurement, whose result decides the shape of lots 1 and 2.

The script MODIFIES NOTHING. It measures, writes a report, and returns an exit code.

**The cross-check is independent** (gate E): the classification recomputed here is set against
the `zone_communale` column of `agents_reclassement.csv`, produced on 2026-08-24 by the other
path during ticket 020. Agreement between two measurements of the same path would be worthless.

Exit codes:
  0  all seven gates pass — lots 1 to 5 can be written as they stand
  1  versioned resource missing
  2  at least one gate FAILS — the ticket must be redesigned before writing code
  3  at least one gate is NOT MEASURABLE (restricted-access data missing), and an
     unmeasured gate is a gate that passes: the script refuses to keep quiet about it

Usage:
    make audit-couronnes
    services/llm-agents/.venv/bin/python -m scripts.data.population.audit_couronne_equivalences \
      --population data/population/toulouse_population_1000.json \
      --trace docs/traces/2026-08-24_couronne_equivalences
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER  # noqa: E402

EXIT_OK, EXIT_RESOURCE_MISSING, EXIT_GATE_FAILED, EXIT_NOT_MEASURABLE = 0, 1, 2, 3

SIG = REPO_ROOT / "data" / "PROGEDO 2023" / "lil-1750-Documentation" / "SIG"
SIG_DTIR = SIG / "EMC2_Toulouse_2023_DTIR_17072023.shp"
SIG_ZF = SIG / "EMC2_Toulouse_2023_ZF_26052023.shp"
ZF_GPKG = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_zones.gpkg"
COURONNE_GEOJSON = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "couronne_perimetre.geojson"
ZF_TABLE = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_couronne.json"
DEFAULT_POPULATION = (REPO_ROOT / "data" / "population"
                      / "toulouse_population_1000.json")
DEFAULT_ARCHIVE = (REPO_ROOT / "docs" / "traces" / "2026-08-24_perimetre_population"
                   / "agents_reclassement.csv")

NOT_MEASURABLE = "NON MESURABLE"


def title(text: str) -> None:
    print(f"\n{'─' * 78}\n{text}\n{'─' * 78}")


def person_id(person: dict) -> str:
    identity = person.get("identity") or {}
    for key in ("id", "person_id", "agent_id"):
        if person.get(key) is not None:
            return str(person[key])
        if identity.get(key) is not None:
            return str(identity[key])
    return ""


def secteur_couronne_map() -> tuple[Optional[dict], str]:
    """`secteur → couronne`, from the published table if it exists, else the GIS layer.

    Order matters: after lot 1 the `zf_couronne.json` table is versioned and this
    audit runs WITHOUT the restricted-access data. Before lot 1, there is only the
    shapefile — and then the gates depending on it are declared not measurable
    rather than silently skipped.
    """
    if ZF_TABLE.exists():
        from mobility_core.residence_zone import CouronneTable
        return (CouronneTable.load(ZF_TABLE).secteurs,
                f"{ZF_TABLE.name} (table versionnée)")
    if SIG_DTIR.exists():
        import geopandas as gpd
        dtir = gpd.read_file(SIG_DTIR)
        return (dict(zip(dtir["NUM_DTIR"].astype(str), dtir["NOM_D2"])),
                f"{SIG_DTIR.name} (accès restreint lil-1750)")
    return None, "aucune source de couronne par secteur"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE,
                        help="Independent cross-check CSV (ticket 020)")
    parser.add_argument("--trace", type=Path, default=None,
                        help="Directory where the JSON report is archived")
    args = parser.parse_args()

    for resource in (ZF_GPKG, COURONNE_GEOJSON):
        if not resource.exists():
            print(f"[ERREUR] resource missing: {resource} "
                  f"(make zones / make communes-couronnes)", file=sys.stderr)
            return EXIT_RESOURCE_MISSING

    import geopandas as gpd
    from pyproj import Transformer

    from mobility_core.residence_zone import CommunalZones
    from mobility_core.zone_resolver import ZoneResolver

    secteurs, source = secteur_couronne_map()
    report: dict = {"generated_at": date.today().isoformat(), "ticket": "021",
                    "lot": 0, "source_couronne_par_secteur": source, "portes": {}}
    portes: dict[str, object] = {}

    # ── A · attaching fine zone → sector ─────────────────────────────────────
    title("A · Attachment by prefix: the 785 fine zones to the 88 sectors")
    gpkg = gpd.read_file(ZF_GPKG)
    gpkg["prefixe"] = gpkg["ZF"].astype(str).str[:3]
    prefixes = sorted(set(gpkg["prefixe"]))

    if secteurs is None:
        print(f"{NOT_MEASURABLE} — {source}")
        portes["A · aucune zone fine orpheline"] = NOT_MEASURABLE
        portes["A · tous les secteurs atteints"] = NOT_MEASURABLE
        orphelines, sans_zone = [], []
    else:
        orphelines = sorted(set(prefixes) - set(secteurs))
        sans_zone = sorted(set(secteurs) - set(prefixes))
        inconnues = sorted(set(secteurs.values()) - set(COURONNES))
        print(f"known sectors                     : {len(secteurs)}")
        print(f"distinct prefixes over 785 zones  : {len(prefixes)}")
        print(f"orphan fine zones                 : {len(orphelines)} {orphelines[:5]}")
        print(f"sectors without any fine zone     : {sans_zone or '—'}")
        print(f"values outside COURONNES          : {inconnues or '—'}")
        portes["A · aucune zone fine orpheline"] = not orphelines
        portes["A · tous les secteurs atteints"] = not sans_zone
    report["A"] = {"n_secteurs": len(secteurs) if secteurs else None,
                   "n_prefixes": len(prefixes), "orphelines": orphelines,
                   "secteurs_sans_zone": sans_zone}

    zones = CommunalZones.load(COURONNE_GEOJSON)

    # ── B · equivalence 1, at fine-zone grain ────────────────────────────────
    title("B · Equivalence 1 — prefix against geometric membership (785 zones)")
    desaccords_b: list[dict] = []
    if secteurs is None:
        print(f"{NOT_MEASURABLE} — {source}")
        portes["B · préfixe == géométrie sur les 785 zones"] = NOT_MEASURABLE
    else:
        to_wgs = Transformer.from_crs(2154, 4326, always_xy=True)
        for row in gpkg.itertuples():
            attendu = secteurs.get(row.prefixe)
            # Two points per zone: the centroid published in the resource — the one the
            # lot 1 test will use — and the polygon's representative point, which stays
            # inside even for a concave zone.
            lon_c, lat_c = to_wgs.transform(row.XL93, row.YL93)
            rep = row.geometry.representative_point()
            lon_r, lat_r = to_wgs.transform(rep.x, rep.y)
            geo_c, geo_r = zones.classify(lat_c, lon_c), zones.classify(lat_r, lon_r)
            if attendu != geo_c or attendu != geo_r:
                desaccords_b.append({"zf": str(row.ZF), "par_prefixe": attendu,
                                     "geo_centroide": geo_c,
                                     "geo_representatif": geo_r})
        accord = len(gpkg) - len(desaccords_b)
        print(f"agreement: {accord}/{len(gpkg)} = {100.0 * accord / len(gpkg):.2f} %")
        for item in desaccords_b[:12]:
            print(f"  ZF {item['zf']} — prefix « {item['par_prefixe']} » / centroid "
                  f"« {item['geo_centroide']} » / representative "
                  f"« {item['geo_representatif']} »")
        portes["B · préfixe == géométrie sur les 785 zones"] = not desaccords_b
    report["B"] = {"n": len(gpkg), "desaccords": desaccords_b}

    # ── C · equivalence 1, at home grain ─────────────────────────────────────
    title("C · Equivalence 1 — the population's homes, prefix against geometry")
    if not args.population.exists():
        print(f"[ERREUR] population missing: {args.population}", file=sys.stderr)
        return EXIT_RESOURCE_MISSING
    raw = json.loads(args.population.read_text(encoding="utf-8"))
    people = raw if isinstance(raw, list) else (raw.get("people") or raw.get("personas"))
    resolver = ZoneResolver.load()

    rows: list[dict] = []
    for person in people:
        home = (person.get("identity") or {}).get("home") or {}
        lat, lon = home.get("lat"), home.get("lon")
        zone = resolver.resolve(lat, lon) if lat is not None and lon is not None else None
        if zone is None:
            par_prefixe = OUT_OF_PERIMETER
        elif secteurs is None:
            par_prefixe = NOT_MEASURABLE
        else:
            par_prefixe = secteurs.get(str(zone.zf)[:3], "SECTEUR INCONNU")
        rows.append({"person_id": person_id(person),
                     "zf": str(zone.zf) if zone else "",
                     "par_prefixe": par_prefixe,
                     "geometrique": zones.classify(lat, lon)})

    if secteurs is None:
        print(f"{NOT_MEASURABLE} — {source}")
        portes["C · préfixe == géométrie sur les domiciles"] = NOT_MEASURABLE
        desaccords_c: list[dict] = []
    else:
        desaccords_c = [r for r in rows if r["par_prefixe"] != r["geometrique"]]
        accord = len(rows) - len(desaccords_c)
        print(f"agreement: {accord}/{len(rows)} = {100.0 * accord / len(rows):.2f} %")
        print("distribution by prefix  :", dict(Counter(r["par_prefixe"] for r in rows)))
        print("geometric distribution  :", dict(Counter(r["geometrique"] for r in rows)))
        for item in desaccords_c[:12]:
            print(f"  {item['person_id']} ZF {item['zf'] or '—'} — prefix "
                  f"« {item['par_prefixe']} » / geometry « {item['geometrique']} »")
        portes["C · préfixe == géométrie sur les domiciles"] = not desaccords_c
    report["C"] = {"n": len(rows), "n_desaccords": len(desaccords_c),
                   "desaccords": desaccords_c[:50]}

    # ── D · equivalence 2, the two footprints ────────────────────────────────
    title("D · Equivalence 2 — outside the fine-zone LAYER against outside the SCOPE")
    hors_couche = {r["person_id"] for r in rows if not r["zf"]}
    hors_perimetre = {r["person_id"] for r in rows
                      if r["geometrique"] == OUT_OF_PERIMETER}
    couche_seule = sorted(hors_couche - hors_perimetre)
    perimetre_seul = sorted(hors_perimetre - hors_couche)
    print(f"outside the fine-zone layer       : {len(hors_couche)}")
    print(f"outside the four rings            : {len(hors_perimetre)}")
    print(f"outside the layer but in scope    : {couche_seule[:10] or '—'} "
          f"({len(couche_seule)})")
    print(f"in the layer but outside scope    : {perimetre_seul[:10] or '—'} "
          f"({len(perimetre_seul)})")
    coverage = resolver.coverage()
    print(f"resolver coverage                 : {coverage}")
    portes["D · hors couche == hors périmètre"] = not (couche_seule or perimetre_seul)
    report["D"] = {"hors_couche": len(hors_couche),
                   "hors_perimetre": len(hors_perimetre),
                   "couche_seule": couche_seule, "perimetre_seul": perimetre_seul,
                   "coverage": coverage}

    # ── E · independent cross-check ──────────────────────────────────────────
    title("E · Cross-check against the ticket 020 trace (independent path)")
    if not args.archive.exists():
        print(f"{NOT_MEASURABLE} — trace missing: {args.archive}")
        portes["E · accord avec la trace du ticket 020"] = NOT_MEASURABLE
        ecarts_e: list[tuple] = []
        archive: dict = {}
    else:
        archive = {row["person_id"]: row
                   for row in csv.DictReader(args.archive.open(encoding="utf-8"))}
        apparies = [r for r in rows if r["person_id"] in archive]
        ecarts_e = [(r["person_id"], r["par_prefixe"],
                     archive[r["person_id"]]["zone_communale"])
                    for r in apparies
                    if r["par_prefixe"] != archive[r["person_id"]]["zone_communale"]]
        print(f"matched personas                  : {len(apparies)}/{len(rows)}")
        print(f"agreement prefix ↔ archived column: "
              f"{len(apparies) - len(ecarts_e)}/{len(apparies)}")
        for pid, recalcule, archive_value in ecarts_e[:12]:
            print(f"  {pid} — recomputed « {recalcule} » / archived « {archive_value} »")
        portes["E · accord avec la trace du ticket 020"] = (
            not ecarts_e if apparies else NOT_MEASURABLE)
    report["E"] = {"n_ecarts": len(ecarts_e), "ecarts": ecarts_e[:50]}

    # ── F · the municipality, which the lot 1 table must publish ─────────────
    title("F · ZF → INSEE: can the municipality be reproduced from the zone code?")
    ecarts_f: list[tuple] = []
    if not SIG_ZF.exists() and not ZF_TABLE.exists():
        print(f"{NOT_MEASURABLE} — neither {ZF_TABLE.name} nor the fine-zone GIS layer")
        portes["F · ZF → INSEE reproduit la commune archivée"] = NOT_MEASURABLE
    elif not archive:
        print(f"{NOT_MEASURABLE} — no cross-check trace")
        portes["F · ZF → INSEE reproduit la commune archivée"] = NOT_MEASURABLE
    else:
        if ZF_TABLE.exists():
            from mobility_core.residence_zone import CouronneTable
            table = CouronneTable.load(ZF_TABLE)
            zf_insee = {z: table.commune_of_zf(z)[0]
                        for z in {r["zf"] for r in rows if r["zf"]}
                        if table.commune_of_zf(z)}
        else:
            zf_shp = gpd.read_file(SIG_ZF)
            zf_insee = {str(z): str(i)
                        for z, i in zip(zf_shp["ZF"], zf_shp["INSEE"])}
        manquantes = sorted({r["zf"] for r in rows if r["zf"] and r["zf"] not in zf_insee})
        for row in rows:
            if not row["zf"] or row["person_id"] not in archive:
                continue
            attendu = archive[row["person_id"]]["INSEE"]
            obtenu = zf_insee.get(row["zf"], "")
            if attendu and obtenu != attendu:
                ecarts_f.append((row["person_id"], row["zf"], obtenu, attendu))
        print(f"fine zones without INSEE          : {len(manquantes)}")
        print(f"INSEE gaps recomputed ↔ archived  : {len(ecarts_f)}")
        for pid, zf, obtenu, attendu in ecarts_f[:12]:
            print(f"  {pid} ZF {zf} — recomputed {obtenu} / archived {attendu}")
        portes["F · ZF → INSEE reproduit la commune archivée"] = not (ecarts_f
                                                                     or manquantes)
        report["F"] = {"zf_sans_insee": manquantes, "n_ecarts": len(ecarts_f),
                       "ecarts": ecarts_f[:50]}

    # ── verdict ──────────────────────────────────────────────────────────────
    title("VERDICT")
    for label, state in portes.items():
        mark = NOT_MEASURABLE if state == NOT_MEASURABLE else ("PASSE" if state
                                                               else "ÉCHOUE")
        print(f"  {mark:14} {label}")
    report["portes"] = {k: (v if v == NOT_MEASURABLE else bool(v))
                        for k, v in portes.items()}

    if args.trace:
        args.trace.mkdir(parents=True, exist_ok=True)
        out = args.trace / "couronne_equivalences.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        # `--trace` may be relative: `relative_to` would then raise, AFTER the write.
        shown = out.resolve()
        try:
            shown = shown.relative_to(REPO_ROOT)
        except ValueError:
            pass
        print(f"\nreport → {shown}")

    if any(state is False for state in portes.values()):
        print("\n  → a gate FAILS: ticket 021 must be redesigned before any code.")
        return EXIT_GATE_FAILED
    if any(state == NOT_MEASURABLE for state in portes.values()):
        print(f"\n  → at least one gate {NOT_MEASURABLE}: an unmeasured gate is a "
              f"gate that passes, it is not a verdict.")
        return EXIT_NOT_MEASURABLE
    print("\n  → the gates pass: lots 1 to 5 can be written as they stand.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
