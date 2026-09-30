"""
export_commune_couronne.py — The missing data of ticket 020: which municipality is
in which ring, and where the survey scope ends.

WHAT IT PRODUCES. Three resources, in `mobility_core/data/`:

1. `commune_couronne.json` — the 453 municipalities of the EMC² 2023 scope with their INSEE
   code and their ring (`Toulouse` / `1ere couronne` / `2eme couronne` /
   `3eme couronne`). It is the survey's reference data, not a reconstruction.
2. `couronne_perimetre.geojson` — the geometry of the four rings, in WGS84,
   allowing a home to be classified by **membership** and not by distance.
3. `zf_couronne.json` — the 785 fine zones with their sampling sector, their ring,
   their INSEE code and their municipality (ticket 021). It is the resource read by
   `mobility_core.residence_zone`: it gives the ring of a home WITHOUT geometry
   at runtime, since `zone_resolver` already gives its fine zone. The grain is the fine
   zone and not the sector, because a `secteur → couronne` table of 88 rows would not
   carry the municipality — and the municipality is what makes the classification auditable.

WHY. `geo_reference.residence_zone` classified a home by its **distance to the
city core** (under 8 km = Toulouse, 20 = 1st ring, 40 = 2nd), with the
comment « these are the modalities of `lieu_residence` of the EMC² reference ». They
are not: the survey splits by LIST OF MUNICIPALITIES. An administrative ring
is not a metric annulus, and ticket 020 measured the consequence on
`toulouse_population_1000.json`: **24.4 % of agents change ring**, including 66
that the 8 km disc labels « Toulouse » while they live in Blagnac, Balma,
Colomiers or Ramonville — compared to a car target of 31 % instead of 64 %.

SOURCE. GIS layer of the survey, `data/PROGEDO 2023/lil-1750-Documentation/SIG/`:

- `EMC2_Toulouse_2023_DTIR_17072023.shp` — the 88 sampling sectors, carrying the field
  `NOM_D2` which IS the survey's split into rings;
- `EMC2_Toulouse_2023_ZF_26052023.shp` — the 785 fine zones, carrying `COM` and `INSEE`.

The fine zone → sector link goes through the code: the first three digits of the
`ZF` code are the `NUM_DTIR`. Checked at 100 % on the 785 zones; the export fails if a
single code does not link, rather than leave a zone without a ring.

⚠ THE GIS LAYER IS AUTHORITATIVE, AND IT CONTRADICTS THE PUBLICATION ON ONE MUNICIPALITY. It gives
1 / 69 / 108 / 275, the CEREMA publication states 1 / 68 / 109 / 275. The total, 453, is
the same. The layer is kept: it is the one that defines the sectors on which the
adjustment weights were computed, hence the one that defines the targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from mobility_core.population_reference import (  # noqa: E402
    COURONNES, COURONNES_FR, couronne_canonique)
SIG_DIR = REPO_ROOT / "data" / "PROGEDO 2023" / "lil-1750-Documentation" / "SIG"
DTIR_SHP = SIG_DIR / "EMC2_Toulouse_2023_DTIR_17072023.shp"
ZF_SHP = SIG_DIR / "EMC2_Toulouse_2023_ZF_26052023.shp"

OUT_DIR = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data"
OUT_TABLE = OUT_DIR / "commune_couronne.json"
OUT_GEOJSON = OUT_DIR / "couronne_perimetre.geojson"
OUT_ZF_TABLE = OUT_DIR / "zf_couronne.json"

# The survey's GIS layer writes its rings in FRENCH in `NOM_D2` — it is the
# source, it does not move. The produced RESOURCE, for its part, carries the canonical modality of the
# system, English since ticket 074: `CouronneTable.load()` validates it against
# `COURONNES` and would refuse a French file. The translation is therefore done here, once,
# at write time — and the inventory of expected modalities remains that of the layer.
COURONNES_SOURCE = COURONNES_FR


def build() -> dict:
    for path in (DTIR_SHP, ZF_SHP):
        if not path.exists():
            raise SystemExit(
                f"GIS layer missing: {path}\n"
                "PROGEDO data are restricted-access (lil-1750); this export "
                "requires them. The produced resource, for its part, is versioned.")

    dtir = gpd.read_file(DTIR_SHP)
    zf = gpd.read_file(ZF_SHP)

    inconnues = set(dtir["NOM_D2"].unique()) - set(COURONNES_SOURCE)
    if inconnues:
        raise SystemExit(
            f"Unexpected ring modalities in NOM_D2: {sorted(inconnues)}. "
            "They must be exactly those of cerema_values.yaml.")

    # ONE single translation, here, right after validation: everything that follows — municipality
    # table, fine-zone table, geometry, counters — then carries the canonical modality.
    # Translating further on would have meant translating it four times, and forgetting one.
    dtir["NOM_D2"] = dtir["NOM_D2"].map(couronne_canonique)

    zf = zf.assign(num_dtir=zf["ZF"].astype(str).str[:3])
    orphelines = ~zf["num_dtir"].isin(dtir["NUM_DTIR"].astype(str))
    if orphelines.any():
        raise SystemExit(
            f"{int(orphelines.sum())} fine zone(s) without sampling sector: "
            f"{zf.loc[orphelines, 'ZF'].tolist()[:10]}. The link by code prefix "
            "no longer holds — a spatial join is needed.")

    # The absence of orphans says that each zone finds a sector; it does not say that
    # each sector is reached. A sector without fine zone would mean that the ZF layer
    # and the DTIR layer do not describe the same scope — and the published table would
    # hide it. Check added in ticket 021, lot 1.
    sans_zone = sorted(set(dtir["NUM_DTIR"].astype(str)) - set(zf["num_dtir"]))
    if sans_zone:
        raise SystemExit(
            f"{len(sans_zone)} sampling sector(s) without any fine zone: {sans_zone}. "
            "The two layers do not describe the same scope.")

    joined = zf.merge(dtir[["NUM_DTIR", "NOM_D2"]], left_on="num_dtir",
                      right_on="NUM_DTIR", how="left")

    # A municipality straddling two rings would make the table ambiguous: we refuse
    # rather than decide through a silent `mode()`.
    a_cheval = joined.groupby("INSEE")["NOM_D2"].nunique()
    if (a_cheval > 1).any():
        coupables = a_cheval[a_cheval > 1].index.tolist()
        raise SystemExit(
            f"Municipalities linked to several rings: {coupables}. "
            "The municipality → ring table is not a function; manual arbitration required.")

    table = (joined.groupby("INSEE")
             .agg(commune=("COM", "first"), couronne=("NOM_D2", "first"))
             .reset_index()
             .sort_values("INSEE"))

    counts = table["couronne"].value_counts().to_dict()
    payload = {
        "version": "cc1",
        "source": {
            "survey": "EMC² Toulouse 2023 (ProGEDO / lil-1750)",
            "dtir_layer": DTIR_SHP.name,
            "zf_layer": ZF_SHP.name,
            "field": "NOM_D2",
        },
        "note": ("Découpage par liste de communes, tel que défini par la couche SIG de "
                 "l'enquête. La publication CEREMA annonce 68 et 109 communes pour les "
                 "1ʳᵉ et 2ᵉ couronnes ; la couche en donne 69 et 108, pour le même total "
                 "de 453. La couche fait foi : c'est sur ses secteurs que les poids de "
                 "redressement ont été calculés."),
        "counts": {z: int(counts.get(z, 0)) for z in COURONNES},
        "n_communes": int(len(table)),
        "communes": [
            {"insee": r.INSEE, "commune": r.commune, "couronne": r.couronne}
            for r in table.itertuples()
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    # Table at FINE ZONE grain (ticket 021). The runtime resolves a home to a fine zone
    # then reads this row: no spatial join, and the municipality comes along.
    zones_table = (joined[["ZF", "num_dtir", "NOM_D2", "INSEE", "COM"]]
                   .rename(columns={"ZF": "zf", "num_dtir": "secteur",
                                    "NOM_D2": "couronne", "INSEE": "insee",
                                    "COM": "commune"})
                   .astype({"zf": str, "secteur": str, "insee": str})
                   .sort_values("zf"))
    zf_counts = zones_table["couronne"].value_counts().to_dict()
    zf_payload = {
        "version": "zc1",
        "source": payload["source"],
        "note": ("Les 785 zones fines de l'enquête avec leur secteur de tirage, leur "
                 "couronne, leur code INSEE et leur commune. Le rattachement zone → "
                 "secteur passe par les TROIS PREMIERS CHIFFRES du code ZF ; mesuré "
                 "identique à 100 % au classement par appartenance géométrique, sur les "
                 "785 zones comme sur 1 021 domiciles (ticket 021, lot 0, trace "
                 "docs/traces/2026-08-24_couronne_equivalences/). Hors de cette couche, "
                 "la couronne ne se devine pas : c'est `hors périmètre`."),
        "n_zones": int(len(zones_table)),
        "n_secteurs": int(zones_table["secteur"].nunique()),
        "counts": {z: int(zf_counts.get(z, 0)) for z in COURONNES},
        "secteurs": [
            {"secteur": s, "couronne": c}
            for s, c in sorted(zones_table.drop_duplicates("secteur")
                               .set_index("secteur")["couronne"].items())
        ],
        "zones": [
            {"zf": r.zf, "secteur": r.secteur, "couronne": r.couronne,
             "insee": r.insee, "commune": r.commune}
            for r in zones_table.itertuples()
        ],
    }
    OUT_ZF_TABLE.write_text(json.dumps(zf_payload, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    # Geometry of the rings: dissolution of the sectors by NOM_D2, reprojected to
    # WGS84 because it is the frame of the homes of the synthetic population.
    zones = (dtir[["NOM_D2", "geometry"]].dissolve(by="NOM_D2").reset_index()
             .to_crs(4326))
    zones = zones.rename(columns={"NOM_D2": "couronne"})
    zones["couronne"] = pd.Categorical(zones["couronne"], categories=COURONNES,
                                       ordered=True)
    zones = zones.sort_values("couronne")
    zones.to_file(OUT_GEOJSON, driver="GeoJSON")

    payload["zf_table"] = {k: v for k, v in zf_payload.items() if k != "zones"}
    return payload


def main() -> None:
    payload = build()
    print(f"→ {OUT_TABLE.relative_to(REPO_ROOT)}  ({payload['n_communes']} municipalities)")
    print(f"→ {OUT_GEOJSON.relative_to(REPO_ROOT)}")
    zf_table = payload["zf_table"]
    print(f"→ {OUT_ZF_TABLE.relative_to(REPO_ROOT)}  "
          f"({zf_table['n_zones']} fine zones, {zf_table['n_secteurs']} sectors)")
    for zone, n in payload["counts"].items():
        print(f"   {zone:16s} {n:4d} municipalities")
    print("\nExpected totals (population_emc2_2023.yaml): 1 / 69 / 108 / 275 = 453")


if __name__ == "__main__":
    main()
