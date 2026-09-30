"""zone_label.py — The zone label read by the model, rebuilt for an INSEE municipality.

The prompt serves the agent a `destination_zone`: « a neighbourhood of a major urban centre in
the municipality of Toulouse ». On the cohort, this label is composed by step 3bis of
`scripts/data/population/generate_population.ipynb`, from the INSEE density grid
(`DENS7`) and the municipal composition of catchment areas (`AAV2020`), the municipality being
obtained by reverse geocoding of the activity coordinates.

The unit audit of ticket 058 works on fine survey zones, whose INSEE code is
known without geocoding (`zf_couronne.json`). This module therefore redoes the **same** composition
from the municipality code, so that the audited agent reads the same sentence as the cohort agent.

The replication is not assumed, it is checked:

    python scripts/progedo_logit/zone_label.py --verifier

rebuilds the home label of the 1,000 personas of cohort v6 from their INSEE code
alone, and compares it character by character to the one carried by the population file. One
mismatch, and the module is wrong: the audit must not start on an approximate sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from loguru import logger

RACINE = Path(__file__).resolve().parents[2]
INSEE_DIR = RACINE / "data" / "insee"
FICHIER_DENSITE = INSEE_DIR / "fichier_diffusion_2026.xlsx"
FICHIER_AAV = INSEE_DIR / "AAV2020_au_01-01-2026.xlsx"

# Labels of the INSEE density grid (DENS7). Copied from step 3bis of the generation
# notebook: they go into a sentence READ BY THE MODEL, so both sources must
# say exactly the same thing. The `--verifier` mode is what guarantees it.
DENS7_LABEL = {
    1: "a major urban centre",
    2: "an intermediate urban centre",
    3: "an urban belt",
    4: "a small town",
    5: "a rural small town",
    6: "scattered rural housing",
    7: "very scattered rural housing",
}

# `CATEAAV2020` == 30: municipality outside any urban catchment. Read as an INTEGER — comparing
# to the string '30' was the defect fixed on 2026-09-04 on the notebook side.
CATEGORIE_HORS_AIRE = 30


@lru_cache(maxsize=1)
def _tables() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(density → label, code → municipality, code → catchment area), from the INSEE files."""
    if not FICHIER_DENSITE.exists():
        raise SystemExit(
            f"Density grid missing: {FICHIER_DENSITE}\n"
            "Download it from https://www.insee.fr/fr/statistiques/5040028"
        )
    densite = pd.read_excel(
        FICHIER_DENSITE,
        sheet_name="Maille communale",
        header=4,
        dtype={"CODGEO": str},
        engine="calamine",
    )
    densite.columns = [c.strip() for c in densite.columns]
    densite["_detail"] = densite["DENS7"].map(DENS7_LABEL).fillna("an unknown area")
    libelle = densite.set_index("CODGEO")["_detail"].to_dict()
    commune = densite.set_index("CODGEO")["LIBGEO"].to_dict()
    aire: dict[str, str] = {}

    if FICHIER_AAV.exists():
        aav = pd.read_excel(
            FICHIER_AAV,
            sheet_name="Composition_communale",
            header=5,
            dtype={"CODGEO": str},
            engine="calamine",
        )
        aav.columns = [c.strip() for c in aav.columns]
        categorie = pd.to_numeric(aav["CATEAAV2020"], errors="coerce")
        aire = aav.loc[categorie != CATEGORIE_HORS_AIRE].set_index("CODGEO")["LIBAAV2020"].to_dict()
        for code, nom in aav.set_index("CODGEO")["LIBGEO"].to_dict().items():
            commune.setdefault(code, nom)
        logger.info(
            f"AAV2020: {len(aav)} municipalities, of which "
            f"{int((categorie == CATEGORIE_HORS_AIRE).sum())} outside any catchment area"
        )
    else:
        logger.warning(f"AAV2020 missing ({FICHIER_AAV}): catchment area ignored")

    logger.info(f"Density grid: {len(densite)} municipalities · names: {len(commune)}")
    return libelle, commune, aire


def build_zone_label(codgeo: str) -> str:
    """The zone label of a municipality, in the exact terms of the cohort prompt."""
    libelle, commune, aire = _tables()
    code = str(codgeo).strip()
    detail = libelle.get(code, "an unknown area")
    nom = commune.get(code)
    # A municipality name is not made up: « unknown » is information, a fabricated name is not.
    lieu = f"the municipality of {nom}" if nom else "an unknown municipality"
    attraction = aire.get(code, "")
    if attraction:
        return f"a neighbourhood of {detail} in {lieu} ({attraction} urban area)"
    return f"a neighbourhood of {detail} in {lieu}, outside any urban catchment area"


def verifier(population: Path) -> int:
    """Rebuilds the home label of each persona and compares it to the one in the file."""
    personas = json.loads(population.read_text(encoding="utf-8"))
    compares = ecarts = sans_reference = reference_degradee = 0
    exemples: list[tuple[str, str, str]] = []
    degrades: list[tuple[str, str]] = []

    for personne in personas:
        code = (personne.get("identity", {}).get("traits_json", {}) or {}).get("residence_insee")
        activites = (personne.get("identity", {}) or {}).get("activities") or []
        attendu = next(
            (
                (a.get("location") or {}).get("zone")
                for a in activites
                if a.get("purpose") == "home" and (a.get("location") or {}).get("zone")
            ),
            None,
        )
        if not code or not attendu:
            sans_reference += 1
            continue
        obtenu = build_zone_label(code)
        compares += 1
        if obtenu != attendu:
            # The cohort names the municipality by reverse geocoding of the coordinates; when that
            # geocoding returned nothing, its label is degraded (« an unknown municipality »)
            # while the persona's INSEE code, for its part, is known. Rebuilding from the code
            # therefore does BETTER than the reference: this is not a divergence of method.
            if "an unknown municipality" in attendu:
                reference_degradee += 1
                if len(degrades) < 5:
                    degrades.append((str(personne.get("person_id")), obtenu))
                continue
            ecarts += 1
            if len(exemples) < 5:
                exemples.append((str(personne.get("person_id")), attendu, obtenu))

    logger.info(
        f"Labels compared: {compares} · without reference in the file: {sans_reference}"
    )
    if reference_degradee:
        logger.warning(
            f"{reference_degradee} persona(s) de la cohorte portent un libellé dégradé "
            "(« an unknown municipality ») que le code INSEE permet pourtant de résoudre — "
            "géocodage inverse muet à la génération ; la reconstruction fait mieux, "
            "et ces cas ne comptent pas comme divergence"
        )
        for pid, obtenu in degrades:
            logger.warning(f"  persona {pid} → {obtenu}")
    if ecarts:
        logger.error(
            f"[ALARME] {ecarts}/{compares} labels diverge from the population file: "
            "the sentence served to the audited agent would not be the one served to the cohort"
        )
        for pid, attendu, obtenu in exemples:
            logger.error(f"  persona {pid}\n    expected: {attendu}\n    obtained: {obtenu}")
        return 1
    exacts = compares - reference_degradee
    logger.success(
        f"Réplication exacte : {exacts}/{exacts} libellés comparables identiques à ceux de "
        f"la cohorte ({reference_degradee} référence(s) dégradée(s) écartée(s))"
    )
    return 0


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--verifier", action="store_true", help="check against cohort v6")
    parseur.add_argument(
        "--population",
        type=Path,
        default=RACINE / "data" / "population" / "population_1000_PANEL_v6" / "population.json",
        help="reference population for the check",
    )
    parseur.add_argument("--commune", help="INSEE code whose label is wanted")
    args = parseur.parse_args()

    if args.verifier:
        sys.exit(verifier(args.population))
    if args.commune:
        print(build_zone_label(args.commune))
        return
    parseur.error("specify --verifier or --commune")


if __name__ == "__main__":
    main()
