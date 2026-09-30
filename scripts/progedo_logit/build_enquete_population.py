"""build_enquete_population.py — The survey's declared days, as a readable population.

The unit audit of ticket 058 measures trip-by-trip agreement between a decision-maker
and the mode that a real person declared. The sealed cohort cannot carry it: none
of its personas declared anything at all. The reference is therefore the survey, and it must be
presented to the decision chain in the form it can read — a population.

What the script writes:

    data/population/population_enquete_058_<split>/population.json   the respondents and their day
    data/population/population_enquete_058_<split>/dates_meteo.json  the described day, per person
    scripts/progedo_logit/verite_058_<split>.csv                     the declared mode, separately

`dates_meteo.json` lives **next to** the population, not inside it: the date of the described day
is not a trait of the person, it must enter neither the narrative nor the key of the
decision cache. The runner picks it up by convention when the file exists, and the
« one weather per agent » mechanism then reads the actual day instead of drawing one.

**The ground truth lives in a second file, and this is structural.** The declared mode is the
answer to the problem put to the decision-maker. In the population, it would end up in `traits_json`, hence
in the prompt narrative and in the key of the decision cache. It stays outside.

## What is served to the model, and what is not

The agent narrative (`_build_profile_narrative`) reads seven fields: first name, age, occupation,
household size, income, usual purposes, ring of residence. Four fall outside the 21
variables of the parity contract, and the author's decision is to serve what the survey
allows to be rebuilt, without fabricating anything:

| Field | Decision | Reason |
|---|---|---|
| `residence_zone` | **served** | the ring is deduced from the fine zone of residence |
| `travel_purposes` | **served** | the purposes of the declared day, as on the cohort |
| `name` | **drawn by seed** | see below: omitting it makes it fabricated downstream, without a seed |
| `income` | omitted | EMC² Toulouse 2023 carries no income variable |
| `professional_activity` | omitted | its nomenclature comes from eqasim, not from the survey; the narrative
  then falls back on `main_occupation`, which is its documented fallback path |

`personal_bike` has not entered the narrative since 2026-08-26, but the chain rule
reads it to know whether the bike can be taken: it is therefore set from `has_bike`, one of the 21.

**The first name cannot simply be omitted.** `eqasim_loader.load_population_from_data`
replaces an empty `name` with a `Faker("fr_FR")` draw **without a seed**: the prompt would change
from one run to another, and with it the key of the decision cache. The first name is therefore drawn
here, by a seed derived from the `person_id`, exactly as cohort v6 has done since
ticket 074. No survey information is fabricated along the way: the first name is synthetic
on both sides of the comparison, and that is what keeps the same form for the prompt.

**The zone label is written, and the loader throws it away.** Each activity carries a `zone`
composed by `zone_label.py`, as on the cohort. `eqasim_loader._parse_activity` nonetheless builds
its `Location` without that field: on the v6 reference arm, the prompt therefore goes out with
`destination_zone: None` (7 narratives out of 3,161 escape it, by another path). The field is
kept here for fidelity to the structure of the cohort file, and parity is held
**because** both sides lose it. This is not to be fixed within the audit:
giving the sentence back to the model would change the prompt of all the arms already played.

## The departures from the field, declared

1. **Origin and destination are fine-zone centroids.** The microdata carry no
   coordinates, and the `lil-1750` agreement forbids redistributing any.
2. **Intra-zone trips are set aside.** Between two coinciding centroids there is no
   itinerary to compute. They are short, hence mostly on foot: the script quantifies the lost
   walking share so that the bias is readable, not just mentioned.
3. **Eleven sequences are broken by the exclusion of intra-zone trips.** The set builder
   routes from one activity to the next: the origin of a trip is therefore the destination of the
   previous one. When an intra-zone trip has been removed in the middle of a day, the routed origin
   is no longer the one the respondent declared. Measured: 11 sequences out of 6,702
   (0.2 %), 11 respondents. These trips carry `chaine_rompue = 1` in the truth file
   and leave the audit — 11 out of 9,632 move no metric, and keeping them would
   compare a choice to a trip that is not that one.
4. **The day starts where the respondent was.** The household vehicles are at home at the
   start of the day, as in the chain rule. A respondent whose first trip does not
   leave from home therefore does not have their car at hand: the count is logged.

The produced file carries respondent attributes under the `lil-1750` agreement. `data/.gitignore`
already ignores `/population/*`: it is not versioned, and is not mounted outside this workstation.

Usage:
    services/llm-agents/.venv/bin/python scripts/progedo_logit/build_enquete_population.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import locale
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

ICI = Path(__file__).resolve().parent
RACINE = ICI.parents[1]
sys.path.insert(0, str(RACINE / "packages" / "mobility_core" / "src"))
sys.path.insert(0, str(ICI))

from faker import Faker  # noqa: E402
from mobility_core.bike_ownership import NO_BIKE, PLAIN_BIKE  # noqa: E402
from mobility_core.residence_zone import CouronneTable  # noqa: E402
from zone_label import build_zone_label  # noqa: E402

try:
    locale.setlocale(locale.LC_NUMERIC, "fr_FR.UTF-8")
except locale.Error:  # pragma: no cover — poste sans locale française
    pass

PROGEDO = RACINE / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV" / "fichiers_standards"
COUCHE_ZONES = (
    RACINE / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "zf_zones.gpkg"
)

CLES_PERSONNE = ("ZF", "ECH", "PER")

# Stable identifiers: two runs must produce the same `person_id` and the same
# `activity_id`, otherwise the decision cache and the join with the ground truth break
# from one run to another.
NAMESPACE_058 = uuid.uuid5(uuid.NAMESPACE_URL, "llm-agents-gama/ticket-058/audit-unitaire")

# Survey `main_occupation` → English persona modality. It is the inverse of
# `population_reference.OCCUPATION_ENQUETE`, whose values are frozen with the fitted
# artefacts: we translate the respondent's answer, we do not invent one.
OCCUPATION_ANGLAIS = {
    "Scolaire (jusqu'au Bac)": "Pupil (up to Baccalaureate)",
    "Étudiant": "Student",
    "Travail à plein temps": "Full-time worker",
    "Travail à temps partiel": "Part-time worker",
    "Chômeur/recherche d'emploi": "Unemployed / job seeker",
    "Personne au foyer": "Homemaker",
    "Retraité": "Retired",
    # The cohort does not expose this modality, but it is the answer given by the respondent:
    # translating it is a translation, dropping it would be a loss.
    "Autre": "Other",
}

# Purposes of the day → `travel_purposes` labels, as the narrative renders them.
MOTIF_LIBELLE = {
    "work": "Work",
    "education": "Education",
    "shop": "Shopping",
    "leisure": "Leisure",
    "other": "Other",
    "home": None,  # home is not an outing purpose
}

VRAI = {"true", "True", "1", "vrai"}

# Distance bands of the aggregated composite score (`prompt_calibration/calibration/metadata.py`,
# `_DIST_BUCKETS`). Mode depends first on distance: the exclusion of intra-zone trips is therefore
# judged not on an overall walking share, but band by band — it is the axis that the
# composite score already weights (weight 0.3).
BANDES_DISTANCE = ((1, "0-1km"), (2, "1-2km"), (5, "2-5km"), (10, "5-10km"), (20, "10-20km"),
                   (50, "20-50km"))
BANDE_LOINTAINE = "plus_50km"


def _bande(km: float) -> str:
    for seuil, nom in BANDES_DISTANCE:
        if km < seuil:
            return nom
    return BANDE_LOINTAINE

# Same locale as `utils.fake`, which the loader uses for populations without a first name.
_FAKER = Faker("fr_FR")


def _nom(person_id: str, gender: str) -> str:
    """A stable first name for a respondent: pure function of (person_id, gender).

    Two runs must serve the same prompt, hence the same first name. The loader, for its part,
    would draw from the clock — it is the defect the cohort fixed in ticket 074.
    """
    graine = int(hashlib.md5(person_id.encode()).hexdigest()[:8], 16)
    _FAKER.seed_instance(graine)
    if gender == "Male":
        return _FAKER.name_male()
    if gender == "Female":
        return _FAKER.name_female()
    return _FAKER.name()

# Beyond this rate of unusable days, the sample is no longer the one we think.
SEUIL_PERTE_ALARME = 0.05


def _bool(valeur: str | None) -> bool:
    return str(valeur or "").strip() in VRAI


def _entier(valeur: str | None, defaut: int = 0) -> int:
    try:
        return int(float(str(valeur).strip()))
    except (TypeError, ValueError):
        return defaut


def _secondes(hhmm: str | None) -> float | None:
    """`0815` → 29,700 s. Hours from 24 to 28 are those of the next day, and stay as such."""
    texte = str(hhmm or "").strip()
    if not texte.isdigit() or not 3 <= len(texte) <= 4:
        return None
    heures, minutes = divmod(int(texte), 100)
    if minutes > 59:
        return None
    return float(heures * 3600 + minutes * 60)


def centroides_wgs84() -> dict[str, tuple[float, float]]:
    """Fine zone → (lon, lat). The centroids are those seen at training time, reprojected."""
    import geopandas as gpd

    couche = gpd.read_file(COUCHE_ZONES)
    colonne = "ZF" if "ZF" in couche.columns else couche.columns[0]
    # The layer carries the centroids in Lambert 93; the simulator works in WGS84.
    points = gpd.GeoSeries(
        gpd.points_from_xy(couche["XL93"], couche["YL93"], crs="EPSG:2154")
    ).to_crs("EPSG:4326")
    return {
        str(zf).strip(): (float(p.x), float(p.y))
        for zf, p in zip(couche[colonne], points, strict=True)
    }


def jours_declares() -> dict[tuple[str, ...], str]:
    """(ZF, ECH, PER) → date of the described day, `YYYY-MM-DD`.

    The survey covers the day before the interview, and it is that day which the persons
    file dates: the weekday computed from `AN/MOIS/DATE` equals `JOUR` (« day of the
    trips ») for 20,462 persons out of 20,463, and never falls on a Saturday.
    """
    fichier = PROGEDO / "Toulouse_2023_std_pers.csv"
    jours: dict[tuple[str, ...], str] = {}
    with fichier.open(encoding="utf-8", errors="replace") as flux:
        for ligne in csv.DictReader(flux):
            an, mois, jour = (
                (ligne.get("AN") or "").strip(),
                (ligne.get("MOIS") or "").strip(),
                (ligne.get("DATE") or "").strip(),
            )
            if not (an and mois and jour):
                continue
            cle = (
                (ligne.get("ZFP") or "").strip(),
                (ligne.get("ECH") or "").strip(),
                (ligne.get("PER") or "").strip(),
            )
            jours[cle] = f"{an}-{int(mois):02d}-{int(jour):02d}"
    logger.info(f"Described-day dates loaded: {len(jours):n} persons")
    return jours


def _traits(
    ligne: dict[str, str], couronne: str | None, motifs: list[str], nom: str
) -> dict[str, Any]:
    """The traits served to the agent: the 12 persona variables, the ring, the purposes."""
    occupation_fr = (ligne.get("main_occupation") or "").strip()
    return {
        "name": nom,
        "age": _entier(ligne.get("age")),
        "gender": (ligne.get("gender") or "").strip(),
        "household_size": _entier(ligne.get("household_size")),
        "has_driving_license": _bool(ligne.get("has_driving_license")),
        "has_pt_subscription": _bool(ligne.get("has_pt_subscription")),
        "number_of_cars": _entier(ligne.get("number_of_cars")),
        "car_availability": (ligne.get("car_availability") or "").strip(),
        "socioprofessional_class": (ligne.get("socioprofessional_class") or "").strip(),
        "main_occupation": OCCUPATION_ANGLAIS.get(occupation_fr, occupation_fr),
        "employed": _bool(ligne.get("employed")),
        "studies": _bool(ligne.get("studies")),
        # Outside the narrative since 2026-08-26, but read by the chain rule.
        "personal_bike": PLAIN_BIKE if _bool(ligne.get("has_bike")) else NO_BIKE,
        "residence_zone": couronne,
        "travel_purposes": motifs,
    }


def construire(
    split: str,
    limite: int | None = None,
    suffixe: str = "",
    echantillon: int | None = None,
    graine: int = 42,
) -> tuple[Path, Path]:
    debut = time.monotonic()
    logger.info(f"=== Respondent population — split « {split} » ===")

    variables = {}
    with (ICI / f"mode_choice_{split}.csv").open(encoding="utf-8") as flux:
        for ligne in csv.DictReader(flux):
            variables[tuple(ligne[c] for c in (*CLES_PERSONNE, "NDEP"))] = ligne
    logger.info(f"Mode-choice set: {len(variables):n} trips")

    deplacements: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    intra = 0
    marche_totale = Counter()
    par_bande: dict[str, Counter] = defaultdict(Counter)
    with (ICI / f"mode_choice_{split}_od.csv").open(encoding="utf-8") as flux:
        for ligne in csv.DictReader(flux):
            cle = tuple(ligne[c] for c in (*CLES_PERSONNE, "NDEP"))
            mode = (variables.get(cle) or {}).get("mode", "")
            bande = _bande(float(ligne.get("vol_oiseau_declare_km") or 0.0))
            marche_totale["tous"] += 1
            marche_totale["tous_walk"] += mode == "walk"
            par_bande[bande]["tous"] += 1
            par_bande[bande]["tous_walk"] += mode == "walk"
            if ligne["ZF_orig"] == ligne["ZF_dest"]:
                intra += 1
                continue
            marche_totale["retenus"] += 1
            marche_totale["retenus_walk"] += mode == "walk"
            par_bande[bande]["retenus"] += 1
            par_bande[bande]["retenus_walk"] += mode == "walk"
            deplacements[tuple(ligne[c] for c in CLES_PERSONNE)].append(ligne)

    part_avant = marche_totale["tous_walk"] / max(marche_totale["tous"], 1)
    part_apres = marche_totale["retenus_walk"] / max(marche_totale["retenus"], 1)
    logger.info(
        f"Intra-zone set aside: {intra:n} · kept: {marche_totale['retenus']:n} "
        f"for {len(deplacements):n} respondents"
    )
    logger.info(
        f"Measured sample bias — declared walking share: "
        f"{part_avant:.1%} over all trips, {part_apres:.1%} over those kept "
        f"({(part_apres - part_avant) * 100:+.1f} pt)"
    )

    logger.info(
        "Retention and walking share per declared distance band "
        "(the axis of the aggregated composite score):"
    )
    for bande in [nom for _, nom in BANDES_DISTANCE] + [BANDE_LOINTAINE]:
        compteur = par_bande.get(bande)
        if not compteur or not compteur["tous"]:
            continue
        garde = compteur["retenus"] / compteur["tous"]
        marche_t = compteur["tous_walk"] / compteur["tous"]
        marche_r = compteur["retenus_walk"] / max(compteur["retenus"], 1)
        logger.info(
            f"  {bande:9s} {compteur['retenus']:5d}/{compteur['tous']:5d} kept ({garde:4.0%}) "
            f"· walking {marche_t:5.1%} → {marche_r:5.1%}"
        )

    centroides = centroides_wgs84()
    couronnes = CouronneTable.load()
    dates = jours_declares()

    personnes: list[dict[str, Any]] = []
    verite: list[dict[str, Any]] = []
    compte = Counter()

    retenues = sorted(deplacements.items())
    if limite:
        # Dry run: the first respondents in key order, hence always the same ones.
        retenues = retenues[:limite]
        logger.info(f"Trial limit: {limite:n} respondent(s) out of {len(deplacements):n}")
    elif echantillon:
        # Calibration sample: respondents drawn at random, NOT the first ones. Key order
        # is the fine-zone order, i.e. geographic: its first N would all live in the same
        # few communes. Re-sorted after the draw so that the output order stays stable.
        if echantillon > len(retenues):
            logger.error(
                f"[ALARME] Sample of {echantillon:n} respondents asked for, only "
                f"{len(retenues):n} available in split « {split} »"
            )
            raise SystemExit(2)
        retenues = sorted(random.Random(graine).sample(retenues, echantillon))
        logger.info(
            f"Random sample: {echantillon:n} respondent(s) out of {len(deplacements):n} "
            f"(seed {graine}), {sum(len(t) for _, t in retenues):n} inter-zone trips"
        )

    for cle, trajets in retenues:
        trajets.sort(key=lambda t: _entier(t["NDEP"]))
        zf_residence = cle[0]

        horaires = [(_secondes(t["depart_hhmm"]), _secondes(t["arrivee_hhmm"])) for t in trajets]
        if any(depart is None or arrivee is None for depart, arrivee in horaires):
            compte["journee_horaires_illisibles"] += 1
            continue
        if any(t["ZF_orig"] not in centroides or t["ZF_dest"] not in centroides for t in trajets):
            compte["journee_zone_hors_couche"] += 1
            continue

        person_id = str(uuid.uuid5(NAMESPACE_058, "|".join(cle)))
        premier = variables[(*cle, trajets[0]["NDEP"])]

        # Purposes of the day: those of the destinations, excluding home, in order of appearance.
        motifs: list[str] = []
        for trajet in trajets:
            libelle = MOTIF_LIBELLE.get(
                (variables[(*cle, trajet["NDEP"])].get("purpose") or "").strip()
            )
            if libelle and libelle not in motifs:
                motifs.append(libelle)

        couronne = couronnes.couronne_of_zf(zf_residence)
        if couronne is None:
            compte["couronne_inconnue"] += 1

        activites: list[dict[str, Any]] = []

        def _lieu(zf: str) -> dict[str, Any]:
            lon, lat = centroides[zf]
            commune = couronnes.commune_of_zf(zf)
            return {
                "lon": lon,
                "lat": lat,
                # Left as None ON PURPOSE: the engine itself computes whether a stop is
                # reachable (`otp._has_reachable_stop`) when the flag is absent. Prefilling
                # it from another source would substitute a fabricated value.
                "public_transport": None,
                "zone": build_zone_label(commune[0]) if commune else None,
            }

        # Origin activity of the first trip: its purpose is that of the declared origin.
        origine_motif = (premier.get("purpose_origin") or "other").strip() or "other"
        activites.append({
            "id": str(uuid.uuid5(NAMESPACE_058, f"{person_id}|0")),
            "scheduled_start_time": 0.0,
            "start_time": 0.0,
            "end_time": horaires[0][0],
            "purpose": origine_motif,
            "location": _lieu(trajets[0]["ZF_orig"]),
        })

        for index, (trajet, (depart, arrivee)) in enumerate(zip(trajets, horaires, strict=True), 1):
            motif = (variables[(*cle, trajet["NDEP"])].get("purpose") or "other").strip()
            # The routed origin is the destination of the previous kept trip. If an
            # intra-zone trip was removed between the two, it is no longer the declared origin.
            rompue = index > 1 and trajet["ZF_orig"] != trajets[index - 2]["ZF_dest"]
            if rompue:
                compte["enchainement_rompu"] += 1
            depart_suivant = horaires[index][0] if index < len(horaires) else 86400.0
            activites.append({
                "id": str(uuid.uuid5(NAMESPACE_058, f"{person_id}|{index}")),
                "scheduled_start_time": arrivee,
                "start_time": arrivee,
                "end_time": max(depart_suivant, arrivee),
                "purpose": motif or "other",
                "location": _lieu(trajet["ZF_dest"]),
            })
            verite.append({
                "person_id": person_id,
                "activity_id": activites[-1]["id"],
                "ordinal": index - 1,
                "ZF": cle[0],
                "ECH": cle[1],
                "PER": cle[2],
                "NDEP": trajet["NDEP"],
                "mode_declare": variables[(*cle, trajet["NDEP"])]["mode"],
                "chaine_rompue": int(rompue),
                "sample_weight": variables[(*cle, trajet["NDEP"])]["sample_weight"],
                "purpose": motif,
                # Declared crow-fly distance (`D11`). Serves to place the trip in
                # a distance band — the axis on which accuracy is published — and comes
                # from the field, hence independently of what a decision-maker chose.
                "vol_oiseau_declare_km": trajet["vol_oiseau_declare_km"],
                "depart_hhmm": trajet["depart_hhmm"],
                "jour_declare": dates.get(cle, ""),
            })
            if depart >= 86400:
                compte["deplacement_apres_minuit"] += 1

        if origine_motif != "home":
            compte["journee_ne_commence_pas_au_domicile"] += 1
        if cle not in dates:
            compte["date_inconnue"] += 1

        lon_dom, lat_dom = centroides.get(zf_residence, centroides[trajets[0]["ZF_orig"]])
        nom = _nom(person_id, (premier.get("gender") or "").strip())
        personnes.append({
            "person_id": person_id,
            "immobile": False,
            "enquete": {
                "zf": cle[0],
                "ech": cle[1],
                "per": cle[2],
                "jour_declare": dates.get(cle, ""),
                "zf_residence": zf_residence,
            },
            "identity": {
                "name": nom,
                "traits_json": _traits(premier, couronne, motifs, nom),
                "home": {"lon": lon_dom, "lat": lat_dom, "public_transport": None},
                "activities": activites,
            },
            "state": {},
            "is_llm_based": True,
        })
        compte["journees_retenues"] += 1

    perdues = sum(compte[c] for c in ("journee_horaires_illisibles", "journee_zone_hors_couche"))
    taux_perte = perdues / max(len(retenues), 1)
    logger.info(
        f"Days kept: {compte['journees_retenues']:n} · set aside: {perdues:n} "
        f"({taux_perte:.2%}) · trips with ground truth: {len(verite):n}"
    )
    for motif in ("journee_horaires_illisibles", "journee_zone_hors_couche", "date_inconnue"):
        if compte[motif]:
            logger.info(f"  {motif}: {compte[motif]:n}")
    if compte["journee_ne_commence_pas_au_domicile"]:
        logger.info(
            f"Days that do not start at home: "
            f"{compte['journee_ne_commence_pas_au_domicile']:n} — the household car is "
            "parked at home there, hence out of reach of the first trip (chain rule)"
        )
    if compte["enchainement_rompu"]:
        logger.warning(
            f"{compte['enchainement_rompu']:n} sequence(s) broken by the removal of an "
            "intra-zone trip: the routed origin is not the declared origin. Marked "
            "`chaine_rompue = 1` in the truth file, to be excluded from the audit"
        )
    if compte["deplacement_apres_minuit"]:
        logger.info(
            f"Trips declared at 24 h or later: {compte['deplacement_apres_minuit']:n} "
            "— times kept as is, beyond the 86,400 s day"
        )
    if compte["couronne_inconnue"]:
        logger.warning(
            f"{compte['couronne_inconnue']:n} respondent(s) without a ring: `residence_zone` "
            "will stay null for them, the narrative will omit the line"
        )
    if taux_perte > SEUIL_PERTE_ALARME:
        logger.error(
            f"[ALARME] {taux_perte:.1%} of days set aside (> {SEUIL_PERTE_ALARME:.0%}): "
            "the audited sample no longer represents the test partition"
        )

    dossier = RACINE / "data" / "population" / f"population_enquete_058_{split}{suffixe}"
    dossier.mkdir(parents=True, exist_ok=True)
    fichier_population = dossier / "population.json"
    fichier_population.write_text(
        json.dumps(personnes, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    dates_meteo = {
        p["person_id"]: p["enquete"]["jour_declare"]
        for p in personnes
        if p["enquete"]["jour_declare"]
    }
    fichier_dates = dossier / "dates_meteo.json"
    fichier_dates.write_text(
        json.dumps(dates_meteo, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info(
        f"Declared weather dates written: {len(dates_meteo):n} person(s) out of "
        f"{len(personnes):n} — {fichier_dates.name}"
    )

    fichier_verite = ICI / f"verite_058_{split}{suffixe}.csv"
    with fichier_verite.open("w", encoding="utf-8", newline="") as flux:
        graveur = csv.DictWriter(flux, fieldnames=list(verite[0].keys()))
        graveur.writeheader()
        graveur.writerows(verite)

    logger.success(
        f"Écrits : {fichier_population.relative_to(RACINE)} "
        f"({len(personnes):n} enquêtés) et {fichier_verite.name} "
        f"({len(verite):n} modes déclarés), en {time.monotonic() - debut:.1f} s"
    )
    return fichier_population, fichier_verite


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--split", default="test", choices=["test", "train"])
    parseur.add_argument(
        "--limite", type=int, help="write only the first N respondents (dry run)"
    )
    parseur.add_argument(
        "--suffixe", default="", help="suffix of the folder and of the truth file (e.g. _essai)"
    )
    parseur.add_argument(
        "--echantillon", type=int,
        help="draw N respondents at random (seeded) — a calibration sample, unlike --limite",
    )
    parseur.add_argument("--graine", type=int, default=42, help="seed of --echantillon")
    args = parseur.parse_args()
    if args.limite and args.echantillon:
        parseur.error("--limite and --echantillon are exclusive")
    construire(args.split, args.limite, args.suffixe, args.echantillon, args.graine)


if __name__ == "__main__":
    main()
