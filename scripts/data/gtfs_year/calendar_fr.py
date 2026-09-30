"""
School calendar of the Toulouse zone and metropolitan French public holidays.

Used to classify each day of the year by SIGNATURE — the pair
(day type, period class) that determines the transit system's service level.

Why it is needed: Tisséo service follows the school calendar very
closely, and the official bounds of zone C coincide exactly with the
breaks measured in the exports (drop from 12 600 to 10 535 trips on Monday
20/04/2026, back to 12 538 on Monday 04/05, bridge Friday 15/05 reduced to
10 877). Classifying a day without this calendar would amount to copying a
holiday day onto a school day.

Two public sources, both cached on disk so that the build
stays replayable offline:

  - data.education.gouv.fr  → school holidays by zone and by locality
  - calendrier.api.gouv.fr  → metropolitan public holidays

A validated snapshot is versioned next to this module
(`calendrier_snapshot.json`) and serves as fallback if the internet is not reachable.

This module does not decide the service: it only labels dates.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MODULE_DIR = Path(__file__).resolve().parent
CACHE_DIR = MODULE_DIR / "cache"
SNAPSHOT = MODULE_DIR / "calendrier_snapshot.json"

API_VACANCES = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/records"
)
API_FERIES = "https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json"

TIMEOUT_S = 30

# Period classes. Summer is split because July and August have distinct
# service (≈ 9 910 trips on a Monday in July, ≈ 10 418 at the end of August).
SCOLAIRE = "scolaire"
CLASSES_VACANCES = (
    "vac_hiver",
    "vac_printemps",
    "pont_ascension",
    "ete_juillet",
    "ete_aout",
    "vac_toussaint",
    "vac_noel",
)

# Label returned by the API → internal class.
_LIBELLE_VERS_CLASSE = {
    "vacances d'hiver": "vac_hiver",
    "vacances de printemps": "vac_printemps",
    "pont de l'ascension": "pont_ascension",
    "vacances d'été": "ete",  # later split into ete_juillet / ete_aout
    "début des vacances d'été": "ete",
    "vacances de la toussaint": "vac_toussaint",
    "vacances de noël": "vac_noel",
}

# When only the START date of the summer holidays is published — the calendar
# of the following school year not being published yet —, the end is set to
# 31 August. It is an assumption, logged as such: it only moves
# the boundary between "summer holidays" and "school" over the very last
# days of August, when service picks up anyway before the official start of term.
FIN_ETE_PAR_DEFAUT = (8, 31)

JOURS = ("lun", "mar", "mer", "jeu", "ven", "sam", "dim")


@dataclass(frozen=True)
class Periode:
    """A school-holiday range, in inclusive local dates."""

    classe: str
    debut: dt.date
    fin: dt.date

    def contient(self, jour: dt.date) -> bool:
        return self.debut <= jour <= self.fin


@dataclass(frozen=True)
class Signature:
    """What determines a day's service level."""

    type_jour: str  # lun..dim, or "ferie"
    periode: str  # scolaire, vac_*, ete_*

    def __str__(self) -> str:  # pragma: no cover - confort de lecture
        return f"{self.type_jour}/{self.periode}"


# ──────────────────────────────────────────────────────────────────────────────
# Fetching and cache
# ──────────────────────────────────────────────────────────────────────────────


def _lire_cache(nom: str) -> dict | None:
    chemin = CACHE_DIR / nom
    if chemin.exists():
        return json.loads(chemin.read_text(encoding="utf-8"))
    return None


def _ecrire_cache(nom: str, charge: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / nom).write_text(
        json.dumps(charge, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _lire_snapshot(cle: str) -> dict | None:
    if not SNAPSHOT.exists():
        return None
    return json.loads(SNAPSHOT.read_text(encoding="utf-8")).get(cle)


def _figer_instantane(cle: str, charge: dict) -> None:
    """Stores a successful fetch in the versioned snapshot.

    The disk cache is disposable; the snapshot, for its part, is committed with the
    repository and lets a build be replayed identically while offline.
    """
    contenu = {}
    if SNAPSHOT.exists():
        contenu = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if contenu.get(cle) == charge:
        return
    contenu[cle] = charge
    SNAPSHOT.write_text(
        json.dumps(contenu, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _contexte_ssl():
    """TLS context with a usable certificate store.

    Python interpreters installed by hand on macOS have no system
    store: without `certifi`, every HTTPS request fails with
    CERTIFICATE_VERIFY_FAILED whereas `curl` succeeds.
    """
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _http_json(url: str) -> dict:
    requete = urllib.request.Request(url, headers={"User-Agent": "llm-agents-gama/gtfs_year"})
    with urllib.request.urlopen(requete, timeout=TIMEOUT_S, context=_contexte_ssl()) as reponse:
        return json.loads(reponse.read().decode("utf-8"))


def _recuperer(nom_cache: str, url: str, rafraichir: bool, journal) -> dict:
    """Disk cache → online fetch → versioned snapshot. Never a silent failure."""
    if not rafraichir:
        depuis_cache = _lire_cache(nom_cache)
        if depuis_cache is not None:
            journal(f"    calendrier : {nom_cache} lu depuis le cache")
            return depuis_cache
    try:
        charge = _http_json(url)
        _ecrire_cache(nom_cache, charge)
        _figer_instantane(nom_cache, charge)
        journal(f"    calendrier : {nom_cache} récupéré en ligne et mis en cache")
        return charge
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        instantane = _lire_snapshot(nom_cache)
        if instantane is None:
            raise RuntimeError(
                f"[ALARME] calendrier indisponible : {url} injoignable ({err}) "
                f"et aucun instantané pour {nom_cache}"
            ) from err
        journal(f"    calendrier : {nom_cache} — réseau indisponible ({err}), repli sur l'instantané versionné")
        return instantane


# ──────────────────────────────────────────────────────────────────────────────
# Public holidays
# ──────────────────────────────────────────────────────────────────────────────


def feries(annee: int, rafraichir: bool = False, journal=print) -> dict[str, str]:
    """Metropolitan public holidays of the year, as `YYYYMMDD` → label."""
    charge = _recuperer(f"feries_{annee}.json", API_FERIES.format(year=annee), rafraichir, journal)
    return {k.replace("-", ""): v for k, v in charge.items()}


# ──────────────────────────────────────────────────────────────────────────────
# School holidays
# ──────────────────────────────────────────────────────────────────────────────


def _date_locale(horodatage: str) -> dt.date:
    """Converts a UTC timestamp from the API into a Europe/Paris local date.

    The API publishes `2026-04-17T22:00:00+00:00` for holidays that start on
    18/04 at 00:00 Paris time. Ignoring this offset would shift all
    bounds by one day.

    The +2 h also holds in winter, for another reason: the API then publishes
    `23:00Z` (winter time = UTC+1), and +2 h gives 01:00 the next day — same
    local date. A season-dependent offset would therefore be useless here, and
    only the DATE of the result is read.
    """
    moment = dt.datetime.fromisoformat(horodatage)
    return (moment + dt.timedelta(hours=2)).date()


def vacances(
    annee: int,
    localite: str = "Toulouse",
    rafraichir: bool = False,
    journal=print,
) -> list[Periode]:
    """School-holiday periods touching the year, class by class.

    A wide window (year-1 to year+1) is queried to catch the
    Christmas holidays straddling two calendar years.
    """
    where = (
        f'location="{localite}" '
        f'and end_date>="{annee - 1}-06-01" '
        f'and start_date<="{annee + 1}-06-30"'
    )
    url = API_VACANCES + "?" + urllib.parse.urlencode(
        {
            "where": where,
            "limit": 100,
            "select": "description,start_date,end_date,zones,population",
            "order_by": "start_date",
        }
    )
    charge = _recuperer(f"vacances_{localite}_{annee}.json", url, rafraichir, journal)

    brut: list[Periode] = []
    vus: set[tuple[str, dt.date, dt.date]] = set()
    for enreg in charge.get("results", []):
        libelle = (enreg.get("description") or "").strip().lower()
        classe = _LIBELLE_VERS_CLASSE.get(libelle)
        if classe is None:
            journal(f"    calendrier : période ignorée, libellé inconnu — {libelle!r}")
            continue
        # Summer is published twice (pupils / teachers): the widest range is kept,
        # the one that actually bounds the transport service.
        debut = _date_locale(enreg["start_date"])
        fin = _date_locale(enreg["end_date"]) - dt.timedelta(days=1)
        if fin < debut:
            if classe == "ete":
                # One-off entry "Début des vacances d'été": the end is not
                # published as long as the calendar of the following year
                # is not.
                fin = dt.date(debut.year, *FIN_ETE_PAR_DEFAUT)
                journal(
                    f"    calendrier : {libelle} du {debut} sans date de fin publiée, "
                    f"bornée au {fin} par hypothèse"
                )
            else:
                # Single-day period (an isolated bridge day).
                fin = debut
        cle = (classe, debut, fin)
        if cle in vus:
            continue
        vus.add(cle)
        brut.append(Periode(classe, debut, fin))

    # Merge overlapping duplicates of the same class (summer pupils /
    # teachers), then split the summer.
    fusionnees: dict[str, Periode] = {}
    for periode in brut:
        cle = f"{periode.classe}:{periode.debut.year}:{periode.debut.month}"
        deja = fusionnees.get(cle)
        if deja is None:
            fusionnees[cle] = periode
        else:
            fusionnees[cle] = Periode(
                periode.classe, min(deja.debut, periode.debut), max(deja.fin, periode.fin)
            )

    resultat: list[Periode] = []
    for periode in fusionnees.values():
        if periode.classe != "ete":
            resultat.append(periode)
            continue
        bascule = dt.date(periode.debut.year, 8, 1)
        if periode.debut < bascule <= periode.fin:
            resultat.append(Periode("ete_juillet", periode.debut, bascule - dt.timedelta(days=1)))
            resultat.append(Periode("ete_aout", bascule, periode.fin))
        else:
            resultat.append(
                Periode("ete_aout" if periode.debut >= bascule else "ete_juillet", periode.debut, periode.fin)
            )

    resultat.sort(key=lambda p: p.debut)
    return resultat


# ──────────────────────────────────────────────────────────────────────────────
# Signatures
# ──────────────────────────────────────────────────────────────────────────────


def dates_annee(annee: int) -> list[str]:
    """All dates of the year as `YYYYMMDD`."""
    jour = dt.date(annee, 1, 1)
    fin = dt.date(annee, 12, 31)
    sortie = []
    while jour <= fin:
        sortie.append(jour.strftime("%Y%m%d"))
        jour += dt.timedelta(days=1)
    return sortie


def to_date(datestr: str) -> dt.date:
    return dt.date(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8]))


def signature(
    datestr: str,
    periodes: Iterable[Periode],
    jours_feries: dict[str, str],
    decalages: dict[str, int] | None = None,
) -> Signature:
    """Signature of a date: (day type, period class).

    `decalages` postpones the start of a period by N days; it is LEARNED on
    the real dates by `ajuster_bornes`, never postulated.
    """
    jour = to_date(datestr)
    decalages = decalages or {}

    classe = SCOLAIRE
    for periode in periodes:
        debut = periode.debut + dt.timedelta(days=decalages.get(periode.classe, 0))
        if debut <= jour <= periode.fin:
            classe = periode.classe
            break

    type_jour = "ferie" if datestr in jours_feries else JOURS[jour.weekday()]
    return Signature(type_jour, classe)


def ajuster_bornes(
    periodes: list[Periode],
    jours_feries: dict[str, str],
    offre_reelle: dict[str, int],
    decalages_testes: list[int],
    journal=print,
) -> dict[str, int]:
    """Learns the start offset of each period on the real dates.

    The official holiday bound does not always coincide with the service
    switch: Saturday 18/04/2026, first official day of the spring
    holidays, still runs as a school Saturday (8 194 trips, like the school
    Saturdays of 04/04 and 11/04, against 8 470 on the next holiday Saturday).
    But summer switches from its first Saturday. Rather than deciding a priori,
    for each period the offset that minimises the relative dispersion
    of the number of trips within each signature is kept.

    Returns {classe_de_période: décalage_en_jours}.
    """
    if not offre_reelle:
        return {}

    def dispersion(decalages: dict[str, int]) -> float:
        groupes: dict[str, list[int]] = {}
        for datestr, trips in offre_reelle.items():
            sig = str(signature(datestr, periodes, jours_feries, decalages))
            groupes.setdefault(sig, []).append(trips)
        total = 0.0
        for valeurs in groupes.values():
            if len(valeurs) < 2:
                continue
            moyenne = sum(valeurs) / len(valeurs)
            if moyenne <= 0:
                continue
            etendue = max(valeurs) - min(valeurs)
            total += etendue / moyenne
        return total

    decalages: dict[str, int] = {}
    for periode in periodes:
        # Without a real date in the period, no offset is measurable.
        couvert = any(
            periode.debut <= to_date(d) <= periode.fin for d in offre_reelle
        )
        if not couvert:
            continue
        meilleur, score_min = 0, None
        for decalage in decalages_testes:
            essai = dict(decalages, **{periode.classe: decalage})
            score = dispersion(essai)
            if score_min is None or score < score_min - 1e-9:
                meilleur, score_min = decalage, score
        if meilleur:
            journal(
                f"    calendrier : début de {periode.classe} recalé de +{meilleur} j "
                f"({periode.debut} → {periode.debut + dt.timedelta(days=meilleur)}), appris sur les dates réelles"
            )
        decalages[periode.classe] = meilleur
    return decalages
