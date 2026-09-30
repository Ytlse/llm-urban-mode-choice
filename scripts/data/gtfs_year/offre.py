"""
Indexing of an export's service offer and detection of its reliable window.

Tisséo exports are "rolling": each covers about 35 days, but
is complete only over the first weeks. Beyond that, the operator publishes
only the backbone lines — the number of active lines collapses from
123 to 3 (metro A, B, TELEO), then to 1. Taking these days as they are
would run the simulation on a transit system reduced to the metro, without any error
signal.

The cut is first made on a ratio RELATIVE to the day type. A demanding absolute
threshold such as "at least 80 % of the maximum" would reject Saturday 11/04/2026
(88 lines) and Sunday 12/04 (48 lines), which are perfectly
normal days: a Sunday legitimately has three times fewer lines than a Tuesday.

A very low absolute floor completes the rule, because the day-type rule
is blind when a day type never appears complete in the export —
its reference is then the level of the tail itself.

A tail can also be truncated **per line** without the overall offer
collapsing: the liO export describes the whole transit system up to the service change
of 13/12/2026, then only extends the lines already filled in. Thirteen lines
`.liO 31` — ten within the study scope, all feeder lines to a station —
stop on 11 or 12/12/2026 and never come back over the following thirty-seven
weeks, while the number of active lines stays at 94 % of its
maximum. No global threshold can see it: hence `_falaise_lignes`.

Both GTFS calendar forms are read. Tisséo and the TER use only
`calendar_dates.txt`, one row per service date; liO publishes a weekly
`calendar.txt` that `calendar_dates.txt` then corrects in both
directions (addition `exception_type=1`, removal `exception_type=2`). The calendar
is unfolded into explicit dates at indexing time; the produced feed, for its part,
always stays in explicit dates with an empty `calendar.txt` (invariant V1).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from . import gtfs_io
from .calendar_fr import JOURS, to_date
from .gtfs_io import Export

# Columns of `calendar.txt`, in the order of `date.weekday()` (0 = Monday).
COLONNES_JOURS_GTFS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_UN_JOUR = dt.timedelta(days=1)


@dataclass
class IndexExport:
    """What an export says about each day it covers."""

    export: Export
    trips: dict[str, dict[str, str]] = field(default_factory=dict)
    trips_par_date: dict[str, list[str]] = field(default_factory=dict)
    lignes_par_date: dict[str, int] = field(default_factory=dict)
    agences: list[dict[str, str]] = field(default_factory=list)

    @property
    def dates(self) -> list[str]:
        return sorted(self.trips_par_date)

    def nb_trips(self, date: str) -> int:
        return len(self.trips_par_date.get(date, ()))


def indexer(export: Export, journal=print) -> IndexExport:
    """Builds the index of an export: which trips run on which day.

    Reads neither `stop_times.txt` nor `shapes.txt` — the two heavy files will
    only be scanned once the days actually kept are known.
    """
    index = IndexExport(export=export)
    index.agences = list(gtfs_io.lire(export, "agency.txt"))

    services_par_trip: dict[str, str] = {}
    trips_par_service: dict[str, list[str]] = {}
    lignes_par_service: dict[str, set[str]] = {}
    for ligne in gtfs_io.lire(export, "trips.txt"):
        trip_id = ligne["trip_id"]
        service_id = ligne["service_id"]
        index.trips[trip_id] = ligne
        services_par_trip[trip_id] = service_id
        trips_par_service.setdefault(service_id, []).append(trip_id)
        lignes_par_service.setdefault(service_id, set()).add(ligne["route_id"])

    # The two ways a GTFS declares when a service runs. Tisséo and the
    # TER use only `calendar_dates.txt` (one row per date); liO publishes
    # a weekly `calendar.txt` (457 services over 396 days) that
    # `calendar_dates.txt` then corrects in both directions — 3,408 additions and
    # 2,925 removals. Ignoring the removals would run coaches on the days when
    # the operator says they do not run: the calendar is unfolded here, and
    # the produced feed, for its part, stays in explicit dates (invariant V1).
    dates_par_service: dict[str, set[str]] = {}
    calendrier_hebdo = list(gtfs_io.lire(export, "calendar.txt"))
    jours_deplies = 0
    for ligne in calendrier_hebdo:
        service_id = ligne["service_id"]
        debut, fin = ligne.get("start_date", ""), ligne.get("end_date", "")
        if not debut or not fin:
            journal(f"[ALARME] {export.etiquette} : service {service_id} sans bornes de validité")
            continue
        jour_courant, dernier = to_date(debut), to_date(fin)
        if jour_courant > dernier:
            journal(
                f"[ALARME] {export.etiquette} : service {service_id} borné à l'envers "
                f"({debut} → {fin}) — ignoré"
            )
            continue
        actives = dates_par_service.setdefault(service_id, set())
        while jour_courant <= dernier:
            if ligne.get(COLONNES_JOURS_GTFS[jour_courant.weekday()]) == "1":
                actives.add(f"{jour_courant:%Y%m%d}")
                jours_deplies += 1
            jour_courant += _UN_JOUR

    ajouts = retraits = retraits_sans_effet = 0
    for ligne in gtfs_io.lire(export, "calendar_dates.txt"):
        service_id, date = ligne["service_id"], ligne["date"]
        exception = ligne.get("exception_type")
        if exception == "1":
            dates_par_service.setdefault(service_id, set()).add(date)
            ajouts += 1
        elif exception == "2":
            actives = dates_par_service.get(service_id)
            if actives and date in actives:
                actives.discard(date)
                retraits += 1
            else:
                retraits_sans_effet += 1
        else:
            journal(
                f"[ALARME] {export.etiquette} : exception_type {exception!r} inconnu "
                f"(service {service_id}, {date}) — ligne ignorée"
            )

    lignes_par_date: dict[str, set[str]] = {}
    for service_id, dates_actives in dates_par_service.items():
        trips_du_service = trips_par_service.get(service_id, ())
        routes_du_service = lignes_par_service.get(service_id, ())
        for date in dates_actives:
            index.trips_par_date.setdefault(date, []).extend(trips_du_service)
            lignes_par_date.setdefault(date, set()).update(routes_du_service)

    if calendrier_hebdo:
        journal(
            f"    {export.etiquette} : calendar.txt déplié — {len(calendrier_hebdo)} service(s) "
            f"hebdomadaire(s), {jours_deplies:,} (service, date) produits"
        )
    if ajouts or retraits or retraits_sans_effet:
        journal(
            f"    {export.etiquette} : calendar_dates — {ajouts:,} ajout(s), {retraits:,} retrait(s), "
            f"{retraits_sans_effet:,} retrait(s) sans effet"
        )

    for date in index.trips_par_date:
        index.trips_par_date[date].sort()
    index.lignes_par_date = {d: len(v) for d, v in lignes_par_date.items()}

    dates = index.dates
    if dates:
        export.date_min, export.date_max = dates[0], dates[-1]
    journal(
        f"    {export.etiquette} : {len(index.trips):,} trips, {len(dates)} dates "
        f"({export.date_min} → {export.date_max})"
    )
    return index


def _falaise_lignes(
    index: IndexExport, dates: list[str], config: dict, journal=print
) -> int | None:
    """First date from which the export describes a truncated transit system.

    The "historical" rule 1 looks for a **global** collapse of the number of
    active lines. It is blind to *per-line* truncation, that of an
    export which describes the whole transit system up to the next service change
    then only extends the lines already filled in. Measured on liO
    (2026-08-01 → 2027-08-31): **thirteen `.liO 31` lines stop on 11 or
    12/12/2026** — ten of which serve the study scope, all
    feeder lines to a station (Muret, Carbonne, Noé, Villefranche,
    Castelnau-d'Estrétefonds, Boussens) — and never resume over the
    thirty-seven remaining weeks. Their `calendar.txt` stops there (services
    from 06 to 12/12) while that of the neighbouring lines runs until 31/08/2027:
    it is the export's horizon, not an operating decision. The overall
    offer, for its part, barely moved (4,303 → 4,165 runs on Monday,
    260 active lines out of 276): no global threshold could see it.

    What separates a cliff from an end of season is not the shape of the
    loss, it is **what the export does next**. The fifty-two school
    lines that stop on 30/06/2027 do not resume either, but
    the export ends nine weeks later, in the middle of the summer holidays:
    their absence is explained. The thirteen of 11/12, on the other hand, are missing for
    thirty-seven weeks including six months of school term. Hence the condition
    `falaise_jours_apres_min`: a season lasts at most ten weeks, so if
    the export still covers thirteen weeks after the loss, the season no longer
    explains it. This condition also puts Tisséo's rolling exports
    (35 days) out of reach of the check by construction.

    Returns the index of the first date to discard, or `None`.
    """
    part_min = float(config.get("falaise_lignes_part_min", 0.04))
    plancher_lignes = int(config.get("falaise_lignes_min", 5))
    jours_apres_min = int(config.get("falaise_jours_apres_min", 91))
    fenetre_jours = int(config.get("falaise_fenetre_jours", 7))
    if part_min <= 0 or jours_apres_min <= 0:
        return None

    # Last service day of each line, within the export's dates.
    derniere_par_ligne: dict[str, str] = {}
    for date in dates:
        for trip_id in index.trips_par_date.get(date, ()):
            ligne = index.trips.get(trip_id, {}).get("route_id", "")
            if ligne:
                derniere_par_ligne[ligne] = date
    if not derniere_par_ligne:
        return None

    # Lines that stop on the export's last day prove nothing:
    # it is the export that stops, not them.
    fins: dict[str, int] = {}
    for ligne, fin in derniere_par_ligne.items():
        if fin != dates[-1]:
            fins[fin] = fins.get(fin, 0) + 1

    actives_max = max(index.lignes_par_date.values(), default=0)
    seuil = max(plancher_lignes, part_min * actives_max)

    for position, date in enumerate(dates):
        if len(dates) - 1 - position < jours_apres_min:
            break
        fenetre = dates[max(0, position - fenetre_jours + 1) : position + 1]
        perdues = {d: fins.get(d, 0) for d in fenetre if fins.get(d)}
        if sum(perdues.values()) < seuil:
            continue
        # The last day of a lost line is still complete; it is the
        # next day that is missing. The cut is therefore made just after the EARLIEST
        # loss in the window.
        derniere_complete = min(perdues)
        indice = dates.index(derniere_complete) + 1
        journal(
            f"    {index.export.etiquette} : falaise de lignes le {derniere_complete} — "
            f"{sum(perdues.values())} ligne(s) cessent définitivement de rouler "
            f"(seuil {seuil:.1f} sur {actives_max} lignes actives au maximum) alors que "
            f"l'export couvre encore {len(dates) - 1 - position} jour(s) : "
            f"queue tronquée par ligne, pas fin de saison"
        )
        journal(
            f"[ALARME] {index.export.etiquette} : l'export cesse de décrire "
            f"{sum(perdues.values())} ligne(s) au {derniere_complete} et couvre pourtant "
            f"{len(dates) - 1 - position} jour(s) de plus — {len(dates) - indice} date(s) "
            f"écartée(s) ; un export publié après ce changement de service les rendrait réelles"
        )
        return indice
    return None


def fenetre_fiable(index: IndexExport, config: dict, journal=print) -> list[str]:
    """Dates of the export to be considered complete.

    Two forms of truncation, and the cut keeps the earlier of the two.

    **Global collapse** — two thresholds, and the cut applies to the longest
    suffix that breaches them:

      - relative to the day type: below `ratio_lignes_min` times the maximum
        number of lines reached by that day type in the export;
      - absolute: below `ratio_plancher_lignes` times the export's maximum, all
        day types combined — the safety net when a day type never appears
        complete.

    **Line cliff** — entire lines stop being described while
    the export still covers months: see `_falaise_lignes`.

    Truncation never reopens: everything after its start is discarded.
    """
    ratio_min = float(config["ratio_lignes_min"])
    ratio_plancher = float(config["ratio_plancher_lignes"])
    jours_min = int(config["jours_min_par_export"])

    dates = index.dates
    if not dates:
        return []

    # Reference: the MAXIMUM number of active lines observed for this day type over
    # the whole export. Taking the median of the first weeks seemed more
    # robust, but an export delivered late — mostly made of its
    # own truncated tail — would contaminate its own reference: the median
    # would drop to the tail's level, nothing would be "below the threshold" any more, and
    # the export would pass intact while injecting days reduced to the metro.
    reference: dict[str, int] = {}
    for date in dates:
        type_jour = JOURS[to_date(date).weekday()]
        lignes = index.lignes_par_date.get(date, 0)
        if lignes > reference.get(type_jour, -1):
            reference[type_jour] = lignes

    # Second safeguard, an absolute one: a fraction of the maximum reached by
    # the export, all day types combined. The day-type rule is
    # blind when a type NEVER appears complete in the export — its
    # reference is then the level of the tail itself, and nothing is cut.
    # The floor is very low to let Sundays through, which legitimately run
    # 39 % of a Tuesday's lines on the Toulouse transit system.
    plancher = ratio_plancher * max(index.lignes_par_date.values(), default=0)

    def sous_seuil(date: str) -> bool:
        lignes = index.lignes_par_date.get(date, 0)
        if lignes < plancher:
            return True
        type_jour = JOURS[to_date(date).weekday()]
        seuil = reference.get(type_jour)
        if seuil is None:
            return False
        return lignes < ratio_min * seuil

    # Truncation is a LASTING collapse that runs until the end of
    # the export, not an isolated dip. We therefore look for the longest suffix whose
    # days are all below the threshold. Cutting at the first dip
    # encountered would reject public holidays, which legitimately run at the
    # level of a Sunday: Easter Monday 06/04 (48 lines against 123
    # for an ordinary Monday) would cut six valid days from the export.
    debut_queue = len(dates)
    for position in range(len(dates) - 1, -1, -1):
        if not sous_seuil(dates[position]):
            break
        debut_queue = position

    # Second form of truncation: the line cliff. The global collapse
    # above sees nothing when the export describes the whole transit system up to the
    # next service change, then only extends the lines already
    # filled in: the number of active lines stays high (liO keeps 94 % of
    # its own on 14/12/2026) but entire lines disappear for
    # good. See `_falaise_lignes`.
    debut_falaise = _falaise_lignes(index, dates, config, journal)
    if debut_falaise is not None:
        debut_queue = min(debut_queue, debut_falaise)

    retenues = dates[:debut_queue]
    if debut_queue < len(dates):
        premiere = dates[debut_queue]
        type_jour = JOURS[to_date(premiere).weekday()]
        journal(
            f"    {index.export.etiquette} : queue tronquée à partir du {premiere} "
            f"({index.lignes_par_date.get(premiere, 0)} lignes actives contre "
            f"{reference.get(type_jour)} attendues pour un {type_jour}) — "
            f"{len(dates) - debut_queue} date(s) écartée(s)"
        )

    creux_isoles = [d for d in retenues if sous_seuil(d)]
    if creux_isoles:
        journal(
            f"    {index.export.etiquette} : {len(creux_isoles)} creux isolé(s) conservé(s) "
            f"(offre réduite mais suivie d'un retour à la normale, typiquement un férié) : "
            f"{', '.join(creux_isoles[:6])}{'…' if len(creux_isoles) > 6 else ''}"
        )

    if len(retenues) < jours_min:
        journal(
            f"[ALARME] {index.export.etiquette} : seulement {len(retenues)} date(s) fiable(s), "
            f"minimum attendu {jours_min} — export inutilisable"
        )
        return []

    index.export.fin_fiable = retenues[-1]
    index.export.jours_fiables = retenues
    return retenues
