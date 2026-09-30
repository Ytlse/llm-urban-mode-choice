"""
Invariants of the produced feed.

"Byte-exact preservation" of real dates makes no sense here: the
`service_id` values are rewritten by construction, and some `trip_id` too. What
must be preserved is the SERVICE — the set of trips run on a given day.

Hence the service fingerprint:

    empreinte(feed, date) = sha256(sorted multiset of the content keys
                                   of the trips active that day)

where a trip's content key is the hash of its route, its direction, its
headsign, its geometry and its sequence of timed stops — never its identifier.
Two feeds serve the same day if and only if their fingerprints
coincide. It is a strict equality, not a bound, and it is recomputed by
rereading the written files: it catches a calendar error just as well
as a write error.

The same checks applied to the feed currently in service fail: it serves
13 250 trips on 08/04/2026 where its two sources give 12 652 and 12 660,
and its shape 14846 mixes two paths. That is why these checks
exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import gtfs_io
from .assemblage import cle_contenu, hash_geometrie
from .calendar_fr import JOURS, to_date
from .donneurs import EXTRAPOLE, REEL, SANS_SERVICE, Provenance
from .gtfs_io import Export
from .offre import IndexExport, indexer

BLOQUANT = "bloquant"
ALARME = "alarme"


@dataclass
class Violation:
    code: str
    gravite: str
    message: str

    def __str__(self) -> str:  # pragma: no cover
        marque = "[ALARME] " if self.gravite == ALARME else "[ÉCHEC] "
        return f"{marque}{self.code} — {self.message}"


def _id_source(trip_id: str, etiquettes: set[str]) -> str:
    """Recovers the original identifier of a forked trip (`<id>__<export>`)."""
    for etiquette in etiquettes:
        suffixe = f"__{etiquette}"
        if trip_id.endswith(suffixe):
            return trip_id[: -len(suffixe)]
    return trip_id


def trips_hors_geometrie(export: Export) -> set[str]:
    """Trips with a `shape_dist_traveled` exceeding the length of their path.

    Used twice: on the produced feed (V6) and, when they carry any, on its
    sources — to tell whether the overrun is a build defect (a chimera
    path, two geometries interleaved under the same identifier) or a defect
    already published by the operator and faithfully copied. liO publishes 29 out of
    7 715 trips; confusing the two would reject a correct feed.
    """
    longueur: dict[str, float] = {}
    for ligne in gtfs_io.lire(export, "shapes.txt"):
        try:
            distance = float(ligne.get("shape_dist_traveled") or 0.0)
        except ValueError:
            continue
        shape_id = ligne["shape_id"]
        if distance > longueur.get(shape_id, -1.0):
            longueur[shape_id] = distance
    shape_par_trip = {l["trip_id"]: l.get("shape_id", "") for l in gtfs_io.lire(export, "trips.txt")}
    hors: set[str] = set()
    for ligne in gtfs_io.lire(export, "stop_times.txt"):
        shape_id = shape_par_trip.get(ligne["trip_id"], "")
        if not shape_id or shape_id not in longueur:
            continue
        try:
            distance = float(ligne.get("shape_dist_traveled") or 0.0)
        except ValueError:
            continue
        if distance > longueur[shape_id] + 1.0:
            hors.add(ligne["trip_id"])
    return hors


def empreintes_par_date(
    export: Export,
    trips_par_date: dict[str, list[str]],
    dates: set[str],
    config: dict,
    journal=print,
) -> dict[str, str]:
    """Service fingerprint of each requested date, computed by rereading the feed."""
    dec_coord = int(config["canonicalisation"]["decimales_coordonnees"])
    dec_dist = int(config["canonicalisation"]["decimales_distance"])

    trips_voulus: set[str] = set()
    for date in dates:
        trips_voulus.update(trips_par_date.get(date, ()))
    if not trips_voulus:
        return {}

    meta_par_trip: dict[str, dict[str, str]] = {}
    for ligne in gtfs_io.lire(export, "trips.txt"):
        if ligne["trip_id"] in trips_voulus:
            meta_par_trip[ligne["trip_id"]] = ligne

    shapes_voulues = {m.get("shape_id", "") for m in meta_par_trip.values()} - {""}
    points_par_shape: dict[str, list[dict[str, str]]] = {}
    for ligne in gtfs_io.lire(export, "shapes.txt"):
        if ligne["shape_id"] in shapes_voulues:
            points_par_shape.setdefault(ligne["shape_id"], []).append(
                gtfs_io.canoniser_point_shape(ligne, dec_coord, dec_dist)
            )
    for points in points_par_shape.values():
        points.sort(key=lambda p: int(p["shape_pt_sequence"]))
    hash_par_shape = {sid: hash_geometrie(pts) for sid, pts in points_par_shape.items()}

    horaires_par_trip: dict[str, list[dict[str, str]]] = {}
    for ligne in gtfs_io.lire(export, "stop_times.txt"):
        if ligne["trip_id"] in trips_voulus:
            horaires_par_trip.setdefault(ligne["trip_id"], []).append(
                gtfs_io.canoniser_horaire(ligne, dec_dist)
            )
    for horaires in horaires_par_trip.values():
        horaires.sort(key=lambda h: int(h["stop_sequence"]))

    cle_par_trip = {
        trip_id: cle_contenu(
            meta, horaires_par_trip.get(trip_id, []), hash_par_shape.get(meta.get("shape_id", ""), "")
        )
        for trip_id, meta in meta_par_trip.items()
    }

    import hashlib

    empreintes: dict[str, str] = {}
    for date in sorted(dates):
        cles = sorted(
            cle_par_trip[t] for t in trips_par_date.get(date, ()) if t in cle_par_trip
        )
        digest = hashlib.sha256()
        for cle in cles:
            digest.update(cle.encode())
        empreintes[date] = digest.hexdigest()
    return empreintes


def controler(
    repertoire: Path,
    plan: dict[str, Provenance],
    index_par_export: dict[str, IndexExport],
    config: dict,
    journal=print,
) -> tuple[list[Violation], dict[str, str], IndexExport]:
    """Applies all invariants to the feed written in `repertoire`."""
    violations: list[Violation] = []
    feed = Export(chemin=repertoire, etiquette=repertoire.name, empreinte="")

    # V1 — conformity with the repository reader.
    calendrier_hebdo = list(gtfs_io.lire(feed, "calendar.txt"))
    if calendrier_hebdo:
        violations.append(
            Violation("V1", BLOQUANT, f"calendar.txt contient {len(calendrier_hebdo)} service(s)")
        )
    exceptions = {l.get("exception_type") for l in gtfs_io.lire(feed, "calendar_dates.txt")}
    if exceptions - {"1"}:
        violations.append(
            Violation("V1", BLOQUANT, f"calendar_dates contient des exception_type {sorted(exceptions - {'1'})}")
        )

    index_sortie = indexer(feed, journal=lambda *_: None)

    # V9 — uniqueness.
    vus_calendrier: set[tuple[str, str]] = set()
    doublons_calendrier = 0
    for ligne in gtfs_io.lire(feed, "calendar_dates.txt"):
        cle = (ligne["service_id"], ligne["date"])
        if cle in vus_calendrier:
            doublons_calendrier += 1
        vus_calendrier.add(cle)
    if doublons_calendrier:
        violations.append(
            Violation("V9", BLOQUANT, f"{doublons_calendrier} couple(s) (service_id, date) en double")
        )

    # V5 / V6 — stop times monotonic and consistent with their geometry.
    longueur_shape: dict[str, float] = {}
    for ligne in gtfs_io.lire(feed, "shapes.txt"):
        try:
            distance = float(ligne.get("shape_dist_traveled") or 0.0)
        except ValueError:
            continue
        shape_id = ligne["shape_id"]
        if distance > longueur_shape.get(shape_id, -1.0):
            longueur_shape[shape_id] = distance
    shape_par_trip = {
        l["trip_id"]: l.get("shape_id", "") for l in gtfs_io.lire(feed, "trips.txt")
    }

    precedent_trip = None
    precedent_sequence = -1
    precedent_temps = -1
    non_monotones = 0
    depassements = 0
    trips_depassants: set[str] = set()
    doublons_horaires = 0
    vus_horaires: set[tuple[str, str]] = set()
    for ligne in gtfs_io.lire(feed, "stop_times.txt"):
        trip_id = ligne["trip_id"]
        cle = (trip_id, ligne["stop_sequence"])
        if cle in vus_horaires:
            doublons_horaires += 1
        vus_horaires.add(cle)

        sequence = int(ligne["stop_sequence"])
        temps = _secondes(ligne.get("departure_time") or ligne.get("arrival_time") or "")
        if trip_id != precedent_trip:
            precedent_trip, precedent_sequence, precedent_temps = trip_id, sequence, temps
        else:
            if sequence <= precedent_sequence or (temps >= 0 and temps < precedent_temps):
                non_monotones += 1
            precedent_sequence, precedent_temps = sequence, max(temps, precedent_temps)

        shape_id = shape_par_trip.get(trip_id, "")
        if shape_id and shape_id in longueur_shape:
            try:
                distance = float(ligne.get("shape_dist_traveled") or 0.0)
            except ValueError:
                distance = 0.0
            if distance > longueur_shape[shape_id] + 1.0:
                depassements += 1
                trips_depassants.add(trip_id)

    if doublons_horaires:
        violations.append(
            Violation("V9", BLOQUANT, f"{doublons_horaires} couple(s) (trip_id, stop_sequence) en double")
        )
    if non_monotones:
        violations.append(
            Violation("V5", BLOQUANT, f"{non_monotones} horaire(s) non monotone(s) dans leur course")
        )
    if depassements:
        # Does the defect come from the build or from the source? A chimera path is
        # produced by a merge of geometries; a `shape_dist_traveled`
        # published too long by the operator is copied as is, and rejecting the
        # feed for that would amount to requiring the pipeline to repair the
        # source. The comparison is only made if there is something to
        # explain: it rereads the exports.
        herites: set[str] = set()
        for index in index_par_export.values():
            herites |= trips_hors_geometrie(index.export)
        etiquettes = set(index_par_export)
        fabriques = {_id_source(t, etiquettes) for t in trips_depassants} - herites
        if fabriques:
            violations.append(
                Violation(
                    "V6",
                    BLOQUANT,
                    f"{depassements} horaire(s) sur {len(trips_depassants)} course(s) dépassent la "
                    f"longueur de leur géométrie, dont {len(fabriques)} course(s) que la source ne "
                    f"portait pas — signature d'un tracé chimère",
                )
            )
        else:
            violations.append(
                Violation(
                    "V6",
                    ALARME,
                    f"{depassements} horaire(s) sur {len(trips_depassants)} course(s) dépassent la "
                    f"longueur de leur géométrie — toutes déjà défectueuses dans la source, recopiées "
                    f"telles quelles et non fabriquées par le build",
                )
            )

    # V2 / V3 — the service of real dates is intact.
    dates_reelles = {d for d, p in plan.items() if p.mode == REEL}
    empreintes_sortie = empreintes_par_date(
        feed, index_sortie.trips_par_date, dates_reelles, config, journal
    )
    ecarts: list[str] = []
    for etiquette, index in index_par_export.items():
        dates_ici = {d for d in dates_reelles if plan[d].export == etiquette}
        if not dates_ici:
            continue
        empreintes_source = empreintes_par_date(
            index.export, index.trips_par_date, dates_ici, config, journal
        )
        for date in sorted(dates_ici):
            if empreintes_sortie.get(date) != empreintes_source.get(date):
                ecarts.append(date)
    if ecarts:
        violations.append(
            Violation(
                "V2",
                BLOQUANT,
                f"{len(ecarts)} date(s) réelle(s) dont l'offre diffère de sa source : "
                f"{', '.join(ecarts[:10])}{'…' if len(ecarts) > 10 else ''}",
            )
        )
    else:
        journal(f"    V2 : offre identique à la source sur les {len(dates_reelles)} dates réelles")

    # V7 — an extrapolated day is a COPY: it must serve exactly
    # the service its donor actually served, fingerprint against fingerprint.
    # This is the direct statement of the pipeline's promise ("no stop time
    # is synthesised"); the volume compared with the signature's envelope,
    # which stood in for it, became note V7c: it wrongly flagged
    # donors taken from the other half of the school year (liO publishes 4 300
    # trips on autumn 2026 working days and 3 450 on spring 2027 ones —
    # a faithful copy of 15/03/2027 then falls "outside the envelope").
    dates_extrapolees = {d for d, p in plan.items() if p.mode == EXTRAPOLE and p.date_source}
    empreintes_copies = empreintes_par_date(
        feed, index_sortie.trips_par_date, dates_extrapolees, config, journal
    )
    copies_infideles: list[str] = []
    for etiquette, index in index_par_export.items():
        cibles = {d for d in dates_extrapolees if plan[d].export == etiquette}
        if not cibles:
            continue
        empreintes_donneurs = empreintes_par_date(
            index.export, index.trips_par_date, {plan[d].date_source for d in cibles}, config, journal
        )
        for date in sorted(cibles):
            if empreintes_copies.get(date) != empreintes_donneurs.get(plan[date].date_source):
                copies_infideles.append(f"{date}←{plan[date].date_source}")
    if copies_infideles:
        violations.append(
            Violation(
                "V7",
                BLOQUANT,
                f"{len(copies_infideles)} journée(s) extrapolée(s) ne reproduisent pas l'offre de leur "
                f"donneur : {', '.join(copies_infideles[:8])}{'…' if len(copies_infideles) > 8 else ''}",
            )
        )
    elif dates_extrapolees:
        journal(
            f"    V7 : les {len(dates_extrapolees)} journées extrapolées servent exactement l'offre "
            f"de leur donneur"
        )

    enveloppe: dict[str, list[int]] = {}
    for date, provenance in plan.items():
        if provenance.mode != REEL:
            continue
        enveloppe.setdefault(provenance.signature, []).append(index_sortie.nb_trips(date))

    # V7c — note: a faithful copy can fall outside the envelope of the real
    # days of the target year when its donor comes from another year.
    hors_enveloppe: list[str] = []
    for date, provenance in sorted(plan.items()):
        if provenance.mode != EXTRAPOLE or "repli" in provenance.motif:
            continue
        bornes = enveloppe.get(provenance.signature)
        if not bornes:
            continue
        nb = index_sortie.nb_trips(date)
        if not (min(bornes) <= nb <= max(bornes)):
            hors_enveloppe.append(f"{date} ({nb} hors [{min(bornes)}, {max(bornes)}])")
    if hors_enveloppe:
        journal(
            f"    V7c : {len(hors_enveloppe)} journée(s) extrapolée(s) hors de l'enveloppe réelle de "
            f"leur signature dans l'année cible (donneur d'une autre année) — "
            f"{', '.join(hors_enveloppe[:5])}{'…' if len(hors_enveloppe) > 5 else ''}"
        )

    # V7b — heterogeneity of the source itself. Informative: it is not a
    # build defect but a property of the transit system, worth knowing to read the
    # feed (the school period is not homogeneous — service decreases over the
    # last weeks of June).
    dispersion_max = float(config["controles"]["dispersion_signature_max"])
    heterogenes: list[str] = []
    for signature_str, valeurs in sorted(enveloppe.items()):
        if len(valeurs) < 3:
            continue
        moyenne = sum(valeurs) / len(valeurs)
        if moyenne <= 0:
            continue
        dispersion = (max(valeurs) - min(valeurs)) / moyenne
        if dispersion > dispersion_max:
            heterogenes.append(
                f"{signature_str} {dispersion:.0%} ({min(valeurs)}→{max(valeurs)})"
            )
    if heterogenes:
        journal(
            f"    V7b : {len(heterogenes)} signature(s) hétérogènes dans la source elle-même — "
            + " · ".join(heterogenes)
        )

    # V8 — continuity of the year. The floor of active routes is calibrated on
    # the transit system itself: an absolute threshold designed for Tisséo (123 routes)
    # would cry wolf on the TER, which has only about twenty.
    ratio_min = float(config["controles"]["ratio_lignes_min_jour_ouvre"])
    lignes_ouvrees = sorted(
        index_sortie.lignes_par_date.get(d, 0)
        for d, p in plan.items()
        if p.mode != SANS_SERVICE
        and JOURS[to_date(d).weekday()] not in ("sam", "dim")
        and p.signature.split("/")[0] != "ferie"
    )
    reference = lignes_ouvrees[len(lignes_ouvrees) // 2] if lignes_ouvrees else 0
    lignes_min = int(ratio_min * reference)
    creux: list[str] = []
    for date, provenance in sorted(plan.items()):
        nb = index_sortie.nb_trips(date)
        if provenance.mode == SANS_SERVICE:
            if nb:
                violations.append(
                    Violation("V8", BLOQUANT, f"{date} déclarée sans service mais sert {nb} trips")
                )
            continue
        if nb == 0:
            violations.append(Violation("V8", BLOQUANT, f"{date} sans aucune offre"))
            continue
        jour = JOURS[to_date(date).weekday()]
        if jour not in ("sam", "dim") and provenance.signature.split("/")[0] != "ferie":
            if index_sortie.lignes_par_date.get(date, 0) < lignes_min:
                creux.append(date)
    if creux:
        violations.append(
            Violation(
                "V8",
                ALARME,
                f"{len(creux)} jour(s) ouvré(s) sous {lignes_min} lignes actives : "
                f"{', '.join(creux[:10])}{'…' if len(creux) > 10 else ''}",
            )
        )

    return violations, empreintes_sortie, index_sortie


def _secondes(heure: str) -> int:
    """GTFS time in seconds, tolerating values beyond 24:00:00."""
    if not heure or heure.count(":") != 2:
        return -1
    try:
        h, m, s = (int(x) for x in heure.split(":"))
    except ValueError:
        return -1
    return h * 3600 + m * 60 + s


def holdout(
    plan_masque: dict[str, Provenance],
    index_sortie_masquee: IndexExport,
    verite: dict[str, int],
    ecart_max: float,
    journal=print,
) -> tuple[list[dict], float]:
    """Compares the extrapolated service of a masked month with its real service.

    Without this measurement, the extrapolation model is plausible but untested.
    """
    rapport: list[dict] = []
    pire = 0.0
    for date in sorted(verite):
        attendu = verite[date]
        obtenu = index_sortie_masquee.nb_trips(date)
        provenance = plan_masque.get(date)
        ecart = abs(obtenu - attendu) / attendu if attendu else 0.0
        pire = max(pire, ecart)
        rapport.append(
            {
                "date": date,
                "signature": provenance.signature if provenance else "",
                "mode": provenance.mode if provenance else "",
                "date_source": provenance.date_source if provenance else "",
                "trips_reels": attendu,
                "trips_extrapoles": obtenu,
                "ecart_relatif": round(ecart, 4),
            }
        )
    hors_tolerance = [r for r in rapport if r["ecart_relatif"] > ecart_max]
    if hors_tolerance:
        journal(
            f"[ALARME] hold-out : {len(hors_tolerance)}/{len(rapport)} journée(s) au-delà de "
            f"{ecart_max:.0%} d'écart (pire : {pire:.1%})"
        )
    else:
        journal(
            f"    hold-out : {len(rapport)} journée(s) extrapolée(s), écart maximal {pire:.1%} "
            f"(tolérance {ecart_max:.0%})"
        )
    return rapport, pire
