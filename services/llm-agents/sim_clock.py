"""The simulation clock: a GAMA timestamp is a local WALL-CLOCK time.

`services/GAMA/CityTransport/models/Settings.gaml` starts the simulated day at
`date([2026,3,16,5,0,0]) // Lundi 5h` and publishes its clock as follows:

    date UTC_START_DATE  <- date([1970,1,1,0,0,0]);
    int  CURRENT_TIMESTAMP -> int(current_date - UTC_START_DATE);

It is the difference of two **naive** dates. The integer that comes out is therefore not
an instant: it is the **wall-clock** time of the simulation, encoded as if it
were UTC. For Monday 16 March 2026 5 a.m., it equals **1773637200** (read from the
"Temps simulé" column of `experiments/archive/2026-09-04_01_09/moves.csv`).

Reading it with `datetime.fromtimestamp(ts)` — without a time zone — sends it through the
**process's** time zone. In the `controller` container (`TZ=Europe/Paris`), 5 a.m.
wall-clock became **6 a.m.**, and the process's time zone therefore decided the time at
which the agents saw the network. Measured on the 2,580 points of the
v4 sealed population (`docs/traces/2026-09-04_13-15_fuseau_otp/`): **235**
points without a transit itinerary at the requested time, **605** at the time the model
thought it was requesting. The bias is not even constant — one hour in March, **two**
for a simulated day in summer.

This module is the ONLY place that translates this integer:

    wall_clock(ts)          -> naive datetime carrying GAMA's wall-clock fields
    to_network_datetime(ts) -> the same instant, AWARE of the network's time zone
    network_iso(ts)         -> its ISO-8601 form (what OTP expects in `dateTime`)
    gama_timestamp(dt)      -> the inverse: from an instant (or a wall-clock time)
                               to the GAMA timestamp

⚠ The time zone is that of the **simulated network**, not that of the process: a badly
configured container must not shift the itineraries. It is read from the GTFS feeds
in service (`agency.txt`, column `agency_timezone`) — the same source OTP
uses to interpret its schedules — and is overridden by
`settings.gtfs.network_timezone`. No hard-coded fallback: without a readable source,
the conversion REFUSES instead of returning a plausible time.
"""

from __future__ import annotations

import calendar
import csv
import io
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from settings import settings

__all__ = [
    "NetworkTimezoneError",
    "gama_timestamp",
    "network_iso",
    "network_timezone",
    "network_timezone_name",
    "reset_cache",
    "to_network_datetime",
    "wall_clock",
]


class NetworkTimezoneError(RuntimeError):
    """The time zone of the simulated network has no readable source.

    Deliberately fatal: guessing "Europe/Paris" would turn a missing datum into a
    plausible result, and that is exactly the defect this module fixes.
    """


# Time zone resolved once (reading opens files), with the trace of its source.
_tz: Optional[ZoneInfo] = None
_tz_source: str = ""
# Daylight-saving anomalies already reported: (wall-clock date, nature) — the alarm
# fires on the rising edge, a simulated day counts thousands of trips.
_dst_alarmed: set[tuple[str, str]] = set()


def reset_cache() -> None:
    """Forget the resolved time zone (tests: the configuration or the feeds change)."""
    global _tz, _tz_source
    _tz = None
    _tz_source = ""
    _dst_alarmed.clear()


# ── The time zone of the simulated network ────────────────────────────────────

def _agency_timezones() -> dict[str, str]:
    """`{feed name: agency_timezone}` for the GTFS feeds in service.

    Feeds are enumerated as OTP does (`trip_helper.otp.feeds_en_service`:
    a directory or a zip carrying `stops.txt` at the first level of the build
    directory), and not only the primary feed — Tisséo, liO and the annual TER run
    in the same graph since 2026-09-04.
    """
    from trip_helper.otp import feeds_en_service  # late import: avoids the cycle

    trouves: dict[str, str] = {}
    for feed in feeds_en_service(settings.gtfs.gtfs_file):
        feed = Path(feed)
        try:
            if feed.is_dir():
                agence = feed / "agency.txt"
                if not agence.exists():
                    continue
                flux = open(agence, encoding="utf-8-sig", newline="")
            else:
                archive = zipfile.ZipFile(feed)
                if "agency.txt" not in archive.namelist():
                    continue
                flux = io.TextIOWrapper(archive.open("agency.txt"), encoding="utf-8-sig", newline="")
            with flux:
                for ligne in csv.DictReader(flux):
                    valeur = (ligne.get("agency_timezone") or "").strip()
                    if valeur:
                        trouves[feed.name] = valeur
                        break
        except (OSError, zipfile.BadZipFile, csv.Error) as exc:
            logger.warning(f"[horloge] unreadable agency.txt in {feed.name} : {exc}")
    return trouves


def network_timezone_name() -> str:
    """IANA name of the simulated network's time zone, and where it comes from.

    Order: the explicit setting `gtfs.network_timezone`, then the `agency_timezone`
    of the feeds in service. Two feeds that disagree, or no readable feed,
    raise `NetworkTimezoneError`: the time of itineraries is not guessed.
    """
    global _tz_source

    surcharge = getattr(settings.gtfs, "network_timezone", None)
    if surcharge:
        _tz_source = f"réglage gtfs.network_timezone={surcharge}"
        return surcharge

    par_feed = _agency_timezones()
    distincts = sorted(set(par_feed.values()))
    if not distincts:
        logger.error(
            "[ALARME] [horloge] fuseau du réseau introuvable : aucun agency.txt lisible "
            f"à côté de {settings.gtfs.gtfs_file}. Les itinéraires ne peuvent pas être "
            "demandés à une heure inconnue — montez les feeds GTFS dans le service, ou "
            "posez explicitement `gtfs.network_timezone`.")
        raise NetworkTimezoneError(
            f"no readable agency_timezone next to {settings.gtfs.gtfs_file} "
            "(set gtfs.network_timezone)")
    if len(distincts) > 1:
        logger.error(
            "[ALARME] [horloge] les feeds GTFS en service ne déclarent pas le même "
            f"fuseau : {par_feed}. En choisir un au hasard décalerait les horaires d'un "
            "réseau entier — posez explicitement `gtfs.network_timezone`.")
        raise NetworkTimezoneError(
            f"contradictory agency_timezone between feeds: {par_feed} "
            "(set gtfs.network_timezone)")

    _tz_source = ("agency.txt de " + ", ".join(f"{nom}={tz}" for nom, tz in sorted(par_feed.items())))
    return distincts[0]


def network_timezone() -> ZoneInfo:
    """Time zone of the simulated network (resolved once, logged with its source).

    Success is logged, not only failure: without this line, "the clock
    reads the network's time zone" cannot be told apart from "the module was never used".
    """
    global _tz
    if _tz is not None:
        return _tz
    nom = network_timezone_name()
    try:
        _tz = ZoneInfo(nom)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        logger.error(
            f"[ALARME] [horloge] time zone '{nom}' unknown to the IANA database ({exc}) — "
            f"source: {_tz_source}. No time is guessed in its place.")
        raise NetworkTimezoneError(f"fuseau inconnu : {nom}") from exc
    logger.info(
        f"[horloge] fuseau du réseau simulé : {nom} (source : {_tz_source}) — "
        f"fuseau du processus : {os.environ.get('TZ', '(non posé)')}, qui n'entre "
        "dans aucune conversion")
    return _tz


# ── The conversions ──────────────────────────────────────────────────────────

def wall_clock(timestamp: int | float) -> datetime:
    """GAMA timestamp → NAIVE datetime carrying the simulation's wall-clock time.

    `wall_clock(1773637200)` equals `datetime(2026, 3, 16, 5, 0)` whatever the
    process's `TZ`. To be used wherever only the FIELDS matter (hour of
    the congestion table, day of the week, date of the simulated day).
    """
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).replace(tzinfo=None)


def _signale_bascule(mur: datetime, tz: ZoneInfo) -> None:
    """Alarm (rising edge) when the wall-clock time does not exist or exists twice.

    GAMA's clock is a wall clock without daylight saving: its day
    of the last Sunday of March counts 24 wall-clock hours where reality only has
    23. The missing hour and the doubled hour are facts, not conversion
    details: they are stated, and the conversion continues on `fold=0` — an announced
    choice rather than a plausible instant produced in silence.
    """
    debut = mur.replace(tzinfo=tz)
    if debut.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) != mur:
        cle = (mur.strftime("%Y-%m-%d"), "inexistante")
        if cle not in _dst_alarmed:
            _dst_alarmed.add(cle)
            logger.error(
                f"[ALARME] [horloge] l'heure murale {mur.isoformat()} n'existe pas en "
                f"{tz.key} (passage à l'heure d'été) : la journée simulée compte une "
                "heure que le réseau n'a pas. Conversion poursuivie sur l'heure d'hiver "
                "(fold=0) ; les itinéraires de cette tranche sont à lire avec prudence.")
        return
    if debut.utcoffset() != mur.replace(tzinfo=tz, fold=1).utcoffset():
        cle = (mur.strftime("%Y-%m-%d"), "ambigue")
        if cle not in _dst_alarmed:
            _dst_alarmed.add(cle)
            logger.error(
                f"[ALARME] [horloge] l'heure murale {mur.isoformat()} existe DEUX fois en "
                f"{tz.key} (retour à l'heure d'hiver) : la première occurrence est "
                "retenue (fold=0). Les horaires GTFS de cette tranche sont ambigus.")


def to_network_datetime(timestamp: int | float) -> datetime:
    """GAMA timestamp → time-zone-AWARE instant, in the network's time zone.

    `to_network_datetime(1773637200)` equals `2026-03-16T05:00:00+01:00`, and
    `to_network_datetime(1783918800)` (5 a.m. wall-clock on 13 July) equals
    `2026-07-13T05:00:00+02:00`: the gap with the former reading is one hour in
    winter and two in summer.
    """
    tz = network_timezone()
    mur = wall_clock(timestamp)
    _signale_bascule(mur, tz)
    return mur.replace(tzinfo=tz)


def network_iso(timestamp: int | float) -> str:
    """GAMA timestamp → ISO-8601 with offset, as OTP expects it in `dateTime`.

    OTP is correct: it translates the received instant into its network's time zone. Sending it
    `05:00+00:00` for 5 a.m. wall-clock made it plan **6 a.m. local**.
    """
    return to_network_datetime(timestamp).isoformat()


def gama_timestamp(moment: datetime) -> int:
    """Inverse of :func:`to_network_datetime`: an instant → the GAMA timestamp.

    Used on both sides of the boundary:

    - the instants OTP RETURNS (`2026-03-16T05:12:00+01:00`) must come back
      into GAMA's clock, otherwise `start_in` — the gap between the requested departure and
      the plan's departure — is off by one hour and the agent sees options that
      "leave in the past";
    - the `gtfs.fixed_day` remapping, which rebuilds a timestamp from a
      fixed date and the requested wall-clock time.

    A naive datetime is read as a WALL-CLOCK time of the network (it is the only
    consistent reading for a transport schedule); a time-zone-aware datetime
    is first brought back into the network's.
    """
    if moment.tzinfo is not None:
        moment = moment.astimezone(network_timezone()).replace(tzinfo=None)
    return int(calendar.timegm(moment.timetuple()))
