"""The time requested from the routing engines is GAMA's WALL-CLOCK time (2026-09-04).

The defect fixed: `services/GAMA/CityTransport/models/Settings.gaml` publishes its clock as
`int(current_date - UTC_START_DATE)`, a difference of two **naive** dates — hence a
local wall-clock time encoded as if it were UTC (1773637200 for Monday 16 March 2026
5:00). The runtime read it with `datetime.fromtimestamp(...)`, without a time zone, hence in
the **process's** one: under `TZ=Europe/Paris`, 5:00 wall-clock was requested from OTP as
**6:00 local**, and the congestion factor priced at 6:00 as well.

What it cost, measured on the 2,580 points of the sealed population v4
(`docs/traces/2026-09-04_13-15_fuseau_otp/`): **235** points without a transit itinerary at
the requested time against **605** at the agents' time. The bias was one hour in March
and **two** for a simulated day in summer — it was not even constant.

These tests fail if the convention changes again: they check the time to the second,
in winter AND in summer, equality across all consumers of `departure_time`,
independence from the process `TZ`, and the explicit refusal when the time zone has no
source (a guessed time would be a plausible result drawn from absent data).
"""

import asyncio
import calendar
import itertools
import os
import time
from datetime import datetime, timezone

import pytest

import sim_clock
from models import Location
from trip_helper.osmnx_persistent_cache import OsmnxPersistentCache
from trip_helper.otp import OTPTripHelper
from trip_helper.otp_persistent_cache import OtpPersistentCache
from settings import settings


def _gama_ts(annee, mois, jour, heure, minute=0, seconde=0) -> int:
    """The timestamp GAMA publishes for a given WALL-CLOCK time.

    Reproduces `int(current_date - UTC_START_DATE)`: a difference of naive dates,
    i.e. wall-clock time counted as if it were UTC.
    """
    return int(calendar.timegm((annee, mois, jour, heure, minute, seconde, 0, 0, 0)))


# Monday 16 March 2026, 5:00 wall-clock — the t0 of `starting_date` in Settings.gaml, and the
# value read in the "Temps simulé" column of moves.csv.
TS_HIVER = 1773637200
# Monday 13 July 2026, 5:00 wall-clock — summer time, when the gap was TWO hours.
TS_ETE = _gama_ts(2026, 7, 13, 5)

ORIGIN = Location(lat=43.6045, lon=1.4440, public_transport=True)
DEST = Location(lat=43.5710, lon=1.4020, public_transport=True)


@pytest.fixture(autouse=True)
def _horloge_propre():
    """The time zone is looked up once and cached: each test starts from scratch."""
    sim_clock.reset_cache()
    yield
    sim_clock.reset_cache()


def test_lhorodatage_de_gama_est_une_heure_murale():
    """The reference of the statement: 1773637200 IS Monday 16 March 2026 5:00 wall-clock."""
    assert TS_HIVER == _gama_ts(2026, 3, 16, 5)
    assert sim_clock.wall_clock(TS_HIVER) == datetime(2026, 3, 16, 5, 0, 0)
    assert sim_clock.wall_clock(TS_ETE) == datetime(2026, 7, 13, 5, 0, 0)


def test_lheure_demandee_a_otp_egale_lheure_murale_de_gama_hiver_et_ete():
    """Zero residual offset, to the second, on both sides of the clock change.

    This is the requested measure: GAMA's wall-clock time and the local time received by OTP
    must be the same, in winter (+01:00) as in summer (+02:00).
    """
    for ts in (TS_HIVER, TS_ETE):
        demande = datetime.fromisoformat(sim_clock.network_iso(ts))
        mur = sim_clock.wall_clock(ts)
        assert demande.replace(tzinfo=None) == mur, ts
        # And the offset carried is indeed the season's: the same wall-clock time
        # is not the same instant in March and in July.
        assert demande.utcoffset().total_seconds() == (3600 if ts == TS_HIVER else 7200)

    assert sim_clock.network_iso(TS_HIVER) == "2026-03-16T05:00:00+01:00"
    assert sim_clock.network_iso(TS_ETE) == "2026-07-13T05:00:00+02:00"


def test_lancienne_convention_demandait_une_heure_de_trop():
    """No-regression safeguard: the reading without a time zone stays wrong, and by how much.

    Without this test, reverting to `datetime.fromtimestamp(ts)` would break nothing while the
    test machine runs in UTC — that is precisely the asymmetry that let the
    defect through (`otp_link_check.py` queried OTP in local time, the runtime did not).
    """
    ancien_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Europe/Paris"
        time.tzset()
        assert datetime.fromtimestamp(TS_HIVER).hour == 6   # one hour too many
        assert datetime.fromtimestamp(TS_ETE).hour == 7     # two hours too many
        # And the old `dateTime`: wall-clock time stamped UTC, which OTP translated
        # into its network's time zone (hence 06:00 then 07:00 local).
        assert datetime.fromtimestamp(TS_HIVER, tz=timezone.utc).isoformat() == \
            "2026-03-16T05:00:00+00:00"
    finally:
        if ancien_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = ancien_tz
        time.tzset()


@pytest.mark.parametrize("tz_processus", ["UTC", "Europe/Paris", "America/Los_Angeles",
                                          "Pacific/Kiritimati"])
def test_le_fuseau_vient_de_la_configuration_pas_du_processus(tz_processus):
    """A misconfigured container must not shift the itineraries.

    The `osmnx` replicas run in UTC, the `controller` in Europe/Paris and the
    populator on the host: three readings of the same integer, three different times.
    """
    ancien_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = tz_processus
        time.tzset()
        sim_clock.reset_cache()
        assert sim_clock.network_iso(TS_HIVER) == "2026-03-16T05:00:00+01:00"
        assert sim_clock.network_iso(TS_ETE) == "2026-07-13T05:00:00+02:00"
        assert sim_clock.wall_clock(TS_HIVER) == datetime(2026, 3, 16, 5, 0, 0)
    finally:
        if ancien_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = ancien_tz
        time.tzset()


# ── Where the time zone comes from ───────────────────────────────────────────

def _feed(repertoire, nom, fuseau):
    """A minimal GTFS feed: `stops.txt` (to be seen as in service) + `agency.txt`."""
    feed = repertoire / nom
    feed.mkdir(parents=True)
    (feed / "stops.txt").write_text("stop_id,stop_lat,stop_lon\nA,43.6,1.44\n", encoding="utf-8")
    if fuseau is not None:
        (feed / "agency.txt").write_text(
            f"agency_id,agency_name,agency_timezone\n1,Test,{fuseau}\n", encoding="utf-8")
    return feed


def test_le_fuseau_est_lu_dans_le_feed_gtfs(tmp_path, monkeypatch):
    """The source is the feed's `agency_timezone`, not a literal in the code.

    It is the only source that cannot diverge from OTP: it is the one OTP itself
    uses to interpret the network's timetables. This test proves it by moving the
    network to New York — a hard-coded "Europe/Paris" would make it fail.
    """
    feed = _feed(tmp_path / "build", "reseau_gtfs", "America/New_York")
    monkeypatch.setattr(settings.gtfs, "gtfs_file", str(feed))
    monkeypatch.setattr(settings.gtfs, "network_timezone", None)
    sim_clock.reset_cache()
    assert sim_clock.network_timezone_name() == "America/New_York"
    # 5:00 wall-clock in New York, mid-March: US daylight saving time, offset −04:00.
    assert sim_clock.network_iso(TS_HIVER) == "2026-03-16T05:00:00-04:00"


def test_les_trois_feeds_de_production_declarent_le_meme_fuseau():
    """Tisséo, liO and the annual TER: `Europe/Paris` in all three `agency.txt`."""
    monkeypatched = getattr(settings.gtfs, "network_timezone", None)
    assert monkeypatched is None, "the production setting must remain reading the feed"
    assert sim_clock.network_timezone_name() == "Europe/Paris"


def test_le_reglage_explicite_prime_sur_le_feed(tmp_path, monkeypatch):
    """The documented escape hatch, for a feed missing from the service or contradictory."""
    feed = _feed(tmp_path / "build", "reseau_gtfs", "America/New_York")
    monkeypatch.setattr(settings.gtfs, "gtfs_file", str(feed))
    monkeypatch.setattr(settings.gtfs, "network_timezone", "Europe/Lisbon")
    sim_clock.reset_cache()
    assert sim_clock.network_timezone_name() == "Europe/Lisbon"
    assert sim_clock.network_iso(TS_HIVER) == "2026-03-16T05:00:00+00:00"


def test_deux_feeds_qui_se_contredisent_refusent(tmp_path, monkeypatch):
    """Picking one at random would shift the timetables of an entire network."""
    build = tmp_path / "build"
    _feed(build, "a_gtfs", "Europe/Paris")
    feed_b = _feed(build, "b_gtfs", "America/New_York")
    monkeypatch.setattr(settings.gtfs, "gtfs_file", str(feed_b))
    monkeypatch.setattr(settings.gtfs, "network_timezone", None)
    sim_clock.reset_cache()
    with pytest.raises(sim_clock.NetworkTimezoneError, match="contradictory"):
        sim_clock.network_timezone()


def test_sans_feed_lisible_la_conversion_refuse(tmp_path, monkeypatch):
    """The absence of a measure does not produce a plausible result.

    A fallback to "Europe/Paris" would return the right time here and the wrong one elsewhere,
    without any signal — exactly the shape of the defect fixed.
    """
    feed = _feed(tmp_path / "build", "reseau_gtfs", None)  # stops.txt but no agency
    monkeypatch.setattr(settings.gtfs, "gtfs_file", str(feed))
    monkeypatch.setattr(settings.gtfs, "network_timezone", None)
    sim_clock.reset_cache()
    with pytest.raises(sim_clock.NetworkTimezoneError, match="agency_timezone"):
        sim_clock.network_timezone()


def test_un_fuseau_inconnu_de_la_base_iana_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.gtfs, "network_timezone", "Mars/Olympus_Mons")
    sim_clock.reset_cache()
    with pytest.raises(sim_clock.NetworkTimezoneError, match="fuseau inconnu"):
        sim_clock.network_timezone()


# ── What the two engines actually receive ────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeSession:
    """Captures the body POSTed to OTP instead of sending it."""

    def __init__(self, reponse):
        self.reponse = reponse
        self.corps = []

    def post(self, url, json=None, timeout=None):
        self.corps.append(json)
        return _FakeResponse(self.reponse)


def _helper(session, fixed_day=None):
    """An `OTPTripHelper` without GTFS or network: only building the request
    is under test, and `GTFSData.DEFAULT()` would cost several seconds of reading."""
    h = object.__new__(OTPTripHelper)
    h.fixed_day = fixed_day
    h._endpoint_iter = itertools.cycle(["http://otp/otp/transmodel/v3"])
    h._semaphore = asyncio.Semaphore(4)
    h._session = None
    h.gtfs_data = None
    h._stop_coords = None

    async def _get_session():
        return session

    h.get_session = _get_session
    return h


_AUCUN_MOTIF = {"data": {"trip": {"tripPatterns": []}}}


@pytest.mark.parametrize("ts,attendu", [(TS_HIVER, "2026-03-16T05:00:00+01:00"),
                                        (TS_ETE, "2026-07-13T05:00:00+02:00")])
def test_le_datetime_envoye_a_otp_porte_lheure_murale_de_gama(ts, attendu):
    """The `dateTime` of the GraphQL request, as OTP receives it."""
    session = _FakeSession(_AUCUN_MOTIF)
    helper = _helper(session)
    asyncio.run(helper.get_itineraries(ORIGIN, DEST, ts, include_direct=False))
    assert len(session.corps) == 1
    assert session.corps[0]["variables"]["dateTime"] == attendu


@pytest.mark.parametrize("ts,heure,jour", [(TS_HIVER, 5, "Mon"), (TS_ETE, 5, "Mon")])
def test_la_congestion_est_lue_a_la_meme_heure_que_otp(monkeypatch, ts, heure, jour):
    """A single time for both engines.

    The TomTom factor is tabulated by day of the week and whole hour
    (`osmnx_direct._zone_factor`): read one hour too late, it priced the morning
    peak on a trip leaving before it. And the routing cache key carries that
    same slot, so the error was memorized.
    """
    vus = []

    async def _fake_direct(origin, destination, trip_mode, departure_time, congestion_dt,
                           _timing_sink=None):
        vus.append((trip_mode, departure_time, congestion_dt))
        return None

    monkeypatch.setattr("trip_helper.otp.get_direct_plan", _fake_direct)
    session = _FakeSession(_AUCUN_MOTIF)
    helper = _helper(session)
    asyncio.run(helper.get_itineraries(ORIGIN, DEST, ts, include_direct=True,
                                       include_car=True, include_transit=False))

    assert vus, "no OSMnx call captured"
    for trip_mode, departure_time, congestion_dt in vus:
        assert departure_time == ts, trip_mode
        assert congestion_dt.hour == heure, trip_mode
        assert congestion_dt.strftime("%a") == jour, trip_mode
        # The instant is aware of the network's time zone: the cache row and the
        # log say which time was priced, not only which one it was
        # in the process. And it is EXACTLY the instant sent to OTP.
        assert congestion_dt.tzinfo is not None, trip_mode
        assert congestion_dt.replace(tzinfo=None) == sim_clock.wall_clock(ts)
        assert congestion_dt.isoformat() == sim_clock.network_iso(ts), trip_mode


def _motif(depart_iso, arrivee_iso):
    place_o = {"name": "Origin", "latitude": ORIGIN.lat, "longitude": ORIGIN.lon}
    place_d = {"name": "Destination", "latitude": DEST.lat, "longitude": DEST.lon}
    leg = {"mode": "foot", "aimedStartTime": depart_iso, "aimedEndTime": arrivee_iso,
           "expectedStartTime": depart_iso, "expectedEndTime": arrivee_iso,
           "realtime": False, "distance": 900.0, "duration": 720,
           "fromPlace": place_o, "toPlace": place_d}
    return {"data": {"trip": {"tripPatterns": [{
        "aimedStartTime": depart_iso, "aimedEndTime": arrivee_iso,
        "expectedStartTime": depart_iso, "expectedEndTime": arrivee_iso,
        "duration": 720, "distance": 900.0, "legs": [leg], "systemNotices": []}]}}}


@pytest.mark.parametrize("ts,depart,arrivee", [
    (TS_HIVER, "2026-03-16T05:12:00+01:00", "2026-03-16T05:24:00+01:00"),
    (TS_ETE, "2026-07-13T05:12:00+02:00", "2026-07-13T05:24:00+02:00"),
])
def test_les_horaires_rendus_par_otp_reviennent_dans_lhorloge_de_gama(ts, depart, arrivee):
    """The way back counts as much as the way out.

    GAMA has no instants: its times are wall-clock. A plan whose times
    stayed in real epoch would give a `start_in` off by the time-zone offset — options
    "leaving 55 minutes in the past" in winter, 1 h 48 in summer — and
    legs pushed to GAMA one hour away from its own `CURRENT_TIMESTAMP`.
    """
    session = _FakeSession(_motif(depart, arrivee))
    helper = _helper(session)
    plans = asyncio.run(helper.get_itineraries(ORIGIN, DEST, ts, include_direct=False))

    assert len(plans) == 1
    plan = plans[0]
    assert plan.start_time // 1000 == ts + 12 * 60      # 12 min after the requested departure
    assert plan.end_time // 1000 == ts + 24 * 60
    assert plan.start_in == 12 * 60
    # And the plan's wall-clock time is the one OTP wrote.
    assert sim_clock.wall_clock(plan.start_time // 1000).strftime("%H:%M") == "05:12"


def test_le_remappage_fixed_day_reste_dans_lhorloge_de_gama(monkeypatch):
    """`gtfs.fixed_day` rebuilds a GAMA timestamp, not a real instant.

    `self.fixed_day.replace(...).timestamp()` went through the process time zone: the
    remapping therefore added its own offset to the request's.
    """
    session = _FakeSession(_AUCUN_MOTIF)
    helper = _helper(session, fixed_day=datetime(2026, 3, 16))
    asyncio.run(helper.get_itineraries(ORIGIN, DEST, TS_ETE, include_direct=False))
    # 5:00 wall-clock on 13 July, moved to 16 March: 5:00 wall-clock, +01:00.
    assert session.corps[0]["variables"]["dateTime"] == "2026-03-16T05:00:00+01:00"


def test_le_remappage_fixed_day_rend_les_horaires_au_jour_reel():
    """And the return path of the remapping: `real_day - fixed_day` in days.

    `real_day` derives from `congestion_dt`, now time-zone AWARE, whereas
    `fixed_day` is naive: subtracting one from the other would raise a `TypeError`. This
    test walks the full branch, which `gtfs.fixed_day: null` leaves silent in
    production.
    """
    session = _FakeSession(_motif("2026-03-16T05:12:00+01:00", "2026-03-16T05:24:00+01:00"))
    helper = _helper(session, fixed_day=datetime(2026, 3, 16))
    plans = asyncio.run(helper.get_itineraries(ORIGIN, DEST, TS_ETE, include_direct=False))

    assert len(plans) == 1
    # The plan is returned on the REAL day (13 July), at the wall-clock time OTP wrote.
    assert sim_clock.wall_clock(plans[0].start_time // 1000) == datetime(2026, 7, 13, 5, 12)


# ── The caches ───────────────────────────────────────────────────────────────

def test_la_cle_du_cache_de_routage_suit_lheure_murale():
    """`OsmnxPersistentCache` indexes the congestion factor's slot.

    Wall-clock time and the time read must coincide: otherwise a duration priced at 6:00 is
    filed under "6:00" while it answers a 5:00 departure, and the row does not say
    which of the two it describes.
    """
    _, date_str, dow, bucket = OsmnxPersistentCache.make_key(
        sim_clock.to_network_datetime(TS_HIVER), "car", 43.6045, 1.4440, 43.5710, 1.4020)
    assert (date_str, dow, bucket) == ("2026-03-16", 0, "05:00")

    _, date_ete, dow_ete, bucket_ete = OsmnxPersistentCache.make_key(
        sim_clock.to_network_datetime(TS_ETE), "car", 43.6045, 1.4440, 43.5710, 1.4020)
    assert (date_ete, dow_ete, bucket_ete) == ("2026-07-13", 0, "05:00")


def test_la_cle_du_cache_otp_porte_linstant_avec_son_decalage():
    """What makes the old generation out of reach, without a manual purge.

    This cache stores serialized `TravelPlan`s and SHIFTS them on reuse
    (`lookup` → `departure_time - stored_departure_time`). An entry filed under
    the label "06:00" while it answered a 5:00 wall-clock departure would be
    served again to a 6:00 wall-clock departure, then shifted by one more hour. Neither
    `data_version()` nor `routing_version` prevented it: the key must carry
    the instant, offset included.
    """
    cle_hiver = OtpPersistentCache.make_key(TS_HIVER, ORIGIN, DEST, False, False, True)
    cle_ete = OtpPersistentCache.make_key(TS_ETE, ORIGIN, DEST, False, False, True)
    assert cle_hiver != cle_ete

    # The same wall-clock time one hour later is ANOTHER entry…
    cle_six_heures = OtpPersistentCache.make_key(
        TS_HIVER + 3600, ORIGIN, DEST, False, False, True)
    assert cle_six_heures != cle_hiver

    # …and the old convention (bare date and time, read in the process time zone)
    # can no longer produce any of these keys: the shape of the string has changed.
    import hashlib

    from trip_helper.terminal_time import data_version

    ancien = (f"{data_version()}|2026-03-16|06:00|{ORIGIN.lat:.5f}|{ORIGIN.lon:.5f}"
              f"|{DEST.lat:.5f}|{DEST.lon:.5f}|0|0|1")
    assert hashlib.sha256(ancien.encode()).hexdigest() not in (cle_hiver, cle_six_heures)

    # The 10 min slice remains the cache granularity (it was already documented).
    assert OtpPersistentCache.make_key(TS_HIVER + 120, ORIGIN, DEST, False, False, True) \
        == cle_hiver


# ── Clock change: stated, never guessed ──────────────────────────────────────

def test_lheure_murale_inexistante_est_alarmee():
    """The wall-clock day of the switch to summer time counts 24 h, reality 23.

    2:30 on 29 March 2026 does not exist in Europe/Paris. The conversion goes on with
    `fold=0` — an announced choice — but it SAYS so: without this line, the itinerary
    for a time that does not exist would be returned like any other.
    """
    from loguru import logger as _loguru

    messages = []
    sink = _loguru.add(lambda m: messages.append(m.record["message"]), level="ERROR")
    try:
        sim_clock.to_network_datetime(_gama_ts(2026, 3, 29, 2, 30))
    finally:
        _loguru.remove(sink)
    assert any("[ALARME]" in m and "n'existe pas" in m for m in messages), messages


def test_lheure_murale_ambigue_est_alarmee():
    """2:30 on 25 October 2026 exists twice: the first is retained, and stated."""
    from loguru import logger as _loguru

    messages = []
    sink = _loguru.add(lambda m: messages.append(m.record["message"]), level="ERROR")
    try:
        sim_clock.to_network_datetime(_gama_ts(2026, 10, 25, 2, 30))
    finally:
        _loguru.remove(sink)
    assert any("[ALARME]" in m and "DEUX fois" in m for m in messages), messages


def test_lalarme_de_bascule_se_leve_sur_front_montant():
    """A simulated day counts thousands of trips: one alarm, not a flood."""
    from loguru import logger as _loguru

    messages = []
    sink = _loguru.add(lambda m: messages.append(m.record["message"]), level="ERROR")
    try:
        for minute in range(0, 60, 5):
            sim_clock.to_network_datetime(_gama_ts(2026, 3, 29, 2, minute))
    finally:
        _loguru.remove(sink)
    assert len([m for m in messages if "n'existe pas" in m]) == 1, messages
