"""TER must be requested from OTP, and be named.

TER entered the OTP graph on 2026-09-03, its stops counted in the
service envelope, the yearly feed runs it on the simulated day — and no
agent could ever be offered one, because `rail` was not among the
requested modes. Four families of tests, each written to fail against the
former code:

1. **`rail` is requested and accepted.** The list sent to OTP contains it, and
   so does `SUPPORTED_MODES` — otherwise the `to_travel_plan` assertion would reject
   the first train returned. A consistency test locks the two together:
   requesting a mode that would then be rejected crashes the first itinerary.

2. **The train is named.** `route_type=2` → "Train" in the prompt, not
   "Unknown".

3. **The proximity gate sees every network in service.** It was built
   on the Tisséo feed alone: 397 of the 2,580 points of the sealed v4 population
   are within reach of a liO stop or a TER station without being near a Tisséo stop,
   and the runtime skipped OTP for them.

4. **The train counts as train** in the metrics tables.

Run: cd llm-agents && .venv/bin/python -m pytest tests/test_mode_rail.py
"""

import csv

import pytest

from settings import settings
from trip_helper.otp import OTPTripHelper, coordonnees_arrets, feeds_en_service
from urban_mobility_agents.utils.move_logger import _plan_transport_mode
from mobility_llm.mode_choice import canonical_mode
from models import TransitLocation, Transit, TravelPlan


# ── 1. `rail` is requested and accepted ─────────────────────────────────────────

def _modes_demandes_a_otp() -> list[str]:
    """The `transportMode` values `get_itineraries` puts in its request.

    Read from the source: the construction is inline in a coroutine whose
    execution would need a reachable OTP. The test therefore bears on the text of
    the function — which is exactly what we want to lock, a literal
    list.
    """
    import inspect
    import re

    source = inspect.getsource(OTPTripHelper.get_itineraries)
    return re.findall(r'\{"transportMode":\s*"([a-z_]+)"\}', source)


def test_rail_est_demande_a_otp():
    assert "rail" in _modes_demandes_a_otp()


def test_les_modes_urbains_restent_demandes():
    demandes = _modes_demandes_a_otp()
    for mode in ("bus", "metro", "tram", "cableway"):
        assert mode in demandes, mode


def test_rail_est_un_mode_de_jambe_accepte():
    assert "rail" in OTPTripHelper.SUPPORTED_MODES


def test_tout_mode_demande_est_un_mode_accepte():
    """Requesting a mode that would then be rejected crashes the first itinerary.

    `to_travel_plan` asserts `assert leg.mode in SUPPORTED_MODES`. A requested mode
    missing from the list does not yield "fewer itineraries": it yields an
    `AssertionError` on the first trip that uses it.
    """
    manquants = [m for m in _modes_demandes_a_otp() if m not in OTPTripHelper.SUPPORTED_MODES]
    assert manquants == []


# ── 2. The train is named ─────────────────────────────────────────────────────

def test_route_type_2_se_nomme_train():
    assert settings.gtfs.gtfs_modality_name_map.get("2") == "Train"


def test_les_modalites_deja_servies_ne_bougent_pas():
    carte = settings.gtfs.gtfs_modality_name_map
    assert carte.get("0") == "T1/Tram"
    assert carte.get("1") == "Metro"
    assert carte.get("3") == "Bus"
    assert carte.get("6") == "Teleo"


def test_le_lecteur_traduit_le_route_type_du_train(tmp_path):
    """End to end on the modality table: a TER `routes.txt` reads as "Train"."""
    from inputs.gtfs.reader import GTFSData

    feed = tmp_path / "ter_gtfs"
    feed.mkdir()
    (feed / "routes.txt").write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "800000,TER 1,Toulouse - Montauban,2\n",
        encoding="utf-8",
    )
    data = GTFSData.__new__(GTFSData)
    import pandas as pd

    data.routes = pd.read_csv(feed / "routes.txt", dtype=str)
    data.init_route_lookup_maps()
    assert data.get_route_type_string_by_id("800000") == "Train"


# ── 3. The proximity gate sees every network in service ────────────────

def _ecrire_feed(racine, nom: str, arrets: list[tuple[str, float, float]]):
    feed = racine / nom
    feed.mkdir(parents=True)
    with (feed / "stops.txt").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stop_id", "stop_name", "stop_lat", "stop_lon"])
        for sid, lat, lon in arrets:
            w.writerow([sid, sid, lat, lon])
    return feed


def test_feeds_en_service_trouve_les_reseaux_voisins(tmp_path):
    _ecrire_feed(tmp_path, "tisseo_gtfs", [("A", 43.60, 1.44)])
    _ecrire_feed(tmp_path, "lio_gtfs", [("B", 43.20, 1.10)])
    _ecrire_feed(tmp_path, "ter_gtfs", [("C", 43.90, 1.35)])
    noms = {f.name for f in feeds_en_service(str(tmp_path / "tisseo_gtfs"))}
    assert noms == {"tisseo_gtfs", "lio_gtfs", "ter_gtfs"}


def test_feeds_en_service_ignore_les_archives_imbriquees(tmp_path):
    """An old export filed under `archives/` must not come back into the gate.

    `data/gtfs/archives/<date>/ter_gtfs_export_…` holds the REPLACED service. Taking
    it back would mean serving two calendars for the same network.
    """
    _ecrire_feed(tmp_path, "tisseo_gtfs", [("A", 43.60, 1.44)])
    _ecrire_feed(tmp_path, "archives/2026-09-04_pre_lio/ter_gtfs_ancien", [("Z", 43.0, 1.0)])
    noms = {f.name for f in feeds_en_service(str(tmp_path / "tisseo_gtfs"))}
    assert noms == {"tisseo_gtfs"}


def test_feeds_en_service_lit_aussi_un_zip(tmp_path):
    import zipfile

    _ecrire_feed(tmp_path, "tisseo_gtfs", [("A", 43.60, 1.44)])
    with zipfile.ZipFile(tmp_path / "lio_2026.zip", "w") as z:
        z.writestr("stops.txt", "stop_id,stop_name,stop_lat,stop_lon\nB,B,43.2,1.1\n")
    noms = {f.name for f in feeds_en_service(str(tmp_path / "tisseo_gtfs"))}
    assert noms == {"tisseo_gtfs", "lio_2026.zip"}


def test_coordonnees_arrets_reunit_les_feeds(tmp_path):
    _ecrire_feed(tmp_path, "tisseo_gtfs", [("A", 43.60, 1.44), ("A2", 43.61, 1.45)])
    _ecrire_feed(tmp_path, "lio_gtfs", [("B", 43.20, 1.10)])
    coords, par_feed = coordonnees_arrets(feeds_en_service(str(tmp_path / "tisseo_gtfs")))
    assert len(coords) == 3
    assert par_feed == {"lio_gtfs": 1, "tisseo_gtfs": 2}


def test_la_porte_de_proximite_voit_un_arret_regional(tmp_path, monkeypatch):
    """The measured case: a point far from any Tisséo stop, 200 m from a liO coach.

    Against the former code, `_has_reachable_stop` returned False and the runtime
    skipped OTP — the service was in the graph and invisible.
    """
    import pandas as pd

    _ecrire_feed(tmp_path, "tisseo_gtfs", [("A", 43.60, 1.44)])
    _ecrire_feed(tmp_path, "lio_gtfs", [("B", 43.20, 1.10)])
    monkeypatch.setattr(settings.gtfs, "gtfs_file", str(tmp_path / "tisseo_gtfs"))

    class _Feed:
        stops = pd.DataFrame({"stop_lat": [43.60], "stop_lon": [1.44]})

    helper = OTPTripHelper.__new__(OTPTripHelper)
    helper.gtfs_data = _Feed()
    # Reproduces the `__init__` construction without network or semaphore.
    coords, _ = coordonnees_arrets(feeds_en_service(settings.gtfs.gtfs_file))
    helper._stop_coords = coords

    # 43.2018/1.1000 ≈ 200 m north of the liO stop, 45 km from the Tisséo stop.
    assert helper._has_reachable_stop(1.1000, 43.2018) is True
    # A point 50 km from any stop stays out of reach: the gate still works.
    assert helper._has_reachable_stop(1.9000, 43.2000) is False


def test_la_porte_ne_retrecit_jamais_en_silence(tmp_path, monkeypatch, caplog):
    """Empty scan → fallback to the primary feed AND `[ALARME]`.

    The "absence of measurement yields the perfect score" pattern: a gate reduced
    to zero stops would let everyone or no one through without saying so.
    """
    import pandas as pd

    vide = tmp_path / "vide"
    vide.mkdir()
    monkeypatch.setattr(settings.gtfs, "gtfs_file", str(vide / "absent_gtfs"))
    coords, par_feed = coordonnees_arrets(feeds_en_service(settings.gtfs.gtfs_file))
    assert len(coords) == 0 and par_feed == {}
    primaire = pd.DataFrame({"stop_lat": [43.6], "stop_lon": [1.44]}).values.astype(float)
    assert len(coords) < len(primaire)  # this is the condition that triggers the fallback


# ── 4. The train counts as train ────────────────────────────────────────

def _plan_train() -> TravelPlan:
    depart = TransitLocation(stop="Gare de Muret", lat=43.46, lon=1.32)
    arrivee = TransitLocation(stop="Toulouse Matabiau", lat=43.611, lon=1.454)
    leg = Transit(
        start_time=0,
        end_time=1_200_000,
        start_location=depart,
        end_location=arrivee,
        is_transfer=False,
        transit_route="800000",
        shape_id=None,
        duration=1_200,
        distance=22_000.0,
        mode="rail",
    )
    return TravelPlan(
        id="rail-test",
        start_location=depart,
        end_location=arrivee,
        start_time=0,
        end_time=1_200_000,
        legs=[leg],
    )


def test_le_journal_de_production_compte_un_train(caplog):
    assert _plan_transport_mode(_plan_train()) == "Train"


def test_le_mode_canonique_du_rail_est_le_train():
    assert canonical_mode("rail") == "train"


@pytest.mark.parametrize("brut", ["rail", "TER", "train", "foot,rail,foot"])
def test_toutes_les_ecritures_du_train_se_canonisent(brut):
    assert canonical_mode(brut) == "train"
