"""The route catalogue gathers the three networks in service, or says it does not.

The GTFS reader loads only one feed (Tisséo) while the OTP graph has carried three since
2026-09-04. A liO or TER route identifier was therefore in no table, and the agent's
prompt read "Trajet en Unknown 392" for the 319 routes of the two regional
networks — mode and number lost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inputs.gtfs import reader as reader_module  # noqa: E402
from inputs.gtfs.reader import GTFSData  # noqa: E402


def _ecrire_feed(dossier: Path, routes: list[dict], avec_stops: bool = True) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(routes).to_csv(dossier / "routes.txt", index=False)
    if avec_stops:
        pd.DataFrame([{"stop_id": "s1", "stop_name": "A", "stop_lat": 43.6, "stop_lon": 1.44}]
                     ).to_csv(dossier / "stops.txt", index=False)
    return dossier


def _gtfs_primaire(routes: list[dict]) -> GTFSData:
    """A minimal GTFSData of which only the routes table matters for these tests."""
    vide = pd.DataFrame()
    g = GTFSData.__new__(GTFSData)
    g.routes = pd.DataFrame(routes)
    g.trips = vide
    g.stop_times = vide
    g.stops = vide
    g.shapes = vide
    g.calendar = vide
    g.calendar_dates = vide
    return g


@pytest.fixture
def trois_feeds(tmp_path, monkeypatch):
    racine = tmp_path / "gtfs"
    primaire = _ecrire_feed(racine / "tisseo_gtfs", [
        {"route_id": "T1", "route_short_name": "1", "route_long_name": "Bus 1", "route_type": "3"},
    ])
    _ecrire_feed(racine / "lio_gtfs", [
        {"route_id": "L392", "route_short_name": "392", "route_long_name": "Car 392", "route_type": "3"},
        {"route_id": "L1", "route_short_name": "1", "route_long_name": "Car 1 (nom court déjà pris)", "route_type": "3"},
    ])
    _ecrire_feed(racine / "ter_gtfs", [
        {"route_id": "R_K1", "route_short_name": "K1", "route_long_name": "Matabiau — Brive", "route_type": "2"},
    ])
    monkeypatch.setattr(reader_module.settings.gtfs, "gtfs_file", str(primaire), raising=False)
    return primaire


def test_les_lignes_des_autres_feeds_entrent_avec_leur_mode(trois_feeds):
    g = _gtfs_primaire([{"route_id": "T1", "route_short_name": "1",
                         "route_long_name": "Bus 1", "route_type": "3"}])
    g.init_route_lookup_maps()
    # The regional coach and the train are named, and their mode is the one from the modes table.
    assert g.route_id_map["L392"]["route_short_name"] == "392"
    assert g.route_id_map["L392"]["route_type"] == "Bus"
    assert g.route_id_map["R_K1"]["route_short_name"] == "K1"
    assert g.route_id_map["R_K1"]["route_type"] == "Train"
    # No "Unknown" left: that was the defect to close.
    assert all(v["route_type"] != "Unknown" for v in g.route_id_map.values())


def test_un_nom_court_deja_pris_reste_au_feed_primaire(trois_feeds):
    g = _gtfs_primaire([{"route_id": "T1", "route_short_name": "1",
                         "route_long_name": "Bus 1", "route_type": "3"}])
    g.init_route_lookup_maps()
    # "1" exists in both networks: the primary feed keeps the short name, and the regional
    # route remains reachable by its identifier.
    assert g.route_name_id_map["1"] == "T1"
    assert "L1" in g.route_id_map


def test_un_identifiant_revendique_deux_fois_s_alarme(tmp_path, monkeypatch, caplog):
    racine = tmp_path / "gtfs"
    primaire = _ecrire_feed(racine / "tisseo_gtfs", [
        {"route_id": "DOUBLON", "route_short_name": "1", "route_long_name": "Bus 1", "route_type": "3"},
    ])
    _ecrire_feed(racine / "autre_gtfs", [
        {"route_id": "DOUBLON", "route_short_name": "9", "route_long_name": "Autre", "route_type": "2"},
    ])
    monkeypatch.setattr(reader_module.settings.gtfs, "gtfs_file", str(primaire), raising=False)
    g = _gtfs_primaire([{"route_id": "DOUBLON", "route_short_name": "1",
                         "route_long_name": "Bus 1", "route_type": "3"}])
    journal = g.init_route_lookup_maps()
    assert journal["collisions_identifiant"] == 1
    # The primary feed wins: the displayed mode stays its own, not the intruder's.
    assert g.route_id_map["DOUBLON"]["route_type"] == "Bus"


def test_un_feed_sans_catalogue_de_lignes_n_est_pas_une_erreur(tmp_path, monkeypatch):
    racine = tmp_path / "gtfs"
    primaire = _ecrire_feed(racine / "tisseo_gtfs", [
        {"route_id": "T1", "route_short_name": "1", "route_long_name": "Bus 1", "route_type": "3"},
    ])
    sans_routes = racine / "arrets_seuls"
    sans_routes.mkdir()
    pd.DataFrame([{"stop_id": "s9", "stop_name": "B", "stop_lat": 43.7, "stop_lon": 1.5}]
                 ).to_csv(sans_routes / "stops.txt", index=False)
    monkeypatch.setattr(reader_module.settings.gtfs, "gtfs_file", str(primaire), raising=False)
    g = _gtfs_primaire([{"route_id": "T1", "route_short_name": "1",
                         "route_long_name": "Bus 1", "route_type": "3"}])
    journal = g.init_route_lookup_maps()
    assert journal["lignes_ajoutees"] == 0
    assert g.route_id_map["T1"]["route_short_name"] == "1"
