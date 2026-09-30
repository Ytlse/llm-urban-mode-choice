"""The shape table comes from the recipe, or the runtime raises an alarm.

THE TARGETED DEFECT
-------------------
To board an agent onto a vehicle, `Inhabitant.gaml` compares the
vehicle's `shape_id` with the list Python put on the itinerary
leg. That list comes from `GTFSData.get_shape_id_from_route_info`, which
read a table built from the **primary feed alone** (Tisséo) whereas
the layers and the vehicle trips carry **three** networks since 2026-09-04.
Measured that day: **80 of the 199 lines** of the vehicle-trip file (17 TER,
58 liO coaches, 5 circular lines) and **2,277 vehicle trips** ran in GAMA
without any itinerary being able to designate them — the function returned `[]`,
**indistinguishable** from "no shape for this pair of stops".

Each test bears on a decision which, taken the wrong way round, produces a plausible
silence:

  * side file missing → **[ALARME]**, table marked partial; without this cry,
    the failure looks like legitimately missing data;
  * a line WITHOUT published geometry (the TER) becomes reachable **because** the
    recipe published the `shape_id`s it made — the runtime remakes
    none of them;
  * a freshness witness that moved (the layers remade alone) → refusal and
    alarm: the table would no longer designate the shapes GAMA draws;
  * counters that do not match the content → refusal: a truncated
    file has the right fingerprints of its siblings;
  * a non-designatable line raises its alarm **once**, not at every itinerary;
  * OTP's line identifier is cross-checked with the catalogue: `lio:305` is
    line `305`, not an unknown line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inputs.gtfs import reader as reader_module  # noqa: E402
from inputs.gtfs import table_traces  # noqa: E402
from inputs.gtfs.reader import GTFSData  # noqa: E402

# A TER shape as the RECIPE names it: `<route_id>:<direction>:<fingerprint>`.
# No feed publishes this identifier; only the recipe knows how to make it, and
# that is the whole point of the side file.
SHAPE_TER = "FR:Line::ABC:1:7955fdab"
ROUTE_TER = "FR:Line::ABC:"
GARE_A, GARE_B = "StopPoint:OCETrain TER-87611830", "StopPoint:OCETrain TER-87611467"


def _feed_tisseo(dossier: Path) -> Path:
    """The primary feed: one bus line, its geometry, two stops."""
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"route_id": "line:8", "route_short_name": "8",
                   "route_long_name": "Bus 8", "route_type": "3"}]
                 ).to_csv(dossier / "routes.txt", index=False)
    pd.DataFrame([{"stop_id": "stop_point:SP_1", "stop_name": "Arènes",
                   "stop_lat": 43.5940, "stop_lon": 1.4165, "location_type": 0},
                  {"stop_id": "stop_point:SP_2", "stop_name": "Patte d'Oie",
                   "stop_lat": 43.5990, "stop_lon": 1.4210, "location_type": 0}]
                 ).to_csv(dossier / "stops.txt", index=False)
    pd.DataFrame([{"route_id": "line:8", "service_id": "SVC_1", "trip_id": "t1",
                   "direction_id": 0, "shape_id": "14852"}]
                 ).to_csv(dossier / "trips.txt", index=False)
    pd.DataFrame([{"trip_id": "t1", "stop_sequence": 0, "stop_id": "stop_point:SP_1",
                   "arrival_time": "07:00:00", "departure_time": "07:00:00",
                   "shape_dist_traveled": 0.0},
                  {"trip_id": "t1", "stop_sequence": 1, "stop_id": "stop_point:SP_2",
                   "arrival_time": "07:05:00", "departure_time": "07:05:00",
                   "shape_dist_traveled": 700.0}]
                 ).to_csv(dossier / "stop_times.txt", index=False)
    pd.DataFrame([{"shape_id": "14852", "shape_pt_lat": 43.5940, "shape_pt_lon": 1.4165,
                   "shape_pt_sequence": 0, "shape_dist_traveled": 0.0},
                  {"shape_id": "14852", "shape_pt_lat": 43.5990, "shape_pt_lon": 1.4210,
                   "shape_pt_sequence": 1, "shape_dist_traveled": 700.0}]
                 ).to_csv(dossier / "shapes.txt", index=False)
    pd.DataFrame([{"service_id": "SVC_1", "date": "20260316", "exception_type": 1}]
                 ).to_csv(dossier / "calendar_dates.txt", index=False)
    (dossier / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n", encoding="utf-8")
    return dossier


def _feed_ter(dossier: Path) -> Path:
    """A TER-style network: `shapes.txt` reduced to its header.

    It serves the line CATALOGUE (joined by `init_route_lookup_maps`); its
    geometry, however, does not exist — hence the side file.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"route_id": ROUTE_TER, "route_short_name": "P2",
                   "route_long_name": "L'Isle-Jourdain — Toulouse", "route_type": "2"}]
                 ).to_csv(dossier / "routes.txt", index=False)
    pd.DataFrame([{"stop_id": GARE_A, "stop_name": "Pibrac",
                   "stop_lat": 43.62133, "stop_lon": 1.28912, "location_type": 0},
                  {"stop_id": GARE_B, "stop_name": "Colomiers",
                   "stop_lat": 43.60373, "stop_lon": 1.33422, "location_type": 0}]
                 ).to_csv(dossier / "stops.txt", index=False)
    (dossier / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n",
        encoding="utf-8")
    return dossier


def _couches(dossier: Path) -> dict[str, str]:
    """The two siblings whose fingerprint the side file records."""
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "routes.shp").write_bytes(b"couche des traces, generation 1")
    (dossier / "routes.dbf").write_bytes(b"attributs des traces, generation 1")
    (dossier / "trip_info.json").write_text('{"trip_list": []}', encoding="utf-8")
    return {"routes.shp": "routes.shp", "routes.dbf": "routes.dbf",
            "trip_info.json": "trip_info.json"}


def _table_de_reference() -> dict:
    return {
        ROUTE_TER: {SHAPE_TER: {GARE_A: 3, GARE_B: 5}},
        "line:8": {"14852": {"stop_point:SP_1": 0, "stop_point:SP_2": 1}},
    }


def _ecrire_annexe(dossier: Path, table=None, arrets=None) -> Path:
    document = table_traces.construire(
        table=table if table is not None else _table_de_reference(),
        arrets=arrets if arrets is not None else {
            GARE_A: {"stop_name": "Pibrac", "stop_lat": 43.62133, "stop_lon": 1.28912},
            GARE_B: {"stop_name": "Colomiers", "stop_lat": 43.60373, "stop_lon": 1.33422},
        },
        dossier_temoins=dossier,
        noms_temoins=("routes.shp", "routes.dbf", "trip_info.json"),
        genere_le="2026-09-04T12:00:00",
        recette="scripts/data/gama/export_trip_info.py",
        reseaux={"ter": {"courses_retenues": 884}},
    )
    chemin = dossier / table_traces.NOM_FICHIER
    table_traces.ecrire(chemin, document)
    return chemin


@pytest.fixture
def alarmes():
    """Captures loguru ERRORs — `caplog` does not see them (no propagation)."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    yield messages
    logger.remove(sink)


@pytest.fixture
def journal():
    """Captures loguru INFOs: success must be logged explicitly."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    yield messages
    logger.remove(sink)


@pytest.fixture
def monde(tmp_path, monkeypatch):
    """A primary feed, a TER feed next to it, layers, and the side table."""
    racine = tmp_path / "gtfs"
    primaire = _feed_tisseo(racine / "tisseo_gtfs")
    _feed_ter(racine / "ter_gtfs")
    includes = tmp_path / "includes"
    _couches(includes)
    annexe = _ecrire_annexe(includes)
    monkeypatch.setattr(reader_module.settings.gtfs, "gtfs_file", str(primaire),
                        raising=False)
    monkeypatch.setattr(reader_module.settings.gtfs, "shape_lookup_file", str(annexe),
                        raising=False)
    return {"primaire": primaire, "includes": includes, "annexe": annexe}


# ── The line without geometry becomes reachable ──────────────────────────────

class TestLigneFerroviaireJoignable:
    def test_le_ter_est_desormais_designable(self, monde):
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.source_table_traces == GTFSData.SOURCE_ANNEXE
        assert g.table_traces_partielle is False
        assert g.get_shape_id_from_route_info(ROUTE_TER, GARE_A, GARE_B) == [SHAPE_TER]

    def test_sans_le_fichier_annexe_la_meme_ligne_rend_une_liste_vide(self, monde):
        """The state before the fix, reproduced: the primary feed alone."""
        monde["annexe"].unlink()
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.get_shape_id_from_route_info(ROUTE_TER, GARE_A, GARE_B) == []
        # And the primary feed's bus stays reachable: the fallback is not a breakdown.
        assert g.get_shape_id_from_route_info(
            "line:8", "stop_point:SP_1", "stop_point:SP_2") == ["14852"]

    def test_le_sens_du_couple_d_arrets_est_respecte(self, monde):
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.get_shape_id_from_route_info(ROUTE_TER, GARE_B, GARE_A) == []

    def test_l_identifiant_du_trace_est_celui_de_la_recette(self, monde):
        """The runtime makes no `shape_id`: it returns the published one.

        This is the invariant that prevents a second implementation of the rule
        `<route_id>:<direction>:<fingerprint>` from existing in the repo.
        """
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        publies = {s for par_trace in json.loads(monde["annexe"].read_text())["table"].values()
                   for s in par_trace}
        rendus = set(g.get_shape_id_from_route_info(ROUTE_TER, GARE_A, GARE_B))
        assert rendus <= publies


# ── The alarms ───────────────────────────────────────────────────────────────

class TestAlarmes:
    def test_fichier_absent_alarme_et_marque_la_table_partielle(self, monde, alarmes):
        monde["annexe"].unlink()
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.table_traces_partielle is True
        assert g.source_table_traces == "feed_primaire:absente"
        assert any("[ALARME]" in m and "absente" in m for m in alarmes), alarmes

    def test_temoin_qui_a_bouge_refuse_la_table(self, monde, alarmes):
        """`make gama-layers` alone remakes `routes.shp`: the pair is mismatched."""
        (monde["includes"] / "routes.shp").write_bytes(b"couche des traces, generation 2")
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.source_table_traces == "feed_primaire:depareillee"
        assert g.get_shape_id_from_route_info(ROUTE_TER, GARE_A, GARE_B) == []
        assert any("[ALARME]" in m and "depareillee" in m for m in alarmes), alarmes

    def test_temoin_disparu_refuse_la_table(self, monde, alarmes):
        (monde["includes"] / "trip_info.json").unlink()
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.source_table_traces == "feed_primaire:temoin_absent"
        assert any("[ALARME]" in m for m in alarmes)

    def test_compteurs_qui_ne_correspondent_pas_refusent_la_table(self, monde, alarmes):
        document = json.loads(monde["annexe"].read_text())
        document["table"].pop("line:8")  # truncated: the fingerprints stay right
        monde["annexe"].write_text(json.dumps(document), encoding="utf-8")
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.source_table_traces == "feed_primaire:comptes"
        assert any("[ALARME]" in m for m in alarmes)

    def test_format_inconnu_refuse_la_table(self, monde, alarmes):
        document = json.loads(monde["annexe"].read_text())
        document["format"] = table_traces.FORMAT + 1
        monde["annexe"].write_text(json.dumps(document), encoding="utf-8")
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.source_table_traces == "feed_primaire:format"
        assert any("[ALARME]" in m for m in alarmes)

    def test_fichier_illisible_refuse_la_table(self, monde, alarmes):
        monde["annexe"].write_text("{ ceci n'est pas du json", encoding="utf-8")
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.source_table_traces == "feed_primaire:illisible"
        assert any("[ALARME]" in m for m in alarmes)

    def test_ligne_indesignable_alarme_une_seule_fois(self, monde, alarmes):
        """Rising edge: the cause is single, the itineraries are many."""
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        for _ in range(5):
            assert g.get_shape_id_from_route_info("inconnue", GARE_A, GARE_B) == []
        vues = [m for m in alarmes if "indésignable" in m]
        assert len(vues) == 1, vues
        assert g.resume_table_traces()["appels_ligne_absente"] == 5
        assert g.resume_table_traces()["lignes_indesignables"] == ["inconnue"]

    def test_jambe_sans_arret_resolu_alarme_aussi(self, monde, alarmes):
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert g.get_shape_id_from_route_info(ROUTE_TER, None, GARE_B) == []
        assert any("arret non resolu" in m for m in alarmes), alarmes
        assert g.resume_table_traces()["appels_arret_non_resolu"] == 1

    def test_le_succes_se_journalise(self, monde, journal):
        """A runtime silent when all is well cannot tell "read" from "nothing read"."""
        GTFSData.from_gtfs_files(str(monde["primaire"]))
        assert any("table des tracés lue" in m for m in journal), journal


# ── The stop catalogue, and the world extent that must not move ──────────────

class TestArretsHorsFeedPrimaire:
    def test_la_gare_ter_se_resout_par_le_catalogue_annexe(self, monde):
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        gare = g.get_stop(GARE_A)
        assert gare.stop_name == "Pibrac"
        assert gare.stop_lat == pytest.approx(43.62133)

    def test_le_catalogue_annexe_n_elargit_pas_l_emprise_du_monde(self, monde):
        """`get_bounding_box` sets the extent of the GAMA world.

        Pouring stations from all of Occitanie into it would stretch it far beyond the
        survey scope — hence a catalogue kept apart from `self.stops`.
        """
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        min_lon, min_lat, max_lon, max_lat = g.get_bounding_box()
        assert (min_lon, max_lon) == pytest.approx((1.4165, 1.4210))
        assert (min_lat, max_lat) == pytest.approx((43.5940, 43.5990))

    def test_un_arret_vraiment_inconnu_leve_toujours(self, monde):
        g = GTFSData.from_gtfs_files(str(monde["primaire"]))
        with pytest.raises(ValueError):
            g.get_stop("StopPoint:jamais-vu")


# ── The line identifier returned by OTP ──────────────────────────────────────

class TestResolutionDesIdentifiantsOTP:
    """OTP prefixes with the feed name. The rule "strip the first segment if there are
    at least two `:`" worked for Tisséo and the TER, and **failed** for the
    liO coach, whose `route_id` contains no `:`: `lio:305` went through
    intact and matched nothing — neither in the shape table, nor in
    `ROUTE_VEHICLE_MAP` on the GAMA side."""

    @pytest.fixture
    def catalogue(self, monde):
        return GTFSData.from_gtfs_files(str(monde["primaire"]))

    def test_le_car_lio_est_reconnu(self, catalogue):
        catalogue.route_id_map["305"] = {"route_short_name": "305",
                                         "route_long_name": "Car 305",
                                         "route_type": "Bus"}
        assert catalogue.resoudre_route_id("lio:305") == "305"

    def test_le_bus_tisseo_est_reconnu(self, catalogue):
        assert catalogue.resoudre_route_id("tisseo:line:8") == "line:8"

    def test_le_ter_est_reconnu(self, catalogue):
        assert catalogue.resoudre_route_id(f"ter:{ROUTE_TER}") == ROUTE_TER

    def test_un_identifiant_deja_propre_est_rendu_tel_quel(self, catalogue):
        assert catalogue.resoudre_route_id("line:8") == "line:8"

    def test_un_identifiant_inconnu_garde_l_ancien_comportement(self, catalogue):
        assert catalogue.resoudre_route_id("feed:jamais:vu") == "jamais:vu"
