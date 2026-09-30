"""Tests of the two recipes that feed `services/GAMA/CityTransport/includes/`
(`scripts/data/gama/export_gtfs_layers.py` and `export_trip_info.py`).

THE REGRESSION TARGETED
-----------------------
On 2026-09-04, `routes.shp` carried **the three networks** — 730 shapes, 34 of them with
`route_type=2` — and `trip_info.json`, produced by hand five months earlier, **only
Tisséo**: 39,343 trips, none with `route_type=2`. GAMA therefore drew
34 TER lines and 68 stations where **no train would ever run**. Nothing said so:
a visible, dead line reads like a line with no service.

These tests each bear on a decision which, taken the wrong way round, produces a
**plausible but wrong** file:

  * a `route_type` drawn in the layer but with no trip → the recipe FAILS, and
    `trip_info.json` is not written (the five-month defect);
  * a drawn `route_type` none of whose trips runs ON THE SIMULATED DAY → same refusal:
    carrying trips in June does not make a train run in March;
  * the simulated date outside the window → refusal, where GAMA settles for a `warn`
    and no longer schedules anything;
  * a window wider than the model's 64-bit binary mask → refusal;
  * two networks that number their services alike (`SVC_0001`: **224
    collisions** measured between the TER and liO feeds) → the calendars stay
    separate, the coaches do not read the train timetable;
  * a `shapes.txt` reduced to its header is not a published geometry — testing
    it by its mere existence yielded zero shapes for the TER, without a word;
  * a network without geometry gets **one shape per distinct stop sequence**, and
    not one per (line, direction): `build_trips` forces the last segment up to the
    last point of the shape, so a Toulouse → Tarbes trip laid on the
    Toulouse → Pau shape would run all the way to Pau;
  * a trip without `direction_id` does not vanish (6 TER trips evaporated,
    their `shape_id` being `NaN`);
  * the two recipes build the SAME `shape_id`s — that is what keeps the
    layer and the trips from silently diverging.

Minimal synthetic feeds: no network access, no large file, no
dependency on the repository data.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.gama import export_trip_info as recette  # noqa: E402
from scripts.data.gama import gtfs_traces  # noqa: E402

# Dates of the test window. The simulated day is the first one.
JOUR_SIMULE = "2026-03-16"
D0, D1, D2 = "20260316", "20260317", "20260318"

# A corner of the Toulouse map, so that lengths in metres are realistic.
LON0, LAT0 = 1.44, 43.60


def _ecrire(chemin: Path, colonnes: list[str], lignes: list[dict]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as fh:
        ecrivain = csv.DictWriter(fh, fieldnames=colonnes, extrasaction="ignore")
        ecrivain.writeheader()
        for ligne in lignes:
            ecrivain.writerow({c: ligne.get(c, "") for c in colonnes})


def _arrets(prefixe: str, noms: list[str]) -> list[dict]:
    """Stops lined up from west to east, about 2 km from one another."""
    return [
        {"stop_id": f"{prefixe}{nom}", "stop_name": f"Arrêt {prefixe}{nom}",
         "stop_lat": f"{LAT0}", "stop_lon": f"{LON0 + 0.025 * i}", "location_type": "0"}
        for i, nom in enumerate(noms)
    ]


def feed_avec_geometrie(dossier: Path, *, route_type: str = "3", dates=(D0, D1, D2),
                        service: str = "SVC_0001") -> Path:
    """A network that publishes its `shapes.txt`: two trips on a 5-point shape."""
    dossier.mkdir(parents=True, exist_ok=True)
    stops = _arrets("B", ["1", "2", "3"])
    _ecrire(dossier / "stops.txt",
            ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"], stops)
    _ecrire(dossier / "routes.txt",
            ["route_id", "route_short_name", "route_long_name", "route_type",
             "route_color", "route_text_color"],
            [{"route_id": "B1", "route_short_name": "B1", "route_long_name": "Ligne bus",
              "route_type": route_type, "route_color": "112233", "route_text_color": "FFFFFF"}])
    # The shape has 5 points; the stops fall on points 0, 2 and 4.
    points = [(LON0 + 0.0125 * i, LAT0) for i in range(5)]
    cumule = recette._cumul(points)
    _ecrire(dossier / "shapes.txt",
            ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence",
             "shape_dist_traveled"],
            [{"shape_id": "SH1", "shape_pt_lat": f"{lat}", "shape_pt_lon": f"{lon}",
              "shape_pt_sequence": str(i), "shape_dist_traveled": f"{cumule[i]}"}
             for i, (lon, lat) in enumerate(points)])
    _ecrire(dossier / "trips.txt",
            ["route_id", "service_id", "trip_id", "direction_id", "shape_id"],
            [{"route_id": "B1", "service_id": service, "trip_id": "bus_1",
              "direction_id": "0", "shape_id": "SH1"},
             {"route_id": "B1", "service_id": service, "trip_id": "bus_2",
              "direction_id": "0", "shape_id": "SH1"}])
    horaires = []
    for trip, heure in (("bus_1", 7), ("bus_2", 8)):
        for rang, (arret, point) in enumerate(zip(stops, (0, 2, 4))):
            horaires.append({"trip_id": trip, "stop_sequence": str(rang),
                             "stop_id": arret["stop_id"],
                             "arrival_time": f"{heure:02d}:{rang * 10:02d}:00",
                             "departure_time": f"{heure:02d}:{rang * 10:02d}:00",
                             "shape_dist_traveled": f"{cumule[point]}"})
    _ecrire(dossier / "stop_times.txt",
            ["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time",
             "shape_dist_traveled"], horaires)
    _ecrire(dossier / "calendar_dates.txt", ["service_id", "date", "exception_type"],
            [{"service_id": service, "date": d, "exception_type": "1"} for d in dates])
    (dossier / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n", encoding="utf-8")
    return dossier


def feed_sans_geometrie(dossier: Path, *, route_type: str = "2", dates=(D0,),
                        service: str = "SVC_0001") -> Path:
    """A TER-style network: `shapes.txt` reduced to its header.

    Three trips on the same line:
      * `rail_complet` serves R1 R2 R3 R4;
      * `rail_partiel` stops at R2 — it is the one that, laid on the shape of the
        full trip, would run all the way to R4;
      * `rail_sans_sens` has no `direction_id`.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    stops = _arrets("R", ["1", "2", "3", "4"])
    _ecrire(dossier / "stops.txt",
            ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"], stops)
    _ecrire(dossier / "routes.txt",
            ["route_id", "route_short_name", "route_long_name", "route_type",
             "route_color", "route_text_color"],
            [{"route_id": "R1", "route_short_name": "TER1", "route_long_name": "Ligne TER",
              "route_type": route_type, "route_color": "AA00AA", "route_text_color": "FFFFFF"}])
    # The header ALONE: the TER feed publishes one like this.
    (dossier / "shapes.txt").write_text(
        "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n",
        encoding="utf-8")
    _ecrire(dossier / "trips.txt",
            ["route_id", "service_id", "trip_id", "direction_id", "shape_id"],
            [{"route_id": "R1", "service_id": service, "trip_id": "rail_complet",
              "direction_id": "0", "shape_id": ""},
             {"route_id": "R1", "service_id": service, "trip_id": "rail_partiel",
              "direction_id": "0", "shape_id": ""},
             {"route_id": "R1", "service_id": service, "trip_id": "rail_sans_sens",
              "direction_id": "", "shape_id": ""}])
    dessertes = {"rail_complet": ["R1", "R2", "R3", "R4"],
                 "rail_partiel": ["R1", "R2"],
                 "rail_sans_sens": ["R4", "R3", "R1"]}
    horaires = []
    for heure, (trip, suite) in enumerate(dessertes.items(), start=6):
        for rang, arret in enumerate(suite):
            horaires.append({"trip_id": trip, "stop_sequence": str(rang), "stop_id": arret,
                             "arrival_time": f"{heure:02d}:{rang * 15:02d}:00",
                             "departure_time": f"{heure:02d}:{rang * 15:02d}:00",
                             "shape_dist_traveled": ""})
    _ecrire(dossier / "stop_times.txt",
            ["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time",
             "shape_dist_traveled"], horaires)
    _ecrire(dossier / "calendar_dates.txt", ["service_id", "date", "exception_type"],
            [{"service_id": service, "date": d, "exception_type": "1"} for d in dates])
    (dossier / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\n", encoding="utf-8")
    return dossier


def couche_depuis_feeds(feeds: dict[str, Path], chemin: Path,
                        garder=lambda shape_id, route_type: True) -> Path:
    """Builds a test `routes.shp`, with the `shape_id`s of the shared module.

    `garder` allows removing shapes from it: that is how we reproduce a
    layer that carries a `route_type` the trips ignore, or the reverse.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    lignes = []
    for reseau, feed in feeds.items():
        stops = {l["stop_id"]: (float(l["stop_lon"]), float(l["stop_lat"]))
                 for l in recette._lire_csv(feed / "stops.txt")}
        types = {l["route_id"]: l["route_type"]
                 for l in recette._lire_csv(feed / "routes.txt")}
        trips = recette._lire_csv(feed / "trips.txt")
        horaires = recette._lire_csv(feed / "stop_times.txt")
        if recette._a_des_geometries(feed):
            points: dict[str, list] = {}
            for l in recette._lire_csv(feed / "shapes.txt"):
                points.setdefault(l["shape_id"], []).append(
                    (int(l["shape_pt_sequence"]), float(l["shape_pt_lon"]),
                     float(l["shape_pt_lat"])))
            traces = {sid: [(lo, la) for _, lo, la in sorted(v)] for sid, v in points.items()}
            route_de = {l["shape_id"]: l["route_id"] for l in trips}
        else:
            suites = gtfs_traces.suites_depuis_stop_times(horaires)
            motifs, course_vers_trace, _ = gtfs_traces.traces_par_suite_d_arrets(
                trips, suites, journal=lambda *_a, **_k: None)
            traces = {sid: [stops[s] for s in suite] for sid, suite in motifs.items()}
            route_de = {course_vers_trace[l["trip_id"]]: l["route_id"]
                        for l in trips if l["trip_id"] in course_vers_trace}
        for shape_id, pts in traces.items():
            route_type = types[route_de[shape_id]]
            if not garder(shape_id, route_type):
                continue
            lignes.append({"shape_id": shape_id, "route_id": route_de[shape_id],
                           "route_type": route_type, "reseau": reseau,
                           "geometry": LineString(pts)})
    chemin.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(lignes, crs="EPSG:4326").to_file(chemin)
    return chemin


class BaseTemporaire(unittest.TestCase):
    def setUp(self):
        self.racine = Path(tempfile.mkdtemp(prefix="gama_includes_"))
        self.addCleanup(shutil.rmtree, self.racine, ignore_errors=True)
        self.feeds = {
            "bus": feed_avec_geometrie(self.racine / "feed_bus"),
            "rail": feed_sans_geometrie(self.racine / "feed_rail"),
        }
        self.sortie = self.racine / "trip_info.json"

    def lancer(self, *, couche: Path, jours: int = 64, date_simulee: str = JOUR_SIMULE,
               debut: str | None = None, sortie: Path | None = None) -> int:
        argv = ["--routes", str(couche), "--sortie", str(sortie or self.sortie),
                "--date-simulee", date_simulee, "--jours", str(jours)]
        if debut:
            argv += ["--debut", debut]
        for nom, chemin in self.feeds.items():
            argv += ["--feed", f"{nom}={chemin}"]
        return recette.main(argv)

    def produit(self, sortie: Path | None = None) -> dict:
        with open(sortie or self.sortie, encoding="utf-8") as fh:
            return json.load(fh)


class TestCoherenceCoucheCourses(BaseTemporaire):
    """The five-month drift: a layer and trips that do not talk about the
    same line types."""

    def test_type_trace_sans_course_fait_echouer_la_recette(self):
        # The layer carries rail; the trips do not — exactly the state of
        # 2026-09-04 (34 TER shapes, zero trips of type 2).
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.feeds.pop("rail")
        code = self.lancer(couche=couche)
        self.assertEqual(code, recette.CODE_REFUS)
        self.assertFalse(self.sortie.exists(),
                         "trip_info.json ne doit PAS être écrit quand un type tracé "
                         "n'a aucune course")

    def test_type_trace_sans_course_LE_JOUR_SIMULE_fait_echouer(self):
        # Rail only runs on the 18th; the simulated day is the 16th. The file
        # would carry type-2 trips — a non-zero count, hence reassuring —
        # none of which would run on the day of the simulation.
        self.feeds["rail"] = feed_sans_geometrie(self.racine / "feed_rail_tardif",
                                                 dates=(D2,), service="SVC_0009")
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        code = self.lancer(couche=couche)
        self.assertEqual(code, recette.CODE_REFUS)
        self.assertFalse(self.sortie.exists())

    def test_cas_nominal_les_deux_types_roulent_le_jour_simule(self):
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche), 0)
        produit = self.produit()
        du_jour = recette.courses_du_jour(produit, D0.replace("-", ""))
        self.assertEqual(set(du_jour), {"2", "3"},
                         "les deux types de ligne de la couche doivent rouler le jour simulé")
        self.assertEqual(du_jour["2"], 3)
        self.assertEqual(du_jour["3"], 2)

    def test_course_dont_le_trace_manque_a_la_couche_est_ecartee(self):
        # A `shape_id` missing from the layer makes `route first_with (...)` nil in
        # `PublicTransport.gaml`: the vehicle would be born without geometry.
        couche = couche_depuis_feeds(
            self.feeds, self.racine / "couche.shp",
            garder=lambda sid, rt: not sid.endswith(
                gtfs_traces.empreinte_suite(["R1", "R2"])))
        self.assertEqual(self.lancer(couche=couche), 0)
        traces_couche = set(recette.types_de_la_couche(couche)[0])
        for trip in self.produit()["trip_list"]:
            self.assertIn(trip["shape_id"], traces_couche)
        self.assertNotIn("rail_partiel",
                         {t["trip_id"] for t in self.produit()["trip_list"]})

    def test_desaccord_de_nombre_de_points_fait_echouer(self):
        """The `shape_segments` are indices into `r.shape.points`.

        A layer in which a shape has fewer points than the feed would push
        GAMA out of the vertex list — at best an error, at worst a motionless
        vehicle.
        """
        import geopandas as gpd
        from shapely.geometry import LineString

        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        gdf = gpd.read_file(couche)
        cible = gdf.index[gdf["shape_id"] == "SH1"][0]
        gdf.loc[cible, "geometry"] = LineString(list(gdf.loc[cible, "geometry"].coords)[:3])
        tronquee = self.racine / "couche_tronquee.shp"
        gdf.to_file(tronquee)
        self.assertEqual(self.lancer(couche=tronquee), recette.CODE_REFUS)
        self.assertFalse(self.sortie.exists())


class TestFenetreEtMasqueBinaire(BaseTemporaire):
    """GAMA's calendar is a 64-bit mask, and it must contain the simulated
    date: outside the calendar, `is_trip_available_today` settles for a `warn`."""

    def test_date_simulee_hors_fenetre_refusee(self):
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        code = self.lancer(couche=couche, debut="2026-06-01", jours=64)
        self.assertEqual(code, recette.CODE_REFUS)
        self.assertFalse(self.sortie.exists())

    def test_fenetre_plus_large_que_le_masque_refusee(self):
        # The ceiling follows `recette.LIMITE_MASQUE`: it went from 64 to 366 days on 2026-09-17
        # (ticket 075, mask turned into a string of "0"/"1"). A hard-coded 65 tested the old
        # ceiling and the recipe, rightly, accepted the window.
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche, jours=recette.LIMITE_MASQUE + 1),
                         recette.CODE_REFUS)
        self.assertFalse(self.sortie.exists())

    def test_le_calendrier_produit_tient_dans_64_bits(self):
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche), 0)
        calendrier = self.produit()["calendar"]
        self.assertLessEqual(len(calendrier["dates"]), recette.LIMITE_MASQUE)
        self.assertIn(D0, calendrier["dates"])

    def test_date_simulee_lue_dans_settings_gaml(self):
        """The date is not copied into the recipe: two sources diverge, and the
        consequence of a divergence is an empty network with no error message."""
        lue = recette.date_simulee_de_settings()
        self.assertIsNotNone(lue, "starting_date doit être lisible dans Settings.gaml")
        self.assertEqual(lue.isoformat(), JOUR_SIMULE)

    def test_settings_illisible_ne_produit_pas_de_date_plausible(self):
        faux = self.racine / "Settings.gaml"
        faux.write_text("global { int x <- 1; }\n", encoding="utf-8")
        self.assertIsNone(recette.date_simulee_de_settings(faux))


class TestFusionDesReseaux(BaseTemporaire):
    def test_service_id_collisionnant_ne_melange_pas_les_calendriers(self):
        """The TER and liO feeds both number their services `SVC_0001`:
        224 collisions measured. Merged as is, the coaches would read
        the train timetable."""
        # The bus runs on the 16th, 17th and 18th; rail on the 17th only. Same `SVC_0001`.
        self.feeds["rail"] = feed_sans_geometrie(self.racine / "feed_rail_j2",
                                                 dates=(D1,), service="SVC_0001")
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(
            self.lancer(couche=couche, date_simulee="2026-03-17", debut=JOUR_SIMULE), 0)
        produit = self.produit()
        self.assertEqual(recette.courses_du_jour(produit, D0), {"3": 2},
                         "aucun train ne roule le 16 : le rail n'a de service que le 17")
        self.assertEqual(recette.courses_du_jour(produit, D1), {"2": 3, "3": 2})
        services = {t["service_id"] for t in produit["trip_list"]}
        self.assertEqual(services, {"bus:SVC_0001", "rail:SVC_0001"})

    def test_les_cles_de_jointure_avec_otp_ne_sont_jamais_renommees(self):
        """`shape_id`, `route_id`, `trip_id` are the join keys with the
        itineraries returned by OTP (`Inhabitant.gaml`: `shape_id_list contains
        each.shape_id`). Prefixing them would have an agent board into thin air."""
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche), 0)
        for trip in self.produit()["trip_list"]:
            self.assertFalse(trip["trip_id"].startswith(("bus:", "rail:")))
            self.assertFalse(trip["shape_id"].startswith(("bus:", "rail:")))
            self.assertFalse(trip["route_id"].startswith(("bus:", "rail:")))


class TestReseauSansGeometrie(BaseTemporaire):
    """The TER publishes neither a populated `shapes.txt` nor `trips.shape_id`."""

    def test_shapes_txt_reduit_a_son_entete_n_est_pas_une_geometrie(self):
        self.assertFalse(recette._a_des_geometries(self.feeds["rail"]))
        self.assertTrue(recette._a_des_geometries(self.feeds["bus"]))

    def test_un_trace_par_suite_d_arrets_distincte(self):
        trips = recette._lire_csv(self.feeds["rail"] / "trips.txt")
        suites = gtfs_traces.suites_depuis_stop_times(
            recette._lire_csv(self.feeds["rail"] / "stop_times.txt"))
        traces, course_vers_trace, mesures = gtfs_traces.traces_par_suite_d_arrets(
            trips, suites, journal=lambda *_a, **_k: None)
        self.assertEqual(mesures["courses_tracees"], 3)
        self.assertEqual(len(traces), 3, "trois dessertes distinctes, trois tracés")
        self.assertNotEqual(course_vers_trace["rail_complet"], course_vers_trace["rail_partiel"])

    def test_une_course_partielle_ne_roule_pas_jusqu_au_bout_de_la_ligne(self):
        """`build_trips` forces `shape_segments[-1]` to the last point of the shape.

        With one shape per (line, direction), the R1→R2 trip would inherit the
        R1→R2→R3→R4 shape and run all the way to R4 within R2's travel time:
        manufactured movement. With one shape per stop pattern, its last segment
        IS its last stop.
        """
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche), 0)
        points = recette.types_de_la_couche(couche)[0]
        partielle = next(t for t in self.produit()["trip_list"]
                         if t["trip_id"] == "rail_partiel")
        self.assertEqual(len(partielle["stop_times"]), 2)
        self.assertEqual(points[partielle["shape_id"]], 2,
                         "le tracé de la course partielle n'a que ses deux arrêts")
        self.assertEqual(partielle["shape_segments"], [1])

    def test_course_sans_direction_id_ne_disparait_pas(self):
        """6 TER trips have no `direction_id`; their `shape_id` was `NaN`,
        and the grouping dropped them without a word."""
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche), 0)
        identifiants = {t["trip_id"] for t in self.produit()["trip_list"]}
        self.assertIn("rail_sans_sens", identifiants)
        shape = next(t["shape_id"] for t in self.produit()["trip_list"]
                     if t["trip_id"] == "rail_sans_sens")
        self.assertIn(f":{gtfs_traces.SENS_ABSENT}:", shape)

    def test_sens_normalise_couvre_les_trois_formes_de_valeur_absente(self):
        for absent in (None, "", "nan", "NaN", "None", float("nan")):
            self.assertEqual(gtfs_traces.sens_normalise(absent), gtfs_traces.SENS_ABSENT)
        self.assertEqual(gtfs_traces.sens_normalise("0"), "0")
        self.assertEqual(gtfs_traces.sens_normalise(1), "1")

    def test_empreinte_du_motif_est_stable_et_distingue_les_dessertes(self):
        self.assertEqual(gtfs_traces.empreinte_suite(["A", "B"]),
                         gtfs_traces.empreinte_suite(["A", "B"]))
        self.assertNotEqual(gtfs_traces.empreinte_suite(["A", "B"]),
                            gtfs_traces.empreinte_suite(["B", "A"]))


class TestRecettesAlignees(BaseTemporaire):
    def test_les_deux_recettes_fabriquent_les_memes_shape_id(self):
        """This is the invariant that keeps the five-month drift from coming back: the
        layer and the trips go through the SAME module to name a shape."""
        from scripts.data.gama import export_gtfs_layers

        couche_routes, _stops, _comptes = export_gtfs_layers.couches(
            {nom: chemin for nom, chemin in self.feeds.items()},
            journal=lambda *_a, **_k: None)
        couche = couche_depuis_feeds(self.feeds, self.racine / "couche.shp")
        self.assertEqual(self.lancer(couche=couche), 0)
        des_courses = {t["shape_id"] for t in self.produit()["trip_list"]}
        self.assertTrue(des_courses <= set(couche_routes["shape_id"]),
                        "tout tracé porté par une course doit exister dans la couche "
                        "produite par l'autre recette")


class TestTableDesTracesPubliee(BaseTemporaire):
    """The recipe publishes the mapping it USED, and the runtime reads it.

    Without this side file, `GTFSData.get_shape_id_from_route_info` only knows
    the primary feed: measured on 2026-09-04, 80 of the 199 lines of the trips
    file ran in GAMA without any itinerary being able to designate them —
    no agent could board a TER or a liO coach. Rebuilding it
    on the runtime side would have kept TWO implementations of the rule
    `<route_id>:<sens>:<empreinte>` alive (the TER publishes no geometry), and the
    runtime would also have to reproduce the trips the recipe drops.
    """

    def _lancer_et_lire(self, **kwargs):
        from inputs.gtfs import table_traces

        couche = kwargs.pop("couche", None) or couche_depuis_feeds(
            self.feeds, self.racine / "routes.shp")
        self.assertEqual(self.lancer(couche=couche, **kwargs), 0)
        annexe = self.sortie.parent / table_traces.NOM_FICHIER
        self.assertTrue(annexe.exists(), "la table des tracés doit être publiée "
                                         "à côté de trip_info.json")
        return couche, annexe, table_traces

    def test_la_table_est_publiee_et_relue_par_le_runtime(self):
        _couche, annexe, table_traces = self._lancer_et_lire()
        table, arrets, journal = table_traces.charger(annexe)
        self.assertEqual(journal["recette"], "scripts/data/gama/export_trip_info.py")
        self.assertEqual(sorted(journal["temoins"]),
                         ["routes.dbf", "routes.shp", "trip_info.json"])
        self.assertTrue(arrets, "le catalogue des arrêts doit être publié")

    def test_toute_course_du_fichier_est_designable(self):
        """The end-to-end invariant: every trip has its pair in the table."""
        _couche, annexe, table_traces = self._lancer_et_lire()
        table, _arrets, _j = table_traces.charger(annexe)
        for trip in self.produit()["trip_list"]:
            self.assertIn(trip["shape_id"], table.get(trip["route_id"], {}),
                          f"la course {trip['trip_id']} n'est désignable par aucun "
                          f"itinéraire")

    def test_la_ligne_sans_geometrie_porte_le_shape_id_fabrique_par_la_recette(self):
        """The TER: the identifier comes from the shared module, not from a feed."""
        _couche, annexe, table_traces = self._lancer_et_lire()
        table, _arrets, _j = table_traces.charger(annexe)
        attendu = gtfs_traces.shape_id_synthetique("R1", "0", ["R1", "R2", "R3", "R4"])
        self.assertIn(attendu, table["R1"])
        # And it serves its stops in order: that is what
        # `get_shape_id_from_route_info` tests to choose a shape.
        self.assertLess(table["R1"][attendu]["R1"], table["R1"][attendu]["R4"])

    def test_les_traces_de_la_table_sont_tous_dans_la_couche(self):
        couche, annexe, table_traces = self._lancer_et_lire()
        table, _arrets, _j = table_traces.charger(annexe)
        de_la_couche = set(recette.types_de_la_couche(couche)[0])
        for route_id, par_trace in table.items():
            for shape_id in par_trace:
                self.assertIn(shape_id, de_la_couche,
                              f"{shape_id} ({route_id}) est désignable mais absent de "
                              f"la couche : `route first_with (...)` rendrait nil")

    def test_une_course_ecartee_par_la_recette_n_est_pas_designable(self):
        """The recipe's exclusions hold for the table: otherwise an itinerary
        would offer a vehicle that will never be born."""
        couche = couche_depuis_feeds(
            self.feeds, self.racine / "routes.shp",
            garder=lambda sid, rt: not sid.endswith(
                gtfs_traces.empreinte_suite(["R1", "R2"])))
        _c, annexe, table_traces = self._lancer_et_lire(couche=couche)
        table, _arrets, _j = table_traces.charger(annexe)
        ecarte = gtfs_traces.shape_id_synthetique("R1", "0", ["R1", "R2"])
        self.assertNotIn(ecarte, table.get("R1", {}))

    def test_une_couche_refaite_seule_depareille_la_table(self):
        """`make gama-layers` alone, on other data: the freshness check refuses.

        This is the check that was missing for five months: nothing said that the table and
        the layer did not come from the same generation. An identical
        rewrite, for its part, triggers nothing — geopandas writes the same bytes,
        and a false alarm teaches people to ignore alarms.
        """
        couche, annexe, table_traces = self._lancer_et_lire()
        import geopandas as gpd

        identique = gpd.read_file(couche)
        identique.to_file(couche)
        table_traces.charger(annexe)  # does not raise: same bytes

        # A layer that REALLY changed — one shape fewer, as after a
        # new scope — and the pair is mismatched.
        identique.iloc[1:].to_file(couche)
        with self.assertRaises(table_traces.TableTracesInvalide) as capture:
            table_traces.charger(annexe)
        self.assertEqual(capture.exception.motif, "depareillee")

    def test_des_courses_reecrites_seules_depareillent_la_table(self):
        couche, annexe, table_traces = self._lancer_et_lire()
        contenu = json.loads(self.sortie.read_text(encoding="utf-8"))
        contenu["trip_list"] = contenu["trip_list"][:-1]
        self.sortie.write_text(json.dumps(contenu), encoding="utf-8")
        with self.assertRaises(table_traces.TableTracesInvalide) as capture:
            table_traces.charger(annexe)
        self.assertEqual(capture.exception.motif, "depareillee")

    def test_courses_absentes_de_la_table_font_echouer_la_recette(self):
        """The internal safeguard: the recipe refuses to deliver a table with holes."""
        from inputs.gtfs.reader import GTFSData

        original = GTFSData.table_traces_serialisable

        def tronquee(self):
            table, arrets = original(self)
            table.pop(next(iter(table)))
            return table, arrets

        GTFSData.table_traces_serialisable = tronquee
        self.addCleanup(setattr, GTFSData, "table_traces_serialisable", original)
        couche = couche_depuis_feeds(self.feeds, self.racine / "routes.shp")
        self.assertEqual(self.lancer(couche=couche), recette.CODE_REFUS)
        self.assertFalse(self.sortie.exists())


if __name__ == "__main__":
    unittest.main()
