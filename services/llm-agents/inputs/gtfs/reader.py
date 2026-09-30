from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel
from models import BBox, Location
# from scipy.spatial import KDTree
import zipfile
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from loguru import logger
from settings import settings
# `table_traces` depends on nothing in the package: importing it here closes no
# cycle, even if `inputs/gtfs/__init__.py` is importing THIS file.
from inputs.gtfs import table_traces


STRING_COLUMNS = [
    'route_id', 'service_id', 'trip_id', 'shape_id', 'stop_id', 'date',
]


class Stop(BaseModel):
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float


def _correct_color_hex_string(value):
    value = str(value)
    if value == 'nan':
        return "#222222"
    if value.startswith('#'):
        return value
    if len(value) == 6:
        return '#' + value
    if len(value) == 3:
        return '#' + ''.join([c * 2 for c in value])
    return value

def _lire_table_gtfs(feed: "Path", nom: str):
    """Read a table of a GTFS feed, whether the feed is a directory or a zip.

    Returns `None` if the table does not exist: a feed without `routes.txt` is not an error,
    it is a feed that has nothing to say about lines.
    """
    if feed.is_dir():
        chemin = feed / nom
        return pd.read_csv(chemin, dtype=str) if chemin.exists() else None
    if feed.suffix == ".zip":
        with zipfile.ZipFile(feed) as archive:
            if nom not in archive.namelist():
                return None
            with archive.open(nom) as f:
                return pd.read_csv(f, dtype=str)
    return None


class GTFSData:
    # Where the shape table comes from:
    #   "annexe" — the file published by the recipe (the runtime case);
    #   "feed"   — recomputed from the loaded tables, without reading the annex file.
    #              This is the mode of the RECIPE itself: it is the one producing the
    #              file, it cannot read it in order to write it.
    SOURCE_ANNEXE = "annexe"
    SOURCE_FEED = "feed"

    def __init__(self, **kwargs):
        self.stop_times = kwargs["stop_times"]
        self.stops = kwargs["stops"]
        self.routes = kwargs["routes"]
        self.trips = kwargs["trips"]
        self.shapes = kwargs["shapes"]
        self.calendar_dates = kwargs["calendar_dates"]
        self.calendar = kwargs["calendar"]
        self._source_demandee = kwargs.get("table_traces", self.SOURCE_ANNEXE)

        # Init lookup maps
        self.init_route_lookup_maps()
        self.init_shape_lookup_maps()

        # init the KDTree for the stops
        # TODO: remove these lines, this is used for python RAPTOR implementation
        # which is deprecated
        # if kwargs.get("index_stop_area") is not True:
        #     self.indexed_stops_df = self.stops[self.stops['location_type'] != 1]
        # else:
        #     self.indexed_stops_df = self.stops.copy()
        # points = self.indexed_stops_df[['stop_lon', 'stop_lat']].values
        # points = points.astype(float)
        # self.stop_kdtree = KDTree(points)

    def init_route_lookup_maps(self):
        self.route_name_id_map = {
            str(row['route_short_name']): str(row['route_id'])
            for _, row in self.routes.iterrows()
        }

        self.route_id_map = {
            str(row['route_id']): {
                "route_short_name": str(row['route_short_name']),
                "route_long_name": str(row['route_long_name']),
                "route_type": settings.gtfs.gtfs_modality_name_map.get(str(row['route_type']), "Unknown"),
            }
            for _, row in self.routes.iterrows()
        }
        return self._joindre_catalogues_des_autres_feeds()

    def _joindre_catalogues_des_autres_feeds(self) -> dict:
        """Add to the line dictionary the catalogue of the OTHER feeds in service.

        The reader only loads one feed (`settings.gtfs.gtfs_file`, Tisséo), whereas the
        OTP graph carries three since 2026-09-04: Tisséo, the TER and the regional coach
        liO. A liO or TER line identifier was therefore in no table, and the
        agent's prompt read "Trajet en **Unknown 392**" for the 319 lines of the two
        regional networks — mode AND number lost, whereas the mode table has known
        the train (`route_type` 2) and the bus (3) since the same day.

        Only the LINE CATALOGUE is joined, not the schedules nor the stops: it is all
        the prompt needs, and the heavy tables of the primary feed stay intact.

        The join is done on `route_id`, measured **without collision** between the three feeds;
        short names, on the other hand, collide (a line "1" exists everywhere), hence two
        dictionaries handled differently: by identifier we complete, by short name we
        keep the primary feed and count. An identifier collision, however, is a
        real ambiguity: it raises an alarm.
        """
        from trip_helper.otp import feeds_en_service

        primaire = Path(settings.gtfs.gtfs_file)
        journal = {"feeds": {}, "lignes_ajoutees": 0,
                   "collisions_identifiant": 0, "collisions_nom_court": 0}
        for feed in feeds_en_service(str(primaire)):
            if feed.resolve() == primaire.resolve():
                continue
            try:
                routes = _lire_table_gtfs(feed, "routes.txt")
            except (OSError, ValueError, KeyError) as exc:
                logger.error(f"[ALARME] catalogue de lignes illisible dans {feed.name} : "
                             f"{exc} — les lignes de ce réseau resteront « Unknown » dans le prompt")
                continue
            if routes is None:
                continue
            ajoutees = 0
            for _, row in routes.iterrows():
                rid = str(row['route_id'])
                if rid in self.route_id_map:
                    journal["collisions_identifiant"] += 1
                    continue
                self.route_id_map[rid] = {
                    "route_short_name": str(row.get('route_short_name', '')),
                    "route_long_name": str(row.get('route_long_name', '')),
                    "route_type": settings.gtfs.gtfs_modality_name_map.get(
                        str(row.get('route_type', '')), "Unknown"),
                }
                nom = str(row.get('route_short_name', ''))
                if nom in self.route_name_id_map:
                    journal["collisions_nom_court"] += 1
                else:
                    self.route_name_id_map[nom] = rid
                ajoutees += 1
            journal["feeds"][feed.name] = ajoutees
            journal["lignes_ajoutees"] += ajoutees

        if journal["collisions_identifiant"]:
            logger.error(
                f"[ALARME] {journal['collisions_identifiant']} line identifier(s) "
                f"claimed by two GTFS feeds: the catalogue of the primary feed is "
                f"kept, the other ignored. The mode and number displayed in the prompt "
                f"may designate the wrong line."
            )
        # Success is logged too: without this line, we cannot tell "the three
        # networks are there" from "the reader found nothing to join".
        logger.info(
            f"[GTFS] line catalogue: {len(self.route_id_map)} line(s) in total, of which "
            f"{journal['lignes_ajoutees']} came from the other feeds in service "
            f"({journal['feeds'] or 'aucun'}) ; {journal['collisions_nom_court']} short name(s) "
            f"already taken, kept for the primary feed"
        )
        return journal

    def init_shape_lookup_maps(self):
        """Load the shape table — the file published by the recipe first.

        It is this table that `get_shape_id_from_route_info` queries to set
        a `shape_id` on an itinerary leg, and it is this `shape_id` that
        `Inhabitant.gaml` compares with the vehicle's to make the agent BOARD.

        Built from the primary feed only — which it was until
        2026-09-04 — it ignored 80 of the 199 lines carrying trips (17 TER,
        58 liO coaches, 5 circular lines): `get_shape_id_from_route_info`
        returned `[]`, indistinguishable from "no shape for this pair of stops".
        See `inputs/gtfs/table_traces.py` for why the annex file exists.

        The fallback on the primary feed remains — without it, a missing annex
        file would deprive the simulation of ALL public transport — but it
        is **announced as [ALARME]**, the table is marked partial, and each
        line it cannot designate raises an alarm in turn (rising edge).
        Silence was the defect; a silent fallback would have been another.
        """
        # Counters of the designation alarms (see `_alarme_ligne_indesignable`).
        self._lignes_indesignables: set = set()
        self._appels_sans_trace = 0
        self._appels_sans_arret = 0

        if self._source_demandee == self.SOURCE_FEED:
            self.route_id_shape_lookup_map = self._table_traces_depuis_le_feed()
            self.arrets_hors_feed_primaire = {}
            self.source_table_traces = self.SOURCE_FEED
            self.table_traces_partielle = False
            self.journal_table_traces = {"source": self.SOURCE_FEED}
            logger.info(
                f"[GTFS] shape table recomputed from the loaded tables "
                f"(recipe mode) : {len(self.route_id_shape_lookup_map)} route_id, "
                f"{sum(len(v) for v in self.route_id_shape_lookup_map.values())} shape(s)")
            return

        chemin = Path(settings.gtfs.shape_lookup_file)
        try:
            table, arrets, journal = table_traces.charger(chemin)
        except table_traces.TableTracesInvalide as exc:
            self.route_id_shape_lookup_map = self._table_traces_depuis_le_feed()
            self.arrets_hors_feed_primaire = {}
            self.source_table_traces = f"feed_primaire:{exc.motif}"
            self.table_traces_partielle = True
            self.journal_table_traces = {"source": self.source_table_traces,
                                         "motif": exc.motif, "detail": exc.detail}
            logger.error(
                f"[ALARME] table des tracés inutilisable ({exc.motif}) : {exc.detail}. "
                f"Repli sur le SEUL feed primaire ({settings.gtfs.gtfs_file}) : "
                f"{len(self.route_id_shape_lookup_map)} route_id désignables. Les lignes "
                f"des autres réseaux (TER, cars liO) rouleront dans GAMA sans qu'aucun "
                f"agent puisse y monter."
            )
            return

        self.route_id_shape_lookup_map = table
        self.arrets_hors_feed_primaire = arrets
        self.source_table_traces = self.SOURCE_ANNEXE
        self.table_traces_partielle = False
        self.journal_table_traces = {"source": self.SOURCE_ANNEXE, **journal}
        # Success is logged explicitly: without this line, we cannot tell
        # "the table of the three networks is in place" from "the runtime read nothing".
        logger.info(
            f"[GTFS] table des tracés lue dans {chemin} (générée le "
            f"{journal.get('genere_le')} par {journal.get('recette')}) : "
            f"{journal['comptes']['route_id']} route_id, {journal['comptes']['traces']} "
            f"tracé(s), {journal['arrets_catalogue']} arrêt(s) au catalogue ; "
            f"réseaux {journal.get('reseaux') or 'non détaillés'} ; fraîcheur vérifiée "
            f"sur {', '.join(journal['temoins'])}"
        )

    def _table_traces_depuis_le_feed(self) -> dict:
        """`route_id → {shape_id → {stop_id: stop_sequence}}` from the loaded tables.

        The historical computation, now reserved for two uses: the RECIPE, which
        uses it to produce the annex file from the merged feed of the
        three networks, and the alarmed fallback when this file is missing.
        """
        stops = self.trips.groupby('shape_id').agg({
            'route_id': 'first',
            'trip_id': 'first',
        }).reset_index()\
        .merge(
            self.stop_times[['trip_id', 'stop_sequence', 'stop_id']],
            on='trip_id',
            how='left',
        ).merge(
            self.stops[['stop_id', 'stop_name']],
            on='stop_id',
            how='left',
        )

        m = defaultdict(dict)
        for _, row in stops.iterrows():
            if row['shape_id'] not in m[row['route_id']]:
                m[row['route_id']][row['shape_id']] = {}
            m[row['route_id']][row['shape_id']][row['stop_id']] = row['stop_sequence']
        return m

    def table_traces_serialisable(self) -> tuple[dict, dict]:
        """The table and the stop catalogue, as JSON types — for the recipe.

        `stop_sequence` arrives from pandas as `int64`, which `json` refuses; the
        `shape_id` of a feed without geometry are already strings. The conversion
        is done here, as close as possible to the computation, and not in the recipe: it is the
        same table that serves the runtime.
        """
        table = {
            str(route_id): {
                str(shape_id): {str(stop_id): int(rang) for stop_id, rang in stops.items()}
                for shape_id, stops in par_trace.items()
            }
            for route_id, par_trace in self.route_id_shape_lookup_map.items()
        }
        arrets_utiles = {stop_id for par_trace in table.values()
                         for stops in par_trace.values() for stop_id in stops}
        arrets = {}
        for _, row in self.stops.iterrows():
            stop_id = str(row['stop_id'])
            if stop_id not in arrets_utiles:
                continue
            arrets[stop_id] = {"stop_name": str(row['stop_name']),
                               "stop_lat": float(row['stop_lat']),
                               "stop_lon": float(row['stop_lon'])}
        return table, arrets

    def load_world_bounding_box(self) -> BBox:
        min_lon, min_lat, max_lon, max_lat = self.get_bounding_box()
        buffer = 0.05  # degrees ~ 5km
        return BBox(
            min_lon=min_lon - buffer,
            min_lat=min_lat - buffer,
            max_lon=max_lon + buffer,
            max_lat=max_lat + buffer,
        )

    def get_shape_id_from_route_info(self, route_id: str, from_stop_id: Optional[str], to_stop_id: Optional[str]) -> list[str]:
        """The `shape_id` of line `route_id` that serve both stops in order.

        An empty list means "no vehicle of this line can be
        designated": `Inhabitant.gaml` will find nobody to board. Two
        causes were confused in this silence — a pair of stops that exists
        on no shape (normal), and a line the table does not know at
        all (the defect). The second now raises an alarm, on the rising edge:
        once per `route_id`, to name the line without flooding the log.
        """
        if not from_stop_id or not to_stop_id:
            self._appels_sans_arret += 1
            self._alarme_ligne_indesignable(
                route_id, "arret non resolu",
                f"jambe sans identifiant d'arrêt (de={from_stop_id!r}, vers={to_stop_id!r}) : "
                f"l'arrêt rendu par OTP n'a pas été retrouvé dans les tables GTFS")
            return []
        if route_id not in self.route_id_shape_lookup_map:
            self._appels_sans_trace += 1
            self._alarme_ligne_indesignable(
                route_id, "ligne absente de la table",
                f"aucun tracé connu pour cette ligne ; la table vient de "
                f"{self.source_table_traces} et porte "
                f"{len(self.route_id_shape_lookup_map)} route_id")
            return []

        results = []
        for shape_id, stops in self.route_id_shape_lookup_map[route_id].items():
            if from_stop_id not in stops or to_stop_id not in stops:
                continue
            if stops[from_stop_id] < stops[to_stop_id]:
                results.append(shape_id)
        return results

    def _alarme_ligne_indesignable(self, route_id: str, motif: str, detail: str) -> None:
        """[ALARME] on the rising edge: once per line, not once per itinerary.

        An agent can request dozens of itineraries on the same line; the
        cause, however, is unique. We therefore alarm at the FIRST occurrence of
        each `route_id` and count the following ones, which the end-of-run
        report picks up (`resume_table_traces`).
        """
        cle = (route_id, motif)
        if cle in self._lignes_indesignables:
            return
        self._lignes_indesignables.add(cle)
        nom = self.get_route_short_name_by_id(route_id)
        mode = self.get_route_type_string_by_id(route_id)
        logger.error(
            f"[ALARME] itinéraire indésignable — ligne {route_id} ({mode} {nom}) : {motif}. "
            f"{detail}. Aucun agent ne pourra monter dans un véhicule de cette ligne : "
            f"le véhicule roule dans GAMA, l'itinéraire ne le nomme pas."
        )

    def resume_table_traces(self) -> dict:
        """Enough to reconstruct afterwards what the table could and could not designate."""
        return {
            "source": getattr(self, "source_table_traces", "inconnue"),
            "partielle": getattr(self, "table_traces_partielle", None),
            "route_id_connus": len(self.route_id_shape_lookup_map),
            "lignes_indesignables": sorted({rid for rid, _ in self._lignes_indesignables}),
            "appels_ligne_absente": self._appels_sans_trace,
            "appels_arret_non_resolu": self._appels_sans_arret,
            "journal": getattr(self, "journal_table_traces", {}),
        }

    def resoudre_route_id(self, identifiant_otp: str) -> str:
        """OTP's line identifier brought back to the GTFS one, through the CATALOGUE.

        OTP prefixes its identifiers with the feed name: `tisseo:line:8`,
        `ter:FR:Line::68c586ae-…`, `lio:305`. `parse_gtfs_entity_id` removes
        "the first segment when there are at least two `:`" — a rule that
        works for the first two and **fails for the third**: the
        `route_id` of a liO coach contains no `:`, so `lio:305` goes through
        intact and matches nothing. Measured on 2026-09-04 on an itinerary
        returned by the OTP in service: `transit_route='lio:305'` where the line
        is called `305` in the GTFS, in `trip_info.json` and hence in
        `ROUTE_VEHICLE_MAP` — the agent could neither designate the shape nor find
        the vehicle.

        So we no longer guess the form of the identifier: we try the raw one,
        then remove the prefixes one by one, and keep the first candidate
        that the LINE CATALOGUE knows (the three feeds are in it since
        2026-09-04). No candidate recognised: we return the raw one stripped of its
        first segment — the former behaviour — and the next call will raise an alarm.
        """
        candidats = [identifiant_otp]
        reste = identifiant_otp
        while ":" in reste:
            reste = reste.split(":", 1)[1]
            candidats.append(reste)
        for candidat in candidats:
            if candidat in self.route_id_map:
                return candidat
        return candidats[1] if len(candidats) > 1 else identifiant_otp

    def get_route_id_by_name(self, route_name: str) -> str:
        # Get the route id by route name
        if route_name in self.route_name_id_map:
            return self.route_name_id_map[route_name]
        raise ValueError(f"Route {route_name} not found")
    
    def get_route_type_string_by_id(self, route_id: str) -> str:
        return self.route_id_map.get(route_id, {}).get("route_type", "Unknown")
    
    def get_route_long_name_by_id(self, route_id: str) -> str:
        return self.route_id_map.get(route_id, {}).get("route_long_name", "Unknown")
    
    def get_route_short_name_by_id(self, route_id: str) -> str:
        return self.route_id_map.get(route_id, {}).get("route_short_name", "Unknown")

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        # Get the bounding box of the stops
        min_lon = self.stops['stop_lon'].min()
        max_lon = self.stops['stop_lon'].max()
        min_lat = self.stops['stop_lat'].min()
        max_lat = self.stops['stop_lat'].max()
        return min_lon, min_lat, max_lon, max_lat

    # def get_nearest_stops(self, lon, lat, stops_count=5) -> tuple[list[Stop], list[float]]:
    #     # Find the nearest stops using KDTree
    #     distances, indices = self.stop_kdtree.query([lon, lat], k=stops_count)
    #     nearest_stops = self.indexed_stops_df.iloc[indices]
    #     stops = [Stop.model_validate(row) for row in nearest_stops.to_dict(orient="records")]
    #     return stops, distances
    
    def get_stop_id_by_name(self, stop_name: str) -> Optional[str]:
        match = self.stops[self.stops['stop_name'] == stop_name]
        if match.empty:
            return None
        return str(match.iloc[0]['stop_id'])

    def get_stop(self, stop_id: str) -> Stop:
        """The stop, looked up in the primary feed THEN in the annex catalogue.

        The annex catalogue carries the stops of the three networks served by the
        published shapes (TER stations, liO coach stops). Without it,
        `_resolve_gtfs_stop` (`trip_helper/otp.py`) did not find the station
        returned by OTP, the leg left with `stop_id=None`, and
        `get_shape_id_from_route_info` returned `[]` before even consulting the
        table: the second link of the chain that prevented boarding a
        train.

        ⚠ This catalogue is kept APART from `self.stops`, on purpose:
        `get_bounding_box` derives from it the footprint of the simulation world
        (`urban_mobility_agents/factory/factory.py`), and pouring stations from
        all of Occitanie into it would extend it far beyond the survey scope.
        """
        stop = self.stops[self.stops['stop_id'] == stop_id]
        if stop.empty:
            annexe = getattr(self, "arrets_hors_feed_primaire", {}).get(stop_id)
            if annexe is not None:
                return Stop(stop_id=stop_id, stop_name=annexe["stop_name"],
                            stop_lat=annexe["stop_lat"], stop_lon=annexe["stop_lon"])
            raise ValueError(f"Stop {stop_id} not found")
        stop = stop.iloc[0]
        return Stop.model_validate(stop.to_dict())
    
    def all_stop_locations(self) -> list[Location]:
        # Get all stop locations
        return [
            Location(lon=row['stop_lon'], lat=row['stop_lat'])
            for _, row in self.stops.iterrows()
        ]

    @classmethod
    def _read_gtfs_file_as_pd(cls, file):
        df = pd.read_csv(file, dtype={col: str for col in STRING_COLUMNS}, low_memory=False)
        return df

    @classmethod
    def read_df_from_zip(cls, zip_path, file_name):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if file_name in zip_ref.namelist():
                with zip_ref.open(file_name) as file:
                    df = cls._read_gtfs_file_as_pd(file)
                    return df
            else:
                raise ValueError(f"File {file_name} not found in {zip_path}")
            
    @classmethod
    def read_file(cls, dir, file_name):
        if not os.path.exists(dir):
            raise ValueError(f"Dir {dir} not found")
        
        if os.path.isdir(dir):
            with open(os.path.join(dir, file_name), 'r') as file:
                return cls._read_gtfs_file_as_pd(file)
        if os.path.isfile(dir) and dir.endswith('.zip'):
            return cls.read_df_from_zip(dir, file_name)
        
        raise ValueError(f"Dir {dir} is not a directory or a zip file")

    @classmethod
    def from_gtfs_files(cls, dir, table_traces=SOURCE_ANNEXE):
        """Load a GTFS feed from a directory or a zip.

        `table_traces="feed"` recomputes the shape table from the loaded
        tables instead of reading the annex file: this is the mode of the recipe
        `scripts/data/gama/export_trip_info.py`, which reads the MERGED feed of the
        three networks to produce this file — it cannot read it in order to
        write it, and the table it publishes must come from the feed it has
        really merged.
        """
        data = GTFSData(**{
            'table_traces': table_traces,
            # 'agency': read_file(dir, 'agency.txt'),
            'stops': cls.read_file(dir, 'stops.txt'),
            'shapes': cls.read_file(dir, 'shapes.txt'),
            'trips': cls.read_file(dir, 'trips.txt'),
            'stop_times': cls.read_file(dir, 'stop_times.txt'),
            'routes': cls.read_file(dir, 'routes.txt'),
            # TODO: support calendar.txt
            # for now, we pretend that all services are available, and calendar.txt file is empty
            'calendar_dates': cls.read_file(dir, 'calendar_dates.txt'),
            'calendar': cls.read_file(dir, 'calendar.txt'),
        })
        assert len(data.calendar) == 0, "calendar.txt is not supported yet"
        assert data.calendar_dates['exception_type'].unique().tolist() == [1], "calendar_dates.txt only supports exception_type = 1"

        return data
    
    @classmethod
    def DEFAULT(cls):
        # Get the GTFS data from the settings
        if not hasattr(cls, "_instance"):
            cls._instance = GTFSData.from_gtfs_files(settings.gtfs.gtfs_file)
        return cls._instance

    def to_stops_shape_file(self, output_path, crs=4326):
        stops_df = self.stops.copy()        
        routes_df = self.routes[['route_id', 'route_type']]
        trips_df = self.trips[['route_id', 'trip_id']]
        stop_times_df = self.stop_times[['stop_id', 'trip_id']]

        route_type_df = trips_df.merge(routes_df, on='route_id', how='left')
        stop_times_df = stop_times_df.merge(route_type_df, on='trip_id', how='left')
        stop_times_df = stop_times_df.groupby('stop_id').agg({'route_type': 'min'}).reset_index()

        stops_df = stops_df[['stop_id', 'stop_name', 'location_type', 'wheelchair_boarding', 'stop_lon', 'stop_lat']]
        stops_df = stops_df.merge(stop_times_df[['stop_id', 'route_type']], on='stop_id', how='left')
        # stops_df['route_type'] = stops_df['route_type'].fillna(-1).astype(float)
        stops_df.dropna(subset=['route_type'], inplace=True)
        gdf = gpd.GeoDataFrame(
            stops_df, geometry=gpd.points_from_xy(stops_df['stop_lon'], stops_df['stop_lat'], z=0)
        )
        gdf.set_crs(epsg=crs, inplace=True)

        gdf.drop(columns=['stop_lon', 'stop_lat'], inplace=True)

        # Save as Shapefile
        gdf.to_file(os.path.join(output_path, 'stops.shp'))
        gdf.to_file(os.path.join(output_path, 'stops.geojson'), driver='GeoJSON')

    def to_route_shape_file(self, output_path, crs=4326):
        shapes_df = self.shapes
        routes_df = self.routes
        trips_df = self.trips

        shapes_list = shapes_df.groupby("shape_id").apply(
            lambda l: LineString(zip(l['shape_pt_lon'], l['shape_pt_lat']))
        )
        shapes_all = pd.DataFrame({
            'shape_id': shapes_list.index,
            'geometry': shapes_list.values
        })
        
        trips_df = trips_df[['route_id', 'service_id', 'trip_id', 'shape_id']].groupby("shape_id").agg(lambda x: x.iloc[0])
        shapes_all = shapes_all.merge(trips_df, on='shape_id', how='left')
        shapes_all = shapes_all.merge(routes_df, on='route_id', how='left')

        # compact the column names
        shapes_all.rename(columns={
            'shape_id': 'shape_id',
            'route_id': 'route_id',
            'service_id': 'service_id',
            'trip_id': 'trip_id',
            'route_short_name': 'short_name',
            'route_long_name': 'long_name',
            'route_color': 'color',
            'route_text_color': 'text_color',
            'route_type': 'route_type',
        }, inplace=True)

        # correct the color hex string
        shapes_all['color'] = shapes_all['color'].apply(_correct_color_hex_string)
        shapes_all['text_color'] = shapes_all['text_color'].apply(_correct_color_hex_string)

        gdf = gpd.GeoDataFrame(shapes_all)
        gdf.set_crs(epsg=crs, inplace=True)

        # Save as Shapefile
        gdf.to_file(os.path.join(output_path, 'routes.shp'))
        gdf.to_file(os.path.join(output_path, 'routes.geojson'), driver='GeoJSON')


if __name__ == '__main__':
    gtfs = GTFSData.from_gtfs_files("../data/gtfs/tisseo_gtfs/")

    output_dir = "../data/exports/gtfs/"
    os.makedirs(output_dir, exist_ok=True)
    gtfs.to_stops_shape_file(output_dir)
    gtfs.to_route_shape_file(output_dir)
