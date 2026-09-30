"""Tests of the complementary recipes of `includes/` (`scripts/data/gama/gama_includes.py`).

Synthetic shapefiles of a few features, no network access, no large file. Each test
covers a verdict which, if reversed, would publish a different layer as identical
— or refuse an identical layer.

    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_gama_includes_recettes.py -q
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.gama import gama_includes as gi


def ecrire_couche(chemin: Path, points: list[tuple[float, float]], noms: list[str]) -> Path:
    import geopandas as gpd
    from shapely.geometry import Point
    gpd.GeoDataFrame({"nom": noms}, geometry=[Point(p) for p in points],
                     crs="EPSG:4326").to_file(chemin)
    return chemin


class Base(unittest.TestCase):
    def setUp(self):
        self.racine = Path(tempfile.mkdtemp(prefix="gama_includes_recettes_"))
        (self.racine / "ref").mkdir()
        (self.racine / "cand").mkdir()

    def couche(self, dossier: str, points, noms, nom="stops.shp") -> Path:
        return ecrire_couche(self.racine / dossier / nom, points, noms)


class TestLectureBrute(Base):
    def test_compte_et_emprise_sans_shx_ni_dbf(self):
        """The original building.shp has neither .shx nor .dbf: the raw read must suffice."""
        shp = self.couche("ref", [(1.0, 43.0), (1.5, 43.5), (2.0, 44.0)], ["a", "b", "c"])
        for ext in (".shx", ".dbf"):
            shp.with_suffix(ext).unlink()
        s = gi.stats_shp_brut(shp)
        self.assertEqual(s["enregistrements"], 3)
        self.assertEqual(s["type_forme"], 1)
        self.assertEqual(s["emprise"], [1.0, 43.0, 2.0, 44.0])


class TestVerdictShapefile(Base):
    POINTS = ((1.0, 43.0), (1.5, 43.5))

    def test_meme_export_identique(self):
        ref = self.couche("ref", self.POINTS, ["a", "b"])
        cand = self.racine / "cand" / "stops.shp"
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            if ref.with_suffix(ext).exists():
                cand.with_suffix(ext).write_bytes(ref.with_suffix(ext).read_bytes())
        self.assertEqual(gi._cmp_shapefile(ref, cand)["verdict"], "identique")

    def test_seule_la_date_d_entete_dbf_differe_equivalent(self):
        ref = self.couche("ref", self.POINTS, ["a", "b"])
        cand = self.couche("cand", self.POINTS, ["a", "b"])
        octets = bytearray(cand.with_suffix(".dbf").read_bytes())
        octets[1:4] = bytes([99, 12, 31])  # another export day
        cand.with_suffix(".dbf").write_bytes(bytes(octets))
        r = gi._cmp_shapefile(ref, cand)
        self.assertEqual(r["verdict"], "équivalent")
        self.assertTrue(r["seul_ecart_date_entete_dbf"])

    def test_un_attribut_change_different(self):
        ref = self.couche("ref", self.POINTS, ["a", "b"])
        cand = self.couche("cand", self.POINTS, ["a", "X"])
        self.assertEqual(gi._cmp_shapefile(ref, cand)["verdict"], "différent")

    def test_une_geometrie_deplacee_different(self):
        ref = self.couche("ref", self.POINTS, ["a", "b"])
        cand = self.couche("cand", [(1.0, 43.0), (1.5, 43.6)], ["a", "b"])
        r = gi._cmp_shapefile(ref, cand)
        self.assertEqual(r["verdict"], "différent")
        self.assertEqual((r["geometries_ref_seules"], r["geometries_cand_seules"]), (1, 1))


class TestRessourcesAbsentes(Base):
    def test_osm_p95_source_absente(self):
        code = gi.main(["osm-p95", "--source", str(self.racine / "absent.osm.pbf"),
                        "--sortie", str(self.racine / "cand")])
        self.assertEqual(code, gi.CODE_RESSOURCE)
        self.assertFalse((self.racine / "cand" / "Toulouse_bbox_p95.osm.pbf").exists())

    def test_batiments_archive_absente(self):
        code = gi.main(["batiments", "--archive", str(self.racine / "absent.7z"),
                        "--sortie", str(self.racine / "cand")])
        self.assertEqual(code, gi.CODE_RESSOURCE)

    def test_fond_carte_refuse_sans_forcer(self):
        """CARTO tiles without a key return an image of the right size, but blank: empty success."""
        t0 = time.monotonic()
        code = gi.main(["fond-carte", "--sortie", str(self.racine / "cand")])
        self.assertEqual(code, gi.CODE_RESSOURCE)
        self.assertFalse((self.racine / "cand" / "toulouse_map.png").exists())
        self.assertLess(time.monotonic() - t0, 5, "le refus ne doit pas attendre le réseau")


class TestComparerDossier(Base):
    def test_couche_absente_comptee_absente(self):
        self.couche("ref", [(1.0, 43.0)], ["a"])
        code = gi.main(["comparer", "--reference", str(self.racine / "ref"),
                        "--candidat", str(self.racine / "cand"),
                        "--json", str(self.racine / "cmp.json")])
        self.assertEqual(code, 0)
        import json
        d = json.loads((self.racine / "cmp.json").read_text())
        self.assertEqual(d["couches"]["stops.shp"]["verdict"], "absent")
        self.assertEqual(d["bilan"]["absent"], len(gi.COUCHES))


if __name__ == "__main__":
    unittest.main()
