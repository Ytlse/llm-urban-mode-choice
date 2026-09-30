"""OTP linking check — breakdown by ring.

The aggregated total does not answer the question raised by the scope report:
"without liO, the outer rings have only a tenth of their PT service". What
is measured is the lack of service *per ring of residence*. These
tests are about classifying the points, not about the OTP requests.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.gtfs.otp_link_check import classer_par_couronne  # noqa: E402

CAPITOLE = (43.6045, 1.4440)
BLAGNAC = (43.6350, 1.3940)     # 1st ring
CAZERES = (43.2100, 1.0850)     # 3rd ring
BORDEAUX = (44.8378, -0.5792)   # outside scope


class TestClassementCouronnes(unittest.TestCase):
    GEOJSON = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "couronne_perimetre.geojson"

    def _classer(self, coordonnees, geojson=None):
        points = [("p", "home", lat, lon) for lat, lon in coordonnees]
        return classer_par_couronne(points, geojson or self.GEOJSON)

    def test_chaque_point_recoit_une_couronne(self):
        classement = self._classer([CAPITOLE, BLAGNAC, CAZERES])
        self.assertEqual(len(classement), 3)
        self.assertEqual(classement[0], "Toulouse")
        self.assertEqual(classement[1], "1st ring")
        self.assertEqual(classement[2], "3rd ring")

    def test_point_hors_perimetre_nest_pas_rattache_de_force(self):
        """A point outside the 453 municipalities is counted apart, not put into the
        nearest ring: linking it would skew the rates."""
        classement = self._classer([BORDEAUX])
        self.assertEqual(classement[0], "hors perimetre")

    def test_absence_de_geometrie_ne_casse_pas_la_mesure(self):
        """Without the rings file, the breakdown is empty — the linking
        check itself must keep running."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._classer([CAPITOLE], Path(tmp) / "absent.geojson"), {})

    def test_geometrie_synthetique(self):
        """The classification does read the `couronne` property of the GeoJSON."""
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "couronnes.geojson"
            carre = [[[1.0, 43.0], [2.0, 43.0], [2.0, 44.0], [1.0, 44.0], [1.0, 43.0]]]
            chemin.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {"couronne": "test"},
                              "geometry": {"type": "Polygon", "coordinates": carre}}],
            }), encoding="utf-8")
            self.assertEqual(self._classer([(43.5, 1.5), (40.0, 1.5)], chemin),
                             {0: "test", 1: "hors perimetre"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
