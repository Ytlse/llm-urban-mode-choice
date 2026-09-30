"""The "How the population is built" page is built from the versioned seal, without
the traces (absent from a clone): the missing sections say so instead of inventing."""
from pathlib import Path

from scripts.panel.synthese_generation_population import build, TEMPLATE_V2, CONFIG_EQASIM, COMMUNES, GRAPHE_META

REPO = Path(__file__).resolve().parents[2]
SCEAU = REPO / "data" / "population" / "population_1000_PANEL_v4"


def test_la_page_se_construit_depuis_le_sceau_seul():
    html = build(SCEAU, None, None, None, None, CONFIG_EQASIM, COMMUNES,
                 GRAPHE_META if GRAPHE_META.exists() else None, None, TEMPLATE_V2, "synthese.html")
    assert "<title>Fabrication de la population v4</title>" in html
    # Figures read from the MANIFEST, never typed in
    assert "11 329" in html and "513" in html and "panel_seal_v4" in html
    assert "9f05c655c3ad2cf4" in html
    # What is not provided is stated as such
    assert "rapport non fourni" in html and "audit non fourni" in html
    # Figures from the log are marked
    assert "<sup class='j'" in html
