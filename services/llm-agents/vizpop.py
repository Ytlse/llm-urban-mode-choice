"""
Population visualization server.
Run with: python vizpop.py [port]  (default port 5050)
"""

import json
import glob
import sys
from pathlib import Path

import folium
import osmnx as ox
from flask import Flask, Response
from shapely import wkt

from geography import TOULOUSE_TRANSIT_SERVICE_WKT

app = Flask(__name__)

_HERE = Path(__file__).resolve().parent
_HG_CACHE = _HERE / ".hg_boundary.geojson"

# Geometry of the four rings of the EMC² 2023 survey = the scope of the 453 communes (ticket
# 031, part 2). Resource of the mobility_core package (editable on the host, mounted under /opt in
# the controller container).
def _couronnes_candidates() -> tuple[Path, ...]:
    cands: list[Path] = []
    try:
        from mobility_core.resources import data_path
        cands.append(data_path("couronne_perimetre.geojson"))
    except ImportError:  # pragma: no cover
        pass
    cands.append(Path("/opt/mobility_core/data/couronne_perimetre.geojson"))
    return tuple(cands)


_COURONNES_CANDIDATES = _couronnes_candidates()
_COURONNE_COLORS = {
    "Toulouse": "#7c3aed", "1ere couronne": "#2563eb", "2eme couronne": "#0891b2", "3eme couronne": "#64748b",
}


def _experiments_current() -> Path:
    """Resolve experiments/current at call time so the server picks it up as
    soon as a new experiment creates the symlink — no restart needed.

    Two candidate locations (tried in order):
      1. <llm-agents>/experiments/current  — Docker volume mounted at /app/experiments
      2. <project-root>/experiments/current — local dev (project root is one level up)
    """
    for candidate in (
        _HERE / "experiments" / "current",
        _HERE.parent / "experiments" / "current",
    ):
        if candidate.exists():
            return candidate
    # Neither exists yet — return the Docker-native path so the error is legible
    return _HERE / "experiments" / "current"


# ── Haute-Garonne boundary (cached on first call) ───────────────────────────

def _get_hg_geojson() -> dict | None:
    if _HG_CACHE.exists():
        try:
            return json.loads(_HG_CACHE.read_text())
        except Exception:
            pass
    try:
        gdf = ox.geocode_to_gdf("Haute-Garonne, France")
        geojson = json.loads(gdf.to_json())
        _HG_CACHE.write_text(json.dumps(geojson))
        return geojson
    except Exception as exc:
        print(f"[vizpop] Could not fetch Haute-Garonne boundary: {exc}", file=sys.stderr)
        return None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _find_population_files() -> tuple[list[Path], Path]:
    current = _experiments_current()
    return sorted(Path(p) for p in glob.glob(str(current / "population_*.json"))), current


def _transit_polygon_coords() -> list[tuple[float, float]]:
    poly = wkt.loads(TOULOUSE_TRANSIT_SERVICE_WKT)
    return [(lat, lon) for lon, lat in poly.exterior.coords]


def _get_couronnes_geojson() -> dict | None:
    """The four rings (WGS84) as GeoJSON, or None if the resource is missing (the log says why)."""
    for cand in _COURONNES_CANDIDATES:
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[vizpop] couronne_perimetre.geojson unreadable ({cand}) : {exc}", file=sys.stderr)
                return None
    print("[vizpop] couronne_perimetre.geojson not found: the scope of the 453 communes is not drawn",
          file=sys.stderr)
    return None


def _build_map(files: list[Path]) -> tuple[folium.Map, int, int]:
    m = folium.Map(location=[43.6, 1.44], zoom_start=11, tiles="CartoDB positron")

    # ── Haute-Garonne department outline (red dashed) ────────────────────────
    hg = _get_hg_geojson()
    if hg:
        folium.GeoJson(
            hg,
            name="Département Haute-Garonne",
            style_function=lambda _: {
                "color": "#ef4444",
                "weight": 2,
                "dashArray": "8 5",
                "fillOpacity": 0,
            },
            tooltip="Haute-Garonne (31)",
        ).add_to(m)

    # ── Simulation scope: the four rings of the 453 communes (ticket 031) ──
    # Before 2026-09-03: the grey rectangle of the 30 km OSMnx graph, which was not the
    # study scope and left the 3rd ring outside.
    couronnes = _get_couronnes_geojson()
    if couronnes:
        folium.GeoJson(
            couronnes,
            name="Périmètre 453 communes",
            style_function=lambda feat: {
                "color": _COURONNE_COLORS.get(feat["properties"].get("couronne"), "#94a3b8"),
                "weight": 1.5,
                "dashArray": "6 4",
                "fillColor": _COURONNE_COLORS.get(feat["properties"].get("couronne"), "#94a3b8"),
                "fillOpacity": 0.05,
            },
            tooltip=folium.GeoJsonTooltip(fields=["couronne"], aliases=["Couronne EMC²"]),
        ).add_to(m)

    # ── Transit service: hull of the Tisséo + TER stops in the polygon (T4, yellow) ─
    folium.Polygon(
        locations=_transit_polygon_coords(),
        color="#eab308",
        weight=2,
        fill=True,
        fill_color="#eab308",
        fill_opacity=0.12,
        tooltip="Desserte TC (arrêts Tisséo + TER dans le périmètre)",
        name="Desserte TC",
    ).add_to(m)

    # ── Population layers ────────────────────────────────────────────────────
    home_layer = folium.FeatureGroup(name="Domiciles", show=True)
    activity_layer = folium.FeatureGroup(name="Activités", show=True)

    n_homes = 0
    n_activities = 0

    for path in files:
        with open(path) as f:
            population = json.load(f)

        for person in population:
            identity = person.get("identity", {})
            name = identity.get("name", person.get("person_id", "?"))
            traits = identity.get("traits_json", {})
            age = traits.get("age", "?")
            occupation = traits.get("main_occupation", "")

            home = identity.get("home", {})
            hlat, hlon = home.get("lat"), home.get("lon")
            if hlat and hlon:
                folium.CircleMarker(
                    location=[hlat, hlon],
                    radius=4,
                    color="#3b82f6",
                    fill=True,
                    fill_color="#3b82f6",
                    fill_opacity=0.7,
                    weight=1,
                    tooltip=f"{name} ({age} ans) — {occupation}",
                ).add_to(home_layer)
                n_homes += 1

            seen: set[tuple[float, float]] = set()
            for act in identity.get("activities", []):
                if act.get("purpose") == "home":
                    continue
                loc = act.get("location", {})
                alat, alon = loc.get("lat"), loc.get("lon")
                if alat and alon:
                    key = (round(alat, 5), round(alon, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    folium.CircleMarker(
                        location=[alat, alon],
                        radius=3,
                        color="#22c55e",
                        fill=True,
                        fill_color="#22c55e",
                        fill_opacity=0.6,
                        weight=1,
                        tooltip=f"{name} — {act.get('purpose', '?')}",
                    ).add_to(activity_layer)
                    n_activities += 1

    home_layer.add_to(m)
    activity_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m, n_homes, n_activities


# ── Routes ───────────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Population Visualizer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0;
           height: 100vh; display: flex; flex-direction: column; }
    header {
      padding: 10px 18px; background: #16213e;
      display: flex; align-items: center; gap: 14px;
      border-bottom: 1px solid #0f3460; flex-shrink: 0;
    }
    h1 { font-size: 1rem; font-weight: 600; letter-spacing: .4px; }
    #status { font-size: .8rem; color: #94a3b8; margin-left: auto; }
    button#refresh {
      background: transparent; border: 1px solid #475569;
      color: #94a3b8; padding: 5px 10px; border-radius: 5px;
      cursor: pointer; font-size: .85rem; transition: all .15s;
    }
    button#refresh:hover { border-color: #e94560; color: #e94560; }
    button#refresh:disabled { opacity: .4; cursor: not-allowed; }
    .legend {
      padding: 5px 18px; background: #16213e;
      border-bottom: 1px solid #0f3460;
      display: flex; gap: 18px; font-size: .75rem; flex-shrink: 0;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; }
    .rect { width: 13px; height: 9px; border-radius: 1px; }
    #mapframe { flex: 1; border: none; width: 100%; }
    #placeholder {
      flex: 1; display: flex; align-items: center;
      justify-content: center; font-size: 1rem; color: #4a5568; gap: 10px;
    }
    .spinner {
      width: 20px; height: 20px;
      border: 2px solid #334155; border-top-color: #3b82f6;
      border-radius: 50%; animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <h1>Population Visualizer — périmètre EMC² (453 communes)</h1>
    <span id="status">Chargement…</span>
    <button id="refresh" onclick="loadMap()" disabled>↻ Actualiser</button>
  </header>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#3b82f6"></div> Domicile</div>
    <div class="legend-item"><div class="dot" style="background:#22c55e"></div> Activité</div>
    <div class="legend-item"><div class="rect" style="background:#eab308;opacity:.6"></div> Desserte TC (Tisséo + TER)</div>
    <div class="legend-item"><div class="rect" style="background:#2563eb;opacity:.5"></div> Périmètre 453 communes (couronnes)</div>
    <div class="legend-item">
      <div style="width:18px;height:0;border-top:2px dashed #ef4444"></div>
      Haute-Garonne
    </div>
  </div>
  <div id="placeholder"><div class="spinner"></div> Génération de la carte…</div>
  <iframe id="mapframe" style="display:none"></iframe>

  <script>
    async function loadMap() {
      const btn    = document.getElementById('refresh');
      const status = document.getElementById('status');
      const frame  = document.getElementById('mapframe');
      const ph     = document.getElementById('placeholder');

      btn.disabled = true;
      ph.style.display = 'flex';
      frame.style.display = 'none';
      ph.innerHTML = '<div class="spinner"></div>&nbsp; Génération de la carte…';
      status.textContent = 'Chargement…';

      try {
        const res = await fetch('/map');
        if (!res.ok) {
          const msg = await res.text();
          ph.innerHTML = '⚠️ ' + msg;
          status.textContent = 'Erreur';
          return;
        }
        const info = res.headers.get('X-Population-Info') || '';
        const html = await res.text();
        frame.srcdoc = html;
        ph.style.display = 'none';
        frame.style.display = 'block';
        status.textContent = info;
      } catch (e) {
        ph.innerHTML = '⚠️ Erreur réseau : ' + e.message;
        status.textContent = 'Erreur';
      } finally {
        btn.disabled = false;
      }
    }

    window.addEventListener('DOMContentLoaded', loadMap);
  </script>
</body>
</html>"""


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/map")
def generate_map():
    files, current = _find_population_files()
    if not files:
        return Response(
            f"Aucun fichier population_*.json dans {current}",
            status=404,
            mimetype="text/plain",
        )

    m, n_homes, n_activities = _build_map(files)
    files_str = ", ".join(p.name for p in files)
    info = f"{n_homes} personnes · {n_activities} activités · {files_str}"

    return Response(
        m._repr_html_(),
        mimetype="text/html",
        headers={"X-Population-Info": info},
    )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"Population visualizer → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
