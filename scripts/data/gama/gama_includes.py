"""Complementary recipes for `services/GAMA/CityTransport/includes/`, and their comparison.

    services/llm-agents/.venv/bin/python scripts/data/gama/gama_includes.py <commande> [options]
    make gama-includes INCLUDES_OUT=/chemin/de/sortie [ANNEXES=1]      # chains all the recipes
    make gama-includes-compare INCLUDES_REF=… INCLUDES_OUT=…           # compares two directories

`includes/` is not published: each layer must be rebuilt from a versioned recipe. The
layers READ by the model (`Settings.gaml`) already have their recipe — `export_gtfs_layers.py`
(routes, stops), `export_trip_info.py` (trip_info.json + shape_lookup.json, read by the runtime),
`export_perimetre_shapefile.py` (perimetre_453). This file adds:

- `perimetre`  : `export_perimetre_shapefile.py` to a chosen directory (the script has no output
                 option; its two module constants are replaced, without modifying it);
- `osm-p95`    : the `Toulouse_bbox_p95.osm.pbf` clip — `osmium extract --bbox` on the
                 Geofabrik Midi-Pyrénées extract of 2022-01-01, bbox P95 RP2022 dept 31 (recipe
                 `make clip-osm` of commit 4dccd17, 2026-05-04, gone since);
- `batiments`  : `building.shp` — the BATI/BATIMENT theme of the IGN BD TOPO, département 31,
                 extracted from the « TOUSTHEMES_SHP_LAMB93 » delivery;
- `fond-carte` : `toulouse_map.png` — CartoDB Positron tiles via contextily, 12 × 12 inches at
                 1 000 dpi over the 30 km rectangle (function `save_clean_map` of
                 `scripts/infra/otp_shape.ipynb`);
- `comparer`   : compares a regenerated directory with a reference directory, layer by layer.

The last three layers are read by NO model (`building.shp` and `toulouse_map.png` are
commented out in Settings.gaml / City.gaml, the pbf is loaded by nobody): `make gama-includes`
produces them only with `ANNEXES=1`.

Exit codes: 0 written, 1 resource missing (source, tool), 2 invariant contradicted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "services" / "llm-agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INCLUDES = REPO_ROOT / "services" / "GAMA" / "CityTransport" / "includes"

CODE_RESSOURCE = 1
CODE_REFUS = 2

# ── Recipe parameters (recovered values, with their source) ─────────────────────────────────────
# BBox P95 RP2022 dept 31, osmium format min_lon,min_lat,max_lon,max_lat — Makefile of commit
# 4dccd17 (`BBOX = 0.8094,43.1237,1.6139,43.7445`), from notebooks/mobility_bbox_analysis.ipynb.
BBOX_P95 = "0.8094,43.1237,1.6139,43.7445"
OSM_SOURCE_DEFAUT = REPO_ROOT / "services" / "eqasim-toulouse" / "data" / "osm_toulouse" / "midi-pyrenees-220101.osm.pbf"
OSM_SOURCE_URL = "https://download.geofabrik.de/europe/france/midi-pyrenees-220101.osm.pbf"

# BD TOPO: the URL follows the Géoportail pattern; the edition is a parameter. The one of the
# original includes (copied on 2026-03-24, 1 177 680 buildings) is NOT one of the two present on
# disk (2024-09-15: 1 163 493; 2025-03-15: 1 163 359); 98.4 % of its geometries are those of
# 2025-03-15 byte for byte, so it is a more recent edition. Published candidates (HEAD of
# 2026-09-28): 3-5 / 2025-09-15 (359 534 893 B) and 3-5 / 2025-12-15 (359 904 394 B); 3-4 /
# 2025-06-15 is no longer served. Not checked: 360 MB to download.
BDTOPO_URL = ("https://data.geopf.fr/telechargement/download/BDTOPO/"
              "BDTOPO_{version}_TOUSTHEMES_SHP_LAMB93_D{dep}_{edition}/"
              "BDTOPO_{version}_TOUSTHEMES_SHP_LAMB93_D{dep}_{edition}.7z")

# 30 km rectangle (TOULOUSE_OSM_ROUTES_30K_BBOX) — `newbox_gdf2` of scripts/infra/otp_shape.ipynb.
FOND_CARTE_BBOX = {"lat_min": 43.336, "lat_max": 43.868, "lon_min": 1.085, "lon_max": 1.815}
FOND_CARTE_FOURNISSEUR = "CartoDB.Positron"

log = logging.getLogger("gama_includes")


def _configurer_journal() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _court(chemin: Path) -> str:
    try:
        return str(Path(chemin).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(chemin)


def sha256(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


class Etape:
    """Step start / end with duration; the end states success or failure explicitly."""

    def __init__(self, nom: str):
        self.nom = nom

    def __enter__(self):
        self.t0 = time.monotonic()
        log.info("[%s] start", self.nom)
        return self

    def __exit__(self, typ, val, tb):
        duree = time.monotonic() - self.t0
        if typ is None:
            log.info("[%s] end — success in %.1f s", self.nom, duree)
        else:
            log.error("[%s] end — FAILURE in %.1f s: %s", self.nom, duree, val)
        return False


# ── Raw reading of a .shp (works without .shx or .dbf, like the original building.shp) ──────────

def stats_shp_brut(chemin: Path, empreintes: bool = False) -> dict:
    """Number of records, type, header extent; optional: fingerprint of each geometry."""
    with open(chemin, "rb") as f:
        entete = f.read(100)
        longueur = struct.unpack(">i", entete[24:28])[0] * 2
        type_forme = struct.unpack("<i", entete[32:36])[0]
        emprise = [round(v, 6) for v in struct.unpack("<4d", entete[36:68])]
        n, vides, hachages = 0, 0, set()
        pos = 100
        while pos < longueur:
            _num, mots = struct.unpack(">ii", f.read(8))
            corps = f.read(mots * 2)
            if struct.unpack("<i", corps[:4])[0] == 0:
                vides += 1
            if empreintes:
                hachages.add(hashlib.blake2b(corps, digest_size=12).digest())
            n += 1
            pos += 8 + mots * 2
    out = {"enregistrements": n, "geometries_vides": vides, "type_forme": type_forme, "emprise": emprise}
    if empreintes:
        out["_empreintes"] = hachages
    return out


# ── perimetre ─────────────────────────────────────────────────────────────────────────────────────

def cmd_perimetre(args) -> int:
    from scripts.data.gama import export_perimetre_shapefile as recette

    sortie = args.sortie.resolve()
    with Etape("perimetre"):
        sortie.mkdir(parents=True, exist_ok=True)
        recette.INCLUDES = sortie
        recette.OUT = sortie / "perimetre_453.shp"
        code = recette.main()
        if code != 0:
            log.error("export_perimetre_shapefile.py returned %s (sortie=%s)", code, sortie)
            return code
        s = stats_shp_brut(recette.OUT)
        log.info("[perimetre] %s: %d feature(s) written, extent %s — success",
                 _court(recette.OUT), s["enregistrements"], s["emprise"])
        if s["enregistrements"] != 1:
            log.error("[ALARME] perimetre_453.shp must carry ONE feature, carries %d (%s)",
                      s["enregistrements"], recette.OUT)
            return CODE_REFUS
    return 0


# ── osm-p95 ───────────────────────────────────────────────────────────────────────────────────────

def _osmium_info(chemin: Path) -> dict:
    brut = subprocess.run(["osmium", "fileinfo", "-e", "-j", str(chemin)],
                          check=True, capture_output=True, text=True).stdout
    d = json.loads(brut)
    return {"emprise_donnees": d["data"]["bbox"], "horodatage": d["data"]["timestamp"],
            "compte": {k: d["data"]["count"][k] for k in ("nodes", "ways", "relations")}}


def cmd_osm_p95(args) -> int:
    source, sortie = args.source, args.sortie / "Toulouse_bbox_p95.osm.pbf"
    if shutil.which("osmium") is None:
        log.error("osmium introuvable (brew install osmium-tool / apt install osmium-tool) — "
                  "source=%s bbox=%s", source, args.bbox)
        return CODE_RESSOURCE
    if not source.exists():
        log.error("OSM source missing: %s — to download from %s (≈ 280 MB)", source, OSM_SOURCE_URL)
        return CODE_RESSOURCE
    with Etape("osm-p95"):
        args.sortie.mkdir(parents=True, exist_ok=True)
        log.info("[osm-p95] osmium extract --bbox %s %s", args.bbox, _court(source))
        r = subprocess.run(["osmium", "extract", "--bbox", args.bbox, str(source),
                            "-o", str(sortie), "--overwrite"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            log.error("osmium extract failed (code %s, bbox=%s, source=%s): %s",
                      r.returncode, args.bbox, source, r.stderr.strip())
            return CODE_REFUS
        info = _osmium_info(sortie)
        log.info("[osm-p95] %s written: %s nodes, %s ways, %s relations; sha256 %s — success",
                 _court(sortie), info["compte"]["nodes"], info["compte"]["ways"],
                 info["compte"]["relations"], sha256(sortie)[:16])
    return 0


# ── batiments ─────────────────────────────────────────────────────────────────────────────────────

def _membres_7z(archive: Path) -> list[str]:
    try:
        import py7zr
        with py7zr.SevenZipFile(archive) as a:
            return a.getnames()
    except ImportError:
        pass
    for outil in (["bsdtar", "-tf"], ["tar", "-tf"], ["7z", "l", "-ba", "-slt"]):
        if shutil.which(outil[0]) is None:
            continue
        r = subprocess.run([*outil, str(archive)], capture_output=True, text=True, check=False)
        if r.returncode == 0 and "BATI" in r.stdout:
            if outil[0] == "7z":
                return [l[7:] for l in r.stdout.splitlines() if l.startswith("Path = ")]
            return r.stdout.splitlines()
    raise RuntimeError("no 7z reader (py7zr, bsdtar, 7z) can list the archive")


def _extraire_7z(archive: Path, membres: list[str], dossier: Path) -> None:
    try:
        import py7zr
        with py7zr.SevenZipFile(archive) as a:
            a.extract(dossier, membres)
        return
    except ImportError:
        pass
    for outil in ("bsdtar", "tar"):
        if shutil.which(outil):
            r = subprocess.run([outil, "-xf", str(archive), "-C", str(dossier), *membres],
                               capture_output=True, text=True, check=False)
            if r.returncode == 0:
                return
    if shutil.which("7z"):
        subprocess.run(["7z", "x", "-y", f"-o{dossier}", str(archive), *membres], check=True,
                       capture_output=True)
        return
    raise RuntimeError("no 7z extractor available (py7zr, bsdtar, 7z)")


def cmd_batiments(args) -> int:
    archive = args.archive
    if archive is None or not archive.exists():
        log.error("BD TOPO archive missing: %s — to download, for example %s", archive,
                  BDTOPO_URL.format(version="3-5", dep="031", edition="2025-12-15"))
        return CODE_RESSOURCE
    with Etape("batiments"):
        membres = [m for m in _membres_7z(archive) if "/BATI/BATIMENT." in m]
        if not any(m.endswith(".shp") for m in membres):
            log.error("[ALARME] no BATI/BATIMENT.shp in %s (%d BATI member(s) read)",
                      archive, len(membres))
            return CODE_REFUS
        log.info("[batiments] %s: %d BATI/BATIMENT.* member(s)", archive.name, len(membres))
        args.sortie.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.sortie, prefix=".bdtopo_") as tmp:
            _extraire_7z(archive, membres, Path(tmp))
            ecrits = []
            for m in membres:
                ext = m.rsplit(".", 1)[1]
                if args.shp_seul and ext != "shp":
                    continue
                cible = args.sortie / f"building.{ext}"
                shutil.move(str(Path(tmp) / m), cible)
                ecrits.append(cible.name)
        s = stats_shp_brut(args.sortie / "building.shp")
        log.info("[batiments] wrote %s: %d building(s), %d empty geometry(ies), type %d, extent %s — success",
                 ", ".join(ecrits), s["enregistrements"], s["geometries_vides"], s["type_forme"], s["emprise"])
    return 0


# ── fond-carte ────────────────────────────────────────────────────────────────────────────────────

def cmd_fond_carte(args) -> int:
    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from shapely.geometry import box
    except ImportError as e:
        log.error("missing dependency for fond-carte: %s", e)
        return CODE_RESSOURCE
    b = FOND_CARTE_BBOX
    sortie = args.sortie / "toulouse_map.png"
    if not args.forcer:
        # Observed on 2026-09-28: CARTO tiles served without a key carry the watermark
        # « API KEY REQUIRED » (carto.com/basemaps/apikey). The rendering "succeeds" and produces an
        # image of the right size — an empty success. So it is refused by default.
        log.error("[ALARME] fond-carte not reproducible as is: the %s tiles now require "
                  "an API key (watermark « API KEY REQUIRED » observed on 2026-09-28); bbox=%s, dpi=%d. "
                  "--forcer to produce an image anyway, to be checked by eye.",
                  args.fournisseur, b, args.dpi)
        return CODE_RESSOURCE
    with Etape("fond-carte"):
        args.sortie.mkdir(parents=True, exist_ok=True)
        fournisseur = ctx.providers
        for part in args.fournisseur.split("."):
            fournisseur = fournisseur[part]
        gdf = gpd.GeoDataFrame({"source": ["Equasim"]},
                               geometry=[box(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])],
                               crs="EPSG:4326").to_crs(epsg=3857)
        # Taken as is from `save_clean_map` (scripts/infra/otp_shape.ipynb, cell 20).
        fig = plt.figure(frameon=False, figsize=(12, 12))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        gdf.plot(ax=ax, edgecolor="blue", facecolor="none", linewidth=1.5)
        try:
            ctx.add_basemap(ax=ax, source=fournisseur)
        except Exception as e:  # noqa: BLE001 — réseau, quota du fournisseur de tuiles
            log.error("tiles %s could not be fetched (bbox=%s): %s", args.fournisseur, b, e)
            plt.close(fig)
            return CODE_RESSOURCE
        x0, y0, x1, y1 = gdf.total_bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        plt.savefig(sortie, dpi=args.dpi, bbox_inches=0, pad_inches=0)
        plt.close(fig)
        log.info("[fond-carte] %s written (%s, %d dpi) — success", _court(sortie), args.fournisseur, args.dpi)
    return 0


# ── comparer ──────────────────────────────────────────────────────────────────────────────────────

COLONNES_PROVENANCE = {"exporte_le"}  # attributes carrying the export date, not the data


def _cmp_shapefile(ref: Path, cand: Path) -> dict:
    composants = {}
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        r, c = ref.with_suffix("." + ext), cand.with_suffix("." + ext)
        if r.exists() or c.exists():
            composants[ext] = {"ref": sha256(r)[:16] if r.exists() else None,
                               "cand": sha256(c)[:16] if c.exists() else None}
    dbf_r, dbf_c = ref.with_suffix(".dbf"), cand.with_suffix(".dbf")
    if dbf_r.exists() and dbf_c.exists():
        # Bytes 1-3 of the dBase header = date of last write (YY MM DD): they change at
        # every export without the table changing. They are neutralised to judge the content.
        br, bc = bytearray(dbf_r.read_bytes()), bytearray(dbf_c.read_bytes())
        br[1:4] = bc[1:4] = b"\0\0\0"
        composants["dbf"]["hors_date_entete"] = ("identique" if hashlib.sha256(br).digest()
                                                 == hashlib.sha256(bc).digest() else "différent")
    sr, sc = stats_shp_brut(ref, empreintes=True), stats_shp_brut(cand, empreintes=True)
    er, ec = sr.pop("_empreintes"), sc.pop("_empreintes")
    out = {"ref": sr, "cand": sc, "sha256": composants,
           "geometries_communes": len(er & ec), "geometries_ref_seules": len(er - ec),
           "geometries_cand_seules": len(ec - er)}
    if ref.with_suffix(".dbf").exists() and cand.with_suffix(".dbf").exists():
        import pyogrio
        dr = pyogrio.read_dataframe(ref, read_geometry=False)
        dc = pyogrio.read_dataframe(cand, read_geometry=False)
        out["colonnes_ref"], out["colonnes_cand"] = list(dr.columns), list(dc.columns)
        commun = [c for c in dr.columns if c in dc.columns and c not in COLONNES_PROVENANCE]
        if len(dr) == len(dc):
            egal = dr[commun].reset_index(drop=True).equals(dc[commun].reset_index(drop=True))
            out["attributs_egaux_hors_provenance"] = bool(egal)
        info_r = pyogrio.read_info(ref)
        info_c = pyogrio.read_info(cand)
        out["crs"] = {"ref": info_r.get("crs"), "cand": info_c.get("crs")}
    tout_egal = all(v["ref"] == v["cand"] for v in composants.values())
    out["seul_ecart_date_entete_dbf"] = (not tout_egal and all(
        v["ref"] == v["cand"] or v.get("hors_date_entete") == "identique" for v in composants.values()))
    if tout_egal:
        verdict = "identique"
    elif (sr == sc and out["geometries_ref_seules"] == 0 and out["geometries_cand_seules"] == 0
          and out.get("attributs_egaux_hors_provenance", True)
          and out.get("colonnes_ref") == out.get("colonnes_cand")):
        verdict = "équivalent"
    else:
        verdict = "différent"
    out["verdict"] = verdict
    return out


def _canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cmp_trip_info(ref: Path, cand: Path) -> dict:
    r, c = json.loads(ref.read_text()), json.loads(cand.read_text())
    def resume(d):
        dates = d["calendar"]["dates"]
        return {"cles": sorted(d), "courses": len(d["trip_list"]), "dates": len(dates),
                "premiere": dates[0] if dates else None, "derniere": dates[-1] if dates else None,
                "contenu": _canon(d)[:16]}
    rr, rc = resume(r), resume(c)
    ids_r = {t[0] if isinstance(t, list) else json.dumps(t, sort_keys=True) for t in r["trip_list"]}
    ids_c = {t[0] if isinstance(t, list) else json.dumps(t, sort_keys=True) for t in c["trip_list"]}
    sha_r, sha_c = sha256(ref), sha256(cand)
    verdict = ("identique" if sha_r == sha_c else
               "équivalent" if rr["contenu"] == rc["contenu"] else "différent")
    return {"ref": rr, "cand": rc, "sha256": {"ref": sha_r[:16], "cand": sha_c[:16]},
            "courses_communes": len(ids_r & ids_c), "courses_ref_seules": len(ids_r - ids_c),
            "courses_cand_seules": len(ids_c - ids_r), "verdict": verdict}


def _cmp_shape_lookup(ref: Path, cand: Path) -> dict:
    r, c = json.loads(ref.read_text()), json.loads(cand.read_text())
    def utile(d):  # excluding timestamp and witnesses (fingerprints of neighbouring files)
        d = dict(d)
        d.pop("genere_le", None)
        conc = dict(d.get("concordance", {}))
        conc.pop("temoins", None)
        d["concordance"] = conc
        return d
    ur, uc = utile(r), utile(c)
    sha_r, sha_c = sha256(ref), sha256(cand)
    verdict = ("identique" if sha_r == sha_c else
               "équivalent" if _canon(ur) == _canon(uc) else "différent")
    return {"ref": r.get("concordance", {}).get("comptes"), "cand": c.get("concordance", {}).get("comptes"),
            "reseaux_egaux": r.get("reseaux") == c.get("reseaux"),
            "table_egale": r.get("table") == c.get("table"), "arrets_egaux": r.get("arrets") == c.get("arrets"),
            "sha256": {"ref": sha_r[:16], "cand": sha_c[:16]}, "verdict": verdict}


def _cmp_pbf(ref: Path, cand: Path) -> dict:
    sha_r, sha_c = sha256(ref), sha256(cand)
    out = {"sha256": {"ref": sha_r[:16], "cand": sha_c[:16]}}
    if shutil.which("osmium"):
        out["ref"], out["cand"] = _osmium_info(ref), _osmium_info(cand)
    out["verdict"] = ("identique" if sha_r == sha_c else
                      "équivalent" if out.get("ref") == out.get("cand") else "différent")
    return out


def _cmp_png(ref: Path, cand: Path) -> dict:
    from PIL import Image, ImageChops, ImageStat
    Image.MAX_IMAGE_PIXELS = None
    a, b = Image.open(ref), Image.open(cand)
    out = {"ref": {"taille": a.size, "mode": a.mode}, "cand": {"taille": b.size, "mode": b.mode},
           "sha256": {"ref": sha256(ref)[:16], "cand": sha256(cand)[:16]}}
    if a.size == b.size:
        pa = a.convert("RGB").resize((1500, 1500))
        pb = b.convert("RGB").resize((1500, 1500))
        diff = ImageStat.Stat(ImageChops.difference(pa, pb)).mean
        out["ecart_moyen_pixel_0_255"] = round(sum(diff) / 3, 2)
    out["verdict"] = ("identique" if out["sha256"]["ref"] == out["sha256"]["cand"] else
                      "équivalent" if out.get("ecart_moyen_pixel_0_255", 99) < 3 else "différent")
    return out


COUCHES = [  # (file, comparator, read by GAML?)
    ("perimetre_453.shp", _cmp_shapefile, "oui (Settings.gaml)"),
    ("routes.shp", _cmp_shapefile, "oui (Settings.gaml)"),
    ("stops.shp", _cmp_shapefile, "oui (Settings.gaml)"),
    ("trip_info.json", _cmp_trip_info, "oui (Settings.gaml)"),
    ("shape_lookup.json", _cmp_shape_lookup, "non — runtime Python (settings.gtfs.shape_lookup_file)"),
    ("Toulouse_bbox_p95.osm.pbf", _cmp_pbf, "non"),
    ("building.shp", _cmp_shapefile, "non (commenté)"),
    ("toulouse_map.png", _cmp_png, "non (commenté)"),
]


def cmd_comparer(args) -> int:
    resultats, comptes = {}, {"identique": 0, "équivalent": 0, "différent": 0, "absent": 0}
    with Etape("comparer"):
        for nom, comparer, lue in COUCHES:
            ref, cand = args.reference / nom, args.candidat / nom
            if not ref.exists() or not cand.exists():
                resultats[nom] = {"lue": lue, "verdict": "absent",
                                  "ref_present": ref.exists(), "cand_present": cand.exists()}
                comptes["absent"] += 1
                log.warning("[comparer] %s: absent (reference %s, candidate %s)",
                            nom, ref.exists(), cand.exists())
                continue
            t0 = time.monotonic()
            try:
                r = comparer(ref, cand)
            except Exception as e:
                log.error("[comparer] %s: comparison impossible (ref=%s, cand=%s): %s", nom, ref, cand, e)
                raise
            r["lue"] = lue
            resultats[nom] = r
            comptes[r["verdict"]] += 1
            log.info("[comparer] %s : %s (%.1f s)", nom, r["verdict"], time.monotonic() - t0)
        log.info("[comparer] summary: %s", comptes)
    texte = json.dumps({"reference": str(args.reference), "candidat": str(args.candidat),
                        "bilan": comptes, "couches": resultats}, ensure_ascii=False, indent=1, default=str)
    print(texte)
    if args.json:
        args.json.write_text(texte, encoding="utf-8")
    return 0


def main(argv=None) -> int:
    _configurer_journal()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)

    s = sub.add_parser("perimetre", help="perimetre_453.shp to --sortie")
    s.add_argument("--sortie", type=Path, default=INCLUDES)
    s.set_defaults(fn=cmd_perimetre)

    s = sub.add_parser("osm-p95", help="Toulouse_bbox_p95.osm.pbf (osmium extract --bbox)")
    s.add_argument("--source", type=Path, default=OSM_SOURCE_DEFAUT)
    s.add_argument("--bbox", default=BBOX_P95)
    s.add_argument("--sortie", type=Path, default=INCLUDES)
    s.set_defaults(fn=cmd_osm_p95)

    s = sub.add_parser("batiments", help="building.shp from a BD TOPO delivery (7z)")
    s.add_argument("--archive", type=Path, default=None)
    s.add_argument("--sortie", type=Path, default=INCLUDES)
    s.add_argument("--shp-seul", action="store_true",
                   help="write only building.shp, like the original includes (without .dbf/.shx/.prj)")
    s.set_defaults(fn=cmd_batiments)

    s = sub.add_parser("fond-carte", help="toulouse_map.png (contextily)")
    s.add_argument("--sortie", type=Path, default=INCLUDES)
    s.add_argument("--fournisseur", default=FOND_CARTE_FOURNISSEUR)
    s.add_argument("--dpi", type=int, default=1000)
    s.add_argument("--forcer", action="store_true",
                   help="produce the image despite the known tile defect (to be checked by eye)")
    s.set_defaults(fn=cmd_fond_carte)

    s = sub.add_parser("comparer", help="compares two includes directories")
    s.add_argument("--reference", type=Path, default=INCLUDES)
    s.add_argument("--candidat", type=Path, required=True)
    s.add_argument("--json", type=Path, default=None)
    s.set_defaults(fn=cmd_comparer)

    args = p.parse_args(argv)
    for attr in ("sortie", "candidat"):
        if getattr(args, attr, None) is not None:
            setattr(args, attr, getattr(args, attr).resolve())
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
