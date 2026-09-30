"""Tests of the residence ring (core/residence_zone.py).

The trait is **observed**, not imputed: a home is in a commune or it is not.
There is therefore nothing to lock on a distribution — what is locked here are
the properties without which the classification by fine zone CODE would not be legitimate:

- **the classification by code is identical to the classification by geometric membership.**
  It is the gate that authorises the whole trait: it was measured once by
  `make audit-couronnes` (trace `docs/traces/2026-08-24_couronne_equivalences/`), and it
  is replayed here at every test run, on the versioned resources. A one-off
  measurement goes stale; a test does not;
- **the categories are exactly those of the EMC² reference** — it is the join key
  of the synthesis page, a one-character divergence would make the axis disappear there
  without an error;
- **`hors périmètre` is not a ring** — confusing it with the 3rd led to publishing a
  stratum where 76% of the inhabitants were not in the survey;
- **nothing is guessed** — an unknown code returns `None`, a commune is never inferred from a
  sector, a missing resource or one of another version raises at load time;
- **the divergence from the metric classification is real and intended** — a test fixes it on
  a known point, so that a future "alignment" of the two is a deliberate act and not a
  side effect.

Offline, without the restricted-access PROGEDO data. Tests that require the
exported resources skip themselves when they have not been produced.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
from mobility_core.residence_zone import (
    COMMUNE_TRAIT_KEY,
    DEFAULT_COMMUNE_TABLE,
    DEFAULT_GEOJSON,
    DEFAULT_TABLE,
    RESOURCE_VERSION,
    SECTOR_PREFIX_LEN,
    TRAIT_KEY,
    CommunalZones,
    CommuneTable,
    CouronneTable,
    ResidenceZoneError,
    ZoneCouronne,
    secteur_of,
)

COMMUNE_TABLE = DEFAULT_COMMUNE_TABLE
ZF_LAYER = DEFAULT_TABLE.parent / "zf_zones.gpkg"

needs_table = pytest.mark.skipif(not DEFAULT_TABLE.exists(),
                                 reason="zf_couronne.json missing (make communes-couronnes)")
needs_geojson = pytest.mark.skipif(not DEFAULT_GEOJSON.exists(),
                                   reason="couronne_perimetre.geojson missing")
needs_layer = pytest.mark.skipif(not ZF_LAYER.exists(),
                                 reason="zf_zones.gpkg missing (make zones)")


@pytest.fixture(scope="module")
def table() -> CouronneTable:
    return CouronneTable.load()


# ── The module, without a resource ───────────────────────────────────────────

def test_secteur_of_prend_les_trois_premiers_chiffres():
    assert SECTOR_PREFIX_LEN == 3
    assert secteur_of("218102000") == "218"
    assert secteur_of(101101000) == "101"
    # A code that is too short is not silently truncated: it has no sector.
    assert secteur_of("21") == ""
    assert secteur_of(None) == ""
    assert secteur_of("") == ""


def test_hors_perimetre_n_est_pas_une_couronne():
    assert OUT_OF_PERIMETER not in COURONNES


def test_les_cles_de_trait_sont_distinctes_et_lisibles():
    assert TRAIT_KEY == "residence_zone"
    assert COMMUNE_TRAIT_KEY == "residence_commune"
    assert TRAIT_KEY != COMMUNE_TRAIT_KEY


def test_un_secteur_a_deux_couronnes_est_refuse():
    """The table must be a FUNCTION of the sector, otherwise the classification is ambiguous."""
    zones = [ZoneCouronne("101101000", "101", "Toulouse", "31555", "Toulouse"),
             ZoneCouronne("101102000", "101", "1st ring", "31555", "Toulouse")]
    with pytest.raises(ResidenceZoneError, match="two rings"):
        CouronneTable(zones)


def test_ressource_absente_leve_au_chargement(tmp_path):
    with pytest.raises(ResidenceZoneError, match="missing"):
        CouronneTable.load(tmp_path / "pas_la.json")


def test_ressource_d_une_autre_version_est_refusee(tmp_path):
    """An outdated resource is not served "as best it can": it is rejected."""
    path = tmp_path / "zf_couronne.json"
    path.write_text(json.dumps({"version": "zc0", "zones": []}), encoding="utf-8")
    with pytest.raises(ResidenceZoneError, match="version"):
        CouronneTable.load(path)


def test_modalite_inattendue_est_refusee(tmp_path):
    path = tmp_path / "zf_couronne.json"
    path.write_text(json.dumps({
        "version": RESOURCE_VERSION,
        "zones": [{"zf": "101101000", "secteur": "101", "couronne": "4eme couronne",
                   "insee": "31555", "commune": "Toulouse"}]}), encoding="utf-8")
    with pytest.raises(ResidenceZoneError, match="unexpected"):
        CouronneTable.load(path)


# ── The published resource ───────────────────────────────────────────────────

@needs_table
def test_la_table_couvre_les_785_zones_et_les_88_secteurs(table: CouronneTable):
    assert len(table) == 785
    assert len(table.secteurs) == 88
    assert table.meta["n_zones"] == 785
    assert table.meta["n_secteurs"] == 88
    assert set(table.meta["counts"]) == set(COURONNES)
    assert sum(table.meta["counts"].values()) == 785


@needs_table
def test_les_codes_de_zone_fine_sont_bien_formes(table: CouronneTable):
    for zf, zone in table._by_zf.items():  # noqa: SLF001 - internal shape check
        assert len(zf) == 9 and zf.isdigit(), zf
        assert zone.secteur == zf[:SECTOR_PREFIX_LEN]
        assert zone.couronne in COURONNES
        assert len(zone.insee) == 5 and zone.insee.isdigit(), zone.insee
        assert zone.commune


@needs_table
def test_la_lecture_par_code_et_par_secteur_concordent(table: CouronneTable):
    for zf, zone in table._by_zf.items():  # noqa: SLF001
        assert table.couronne_of_zf(zf) == zone.couronne
        assert table.couronne_of_secteur(zone.secteur) == zone.couronne


@needs_table
def test_rien_ne_se_devine_hors_de_la_table(table: CouronneTable):
    # Unknown code but known sector: the ring comes from the sector, which is the true
    # carrier of the information in the survey.
    assert table.couronne_of_zf("101999999") == table.couronne_of_secteur("101")
    # The COMMUNE, however, is not inferred from a sector: several communes per sector.
    assert table.commune_of_zf("101999999") is None
    # Unknown sector: nothing.
    assert table.couronne_of_zf("999999999") is None
    assert table.couronne_of_zf(None) is None
    assert table.commune_of_zf("") is None


@needs_table
@pytest.mark.skipif(not COMMUNE_TABLE.exists(), reason="commune_couronne.json missing")
def test_la_table_de_zones_est_coherente_avec_celle_des_communes(table: CouronneTable):
    """Two resources produced by the same export must tell the same story."""
    communes = json.loads(COMMUNE_TABLE.read_text(encoding="utf-8"))
    par_insee = {row["insee"]: row["couronne"] for row in communes["communes"]}
    vues = {}
    for zone in table._by_zf.values():  # noqa: SLF001
        assert par_insee.get(zone.insee) == zone.couronne, zone
        vues[zone.insee] = zone.commune
    # All communes of the scope carry at least one fine zone: otherwise the zone table
    # would describe a smaller scope than the commune table, silently.
    assert set(vues) == set(par_insee)
    assert len(vues) == communes["n_communes"] == 453


# ── The gate: code against geometry ──────────────────────────────────────

@needs_table
@needs_geojson
@needs_layer
def test_le_classement_par_code_egale_le_classement_geometrique(table: CouronneTable):
    """Gate B, replayed at every test run.

    Two independent paths: a mapping by the first three digits of the code,
    and a spatial join against the dissolution of the sectors. The trait rests
    entirely on their equality — measured once on 2026-08-24, locked here.
    """
    geopandas = pytest.importorskip("geopandas")
    pyproj = pytest.importorskip("pyproj")

    layer = geopandas.read_file(ZF_LAYER)
    zones = CommunalZones.load()
    to_wgs = pyproj.Transformer.from_crs(2154, 4326, always_xy=True)

    desaccords = []
    for row in layer.itertuples():
        par_code = table.couronne_of_zf(row.ZF)
        lon, lat = to_wgs.transform(row.XL93, row.YL93)
        par_geometrie = zones.classify(lat, lon)
        if par_code != par_geometrie:
            desaccords.append((str(row.ZF), par_code, par_geometrie))

    assert len(layer) == 785
    assert not desaccords, (
        f"{len(desaccords)} fine zone(s) classified differently depending on the path: "
        f"{desaccords[:5]}. The classification by code is no longer legitimate — redo "
        f"batch 0 of ticket 021 before serving this trait.")


@needs_geojson
def test_la_geometrie_rend_hors_perimetre_et_pas_une_couronne():
    zones = CommunalZones.load()
    # Capitole: the heart of the commune of Toulouse.
    assert zones.classify(43.6045, 1.4440) == "Toulouse"
    # A point clearly outside the survey scope (south of Ariège).
    assert zones.classify(43.10, 1.10) == OUT_OF_PERIMETER
    # An unknown point is not a category: empty, just as an empty probability cell
    # is not a 0.
    assert zones.classify(None, None) == ""
    assert zones.classify(43.6, None) == ""


@needs_geojson
def test_la_geometrie_couvre_les_quatre_couronnes():
    zones = CommunalZones.load()
    assert Counter(zones._names) == Counter(COURONNES)  # noqa: SLF001


# ── The accepted divergence from the metric classification ───────────────────

@needs_table
@needs_geojson
def test_le_classement_metrique_diverge_et_c_est_documente(table: CouronneTable):
    """A "false Toulousain": 7.57 km from the Capitole, but in Auzeville-Tolosane.

    The point comes from `docs/traces/2026-08-24_perimetre_population/agents_reclassement.csv`
    (persona 39705), one of the 66 that the 8 km disc names "Toulouse". This test fixes
    the divergence between the two definitions: it is intended, bounded to 34 s of terminal
    time on the worst observed pair, and closing it requires re-exporting the laws of
    `terminal_time_emc2.json` — another work item, not a side effect.
    """
    from mobility_core.geo_reference import residence_zone as metrique

    lat, lon = 43.535540410127076, 1.484410194839593
    assert metrique(lat, lon) == "Toulouse"
    assert CommunalZones.load().classify(lat, lon) == "1st ring"


# ── The sampling frame ──────────────────────────────────────────

needs_communes = pytest.mark.skipif(
    not COMMUNE_TABLE.exists(),
    reason="commune_couronne.json missing (make communes-couronnes)")


@needs_communes
def test_le_perimetre_compte_453_communes_sur_six_departements():
    """The figure is surprising, so it is locked: 453 communes, 6 departments.

    Cross-checked by area (5,428 km² against 5,400 km² published by the auat)
    and by the survey report itself.
    """
    table = CommuneTable.load()
    assert len(table) == 453
    departements = {c[:2] for c in table.communes()}
    assert departements == {"31", "32", "81", "82", "09", "11"}
    assert sum(table.counts().values()) == 453


@needs_communes
def test_le_cadre_haute_garonne_est_un_sous_ensemble_strict():
    """The light version of the frame: 346 communes, and the 3rd ring cut down."""
    table = CommuneTable.load()
    complet, cadre = table.counts(), table.counts(["31"])
    assert len(table.communes(["31"])) == 346
    assert cadre["Toulouse"] == complet["Toulouse"] == 1
    assert cadre["1st ring"] == complet["1st ring"]
    # This is where the limit bites, and it is published (perimetre-population.md, no. 6).
    assert cadre["3rd ring"] == 175 < complet["3rd ring"] == 275


@needs_communes
def test_un_cadre_vide_leve_au_lieu_de_retomber_sur_le_departement():
    """Without this safeguard, a typo would populate the whole department."""
    table = CommuneTable.load()
    with pytest.raises(ResidenceZoneError, match="empty sampling frame"):
        table.communes(["75"])
    with pytest.raises(ResidenceZoneError, match="empty list of departments"):
        table.communes([])


@needs_communes
def test_l_appartenance_au_perimetre_se_lit_sur_le_code_insee():
    table = CommuneTable.load()
    assert table.couronne_of_insee("31555") == "Toulouse"
    assert table.contains("09038")          # La Bastide-de-Besplas, 3rd ring (Ariège)
    assert not table.contains("75056")      # Paris
    assert table.couronne_of_insee(None) is None


@needs_communes
@needs_table
def test_les_deux_tables_racontent_la_meme_geographie(table: CouronneTable):
    """`zf_couronne.json` (fine zones) and `commune_couronne.json` (communes) are
    produced by the same export: their communes and their rings must match."""
    communes = CommuneTable.load()
    vues = {z.insee for z in table._by_zf.values()}  # noqa: SLF001
    assert vues == set(communes.communes())
    for zone in table._by_zf.values():  # noqa: SLF001
        assert communes.couronne_of_insee(zone.insee) == zone.couronne
