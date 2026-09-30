"""core/residence_zone.py — The residence ring, READ and not computed.

The EMC² 2023 survey splits its scope into four rings **by list of communes**
(`Toulouse`, `1ere couronne`, `2eme couronne`, `3eme couronne`), and it is against this
split that its modal shares are published. This module serves this split from the
**fine zone code** of the home, which `zone_resolver` already resolves.

⚠ **DO NOT CONFUSE WITH `geo_reference.residence_zone`.** The latter classifies a point
by its **distance to the hypercentre** (8 / 20 / 40 km). It is not the survey's
definition — ticket 020 measured 24.4% of misclassified personas and 66 false Toulousains
— and it only survives for **terminal time**, which classifies arbitrary points and
whose laws are stratified with it (`terminal_time_emc2.json`, `meta.crown_definition`).
The divergence between the two is accepted, bounded and documented (ticket 021): 34 s per
trip end on the worst observed pair. A caller that wants the ring of a **home**
reads the persona trait; it never calls the metric function again.

**The mapping goes through the code, not through a geometry.** The `ZF` code has 9
digits whose **first three are the sampling sector number** (`NUM_DTIR`), and
the sector carries the ring. Measured (ticket 021, batch 0): this classification is 100%
identical to the classification by geometric membership, on the 785 fine zones as on the
1,021 homes of the reference population.

*In passing*: `housing_type` splits the same code on **four** digits for its sector
fallbacks. The two partitions are identical — 88 classes on each side, and a
longer split can only refine — so the two modules do talk about the same
sectors despite the different length.

**Out of scope is not a ring.** A home outside the 453 communes receives
`population_reference.OUT_OF_PERIMETER`, never "3rd ring": it has no EMC² target
to be compared with, and confusing it with the outermost ring led to publishing a
stratum of which 76% of the inhabitants were not in the survey. Two footprints coexist —
the fine-zone layer (`zone_resolver.resolve()` returns `None` outside) and the dissolution
of the four rings (`CommunalZones`) — and batch 0 measured that they designate exactly
the same set. **The normative footprint is the second one**: it is the survey's.

Inputs/outputs are confined to the `load` methods of the two classes, in line with the
`mobility_core` architecture contract.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
from mobility_core.resources import restricted_data_path, restricted_resource_hint

# Trait keys in `traits_json`. The persona carries the survey LABEL
# (`1ere couronne`), not a technical key: it is what the trip log
# reads back, and what a human reads back in the population JSON.
TRAIT_KEY = "residence_zone"

# The commune is not decorative: it is what makes the classification auditable, and it
# survives a redrawing of the rings. The INSEE code comes with it, because a commune
# name is not a join key.
COMMUNE_TRAIT_KEY = "residence_commune"
INSEE_TRAIT_KEY = "residence_insee"

# Length of the sampling-sector prefix in the fine zone code (`218102000` → `218`).
SECTOR_PREFIX_LEN = 3

# Accepted resource version. A resource of another version is not served "as best
# it can": it is rejected, because a silently outdated residence classification is
# read as a modal share and not as a bug.
RESOURCE_VERSION = "zc1"

# RESTRICTED-ACCESS resource: present in the repository, never in the distributed package —
# it reproduces the survey's sampling plan (ticket 038). Resolved by
# `restricted_data_path`, which looks at `MOBILITY_CORE_EMC2_DATA_DIR` first.
TABLE_RESOURCE = "zf_couronne.json"

# Evaluated at import, for callers that test its existence. `load()` resolves again on
# each call: an environment variable set after the import is still taken into account.
DEFAULT_TABLE = restricted_data_path(TABLE_RESOURCE)
DEFAULT_GEOJSON = (Path(__file__).resolve().parent / "data"
                   / "couronne_perimetre.geojson")
DEFAULT_COMMUNE_TABLE = (Path(__file__).resolve().parent / "data"
                         / "commune_couronne.json")


class ResidenceZoneError(ValueError):
    """Resource missing, outdated or inconsistent. Never a silent fallback."""


def secteur_of(zf: object) -> str:
    """Sector prefix of a fine zone code. Empty string if the code is unusable."""
    code = str(zf or "").strip()
    return code[:SECTOR_PREFIX_LEN] if len(code) >= SECTOR_PREFIX_LEN else ""


@dataclass(frozen=True)
class ZoneCouronne:
    """A fine zone, reduced to what the residence trait needs."""

    zf: str
    secteur: str
    couronne: str
    insee: str
    commune: str


class CouronneTable:
    """`zf_couronne.json`: 785 fine zones → sector, ring, commune.

    Produced by `scripts/progedo_logit/export_commune_couronne.py` from the restricted-access
    GIS layer; the resource itself is versioned and mounted in the containers.
    """

    def __init__(self, zones: Sequence[ZoneCouronne], meta: dict | None = None) -> None:
        self._by_zf = {z.zf: z for z in zones}
        self._by_secteur: dict[str, str] = {}
        for zone in zones:
            known = self._by_secteur.setdefault(zone.secteur, zone.couronne)
            if known != zone.couronne:
                raise ResidenceZoneError(
                    f"sector {zone.secteur} mapped to two rings "
                    f"({known} and {zone.couronne}): the table is not a function.")
        self.meta = meta or {}

    # -- Loading ------------------------------------------------------------

    @classmethod
    def load(cls, resource: Path | None = None) -> CouronneTable:
        """Loads the table (the class's only I/O point)."""
        path = Path(resource) if resource else restricted_data_path(TABLE_RESOURCE)
        if not path.exists():
            raise ResidenceZoneError(
                "ring table missing: the ring of a home is not guessed.\n"
                + restricted_resource_hint(TABLE_RESOURCE))
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = str(payload.get("version", ""))
        if version != RESOURCE_VERSION:
            raise ResidenceZoneError(
                f"{path.name} is at version '{version}', expected '"
                f"{RESOURCE_VERSION}'. Rerun `make communes-couronnes`.")
        rows = payload.get("zones") or []
        if not rows:
            raise ResidenceZoneError(f"{path.name} carries no zone.")
        inconnues = sorted({row["couronne"] for row in rows} - set(COURONNES))
        if inconnues:
            raise ResidenceZoneError(
                f"unexpected rings in {path.name}: {inconnues}. The categories "
                f"must be exactly {list(COURONNES)}.")
        zones = [ZoneCouronne(zf=str(row["zf"]), secteur=str(row["secteur"]),
                              couronne=row["couronne"], insee=str(row["insee"]),
                              commune=str(row["commune"]))
                 for row in rows]
        meta = {k: v for k, v in payload.items() if k != "zones"}
        return cls(zones, meta)

    # -- Reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_zf)

    @property
    def secteurs(self) -> dict[str, str]:
        """`sector → ring`, the survey's 88 sampling sectors."""
        return dict(self._by_secteur)

    def zone(self, zf: object) -> ZoneCouronne | None:
        """The row of a fine zone. `None` if the code is unknown — no guessing."""
        return self._by_zf.get(str(zf or "").strip())

    def couronne_of_zf(self, zf: object) -> str | None:
        """Ring of a fine zone code, by its row then by its sector.

        The sector fallback is not an approximation: the sector IS the carrier of
        the ring in the survey, the zone row is only a projection of it. It serves
        codes that a more recent zone layer would know without the table having
        seen them yet. `None` when even the sector is unknown.
        """
        zone = self.zone(zf)
        if zone is not None:
            return zone.couronne
        return self._by_secteur.get(secteur_of(zf))

    def couronne_of_secteur(self, secteur: object) -> str | None:
        """Ring of a sampling sector. `None` if the sector is unknown."""
        return self._by_secteur.get(str(secteur or "").strip())

    def commune_of_zf(self, zf: object) -> tuple[str, str] | None:
        """`(insee, commune)` of a fine zone. `None` if the code is unknown.

        No sector fallback here: a sector covers several communes, and returning
        "a commune of the sector" would be an invention.
        """
        zone = self.zone(zf)
        return (zone.insee, zone.commune) if zone is not None else None


class CommuneTable:
    """`commune_couronne.json`: the 453 communes of the scope and their ring.

    Serves two uses that nothing else covers:

    - the **sampling frame** of population generation — which communes the
      eqasim pipeline is allowed to populate (ticket 026);
    - the **membership test** for the survey scope from an INSEE code, without
      geometry and without resolving a fine zone.

    The two are not the same: the frame may be a subset (Haute-Garonne
    version of ticket 026), the scope is always the 453.
    """

    def __init__(self, communes: dict[str, str], meta: dict | None = None) -> None:
        self._couronne_by_insee = dict(communes)
        self.meta = meta or {}

    @classmethod
    def load(cls, resource: Path | None = None) -> CommuneTable:
        path = Path(resource) if resource else DEFAULT_COMMUNE_TABLE
        if not path.exists():
            raise ResidenceZoneError(
                f"commune table missing: {path} (`make communes-couronnes`).")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("communes") or []
        if not rows:
            raise ResidenceZoneError(f"{path.name} carries no commune.")
        inconnues = sorted({r["couronne"] for r in rows} - set(COURONNES))
        if inconnues:
            raise ResidenceZoneError(
                f"unexpected rings in {path.name}: {inconnues}.")
        communes = {str(r["insee"]).zfill(5): r["couronne"] for r in rows}
        meta = {k: v for k, v in payload.items() if k != "communes"}
        return cls(communes, meta)

    def __len__(self) -> int:
        return len(self._couronne_by_insee)

    def couronne_of_insee(self, insee: object) -> str | None:
        """Ring of a commune. `None` if it is not in the scope."""
        return self._couronne_by_insee.get(str(insee or "").strip().zfill(5))

    def contains(self, insee: object) -> bool:
        return self.couronne_of_insee(insee) is not None

    def communes(self, departments: Sequence[str] | None = None) -> list[str]:
        """INSEE codes of the scope, optionally restricted to departments.

        `departments=None` returns the 453 — the whole survey scope.
        `departments=["31"]` returns the sampling frame of the Haute-Garonne version of
        ticket 026 (346 communes). The restriction is a WORKING CHOICE, never a
        definition: the scope remains the 453, and it is what serves as the admission
        filter at load time.
        """
        codes = sorted(self._couronne_by_insee)
        if departments is None:
            return codes
        prefixes = tuple(str(d).zfill(2) for d in departments)
        if not prefixes:
            raise ResidenceZoneError(
                "empty list of departments: refusing to return an ambiguous sampling "
                "frame. Pass `None` for the whole scope.")
        retenus = [c for c in codes if c.startswith(prefixes)]
        if not retenus:
            raise ResidenceZoneError(
                f"no commune of the scope in departments {list(prefixes)} — "
                f"empty sampling frame. Without this safeguard, the pipeline would silently "
                f"fall back on the whole department.")
        return retenus

    def counts(self, departments: Sequence[str] | None = None) -> dict[str, int]:
        """Number of communes per ring, for the requested frame."""
        retenus = self.communes(departments)
        out = {z: 0 for z in COURONNES}
        for insee in retenus:
            out[self._couronne_by_insee[insee]] += 1
        return out


class CommunalZones:
    """Classifies a point by MEMBERSHIP of a ring, not by distance.

    It is the survey's definition, and the normative footprint of out-of-scope. It
    lived in `scripts/data/population/audit_perimetre.py` (ticket 020); it moved
    here when a second caller appeared — the post-processing of ticket 021 — because
    two copies of a reference classification end up diverging.

    A point outside the four rings receives `hors périmètre`; an unknown point (`None`)
    receives the empty string, which is not a category.
    """

    def __init__(self, names: Sequence[str], geometries) -> None:
        from shapely import STRtree
        from shapely import points as shapely_points

        self._names = [str(name) for name in names]
        self._tree = STRtree(list(geometries))
        self._points = shapely_points

    @classmethod
    def load(cls, geojson: Path | None = None) -> CommunalZones:
        """Loads the ring geometry (the class's only I/O point)."""
        path = Path(geojson) if geojson else DEFAULT_GEOJSON
        if not path.exists():
            raise ResidenceZoneError(
                f"ring geometry missing: {path} "
                f"(`make communes-couronnes`).")
        try:
            import geopandas as gpd
        except ImportError as exc:  # pragma: no cover - dépend de l'image
            raise ResidenceZoneError(
                f"geopandas required to read {path.name}: {exc}") from exc
        layer = gpd.read_file(path).to_crs(4326)
        inconnues = sorted(set(map(str, layer["couronne"])) - set(COURONNES))
        if inconnues:
            raise ResidenceZoneError(
                f"unexpected rings in {path.name}: {inconnues}.")
        return cls(list(layer["couronne"]), list(layer.geometry))

    def classify(self, lat: float | None, lon: float | None) -> str:
        """Ring of a point, `hors périmètre` outside, empty string if the point is missing."""
        if lat is None or lon is None:
            return ""
        hits = self._tree.query(self._points([lon], [lat]), predicate="within")
        # STRtree.query returns (input indices, tree indices) in shapely 2.
        indices = hits[1] if getattr(hits, "ndim", 1) == 2 else hits
        for index in indices:
            return self._names[int(index)]
        return OUT_OF_PERIMETER
