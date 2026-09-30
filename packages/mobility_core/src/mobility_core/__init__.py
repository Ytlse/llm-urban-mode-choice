"""mobility_core — the EMC² Toulouse survey domain, with no LLM dependency.

Residence rings, fine zones, mode hierarchy, bike equipment, housing
type, individual propensities and framing of the surveyed population. Each module
reads a frozen resource in ``mobility_core/data/`` and refuses to guess what it does not
contain.

Extra ``geo``: ``zone_resolver`` and ``residence_zone.CommunalZones`` require geopandas,
shapely and pyproj.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
