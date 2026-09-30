"""resources — where the files this package reads are.

Three families:

* the **package resources** (``data/``), shipped with it: :func:`data_path`;
* the **restricted-access resources**, which live in the same place but are **never**
  redistributed with the package because they reproduce the EMC² Toulouse 2023 microdata
  or their zoning (ProGEDO / ADISP agreement lil-1750): :func:`restricted_data_path`.
  A ``pip install mobility-core`` does not have them; the repository and containers do;
* the **repository reference files** (``scripts/data/population/…``,
  ``scripts/progedo_logit/feature_spec.json``), which are not copied into the package
  so as not to create a second source of truth: :func:`find_repo_file`.

A repository file is looked up, in order: the dedicated environment variable, the
root designated by ``MOBILITY_CORE_REPO_ROOT``, a walk up from the current directory,
then ``/app`` (the ``controller`` container mounts ``scripts/`` under ``/app/scripts``).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT_ENV = "MOBILITY_CORE_REPO_ROOT"
EMC2_DATA_DIR_ENV = "MOBILITY_CORE_EMC2_DATA_DIR"

# Location of the resources in a repository tree, when the package itself does not
# carry them (the `pip install` case).
_REPO_DATA_RELATIVE = "mobility_core/src/mobility_core/data"

# Restricted-access resources → command that (re)produces them. This table is the source of
# truth of the distribution regime: it must stay aligned with the whitelist in
# `MANIFEST.in` and with `package-data`, which `tests/test_packaging_licence.py` checks.
#
# Why these and not the others (ticket 038): the lil-1750 agreement allows distributing
# RESULTS, not DATA. `zf_couronne.json` reproduces the survey's sampling plan
# (785 fine zones → sampling sector → commune); `zf_housing_type.json`
# publishes the count per fine zone (median 12 households, 195 zones under 5); `zf_zones.gpkg`
# is the survey's GIS layer. The other resources are model coefficients or
# ring-level aggregates.
RESTRICTED_RESOURCES: dict[str, str] = {
    "zf_couronne.json": "make communes-couronnes",
    "zf_housing_type.json": "make housing-type",
    "zf_zones.gpkg": "make zones",
    "zf_zones.meta.json": "make zones",
}

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CONTAINER_ROOT = Path("/app")
_MAX_UPWARD_STEPS = 8


def data_dir() -> Path:
    """Directory of the resources shipped with the package."""
    return _DATA_DIR


def data_path(name: str) -> Path:
    """Path of a package resource (``bike_ownership.json``, ``zf_zones.gpkg``…).

    The file may not exist (``zf_zones.gpkg`` is produced by ``make zones``):
    the caller decides whether to make it an error.
    """
    return _DATA_DIR / name


def repo_root_candidates() -> list[Path]:
    """Plausible repository roots, in order of preference."""
    roots: list[Path] = []
    env_root = os.getenv(REPO_ROOT_ENV)
    if env_root:
        roots.append(Path(env_root))
    here = Path.cwd().resolve()
    roots.append(here)
    roots.extend(here.parents[:_MAX_UPWARD_STEPS])
    roots.append(_CONTAINER_ROOT)
    return roots


def find_repo_file(relative: str, env_var: str | None = None) -> Path | None:
    """First repository file found at ``relative`` from a plausible root.

    ``env_var`` names an environment variable that, when set, takes precedence over
    any search (full path of the file). Returns ``None`` if nothing exists.
    """
    if env_var:
        override = os.getenv(env_var)
        if override:
            candidate = Path(override)
            return candidate.resolve() if candidate.exists() else None
    for root in repo_root_candidates():
        candidate = root / relative
        if candidate.exists():
            return candidate.resolve()
    return None


def repo_file_candidates(relative: str) -> list[Path]:
    """The paths tried by :func:`find_repo_file`, for a useful error message."""
    return [root / relative for root in repo_root_candidates()]


# ---------------------------------------------------------------------------
# Restricted-access resources — present in the repository, never in the package.
# ---------------------------------------------------------------------------

def _check_restricted(name: str) -> None:
    """Rejects a name not declared restricted: it is a caller error."""
    if name not in RESTRICTED_RESOURCES:
        raise ValueError(
            f"'{name}' is not a restricted-access resource. The resources shipped "
            f"with the package are read through data_path(). Declared restricted: "
            f"{sorted(RESTRICTED_RESOURCES)}.")


def restricted_data_candidates(name: str) -> list[Path]:
    """Locations tried for a restricted resource, in order of preference.

    1. ``$MOBILITY_CORE_EMC2_DATA_DIR`` — the directory where the agreement holder
       stores the resources they produced. It takes precedence over everything else;
    2. the package's ``data/`` — the case of the repository, an editable install and the
       containers, which mount ``mobility_core/src/mobility_core`` as a volume;
    3. ``mobility_core/src/mobility_core/data/`` under a plausible repository root — the
       case of a package installed next to a copy of the repository.
    """
    _check_restricted(name)
    candidates: list[Path] = []
    env_dir = os.getenv(EMC2_DATA_DIR_ENV)
    if env_dir:
        candidates.append(Path(env_dir) / name)
    candidates.append(_DATA_DIR / name)
    candidates.extend(root / _REPO_DATA_RELATIVE / name for root in repo_root_candidates())
    # Walking up directories produces duplicates (the package IS in the repository, as an
    # editable install): keeping them would make the error message unreadable where it matters.
    uniques: list[Path] = []
    vus: set[str] = set()
    for candidate in candidates:
        cle = str(candidate)
        if cle not in vus:
            vus.add(cle)
            uniques.append(candidate)
    return uniques


def restricted_data_path(name: str) -> Path:
    """Path of a restricted-access resource: the first location that exists.

    When none exists, returns the **first candidate** rather than ``None``: the caller
    then raises its domain error with a path to show, and :func:`restricted_resource_hint`
    says what to do. No silent fallback — a ring or a housing type that were
    guessed would later be read as a modal share, not as a bug.
    """
    candidates = restricted_data_candidates(name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def restricted_resource_hint(name: str) -> str:
    """Actionable error message for a restricted resource that cannot be found.

    Names the command that produces it, the environment variable that designates it and the
    locations tried — a bare ``FileNotFoundError`` would force a reread of the code.
    """
    _check_restricted(name)
    essayes = "\n".join(f"    - {c}" for c in restricted_data_candidates(name))
    return (
        f"'{name}' derives from the EMC² Toulouse 2023 microdata (ProGEDO / ADISP lil-1750, "
        f"restricted access): it is NOT shipped with the package (ticket 038).\n"
        f"  Produce it: {RESTRICTED_RESOURCES[name]} — requires the data under "
        f"'data/PROGEDO 2023/'.\n"
        f"  Or point to the directory that contains it: {EMC2_DATA_DIR_ENV}=/path/to/data\n"
        f"  Locations tried:\n{essayes}"
    )
