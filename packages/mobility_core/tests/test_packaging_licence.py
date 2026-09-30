"""What the package is allowed to ship.

`mobility_core` is installable, hence redistributable, and it embeds resources
derived from the EMC² Toulouse 2023 survey whose microdata are restricted-access
(ProGEDO / ADISP agreement lil-1750). Agreements of this kind allow distributing
**results**, not **data**: a resource that reproduces the survey's sampling plan,
or that publishes cell counts that are too small, is data.

This test judges nothing: it checks that the `package-data` list of `pyproject.toml` says
explicitly what it ships, and that no resource classed as restricted appears in it.

Two regressions make it fail, and they are the two that have already happened:

- **the return of a glob** (`data/*.json`). A glob picks up any file present on the
  disk at build time, including those that `.gitignore` excludes: that is how
  `zf_housing_type.json`, ignored by git, ended up in `build/lib/` then in the
  wheel. `.gitignore` does not protect the package;
- **the addition of a restricted resource** to the list, for deployment convenience.

Two levels: the declaration tests read `pyproject.toml` and `MANIFEST.in` (fast,
offline); the artefact test actually builds the sdist and the wheel and looks at what
they contain. The second exists because the first is not enough — the explicit list
of `package-data` was correct while BOTH artefacts still shipped the
restricted resources, through `include-package-data` (True by default, it merges with the list)
for the wheel and through the absence of `MANIFEST.in` for the sdist. What is authoritative is
the archive, not the declaration.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Resources derived from the survey microdata or zoning: never in the package.
# `zf_couronne` reproduces the sampling plan (785 fine zones → sector → commune);
# `zf_housing_type` publishes the count per fine zone (median 12 households, 195 zones under 5);
# `zf_zones` is the survey's GIS layer itself.
RESSOURCES_RESTREINTES = (
    "zf_couronne.json",
    "zf_housing_type.json",
    "zf_zones.gpkg",
    "zf_zones.meta.json",
)


@pytest.fixture(scope="module")
def package_data() -> list[str]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    entries = config["tool"]["setuptools"]["package-data"]["mobility_core"]
    assert entries, "package-data empty: the package would ship no resource"
    return list(entries)


def test_aucun_glob_dans_package_data(package_data: list[str]) -> None:
    """A glob would make the wheel content depend on whatever lies around on the disk."""
    globs = [entry for entry in package_data if any(c in entry for c in "*?[")]
    assert not globs, (
        "package-data must list the resources one by one (ticket 038); "
        f"patterns found: {globs}. A glob ships the restricted-access resources "
        "present on the build disk, which `.gitignore` does not keep out of the package."
    )


@pytest.mark.parametrize("ressource", RESSOURCES_RESTREINTES)
def test_ressource_restreinte_absente_du_paquet(package_data: list[str], ressource: str) -> None:
    """Adding one of these resources to the package means redistributing the survey."""
    porteurs = [entry for entry in package_data if entry.endswith(ressource)]
    assert not porteurs, (
        f"`{ressource}` derives from the ProGEDO lil-1750 microdata (restricted access) and "
        f"cannot be redistributed with the package; offending entry: {porteurs}. "
        "It is produced locally (`make communes-couronnes`, `make housing-type`, "
        "`make zones`) and read outside the package."
    )


def test_ressources_declarees_existent_sur_le_disque(package_data: list[str]) -> None:
    """An entry that matches no file is a resource silently lost."""
    racine = PYPROJECT.parent / "src" / "mobility_core"
    manquantes = [entry for entry in package_data if not (racine / entry).exists()]
    assert not manquantes, (
        f"declared in package-data but missing from {racine}: {manquantes}"
    )


# ---------------------------------------------------------------------------
# What is authoritative: the built artefact.
# ---------------------------------------------------------------------------

def _noms_data(chemin: Path) -> set[str]:
    """File names under `data/` in an sdist (.tar.gz) or a wheel (.whl)."""
    if chemin.suffix == ".whl":
        import zipfile
        membres = zipfile.ZipFile(chemin).namelist()
    else:
        import tarfile
        membres = tarfile.open(chemin).getnames()
    return {m.split("/data/")[-1] for m in membres if "/data/" in m}


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Builds sdist and wheel in a throwaway directory. Skips if `build` is missing."""
    pytest.importorskip("build", reason="`pip install build` to check the artefacts")
    import subprocess
    import sys

    sortie = tmp_path_factory.mktemp("dist")
    projet = PYPROJECT.parent
    # `--outdir` outside the project: a leftover `build/` of the repository must not take part.
    process = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(sortie), str(projet)],
        capture_output=True, text=True,
    )
    assert process.returncode == 0, (
        f"package build failed (code {process.returncode})\n{process.stderr[-2000:]}"
    )
    construits = sorted(sortie.glob("*.tar.gz")) + sorted(sortie.glob("*.whl"))
    assert len(construits) == 2, f"expected an sdist and a wheel, got: {construits}"
    return construits


@pytest.mark.slow
def test_artefacts_sans_ressource_restreinte(artefacts: list[Path]) -> None:
    """The sdist AND the wheel: neither archive redistributes the survey."""
    fautes = {
        artefact.name: sorted(_noms_data(artefact) & set(RESSOURCES_RESTREINTES))
        for artefact in artefacts
    }
    coupables = {nom: ressources for nom, ressources in fautes.items() if ressources}
    assert not coupables, (
        f"restricted-access resources present in the artefacts: {coupables}. "
        "The wheel is set by `package-data` + `include-package-data = false`, the sdist "
        "by the `MANIFEST.in` whitelist — both, not one of the two."
    )
