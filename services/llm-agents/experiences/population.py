"""Loading a population for the platform — name, fingerprint, seal (spec 01 J1, spec 06 E2).

Any population is admissible (EF-02): a sealed folder (`MANIFEST.yaml` + `population.json`)
as well as a bare JSON file. The result says which of the two situations applied. Persons
are built by the **same** loader as the simulation (`EqasimJSONPopulationLoader`), hence
with the same scheduled times: this is what makes the number of expected trips derivable
identically on both sides (J2, E22).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from experiences import froid
from models import Person

# The cold-archive guard lives in `experiences.froid` since ticket 074: populations,
# sets and prompts now share the same rule, and a rule written twice ends up
# diverging. The names here remain exported — code and tests import them from this
# module since ticket 045.
SEGMENT_ARCHIVE = froid.SEGMENT_ARCHIVE
COHORTE_DE_REFERENCE = "data/population/population_1000_PANEL_v6"

# `PopulationArchivee` remains a subclass of ValueError, as before: callers that
# caught it keep working.
PopulationArchivee = froid.ContenuArchive


def _sous_archive(chemin: Path) -> bool:
    """Does the path go through an `archive` folder?"""
    return froid.sous_archive(chemin)


def _verifier_non_archivee(chemin: Path, archivee_confirmee: str | None) -> None:
    """Refuses an archived cohort, unless an explicit reason is given — and logs the exemption.

    Why a refusal in the code and not a sentence in a README: the platform's 36 runs
    all read cohort v1 while the article's reference was v5,
    and nothing stood in the way. A safeguard that exists only in prose never
    fires.

    Lifting it requires a REASON, not a boolean: `archivee_confirmee=True` gets ticked without
    thinking, `archivee_confirmee="ticket 045 control"` gets written and read back. It is set in
    the experiment definition: `population.archivee_confirmee: <motif>`.

    The body of the rule lives in `experiences.froid` since ticket 074 — populations, sets
    and prompts share it.
    """
    froid.verifier(
        chemin,
        archivee_confirmee,
        quoi="a population cohort",
        comment_lever=(
            "the experiment definition must carry `population.archivee_confirmee: <reason>`"
        ),
        repli=COHORTE_DE_REFERENCE,
    )


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


@dataclass
class InfoPopulation:
    nom: str
    chemin: str  # population.json file actually read
    sha256: str  # IDENTITY fingerprint: MANIFEST.yaml if sealed, otherwise the file
    fichier_sha256: str  # fingerprint of population.json in both cases
    scellee: bool
    n: int
    manifest: dict | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "nom": self.nom,
            "chemin": self.chemin,
            "sha256": self.sha256,
            "fichier_sha256": self.fichier_sha256,
            "scellee": self.scellee,
            "n": self.n,
        }


def resoudre_population(
    chemin: str | Path, *, archivee_confirmee: str | None = None
) -> tuple[Path, Path | None]:
    """(population.json, MANIFEST.yaml or None) from a sealed folder or a file.

    Refuses a cohort stored under `archive/` (R15): this is the single mandatory gateway
    of any population read, hence the right place for the guard. `archivee_confirmee`
    carries the REASON for the exemption, which is logged.
    """
    p = Path(chemin)
    _verifier_non_archivee(p, archivee_confirmee)
    if p.is_dir():
        manifest = p / "MANIFEST.yaml"
        fichier = p / "population.json"
        if not fichier.exists():
            raise FileNotFoundError(f"population not found: {fichier}")
        return fichier, (manifest if manifest.exists() else None)
    if not p.exists():
        raise FileNotFoundError(f"population not found: {p}")
    manifest = p.parent / "MANIFEST.yaml"
    return p, (manifest if manifest.exists() else None)


def info_population(
    chemin: str | Path, *, archivee_confirmee: str | None = None
) -> InfoPopulation:
    fichier, manifest_path = resoudre_population(
        chemin, archivee_confirmee=archivee_confirmee
    )
    fichier_sha = sha256_fichier(fichier)
    manifest = None
    scellee = False
    if manifest_path is not None:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        attendu = (manifest.get("population") or {}).get("sha256")
        scellee = bool(attendu)
        if attendu and attendu != fichier_sha:
            raise ValueError(
                f"sealed population altered: MANIFEST states {attendu[:12]}…, the file is {fichier_sha[:12]}… ({fichier})"
            )
    nom = (manifest or {}).get("nom") or (
        fichier.parent.name if fichier.name == "population.json" else fichier.stem
    )
    with open(fichier, encoding="utf-8") as f:
        n = len(json.load(f))
    return InfoPopulation(
        nom=str(nom),
        chemin=str(fichier),
        sha256=sha256_fichier(manifest_path) if scellee else fichier_sha,
        fichier_sha256=fichier_sha,
        scellee=scellee,
        n=n,
        manifest=manifest,
    )


def charger_population(
    chemin: str | Path, *, archivee_confirmee: str | None = None
) -> tuple[list[Person], InfoPopulation]:
    """Persons + info. Same loader as the simulation, with no scope or size filter.

    `archivee_confirmee` carries the reason for an exemption from the archive guard (R15); without
    it, a cohort stored under `archive/` is refused.
    """
    from inputs.population.eqasim_loader import EqasimJSONPopulationLoader

    info = info_population(chemin, archivee_confirmee=archivee_confirmee)
    with open(info.chemin, encoding="utf-8") as f:
        brut = json.load(f)
    if (
        brut
        and "name" in (brut[0].get("identity") or {})
        and "traits_json" in brut[0]["identity"]
        and brut[0]["identity"].get("activities")
        and "id" in brut[0]["identity"]["activities"][0]
        and "state" in brut[0]
        and brut[0]["identity"]["activities"][0].get("start_time") is not None
    ):
        # eqasim format (that of sealed populations) — the loader sets scheduled_start_time.
        pass
    loader = EqasimJSONPopulationLoader()
    personnes = loader.load_population_from_data(brut, max_size=len(brut), bbox=None)
    return personnes, info
