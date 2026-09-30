"""An archived population is never read by accident.

R15: `resoudre_population` refuses a path under `data/population/archive/`, unless explicitly
marked `archivee_confirmee`, and the refusal names the reference cohort.

Why a refusal in the code, and not just a sentence in a README: the 36 runs of the
platform all read cohort v1 while the article's reference is v5, and
nothing stood in the way. A safeguard that only exists in prose never fires. The
refusal must be loud, and lifting it must leave a trace — a reason written in the definition,
not a boolean flag ticked without thinking.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from experiences.population import (
    PopulationArchivee,
    charger_population,
    info_population,
    resoudre_population,
)

HOME = {"lon": 1.4400, "lat": 43.6000, "public_transport": True, "zone": "centre"}
WORK = {"lon": 1.4500, "lat": 43.6100, "public_transport": True, "zone": "nord"}


def _population_minimale() -> list[dict]:
    traits = {
        "age": 35,
        "gender": "Female",
        "main_occupation": "actif",
        "household_size": 1,
        "number_of_cars": 1,
        "has_driving_license": True,
        "personal_bike": "vélo normal",
        "residence_zone": "Toulouse",
    }
    acts = [
        {
            "id": "a0",
            "scheduled_start_time": 20 * 3600.0,
            "start_time": 0.0,
            "end_time": 8 * 3600.0,
            "purpose": "home",
            "location": HOME,
        },
        {
            "id": "a1",
            "scheduled_start_time": 8 * 3600.0,
            "start_time": 9 * 3600.0,
            "end_time": 20 * 3600.0,
            "purpose": "work",
            "location": WORK,
        },
    ]
    return [
        {
            "person_id": "p1",
            "identity": {"traits_json": traits, "home": HOME, "activities": acts},
            "state": {"last_activity_index": 0},
            "is_llm_based": True,
        }
    ]


def _cohorte(racine, nom: str):
    """Writes a sealed cohort under `racine/nom` and returns its directory."""
    d = racine / nom
    d.mkdir(parents=True)
    fichier = d / "population.json"
    fichier.write_text(json.dumps(_population_minimale()), encoding="utf-8")
    sha = hashlib.sha256(fichier.read_bytes()).hexdigest()
    (d / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {"nom": nom, "population": {"fichier": "population.json", "sha256": sha}}
        ),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def arborescence(tmp_path):
    """`data/population/` with a live cohort and an archived cohort."""
    pop = tmp_path / "data" / "population"
    vivante = _cohorte(pop, "population_1000_PANEL_v5")
    archivee = _cohorte(pop / "archive", "population_1000_PANEL")
    return {"racine": pop, "vivante": vivante, "archivee": archivee}


# ── The refusal ─────────────────────────────────────────────────────────────


def test_r15_une_cohorte_archivee_est_refusee(arborescence):
    with pytest.raises(PopulationArchivee):
        resoudre_population(arborescence["archivee"])


def test_r15_le_refus_nomme_le_chemin_et_la_marche_a_suivre(arborescence):
    """A refusal that does not say what to do gets bypassed by guesswork, or endured."""
    with pytest.raises(PopulationArchivee) as e:
        resoudre_population(arborescence["archivee"])
    message = str(e.value)
    assert "population_1000_PANEL" in message
    assert "archive" in message
    assert "archivee_confirmee" in message, "the refusal must say how to lift it"


def test_r15_le_fichier_nu_sous_archive_est_refuse_aussi(arborescence):
    """The guard is about location, not form: a bare JSON under `archive/` too."""
    nu = arborescence["racine"] / "archive" / "vieille_population.json"
    nu.write_text(json.dumps(_population_minimale()), encoding="utf-8")
    with pytest.raises(PopulationArchivee):
        resoudre_population(nu)


def test_r15_un_sous_dossier_profond_de_archive_est_refuse(arborescence):
    """`archive/2026/pop/` is under archive: depth does not bypass the guard."""
    profonde = _cohorte(arborescence["racine"] / "archive" / "2026", "vieille")
    with pytest.raises(PopulationArchivee):
        resoudre_population(profonde)


# ── The explicit lifting ────────────────────────────────────────────────────


def test_r15_un_motif_explicite_leve_le_refus(arborescence):
    fichier, manifest = resoudre_population(
        arborescence["archivee"], archivee_confirmee="témoin du ticket 045"
    )
    assert fichier.is_file()
    assert manifest is not None


def test_r15_la_levee_est_journalisee_avec_son_motif(arborescence, caplog):
    """The reason must appear in the log: a silent exemption is not one."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        resoudre_population(
            arborescence["archivee"], archivee_confirmee="témoin du ticket 045"
        )
    finally:
        logger.remove(sink)
    assert any("témoin du ticket 045" in m for m in messages), messages


def test_r15_un_motif_vide_ne_leve_rien(arborescence):
    """`archivee_confirmee: ""` or `True` is not a reason: one must say WHY."""
    for faux_motif in ("", "   ", None):
        with pytest.raises(PopulationArchivee):
            resoudre_population(arborescence["archivee"], archivee_confirmee=faux_motif)


# ── What the guard must not break ───────────────────────────────────────────


def test_r15_une_cohorte_vivante_se_charge_sans_rien_demander(arborescence):
    fichier, manifest = resoudre_population(arborescence["vivante"])
    assert fichier.is_file()
    assert manifest is not None
    info = info_population(arborescence["vivante"])
    assert info.scellee and info.n == 1


def test_r15_un_dossier_nomme_archives_ailleurs_nest_pas_vise(tmp_path):
    """The guard targets the `archive` segment of a population path, not the word anywhere.

    A cohort whose name contains "archive" stays readable: the location
    decides, never a text match.
    """
    pop = tmp_path / "data" / "population"
    d = _cohorte(pop, "population_archivistes_2026")
    fichier, _ = resoudre_population(d)
    assert fichier.is_file()


def test_r15_charger_population_propage_le_garde(arborescence):
    with pytest.raises(PopulationArchivee):
        charger_population(arborescence["archivee"])
    personnes, info = charger_population(
        arborescence["archivee"], archivee_confirmee="témoin du ticket 045"
    )
    assert len(personnes) == 1 and info.n == 1
