"""The cold archive is unreachable at run time, and the code enforces it.

    Cold = restorable and auditable, never used nor referenced.

The acceptance criterion is explicit: "a test proves that no loader in the code
resolves a name from `archive/`". Prove, not promise — a guarantee that only exists
in a README never fires (lesson from a past incident: 36 runs read the wrong
cohort without anything objecting).

Three loaders, three mandatory checkpoints, three refusals:

1. `experiences.population.resoudre_population` — every cohort read goes through it;
2. `experiences.jeu.Jeu.charger` — every sealed-set read goes through it;
3. `llm_gateway.prompts.engine.PromptManager` — every system-prompt serving goes through it.

And two anchoring properties, which are the reason why the registry and the queue never see
the archive: `dossier_experiences()` and `dossier_jeux()` are anchored on `data/`, and
the archive lives elsewhere.

The exemption remains possible — the D-7 comparability guard must be able to re-read the frozen v5 —
but it requires a written REASON, never a boolean.

Running:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_cold_archive.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))
sys.path.insert(0, str(RACINE / "packages" / "llm_gateway" / "src"))

from experiences import froid  # noqa: E402
from experiences.experience import dossier_experiences, dossier_jeux  # noqa: E402
from experiences.population import PopulationArchivee, resoudre_population  # noqa: E402
from llm_gateway.prompts.engine import PromptManager, PromptsArchives  # noqa: E402

MOTIF = "garde de comparabilité D-7, ticket 074"


# ── The predicate itself ──────────────────────────────────────────────────────────────────


def test_le_garde_porte_sur_l_emplacement_pas_sur_le_nom(tmp_path: Path) -> None:
    """An `archive` segment in the path is enough; a NAME that contains "archive" is not.

    The distinction is not a subtlety: a cohort that would be called
    `population_archivistes` must remain perfectly readable.
    """
    assert froid.sous_archive(tmp_path / "archive" / "2026-09-14_avant_bascule_anglaise" / "x")
    assert froid.sous_archive(tmp_path / "data" / "archive" / "jeux")
    assert not froid.sous_archive(tmp_path / "data" / "population_archivistes")
    assert not froid.sous_archive(tmp_path / "data" / "jeux" / "archivage.yaml")


def test_la_derogation_exige_un_motif_ecrit(tmp_path: Path) -> None:
    chemin = tmp_path / "archive" / "gel" / "population.json"
    garde = dict(quoi="une cohorte de population", comment_lever="écrire `archivee_confirmee`")
    with pytest.raises(froid.ContenuArchive):
        froid.verifier(chemin, None, **garde)
    with pytest.raises(froid.ContenuArchive):
        froid.verifier(chemin, "   ", **garde)   # an empty string is not a reason
    with pytest.raises(froid.ContenuArchive):
        froid.verifier(chemin, True, **garde)    # type: ignore[arg-type]
    froid.verifier(chemin, MOTIF, **garde)       # written reason → passes


def test_le_refus_dit_toujours_comment_le_lever(tmp_path: Path) -> None:
    """A refusal that does not say what to do gets worked around by guesswork, or endured.

    The three loaders must name the concrete field, not "provide a reason".
    """
    from experiences.jeu import Jeu

    with pytest.raises(PopulationArchivee) as pop:
        resoudre_population(_cohorte(tmp_path / "archive" / "gel" / "v5"))
    assert "population.archivee_confirmee" in str(pop.value)

    with pytest.raises(froid.ContenuArchive) as jeu:
        Jeu.charger(_jeu(tmp_path / "archive" / "gel" / "j5"))
    assert "archive_confirmee" in str(jeu.value)

    with pytest.raises(PromptsArchives) as pm:
        PromptManager(
            templates_dir=tmp_path,
            prompts_file=_store(tmp_path / "archive" / "gel" / "prompts.yaml"),
            exiger_avis_neutralite=False,
        )
    # The prompt store, for its part, has NO exemption: an archived prompt is never served.
    # So the message says where the live store is, instead.
    assert "mobility_llm/prompts/prompts.yaml" in str(pm.value)


# ── 1. Populations ────────────────────────────────────────────────────────────────────────


def _cohorte(dossier: Path) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "population.json").write_text("[]", encoding="utf-8")
    return dossier


def test_aucune_population_ne_se_resout_depuis_l_archive(tmp_path: Path) -> None:
    gelee = _cohorte(tmp_path / "archive" / "2026-09-14_avant_bascule_anglaise" / "population" / "v5")
    with pytest.raises(PopulationArchivee) as e:
        resoudre_population(gelee)
    assert "cold archive" in str(e.value)

    vivante = _cohorte(tmp_path / "data" / "population" / "population_1000_PANEL_v6")
    fichier, _ = resoudre_population(vivante)
    assert fichier == vivante / "population.json"


def test_une_population_archivee_se_relit_avec_un_motif(tmp_path: Path) -> None:
    gelee = _cohorte(tmp_path / "archive" / "gel" / "v5")
    fichier, _ = resoudre_population(gelee, archivee_confirmee=MOTIF)
    assert fichier == gelee / "population.json"


# ── 2. Sealed sets ────────────────────────────────────────────────────────────────────────


def _jeu(dossier: Path) -> Path:
    """A minimal but VALID set: the refusal must come from the guard, not a shaky manifest."""
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "jeu1",
                "nom": dossier.name,
                "clos": False,
                "population": {"sha256": "0" * 64, "nom": "x", "n": 0},
                "dependances": {"commit": None},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (dossier / "propositions.jsonl").write_text("", encoding="utf-8")
    return dossier


def test_aucun_jeu_ne_se_charge_depuis_l_archive(tmp_path: Path) -> None:
    from experiences.jeu import Jeu

    gele = _jeu(tmp_path / "archive" / "gel" / "plateforme" / "jeux" / "v5_20260316")
    with pytest.raises(froid.ContenuArchive) as e:
        Jeu.charger(gele)
    assert "a sealed set in cold archive" in str(e.value)

    # The same folder, elsewhere, loads: it is indeed the location that refuses.
    vivant = _jeu(tmp_path / "data" / "jeux" / "v6_20260316")
    assert Jeu.charger(vivant).nom == "v6_20260316"


def test_un_jeu_archive_se_relit_avec_un_motif(tmp_path: Path) -> None:
    from experiences.jeu import Jeu

    gele = _jeu(tmp_path / "archive" / "gel" / "v5_20260316")
    assert Jeu.charger(gele, archive_confirmee=MOTIF).nom == "v5_20260316"


# ── 3. System prompts ─────────────────────────────────────────────────────────────────────


def _store(chemin: Path) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        yaml.safe_dump(
            {
                "active": {"itinary_multi_agent": "v"},
                "prompts": {"v": {"content": "Pick a mode.\n"}},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return chemin


def test_le_prompt_manager_refuse_un_store_archive(tmp_path: Path) -> None:
    gele = _store(tmp_path / "archive" / "gel" / "prompts" / "prompts.yaml")
    with pytest.raises(PromptsArchives) as e:
        PromptManager(templates_dir=tmp_path, prompts_file=gele, exiger_avis_neutralite=False)
    assert "cold archive" in str(e.value)


def test_le_prompt_manager_sert_le_store_vivant(tmp_path: Path) -> None:
    vivant = _store(tmp_path / "packages" / "prompts" / "prompts.yaml")
    pm = PromptManager(templates_dir=tmp_path, prompts_file=vivant, exiger_avis_neutralite=False)
    assert pm.get_system_prompt("itinary_multi_agent") == "Pick a mode."


# ── 4. Anchoring of the registry and the queue ────────────────────────────────────────────


def test_le_registre_et_la_file_s_ancrent_hors_archive() -> None:
    """`registre` and the FIFO queue read `dossier_experiences()`, `dossier_jeux()` — not the archive.

    This is not a filter added somewhere, it is an anchoring property: these two
    functions point under `data/`, and the cold archive lives under `archive/`, next to it. The test
    locks the property so that a future move of either one shows.
    """
    for dossier in (dossier_experiences(), dossier_jeux()):
        assert not froid.sous_archive(dossier), (
            f"{dossier} crosses an `archive` segment: the registry would list frozen content"
        )


def test_l_archive_du_ticket_074_n_est_pas_sous_data() -> None:
    """The archive is a sibling of `data/`, not a child — otherwise the anchors above would see it."""
    archive = RACINE / "archive" / "2026-09-14_avant_bascule_anglaise"
    assert froid.sous_archive(archive)
    assert (RACINE / "data") not in archive.parents
