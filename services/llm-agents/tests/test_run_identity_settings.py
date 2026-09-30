"""An experiment setting belongs to the run, and the run says so.

On 18 September, four experiment settings were put in `config/config.yaml`. They
became the repository default — thirty red tests since — and they appeared nowhere in
`identite_run.json`: three arms with opposite settings carried the same identity.
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urban_mobility_agents.utils import identite_run as I

RACINE = Path(__file__).resolve().parents[1]
CONFIG = RACINE / "config" / "config.yaml"
COHORTE = RACINE.parents[1] / "scripts" / "experiment" / "run_sequential_cohort.py"

# What an experiment arm varies. Removed from `config.yaml` (K1), set by
# the environment (K7), recorded in the run identity (K2).
REGLAGES_D_EXPERIENCE = (
    "vehicle_chain_enabled",
    "vehicle_return_home_lock",
    "mode_choice_truncation_threshold",
    "stm_reflection_min_entries",
)


def _reglages(**s):
    base = {
        "chaine": True,
        "verrou": True,
        "troncature": 0.0,
        "seuil_choc": 0.7,
        "fenetre": 14,
        "changements_max": 3,
        "reflexion_min": 10,
        "meteo_par_agent": True,
    }
    base.update(s)
    return SimpleNamespace(
        llm=SimpleNamespace(instances_admises=[], providers={}),
        agent=SimpleNamespace(
            llm_params={"prompt_variant": "prompt_expert_05"},
            long_term_memory_enabled=True,
            long_term_self_reflect_enabled=True,
            mode_draw_seed=42,
            option_order_seed=42,
            weather_draw_seed=42,
            vehicle_chain_enabled=base["chaine"],
            vehicle_return_home_lock=base["verrou"],
            mode_choice_truncation_threshold=base["troncature"],
            memoire__importance_choc=base["seuil_choc"],
            memoire__fenetre_changements_jours=base["fenetre"],
            memoire__changements_max=base["changements_max"],
            stm_reflection_min_entries=base["reflexion_min"],
            weather_per_agent_dates=base["meteo_par_agent"],
        ),
        data=SimpleNamespace(population_file="/data/p.json"),
        cache=SimpleNamespace(enabled=False),
    )


# ══════════════════════ K1 — outside the repository configuration ══════════════════


def test_K1_config_yaml_ne_declare_aucun_reglage_d_experience():
    """K1 — set there, they become the repository norm without anyone having decided it."""
    agent = (yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("agent") or {}
    fautifs = sorted(set(agent) & set(REGLAGES_D_EXPERIENCE))
    assert not fautifs, (
        "config.yaml déclare des réglages d'expérience : "
        + ", ".join(fautifs)
        + " — à passer par l'environnement, où le run les enregistre dans son identité."
    )


# ══════════════════════ K2 to K6 — the identity carries them ══════════════════════════


def test_K2_l_identite_porte_les_reglages_d_experience():
    """K2 — without them, three experiment arms carry the same identity."""
    ident = I.composer(_reglages())
    for champ in (
        "chaine_vehicules",
        "verrou_retour_domicile",
        "seuil_troncature",
        "seuil_choc",
        "fenetre_changements_jours",
        "changements_max",
        "reflexion_stm_min_entrees",
        "meteo_par_agent",
    ):
        assert champ in ident, f"l'identité du run ne porte pas {champ}"


def test_K3_le_seuil_de_troncature_separe_deux_runs():
    """K3 — it changes the draw law: two runs that differ on it are not comparable."""
    ecarts = I.differences(I.composer(_reglages()), I.composer(_reglages(troncature=0.15)))
    assert len(ecarts) == 1
    assert "troncature" in ecarts[0]


def test_K4_la_fenetre_de_changements_separe_deux_runs():
    """K4 — it is the ablation arm parameter: it must not be able to go unnoticed."""
    ecarts = I.differences(I.composer(_reglages()), I.composer(_reglages(fenetre=7)))
    assert len(ecarts) == 1
    assert "fenêtre" in ecarts[0].lower()


def test_K5_un_champ_absent_est_dit_absent(tmp_path):
    """K5 — a run from before 2026-09-19 cannot be resumed, and the reason is its AGE."""
    ancienne = {k: v for k, v in I.composer(_reglages()).items() if k != "seuil_troncature"}
    I.ecrire(tmp_path, ancienne)
    with pytest.raises(I.IdentiteIncompatible) as exc:
        I.verifier(tmp_path, I.composer(_reglages()))
    assert "absent du run repris" in str(exc.value)


def test_K6_les_defauts_de_l_identite_sont_ceux_du_depot():
    """K6 — a nominal run must never believe itself different from itself."""
    from settings import settings

    ident = I.composer(settings)
    assert ident["chaine_vehicules"] == bool(settings.agent.vehicle_chain_enabled)
    assert ident["seuil_choc"] == pytest.approx(float(settings.agent.memoire__importance_choc))
    assert ident["fenetre_changements_jours"] == int(
        settings.agent.memoire__fenetre_changements_jours
    )


# ══════════════════════ K7 — the campaign sets them ═══════════════════════════════


def test_K7_la_cohorte_pose_tous_les_reglages_par_l_environnement():
    """K7 — otherwise the campaign runs on repository defaults without anything saying so."""
    if not COHORTE.is_file():
        pytest.skip("scripts/experiment/run_sequential_cohort.py absent")
    source = COHORTE.read_text(encoding="utf-8")
    for reglage in REGLAGES_D_EXPERIENCE:
        # ⚠ BARE NAME. This rule was first written with an `AGENT__` prefix, by analogy
        # with other repositories — and it passed, because the cohort set the same wrong
        # form. The `settings.py` sub-configurations are `BaseSettings` WITHOUT a
        # prefix: only `VEHICLE_CHAIN_ENABLED` is read. Checked at run time on 2026-09-19,
        # after a whole arm had run on default values.
        attendu = reglage.upper()
        assert re.search(rf'"{attendu}"', source) or f'"{attendu}"' in source, (
            f"{attendu} is not set by the cohort: removed from config.yaml and not replaced, "
            f"this setting would fall back to its default value on the next run."
        )
        actives = [
            l for l in source.splitlines()
            if f"AGENT__{attendu}" in l and not l.strip().startswith("#")
        ]
        assert not actives, (
            f"AGENT__{attendu} encore posé : ce préfixe n'est lu par personne dans ce dépôt.\n"
            + "\n".join(actives)
        )


# ══════════════════════ K8 to K10 — the campaign really separates its arms ═════════


def _source_cohorte() -> str:
    if not COHORTE.is_file():
        pytest.skip("scripts/experiment/run_sequential_cohort.py absent")
    return COHORTE.read_text(encoding="utf-8")


def test_K8_la_campagne_declare_un_choc_par_branche():
    """K8 — without `CHOC=`, `make run` keeps the config.yaml declaration and the control is treated."""
    source = _source_cohorte()
    assert 'f"CHOC={choc}"' in source or "CHOC=" in source, (
        "the campaign command carries no CHOC=: the control branch would run with "
        "the shock declared in config.yaml"
    )
    assert '"treated"' in source and 'else "0"' in source, (
        "the two branches do not explicitly split on the shock"
    )


def test_K9_le_texte_du_choc_n_est_pas_recopie_dans_le_lanceur():
    """K9 — a copy drifts from the tested file; it happened, and the shock cadence never reached the campaign."""
    source = _source_cohorte()
    assert "CHOC_TEMPLATE" not in source, "the launcher still carries its own copy of the shock"
    assert "CHOC_REFERENCE" in source, "the launcher does not derive the shock from the reference file"
    for mot in ("grinding", "rattling", "Warning lights", "vecu:"):
        assert mot not in source, f"shock text copied into the launcher: « {mot} »"


def test_K10_la_campagne_coupe_le_cache():
    """K10 — the cache key carries no duration: a decision from before the shock would be served again."""
    assert "CACHE=0" in _source_cohorte()
