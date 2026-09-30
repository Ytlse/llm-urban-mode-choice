"""The duration of a shock memory, derived from its severity.

Why this file exists. The paired campaign on the suspect car showed a shock effect that
dies out **on the exact day** the memory leaves the prompt block — 13 April at window 14, 6 April
at window 7. The duration of an effect was thus an integer set in `settings.py`. That 14.6 — the
computed lifetime of the memory — falls close to 14 is the coincidence that masked the defect:
both explanations predicted the same date, and the arm at window 7 separated them.

The numbers tested here are those of the contract, set BEFORE the code:
`force = min(2.8 × (1 + 6 × severity), 30)` · `duration = force × ln(1/0.35)` · `ln(1/0.35) = 1.049822`.
"""

import math
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import noyau as noyau_module
from llm.memory import MemoryEntry, MemoryType
from llm.noyau import MODE_DERIVEE, MODE_FIXE, bloc_changements, duree_service_jours
from settings import settings
from sim_clock import wall_clock

T0 = 1773637200
LN = math.log(1 / 0.35)


def _choc(texte="panne sur voie rapide", gravite=0.70, jours=1.0, force=None):
    """A shock memory. `force=None` = never qualified: it is recomputed from the severity."""
    return MemoryEntry(
        content=texte,
        timestamp=wall_clock(T0) - timedelta(days=jours),
        memory_type=MemoryType.REFLECTION,
        person_id="899549",
        importance=gravite,
        force=force,
    )


@pytest.fixture(autouse=True)
def _propre(monkeypatch):
    noyau_module.reinitialiser()
    monkeypatch.setattr(
        settings.agent, "memoire__mode_fenetre_changements", MODE_DERIVEE, raising=False
    )
    yield
    noyau_module.reinitialiser()


@pytest.fixture
def journal():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(
        lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO"
    )
    yield lignes
    logger.remove(sink)


@pytest.fixture
def regler(monkeypatch):
    def _poser(**kw):
        for nom, valeur in kw.items():
            monkeypatch.setattr(settings.agent, f"memoire__{nom}", valeur, raising=False)
    return _poser


# ══════════════════════ A — the derived duration ════════════════════════════════


def test_A1_la_force_absente_se_recalcule_depuis_la_gravite():
    """A1 — 15.285 d for a severity of 0.70, the reference value (C6 measured)."""
    d = duree_service_jours(_choc(gravite=0.70, force=None))
    assert d.force == pytest.approx(14.56)
    assert d.jours == pytest.approx(15.285, abs=1e-3)
    assert d.borne == ""


def test_A2_une_force_portee_est_utilisee_telle_quelle():
    """A2 — strength may have grown on recall; recomputing it would erase this reinforcement."""
    d = duree_service_jours(_choc(gravite=0.70, force=20.0))
    assert d.force == 20.0
    assert d.jours == pytest.approx(20.0 * LN, abs=1e-6)


def test_A3_trois_gravites_croissantes_donnent_trois_durees_croissantes():
    """A3 — this is the whole thesis of the lot: the duration follows the severity."""
    durees = [duree_service_jours(_choc(gravite=g)).jours for g in (0.30, 0.70, 0.90)]
    assert durees == sorted(durees)
    assert durees[0] == pytest.approx(8.231, abs=1e-3)
    assert durees[1] == pytest.approx(15.285, abs=1e-3)
    assert durees[2] == pytest.approx(18.813, abs=1e-3)


def test_A4_le_plancher_ne_mord_pas_a_gravite_faible():
    """A4 — 3.12 d at severity 0.01: above the floor, hence served as is."""
    d = duree_service_jours(_choc(gravite=0.01))
    assert d.force == pytest.approx(2.968, abs=1e-3)
    assert d.jours == pytest.approx(3.116, abs=1e-3)
    assert d.borne == ""


def test_A5_le_plancher_ne_mord_pas_meme_a_gravite_nulle():
    """A5 — the floor is a safety bound, NOT an observed minimum duration.

    At S0 = 2.8 d, threshold 0.35, a zero-severity memory already lasts 2.94 d. Citing
    "2 days" as a measured minimum duration would be wrong.
    """
    d = duree_service_jours(_choc(gravite=0.0))
    assert d.jours == pytest.approx(2.8 * LN, abs=1e-6)
    assert d.borne == ""


def test_A6_le_plafond_de_duree_ne_mord_plus_jamais(regler):
    """A6 — author's decision of 2026-09-22: the ceiling goes to 50 d and becomes a witness.

    What really bounds the duration is `memoire__force_max_jours`, because
    `duration = strength × 1.0498`. Saturated at 30, strength gives 31.49 d — and no number of
    recalls goes beyond it. The duration ceiling could only bite in the band
    `strength ∈ ]28.58; 30]`, where it shaved off at most 1.49 d.

    ⚠ The former version of this test claimed that without a ceiling "a memory often recalled
    would push back its own deadline indefinitely". That was wrong: `force_apres_rappel` saturates.
    """
    d = duree_service_jours(_choc(gravite=1.0, force=30.0))
    assert d.brute == pytest.approx(31.495, abs=1e-3)
    assert d.jours == pytest.approx(31.495, abs=1e-3), "the 50 d ceiling must no longer bite"
    assert d.borne == ""
    # Severity alone never gets there: 19.6 d of strength at most.
    assert duree_service_jours(_choc(gravite=1.0)).jours == pytest.approx(20.577, abs=1e-3)

    # The MECHANISM stays exercised: set below reachable duration, the ceiling bites and says so.
    regler(plafond_changement_jours=30.0)
    rabote = duree_service_jours(_choc(gravite=1.0, force=30.0))
    assert rabote.jours == 30.0
    assert rabote.borne == "plafond"


def test_A7_un_souvenir_sort_le_jour_que_sa_duree_predit():
    """A7 — severity 0.70: served on day 15, no longer on day 16."""
    assert bloc_changements([_choc(gravite=0.70, jours=15.0)], wall_clock(T0))
    assert bloc_changements([_choc(gravite=0.70, jours=16.0)], wall_clock(T0)) == []


def test_A8_la_date_d_extinction_se_deplace_avec_la_gravite():
    """A8 — the core of E3, exercised as a unit test before the run."""
    quand = wall_clock(T0)
    assert bloc_changements([_choc(gravite=0.90, jours=16.0)], quand)
    assert bloc_changements([_choc(gravite=0.70, jours=16.0)], quand) == []


def test_A9_l_age_se_compte_depuis_l_evenement_pas_depuis_le_rappel():
    """A9 — a shock reread yesterday is not a shock from yesterday.

    The block announces the age of a CHANGE. Reinforcement on recall still plays a part,
    but through the `force`, hence on the duration — not by making the event younger.
    """
    vieux = _choc(gravite=0.70, jours=16.0)
    vieux.dernier_rappel = wall_clock(T0) - timedelta(hours=1)
    assert bloc_changements([vieux], wall_clock(T0)) == []


def test_A10_sous_le_seuil_de_choc_rien_n_entre():
    """A10 — the duration does not redeem a severity below the shock threshold."""
    assert bloc_changements([_choc(gravite=0.10, jours=0.5)], wall_clock(T0)) == []


# ══════════════════════ B — mode, settings and bounds ═══════════════════════════


def test_B1_en_mode_derive_la_fenetre_fixe_n_est_pas_lue(regler):
    """B1 — else two settings would govern one thing, and we could not tell which one acts."""
    souvenir = _choc(gravite=0.70, jours=10.0)
    regler(fenetre_changements_jours=1)
    assert bloc_changements([souvenir], wall_clock(T0))
    regler(fenetre_changements_jours=40)
    assert bloc_changements([souvenir], wall_clock(T0))


def test_B2_le_mode_fixe_ignore_la_gravite(regler):
    """B2 — the methodological control arm must reproduce the campaign of 19 September."""
    regler(mode_fenetre_changements=MODE_FIXE, fenetre_changements_jours=14)
    quand = wall_clock(T0)
    assert bloc_changements([_choc(gravite=0.90, jours=15.0)], quand) == []
    assert bloc_changements([_choc(gravite=0.70, jours=13.0)], quand)


def test_B3_le_mode_fixe_a_zero_reste_l_ablation(regler):
    """B3 — including the memory of the very instant."""
    regler(mode_fenetre_changements=MODE_FIXE, fenetre_changements_jours=0)
    assert bloc_changements([_choc(jours=0.0)], wall_clock(T0)) == []


def test_B4_un_seuil_hors_domaine_est_une_alarme_et_un_repli(regler, journal):
    """B4 — at threshold 1, the duration would be zero: an ablation nobody declared."""
    regler(seuil_service_changement=1.0)
    d = duree_service_jours(_choc(gravite=0.70))
    assert d.jours == pytest.approx(15.285, abs=1e-3)
    assert [m for n, m in journal if n == "ERROR" and "[ALARME]" in m and "seuil" in m]


def test_B5_un_mode_inconnu_ne_s_interprete_pas(regler, journal):
    """B5 — and the faulty value is quoted: one must be able to fix it without guessing it."""
    regler(mode_fenetre_changements="glissante")
    assert bloc_changements([_choc(gravite=0.70, jours=10.0)], wall_clock(T0))
    alarmes = [m for n, m in journal if n == "ERROR" and "[ALARME]" in m]
    assert alarmes and "glissante" in alarmes[0]


def test_B6_plancher_superieur_au_plafond_est_rattrape_et_dit(regler, journal):
    """B6 — without the swap, NO shock memory would ever be served.

    Bounds put back in order: (2; 30), so the duration of 15.29 d passes without being bounded.
    """
    regler(plancher_changement_jours=30.0, plafond_changement_jours=2.0)
    d = duree_service_jours(_choc(gravite=0.70))
    assert d.jours == pytest.approx(15.285, abs=1e-3)
    assert d.borne == ""
    assert [m for n, m in journal if n == "ERROR" and "plancher" in m and "plafond" in m]


def test_B7_les_reglages_sont_relus_a_chaque_appel(regler):
    """B7 — frozen at import, an environment override — hence an arm — would be of no use."""
    souvenir = _choc(gravite=0.70, jours=10.0)
    regler(seuil_service_changement=0.9)  # duration = 14.56 × 0.105 = 1.53 d → floor 2 d
    assert bloc_changements([souvenir], wall_clock(T0)) == []
    regler(seuil_service_changement=0.35)
    assert bloc_changements([souvenir], wall_clock(T0))


def test_B8_le_plafond_a_une_trace_propre(journal, regler):
    """B8 — "the ceiling bit" must be readable in the log, not deduced from a computation.

    The production ceiling (50 d) no longer bites: it is lowered here to exercise the trace. Without
    that, the day the law or the strength ceiling moved, the line would no longer be tested.
    """
    regler(plafond_changement_jours=30.0)
    entrees = [_choc(gravite=1.0, jours=31.0, force=30.0)]
    bloc_changements(entrees, wall_clock(T0), person_id="899549")
    lignes = [m for n, m in journal if "sorti du bloc" in m]
    assert lignes and "plafond" in lignes[0]


# ══════════════════════ C — the exit line ═══════════════════════════════════════


def test_C1_la_ligne_porte_la_cause_et_pas_seulement_la_date(journal):
    """C1 — a duration served without its cause cannot be checked afterwards."""
    bloc_changements([_choc(gravite=0.70, jours=16.0)], wall_clock(T0), person_id="899549")
    ligne = next(m for n, m in journal if "sorti du bloc" in m)
    assert "0.70" in ligne          # the severity
    assert "14.56" in ligne         # the strength
    assert "15.29" in ligne         # the served duration, rounded to the hundredth
    assert "plus aucun souvenir de choc" in ligne


def test_C2_front_montant_une_seule_ligne(journal):
    """C2 — repeated at every decision, the line would drown what it says."""
    entrees = [_choc(gravite=0.70, jours=16.0)]
    bloc_changements(entrees, wall_clock(T0), person_id="899549")
    bloc_changements(entrees, wall_clock(T0) + timedelta(days=1), person_id="899549")
    assert len([m for _, m in journal if "sorti du bloc" in m]) == 1


def test_C3_deux_gravites_sortent_a_deux_dates_et_font_deux_lignes(journal):
    """C3 — and each line names ITS severity: this is what makes E3 readable in the log."""
    # Two distinct EVENTS, hence two distinct timestamps: the rising edge is
    # memorised per (agent, memory instant), and two simultaneous memories make only one.
    leger = _choc("crevaison", gravite=0.70, jours=16.0)
    lourd = _choc("panne réseau", gravite=0.90, jours=16.2)
    bloc_changements([leger, lourd], wall_clock(T0), person_id="899549")
    bloc_changements([leger, lourd], wall_clock(T0) + timedelta(days=3), person_id="899549")
    lignes = [m for _, m in journal if "sorti du bloc" in m]
    assert len(lignes) == 2
    assert any("0.70" in m for m in lignes) and any("0.90" in m for m in lignes)


def test_C4_en_mode_fixe_la_ligne_nomme_la_fenetre(regler, journal):
    """C4 — it does not claim a derived duration it did not compute."""
    regler(mode_fenetre_changements=MODE_FIXE, fenetre_changements_jours=14)
    bloc_changements([_choc(jours=15.0)], wall_clock(T0), person_id="899549")
    ligne = next(m for n, m in journal if "sorti du bloc" in m)
    assert "fenêtre fixe de 14 j" in ligne
    assert "gravité" not in ligne
