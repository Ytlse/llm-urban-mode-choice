"""A departure postponed by the calendar is not an experienced delay.

`expected_arrive_at` is computed by GAMA from the original `schedule_at` and is never
recomputed when the departure is postponed — not by D+1 wrap-around, not by `no_weekend_departures`.
The 2026-09-23 run carried two of them, found in `gama_arrivals.csv`: 23.6 h and 71.6 h of
"delay" for twelve-minute trips. Each produced a severity of 0.70 and a
"severe" memory in the baseline, indistinguishable from the shock declared at 0.75 — and, in the
agent's prompt, a "Late by: 23 hours" it had not experienced.
"""

from text_helper.models.arrival import (
    SEUIL_REPORT_CALENDAIRE_S,
    EnvObArrival,
    retard_d_arrivee,
)

H = 3600
J = 86400


# ── The two MEASURED cases of the 2026-09-23 run ─────────────────────────────────────────

# schedule_at Wed 18 Mar 19:35 · started_at Thu 19 Mar 19:15 · expected Wed 18 Mar 19:50
# arrived Thu 19 Mar 19:27 — D+1 wrap-around, 12-min trip versus 15 planned.
BOUCLAGE_J1 = dict(
    schedule_at=1773862500, started_at=1773947700,
    expected_arrive_at=1773863400, arrive_at=1773948420,
)
# schedule_at Fri 20 Mar 19:35 · started_at Mon 23 Mar 19:15 — weekend postponement.
REPORT_WEEKEND = dict(
    schedule_at=1774035300, started_at=1774293300,
    expected_arrive_at=1774036200, arrive_at=1774294020,
)


def test_R1_le_bouclage_J1_ne_produit_aucun_retard():
    """23.6 h of "delay" for a trip that arrived AHEAD of its own departure."""
    brut = BOUCLAGE_J1["arrive_at"] - BOUCLAGE_J1["expected_arrive_at"]
    assert brut > 23 * H, "the measured case must indeed carry the original defect"
    assert retard_d_arrivee(**BOUCLAGE_J1) == 0


def test_R2_le_report_de_week_end_ne_produit_aucun_retard():
    """71.6 h of "delay" for a Friday evening trip played on Monday."""
    brut = REPORT_WEEKEND["arrive_at"] - REPORT_WEEKEND["expected_arrive_at"]
    assert brut > 71 * H, "the measured case must indeed carry the original defect"
    assert retard_d_arrivee(**REPORT_WEEKEND) == 0


def test_R3_ces_deux_cas_ne_produisent_plus_de_souvenir_grave():
    """The point that matters for the experiment: no more severity in the baseline.

    0.70 was the value served, versus 0.75 for the declared c3 shock: at five hundredths, the
    baseline became indistinguishable from what is injected.
    """
    from llm.gravite import gravite_deterministe

    for cas in (BOUCLAGE_J1, REPORT_WEEKEND):
        avant, _ = gravite_deterministe(
            retard_s=float(cas["arrive_at"] - cas["expected_arrive_at"])
        )
        apres, _ = gravite_deterministe(retard_s=float(retard_d_arrivee(**cas)))
        assert avant >= 0.68, f"the measured case was worth {avant:.2f} before the fix"
        assert apres == 0.0


# ── What MUST NOT change ─────────────────────────────────────────────────────────────────


def test_R4_un_glissement_ordinaire_reste_un_retard_vecu():
    """An activity that overruns by twenty minutes: the agent did experience it."""
    assert retard_d_arrivee(
        arrive_at=1_000_000 + 20 * 60, expected_arrive_at=1_000_000,
        schedule_at=999_000, started_at=999_000 + 20 * 60,
    ) == 20 * 60


def test_R5_un_vrai_retard_sur_un_depart_a_l_heure_est_conserve():
    assert retard_d_arrivee(
        arrive_at=1_000_000 + 1800, expected_arrive_at=1_000_000,
        schedule_at=998_200, started_at=998_200,
    ) == 1800


def test_R6_une_arrivee_en_avance_vaut_zero_et_non_un_retard_negatif():
    """Otherwise an early arrival would offset a real incident in the same entry."""
    assert retard_d_arrivee(
        arrive_at=1_000_000 - 600, expected_arrive_at=1_000_000,
        schedule_at=998_200, started_at=998_200,
    ) == 0


def test_R7_sans_les_deux_champs_le_calcul_est_celui_d_avant():
    """No existing caller changes behaviour (backward compatibility)."""
    assert retard_d_arrivee(arrive_at=1_000_000 + 900, expected_arrive_at=1_000_000) == 900
    assert retard_d_arrivee(
        arrive_at=1_000_000 + 900, expected_arrive_at=1_000_000, schedule_at=999_000,
    ) == 900


def test_R8_le_seuil_separe_bien_les_deux_familles():
    """Just below the threshold: experienced. Just above: postponed."""
    base = 1_000_000
    sous = retard_d_arrivee(
        arrive_at=base + SEUIL_REPORT_CALENDAIRE_S - 1, expected_arrive_at=base,
        schedule_at=base, started_at=base + SEUIL_REPORT_CALENDAIRE_S - 1,
    )
    dessus = retard_d_arrivee(
        arrive_at=base + SEUIL_REPORT_CALENDAIRE_S, expected_arrive_at=base,
        schedule_at=base, started_at=base + SEUIL_REPORT_CALENDAIRE_S,
    )
    assert sous == SEUIL_REPORT_CALENDAIRE_S - 1
    assert dessus == 0


# ── The text the agent READS ─────────────────────────────────────────────────────────────


def _observation(**kw) -> EnvObArrival:
    return EnvObArrival(
        type="arrival", timestamp=kw["arrive_at"], purpose="home",
        duration=720.0, plan_duration=900.0, **kw,
    )


def test_R9_le_prompt_ne_dit_plus_un_retard_que_l_agent_n_a_pas_vecu():
    """The template renders "Late by: …" from `ob.late`: that went into its memory."""
    ob = _observation(**REPORT_WEEKEND)
    assert ob.late == 0
    assert not ob.is_late
    assert "Late by" not in ob.describe()
    assert "On time" in ob.describe()


def test_R10_un_vrai_retard_se_dit_toujours_dans_le_prompt():
    ob = _observation(
        schedule_at=998_200, started_at=998_200,
        expected_arrive_at=1_000_000, arrive_at=1_000_000 + 1800,
    )
    assert ob.late == 1800
    assert "Late by" in ob.describe()
