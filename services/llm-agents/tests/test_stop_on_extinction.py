"""Early stop fires on a DATED EVENT, never on a statistic.

The core is pure: `analyser` reads lines and returns (extinction seen, lived days elapsed).
The cases come from the real log of the 2026-09-21 campaign.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "experiment"))

from arret_sur_extinction import analyser  # noqa: E402

SORTIE_PARTIELLE = (
    "2026-09-21 17:36:52 | INFO | llm.noyau - [noyau] 861500 : le souvenir de choc du "
    "2026-03-30 est sorti du bloc « ce qui a changé récemment » (durée 15.29 j)."
)
SORTIE_TOTALE = SORTIE_PARTIELLE[:-1] + " — plus aucun souvenir de choc ne pèse sur ses décisions."


def _jour(d):
    return f"2026-09-21 18:00:00 | INFO | [sync] END sim_time={d} 2026, 05:00"


def test_A_une_sortie_partielle_ne_declenche_pas():
    """A memory that exits while others remain served ends nothing."""
    vue, _ = analyser([SORTIE_PARTIELLE, _jour("15 April")])
    assert vue is False


def test_B_la_sortie_totale_declenche():
    vue, _ = analyser([SORTIE_PARTIELLE, SORTIE_TOTALE, _jour("15 April")])
    assert vue is True


def test_C_le_souvenir_declare_declenche_a_sa_date():
    """Recommended key: the agent makes its own severe memories, and the TOTAL exit
    only comes at the end of the run. Measured on 2026-09-21: 5 lived days left versus 13."""
    vue, _ = analyser([SORTIE_PARTIELLE], souvenir_du="2026-03-30")
    assert vue is True
    vue_autre, _ = analyser([SORTIE_PARTIELLE], souvenir_du="2026-04-07")
    assert vue_autre is False, "another memory must not trigger"


def test_D_les_jours_sont_comptes_en_jours_VECUS():
    """The simulation skips weekends: counting calendar days would remove two days out of seven."""
    lignes = [SORTIE_TOTALE] + [_jour(d) for d in
              ("14 April", "14 April", "15 April", "16 April", "17 April")]
    vue, n = analyser(lignes)
    assert vue is True
    assert n == 3, "four distinct dates after the extinction, including that same day"


def test_E_rien_avant_l_extinction_n_est_compte():
    lignes = [_jour("01 April"), _jour("02 April"), SORTIE_TOTALE, _jour("15 April")]
    _, n = analyser(lignes)
    assert n == 0


def test_F_le_journal_reel_du_2026_09_21():
    """End-to-end test: both keys, on the log of the treated arm."""
    p = (Path(__file__).resolve().parents[3] / "experiments" / "archive"
         / "2026-09-21_15_13" / "app.log")
    if not p.is_file():
        import pytest
        pytest.skip("archive missing from this machine")
    with open(p, encoding="utf-8", errors="replace") as f:
        vue_totale, n_totale = analyser(f)
    with open(p, encoding="utf-8", errors="replace") as f:
        vue_declare, n_declare = analyser(f, souvenir_du="2026-03-30")
    assert vue_totale and vue_declare
    assert n_declare > n_totale, (
        "the exit of the DECLARED memory must leave more margin than the total extinction — "
        "that is what makes the early stop worthwhile"
    )


# ══ The wiring into the campaign orchestrator (2026-09-22) ═══════════════════════════════════
# Until now the detector was a script to be run by hand, with `--souvenir-du` to be copied
# by oneself from the trace. A fifty-day campaign is not monitored by hand:
# the orchestrator reads the date in `evenements.jsonl` and stops by itself.

import json  # noqa: E402

from run_sequential_cohort import date_du_souvenir_declare  # noqa: E402


def _trace(tmp_path, dates):
    """A run directory carrying an event trace at the given simulated dates."""
    (tmp_path / "evenements.jsonl").write_text(
        "\n".join(
            json.dumps({"person_id": "861500", "horodatage_simule": f"{d}T14:07:37"})
            for d in dates
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_la_date_surveillee_est_celle_de_la_DERNIERE_application(tmp_path):
    """c6 strikes two days. Watching the first would stop the run with the memory still served."""
    assert date_du_souvenir_declare(_trace(tmp_path, ["2026-03-16", "2026-03-17"])) == "2026-03-17"


def test_sans_trace_aucune_date_donc_aucun_arret(tmp_path):
    """The CONTROL arm undergoes nothing: its trace is empty and it must reach its horizon.

    This is the condition that keeps the paired comparison readable — the control is never
    truncated by surprise, it is the analysis that brings it back to the day of the treated arm.
    """
    assert date_du_souvenir_declare(tmp_path) is None
    (tmp_path / "evenements.jsonl").write_text("", encoding="utf-8")
    assert date_du_souvenir_declare(tmp_path) is None
    assert date_du_souvenir_declare(None) is None


def test_une_ligne_tronquee_ne_fait_pas_tomber_la_surveillance(tmp_path):
    """The trace is written WHILE it is read: the last line may be cut in two."""
    p = _trace(tmp_path, ["2026-03-16"])
    with open(p / "evenements.jsonl", "a", encoding="utf-8") as f:
        f.write('{"person_id": "861500", "horodatage_sim')
    assert date_du_souvenir_declare(p) == "2026-03-16"


def test_le_detecteur_ne_compte_que_les_jours_qui_suivent_la_date_surveillee():
    """End to end: the date comes from the trace, and `analyser` finds it in the log."""
    sortie = (
        "2026-09-22 10:00:00 | INFO | llm.noyau - [noyau] 861500 : le souvenir de choc du "
        "2026-03-17 est sorti du bloc « ce qui a changé récemment » (durée 16.17 j)."
    )
    lignes = [_jour("10 April"), sortie, _jour("10 April"), _jour("11 April"), _jour("13 April")]
    vue, ecoules = analyser(lignes, "2026-03-17")
    assert vue is True
    # Three dates after the extinction, the first counting as zero: two lived days.
    assert ecoules == 2
    # And the memory of ANOTHER date triggers nothing.
    assert analyser(lignes, "2026-03-16") == (False, 0)
