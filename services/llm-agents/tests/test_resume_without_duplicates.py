"""A warm resumption no longer duplicates replayed trips.

What these tests would have caught: `restaurer_si_demande` only set `moves.csv` aside in the
"no resume point found" branch. When a point IS found, GAMA still restarts from
its t0 and replays the days already lived; `move_logger` knows nothing about the freeze, so the
replayed trips were appended to the file, indistinguishable from the originals. That is exactly
the defect that had duplicated 84 trips on a thirty-day run — and the function written to
prevent it was not called on that path.

Each test of the first class first checks that the case is indeed the one believed, then that
the defect is gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urban_mobility_agents.utils import reprise

LIGNES = (
    "Temps simulé,Mode de transport Choisi,Méthode de sélection\n1774000000,Train,LLM\n"
)


@pytest.fixture(autouse=True)
def _etat_propre():
    reprise.reinitialiser()
    yield
    reprise.reinitialiser()


def _point_valide(workdir: Path, jour: int = 17, timestamp: int = 1774000000) -> Path:
    """A minimal but VALID resume point: `reprise.json` is what makes it valid."""
    point = workdir / reprise.POINTS / f"jour_{jour:03d}"
    point.mkdir(parents=True)
    (point / reprise.DESCRIPTION).write_text(
        json.dumps(
            {
                "jour_simule": jour,
                "timestamp_simule": timestamp,
                "horodatage_simule": "2026-04-01T03:00:00",
            }
        ),
        encoding="utf-8",
    )
    return point


class TestLesDeuxBranchesEcartentLesSorties:
    def test_avec_un_point_valide_moves_est_ecarte(self, tmp_path):
        """THE FIXED CASE: a point found does not exempt from setting the measurements aside."""
        _point_valide(tmp_path)
        (tmp_path / "moves.csv").write_text(LIGNES, encoding="utf-8")

        meta = reprise.restaurer_si_demande(tmp_path, reprise_demandee=True)

        assert meta is not None, "the point should have been restored"
        assert not (tmp_path / "moves.csv").exists(), (
            "moves.csv stayed in place: the replay will append duplicate trips to it"
        )
        ecartes = list(tmp_path.glob("moves.csv.*.avant_rejeu"))
        assert len(ecartes) == 1
        assert ecartes[0].read_text(encoding="utf-8") == LIGNES, (
            "the measurements are not lost"
        )

    def test_sans_point_le_comportement_du_077_est_conserve(self, tmp_path):
        """The branch that already worked must not regress."""
        (tmp_path / "moves.csv").write_text(LIGNES, encoding="utf-8")

        meta = reprise.restaurer_si_demande(
            tmp_path, reprise_demandee=True, alarme_si_absent=True
        )

        assert meta is None
        assert not (tmp_path / "moves.csv").exists()
        assert len(list(tmp_path.glob("moves.csv.*.avant_rejeu"))) == 1


class TestCeQuiNeDoitPasChanger:
    def test_un_run_neuf_ne_touche_a_rien(self, tmp_path):
        """`reprise_demandee=False`: no replay, hence no measurement to set aside."""
        (tmp_path / "moves.csv").write_text(LIGNES, encoding="utf-8")

        assert reprise.restaurer_si_demande(tmp_path, reprise_demandee=False) is None

        assert (tmp_path / "moves.csv").exists()
        assert not list(tmp_path.glob("moves.csv.*.avant_rejeu"))

    def test_un_init_de_run_neuf_sans_point_ne_touche_a_rien(self, tmp_path):
        """`alarme_si_absent=False` is the `/init` of a fresh run: it replays nothing."""
        (tmp_path / "moves.csv").write_text(LIGNES, encoding="utf-8")

        assert (
            reprise.restaurer_si_demande(
                tmp_path, reprise_demandee=True, alarme_si_absent=False
            )
            is None
        )

        assert (tmp_path / "moves.csv").exists()

    def test_le_gel_est_pose_apres_la_restauration(self, tmp_path):
        """Without a freeze, the replay would rewrite memories already written."""
        _point_valide(tmp_path, jour=17, timestamp=1774000000)

        reprise.restaurer_si_demande(tmp_path, reprise_demandee=True)

        assert reprise.gel_actif() is True
        assert reprise.point_de_reprise_timestamp() == 1774000000

    def test_le_point_le_plus_recent_gagne(self, tmp_path):
        """Removing a point is the only way to resume earlier: that must remain true."""
        _point_valide(tmp_path, jour=16, timestamp=1773900000)
        _point_valide(tmp_path, jour=17, timestamp=1774000000)

        meta = reprise.restaurer_si_demande(tmp_path, reprise_demandee=True)

        assert meta["jour_simule"] == 17

    def test_un_point_sans_description_n_est_pas_valide(self, tmp_path):
        """A half-written point would be worse than no point: it would be restored silently."""
        (tmp_path / reprise.POINTS / "jour_018").mkdir(parents=True)
        _point_valide(tmp_path, jour=17, timestamp=1774000000)

        meta = reprise.restaurer_si_demande(tmp_path, reprise_demandee=True)

        assert meta["jour_simule"] == 17, (
            "point 018 without a description should have been ignored"
        )
