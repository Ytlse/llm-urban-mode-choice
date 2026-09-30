"""Rewriting the terminal time of a frozen set (alignment on EMC²).

What is locked here are the properties without which the measurement would be worth
nothing:

- **the rendering invariant**: the displayed total is EXACTLY the sum of the displayed
  sub-steps. It is an acceptance criterion, and it holds because all
  the written values are multiples of 60 s — `floor(a + k·60) == floor(a) + k`;
- **driving time is intact**: we cannot recompute it without replaying the
  routing, so touching it would invalidate the comparison;
- **the draw is deterministic**: two runs give the same set, otherwise the store's
  eval cache becomes wrong;
- **only one variable moves**: the rendering structure is preserved, zero sub-bullets
  included. Removing the zero components would be a second change, and the A/B
  would measure two things at once;
- **the ring is read, not guessed**: it is the egress applied by the config that
  identifies it.

Offline, without the PROGEDO data: the law is a stub whose answer is known.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pytest

CALIB = Path(__file__).resolve().parents[2] / "prompt_calibration"
if str(CALIB) not in sys.path:
    sys.path.insert(0, str(CALIB))

rewrite = pytest.importorskip("rewrite_terminal_time")


class _Law:
    """Test law: `minutes` is certain, which makes every assertion exact."""

    def __init__(self, minutes: int):
        self._pmf = {str(minutes): 1.0}

    def pmf(self, mode: str, end: str, crown):  # noqa: ARG002 — signature du vrai objet
        return dict(self._pmf)


OPTION = """--- agent_id=42 | Destination : work ---
- [0] foot: Durée estimée : 25 minutes. Distance : 2.0 km.
- [1] car: Temps de trajet : 15 minutes, dont 10 minutes d'accès et de stationnement. Distance : 5.0 km.
    · Rejoindre la voiture : 3 minutes.
    · Conduite : 5 minutes.
    · Stationnement et marche jusqu'à 'work' : 7 minutes.
- [2] bicycle: Temps de trajet : 12 minutes, dont 2 minutes d'accès et d'attache. Distance : 2.0 km.
    · Déverrouiller le vélo : 1 minute.
    · Trajet à vélo : 10 minutes.
    · Attacher le vélo à 'work' : 1 minute.
"""


def _minutes(text: str) -> int:
    return rewrite.to_seconds(text) // 60


def _car_block(section: str) -> str:
    match = rewrite.CAR_OPTION.search(section)
    assert match, "car option not found"
    return match.group(0)


def _rewrite(minutes: int, section: str = OPTION,
             modes: tuple[str, ...] = ("car",)) -> tuple[str, Counter]:
    stats: Counter = Counter()
    out = rewrite.rewrite_section(section, _Law(minutes), "42", "0", stats, modes)
    return out, stats


class TestInvariantDuRendu:

    @pytest.mark.parametrize("minutes", [0, 1, 2, 5])
    def test_le_total_est_la_somme_des_sous_etapes(self, minutes):
        """The invariant on which the whole rendering rests."""
        out, _ = _rewrite(minutes)
        block = _car_block(out)
        # Capture ONLY the leading duration: `[^.]+` would also swallow the clause
        # `dont X minutes d'accès`, and `to_seconds` would sum both numbers —
        # the test would announce 25 minutes where the header shows 15.
        head = re.search(r"Temps de trajet\s*:\s*([^,.]+(?:,\s*\d+\s*minutes?)?)",
                         block).group(1)
        head = re.split(r",\s*dont", head)[0]
        total = _minutes(head)
        steps = [_minutes(m.group(1)) for m in
                 re.finditer(r"·[^:]+:\s*([^.\n]+)\.", block)]
        assert total == sum(steps), (total, steps, block)

    @pytest.mark.parametrize("minutes", [0, 1, 3])
    def test_le_temps_de_conduite_est_intact(self, minutes):
        """We cannot recompute it: touching it would invalidate the comparison."""
        out, _ = _rewrite(minutes)
        drive = rewrite.DRIVE_STEP.search(_car_block(out))
        assert _minutes(drive.group("val")) == 5

    def test_la_clause_terminale_disparait_quand_le_terminal_est_nul(self):
        """`dont 0 minute d'accès` would be noise: the clause is removed."""
        out, _ = _rewrite(0)
        assert "d'accès et de stationnement" not in _car_block(out)

    def test_la_clause_terminale_est_presente_sinon(self):
        out, _ = _rewrite(2)
        block = _car_block(out)
        assert "dont 4 minutes d'accès et de stationnement" in block

    def test_la_distance_est_conservee(self):
        out, _ = _rewrite(1)
        assert "Distance : 5.0 km." in _car_block(out)


class TestUneSeuleVariable:

    def test_les_sous_puces_nulles_sont_conservees(self):
        """Removing the zero components would change the STRUCTURE along with
        the durations: the A/B would measure two things at once."""
        out, _ = _rewrite(0)
        block = _car_block(out)
        assert "Rejoindre la voiture : 0 minute." in block
        assert "Stationnement et marche jusqu'à 'work' : 0 minute." in block

    def test_les_autres_modes_ne_sont_pas_touches(self):
        """Bike and walking also carry a terminal time; it is not the subject
        of this rewrite, and changing it would conflate two corrections."""
        out, _ = _rewrite(0)
        assert "- [0] foot: Durée estimée : 25 minutes. Distance : 2.0 km." in out
        assert "· Déverrouiller le vélo : 1 minute." in out
        assert "dont 2 minutes d'accès et d'attache" in out

    def test_le_nombre_doptions_est_inchange(self):
        out, _ = _rewrite(2)
        assert out.count("- [") == OPTION.count("- [")


class TestDeterminisme:

    def test_deux_reecritures_donnent_le_meme_texte(self):
        first, _ = _rewrite(2)
        second, _ = _rewrite(2)
        assert first == second

    def test_le_tirage_suit_la_loi(self):
        """Over many keys, the drawn frequency must approach the served law."""
        pmf = {"0": 0.9, "5": 0.1}
        drawn = Counter(rewrite.draw_minutes(pmf, f"clé-{i}") for i in range(4000))
        assert 0.86 < drawn[0] / 4000 < 0.94, drawn

    def test_une_loi_certaine_rend_sa_valeur(self):
        assert rewrite.draw_minutes({"3": 1.0}, "quelconque") == 3

    def test_deux_bouts_tirent_independamment(self):
        """Access and egress have distinct keys: otherwise they would be correlated at 1
        and the terminal time would carry only one source of variation."""
        pmf = {"0": 0.5, "4": 0.5}
        pairs = {(rewrite.draw_minutes(pmf, f"{i}:access"),
                  rewrite.draw_minutes(pmf, f"{i}:egress")) for i in range(200)}
        assert len(pairs) == 4, pairs


class TestLectureDeLaCouronne:

    def test_l_egression_identifie_la_couronne_de_destination(self):
        """It is the inverse table of `terminal_time.yaml`, and it identifies uniquely."""
        assert rewrite.EGRESS_TO_CROWN[7] == "Toulouse"
        assert rewrite.EGRESS_TO_CROWN[4] == "1ere couronne"
        assert rewrite.EGRESS_TO_CROWN[3] == "2eme couronne"
        assert rewrite.EGRESS_TO_CROWN[1] == "3eme couronne"

    def test_la_couronne_lue_est_comptee(self):
        _, stats = _rewrite(1)
        assert stats["couronne_dest:Toulouse"] == 1, dict(stats)

    def test_une_option_non_decomposee_est_laissee_telle_quelle(self):
        """Better an identifiable option than an invented breakdown."""
        section = ("- [0] car: Durée estimée : 27 minutes. Distance : 11.3 km.\n")
        out, stats = _rewrite(3, section)
        assert out == section
        assert stats["options_car_non_decomposees"] == 1


class TestHumanize:

    @pytest.mark.parametrize("seconds,expected", [
        (0, "0 minute"), (60, "1 minute"), (120, "2 minutes"),
        (3600, "1 hour"), (3660, "1 hour, 1 minute"), (7320, "2 hours, 2 minutes"),
    ])
    def test_rendu_des_durees(self, seconds, expected):
        assert rewrite.humanize(seconds) == expected

    def test_aller_retour_secondes(self):
        for seconds in (0, 60, 300, 3600, 3660):
            assert rewrite.to_seconds(rewrite.humanize(seconds)) == seconds


# ── Multi-mode alignment (set v7) ────────────────────────────────────────────

class TestPerimetreDesModes:
    """The rewritten scope must be the declared one, no more, no less.

    This is the defect that motivated this class: the fix shipped to production
    (`tt3`) aligned the car AND the bike, whereas the measured set (`v6`) aligned
    only the car. The published figure therefore did not describe what was running.
    """

    def test_voiture_seule_laisse_le_velo_intact(self):
        out, stats = _rewrite(0, modes=("car",))
        assert "· Déverrouiller le vélo : 1 minute." in out
        assert "· Attacher le vélo à 'work' : 1 minute." in out
        assert "dont 2 minutes d'accès et d'attache" in out
        assert stats["options_bicycle"] == 0

    def test_les_deux_modes_alignent_les_deux(self):
        out, stats = _rewrite(0, modes=("car", "bicycle"))
        assert "· Déverrouiller le vélo : 0 minute." in out
        assert "· Attacher le vélo à 'work' : 0 minute." in out
        assert "· Rejoindre la voiture : 0 minute." in out
        assert stats["options_car"] == 1 and stats["options_bicycle"] == 1

    def test_chaque_mode_garde_sa_clause_terminale(self):
        """`d'accès et de stationnement` for the car, `d'accès et d'attache`
        for the bike: swapping them would give a text that production never
        writes."""
        out, _ = _rewrite(2, modes=("car", "bicycle"))
        assert "dont 4 minutes d'accès et de stationnement" in out
        assert "dont 4 minutes d'accès et d'attache" in out

    def test_le_temps_de_trajet_de_chaque_mode_est_intact(self):
        """Driving and bike trip cannot be recomputed without replaying the routing."""
        out, _ = _rewrite(0, modes=("car", "bicycle"))
        assert "· Conduite : 5 minutes." in out
        assert "· Trajet à vélo : 10 minutes." in out

    def test_la_marche_nest_jamais_touchee(self):
        """Walking is door-to-door: it has no terminal time to align."""
        for modes in (("car",), ("car", "bicycle")):
            out, _ = _rewrite(0, modes=modes)
            assert "- [0] foot: Durée estimée : 25 minutes. Distance : 2.0 km." in out

    def test_les_deux_modes_tirent_independamment(self):
        """The draw key carries the mode: without this prefix, car and bike of the same
        option would receive the SAME terminal time, and the set would carry only one
        source of variation instead of two."""
        pmf = {"0": 0.5, "4": 0.5}
        pairs = {(rewrite.draw_minutes(pmf, f"car:{i}:access"),
                  rewrite.draw_minutes(pmf, f"bicycle:{i}:access")) for i in range(200)}
        assert len(pairs) == 4, pairs

    def test_un_mode_inconnu_est_refuse(self):
        with pytest.raises(KeyError):
            _rewrite(0, modes=("scooter",))

    def test_les_specs_couvrent_les_deux_modes_vehicules(self):
        assert set(rewrite.MODE_SPECS) == {"car", "bicycle"}
        for mode, spec in rewrite.MODE_SPECS.items():
            assert {"option", "access", "egress", "main", "clause",
                    "spatialise"} <= set(spec)
        # Only the car is spatialized: the bike has only 2 047 surveyed trips,
        # its per-ring cells would be too thin.
        assert rewrite.MODE_SPECS["car"]["spatialise"] is True
        assert rewrite.MODE_SPECS["bicycle"]["spatialise"] is False
