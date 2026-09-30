"""An exhausted execution can be resumed; only a closed archive cannot.

The runner marks "exhausted" WITHOUT sealing the archive (R4: `not exe.cloturee`), precisely so
that it can be resumed when the quota window reopens — or right away, if the exhaustion was
a false diagnosis. On 2026-09-08, `make experience-reprendre` nevertheless answered "nothing to
resume" on such a run: the CLI refused every state of `ETATS_FINAUX`, exhausted
included, contradicting its own message, the spec and the "Resume" button.
"""

from experiences.cli import motif_non_reprenable


class _Exe:
    def __init__(self, etat: str, cloturee: bool = False, nom: str = "2026-09-08_12_01_16"):
        self._etat, self.cloturee, self.nom = etat, cloturee, nom

    def etat(self) -> dict:
        return {"etat": self._etat, "raison": None, "reprise_possible_a": None}


def test_epuisee_non_cloturee_se_reprend():
    assert motif_non_reprenable(_Exe("epuisee")) is None


def test_en_pause_interrompue_et_en_cours_sans_runner_se_reprennent():
    for etat in ("en_pause", "interrompue", "en_cours", "definie", "en_attente_quota"):
        assert motif_non_reprenable(_Exe(etat)) is None, etat


def test_une_archive_cloturee_ne_se_reprend_pas_quel_que_soit_son_etat():
    for etat in ("arretee", "terminee", "epuisee"):
        motif = motif_non_reprenable(_Exe(etat, cloturee=True))
        assert motif and "clôturée" in motif and "Rejouer" in motif and "2026-09-08_12_01_16" in motif, etat


def test_terminee_ne_se_reprend_pas_meme_sans_bloc_cloture():
    """Belt and braces: a finished run is always closed by the runner, but a hand-made etat.json must not reopen it."""
    assert motif_non_reprenable(_Exe("terminee", cloturee=False))
