"""Experiment parallelisation: scheduler.

`tour()` is tested with its effects injected (no Docker): we check that it reconciles then
promotes in FIFO order the experiments whose keys are free, and that it relaunches each promoted one.
"""

from __future__ import annotations

import pytest

from experiences import ordonnanceur as O
from experiences import reservations as R
from experiences.archive import Execution


@pytest.fixture
def experiences_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    return tmp_path


def _execution(experiences_dir, nom):
    return Execution.creer(
        experiences_dir / nom,
        experience={"nom": nom},
        empreintes={},
        regime_demande={},
        sources_alea={},
    )


def test_R2b_tour_promeut_les_pretes_en_fifo(experiences_dir):
    a = _execution(experiences_dir, "A")
    b = _execution(experiences_dir, "B")
    c = _execution(experiences_dir, "C")
    # A holds `google`; B then C wait for the SAME key.
    assert R.admettre({"google"}, a.dossier, "A") == "lance"
    assert R.admettre({"google"}, b.dossier, "B") == "file"
    assert R.admettre({"google"}, c.dossier, "C") == "file"

    relances: list[str] = []
    reconciliations = {"n": 0}

    def faux_reconcilieur():
        reconciliations["n"] += 1

    def faux_lanceur(entree):
        relances.append(entree["exp"])

    # Key held by A: nothing is promoted (R2c, no preemption).
    assert O.tour(reconcilieur=faux_reconcilieur, lanceur=faux_lanceur) == []
    assert reconciliations["n"] == 1
    assert [e["exp"] for e in R.lister_file()] == ["B", "C"]

    # A releases `google`. The fake launcher does not reserve; we check FIFO order over two rounds:
    # B first, then C.
    R.liberer(a.dossier)
    assert O.tour(reconcilieur=faux_reconcilieur, lanceur=faux_lanceur) == ["B"]
    assert O.tour(reconcilieur=faux_reconcilieur, lanceur=faux_lanceur) == ["C"]
    assert relances == ["B", "C"]
    assert R.lister_file() == []


def test_argv_lancer_reporte_les_options():
    entree = {"exp": "Exp", "args": {"reprendre": True, "accepter_perime": True}}
    argv = O._argv_lancer(entree)
    i = argv.index("lancer")
    assert argv[i : i + 3] == ["lancer", "--experience", "Exp"]
    assert "--reprendre" in argv
    assert "--accepter-perime" in argv
    assert "--attendre-fenetre" not in argv



def test_argv_lancer_transmet_l_attente_de_fenetre_dans_les_deux_sens():
    """The parser default is "do not wait": a request that wanted to wait must
    say so, otherwise it loses it at promotion."""
    oui = O._argv_lancer({"exp": "Exp", "args": {"attendre_fenetre": True}})
    non = O._argv_lancer({"exp": "Exp", "args": {"attendre_fenetre": False}})
    assert "--attendre-fenetre" in oui and "--ne-pas-attendre-fenetre" not in oui
    assert "--ne-pas-attendre-fenetre" in non and "--attendre-fenetre" not in non


def test_une_relance_de_l_ordonnanceur_garde_sa_sortie(experiences_dir, monkeypatch):
    """Its output went to /dev/null: the relaunch of go123 on 29/09 left no trace."""
    (experiences_dir / "A").mkdir()
    (experiences_dir / "A" / "experience.yaml").write_text("nom: A\n", encoding="utf-8")
    vus: list[dict] = []
    monkeypatch.setattr(O.subprocess, "Popen", lambda argv, **kw: vus.append(kw))
    O._lancer_detache({"exp": "A", "args": {}})
    flux = vus[-1]["stdout"]
    assert flux is not O.subprocess.DEVNULL
    assert list((experiences_dir / "A" / "lancements").glob("*.log"))


def test_le_defaut_du_parseur_est_de_ne_pas_attendre_et_son_aide_le_dit():
    from experiences.cli import construire_parser

    p = construire_parser()
    assert p.parse_args(["lancer", "--experience", "x"]).attendre_fenetre is False
    assert p.parse_args(["lancer", "--experience", "x", "--attendre-fenetre"]).attendre_fenetre
    lancer = next(a for a in p._subparsers._group_actions[0].choices.values()
                  if any(o.dest == "reprendre" for o in a._actions))
    aide = next(o.help for o in lancer._actions if "--attendre-fenetre" in o.option_strings)
    assert "désormais le défaut" not in aide, "the help announced the opposite of the behaviour"
