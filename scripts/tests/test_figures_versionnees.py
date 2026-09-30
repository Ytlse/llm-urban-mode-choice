"""A figure regenerated into a versioned folder must announce it.

`plot_experiences.py` and `plot_familles.py` write BY DEFAULT into `figures/` of the papers
repo (`PAPER_DIR`, ticket 115), which is tracked by git. Each committed regeneration adds one more blob, permanently.
The warning blocks nothing: it makes visible, at write time, a cost that would otherwise
only show up in `git status` — or never.
"""

import logging
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis.figures_versionnees import signaler
from scripts.depot_papiers import chemin_papier


def test_une_figure_hors_git_ne_declenche_rien(tmp_path, caplog):
    figure = tmp_path / "exploration.png"
    figure.write_bytes(b"x" * 1024)
    with caplog.at_level(logging.INFO, logger="figures"):
        signaler([figure])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Figure écrite" in r.message for r in caplog.records), \
        "success is logged too, not only the anomaly"


def test_une_figure_suivie_par_git_avertit_et_chiffre(caplog):
    suivie = chemin_papier("figures", "comparaison_experiences.png")
    if not suivie.is_file():
        import pytest
        pytest.skip(f"reference figure missing from disk (papers repo: {suivie.parent})")
    with caplog.at_level(logging.INFO, logger="figures"):
        signaler([suivie])
    alertes = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(alertes) == 1, "an overwritten versioned figure must warn once"
    # The message must carry what is needed to ACT: a size and the way out.
    assert "Ko" in alertes[0].getMessage()
    assert "--sortie" in alertes[0].getMessage()


def test_git_indisponible_ne_fait_pas_echouer(tmp_path, monkeypatch, caplog):
    """Fail-open: a lost warning is better than a figure not produced."""
    import subprocess

    from scripts.analysis import figures_versionnees

    def tombe(*a, **k):
        raise OSError("git introuvable")

    monkeypatch.setattr(subprocess, "run", tombe)
    figure = tmp_path / "f.png"
    figure.write_bytes(b"x")
    with caplog.at_level(logging.INFO, logger="figures"):
        figures_versionnees.signaler([figure])  # must not raise
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_les_generateurs_ecrivent_par_defaut_dans_un_dossier_versionne():
    """The safeguard only makes sense as long as this holds: if the default output leaves
    `figures/` of the papers repo, this test turns red and the rule of step 6 must be revisited."""
    for module in ("plot_experiences", "plot_familles"):
        source = (RACINE / "scripts" / "analysis" / f"{module}.py").read_text(encoding="utf-8")
        assert 'chemin_papier("figures"' in source, f"{module}: default output moved"
        assert "signaler(ecrits)" in source, f"{module}: warning unplugged"
