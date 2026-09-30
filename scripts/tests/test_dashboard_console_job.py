"""A job's console stays open once opened (📟 Activités en cours).

The jobs panel lives in a `run_every="10s"` fragment, and the `expanded` argument of an
`st.expander` wins on EVERY pass. As long as it was `index == 0`, only the console of the most
recent job could stay open: opening another job's console closed it again ten seconds
later. Real case of 2026-09-07 — two experiments in parallel, impossible to read the
log of the older one.

The fold chosen by the reader is now written to `session_state` (`key` +
`on_change="rerun"`) and read back on every pass; `expanded` is only the default of the first
display.
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

SCRIPT = '''
import sys, time
from pathlib import Path
sys.path.insert(0, "__RACINE__")
import streamlit as st
from scripts.dashboard import app, runner

class FauxJob:
    def __init__(self, ident, label):
        self.id, self.label = ident, label
        self.argv, self.cwd = [f"EXP={label}"], Path("__RACINE__")
        self.log_path = Path("__RACINE__") / "experiments" / ".dashboard" / "faux.log"
        self.started_at, self.finished_at = time.time() - 60, None
        self.returncode, self.error, self.flags = None, None, ()
        self.running = True
        self.duration = 60.0
        self.command_line = f"make {label}"
        self.state = "en cours"

# Deux lancements en parallèle, le plus récent en tête (comme `par_priorite`).
for index, job in enumerate([FauxJob("recent", "gemini"), FauxJob("ancien", "mistral")]):
    app.render_job(job, expanded=(index == 0))
'''


def _apptest():
    # The log does not need to exist (`runner.tail` returns "log indisponible"), but
    # its path must live under the repo root: `render_job` displays it as relative.
    # `import app` renders the whole dashboard (≈ 5 s on a loaded machine): same margin as
    # the other AppTests of the suite, instead of the default 3 s.
    at = AppTest.from_string(SCRIPT.replace("__RACINE__", str(RACINE)), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _plis(at) -> dict[str, bool]:
    """The fold of BOTH panels of this test — importing `app` renders the whole page, panels too."""
    plis = {}
    for e in at.expander:
        for nom in ("gemini", "mistral"):
            if f" {nom} — " in e.proto.label:
                plis[nom] = e.proto.expanded
    return plis


def test_par_defaut_seul_le_job_de_tete_est_ouvert():
    at = _apptest()
    assert _plis(at) == {"gemini": True, "mistral": False}


def test_le_pli_choisi_par_le_lecteur_survit_au_rafraichissement():
    """The fragment replays every 10 s: it must no longer force anything back."""
    at = _apptest()
    at.session_state["job-volet-ancien"] = True   # the reader opens the mistral console
    at.run()
    assert _plis(at)["mistral"], "the open console stays open on the next pass"


def test_le_lecteur_peut_aussi_refermer_le_job_de_tete():
    """The default must not force-reopen what was just closed."""
    at = _apptest()
    at.session_state["job-volet-recent"] = False
    at.run()
    assert not _plis(at)["gemini"]
