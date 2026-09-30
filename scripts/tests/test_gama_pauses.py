"""The headless launcher tolerates GAMA's normal pauses.

By default `websockets.connect` closes the connection if a ping gets no answer for 20 s, and
GAMA Server stops the experiment whose client disconnected. Yet GAMA blocks while
it waits for the LLM decisions: with `CACHE=0` this is the NORMAL regime, and when tokens
run short the wait is longer still — the repository rule being to wait for the
renewal rather than degrade.

Measured on run 2026-09-16_12_48: 23 pauses, median 9.0 s, maximum 40.9 s, three
above 20 s. The run died at the third one.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
SOURCE = RACINE / "scripts" / "gama" / "launch_headless.py"

VINGT_MINUTES = 20 * 60


def _charger(monkeypatch, **env):
    """Reloads the module with a given environment (thresholds are read at import time)."""
    for cle, valeur in env.items():
        monkeypatch.setenv(cle, valeur)
    spec = importlib.util.spec_from_file_location("launch_headless_sous_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_headless_sous_test"] = module
    spec.loader.exec_module(module)
    return module


def _kwargs_de_connect(module, monkeypatch) -> dict:
    """Captures the arguments passed to websockets.connect without opening a socket."""
    vus: dict = {}

    async def faux_connect(url, **kwargs):
        vus.update(kwargs)
        vus["url"] = url
        return object()

    monkeypatch.setattr(module.websockets, "connect", faux_connect)
    asyncio.run(module.connect_with_retries())
    return vus


class TestToleranceAuxPauses:
    def test_B1_le_ping_tolere_au_moins_vingt_minutes(self, monkeypatch):
        module = _charger(monkeypatch)
        kwargs = _kwargs_de_connect(module, monkeypatch)
        assert kwargs.get("ping_timeout") is not None, (
            "without an explicit ping_timeout, the websockets default (20 s) kills the run "
            "at the first somewhat long pause"
        )
        assert kwargs["ping_timeout"] >= VINGT_MINUTES

    def test_B2_le_ping_reste_emis(self, monkeypatch):
        """We loosen the answer delay, we do not switch off detection: a connection that
        is really dead must still show."""
        module = _charger(monkeypatch)
        kwargs = _kwargs_de_connect(module, monkeypatch)
        interval = kwargs.get("ping_interval", 20)
        assert interval is not None and interval > 0

    def test_B3_le_seuil_est_reglable_et_vaut_1200_par_defaut(self, monkeypatch):
        module = _charger(monkeypatch)
        assert module.PING_TIMEOUT_S == 1200

        autre = _charger(monkeypatch, GAMA_PING_TIMEOUT_S="2400")
        assert autre.PING_TIMEOUT_S == 2400


class TestLeSymptomeResteVisible:
    def test_B4_la_source_ne_masque_pas_les_pauses(self):
        """The fix loosens a threshold; it must not silence the slow pauses,
        which remain the signal of a struggling run."""
        texte = SOURCE.read_text(encoding="utf-8")
        assert "sync lent" not in texte, (
            "pauses are logged on the controller side; the launcher must not filter them"
        )
