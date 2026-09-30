"""A JEV never replays a memory A/B, even when launched in an A/B's container (2026-09-28).

On 26/09, JEVs launched in the controller container inherited `REJEU_AB`: they
stored their responses in a13's store and read them back from there.
"""

from __future__ import annotations

import pytest

from experiences import cli
from settings import settings


def test_la_cli_coupe_le_rejeu_herite_avant_toute_commande(monkeypatch):
    monkeypatch.setattr(settings.llm, "rejeu_ab", "exp_mem_presse_a13_v5")
    monkeypatch.setattr(settings.llm, "rejeu_strict_avant_ts", 1774597500)
    with pytest.raises(SystemExit):
        cli.main(["--commande-qui-n-existe-pas"])
    assert settings.llm.rejeu_ab == ""
    assert settings.llm.rejeu_strict_avant_ts is None


def test_le_make_des_jev_ne_transmet_pas_le_rejeu():
    from pathlib import Path

    mk = (Path(__file__).resolve().parents[3] / "make" / "experiences.mk").read_text(encoding="utf-8")
    ligne = next(l for l in mk.splitlines() if l.startswith("EXPERIENCES_PY ="))
    assert "-e REJEU_AB= " in ligne and "-e REJEU_STRICT_AVANT_TS=0" in ligne
