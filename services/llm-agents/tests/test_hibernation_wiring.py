"""The WIRING of hibernation and of the daily measures.

Why this file exists. The run `experiments/archive/2026-09-19_07_31` produced none
of the per-day measurement CSVs, although `mesures_jour_enabled` was `true`. Cause: the method
`_declencher_hibernation_propre` had been inserted in the MIDDLE of `_ecrire_point_de_reprise`,
and the call to the measurements ended up after a `sys.exit(0)` — unreachable. The per-day
measurement tests stayed green throughout: they checked the module, not its wiring.

These tests therefore cover the wiring and the shape of the code, never the computation of the
measurements, which is covered elsewhere (`test_daily_measures.py`).
"""

import ast
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urban_mobility_agents import simulation_controller as sc
from urban_mobility_agents.utils import mesures_jour

SOURCE = Path(sc.__file__)
T0 = 1773637200  # 16 March 2026, 05:00 in simulation wall-clock time


# ══════════════════════════ test doubles ════════════════════════════════════════


class _PopulationDouble:
    def get_people_list(self):
        return []


def _boucle(timestamp: int = T0) -> SimpleNamespace:
    """The loop reduced to what the two methods under test actually read."""
    return SimpleNamespace(
        agent=None,
        population=_PopulationDouble(),
        _current_sim_timestamp=timestamp,
        # The stop marker now carries the counter of consecutive fallbacks.
        # The dummy must mirror the class: a defensive `getattr` on the production side
        # would mask a real AttributeError the day the counter disappeared.
        _replis_consecutifs=0,
        # Hibernation rising edge (2026-09-24).
        _hibernation_declenchee=False,
        _hibernations_ignorees=0,
    )


def _poser_workdir(monkeypatch, workdir: Path) -> None:
    """`settings` is a `__getattribute__` proxy: we replace the name in the target module.

    Same gesture as `test_accidents_switch.py` — a double placed on the proxy would
    never take, and the test would pass reading the process's real workdir.
    """
    monkeypatch.setattr(sc, "settings", SimpleNamespace(workdir=workdir))


def _fonction(nom: str) -> ast.AST:
    arbre = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for noeud in ast.walk(arbre):
        if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef)) and noeud.name == nom:
            return noeud
    raise AssertionError(f"function not found: {nom}")


def _est_appel_terminal(instruction: ast.stmt) -> bool:
    """`sys.exit(…)`, `os._exit(…)`, or a signal sent to its own process."""
    if not (isinstance(instruction, ast.Expr) and isinstance(instruction.value, ast.Call)):
        return False
    appel = ast.unparse(instruction.value.func)
    return appel in {"sys.exit", "exit", "os._exit", "os.kill"}


# ══════════════════════════ G1 — the daily measurements are wired ═══════════════


def _armer(monkeypatch, tmp_path, actif=True, point_leve=False):
    vus: list[Path] = []

    def _ecriture(workdir):
        vus.append(Path(workdir))
        return 1

    def _point(*_a, **_k):
        if point_leve:
            raise OSError("disque plein")

    monkeypatch.setattr(mesures_jour, "actif", lambda: actif)
    monkeypatch.setattr(mesures_jour, "ecrire_mesures_du_jour", _ecriture)
    monkeypatch.setattr(sc, "ecrire_point", _point)
    _poser_workdir(monkeypatch, tmp_path)
    return vus


def test_G1_le_point_de_reprise_ecrit_les_mesures_du_jour(monkeypatch, tmp_path):
    """G1 — the rule that fails if the code is put back into its 19 September state."""
    vus = _armer(monkeypatch, tmp_path)
    asyncio.run(sc.SimulationLoopV1._ecrire_point_de_reprise(_boucle(), T0))
    assert vus == [tmp_path], "the daily measurements are no longer called at the checkpoint"


def test_G1b_reglage_eteint_aucune_mesure(monkeypatch, tmp_path):
    """G1b — switched off, the module reads nothing and writes nothing."""
    vus = _armer(monkeypatch, tmp_path, actif=False)
    asyncio.run(sc.SimulationLoopV1._ecrire_point_de_reprise(_boucle(), T0))
    assert vus == []


def test_G1c_un_point_qui_leve_n_emporte_pas_les_mesures(monkeypatch, tmp_path):
    """G1c — the checkpoint's fail-open must not take the measurements' one down with it."""
    vus = _armer(monkeypatch, tmp_path, point_leve=True)
    asyncio.run(sc.SimulationLoopV1._ecrire_point_de_reprise(_boucle(), T0))
    assert vus == [tmp_path], "a failing checkpoint knocked out the daily measurements"


# ══════════════════════════ G2 — generic guard against dead code ════════════════


def test_G2_aucune_instruction_apres_un_appel_terminal():
    """G2 — the mistake is caught anywhere in the file, not only where it was born."""
    arbre = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fautes = []
    for noeud in ast.walk(arbre):
        for champ in ("body", "orelse", "finalbody"):
            bloc = getattr(noeud, champ, None)
            if not isinstance(bloc, list):
                continue
            for rang, instruction in enumerate(bloc[:-1]):
                if _est_appel_terminal(instruction):
                    fautes.append(
                        f"ligne {instruction.lineno} : {len(bloc) - rang - 1} instruction(s) "
                        f"morte(s) après {ast.unparse(instruction).strip()}"
                    )
    assert not fautes, "code mort après un appel terminal :\n" + "\n".join(fautes)


# ══════════════════════════ G3 to G5 — hibernation ══════════════════════════════


def _armer_hibernation(monkeypatch, workdir):
    signaux: list[tuple] = []
    _poser_workdir(monkeypatch, workdir)
    monkeypatch.setattr(sc.os, "kill", lambda pid, sig: signaux.append((pid, sig)))

    async def _point(_ts):
        return None

    boucle = _boucle()
    boucle._ecrire_point_de_reprise = _point
    return boucle, signaux


def test_G3_le_marqueur_est_ecrit_dans_le_workdir_du_run(monkeypatch, tmp_path):
    """G3 — `settings.workdir`, and not a `settings.data.workdir` that does not exist."""
    boucle, _ = _armer_hibernation(monkeypatch, tmp_path)
    asyncio.run(sc.SimulationLoopV1._declencher_hibernation_propre(boucle, "2026-09-19T12:00", "899549"))

    marqueur = tmp_path / "en_attente_quota.json"
    assert marqueur.is_file(), "no waiting marker written"
    contenu = json.loads(marqueur.read_text(encoding="utf-8"))
    assert contenu["person_id"] == "899549"
    assert contenu["resume_at"] == "2026-09-19T12:00"
    assert contenu["timestamp"] == T0
    assert contenu["jour_simule"] >= 1


def test_G4_l_arret_est_demande_une_seule_fois_par_signal(monkeypatch, tmp_path):
    """G4 — SIGTERM to the process: the sequential orchestrator expects a return code 0."""
    import os
    import signal

    boucle, signaux = _armer_hibernation(monkeypatch, tmp_path)
    asyncio.run(sc.SimulationLoopV1._declencher_hibernation_propre(boucle, "2026-09-19T12:00", "899549"))

    assert len(signaux) == 1, f"stop requested {len(signaux)} times"
    pid, sig = signaux[0]
    assert pid == os.getpid()
    assert sig == signal.SIGTERM


def test_G5_un_marqueur_impossible_n_annule_pas_l_arret(monkeypatch, tmp_path, caplog):
    """G5 — otherwise hibernation becomes a run that goes on with default fallbacks."""
    boucle, signaux = _armer_hibernation(monkeypatch, tmp_path / "chemin" / "inexistant")
    asyncio.run(sc.SimulationLoopV1._declencher_hibernation_propre(boucle, "2026-09-19T12:00", "899549"))
    assert len(signaux) == 1, "the stop was not requested although the marker failed"


def test_G5b_des_declenchements_concurrents_n_arretent_qu_une_fois(monkeypatch, tmp_path):
    """2026-09-24: eight consumers in fallback each triggered the stop, and their concurrent
    checkpoints trod on each other (Errno 39). Only the first one acts."""
    boucle, signaux = _armer_hibernation(monkeypatch, tmp_path)
    points: list[int] = []

    async def _point_lent(ts):
        points.append(ts)
        await asyncio.sleep(0.01)   # gives the other callers time to get in

    boucle._ecrire_point_de_reprise = _point_lent

    async def _huit():
        await asyncio.gather(*(
            sc.SimulationLoopV1._declencher_hibernation_propre(
                boucle, "2026-09-25T00:00", str(i)
            )
            for i in range(8)
        ))

    asyncio.run(_huit())
    assert len(signaux) == 1
    assert len(points) == 1
    assert boucle._hibernations_ignorees == 7
    assert json.loads((tmp_path / "en_attente_quota.json").read_text())["person_id"] == "0"


# ══════════════════════════ G6 — the shape of the return ════════════════════════


def test_G6_compute_move_rend_toujours_un_couple():
    """G6 — the four callers do `move, _ = await …`: a bare `None` breaks the unpacking."""
    fonction = _fonction("_compute_move_for_activity")
    imbriquees = {
        n for f in ast.walk(fonction)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f is not fonction
        for n in ast.walk(f)
    }
    fautes = []
    for noeud in ast.walk(fonction):
        if isinstance(noeud, ast.Return) and noeud not in imbriquees:
            valeur = noeud.value
            if not (isinstance(valeur, ast.Tuple) and len(valeur.elts) == 2):
                fautes.append(f"ligne {noeud.lineno} : {ast.unparse(noeud).strip()}")
    assert not fautes, "retour non conforme à la signature (couple attendu) :\n" + "\n".join(fautes)
