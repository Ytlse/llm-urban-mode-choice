"""The lived experience does not conclude, and a shock has a cadence.

The cases carry the rule numbers.

Why this file exists. Run `experiments/archive/2026-09-19_07_31` measured a modal
shift whose conclusion was written in the stimulus: the `vecu` said « I no longer trust
this car at all », the reflection drew « consider alternative transport options » from it, and that
sentence was served to the forty decisions of the next fourteen days. Rule R7/R8 of declared shocks
already refused addressing the agent; it let through concluding in its place.

Everything here is PURE: no simulator, no model, no network call.
"""

import sys
from pathlib import Path

import pytest
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import chocs as chocs_module
from llm.chocs import RefusDeChoc, RegistreChocs, charger

CONFIG_CHOCS = Path(__file__).resolve().parents[1] / "config" / "chocs"

VECU_VALIDE = "I was stuck for an hour on the ring road, and I arrived in a foul mood."


def _declaration(**surcharges) -> dict:
    base = {
        "choc": "test_choc",
        "libelle": "Choc de test",
        "source": "test",
        "exposition": {"regle": "mode", "modes": ["car"]},
        "jours": [{"jour": 12, "retard_min": 60, "vecu": VECU_VALIDE}],
    }
    base.update(surcharges)
    return base


def _ecrire(tmp_path: Path, declaration: dict) -> Path:
    p = tmp_path / "choc.yaml"
    p.write_text(yaml.safe_dump(declaration, allow_unicode=True), encoding="utf-8")
    return p


def _charger_vecu(tmp_path: Path, vecu: str):
    return charger(_ecrire(tmp_path, _declaration(jours=[{"jour": 12, "retard_min": 60, "vecu": vecu}])))


@pytest.fixture(autouse=True)
def _registre_propre():
    chocs_module.reinitialiser()
    yield
    chocs_module.reinitialiser()


@pytest.fixture
def journal():
    """Collects the module's loguru messages, level and text."""
    lignes: list[tuple[str, str]] = []
    sink = logger.add(lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO")
    yield lignes
    logger.remove(sink)


# ══════════════════════ H1 — no verdict, no intention ═══════════════════════════


@pytest.mark.parametrize(
    "vecu",
    [
        "The engine stalled. I no longer trust this car at all.",
        "The bus never came. This bus line is unreliable.",
        "Encore en panne. Je ne fais plus confiance à cette voiture.",
    ],
)
def test_H1_1_un_verdict_sur_un_mode_est_refuse(tmp_path, vecu):
    """H1.1 — the belief is what the reflection must PRODUCE, not what it is told."""
    with pytest.raises(RefusDeChoc) as exc:
        _charger_vecu(tmp_path, vecu)
    assert "verdict" in str(exc.value).lower()


@pytest.mark.parametrize(
    "vecu",
    [
        "Arrived 20 minutes late. I am seriously thinking about not using this car anymore.",
        "Stuck again. From now on I will take the metro.",
        "Encore une heure de bouchon. Je ne prendrai plus la voiture.",
    ],
)
def test_H1_2_une_intention_modale_est_refusee(tmp_path, vecu):
    """H1.2 — message distinct from H1.1: the two families are relaxed separately."""
    with pytest.raises(RefusDeChoc) as exc:
        _charger_vecu(tmp_path, vecu)
    assert "intention" in str(exc.value).lower()


@pytest.mark.parametrize("fichier", ["c1_bouchon_rocade", "c2_crevaison", "c3_panne_reseau",
                                     "c4_train_supprime", "c5_orage_grele"])
def test_H1_3_les_chocs_existants_se_chargent(fichier):
    """H1.3 — the rule makes the practice of the other five files checkable, it does not condemn it."""
    charger(CONFIG_CHOCS / f"{fichier}.yaml")


def test_H1_4_un_doute_est_accepte(tmp_path):
    """H1.4 — the lower bound, deliberately kept (c1, day 14)."""
    _charger_vecu(tmp_path, "Still crawling along. I am starting to wonder whether this is worth it.")


def test_H1_5_un_fait_passe_meme_modal_est_accepte(tmp_path):
    """H1.5 — « I did otherwise yesterday » is not « I will do otherwise tomorrow » (c2, day 13)."""
    _charger_vecu(
        tmp_path,
        "My bike is still out of action after yesterday. I had to sort out another way of getting around.",
    )


def test_H1_6_la_redaction_du_18_septembre_est_refusee(tmp_path):
    """H1.6 — the test that should have failed before the forty-two-day run."""
    j15 = (
        "The engine made a grinding noise and the car stalled on the expressway. I had to pull "
        "over and wait 30 minutes for roadside assistance before the car would restart. "
        "I arrived very late and stressed. I no longer trust this car at all."
    )
    j16 = (
        "Warning lights flashing on the dashboard again. The grinding noise is back and louder "
        "than yesterday. I drove very slowly and considered pulling over again. Arrived 20 "
        "minutes late. I am seriously thinking about not using this car anymore."
    )
    with pytest.raises(RefusDeChoc, match="(?i)verdict"):
        _charger_vecu(tmp_path, j15)
    with pytest.raises(RefusDeChoc, match="(?i)intention"):
        _charger_vecu(tmp_path, j16)


def test_H1_7_c6_corrige_se_charge_et_ne_conclut_plus():
    """H1.7 — and the check also holds on the text, not only on loading."""
    choc = charger(CONFIG_CHOCS / "c6_voiture_suspecte.yaml")
    for jour in choc.jours.values():
        bas = jour.vecu.lower()
        assert "trust" not in bas
        assert "anymore" not in bas
        for mode in ("bus", "metro", "bike", "bicycle", "walk", "public transport"):
            assert mode not in bas, f"day {jour.jour}: the lived experience names a shift mode ({mode})"


# ══════════════════════ H2 — the cadence ═════════════════════════════════════════


def _registre(tmp_path, monkeypatch, jour=12, **surcharges):
    choc = charger(_ecrire(tmp_path, _declaration(**surcharges)))
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: jour))
    return RegistreChocs(choc)


def test_H2_1_cadence_jour_ne_touche_qu_une_fois(tmp_path, monkeypatch):
    """H2.1 — a repaired breakdown does not recur identically three hours later."""
    r = _registre(tmp_path, monkeypatch, cadence="jour")
    assert r.applique("609", "car", 1000) is not None
    assert r.applique("609", "car", 2000) is None
    assert r.applique("610", "car", 3000) is not None, "the cadence is per AGENT, not per global day"


def test_H2_2_cadence_trajet_touche_chaque_arrivee(tmp_path, monkeypatch):
    """H2.2 — historical behaviour, that of a traffic jam lasting all day."""
    r = _registre(tmp_path, monkeypatch, cadence="trajet")
    assert r.applique("609", "car", 1000) is not None
    assert r.applique("609", "car", 2000) is not None


def test_H2_3_cadence_absente_vaut_trajet_et_se_journalise(tmp_path, monkeypatch, journal):
    """H2.3 — no existing file changes behaviour silently."""
    r = _registre(tmp_path, monkeypatch)
    assert r.applique("609", "car", 1000) is not None
    assert r.applique("609", "car", 2000) is not None
    assert any("cadence" in m for _, m in journal), "the default cadence is not logged"


def test_H2_4_cadence_inconnue_est_refusee(tmp_path):
    """H2.4 — like an unknown exposure rule: we refuse, we do not ignore."""
    with pytest.raises(RefusDeChoc, match="(?i)cadence"):
        charger(_ecrire(tmp_path, _declaration(cadence="parfois")))


def test_H2_5_le_compteur_se_rearme_chaque_journee(tmp_path, monkeypatch):
    """H2.5 — hit once per day, not once for the whole run."""
    choc = charger(
        _ecrire(
            tmp_path,
            _declaration(
                cadence="jour",
                jours=[
                    {"jour": 12, "retard_min": 60, "vecu": VECU_VALIDE},
                    {"jour": 13, "retard_min": 30, "vecu": VECU_VALIDE},
                ],
            ),
        )
    )
    r = RegistreChocs(choc)
    jours = {"n": 12}
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: jours["n"]))
    assert r.applique("609", "car", 1000) is not None
    assert r.applique("609", "car", 2000) is None
    jours["n"] = 13
    assert r.applique("609", "car", 3000) is not None


def test_H2_6_c6_declare_la_cadence_jour():
    """H2.6 — the run file says itself what it does."""
    assert charger(CONFIG_CHOCS / "c6_voiture_suspecte.yaml").cadence == "jour"


# ══════════════════════ H3 — a shock day with no exposed agent ════════════════════════


def test_H3_1_jour_de_choc_sans_expose_leve_une_alarme(tmp_path, monkeypatch, journal):
    """H3.1 — the second shock of c6 never happened, and the report kept talking about it."""
    r = _registre(tmp_path, monkeypatch)
    r.applique("609", "walking", 1000)  # spared: wrong mode
    r.journaliser_compteurs()
    alarmes = [m for n, m in journal if n == "ERROR" and "[ALARME]" in m]
    assert alarmes, "a shock day closed without a single exposed agent raised no alarm"
    assert "12" in alarmes[0]


def test_H3_2_journee_nominale_sans_expose_reste_en_info(tmp_path, monkeypatch, journal):
    """H3.2 — the normal case: most days of a run are not shock days."""
    r = _registre(tmp_path, monkeypatch, jour=10)
    r.applique("609", "car", 1000)
    r.journaliser_compteurs()
    assert not [m for n, m in journal if n == "ERROR"]


def test_H3_3_jour_de_choc_avec_expose_ne_leve_rien(tmp_path, monkeypatch, journal):
    """H3.3 — the alarm fires only on absence, never on presence."""
    r = _registre(tmp_path, monkeypatch)
    assert r.applique("609", "car", 1000) is not None
    r.journaliser_compteurs()
    assert not [m for n, m in journal if n == "ERROR"]
