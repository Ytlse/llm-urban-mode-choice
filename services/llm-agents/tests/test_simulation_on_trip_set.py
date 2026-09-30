"""The simulation served from a recorded trip set.

Without GAMA: exercises the controller methods that decide to serve the set or to recompute,
on an instance built without its constructor (no world model required).
"""

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest
from loguru import logger

from experiences import jeu as J
from experiences.decision import SOURCE_ENREGISTREE
from experiences.experience import ToleranceHoraire
from experiences.population import charger_population, info_population
from models import Location
from tests.test_trip_set import GYM, HOME, WORK, MoteurFactice, _pas_de_locale, _population, _plan, _sceller
from urban_mobility_agents.simulation_controller import SimulationLoopV1, _resume_sources

TOL = {"walk": "insensible", "bike": "insensible", "car": "heure", "transit": {"pas_min": 10}, "rail": {"pas_min": 10}}


@pytest.fixture
def banc(tmp_path, monkeypatch):
    pop = tmp_path / "pop"; pop.mkdir()
    (pop / "population.json").write_text(json.dumps(_population()), encoding="utf-8")
    _sceller(pop)
    personnes, info = charger_population(pop)
    prep = J.JeuEnPreparation.ouvrir(tmp_path / "jeu", "jeu_t", info, "2026-03-16", dependances={"commit": "abc"})
    asyncio.run(J.preparer(prep, personnes, MoteurFactice(n_transit=2), fabrique_locale=_pas_de_locale, progression_s=100))
    prep.clore(J.deplacements_attendus(personnes, "2026-03-16"), len(personnes))
    from settings import settings
    monkeypatch.setattr(settings.data, "jeu_tolerances_horaires", dict(TOL))
    monkeypatch.setattr(settings.app, "log_file", str(tmp_path / "run" / "app.log"))
    (tmp_path / "run").mkdir()
    return {"pop": pop, "info": info, "personnes": personnes, "jeu_dir": tmp_path / "jeu", "tmp": tmp_path}


def _controleur() -> SimulationLoopV1:
    c = SimulationLoopV1.__new__(SimulationLoopV1)
    c.jeu = None; c.jeu_tolerances = {}; c._jeu_stats = Counter(); c._jeu_propositions_par_source = Counter()  # collections.Counter
    c._jeu_alarme_sterile_on = False; c._jeu_stats_depuis_ecriture = 0
    return c


def test_G1_refus_population_differente(banc, tmp_path):
    autre = tmp_path / "autre"; autre.mkdir()
    (autre / "population.json").write_text(json.dumps(_population()[:2]), encoding="utf-8")
    c = _controleur()
    with pytest.raises(RuntimeError, match="préparé pour la population"):
        c.charger_jeu(str(banc["jeu_dir"]), info_population(autre / "population.json"))
    assert c.jeu is None


def test_G5_refus_sans_tolerances(banc, monkeypatch):
    from settings import settings
    monkeypatch.setattr(settings.data, "jeu_tolerances_horaires", {"walk": "insensible"})
    with pytest.raises(RuntimeError, match="hourly tolerances not declared"):
        _controleur().charger_jeu(str(banc["jeu_dir"]), banc["info"])


def test_G3_regime_nominal_servi_du_jeu(banc):
    c = _controleur(); c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    p3 = next(p for p in banc["personnes"] if p.person_id == "p3")
    a1 = next(a for a in p3.identity.activities if a.id == "a1")
    ligne = c.jeu.ligne("p3", "a1")
    du_jeu = c._propositions_du_jeu(p3, a1, ligne.depart_ts + 7 * 60)     # 7 min late: within all tolerances
    assert du_jeu is not None and du_jeu["a_recalculer"] == set() and len(du_jeu["propositions"]) == 5
    assert all(p.source == SOURCE_ENREGISTREE for p in du_jeu["propositions"])
    # Decision 18: another day plays that day's PT supply → transit recomputed, walk/bike/car served…
    mardi = c._propositions_du_jeu(p3, a1, ligne.depart_ts + 86400 + 60)
    assert mardi["a_recalculer"] == {"transit"} and mardi["motifs"]["transit"] == "offre_jour:2026-03-17"
    # …unless the supply equivalence was MEASURED and declared (EQUIVALENCES.yaml, MANIFEST intact).
    empreinte = c.jeu.empreinte
    c.jeu.declarer_jour_equivalent("2026-03-17", {"compares": 100, "identiques": 100})
    assert c._propositions_du_jeu(p3, a1, ligne.depart_ts + 86400 + 60)["a_recalculer"] == set()
    assert J.Jeu.charger(banc["jeu_dir"]).empreinte == empreinte and J.Jeu.charger(banc["jeu_dir"]).jours_equivalents() == ["2026-03-17"]
    assert c._propositions_du_jeu(p3, type("A", (), {"id": "inconnue"})(), ligne.depart_ts) is None   # not in set


def test_G5_tolerances_par_groupe(banc):
    c = _controleur(); c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    p3 = next(p for p in banc["personnes"] if p.person_id == "p3")
    a1 = next(a for a in p3.identity.activities if a.id == "a1")
    ref = c.jeu.ligne("p3", "a1").depart_ts                    # 08:30 → full hour 8
    assert c._propositions_du_jeu(p3, a1, ref + 7 * 60)["a_recalculer"] == set()             # 08:37
    assert c._propositions_du_jeu(p3, a1, ref + 12 * 60)["a_recalculer"] == {"transit"}      # 08:42: PT beyond 10 min
    assert c._propositions_du_jeu(p3, a1, ref + 32 * 60)["a_recalculer"] == {"transit", "car"}   # 09:02: full hour 9 → car too
    # walk and bike are never recomputed
    assert not {"walk", "bike"} & c._propositions_du_jeu(p3, a1, ref + 5 * 3600)["a_recalculer"]


def test_G6_G7_fusion_recalcul_et_sterile(banc):
    c = _controleur(); c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    p3 = next(p for p in banc["personnes"] if p.person_id == "p3")
    a1 = next(a for a in p3.identity.activities if a.id == "a1")
    ref = c.jeu.ligne("p3", "a1").depart_ts
    du_jeu = c._propositions_du_jeu(p3, a1, ref + 12 * 60)
    o, d = Location(**HOME), Location(**WORK)
    # PT recomputation returning the SAME buses → no effect; the recomputed car is not kept (within tolerance)
    recalculees = [_plan("bus", o, d, ref + 12 * 60, 1000, route="L0"), _plan("bus", o, d, ref + 12 * 60, 1060, route="L1"),
                   _plan("car", o, d, ref + 12 * 60, 600)]
    plans, sources = c._fusionner_recalcul(du_jeu, recalculees)
    modes = sorted(pl.mode_label() for pl in plans)
    assert modes == ["bicycle", "bus", "bus", "car", "foot"]
    src = Counter(v.split(":")[0] for v in sources.values())
    assert src == {"enregistree": 3, "recalculee": 2}
    assert any(v.startswith("recalculee:horaire:+12/pas10min") for v in sources.values())
    assert c._jeu_stats["recalculs_horaire"] == 1 and c._jeu_stats["recalcul_sans_effet"] == 1
    # a recomputation that changes durations is not sterile
    plans2, _ = c._fusionner_recalcul(du_jeu, [_plan("bus", o, d, ref, 1500, route="L0")])
    assert c._jeu_stats["recalculs_horaire"] == 2 and c._jeu_stats["recalcul_sans_effet"] == 1
    # ALARME on rising edge beyond the threshold (≥ 20 recomputations)
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        for _ in range(25):
            c._fusionner_recalcul(du_jeu, recalculees)
    finally:
        logger.remove(sink)
    assert sum("[ALARME]" in m and "too sensitive" in m for m in messages) == 1


def test_G6_source_offre_jour(banc):
    c = _controleur(); c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    p3 = next(p for p in banc["personnes"] if p.person_id == "p3")
    a1 = next(a for a in p3.identity.activities if a.id == "a1")
    ref = c.jeu.ligne("p3", "a1").depart_ts
    du_jeu = c._propositions_du_jeu(p3, a1, ref + 86400)
    o, d = Location(**HOME), Location(**WORK)
    plans, sources = c._fusionner_recalcul(du_jeu, [_plan("bus", o, d, ref + 86400, 1200, route="L9")])
    assert sorted(pl.mode_label() for pl in plans) == ["bicycle", "bus", "car", "foot"]
    assert [v for v in sources.values() if v.startswith("recalculee:offre_jour:2026-03-17")]
    assert c._jeu_stats["recalculs_offre_jour"] == 1 and c._jeu_stats["recalculs_horaire"] == 0


def test_G6_colonne_source_moves(banc):
    from models import TravelPlan
    o, d = Location(**HOME), Location(**WORK)
    a, b, cpl = _plan("foot", o, d, 0), _plan("bus", o, d, 0, route="L0"), _plan("car", o, d, 0)
    sources = {id(a): "enregistree", id(b): "recalculee:horaire:+12/pas10min", id(cpl): "enregistree"}
    assert _resume_sources(sources, [a, b, cpl]) == "enregistree:2,recalculee:1"
    assert _resume_sources({}, [a]) == "en_vol:1"


def test_G13_le_jeu_n_est_jamais_modifie(banc):
    avant = J.Jeu.charger(banc["jeu_dir"]).empreinte
    c = _controleur(); c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    p3 = next(p for p in banc["personnes"] if p.person_id == "p3")
    a1 = next(a for a in p3.identity.activities if a.id == "a1")
    du_jeu = c._propositions_du_jeu(p3, a1, c.jeu.ligne("p3", "a1").depart_ts + 12 * 60)
    c._fusionner_recalcul(du_jeu, [_plan("bus", Location(**HOME), Location(**WORK), 0, 1500, route="L0")])
    c._ecrire_jeu_stats(force=True)
    assert J.Jeu.charger(banc["jeu_dir"]).empreinte == avant


def test_G14_jeu_stats_et_rapport(banc, tmp_path):
    c = _controleur(); c.charger_jeu(str(banc["jeu_dir"]), banc["info"])
    c._jeu_stats.update({"deplacements_servis": 40, "appels_moteur": 2, "recalculs_horaire": 2, "recalcul_sans_effet": 1, "hors_jeu": 1})
    c._jeu_propositions_par_source.update({"enregistree": 190, "recalculee": 4, "hors_jeu": 6})
    c._ecrire_jeu_stats(force=True)
    stats = json.loads((tmp_path / "run" / "jeu_stats.json").read_text())
    assert stats["jeu"] == "jeu_t" and stats["deplacements_servis"] == 40 and stats["propositions_par_source"]["enregistree"] == 190
    assert stats["declencheurs_jamais_declenches"] == ["offre (événements non implémentés)", "offre_jour (aucun autre jour simulé que celui du jeu)"]
    # 5 expected on the bench: the day closes (p3: a0→a1, a1→a2, a2→a0; p2: b0→b1, b1→b0),
    # and p3's HOME → HOME loop counts in the raw expected even though it stays unusable.
    assert stats["couverture_jeu"]["deplacements_attendus"] == 5
    # `make report`: the five sections
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("run_report", Path(__file__).resolve().parents[3] / "scripts" / "debug" / "run_report.py")
    rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
    out, alarms = [], []
    rr.section_jeu(tmp_path / "run", out, alarms)
    texte = "\n".join(out)
    for rubrique in ("Appels au trip helper", "Propositions `enregistree`", "Recalculs sans effet", "Déclencheurs jamais déclenchés", "Couverture du jeu"):
        assert rubrique in texte, rubrique
    assert any("fonction fantôme" in a for a in alarms)
