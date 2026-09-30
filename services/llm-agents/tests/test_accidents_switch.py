"""The accidents switch and the accident draw.

One test per rule of the switch specification, the rule number in the
name. Rules R1 and R3 concern the GAMA GUI and its persistence: they are checked on the
text of the `.gaml` models, which no Python test can run — it is a presence check,
and it is declared as such rather than omitted.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import pytest
from settings import AccidentsConfig
from trip_helper import accidents as accidents_module
from trip_helper.accidents import (
    Accident,
    LoiBaac,
    RegistreAccidents,
    classe_de_vitesse,
    jour_simule,
)

MODELES = (
    pathlib.Path(__file__).resolve().parents[3]
    / "services"
    / "GAMA"
    / "CityTransport"
    / "models"
)
JOUR = 86400
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
T0 = 1_773_637_200  # a simulated Monday 05:00, like the reference start


class GrapheFactice:
    """A minimal graph: edges, a length, a speed limit."""

    def __init__(self, aretes):
        self._aretes = aretes

    def edges(self, data=False):
        if data:
            return [
                (u, v, {"length": lg, "maxspeed": vit})
                for u, v, lg, vit in self._aretes
            ]
        return [(u, v) for u, v, _, _ in self._aretes]


# Three edges of the 31-50 class, of very unequal lengths, plus a fast edge: enough to
# check that the class is drawn from the law and the edge pro rata to its length WITHIN the class.
ARETES = [(1, 2, 1000.0, 50), (2, 3, 500.0, 50), (3, 4, 2500.0, 50), (4, 5, 800.0, 80)]


def _registre(
    taux=None, graine=70, loi=None, aretes=ARETES, **kwargs
) -> RegistreAccidents:
    config = AccidentsConfig(
        enabled=True, taux_journalier=taux, graine=graine, **kwargs
    )
    registre = RegistreAccidents(config=config, loi=loi)
    registre.charger_aretes(GrapheFactice(aretes))
    return registre


@pytest.fixture(autouse=True)
def _registre_propre():
    """No test may inherit another test's registry."""
    accidents_module.reinitialiser()
    yield
    accidents_module.reinitialiser()


# ── R1 / R3 — the switch in the GAMA GUI and its persistence ─────────────────


def test_r1_interrupteur_expose_dans_l_ihm_gama():
    """R1 — an "Accidents sur les axes" parameter, Simulation category."""
    city = (MODELES / "City.gaml").read_text(encoding="utf-8")
    assert 'parameter "Accidents sur les axes"' in city
    assert 'category: "Simulation" var: accidents_enabled' in city


def test_r3_interrupteur_persiste_et_recharge():
    """R3 — written to sim_params.yaml and read back at startup."""
    settings_gaml = (MODELES / "Settings.gaml").read_text(encoding="utf-8")
    assert '"accidents_enabled: " + string(accidents_enabled)' in settings_gaml, (
        "not written"
    )
    assert "accidents_enabled <- (_cfg_acc != nil)" in settings_gaml, "not read back"


def test_r2_vrai_par_defaut_partout():
    """R2 — true by default on the GAMA side as on the controller side (decision of 2026-09-15).

    The realistic regime is the ordinary one: it is its DEACTIVATION that takes a gesture. An
    older `sim_params.yaml`, without the key, must therefore enable accidents, not silence them.
    """
    settings_gaml = (MODELES / "Settings.gaml").read_text(encoding="utf-8")
    assert "bool accidents_enabled <- true;" in settings_gaml
    assert 'contains "true") : true;' in settings_gaml, "the fallback on missing config must be true"
    city = (MODELES / "City.gaml").read_text(encoding="utf-8")
    assert "var: accidents_enabled <- true;" in city
    assert AccidentsConfig().enabled is True


def test_r4_interrupteur_transmis_au_init():
    """R4 — the /init payload carries `accidents_enabled`."""
    llm_agent = (MODELES / "LLMAgent.gaml").read_text(encoding="utf-8")
    assert '"accidents_enabled"::accidents_enabled' in llm_agent

    from gama_models import WorldInitRequest

    requete = WorldInitRequest(timestamp=T0, accidents_enabled=True)
    assert requete.accidents_enabled is True
    # Absent from the request ≠ disabled: the controller must be able to tell them apart.
    assert WorldInitRequest(timestamp=T0).accidents_enabled is None


def test_r5_etat_effectif_ecrit_meme_a_faux(tmp_path, monkeypatch):
    """R5 — `scenario_params.yaml` carries the state, including when it is false."""
    import types

    import yaml
    from urban_mobility_agents.factory import factory

    # `settings` is a `FactorySettings` proxy: assigning `workdir` to it changes nothing —
    # the assignment does not reach the underlying object (verified). So we replace the name
    # the module reads, which `getattr(settings, "workdir")` resolves at call time.
    monkeypatch.setattr(factory, "settings", types.SimpleNamespace(workdir=tmp_path))
    factory._save_scenario_params(
        population_size=10,
        llm_agents=10,
        long_term_memory_enabled=True,
        long_term_self_reflect_enabled=True,
        accidents_enabled=False,
    )
    ecrit = yaml.safe_load(
        (tmp_path / "scenario_params.yaml").read_text(encoding="utf-8")
    )
    assert "accidents_enabled" in ecrit, "a run silent about the regime is unreadable"
    assert ecrit["accidents_enabled"] is False


# ── R6 / R7 — what the switch controls ───────────────────────────────────────


def test_r6_interrupteur_a_faux_ne_tire_rien(monkeypatch):
    """R6 — regime explicitly disabled: no registry, hence no accident.

    Since the default became true, this test must disable EXPLICITLY: it is the
    experimenter's gesture of unticking, and that is what is checked.
    """
    from settings import settings

    monkeypatch.setattr(settings.accidents, "enabled", False)
    assert accidents_module.initialiser() is None
    assert accidents_module.registre() is None


def test_r7_interrupteur_a_vrai_tire_des_accidents_poses_sur_une_arete():
    """R7 — each accident carries an edge, a start and a duration."""
    registre = _registre(taux=5.0)
    poses = registre.tirer_journee(jour=1, debut_jour_ts=T0)

    assert poses, "a rate of 5/day must produce accidents"
    for accident in poses:
        assert accident.arete in {(1, 2), (2, 3), (3, 4), (4, 5)}
        assert accident.classe_vitesse in {"31-50", "71-90"}
        assert T0 <= accident.debut_ts < T0 + JOUR
        assert accident.duree_s > 0
        assert accident.fin_ts == accident.debut_ts + accident.duree_s


def test_r7_tirage_idempotent_par_journee():
    """R7 — called at every sync, the draw produces the day only once."""
    registre = _registre(taux=5.0)
    premier = registre.tirer_journee(jour=1, debut_jour_ts=T0)
    second = registre.tirer_journee(jour=1, debut_jour_ts=T0)
    assert second == []
    assert len(registre.accidents) == len(premier)


def test_r7_tirage_deterministe_a_graine_fixee():
    """R7 — two runs of the same scenario place the same accidents."""
    a = _registre(taux=5.0, graine=70).tirer_journee(1, T0)
    b = _registre(taux=5.0, graine=70).tirer_journee(1, T0)
    assert a == b
    autre = _registre(taux=5.0, graine=71).tirer_journee(1, T0)
    assert autre != a, "a different seed must produce another world"


def test_r7_arete_tiree_proportionnellement_a_sa_longueur_dans_sa_classe():
    """R7 — within the same class, the 2,500 m edge is hit more often than the 500 m one."""
    registre = _registre(taux=40.0, graine=1)
    for jour in range(1, 40):
        registre.tirer_journee(jour, T0 + (jour - 1) * JOUR)
    comptes = {}
    for accident in registre.accidents:
        comptes[accident.arete] = comptes.get(accident.arete, 0) + 1
    assert comptes.get((3, 4), 0) > comptes.get((2, 3), 0), (
        "le tirage ignore la longueur à l'intérieur d'une classe : la géographie des "
        f"accidents serait celle du découpage OSM, pas du réseau — {comptes}"
    )


# ── R11 / R12 / R13 — the derived rules ──────────────────────────────────────


def test_r11_compteurs_publies_meme_a_zero(caplog):
    """R11 — a day without accidents says so, otherwise it looks like a failure."""
    registre = _registre(taux=0.0001, graine=3)
    registre.tirer_journee(jour=1, debut_jour_ts=T0)
    compteurs = registre.compteurs
    assert len(compteurs) == 1, "the day must be counted even when empty"
    assert compteurs[0].jour == 1
    assert compteurs[0].tires == 0


def test_r12_accident_de_duree_nulle_est_refuse():
    """R12 — refused when placed, and nothing enters the world state."""
    registre = _registre()
    assert registre._poser(Accident(arete=(1, 2), debut_ts=T0, duree_s=0)) is False
    assert registre.accidents == []


def test_r12_accident_sur_arete_inconnue_est_refuse():
    """R12 — an edge absent from the graph cannot carry an accident."""
    registre = _registre()
    assert registre._poser(Accident(arete=(99, 98), debut_ts=T0, duree_s=600)) is False
    assert registre.accidents == []


def test_r12_taux_hors_bornes_ne_tire_rien():
    """R12 — an aberrant rate is refused rather than endured."""
    assert _registre(taux=0.0).tirer_journee(1, T0) == []
    assert _registre(taux=10_000.0).tirer_journee(1, T0) == []


def test_r13_journee_simulee_ancree_sur_le_debut_du_run():
    """R13 — the split into days depends only on the start of the run."""
    assert jour_simule(T0, T0) == (1, T0)
    assert jour_simule(T0 + JOUR - 1, T0) == (1, T0)
    assert jour_simule(T0 + JOUR, T0) == (2, T0 + JOUR)
    assert jour_simule(T0 + 3 * JOUR + 7200, T0) == (4, T0 + 3 * JOUR)


# ── Checked non-goal: no duration is modified by this slice ──────────────────


# ── R8 / R9 / R10 — the delay incurred and its two guards ────────────────────


def _gdf(aretes, travel_times):
    """A minimal `route_to_gdf` output: a (u, v, k) index and a travel_time column."""
    import pandas as pd

    index = pd.MultiIndex.from_tuples([(u, v, 0) for u, v in aretes])
    return pd.DataFrame({"travel_time": travel_times}, index=index)


class _GrapheZones:
    """A graph reduced to what `_congested_travel_time` asks of it: the zone of its nodes."""

    def __init__(self, noeuds):
        from trip_helper.congestion_zones import NODE_ZONE_KEY, ZONE_OUTSIDE

        self.nodes = {n: {NODE_ZONE_KEY: ZONE_OUTSIDE} for n in noeuds}


def test_r8_itineraire_traversant_un_accident_est_allonge(monkeypatch):
    """R8 — the returned duration is strictly greater, and the faulty edge is counted."""
    from trip_helper import accidents as mod
    from trip_helper.osmnx_direct import _congested_travel_time

    registre = _registre(taux=5.0)
    registre.tirer_journee(1, T0)
    assert registre._poser(
        Accident(arete=(2, 3), debut_ts=T0, duree_s=3600, classe_vitesse="31-50")
    )
    monkeypatch.setattr(mod, "_registre", registre)

    cong_s, free_s, n_acc = _congested_travel_time(
        _GrapheZones([1, 2, 3, 4]),
        _gdf([(1, 2), (2, 3), (3, 4)], [100.0, 200.0, 100.0]),
        datetime.fromtimestamp(T0 + 600, tz=timezone.utc),
    )

    assert free_s == 400.0
    assert n_acc == 1, "the accident edge was not recognised"
    attendu = 100.0 + 200.0 * AccidentsConfig().facteur_ralentissement + 100.0
    assert cong_s == pytest.approx(attendu)
    assert cong_s > free_s


def test_r8_itineraire_hors_accident_est_inchange(monkeypatch):
    """R8 — an itinerary that crosses nothing keeps its free-flow duration."""
    from trip_helper import accidents as mod
    from trip_helper.osmnx_direct import _congested_travel_time

    registre = _registre(taux=5.0)
    registre.tirer_journee(1, T0)
    registre._poser(Accident(arete=(1, 2), debut_ts=T0, duree_s=3600))
    monkeypatch.setattr(mod, "_registre", registre)

    cong_s, free_s, n_acc = _congested_travel_time(
        _GrapheZones([2, 3, 4]),
        _gdf([(2, 3), (3, 4)], [100.0, 100.0]),
        datetime.fromtimestamp(T0 + 600, tz=timezone.utc),
    )
    assert n_acc == 0
    assert cong_s == pytest.approx(free_s)


def test_r8_accident_expire_ne_ralentit_plus(monkeypatch):
    """R8 — outside its window, the accident has no effect any more."""
    from trip_helper import accidents as mod
    from trip_helper.osmnx_direct import _congested_travel_time

    registre = _registre(taux=5.0)
    registre.tirer_journee(1, T0)
    registre._poser(Accident(arete=(1, 2), debut_ts=T0, duree_s=600))
    monkeypatch.setattr(mod, "_registre", registre)

    _cong, _free, n_acc = _congested_travel_time(
        _GrapheZones([1, 2]),
        _gdf([(1, 2)], [100.0]),
        datetime.fromtimestamp(T0 + 1200, tz=timezone.utc),
    )
    assert n_acc == 0


def test_r9_le_cache_d_itineraires_est_contourne_pendant_un_accident():
    """R9 — neither read nor write while an accident is active.

    The predicate is exercised here; the fact that the guard is WIRED to it is checked on the
    module text — the full path is asynchronous and depends on the network, it cannot
    be exercised in a unit test. Presence check, declared as such.
    """
    registre = _registre(taux=5.0)
    registre.tirer_journee(1, T0)
    registre._poser(Accident(arete=(1, 2), debut_ts=T0 + 3600, duree_s=1800))

    assert registre.a_des_accidents_actifs(T0 + 4000) is True
    assert registre.a_des_accidents_actifs(T0) is False, "before the accident"
    assert registre.a_des_accidents_actifs(T0 + 7200) is False, "after the accident"

    source = (
        pathlib.Path(__file__).resolve().parents[1] / "trip_helper" / "osmnx_direct.py"
    ).read_text(encoding="utf-8")
    assert "_reg.a_des_accidents_actifs(" in source
    assert "if _persistent_cache is not None and not _accidents_actifs:" in source, (
        "the guard is not wired to the predicate: a disturbed duration can enter the cache"
    )


def test_r10_la_cle_de_decision_porte_les_accidents_actifs():
    """R10 — two distinct world states give two distinct signatures."""
    registre = _registre(taux=5.0)
    registre.tirer_journee(1, T0)
    assert registre.signature_active(T0 + 600) == "", "no accident placed: empty signature"

    registre._poser(Accident(arete=(1, 2), debut_ts=T0, duree_s=3600))
    avec = registre.signature_active(T0 + 600)
    assert avec, "an active accident must produce a signature"
    assert registre.signature_active(T0 + 7200) == "", "outside window: empty signature"

    registre._poser(Accident(arete=(3, 4), debut_ts=T0, duree_s=3600))
    assert registre.signature_active(T0 + 600) != avec, "two accidents ≠ one accident"

    agent = (
        pathlib.Path(__file__).resolve().parents[1]
        / "urban_mobility_agents"
        / "agents"
        / "llm_agent.py"
    ).read_text(encoding="utf-8")
    assert 'anticipation_key = f"{anticipation_key}|accidents:{_sig}"' in agent, (
        "the signature does not enter the key: a delayed agent would be served again its "
        "decision from before the delay"
    )


def test_le_retard_et_ses_deux_gardes_sont_livres_ensemble():
    """The C/D/E block is indivisible, and this test guards it.

    Shipping the delay without the itinerary cache guard would poison runs that
    asked for no accident; without the decision key, a delayed agent would replay its choice
    from before. If any of the three disappears, this test fails.
    """
    racine = pathlib.Path(__file__).resolve().parents[1]
    osmnx = (racine / "trip_helper" / "osmnx_direct.py").read_text(encoding="utf-8")
    agent = (racine / "urban_mobility_agents" / "agents" / "llm_agent.py").read_text(
        encoding="utf-8"
    )
    assert "registre.facteur_arete((u, v), ts)" in osmnx, "C — the delay has disappeared"
    assert "not _accidents_actifs" in osmnx, "D — the itinerary cache guard has disappeared"
    assert "accidents:{_sig}" in agent, "E — the decision key has disappeared"


def test_accident_actif_sur_sa_fenetre_seulement():
    """The window is half-open: the end instant is no longer active."""
    accident = Accident(arete=(1, 2), debut_ts=T0, duree_s=600)
    assert accident.actif_a(T0) is True
    assert accident.actif_a(T0 + 599) is True
    assert accident.actif_a(T0 + 600) is False
    assert accident.actif_a(T0 - 1) is False


# ── Work A — the draw is conditioned on BAAC statistics ──────────────────────


def test_loi_baac_livree_est_coherente():
    """The repository law sums to 1 where it should, and its factors average 1.

    This is what guarantees that the base rate REMAINS the mean rate: a factor averaging
    1.2 would quietly multiply every run by 1.2.
    """
    loi = LoiBaac.charger()
    assert 0.99 <= sum(loi.distribution_horaire.values()) <= 1.01
    assert 0.99 <= sum(loi.distribution_classe_vitesse.values()) <= 1.01
    moyenne_jours = sum(loi.facteur_jour_semaine.values()) / 7
    assert 0.99 <= moyenne_jours <= 1.01
    assert len(loi.distribution_horaire) == 24
    assert loi.taux_base_par_jour > 0


def test_loi_incoherente_est_refusee():
    """A distribution that does not sum to 1 must raise, not draw silently."""
    with pytest.raises(ValueError, match="sums to"):
        LoiBaac(
            taux_base_par_jour=1.5,
            distribution_horaire={h: 0.01 for h in range(24)},  # sum 0.24
            facteur_jour_semaine={j: 1.0 for j in JOURS},
            distribution_classe_vitesse={"31-50": 1.0},
            facteur_meteo={},
            facteur_meteo_etabli=False,
            millesimes=[],
        ).verifier()


def test_heure_suit_la_distribution_mesuree_et_non_l_uniforme():
    """The most marked conditioning: 17:00 must dominate 03:00, clearly.

    Without it, the uniform draw of the first slice placed as many accidents at 3 in
    the morning as at the evening peak.
    """
    registre = _registre(taux=30.0, graine=5)
    for jour in range(1, 60):
        registre.tirer_journee(jour, T0 + (jour - 1) * JOUR)
    heures = [(a.debut_ts - T0) % JOUR // 3600 for a in registre.accidents]
    assert len(heures) > 500, "sample too small to conclude"
    creux = sum(1 for h in heures if h == 3)
    pointe = sum(1 for h in heures if h == 17)
    assert pointe > 3 * max(creux, 1), (
        f"the 17:00 peak ({pointe}) does not dominate the 03:00 trough ({creux}): "
        "the hourly draw probably stayed uniform"
    )


def test_le_nombre_depend_du_jour_de_semaine():
    """Friday (×1.19) must produce more accidents than Sunday (×0.83)."""
    loi = LoiBaac.charger()
    assert loi.facteur_jour_semaine["vendredi"] > loi.facteur_jour_semaine["dimanche"]

    # T0 is a Monday 05:00: we align on the Friday and the Sunday of the same week.
    vendredi_ts = T0 + 4 * JOUR
    dimanche_ts = T0 + 6 * JOUR
    registre = _registre(taux=20.0, graine=11)
    taux_vendredi = registre._taux_du_jour(
        accidents_module.wall_clock(vendredi_ts).weekday(), None
    )
    taux_dimanche = registre._taux_du_jour(
        accidents_module.wall_clock(dimanche_ts).weekday(), None
    )
    assert taux_vendredi > taux_dimanche


def test_la_classe_d_axe_suit_baac_et_non_la_longueur_du_reseau():
    """The result that justifies all of work A.

    The simulated network is 58 % traffic-calmed zone (≤ 30 km/h), which carries only 8 % of
    real accidents. A draw pro rata to length alone would therefore place the majority
    of accidents there. With the law, the 31-50 class — 57 % of real accidents — must dominate
    although it weighs only a third of the kilometres.
    """
    aretes = [
        (1, 2, 58_000.0, 30),  # traffic-calmed zone: 58 % of the metres
        (2, 3, 34_000.0, 50),  # urban: 34 % of the metres
        (3, 4, 8_000.0, 80),  # expressway: 8 % of the metres
    ]
    registre = _registre(taux=30.0, graine=7, aretes=aretes)
    for jour in range(1, 60):
        registre.tirer_journee(jour, T0 + (jour - 1) * JOUR)

    comptes = {}
    for a in registre.accidents:
        comptes[a.classe_vitesse] = comptes.get(a.classe_vitesse, 0) + 1
    total = sum(comptes.values())
    assert total > 500, "sample too small to conclude"
    part_apaisee = comptes.get("<=30", 0) / total
    part_urbaine = comptes.get("31-50", 0) / total
    assert part_urbaine > part_apaisee, (
        f"the 31-50 class ({part_urbaine:.1%}) does not dominate the traffic-calmed zone "
        f"({part_apaisee:.1%}) although it carries 57 % of real accidents for 34 % of the "
        "metres: the draw probably still follows length alone"
    )
    assert part_apaisee < 0.20, (
        f"{part_apaisee:.1%} of accidents in traffic-calmed zone, against 8 % in BAAC"
    )


def test_facteur_meteo_est_neutre_par_decision():
    """Weather does NOT condition accident rates — author's decision of 2026-09-21.

    This is not work postponed but an accepted limit, and it rests on a
    measured fact: risky conditions do not exist in the simulated world. Out of the 2,920
    slots of a year served by `weather_loader`, fog takes ONE and snow
    FOUR — a factor for these conditions would never trigger. For the ones that
    remain, the `atm` and local weather nomenclatures are incommensurable.

    If this test fails because `facteur_meteo_etabli` became true, the only thing that
    could justify it is a weather set served to the simulation that really carries these
    conditions — not a new mapping fitted on the same data.
    """
    loi = LoiBaac.charger()
    assert loi.facteur_meteo_etabli is False
    assert all(v == 1.0 for v in loi.facteur_meteo.values())

    registre = _registre(taux=10.0)
    lundi = accidents_module.wall_clock(T0).weekday()
    assert registre._taux_du_jour(lundi, "pluie forte") == registre._taux_du_jour(
        lundi, None
    )


def test_classe_de_vitesse_tolere_les_formes_osm():
    """OSM writes speed in several ways: list, string, unit, absence."""
    assert classe_de_vitesse(50) == "31-50"
    assert classe_de_vitesse("50") == "31-50"
    assert classe_de_vitesse("50 km/h") == "31-50"
    assert classe_de_vitesse(["80", "90"]) == "71-90"
    assert classe_de_vitesse(30) == "<=30"
    assert classe_de_vitesse(130) == ">90"
    assert classe_de_vitesse(None) is None
    assert classe_de_vitesse("") is None
    assert classe_de_vitesse(0) is None
    assert classe_de_vitesse(500) is None


def test_surcharge_de_taux_remplace_la_mesure():
    """Non-null `taux_journalier` short-circuits the law — reserved for development."""
    registre_mesure = _registre(taux=None)
    registre_force = _registre(taux=50.0)
    lundi = accidents_module.wall_clock(T0).weekday()
    assert registre_force._taux_du_jour(lundi, None) > registre_mesure._taux_du_jour(
        lundi, None
    )


# ── F — manual placement, where the figures will come from ───────────────────


def test_f_pose_manuelle_accroche_l_arete_la_plus_proche():
    """F — a point resolves to an edge, and the accident enters the same registry."""
    aretes = [
        (1, 2, 1000.0, 50),
        (2, 3, 1000.0, 50),
        (3, 4, 1000.0, 80),
    ]
    registre = _registre(taux=5.0, aretes=aretes)
    registre._positions = {1: (43.60, 1.44), 2: (43.57, 1.43), 3: (43.50, 1.40), 4: (43.40, 1.30)}
    registre.tirer_journee(1, T0)
    avant = len(registre.accidents)

    pose = registre.poser_manuellement(lat=43.5701, lon=1.4301, debut_ts=T0 + 3 * 3600, duree_minutes=45)

    assert pose is not None
    assert pose.arete == (2, 3), f"nearest edge wrongly chosen: {pose.arete}"
    assert pose.duree_s == 45 * 60
    assert len(registre.accidents) == avant + 1, "the placed accident must enter the registry"
    # And it acts exactly like a drawn accident: same window, same factor.
    assert registre.facteur_arete((2, 3), T0 + 3 * 3600 + 60) > 1.0
    assert registre.facteur_arete((2, 3), T0) == 1.0


def test_f_pose_manuelle_refuse_une_duree_non_positive():
    """F — refusal logged, and nothing enters the world state."""
    registre = _registre(taux=5.0)
    registre._positions = {1: (43.6, 1.4), 2: (43.6, 1.4), 3: (43.6, 1.4), 4: (43.6, 1.4), 5: (43.6, 1.4)}
    registre.tirer_journee(1, T0)
    avant = len(registre.accidents)
    assert registre.poser_manuellement(43.6, 1.4, T0, 0) is None
    assert len(registre.accidents) == avant


def test_f_bouton_et_endpoint_existent():
    """F — the gesture is offered in the GUI and by the controller.

    Presence check, like R1 and R3: neither the GAMA GUI nor the HTTP route can be
    exercised from a unit test.
    """
    city = (MODELES / "City.gaml").read_text(encoding="utf-8")
    assert 'user_command "Poser un accident maintenant"' in city
    llm_agent = (MODELES / "LLMAgent.gaml").read_text(encoding="utf-8")
    assert 'do send to: "/accidents"' in llm_agent
    app = (
        pathlib.Path(__file__).resolve().parents[1] / "handle" / "application.py"
    ).read_text(encoding="utf-8")
    assert '@app.post(\n    "/accidents",' in app
    assert "régime d'accidents désactivé pour ce run" in app, (
        "a placement in a run where the regime is unticked must be refused explicitly"
    )
