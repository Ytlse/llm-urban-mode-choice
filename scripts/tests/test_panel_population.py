"""Tests of the control and sealing of the test set population (scripts/panel).

What is locked here, and why:
  * largest-remainder rounding sums EXACTLY to N — one persona too many or too few
    would make the name of the sealed directory lie;
  * the TOST returns `équivalent` when the CI90 fits within the bound, `écart` when the CI95 excludes the
    target and the gap exceeds the bound, `non concluant` in between;
  * the selection is deterministic, excludes those outside the scope and those under 5, and logs
    a deficit instead of hiding it;
  * the control returns `non mesurable` — never 0 — for a margin without a published target.

    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_panel_population.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.panel import control_population as ctl  # noqa: E402
from scripts.panel import seal_population as seal  # noqa: E402
from scripts.panel.reference_marges import (  # noqa: E402
    JOINT_TARGET, MOTORISATION, age_class, cible_jointe, marges, motorisation_class)


def _persona(pid: int, age: int, gender: str, occupation: str, cars: int, size: int,
             couronne: str) -> dict:
    return {
        "person_id": str(pid),
        "identity": {
            "name": f"P{pid}",
            "traits_json": {"age": age, "gender": gender, "main_occupation": occupation,
                            "number_of_cars": cars, "household_size": size,
                            "has_driving_license": age >= 18, "residence_zone": couronne},
            "home": {"lon": 1.44, "lat": 43.6},
            "activities": [],
        },
        "state": {}, "is_llm_based": True,
    }


def _pool(n_per_cell: int = 40) -> list[dict]:
    """A synthetic pool: each ring × car ownership cell populated with `n_per_cell`."""
    from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
    pool, pid = [], 0
    occupations = ["Travail à plein temps", "Retraité", "Scolaire (jusqu'au Bac)", "Étudiant",
                   "Travail à temps partiel", "Chômeur/recherche d'emploi", "Personne au foyer"]
    for c in COURONNES:
        for cars in (0, 1, 2):
            for k in range(n_per_cell):
                pid += 1
                pool.append(_persona(pid, 5 + (pid * 7) % 80, "Female" if pid % 2 else "Male",
                                     occupations[pid % 7], cars, 1 + pid % 4, c))
    # Noise to exclude: outside the scope, and a 3-year-old child.
    pid += 1
    pool.append(_persona(pid, 40, "Male", "Retraité", 1, 2, OUT_OF_PERIMETER))
    pid += 1
    pool.append(_persona(pid, 3, "Female", "Scolaire (jusqu'au Bac)", 1, 3, COURONNES[0]))
    return pool


# ── References ────────────────────────────────────────────────────────────────

def test_la_cible_jointe_gelee_est_lisible_et_somme_a_100():
    doc = cible_jointe(JOINT_TARGET)
    total = sum(v for row in doc["cible_pct"].values() for v in row.values())
    assert abs(total - 100.0) < 0.05
    # The counter-check recovers page 21 of the report (household base) within ± 0.5 pt.
    menage = doc["contre_epreuves"]["motorisation_base_menage_pct"]
    assert abs(menage["sans voiture"] - 19) < 0.5
    assert abs(menage["une voiture"] - 45) < 0.5
    assert abs(menage["deux voitures et +"] - 35) < 0.5


def test_les_marges_recalculees_disent_leur_source():
    """Gender, licence, pass… are not published by the report: their target is a
    frozen recomputation (cm1), and the source of each margin says so."""
    items = {m.nom: m for m in marges()}
    for nom in ("genre", "permis_adultes", "abonnement_tc", "logement", "immobile",
                "age_quinquennal", "taille_menage_personne"):
        assert items[nom].cible_pct is not None, nom
        assert abs(sum(items[nom].cible_pct.values()) - 100) < 0.05, nom
        assert "recalcul" in items[nom].source_cible and "non publié" in items[nom].source_cible, nom
    assert items["classe_age"].source_cible.startswith("AUAT")          # this one is published
    assert items["motorisation_personne"].cible_pct["deux voitures et +"] > 45   # person base
    assert 9 < items["immobile"].cible_pct["Oui"] < 12


def test_recodages():
    assert motorisation_class(0) == "sans voiture"
    assert motorisation_class("1") == "une voiture"
    assert motorisation_class(3) == "deux voitures et +"
    assert motorisation_class(None) is None
    assert age_class(4) is None          # below the surveyed population
    assert age_class(17) == "5-17 ans"
    assert age_class(18) == "18-24 ans"
    assert age_class(90) == "65 ans et +"


# ── Statistics ────────────────────────────────────────────────────────────────

def test_plus_fort_reste_somme_exactement_a_n():
    shares = {"a": 33.3333, "b": 33.3333, "c": 33.3334}
    for n in (10, 999, 1000, 1001):
        out = seal.largest_remainder(shares, n)
        assert sum(out.values()) == n
    doc = cible_jointe(JOINT_TARGET)
    cells = {f"{c} × {m}": doc["cible_pct"][c][m] for c in doc["cible_pct"] for m in MOTORISATION}
    assert sum(seal.largest_remainder(cells, 1000).values()) == 1000


def test_tost_trois_verdicts():
    # CI90 contained within ± 1 around the target → equivalent.
    assert ctl.tost(50.2, (49.6, 50.8), (49.4, 51.0), 50.0, 1.0) == ctl.TOST_EQUIVALENT
    # CI95 excludes the target AND gap > bound → gap.
    assert ctl.tost(55.0, (53.5, 56.5), (53.0, 57.0), 50.0, 1.0) == ctl.TOST_ECART
    # CI95 excludes the target but gap below the bound → inconclusive, not gap.
    assert ctl.tost(50.8, (50.5, 51.1), (50.4, 51.2), 50.0, 1.0) == ctl.TOST_INCONCLUSIF
    # Wide CI: neither equivalent nor gap.
    assert ctl.tost(52.0, (48.0, 56.0), (47.0, 57.0), 50.0, 1.0) == ctl.TOST_INCONCLUSIF


def test_clopper_pearson_bornes():
    lo, hi = ctl.clopper_pearson(0, 100, 0.05)
    assert lo == 0.0 and 0 < hi < 5
    lo, hi = ctl.clopper_pearson(100, 100, 0.05)
    assert hi == 100.0 and 95 < lo < 100
    lo, hi = ctl.clopper_pearson(50, 100, 0.05)
    assert lo < 50 < hi


# ── Selection ─────────────────────────────────────────────────────────────────

def test_selection_deterministe_et_exclusions():
    # 40 personas per cell; the largest share of the joint target is ≈ 20 %, so
    # n = 150 (max target ≈ 30) fits in each cell: no deficit expected.
    pool = _pool(40)
    chosen_a, journal_a = seal.select(pool, 150)
    chosen_b, journal_b = seal.select(list(reversed(pool)), 150)   # file order reversed
    assert len(chosen_a) == 150
    assert [p["person_id"] for p in chosen_a] == [p["person_id"] for p in chosen_b]
    ex = journal_a["vivier"]["exclus"]
    assert ex["hors_perimetre"] == 1 and ex["moins_de_5_ans"] == 1
    # Without household.id, each person is a one-person household — and the log says so.
    assert ex["sans_household_id_menage_d_une_personne"] == 480
    assert sum(journal_a["retenus_par_cellule"].values()) == 150
    assert not journal_a["deficits"]


def test_selection_journalise_le_deficit_au_lieu_de_le_cacher():
    pool = _pool(40)   # 40 per cell: the "1ere couronne × deux voitures et +" cell
    chosen, journal = seal.select(pool, 480)   # asks 20 % more than some cells
    assert len(chosen) == 480
    assert journal["deficits"], "a pool that is too small must produce a visible deficit"
    assert journal["reports"] and all(r["n"] > 0 for r in journal["reports"])
    # A deficit is filled first within the SAME ring.
    for r in journal["reports"]:
        if r["portee"] == "même couronne":
            assert r["deficit"].split(" × ")[0] == r["vers"].split(" × ")[0]


def test_descente_reduit_la_perte_sans_bouger_les_cellules():
    """The descent swaps households of the same SUB-CELL (cell, present count, declared
    size): the multi-margin loss decreases, the 12 cell counts stay those of
    the allocation."""
    from collections import Counter
    pool = _pool(40)
    chosen, journal = seal.select(pool, 150)
    d = journal["descente"]
    assert d["echanges"] > 0
    assert d["perte_apres_pt"] < d["perte_avant_pt"]
    assert "occupation" in d["marges"] and d["marges"]["occupation"]["mesuree"]
    # The synthetic pool has neither housing nor pass: the descent says so, does not invent it.
    assert "logement" in d["marges_non_mesurees"]
    cells = Counter(f"{p['identity']['traits_json']['residence_zone']} × "
                    f"{seal.motorisation_class(p['identity']['traits_json']['number_of_cars'])}"
                    for p in chosen)
    assert dict(cells) == {c: n for c, n in journal["retenus_par_cellule"].items() if n}
    assert journal["version"] == seal.SELECTION_RULE


def _pool_menages(n_menages_par_cellule: int = 25) -> list[dict]:
    """A pool of HOUSEHOLDS: sizes 1 to 3, one cell per household, household.id at the root."""
    from mobility_core.population_reference import COURONNES
    pool, pid, hid = [], 0, 0
    occupations = ["Travail à plein temps", "Retraité", "Scolaire (jusqu'au Bac)", "Étudiant",
                   "Travail à temps partiel", "Chômeur/recherche d'emploi", "Personne au foyer"]
    for c in COURONNES:
        for cars in (0, 1, 2):
            for k in range(n_menages_par_cellule):
                hid += 1
                size = 1 + (hid % 3)
                for j in range(size):
                    pid += 1
                    rec = _persona(pid, 5 + (pid * 7) % 80, "Female" if pid % 2 else "Male",
                                   occupations[pid % 7], cars, size, c)
                    rec["household"] = {"id": f"h{hid}", "iris_id": None, "commune_id": None}
                    rec["immobile"] = (pid % 9 == 0)
                    rec["identity"]["activities"] = ([{"purpose": "home"}] if rec["immobile"] else
                                                     [{"purpose": "home"}, {"purpose": "work"}, {"purpose": "home"}])
                    pool.append(rec)
    return pool


def test_selection_par_menage_retient_des_menages_entiers():
    """A household enters or leaves whole: never one member without the others."""
    from collections import Counter, defaultdict
    pool = _pool_menages(25)
    chosen, journal = seal.select(pool, 300)
    assert len(chosen) == 300
    membres, retenus = defaultdict(set), defaultdict(set)
    for r in pool:
        membres[r["household"]["id"]].add(r["person_id"])
    for r in chosen:
        retenus[r["household"]["id"]].add(r["person_id"])
    for h, ids in retenus.items():
        assert ids == membres[h], f"household {h} fragmented"
    assert journal["menages_retenus"]["n"] == len(retenus)
    assert journal["menages_retenus"]["membres_presents"] == 300
    # Immobile persons are a margin: measured, and brought closer to the target (10.6 %).
    im = journal["descente"]["marges"]["immobile"]
    assert im["mesuree"] and im["ecart_max_apres_pt"] <= im["ecart_max_avant_pt"]
    cells = Counter(f"{p['identity']['traits_json']['residence_zone']} × "
                    f"{seal.motorisation_class(p['identity']['traits_json']['number_of_cars'])}"
                    for p in chosen)
    assert dict(cells) == {c: n for c, n in journal["retenus_par_cellule"].items() if n}


def test_selection_refuse_un_vivier_insuffisant():
    with pytest.raises(ValueError, match="pool too small"):
        seal.select(_pool(2), 1000)


# ── Control ───────────────────────────────────────────────────────────────────

def test_controle_rend_non_mesurable_et_jamais_zero_sans_cible(tmp_path):
    pool = _pool(40)
    path = tmp_path / "pop.json"
    path.write_text(json.dumps(pool), encoding="utf-8")
    report = ctl.run_control(path, borne=1.0, n_min=30, n_min_cellule=50)
    by_name = {m["marge"]: m for m in report["marges"]}
    # The synthetic pool carries neither housing nor pass: these margins come out
    # "non mesurable — aucun persona ne porte cette variable", never 0.
    for nom in ("logement", "abonnement_tc"):
        assert by_name[nom]["verdict"] == ctl.NON_MESURABLE, nom
        assert by_name[nom]["chi2"] is None
    # Gender, for its part, is now measurable (frozen recomputed target).
    assert by_name["genre"]["chi2"] is not None
    # All immobile (no activity): the margin says so, the mobility section too.
    assert by_name["immobile"]["constats"][0]["observe_pct"] == 100.0
    assert report["menages_et_mobilite"]["part_immobiles_pct"] == 100.0
    assert report["compteurs"]["hors_perimetre"] == 1
    assert report["compteurs"]["age_sous_5_ans"] == 1
    assert set(report["verdicts"]) == {ctl.CONFORME, ctl.A_CORRIGER, ctl.A_PUBLIER, ctl.NON_MESURABLE}
    # The summary lists out-of-scope as closable at sealing.
    assert any("hors des 453 communes" in row["ecart"] for row in report["synthese"])
    # The cross-check log carries the nine lines of the protocol.
    assert len(report["recoupement"]) == 13  # table §2.1 of protocol v1.5 (13 lines)


def test_controle_uniforme_est_a_corriger_sur_la_couronne(tmp_path):
    """A pool with equal cells puts 25 % per ring: the 3rd (target 15.4 %) is off target."""
    pool = _pool(60)
    path = tmp_path / "pop.json"
    path.write_text(json.dumps(pool), encoding="utf-8")
    report = ctl.run_control(path, borne=1.0, n_min=30, n_min_cellule=50)
    couronne = {m["marge"]: m for m in report["marges"]}["couronne"]
    assert couronne["verdict"] == ctl.A_CORRIGER
    assert report["verdicts"][ctl.A_CORRIGER] >= 1


# ── Rule v5: scope, schoolchildren line, household-base car ownership ────────

def test_regle_v5_journalise_le_perimetre_et_les_departements():
    """The rule is `panel_seal_v5`, holds the six age classes, and the log says where
    the kept ones come from (department of `household.commune_id`) and which scope is declared.

    The hash **salt** stays that of v4, deliberately: v5 only changes the loss function
    of the descent, not the order in which households come up."""
    assert seal.SELECTION_RULE == "panel_seal_v5"
    assert seal.SELECTION_NAMESPACE == "panel_seal_v4"
    assert "classe_age" in seal.DESCENTE_MARGES and "occupation" in seal.DESCENTE_MARGES
    assert seal.DEFAULT_SEAL_DIR.name == "population_1000_PANEL_v6"
    # The two margins that the ALLOCATION holds (and not the descent) since v5.
    assert seal.ALLOCATION_MARGES == ("taille_menage_personne", "motorisation_menage")
    assert seal.ALLOCATION_TOLERANCE_ALARME_PT == 1.0   # the control's indifference bound
    pool = _pool_menages(25)
    deps = ["31", "32", "81", "82", "09", "11"]
    for i, rec in enumerate(pool):
        rec["household"]["commune_id"] = f"{deps[int(rec['household']['id'][1:]) % 6]}123"
    chosen, journal = seal.select(pool, 300)
    per = journal["perimetre"]
    assert "453 communes" in per["definition"] and "polygone" in per["definition"]
    assert set(per["departements_attendus"]) == set(deps)
    assert sum(per["retenus_par_departement"].values()) == 300
    assert per["departements_representes"] == 6
    assert per["retenus_sans_commune"] == 0
    assert "panel_seal_v4" in journal["regle"]   # the hash salt, quoted in the rule


def test_commune_du_domicile_lit_household_puis_le_trait():
    rec = {"household": {"commune_id": "undefined"}, "identity": {"traits_json": {"residence_insee": "9038"}}}
    assert seal._commune_of(rec) == "09038"
    rec["household"]["commune_id"] = "31555"
    assert seal._commune_of(rec) == "31555"
    assert seal._commune_of({"household": {}, "identity": {"traits_json": {}}}) is None


def test_independance_rend_non_mesurable_sur_une_table_degeneree(tmp_path):
    """A population where a whole car ownership column is empty does not crash the
    control: the cross-tabulation comes out `non mesurable`, with the empty level named."""
    pool = [p for p in _pool(40) if p["identity"]["traits_json"]["number_of_cars"] == 1]
    path = tmp_path / "pop.json"
    path.write_text(json.dumps(pool), encoding="utf-8")
    report = ctl.run_control(path, borne=1.0, n_min=30, n_min_cellule=50)
    assert report["independance"]["verdict"] == ctl.NON_MESURABLE
    assert "sans voiture" in report["independance"]["raison"]


def _pool_scolaires(n_menages: int = 30, part_etudes: float = 0.5) -> list[dict]:
    """A pool with parent + schoolchild households; `part_etudes` of the children go to school."""
    from mobility_core.population_reference import COURONNES
    pool, pid = [], 0
    for h in range(n_menages):
        c = COURONNES[h % 4]
        for age, occ in ((40, "Travail à plein temps"), (9 + h % 9, "Scolaire (jusqu'au Bac)")):
            pid += 1
            rec = _persona(pid, age, "Female" if pid % 2 else "Male", occ, h % 3, 2, c)
            rec["household"] = {"id": f"h{h}", "iris_id": None, "commune_id": "31555"}
            rec["immobile"] = False
            if occ.startswith("Scolaire"):
                purpose = "education" if (h / n_menages) < part_etudes else "leisure"
            else:
                purpose = "work"
            rec["identity"]["activities"] = [{"purpose": "home"}, {"purpose": purpose}, {"purpose": "home"}]
            pool.append(rec)
    return pool


def test_controle_mesure_les_scolaires_avec_activite_etudes(tmp_path):
    """The "scolaires avec activité d'études" line counts mobile schoolchildren aged 6-17, and a
    rate below 88 % comes out in the summary as a gap to publish (eqasim lever, not the selection)."""
    path = tmp_path / "pop.json"
    path.write_text(json.dumps(_pool_scolaires(30, 0.5)), encoding="utf-8")
    report = ctl.run_control(path, borne=1.0, n_min=30, n_min_cellule=50)
    m = report["menages_et_mobilite"]
    assert m["scolaires_6_17"] == 30 and m["scolaires_mobiles"] == 30
    assert m["scolaires_avec_activite_etudes"] == 15
    assert m["part_scolaires_avec_etudes_pct"] == 50.0
    assert m["reference_enquete"]["part_scolaires_avec_etudes_pct"] == [90.0, 95.0]
    rows = [r for r in report["synthese"] if r["ecart"] == "scolaires sans activité d'études"]
    assert len(rows) == 1 and rows[0]["verdict"] == ctl.A_PUBLIER
    assert "eqasim" in rows[0]["refermable_au_scellement"]
    assert "Scolaires (6-17 ans) avec activité d'études : 15/30" in ctl.render_text(report)
    assert "50.0 %" in ctl.render_markdown(report)

    # Above the threshold: no gap in the summary.
    path.write_text(json.dumps(_pool_scolaires(30, 0.95)), encoding="utf-8")
    report = ctl.run_control(path, borne=1.0, n_min=30, n_min_cellule=50)
    assert report["menages_et_mobilite"]["part_scolaires_avec_etudes_pct"] >= 88.0
    assert not [r for r in report["synthese"] if r["ecart"] == "scolaires sans activité d'études"]


def test_le_compte_des_activites_hors_perimetre_distingue_controle_et_zero():
    """Without the `perimetre` key, the population was not checked: the log says so, it does not invent 0."""
    pool = _pool_menages(5)
    j = seal.count_removed_out_of_perimeter(pool)
    assert j["controle"] is False and j["personas_controles"] == 0
    for i, rec in enumerate(pool):
        rec["perimetre"] = {"activites_hors_perimetre_supprimees": 2 if i < 3 else 0}
    j = seal.count_removed_out_of_perimeter(pool)
    assert j == {"controle": True, "personas_controles": len(pool),
                 "activites_hors_perimetre_supprimees": 6, "personas_touches": 3}
    chosen, journal = seal.select(pool, 60)
    assert "activites_hors_perimetre" in journal["perimetre"]
    assert journal["perimetre"]["activites_hors_perimetre"]["controle"] is True


def test_deplacements_comptent_le_retour_au_domicile(tmp_path):
    """The activity chain is cyclic (evening home merged with the morning one):
    n activities = n trips for a mobile person, 0 for an immobile one (a single activity)."""
    def acts(purposes):
        return [{"id": str(i), "purpose": p, "start_time": 3600.0 * (8 + 3 * i),
                 "end_time": 3600.0 * (10 + 3 * i), "scheduled_start_time": None,
                 "location": {"lon": 1.44, "lat": 43.6}} for i, p in enumerate(purposes)]
    pool = _pool_menages(n_menages_par_cellule=10)
    for k, rec in enumerate(pool):
        if k % 10 == 0:
            rec["identity"]["activities"] = acts(["home"])          # immobile
            rec["immobile"] = True
        elif k % 2 == 0:
            rec["identity"]["activities"] = acts(["home", "work"])  # 2 trips (outbound, return)
            rec["immobile"] = False
        else:
            rec["identity"]["activities"] = acts(["home", "work", "shop"])  # 3 trips
            rec["immobile"] = False
    path = tmp_path / "pop.json"
    path.write_text(json.dumps(pool), encoding="utf-8")
    report = ctl.run_control(path, borne=1.0, n_min=30, n_min_cellule=50)
    m = report["menages_et_mobilite"]
    n_imm = sum(1 for k in range(len(pool)) if k % 10 == 0)
    n_two = sum(1 for k in range(len(pool)) if k % 10 != 0 and k % 2 == 0)
    n_three = len(pool) - n_imm - n_two
    attendu_mobile = (2 * n_two + 3 * n_three) / (n_two + n_three)
    assert m["immobiles"] == n_imm
    # The report rounds to 3 decimals.
    assert abs(m["deplacements_par_persona_mobile"] - attendu_mobile) < 5e-4
    assert abs(m["deplacements_par_persona"] - (2 * n_two + 3 * n_three) / len(pool)) < 5e-4


def test_la_motorisation_base_menage_est_une_marge_ponderee_de_la_descente():
    """Household-base car ownership enters the descent's loss, and it is
    counted there on a **household base** — each persona weighs the inverse of the declared size of its
    household. Counting it with weight 1 would compare a population of persons to a target of
    households: this is the base error the control page forbids."""
    assert "motorisation_menage" in seal.DESCENTE_MARGES
    assert "motorisation_menage" in seal.DESCENTE_MARGES_MENAGE
    assert "motorisation_personne" not in seal.DESCENTE_MARGES_MENAGE

    # Two personas of the same level but of different household sizes do not weigh
    # the same in the household margin's field, and weigh the same in a person margin.
    defs = [("motorisation_menage", lambda p: "sans voiture", {"sans voiture": 100.0}),
            ("genre", lambda p: "Femmes", {"Femmes": 100.0})]
    etat = seal._Etat(defs)

    class _P:
        def __init__(self, taille): self.poids_menage = 1.0 / taille
    etat.add(_P(1))
    etat.add(_P(4))
    assert etat.fields["motorisation_menage"] == pytest.approx(1.25)
    assert etat.fields["genre"] == 2


# ── Rule v5: allocation per sub-cell (cell × present × declared) ──────────────

def _pool_v5(par_taille: int = 26, avec_enfants_absents: bool = True) -> list[dict]:
    """A pool of households of declared sizes 1 to 6, in the 12 cells.

    Some of the households carry a child under 5, outside the surveyed population: their
    PRESENT count is lower than their DECLARED size, which is exactly the dimension
    that the v5 sub-cell distinguishes (and the only `1/taille` weight lever of the real
    pool: 52 members out of 1,052 in v4).
    """
    from mobility_core.population_reference import COURONNES
    occupations = ["Travail à plein temps", "Retraité", "Scolaire (jusqu'au Bac)", "Étudiant",
                   "Travail à temps partiel", "Chômeur/recherche d'emploi", "Personne au foyer"]
    pool, pid, hid = [], 0, 0
    for c in COURONNES:
        for cars in (0, 1, 2):
            for taille in (1, 2, 3, 4, 5, 6):
                for k in range(par_taille):
                    hid += 1
                    bebe = avec_enfants_absents and taille >= 2 and k % 5 == 0
                    ages = [18 + (hid * 7) % 60] + [5 + (hid * 3 + j * 11) % 70
                                                    for j in range(taille - 1)]
                    if bebe:
                        ages[-1] = 3            # excluded from the surveyed population
                    for j, age in enumerate(ages):
                        pid += 1
                        rec = _persona(pid, age, "Female" if pid % 2 else "Male",
                                       occupations[pid % 7], cars, taille, c)
                        rec["household"] = {"id": f"h{hid}", "iris_id": None,
                                            "commune_id": f"{(31, 32, 81, 82, 9, 11)[hid % 6]:02d}123"}
                        rec["immobile"] = (hid % 9 == 0)
                        rec["identity"]["activities"] = (
                            [{"purpose": "home"}] if rec["immobile"] else
                            [{"purpose": "home"}, {"purpose": "work"}, {"purpose": "home"}])
                        pool.append(rec)
    return pool


def test_les_cibles_de_l_allocation_sont_lues_sur_les_cibles_gelees():
    """No literal: the two allocated margins take their target from `cj1` / `cm1`,
    renormalised to 100 — a target that moves must move the selection."""
    cibles = seal.cibles_allocation()
    assert set(cibles) == set(seal.ALLOCATION_MARGES)
    reference = {m.nom: m for m in marges()}
    for nom, row in cibles.items():
        assert set(row) == set(reference[nom].cible_pct), nom
        assert sum(row.values()) == pytest.approx(100.0), nom
        brut = reference[nom].cible_pct
        total = sum(brut.values())
        assert row == pytest.approx({k: 100.0 * v / total for k, v in brut.items()})
    # Household car ownership is indeed the report's target p. 21 (19 / 45 / 35), not the person
    # base (13.6 / 37.8 / 48.7) — this is the base confusion that v4 left open.
    assert 18.5 < cibles["motorisation_menage"]["sans voiture"] < 20.0


def test_l_allocation_par_sous_cellule_somme_aux_cibles_de_cellule():
    """The per-sub-cell counts sum EXACTLY to the twelve cell counts in
    persons, and hence to N: this is the constraint the extra dimension must not drop."""
    from collections import Counter
    pool = _pool_v5()
    menages, _ = seal.group_households(pool)
    targets = seal.largest_remainder(
        {f"{c} × {m}": cible_jointe(JOINT_TARGET)["cible_pct"][c][m]
         for c in cible_jointe(JOINT_TARGET)["cible_pct"] for m in MOTORISATION}, 400)
    effectif, deficits, reports = seal.effectifs_par_cellule(menages, targets)
    assert not deficits and not reports        # the test pool fills all the cells
    cibles_sc, journal = seal.allouer_sous_cellules(seal.inventaire(menages), effectif, 400,
                                                    seal.cibles_allocation())
    par_cellule: Counter = Counter()
    for (cellule, presents, _declares), n in cibles_sc.items():
        par_cellule[cellule] += n * presents
    assert dict(par_cellule) == {c: v for c, v in effectif.items() if v}
    assert sum(par_cellule.values()) == 400 == journal["personnes"]
    # Each sub-cell stays within the pool inventory: what does not exist is not allocated.
    inv = seal.inventaire(menages)
    assert all(n <= inv[key] for key, n in cibles_sc.items())
    assert journal["sous_cellules"]["servies"] == len(cibles_sc) <= journal["sous_cellules"]["vivier"]


def test_l_allocation_tient_les_deux_marges_dans_la_tolerance_qu_elle_annonce():
    """The retained tolerance is found by bisection, and the obtained gaps respect it.

    The log says what was asked (targets), what was obtained (shares), the gap, the
    tolerance and the number of programs solved — enough to replay the decision."""
    pool = _pool_v5()
    _retenus, journal = seal.select(pool, 400)
    a = journal["allocation"]
    assert a["marges_allouees"] == list(seal.ALLOCATION_MARGES)
    assert a["tolerance"]["retenue_pt"] <= seal.ALLOCATION_TOLERANCE_ALARME_PT
    assert a["tolerance"]["depassee"] is False
    assert a["ecart_max_pt"] <= a["tolerance"]["retenue_pt"] + 1e-6
    for nom, cibles in a["cibles_pct"].items():
        for modalite, cible in cibles.items():
            assert abs(a["parts_obtenues_pct"][nom][modalite] - cible) <= a["tolerance"]["retenue_pt"] + 1e-6
    # The bisection goes down to the announced step: the tolerance just below is infeasible.
    essais = a["tolerance"]["essais"]
    assert len(essais) >= 2 and essais[0]["tolerance_pt"] == seal.ALLOCATION_TOLERANCE_MAX_PT
    assert any(not e["faisable"] for e in essais), "the bisection must have hit the infeasible"
    assert a["tolerance"]["programmes_resolus"] == len(essais) + 1
    assert a["duree_s"] >= 0


def test_l_allocation_est_deterministe_et_independante_de_l_ordre_du_fichier():
    """Same pool, same allocation and same kept ones — whatever the order of the rows."""
    pool = _pool_v5()
    retenus_a, journal_a = seal.select(pool, 400)
    retenus_b, journal_b = seal.select(list(reversed(pool)), 400)
    assert [p["person_id"] for p in retenus_a] == [p["person_id"] for p in retenus_b]
    assert journal_a["household_ids"] == journal_b["household_ids"]
    for cle in ("cibles_menages",):
        assert journal_a["allocation"]["sous_cellules"][cle] == journal_b["allocation"]["sous_cellules"][cle]
    assert journal_a["allocation"]["tolerance"]["retenue_pt"] == journal_b["allocation"]["tolerance"]["retenue_pt"]


def test_la_descente_ne_deplace_aucune_sous_cellule():
    """The swap operator pairs at constant sub-cell: the two allocated margins
    come out of the descent INTACT, and so do the twelve cell counts.

    This is what replaces v4's impossible trade-off between household-base car ownership
    and household size: what the allocation fixes, the descent can no longer undo."""
    from collections import Counter
    pool = _pool_v5()
    retenus, journal = seal.select(pool, 400)
    d = journal["descente"]
    assert d["echanges"] > 0, "the descent must still work on the other margins"
    for nom in seal.ALLOCATION_MARGES:
        marge = d["marges"][nom]
        assert marge["mesuree"]
        assert marge["avant_pct"] == marge["apres_pct"], nom
        assert marge["ecart_max_apres_pt"] == marge["ecart_max_avant_pt"] <= 1.0, nom
    # The per-sub-cell counts of the kept ones are those the allocation asked for.
    par_sc: Counter = Counter()
    for rec in retenus:
        tr = rec["identity"]["traits_json"]
        par_sc[(f"{tr['residence_zone']} × {seal.motorisation_class(tr['number_of_cars'])}",
                rec["household"]["id"])] += 1
    tailles = Counter()
    for (cellule, _hid), presents in par_sc.items():
        tailles[cellule] += presents
    assert dict(tailles) == {c: v for c, v in journal["retenus_par_cellule"].items() if v}
    assert journal["allocation"]["retenus_par_sous_cellule"] == \
        journal["allocation"]["sous_cellules"]["cibles_menages"]


def test_une_sous_cellule_sous_remplie_se_reporte_dans_sa_cellule_et_s_alarme(monkeypatch, caplog):
    """A deficit is never filled silently. The inventory bounds the program, so an
    under-filled sub-cell is impossible by construction: we force the case and check
    that the carry-over is logged, alarmed, that the twelve cell counts stay exact,
    and that the selection exits with code 1."""
    import logging
    from collections import Counter
    pool = _pool_v5()
    vrai = seal.allouer_sous_cellules

    def _sur_alloue(inv, effectif, n, cibles):
        cibles_sc, journal = vrai(inv, effectif, n, cibles)
        # We ask for one household MORE than exist in a sub-cell, and as many
        # fewer in a sub-cell of the SAME cell and the SAME present count: the
        # person count of the cell stays that of the allocation, but a sub-cell
        # becomes impossible to serve.
        for cible in sorted(cibles_sc):
            delta = inv[cible] + 1 - cibles_sc[cible]
            jumelles = [k for k in sorted(cibles_sc)
                        if k != cible and k[:2] == cible[:2] and cibles_sc[k] >= delta]
            if delta >= 1 and jumelles:
                cibles_sc[cible] = inv[cible] + 1
                cibles_sc[jumelles[0]] -= delta
                return cibles_sc, journal
        raise AssertionError("the test pool does not allow forcing an empty sub-cell")

    monkeypatch.setattr(seal, "allouer_sous_cellules", _sur_alloue)
    with caplog.at_level(logging.ERROR, logger="panel.seal"):
        retenus, journal = seal.select(pool, 400)
    a = journal["allocation"]
    assert a["manques"] and a["manques"][0]["manque_menages"] == 1
    assert any("[ALARME]" in r.message and "sous-remplie" in r.message for r in caplog.records)
    reports = [r for r in journal["reports"] if r["portee"].startswith("même cellule")]
    assert reports and sum(r["n"] for r in reports) == a["manques"][0]["manque_personnes"]
    assert journal["deficits"], "a carried-over shortfall remains a visible deficit"
    # The twelve cell counts in persons stay exact despite the carry-over.
    cellules: Counter = Counter()
    for rec in retenus:
        tr = rec["identity"]["traits_json"]
        cellules[f"{tr['residence_zone']} × {seal.motorisation_class(tr['number_of_cars'])}"] += 1
    assert dict(cellules) == {c: v for c, v in journal["cibles"].items() if v}
    assert len(retenus) == 400


def test_un_vivier_qui_ne_porte_pas_les_grands_menages_le_dit_et_sort_en_code_1(tmp_path, caplog):
    """The `_pool` pool has no household of declared size 5 or more: the size margin
    CANNOT be held. The allocation does not silently get "as close as possible" — it
    states the tolerance that was needed, raises the alarm, and the selection exits with code 1."""
    import argparse
    import logging
    pool = _pool(60)
    path = tmp_path / "vivier.json"
    path.write_text(json.dumps(pool), encoding="utf-8")
    args = argparse.Namespace(pool=path, n=150, out=tmp_path / "out.json", selection_json=None,
                              exclure=None)   # whole pool — cf. `--exclure`
    with caplog.at_level(logging.ERROR, logger="panel.seal"):
        code = seal.cmd_select(args)
    assert code == 1
    assert any("[ALARME]" in r.message and "marges allouées" in r.message for r in caplog.records)
    journal = json.loads((tmp_path / "out_selection.json").read_text(encoding="utf-8"))
    a = journal["allocation"]
    assert a["tolerance"]["depassee"] is True
    assert a["ecart_max_pt"] > seal.ALLOCATION_TOLERANCE_ALARME_PT
    # The gap is that of the level the pool does not carry, and it is named.
    assert abs(a["ecarts_pt"]["taille_menage_personne"]["5 et +"]) > 1.0
    assert (tmp_path / "out.json").exists(), "the N are delivered, the gap is declared"
