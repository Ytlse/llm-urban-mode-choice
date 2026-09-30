"""Drawing a cohort disjoint from the reference cohort (by household).

One test per rule of the spec, the number in the name. What is locked here:

  * the disjunction bears on the HOUSEHOLD and is checked on the RESULT, not only at the input —
    a filter using the wrong key would silently produce an overlapping cohort, and axis 2
    would be published on sand;
  * the hash salt does not move: with an empty exclusion, the chain gives back the v6 cohort to the
    sha256, and that is the only proof that the gap between two cohorts comes from the exclusion;
  * a cohort whose file has moved since its sealing is no longer the one we think we are
    excluding, and there is no knowing in which direction: it is an error, not a warning.

    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_disjoint_cohort.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.panel import seal_population as seal  # noqa: E402
from scripts.tests.test_panel_population import _pool  # noqa: E402

COHORTE_1 = REPO_ROOT / "data" / "population" / "population_1000_PANEL_v6"
COHORTE_2 = REPO_ROOT / "data" / "population" / "population_1000_PANEL_v6_c2"
VIVIER = (REPO_ROOT / "scripts" / "data" / "population" / "Temp" / "4_zone_enriched"
          / "toulouse_population_10000.json")

SHA_V6 = "412efada802f79e8a72976ba25e0c7db8c9404adaed7c1f5e3e3d6afa3531db6"

besoin_cohortes = pytest.mark.skipif(
    not (COHORTE_1.exists() and COHORTE_2.exists()),
    reason="both sealed cohorts are required")
besoin_vivier = pytest.mark.skipif(not VIVIER.exists(), reason="pool missing")


def _menages(pop: Path) -> set[str]:
    return {str((r.get("household") or {}).get("id"))
            for r in json.loads(pop.read_text(encoding="utf-8"))}


def _personnes(pop: Path) -> set[str]:
    return {str(r.get("person_id"))
            for r in json.loads(pop.read_text(encoding="utf-8"))}


def _pool_mixte(n_per_cell: int = 40) -> list[dict]:
    """Synthetic pool where half of the personas live in households of TWO, the other half alone.

    The mix is not cosmetic: the twelve cell counts are equalities in
    persons, and a pool made only of two-person households can only compose even
    counts. The joint target produces odd ones, and the integer allocation then has no solution.

    Pairs form between immediate neighbours, which belong to the same cell — `_pool`
    fills each cell in blocks of `n_per_cell` —, otherwise the household would be "mixed" and
    discarded. The declared size follows the present size.
    """
    pool = _pool(n_per_cell)
    for i, rec in enumerate(pool):
        apparie = (i % 4) < 2 and (i + 1) < len(pool)
        rec["household"] = {"id": f"h{i // 2}" if apparie else f"s{i}"}
        rec["identity"]["traits_json"]["household_size"] = 2 if apparie else 1
    return pool


# ── R1: the exclusion cuts down the pool ──────────────────────────────────────

@besoin_cohortes
def test_R1_l_exclusion_retire_les_menages_de_la_cohorte_citee():
    exclus, journal = seal.charger_exclusions([COHORTE_1])
    assert len(exclus) == 499, "v6 retains 499 households (MANIFEST: menages_retenus.n)"
    assert journal[0]["nom"] == "population_1000_PANEL_v6"
    assert journal[0]["personnes"] == 1000


def test_R1_le_vivier_ampute_se_lit_dans_le_journal():
    pool = _pool_mixte()
    # Two-person households carry an EVEN identifier (`h{i // 2}` for i = 0, 1, 4, 5, 8, 9…):
    # h0 groups personas 0 and 1, h2 personas 4 and 5. s2 is a one-person household.
    _, journal = seal.select(pool, 150, exclure_menages={"h0", "h2", "s2"})
    assert journal["vivier"]["menages_exclus"] == 3
    assert journal["vivier"]["exclus"]["exclus_cohorte_anterieure"] == 5


# ── R2: the disjunction bears on the household, not on the person ─────────────

def test_R2_exclure_un_menage_emporte_tous_ses_membres():
    pool = _pool_mixte()
    retenus, _ = seal.select(pool, 150, exclure_menages={"h0"})
    # h0 groups the first two personas of the pool: neither of them must reappear.
    membres_h0 = {str(r["person_id"]) for r in pool if r["household"]["id"] == "h0"}
    assert len(membres_h0) == 2
    assert membres_h0 & {str(r["person_id"]) for r in retenus} == set()


# ── R3: no overlap, neither household nor person ──────────────────────────────

@besoin_cohortes
def test_R3_les_deux_cohortes_scellees_ne_se_recouvrent_pas():
    p1, p2 = COHORTE_1 / "population.json", COHORTE_2 / "population.json"
    assert _menages(p1) & _menages(p2) == set()
    assert _personnes(p1) & _personnes(p2) == set()
    assert len(_personnes(p1)) == len(_personnes(p2)) == 1000


# ── R4: the manifest says what the cohort is disjoint from ────────────────────

@besoin_cohortes
def test_R4_le_manifeste_nomme_les_cohortes_exclues_et_leur_sha256():
    m = yaml.safe_load((COHORTE_2 / "MANIFEST.yaml").read_text(encoding="utf-8"))
    dis = m["disjonction"]
    assert dis["menages_exclus"] == 499
    citees = {c["nom"]: c["sha256"] for c in dis["cohortes"]}
    assert citees == {"population_1000_PANEL_v6": SHA_V6}


@besoin_cohortes
def test_R4_une_cohorte_sur_vivier_entier_porte_une_disjonction_nulle():
    """`null`, not a missing key: otherwise an old manifest cannot be told from a draw
    on the whole pool."""
    m = yaml.safe_load((COHORTE_1 / "MANIFEST.yaml").read_text(encoding="utf-8"))
    assert m.get("disjonction", "ABSENTE") in (None, "ABSENTE")


# ── R5: the selection rule changes name ───────────────────────────────────────

@besoin_cohortes
def test_R5_la_regle_declaree_distingue_le_vivier_ampute():
    m2 = yaml.safe_load((COHORTE_2 / "MANIFEST.yaml").read_text(encoding="utf-8"))
    m1 = yaml.safe_load((COHORTE_1 / "MANIFEST.yaml").read_text(encoding="utf-8"))
    assert m2["selection"]["version"] == seal.SELECTION_RULE_DISJOINT
    assert m1["selection"]["version"] == seal.SELECTION_RULE
    assert seal.SELECTION_RULE_DISJOINT != seal.SELECTION_RULE


def test_R5_le_journal_porte_la_regle_du_vivier_entier_sans_exclusion():
    _, journal = seal.select(_pool_mixte(), 150)
    assert journal["version"] == seal.SELECTION_RULE


# ── R6: the territorial check does not weaken ─────────────────────────────────

@besoin_cohortes
def test_R6_la_cohorte_disjointe_passe_le_controle_sans_indulgence():
    m = yaml.safe_load((COHORTE_2 / "MANIFEST.yaml").read_text(encoding="utf-8"))
    ctl_ = m["controle"]
    assert ctl_["verdicts"]["à corriger"] == 0
    assert ctl_["borne_tost_pt"] == 1.0, "the whole-pool bound, not a loosened bound"
    assert ctl_["n_min"] == 30 and ctl_["n_min_cellule"] == 50


# ── R7: a deficit due to the exclusion is told apart from a pool deficit ──────

def test_R7_un_deficit_cause_par_l_exclusion_est_signale_comme_tel(caplog):
    """Emptying a cell by exclusion, whereas the whole pool filled it."""
    pool = _pool(40)   # no household.id: each persona is a one-person household
    cible = "Toulouse × sans voiture"
    a_exclure = {f"p:{r['person_id']}" for r in pool
                 if r["identity"]["traits_json"]["residence_zone"] == "Toulouse"
                 and r["identity"]["traits_json"]["number_of_cars"] == 0}
    assert a_exclure, "the synthetic pool must populate the targeted cell"
    _, journal = seal.select(pool, 150, exclure_menages=a_exclure)
    assert journal["deficits_imputables_a_l_exclusion"], (
        "the cell was served by the whole pool: the deficit comes from the exclusion")
    assert cible in journal["deficits_imputables_a_l_exclusion"]


def test_R7_sans_exclusion_aucun_deficit_n_est_impute_a_l_exclusion():
    _, journal = seal.select(_pool_mixte(), 150)
    assert journal["deficits_imputables_a_l_exclusion"] == {}


# ── R8: the salt is fixed — with an empty exclusion, v6 comes back identical ───

@besoin_vivier
def test_R8_a_exclusion_vide_le_tirage_redonne_la_cohorte_v6():
    """The proof that the gap between the two cohorts comes from the EXCLUSION and nothing else.

    Slow (≈ 20 s: loading the 26 MB pool, whole programme, descent). That is the price
    of a non-regression over the whole draw chain.
    """
    from scripts.panel import control_population as ctl
    records = ctl.load_population(VIVIER)
    seal.ensure_residence_zone(records)
    retenus, journal = seal.select(records, 1000)
    assert journal["version"] == seal.SELECTION_RULE
    attendus = json.loads((COHORTE_1 / "selection.json").read_text(encoding="utf-8"))
    assert [str(r["person_id"]) for r in retenus] == attendus["person_ids"]


# ── R9: a cohort modified since its sealing is not excluded ───────────────────

@besoin_cohortes
def test_R9_une_cohorte_dont_le_sha256_a_bouge_est_refusee(tmp_path):
    faux = tmp_path / "cohorte_trafiquee"
    faux.mkdir()
    (faux / "MANIFEST.yaml").write_text(
        (COHORTE_1 / "MANIFEST.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    records = json.loads((COHORTE_1 / "population.json").read_text(encoding="utf-8"))
    records.pop()  # one persona fewer: the file is no longer the one the manifest seals
    (faux / "population.json").write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="modified since its sealing"):
        seal.charger_exclusions([faux])


def test_R9_une_cohorte_incomplete_est_refusee(tmp_path):
    vide = tmp_path / "sans_rien"
    vide.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        seal.charger_exclusions([vide])


# ── R10: the disjunction is checked on the result ─────────────────────────────

def test_R10_la_garde_leve_quand_un_menage_exclu_figure_dans_le_resultat():
    with pytest.raises(ValueError, match="disjointness broken"):
        seal.verifier_disjonction({"h1", "h2", "h3"}, {"h3", "h9"})


def test_R10_la_garde_laisse_passer_un_resultat_disjoint():
    seal.verifier_disjonction({"h1", "h2"}, {"h8", "h9"})
    seal.verifier_disjonction({"h1", "h2"}, set())


def test_R10_select_appelle_la_garde_sur_son_resultat(monkeypatch):
    """The guard is useless if the production path forgets it: we check the call."""
    vus: list[tuple[set, set]] = []
    monkeypatch.setattr(seal, "verifier_disjonction",
                        lambda retenus, exclus: vus.append((set(retenus), set(exclus))))
    seal.select(_pool_mixte(), 150, exclure_menages={"h0"})
    assert len(vus) == 1, "select must check its result exactly once"
    retenus, exclus = vus[0]
    assert exclus == {"h0"}
    assert retenus, "the guard receives the retained households, not an empty set"
