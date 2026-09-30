"""Tests of the two-oracle composite score.

One test per rule of the specification of the two-oracle score, R8 to R20. The distributions
are made by hand: what is checked is **the arithmetic of the score and its
refusals**, never the quality of a model. The two properties that would cost the most if
they gave way silently:

- an unmeasured block must return `null`, not `0.0` — in this project, the absence of measurement
  produces the perfect score;
- a numerator and a denominator measured on two substrates must never make up a
  figure.

Offline: no network call, no LLM, no real parquet.
"""

from __future__ import annotations

import math

import pytest

from scripts.synthesis.bi_oracle import (
    DIST_ORDER,
    EPSILON,
    arc_elasticities,
    block_b,
    block_c1,
    block_c2,
    check_substrate,
    compose,
    entropy_bits,
    jsd_bits,
    kl_bits,
    llm_distributions,
    mass_by_stratum,
)

# The triplet of the spec example: prompt, second oracle, supervised oracle, on an
# offer with three modes. The expected values below are computed by hand.
PROMPT = {"marche": 0.10, "voiture": 0.65, "transports_collectifs": 0.25}
MNL = {"marche": 0.22, "voiture": 0.55, "transports_collectifs": 0.23}
BOOSTER = {"marche": 0.05, "voiture": 0.80, "transports_collectifs": 0.15}
OFFER = sorted(PROMPT)


def entry(distribution: dict, offered=None, dist_cat: str = "2-5km") -> dict:
    return {"p": dict(distribution), "offered": offered or sorted(distribution),
            "dist_cat": dist_cat, "degenere": max(distribution.values()) > 0.999}


def triple(n: int = 1, dist_cat: str = "2-5km"):
    """`n` identical decisions, seen by the three decision-makers."""
    keys = [(f"a{i}", f"t{i}") for i in range(n)]
    return (
        {k: entry(PROMPT, OFFER, dist_cat) for k in keys},
        {k: entry(MNL, OFFER, dist_cat) for k in keys},
        {k: entry(BOOSTER, OFFER, dist_cat) for k in keys},
    )


# ── R8: the formula of block B ───────────────────────────────────────────────

def test_R8_formule_du_bloc_B():
    """On the spec triplet, `s_B` equals the ratio of the divergences, by hand."""
    llm, mnl, lgb = triple(3)
    out = block_b(llm, mnl, lgb)
    assert out["jsd_prompt_mnl_bits_mean"] == pytest.approx(0.019944, abs=1e-6)
    assert out["jsd_booster_mnl_bits_mean"] == pytest.approx(0.064591, abs=1e-6)
    assert out["s_B"] == pytest.approx(30.8769, abs=1e-3)
    # Kullback-Leibler variant: 0.0900 bit vs 0.3148 bit.
    assert out["kl_prompt_mnl_bits_mean"] == pytest.approx(0.090029, abs=1e-6)
    assert out["s_B_kl"] == pytest.approx(28.6012, abs=1e-3)


def test_R8_la_cel_brute_n_est_jamais_publiee_seule():
    """The CEL against a soft target has a floor: it is the excess that is published.

    `CEL(p‖q) = H(q) + KL(q‖p)`: on this triplet, 1.53 bit of which 1.44 is floor. Publishing
    the CEL without its floor would suggest a disagreement of 1.53 bit where it is
    0.09.
    """
    llm, mnl, lgb = triple(1)
    out = block_b(llm, mnl, lgb)
    assert "cel" not in json_keys(out)
    kl, _ = kl_bits(MNL, PROMPT)
    cel = -sum(p * math.log2(PROMPT[m]) for m, p in MNL.items())
    assert cel == pytest.approx(entropy_bits(MNL) + kl, abs=1e-9)
    assert out["entropie_mnl_bits_mean"] == pytest.approx(entropy_bits(MNL), abs=1e-9)


def json_keys(mapping: dict) -> set:
    return {k for k in mapping}


# ── R9: no denominator, no figure ────────────────────────────────────────────

def test_R9_denominateur_absent_non_publie():
    """Two oracles in exact agreement: there is no scale, hence no score."""
    llm, mnl, _ = triple(2)
    out = block_b(llm, mnl, {k: entry(MNL, OFFER) for k in mnl})
    assert out["s_B"] is None
    assert out["mesure"] == "non mesuré"
    assert "pas d'échelle" in out["raison"]


def test_R9_substrat_divergent_refuse():
    """Different `moves.csv` fingerprint: a hot resume keeps the name of the run."""
    mnl_meta = {"run": "experiments/archive/r1", "moves_sha256": "a" * 64,
                "spec_version": 2}
    lgb_meta = {"run": "experiments/archive/r1", "moves_sha256": "b" * 64,
                "spec_version": 2}
    verdict = check_substrate(mnl_meta, lgb_meta, "experiments/archive/r1", "a" * 64)
    assert verdict["ok"] is False
    assert any("empreinte moves.csv divergente" in p for p in verdict["problemes"])


def test_R9_contrats_de_variables_differents_refuses():
    verdict = check_substrate({"spec_version": 2}, {"spec_version": 3}, None, None)
    assert verdict["ok"] is False
    assert any("contrat de variables" in p for p in verdict["problemes"])


# ── R10 and R20: block C1, signs and arbiter ─────────────────────────────────

def _curve(walk_by_stratum: list[float], strata: list[str], n: int = 40) -> dict:
    """Synthetic decisions: walking share imposed per stratum, the rest by car."""
    out = {}
    for index, (cat, walk) in enumerate(zip(strata, walk_by_stratum)):
        for i in range(n):
            out[(f"a{index}_{i}", "t")] = entry(
                {"marche": walk, "voiture": 1 - walk}, ["marche", "voiture"], cat)
    return out


def test_R10_bloc_C_signes():
    """Two contradicted transitions out of ten measured → `s_C1 = 20`."""
    strata = DIST_ORDER[:6]
    # The arbiter is monotone: walking declines as distance grows.
    mnl = _curve([0.80, 0.65, 0.50, 0.38, 0.25, 0.12], strata)
    # The prompt follows it, except between strata 3 and 4 where it goes up: this transition
    # contradicts the arbiter for walking AND for car, i.e. 2 disagreements.
    llm = _curve([0.75, 0.60, 0.45, 0.55, 0.30, 0.15], strata)
    out = block_c1(llm, mnl)
    assert out["n_transitions"] == 10
    assert out["n_desaccords"] == 2
    assert out["s_C1"] == pytest.approx(20.0)
    faulty = {(d["mode"], d["de"], d["vers"]) for d in out["desaccords"]}
    assert faulty == {("marche", strata[2], strata[3]), ("voiture", strata[2], strata[3])}


def test_R20_arbitre_non_monotone_exclu():
    """A transition where the arbiter does not move is neither an agreement nor a disagreement."""
    strata = DIST_ORDER[:2]
    mnl = _curve([0.50, 0.50], strata)          # flat arbiter: no sign to give
    llm = _curve([0.40, 0.60], strata)
    out = block_c1(llm, mnl)
    assert out["s_C1"] is None
    assert out["mesure"] == "non mesuré"
    assert out["ecartees"]["arbitre_sans_signe"] == 4


def test_R20_strate_trop_mince_ecartee_et_comptee():
    strata = DIST_ORDER[:2]
    mnl = _curve([0.80, 0.20], strata, n=3)
    llm = _curve([0.20, 0.80], strata, n=3)
    out = block_c1(llm, mnl, min_stratum=30)
    assert out["s_C1"] is None
    assert out["ecartees"]["effectif_insuffisant"] == 4


# ── R11, R12, R13: composition ───────────────────────────────────────────────

def test_R11_composition_lineaire():
    weights = {"accord_mnl": 0.1, "coherence_mnl": 0.05}
    out = compose(16.16, 30.0, 20.0, weights)
    assert out["S2"] == pytest.approx(16.16 + 0.1 * 30.0 + 0.05 * 20.0)
    # Exact back-application: promoting a weight from 0 is an addition.
    base = compose(16.16, 30.0, 20.0, {"accord_mnl": 0.0, "coherence_mnl": 0.0})
    assert base["S2"] + 0.1 * 30.0 + 0.05 * 20.0 == pytest.approx(out["S2"])


def test_R12_poids_nuls_par_defaut():
    """Without configuration, the two-oracle composite score equals the fidelity one."""
    out = compose(16.16, 899.4, 5.0, {})
    assert out["S2"] == pytest.approx(16.16)
    assert out["poids"] == {"accord_mnl": 0.0, "coherence_mnl": 0.0}


def test_R13_mesure_publiee_malgre_poids_nul():
    out = compose(16.16, 899.4, 5.0, {})
    scores = {t["terme"]: t["score"] for t in out["termes"]}
    assert scores["accord_mnl"] == 899.4 and scores["coherence_mnl"] == 5.0
    llm, mnl, lgb = triple(4)
    detail = block_b(llm, mnl, lgb, n_perimeter=8)
    assert detail["couverture"] == pytest.approx(0.5)
    assert detail["base_couverture"] == 8


def test_R13_un_terme_pese_mais_non_mesure_bloque_le_total():
    """A non-zero weight on a missing term is not replaced by zero."""
    out = compose(16.16, None, 5.0, {"accord_mnl": 0.1})
    assert out["S2"] is None
    assert out["non_mesures_bloquants"] == ["accord_mnl"]


# ── R14: vacuity ≠ perfection ────────────────────────────────────────────────

def test_R14_vacuite_non_nulle():
    out = block_b({}, {}, {})
    assert out["s_B"] is None and out["mesure"] == "non mesuré"
    assert out["s_B"] != 0.0
    assert block_c1({}, {})["s_C1"] is None


def test_R14_une_offre_unique_n_est_pas_un_accord():
    """Forced decision: the three decision-makers agree without anything having been measured."""
    moves = [{"agent_id": "a", "activity_id": "t", "offered": ["voiture"],
              "probas": {"voiture": 100.0}, "dist_cat": "2-5km"}]
    distributions, skipped = llm_distributions(moves)
    assert distributions == {}
    assert skipped == {"offre_unique": 1}


# ── R15: identity of the substrate and of the two oracles ────────────────────

def test_R15_identite_du_substrat():
    mnl_meta = {"run": "r", "moves_sha256": "s", "spec_version": 2,
                "policy_format": "mnl_mode_choice_policy", "policy_sha256": "m" * 64}
    lgb_meta = {"run": "r", "moves_sha256": "s", "spec_version": 2,
                "policy_format": "lightgbm_mode_choice_policy", "policy_sha256": "l" * 64}
    verdict = check_substrate(mnl_meta, lgb_meta, "r", "s")
    assert verdict["ok"] is True
    assert verdict["oracles"]["mnl"]["sha256"] == "m" * 64
    assert verdict["oracles"]["booster"]["sha256"] == "l" * 64
    assert verdict["oracles"]["mnl"]["format"] == "mnl_mode_choice_policy"


def test_R15_parquet_illisible_est_un_probleme_nomme():
    verdict = check_substrate({"error": "parquet absent"}, {"spec_version": 2}, "r", "s")
    assert verdict["ok"] is False
    assert verdict["problemes"][0].startswith("mnl : parquet absent")


# ── R16 and R17: matching and support ────────────────────────────────────────

def test_R16_decisions_non_appariees_comptees():
    llm, mnl, lgb = triple(3)
    mnl.pop(("a0", "t0"))
    out = block_b(llm, mnl, lgb, llm_skipped={"offre_unique": 7})
    assert out["n_decisions"] == 2
    assert out["exclusions"]["sans_mnl"] == 1
    assert out["exclusions"]["prompt::offre_unique"] == 7


def test_R16_offre_divergente_exclue():
    llm, mnl, lgb = triple(1)
    key = ("a0", "t0")
    llm[key] = entry(PROMPT, ["marche", "voiture"])
    out = block_b(llm, mnl, lgb)
    assert out["s_B"] is None
    assert out["exclusions"]["offre_divergente"] == 1


def test_R17_support_de_l_offre():
    """A mode outside the offer does not contribute: it is removed before renormalisation."""
    moves = [{"agent_id": "a", "activity_id": "t",
              "offered": ["marche", "voiture", "autres"],
              "probas": {"marche": 30.0, "voiture": 50.0, "autres": 20.0},
              "dist_cat": "2-5km"}]
    distributions, _ = llm_distributions(moves)
    p = distributions[("a", "t")]["p"]
    assert set(p) == {"marche", "voiture"}
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["marche"] == pytest.approx(30 / 80)


# ── R18: the floor is declared and counted ───────────────────────────────────

def test_R18_epsilon_declare():
    """Zero probability on a mode loaded by the arbiter: finite divergence, and counted."""
    categorical = {"marche": 0.0, "voiture": 1.0, "transports_collectifs": 0.0}
    key = ("a0", "t0")
    llm = {key: entry(categorical, OFFER)}
    mnl = {key: entry(MNL, OFFER)}
    lgb = {key: entry(BOOSTER, OFFER)}
    value, floored = kl_bits(MNL, categorical)
    assert math.isfinite(value) and floored is True
    out = block_b(llm, mnl, lgb)
    assert out["epsilon"] == EPSILON
    assert out["n_plancher_applique"] == 1
    assert out["n_prompt_degenere"] == 1
    # The JSD, for its part, needs no floor: bounded by 1 bit on the same data.
    assert jsd_bits(MNL, categorical) < 1.0


# ── R19: an agreement, never an error ────────────────────────────────────────

def test_R19_libelle_accord():
    llm, mnl, lgb = triple(2)
    out = block_b(llm, mnl, lgb)
    assert "accord" in out["mesure"]
    assert "erreur" not in out["mesure"]
    assert "inter-oracles" in out["mesure"]


# ── Elasticities of block C2 ─────────────────────────────────────────────────

def test_C2_non_mesure_sans_paire_ab():
    out = block_c2(None, None, None, None, None)
    assert out["s_C2"] is None
    assert "rejouées" in out["raison"]
    assert "--ab-variable" in out["comment"]


def test_C2_signes_d_elasticite_compares():
    before = {("a", "t"): entry({"marche": 0.5, "voiture": 0.5}, ["marche", "voiture"])}
    after = {("a", "t"): entry({"marche": 0.7, "voiture": 0.3}, ["marche", "voiture"])}
    mnl_before = before
    mnl_after = {("a", "t"): entry({"marche": 0.3, "voiture": 0.7},
                                   ["marche", "voiture"])}
    elasticities = arc_elasticities(before, after)
    assert elasticities["marche"] == pytest.approx(0.4)
    out = block_c2(before, after, mnl_before, mnl_after, "has_pt_subscription")
    assert out["s_C2"] == pytest.approx(100.0)
    assert out["variable"] == "has_pt_subscription"


def test_les_parts_par_strate_sont_des_masses_de_probabilite():
    strata = mass_by_stratum({("a", "t"): entry(PROMPT, OFFER, "2-5km"),
                              ("b", "t"): entry(BOOSTER, OFFER, "2-5km")})
    assert strata["2-5km"]["n"] == 2
    assert strata["2-5km"]["shares"]["voiture"] == pytest.approx(72.5)
    assert sum(strata["2-5km"]["shares"].values()) == pytest.approx(100.0)


# ── K11: block C with two arbiters ──────────────────────────────
#
# A single arbiter cannot be refuted. With the KLR as second arbiter, a transition
# enters the score only if both behaviour models carry a clear sign and the
# same one; otherwise it leaves the score and is counted, rather than attributing to the prompt a
# disagreement that the models have not settled between themselves.

def test_K11_sans_klr_le_regime_a_un_arbitre_est_inchange():
    """The figure from before the second arbiter does not move, and the output states its regime."""
    strata = DIST_ORDER[:6]
    mnl = _curve([0.80, 0.65, 0.50, 0.38, 0.25, 0.12], strata)
    llm = _curve([0.75, 0.60, 0.45, 0.55, 0.30, 0.15], strata)
    out = block_c1(llm, mnl)
    assert out["s_C1"] == pytest.approx(20.0)
    assert out["regime"] == "un seul arbitre (logit)"
    assert "arbitres_en_desaccord" not in out["ecartees"]
    assert all("delta_klr_pt" not in t for t in out["transitions"])


def test_K11_les_deux_arbitres_d_accord_la_transition_compte():
    strata = DIST_ORDER[:6]
    mnl = _curve([0.80, 0.65, 0.50, 0.38, 0.25, 0.12], strata)
    klr = _curve([0.78, 0.66, 0.52, 0.40, 0.22, 0.10], strata)   # same direction everywhere
    llm = _curve([0.75, 0.60, 0.45, 0.55, 0.30, 0.15], strata)
    out = block_c1(llm, mnl, klr)
    assert out["n_transitions"] == 10
    assert out["s_C1"] == pytest.approx(20.0)
    assert out["regime"].startswith("deux arbitres")
    assert out["ecartees"]["arbitres_en_desaccord"] == 0
    assert all("delta_klr_pt" in t for t in out["transitions"])


def test_K11_arbitres_en_desaccord_la_transition_sort_du_score():
    """The KLR goes up where the logit goes down: the test stays silent on this transition."""
    strata = DIST_ORDER[:3]
    mnl = _curve([0.80, 0.50, 0.20], strata)
    klr = _curve([0.80, 0.90, 0.20], strata)    # opposite direction on the 1st transition
    llm = _curve([0.70, 0.40, 0.10], strata)
    out = block_c1(llm, mnl, klr)
    # 2 transitions × 2 modes = 4 pairs; the first transition leaves (2 pairs).
    assert out["ecartees"]["arbitres_en_desaccord"] == 2
    assert out["n_transitions"] == 2
    assert {(t["de"], t["vers"]) for t in out["transitions"]} == {(strata[1], strata[2])}


def test_K11_second_arbitre_sans_signe_ecarte_et_compte():
    """The flat second arbiter is not agreement: the transition is discarded, not attributed."""
    strata = DIST_ORDER[:2]
    mnl = _curve([0.80, 0.20], strata)
    klr = _curve([0.50, 0.50], strata)          # no sign to give
    llm = _curve([0.20, 0.80], strata)
    out = block_c1(llm, mnl, klr)
    assert out["s_C1"] is None
    assert out["mesure"] == "non mesuré"
    # The two modes that the curve moves (walking, car) reach the second
    # arbiter and stop there; the two others (bike, public transport) are flat
    # at the FIRST arbiter and leave before. The two counters are distinct because
    # they do not say the same thing about what the test could not measure.
    assert out["ecartees"]["second_arbitre_sans_signe"] == 2
    assert out["ecartees"]["arbitre_sans_signe"] == 2
    assert "deux arbitres" in out["regime"]


def test_K11_la_troisieme_colonne_est_le_meme_calcul():
    """`block_b(arbitre="klr")` returns the same quantities, named after their arbiter."""
    llm, mnl, lgb = triple(40)
    contre_mnl = block_b(llm, mnl, lgb)
    contre_klr = block_b(llm, mnl, lgb, arbitre="klr")
    assert contre_klr["s_B"] == pytest.approx(contre_mnl["s_B"])
    assert contre_klr["arbitre"] == "klr"
    assert "jsd_prompt_klr_bits_mean" in contre_klr
    # No key must announce "mnl" in an output measured against the KLR: a
    # wrong label goes unnoticed, which is what makes it worse than a missing label.
    assert not [k for k in contre_klr if k.endswith("_mnl_bits_mean")]
    assert "sans_klr" in contre_klr["exclusions"]


def test_K11_substrat_du_troisieme_oracle_verifie_a_part():
    """A KLR outside the substrate deprives block C of its second arbiter, not block B of s_B."""
    bon = {"run": "experiments/r1", "moves_sha256": "abc", "spec_version": 2,
           "policy_format": "mnl_mode_choice_policy"}
    verdict = check_substrate(bon, dict(bon), "experiments/r1", "abc",
                              {"run": "experiments/AUTRE", "moves_sha256": "abc",
                               "spec_version": 2})
    assert verdict["ok"] is True                     # the first two remain measurable
    assert verdict["klr_ok"] is False
    assert any("klr" in p for p in verdict["problemes_klr"])


def test_K11_klr_absent_est_un_regime_declare_pas_un_zero():
    verdict = check_substrate({"run": "r", "moves_sha256": "a", "spec_version": 2},
                              {"run": "r", "moves_sha256": "a", "spec_version": 2},
                              "r", "a", {"error": "parquet absent : …"})
    assert verdict["klr_ok"] is False
    assert verdict["problemes_klr"] == ["klr : parquet absent : …"]
