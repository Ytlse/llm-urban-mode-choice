"""Fitness of a model to carry an experiment — hygiene spec §6, proposal P3.

Severity is the subject of these tests. Refusing too broadly blocks legitimate work: the
limits in `providers.yaml` are declarative and sometimes wrong (runs on
`gemini-3.5-flash-lite` succeeded beyond the declared `rpd_limit`). So we only refuse
the impossible, and warn about the slow.
"""

from __future__ import annotations

import pytest

from experiences import aptitude as APT

CHARGE = dict(
    sollicitations=2285, jetons_entree=1289, jetons_sortie=400, max_tokens_demande=4096
)


def _verifier(providers, instances=None, **kw):
    args = {**CHARGE, **kw}
    return APT.verifier(
        modele=kw.pop("modele", "m") if "modele" in kw else "m",
        providers=providers,
        instances=instances if instances is not None else list(providers),
        **{k: v for k, v in args.items() if k != "modele"},
    )


def test_quota_court_meme_tres_long_avertit_sans_refuser():
    """Even for 114 theoretical days, a short quota stays a warning and does not refuse."""
    refus, avert = _verifier({"i": {"rpd_limit": 20, "rpm_limit": 5}})
    assert refus == []
    assert any("plus court que la charge" in m for m in avert)
    assert any("ce n'est pas un refus" in m for m in avert)


def test_quota_juste_court_avertit_sans_refuser():
    """1,000/day for 2,285: two quota windows and a resume, not an impossibility."""
    refus, avert = _verifier({"a": {"rpd_limit": 500}, "b": {"rpd_limit": 500}})
    assert refus == []
    assert any("plus court que la charge" in m for m in avert)
    assert any("ce n'est pas un refus" in m for m in avert)


def test_quota_suffisant_ne_dit_rien():
    refus, avert = _verifier({"i": {"rpd_limit": 5000, "rpm_limit": 60, "tpm_limit": 500000}})
    assert refus == [] and avert == []


def test_plafond_par_requete_refuse():
    """A cap below the need truncates EVERY call: it is structural, not a quota matter."""
    refus, _ = _verifier(
        {"i": {"rpd_limit": 99999, "max_tokens_per_request": 1000}}
    )
    assert refus and "plafonne à 1000 jetons par requête" in refus[0]
    assert "tronqué" in refus[0]


def test_plafond_de_sortie_refuse():
    refus, _ = _verifier({"i": {"rpd_limit": 99999, "max_output_tokens": 2048}})
    assert refus and "plafonne la sortie à 2048" in refus[0]


def test_plafond_par_requete_suffisant_passe():
    refus, _ = _verifier(
        {"i": {"rpd_limit": 99999, "max_tokens_per_request": 8000, "max_output_tokens": 4096}}
    )
    assert refus == []


def test_tpm_bride_le_debit_et_le_dit():
    """rpm_limit 30 announced, but tpm 8,000 ÷ 1,689 tokens = 4.7 real req/min."""
    _, avert = _verifier(
        {"i": {"rpd_limit": 99999, "rpm_limit": 30, "tpm_limit": 8000}}
    )
    joint = " ".join(avert)
    assert "débit bridé par les jetons" in joint
    assert "4.7 requêtes/min" in joint and "rpm_limit 30" in joint


def test_duree_longue_avertit():
    _, avert = _verifier({"i": {"rpd_limit": 99999, "rpm_limit": 5}})
    assert any("durée estimée" in m and "fenêtre de renouvellement" in m for m in avert)


def test_jetons_inconnus_ne_refusent_pas():
    """Refusing on an unknown value would block any never-measured model, so every new model."""
    refus, _ = _verifier(
        {"i": {"rpd_limit": 99999, "max_tokens_per_request": 100}},
        jetons_entree=None,
        jetons_sortie=None,
    )
    assert refus == []


def test_limites_non_declarees_ne_refusent_pas():
    """An instance that declares no limit is not deemed unfit."""
    refus, avert = _verifier({"i": {}})
    assert refus == [] and avert == []


def test_sans_instance_aucun_verdict():
    """No instance serving the model is already refused upstream: do not duplicate."""
    refus, avert = _verifier({"i": {"rpd_limit": 20}}, instances=[])
    assert refus == [] and avert == []


def test_marge_de_quota_appliquee():
    """Consuming the quota to the last token fails on the first retried error."""
    refus, avert = _verifier({"i": {"rpd_limit": 2285}})
    assert refus == []
    assert any("plus court que la charge" in m for m in avert), (
        "a quota of 2285 for 2285 requests must warn: the margin is 5 %"
    )


# ── Thinking depth (2026-09-10) ──────────────────────────────────────────────


def test_reflexion_au_dela_du_plafond_declare_refuse():
    """The provider would trim silently: the fingerprint would carry a budget not applied."""
    refus, _ = _verifier(
        {"i": {"rpd_limit": 99999, "thinking_budget_max": 8192}}, reflexion_demandee=16384
    )
    assert refus and "au-delà du plafond déclaré 8192" in refus[0]
    assert "non appliqué" in refus[0]


def test_reflexion_dans_le_plafond_passe():
    refus, avert = _verifier(
        {"i": {"rpd_limit": 99999, "thinking_budget_max": 8192}}, reflexion_demandee=8192
    )
    assert refus == []
    assert not [m for m in avert if "réflexion" in m]


def test_reflexion_sans_plafond_declare_avertit_sans_refuser():
    """We do not refuse on a missing measure — we say we cannot check."""
    refus, avert = _verifier({"i": {"rpd_limit": 99999}}, reflexion_demandee=4096)
    assert refus == []
    assert any("aucun `thinking_budget_max`" in m for m in avert)


def test_reflexion_absente_ou_nulle_ne_declenche_rien():
    for valeur in (None, 0, -1):
        refus, avert = _verifier({"i": {"rpd_limit": 99999}}, reflexion_demandee=valeur)
        assert refus == []
        assert not [m for m in avert if "réflexion" in m], valeur


def test_niveau_non_accepte_refuse():
    """`minimal` does not exist on gemini-3.7/3.8: every call would end in a 400."""
    refus, _ = _verifier(
        {"i": {"rpd_limit": 99999, "thinking_levels": ["low", "medium", "high"]}},
        niveau_demande="minimal",
    )
    assert refus and "non accepté" in refus[0] and "400" in refus[0]


def test_niveau_accepte_passe():
    refus, avert = _verifier(
        {"i": {"rpd_limit": 99999, "thinking_levels": ["low", "medium", "high"]}},
        niveau_demande="high",
    )
    assert refus == []
    assert not [m for m in avert if "niveau" in m]


def test_niveau_sans_declaration_avertit():
    refus, avert = _verifier({"i": {"rpd_limit": 99999}}, niveau_demande="high")
    assert refus == []
    assert any("aucun `thinking_levels`" in m for m in avert)


def test_niveau_et_budget_ensemble_refuses():
    """The API returns 400 if both coexist — say so at launch, not in flight."""
    refus, _ = _verifier(
        {"i": {"rpd_limit": 99999, "thinking_levels": ["high"]}},
        niveau_demande="high", reflexion_demandee=1024,
    )
    assert refus and "ensemble" in refus[0] and "400" in refus[0]


# ── Quotas are counted in REQUESTS, not in trips (2026-09-22) ────────────────
#
# The gateway merges several agents per call. Comparing an `rpd_limit` with a number of
# trips overestimated the load by a factor of 2 to 8 and raised short-quota warnings on
# arms that fitted easily.


def test_le_quota_se_compare_aux_requetes_pas_aux_deplacements():
    """2,285 trips grouped by 8 fit in 1,000 requests/day: no more warning."""
    providers = {"a": {"rpd_limit": 500}, "b": {"rpd_limit": 500}}
    _, avert_avant = _verifier(providers)
    assert any("plus court que la charge" in m for m in avert_avant), "test safeguard"
    refus, avert = APT.verifier(
        modele="m",
        providers=providers,
        instances=list(providers),
        requetes=286,
        regroupement={"prudent": 8.0, "source": "2 exécutions archivées"},
        **CHARGE,
    )
    assert refus == []
    assert not any("plus court que la charge" in m for m in avert)


def test_lavertissement_nomme_les_deux_unites_et_sa_source():
    refus, avert = APT.verifier(
        modele="m",
        providers={"a": {"rpd_limit": 500}},
        instances=["a"],
        requetes=1143,
        regroupement={"prudent": 2.0, "source": "2 exécutions archivées du même gabarit"},
        **CHARGE,
    )
    (msg,) = [m for m in avert if "plus court que la charge" in m]
    assert "1143 requêtes" in msg and "2285 déplacements" in msg
    assert "2.00 agent(s)/requête" in msg and "exécutions archivées" in msg


def test_sans_regroupement_declare_le_verdict_est_celui_davant():
    """`requetes=None` ⇒ one request per trip: the caution cannot decrease."""
    avant = _verifier({"a": {"rpd_limit": 500}, "b": {"rpd_limit": 500}})
    apres = APT.verifier(
        modele="m",
        providers={"a": {"rpd_limit": 500}, "b": {"rpd_limit": 500}},
        instances=["a", "b"],
        requetes=None,
        **CHARGE,
    )
    assert avant == apres


def _duree_h(providers, *, facteur, deplacements, appels):
    _, avert = APT.verifier(
        modele="m",
        providers=providers,
        instances=list(providers),
        requetes=appels,
        regroupement={"prudent": facteur, "source": "test"},
        **{**CHARGE, "sollicitations": deplacements},
    )
    (msg,) = [m for m in avert if "durée estimée" in m]
    return float(msg.split("~")[1].split(" h")[0])


def test_le_bornage_par_les_jetons_est_insensible_au_regroupement():
    """n·jetons_agent ÷ tpm on both sides: grouping does not gain any TPM.

    The trap this test guards: the tokens of a request are those of ALL its agents.
    Dividing the number of requests without multiplying the tokens per request would have
    divided the duration by the grouping factor — a wrong, and optimistic, duration.
    """
    providers = {"i": {"rpd_limit": 999999, "rpm_limit": 600, "tpm_limit": 4_000}}
    seul = _duree_h(providers, facteur=1.0, deplacements=2288, appels=2288)
    groupe = _duree_h(providers, facteur=8.0, deplacements=2288, appels=286)
    assert seul == pytest.approx(groupe, rel=0.01), (seul, groupe)


def test_le_bornage_par_le_rpm_est_divise_par_le_regroupement():
    """Without a token cap, grouping by 8 does divide the duration by 8."""
    providers = {"i": {"rpd_limit": 999999, "rpm_limit": 1}}
    seul = _duree_h(providers, facteur=1.0, deplacements=22880, appels=22880)
    groupe = _duree_h(providers, facteur=8.0, deplacements=22880, appels=2860)
    assert seul / groupe == pytest.approx(8.0, rel=0.01)


def test_un_agent_seul_trop_gros_reste_un_refus():
    """The per-request cap is judged on the AGENT: a batch too big, the gateway shrinks it."""
    refus, _ = APT.verifier(
        modele="m",
        providers={"i": {"rpd_limit": 99999, "max_tokens_per_request": 1000}},
        instances=["i"],
        requetes=286,
        regroupement={"prudent": 8.0, "source": "test"},
        **CHARGE,
    )
    assert refus and "même sans regroupement" in refus[0]


def test_lavertissement_dit_aussi_ce_que_le_bras_coutera_vraisemblablement():
    """No measure: cautious = trips; without the expected value beside it, we read the worst alone."""
    refus, avert = APT.verifier(
        modele="m",
        providers={"a": {"rpd_limit": 500}},
        instances=["a"],
        requetes=None,
        regroupement={"prudent": 1.0, "attendu": 8.0, "source": "aucune mesure"},
        **CHARGE,
    )
    (msg,) = [m for m in avert if "plus court que la charge" in m]
    assert "attendu plutôt ~286 requêtes" in msg and "aucune mesure ne le garantit" in msg
