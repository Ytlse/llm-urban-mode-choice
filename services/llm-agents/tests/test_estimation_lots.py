"""From trip to request: batching enters the cost estimate.

The defect fixed: `estimer` assumed one provider request per trip, whereas the
gateway merges eight of them. Benchmark from ticket 073 § 4: about 310 requests for a full
arm, i.e. roughly 0.3 day of quota — the old estimate announced 2,500.

What these tests hold, in order of importance:

1. **caution never decreases** — without a comparable archived run, the deciding figure
   becomes exactly the one from before the fix again (one request per trip);
2. the two units are distinct and named in the output;
3. an archived measurement that does not describe the run is discarded, with its reason.
"""

from __future__ import annotations

import json

import pytest
import yaml
from experiences import lots as L

# ── Derived ceiling ──────────────────────────────────────────────────────────

PROVIDERS = {
    "k1": {"rpm_limit": 15, "tpm_limit": 250_000},  # 250000/3000=83, capped by rpm → 15
    "k2": {"rpm_limit": 15, "tpm_limit": 250_000},
    "petite": {"rpm_limit": 60, "tpm_limit": 6_000},  # 6000/3000 = 2
}


def test_plafond_derive_de_la_formule_de_la_passerelle():
    """Without `/health`, the `llm_gateway` formula is replayed on providers.yaml."""
    plafond, source = L.plafond_lot(PROVIDERS, ["k1", "k2"])
    assert plafond == 15
    assert "batch_max_agents=15" in source and "providers.yaml" in source


def test_plafond_pris_sur_la_plus_petite_instance():
    """The decision-maker exhausts the instances IN SERIES: the smallest capacity ends up serving."""
    plafond, _ = L.plafond_lot(PROVIDERS, ["k1", "petite"])
    assert plafond == 2


def test_la_passerelle_lemporte_sur_le_fichier():
    """`/health` publishes the value computed by the serving container: that is the one that counts."""
    plafond, source = L.plafond_lot(
        PROVIDERS, ["k1"], etat_passerelle={"k1": {"batch_max_agents": 7}}
    )
    assert plafond == 7 and "passerelle" in source


def test_le_parallelisme_borne_le_lot_sans_simulateur():
    """8 persons in flight, serial trips: never more than 8 tasks to merge."""
    plafond, source = L.plafond_lot(PROVIDERS, ["k1"], parallelisme=8)
    assert plafond == 8 and "parallélisme 8" in source
    # And it does not cap in simulator mode: GAMA feeds the queue, batches of 15 are observed.
    assert L.plafond_lot(PROVIDERS, ["k1"], parallelisme=None)[0] == 15


def test_sans_instance_aucun_regroupement_suppose():
    plafond, source = L.plafond_lot(PROVIDERS, [])
    assert plafond == 1 and "aucun regroupement" in source


# ── Measurement on archived runs ─────────────────────────────────────────────

SHA = "a" * 64


def _archiver(
    racine,
    nom,
    *,
    sollicitations,
    requetes=None,
    requetes_jour=None,
    parallelisme=8,
    troncature=False,
    cree_le="2026-09-21T09:00:00+00:00",
    cloture_le="2026-09-21T15:00:00+00:00",
    etat="terminee",
    interruptions=None,
    limite_jour=None,
    epuisee=False,
    sha=SHA,
):
    d = racine / nom / "executions" / "2026-09-21_09_00_00"
    d.mkdir(parents=True)
    (d / "execution.yaml").write_text(
        yaml.safe_dump(
            {
                "cree_le": cree_le,
                "experience": {
                    "regroupement": {"parallelisme": parallelisme},
                    "troncature_15": troncature,
                },
                "empreintes": {"gabarit": {"sha256": sha}},
                "interruptions": interruptions or [],
                "cloture": {"le": cloture_le, "etat": etat},
            }
        ),
        encoding="utf-8",
    )
    compteurs = {"sollicitations": sollicitations}
    if requetes is not None:
        compteurs["requetes"] = {"delta": requetes, "fiable": True}
    if requetes_jour is not None:
        compteurs["quota"] = [
            {
                "instance": "k1",
                "requetes_jour": requetes_jour,
                "limite_jour": limite_jour,
                "epuisee": epuisee,
            }
        ]
    (d / "compteurs.json").write_text(json.dumps(compteurs), encoding="utf-8")
    return d


def _mesure(racine, **kw):
    return L.mesures_archivees(
        SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=racine, **kw
    )


# The daily counter is only valid as the cost of a run under three conditions. They are not
# decorative: read without them on run `2026-09-21_17_00_49`, whose key was at 506
# requests for a limit of 500, it announces 2.4 agents/request where the real batching is
# around 8 (ticket 073 § 4). Retries and the day's other runs inflate the
# denominator — and a factor that is too low only costs caution.


def test_execution_non_terminee_ecartee(tmp_path):
    _archiver(tmp_path, "a", sollicitations=2229, requetes_jour=290, etat="epuisee")
    assert _mesure(tmp_path) is None


def test_execution_reprise_ecartee(tmp_path):
    """A resume re-serves archived decisions: its requests are not its own."""
    _archiver(
        tmp_path,
        "a",
        sollicitations=2229,
        requetes_jour=290,
        interruptions=[{"type": "reprise", "instant": "2026-09-21T11:00:00+00:00"}],
    )
    assert _mesure(tmp_path) is None


def test_quota_au_plafond_ecarte(tmp_path):
    """At the ceiling, the provider refused requests that were nevertheless counted."""
    _archiver(
        tmp_path, "a", sollicitations=2229, requetes_jour=506, limite_jour=500
    )
    assert _mesure(tmp_path) is None


def test_ces_garde_fous_ne_touchent_pas_la_mesure_fiable(tmp_path):
    """The run delta does without these conditions: it counts only this run, by construction."""
    _archiver(
        tmp_path,
        "a",
        sollicitations=2229,
        requetes=290,
        etat="epuisee",
        interruptions=[{"type": "reprise"}],
    )
    m = _mesure(tmp_path)
    assert m and m["n"] == 1 and m["fiabilite"] == "mesurée"


def test_mesure_fiable_lue_sur_le_delta_du_run(tmp_path):
    """The delta of the gateway counters measures THIS run: it is the right source."""
    _archiver(tmp_path, "exp_a", sollicitations=2442, requetes=318)
    m = _mesure(tmp_path)
    assert m["n"] == 1
    assert m["mediane"] == pytest.approx(7.68, abs=0.01)
    assert m["fiabilite"] == "mesurée"


def test_compteur_journalier_accepte_mais_signale_comme_minore(tmp_path):
    """Lacking a delta, the DAY counter is used — it aggregates other runs, so it underestimates."""
    _archiver(tmp_path, "exp_a", sollicitations=2442, requetes_jour=318)
    m = _mesure(tmp_path)
    assert m["n"] == 1 and m["fiabilite"].startswith("minorée")


def test_rapport_sous_un_ecarte(tmp_path):
    """Less than one agent per request is impossible: the counter describes more than this run."""
    _archiver(tmp_path, "exp_a", sollicitations=194, requetes_jour=760)
    assert (
        L.mesures_archivees(
            SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=tmp_path
        )
        is None
    )


def test_rapport_au_dela_du_plafond_ecarte(tmp_path):
    """14 agents/request for a ceiling of 8: the counter was reset, not a record."""
    _archiver(tmp_path, "exp_a", sollicitations=1400, requetes_jour=100)
    m = L.mesures_archivees(
        SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=tmp_path
    )
    assert m is None


def test_fenetre_de_quota_traversee_ecarte_le_compteur_journalier(tmp_path):
    """Run straddling midnight: the counter did not see the whole run, the ratio inflates."""
    _archiver(
        tmp_path,
        "exp_a",
        sollicitations=2442,
        requetes_jour=318,
        cree_le="2026-09-20T22:00:00-07:00",
        cloture_le="2026-09-21T03:00:00-07:00",
    )
    assert (
        L.mesures_archivees(
            SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=tmp_path
        )
        is None
    )


def test_la_fenetre_ne_disqualifie_pas_une_mesure_fiable(tmp_path):
    """The delta is computed on both bounds: crossing midnight does not skew it."""
    _archiver(
        tmp_path,
        "exp_a",
        sollicitations=2442,
        requetes=318,
        cree_le="2026-09-20T22:00:00-07:00",
        cloture_le="2026-09-21T03:00:00-07:00",
    )
    m = L.mesures_archivees(
        SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=tmp_path
    )
    assert m and m["n"] == 1


def test_execution_trop_courte_ecartee(tmp_path):
    """A ratio built on two requests measures nothing: ten full batches are required."""
    _archiver(tmp_path, "exp_a", sollicitations=20, requetes=5)
    assert (
        L.mesures_archivees(
            SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=tmp_path
        )
        is None
    )


def test_troncature_differente_non_melangee(tmp_path):
    """Full context and truncated context do not batch alike: their mean describes neither."""
    _archiver(tmp_path, "plein", sollicitations=2400, requetes=800, troncature=False)
    _archiver(tmp_path, "tronque", sollicitations=2442, requetes=318, troncature=True)
    plein = L.mesures_archivees(
        SHA, parallelisme=8, troncature=False, plafond=8, dossier_experiences_=tmp_path
    )
    tronque = L.mesures_archivees(
        SHA, parallelisme=8, troncature=True, plafond=8, dossier_experiences_=tmp_path
    )
    assert plein["n"] == 1 and plein["mediane"] == pytest.approx(3.0, abs=0.01)
    assert tronque["n"] == 1 and tronque["mediane"] == pytest.approx(7.68, abs=0.01)


# ── The three factors, and the one that decides ──────────────────────────────


def test_sans_mesure_le_facteur_prudent_vaut_un(tmp_path):
    """THE non-regression guarantee: no archive ⇒ one request per trip."""
    f = L.facteurs(
        providers=PROVIDERS,
        instances=["k1"],
        parallelisme=8,
        empreinte_gabarit_=SHA,
        dossier_experiences_=tmp_path,
    )
    assert f["prudent"] == 1.0
    assert f["attendu"] == 8.0 and f["plafond"] == 8
    assert L.requetes(2229, f["prudent"]) == 2229


def test_avec_mesures_le_prudent_est_le_minimum(tmp_path):
    """The minimum, not the median: the deciding figure takes the least batched run."""
    _archiver(tmp_path, "a", sollicitations=2400, requetes=800)  # 3.0
    _archiver(tmp_path, "b", sollicitations=2400, requetes=400)  # 6.0
    f = L.facteurs(
        providers=PROVIDERS,
        instances=["k1"],
        parallelisme=8,
        empreinte_gabarit_=SHA,
        dossier_experiences_=tmp_path,
    )
    assert f["prudent"] == pytest.approx(3.0, abs=0.01)
    assert f["attendu"] == pytest.approx(4.5, abs=0.01)  # median of the two
    assert f["plafond"] == 8


def test_ordre_des_trois_chiffres(tmp_path):
    """prudente ≥ attendue ≥ plancher, always: otherwise one of the three lies."""
    _archiver(tmp_path, "a", sollicitations=2442, requetes=318)
    f = L.facteurs(
        providers=PROVIDERS,
        instances=["k1"],
        parallelisme=8,
        empreinte_gabarit_=SHA,
        dossier_experiences_=tmp_path,
    )
    p, a, pl = (
        L.requetes(2229, f["prudent"]),
        L.requetes(2229, f["attendu"]),
        L.requetes(2229, f["plafond"]),
    )
    assert p >= a >= pl


def test_requetes_arrondit_vers_le_haut():
    """A remainder of trips still goes out in a request."""
    assert L.requetes(10, 3) == 4
    assert L.requetes(0, 8) == 0
    # A factor below 1 cannot create more requests than trips.
    assert L.requetes(10, 0.5) == 10


# ── The output of `estimer`: two named units ─────────────────────────────────


class _JeuFactice:
    nom = "jeu_t"

    def couverture(self):
        return {"deplacements_couverts": 2229, "deplacements_attendus": 2300}


def _exp_factice(**kw):
    from types import SimpleNamespace

    from experiences.experience import MODE_SANS_SIMULATEUR

    base = {
        "nom": "exp_t",
        "mode": MODE_SANS_SIMULATEUR,
        "troncature_15": False,
        "regroupement": SimpleNamespace(parallelisme=8),
        "gabarit": SimpleNamespace(categorie="itinary_multi_agent", variante=None),
        "decideur": SimpleNamespace(
            type="passerelle", modele="m", portee=None, parametres={"temperature": 0}
        ),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _moniteur(providers=None, etat=None):
    from experiences.ressources import MoniteurRessources

    providers = providers or {
        "k1": {"default_model": "m", "rpd_limit": 500, "rpm_limit": 15, "tpm_limit": 250_000}
    }
    m = MoniteurRessources(
        list(providers), providers, lecteur=lambda url: etat or {"k1": {"daily_requests": 0}}
    )
    m.rafraichir()
    return m


def test_estimer_distingue_deplacements_et_requetes(tmp_path, monkeypatch):
    """The two units carry different names and different values."""
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    _archiver(tmp_path, "a", sollicitations=2442, requetes=318)
    exp = _exp_factice()
    sha = E.empreinte_gabarit("itinary_multi_agent", None)["sha256"]
    for d in tmp_path.glob("*/executions/*/execution.yaml"):
        d.write_text(
            d.read_text(encoding="utf-8").replace(SHA, sha), encoding="utf-8"
        )
    est = E.estimer(
        exp,
        _JeuFactice(),
        moniteur=_moniteur(),
        jetons={"entree": 576, "sortie": 697, "source": "test"},
    )
    assert est["deplacements"]["valeur"] == 2229
    assert est["deplacements"]["unite"] == "déplacement"
    assert est["sollicitations"]["unite"] == "déplacement"  # same unit, historical name
    assert est["requetes"]["unite"] == "requête fournisseur"
    # 2,229 ÷ 7.68 ≈ 291 requests: the old costing announced 2,229.
    assert est["requetes"]["prudente"] < 400
    assert est["requetes"]["prudente"] >= est["requetes"]["attendue"]
    assert est["requetes"]["attendue"] >= est["requetes"]["plancher"]
    assert est["regroupement"]["decide_par"] == "prudent"
    # Tokens stay per AGENT, and the new field says what the provider sees.
    assert est["jetons"]["par_sollicitation"]["entree"] == 576
    assert est["jetons"]["par_requete"]["entree"] > 576


def test_estimer_sans_archive_ne_devient_pas_plus_optimiste(tmp_path, monkeypatch):
    """Without a comparable measurement, cautious requests == trips: the previous behaviour."""
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    est = E.estimer(
        _exp_factice(),
        _JeuFactice(),
        moniteur=_moniteur(),
        jetons={"entree": 1, "sortie": 1, "source": "test"},
    )
    assert est["requetes"]["prudente"] == est["deplacements"]["valeur"] == 2229
    assert est["quota"]["part"] == pytest.approx(2229 / 500)


def test_estimer_decideur_local_ne_compte_aucune_requete(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    exp = _exp_factice(
        decideur=SimpleNamespace(type="aleatoire", modele=None, portee=None, parametres={})
    )
    est = E.estimer(exp, _JeuFactice())
    assert est["deplacements"]["valeur"] == 2229 and est["requetes"]["valeur"] == 0


# ── The tokens of a log line are those of the BATCH, not of one agent ────────


def test_jetons_ramenes_a_lagent_par_la_taille_du_lot(tmp_path, monkeypatch):
    """4,607 tokens for `batch_..._8` make 576 per agent, not 4,607."""
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    sha = E.empreinte_gabarit("itinary_multi_agent", None)["sha256"]
    d = _archiver(tmp_path, "a", sollicitations=2442, requetes=318, sha=sha)
    # The real file is a sequence of INDENTED JSON objects, not JSONL: a line-by-line
    # parser read nothing from it, silently.
    (d / "llm_exchanges.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"task_id": "batch_7cc7ed2c_8", "tokens_in": 4607, "tokens_out": 5576},
                indent=2,
            )
            for _ in range(3)
        ),
        encoding="utf-8",
    )
    j = E.jetons_mesures(tmp_path, sha)
    assert j["entree"] == 575 and j["sortie"] == 697  # 4607/8 and 5576/8
    assert "ramenée" in j["source"]


def test_ligne_sans_taille_de_lot_ignoree_plutot_que_comptee_pour_un(tmp_path, monkeypatch):
    """Counting it as one agent would be exactly the error being fixed."""
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    sha = E.empreinte_gabarit("itinary_multi_agent", None)["sha256"]
    d = _archiver(tmp_path, "a", sollicitations=2442, requetes=318, sha=sha)
    (d / "llm_exchanges.jsonl").write_text(
        json.dumps({"task_id": "inconnu", "tokens_in": 4607, "tokens_out": 5576}),
        encoding="utf-8",
    )
    assert E.jetons_mesures(tmp_path, sha) is None


def test_repli_sur_le_nombre_de_reponses_dagents(tmp_path, monkeypatch):
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    sha = E.empreinte_gabarit("itinary_multi_agent", None)["sha256"]
    d = _archiver(tmp_path, "a", sollicitations=2442, requetes=318, sha=sha)
    (d / "llm_exchanges.jsonl").write_text(
        json.dumps(
            {"tokens_in": 400, "tokens_out": 200, "response": [{"agent_id": i} for i in range(4)]}
        ),
        encoding="utf-8",
    )
    j = E.jetons_mesures(tmp_path, sha)
    assert j["entree"] == 100 and j["sortie"] == 50


# ── An unclosed set does not produce negative figures (reported on 2026-09-22) ──


class _JeuEnPreparation:
    """Manifest without `attendus`: this is the state of a set still being prepared."""

    nom = "jeu_en_cours"

    def couverture(self):
        return {"deplacements_couverts": 773, "deplacements_attendus": 0}


def test_jeu_non_clos_ne_rend_pas_un_nombre_negatif(tmp_path, monkeypatch):
    """`0 - 773 = -773` was displayed as "déplacements non couverts"."""
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    est = E.estimer(
        _exp_factice(),
        _JeuEnPreparation(),
        moniteur=_moniteur(),
        jetons={"entree": 1, "sortie": 1, "source": "test"},
    )
    assert est["non_couverts"]["valeur"] is None
    assert "non clos" in est["non_couverts"]["source"]
    # And the trip figure presents itself for what it is: a progress count.
    assert est["deplacements"]["jeu_clos"] is False
    assert "NON CLOS" in est["deplacements"]["source"]


def test_jeu_clos_soustrait_normalement(tmp_path, monkeypatch):
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    est = E.estimer(
        _exp_factice(),
        _JeuFactice(),
        moniteur=_moniteur(),
        jetons={"entree": 1, "sortie": 1, "source": "test"},
    )
    assert est["non_couverts"]["valeur"] == 2300 - 2229
    assert est["deplacements"]["jeu_clos"] is True


# ── Runs filed in families (reported on 2026-09-28) ──────────────────────────
#
# Experiments live under `regime_nominal/<jeu>/<exp>/executions/<horodatage>/`. A single-level
# `glob` (`*/executions/*/execution.yaml`) found nothing there: on the tree of
# 2026-09-28, 0 runs read out of 88, and the estimate returned "no measurement", which reads
# as a result. The tests above archive flat; these ones file into a family.

FAMILLE = ("regime_nominal", "jeu_x")


def test_mesure_lit_une_execution_rangee_en_famille(tmp_path):
    _archiver(tmp_path.joinpath(*FAMILLE), "exp_a", sollicitations=2442, requetes=318)
    m = _mesure(tmp_path)
    assert m is not None, "run filed in a family not seen"
    assert m["n"] == 1 and m["mediane"] == pytest.approx(7.68, abs=0.01)


def test_jetons_lus_sur_une_execution_rangee_en_famille(tmp_path, monkeypatch):
    from experiences import experience as E

    monkeypatch.setenv("EXPERIENCES_DIR", str(tmp_path))
    sha = E.empreinte_gabarit("itinary_multi_agent", None)["sha256"]
    d = _archiver(
        tmp_path.joinpath(*FAMILLE), "exp_a", sollicitations=2442, requetes=318, sha=sha
    )
    (d / "llm_exchanges.jsonl").write_text(
        json.dumps(
            {"task_id": "batch_7cc7ed2c_8", "tokens_in": 4607, "tokens_out": 5576}
        ),
        encoding="utf-8",
    )
    j = E.jetons_mesures(tmp_path, sha)
    assert j is not None, "run filed in a family not seen"
    assert j["entree"] == 575 and j["sortie"] == 697
    assert "1 exécutions" in j["source"]


def test_archive_et_system_generated_ignores(tmp_path):
    """Same exclusion as `trouver_dossier_experience`: neither archive nor generated copy."""
    _archiver(tmp_path.joinpath(*FAMILLE), "exp_a", sollicitations=2442, requetes=318)
    # Two decoys with a different ratio (3.0 and 4.0): if they were read, the median would move.
    _archiver(
        tmp_path.joinpath("archive", *FAMILLE),
        "exp_b",
        sollicitations=2400,
        requetes=800,
    )
    _archiver(
        tmp_path.joinpath(*FAMILLE, ".system_generated"),
        "exp_c",
        sollicitations=2400,
        requetes=600,
    )
    m = _mesure(tmp_path)
    assert m is not None, "run filed in a family not seen"
    assert m["n"] == 1 and m["mediane"] == pytest.approx(7.68, abs=0.01)


def test_exclusion_calculee_depuis_la_racine(tmp_path):
    """A root placed under an `archive` folder does not exclude itself.

    Testing `archive` on the absolute path would empty the whole tree as soon as `EXPERIENCES_DIR`
    goes through such a folder: the exclusion applies to the path relative to the root.
    """
    racine = tmp_path / "archive" / "experiences"
    _archiver(racine.joinpath(*FAMILLE), "exp_a", sollicitations=2442, requetes=318)
    m = _mesure(racine)
    assert m is not None, "run filed under a root named `archive` not seen"
    assert m["n"] == 1
