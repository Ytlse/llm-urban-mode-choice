"""Exact-prompt replay between the two arms of a memory A/B — orchestration side (2026-09-25).

The gateway serves the control the treated arm's response again when the prompt is word for word
identical (`packages/llm_gateway/tests/integration/test_rejeu_ab.py`). Here: the space is the same
for both arms, a treated arm that restarts from scratch does not replay a previous attempt, the
setting reaches the container and its identity, and the summary flags a call paid before the event.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for chemin in (REPO_ROOT, REPO_ROOT / "services" / "llm-agents"):
    if str(chemin) not in sys.path:
        sys.path.insert(0, str(chemin))

from experiences import rejeu_ab  # noqa: E402
from scripts.experiment import orchestrateur_memoire as O  # noqa: E402
from scripts.experiment import run_sequential_cohort as C  # noqa: E402

NOM = "exp_mem_presse_a13_punaises_gem31flite_pop_2_foyers_12j_foyer"


def _echange(provider: str, jour: str, categorie: str = "itinary_multi_agent", agent: str = "1") -> dict:
    return {"provider": provider, "sim_day": jour, "category": categorie,
            "response": [{"agent_id": agent}], "messages": []}


# ── The space ────────────────────────────────────────────────────────────────────────────


def test_l_espace_est_le_nom_de_l_experience_et_le_meme_pour_les_deux_bras():
    assert O.espace_rejeu(NOM, {"rejeu_ab": True}) == NOM
    assert O.espace_rejeu(NOM, {}) is None, "a declaration without the key runs without replay"


def test_une_experience_neuve_rejoue_par_defaut():
    from experiences import memoire

    assert memoire.defauts()["rejeu_ab"] is True


def test_le_reglage_est_verifie_dans_l_identite_du_run():
    """A container that did not receive it would pay for the whole control without saying so."""
    assert C.reglages_attendus({"REJEU_AB": NOM})["rejeu_ab"] == NOM
    from urban_mobility_agents.utils import identite_run as I

    assert "rejeu_ab" not in I.LIBELLES, "outside the resume comparison: an old run stays resumable"


def test_le_client_de_l_agent_pose_l_espace():
    source = (REPO_ROOT / "services/llm-agents/urban_mobility_agents/agents/llm_agent.py").read_text()
    assert "espace_rejeu=settings.llm.rejeu_ab or None" in source


# ── The treated arm that restarts from scratch ───────────────────────────────────────────


def test_un_magasin_perime_est_mis_de_cote_pas_efface(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "RACINE_REJEU", tmp_path)
    (tmp_path / NOM).mkdir()
    (tmp_path / NOM / "abc.json").write_text("{}")
    cible = O.mettre_de_cote_magasin(NOM)
    assert cible is not None and (cible / "abc.json").is_file(), "the paid responses are kept"
    assert not (tmp_path / NOM).exists()
    assert O.mettre_de_cote_magasin(NOM) is None, "nothing to do without a store"


# ── The summary ──────────────────────────────────────────────────────────────────────────


def test_un_temoin_entierement_rejoue_avant_l_evenement_est_conforme():
    echanges = [_echange("rejeu_ab:k1", "2026-03-20"), _echange("rejeu_ab:k1", "2026-03-24"),
                _echange("k1", "2026-03-25"), _echange("rejeu_ab:k2", "2026-03-26", "stm_reflection")]
    b = rejeu_ab.bilan(echanges, "2026-03-25")
    assert b["payes_avant_evenement"] == 0
    assert (b["servis"], b["payes"]) == (3, 1)
    assert b["par_categorie"]["stm_reflection"] == {"servis": 1, "payes": 0}


def test_un_appel_paye_avant_l_evenement_est_denonce():
    b = rejeu_ab.bilan([_echange("k1", "2026-03-21", agent="643030")], "2026-03-25")
    assert b["payes_avant_evenement"] == 1
    assert b["premiers_payes_avant"][0] == {"categorie": "itinary_multi_agent", "jour": "2026-03-21",
                                             "agents": ["643030"]}


def test_un_echange_sans_jour_n_est_pas_accuse():
    b = rejeu_ab.bilan([{"provider": "k1", "sim_day": None, "category": "x", "response": []}], "2026-03-25")
    assert b["payes_avant_evenement"] == 0 and b["payes"] == 1


def test_le_prefixe_compare_les_deplacements_pas_seulement_les_appels(tmp_path):
    champs = ["ID Personne", "ID Activité", "Heure de départ", "Mode de transport Choisi"]
    for bras, mode in (("traite", "Vélo"), ("temoin", "Voiture Privée")):
        dossier = tmp_path / bras
        dossier.mkdir()
        with (dossier / "moves.csv").open("w", newline="", encoding="utf-8") as flux:
            writer = csv.DictWriter(flux, fieldnames=champs)
            writer.writeheader()
            writer.writerow({"ID Personne": "1", "ID Activité": "2",
                             "Heure de départ": "2026-03-20T08:00:00", "Mode de transport Choisi": mode})
    bilan = O.verifier_prefixe_deplacements(tmp_path, "2026-03-25")
    assert bilan["conforme"] is False
    assert bilan["cles_divergentes"] == 1


def test_le_prefixe_compare_aussi_le_matin_avant_la_publication(tmp_path):
    champs = ["ID Personne", "ID Activité", "Heure de départ", "Mode de transport Choisi"]
    for bras, mode in (("traite", "Vélo"), ("temoin", "Voiture Privée")):
        dossier = tmp_path / bras
        dossier.mkdir()
        with (dossier / "moves.csv").open("w", newline="", encoding="utf-8") as flux:
            writer = csv.DictWriter(flux, fieldnames=champs)
            writer.writeheader()
            writer.writerow({"ID Personne": "1", "ID Activité": "2",
                             "Heure de départ": "2026-03-25T07:00:00", "Mode de transport Choisi": mode})
    assert O.verifier_prefixe_deplacements(tmp_path, "2026-03-25T07:45:00")["cles_divergentes"] == 1
    assert O.verifier_prefixe_deplacements(tmp_path, "2026-03-25T06:45:00")["cles_divergentes"] == 0


def test_la_date_de_l_evenement_est_la_premiere_injection(tmp_path):
    f = tmp_path / "evenements.jsonl"
    f.write_text("\n".join(json.dumps({"horodatage_simule": h}) for h in
                           ["2026-03-26T07:00:00", "2026-03-25T07:00:00"]))
    assert rejeu_ab.date_premiere_injection(f) == "2026-03-25"
    assert rejeu_ab.date_premiere_injection(tmp_path / "absent.jsonl") is None


def test_borne_stricte_vient_de_l_instant_exact_du_traite(tmp_path):
    dossier = tmp_path / "traite"
    dossier.mkdir()
    (dossier / "evenements.jsonl").write_text(
        json.dumps({"horodatage_simule": "2026-03-27T07:45:00"}) + "\n"
    )
    from datetime import datetime, timezone

    attendu = int(datetime(2026, 3, 27, 7, 45, tzinfo=timezone.utc).timestamp())
    assert O.borne_prefixe_commun(tmp_path) == attendu
    attendus = C.reglages_attendus({"WORLD__PREFIXE_COMMUN": "true",
                                     "REJEU_STRICT_AVANT_TS": str(attendu)})
    assert attendus["prefixe_commun"] is True
    assert attendus["rejeu_strict_avant_ts"] == attendu


def _traite(tmp_path: Path, injection: str, services: list[str] | None, canal: str = "lu") -> None:
    dossier = tmp_path / "traite"
    dossier.mkdir()
    (dossier / "evenements.jsonl").write_text(json.dumps({"horodatage_simule": injection}) + "\n")
    if services is not None:
        (dossier / rejeu_ab.FICHIER_PREMIERS_SERVICES).write_text(
            "".join(json.dumps({"instant_simule": h}) + "\n" for h in services)
        )
    (tmp_path / "experience_memoire.yaml").write_text(f"canal: {canal}\n")


def _ts(*args: int) -> int:
    from datetime import datetime, timezone

    return int(datetime(*args, tzinfo=timezone.utc).timestamp())


def test_la_borne_est_le_premier_prompt_porteur_de_l_article_pas_l_injection(tmp_path):
    """a13 v5, 2026-09-28: the article of 27/03 entered a decision on 26/03 at 07:30.
    The bound at 00:00 on the 27th made the control request a prompt the treated arm never sent."""
    _traite(tmp_path, "2026-03-27T00:00:00", ["2026-03-26T17:00:00", "2026-03-26T07:30:00"])
    assert rejeu_ab.instant_debut_traitement(tmp_path / "traite" / "evenements.jsonl") == (
        "2026-03-26T07:30:00", "service"
    )
    assert O.borne_prefixe_commun(tmp_path) == _ts(2026, 3, 26, 7, 30)


def test_une_injection_plus_precoce_que_tout_service_fait_foi(tmp_path):
    _traite(tmp_path, "2026-03-25T08:00:00", ["2026-03-26T07:30:00"])
    assert O.borne_prefixe_commun(tmp_path) == _ts(2026, 3, 25, 8, 0)


def test_un_article_lu_sans_premiers_services_est_refuse(tmp_path):
    """a13 v6, 2026-09-29: the file had not been brought back into `traite/`, the bound fell back to
    the injection (27/03 00:00) with a WARNING, and the control arm demanded in strict replay the
    reader's 26/03 07:30 decision — a 409 at the same instant on three relaunches. No bound, no arm."""
    _traite(tmp_path, "2026-03-27T00:00:00", None)
    with pytest.raises(O.BorneIncertaine, match="premiers_services.jsonl"):
        O.borne_prefixe_commun(tmp_path)


def test_la_borne_de_la_v6_une_fois_le_fichier_recopie(tmp_path):
    """The real first line of a13 v6's treated arm: the bound is 26/03 07:30, the instant of the 409."""
    _traite(tmp_path, "2026-03-27T00:00:00", None)
    (tmp_path / "traite" / rejeu_ab.FICHIER_PREMIERS_SERVICES).write_text(json.dumps({
        "person_id": "1127260", "evenement_id": "a13_punaises_metro", "jour_service": "2026-03-27",
        "genre": "lectures_servies", "instant_ts": 1774510200,
        "instant_simule": "2026-03-26T07:30:00", "instant_source": "sync",
        "decision_ts": 1774595419, "avant_injection": True,
    }) + "\n")
    assert O.borne_prefixe_commun(tmp_path) == 1774510200


def test_le_lanceur_recopie_les_premiers_services_avec_les_livrables():
    """Upstream cause: `run_sequential_cohort` copied the arm's deliverables without this file, and
    the orchestrator, which reads `traite/`, never saw it (placed by hand for v5 on 28/09 09:30)."""
    import inspect

    bloc = inspect.getsource(C).split("Deliverables copied")[0].rsplit("for fname in [", 1)[1]
    liste = bloc.split("]:")[0]
    assert f'"{rejeu_ab.FICHIER_PREMIERS_SERVICES}"' in liste
    assert '"evenements.jsonl"' in liste


def test_un_choc_vecu_sans_premiers_services_ne_crie_pas(tmp_path, caplog):
    """A lived shock serves no line to the prompt: its timestamp IS its first effect."""
    _traite(tmp_path, "2026-03-27T08:15:00", None, canal="vecu")
    with caplog.at_level("WARNING", logger="orchestrateur_memoire"):
        assert O.borne_prefixe_commun(tmp_path) == _ts(2026, 3, 27, 8, 15)
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_le_controle_lit_le_journal_du_temoin_par_le_manifeste(tmp_path, monkeypatch):
    archive = tmp_path / "archive" / "2026-09-26_09_00"
    archive.mkdir(parents=True)
    (archive / "llm_exchanges.jsonl").write_text(
        "\n".join(json.dumps({**e, "origine": archive.name}, indent=2) for e in
                  [_echange("rejeu_ab:k1", "2026-03-20"), _echange("k1", "2026-03-27")]) + "\n")
    runs = tmp_path / "experiments" / "runs" / f"{NOM}_control"
    runs.mkdir(parents=True)
    (runs / "manifeste.json").write_text(json.dumps([{"branche": "control", "archive": str(archive)}]))
    racine_bras = tmp_path / "exp"
    (racine_bras / "traite").mkdir(parents=True)
    (racine_bras / "traite" / "evenements.jsonl").write_text(json.dumps({"horodatage_simule": "2026-03-25T07:00:00"}))
    monkeypatch.setattr(O, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(O, "RACINE_REJEU", tmp_path / "rejeu_ab")
    b = O.controler_rejeu(NOM, racine_bras)
    assert b["payes_avant_evenement"] == 0 and b["servis"] == 1 and b["payes"] == 1
    assert b["date_evenement"] == "2026-03-25"


def test_les_journaux_du_bras_ecartent_les_autres_clients(tmp_path):
    archive, cible = tmp_path / "archive" / "run_a", tmp_path / "sortie"
    archive.mkdir(parents=True)
    (archive / "llm_exchanges.jsonl").write_text("\n".join(
        json.dumps({**_echange("k", "2026-03-20"), "origine": origine}, indent=2)
        for origine in ("run_a", "run_b")) + "\n")
    (archive / "llm_errors.jsonl").write_text("\n".join(
        json.dumps({"task_id": origine, "origine": origine}) for origine in
        ("run_a", "run_b", None)) + "\n")
    bilan = O.isoler_journaux(archive, cible, "run_a")
    assert bilan == {"echanges": 1, "erreurs": 1, "erreurs_sans_origine": 1}
    assert {e["origine"] for e in rejeu_ab.lire_echanges(cible / "llm_exchanges.jsonl")} == {"run_a"}
    assert [json.loads(l)["origine"] for l in (cible / "llm_errors.jsonl").read_text().splitlines()] == ["run_a"]


def test_sans_journal_le_controle_le_dit(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "REPO_ROOT", tmp_path)
    assert O.controler_rejeu(NOM, tmp_path) is None
