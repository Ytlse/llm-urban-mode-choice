"""Measures per simulated day.

Families B (mode choice), C (habit), D (memory), E (shock), and the "writing" part
of F (continuity).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts.analysis.mesures import calculer
from scripts.analysis.mesures.calcul import journee_du_moment, journees_vecues
from scripts.analysis.mesures.ecriture import TABLES, ecrire, mesurer_et_ecrire

COLONNES = [
    "ID Personne", "ID Activité", "Temps simulé", "Heure de départ", "Heure de calcul",
    "Mode de transport Choisi", "Options présentées", "Motifs de déplacement",
]


def _t(personne, jour, heure, mode, options=3, activite="act-travail", motif="work"):
    return {
        "ID Personne": personne,
        "ID Activité": activite,
        "Temps simulé": f"{jour} {heure}",
        "Heure de départ": f"{jour} {heure}",
        "Heure de calcul": f"2026-09-16T10:00:00+00:00",
        "Mode de transport Choisi": mode,
        "Options présentées": str(options),
        "Motifs de déplacement": motif,
    }


def _run(tmp_path, trajets, *, rappels=(), operations=None, chocs=(), points=None) -> Path:
    """A minimal run directory, with only the requested sources."""
    racine = tmp_path / "run"
    racine.mkdir(parents=True, exist_ok=True)
    with (racine / "moves.csv").open("w", newline="", encoding="utf-8") as flux:
        ecrivain = csv.DictWriter(flux, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(trajets)
    if rappels:
        (racine / "trace_rappel.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rappels), encoding="utf-8")
    if operations is not None:
        (racine / "operations_concept.jsonl").write_text(
            "\n".join(json.dumps(o) for o in operations), encoding="utf-8")
    if chocs:
        (racine / "chocs.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chocs), encoding="utf-8")
    for numero, (horodatage, entrees) in (points or {}).items():
        _point(racine, numero, horodatage, entrees)
    return racine


def _point(racine: Path, numero: int, horodatage: str, entrees: dict[str, list[dict]]) -> None:
    dossier = racine / "checkpoints_memoire" / f"jour_{numero:03d}"
    (dossier).mkdir(parents=True, exist_ok=True)
    (dossier / "reprise.json").write_text(json.dumps({
        "jour_simule": numero, "horodatage_simule": horodatage,
        "timestamp_simule": 0, "ancre_run": 0,
    }), encoding="utf-8")
    metadonnees = dossier / "long_term_memory" / "user_metadata" / "shard_0"
    metadonnees.mkdir(parents=True, exist_ok=True)
    for agent, liste in entrees.items():
        (metadonnees / f"{agent}.json").write_text(
            json.dumps({"person_id": agent, "entries": liste}), encoding="utf-8")


def _souvenir(doc_id, type_souvenir, force, contenu="peu importe"):
    return {"doc_id": doc_id, "memory_type": type_souvenir, "force": force,
            "content": contenu, "timestamp": "2026-03-16T05:00:00"}


def _echanges(racine: Path, echanges) -> None:
    """`llm_exchanges.jsonl`: concatenated indented JSON objects, as the worker writes them."""
    (racine / "llm_exchanges.jsonl").write_text(
        "\n\n".join(json.dumps(e, indent=2) for e in echanges), encoding="utf-8")


def _prompt(agent, jour, depart, historique):
    """A decision prompt in the format of the `itinary_multi_agent` template."""
    lignes = [f"--- agent_id={agent} | Destination: work (somewhere) | Departure: {depart} ---",
              "**Trip options** (2 options, indices 0 to 1):",
              "- [0] foot: Estimated duration: 14 minutes.",
              "- [1] car: Estimated duration: 3 minutes.", ""]
    if historique:
        lignes += ["**History:**", *(f"- {h}" for h in historique), ""]
    lignes += ["", "Reply with the final JSON object containing the recommendations."]
    return {"category": "itinary_multi_agent", "sim_day": jour,
            "messages": [{"role": "user", "content": "\n".join(lignes)}]}


def _ligne(mesures, agent, jour, famille="choix_modal"):
    return next(l for l in getattr(mesures, famille)
                if l.person_id == agent and l.jour_simule == jour)


# ── B — Mode choice ────────────────────────────────────────────────────────────────────

class TestChoixModal:
    def test_B1_les_parts_modales_somment_a_un(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="a"),
            _t("1", "2026-03-16", "12:00:00", "Marche", activite="b"),
            _t("1", "2026-03-16", "18:00:00", "Marche", activite="c"),
        ])
        ligne = _ligne(calculer(run), "1", 1)
        assert sum(ligne.parts.values()) == pytest.approx(1.0)
        assert ligne.parts["walking"] == pytest.approx(2 / 3)

    def test_B2_la_part_decidee_compte_les_trajets_a_deux_options_ou_plus(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", options=4, activite="a"),
            _t("1", "2026-03-16", "12:00:00", "Voiture Privée", options=1, activite="b"),
        ])
        ligne = _ligne(calculer(run), "1", 1)
        assert (ligne.trajets, ligne.trajets_decides, ligne.part_decidee) == (2, 1, 0.5)

    def test_B3_une_journee_entierement_mono_option_donne_zero_et_non_vide(self, tmp_path):
        """0.0 is a MEASURE here: it was offered a single route every time."""
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", options=1, activite="a"),
            _t("1", "2026-03-16", "12:00:00", "Voiture Privée", options=1, activite="b"),
        ])
        ligne = _ligne(calculer(run), "1", 1)
        assert ligne.part_decidee == 0.0
        assert ligne.part_decidee is not None

    def test_B4_un_agent_sans_trajet_ce_jour_la_n_a_pas_de_ligne(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée"),
            _t("2", "2026-03-17", "08:00:00", "Marche"),
        ])
        mesures = calculer(run)
        assert {(l.person_id, l.jour_simule) for l in mesures.choix_modal} == {("1", 1), ("2", 2)}

    def test_B5_les_trajets_rejoues_ne_comptent_qu_une_fois(self, tmp_path):
        ligne = _t("1", "2026-03-16", "08:00:00", "Voiture Privée")
        rejouee = dict(ligne, **{"Heure de calcul": "2026-09-16T18:00:00+00:00",
                                 "Mode de transport Choisi": "Marche"})
        run = _run(tmp_path, [ligne, rejouee])
        mesures = calculer(run)
        assert mesures.trajets_rejoues == 1
        assert _ligne(mesures, "1", 1).trajets == 1

    def test_B6_chaque_ligne_porte_le_jour_simule_et_la_date(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-18", "08:00:00", "Marche")])
        ligne = calculer(run).choix_modal[0]
        assert (ligne.jour_simule, ligne.date_simulee) == (1, "2026-03-18")

    def test_B7_un_mode_hors_vocabulaire_est_compte_a_part(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Marche", activite="a"),
            _t("1", "2026-03-16", "12:00:00", "Téléportation", activite="b"),
        ])
        ligne = _ligne(calculer(run), "1", 1)
        assert ligne.trajets_mode_inconnu == 1
        assert sum(ligne.parts.values()) == pytest.approx(0.5)

    def test_B8_un_depart_a_minuit_trente_compte_dans_la_journee_de_la_veille(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "20:00:00", "Marche", activite="a"),
            _t("1", "2026-03-17", "00:30:00", "Voiture Privée", activite="b"),
        ])
        mesures = calculer(run)
        assert len(mesures.journees) == 1
        assert _ligne(mesures, "1", 1).trajets == 2

    def test_B8bis_un_depart_a_quatre_heures_ouvre_la_journee_suivante(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "20:00:00", "Marche", activite="a"),
            _t("1", "2026-03-17", "04:00:00", "Voiture Privée", activite="b"),
        ])
        assert [j.date for j in calculer(run).journees] == ["2026-03-16", "2026-03-17"]

    def test_B9_le_numero_de_jour_se_compte_depuis_la_premiere_journee_vecue(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Marche"),
            _t("1", "2026-03-20", "08:00:00", "Marche"),
        ])
        assert [(j.index, j.date) for j in calculer(run).journees] == [
            (1, "2026-03-16"), (5, "2026-03-20")]


# ── C — Habit and break ────────────────────────────────────────────────────────────────

def _semaine(agent, jours_et_modes, activite="act-travail"):
    return [_t(agent, jour, "08:00:00", mode, activite=activite)
            for jour, mode in jours_et_modes]


class TestHabitude:
    def test_C1_meme_mode_que_le_jour_vecu_precedent(self, tmp_path):
        run = _run(tmp_path, _semaine("1", [("2026-03-16", "Voiture Privée"),
                                            ("2026-03-17", "Voiture Privée")]))
        ligne = _ligne(calculer(run), "1", 2, "habitudes")
        assert (ligne.mode_veille, ligne.reprise_veille) == ("car", True)

    def test_C2_mode_different_de_la_veille(self, tmp_path):
        run = _run(tmp_path, _semaine("1", [("2026-03-16", "Voiture Privée"),
                                            ("2026-03-17", "Marche")]))
        assert _ligne(calculer(run), "1", 2, "habitudes").reprise_veille is False

    def test_C2bis_un_week_end_saute_n_est_pas_une_veille(self, tmp_path):
        """Monday is compared with Friday: nothing was lived on Saturday or Sunday."""
        run = _run(tmp_path, _semaine("1", [("2026-03-20", "Voiture Privée"),   # Friday
                                            ("2026-03-23", "Voiture Privée")]))  # Monday
        ligne = _ligne(calculer(run), "1", 4, "habitudes")
        assert (ligne.mode_veille, ligne.reprise_veille) == ("car", True)

    def test_C3_activite_absente_du_jour_precedent_reste_vide(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="travail"),
            _t("1", "2026-03-17", "08:00:00", "Marche", activite="loisir"),
        ])
        ligne = _ligne(calculer(run), "1", 2, "habitudes")
        assert ligne.reprise_veille is None and ligne.mode_veille is None

    def test_C4_la_conformite_suit_le_mode_majoritaire_de_la_fenetre(self, tmp_path):
        run = _run(tmp_path, _semaine("1", [
            ("2026-03-16", "Voiture Privée"), ("2026-03-17", "Voiture Privée"),
            ("2026-03-18", "Marche"), ("2026-03-19", "Voiture Privée"),
        ]))
        ligne = _ligne(calculer(run), "1", 4, "habitudes")
        assert (ligne.mode_habituel, ligne.conforme_habitude) == ("car", True)

    def test_C5_aucune_observation_anterieure_laisse_vide(self, tmp_path):
        run = _run(tmp_path, _semaine("1", [("2026-03-16", "Voiture Privée")]))
        ligne = _ligne(calculer(run), "1", 1, "habitudes")
        assert ligne.mode_habituel is None and ligne.conforme_habitude is None

    def test_C6_un_ex_aequo_ne_fabrique_pas_une_habitude(self, tmp_path):
        run = _run(tmp_path, _semaine("1", [
            ("2026-03-16", "Voiture Privée"), ("2026-03-17", "Marche"),
            ("2026-03-18", "Marche"), ("2026-03-19", "Voiture Privée"),
            ("2026-03-20", "Voiture Privée"),
        ]))
        ligne = _ligne(calculer(run), "1", 5, "habitudes")
        assert ligne.observations_fenetre == 4
        assert ligne.mode_habituel is None and ligne.conforme_habitude is None

    def test_C7_la_fenetre_porte_sur_les_observations_et_saute_les_trous(self, tmp_path):
        """Five OBSERVED days, not five simulated days: gaps do not count."""
        jours = ["2026-03-16", "2026-03-19", "2026-03-20", "2026-03-22", "2026-03-24"]
        trajets = _semaine("1", [(j, "Voiture Privée") for j in jours])
        trajets += _semaine("1", [("2026-03-27", "Marche")])
        run = _run(tmp_path, trajets)
        ligne = _ligne(calculer(run), "1", 12, "habitudes")
        assert ligne.observations_fenetre == 5
        assert (ligne.mode_habituel, ligne.conforme_habitude) == ("car", False)

    def test_C8_moins_de_cinq_observations_la_fenetre_porte_sur_ce_qui_existe(self, tmp_path):
        run = _run(tmp_path, _semaine("1", [("2026-03-16", "Marche"),
                                            ("2026-03-17", "Marche")]))
        assert _ligne(calculer(run), "1", 2, "habitudes").observations_fenetre == 1

    def test_C9_deux_activites_basculent_independamment(self, tmp_path):
        trajets = []
        for jour, travail, loisir in [("2026-03-16", "Voiture Privée", "Marche"),
                                      ("2026-03-17", "Voiture Privée", "Vélo")]:
            trajets.append(_t("1", jour, "08:00:00", travail, activite="travail"))
            trajets.append(_t("1", jour, "19:00:00", loisir, activite="loisir", motif="leisure"))
        lignes = {l.activite: l for l in calculer(_run(tmp_path, trajets)).habitudes
                  if l.jour_simule == 2}
        assert lignes["travail"].reprise_veille is True
        assert lignes["loisir"].reprise_veille is False

    def test_C10_plusieurs_trajets_d_une_activite_le_meme_jour(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="travail"),
            _t("1", "2026-03-16", "14:00:00", "Marche", activite="travail"),
        ])
        ligne = _ligne(calculer(run), "1", 1, "habitudes")
        assert (ligne.mode, ligne.occurrences) == ("car", 2)


# ── D — Memory ─────────────────────────────────────────────────────────────────────────

class TestMemoire:
    _instant = [1773637200]

    def _rappel(self, agent, jour, candidats, servis=(), instant=None):
        self._instant[0] += 60
        return {"sim_ts": instant or self._instant[0], "sim_day": jour, "person_id": agent,
                "candidats": candidats, "servis": [{"doc_id": d} for d in servis]}

    def test_D1_D2_le_vivier_est_resume_et_un_vivier_vide_compte(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], rappels=[
            self._rappel("1", "2026-03-16", 0),
            self._rappel("1", "2026-03-16", 4),
            self._rappel("1", "2026-03-16", 8),
        ])
        ligne = _ligne(calculer(run), "1", 1, "memoire")
        assert (ligne.rappels, ligne.vivier_min, ligne.vivier_median, ligne.vivier_max) == (
            3, 0, 4, 8)

    def test_D2bis_un_rappel_RETRACE_pendant_le_rejeu_ne_compte_qu_une_fois(self, tmp_path):
        """The recall trace is not frozen during replay (measured on 2026-09-16).

        Same key, two lines: the one from the live run — pool at 0 on the first day — then the
        replay's, which sees memory frozen at 22. Without deduplication, the pool growth, which
        is the very object of the measure, is drowned under the frozen state.
        """
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], rappels=[
            self._rappel("1", "2026-03-16", 0, instant=1773637201),
            self._rappel("1", "2026-03-16", 22, instant=1773637201),
        ])
        mesures = calculer(run)
        ligne = _ligne(mesures, "1", 1, "memoire")
        assert (ligne.rappels, ligne.vivier_max) == (1, 0)
        assert mesures.rappels_rejoues == 1

    def test_D3_les_souvenirs_servis_se_comptent_a_part_du_vivier(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], rappels=[
            self._rappel("1", "2026-03-16", 9, servis=("a", "b", "c")),
        ])
        assert _ligne(calculer(run), "1", 1, "memoire").souvenirs_servis == 3

    def test_D4_les_quatre_operations_de_concept_sont_comptees(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], operations=[
            {"sim_day": "2026-03-16", "person_id": "1", "operation": "créé"},
            {"sim_day": "2026-03-16", "person_id": "1", "operation": "confirmé"},
            {"sim_day": "2026-03-16", "person_id": "1", "operation": "confirmé"},
            {"sim_day": "2026-03-16", "person_id": "1", "operation": "contredit"},
        ])
        ligne = _ligne(calculer(run), "1", 1, "memoire")
        assert ligne.operations == {"créé": 1, "confirmé": 2, "précisé": 0, "contredit": 1}

    def test_D6_une_operation_hors_vocabulaire_est_comptee_a_part(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], operations=[
            {"sim_day": "2026-03-16", "person_id": "1", "operation": "fusionné"},
        ])
        ligne = _ligne(calculer(run), "1", 1, "memoire")
        assert ligne.operations_hors_vocabulaire == 1
        assert sum(v or 0 for v in ligne.operations.values()) == 0

    def test_D7_trace_eteinte_les_colonnes_sont_vides_et_non_a_zero(self, tmp_path):
        """"No contradiction" and "we do not measure" must not be quoted the same way."""
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")])
        ligne = _ligne(calculer(run), "1", 1, "memoire")
        assert all(valeur is None for valeur in ligne.operations.values())

    def test_D8_la_duree_de_vie_vient_du_point_qui_clot_la_journee(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], points={
            2: ("2026-03-17T03:00:00", {"1": [
                _souvenir("1_0", "concept", 10.0), _souvenir("1_1", "concept", 20.0),
                _souvenir("1_2", "reflection", 4.0)]}),
        })
        mesures = calculer(run)
        par_type = {l.type_souvenir: l for l in mesures.durees_de_vie if l.jour_simule == 1}
        assert par_type["concept"].duree_vie_mediane_jours == 15.0
        assert par_type["reflection"].entrees == 1
        assert _ligne(mesures, "1", 1, "memoire").entrees_ltm == 3

    def test_D8bis_une_journee_sans_point_a_ses_colonnes_d_etat_vides(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")])
        ligne = _ligne(calculer(run), "1", 1, "memoire")
        assert ligne.entrees_ltm is None and ligne.etat_lu_dans is None
        assert calculer(run).durees_de_vie == []

    def test_D9_un_type_absent_n_a_pas_de_ligne_plutot_qu_un_zero(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")], points={
            2: ("2026-03-17T03:00:00", {"1": [_souvenir("1_0", "concept", 10.0)]}),
        })
        types = {l.type_souvenir for l in calculer(run).durees_de_vie}
        assert types == {"concept"}


# ── E — Shock ──────────────────────────────────────────────────────────────────────────

class TestChoc:
    def _choc(self, agent, horodatage, retard=600, vecu="La voiture a calé."):
        return {"person_id": agent, "horodatage_simule": horodatage,
                "choc_id": "c6", "retard_injecte_s": retard, "incident_reseau": True,
                "correspondance_ratee": False, "vecu": vecu}

    def test_E1_exposes_et_minutes_injectees(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Voiture Privée")], chocs=[
            self._choc("1", "2026-03-16T08:05:00", retard=600),
            self._choc("1", "2026-03-16T18:05:00", retard=300),
        ])
        ligne = calculer(run).chocs[0]
        assert (ligne.expositions, ligne.minutes_injectees) == (2, 15.0)
        assert ligne.incidents_reseau == 2

    def test_E2_E3_un_souvenir_dans_le_prompt_du_jour_est_compte(self, tmp_path):
        """"Served" is read in the prompt's History section — what the model has read."""
        vecu = "The engine made a grinding noise and the car stalled twice near the ring road."
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Voiture Privée")],
                   chocs=[self._choc("1", "2026-03-16T08:05:00", vecu=vecu)])
        _echanges(run, [_prompt("1", "2026-03-16", "17:30", [
            "[Monday, March 16] My engine was grinding and it stalled on the ring road."])])
        ligne = calculer(run).chocs[0]
        assert ligne.appariement == "mots"
        assert ligne.souvenir_choc_servi is True
        assert ligne.decisions_avec_souvenir_choc == 1

    def test_E4_sans_prompt_la_colonne_est_VIDE_et_non_fausse(self, tmp_path):
        """No exchange log: we do not know, and "not served" would be a lie."""
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Voiture Privée")],
                   chocs=[self._choc("1", "2026-03-16T08:05:00")])
        ligne = calculer(run).chocs[0]
        assert ligne.appariement == "sans prompt"
        assert ligne.souvenir_choc_servi is None
        assert ligne.decisions_avec_souvenir_choc is None

    def test_E4bis_des_prompts_sans_trace_valent_zero_mesure(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Voiture Privée")],
                   chocs=[self._choc("1", "2026-03-16T08:05:00",
                                     vecu="The engine made a grinding noise and stalled.")])
        _echanges(run, [_prompt("1", "2026-03-16", "17:30", ["[Monday, March 16] Calm day."])])
        ligne = calculer(run).chocs[0]
        assert (ligne.appariement, ligne.decisions_avec_souvenir_choc) == ("aucune trace", 0)
        assert ligne.souvenir_choc_servi is False

    def test_E5_aucun_choc_declare_aucune_ligne(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")])
        assert calculer(run).chocs == []

    # ── Article read on waking (defect of run 2026-09-24_17_50) ─────────────────────────
    def _lecture(self, agent, horodatage, jour_run):
        return {"person_id": agent, "horodatage_simule": horodatage, "choc_id": "a09",
                "evenement_id": "a09", "canal": "lu", "moment": "reveil",
                "jour_run": jour_run, "retard_injecte_s": 0, "vecu": "Parks closed."}

    def _semaine_de_trajets(self):
        return [_t("1", f"2026-03-{j}", "08:00:00", "Voiture Privée") for j in (16, 17, 18)]

    def test_E6_un_article_lu_a_minuit_appartient_au_jour_ou_il_est_lu(self, tmp_path):
        """Injected at 00:00 on the 17th, it came out on the 16th: the trips' 3 h boundary moved it back."""
        run = _run(tmp_path, self._semaine_de_trajets(),
                   chocs=[self._lecture("1", "2026-03-17T00:00:00", jour_run=2)])
        ligne = calculer(run).chocs[0]
        assert (ligne.jour_simule, ligne.date_simulee) == (2, "2026-03-17")

    def test_E7_un_choc_vecu_apres_minuit_garde_la_frontiere_de_3_h(self, tmp_path):
        """Joined to a trip before 3 h, it falls on the same day: declared-shock figures unchanged."""
        choc = self._choc("1", "2026-03-18T01:30:00")
        run = _run(tmp_path, self._semaine_de_trajets(), chocs=[choc])
        ligne = calculer(run).chocs[0]
        assert (ligne.jour_simule, ligne.date_simulee) == (2, "2026-03-17")

    def test_E8_une_lecture_un_jour_sans_trajet_garde_sa_ligne(self, tmp_path):
        """One reads the paper even on a day one does not go out: the exposure happened."""
        trajets = [_t("1", "2026-03-16", "08:00:00", "Marche"),
                   _t("1", "2026-03-19", "08:00:00", "Marche")]
        run = _run(tmp_path, trajets,
                   chocs=[self._lecture("1", "2026-03-17T00:00:00", jour_run=2)])
        ligne = calculer(run).chocs[0]
        assert (ligne.jour_simule, ligne.date_simulee) == (2, "2026-03-17")

    def test_E9_le_canal_est_ecrit_dans_le_csv(self, tmp_path):
        """The header carried the column; the row never wrote it."""
        run = _run(tmp_path, self._semaine_de_trajets(),
                   chocs=[self._lecture("1", "2026-03-17T00:00:00", jour_run=2)])
        chemin = ecrire(calculer(run), tmp_path / "sortie")["chocs"]
        with chemin.open(encoding="utf-8") as f:
            assert [l["canal"] for l in csv.DictReader(f)] == ["lu"]

    def test_E10_un_run_du_079_sans_canal_laisse_la_colonne_vide(self, tmp_path):
        run = _run(tmp_path, self._semaine_de_trajets(),
                   chocs=[self._choc("1", "2026-03-17T08:05:00")])
        chemin = ecrire(calculer(run), tmp_path / "sortie")["chocs"]
        with chemin.open(encoding="utf-8") as f:
            assert [l["canal"] for l in csv.DictReader(f)] == [""]


# ── F (writing) — rewriting of flows, preservation of states ───────────────────────────

class TestEcriture:
    def _run_simple(self, tmp_path):
        return _run(tmp_path, [
            _t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="a"),
            _t("1", "2026-03-17", "08:00:00", "Marche", activite="a"),
        ], points={2: ("2026-03-17T03:00:00", {"1": [_souvenir("1_0", "concept", 12.0)]})})

    def test_F2_aucune_cle_n_apparait_deux_fois(self, tmp_path):
        chemins = mesurer_et_ecrire(self._run_simple(tmp_path))
        for nom, chemin in chemins.items():
            with chemin.open(newline="", encoding="utf-8") as flux:
                lignes = list(csv.DictReader(flux))
            if nom not in TABLES:
                continue  # `souvenirs_derives`: a snapshot, without a day key
            cles = [tuple(l[c] for c in TABLES[nom].cle) for l in lignes]
            assert len(cles) == len(set(cles)), nom

    def test_F4_les_csv_sont_reecrits_en_entier_sans_reliquat(self, tmp_path):
        run = self._run_simple(tmp_path)
        chemins = mesurer_et_ecrire(run)
        avant = chemins["choix_modal"].read_text(encoding="utf-8")
        mesurer_et_ecrire(run)
        assert chemins["choix_modal"].read_text(encoding="utf-8") == avant

    def test_D8ter_un_etat_deja_ecrit_n_est_jamais_reecrit(self, tmp_path):
        """The replay of a resume overwrites the points with the FROZEN state (measured 2026-09-16).

        The first pass fixes the state, at the time the information still exists.
        """
        run = self._run_simple(tmp_path)
        chemins = mesurer_et_ecrire(run)
        # The point is rewritten with an impoverished state, as a replay does.
        _point(run, 2, "2026-03-17T03:00:00", {"1": []})
        mesurer_et_ecrire(run)
        with chemins["memoire"].open(newline="", encoding="utf-8") as flux:
            ligne = next(l for l in csv.DictReader(flux) if l["jour_simule"] == "1")
        assert ligne["entrees_ltm"] == "1"

    def test_F1_les_flux_d_un_jour_anterieur_sont_identiques_apres_une_reprise(self, tmp_path):
        run = self._run_simple(tmp_path)
        chemins = mesurer_et_ecrire(run)
        avant = [l for l in csv.DictReader(chemins["habitudes"].open(encoding="utf-8"))
                 if l["jour_simule"] == "1"]
        # A resume replays day 1: the same decision is rewritten into moves.csv.
        with (run / "moves.csv").open("a", newline="", encoding="utf-8") as flux:
            csv.DictWriter(flux, fieldnames=COLONNES).writerow(
                dict(_t("1", "2026-03-16", "08:00:00", "Vélo", activite="a"),
                     **{"Heure de calcul": "2026-09-16T20:00:00+00:00"}))
        mesurer_et_ecrire(run)
        apres = [l for l in csv.DictReader(chemins["habitudes"].open(encoding="utf-8"))
                 if l["jour_simule"] == "1"]
        assert apres == avant

    def test_H2_le_repertoire_est_cree_a_la_premiere_ecriture(self, tmp_path):
        run = self._run_simple(tmp_path)
        assert not (run / "mesures").exists()
        mesures = calculer(run)
        assert not (run / "mesures").exists(), "the computation alone must not write anything"
        ecrire(mesures)
        assert (run / "mesures").is_dir()

    def test_H4_une_source_absente_n_empeche_pas_les_mesures_calculables(self, tmp_path):
        run = _run(tmp_path, [_t("1", "2026-03-16", "08:00:00", "Marche")])
        chemins = mesurer_et_ecrire(run)
        assert chemins["choix_modal"].is_file() and chemins["chocs"].is_file()
        assert chemins["chocs"].read_text(encoding="utf-8").strip().count("\n") == 0


def test_journee_du_moment_rend_none_sur_une_valeur_illisible():
    assert journee_du_moment("") is None
    assert journee_du_moment("pas une date") is None
    assert journees_vecues([]) == []


# ── F — Continuity on both sides of an interruption ────────────────────────────────────

class TestContinuite:
    """The continuity check: an interruption neither duplicates, gaps nor shifts a curve."""

    def _deux_jours(self, tmp_path, jours=("2026-03-16", "2026-03-17")):
        return [_t("1", jour, "08:00:00", "Voiture Privée", activite="a") for jour in jours]

    def test_F6_un_dedoublement_fabrique_est_REFUSE(self, tmp_path):
        from scripts.analysis.mesures.continuite import verifier

        run = _run(tmp_path, self._deux_jours(tmp_path))
        chemins = mesurer_et_ecrire(run)
        dossier = chemins["choix_modal"].parent
        with chemins["choix_modal"].open("a", encoding="utf-8") as flux:
            flux.write("1,2026-03-16,1,1,1,1,1,1,0,0,0,0,0,0,0\n")
        verdict = verifier(dossier, dossier)
        assert not verdict.continu
        assert any(r.genre == "dédoublement" for r in verdict.ruptures)

    def test_F6bis_un_trou_de_jour_OUVRE_est_refuse(self, tmp_path):
        from scripts.analysis.mesures.continuite import verifier

        run = _run(tmp_path, self._deux_jours(tmp_path, ("2026-03-16", "2026-03-18")))
        dossier = mesurer_et_ecrire(run)["choix_modal"].parent
        verdict = verifier(dossier, dossier)
        assert any(r.genre == "trou" and "2026-03-17" in r.detail for r in verdict.ruptures)

    def test_F6ter_un_week_end_saute_n_est_PAS_un_trou(self, tmp_path):
        """The simulation postpones weekend departures to Monday: it is not an interruption."""
        from scripts.analysis.mesures.continuite import verifier

        run = _run(tmp_path, self._deux_jours(tmp_path, ("2026-03-20", "2026-03-23")))
        dossier = mesurer_et_ecrire(run)["choix_modal"].parent
        assert verifier(dossier, dossier).continu

    def test_F6quater_un_saut_de_valeur_anterieure_est_refuse(self, tmp_path):
        from scripts.analysis.mesures.continuite import verifier

        run = _run(tmp_path, self._deux_jours(tmp_path))
        avant = tmp_path / "avant"
        apres = mesurer_et_ecrire(run)["choix_modal"].parent
        avant.mkdir()
        for fichier in apres.glob("*.csv"):
            (avant / fichier.name).write_text(fichier.read_text(encoding="utf-8"),
                                              encoding="utf-8")
        # A day-1 value changes after the interruption: this is precisely what we track.
        texte = (apres / "choix_modal_par_jour.csv").read_text(encoding="utf-8")
        (apres / "choix_modal_par_jour.csv").write_text(
            texte.replace("1,2026-03-16,1,1,1,1", "1,2026-03-16,1,9,1,1"), encoding="utf-8")
        verdict = verifier(avant, apres)
        assert any(r.genre == "saut" and "trajets" in r.detail for r in verdict.ruptures)

    def test_F7_un_run_coupe_puis_repris_donne_les_MEMES_lignes_qu_un_run_continu(self, tmp_path):
        """The heart of the check: the interruption must change nothing in what precedes."""
        from scripts.analysis.mesures.continuite import verifier

        continu = _run(tmp_path / "continu", self._deux_jours(tmp_path))
        dossier_continu = mesurer_et_ecrire(continu)["choix_modal"].parent

        # The interrupted run: day 1 is written, then the resume REPLAYS it before writing day 2.
        coupe = _run(tmp_path / "coupe", [self._deux_jours(tmp_path)[0]])
        dossier_coupe = mesurer_et_ecrire(coupe)["choix_modal"].parent
        rejouee = dict(self._deux_jours(tmp_path)[0],
                       **{"Heure de calcul": "2026-09-16T23:00:00+00:00"})
        with (coupe / "moves.csv").open("a", newline="", encoding="utf-8") as flux:
            ecrivain = csv.DictWriter(flux, fieldnames=COLONNES)
            ecrivain.writerow(rejouee)
            ecrivain.writerow(self._deux_jours(tmp_path)[1])
        mesurer_et_ecrire(coupe)

        assert verifier(dossier_continu, dossier_coupe).continu
        for fichier in ("choix_modal_par_jour.csv", "habitudes_par_activite.csv"):
            assert (dossier_coupe / fichier).read_text(encoding="utf-8") == (
                dossier_continu / fichier).read_text(encoding="utf-8")


class TestLesDeuxReperesTemporels:
    """`moves.csv` carries the time of the DECISION and the time of DEPARTURE. They differ.

    Measured on `2026-09-16_15_58`: 30 rows out of 270 set them more than an hour and a half apart,
    19 of them by exactly 48 hours — decided on Saturday, departed on Monday.
    """

    def _decidee_samedi_partie_lundi(self):
        """A decision of Saturday March 21 for a departure on Monday 23."""
        ligne = _t("1", "2026-03-23", "08:00:00", "Voiture Privée", activite="travail")
        ligne["Temps simulé"] = "2026-03-21 08:00:00"
        return ligne

    def test_la_journee_est_datee_par_le_DEPART_et_non_par_la_decision(self, tmp_path):
        """Dating by the decision would fabricate a "Saturday" day on which nothing moves."""
        run = _run(tmp_path, [
            _t("1", "2026-03-20", "08:00:00", "Voiture Privée", activite="travail"),
            self._decidee_samedi_partie_lundi(),
        ])
        mesures = calculer(run)
        assert [j.date for j in mesures.journees] == ["2026-03-20", "2026-03-23"]
        assert all(l.date_simulee != "2026-03-21" for l in mesures.choix_modal)

    def test_et_la_veille_du_lundi_reste_le_vendredi(self, tmp_path):
        run = _run(tmp_path, [
            _t("1", "2026-03-20", "08:00:00", "Voiture Privée", activite="travail"),
            self._decidee_samedi_partie_lundi(),
        ])
        ligne = _ligne(calculer(run), "1", 4, "habitudes")
        assert (ligne.date_simulee, ligne.mode_veille, ligne.reprise_veille) == (
            "2026-03-23", "car", True)

    def test_la_deduplication_du_rejeu_utilise_l_instant_de_DECISION(self, tmp_path):
        """Two departures of the same activity, decided at two times: two trips, not one.

        That is why the deduplication key is `(personne, activité, Temps simulé)` and not
        `(personne, activité, jour)`: per day, these two distinct decisions would merge.
        """
        premier = _t("1", "2026-03-16", "08:00:00", "Voiture Privée", activite="travail")
        premier["Temps simulé"] = "2026-03-16 05:00:00"
        second = _t("1", "2026-03-16", "14:00:00", "Marche", activite="travail")
        second["Temps simulé"] = "2026-03-16 11:00:00"
        mesures = calculer(_run(tmp_path, [premier, second]))
        assert mesures.trajets_rejoues == 0
        assert _ligne(mesures, "1", 1).trajets == 2

    def test_la_colonne_Jour_relatif_au_choc_n_est_JAMAIS_lue(self):
        """It mixes two conventions if the shock configuration moves during the run.

        Relative days are re-derived from the timestamps of `chocs.jsonl`.
        """
        from scripts.analysis.mesures import calcul, ecriture

        for module in (calcul, ecriture):
            source = Path(module.__file__).read_text(encoding="utf-8")
            code = "\n".join(l for l in source.splitlines()
                             if not l.strip().startswith(("#", "⚠", "|")))
            assert "Jour relatif au choc" not in code, module.__name__


# ── S — The memory of the event, day after day (analysis of 2026-09-25) ─────────────────
# The treated arm 2026-09-24_17_50: six prompts carried article a09 on the following days, and
# `evenement_par_jour.csv` said 0 — literal link searched in memory, recall trace
# instead of the prompt, and on the same day only.

ARTICLE = ("(Translated from French)\nGusts above 80 km/h: Toulouse closes its parks and "
           "gardens this Thursday evening under a yellow storm warning.")


class TestSouvenir:
    def _lecture(self, agent="1", horodatage="2026-03-17T00:00:00"):
        return {"person_id": agent, "horodatage_simule": horodatage, "choc_id": "a09",
                "evenement_id": "a09", "canal": "lu", "moment": "reveil", "jour_run": 2,
                "retard_injecte_s": 0, "vecu": ARTICLE}

    def _run(self, tmp_path, prompts, *, ltm=None, foyer=("1",), trajets=None):
        trajets = trajets or [
            _t(a, f"2026-03-{j}", "08:00:00", "Voiture Privée", activite=f"act-{j}")
            for a in foyer for j in (16, 17, 18, 19)
        ]
        run = _run(tmp_path, trajets, chocs=[self._lecture()])
        (run / "population_4.json").write_text(json.dumps([
            {"person_id": a, "household": {"id": "h1"}} for a in foyer
        ]), encoding="utf-8")
        _echanges(run, prompts)
        racine = run / "long_term_memory" / "user_metadata" / "shard_0"
        racine.mkdir(parents=True, exist_ok=True)
        for agent, entrees in (ltm or {}).items():
            (racine / f"{agent}.json").write_text(
                json.dumps({"person_id": agent, "entries": entrees}), encoding="utf-8")
        return run

    def _du(self, mesures, agent, date):
        return next(l for l in mesures.souvenirs
                    if l.person_id == agent and l.date_simulee == date)

    def test_S1_une_reformulation_servie_deux_jours_apres_est_comptee(self, tmp_path):
        run = self._run(tmp_path, [
            _prompt("1", "2026-03-19", "08:00", [
                "Ce que je sais",
                "- Toulouse closed its parks and gardens because of a yellow storm warning.  (0 obs.)",
            ]),
        ])
        ligne = self._du(calculer(run), "1", "2026-03-19")
        assert (ligne.role, ligne.jours_depuis_j0) == ("expose", 2)
        assert (ligne.prompts_avec_souvenir, ligne.souvenir_mots) == (1, 1)
        assert (ligne.via_connaissances, ligne.via_changements, ligne.via_rappel) == (1, 0, 0)

    def test_S1_bis_les_titres_anglais_se_lisent_comme_les_francais(self, tmp_path):
        """2026-09-25 — the block speaks English; earlier archives keep their headings."""
        run = self._run(tmp_path, [
            _prompt("1", "2026-03-19", "08:00", [
                "What I know",
                "- Toulouse closed its parks and gardens because of a yellow storm warning.  (0 obs.)",
            ]),
        ])
        ligne = self._du(calculer(run), "1", "2026-03-19")
        assert (ligne.via_connaissances, ligne.via_changements, ligne.via_rappel) == (1, 0, 0)

    def test_S2_le_co_resident_est_suivi_aussi(self, tmp_path):
        run = self._run(tmp_path, [
            _prompt("2", "2026-03-18", "17:00", [
                "[Tuesday, March 17] Dad said the parks and gardens close tonight: storm warning."]),
        ], foyer=("1", "2"))
        ligne = self._du(calculer(run), "2", "2026-03-18")
        assert ligne.role == "co_resident"
        assert (ligne.prompts_avec_souvenir, ligne.via_rappel) == (1, 1)

    def test_S3_les_mots_que_le_foyer_ecrivait_deja_ne_font_pas_un_souvenir(self, tmp_path):
        """"yellow storm warning" written BEFORE the reading is no longer the article's trace."""
        avant = {"doc_id": "1_1", "memory_type": "reflection", "force": 1.0,
                 "content": "A yellow storm warning with gusts, so the parks were empty.",
                 "timestamp": "2026-03-16T20:00:00"}
        run = self._run(tmp_path, [
            _prompt("1", "2026-03-19", "08:00", [
                "[Wednesday, March 18] Another yellow storm warning, gusts again near the parks."]),
        ], ltm={"1": [avant]})
        assert self._du(calculer(run), "1", "2026-03-19").prompts_avec_souvenir == 0

    def test_S4_la_date_d_une_episodique_ne_compte_pas(self, tmp_path):
        """The template dates each episodic memory; the article says "this Thursday"."""
        run = self._run(tmp_path, [
            _prompt("1", "2026-03-19", "08:00", [
                "[Thursday, March 19] The drive closely matched my plan; the gardens looked nice."]),
        ])
        assert self._du(calculer(run), "1", "2026-03-19").prompts_avec_souvenir == 0

    def test_S5_l_article_dans_la_ligne_garantie_se_lit_comme_texte_par_les_changements(
            self, tmp_path):
        run = self._run(tmp_path, [
            _prompt("1", "2026-03-17", "08:00", [
                "Ce qui a changé récemment",
                # A single entry, on two lines: the template prefixes only the first.
                "- [ PRESSE ] This morning I read in the paper: « (Translated from French)\n"
                "Gusts above 80 km/h: Toulouse closes its parks and gardens this Thursday "
                "evening under a yellow storm warning. »",
            ]),
        ])
        mesures = calculer(run)
        ligne = self._du(mesures, "1", "2026-03-17")
        assert (ligne.souvenir_texte, ligne.via_changements) == (1, 1)
        choc = mesures.chocs[0]
        assert (choc.appariement, choc.decisions_avec_souvenir_choc) == ("texte", 1)

    def test_S6_une_decision_sans_prompt_se_montre(self, tmp_path):
        """A decision taken from the cache has no prompt: neither with nor without memory."""
        trajets = [_t("1", "2026-03-18", "08:00:00", "Marche", activite="a"),
                   _t("1", "2026-03-18", "17:00:00", "Marche", activite="b"),
                   _t("1", "2026-03-16", "08:00:00", "Marche", activite="c")]
        run = self._run(tmp_path, [_prompt("1", "2026-03-18", "08:00", ["Calm."])],
                        trajets=trajets)
        ligne = self._du(calculer(run), "1", "2026-03-18")
        assert (ligne.decisions, ligne.prompts, ligne.decisions_sans_prompt) == (2, 1, 1)

    def test_S7_les_deux_fichiers_sont_ecrits(self, tmp_path):
        concept = {"doc_id": "1_9", "memory_type": "concept", "force": 1.0,
                   "content": json.dumps(["Parks and gardens closed under a yellow storm warning",
                                          "", "", "", ""]),
                   "timestamp": "2026-03-17T20:00:00"}
        run = self._run(tmp_path, [], ltm={"1": [concept]})
        chemins = ecrire(calculer(run), tmp_path / "sortie")
        assert chemins["souvenirs"].name == "souvenir_evenement_par_jour.csv"
        with chemins["souvenirs_derives"].open(encoding="utf-8") as f:
            derives = list(csv.DictReader(f))
        assert [(d["doc_id"], d["appariement"]) for d in derives] == [("1_9", "mots")]

    def test_S8_un_prompt_rejoue_ne_compte_qu_une_fois(self, tmp_path):
        bloc = _prompt("1", "2026-03-19", "08:00", [
            "Ce que je sais", "- Parks and gardens closed under a yellow storm warning.  (0 obs.)"])
        run = self._run(tmp_path, [bloc, bloc])
        assert self._du(calculer(run), "1", "2026-03-19").prompts == 1
