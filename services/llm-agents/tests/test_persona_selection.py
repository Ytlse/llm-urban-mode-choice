"""Choosing personas whose decisions are observable.

Three personas out of the five of run `2026-09-16_15_58` bring nothing: two have a single
itinerary offered one time out of two, the third only lives 19 trips. The criterion of this lot is
measurable BEFORE the run, on a reference day — hence reproducible, and enforceable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.data.population import selectionner_personas_mesurables as S

RACINE = next(
    a
    for a in Path(__file__).resolve().parents
    if (a / "scripts" / "synthesis").is_dir()
)

# Reference run: 234 candidates, and 920187 has 15 trips in it. It is not
# versioned (`data/experiences/*` is ignored), hence the skip rather than a failure elsewhere.
#
# SINCE TICKET 098, IT IS NOT THERE AT ALL: it was decided on the old v6 EN set, and
# that whole set went to cold archive. The two tests that rely on it therefore
# now always skip. The path stays written under `data/` ON PURPOSE — pointing it at the archive
# would amount to referencing it, which the archive forbids.
#
# This skip is NOT the solution, it is the finding pending a decision.
# Replaying the same criterion on the counterpart of the corrected set
# (`…_EN_c_t0_nosim/executions/2026-09-17_07_34_55`) gives DIFFERENT figures:
#   - 246 candidates and not 234;
#   - 920187 holds (15 trips, 9 walking);
#   - but 41275 only makes a single mode and CEASES to be a candidate — yet it is precisely
#     the one that carries the demonstration of test A5bis.
# Repointing this path without deciding would thus silently rewrite the reference figures
# that `data/population/population_10_mesurables_093/MANIFEST.yaml` seals
# (`run_de_reference`, `candidats: 234`). The selection of the 10 measurable personas must be
# re-derived on the corrected substrate, and the MANIFEST rewritten with it. It is a scientific
# decision, not a test adjustment.
RUN_REFERENCE = (
    RACINE
    / "data/experiences"
    / "exp_gemini-35-fl_proexp08_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_t0_nosim"
    / "executions/2026-09-15_12_38_15"
)

MOTIF_SAUT = (
    "run de référence absent : gelé en archive froide avec l'ancien jeu v6 EN (ticket 098). "
    "La sélection des personas mesurables est à re-dériver sur le jeu corrigé — voir le "
    "commentaire au-dessus de RUN_REFERENCE."
)

COLONNES = [
    "ID Personne",
    "ID Activité",
    "Temps simulé",
    "Heure de départ",
    "Heure de calcul",
    "Mode de transport Choisi",
    "Options présentées",
    "Motifs de déplacement",
]


def _trajet(
    personne, mode, options, *, heure="08:00:00", jour="2026-03-16", activite="a1"
):
    return {
        "ID Personne": personne,
        "ID Activité": activite,
        "Temps simulé": f"{jour} {heure}",
        "Heure de départ": f"{jour} {heure}",
        "Heure de calcul": f"{jour} {heure}",
        "Mode de transport Choisi": mode,
        "Options présentées": str(options),
        "Motifs de déplacement": "work",
    }


def _ecrire_run(dossier: Path, lignes: list[dict]) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    with (dossier / "moves.csv").open("w", newline="", encoding="utf-8") as flux:
        ecrivain = csv.DictWriter(flux, fieldnames=COLONNES)
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    return dossier


def _agent(personne: str) -> dict:
    return {
        "person_id": personne,
        "identity": {
            "traits_json": {"name": f"Agent {personne}", "age": 40},
            "activities": [{"purpose": "work"}, {"purpose": "home"}],
        },
    }


def _ecrire_source(chemin: Path, personnes: list[str]) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps([_agent(p) for p in personnes]), encoding="utf-8")
    return chemin


def _journee_variee(
    personne: str, *, trajets=4, options=5, modes=("Voiture Privée", "Marche")
):
    """A day that meets the criterion: enough trips, options, several modes."""
    return [
        _trajet(
            personne,
            modes[i % len(modes)],
            options,
            heure=f"{6 + i:02d}:00:00",
            activite=f"a{i}",
        )
        for i in range(trajets)
    ]


class TestCritere:
    def test_A1_moins_de_quatre_trajets_recale(self, tmp_path):
        run = _ecrire_run(tmp_path / "run", _journee_variee("11195", trajets=3))
        mesures = S.mesurer(run)
        assert mesures["11195"].trajets == 3
        assert not S.est_candidat(mesures["11195"])

    def test_A2_un_seul_trajet_mono_option_recale_quel_que_soit_le_reste(
        self, tmp_path
    ):
        lignes = _journee_variee("42", trajets=30)
        lignes[17]["Options présentées"] = "1"
        run = _ecrire_run(tmp_path / "run", lignes)
        mesure = S.mesurer(run)["42"]
        assert mesure.trajets == 30 and mesure.options_min == 1
        assert not S.est_candidat(mesure)

    def test_A3_un_seul_mode_recale_meme_avec_des_options(self, tmp_path):
        run = _ecrire_run(
            tmp_path / "run",
            _journee_variee("609", trajets=8, options=3, modes=("Voiture Privée",)),
        )
        mesure = S.mesurer(run)["609"]
        assert mesure.options_min == 3 and len(mesure.modes) == 1
        assert not S.est_candidat(mesure)

    def test_A4_les_trois_conditions_reunies_font_un_candidat(self, tmp_path):
        run = _ecrire_run(tmp_path / "run", _journee_variee("920187", trajets=15))
        assert S.est_candidat(S.mesurer(run)["920187"])

    def test_A4bis_le_critere_ne_juge_pas_les_modes_proposes_mais_les_modes_CHOISIS(
        self, tmp_path
    ):
        """An agent offered everything who always takes the car is not a candidate.

        This is the case of 609 and 41275: their "decisions" are visible, but they do not vary.
        """
        run = _ecrire_run(
            tmp_path / "run",
            _journee_variee("41275", trajets=6, options=9, modes=("Voiture Privée",)),
        )
        assert not S.est_candidat(S.mesurer(run)["41275"])

    @pytest.mark.skipif(
        not (RUN_REFERENCE / "moves.csv").is_file(),
        reason=MOTIF_SAUT,
    )
    def test_A5_le_run_de_reference_retient_bien_234_candidats(self):
        mesures = S.mesurer(RUN_REFERENCE)
        candidats = [p for p, m in mesures.items() if S.est_candidat(m)]
        assert len(candidats) == 234
        # The reference figures, checked at the source.
        assert mesures["920187"].trajets == 15
        assert dict(mesures["920187"].modes)["walking"] == 9
        # 609 (car only, 3 fixed options) and 11195 (2 trips) are rejected BY THE
        # CRITERION, not by decree.
        for recale in ("609", "11195"):
            assert not S.est_candidat(mesures[recale])
        # The two kept ones pass the criterion: keeping them is not a special favour.
        for conserve in ("899549", "616478"):
            assert S.est_candidat(mesures[conserve])

    @pytest.mark.skipif(
        not (RUN_REFERENCE / "moves.csv").is_file(),
        reason=MOTIF_SAUT,
    )
    def test_A5bis_le_critere_juge_un_run_et_pas_un_agent_en_soi(self):
        """41275 PASSES the criterion on the reference run, whereas the selection rejects it.

        It is not a contradiction, it is the exact scope of the criterion: on that run —
        another model, another prompt variant — it makes two modes over four trips; on the
        five-persona run of 2026-09-16, it only drives, with a single itinerary
        one time out of two. A selection is thus only valid FOR ITS REFERENCE RUN, and the
        MANIFEST must name that run so that the selection stays enforceable.
        """
        mesure = S.mesurer(RUN_REFERENCE)["41275"]
        assert S.est_candidat(mesure)
        assert mesure.modes_distincts == 2


class TestChoix:
    def _fixture(self, tmp_path, personnes_candidates, modes_par_agent=None):
        lignes = []
        for personne in personnes_candidates:
            modes = (modes_par_agent or {}).get(personne, ("Voiture Privée", "Marche"))
            lignes += _journee_variee(personne, trajets=6, modes=modes)
        run = _ecrire_run(tmp_path / "run", lignes)
        source = _ecrire_source(tmp_path / "pop.json", personnes_candidates)
        return run, source

    def test_A6_les_conserves_sont_retenus_de_droit_et_leur_critere_est_ecrit(
        self, tmp_path
    ):
        # 616478 only makes a single mode: it is kept, AND the manifest says it fails.
        run, source = self._fixture(
            tmp_path,
            ["899549", "616478"] + [str(900 + i) for i in range(12)],
            modes_par_agent={"616478": ("Marche",)},
        )
        retenus = S.choisir(
            S.mesurer(run),
            json.loads(source.read_text()),
            conserver=["899549", "616478"],
            combien=10,
        )
        par_id = {r.person_id: r for r in retenus}
        assert {"899549", "616478"} <= set(par_id)
        assert par_id["899549"].passe_le_critere is True
        assert par_id["616478"].passe_le_critere is False
        assert par_id["616478"].motif == "conservé"

    def test_A7_dix_agents_et_les_modes_dominants_couverts(self, tmp_path):
        modes = {
            "1001": ("Voiture Privée", "Marche"),
            "1002": ("Marche", "Transports_collectifs"),
            "1003": ("Transports_collectifs", "Marche"),
            "1004": ("Vélo", "Marche"),
        }
        run, source = self._fixture(
            tmp_path,
            list(modes) + [str(2000 + i) for i in range(10)],
            modes_par_agent=modes,
        )
        retenus = S.choisir(
            S.mesurer(run), json.loads(source.read_text()), conserver=[], combien=10
        )
        assert len(retenus) == 10
        dominants = {r.mode_dominant for r in retenus}
        assert {"car", "walking", "public_transport", "cycling"} <= dominants

    def test_A8_deux_executions_rendent_la_meme_liste_dans_le_meme_ordre(
        self, tmp_path
    ):
        run, source = self._fixture(tmp_path, [str(3000 + i) for i in range(14)])
        population = json.loads(source.read_text())
        mesures = S.mesurer(run)
        premier = [
            r.person_id
            for r in S.choisir(mesures, population, conserver=[], combien=10)
        ]
        second = [
            r.person_id
            for r in S.choisir(mesures, population, conserver=[], combien=10)
        ]
        assert premier == second

    def test_A9_les_agents_sont_recopies_tels_quels(self, tmp_path):
        run, source = self._fixture(tmp_path, [str(4000 + i) for i in range(12)])
        population = json.loads(source.read_text())
        retenus = S.choisir(S.mesurer(run), population, conserver=[], combien=10)
        par_id = {str(p["person_id"]): p for p in population}
        for retenu in retenus:
            assert retenu.persona == par_id[retenu.person_id]

    def test_A12_a_mode_dominant_egal_le_plus_observable_passe_avant(self, tmp_path):
        """The criterion is a floor, not a target.

        Sorting by identifier gave ten minimally compliant agents — four trips, two
        modes — and let in `41275`, the very one the selection discards for teaching
        nothing. An agent that decides fifteen times offers fifteen handholds to memory.
        """
        lignes = _journee_variee("9001", trajets=4)  # compliant, but minimal
        lignes += _journee_variee(
            "9002", trajets=12, modes=("Voiture Privée", "Marche", "Vélo")
        )  # rich
        run = _ecrire_run(tmp_path / "run", lignes)
        source = _ecrire_source(tmp_path / "pop.json", ["9001", "9002"])
        retenus = S.choisir(
            S.mesurer(run), json.loads(source.read_text()), conserver=[], combien=1
        )
        assert [r.person_id for r in retenus] == ["9002"]

    def test_A10_moins_de_candidats_que_demande_refuse_au_lieu_de_completer(
        self, tmp_path
    ):
        run, source = self._fixture(tmp_path, [str(5000 + i) for i in range(4)])
        with pytest.raises(SystemExit) as refus:
            S.choisir(
                S.mesurer(run), json.loads(source.read_text()), conserver=[], combien=10
            )
        assert "4" in str(refus.value) and "10" in str(refus.value)

    def test_A10bis_un_conserve_absent_de_la_source_refuse_en_le_nommant(
        self, tmp_path
    ):
        run, source = self._fixture(tmp_path, [str(6000 + i) for i in range(12)])
        with pytest.raises(SystemExit) as refus:
            S.choisir(
                S.mesurer(run),
                json.loads(source.read_text()),
                conserver=["999999"],
                combien=10,
            )
        assert "999999" in str(refus.value)


class TestManifeste:
    def test_A11_le_manifeste_porte_le_critere_le_run_et_les_chiffres_mesures(
        self, tmp_path
    ):
        lignes = []
        for personne in [str(7000 + i) for i in range(12)]:
            lignes += _journee_variee(personne, trajets=6)
        run = _ecrire_run(tmp_path / "run", lignes)
        source = _ecrire_source(
            tmp_path / "pop.json", [str(7000 + i) for i in range(12)]
        )
        sortie = tmp_path / "population_10"

        S.ecrire(run, source, sortie, conserver=[], combien=10)

        manifeste = (sortie / "MANIFEST.yaml").read_text(encoding="utf-8")
        assert "run_de_reference" in manifeste
        assert str(S.CRITERE.trajets_min) in manifeste
        assert "trajets: 6" in manifeste
        assert "PAS UN SCEAU" in manifeste.upper()
        produite = json.loads((sortie / "population.json").read_text(encoding="utf-8"))
        assert len(produite) == 10

    def test_A11bis_la_population_ecrite_ne_contient_que_des_agents_de_la_source(
        self, tmp_path
    ):
        personnes = [str(8000 + i) for i in range(12)]
        lignes = []
        for personne in personnes:
            lignes += _journee_variee(personne, trajets=6)
        run = _ecrire_run(tmp_path / "run", lignes)
        source = _ecrire_source(tmp_path / "pop.json", personnes)
        sortie = tmp_path / "population_10"
        S.ecrire(run, source, sortie, conserver=[], combien=10)
        produite = json.loads((sortie / "population.json").read_text(encoding="utf-8"))
        assert {str(p["person_id"]) for p in produite} <= set(personnes)


class TestVerification:
    """The "sealing" useful for ten agents: the delivered file is the one the manifest describes.

    It is NOT a cohort seal — `scripts.panel.seal_population` draws households by strata and
    checks thirteen margins at ± 1 point, which makes no sense for ten agents and is not the goal.
    """

    def _produire(self, tmp_path):
        personnes = [str(9000 + i) for i in range(12)]
        lignes = []
        for personne in personnes:
            lignes += _journee_variee(personne, trajets=6)
        run = _ecrire_run(tmp_path / "run", lignes)
        source = _ecrire_source(tmp_path / "pop.json", personnes)
        sortie = tmp_path / "population_10"
        S.ecrire(run, source, sortie, conserver=[], combien=10)
        return sortie

    def test_une_population_intacte_ne_leve_aucune_anomalie(self, tmp_path):
        assert S.verifier(self._produire(tmp_path)) == []

    def test_une_population_modifiee_a_la_main_est_detectee(self, tmp_path):
        """Without this check, an edited file would keep claiming to follow the criterion."""
        sortie = self._produire(tmp_path)
        agents = json.loads((sortie / "population.json").read_text(encoding="utf-8"))
        (sortie / "population.json").write_text(
            json.dumps(agents[:9]), encoding="utf-8"
        )
        anomalies = S.verifier(sortie)
        assert any("a changé" in a for a in anomalies)
        assert any("9 agent(s)" in a for a in anomalies)

    def test_une_source_disparue_est_dite_plutot_que_supposee_identique(self, tmp_path):
        sortie = self._produire(tmp_path)
        (tmp_path / "pop.json").unlink()
        assert any("plus rejouable" in a for a in S.verifier(sortie))

    def test_un_dossier_incomplet_le_dit_au_lieu_de_passer(self, tmp_path):
        vide = tmp_path / "rien"
        vide.mkdir()
        assert S.verifier(vide)


def test_la_population_du_ticket_est_livree_avec_le_depot():
    """The reference run lives under `data/experiences/`, which git ignores.

    The population of ten must not share that fate: kept on the machine that produced it only,
    the selection could not be replayed or checked elsewhere; its criterion would remain
    theoretical. It ships with the repository, as the very file its MANIFEST describes, and git
    does not ignore it (checked where a git work tree exists; a copy without one ships the files).
    """
    dossier = RACINE / "data" / "population" / "population_10_mesurables_093"
    fichier = dossier / "population.json"
    manifeste = yaml.safe_load((dossier / "MANIFEST.yaml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fichier.read_bytes()).hexdigest() == manifeste["population"]["sha256"]
    if (RACINE / ".git").exists() and shutil.which("git"):
        ignore = subprocess.run(["git", "-C", str(RACINE), "check-ignore", "-q", str(fichier)], check=False)
        assert ignore.returncode == 1, "data/.gitignore ignores the population of ten"
