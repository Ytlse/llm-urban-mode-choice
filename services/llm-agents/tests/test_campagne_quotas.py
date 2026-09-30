"""Ticket 074, lot D — the campaign carries 22 experiments to the end, across quotas.

What is checked here is not "the code runs" but the four promises that make a
campaign better than a human who relaunches:

1. it **refuses** rather than start on a definition that would make it fail midway;
2. it respects the **order of phases** — the free controls before the quota;
3. it **resumes where it stopped**, never at the start, even after a restart;
4. it **says** what it does — successes, failures, sleeps — with the doctrine's alarms.

Time is INJECTED (`dormir`, `max_tours`). A loop only tested by actually waiting
thirty seconds is not tested: it is only slow at not being so.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from loguru import logger
from experiences import campagne as C

# ── Bench ────────────────────────────────────────────────────────────────────


def _definir_experience(racine: Path, nom: str) -> Path:
    """A minimal definition: the campaign only reads its existence."""
    d = racine / nom
    d.mkdir(parents=True, exist_ok=True)
    (d / "experience.yaml").write_text(yaml.safe_dump({"nom": nom}), encoding="utf-8")
    return d


def _poser_etat(
    racine: Path,
    nom: str,
    etat: str,
    *,
    horodatage: str = "2026-09-14_10_00_00",
    raison: str | None = None,
    maj: str = "2026-09-14T10:00:00+00:00",
    interruptions: list[dict] | None = None,
) -> None:
    d = racine / nom / "executions" / horodatage
    d.mkdir(parents=True, exist_ok=True)
    (d / "etat.json").write_text(
        json.dumps({"etat": etat, "raison": raison, "maj": maj}),
        encoding="utf-8",
    )
    if interruptions is not None:
        # The `execution.yaml` carries the STRUCTURED trace of the interruption: it is what
        # tells an undergone pause from a requested pause, not the message of `etat.json`.
        (d / "execution.yaml").write_text(
            yaml.safe_dump({"interruptions": interruptions}, allow_unicode=True),
            encoding="utf-8",
        )


def _pause_chien_de_garde(racine: Path, nom: str, **kw) -> None:
    """What the runner writes after 420 s without progress: the process is dead."""
    _poser_etat(
        racine, nom, "en_pause",
        raison="pause automatique — 420s sans avancée ; 0/3299 archivées",
        interruptions=[{"instant": "2026-09-14T10:00:00+00:00", "cause": "pause",
                        "decisions_archivees": 0, "raison": "inactivite:420s"}],
        **kw,
    )


def _pause_humaine(racine: Path, nom: str, **kw) -> None:
    """What the runner writes when someone requested the pause: it waits for a human."""
    _poser_etat(
        racine, nom, "en_pause", raison="pause — 120/3299 archivées",
        interruptions=[{"instant": "2026-09-14T10:00:00+00:00", "cause": "pause",
                        "decisions_archivees": 120, "raison": "manuelle"}],
        **kw,
    )


@pytest.fixture
def journal():
    """The loguru messages, which `caplog` does not see (it only listens to `logging`)."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink)


@pytest.fixture
def banc(tmp_path, monkeypatch):
    """A pocket repository: defined experiments, a campaigns directory, no Docker."""
    experiences = tmp_path / "experiences"
    campagnes = tmp_path / "campagnes"
    experiences.mkdir()
    campagnes.mkdir()
    monkeypatch.setenv("EXPERIENCES_DIR", str(experiences))
    monkeypatch.setenv("CAMPAGNES_DIR", str(campagnes))

    lances: list[str] = []
    vrai_lancement = C._lancer_experience
    # The real `_lancer_experience` takes (exp, lanceurs): the stand-in must accept
    # both, otherwise it hides the signature instead of replacing it.
    monkeypatch.setattr(C, "_lancer_experience",
                        lambda exp, lanceurs=None, **kw: lances.append(exp))
    monkeypatch.setattr(C, "_tour_ordonnanceur", lambda: None)
    # Outside the "across quotas" mode, the campaign only reconciles the keys
    # (docker exec): the bench has no container.
    monkeypatch.setattr(C, "_reconcilier_cles", lambda: None, raising=False)
    return {"experiences": experiences, "campagnes": campagnes, "lances": lances,
            "vrai_lancement": vrai_lancement}


def _ecrire_campagne(banc, nom: str, phases: list[dict], **extra) -> None:
    doc = {"nom": nom, "version": C.VERSION_CAMPAGNE, "phases": phases, **extra}
    (banc["campagnes"] / f"{nom}.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _a_travers_les_quotas(banc, nom: str = "c") -> None:
    """The historical mode — batch sleep, deferral, retrieval —, opt-in since 2026-09-29."""
    f = banc["campagnes"] / f"{nom}.yaml"
    doc = yaml.safe_load(f.read_text(encoding="utf-8"))
    doc["a_travers_les_quotas"] = True
    f.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


@pytest.fixture
def deux_phases(banc):
    for n in ("t1", "t2", "llm1", "llm2"):
        _definir_experience(banc["experiences"], n)
    _ecrire_campagne(
        banc,
        "c",
        [
            {"nom": "temoins", "raison": "gratuits", "experiences": ["t1", "t2"]},
            {"nom": "llm", "experiences": ["llm1", "llm2"]},
        ],
    )
    return banc


# ── 1. It refuses rather than fail midway ────────────────────────────────────


class TestRefus:
    def test_une_campagne_absente_dit_lesquelles_existent(self, banc):
        _ecrire_campagne(banc, "reelle", [{"nom": "p", "experiences": ["x"]}])
        _definir_experience(banc["experiences"], "x")
        with pytest.raises(C.CampagneInvalide, match="reelle"):
            C.charger("fantome")

    def test_une_version_inattendue_est_refusee(self, banc):
        (banc["campagnes"] / "c.yaml").write_text(
            yaml.safe_dump(
                {
                    "nom": "c",
                    "version": "campagne0",
                    "phases": [{"nom": "p", "experiences": ["x"]}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(C.CampagneInvalide, match="campagne0"):
            C.charger("c")

    def test_une_experience_inexistante_est_refusee_AVANT_de_depenser(self, banc):
        _definir_experience(banc["experiences"], "existe")
        _ecrire_campagne(banc, "c", [{"nom": "p", "experiences": ["existe", "manque"]}])
        with pytest.raises(C.CampagneInvalide, match="manque"):
            C.charger("c")

    def test_une_phase_vide_est_refusee(self, banc):
        _ecrire_campagne(banc, "c", [{"nom": "vide", "experiences": []}])
        with pytest.raises(C.CampagneInvalide, match="carries no experiment"):
            C.charger("c")

    def test_une_experience_dans_deux_phases_est_refusee(self, banc):
        _definir_experience(banc["experiences"], "x")
        _ecrire_campagne(
            banc,
            "c",
            [{"nom": "a", "experiences": ["x"]}, {"nom": "b", "experiences": ["x"]}],
        )
        with pytest.raises(C.CampagneInvalide, match="two phases"):
            C.charger("c")


# ── 2. The order of phases ───────────────────────────────────────────────────


class TestPhases:
    def test_la_phase_2_ne_demarre_pas_avant_la_fin_de_la_phase_1(self, deux_phases):
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        # A single experiment in flight at a time, and it is an experiment of phase 1.
        assert deux_phases["lances"] == ["t1"], deux_phases["lances"]

    def test_la_phase_2_demarre_quand_la_phase_1_est_close(self, deux_phases):
        for n in ("t1", "t2"):
            _poser_etat(deux_phases["experiences"], n, "terminee")
        C.lancer("c", max_tours=4, dormir=lambda _s: None)
        assert deux_phases["lances"] == ["llm1"], deux_phases["lances"]
        etat = C.lire_etat("c")
        assert etat["phase_courante"] == "llm"
        assert set(etat["faites"]) == {"t1", "t2"}

    def test_une_campagne_toute_faite_se_declare_terminee(self, deux_phases):
        for n in ("t1", "t2", "llm1", "llm2"):
            _poser_etat(deux_phases["experiences"], n, "terminee")
        assert C.lancer("c", max_tours=10, dormir=lambda _s: None) == 0
        etat = C.lire_etat("c")
        assert etat["terminee_le"]
        assert len(etat["faites"]) == 4 and not etat["restantes"]
        assert deux_phases["lances"] == []


# ── 3. Resumption ────────────────────────────────────────────────────────────


class TestReprise:
    def test_elle_reprend_ou_elle_en_etait_pas_au_debut(self, deux_phases):
        """Promise D-3, and the only one that counts when a sleep lasts the night."""
        _poser_etat(deux_phases["experiences"], "t1", "terminee")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert deux_phases["lances"] == ["t2"]

        # Restart: the state is read back from disk, t1 is not replayed.
        deux_phases["lances"].clear()
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert "t1" not in deux_phases["lances"]

    def test_une_execution_interrompue_est_reprise(self, deux_phases):
        _poser_etat(
            deux_phases["experiences"], "t1", "interrompue", raison="processus mort"
        )
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert "t1" in deux_phases["lances"]

    def test_recommencer_ignore_l_etat_existant(self, deux_phases):
        C.ecrire_etat(
            "c",
            {
                **C.etat_par_defaut(C.charger("c")),
                "faites": ["t1", "t2"],
                "phase_courante": "llm",
            },
        )
        C.lancer("c", reprendre=False, max_tours=2, dormir=lambda _s: None)
        assert C.lire_etat("c")["phase_courante"] == "temoins"

    def test_l_etat_survit_a_un_json_tronque(self, deux_phases):
        (deux_phases["campagnes"] / "c").mkdir(parents=True, exist_ok=True)
        (deux_phases["campagnes"] / "c" / "etat.json").write_text(
            "{tronqué", encoding="utf-8"
        )
        assert C.lire_etat("c") is None  # says so, and starts from zero — without raising


# ── 4. Failures, stop, sleep ─────────────────────────────────────────────────


class TestEchecsEtSommeil:
    def test_deux_echecs_de_suite_alarment_et_la_campagne_continue(
        self, deux_phases, caplog, monkeypatch
    ):
        # Each launch rewrites a NEW `arretee` state: that is what makes it a failure of the
        # launch, and not the previous state (2026-09-29).
        _poser_etat(deux_phases["experiences"], "t1", "arretee", raison="clé absente")
        TestEnchainement._lanceur_qui_ecrit(deux_phases, monkeypatch, "arretee",
                                            raison="clé absente")
        import logging

        with caplog.at_level(logging.ERROR):
            C.lancer("c", max_tours=6, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "t1" in etat["echouees"]
        assert etat["echouees"]["t1"]["tentatives"] > C.TENTATIVES_MAX
        # And the next one starts anyway: a failure does not cancel the rest.
        assert "t2" in deux_phases["lances"]

    def test_un_arret_pose_PENDANT_la_boucle_sort_en_130(self, deux_phases):
        """A stop requested along the way is honoured at the next round."""

        def dormir_puis_arreter(_s):
            C.arreter("c")

        assert C.lancer("c", max_tours=5, dormir=dormir_puis_arreter) == 130
        # A single experiment was launched: the stop prevented the rest.
        assert deux_phases["lances"] == ["t1"], deux_phases["lances"]

    def test_lancer_leve_un_arret_precedent(self, deux_phases):
        C.arreter("c")
        assert C.demande_arret("c")
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        assert not C.demande_arret("c")

    def test_quota_epuise_et_RIEN_d_autre_a_faire_met_en_sommeil(self, banc):
        """Sleeping stays right when the queue behind is empty: it is the only case."""
        for n in ("t1", "t2"):
            _definir_experience(banc["experiences"], n)
        _ecrire_campagne(banc, "c", [{"nom": "p", "experiences": ["t1", "t2"]}],
                         a_travers_les_quotas=True)
        _poser_etat(banc["experiences"], "t1", "en_attente_quota")
        _poser_etat(banc["experiences"], "t2", "en_attente_quota")
        dormi: list[float] = []
        C.lancer("c", max_tours=2, dormir=dormi.append)
        etat = C.lire_etat("c")
        assert etat["sommeils"], "a sleep must be recorded, not undergone in silence"
        assert etat["sommeils"][0]["jusqu"], "the wake-up time must be written"
        assert dormi and dormi[0] > 0
        assert not etat["reportees"], "nothing to defer when nothing else can run"

    def test_quota_epuise_mais_AUTRE_CHOSE_a_faire_reporte_au_lieu_de_dormir(
        self, deux_phases
    ):
        """The defect fixed on 2026-09-15: sleeping 24 h in front of a full queue.

        A quota exhausted at one provider says nothing of the others. As long as an experiment
        can run, the sleeping one is set aside and the campaign moves on.
        """
        _a_travers_les_quotas(deux_phases)
        _poser_etat(deux_phases["experiences"], "t1", "en_attente_quota")
        _poser_etat(deux_phases["experiences"], "t2", "en_attente_quota")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert not etat["sommeils"], "the campaign must NOT sleep: llm1 and llm2 are waiting"
        assert set(etat["reportees"]) == {"t1", "t2"}
        assert etat["reportees"]["t1"]["motif"], "the reason for the deferral must be recorded"
        for exp in ("t1", "t2"):
            stop = deux_phases["experiences"] / exp / "executions" / "2026-09-14_10_00_00" / "STOP"
            assert stop.is_file(), (
                "a deferred experiment must be STOPPED, otherwise its process still sleeps "
                "and holds its key"
            )

    def test_une_seule_en_attente_ne_met_PAS_la_campagne_en_sommeil(self, deux_phases):
        """The trap: sleeping because ONE run sleeps would freeze the campaign for nothing."""
        _poser_etat(deux_phases["experiences"], "t1", "en_attente_quota")
        _poser_etat(deux_phases["experiences"], "t2", "en_cours")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert not (C.lire_etat("c")["sommeils"])

    def test_un_sommeil_trop_long_leve_une_alarme(
        self, deux_phases, monkeypatch, journal
    ):
        monkeypatch.setattr(
            C,
            "prochain_reveil",
            lambda *a, **k: ("2026-09-17T00:00:00+00:00", 40 * 3600),
        )
        _a_travers_les_quotas(deux_phases)
        _poser_etat(deux_phases["experiences"], "llm1", "en_attente_quota")
        _poser_etat(deux_phases["experiences"], "llm2", "en_attente_quota")
        _poser_etat(deux_phases["experiences"], "t1", "terminee")
        _poser_etat(deux_phases["experiences"], "t2", "terminee")
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert sum("[ALARME]" in m and "24" in m for m in journal) == 1, \
            "one alarm per sleep, not one per round"


# ── Reading ──────────────────────────────────────────────────────────────────


class TestLecture:
    def test_etat_lisible_ne_leve_pas_sur_une_campagne_jamais_lancee(self, deux_phases):
        vue = C.etat_lisible("c")
        assert vue["etat"] is None and vue["total"] == 4 and vue["faites"] == []
        assert [p["nom"] for p in vue["phases"]] == ["temoins", "llm"]

    def test_etat_lisible_compte_les_faites_par_phase(self, deux_phases):
        _poser_etat(deux_phases["experiences"], "t1", "terminee")
        vue = C.etat_lisible("c")
        assert vue["phases"][0]["faites"] == ["t1"]
        assert vue["faites"] == ["t1"]

    def test_une_experience_jamais_lancee_est_definie_pas_echouee(self, deux_phases):
        assert C.etat_experience("t1")["etat"] == "definie"

    def test_la_derniere_execution_est_la_plus_recente(self, deux_phases):
        _poser_etat(
            deux_phases["experiences"],
            "t1",
            "arretee",
            horodatage="2026-09-14_09_00_00",
        )
        _poser_etat(
            deux_phases["experiences"],
            "t1",
            "terminee",
            horodatage="2026-09-14_11_00_00",
        )
        assert C.etat_experience("t1")["etat"] == "terminee"

    def test_prochain_reveil_rend_une_date_future_et_ses_secondes(self):
        quand, secondes = C.prochain_reveil()
        assert quand.endswith("+00:00") and 0 < secondes <= 24 * 3600

# ── The launch itself ────────────────────────────────────────────────────────


class TestLancementReel:
    """These tests exercise the REAL `_lancer_experience` (only `Popen` is simulated).

    The tests above replace it with a list, which makes them fast and readable —
    but also made them blind to what happened on 2026-09-15: the campaign
    passed `--reprendre` by default, the CLI refuses it on an experiment that never
    ran, and the launch output went to `/dev/null`. The campaign stopped
    on its very first action, without a word.
    """

    @pytest.fixture
    def lancer_vrai(self, deux_phases, monkeypatch):
        """Returns `(lancement, argv_vus, kwargs_vus)` — without undoing the bench."""
        argv_vus: list[list[str]] = []
        kwargs_vus: list[dict] = []

        def _popen(argv, **kw):
            argv_vus.append(list(argv))
            kwargs_vus.append(kw)
            return None

        monkeypatch.setattr(C.subprocess, "Popen", _popen)
        return deux_phases["vrai_lancement"], argv_vus, kwargs_vus

    def test_une_experience_neuve_est_lancee_SANS_reprendre(self, lancer_vrai):
        lancement, argv, _ = lancer_vrai
        lancement("t1")
        assert "--reprendre" not in argv[-1], (
            "`lancer --reprendre` refuses when there is nothing to resume: passing it by default "
            "makes the FIRST launch of every experiment fail.")
        assert "--experience" in argv[-1] and "t1" in argv[-1]

    def test_une_experience_deja_tentee_est_lancee_AVEC_reprendre(self, deux_phases,
                                                                  lancer_vrai):
        _poser_etat(deux_phases["experiences"], "t1", "interrompue")
        lancement, argv, _ = lancer_vrai
        lancement("t1")
        assert "--reprendre" in argv[-1]

    def test_la_sortie_du_lancement_est_CONSERVEE_pas_jetee(self, deux_phases, lancer_vrai):
        """A detached launch whose output is thrown away is a launch whose fate is unknown."""
        lancement, _, kwargs = lancer_vrai
        lancement("t1")
        flux = kwargs[-1].get("stdout")
        assert flux is not None and flux != C.subprocess.DEVNULL
        journaux = list((deux_phases["experiences"] / "t1" / "lancements").glob("*.log"))
        assert journaux, "the launch's possible refusal must stay readable somewhere"

# ── The launch that never succeeds ───────────────────────────────────────────


class TestLancementQuiNAboutitPas:
    """A launch refused by the platform never writes a state. What does the campaign do?

    It relaunched it ENDLESSLY: measured on 2026-09-15, 178 relaunches in six hours on the
    random forest control, whose artefact is refused by `decideur_modele` (rule R7 of
    ticket 044). The controls phase was never closed, and the six LLM arms were never
    reached — a whole night lost, without a single alarm.

    A silent blockage is worth less than a declared failure: at least the failure lets
    the rest through.
    """

    def test_un_lancement_muet_finit_par_etre_declare_en_echec(self, deux_phases, monkeypatch,
                                                               journal):
        # The grace period is brought down to zero: we test the logic, not patience.
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        C.lancer("c", max_tours=8, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "t1" in etat["echouees"], "a launch that never succeeds must be declared"
        assert "sans jamais écrire d'état" in etat["echouees"]["t1"]["motif"]
        assert sum("[ALARME]" in m and "t1" in m for m in journal) == 1

    def test_et_la_campagne_passe_a_la_SUIVANTE(self, deux_phases, monkeypatch):
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        C.lancer("c", max_tours=8, dormir=lambda _s: None)
        assert "t2" in deux_phases["lances"], (
            "a failure must not block the phase: this is exactly what cost the night "
            "of 2026-09-15.")

    def test_le_nombre_de_relances_est_BORNE(self, deux_phases, monkeypatch):
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        C.lancer("c", max_tours=20, dormir=lambda _s: None)
        relances_t1 = deux_phases["lances"].count("t1")
        # The bound changed on 2026-09-15 with the retrieval pass: a failed experiment
        # is entitled to ONE more relaunch per pass, never to a counter reset. The total
        # therefore stays bounded and computable, which is all this test protects.
        borne = C.TENTATIVES_MAX + 1 + C.REPECHAGES_MAX
        assert relances_t1 <= borne, (
            f"t1 relaunched {relances_t1} times — the bound is {borne} "
            f"({C.TENTATIVES_MAX} attempts + 1, plus {C.REPECHAGES_MAX} retrievals)")


# ── End-of-campaign retrieval ────────────────────────────────────────────────


class TestRepechage:
    """Nothing is abandoned without a second chance (2026-09-15).

    Before, an experiment set aside left observation and never came back, even
    when relaunching the campaign: its failure was written in the state, and the state excluded what
    appeared in it. A quota exhausted mid-campaign therefore cost the experiment for good.
    """

    def test_une_reportee_repasse_a_la_fin(self, deux_phases):
        _a_travers_les_quotas(deux_phases)
        _poser_etat(deux_phases["experiences"], "t1", "en_attente_quota")
        _poser_etat(deux_phases["experiences"], "t2", "terminee")
        _poser_etat(deux_phases["experiences"], "llm1", "terminee")
        _poser_etat(deux_phases["experiences"], "llm2", "terminee")
        C.lancer("c", max_tours=6, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert etat["repechages"] >= 1, "the end of the campaign must trigger a retrieval"
        assert "t1" not in etat["reportees"], (
            "the deferred one must be TAKEN OUT of the list to be replayed, not only "
            "counted"
        )
        assert etat["phase_courante"] == "temoins", (
            "the retrieval must go back to the phase holding the deferred one, not stay at the end"
        )
        # The relaunch itself is not observable here: the fixture writes a frozen
        # `etat.json`, so t1 reads back as `en_attente_quota` even after its STOP file. On a
        # real run, the runner sees the STOP, closes as `arretee`, and the standard resumption
        # relaunches it — this is the path `TestEchecsEtSommeil` already covers.

    def test_le_repechage_est_borne(self, deux_phases, monkeypatch):
        """A campaign looping endlessly is not visible: the number of passes is fixed."""
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        _poser_etat(deux_phases["experiences"], "t2", "terminee")
        _poser_etat(deux_phases["experiences"], "llm1", "terminee")
        _poser_etat(deux_phases["experiences"], "llm2", "terminee")
        _poser_etat(deux_phases["experiences"], "t1", "arretee", raison="cassée")
        C.lancer("c", max_tours=60, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert etat["repechages"] <= C.REPECHAGES_MAX
        assert etat["terminee_le"], "the campaign must finish, even with a broken experiment"

    def test_une_reportee_qui_ne_revient_pas_leve_une_alarme(self, deux_phases, journal):
        """Silence about an arm never played would be the worst defect of this mechanism."""
        monkeypatch_cible = deux_phases["experiences"]
        for n in ("t2", "llm1", "llm2"):
            _poser_etat(monkeypatch_cible, n, "terminee")
        _poser_etat(monkeypatch_cible, "t1", "en_attente_quota")
        C.lancer("c", max_tours=40, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        if etat["reportees"]:
            assert any("[ALARME]" in m and "REPORTÉE" in m for m in journal), (
                "an experiment left behind must say so loudly"
            )


# ── Dedicated launcher ───────────────────────────────────────────────────────


class TestLanceurDedie:
    """Some experiments do not go through `experiences lancer`.

    The random forest control is one: rule R7 of ticket 044 keeps it out of
    `decideur_modele.FAMILLES` so that it does not become an arbiter, and its dedicated launcher
    registers its family for the duration of its own process. The campaign must be able to say so,
    rather than stumble on it.
    """

    def test_un_lanceur_declare_est_utilise_tel_quel(self, banc, monkeypatch):
        _definir_experience(banc["experiences"], "special")
        _ecrire_campagne(banc, "c", [{"nom": "p", "experiences": ["special"]}],
                         lanceurs={"special": ["/bin/echo", "--experience", "{exp}"]})
        argv_vus: list[list[str]] = []
        monkeypatch.setattr(C.subprocess, "Popen",
                            lambda argv, **kw: argv_vus.append(list(argv)))
        banc["vrai_lancement"]("special", C.charger("c").lanceurs)
        assert argv_vus[-1] == ["/bin/echo", "--experience", "special"], (
            "`{exp}` must be replaced by the experiment name")

    def test_sans_lanceur_declare_on_passe_par_la_plateforme(self, deux_phases, monkeypatch):
        argv_vus: list[list[str]] = []
        monkeypatch.setattr(C.subprocess, "Popen",
                            lambda argv, **kw: argv_vus.append(list(argv)))
        deux_phases["vrai_lancement"]("t1", C.charger("c").lanceurs)
        assert "experiences" in argv_vus[-1] and "lancer" in argv_vus[-1]

    def test_un_lanceur_pour_une_experience_absente_est_refuse(self, banc):
        _definir_experience(banc["experiences"], "x")
        _ecrire_campagne(banc, "c", [{"nom": "p", "experiences": ["x"]}],
                         lanceurs={"absente": ["/bin/echo"]})
        with pytest.raises(C.CampagneInvalide, match="absente"):
            C.charger("c")


# ── 9. The pause that froze the campaign ─────────────────────────────────────


class TestPauseQuiBloque:
    """Measured on 2026-09-16: `en_pause` fell into the "in flight" catch-all.

    The experiment was then neither resumed (the state is not in `ETATS_REPRENABLES`), nor
    put to sleep (it is not a quota), nor relaunched (the grace period only watches the
    launches of the current process). The campaign spun idle, for six hours, without
    launching anything behind — and without saying anything, since it only declares itself alive one round
    in twenty.
    """

    def test_une_pause_de_chien_de_garde_est_reprise(self, deux_phases):
        _pause_chien_de_garde(deux_phases["experiences"], "t1")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert "t1" in deux_phases["lances"]

    def test_une_execution_incomplete_sans_interruption_est_reprise(self, deux_phases):
        """The runner's third path: incomplete stop, pause never requested.

        Its own message says "reprendre" — the campaign still has to do it.
        """
        _poser_etat(deux_phases["experiences"], "t1", "en_pause",
                    raison="incomplète : 12/3299 archivées — reprendre")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert "t1" in deux_phases["lances"]

    def test_un_execution_yaml_illisible_penche_vers_la_reprise(self, deux_phases):
        """An unbounded blockage is worse than one resumption too many, which `TENTATIVES_MAX` caps."""
        _pause_chien_de_garde(deux_phases["experiences"], "t1")
        d = deux_phases["experiences"] / "t1" / "executions" / "2026-09-14_10_00_00"
        (d / "execution.yaml").write_text("{ceci n'est pas: du yaml: valide", encoding="utf-8")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert "t1" in deux_phases["lances"]

    def test_une_pause_demandee_par_un_humain_n_est_pas_reprise(self, deux_phases):
        """The fix must not go from one excess to the other.

        Someone paused this run; the campaign does not relaunch it behind their
        back, nor does it skip over it to launch the next one.
        """
        _pause_humaine(deux_phases["experiences"], "t1")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert deux_phases["lances"] == []

    def test_le_scenario_reel_la_phase_llm_ne_reste_pas_a_zero_lancement(self, banc):
        """An undergone `en_pause` ahead of a `definie`: this is exactly the 16/09 blockage."""
        for n in ("bloquee", "suivante"):
            _definir_experience(banc["experiences"], n)
        _ecrire_campagne(banc, "c",
                         [{"nom": "llm", "experiences": ["bloquee", "suivante"]}])
        _pause_chien_de_garde(banc["experiences"], "bloquee")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert banc["lances"], "the campaign must launch something, not spin idle"


# ── 10. The "in flight, but frozen" alarm ────────────────────────────────────


class TestEnVolFige:
    """The missing safeguard: the campaign SAYS it is waiting, and for whom."""

    def _pause_humaine_ancienne(self, banc, nom: str) -> None:
        vieux = (
            datetime.now(timezone.utc) - timedelta(seconds=C.EN_VOL_FIGE_S + 600)
        ).isoformat()
        _pause_humaine(banc["experiences"], nom, maj=vieux)

    def test_une_experience_en_vol_figee_leve_une_alarme(self, deux_phases, journal):
        self._pause_humaine_ancienne(deux_phases, "t1")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        alarmes = [m for m in journal if "[ALARME]" in m and "EN VOL" in m]
        assert alarmes, "a silent blockage is not visible — it must announce itself"
        assert "t1" in alarmes[0]
        assert "experience-reprendre" in alarmes[0], "an alarm says what to do"

    def test_l_alarme_ne_se_repete_pas_a_chaque_tour(self, deux_phases, journal):
        """Rising edge: an alarm twice a minute drowns the log it sheds light on."""
        self._pause_humaine_ancienne(deux_phases, "t1")
        C.lancer("c", max_tours=8, dormir=lambda _s: None)
        assert len([m for m in journal if "[ALARME]" in m and "EN VOL" in m]) == 1

    def test_une_experience_en_vol_recente_n_alarme_pas(self, deux_phases, journal):
        _pause_humaine(deux_phases["experiences"], "t1",
                       maj=datetime.now(timezone.utc).isoformat())
        C.lancer("c", max_tours=4, dormir=lambda _s: None)
        assert not [m for m in journal if "[ALARME]" in m and "EN VOL" in m]

    def test_l_alarme_ne_touche_a_rien(self, deux_phases, journal):
        """It logs. It does not resume, does not declare a failure, does not skip."""
        self._pause_humaine_ancienne(deux_phases, "t1")
        C.lancer("c", max_tours=4, dormir=lambda _s: None)
        assert deux_phases["lances"] == []
        assert C.lire_etat("c")["echouees"] == {}

    def test_une_experience_qui_dort_sur_son_quota_n_alarme_pas(self, deux_phases, journal):
        """Its state does not move either — but it is an understood wait, already stated.

        The batch sleep records it, and the 26 h alarm covers it. Raising an alarm here would
        pass a normal wait off as an anomaly.
        """
        vieux = (
            datetime.now(timezone.utc) - timedelta(seconds=C.EN_VOL_FIGE_S + 600)
        ).isoformat()
        _poser_etat(deux_phases["experiences"], "t1", "en_attente_quota", maj=vieux)
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert not [m for m in journal if "[ALARME]" in m and "EN VOL" in m]


# ── 5. A single campaign at a time ───────────────────────────────────────────


class TestDoubleLancement:
    """On 2026-09-16, two `campagne-lancer bascule_anglaise_v6` ran in parallel
    for twenty minutes: they wrote the same `etat.json` each in turn, marked two
    arms "stop requested", and declared the campaign finished with an arm never launched."""

    @staticmethod
    def _dire_que_le_pid_tourne(monkeypatch, ligne: str | None) -> None:
        monkeypatch.setattr(C, "_ligne_de_commande", lambda pid: ligne)

    def test_un_second_lancement_REFUSE_et_ne_touche_a_rien(
        self, deux_phases, monkeypatch, journal
    ):
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        avant = C.lire_etat("c")
        # The first process is no longer the caller: we give it a borrowed pid,
        # and we say that this pid does carry a campaign command line.
        avant["pid"] = os.getpid() + 1
        C.ecrire_etat("c", avant)
        self._dire_que_le_pid_tourne(
            monkeypatch, "python -m experiences campagne-lancer --nom c")

        deux_phases["lances"].clear()
        assert C.lancer("c", max_tours=5, dormir=lambda _s: None) == C.CODE_DEJA_EN_VOL
        assert deux_phases["lances"] == []
        alarme = [l for l in journal if "[ALARME]" in l and "TOURNE DÉJÀ" in l]
        assert alarme, journal
        # It ALSO says what to do: observe, or stop the one running.
        assert "campagne-etat" in alarme[0] and "campagne-arreter" in alarme[0]

    def test_le_refus_ne_LEVE_PAS_le_drapeau_d_arret_du_premier(
        self, deux_phases, monkeypatch
    ):
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        etat["pid"] = os.getpid() + 1
        C.ecrire_etat("c", etat)
        C.arreter("c")
        self._dire_que_le_pid_tourne(monkeypatch, "campagne-lancer --nom c")

        C.lancer("c", max_tours=5, dormir=lambda _s: None)
        assert C.demande_arret("c"), "the second launch erased the first one's STOP"

    def test_un_pid_MORT_ne_bloque_pas_la_reprise(self, deux_phases, monkeypatch):
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        etat["pid"] = os.getpid() + 1
        C.ecrire_etat("c", etat)
        self._dire_que_le_pid_tourne(monkeypatch, None)  # `ps` no longer knows this pid

        deux_phases["lances"].clear()
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert deux_phases["lances"], "a dead campaign must be able to resume"

    def test_un_pid_RECYCLE_ne_bloque_pas_non_plus(self, deux_phases, monkeypatch):
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        etat["pid"] = os.getpid() + 1
        C.ecrire_etat("c", etat)
        # Same pid, but it is no longer a campaign: the system reassigned it.
        self._dire_que_le_pid_tourne(monkeypatch, "/usr/bin/vim notes.md")

        deux_phases["lances"].clear()
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert deux_phases["lances"]

    def test_une_AUTRE_campagne_en_vol_ne_bloque_pas_celle_ci(self, deux_phases, monkeypatch):
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        etat["pid"] = os.getpid() + 1
        C.ecrire_etat("c", etat)
        self._dire_que_le_pid_tourne(monkeypatch, "campagne-lancer --nom une_autre")

        deux_phases["lances"].clear()
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert deux_phases["lances"]

    def test_un_ps_MUET_laisse_passer_plutot_que_de_bloquer(self, deux_phases, monkeypatch):
        """Fail-open: a failing `ps` must not prevent a campaign from starting."""
        def _ps_casse(*_a, **_k):
            raise OSError("ps introuvable")
        monkeypatch.setattr(C.subprocess, "run", _ps_casse)
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        etat["pid"] = os.getpid() + 1
        C.ecrire_etat("c", etat)

        deux_phases["lances"].clear()
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert deux_phases["lances"]


# ── Launch refusal: a quota shortage is not a breakdown (2026-09-17) ───────────────────


class TestRefusDeLancement:
    """The launcher sometimes refuses to create the run. The campaign must know WHY.

    On 2026-09-16, four arms were declared "lancée 3 fois sans jamais écrire d'état"
    whereas their only fault was waiting for a quota renewal. The campaign launches
    in the background: it reads neither the launcher output nor its return code. The marker
    dropped next to the launch log is the only channel left.
    """

    def _marqueur(self, racine, nom, classe, motifs):
        base = racine / nom / "lancements"
        base.mkdir(parents=True, exist_ok=True)
        (base / f"2026-09-17_10_00_00{C.REFUS.SUFFIXE_MARQUEUR}").write_text(
            json.dumps({"classe": classe,
                        "reportable": C.REFUS.est_reportable(classe),
                        "motifs": motifs, "le": "2026-09-17T10:00:00+00:00"},
                       ensure_ascii=False),
            encoding="utf-8")

    def test_penurie_de_quota_reporte_sans_decompter_de_tentative(
        self, deux_phases, monkeypatch
    ):
        _a_travers_les_quotas(deux_phases)
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        self._marqueur(deux_phases["experiences"], "t1", C.REFUS.QUOTA_EPUISE,
                       ["les 2 instance(s) qui servent 'x' sont momentanément épuisées"])
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "t1" in etat["reportees"], "a quota shortage is DEFERRED"
        assert "t1" not in etat["echouees"], "and does not count as a breakdown"
        assert "momentanément épuisées" in etat["reportees"]["t1"]["motif"], (
            "the launcher's reason must survive up to the campaign state"
        )

    def test_charge_hors_quota_echoue_en_disant_pourquoi(self, deux_phases, monkeypatch):
        """Waiting does not fix a load above the daily quota: it is a failure,
        but a failure that names its cause instead of speaking of a never-written state."""
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        self._marqueur(deux_phases["experiences"], "t1", C.REFUS.QUOTA_INSUFFISANT,
                       ["quota journalier hors d'atteinte : ~10 jours de quota"])
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "t1" in etat["echouees"]
        assert "t1" not in etat["reportees"], "deferring would be an endless loop"
        motif = etat["echouees"]["t1"]["motif"]
        assert "hors d'atteinte" in motif and "sans jamais écrire" not in motif

    def test_sans_marqueur_le_comportement_ne_change_pas(self, deux_phases, monkeypatch):
        """Non-regression: a silent launch stays a breakdown after TENTATIVES_MAX."""
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        C.lancer("c", max_tours=20, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "t1" in etat["echouees"]
        assert "sans jamais écrire d'état" in etat["echouees"]["t1"]["motif"]

    def test_un_marqueur_perime_est_ignore(self, deux_phases, monkeypatch):
        """Yesterday's marker says nothing of today's launch: if it predates the
        last launch log, it does not count."""
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        self._marqueur(deux_phases["experiences"], "t1", C.REFUS.QUOTA_EPUISE,
                       ["momentanément épuisées"])
        base = deux_phases["experiences"] / "t1" / "lancements"
        (base / "2026-09-18_09_00_00.log").write_text("lancement postérieur\n", encoding="utf-8")
        C.lancer("c", max_tours=20, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "t1" not in etat["reportees"], "the marker is older than the log"
        assert "t1" in etat["echouees"]


class TestClassementDesRefus:
    def test_les_trois_classes(self):
        from experiences import refus as R
        assert R.classer(["… sont momentanément épuisées — …"]) == R.QUOTA_EPUISE
        assert R.classer(["quota journalier hors d'atteinte : ~10 jours de quota"]) == R.QUOTA_INSUFFISANT
        assert R.classer(["jeu périmé"]) == R.DEFINITIF
        assert R.classer([]) == R.DEFINITIF

    def test_le_plus_grave_l_emporte(self):
        """A definitive refusal together with a shortage stays definitive: waiting does not lift it."""
        from experiences import refus as R
        assert R.classer(["momentanément épuisées", "jeu périmé"]) == R.DEFINITIF
        assert R.est_reportable(R.QUOTA_EPUISE) is True
        assert R.est_reportable(R.QUOTA_INSUFFISANT) is False


# ── Experiments sorted by families (2026-09-28) ──────────────────────────────


class TestExperiencesRangeesParFamilles:
    """Experiments live under `regime_nominal/<jeu>/<exp>/`, not at the root.

    On 2026-09-26, the `pilotage_gemini` campaign looked for `executions/` and wrote
    `lancements/` at the root, where nothing exists. It never saw the runs of its six
    arms, declared them "lancée 3 fois sans jamais écrire d'état" — yet proexp32 had
    run — and left six orphan directories that only contained launch
    logs. The bench above defines everything flat: it could not see it.
    """

    FAMILLE = Path("regime_nominal") / "jeu_x"

    @pytest.fixture
    def rangees(self, banc):
        for n in ("t1", "t2"):
            _definir_experience(banc["experiences"] / self.FAMILLE, n)
        _ecrire_campagne(banc, "c", [{"nom": "temoins", "experiences": ["t1", "t2"]}])
        return banc

    def _orphelins(self, racine: Path) -> list[str]:
        return sorted(p.name for p in racine.iterdir() if p.name != "regime_nominal")

    def test_la_campagne_accepte_une_experience_rangee(self, rangees):
        assert list(C.charger("c").toutes) == ["t1", "t2"]

    def test_l_etat_est_lu_dans_la_famille(self, rangees):
        racine = rangees["experiences"]
        _poser_etat(racine / self.FAMILLE, "t1", "terminee")
        infos = C.etat_experience("t1")
        assert infos["etat"] == "terminee", "the campaign only saw \"definie\""
        assert Path(infos["dossier"]).parent.parent == racine / self.FAMILLE / "t1"

    def test_une_campagne_ne_relance_pas_ce_qui_est_fait(self, rangees):
        racine = rangees["experiences"]
        for n in ("t1", "t2"):
            _poser_etat(racine / self.FAMILLE, n, "terminee")
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert rangees["lances"] == [], "finished arms relaunched in a loop"
        assert C.lire_etat("c")["faites"] == ["t1", "t2"]

    def test_le_journal_de_lancement_va_dans_la_famille(self, rangees):
        racine = rangees["experiences"]
        sortie = C.journal_lancement("t1")
        assert sortie.parent == racine / self.FAMILLE / "t1" / "lancements"
        assert self._orphelins(racine) == [], "no orphan directory at the root"

    def test_le_lancement_reel_reprend_et_dit_ou_est_sa_sortie(self, rangees, monkeypatch,
                                                                journal):
        racine = rangees["experiences"]
        _poser_etat(racine / self.FAMILLE, "t1", "interrompue")
        argv_vus: list[list[str]] = []
        monkeypatch.setattr(C.subprocess, "Popen",
                            lambda argv, **kw: argv_vus.append(list(argv)))
        rangees["vrai_lancement"]("t1")
        assert "--reprendre" in argv_vus[-1], "an interrupted run is RESUMED"
        attendu = (self.FAMILLE / "t1" / "lancements").as_posix()
        assert any(attendu in m for m in journal), "the log must name the family"
        assert self._orphelins(racine) == []

    def test_le_marqueur_de_refus_est_lu_dans_la_famille(self, rangees, monkeypatch):
        _a_travers_les_quotas(rangees)
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        base = rangees["experiences"] / self.FAMILLE / "t1" / "lancements"
        base.mkdir(parents=True)
        (base / f"2026-09-17_10_00_00{C.REFUS.SUFFIXE_MARQUEUR}").write_text(
            json.dumps({"classe": C.REFUS.QUOTA_EPUISE, "reportable": True,
                        "motifs": ["momentanément épuisées"],
                        "le": "2026-09-17T10:00:00+00:00"}), encoding="utf-8")
        C.lancer("c", max_tours=3, dormir=lambda _s: None)
        assert "t1" in C.lire_etat("c")["reportees"]

    def test_le_motif_d_echec_designe_le_vrai_dossier(self, rangees, monkeypatch):
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        C.lancer("c", max_tours=20, dormir=lambda _s: None)
        motif = C.lire_etat("c")["echouees"]["t1"]["motif"]
        attendu = rangees["experiences"] / self.FAMILLE / "t1" / "lancements"
        assert str(attendu) in motif

    def test_le_marqueur_de_la_cli_va_dans_la_famille(self, rangees):
        from experiences import cli as CLI

        CLI._ecrire_marqueur_refus("t1", ["les 2 instance(s) sont momentanément épuisées"])
        base = rangees["experiences"] / self.FAMILLE / "t1" / "lancements"
        assert list(base.glob(f"*{C.REFUS.SUFFIXE_MARQUEUR}")), "marker missing from the family"
        assert self._orphelins(rangees["experiences"]) == []

    def test_une_experience_introuvable_garde_son_journal_a_plat_et_le_dit(self, banc,
                                                                           journal):
        sortie = C.journal_lancement("fantome")
        assert sortie.parent == banc["experiences"] / "fantome" / "lancements"
        assert any("fantome" in m and "introuvable" in m for m in journal)


# ── 11. Chaining, without sleeping or relaunching (2026-09-29) ──────────────


class TestEnchainement:
    """The defect of the night of 28 to 29/09, and the behaviour wanted by the author.

    The `papier_court_nominal_20260928` campaign finished at 01:56 without having played anything.
    An old request from the key queue, which the campaign itself promoted, held the
    two Google keys; the campaign's experiments queued up behind it, and
    the campaign counted each queuing as a failure. Resumptions, it judged them on
    their state of the day before: three "failures" in one minute, without any having run.

    The wanted behaviour (the author, 2026-09-29): chain experiments one by one; an
    experiment stopped on quota is not relaunched; keys held by another experiment leave it
    "non jouée"; neither sleep nor retrieval — at the end of the list, the campaign
    finishes and hands back control.
    """

    @staticmethod
    def _campagne(banc, noms, **extra):
        for n in noms:
            _definir_experience(banc["experiences"], n)
        _ecrire_campagne(banc, "c", [{"nom": "p", "experiences": list(noms)}], **extra)

    @staticmethod
    def _lanceur_qui_ecrit(banc, monkeypatch, etat, raison=None):
        """A launcher that does what the real one does: it rewrites the state, dated now."""

        def _lancer(exp, lanceurs=None, **kw):
            banc["lances"].append(exp)
            base = banc["experiences"] / exp / "executions"
            existantes = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
            horodatage = existantes[-1].name if existantes else "2026-09-29_08_00_00"
            _poser_etat(banc["experiences"], exp, etat, horodatage=horodatage, raison=raison,
                        maj=datetime.now(timezone.utc).isoformat())

        monkeypatch.setattr(C, "_lancer_experience", _lancer)

    @staticmethod
    def _cles_tenues(racine: Path, par: str = "autre_exp") -> None:
        tenue = {"execution": str(racine / par / "executions" / "2026-09-29_06_00_00"),
                 "exp": par, "pid": 1, "depuis": "2026-09-29T06:00:00+00:00"}
        (racine / ".reservations.json").write_text(
            json.dumps({"google_key1": tenue, "google_key2": tenue}), encoding="utf-8")

    @staticmethod
    def _lanceur_mis_en_file(banc, monkeypatch):
        """What `lancer` does when the keys are taken: a queued request, no state."""
        from experiences import file as F

        def _lancer(exp, lanceurs=None, **kw):
            banc["lances"].append(exp)
            F.sauver([*[e for e in F.charger() if e.get("exp") != exp],
                      F.entree(exp, {"google_key1", "google_key2"}, {"reprendre": False})])

        monkeypatch.setattr(C, "_lancer_experience", _lancer)

    def test_une_reprise_n_est_pas_jugee_sur_son_etat_de_la_veille(self, banc):
        """The core of the defect: the `epuisee` state of 28/09 counted as the failure of 29/09."""
        self._campagne(banc, ["a", "b"])
        _poser_etat(banc["experiences"], "a", "epuisee", raison="quota du 28/09")
        C.lancer("c", max_tours=4, dormir=lambda _s: None)
        assert banc["lances"] == ["a"], (
            f"launched once then awaited, not relaunched at each round: {banc['lances']}")
        assert "a" not in C.lire_etat("c")["echouees"]

    def test_une_experience_epuisee_apres_son_lancement_n_est_pas_relancee(
        self, banc, monkeypatch
    ):
        self._campagne(banc, ["a", "b"])
        self._lanceur_qui_ecrit(banc, monkeypatch, "epuisee",
                                raison="google_gemini35_key1 : 500/500 requêtes/jour")
        C.lancer("c", max_tours=6, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert banc["lances"].count("a") == 1, banc["lances"]
        assert "a" in etat["echouees"]
        assert "non terminée" in etat["echouees"]["a"]["motif"]
        assert "500/500" in etat["echouees"]["a"]["motif"], "le motif du run doit survivre"
        assert "b" in banc["lances"], "the campaign moves on to the next one"

    def test_des_cles_tenues_laissent_non_jouee_et_retirent_la_demande_de_la_file(
        self, banc, monkeypatch
    ):
        from experiences import reservations as R

        self._campagne(banc, ["a", "b"])
        self._cles_tenues(banc["experiences"])
        self._lanceur_mis_en_file(banc, monkeypatch)
        C.lancer("c", max_tours=6, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert banc["lances"].count("a") == 1, banc["lances"]
        motif = etat["echouees"]["a"]["motif"]
        assert "non jouée" in motif and "autre_exp" in motif, motif
        assert "sans jamais écrire" not in motif, "a queuing is not a breakdown"
        assert R.lister_file() == [], (
            "a request left in the queue would be started later, outside the campaign")
        assert "b" in banc["lances"]

    def test_une_seule_a_la_fois_dans_l_ordre_du_yaml(self, banc):
        self._campagne(banc, ["a", "b", "d"])
        _poser_etat(banc["experiences"], "a", "epuisee")
        _poser_etat(banc["experiences"], "b", "interrompue")
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert banc["lances"] == ["a"], (
            f"two resumptions launched together fight over the same keys: {banc['lances']}")

    def test_un_lancement_qui_echoue_vraiment_est_relance_puis_abandonne(
        self, banc, monkeypatch
    ):
        """Counter-test: a failure that is not a quota keeps its bounded relaunches."""
        self._campagne(banc, ["a", "b"])
        self._lanceur_qui_ecrit(banc, monkeypatch, "interrompue", raison="processus mort")
        C.lancer("c", max_tours=10, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert banc["lances"].count("a") == C.TENTATIVES_MAX + 1, banc["lances"]
        assert "a" in etat["echouees"]

    def test_ni_sommeil_ni_repechage_la_campagne_se_termine(self, banc, monkeypatch):
        self._campagne(banc, ["a"])
        self._lanceur_qui_ecrit(banc, monkeypatch, "epuisee", raison="quota")
        dormi: list[float] = []
        code = C.lancer("c", max_tours=12, dormir=dormi.append)
        etat = C.lire_etat("c")
        assert etat["terminee_le"], "at the end of the list, the campaign finishes"
        assert etat["repechages"] == 0 and etat["sommeils"] == []
        assert banc["lances"] == ["a"]
        assert code == 1, "an unfinished experiment shows in the exit code"
        assert all(s <= C.INTERVALLE_S for s in dormi), "no wait for a quota window"

    def test_la_campagne_ne_demarre_pas_la_file_des_autres(self, banc, monkeypatch):
        """It was the campaign itself that relaunched go123's stale request at 01:38."""
        self._campagne(banc, ["a"])
        appels: list[str] = []
        monkeypatch.setattr(C, "_tour_ordonnanceur", lambda: appels.append("tour"))
        monkeypatch.setattr(C, "_reconcilier_cles", lambda: appels.append("reconcilier"),
                            raising=False)
        C.lancer("c", max_tours=2, dormir=lambda _s: None)
        assert "tour" not in appels, "promoting the queue starts experiments outside the campaign"
        assert "reconcilier" in appels, "the keys of dead runs must be released"

    def test_l_ancien_comportement_reste_disponible_par_le_yaml(self, banc, monkeypatch):
        self._campagne(banc, ["a"], a_travers_les_quotas=True)
        assert C.charger("c").a_travers_les_quotas is True
        appels: list[str] = []
        monkeypatch.setattr(C, "_tour_ordonnanceur", lambda: appels.append("tour"))
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        assert appels == ["tour"]

    def test_un_refus_pour_quota_au_lancement_laisse_non_jouee(self, banc, monkeypatch):
        monkeypatch.setattr(C, "DELAI_ECRITURE_ETAT_S", 0.0)
        self._campagne(banc, ["a"])
        base = banc["experiences"] / "a" / "lancements"
        base.mkdir(parents=True)
        (base / f"2026-09-17_10_00_00{C.REFUS.SUFFIXE_MARQUEUR}").write_text(
            json.dumps({"classe": C.REFUS.QUOTA_EPUISE, "reportable": True,
                        "motifs": ["les 2 instance(s) sont momentanément épuisées"],
                        "le": "2026-09-17T10:00:00+00:00"}), encoding="utf-8")
        C.lancer("c", max_tours=4, dormir=lambda _s: None)
        etat = C.lire_etat("c")
        assert "a" not in etat["reportees"], "nothing is deferred: there is no retrieval"
        assert "momentanément épuisées" in etat["echouees"]["a"]["motif"]

    def test_au_demarrage_des_cles_tenues_sont_signalees(self, banc, journal):
        self._campagne(banc, ["a"])
        self._cles_tenues(banc["experiences"])
        C.lancer("c", max_tours=1, dormir=lambda _s: None)
        assert any("autre_exp" in m and "tenue" in m for m in journal), journal[-5:]

    def test_le_lancement_dit_explicitement_s_il_attend_la_fenetre(self, banc, monkeypatch):
        """The parser's default is not the one its doc announced: stop depending on it."""
        self._campagne(banc, ["a"])
        argv: list[list[str]] = []
        monkeypatch.setattr(C.subprocess, "Popen", lambda a, **k: argv.append(list(a)))
        banc["vrai_lancement"]("a")
        assert "--ne-pas-attendre-fenetre" in argv[-1]
        banc["vrai_lancement"]("a", attendre_fenetre=True)
        assert "--attendre-fenetre" in argv[-1]
        assert "--ne-pas-attendre-fenetre" not in argv[-1]
