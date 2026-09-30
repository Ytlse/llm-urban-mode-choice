"""A truncated moves log never yields a score, and a resumption regenerates it.

What these tests lock down fits in one sentence: `moves.csv` is the SUBSTRATE of the composite score,
and nobody ever compared its count with that of the archived decisions. Run
`2026-09-12_11_24_28` was thus published at a 5.35 composite on 274 lines when its archive
held 3,299 — the figure went through three documents before being recognised as wrong. Rescored
on the rebuilt log, it is 6.17.

The two halves of the remedy are tested separately, because they fail separately:

- the SCORER refuses (R24) — even with regeneration forgotten, no figure comes out of an
  incomplete log; this is the guard that holds whatever happens elsewhere;
- RESUMPTION regenerates — the log becomes complete again, so the score becomes possible again.

The threshold (2 %) is not a whim. Sweep of the repository's 38 runs on
2026-09-15: a healthy run holds 3,161 lines for 3,151 to 3,155 decided, i.e. a
deficit that is ALWAYS negative (the log additionally holds the `sans_solution` lines, which `decides`
excludes); the faulty run is at +91.3 %. The two regimes are a factor of 45 apart.
"""

import asyncio
import csv
import json
import shutil
from pathlib import Path

import pytest
from experiences import formule as F
from experiences import journal as JN
from experiences import score as S
from experiences.archive import METHODE_INEXPLOITABLE, METHODE_NON_COUVERT

# The synthetic bench of the simulator-free run: sealed population, closed set, isolated roots. Imported
# as is — a second bench would drift from the first without anything saying so.
from tests.test_offline_execution import (  # noqa: F401
    _exp,
    _lancer,
    banc,
    sans_anticipation,
)

REPO = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.usefixtures("sans_anticipation")


# ── Real substrate: a terminated run from the repository ────────────────────


def _exec_reelle() -> Path | None:
    """A terminated, complete run, with its decision archive and its score.

    Hard-coded, the reference would disappear with the experiment that holds it. We require
    `decisions.jsonl`: without it, regeneration has nothing to read back and the fidelity tests
    would pass while measuring nothing.
    """
    racine = REPO / "data" / "experiences"
    for d in JN.executions(racine):
        if not (d / "decisions.jsonl").is_file() or not (d / "scores.json").is_file():
            continue
        constat = JN.verifier(d)
        if constat["etat"] == "terminee" and constat["complet"] is True:
            return d
    return None


EXEC_REELLE = _exec_reelle()

sans_substrat = pytest.mark.skipif(
    EXEC_REELLE is None,
    reason="no terminated, complete and archived run in data/experiences",
)


@pytest.fixture
def copie_reelle(tmp_path):
    dst = tmp_path / EXEC_REELLE.name
    shutil.copytree(EXEC_REELLE, dst)
    return dst


@pytest.fixture(scope="module")
def registre():
    return F.charger()


def _lignes(chemin: Path) -> list[dict]:
    with Path(chemin).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _tronquer(dossier: Path, fraction: float) -> int:
    """Keeps only `fraction` of the log lines, header kept. Returns the kept count."""
    chemin = dossier / "moves.csv"
    lignes = _lignes(chemin)
    gardees = lignes[: max(1, int(len(lignes) * fraction))]
    with chemin.open("w", encoding="utf-8", newline="") as fh:
        redacteur = csv.DictWriter(fh, fieldnames=list(lignes[0].keys()))
        redacteur.writeheader()
        redacteur.writerows(gardees)
    return len(gardees)


@pytest.fixture(autouse=True)
def alarme_desarmee():
    """The rising edge is a PROCESS state: two tests would share it otherwise.

    Without this cleanup, the rising-edge test would pass or not depending on execution order —
    a green guard for a wrong reason, which is precisely what this file fights.
    """
    S._JOURNAUX_SIGNALES.clear()
    yield
    S._JOURNAUX_SIGNALES.clear()


# ═══════════ The scorer refuses an incomplete log (R24) ═══════════


@sans_substrat
def test_un_journal_tronque_a_10_pour_cent_ne_produit_aucun_score(
    copie_reelle, registre
):
    """No `scores.json`, and the refusal is a named exception — not an approximate score."""
    (copie_reelle / "scores.json").unlink(missing_ok=True)
    garde = _tronquer(copie_reelle, 0.10)
    with pytest.raises(S.JournalIncomplet) as exc:
        S.calculer(copie_reelle, registre.reference)
    assert str(garde) in str(exc.value), "the refusal must cite the log count"
    assert S.score_execution(copie_reelle, registre.reference, registre) is None
    assert not (copie_reelle / "scores.json").exists()


@sans_substrat
def test_l_alarme_part_une_seule_fois_par_execution(copie_reelle, registre, caplog):
    """Rising edge: `score --toutes` must not replay the same ERROR on every pass.

    A repeated alarm drowns `make error` in its own repetition, and that log is the one
    read to find out what is wrong.
    """
    import logging

    from loguru import logger

    (copie_reelle / "scores.json").unlink(missing_ok=True)
    _tronquer(copie_reelle, 0.10)
    poste = logger.add(caplog.handler, level="ERROR", format="{message}")
    try:
        for _ in range(3):
            with pytest.raises(S.JournalIncomplet):
                S.calculer(copie_reelle, registre.reference)
    finally:
        logger.remove(poste)
    alarmes = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR
        and "[ALARME] Journal des mouvements incomplet" in r.getMessage()
    ]
    assert len(alarmes) == 1, f"{len(alarmes)} alarms for 3 scorings"
    message = alarmes[0].getMessage()
    assert "274" not in message  # no hard-coded value: the message is computed
    # The alarm must say WHAT TO DO, not only that something is wrong: the command is in it,
    # naming the run concerned between the subcommand and the option.
    assert "python -m experiences journal" in message and "--regenerer" in message
    assert copie_reelle.name in message


@sans_substrat
def test_le_front_montant_se_rearme_quand_le_journal_redevient_complet(
    copie_reelle, registre, caplog
):
    """A log regenerated then truncated again must re-alarm: otherwise the guard wears out with use."""
    import logging

    from loguru import logger

    original = _lignes(EXEC_REELLE / "moves.csv")
    (copie_reelle / "scores.json").unlink(missing_ok=True)
    poste = logger.add(caplog.handler, level="ERROR", format="{message}")
    try:
        _tronquer(copie_reelle, 0.10)
        with pytest.raises(S.JournalIncomplet):
            S.calculer(copie_reelle, registre.reference)
        shutil.copy(EXEC_REELLE / "moves.csv", copie_reelle / "moves.csv")
        S.calculer(copie_reelle, registre.reference)  # complete log: passes
        _tronquer(copie_reelle, 0.10)
        with pytest.raises(S.JournalIncomplet):
            S.calculer(copie_reelle, registre.reference)
    finally:
        logger.remove(poste)
    alarmes = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR
        and "[ALARME] Journal des mouvements incomplet" in r.getMessage()
    ]
    assert len(alarmes) == 2, f"{len(alarmes)} alarms for two distinct truncations"
    assert len(original) > 0


@sans_substrat
def test_un_score_deja_ecrit_sur_un_journal_tronque_est_invalide(
    copie_reelle, registre
):
    """Refusal is not enough: the published `scores.json` stays readable until it is removed.

    This is exactly what happened on 2026-09-12 — the invalid composite score stayed on
    disk, served by the page and by the dashboard, while the rule did not exist.
    """
    assert (copie_reelle / "scores.json").is_file()
    (copie_reelle / "synthese_scores.html").write_text("page", encoding="utf-8")
    _tronquer(copie_reelle, 0.10)
    assert S.score_execution(copie_reelle, registre.reference, registre) is None
    assert not (copie_reelle / "scores.json").exists()
    assert not (copie_reelle / "synthese_scores.html").exists()
    invalide = copie_reelle / "scores.invalide.json"
    assert invalide.is_file(), "the old score must stay auditable, not disappear"
    contenu = json.loads(invalide.read_text(encoding="utf-8"))
    assert contenu["invalide"]["motif"]
    assert contenu["composite"]["emd_jsd"] is not None


@sans_substrat
def test_un_score_sans_perimetre_verifie_est_perime(copie_reelle):
    """Without this criterion, a score predating the rule would be replayed indefinitely unchecked.

    Offline replay never reads `moves.csv` back: it recomposes the composite score from the
    raw scores. A wrong score would thus cross every scoring formula change intact.
    """
    scores = json.loads((copie_reelle / "scores.json").read_text(encoding="utf-8"))
    scores.pop("perimetre_verifie", None)
    (copie_reelle / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False), encoding="utf-8"
    )
    assert S.scores_perimes(copie_reelle) is True


# ═══════════ Non-regression: a complete run scores as before ═══════════


def test_les_executions_completes_gardent_leur_composite_au_centieme(registre):
    """The guard must change nothing in what was fine — otherwise it costs more than it brings."""
    racine = REPO / "data" / "experiences"
    compares = 0
    for d in JN.executions(racine):
        chemin = d / "scores.json"
        if not chemin.is_file():
            continue
        ancien = json.loads(chemin.read_text(encoding="utf-8"))
        if (ancien.get("formule") or {}).get("sha256") != registre.reference.sha256:
            continue
        if ancien.get("composite", {}).get("emd_jsd") is None:
            continue
        frais = S.calculer(d, registre.reference)
        assert frais["composite"]["emd_jsd"] == pytest.approx(
            ancien["composite"]["emd_jsd"], abs=0.005
        ), d.name
        assert frais["perimetre_verifie"]["complet"] is True, d.name
        compares += 1
    if compares == 0:
        pytest.skip("no run scored under the reference scoring formula")


def test_le_seuil_laisse_passer_le_deficit_reel_des_executions_saines():
    """The deficit of a healthy run is NEGATIVE, and the rule never fires on it.

    Measured on 2026-09-15: from −0.19 % to −0.32 % over 37 runs. The log holds the
    `sans_solution` lines, which `couverture.decides` excludes; so it always has a few
    more. A threshold that fired there would make the repository unscorable overnight.
    """
    synthese = {"couverture": {"decides": 3154}}
    assert S.mesurer_perimetre(3161, synthese)["complet"] is True
    assert S.mesurer_perimetre(3154, synthese)["complet"] is True
    # 2 % of 3,154, i.e. 63 lines of margin: the last line accepted, then the first refused.
    assert S.mesurer_perimetre(3092, synthese)["complet"] is True
    assert S.mesurer_perimetre(3090, synthese)["complet"] is False
    assert S.mesurer_perimetre(274, synthese)["complet"] is False


def test_sans_compte_de_decisions_le_journal_n_est_pas_declare_complet():
    """Vacuity ≠ verification. Being unable to compare anything is not having compared successfully."""
    for synthese in ({}, {"couverture": {}}, {"couverture": {"decides": 0}}):
        assert S.mesurer_perimetre(3161, synthese)["complet"] is None


# ═══════════ Resumption regenerates the log ═══════════


@sans_substrat
def test_la_regeneration_reproduit_le_journal_a_l_identique(copie_reelle, registre):
    """Column by column, except the two the archive cannot return.

    "Trajet" is a process counter (the write order depends on task
    scheduling) and "Heure de calcul" is not archived decision by decision. Everything else —
    persona, chosen mode, distance, distribution, options, discarded — must be identical,
    otherwise regeneration would fabricate a plausible but wrong log.
    """
    jeu, personnes, info = _sources_reelles(copie_reelle)
    execution = JN.ouvrir(copie_reelle)
    etiquette = JN.EtiquetteDecideur.depuis_execution(execution.config)
    asyncio.run(
        JN.regenerer(execution, jeu, personnes, etiquette, info_population=info)
    )
    origine = {r["ID Trajet"]: r for r in _lignes(EXEC_REELLE / "moves.csv")}
    regenere = {r["ID Trajet"]: r for r in _lignes(copie_reelle / "moves.csv")}
    assert set(regenere) == set(origine)
    volatiles = {"Trajet", "Heure de calcul"}
    divergentes = {
        colonne
        for cle in origine
        for colonne in origine[cle]
        if colonne not in volatiles
        and (origine[cle].get(colonne) or "") != (regenere[cle].get(colonne) or "")
    }
    assert not divergentes, f"diverging columns: {sorted(divergentes)}"


@sans_substrat
def test_le_journal_regenere_donne_le_meme_score_au_centieme(copie_reelle, registre):
    avant = S.calculer(EXEC_REELLE, registre.reference)
    jeu, personnes, info = _sources_reelles(copie_reelle)
    execution = JN.ouvrir(copie_reelle)
    asyncio.run(
        JN.regenerer(
            execution,
            jeu,
            personnes,
            JN.EtiquetteDecideur.depuis_execution(execution.config),
            info_population=info,
        )
    )
    apres = S.calculer(copie_reelle, registre.reference)
    for cle in ("emd_jsd", "l1", "emd_jsd_hors_choix_unique", "l1_hors_choix_unique"):
        assert apres["composite"][cle] == pytest.approx(
            avant["composite"][cle], abs=0.005
        ), cle


@sans_substrat
def test_regenerer_depuis_une_autre_cohorte_est_refuse(copie_reelle):
    """A log rebuilt on the wrong population would be plausible, scorable, and wrong.

    It is the same family of failure as the truncated log: a normal-looking file of which
    nothing says that it describes something other than what one believes.
    """
    jeu, personnes, info = _sources_reelles(copie_reelle)
    execution = JN.ouvrir(copie_reelle)
    empreintes = dict(execution.config.get("empreintes") or {})
    empreintes["population"] = {
        **(empreintes.get("population") or {}),
        "fichier_sha256": "0" * 64,
    }
    execution.config["empreintes"] = empreintes
    with pytest.raises(JN.JournalIrregenerable, match="population différente"):
        asyncio.run(
            JN.regenerer(
                execution,
                jeu,
                personnes,
                JN.EtiquetteDecideur.depuis_execution(execution.config),
                info_population=info,
            )
        )


def _sources_reelles(dossier: Path):
    """(jeu, personnes, info) of the real run, or skip if the repository no longer holds them."""
    from experiences import journal as JN

    execution = JN.ouvrir(dossier)
    exp = execution.config.get("experience") or {}
    nom_jeu = (exp.get("jeu") or {}).get("nom")
    chemin_jeu = REPO / "data" / "jeux" / str(nom_jeu)
    chemin_pop = (
        REPO
        / "data"
        / "population"
        / Path(str((exp.get("population") or {}).get("chemin") or "")).name
    )
    if not chemin_jeu.is_dir() or not chemin_pop.is_dir():
        pytest.skip(f"set or cohort missing from the repository: {nom_jeu}")
    return JN.resoudre_sources(execution, jeu=chemin_jeu, population=chemin_pop)


# ═══════════ Cold resumption on the synthetic bench ═══════════


def test_une_reprise_a_froid_laisse_un_journal_complet(banc, monkeypatch):  # noqa: F811
    """The original case, replayed end to end: interruption, lost log, resumption.

    On resumption, archived decisions are SERVED AGAIN without querying the decision-maker, and this
    path wrote no log line. We simulate the loss of the log left by the
    interrupted process, then resume: the log must end up complete, that is,
    hold one line per archived decision EXCEPT the uncovered and unusable ones, which
    have neither a chosen mode nor a distance to record.
    """
    from experiences.decideurs import DecideurDureeMinimale
    from experiences.runner import Controle

    exp = _exp(banc, regroupement={"parallelisme": 1})

    class PauseApres(DecideurDureeMinimale):
        nom = "duree_minimale"
        sans_quota = True

        def __init__(self, controle, apres):
            self.controle, self.apres, self.appels = controle, apres, 0

        async def choisir(self, person, ctx, presentees):
            self.appels += 1
            if self.appels >= self.apres:
                self.controle.pause = True
            return await super().choisir(person, ctx, presentees)

    from experiences.archive import Execution

    _, execution = _lancer(banc, exp)  # first run, complete
    total_attendu = sum(
        1
        for t in execution.decisions
        if t.get("methode") not in (METHODE_NON_COUVERT, METHODE_INEXPLOITABLE)
    )
    assert total_attendu > 0
    execution.fermer()

    # The interrupted process took its log with it: this is the situation of 2026-09-12, where
    # `moves.csv` held only a fraction of the archived decisions.
    (execution.dossier / "moves.csv").unlink()
    rouverte = Execution.ouvrir(execution.dossier)
    assert len(rouverte.decisions) > 0, (
        "resumption must have decisions to serve again"
    )

    compteurs, _ = _lancer(banc, exp, execution=rouverte)
    assert compteurs["resservies"] == len(rouverte.decisions)
    assert compteurs["sollicitations"] == 0, "a complete resumption queries nothing"

    lignes = _lignes(execution.dossier / "moves.csv")
    assert len(lignes) == total_attendu, (
        f"{len(lignes)} log line(s) for {total_attendu} usable "
        f"decision(s) — a resumption must leave a complete log"
    )
    assert Controle  # the interruption harness stays importable (refactor guard)


def test_le_journal_regenere_a_la_reprise_est_scorable(banc):  # noqa: F811
    """The loop closes: complete log → rule R24 lets the score through.

    Without this check, one could regenerate a log that the safeguard would still
    refuse, and resumption would remain unusable for measurement.
    """
    from experiences.archive import Execution

    exp = _exp(banc, regroupement={"parallelisme": 1})
    _, execution = _lancer(banc, exp)
    execution.fermer()
    (execution.dossier / "moves.csv").unlink()
    rouverte = Execution.ouvrir(execution.dossier)
    _lancer(banc, exp, execution=rouverte)

    synthese = json.loads(
        (execution.dossier / "synthese.json").read_text(encoding="utf-8")
    )
    lignes = len(_lignes(execution.dossier / "moves.csv"))
    constat = S.mesurer_perimetre(lignes, synthese)
    assert constat["complet"] is True, constat


# ═══════════ Prerequisite — reading back a cohort from before the switch ═══════════
#
# Found while regenerating the run of 12/09: the rebuilt log stopped at 1,158
# lines out of 3,161, with 2,003 decisions "without persona". The loader rejected 637 personas
# out of 1,000 as "zone inconnue (1ere couronne)" — the frozen v5 cohort carries the labels
# from before the English switch, and the filter only accepted the English canon.
# `charger_population` nevertheless announces "no scope filter": the caller received
# 363 persons believing it had them all.


def _persona(zone):
    """A minimal persona that carries only what the scope verdict looks at."""
    from types import SimpleNamespace

    return SimpleNamespace(
        identity=SimpleNamespace(
            home=SimpleNamespace(lon=1.44, lat=43.60),
            traits_json={"residence_zone": zone} if zone else {},
        )
    )


@pytest.mark.parametrize(
    "fr, en",
    [
        ("1ere couronne", "1st ring"),
        ("2eme couronne", "2nd ring"),
        ("3eme couronne", "3rd ring"),
        ("Toulouse", "Toulouse"),
    ],
)
def test_une_couronne_d_avant_la_bascule_est_admise_comme_sa_traduction(fr, en):
    from inputs.population.eqasim_loader import perimeter_verdict

    assert perimeter_verdict(_persona(fr), None)[0] is True, fr
    assert perimeter_verdict(_persona(en), None)[0] is True, en


def test_hors_perimetre_reste_rejete_et_garde_son_nom():
    """Rejected, but NAMED: outside the scope is a survey category, not an unreadable value.

    Confusing them would hide a badly translated cohort behind a legitimate rejection — and that
    is exactly what made the failure invisible: 637 "zone inconnue" rejections read
    as normal filtering.
    """
    from inputs.population.eqasim_loader import perimeter_verdict

    for libelle in ("hors périmètre", "outside perimeter"):
        admis, motif = perimeter_verdict(_persona(libelle), None)
        assert admis is False
        assert motif == "outside perimeter", libelle


def test_une_modalite_vraiment_inconnue_reste_rejetee_et_se_nomme():
    from inputs.population.eqasim_loader import perimeter_verdict

    admis, motif = perimeter_verdict(_persona("4eme couronne"), None)
    assert admis is False
    assert "4eme couronne" in motif
