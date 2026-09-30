"""Composite scoring of an execution.

Substrate: a real finished run from the
repo, copied to tmp so that writes do not pollute `data/`.
"""

import json
import shutil
from pathlib import Path

import pytest
from experiences import formule as F
from experiences import score as S

REPO = Path(__file__).resolve().parents[3]


def _exec_reelle() -> Path | None:
    """The first finished and usable run of the repo, whatever its name.

    When hard-coded, the reference disappeared with the experiment that carried it (a
    deletion from the dashboard is enough) and the WHOLE suite started to
    skip without anything flagging it: green tests that test nothing.
    """
    racine = REPO / "data" / "experiences"
    for exp in sorted(racine.iterdir()) if racine.is_dir() else []:
        for d in sorted((exp / "executions").iterdir()) if (exp / "executions").is_dir() else []:
            if not (d / "moves.csv").exists() or not (d / "synthese.json").exists():
                continue
            try:
                synthese = json.loads((d / "synthese.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (synthese.get("etat") or {}).get("etat") == "terminee":
                return d
    return None


EXEC_REELLE = _exec_reelle()

CHAMPS_OBLIGATOIRES = (
    "execution",
    "volet",
    "formule",
    "moves_sha256",
    "couverture",
    "composite",
    "scores_bruts",
    "detail",
)


pytestmark = pytest.mark.skipif(
    EXEC_REELLE is None,
    reason="no finished run in data/experiences (substrate missing)",
)


@pytest.fixture
def exec_tmp(tmp_path):
    dst = tmp_path / EXEC_REELLE.name
    shutil.copytree(EXEC_REELLE, dst)
    # Clean substrate: remove any score artefacts already present in
    # the real run (a `score --toutes` may have written some), otherwise the tests that
    # check the absence of scores.json start out biased.
    for f in ("scores.json", "synthese_scores.html"):
        (dst / f).unlink(missing_ok=True)
    return dst


@pytest.fixture(scope="module")
def registre():
    return F.charger()


def test_R1_composite_vient_du_scorer(exec_tmp, registre):
    # The composite of calculer() equals that of a Scorer applied to the same frame.
    contenu = S.calculer(exec_tmp, registre.reference)
    scorer, _ = S.scorer_pour(registre.reference)
    # The scope is read through `lire_perimetre`, the only place that decides it.
    rows, _ = S.lire_perimetre(exec_tmp, S.EXCLURE_METHODES)
    attendu = S.frames.simulation_frames(rows)["attendu"]
    cerema = S.frames.load_cerema(
        S._resoudre_cerema(json.loads((exec_tmp / "synthese.json").read_text()))
    )
    direct = scorer.score(attendu, cerema)["emd_jsd"]["composite"]
    assert abs(contenu["composite"]["emd_jsd"] - direct) < 1e-9


def test_R2_length_penalty_hors_composite(exec_tmp, registre):
    from scripts.synthesis.formule_score import metrics as m

    contenu = S.calculer(exec_tmp, registre.reference)
    emd = contenu["scores_bruts"]["emd_jsd"]
    poids = registre.reference.poids_scorer()
    assert "length_penalty" not in poids
    # The composite is exactly the weighted sum over the 7 dimensions; adding a
    # length_penalty weight via the formula weights is impossible (R19), so it never
    # enters: the stored value matches the sum over the 7 dimensions.
    assert (
        abs(contenu["composite"]["emd_jsd"] - m.weighted_composite(emd, poids)) < 1e-9
    )


def test_R5_scores_json_complet(exec_tmp, registre):
    contenu = S.calculer(exec_tmp, registre.reference)
    for champ in CHAMPS_OBLIGATOIRES:
        assert champ in contenu, f"missing mandatory field: {champ}"
    # detail per stratum: target / actual / L1 / count present
    strate = contenu["detail"]["age"]["strates"][0]
    for cle in ("cat", "n", "actual", "target", "l1"):
        assert cle in strate


def test_R6_rejeu_exact(exec_tmp, registre):
    ref = S.calculer(exec_tmp, registre.reference)
    poids_alt = {**registre.reference.poids, "age": 0.9, "distance": 0.1}
    alt = F.Formule("alt", poids_alt, F.empreinte(poids_alt))
    rejoue = S.rejouer(ref, alt)
    frais = S.calculer(exec_tmp, alt)
    assert abs(rejoue["composite"]["emd_jsd"] - frais["composite"]["emd_jsd"]) < 1e-9
    assert abs(rejoue["composite"]["l1"] - frais["composite"]["l1"]) < 1e-9
    assert rejoue["formule"]["sha256"] == alt.sha256


def test_R8_dimension_vide_non_mesuree(registre):
    # A frame where one dimension has no measured stratum: the engine declares it
    # `undefined` (max-loss fallback), never a silent 0, and the composite does not drop to 0.
    scorer, _ = S.scorer_pour(registre.reference)
    cerema = S.frames.load_cerema(S.CEREMA_DEPOT)
    # A single decision, with no distance or age category filled in.
    rows = [
        {
            "agent_id": "a1",
            "mode_cat": "voiture",
            "weight": 1.0,
            "genre": "Homme",
            "age_cat": None,
            "occupation": "actif_temps_plein",
            "motif": "travail",
            "dist_cat": None,
            "lieu_residence": None,
            "type_logement": None,
        }
    ]
    scores, mesure = scorer.primary.compute_detailed(scorer._pd.DataFrame(rows), cerema)
    assert mesure.undefined, "at least one dimension must be declared unmeasured"
    assert scores.composite != 0.0


def test_R15_routage_volet(exec_tmp):
    # The reference run is NOT hard-coded (cf. _exec_reelle): its decision-maker
    # is that of the first finished run of the repo, `passerelle` or `modele` depending on
    # what is there on that day. Asserting `== "1"` on it as is meant testing that
    # draw rather than the rule — the suite turned red on 2026-09-08 as soon as
    # `Light_GBM` (decision-maker `modele`, first in alphabetical order) produced a
    # finished run. So the type is set explicitly, and BOTH branches of R15
    # are covered whatever is on disk.
    synthese = json.loads((exec_tmp / "synthese.json").read_text())
    synthese["empreintes"]["decideur"]["type"] = "modele"
    assert S.volet_pour(synthese) == "3"
    synthese["empreintes"]["decideur"]["type"] = "passerelle"
    assert S.volet_pour(synthese) == "1"


def test_R17_substrat_lie_moves(exec_tmp, registre):
    S.score_execution(exec_tmp, registre.reference)
    assert not S.scores_perimes(exec_tmp)
    (exec_tmp / "moves.csv").write_text("Référence\nmodifié\n", encoding="utf-8")
    assert S.scores_perimes(exec_tmp)


def test_082_couronnes_fantomes_periment_le_scores_json(exec_tmp, registre):
    """A scores.json carrying `1st_ring` must force a full computation.

    The offline replay only recomputes the composite: without this criterion, `--toutes`
    would rewrite the v6 runs while keeping a "place of residence" dimension
    missing three rings out of four.
    """
    S.score_execution(exec_tmp, registre.reference)
    assert not S.scores_perimes(exec_tmp)
    scores = json.loads((exec_tmp / "scores.json").read_text())
    strates = scores["detail"]["lieu_residence"]["strates"]
    strates.append({
        "cat": S.frames.OFF_REFERENCE_ROW, "n": 12, "actual": {}, "target": {},
        "l1": None, "covered": False, "excluded_mass": 12.0,
        "categories": {"1st_ring": {"mass": 12.0, "n": 12}},
    })
    (exec_tmp / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    assert S.scores_perimes(exec_tmp)


def test_082_hors_perimetre_ne_perime_rien(exec_tmp, registre):
    """The "off reference data" row is LEGITIMATE when it carries `hors_perimetre`:
    confusing it with the failure would recompute the whole history on
    every pass, without any figure changing."""
    S.score_execution(exec_tmp, registre.reference)
    scores = json.loads((exec_tmp / "scores.json").read_text())
    scores["detail"]["lieu_residence"]["strates"].append({
        "cat": S.frames.OFF_REFERENCE_ROW, "n": 3, "actual": {}, "target": {},
        "l1": None, "covered": False, "excluded_mass": 3.0,
        "categories": {S.frames.OUT_OF_PERIMETER_KEY: {"mass": 3.0, "n": 3}},
    })
    (exec_tmp / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    assert not S.scores_perimes(exec_tmp)


def test_R21_partielle_non_scoree(exec_tmp, registre):
    synthese = json.loads((exec_tmp / "synthese.json").read_text())
    synthese["etat"]["etat"] = "interrompue"
    (exec_tmp / "synthese.json").write_text(json.dumps(synthese), encoding="utf-8")
    assert S.score_execution(exec_tmp, registre.reference) is None
    assert not (exec_tmp / "scores.json").exists()


def _racine_avec(exec_tmp, tmp_path) -> Path:
    racine = tmp_path / "experiences"
    (racine / "Exp" / "executions").mkdir(parents=True)
    shutil.copytree(exec_tmp, racine / "Exp" / "executions" / exec_tmp.name)
    return racine


def test_R10_recalcul_hors_ligne(exec_tmp, tmp_path, registre, monkeypatch):
    # A run already scored (moves unchanged) is recomputed by REPLAY, without
    # rebuilding the Scorer: we prove it by making scorer_pour fail.
    racine = _racine_avec(exec_tmp, tmp_path)
    cible = racine / "Exp" / "executions" / exec_tmp.name
    S.score_execution(cible, registre.reference)  # first computation (once)

    def _interdit(*a, **k):
        raise AssertionError(
            "scorer_pour must not be called on the replay path (R10)"
        )

    monkeypatch.setattr(S, "scorer_pour", _interdit)
    poids = {**registre.reference.poids, "motif": 0.8}
    alt = F.Formule("alt", poids, F.empreinte(poids))
    bilan = S.rescorer_tout(alt, registre, racine=racine)
    assert bilan["rejouees"] == 1 and bilan["calculees"] == 0
    rescore = json.loads((cible / "scores.json").read_text())
    assert rescore["formule"]["sha256"] == alt.sha256


def test_rescorer_tout_ignore_partielles(exec_tmp, tmp_path, registre):
    racine = _racine_avec(exec_tmp, tmp_path)
    cible = racine / "Exp" / "executions" / exec_tmp.name
    synthese = json.loads((cible / "synthese.json").read_text())
    synthese["etat"]["etat"] = "interrompue"
    (cible / "synthese.json").write_text(json.dumps(synthese), encoding="utf-8")
    bilan = S.rescorer_tout(registre.reference, registre, racine=racine)
    assert bilan["ignorees"] == 1
    assert not (cible / "scores.json").exists()


def test_R7_est_reference_derive(exec_tmp, registre):
    contenu = S.calculer(exec_tmp, registre.reference)
    assert S.est_reference(contenu, registre)
    # Simulate a different reference formula without touching scores.json.
    autre = F.Formule(
        "autre",
        {**registre.reference.poids, "genre": 0.99},
        F.empreinte({**registre.reference.poids, "genre": 0.99}),
    )
    faux_registre = F.RegistreFormules([autre], "autre")
    assert not S.est_reference(contenu, faux_registre)


def test_R6_rejeu_dans_un_interprete_vierge(exec_tmp, registre, tmp_path):
    """The offline replay depends on no prior `scorer_pour`.

    Only `scorer_pour` puts `prompt_calibration/` on `sys.path`. Assuming it already
    done, `rejouer` died with ModuleNotFoundError in a process that had computed
    nothing — exactly the case of `score --toutes` and of the "Recalculer toutes
    les expériences" button when the first run encountered is already scored.
    """
    import subprocess
    import sys as _sys

    fichier = tmp_path / "scores.json"
    fichier.write_text(
        json.dumps(S.calculer(exec_tmp, registre.reference)), encoding="utf-8"
    )
    code = (
        "import json;"
        "from experiences import formule as F, score as S;"
        "reg = F.charger();"
        f"s = json.load(open({str(fichier)!r}));"
        "print(S.rejouer(s, reg.reference)['composite']['emd_jsd'])"
    )
    proc = subprocess.run(
        [_sys.executable, "-c", code],
        cwd=REPO / "services" / "llm-agents",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    attendu = S.rejouer(json.loads(fichier.read_text()), registre.reference)
    assert abs(float(proc.stdout.strip()) - attendu["composite"]["emd_jsd"]) < 1e-9
