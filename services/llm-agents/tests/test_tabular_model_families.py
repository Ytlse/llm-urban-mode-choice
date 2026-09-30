"""The four tabular reference methods and how they are loaded.

The four tabular reference methods — LightGBM booster, multinomial logit, kernel
logistic regression, random forest — are wired the same way: a format declared in
`POLICY_FORMATS`, a label in the family table, loading through `load_policy`.

**This file said the opposite until 2026-09-16**, and the reason for the change must be stated,
otherwise the next session will undo this wiring believing it restores the earlier state.

The forest was kept out of both tables in the name of an earlier rule — *a control does not
become an arbiter*. Two things have changed since:

1. **The practical reason is gone.** The forest's dedicated launcher was written on 2026-09-11
   because the work on kernel logistic regression was modifying these same two tables at the
   same time, and writing to them in parallel would have overwritten one of the two. That work
   is closed. The workaround — registering the family in memory and replacing `load_policy` in
   the decision-maker's namespace — did cost, however, an experiment the CLI could not replay.

2. **The word "arbiter" means something other than presence in these tables.** The arbiter is
   the model whose preference decides in the two-oracle composite score, and
   the two-oracle score specification names it: it is the MNL. Being listed in
   `FAMILLES` confers no arbitration role; these tables only say "here is how to
   reload this artefact". The forest has in fact lost its control label in the article
   itself (chapter 4, draft v0.15): it is the fourth reference method, and the
   ceiling is read axis by axis over the four.

What remains true and these tests keep pinning down: the family label is **derived from the
format declared by the artefact**, never hard-coded — a forest run that
announced itself as "lightgbm" in the traces would be worse than a run with no label.

What these tests do not cover: the forest runs from the HOST and not from the `controller`
container, because it refits at load time and scikit-learn there is not the
estimation version. This is not a wiring matter but an environment one, and it is the
reproduction safeguard of `mode_choice_rf.py` that enforces it, by measuring.
"""

from __future__ import annotations

import json

import pytest
from experiences.chemins import racine_depot
from experiences.decideur_modele import FAMILLES

METHODES = {
    "lightgbm_mode_choice_policy": "scripts/progedo_logit/mode_choice_policy.json",
    "mnl_mode_choice_policy": "scripts/progedo_logit/mnl_model.json",
    "klr_mode_choice_policy": "scripts/progedo_logit/klr_model.json",
    "rf_mode_choice_policy": "scripts/progedo_logit/rf_mode_choice_policy.json",
}
FORMAT_RF = "rf_mode_choice_policy"


# ── The four methods are wired the same way ─────────────────────────────────


def test_les_quatre_methodes_sont_dans_la_table_des_libelles():
    assert set(METHODES) <= set(FAMILLES), sorted(set(METHODES) - set(FAMILLES))


def test_les_quatre_methodes_sont_acceptees_par_load_policy():
    from scripts.synthesis.model_on_common_set import POLICY_FORMATS

    assert set(METHODES) <= set(POLICY_FORMATS)


def test_chaque_famille_a_un_libelle_distinct():
    """Two families under the same label would make the traces uninterpretable."""
    libelles = [FAMILLES[f] for f in METHODES]
    assert len(set(libelles)) == len(libelles), libelles


@pytest.mark.parametrize("format_, chemin", sorted(METHODES.items()))
def test_le_libelle_vient_du_format_declare_par_lartefact(format_, chemin):
    """The label is not guessed from the file name: it comes from the `format` field."""
    p = racine_depot() / chemin
    if not p.is_file():
        pytest.skip(f"artefact missing: {chemin}")
    # Head of the file only: `mode_choice_policy.json` weighs 18 MB.
    tete = p.read_text(encoding="utf-8")[:400]
    assert f'"format": "{format_}"' in " ".join(tete.split())


def test_lartefact_de_la_foret_declare_bien_son_format():
    p = racine_depot() / METHODES[FORMAT_RF]
    if not p.is_file():
        pytest.skip("artefact missing")
    assert json.loads(p.read_text(encoding="utf-8"))["format"] == FORMAT_RF


# ── The forest loads through the official path, and returns an RFPredictor ─


def test_la_foret_porte_son_propre_libelle_et_pas_celui_dune_autre():
    """The risk the launcher documented: an rf run announced as "lightgbm"."""
    assert FAMILLES[FORMAT_RF] == "rf"


def test_load_policy_aiguille_la_foret_vers_son_chargeur():
    """`load_policy` must return an RFPredictor, not attempt a LightGBM Booster.

    The refit costs about fifteen seconds and checks along the way that the forest
    reproduces its published metrics: it is the slowest test in the file, and the only one that
    proves the wiring leads somewhere.
    """
    from scripts.synthesis.model_on_common_set import load_policy

    artefact = racine_depot() / METHODES[FORMAT_RF]
    spec_file = racine_depot() / "scripts/progedo_logit/feature_spec.json"
    trainset = racine_depot() / "scripts/progedo_logit/rf_mode_choice_trainset.npz"
    for p in (artefact, spec_file, trainset):
        if not p.is_file():
            pytest.skip(f"missing: {p.name}")

    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    modele, lu = load_policy(artefact, spec)
    assert type(modele).__name__ == "RFPredictor"
    assert lu["format"] == FORMAT_RF
    assert modele.feature_name() == [f["name"] for f in spec["features"]]
    # The safeguard ran and concluded: under another scikit-learn version it would have
    # raised, which makes this assertion an environment check as much as a wiring one.
    assert modele.controle["reproduit"] is True


def test_le_lanceur_dedie_ne_remplace_plus_rien():
    """It checks the declaration instead of injecting it — otherwise the wiring would be useless."""
    from scripts.progedo_logit import lancer_experience_rf

    assert hasattr(lancer_experience_rf, "verifier_famille_rf")
    assert not hasattr(lancer_experience_rf, "enregistrer_famille_rf"), (
        "in-memory injection was removed in ticket 088 § 3.3: the family is declared"
    )


def test_le_lanceur_accepte_la_declaration_en_place():
    from scripts.progedo_logit.lancer_experience_rf import verifier_famille_rf

    verifier_famille_rf()  # does not raise: both tables carry the family
