"""An experiment name states the REAL family of its model.

`ABREV_DECIDEUR["modele"]` was `lgbm`, so that **every** experiment with a model decision-maker
announced itself as LightGBM, whatever its family:

| Artefact | Name produced before | What the name said |
|---|---|---|
| `mnl_model.json` | `exp_lgbm-mnl_model_…` | a booster, for a multinomial logit |
| `klr_model.json` | `exp_lgbm-klr_model_…` | a booster, for a kernel regression |
| `rf_mode_choice_policy.json` | `exp_lgbm-rf_mode_choi_…` | a booster, for a forest |

A comment in `decideur_modele.py` says exactly why this is serious:
*"a wrong label is worse than a missing label, because it goes unnoticed"*. That
label had been fixed in the traces and in the fingerprint, but not in the **name**,
which is however what one reads first and what names the archive folder.

The family is read from the artefact's `format` field, at the top of the file — so without loading
18 MB of booster. An unreadable artefact or one of unknown format is assigned **no**
family: the segment becomes `mod-<fichier>`, which asserts nothing.
"""

from __future__ import annotations

from experiences.nommage import segment_decideur


def _modele(artefact: str | None) -> dict:
    return {"type": "modele", "artefact": artefact}


def test_chaque_famille_se_nomme_elle_meme():
    attendus = {
        "scripts/progedo_logit/mode_choice_policy.json": "lgbm",
        "scripts/progedo_logit/mnl_model.json": "mnl",
        "scripts/progedo_logit/klr_model.json": "klr",
        "scripts/progedo_logit/rf_mode_choice_policy.json": "rf",
    }
    for artefact, famille in attendus.items():
        assert segment_decideur(_modele(artefact)) == famille, artefact


def test_quatre_familles_quatre_segments_distincts():
    """Two families under the same segment would share the archive folder."""
    segments = [
        segment_decideur(_modele(a))
        for a in (
            "scripts/progedo_logit/mode_choice_policy.json",
            "scripts/progedo_logit/mnl_model.json",
            "scripts/progedo_logit/klr_model.json",
            "scripts/progedo_logit/rf_mode_choice_policy.json",
        )
    ]
    assert len(set(segments)) == 4, segments


def test_sans_artefact_cest_le_booster_par_defaut():
    """The decision-maker's default artefact IS the booster: naming it `lgbm` stays true."""
    assert segment_decideur(_modele(None)) == "lgbm"


def test_un_artefact_introuvable_naffirme_aucune_famille():
    """Better to say nothing than to say "lgbm" of a file that was not read."""
    seg = segment_decideur(_modele("scripts/progedo_logit/inexistant_v7.json"))
    assert seg.startswith("mod-"), seg
    assert "lgbm" not in seg


def test_un_format_inconnu_naffirme_aucune_famille(tmp_path):
    f = tmp_path / "exotique.json"
    f.write_text('{"format": "quelque_chose_de_neuf", "spec_version": 2}', encoding="utf-8")
    seg = segment_decideur(_modele(str(f)))
    assert seg.startswith("mod-") and "lgbm" not in seg, seg
