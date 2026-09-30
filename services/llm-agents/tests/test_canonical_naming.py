"""Canonical experiment naming: the name is computed.

One test per rule, named by its number. No disk outside `tmp_path`: `attribuer_nom` reads
a temporary experiments directory.

Since rule R18, the population is no longer a reference value: it is
absent from `DEFAUTS_NOMMAGE`, so no cohort is silent. Every name expected here carries
its `pop-<abbreviated>` segment at the head of block N7 — including that of the reference
definition, which is `exp_durmin_pop-1000_PANEL_nosim`. This is precisely what the rule
fixes: a name without substrate read as "nothing to report" and meant "cohort v1".
"""

from __future__ import annotations

import copy

import pytest
import yaml

from experiences import nommage as N

BASE = {
    "nom": "peu-importe",
    "population": {"chemin": "/data/eqasim-output/population_1000_PANEL"},
    "jeu": {"nom": "population_1000_PANEL_20260316", "dossier": None},
    "gabarit": {"categorie": "itinary_multi_agent", "variante": "minimal_persona"},
    "decideur": {
        "type": "duree_minimale",
        "modele": None,
        "parametres": {},
        "rejeu_de": None,
        "graine": None,
        "artefact": None,
    },
    "mode": "sans_simulateur",
    "calendrier": {"politique": "commune", "date": "2026-03-16", "graine": 42},
    "horizon_jours": 1,
    "memoire": False,
    "evenements": [],
    "graine_ordre": 42,
    "graine_tirage": 42,
    "regroupement": {"parallelisme": 8},
    "tolerances_horaires": dict(N.TOLERANCES_REFERENCE),
    "max_candidats": 6,
    "attente_max_s": 120,
    "derive_de": None,
    "executions_connues": [],
}


def exp(**changements) -> dict:
    """A reference definition, n fields changed (dotted paths: `decideur.modele`)."""
    d = copy.deepcopy(BASE)
    for cle, valeur in changements.items():
        cible, *reste = cle.split("__")
        if reste:
            d[cible][reste[0]] = valeur
        else:
            d[cible] = valeur
    return d


def ecrire(dossier, nom: str, definition: dict) -> None:
    d = dossier / nom
    d.mkdir(parents=True, exist_ok=True)
    (d / "experience.yaml").write_text(
        yaml.safe_dump({**definition, "nom": nom}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# ── N2, N3, N8: grammar, decision-maker, temperature and mode always named ───


def test_n2_grammaire_et_mode():
    # `exp` · decision-maker · population · mode: the cohort sits between the decision-maker and
    # the mode because it opens block N7, before all other deviations (R18).
    assert N.nom_canonique(exp()) == "exp_durmin_pop-1000_PANEL_nosim"
    assert N.nom_canonique(exp(mode="simulateur")) == "exp_durmin_pop-1000_PANEL_sim"


def test_n3_chaque_decideur_a_son_segment():
    attendus = {
        "aleatoire": "exp_alea_pop-1000_PANEL_nosim",
        "duree_minimale": "exp_durmin_pop-1000_PANEL_nosim",
        "majoritaire_voiture": "exp_majvoiture_pop-1000_PANEL_nosim",
        "modele": "exp_lgbm_pop-1000_PANEL_nosim",
    }
    for type_, attendu in attendus.items():
        assert N.nom_canonique(exp(decideur__type=type_)) == attendu


def test_n3_artefact_et_rejeu_nomment_leur_source():
    # `police_v7.json` does not exist on disk: the family therefore cannot be read from
    # its `format` field, and the segment asserts NONE. The file name
    # stays in the segment — it is indeed its source that names —, but without claiming it
    # is a booster: the former `lgbm-` prefix lied about every artefact that was not
    # one. A real artefact is named by its family: see
    # `test_model_family_in_name.py`.
    inconnu = exp(
        decideur__type="modele",
        decideur__artefact="scripts/progedo_logit/police_v7.json",
    )
    assert N.nom_canonique(inconnu) == "exp_mod-police_v7_pop-1000_PANEL_nosim"
    rejeu = exp(
        decideur__type="rejeu",
        decideur__rejeu_de="/app/data/experiences/e/executions/2026-09-07_22_09_48",
    )
    assert N.nom_canonique(rejeu) == "exp_rejeu-260907-2209_pop-1000_PANEL_nosim"


def test_n3_passerelle_sans_modele_refuse_de_nommer():
    with pytest.raises(N.NommageImpossible, match="decideur.modele"):
        N.nom_canonique(exp(decideur__type="passerelle"))


def test_n8_temperature_toujours_nommee_pour_la_passerelle():
    def nom(t):
        return N.nom_canonique(
            exp(
                decideur__type="passerelle",
                decideur__modele="mistral-small-latest",
                decideur__parametres={"temperature": t, "top_p": 1.0},
            )
        )

    assert nom(0.0).endswith("_t0_nosim")
    assert nom(0.7).endswith("_t07_nosim")
    assert nom(1.0).endswith("_t1_nosim")
    assert nom(1.25).endswith("_t125_nosim")
    # A decision-maker without temperature carries none in its name.
    assert "_t" not in N.nom_canonique(exp()).replace("_tir", "")


# ── N4: model slug ───────────────────────────────────────────────────────────


def test_n4_slugs_de_la_campagne():
    assert N.abreger_modele("gemini-3.1-flash-lite-preview") == "gemini-31-fl"
    # `preview` is BRUIT_MODELE: the withdrawn alias and the exact name give the SAME
    # slug. This is what made it possible to fix the judge's name on 2026-09-10 without
    # renaming a single already archived experiment.
    assert N.abreger_modele("gemini-3.1-flash-lite") == "gemini-31-fl"
    assert N.abreger_modele("gemini-3.5-flash-lite") == "gemini-35-fl"
    assert N.abreger_modele("mistral-small-latest") == "mistral-s"
    assert N.abreger_modele("qwen/qwen3.6-27b") == "qwen36-27b"
    assert N.abreger_modele("gpt-oss-120b") == "gpt-oss-120b"
    assert N.abreger_modele("Qwen/Qwen2.5-32B-Instruct-AWQ") == "qwen25-32b"


def test_n4_modele_inconnu_donne_un_slug_court_et_non_vide():
    slug = N.abreger_modele("acme/foo-bar-9b-latest")
    assert slug and len(slug) <= N.BUDGET_MODELE
    assert N.abreger_modele("") == ""


# ── N5: the prompt only counts for a decision-maker that reads one ───────────


def test_n5_prompt_absent_hors_passerelle():
    # This was the defect of `Light_GBM`: a displayed variant that decided nothing.
    assert "minper" not in N.nom_canonique(exp(decideur__type="modele"))
    passerelle = exp(
        decideur__type="passerelle", decideur__modele="mistral-small-latest"
    )
    assert "minper" in N.nom_canonique(passerelle)


def test_n5_variante_absente_se_nomme_actif():
    d = exp(decideur__type="passerelle", decideur__modele="mistral-small-latest")
    d["gabarit"]["variante"] = None
    assert N.nom_canonique(d) == "exp_mistral-s_actif_pop-1000_PANEL_t0_nosim"


# ── N6: calendar ────────────────────────────────────────────────────────────


def test_n6_calendrier():
    assert (
        N.nom_canonique(
            exp(
                calendrier={
                    "politique": "aleatoire",
                    "date": "2026-03-16",
                    "graine": 42,
                }
            )
        )
        == "exp_durmin_jtir_pop-1000_PANEL_nosim"
    )
    assert (
        N.nom_canonique(
            exp(calendrier={"politique": "propre", "date": "2026-03-16", "graine": 42})
        )
        == "exp_durmin_jpers_pop-1000_PANEL_nosim"
    )
    # Common date = day of the set → silent; another day is stated. The calendar segment
    # (N6) precedes the population one (first deviation of block N7).
    assert N.nom_canonique(exp()) == "exp_durmin_pop-1000_PANEL_nosim"
    autre = exp(calendrier={"politique": "commune", "date": "2026-03-17", "graine": 42})
    assert N.nom_canonique(autre) == "exp_durmin_j0317_pop-1000_PANEL_nosim"


# ── N7: the deviations, and nothing but the deviations ──────────────────────


def test_n7_defauts_muets():
    """All parameters at their reference value: only the population speaks.

    The cohort is NOT a default (R18): there is no implicit substrate any more, so even
    the most ordinary definition announces on which population it was measured.
    """
    assert N.nom_canonique(exp()) == "exp_durmin_pop-1000_PANEL_nosim"


@pytest.mark.parametrize(
    "changement,attendu",
    [
        ({"regroupement": {"parallelisme": 16}}, "exp_durmin_pop-1000_PANEL_p16_nosim"),
        ({"memoire": True}, "exp_durmin_pop-1000_PANEL_mem_nosim"),
        ({"horizon_jours": 5}, "exp_durmin_pop-1000_PANEL_h5j_nosim"),
        ({"max_candidats": 10}, "exp_durmin_pop-1000_PANEL_c10_nosim"),
        ({"attente_max_s": 300}, "exp_durmin_pop-1000_PANEL_a300_nosim"),
        ({"graine_ordre": 7}, "exp_durmin_pop-1000_PANEL_go7_nosim"),
        ({"graine_tirage": 7}, "exp_durmin_pop-1000_PANEL_gt7_nosim"),
        (
            {"calendrier": {"politique": "commune", "date": "2026-03-16", "graine": 7}},
            "exp_durmin_pop-1000_PANEL_gc7_nosim",
        ),
        (
            {"decideur__type": "aleatoire", "decideur__graine": 7},
            "exp_alea_pop-1000_PANEL_gd7_nosim",
        ),
    ],
)
def test_n7_un_ecart_un_segment(changement, attendu):
    # The population opens block N7: every deviation named below goes AFTER it.
    assert N.nom_canonique(exp(**changement)) == attendu


def test_n7_population_nommee_sans_son_prefixe_commun():
    """Another cohort gives another segment — that is the whole point of R18.

    `abreger_population` removes `population_`, which all cohorts share and which therefore
    distinguishes nothing, before truncating to 16: what remains carries the size and the
    version. Here the set, still that of the reference cohort, no longer follows the convention
    `<population>_<AAAAMMJJ>`: the day of the set becomes unknown, so the date (`j0316`) and the
    set name enter the name in turn — a substrate/set mismatch must be visible.
    """
    d = exp(population={"chemin": "/data/eqasim-output/population_10000_v2"})
    nom = N.nom_canonique(d)
    # The set segment keeps its DATE, which is all its distinguishing power. Truncating from
    # the head gave `population_1000_` for `…_20260316` as for `…_20260317`: two different
    # sets, a single experiment name. `abreger_jeu` removes the common prefix then cuts
    # from the TAIL — the same remedy as `abreger_population`, applied to the same defect.
    assert nom == "exp_durmin_j0316_pop-10000_v2_jeu-0_PANEL_20260316_nosim"
    assert N.MOTIF_NOM.match(nom)


@pytest.mark.parametrize(
    ("population", "jeu", "attendu"),
    [
        # Survey samples: the set repeats the population stem WITHOUT `population_`. Before
        # 2026-09-30 the stem was not stripped, and both truncations cut mid-word:
        # `pop-enquete_058_trai_jeu-ain_cal_20260316`.
        (
            "population_enquete_058_train_cal",
            "enquete_058_train_cal_20260316",
            "exp_durmin_j0316_pop-enquete_058_train_cal_jeu-20260316_nosim",
        ),
        (
            "population_enquete_058_test",
            "enquete_058_test_20260316",
            "exp_durmin_j0316_pop-enquete_058_test_jeu-20260316_nosim",
        ),
        # Cohorts, whose set carries the prefixed population name: unchanged by the fix.
        (
            "population_1000_PANEL_v6",
            "population_1000_PANEL_v6_20260316_EN_c",
            "exp_durmin_j0316_pop-1000_PANEL_v6_jeu-20260316_EN_c_nosim",
        ),
        (
            "population_1000_PANEL_v6_c2",
            "population_1000_PANEL_v6_c2_20260316_EN_c",
            "exp_durmin_j0316_pop-1000_PANEL_v6_c2_jeu-20260316_EN_c_nosim",
        ),
    ],
)
def test_n7_jeu_d_enquete_sans_prefixe_population(population, jeu, attendu):
    d = exp(
        population={"chemin": f"/app/data/population/{population}"},
        jeu={"nom": jeu, "dossier": None},
    )
    nom = N.nom_canonique(d)
    assert nom == attendu
    assert N.MOTIF_NOM.match(nom)


def test_n7_evenements_et_tolerances():
    ev = exp(
        evenements=[
            {"type": "incident", "jour": 1, "heure_debut": "08:00", "description": "x"},
            {
                "type": "information",
                "jour": 2,
                "heure_debut": "09:00",
                "description": "y",
            },
        ]
    )
    assert "ev2" in N.nom_canonique(ev)
    tol = exp(tolerances_horaires={**N.TOLERANCES_REFERENCE, "car": "insensible"})
    nom = N.nom_canonique(tol)
    assert "tol-" in nom and nom == N.nom_canonique(
        tol
    )  # fingerprint stable from one call to the next
    # The two spellings of the same tolerance are not a deviation.
    equivalent = exp(
        tolerances_horaires={
            **N.TOLERANCES_REFERENCE,
            "transit": {"type": "pas", "pas_min": 10},
        }
    )
    assert "tol-" not in N.nom_canonique(equivalent)


def test_n7_jeu_hors_convention_se_nomme():
    # Outside the `<population>_<AAAAMMJJ>` convention, the day of the set is unknown: the set AND the
    # date enter the name, otherwise two different frozen sets would be confused. The
    # population segment stays at the head of block N7, before the set one.
    d = exp(jeu={"nom": "population_1000_PANEL_gele_v5", "dossier": None})
    assert N.nom_canonique(d) == "exp_durmin_j0316_pop-1000_PANEL_jeu-gele_v5_nosim"


# ── N9: the name is usable as a directory and as EXP= ───────────────────────


def test_n9_le_nom_satisfait_le_motif():
    for d in (
        exp(),
        exp(decideur__type="passerelle", decideur__modele="qwen/qwen3.6-27b"),
        exp(population={"chemin": "/data/eqasim-output/pop lente; rm -rf"}),
    ):
        nom = N.nom_canonique(d)
        assert N.MOTIF_NOM.match(nom), nom
        assert len(nom) <= N.LONGUEUR_MAX


def test_n9_mode_inconnu_refuse():
    with pytest.raises(N.NommageImpossible, match="mode"):
        N.nom_canonique(exp(mode="turbo"))


# ── N10: collision ──────────────────────────────────────────────────────────


def test_n10_definition_identique_reutilise_le_nom(tmp_path):
    ecrire(tmp_path, "exp_durmin_pop-1000_PANEL_nosim", exp())
    a = N.attribuer_nom(exp(nom="autre-chose"), tmp_path)
    assert a.nom == "exp_durmin_pop-1000_PANEL_nosim"
    assert a.reutilise == "exp_durmin_pop-1000_PANEL_nosim"
    assert a.indice == 1


def test_n10_definition_differente_prend_un_indice(tmp_path):
    # Two definitions the grammar abbreviates alike: only their tolerances differ…
    ecrire(tmp_path, "exp_durmin_pop-1000_PANEL_nosim", exp())
    autre = exp(attente_max_s=120, max_candidats=6)
    autre["gabarit"]["categorie"] = (
        "itinary_solo"  # same composed name, other definition
    )
    a = N.attribuer_nom(autre, tmp_path)
    assert a.nom == "exp_durmin_pop-1000_PANEL_nosim_2"
    assert a.reutilise is None
    assert a.voisins == ["exp_durmin_pop-1000_PANEL_nosim"]
    ecrire(tmp_path, "exp_durmin_pop-1000_PANEL_nosim_2", autre)
    troisieme = copy.deepcopy(autre)
    troisieme["gabarit"]["categorie"] = "itinary_duo"
    assert (
        N.attribuer_nom(troisieme, tmp_path).nom == "exp_durmin_pop-1000_PANEL_nosim_3"
    )


def test_n10_signature_ignore_la_seule_identite():
    a, b = exp(nom="un"), exp(nom="deux")
    b["derive_de"], b["executions_connues"] = "un", ["2026-09-08_05_00_00"]
    assert N.signature(a) == N.signature(b)
    assert N.signature(exp(memoire=True)) != N.signature(exp())


# ── N11: the index never drops ──────────────────────────────────────────────


def test_n11_indice_conserve_quand_le_nom_est_long(tmp_path, monkeypatch):
    longue = "exp_" + "a" * (N.LONGUEUR_MAX - 4)
    prises = {longue: "une-autre-signature"}
    monkeypatch.setattr(N, "nom_canonique", lambda _exp: longue)
    a = N.attribuer_nom(exp(), tmp_path, existantes=prises)
    assert a.nom.endswith("_2")
    assert len(a.nom) <= N.LONGUEUR_MAX


# ── N12/N13: the file is authoritative, `definir` checks it ─────────────────


def test_n13_verifier_nom():
    assert N.verifier_nom(exp(nom="exp_durmin_pop-1000_PANEL_nosim")) is None
    assert (
        N.verifier_nom(exp(nom="exp_durmin_pop-1000_PANEL_nosim_2")) is None
    )  # collision index accepted
    assert (
        N.verifier_nom(exp(nom="Mon_Experience")) == "exp_durmin_pop-1000_PANEL_nosim"
    )


def test_n12_le_nom_ecrit_est_autoritaire(tmp_path):
    """A historical name is still read as is: naming renames nothing that exists.

    It occupies its path without occupying the canonical name: a new identical definition
    therefore takes the canonical name, which is free, and the historical directory is untouched.
    """
    ecrire(tmp_path, "un_nom_historique", exp())
    prises = N.definitions_existantes(tmp_path)
    assert prises["un_nom_historique"] == N.signature(exp())
    a = N.attribuer_nom(exp(), tmp_path)
    assert a.nom == "exp_durmin_pop-1000_PANEL_nosim"
    assert a.reutilise is None
    assert (tmp_path / "un_nom_historique" / "experience.yaml").is_file()


def test_n7_troncature_15_consideration_set():
    """troncature_15 False is silent (default); True adds cset15."""
    nom_defaut = N.nom_canonique(exp(troncature_15=False))
    assert nom_defaut == "exp_durmin_pop-1000_PANEL_nosim"
    assert "cset15" not in nom_defaut

    nom_cset = N.nom_canonique(exp(troncature_15=True))
    assert "cset15" in nom_cset
    assert nom_cset == "exp_durmin_pop-1000_PANEL_cset15_nosim"


def test_experience_pydantic_troncature_15():
    from experiences import experience as E

    # Without the field (backward compatibility)
    e_sans = E.Experience.model_validate(exp())
    assert e_sans.troncature_15 is False

    # With the field
    e_avec = E.Experience.model_validate(exp(troncature_15=True))
    assert e_avec.troncature_15 is True
    assert E.experience_vers_dict(e_avec)["troncature_15"] is True


def test_cli_reglages_troncature_15():
    from experiences.cli import reglages_herites_de
    from settings import settings

    class FausseExp:
        vehicule_chaine = True
        verrou_retour = True
        troncature_15 = True

    e = FausseExp()
    settings.agent.mode_choice_truncation_threshold = 0.15 if e.troncature_15 else 0.0
    reg = reglages_herites_de(e)
    assert reg["troncature_15"] is True
    assert reg["mode_choice_truncation_threshold"] == 0.15

    e.troncature_15 = False
    settings.agent.mode_choice_truncation_threshold = 0.15 if e.troncature_15 else 0.0
    reg = reglages_herites_de(e)
    assert reg["troncature_15"] is False
    assert reg["mode_choice_truncation_threshold"] == 0.0
