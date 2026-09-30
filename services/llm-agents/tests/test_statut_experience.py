"""Status of an experiment — spec `hygiene-prompts-et-plateforme-experiences.md` §3.2 and §5.

The cardinal point of these tests: **nothing is deleted or moved**. A status is a
marker; runs, traces and scores remain readable and verifiable afterwards.
"""

from __future__ import annotations

import json

import pytest
import yaml

from experiences import statut as S
from experiences.registre import lister


def _experience(tmp_path, nom="exp_test_statut", avec_execution=True):
    d = tmp_path / nom
    (d / "executions" / "2026-01-01_00_00_00").mkdir(parents=True)
    (d / "experience.yaml").write_text(
        yaml.safe_dump({"nom": nom, "mode": "sans_simulateur"}), encoding="utf-8"
    )
    if avec_execution:
        ex = d / "executions" / "2026-01-01_00_00_00"
        (ex / "etat.json").write_text(json.dumps({"etat": "terminee"}), encoding="utf-8")
        (ex / "compteurs.json").write_text(
            json.dumps({"couverture": {"decides": 10, "attendus": 10, "taux": 1.0}}),
            encoding="utf-8",
        )
    return d


def test_defaut_est_actif_sans_fichier(tmp_path):
    d = _experience(tmp_path)
    st = S.lire(d)
    assert st["statut"] == S.ACTIF
    assert not S.est_masquee(st)
    assert not (d / S.F_STATUT).exists(), "the default state must write nothing"


@pytest.mark.parametrize("cible", [S.ARCHIVEE, S.INVALIDE])
def test_poser_un_statut_masque_sans_rien_supprimer(tmp_path, cible):
    d = _experience(tmp_path)
    avant = sorted(p.relative_to(d).as_posix() for p in d.rglob("*"))
    corps = S.ecrire(d, cible, motif="parce que", reference="specs/x.md")

    assert corps["statut"] == cible
    assert corps["donnees"] == "conservees"
    assert S.est_masquee(S.lire(d))
    apres = sorted(p.relative_to(d).as_posix() for p in d.rglob("*"))
    assert set(avant) - set(apres) == set(), "no file must disappear"
    assert set(apres) - set(avant) == {S.F_STATUT}, "only the marker is added"


def test_motif_obligatoire_hors_actif(tmp_path):
    d = _experience(tmp_path)
    for cible in (S.ARCHIVEE, S.INVALIDE):
        with pytest.raises(S.StatutInvalide, match="motif"):
            S.ecrire(d, cible)
        with pytest.raises(S.StatutInvalide, match="motif"):
            S.ecrire(d, cible, motif="   ")


def test_statut_inconnu_refuse(tmp_path):
    d = _experience(tmp_path)
    with pytest.raises(S.StatutInvalide, match="inconnu"):
        S.ecrire(d, "obsolete", motif="m")


def test_pas_une_experience_refuse(tmp_path):
    (tmp_path / "vide").mkdir()
    with pytest.raises(S.StatutInvalide, match="not an experiment"):
        S.ecrire(tmp_path / "vide", S.ARCHIVEE, motif="m")


def test_historique_empile_les_transitions(tmp_path):
    d = _experience(tmp_path)
    S.ecrire(d, S.ARCHIVEE, motif="doublon de nommage")
    S.ecrire(d, S.INVALIDE, motif="gabarit invalidé")
    S.ecrire(d, S.ACTIF)
    st = S.lire(d)
    assert st["statut"] == S.ACTIF
    assert [h["vers"] for h in st["historique"]] == [S.ARCHIVEE, S.INVALIDE, S.ACTIF]
    assert st["historique"][0]["motif"] == "doublon de nommage"


def test_marqueur_illisible_vaut_actif(tmp_path):
    """A registry must remain listable: the worst acceptable is "everything is shown"."""
    d = _experience(tmp_path)
    (d / S.F_STATUT).write_text("{ ceci n'est pas du json", encoding="utf-8")
    assert S.lire(d)["statut"] == S.ACTIF
    (d / S.F_STATUT).write_text(json.dumps({"statut": "n_importe_quoi"}), encoding="utf-8")
    assert S.lire(d)["statut"] == S.ACTIF


def test_registre_masque_par_defaut_et_reaffiche_sur_demande(tmp_path):
    _experience(tmp_path, "exp_visible")
    d = _experience(tmp_path, "exp_masquee")
    S.ecrire(d, S.INVALIDE, motif="gabarit invalidé (règle M1)")

    noms = {l["experience"] for l in lister(tmp_path)}
    assert noms == {"exp_visible"}

    lignes = lister(tmp_path, inclure_masquees=True)
    par_nom = {l["experience"]: l for l in lignes}
    assert set(par_nom) == {"exp_visible", "exp_masquee"}
    assert par_nom["exp_masquee"]["statut"] == S.INVALIDE
    assert "M1" in par_nom["exp_masquee"]["statut_motif"]
    assert par_nom["exp_visible"]["statut"] == S.ACTIF


def test_visible_par_defaut_garde_la_ligne(tmp_path):
    """Marking without hiding: useful for an invalidation one wants to keep in view."""
    d = _experience(tmp_path, "exp_marquee_visible")
    S.ecrire(d, S.INVALIDE, motif="à discuter", visible_par_defaut=True)
    lignes = lister(tmp_path)
    assert [l["experience"] for l in lignes] == ["exp_marquee_visible"]
    assert lignes[0]["statut"] == S.INVALIDE


def test_bandeau_sur_la_page_de_scores():
    from experiences.rendu_scores import _bandeau_statut

    assert _bandeau_statut(None) == ""
    assert _bandeau_statut({"statut": "actif"}) == ""
    html = _bandeau_statut(
        {"statut": "invalide", "motif": "gabarit invalidé", "le": "2026-09-10", "reference": "specs/x.md"}
    )
    assert "mesure invalidée" in html and "gabarit invalidé" in html and "specs/x.md" in html
    assert "ne doivent pas être cités sans ce motif" in html


def test_statuts_par_experience(tmp_path):
    _experience(tmp_path, "exp_a")
    S.ecrire(_experience(tmp_path, "exp_b"), S.ARCHIVEE, motif="vide")
    tous = S.statuts_par_experience(tmp_path)
    assert {n: s["statut"] for n, s in tous.items()} == {
        "exp_a": S.ACTIF,
        "exp_b": S.ARCHIVEE,
    }


# ---------------------------------------------------------------------------
# P6 — comparability lock (hygiene spec §8)


def _execution_comparable(tmp_path, nom, jour="2026-03-16"):
    """Two runs with identical shared fingerprints, to isolate the effect of the status."""
    d = tmp_path / nom
    ex = d / "executions" / "2026-01-01_00_00_00"
    ex.mkdir(parents=True)
    (d / "experience.yaml").write_text(
        yaml.safe_dump({"nom": nom, "mode": "sans_simulateur"}), encoding="utf-8"
    )
    (ex / "execution.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "execution1",
                "cree_le": "2026-01-01T00:00:00+00:00",
                "experience": {
                    "nom": nom,
                    "mode": "sans_simulateur",
                    "calendrier": {"politique": "aleatoire", "date": jour, "graine": 42},
                    "graine_ordre": 42,
                    "graine_tirage": 42,
                    "max_candidats": 6,
                    "horizon_jours": 1,
                    "memoire": False,
                },
                "empreintes": {"gabarit": {"categorie": "itinary_multi_agent", "sha256": "abc"}},
                "regime_demande": {"parallelisme": 1, "unite_sollicitation": "deplacement"},
                "regime_applique": {"parallelisme": 1, "unite_sollicitation": "deplacement"},
                "sources_alea": {"graine_ordre": 42, "graine_tirage": 42},
                "interruptions": [],
            }
        ),
        encoding="utf-8",
    )
    (ex / "etat.json").write_text(json.dumps({"etat": "terminee"}), encoding="utf-8")
    (ex / "compteurs.json").write_text(
        json.dumps({"couverture": {"decides": 10, "attendus": 10, "taux": 1.0}}),
        encoding="utf-8",
    )
    (ex / "moves.csv").write_text("Mode de transport Choisi\nVoiture Privée\n", encoding="utf-8")
    return ex


def test_comparer_refuse_une_experience_invalidee(tmp_path):
    from experiences.registre import ComparaisonRefusee, comparer

    a = _execution_comparable(tmp_path, "exp_a")
    b = _execution_comparable(tmp_path, "exp_b")
    S.ecrire(tmp_path / "exp_a", S.INVALIDE, motif="gabarit invalidé (M1)")

    with pytest.raises(ComparaisonRefusee) as exc:
        comparer(a, b)
    assert "invalide" in str(exc.value) and "M1" in str(exc.value)
    assert "--inclure-invalides" in str(exc.value), "the refusal must say how to override it"


def test_comparer_refuse_aussi_une_archivee(tmp_path):
    from experiences.registre import ComparaisonRefusee, comparer

    a = _execution_comparable(tmp_path, "exp_a")
    b = _execution_comparable(tmp_path, "exp_b")
    S.ecrire(tmp_path / "exp_b", S.ARCHIVEE, motif="doublon de nommage")
    with pytest.raises(ComparaisonRefusee):
        comparer(a, b)


def test_comparer_force_rappelle_le_motif(tmp_path):
    """Forcing is allowed, but the reason travels with the gap table."""
    from experiences.registre import comparer, formater_comparaison

    a = _execution_comparable(tmp_path, "exp_a")
    b = _execution_comparable(tmp_path, "exp_b")
    S.ecrire(tmp_path / "exp_a", S.INVALIDE, motif="gabarit invalidé (M1)")

    c = comparer(a, b, inclure_invalides=True)
    assert c["force"] is True
    assert c["statuts"] == {"a": S.INVALIDE, "b": S.ACTIF}
    texte = formater_comparaison(c)
    assert "COMPARAISON FORCÉE" in texte
    assert "gabarit invalidé (M1)" in texte
    assert "[a] [invalide]" in texte


def test_comparer_deux_actives_ne_change_pas(tmp_path):
    """The lock is targeted: two active experiments compare as before."""
    from experiences.registre import comparer

    a = _execution_comparable(tmp_path, "exp_a")
    b = _execution_comparable(tmp_path, "exp_b")
    c = comparer(a, b)
    assert c["force"] is False
    assert c["comparable"] is True
    assert c["statuts"] == {"a": S.ACTIF, "b": S.ACTIF}
