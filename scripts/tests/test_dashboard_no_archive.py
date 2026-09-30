"""The dashboard offers NOTHING archived, anywhere.

R16: `populations()` exposes no archived cohort, and its default is the latest sealed one
      by SEAL DATE, never the first in alphabetical order.
R17: no archived item is offered for selection, including when « Masquer les obsolètes »
      is UNCHECKED — this button sets the visibility of what is obsolete, not of what is
      archived.

What these tests lock, and why. Two distinct notions carried until now the same
word "obsolete":

- an *obsolete* run is a run that a more recent one of the SAME experiment has
  replaced; it is the one the checkbox governs, and that is quite right;
- an *archived* experiment is an experiment withdrawn from service by a decision.

The second must never depend on the first. Yet the exclusion of archived ones only held
in the « Mes expériences » table: the « S'inspirer d'une expérience existante » selector
and the « Dupliquer » button read `experiences()`, which nothing filters. An archived experiment
thus stayed offered there, and a click copied it into the form.

For populations, the exclusion of an `archive/` folder was only a **structural
accident**: `populations()` only keeps a folder if it carries a `MANIFEST.yaml` at its
direct root, which an `archive/` folder does not have. Nothing stated it, nothing tested it, and
the rule would have fallen at the first archived cohort keeping its manifest at the wrong level.
The form default, for its part, was `populations()[0]`, that is the FIRST in alphabetical
order: `population_1000_PANEL` precedes `_v3`, `_v4`, `_v5`. This is the root cause of the
36 runs made on the wrong cohort.
"""

import json
import sys
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


def _cohorte(dossier: Path, nom: str, scelle_le: str | None) -> Path:
    """A CREDIBLE sealed cohort: the seal carries the true fingerprint of the file.

    An invented seal (`"x" * 64`) would be refused on loading — `info_population` checks
    that a sealed population has not been altered. A fixture that bypasses this check
    makes tests pass that prove nothing.
    """
    import hashlib

    d = dossier / nom
    d.mkdir(parents=True)
    fichier = d / "population.json"
    fichier.write_text("[]", encoding="utf-8")
    manifest = {
        "nom": nom,
        "population": {
            "fichier": "population.json",
            "sha256": hashlib.sha256(fichier.read_bytes()).hexdigest(),
        },
    }
    if scelle_le:
        manifest["scelle_le"] = scelle_le
    (d / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return d


def _sha_contenu(dossier_cohorte: Path) -> str:
    """The CONTENT fingerprint of a cohort: that of its `population.json`."""
    import hashlib

    return hashlib.sha256((dossier_cohorte / "population.json").read_bytes()).hexdigest()


def _experience(dossier: Path, nom: str, statut: str | None = None) -> Path:
    d = dossier / nom
    d.mkdir(parents=True)
    (d / "experience.yaml").write_text(yaml.safe_dump({"nom": nom}), encoding="utf-8")
    if statut:
        (d / "statut.json").write_text(
            json.dumps({"statut": statut, "motif": "essai"}), encoding="utf-8"
        )
    return d


def _jeu(dossier: Path, nom: str) -> Path:
    d = dossier / nom
    d.mkdir(parents=True)
    (d / "MANIFEST.yaml").write_text(
        yaml.safe_dump({"nom": nom, "jour_simule": "2026-03-16", "clos": True}),
        encoding="utf-8",
    )
    return d


# ── R16: the populations ────────────────────────────────────────────────────


def test_r16_une_cohorte_sous_archive_nest_jamais_proposee(monkeypatch, tmp_path):
    """Even if it keeps its MANIFEST.yaml: the location decides, not the shape."""
    pop = tmp_path / "data" / "population"
    _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    _cohorte(pop / "archive", "population_1000_PANEL", "2026-09-02T00:00:00+00:00")
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    listees = experiences.populations()
    assert any("population_1000_PANEL_v5" in p for p in listees)
    assert not any("archive" in p for p in listees), listees


def test_r16_un_json_nu_sous_archive_nest_pas_propose(monkeypatch, tmp_path):
    pop = tmp_path / "data" / "population"
    _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    (pop / "archive").mkdir(parents=True, exist_ok=True)
    (pop / "archive" / "vieille.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    assert not any("archive" in p for p in experiences.populations())


def test_r16_le_defaut_est_la_derniere_scellee_pas_la_premiere_alphabetique(
    monkeypatch, tmp_path
):
    """The root cause of the 36 runs on the wrong cohort.

    `population_1000_PANEL` precedes `_v5` alphabetically; yet `_v5` is the
    reference, because it was sealed later.
    """
    pop = tmp_path / "data" / "population"
    _cohorte(pop, "population_1000_PANEL", "2026-09-02T20:39:00+00:00")
    _cohorte(pop, "population_1000_PANEL_v3", "2026-09-03T18:58:00+00:00")
    _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    assert experiences.population_par_defaut().endswith("population_1000_PANEL_v5")


def test_r16_une_cohorte_sans_date_de_sceau_ne_prend_pas_la_tete(monkeypatch, tmp_path):
    """A cohort without `scelle_le` cannot be "the most recent": it comes after.

    Without this rule, a badly sealed cohort would become the default by the mere fact that its
    date is missing — again the "absence of measurement makes the best score" pattern.
    """
    pop = tmp_path / "data" / "population"
    _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    _cohorte(pop, "zz_population_sans_date", None)
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    assert experiences.population_par_defaut().endswith("population_1000_PANEL_v5")


# ── R17: the frozen sets ────────────────────────────────────────────────────


def test_r17_un_jeu_sous_archive_nest_pas_propose(monkeypatch, tmp_path):
    jeux = tmp_path / "data" / "jeux"
    _jeu(jeux, "population_1000_PANEL_v5_20260316")
    _jeu(jeux / "archive", "population_1000_PANEL_20260316")
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)

    noms = [j["nom"] for j in experiences.jeux()]
    assert noms == ["population_1000_PANEL_v5_20260316"]


# ── R17: the experiments, wherever they are offered ─────────────────────────


def test_r17_experiences_exclut_les_archivees_et_les_invalides(monkeypatch, tmp_path):
    """`experiences()` serves « S'inspirer de » and « Dupliquer »: it must filter too.

    That was the gap: the « Mes expériences » table filtered, those two did not.
    """
    _experience(tmp_path, "exp_vivante")
    _experience(tmp_path, "exp_rangee", statut="archivee")
    _experience(tmp_path, "exp_cassee", statut="invalide")
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)

    proposees = experiences.experiences()
    assert "exp_vivante" in proposees
    assert "exp_rangee" not in proposees
    assert "exp_cassee" not in proposees


def test_r17_experiences_toutes_reste_disponible_pour_la_lecture(monkeypatch, tmp_path):
    """Filtering the proposals must not make the archived ones UNREADABLE.

    A detail page, a report, a history replay must still be able to
    open them: the rule is "they are no longer OFFERED", not "they are erased".
    """
    _experience(tmp_path, "exp_vivante")
    _experience(tmp_path, "exp_rangee", statut="archivee")
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)

    toutes = experiences.experiences(inclure_masquees=True)
    assert {"exp_vivante", "exp_rangee"} <= set(toutes)


def test_r17_le_bouton_masquer_les_obsoletes_ne_gouverne_pas_les_archivees(
    monkeypatch, tmp_path
):
    """Unchecked, it brings back the obsolete ones — never the archived ones.

    The two notions are distinct and must not be confused: "obsolete" is a
    run replaced by a more recent one of the same experiment, "archived" is a
    withdrawal decision.
    """
    _experience(tmp_path, "exp_vivante")
    _experience(tmp_path, "exp_rangee", statut="archivee")
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)

    lignes = experiences.lister(tmp_path)
    rangees = [l for l in lignes if l.get("experience") == "exp_rangee"]
    # `lister()` exposes everything — the view does the filtering — but it must MARK the status,
    # otherwise no view can decide.
    for ligne in rangees:
        assert experiences.masquee(ligne) is True


# ── R19: the substrate is displayed next to the launch button ───────────────


def test_r19_lempreinte_de_la_cohorte_est_lisible_avant_le_lancement(monkeypatch, tmp_path):
    """The first 36 runs read the wrong cohort: nothing said so BEFORE paying.

    ⚠ This test fixes the distinction that produced a false « substrat incohérent » on
    2026-09-11: **two quantities carry the name `sha256`**. The IDENTITY of a cohort is
    the fingerprint of its manifest FILE — it is the one the frozen set records and compares. Its
    CONTENT is the fingerprint of `population.json`, the `population.sha256` field written IN the
    manifest. Confusing them makes the inconsistency alarm go off on a perfectly sound cohort.
    """
    import hashlib

    pop = tmp_path / "data" / "population"
    d = _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    info = experiences.empreinte_population("data/population/population_1000_PANEL_v5")
    assert info["scellee"] is True
    assert info["scelle_le"].startswith("2026-09-04")
    # The identity: the manifest FILE, not the field it contains.
    assert info["sha256"] == hashlib.sha256((d / "MANIFEST.yaml").read_bytes()).hexdigest()
    # The content: the manifest field, which describes `population.json`.
    assert info["fichier_sha256"] == _sha_contenu(d)
    assert info["sha256"] != info["fichier_sha256"], "the two fingerprints are distinct"


def test_r19_une_cohorte_non_scellee_se_signale(monkeypatch, tmp_path):
    """Without a seal, no stable identity: the measurement could be attached to nothing."""
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)
    info = experiences.empreinte_population("data/population/toulouse_population_1000.json")
    assert info["scellee"] is False
    assert info["sha256"] == ""


def test_r19_lidentite_est_celle_que_le_jeu_compare(monkeypatch, tmp_path):
    """The dashboard guard must read the SAME fingerprint as the launch guard.

    Two implementations of the same concept end up diverging: that is the defect fixed
    elsewhere, and it happened again here. This test forbids it by comparing
    against the source — `info_population()`, whose result `Jeu.verifier_population` consumes.
    """
    from experiences.population import info_population

    pop = tmp_path / "data" / "population"
    d = _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    vue = experiences.empreinte_population("data/population/population_1000_PANEL_v5")
    plateforme = info_population(d)
    assert vue["sha256"] == plateforme.sha256
    assert vue["fichier_sha256"] == plateforme.fichier_sha256


def test_r19_un_jeu_prepare_pour_une_autre_cohorte_est_refuse_avant_le_clic(
    monkeypatch, tmp_path
):
    """The guard exists at launch (G1); moving it before the click avoids paying for nothing."""
    pop = tmp_path / "data" / "population"
    _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    jeux = tmp_path / "data" / "jeux"
    d = jeux / "jeu_de_la_v1"
    d.mkdir(parents=True)
    (d / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "nom": "jeu_de_la_v1",
                # Identity of ANOTHER cohort: plausible in shape, different in value.
                "population": {"nom": "population_1000_PANEL", "sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    msg = experiences.coherence_population_jeu(
        "data/population/population_1000_PANEL_v5", "jeu_de_la_v1"
    )
    assert msg and "population_1000_PANEL" in msg


def test_r19_un_jeu_de_la_bonne_cohorte_ne_declenche_rien(monkeypatch, tmp_path):
    pop = tmp_path / "data" / "population"
    _cohorte(pop, "population_1000_PANEL_v5", "2026-09-04T13:47:53+00:00")
    jeux = tmp_path / "data" / "jeux"
    d = jeux / "jeu_v5"
    d.mkdir(parents=True)
    # The frozen set records the IDENTITY fingerprint of the cohort: that of its manifest
    # FILE, not the `sha256` field it contains. Writing the content fingerprint here
    # would make this test pass as a success while the guard compared two
    # different things — this is exactly the false positive of 2026-09-11.
    import hashlib

    identite = hashlib.sha256(
        (pop / "population_1000_PANEL_v5" / "MANIFEST.yaml").read_bytes()
    ).hexdigest()
    (d / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "nom": "jeu_v5",
                "population": {"nom": "population_1000_PANEL_v5", "sha256": identite},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiences, "DOSSIER_POP", pop)
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)

    assert (
        experiences.coherence_population_jeu(
            "data/population/population_1000_PANEL_v5", "jeu_v5"
        )
        is None
    )


# ── The vehicle chain is spelled out in full ────────────────────────────────


def test_la_colonne_chaine_dit_active_par_defaut():
    """A silent definition describes the nominal behaviour, not a gap."""
    assert experiences.libelle_chaine({}) == "active"
    assert experiences.libelle_chaine({"vehicule_chaine": True, "verrou_retour": True}) == "active"


def test_la_colonne_chaine_dit_coupee_sans_faire_decoder_le_nom():
    """`nochn_noret` in the name assumes knowing the convention; the column does not."""
    assert (
        experiences.libelle_chaine({"vehicule_chaine": False, "verrou_retour": False})
        == "coupée"
    )


def test_la_colonne_distingue_les_deux_interrupteurs():
    """Cutting one without the other is possible, and the label says what REMAINS.

    « position » on its own would not say whether it is active or cut: the label names
    the switch still in service, never the one that has dropped.
    """
    seule_position = {"vehicule_chaine": True, "verrou_retour": False}
    seul_verrou = {"vehicule_chaine": False, "verrou_retour": True}
    assert experiences.libelle_chaine(seule_position) == "position seule"
    assert experiences.libelle_chaine(seul_verrou) == "verrou seul"


def test_la_colonne_chaine_est_affichable_dans_le_tableau():
    """A value computed but never displayable is useless.

    Since 2026-09-11 the table opens on ten columns and `chaine` is no longer
    among them: it remains RECALLABLE in the column selector, which is enough for it to
    serve — the comparability of two runs is checked on demand, not permanently.
    """
    assert "chaine" in experiences.COLONNES_REGISTRE
    assert "chaine" not in experiences.COLONNES_REGISTRE_DEFAUT


def test_chaque_ligne_listee_porte_son_etat_de_chaine(monkeypatch, tmp_path):
    d = _experience(tmp_path, "exp_test")
    (d / "experience.yaml").write_text(
        yaml.safe_dump({"nom": "exp_test", "vehicule_chaine": False, "verrou_retour": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(experiences, "DOSSIER", tmp_path)
    lignes = experiences.lister(tmp_path)
    assert lignes and all(l.get("chaine") == "coupée" for l in lignes)
