"""Saving an experiment without launching it (spec `enregistrer-experience-non-lancee.md`).

The thread of these tests: write one's plan today and launch tomorrow. The write is local and must
depend neither on the `controller` container nor on the existence of the trip set — the platform
schema accepts an experiment that names a set not yet built, and the warm-up lasts
an hour.
"""

import functools
import json
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402

VRAI_LISTER = experiences.lister


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    texte = json.dumps(contenu) if chemin.suffix == ".json" else yaml.safe_dump(contenu, allow_unicode=True)
    chemin.write_text(texte, encoding="utf-8")


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    """A pocket repository: two populations, a closed set for one, nothing for the other."""
    exps, jeux, pops = tmp_path / "experiences", tmp_path / "jeux", tmp_path / "population"
    for nom in ("pop_avec_jeu", "pop_sans_jeu"):
        _ecrire(pops / nom / "MANIFEST.yaml", {"nom": nom, "n": 10})
    _ecrire(jeux / "pop_avec_jeu_20260316" / "MANIFEST.yaml",
            {"nom": "pop_avec_jeu_20260316", "clos": True, "population": {"nom": "pop_avec_jeu"},
             "jour_simule": "2026-03-16", "attendus": {"deplacements": 100}, "couverts": {"deplacements": 98}})
    monkeypatch.setattr(experiences, "DOSSIER", exps)
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)
    monkeypatch.setattr(experiences, "DOSSIER_POP", pops)
    monkeypatch.setattr(experiences, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(experiences, "lister", functools.partial(VRAI_LISTER, dossier=exps))
    return tmp_path


def _valeurs(**surcharges) -> dict:
    """The form values, as `_formulaire` returns them (no name: it is computed)."""
    base = {
        "population": "data/population/pop_sans_jeu", "jeu": "", "sans_jeu": True,
        "variante": "b_min", "decideur_type": "passerelle", "modele": "m1", "temperature": 0.0,
        "graine_decideur": 42, "rejeu_de": "", "mode": "sans_simulateur", "politique": "commune",
        "date": "2026-03-16", "graine_calendrier": 42, "horizon_jours": 1, "memoire": False,
        "graine_ordre": 42, "graine_tirage": 42, "parallelisme": 8, "max_candidats": 6,
        "attente_max_s": 120, "tolerances": dict(experiences.TOLERANCES_PROPOSEES), "derive_de": None,
    }
    base.update(surcharges)
    return base


# The name is computed from the parameters (N1): the tests read it instead of setting it. Two
# experiments are therefore told apart by a PARAMETER — here the model — never by a label.
NOM_DEFAUT = "exp_m1_bmin_pop-pop_sans_jeu_t0_nosim"


def _nom(**surcharges) -> str:
    return experiences.construire_experience(_valeurs(**surcharges))["nom"]


def test_R1_enregistrer_ne_demande_ni_jeu_ni_conteneur(plateforme):
    exp = experiences.construire_experience(_valeurs())
    motifs = experiences.motifs_indisponibilite(exp, jeu_clos=False, controleur_ok=False, registre=False)
    assert motifs["enregistrer"] == [], "an admissible name is enough to save"
    assert motifs["lancer"], "launching, however, stays blocked"

    chemin, change = experiences.enregistrer(exp)
    assert change is True and chemin.is_file()
    assert yaml.safe_load(chemin.read_text(encoding="utf-8"))["nom"] == NOM_DEFAUT
    assert not (chemin.parent / "executions").exists(), "no run must be created"


def test_R2_l_experience_nomme_le_jeu_qu_elle_attend(plateforme):
    exp = experiences.construire_experience(_valeurs())
    attendu = experiences.nom_jeu_attendu("data/population/pop_sans_jeu", "2026-03-16")
    assert exp["jeu"]["nom"] == attendu == "pop_sans_jeu_20260316"

    # When a set exists, it is the one named, not a computed name
    avec = experiences.construire_experience(
        _valeurs(population="data/population/pop_avec_jeu", jeu="pop_avec_jeu_20260316", sans_jeu=False))
    assert avec["jeu"]["nom"] == "pop_avec_jeu_20260316"


def test_R3_la_validation_est_une_etape_distincte(plateforme):
    exp = experiences.construire_experience(_valeurs())

    arrete = experiences._enregistrer_et_dire(
        exp, inline=None, sans_validation="le service `controller` ne tourne pas")
    assert "validation non tentée" in arrete
    assert "`controller` ne tourne pas" in arrete
    assert "raté" not in arrete and "erreur" not in arrete.lower()

    appels = []
    def faux_inline(cible, variables, **kwargs):
        appels.append((cible, variables, kwargs))
        return f"expérience {NOM_DEFAUT!r} validée et rangée"

    actif = experiences._enregistrer_et_dire(exp, inline=faux_inline, sans_validation=None)
    assert appels and appels[0][0] == "experience-definir"
    assert "validée et rangée" in actif


def test_R4_le_message_dit_le_chemin_et_l_etat_du_jeu(plateforme):
    message = experiences._enregistrer_et_dire(
        experiences.construire_experience(_valeurs()), inline=None, sans_validation="pas de conteneur")
    # A new experiment is filed in the family of its set, as by the CLI.
    assert f"experiences/regime_nominal/pop_sans_jeu_20260316/{NOM_DEFAUT}/experience.yaml" in message
    assert "jeu attendu « pop_sans_jeu_20260316 » : absent" in message

    pret = experiences._enregistrer_et_dire(
        experiences.construire_experience(
            _valeurs(population="data/population/pop_avec_jeu",
                     jeu="pop_avec_jeu_20260316", sans_jeu=False)),
        inline=None, sans_validation="pas de conteneur")
    assert "clos et prêt" in pret


def test_R5_seul_un_nom_impossible_empeche_d_enregistrer(plateforme):
    """The name no longer being typed (N1), the only remaining case is a naming that REFUSES."""
    sans_modele = experiences.construire_experience(_valeurs(modele=""))
    assert sans_modele["nom"] == ""
    motifs = experiences.motifs_indisponibilite(sans_modele, jeu_clos=False, controleur_ok=False,
                                                registre=False)
    assert any("decideur.modele" in m for m in motifs["enregistrer"]), motifs["enregistrer"]

    exp = experiences.construire_experience(_valeurs())
    assert exp["nom"] == NOM_DEFAUT
    assert experiences.MOTIF_NOM.match(exp["nom"])
    assert experiences.motifs_indisponibilite(exp, jeu_clos=False, controleur_ok=False,
                                              registre=False)["enregistrer"] == []


def test_R6_un_nom_deja_execute_demande_confirmation(plateforme):
    exp = experiences.construire_experience(_valeurs())
    chemin, _ = experiences.enregistrer(exp)
    dossier = chemin.parent / "executions"
    for horodatage in ("2026-09-01_10_00_00", "2026-09-02_11_00_00"):
        _ecrire(dossier / horodatage / "execution.yaml", {"experience": {"nom": exp["nom"]}})
        _ecrire(dossier / horodatage / "etat.json", {"etat": "terminee"})
    assert experiences.executions_connues(exp["nom"]) == 2

    empreintes = {p: p.read_bytes() for p in dossier.rglob("*") if p.is_file()}

    non_confirme = experiences.motifs_indisponibilite(
        exp, jeu_clos=False, controleur_ok=False, registre=False,
        ecrasement=f"« {exp['nom']} » porte déjà 2 exécution(s) : cochez la confirmation ci-dessus")
    assert any("2 exécution(s)" in m for m in non_confirme["enregistrer"])

    confirme = experiences.motifs_indisponibilite(exp, jeu_clos=False, controleur_ok=False, registre=False)
    assert confirme["enregistrer"] == []

    experiences.enregistrer({**exp, "max_candidats": 9})
    assert {p: p.read_bytes() for p in dossier.rglob("*") if p.is_file()} == empreintes, \
        "archived runs keep their frozen copy of the definition"


def test_R6bis_estimer_et_lancer_passent_aussi_par_la_confirmation(plateforme):
    """The overwrite safeguard (R6) must cover Estimate and Launch, not only Save.

    These two buttons write the definition (enregistrer() upstream): as long as they ignored it,
    launching a run with a modified decision-maker silently switched the definition back.
    """
    exp = experiences.construire_experience(
        _valeurs(population="data/population/pop_avec_jeu",
                 jeu="pop_avec_jeu_20260316", sans_jeu=False))
    chemin, _ = experiences.enregistrer(exp)
    dossier = chemin.parent / "executions"
    _ecrire(dossier / "2026-09-01_10_00_00" / "execution.yaml", {"experience": {"nom": exp["nom"]}})
    _ecrire(dossier / "2026-09-01_10_00_00" / "etat.json", {"etat": "terminee"})
    assert experiences.executions_connues(exp["nom"]) == 1

    motif = f"« {exp['nom']} » porte déjà 1 exécution(s) : cochez la confirmation ci-dessus"
    bloques = experiences.motifs_indisponibilite(
        exp, jeu_clos=True, controleur_ok=True, registre=True, ecrasement=motif)
    assert any("exécution(s)" in m for m in bloques["lancer"]), bloques["lancer"]
    assert any("exécution(s)" in m for m in bloques["estimer"]), bloques["estimer"]

    # Confirmed (ecrasement=None): nothing blocks on the overwrite side any more.
    ok = experiences.motifs_indisponibilite(exp, jeu_clos=True, controleur_ok=True, registre=True)
    assert ok["lancer"] == [] and ok["estimer"] == []


def test_R6ter_le_decideur_affiche_est_celui_fige_par_execution(plateforme):
    """Two runs of the same experiment may carry different decision-makers: the table
    shows the one frozen in each run's snapshot, never the (mutable) one of the definition."""
    definition = experiences.construire_experience(_valeurs(modele="mistral-small-latest"))
    chemin, _ = experiences.enregistrer(definition)
    nom = definition["nom"]
    dossier = chemin.parent / "executions"
    fige = {"2026-09-01_10_00_00": "gemini-3.5-flash-lite", "2026-09-02_11_00_00": "mistral-small-latest"}
    for horodatage, modele in fige.items():
        _ecrire(dossier / horodatage / "execution.yaml",
                {"experience": {"nom": nom,
                                "decideur": {"type": "passerelle", "modele": modele}}})
        _ecrire(dossier / horodatage / "etat.json", {"etat": "terminee"})

    par_exec = {l["execution"]: l["decideur"] for l in experiences.lister()
                if l["experience"] == nom}
    assert par_exec["2026-09-01_10_00_00"] == "gemini-3.5-flash-lite"
    assert par_exec["2026-09-02_11_00_00"] == "mistral-small-latest"


def test_R7_le_jeu_attendu_suit_le_jour_choisi(plateforme):
    veille = experiences.construire_experience(_valeurs(date="2026-03-16"))
    lendemain = experiences.construire_experience(_valeurs(date="2026-03-17"))
    assert veille["jeu"]["nom"] == "pop_sans_jeu_20260316"
    assert lendemain["jeu"]["nom"] == "pop_sans_jeu_20260317"

    chemin, _ = experiences.enregistrer(lendemain)
    assert yaml.safe_load(chemin.read_text(encoding="utf-8"))["jeu"]["nom"] == "pop_sans_jeu_20260317"


def test_R8_enregistrer_deux_fois_sans_changement_ne_fait_rien(plateforme):
    exp = experiences.construire_experience(_valeurs())
    chemin, premier = experiences.enregistrer(exp)
    empreinte = chemin.stat().st_mtime_ns

    chemin2, second = experiences.enregistrer(exp)
    assert (premier, second) == (True, False)
    assert chemin2 == chemin and chemin.stat().st_mtime_ns == empreinte

    message = experiences._enregistrer_et_dire(exp, inline=None, sans_validation="pas de conteneur")
    assert "inchangé" in message and "le fichier disait déjà cela" in message
    assert not (chemin.parent / "executions").exists()


def test_R9_le_registre_dit_si_le_jeu_est_pret(plateforme):
    a_construire = experiences.construire_experience(_valeurs())
    deja_pret = experiences.construire_experience(
        _valeurs(population="data/population/pop_avec_jeu", jeu="pop_avec_jeu_20260316",
                 sans_jeu=False))
    for exp in (a_construire, deja_pret):
        experiences.enregistrer(exp)

    par_nom = {l["experience"]: l for l in experiences.lister()}
    assert par_nom[a_construire["nom"]]["jeu_etat"] == "absent"
    assert par_nom[deja_pret["nom"]]["jeu_etat"] == "clos et prêt"
    assert a_construire["nom"] != deja_pret["nom"], "two populations, two names (N7)"


def test_R10_la_validation_n_est_pas_tentee_sur_un_etat_inconnu(plateforme):
    """The unknown is not a "yes": otherwise a hanging Docker daemon would make the page wait."""
    appels = []
    faux_inline = lambda *a, **k: appels.append(a) or "ne devrait pas être appelé"  # noqa: E731

    assert experiences.motif_sans_validation(None, faux_inline) == \
        "l'état des services Docker est inconnu (docker injoignable)"
    assert experiences.motif_sans_validation({"api", "worker"}, faux_inline) == \
        "le service `controller` ne tourne pas"
    assert experiences.motif_sans_validation({"controller"}, None) is not None
    assert experiences.motif_sans_validation({"controller", "api"}, faux_inline) is None

    exp = experiences.construire_experience(_valeurs())
    for services in (None, {"api"}):
        message = experiences._enregistrer_et_dire(
            exp, inline=faux_inline, sans_validation=experiences.motif_sans_validation(services, faux_inline))
        assert "validation non tentée" in message
    assert appels == [], "no command must be launched when validation is not attempted"


def test_R11_la_sortie_de_validation_est_etiquetee_et_bornee(plateforme):
    """A validation failure must not be readable as a save failure."""
    vus = []

    def faux_inline(cible, variables, **kwargs):
        vus.append(kwargs)
        return "expérience refusée : decideur.modele est obligatoire"

    message = experiences._enregistrer_et_dire(
        experiences.construire_experience(_valeurs()), inline=faux_inline, sans_validation=None)

    assert "écrit :" in message, "the save is announced first"
    lignes = message.splitlines()
    etiquette = next(i for i, l in enumerate(lignes) if l.startswith("validation par la plateforme"))
    assert "refusée" in "\n".join(lignes[etiquette:]), "the failure is under the validation label"
    assert vus and vus[0]["timeout"] == experiences.DELAI_VALIDATION_S == 20


def test_le_prompt_affiche_ne_l_est_que_pour_un_decideur_qui_en_lit_un(plateforme):
    """The table shows a prompt variant only if the run will read one (N5).

    Two SYMMETRIC defects are covered here, and it took both for this test to say
    anything. On 2026-09-08, `exp_lgbm_jtir_nosim` (LightGBM) announced
    `minimal_persona` whereas its archive carries not a single LLM exchange: a prompt
    shown for a decision-maker that reads none. On 2026-09-21, the reverse: `typesafe`
    was missing from the list and EVERY Jev run showed "—", whereas its instruction names
    its experiment and is sealed in its hash. This test only listed silent
    decision-makers, so it passed in both cases — that omission is what let the
    second defect slip through.

    `typesafe` receives the variant stripped of its output block; the COLUMN still
    shows the bare name, to keep a single value per variant in the filter. The truncation
    is stated in the detail sheet (cf. `test_la_fiche_dit_la_troncature_du_prompt_de_jev`).
    """
    attendu = {}
    for surcharges, prompt in (
        ({"decideur_type": "passerelle", "modele": "m1"}, "b_min"),
        ({"decideur_type": "typesafe", "modele": "jev-1.13.0"}, "b_min"),
        ({"decideur_type": "modele"}, "—"),
        ({"decideur_type": "aleatoire"}, "—"),
        ({"decideur_type": "duree_minimale"}, "—"),
        ({"decideur_type": "rejeu", "rejeu_de": "/app/data/experiences/x/executions/y"}, "—"),
    ):
        exp = experiences.construire_experience(_valeurs(variante="b_min", **surcharges))
        experiences.enregistrer(exp)
        assert exp["gabarit"]["variante"] == "b_min", "the definition keeps the field as is"
        attendu[exp["nom"]] = prompt

    par_nom = {l["experience"]: l["prompt"] for l in experiences.lister()}
    assert {n: par_nom[n] for n in attendu} == attendu


def test_le_prompt_affiche_suit_le_decideur_fige_de_chaque_execution(plateforme):
    """Like the decision-maker, the prompt shown is the one of the run's snapshot, not of the
    current definition: an experiment redefined from the gateway to the statistical model
    keeps a prompt on its old runs and "—" on the new ones."""
    definition = experiences.construire_experience(_valeurs(variante="b_min"))
    chemin, _ = experiences.enregistrer(definition)
    nom = definition["nom"]
    dossier = chemin.parent / "executions"
    fige = {
        "2026-09-01_10_00_00": {"gabarit": {"variante": "expert_chaine"},
                                "decideur": {"type": "passerelle", "modele": "m1"}},
        "2026-09-02_11_00_00": {"gabarit": {"variante": "expert_chaine"},
                                "decideur": {"type": "modele"}},
        # Partial snapshot (no decision-maker): the definition is authoritative, for lack of better.
        "2026-09-03_12_00_00": {},
    }
    for horodatage, instantane in fige.items():
        _ecrire(dossier / horodatage / "execution.yaml",
                {"experience": {"nom": nom, **instantane}})
        _ecrire(dossier / horodatage / "etat.json", {"etat": "terminee"})

    par_exec = {l["execution"]: l["prompt"] for l in experiences.lister()
                if l["experience"] == nom}
    assert par_exec["2026-09-01_10_00_00"] == "expert_chaine"
    assert par_exec["2026-09-02_11_00_00"] == "—"
    assert par_exec["2026-09-03_12_00_00"] == "b_min"


def test_sans_variante_designee_la_passerelle_dit_le_prompt_actif(plateforme):
    """Empty variant = the gateway's ACTIVE prompt, "today's one": the table says so
    instead of leaving the column empty."""
    assert experiences._prompt_affiche(None, {"type": "passerelle", "modele": "m1"}) == "actif"
    assert experiences._prompt_affiche("", {"type": "passerelle"}) == "actif"
    assert experiences._prompt_affiche(None, {"type": "modele"}) == "—"
    assert experiences._prompt_affiche(None, None) == "—", "unknown decision-maker: nothing to promise"


def test_terminee_ne_se_confond_pas_avec_les_autres_etats_finaux():
    """The `Terminées seulement` checkbox keeps only `terminee`. `epuisee`, `arretee` and
    `interrompue` are final — nothing writes any more — but their coverage is partial:
    keeping them would pass an incomplete result off as a result (E14)."""
    assert experiences.est_terminee({"etat": "terminee"})
    for etat in ("definie", "en_cours", "en_pause", "en_attente_quota",
                 "epuisee", "arretee", "interrompue", "archive manquante", "?"):
        assert not experiences.est_terminee({"etat": etat}), etat
    assert not experiences.est_terminee({}) and not experiences.est_terminee(None)


def test_le_formulaire_sait_declarer_une_experience_jev(plateforme):
    """A Jev experiment is created from the dashboard, not only through its YAML.

    Defect of 2026-09-21: the form wrote `modele` only for the gateway and
    Antigravity. For `typesafe`, `construire_experience` left `modele: None`; yet
    `segment_decideur` names this decision-maker after its VERSION (`jev-1130`), so the name
    came out EMPTY and `enregistrer` refused — without anything saying that the missing field
    was the version. The six experiments of the mutation batch had to be written by hand.

    This test holds the three links together: the field is written, the name is computed, and the
    file is filed. Breaking any of the three makes it fail.
    """
    exp = experiences.construire_experience(
        _valeurs(variante="b_min", decideur_type="typesafe", modele="jev-1.13.0")
    )
    assert exp["decideur"]["type"] == "typesafe"
    assert exp["decideur"]["modele"] == "jev-1.13.0", "the version is WRITTEN, otherwise the name is not computed"
    assert exp["decideur"]["parametres"] == {}, (
        "neither temperature nor thinking: Jev has none, and sealing them would suggest an applied setting"
    )
    assert "jev-1130" in exp["nom"], f"the name carries the version: {exp['nom']}"
    assert "bmin" in exp["nom"], f"le nom porte la variante de prompt (N5) : {exp['nom']}"

    chemin, _ = experiences.enregistrer(exp)
    assert chemin.is_file()
    relu = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    assert relu["decideur"]["modele"] == "jev-1.13.0"
    assert relu["gabarit"]["variante"] == "b_min"


def test_l_horizon_declarable_est_borne_au_garde_fou_de_cinquante_jours():
    """The observation cap lives in ONE constant, and it does bound the form.

    Decision of 2026-09-22: fifty days. The figure was 31 as long as the duration cap
    of a memory was 30; both moved together, and nothing said so. This test ties
    the form to the constant — the day one of the two moves again, the other shows it.
    """
    assert experiences.HORIZON_MAX_JOURS == 50
    assert experiences._BORNES["horizon_jours"] == (1, 50, int)

    # A draft asking for more returns to the cap, it does not break the page.
    assert experiences._valider_base({"horizon_jours": 400})["horizon_jours"] == 50
    assert experiences._valider_base({"horizon_jours": 0})["horizon_jours"] == 1
    assert experiences._valider_base({"horizon_jours": 42})["horizon_jours"] == 42
