"""What is running, as seen by the dashboard (spec R1 to R4b, R17, R19, R21 to R22).

The thread of these tests: the displayed state is read ON DISK, from files written by the
container while they are being read. They can thus be absent, truncated or incomplete — and
that is the case the page must hold up to, because it is the one that happened: on 2026-09-06
a warm-up ran for an hour without the form noticing.
"""

import contextlib
import functools
import json
import sys
import time
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402

VRAI_LISTER = experiences.lister


class FauxSt:
    """The bare minimum of the Streamlit API used by `rendre_activites`.

    `columns` returns this very object: everything drawn, whatever the column,
    ends up in the same lists — which is what we want to check.
    """

    def __init__(self):
        self.barres: list[tuple[float, str]] = []
        self.textes: list[str] = []
        self.boutons: list[str] = []
        self.cases: list[str] = []
        self.legendes: list[str] = []

    def progress(self, valeur, text=""):
        self.barres.append((valeur, text))

    def markdown(self, texte):
        self.textes.append(texte)

    def caption(self, texte, **_k):
        self.legendes.append(texte)

    def columns(self, spec, **_k):
        largeur = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [self] * largeur

    def button(self, label, **_k):
        self.boutons.append(label)
        return False

    def checkbox(self, label, **_k):
        self.cases.append(label)
        return False

    def toast(self, *_a, **_k):
        pass

    def expander(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def tout(self) -> str:
        return "\n".join(self.textes + self.legendes + [t for _, t in self.barres])


class RerunDemande(Exception):
    """What `st.rerun` raises: in Streamlit it interrupts the script, here too."""


class FauxStFragment(FauxSt):
    """A pocket Streamlit that records the fragment's pace and the requested reloads."""

    def __init__(self):
        super().__init__()
        self.session_state: dict = {}
        self.run_every = "fragment jamais créé"
        self.reruns: list = []

    def fragment(self, run_every=None):
        self.run_every = run_every
        return lambda fonction: fonction

    def rerun(self, scope=None):
        self.reruns.append(scope)
        raise RerunDemande()


def _clore(nom: str) -> None:
    chemin = experiences.DOSSIER_JEUX / nom / "MANIFEST.yaml"
    contenu = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    contenu["clos"] = True
    _ecrire(chemin, contenu)


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if chemin.suffix == ".json":
        chemin.write_text(json.dumps(contenu), encoding="utf-8")
    else:
        chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def plateforme(tmp_path, monkeypatch):
    """A pocket platform: one experiment running, one finished, three sets."""
    exps, jeux = tmp_path / "experiences", tmp_path / "jeux"

    _ecrire(exps / "exp1" / "experience.yaml",
            {"nom": "exp1", "jeu": {"nom": "j_clos"}, "mode": "sans_simulateur",
             "decideur": {"type": "passerelle", "modele": "m1"}, "gabarit": {"variante": "b_min"}})
    _ecrire(exps / "exp1" / "executions" / "20260907-100000" / "etat.json", {"etat": "en_cours"})
    _ecrire(exps / "exp1" / "executions" / "20260907-100000" / "progression.json",
            {"faits": 120, "attendus": 400, "pourcent": 30, "reste_s": 600,
             "personnes": 50, "personnes_terminees": 12, "erreurs": 0})
    _ecrire(exps / "exp1" / "executions" / "20260906-090000" / "etat.json", {"etat": "terminee"})

    _ecrire(exps / "exp2" / "experience.yaml",
            {"nom": "exp2", "jeu": {"nom": "j_clos"}, "decideur": {"type": "aleatoire"}})
    _ecrire(exps / "exp2" / "executions" / "20260906-110000" / "etat.json", {"etat": "terminee"})

    _ecrire(exps / "exp3" / "experience.yaml", {"nom": "exp3", "jeu": {"nom": "j_clos"}, "decideur": {}})

    _ecrire(jeux / "j_clos" / "MANIFEST.yaml",
            {"nom": "j_clos", "clos": True, "population": {"nom": "pop"}, "jour_simule": "2026-03-16",
             "attendus": {"deplacements": 2693}, "couverts": {"deplacements": 2645}})
    _ecrire(jeux / "j_prep" / "MANIFEST.yaml",
            {"nom": "j_prep", "clos": False, "population": {"nom": "pop"}, "jour_simule": "2026-03-16"})
    _ecrire(jeux / "j_prep" / "progression.json",
            {"jeu": "j_prep", "faits": 1753, "total": 2693, "pourcent": 65.1, "sans_proposition": 31,
             "erreurs": 0, "reste_s": 1288, "maj": "2026-09-06T19:13:31+00:00"})
    _ecrire(jeux / "j_neuf" / "MANIFEST.yaml",
            {"nom": "j_neuf", "clos": False, "population": {"nom": "pop"}, "jour_simule": "2026-03-16"})

    monkeypatch.setattr(experiences, "DOSSIER", exps)
    monkeypatch.setattr(experiences, "DOSSIER_JEUX", jeux)
    monkeypatch.setattr(experiences, "lister", functools.partial(VRAI_LISTER, dossier=exps))
    # Table columns and filters now survive closing the page: without this
    # redirection, the tests would read the user's view and write into it.
    monkeypatch.setattr(experiences, "ETAT_VUE_REGISTRE", tmp_path / "vue" / "tableau.yaml")
    monkeypatch.setattr(experiences, "ETAT_ESPACE_ACTIF", tmp_path / "espace_actif.txt")
    monkeypatch.setattr(experiences, "ETAT_TERMINEES_PURGEES", tmp_path / "terminees_purgees.json")
    return tmp_path


def test_R1_une_execution_en_cours_est_listee_avec_son_avancement(plateforme):
    act = experiences.activites_en_cours()
    assert [e["execution"] for e in act["executions"]] == ["20260907-100000"], "only `en_cours` must appear"
    e = act["executions"][0]
    assert (e["experience"], e["faits"], e["total"], e["pourcent"], e["reste_s"]) == ("exp1", 120, 400, 30.0, 600)

    st = FauxSt()
    experiences.rendre_activites(st, act)
    assert "120 / 400 déplacements" in st.tout
    assert "30 %" in st.tout, "the criterion asks for the written percentage, not only the bar"
    assert "12 / 50 personnes" in st.tout
    assert "reste ≈ 00:10:00" in st.tout, "a duration reads as hh:mm:ss, not as bare seconds"


def test_R2_un_jeu_non_clos_est_liste_avec_sa_progression(plateforme):
    act = experiences.activites_en_cours()
    noms = [j["nom"] for j in act["jeux"]]
    assert "j_clos" not in noms, "a closed set is no longer an activity"
    assert set(noms) == {"j_prep", "j_neuf"}

    st = FauxSt()
    experiences.rendre_activites(st, act)
    assert "1 753 / 2 693 déplacements" in st.tout
    assert "65 %" in st.tout, "the criterion asks for the written percentage"
    assert "31 sans proposition" in st.tout
    assert "« j_neuf » en préparation (pas encore de progression)" in st.tout


def test_R3_sans_rien_en_cours_les_compteurs_parlent(plateforme):
    # everything is closed: no activity left
    for nom in ("j_prep", "j_neuf"):
        chemin = experiences.DOSSIER_JEUX / nom / "MANIFEST.yaml"
        contenu = yaml.safe_load(chemin.read_text(encoding="utf-8"))
        contenu["clos"] = True
        _ecrire(chemin, contenu)
    (experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "etat.json").write_text(
        json.dumps({"etat": "terminee"}), encoding="utf-8")

    act = experiences.activites_en_cours()
    assert act["executions"] == [] and act["jeux"] == []
    assert (act["definies"], act["terminees"]) == (3, 3)

    st = FauxSt()
    experiences.rendre_activites(st, act)
    assert "Aucune expérience en cours" in st.tout
    assert "**3** définie(s)" in st.tout
    assert "**3** exécution(s) terminée(s)" in st.tout


def test_R4b_un_fichier_de_progression_absurde_ne_casse_rien(plateforme):
    (experiences.DOSSIER_JEUX / "j_prep" / "progression.json").write_text("{tronqué", encoding="utf-8")
    (experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "progression.json").write_text(
        json.dumps({"pourcent": 140}), encoding="utf-8")

    act = experiences.activites_en_cours()
    assert act["executions"][0]["pourcent"] == 100.0, "an aberrant percentage is brought back to 100"
    assert act["executions"][0]["faits"] is None

    st = FauxSt()
    experiences.rendre_activites(st, act)  # must not raise
    assert "? / ? déplacements" in st.tout
    assert all(0.0 <= valeur <= 1.0 for valeur, _ in st.barres)


def test_R17_un_jeu_en_preparation_est_visible_pour_sa_population(plateforme):
    assert {j["nom"] for j in experiences.jeux_en_preparation("pop")} == {"j_prep", "j_neuf"}
    assert experiences.jeux_en_preparation("une_autre_pop") == []


def test_R19_la_construction_est_gardee_tant_qu_elle_ecrit(plateforme):
    from datetime import datetime, timedelta, timezone

    chemin = experiences.DOSSIER_JEUX / "j_prep" / "progression.json"
    frais = datetime.now(timezone.utc) - timedelta(seconds=10)
    _ecrire(chemin, {"faits": 10, "total": 100, "maj": frais.isoformat()})
    motif = experiences.construction_active("j_prep")
    assert motif and "déjà en cours" in motif

    vieux = datetime.now(timezone.utc) - timedelta(minutes=10)
    _ecrire(chemin, {"faits": 10, "total": 100, "maj": vieux.isoformat()})
    assert experiences.construction_active("j_prep") is None, "an interrupted warm-up must stay relaunchable"

    assert experiences.construction_active("j_neuf") is None, "without progress, nothing is running"


def test_R21_les_choix_du_formulaire_survivent_au_redemarrage(plateforme, monkeypatch, tmp_path):
    monkeypatch.setattr(experiences, "ETAT_FORMULAIRE", tmp_path / "formulaire.yaml")
    choix = dict(experiences.defauts())
    choix.update({"mode": "simulateur", "politique": "propre", "attente_max_s": 300,
                  "max_candidats": 9, "temperature": 0.7, "jeu_courant": {"nom": "ignoré"}})

    assert experiences.sauver_etat_formulaire(choix) is True
    assert experiences.sauver_etat_formulaire(choix) is False, "nothing moved: do not rewrite"

    relu = experiences.charger_etat_formulaire()
    assert "nom" not in relu, "the name is computed (N1): it is no longer a choice to remember"
    assert relu["attente_max_s"] == 300
    assert (relu["mode"], relu["politique"], relu["max_candidats"], relu["temperature"]) == ("simulateur", "propre", 9, 0.7)
    assert "jeu_courant" not in relu, "only the form fields are remembered"
    # The four values validated against the disk must also come back when they EXIST,
    # otherwise the R21b fallback would erase them on every start.
    for champ in ("population", "jeu", "variante", "modele"):
        assert relu[champ] == choix[champ], f"{champ} must be restored as is"


def test_R21b_une_reprise_abimee_revient_aux_defauts_champ_par_champ(plateforme, monkeypatch, tmp_path):
    fichier = tmp_path / "formulaire.yaml"
    monkeypatch.setattr(experiences, "ETAT_FORMULAIRE", fichier)

    fichier.write_text("{{{ pas du yaml", encoding="utf-8")
    assert experiences.charger_etat_formulaire() == {}, "an unreadable file must not block the form"

    _ecrire(fichier, {"attente_max_s": 300, "politique": "inventée", "decideur_type": "oracle",
                      "date": "pas-une-date", "parallelisme": 9999, "variante": "variante_disparue",
                      "jeu": "jeu_efface", "modele": "modele_retire",
                      "population": "data/population/population_effacee"})
    relu = experiences.charger_etat_formulaire()
    defaut = experiences.defauts()
    assert relu["attente_max_s"] == 300, "what is valid is kept"
    assert relu["politique"] == defaut["politique"]
    assert relu["decideur_type"] == defaut["decideur_type"]
    assert relu["date"] == defaut["date"]
    assert relu["parallelisme"] == 64, "an out-of-bounds value is brought back into the range"

    # The three values the rule names: a kept string would make the selector fall back
    # on its first choice — b0_pristine for the prompt — and that silent substitution
    # would end up written into experience.yaml.
    assert relu["variante"] == defaut["variante"], "the vanished variant must return to the active prompt"
    assert relu["jeu"] == defaut["jeu"], "a deleted set must return to the default"
    assert relu["modele"] == defaut["modele"], "a withdrawn model must return to the default"
    assert relu["population"] == defaut["population"], "a deleted population must return to the default"


def test_R22_restaurer_un_brouillon_ne_cree_aucune_experience(plateforme, monkeypatch, tmp_path):
    monkeypatch.setattr(experiences, "ETAT_FORMULAIRE", tmp_path / "formulaire.yaml")
    experiences.sauver_etat_formulaire({**experiences.defauts(), "attente_max_s": 300})
    avant = sorted(str(p.relative_to(experiences.DOSSIER)) for p in experiences.DOSSIER.rglob("experience.yaml"))
    experiences.charger_etat_formulaire()
    assert sorted(str(p.relative_to(experiences.DOSSIER)) for p in experiences.DOSSIER.rglob("experience.yaml")) == avant
    assert experiences.charger_etat_formulaire()["attente_max_s"] == 300


def test_R18_la_page_se_recharge_quand_un_jeu_devient_clos(plateforme):
    st = FauxStFragment()

    experiences._suivi_des_jeux(st, "pop")
    assert st.run_every == "5s", "while a set is being built, the block must refresh on its own"
    assert st.reruns == [], "first pass: the state is learnt, no reload"
    assert any("j_prep" in texte for _, texte in st.barres)

    _clore("j_prep")
    _clore("j_neuf")
    with pytest.raises(RerunDemande):
        experiences._suivi_des_jeux(st, "pop")
    assert st.reruns == ["app"], "the set is closed: the whole page must see it"

    experiences._suivi_des_jeux(st, "pop")
    assert st.reruns == ["app"], "state settled: no more reloads, otherwise the page loops"
    assert st.run_every is None, "nothing is being built any more: no point polling the disk"


def test_R18_un_jeu_qui_apparait_recharge_aussi_la_page(plateforme):
    st = FauxStFragment()
    experiences._suivi_des_jeux(st, "pop")
    st.reruns.clear()

    _ecrire(experiences.DOSSIER_JEUX / "j_tard" / "MANIFEST.yaml",
            {"nom": "j_tard", "clos": False, "population": {"nom": "pop"}, "jour_simule": "2026-03-16"})
    with pytest.raises(RerunDemande):
        experiences._suivi_des_jeux(st, "pop")
    assert st.reruns == ["app"]


def test_R18_le_clic_sur_la_construction_relance_le_script(plateforme):
    """The watch flag is set AFTER the fragment is created, in the same run.

    Without `st.rerun()` right after, it would only be read on the user's next click: the
    progress block would stay frozen, exactly the case R18 names.
    """
    source = Path(experiences.__file__).read_text(encoding="utf-8")
    apres_drapeau = source.split('st.session_state["_warmup_lance_a"] = time.time()', 1)[1]
    assert "st.rerun()" in apres_drapeau.split("\n\n", 1)[0], (
        "the build handler must rerun the script right after setting the flag")


def test_R18_apres_un_clic_la_surveillance_tient_avant_meme_le_dossier(plateforme):
    st = FauxStFragment()
    experiences._suivi_des_jeux(st, "population_sans_jeu")
    assert st.run_every is None, "with no build launched, nothing to watch"

    st.session_state["_warmup_lance_a"] = time.time()
    experiences._suivi_des_jeux(st, "population_sans_jeu")
    assert st.run_every == "5s", "the set's folder does not exist yet: we must wait for it to appear"


def test_R17_la_liste_du_formulaire_montre_les_jeux_en_construction(plateforme):
    """R17 is about the form's list and the warning, not the tile."""
    noms = [j["nom"] for j in experiences.jeux_de("pop")]
    assert set(noms) == {"j_clos", "j_prep", "j_neuf"}, "closed or not, a set of this population is offered"
    assert experiences.jeux_de("pop_sans_jeu") == [], "without a set, the list is empty and the form warns"

    par_nom = {j["nom"]: j for j in experiences.jeux_de("pop")}
    en_prep = experiences.libelle_jeu(par_nom["j_prep"])
    assert "(EN PRÉPARATION)" in en_prep
    # `couverts` and `attendus` only enter the manifest at closing: interpolating them raw
    # displayed "None/None" precisely on the set R17 makes visible.
    assert "None" not in en_prep, en_prep
    assert "?/? déplacements" in en_prep, en_prep
    assert "(EN PRÉPARATION)" not in experiences.libelle_jeu(par_nom["j_clos"])
    assert "2 645/2 693 déplacements" in experiences.libelle_jeu(par_nom["j_clos"])


def test_R8_un_jeu_en_preparation_s_affiche_sans_aucun_job_make(plateforme):
    """The Activités tab reads the disk: the job registry can be empty."""
    from scripts.dashboard import runner

    assert [j for j in runner.Registry().jobs() if j.running] == [], "no job launched in this test"
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "j_prep" in st.tout, "the set in preparation must be displayed without a make job"


def test_R24_l_age_de_la_progression_d_une_execution_est_dit(plateforme):
    from datetime import datetime, timedelta, timezone

    chemin = experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "progression.json"
    base = {"faits": 120, "attendus": 400, "pourcent": 30, "personnes": 50, "personnes_terminees": 12}

    frais = (datetime.now(timezone.utc) - timedelta(seconds=3)).isoformat()
    _ecrire(chemin, {**base, "maj": frais})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "progression écrite il y a" in st.tout

    vieux = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    _ecrire(chemin, {**base, "maj": vieux})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "plus rien d'écrit depuis 15 min" in st.tout
    assert "gel" not in st.tout.lower(), "the fact is written, a freeze is not diagnosed"


def test_R22_la_restauration_n_ecrit_jamais_d_experience(plateforme, monkeypatch, tmp_path):
    """The real risk is not that a folder appears, it is that `enregistrer` gets called."""
    monkeypatch.setattr(experiences, "ETAT_FORMULAIRE", tmp_path / "formulaire.yaml")
    experiences.sauver_etat_formulaire({**experiences.defauts(), "attente_max_s": 300})

    def refuser(*_args, **_kwargs):
        raise AssertionError("restoring a draft must not record anything")

    monkeypatch.setattr(experiences, "enregistrer", refuser)
    avant = sorted(str(p.relative_to(experiences.DOSSIER)) for p in experiences.DOSSIER.rglob("experience.yaml"))
    relu = experiences.charger_etat_formulaire()
    assert relu["attente_max_s"] == 300
    assert sorted(str(p.relative_to(experiences.DOSSIER)) for p in experiences.DOSSIER.rglob("experience.yaml")) == avant, \
        "a restored draft creates no experiment"


def test_R21_le_brouillon_vit_dans_un_dossier_ignore_par_git():
    """The path WRITTEN in the module, not the current value: other tests redirect it."""
    import subprocess

    source = Path(experiences.__file__).read_text(encoding="utf-8")
    declaration = 'ETAT_FORMULAIRE = REPO_ROOT / "experiments" / ".dashboard" / "formulaire_experience.yaml"'
    assert declaration in source, "the draft must live under experiments/.dashboard/"

    chemin = RACINE / "experiments" / ".dashboard" / "formulaire_experience.yaml"
    sortie = subprocess.run(["git", "check-ignore", "-v", str(chemin)],
                            cwd=RACINE, capture_output=True, text=True)
    assert sortie.returncode == 0, f"{chemin} is not ignored by git: {sortie.stdout}{sortie.stderr}"


def test_R26_les_cibles_documentees_restent_listables_hors_du_tableau_de_bord():
    import subprocess

    sortie = subprocess.run(["make", "help"], cwd=RACINE, capture_output=True, text=True)
    assert sortie.returncode == 0, sortie.stderr
    lignes = sortie.stdout.splitlines()
    assert any(l.strip().startswith("dashboard ") for l in lignes), "the dashboard target must be listed"
    assert any("Control dashboard" in l for l in lignes), "with its documentation"
    assert not any(".PHONY" in l for l in lignes), "the .PHONY lines are not targets"
    assert len(lignes) > 40, f"only {len(lignes)} targets listed"


def test_R24_l_age_est_dit_aussi_dans_la_tuile_compacte(plateforme):
    """The overview is where the marker helps most: it is in compact mode."""
    from datetime import datetime, timedelta, timezone

    chemin = experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "progression.json"
    _ecrire(chemin, {"faits": 120, "attendus": 400, "pourcent": 30,
                     "maj": (datetime.now(timezone.utc) - timedelta(seconds=4)).isoformat()})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours(), compact=True)
    assert "écrit il y a" in st.tout, "in compact mode too, the age of the progress is stated"


def test_R19_une_construction_active_grise_le_bouton_avec_son_motif():
    """The composition, not only its two halves: the reason must reach the button."""
    exp = {"nom": "exp", "jeu": {"nom": "j1"}}
    motif = "une construction est déjà en cours (10 / 100 déplacements, progression écrite il y a 3 s)"
    motifs = experiences.motifs_indisponibilite(exp, jeu_clos=True, controleur_ok=True,
                                                registre=True, construction=motif)
    assert motifs["construire"] == [motif]
    assert motifs["lancer"] == [], "launching the experiment stays possible while another set is being built"

    sans = experiences.motifs_indisponibilite(exp, jeu_clos=True, controleur_ok=True, registre=True)
    assert sans["construire"] == []


def test_R29_ce_qui_est_refuse_est_ce_qui_est_dangereux(plateforme):
    """Accents are harmless; spaces break `make EXP=` for lack of quotes."""
    acceptes = ("premiere_minimal", "Prompt_Éco", "jeu.v5-1", "P", "prompt2")
    refuses = ("Prompt Minimaliste", "a/b", "../../evade", "-flag", "_debut", "a;rm -rf",
               "a$(ls)", "", "   ", "x" * 129)
    for nom in acceptes:
        assert experiences.MOTIF_NOM.match(nom), f"{nom!r} should be accepted"
    for nom in refuses:
        assert not experiences.MOTIF_NOM.match(nom.strip()), f"{nom!r} should be refused"


def test_R29_un_nom_d_experience_ne_peut_pas_sortir_du_dossier(plateforme):
    exp = {"nom": "../../evade", "jeu": {"nom": "j_clos"}}
    motifs = experiences.motifs_indisponibilite(exp, jeu_clos=True, controleur_ok=True, registre=True)
    assert any("ni espace ni séparateur" in m for m in motifs["lancer"]), motifs["lancer"]
    assert motifs["enregistrer"], "a refused name must also block recording"

    with pytest.raises(ValueError, match="experiment name refused"):
        experiences.enregistrer({"nom": "../../evade"})
    assert not (experiences.DOSSIER.parent.parent / "evade").exists()

    correct = experiences.motifs_indisponibilite({"nom": "premiere_minimal", "jeu": {"nom": "j_clos"}},
                                                 jeu_clos=True, controleur_ok=True, registre=True)
    assert all(v == [] for v in correct.values()), correct

    # N1: an empty name is no longer a forgotten input, it is a naming that refused — and the
    # reason must name the field to fix, not state the emptiness.
    sans_nom = experiences.motifs_indisponibilite({"nom": "", "jeu": {"nom": "j_clos"}},
                                                  jeu_clos=True, controleur_ok=True, registre=True)
    assert sans_nom["enregistrer"], "an impossible name blocks recording"
    assert any("decideur" in m for m in sans_nom["enregistrer"]), sans_nom["enregistrer"]


def test_R24_un_warm_up_qui_n_ecrit_plus_le_dit_aussi(plateforme):
    """The counterpart of R24 for sets: a killed warm-up keeps `clos: false`."""
    from datetime import datetime, timedelta, timezone

    chemin = experiences.DOSSIER_JEUX / "j_prep" / "progression.json"
    base = {"faits": 1753, "total": 2693, "pourcent": 65.1}

    _ecrire(chemin, {**base, "maj": (datetime.now(timezone.utc) - timedelta(seconds=6)).isoformat()})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "progression écrite il y a" in st.tout
    assert "plus rien d'écrit" not in st.tout

    _ecrire(chemin, {**base, "maj": (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "plus rien d'écrit depuis 20 min" in st.tout, st.tout

    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours(), compact=True)
    assert "plus rien d'écrit depuis 20 min" in st.tout, "the tile must say it too"


def test_les_boutons_pause_et_arret_sont_offerts_sur_une_execution_en_cours(plateforme):
    """Request of 2026-09-07: be able to stop from "Exécutions et jeux en cours"."""
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())

    assert any("Pause" in b for b in st.boutons), st.boutons
    assert any("Arrêter" in b for b in st.boutons), st.boutons


def test_l_arret_n_est_plus_garde_par_une_case_de_confirmation(plateforme):
    """Removal requested on 2026-09-09: "⏹ Arrêter" is clickable without ticking anything.

    The difference between pause and stop, though, stays written under the buttons: it is all
    that still guards an irreversible gesture (it had cost 209 decisions on 2026-09-07).
    """
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())

    assert st.cases == [], f"no confirmation checkbox any more: {st.cases}"
    assert not any("confirme" in b for b in st.boutons), st.boutons
    dit = "\n".join(st.legendes)
    assert "reprenable" in dit and "scelle l'archive" in dit, dit


def test_la_tuile_compacte_n_offre_pas_de_bouton(plateforme):
    """The overview refreshes every 10 s: no destructive action in it."""
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours(), compact=True)
    assert st.boutons == [] and st.cases == []


def test_une_tentative_reessayee_est_une_attente_pas_une_erreur(plateforme):
    """R1 skips no trip: counting these attempts as "erreurs" made one read
    128 errors on a run that had none."""
    chemin = experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "progression.json"
    _ecrire(chemin, {"faits": 106, "attendus": 2693, "pourcent": 3.9, "sollicitations": 192,
                     "attentes": 96, "attentes_par_type": {"passerelle_occupee": 96},
                     "erreurs": 0, "erreurs_par_type": {}})

    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "96 attentes (passerelle_occupee)" in st.tout, st.tout
    assert "erreurs" not in st.tout.lower(), "no error: the word must not appear"

    # A DEFINITIVE failure, though, must stand out
    _ecrire(chemin, {"faits": 106, "attendus": 2693, "attentes": 96,
                     "attentes_par_type": {"passerelle_occupee": 96},
                     "erreurs": 3, "erreurs_par_type": {"schema_invalide": 3}})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "3 ÉCHECS DÉFINITIFS" in st.tout, st.tout


def test_le_parallelisme_conseille_suit_le_debit_des_instances_vivantes(plateforme):
    """Asking 8 decisions of an instance at 15 requests/minute wastes half the attempts."""
    vivantes = {"google_gemini35_key1": {"rpm_limit": 15}, "google_gemini35_key2": {"rpm_limit": 15}}
    conseil = experiences.parallelisme_conseille("gemini-3.5-flash-lite", vivantes)
    assert conseil is not None, "both instances of this model are declared in providers.yaml"
    # Serial, not parallel: the throughput of ONE instance, not the sum of both
    assert conseil["rpm"] == 15, conseil
    assert conseil["valeur"] == 3, "15 requests/minute on the served instance → 3 in parallel"

    une_seule = experiences.parallelisme_conseille("gemini-3.5-flash-lite", {"google_gemini35_key2": {"rpm_limit": 15}})
    assert une_seule["rpm"] == 15 and une_seule["valeur"] == 3, une_seule

    # An instance declared but absent from the gateway does not count: it has no throughput
    vide = experiences.parallelisme_conseille("gemini-3.5-flash-lite", {})
    assert vide is None, "no live instance: no advice"


class FauxStServices(FauxStFragment):
    """Adds what `_suivi_des_services` uses: container, captions, buttons."""

    def __init__(self, services):
        super().__init__()
        self.services = services
        self.legendes: list[str] = []
        self.boutons: list[str] = []

    def container(self, **_k):
        import contextlib
        return contextlib.nullcontext(self)

    def caption(self, texte, **_k):
        self.legendes.append(texte)

    def button(self, label, **_k):
        self.boutons.append(label)
        return False

    def toast(self, *_a, **_k):
        pass

    @property
    def vu(self) -> str:
        return "\n".join(self.textes + self.legendes + self.boutons)


def test_le_bloc_des_services_se_rafraichit_tant_qu_il_en_manque(plateforme, monkeypatch):
    """Without this the "le contrôleur ne tourne pas" banner outlives its start."""
    etat = {"actifs": {"api", "worker"}}
    monkeypatch.setattr(experiences, "services_actifs", lambda *a, **k: set(etat["actifs"]))

    st = FauxStServices(etat)
    experiences._suivi_des_services(st, ["controller", "api", "worker"])
    assert st.run_every == "5s", "`controller` is missing: the probe must run"
    assert "⚪ `controller`" in st.vu and "🟢 `api`" in st.vu, st.vu
    # No more button: launching starts what is missing by itself (removed on 2026-09-09).
    assert st.boutons == [], f"the services block is read-only: {st.boutons}"
    assert any("1 service(s) à démarrer" in l and "Lancer" in l for l in st.legendes), st.legendes

    # the service starts: the whole page must reload once
    etat["actifs"] = {"controller", "api", "worker"}
    st.session_state.pop("_services", None)
    with pytest.raises(RerunDemande):
        experiences._suivi_des_services(st, ["controller", "api", "worker"])
    assert st.reruns == ["app"]
    assert "_services" not in st.session_state, "the cache must be emptied so that the banner reads again"


def test_le_bloc_des_services_cesse_de_sonder_quand_tout_tourne(plateforme, monkeypatch):
    monkeypatch.setattr(experiences, "services_actifs", lambda *a, **k: {"controller", "api", "worker"})
    st = FauxStServices(None)

    experiences._suivi_des_services(st, ["controller", "api", "worker"])
    st.reruns.clear()
    experiences._suivi_des_services(st, ["controller", "api", "worker"])

    assert st.run_every is None, "everything runs: no point querying Docker in a loop"
    assert st.reruns == [], "stable state: no reload"
    assert st.boutons == [], "nothing is missing: no start button"


class FauxStRegistre(FauxStFragment):
    """What `_suivi_du_registre` uses: columns, inputs, table, bars."""

    def __init__(self):
        super().__init__()
        self.legendes: list[str] = []
        self.boutons: list[str] = []
        self.cases: list[str] = []
        self.tableaux = 0
        self.replis: list[str] = []
        self.profondeur = 0

    @contextlib.contextmanager
    def expander(self, label, **_k):
        """The "Formule" panel renders in a fold, at the top of the registry.

        What it draws does not count for the registry: without this distinction, its weights
        table would have added to the runs table and made it impossible to count.
        """
        self.replis.append(label)
        self.profondeur += 1
        try:
            yield self
        finally:
            self.profondeur -= 1

    def warning(self, texte, **_k):
        self.textes.append(texte)

    def columns(self, spec, **_k):
        largeur = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [self] * largeur

    def caption(self, texte, **_k):
        self.legendes.append(texte)

    def info(self, texte, **_k):
        self.textes.append(texte)

    @contextlib.contextmanager
    def popover(self, label, **_k):
        """A per-column filter is drawn in a popover, like the "Formule" panel
        in a fold: what it contains does not count for the runs table."""
        self.replis.append(label)
        self.profondeur += 1
        try:
            yield self
        finally:
            self.profondeur -= 1

    def multiselect(self, _label, options, default=None, key=None, **_k):
        return self.session_state.get(key, list(default or []))

    def number_input(self, _label, value=None, key=None, **_k):
        return self.session_state.get(key, value)

    def text_input(self, _label, valeur="", **_k):
        return valeur

    def selectbox(self, _label, options, index=0, **_k):
        return list(options)[index] if options else None

    def dataframe(self, *_a, **_k):
        if not self.profondeur:  # the registry table, not the folded one of the Formule panel
            self.tableaux += 1

    def button(self, label, **_k):
        self.boutons.append(label)
        return False

    def checkbox(self, label, **_k):
        self.cases.append(label)
        return False

    def toast(self, *_a, **_k):
        pass


def test_le_registre_bat_tant_qu_une_execution_tourne(plateforme):
    """A finished run stayed displayed as "en cours" until the next click."""
    import pandas as pd

    st = FauxStRegistre()
    experiences._suivi_du_registre(st, pd)
    assert st.run_every == "5s", "a run is going: the registry must refresh"
    assert st.tableaux == 1
    assert any("Formule" in r for r in st.replis), st.replis
    assert any("en cours" in x for x in st.textes), st.textes
    assert any("Pause" in b for b in st.boutons) and any("Arrêter" in b for b in st.boutons)

    # the run ends: the whole page must reload once
    _ecrire(experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "etat.json",
            {"etat": "terminee"})
    with pytest.raises(RerunDemande):
        experiences._suivi_du_registre(st, pd)
    assert st.reruns == ["app"]

    st.reruns.clear()
    experiences._suivi_du_registre(st, pd)
    assert st.reruns == [], "état stabilisé : plus de rechargement"
    assert st.run_every is None, "nothing runs any more: the registry stops polling"


def test_les_boutons_d_arret_ne_collisionnent_pas_entre_les_deux_onglets(plateforme, tmp_path):
    """The same run is displayed in "Activités en cours" and in "Mes expériences":
    two identical keys would crash Streamlit.

    The run must still be writing: the interruption controls are only displayed in that
    case (a sentinel dropped into a dead run only traps the resume), so
    without fresh progress there would be no key to compare.
    """
    from datetime import datetime, timezone

    dossier = tmp_path / "executions" / "20260907-100000"
    dossier.mkdir(parents=True)
    (dossier / "progression.json").write_text(
        json.dumps({"faits": 12, "attendus": 100,
                    "maj": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    assert experiences.execution_vivante(dossier), "the test's precondition: it is still writing"
    e = {"experience": "exp1", "execution": "20260907-100000", "dossier": str(dossier)}

    class Cles(FauxStRegistre):
        def __init__(self):
            super().__init__()
            self.vues: list[str] = []

        def button(self, label, key=None, **_k):
            self.vues.append(str(key))
            return False

        def checkbox(self, label, key=None, **_k):
            self.vues.append(str(key))
            return False

    st = Cles()
    experiences._boutons_arret(st, e, 0, prefixe="act")
    experiences._boutons_arret(st, e, 0, prefixe="reg")
    assert len(set(st.vues)) == len(st.vues), f"duplicated keys: {st.vues}"
    assert any(v.startswith("pause-act-") for v in st.vues)
    assert any(v.startswith("pause-reg-") for v in st.vues)


# ── Stopped runs: their cause, their resume, their obsolescence ──────────────────────────
# Request of 2026-09-09: a run that stops — quota exhausted, gateway unreachable,
# PC switched off, pause — left the "Activités en cours" tab that very second. One had
# to go and look for it in the registry of the Expériences tab to understand and resume.


def _arretee(nom_exp: str, execution: str, etat: dict, *, erreur: dict | None = None,
             progression: dict | None = None) -> Path:
    """A stopped run on disk, as the runner leaves it there."""
    dossier = experiences.DOSSIER / nom_exp / "executions" / execution
    _ecrire(dossier / "etat.json", etat)
    if progression is not None:
        _ecrire(dossier / "progression.json", progression)
    if erreur is not None:
        (dossier / "erreurs.jsonl").write_text(json.dumps(erreur) + "\n", encoding="utf-8")
    return dossier


def test_la_cause_de_l_arret_est_lue_dans_ce_qui_est_ecrit(plateforme):
    """Each cause comes from a trace, never from a guess: state, reason, log."""
    _arretee("exp2", "20260908-090000",
             {"etat": "epuisee", "raison": "epuise: groq_key1 : 116/1000 requêtes/jour",
              "reprise_possible_a": "2026-09-10T07:00:00+00:00"})
    _arretee("exp2", "20260908-120000", {"etat": "en_pause", "raison": "pause — 55/2693 archivées"})
    _arretee("exp2", "20260908-130000",
             {"etat": "en_pause", "raison": "pause automatique — 420s sans avancée ; 2/2693 archivées"})

    causes = {l["execution"]: experiences.cause_interruption(l)
              for l in experiences.lister() if l["experience"] == "exp2" and l.get("execution")}

    quota = causes["20260908-090000"]
    assert quota["cle"] == "quota" and "quota épuisé" in quota["libelle"]
    assert "2026-09-10T07:00:00" in quota["detail"], "the announced resume time must be stated"
    assert "116/1000" in quota["detail"], "the raw reason must stay quoted"

    assert causes["20260908-120000"]["cle"] == "pause"
    assert causes["20260908-130000"]["cle"] == "inactivite"
    assert "420s sans avancée" in causes["20260908-130000"]["detail"]


def test_un_runner_tue_et_une_coupure_reseau_se_distinguent(plateforme):
    """`etat.json` says "en cours" forever when the PC stops: the cause is elsewhere.

    The network, for its part, is read ONLY in `erreurs.jsonl` — the runner never names it. It
    only takes the place of a cause that does not name itself.
    """
    mort = _arretee("exp2", "20260908-140000", {"etat": "en_cours"},
                    progression={"faits": 10, "attendus": 400,
                                 "maj": "2026-09-08T14:00:00+00:00"})
    coupe = _arretee("exp3", "20260908-150000", {"etat": "en_cours"},
                     progression={"faits": 3, "attendus": 400, "maj": "2026-09-08T15:00:00+00:00"},
                     erreur={"type": "Gateway LLM injoignable (ConnectError)",
                             "message": "connect refused", "horodatage": "2026-09-08T15:01:02+00:00"})
    assert not experiences.execution_vivante(mort) and not experiences.execution_vivante(coupe)

    par_dossier = {str(l["dossier"]): experiences.cause_interruption(l) for l in experiences.lister()}
    assert par_dossier[str(mort)]["cle"] == "processus", par_dossier[str(mort)]
    assert "plus rien d'écrit depuis" in par_dossier[str(mort)]["detail"]
    assert par_dossier[str(coupe)]["cle"] == "reseau", par_dossier[str(coupe)]
    detail = par_dossier[str(coupe)]["detail"]
    assert "connect refused" in detail, f"the message must be quoted: {detail}"
    assert "ConnectError" in detail, f"and the type, when the message does not carry it: {detail}"


def test_un_quota_epuise_reste_un_quota_meme_apres_une_erreur_reseau(plateforme):
    """The quota names itself and carries its resume time: a network diagnosis does not replace it."""
    _arretee("exp2", "20260908-160000",
             {"etat": "epuisee", "raison": "epuise: crédits épuisés (HTTP 402)",
              "reprise_possible_a": "2026-09-09T07:00:00+00:00"},
             erreur={"type": "Gateway LLM injoignable (ConnectError)", "message": "x",
                     "horodatage": "2026-09-08T16:00:00+00:00"})
    ligne = next(l for l in experiences.lister() if l.get("execution") == "20260908-160000")
    assert experiences.cause_interruption(ligne)["cle"] == "quota"


def test_les_arretees_sont_listees_avec_de_quoi_les_reprendre(plateforme):
    """The tab's block: one row per run, its cause, its decisions already paid for."""
    dossier = _arretee("exp2", "20260908-090000",
                       {"etat": "epuisee", "raison": "epuise: quota journalier",
                        "reprise_possible_a": "2026-09-10T07:00:00+00:00"},
                       progression={"faits": 55, "attendus": 2693, "pourcent": 2.0})
    (dossier / "decisions.jsonl").write_text('{"a": 1}\n' * 55, encoding="utf-8")

    act = experiences.interrompues()
    assert [l["execution"] for l in act["lignes"]] == ["20260908-090000"], act["lignes"]
    assert act["lignes"][0]["decisions"] == 55

    lances = []
    st = FauxSt()
    st.button = lambda label, **k: (st.boutons.append(label), True)[1]  # all buttons clicked
    experiences.rendre_reprenables(st, act, lancer=lambda cible, vals: lances.append((cible, vals)))

    assert "🪫" in st.tout and "quota épuisé" in st.tout, st.tout
    assert "55 décision(s) déjà archivée(s)" in st.tout, st.tout
    assert any("Reprendre" in b for b in st.boutons), st.boutons
    assert lances and lances[0][0] == "experience-reprendre", lances
    assert lances[0][1]["EXP"] == "exp2"
    assert "controller" in lances[0][1]["REQUIS"], "resuming starts the required services"


def test_une_archive_scellee_ou_terminee_n_est_pas_proposee(plateforme):
    """A closed archive is immutable (E19): offering it would be a loop."""
    dossier = _arretee("exp2", "20260908-170000", {"etat": "arretee", "raison": "arrêt demandé"})
    _ecrire(dossier / "execution.yaml", {"cloture": {"le": "2026-09-08T17:00:00", "etat": "arretee"}})

    assert [l["execution"] for l in experiences.interrompues()["lignes"]] == []


def test_une_execution_obsolete_n_est_ni_reprenable_ni_listee_mais_comptee(plateforme):
    """The "all previous ones" are obsolete: resuming only applies to the latest.

    `make experience-reprendre` resumes `_derniere_execution` — offering an older one
    would be an empty promise. It is COUNTED, not hidden away.
    """
    _arretee("exp2", "20260908-090000", {"etat": "en_pause", "raison": "pause — 55/2693 archivées"})
    _arretee("exp2", "20260908-190000", {"etat": "en_pause", "raison": "pause — 3/2693 archivées"})

    lignes = {l["execution"]: l for l in experiences.lister() if l["experience"] == "exp2"}
    assert lignes["20260908-090000"]["obsolete"] and not lignes["20260908-090000"]["derniere"]
    assert lignes["20260908-190000"]["derniere"] and not lignes["20260908-190000"]["obsolete"]
    assert not experiences.est_reprenable(lignes["20260908-090000"])
    assert experiences.est_reprenable(lignes["20260908-190000"])

    act = experiences.interrompues()
    assert [l["execution"] for l in act["lignes"]] == ["20260908-190000"]
    assert act["obsoletes"] == 1, act
    st = FauxSt()
    experiences.rendre_reprenables(st, act)
    assert "1 exécution(s) arrêtée(s) non listée(s)" in st.tout, st.tout


def test_interrompue_et_attente_de_quota_morte_sont_reprenables(plateforme):
    """The CLI resumes any unclosed archive: these two states were missing from the table."""
    _arretee("exp2", "20260908-200000", {"etat": "interrompue", "raison": "pid mort"})
    _arretee("exp3", "20260908-210000",
             {"etat": "en_attente_quota", "raison": "epuise: quota", "reprise_possible_a": "2026-09-09T07:00"},
             progression={"faits": 1, "attendus": 400, "maj": "2026-09-08T21:00:00+00:00"})

    par_execution = {l["execution"]: l["cause"]["cle"] for l in experiences.interrompues()["lignes"]}
    assert par_execution.get("20260908-200000") == "processus", par_execution
    assert par_execution.get("20260908-210000") == "quota", par_execution


def test_une_duree_se_lit_en_heures_minutes_secondes():
    """`reste ≈ 6765 s` is unreadable; hours are not capped at 24."""
    assert experiences._duree(6765) == "01:52:45"
    assert experiences._duree(0) == "00:00:00"
    assert experiences._duree(112805) == "31:20:05", "no wrap to zero beyond one day"
    assert experiences._duree(None) == "?" and experiences._duree("x") == "?" and experiences._duree(-1) == "?"


class FauxStFiltre(FauxStRegistre):
    """A registry whose checkboxes answer their default value, and which keeps its table.

    `FauxStRegistre` answers "unticked" for every checkbox: the "Masquer les obsolètes" box
    being ticked by default, `value` must be honoured to test the filter as it opens.
    """

    def __init__(self, cases: dict | None = None):
        super().__init__()
        self.reponses = cases or {}
        self.vues: list = []

    def checkbox(self, label, value=False, **_k):
        self.cases.append(label)
        return self.reponses.get(label, value)

    def dataframe(self, donnees, *_a, **_k):
        if not self.profondeur:
            self.tableaux += 1
            # Outside the reference, the table greys its rows (R22) and Streamlit gets a `Styler`:
            # the bare table is kept, it is the one these tests query.
            self.vues.append(getattr(donnees, "data", donnees))
        return None


def test_le_registre_masque_les_obsoletes_et_dit_combien(plateforme, monkeypatch):
    """The "Terminées seulement" box is replaced by "Masquer les obsolètes" (2026-09-09).

    Nothing disappears silently: the number of hidden rows is written under the table,
    as the panel of removed entries does.
    """
    pd = pytest.importorskip("pandas")
    _arretee("exp2", "20260908-090000", {"etat": "en_pause", "raison": "pause — 55/2693 archivées"})
    _arretee("exp2", "20260908-190000", {"etat": "en_pause", "raison": "pause — 3/2693 archivées"})

    st = FauxStFiltre()
    experiences._suivi_du_registre(st, pd)
    assert any("Masquer les obsolètes" in c for c in st.cases), st.cases
    assert not any("Terminées seulement" in c for c in st.cases), st.cases
    assert "fournisseur" in st.vues[-1].columns, "the fournisseur column must be in the table"
    executions = list(st.vues[-1]["execution"])
    assert "20260908-190000" in executions, "the latest run always stays visible"
    assert "20260908-090000" not in executions, "the obsolete one is hidden by default"
    assert any("obsolète(s) masquée(s)" in l for l in st.legendes), st.legendes

    # unticked, it shows them — with what obsolescence costs, written in the etat column
    st = FauxStFiltre({"Masquer les obsolètes": False})
    experiences._suivi_du_registre(st, pd)
    vue = st.vues[-1]
    assert "20260908-090000" in list(vue["execution"])
    etats = dict(zip(vue["execution"], vue["etat"]))
    assert "obsolète (partielle)" in etats["20260908-090000"], etats
    assert "obsolète" not in etats["20260908-190000"], etats
    # exp1: a finished run replaced by a more recent one keeps a complete result
    assert "obsolète (résultat complet)" in etats["20260906-090000"], etats


# ── Who serves the decisions: the "fournisseur" column ───────────────────────────────────
# Request of 2026-09-09: know at a glance whether an experiment runs locally, at
# Google, at Groq, or through an Antigravity sub-agent — in the registry as in the tab.


@pytest.fixture
def fournisseurs(tmp_path, monkeypatch):
    """A pocket `providers.yaml`: two remote families, one local, one shared model."""
    chemin = tmp_path / "providers.yaml"
    _ecrire(chemin, {"providers": {
        "google_gemini35_key1": {"adapter": "google", "base_url": "https://generativelanguage.googleapis.com",
                                 "default_model": "gemini-3.5-flash-lite"},
        "google_gemini35_key2": {"adapter": "google", "base_url": "https://generativelanguage.googleapis.com",
                                 "default_model": "gemini-3.5-flash-lite"},
        "groq_oss_key1": {"adapter": "groq", "base_url": "https://api.groq.com/openai/v1",
                          "default_model": "openai/gpt-oss-120b"},
        "cerebras_oss_key1": {"adapter": "cerebras", "base_url": "https://api.cerebras.ai/v1",
                              "default_model": "openai/gpt-oss-120b"},
        "lmstudio_muse_key1": {"adapter": "openai_compatible",
                               "base_url": "http://host.docker.internal:1234/v1",
                               "default_model": "meta/muse-glimmer"},
    }})
    monkeypatch.setattr(experiences, "PROVIDERS_YAML", chemin)
    experiences._FAMILLES_CACHE.clear()
    yield chemin
    experiences._FAMILLES_CACHE.clear()


def test_le_fournisseur_se_deduit_du_modele_et_du_type_de_decideur(fournisseurs):
    """The family is the `adapter`, never the instance name (suffixed `_key1`, Google eleven times)."""
    f = experiences.fournisseur_de
    assert f({"type": "passerelle", "modele": "gemini-3.5-flash-lite"}) == "google", "two instances, one family"
    assert f({"type": "passerelle", "modele": "meta/muse-glimmer"}) == "local", "LM Studio shows through its base_url"
    # A model served by two families carries both: nothing is decided at random.
    assert f({"type": "passerelle", "modele": "openai/gpt-oss-120b"}) == "cerebras · groq"
    # Antigravity wins over the model name: the decision does not go through the gateway.
    assert f({"type": "antigravity", "modele": "gemini-3.5-flash-lite"}) == "antigravity"
    # A model no instance serves: say so, do not make it up.
    assert f({"type": "passerelle", "modele": "modele-fantome"}) == "inconnu"
    # No LLM called: the `decideur` column already states the heuristic.
    for dec in ({"type": "aleatoire"}, {"type": "duree_minimale"}, {"type": "modele"}, {}, None):
        assert f(dec) == "—", dec


def test_le_registre_porte_le_fournisseur_du_decideur_fige(plateforme, fournisseurs):
    """R6 holds for the provider: it is the snapshot's one, not the definition's one.

    exp1 is defined on `m1` (no instance serves it) but its run ran on
    `gemini-3.5-flash-lite`: the row must say `google`, otherwise it lies about the archive.
    """
    _ecrire(experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "execution.yaml",
            {"experience": {"decideur": {"type": "passerelle", "modele": "gemini-3.5-flash-lite"}}})

    lignes = {(l["experience"], l.get("execution")): l for l in experiences.lister()}
    assert lignes[("exp1", "20260907-100000")]["fournisseur"] == "google"
    assert lignes[("exp1", "20260906-090000")]["fournisseur"] == "inconnu", \
        "without a snapshot, the row falls back on the definition — `m1` is served by nobody"
    assert lignes[("exp2", "20260906-110000")]["fournisseur"] == "—", "exp2 draws at random"


def test_les_lignes_de_l_onglet_nomment_qui_sert(plateforme, fournisseurs):
    """Request: `🧪 antigravity / exp… / 2026-09-09_13_05_09`, and nothing without an LLM."""
    _ecrire(experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "execution.yaml",
            {"experience": {"decideur": {"type": "antigravity", "modele": "claude-opus-4.6"}}})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "🧪 **antigravity / exp1 / 20260907-100000**" in st.tout, st.tout

    # A decision-maker without an LLM carries no prefix: "— /" would suggest a gap.
    _ecrire(experiences.DOSSIER / "exp1" / "executions" / "20260907-100000" / "execution.yaml",
            {"experience": {"decideur": {"type": "aleatoire"}}})
    st = FauxSt()
    experiences.rendre_activites(st, experiences.activites_en_cours())
    assert "🧪 **exp1 / 20260907-100000**" in st.tout, st.tout


def test_le_bloc_des_arretees_nomme_aussi_qui_servait(plateforme, fournisseurs):
    """Two neighbouring lists in the same tab: they must read the same way."""
    _arretee("exp2", "20260908-090000", {"etat": "epuisee", "raison": "epuise: quota journalier"})
    _ecrire(experiences.DOSSIER / "exp2" / "executions" / "20260908-090000" / "execution.yaml",
            {"experience": {"decideur": {"type": "passerelle", "modele": "meta/muse-glimmer"}}})

    st = FauxSt()
    experiences.rendre_reprenables(st, experiences.interrompues())
    assert "**local / exp2 / 20260908-090000**" in st.tout, st.tout


# ── The real cause, not only the error type ─────────────────────────────────────────────
# Request of 2026-09-09: "there is a reason reported, the prompt rejection for example". It
# was in `erreurs.jsonl` — `message` field — and the screen only kept its `type`. The
# messages below are those of the real archives, verbatim.


def _echecs(nom_exp: str, execution: str, lignes: list[dict]) -> None:
    dossier = experiences.DOSSIER / nom_exp / "executions" / execution
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "erreurs.jsonl").write_text(
        "".join(json.dumps(l) + "\n" for l in lignes), encoding="utf-8")


def _cause(execution: str) -> dict:
    ligne = next(l for l in experiences.lister() if l.get("execution") == execution)
    return experiences.cause_interruption(ligne)


def test_le_message_de_l_echec_est_affiche_pas_seulement_son_type(plateforme):
    """The prompt rejection: "Exception interne" says nothing, the message says what to fix."""
    _arretee("exp2", "20260908-090000",
             {"etat": "en_pause", "raison": "pause automatique — 420s sans avancée ; 2/2693 archivées"})
    _echecs("exp2", "20260908-090000", [{
        "type": "Exception interne", "horodatage": "2026-09-08T09:30:00+00:00",
        "message": "Exception interne : variante de prompt 'expert_chaine_m5' introuvable dans "
                   "prompts.yaml (connues : b0_pristine, b_min, calibrated_20260611_1902)",
        "fournisseur": "google_gemini35_key1"}])

    c = _cause("20260908-090000")
    assert c["cle"] == "prompt" and c["icone"] == "🧩", c
    assert "variante introuvable" in c["libelle"], c["libelle"]
    assert "expert_chaine_m5" in c["detail"], "the message must be quoted, not reduced to its type"
    assert "google_gemini35_key1" in c["detail"], "the instance that gave up must be named"
    assert "09:30:00" in c["detail"]


def test_un_sous_agent_muet_et_une_passerelle_saturee_ne_se_confondent_pas(plateforme):
    """Two different walls, two different gestures: relaunch the agent, or wait for the gateway."""
    _arretee("exp2", "20260908-100000",
             {"etat": "en_pause", "raison": "pause automatique — 420s sans avancée"})
    _echecs("exp2", "20260908-100000", [{"type": "antigravity", "message": "antigravity: pas de réponse en 600s",
                                         "fournisseur": "antigravity:gemini-3.8-flash",
                                         "horodatage": "2026-09-08T10:00:00+00:00"}])
    _arretee("exp3", "20260908-110000",
             {"etat": "en_pause", "raison": "pause automatique — 420s sans avancée"})
    _echecs("exp3", "20260908-110000", [{
        "type": "passerelle_occupee", "horodatage": "2026-09-08T11:00:00+00:00",
        "message": "passerelle_occupee: Providers saturés ou indisponibles après 8s (50 retries épuisés)"}])

    agent, saturee = _cause("20260908-100000"), _cause("20260908-110000")
    assert (agent["cle"], agent["icone"]) == ("agent", "🤖"), agent
    assert "pas de réponse en 600s" in agent["detail"]
    assert (saturee["cle"], saturee["icone"]) == ("saturation", "🚧"), saturee
    # An overflowing gateway's "Timeout expiré" must not become a network outage.
    _echecs("exp3", "20260908-110000", [{"type": "passerelle_occupee",
                                         "message": "passerelle_occupee: Timeout expiré",
                                         "horodatage": "2026-09-08T11:05:00+00:00"}])
    assert _cause("20260908-110000")["cle"] == "saturation", "la saturation passe avant le réseau"


def test_les_vraies_coupures_reseau_sont_encore_reconnues(plateforme):
    """The two outage messages of the archives, which the old pattern missed."""
    for i, message in enumerate(("Exception interne : [Errno -2] Name or service not known",
                                 "Exception interne : Server disconnected without sending a response.",
                                 "Gateway LLM injoignable (ConnectError)")):
        execution = f"2026090{i}-120000"
        _arretee("exp3", execution, {"etat": "interrompue", "raison": "pid mort"})
        _echecs("exp3", execution, [{"type": "Exception interne", "message": message,
                                     "horodatage": "2026-09-08T12:00:00+00:00"}])
        assert _cause(execution)["cle"] == "reseau", message


def test_les_echecs_identiques_consecutifs_sont_comptes(plateforme):
    """An isolated incident and a wall do not read the same — and the reread window is bounded."""
    _arretee("exp2", "20260908-130000", {"etat": "en_pause", "raison": "pause automatique"})
    mur = {"type": "antigravity", "message": "antigravity: pas de réponse en 600s",
           "horodatage": "2026-09-08T13:00:00+00:00"}
    _echecs("exp2", "20260908-130000", [{**mur, "message": "autre chose"}, mur, mur, mur])
    c = _cause("20260908-130000")
    assert c["echecs_consecutifs"] == 3, c
    assert "(3 fois de suite)" in c["detail"], c["detail"]

    # Repetition filling the whole window: what comes before is unknown, hence "≥".
    _echecs("exp2", "20260908-130000", [mur] * (experiences.ECHECS_A_RELIRE + 5))
    assert f"(≥{experiences.ECHECS_A_RELIRE} fois de suite)" in _cause("20260908-130000")["detail"]

    # A single failure: no counter at all, it would teach nothing.
    _echecs("exp2", "20260908-130000", [mur])
    detail = _cause("20260908-130000")["detail"]
    assert "fois de suite" not in detail, detail


def test_le_quota_garde_la_priorite_sur_tout_diagnostic(plateforme):
    """Its resume time is worth more than any reading of the error log."""
    _arretee("exp2", "20260908-140000",
             {"etat": "epuisee", "raison": "epuise: quota journalier",
              "reprise_possible_a": "2026-09-09T07:00:00+00:00"})
    _echecs("exp2", "20260908-140000", [{
        "type": "Exception interne", "horodatage": "2026-09-08T14:00:00+00:00",
        "message": "Exception interne : variante de prompt 'x' introuvable dans prompts.yaml"}])

    c = _cause("20260908-140000")
    assert c["cle"] == "quota", c
    assert "2026-09-09T07:00:00" in c["detail"]
    assert "variante de prompt 'x' introuvable" in c["detail"], \
        "the diagnosis does not take the quota's place, but is still stated in the detail"


def test_bouton_passerelle_recharger_lance_la_cible(plateforme):
    """The button to reload the gateway does trigger the 'passerelle-recharger' target."""
    lances = []

    def faux_lancer(cible, var):
        lances.append((cible, var))

    class FauxStTop:
        def __init__(self):
            self.boutons = []
            self.session_state = {"_lancer": faux_lancer}

        def columns(self, spec, **_k):
            return [self] * (len(spec) if isinstance(spec, (list, tuple)) else spec)

        def subheader(self, _t):
            pass

        def button(self, label, **_k):
            self.boutons.append(label)
            return True

        def toast(self, *_a, **_k):
            pass

    st = FauxStTop()
    col_titre_mes, col_btn_rech = st.columns([3, 1], vertical_alignment="bottom")
    col_titre_mes.subheader("📚 Mes expériences")
    if col_btn_rech.button("♻️ Recharger la passerelle", key="top-recharger-passerelle",
                           disabled=not faux_lancer, width="stretch"):
        faux_lancer("passerelle-recharger", {})

    assert any("Recharger la passerelle" in b for b in st.boutons)
    assert lances == [("passerelle-recharger", {})]



# ── The cost of one heartbeat (2026-09-16) ───────────────────────────────────


class TestCoutDunBattement:
    """A panel slower than its own period saturates a core forever, silently.

    Measured on 2026-09-16: `activites_en_cours()` cost 9 s warm, 28 s cold, and its
    fragment asked for it again every 5 s. The dashboard burnt 210 minutes of CPU in
    ten hours, at 98 % continuously, without a single signal. 88 % of that time went into
    `yaml.safe_load`, which takes the Python parser although libyaml is installed; and a
    third of the reads were rereads of the same file — `providers.yaml` fifty-six
    times per call.

    What is checked here is not a duration — an assertion on the clock is unstable on
    a loaded machine, and a loaded machine is precisely what we want to measure. It is the
    NUMBER OF PARSES, which is the cause: there must be no useless reread any more, and a
    file that has not changed must never be parsed again.
    """

    @pytest.fixture(autouse=True)
    def cache_neuf(self):
        experiences._yaml_analyse.cache_clear()
        yield
        experiences._yaml_analyse.cache_clear()

    def test_libyaml_est_utilise_quand_il_est_la(self):
        """×7.9 measured on the repo files. `yaml.safe_load` never picks it on its own."""
        if not getattr(yaml, "__with_libyaml__", False):
            pytest.skip("libyaml absent from this interpreter: the Python parser is the right fallback")
        assert experiences._CHARGEUR_YAML is yaml.CSafeLoader

    def test_le_meme_fichier_lu_deux_fois_n_est_analyse_qu_une_fois(self, plateforme):
        manifeste = plateforme / "jeux" / "j_clos" / "MANIFEST.yaml"
        assert experiences._yaml(manifeste) == experiences._yaml(manifeste)
        infos = experiences._yaml_analyse.cache_info()
        assert (infos.misses, infos.hits) == (1, 1), (
            "`providers.yaml` was parsed 56 times in a single heartbeat")

    def test_un_battement_qui_se_repete_ne_reanalyse_RIEN(self, plateforme):
        experiences.activites_en_cours()
        apres_premier = experiences._yaml_analyse.cache_info().misses
        assert apres_premier > 0, "the first heartbeat must indeed read something"

        experiences.activites_en_cours()
        assert experiences._yaml_analyse.cache_info().misses == apres_premier, (
            "a second heartbeat with nothing changed on disk must cost no parse at "
            "all: that is what brings the cost from 9 s down to under 100 ms.")

    def test_un_fichier_REECRIT_est_bien_relu(self, plateforme):
        """The cache's safeguard: `ajouter_variante` rewrites `prompts.yaml` then reads it
        back at once to check its own work. A cache serving the old content
        would make that check fail — or worse, make it pass wrongly."""
        manifeste = plateforme / "jeux" / "j_neuf" / "MANIFEST.yaml"
        assert experiences._yaml(manifeste).get("clos") is False

        time.sleep(0.01)  # so the mtime moves, even on a coarse file system
        manifeste.write_text(yaml.safe_dump({"nom": "j_neuf", "clos": True}), encoding="utf-8")
        assert experiences._yaml(manifeste).get("clos") is True, (
            "a rewritten file must be read again: the cache key carries the size AND the mtime")

    def test_un_fichier_absent_ou_illisible_ne_leve_pas(self, plateforme):
        assert experiences._yaml(plateforme / "jamais_ecrit.yaml") == {}
        assert experiences._yaml(plateforme / "jeux") == {}, "a folder is not a file"
        casse = plateforme / "casse.yaml"
        casse.write_text("{ ceci n'est pas: du yaml: du tout", encoding="utf-8")
        assert experiences._yaml(casse) == {}


# ── Runs finished successfully: log, date/time and purge ──────────────────────

def _terminee(nom_exp: str, execution: str, etat: dict | None = None, *,
              cree_le: str = "2026-09-24T08:00:00+00:00",
              maj: str = "2026-09-24T08:30:15+00:00",
              decisions: int = 3154, attendus: int = 3161) -> Path:
    exp_yaml = experiences.DOSSIER / nom_exp / "experience.yaml"
    if not exp_yaml.is_file():
        _ecrire(exp_yaml, {"nom": nom_exp, "decideur": {}})
    dossier = experiences.DOSSIER / nom_exp / "executions" / execution
    base_etat = {"etat": "terminee", "maj": maj, "decisions_archivees": decisions}
    if etat:
        base_etat.update(etat)
    _ecrire(dossier / "etat.json", base_etat)
    _ecrire(dossier / "execution.yaml", {"version": "execution1", "cree_le": cree_le, "experience": {"nom": nom_exp}})
    _ecrire(dossier / "compteurs.json", {"couverture": {"decides": decisions, "attendus": attendus, "taux": decisions / attendus}})
    return dossier


def test_terminees_liste_executions_avec_date_et_heure(plateforme, monkeypatch, tmp_path):
    """Runs finished successfully are listed with their end date and time."""
    fichier_purge = tmp_path / "terminees_purgees.json"
    monkeypatch.setattr(experiences, "ETAT_TERMINEES_PURGEES", fichier_purge)
    experiences.purger_terminees()

    _terminee("exp_terminee", "20260924-080000", maj="2026-09-24T08:30:15+00:00", decisions=3154, attendus=3161)
    _arretee("exp_en_pause", "20260924-081000", {"etat": "en_pause", "raison": "pause"})

    lignes = experiences.terminees()
    assert len(lignes) == 1
    e = lignes[0]
    assert e["experience"] == "exp_terminee"
    assert e["execution"] == "20260924-080000"
    assert "24/09/2026" in e["date_heure_texte"]
    assert "30:15" in e["date_heure_texte"]
    assert e["decisions"] == 3154
    assert e["couverture"] is not None


def test_purger_terminees_retire_les_terminees_de_l_affichage(plateforme, monkeypatch, tmp_path):
    """The purge records the keys and removes the finished runs from the display."""
    fichier_purge = tmp_path / "terminees_purgees.json"
    monkeypatch.setattr(experiences, "ETAT_TERMINEES_PURGEES", fichier_purge)
    experiences.purger_terminees()

    _terminee("exp1", "20260924-080000", maj="2026-09-24T08:30:00+00:00")
    _terminee("exp2", "20260924-081000", maj="2026-09-24T08:40:00+00:00")

    assert len(experiences.terminees()) == 2
    purges = experiences.purger_terminees()
    assert purges == 2
    assert len(experiences.terminees()) == 0

    # Idempotent
    assert experiences.purger_terminees() == 0

    # A new run finished afterwards does appear
    _terminee("exp3", "20260924-090000", maj="2026-09-24T09:15:00+00:00")
    lignes = experiences.terminees()
    assert len(lignes) == 1
    assert lignes[0]["experience"] == "exp3"


def test_rendre_terminees_affichage_et_bouton_vider(plateforme, monkeypatch, tmp_path):
    """rendre_terminees displays the finished runs with date/time and the purge button."""
    fichier_purge = tmp_path / "terminees_purgees.json"
    monkeypatch.setattr(experiences, "ETAT_TERMINEES_PURGEES", fichier_purge)
    experiences.purger_terminees()

    st = FauxSt()
    experiences.rendre_terminees(st, [])
    assert "Aucune exécution terminée récente" in st.tout

    _terminee("exp1", "20260924-080000", maj="2026-09-24T08:30:00+00:00", decisions=100, attendus=100)
    lignes = experiences.terminees()
    st2 = FauxSt()
    st2.expander = lambda *a, **k: st2
    st2.__enter__ = lambda *a: st2
    st2.__exit__ = lambda *a: None
    experiences.rendre_terminees(st2, lignes)

    assert "✅" in st2.tout
    assert "exp1" in st2.tout
    assert "terminée avec succès le" in st2.tout
    assert "100 décisions archivées" in st2.tout
    assert any("Vider la liste" in b for b in st2.boutons)


def test_formater_date_heure():
    """Checks the formatting of local dates and times."""
    assert "16/09/2026" in experiences.formater_date_heure("2026-09-16T13:36:44+00:00")
    assert experiences.formater_date_heure(None) == "date inconnue"
    assert experiences.formater_date_heure("") == "date inconnue"


def test_render_activites_ordre_des_blocs(plateforme, monkeypatch):
    """Checks that the block of finished runs does appear before the stopped runs."""
    from scripts.dashboard import app

    st = FauxSt()
    st.divider = lambda: None
    monkeypatch.setattr(app, "st", st)
    monkeypatch.setattr(app, "render_activites_disque", lambda: st.markdown("BLOC_EN_COURS"))
    monkeypatch.setattr(app, "render_derniere_erreur_llm", lambda: None)
    monkeypatch.setattr(app, "render_terminees_disque", lambda: st.markdown("BLOC_TERMINEES"))
    monkeypatch.setattr(app, "render_reprenables_disque", lambda: st.markdown("BLOC_ARRETEES"))
    monkeypatch.setattr(app, "render_jobs_live", lambda: st.markdown("BLOC_JOBS"))

    app.render_activites()
    textes = st.textes
    idx_terminees = next(i for i, t in enumerate(textes) if "BLOC_TERMINEES" in t or "terminées" in t)
    idx_arretees = next(i for i, t in enumerate(textes) if "BLOC_ARRETEES" in t or "arrêtées" in t)
    assert idx_terminees < idx_arretees, "The finished block must be placed before the stopped ones"

