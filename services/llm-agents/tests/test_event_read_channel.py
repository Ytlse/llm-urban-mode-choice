"""The `lu` (read) channel, the `reveil` hook, the citation, households.

The `lu` channel is the second regime: the agent knows BEFORE deciding. All that distinguishes it
from the shock lies in that word — and in the fact that nothing, in the world, has moved.

Everything here is PURE: no simulator, no model, no network call.
"""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import evenements as ev
from llm.evenements import RefusDEvenement, RegistreEvenements, charger
from settings import settings

RACINE = Path(__file__).resolve().parents[1]
CONFIG_EVENEMENTS = RACINE / "config" / "evenements"
CORPUS = RACINE.parents[1] / "data" / "presse" / "articles_txt"

ARTICLES = (
    "a07_greve_eboueurs", "a09_vent_autan", "a13_punaises_metro",
    "a18_la_machine", "a25_velotoulouse",
)


@dataclass
class FauxAgent:
    """The minimum that the `foyers` rule reads on an agent: its identity and its household."""

    person_id: str
    household_id: str | None = None
    immobile: bool = False


def _population(*couples) -> list:
    return [FauxAgent(person_id=p, household_id=h) for p, h in couples]


def _texte(tmp_path: Path, contenu: str, nom: str = "brut.txt") -> tuple[Path, str]:
    dossier = tmp_path / "articles_txt" / "a99_test"
    dossier.mkdir(parents=True, exist_ok=True)
    fichier = dossier / nom
    fichier.write_text(contenu, encoding="utf-8")
    return fichier, hashlib.sha256(fichier.read_bytes()).hexdigest()


ARTICLE = (
    "Toulouse: the metro will run a reduced service on Line A next week while signalling "
    "work is carried out at Jean-Jaures. The operator says trains will be spaced further "
    "apart between 10am and 4pm."
)


def _declaration(fichier: Path, sha: str, **surcharges) -> dict:
    base = {
        "evenement": "a99_test",
        "libelle": "Article de test",
        "source": "test",
        "canal": "lu",
        "moment": "reveil",
        "jugement": "aucun",
        "texte": {"fichier": str(fichier), "sha256": sha},
        "calendrier": {"fenetre_jours": [9, 13], "graine": 59},
        "exposition": {
            "regle": "foyers", "foyers": ["5177", "18056"],
            "lecteurs_par_foyer": 1, "graine": 59,
        },
        "cadence": "jour",
    }
    base.update(surcharges)
    return base


def _ecrire(tmp_path: Path, declaration: dict) -> Path:
    p = tmp_path / "evenement.yaml"
    p.write_text(yaml.safe_dump(declaration, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _registre_propre():
    ev.reinitialiser()
    settings.evenements.enabled = False
    settings.evenements.fichier = None
    settings.chocs.enabled = False
    yield
    ev.reinitialiser()
    settings.evenements.enabled = False
    settings.evenements.fichier = None
    settings.chocs.enabled = False


# ── R13, R14. The world does not move ────────────────────────────────────────────────────
def test_R13_un_article_ne_fait_subir_aucun_retard(tmp_path):
    f, sha = _texte(tmp_path, ARTICLE)
    e = charger(_ecrire(tmp_path, _declaration(f, sha)))
    assert e.jours == {}  # no day-by-day profile: an article is the same every morning
    assert e.texte_cite is not None


def test_R14_un_effet_physique_au_reveil_est_refuse(tmp_path):
    """The world does not change before the decision.

    Without this rule, the day of the event would measure both an anticipation and a
    constraint, and nothing could separate them.
    """
    d = {
        "evenement": "e_test", "canal": "vecu", "moment": "reveil", "jugement": "aucun",
        "exposition": {"regle": "agents", "agents": ["609"]},
        "jours": [{"jour": 3, "retard_min": 30, "texte": "I could not start my car."}],
    }
    with pytest.raises(RefusDEvenement, match="WAKE-UP"):
        charger(_ecrire(tmp_path, d))


def test_R14bis_un_canal_lu_pose_a_larrivee_est_refuse(tmp_path):
    f, sha = _texte(tmp_path, ARTICLE)
    with pytest.raises(RefusDEvenement, match="BEFORE deciding"):
        charger(_ecrire(tmp_path, _declaration(f, sha, moment="arrivee")))


# ── R15, R16, R17. The citation guard ────────────────────────────────────────────────────
def test_R15_un_texte_modifie_depuis_la_declaration_est_refuse(tmp_path):
    f, sha = _texte(tmp_path, ARTICLE)
    declaration = _ecrire(tmp_path, _declaration(f, sha))
    f.write_text(ARTICLE + " One more sentence.", encoding="utf-8")
    with pytest.raises(RefusDEvenement, match="has CHANGED"):
        charger(declaration)


def test_R15bis_une_empreinte_absente_est_refusee(tmp_path):
    f, sha = _texte(tmp_path, ARTICLE)
    with pytest.raises(RefusDEvenement, match="sha256` missing"):
        charger(_ecrire(tmp_path, _declaration(f, "")))


def test_R15ter_la_seconde_comparaison_attrape_ce_que_la_premiere_laisse_passer(tmp_path):
    """Declaration and file agree, manifest disagrees: refused.

    This is the case that comparing to the file alone would let through — a text edited at
    the same time as its declaration. Two campaigns would believe they played the same article.
    """
    f, sha = _texte(tmp_path, ARTICLE)
    manifeste = f.parent.parent / "MANIFEST.yaml"
    manifeste.write_text(
        yaml.safe_dump({"articles": {"a99_test": {"brut": {"en": {"sha256": "0" * 64}}}}}),
        encoding="utf-8",
    )
    with pytest.raises(RefusDEvenement, match="manifest"):
        charger(_ecrire(tmp_path, _declaration(f, sha)))


def test_R15quater_un_texte_introuvable_dit_ou_il_a_cherche(tmp_path):
    with pytest.raises(RefusDEvenement, match="Looked in"):
        charger(_ecrire(tmp_path, _declaration(Path("nulle/part/brut.txt"), "0" * 64)))


def test_R16_les_cinq_articles_du_059_concordent_avec_leur_manifeste():
    """Loaded from the repository, fingerprints checked against `MANIFEST.yaml`."""
    for nom in ARTICLES:
        chemin = CONFIG_EVENEMENTS / f"{nom}.yaml"
        assert chemin.is_file(), f"declared article missing: {nom}"
        e = charger(chemin)  # the citation guard runs here
        assert e.canal == "lu" and e.moment == "reveil"
        assert e.texte_cite is not None and e.texte_cite.contenu
        assert e.exposition.regle == "foyers"
        assert e.calendrier.fenetre == (9, 13)


def test_R17_la_mention_de_traduction_est_dans_lentree_servie():
    """A translation served as an original would be an undeclared condition."""
    e = charger(CONFIG_EVENEMENTS / "a13_punaises_metro.yaml")
    assert e.texte_cite.mention == "Translated from French"
    assert "Translated from French" in e.texte_cite.servi
    assert e.texte_cite.servi.endswith(e.texte_cite.contenu)


# ── The content guards apply to the cited text ───────────────────────────────────────────
def test_un_article_qui_dicte_un_mode_est_refuse(tmp_path):
    f, sha = _texte(tmp_path, "Residents should avoid the metro this week, the city says.")
    with pytest.raises(RefusDEvenement, match="avoid"):
        charger(_ecrire(tmp_path, _declaration(f, sha)))


def test_une_exemption_ne_vaut_que_declaree_motivee_et_presente(tmp_path):
    texte = "City staff must first inspect each site to make sure there is no danger."
    f, sha = _texte(tmp_path, texte)
    # Without exemption: refused, and the message says how to exempt.
    with pytest.raises(RefusDEvenement, match="marqueurs_exemptes"):
        charger(_ecrire(tmp_path, _declaration(f, sha)))
    # Exemption without reason: refused.
    d = _declaration(f, sha)
    d["texte"]["marqueurs_exemptes"] = [{"marqueur": "make sure"}]
    with pytest.raises(RefusDEvenement, match="without a reason"):
        charger(_ecrire(tmp_path, d))
    # Exemption of an ABSENT marker: refused — one does not exempt as a precaution.
    d["texte"]["marqueurs_exemptes"] = [{"marqueur": "you should", "motif": "au cas où"}]
    with pytest.raises(RefusDEvenement, match="DOES NOT APPEAR"):
        charger(_ecrire(tmp_path, d))
    # Exemption in order: the text passes.
    d["texte"]["marqueurs_exemptes"] = [
        {"marqueur": "make sure", "motif": "le sujet est le personnel municipal"}
    ]
    assert charger(_ecrire(tmp_path, d)).texte_cite is not None


def test_ladresse_a_la_deuxieme_personne_ne_sexempte_jamais(tmp_path):
    f, sha = _texte(tmp_path, "The city reminds you that the metro closes at midnight.")
    d = _declaration(f, sha)
    d["texte"]["marqueurs_exemptes"] = [{"marqueur": "you", "motif": "essayons"}]
    with pytest.raises(RefusDEvenement, match="second person"):
        charger(_ecrire(tmp_path, d))


# ── R18, R19. The `foyers` rule ──────────────────────────────────────────────────────────
def _registre(tmp_path, **surcharges) -> RegistreEvenements:
    f, sha = _texte(tmp_path, ARTICLE)
    return RegistreEvenements(charger(_ecrire(tmp_path, _declaration(f, sha, **surcharges))))


POPULATION = _population(
    ("12", "5177"), ("49", "5177"),          # Constance and Jacques
    ("66", "18056"), ("65", "18056"),        # Valérie and Maurice
    ("40", "312"),                           # Xavier, alone — outside declared households
)


def test_R18_un_lecteur_par_foyer_et_le_co_resident_est_temoin(tmp_path):
    lecteurs = _registre(tmp_path).lecteurs(POPULATION)
    assert len(lecteurs) == 2
    assert {h for _, (h, _) in lecteurs.items()} == {"5177", "18056"}
    assert "40" not in lecteurs  # undeclared household: nothing
    for foyer in ("5177", "18056"):
        membres = {p for p, (h, _) in lecteurs.items() if h == foyer}
        assert len(membres) == 1, "the co-resident not drawn is the internal control"


def test_R18bis_le_tirage_des_lecteurs_est_stable_sur_trois_executions(tmp_path):
    tirages = [
        set(RegistreEvenements(charger(
            _ecrire(tmp_path, _declaration(*_texte(tmp_path, ARTICLE)))
        )).lecteurs(POPULATION))
        for _ in range(3)
    ]
    assert tirages[0] == tirages[1] == tirages[2]
    # And the order of the population changes nothing.
    autre = RegistreEvenements(charger(
        _ecrire(tmp_path, _declaration(*_texte(tmp_path, ARTICLE)))
    )).lecteurs(list(reversed(POPULATION)))
    assert set(autre) == tirages[0]


def test_R18ter_deux_lecteurs_par_foyer_en_donnent_deux(tmp_path):
    r = _registre(tmp_path, exposition={
        "regle": "foyers", "foyers": ["5177"], "lecteurs_par_foyer": 2, "graine": 59,
    })
    assert set(r.lecteurs(POPULATION)) == {"12", "49"}


def test_R19_un_foyer_sans_membre_mobile_leve_une_alarme_et_ne_lit_pas(tmp_path):
    r = _registre(tmp_path, exposition={
        "regle": "foyers", "foyers": ["99999"], "lecteurs_par_foyer": 1, "graine": 59,
    })
    assert r.lecteurs(POPULATION) == {}


def test_R19bis_une_regle_foyers_sans_foyer_est_refusee(tmp_path):
    f, sha = _texte(tmp_path, ARTICLE)
    d = _declaration(f, sha, exposition={"regle": "foyers", "foyers": []})
    with pytest.raises(RefusDEvenement, match="without any household identifier"):
        charger(_ecrire(tmp_path, d))


def test_R19ter_les_immobiles_ne_lisent_pas(tmp_path):
    population = [FauxAgent("12", "5177", immobile=True), FauxAgent("49", "5177")]
    assert set(_registre(tmp_path).lecteurs(population)) == {"49"}


# ── R20, R21. The drawn window ───────────────────────────────────────────────────────────
def test_R20_le_jour_est_dans_la_fenetre_stable_et_differe_dun_foyer_a_lautre(tmp_path):
    r = _registre(tmp_path)
    cibles = [str(1000 + i) for i in range(20)]
    jours = {c: r.evenement.calendrier.jour_de("a99_test", c) for c in cibles}
    assert all(9 <= j <= 13 for j in jours.values()), jours
    # Stable: replayed, the same household draws the same day.
    assert jours == {c: r.evenement.calendrier.jour_de("a99_test", c) for c in cibles}
    # And not all read on the same morning — that is what separates the article from the calendar.
    assert len(set(jours.values())) > 1


def test_R20bis_les_membres_dun_meme_foyer_lisent_le_meme_jour(tmp_path):
    """Otherwise the control co-resident would no longer be comparable to the reader on the same day."""
    r = _registre(tmp_path, exposition={
        "regle": "foyers", "foyers": ["5177"], "lecteurs_par_foyer": 2, "graine": 59,
    })
    assert r.jour_de("12", "5177") == r.jour_de("49", "5177")


@pytest.mark.parametrize(
    "fenetre, motif",
    [([13, 9], "is empty"), ([0, 5], "number 1"), ([9], "two bounds")],
)
def test_R21_une_fenetre_impossible_est_refusee(tmp_path, fenetre, motif):
    f, sha = _texte(tmp_path, ARTICLE)
    d = _declaration(f, sha, calendrier={"fenetre_jours": fenetre, "graine": 59})
    with pytest.raises(RefusDEvenement) as err:
        charger(_ecrire(tmp_path, d))
    assert motif in str(err.value)


# ── R12, R22. The `reveil` hook ──────────────────────────────────────────────────────────
def _le_jour(registre, jour: int):
    RegistreEvenements.jour_du_run = staticmethod(lambda ts, _j=jour: _j)


def _rendre_le_calendrier():
    from llm.evenements import calendrier as _cal

    RegistreEvenements.jour_du_run = staticmethod(_cal.jour_du_run)


def test_R12_linjection_na_lieu_quune_fois_par_agent(tmp_path):
    r = _registre(tmp_path)
    try:
        lecteurs = r.lecteurs(POPULATION)
        servis = {}
        for jour in range(9, 14):
            _le_jour(r, jour)
            for person_id, applique in r.dus_au_reveil(1_700_000_000, POPULATION):
                servis.setdefault(person_id, []).append(jour)
            # Called three times on the same day: nothing more.
            assert r.dus_au_reveil(1_700_000_000, POPULATION) == []
            assert r.dus_au_reveil(1_700_000_000, POPULATION) == []
        assert set(servis) == set(lecteurs)
        assert all(len(j) == 1 for j in servis.values()), servis
    finally:
        _rendre_le_calendrier()


def test_R22_un_co_resident_non_tire_ne_recoit_rien(tmp_path):
    r = _registre(tmp_path)
    try:
        lecteurs = set(r.lecteurs(POPULATION))
        recus = set()
        for jour in range(9, 14):
            _le_jour(r, jour)
            recus |= {p for p, _ in r.dus_au_reveil(1_700_000_000, POPULATION)}
        assert recus == lecteurs
        temoins = {a.person_id for a in POPULATION} - lecteurs
        assert temoins and not (temoins & recus)
    finally:
        _rendre_le_calendrier()


def test_R12bis_lentree_servie_porte_le_prefixe_et_le_texte(tmp_path):
    r = _registre(tmp_path)
    try:
        for jour in range(9, 14):
            _le_jour(r, jour)
            for _, applique in r.dus_au_reveil(1_700_000_000, POPULATION):
                entree = ev.entree_de_lecture(applique)
                assert entree.startswith("[ PRESSE ] I read in the paper:")
                assert ARTICLE in entree
                assert applique.retard_injecte_s == 0
                assert applique.incident_reseau is False
                return
        pytest.fail("no reader served: the test proves nothing")
    finally:
        _rendre_le_calendrier()


def test_une_prise_arrivee_ne_rend_rien_au_reveil(tmp_path):
    """The two hooks stay in their place: that is the whole contrast of chapter 7."""
    from llm.evenements.declaration import Calendrier

    chemin = RACINE / "config" / "evenements" / "c6_voiture_suspecte.yaml"
    r = RegistreEvenements(charger(chemin))
    try:
        _le_jour(r, 15)
        assert r.dus_au_reveil(1_700_000_000, POPULATION) == []
    finally:
        _rendre_le_calendrier()


# ── The cache, switched off on the day of the event (Q5) ─────────────────────────────────
def test_le_cache_est_coupe_le_jour_de_levenement_et_pas_les_autres(tmp_path):
    chemin = RACINE / "config" / "evenements" / "c6_voiture_suspecte.yaml"
    r = RegistreEvenements(charger(chemin))  # days 15 and 16
    try:
        for jour, attendu in ((14, False), (15, True), (16, True), (17, False)):
            _le_jour(r, jour)
            assert r.cache_coupe(1_700_000_000) is attendu, f"day {jour}"
    finally:
        _rendre_le_calendrier()


def test_la_coupure_couvre_toute_la_fenetre_quand_les_jours_sont_tires(tmp_path):
    """Without the population, one does not know WHO reads on which morning: cut wide.

    Cutting wide costs a few days of calls; cutting off target would let a decision be served
    again on the very morning of publication.
    """
    r = _registre(tmp_path)  # window [9, 13]
    try:
        for jour, attendu in ((8, False), (9, True), (11, True), (13, True), (14, False)):
            _le_jour(r, jour)
            assert r.cache_coupe(1_700_000_000) is attendu, f"day {jour}"
    finally:
        _rendre_le_calendrier()


def test_sans_evenement_le_cache_nest_jamais_coupe():
    assert ev.registre() is None
    assert ev.cache_coupe(1_700_000_000) is False


# ── The fix of the 2026-09-22 defect ─────────────────────────────────────────────────────
def test_la_prise_reveil_ecrit_AUSSI_en_memoire_longue():
    """Without this write, the `reveil` regime does not do what it announces.

    Measured in the code: the decision reads ONLY long-term memory
    (`query_past_experiences_for_travel` → `aquery_user_memories`), and the evening consolidation
    CONSUMES the short-term buffer (`remove_batch`) while writing only the reflection and concepts.
    An entry placed at 3:00 in short-term memory would thus reach no decision of the day, and
    "the agent knows before deciding" would be false.

    Checked by reading the source, like the boundary tests R14 and R15 of declared shocks: setting
    up a full controller would require GAMA, a model and a vector index.
    """
    ctrl = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text("utf-8")
    debut = ctrl.index("def _injecter_evenements_du_reveil")
    fin = ctrl.index("async def _tirer_accidents_du_jour")
    methode = ctrl[debut:fin]
    assert "add_short_term_memory" in methode, "the evening reflection must see what was read"
    assert "aadd_long_term_memory" in methode, (
        "the SAME-DAY decision reads only long-term memory: without this write, "
        "the article reaches no decision before the next day"
    )
    # And the decision, for its part, still reads only long-term memory: if that changed, the
    # double write above would become a duplication instead of a necessity.
    agent = (RACINE / "urban_mobility_agents" / "agents" / "llm_agent.py").read_text("utf-8")
    assert "hist = await self.long_term_memory.aquery_user_memories(" in agent


def test_le_choc_lui_necrit_pas_en_memoire_longue():
    """The asymmetry is the regime, not an oversight.

    A shock applies after the decision: that its effect only starts the next day is
    exactly what is wanted, and it is what makes the following days attributable to the
    memory alone. Opening the same path to it would erase the difference between the two regimes.
    """
    ctrl = (RACINE / "urban_mobility_agents" / "simulation_controller.py").read_text("utf-8")
    debut = ctrl.index("# ── Ticket 100 — the declared event")
    fin = ctrl.index("await GamaArrivalsLogger")
    prise_arrivee = ctrl[debut:fin]
    assert "joindre" in prise_arrivee
    assert "aadd_long_term_memory" not in prise_arrivee


def test_un_article_peut_nommer_des_modes_de_transport(tmp_path):
    """Author's decision, 2026-09-22. Not to be confused with the press lexicon list.

    That one forbids mobility vocabulary in the PARAPHRASE (C3) and the CONTROL (C4),
    so that these conditions do not talk about transport at all. It does not apply to the
    raw cited text, which is the article. The boundary of this module is not the vocabulary,
    it is the destination of the sentence.
    """
    texte = (
        "Toulouse: metro Line A, the bus network and the VélôToulouse bike scheme will all be "
        "affected by Thursday's strike. Drivers are expected to face heavy congestion on the "
        "ring road, and the tram will run every twenty minutes instead of six."
    )
    f, sha = _texte(tmp_path, texte)
    e = charger(_ecrire(tmp_path, _declaration(f, sha)))
    for mode in ("metro", "bus", "bike", "ring road", "tram"):
        assert mode in e.texte_cite.contenu


def test_les_foyers_declares_existent_dans_la_population_du_059():
    """Found on 2026-09-22: the five articles named households of ANOTHER population.

    The identifiers came from a household study list, which covers the whole v6
    cohort. The campaign population is `population_20_foyers_059`, ten entirely different
    households: a campaign would have raised the [ALARME] "aucun lecteur retenu" and would
    have run without a single agent reading anything at all.

    The guard did its job — it would have shouted — but shouting at the launch of a forty-day
    campaign costs more than this test.
    """
    import yaml as _yaml

    manifeste = (
        RACINE.parents[1] / "data" / "population" / "population_20_foyers_059" / "MANIFEST.yaml"
    )
    if not manifeste.is_file():
        pytest.skip("campaign population absent from this machine")
    m = _yaml.safe_load(manifeste.read_text("utf-8"))
    attendus = {f["household_id"] for f in m["groupes"]["expose"]}
    temoins = {f["household_id"] for f in m["groupes"]["temoin"]}

    for nom in ARTICLES:
        e = charger(CONFIG_EVENEMENTS / f"{nom}.yaml")
        assert set(e.exposition.foyers) == attendus, (
            f"{nom} does not expose the households the manifest declares exposed"
        )
        assert not (set(e.exposition.foyers) & temoins), (
            f"{nom} would expose a CONTROL household — the control is in the same run, and "
            f"exposing it would erase the baseline"
        )


# ── "0 exposé" alarm of a window drawn per household (run 2026-09-24_17_50) ─────────────
@pytest.fixture
def journal():
    """Loguru messages, level and text."""
    from loguru import logger

    lignes: list[tuple[str, str]] = []
    sink = logger.add(
        lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO"
    )
    yield lignes
    logger.remove(sink)


def _erreurs(journal) -> list[str]:
    """End-of-day alarms only: the test declaration, without judgement, raises its
    own at load time, and it has nothing to do with what is checked here."""
    return [m for n, m in journal if n == "ERROR" and " du run" in m]


def test_un_jour_de_fenetre_sans_lecteur_tire_ne_leve_pas_dalarme(tmp_path, journal):
    """Window 9-13, one day drawn per household: the other days have nothing to expose.

    The a09 run raised "JOUR D'ÉVÉNEMENT clos avec 0 exposé" on days 9, 10, 12 and 13, while
    its only household had drawn day 11.
    """
    r = _registre(tmp_path)
    try:
        lecteurs = r.lecteurs(POPULATION)
        tires = {r.jour_de(p, h) for p, (h, _) in lecteurs.items()}
        for jour in range(9, 14):
            _le_jour(r, jour)
            r.dus_au_reveil(1_700_000_000, POPULATION)
        r.journaliser_compteurs()
    finally:
        _rendre_le_calendrier()
    assert not _erreurs(journal), _erreurs(journal)
    parutions = [m for n, m in journal if n == "INFO" and "JOUR DE PARUTION" in m]
    assert len(parutions) == len(tires), "a drawn publication day must say so"
    assert set(range(9, 14)) - tires, "the test assumes at least one window day without a draw"
    sans_tirage = [m for n, m in journal if "aucun lecteur tiré pour ce jour" in m]
    assert len(sans_tirage) == 5 - len(tires)


def test_un_lecteur_tire_qui_na_pas_lu_leve_une_alarme_qui_le_nomme(tmp_path, journal):
    r = _registre(tmp_path)
    try:
        lecteurs = r.lecteurs(POPULATION)
        lecteur, (foyer, _) = sorted(lecteurs.items())[0]
        jour = r.jour_de(lecteur, foyer)
        _le_jour(r, jour)
        # A decision rolls the day over, but the wake-up hook does not run.
        r.noter_decision(1_700_000_000, depuis_cache=False)
        r.journaliser_compteurs()
    finally:
        _rendre_le_calendrier()
    alarmes = _erreurs(journal)
    assert len(alarmes) == 1 and "[ALARME]" in alarmes[0]
    assert lecteur in alarmes[0] and foyer in alarmes[0]


def test_des_lecteurs_jamais_tires_ne_se_disent_quune_fois(tmp_path, journal):
    """The wake-up hook never ran: it is a defect, said once, not five times."""
    r = _registre(tmp_path)
    try:
        for jour in (9, 10, 11):
            _le_jour(r, jour)
            r.noter_decision(1_700_000_000, depuis_cache=False)
        r.journaliser_compteurs()
    finally:
        _rendre_le_calendrier()
    alarmes = _erreurs(journal)
    assert len(alarmes) == 1 and "jamais été tirés" in alarmes[0]
