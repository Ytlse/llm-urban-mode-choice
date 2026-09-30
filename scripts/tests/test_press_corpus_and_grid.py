"""The twenty-sign grid and the press corpus.

Cases carry the rule numbers (G1…G5 for the grid, C1…C7 for the corpus).

Everything here is PURE: no simulator, no model, no network call.

Running:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_press_corpus_and_grid.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis.presse.corpus import (  # noqa: E402
    CONDITIONS,
    ECART_LONGUEUR_MAX,
    MENTION_TRADUCTION,
    RefusDeCorpus,
    charger_corpus,
)
from scripts.analysis.presse.grille import (  # noqa: E402
    CELLULES_ATTENDUES,
    RefusDeGrille,
    charger_grille,
)
from scripts.analysis.presse.lexique import mots_de_mobilite_trouves  # noqa: E402

GRILLE = RACINE / "data" / "presse" / "grille_signes.yaml"
CORPUS = RACINE / "data" / "presse" / "articles_txt"


def _grille_valide() -> dict:
    return yaml.safe_load(GRILLE.read_text(encoding="utf-8"))


def _ecrire(tmp_path: Path, d: dict) -> Path:
    p = tmp_path / "grille.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


# ── The shipped grid ────────────────────────────────────────────────────────────────────


def test_G1_la_grille_du_depot_porte_vingt_cellules():
    """Chapter 7 announces twenty predictions and sets the binomial bar at fifteen."""
    g = charger_grille(GRILLE)
    assert len(g.cellules) == CELLULES_ATTENDUES
    assert len(g.articles) == 5
    assert set(g.modes) == {"voiture", "velo", "marche", "tc"}


def test_G5_l_empreinte_est_stable_et_non_vide():
    """It is the only point that makes the "pre-registered" checkable after the fact."""
    assert charger_grille(GRILLE).empreinte == charger_grille(GRILLE).empreinte
    assert len(charger_grille(GRILLE).empreinte) == 64


def test_G3_les_trois_cellules_ambigues_sont_declarees_sans_effet_attendu():
    """Walking under La Machine, car under the refuse collectors, walking under VélôToulouse.

    They are not read after the measurement: they are declared, with their reason.
    """
    g = charger_grille(GRILLE)
    sans_effet = {(c.article, c.mode) for c in g.cellules if c.sans_effet_attendu}
    assert sans_effet == {
        ("a07_greve_eboueurs", "voiture"),
        ("a18_la_machine", "marche"),
        ("a25_velotoulouse", "marche"),
    }
    for article, mode in sans_effet:
        assert g.de(article, mode).motif, "a prediction without a reason cannot be discussed"


def test_G4_toute_cellule_porte_un_motif():
    assert all(c.motif.strip() for c in charger_grille(GRILLE).cellules)


def test_G_concordance_du_signe_zero_est_une_concordance():
    """Predicting no effect and observing it is a match, not an abstention."""
    c = charger_grille(GRILLE).de("a18_la_machine", "marche")
    assert c.concorde("0") is True
    assert c.concorde("+") is False
    assert c.concorde("-") is False


# ── What the grid refuses ───────────────────────────────────────────────────────────────


def test_G1_refus_cellule_manquante(tmp_path):
    d = _grille_valide()
    del d["articles"]["a09_vent_autan"]["cellules"]["velo"]
    with pytest.raises(RefusDeGrille, match="has no cell"):
        charger_grille(_ecrire(tmp_path, d))


def test_G1_refus_compte_different_de_vingt(tmp_path):
    d = _grille_valide()
    del d["articles"]["a25_velotoulouse"]
    with pytest.raises(RefusDeGrille, match="instead of 20"):
        charger_grille(_ecrire(tmp_path, d))


def test_G2_refus_signe_hors_domaine(tmp_path):
    d = _grille_valide()
    d["articles"]["a09_vent_autan"]["cellules"]["velo"]["signe"] = "?"
    with pytest.raises(RefusDeGrille, match="carries the sign"):
        charger_grille(_ecrire(tmp_path, d))


def test_G3_refus_signe_sans_intensite(tmp_path):
    """A cell that declares a sign without intensity could fall back on "no opinion"."""
    d = _grille_valide()
    d["articles"]["a09_vent_autan"]["cellules"]["velo"]["intensite"] = 0
    with pytest.raises(RefusDeGrille, match="equivalence is strict"):
        charger_grille(_ecrire(tmp_path, d))


def test_G3_refus_intensite_sans_signe(tmp_path):
    d = _grille_valide()
    d["articles"]["a18_la_machine"]["cellules"]["marche"]["intensite"] = 2
    with pytest.raises(RefusDeGrille, match="equivalence is strict"):
        charger_grille(_ecrire(tmp_path, d))


def test_G4_refus_motif_vide(tmp_path):
    d = _grille_valide()
    d["articles"]["a09_vent_autan"]["cellules"]["velo"]["motif"] = "  "
    with pytest.raises(RefusDeGrille, match="has no rationale"):
        charger_grille(_ecrire(tmp_path, d))


# ── The mobility lexicon — what makes C3 checkable ──────────────────────────────────────


@pytest.mark.parametrize(
    "texte,langue,attendu",
    [
        ("Les trottoirs sont encombrés et la chaussée glissante.", "fr", True),
        ("La mairie ferme les jardins par précaution ce soir.", "fr", False),
        ("Commuters avoided the metro and took their bikes.", "en", True),
        ("The council closed the gardens as a precaution.", "en", False),
    ],
)
def test_C2_le_lexique_voit_les_mots_de_mobilite(texte, langue, attendu):
    assert bool(mots_de_mobilite_trouves(texte, langue)) is attendu


def test_C2_le_lexique_ne_se_declenche_pas_sur_un_prefixe():
    """`car` must not come out on `careful`, nor `lane` on `planet`.

    An open suffix would produce incomprehensible refusals, and an incomprehensible refusal
    ends up being worked around.
    """
    trouves = mots_de_mobilite_trouves("He was careful, the planet is round, she is walking.", "en")
    assert "car" not in trouves
    assert "lane" not in trouves
    assert "walking" in trouves


def test_C2_le_lexique_voit_les_flexions_et_les_accents():
    assert "trottoir" in mots_de_mobilite_trouves("Les trottoirs sont pleins.", "fr")
    assert "chaussee" in mots_de_mobilite_trouves("La chaussée est glissante.", "fr")
    assert "ring road" in mots_de_mobilite_trouves("Stuck on the ring road again.", "en")


def test_C2_langue_inconnue_refusee():
    with pytest.raises(ValueError, match="unknown language"):
        mots_de_mobilite_trouves("texte", "es")


# ── The shipped corpus ──────────────────────────────────────────────────────────────────

pytestmark_corpus = pytest.mark.skipif(
    not (CORPUS / "MANIFEST.yaml").is_file(),
    reason="corpus not extracted — run `python -m scripts.data.presse.extraire_textes extraire`",
)


@pytestmark_corpus
def test_C1_les_cinq_textes_bruts_francais_existent_et_sont_non_vides():
    for article_id in (
        "a09_vent_autan",
        "a13_punaises_metro",
        "a07_greve_eboueurs",
        "a18_la_machine",
        "a25_velotoulouse",
    ):
        p = CORPUS / article_id / "brut.fr.txt"
        assert p.is_file(), f"{p} missing"
        assert len(p.read_text(encoding="utf-8").split()) > 50


@pytestmark_corpus
def test_C1_les_identifiants_du_corpus_sont_ceux_de_la_grille():
    """Two diverging files would score a prediction against another article."""
    manifeste = yaml.safe_load((CORPUS / "MANIFEST.yaml").read_text(encoding="utf-8"))
    assert set(manifeste["articles"]) == set(charger_grille(GRILLE).articles)


@pytestmark_corpus
def test_C7_le_manifeste_exige_de_nommer_qui_a_traduit(tmp_path):
    """An anonymous translation cannot be verified."""
    manifeste = yaml.safe_load((CORPUS / "MANIFEST.yaml").read_text(encoding="utf-8"))
    manifeste["traduction"] = {"par": None, "le": None}
    p = tmp_path / "MANIFEST.yaml"
    p.write_text(yaml.safe_dump(manifeste, allow_unicode=True), encoding="utf-8")
    with pytest.raises(RefusDeCorpus, match="anonymous translation"):
        charger_corpus(p)


def test_C6_la_mention_de_traduction_est_declaree_une_seule_fois():
    """The agent reads that it is reading a translation; the setup has no need to hide it."""
    assert MENTION_TRADUCTION == "Translated from French"


def test_C4_l_ecart_de_longueur_admis_est_declare():
    assert ECART_LONGUEUR_MAX == 0.15
    assert CONDITIONS == ("brut", "paraphrase", "temoin")


# ── The texts actually in the repository ────────────────────────────────────────────────

ARTICLES = (
    "a09_vent_autan",
    "a13_punaises_metro",
    "a07_greve_eboueurs",
    "a18_la_machine",
    "a25_velotoulouse",
)


@pytestmark_corpus
@pytest.mark.parametrize("article_id", ARTICLES)
def test_C1_chaque_article_porte_son_brut_dans_les_deux_langues(article_id):
    for nom in ("brut.fr.txt", "brut.txt"):
        p = CORPUS / article_id / nom
        assert p.is_file(), f"{p} missing"
        assert len(p.read_text(encoding="utf-8").split()) > 100


def _exemptions(article_id: str, langue: str) -> tuple[str, ...]:
    manifeste = yaml.safe_load((CORPUS / "MANIFEST.yaml").read_text(encoding="utf-8"))
    declare = (manifeste["articles"][article_id].get("c3_mots_autorises") or {}).get(langue) or {}
    return tuple(declare.get("mots") or ())


@pytestmark_corpus
@pytest.mark.parametrize("article_id", ARTICLES)
@pytest.mark.parametrize("nom,langue", [("paraphrase.fr.txt", "fr"), ("paraphrase.txt", "en")])
def test_C2_aucune_paraphrase_livree_ne_nomme_un_mode_de_report(article_id, nom, langue):
    """C3 only holds if the text names no mode TOWARDS WHICH a modal shift would be possible.

    The article's subject, however, is named: an article about bike sharing talks about bikes, and
    hiding it would make the paraphrase unintelligible without proving anything. It is the shift
    modes that must disappear — that is where, and only where, lexical copying would hide.

    Two words got caught at the first writing — `mises en ligne` and
    `underground` — that no human proofreading would have flagged.
    """
    p = CORPUS / article_id / nom
    trouves = mots_de_mobilite_trouves(
        p.read_text(encoding="utf-8"), langue, _exemptions(article_id, langue)
    )
    assert not trouves, f"{p} carries {list(trouves)}"


@pytestmark_corpus
@pytest.mark.parametrize("article_id", ARTICLES)
@pytest.mark.parametrize("langue", ["fr", "en"])
def test_C2_un_mot_exempte_figure_dans_le_texte_brut(article_id, langue):
    """A word is not exempted as a precaution; the one the subject imposes is exempted.

    Without this guard, exempting `voiture` on the bedbug article would be enough for the
    paraphrase to suggest the shift — and condition C3 would silently empty out.
    """
    nom = "brut.txt" if langue == "en" else "brut.fr.txt"
    brut = (CORPUS / article_id / nom).read_text(encoding="utf-8")
    presents = set(mots_de_mobilite_trouves(brut, langue))
    for mot in _exemptions(article_id, langue):
        assert mot in presents, f"{article_id} ({langue}) exempts {mot!r}, absent from the raw text"


@pytestmark_corpus
def test_C2_les_exemptions_ne_couvrent_jamais_un_mode_de_report():
    """No article exempts a mode towards which its event would push.

    This is the underlying guard: `métro` is exempted on the bedbug article because it is its
    subject; `vélo` is not exempted there, because it is its expected shift.
    """
    manifeste = yaml.safe_load((CORPUS / "MANIFEST.yaml").read_text(encoding="utf-8"))
    reports_interdits = {
        "a13_punaises_metro": {"velo", "voiture", "marche", "bike", "bicycle", "car", "walk"},
        "a25_velotoulouse": {"voiture", "bus", "marche", "car", "walk", "walking"},
    }
    for article_id, interdits in reports_interdits.items():
        declare = manifeste["articles"][article_id].get("c3_mots_autorises") or {}
        for langue in ("fr", "en"):
            mots = set((declare.get(langue) or {}).get("mots") or ())
            assert not (mots & interdits), f"{article_id} ({langue}) exempts a shift mode"


@pytestmark_corpus
@pytest.mark.parametrize("article_id", ARTICLES)
def test_C1_les_empreintes_du_manifeste_correspondent_aux_fichiers(article_id):
    """An article is QUOTED, never rewritten: a divergence makes loading refuse."""
    import hashlib

    manifeste = yaml.safe_load((CORPUS / "MANIFEST.yaml").read_text(encoding="utf-8"))
    declare = manifeste["articles"][article_id]
    verifiees = 0
    for condition in CONDITIONS:
        for langue, meta in (declare.get(condition) or {}).items():
            nom = f"{condition}.txt" if langue == "en" else f"{condition}.{langue}.txt"
            donnees = (CORPUS / article_id / nom).read_bytes()
            assert hashlib.sha256(donnees).hexdigest() == meta["sha256"], f"{article_id}/{nom}"
            verifiees += 1
    assert verifiees >= 4, f"{article_id}: {verifiees} declared hashes, raw and paraphrase expected"


# ── The common control (author's decision, 2026-09-21) ──────────────────────────────────


@pytestmark_corpus
def test_C4_le_corpus_complet_se_charge():
    """Loading is the check: it refuses everything that rules C1 to C9 forbid."""
    c = charger_corpus(CORPUS / "MANIFEST.yaml")
    assert set(c.articles) == set(ARTICLES)
    assert c.traduction_par and c.traduction_le


@pytestmark_corpus
def test_C4_un_seul_temoin_sert_les_cinq_articles():
    """A single control makes C4 COMPARABLE from one article to another.

    Five controls would make the specificity check depend on five distinct choices; what
    C4 measures — the effect of adding any text at all — requires only one.
    """
    c = charger_corpus(CORPUS / "MANIFEST.yaml")
    empreintes = {c.articles[a].texte("temoin", "en").empreinte for a in ARTICLES}
    assert len(empreintes) == 1


@pytestmark_corpus
@pytest.mark.parametrize("article_id", ARTICLES)
def test_C4_le_temoin_est_apparie_en_longueur_a_chaque_article(article_id):
    """What distinguishes C4 from C2 must not be the size of the block added to the context."""
    c = charger_corpus(CORPUS / "MANIFEST.yaml")
    a = c.articles[article_id]
    brut, temoin = a.texte("brut", "en").mots, a.texte("temoin", "en").mots
    assert abs(temoin - brut) / brut <= ECART_LONGUEUR_MAX


@pytestmark_corpus
def test_C6_l_entree_servie_a_l_agent_porte_la_mention_de_traduction():
    c = charger_corpus(CORPUS / "MANIFEST.yaml")
    for article_id in ARTICLES:
        assert c.articles[article_id].entree_pour_agent().startswith(f"[{MENTION_TRADUCTION}]")
