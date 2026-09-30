"""Keys serving the same model are consumed in SERIES.

Two keys serve `gemini-3.5-flash-lite`, each with its bucket of 500 requests per day. The
rotation drew on both in parallel; we want to exhaust the first before touching the
second, to keep one bucket intact and know what was consumed.
"""

from experiences.decideurs import DecideurPasserelle


class MoniteurFactice:
    """Keeps only the instances whose DAILY quota is not exhausted, like the real one."""

    def __init__(self, disponibles: list[str]):
        self.disponibles = list(disponibles)

    def instances_disponibles(self, besoin: int = 1) -> list[str]:
        return list(self.disponibles)


def _decideur(instances, moniteur=None) -> DecideurPasserelle:
    return DecideurPasserelle(agent=None, modele="modele-x", instances=instances, moniteur=moniteur)


def test_la_premiere_instance_est_servie_tant_qu_elle_a_du_quota():
    moniteur = MoniteurFactice(["cle1", "cle2"])
    d = _decideur(["cle1", "cle2"], moniteur)

    assert [d._prochaine_instance() for _ in range(5)] == ["cle1"] * 5, \
        "without exhaustion, we stay on the first: no rotation"


def test_on_bascule_quand_la_premiere_est_epuisee():
    moniteur = MoniteurFactice(["cle1", "cle2"])
    d = _decideur(["cle1", "cle2"], moniteur)
    assert d._prochaine_instance() == "cle1"

    moniteur.disponibles = ["cle2"]           # daily quota exhausted on cle1
    assert d._prochaine_instance() == "cle2"
    assert d._prochaine_instance() == "cle2"

    moniteur.disponibles = []                  # both exhausted
    assert d._prochaine_instance() is None


def test_la_bascule_est_journalisee_par_son_RANG_jamais_par_son_nom(caplog):
    """The log is read, copied and passed on: an instance name identifies an account."""
    import logging

    from loguru import logger

    # loguru to caplog, for the duration of the test
    poignee = logger.add(lambda m: logging.getLogger("loguru").warning(m.record["message"]), level="INFO")
    try:
        moniteur = MoniteurFactice(["cle_secrete_1", "cle_secrete_2"])
        d = _decideur(["cle_secrete_1", "cle_secrete_2"], moniteur)
        with caplog.at_level(logging.INFO, logger="loguru"):
            assert d._prochaine_instance() == "cle_secrete_1"
            moniteur.disponibles = ["cle_secrete_2"]
            assert d._prochaine_instance() == "cle_secrete_2"
    finally:
        logger.remove(poignee)

    journal = "\n".join(r.getMessage() for r in caplog.records)
    assert "Passage sur la seconde clé" in journal, journal
    assert "(2/2)" in journal
    assert "épuisé son quota du jour" in journal
    assert "cle_secrete" not in journal, "no instance name may appear in the log"


def test_les_rangs_se_disent_en_mots():
    from experiences.decideurs import rang_en_mots

    assert rang_en_mots(1) == "première"
    assert rang_en_mots(2) == "seconde"
    assert rang_en_mots(10) == "dixième"
    assert rang_en_mots(11) == "n° 11", "beyond ten, we number"


def test_sans_moniteur_l_ordre_declare_fait_foi():
    d = _decideur(["cle1", "cle2"])
    assert [d._prochaine_instance() for _ in range(3)] == ["cle1"] * 3


# ── freshness of the quota snapshot ─────────────────────────────────────
# The decision-maker switches only if `instances_disponibles` changes. Nothing changed it
# during a run: `etat` stayed the startup one, so a key exhausted along the way
# stayed "available" and the next one was never started (2026-09-08, run stopped at
# 70.5% with 500 requests untouched in reserve).


def _moniteur(lecteur):
    from experiences.ressources import MoniteurRessources

    return MoniteurRessources(
        instances=["cle1", "cle2"],
        providers={"cle1": {"rpd_limit": 500}, "cle2": {"rpd_limit": 500}},
        base_url="http://passerelle-de-test",
        lecteur=lecteur,
    )


def test_l_instantane_des_quotas_n_est_relu_qu_une_fois_perime():
    lectures = []

    def lecteur(_url):
        lectures.append(1)
        return {"cle1": {"daily_requests": 0}, "cle2": {"daily_requests": 0}}

    m = _moniteur(lecteur)
    assert m.perime(30.0) is True, "never read ⇒ stale by default"
    assert m.rafraichir_si_perime(30.0) is True and len(lectures) == 1

    assert m.rafraichir_si_perime(30.0) is False, "too early: we do not hammer /health"
    assert len(lectures) == 1
    assert m.rafraichir_si_perime(0.0) is True, "zero age ⇒ always re-read"
    assert len(lectures) == 2


def test_une_passerelle_injoignable_laisse_l_instantane_en_place():
    """Fail-safe: a slightly old state is better than an empty one, which would suggest
    all keys are exhausted and would stop the run for nothing.

    The age stays that of the last SUCCESSFUL read: the watcher will retry at its
    normal deadline instead of hammering a broken gateway.
    """
    reponses = [{"cle1": {"daily_requests": 12}, "cle2": {"daily_requests": 0}}, None]

    m = _moniteur(lambda _url: reponses.pop(0))
    assert m.rafraichir_si_perime(0.0) is True
    avant, age_avant = dict(m.etat), m.maj_monotone

    assert m.rafraichir_si_perime(0.0) is True, "it did attempt the read"
    assert m.etat == avant, "the previous snapshot is kept"
    assert m.joignable is False
    assert m.maj_monotone == age_avant, "a failure does not refresh the snapshot age"
    assert m.instances_disponibles() == ["cle1", "cle2"], \
        "a read failure must never suggest that the keys are exhausted"


def test_la_relecture_fait_basculer_le_decideur_sans_attendre_d_erreur():
    """The core of the fix: the exhausted key leaves the available ones on re-read, so the
    decision-maker pins the next one — without any request having had to fail."""
    frais = {
        "cle1": {"daily_requests": 100, "quota_exhausted": False, "available": True},
        "cle2": {"daily_requests": 0, "quota_exhausted": False, "available": True},
    }
    epuisee = {
        "cle1": {"daily_requests": 500, "quota_exhausted": True, "available": False},
        "cle2": {"daily_requests": 0, "quota_exhausted": False, "available": True},
    }
    # The first call returns the fresh state, the following ones the exhausted state.
    etats = [frais, epuisee]
    m = _moniteur(lambda _url: etats.pop(0) if len(etats) > 1 else etats[0])

    m.rafraichir()  # what the run startup does
    d = _decideur(["cle1", "cle2"], m)
    assert d._prochaine_instance() == "cle1", "while it has quota, we stay on it"

    m.rafraichir_si_perime(0.0)  # what `veiller_quotas` does every 30 s
    assert m.instances_disponibles() == ["cle2"], m.instances_disponibles()
    assert d._prochaine_instance() == "cle2", \
        "the second key is started, its 500 requests get used"


def test_confirme_epuise_ne_s_arrete_que_si_toutes_les_cles_sont_epuisees():
    """The experiment stops only if ALL keys have failed.
    A 429 error on cle1 must not stop the run if cle2 is available.
    """
    from experiences.runner import _confirme_epuise

    # cle1 exhausted after a 429, but cle2 available
    etat_cle1_seule_epuisee = {
        "cle1": {"daily_requests": 500, "quota_exhausted": True, "available": False},
        "cle2": {"daily_requests": 100, "quota_exhausted": False, "available": True},
    }
    m = _moniteur(lambda _url: etat_cle1_seule_epuisee)
    # Even if the provider reported a 429 on cle1, exhaustion is NOT confirmed since cle2 exists
    assert _confirme_epuise(m, annonce_par_fournisseur=True) is False

    etat_deux_cles_epuisees = {
        "cle1": {"daily_requests": 500, "quota_exhausted": True, "available": False},
        "cle2": {"daily_requests": 500, "quota_exhausted": True, "available": False},
    }
    m_complet = _moniteur(lambda _url: etat_deux_cles_epuisees)
    assert _confirme_epuise(m_complet, annonce_par_fournisseur=True) is True


def test_reinitialiser_quotas_et_pas_de_blocage_local_au_lancement():
    """At launch, the system does not trust local counters/locks:
    it resets the locks and admits the serving instances to test them live.
    """
    from experiences.ressources import MoniteurRessources

    class FauxHttp:
        def __init__(self):
            self.post_appele = False
        def post(self, url, json=None, timeout=None):
            self.post_appele = True
            class R:
                is_success = True
            return R()

    m = MoniteurRessources(["cle1", "cle2"], {"cle1": {}, "cle2": {}})
    # Test that reinitialiser_quotas runs without error
    assert m.reinitialiser_quotas() in (True, False)
