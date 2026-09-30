"""Declared shocks, experienced by the agents.

The cases carry the rule numbers (R1, R7, R21…).

Everything here is PURE: no simulator, no model, no network call. The injection into the
controller is checked by reading the source (R14, R15), as the boundary test of the
accidents switch does.
"""

import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import chocs as chocs_module
from llm.chocs import RefusDeChoc, RegistreChocs, charger
from llm.gravite import composantes_sans_source, force_initiale, gravite_deterministe
from settings import settings

CONFIG_CHOCS = Path(__file__).resolve().parents[1] / "config" / "chocs"

VECU_VALIDE = "I was stuck for an hour on the ring road, and I arrived in a foul mood."


def _declaration(**surcharges) -> dict:
    base = {
        "choc": "test_choc",
        "libelle": "Choc de test",
        "source": "test",
        "exposition": {"regle": "mode", "modes": ["car"]},
        "jours": [{"jour": 12, "retard_min": 60, "vecu": VECU_VALIDE}],
    }
    base.update(surcharges)
    return base


def _ecrire(tmp_path: Path, declaration: dict) -> Path:
    p = tmp_path / "choc.yaml"
    p.write_text(yaml.safe_dump(declaration, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _registre_propre():
    """Registry reset, AND BOTH CONFIGURATION BLOCKS NEUTRALISED.

    ⚠ These tests set `settings.chocs.*`, the historical key. Since the event channel,
    `_declaration_demandee()` looks at `settings.evenements` FIRST and only falls back on `chocs`
    if the first block is off. As long as the repository's `config.yaml` still carried
    the old key, `settings.evenements.enabled` was false and these tests passed — for the
    wrong reason. On 2026-09-22, a campaign wrote the new key into that file and five
    of them failed at once, looking for a declaration at a container path.

    A test that depends on the state of a configuration file that every run rewrites does not test
    what it thinks. Both blocks are therefore turned off on entry and restored to their original
    value on exit: this file no longer reads `config.yaml` at all.
    """
    garde = {}
    for bloc in ("chocs", "evenements"):
        config = getattr(settings, bloc, None)
        if config is None:
            continue
        garde[bloc] = (getattr(config, "enabled", False), getattr(config, "fichier", None))
        config.enabled = False
        config.fichier = None

    chocs_module.reinitialiser()
    yield
    chocs_module.reinitialiser()

    for bloc, (enabled, fichier) in garde.items():
        config = getattr(settings, bloc)
        config.enabled = enabled
        config.fichier = fichier


# ── A. Reading and refusal ─────────────────────────────────────────────────────────────
def test_R1_declaration_valide_se_charge(tmp_path):
    choc = charger(_ecrire(tmp_path, _declaration()))
    assert choc.choc_id == "test_choc"
    assert choc.premier_jour == 12 and choc.dernier_jour == 12
    assert choc.jours[12].retard_s == 3600
    assert choc.jours[12].incident_reseau is True  # default: a shock IS an incident
    assert choc.empreinte  # the file fingerprint travels with the declaration


def test_R2_sans_fichier_le_registre_reste_vide():
    """A run that asks for nothing behaves exactly as without declared shocks."""
    settings.chocs.enabled = False
    settings.chocs.fichier = None
    assert chocs_module.initialiser() is None
    assert chocs_module.registre() is None
    assert chocs_module.incident_reseau_a_une_source() is False


def test_R3_le_chargement_se_journalise(tmp_path, caplog):
    settings.chocs.enabled = True
    settings.chocs.fichier = str(_ecrire(tmp_path, _declaration()))
    settings.cache.enabled = False
    chocs_module.initialiser()
    assert chocs_module.registre() is not None
    assert chocs_module.registre().choc.libelle == "Choc de test"


@pytest.mark.parametrize(
    "surcharge, motif",
    [
        ({"jours": []}, "no day declared"),
        ({"jours": [{"jour": 0, "retard_min": 10, "vecu": VECU_VALIDE}]}, "number 1"),
        (
            {
                "jours": [
                    {"jour": 3, "retard_min": 10, "vecu": VECU_VALIDE},
                    {"jour": 3, "retard_min": 20, "vecu": VECU_VALIDE},
                ]
            },
            "two entries",
        ),
        ({"jours": [{"jour": 3, "retard_min": -5, "vecu": VECU_VALIDE}]}, "negative"),
        ({"jours": [{"jour": 3, "retard_min": 10, "vecu": "  "}]}, "empty"),
    ],
)
def test_R4_declarations_invalides_refusees(tmp_path, surcharge, motif):
    with pytest.raises(RefusDeChoc) as err:
        charger(_ecrire(tmp_path, _declaration(**surcharge)))
    assert motif in str(err.value)


def test_R5_regle_exposition_inconnue_refusee(tmp_path):
    d = _declaration(exposition={"regle": "au_hasard", "modes": ["car"]})
    with pytest.raises(RefusDeChoc, match="unknown"):
        charger(_ecrire(tmp_path, d))


def test_R6_mode_hors_hierarchie_refuse(tmp_path):
    """The vocabulary comes from `llm/axes.py`, never from a list copied here."""
    d = _declaration(exposition={"regle": "mode", "modes": ["teleportation"]})
    with pytest.raises(RefusDeChoc, match="outside the repository"):
        charger(_ecrire(tmp_path, d))


def test_R5bis_mode_sans_liste_refuse(tmp_path):
    with pytest.raises(RefusDeChoc, match="without any declared mode"):
        charger(_ecrire(tmp_path, _declaration(exposition={"regle": "mode", "modes": []})))


# ── B. The lived experience is lived experience ────────────────────────────────────────
@pytest.mark.parametrize(
    "texte",
    [
        "You should avoid the ring road tomorrow.",
        "Avoid the metro from now on.",
        "Remember to take the bike instead.",
        "Tu devrais éviter la rocade demain.",
        "Don't take the car again.",
    ],
)
def test_R7_consigne_deguisee_en_vecu_refusee(tmp_path, texte):
    """A FIRM refusal, not a warning: an instruction that gets through fabricates the result."""
    d = _declaration(jours=[{"jour": 3, "retard_min": 10, "vecu": texte}])
    with pytest.raises(RefusDeChoc):
        charger(_ecrire(tmp_path, d))


def test_R8_vecu_sans_premiere_personne_avertit_mais_passe(tmp_path):
    d = _declaration(
        jours=[{"jour": 3, "retard_min": 10, "vecu": "Flat tyre, hands covered in grease."}]
    )
    choc = charger(_ecrire(tmp_path, d))  # does not raise
    assert choc.jours[3].vecu.startswith("Flat tyre")


# The five originally SHIPPED cases, named and not counted: the directory also holds
# the case studies declared over the experiments (c6 on 2026-09-15), and a test that counts the
# files would forbid adding one without touching the test — which is not what it means.
CAS_DU_TICKET = (
    "c1_bouchon_rocade",
    "c2_crevaison",
    "c3_panne_reseau",
    "c4_train_supprime",
    "c5_orage_grele",
)


def test_R9_les_cinq_cas_livres_passent():
    modes_couverts = set()
    for nom in CAS_DU_TICKET:
        chemin = CONFIG_CHOCS / f"{nom}.yaml"
        assert chemin.is_file(), f"ticket case missing: {nom}"
        choc = charger(chemin)  # no exception, no refusal
        assert choc.jours
        modes_couverts |= set(choc.exposition.modes)
    # The repository's six modes are covered by at least one case.
    assert modes_couverts == {
        "car", "cycling", "public_transport", "walking", "train", "motorbike",
    }


def test_R9bis_tous_les_cas_du_repertoire_se_chargent():
    """Including case studies added later: no dead file in this folder."""
    fichiers = sorted(CONFIG_CHOCS.glob("c*.yaml"))
    assert len(fichiers) >= len(CAS_DU_TICKET)
    for f in fichiers:
        choc = charger(f)
        assert choc.jours, f"{f.name} declares no day"


# ── C. Application ─────────────────────────────────────────────────────────────────────
def _registre(tmp_path, **surcharges) -> RegistreChocs:
    return RegistreChocs(charger(_ecrire(tmp_path, _declaration(**surcharges))))


def test_R10_agent_expose_recoit_retard_texte_et_incident(tmp_path, monkeypatch):
    r = _registre(tmp_path)
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    a = r.applique("609", "car", 1000)
    assert a is not None
    assert a.retard_injecte_s == 3600
    assert a.incident_reseau is True
    assert a.vecu == VECU_VALIDE
    assert a.raison == "mode:car"


def test_R12_agent_non_expose_ne_recoit_rien(tmp_path, monkeypatch):
    r = _registre(tmp_path)
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    assert r.applique("609", "cycling", 1000) is None
    assert r._compteurs.epargnes == 1  # the internal control is COUNTED, not deduced


def test_R13_jour_nominal_nappelle_rien_mais_garde_labscisse(tmp_path, monkeypatch):
    r = _registre(tmp_path)
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 10))
    assert r.applique("609", "car", 1000) is None
    assert r.jour_relatif(1000) == -2  # two days BEFORE the shock
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 15))
    assert r.jour_relatif(1000) == +3


def test_R13bis_jour_zero_est_lexposition_effective_du_foyer(tmp_path, monkeypatch):
    """A D9–D13 window must not show +3 to the household actually exposed on D12."""
    r = _registre(tmp_path)
    monkeypatch.setattr(r, "_date_injection", lambda person_id: date(2026, 3, 27))
    jours = iter((date(2026, 3, 27), date(2026, 3, 30)))
    monkeypatch.setattr(r, "_date_de", lambda timestamp: next(jours))
    assert r.jour_relatif(1000, "1127259") == 0
    assert r.jour_relatif(1000, "1127259") == 3


def test_R14_aucun_moteur_ditineraire_nest_sollicite():
    """Boundary of the endured regime: the module IMPORTS no itinerary engine.

    Checked on the imports and not on the file's text: the docstring mentions the engines
    precisely to say it does not touch them, and a test reading the raw text
    would forbid explaining the rule it checks.
    """
    import ast

    arbre = ast.parse((Path(__file__).resolve().parents[1] / "llm" / "chocs.py").read_text("utf-8"))
    importes = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            importes |= {a.name for a in noeud.names}
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            importes.add(noeud.module)
    for interdit in ("trip_helper", "otp", "osmnx", "gtfs"):
        assert not any(interdit in m for m in importes), (
            f"chocs.py imports '{interdit}': the endured regime calls NO itinerary "
            f"engine. A shock that degrades the offer belongs to another ticket."
        )


def test_R15_lagenda_est_decale_par_le_retard_MESURE_seulement():
    """The injected delay reschedules nothing: rescheduling reads GAMA's observation."""
    ctrl = (
        Path(__file__).resolve().parents[1]
        / "urban_mobility_agents"
        / "simulation_controller.py"
    ).read_text("utf-8")
    assert "arrival_late_seconds=ob.late" in ctrl, (
        "rescheduling must read the delay from the OBSERVATION"
    )
    for ligne in ctrl.splitlines():
        if "reschedule" in ligne.lower():
            assert "_retard_injecte_s" not in ligne, (
                "the injected delay must never enter a rescheduling: "
                "the agenda would stay comparable between a shock run and its nominal twin"
            )


# ── D. Exposure ──────────────────────────────────────────────────────────────────────────
def test_R17_exposition_par_mode(tmp_path, monkeypatch):
    r = _registre(tmp_path, exposition={"regle": "mode", "modes": ["car", "motorbike"]})
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    assert r.applique("1", "car", 0) is not None
    assert r.applique("2", "motorbike", 0) is not None
    assert r.applique("3", "walking", 0) is None
    assert r.applique("4", None, 0) is None


def test_R18_tirage_deterministe_et_stable(tmp_path, monkeypatch):
    d = {"regle": "tirage", "part": 0.3, "graine": 79}
    r1 = _registre(tmp_path, exposition=d)
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    touches_1 = {p for p in map(str, range(200)) if r1.applique(p, "car", 0)}
    # A second registry, built separately: same seed, same agents hit.
    r2 = _registre(tmp_path, exposition=d)
    touches_2 = {p for p in map(str, range(200)) if r2.applique(p, "car", 0)}
    assert touches_1 == touches_2
    # And the arrival order of observations changes nothing.
    r3 = _registre(tmp_path, exposition=d)
    touches_3 = {p for p in reversed(list(map(str, range(200)))) if r3.applique(p, "car", 0)}
    assert touches_1 == touches_3


def test_R19_la_part_tiree_est_celle_declaree(tmp_path, monkeypatch):
    r = _registre(tmp_path, exposition={"regle": "tirage", "part": 0.30, "graine": 79})
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    touches = sum(1 for p in map(str, range(2000)) if r.applique(p, "car", 0))
    assert 0.27 <= touches / 2000 <= 0.33


def test_R20_exposition_par_agents_designes(tmp_path, monkeypatch):
    r = _registre(tmp_path, exposition={"regle": "agents", "agents": ["609", "41275"]})
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    assert r.applique("609", "walking", 0) is not None  # without `modes`, the mode is not taken into account
    assert r.applique("99999", "car", 0) is None


def test_R20bis_agents_designes_et_modes_declares_se_conjuguent(tmp_path, monkeypatch):
    """`modes` was parsed, validated, then IGNORED by the `agents` rule (2026-09-15).

    A field accepted and without effect is worse than a refused field: nothing flags it. The case
    that requires it is a CAR incident placed on a multimodal agent — without conjunction, it
    read "the engine made a grinding noise" on returning from a bus trip, and its memory
    recorded an impossible story.
    """
    r = _registre(
        tmp_path,
        exposition={"regle": "agents", "agents": ["899549"], "modes": ["car"]},
    )
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    assert r.applique("899549", "car", 0) is not None
    assert r.applique("899549", "public_transport", 0) is None
    assert r.applique("899549", "walking", 0) is None
    assert r.applique("899549", None, 0) is None  # unknown mode: we do not expose at random
    assert r.applique("609", "car", 0) is None  # the non-designated agent stays spared


# ── E. Severity and memory ───────────────────────────────────────────────────────────────
def test_R21_un_bouchon_dune_heure_depasse_le_seuil_de_choc():
    """The computation is not copied: it comes from `llm/gravite.py`.

    ⚠ An hour of traffic jam is NO LONGER worth the same as half an hour. Under the
    plateau, both gave 0.70: a one-hour jam and a thirty-minute jam
    were the same event for memory. The asymptotic component separates them, and that is
    the whole point of the change.
    """
    gravite, detail = gravite_deterministe(retard_s=3600, incident_reseau=True)
    assert gravite == pytest.approx(0.8835830003, abs=1e-9)
    assert gravite >= settings.agent.memoire__importance_choc  # enters the shock pool
    assert detail.retard == pytest.approx(0.6835830003, abs=1e-9)
    assert detail.incident_reseau == pytest.approx(0.20)
    assert force_initiale(gravite) == pytest.approx(17.64, abs=0.01)  # days, versus 2.8
    # And half an hour stays at the anchor point: it has not moved.
    demi, _ = gravite_deterministe(retard_s=1800, incident_reseau=True)
    assert demi == pytest.approx(0.70, abs=1e-9)
    assert gravite > demi


def test_R21bis_la_panne_reseau_est_le_choc_le_plus_marquant():
    gravite, _ = gravite_deterministe(
        retard_s=2700, correspondance_ratee=True, incident_reseau=True
    )
    # ⚠ It now SATURATES the scale. 45 minutes of delay are worth 0.64 instead
    # of 0.50; with the missed connection (0.20) and the incident (0.20), the total exceeds 1 and
    # the clamping bites. It is the only shock of the catalogue in this case, and it lives up to its name.
    assert gravite == pytest.approx(1.00, abs=1e-9)
    assert force_initiale(gravite) == pytest.approx(19.60, abs=0.01)


def test_R22_incident_reseau_cesse_detre_inactif_quand_un_choc_le_porte(tmp_path):
    assert "incident_reseau" in composantes_sans_source()  # without shock: declared inactive
    settings.chocs.enabled = True
    settings.chocs.fichier = str(_ecrire(tmp_path, _declaration()))
    settings.cache.enabled = False
    chocs_module.initialiser()
    assert chocs_module.incident_reseau_a_une_source() is True
    assert "incident_reseau" not in composantes_sans_source()


def test_R22bis_un_choc_sans_incident_reseau_ne_la_reveille_pas(tmp_path):
    d = _declaration(
        jours=[{"jour": 3, "retard_min": 10, "vecu": VECU_VALIDE, "incident_reseau": False}]
    )
    settings.chocs.enabled = True
    settings.chocs.fichier = str(_ecrire(tmp_path, d))
    settings.cache.enabled = False
    chocs_module.initialiser()
    assert chocs_module.incident_reseau_a_une_source() is False
    assert "incident_reseau" in composantes_sans_source()


# ── F. Trace ───────────────────────────────────────────────────────────────────────────
def test_R27_chaque_application_laisse_une_ligne(tmp_path, monkeypatch):
    import json

    journal = tmp_path / "chocs.jsonl"
    r = RegistreChocs(charger(_ecrire(tmp_path, _declaration())), journal=journal)
    monkeypatch.setattr(RegistreChocs, "jour_du_run", staticmethod(lambda ts: 12))
    a = r.applique("609", "car", 1_700_000_000)
    gravite, detail = gravite_deterministe(retard_s=a.retard_injecte_s, incident_reseau=True)
    r.tracer(a, "609", 1_700_000_000, gravite, detail)
    ligne = json.loads(journal.read_text("utf-8").strip())
    assert ligne["person_id"] == "609"
    assert ligne["choc_id"] == "test_choc"
    assert ligne["jour_relatif"] == 0
    assert ligne["retard_injecte_s"] == 3600
    assert ligne["vecu"] == VECU_VALIDE
    assert ligne["gravite"] == pytest.approx(0.8836, abs=1e-4)  # log rounding
    assert ligne["gravite_detail"]["incident_reseau"] == pytest.approx(0.20)


def test_R28_la_declaration_est_archivee_dans_le_run(tmp_path):
    workdir = tmp_path / "run"
    settings.chocs.enabled = True
    settings.chocs.fichier = str(_ecrire(tmp_path, _declaration()))
    settings.cache.enabled = False
    chocs_module.initialiser(workdir=workdir)
    assert (workdir / "choc.yaml").is_file()
    assert "test_choc" in (workdir / "choc.yaml").read_text("utf-8")


def test_R11_les_deux_retards_ne_se_confondent_jamais():
    """The controller adds up for the severity, but logs SEPARATELY."""
    ctrl = (
        Path(__file__).resolve().parents[1]
        / "urban_mobility_agents"
        / "simulation_controller.py"
    ).read_text("utf-8")
    assert "_retard_observe_s + _retard_injecte_s" in ctrl  # the severity sees the sum
    assert "retard_injecte_s=_retard_injecte_s" in ctrl  # the log sees both
    logger = (
        Path(__file__).resolve().parents[1]
        / "urban_mobility_agents"
        / "utils"
        / "move_logger.py"
    ).read_text("utf-8")
    assert '"retard_injecte_s"' in logger and '"delay_s"' in logger
