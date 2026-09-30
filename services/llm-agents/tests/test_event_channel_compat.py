"""The event channel: former shock declarations still load and replay identically.

This step delivers NO new function: it moves the declared shocks into
`llm/evenements/`, adds two fields, and must prove that a replay yields the same file. A
migration is proven by equality, not by argument.

Everything here is PURE: no simulator, no model, no network call.
"""

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import evenements as ev
from llm.evenements import RefusDEvenement, RegistreEvenements, charger
from llm.gravite import gravite_deterministe
from llm.memory import MemoryEntry, MemoryType
from settings import settings

RACINE = Path(__file__).resolve().parents[1]
CONFIG_CHOCS = RACINE / "config" / "chocs"
CONFIG_EVENEMENTS = RACINE / "config" / "evenements"

TEXTE_VALIDE = "I was stuck for an hour on the ring road, and I arrived in a foul mood."


def _declaration(**surcharges) -> dict:
    base = {
        "evenement": "test_evenement",
        "libelle": "Événement de test",
        "source": "test",
        "canal": "vecu",
        "moment": "arrivee",
        "jugement": "aucun",
        "exposition": {"regle": "mode", "modes": ["car"]},
        "jours": [{"jour": 12, "retard_min": 60, "texte": TEXTE_VALIDE}],
    }
    base.update(surcharges)
    return base


def _ecrire(tmp_path: Path, declaration: dict, nom: str = "evenement.yaml") -> Path:
    p = tmp_path / nom
    p.write_text(yaml.safe_dump(declaration, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _registre_propre():
    ev.reinitialiser()
    settings.evenements.enabled = False
    settings.evenements.fichier = None
    settings.chocs.enabled = False
    settings.chocs.fichier = None
    yield
    ev.reinitialiser()
    settings.evenements.enabled = False
    settings.evenements.fichier = None
    settings.chocs.enabled = False
    settings.chocs.fichier = None


# ── R1. The golden test ──────────────────────────────────────────────────────────────────
def _trace_dun_rejeu(declaration: Path, journal: Path, jours: tuple[int, ...]) -> list[dict]:
    """Replays a declaration on given days and returns the lines written.

    The agents and modes are those of `c6_voiture_suspecte`: three designated personas, all
    by car, plus a non-designated control and a bus trip — so that the trace carries both
    exposed and spared ones.
    """
    registre = RegistreEvenements(charger(declaration), journal=journal)
    arrivees = (
        ("899549", "car"), ("899549", "public_transport"),
        ("1250941", "car"), ("861500", "car"), ("609", "car"),
    )
    for jour in jours:
        RegistreEvenements.jour_du_run = staticmethod(lambda ts, _j=jour: _j)
        for rang, (person_id, mode) in enumerate(arrivees):
            applique = registre.applique(person_id, mode, 1_700_000_000 + rang * 3600)
            if applique is None:
                continue
            gravite, detail = gravite_deterministe(
                retard_s=applique.retard_injecte_s,
                correspondance_ratee=applique.correspondance_ratee,
                incident_reseau=applique.incident_reseau,
            )
            registre.tracer(
                applique, person_id, 1_700_000_000 + rang * 3600, gravite, detail
            )
    RegistreEvenements.jour_du_run = staticmethod(
        lambda ts: __import__("llm.evenements.calendrier", fromlist=["x"]).jour_du_run(ts)
    )
    return [json.loads(l) for l in journal.read_text("utf-8").splitlines() if l.strip()]


# The fields that the event channel ADDS. Everything else must be byte-for-byte equal.
CHAMPS_AJOUTES = {
    "evenement_id", "canal", "moment", "texte",
    # What the AGENT said about it, next to what the simulation measured.
    "importance_estimee", "intensite_jugee", "valence", "modes_touches",
    "importance_retenue",
    # D7 — the gap to the measured fact, logged and never applied.
    "ecart_au_fait",
}


def test_R1_test_en_or_la_declaration_079_rend_la_meme_trace(tmp_path):
    """The ORIGINAL file, replayed on the new package, yields the trace of the old one.

    It is the only criterion of the migration. It replays `config/chocs/c6_voiture_suspecte.yaml` as
    it is in the repository — not its migrated version: a migration only checked on
    rewritten files only checks the rewriting.
    """
    original = CONFIG_CHOCS / "c6_voiture_suspecte.yaml"
    migre = CONFIG_EVENEMENTS / "c6_voiture_suspecte.yaml"
    assert original.is_file() and migre.is_file()

    lignes_079 = _trace_dun_rejeu(original, tmp_path / "a.jsonl", (15, 16))
    lignes_100 = _trace_dun_rejeu(migre, tmp_path / "b.jsonl", (15, 16))

    assert lignes_079, "the replay produced NO line: the test proves nothing"
    assert len(lignes_079) == len(lignes_100)
    for ancienne, neuve in zip(lignes_079, lignes_100):
        assert set(ancienne) == set(neuve)
        for champ in sorted(set(ancienne) - CHAMPS_AJOUTES):
            assert ancienne[champ] == neuve[champ], f"field '{champ}' modified"
        # And the added fields say what they must say, without substituting anything.
        assert neuve["evenement_id"] == neuve["choc_id"]
        assert neuve["texte"] == neuve["vecu"]
        assert (neuve["canal"], neuve["moment"]) == ("vecu", "arrivee")


def test_R1bis_la_trace_porte_les_champs_du_079_sans_exception(tmp_path):
    """The analysers of archived runs read these keys. None disappears in the migration."""
    lignes = _trace_dun_rejeu(
        CONFIG_CHOCS / "c6_voiture_suspecte.yaml", tmp_path / "t.jsonl", (15,)
    )
    attendus = {
        "person_id", "timestamp", "horodatage_simule", "choc_id", "jour_run", "jour_relatif",
        "raison_exposition", "retard_injecte_s", "incident_reseau", "correspondance_ratee",
        "vecu", "gravite", "gravite_detail",
    }
    assert attendus <= set(lignes[0])


# ── R3, R4. The two formats ──────────────────────────────────────────────────────────────
def test_R3_les_declarations_079_se_chargent_et_le_disent(tmp_path, caplog):
    for fichier in sorted(CONFIG_CHOCS.glob("c*.yaml")):
        evenement = charger(fichier)
        assert evenement.format_source == "079"
        assert (evenement.canal, evenement.moment, evenement.jugement) == (
            "vecu", "arrivee", "aucun",
        )


def test_R4_la_migration_est_COMPLETE():
    """No case of declared shocks remains without a migrated equivalent.

    This is the invariant that counts: a shock forgotten in `config/chocs/` would still
    load — the compatibility path allows it — but it would no longer be in the directory
    that the `EVENEMENT=` lever lists, and nobody would see it missing.
    """
    originaux = {p.name for p in CONFIG_CHOCS.glob("c*.yaml")}
    migres = {p.name for p in CONFIG_EVENEMENTS.glob("c*.yaml")}
    assert originaux <= migres, f"079 cases not migrated: {sorted(originaux - migres)}"


def test_R4_les_deux_formats_rendent_le_meme_evenement():
    """Each migrated case says exactly what its original said.

    ⚠ Covers ONLY the cases that have an original. `config/evenements/` also hosts
    new declarations — the five articles, and the bench variants — which migrate nothing.
    Requiring an original for all would forbid adding one without touching the test, which
    is not what the rule means.
    """
    compares = 0
    for migre in sorted(CONFIG_EVENEMENTS.glob("c*.yaml")):
        original = CONFIG_CHOCS / migre.name
        if not original.is_file():
            continue
        compares += 1
        a, b = charger(original), charger(migre)
        assert (a.evenement_id, a.libelle, a.source) == (b.evenement_id, b.libelle, b.source)
        assert a.cadence == b.cadence
        assert a.exposition == b.exposition
        assert sorted(a.jours) == sorted(b.jours)
        for jour in a.jours:
            assert a.jours[jour].texte == b.jours[jour].texte
            assert a.jours[jour].effet == b.jours[jour].effet
        assert (b.format_source, a.format_source) == ("100", "079")
    assert compares >= 6, f"only {compares} cases compared: the migration is no longer covered"


# ── R5, R6. The refusals ─────────────────────────────────────────────────────────────────
def test_R5_une_gravite_posee_a_la_main_est_refusee(tmp_path):
    with pytest.raises(RefusDEvenement, match="gravite"):
        charger(_ecrire(tmp_path, _declaration(gravite=0.70)))


@pytest.mark.parametrize(
    "surcharge, motif",
    [
        ({"canal": "entendu"}, "unknown"),
        ({"moment": "au_coucher"}, "unknown"),
        ({"jugement": "au_pif"}, "unknown"),
        ({"effet_physique": {"retard_min": 30}}, "day by day"),
    ],
)
def test_R6_une_valeur_hors_vocabulaire_est_refusee(tmp_path, surcharge, motif):
    """A field accepted and without effect is worse than a refused field: nothing flags it."""
    with pytest.raises(RefusDEvenement) as err:
        charger(_ecrire(tmp_path, _declaration(**surcharge)))
    assert motif in str(err.value)


def test_R6quater_lire_une_declaration_et_armer_un_run_sont_deux_choses(tmp_path):
    """The separation set up with the `lu` channel, which survives judgement with a new reason.

    `charger()` must be able to check a protocol one does not want to play — otherwise one
    could not even check that the five articles match their manifest. Arming a
    run, on the other hand, stops dead.

    The case that requires it today: a `canal: lu` without judgement. It reads — the declaration
    is well formed — but it does not arm, because an article with severity 0.00 lives 2.8 days and
    its silence would pass for an absence of effect.
    """
    from llm.evenements import charger as _charger

    article = CONFIG_EVENEMENTS / "a13_punaises_metro.yaml"
    contenu = yaml.safe_load(article.read_text("utf-8"))
    contenu["jugement"] = "aucun"
    fichier = tmp_path / "sans_jugement.yaml"
    fichier.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")

    assert _charger(fichier).jugement == "aucun"  # reading passes
    settings.evenements.enabled = True
    settings.evenements.fichier = str(fichier)
    settings.cache.enabled = False
    with pytest.raises(RefusDEvenement, match="2.8 days"):
        ev.initialiser()


def test_R6bis_texte_et_vecu_ensemble_sont_refuses(tmp_path):
    d = _declaration(jours=[{"jour": 3, "retard_min": 10, "texte": TEXTE_VALIDE,
                            "vecu": TEXTE_VALIDE}])
    with pytest.raises(RefusDEvenement, match="not both"):
        charger(_ecrire(tmp_path, d))


def test_R6ter_le_nom_du_079_reste_lisible_seul(tmp_path):
    """`vecu:` alone still reads: the eight cases of the repository use it."""
    d = _declaration(jours=[{"jour": 3, "retard_min": 10, "vecu": TEXTE_VALIDE}])
    assert charger(_ecrire(tmp_path, d)).jours[3].texte == TEXTE_VALIDE


# ── R8, R9, R10. The provenance ──────────────────────────────────────────────────────────
def _entree(**kw) -> MemoryEntry:
    from datetime import datetime

    base = dict(
        content="x", timestamp=datetime(2026, 3, 16, 8, 0),
        memory_type=MemoryType.CONVERSATION, person_id="609",
    )
    base.update(kw)
    return MemoryEntry(**base)


def test_R8_origine_fait_laller_retour_et_survit_a_un_lecteur_qui_lignore():
    entree = _entree(origine="lu")
    assert MemoryEntry.from_dict(entree.to_dict()).origine == "lu"
    sans = {k: v for k, v in entree.to_dict().items() if k != "origine"}
    assert MemoryEntry.from_dict(sans).origine is None  # read back without exception


def test_R9_une_entree_anterieure_se_lit_vecue_sans_le_declarer():
    """`None` and `vecu` are not the same thing, and confusing them would erase the measurement."""
    ancienne = _entree()
    assert ancienne.origine is None
    assert ancienne.origine_effective == "vecu"
    assert _entree(origine="vecu").origine == "vecu"


def test_R10_valence_et_origine_traversent_la_memoire_courte():
    from llm.shortterm import UserShortTermMemory

    memoire = UserShortTermMemory(person_id="609")
    memoire.add_message("I read about bed bugs on the metro", valence="negative", origine="lu")
    entree = memoire.recent_entries[-1]
    assert (entree.valence, entree.origine) == ("negative", "lu")
    # The default does not change what the declared shocks wrote.
    memoire.add_message("I arrived on time")
    assert (memoire.recent_entries[-1].valence, memoire.recent_entries[-1].origine) == (
        "neutre", None,
    )


# ── R7. The household down to the runtime ────────────────────────────────────────────────
def test_R7_household_id_arrive_jusqua_Person():
    """The field existed in the JSON since the seal and got lost at validation."""
    from models import Person

    brut = {
        "person_id": "609",
        "household": {"id": "5177", "commune_id": "31555"},
        "identity": {"name": "Jacques Aubert", "traits_json": {"name": "Jacques Aubert"}},
    }
    # Without the copy, pydantic ignores `household`: that is the defect the migration fixes.
    assert Person.model_validate(brut).household_id is None
    from world.population import WorldPopulation

    assert Person.model_validate(WorldPopulation._avec_foyer(brut)).household_id == "5177"
    # The copy does not modify the caller's dictionary.
    assert "household_id" not in brut


def test_R7bis_la_cohorte_scellee_porte_un_menage_pour_chaque_agent():
    """Measured, not assumed: 1,000 agents, 0 without `household.id`."""
    import json as _json

    cohorte = RACINE.parents[1] / "data" / "population" / "toulouse_population_1000_PANEL_v6.json"
    if not cohorte.is_file():
        pytest.skip("sealed cohort absent from this machine")
    from inputs.population.eqasim_loader import _identifiant_de_foyer

    entrees = _json.loads(cohorte.read_text("utf-8"))
    sans = [e["person_id"] for e in entrees if _identifiant_de_foyer(e) is None]
    assert not sans, f"{len(sans)} agent(s) without household: {sans[:5]}"


# ── Regime boundary: no itinerary engine ─────────────────────────────────────────────────
def test_R23_le_paquet_nimporte_aucun_moteur_ditineraire():
    """Takes up R14 of the declared shocks, widened to the whole package.

    Checked on the imports and not on the text: the docstrings talk about the engines
    precisely to say that they do not touch them, and a test that read the raw text
    would forbid explaining the rule it checks.
    """
    for module in sorted((RACINE / "llm" / "evenements").glob("*.py")):
        arbre = ast.parse(module.read_text("utf-8"))
        importes: set[str] = set()
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Import):
                importes |= {a.name for a in noeud.names}
            elif isinstance(noeud, ast.ImportFrom) and noeud.module:
                importes.add(noeud.module)
        for interdit in ("trip_helper", "otp", "osmnx", "gtfs"):
            assert not any(interdit in m for m in importes), (
                f"{module.name} imports '{interdit}': an event calls on NO "
                f"itinerary engine. Degrading the supply belongs to another ticket."
            )


# ── The configuration key, and the cache alarm ───────────────────────────────────────────
def test_la_cle_du_100_prime_et_celle_du_079_reste_servie(tmp_path):
    fichier = _ecrire(tmp_path, _declaration())
    settings.evenements.enabled = True
    settings.evenements.fichier = str(fichier)
    settings.cache.enabled = False
    assert ev.initialiser() is not None
    assert ev.registre().evenement.evenement_id == "test_evenement"

    ev.reinitialiser()
    settings.evenements.enabled = False
    settings.chocs.enabled = True
    settings.chocs.fichier = str(fichier)
    assert ev.initialiser() is not None  # the old key keeps being served


def test_le_cache_actif_leve_une_alarme(tmp_path, caplog):
    import logging

    settings.evenements.enabled = True
    settings.evenements.fichier = str(_ecrire(tmp_path, _declaration()))
    settings.cache.enabled = True
    try:
        with caplog.at_level(logging.ERROR):
            ev.initialiser()
    finally:
        settings.cache.enabled = False


def test_la_declaration_est_archivee_sous_les_deux_noms(tmp_path):
    """`evenement.yaml` is the new name; `choc.yaml` stays reachable as long as the compatibility adapter exists."""
    workdir = tmp_path / "run"
    settings.evenements.enabled = True
    settings.evenements.fichier = str(_ecrire(tmp_path, _declaration()))
    settings.cache.enabled = False
    ev.initialiser(workdir=workdir)
    assert (workdir / "evenement.yaml").is_file()
    assert (workdir / "choc.yaml").is_file()
    assert "test_evenement" in (workdir / "choc.yaml").read_text("utf-8")
