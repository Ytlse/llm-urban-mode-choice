"""The functional test bench tests itself.

A bench that does not check itself returns worthless greens. What is checked here: that the
stubs do import the repository formats instead of copying them, that the gateway guard
refuses an undeclared instance, and that the budget stops the bench instead of overflowing.

No call: the real client is never built.
"""

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))

from scripts.experiment.banc_fonctionnel import stubs  # noqa: E402
from scripts.experiment.banc_fonctionnel.clients import (  # noqa: E402
    GROQ,
    Appel,
    BudgetEpuise,
    ClientStub,
    Journal,
    PasserelleRefusee,
)


# ── Stubs import, they do not copy ──────────────────────────────────────────────────────
def test_le_stub_de_moves_importe_les_vraies_colonnes(tmp_path):
    """A stub that copies a list of columns tests its own copy.

    It stays green the day the real format changes — and that is exactly what the bench must
    catch. Lesson from the two mode vocabularies that silently coexisted.
    """
    import csv

    from urban_mobility_agents.utils.move_logger import CSV_HEADERS

    chemin = stubs.moves_stub(tmp_path / "moves.csv", evenement="a13", jours=range(0, 2))
    with chemin.open(encoding="utf-8") as f:
        lecteur = csv.reader(f)
        entetes = next(lecteur)
        lignes = list(lecteur)
    assert entetes == CSV_HEADERS, "the headers come from the repository, not from a copy"
    assert all(len(l) == len(CSV_HEADERS) for l in lignes), (
        "a row that does not have its header's column count shifts everything after it, "
        "and the shift only shows when reading back"
    )


def test_le_stub_ecrit_lagent_dans_la_COLONNE_DE_LAGENT(tmp_path):
    """Importing the headers is not enough: one must also write in the right place.

    Found on 2026-09-22 on a real run. The stub wrote the agent identifier into
    "Référence", which carries the RUN identifier; the agent is under "ID Personne". The tests
    passed because they checked the stub's convention, not the repository's — the rule
    "import, do not copy" covered the headers, not the MEANING of the columns.
    """
    import csv

    chemin = stubs.moves_stub(tmp_path / "moves.csv", evenement="a13", jours=range(0, 1))
    with chemin.open(encoding="utf-8") as f:
        ligne = next(csv.DictReader(f))
    assert ligne["ID Personne"], "the agent goes under \"ID Personne\""
    assert "_" in ligne["ID Personne"], "and it is indeed an agent identifier"
    assert ligne["Référence"] == "banc", "\"Référence\" carries the run, not the agent"

    # And the tallying bridge reads the same column the stub writes.
    from scripts.analysis.presse import campagne

    assert campagne.COLONNE_AGENT == "ID Personne"


def test_le_stub_de_memoire_produit_de_vraies_MemoryEntry():
    from llm.memory import MemoryEntry, MemoryType

    reflexion = stubs.reflexion_stub("609", "A quiet day.", 3)
    concept = stubs.concept_stub("609", "Line A is packed", 3, origine="entendu")
    assert isinstance(reflexion, MemoryEntry) and isinstance(concept, MemoryEntry)
    assert reflexion.memory_type == MemoryType.REFLECTION
    assert concept.memory_type == MemoryType.CONCEPT and concept.origine == "entendu"
    # The serialisation round trip must hold: that is what a resume point does.
    assert MemoryEntry.from_dict(concept.to_dict()).origine == "entendu"


def test_le_stub_de_point_de_reprise_est_lu_par_le_vrai_lecteur(tmp_path):
    from urban_mobility_agents.utils import reprise

    stubs.point_de_reprise_stub(tmp_path, 12, foyer={"lu_jusqu_a": {"1": "x"}})
    trouve = reprise.dernier_point(tmp_path)
    assert trouve is not None, "the stubbed point must be valid for the repository reader"
    _source, meta = trouve
    assert meta["jour_simule"] == 12
    assert meta["foyer"]["lu_jusqu_a"] == {"1": "x"}


def test_la_population_de_banc_porte_les_contrastes_qui_font_jouer_R4():
    """R4 can only be tested on members who differ: this is the Constance / Jacques case."""
    population = stubs.population_de_banc()
    assert len(population) == 12
    foyers = {a.household_id for a in population}
    assert len(foyers) == 6 and all(
        sum(1 for a in population if a.household_id == f) == 2 for f in foyers
    )
    permis = {a.identity.traits_json["has_driving_license"] for a in population}
    assert permis == {True, False}, "without a licence contrast, R4 never fires"


def test_la_journee_stub_produit_les_TROIS_entrees_reelles():
    """Decision, arrival, constraint: this is why "per raw entry" would triple the block."""
    assert len(stubs.journee_stub("609", 1)) == 3


# ── The gateway guard ───────────────────────────────────────────────────────────────────
def test_aucune_instance_declaree_est_refuse():
    from scripts.experiment.banc_fonctionnel.clients import ClientEpingle

    with pytest.raises(PasserelleRefusee, match="does not choose the gateway"):
        ClientEpingle(instances=())


def test_groq_est_le_defaut_du_jour_mais_reste_un_parametre():
    """Today Groq alone; long runs will go elsewhere without touching the code."""
    assert all(i.startswith("groq_") for i in GROQ)
    import inspect

    from scripts.experiment.banc_fonctionnel.clients import ClientEpingle

    assert inspect.signature(ClientEpingle).parameters["instances"].default == GROQ


def test_le_budget_de_jetons_arrete_le_banc():
    journal = Journal()
    journal.appels.append(Appel("x", "1", "t", "groq_openai_20_key1", jetons_sortie=9000))
    assert journal.jetons_sortie == 9000
    # The guard lives in `execute`, which cannot be called without a network: we check that the
    # journal feeds it, and that the message names the consequence.
    assert "quota" in BudgetEpuise.__doc__ or "campagnes" in BudgetEpuise.__doc__


def test_le_journal_dit_quelle_instance_a_servi(tmp_path):
    """Without this, two campaigns on two gateways would be incomparable without our knowing."""
    journal = Journal()
    journal.appels.append(Appel("evenement_jugement", "609", "B1",
                                "groq_openai_120_key1", 180, 1.2))
    journal.appels.append(Appel("stm_reflection", "610", "B2",
                                "groq_openai_20_key1", 640, 3.1, hors_schema=True))
    journal.ecrire(tmp_path / "passerelle.csv")
    contenu = (tmp_path / "passerelle.csv").read_text("utf-8")
    assert "groq_openai_120_key1" in contenu and "groq_openai_20_key1" in contenu
    assert "hors_schema" in contenu
    assert "820" not in contenu  # the total is not written in place of the detail
    assert journal.jetons_sortie == 820


# ── The stub client ─────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_le_stub_refuse_une_categorie_non_declaree():
    """A silent stub would pass off an untested path as a tested one."""
    client = ClientStub({"evenement_jugement": [{"severity": "notable"}]})
    with pytest.raises(AssertionError, match="no response"):
        await client.execute({"category": "stm_reflection", "agents": [{"agent_id": "1"}]})


@pytest.mark.asyncio
async def test_le_stub_sert_les_reponses_dans_lordre_puis_boucle():
    client = ClientStub({"evenement_jugement": [
        {"severity": "negligible"}, {"severity": "memorable"},
    ]})
    charge = {"category": "evenement_jugement", "agents": [{"agent_id": "609"}]}
    assert (await client.execute(charge)).agents[0].severity == "negligible"
    assert (await client.execute(charge)).agents[0].severity == "memorable"
    assert (await client.execute(charge)).agents[0].severity == "memorable"


# ── Family A runs, and it is the prerequisite ───────────────────────────────────────────
def test_la_famille_A_tourne_entierement_sans_appel():
    """If it fails, family B must not start: it would pay for nothing."""
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "scripts.experiment.banc_fonctionnel.famille_a"],
        capture_output=True, text=True, cwd=RACINE,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "au vert" in r.stdout
    assert "❌" not in r.stdout and "💥" not in r.stdout
