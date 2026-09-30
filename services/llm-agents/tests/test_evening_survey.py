"""The evening survey, aligned with Adam & Gaudou (2025).

Why this file exists. The survey had existed for a long time and had never run: the
four `EXPERIMENT_*` variables were not declared in compose. But the pass-through was not
enough — the perception served to the model fitted in four fields (name, age, gender,
occupation). As written, the probe measured the prior of the base model about a
53-year-old part-time woman: identical on day 12 and on day 29, identical in both arms. It
would have detected nothing, and its silence would have been taken for an absence of effect.

Two distinct requirements are tested here: FIDELITY on input (section D) and SEALING
on output (section E). The module only carried the second one.
"""

import asyncio
import csv
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import noyau as noyau_module
from llm.memory import MemoryEntry, MemoryType
from llm.noyau import memoire_noyau, noter_trajet
from settings import settings
from sim_clock import wall_clock
from urban_mobility_agents import enquetes

T0 = 1773637200


# ── A paper LLM agent, carrying just what the probe needs ────────────


class _MemoireFactice:
    def __init__(self, entrees, journal):
        self.user_metadata = {"899549": {"entries": entrees}}
        self._journal = journal
        self.rappels = 0  # counts the reinforcements: must stay at 0

    def journal_trajets(self, person_id):
        return self._journal


class _ClientFactice:
    """Notes each call and returns six scores. The provider is fixed and known to the tests."""

    def __init__(self, scores=None, echoue_sur=None, vide_sur=None):
        self.appels: list[dict] = []
        self._scores = scores or dict.fromkeys(enquetes.CRITERES, 5)
        self._echoue_sur = echoue_sur or set()
        self._vide_sur = vide_sur or set()

    async def execute(self, payload):
        self.appels.append(payload)
        mode = payload["agents"][0].get("mode_interroge")
        if mode in self._echoue_sur:
            raise RuntimeError("passerelle indisponible")
        if mode in self._vide_sur:
            return SimpleNamespace(agents=[], provider_used="google_gemini31_key1")
        return SimpleNamespace(
            agents=[
                SimpleNamespace(
                    agent_id=payload["agents"][0]["agent_id"],
                    scores=dict(self._scores),
                    justification="Une seule phrase, pour l'ensemble.",
                )
            ],
            provider_used="google_gemini31_key1",
        )


def _choc(texte="panne sur voie rapide", gravite=0.70, jours=1.0):
    return MemoryEntry(
        content=texte,
        timestamp=wall_clock(T0) - timedelta(days=jours),
        memory_type=MemoryType.REFLECTION,
        person_id="899549",
        importance=gravite,
        force=14.56,
    )


def _agent(entrees=None, journal=None, client=None, recit="Corinne, 53 ans, temps partiel."):
    entrees = entrees if entrees is not None else [_choc()]
    journal = journal if journal is not None else {}
    memoire = _MemoireFactice(entrees, journal)
    return SimpleNamespace(
        llm_client=client or _ClientFactice(),
        long_term_memory=memoire,
        get_person_identity_description=lambda person: recit,
    )


def _personne(pid="899549"):
    return SimpleNamespace(person_id=pid, identity=SimpleNamespace(name="Corinne"))


def _jouer(agent, dossier, jour=17, personnes=None):
    return asyncio.run(
        enquetes.executer_enquetes_jalon(
            jour, T0, personnes or [_personne()], agent, Path(dossier)
        )
    )


@pytest.fixture(autouse=True)
def _propre(monkeypatch):
    enquetes.reinitialiser()
    noyau_module.reinitialiser()
    monkeypatch.delenv("EXPERIMENT_SURVEY_MODES", raising=False)
    monkeypatch.delenv("EXPERIMENT_TARGET_PERSONAS", raising=False)
    yield
    enquetes.reinitialiser()
    noyau_module.reinitialiser()


@pytest.fixture
def journal_logs():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(
        lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO"
    )
    yield lignes
    logger.remove(sink)


def _lire(dossier) -> list[dict]:
    fichier = Path(dossier) / "affinites_declarees.csv"
    if not fichier.is_file():
        return []
    with open(fichier, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ══════════════════════ D — fidélité en entrée ══════════════════════════════════


def test_D1_la_perception_porte_les_trois_blocs_de_la_memoire_noyau():
    """D1 — without them, the probe questions the base model about an age and an occupation."""
    journal = {}
    for _ in range(4):
        noter_trajet(journal, "work", "matin", "car")
    agent = _agent(entrees=[_choc()], journal=journal)
    perception = enquetes.perception_de(agent, _personne(), T0)
    assert "My habits" in perception
    assert "What changed recently" in perception
    assert "panne sur voie rapide" in perception


def test_D2_la_perception_porte_le_recit_d_identite_complet():
    """D2 — the SAME as the one of the decision prompt, not a parallel summary."""
    perception = enquetes.perception_de(_agent(recit="Corinne, 53 ans, deux véhicules."), _personne(), T0)
    assert "Corinne, 53 ans, deux véhicules." in perception


def test_D3_un_bloc_memoire_vide_ne_fabrique_pas_de_titre_creux():
    """D3 — a title without content tells the model there should be something."""
    perception = enquetes.perception_de(_agent(entrees=[], journal={}), _personne(), T0)
    assert "What changed recently" not in perception
    assert "My habits" not in perception


def test_D4_la_sonde_voit_passer_le_temps():
    """D4 — D17 and D40 do not return the same perception: that is the whole point of the batch.

    On day 17 the shock memory is in the block; 30 days later, its derived duration
    (15.29 d at severity 0.70) has taken it out.
    """
    agent = _agent(entrees=[_choc(jours=1.0)])
    tot = enquetes.perception_de(agent, _personne(), T0)
    tard = enquetes.perception_de(agent, _personne(), T0 + 30 * 86400)
    assert "panne sur voie rapide" in tot
    assert "panne sur voie rapide" not in tard


def test_D5_le_bloc_est_celui_du_prompt_de_decision_caractere_pour_caractere():
    """D5 — a « lighter » variant would diverge silently, and we would no longer know what to compare."""
    journal = {}
    for _ in range(4):
        noter_trajet(journal, "work", "matin", "car")
    entrees = [_choc()]
    agent = _agent(entrees=entrees, journal=journal)
    attendu = "\n".join(memoire_noyau(journal, entrees, wall_clock(T0), "899549"))
    assert attendu in enquetes.perception_de(agent, _personne(), T0)


# ══════════════════════ E — étanchéité en sortie ════════════════════════════════


def test_E3_l_enquete_n_est_pas_un_rappel(tmp_path):
    """E3 — a probe that strengthens the force extends the lifetime of what it observes."""
    souvenir = _choc()
    force_avant, rappel_avant = souvenir.force, souvenir.dernier_rappel
    agent = _agent(entrees=[souvenir])
    _jouer(agent, tmp_path)
    assert souvenir.force == force_avant
    assert souvenir.dernier_rappel == rappel_avant
    assert souvenir.rappels == 0


def test_E2_aucune_entree_n_est_ajoutee_en_memoire(tmp_path):
    """E2 — la mémoire longue compte autant d'entrées après qu'avant."""
    entrees = [_choc()]
    agent = _agent(entrees=entrees)
    _jouer(agent, tmp_path)
    assert agent.long_term_memory.user_metadata["899549"]["entries"] == entrees
    assert len(entrees) == 1


def test_E5_la_reponse_n_atteint_que_le_csv(tmp_path):
    """E5 — a single file written, and it is this one."""
    _jouer(_agent(), tmp_path)
    ecrits = sorted(p.name for p in Path(tmp_path).iterdir())
    assert ecrits == ["affinites_declarees.csv"]


# ══════════════════════ F — the five prompts and their output ═════════════════════


def test_F1_un_jalon_declenche_cinq_appels(tmp_path):
    """F1 — four modes plus the priorities."""
    client = _ClientFactice()
    _jouer(_agent(client=client), tmp_path)
    assert len(client.appels) == 5


def test_F2_un_prompt_de_mode_ne_cite_aucun_autre_mode(tmp_path):
    """F2 — named alone, the mode is rated in ABSOLUTE terms; the 24 scores at once gave a flat grid."""
    client = _ClientFactice()
    _jouer(_agent(client=client), tmp_path)
    interroges = [a["agents"][0]["mode_interroge"] for a in client.appels]
    assert interroges == [
        "the car", "public transport", "the bicycle", "walking", None
    ]
    assert len(set(interroges[:4])) == 4


def test_F3_le_prompt_de_priorites_ne_nomme_aucun_mode(tmp_path):
    """F3 — c'est le second vecteur d'Adam & Gaudou, indépendant de tout mode."""
    client = _ClientFactice()
    _jouer(_agent(client=client), tmp_path)
    assert client.appels[-1]["agents"][0]["mode_interroge"] is None


def test_F6_les_modes_sont_declares(monkeypatch, tmp_path):
    """F6 — hard-coded, they would make the instrument blind to shocks C4 and C5."""
    monkeypatch.setenv("EXPERIMENT_SURVEY_MODES", "voiture,train")
    enquetes.reinitialiser()
    client = _ClientFactice()
    _jouer(_agent(client=client), tmp_path)
    assert len(client.appels) == 3  # deux modes + priorités
    assert client.appels[1]["agents"][0]["mode_interroge"] == "the train"


def test_F6b_un_mode_inconnu_est_ecarte_et_signale(monkeypatch, journal_logs):
    """F6b — a mode without a label would ask six questions into the void."""
    monkeypatch.setenv("EXPERIMENT_SURVEY_MODES", "voiture,trottinette")
    enquetes.reinitialiser()
    assert enquetes.modes_interroges() == ("voiture",)
    assert [m for n, m in journal_logs if n == "ERROR" and "trottinette" in m]


def test_F7_le_csv_est_en_format_long(tmp_path):
    """F7 — the wide format with 24 columns forced rewriting the header at each mode added."""
    _jouer(_agent(), tmp_path)
    lignes = _lire(tmp_path)
    assert len(lignes) == 5 * len(enquetes.CRITERES)
    assert set(lignes[0]) == set(enquetes.COLONNES_CSV)
    voiture = [l for l in lignes if l["mode"] == "voiture"]
    assert sorted(l["critere"] for l in voiture) == sorted(enquetes.CRITERES)


def test_F8_les_priorites_sont_dans_le_meme_fichier(tmp_path):
    """F8 — a second file would force bringing them together to compute the score."""
    _jouer(_agent(), tmp_path)
    prio = [l for l in _lire(tmp_path) if l["mode"] == enquetes.MODE_PRIORITES]
    assert len(prio) == len(enquetes.CRITERES)


def test_F9_chaque_ligne_porte_le_fournisseur_et_le_modele(tmp_path):
    """F9 — a decision read without its model forces cross-checking two files."""
    lignes = _jouer(_agent(), tmp_path) or _lire(tmp_path)
    for ligne in _lire(tmp_path):
        assert ligne["provider"] == "google_gemini31_key1"
        assert ligne["model"] and ligne["model"] != ""


def test_F10_un_score_hors_domaine_est_alarme_et_ecrit_tel_quel(tmp_path, journal_logs):
    """F10 — clipping a 14 to 10 would fabricate a maximal opinion where the scale was not followed."""
    scores = dict.fromkeys(enquetes.CRITERES, 5) | {"securite": 14}
    _jouer(_agent(client=_ClientFactice(scores=scores)), tmp_path)
    alarmes = [m for n, m in journal_logs if n == "ERROR" and "hors du domaine" in m]
    assert alarmes
    valeurs = {l["score"] for l in _lire(tmp_path) if l["critere"] == "securite"}
    assert valeurs == {"14"}


def test_F11_un_jalon_muet_leve_une_alarme(tmp_path, journal_logs):
    """F11 — a silence would read as « nothing moved », which is the worst of reports."""
    tous = {"the car", "public transport", "the bicycle", "walking", None}
    _jouer(_agent(client=_ClientFactice(vide_sur=tous)), tmp_path)
    assert [m for n, m in journal_logs if n == "ERROR" and "INCOMPLET : 5/5" in m]
    assert _lire(tmp_path) == []


def test_F12_un_appel_en_echec_met_le_jalon_en_attente(tmp_path, journal_logs):
    """F12, revised by ticket 118 (O2) — a milestone is complete or is not written.

    Before: the four other answers were written, and the fifth was lost for good.
    Now nothing enters the CSV; the answers obtained wait with the photographed
    perception, so that only the missing question is paid for again.
    """
    import json as _json

    _jouer(_agent(client=_ClientFactice(echoue_sur={"the bicycle"})), tmp_path)
    assert _lire(tmp_path) == []
    assert [m for n, m in journal_logs if n == "ERROR" and "INCOMPLET" in m]
    attente = _json.loads(next(tmp_path.glob("enquete_en_attente_J*.json")).read_text("utf-8"))
    assert [m for _p, m in attente["manquantes"]] == ["velo"]
    assert {l["mode"] for l in attente["lignes"]} == {
        "voiture", "transports_collectifs", "marche", enquetes.MODE_PRIORITES
    }
    assert set(attente["perceptions"]) == {p for p, _m in attente["manquantes"]}


def test_F13_la_formule_de_score_se_calcule_depuis_le_seul_csv(tmp_path):
    """F13 — `score(mode) = Σ val(mode, critère) × prio(critère)`, sans autre source."""
    scores = {"rapidite": 8, "praticite": 7, "confort": 6, "securite": 5, "cout": 4, "ecologie": 3}
    _jouer(_agent(client=_ClientFactice(scores=scores)), tmp_path)
    lignes = _lire(tmp_path)
    prio = {
        l["critere"]: int(l["score"])
        for l in lignes
        if l["mode"] == enquetes.MODE_PRIORITES
    }
    val = {
        l["critere"]: int(l["score"]) for l in lignes if l["mode"] == "voiture"
    }
    assert sum(val[c] * prio[c] for c in enquetes.CRITERES) == sum(
        scores[c] * scores[c] for c in enquetes.CRITERES
    )


def test_E1_E4_frontiere_le_module_n_importe_rien_qui_ecrive_la_memoire():
    """E1 and E4 — sealing is proven on the IMPORTS, not on an intention.

    A test that checks « the STM has not moved » on a paper agent proves nothing: it is
    the paper agent that has no STM. What can be proven is that the module has no means
    of writing — neither short-term memory, nor long-term memory, nor ChromaDB, nor memory journal.

    `memoire_noyau` is the only door open onto the memory, and it only READS.
    """
    import ast

    source = Path(enquetes.__file__).read_text(encoding="utf-8")
    arbre = ast.parse(source)
    importes = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.ImportFrom) and noeud.module:
            importes.add(noeud.module)
        elif isinstance(noeud, ast.Import):
            importes.update(a.name for a in noeud.names)

    interdits = {
        "llm.shortterm", "llm.longterm", "llm.memory", "llm.journal_memoire",
        "llm.concepts", "chromadb",
    }
    assert not (importes & interdits), (
        "le module d'enquête importe de quoi écrire la mémoire : "
        + ", ".join(sorted(importes & interdits))
    )
    assert "llm.noyau" in importes, (
        "la mémoire noyau est la porte d'ENTRÉE de la sonde : sans elle, l'enquête mesure le "
        "modèle de base."
    )
    # No memory write, not even through an attribute of an object passed as a parameter. The
    # search bears on the CALLS of the syntax tree and not on the text: the module
    # NAMES `force_apres_rappel` in its docstring, precisely to say that it does not call it.
    appeles = {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(arbre)
        if isinstance(n, ast.Call)
    }
    interdits_appels = {"aadd_memory", "add_message", "remove_batch", "force_apres_rappel"}
    assert not (appeles & interdits_appels), (
        "le module appelle " + ", ".join(sorted(appeles & interdits_appels))
        + " : l'étanchéité est rompue."
    )
