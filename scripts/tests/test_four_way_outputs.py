"""The single figure and the four-way table.

Both scripts are written BEFORE the first run, on shape data: a figure whose
script only exists after the measurement gets tailored to what it found.
"""

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.analysis import figure_evenement as fig
from scripts.analysis import tableau_quatre_voies as quatre


TEXTE = "The engine made a grinding noise and the car stalled on the expressway."

# Instants, in simulated UTC seconds as in the real files: `evenements.jsonl`
# carries `timestamp`, each exchange `sim_ts` and `sim_day`. An undated decision is dropped
# from the table — it falls neither before nor after the exposure.
EXPOSITION = 1774483200            # 2026-03-26T00:00:00, wake-up on the reading day
APRES = EXPOSITION + 21438         # 2026-03-26 05:57:18, the first departure after it
AVANT = APRES - 86400              # the day before, same time
DATEE = {"sim_ts": APRES, "sim_day": "2026-03-26"}


def _moves(dossier: Path, lignes: list[dict], avec_role: bool = True) -> Path:
    entetes = ["Référence", "ID Personne", "Heure de départ", "P(Voiture Privée) %", "Choc",
               "Jour relatif au choc"] + (["Rôle"] if avec_role else [])
    chemin = dossier / "moves.csv"
    with chemin.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=entetes, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)
    return chemin


def _jeu_de_forme(dossier: Path, effectif: int = 6) -> Path:
    lignes = []
    for relatif in range(-3, 8):
        for role, base, creux in (("expose", 60, 25), ("co_resident", 60, 6), ("temoin", 60, 0)):
            for i in range(effectif):
                lignes.append({
                    "Référence": f"{role}_{i}", "Heure de départ": "2026-03-16 08:00",
                    "P(Voiture Privée) %": base - (creux if 0 <= relatif <= 5 else 0),
                    "Choc": "c6_voiture_suspecte", "Jour relatif au choc": relatif,
                    "Rôle": role,
                })
    return _moves(dossier, lignes)


# ── R47, R49. The vacuity guard ──────────────────────────────────────────────────────────
def test_R49_sous_leffectif_minimal_la_figure_sort_non_concluant(tmp_path):
    """A curve drawn on two decisions looks like a curve drawn on two hundred."""
    _jeu_de_forme(tmp_path, effectif=fig.EFFECTIF_MINIMAL - 1)
    moyennes, _effectifs, _ev = fig.series(tmp_path, "car")
    assert all(not points for points in moyennes.values())
    with pytest.raises(SystemExit, match="inconclusive"):
        fig.composer(moyennes, {}, "car", "c6")


def test_R47_un_role_absent_est_ECRIT_sur_la_figure(tmp_path):
    """A missing series reads as a null effect to whoever does not know it is missing."""
    lignes = []
    for relatif in range(-2, 5):
        for i in range(5):
            lignes.append({
                "Référence": f"e{i}", "Heure de départ": "2026-03-16 08:00",
                "P(Voiture Privée) %": 55, "Choc": "c6", "Jour relatif au choc": relatif,
                "Rôle": "expose",
            })
    _moves(tmp_path, lignes)
    moyennes, effectifs, ev = fig.series(tmp_path, "car")
    svg = fig.composer(moyennes, effectifs, "car", ev)
    assert "not conclusive for" in svg
    assert "Co-resident" in svg and "Control" in svg


def test_une_cellule_vide_nest_pas_un_zero(tmp_path):
    """A decision without a distribution is not a decision at 0 % car."""
    lignes = [
        {"Référence": "a", "Heure de départ": "2026-03-16 08:00", "P(Voiture Privée) %": "",
         "Choc": "c6", "Jour relatif au choc": 0, "Rôle": "expose"},
    ] + [
        {"Référence": f"b{i}", "Heure de départ": "2026-03-16 08:00",
         "P(Voiture Privée) %": 50, "Choc": "c6", "Jour relatif au choc": 0, "Rôle": "expose"}
        for i in range(3)
    ]
    _moves(tmp_path, lignes)
    moyennes, effectifs, _ = fig.series(tmp_path, "car")
    assert effectifs["expose"][0] == 3, "the empty row does not count in the sample size"
    assert moyennes["expose"][0] == 50.0, "it does not pull the mean down either"


# ── R50. The x-axis exists every day ─────────────────────────────────────────────────────
def test_R50_le_jour_relatif_couvre_avant_et_apres(tmp_path):
    _jeu_de_forme(tmp_path)
    moyennes, _e, _ev = fig.series(tmp_path, "car")
    jours = sorted(moyennes["temoin"])
    assert min(jours) < 0 < max(jours), (
        "an x-axis that existed only on event days would plot nothing"
    )


def test_la_figure_se_compose_et_nomme_le_resultat(tmp_path):
    _jeu_de_forme(tmp_path)
    moyennes, effectifs, ev = fig.series(tmp_path, "car")
    svg = fig.composer(moyennes, effectifs, "car", ev)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "Who changes, and for how long" in svg
    for _role, nom, *_style in fig.ROLES:
        assert nom in svg
    # Figure in ENGLISH, legend included: all of the article's figures are.
    for mot_francais in ("exposé", "témoin", "jour relatif"):
        assert mot_francais not in svg


def test_un_run_sans_colonne_role_le_dit_au_lieu_de_deviner(tmp_path):
    _moves(tmp_path, [{"Référence": "a", "Heure de départ": "2026-03-16 08:00",
                       "P(Voiture Privée) %": 50, "Choc": "c6",
                       "Jour relatif au choc": 0}], avec_role=False)
    with pytest.raises(SystemExit, match="Rôle"):
        fig.series(tmp_path, "car")


def test_un_mode_inconnu_est_refuse_et_non_devine():
    assert "teleportation" not in fig.COLONNE_PROBA
    assert set(fig.COLONNE_PROBA) == set(fig.LIBELLE_EN)


# ── The four-way table ───────────────────────────────────────────────────────────────────
def test_les_en_tetes_de_bloc_sont_ceux_de_noyau_py():
    """A renamed header would output a way at zero without any error showing."""
    noyau = (RACINE / "services" / "llm-agents" / "llm" / "noyau.py").read_text("utf-8")
    for actuel, _archive in quatre.EN_TETES.values():
        assert f'"{actuel}"' in noyau, f"'{actuel}' is no longer set by noyau.py"


def test_les_quatre_voies_sont_celles_du_manuscrit():
    assert quatre.VOIES == ("habitudes", "connaissances", "changements", "rappel")


def test_les_mots_saillants_ecartent_le_banal_et_gardent_le_rare():
    saillants = quatre.mots_saillants(TEXTE)
    assert "grinding" in saillants and "expressway" in saillants
    assert "the" not in saillants and "and" not in saillants


def test_un_run_sans_evenement_le_dit_au_lieu_de_rendre_un_tableau_vide(tmp_path):
    with pytest.raises(SystemExit, match="run without an event"):
        quatre.depouiller(tmp_path)


def _run_complet(dossier: Path) -> Path:
    (dossier / "evenements.jsonl").write_text(json.dumps({
        "person_id": "899549", "canal": "vecu", "texte": TEXTE, "evenement_id": "c6",
        "timestamp": EXPOSITION, "horodatage_simule": "2026-03-26T00:00:00",
    }) + "\n", encoding="utf-8")
    # As in the real log: `Référence` carries the run name, the agent is in `ID Personne`.
    _moves(dossier, [{"Référence": "2026-09-24_17_50", "ID Personne": "899549",
                      "Heure de départ": "2026-03-16 08:00",
                      "P(Voiture Privée) %": 30, "Choc": "c6",
                      "Jour relatif au choc": 1, "Rôle": "expose"}])
    return dossier


def test_une_decision_sans_aucune_voie_est_NOMMEE_et_non_lue_comme_un_zero(tmp_path):
    """'We cannot tell' is not 'the event weighed on nothing'."""
    _run_complet(tmp_path)
    (tmp_path / "llm_exchanges.jsonl").write_text(json.dumps({
        "category": "itinary_multi_agent", **DATEE,
        "messages": "--- agent_id=899549 ---\nMes habitudes\n- rien de notable",
    }) + "\n", encoding="utf-8")
    resultat = quatre.depouiller(tmp_path)
    assert resultat["decisions"][("vecu", "expose")] == 1
    assert not any(resultat["compte"].get(("vecu", "expose", v)) for v in quatre.VOIES)
    rendu = quatre.rendre(resultat)
    assert "AUCUNE voie" in rendu
    assert "n'a pesé sur rien" in rendu, "the output must SAY what it cannot tell"
    assert "absence de mesure" in rendu


def test_un_run_dont_aucune_decision_ne_voit_lagent_sort_NON_CONCLUANT(tmp_path):
    _run_complet(tmp_path)
    (tmp_path / "llm_exchanges.jsonl").write_text(json.dumps({
        "category": "itinary_multi_agent", "messages": "--- agent_id=999 ---\nMes habitudes",
    }) + "\n", encoding="utf-8")
    assert "non concluant" in quatre.rendre(quatre.depouiller(tmp_path))


def test_le_texte_retrouve_dans_un_bloc_est_impute_a_CE_bloc(tmp_path):
    _run_complet(tmp_path)
    prompt = (
        "--- agent_id=899549 ---\n"
        "Mes habitudes\n- souvent la voiture\n"
        f"Ce qui a changé récemment\n- {TEXTE}\n"
    )
    (tmp_path / "llm_exchanges.jsonl").write_text(json.dumps({
        "category": "itinary_multi_agent", **DATEE, "messages": prompt,
    }) + "\n", encoding="utf-8")
    resultat = quatre.depouiller(tmp_path)
    assert resultat["compte"][("vecu", "expose", "changements")]["exact"] == 1
    assert not resultat["compte"].get(("vecu", "expose", "habitudes"))


def test_lappariement_par_mots_saillants_est_marque_comme_indicatif(tmp_path):
    """The reflection REPHRASES the lived event: exact matching fails, and that must be said."""
    _run_complet(tmp_path)
    reformule = "my car started making a terrible grinding noise on the expressway"
    (tmp_path / "llm_exchanges.jsonl").write_text(json.dumps({
        "category": "itinary_multi_agent", **DATEE,
        "messages": f"--- agent_id=899549 ---\nCe que je sais\n- {reformule}\n",
    }) + "\n", encoding="utf-8")
    resultat = quatre.depouiller(tmp_path)
    assert resultat["compte"][("vecu", "expose", "connaissances")]["saillant"] == 1
    assert "~" in quatre.rendre(resultat)


def test_une_reflexion_nocturne_nest_pas_une_decision(tmp_path):
    _run_complet(tmp_path)
    (tmp_path / "llm_exchanges.jsonl").write_text(json.dumps({
        "category": "stm_reflection",
        "messages": f"--- agent_id=899549 ---\nCe que je sais\n- {TEXTE}",
    }) + "\n", encoding="utf-8")
    assert not quatre.depouiller(tmp_path)["decisions"]


def test_le_role_se_lit_dans_ID_Personne_et_non_dans_Reference(tmp_path):
    """`Référence` carries the run name (`experiences/journal.py`). Read as an agent
    identifier, it output role `?` on every row of run 2026-09-24_17_50."""
    _moves(tmp_path, [
        {"Référence": "2026-09-24_17_50", "ID Personne": "286920", "Rôle": "expose"},
        {"Référence": "2026-09-24_17_50", "ID Personne": "286921", "Rôle": "co_resident"},
    ])
    assert quatre.roles_du_run(tmp_path) == {"286920": "expose", "286921": "co_resident"}


# ── Exchanges as the gateway writes them ─────────────────────────────────────────────────
def _echanges_passerelle(dossier: Path, objets: list[dict], separateur: str = "\n") -> None:
    """One INDENTED JSON object per exchange, as `llm_gateway/telemetry/exchanges.py` does: this
    is not JSONL despite the extension."""
    (dossier / "llm_exchanges.jsonl").write_text(
        "".join(json.dumps(o, ensure_ascii=False, indent=2) + separateur for o in objets),
        encoding="utf-8",
    )


def _echanges_de_forme(origine: str) -> list[dict]:
    """`origine` is the name of the run that signed the exchange: `lire_echanges` drops the others."""
    return [
        {"origine": origine, "category": "itinary_multi_agent", "task_id": "t1", **DATEE,
         "messages": [
             {"role": "system", "content": "Select the optimal travel mode."},
             {"role": "user", "content": "--- agent_id=899549 ---\nMes habitudes\n- la voiture\n"
                                         f"Ce qui a changé récemment\n- {TEXTE}\n"},
         ],
         # A list of strings: indented, each one sits alone on its line and decodes,
         # read line by line, as a bare string — that is what broke `.get()`.
         "response": {"modes_touches": ["walking", "car"]}},
        {"origine": origine, "category": "stm_reflection", "task_id": "t2", **DATEE,
         "messages": [{"role": "user", "content": f"--- agent_id=899549 ---\n{TEXTE}"}],
         "response": ["walking"]},
    ]


@pytest.mark.parametrize("separateur", ["\n\n", "\n"], ids=["ligne_vide", "saut_de_ligne"])
def test_les_echanges_indentes_de_la_passerelle_se_lisent(tmp_path, separateur):
    """The gateway docstring announces a blank line between objects; run
    2026-09-24_17_50 has none. Both are read."""
    _run_complet(tmp_path)
    _echanges_passerelle(tmp_path, _echanges_de_forme(tmp_path.name), separateur)
    resultat = quatre.depouiller(tmp_path)
    assert resultat["echanges"] == 2
    assert resultat["decisions"][("vecu", "expose")] == 1
    assert resultat["compte"][("vecu", "expose", "changements")]["exact"] == 1
    assert not resultat["compte"].get(("vecu", "expose", "habitudes"))
    assert "Lu : 2 échanges, dont 1 décision" in quatre.rendre(resultat)


def test_un_echange_signe_par_un_autre_run_nest_pas_compte(tmp_path):
    """The worker writes this log for ALL its clients: a decision signed by another run
    says nothing about this one, even if it carries the same `agent_id`."""
    _run_complet(tmp_path)
    _echanges_passerelle(tmp_path, _echanges_de_forme("un_autre_run")
                         + _echanges_de_forme(tmp_path.name)[:1])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["echanges"] == 1
    assert resultat["decisions"][("vecu", "expose")] == 1


@pytest.mark.parametrize("contenu", [None, ""], ids=["absent", "vide"])
def test_un_run_sans_echanges_le_dit_au_lieu_de_conclure_a_labsence_de_decision(tmp_path, contenu):
    """With no exchange read, `aucune décision ne porte le texte` would be false: no decision
    was read at all. The table is still output (A9.3 of the bench), it does not raise."""
    _run_complet(tmp_path)
    if contenu is not None:
        (tmp_path / "llm_exchanges.jsonl").write_text(contenu, encoding="utf-8")
    rendu = quatre.rendre(quatre.depouiller(tmp_path))
    assert "non concluant" in rendu and "n'a pesé sur rien" in rendu
    assert "aucun échange lu dans `llm_exchanges.jsonl`" in rendu
    assert "aucune décision ne porte le texte" not in rendu


def test_le_tableau_se_lance_en_ligne_de_commande_sur_des_echanges_indentes(tmp_path):
    """Launched by its path, outside the root: the import of `memoire.sources` must hold."""
    _run_complet(tmp_path)
    _echanges_passerelle(tmp_path, _echanges_de_forme(tmp_path.name))
    r = subprocess.run(
        [sys.executable, str(RACINE / "scripts" / "analysis" / "tableau_quatre_voies.py"),
         str(tmp_path), "--markdown"],
        capture_output=True, text=True, cwd=tmp_path, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "expose" in r.stdout and "Lu : 2 échanges" in r.stdout


# ── Recall, date, raw content — on the shape of run 2026-09-24_17_50 ─────────────────────
# Three defects found on 2026-09-25 on this run: the `rappel` column counted every decision
# of an agent that had been served ANY memory (15 out of 15); 11 of these 15 decisions
# preceded the reading; and an article's text, searched in `json.dumps(messages)`, could
# never be found there — line breaks and quotes are escaped in it.
LECTEUR = "286920"
ARTICLE = (
    "(Translated from French)\nGusts above 80 km/h: Toulouse closes its parks and gardens at "
    "short notice this Thursday evening\n\n\"A weather alert forecasts violent winds,\" the "
    "municipality states on its website."
)
# The entry the reading puts in memory (`llm/evenements/injection.py`), of type
# `conversation`: `llm_agent.py` only formats recalled reflections and concepts.
SOUVENIR_LU = f"[ PRESSE ] I read in the paper: « {ARTICLE} »"
# A reflection that QUOTES the article: exact link, and it can reach the prompt itself.
REFLEXION_CITANT = f"This morning the paper said: {ARTICLE} I kept driving."


def test_les_en_tetes_anglais_s_imputent_comme_les_francais(tmp_path):
    """2026-09-25 — the block speaks English; older archives keep their French titles."""
    _run_complet(tmp_path)
    prompt = (
        "--- agent_id=899549 ---\n"
        "My habits\n- often the car\n"
        f"What changed recently\n- {TEXTE}\n"
    )
    (tmp_path / "llm_exchanges.jsonl").write_text(json.dumps({
        "category": "itinary_multi_agent", **DATEE, "messages": prompt,
    }) + "\n", encoding="utf-8")
    resultat = quatre.depouiller(tmp_path)
    assert resultat["compte"][("vecu", "expose", "changements")]["exact"] == 1
    assert not resultat["compte"].get(("vecu", "expose", "habitudes"))


def _jour(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _section(depart: int, *, agent: str = LECTEUR, changements: str = "",
             rappel: tuple[str, ...] = ()) -> str:
    """An agent's section, composed as `llm_agent.py` composes it: the header and its departure
    time, the core blocks prefixed by the template, then the recalled memories."""
    heure = datetime.fromtimestamp(depart, tz=timezone.utc).strftime("%H:%M")
    lignes = [f"--- agent_id={agent} | Destination: work (Toulouse) | Departure: {heure} ---",
              "**History:**",
              "- Mes habitudes", "- - work le matin : en voiture, 8 fois sur 8",
              "- Ce que je sais", "- - Driving to work is fast and reliable.  (6 obs.)"]
    if changements:
        lignes += ["- Ce qui a changé récemment", f"- - {changements}"]
    lignes += [f"- {r}" for r in rappel]
    return "\n".join(lignes) + "\n\n"


def _run_lu(dossier: Path, *, souvenirs=(), traces=(), decisions=(), exposition=True) -> Path:
    """A reader exposed at wake-up on March 26.

    `souvenirs`: `(doc_id, memory_type, content, timestamp)`, in long-term memory;
    `traces`: `(sim_ts, [doc_id, …])`, in `trace_rappel.jsonl`;
    `decisions`: `(sim_ts, contenu_user)`, one decision prompt each.
    """
    evenement = {"person_id": LECTEUR, "canal": "lu", "moment": "reveil", "texte": ARTICLE,
                 "evenement_id": "a09_vent_autan"}
    if exposition:
        evenement |= {"timestamp": EXPOSITION, "horodatage_simule": "2026-03-26T00:00:00"}
    (dossier / "evenements.jsonl").write_text(json.dumps(evenement) + "\n", encoding="utf-8")
    _moves(dossier, [{"Référence": dossier.name, "ID Personne": LECTEUR, "Rôle": "expose"}])
    shard = dossier / "long_term_memory" / "user_metadata" / "shard_84"
    shard.mkdir(parents=True)
    (shard / f"{LECTEUR}.json").write_text(json.dumps({
        "person_id": LECTEUR,
        "entries": [{"doc_id": d, "memory_type": t, "content": c, "timestamp": ts,
                     "person_id": LECTEUR} for d, t, c, ts in souvenirs],
    }), encoding="utf-8")
    (dossier / "trace_rappel.jsonl").write_text("".join(
        json.dumps({"sim_ts": ts, "sim_day": _jour(ts), "person_id": LECTEUR,
                    "candidats": 20, "concentration": None,
                    "servis": [{"rang": i, "doc_id": d, "type": "reflection", "vivier": "A",
                                "score": 0.5} for i, d in enumerate(docs)]}) + "\n"
        for ts, docs in traces), encoding="utf-8")
    _echanges_passerelle(dossier, [
        {"origine": dossier.name, "category": "itinary_multi_agent",
         **({"sim_ts": ts, "sim_day": _jour(ts)} if ts is not None else {}),
         "messages": [{"role": "system", "content": "Select the optimal travel mode."},
                      {"role": "user", "content": contenu}]}
        for ts, contenu in decisions])
    return dossier


def _ligne(rendu: str, canal: str = "lu") -> list[str]:
    """The NON-EMPTY cells of the channel's row: `[canal, rôle, décisions, …]`."""
    return next(l for l in rendu.splitlines() if l.startswith(canal)).split()


def test_le_rappel_ne_compte_que_le_souvenir_NE_DE_LEVENEMENT(tmp_path):
    """Any served memory is not the recalled event. On run 2026-09-24_17_50,
    the trace serves `286920_19` — a reflection from March 24 — and the old table derived
    "rappel 15" from it. The article's memory, `286920_23`, is served nowhere."""
    _run_lu(tmp_path,
            souvenirs=[("286920_19", "reflection", "Today went smoothly.", "2026-03-24T19:21:39"),
                       ("286920_23", "conversation", SOUVENIR_LU, "2026-03-26T00:00:00")],
            traces=[(APRES, ["286920_19"])],
            decisions=[(APRES, _section(APRES, rappel=("[Tuesday, March 24] Today went smoothly.",)))])
    resultat = quatre.depouiller(tmp_path)
    assert not resultat["compte"].get(("lu", "expose", "rappel"))
    assert resultat["rappel"][("lu", "expose")]["mesurables_exact"] == 1
    rendu = quatre.rendre(resultat)
    # Measured and zero: `0`, not an empty cell — the memory is identified, the trace exists.
    assert _ligne(rendu) == ["lu", "expose", "1", "0"]
    assert "286920_23" in rendu


def test_le_souvenir_servi_ET_present_dans_le_prompt_compte_au_rappel_et_a_lui_seul(tmp_path):
    """The recalled memory appears AFTER the last named block. The old split put it
    in that block — here `Ce que je sais` — and so assigned it to the wrong way."""
    _run_lu(tmp_path,
            souvenirs=[("286920_24", "reflection", REFLEXION_CITANT, "2026-03-26T00:00:00")],
            traces=[(APRES, ["286920_1", "286920_24"])],
            decisions=[(APRES, _section(APRES, rappel=(f"[Thursday, March 26] {REFLEXION_CITANT}",)))])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["compte"][("lu", "expose", "rappel")]["exact"] == 1
    assert not resultat["compte"].get(("lu", "expose", "connaissances"))
    assert _ligne(quatre.rendre(resultat)) == ["lu", "expose", "1", "1"]


def test_un_souvenir_servi_au_top_K_sans_atteindre_le_prompt_nest_pas_compte(tmp_path):
    """The trace lists the top-K (ten); the prompt only receives the three most recent, and
    never a `conversation` entry. Served is not seen."""
    _run_lu(tmp_path,
            souvenirs=[("286920_23", "conversation", SOUVENIR_LU, "2026-03-26T00:00:00")],
            traces=[(APRES, ["286920_23", "286920_19"])],
            decisions=[(APRES, _section(APRES, rappel=("[Tuesday, March 24] Today went smoothly.",)))])
    resultat = quatre.depouiller(tmp_path)
    assert not resultat["compte"].get(("lu", "expose", "rappel"))
    assert resultat["rappel"][("lu", "expose")]["hors_prompt"] == 1
    rendu = quatre.rendre(resultat)
    assert _ligne(rendu) == ["lu", "expose", "1", "0"]
    assert "servi au top-K sans atteindre le prompt : 1" in rendu


def test_sans_trace_appariee_le_rappel_reste_VIDE_et_non_zero(tmp_path):
    """On March 30, the decision of 286920 has no recall trace: we do not know what was
    served, and a `0` would claim it."""
    _run_lu(tmp_path,
            souvenirs=[("286920_23", "conversation", SOUVENIR_LU, "2026-03-26T00:00:00")],
            traces=[],
            decisions=[(APRES, _section(APRES))])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["rappel"][("lu", "expose")]["sans_trace"] == 1
    rendu = quatre.rendre(resultat)
    assert _ligne(rendu) == ["lu", "expose", "1"], "all cells empty"
    assert "sans trace de rappel appariée : 1" in rendu


def test_sans_souvenir_de_levenement_en_memoire_longue_le_rappel_reste_VIDE(tmp_path):
    _run_lu(tmp_path,
            souvenirs=[("286920_19", "reflection", "Today went smoothly.", "2026-03-24T19:21:39")],
            traces=[(APRES, ["286920_19"])],
            decisions=[(APRES, _section(APRES))])
    rendu = quatre.rendre(quatre.depouiller(tmp_path))
    assert _ligne(rendu) == ["lu", "expose", "1"]
    assert "aucun souvenir de l'événement" in rendu


@pytest.mark.parametrize("servi", [True, False], ids=["servi", "non_servi"])
def test_un_lien_seulement_INDICATIF_marque_le_rappel_et_ne_rend_jamais_zero(tmp_path, servi):
    """A rephrased lived event links to its memory only through salient words. A recall found
    this way carries `~`; a recall NOT found proves nothing — another rephrased memory may
    exist that the words do not catch — and the cell stays empty."""
    reformule = "The municipality translated its storm notice for Toulouse residents."
    _run_lu(tmp_path,
            souvenirs=[("286920_24", "reflection", reformule, "2026-03-26T00:00:00")],
            traces=[(APRES, ["286920_24"] if servi else ["286920_19"])],
            decisions=[(APRES, _section(APRES, rappel=(f"[Thursday, March 26] {reformule}",)
                                        if servi else ()))])
    resultat = quatre.depouiller(tmp_path)
    assert _ligne(quatre.rendre(resultat)) == (["lu", "expose", "1", "~1"] if servi
                                              else ["lu", "expose", "1"])


def test_les_decisions_ANTERIEURES_a_lexposition_sont_ecartees_et_comptees(tmp_path):
    """11 of the 15 decisions of 286920 precede the reading of March 26. They cannot
    have seen the article. Nor can a decision taken AT the instant of exposure: wake-up
    precedes the reading."""
    _run_lu(tmp_path, decisions=[(AVANT, _section(AVANT)), (EXPOSITION, _section(EXPOSITION)),
                                 (APRES, _section(APRES))])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["decisions"][("lu", "expose")] == 1
    assert resultat["ecartees"]["anterieures"] == 2
    rendu = quatre.rendre(resultat)
    assert "3 lue(s), 1 postérieure(s) à l'exposition retenue(s), 2 antérieure(s)" in rendu


def test_quand_toutes_les_decisions_precedent_lexposition_le_tableau_le_DIT(tmp_path):
    _run_lu(tmp_path, decisions=[(AVANT, _section(AVANT))])
    rendu = quatre.rendre(quatre.depouiller(tmp_path))
    assert "non concluant" in rendu and "n'a pesé sur rien" in rendu
    assert "aucune décision postérieure à l'exposition" in rendu
    assert "1 antérieure(s)" in rendu


@pytest.mark.parametrize("manque", ["sim_ts", "timestamp"], ids=["decision", "exposition"])
def test_une_decision_NON_DATEE_est_ecartee_et_comptee(tmp_path, manque):
    """Without an instant on one side or the other, the decision falls neither before nor after."""
    _run_lu(tmp_path, exposition=(manque != "timestamp"),
            decisions=[(None if manque == "sim_ts" else APRES, _section(APRES))])
    resultat = quatre.depouiller(tmp_path)
    assert not resultat["decisions"]
    assert resultat["ecartees"]["non_datees"] == 1
    assert "1 non datée(s)" in quatre.rendre(resultat)


def test_linstant_dexposition_se_lit_aussi_dans_horodatage_simule(tmp_path):
    _run_lu(tmp_path, decisions=[(AVANT, _section(AVANT)), (APRES, _section(APRES))])
    ligne = json.loads((tmp_path / "evenements.jsonl").read_text("utf-8"))
    del ligne["timestamp"]
    (tmp_path / "evenements.jsonl").write_text(json.dumps(ligne) + "\n", encoding="utf-8")
    resultat = quatre.depouiller(tmp_path)
    assert resultat["decisions"][("lu", "expose")] == 1
    assert resultat["ecartees"]["anterieures"] == 1


def test_un_article_multiligne_se_retrouve_TEL_QUEL_dans_le_contenu_brut(tmp_path):
    """`json.dumps(messages)` escapes `\\n` and `"`: the start of the article was never
    found there, and exact matching was structurally impossible for the press."""
    _run_lu(tmp_path, decisions=[(APRES, _section(APRES, changements=SOUVENIR_LU))])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["compte"][("lu", "expose", "changements")] == Counter(exact=1)


def test_le_bloc_dun_AUTRE_AGENT_du_meme_lot_nest_pas_impute(tmp_path):
    """A decision prompt groups several agents. Blocks are searched in the agent's
    section, and `agent_id=286920` is not `agent_id=2869201`."""
    voisin = "2869201"
    _run_lu(tmp_path, decisions=[
        (APRES, _section(APRES, agent=voisin, changements=SOUVENIR_LU) + _section(APRES)),
        (APRES + 60, _section(APRES + 60, agent=voisin, changements=SOUVENIR_LU)),
    ])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["decisions"][("lu", "expose")] == 1
    assert not resultat["compte"].get(("lu", "expose", "changements"))


def test_un_rappel_du_souvenir_SANS_DECISION_LUE_est_signale(tmp_path):
    """On run c3 (2026-09-24_00_15), the memory of the breakdown is served only once, on
    March 28 at 05:42 — and no decision prompt is logged that day. This recall is not
    counted, but it is reported: without it, the table would contradict the results log."""
    deux_jours_apres = APRES + 2 * 86400
    _run_lu(tmp_path,
            souvenirs=[("286920_23", "conversation", SOUVENIR_LU, "2026-03-26T00:00:00")],
            traces=[(APRES, ["286920_19"]), (deux_jours_apres, ["286920_23"])],
            decisions=[(APRES, _section(APRES))])
    resultat = quatre.depouiller(tmp_path)
    assert resultat["rappels_sans_decision"] == [(LECTEUR, "2026-03-28 05:57", "286920_23")]
    assert "sans décision lue" in quatre.rendre(resultat)


def test_les_deux_scripts_se_lancent_en_ligne_de_commande(tmp_path):
    _jeu_de_forme(tmp_path)
    sortie = tmp_path / "f.svg"
    r = subprocess.run(
        [sys.executable, str(RACINE / "scripts" / "analysis" / "figure_evenement.py"),
         str(tmp_path), "--mode", "car", "-o", str(sortie)],
        capture_output=True, text=True, cwd=RACINE,
    )
    assert r.returncode == 0, r.stderr
    assert sortie.is_file() and sortie.read_text("utf-8").startswith("<svg")
