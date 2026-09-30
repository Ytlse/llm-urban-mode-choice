"""Annex L: a real, verifiable example of the long-term memory recall score.

Replays, OFFLINE and without any call to the model, the recall served to the decision of
31/03/2026 at 05:42 (persona 861500, day after the car breakdown injected on 30/03 at
05:32) in the traced run `experiments/archive/simulations_septembre_2026/2026-09-21_15_13`.

What the script does, in order:

1. **The memory state at the moment of the recall.** The checkpoint `jour_016` carries
   the label "31/03 03:00", but it was written AFTER the 05:42 decision: the
   decisions are precomputed (the 05:42 decision was made when the simulated clock
   read 30/03 at 21:45). The checkpoint therefore holds the effects of the 05:42 recall and
   of the 07:05 one, plus two entries written by a consolidation that finished just after the
   recall. The script copies this checkpoint into a working directory and brings it back to
   the moment of the recall:
   - entries inserted into Chroma AFTER the end of the query (Chroma `created_at`
     timestamp against `T_ltm_end` from `pipeline_timing.csv`) are removed;
   - reinforcements from later recalls (counter `rappels`, `dernier_rappel`, `force`)
     are undone from `trace_rappel.jsonl`, after checking that this trace explains
     EXACTLY the counter and the last-recall date of each entry of the checkpoint.
2. **The query**, built by the project's own code:
   `LlmAgent.query_past_experiences_for_travel` is called on a minimal agent (prompt
   manager, long-term memory, weather resolution), with the six options read from the actual
   prompt. The query text, the offered modes and the axis context are captured on the way.
3. **The ranking**, by the code's function: `MultiUserLongTermMemory.aquery_user_memories`
   (pools A, B, C, filters, `rank_nodes`, top-K). `rank_nodes` only returns the total score;
   the five components are recomputed with the SAME functions as it
   (`affinite_meteo`, `_time_decay_score`, `importance`, `affinite_axes`, bounded similarity),
   and their weighted sum must equal the score returned by the code.
4. **The checks** against what the run actually did: top-10, scores and number of
   candidates in `trace_rappel.jsonl`; history lines of the real prompt (`llm_exchanges.jsonl`).

Nothing is written into the archived run or into the application code. Any divergence is
logged as ERROR `[ALARME]` and the script exits with code 1.

Usage:
    services/llm-agents/.venv/bin/python scripts/annexes/exemple_L_rappel.py [--garder]
Output: scripts/annexes/data/exemple_L_rappel.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
SA = RACINE / "services" / "llm-agents"
RUN = RACINE / "experiments/archive/simulations_septembre_2026/2026-09-21_15_13"
POINT = RUN / "checkpoints_memoire" / "jour_016"
SORTIE = RACINE / "scripts/annexes/data/exemple_L_rappel.json"

PERSONNE = "861500"
QUERY_AT = 1774935736  # 2026-03-31 05:42:16, GAMA wall-clock time
MOTS_MAX = 25
TOLERANCE_SOMME = 1e-12  # recombination of the five components against the code's score
TOLERANCE_TRACE = 6e-5  # the trace rounds scores to 4 decimals (5e-5), plus a computation margin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exemple_L")

_ALARMES: list[str] = []


def alarme(message: str) -> None:
    _ALARMES.append(message)
    log.error(f"[ALARME] {message}")


class Etape:
    """Start, end and duration of a step, in the log."""

    def __init__(self, nom: str) -> None:
        self.nom = nom

    def __enter__(self):
        self.t0 = time.monotonic()
        log.info(f"▶ {self.nom}")
        return self

    def __exit__(self, *exc):
        log.info(f"■ {self.nom} — {time.monotonic() - self.t0:.2f} s")
        return False


# ── Reading the run ─────────────────────────────────────────────────────────────────────


def lire_jsonl_concatene(chemin: Path):
    """`llm_exchanges.jsonl` concatenates indented JSON objects: we decode them one by one."""
    texte = chemin.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
    while i < len(texte):
        while i < len(texte) and texte[i].isspace():
            i += 1
        if i >= len(texte):
            break
        objet, i = dec.raw_decode(texte, i)
        yield objet


def prompt_de_la_decision() -> str:
    for ech in lire_jsonl_concatene(RUN / "llm_exchanges.jsonl"):
        if ech.get("category") != "itinary_multi_agent":
            continue
        if int(ech.get("sim_ts") or 0) != QUERY_AT:
            continue
        for m in ech["messages"]:
            if m["role"] == "user" and f"agent_id={PERSONNE}" in m["content"]:
                return m["content"]
    raise SystemExit(f"decision {PERSONNE} @ {QUERY_AT} missing from llm_exchanges.jsonl")


def lignes_historique(prompt: str) -> list[str]:
    bloc = prompt.split("**History:**", 1)[1].split("\n\n", 1)[0]
    return [ligne[2:] for ligne in bloc.strip().splitlines() if ligne.startswith("- ")]


def trace_des_rappels() -> list[dict]:
    lignes = [json.loads(l) for l in (RUN / "trace_rappel.jsonl").open(encoding="utf-8")]
    return [l for l in lignes if str(l.get("person_id")) == PERSONNE]


def fenetre_de_la_requete() -> tuple[float, float]:
    import csv

    with (RUN / "pipeline_timing.csv").open(encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            if ligne["agent_id"] == PERSONNE and ligne["sim_time"] and int(float(ligne["sim_time"])) == QUERY_AT:
                return float(ligne["T_ltm_start"]), float(ligne["T_ltm_end"])
    raise SystemExit("no pipeline_timing.csv line for this decision")


# ── Options: read from the actual prompt, turned back into TravelPlan ────────────────────


def _secondes(texte: str) -> int:
    total = 0
    for n, unite in re.findall(r"(\d+)\s*(hour|minute|second)s?", texte):
        total += int(n) * {"hour": 3600, "minute": 60, "second": 1}[unite]
    return total


def options_du_prompt(prompt: str, purpose: str):
    """The options as the prompt presents them, in the presented order.

    The memory query is built on the list already ordered for presentation
    (`build_travel_plan_payload(context, shuffled_options, …)`), hence on this order. The
    short query template (`travel_plan_describe_lite.j2`) only reads: the mode, the total
    duration and the distance of a direct trip; the type, the short name, the boarding stop and
    the alighting stop of a public transport leg. Only these fields are filled
    with values from the prompt; coordinates are neutral, the template does not read them.
    """
    from models import Location, Transit, TransitLocation, TravelPlan
    from text_helper.templates.repository import gtfs_data
    from trip_helper.osmnx_direct import DIRECT_ROUTE_MARKER

    bloc = prompt.split("**Trip options**", 1)[1].split("**History:**", 1)[0]
    entetes = list(re.finditer(r"^- \[(\d+)\] ([^:]+): (.*)$", bloc, re.MULTILINE))
    lieu = Location(lon=0.0, lat=0.0)
    arret = lambda nom: TransitLocation(stop=nom, lat=0.0, lon=0.0)
    t0 = QUERY_AT * 1000
    plans = []
    for k, m in enumerate(entetes):
        fin = entetes[k + 1].start() if k + 1 < len(entetes) else len(bloc)
        etapes = re.findall(r"^\s+· (.*)$", bloc[m.end():fin], re.MULTILINE)
        mode, resume = m.group(2).strip(), m.group(3)
        direct = re.match(r"Estimated duration: (.*?)\. Distance: ([\d.]+) (km|m)\.", resume)
        if direct:
            duree = _secondes(direct.group(1))
            dist = float(direct.group(2)) * (1000 if direct.group(3) == "km" else 1)
            jambes = [Transit(start_time=t0, end_time=t0 + duree * 1000, start_location=arret(""),
                              end_location=arret(""), transit_route=DIRECT_ROUTE_MARKER[mode],
                              mode=mode, duration=duree, distance=dist)]
        else:
            jambes, precedent = [], ""
            for etape in etapes:
                marche = re.match(r"Walk to '(.*)': (.*)\.$", etape)
                ligne = re.match(r"(\w[\w ]*) '(.+)' to '(.*)': (.*)\.$", etape)
                if marche:
                    fin_arret = "" if marche.group(1) == purpose else marche.group(1)
                    jambes.append(Transit(start_time=t0, end_time=t0, start_location=arret(precedent),
                                          end_location=arret(fin_arret), is_transfer=True, mode="foot",
                                          duration=_secondes(marche.group(2))))
                    precedent = fin_arret
                elif ligne:
                    route_id = gtfs_data.get_route_id_by_name(ligne.group(2))
                    type_lu = gtfs_data.get_route_type_string_by_id(route_id)
                    if type_lu != ligne.group(1):
                        alarme(f"option {m.group(1)} : la ligne « {ligne.group(2)} » est de type "
                               f"« {type_lu} » au GTFS, le prompt dit « {ligne.group(1)} »")
                    jambes.append(Transit(start_time=t0, end_time=t0, start_location=arret(precedent),
                                          end_location=arret(ligne.group(3)), transit_route=route_id,
                                          mode=ligne.group(1).lower(), duration=_secondes(ligne.group(4))))
                    precedent = ligne.group(3)
                else:
                    alarme(f"option {m.group(1)} : étape illisible « {etape} »")
            duree = _secondes(resume.split(",")[0])
        plan = TravelPlan(id=f"option_{m.group(1)}", start_location=lieu, end_location=lieu,
                          start_time=t0, end_time=t0 + duree * 1000, purpose=purpose, legs=jambes)
        if plan.mode_label() != mode:
            alarme(f"option {m.group(1)} : étiquette reconstruite « {plan.mode_label()} » ≠ « {mode} »")
        plans.append(plan)
    return plans


# ── Memory state at the moment of the recall ─────────────────────────────────────────────


def ramener_a_l_instant_du_rappel(copie: Path, trace: list[dict], t_ltm: tuple[float, float]) -> dict:
    """Removes later entries and undoes later reinforcements. Returns a summary."""
    from llm.gravite import force_initiale
    from settings import settings
    from sim_clock import wall_clock

    ltm_dir = copie / "long_term_memory"
    meta_path = next((ltm_dir / "user_metadata").glob(f"*/{PERSONNE}.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entrees = meta["entries"]
    delta = float(settings.agent.memoire__force_delta_rappel_jours)
    plafond = float(settings.agent.memoire__force_max_jours)

    # 1. Entries written after the end of the query (Chroma clock, in seconds, UTC).
    base = sqlite3.connect(ltm_dir / "chroma_db" / "chroma.sqlite3")
    insertions = dict(base.execute(
        "select m.string_value, e.created_at from embeddings e join embedding_metadata m "
        "on m.id = e.id and m.key = 'doc_id'"
    ).fetchall())
    base.close()
    debut, fin = t_ltm
    posterieures, ambigues = [], []
    for doc, cree in insertions.items():
        t = datetime.strptime(cree, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        if t > fin:
            posterieures.append(doc)
        elif t >= math.floor(debut):
            ambigues.append(doc)
    if ambigues:
        alarme(f"entrées insérées dans la même seconde que la requête, ordre indécidable : {ambigues}")
    sans_index = [e["doc_id"] for e in entrees if e["doc_id"] not in insertions]
    if sans_index:
        alarme(f"entrées des métadonnées absentes de l'index Chroma : {sans_index}")
    log.info(f"checkpoint entries: {len(entrees)}; inserted after the query (removed): "
             f"{sorted(posterieures)}")

    # 2. The recalls, in the order they were PROCESSED (file order, not sim_ts order:
    # decisions are precomputed). The target recall, then those the checkpoint contains.
    cible = [i for i, l in enumerate(trace) if int(l["sim_ts"]) == QUERY_AT]
    if len(cible) != 1:
        raise SystemExit(f"{len(cible)} recalls at {QUERY_AT} in trace_rappel.jsonl, 1 expected")
    cible = cible[0]

    def historique(jusqu_a: int) -> dict[str, tuple[int, int | None]]:
        compte: dict[str, list[int]] = {}
        for l in trace[: jusqu_a + 1]:
            for s in l["servis"]:
                compte.setdefault(s["doc_id"], []).append(int(l["sim_ts"]))
        return {d: (len(v), v[-1]) for d, v in compte.items()}

    def date(ts):
        return wall_clock(ts).isoformat() if ts is not None else None

    conservees = [e for e in entrees if e["doc_id"] not in posterieures]
    dernier_inclus = None
    for k in range(cible, len(trace)):
        h = historique(k)
        if all((e["rappels"], e["dernier_rappel"]) == (h.get(e["doc_id"], (0, None))[0],
                                                      date(h.get(e["doc_id"], (0, None))[1]))
               for e in conservees):
            dernier_inclus = k
            break
    if dernier_inclus is None:
        alarme("aucun préfixe de trace_rappel.jsonl n'explique les compteurs de rappel du point : "
               "l'état au rappel n'est pas reconstructible")
        raise SystemExit(1)
    rappels_annules = [int(l["sim_ts"]) for l in trace[cible: dernier_inclus + 1]]
    log.info(f"the checkpoint contains the recalls {[date(t) for t in rappels_annules]}; "
             f"counters and last-recall dates explained for {len(conservees)}/{len(conservees)} entries")

    avant = historique(cible - 1)
    apres = historique(dernier_inclus)
    modifiees = 0
    for e in conservees:
        n_avant = avant.get(e["doc_id"], (0, None))[0]
        n_retire = apres.get(e["doc_id"], (0, None))[0] - n_avant
        if n_retire == 0:
            continue
        modifiees += 1
        force_point = float(e["force"])
        if e["memory_type"] == "concept":
            # A concept's strength does not enter the score (time there is the confidence).
            if force_point < plafond:
                e["force"] = force_point - delta * n_retire
            else:
                log.warning(f"{e['doc_id']} (concept): strength at the ceiling, left as is — "
                            f"no effect on the score")
        else:
            attendu = min(force_initiale(float(e["importance"] or 0.0)) + delta * e["rappels"], plafond)
            if abs(attendu - force_point) > 1e-9:
                alarme(f"{e['doc_id']} : force {force_point} ≠ force initiale + δ·rappels = {attendu}")
            e["force"] = min(force_initiale(float(e["importance"] or 0.0)) + delta * n_avant, plafond)
        e["rappels"] = n_avant
        e["dernier_rappel"] = date(avant.get(e["doc_id"], (0, None))[1])
    meta["entries"] = conservees
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if posterieures:
        import chromadb

        client = chromadb.PersistentClient(path=str(ltm_dir / "chroma_db"))
        collection = client.get_collection("memory_collection")
        collection.delete(where={"doc_id": {"$in": sorted(posterieures)}})
        log.info(f"Chroma index of the copy: {collection.count()} vectors after removal")
    return {
        "point_de_reprise": str(POINT.relative_to(RACINE)),
        "entrees_du_point": len(entrees),
        "entrees_retirees_posterieures": sorted(posterieures),
        "rappels_du_point_annules": [date(t) for t in rappels_annules],
        "entrees_dont_le_renforcement_est_annule": modifiees,
        "entrees_a_l_instant_du_rappel": len(conservees),
    }


# ── Replay ───────────────────────────────────────────────────────────────────────────────


def appliquer_la_configuration_du_run(trace_tmp: Path) -> None:
    import yaml
    from settings import settings

    agent = yaml.safe_load((RUN / "static_config.yaml").read_text(encoding="utf-8"))["agent"]
    changes = []
    for cle, valeur in agent.items():
        if hasattr(settings.agent, cle) and getattr(settings.agent, cle) != valeur:
            setattr(settings.agent, cle, valeur)
            changes.append(cle)
    # Observation only: nothing is written into the run's trace or log.
    settings.agent.trace_rappel_enabled = False
    settings.agent.journal_memoire_enabled = False
    settings.app.trace_rappel_file = str(trace_tmp)
    log.info(f"configuration « agent » du run appliquée ({len(changes)} clé(s) différente(s) du "
             f"config.yaml courant : {changes})")


def court(texte: str) -> str:
    mots = texte.split()
    return " ".join(mots[:MOTS_MAX]) + (" …" if len(mots) > MOTS_MAX else "")


async def rejouer(copie: Path, prompt: str) -> dict:
    import numpy as np
    from llm.axes import affinite_axes, affinite_meteo
    from llm.longterm import MultiUserLongTermMemory
    from settings import settings
    from urban_mobility_agents.agents.llm_agent import LlmAgent
    from urban_mobility_agents.agents.prompt_manager import PromptManager

    ltm = MultiUserLongTermMemory(
        storage_dir=str(copie / "long_term_memory"),
        long_term_memory_filter_by_datetime=settings.agent.long_term_memory_filter_by_datetime,
        max_loaded_metadata=settings.agent.long_term_max_loaded_metadata,
    )
    ltm.ensure_user_initialized(PERSONNE)
    entrees = {e.doc_id: e for e in ltm.user_metadata[PERSONNE]["entries"]}
    log.info(f"memory loaded: {len(entrees)} entries ({sum(e.est_episodique for e in entrees.values())} "
             f"episodic, {sum(not e.est_episodique for e in entrees.values())} concepts)")

    capture: dict = {}
    requete_code = ltm.aquery_user_memories
    classement_code = ltm.rank_nodes

    async def requete(**kw):
        capture["requete"] = kw
        # Membership of the structured pools, read BEFORE the recall reinforces anything
        # at all (pool B sorts on the last-recall date).
        capture["vivier_b"] = {r.metadata["doc_id"] for r in ltm.viviers_structures(PERSONNE, kw["modes_offerts"])
                               if r.metadata["vivier"] == "B"}
        capture["vivier_c"] = {r.metadata["doc_id"] for r in ltm.viviers_structures(PERSONNE, [])}
        capture["resultat"] = await requete_code(**kw)
        return capture["resultat"]

    poids = {
        "sim": settings.agent.long_term_retrieval__sim_weight,
        "temps": settings.agent.long_term_retrieval__time_weight,
        "g": settings.agent.long_term_retrieval__importance_weight,
        "axes": settings.agent.long_term_retrieval__affinite_weight,
        "meteo": settings.agent.long_term_retrieval__keyword_weight,
    }

    def composantes(n, e, ctx, query_at) -> dict:
        """The five components, by the very functions of `rank_nodes`."""
        return {
            "sim": float(np.clip(n.score, 0.0, 1.0)),
            "temps": float(ltm._time_decay_score(n.metadata.get("timestamp"), query_at, entree=e)),
            "g": float(getattr(e, "importance", 0.0) or 0.0),
            "axes": float(affinite_axes(
                getattr(e, "axe_objet", None), getattr(e, "axe_lieu", None),
                getattr(e, "axe_creneau", None), getattr(e, "axe_motif", None),
                objet_courant=ctx.get("axe_objet"), lieu_courant=ctx.get("axe_lieu"),
                creneau_courant=ctx.get("axe_creneau"), motif_courant=ctx.get("axe_motif"))),
            "meteo": float(affinite_meteo(getattr(e, "axe_meteo", None), ctx.get("axe_meteo"))),
        }

    def classement(query, query_at, nodes, entrees_par_doc=None, contexte=None):
        scores = classement_code(query, query_at, nodes, entrees_par_doc, contexte)
        # Components computed HERE, before `_renforcer_les_servis` touches the served ones
        # (strength, last recall): afterwards, their time weight would be 1.
        ctx, ep = contexte or {}, entrees_par_doc or {}
        lignes = []
        for n, score_code in zip(nodes, scores):
            e = ep.get(n.metadata.get("doc_id"))
            comp = composantes(n, e, ctx, query_at)
            somme = sum(poids[k] * comp[k] for k in poids)
            if abs(somme - float(score_code)) > TOLERANCE_SOMME:
                alarme(f"{n.metadata.get('doc_id')} : somme pondérée {somme!r} ≠ score du code "
                       f"{float(score_code)!r} (composantes {comp})")
            etat = {"dernier_rappel": e.dernier_rappel.isoformat() if e and e.dernier_rappel else None,
                    "force": float(e.force) if e and e.force is not None else None,
                    "rappels": int(e.rappels or 0) if e else None}
            lignes.append((n, e, comp, float(score_code), etat))
        capture.update(candidats=lignes, contexte=ctx)
        log.info(f"{len(lignes)} candidates ranked; weighted sum of the five components = code "
                 f"score for each (tolerance {TOLERANCE_SOMME})")
        return scores

    ltm.aquery_user_memories = requete
    ltm.rank_nodes = classement

    agent = types.SimpleNamespace(
        prompt_manager=PromptManager(str(SA / "urban_mobility_agents/agents/prompts")),
        long_term_memory=ltm,
    )
    agent._weather_info = types.MethodType(LlmAgent._weather_info, agent)
    agent._weather_timestamp = types.MethodType(LlmAgent._weather_timestamp, agent)
    contexte = types.SimpleNamespace(timestamp=QUERY_AT, person=types.SimpleNamespace(person_id=PERSONNE))
    purpose = re.search(r"Destination: (\w+)", prompt).group(1)
    options = options_du_prompt(prompt, purpose)
    log.info(f"options read from the prompt: {[o.mode_label() for o in options]} (purpose « {purpose} »)")

    historique = await LlmAgent.query_past_experiences_for_travel(agent, contexte, options, ())
    if ltm._flush_task is not None:
        ltm._flush_task.cancel()  # the copy does not need to be rewritten

    candidats = capture["candidats"]
    vivier_a = {n.metadata["doc_id"] for n, *_ in candidats if n.metadata.get("vivier") == "A"}
    servis = [r.metadata["doc_id"] for r in capture["resultat"]]
    return {"ltm": ltm, "capture": capture, "candidats": candidats, "servis": servis,
            "historique": historique, "poids": poids, "vivier_a": vivier_a, "entrees": entrees}


def est_concept(n) -> bool:
    from llm.memory import MemoryType

    return str(n.metadata["memory_type"]) == str(MemoryType.CONCEPT.value)


def ligne_injectee(n) -> str:
    """Exact form of a memory in the prompt (see `query_past_experiences_for_travel`)."""
    if est_concept(n):
        return f"[Concept] {json.loads(n.content)[0]}"
    return f"[{datetime.fromisoformat(n.metadata['timestamp']).strftime('%A, %B %d')}] {n.content}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--garder", action="store_true", help="keep the checkpoint's working copy")
    args = parser.parse_args()

    os.environ.setdefault("APP_NO_RUN_ARTIFACTS", "1")
    sys.path.insert(0, str(SA))
    t_debut = time.monotonic()
    log.info(f"example K — recall of {PERSONNE} on {datetime.fromtimestamp(QUERY_AT, timezone.utc):%Y-%m-%d} "
             f"(GAMA timestamp {QUERY_AT}) — run {RUN.name}")

    travail = Path(tempfile.mkdtemp(prefix="exemple_L_"))
    try:
        with Etape("configuration du run"):
            appliquer_la_configuration_du_run(travail / "trace_rappel_rejeu.jsonl")
        with Etape("lecture du run (prompt, trace des rappels, chronométrage)"):
            prompt = prompt_de_la_decision()
            trace = trace_des_rappels()
            reel = next(l for l in trace if int(l["sim_ts"]) == QUERY_AT)
            t_ltm = fenetre_de_la_requete()
            log.info(f"{len(trace)} recalls traced for {PERSONNE}; actual query between "
                     f"{datetime.fromtimestamp(t_ltm[0], timezone.utc):%H:%M:%S.%f} and "
                     f"{datetime.fromtimestamp(t_ltm[1], timezone.utc):%H:%M:%S.%f} UTC; "
                     f"{reel['candidats']} candidates according to the trace")
        with Etape("copie du point et retour à l'instant du rappel"):
            shutil.copytree(POINT, travail / "point")
            bilan_etat = ramener_a_l_instant_du_rappel(travail / "point", trace, t_ltm)
        with Etape("rejeu de la requête et du classement par le code du projet"):
            r = asyncio.run(rejouer(travail / "point", prompt))
    finally:
        if args.garder:
            log.info(f"working copy kept: {travail}")
        else:
            shutil.rmtree(travail, ignore_errors=True)

    # ── Checks against the run ───────────────────────────────────────────────────────────
    with Etape("contrôles contre le run"):
        par_doc = {c[0].metadata["doc_id"]: c for c in r["candidats"]}
        if len(r["candidats"]) != int(reel["candidats"]):
            alarme(f"{len(r['candidats'])} candidats rejoués, {reel['candidats']} dans la trace")
        servis_reels = [s["doc_id"] for s in reel["servis"]]
        if r["servis"] != servis_reels:
            alarme(f"top-10 rejoué {r['servis']} ≠ top-10 tracé {servis_reels}")
        ecart_max, arrondis_egaux = 0.0, 0
        for s in reel["servis"]:
            if s["doc_id"] in par_doc:
                rejoue = par_doc[s["doc_id"]][3]
                ecart = abs(rejoue - float(s["score"]))
                ecart_max = max(ecart_max, ecart)
                arrondis_egaux += round(rejoue, 4) == float(s["score"])
                if ecart > TOLERANCE_TRACE:
                    alarme(f"{s['doc_id']} : score rejoué {rejoue:.6f}, tracé {s['score']}")
        log.info(f"top-10 scores: {arrondis_egaux}/10 equal to the traced score once rounded to 4 decimals; "
                 f"largest gap {ecart_max:.2e} (tolerance {TOLERANCE_TRACE})")

        historique_reel = lignes_historique(prompt)
        if r["historique"] != historique_reel:
            for i, (a, b) in enumerate(zip(r["historique"], historique_reel)):
                if a != b:
                    log.warning(f"history, line {i}: replayed « {a[:90]} » / actual « {b[:90]} »")
            if len(r["historique"]) != len(historique_reel):
                log.warning(f"history: {len(r['historique'])} lines replayed, {len(historique_reel)} actual")
        souvenirs_reels = [l for l in historique_reel if l.startswith("[")]
        souvenirs_rejoues = [l for l in r["historique"] if l.startswith("[")]
        if souvenirs_rejoues != souvenirs_reels:
            alarme(f"souvenirs injectés rejoués {souvenirs_rejoues} ≠ réels {souvenirs_reels}")
        else:
            log.info(f"the {len(souvenirs_reels)} injected memories of the real prompt are reproduced exactly")
        noyau_identique = [l for l in r["historique"] if not l.startswith("[")] == \
                          [l for l in historique_reel if not l.startswith("[")]
        if not noyau_identique:
            # Outside the score: the core block is not ranked. Its titles and labels were
            # translated in `llm/noyau.py` after the run, and the checkpoint's trip log
            # counts the trips completed between the recall and the writing of the checkpoint.
            log.warning("core block differs from the real prompt (not ranked): labels translated "
                        "since the run, checkpoint trip log later than the recall")

    # ── Output ──────────────────────────────────────────────────────────────────────────
    from sim_clock import gama_timestamp, wall_clock

    kw = r["capture"]["requete"]
    ctx = r["capture"]["contexte"]
    top = []
    for rang, doc in enumerate(r["servis"]):
        n, e, comp, s, etat = par_doc[doc]
        concept = est_concept(n)
        viviers = [v for v, ens in (("A", r["vivier_a"]), ("B", r["capture"]["vivier_b"]),
                                    ("C", r["capture"]["vivier_c"])) if doc in ens]
        texte = json.loads(n.content)[0] if concept else n.content
        trace_score = next(x["score"] for x in reel["servis"] if x["doc_id"] == doc)
        top.append({
            "rang": rang + 1,
            "id": doc,
            "type": "concept" if concept else "episodique",
            "memory_type": str(n.metadata["memory_type"]),
            "texte_court": court(texte),
            "ecrit_le": n.metadata["timestamp"],
            "rappels_avant": etat["rappels"],
            "dernier_rappel_avant": etat["dernier_rappel"],
            "delta_t_jours": None if concept else round(
                (QUERY_AT - gama_timestamp(datetime.fromisoformat(etat["dernier_rappel"] or n.metadata["timestamp"])))
                / 86400, 4),
            "force_jours": None if concept else round(etat["force"], 4),
            "observations": e.observations if concept else None,
            "contre_exemples": e.contre_exemples if concept else None,
            "viviers": viviers,
            "sim": round(comp["sim"], 4),
            "temps_ou_confiance": round(comp["temps"], 4),
            "composante_temps": "confiance (Laplace)" if concept else "exp(-Δt/force)",
            "g": round(comp["g"], 4),
            "axes": round(comp["axes"], 4),
            "meteo": round(comp["meteo"], 4),
            "S": round(s, 4),
            "S_trace_run": trace_score,
            "axes_du_souvenir": {"objet": e.axe_objet, "lieu": e.axe_lieu, "creneau": e.axe_creneau,
                                 "motif": e.axe_motif, "meteo": e.axe_meteo},
            "injecte": ligne_injectee(n) in r["historique"],
        })
    # Injection rule: non-empty core block → only the N most RECENT (writing date) of the
    # top-K enter the prompt.
    n_injectes = int(__import__("settings").settings.agent.memoire__episodiques_avec_noyau)
    plus_recents = {t["id"] for t in sorted(top, key=lambda t: t["ecrit_le"])[-n_injectes:]}
    injectes = {t["id"] for t in top if t["injecte"]}
    if injectes != plus_recents:
        alarme(f"injectés {sorted(injectes)} ≠ {n_injectes} plus récents du top-10 {sorted(plus_recents)}")
    else:
        log.info(f"injection rule verified: the {n_injectes} injected memories are the most recent of the top-10 "
                 f"({sorted(injectes)})")
    sortie = {
        "description": "Annexe L — rappel réel du 2026-03-31 05:42 (persona 861500), rejoué hors ligne "
                       "par le code du projet ; produit par scripts/annexes/exemple_L_rappel.py",
        "run": str(RUN.relative_to(RACINE)),
        "etat_memoire": bilan_etat,
        "requete": {
            "query_at": QUERY_AT,
            "heure_murale": wall_clock(QUERY_AT).isoformat(),
            "texte": kw["query"],
            "modes_offerts": kw["modes_offerts"],
            "axe_objet": sorted(ctx["axe_objet"]),
            "axe_lieu": ctx["axe_lieu"],
            "axe_creneau": ctx["axe_creneau"],
            "axe_motif": ctx["axe_motif"],
            "axe_meteo": ctx["axe_meteo"],
            "top_k": kw["top_k"],
            "max_past_days": kw["max_past_days"],
        },
        "poids": r["poids"],
        "candidats": {
            "total": len(r["candidats"]),
            "vivier_A": len(r["vivier_a"]),
            "vivier_B": len(r["capture"]["vivier_b"]),
            "vivier_C": len(r["capture"]["vivier_c"]),
            "ids_vivier_B": sorted(r["capture"]["vivier_b"], key=lambda d: int(d.split("_")[1])),
            "ids_vivier_C": sorted(r["capture"]["vivier_c"], key=lambda d: int(d.split("_")[1])),
            "hors_candidats_gravite_choc": sorted(
                (d for d, e in r["entrees"].items()
                 if float(e.importance or 0.0) >= float(__import__("settings").settings.agent.memoire__importance_choc)
                 and d not in {c[0].metadata["doc_id"] for c in r["candidats"]}),
                key=lambda d: int(d.split("_")[1])),
            "par_origine": {v: sum(1 for n, *_ in r["candidats"] if n.metadata.get("vivier") == v) for v in "ABC"},
        },
        "top10": top,
        "tous_les_candidats": [
            {"rang": i + 1, "id": c[0].metadata["doc_id"],
             "type": "concept" if est_concept(c[0]) else "episodique",
             "vivier_d_origine": c[0].metadata.get("vivier"),
             **{k: round(v, 4) for k, v in c[2].items()}, "S": round(c[3], 4),
             "debut": court(json.loads(c[0].content)[0] if est_concept(c[0]) else c[0].content)[:80]}
            for i, c in enumerate(sorted(r["candidats"], key=lambda c: -c[3]))
        ],
        "controles": {
            "candidats_trace_run": int(reel["candidats"]),
            "top10_identique_a_la_trace": r["servis"] == servis_reels,
            "ecart_max_score_trace": ecart_max,
            "scores_egaux_a_4_decimales": arrondis_egaux,
            "souvenirs_injectes_identiques_au_prompt": souvenirs_rejoues == souvenirs_reels,
            "bloc_noyau_identique_au_prompt": noyau_identique,
            "injectes_egaux_aux_plus_recents_du_top10": injectes == plus_recents,
            "episodiques_avec_noyau": n_injectes,
            "alarmes": _ALARMES,
        },
    }
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"written: {SORTIE.relative_to(RACINE)} ({len(top)} memories, "
             f"{sum(t['injecte'] for t in top)} injected) — {time.monotonic() - t_debut:.1f} s")
    if _ALARMES:
        log.error(f"[ALARME] {len(_ALARMES)} divergence(s) from the run — example NOT verified")
        return 1
    log.info("SUCCESS: top-10, scores, number of candidates and injected memories match the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
