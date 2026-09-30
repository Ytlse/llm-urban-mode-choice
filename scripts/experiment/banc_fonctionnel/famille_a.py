"""Family A — the synthetic run, ZERO calls (ticket 100, functional tests).

WHAT NO UNIT TEST CATCHES
------------------------------------
The repository carries 1,900 tests on this ticket. Each one calls a function and looks at what
it returns. What nobody checked is that **the links hold together**: a badly named
column, a role that does not propagate, a trace the extractor cannot read, a relative
day shifted by one notch. Those defects only show at the end of the chain, and a
forty-day campaign is a bad place to discover them.

So we build a complete run — population, decisions, memory, event, checkpoints
— without a simulator and without a model, then run **the real analysis chain** on it.

    python -m scripts.experiment.banc_fonctionnel.famille_a
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta, timezone
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
AGENTS = RACINE / "services" / "llm-agents"
for chemin in (str(RACINE), str(AGENTS)):
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

from scripts.experiment.banc_fonctionnel import stubs  # noqa: E402


class Echec(AssertionError):
    """A link does not hold. The message says which one and what was expected."""


_resultats: list[tuple[str, str, str]] = []


def _verifier(code: str, condition: bool, attendu: str, constate: str = "") -> None:
    _resultats.append((code, "✅" if condition else "❌", attendu))
    if not condition:
        raise Echec(f"{code} — expected: {attendu}. Observed: {constate or 'non'}")


# ── A2, A4: the household memory, without a clock ──────────────────────────────────────────
def a2_a4_le_foyer(tmp: Path) -> None:
    """The evening summary, the single hop, and the resume — three nights in three seconds."""
    from llm import foyer
    from settings import settings

    population = stubs.population_de_banc()
    foyer.reinitialiser()
    settings.agent.memoire__partage_foyer_enabled = True
    try:
        foyer.initialiser(population)
        ltm = stubs.MemoireLongueStub()
        a, b = population[0], population[1]  # same household 605813
        # Evening = 9 pm (ANCRE is at 5 am, hence `hours=16`). The household now only talks at
        # the first consolidation after `memoire__recit_soir_heure` (2026-09-25): the old
        # `hours=22` fell at 3 am — the day boundary, not an evening.

        # NIGHT 1 — A has written its summary, B hears it.
        ltm.ajouter(stubs.reflexion_stub(
            a.person_id, "Line A was packed again this morning; I gave up and walked.", 1))
        bloc1 = foyer.bloc_du_soir(ltm, b, stubs.ANCRE + timedelta(days=1, hours=16))
        _verifier("A4.1", "Line A was packed" in bloc1,
                  "le bilan de A est CITÉ dans le bloc de B", bloc1[:80])

        # NIGHT 2 — nothing new at A's: B must hear nothing again.
        bloc2 = foyer.bloc_du_soir(ltm, b, stubs.ANCRE + timedelta(days=2, hours=16))
        _verifier("A4.2", bloc2 == "",
                  "un bilan déjà entendu ne repart pas (G1)", bloc2[:80])

        # NIGHT 3 — A has an ANCHORED belief: it crosses. A HEARD belief: never.
        ltm.ajouter(stubs.concept_stub(
            a.person_id, "Line A is packed before 9am", 3, observations=3, origine="vecu"))
        ltm.ajouter(stubs.concept_stub(
            a.person_id, "The ring road is hopeless", 3, mode="car",
            observations=9, origine="entendu"))
        bloc3 = foyer.bloc_du_soir(ltm, b, stubs.ANCRE + timedelta(days=3, hours=16))
        _verifier("A4.3", "Line A is packed before 9am" in bloc3,
                  "une croyance ancrée et vécue traverse", bloc3[:120])
        _verifier("A4.4", "ring road" not in bloc3,
                  "une croyance ENTENDUE ne repart jamais, même confirmée 9 fois (D2)",
                  bloc3[:160])

        # A5 — the hot resume does not make anyone hear again.
        sauvegarde = foyer.etat_pour_reprise()
        foyer.reinitialiser()
        foyer.initialiser(population)
        foyer.charger_etat(sauvegarde)
        bloc4 = foyer.bloc_du_soir(ltm, b, stubs.ANCRE + timedelta(days=4, hours=16))
        _verifier("A5.1", bloc4 == "",
                  "après une reprise, rien de déjà entendu ne repart", bloc4[:80])

        # Measure no. 1 of 078 § 6: it must be EMITTED, not only computed.
        compteurs = foyer.compteurs()
        _verifier("A4.5", compteurs.get("candidats", 0) > 0 and "refus_R1" in compteurs,
                  "les refus par règle sont comptés (mesure n° 1 du 078 § 6)", str(compteurs))
    finally:
        settings.agent.memoire__partage_foyer_enabled = False
        foyer.reinitialiser()


# ── A7: the cache, cut on event days and not on the others ─────────────────────────────────
def a7_le_cache(tmp: Path) -> None:
    from llm import evenements as ev
    from llm.evenements import RegistreEvenements, charger
    from llm.evenements import calendrier as cal

    registre = RegistreEvenements(charger(
        AGENTS / "config" / "evenements" / "c6_voiture_suspecte.yaml"))
    try:
        for jour, attendu in ((14, False), (15, True), (16, True), (17, False)):
            RegistreEvenements.jour_du_run = staticmethod(lambda ts, _j=jour: _j)
            _verifier(f"A7.{jour}", registre.cache_coupe(0) is attendu,
                      f"jour {jour} : cache {'coupé' if attendu else 'actif'}")
        # The counters must count, even on a nominal day.
        RegistreEvenements.jour_du_run = staticmethod(lambda ts: 18)
        for _ in range(3):
            registre.noter_decision(0, depuis_cache=True)
        registre.noter_decision(0, depuis_cache=False)
        _verifier("A7.c", registre._compteurs.decisions_depuis_cache == 3,
                  "les décisions servies par le cache sont comptées, jour par jour",
                  str(registre._compteurs))
    finally:
        RegistreEvenements.jour_du_run = staticmethod(cal.jour_du_run)
    _verifier("A7.0", ev.cache_coupe(0) is False,
              "sans événement déclaré, le cache n'est jamais coupé")


# ── A6: a day due during the replay ────────────────────────────────────────────────────────
def a6_le_gel(tmp: Path) -> None:
    """The controller must REFUSE to inject during a replay, and say so."""
    source = (AGENTS / "urban_mobility_agents" / "simulation_controller.py").read_text("utf-8")
    debut = source.index("async def _injecter_evenements_du_reveil")
    fin = source.index("async def _tirer_accidents_du_jour")
    methode = source[debut:fin]
    _verifier("A6.1", "gel_actif()" in methode,
              "l'injection au réveil vérifie le gel du rejeu")
    _verifier("A6.2", "[ALARME]" in methode and "REJEU" in methode,
              "un jour dû pendant le rejeu lève une alarme nommée")
    # ⚠ We compare the CALL, not the word: the method's docstring cites
    # `add_short_term_memory` to explain the rule, and cites it BEFORE the code. Searching for
    # the bare word made the bench fail on its own prose — a test that reads the comments
    # forbids explaining the rule it checks (same pattern as R14 of ticket 079).
    _verifier("A6.3",
              methode.index("gel_actif()")
              < methode.index("self.agent.add_short_term_memory("),
              "le gel est vérifié AVANT le premier appel d'écriture")


# ── A1, A3, A8: the wake-up injection, end to end ──────────────────────────────────────────
def a1_a3_a8_linjection(tmp: Path) -> None:
    from llm.evenements import RegistreEvenements, charger
    from llm.evenements import calendrier as cal
    from llm.gravite import force_initiale

    population = stubs.population_de_banc()
    article = AGENTS / "config" / "evenements" / "a13_punaises_metro.yaml"
    registre = RegistreEvenements(charger(article), journal=tmp / "evenements.jsonl")
    lecteurs = registre.lecteurs(population)

    _verifier("A3.1", len(lecteurs) == 6,
              "un lecteur par foyer exposé, six foyers", f"{len(lecteurs)} lecteur(s)")
    foyers_servis = {h for h, _ in lecteurs.values()}
    _verifier("A3.2", len(foyers_servis) == 6,
              "six foyers distincts", str(sorted(foyers_servis)))
    co_residents = {a.person_id for a in population} - set(lecteurs)
    _verifier("A3.3", len(co_residents) == 6,
              "six co-résidents ne reçoivent RIEN — c'est le témoin interne",
              str(len(co_residents)))

    # Publication days are drawn PER HOUSEHOLD: not all read on the same morning.
    jours = {h: registre.jour_de(p, h) for p, (h, _r) in lecteurs.items()}
    _verifier("A1.1", all(9 <= j <= 13 for j in jours.values()),
              "chaque parution tombe dans la fenêtre déclarée [9, 13]", str(jours))
    _verifier("A1.2", len(set(jours.values())) > 1,
              "deux foyers ne lisent pas le même matin — sinon le calendrier et l'article "
              "seraient inséparables", str(jours))

    # Each reader is served ONCE, and only once.
    servis: dict[str, int] = {}
    try:
        for jour in range(8, 15):
            RegistreEvenements.jour_du_run = staticmethod(lambda ts, _j=jour: _j)
            for _ in range(3):  # three syncs in the day: idempotence
                for pid, applique in registre.dus_au_reveil(0, population):
                    servis[pid] = servis.get(pid, 0) + 1
                    registre.tracer(applique, pid, 0, 0.30, None)
    finally:
        RegistreEvenements.jour_du_run = staticmethod(cal.jour_du_run)
    _verifier("A1.3", set(servis) == set(lecteurs) and all(n == 1 for n in servis.values()),
              "chaque lecteur servi une fois, et une seule", str(servis))

    # A8 — one line per exposure, not two.
    lignes = [json.loads(l) for l in
              (tmp / "evenements.jsonl").read_text("utf-8").splitlines() if l.strip()]
    _verifier("A8.1", len(lignes) == len(lecteurs),
              "une ligne de trace par exposition", f"{len(lignes)} ligne(s)")
    _verifier("A8.2", all(l["canal"] == "lu" and l["moment"] == "reveil" for l in lignes),
              "la trace porte le canal et le moment")
    _verifier("A8.3", all(l["retard_injecte_s"] == 0 for l in lignes),
              "un article ne fait subir AUCUN retard")

    # A2 — the severity is that of the judgement, ALONE (D7).
    _verifier("A2.1", abs(force_initiale(0.30) - 7.84) < 0.01,
              "un jugement `notable` (0,30) donne 7,84 jours de durée de vie",
              f"{force_initiale(0.30):.2f}")
    _verifier("A2.2", abs(force_initiale(0.70) - 14.56) < 0.01,
              "et non 14,56, qui serait la valeur sous l'ancien plancher")


# ── A9, A10: the analysis chain reads what the run writes ──────────────────────────────────
def a9_a10_lanalyse(tmp: Path) -> None:
    from scripts.analysis import figure_evenement as fig
    from scripts.analysis import tableau_quatre_voies as quatre
    from scripts.analysis.presse import campagne
    from scripts.analysis.presse.grille import charger_grille

    grille = RACINE / "data" / "presse" / "grille_signes.yaml"
    articles = sorted({c.article for c in charger_grille(grille).cellules})

    # Dated like `evenements.jsonl`: the four-channel table only keeps decisions
    # after the exposure, and discards those that cannot be placed.
    exposition = stubs.ANCRE.replace(tzinfo=timezone.utc)
    runs = []
    for article in articles:
        run = tmp / "runs" / article
        stubs.moves_stub(run / "moves.csv", evenement=article)
        (run / "evenements.jsonl").write_text(
            json.dumps({"evenement_id": article, "canal": "lu", "moment": "reveil",
                        "person_id": "expose_0", "texte": "Bed bugs on line A.",
                        "timestamp": int(exposition.timestamp()),
                        "horodatage_simule": exposition.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n",
            encoding="utf-8")
        runs.append(run)

    # A9 — the figure comes out, with its three roles.
    moyennes, effectifs, evenement = fig.series(runs[0], "car")
    svg = fig.composer(moyennes, effectifs, "car", evenement)
    _verifier("A9.1", svg.startswith("<svg") and "Exposed" in svg and "Control" in svg,
              "la figure se compose avec les trois rôles")
    _verifier("A9.2", all(len(moyennes[r]) > 0 for r in ("expose", "co_resident", "temoin")),
              "les trois rôles portent des points", str({k: len(v) for k, v in moyennes.items()}))

    # A9 — the four-channel table reads the trace AND the prompts.
    # ⚠ Without `llm_exchanges.jsonl`, it outputs "non concluant" — which is correct, and it is
    # in fact the first case to check: an absence of prompts must not read as
    # "the event weighed on nothing".
    muet = quatre.rendre(quatre.depouiller(runs[0]))
    _verifier("A9.3", "non concluant" in muet and "n'a pesé sur rien" in muet,
              "sans prompts, le tableau DIT qu'il ne sait pas dire", muet[:120])

    stubs.echanges_stub(runs[0] / "llm_exchanges.jsonl", agent="expose_0",
                        texte="Bed bugs on line A.", exposition=int(exposition.timestamp()))
    resultat = quatre.depouiller(runs[0])
    rendu = quatre.rendre(resultat)
    _verifier("A9.4", "habitudes" in rendu and "changements" in rendu,
              "avec des prompts, le tableau nomme les quatre voies", rendu[:160])

    # ⚠ We read the CELLS, not the whole rendering. A9.5 checked `"~" in rendu`, and the legend
    # ("`~` = matching by salient words…") prints it at each rendering: the check stayed
    # green without any indicative cell coming out — and none came out, the stub putting
    # no salient word in its rephrased prompt (2026-09-25).
    compte = resultat["compte"]
    connaissances = compte[("lu", "expose", "connaissances")]
    changements = compte[("lu", "expose", "changements")]
    _verifier("A9.5",
              connaissances["saillant"] == 1 and connaissances["exact"] == 0
              and changements["exact"] == 1 and changements["saillant"] == 0,
              "la voie exacte (« changements ») et la voie par mots saillants "
              "(« connaissances ») se distinguent dans le dépouillement",
              f"connaissances={dict(connaissances)}, changements={dict(changements)}")

    # And the rendering marks the cell, not only the legend. Columns read by their name.
    tableau = [l.split("|") for l in quatre.rendre(resultat, markdown=True).splitlines()
               if l.count("|") >= len(quatre.VOIES)]
    en_tete = [c.strip() for c in tableau[0]]
    ligne = next((dict(zip(en_tete, (c.strip() for c in l))) for l in tableau[1:]
                  if [c.strip() for c in l[:2]] == ["lu", "expose"]), {})
    _verifier("A9.6", ligne.get("connaissances") == "~1" and ligne.get("changements") == "1",
              "la cellule « connaissances » porte `~1`, la cellule « changements » `1`",
              str(ligne))

    # A10 — the campaign extraction, on five articles.
    rapport = campagne.rendre(campagne.depouiller(
        runs, grille_chemin=grille, plancher=0.032))
    _verifier("A10.1", "grille gelée" in rapport and "0.032" in rapport,
              "le rapport recopie l'empreinte de la grille et le plancher déclaré")
    _verifier("A10.2", "Pas de test binomial" in rapport,
              "le binomial est retiré — les cellules d'un événement ne sont pas indépendantes")
    _verifier("A10.3", "par événement" in rapport,
              "l'incertitude est groupée par événement")


def main() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="banc_a_"))
    etapes = (
        ("A1/A3/A8 — injection au réveil, rôles, trace", a1_a3_a8_linjection),
        ("A2/A4 — le foyer, le saut unique, la reprise", a2_a4_le_foyer),
        ("A6 — le gel du rejeu", a6_le_gel),
        ("A7 — le cache coupé les jours d'événement", a7_le_cache),
        ("A9/A10 — la chaîne d'analyse", a9_a10_lanalyse),
    )
    print(f"Functional bench — family A (zero calls). Sandbox: {tmp}\n")
    echecs = 0
    for titre, etape in etapes:
        try:
            etape(tmp)
            print(f"  ✅ {titre}")
        except Echec as err:
            echecs += 1
            print(f"  ❌ {titre}\n     {err}")
        except Exception as err:  # noqa: BLE001
            echecs += 1
            print(f"  💥 {titre}\n     {type(err).__name__}: {err}")

    print(f"\n{len(_resultats)} vérification(s), "
          f"{sum(1 for _c, s, _a in _resultats if s == '✅')} au vert.")
    if echecs:
        print(f"⚠ {echecs} step(s) failed — do NOT launch family B before fixing.")
    else:
        print("The chain holds. Family B can start (27 calls, Groq only).")
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
