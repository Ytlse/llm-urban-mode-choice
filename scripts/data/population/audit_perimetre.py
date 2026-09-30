"""
audit_perimetre.py — The nine baseline gaps between the surveyed population and the
simulated population, measured one by one.

    services/llm-agents/.venv/bin/python -m scripts.data.population.audit_perimetre
    services/llm-agents/.venv/bin/python -m scripts.data.population.audit_perimetre \
        --population data/population/toulouse_population_1000.json \
        --run experiments/current \
        --trace docs/traces/2026-08-24_perimetre_population

WHAT IT MEASURES, AND WHY IT EXISTS (ticket 020). The whole measurement chain of the
repository compares simulated modal shares with the targets of `cerema_values.yaml` —
overall and in eight subcategories. This comparison assumes that both sides
talk about the same population and the same counted object. That was not established: it was
assumed. Tickets 015, 016, 017 and 019 all showed the same pattern — a
coefficient learned on one variable, applied to another, and the gap invisible in
the aggregates. The population scope is the most upstream link: a scope
bias shifts ALL targets at once.

THE OUTPUT PRINCIPLE. Each axis returns one line: survey value, simulated value,
gap, and verdict. **An unmeasured axis is an axis that passes**, and that is exactly the
emptiness pattern the project tracks: the script therefore returns `non mesurable` with its
reason, never a silence nor a 0.

THREE POSSIBLE VERDICTS, and they are not interchangeable:
  * `conforme`     — the gap is under the tolerance; nothing to correct.
  * `à corriger`   — the gap affects the results; it opens a ticket.
  * `à publier`    — the gap is real and not correctable in this ticket; it appears
                     among the publication's limits WITH ITS MAGNITUDE.

WHAT IT DOES NOT DO. No correction. Ticket 020 establishes and qualifies the
gaps; corrections that go beyond a measurement adjustment open their own
tickets. That is what happened for A2 and A4: ticket 021 set the residence
ring ON THE PERSONA (trait `residence_zone`, by list of communes), ticket 028
re-stratified terminal time on the same table (`tt4`) and made "out of scope"
explicit. Axis A2 now checks that neither of them goes back to distance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mobility_core.geo_reference import haversine_km, hypercenter  # noqa: E402
from mobility_core.population_reference import (  # noqa: E402
    COURONNES, MIN_AGE, OUT_OF_PERIMETER, couronne_canonique, couronne_commune_counts,
    couronne_population_shares, household_targets, household_weight,
    population_reference, survey_window, surveyed_weekdays)
from mobility_core.residence_zone import CommunalZones  # noqa: E402
# Detail table and aggregation table of mode labels. They live in a
# module shared with the analysis notebooks (`scripts/analysis/mode_labels.py`) for
# the reason that moved `CommunalZones` up into `mobility_core` in ticket 021: two
# copies of a reference classification end up diverging, and that is
# exactly what happened here — the notebook and the audit each ignored "Train"
# on its own side.
from scripts.analysis.mode_labels import (  # noqa: E402
    SCORED_CATEGORIES, SURVEY_CATEGORIES, UNKNOWN as UNKNOWN_MODE,
    aggregation_table, category_of, log_alarm, tally_labels)

DEFAULT_RUN = REPO_ROOT / "experiments" / "current"
# The audited population is THE ONE THAT RAN, taken from the run directory. Decision of
# the repository's author (2026-09-04): « il faut prendre celui qui a tourné ; ça ne me paraît pas
# correct de faire un audit sur un fichier pas utilisé. »
#
# What made it necessary. The historical default pointed to
# `data/population/toulouse_population_1000.json` — the raw generator output lying around
# in that folder, 1,021 personas — whereas the run simulated the sealed cohort, 1,000
# personas drawn separately. The two populations share **only one identifier in a thousand**:
# axis A2 thus joined 6 trips out of the run's 5,322 and published a gap of 154.3 pt
# computed on those six. On the run's population, the join is complete and the gap is
# 41.2 pt.
#
# ⚠ This is not a measurement adjustment, it is a change of OBJECT: the nine axes
# measured a file nobody had simulated. Verdicts published before that date on
# the historical default are therefore **obsolete**, not "improved" — in particular A4, which goes from
# "à corriger" to "conforme", and the exit code, which goes from 2 to 0. Auditing the generation
# chain is the job of `control_population.py` and of sealing; auditing the experiment
# that ran is the job of this script.
#
# `--population` remains to audit a named file (a reference population, a specific
# seal): the default closes off no reading, it picks the right one by default.
DEFAULT_POPULATION_DANS_LE_RUN = "population_1000.json"
DEFAULT_POPULATION = REPO_ROOT / "data" / "population" / "toulouse_population_1000.json"
CEREMA_VALUES = REPO_ROOT / "scripts" / "data" / "population" / "cerema_values.yaml"
COURONNE_GEOJSON = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "couronne_perimetre.geojson"
WEATHER_CSV = REPO_ROOT / "data" / "weather" / "meteo_toulouse_12_mois.csv"
# Terminal time resource: its `meta.crown_definition` says on which zoning the
# laws are stratified. It is the third place where distance could come back (A2).
TERMINAL_TIME_JSON = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "terminal_time_emc2.json"

# The four scored categories, read from the shared module. `MOVE_MODE_MAP` was
# removed on 2026-09-04: it was a four-entry table WITHOUT "Train", and a
# train trip left the modal share audit through a silent `continue`.
# The replacement counts everything, aggregates to the survey categories and raises an alarm on
# any label outside the table.
SCORED_MODES = SCORED_CATEGORIES

# Thresholds of the persona ↔ trip join of the per-zone shares (A2). Beyond
# JOIN_ALARM_PCT of trips without a persona, an [ALARME] fires; under JOIN_VOID_PCT
# of join, the table declares itself NOT MEASURABLE rather than publishing an L1 computed
# on a handful of rows. "An unmeasured axis is an axis that passes": here it was
# an unmeasured SUB-TABLE that passed.
JOIN_ALARM_PCT = 5.0
JOIN_VOID_PCT = 50.0

# Name displayed in app.log: `make error` must point to the audit, not to the
# module that carries the handler.
_ALARM_LOGGER = "scripts.data.population.audit_perimetre"

CONFORME = "conforme"
A_CORRIGER = "à corriger"
A_PUBLIER = "à publier"
NON_MESURABLE = "non mesurable"


@dataclass
class Finding:
    """An examined axis. `simule=None` means not measurable, with its reason."""

    axe: str
    titre: str
    enquete: str
    simule: str
    ecart: str
    verdict: str
    detail: str = ""
    tables: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"axe": self.axe, "titre": self.titre, "enquete": self.enquete,
                "simule": self.simule, "ecart": self.ecart, "verdict": self.verdict,
                "detail": self.detail, "tables": self.tables}


# ── Reading inputs ────────────────────────────────────────────────────────────

def load_population(path: Path) -> list[dict]:
    people = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(people, list):
        raise SystemExit(f"{path}: a population is a list of personas.")
    return people


def load_moves(run_dir: Path) -> tuple[list[dict], Optional[str]]:
    """Rows of the run's `moves.csv`, or `(<empty>, reason)`."""
    import csv

    path = run_dir / "moves.csv"
    if not path.exists():
        return [], f"{path} absent"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, None


def traits(person: dict) -> dict:
    return (person.get("identity") or {}).get("traits_json") or {}


def home(person: dict) -> dict:
    return (person.get("identity") or {}).get("home") or {}


# ── Communal classification (the survey's definition) ─────────────────────────

# `CommunalZones` used to live here. It moved up into `mobility_core.residence_zone` in
# ticket 021, lot 1, when a second caller appeared (the post-processing that sets the
# ring on the persona): two copies of a reference classification end up
# diverging, and this one defines what is "out of scope". The audit now reads it in the
# same place as production, which is the only way for its verdicts to apply to what
# runs.

# ── Axes ──────────────────────────────────────────────────────────────────────

def axis_a1_age(people: list[dict]) -> Finding:
    ages = [traits(p).get("age") for p in people]
    known = [float(a) for a in ages if isinstance(a, (int, float))]
    missing = len(ages) - len(known)
    under = [a for a in known if a < MIN_AGE]
    share = 100.0 * len(under) / len(known) if known else 0.0
    verdict = CONFORME if not under else A_CORRIGER
    detail = (
        f"Âge minimum observé : {min(known):.0f} ans ; {missing} persona(s) sans âge. "
        "Le contrôle n'est pas gratuit même à zéro : `frames.age_to_cat` teste "
        "`a <= 9`, donc un persona de 3 ans tomberait dans la classe « 5-9 » et serait "
        "comparé à la cible d'une classe dont il ne fait pas partie, sans qu'aucun log "
        "ne le signale. La conformité est ici HÉRITÉE de la chaîne eqasim (les chaînes "
        "d'activités sont appariées sur une enquête qui commence à 5 ans), elle n'est "
        "garantie par aucune assertion.")
    return Finding(
        "A1", "Âge minimum de la population comptée",
        f"population cible = {MIN_AGE} ans et plus",
        f"{len(under)} persona(s) de moins de {MIN_AGE} ans sur {len(known)}",
        f"{share:.2f} pt de population", verdict, detail,
        {"age_min": min(known) if known else None, "n_sous_seuil": len(under),
         "n_total": len(known), "n_sans_age": missing})


def axis_a2_couronnes(people: list[dict], zones: Optional[CommunalZones],
                      moves: list[dict], cerema: dict,
                      run_dir: Optional[Path] = None) -> Finding:
    if zones is None:
        return Finding(
            "A2", "Définition des couronnes", "découpage par liste de communes",
            "—", "—", NON_MESURABLE,
            "Ressource `mobility_core/data/couronne_perimetre.geojson` absente : "
            "`make communes-couronnes` l'exige, et cette cible exige les données "
            "PROGEDO d'accès restreint.")

    # Since tickets 021 and 028, nothing in production classifies by distance any more: the
    # log READS the `residence_zone` trait, terminal time classifies by ring
    # membership and its laws are stratified by the survey table. The axis therefore checks
    # the three doors through which the gap would come back: a missing trait, a trait that
    # does not match the geometry, a terminal time resource still stratified
    # by distance. The geometry (`CommunalZones`) is the INDEPENDENT measurement of the trait.
    confusion: Counter = Counter()
    per_person: dict[str, tuple[str, str]] = {}
    missing = 0
    for person in people:
        h = home(person)
        trait = str(traits(person).get("residence_zone") or "")
        communal = zones.classify(h.get("lat"), h.get("lon"))
        if not trait:
            missing += 1
        confusion[(trait or "∅", communal)] += 1
        per_person[str(person.get("person_id"))] = (trait, communal)

    total = sum(confusion.values())
    mismatched = sum(n for (t, c), n in confusion.items() if t != "∅" and t != c)

    crown_definition = ""
    try:
        crown_definition = str(json.loads(TERMINAL_TIME_JSON.read_text(encoding="utf-8"))
                               ["meta"]["crown_definition"])
    except (OSError, KeyError, ValueError, TypeError):
        crown_definition = ""
    terminal_communal = ("CouronneTable" in crown_definition
                         and "geo_reference" not in crown_definition)

    tables: dict[str, Any] = {
        "confusion_trait_vs_geometrie": {f"{t} → {c}": n
                                         for (t, c), n in sorted(confusion.items())},
        "trait_absent": missing,
        "trait_different_de_la_geometrie": mismatched,
        "temps_terminal_crown_definition": crown_definition or "(ressource illisible)",
    }
    if moves:
        tables["parts_par_zone"] = modal_shares_by_zone(moves, per_person, cerema,
                                                        log_dir=run_dir)

    problems: list[str] = []
    if missing:
        problems.append(f"{missing} persona(s) sans trait `residence_zone`")
    if mismatched:
        problems.append(f"{mismatched} trait(s) qui ne coïncident pas avec la géométrie")
    if not terminal_communal:
        problems.append("temps terminal encore stratifié à la distance "
                        "(`meta.crown_definition`)")

    detail = (
        "Une couronne administrative n'est pas un anneau métrique. Le disque de 8 km "
        "autour du Capitole sort largement de la commune de Toulouse et mord sur "
        "Blagnac, Balma, Colomiers, Tournefeuille et Ramonville — de 1ʳᵉ couronne dans "
        "l'enquête. Le ticket 020 a mesuré 24,4 % de personas reclassés entre les deux "
        "définitions, unidirectionnellement (aucun Toulousain classé dehors). Enjeu "
        "direct : la cible `voiture` vaut 31 % à Toulouse et 64 % en 1ʳᵉ couronne — un "
        "agent mal classé est comparé à une cible qui diffère de plus de 30 points. "
        "Le ticket 021 a posé la couronne SUR le persona ; le ticket 028 a re-stratifié "
        "le temps terminal sur la même table (tt4). L'axe ne remesure pas cet écart "
        "historique : il garde les trois portes fermées.")
    ecart = "aucun écart : trait, géométrie et temps terminal sur le même découpage" \
        if not problems else " ; ".join(problems)
    # The per-zone share sub-table can be meaningless without the three doors
    # of the axis being open: it is enough that the audited population and the run do not
    # share their person identifiers. The axis verdict remains that of the
    # three doors — that is indeed what it measures — but the gap SAYS so, and an [ALARME]
    # fires from `modal_shares_by_zone`.
    void = ((tables.get("parts_par_zone") or {}).get("communal") or {}).get(
        "non_mesurable")
    if void:
        ecart += f" ⚠ parts modales par zone NON MESURABLES : {void}"
    return Finding(
        "A2", "Définition des couronnes",
        "découpage par liste de communes : "
        + " / ".join(f"{n}" for n in couronne_commune_counts().values())
        + " — pour la résidence ET le temps terminal",
        f"trait `residence_zone` posé sur {total - missing}/{total} personas ; temps "
        f"terminal stratifié par {'la table de l’enquête (tt4)' if terminal_communal else 'la distance à l’hypercentre'}",
        ecart, CONFORME if not problems else A_CORRIGER, detail, tables)


def modal_shares_by_zone(moves: list[dict], per_person: dict[str, tuple[str, str]],
                         cerema: dict, log_dir: Optional[Path] = None) -> dict:
    """Modal shares by ring — under the persona's trait and under the geometry —, plus
    the L1 gap to the targets. The two columns must match; the gap between them is
    exactly what axis A2 counts.

    The L1 is computed on the four scored categories, against a target renormalised
    to the same four — the comparison published since ticket 020, unchanged. What
    changes on 2026-09-04 is the path leading to it: a fine label is first
    aggregated to its survey category ("Train" → public transport), and everything
    that does NOT enter the modal shares is counted and named instead of being
    `continue`d. A shrinking denominator raises the remaining shares: it is a
    false score, not a lack of score.
    """
    targets = (cerema.get("parts_modales_2023") or {}).get("lieu_residence") or {}
    out: dict[str, Any] = {}
    for index, label in ((0, "trait"), (1, "communal")):
        mass: dict[str, Counter] = defaultdict(Counter)
        ecarte: Counter = Counter()
        for row in moves:
            category = category_of(row.get("Mode de transport Choisi"))
            if category is None:
                ecarte[UNKNOWN_MODE] += 1
                continue
            if category not in SCORED_MODES:
                # `autres_modes` (survey residual), `non_deplacement` ("Aucun")
                # and the empty cell: outside the four scored categories, but stated.
                ecarte[category] += 1
                continue
            pair = per_person.get((row.get("ID Personne") or "").strip())
            if pair is None:
                ecarte["persona_introuvable"] += 1
                continue
            mass[pair[index]][category] += 1
        # The equality is checked, not hoped for: everything read is either in a
        # zone × mode cell, or in a named discard counter.
        retenus = sum(sum(c.values()) for c in mass.values())
        if retenus + sum(ecarte.values()) != len(moves):
            raise AssertionError(
                f"parts_par_zone[{label}]: {len(moves)} row(s) read, "
                f"{retenus} kept and {sum(ecarte.values())} discarded — "
                "a count got lost, so a denominator is wrong.")
        zones: dict[str, Any] = {}
        for zone, counter in mass.items():
            total = sum(counter.values())
            if not total:
                continue
            shares = {m: 100.0 * counter[m] / total for m in SCORED_MODES}
            node = targets.get(zone.replace(" ", "_"))
            l1 = None
            if node:
                renorm = sum(float(node[m]) for m in SCORED_MODES)
                target = {m: 100.0 * float(node[m]) / renorm for m in SCORED_MODES}
                l1 = sum(abs(shares[m] - target[m]) for m in SCORED_MODES)
            zones[zone] = {"n": total, "shares": shares, "l1": l1}
        weighted = [(z["l1"], z["n"]) for z in zones.values() if z["l1"] is not None]
        # Persona ↔ trip join rate. The join is on "ID
        # Personne": a population and a run that do not share their identifier
        # space join NOTHING, and the original `continue` then returned an
        # L1 computed on a handful of rows — a plausible and meaningless figure,
        # without a word in the log. The rate is therefore published, alarmed, and under
        # `JOIN_VOID_PCT` the table declares itself NOT MEASURABLE instead of returning that figure.
        joinable = retenus + ecarte["persona_introuvable"]
        join_pct = 100.0 * retenus / joinable if joinable else 0.0
        void_reason = ""
        if join_pct < JOIN_VOID_PCT:
            void_reason = (
                f"{ecarte['persona_introuvable']}/{joinable} déplacement(s) sans "
                f"persona correspondant ({100.0 - join_pct:.1f} %) : la population "
                f"auditée et le run ne partagent pas leurs identifiants de personne. "
                f"Les parts par zone porteraient sur {retenus} ligne(s) — non mesurable.")
            for zone in zones.values():
                zone["l1"] = None
            weighted = []
        out[label] = {
            "zones": zones,
            "l1_pondere": (sum(l * n for l, n in weighted) / sum(n for _, n in weighted)
                           if weighted else None),
            # What the table does NOT count, named row by row. Key added on
            # 2026-09-04: before, these rows vanished from the denominator without
            # a trace, and the shares of the remaining modes rose accordingly.
            "hors_parts_modales": dict(ecarte.most_common()),
            "n_lignes_lues": len(moves),
            "n_lignes_retenues": retenus,
            "taux_jointure_persona_pct": join_pct,
            "non_mesurable": void_reason,
        }
        if void_reason and label == "communal":
            log_alarm(f"[ALARME] A2 parts_par_zone : {void_reason}",
                      log_dir=log_dir, logger_name=_ALARM_LOGGER)
        elif ecarte["persona_introuvable"] and label == "communal":
            perdu = 100.0 - join_pct
            if perdu > JOIN_ALARM_PCT:
                log_alarm(
                    f"[ALARME] A2 parts_par_zone : "
                    f"{ecarte['persona_introuvable']}/{joinable} déplacement(s) "
                    f"({perdu:.1f} %) sans persona correspondant — autant de lignes "
                    f"hors du dénominateur des parts par zone.",
                    log_dir=log_dir, logger_name=_ALARM_LOGGER)
    return out


def axis_a3_ponderation(people: list[dict], moves: list[dict]) -> Finding:
    targets = household_targets()
    sizes = [(traits(p).get("household_size"), traits(p).get("number_of_cars"))
             for p in people]
    usable = [(float(s), float(c)) for s, c in sizes
              if isinstance(s, (int, float)) and s and isinstance(c, (int, float))]
    if not usable:
        return Finding("A3", "Base de pondération", "poids COE0 (ménages) / COEP (personnes)",
                       "—", "—", NON_MESURABLE,
                       "Aucun persona ne porte à la fois `household_size` et `number_of_cars`.")

    raw_size = sum(s for s, _ in usable) / len(usable)
    raw_cars = sum(c for _, c in usable) / len(usable)
    raw_zero = 100.0 * sum(1 for _, c in usable if c == 0) / len(usable)
    weights = [household_weight(s) for s, _ in usable]
    mass = sum(weights)
    w_size = sum(w * s for w, (s, _) in zip(weights, usable)) / mass
    w_cars = sum(w * c for w, (_, c) in zip(weights, usable)) / mass
    w_zero = 100.0 * sum(w for w, (_, c) in zip(weights, usable) if c == 0) / mass

    detail = (
        "L'écart brut n'est pas un défaut de la population : c'est un défaut de BASE. "
        "Une population synthétique échantillonne des PERSONNES, donc un ménage de 5 y "
        "apparaît cinq fois et un ménage de 1 une seule ; la moyenne brute d'un attribut "
        "de ménage y est mécaniquement tirée vers les grands ménages. Pondérer chaque "
        "personne par 1/taille rend à chaque ménage un poids de 1 — et l'écart de 30 % "
        "sur la taille de ménage tombe à 3 %. C'est le même raisonnement que le ticket "
        "019 a appliqué à la loi du logement. Les parts modales, elles, sont des "
        "comptages de DÉPLACEMENTS non pondérés : c'est la bonne base pour une cible "
        "`COEP`, à ceci près qu'un persona qui se déplace beaucoup y pèse plus qu'un "
        "sédentaire, ce que le redressement d'enquête corrige et que la simulation ne "
        "corrige pas.")
    return Finding(
        "A3", "Base de pondération des cibles",
        f"taille {targets['taille_moyenne_menage']:.2f} · "
        f"{targets['voitures_par_menage']:.2f} voiture/ménage · "
        f"{targets['sans_voiture_pct']:.0f} % sans voiture (poids COE0)",
        f"brut : {raw_size:.2f} · {raw_cars:.2f} · {raw_zero:.1f} % — "
        f"pondéré ménages : {w_size:.2f} · {w_cars:.2f} · {w_zero:.1f} %",
        f"taille : {raw_size - targets['taille_moyenne_menage']:+.2f} en brut, "
        f"{w_size - targets['taille_moyenne_menage']:+.2f} pondéré",
        A_PUBLIER, detail,
        {"brut": {"taille": raw_size, "voitures": raw_cars, "sans_voiture_pct": raw_zero},
         "pondere_menages": {"taille": w_size, "voitures": w_cars,
                             "sans_voiture_pct": w_zero},
         "cible": targets,
         "n_deplacements_non_ponderes": len(moves)})


def axis_a4_exclusions(people: list[dict], zones: Optional[CommunalZones]) -> Finding:
    if zones is None:
        return Finding("A4", "Populations et déplacements hors périmètre",
                       "touristes, EHPAD, marchandises exclus ; 95,9 % des "
                       "déplacements internes au périmètre",
                       "—", "—", NON_MESURABLE,
                       "Ressource `couronne_perimetre.geojson` absente.")
    center = hypercenter()
    outside, distances = 0, []
    activities_total = activities_outside = 0
    residents_with_external = 0
    for person in people:
        h = home(person)
        zone = zones.classify(h.get("lat"), h.get("lon"))
        inside_home = zone != OUT_OF_PERIMETER and zone != ""
        if not inside_home:
            outside += 1
            if h.get("lat") is not None:
                distances.append(haversine_km(center[0], center[1], h["lat"], h["lon"]))
        has_external = False
        for activity in (person.get("identity") or {}).get("activities") or []:
            loc = activity.get("location") or {}
            if loc.get("lat") is None:
                continue
            activities_total += 1
            if zones.classify(loc.get("lat"), loc.get("lon")) == OUT_OF_PERIMETER:
                activities_outside += 1
                has_external = True
        if inside_home and has_external:
            residents_with_external += 1
    total = len(people)
    share = 100.0 * outside / total if total else 0.0
    verdict = A_CORRIGER if share > 1.0 else CONFORME
    detail = (
        "L'enquête ne compte QUE le périmètre de 453 communes : ni touristes, ni "
        "EHPAD, ni marchandises, et 95,9 % de ses déplacements pondérés sont internes "
        "au périmètre (le recalcul sur les internes seuls redonne exactement la cible "
        "publiée de 55 % voiture). Un domicile hors périmètre n'a donc AUCUNE cible à "
        "laquelle se comparer — et le classement métrique lui en donne une quand même, "
        "celle de la 3ᵉ couronne, parce que « au-delà de 40 km » n'a pas de borne "
        "supérieure. C'est le mécanisme exact du motif de vacuité : l'absence de "
        "périmètre produit une classification, pas une erreur.")
    return Finding(
        "A4", "Populations et déplacements hors périmètre",
        "453 communes ; 95,9 % des déplacements internes au périmètre",
        f"{outside} domicile(s) hors périmètre sur {total}",
        f"{share:.1f} % des personas"
        + (f", jusqu'à {max(distances):.0f} km du Capitole" if distances else ""),
        verdict, detail,
        {"n_hors_perimetre": outside, "n_total": total,
         "distance_max_km": max(distances) if distances else None,
         "distance_min_km": min(distances) if distances else None,
         "activites_hors_perimetre_pct":
             100.0 * activities_outside / activities_total if activities_total else None,
         "residents_avec_activite_externe": residents_with_external})


def axis_a5_saison(moves: list[dict]) -> Finding:
    debut, fin = survey_window()
    md_start, md_end = debut[5:], fin[5:]

    def in_window(month_day: str) -> bool:
        return month_day >= md_start or month_day <= md_end

    rows = _read_weather()
    window_stats = year_stats = None
    if rows:
        year_stats = _rain_stats(rows)
        window_stats = _rain_stats([r for r in rows if in_window(r["md"])])

    dates = sorted({(row.get("Heure de départ") or "")[:10] for row in moves
                    if (row.get("Heure de départ") or "")[:10]})
    in_win = [d for d in dates if in_window(d[5:])]
    rain_share = None
    if moves:
        wet = sum(1 for row in moves
                  if _to_float(row.get("Météo Précipitations (mm)")) > 0)
        rain_share = 100.0 * wet / len(moves)

    detail = (
        "CE QUE L'ENQUÊTE MESURE, vérifié deux fois. La méthode EMC² recueille les "
        "« déplacements de la VEILLE » (passation du mardi au samedi hors fériés et "
        "vacances scolaires, jour de référence du lundi au vendredi) : elle n'interroge "
        "personne sur ses habitudes annuelles. Les dates de référence des microdonnées "
        "le confirment — seuls les mois 09 à 12 de 2022 et 01 à 02 de 2023 y "
        "apparaissent, aucune observation de mars à août, et le jour de référence est "
        "toujours ouvré. Les cibles sont donc bien des déplacements d'automne-hiver.\n\n"
        "MAIS CE QU'ELLE PUBLIE est « un jour moyen de semaine ». La fenêtre "
        "automne-hiver et l'exclusion des congés sont le MOYEN d'obtenir une journée "
        "ordinaire, pas une revendication saisonnière. L'écart n'est donc pas « cible "
        "d'automne contre simulation de printemps » : c'est un écart de MOYENNAGE, et il "
        "se sépare en deux.\n\n"
        "(1) LES JEUX GELÉS MOYENNENT, mais sur la mauvaise fenêtre. Un jour est tiré "
        "indépendamment par décision, sur 365 jours : la pluie y est donc bien "
        "représentée (42,5 % contre 44,7 % en fenêtre), et l'écart est THERMIQUE — 18,0 "
        "contre 12,7 °C à midi. Restreindre le tirage à la fenêtre d'enquête corrige "
        "cet écart-là, et lui seul.\n\n"
        "(2) UN RUN NE MOYENNE PAS. Il rejoue des jours calendaires consécutifs réels — "
        "ici cinq jours de mi-mars, tous secs. Thermiquement ces jours sont TYPIQUES de "
        "la fenêtre d'enquête (14,6 °C à midi, chacun entre le 56ᵉ et le 81ᵉ centile de "
        "sa distribution), et mi-mars est une semaine scolaire ordinaire : le grief "
        "calendaire est faible. Le grief réel est qu'une réalisation de 5 jours est "
        "comparée à une moyenne de 152 jours. Et 0 % de pluie n'est pas un tirage "
        "exotique qu'il suffirait d'éviter : 27,7 % des fenêtres de 5 jours consécutifs "
        "de la période d'enquête sont elles aussi entièrement sèches. AUCUN choix de "
        "jours ne rend un run de 5 jours comparable à la moyenne sur le mode le plus "
        "sensible à la météo — le vélo, dont les mouvements de 4 à 5 points ont déjà "
        "arbitré les tickets 013 et 014. C'est une limite de variance, à publier, pas un "
        "réglage à trouver.")
    simule = (f"{len(dates)} jour(s) simulé(s) : {', '.join(dates)}" if dates
              else "aucun run exploitable")
    if rain_share is not None:
        simule += f" ; {rain_share:.1f} % des trajets sous la pluie"
    ecart = "—"
    if window_stats and year_stats:
        ecart = (f"pluie : {window_stats['rain_pct']:.1f} % de jours en fenêtre "
                 f"d'enquête, {year_stats['rain_pct']:.1f} % sur l'année tirée par les "
                 f"jeux gelés, {rain_share:.1f} % dans ce run — et T° midi "
                 f"{window_stats['temp_midi_moy']:.1f} contre "
                 f"{year_stats['temp_midi_moy']:.1f} °C"
                 if rain_share is not None else
                 f"jours pluvieux : {window_stats['rain_pct']:.1f} % contre "
                 f"{year_stats['rain_pct']:.1f} %")
    return Finding(
        "A5", "Fenêtre saisonnière de la mesure",
        f"{debut} → {fin}, hors vacances scolaires", simule, ecart,
        # Verdict A_PUBLIER even if the simulated days fall INSIDE the window: the
        # main grievance is the variance of a run of a few days against an average
        # of 152, and changing the dates does not remove it.
        A_PUBLIER, detail,
        {"jours_simules": dates, "jours_dans_la_fenetre": in_win,
         "fenetre": window_stats, "annee": year_stats,
         "part_trajets_sous_la_pluie": rain_share})


def _read_weather() -> list[dict]:
    if not WEATHER_CSV.exists():
        return []
    import csv

    out = []
    with WEATHER_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = (row.get("DATE") or "")
            if len(stamp) < 10:
                continue
            out.append({"md": stamp[5:10],
                        "precip": _to_float(row.get("PRECIP_TOTAL_DAY_MM")),
                        "noon": _to_float(row.get("TEMPERATURE_NOON_C_12H"))})
    return out


def _rain_stats(rows: list[dict]) -> Optional[dict]:
    if not rows:
        return None
    return {"n_jours": len(rows),
            "rain_pct": 100.0 * sum(1 for r in rows if r["precip"] > 0) / len(rows),
            "precip_moy_mm": sum(r["precip"] for r in rows) / len(rows),
            "temp_midi_moy": sum(r["noon"] for r in rows) / len(rows)}


def axis_a6_jour(moves: list[dict]) -> Finding:
    if not moves:
        return Finding("A6", "Jour de la semaine",
                       "veille enquêtée = lundi à vendredi", "—", "—", NON_MESURABLE,
                       "Aucun `moves.csv` exploitable.")
    days: Counter = Counter()
    for row in moves:
        stamp = (row.get("Heure de départ") or "")[:10]
        if len(stamp) != 10:
            continue
        try:
            days[date.fromisoformat(stamp).isoweekday()] += 1
        except ValueError:
            continue
    noms = {1: "lundi", 2: "mardi", 3: "mercredi", 4: "jeudi", 5: "vendredi",
            6: "samedi", 7: "dimanche"}
    weekend = sum(n for d, n in days.items() if d >= 6)
    total = sum(days.values()) or 1
    expected = set(surveyed_weekdays())
    detail = (
        "Le garde-fou existe et il est actif : `no_weekend_departures` reporte tout "
        "départ de samedi ou dimanche au lundi suivant à la même heure. Il n'est "
        "jamais exercé sur les runs courants, qui démarrent un lundi et durent au plus "
        "cinq jours. Deux nuances mesurées à l'intérieur d'EMC² : les parts modales ne "
        "varient que de 1,3 point au plus entre les cinq jours ouvrés (voiture 54,5 à "
        "55,8 %), donc un run mono-journalier ne biaise pas les PARTS ; mais le lundi "
        "porte 3,16 déplacements par personne contre 3,51 le mercredi, soit 10 % de "
        "volume en moins — l'écart compterait si un VOLUME était un jour comparé. "
        "Attention au report : sur un run de plus de cinq jours, il empilerait les "
        "départs de week-end sur le lundi et fabriquerait un lundi atypique.")
    return Finding(
        "A6", "Jour de la semaine",
        f"veille enquêtée = jours {sorted(expected)} (lundi à vendredi), "
        "répartis à peu près uniformément",
        " · ".join(f"{noms[d]} {100.0 * n / total:.0f} %"
                  for d, n in sorted(days.items())),
        f"{100.0 * weekend / total:.1f} % de trajets de week-end",
        CONFORME if not weekend and set(days) <= expected else A_CORRIGER,
        detail, {"trajets_par_jour_semaine": {noms[d]: n
                                             for d, n in sorted(days.items())}})


def axis_a7_objet_compte(moves: list[dict], cerema: dict,
                         run_dir: Optional[Path] = None) -> Finding:
    if not moves:
        return Finding("A7", "Objet compté : le déplacement à mode principal",
                       "un déplacement = un mode principal", "—", "—", NON_MESURABLE,
                       "Aucun `moves.csv` exploitable.")
    trip_ids = [row.get("ID Trajet") for row in moves if row.get("ID Trajet")]
    unique = len(set(trip_ids))
    # The DETAILED count, by label as it appears in `moves.csv`, plus the
    # table that aggregates it to the survey categories. Both levels are
    # published because they do not answer the same question: the detail says what
    # the simulation produced (how much train, how many non-trips), the
    # category says what to compare it with. A label outside the table is counted under
    # `libelle_inconnu` and raises an [ALARME] in the run's app.log.
    tally = tally_labels((row.get("Mode de transport Choisi") for row in moves),
                         source="moves.csv · Mode de transport Choisi",
                         log_dir=run_dir)
    modes = tally.detail
    shares = tally.shares(SCORED_MODES)
    global_target = (cerema.get("parts_modales_2023") or {}).get("global") or {}
    l1_global = None
    cible: Optional[dict[str, float]] = None
    if all(m in global_target for m in SCORED_MODES):
        renorm = sum(float(global_target[m]) for m in SCORED_MODES)
        cible = {m: 100.0 * float(global_target[m]) / renorm for m in SCORED_MODES}
        l1_global = sum(abs(shares[m] - cible[m]) for m in SCORED_MODES)
    detail = (
        "Deux questions distinctes, et les réponses ne vont pas dans le même sens.\n\n"
        "CE QUI EST CONFORME. Une ligne de `moves.csv` est bien un DÉPLACEMENT, pas une "
        "jambe : les jambes terminales du ticket 013 portent `is_transfer=True` et "
        "`_plan_transport_mode` ne regarde que les jambes non-transfert, donc la marche "
        "d'accès à une voiture ou à un bus n'est jamais comptée comme un déplacement à "
        "pied. L'enquête fait la même chose, et c'est vérifié dans ses microdonnées : "
        "AUCUN de ses déplacements en voiture ou en transports collectifs ne porte de "
        "trajet à pied — l'accès y est une DURÉE (T2/T6), pas un trajet. La marche n'est "
        "donc pas surestimée par construction.\n\n"
        "CE QUI DIVERGEAIT, ET NE DIVERGE PLUS. La hiérarchie de mode principal était "
        "INVERSÉE : `_plan_transport_mode` testait la voiture AVANT les transports "
        "collectifs, alors que l'enquête fait le contraire — sur ses 770 déplacements "
        "mêlant voiture et transports collectifs, 760 sont codés « transports "
        "collectifs » et 10 seulement « voiture » (757 sur 767 en recomptant avec les "
        "listes de modes complètes). Refermé le 2026-09-04 par le ticket 022, et par la "
        "source plutôt que par une convention : le rapport publie en annexe p. 53 la "
        "hiérarchie des 36 modes enquêtés, gelée dans "
        "`mobility_core/data/mode_hierarchy_emc2.json` et servie par "
        "`mobility_core.mode_hierarchy` à toutes les tables du dépôt — "
        "`_plan_transport_mode` appelle `primary_label`, il n'y a plus de cascade de "
        "`if`. Un mode que la hiérarchie ne connaît pas lève une [ALARME] au lieu d'être "
        "absorbé. Mais l'effet miroir reste ENTIER : OTP est interrogé mode par mode, "
        "donc aucun itinéraire simulé ne mêle voiture et transports collectifs, et la "
        "simulation ne peut STRUCTURELLEMENT pas produire les 1,4 point de rabattement "
        "que la cible compte en transports collectifs — c'est l'objet des lots 2 à 5 du "
        "ticket 022.\n\n"
        "DEUX NIVEAUX DE LECTURE, ET LA TABLE POUR PASSER DE L'UN À L'AUTRE. Le "
        "journal écrit des libellés FINS (Marche, Vélo, Voiture Privée, "
        "Transports_collectifs, Train, Deux-roues motorisé, Autres modes, Aucun) ; la "
        "référence publie des CATÉGORIES (marche, velo, voiture, "
        "transports_collectifs, autres_modes). L'audit rend les deux et la table qui "
        "les relie, de sorte que l'agrégat se recompose depuis le détail — le train "
        "va avec les transports collectifs, les deux-roues motorisés avec le résidu "
        "`autres_modes` de l'enquête. Jusqu'au 2026-09-04, la table de l'audit "
        "n'avait que quatre entrées et pas de « Train » : depuis le routage du TER "
        "(16,7 % des itinéraires portent un train), un déplacement en train sortait "
        "des parts modales par un `continue` MUET. Le dénominateur baissait, les "
        "parts des autres modes montaient, et rien ne le disait. Deux valeurs du "
        "journal ne sont pas des déplacements et n'entrent donc dans aucune part "
        "modale — « Aucun » (même localisation, l'agent n'a pas bougé) et la cellule "
        "vide (aucun itinéraire) : elles sont désormais comptées et nommées plutôt "
        "que jetées. Un libellé hors table est compté sous « libelle_inconnu », "
        "nommé, et lève une [ALARME] dans l'app.log du run — visible par "
        "`make error`.")
    ecart = ("0 déplacement mal classé aujourd'hui ; 1,4 pt de rabattement "
             "inatteignable")
    if l1_global is not None:
        ecart += (f" ; L1 aux cibles globales sur les 4 catégories scorées "
                  f"{l1_global:.1f} pt (comptages non redressés, cf. A3)")
    verdict = A_PUBLIER
    if tally.unknown:
        named = ", ".join(f"« {k} » ({n})" for k, n in tally.unknown.most_common())
        ecart += (f" ; {tally.n_unknown} ligne(s) sous un libellé hors table "
                  f"d'agrégation : {named}")
        verdict = A_CORRIGER
    if tally.hierarchy_gap:
        # LATENT defect: the aggregation table does not cover the whole mode hierarchy
        # of the repository. Nothing is lost in THIS run, everything will be lost in the first run that
        # carries the missing mode — that is exactly the "Train" story.
        ecart += f" ⚠ {tally.hierarchy_gap}"
        verdict = A_CORRIGER
    return Finding(
        "A7", "Objet compté : le déplacement à mode principal",
        "un déplacement, un mode principal ; hiérarchie plaçant les transports "
        "collectifs au-dessus de la voiture (760/770 déplacements mixtes) ; parts "
        "publiées par catégorie : "
        + " · ".join(f"{m} {float(global_target[m]):.0f} %" for m in SURVEY_CATEGORIES
                     if m in global_target),
        f"{len(moves)} lignes pour {unique} identifiants de trajet distincts ; "
        f"aucune ligne de jambe ; {tally.n_trips} déplacement(s) en "
        f"{len(tally.detail)} libellé(s) fin(s) → "
        + " · ".join(f"{m} {shares[m]:.1f} %" for m in SCORED_MODES),
        ecart,
        verdict, detail,
        {"n_lignes": len(moves), "n_trajets_uniques": unique,
         # Historical key, unchanged: the count by raw label.
         "modes": dict(modes.most_common()),
         # The publishable detail — a label, its category, its count, its share.
         "modes_detail": tally.detail_rows(),
         # The aggregation table itself: publishing the result without the table does
         # not allow checking the aggregation, publishing both does.
         "agregation_libelle_vers_categorie": aggregation_table(),
         # The aggregate, on the survey categories then on the 4 scored ones.
         "effectifs_par_categorie": dict(tally.categories.most_common()),
         "parts_par_categorie_enquete": tally.shares(SURVEY_CATEGORIES),
         "parts_par_categorie_scoree": shares,
         "cible_globale_scoree_pct": cible,
         "l1_global_scorees_pt": l1_global,
         # What does not enter a modal share, and the invariant that guarantees it.
         "hors_parts_modales": {c: n for c, n in tally.categories.items()
                                if c not in SURVEY_CATEGORIES},
         "libelles_inconnus": dict(tally.unknown.most_common()),
         # Coverage of the aggregation table by the repository's mode hierarchy
         # (`mobility_core.mode_hierarchy`, ticket 022): "" = all covered.
         "couverture_hierarchie_modes": tally.hierarchy_gap or "complète",
         "invariant_total": {"n_lignes_lues": tally.total,
                             "somme_detail": sum(tally.detail.values()),
                             "somme_categories": sum(tally.categories.values()),
                             "verifie": True},
         "part_rabattement_dans_cible_tc_pct": 11.5,
         "part_rabattement_absolue_pct": 1.41})


def axis_a8_menages(people: list[dict]) -> Finding:
    clusters: dict[tuple, list[dict]] = defaultdict(list)
    for person in people:
        h = home(person)
        if h.get("lat") is None:
            continue
        clusters[(round(float(h["lat"]), 6), round(float(h["lon"]), 6))].append(person)
    if not clusters:
        return Finding("A8", "Structure de ménage", "2,08 personnes par ménage", "—",
                       "—", NON_MESURABLE, "Aucun domicile géolocalisé.")
    complete = incomplete = collisions = 0
    declared = present = 0
    for members in clusters.values():
        sizes = {traits(m).get("household_size") for m in members}
        if len(sizes) > 1:
            collisions += 1
        size = traits(members[0]).get("household_size") or 0
        declared += size
        present += len(members)
        if len(members) == size:
            complete += 1
        elif len(members) < size:
            incomplete += 1
    target = household_targets()["taille_moyenne_menage"]
    declared_mean = declared / len(clusters)
    present_mean = present / len(clusters)
    detail = (
        "La taille de ménage DÉCLARÉE est juste : la moyenne par adresse tombe sur la "
        "cible. Ce qui manque, ce sont des MEMBRES : environ un membre déclaré sur dix "
        "n'existe pas comme persona, et un quart des grappes est incomplet. Deux "
        "conséquences à ne pas confondre. Sur les cibles de ménage, aucune : elles se "
        "lisent sur la taille déclarée, qui est correcte. Sur tout ce qui dépend des "
        "CO-RÉSIDENTS — partage de voiture du foyer, verrous de chaîne, attribution de "
        "vélo — l'effet est réel, et c'est le mécanisme déjà documenté par le ticket 015.")
    return Finding(
        "A8", "Structure de ménage",
        f"{target:.2f} personnes par ménage ; 674 000 ménages",
        f"{declared_mean:.2f} déclarée par adresse, {present_mean:.2f} réellement "
        f"présente ; {complete}/{len(clusters)} grappes complètes",
        f"{declared_mean - target:+.2f} sur la taille déclarée ; "
        f"{100.0 * (declared - present) / declared:.1f} % de membres absents",
        A_PUBLIER, detail,
        {"n_adresses": len(clusters), "grappes_completes": complete,
         "grappes_incompletes": incomplete, "collisions_adresse": collisions,
         "taille_declaree_moyenne": declared_mean,
         "membres_presents_moyenne": present_mean,
         "membres_absents_pct": 100.0 * (declared - present) / declared})


def axis_a9_spatial(people: list[dict], zones: Optional[CommunalZones]) -> Finding:
    if zones is None:
        return Finding("A9", "Représentativité spatiale", "70 % en Toulouse + 1ʳᵉ couronne",
                       "—", "—", NON_MESURABLE,
                       "Ressource `couronne_perimetre.geojson` absente.")
    target = couronne_population_shares()
    # The ring geometry is the independent measurement; since tickets 021 and
    # 028 it is also what the simulation publishes (trait) and charges (terminal time), and
    # axis A2 checks that all three match. There is thus no longer a "published
    # concentration" distinct from the real concentration.
    counts: Counter = Counter()
    for person in people:
        h = home(person)
        counts[zones.classify(h.get("lat"), h.get("lon"))] += 1
    inside = sum(n for z, n in counts.items() if z in COURONNES)
    observed = {z: 100.0 * counts.get(z, 0) / inside for z in COURONNES} if inside else {}
    # Urban core = the first two categories of COURONNES, taken AT THE
    # SOURCE. Hard-coding them ("1ere couronne") would have raised a KeyError in ticket 074;
    # hard-coding them in English would raise it again at the next switch.
    _coeur = COURONNES[:2]
    core_target = sum(target[z] for z in _coeur)
    core_observed = sum(observed.get(z, 0) for z in _coeur)
    l1 = sum(abs(observed.get(z, 0) - target[z]) for z in COURONNES)
    detail = (
        "Une surconcentration en cœur d'agglomération tire mécaniquement la part "
        "voiture vers le bas, sans qu'aucun modèle de choix ne soit en cause : la cible "
        "voiture vaut 31 % à Toulouse et 71 à 74 % dans les couronnes externes. L'écart "
        "est un écart de CADRE DE TIRAGE (Haute-Garonne : 346 des 453 communes, la 3ᵉ "
        "couronne plafonne à 10,6 % de la population pour 15,4 % dans l'enquête) — il se "
        "referme par une sélection stratifiée sur un vivier assez large, pas par un "
        "meilleur classement.")
    return Finding(
        "A9", "Représentativité spatiale de l'échantillon",
        " · ".join(f"{z} {target[z]:.1f} %" for z in COURONNES),
        " · ".join(f"{z} {observed.get(z, 0):.1f} %" for z in COURONNES),
        f"Toulouse + 1ʳᵉ couronne : {core_observed:.1f} % contre {core_target:.1f} % "
        f"cible (L1 = {l1:.1f} pt) ; {counts.get(OUT_OF_PERIMETER, 0)} domicile(s) hors "
        "périmètre exclus du calcul",
        A_PUBLIER if l1 > 4 else CONFORME, detail,
        {"cible": target, "observe": observed, "l1": l1,
         "coeur_cible_pct": core_target, "coeur_reel_pct": core_observed,
         "n_dans_le_perimetre": inside,
         "n_hors_perimetre": counts.get(OUT_OF_PERIMETER, 0)})


def _to_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0



# ── Cross-check of the framing from the microdata ─────────────────────────────
# The `population_reference` loader checks that the framing is CONSISTENT with itself.
# It cannot check that it is true: that requires recomputing each value
# from the survey microdata. A figure copied from a publication and a
# recomputed figure are two independent measurements — the cross-check is only worth anything if
# they are. Hence this separate mode, which requires the restricted-access data.

PROGEDO_STD = (REPO_ROOT / "data" / "PROGEDO 2023" / "lil-1750-Donnees_CSV"
               / "fichiers_standards")
SIG_DTIR = (REPO_ROOT / "data" / "PROGEDO 2023" / "lil-1750-Documentation" / "SIG"
            / "EMC2_Toulouse_2023_DTIR_17072023.shp")

# Grouping of `MODP` categories (main mode of the trip) into the four
# scored modes. The codes come from the survey dictionary (`MODPf`).
_MODP_GROUPS = {
    "marche": {"01"},
    "velo": {"10", "11", "12", "17", "18"},
    "voiture": {"21", "22", "61", "62", "71", "81", "82"},
    "transports_collectifs": {"31", "32", "33", "34", "37", "38", "39",
                              "41", "42", "43", "51", "52", "53", "54"},
}


def _modp_group(code: Any) -> str:
    code = str(code).zfill(2)
    for group, codes in _MODP_GROUPS.items():
        if code in codes:
            return group
    return "autres"


def recompute_from_microdata() -> dict:
    """Recomputes the framing values from EMC², and compares them with the YAML.

    Returns `{"disponible": False, "raison": ...}` when the microdata are missing —
    this is the NORMAL case on a workstation or container without ProGEDO access.
    """
    if not PROGEDO_STD.exists() or not SIG_DTIR.exists():
        return {"disponible": False,
                "raison": f"microdonnées absentes ({PROGEDO_STD}) — accès restreint lil-1750"}
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as exc:
        return {"disponible": False, "raison": f"pandas/geopandas requis : {exc}"}

    dtir = gpd.read_file(SIG_DTIR)
    couronne_of = dict(zip(dtir["NUM_DTIR"].astype(str), dtir["NOM_D2"]))

    men = pd.read_csv(PROGEDO_STD / "Toulouse_2023_std_men.csv", dtype=str)
    per = pd.read_csv(PROGEDO_STD / "Toulouse_2023_std_pers.csv", dtype=str)
    dep = pd.read_csv(PROGEDO_STD / "Toulouse_2023_std_depl.csv", dtype=str)
    men["COE0"] = pd.to_numeric(men["COE0"], errors="coerce")
    per["COEP"] = pd.to_numeric(per["COEP"], errors="coerce")

    # `NOM_D2` is the survey's GIS layer: it says "1ere couronne". The platform's
    # category is English since ticket 074 — we canonicalise ON READ,
    # otherwise the join with the target finds no row and the cross-check, which is
    # the whole point of this function, happens between two foreign vocabularies.
    per["couronne"] = (per["ZFP"].str[:3].map(couronne_of).map(couronne_canonique))
    par_couronne = per.groupby("couronne")["COEP"].sum()
    population_5plus = float(par_couronne.sum())

    men["cars"] = pd.to_numeric(men["M6"], errors="coerce").fillna(0)
    poids = men["COE0"]
    menages = float(poids.sum())

    def part(mask) -> float:
        return 100.0 * float((poids * mask.astype(float)).sum()) / menages

    # Modal shares: COEP-weighted, restricted to INTERNAL trips (TYPD = 1),
    # which is the definition the published target reproduces.
    joined = dep.merge(per[["ZFP", "ECH", "PER", "COEP"]].rename(columns={"ZFP": "ZFD"}),
                       on=["ZFD", "ECH", "PER"], how="left")
    joined["groupe"] = joined["MODP"].map(_modp_group)
    interne = joined[joined["TYPD"] == "1"]
    masse = interne.groupby("groupe")["COEP"].sum()
    parts_internes = {g: 100.0 * float(masse.get(g, 0.0)) / float(masse.sum())
                      for g in list(_MODP_GROUPS) + ["autres"]}
    masse_tous = joined.groupby("groupe")["COEP"].sum()
    parts_tous = {g: 100.0 * float(masse_tous.get(g, 0.0)) / float(masse_tous.sum())
                  for g in list(_MODP_GROUPS) + ["autres"]}
    localisation = joined.groupby("TYPD")["COEP"].sum()
    localisation_pct = {t: 100.0 * float(v) / float(localisation.sum())
                        for t, v in localisation.items()}

    reference = population_reference()
    totaux = reference["population"]["totaux_perimetre_2023"]
    equip = reference["menages_equipement_voiture"]["perimetre_2023"]
    cible_couronnes = couronne_population_shares()

    lignes = [
        ("ménages enquêtés", len(men),
         reference["enquete"]["echantillon"]["menages_enquetes"]),
        ("personnes interrogées (PENQ=1)", int((per["PENQ"] == "1").sum()),
         reference["enquete"]["echantillon"]["personnes_interrogees"]),
        ("déplacements recensés", len(dep),
         reference["enquete"]["echantillon"]["deplacements_recenses"]),
        ("secteurs de tirage", len(dtir),
         reference["enquete"]["echantillon"]["secteurs_de_tirage"]),
        ("habitants de 5 ans et + (milliers)", round(population_5plus / 1000),
         round(totaux["habitants_5_ans_et_plus"] / 1000)),
        ("ménages (milliers)", round(menages / 1000),
         round(totaux["nombre_menages"] / 1000)),
        ("voitures par ménage", round(float((poids * men["cars"]).sum()) / menages, 2),
         equip["voitures_par_menage_moyen"]),
        ("ménages sans voiture (%)", round(part(men["cars"] == 0), 1),
         equip["repartition_motorisation"]["sans_voiture"]),
        ("ménages à une voiture (%)", round(part(men["cars"] == 1), 1),
         equip["repartition_motorisation"]["une_voiture"]),
        ("ménages à 2 voitures et + (%)", round(part(men["cars"] >= 2), 1),
         equip["repartition_motorisation"]["deux_voitures_et_plus"]),
        ("déplacements internes au périmètre (%)",
         round(localisation_pct.get("1", 0.0), 2),
         reference["enquete"]["localisation_deplacements"]["interne_au_perimetre"]),
    ]
    for zone, cible in cible_couronnes.items():
        # `par_couronne` and `cible_couronnes` both come from `population_reference`,
        # hence from the same canonical category: the identity mapping table that
        # lived here only masked a possible disagreement behind a KeyError.
        observe = 100.0 * float(par_couronne.get(zone, 0.0)) / population_5plus
        lignes.append((f"part de population — {zone} (%)", round(observe, 1),
                       round(cible, 1)))

    controles = [{"grandeur": nom, "recalcule": recalc, "cadrage": cadre,
                  "ecart": (round(float(recalc) - float(cadre), 2)
                            if isinstance(recalc, (int, float))
                            and isinstance(cadre, (int, float)) else None)}
                 for nom, recalc, cadre in lignes]
    return {
        "disponible": True,
        "controles": controles,
        "parts_modales_recalculees": {"internes": parts_internes, "tous": parts_tous},
        "localisation_deplacements_pct": localisation_pct,
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

def run_audit(population_path: Path, run_dir: Path,
              recompute: Optional[dict] = None) -> dict:
    population_reference()          # raises if the framing is missing or inconsistent
    people = load_population(population_path)
    moves, moves_error = load_moves(run_dir)
    cerema = yaml.safe_load(CEREMA_VALUES.read_text(encoding="utf-8"))

    zones: Optional[CommunalZones] = None
    zones_error = None
    if COURONNE_GEOJSON.exists():
        try:
            zones = CommunalZones.load(COURONNE_GEOJSON)
        except RuntimeError as exc:
            zones_error = str(exc)
    else:
        zones_error = f"{COURONNE_GEOJSON} absent (make communes-couronnes)"

    findings = [
        axis_a1_age(people),
        axis_a2_couronnes(people, zones, moves, cerema, run_dir),
        axis_a3_ponderation(people, moves),
        axis_a4_exclusions(people, zones),
        axis_a5_saison(moves),
        axis_a6_jour(moves),
        axis_a7_objet_compte(moves, cerema, run_dir),
        axis_a8_menages(people),
        axis_a9_spatial(people, zones),
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticket": "020",
        "recoupement_microdonnees": recompute if recompute is not None else None,
        "inputs": {
            "population": str(population_path.relative_to(REPO_ROOT)
                              if population_path.is_relative_to(REPO_ROOT)
                              else population_path),
            "n_personas": len(people),
            "run": str(run_dir.relative_to(REPO_ROOT)
                       if run_dir.is_relative_to(REPO_ROOT) else run_dir),
            "run_resolved": str(run_dir.resolve().name),
            "n_moves": len(moves),
            "moves_error": moves_error,
            "zones_error": zones_error,
        },
        "findings": [f.as_dict() for f in findings],
        "verdicts": dict(Counter(f.verdict for f in findings)),
    }


def _print_mode_detail(tables: dict) -> None:
    """The detail by label, its category, and the aggregate — one under the other.

    Deliberately rendered as two adjacent blocks: the reader must be able to add up
    the detail rows and get back the category row. An aggregation one
    cannot redo by hand is an aggregation one takes on faith.
    """
    rows = tables.get("modes_detail")
    if not rows:
        return
    print("   detail by mode label (as written in moves.csv):")
    print(f"      {'libellé':24s} {'→ catégorie':24s} {'n':>7s} {'part':>7s}")
    for row in rows:
        print(f"      {row['libelle']:24s} {row['categorie']:24s} "
              f"{row['n']:7d} {row['part_pct']:6.1f} %")
    survey = tables.get("parts_par_categorie_enquete") or {}
    if survey:
        print("   aggregated to the 5 survey categories: "
              + " · ".join(f"{k} {v:.1f} %" for k, v in survey.items()))
    parts = tables.get("parts_par_categorie_scoree") or {}
    cible = tables.get("cible_globale_scoree_pct") or {}
    if parts:
        print("   then restricted to the 4 scored categories, against the overall target "
              "renormalised to the same 4:")
        for mode, value in parts.items():
            attendu = (f"  cible {cible[mode]:5.1f} %  écart {value - cible[mode]:+5.1f} pt"
                       if mode in cible else "")
            print(f"      {mode:24s} {value:6.1f} %{attendu}")
    hors = tables.get("hors_parts_modales") or {}
    if hors:
        print("   outside modal shares, counted and named: "
              + " · ".join(f"{k} {v}" for k, v in hors.items()))
    invariant = tables.get("invariant_total") or {}
    if invariant:
        print(f"   invariant checked: {invariant['n_lignes_lues']} row(s) read = "
              f"{invariant['somme_detail']} in detail = "
              f"{invariant['somme_categories']} in categories")


def print_report(report: dict) -> None:
    inputs = report["inputs"]
    print("═" * 78)
    print("SCOPE AUDIT — EMC² surveyed population against simulated population")
    print(f"ticket 020 · {report['generated_at']}")
    print("═" * 78)
    print(f"population : {inputs['population']} ({inputs['n_personas']} personas)")
    print(f"run        : {inputs['run']} → {inputs['run_resolved']} "
          f"({inputs['n_moves']} trips)")
    for key in ("moves_error", "zones_error"):
        if inputs.get(key):
            print(f"⚠ {inputs[key]}")
    print()
    for finding in report["findings"]:
        print("─" * 78)
        print(f"{finding['axe']} · {finding['titre']}   [{finding['verdict'].upper()}]")
        print(f"   survey   : {finding['enquete']}")
        print(f"   simulated: {finding['simule']}")
        print(f"   gap      : {finding['ecart']}")
        _print_mode_detail(finding.get("tables") or {})
        if finding["detail"]:
            for line in finding["detail"].split("\n"):
                print(f"   │ {line}" if line else "   │")
    recoupement = report.get("recoupement_microdonnees")
    if recoupement:
        print("─" * 78)
        print("FRAMING CROSS-CHECK — each value recomputed from EMC²")
        if not recoupement.get("disponible"):
            print(f"   not available: {recoupement.get('raison')}")
        else:
            print(f"   {'grandeur':40s} {'recalculé':>12s} {'cadrage':>10s} {'écart':>8s}")
            for row in recoupement["controles"]:
                ecart = "—" if row["ecart"] is None else f"{row['ecart']:+.2f}"
                print(f"   {row['grandeur']:40s} {str(row['recalcule']):>12s} "
                      f"{str(row['cadrage']):>10s} {ecart:>8s}")
            parts = recoupement["parts_modales_recalculees"]
            print("   recomputed modal shares (COEP-weighted):")
            for scope, values in parts.items():
                rendu = " · ".join(f"{k} {v:.1f} %" for k, v in values.items())
                print(f"      {scope:9s} {rendu}")
    print("─" * 78)
    print("Verdicts: " + " · ".join(f"{k} : {v}" for k, v in
                                     sorted(report["verdicts"].items())))
    if NON_MESURABLE in report["verdicts"]:
        print("⚠ An unmeasured axis is an axis that passes. The axes above marked "
              "\"non mesurable\" are NOT compliant: they are unknown.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=Path, default=None,
                        help="population to audit; by default that of the run "
                             f"(<run>/{DEFAULT_POPULATION_DANS_LE_RUN})")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--trace", type=Path, default=None,
                        help="archive folder (docs/traces/...); the JSON is written there")
    parser.add_argument("--json", action="store_true", help="output only the JSON")
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the framing values from the EMC² microdata "
                             "(restricted access) and compare them with the YAML")
    args = parser.parse_args(argv)

    population = args.population
    if population is None:
        population = args.run / DEFAULT_POPULATION_DANS_LE_RUN
        if not population.exists():
            # No fallback to the reference population: auditing a file other than the one
            # that ran is exactly the defect being closed, and doing it silently would be
            # worse than stopping.
            raise SystemExit(
                f"population of the run not found: {population}\n"
                f"The run did not store its population ({DEFAULT_POPULATION_DANS_LE_RUN}). "
                f"Name the file to audit with --population, knowing that the audit "
                f"will then bear on a population this run did not simulate."
            )
        print(f"population auditée : celle du run ({population})", file=sys.stderr)
    else:
        if not population.exists():
            raise SystemExit(f"population introuvable : {population}")
        print(f"audited population: named as argument ({population}) — check "
              f"that it is indeed the one the run simulated", file=sys.stderr)
    recompute = recompute_from_microdata() if args.recompute else None
    report = run_audit(population, args.run, recompute)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print_report(report)

    if args.trace:
        args.trace.mkdir(parents=True, exist_ok=True)
        out = args.trace / "audit_perimetre.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        if not args.json:
            print(f"\nTrace archived: {out}")

    # Exit code: 0 all compliant, 2 at least one axis to correct, 3 at least one
    # unmeasurable axis. "À publier" does not fail — it is an accepted limit.
    if NON_MESURABLE in report["verdicts"]:
        return 3
    if A_CORRIGER in report["verdicts"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
