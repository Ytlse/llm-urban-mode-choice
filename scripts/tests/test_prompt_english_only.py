"""A rendered prompt contains no French, proper nouns aside.

The acceptance criterion rests on this, and it is tricky: the served prompt is not a
file, it is an ASSEMBLY of eight independent surfaces that have all changed.

    system prompt (prompts.yaml)
  + category template (itinary_multi_agent/template.md.j2)
  + output schema (output_schema.json)
  + persona narrative (llm_agent._build_profile_narrative)
  + itinerary description (travel_plan_describe_v2.j2)
  + terminal labels (config/terminal_time.yaml)
  + weather bulletin (weather_loader)
  + condition labels (data/weather/meteo_toulouse_codes.csv)

Testing each piece separately proves nothing about the assembly: that is precisely where a
forgotten label slips in, and that is what happened during the switch — the precipitation
families still rendered "Pluie" in the middle of an English sentence, without any piece
test failing. So this test builds the ENTIRE prompt.

What it does not do: judge the quality of the English. It detects what an incomplete switch
leaves behind — accents and common French vocabulary — while sparing proper nouns,
which stay French in any language.

Run:
    services/llm-agents/.venv/bin/python -m pytest scripts/tests/test_prompt_english_only.py -q
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "services" / "llm-agents"))
sys.path.insert(0, str(RACINE / "packages" / "llm_gateway" / "src"))
sys.path.insert(0, str(RACINE / "packages" / "mobility_llm" / "src"))

ACCENTS = re.compile(r"[àâäéèêëîïôöùûüÿçœæÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇŒÆ]")

# Frequent French words in the translated surfaces. Deliberately SHORT and targeted: an
# exhaustive French list would catch words common to both languages (`train`, `bus`,
# `distance`, `option`, `mode`, `persona`) and the test would cry wolf on every run.
MOTS_FRANCAIS = re.compile(
    r"\b(durée|estimée|trajet|marche|marché|jusqu|météo|pluie|neige|ans|revenu|départ|"
    r"destination\s*:|attente|arrivée|correspondance|gratuit|scolaire|stationnement|"
    r"accès|conduite|vélo|voiture|à pied|aujourd|lever|coucher|rafales|verglas|"
    r"seul\(e\)|famille de|prévue|prévues|dont|nuageux|dégagé|ensoleillé|brouillard|brume)\b",
    re.IGNORECASE,
)


# ── Whitelist of proper nouns ────────────────────────────────────────────────────────────


def _noms_propres() -> set[str]:
    """GTFS stop names and municipality names of the scope — they stay French, as expected.

    Built from the SOURCES, never written by hand: a stop renamed in the GTFS must not
    make this test fail, and a frozen list would.
    """
    blancs: set[str] = set()
    for feed in ("tisseo_gtfs", "lio_gtfs", "ter_gtfs"):
        stops = RACINE / "data" / "gtfs" / feed / "stops.txt"
        if not stops.is_file():
            continue
        with stops.open(encoding="utf-8-sig", newline="") as f:
            for ligne in csv.DictReader(f):
                blancs.update(_mots(ligne.get("stop_name") or ""))
    communes = (RACINE / "packages" / "mobility_core" / "src" / "mobility_core"
                / "data" / "commune_couronne.json")
    if communes.is_file():
        for c in json.loads(communes.read_text(encoding="utf-8"))["communes"]:
            blancs.update(_mots(c.get("commune") or ""))
    # Persona first names come from the population; they are French by construction.
    blancs.update({"toulouse", "tissÉo", "lio", "ter", "sncf"})
    return blancs


def _mots(texte: str) -> set[str]:
    return {m.lower() for m in re.findall(r"[^\W\d_]+", texte, flags=re.UNICODE) if len(m) > 1}


@pytest.fixture(scope="module")
def blancs() -> set[str]:
    n = _noms_propres()
    assert len(n) > 500, (
        f"suspicious proper-noun whitelist ({len(n)} words): the GTFS sources or the "
        "municipality table were not read. Without it, the test would pass excusing nothing — "
        "or fail on every stop."
    )
    return n


def _mots_francais_hors_liste(texte: str, blancs: set[str]) -> list[str]:
    trouves = []
    for m in MOTS_FRANCAIS.finditer(texte):
        mot = m.group(0).lower().strip()
        if mot.split()[0] not in blancs:
            trouves.append(texte[max(0, m.start() - 60):m.end() + 40].replace("\n", " ⏎ "))
    return trouves


def _accents_hors_liste(texte: str, blancs: set[str]) -> list[str]:
    trouves = []
    for m in re.finditer(r"[^\W\d_]+", texte, flags=re.UNICODE):
        mot = m.group(0)
        if not ACCENTS.search(mot):
            continue
        if mot.lower() in blancs:
            continue
        # An accented word next to a known proper noun ("Baziège" in "Gare SNCF Baziège")
        # stays excused: we look at the immediate neighbourhood.
        voisinage = _mots(texte[max(0, m.start() - 30):m.end() + 30])
        if voisinage & blancs:
            continue
        trouves.append(texte[max(0, m.start() - 60):m.end() + 40].replace("\n", " ⏎ "))
    return trouves


# ── Assembly of the entire prompt ────────────────────────────────────────────────────────


def _reposer_libelles(plan: dict) -> dict:
    """Resets the `step_label` of terminal legs from the LIVE configuration.

    Without this, this test would fail on "Rejoindre la voiture" and "Conduite" — not because
    today's code produces them, but because they are **frozen** in the plans of the archived
    set, written under the French labels from before the switch.

    This is exactly the reason for the bump `terminal_time.yaml: tt4 → tt5`: `step_label` is
    serialised in the OTP cache and in the sealed sets, so a warm cache would have served
    these labels again to the v6 set at the time it was sealed. The test checks here what the
    current configuration produces; the bump itself is checked by
    `test_le_bump_de_version_protege_les_plans_en_cache`.
    """
    from trip_helper.terminal_time import terminal_profile

    mode = next((l.get("mode") for l in plan["legs"]
                 if l.get("mode") and terminal_profile(str(l["mode"]))), None)
    profil = terminal_profile(str(mode)) if mode else None
    if profil is None:
        return plan
    plan = dict(plan)
    plan["legs"] = [dict(l) for l in plan["legs"]]
    terminales = [l for l in plan["legs"] if l.get("step_label")]
    for i, leg in enumerate(terminales):
        if leg.get("mode") == mode:
            leg["step_label"] = profil.labels["main"]
        elif i == 0:
            leg["step_label"] = profil.labels["access"]
        else:
            leg["step_label"] = profil.labels["egress"]
    return plan


@pytest.fixture(scope="module")
def prompt_rendu() -> str:
    """The COMPLETE prompt — system + template + described options + persona + weather.

    The option plans come from the cold archive, READ ONLY and by justified
    exception: they are real itineraries (direct walk, car with terminal legs,
    public transport chain with a transfer), and a hand-made plan would never
    drive the template through its four branches.
    """
    from mobility_llm.persona import AgentSpec
    from mobility_llm.prompts import CATEGORIES_DIR, PROMPTS_FILE
    from models import TravelPlan
    from text_helper import env_ob_to_text
    from urban_mobility_agents.agents.llm_agent import _build_profile_narrative
    from urban_mobility_agents.utils import weather_loader

    from llm_gateway.prompts.engine import AvisNeutraliteManquant, PromptManager

    propositions = (RACINE / "archive" / "2026-09-14_avant_bascule_anglaise" / "plateforme"
                    / "jeux" / "population_1000_PANEL_v5_20260316" / "propositions.jsonl")
    if not propositions.is_file():
        pytest.skip(f"reference plans missing ({propositions}) — cold archive moved?")

    # One plan per template branch: direct, with terminal legs, multi-leg public transport.
    from text_helper.models.travel_plan import TravelPlanWrapper
    par_branche: dict[str, dict] = {}

    with propositions.open(encoding="utf-8") as f:
        for i, ligne in enumerate(f):
            if i > 3000 or len(par_branche) >= 3:
                break
            d = json.loads(ligne)
            for p in d["propositions"]:
                brut = dict(p["plan"])
                brut["purpose"] = brut.get("purpose") or d["purpose"]
                w = TravelPlanWrapper(**brut)
                cle = ("terminal" if w.has_terminal_legs
                       else "direct" if len(w.legs) <= 1 else "transit")
                par_branche.setdefault(cle, _reposer_libelles(brut))
    assert len(par_branche) == 3, f"branches covered: {sorted(par_branche)}"

    trajectoires = [
        {"index": i,
         "mode": TravelPlan.model_validate(brut).mode_label() or "unknown",
         "description": env_ob_to_text("travel_plan", brut),
         "total_distance_m": brut.get("distance") or 0.0}
        for i, brut in enumerate(par_branche.values())
    ]

    weather_loader._load()
    bulletin = weather_loader.weather_to_natural_language({
        "temperature": 2, "weather_label": weather_loader._code_labels[296],
        "precip_mm": 0.2, "temp_min": -1, "temp_max": 7,
        "sunrise": "07:55", "sunset": "17:25",
        "precip_slots": [("morning", "rain"), ("evening", "snow")], "wind_max_kmh": 95,
    })

    traits = {"name": "Thibault Marty", "age": 58, "main_occupation": "Travail à plein temps",
              "professional_activity": "Full-Time Worker", "household_size": 4,
              "income": "Medium-High"}

    agent = AgentSpec(
        agent_id="2348",
        perception=_build_profile_narrative(traits),
        destination="work",
        # ⚠ `destination_zone` is null in the simulator-less path (checked on 1,662 plans
        # of the v5 set), but NOT necessarily in the GAMA path, where the activity zone is
        # a French sentence from the population. So we inject it on purpose: it is the only
        # place where this channel is exercised.
        destination_zone="dense urban neighbourhood",
        departure_time="06:48",
        context=bulletin,
        day_outlook=weather_loader.day_weather_outlook(1773737984) or "afternoon 12°C, Clear/Sunny",
        agenda=["18:10 → home"],
        trajectories=trajectoires,
    )

    pm = PromptManager(
        templates_dir=CATEGORIES_DIR,
        prompts_file=PROMPTS_FILE,
        template_names={"itinary_multi_agent": "itinary_multi_agent/template.md.j2"},
        schema_paths={"itinary_multi_agent": CATEGORIES_DIR / "itinary_multi_agent"
                      / "output_schema.json"},
    )
    try:
        messages = pm.render("itinary_multi_agent", [agent], {"prompt_variant": "expert_m4"})
    except AvisNeutraliteManquant as e:
        # We DELIBERATELY go through the SERVING path, not through a lenient read:
        # what must be checked is the text actually sent to the model. A stale seal
        # prevents this text from existing — that is no reason to skip the test with a
        # `skip` (the absence of measurement would pass for success), it is a failure in itself,
        # and the message must say which one rather than let it read as a language regression.
        pytest.fail(
            "the prompt is not SERVABLE, so its text does not exist and nothing was "
            f"checked:\n  {e}\n"
            "→ B-8: have the translated variants re-examined by the `prompt-auditor` agent, "
            "then rewrite `_neutralite` (verdict, date, sha256_texte) before rerunning."
        )
    return "\n".join(m.content for m in messages)


# ── The checks ───────────────────────────────────────────────────────────────────────────


def test_le_montage_couvre_bien_les_surfaces(prompt_rendu: str) -> None:
    """Safeguard of the test itself: a truncated prompt would pass all the following checks.

    This is the "absence of measurement yields the perfect score" pattern: if the template rendered
    an empty page, there would be no French word in it.
    """
    for attendu, surface in (
        ("Thibault", "récit de persona"),
        ("Weather:", "bulletin météo"),
        ("Travel time:", "description d'itinéraire multi-jambes"),
        ("of access and parking", "libellés terminaux de terminal_time.yaml"),
        ("Trip options", "gabarit de catégorie"),
        ("Strict filtering", "prompt système"),
        ("dense urban neighbourhood", "destination_zone"),
        ('"agent_id"', "schéma de sortie"),
    ):
        assert attendu in prompt_rendu, f"surface missing from assembly: {surface} ({attendu!r})"
    assert len(prompt_rendu) > 2500, "suspiciously short prompt"


def test_aucun_caractere_accentue_hors_noms_propres(prompt_rendu: str, blancs: set[str]) -> None:
    restes = _accents_hors_liste(prompt_rendu, blancs)
    assert not restes, (
        f"{len(restes)} passage(s) accentué(s) hors noms propres :\n  - "
        + "\n  - ".join(restes[:12])
    )


def test_aucun_mot_francais_courant_hors_noms_propres(prompt_rendu: str,
                                                      blancs: set[str]) -> None:
    restes = _mots_francais_hors_liste(prompt_rendu, blancs)
    assert not restes, (
        f"{len(restes)} passage(s) en français :\n  - " + "\n  - ".join(restes[:12])
    )


def test_les_noms_propres_sont_bien_preserves(prompt_rendu: str) -> None:
    """The flip side of the test: the switch must not have anglicised a stop or municipality name.

    Without this check, "translating" `Gare SNCF Baziège` into `Baziege railway station` would pass
    the two previous tests with flying colours.
    """
    noms = [m for m in re.findall(r"'([^']{3,60})'", prompt_rendu)]
    assert noms, "no proper noun between apostrophes in the prompt — assembly to review"
    connus = _noms_propres()
    reconnus = [n for n in noms if _mots(n) & connus]
    assert reconnus, (
        f"none of the quoted names is a known stop or municipality of the scope: {noms[:10]}"
    )


def test_les_etiquettes_de_mode_restent_intactes(prompt_rendu: str) -> None:
    """Mode labels are read back IN the prompt text by `parse_option_modes`.

    They were already English; the switch was not supposed to touch them. The check is not
    about the language but about the identity of the served labels.
    """
    from mobility_llm.mode_choice import canonical_mode

    etiquettes = re.findall(r"^- \[\d+\] ([^:]+):", prompt_rendu, flags=re.M)
    assert etiquettes, "no option line \"- [n] mode:\" in the prompt"
    for e in etiquettes:
        assert not ACCENTS.search(e), f"accented mode label: {e!r}"
        famille = canonical_mode(e)
        assert famille and famille != "unknown", (
            f"mode label cannot be categorised: {e!r} → {famille!r}. `parse_option_modes` "
            "reads these labels back IN the prompt text; translating them would break "
            "`canonical_mode`, the calibration loss and the modal shares of moves.csv."
        )


def test_les_libelles_meteo_servis_sont_anglais() -> None:
    """The 48 served conditions come from `Condition_EN`, not from `Condition`.

    Check separate from the assembly: a single condition is rendered in a given prompt, and
    it is the ENTIRE table that must be switched.
    """
    from urban_mobility_agents.utils import weather_loader

    weather_loader._load()
    assert weather_loader._code_labels, "empty condition table"
    fautifs = {c: l for c, l in weather_loader._code_labels.items() if ACCENTS.search(l)}
    assert not fautifs, f"weather conditions still French: {fautifs}"


def test_les_familles_de_precipitation_reconnaissent_les_libelles_anglais() -> None:
    """The trap of the switch: expressions left in French no longer recognise anything.

    `_SNOW_RE` / `_RAIN_RE` are the ONLY link between the condition table and the precipitation
    sentence. Left in French, they would no longer have detected anything: `precip_slots`
    always empty, and "No precipitation expected." asserted every day of the year,
    even during a thunderstorm. No exception, no log — a false statement, not a gap.
    """
    from urban_mobility_agents.utils import weather_loader

    weather_loader._load()
    attendu = {296: "rain", 326: "snow", 113: None, 248: None, 350: "snow", 200: "rain"}
    for code, famille in attendu.items():
        libelle = weather_loader._code_labels[code]
        assert weather_loader._precip_family(libelle) == famille, (
            f"code {code} ({libelle!r}) → {weather_loader._precip_family(libelle)!r}, "
            f"expected {famille!r}"
        )
    # And overall: the table must not have gone silent.
    familles = {weather_loader._precip_family(l) for l in weather_loader._code_labels.values()}
    assert familles == {"rain", "snow", None}, f"families produced: {familles}"


def test_le_bump_de_version_protege_les_plans_en_cache() -> None:
    """`terminal_time.yaml` did change version, and only the one that had to.

    Terminal labels are serialised in `Transit.step_label`, and the OTP cache stores
    complete `TravelPlan`s: without a `data_version` bump, a warm cache would have served
    plans carrying "Rejoindre la voiture" again to the v6 set, at the time it was sealed, with no
    log reporting it. This can be checked on evidence — the archived v5 set's plans carry them.

    `routing_version` was NOT supposed to move: the OSMnx routing cache only stores network
    time, which no label touches. Bumping it would have cost ~2 h of recomputation for nothing.
    """
    from trip_helper.terminal_time import data_version, routing_version, terminal_profile

    assert data_version() != "tt4", (
        "data_version stayed at tt4 although the terminal labels were translated: "
        "the OTP cache will serve plans with French sub-steps again"
    )
    assert routing_version() == "r3", (
        "routing_version a bougé sans nécessité : le cache de routage OSMnx sera recalculé à "
        "froid (~2 h) alors qu'aucune durée réseau n'a changé"
    )
    for mode in ("car", "bicycle"):
        profil = terminal_profile(mode)
        for cle in ("access", "main", "egress", "egress_sans_destination", "terminal"):
            libelle = profil.labels[cle]
            assert not ACCENTS.search(libelle), (
                f"terminal label still French: {mode}.{cle} = {libelle!r}"
            )


def _jour_a_meteo_variable() -> int:
    """A timestamp falling on a day whose weather CHANGES across the time slots.

    The day of the sealed set (2026-03-16) is "Clear/Sunny" from morning to evening: on it, the
    weather branch of `_agenda_lines` never fires and the language check covers
    nothing. The day is therefore chosen from the data — 228 of the 365 days carry at least
    three distinct labels — rather than fixed at random.
    """
    import calendar
    from datetime import datetime, timezone

    from urban_mobility_agents.utils import weather_loader as W

    W._load()
    for (mois, jour), ligne in sorted(W._weather_index.items()):
        libelles = set()
        for col in W._FINE_CODE_COLS.values():
            try:
                libelles.add(W._code_labels[int(float(ligne[col]))])
            except (KeyError, ValueError, TypeError):
                continue
        if len(libelles) >= 3:
            return calendar.timegm(
                datetime(2026, mois, jour, 6, 0, tzinfo=timezone.utc).timetuple()
            )
    raise AssertionError("no day with variable weather in data/weather — test inoperative")


def test_l_agenda_d_anticipation_est_anglais() -> None:
    """The NINTH surface — `_agenda_lines`, missing from the initial inventory.

    This agenda line ("18:10 → home (≈4,2 km) — light rain expected") goes into the prompt
    via `agent.agenda`, and its `signature` enters the decision cache key. It is only
    rendered for agents with a vehicle to chain AND when the forecast weather differs from
    the one at departure: rare enough for a sample to miss it, frequent enough for it to
    reach production. It was indeed missed in the first translation pass, because
    the prompt fixture supplied an agenda line WRITTEN BY HAND — an assembly that tests
    the template, never the producer.

    This test therefore exercises the real producer on real personas, and **refuses to pass
    without seeing the weather branch**: otherwise it would go green the day the branch stops
    being reached, which is the "absence of measurement yields the perfect score" pattern.
    """
    from experiences.population import charger_population
    from urban_mobility_agents.simulation_controller import _agenda_lines

    cohorte = (RACINE / "archive" / "2026-09-14_avant_bascule_anglaise" / "population"
               / "population_1000_PANEL_v5")
    if not (cohorte / "population.json").is_file():
        pytest.skip(f"reference cohort missing ({cohorte})")

    personnes, _ = charger_population(
        cohorte,
        archivee_confirmee=(
            "ticket 074 B-9 : contrôle linguistique de l'agenda d'anticipation sur des personas "
            "réels — lecture seule, aucune mesure produite"
        ),
    )
    # ⚠ A REAL timestamp, not a 24h time. `_agenda_lines` calls `get_weather`, which reads
    # wall-clock time: passing it `acte.end_time` (0..86400) puts everything in 1970, all
    # slots fall in the same place, and the weather branch NEVER fires. The test
    # would then pass having exercised nothing — which is what it refused to do on the first draft.
    from helper import to_timestamp_based_on_day

    jour = _jour_a_meteo_variable()

    lignes: list[str] = []
    avec_meteo = 0
    for personne in personnes:
        actes = personne.identity.activities or []
        for i, acte in enumerate(actes[:-1]):
            brut = acte.end_time
            if brut is None:
                continue
            depart = to_timestamp_based_on_day(int(brut), jour)
            produites = _agenda_lines(personne, actes[i + 1], depart)
            lignes.extend(produites)
            avec_meteo += sum(1 for l in produites if " — " in l)
        if len(lignes) > 4000:
            break

    assert lignes, "no agenda line produced — the test measures nothing"
    assert avec_meteo > 0, (
        f"{len(lignes)} agenda lines produced, NONE carries the weather suffix \"— …\": "
        "the branch that carried the French was not exercised, so nothing was checked"
    )
    blancs = _noms_propres()
    fautives = [l for l in lignes if _accents_hors_liste(l, blancs)
                or _mots_francais_hors_liste(l, blancs)]
    assert not fautives, (
        f"{len(fautives)} ligne(s) d'agenda en français sur {len(lignes)} "
        f"({avec_meteo} avec suffixe météo) :\n  - " + "\n  - ".join(fautives[:8])
    )
