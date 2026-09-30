"""Terminal time of itineraries.

Covers the acceptance criteria that can be checked without an LLM call:

1. 100 % of car and bike options describe a terminal time;
2. the displayed total is GREATER than the driving time alone, and the
   breakdown **sums** to the total;
4. no public transport option has changed (no double counting);

plus the invariants that protect the measurement: unchanged plan code (key of the
decision cache), unchanged mode label (it feeds the calibration loss),
``Distance`` line kept (it carries ``dist_km``), and versioning of the two
caches which, without it, would serve stale durations again.
"""

import sys
from pathlib import Path

import pytest
import yaml

_LLM_AGENTS = Path(__file__).resolve().parents[1]
if str(_LLM_AGENTS) not in sys.path:
    sys.path.insert(0, str(_LLM_AGENTS))

from helper import humanize_duration                      # noqa: E402
from models import Location, Transit, TransitLocation     # noqa: E402
from text_helper.models.travel_plan import (              # noqa: E402
    TravelPlanLiteWrapper, TravelPlanWrapper)
from text_helper.templates import repository              # noqa: E402
from trip_helper import terminal_time                     # noqa: E402
from trip_helper.osmnx_direct import _make_travel_plan    # noqa: E402

# REAL GTFS route (bus line 13): with a made-up identifier, the template
# would render "Unknown 'Unknown'" and the non-regression check of the
# public transport options would prove nothing.
_REAL_BUS_ROUTE = "line:142"

# Points anchored in KNOWN rings (the terminal time has been spatialised since
# version tt2): without anchoring, the expected values would depend on a latitude fluke.
ORIGIN = Location(lon=1.4450, lat=43.5973)   # hypercentre → "Toulouse"
DEST = Location(lon=1.4400, lat=43.6000)     # 500 m away → "Toulouse" too
ZONE_LOINTAINE = Location(lon=1.10, lat=43.40)  # ~35 km south-west; "3rd ring" by commune ("2nd" by distance — the gap since closed)
T0 = 1773723994


@pytest.fixture(autouse=True)
def minutes_not_buckets():
    """Forces rendering in minutes, like production.

    ``settings.agent.quantify_time_window`` is ``True`` by default in the code
    but ``false`` in every run config (``config/config_*.yaml``): the
    frozen sets carry minutes, not "moderate (under 10 minutes)". Without this
    override, the tests would measure a rendering that production does not use.
    """
    previous = repository.env.filters['duration_to_bucket_text']
    repository.env.filters['duration_to_bucket_text'] = humanize_duration
    yield
    repository.env.filters['duration_to_bucket_text'] = previous


@pytest.fixture(autouse=True)
def fresh_config():
    """Cache emptied before/after, AND config path restored.

    ⚠ `_write_config` reassigns `terminal_time._CONFIG_PATH` to a temporary
    file without putting it back: without this restore, every test run AFTER a
    configuration test read a throwaway test YAML in `tmp_path`. The leak was
    invisible as long as no test depended on the PRODUCTION values; the
    survey-alignment guard does depend on them — it passed alone and
    failed in the suite.
    """
    original_path = terminal_time._CONFIG_PATH
    terminal_time.reset()
    yield
    terminal_time._CONFIG_PATH = original_path
    terminal_time.reset()


# CERTAIN terminal times, in minutes, for the structure and rendering tests.
# Since tt3 the terminal time is DRAWN from the survey law, massed at zero (88 to
# 96 % of trips have none): a production plan therefore most often has
# a single leg. That is the intended behaviour, but it makes undecidable a test
# that wants to check the BREAKDOWN — it needs a case where both ends exist.
# This fixture installs a certain law, which restores determinism without
# going back to constants: the mechanism tested is still the draw.
CERTAIN_ACCESS_MIN = 2
CERTAIN_EGRESS_MIN = 3


@pytest.fixture
def certain_terminal():
    """Forces access and egress to certain values, for both vehicle modes."""
    conf = terminal_time._load()
    for mode in ("car", "bicycle"):
        profile = conf["modes"][mode]
        conf["modes"][mode] = terminal_time.TerminalProfile(
            mode=profile.mode,
            access_by_zone=profile.access_by_zone,
            egress_by_zone=profile.egress_by_zone,
            provenance=profile.provenance,
            spatialise=profile.spatialise,
            labels=profile.labels,
            access_law_by_zone={"default": {CERTAIN_ACCESS_MIN * 60: 1.0}},
            egress_law_by_zone={"default": {CERTAIN_EGRESS_MIN * 60: 1.0}},
        )
    yield
    terminal_time.reset()


def _wrap(plan, purpose="shop"):
    """Applies the millisecond conversion done by ``get_itineraries``."""
    plan.purpose = purpose
    plan.start_time = int(plan.start_time * 1000)
    plan.end_time = int(plan.end_time * 1000)
    for leg in plan.legs:
        leg.start_time = int(leg.start_time * 1000)
        leg.end_time = int(leg.end_time * 1000)
    return TravelPlanWrapper(**plan.model_dump())


def direct(mode, network_s, distance_m=1800.0, purpose="shop"):
    """Direct plan from a pure NETWORK TRAVEL duration."""
    return _wrap(_make_travel_plan(ORIGIN, DEST, mode, T0, network_s, distance_m),
                 purpose)


def transit_plan(purpose="shop"):
    """``foot,bus,foot`` plan as produced by the OTP parser."""
    stop_a = TransitLocation(stop="Pradettes", lat=43.60, lon=1.40)
    stop_b = TransitLocation(stop="Gare SNCF Baziège", lat=43.61, lon=1.41)
    orig = TransitLocation(stop="", lat=43.60, lon=1.40)
    dest = TransitLocation(stop="", lat=43.61, lon=1.41)
    base = T0 * 1000
    legs = [
        Transit(start_time=base, end_time=base + 180_000, duration=180, mode="foot",
                start_location=orig, end_location=stop_a, is_transfer=True),
        Transit(start_time=base + 180_000, end_time=base + 300_000, duration=120,
                mode="bus", start_location=stop_a, end_location=stop_b,
                is_transfer=False, transit_route=_REAL_BUS_ROUTE),
        Transit(start_time=base + 300_000, end_time=base + 780_000, duration=480,
                mode="foot", start_location=stop_b, end_location=dest,
                is_transfer=True),
    ]
    return TravelPlanWrapper(id="t", start_location=ORIGIN, end_location=DEST,
                             start_time=base, end_time=base + 780_000, duration=780,
                             distance=1400.0, purpose=purpose, legs=legs)


# ── Plan structure ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode, expected_legs", [
    ("car", 3), ("bicycle", 3), ("foot", 1),
])
def test_nombre_de_jambes_par_mode(mode, expected_legs, certain_terminal):
    """Car and bike carry access + trip + egress; walking does not.

    Walking is door-to-door by construction (§4.1): adding a terminal time to it
    would invent a cost that does not exist.
    """
    plan = direct(mode, 300)
    assert len(plan.legs) == expected_legs


def test_marche_et_tc_sans_temps_terminal():
    """Criterion 4: no terminal time added where it would be double counting."""
    assert not direct("foot", 600).has_terminal_legs
    assert not transit_plan().has_terminal_legs
    assert terminal_time.terminal_profile("foot") is None
    assert terminal_time.terminal_profile("bus") is None


def test_code_de_plan_inchange():
    """INVARIANT: breaking down the display must not change the plan's identity.

    ``get_code()`` is the key of the LLM decision cache and of itinerary
    deduplication. If the terminal legs entered it, every car option
    would become a "new" option and the whole cache would be silently lost.
    """
    assert direct("car", 300).get_code() == "__DIRECT_CAR__^^"
    assert direct("bicycle", 300).get_code() == "__DIRECT_BICYCLE__^^"
    assert direct("foot", 300).get_code() == "__DIRECT_FOOT__^^"


def test_etiquette_de_mode_inchangee():
    """The label stays ``car`` — it is what ``parse_option_modes`` reads.

    Without excluding the terminal legs, it would be ``"None,car,None"``: the
    mode of each option would be misassigned, hence ``categorize_mode``, hence the
    calibration loss and the modal shares of ``moves.csv``.
    """
    assert direct("car", 300).mode_label() == "car"
    assert direct("bicycle", 300).mode_label() == "bicycle"
    assert direct("foot", 300).mode_label() == "foot"
    assert transit_plan().mode_label() == "foot,bus,foot"


# ── Rendering ────────────────────────────────────────────────────────────────

def test_rendu_voiture_decompose(certain_terminal):
    """Criteria 1 and 2: the breakdown is displayed and it sums to its total.

    The expected durations are DERIVED from the fixture, not copied: since tt3 the
    terminal time is drawn from the survey law, and hard-coding "3 min of access" would make
    this test a test of the tt2 table rather than of the rendering.
    """
    drive_min, terminal_min = 3, CERTAIN_ACCESS_MIN + CERTAIN_EGRESS_MIN
    desc = direct("car", drive_min * 60, distance_m=1800.0).describe()
    assert desc.startswith(f" Travel time: {drive_min + terminal_min} minutes, "
                           f"including {terminal_min} minutes of access and parking. "
                           f"Distance: 1.8 km.")
    assert f"\n- Walk to the car: {CERTAIN_ACCESS_MIN} minutes." in desc
    assert f"\n- Driving: {drive_min} minutes." in desc
    assert (f"\n- Parking and walk to 'shop': "
            f"{CERTAIN_EGRESS_MIN} minutes.") in desc


def test_rendu_velo_decompose(certain_terminal):
    """The bike changes RENDERING without changing duration — and that is intended (T5).

    The 2 terminal minutes are the ones ``park_base`` already added
    silently: the terminal time makes them visible, it does not invent them. If the bike
    shares move, it will be through salience alone.
    """
    ride_min, terminal_min = 5, CERTAIN_ACCESS_MIN + CERTAIN_EGRESS_MIN
    desc = direct("bicycle", ride_min * 60, distance_m=1400.0).describe()
    assert desc.startswith(f" Travel time: {ride_min + terminal_min} minutes, "
                           f"including {terminal_min} minutes of access and locking. "
                           f"Distance: 1.4 km.")
    assert f"\n- Unlock the bike: {CERTAIN_ACCESS_MIN} minutes." in desc
    assert f"\n- Cycling: {ride_min} minutes." in desc
    assert f"\n- Lock the bike at 'shop': {CERTAIN_EGRESS_MIN} minutes." in desc


def test_rendu_marche_inchange():
    assert direct("foot", 960, distance_m=1400.0).describe() == (
        " Estimated duration: 16 minutes. Distance: 1.4 km.")


def test_rendu_transports_collectifs_inchange():
    """Criterion 4: the text of public transport options is identical to the character.

    Expected string copied from the rendering PRIOR to the terminal time (format of the v3
    frozen sets): it is the only check that detects a regression introduced by
    the new template branch.
    """
    assert transit_plan().describe() == (
        " Travel time: 13 minutes, including 11 minutes of walking."
        "\n- Walk to 'Pradettes': 3 minutes."
        "\n- Bus '13' to 'Gare SNCF Baziège': 2 minutes."
        "\n- Walk to 'shop': 8 minutes.")


def test_distance_conservee_sur_voiture_et_velo(certain_terminal):
    """The ``Distance`` line carries ``dist_km`` of the calibration records.

    ``metadata.extract_min_distance_km`` takes the MINIMUM of the distances displayed
    in the section. Measured on ``v3``: 579 records (13.5 %) get their
    distance ONLY from car/bike lines. The public transport branch shows no distance;
    if direct plans switched to it without keeping theirs, those decisions
    would lose ``dist_cat`` and drop out of the distance stratum of the measurement —
    a score that improves because less is measured.
    """
    for mode in ("car", "bicycle"):
        assert "Distance: 2.3 km." in direct(mode, 300, distance_m=2300.0).describe()


@pytest.mark.parametrize("mode", ["car", "bicycle"])
@pytest.mark.parametrize("network_s", [1, 59, 60, 61, 119, 137, 300, 613, 3599, 4741])
def test_total_egale_somme_des_sous_etapes(mode, network_s, certain_terminal):
    """Criterion 2, tested on a grid of durations not aligned on the minute.

    ``humanize_duration`` TRUNCATES to the minute. The equality only holds because
    the terminal times are multiples of 60 s
    (``floor(a + k×60) == floor(a) + k``), which the configuration loader
    enforces. This test is what would make visible a 90 s value slipped into the
    YAML.
    """
    plan = direct(mode, network_s)
    total_minutes = plan.total_seconds // 60
    sous_etapes = sum(seconds // 60 for _, seconds in plan.described_steps)
    assert total_minutes == sous_etapes
    # The total is strictly greater than the travel duration alone (criterion 2).
    assert plan.total_seconds > network_s


def test_forme_courte_reconnait_les_plans_a_trois_jambes(certain_terminal):
    """The memory query must still describe the trip, not an empty list.

    The test was on ``legs | length == 1``; a car plan has three
    since the terminal time and would have fallen into the "List of transits" branch, which
    only shows public transport legs — hence nothing.
    """
    total = 3 + CERTAIN_ACCESS_MIN + CERTAIN_EGRESS_MIN
    lite = TravelPlanLiteWrapper(**direct("car", 180).model_dump())
    assert lite.describe() == (f"Direct car; Duration: {total} minutes; "
                               f"Distance: 1.8 km.")


def test_libelle_de_diffusion_sans_destination_connue(certain_terminal):
    """Without ``purpose``, no destination name is made up."""
    desc = direct("car", 180, purpose=None).describe()
    assert (f"Parking and walk to the destination: "
            f"{CERTAIN_EGRESS_MIN} minutes.") in desc
    assert "{destination}" not in desc


# ── Configuration ────────────────────────────────────────────────────────────

def _write_config(tmp_path, payload):
    path = tmp_path / "terminal_time.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    terminal_time._CONFIG_PATH = path
    terminal_time.reset()
    return path


_LABELS = {"access": "a", "main": "m", "egress": "e '{destination}'",
           "egress_sans_destination": "e", "terminal": "t"}


def test_config_refuse_un_temps_non_multiple_de_60(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_time, "_CONFIG_PATH", terminal_time._CONFIG_PATH)
    original = terminal_time._CONFIG_PATH
    try:
        _write_config(tmp_path, {"version": "x",
                                 "modes": {"car": {"access_s": {"default": 90},
                                                   "egress_s": {"default": 120},
                                                   "labels": _LABELS}}})
        with pytest.raises(ValueError, match="multiple of 60"):
            terminal_time.terminal_profile("car")
    finally:
        terminal_time._CONFIG_PATH = original
        terminal_time.reset()


def test_config_refuse_une_version_absente(tmp_path):
    original = terminal_time._CONFIG_PATH
    try:
        _write_config(tmp_path, {"modes": {}})
        with pytest.raises(ValueError, match="version"):
            terminal_time.data_version()
    finally:
        terminal_time._CONFIG_PATH = original
        terminal_time.reset()


def test_config_refuse_des_libelles_incomplets(tmp_path):
    original = terminal_time._CONFIG_PATH
    try:
        _write_config(tmp_path, {"version": "x",
                                 "modes": {"car": {"access_s": {"default": 60},
                                                   "egress_s": {"default": 60},
                                                   "labels": {"access": "a"}}}})
        with pytest.raises(ValueError, match="missing labels"):
            terminal_time.terminal_profile("car")
    finally:
        terminal_time._CONFIG_PATH = original
        terminal_time.reset()


def test_provenance_du_velo_est_sourcee_depuis_tt3():
    """T2: a value without a source must declare itself, not hide.

    No quantified reference was found for the terminal time of a personal
    bike; the value reuses the one from the earlier code. Writing it in the
    configuration is what prevents citing it later as sourced.
    """
    # tt2 declared the bike `unsourced`, for lack of a published value for the terminal
    # time of a PERSONAL bike. tt3 has one: the survey measures it (T2/T6 on
    # T3 ∈ {11, 17}, 2,047 trips, 0.11 min per end). Both modes are therefore
    # sourced — and leaving the bike unsourced next to a corrected car would have
    # left an undocumented bias against a documented one.
    assert terminal_time.terminal_profile("bicycle").provenance == "sourced"
    assert terminal_time.terminal_profile("car").provenance == "sourced"


def test_grille_de_sensibilite_ordonnee():
    """T6: three variants, strictly increasing factors.

    A variant applies a FACTOR to all rings, not a single
    value: a constant would crush the spatialisation, and the sensitivity
    would measure "spatialised or not" at the same time as the terminal time's magnitude.
    """
    variants = terminal_time.sensitivity_variants()
    assert set(variants) == {"low", "central", "high"}
    factors = [variants[n]["car"]["factor"] for n in ("low", "central", "high")]
    assert factors[0] < factors[1] < factors[2]


def test_variante_conserve_le_gradient_par_zone():
    """A variant scales WITHOUT flattening the spatialisation."""
    # On the law's EXPECTATION: since tt3 `egress_s()` draws, and comparing two
    # draws would say nothing about the scaling.
    base = terminal_time.terminal_profile("car")
    ecart_base = (base.mean_s("egress", "Toulouse")
                  - base.mean_s("egress", "3rd ring"))
    terminal_time.apply_variant("high")
    haut = terminal_time.terminal_profile("car")
    assert haut.mean_s("egress", "Toulouse") > base.mean_s("egress", "Toulouse")
    assert (haut.mean_s("egress", "Toulouse")
            - haut.mean_s("egress", "3rd ring")) > ecart_base
    # The multiples-of-60 invariant survives scaling — including on
    # the law's KEYS, which are the durations actually displayable.
    for zone in ("Toulouse", "1st ring", "2nd ring", "3rd ring",
                 "default"):
        for laws in (haut.access_law_by_zone, haut.egress_law_by_zone):
            for seconds in (laws.get(zone) or {}):
                assert seconds % 60 == 0, (zone, seconds)


def test_variante_de_sensibilite_change_la_version_de_donnees():
    """The three sensitivity sets must not share a cache key.

    Without a version suffix, an eval under the high variant would read the cache of the
    central variant: the three T6 measurements would merge.
    """
    base = terminal_time.data_version()
    terminal_time.apply_variant("high")
    assert terminal_time.data_version() == f"{base}-high"
    # The high variant does raise the central expectation (600 s was the expected value in
    # tt2, where the terminal time was constant; under a law it is the expectation that rises).
    haut = terminal_time.terminal_profile("car")
    terminal_time.reset()
    central = terminal_time.terminal_profile("car")
    assert (haut.mean_s("access", "Toulouse") + haut.mean_s("egress", "Toulouse")
            > central.mean_s("access", "Toulouse")
            + central.mean_s("egress", "Toulouse"))
    terminal_time.apply_variant("high")
    # Two successive switches do not stack the suffixes.
    terminal_time.apply_variant("low")
    assert terminal_time.data_version() == f"{base}-low"


def test_bascules_successives_nempilent_pas_les_facteurs():
    """Chaining two variants must give the second one, not the product of both.

    This is the NORMAL use case of the T6 grid: a loop going through low,
    central, high in the same process. Starting from the current profiles instead
    of the central ones, `high` then `low` applied 1.5 × 0.5 = 0.75 — terminal
    times that no variant declares, under a correct variant label.
    """
    def esperance():
        return terminal_time.terminal_profile("car").mean_s("egress", "Toulouse")

    central = esperance()
    terminal_time.apply_variant("low")
    attendu_low = esperance()

    terminal_time.reset()
    terminal_time.apply_variant("high")
    terminal_time.apply_variant("low")
    assert esperance() == pytest.approx(attendu_low)
    assert attendu_low < central  # and the low variant does stay low

    # Same the other way, and going back to `central` gives the base values.
    terminal_time.apply_variant("central")
    assert esperance() == pytest.approx(central)


def test_version_de_donnees_dans_les_trois_cles_de_cache(monkeypatch):
    """The THREE caches that survive across runs must move with the parameter.

    None of the three can guess that a terminal time has changed:

    - **OSMnx routing** — addressed by (mode, coordinates, time slot): it would serve
      durations computed under the old definition;
    - **OTP itineraries** — the most serious, because it does not store durations but
      the serialised ``TravelPlan``s, car and bike options included: a warm cache
      would serve SINGLE-leg plans with the old parking melted
      into them, i.e. the original defect brought back after its fix;
    - **LLM decisions** — addressed by ``get_code()`` (routes and stops), hence
      insensitive to durations by construction: it would replay decisions made on
      options that no longer exist, and **nothing would flag it in the logs**.
    """
    from datetime import datetime

    from llm.cache import LlmSemanticCache
    from models import Location
    from trip_helper.osmnx_persistent_cache import OsmnxPersistentCache
    from trip_helper.otp_persistent_cache import OtpPersistentCache

    options = [direct("car", 180)]
    dt = datetime(2026, 3, 17, 10, 55)
    ts = int(dt.timestamp())

    before_state = LlmSemanticCache._make_state_hash(options)
    before_route, *_ = OsmnxPersistentCache.make_key(dt, "car", 43.6, 1.4, 43.61, 1.41)
    before_otp = OtpPersistentCache.make_key(ts, ORIGIN, DEST, True, False, True)
    before_bl = OtpPersistentCache.make_blacklist_key(ORIGIN, DEST)

    # Changing the TERMINAL TIME must invalidate the PLAN and DECISION caches…
    monkeypatch.setattr(terminal_time, "data_version", lambda: "tt-autre")
    assert LlmSemanticCache._make_state_hash(options) != before_state
    assert OtpPersistentCache.make_key(ts, ORIGIN, DEST, True, False, True) != before_otp

    # …but NOT the ROUTING cache, which only stores network time. Invalidating it
    # would recompute thousands of routes from cold for an identical result
    # (~2 h for 930 personas). That is the distinction `routing_version` carries.
    assert OsmnxPersistentCache.make_key(dt, "car", 43.6, 1.4, 43.61, 1.41)[0] == before_route

    # Routing has its own version, which does invalidate it when it moves.
    monkeypatch.setattr(terminal_time, "routing_version", lambda: "r-autre")
    after_route, *_ = OsmnxPersistentCache.make_key(dt, "car", 43.6, 1.4, 43.61, 1.41)
    assert after_route != before_route

    # The blacklist, for its part, MUST NOT move: "OTP does not link these two points"
    # is a fact of network topology, independent of any terminal time. Versioning
    # it would query OTP again for nothing on all pairs known to be
    # unlinked — half of a run's "No usable itinerary" warnings.
    assert OtpPersistentCache.make_blacklist_key(ORIGIN, DEST) == before_bl


# ── Spatialisation ────────────────────────────

def test_le_temps_terminal_depend_de_la_couronne():
    """Parking costs more in the centre than in the rings — and it is sourced.

    This is the refinement §4.1 announced: "the parameter should depend on the
    place of residence / destination (parking does not cost the same inside the Toulouse
    ring road and in the 3rd ring)". A global constant underestimated the cost
    of using the car in the centre and overestimated it on the outskirts — hence flattened
    precisely the spatial elasticity we want to measure.
    """
    # ⚠ Since tt3 the duration is DRAWN: comparing `access_s()` would compare two
    # draws, not two rings. The gradient is expressed on the law's EXPECTATION.
    p = terminal_time.terminal_profile("car")
    assert p.spatialise is True
    assert (p.mean_s("access", "Toulouse") > p.mean_s("access", "1st ring")
            and p.mean_s("access", "2nd ring")
            > p.mean_s("access", "3rd ring"))
    assert (p.mean_s("egress", "Toulouse") > p.mean_s("egress", "1st ring")
            and p.mean_s("egress", "2nd ring")
            > p.mean_s("egress", "3rd ring"))


def test_zone_inconnue_retombe_sur_le_defaut():
    """A point outside the EMC² layer must not make the terminal time vanish.

    ``residence_zone`` returns an empty string for an unknown point. Without a
    ``default`` entry, the table would be looked up empty — and the mode would become free again,
    that is, the fixed bug, brought back by a zoning gap.
    """
    p = terminal_time.terminal_profile("car")
    # The `default` law is served for any unknown ring, and its expectation is
    # non-zero: a zoning gap does not make the car free. The test is on
    # the expectation and not on a draw — a draw is 0 in ~92 % of cases, which is
    # the intended behaviour and says nothing about the fallback.
    assert p.mean_s("access", "") == p.mean_s("access", "default")
    assert p.mean_s("egress", "zone inexistante") == p.mean_s("egress", "default")
    assert p.mean_s("access", "") > 0 and p.mean_s("egress", "") > 0


def test_les_deux_bouts_sont_tarifes_separement():
    """Access on the ORIGIN ring, parking on the DESTINATION one."""
    from trip_helper.osmnx_direct import _make_travel_plan

    vers_centre = _wrap(_make_travel_plan(ZONE_LOINTAINE, ORIGIN, "car", T0, 180, 1800.0))
    vers_peripherie = _wrap(_make_travel_plan(ORIGIN, ZONE_LOINTAINE, "car", T0, 180, 1800.0))
    # Going to the centre costs more than leaving it: parking dominates. In
    # EXPECTATION — a pair of draws would prove nothing, the law being massed at zero.
    p = terminal_time.terminal_profile("car")
    attendu_centre = (p.mean_s("access", "3rd ring")
                      + p.mean_s("egress", "Toulouse"))
    attendu_peripherie = (p.mean_s("access", "Toulouse")
                          + p.mean_s("egress", "3rd ring"))
    assert attendu_centre > attendu_peripherie
    # And the total is still the sum of the displayed sub-steps.
    for plan in (vers_centre, vers_peripherie):
        assert plan.total_seconds // 60 == sum(s // 60 for _, s in plan.described_steps)


def test_les_deux_classements_convergent_depuis_tt4():
    """This test flips for the second time, and each flip was a decision.

    Originally, it required the log and the terminal time to classify identically
    — by distance. The residence ring split them: the log READS the persona's commune trait,
    the terminal time kept classifying its points by distance, a divergence capped at
    34 s per trip end and documented. A later fix closes it from the other end: the
    laws of `terminal_time_emc2.json` are re-stratified by the survey table (tt4),
    and `_make_travel_plan` classifies its points by MEMBERSHIP of the rings. The two
    definitions therefore coincide again — but on the survey's, not on the
    metric one.

    Blagnac is the case that told them apart: 6 km from the Capitole, "Toulouse" by
    distance, 1st ring by commune. The terminal time must call it 1st ring.
    """
    import json
    from pathlib import Path

    import urban_mobility_agents.utils.move_logger as move_logger
    from mobility_core.geo_reference import residence_zone as classement_metrique
    from trip_helper import osmnx_direct

    blagnac = (43.635, 1.39)
    assert classement_metrique(*blagnac) == "Toulouse"                      # the old definition
    assert osmnx_direct._terminal_zone(*blagnac, "access") == "1st ring"  # the survey's
    assert osmnx_direct._terminal_zone(43.5973, 1.4450, "egress") == "Toulouse"

    # The resource is stratified on the same definition — otherwise the drawn durations
    # would be compared with strata that do not denote the same territories.
    from mobility_core.resources import data_path
    repo = Path(__file__).resolve().parents[3]
    meta = json.loads(data_path("terminal_time_emc2.json").read_text(encoding="utf-8"))["meta"]
    assert "CouronneTable" in meta["crown_definition"]
    assert "geo_reference" not in meta["crown_definition"]

    # The log reads the trait, and nothing else.
    assert move_logger._residence_zone({"residence_zone": "1st ring"}) == "1st ring"
    assert move_logger._residence_zone({}) == ""

    # And the distance fallback is IMPOSSIBLE in the THREE modules, not merely
    # discouraged: none imports the metric function. Without this check, a "reasonable
    # fallback" would come back in one line at the first careless review.
    assert not hasattr(move_logger, "residence_zone")
    for source_path in (Path(osmnx_direct.__file__),
                        repo / "scripts" / "progedo_logit" / "export_terminal_time.py"):
        assert "geo_reference import residence_zone" not in source_path.read_text(
            encoding="utf-8"), source_path.name


def test_hors_perimetre_est_compte_et_tombe_sur_la_loi_default():
    """Axis A4 of the audit, closed on the terminal time side.

    Before tt4, a point 100 km from the Capitole was classified "3rd ring" because "beyond
    40 km" had no bound: it paid the law of a ring it is not in,
    silently. It now gets `hors périmètre`, which has no law of its own: the
    `default` law is served to it, and the case is COUNTED — a high volume would flag a
    population or a scope that has changed, not a normal case to absorb.
    """
    from mobility_core.population_reference import OUT_OF_PERIMETER
    from trip_helper import osmnx_direct

    lot = (44.50, 1.00)   # Lot département, ~100 km north — outside the 453 communes
    counter = osmnx_direct.TERMINAL_OUT_OF_PERIMETER.labels(end="egress")
    before = counter._value.get()
    assert osmnx_direct._terminal_zone(*lot, "egress") == OUT_OF_PERIMETER
    assert counter._value.get() == before + 1
    p = terminal_time.terminal_profile("car")
    assert p.mean_s("egress", OUT_OF_PERIMETER) == p.mean_s("egress", "default")
    assert p.mean_s("egress", OUT_OF_PERIMETER) > 0


# ── tt3: the terminal time is DRAWN from the survey law ──────────────────────

def test_la_loi_remplace_la_constante_quand_elle_est_servie():
    """A law, when present, prevails; the constant is no longer consulted.

    The two mechanisms coexist on purpose — a future mode may stay on a
    constant — but they must not mix: serving both and reading the
    constant would make the config file misleading.
    """
    profile = terminal_time.TerminalProfile(
        mode="car", access_by_zone={"default": 999 * 60},
        egress_by_zone={"default": 999 * 60}, provenance="sourced",
        spatialise=False, labels={},
        access_law_by_zone={"default": {120: 1.0}},
        egress_law_by_zone={"default": {180: 1.0}})
    assert profile.access_s("", "k") == 120
    assert profile.egress_s("", "k") == 180
    assert profile.mean_s("access") == 120


def test_sans_loi_la_constante_fait_foi():
    """Backward compatibility: a mode on a constant table keeps its tt2 behaviour."""
    profile = terminal_time.TerminalProfile(
        mode="car", access_by_zone={"default": 120}, egress_by_zone={"default": 180},
        provenance="sourced", spatialise=False, labels={})
    assert profile.access_s("", "k") == 120
    assert profile.egress_s("zone inconnue", "k") == 180


def test_le_tirage_est_deterministe_par_trajet():
    """The same trip always draws the same — plans and LLM decisions are
    cached, an unstable draw would make a run diverge from its resumption."""
    profile = terminal_time.TerminalProfile(
        mode="car", access_by_zone={"default": 0}, egress_by_zone={"default": 0},
        provenance="sourced", spatialise=False, labels={},
        access_law_by_zone={"default": {0: 0.5, 300: 0.5}},
        egress_law_by_zone={"default": {0: 0.5, 300: 0.5}})
    for key in ("a", "b", "trajet:43.6,1.4→43.7,1.5"):
        assert profile.access_s("", key) == profile.access_s("", key)


def test_deux_trajets_tirent_independamment():
    """Otherwise all trips of a ring would get the same value, and the law
    would be useless."""
    profile = terminal_time.TerminalProfile(
        mode="car", access_by_zone={"default": 0}, egress_by_zone={"default": 0},
        provenance="sourced", spatialise=False, labels={},
        access_law_by_zone={"default": {0: 0.5, 300: 0.5}},
        egress_law_by_zone={"default": {0: 0.5, 300: 0.5}})
    drawn = {profile.access_s("", f"trajet-{i}") for i in range(50)}
    assert drawn == {0, 300}


def test_les_deux_bouts_tirent_independamment():
    """Access and egress must not be correlated at 1: the terminal time
    would then carry a single source of variation instead of two."""
    profile = terminal_time.TerminalProfile(
        mode="car", access_by_zone={"default": 0}, egress_by_zone={"default": 0},
        provenance="sourced", spatialise=False, labels={},
        access_law_by_zone={"default": {0: 0.5, 300: 0.5}},
        egress_law_by_zone={"default": {0: 0.5, 300: 0.5}})
    pairs = {(profile.access_s("", f"t{i}"), profile.egress_s("", f"t{i}"))
             for i in range(200)}
    assert len(pairs) == 4, pairs


def test_le_tirage_suit_la_loi():
    profile = terminal_time.TerminalProfile(
        mode="car", access_by_zone={"default": 0}, egress_by_zone={"default": 0},
        provenance="sourced", spatialise=False, labels={},
        access_law_by_zone={"default": {0: 0.9, 600: 0.1}},
        egress_law_by_zone={"default": {0: 1.0}})
    zeros = sum(profile.access_s("", f"t{i}") == 0 for i in range(4000))
    assert 0.86 < zeros / 4000 < 0.94, zeros / 4000


def test_config_refuse_une_loi_qui_ne_somme_pas_a_un(tmp_path):
    """A law that does not sum to 1 silences part of the mass without saying so."""
    _write_config(tmp_path, {"version": "x", "routing_version": "r",
                             "modes": {"car": {"access_law": {"default": {0: 0.5}},
                                               "egress_law": {"default": {0: 1.0}},
                                               "labels": _LABELS}}})
    with pytest.raises(ValueError, match="sums to"):
        terminal_time.terminal_profile("car")


def test_config_refuse_une_loi_vide(tmp_path):
    """Drawing from an empty law would return 0 — plausible, hence undetectable."""
    _write_config(tmp_path, {"version": "x", "routing_version": "r",
                             "modes": {"car": {"access_law": {"default": {}},
                                               "egress_law": {"default": {0: 1.0}},
                                               "labels": _LABELS}}})
    with pytest.raises(ValueError, match="empty"):
        terminal_time.terminal_profile("car")


def test_config_refuse_une_loi_sans_default(tmp_path):
    """A zone outside the EMC² layer would fall into the void, and the mode would become
    free again — the very bug the terminal time fixes."""
    _write_config(tmp_path, {"version": "x", "routing_version": "r",
                             "modes": {"car": {"access_law": {"Toulouse": {0: 1.0}},
                                               "egress_law": {"default": {0: 1.0}},
                                               "labels": _LABELS}}})
    with pytest.raises(ValueError, match="default"):
        terminal_time.terminal_profile("car")


def test_la_config_de_production_est_alignee_sur_lenquete():
    """Alignment guard: the served expectations must stay those of
    the survey (0.55 min for the car, 0.22 for the bike, all ends combined).

    Without it, a regression to the tt2 values (2 to 10 min) would go unnoticed —
    and that is exactly the regression that cost 2 composite score points to the run of
    2026-08-21.
    """
    car = terminal_time.terminal_profile("car")
    bike = terminal_time.terminal_profile("bicycle")
    total_car = (car.mean_s("access", "Toulouse")
                 + car.mean_s("egress", "Toulouse")) / 60
    total_bike = (bike.mean_s("access") + bike.mean_s("egress")) / 60
    assert 0.5 < total_car < 1.5, total_car      # Toulouse, the most expensive ring
    assert 0.1 < total_bike < 0.5, total_bike
    assert car.provenance == "sourced" and bike.provenance == "sourced"


def test_la_routing_version_de_production_est_r3_pour_le_fuseau_du_reseau():
    """Two moves locked here, not a cosmetic value.

    `r2`: the graph served at runtime is the one of the POLYGON of the
    453 communes, and the bike speeds of `config/osmnx.yaml` were completed. The
    key of the SQLite cache of direct itineraries does NOT carry the graph
    (`osmnx_persistent_cache.make_key` = `routing_version` + mode + coordinates): a
    return to `r1` would serve durations computed on the old 30 km disc — including
    the 70 km/h fallbacks of one 3rd-ring trip in six — without anything
    flagging it.

    `r3` (2026-09-04): the departure time is read in the NETWORK time zone and not in
    the process one. The key carries the time slot of the congestion factor, and
    nothing in a cache row says under which time zone it was computed — the
    `osmnx` replicas run in UTC, the `controller` in Europe/Paris.
    """
    assert terminal_time.routing_version() == "r3"


def test_le_bump_de_routing_version_change_la_cle_du_cache_direct(monkeypatch):
    """What makes the bump effective: the version enters the cache key.

    Without this dependency, moving to `r2` would be pointless — an entry computed on
    the old 30 km graph would be served as is on the polygon graph.
    """
    from datetime import datetime

    from trip_helper.osmnx_persistent_cache import OsmnxPersistentCache

    dt = datetime(2026, 3, 16, 8, 0)
    # Capitole → Villeneuve-du-Latou: a trip the 30 km rectangle did not route.
    args = dict(congestion_dt=dt, lat_from=43.6045, lon_from=1.4440,
                lat_to=43.2104, lon_to=1.4223)
    for mode in ("bicycle", "car"):
        key_r2, *_ = OsmnxPersistentCache.make_key(mode=mode, **args)
        monkeypatch.setattr(
            "trip_helper.terminal_time.routing_version", lambda: "r1", raising=True)
        key_r1, *_ = OsmnxPersistentCache.make_key(mode=mode, **args)
        monkeypatch.undo()
        assert key_r1 != key_r2, f"{mode}: the routing_version does not enter the key"
