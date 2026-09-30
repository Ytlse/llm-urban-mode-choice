"""The synthetic school bus: who may take it, when, and how it is shown and counted."""

# ==============================================================================
# SYNTHETIC SCHOOL BUS
# ==============================================================================

def test_tf22_car_scolaire_eligibilite_age():
    """TF-22: Strict eligibility by age band 5-17 years."""
    import trip_helper.school_bus as sb
    cfg = sb._config()
    assert cfg.age_min == 5
    assert cfg.age_max == 17


def test_tf23_car_scolaire_eligibilite_hors_tisseo():
    """TF-23: Territorial eligibility restricted to residences not served by Tisséo."""
    from models import Activity, Location, Person, PersonalIdentity
    from trip_helper.school_bus import build_school_bus_option

    home_tisseo = Location(lon=1.44, lat=43.60, public_transport=True)
    home_rural = Location(lon=1.10, lat=43.20, public_transport=False)
    school = Location(lon=1.15, lat=43.25, public_transport=False)
    edu = Activity(id="edu", start_time=28800, end_time=57600, purpose="education", location=school)

    p_tisseo = Person(person_id="p1", identity=PersonalIdentity(name="A", traits_json={"age": 12}, home=home_tisseo, activities=[edu]))
    p_rural = Person(person_id="p2", identity=PersonalIdentity(name="B", traits_json={"age": 12}, home=home_rural, activities=[edu]))

    # In the Tisséo zone (public_transport=True), no school_bus option
    assert build_school_bus_option(p_tisseo, home_tisseo, edu, 28800, 28800) is None
    # Outside the Tisséo zone (public_transport=False), school_bus option generated
    opt_rural = build_school_bus_option(p_rural, home_rural, edu, 28800, 28800)
    assert opt_rural is not None


def test_tf24_car_scolaire_motif_etudes():
    """TF-24: Activation of the school service only on a school/study purpose."""
    from models import Activity, Location, Person, PersonalIdentity
    from trip_helper.school_bus import build_school_bus_option
    home = Location(lon=1.10, lat=43.20, public_transport=False)
    school = Location(lon=1.15, lat=43.25, public_transport=False)
    shop = Activity(id="s", start_time=28800, end_time=32000, purpose="shop", location=school)
    p = Person(person_id="p", identity=PersonalIdentity(name="A", traits_json={"age": 12}, home=home, activities=[shop]))
    assert build_school_bus_option(p, home, shop, 28800, 28800) is None


def test_tf25_car_scolaire_gratuite():
    """TF-25: The school bus service is free (the mention in the text SERVED to the model).

    The text switched to English in ticket 074: it is « free », not « gratuit ». Both
    are accepted because the archived traces still carry the French — but it really is
    the free-of-charge status that is checked, not a word: without it, a pupil compares a free bus
    with a paid pass without knowing it.
    """
    from models import Activity, Location, Person, PersonalIdentity
    from text_helper import env_ob_to_text
    from trip_helper.school_bus import build_school_bus_option
    home = Location(lon=1.10, lat=43.20, public_transport=False)
    school = Location(lon=1.15, lat=43.25, public_transport=False)
    edu = Activity(id="edu", start_time=28800, end_time=57600, purpose="education", location=school)
    p = Person(person_id="p", identity=PersonalIdentity(name="A", traits_json={"age": 12}, home=home, activities=[edu]))
    plan = build_school_bus_option(p, home, edu, 28800, 28800)
    assert plan is not None
    text = env_ob_to_text("travel_plan", plan.model_dump())
    assert "free" in text.lower() or "gratuit" in text.lower()


def test_tf26_car_scolaire_plage_horaire():
    """TF-26: Time opportunity window of 30 minutes."""
    import trip_helper.school_bus as sb
    cfg = sb._config()
    assert cfg.schedule_margin_minutes == 30.0


def test_tf27_car_scolaire_categorisation_tc():
    """TF-27: Inclusion of the school bus in the Public Transport category."""
    from mobility_llm.mode_choice import canonical_mode

    from scripts.synthesis.formule_score.metrics import categorize_mode
    assert categorize_mode("school_bus") == "transports_collectifs"
    assert canonical_mode("school_bus") == "public_transport"


def test_tf28_car_scolaire_rendu_gama_direct_car():
    """TF-28: GAMA direct route marker (__DIRECT_CAR__)."""
    import trip_helper.school_bus as sb
    assert sb.SCHOOL_BUS_ROUTE_MARKER == "__DIRECT_CAR__"


def test_tf29_car_scolaire_compteurs_options_choix():
    """TF-29: Double logging of the school options proposed and chosen."""
    import trip_helper.school_bus as sb
    assert hasattr(sb, "SCHOOL_BUS_OPTIONS")
    assert hasattr(sb, "SCHOOL_BUS_CHOSEN")
