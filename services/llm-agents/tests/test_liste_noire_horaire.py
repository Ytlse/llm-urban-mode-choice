"""A pair without public transport at 5 a.m. is not a pair without transport.

OTP's blacklist avoids calling the router again for a pair known to return
nothing. Its key carried only the coordinates, whereas `cached_triphelper` fills it as soon as the
result is **empty**, whatever the reason — and `noTransitConnectionInSearchWindow` is time-dependent:
measured on 2026-09-04, 29 points without itinerary at 6:00 against **341 at 5:00** out of the same 2,580.
A pair blacklisted in the early morning therefore returned "no public transport" at 17:00, with no call
and no log — and the pre-planning waves query the same pair at successive
times, so the defect triggered within a single run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import Location  # noqa: E402
from trip_helper.otp_persistent_cache import OtpPersistentCache  # noqa: E402

_O = Location(lat=43.6045, lon=1.4440)
_D = Location(lat=43.5290, lon=1.3270)
# 2026-03-16, GAMA wall clock (naive-as-UTC): 5:00, 6:00, 17:00.
_5H, _6H, _17H = 1773637200, 1773640800, 1773680400


def test_deux_heures_differentes_donnent_deux_cles():
    assert (OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
            != OtpPersistentCache.make_blacklist_key(_O, _D, _17H))


def test_la_meme_heure_donne_la_meme_cle():
    assert (OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
            == OtpPersistentCache.make_blacklist_key(_O, _D, _5H))


def test_le_creneau_est_de_dix_minutes_comme_le_cache_de_plans():
    """Two departures in the same ten-minute slot share the key: that is what still
    saves calls within a pre-planning wave."""
    assert (OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
            == OtpPersistentCache.make_blacklist_key(_O, _D, _5H + 599))
    assert (OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
            != OtpPersistentCache.make_blacklist_key(_O, _D, _5H + 601))


def test_le_sens_du_trajet_compte():
    assert (OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
            != OtpPersistentCache.make_blacklist_key(_D, _O, _5H))


def test_sans_heure_la_cle_ne_collisionne_pas_avec_une_cle_horaire():
    """A caller without a time keeps a topology key, but it is marked: the two
    families must not mix, otherwise an entry without a time would mask all the
    times — the defect being closed."""
    sans = OtpPersistentCache.make_blacklist_key(_O, _D)
    assert sans != OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
    assert sans == OtpPersistentCache.make_blacklist_key(_O, _D)


def test_une_paire_noircie_a_cinq_heures_est_rejouee_a_dix_sept(tmp_path):
    """The end-to-end behaviour, on a real database."""
    cache = OtpPersistentCache(str(tmp_path))
    k5 = OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
    k17 = OtpPersistentCache.make_blacklist_key(_O, _D, _17H)

    cache.blacklist_add(k5)

    assert cache.is_blacklisted(k5), "the pair stays blacklisted at its time"
    assert not cache.is_blacklisted(k17), "but it is queried again at another time"


def test_l_heure_est_celle_du_reseau_pas_celle_du_processus(monkeypatch):
    """The key goes through `sim_clock`, hence through the GTFS feeds' time zone: the
    process `TZ` must not shift it."""
    import os
    import time as _time
    avant = OtpPersistentCache.make_blacklist_key(_O, _D, _5H)
    monkeypatch.setenv("TZ", "America/New_York")
    if hasattr(_time, "tzset"):
        _time.tzset()
    try:
        assert OtpPersistentCache.make_blacklist_key(_O, _D, _5H) == avant
    finally:
        os.environ.pop("TZ", None)
        if hasattr(_time, "tzset"):
            _time.tzset()
