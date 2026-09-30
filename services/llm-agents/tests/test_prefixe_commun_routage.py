"""A control does not recompute cold a trip dating from before the article."""

import asyncio

import pytest

from models import Location
from settings import settings
from trip_helper import cached_triphelper, strict_replay


def test_cache_otp_absent_arrete_avant_l_appel_reseau(monkeypatch):
    class CacheVide:
        async def is_blacklisted_async(self, _key):
            return False

        async def lookup_async(self, _key):
            return None

        async def blacklist_add_async(self, _key):
            return None

    class Fournisseur:
        appels = 0

        async def get_itineraries(self, *_args, **_kwargs):
            self.appels += 1
            return []

    monkeypatch.setattr(cached_triphelper, "_otp_persistent_cache", CacheVide())
    monkeypatch.setattr(settings.llm, "rejeu_strict_avant_ts", 100)
    monkeypatch.setattr(strict_replay, "_ECHEC_ROUTE_STRICT", None)
    fournisseur = Fournisseur()
    helper = cached_triphelper.OtpCachedTripHelper(fournisseur)
    lieu = Location(lat=43.6, lon=1.44)

    async def verifier():
        with pytest.raises(RuntimeError, match="cache OTP absent"):
            await helper.get_itineraries(lieu, lieu, 99)
        assert fournisseur.appels == 0
        assert strict_replay.echec_route_strict() is not None
        await helper.get_itineraries(lieu, lieu, 100)
        assert fournisseur.appels == 1

    try:
        asyncio.run(verifier())
    finally:
        monkeypatch.setattr(strict_replay, "_ECHEC_ROUTE_STRICT", None)
