"""Withdrawing an instance on daily quota: until the reset of ITS day, not ours.

Contract shared by both implementations of the RateLimiter port (memory and Redis).

What these tests would have caught on 2026-09-08: `mark_quota_exhausted_until` did not exist,
and the only path that set an instance aside went through the local counter — which emptied at
UTC midnight, i.e. 02:00 Paris time, for a Gemini quota that only reopened at 09:00. A key
refused by Google was therefore announced as available for seven hours.
"""

from datetime import UTC, datetime, timedelta

from llm_gateway.core.quota import next_quota_reset


class TestRetraitSurParoleDuFournisseur:
    """A "per day" 429 sets the instance aside without going through the local counter."""

    def test_l_instance_est_ecartee_immediatement(self, ports):
        assert ports.limiter.is_quota_exhausted("p_pacifique") is False
        ports.limiter.mark_quota_exhausted_until("p_pacifique", kind="rpd")
        assert ports.limiter.is_quota_exhausted("p_pacifique") is True

    def test_le_retrait_vise_le_reset_pacifique(self, ports):
        """The TTL must cover the real wait, not the ~30 s of the announced `retryDelay`."""
        attendu = (next_quota_reset("America/Los_Angeles") - datetime.now(UTC)).total_seconds()
        ttl = ports.limiter.mark_quota_exhausted_until("p_pacifique", kind="rpd")
        assert abs(ttl - attendu) <= 2
        assert ttl > 60, "a daily quota is never settled in less than a minute"

    def test_une_heure_de_reprise_explicite_est_respectee(self, ports):
        """When the provider gives its time, that is the one that counts."""
        cible = datetime.now(UTC) + timedelta(hours=3)
        ttl = ports.limiter.mark_quota_exhausted_until("p_pacifique", kind="rpd", until=cible)
        assert abs(ttl - 3 * 3600) <= 2
        assert ports.limiter.is_quota_exhausted("p_pacifique") is True

    def test_le_retrait_ne_touche_que_l_instance_visee(self, ports):
        """Two models on the same key have separate buckets: set aside only the one that refuses."""
        ports.limiter.mark_quota_exhausted_until("p_pacifique", kind="rpd")
        assert ports.limiter.is_quota_exhausted("p3") is False


class TestFenetreDuCompteur:
    """The day's counters are dated in the provider's time zone."""

    def test_le_compteur_suit_le_fuseau_du_fournisseur(self, ports):
        """Writing and reading must target the SAME day key.

        Their disagreement is what produced 49 requests counted for 500 consumed: if the
        read dated the day differently from the write, the counter read back would be 0.
        (A single reservation: the 60 s/rpm smoothing refuses two calls in the same
        microsecond, which says nothing about the dating.)
        """
        assert ports.limiter.daily_requests_local_seulement("p_pacifique") == 0
        assert ports.limiter.try_reserve("p_pacifique") is True
        assert ports.limiter.daily_requests_local_seulement("p_pacifique") == 1

    def test_les_tokens_du_jour_suivent_la_meme_fenetre(self, ports):
        ports.limiter.record_tokens("p_pacifique", 1_500)
        assert ports.limiter.daily_tokens_local_seulement("p_pacifique") == 1_500
