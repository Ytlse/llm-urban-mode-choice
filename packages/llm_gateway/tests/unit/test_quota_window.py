"""Daily quota window: the right time zone, and the `retryDelay` put back in its place.

Reference incident (2026-09-08). An experiment stops at 10%: `google_gemini35_key2`
answers 429 in a loop for 15 min. Google's body is explicit —
`quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: 500` — but two
things made it unreadable for the platform:

1. the internal counters were dated in UTC, hence emptied at 02:00 Paris time, while the
   Gemini billing day only ended at 09:00 (Pacific midnight). Redis showed 49 requests
   out of 500 when Google refused for exceeding the 500;
2. the announced `retryDelay` was 0.7 s to 57 s — a per-minute rate delay, which does NOT
   measure the time until the reset. Following it looped indefinitely.

These tests freeze both points on the values actually observed that day.
"""

from datetime import UTC, datetime

from llm_gateway.core.quota import (
    is_daily_quota_error,
    next_quota_reset,
    quota_day,
    seconds_until_quota_reset,
)

# Real body of the 429 of 2026-09-08 (taken from experiments/.../llm_errors.jsonl), reduced to
# what carries the meaning: the quotaId, and a short retryDelay that says nothing of the reset.
CORPS_429_JOURNALIER = """{"error": {"code": 429,
 "message": "You exceeded your current quota... Quota exceeded for metric:
   generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 500,
   model: gemini-3.5-flash-lite. Please retry in 33.173905726s.",
 "status": "RESOURCE_EXHAUSTED",
 "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
   "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                   "quotaValue": "500"}]},
  {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "33s"}]}}"""

CORPS_429_PAR_MINUTE = """{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
 "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
   "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                   "quotaValue": "15"}]}]}}"""


class TestNatureDuRefus:
    def test_le_429_reel_du_8_septembre_est_journalier(self):
        assert is_daily_quota_error(CORPS_429_JOURNALIER) is True

    def test_un_429_par_minute_ne_l_est_pas(self):
        """Not to be confused: this one is handled by a short cooldown, not by a wait."""
        assert is_daily_quota_error(CORPS_429_PAR_MINUTE) is False

    def test_corps_absent_ou_vide(self):
        assert is_daily_quota_error(None) is False
        assert is_daily_quota_error("") is False

    def test_libelles_en_clair(self):
        assert is_daily_quota_error("limit: RequestsPerDay exceeded") is True
        assert is_daily_quota_error("quota per day exhausted") is True
        assert is_daily_quota_error("TokensPerDay limit reached") is True


class TestFenetreDuJour:
    # 2026-09-08 06:44 UTC = 08:44 in Paris = 23:44 on the 7th in the Pacific.
    INSTANT_DU_BLOCAGE = datetime(2026, 9, 8, 6, 44, tzinfo=UTC)

    def test_les_deux_fuseaux_ne_datent_pas_la_meme_journee(self):
        """The root cause: at 08:44 Paris, the UTC counter has rolled over, Google's has not."""
        assert quota_day("UTC", self.INSTANT_DU_BLOCAGE) == "20260908"
        assert quota_day("America/Los_Angeles", self.INSTANT_DU_BLOCAGE) == "20260907"

    def test_la_reouverture_tombe_a_9h_a_paris(self):
        """Measured: the probes on both keys return to HTTP 200 at 09:01 Paris time."""
        assert next_quota_reset("America/Los_Angeles", self.INSTANT_DU_BLOCAGE) == datetime(
            2026, 9, 8, 7, 0, tzinfo=UTC
        )

    def test_l_attente_reelle_depasse_de_loin_le_retry_delay_annonce(self):
        """33 s announced by Google against 16 min of real wait: the delay must be discarded."""
        restant = seconds_until_quota_reset("America/Los_Angeles", self.INSTANT_DU_BLOCAGE)
        assert restant == 960
        assert restant > 33

    def test_passage_a_l_heure_d_hiver_pacifique(self):
        """On 1 November 2026, the Pacific goes back to PST: local midnight = 08:00 UTC."""
        apres = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
        assert next_quota_reset("America/Los_Angeles", apres) == datetime(
            2026, 11, 6, 8, 0, tzinfo=UTC
        )

    def test_passage_a_l_heure_d_ete_pacifique(self):
        """End of March, PDT: local midnight = 07:00 UTC."""
        ete = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
        assert next_quota_reset("America/Los_Angeles", ete) == datetime(
            2026, 3, 31, 7, 0, tzinfo=UTC
        )

    def test_fuseau_inconnu_retombe_sur_utc_sans_lever(self):
        """An unreadable time zone (tzdata missing) must not prevent the gateway from starting."""
        t = datetime(2026, 9, 8, 6, 44, tzinfo=UTC)
        assert next_quota_reset("Mars/Olympus_Mons", t) == datetime(
            2026, 9, 9, 0, 0, tzinfo=UTC
        )
        assert next_quota_reset(None, t) == datetime(2026, 9, 9, 0, 0, tzinfo=UTC)

    def test_l_attente_est_toujours_strictement_positive(self):
        """One second before the reset, we wait 1 s — never 0, which would loop the caller."""
        juste_avant = datetime(2026, 9, 8, 6, 59, 59, tzinfo=UTC)
        assert seconds_until_quota_reset("America/Los_Angeles", juste_avant) >= 1


class TestReinitialisationQuota:
    def test_in_memory_clear_quota_exhausted(self):
        from llm_gateway.config import ProviderConfig
        from llm_gateway.infra.memory.rate_limiter import InMemoryRateLimiter

        cfg = ProviderConfig(
            rpm_limit=15,
            rpd_limit=500,
            adapter="google",
            base_url="https://example.com",
            default_model="gemini-test",
        )
        limiter = InMemoryRateLimiter({"p1": cfg})
        limiter.mark_quota_exhausted_until("p1", kind="rpd")
        assert limiter.is_quota_exhausted("p1") is True

        limiter.clear_quota_exhausted("p1")
        assert limiter.is_quota_exhausted("p1") is False
