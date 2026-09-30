"""The daily request counter says, in its own name, that it only sees this process.

What these tests would have caught: on 2026-09-23, `redis-cli GET rpd:google_gemini31_key1` was
read as a remaining budget and got a relaunch postponed by a day, while the code already said
the opposite in a docstring — which is not where the number is read.

They lock three things: the key name carries its limit, so does the accessor name, and
the function that would decide from this counter no longer exists.
"""

from __future__ import annotations

import inspect

from llm_gateway.infra.memory import rate_limiter as memoire
from llm_gateway.infra.redis import rate_limiter as redis_rl
from llm_gateway.ports import rate_limiter as port


class TestNomDeLaCle:
    def test_le_prefixe_rpd_dit_qu_il_est_local(self):
        """`redis-cli --scan` is what gets read when looking for where a quota stands."""
        assert redis_rl.RPD_KEY_PREFIX == "rpd_local_seulement:"

    def test_le_prefixe_tpd_dit_qu_il_est_local(self):
        assert redis_rl.TPD_KEY_PREFIX == "tpd_local_seulement:"

    def test_le_verrou_de_quota_garde_son_prefixe(self):
        """This one is authoritative — it is set on a provider 429, not on a counter."""
        assert redis_rl.QUOTA_EXHAUSTED_PREFIX == "quota_exhausted:"


class TestNomDeLAccesseur:
    def test_les_deux_implementations_exposent_le_nom_suffixe(self):
        for impl in (redis_rl.RedisRateLimiter, memoire.InMemoryRateLimiter):
            assert hasattr(impl, "daily_requests_local_seulement"), impl.__name__
            assert hasattr(impl, "daily_tokens_local_seulement"), impl.__name__

    def test_l_ancien_nom_n_existe_plus(self):
        """No alias: a neutral name that still works is an invitation to get it wrong again."""
        for impl in (redis_rl.RedisRateLimiter, memoire.InMemoryRateLimiter):
            assert not hasattr(impl, "daily_requests"), impl.__name__
            assert not hasattr(impl, "daily_tokens"), impl.__name__

    def test_le_contrat_de_port_porte_le_meme_nom(self):
        source = inspect.getsource(port)
        assert "daily_requests_local_seulement" in source
        assert "daily_tokens_local_seulement" in source


class TestCodeMort:
    def test_mark_quota_exhausted_depuis_le_compteur_local_a_disparu(self):
        """No caller, and looking like live code: removed."""
        assert not hasattr(redis_rl.RedisRateLimiter, "_mark_quota_exhausted")

    def test_la_voie_qui_fait_autorite_reste(self):
        """The one that relies on the provider's 429 stays put."""
        for impl in (redis_rl.RedisRateLimiter, memoire.InMemoryRateLimiter):
            assert hasattr(impl, "mark_quota_exhausted_until"), impl.__name__


class TestLeCompteurNeFermeToujoursRien:
    def test_un_compteur_au_dela_du_plafond_n_ecarte_pas_la_cle(self, ports):
        """Relocked here: only the provider closes a key."""
        limiter = ports.limiter
        for _ in range(50):
            limiter.record_tokens("p_pacifique", 10_000)
        assert limiter.is_quota_exhausted("p_pacifique") is False
