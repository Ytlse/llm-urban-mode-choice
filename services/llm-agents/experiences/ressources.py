"""Resources of remote decision-makers: instances, quotas, availability (ticket 035, spec 05).

Everything comes from the gateway and its configuration (`config/llm_gateway/providers.yaml`, `/health`),
never from a constant written here (Q1). An **instance** is a key + a quota; the decision-maker of an
experiment is a **model** + parameters (Q2): the admitted instances are those that serve
exactly this model.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from loguru import logger

_SOUS_CHEMIN_PROVIDERS = ("config", "llm_gateway", "providers.yaml")


def candidats_providers() -> list[Path]:
    """Locations probed for `providers.yaml`, in priority order.

    Since ticket 037 (iteration 2) the file lives OUTSIDE the package: at the repository root
    on the host, mounted under `/app/config/llm_gateway` in the containers. We walk up the
    module's ancestors instead of indexing a fixed `parents[N]`: this module lives under
    `services/llm-agents/experiences/` on the host (repository root two levels up) but under
    `/app/experiences/` in the container (`./llm-agents` is mounted on `/app`, so the
    root is ONE level up and `parents[2]` is `/`).
    """
    candidats: list[Path] = []
    depuis_env = os.environ.get("LLM_GATEWAY_PROVIDERS_FILE")
    if depuis_env:
        candidats.append(Path(depuis_env))
    for ancetre in Path(__file__).resolve().parents[:4]:
        candidats.append(ancetre.joinpath(*_SOUS_CHEMIN_PROVIDERS))
    candidats.append(
        Path("/app").joinpath(*_SOUS_CHEMIN_PROVIDERS)
    )  # container mount
    vus: set[str] = set()
    return [c for c in candidats if not (str(c) in vus or vus.add(str(c)))]


def chemin_providers() -> Path:
    """The first readable candidate; failing that the last one, so the error message cites a path."""
    candidats = candidats_providers()
    for c in candidats:
        if c.is_file():
            return c
    return candidats[-1]


def charger_providers(chemin: Path | None = None) -> dict[str, dict]:
    """{instance: configuration} from providers.yaml (root key `providers:` or flat)."""
    p = Path(chemin) if chemin else chemin_providers()
    if not p.is_file():
        # Deployment configuration missing: every `passerelle` decision-maker will be refused
        # for lack of an instance. Silent, this case sends one to check a file that was not read
        # (outage 2026-09-07).
        logger.error(
            f"[ALARME] no readable providers file — {diagnostic_providers(p)}"
        )
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    providers = data.get("providers", data) if isinstance(data, dict) else {}
    charges = {
        str(k): dict(v or {}) for k, v in providers.items() if isinstance(v, dict)
    }
    logger.debug(f"[ressources] {len(charges)} instance(s) read from {p}")
    return charges


def diagnostic_providers(
    chemin: Path | None = None, providers: dict | None = None
) -> str:
    """One line for error messages: WHICH file was read and HOW MANY instances it holds.

    A refusal that points to "providers.yaml" without saying which one was read costs time:
    this is exactly what happened on 2026-09-07 with a container that was not recreated.
    """
    p = Path(chemin) if chemin else chemin_providers()
    if p.is_file():
        n = len(providers) if providers is not None else len(charger_providers(p))
        return f"{n} instance(s) lue(s) dans {p}"
    env = os.environ.get("LLM_GATEWAY_PROVIDERS_FILE") or "non définie"
    sondes = ", ".join(str(c) for c in candidats_providers())
    return (
        f"aucun fichier lisible (LLM_GATEWAY_PROVIDERS_FILE : {env}) ; "
        f"{len(candidats_providers())} emplacements sondés : {sondes}"
    )


# Port of the local LM Studio server, and hosts that designate it from a container or from the
# machine itself. Same rule as `scripts/dashboard/lmstudio.est_instance_lmstudio`, rewritten
# here because this module runs in the `controller` container, without the dashboard
# package: `test_decider_scope.py` checks that both verdicts agree on the real
# providers.yaml, so that the duplication cannot diverge silently.
PORT_LMSTUDIO = 1234
_HOTES_LOCAUX = (
    "host.docker.internal",
    "gateway.docker.internal",
    "localhost",
    "127.0.0.1",
)
PORTEES = ("local", "distant")


def est_instance_locale(cfg: dict) -> bool:
    """Is this instance served by LM Studio on this machine?"""
    from urllib.parse import urlsplit

    if not isinstance(cfg, dict):
        return False
    u = urlsplit(str(cfg.get("base_url") or ""))
    if u.hostname not in _HOTES_LOCAUX:
        return False
    return (u.port or (443 if u.scheme == "https" else 80)) == PORT_LMSTUDIO


def portee_instance(cfg: dict) -> str:
    """`local` (LM Studio on this machine) or `distant` (a quota-bound API)."""
    return "local" if est_instance_locale(cfg) else "distant"


def instances_pour_modele(
    modele: str, providers: dict[str, dict], portee: str | None = None
) -> list[str]:
    """Instances that serve EXACTLY this model (Q2) — switching keys is not a substitution.

    `portee` restricts to the requested side. The same model identifier can be served on both
    sides — `qwen/qwen3.8-27b` is at Groq AND in LM Studio (two quantizations) — and without
    this filter the experiment would start on one and end on the other once the quota ran out,
    under a single name. `None` (definitions written before this field) returns both sides: the
    archive behaviour does not change; it is `refuser_si_impossible` that requires the scope
    when it removes a real ambiguity.
    """
    noms = sorted(
        nom
        for nom, cfg in providers.items()
        if str(cfg.get("default_model", "")) == modele
    )
    if portee is None:
        return noms
    return [n for n in noms if portee_instance(providers.get(n, {})) == portee]


def portees_pour_modele(modele: str, providers: dict[str, dict]) -> dict[str, list[str]]:
    """{scope: instances} for this model, empty scopes omitted — the basis of the ambiguity diagnostic."""
    out: dict[str, list[str]] = {}
    for nom in instances_pour_modele(modele, providers):
        out.setdefault(portee_instance(providers.get(nom, {})), []).append(nom)
    return {p: sorted(v) for p, v in sorted(out.items())}


def url_passerelle() -> str:
    return os.getenv("LLM_API_URL", "http://localhost:8000")


def lire_etat_passerelle(
    base_url: str | None = None, timeout: float = 5.0
) -> dict | None:
    """GET /health → {instance: {daily_requests, rpd_limit, quota_exhausted, available, current_rpm…}}; None if unreachable."""
    import httpx

    url = (base_url or url_passerelle()).rstrip("/") + "/health"
    try:
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
        return (r.json() or {}).get("providers") or {}
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"[ressources] passerelle injoignable ({url}) : {type(e).__name__}: {e}"
        )
        return None


# Default time zone of the daily reset when the config does not say. Gemini free tier
# counts its day in Pacific time: aiming at UTC midnight woke a run at
# 02:00 Paris time for a window that only reopened at 09:00 (incident 2026-09-08).
FUSEAU_QUOTA_DEFAUT = "America/Los_Angeles"


def prochaine_fenetre_quota(
    maintenant: datetime | None = None, fuseau: str | None = None
) -> str:
    """End of the daily window of `rpd` quotas: next midnight in `fuseau`, in UTC.

    This is only a FALLBACK: when the provider itself announces its reopening time
    (429 "per day"), that time prevails — cf. `runner._attendre_fenetre_quota`.
    """
    now = maintenant or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(fuseau or FUSEAU_QUOTA_DEFAUT)
    except Exception:  # noqa: BLE001 — fuseau inconnu / tzdata absente
        tz = timezone.utc
    local = now.astimezone(tz)
    lendemain = (local + timedelta(days=1)).date()
    minuit = datetime(lendemain.year, lendemain.month, lendemain.day, tzinfo=tz)
    return minuit.astimezone(timezone.utc).isoformat(timespec="seconds")


def hors_service(e: dict | None) -> bool:
    """Has the gateway taken the instance out of service — disabled after consecutive
    errors (402 credits, 5xx…) or in cooldown?

    `/health` publishes `disabled` and `cooldown`, which say exactly that. `available`, for its
    part, answers another question — "can it take a request NOW?" — and is
    also false when the instance is simply BUSY (`active_tasks` ≥ `concurrency_limit`).
    On 2026-09-08, read as "out of service", it had an experiment on a local model with one call
    at a time declared exhausted until the next day, two decisions after it resumed.
    Without these two fields (older gateway), `available` remains the only signal and prevails.
    """
    e = e or {}
    if "disabled" in e or "cooldown" in e:
        return bool(e.get("disabled")) or bool(e.get("cooldown"))
    return not e.get("available", True)


def occupee(e: dict | None) -> bool:
    """Servable but saturated: all its concurrent calls are taken. Information, never a refusal."""
    e = e or {}
    return (not e.get("available", True)) and not hors_service(e) and not bool(e.get("quota_exhausted"))


class MoniteurRessources:
    """View of a decision-maker's admitted instances: margin, availability, exhaustion (Q1, Q4, Q11)."""

    def __init__(
        self,
        instances: list[str],
        providers: dict[str, dict],
        base_url: str | None = None,
        lecteur=lire_etat_passerelle,
    ):
        self.instances = list(instances)
        self.providers = providers
        self.base_url = base_url or url_passerelle()
        self._lecteur = lecteur
        self.etat: dict[str, dict] = {}
        self.joignable: bool | None = None
        self.compteurs = {"429": 0, "402": 0, "substitution_refusee": 0}
        # Age of the snapshot, on the monotonic clock: `rafraichir_si_perime` uses it to
        # avoid hammering `/health`. None = never read, hence stale by default.
        self.maj_monotone: float | None = None
        # Rising edges of the two traces of the declared cap (ticket 097). Since
        # 2026-09-21 `rpd_limit` no longer excludes an instance: all that remains of it is these
        # logs, and without them the repeal would remove the safeguard AND the measurement.
        self._plafond_franchi: set[str] = set()
        self._refus_signale: set[str] = set()

    def rafraichir(self) -> None:
        etat = self._lecteur(self.base_url)
        self.joignable = etat is not None
        if etat is not None:
            self.etat = {i: etat.get(i, {}) for i in self.instances}
            # The age restarts ONLY on a successful read: an unreachable gateway leaves
            # the previous snapshot in place (fail-safe) and leaves it stale, hence retried.
            self.maj_monotone = time.monotonic()
            # A single read of `/health` ⇒ a single observation point. Above all not
            # in `disponible()`, called at every decision: the log would be flooded.
            self._journaliser_plafond()

    def _journaliser_plafond(self) -> None:
        """Logs what the declared cap no longer decides (ticket 097).

        `rpd_limit` is copied by hand from provider documentation that is outdated,
        silent or gone; it no longer excludes a key. Two facts remain to be recorded, on a
        rising edge so as not to repeat the same line at every refresh:

        1. the provider serves BEYOND the announced cap — the file value is too
           low, and this is the only chance to learn it;
        2. the provider refuses — `daily_requests` at that moment IS its real limit, the one
           the declared cap claimed to know.
        """
        for i in self.instances:
            e = self.etat.get(i) or {}
            limite = (self.providers.get(i) or {}).get("rpd_limit")
            compteur = e.get("daily_requests")
            if limite is None or compteur is None:
                # Nothing measured to compare: stay silent rather than assume a figure —
                # this is exactly the defect that ticket 097 fixes.
                continue
            limite, consomme = int(limite), int(compteur)

            # (1) Silent overrun. Strictly beyond: at `consomme == limite`, the
            # cap is not yet disproved.
            if consomme > limite and not e.get("quota_exhausted"):
                if i not in self._plafond_franchi:
                    self._plafond_franchi.add(i)
                    logger.warning(
                        f"[ressources] [PLAFOND] {i}: {consomme}/{limite} requests/day — "
                        f"declared cap exceeded by {consomme - limite} and the provider "
                        f"still serves; providers.yaml value to re-measure "
                        f"(x-ratelimit-* headers)"
                    )
            elif consomme <= limite:
                # Counter back under the cap: next window, the trace will fire again.
                self._plafond_franchi.discard(i)

            # (2) Real refusal: the measured limit, at last.
            if e.get("quota_exhausted"):
                if i not in self._refus_signale:
                    self._refus_signale.add(i)
                    logger.warning(
                        f"[ressources] [PLAFOND] {i} refused by the provider (429) at "
                        f"{consomme} requests/day — real limit observed; declared cap "
                        f"{limite} (gap {consomme - limite:+d})"
                    )
            else:
                self._refus_signale.discard(i)

    def reinitialiser_quotas(self, instances: list[str] | None = None) -> bool:
        """Lifts the local quota-exhaustion locks (gateway / Redis) to test the keys live."""
        import httpx

        cibles = list(instances) if instances is not None else self.instances
        url = self.base_url.rstrip("/") + "/providers/reset-quota"
        reussi = False
        try:
            r = httpx.post(url, json={"providers": cibles}, timeout=5.0)
            if r.is_success:
                reussi = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ressources] quota reset via gateway not performed ({e})")

        try:
            import redis
            redis_url = os.getenv("REDIS_URL") or os.getenv("LLM_GATEWAY_REDIS__URL")
            if redis_url:
                cli = redis.Redis.from_url(redis_url)
                for inst in cibles:
                    cli.delete(f"quota_exhausted:{inst}")
                reussi = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ressources] direct Redis quota reset not performed ({e})")
        return reussi

    def perime(self, age_max_s: float) -> bool:
        """Has the snapshot exceeded `age_max_s`? Never read ⇒ stale."""
        if self.maj_monotone is None:
            return True
        return (time.monotonic() - self.maj_monotone) >= age_max_s

    def rafraichir_si_perime(self, age_max_s: float) -> bool:
        """Rereads `/health` only if the snapshot is too old. Returns True if it read.

        Without this, `etat` stayed frozen on the STARTUP read for the whole run:
        a key that ran out along the way remained "available" in the eyes of
        `instances_disponibles`, so `DecideurPasserelle._prochaine_instance` kept
        pinning it and the next key — 500 untouched requests — was never started
        (incident of 2026-09-08: run stopped at 70.5 % with a full bucket in reserve).
        """
        if not self.perime(age_max_s):
            return False
        self.rafraichir()
        return True

    def marge(self, instance: str) -> int | None:
        """Requests left in today's window — None if the instance has no daily limit."""
        cfg = self.providers.get(instance, {})
        limite = cfg.get("rpd_limit")
        if limite is None:
            return None
        consomme = int((self.etat.get(instance) or {}).get("daily_requests") or 0)
        return max(0, int(limite) - consomme)

    def disponible(self, instance: str, besoin: int = 1) -> bool:
        e = self.etat.get(instance) or {}
        if hors_service(e):
            # The gateway took the instance out of service (credits exhausted, consecutive
            # errors, cooldown). The go/no-go did not read this until 2026-09-07
            # (`cerebras_gpt-oss-120b` at 402 with `quota_exhausted: false`: admitted, then
            # nine requests burned), then it read `available` until 2026-09-08 — which
            # is also false when the instance is merely BUSY: cf. `hors_service`.
            return False
        if e.get("quota_exhausted"):
            return False
        # The local cap is declarative and informative: we no longer block in advance
        # (marge < besoin), so as to measure real overruns. Only a real 429
        # setting `quota_exhausted: True` excludes the instance.
        return True

    def instances_disponibles(self, besoin: int = 1) -> list[str]:
        return [i for i in self.instances if self.disponible(i, besoin)]

    def epuise(self, besoin: int = 1) -> bool:
        return not self.instances_disponibles(besoin)

    def raison_epuisement(self) -> str:
        parts = []
        for i in self.instances:
            cfg = self.providers.get(i, {})
            e = self.etat.get(i) or {}
            parts.append(
                f"{i} : {e.get('daily_requests', '?')}/{cfg.get('rpd_limit', '∞')} requêtes/jour"
                + (" (épuisée)" if e.get("quota_exhausted") else "")
                + (" (désactivée côté passerelle : erreurs consécutives ou cooldown)" if hors_service(e) else "")
            )
        return " ; ".join(parts) or "aucune instance"

    def tableau(self) -> list[dict]:
        """What the user sees during the run (Q1)."""
        lignes = []
        for i in self.instances:
            cfg = self.providers.get(i, {})
            e = self.etat.get(i) or {}
            lignes.append(
                {
                    "instance": i,
                    "modele": cfg.get("default_model"),
                    "requetes_jour": e.get("daily_requests"),
                    "limite_jour": cfg.get("rpd_limit"),
                    "marge": self.marge(i),
                    "jetons_jour": e.get("daily_tokens"),
                    "limite_jetons_jour": cfg.get("tpd_limit"),
                    "rpm_observe": e.get("current_rpm"),
                    "rpm_limite": cfg.get("rpm_limit"),
                    "epuisee": bool(e.get("quota_exhausted")),
                    "disponible": not hors_service(e),
                    "occupee": occupee(e),
                }
            )
        return lignes


__all__ = [
    "MoniteurRessources",
    "charger_providers",
    "chemin_providers",
    "est_instance_locale",
    "instances_pour_modele",
    "portee_instance",
    "portees_pour_modele",
    "lire_etat_passerelle",
    "FUSEAU_QUOTA_DEFAUT",
    "prochaine_fenetre_quota",
    "url_passerelle",
]
