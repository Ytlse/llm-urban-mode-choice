"""The bench's two clients: the stub, and the real one — ticket 100, functional tests.

THE STUB serves family A: it returns a declared answer, with no network and no token. It makes
it possible to play a whole chain — injection, memory, trace, measure, figure — for zero calls.

THE REAL ONE serves family B, and it carries the gateway constraint.

⚠ **NO FALLBACK, EVER.** The author decided on 2026-09-22: Groq only today, other
models for long runs, and **if the quota is exhausted we wait**. A measurement obtained on
a gateway that was not declared is not the measurement one thinks one is reading — and this
repository has already paid the price of a silent fallback.

⚠ **The limit that bites is not the RPM.** Groq was dropped from campaigns on 2026-09-08 for
a measured limit of **1,000 OUTPUT tokens per minute**, invisible outside the body of the 429s,
and the gateway was not fixed. The 30 req/min and 1,000 req/day will never be reached
here; the OTPM will. Hence: calls in SERIES, a tight `max_tokens`, and an output-token
budget that the bench watches itself rather than discovering the limit in a silent 429.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

# The instances declared for TODAY. A parameter, not a constant: long-running tests
# will move to other models, and the bench writes into each result the one that served —
# otherwise two campaigns on two gateways would be incomparable without anyone knowing.
GROQ = ("groq_openai_120_key1", "groq_openai_20_key1", "groq_qwen_qwen3_8_27b_key1")

# OUTPUT token budget. At 1,000/minute measured, 6,000 tokens are worth six minutes of waiting
# at worst. Exceeding it STOPS the bench: a clean stop is better than a burst of 429s on a
# key shared with the campaigns.
BUDGET_JETONS_SORTIE = 6000

# Pause between two calls, in seconds. Deliberately coarse: 1,000 tokens/minute, ~200
# tokens per answer, i.e. five answers per minute at most.
PAUSE_ENTRE_APPELS = 12.0


class BudgetEpuise(RuntimeError):
    """The bench stops rather than overflowing onto the campaigns' quota."""


class PasserelleRefusee(RuntimeError):
    """An undeclared instance served, or was about to serve."""


@dataclass
class Appel:
    """What is kept of each call. It is the material of `passerelle.csv`."""

    categorie: str
    agent_id: str
    etiquette: str
    instance: str = ""
    jetons_sortie: int = 0
    secondes: float = 0.0
    hors_schema: bool = False
    erreur: str = ""


@dataclass
class Journal:
    appels: list[Appel] = field(default_factory=list)

    @property
    def jetons_sortie(self) -> int:
        return sum(a.jetons_sortie for a in self.appels)

    def resume(self) -> str:
        hors = sum(1 for a in self.appels if a.hors_schema)
        erreurs = sum(1 for a in self.appels if a.erreur)
        par_instance: dict[str, int] = {}
        for a in self.appels:
            par_instance[a.instance or "?"] = par_instance.get(a.instance or "?", 0) + 1
        return (
            f"{len(self.appels)} appel(s), {self.jetons_sortie} jeton(s) de sortie, "
            f"{hors} hors schéma, {erreurs} en erreur — instances : {par_instance}"
        )

    def ecrire(self, chemin: Path) -> None:
        import csv

        chemin.parent.mkdir(parents=True, exist_ok=True)
        with chemin.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["categorie", "agent_id", "etiquette", "instance",
                        "jetons_sortie", "secondes", "hors_schema", "erreur"])
            for a in self.appels:
                w.writerow([a.categorie, a.agent_id, a.etiquette, a.instance,
                            a.jetons_sortie, round(a.secondes, 2),
                            "oui" if a.hors_schema else "", a.erreur])


# ── The stub client ─────────────────────────────────────────────────────────────────────────
class _Rendu:
    def __init__(self, **champs):
        for k, v in champs.items():
            setattr(self, k, v)


class _Reponse:
    def __init__(self, agents, provider_used="stub"):
        self.agents = agents
        self.provider_used = provider_used


class ClientStub:
    """Returns DECLARED answers. No network, no token, no latency.

    `reponses` maps a category to a list of dictionaries served in order, then in a
    loop on the last one. A missing category raises: a silent stub would pass an untested
    path off as a tested one.
    """

    def __init__(self, reponses: dict[str, list[dict]]) -> None:
        self._reponses = {k: list(v) for k, v in reponses.items()}
        self.appels: list[dict] = []

    async def execute(self, payload: dict):
        categorie = str(payload.get("category") or "")
        self.appels.append(payload)
        file = self._reponses.get(categorie)
        if not file:
            raise AssertionError(
                f"the bench called the category « {categorie} », for which no response "
                f"is declared. A silent stub would pass an untested path off as tested."
            )
        rendu = file.pop(0) if len(file) > 1 else file[0]
        agents = payload.get("agents") or [{}]
        return _Reponse([_Rendu(agent_id=str(agents[0].get("agent_id", "")), **rendu)])


# ── The real client, pinned ────────────────────────────────────────────────────────────────
class ClientEpingle:
    """The gateway client, restricted to the DECLARED instances, with no fallback.

    It counts output tokens, serialises calls, and refuses to go on beyond the
    budget. It catches nothing: an error goes up, logged, and the bench decides.
    """

    def __init__(
        self,
        instances: tuple[str, ...] = GROQ,
        *,
        base_url: str = "http://localhost:8000",
        budget_jetons: int = BUDGET_JETONS_SORTIE,
        pause: float = PAUSE_ENTRE_APPELS,
        journal: Journal | None = None,
    ) -> None:
        if not instances:
            raise PasserelleRefusee(
                "no instance declared. The bench does not choose the gateway for "
                "you: declare it, it is written in each result."
            )
        from llm_gateway.sdk.client import LLMGatewayClient

        self.instances = tuple(instances)
        self._budget = int(budget_jetons)
        self._pause = float(pause)
        self.journal = journal or Journal()
        self._dernier_appel = 0.0
        self._client = LLMGatewayClient(
            base_url=base_url, instances_admises=list(self.instances)
        )

    async def execute(self, payload: dict, *, etiquette: str = ""):
        """One call, counted. Raises `BudgetEpuise` rather than overflowing."""
        if self.journal.jetons_sortie >= self._budget:
            raise BudgetEpuise(
                f"budget reached: {self.journal.jetons_sortie} output tokens out of "
                f"{self._budget}. The bench stops — it does not overflow onto the campaigns' "
                f"quota. Run it again with a higher declared budget if that is intended."
            )
        # The instances travel IN the payload AND in the client: the gateway reads
        # one or the other depending on the path, and leaving the choice open on one side would
        # be enough to have an undeclared instance serve.
        payload = dict(payload)
        payload["instances_admises"] = list(self.instances)

        attente = self._pause - (time.monotonic() - self._dernier_appel)
        if self._dernier_appel and attente > 0:
            await asyncio.sleep(attente)

        agents = payload.get("agents") or [{}]
        trace = Appel(
            categorie=str(payload.get("category") or ""),
            agent_id=str(agents[0].get("agent_id", "")),
            etiquette=etiquette,
        )
        debut = time.monotonic()
        try:
            reponse = await self._client.execute(payload)
        except Exception as err:  # noqa: BLE001 — journalisé puis relevé, jamais avalé
            trace.erreur = f"{type(err).__name__}: {err}"[:200]
            trace.secondes = time.monotonic() - debut
            self.journal.appels.append(trace)
            self._dernier_appel = time.monotonic()
            raise
        trace.secondes = time.monotonic() - debut
        trace.instance = str(getattr(reponse, "provider_used", "") or "")
        # ⚠ An EMPTY answer is not an off-grid answer, and confusing them cost half an hour
        # of diagnosis on 2026-09-22: the bench said "off grid" where the
        # gateway had returned nothing at all. On Groq, it is the signature of the OUTPUT
        # tokens per minute limit — invisible outside the body of the 429s, and the gateway
        # does not report it (measured on 2026-09-08, never fixed). The status and the error
        # are therefore logged, even when the gateway says "success".
        statut = str(getattr(reponse, "status", "") or "")
        erreur = str(getattr(reponse, "error", "") or "")
        if not getattr(reponse, "agents", None):
            trace.erreur = (erreur or f"réponse VIDE (statut {statut})")[:200]
        trace.jetons_sortie = int(getattr(reponse, "tokens_out", 0) or 0) or _estimer(reponse)
        self.journal.appels.append(trace)
        self._dernier_appel = time.monotonic()

        # ⚠ The guard that matters: if an UNdeclared instance served, we say so and we
        # stop. The result would be unusable, and worse: we would not know it.
        if trace.instance and trace.instance not in self.instances:
            raise PasserelleRefusee(
                f"instance « {trace.instance} » served whereas the bench only admits "
                f"{list(self.instances)}. The result is discarded: a measurement obtained on an "
                f"undeclared gateway is not the measurement one thinks one is reading."
            )
        return reponse


def _estimer(reponse) -> int:
    """Lacking a counter returned by the gateway, a coarse estimate, and SAID to be one.

    Four characters per token: it is wrong in the detail and enough for a budget
    guard. What matters is not leaving the counter at zero — a budget that counts
    nothing stops nothing.
    """
    try:
        agents = getattr(reponse, "agents", None) or []
        return max(1, len(str(agents)) // 4)
    except Exception:  # noqa: BLE001
        return 1
