"""The event in force for this run, and what it did — ticket 100, lot 1.

A single event at a time, deliberately. Two overlapping events would make attribution
impossible — this is exactly what the protocol tries to establish, and the mechanism must not
itself manufacture the confusion it serves to clear up.

The registry counts THREE distinct things and never mixes them: the exposed, the spared, and
those who already were today under `cadence: jour`. The second is the INTERNAL WITNESS of the
run and it must be readable; the third is neither one nor the other, and confusing it with a
spared agent would make an event applied once pass for an event that misses three trips in four.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from llm.evenements import calendrier
from llm.evenements import relais as relais_module
from llm.evenements.declaration import Evenement, EvenementApplique
from llm.evenements.exposition import expose, lecteurs
from llm.evenements.injection import ligne_de_foyer, ligne_de_lecture
from settings import settings
from sim_clock import gama_timestamp, wall_clock

# 2026-09-28 — next to `evenements.jsonl`; read by `experiences/rejeu_ab.py`.
FICHIER_PREMIERS_SERVICES = "premiers_services.jsonl"


@dataclass
class CompteursJournee:
    jour_run: int = 0
    exposes: int = 0
    epargnes: int = 0
    retard_injecte_s: int = 0
    # Eligible arrivals NOT affected because the agent had already been that same day
    # (`cadence: jour`). Counted apart: this is neither an exposure nor a spared agent.
    deja_touches: int = 0
    # Decisions served FROM THE CACHE that day. Counted for a precise reason: the cache cut
    # applies to the event day, but what is measured is the window AFTER it. This counter
    # says, day by day, how many decisions the caveat concerns.
    decisions_depuis_cache: int = 0
    # Decisions for which the cache was deliberately bypassed (event day).
    decisions_cache_coupe: int = 0
    # ── Ticket 111 — guaranteed presence in the prompt ──────────────────────────────────
    lectures_servies: int = 0            # renderings of `[ PRESSE ]` to a reader
    messages_servis: int = 0             # renderings of `[ FOYER ]` to an informed member
    servies_avant_injection: int = 0     # of which decisions computed BEFORE the 00:00 injection
    relais_produits: int = 0
    relais_refuses: int = 0
    relais_sans_membre: int = 0          # reader alone (or only mobile one): nothing to pass on
    cache_contourne_ligne: int = 0       # decisions that had to carry a line: never from cache
    # Service-day decisions built WITHOUT their line. Must stay at zero.
    decisions_sans_ligne: int = 0
    reflexions_presse_presentees: int = 0
    reflexions_presse_validees: int = 0
    reflexions_presse_invalides: int = 0


class RegistreEvenements:
    """What the declared event did, day by day."""

    def __init__(self, evenement: Evenement, journal: Path | None = None) -> None:
        self.evenement = evenement
        self._journal = journal
        self._compteurs = CompteursJournee()
        self._exposes_vus: set[str] = set()
        # `cadence: jour` — who has already been affected WITHIN the current day. Reset at the
        # day switch, never accumulated over the run: an agent is affected once per event
        # day, not once for the whole run.
        self._touches_du_jour: set[str] = set()
        # ── `reveil` injection point (lot 2) ─────────────────────────────────────────────
        # The readers, drawn ONCE over the loaded population, and who has already read. An
        # article is read once: rereading it every morning would make the stimulus a repetition,
        # and the persistence of the memory could no longer be told from its cause's repetition.
        self._lecteurs: dict[str, tuple[str, str]] | None = None
        self._ont_lu: set[str] = set()
        # Rising edge of the "window open, readers never drawn" alarm.
        self._alarme_lecteurs_non_tires = False
        # ── Ticket 111 — the line served to the prompt, and the relay to the household ───
        self._journal_relais = (
            Path(journal).parent / "relais_foyer.jsonl" if journal is not None else None
        )
        self._journal_reflexions_presse = (
            Path(journal).parent / "reflexions_presse.jsonl" if journal is not None else None
        )
        # 2026-09-28 — the instant the event enters a prompt for the first time. The bound of
        # the witness's strict replay is read here, not in `evenements.jsonl`.
        self._journal_services = (
            Path(journal).parent / FICHIER_PREMIERS_SERVICES if journal is not None else None
        )
        # Relays already produced by THIS run: read back on resume, never regenerated.
        self._relais: dict[str, relais_module.RelaisFoyer] = relais_module.relire(
            self._journal_relais
        )
        # Technical failures to retry: taken out of `_relais` to be requested again, with the
        # number of the attempt already made.
        self._tentatives: dict[str, int] = {
            hh: r.tentative for hh, r in self._relais.items() if r.a_retenter
        }
        for hh in self._tentatives:
            del self._relais[hh]
        self._relais_taches: dict[str, asyncio.Future] = {}
        self._producteur_relais: Callable[[str, str, int], Awaitable[Any]] | None = None
        self._non_avenus: set[str] = set()
        self._non_avenus_alarmes: set[str] = set()
        self._servies: Counter = Counter()  # (person_id, ISO date) → renderings
        self._servies_avant_injection: Counter = Counter()  # person_id → renderings
        self._entendus: set[str] = set()  # informed members already written to memory at 00:00
        self._lecteurs_inconnus_dit = False
        self._producteur_absent_dit = False
        self._jamais_injectes_dits: set[str] = set()
        # On resume, who has ALREADY read and who has already been informed is read back from the
        # journal: otherwise, a replay passing through 00:00 of the reading day would make the
        # reading due a second time, and the replay freeze would declare it void — a successful
        # reading would lose its lines. The journal carries only completed injections (`tracer`
        # follows the memory write).
        self._relire_journal()
        # Simulated instant of the last /sync: this is what says a decision was computed
        # BEFORE the injection, and by how much.
        self._maintenant: int | None = None

    def _relire_journal(self) -> None:
        if self._journal is None or self.evenement.moment != "reveil":
            return
        chemin = Path(self._journal)
        if not chemin.is_file():
            return
        lus = entendus = 0
        for brute in chemin.read_text(encoding="utf-8").splitlines():
            try:
                ligne = json.loads(brute) if brute.strip() else None
            except json.JSONDecodeError:
                continue  # last line truncated by a hard stop: skipped, never fatal
            if not ligne or ligne.get("evenement_id") != self.evenement.evenement_id:
                continue
            pid = str(ligne.get("person_id") or "")
            if not pid:
                continue
            if ligne.get("origine") == "entendu":
                self._entendus.add(pid)
                entendus += 1
            else:
                self._ont_lu.add(pid)
                lus += 1
        if lus or entendus:
            logger.info(
                f"[evenements] reprise — « {self.evenement.evenement_id} » : {lus} lecture(s) et "
                f"{entendus} membre(s) informé(s) relus dans {chemin.name} ; ni relus ni "
                f"réinjectés."
            )

    def _injection_manquee(self, lecteur_id: str) -> bool:
        """The reading day has passed and this reader was never injected.

        Happens on resume, when the injection did not complete before the stop (the new
        registry does not know it was declared void): serving the line would make the agent
        read an article its memory does not carry. On the day itself, we serve: the decision
        may have been computed before the injection, and that is the whole point of the ticket.
        """
        if lecteur_id in self._ont_lu or self._maintenant is None:
            return False
        lecture = self._date_injection(lecteur_id)
        if lecture is None or self._date_de(self._maintenant) <= lecture:
            return False
        if lecteur_id not in self._jamais_injectes_dits:
            self._jamais_injectes_dits.add(lecteur_id)
            logger.error(
                f"[ALARME] [evenements] « {self.evenement.evenement_id} » : le lecteur "
                f"{lecteur_id} devait lire le {lecture:%d/%m} et n'a jamais été injecté (reprise "
                f"après une exposition non avenue ?). Ses lignes et celles de son foyer ne sont "
                f"PAS servies ; {self._servies_avant_injection.get(lecteur_id, 0)} décision(s) "
                f"les avaient déjà portées avant."
            )
        return True

    @property
    def choc(self) -> Evenement:
        """079 compatibility — goes in lot 6 with `llm/chocs.py`."""
        return self.evenement

    # ── Time ─────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def jour_du_run(timestamp: int) -> int:
        """1 for the first simulated day of the run.

        Stays a class method delegating to `calendrier.jour_du_run`: the 079 tests replace it
        on the CLASS to play a given day without a simulator, and moving the call into the
        module would take this hook away from them.
        """
        return calendrier.jour_du_run(timestamp)

    def jour_relatif(self, timestamp: int, person_id: str | None = None) -> int:
        """Days elapsed since the effective exposure: −2, −1, 0, +1…

        Defined on every day of the run, including before and long after: it is the x-axis of
        every drop-off and return curve, and an x-axis that existed only on event days would
        plot nothing. For a window drawn per household, day zero is that of the reader of
        ``person_id``'s household; using the first possible day of the window shifted the
        whole curve (A13: exposure on D12 shown as +3 because of window D9–D13).

        Without a person — old calls and endured events resolved trip by trip — the historical
        declarative origin is kept.
        """
        if person_id is not None:
            date_injection = self._date_injection(str(person_id))
            if date_injection is not None:
                return (self._date_de(timestamp) - date_injection).days
        return self.jour_du_run(timestamp) - self.evenement.premier_jour

    # ── Exposure ─────────────────────────────────────────────────────────────────────────
    def _expose(self, person_id: str, mode: str | None) -> tuple[bool, str]:
        return expose(
            self.evenement.exposition, self.evenement.evenement_id, person_id, mode
        )

    def applique(
        self, person_id: str, mode: str | None, timestamp: int
    ) -> EvenementApplique | None:
        """What this agent endures at this arrival, or `None` if it endures nothing.

        Returns `None` in three perfectly distinct cases, and counts them separately: the day
        is not an event day, the agent is not exposed, or the day's profile inflicts
        nothing.
        """
        jour = self.jour_du_run(timestamp)
        if jour != self._compteurs.jour_run:
            self._basculer_de_journee(jour)
        profil = self.evenement.jours.get(jour)
        if profil is None:
            return None
        est_expose, raison = self._expose(str(person_id), mode)
        if not est_expose:
            self._compteurs.epargnes += 1
            return None
        if self.evenement.cadence == "jour" and str(person_id) in self._touches_du_jour:
            self._compteurs.deja_touches += 1
            return None
        self._touches_du_jour.add(str(person_id))
        self._compteurs.exposes += 1
        self._compteurs.retard_injecte_s += profil.retard_s
        self._exposes_vus.add(str(person_id))
        return EvenementApplique(
            evenement_id=self.evenement.evenement_id,
            canal=self.evenement.canal,
            moment=self.evenement.moment,
            jour_run=jour,
            jour_relatif=jour - self.evenement.premier_jour,
            retard_injecte_s=profil.retard_s,
            texte=profil.texte,
            incident_reseau=profil.incident_reseau,
            correspondance_ratee=profil.correspondance_ratee,
            raison=raison,
        )

    # ── `reveil` injection point ─────────────────────────────────────────────────────────
    def lecteurs(self, population) -> dict[str, tuple[str, str]]:
        """Who receives this event, and for what reason. Computed once, then kept."""
        if self._lecteurs is not None:
            return self._lecteurs
        e = self.evenement
        if e.exposition.regle == "foyers":
            self._lecteurs = lecteurs(e.exposition, e.evenement_id, population)
        else:
            retenus: dict[str, tuple[str, str]] = {}
            for personne in population:
                if getattr(personne, "immobile", False):
                    continue
                touche, raison = expose(
                    e.exposition, e.evenement_id, str(personne.person_id), None
                )
                if touche:
                    retenus[str(personne.person_id)] = (
                        str(getattr(personne, "household_id", "") or ""), raison,
                    )
            self._lecteurs = retenus
        logger.info(
            f"[evenements] « {e.evenement_id} » : {len(self._lecteurs)} lecteur(s) retenu(s) "
            f"sur {sum(1 for _ in population)} agent(s) — règle {e.exposition.regle}"
        )
        return self._lecteurs

    def role_de(self, person_id: str) -> tuple[str, str]:
        """This agent's role in the setup, and the reason. Ticket 100, lot 5.

        Three roles, and they cannot be deduced from one another:

        - `expose` — the event reached them;
        - `co_resident` — lives under the same roof as an exposed agent, and received NOTHING.
          What is transmitted is read in them, and without them the diffusion stage is moot;
        - `temoin` — neither. This is the baseline of the run.

        Empty when the setup does not yet know who reads: an empty column is not a role, and
        writing it `temoin` would pass ignorance off as a measurement.
        """
        if self._lecteurs is None:
            # `arrivee` injection point: exposure is resolved trip by trip, never in advance.
            # The role is then read in `evenements.jsonl`, not here.
            return ("", "")
        pid = str(person_id)
        if pid in self._lecteurs:
            return ("expose", self._lecteurs[pid][1])
        foyers_exposes = {h for h, _ in self._lecteurs.values() if h}
        from llm import foyer as _foyer

        mien = _foyer.foyer_de(pid)
        if mien and mien in foyers_exposes:
            return ("co_resident", f"foyer:{mien}")
        return ("temoin", "")

    def jour_de(self, person_id: str, household_id: str) -> int:
        """The day on which THIS agent receives the event.

        The draw target is the HOUSEHOLD when the rule is `foyers`: the members of one
        household must receive on the same day, otherwise the witness co-resident would no
        longer be comparable with the reader on the same day.
        """
        e = self.evenement
        if e.calendrier is None:
            return e.premier_jour
        cible = household_id if e.exposition.regle == "foyers" and household_id else person_id
        return e.calendrier.jour_de(e.evenement_id, cible)

    def dus_au_reveil(self, timestamp: int, population) -> list:
        """The agents who receive the event this morning, and what they receive.

        Returns a list of `(person_id, EvenementApplique)`, empty on other days. Idempotent per
        agent: called three times in the same day, it returns nothing the second time.
        """
        e = self.evenement
        if e.moment != "reveil":
            return []
        jour = self.jour_du_run(timestamp)
        if jour != self._compteurs.jour_run:
            self._basculer_de_journee(jour)
        dus = []
        for person_id, (household_id, raison) in self.lecteurs(population).items():
            if person_id in self._ont_lu:
                continue
            if self.jour_de(person_id, household_id) != jour:
                continue
            self._ont_lu.add(person_id)
            self._compteurs.exposes += 1
            dus.append((
                person_id,
                EvenementApplique(
                    evenement_id=e.evenement_id,
                    canal=e.canal,
                    moment=e.moment,
                    jour_run=jour,
                    jour_relatif=0,  # the day of receipt IS the origin, per agent
                    retard_injecte_s=0,
                    texte=e.texte_cite.servi if e.texte_cite else "",
                    incident_reseau=False,
                    correspondance_ratee=False,
                    raison=raison,
                ),
            ))
        if dus:
            restants = len(self.lecteurs(population)) - len(self._ont_lu)
            logger.info(
                f"[evenements] jour {jour} — « {e.evenement_id} » servi à {len(dus)} "
                f"lecteur(s) au réveil ; {restants} lecteur(s) attendent encore leur jour de "
                f"parution. Les co-résidents non tirés sont le TÉMOIN INTERNE : ils ne "
                f"reçoivent rien, et c'est ce qui rend l'étage de diffusion observable."
            )
        return dus

    # ── Ticket 111 — what is served to the prompt during the service days ────────────────
    def noter_instant(self, timestamp: int) -> None:
        """The current simulated instant, set at every /sync. Used by the journal, never computing."""
        self._maintenant = int(timestamp)

    def brancher_producteur_relais(
        self, producteur: Callable[[str, str, int], Awaitable[Any]]
    ) -> None:
        """The relay producer: only the controller holds population, client and identities."""
        self._producteur_relais = producteur

    def _jours_de_service(self, cible_id: str, household_id: str) -> tuple:
        """This target's service dates, deterministic: neither horizon nor recall enter them."""
        service = self.evenement.service
        if service is None:
            return ()
        debut = calendrier.date_du_jour_run(self.jour_de(cible_id, household_id))
        if debut is None:
            return ()
        return calendrier.jours_de_service(
            debut,
            service.jours_de_deplacement,
            bool(getattr(settings.agent, "no_weekend_departures", False)),
        )

    def _lecteur_du_foyer(self, household_id: str) -> str | None:
        """The reader who relays in this household. The first in numeric order if there are several."""
        ids = [p for p, (h, _) in (self._lecteurs or {}).items() if h == household_id]
        if not ids:
            return None
        return sorted(ids, key=lambda p: (len(p), p))[0]

    def foyer_expose(self, person_id: str) -> tuple[str, str] | None:
        """`(household_id, lecteur_id)` if this agent is a NON-reader of an exposed household."""
        if self._lecteurs is None:
            return None
        pid = str(person_id)
        if pid in self._lecteurs:
            return None
        from llm import foyer as _foyer

        mien = _foyer.foyer_de(pid)
        if not mien:
            return None
        lecteur = self._lecteur_du_foyer(mien)
        return (mien, lecteur) if lecteur else None

    def relais_connu(self, household_id: str):
        """This household's relay if already produced (or refused), without launching anything."""
        return self._relais.get(str(household_id))

    async def relais_du_foyer(self, household_id: str):
        """This household's relay, produced on first request. A single call per household.

        Returns `None` when there is no relay to serve: no relay declared, no producer
        plugged in, or relay refused (the alarm was raised at production).
        """
        hh = str(household_id)
        if self.evenement.relais is None:
            return None
        deja = self._relais.get(hh)
        if deja is not None:
            return None if deja.refus else deja
        if self._producteur_relais is None:
            if not self._producteur_absent_dit:
                self._producteur_absent_dit = True
                logger.error(
                    f"[ALARME] [evenements] « {self.evenement.evenement_id} » declares a relay "
                    f"to the household, but no producer is plugged in: NO member of an exposed "
                    f"household will be informed. Do not count the co-residents as informed."
                )
            return None
        tache = self._relais_taches.get(hh)
        if tache is None:
            tache = asyncio.ensure_future(self._produire_relais(hh))
            self._relais_taches[hh] = tache
        return await asyncio.shield(tache)

    async def _produire_relais(self, hh: str):
        lecteur = self._lecteur_du_foyer(hh)
        jour = self.jour_de(lecteur or "", hh)
        tentative = self._tentatives.get(hh, 0) + 1
        if tentative > 1:
            logger.warning(
                f"[evenements] relay of household {hh} — attempt {tentative}/"
                f"{relais_module.TENTATIVES_MAX} after a technical failure in the previous run"
            )
        try:
            produit = await self._producteur_relais(lecteur, hh, jour)
        except relais_module.RelaisRefuse as err:
            produit = relais_module.RelaisFoyer(
                household_id=hh, lecteur_id=str(lecteur), lecteur_prenom="",
                evenement_id=self.evenement.evenement_id, jour_run=int(jour),
                refus=str(err) or "refusé", technique=err.technique, tentative=tentative,
            )
        except Exception as err:  # noqa: BLE001 — the relay never brings down a decision
            logger.error(
                f"[ALARME] [evenements] relay of household {hh} impossible ({type(err).__name__}: "
                f"{err}) — reader {lecteur}, day {jour}. No message is served in this "
                f"household."
            )
            produit = relais_module.RelaisFoyer(
                household_id=hh, lecteur_id=str(lecteur), lecteur_prenom="",
                evenement_id=self.evenement.evenement_id, jour_run=int(jour),
                refus=f"{type(err).__name__}: {err}", technique=True, tentative=tentative,
            )
        if produit.refus and produit.technique:
            reste = relais_module.TENTATIVES_MAX - produit.tentative
            logger.error(
                f"[ALARME] [evenements] relais du foyer {hh} : échec technique, tentative "
                f"{produit.tentative}/{relais_module.TENTATIVES_MAX}. "
                + (f"Pas de nouvel essai dans ce run ; une reprise retentera ({reste} essai(s) "
                   f"restant(s))." if reste > 0 else
                   "Plus aucun essai : le refus est définitif pour ce run et ses reprises.")
            )
        self._relais[hh] = produit
        if not produit.refus and tentative > 1:
            logger.warning(
                f"[evenements] relais du foyer {hh} produit à la tentative {tentative} : ses "
                f"membres informés voient leur ligne pour les jours de service restants. Si le "
                f"00:00 du jour de lecture est passé, ils n'ont PAS l'entrée en mémoire de ce "
                f"jour-là (coût accepté par l'auteur le 2026-09-25)."
            )
        if produit.refus:
            self._compteurs.relais_refuses += 1
        elif not produit.messages:
            self._compteurs.relais_sans_membre += 1
        else:
            self._compteurs.relais_produits += 1
        relais_module.ecrire(self._journal_relais, produit)
        return None if produit.refus else produit

    def _date_de(self, timestamp: int) -> date:
        return wall_clock(int(timestamp)).date()

    async def lignes_du_jour(
        self, person_id: str, timestamp: int, *, compter: bool = True
    ) -> list[str]:
        """The lines GUARANTEED in this agent's prompt for a decision held at `timestamp`.

        `timestamp` is the instant of the TRIP, not of the computation: a reading-day decision
        is computed the day before, and that is precisely what deprived it of the article.

        - reader, day within their service days → `[ PRESSE ] I read in the paper…`;
        - informed member of an exposed household, day within their service days → their
          `[ FOYER ]` line, taken from the relay (produced on demand if not yet produced);
        - person declared void, or not informed → nothing.

        `compter=False` for a survey: it sees the same line, but is not a decision.
        """
        e = self.evenement
        if e.service is None or e.moment != "reveil":
            return []
        if self._lecteurs is None:
            if not self._lecteurs_inconnus_dit:
                self._lecteurs_inconnus_dit = True
                logger.error(
                    f"[ALARME] [evenements] « {e.evenement_id} »: a decision asks for its "
                    f"service lines while the readers are not yet drawn. The "
                    f"controller must call `lecteurs(population)` as soon as the population "
                    f"is loaded; otherwise, a reading on day 1 would be invisible to the "
                    f"bootstrap decisions."
                )
            return []
        pid = str(person_id)
        if pid in self._non_avenus:
            return []
        jour = self._date_de(timestamp)
        ligne = None
        avant = False
        if pid in self._lecteurs:
            hh = self._lecteurs[pid][0]
            if jour not in self._jours_de_service(pid, hh):
                return []
            if self._injection_manquee(pid):
                return []
            ligne = ligne_de_lecture(e.texte_cite.servi if e.texte_cite else "")
            avant = pid not in self._ont_lu
            genre = "lectures_servies"
        else:
            expose_ = self.foyer_expose(pid)
            if expose_ is None or e.relais is None:
                return []
            hh, lecteur = expose_
            if jour not in self._jours_de_service(pid, hh):
                return []
            if lecteur is not None and self._injection_manquee(str(lecteur)):
                return []
            produit = await self.relais_du_foyer(hh)
            if pid in self._non_avenus or produit is None:
                return []
            message = produit.message_pour(pid)
            if message is None or not message.parle:
                return []
            ligne = ligne_de_foyer(message.texte, produit.lecteur_prenom, message.mineur)
            avant = pid not in self._entendus
            genre = "messages_servis"
        if compter:
            self._noter_service(pid, jour, timestamp, avant, genre)
        return [ligne]

    def _noter_service(self, pid: str, jour: date, timestamp: int, avant: bool,
                       genre: str) -> None:
        cle = (pid, jour.isoformat())
        premier = cle not in self._servies
        self._servies[cle] += 1
        setattr(self._compteurs, genre, getattr(self._compteurs, genre) + 1)
        if avant:
            self._servies_avant_injection[pid] += 1
            self._compteurs.servies_avant_injection += 1
        if not premier:
            return
        self._consigner_premier_service(pid, jour, timestamp, genre, avant)
        quoi = "article du jour" if genre == "lectures_servies" else "message du foyer"
        calcul = ""
        if avant and self._maintenant is not None:
            injection = gama_timestamp(
                datetime.combine(self._date_injection(pid) or jour, datetime.min.time())
            )
            ecart_h = (injection - self._maintenant) / 3600.0
            if ecart_h > 0:
                calcul = f", calculée {ecart_h:.0f} h avant l'injection"
        logger.info(
            f"[evenements] {pid} : {quoi} servi à la décision du "
            f"{wall_clock(int(timestamp)):%d/%m %H:%M}{calcul} "
            f"(jour de service {jour.isoformat()})"
        )

    def _consigner_premier_service(self, pid: str, jour: date, timestamp: int, genre: str,
                                   avant: bool) -> None:
        """One line per (person, service day): the simulated instant at which the event line
        first entered a prompt (2026-09-28).

        That is where the two arms of an A/B stop asking the same questions, and not at the
        injection timestamp. A next-day decision is computed from the day before: in
        a13 v5, the article of 27/03 entered on 26/03 at 07:30 the decision for the 07:10 trip
        of the next day, 16 h 30 before the 00:00 injection. The bound of the witness's strict
        replay is read here (`experiences/rejeu_ab.instant_debut_traitement`). Never an
        exception towards the caller.
        """
        if self._journal_services is None:
            return
        depuis_sync = self._maintenant is not None
        instant = self._maintenant if depuis_sync else int(timestamp)
        if not depuis_sync:
            # Without /sync, the decision instant is later than the prompt's: a bound read
            # here would arrive too late. The controller sets the instant at every /sync.
            logger.warning(
                f"[evenements] {pid}: first service recorded without a /sync instant — "
                f"the decision instant ({wall_clock(instant):%d/%m %H:%M}) replaces it."
            )
        try:
            ligne = {
                "person_id": str(pid),
                "evenement_id": self.evenement.evenement_id,
                "jour_service": jour.isoformat(),
                "genre": genre,
                "instant_ts": instant,
                "instant_simule": wall_clock(instant).isoformat(),
                "instant_source": "sync" if depuis_sync else "decision",
                "decision_ts": int(timestamp),
                "avant_injection": bool(avant),
            }
            self._journal_services.parent.mkdir(parents=True, exist_ok=True)
            with self._journal_services.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        except Exception as err:  # noqa: BLE001 — a trace never brings down a run
            logger.warning(f"[evenements] first service not recorded ({err})")

    def _date_injection(self, pid: str) -> date | None:
        hh = (self._lecteurs or {}).get(pid, (None,))[0] or (self.foyer_expose(pid) or (None,))[0]
        if hh is None:
            return None
        return calendrier.date_du_jour_run(self.jour_de(pid, hh))

    def noter_entendu(self, person_id: str) -> None:
        """This informed member's message is written to memory (00:00 injection)."""
        self._entendus.add(str(person_id))

    def declarer_non_avenue(self, person_id: str, motif: str) -> None:
        """This agent's exposure did not take place. Its lines stop being served.

        If decisions have ALREADY carried the line — computed the day before, before the
        injection was refused —, they are not undone: an `[ALARME]`, once per agent,
        says how many, so that they are excluded from the analysis.
        """
        pid = str(person_id)
        self._non_avenus.add(pid)
        deja = sum(n for (p, _), n in self._servies.items() if p == pid)
        if deja and pid not in self._non_avenus_alarmes:
            self._non_avenus_alarmes.add(pid)
            logger.error(
                f"[ALARME] [evenements] « {self.evenement.evenement_id} » : l'exposition de "
                f"{pid} est déclarée NON AVENUE ({motif}), mais {deja} décision(s) ont déjà vu "
                f"sa ligne au prompt. Elles ne se défont pas : les écarter de l'analyse."
            )
        else:
            logger.warning(
                f"[evenements] « {self.evenement.evenement_id} »: exposure of {pid} declared "
                f"void ({motif}) — no decision had seen its line yet."
            )

    def noter_rendu(self, person_id: str, timestamp: int, lignes, historique) -> None:
        """Did a service-day decision actually carry its lines? Never raises."""
        try:
            if not lignes:
                return
            rendu = "\n".join(str(h) for h in (historique or []))
            manquantes = [ligne for ligne in lignes if ligne not in rendu]
            if not manquantes:
                return
            self._compteurs.decisions_sans_ligne += 1
            logger.error(
                f"[ALARME] [evenements] decision of {person_id} on "
                f"{wall_clock(int(timestamp)):%d/%m %H:%M} built WITHOUT its service "
                f"line ({manquantes[0][:40]}…) — the memory block did not render it. The "
                f"guarantee of ticket 111 does not hold for this decision."
            )
        except Exception:  # noqa: BLE001
            pass

    def noter_contournement_cache(self) -> None:
        self._compteurs.cache_contourne_ligne += 1

    def tracer_reflexion_presse(
        self,
        person_id: str,
        timestamp: int,
        lignes: list[str],
        *,
        etape: str,
        prise_en_compte: bool | None = None,
        reflection: str = "",
    ) -> None:
        """Traces the presentation of the journal to the consolidation and its read receipt.

        The trace is deliberately separate from the injection journal: an initial reading
        and a daily presentation during service are two different mechanisms.
        """
        if not lignes:
            return
        if etape == "presentee":
            self._compteurs.reflexions_presse_presentees += 1
        elif prise_en_compte:
            self._compteurs.reflexions_presse_validees += 1
        else:
            self._compteurs.reflexions_presse_invalides += 1
        chemin = self._journal_reflexions_presse
        if chemin is None:
            return
        try:
            texte = "\n".join(str(l) for l in lignes)
            ligne = {
                "evenement_id": self.evenement.evenement_id,
                "person_id": str(person_id),
                "timestamp": int(timestamp),
                "date": self._date_de(int(timestamp)).isoformat(),
                "jour_relatif": self.jour_relatif(int(timestamp), str(person_id)),
                "etape": str(etape),
                "prise_en_compte": prise_en_compte,
                "service_sha256": hashlib.sha256(texte.encode("utf-8")).hexdigest(),
                "service_apercu": texte[:240],
                "reflection": str(reflection or ""),
            }
            chemin.parent.mkdir(parents=True, exist_ok=True)
            with chemin.open("a", encoding="utf-8") as flux:
                flux.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        except Exception as err:  # noqa: BLE001 — observability does not bring down the run
            logger.error(
                f"[ALARME] [evenements] press reflection trace impossible for "
                f"{person_id} ({err})"
            )

    # ── Decision cache ───────────────────────────────────────────────────────────────────
    def cache_coupe(self, timestamp: int) -> bool:
        """Must the decision cache be bypassed at this instant? (Q5, 2026-09-22)

        True on EVENT DAYS, and only those. The cut is automatic: it does not depend on a
        `CACHE=0` one would remember to set, and the 19 September run showed what a guard
        costs that depends on the operator's memory.

        ⚠ **What this cut does not protect.** The event day is not the day being measured:
        the window after it is. The exact cache key carries neither the date nor the agent's
        memory — a decision taken on day 5, same options, same weather, same traits, same
        time slot, stays servable on day 20. What protects the window is the SEMANTIC branch
        of the cache, which requires 0.95 similarity between today's memory block and the one
        that had produced the decision. Fifteen days apart move this block a lot — probably
        enough. Probably: nobody has measured it. Hence the two counters, which make the
        caveat readable instead of leaving it to be argued about.
        """
        jour = self.jour_du_run(timestamp)
        if self.evenement.jours:
            return jour in self.evenement.jours
        # Form B: publication days are drawn per target. The cut covers the whole WINDOW,
        # for lack of knowing, at this instant and without the population, who is concerned.
        # Cutting wide is better than cutting beside: the cost is a few days of calls, the
        # opposite risk is a decision served again on the very morning of publication.
        return self.evenement.premier_jour <= jour <= self.evenement.dernier_jour

    def noter_decision(self, timestamp: int, depuis_cache: bool) -> None:
        """A decision has just been taken; say whether the cache served it. Never raises."""
        try:
            jour = self.jour_du_run(timestamp)
            if jour != self._compteurs.jour_run:
                self._basculer_de_journee(jour)
            if depuis_cache:
                self._compteurs.decisions_depuis_cache += 1
            else:
                self._compteurs.decisions_cache_coupe += 1
        except Exception:  # noqa: BLE001 — a counter never brings down a decision
            pass

    # ── Journal ──────────────────────────────────────────────────────────────────────────
    def _basculer_de_journee(self, jour: int) -> None:
        if self._compteurs.jour_run:
            self.journaliser_compteurs()
        self._compteurs = CompteursJournee(jour_run=jour)
        self._touches_du_jour.clear()

    def journaliser_compteurs(self) -> None:
        """Counters of the day just ended, logged EVEN AT ZERO.

        A silent counter cannot tell "nothing happened" from "the mechanism is not running",
        and that is precisely the confusion that cost ticket 075 thirty days.
        """
        c = self._compteurs
        if not c.jour_run:
            return
        actif = (
            c.jour_run in self.evenement.jours
            if self.evenement.jours
            else self.evenement.premier_jour <= c.jour_run <= self.evenement.dernier_jour
        )
        # Form B — a window, and a publication day DRAWN per household. A day of the window
        # is not an event day: only the one on which a reader was drawn is. The run
        # `2026-09-24_17_50` (window 9-13, a single household drawn on day 11) raised the
        # « 0 exposé » alarm on days 9, 10, 12 and 13, when there was nothing to expose.
        tire_par_foyer = actif and not self.evenement.jours
        attendus = self._lecteurs_tires_pour(c.jour_run) if tire_par_foyer else None
        if not tire_par_foyer:
            etat = "JOUR D_EVENEMENT" if actif else "nominal"
        elif attendus is None:
            etat = "fenêtre de parution, lecteurs PAS ENCORE TIRÉS"
        elif attendus:
            etat = f"JOUR DE PARUTION ({len(attendus)} lecteur(s) tiré(s) pour ce jour)"
        else:
            etat = "fenêtre de parution, aucun lecteur tiré pour ce jour"
        deja = f", {c.deja_touches} déjà touché(s) ce jour" if c.deja_touches else ""
        relatif = c.jour_run - self.evenement.premier_jour
        repere = "à l'ouverture de la fenêtre" if not self.evenement.jours else "au 1er jour"
        logger.info(
            f"[evenements] jour {c.jour_run} du run "
            f"(relatif {relatif:+d} {repere}) — "
            f"{etat} : {c.exposes} exposé(s), "
            f"{c.epargnes} épargné(s){deja}, {c.retard_injecte_s // 60} min de retard injecté au "
            f"total (canal « {self.evenement.canal} », cadence « {self.evenement.cadence} »)"
        )
        # The cache, day by day. Logged EVEN AT ZERO over the whole window after: it is the
        # only figure that says whether the caveat of the § "cut on the event day" has an
        # object, and it is useless if it is only recorded when there is a problem.
        if actif:
            motif_coupure = "fenêtre de parution" if tire_par_foyer else "jour d'événement"
            logger.info(
                f"[evenements] jour {c.jour_run} — cache de décisions COUPÉ ({motif_coupure}) : "
                f"{c.decisions_cache_coupe} décision(s) passée(s) par le modèle"
            )
        elif relatif > 0:
            logger.info(
                f"[evenements] jour {c.jour_run} (relatif {relatif:+d}, fenêtre d'après) — "
                f"{c.decisions_depuis_cache} décision(s) servie(s) DEPUIS LE CACHE sur "
                f"{c.decisions_depuis_cache + c.decisions_cache_coupe}. Un compte non nul ne "
                f"prouve rien à lui seul : la branche sémantique exige 0,95 de similarité de "
                f"mémoire. Il dit sur quoi porte le doute avant de publier la courbe."
            )
        if self.evenement.service is not None:
            # Ticket 111 — logged EVEN AT ZERO: a silent counter cannot tell "nothing to
            # serve today" from "the mechanism no longer runs".
            logger.info(
                f"[evenements] jour {c.jour_run} — service garanti : {c.lectures_servies} "
                f"lecture(s) servie(s), {c.messages_servis} message(s) du foyer servi(s), dont "
                f"{c.servies_avant_injection} dans une décision calculée avant l'injection ; "
                f"relais {c.relais_produits} produit(s), {c.relais_refuses} refusé(s), "
                f"{c.relais_sans_membre} sans autre membre ; "
                f"{c.cache_contourne_ligne} décision(s) tenue(s) hors cache pour porter leur "
                f"ligne ; {c.decisions_sans_ligne} décision(s) de jour de service sans ligne"
            )
            logger.info(
                f"[evenements] jour {c.jour_run} — journal dans la réflexion : "
                f"{c.reflexions_presse_presentees} présentation(s), "
                f"{c.reflexions_presse_validees} prise(s) en compte validée(s), "
                f"{c.reflexions_presse_invalides} réponse(s) invalide(s)"
            )
            if c.decisions_sans_ligne:
                logger.error(
                    f"[ALARME] [evenements] jour {c.jour_run} : {c.decisions_sans_ligne} "
                    f"décision(s) d'un jour de service construite(s) SANS leur ligne. La "
                    f"présence garantie au prompt ne tient pas ce jour-là."
                )
        if tire_par_foyer:
            self._alarmer_parution(c.jour_run, attendus)
        elif actif and c.exposes == 0:
            # An event day that affects nobody is a protocol that did not take place. The
            # 19 September run had one — a second shock restricted to car trips, on an agent
            # who no longer drove — reported at INFO, hence read by nobody, and the report
            # went on announcing two shock days.
            logger.error(
                f"[ALARME] [evenements] jour {c.jour_run} du run déclaré JOUR D'ÉVÉNEMENT et "
                f"clos avec 0 exposé sur {c.epargnes} arrivée(s) éligible(s) examinée(s) : "
                f"l'événement « {self.evenement.evenement_id} » n'a PAS eu lieu ce jour-là. "
                f"Ne pas le compter comme une journée d'événement dans l'analyse."
            )

    def _lecteurs_tires_pour(self, jour: int) -> list[str] | None:
        """The readers whose drawn publication day is `jour`. `None` if they are not drawn.

        `None` and `[]` are not the same: the first says the waking injection point never ran,
        the second that no household drew that day — which is expected.
        """
        if self._lecteurs is None:
            return None
        return sorted(
            pid for pid, (household_id, _raison) in self._lecteurs.items()
            if self.jour_de(pid, household_id) == jour
        )

    def _alarmer_parution(self, jour: int, attendus: list[str] | None) -> None:
        """The alarm of a window drawn per household, against the readers drawn for THIS day."""
        e = self.evenement
        if attendus is None:
            # Rising edge: once per run. Every window day without drawn readers is the same
            # defect, and repeating it five times would drown the first occurrence.
            if not self._alarme_lecteurs_non_tires:
                self._alarme_lecteurs_non_tires = True
                logger.error(
                    f"[ALARME] [evenements] jour {jour} du run, dans la fenêtre de parution "
                    f"{e.premier_jour}-{e.dernier_jour} de « {e.evenement_id} », et les lecteurs "
                    f"n'ont jamais été tirés : la prise « {e.moment} » n'a pas tourné. Personne "
                    f"ne lira l'article tant qu'elle ne tourne pas."
                )
            return
        manquants = [pid for pid in attendus if pid not in self._ont_lu]
        if manquants:
            foyers = sorted({(self._lecteurs or {}).get(p, ("",))[0] for p in manquants})
            logger.error(
                f"[ALARME] [evenements] jour {jour} du run, jour de parution tiré pour "
                f"{len(attendus)} lecteur(s) de « {e.evenement_id} » : {len(manquants)} n'ont "
                f"PAS lu — {', '.join(manquants[:10])} (foyer(s) {', '.join(foyers[:10])}). Ne "
                f"pas compter ce jour comme une exposition de ces foyers dans l'analyse."
            )

    def tracer(self, applique: EvenementApplique, person_id: str, timestamp: int,
               gravite: float, detail: Any, jugement: Any = None,
               extra: dict | None = None) -> None:
        """One line per application in `evenements.jsonl`. Never an exception towards the caller.

        ⚠ **Ticket 079 fields are KEPT, new ones are ADDED.** `choc_id` and `vecu` are still
        written next to `evenement_id` and `texte`, for one version. This is the condition of
        the golden test — an `evenements.jsonl` equal field by field to the old
        `chocs.jsonl` — and it is what lets the analysers of already archived runs read the
        new runs without being touched. The aliases go in lot 6, with `llm/chocs.py`.
        """
        if not self._journal:
            return
        try:
            ligne = {
                "person_id": str(person_id),
                "timestamp": int(timestamp),
                "horodatage_simule": wall_clock(int(timestamp)).isoformat(),
                "choc_id": applique.evenement_id,
                "jour_run": applique.jour_run,
                "jour_relatif": applique.jour_relatif,
                "raison_exposition": applique.raison,
                "retard_injecte_s": applique.retard_injecte_s,
                "incident_reseau": applique.incident_reseau,
                "correspondance_ratee": applique.correspondance_ratee,
                "vecu": applique.texte,
                "gravite": round(float(gravite), 4),
                "gravite_detail": {
                    "retard": round(float(getattr(detail, "retard", 0.0)), 4),
                    "correspondance_ratee": round(
                        float(getattr(detail, "correspondance_ratee", 0.0)), 4
                    ),
                    "incident_reseau": round(
                        float(getattr(detail, "incident_reseau", 0.0)), 4
                    ),
                    "mode_contraint": round(float(getattr(detail, "mode_contraint", 0.0)), 4),
                },
                # ── Ticket 100 additions ────────────────────────────────────────────────
                "evenement_id": applique.evenement_id,
                "canal": applique.canal,
                "moment": applique.moment,
                "texte": applique.texte,
                # Ticket 100, lot 3 — what the AGENT said about it, next to what the simulation
                # measured. Both, never one instead of the other: without both columns, one
                # could no longer tell whether a severity comes from a fact or from an opinion.
                # Empty — never zero — when no judgement was requested: zero is the value of a
                # perfect trip, and the absence of measurement must not carry it.
                "importance_estimee": (
                    round(float(jugement.importance_estimee), 4) if jugement else None
                ),
                "intensite_jugee": jugement.intensite if jugement else None,
                "valence": jugement.valence if jugement else None,
                "modes_touches": list(jugement.modes) if jugement else None,
                "importance_retenue": round(float(gravite), 4),
                # D7 — the severity is the estimate alone. The gap to the measured fact is
                # logged and NEVER applied: it is the guard that replaces the floor. Without
                # this column, a campaign where the model systematically underestimates would
                # look feature for feature like a campaign where nothing happened.
                "ecart_au_fait": (
                    round(float(jugement.ecart_au_fait), 4) if jugement else None
                ),
            }
            # Ticket 111 — an informed member also carries the message received, who said it, and
            # whether they are a minor. Added, never substituted: the reader's line is unchanged.
            if extra:
                ligne.update(extra)
            self._journal.parent.mkdir(parents=True, exist_ok=True)
            with self._journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        except Exception as err:  # noqa: BLE001 — a trace never brings down a run
            logger.warning(f"[evenements] trace not written ({err})")
