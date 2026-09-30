"""Accidents drawn at random on the roads — their existence, not yet their consequence.

Ticket 070, first slice. This module draws accidents, places them on an edge of the road
graph and logs them. **It modifies no itinerary duration.** The delay suffered, the
cache guards and the agent's memory come in a later slice.

THIS SPLIT IS NOT AN EASY WAY OUT. As long as no duration changes, no disrupted itinerary
can enter the itinerary cache — which is addressed WITHOUT THE DATE and would serve
an accident duration to every Tuesday 8 a.m. — and no agent decision can be wrongly served
again by the decision cache, whose key is built on the option codes and ignores
durations. Both traps are neutralised by construction, not by vigilance.

WHAT THE DRAWING LAW IS WORTH TODAY. It is MEASURED on BAAC/ONISR 2019-2024 (3,789
injury accidents of département 31, 3,414 within the graph footprint) and conditions three
variables: the NUMBER by the day of the week, the HOUR by the observed hourly distribution, the
ROAD CLASS by the distribution by speed limit — the edge then being drawn in proportion
to its length WITHIN its class. The coefficients live in `config/accidents_baac.yaml`, which
also carries the reason for each choice.

⚠ A SINGLE VARIABLE REMAINS UNESTABLISHED: the weather factor. Its estimate was made and
REJECTED (the `atm` nomenclature and that of the local weather source do not split the same
world), so it equals 1. The coefficient file keeps the rejected computation so that the refusal
can be checked.

⚠ A ZERO COUNT IS NOT A FAILURE, and the reverse is true too: at 1.56 accidents per
day over the simulated footprint, many simulated days will see none. That is the
correct behaviour. That is why the log always says how many were drawn, including
zero — without this counter, "no accident" and "the draw is not running" look too alike.
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yaml
from loguru import logger
from settings import settings
from sim_clock import wall_clock

# Coefficients estimated on BAAC/ONISR. Measured, replayable, and documented in the file
# itself: this module only applies them.
_LOI_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "accidents_baac.yaml"

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# An accident is placed on a directed edge of the `drive` graph, designated by its two OSM
# nodes. The key (u, v) is enough: the parallel edges of the same pair share the lane.
CleArete = tuple[int, int]

# Beyond this, we refuse to draw: a world state that swells without bound is a bug, not a
# scenario. The threshold is far above any plausible draw (1.56/day measured).
MAX_ACCIDENTS_PAR_JOUR = 200


# Speed-limit classes, bounds identical to those of the BAAC measurement. Each edge
# of the graph carries a speed (OSM `maxspeed`, otherwise osmnx's table): no mapping
# table to invent between the two worlds.
CLASSES_VITESSE = [("<=30", 0, 30), ("31-50", 31, 50), ("51-70", 51, 70),
                   ("71-90", 71, 90), (">90", 91, 999)]


def classe_de_vitesse(vitesse) -> str | None:
    """Class of a speed limit, or `None` if it is not usable.

    Tolerates the forms OSM really uses: list of values on a merged edge,
    string "50" or "50 km/h", integer, absence.
    """
    if isinstance(vitesse, list):
        vitesse = vitesse[0] if vitesse else None
    if isinstance(vitesse, str):
        chiffres = "".join(ch for ch in vitesse if ch.isdigit())
        vitesse = chiffres or None
    if vitesse is None:
        return None
    try:
        v = int(float(vitesse))
    except (TypeError, ValueError):
        return None
    if v <= 0 or v > 200:
        return None
    for nom, bas, haut in CLASSES_VITESSE:
        if bas <= v <= haut:
            return nom
    return None


@dataclass(frozen=True)
class LoiBaac:
    """The measured accident law: what is drawn, and with which weights.

    Three variables condition the draw, and a fourth is declared unestablished.
    The configuration file carries the detail of each choice, including the reasoned rejection
    of the weather factor — read it before touching these numbers.
    """

    taux_base_par_jour: float
    distribution_horaire: dict[int, float]
    facteur_jour_semaine: dict[str, float]
    distribution_classe_vitesse: dict[str, float]
    facteur_meteo: dict[str, float]
    facteur_meteo_etabli: bool
    millesimes: list

    @classmethod
    def charger(cls, chemin: pathlib.Path | None = None) -> LoiBaac:
        chemin = chemin or _LOI_PATH
        brut = yaml.safe_load(chemin.read_text(encoding="utf-8"))
        loi = cls(
            taux_base_par_jour=float(brut["taux_base_par_jour"]),
            distribution_horaire={int(h): float(v) for h, v in brut["distribution_horaire"].items()},
            facteur_jour_semaine={str(j): float(v) for j, v in brut["facteur_jour_semaine"].items()},
            distribution_classe_vitesse={
                str(c): float(v) for c, v in brut["distribution_classe_vitesse"].items()
            },
            facteur_meteo={str(m): float(v) for m, v in (brut.get("facteur_meteo") or {}).items()},
            facteur_meteo_etabli=bool(brut.get("facteur_meteo_etabli", False)),
            millesimes=list(brut.get("millesimes") or []),
        )
        loi.verifier()
        return loi

    def verifier(self) -> None:
        """Refuse an inconsistent law rather than draw with it.

        A distribution that does not sum to 1 or a factor whose mean differs from 1
        would shift the mean rate without anything saying so: the whole run would be wrong and
        silent.
        """
        for nom, dist in (
            ("distribution_horaire", self.distribution_horaire),
            ("distribution_classe_vitesse", self.distribution_classe_vitesse),
        ):
            somme = sum(dist.values())
            if not 0.99 <= somme <= 1.01:
                raise ValueError(f"{nom} sums to {somme:.4f} instead of 1 ({_LOI_PATH})")
        moyenne = sum(self.facteur_jour_semaine.values()) / max(1, len(self.facteur_jour_semaine))
        if not 0.99 <= moyenne <= 1.01:
            raise ValueError(
                f"facteur_jour_semaine has mean {moyenne:.4f} instead of 1: the base "
                f"rate would no longer be the mean rate ({_LOI_PATH})"
            )
        if len(self.distribution_horaire) != 24:
            raise ValueError(f"distribution_horaire covers {len(self.distribution_horaire)} h out of 24")


@dataclass(frozen=True)
class Accident:
    """A placed accident: where, when it starts, how long it lasts."""

    arete: CleArete
    debut_ts: int
    duree_s: int
    classe_vitesse: str = ""

    @property
    def fin_ts(self) -> int:
        return self.debut_ts + self.duree_s

    def actif_a(self, ts: int) -> bool:
        return self.debut_ts <= ts < self.fin_ts

    def __str__(self) -> str:
        debut = datetime.fromtimestamp(self.debut_ts, tz=timezone.utc)
        classe = f" [{self.classe_vitesse} km/h]" if self.classe_vitesse else ""
        return (
            f"arête {self.arete[0]}→{self.arete[1]}{classe} "
            f"à {debut:%Y-%m-%d %H:%M} pour {self.duree_s // 60} min"
        )


@dataclass
class CompteursJournee:
    """What a simulated day produced. Published even when everything is zero."""

    jour: int
    tires: int = 0
    refuses: int = 0
    # Trips whose itinerary crossed an accident edge. Published even at zero:
    # without it, "no measured effect" and "no trip affected" are indistinguishable, and
    # it is the recurring pattern of the repository — the absence of measurement disguised as a result.
    trajets_touches: int = 0


class RegistreAccidents:
    """The world state: which accidents exist, and when.

    A single register per run. The draw is deterministic with a fixed seed: two runs of the
    same scenario place the same accidents at the same places — otherwise the gap between two
    runs would be blamed on the agents.
    """

    def __init__(self, config=None, loi: LoiBaac | None = None) -> None:
        self._config = config if config is not None else settings.accidents
        self._loi = loi if loi is not None else LoiBaac.charger()
        self._accidents: list[Accident] = []
        self._compteurs: list[CompteursJournee] = []
        self._jours_tires: set[int] = set()
        self._alea = random.Random(self._config.graine)
        # Edges indexed BY SPEED CLASS: (key, cumulative length in the class).
        # The class is drawn on the BAAC law, the edge in proportion to its length WITHIN the class.
        self._aretes: dict[str, list[tuple[CleArete, float]]] = {}
        self._cles: set[CleArete] = set()
        # (lat, lon) of the nodes, to map a point to an edge during a manual placement.
        self._positions: dict[int, tuple[float, float]] | None = None

    # ── Preparation ──────────────────────────────────────────────────────────

    def charger_aretes(self, graphe) -> int:
        """Index the edges of the `drive` graph by SPEED CLASS, with their length.

        The draw is done in two steps, and the order matters. The CLASS is drawn on the
        BAAC distribution; the EDGE is then drawn in proportion to its length WITHIN
        that class. Drawing directly in proportion to length, as the
        first slice did, placed 58% of the accidents in traffic-calmed zones (≤ 30 km/h), which carry
        58% of the graph's kilometres but 8% of real accidents.
        """
        par_classe: dict[str, list[tuple[CleArete, float]]] = {}
        cumuls: dict[str, float] = {}
        sans_classe = 0
        for u, v, data in graphe.edges(data=True):
            longueur = float(data.get("length") or 0.0)
            if longueur <= 0:
                continue
            classe = classe_de_vitesse(data.get("maxspeed") or data.get("speed_kph"))
            if classe is None:
                sans_classe += 1
                continue
            cumul = cumuls.get(classe, 0.0) + longueur
            cumuls[classe] = cumul
            par_classe.setdefault(classe, []).append(((u, v), cumul))

        self._aretes = par_classe
        self._cles = {cle for aretes in par_classe.values() for cle, _ in aretes}
        # Node positions: only manual placement uses them, but they are
        # only available here, when the graph is at hand.
        try:
            self._positions = {
                n: (float(d["y"]), float(d["x"]))
                for n, d in graphe.nodes(data=True)
                if "x" in d and "y" in d
            }
        except (AttributeError, TypeError, KeyError, ValueError):
            self._positions = None

        if not par_classe:
            logger.error(
                "[ALARME] No usable edge in the road graph: the accident "
                "draw will not be able to place anything. The regime stays active but inert."
            )
            return 0

        total_km = sum(cumuls.values()) / 1000.0
        detail = ", ".join(f"{c} {cumuls[c] / 1000:.0f} km" for c in sorted(cumuls))
        logger.info(
            f"[accidents] Network indexed by speed class: "
            f"{sum(len(a) for a in par_classe.values())} edges, {total_km:.0f} km — {detail}"
        )
        if sans_classe:
            logger.info(f"[accidents] {sans_classe} edge(s) without a usable speed, discarded")

        # A class that the law draws but that the network does not carry would give
        # accidents impossible to place. Say it at loading, not at the first failure.
        manquantes = [
            c for c, part in self._loi.distribution_classe_vitesse.items()
            if part > 0 and c not in par_classe
        ]
        if manquantes:
            logger.error(
                f"[ALARME] Classes present in the BAAC law but missing from the graph: "
                f"{manquantes}. The accidents that would be meant for them will be redirected "
                f"to the available classes — the geography of the draw is biased as a result."
            )
        return sum(len(a) for a in par_classe.values())

    @property
    def pret(self) -> bool:
        return bool(self._aretes)

    @property
    def loi(self) -> LoiBaac:
        return self._loi

    # ── Draw ─────────────────────────────────────────────────────────────────

    def _tirer_classe(self) -> str | None:
        """A speed class, drawn on the BAAC distribution, restricted to the network present.

        Restrict rather than retry: if a class is missing from the graph, its mass is
        redistributed over the others proportionally, and the gap has already been reported at loading.
        """
        disponibles = {c: p for c, p in self._loi.distribution_classe_vitesse.items()
                       if p > 0 and self._aretes.get(c)}
        if not disponibles:
            return None
        total = sum(disponibles.values())
        cible = self._alea.uniform(0.0, total)
        cumul = 0.0
        for classe, part in disponibles.items():
            cumul += part
            if cible <= cumul:
                return classe
        return next(iter(disponibles))

    def _tirer_arete(self, classe: str) -> CleArete:
        """An edge of this class, the probability being proportional to its length."""
        aretes = self._aretes[classe]
        cible = self._alea.uniform(0.0, aretes[-1][1])
        bas, haut = 0, len(aretes) - 1
        while bas < haut:
            milieu = (bas + haut) // 2
            if aretes[milieu][1] < cible:
                bas = milieu + 1
            else:
                haut = milieu
        return aretes[bas][0]

    def _tirer_heure(self) -> int:
        """The time of the crash, on the measured hourly distribution.

        It is the most marked conditioning of the law: 9.77% of accidents at 5 p.m.
        against 1.45% at 3 a.m. The uniform draw of the first slice made them equal.
        """
        cible = self._alea.random()
        cumul = 0.0
        for heure in range(24):
            cumul += self._loi.distribution_horaire.get(heure, 0.0)
            if cible <= cumul:
                return heure
        return 23

    def _taux_du_jour(self, jour_semaine: int, meteo: str | None) -> float:
        """Conditioned daily rate: base × day of week × weather.

        The weather factor equals 1 as long as it is not established — see the
        coefficient file, which carries the rejected computation and the reason for the rejection.
        """
        taux = self._loi.taux_base_par_jour
        if self._config.taux_journalier is not None:
            # Explicit configuration override: useful to make the mechanism
            # observable in development, never for a measurement.
            taux = float(self._config.taux_journalier)
        taux *= self._loi.facteur_jour_semaine.get(JOURS[jour_semaine], 1.0)
        if meteo and self._loi.facteur_meteo_etabli:
            taux *= self._loi.facteur_meteo.get(meteo, 1.0)
        return taux

    def _tirer_nombre(self, taux: float) -> int:
        """Number of accidents for a day, Poisson law at the conditioned rate.

        Poisson and not "rounded rate": at 1.56 accidents per day, rounding would give
        two accidents every day, i.e. a world more regular than the real one. The
        variance is part of the phenomenon.
        """
        # Poisson draw by Knuth's method — sufficient at these rates, and it adds
        # no dependency (random.Random does not expose a Poisson law).
        seuil = 2.718281828459045 ** (-taux)
        produit, n = self._alea.random(), 0
        while produit > seuil:
            produit *= self._alea.random()
            n += 1
        return n

    def tirer_journee(
        self, jour: int, debut_jour_ts: int, meteo: str | None = None
    ) -> list[Accident]:
        """Draw and place the accidents of a simulated day. Idempotent per day.

        The conditioning is that of the measured BAAC law: the NUMBER depends on the day of
        the week (and on the weather once that factor is established), the HOUR follows the hourly
        distribution, the ROAD CLASS the distribution by speed limit, and the EDGE is drawn in
        proportion to its length within its class.
        """
        if jour in self._jours_tires:
            return []
        self._jours_tires.add(jour)

        jour_semaine = wall_clock(debut_jour_ts).weekday()
        compteurs = CompteursJournee(jour=jour)
        self._compteurs.append(compteurs)

        taux = self._taux_du_jour(jour_semaine, meteo)
        if taux <= 0 or taux > float(self._config.taux_journalier_max):
            logger.error(
                f"[ALARME] Daily accident rate out of bounds: {taux} "
                f"(expected in ]0 ; {self._config.taux_journalier_max}]). No accident "
                f"drawn for day {jour} — check the law and `accidents.taux_journalier`."
            )
            return []
        if not self.pret:
            logger.error(
                f"[ALARME] Network not indexed: no drawable accident for day {jour}. "
                "The regime is active but places nothing."
            )
            return []

        nombre = self._tirer_nombre(taux)
        if nombre > MAX_ACCIDENTS_PAR_JOUR:
            logger.error(
                f"[ALARME] Aberrant draw for day {jour}: {nombre} accidents "
                f"(ceiling {MAX_ACCIDENTS_PAR_JOUR}). Draw brought down to the ceiling — "
                f"check the law ({taux} accident/day expected)."
            )
            nombre = MAX_ACCIDENTS_PAR_JOUR

        poses: list[Accident] = []
        for _ in range(nombre):
            classe = self._tirer_classe()
            if classe is None:
                compteurs.refuses += 1
                continue
            heure = self._tirer_heure()
            accident = Accident(
                arete=self._tirer_arete(classe),
                debut_ts=debut_jour_ts + heure * 3600 + self._alea.randrange(3600),
                duree_s=60
                * self._alea.randint(
                    int(self._config.duree_min_minutes), int(self._config.duree_max_minutes)
                ),
                classe_vitesse=classe,
            )
            if not self._poser(accident):
                compteurs.refuses += 1
                continue
            poses.append(accident)
            compteurs.tires += 1

        # Success is logged explicitly, and zero is a result: without this line,
        # "no accident that day" cannot be told apart from "the draw no longer runs".
        logger.info(
            f"[accidents] Day {jour} ({JOURS[jour_semaine]}) — conditioned rate "
            f"{taux:.3f}/day, accidents tirés={compteurs.tires}, refusés={compteurs.refuses}, "
            f"actifs au total={len(self._accidents)}, "
            f"trajets touchés la veille={self._compteurs[-2].trajets_touches if len(self._compteurs) > 1 else 0}"
        )
        for accident in poses:
            logger.info(f"[accidents]   placed: {accident}")
        return poses

    def _poser(self, accident: Accident) -> bool:
        """Validate then record an accident. Return false, saying so, if the placement is refused."""
        if accident.duree_s <= 0:
            logger.error(
                f"[ALARME] Accident refused: non-positive duration ({accident.duree_s} s) "
                f"on edge {accident.arete}. Nothing entered the world state."
            )
            return False
        if not self._arete_connue(accident.arete):
            logger.error(
                f"[ALARME] Accident refused: edge {accident.arete} missing from the road "
                "graph. Nothing entered the world state."
            )
            return False
        self._accidents.append(accident)
        return True

    def _arete_connue(self, arete: CleArete) -> bool:
        return arete in self._cles

    # ── Reading ──────────────────────────────────────────────────────────────

    def actifs_a(self, ts: int) -> list[Accident]:
        """The accidents in progress at this instant."""
        return [a for a in self._accidents if a.actif_a(ts)]

    def accident_sur(self, arete: CleArete, ts: int) -> Accident | None:
        """The accident active on this edge at this instant, if there is one."""
        for a in self._accidents:
            if a.arete == arete and a.actif_a(ts):
                return a
        return None

    def facteur_arete(self, arete: CleArete, ts: int) -> float:
        """What to multiply this edge's travel time by at this instant.

        Returns 1.0 when no accident is active there — by far the most frequent case,
        and that is why the search stops at the first test.
        """
        if not self._accidents:
            return 1.0
        return (
            float(self._config.facteur_ralentissement)
            if self.accident_sur(arete, ts) is not None
            else 1.0
        )

    def poser_manuellement(
        self, lat: float, lon: float, debut_ts: int, duree_minutes: int
    ) -> Accident | None:
        """Place a CHOSEN accident, on the graph edge closest to a point.

        This is the experimenter's gesture, and it is from HIM that the figures will come: the
        random draw, for its part, can show nothing on these cohorts (0.6 trip affected per
        simulated day at 1,000 agents). Same mechanism, same world, another trigger —
        the placed accident enters the same register and follows exactly the same path.

        Returns the placed accident, or `None`, logging it, if the placement is refused.
        """
        if not self.pret:
            logger.error(
                "[ALARME] Pose manuelle impossible : réseau non indexé. "
                f"Demande ({lat}, {lon}) à {debut_ts} ignorée."
            )
            return None
        if duree_minutes <= 0:
            logger.error(
                f"[ALARME] Manual placement refused: duration {duree_minutes} min not positive."
            )
            return None

        arete, classe = self._arete_la_plus_proche(lat, lon)
        if arete is None:
            logger.error(
                f"[ALARME] Manual placement refused: no edge found near ({lat}, {lon})."
            )
            return None

        accident = Accident(
            arete=arete,
            debut_ts=int(debut_ts),
            duree_s=int(duree_minutes) * 60,
            classe_vitesse=classe or "",
        )
        if not self._poser(accident):
            return None
        if self._compteurs:
            self._compteurs[-1].tires += 1
        logger.info(f"[accidents] PLACED BY HAND: {accident}")
        return accident

    def _arete_la_plus_proche(self, lat: float, lon: float) -> tuple[CleArete | None, str | None]:
        """The edge one end of which is closest to the point, by Euclidean distance.

        Deliberate approximation: we compare degrees, not metres, and only look at
        the nodes. At the scale of an urban area and to designate "the ring road here", it is
        enough — and it avoids depending on osmnx in this module.
        """
        if self._positions is None:
            logger.error(
                "[ALARME] Node positions not indexed: manual placement cannot "
                "map a point to an edge. Has the graph been loaded?"
            )
            return None, None
        meilleure, distance2 = None, float("inf")
        for classe, aretes in self._aretes.items():
            for cle, _cumul in aretes:
                pos = self._positions.get(cle[0])
                if pos is None:
                    continue
                d2 = (pos[0] - lat) ** 2 + (pos[1] - lon) ** 2
                if d2 < distance2:
                    meilleure, distance2 = (cle, classe), d2
        return meilleure if meilleure else (None, None)

    def compter_trajet_touche(self) -> None:
        """One more trip whose itinerary crossed an accident edge."""
        if self._compteurs:
            self._compteurs[-1].trajets_touches += 1

    def signature_active(self, ts: int) -> str:
        """Signature of the accidents active at this instant, for the cache keys.

        DELIBERATELY COARSE: it describes the world state, not what the agent crosses.
        Two agents none of whose itineraries crosses the accident will still have a key
        distinct from that of a world without accidents. That is over-invalidating, never
        under-invalidating — the error is on the right side, and the proper remedy would be to
        know the itinerary when building the key, which is not the case here.
        """
        actifs = self.actifs_a(ts)
        if not actifs:
            return ""
        return ",".join(
            sorted(f"{a.arete[0]}-{a.arete[1]}@{a.debut_ts}" for a in actifs)
        )

    def a_des_accidents_actifs(self, ts: int) -> bool:
        """Is there an accident in progress? Decides whether to bypass the caches."""
        return any(a.actif_a(ts) for a in self._accidents)

    @property
    def accidents(self) -> list[Accident]:
        return list(self._accidents)

    @property
    def compteurs(self) -> list[CompteursJournee]:
        return list(self._compteurs)

    def resume(self) -> str:
        total = sum(c.tires for c in self._compteurs)
        return f"{total} accident(s) posé(s) sur {len(self._compteurs)} journée(s) simulée(s)"


# ── Register of the run ──────────────────────────────────────────────────────

_registre: RegistreAccidents | None = None


def registre() -> RegistreAccidents | None:
    """The register of the current run, or `None` if the regime is not active."""
    return _registre


def initialiser() -> RegistreAccidents | None:
    """Open the run's register if the GAMA switch is true. Idempotent.

    Logs in BOTH cases: a run without accidents must say so, otherwise nothing
    distinguishes "disabled" from "the feature is broken".
    """
    global _registre
    if not settings.accidents.enabled:
        logger.info("[accidents] Regime DISABLED — no accident will be drawn.")
        _registre = None
        return None
    if _registre is None:
        _registre = RegistreAccidents()
        logger.info(
            f"[accidents] Regime ACTIVE — rate {settings.accidents.taux_journalier}/day, "
            f"duration {settings.accidents.duree_min_minutes}-"
            f"{settings.accidents.duree_max_minutes} min, seed {settings.accidents.graine}. "
            "⚠ PROVISIONAL law: not conditioned on hour, day, weather or road "
            "type. No itinerary duration is modified by this slice."
        )
    return _registre


def reinitialiser() -> None:
    """Close the register. Called between two runs, and by the tests."""
    global _registre
    _registre = None


def jour_simule(ts: int, debut_run_ts: int) -> tuple[int, int]:
    """`(simulated day number, timestamp of its start)`, anchored on the start of the run.

    Anchored on the first observed instant and not on the wall calendar, like the rest of the
    controller's time tracking: it is the same day as that of the `SIM_DAY` logs.
    """
    jour = (ts - debut_run_ts) // 86400
    return jour + 1, debut_run_ts + jour * 86400


def horizon_lisible(accident: Accident) -> str:
    """Accident window in wall-clock time, for logs read by a human."""
    debut = datetime.fromtimestamp(accident.debut_ts, tz=timezone.utc)
    return f"{debut:%H:%M}–{(debut + timedelta(seconds=accident.duree_s)):%H:%M}"
