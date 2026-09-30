"""Recorded set of trips (ticket 035, spec 01).

A **named, dated, portable** object — not a cache: all the proposals the engines
can produce for all the trips of a population's day, computed once,
**without** the vehicle filter or the option cap (the filter applies at decision time, spec 02).

On disk, a folder `data/jeux/<nom>/`:

    MANIFEST.yaml        identity (population + fingerprint), dependencies, completeness, sha256 of the lines
    propositions.jsonl   one JSON line per trip (person, activities, time, proposals)

Rules held here: J1 (population by name + fingerprint), J2 (trips derived from the agendas),
J3 (superset: all modes, no cap), J5 (identity = content fingerprint), J7/J8
(visible completeness), J9/J10 (dependencies and staleness), J11 (resumption), J12/J13 (portable, inert,
refusal of an altered or malformed set), J14 (immutable after closing), J15 (log + ALARM), J16
(no persona attribute).
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chaine_activites import paires_de_la_journee
from experiences import froid
from experiences.decision import SOURCE_ENREGISTREE, SOURCE_LOCALE, Proposition
from experiences.population import InfoPopulation, sha256_fichier
from helper import shift_weekend_departure_to_monday, to_timestamp_based_on_day
from models import Location, Person, TravelPlan
from settings import settings

VERSION_JEU = "jeu1"
FICHIER_MANIFEST = "MANIFEST.yaml"
FICHIER_PROPOSITIONS = "propositions.jsonl"
FICHIER_EQUIVALENCES = (
    "EQUIVALENCES.yaml"  # days whose transit supply was MEASURED identical to the set's
)
HEURE_REFERENCE_DEPART = "depart_programme"

MOTIF_ORIGINE_EGALE_DESTINATION = "origine_egale_destination"
MOTIF_AUCUNE_PROPOSITION = "aucune_proposition"


def est_defaillance_moteur(motif: str | None) -> bool:
    """Does this absence reason say that an engine FAILED, or only that there was nothing to do?

    A structuring distinction, and confusing it made three counters lie at once
    (ticket 045). Between a point and itself there is no itinerary to compute: an
    on-the-spot closure is not a failure, it is a property of the day. It
    therefore enters neither the engines' health alarm nor the denominator of a
    set's completeness.

    The stakes are quantified: in the v5 cohort, 138 of the 3,299 trips have their origin as
    destination — 77 day closures (the person already ends at home) and 61 intermediate
    trips between two activities at the same place. That is 4.2 %, very close to the 5 %
    alarm threshold: counting them would make the alarm scream at the slightest real failure, which
    amounts to switching it off. The v5 set prepared on 2026-09-11 confirms it — 138 without a proposal,
    and ZERO engine failures.
    """
    return bool(motif) and motif != MOTIF_ORIGINE_EGALE_DESTINATION


# Files whose fingerprint enters the dependencies (J9). Relative to the GTFS folder in
# service for the feeds (and the OTP graph that lives there), to `services/llm-agents/config` for the rules.
_FICHIERS_GTFS = (
    "feed_info.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "routes.txt",
    "stops.txt",
    "trips.txt",
    "stop_times.txt",
)
_FICHIER_GRAPHE_OTP = "graph.obj"
_FICHIERS_CONFIG = ("osmnx.yaml", "terminal_time.yaml", "school_bus.yaml")


class JeuInvalide(ValueError):
    """The set cannot be loaded: altered, malformed, without dependencies, other population."""


class JeuClos(RuntimeError):
    """Write refused: the set is closed (J14)."""


# ── Line models ──────────────────────────────────────────────────────────────


class Deplacement(BaseModel):
    """A person's trip between two consecutive located activities (J2)."""

    model_config = ConfigDict(extra="forbid")

    person_id: str
    activity_id: str  # DESTINATION activity — the key of the controller's decisions
    origine_activity_id: str
    ordinal: int  # rank of the trip in the person's day (0, 1, …)
    purpose: str
    depart_24h: int  # scheduled departure time, seconds since midnight
    depart_ts: int  # GAMA (wall-clock) timestamp of the departure for `jour_simule`
    origine: Location
    destination: Location

    @property
    def cle(self) -> tuple[str, str]:
        return (self.person_id, self.activity_id)


class PropositionEnregistree(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = SOURCE_ENREGISTREE
    plan: dict  # TravelPlan.model_dump() — read back by TravelPlan.model_validate


class LigneJeu(Deplacement):
    propositions: list[PropositionEnregistree] = Field(default_factory=list)
    motif_absence: str | None = None

    def vers_propositions(self) -> list[Proposition]:
        return [
            Proposition(TravelPlan.model_validate(p.plan), p.source)
            for p in self.propositions
        ]


# ── Expected trips (J2, E22) ─────────────────────────────────────────────────


def jour_base_ts(jour_simule: str) -> int:
    """Midnight of the simulated day, as a GAMA timestamp (wall-clock time encoded as UTC, cf. sim_clock)."""
    return calendar.timegm(datetime.strptime(jour_simule, "%Y-%m-%d").timetuple())


def _purpose_str(purpose) -> str:
    return str(getattr(purpose, "value", purpose) or "")


def deplacements_attendus(
    personnes: Sequence[Person], jour_simule: str
) -> list[Deplacement]:
    """Derives each person's trips of the day from their agenda — never entered by hand (J2).

    **The day closes on itself** (ticket 045, A1). The enumeration goes through `paires_de_la_journee`,
    the same cyclic chain as the simulation controller: n activities make n trips,
    the last one being the return to the first activity. Previously this function stopped at the
    last activity and returned n − 1, so that the return home was never decided —
    27 % of the day on v1, and precisely the part where the vehicle chain constraint
    constrains the choice the most.

    Same time rule as the controller: departure at `scheduled_start_time` (otherwise `end_time`) of
    the destination activity, resolved on the simulated day; a time already past when the person
    arrives at the previous activity rolls over to the next day; a weekend departure is postponed to Monday
    when `agent.no_weekend_departures` requires it. For the closure, this rule designates the same
    instant as "end of the last activity": the sealed populations encode the loop
    (`scheduled_start_time` of the first activity == `end_time` of the last, with no gap across
    the 1,894 persons of the v1 and v5 cohorts). An activity without a location is neither origin nor
    destination.
    """
    base = jour_base_ts(jour_simule)
    attendus: list[Deplacement] = []
    for personne in personnes:
        for ordinal, (precedente, act) in enumerate(
            paires_de_la_journee(personne.identity.activities or [])
        ):
            cible_24h = (
                act.scheduled_start_time
                if act.scheduled_start_time is not None
                else act.end_time
            )
            # Ticket 057 — the origin activity can SPAN MIDNIGHT, and that is the case of the
            # first one of each day: the cohort's chains start with "home",
            # which begins the evening before. Its `start_time` therefore belongs to the PREVIOUS day,
            # and anchoring it on `base` puts the cursor at 20:07 of the simulated day. The morning
            # departure being earlier, the rollover line dated it to the NEXT DAY: 797 of the
            # 894 personas of v6 lost their first trip, then discarded by the
            # scorer's cut, and the set recorded for them an itinerary computed on the
            # wrong day. An activity that spans midnight is recognised by its `start_time`
            # later than its `end_time`; the cursor is then moved back by one day.
            debut_precedente = int(precedente.start_time)
            if precedente.end_time is not None and debut_precedente > int(
                precedente.end_time
            ):
                debut_precedente -= 86400
            maintenant = base + debut_precedente
            depart = to_timestamp_based_on_day(int(cible_24h), maintenant)
            if depart < maintenant:
                depart += 86400
            if settings.agent.no_weekend_departures:
                depart = shift_weekend_departure_to_monday(depart)
            attendus.append(
                Deplacement(
                    person_id=personne.person_id,
                    activity_id=act.id,
                    origine_activity_id=precedente.id,
                    ordinal=ordinal,
                    purpose=_purpose_str(act.purpose),
                    depart_24h=int(cible_24h) % 86400,
                    depart_ts=int(depart),
                    origine=precedente.location,
                    destination=act.location,
                )
            )
    return attendus


# ── Dependencies (J9, J10) ───────────────────────────────────────────────────


def _racine_depot() -> Path:
    # Anchor rather than counting levels — cf. `experiences.chemins` (ticket 039).
    from experiences.chemins import racine_depot

    return racine_depot()


# Repository state passed by the HOST to the container (R7). `git` is not installed in the
# `controller` image, so `_git` always returned `None` there: the platform's 36 runs
# carry `empreintes.depot = {commit: null, arbre_propre: null}` and none of them can
# be linked to a state of the code. The Makefile fills in these two variables at launch.
ENV_COMMIT = "EXP_DEPOT_COMMIT"
ENV_ARBRE_PROPRE = "EXP_DEPOT_ARBRE_PROPRE"


def etat_depot() -> tuple[str | None, bool | None]:
    """(commit, clean tree) — from the environment if it carries them, otherwise from `git`.

    The environment first: it is the only channel available in the container, and when it
    is filled in, the host measured the repository state at launch time, which
    is precisely the question. `git` remains the fallback for a direct launch on the host.

    Neither one guesses: a missing value is `None` and reads as "not
    verifiable", never as "clean".
    """
    commit = (os.getenv(ENV_COMMIT) or "").strip() or None
    brut = (os.getenv(ENV_ARBRE_PROPRE) or "").strip().lower()
    propre = {
        "1": True,
        "true": True,
        "oui": True,
        "0": False,
        "false": False,
        "non": False,
    }.get(brut)
    if commit is not None:
        return commit, propre
    statut = _git("status", "--porcelain", "--untracked-files=no")
    return _git("rev-parse", "HEAD"), ((statut == "") if statut is not None else None)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_racine_depot(),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cle_graphe_osmnx() -> str | None:
    """Key of the OSMnx graph actually served, or `None` if the routing module is missing.

    Deferred, tolerant import: `dependances_courantes` must remain callable on a host
    without the full routing chain (tests, tooling). But the absence is
    logged — a silent `None` is exactly the defect being fixed here.
    """
    try:
        from trip_helper.osmnx_direct import graph_key

        return graph_key()
    except Exception as e:  # noqa: BLE001 — an unreadable dependency is stated, it is not invented
        logger.warning(
            f"[jeu] OSMnx graph key unreadable ({e}): the dependency will be recorded "
            f"as unverifiable, and a graph change will not make this set stale."
        )
        return None


def chemin_graphe_otp(dossier_flux: Path) -> Path | None:
    """The `graph.obj` that OTP actually loads, or `None` if there is none.

    OTP is started with `--load /var/otp/toulouse`, where `data/gtfs` is mounted: the graph is
    therefore at the ROOT of that folder, not in a feed's subfolder. Looking under
    `settings.gtfs.gtfs_file` — which designates the Tisséo feed — hit a stale namesake:
    84.3 MB from 4 September at the root, 79.3 MB from 19 May in the feed, two different
    fingerprints, and the manifest named the wrong one (observed on 2026-09-11).

    Staleness (J10) could therefore never see a rebuild of the graph in service.
    Watching this one is also more complete: it embeds the THREE feeds — Tisséo, liO,
    TER — and the OSM base, whereas the text-file fingerprints only cover Tisséo.

    The feed folder remains a fallback, for a repository organised differently. A total absence is
    logged: returning `None` silently is indistinguishable from an unchanged graph.
    """
    dossier_flux = Path(dossier_flux)
    for candidat in (
        dossier_flux.parent / _FICHIER_GRAPHE_OTP,
        dossier_flux / _FICHIER_GRAPHE_OTP,
    ):
        if candidat.is_file():
            return candidat
    logger.warning(
        f"[jeu] {_FICHIER_GRAPHE_OTP} not found (neither {dossier_flux.parent} nor {dossier_flux}): "
        f"the dependency will be recorded as unverifiable, and a rebuild of the graph "
        f"will not make this set stale."
    )
    return None


def _sha_si_present(chemin: Path) -> str | None:
    return sha256_fichier(chemin) if chemin.is_file() else None


def dependances_courantes(
    dossier_gtfs: Path | None = None, dossier_config: Path | None = None
) -> dict:
    """What the proposals depend on: transport-supply data, graph, routing rules, repository.

    A dependency that cannot be found is `None` — recorded as is, never invented; the
    comparison (`perime`) then files it under "unverifiable".
    """
    gtfs = Path(dossier_gtfs) if dossier_gtfs else Path(settings.gtfs.gtfs_file)
    config = (
        Path(dossier_config)
        if dossier_config
        else Path(__file__).resolve().parents[1] / "config"
    )
    commit, arbre_propre = etat_depot()
    if commit is None:
        logger.warning(
            "[jeu] repository state unverifiable: neither "
            f"${ENV_COMMIT} nor `git` answers. The measurement cannot be linked "
            "to a state of the code (ticket 045, A4)."
        )
    return {
        "commit": commit,
        "arbre_propre": arbre_propre,
        "gtfs": {nom: _sha_si_present(gtfs / nom) for nom in _FICHIERS_GTFS},
        # The graph THAT OTP LOADS, not a namesake from the feed folder (ticket 045).
        "otp_graph_sha256": (
            _sha_si_present(chemin) if (chemin := chemin_graphe_otp(gtfs)) else None
        ),
        # The EFFECTIVE key, not the raw setting (R8). `settings.gtfs.osmnx_graph_key` is
        # `None` in the usual case — the graph served is then that of the polygon of the 453
        # communes — and the manifest therefore recorded "nothing" precisely when all was
        # well. An unrecorded dependency cannot be compared: a set's staleness could
        # not see a graph change.
        "osmnx_graph_key": _cle_graphe_osmnx(),
        "config": {nom: _sha_si_present(config / nom) for nom in _FICHIERS_CONFIG},
    }


def _aplatir(d: dict, prefixe: str = "") -> dict[str, object]:
    plat: dict[str, object] = {}
    for k, v in (d or {}).items():
        cle = f"{prefixe}/{k}" if prefixe else str(k)
        if isinstance(v, dict):
            plat.update(_aplatir(v, cle))
        else:
            plat[cle] = v
    return plat


def comparer_dependances(
    enregistrees: dict, courantes: dict
) -> tuple[list[str], list[str]]:
    """(dependencies that differ, unverifiable dependencies). `arbre_propre` and `commit` are informative (audit trail)."""
    a, b = _aplatir(enregistrees), _aplatir(courantes)
    differentes, non_verifiables = [], []
    for cle in sorted(set(a) | set(b)):
        if cle in ("arbre_propre", "commit"):
            continue
        va, vb = a.get(cle), b.get(cle)
        if va is None or vb is None:
            if va is not None or vb is not None:
                non_verifiables.append(cle)
        elif va != vb:
            differentes.append(cle)
    return differentes, non_verifiables


# ── Reading / writing lines ──────────────────────────────────────────────────


def _lire_lignes(
    chemin: Path, *, tolerer_troncature: bool
) -> tuple[list[LigneJeu], int]:
    """Valid lines + bytes up to the last complete line (to truncate a half-written line).

    A malformed line is an error that names its position (J13) — except, during preparation, the
    last line of a file interrupted mid-write, discarded with a WARNING (J11).
    """
    lignes: list[LigneJeu] = []
    if not chemin.exists():
        return lignes, 0
    octets_valides = 0
    with open(chemin, "rb") as f:
        brut = f.read()
    for numero, ligne in enumerate(brut.split(b"\n"), start=1):
        if not ligne.strip():
            octets_valides += len(ligne) + 1
            continue
        try:
            lignes.append(LigneJeu.model_validate(json.loads(ligne.decode("utf-8"))))
        except (
            ValueError,
            ValidationError,
        ) as e:  # json.JSONDecodeError is a ValueError
            derniere = octets_valides + len(ligne) >= len(brut.rstrip(b"\n"))
            if tolerer_troncature and derniere:
                logger.warning(
                    f"[jeu] Last line of {chemin.name} truncated or malformed (line {numero}) — discarded, it will be recomputed"
                )
                return lignes, octets_valides
            raise JeuInvalide(
                f"{chemin.name} line {numero}: malformed content ({str(e).splitlines()[0][:120]})"
            ) from e
        octets_valides += len(ligne) + 1
    return lignes, min(octets_valides, len(brut))


def _ecrire_yaml_atomique(chemin: Path, contenu: dict) -> None:
    tmp = chemin.with_suffix(chemin.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(contenu, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(tmp, chemin)


def _maintenant_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Preparation ──────────────────────────────────────────────────────────────


class JeuEnPreparation:
    """A set open for writing: line-by-line append, resumption, closing (J11, J14, J15)."""

    def __init__(
        self,
        dossier: Path,
        manifest: dict,
        cles: set[tuple[str, str]],
        lignes: list[LigneJeu],
    ):
        self.dossier = Path(dossier)
        self.manifest = manifest
        self.cles = cles
        self._lignes = lignes
        self._fh = open(self.dossier / FICHIER_PROPOSITIONS, "ab")

    @classmethod
    def ouvrir(
        cls,
        dossier: str | Path,
        nom: str,
        population: InfoPopulation,
        jour_simule: str,
        *,
        heure_reference: str = HEURE_REFERENCE_DEPART,
        dependances: dict | None = None,
    ) -> JeuEnPreparation:
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        chemin_manifest = dossier / FICHIER_MANIFEST
        if chemin_manifest.exists():
            manifest = yaml.safe_load(chemin_manifest.read_text(encoding="utf-8")) or {}
            if manifest.get("clos"):
                raise JeuClos(
                    f"set {manifest.get('nom')!r} is closed: correcting it produces a NEW set (J14)"
                )
            if (manifest.get("population") or {}).get("sha256") != population.sha256:
                raise JeuInvalide(
                    f"folder {dossier} holds a set for another population "
                    f"({(manifest.get('population') or {}).get('sha256', '?')[:12]}… ≠ {population.sha256[:12]}…)"
                )
        else:
            manifest = {
                "version": VERSION_JEU,
                "nom": nom,
                "cree_le": _maintenant_iso(),
                "clos": False,
                "clos_le": None,
                "population": population.as_dict(),
                "jour_simule": jour_simule,
                "heure_reference": heure_reference,
                "dependances": dependances
                if dependances is not None
                else dependances_courantes(),
            }
            _ecrire_yaml_atomique(chemin_manifest, manifest)
        lignes, octets = _lire_lignes(
            dossier / FICHIER_PROPOSITIONS, tolerer_troncature=True
        )
        chemin_lignes = dossier / FICHIER_PROPOSITIONS
        if chemin_lignes.exists() and octets < chemin_lignes.stat().st_size:
            with open(chemin_lignes, "r+b") as f:
                f.truncate(octets)
        return cls(dossier, manifest, {l.cle for l in lignes}, lignes)

    @property
    def nom(self) -> str:
        return str(self.manifest.get("nom"))

    def ecrire(self, ligne: LigneJeu) -> None:
        if self.manifest.get("clos"):
            raise JeuClos("set closed")
        if ligne.cle in self.cles:
            return
        self._fh.write(
            (
                json.dumps(ligne.model_dump(mode="json"), ensure_ascii=False) + "\n"
            ).encode("utf-8")
        )
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.cles.add(ligne.cle)
        self._lignes.append(ligne)

    def clore(self, attendus: Sequence[Deplacement], personnes_total: int) -> dict:
        """Freezes the set: completeness, list of trips without proposals, content fingerprint (J5, J7, J14)."""
        self._fh.close()
        cles_attendues = {d.cle for d in attendus}
        couvertes = [
            l for l in self._lignes if l.cle in cles_attendues and l.propositions
        ]
        sans = [
            l for l in self._lignes if l.cle in cles_attendues and not l.propositions
        ]
        self.manifest.update(
            {
                "clos": True,
                "clos_le": _maintenant_iso(),
                "attendus": {
                    "deplacements": len(cles_attendues),
                    "personnes_total": int(personnes_total),
                    "personnes_avec_deplacement": len({d.person_id for d in attendus}),
                },
                "couverts": {
                    "deplacements": len(couvertes),
                    "personnes": len({l.person_id for l in couvertes}),
                },
                "non_calcules": len(cles_attendues) - len(couvertes) - len(sans),
                "sans_proposition": [
                    {
                        "person_id": l.person_id,
                        "activity_id": l.activity_id,
                        "motif": l.motif_absence or MOTIF_AUCUNE_PROPOSITION,
                    }
                    for l in sans
                ],
                "propositions_sha256": sha256_fichier(
                    self.dossier / FICHIER_PROPOSITIONS
                ),
            }
        )
        _ecrire_yaml_atomique(self.dossier / FICHIER_MANIFEST, self.manifest)
        return self.manifest

    def fermer(self) -> None:
        if not self._fh.closed:
            self._fh.close()


async def preparer(
    prep: JeuEnPreparation,
    personnes: Sequence[Person],
    trip_helper,
    *,
    concurrence: int = 8,
    seuil_sans_proposition: float = 0.05,
    fabrique_locale: Callable[..., TravelPlan | None] | None = None,
    progression_s: float = 5.0,
) -> dict:
    """Computes all the proposals of the trips not yet recorded (J3, J11, J15).

    `trip_helper.get_itineraries(origin, destination, departure_time, include_car, include_bike,
    arrive_by)` is called with **all modes open**, no cap. `fabrique_locale` (by default the
    synthetic school bus) adds the proposals produced without an engine. An engine error
    writes nothing: the trip remains to be computed at the next resumption.
    """
    debut = time.monotonic()
    jour = str(prep.manifest["jour_simule"])
    attendus = deplacements_attendus(personnes, jour)
    restants = [d for d in attendus if d.cle not in prep.cles]
    par_personne = {p.person_id: p for p in personnes}
    activites = {
        (p.person_id, a.id): a for p in personnes for a in (p.identity.activities or [])
    }
    logger.info(
        f"[jeu] Préparation de {prep.nom!r} : {len(attendus)} déplacements attendus pour "
        f"{len(personnes)} personnes — {len(attendus) - len(restants)} déjà enregistrés, reste {len(restants)}"
    )
    if fabrique_locale is None:
        from trip_helper.school_bus import build_school_bus_option

        fabrique_locale = build_school_bus_option

    compteurs: Counter = Counter()
    sem = asyncio.Semaphore(max(1, concurrence))
    alarme_levee = False

    async def traiter(dep: Deplacement) -> None:
        nonlocal alarme_levee
        async with sem:
            props: list[PropositionEnregistree] = []
            motif: str | None = None
            if (
                abs(dep.origine.lat - dep.destination.lat) < 1e-9
                and abs(dep.origine.lon - dep.destination.lon) < 1e-9
            ):
                motif = MOTIF_ORIGINE_EGALE_DESTINATION
                compteurs["ignores_meme_lieu"] += 1
            else:
                try:
                    itineraires = await trip_helper.get_itineraries(
                        origin=dep.origine,
                        destination=dep.destination,
                        departure_time=dep.depart_ts,
                        include_car=True,
                        include_bike=True,
                        arrive_by=False,
                    )
                except Exception as e:  # noqa: BLE001 — journalisé avec de quoi agir, jamais enregistré
                    compteurs["erreurs"] += 1
                    logger.error(
                        f"[jeu] Engine error for {dep.person_id}/{dep.activity_id} "
                        f"({dep.origine.lat:.5f},{dep.origine.lon:.5f} → {dep.destination.lat:.5f},{dep.destination.lon:.5f} "
                        f"at {dep.depart_ts}): {type(e).__name__}: {e}"
                    )
                    return
                for it in itineraires or []:
                    it.purpose = dep.purpose
                    it.start_location = dep.origine
                    it.end_location = dep.destination
                    props.append(
                        PropositionEnregistree(
                            source=SOURCE_ENREGISTREE, plan=it.model_dump()
                        )
                    )
                personne = par_personne.get(dep.person_id)
                activite = activites.get((dep.person_id, dep.activity_id))
                if personne is not None and activite is not None:
                    try:
                        locale = fabrique_locale(
                            person=personne,
                            from_location=dep.origine,
                            next_activity=activite,
                            timestamp=dep.depart_ts,
                            departure_time=dep.depart_ts,
                        )
                    except Exception as e:  # noqa: BLE001
                        locale = None
                        logger.warning(
                            f"[jeu] local proposal impossible for {dep.person_id}/{dep.activity_id}: {e}"
                        )
                    if locale is not None:
                        locale.purpose = dep.purpose
                        props.append(
                            PropositionEnregistree(
                                source=SOURCE_LOCALE, plan=locale.model_dump()
                            )
                        )
                        compteurs["locales"] += 1
                if not props:
                    motif = MOTIF_AUCUNE_PROPOSITION
            prep.ecrire(
                LigneJeu(**dep.model_dump(), propositions=props, motif_absence=motif)
            )
            compteurs["calcules"] += 1
            if not props:
                compteurs["sans_proposition"] += 1
                # The alarm is about ENGINE HEALTH, so only real failures
                # enter it: an on-the-spot closure has no itinerary to compute and says
                # nothing about OTP or OSMnx. Counting them would cross the 5 % threshold at every
                # preparation (5.5 % on-the-spot closures in the v5 cohort), which amounts
                # to switching the alarm off by making it scream all the time.
                if est_defaillance_moteur(motif):
                    compteurs["defaillances_moteur"] += 1
                    part = compteurs["defaillances_moteur"] / max(1, len(attendus))
                    if part > seuil_sans_proposition and not alarme_levee:
                        alarme_levee = True
                        logger.error(
                            f"[ALARME] Set {prep.nom!r}: {compteurs['defaillances_moteur']} trips without "
                            f"engine proposals out of {len(attendus)} expected "
                            f"({100 * part:.1f} % > {100 * seuil_sans_proposition:.1f} %) — "
                            f"excluding on-the-spot closures, which have no itinerary to compute"
                        )

    def ecrire_progression() -> None:
        faits = compteurs["calcules"] + compteurs["erreurs"]
        ecoule = time.monotonic() - debut
        debit = faits / ecoule if ecoule > 0 else 0.0
        reste = (len(restants) - faits) / debit if debit > 0 else None
        contenu = {
            "jeu": prep.nom,
            "faits": faits + (len(attendus) - len(restants)),
            "total": len(attendus),
            "restants_au_depart": len(restants),
            "pourcent": round(
                100 * (faits + len(attendus) - len(restants)) / len(attendus), 1
            )
            if attendus
            else None,
            "sans_proposition": int(compteurs["sans_proposition"]),
            "erreurs": int(compteurs["erreurs"]),
            "ecoule_s": round(ecoule, 1),
            "reste_s": (round(reste) if reste is not None else None),
            "maj": _maintenant_iso(),
        }
        try:
            tmp = prep.dossier / "progression.json.tmp"
            tmp.write_text(json.dumps(contenu, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, prep.dossier / "progression.json")
        except OSError as e:
            logger.warning(f"[jeu] progression.json not written: {e}")

    async def progression() -> None:
        while True:
            await asyncio.sleep(progression_s)
            faits = compteurs["calcules"] + compteurs["erreurs"]
            ecoule = time.monotonic() - debut
            debit = faits / ecoule if ecoule > 0 else 0.0
            reste = (len(restants) - faits) / debit if debit > 0 else float("nan")
            logger.info(
                f"[jeu] {faits}/{len(restants)} trips processed · {ecoule:.0f} s elapsed · ≈ {reste:.0f} s left"
            )
            ecrire_progression()

    ecrire_progression()
    suivi = asyncio.create_task(progression())
    try:
        await asyncio.gather(*(traiter(d) for d in restants))
    finally:
        suivi.cancel()
        ecrire_progression()
    duree = time.monotonic() - debut
    compteurs["restants_apres"] = len(attendus) - len(
        prep.cles & {d.cle for d in attendus}
    )
    if compteurs["sans_proposition"]:
        # Two kinds, two sentences: an on-the-spot closure is EXPECTED (it has no
        # itinerary), an engine failure is not. Adding them up under
        # "without ANY engine proposal" blamed the engines for a fact of the day.
        logger.warning(
            f"[jeu] {compteurs['sans_proposition']} unusable trip(s) out of {len(attendus)} "
            f"({100 * compteurs['sans_proposition'] / max(1, len(attendus)):.1f} %) — including "
            f"{compteurs['ignores_meme_lieu']} on-the-spot closure(s) (origin = destination, no "
            f"itinerary to compute) and {compteurs['defaillances_moteur']} without engine proposals. "
            f"All excluded from the experiments' expected trips (decision of 2026-09-06); details: `consulter-jeu`"
        )
    niveau = logger.error if compteurs["erreurs"] else logger.info
    niveau(
        f"[jeu] {'Préparation terminée' if not compteurs['erreurs'] else 'Préparation INCOMPLÈTE'} pour {prep.nom!r} en {duree:.1f} s — "
        f"calculés {compteurs['calcules']}, sans proposition {compteurs['sans_proposition']}, "
        f"même lieu {compteurs['ignores_meme_lieu']}, locales {compteurs['locales']}, erreurs {compteurs['erreurs']}, "
        f"reste {compteurs['restants_apres']}"
    )
    compteurs["duree_s"] = round(duree, 3)
    return dict(compteurs)


# ── Loading ──────────────────────────────────────────────────────────────────


class Jeu:
    """A closed (or in-progress) set, read from disk, read-only (J6, J7, J12)."""

    def __init__(self, dossier: Path, manifest: dict, lignes: list[LigneJeu]):
        self.dossier = Path(dossier)
        self.manifest = manifest
        self._index: dict[tuple[str, str], LigneJeu] = {}
        for l in lignes:
            if l.cle in self._index:
                logger.warning(
                    f"[jeu] duplicate key {l.cle} in {self.dossier.name} — first occurrence kept"
                )
                continue
            self._index[l.cle] = l
        self._par_personne: dict[str, list[LigneJeu]] = {}
        for l in self._index.values():
            self._par_personne.setdefault(l.person_id, []).append(l)
        for lst in self._par_personne.values():
            lst.sort(key=lambda x: (x.depart_ts, x.ordinal))

    @classmethod
    def charger(
        cls,
        dossier: str | Path,
        *,
        verifier: bool = True,
        archive_confirmee: str | None = None,
    ) -> Jeu:
        """Opens a sealed set. The SINGLE entry point for any reading of a set.

        `archive_confirmee` (ticket 074, A-4) carries the REASON for a cold-archive read —
        the D-7 comparability guard, for instance, which compares the activity chains of
        v6 with those of the frozen v5. Without a reason, a set stored under `archive/` is refused: this is
        where the refusal is placed, because this is where everyone goes through.
        """
        dossier = Path(dossier)
        froid.verifier(
            dossier,
            archive_confirmee,
            quoi="a sealed set",
            comment_lever='pass `archive_confirmee="<reason>"` to `Jeu.charger`',
        )
        chemin_manifest = dossier / FICHIER_MANIFEST
        if not chemin_manifest.is_file():
            raise JeuInvalide(f"no {FICHIER_MANIFEST} in {dossier}")
        try:
            manifest = yaml.safe_load(chemin_manifest.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise JeuInvalide(f"{FICHIER_MANIFEST} unreadable: {e}") from e
        if manifest.get("version") != VERSION_JEU:
            raise JeuInvalide(
                f"unknown set version: {manifest.get('version')!r} (expected {VERSION_JEU!r})"
            )
        if (
            not isinstance(manifest.get("dependances"), dict)
            or not manifest["dependances"]
        ):
            raise JeuInvalide("set without a `dependances` block: refused (J9)")
        if not (manifest.get("population") or {}).get("sha256"):
            raise JeuInvalide("set without a population fingerprint: refused (J1)")
        chemin_lignes = dossier / FICHIER_PROPOSITIONS
        if manifest.get("clos") and verifier:
            attendu = manifest.get("propositions_sha256")
            reel = sha256_fichier(chemin_lignes) if chemin_lignes.exists() else None
            if not attendu or attendu != reel:
                raise JeuInvalide(
                    f"set {manifest.get('nom')!r} altered: fingerprint {str(reel)[:12]}… ≠ MANIFEST {str(attendu)[:12]}… (J12)"
                )
        lignes, _ = _lire_lignes(
            chemin_lignes, tolerer_troncature=not manifest.get("clos")
        )
        return cls(dossier, manifest, lignes)

    # ── identity ──
    @property
    def nom(self) -> str:
        return str(self.manifest.get("nom"))

    @property
    def empreinte(self) -> str | None:
        return self.manifest.get("propositions_sha256")

    @property
    def clos(self) -> bool:
        return bool(self.manifest.get("clos"))

    @property
    def population(self) -> dict:
        return dict(self.manifest.get("population") or {})

    @property
    def jour_simule(self) -> str:
        return str(self.manifest.get("jour_simule"))

    def jours_equivalents(self) -> list[str]:
        """Days (YYYY-MM-DD) whose transport supply was measured identical to the set's.

        Declared by `verifier-jours --declarer` in `EQUIVALENCES.yaml`, next to the MANIFEST (which
        remains immutable, J14). Without this file, no other day is deemed equivalent: the
        simulation then recomputes public transport for any other day (decision 18).
        """
        p = self.dossier / FICHIER_EQUIVALENCES
        if not p.is_file():
            return []
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return []
        return [str(j) for j in (data.get("jours_equivalents") or [])]

    def _equivalences(self) -> dict:
        p = self.dossier / FICHIER_EQUIVALENCES
        if not p.is_file():
            return {}
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return {}

    def declarer_jour_equivalent(self, jour: str, mesure: dict) -> None:
        """The whole day is equivalent (transit supply measured identical on a sample, "moteurs" method)."""
        data = self._equivalences()
        jours = [str(j) for j in (data.get("jours_equivalents") or [])]
        if jour not in jours:
            jours.append(jour)
        data["jours_equivalents"] = sorted(jours)
        data.setdefault("mesures", {})[jour] = mesure
        _ecrire_yaml_atomique(self.dossier / FICHIER_EQUIVALENCES, data)

    def declarer_verification_jour(self, jour: str, resultat: dict) -> None:
        """PER-TRIP validity for `jour` ("gtfs_existence" method): what is missing is named.

        If everything is valid, the day also becomes equivalent; otherwise only the trips
        listed as `invalides` will have their public transport recomputed on that day.
        """
        data = self._equivalences()
        data.setdefault("par_jour", {})[jour] = {
            k: v for k, v in resultat.items() if k not in ("invalides_detail",)
        }
        if resultat.get("equivalent"):
            jours = [str(j) for j in (data.get("jours_equivalents") or [])]
            if jour not in jours:
                jours.append(jour)
            data["jours_equivalents"] = sorted(jours)
        _ecrire_yaml_atomique(self.dossier / FICHIER_EQUIVALENCES, data)

    def ligne_valide_le(self, jour: str, cle: tuple[str, str]) -> bool | None:
        """Do this trip's transit proposals hold for `jour`?

        True: the set's day, a day declared equivalent, or a trip verified valid on that day;
        False: verified and a service run is missing; None: nothing was verified for that day.
        """
        if jour == self.jour_simule or jour in self.jours_equivalents():
            return True
        par_jour = (self._equivalences().get("par_jour") or {}).get(jour)
        if not par_jour:
            return None
        return f"{cle[0]}|{cle[1]}" not in set(par_jour.get("invalides_cles") or [])

    def verifier_population(self, info: InfoPopulation) -> str | None:
        """Refusal message if the population is not the set's (G1, E6), otherwise None."""
        attendu = self.population.get("sha256")
        if attendu != info.sha256:
            return (
                f"le jeu {self.nom!r} a été préparé pour la population {self.population.get('nom')!r} "
                f"(empreinte {str(attendu)[:12]}…), pas pour {info.nom!r} (empreinte {info.sha256[:12]}…)"
            )
        return None

    # ── reading ──
    def ligne(self, person_id: str, activity_id: str) -> LigneJeu | None:
        return self._index.get((person_id, activity_id))

    def propositions(
        self, person_id: str, activity_id: str
    ) -> list[Proposition] | None:
        """Raw proposals of the trip — `None` if the set does not cover it, `[]` if there are none."""
        l = self._index.get((person_id, activity_id))
        return None if l is None else l.vers_propositions()

    def couvre(self, person_id: str, activity_id: str) -> bool:
        l = self._index.get((person_id, activity_id))
        return l is not None and bool(l.propositions)

    def consulter(self, person_id: str) -> list[LigneJeu]:
        return list(self._par_personne.get(person_id, []))

    def personnes(self) -> list[str]:
        return sorted(self._par_personne)

    def __len__(self) -> int:
        return len(self._index)

    # ── completeness (J7, J8) ──
    def couverture(self) -> dict:
        attendus = self.manifest.get("attendus") or {}
        couverts = self.manifest.get("couverts") or {}
        n_attendus = int(attendus.get("deplacements") or 0)
        n_couverts = int(
            couverts.get("deplacements")
            or sum(1 for l in self._index.values() if l.propositions)
        )
        # A trip whose origin is its destination CANNOT be covered: there is
        # no itinerary to compute. Leaving it in the denominator made `est_complet`
        # unreachable as soon as the day closes back at home, that is, always
        # (ticket 045). The count is read back from the MANIFEST's reasons, so already
        # sealed sets are read without being touched.
        n_hors_portee = sum(
            1
            for s in (self.manifest.get("sans_proposition") or [])
            if not est_defaillance_moteur(s.get("motif"))
        )
        n_exploitables = max(0, n_attendus - n_hors_portee)
        return {
            "deplacements_attendus": n_attendus,
            "deplacements_exploitables": n_exploitables,
            "deplacements_couverts": n_couverts,
            "personnes_total": int(attendus.get("personnes_total") or 0),
            "personnes_avec_deplacement": int(
                attendus.get("personnes_avec_deplacement") or 0
            ),
            "personnes_couvertes": int(
                couverts.get("personnes")
                or len({l.person_id for l in self._index.values() if l.propositions})
            ),
            "sans_proposition": list(self.manifest.get("sans_proposition") or []),
            "non_calcules": int(self.manifest.get("non_calcules") or 0),
            # The rate is measured on what COULD be covered: otherwise a perfect set
            # caps below 100 % merely because the days return home.
            "taux": (n_couverts / n_exploitables) if n_exploitables else None,
        }

    @property
    def est_complet(self) -> bool:
        """Everything that COULD be covered is.

        The denominator is `deplacements_exploitables`, not the raw expected trips: an
        on-the-spot closure has no itinerary, requiring it would make completeness
        unreachable for any population whose days return home.
        """
        c = self.couverture()
        return (
            bool(c["deplacements_exploitables"])
            and c["deplacements_couverts"] == c["deplacements_exploitables"]
        )

    def inexploitables(self, attendus) -> set[tuple[str, str]]:
        """The trips this set covers but WITHOUT any proposal (R9).

        Derived from the set, hence known when a run opens and identical from one
        run to the next. This is the fix for alert A3: the denominator was
        until then accumulated over the decisions, so that an interrupted run
        did not announce the same number as a complete run on the same pair
        (population, set), and two columns ceased to be comparable.

        A trip ABSENT from the set is not part of it: it is a preparation gap
        (`non_couvert`), another category, which must not vary with this one.
        """
        hors = set()
        for d in attendus:
            ligne = self._index.get((d.person_id, d.activity_id))
            if ligne is not None and not ligne.propositions:
                hors.add((d.person_id, d.activity_id))
        return hors

    def resume(self) -> str:
        c = self.couverture()
        taux = (
            f"{100 * c['taux']:.1f} %".replace(".", ",")
            if c["taux"] is not None
            else "n/a"
        )
        etat = "clos" if self.clos else "EN PRÉPARATION"
        pop = self.population
        lignes = [
            f"Jeu {self.nom!r} — {etat}, empreinte {str(self.empreinte)[:12] if self.empreinte else 'non figée'}…",
            f"  population : {pop.get('nom')} ({'scellée' if pop.get('scellee') else 'non scellée'}, {str(pop.get('sha256'))[:12]}…)",
            f"  jour simulé : {self.jour_simule} · heure de référence : {self.manifest.get('heure_reference')}",
            f"  couverture : {c['deplacements_couverts']} / {c['deplacements_attendus']} déplacements ({taux}) · "
            f"{c['personnes_couvertes']} / {c['personnes_avec_deplacement']} personnes avec déplacement "
            f"(population : {c['personnes_total']})",
        ]
        if c["sans_proposition"]:
            motifs = Counter(s.get("motif") for s in c["sans_proposition"])
            lignes.append(
                f"  INEXPLOITABLES (aucune proposition des moteurs, exclus des attendus d'une expérience) : "
                f"{len(c['sans_proposition'])} — "
                + ", ".join(f"{m} × {n}" for m, n in motifs.most_common())
            )
        if c["non_calcules"]:
            lignes.append(
                f"  NON CALCULÉS : {c['non_calcules']} (préparation interrompue : reprendre)"
            )
        return "\n".join(lignes)


def _signature_tc(propositions: Sequence[Proposition]) -> list[tuple]:
    """What distinguishes one public transport supply from another: lines used, duration to the minute."""
    from urban_mobility_agents.candidats import _primary_mode

    return sorted(
        (
            p.plan.get_code() if p.plan.legs else p.plan.id,
            p.mode,
            (p.plan.duration or 0) // 60,
        )
        for p in propositions
        if _primary_mode(p.plan) == "transit"
    )


async def comparer_offre_jour(
    jeu: Jeu,
    trip_helper,
    jour: str,
    *,
    echantillon: int = 100,
    graine: int = 42,
    concurrence: int = 8,
) -> dict:
    """Decision 18 — is the transit supply of `jour` that of the set's day? MEASURED, never assumed.

    Draws `echantillon` covered trips (declared seed), asks the engines for the proposals
    at the same time on day `jour`, and compares public transport (lines, durations to the minute)
    with the recorded ones. Returns the detail; the caller decides whether to declare equivalence.
    """
    import random as _random
    from datetime import date as _date

    decalage = (
        _date.fromisoformat(jour) - _date.fromisoformat(jeu.jour_simule)
    ).days * 86400
    lignes = [
        l for l in jeu._index.values() if any(_signature_tc(l.vers_propositions()))
    ]
    rng = _random.Random(graine)
    tirees = rng.sample(lignes, min(echantillon, len(lignes))) if lignes else []
    sem = asyncio.Semaphore(max(1, concurrence))
    resultats: list[dict] = []

    async def un(ligne: LigneJeu) -> None:
        async with sem:
            avant = _signature_tc(ligne.vers_propositions())
            try:
                its = await trip_helper.get_itineraries(
                    origin=ligne.origine,
                    destination=ligne.destination,
                    departure_time=ligne.depart_ts + decalage,
                    include_car=True,
                    include_bike=True,
                    arrive_by=False,
                )
            except Exception as e:  # noqa: BLE001
                resultats.append(
                    {"cle": list(ligne.cle), "erreur": f"{type(e).__name__}: {e}"}
                )
                return
            apres = _signature_tc(
                [Proposition(it, SOURCE_ENREGISTREE) for it in its or []]
            )
            resultats.append(
                {
                    "cle": list(ligne.cle),
                    "identique": avant == apres,
                    "avant": avant,
                    "apres": apres,
                }
            )

    await asyncio.gather(*(un(l) for l in tirees))
    identiques = sum(1 for r in resultats if r.get("identique"))
    erreurs = sum(1 for r in resultats if "erreur" in r)
    compares = len(resultats) - erreurs
    return {
        "jeu": jeu.nom,
        "jour_jeu": jeu.jour_simule,
        "jour": jour,
        "echantillon": len(tirees),
        "graine": graine,
        "compares": compares,
        "identiques": identiques,
        "differents": compares - identiques,
        "erreurs": erreurs,
        "part_identique": (identiques / compares) if compares else None,
        "equivalent": bool(compares) and identiques == compares and erreurs == 0,
        "mesure_le": _maintenant_iso(),
        "differences": [r for r in resultats if r.get("identique") is False][:20],
    }


# ── Config files: values, not bytes ──────────────────────────────────────────
#
# A set records the sha256 of the BYTES of each config file. Rewording a comment changes
# those bytes without changing a single routing value: on 2026-09-30, translating the
# comments of the three files into English made all three frozen sets stale, while
# `yaml.safe_load` returned the same data before and after. The registry below says, for
# each committed version of a config file, what VALUES its bytes carry — computed from
# the git history on the host (`make config-empreintes`), because the container has no
# git. A byte mismatch whose values are identical is therefore not a staleness.

FICHIER_REGISTRE_VALEURS = "empreintes_valeurs.yaml"
VERSION_REGISTRE_VALEURS = "valeurs1"


def dossier_config_defaut() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def empreinte_valeurs(texte: str | bytes) -> str:
    """sha256 of the DATA of a YAML document: comments, spacing and key order do not count."""
    donnees = yaml.safe_load(texte)
    canon = json.dumps(
        donnees, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def construire_registre_valeurs(versions: dict[str, Sequence[bytes]]) -> dict:
    """{file: [contents of each version]} → registry {file: {bytes sha256: values sha256}}."""
    fichiers: dict[str, dict[str, str]] = {}
    for nom, contenus in sorted(versions.items()):
        table: dict[str, str] = {}
        for contenu in contenus:
            try:
                table[hashlib.sha256(contenu).hexdigest()] = empreinte_valeurs(contenu)
            except yaml.YAMLError:
                continue  # a historical version that no longer parses proves nothing
        fichiers[nom] = dict(sorted(table.items()))
    return {"version": VERSION_REGISTRE_VALEURS, "fichiers": fichiers}


def versions_git_config(dossier_config: Path | None = None) -> dict[str, list[bytes]]:
    """Every committed version of each config file, plus the working-tree one (host only)."""
    config = Path(dossier_config) if dossier_config else dossier_config_defaut()
    racine = _racine_depot()
    versions: dict[str, list[bytes]] = {}
    for nom in _FICHIERS_CONFIG:
        chemin = config / nom
        relatif = chemin.resolve().relative_to(racine.resolve()).as_posix()
        commits = _git("log", "--format=%H", "--follow", "--", relatif)
        if commits is None:
            raise RuntimeError(
                f"git log failed for {relatif}: run this on the host, in the repository"
            )
        contenus: list[bytes] = []
        for c in commits.split():
            out = subprocess.run(
                ["git", "show", f"{c}:{relatif}"],
                cwd=racine,
                capture_output=True,
                timeout=10,
                check=False,
            )
            if out.returncode == 0:
                contenus.append(out.stdout)
        if chemin.is_file():
            contenus.append(chemin.read_bytes())
        versions[nom] = contenus
    return versions


def lire_registre_valeurs(
    dossier_config: Path | None = None,
) -> dict[str, dict[str, str]] | None:
    """{file: {bytes sha256: values sha256}}, or None if the registry is absent or unreadable."""
    config = Path(dossier_config) if dossier_config else dossier_config_defaut()
    chemin = config / FICHIER_REGISTRE_VALEURS
    try:
        doc = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning(
            f"[jeu] config values registry unreadable ({chemin}: {e}) — a config file whose "
            "bytes changed stays STALE, even if only its comments changed"
        )
        return None
    if not isinstance(doc, dict) or doc.get("version") != VERSION_REGISTRE_VALEURS:
        logger.warning(
            f"[jeu] config values registry {chemin} malformed or of another version "
            f"(expected {VERSION_REGISTRE_VALEURS!r}) — byte changes stay STALE"
        )
        return None
    fichiers = doc.get("fichiers")
    return fichiers if isinstance(fichiers, dict) else None


def _config_equivalentes(
    differentes: list[str], enregistrees: dict, dossier_config: Path
) -> list[str]:
    """The `config/<file>` among `differentes` whose recorded version has the current values."""
    candidates = [c for c in differentes if c.startswith("config/")]
    if not candidates:
        return []
    registre = lire_registre_valeurs(dossier_config)
    if registre is None:
        return []
    config_enregistree = (enregistrees or {}).get("config") or {}
    equivalentes: list[str] = []
    for cle in candidates:
        nom = cle.split("/", 1)[1]
        ancienne = config_enregistree.get(nom)
        valeurs_anciennes = (registre.get(nom) or {}).get(ancienne)
        chemin = dossier_config / nom
        if valeurs_anciennes is None:
            logger.info(
                f"[jeu] {cle}: bytes changed and the set's version ({str(ancienne)[:12]}…) is "
                f"not in {FICHIER_REGISTRE_VALEURS} → stale (`make config-empreintes` if it was committed)"
            )
            continue
        try:
            valeurs_courantes = empreinte_valeurs(chemin.read_bytes())
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"[jeu] {cle}: current file unreadable ({e}) → stale")
            continue
        if valeurs_courantes == valeurs_anciennes:
            logger.info(
                f"[jeu] {cle}: text differs, values identical to the set's version → not stale"
            )
            equivalentes.append(cle)
        else:
            logger.info(
                f"[jeu] {cle}: values changed since the set was prepared → stale"
            )
    return equivalentes


def perime(
    jeu: Jeu, courantes: dict | None = None, dossier_config: Path | None = None
) -> tuple[list[str], list[str]]:
    """J10 — (dependencies changed since preparation, unverifiable dependencies).

    A config file counts as changed when its VALUES changed, not its text: a byte mismatch
    whose values are identical (registry `empreintes_valeurs.yaml`) is dropped. Without the
    registry, or for a version it does not know, the byte comparison stands.
    """
    enregistrees = jeu.manifest.get("dependances") or {}
    differentes, non_verif = comparer_dependances(
        enregistrees,
        courantes if courantes is not None else dependances_courantes(),
    )
    config = Path(dossier_config) if dossier_config else dossier_config_defaut()
    equivalentes = set(_config_equivalentes(differentes, enregistrees, config))
    return [c for c in differentes if c not in equivalentes], non_verif


__all__ = [
    "FICHIER_EQUIVALENCES",
    "FICHIER_MANIFEST",
    "FICHIER_PROPOSITIONS",
    "HEURE_REFERENCE_DEPART",
    "MOTIF_AUCUNE_PROPOSITION",
    "MOTIF_ORIGINE_EGALE_DESTINATION",
    "VERSION_JEU",
    "Deplacement",
    "Jeu",
    "JeuClos",
    "JeuEnPreparation",
    "JeuInvalide",
    "LigneJeu",
    "PropositionEnregistree",
    "comparer_dependances",
    "comparer_offre_jour",
    "dependances_courantes",
    "deplacements_attendus",
    "jour_base_ts",
    "perime",
    "preparer",
]
