"""
Unit tests of the pipeline that builds the annual GTFS feed
(`scripts/data/gtfs_year/`).

Each test covers a decision which, taken the wrong way round, produces a
plausible but wrong feed — the failure mode that motivated this pipeline. The
targeted regressions are named explicitly:

Reliable window (offre.py)
  - the truncated tail of an export is cut
  - an isolated dip (public holiday) is KEPT: cutting at the first dip cut
    six valid days off the March export
  - an export reduced to almost nothing is declared unusable, and its own tail
    does not contaminate the reference used to detect it
  - a tail truncated PER LINE is cut too: the liO export describes the whole
    network up to the service change of 13/12/2026 then only extends
    the lines already filled in — thirteen `.liO 31` lines disappear while
    94% of the network keeps running, and no global threshold sees it
  - but an end of season is NOT a truncation: school lines that
    stop at the end of June in an export that ends in August are on holiday,
    not missing
  - two trips with identical content on the SAME day raise an alarm instead
    of being merged (the day's service would be cut short)

Authority and donors (donneurs.py)
  - a date covered by two exports takes its service from only one
  - the seasonal gap is cyclic: 31/12 is one day from 01/01
  - a holiday looks for another holiday before falling back on a Sunday
  - a period class with no data switches to its fallback chain, at
    low confidence
  - a date declared without service stays empty, and the declaration carries
    over from one year to the next

Assembly (assemblage.py)
  - NO OVER-SERVICE on overlap: the feed serves the trips of the authoritative
    export, never their union (the defect of the feed in service: 13,250
    trips on 08/04/2026 against 12,652 and 12,660 in its sources)
  - two exports describing the same trip share it
  - a `trip_id` recycled for a different trip is forked, not arbitrated
  - a diverging geometry is DUPLICATED, never merged point by point
    (the chimera path of shape 14846)
  - an extrapolated day is the exact copy of its donor
  - `calendar.txt` stays empty and `exception_type` equals 1 — the contract of
    `services/llm-agents/inputs/gtfs/reader.py`
  - referential closure, parent stations included

Validation (validation.py)
  - the service hash ignores identifiers and sees a missing trip
  - V2 unmasks an injected over-service
  - V6 unmasks a chimera path

Windowing (window_feed.py)
  - beyond 64 dates, refusal: it is the limit of GAMA's bit mask
  - the window is referentially closed

No network access: both calendar APIs are replaced by
doubles. All feeds are synthetic and written in a temporary
directory.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from scripts.data.gtfs_year import (  # noqa: E402
    assemblage,
    calendar_fr,
    donneurs,
    gtfs_io,
    offre,
    validation,
    window_feed,
)
from scripts.data.gtfs_year.gtfs_io import Export  # noqa: E402

CONFIG = yaml.safe_load(
    (REPO_ROOT / "scripts" / "data" / "gtfs_year" / "feed_year.yaml").read_text(encoding="utf-8")
)

MUET = lambda *_a, **_k: None  # noqa: E731


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic feed factory
# ──────────────────────────────────────────────────────────────────────────────


def ecrire_feed(
    dossier: Path,
    *,
    lignes: list[str],
    arrets: list[tuple],
    courses: list[dict],
    calendrier: dict[str, list[str]],
    geometries: dict[str, list[tuple]] | None = None,
    agency_id: str = "network:1",
    correspondances: list[tuple[str, str]] | None = None,
    calendrier_hebdo: list[dict] | None = None,
    retraits: list[tuple[str, str]] | None = None,
) -> Path:
    """Writes a minimal but compliant GTFS set.

    `courses`: {id, ligne, service, sens, girouette, geometrie, horaires}
    where `horaires` is a sequence of (stop_id, arrivee, depart, distance).
    `arrets`: (stop_id, lat, lon, parent_station|"").
    `calendrier_hebdo`: rows of `calendar.txt` (liO form), each
    {service_id, monday…sunday, start_date, end_date}.
    `retraits`: (service_id, date) written as `exception_type=2`.
    """
    dossier.mkdir(parents=True, exist_ok=True)

    def table(nom, colonnes, rangs):
        with open(dossier / nom, "w", encoding="utf-8", newline="") as fichier:
            writer = csv.DictWriter(fichier, fieldnames=colonnes, lineterminator="\n")
            writer.writeheader()
            for rang in rangs:
                writer.writerow(rang)

    table("agency.txt", ["agency_id", "agency_name", "agency_timezone"],
          [{"agency_id": agency_id, "agency_name": "Test", "agency_timezone": "Europe/Paris"}])
    table("calendar.txt",
          ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
           "saturday", "sunday", "start_date", "end_date"], calendrier_hebdo or [])
    table("routes.txt", ["route_id", "agency_id", "route_short_name", "route_type"],
          [{"route_id": r, "agency_id": agency_id, "route_short_name": r, "route_type": "3"}
           for r in lignes])
    table("stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
          [{"stop_id": a[0], "stop_name": a[0], "stop_lat": f"{a[1]:.6f}", "stop_lon": f"{a[2]:.6f}",
            "location_type": "0", "parent_station": a[3] if len(a) > 3 else ""}
           for a in arrets])
    table("trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"],
          [{"route_id": c["ligne"], "service_id": c["service"], "trip_id": c["id"],
            "trip_headsign": c.get("girouette", "dest"), "direction_id": c.get("sens", "0"),
            "shape_id": c.get("geometrie", "")}
           for c in courses])
    table("stop_times.txt",
          ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence",
           "pickup_type", "drop_off_type", "stop_headsign", "shape_dist_traveled"],
          [{"trip_id": c["id"], "arrival_time": h[1], "departure_time": h[2], "stop_id": h[0],
            "stop_sequence": str(i + 1), "pickup_type": "0", "drop_off_type": "0",
            "stop_headsign": "", "shape_dist_traveled": f"{h[3]:.1f}"}
           for c in courses for i, h in enumerate(c["horaires"])])
    table("calendar_dates.txt", ["service_id", "date", "exception_type"],
          [{"service_id": s, "date": d, "exception_type": "1"}
           for s, dates in calendrier.items() for d in dates]
          + [{"service_id": s, "date": d, "exception_type": "2"} for s, d in (retraits or [])])
    table("shapes.txt",
          ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"],
          [{"shape_id": sid, "shape_pt_lat": f"{p[0]:.6f}", "shape_pt_lon": f"{p[1]:.6f}",
            "shape_pt_sequence": str(i + 1), "shape_dist_traveled": f"{p[2]:.1f}"}
           for sid, points in (geometries or {}).items() for i, p in enumerate(points)])
    table("transfers.txt", ["from_stop_id", "to_stop_id", "transfer_type"],
          [{"from_stop_id": a, "to_stop_id": b, "transfer_type": "2"}
           for a, b in (correspondances or [])])
    return dossier


def course(id_, *, ligne="L1", service="S", geometrie="SH1", girouette="dest",
           depart="08:00:00", arrivee="08:10:00", arrets=("A", "B")) -> dict:
    return {
        "id": id_, "ligne": ligne, "service": service, "geometrie": geometrie,
        "girouette": girouette,
        "horaires": [(arrets[0], depart, depart, 0.0), (arrets[1], arrivee, arrivee, 1000.0)],
    }


GEOM = {"SH1": [(43.600000, 1.440000, 0.0), (43.610000, 1.450000, 1000.0)]}
ARRETS = [("A", 43.600000, 1.440000), ("B", 43.610000, 1.450000)]


def export_de(dossier: Path) -> Export:
    return Export(chemin=dossier, etiquette=dossier.name, empreinte="md5-" + dossier.name)


class BaseTemporaire(unittest.TestCase):
    def setUp(self) -> None:
        self.racine = Path(tempfile.mkdtemp(prefix="gtfs_year_"))
        self.addCleanup(shutil.rmtree, self.racine, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Reliable window
# ──────────────────────────────────────────────────────────────────────────────


class FabriqueOffreParDate(BaseTemporaire):
    """Base of the two families of reliable-window tests, with no test of its own.

    Both forms of truncation — the global collapse and the cliff of lines —
    are described with the same tool: a profile "n active lines on this date".
    The base holds no test, so as not to replay those of one by inheriting
    them in the other.
    """

    def _export(self, offre_par_date: dict[str, int]) -> offre.IndexExport:
        """An export where each date activates `n` distinct lines."""
        lignes = [f"L{i}" for i in range(1, 1 + max(offre_par_date.values()))]
        courses, calendrier = [], {}
        for date, nb in offre_par_date.items():
            service = f"SVC_{date}"
            calendrier[service] = [date]
            for i in range(nb):
                courses.append(course(f"T_{date}_{i}", ligne=lignes[i], service=service,
                                      depart=f"{6 + i // 6:02d}:{(i % 6) * 10:02d}:00"))
        dossier = ecrire_feed(
            self.racine / f"exp_{len(list(self.racine.iterdir()))}",
            lignes=lignes, arrets=ARRETS, courses=courses,
            calendrier=calendrier, geometries=GEOM,
        )
        return offre.indexer(export_de(dossier), MUET)

    @staticmethod
    def _semaines(depart: str, profil: list[int]) -> dict[str, int]:
        premier = calendar_fr.to_date(depart)
        return {
            (premier + dt.timedelta(days=i)).strftime("%Y%m%d"): nb
            for i, nb in enumerate(profil)
        }


class TestFenetreFiable(FabriqueOffreParDate):
    """The truncated tail must go, a holiday dip must stay."""

    def test_queue_tronquee_coupee(self):
        # 21 full days (the reference profile), then 7 days at 2 lines.
        profil = [20] * 21 + [2] * 7
        index = self._export(self._semaines("20260302", profil))
        retenues = offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)
        self.assertEqual(len(retenues), 21)
        self.assertEqual(retenues[-1], "20260322")

    def test_creux_isole_conserve(self):
        # A holiday on the 15th day: collapse followed by a return to normal.
        profil = [20] * 14 + [2] + [20] * 6 + [2] * 7
        profil_date = self._semaines("20260302", profil)
        index = self._export(profil_date)
        retenues = offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)
        creux = (calendar_fr.to_date("20260302") + dt.timedelta(days=14)).strftime("%Y%m%d")
        self.assertIn(creux, retenues, "un creux isolé n'est pas une troncature")
        self.assertEqual(len(retenues), 21)

    def test_export_sans_queue_entierement_retenu(self):
        index = self._export(self._semaines("20260302", [20] * 28))
        self.assertEqual(len(offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)), 28)

    def test_export_trop_court_declare_inutilisable(self):
        # The reference profile is taken from the start: here everything is low, so
        # nothing is "below the threshold" and the export is kept as is. But an
        # export with only 3 days left after the cut is discarded.
        # An export delivered late: three full days, then 25 days of
        # tail. Not enough reliable material remains — and the reference MUST
        # NOT be contaminated by the tail, otherwise nothing would be cut.
        profil = [20] * 3 + [2] * 25
        index = self._export(self._semaines("20260302", profil))
        self.assertEqual(offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET), [])


class TestFalaiseDeLignes(FabriqueOffreParDate):
    """Truncation PER LINE: the global service holds, lines disappear.

    Measured on the liO export of 2026-09-04 (2026-08-01 → 2027-08-31): thirteen
    `.liO 31` lines — ten of which serve the study scope, all feeders to a
    station — stop on 11 or 12/12/2026 at the SNCF service change and never
    resume over the following thirty-seven weeks. Their `calendar.txt`
    stops there (services from 06 to 12/12) while that of the neighbouring lines runs
    until 31/08/2027. Meanwhile the global service holds: 4,303 → 4,165
    trips on Monday, 260 active lines out of a maximum of 276. No global threshold
    could see it, and the donor of the simulated day served 21% too few
    trips.

    What separates the cliff from an end of season is not the shape of the loss but
    what the export does AFTERWARDS — hence the two twin tests below.
    """

    def test_falaise_coupee_quand_l_export_couvre_encore_des_mois(self):
        # 30 days at 20 lines, then 120 days at 14: the global service stays at 70%
        # of the maximum (the global thresholds, at 50%, cut nothing) but six lines
        # have disappeared for good while four months of coverage remain.
        profil = [20] * 30 + [14] * 120
        index = self._export(self._semaines("20260801", profil))
        retenues = offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)
        self.assertEqual(len(retenues), 30, "la falaise de lignes n'a pas été coupée")
        self.assertEqual(retenues[-1], "20260830")

    def test_fin_de_saison_conservee_quand_l_export_s_acheve_peu_apres(self):
        # Same cliff, but the export only covers 30 more days after it: school
        # lines that stop at the end of June in an export that ends in August
        # are not a truncation, they are on holiday.
        profil = [20] * 30 + [14] * 30
        index = self._export(self._semaines("20260801", profil))
        retenues = offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)
        self.assertEqual(len(retenues), 60, "une fin de saison a été prise pour une troncature")

    def test_renouvellement_saisonnier_sous_le_plancher_conserve(self):
        # Three summer lines that stop: below the absolute floor (5 lines).
        # liO loses at most four per week over the whole observed period.
        profil = [20] * 30 + [17] * 120
        index = self._export(self._semaines("20260801", profil))
        retenues = offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)
        self.assertEqual(len(retenues), 150, "un renouvellement saisonnier a été coupé")

    def test_la_fin_de_l_export_n_est_pas_une_falaise(self):
        # All lines stop on the last day, by construction: the
        # check must ignore this date, otherwise no export would ever be
        # usable.
        index = self._export(self._semaines("20260801", [20] * 150))
        self.assertEqual(len(offre.fenetre_fiable(index, CONFIG["fiabilite"], MUET)), 150)

    def test_la_falaise_est_journalisee_et_alarmee(self):
        messages = []
        profil = [20] * 30 + [14] * 120
        index = self._export(self._semaines("20260801", profil))
        offre.fenetre_fiable(index, CONFIG["fiabilite"], lambda m, *a, **k: messages.append(str(m)))
        joints = "\n".join(messages)
        self.assertIn("falaise de lignes", joints)
        self.assertIn("[ALARME]", joints)

    def test_le_controle_se_desactive_par_configuration(self):
        # A zero threshold makes the check inert: the way out if a legitimate
        # export were wrongly cut.
        config = {**CONFIG["fiabilite"], "falaise_lignes_part_min": 0}
        index = self._export(self._semaines("20260801", [20] * 30 + [14] * 120))
        self.assertEqual(len(offre.fenetre_fiable(index, config, MUET)), 150)


class TestCalendrierHebdomadaire(BaseTemporaire):
    """Weekly `calendar.txt` — the form published by liO.

    Tisséo and the TER use only explicit dates; liO declares weekly
    services that `calendar_dates.txt` corrects in both directions.
    Each test covers a reading which, taken the wrong way round, runs
    coaches on a day the operator says they do not run — or the reverse.
    """

    SEMAINE = {"service_id": "S", "monday": "1", "tuesday": "1", "wednesday": "1",
               "thursday": "1", "friday": "1", "saturday": "0", "sunday": "0",
               "start_date": "20260316", "end_date": "20260322"}

    def _indexer(self, calendrier=None, **extra):
        dossier = ecrire_feed(
            self.racine / f"hebdo_{len(list(self.racine.iterdir()))}", lignes=["L1"],
            arrets=ARRETS, courses=[course("T1")], calendrier=calendrier or {},
            geometries=GEOM, **extra,
        )
        return offre.indexer(export_de(dossier), MUET)

    def test_semaine_depliee_en_dates(self):
        index = self._indexer(calendrier_hebdo=[dict(self.SEMAINE)])
        # From Monday 16 to Friday 20 March: five days, Saturday and Sunday excluded.
        self.assertEqual(index.dates, ["20260316", "20260317", "20260318", "20260319", "20260320"])
        self.assertEqual(index.nb_trips("20260318"), 1)

    def test_exception_type_2_retire_une_date(self):
        index = self._indexer(calendrier_hebdo=[dict(self.SEMAINE)], retraits=[("S", "20260318")])
        self.assertNotIn("20260318", index.dates)
        self.assertEqual(len(index.dates), 4)

    def test_exception_type_1_ajoute_un_jour_hors_semaine(self):
        index = self._indexer(calendrier_hebdo=[dict(self.SEMAINE)], calendrier={"S": ["20260321"]})
        self.assertIn("20260321", index.dates)   # a Saturday added by hand
        self.assertEqual(index.nb_trips("20260321"), 1)

    def test_service_borne_a_l_envers_ignore(self):
        borne = dict(self.SEMAINE, start_date="20260322", end_date="20260316")
        index = self._indexer(calendrier_hebdo=[borne])
        self.assertEqual(index.dates, [])

    def test_dates_explicites_seules_inchangees(self):
        """Without `calendar.txt`, the index stays that of Tisséo and the TER."""
        index = self._indexer(calendrier={"S": ["20260316", "20260317"]})
        self.assertEqual(index.dates, ["20260316", "20260317"])


# ──────────────────────────────────────────────────────────────────────────────
# Calendar
# ──────────────────────────────────────────────────────────────────────────────


class TestCalendrier(unittest.TestCase):
    PERIODES = [
        calendar_fr.Periode("vac_printemps", dt.date(2026, 4, 18), dt.date(2026, 5, 3)),
        calendar_fr.Periode("ete_juillet", dt.date(2026, 7, 4), dt.date(2026, 7, 31)),
    ]
    FERIES = {"20260501": "1er mai", "20260406": "Lundi de Pâques"}

    def test_signature_hors_vacances(self):
        sig = calendar_fr.signature("20260316", self.PERIODES, self.FERIES)
        self.assertEqual((sig.type_jour, sig.periode), ("lun", "scolaire"))

    def test_signature_en_vacances(self):
        sig = calendar_fr.signature("20260420", self.PERIODES, self.FERIES)
        self.assertEqual((sig.type_jour, sig.periode), ("lun", "vac_printemps"))

    def test_ferie_a_son_propre_type_de_jour(self):
        # A holiday is not a Monday, nor is it a Sunday: 14/07
        # serves 5,674 trips against 4,683 to 5,054 on July Sundays.
        sig = calendar_fr.signature("20260406", self.PERIODES, self.FERIES)
        self.assertEqual(sig.type_jour, "ferie")

    def test_decalage_reporte_le_debut_de_periode(self):
        sans = calendar_fr.signature("20260418", self.PERIODES, self.FERIES)
        avec = calendar_fr.signature("20260418", self.PERIODES, self.FERIES, {"vac_printemps": 1})
        self.assertEqual(sans.periode, "vac_printemps")
        self.assertEqual(avec.periode, "scolaire")

    def test_date_locale_corrige_le_decalage_utc(self):
        # The API publishes 22:00 UTC for holidays that begin the next day in
        # Paris; ignoring it would shift all bounds by one day.
        self.assertEqual(calendar_fr._date_locale("2026-04-17T22:00:00+00:00"), dt.date(2026, 4, 18))

    def test_bornes_apprises_sur_les_donnees(self):
        # The opening Saturday still runs as a school Saturday: the shift
        # that minimises the dispersion must be +1.
        offre_reelle = {
            "20260404": 8194, "20260411": 8194, "20260418": 8194,  # school Saturdays
            "20260425": 8470, "20260502": 8470,                    # holiday Saturdays
        }
        decalages = calendar_fr.ajuster_bornes(
            self.PERIODES, self.FERIES, offre_reelle,
            CONFIG["calendrier"]["decalages_debut_testes"], MUET,
        )
        self.assertEqual(decalages.get("vac_printemps"), 1)

    def test_ecart_saisonnier_est_cyclique(self):
        self.assertEqual(donneurs.ecart_saisonnier("20261231", "20260101"), 1)
        self.assertEqual(donneurs.ecart_saisonnier("20260101", "20260311"), 69)
        self.assertEqual(
            donneurs.ecart_saisonnier("20270316", "20260316"),
            0,
            "la même date d'une autre année est à distance nulle",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Authority and donors
# ──────────────────────────────────────────────────────────────────────────────


class TestAutoriteEtDonneurs(BaseTemporaire):
    def _index(self, nom: str, calendrier: dict[str, list[str]], courses: list[dict]):
        dossier = ecrire_feed(
            self.racine / nom, lignes=["L1"], arrets=ARRETS, courses=courses,
            calendrier=calendrier, geometries=GEOM,
        )
        return offre.indexer(export_de(dossier), MUET)

    def test_une_seule_source_par_date(self):
        ancien = self._index("ancien", {"S": ["20260316", "20260317"]},
                             [course("T1", service="S"), course("T2", service="S")])
        recent = self._index("recent", {"S": ["20260317", "20260318"]},
                             [course("T3", service="S")])
        choix = donneurs.autorite(
            {"ancien": ancien, "recent": recent},
            {"ancien": ["20260316", "20260317"], "recent": ["20260317", "20260318"]},
            MUET,
        )
        self.assertEqual(choix["20260316"], "ancien")
        self.assertEqual(choix["20260318"], "recent")
        self.assertEqual(choix["20260317"], "recent", "le plus récemment publié fait autorité")

    def _plan(self, dates_reelles: list[str], annee=2026, dates_sans_service=None,
              periodes=None, feries=None):
        index = self._index(
            "src", {f"S{d}": [d] for d in dates_reelles},
            [course(f"T{d}", service=f"S{d}") for d in dates_reelles],
        )
        return donneurs.plan_annee(
            annee=annee,
            dates_annee=calendar_fr.dates_annee(annee),
            source_par_date={d: "src" for d in dates_reelles},
            index_par_export={"src": index},
            periodes=periodes if periodes is not None else TestCalendrier.PERIODES,
            feries=feries if feries is not None else TestCalendrier.FERIES,
            decalages={},
            config_extrap=CONFIG["extrapolation"],
            dates_sans_service=dates_sans_service or set(),
            journal=MUET,
        )

    def test_date_reelle_en_confiance_haute(self):
        plan = self._plan(["20260316"])
        self.assertEqual(plan["20260316"].mode, donneurs.REEL)
        self.assertEqual(plan["20260316"].confiance, donneurs.HAUTE)
        self.assertEqual(plan["20260316"].date_source, "20260316")

    def test_donneur_de_signature_exacte_le_plus_proche(self):
        # Three real school Mondays; 09/03 must take the closest.
        plan = self._plan(["20260316", "20260323", "20260921"])
        provenance = plan["20260309"]
        self.assertEqual(provenance.mode, donneurs.EXTRAPOLE)
        self.assertEqual(provenance.date_source, "20260316")
        self.assertEqual(provenance.motif, "signature exacte")

    def test_ferie_cherche_un_ferie_avant_un_dimanche(self):
        # 06/04 is a holiday, 05/04 is a Sunday: 01/05, a holiday, must take
        # the holiday and not the Sunday, although they are one day apart.
        plan = self._plan(["20260405", "20260406"],
                          periodes=[], feries={"20260406": "Pâques", "20260501": "1er mai"})
        self.assertEqual(plan["20260501"].date_source, "20260406")

    def test_repli_de_periode_en_confiance_basse(self):
        # A single real Monday, in spring holidays. A school-period Monday
        # has no exact donor: `scolaire` has no declared fallback,
        # so the day stays without service rather than invented.
        plan = self._plan(["20260420"])
        self.assertEqual(plan["20260420"].mode, donneurs.REEL)
        scolaire = plan["20260316"]
        self.assertEqual(scolaire.mode, donneurs.SANS_SERVICE)
        self.assertEqual(scolaire.confiance, donneurs.BASSE)

    def test_repli_inverse_vers_les_vacances(self):
        # Symmetric: a winter-holiday Monday falls back on the spring
        # holidays, at low confidence and saying so.
        periodes = TestCalendrier.PERIODES + [
            calendar_fr.Periode("vac_hiver", dt.date(2026, 2, 21), dt.date(2026, 3, 8))
        ]
        plan = self._plan(["20260420"], periodes=periodes)
        hiver = plan["20260223"]
        self.assertEqual(hiver.mode, donneurs.EXTRAPOLE)
        self.assertEqual(hiver.date_source, "20260420")
        self.assertEqual(hiver.confiance, donneurs.BASSE)
        self.assertIn("repli", hiver.motif)

    def test_date_sans_service_reste_vide(self):
        plan = self._plan(["20260420", "20260427"], dates_sans_service={"20260501"})
        provenance = plan["20260501"]
        self.assertEqual(provenance.mode, donneurs.SANS_SERVICE)
        self.assertEqual(provenance.date_source, "")
        self.assertEqual(provenance.confiance, donneurs.HAUTE)

    def test_annee_entiere_couverte(self):
        plan = self._plan(["20260316"])
        self.assertEqual(len(plan), 365)
        self.assertEqual(sorted(plan)[0], "20260101")
        self.assertEqual(sorted(plan)[-1], "20261231")

    def test_declaration_sans_service_reconduite(self):
        from scripts.data.gtfs_year.build_year_feed import dates_sans_service

        parametres = {"dates_sans_service_confirme": ["20260501"]}
        self.assertEqual(dates_sans_service(parametres, 2026, MUET), {"20260501"})
        self.assertEqual(dates_sans_service(parametres, 2027, MUET), {"20270501"})


# ──────────────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────────────


class TestAssemblage(BaseTemporaire):
    IDENTITE = {"feed_id": "test", "publisher": "Test", "url": "https://x", "version": "v1"}

    def _index(self, nom, calendrier, courses, geometries=None, arrets=None,
               correspondances=None):
        dossier = ecrire_feed(
            self.racine / nom, lignes=["L1", "L2"], arrets=arrets or ARRETS,
            courses=courses, calendrier=calendrier,
            geometries=geometries if geometries is not None else GEOM,
            correspondances=correspondances,
        )
        return offre.indexer(export_de(dossier), MUET)

    def _construire(self, plan, index_par_export):
        sortie = self.racine / "sortie"
        stats = assemblage.construire(sortie, plan, index_par_export, CONFIG, self.IDENTITE, MUET)
        return sortie, stats, offre.indexer(export_de(sortie), MUET)

    @staticmethod
    def _reel(date, export, **extra):
        return donneurs.Provenance(
            date=date, signature="lun/scolaire", mode=donneurs.REEL,
            confiance=donneurs.HAUTE, export=export, date_source=date, **extra,
        )

    @staticmethod
    def _extrapole(date, export, source):
        return donneurs.Provenance(
            date=date, signature="lun/scolaire", mode=donneurs.EXTRAPOLE,
            confiance=donneurs.MOYENNE, export=export, date_source=source,
        )

    def test_pas_de_suroffre_sur_recouvrement(self):
        """The central defect: the union of two exports manufactures service.

        On the feed in service, 08/04/2026 serves 13,250 trips where its two
        sources give 12,652 and 12,660.
        """
        # Each trip has its own timetable: two distinct trips of the
        # same day must not be merged by their content.
        ancien = self._index("ancien", {"S": ["20260316"]},
                             [course("T1", service="S", depart="08:00:00"),
                              course("T2", service="S", depart="08:30:00")])
        recent = self._index("recent", {"S": ["20260316"]},
                             [course("T2", service="S", depart="08:30:00"),
                              course("T3", service="S", depart="09:00:00")])
        plan = {"20260316": self._reel("20260316", "recent")}
        _, _, sortie = self._construire(plan, {"ancien": ancien, "recent": recent})
        self.assertEqual(
            sortie.nb_trips("20260316"), 2,
            "le feed doit servir les 2 courses de l'export autoritaire, pas les 3 de l'union",
        )
        self.assertEqual(set(sortie.trips_par_date["20260316"]), {"T2", "T3"})

    def test_meme_contenu_dans_deux_exports_fusionne(self):
        a = self._index("a", {"S": ["20260316"]}, [course("T1", service="S")])
        b = self._index("b", {"S": ["20260323"]}, [course("T1", service="S")])
        plan = {
            "20260316": self._reel("20260316", "a"),
            "20260323": self._reel("20260323", "b"),
        }
        _, stats, sortie = self._construire(plan, {"a": a, "b": b})
        self.assertEqual(stats.trips_ecrits, 1)
        self.assertEqual(stats.trips_fusionnes, 1)
        self.assertEqual(stats.trips_forkes, 0)
        self.assertEqual(sortie.nb_trips("20260316"), 1)
        self.assertEqual(sortie.nb_trips("20260323"), 1)

    def test_doublon_de_contenu_du_meme_export_preserve(self):
        """Two identical trips of the same export are two trips.

        liO publishes 45 of them on Monday 14/09/2026: two mission numbers for the
        same timetable on the same line. Merging them would remove a trip from
        the day's service — and V2, which compares the produced service with that of
        the source, would refuse it.
        """
        index = self._index(
            "doublons", {"S": ["20260316"]},
            [course("T1", service="S", depart="08:00:00"),
             course("T1bis", service="S", depart="08:00:00")],
        )
        plan = {"20260316": self._reel("20260316", "doublons")}
        _, stats, sortie = self._construire(plan, {"doublons": index})
        self.assertEqual(stats.doublons_de_contenu, 1)
        self.assertEqual(stats.trips_fusionnes, 0)
        self.assertEqual(stats.collisions_meme_jour, 0)
        self.assertEqual(sortie.nb_trips("20260316"), 2)

    def test_jours_disjoints_restent_fusionnes(self):
        """The same content on disjoint days stays ONE trip.

        That is the feed's compression: an operator splits its calendar into
        periods and republishes the same trip in each. Telling them apart
        would inflate the feed without adding anything to the service — measured on Tisséo
        2026: 29.4 MB instead of 22.1 MB.
        """
        index = self._index(
            "periodes", {"S1": ["20260316"], "S2": ["20260323"]},
            [course("T1", service="S1", depart="08:00:00"),
             course("T2", service="S2", depart="08:00:00")],
        )
        plan = {"20260316": self._reel("20260316", "periodes"),
                "20260323": self._reel("20260323", "periodes")}
        _, stats, sortie = self._construire(plan, {"periodes": index})
        self.assertEqual(stats.doublons_de_contenu, 0)
        self.assertEqual(stats.trips_ecrits, 1)
        self.assertEqual(sortie.nb_trips("20260316"), 1)
        self.assertEqual(sortie.nb_trips("20260323"), 1)

    def test_doublon_de_contenu_preserve_l_empreinte_de_la_journee(self):
        """Check V2 passes: the produced service equals that of the source."""
        index = self._index(
            "doublons_v2", {"S": ["20260316"]},
            [course("T1", service="S", depart="08:00:00"),
             course("T1bis", service="S", depart="08:00:00")],
        )
        plan = {"20260316": self._reel("20260316", "doublons_v2")}
        _, _, sortie = self._construire(plan, {"doublons_v2": index})
        produite = validation.empreintes_par_date(
            sortie.export, sortie.trips_par_date, {"20260316"}, CONFIG, MUET
        )
        source = validation.empreintes_par_date(
            index.export, index.trips_par_date, {"20260316"}, CONFIG, MUET
        )
        self.assertEqual(produite, source)

    def test_identifiant_recycle_est_forke(self):
        """A `trip_id` reused for another trip must not be arbitrated.

        The `trip_id` is not stable over the year: the Jaccard index between a
        Tuesday in March and a Tuesday in September is 0.00.
        """
        a = self._index("a", {"S": ["20260316"]}, [course("T1", service="S", depart="08:00:00")])
        b = self._index("b", {"S": ["20260921"]}, [course("T1", service="S", depart="09:00:00")])
        plan = {
            "20260316": self._reel("20260316", "a"),
            "20260921": self._reel("20260921", "b"),
        }
        _, stats, sortie = self._construire(plan, {"a": a, "b": b})
        self.assertEqual(stats.trips_forkes, 1)
        self.assertEqual(stats.trips_ecrits, 2)
        self.assertEqual(sortie.trips_par_date["20260316"], ["T1"])
        self.assertEqual(sortie.trips_par_date["20260921"], ["T1__b"])

    def test_geometrie_divergente_dupliquee_jamais_fusionnee(self):
        """The chimera path: deduplicating on (shape_id, shape_pt_sequence)
        interleaves two geometries and breaks the link with the stop distances.
        """
        geom_a = {"SH1": [(43.60, 1.44, 0.0), (43.61, 1.45, 1000.0)]}
        geom_b = {"SH1": [(43.60, 1.44, 0.0), (43.62, 1.46, 1500.0), (43.63, 1.47, 2000.0)]}
        a = self._index("a", {"S": ["20260316"]}, [course("T1", service="S")], geometries=geom_a)
        b = self._index("b", {"S": ["20260921"]}, [course("T9", service="S")], geometries=geom_b)
        plan = {
            "20260316": self._reel("20260316", "a"),
            "20260921": self._reel("20260921", "b"),
        }
        sortie, stats, _ = self._construire(plan, {"a": a, "b": b})
        self.assertEqual(stats.shapes_dupliquees, 1)

        points = {}
        for ligne in gtfs_io.lire(export_de(sortie), "shapes.txt"):
            points.setdefault(ligne["shape_id"], []).append(ligne)
        self.assertEqual(sorted(points), ["SH1", "SH1__b"])
        self.assertEqual(len(points["SH1"]), 2)
        self.assertEqual(len(points["SH1__b"]), 3)
        # Each geometry keeps a complete sequence restarting at 1: that is
        # exactly what the interleaving destroyed.
        for identifiant, rangs in points.items():
            sequences = sorted(int(r["shape_pt_sequence"]) for r in rangs)
            self.assertEqual(sequences, list(range(1, len(rangs) + 1)), identifiant)

        par_trip = {l["trip_id"]: l["shape_id"] for l in gtfs_io.lire(export_de(sortie), "trips.txt")}
        self.assertEqual(par_trip["T1"], "SH1")
        self.assertEqual(par_trip["T9"], "SH1__b")

    def test_journee_extrapolee_est_une_copie_exacte(self):
        source = self._index("src", {"S": ["20260316"]},
                             [course("T1", service="S"), course("T2", service="S")])
        plan = {
            "20260316": self._reel("20260316", "src"),
            "20261214": self._extrapole("20261214", "src", "20260316"),
        }
        _, _, sortie = self._construire(plan, {"src": source})
        self.assertEqual(
            sortie.trips_par_date["20261214"], sortie.trips_par_date["20260316"]
        )

    def test_calendrier_regroupe_par_ensemble_de_dates(self):
        # T1 and T2 run on the same days → a single service; T3 on another day.
        source = self._index(
            "src", {"SA": ["20260316", "20260317"], "SB": ["20260318"]},
            [course("T1", service="SA", depart="08:00:00"),
             course("T2", service="SA", depart="08:30:00"),
             course("T3", service="SB", depart="09:00:00")],
        )
        plan = {d: self._reel(d, "src") for d in ("20260316", "20260317", "20260318")}
        sortie, stats, _ = self._construire(plan, {"src": source})
        self.assertEqual(stats.services, 2)
        # 2 dates for the service of T1/T2, 1 for that of T3.
        self.assertEqual(stats.lignes_calendrier, 3)
        services = {}
        for ligne in gtfs_io.lire(export_de(sortie), "trips.txt"):
            services[ligne["trip_id"]] = ligne["service_id"]
        self.assertEqual(services["T1"], services["T2"])
        self.assertNotEqual(services["T1"], services["T3"])

    def test_contrat_du_lecteur_du_depot(self):
        """Empty `calendar.txt` and `exception_type=1`: the two asserts of
        `services/llm-agents/inputs/gtfs/reader.py`."""
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")])
        sortie, _, _ = self._construire({"20260316": self._reel("20260316", "src")}, {"src": source})
        self.assertEqual(list(gtfs_io.lire(export_de(sortie), "calendar.txt")), [])
        exceptions = {l["exception_type"] for l in gtfs_io.lire(export_de(sortie), "calendar_dates.txt")}
        self.assertEqual(exceptions, {"1"})

    def test_fermeture_referentielle_et_station_parente(self):
        arrets = [("A", 43.60, 1.44, "GARE"), ("B", 43.61, 1.45, ""), ("GARE", 43.60, 1.44, ""),
                  ("Z", 43.70, 1.50, "")]
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")],
                             arrets=arrets, correspondances=[("A", "B"), ("A", "Z")])
        sortie, stats, _ = self._construire({"20260316": self._reel("20260316", "src")}, {"src": source})
        self.assertEqual(sum(stats.orphelins.values()), 0)
        arrets_sortie = {l["stop_id"] for l in gtfs_io.lire(export_de(sortie), "stops.txt")}
        self.assertEqual(arrets_sortie, {"A", "B", "GARE"}, "la station parente suit ses quais")
        transferts = {(l["from_stop_id"], l["to_stop_id"])
                      for l in gtfs_io.lire(export_de(sortie), "transfers.txt")}
        self.assertEqual(transferts, {("A", "B")}, "la correspondance vers un arrêt écarté tombe")

    def test_arret_deplace_leve_une_alarme(self):
        a = self._index("a", {"S": ["20260316"]}, [course("T1", service="S")])
        b = self._index(
            "b", {"S": ["20260323"]}, [course("T2", service="S")],
            arrets=[("A", 43.601000, 1.440000), ("B", 43.610000, 1.450000)],  # ~111 m
        )
        plan = {
            "20260316": self._reel("20260316", "a"),
            "20260323": self._reel("20260323", "b"),
        }
        _, stats, _ = self._construire(plan, {"a": a, "b": b})
        self.assertTrue(stats.arrets_deplaces)
        identifiants = {s for s, _ in stats.arrets_deplaces}
        self.assertIn("A", identifiants)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


class TestValidation(BaseTemporaire):
    IDENTITE = TestAssemblage.IDENTITE

    def _index(self, nom, calendrier, courses, geometries=None):
        dossier = ecrire_feed(
            self.racine / nom, lignes=["L1"], arrets=ARRETS, courses=courses,
            calendrier=calendrier, geometries=geometries if geometries is not None else GEOM,
        )
        return offre.indexer(export_de(dossier), MUET)

    def test_empreinte_ignore_les_identifiants(self):
        """Two feeds serving the same service under different identifiers
        must have the same hash — that is what makes V2 usable
        although the `service_id`s are rewritten by construction."""
        a = self._index("a", {"SX": ["20260316"]}, [course("T1", service="SX")])
        b = self._index("b", {"SY": ["20260316"]}, [course("T999", service="SY")])
        empreinte_a = validation.empreintes_par_date(
            a.export, a.trips_par_date, {"20260316"}, CONFIG, MUET)
        empreinte_b = validation.empreintes_par_date(
            b.export, b.trips_par_date, {"20260316"}, CONFIG, MUET)
        self.assertEqual(empreinte_a["20260316"], empreinte_b["20260316"])

    def test_empreinte_voit_une_course_manquante(self):
        a = self._index("a", {"S": ["20260316"]},
                        [course("T1", service="S"), course("T2", service="S", depart="09:00:00")])
        b = self._index("b", {"S": ["20260316"]}, [course("T1", service="S")])
        self.assertNotEqual(
            validation.empreintes_par_date(a.export, a.trips_par_date, {"20260316"}, CONFIG, MUET),
            validation.empreintes_par_date(b.export, b.trips_par_date, {"20260316"}, CONFIG, MUET),
        )

    def test_v2_demasque_une_suroffre_injectee(self):
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")])
        plan = {"20260316": TestAssemblage._reel("20260316", "src")}
        sortie = self.racine / "sortie"
        assemblage.construire(sortie, plan, {"src": source}, CONFIG, self.IDENTITE, MUET)

        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        self.assertEqual([v for v in violations if v.gravite == validation.BLOQUANT], [])

        # We inject an extra trip on that day, as a merge by union of
        # two exports would.
        service = next(iter(gtfs_io.lire(export_de(sortie), "calendar_dates.txt")))["service_id"]
        with open(sortie / "trips.txt", "a", encoding="utf-8") as fichier:
            fichier.write(f"L1,{service},TDOUBLON,dest,0,SH1\n")
        with open(sortie / "stop_times.txt", "a", encoding="utf-8") as fichier:
            fichier.write("TDOUBLON,1,08:00:00,08:00:00,0,0,,0.0,A\n")

        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        codes = {v.code for v in violations if v.gravite == validation.BLOQUANT}
        self.assertIn("V2", codes, "une sur-offre doit être bloquante")

    def test_v6_demasque_un_trace_chimere(self):
        """A stop distance beyond the path length is the signature
        of an interleaved geometry — the defect of shape 14846."""
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")])
        plan = {"20260316": TestAssemblage._reel("20260316", "src")}
        sortie = self.racine / "sortie"
        assemblage.construire(sortie, plan, {"src": source}, CONFIG, self.IDENTITE, MUET)

        # We shorten the geometry without touching the timetable: the last stop
        # is now beyond the end of the path.
        lignes = list(gtfs_io.lire(export_de(sortie), "shapes.txt"))
        gtfs_io.ecrire_table(
            sortie / "shapes.txt",
            ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence", "shape_dist_traveled"],
            [{**l, "shape_dist_traveled": "10.0"} for l in lignes],
        )
        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        self.assertIn("V6", {v.code for v in violations if v.gravite == validation.BLOQUANT})

    def test_v6_defaut_deja_publie_par_la_source_nest_pas_bloquant(self):
        """A too-long `shape_dist_traveled` at the operator is not a
        chimera path: the build copied it, it did not make it. liO
        publishes 29 out of 7,715 trips; blocking on that would amount to requiring the
        pipeline to repair the source."""
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")],
                             geometries={"SH1": [(43.60, 1.44, 0.0), (43.61, 1.45, 10.0)]})
        plan = {"20260316": TestAssemblage._reel("20260316", "src")}
        sortie = self.racine / "sortie"
        assemblage.construire(sortie, plan, {"src": source}, CONFIG, self.IDENTITE, MUET)
        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        codes_bloquants = {v.code for v in violations if v.gravite == validation.BLOQUANT}
        self.assertNotIn("V6", codes_bloquants)
        self.assertIn("V6", {v.code for v in violations if v.gravite == validation.ALARME})

    def test_v7_demasque_une_copie_infidele(self):
        """An extrapolated day must serve exactly the service of its donor."""
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")])
        plan = {"20260317": TestAssemblage._extrapole("20260317", "src", "20260316")}
        sortie = self.racine / "sortie"
        assemblage.construire(sortie, plan, {"src": source}, CONFIG, self.IDENTITE, MUET)
        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        self.assertNotIn("V7", {v.code for v in violations if v.gravite == validation.BLOQUANT})

        # One more trip on the copied day: the copy is no longer verbatim.
        service = next(iter(gtfs_io.lire(export_de(sortie), "calendar_dates.txt")))["service_id"]
        with open(sortie / "trips.txt", "a", encoding="utf-8") as fichier:
            fichier.write(f"L1,{service},TDOUBLON,dest,0,SH1\n")
        with open(sortie / "stop_times.txt", "a", encoding="utf-8") as fichier:
            fichier.write("TDOUBLON,1,08:00:00,08:00:00,0,0,,0.0,A\n")
        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        self.assertIn("V7", {v.code for v in violations if v.gravite == validation.BLOQUANT})

    def test_v8_signale_une_journee_sans_offre(self):
        source = self._index("src", {"S": ["20260316"]}, [course("T1", service="S")])
        plan = {
            "20260316": TestAssemblage._reel("20260316", "src"),
            # A planned date about which the export says nothing: the feed will
            # serve nothing that day.
            "20260317": TestAssemblage._extrapole("20260317", "src", "20260401"),
        }
        sortie = self.racine / "sortie"
        assemblage.construire(sortie, plan, {"src": source}, CONFIG, self.IDENTITE, MUET)
        violations, _, _ = validation.controler(sortie, plan, {"src": source}, CONFIG, MUET)
        self.assertIn("V8", {v.code for v in violations})


# ──────────────────────────────────────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────────────────────────────────────


class TestFenetrage(BaseTemporaire):
    def _feed_annuel(self) -> Path:
        dates = [
            (dt.date(2026, 3, 16) + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(120)
        ]
        courses, calendrier = [], {}
        for i, date in enumerate(dates):
            service = f"SVC_{i:04d}"
            calendrier[service] = [date]
            courses.append(course(f"T{i}", service=service))
        return ecrire_feed(
            self.racine / "annuel", lignes=["L1"], arrets=ARRETS, courses=courses,
            calendrier=calendrier, geometries=GEOM,
        )

    def test_refuse_au_dela_du_masque_binaire(self):
        """GAMA encodes the calendar as a 64-bit mask: beyond that, the export
        fails (`assert len(all_dates) <= 64` in inputs/gtfs/gama.py)."""
        code = window_feed.fenetrer(self._feed_annuel(), "20260316", 90, self.racine / "f", MUET)
        self.assertEqual(code, 2)

    def test_fenetre_bornee_et_fermee(self):
        source = self._feed_annuel()
        sortie = self.racine / "fenetre"
        self.assertEqual(window_feed.fenetrer(source, "20260316", 64, sortie, MUET), 0)

        index = offre.indexer(export_de(sortie), MUET)
        self.assertEqual(index.dates[0], "20260316")
        self.assertEqual(len(index.dates), 64)
        self.assertLessEqual(
            len(index.dates), window_feed.LIMITE_MASQUE, "le masque binaire ne tient pas au-delà"
        )
        # No trip outside the window, and no dangling service.
        services_calendrier = {
            l["service_id"] for l in gtfs_io.lire(export_de(sortie), "calendar_dates.txt")
        }
        services_trips = {l["service_id"] for l in gtfs_io.lire(export_de(sortie), "trips.txt")}
        self.assertEqual(services_trips, services_calendrier)

    def test_fenetre_hors_calendrier_refusee(self):
        code = window_feed.fenetrer(self._feed_annuel(), "20250101", 64, self.racine / "f", MUET)
        self.assertEqual(code, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Shipped configuration
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigurationLivree(unittest.TestCase):
    """The shipped YAML must carry every key the code reads.

    Without this test, removing a key from `feed_year.yaml` breaks nothing before the
    next full build — and the build takes two minutes.
    """

    def test_cles_attendues_presentes(self):
        for chemin in [
            ("fiabilite", "ratio_lignes_min"),
            ("fiabilite", "ratio_plancher_lignes"),
            ("fiabilite", "jours_min_par_export"),
            ("fiabilite", "falaise_lignes_part_min"),
            ("fiabilite", "falaise_lignes_min"),
            ("fiabilite", "falaise_fenetre_jours"),
            ("fiabilite", "falaise_jours_apres_min"),
            ("calendrier", "localite"),
            ("calendrier", "decalages_debut_testes"),
            ("extrapolation", "repli_periodes"),
            ("extrapolation", "repli_ferie"),
            ("extrapolation", "confiance", "moyenne_si_ecart_jours_max"),
            ("controles", "deplacement_arret_max_m"),
            ("controles", "dispersion_signature_max"),
            ("controles", "ratio_lignes_min_jour_ouvre"),
            ("controles", "dates_confiance_basse_max"),
            ("controles", "holdout_ecart_max"),
            ("canonicalisation", "decimales_coordonnees"),
            ("canonicalisation", "decimales_distance"),
        ]:
            noeud = CONFIG
            for cle in chemin:
                self.assertIn(cle, noeud, f"clé manquante : {' → '.join(chemin)}")
                noeud = noeud[cle]

    def test_chaque_classe_de_periode_a_un_repli(self):
        replis = CONFIG["extrapolation"]["repli_periodes"]
        for classe in calendar_fr.CLASSES_VACANCES + (calendar_fr.SCOLAIRE,):
            self.assertIn(classe, replis, f"classe sans chaîne de repli déclarée : {classe}")

    def test_reseaux_declares_ont_une_identite(self):
        from scripts.data.gtfs_year.build_year_feed import IDENTITES

        for reseau in CONFIG["reseaux"]:
            self.assertIn(reseau, IDENTITES, f"réseau sans identité de feed : {reseau}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
