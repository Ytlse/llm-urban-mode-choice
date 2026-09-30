"""
Choice of the source of each day of the year.

Two decisions, in this order:

1. **Authority.** A date covered by several exports takes its supply from
   ONE only. Taking the union would over-serve: on 04/05/2026, two exports
   give 12,538 and 12,484 trips respectively, with only 11,282 in
   common; their union would make 13,740 of them. This is exactly the defect of the
   feed currently in service, which serves 13,250 trips on 08/04 where its two
   sources give 12,652 and 12,660.

2. **Donor.** A date without coverage receives the verbatim copy of a real
   day with the same signature — the same weekday in the same school-period
   class — preferring the closest one in time. No timetable
   is synthesised: what is served was published by the operator, only
   not on that day.

Dates that the source explicitly declares without service (1 May, omitted by
the two exports that span it) are not gaps: extrapolating them
would invent supply on a public holiday when the transit system does not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .calendar_fr import Periode, Signature, signature, to_date
from .offre import IndexExport

REEL = "reel"
EXTRAPOLE = "extrapole"
SANS_SERVICE = "sans_service"

HAUTE = "haute"
MOYENNE = "moyenne"
BASSE = "basse"


@dataclass
class Provenance:
    """Where the supply of a day of the produced feed comes from."""

    date: str
    signature: str
    mode: str
    confiance: str
    export: str = ""
    date_source: str = ""
    ecart_jours: int = 0
    ecart_saison: int = 0
    motif: str = ""
    nb_trips: int = 0
    nb_lignes: int = 0

    def en_dict(self) -> dict:
        return {
            "date": self.date,
            "signature": self.signature,
            "mode": self.mode,
            "confiance": self.confiance,
            "export": self.export,
            "date_source": self.date_source,
            "ecart_jours": self.ecart_jours,
            "ecart_saison": self.ecart_saison,
            "motif": self.motif,
            "nb_trips": self.nb_trips,
            "nb_lignes": self.nb_lignes,
        }


def autorite(
    index_par_export: dict[str, IndexExport],
    dates_fiables: dict[str, list[str]],
    journal=print,
) -> dict[str, str]:
    """date → label of the export that is authoritative on that day.

    In case of overlap, the most recently published export wins: it is
    the one that includes the latest operating decisions.
    """
    rang = {
        etiquette: (index.export.date_min, etiquette)
        for etiquette, index in index_par_export.items()
    }
    choix: dict[str, str] = {}
    recouvrements = 0
    for etiquette, dates in dates_fiables.items():
        for date in dates:
            actuel = choix.get(date)
            if actuel is None:
                choix[date] = etiquette
            else:
                recouvrements += 1
                if rang[etiquette] > rang[actuel]:
                    choix[date] = etiquette
    journal(
        f"    autorité : {len(choix)} date(s) réelle(s), "
        f"{recouvrements} recouvrement(s) arbitré(s) en faveur de l'export le plus récent"
    )
    return choix


def _signatures_reelles(
    dates_reelles: list[str],
    periodes: list[Periode],
    feries: dict[str, str],
    decalages: dict[str, int],
) -> dict[str, list[str]]:
    """signature → available real dates, sorted."""
    par_signature: dict[str, list[str]] = {}
    for date in dates_reelles:
        sig = str(signature(date, periodes, feries, decalages))
        par_signature.setdefault(sig, []).append(date)
    for dates in par_signature.values():
        dates.sort()
    return par_signature


def ecart_saisonnier(date_a: str, date_b: str) -> int:
    """Distance between two dates in the seasonal sense, in days.

    This is the right notion of proximity for choosing a donor: what matters
    is not the number of calendar days elapsed but the resemblance of the
    period of the year. Without it, a 5 January would look for its donor
    259 days away (21 September) whereas 16 March, 70 season days away,
    resembles it more — and a year entirely copied from the previous one
    would only have "distant" donors.
    """
    doy_a = to_date(date_a).timetuple().tm_yday
    doy_b = to_date(date_b).timetuple().tm_yday
    ecart = abs(doy_a - doy_b)
    return min(ecart, 365 - ecart)


def _candidats(sig: Signature, config_extrap: dict) -> list[tuple[str, bool]]:
    """Acceptable signatures for a target, from best to worst.

    The boolean says whether it is the exact signature. A public holiday first looks for
    another public holiday: 14/07 serves 5,674 trips against 4,683 to 5,054 for the
    Sundays of July, and 08/05 4,782 against 4,644 — treating a public holiday
    as a Sunday is off by about 10 %.
    """
    types_jour = [sig.type_jour]
    if sig.type_jour == "ferie":
        types_jour = list(config_extrap.get("repli_ferie", ["ferie", "dim"]))

    periodes = [sig.periode] + list(config_extrap.get("repli_periodes", {}).get(sig.periode, []))

    sorties: list[tuple[str, bool]] = []
    for i_periode, periode in enumerate(periodes):
        for i_jour, type_jour in enumerate(types_jour):
            sorties.append((f"{type_jour}/{periode}", i_periode == 0 and i_jour == 0))
    return sorties


def plan_annee(
    annee: int,
    dates_annee: list[str],
    source_par_date: dict[str, str],
    index_par_export: dict[str, IndexExport],
    periodes: list[Periode],
    feries: dict[str, str],
    decalages: dict[str, int],
    config_extrap: dict,
    dates_sans_service: set[str],
    journal=print,
) -> dict[str, Provenance]:
    """Decides, for each day of the year, where its supply comes from."""
    dates_reelles = sorted(source_par_date)
    par_signature = _signatures_reelles(dates_reelles, periodes, feries, decalages)
    ecart_moyenne_max = int(config_extrap["confiance"]["moyenne_si_ecart_jours_max"])

    plan: dict[str, Provenance] = {}
    sans_donneur: list[str] = []

    for date in dates_annee:
        sig = signature(date, periodes, feries, decalages)
        sig_str = str(sig)

        if date in dates_sans_service:
            plan[date] = Provenance(
                date=date,
                signature=sig_str,
                mode=SANS_SERVICE,
                confiance=HAUTE,
                motif="déclaré sans service par la source",
            )
            continue

        if date in source_par_date:
            etiquette = source_par_date[date]
            index = index_par_export[etiquette]
            plan[date] = Provenance(
                date=date,
                signature=sig_str,
                mode=REEL,
                confiance=HAUTE,
                export=etiquette,
                date_source=date,
                nb_trips=index.nb_trips(date),
                nb_lignes=index.lignes_par_date.get(date, 0),
            )
            continue

        meilleur: tuple[int, str, str, bool] | None = None
        for sig_candidate, exacte in _candidats(sig, config_extrap):
            for candidate in par_signature.get(sig_candidate, ()):
                ecart = ecart_saisonnier(date, candidate)
                cle = (ecart, candidate)
                if meilleur is None or cle < (meilleur[0], meilleur[1]):
                    meilleur = (ecart, candidate, sig_candidate, exacte)
            if meilleur is not None:
                break  # a worse fallback cannot beat a better rank

        if meilleur is None:
            sans_donneur.append(date)
            plan[date] = Provenance(
                date=date,
                signature=sig_str,
                mode=SANS_SERVICE,
                confiance=BASSE,
                motif="aucun donneur de signature compatible",
            )
            continue

        ecart, date_source, sig_source, exacte = meilleur
        if exacte and ecart <= ecart_moyenne_max:
            confiance = MOYENNE
        else:
            confiance = BASSE
        etiquette = source_par_date[date_source]
        index = index_par_export[etiquette]
        plan[date] = Provenance(
            date=date,
            signature=sig_str,
            mode=EXTRAPOLE,
            confiance=confiance,
            export=etiquette,
            date_source=date_source,
            ecart_jours=abs((to_date(date_source) - to_date(date)).days),
            ecart_saison=ecart,
            motif="signature exacte" if exacte else f"repli sur {sig_source}",
            nb_trips=index.nb_trips(date_source),
            nb_lignes=index.lignes_par_date.get(date_source, 0),
        )

    if sans_donneur:
        journal(
            f"[ALARME] {len(sans_donneur)} date(s) sans donneur compatible : "
            f"{', '.join(sans_donneur[:8])}{'…' if len(sans_donneur) > 8 else ''}"
        )

    reels = sum(1 for p in plan.values() if p.mode == REEL)
    extrapoles = sum(1 for p in plan.values() if p.mode == EXTRAPOLE)
    vides = sum(1 for p in plan.values() if p.mode == SANS_SERVICE)
    basses = sum(1 for p in plan.values() if p.confiance == BASSE)
    journal(
        f"    plan {annee} : {reels} jour(s) réel(s), {extrapoles_txt(extrapoles)}, "
        f"{vides} sans service — dont {basses} en confiance basse"
    )
    return plan


def extrapoles_txt(n: int) -> str:  # pragma: no cover - confort de lecture
    return f"{n} extrapolé(s)"
