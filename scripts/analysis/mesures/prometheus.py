"""Publishing the measurements to Prometheus — ticket 093, lot 3.

⚠ **EVERYTHING IS A GAUGE, AND THAT IS WHY THIS MODULE EXISTS.** A `Counter` incremented as
decisions go by doubles mechanically on the replay of a resume: days already lived are
redone, and the counter counts them twice. Measured on `2026-09-16_15_58`, where 113 trips and
81 recalls were replayed — a counter would have shown nearly double without any line
flagging it. A gauge SET to the recomputed value cannot lie: it is worth what the CSV says,
and the replay rewrites it identically.

⚠ **Grafana's axis is REAL time, not simulated time.** A simulated day lasts about
six minutes: the curves are readable, but the x-axis is not "day 1, day 2". The dashboard
is a monitoring instrument; the CSV is authoritative.

⚠ **A missing measurement is not published.** A habit the window does not determine, a memory
state without a checkpoint: the series is not set to zero, it is not written. In
Prometheus as in the CSV, a zero reads as perfect conformity.
"""

from __future__ import annotations

from typing import Any

from scripts.analysis.mesures.calcul import Mesures

PREFIXE = "persona"


def _familles(gauge_factory: Any) -> dict[str, Any]:
    """Declares the families. `gauge_factory(nom, aide, labels)` returns a gauge."""
    return {
        "jour": gauge_factory(
            f"{PREFIXE}_dernier_jour_clos",
            "Dernier jour simulé pour lequel les mesures sont écrites", []),
        "trajets": gauge_factory(
            f"{PREFIXE}_jour_trajets",
            "Trajets du dernier jour clos, par persona", ["person_id"]),
        "part_decidee": gauge_factory(
            f"{PREFIXE}_part_decidee",
            "Part des trajets du jour ayant eu au moins deux itinéraires proposés",
            ["person_id"]),
        "part_modale": gauge_factory(
            f"{PREFIXE}_part_modale",
            "Part d'un mode dans les trajets du jour, par persona", ["person_id", "mode"]),
        "reprise": gauge_factory(
            f"{PREFIXE}_reprise_veille",
            "Part des activités reprises dans le même mode que le jour vécu précédent",
            ["person_id"]),
        "conformite": gauge_factory(
            f"{PREFIXE}_conformite_habitude",
            "Part des activités conformes au mode majoritaire des 5 dernières observations",
            ["person_id"]),
        "vivier": gauge_factory(
            f"{PREFIXE}_vivier_rappel",
            "Taille maximale du vivier de rappel du jour, par persona", ["person_id"]),
        "servis": gauge_factory(
            f"{PREFIXE}_souvenirs_servis",
            "Souvenirs servis aux décisions du jour, par persona", ["person_id"]),
        "operations": gauge_factory(
            f"{PREFIXE}_operations_concept",
            "Opérations de concept du jour, par persona et par opération",
            ["person_id", "operation"]),
        "duree_vie": gauge_factory(
            f"{PREFIXE}_duree_vie_mediane_jours",
            "Durée de vie médiane des souvenirs, en jours, par persona et par type",
            ["person_id", "type"]),
        "choc_expositions": gauge_factory(
            f"{PREFIXE}_choc_expositions",
            "Expositions au choc du jour, par persona", ["person_id"]),
        "choc_minutes": gauge_factory(
            f"{PREFIXE}_choc_minutes_injectees",
            "Minutes de retard injectées par le choc du jour, par persona", ["person_id"]),
        "choc_servi": gauge_factory(
            f"{PREFIXE}_choc_souvenir_servi",
            "Le souvenir du choc a-t-il été servi à une décision du jour (1/0). "
            "Non publiée quand aucun souvenir n'est appariable au choc.", ["person_id"]),
    }


def publier(mesures: Mesures, gauges: dict[str, Any]) -> int:
    """Sets the gauges on the LAST CLOSED DAY. Returns the index of that day, or 0.

    The last closed day, not the current day: a day still open would see its modal
    shares rise as departures go by, which would read as a change of behaviour.
    """
    if len(mesures.journees) < 2:
        return 0
    jour = mesures.journees[-2]
    gauges["jour"].set(jour.index)

    for ligne in mesures.choix_modal:
        if ligne.jour_simule != jour.index:
            continue
        etiquette = {"person_id": ligne.person_id}
        gauges["trajets"].labels(**etiquette).set(ligne.trajets)
        gauges["part_decidee"].labels(**etiquette).set(ligne.part_decidee)
        for mode, part in ligne.parts.items():
            gauges["part_modale"].labels(person_id=ligne.person_id, mode=mode).set(part)

    for person_id, (reprises, conformes) in _habitudes_par_agent(mesures, jour.index).items():
        # Published only if at least one activity could be judged: otherwise an agent none of
        # whose activities has a precedent would show 0 % carry-over, which reads as a
        # total break although nothing was measured.
        if reprises is not None:
            gauges["reprise"].labels(person_id=person_id).set(reprises)
        if conformes is not None:
            gauges["conformite"].labels(person_id=person_id).set(conformes)

    for ligne in mesures.memoire:
        if ligne.jour_simule != jour.index:
            continue
        etiquette = {"person_id": ligne.person_id}
        if ligne.vivier_max is not None:
            gauges["vivier"].labels(**etiquette).set(ligne.vivier_max)
        gauges["servis"].labels(**etiquette).set(ligne.souvenirs_servis)
        for operation, compte in ligne.operations.items():
            if compte is not None:
                gauges["operations"].labels(
                    person_id=ligne.person_id, operation=operation).set(compte)

    for ligne in mesures.durees_de_vie:
        if ligne.jour_simule == jour.index:
            gauges["duree_vie"].labels(
                person_id=ligne.person_id,
                type=ligne.type_souvenir).set(ligne.duree_vie_mediane_jours)

    for ligne in mesures.chocs:
        if ligne.jour_simule != jour.index:
            continue
        etiquette = {"person_id": ligne.person_id}
        gauges["choc_expositions"].labels(**etiquette).set(ligne.expositions)
        gauges["choc_minutes"].labels(**etiquette).set(ligne.minutes_injectees)
        if ligne.souvenir_choc_servi is not None:
            gauges["choc_servi"].labels(**etiquette).set(1 if ligne.souvenir_choc_servi else 0)
    return jour.index


def _habitudes_par_agent(mesures: Mesures,
                         jour: int) -> dict[str, tuple[float | None, float | None]]:
    """Per agent: share of activities carried over, and share of conforming activities.

    Activities without a precedent count NEITHER in the numerator NOR in the denominator. Counting
    them in the denominator would drop the share on the day an agent discovers an activity, which
    would read as a habit break.
    """
    reprises: dict[str, list[bool]] = {}
    conformes: dict[str, list[bool]] = {}
    for ligne in mesures.habitudes:
        if ligne.jour_simule != jour:
            continue
        if ligne.reprise_veille is not None:
            reprises.setdefault(ligne.person_id, []).append(ligne.reprise_veille)
        if ligne.conforme_habitude is not None:
            conformes.setdefault(ligne.person_id, []).append(ligne.conforme_habitude)
    agents = set(reprises) | set(conformes)
    return {
        agent: (
            _part(reprises.get(agent)),
            _part(conformes.get(agent)),
        )
        for agent in agents
    }


def _part(valeurs: list[bool] | None) -> float | None:
    return sum(valeurs) / len(valeurs) if valeurs else None
