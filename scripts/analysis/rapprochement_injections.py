"""Reconciliation of declared vs produced injections — Ticket 108.

A run may declare injection days in `evenement.yaml` and produce none of them,
or fewer than declared, without any alarm or report flagging it — and keep measuring
the effect of an event that did not take place.

This module compares, for each exposed agent, the planned injections with the lines actually written
in `evenements.jsonl` (ignoring the intra-household relays `origine: entendu`, ticket 111).
For each missing injection, it identifies the plausible reason from `moves.csv`:
- mode not chosen (with an explicit mention if following the previous shock — protocol side effect);
- motionless agent (no trip that day);
- run stopped before the declared day.

Usage:
    python scripts/analysis/rapprochement_injections.py experiments/archive/simulations_septembre_2026/2026-09-23_20_35
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

# Normalisation of mode labels
MODE_NORMALISATION = {
    "voiture privée": "car",
    "voiture": "car",
    "car": "car",
    "transports_collectifs": "public_transport",
    "transports collectifs": "public_transport",
    "public_transport": "public_transport",
    "pt": "public_transport",
    "train": "train",
    "rail": "train",
    "marche": "walk",
    "pied": "walk",
    "walk": "walk",
    "foot": "walk",
    "vélo": "bike",
    "velo": "bike",
    "bike": "bike",
    "deux-roues motorisé": "motorcycle",
    "deux-roues": "motorcycle",
    "motorcycle": "motorcycle",
}


def normaliser_mode(mode: str | None) -> str:
    if not mode:
        return ""
    m = str(mode).strip().lower()
    return MODE_NORMALISATION.get(m, m)


def _jsonl(chemin: Path) -> list[dict]:
    if not chemin.is_file():
        return []
    sortie = []
    for brute in chemin.read_text(encoding="utf-8").splitlines():
        brute = brute.strip()
        if brute:
            try:
                sortie.append(json.loads(brute))
            except json.JSONDecodeError:
                continue
    return sortie


@dataclass
class LigneRapprochement:
    person_id: str
    evenement_id: str
    jour_declare: int | None
    date_simulee: date | None
    statut: str  # "produite" | "manquante"
    detail: str = ""
    raison: str = ""
    choc_precedent_jour: int | None = None

    @property
    def est_produite(self) -> bool:
        return self.statut == "produite"


@dataclass
class RapprochementRun:
    evenement_id: str = ""
    canal: str = ""
    forme: str = ""  # "A" (jours) | "B" (calendrier) | ""
    aucun_evenement: bool = False
    lignes: list[LigneRapprochement] = field(default_factory=list)
    declarees: int = 0
    produites: int = 0
    manquantes: int = 0
    conforme: bool = True

    @property
    def alarmes(self) -> list[str]:
        alarmes = []
        for l in self.lignes:
            if not l.est_produite:
                date_str = f" ({l.date_simulee:%Y-%m-%d})" if l.date_simulee else ""
                jour_str = f" au jour {l.jour_declare}" if l.jour_declare is not None else ""
                cause = f" : {l.raison}" if l.raison else ""
                alarmes.append(
                    f"🔴 INJECTION MANQUANTE — agent `{l.person_id}`, événement « {l.evenement_id} »"
                    f"{jour_str}{date_str}{cause}."
                )
        return alarmes


def _charger_declaration(run: Path, fichier_specifique: Path | None = None) -> dict[str, Any]:
    if fichier_specifique is not None and fichier_specifique.is_file():
        try:
            raw = yaml.safe_load(fichier_specifique.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass

    for nom in ("evenement.yaml", "choc.yaml"):
        f = run / nom
        if f.is_file():
            try:
                raw = yaml.safe_load(f.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and (raw.get("evenement") or raw.get("choc") or raw.get("jours")):
                    return raw
            except Exception:
                continue
    # Lookup in identite_run.json
    identite = run / "identite_run.json"
    if identite.is_file():
        try:
            data = json.loads(identite.read_text(encoding="utf-8"))
            evt_id = data.get("evenement") or data.get("choc")
            if evt_id and isinstance(evt_id, str) and not re.match(r"^[0-9a-f]{40,64}$", evt_id):
                # Symbolic reference
                for dossier in ("evenements", "chocs"):
                    f_cfg = run.parents[2] / "services" / "llm-agents" / "config" / dossier / f"{evt_id}.yaml"
                    if f_cfg.is_file():
                        raw = yaml.safe_load(f_cfg.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            return raw
        except Exception:
            pass
    return {}


def _lire_dates_et_trajets_moves(run: Path) -> tuple[date | None, dict[str, dict[date, list[str]]], date | None]:
    """Extracts the run start date, the trips per agent and date, and the maximum date."""
    chemin = run / "moves.csv"
    if not chemin.is_file():
        return None, {}, None

    date_debut: date | None = None
    date_fin: date | None = None
    trajets_par_agent: dict[str, dict[date, list[str]]] = {}

    with chemin.open(encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        for row in lecteur:
            pid = str(row.get("ID Personne") or "").strip()
            # Date from Heure de départ or Temps simulé
            h_dep = str(row.get("Heure de départ") or "").strip()
            d: date | None = None
            if len(h_dep) >= 10:
                try:
                    d = date.fromisoformat(h_dep[:10])
                except ValueError:
                    pass
            if d is None:
                ts_str = row.get("Temps simulé")
                if ts_str:
                    try:
                        d = datetime.fromtimestamp(float(ts_str), timezone.utc).date()
                    except (ValueError, TypeError):
                        pass

            if d is not None:
                if date_debut is None or d < date_debut:
                    date_debut = d
                if date_fin is None or d > date_fin:
                    date_fin = d

                if pid:
                    mode = str(row.get("Mode de transport Choisi") or "").strip()
                    agent_jours = trajets_par_agent.setdefault(pid, {})
                    agent_jours.setdefault(d, []).append(mode)

    return date_debut, trajets_par_agent, date_fin


def rapprocher(run_dir: Path | str, fichier_declaration: Path | None = None) -> RapprochementRun:
    """Performs the declared vs produced reconciliation for a given run directory."""
    run = Path(run_dir)
    declaration = _charger_declaration(run, fichier_declaration)
    evenement_id = str(declaration.get("evenement") or declaration.get("choc") or "")
    if not declaration and not (run / "evenements.jsonl").is_file() and not (run / "chocs.jsonl").is_file():
        return RapprochementRun(aucun_evenement=True)

    canal = str(declaration.get("canal") or "vecu")
    expo = declaration.get("exposition") or {}
    modes_attendus = [normaliser_mode(m) for m in expo.get("modes") or []]

    # Determine the declared target agents
    agents_cibles: list[str] = []
    if expo.get("agents"):
        agents_cibles = [str(a) for a in expo["agents"]]
    elif expo.get("lecteurs"):
        agents_cibles = [str(a) for a in expo["lecteurs"]]

    # Events actually produced in evenements.jsonl (or chocs.jsonl)
    lignes_evt = []
    for nom in ("evenements.jsonl", "chocs.jsonl"):
        f = run / nom
        if f.is_file():
            lignes_evt = _jsonl(f)
            break

    # Ticket 111: ignore the `origine: entendu` lines (household relay)
    lignes_produites = [
        l for l in lignes_evt
        if l.get("origine") != "entendu"
    ]

    # Indexing of the productions by (person_id, jour_run)
    produits_par_agent_jour: dict[tuple[str, int], dict] = {}
    chocs_produits_par_agent: dict[str, list[int]] = {}
    for p in lignes_produites:
        pid = str(p.get("person_id") or "")
        j = p.get("jour_run")
        if isinstance(j, int) and pid:
            produits_par_agent_jour[(pid, j)] = p
            chocs_produits_par_agent.setdefault(pid, []).append(j)

    # Retrieval of the trip calendar
    date_debut, trajets_par_agent, date_fin = _lire_dates_et_trajets_moves(run)

    # If no explicit target agents in the declaration, look in the produced events
    # or in moves.csv
    if not agents_cibles:
        pids_prod = {str(p.get("person_id")) for p in lignes_produites if p.get("person_id")}
        if pids_prod:
            agents_cibles = sorted(pids_prod)
        elif trajets_par_agent:
            agents_cibles = sorted(trajets_par_agent.keys())

    # Form A: days declared in `jours:`
    jours_decl = declaration.get("jours")
    lignes_rapprochement: list[LigneRapprochement] = []
    declarees_tot = 0
    produites_tot = 0

    if isinstance(jours_decl, list) and jours_decl:
        forme = "A"
        for agent_id in agents_cibles:
            chocs_anterieurs = sorted(chocs_produits_par_agent.get(agent_id, []))
            for item in jours_decl:
                if not isinstance(item, dict):
                    continue
                declarees_tot += 1
                j_decl = item.get("jour")
                if not isinstance(j_decl, int):
                    continue

                date_prevue: date | None = None
                if date_debut is not None:
                    date_prevue = date_debut + timedelta(days=j_decl - 1)

                prod = produits_par_agent_jour.get((agent_id, j_decl))
                if prod is not None:
                    produites_tot += 1
                    retard_s = prod.get("retard_injecte_s") or (item.get("retard_min", 0) * 60)
                    retard_txt = f"{int(retard_s) // 60} min" if retard_s else "0 min"
                    lignes_rapprochement.append(
                        LigneRapprochement(
                            person_id=agent_id,
                            evenement_id=evenement_id or str(prod.get("evenement_id") or prod.get("choc_id") or ""),
                            jour_declare=j_decl,
                            date_simulee=date_prevue,
                            statut="produite",
                            detail=f"injectée ({retard_txt})",
                        )
                    )
                else:
                    # Diagnosis of the cause
                    raison = ""
                    max_j_run = (date_fin - date_debut).days + 1 if (date_debut and date_fin) else None
                    if max_j_run is not None and j_decl > max_j_run:
                        raison = f"le run s'est arrêté avant ce jour (simulation arrêtée à J{max_j_run})"
                    elif date_prevue is not None and agent_id in trajets_par_agent:
                        trajets_du_jour = trajets_par_agent[agent_id].get(date_prevue, [])
                        if not trajets_du_jour:
                            raison = f"l'agent ne s'est pas déplacé ce jour-là (aucun trajet le {date_prevue:%Y-%m-%d})"
                        else:
                            modes_empruntes_norm = [normaliser_mode(m) for m in trajets_du_jour]
                            modes_distincts = list(dict.fromkeys(trajets_du_jour))
                            modes_str = ", ".join(modes_distincts)
                            modes_attendus_str = ", ".join(modes_attendus) if modes_attendus else "déclaré"

                            # Was the exposure mode used?
                            mode_choisi = any(m in modes_attendus for m in modes_empruntes_norm) if modes_attendus else True
                            if not mode_choisi:
                                # Check whether an earlier shock took place
                                chocs_avant = [cj for cj in chocs_anterieurs if cj < j_decl]
                                if chocs_avant:
                                    dernier_choc = chocs_avant[-1]
                                    raison = (
                                        f"mode « {modes_attendus_str} » non choisi ce jour-là "
                                        f"({modes_str} après le choc du J{dernier_choc})"
                                    )
                                else:
                                    raison = (
                                        f"mode « {modes_attendus_str} » non choisi ce jour-là "
                                        f"({modes_str} emprunté(s) à la place)"
                                    )
                            else:
                                raison = "mode choisi mais conditions d'arrivée non réunies"
                    elif date_prevue is not None:
                        raison = f"aucun déplacement enregistré pour cet agent le {date_prevue:%Y-%m-%d}"
                    else:
                        raison = "non injectée (date simulée indéterminée)"

                    lignes_rapprochement.append(
                        LigneRapprochement(
                            person_id=agent_id,
                            evenement_id=evenement_id,
                            jour_declare=j_decl,
                            date_simulee=date_prevue,
                            statut="manquante",
                            raison=raison,
                        )
                    )

    elif "calendrier" in declaration:
        # Form B: publication calendar
        forme = "B"
        if expo.get("regle") == "foyers" and expo.get("lecteurs"):
            nb_attendus = len(expo.get("lecteurs") or [])
        elif expo.get("regle") == "foyers":
            nb_attendus = len(expo.get("foyers") or []) * int(expo.get("lecteurs_par_foyer") or 1)
        elif expo.get("regle") == "agents":
            nb_attendus = len(expo.get("agents") or [])
        else:
            nb_attendus = len(agents_cibles) or 1

        declarees_tot = nb_attendus

        pids_attendus = set(agents_cibles)
        pids_vus: set[str] = set()

        for p in lignes_produites:
            produites_tot += 1
            pid = str(p.get("person_id") or "?")
            pids_vus.add(pid)
            j0 = p.get("jour_run")
            date_p0: date | None = None
            if p.get("horodatage_simule"):
                try:
                    date_p0 = datetime.fromisoformat(str(p["horodatage_simule"])[:10]).date()
                except ValueError:
                    pass
            lignes_rapprochement.append(
                LigneRapprochement(
                    person_id=pid,
                    evenement_id=evenement_id or str(p.get("evenement_id") or ""),
                    jour_declare=j0 if isinstance(j0, int) else None,
                    date_simulee=date_p0,
                    statut="produite",
                    detail="lecture effectuée",
                )
            )

        if pids_attendus and pids_attendus.difference(pids_vus):
            for manquant_id in sorted(pids_attendus - pids_vus):
                lignes_rapprochement.append(
                    LigneRapprochement(
                        person_id=manquant_id,
                        evenement_id=evenement_id,
                        jour_declare=None,
                        date_simulee=None,
                        statut="manquante",
                        raison="lecture non produite dans la fenêtre déclarée",
                    )
                )
        elif produites_tot < declarees_tot:
            for i in range(declarees_tot - produites_tot):
                lignes_rapprochement.append(
                    LigneRapprochement(
                        person_id=f"lecteur_manquant_{i + 1}",
                        evenement_id=evenement_id,
                        jour_declare=None,
                        date_simulee=None,
                        statut="manquante",
                        raison="lecture non produite dans la fenêtre déclarée",
                    )
                )
    else:
        # No days nor structured calendar but an event present or lines produced
        forme = ""
        if lignes_produites:
            for p in lignes_produites:
                produites_tot += 1
                declarees_tot += 1
                pid = str(p.get("person_id") or "?")
                j = p.get("jour_run")
                d_p: date | None = None
                if p.get("horodatage_simule"):
                    try:
                        d_p = datetime.fromisoformat(str(p["horodatage_simule"])[:10]).date()
                    except ValueError:
                        pass
                lignes_rapprochement.append(
                    LigneRapprochement(
                        person_id=pid,
                        evenement_id=evenement_id or str(p.get("evenement_id") or p.get("choc_id") or ""),
                        jour_declare=j if isinstance(j, int) else None,
                        date_simulee=d_p,
                        statut="produite",
                        detail="injectée",
                    )
                )

    manquantes_tot = max(0, declarees_tot - produites_tot)
    conforme = (manquantes_tot == 0 and declarees_tot > 0)

    return RapprochementRun(
        evenement_id=evenement_id,
        canal=canal,
        forme=forme,
        aucun_evenement=not declaration and not lignes_rapprochement,
        lignes=lignes_rapprochement,
        declarees=declarees_tot,
        produites=produites_tot,
        manquantes=manquantes_tot,
        conforme=conforme,
    )


def rendre(rapprochement: RapprochementRun) -> list[str]:
    """Generates the Markdown lines of the report for ticket 108."""
    if rapprochement.aucun_evenement or (not rapprochement.declarees and not rapprochement.produites):
        return ["_Aucun événement déclaré dans ce run : rien à rapprocher._\n"]

    out = [
        "| Agent | Événement | Jour déclaré | Date simulée | Statut | Détail / Raison |",
        "|:--|:--|:--|:--|:--|:--|",
    ]

    for l in rapprochement.lignes:
        jour_txt = f"J{l.jour_declare}" if l.jour_declare is not None else "—"
        date_txt = f"{l.date_simulee:%Y-%m-%d}" if l.date_simulee else "—"
        if l.est_produite:
            statut_txt = "✅ produite"
            info_txt = l.detail or "injectée"
        else:
            statut_txt = "🔴 **manquante**"
            info_txt = l.raison or "non produite"

        out.append(
            f"| `{l.person_id}` | `{l.evenement_id}` | {jour_txt} | {date_txt} | {statut_txt} | {info_txt} |"
        )

    out.append("")
    if rapprochement.forme == "B":
        out.append(f"_Forme B (calendrier / parution) : {rapprochement.declarees} cible(s) examinée(s)._\n")

    if rapprochement.conforme:
        out.append(
            f"✅ **{rapprochement.produites}/{rapprochement.declarees} injection(s) déclarée(s) produite(s)** "
            f"— aucune injection manquante.\n"
        )
    else:
        out.append(
            f"⚠️ **{rapprochement.declarees} injection(s) déclarée(s)** : "
            f"{rapprochement.produites} produite(s), **{rapprochement.manquantes} manquante(s)**.\n"
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="Run directory")
    args = parser.parse_args()

    r = rapprocher(args.run)
    print("\n".join(rendre(r)))
    if r.alarmes:
        print("\nAlarms:")
        for a in r.alarmes:
            print(f"  {a}")
    return 1 if not r.conforme and r.declarees > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
