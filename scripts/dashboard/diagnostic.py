"""Readable reasons drawn from the recorded errors, without querying the providers."""
from __future__ import annotations

import json
import re
from pathlib import Path


SIGNAUX = (
    ("quota", "quota journalier atteint — attendre le renouvellement",
     r"quota_journalier|(?:quota|limit).*(?:per.day|daily).*(?:exceed|exhaust)|(?:per.day|daily).*(?:quota|limit).*(?:exceed|exhaust)|quota journalier (?:atteint|épuisé)|Generate\w*PerDay"),
    ("saturation", "forte demande — réessayez plus tard",
     r"high demand|overloaded|forte demande|model.*overload"),
    ("quota", "limite par minute atteinte — réessayez dans un instant",
     r"quota.*per.minute|per.minute.*(?:exceed|exhaust)|Generate\w*PerMinute|rate.limit.exceeded"),
    ("quota", "limite du fournisseur atteinte — durée non précisée",
     r"\b429\b|RESOURCE_EXHAUSTED"),
    ("saturation", "service temporairement indisponible — réessayez plus tard",
     r"\b503\b|service unavailable|UNAVAILABLE"),
    ("reservation", "en attente d’une clé disponible — clés déjà réservées",
     r"EN FILE|clé\(s\) tenue\(s\)"),
    ("configuration", "authentification refusée — vérifiez la clé API",
     r"API_KEY_INVALID|invalid api key|\b401\b|UNAUTHENTICATED"),
)


def diagnostic(message: str) -> tuple[str, str] | None:
    for categorie, libelle, motif in SIGNAUX:
        if re.search(motif, message, re.IGNORECASE):
            return categorie, libelle
    return None


def expliquer(erreur: dict) -> str:
    message = str(erreur.get("message") or erreur.get("raison") or "")
    signal = f"{erreur.get('type', '')} {erreur.get('genre_erreur', '')} {message}"
    resultat = diagnostic(signal)
    if not resultat:
        return ""
    fournisseur = str(erreur.get("fournisseur") or "")
    cle = re.search(r"\b[\w-]*key[_-]?(\d+)\b", fournisseur or message)
    prefixe = f"clé {cle[1]} : " if cle else ""
    heure = erreur.get("horodatage")
    return prefixe + resultat[1] + (f" · constaté à {heure}" if heure else "")


def queue(chemin: Path) -> list[str]:
    try:
        with chemin.open("rb") as fichier:
            fichier.seek(0, 2)
            debut = max(0, fichier.tell() - 131072)
            fichier.seek(debut)
            if debut:
                fichier.readline()
            return fichier.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []


def motifs_execution(dossier: Path) -> list[str]:
    """Last identifiable incident per instance, within a bounded read."""
    derniers = {}
    for ligne in queue(dossier / "erreurs.jsonl"):
        try:
            erreur = json.loads(ligne)
        except (ValueError, TypeError):
            continue
        if not isinstance(erreur, dict):
            continue
        motif = expliquer(erreur)
        if motif:
            derniers[str(erreur.get("fournisseur") or "inconnu")] = motif
    return list(derniers.values())


def motif_lancement(base: Path, avant: str | None = None) -> str:
    # Log names are in UTC. Do not explain an old failure by a later replay.
    limite = str(avant or "9999")[:19].replace("T", "_").replace(":", "_")
    journaux = sorted(p for p in (base / "lancements").glob("*.log") if p.stem <= limite)
    if not journaux:
        return ""
    for ligne in reversed(queue(journaux[-1])):
        resultat = diagnostic(ligne)
        if resultat:
            return resultat[1]
    return ""
