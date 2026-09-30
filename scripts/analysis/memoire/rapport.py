"""Self-contained HTML report on a memory run, one section per persona.

    python -m scripts.analysis.memoire.rapport <dossier_run> -o <sortie.html>

A single output file: inline CSS, inline SVG, no external resource.
Two runs on the same run produce the **same bytes** (contract F7): nothing in
the body carries the generation time, and every set traversal is sorted.

Everything that comes from the data goes through ``html.escape``: concept statements
are written by a language model, and a memory log is not trusted
HTML.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

from . import graphiques, mesures, sources
from .sources import MODE_LIBELLES, MODES, Run, libelle_motif

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #FAFAF8; color: #1D1D1F;
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.mx-page { max-width: 1120px; margin: 0 auto; padding-block: 28px; padding-left: 20px; padding-right: 20px; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 20px; margin: 40px 0 10px; padding-top: 14px; border-top: 2px solid #1D1D1F; }
h3 { font-size: 16px; margin: 26px 0 8px; }
h4 { font-size: 14px; margin: 18px 0 6px; color: #444; text-transform: uppercase; letter-spacing: .04em; }
p { margin: 8px 0; }
.mx-sous-titre { color: #5A5A5F; margin-top: 0; }
.mx-cartouche { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 6px; }
.mx-fait { background: #fff; border: 1px solid #E3E3E0; border-radius: 8px;
           padding: 8px 12px; min-width: 120px; flex: 1 1 140px; }
.mx-fait b { display: block; font-size: 19px; }
.mx-fait span { color: #5A5A5F; font-size: 12px; }
.mx-alerte { background: #FFF4E5; border: 1px solid #E8C89A; border-radius: 8px;
             padding: 10px 14px; margin: 14px 0; }
.mx-vide { background: #F2F2EF; border: 1px dashed #C9C9C4; border-radius: 8px;
           padding: 10px 14px; margin: 10px 0; color: #5A5A5F; }
.mx-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 4px; }
table { border-collapse: collapse; font-size: 13px; min-width: 100%; }
th, td { border-bottom: 1px solid #E3E3E0; padding: 6px 9px; text-align: left; vertical-align: top;
         white-space: nowrap; }
th { background: #F2F2EF; font-weight: 600; position: sticky; top: 0; }
td.mx-num, th.mx-num { text-align: right; font-variant-numeric: tabular-nums; }
.mx-legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 6px 0 14px; font-size: 12px; color: #444; }
.mx-chip { display: inline-flex; align-items: center; gap: 6px; }
.mx-chip i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.mx-chip-note { color: #5A5A5F; font-style: italic; }
.mx-svg { display: block; }
.mx-label { font-size: 11px; fill: #1D1D1F; }
.mx-tick { font-size: 10px; fill: #5A5A5F; }
.mx-tick-bas { fill: #B04A22; }
.mx-axis { font-size: 10px; fill: #5A5A5F; }
.mx-value { font-size: 11px; fill: #1D1D1F; font-variant-numeric: tabular-nums; }
.mx-grid { stroke: #E3E3E0; stroke-width: 1; }
.mx-track { fill: #EDEDEA; }
.mx-case-vide { fill: #FFFFFF; stroke: #EDEDEA; stroke-width: 1; }
.mx-itineraires { font-size: 12.5px; color: #333; padding-left: 20px; margin: 4px 0 18px; }
.mx-itineraires > li { margin-bottom: 5px; }
.mx-etapes { color: #5A5A5F; font-size: 12px; margin: 2px 0 4px; padding-left: 16px; }
.mx-n { color: #5A5A5F; font-size: 11.5px; }
.mx-citation { background: #fff; border-left: 3px solid #1A4C8B; padding: 8px 12px;
               margin: 8px 0; font-size: 13.5px; color: #333; }
.mx-theme { background: #fff; border: 1px solid #E3E3E0; border-radius: 8px;
            padding: 10px 12px; margin: 10px 0; }
.mx-theme > b { display: block; margin-bottom: 4px; }
.mx-reformulations { margin: 6px 0 0; padding-left: 18px; font-size: 12.5px; color: #444; }
.mx-persona { border-top: 1px solid #E3E3E0; }
.mx-profil { font-size: 13.5px; color: #333; }
.mx-note { font-size: 12.5px; color: #5A5A5F; }
@media (max-width: 640px) {
  .mx-page { padding-left: 16px; padding-right: 16px; }
  h1 { font-size: 21px; }
  .mx-fait { flex: 1 1 100%; }
}
"""


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _pct(valeur: float | None, decimales: int = 1) -> str:
    return "—" if valeur is None else f"{valeur:.{decimales}f} %"


def _fait(valeur: str, libelle: str) -> str:
    return (f'<div class="mx-fait"><b>{escape(valeur)}</b>'
            f'<span>{escape(libelle)}</span></div>')


def _tableau(entetes: Sequence[tuple[str, bool]], lignes: Sequence[Sequence[str]]) -> str:
    """HTML table; the header boolean says whether the column is numeric."""
    th = "".join(f'<th class="mx-num">{escape(t)}</th>' if num else f'<th>{escape(t)}</th>'
                 for t, num in entetes)
    corps = []
    for ligne in lignes:
        cellules = "".join(
            f'<td class="mx-num">{v}</td>' if entetes[i][1] else f"<td>{v}</td>"
            for i, v in enumerate(ligne))
        corps.append(f"<tr>{cellules}</tr>")
    return (f'<div class="mx-scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(corps)}</tbody></table></div>')


# ── Header ────────────────────────────────────────────────────────────────────

def entete(run: Run) -> str:
    jours = run.jours
    periode = f"{jours[0]} → {jours[-1]}" if jours else "aucun jour"
    # The « Fournisseur & Modèle » column also carries the answers served by the
    # cache (`cache:<mode>`): it is not a provider, it is a trip without a call.
    modeles = sorted({t.fournisseur for t in run.trajets
                      if t.fournisseur and not t.fournisseur.startswith("cache:")})
    depuis_cache = sum(1 for t in run.trajets if t.fournisseur.startswith("cache:"))
    if not modeles:
        modeles = run.fournisseurs
    total = len(run.trajets)
    rejeu = run.rejeu
    declaration = (
        f'<div class="mx-alerte"><b>Rejeu exclu.</b> {rejeu} trajet(s) rejoué(s) après '
        f'redémarrage ont été écartés des mesures : la même décision (même personne, '
        f'même activité, même instant simulé) figurait plusieurs fois dans '
        f'<code>moves.csv</code>. On a gardé la première par heure de calcul. '
        f'Les {total} trajets restants sont ceux qui comptent.</div>'
        if rejeu else
        '<div class="mx-alerte"><b>Rejeu exclu.</b> Aucun trajet rejoué détecté : '
        'chaque décision ne figure qu\'une fois dans <code>moves.csv</code>.</div>')
    faits = "".join([
        _fait(periode, "période simulée"),
        _fait(str(len(jours)), "jours simulés"),
        _fait(str(len(run.agents)), "agents"),
        _fait(str(total), "trajets retenus"),
        _fait(str(rejeu), "trajets rejoués, exclus"),
        _fait(" · ".join(modeles) or "—", "fournisseur & modèle"),
        _fait(str(depuis_cache), "décisions servies par le cache"),
    ])
    return (f'<h1>Mémoire des agents — rapport par persona</h1>'
            f'<p class="mx-sous-titre">Run <code>{escape(run.chemin.name)}</code>. '
            f'Ticket 077, lot F.</p>'
            f'<div class="mx-cartouche">{faits}</div>{declaration}')


# ── Cross-cutting summary ────────────────────────────────────────────────────

def synthese(run: Run, contexte: dict[str, Any]) -> str:
    entetes = [("Agent", False), ("Profil", False), ("Trajets", True)]
    entetes += [(MODE_LIBELLES[m], True) for m in MODES[:4]]
    entetes += [("Concepts", True), ("Thèmes", True), ("Redondance", True),
                ("Consolidations", True), ("créé", True), ("confirmé", True),
                ("précisé", True), ("contredit", True)]
    lignes = []
    for agent in run.agents:
        vue = contexte[agent]
        profil = vue["profil"]
        parts = vue["parts"]
        operations = vue["journal"].operations if vue["journal"] else {}
        lignes.append([
            escape(f'{profil["nom"]} ({agent})'),
            escape(_profil_court(profil)),
            str(len(vue["trajets"])),
            *[_pct(parts.get(m), 0) if parts.get(m) is not None else "—" for m in MODES[:4]],
            str(len(vue["concepts"])),
            str(len(vue["themes"])),
            _pct(None if vue["redondance"] is None else 100 * vue["redondance"], 0),
            str(vue["journal"].consolidations if vue["journal"] else 0),
            str(operations.get("créé", 0)),
            str(operations.get("confirmé", 0)),
            str(operations.get("précisé", 0)),
            str(operations.get("contredit", 0)),
        ])
    sans_concept = [a for a in run.agents if not contexte[a]["concepts"]]
    note = ""
    if sans_concept:
        note = ('<p class="mx-note">Agents sans aucun concept : '
                + escape(", ".join(sans_concept))
                + ' — leur section est produite et déclarée vide, pas omise.</p>')
    return ("<h2>Synthèse transversale</h2>"
            "<p>Une ligne par agent. Les parts modales portent sur les trajets retenus "
            "(rejeu exclu). La redondance est la part des concepts qui reformulent un thème "
            "déjà présent.</p>" + _tableau(entetes, lignes) + note)


def _profil_court(profil: dict[str, Any]) -> str:
    morceaux = [str(profil.get("age") or "—") + " ans",
                str(profil.get("occupation") or "—"),
                str(profil.get("commune") or "—")]
    return " · ".join(morceaux)


# ── Section per persona ──────────────────────────────────────────────────────

def section_persona(run: Run, agent: str, vue: dict[str, Any]) -> str:
    profil = vue["profil"]
    titre = (f'<h2 class="mx-persona">{escape(profil["nom"])} '
             f'<span class="mx-n">({escape(agent)})</span></h2>')
    parties = [titre]
    parties.append(_fiche_profil(profil, vue))
    parties.append("<h3>Frise des modes</h3>")
    parties.append("<p>Une ligne par motif, une colonne par jour simulé ; la case porte "
                   "le mode retenu. Plusieurs trajets le même jour : la case se subdivise.</p>")
    parties.append(graphiques.frise_modes(vue["frise"], run.jours))

    parties.append("<h3>Itinéraires proposés et retenus</h3>")
    if not vue["tableaux"]:
        parties.append('<p class="mx-vide">Aucun trajet pour cet agent : '
                       'aucun itinéraire à montrer.</p>')
    for tableau in vue["tableaux"]:
        parties.append(f"<h4>{escape(libelle_motif(tableau.motif))}</h4>")
        parties.append(graphiques.tableau_itineraires(tableau, run.jours))
    appariement = vue["appariement"]
    parties.append(
        f'<p class="mx-note">Appariement trajet ↔ prompt : {appariement.apparies} sur '
        f'{appariement.total} ({_pct(appariement.taux, 0)}). Les options proposées '
        f"n'existent que dans le texte des prompts ; l'appariement se fait par "
        f"(agent, jour, motif) et heure de départ la plus proche. L'index effectivement "
        f"tiré n'étant journalisé nulle part dans ce run, l'option « retenue » est "
        f"celle du mode choisi — à mode égal, la plus courte.</p>")

    parties.append("<h3>Décisivité et entropie, par semaine</h3>")
    parties.append(graphiques.courbe_semaines(vue["semaines"]))
    part_exp, n_raisons = vue["raisons_experience"]
    parties.append(
        f'<p class="mx-note">Raisonnements citant explicitement le passé de l\'agent : '
        f'{_pct(part_exp, 0)} sur {n_raisons} raisonnement(s) lisible(s).</p>')

    parties.append("<h3>Concepts</h3>")
    parties.append(_bloc_concepts(vue))

    parties.append("<h3>Rappels servis</h3>")
    parties.append(graphiques.barres_rappels(vue["rappels"]))

    parties.append("<h3>Habitudes calculées et dernière auto-réflexion</h3>")
    parties.append(_bloc_habitudes(vue))
    return "".join(parties)


def _fiche_profil(profil: dict[str, Any], vue: dict[str, Any]) -> str:
    abonnement = profil.get("abonnement_tc")
    lignes = [
        ("Âge", str(profil.get("age") or "—")),
        ("Occupation", str(profil.get("occupation") or "—")),
        ("Revenu du ménage", str(profil.get("revenu") or "—")),
        ("Résidence", f'{profil.get("commune") or "—"} ({profil.get("zone") or "—"})'),
        ("Voiture", str(profil.get("voiture") or "—")),
        ("Vélo", str(profil.get("velo") or "—")),
        ("Abonnement TC", "oui" if abonnement else "non"),
        ("Motifs au programme", ", ".join(libelle_motif(m) for m in profil.get("motifs") or [])
         or "—"),
        ("Trajets retenus", str(len(vue["trajets"]))),
    ]
    items = "".join(f"<li><b>{escape(cle)}</b> : {escape(valeur)}</li>" for cle, valeur in lignes)
    return f'<ul class="mx-profil">{items}</ul>'


def _bloc_concepts(vue: dict[str, Any]) -> str:
    liste = vue["concepts"]
    if not liste:
        # Contract F6: an empty section is declared. In this repository, the absence of
        # a measure readily produces the perfect score; this silence is refused.
        return ('<div class="mx-vide"><b>Aucun concept en mémoire longue pour cet agent.</b> '
                'Ce n\'est pas un résultat propre : c\'est une absence de mesure. '
                'Ni redondance, ni confiance, ni force ne peuvent être calculées ici.</div>')
    redondance = _pct(None if vue["redondance"] is None else 100 * vue["redondance"], 0)
    resume = (f'<p class="mx-note">{len(liste)} concept(s) pour {len(vue["themes"])} '
              f'thème(s) — redondance {redondance}.</p>')
    blocs = [resume]
    for theme in sorted(vue["themes"], key=lambda t: (-len(t.concepts), t.libelle)):
        premier = theme.concepts[0]
        reformulations = "".join(
            f"<li>{escape(c.enonce)} <span class='mx-n'>"
            f"obs {c.observations} · contre {c.contre_exemples} · "
            f"confiance {mesures.confiance(c.observations, c.contre_exemples):.2f} · "
            f"force {c.force:.1f} j · rappels {c.rappels}</span></li>"
            for c in theme.concepts[1:])
        suite = (f'<ul class="mx-reformulations">{reformulations}</ul>'
                 if reformulations else "")
        blocs.append(
            f'<div class="mx-theme"><b>{escape(theme.libelle)}</b>'
            f'<span class="mx-n">{len(theme.concepts)} formulation(s) · '
            f'observations {theme.observations} · contre-exemples {theme.contre_exemples} · '
            f'confiance {mesures.confiance(premier.observations, premier.contre_exemples):.2f} · '
            f'force {premier.force:.1f} j · rappels {theme.rappels}</span>{suite}</div>')
    return "".join(blocs)


def _bloc_habitudes(vue: dict[str, Any]) -> str:
    parties = []
    habitudes = vue["habitudes"]
    if habitudes:
        lignes = []
        for cle in sorted(habitudes):
            valeur = habitudes[cle] or {}
            modes = valeur.get("modes") or {}
            detail = " · ".join(f"{escape(str(m))} {n}" for m, n in sorted(modes.items()))
            lignes.append([escape(cle), detail or "—",
                           str(valeur.get("total", 0)), str(valeur.get("retards", 0))])
        parties.append(_tableau([("Motif | créneau", False), ("Modes", False),
                                 ("Total", True), ("Retards", True)], lignes))
    else:
        parties.append('<div class="mx-vide">Journal des habitudes vide : aucune arrivée '
                       'n\'a été rangée sous un motif. Une absence, pas une régularité.</div>')
    auto = vue["autoreflexions"]
    if auto:
        jour, texte = auto[-1]
        parties.append(f'<p class="mx-note">Dernière auto-réflexion long terme '
                       f'({escape(jour)}), sur {len(auto)} au total :</p>')
        parties.append(f'<div class="mx-citation">{escape(texte)}</div>')
    else:
        parties.append('<div class="mx-vide">Aucune auto-réflexion long terme pour cet '
                       'agent sur ce run.</div>')
    return "".join(parties)


# ── Cross-cutting diagnosis ──────────────────────────────────────────────────

def diagnostic(run: Run, contexte: dict[str, Any]) -> str:
    contamination = mesures.contamination(run)
    croyances = mesures.croyances(sources.blocs_reflexion(run.echanges))
    intro = ("<p>Ce que la mémoire a appris se juge d'abord sur ce qu'on lui a donné "
             "à voir. Trois contrôles, sur les fichiers du run.</p>")
    parties = ["<h2>Diagnostic transversal</h2>", intro]

    parties.append("<h3>Contamination des observations</h3>")
    part_marches = (100.0 * contamination.marches_fantomes / contamination.marches_examinees
                    if contamination.marches_examinees else None)
    part_tc = (100.0 * contamination.tc_anonymes / contamination.tc_total
               if contamination.tc_total else None)
    parties.append("".join([
        '<div class="mx-cartouche">',
        _fait(f"{contamination.marches_fantomes} / {contamination.marches_examinees}",
              "marches fantômes (durée = retard au départ)"),
        _fait(_pct(part_marches, 0), "part des marches à destination vide"),
        _fait(f"{contamination.tc_anonymes} / {contamination.tc_total}",
              "trajets « Unknown Unknown » en transport collectif"),
        _fait(_pct(part_tc, 0), "part des observations TC anonymes"),
        _fait(str(contamination.voiture_en_tc),
              "trajets voiture journalisés en transport collectif"),
        "</div>"]))
    parties.append(
        '<p class="mx-note">Règle appliquée : une marche est dite fantôme quand son '
        'événement porte une destination vide et que sa durée, arrondie à la minute comme '
        'le fait le gabarit, égale exactement le retard au départ du trajet qui la contient. '
        f'{contamination.marches_sans_arrivee} marche(s) n\'ont pu être rattachées à aucun '
        'trajet : elles ne sont comptées ni d\'un côté ni de l\'autre.</p>')
    lignes = []
    for agent in run.agents:
        compteurs = contamination.par_agent.get(agent) or {}
        lignes.append([escape(f'{contexte[agent]["profil"]["nom"]} ({agent})'),
                       str(compteurs.get("marches_fantomes", 0)),
                       str(compteurs.get("tc_anonymes", 0)),
                       str(compteurs.get("voiture_en_tc", 0))])
    parties.append(_tableau([("Agent", False), ("Marches fantômes", True),
                             ("TC anonymes", True), ("Voiture en TC", True)], lignes))

    parties.append("<h3>Croyances servies à la réflexion</h3>")
    if croyances.blocs:
        parties.append(
            f'<p>{croyances.vides} bloc(s) de réflexion sur {croyances.blocs} sont arrivés '
            f'avec un <code>known_beliefs</code> vide, soit {_pct(croyances.taux_vide, 0)}. '
            f'Un bloc vide ne laisse rien à confirmer ni à contredire : la redondance qui '
            f'suit se produit en amont du modèle.</p>')
        lignes = [[escape(f'{contexte.get(agent, {}).get("profil", {}).get("nom", agent)} ({agent})'),
                   str(total), str(vides),
                   _pct(100.0 * vides / total if total else None, 0)]
                  for agent, (total, vides) in croyances.par_agent.items()]
        parties.append(_tableau([("Agent", False), ("Blocs", True), ("Vides", True),
                                 ("Part vide", True)], lignes))
    else:
        parties.append('<div class="mx-vide">Aucun bloc de réflexion lisible dans '
                       '<code>llm_exchanges.jsonl</code> : le taux de croyances vides ne '
                       'se mesure pas ici.</div>')

    parties.append("<h3>Concentration des rappels</h3>")
    lignes = []
    for agent in run.agents:
        rappels = contexte[agent]["rappels"]
        lignes.append([escape(f'{contexte[agent]["profil"]["nom"]} ({agent})'),
                       str(rappels.evenements), str(rappels.servis),
                       _pct(rappels.concentration, 0),
                       escape(rappels.top[0][0][:60]) if rappels.top else "—",
                       str(rappels.top[0][1]) if rappels.top else "—"])
    parties.append(_tableau([("Agent", False), ("Rappels", True), ("Souvenirs servis", True),
                             ("Part du top 10", True), ("Souvenir le plus servi", False),
                             ("Fois", True)], lignes))
    parties.append('<p class="mx-note">Un rappel augmente la force du souvenir, donc sa '
                   'probabilité d\'être rappelé. Une part du top 10 proche de 100 % dit que '
                   'la mémoire ne grandit plus : elle tourne.</p>')
    return "".join(parties)


# ── Assembly ──────────────────────────────────────────────────────────────────

def _contexte(run: Run) -> dict[str, Any]:
    contexte: dict[str, Any] = {}
    for agent in run.agents:
        trajets = [t for t in run.trajets if t.person_id == agent]
        appariement = mesures.apparier(trajets, [b for b in run.blocs if b.agent == agent])
        concepts = mesures.concepts(run.ltm.get(agent, []))
        contexte[agent] = {
            "profil": sources.profil(run.personas.get(agent, {})),
            "trajets": trajets,
            "parts": mesures.parts_modales(trajets),
            "appariement": appariement,
            "tableaux": mesures.tableaux_itineraires(trajets, appariement),
            "frise": mesures.frise(trajets, run.jours),
            "semaines": mesures.semaines(trajets, run.jours),
            "concepts": concepts,
            "themes": mesures.themes(concepts),
            "redondance": mesures.taux_redondance(concepts),
            "journal": run.journaux.get(agent),
            "rappels": mesures.rappels(run.journaux.get(agent)),
            "habitudes": run.habitudes.get(agent) or {},
            "autoreflexions": run.autoreflexions.get(agent) or [],
            "raisons_experience": mesures.part_raisons_experience(trajets),
        }
    return contexte


def construire(run: Run) -> str:
    """The full document, as one string. No generation timestamp (F7)."""
    contexte = _contexte(run)
    corps = [entete(run), synthese(run, contexte)]
    for agent in run.agents:
        corps.append(section_persona(run, agent, contexte[agent]))
    corps.append(diagnostic(run, contexte))
    return (
        "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>Mémoire des agents — {escape(run.chemin.name)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f'<main class="mx-page">{"".join(corps)}</main>\n</body>\n</html>\n')


def ecrire(chemin_run: Path | str, sortie: Path | str) -> Path:
    run = sources.charger(chemin_run)
    cible = Path(sortie)
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(construire(run), encoding="utf-8")
    return cible


def main(argv: Sequence[str] | None = None) -> int:
    analyseur = argparse.ArgumentParser(
        prog="python -m scripts.analysis.memoire.rapport",
        description="Self-contained HTML report on a memory run, one section per persona.")
    analyseur.add_argument("run", help="dossier du run (ex. experiments/archive/2026-09-14_23_58)")
    analyseur.add_argument("-o", "--out", required=True, help="output HTML file")
    arguments = analyseur.parse_args(argv)
    try:
        cible = ecrire(arguments.run, arguments.out)
    except FileNotFoundError as erreur:
        print(f"[ERREUR] {erreur}", file=sys.stderr)
        return 2
    taille = cible.stat().st_size
    print(f"[OK] report written: {cible} ({taille} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
