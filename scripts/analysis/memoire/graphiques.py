"""Inline SVG charts, with no dependency and no script.

Same constraint as `scripts/synthesis/charts.py`: the page opens from a
``file://`` and reads offline. No CDN, no chart library, no
remote font — only hand-written SVG and inline CSS.

Four forms, chosen for what the data have to say:

* **mode timeline** — one row per purpose, one column per day; the cell carries
  the colour of the chosen mode, and splits when the purpose has several trips
  that day. It is the form that shows a habit settling in;
* **itinerary table** — one row per distinct proposed itinerary, one
  column per day: light blue proposed, dark blue chosen, grey when
  the matching failed. It is the centrepiece: it says what the agent
  had before its eyes, not only what it did;
* **weekly curve** — decisiveness and entropy, two scales, a line
  broken on weeks without a decision;
* **recall bars** — the most served memories, to see the loop.
"""
from __future__ import annotations

from collections.abc import Sequence
from html import escape

from .mesures import LigneItineraire, Rappels, Semaine, TableauItineraires
from .sources import MODE_COULEURS, MODE_LIBELLES, libelle_motif

# Blues of the itinerary table: proposed / chosen / missed matching.
BLEU_PROPOSE = "#BBD5F0"
BLEU_RETENU = "#1A4C8B"
GRIS_MANQUANT = "#D8D8D8"

_CASE = 18
_HAUTEUR_LIGNE = 22
_LARGEUR_LIBELLE = 190


def legende(items: Sequence[tuple[str, str]], note: str = "") -> str:
    """Swatch + label, in text ink (never the series colour)."""
    chips = "".join(
        f'<span class="mx-chip"><i style="background:{color}"></i>{escape(label)}</span>'
        for label, color in items)
    if note:
        chips += f'<span class="mx-chip mx-chip-note">{escape(note)}</span>'
    return f'<div class="mx-legend">{chips}</div>'


def legende_modes(modes: Sequence[str]) -> str:
    return legende([(MODE_LIBELLES.get(m, m), MODE_COULEURS.get(m, "#9AA0A6")) for m in modes])


def _etiquettes_jours(jours: Sequence[str]) -> list[tuple[int, str]]:
    """One date tick every n days, n chosen so that nothing overlaps."""
    if not jours:
        return []
    pas = 1 if len(jours) <= 10 else (2 if len(jours) <= 20 else 4)
    reperes = []
    for i, jour in enumerate(jours):
        if i % pas:
            continue
        reperes.append((i, jour[8:10] + "/" + jour[5:7]))
    return reperes


def _cadre(largeur: int, hauteur: int) -> str:
    return (f'<svg viewBox="0 0 {largeur} {hauteur}" width="{largeur}" height="{hauteur}" '
            f'role="img" preserveAspectRatio="xMinYMin meet" class="mx-svg">')


def frise_modes(lignes: Sequence[tuple[str, dict[str, list[str]]]],
                jours: Sequence[str]) -> str:
    """One row per purpose, one column per day, the colour of the chosen mode.

    Several trips on the same day for the same purpose: the cell splits into
    as many vertical bands. Overwriting the second trip would make the
    returns home disappear, which are half of the trips.
    """
    if not lignes or not jours:
        return '<p class="mx-vide">Aucun trajet : pas de frise à tracer.</p>'
    largeur = _LARGEUR_LIBELLE + _CASE * len(jours) + 8
    hauteur = _HAUTEUR_LIGNE * len(lignes) + 30
    parts = [_cadre(largeur, hauteur)]
    for rang, (motif, par_jour) in enumerate(lignes):
        y = rang * _HAUTEUR_LIGNE + 4
        parts.append(f'<text class="mx-label" x="0" y="{y + 13}">'
                     f'{escape(libelle_motif(motif))}</text>')
        for colonne, jour in enumerate(jours):
            x = _LARGEUR_LIBELLE + colonne * _CASE
            modes = par_jour.get(jour) or []
            if not modes:
                parts.append(f'<rect class="mx-case-vide" x="{x + 1}" y="{y + 1}" '
                             f'width="{_CASE - 2}" height="{_HAUTEUR_LIGNE - 4}" rx="2"/>')
                continue
            tranche = (_CASE - 2) / len(modes)
            for i, mode in enumerate(modes):
                couleur = MODE_COULEURS.get(mode, "#9AA0A6")
                parts.append(
                    f'<rect x="{x + 1 + i * tranche:.2f}" y="{y + 1}" '
                    f'width="{tranche:.2f}" height="{_HAUTEUR_LIGNE - 4}" '
                    f'fill="{couleur}"><title>{escape(jour)} · '
                    f'{escape(libelle_motif(motif))} · '
                    f'{escape(MODE_LIBELLES.get(mode, mode))}</title></rect>')
    axe_y = hauteur - 10
    for index, etiquette in _etiquettes_jours(jours):
        x = _LARGEUR_LIBELLE + index * _CASE + _CASE / 2
        parts.append(f'<text class="mx-tick" x="{x:.1f}" y="{axe_y}" '
                     f'text-anchor="middle">{escape(etiquette)}</text>')
    parts.append("</svg>")
    modes_vus = sorted({m for _, par_jour in lignes for modes in par_jour.values()
                        for m in modes})
    return (f'<div class="mx-scroll">{"".join(parts)}</div>' + legende_modes(modes_vus))


def tableau_itineraires(tableau: TableauItineraires, jours: Sequence[str]) -> str:
    """Itinerary × day grid: proposed, chosen, missed matching.

    An empty cell says "this itinerary was not proposed that day". A grey
    cell says "a trip took place and what was proposed to it was not
    found" — it is a declared measurement gap, not a silence (contract F3).
    """
    if not tableau.lignes and not tableau.jours_gris:
        return ('<p class="mx-vide">Aucun itinéraire retrouvé pour ce motif : '
                'aucun prompt de décision ne lui correspond.</p>')
    lignes = tableau.lignes
    largeur = _LARGEUR_LIBELLE + _CASE * len(jours) + 8
    # A dedicated row carries the days on which the matching failed: the grey cell
    # shows even when NO itinerary was found for this purpose — which is
    # precisely the case where a silent table would lie the most (contract F3).
    ligne_grise = 1 if tableau.jours_gris else 0
    hauteur = _HAUTEUR_LIGNE * max(1, len(lignes) + ligne_grise) + 30
    parts = [_cadre(largeur, hauteur)]
    if ligne_grise:
        parts.append(f'<text class="mx-label" x="0" y="{17}">appariement manqué</text>')
        for colonne, jour in enumerate(jours):
            if jour not in tableau.jours_gris:
                continue
            x = _LARGEUR_LIBELLE + colonne * _CASE
            parts.append(
                f'<rect x="{x + 1}" y="{5}" width="{_CASE - 2}" '
                f'height="{_HAUTEUR_LIGNE - 4}" rx="2" fill="{GRIS_MANQUANT}">'
                f'<title>{escape(jour)} · option non retrouvée dans les prompts</title></rect>')
    for rang, ligne in enumerate(lignes):
        y = (rang + ligne_grise) * _HAUTEUR_LIGNE + 4
        parts.append(f'<text class="mx-label" x="0" y="{y + 13}">'
                     f'{escape(_etiquette_ligne(rang, ligne))}</text>')
        for colonne, jour in enumerate(jours):
            x = _LARGEUR_LIBELLE + colonne * _CASE
            etat = ligne.cases.get(jour)
            if etat is None:
                parts.append(f'<rect class="mx-case-vide" x="{x + 1}" y="{y + 1}" '
                             f'width="{_CASE - 2}" height="{_HAUTEUR_LIGNE - 4}" rx="2"/>')
                continue
            couleur = BLEU_RETENU if etat == "retenu" else BLEU_PROPOSE
            mot = "retenu" if etat == "retenu" else "proposé"
            parts.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{_CASE - 2}" '
                f'height="{_HAUTEUR_LIGNE - 4}" rx="2" fill="{couleur}">'
                f'<title>{escape(jour)} · {mot} · {escape(ligne.mode_brut)}</title></rect>')
    axe_y = hauteur - 10
    for index, etiquette in _etiquettes_jours(jours):
        x = _LARGEUR_LIBELLE + index * _CASE + _CASE / 2
        parts.append(f'<text class="mx-tick" x="{x:.1f}" y="{axe_y}" '
                     f'text-anchor="middle">{escape(etiquette)}</text>')
    parts.append("</svg>")

    note = ""
    if tableau.trajets_non_apparies:
        note = (f"{tableau.trajets_non_apparies} trajet(s) sur {tableau.trajets} sans option "
                f"retrouvée dans les prompts — colonnes grises")
    bloc = [f'<div class="mx-scroll">{"".join(parts)}</div>',
            legende([("proposé ce jour-là", BLEU_PROPOSE),
                     ("retenu ce jour-là", BLEU_RETENU),
                     ("appariement manqué", GRIS_MANQUANT)], note),
            _detail_itineraires(lignes)]
    return "".join(bloc)


def _etiquette_ligne(rang: int, ligne: LigneItineraire) -> str:
    mode = ligne.mode_brut if len(ligne.mode_brut) <= 20 else ligne.mode_brut[:19] + "…"
    return f"[{rang + 1}] {mode}"


def _detail_itineraires(lignes: Sequence[LigneItineraire]) -> str:
    """The full text of each itinerary, under the grid: nothing is truncated."""
    if not lignes:
        return ""
    items = []
    for rang, ligne in enumerate(lignes):
        etapes = "".join(f"<li>{escape(e)}</li>" for e in ligne.etapes)
        detail = f"<ul class='mx-etapes'>{etapes}</ul>" if etapes else ""
        items.append(
            f'<li><b>[{rang + 1}] {escape(ligne.mode_brut)}</b> — '
            f'{escape(ligne.description) or "—"} '
            f'<span class="mx-n">proposé {ligne.proposes} j · retenu {ligne.retenus} j</span>'
            f'{detail}</li>')
    return f'<ol class="mx-itineraires">{"".join(items)}</ol>'


def courbe_semaines(semaines: Sequence[Semaine]) -> str:
    """Decisiveness (0–1) and entropy (bits), one value per week.

    Two series, two scales: decisiveness fits in [0, 1], entropy rises
    with the number of options. A week without a probabilised decision breaks the
    line — a point at zero would read as perfect certainty, whereas there
    was nothing to measure.
    """
    if not semaines:
        return '<p class="mx-vide">Aucune semaine mesurable.</p>'
    largeur, hauteur = 620, 210
    marge_g, marge_h, marge_b, marge_d = 42, 16, 38, 42
    plot_l = largeur - marge_g - marge_d
    plot_h = hauteur - marge_h - marge_b
    pas = plot_l / max(1, len(semaines) - 1)
    entropies = [s.entropie for s in semaines if s.entropie is not None]
    haut_entropie = max(2.0, max(entropies) if entropies else 2.0)

    parts = [_cadre(largeur, hauteur)]
    for frac in (0.0, 0.5, 1.0):
        y = marge_h + plot_h * (1 - frac)
        parts.append(f'<line class="mx-grid" x1="{marge_g}" y1="{y:.1f}" '
                     f'x2="{marge_g + plot_l}" y2="{y:.1f}"/>')
        parts.append(f'<text class="mx-axis" x="{marge_g - 6}" y="{y + 3:.1f}" '
                     f'text-anchor="end">{frac:.1f}</text>')
        parts.append(f'<text class="mx-axis" x="{marge_g + plot_l + 6}" y="{y + 3:.1f}">'
                     f'{haut_entropie * frac:.1f}</text>')

    def trace(valeurs: list[float | None], plafond: float, couleur: str,
              pointille: bool) -> None:
        segment: list[str] = []
        for i, valeur in enumerate(valeurs):
            if valeur is None:
                if len(segment) >= 2:
                    parts.append(_polyline(segment, couleur, pointille))
                segment = []
                continue
            x = marge_g + i * pas
            y = marge_h + plot_h * (1 - min(valeur, plafond) / plafond)
            segment.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{couleur}">'
                         f'<title>{escape(semaines[i].libelle)} — {valeur:.2f}</title></circle>')
        if len(segment) >= 2:
            parts.append(_polyline(segment, couleur, pointille))

    trace([s.decisivite for s in semaines], 1.0, "#1A4C8B", False)
    trace([s.entropie for s in semaines], haut_entropie, "#C2571A", True)

    for i, semaine in enumerate(semaines):
        x = marge_g + i * pas
        etiquette = semaine.libelle if len(semaines) <= 12 or i % 2 == 0 else ""
        if etiquette:
            parts.append(f'<text class="mx-tick" x="{x:.1f}" y="{hauteur - 20}" '
                         f'text-anchor="middle">{escape(etiquette)}</text>')
        if semaine.n == 0:
            parts.append(f'<text class="mx-tick mx-tick-bas" x="{x:.1f}" y="{hauteur - 8}" '
                         f'text-anchor="middle">n=0</text>')
    parts.append("</svg>")
    return (f'<div class="mx-scroll">{"".join(parts)}</div>'
            + legende([("décisivité (gauche, 0–1)", "#1A4C8B"),
                       ("entropie (droite, bits)", "#C2571A")]))


def _polyline(points: Sequence[str], couleur: str, pointille: bool) -> str:
    tirets = ' stroke-dasharray="5 3"' if pointille else ""
    return (f'<polyline points="{" ".join(points)}" fill="none" stroke="{couleur}" '
            f'stroke-width="2" stroke-linejoin="round"{tirets}/>')


def barres_rappels(mesure: Rappels, maximum: int = 10) -> str:
    """The most served memories: the recall loop shows or does not show."""
    if not mesure.servis:
        return ('<p class="mx-vide">Aucun rappel journalisé : la concentration ne '
                'se mesure pas, elle n\'est pas nulle.</p>')
    top = mesure.top[:maximum]
    hauteur_ligne = 24
    largeur = 620
    libelle_l = 250
    piste = largeur - libelle_l - 54
    haut = max(n for _, n in top)
    hauteur = hauteur_ligne * len(top) + 10
    parts = [_cadre(largeur, hauteur)]
    for rang, (incipit, compte) in enumerate(top):
        y = rang * hauteur_ligne + 4
        texte = incipit if len(incipit) <= 42 else incipit[:41] + "…"
        parts.append(f'<text class="mx-label" x="0" y="{y + 13}">{escape(texte)}'
                     f'<title>{escape(incipit)}</title></text>')
        parts.append(f'<rect class="mx-track" x="{libelle_l}" y="{y + 3}" '
                     f'width="{piste}" height="13" rx="3"/>')
        parts.append(f'<rect x="{libelle_l}" y="{y + 3}" '
                     f'width="{piste * compte / haut:.1f}" height="13" rx="3" '
                     f'fill="{BLEU_RETENU}"/>')
        parts.append(f'<text class="mx-value" x="{libelle_l + piste + 8}" y="{y + 13}">'
                     f'{compte}</text>')
    parts.append("</svg>")
    concentration = mesure.concentration
    note = (f"top 10 = {concentration:.0f} % des {mesure.servis} souvenirs servis"
            if concentration is not None else "")
    return f'<div class="mx-scroll">{"".join(parts)}</div>' + legende([], note)
