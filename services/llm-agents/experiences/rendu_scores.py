"""Synthesis page of a run: composite score + per-subcategory detail.

Spec: R5, R7, R9, R15, R16.

The per-subcategory cells are rendered by the **same** ``_dimension_blocks``
as the historical `docs/synthesis` page (cf. `docs/arch/score-synthesis.md`): they
cannot diverge. Only the dressing — composite header, scoring formula and its
fingerprint, coverage, volet label — belongs to the experiment platform.

Volet 1 (LLM decision-maker) and volet 3 (model decision-maker) share this rendering; they
differ only by the header (the named decision-maker, and the model's SHA in volet 3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiences import formule as F
from experiences import score as S
from experiences.chemins import racine_depot

# Anchor rather than counting levels — cf. `experiences.chemins` (ticket 039).
REPO_ROOT = racine_depot()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.synthesis import charts
from scripts.synthesis.render import (
    CSS,
    _dimension_blocks,
    escape,
    missing_card,
    tiles,
)

F_SCORES = "scores.json"
F_SYNTHESE = "synthese.json"
F_PAGE = "synthese_scores.html"


def _num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _en_tete_decideur(scores: dict, synthese: dict) -> str:
    """Describes the decision-maker; in volet 3, shows the model's SHA (R15)."""
    dec = (synthese.get("empreintes") or {}).get("decideur") or {}
    if scores.get("volet") == "3":
        return (
            f"Volet 3 — Modèle statistique · artefact "
            f"<code>{escape((dec.get('sha256') or '')[:12])}</code>"
        )
    modele = dec.get("modele") or dec.get("type") or "décideur"
    return f"Volet 1 — Décideur LLM · <code>{escape(str(modele))}</code>"


def _tuile_journal(perimetre: dict | None) -> tuple[str, str, str]:
    """Ticket 081 — the check that authorised this score, visible on the page that publishes it.

    The page is self-contained: someone citing a composite score must be able to check, without
    opening a log or a terminal, that the scored file did cover the archived decisions.
    A score published on 2026-09-12 covered 274 rows for 3,154 decisions, and nothing on the
    page said so.
    """
    if not perimetre:
        # Score older than the rule: do not show "checked" for a check that did not
        # take place. In this repo, the absence of a measurement too readily passes for
        # a perfect result.
        return ("Journal vérifié", "—", "score antérieur au contrôle")
    lignes, decides = perimetre.get("lignes_journal"), perimetre.get("decides")
    if perimetre.get("complet") is None:
        return ("Journal vérifié", "—", f"{lignes} ligne(s), décisions non comparables")
    relatif = perimetre.get("ecart_relatif") or 0.0
    return (
        "Journal vérifié",
        "complet" if perimetre.get("complet") else "INCOMPLET",
        f"{lignes} ligne(s) pour {decides} décisions ({relatif * 100:+.1f} %)",
    )


def _bandeau_statut(st: dict | None) -> str:
    """Top banner when the experiment is invalidated or archived (hygiene spec §3.2).

    The figure stays displayed — it is exact, it comes from the sealed archive. What the banner
    says is not to cite it without its reason.
    """
    if not st or st.get("statut") == "actif":
        return ""
    libelle = {
        "invalide": "mesure invalidée",
        "archivee": "expérience archivée",
    }.get(st["statut"], st["statut"])
    motif = escape(str(st.get("motif") or "sans motif consigné"))
    quand = escape(str(st.get("le") or "?"))
    ref = st.get("reference")
    suite = f" · <code>{escape(str(ref))}</code>" if ref else ""
    return (
        f'<p class="bandeau-statut"><strong>{libelle}</strong> le {quand} — {motif}{suite}<br>'
        "Les chiffres ci-dessous sont ceux de l'archive, inchangés : "
        "ils ne doivent pas être cités sans ce motif.</p>"
    )


def rendu(
    scores: dict,
    synthese: dict,
    registre: F.RegistreFormules | None = None,
    statut: dict | None = None,
) -> str:
    """Produces the self-contained HTML of a run's synthesis page."""
    registre = registre or F.charger()
    volet = scores.get("volet", "1")
    prefix = "mod" if volet == "3" else "sim"
    comp = scores.get("composite") or {}
    couv = scores.get("couverture") or {}
    formule = scores.get("formule") or {}
    gview = scores.get("global") or {}
    non_mesurees = scores.get("dimensions_non_mesurees") or []

    perimee = not S.est_reference(scores, registre)
    badge = (
        '<span class="badge">⚠ formule périmée</span>'
        if perimee
        else '<span class="badge ok">formule de référence</span>'
    )

    taux = couv.get("taux")
    taux_txt = "—" if taux is None else f"{taux * 100:.1f}%"
    couv_txt = f"{couv.get('decides', '—')} / {couv.get('attendus', '—')} décisions"

    # Header: composite (both losses), global L1, coverage (R9), and the two quantities
    # of ticket 047 — the second reading of the composite and the count that explains it.
    # They are in the header and not in an appendix: the main composite counts
    # decisions nobody made, and nothing else on this page says so.
    forces = scores.get("choix_forces") or {}
    part_forces = forces.get("part")
    forces_txt = (
        "—"
        if forces.get("n") is None
        else f"{forces['n']} / {forces.get('n_scorees', '—')} décisions scorées"
    )
    entete_tiles = tiles(
        [
            ("Composite (EMD·JSD)", _num(comp.get("emd_jsd")), "toutes décisions"),
            (
                "Composite hors it. unique",
                _num(comp.get("emd_jsd_hors_choix_unique")),
                "ce que le décideur a décidé",
            ),
            ("L1 Composite", _num(comp.get("l1"), 1), "points de %"),
            ("Composite (tiré)", _num(comp.get("emd_jsd_tire")), "après tirage"),
            (
                "Choix forcés",
                "—" if part_forces is None else f"{part_forces * 100:.1f}%",
                forces_txt,
            ),
            ("Couverture", taux_txt, couv_txt),
            _tuile_journal(scores.get("perimetre_verifie")),
        ]
    )

    alerte = ""
    if non_mesurees:
        # R8 — never a silent 0: the unmeasured dimensions are cited.
        alerte = (
            f'<p class="warn">Dimension(s) non mesurée(s) — repli vers la perte '
            f"maximale, pas un score parfait : "
            f"<strong>{escape(', '.join(non_mesurees))}</strong>.</p>"
        )
    # Ticket 047 — the page is self-contained: it is read without the scoring log, so it
    # carries the gap between the two readings itself when that gap is decisive.
    principal, seconde = comp.get("emd_jsd"), comp.get("emd_jsd_hors_choix_unique")
    if (
        principal is not None
        and seconde is not None
        and abs(seconde - principal) >= 1.0
    ):
        alerte += (
            f'<p class="warn">Ce composite <strong>dépend des décisions à itinéraire '
            f"unique</strong> — une seule option existait, personne n'a choisi. En les "
            f"retirant, il passe de <strong>{principal:.2f}</strong> à "
            f"<strong>{seconde:.2f}</strong> ({seconde - principal:+.2f}). Leur nombre "
            f"dépend du bras : les deux lectures ne classent pas les bras dans le même "
            f"ordre, et aucune des deux n'est fausse.</p>"
        )

    global_bloc = ""
    if gview:
        global_bloc = f"<h3>Parts modales globales</h3>{charts.global_bullet(gview)}"

    details = {
        k: (v or {}).get("strates") for k, v in (scores.get("detail") or {}).items()
    }
    blocs = (
        _dimension_blocks(details, prefix)
        if details
        else missing_card(
            "Détail par sous-catégorie",
            "Aucun détail par strate dans ce résultat.",
            [],
            "",
        )
    )

    formule_ligne = (
        f"Formule <strong>{escape(formule.get('nom', '?'))}</strong> "
        f"<code>{escape((formule.get('sha256') or '')[:12])}</code> {badge}"
    )

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scores — {escape(str(scores.get("experience") or ""))} / {escape(str(scores.get("execution") or ""))}</title>
<style>{CSS}
.solo{{max-width:1040px;margin:0 auto;padding:32px 40px 96px}}
.solo h1{{font-size:21px;font-weight:500;margin:0 0 4px;letter-spacing:-.015em}}
.solo .sub{{font-size:13px;color:var(--ink3);margin-bottom:6px}}
.warn{{color:var(--warn);font-size:13px;margin:10px 0}}
.bandeau-statut{{border-left:3px solid var(--warn);background:rgba(180,120,0,.07);
  padding:10px 14px;margin:0 0 18px;font-size:13px;line-height:1.55}}
@media(max-width:880px){{.solo{{padding:24px 20px 64px}}}}
</style></head>
<body><main class="solo">
<h1>{escape(str(scores.get("experience") or ""))} — {escape(str(scores.get("execution") or ""))}</h1>
{_bandeau_statut(statut)}
<div class="sub">{_en_tete_decideur(scores, synthese)}</div>
<div class="sub">{formule_ligne}</div>
<p style="color:var(--ink3);font-size:12.5px;margin:6px 0 18px">
Généré le {escape(scores.get("generate_le") or scores.get("genere_le") or "")} ·
composite importé du moteur de calibration (loss non réimplémentée).</p>
{entete_tiles}
{alerte}
{global_bloc}
<h3>Détail par sous-catégorie</h3>
<p>Une cellule par sous-catégorie&nbsp;: parts modales observées face à l'enquête EMC²
2023 (repère pointillé). <strong>Plus l'écart au repère est faible, meilleur c'est.</strong>
Les dimensions <span class="badge ok">dans le composite</span> entrent dans le score&nbsp;;
<span class="badge">hors composite</span> sont rapportées pour lecture seule.</p>
{blocs}
<footer>Recalculer&nbsp;: <code>python -m experiences score --toutes</code> ·
Chiffres produits par le même moteur que <code>docs/synthesis</code>.</footer>
</main></body></html>"""


def ecrire(
    dossier: str | Path, registre: F.RegistreFormules | None = None
) -> Path | None:
    """Reads a run's scores.json + synthese.json, writes synthese_scores.html.

    Returns None if the run has no scores.json yet (not scored / partial)."""
    dossier = Path(dossier)
    scores_path = dossier / F_SCORES
    if not scores_path.exists():
        return None
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    synthese = json.loads((dossier / F_SYNTHESE).read_text(encoding="utf-8"))
    # The status lives at experiment level: two levels above the run directory.
    from experiences import statut as ST

    st = ST.lire(dossier.parent.parent)
    page = rendu(scores, synthese, registre, statut=st)
    cible = dossier / F_PAGE
    cible.write_text(page, encoding="utf-8")
    return cible


__all__ = ["ecrire", "rendu"]
