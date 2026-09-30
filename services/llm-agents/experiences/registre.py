"""Registry, comparison, reporting (ticket 035, spec 06 E12–E17, E20).

The registry is the reading of `data/experiences/*`: nothing is written there. The summary of a run
is a data file (`synthese.json`) of which the HTML page is only a rendering (E16). The
platform holds **no** survey value: when it compares, it reads the repository's source of truth
and cites path + fingerprint (E17).
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

import yaml
from experiences.archive import (
    F_COMPTEURS,
    F_ETAT,
    F_EXECUTION,
    F_SCORES,
    F_SYNTHESE,
    F_SYNTHESE_HTML,
    METHODE_INEXPLOITABLE,
    METHODE_NON_COUVERT,
    Execution,
)
from experiences.decision import (
    METHODE_CHOIX_UNIQUE,
    METHODE_REPLI_UNIFORME,
    METHODE_SANS_SOLUTION,
)
from experiences.experience import dossier_experiences
from experiences.population import sha256_fichier

ETAT_ARCHIVE_MANQUANTE = "archive manquante"

# Canonical families (mobility_llm.mode_choice) → key in the survey reference data. It is a
# NAME MAPPING, not a value: the values are read from the cited file.
CORRESPONDANCE_REFERENTIEL = {
    "car": "voiture",
    "walking": "marche",
    "cycling": "velo",
    "public_transport": "transports_collectifs",
    "train": "transports_collectifs",
}


def _racine() -> Path:
    # Anchor rather than a count of levels — cf. `experiences.chemins` (ticket 039).
    from experiences.chemins import racine_depot

    return racine_depot()


def chemin_referentiel() -> Path:
    """Source of truth for the survey modal shares; `REFERENTIEL_ENQUETE` points to it in the container."""
    import os

    return Path(
        os.getenv("REFERENTIEL_ENQUETE")
        or (_racine() / "scripts" / "data" / "population" / "cerema_values.yaml")
    )


# ── Registry (E12, E20) ──────────────────────────────────────────────────────


def _lire_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except ValueError:
        return {}


def _decideur_label(dec: dict | None) -> str:
    """`type:modele` (e.g. passerelle:gemini-3.5-flash-lite); empty if there is no type."""
    dec = dec or {}
    t = dec.get("type")
    if not t:
        return ""
    return f"{t}:{dec.get('modele') or ''}".rstrip(":")


def lister(
    dossier: Path | None = None, *, inclure_masquees: bool = False
) -> list[dict]:
    """One row per run (and one `definie` row for an experiment without a run).

    `inclure_masquees` (hygiene spec §3.2): by default, experiments marked `archivee`
    or `invalide` leave the listing — their data stay intact on disk, only the
    visibility changes. Each row carries `statut` and `statut_motif` so that the caller
    can show the reason rather than make the row disappear without explanation.
    """
    from experiences import statut as S

    racine = Path(dossier) if dossier else dossier_experiences()
    lignes: list[dict] = []
    if not racine.is_dir():
        return lignes
    # SHA of the current reference scoring formula, to derive the "stale" flag
    # (R7) without touching scores.json. Best-effort: an unreadable registry must
    # not break the list of experiments.
    ref_sha = None
    try:
        from experiences import formule as _F

        ref_sha = _F.charger().reference.sha256
    except Exception:  # noqa: BLE001 — the registry stays readable even without a formula
        pass
    # Recursive discovery of all experiments (direct or in set sub-folders)
    fichiers_exp = []
    if racine.is_dir():
        for p in racine.rglob("experience.yaml"):
            # Relative to the root (ticket 113): a root under `archive/` lists itself.
            parties = p.relative_to(racine).parts
            if "archive" in parties or ".system_generated" in parties:
                continue
            fichiers_exp.append(p.parent)

    for exp_dir in sorted(fichiers_exp, key=lambda p: p.name):
        try:
            exp = (
                yaml.safe_load(
                    (exp_dir / "experience.yaml").read_text(encoding="utf-8")
                )
                or {}
            )
        except yaml.YAMLError:
            continue
        st = S.lire(exp_dir)
        if S.est_masquee(st) and not inclure_masquees:
            continue
        base = {
            "experience": exp.get("nom", exp_dir.name),
            "mode": exp.get("mode"),
            "decideur": _decideur_label(exp.get("decideur")),
            "gabarit": (exp.get("gabarit") or {}).get("categorie"),
            "derive_de": exp.get("derive_de"),
            "statut": st["statut"],
            "statut_motif": st.get("motif"),
        }
        connues = set(exp.get("executions_connues") or [])
        sur_disque = (
            sorted(p.name for p in (exp_dir / "executions").iterdir())
            if (exp_dir / "executions").is_dir()
            else []
        )
        for nom in sur_disque:
            d = exp_dir / "executions" / nom
            etat = _lire_json(d / F_ETAT)
            compteurs = _lire_json(d / F_COMPTEURS)
            synthese = _lire_json(d / F_SYNTHESE)
            conf = {}
            try:
                conf = (
                    yaml.safe_load((d / F_EXECUTION).read_text(encoding="utf-8")) or {}
                )
            except (yaml.YAMLError, OSError):
                pass
            couv = compteurs.get("couverture") or {}
            scores = _lire_json(d / F_SCORES)
            comp = scores.get("composite") or {}
            f_score = scores.get("formule") or {}
            f_sha = f_score.get("sha256")
            # R6 — decision-maker ACTUALLY used, frozen in the run snapshot, rather than
            # the (mutable) one of the current definition. Falls back on the definition if the
            # snapshot says nothing (old formats).
            dec_fige = (
                _decideur_label((conf.get("experience") or {}).get("decideur"))
                or base["decideur"]
            )
            lignes.append(
                {
                    **base,
                    "decideur": dec_fige,
                    "execution": nom,
                    "etat": etat.get("etat", "?"),
                    "raison": etat.get("raison"),
                    "date": conf.get("cree_le"),
                    "gabarit_sha256": (
                        (conf.get("empreintes") or {}).get("gabarit") or {}
                    ).get("sha256"),
                    "decides": couv.get("decides"),
                    "attendus": couv.get("attendus"),
                    "couverture": couv.get("taux"),
                    "parts_modales": (synthese.get("parts_modales") or {}).get(
                        "pourcent"
                    ),
                    # Ticket 047 — the forced-choice count travels with the shares and with
                    # the composite score, always. TWO counts, because there are two
                    # scopes and they differ: the run's goes with the
                    # coverage and the shares, the score's goes with the composite score. A
                    # single figure for both would be misread half of the time.
                    "choix_forces": (synthese.get("choix_forces") or {}).get("n"),
                    "part_forces": (synthese.get("choix_forces") or {}).get("part"),
                    "choix_forces_score": (scores.get("choix_forces") or {}).get("n"),
                    # Scores (R5, R9): None → "—" on display, never 0.
                    "composite_emd": comp.get("emd_jsd"),
                    "composite_l1": comp.get("l1"),
                    "composite_emd_hors_forces": comp.get("emd_jsd_hors_choix_unique"),
                    "composite_l1_hors_forces": comp.get("l1_hors_choix_unique"),
                    "volet": scores.get("volet"),
                    "formule": f_score.get("nom"),
                    "formule_sha256": f_sha,
                    # R7 — flag derived on read: stale if SHA ≠ reference.
                    "formule_perimee": bool(f_sha and ref_sha and f_sha != ref_sha),
                    "dossier": str(d),
                }
            )
        for nom in sorted(connues - set(sur_disque)):
            lignes.append(
                {
                    **base,
                    "execution": nom,
                    "etat": ETAT_ARCHIVE_MANQUANTE,
                    "raison": "dossier disparu",
                    "date": None,
                    "decides": None,
                    "attendus": None,
                    "couverture": None,
                    "parts_modales": None,
                    "dossier": str(exp_dir / "executions" / nom),
                }
            )
        if not sur_disque and not connues:
            lignes.append(
                {
                    **base,
                    "execution": None,
                    "etat": "definie",
                    "raison": None,
                    "date": None,
                    "decides": None,
                    "attendus": None,
                    "couverture": None,
                    "parts_modales": None,
                    "dossier": str(exp_dir),
                }
            )
    return lignes


def trier_filtrer(
    lignes: list[dict],
    *,
    trier: str | None = None,
    decroissant: bool = False,
    filtres: dict[str, str] | None = None,
) -> list[dict]:
    """E12 — sort on a field, filter `field=substring` (case-insensitive)."""
    out = lignes
    for champ, valeur in (filtres or {}).items():
        out = [l for l in out if valeur.lower() in str(l.get(champ, "")).lower()]
    if trier:
        out = sorted(
            out,
            key=lambda l: (
                l.get(trier) is None,
                l.get(trier) if l.get(trier) is not None else "",
            ),
            reverse=decroissant,
        )
    return out


def formater_table(lignes: list[dict], colonnes: list[str] | None = None) -> str:
    colonnes = colonnes or [
        "experience",
        "execution",
        "etat",
        "decideur",
        "mode",
        "couverture",
        "choix_forces",
        "composite_emd",
        # Ticket 047 — the second reading is shown BY DEFAULT, next to the first.
        # Hidden behind an option, it would not be read, and it is precisely the
        # column that says whether the composite score rates a decision-maker or an offer.
        "composite_emd_hors_forces",
        "composite_l1",
        "formule",
    ]

    def cell(l, c):
        v = l.get(c)
        if c == "couverture" and isinstance(v, float):
            return f"{100 * v:.1f} %".replace(".", ",")
        if c == "part_forces" and isinstance(v, float):
            return f"{100 * v:.1f} %".replace(".", ",")
        # Forced-choice count: "—" if the run has no readable summary. A 0
        # shown in its place would suggest that no decision was constrained.
        if c in ("choix_forces", "choix_forces_score"):
            return "—" if v is None else str(v)
        # Composite score: "—" when the run is not scored (never 0, R9/vacuity).
        if c in (
            "composite_emd",
            "composite_l1",
            "composite_emd_hors_forces",
            "composite_l1_hors_forces",
        ):
            return "—" if v is None else f"{v:.2f}".replace(".", ",")
        if c == "formule":
            if v is None:
                return "—"
            return f"{v} ⚠périmée" if l.get("formule_perimee") else str(v)
        return "" if v is None else str(v)

    largeur = {
        c: max(len(c), *(len(cell(l, c)) for l in lignes)) if lignes else len(c)
        for c in colonnes
    }
    entete = " | ".join(c.ljust(largeur[c]) for c in colonnes)
    sep = "-+-".join("-" * largeur[c] for c in colonnes)
    corps = [" | ".join(cell(l, c).ljust(largeur[c]) for c in colonnes) for l in lignes]
    return "\n".join([entete, sep, *corps])


# ── Summary (E14, E16, E17) ──────────────────────────────────────────────────


def lire_referentiel(chemin: Path | None = None) -> dict:
    """Overall survey modal shares, read from the source of truth — with path + sha256."""
    p = Path(chemin) if chemin else chemin_referentiel()
    if not p.is_file():
        return {"source": str(p), "sha256": None, "valeurs": {}, "disponible": False}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    glob = ((data.get("parts_modales_2023") or {}).get("global")) or {}
    return {
        "source": str(p.relative_to(_racine()))
        if p.is_relative_to(_racine())
        else str(p),
        "sha256": sha256_fichier(p),
        "valeurs": {k: float(v) for k, v in glob.items()},
        "disponible": True,
    }


def synthese(dossier_execution: str | Path, referentiel: Path | None = None) -> dict:
    """All the values of the page, and nothing else: the page computes nothing this JSON lacks (E16)."""
    from mobility_llm.mode_choice import CANONICAL_MODES, canonical_mode

    d = Path(dossier_execution)
    ex = Execution.ouvrir(d)
    traces = ex.decisions
    compteurs = ex.compteurs or {}
    couv_c = compteurs.get("couverture") or {}
    attendus = int(
        couv_c.get("attendus")
        or compteurs.get("attendus_exploitables")
        or compteurs.get("attendus")
        or 0
    )
    methodes = Counter(str(t.get("methode")) for t in traces)
    exclus = methodes.get(METHODE_INEXPLOITABLE, 0)
    comptees = [
        t
        for t in traces
        if t.get("retenue")
        and t.get("methode")
        not in (
            METHODE_NON_COUVERT,
            METHODE_INEXPLOITABLE,
            METHODE_SANS_SOLUTION,
            METHODE_REPLI_UNIFORME,
        )
    ]
    par_mode: Counter = Counter(
        canonical_mode((t.get("retenue") or {}).get("mode")) for t in comptees
    )
    n = sum(par_mode.values())
    # A5 — forced choices enter the shares, and their NUMBER depends on the arm. A trip
    # with a single itinerary was decided by nobody: it follows from an earlier choice of the
    # same arm (leaving by car in the morning means coming back by car at night). Measured on
    # the 23 complete runs of v1: 393 for `duree_minimale`, 286 for `aleatoire`.
    # We do NOT remove them from the main reading — the constraint is the frame, not a
    # defect, and a real day contains some — but we publish both readings, because a
    # single figure mixes what the decision-maker decided and what the situation imposed.
    decidees = [t for t in comptees if t.get("methode") != METHODE_CHOIX_UNIQUE]
    par_mode_decidees: Counter = Counter(
        canonical_mode((t.get("retenue") or {}).get("mode")) for t in decidees
    )
    n_decidees = sum(par_mode_decidees.values())
    n_forces = n - n_decidees
    sources: Counter = Counter()
    fournisseurs: Counter = Counter()
    ecartees_motifs: Counter = Counter()
    for t in traces:
        for s in (t.get("sources") or {}).values():
            sources[str(s).split(":")[0]] += 1
        if t.get("fournisseur"):
            fournisseurs[str(t["fournisseur"])] += 1
        for e in t.get("ecartees") or []:
            ecartees_motifs[str(e.get("motif"))] += 1
    ref = lire_referentiel(referentiel)
    ref_par_mode = {}
    if ref.get("disponible"):
        total_ref = sum(
            v
            for k, v in ref["valeurs"].items()
            if k in set(CORRESPONDANCE_REFERENTIEL.values())
        )
        for mode, cle in CORRESPONDANCE_REFERENTIEL.items():
            if cle in ref["valeurs"] and total_ref:
                ref_par_mode[mode] = ref["valeurs"][cle] / total_ref * 100.0
    decides = len(comptees) + methodes.get(METHODE_REPLI_UNIFORME, 0)
    couverture = {
        "decides": decides,
        "comptes_dans_les_parts": n,
        "attendus": attendus,
        "attendus_bruts": int(
            couv_c.get("attendus_bruts") or compteurs.get("attendus") or attendus
        ),
        "inexploitables_exclus": exclus,
        "taux": decides / attendus if attendus else None,
        "complet": bool(attendus)
        and (decides + methodes.get(METHODE_SANS_SOLUTION, 0)) >= attendus,
    }
    return {
        "execution": ex.nom,
        "experience": (ex.config.get("experience") or {}).get("nom"),
        "etat": ex.etat(),
        "empreintes": ex.config.get("empreintes"),
        "regime_applique": ex.config.get("regime_applique"),
        "sources_alea": ex.config.get("sources_alea"),
        "interruptions": ex.config.get("interruptions"),
        "couverture": couverture,
        "methodes": dict(methodes),
        "parts_modales": {
            "n": n,
            "effectifs": {m: par_mode.get(m, 0) for m in CANONICAL_MODES}
            | ({"other": par_mode["other"]} if par_mode.get("other") else {}),
            "pourcent": {
                m: (100.0 * par_mode.get(m, 0) / n if n else None)
                for m in CANONICAL_MODES
            },
            "couverture": couverture,  # E14: coverage ALWAYS goes with the figure
            "lecture": "toutes décisions, choix forcés compris — porte la conclusion",
        },
        # Second reading (R10): what the decision-maker ACTUALLY decided. It does not replace
        # the first, it complements it; publishing both is the author's decision.
        "parts_modales_hors_choix_unique": {
            "n": n_decidees,
            "effectifs": {m: par_mode_decidees.get(m, 0) for m in CANONICAL_MODES}
            | (
                {"other": par_mode_decidees["other"]}
                if par_mode_decidees.get("other")
                else {}
            ),
            "pourcent": {
                m: (
                    100.0 * par_mode_decidees.get(m, 0) / n_decidees
                    if n_decidees
                    else None
                )
                for m in CANONICAL_MODES
            },
            "couverture": couverture,
            "lecture": "hors décisions à itinéraire unique — ce que le décideur a décidé",
        },
        # R11: the forced-choice count, next to the shares and not lost in the counters.
        "choix_forces": {
            "n": n_forces,
            "part": (n_forces / n) if n else None,
            "lecture": (
                "décisions à itinéraire unique : une seule option existait, personne n'a "
                "choisi. Leur nombre dépend du bras, puisqu'il découle de ses propres choix "
                "antérieurs — les bras se ressemblent donc plus qu'ils ne le sont."
            ),
        },
        "referentiel": {
            "source": ref["source"],
            "sha256": ref["sha256"],
            "disponible": ref["disponible"],
            "pourcent_normalise_4_modes": ref_par_mode,
            "note": "valeurs lues dans la source citée, renormalisées sur les modes qu'elle partage avec les canoniques",
        },
        "sources_propositions": dict(sources),
        "fournisseurs": dict(fournisseurs),
        "ecartees_par_motif": dict(ecartees_motifs),
        "compteurs": compteurs,
    }


def synthese_html(s: dict) -> str:
    """Rendering of a `synthese.json`: each figure is read from the JSON, coverage beside it (E14)."""
    e = html.escape
    couv = s.get("couverture") or {}
    taux = couv.get("taux")
    badge = "COMPLET" if couv.get("complet") else "PARTIEL"
    taux_txt = f"{100 * taux:.1f} %" if taux is not None else "n/a"
    lignes = []
    for mode, pct in (s.get("parts_modales") or {}).get("pourcent", {}).items():
        ref = (
            (s.get("referentiel") or {}).get("pourcent_normalise_4_modes", {}).get(mode)
        )
        # BOTH readings side by side (R10): all decisions, and excluding single itineraries.
        # A single figure would mix what the decision-maker decided and what the situation
        # imposed on it, and the second quantity varies from one arm to another.
        hors = s.get("parts_modales_hors_choix_unique") or {}
        pct_hors = (hors.get("pourcent") or {}).get(mode)
        eff_hors = (hors.get("effectifs") or {}).get(mode, 0)
        lignes.append(
            f"<tr><td>{e(mode)}</td><td>{'' if pct is None else f'{pct:.1f} %'}</td>"
            f"<td>{e(str((s['parts_modales']['effectifs'] or {}).get(mode, 0)))}</td>"
            f"<td>{'' if pct_hors is None else f'{pct_hors:.1f} %'}</td>"
            f"<td>{e(str(eff_hors))}</td>"
            f"<td>{'' if ref is None else f'{ref:.1f} %'}</td><td>{e(str(couv.get('decides')))} / {e(str(couv.get('attendus')))} ({e(taux_txt)})</td></tr>"
        )
    ref = s.get("referentiel") or {}
    return f"""<!doctype html><meta charset="utf-8"><title>Synthèse {e(str(s.get("execution")))}</title>
<style>body{{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.3rem .6rem}}.partiel{{color:#b45309;font-weight:bold}}</style>
<h1>Exécution {e(str(s.get("execution")))} — expérience {e(str(s.get("experience")))}</h1>
<p>État : <b>{e(str((s.get("etat") or {}).get("etat")))}</b> · résultat <span class="{"partiel" if badge == "PARTIEL" else ""}">{badge}</span>
· couverture <b>{e(str(couv.get("decides")))} / {e(str(couv.get("attendus")))} déplacements exploitables ({e(taux_txt)})</b>
{f" · <b>{e(str(couv.get('inexploitables_exclus')))} inexploitables exclus</b> (aucune proposition des moteurs, sur {e(str(couv.get('attendus_bruts')))} attendus)" if couv.get("inexploitables_exclus") else ""}</p>
<h2>Parts modales (n = {e(str((s.get("parts_modales") or {}).get("n")))} décisions comptées)</h2>
<p>Deux lectures. <b>Toutes décisions</b> porte la conclusion : un trajet contraint par un choix
antérieur reste un fait de la journée simulée. <b>Hors itinéraire unique</b> dit ce que le
décideur a réellement décidé — {e(str((s.get("choix_forces") or {}).get("n")))} décisions
({"—" if (s.get("choix_forces") or {}).get("part") is None else f"{100 * s['choix_forces']['part']:.1f} %"})
n'offraient qu'une seule option, personne ne les a choisies. <b>Leur nombre dépend du bras</b>,
puisqu'il découle de ses propres choix antérieurs : deux bras se ressemblent donc plus qu'ils ne
le sont. Pour comparer deux bras terme à terme, restreindre au périmètre qu'ils ont tous deux
réellement décidé.</p>
<table><tr><th>Mode</th><th>Part (toutes)</th><th>Effectif</th><th>Part (hors it. unique)</th><th>Effectif</th><th>Enquête</th><th>Couverture</th></tr>{"".join(lignes)}</table>
<p>Référentiel : <code>{e(str(ref.get("source")))}</code> (sha256 {e(str(ref.get("sha256"))[:16])}…) — {e(str(ref.get("note")))}</p>
<h2>Méthodes</h2><pre>{e(json.dumps(s.get("methodes"), ensure_ascii=False, indent=1))}</pre>
<h2>Sources des propositions</h2><pre>{e(json.dumps(s.get("sources_propositions"), ensure_ascii=False, indent=1))}</pre>
<h2>Écartées par motif</h2><pre>{e(json.dumps(s.get("ecartees_par_motif"), ensure_ascii=False, indent=1))}</pre>
<h2>Sources d'aléa</h2><pre>{e(json.dumps(s.get("sources_alea"), ensure_ascii=False, indent=1))}</pre>
<h2>Régime appliqué · interruptions</h2><pre>{e(json.dumps({"regime_applique": s.get("regime_applique"), "interruptions": s.get("interruptions")}, ensure_ascii=False, indent=1))}</pre>
<h2>Empreintes</h2><pre>{e(json.dumps(s.get("empreintes"), ensure_ascii=False, indent=1))}</pre>
"""


def ecrire_synthese(
    dossier_execution: str | Path, referentiel: Path | None = None
) -> Path:
    d = Path(dossier_execution)
    s = synthese(d, referentiel)
    (d / F_SYNTHESE).write_text(
        json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    (d / F_SYNTHESE_HTML).write_text(synthese_html(s), encoding="utf-8")
    return d / F_SYNTHESE


# ── Comparison (E13) ─────────────────────────────────────────────────────────

CHAMPS_PARTAGES = (
    (
        "population",
        lambda c: (c.get("empreintes") or {}).get("population", {}).get("sha256"),
    ),
    ("jeu", lambda c: (c.get("empreintes") or {}).get("jeu", {}).get("sha256")),
    (
        "calendrier",
        lambda c: json.dumps(
            (c.get("experience") or {}).get("calendrier"), sort_keys=True
        ),
    ),
    ("graine_ordre", lambda c: (c.get("experience") or {}).get("graine_ordre")),
    ("graine_tirage", lambda c: (c.get("experience") or {}).get("graine_tirage")),
    (
        "tolerances_horaires",
        lambda c: json.dumps(
            (c.get("experience") or {}).get("tolerances_horaires"), sort_keys=True
        ),
    ),
    ("max_candidats", lambda c: (c.get("experience") or {}).get("max_candidats")),
    ("horizon_jours", lambda c: (c.get("experience") or {}).get("horizon_jours")),
    ("memoire", lambda c: (c.get("experience") or {}).get("memoire")),
)


class ComparaisonRefusee(ValueError):
    """One of the two runs belongs to an invalidated or archived experiment.

    The refusal is a usage safeguard, not a computational constraint (hygiene spec, P6): the
    figures stay readable in the archive and on the summary page. What is held back
    is the automatic matching — the kind whose table ends up cited six months
    later with nobody remembering that one of the two sides was faulty.
    """


def _statut_de_lexecution(dossier: str | Path) -> dict:
    """Status of the experiment that a run folder belongs to.

    The tree is `<exp>/executions/<horodatage>`: the experiment is two levels up.
    """
    from experiences import statut as S

    return S.lire(Path(dossier).resolve().parent.parent)


def perimetre_commun(traces_par_bras: dict[str, list[dict]]) -> dict:
    """The intersection of the trips that ALL arms actually decided (R12).

    Neither "all decisions" nor "excluding single itineraries" is enough to compare two arms
    term by term. The first mixes what was decided and what was imposed. The second
    removes **different** trips depending on the arm — 393 for `duree_minimale`, 286 for
    `aleatoire` on the same substrate — so it does not make the comparison fairer, it
    moves it: the two columns stop covering the same set.

    So we take the intersection. A trip enters it if, in **each** arm, it is
    present AND is not a single-itinerary choice. The result states its count and what
    each arm loses there: a restricted scope without its reason reads as missing
    data.
    """
    decidees_par_bras: dict[str, set[tuple[str, str]]] = {}
    for bras, traces in traces_par_bras.items():
        decidees_par_bras[bras] = {
            (str(t.get("person_id")), str(t.get("activity_id")))
            for t in traces
            if t.get("retenue") and t.get("methode") != METHODE_CHOIX_UNIQUE
        }
    if not decidees_par_bras:
        return {"cles": set(), "n": 0, "retires": {}, "motif": "aucun bras fourni"}
    commun: set[tuple[str, str]] = set.intersection(*decidees_par_bras.values())
    retires = {
        bras: len(
            {(str(t.get("person_id")), str(t.get("activity_id"))) for t in traces}
        )
        - len(commun)
        for bras, traces in traces_par_bras.items()
    }
    return {
        "cles": commun,
        "n": len(commun),
        "retires": retires,
        "bras": sorted(decidees_par_bras),
        "motif": (
            "déplacements décidés par TOUS les bras : sont exclus les choix à itinéraire "
            "unique, qui varient d'un bras à l'autre, et les déplacements qu'un bras au moins "
            "n'a pas traités"
        ),
    }


def comparer(a: str | Path, b: str | Path, *, inclure_invalides: bool = False) -> dict:
    """Comparable ⇔ fingerprints of everything shared are identical; never on the strength of names (RG-4).

    P6 — refuses to pair a run of an `invalide` or `archivee` experiment unless
    `inclure_invalides=True`. The message names the status, the date and the reason.
    """
    from experiences import statut as S

    statuts = {"a": _statut_de_lexecution(a), "b": _statut_de_lexecution(b)}
    fautifs = [(cote, st) for cote, st in statuts.items() if st["statut"] != S.ACTIF]
    if fautifs and not inclure_invalides:
        details = " · ".join(
            f"[{cote}] {st['statut']} le {st.get('le') or '?'} — "
            f"{st.get('motif') or 'sans motif consigné'}"
            for cote, st in fautifs
        )
        raise ComparaisonRefusee(
            f"comparison refused: {details}. "
            "The figures stay readable in the archive; to compare them anyway, "
            "rerun with --inclure-invalides (the status will be repeated in the output)."
        )
    ca = Execution.ouvrir(a).config
    cb = Execution.ouvrir(b).config
    differences = []
    for nom, lecteur in CHAMPS_PARTAGES:
        va, vb = lecteur(ca), lecteur(cb)
        if va != vb:
            differences.append({"champ": nom, "a": va, "b": vb})
    ga, gb = (
        (ca.get("empreintes") or {}).get("gabarit", {}),
        (cb.get("empreintes") or {}).get("gabarit", {}),
    )
    if ga.get("categorie") == gb.get("categorie") and ga.get("sha256") != gb.get(
        "sha256"
    ):
        differences.append(
            {
                "champ": "gabarit (même catégorie, texte différent)",
                "a": ga.get("sha256"),
                "b": gb.get("sha256"),
            }
        )
    sa, sb = synthese(a), synthese(b)
    return {
        "comparable": not differences,
        "differences": differences,
        # Repeated in the output even when the comparison is forced: a table of gaps
        # gets copied, the reason must travel with it.
        "statuts": {cote: st["statut"] for cote, st in statuts.items()},
        "statuts_motifs": {
            cote: st.get("motif") for cote, st in statuts.items() if st.get("motif")
        },
        "force": bool(fautifs),
        "a": {
            "execution": sa["execution"],
            "experience": sa["experience"],
            "decideur": (ca.get("empreintes") or {}).get("decideur"),
            "parts_modales": sa["parts_modales"],
            # Both readings travel with the comparison: whoever copies it must see
            # that the shares mix decisions and constraints, and by how much (R10, R11).
            "parts_modales_hors_choix_unique": sa.get(
                "parts_modales_hors_choix_unique"
            ),
            "choix_forces": sa.get("choix_forces"),
            "couverture": sa["couverture"],
            "statut": statuts["a"]["statut"],
        },
        "b": {
            "execution": sb["execution"],
            "experience": sb["experience"],
            "decideur": (cb.get("empreintes") or {}).get("decideur"),
            "parts_modales": sb["parts_modales"],
            # Both readings travel with the comparison: whoever copies it must see
            # that the shares mix decisions and constraints, and by how much (R10, R11).
            "parts_modales_hors_choix_unique": sb.get(
                "parts_modales_hors_choix_unique"
            ),
            "choix_forces": sb.get("choix_forces"),
            "couverture": sb["couverture"],
            "statut": statuts["b"]["statut"],
        },
    }


def formater_comparaison(c: dict) -> str:
    out = []
    if c.get("force"):
        out.append("⚠ COMPARAISON FORCÉE — un côté au moins n'est pas actif :")
        for cote, statut in (c.get("statuts") or {}).items():
            if statut != "actif":
                motif = (c.get("statuts_motifs") or {}).get(
                    cote
                ) or "sans motif consigné"
                out.append(f"  [{cote}] {statut} — {motif}")
        out.append("  Les écarts ci-dessous ne doivent pas être cités sans ce motif.")
    if not c["comparable"]:
        out.append("NON COMPARABLE — éléments partagés qui diffèrent :")
        out += [
            f"  - {d['champ']} : {str(d['a'])[:24]} ≠ {str(d['b'])[:24]}"
            for d in c["differences"]
        ]
    else:
        out.append("Comparables (toutes les empreintes partagées sont identiques).")
    for cle in ("a", "b"):
        x = c[cle]
        couv = x["couverture"]
        taux = f"{100 * couv['taux']:.1f} %" if couv.get("taux") is not None else "n/a"
        marque = "" if x.get("statut", "actif") == "actif" else f" [{x['statut']}]"
        out.append(
            f"[{cle}]{marque} {x['execution']} ({x['experience']}, décideur {(x['decideur'] or {}).get('type')}:{(x['decideur'] or {}).get('modele')}) "
            f"— couverture {couv.get('decides')}/{couv.get('attendus')} ({taux})"
        )
        out.append(
            "     toutes décisions : "
            + " · ".join(
                f"{m} {v:.1f} %"
                for m, v in (x["parts_modales"]["pourcent"] or {}).items()
                if v is not None
            )
        )
        hors = x.get("parts_modales_hors_choix_unique") or {}
        forces = x.get("choix_forces") or {}
        if hors.get("pourcent"):
            out.append(
                "     hors it. unique  : "
                + " · ".join(
                    f"{m} {v:.1f} %"
                    for m, v in hors["pourcent"].items()
                    if v is not None
                )
            )
        if forces.get("n") is not None:
            part = forces.get("part")
            out.append(
                f"     choix forcés     : {forces['n']}"
                + (f" ({100 * part:.1f} %)" if part is not None else "")
                + " — leur nombre dépend du bras"
            )
    # The reminder most often missing when two columns are copied side by side (R12).
    out.append(
        "Pour comparer terme à terme : restreindre au périmètre que les DEUX bras ont "
        "réellement décidé (`perimetre_commun`), les choix forcés n'étant pas les mêmes de "
        "part et d'autre."
    )
    return "\n".join(out)


__all__ = [
    "CORRESPONDANCE_REFERENTIEL",
    "ETAT_ARCHIVE_MANQUANTE",
    "chemin_referentiel",
    "comparer",
    "ecrire_synthese",
    "formater_comparaison",
    "formater_table",
    "lire_referentiel",
    "lister",
    "synthese",
    "synthese_html",
    "trier_filtrer",
]
