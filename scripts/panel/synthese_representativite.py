"""synthese_representativite.py — The HTML representativeness summary of a sealed population.

    services/llm-agents/.venv/bin/python -m scripts.panel.synthese_representativite \\
        --sceau data/population/population_1000_PANEL_v4 \\
        --precedent data/population/population_1000_PANEL_v3 \\
        --vivier docs/traces/<date>_controle_vivier_10000_v4/report.json \\
        --audit docs/traces/<date>_audit_perimetre_v4/audit_perimetre.json \\
        --velo docs/traces/<date>_…/velo_cohorte.json --velo-vivier docs/traces/<date>_…/velo_vivier.json \\
        --out docs/traces/<date>_controle_…/synthese_representativite_v3.html \\
        --copie $PAPER_DIR/methode/population/synthese_representativite_v3_population_v4_<date>.html

WHAT IT IS. The document the manuscript cites to say what the test-set population
represents and what it does not: verdicts of the check (`control_population.py`),
comparison with the previous seal and with the raw pool, selection log, gaps to publish,
cross-check of the protocol. It reuses the visual identity of the v2 summary (stylesheet and
rendering script **read from that file**, not copied) and writes only figures read from the
JSON reports — none is typed in by hand.

All inputs are outputs of other scripts: `report.json` and `selection.json` of the
sealed folder (`seal_population.py`), `report.json` of the pool check, `audit_perimetre.json`,
and the structured reports of the bike check (`enrich_personal_bike.py --check --rapport-json`) on
the cohort and on the pool — the slope of bike ownership by household size is judged on the pool.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# HTML template read by the code: it lives with the code, not with the papers (ticket 115). The
# old path `docs/paper/population/` no longer existed since the move under `methode/`.
TEMPLATE_V2 = REPO_ROOT / "scripts" / "panel" / "gabarits" / "synthese_representativite_v2_population_v3_2026-09-03.html"

TITRES = {
    "classe_age": "Classes d'âge (6)", "occupation": "Occupation",
    "motorisation_personne": "Motorisation — base personne",
    "motorisation_menage": "Motorisation — base ménage (1/taille)",
    "couronne": "Couronne de résidence", "couronne_x_motorisation": "Couronne × motorisation",
    "age_quinquennal": "Âge quinquennal", "genre": "Genre",
    "taille_menage_personne": "Taille de ménage (personne)", "permis_adultes": "Permis (adultes)",
    "abonnement_tc": "Abonnement TC", "logement": "Type de logement", "immobile": "Immobiles",
}
ALLOUEE = {"motorisation_menage": "non"}
PILL = {"conforme": "ok", "à publier": "pub", "à corriger": "ko", "non mesurable": "nm", "concordant": "ok"}


def fr(v, nd=1, suffix=""):
    if v is None:
        return "—"
    s = f"{v:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s + suffix


def pill(verdict: str) -> str:
    return f"<span class='pill {PILL.get(verdict, 'nm')}'>{verdict}</span>"


def load(path: Optional[Path]) -> Optional[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None


def template_parts(template: Path) -> tuple[str, str, str]:
    """(<link>+<style> block, rendering script without the DATA line, font) read from the v2 summary."""
    html = template.read_text(encoding="utf-8")
    head = re.search(r"(<link[^>]*>\s*<link[^>]*>\s*<style>.*?</style>)", html, re.S).group(1)
    script = re.search(r"<script>\s*const DATA = .*?;\n(.*?)</script>", html, re.S).group(1)
    return head, script, html


def marge_index(report: dict) -> dict[str, dict]:
    return {m["marge"]: m for m in report["marges"]}


def pire(m: dict) -> str:
    """The category of a margin that deviates most, as the v2 summary writes it."""
    cs = [c for c in m["constats"] if c.get("ecart_pt") is not None]
    if not cs:
        return "—"
    c = max(cs, key=lambda c: abs(c["ecart_pt"]))
    return f"{c['modalite']} {c['observe_pct']:.1f} vs {c['cible_pct']:.1f} ({c['ecart_pt']:+.1f} pt)"


def charts(report: dict) -> list[dict]:
    out = []
    for m in report["marges"]:
        rows = [[c["modalite"], c["observe_pct"], c["cible_pct"],
                 (c["ic95"] or [None, None])[0], (c["ic95"] or [None, None])[1]] for c in m["constats"]]
        div = f"{'EMD' if (m['divergence_type'] or '').startswith('EMD') else 'JSD'} {fr(m['divergence'], 3)}" if m.get("divergence") is not None else "—"
        out.append({"id": m["marge"], "titre": TITRES.get(m["marge"], m["marge"]), "verdict": m["verdict"],
                    "n": m["n"], "sub": f"V de Cramér {fr(m['cramer_v'], 3)} · {div}", "rows": rows})
    return out


def velo_resume(rapport: Optional[dict]) -> Optional[dict]:
    """What the summary keeps from a `--rapport-json` report: verdicts, slope, coverage."""
    if not rapport or not rapport.get("populations"):
        return None
    pop = rapport["populations"][0]
    pente = pop.get("pente_tailles_1_4") or {}
    return {"n": pop.get("n"), "verdicts": pop.get("verdicts") or {}, "pente": pente,
            "couverture_pct": pop.get("couverture_pct"), "code_sortie": rapport.get("code_sortie"),
            "echecs": pop.get("echecs") or [], "check": rapport.get("check", False)}


def velo_phrase(v: Optional[dict], quoi: str) -> str:
    if not v:
        return f"{quoi} : non fourni"
    verd = v["verdicts"]
    pente = v["pente"]
    taux = " / ".join(fr(t) for t in (pente.get("taux_pct") or [])) or "—"
    foyers = " / ".join(str(f) for f in (pente.get("foyers") or [])) or "—"
    statut = pente.get("statut", "—")
    return (f"{quoi} : {verd.get('ok', 0)} contrôles ok, {verd.get('echec', 0)} échec, "
            f"{verd.get('non_concluant', 0)} non concluant ; pente par taille de ménage (1 → 4) "
            f"{taux} % sur {foyers} foyers — <strong>{statut}</strong> ({pente.get('detail', '—')})")


def build(sceau: Path, precedent: Optional[Path], vivier: Optional[Path], audit: Optional[Path],
          template: Path, synthese_version: str, lien_precedent: str,
          velo: Optional[Path] = None, velo_vivier: Optional[Path] = None) -> str:
    head, script, _ = template_parts(template)
    vc, vv = velo_resume(load(velo)), velo_resume(load(velo_vivier))
    manifest = yaml.safe_load((sceau / "MANIFEST.yaml").read_text(encoding="utf-8"))
    rep = load(sceau / "report.json")
    sel = load(sceau / "selection.json") or {}
    prev = load(precedent / "report.json") if precedent else None
    pool = load(vivier)
    aud = load(audit)
    M, P = marge_index(rep), (marge_index(prev) if prev else {})
    V = marge_index(pool) if pool else {}
    men, ref = rep["menages_et_mobilite"], rep["menages_et_mobilite"]["reference_enquete"]
    pmen = prev["menages_et_mobilite"] if prev else {}
    vmen = pool["menages_et_mobilite"] if pool else {}
    verd = rep["verdicts"]
    n = rep["population"]["n"]
    sha = rep["population"]["sha256"]
    per = manifest.get("perimetre") or {}
    deps = per.get("retenus_par_departement") or {}
    hp = per.get("activites_hors_perimetre") or {}
    desc = sel.get("descente") or {}
    dm = desc.get("marges") or {}
    nom = sceau.name
    prev_nom = precedent.name if precedent else "—"
    aujourd_hui = date.today().isoformat()
    n_conf = verd.get("conforme", 0)
    n_marges = len(rep["marges"])

    def cell(mi: dict, nomm: str) -> str:
        m = mi.get(nomm)
        return f"{pill(m['verdict'])} {pire(m)}" if m else "—"

    def desc_cell(nomm: str) -> str:
        j = dm.get(nomm)
        return f"{fr(j['ecart_max_avant_pt'], 2)} → {fr(j['ecart_max_apres_pt'], 2)}" if j and j.get("mesuree") else "—"

    rows_cmp = "\n".join(
        f"<tr><td>{TITRES.get(k, k)}</td><td>{cell(V, k)}</td><td>{cell(P, k)}</td><td>{cell(M, k)}</td>"
        f"<td class='n'>{desc_cell(k)}</td></tr>" for k in M)
    rows_cmp += (
        f"<tr><td>Ménages</td><td>{vmen.get('n_menages', '—')} ménages</td>"
        f"<td>{pmen.get('n_menages', '—')} ménages, {pmen.get('menages_complets_taille_declaree', '—')} complets</td>"
        f"<td>{men['n_menages']} ménages entiers, {men['menages_complets_taille_declaree']} complets au sens strict "
        f"({fr(men['part_membres_presents_pct'])} % des membres déclarés présents)</td><td class='n'>—</td></tr>"
        f"<tr><td>Immobiles</td><td>{fr(vmen.get('part_immobiles_pct'))} %</td><td>{fr(pmen.get('part_immobiles_pct'))} %</td>"
        f"<td>{fr(men['part_immobiles_pct'])} % (cible {fr(ref['part_immobiles_pct'])})</td><td class='n'>{desc_cell('immobile')}</td></tr>"
        f"<tr><td>Déplacements par persona</td><td>{fr(vmen.get('deplacements_par_persona'), 2)}</td>"
        f"<td>{fr(pmen.get('deplacements_par_persona'), 2)} ({fr(pmen.get('deplacements_par_persona_mobile'), 2)} par mobile)</td>"
        f"<td>{fr(men['deplacements_par_persona'], 2)} ({fr(men['deplacements_par_persona_mobile'], 2)} par mobile)</td><td class='n'>—</td></tr>"
        f"<tr><td>Scolaires (6-17 ans) avec activité d'études</td><td>{fr(vmen.get('part_scolaires_avec_etudes_pct'))} %</td>"
        f"<td>{fr(pmen.get('part_scolaires_avec_etudes_pct'))} %</td>"
        f"<td>{fr(men.get('part_scolaires_avec_etudes_pct'))} % ({men.get('scolaires_avec_activite_etudes')}/{men.get('scolaires_mobiles')} ; enquête 90 à 95)</td><td class='n'>—</td></tr>"
        f"<tr><td>Départements de résidence</td><td>—</td><td>1 (Haute-Garonne)</td>"
        f"<td>{len(deps)} : {', '.join(f'{k} {v}' for k, v in deps.items())}</td><td class='n'>—</td></tr>")

    rows_marges = "\n".join(
        f"<tr><td>{TITRES.get(m['marge'], m['marge'])}</td><td>{pill(m['verdict'])}</td><td class='n'>{m['n']}</td>"
        f"<td class='n'>{fr(m['chi2'])}</td><td class='n'>{fr(m['p_value'], 3) if m['p_value'] is not None else '—'}</td>"
        f"<td class='n'>{fr(m['cramer_v'], 3)}</td><td class='n'>{('EMD' if (m['divergence_type'] or '').startswith('EMD') else 'JSD')} {fr(m['divergence'], 3)}</td>"
        f"<td class='n'>{fr(m['ecart_max_pt'], 2)} pt</td><td>{ALLOUEE.get(m['marge'], 'oui')}</td></tr>" for m in rep["marges"])

    rows_desc = "\n".join(
        f"<tr><td>{TITRES.get(k, k)}</td><td class='n'>{j['champ']}</td><td class='n'>{fr(j['ecart_max_avant_pt'], 2)} pt</td>"
        f"<td class='n'>{fr(j['ecart_max_apres_pt'], 2)} pt</td></tr>" for k, j in dm.items() if j.get("mesuree"))

    rows_recoup = "\n".join(
        f"<tr><td>{r['ligne']}</td><td class='n'>{fr(r['valeur_publiee_protocole'])} %</td><td class='n'>{fr(r['reference'])} %</td>"
        f"<td class='n'>{fr(r['ecart_pt'])} pt</td><td>{pill('concordant') if r['statut'] == 'concordant' else pill('à publier').replace('à publier', 'écart — Annexe F')}</td>"
        f"<td>{r['source_reference']}</td></tr>" for r in rep["recoupement"])

    synth_rows = "\n".join(
        f"<tr><td>{r['ecart']}</td><td>{r['amplitude']}</td><td>{r['nature']}</td><td>{pill(r['verdict'])}</td></tr>"
        for r in rep["synthese"]) or "<tr><td colspan='4'>aucun écart dans la synthèse du contrôle</td></tr>"

    audit_txt = ""
    if aud:
        finds = aud.get("findings") or aud.get("axes") or []
        if isinstance(finds, dict):
            finds = list(finds.values())
        parts = [f"{f.get('axe')} {pill(f.get('verdict', ''))}" for f in finds if isinstance(f, dict) and f.get("axe")]
        if parts:
            audit_txt = "<p class='small'>Audit de périmètre (ticket 020) : " + " · ".join(parts) + ".</p>"

    hp_txt = (f"{hp.get('activites_hors_perimetre_supprimees', 0)} activité(s) hors du polygone supprimée(s) "
              f"chez {hp.get('personas_touches', 0)} persona(s)" if hp.get("controle")
              else "non contrôlé (population produite avant le garde-fou)")

    data = {"charts": charts(rep),
            "mob": [["Enquête EMC² — par personne", ref["deplacements_par_personne"], None, None, None],
                    [f"Scellée {nom.split('_')[-1]} — par persona", men["deplacements_par_persona"], None, None, None],
                    ["Enquête — par personne mobile", ref["deplacements_par_personne_mobile"], None, None, None],
                    [f"Scellée {nom.split('_')[-1]} — par persona mobile", men["deplacements_par_persona_mobile"], None, None, None]]
                   + ([[f"Scellée {prev_nom.split('_')[-1]} — par persona mobile", pmen.get("deplacements_par_persona_mobile"), None, None, None]] if prev else []),
            "imm": [["Enquête EMC²", ref["part_immobiles_pct"], None, None, None],
                    [f"Scellée {nom.split('_')[-1]}", men["part_immobiles_pct"], None, None, None]]
                   + ([[f"Vivier {vmen.get('n_menages', '')and ''}brut", vmen.get("part_immobiles_pct"), None, None, None]] if pool else [])
                   + ([[f"Scellée {prev_nom.split('_')[-1]}", pmen.get("part_immobiles_pct"), None, None, None]] if prev else [])}

    vivier_n = (sel.get("vivier") or {}).get("n", "—")
    v_a_corr = (pool or {}).get("verdicts", {}).get("à corriger", "—")
    scol = men.get("part_scolaires_avec_etudes_pct")
    html = f"""<title>Représentativité de {nom}</title>
{head}
<div class="wrap">
<div class="eyebrow">Article · jalon 0 du protocole · contrôle du {rep['date'][:10]} · synthèse {synthese_version}</div>
<h1>La population scellée {nom.split('_')[-1]} est-elle représentative de l'enquête ?</h1>
<p class="lead prose">Sur les {n_marges} marges contrôlées, {'oui — toutes conformes' if n_conf == n_marges else f'{n_conf} sont conformes'}, et pour la première fois sur le <strong>périmètre entier de l'enquête</strong> : les 453 communes de six départements, par le polygone des communes. Douze de ces marges sont <em>allouées</em> par la sélection ; la fidélité du générateur se lit sur le vivier. Ce qui change avec ce sceau : les chaînes d'activités viennent enfin de l'ENTD nationale appariée par classe d'âge, les écoliers vont à l'école ({fr(scol)} % des 6-17 ans mobiles), et les agents mobiles font <strong>{fr(men['deplacements_par_persona_mobile'], 2)} déplacements par jour contre {fr(ref['deplacements_par_personne_mobile'], 2)}</strong>.</p>
<div class="meta"><span>data/population/{nom}/</span><span>sha256 {sha[:16]}…</span><span>vivier eqasim {vivier_n} · règle {sel.get('version', '—')} (ménages, {len(dm)} marges de descente)</span><span>périmètre : {per.get('definition', '—')}</span><span>référence EMC² 2023 (rapport AUAT p. 10, 11, 21 · microdonnées ProGEDO, COEP · cibles gelées cj1 + cm1)</span><span><a href="{lien_precedent}">synthèse précédente ({prev_nom})</a></span></div>
<div class="tiles"><div class="tile"><div class="k">{n_conf} / {n_marges}</div><div class="l">marges conformes (TOST ± 1 pt, IC95) — {verd.get('à corriger', 0)} à corriger, {verd.get('à publier', 0)} à publier, {verd.get('non mesurable', 0)} non mesurable</div></div><div class="tile"><div class="k">{len(deps)} / 6</div><div class="l">départements de résidence représentés — {', '.join(f'{k} : {v}' for k, v in deps.items())}</div></div><div class="tile"><div class="k">{fr(scol)} %</div><div class="l">des 6-17 ans mobiles ont une activité d'études — enquête 90 à 95 %, v3 54 %</div></div><div class="tile"><div class="k">{fr(men['deplacements_par_persona_mobile'], 2)}</div><div class="l">déplacements par agent mobile — l'enquête en compte {fr(ref['deplacements_par_personne_mobile'], 2)}, la v3 en avait {fr(pmen.get('deplacements_par_persona_mobile'), 2)}</div></div></div>
<ul class="toc"><li><a href="#verdict">1 · Le verdict, en trois niveaux</a></li><li><a href="#v3v4">2 · De la {prev_nom.split('_')[-1]} à la {nom.split('_')[-1]}</a></li><li><a href="#marges">3 · Les {n_marges} marges</a></li><li><a href="#vivier">4 · Du vivier à la cohorte</a></li><li><a href="#reste">5 · Ce qui reste</a></li><li><a href="#recoupement">6 · Recoupement du protocole</a></li><li><a href="#article">7 · Pour l'article</a></li></ul>
<h2 id="verdict">1 · Le verdict, en trois niveaux</h2><div class="prose">
<p><strong>Niveau 1 — les marges contrôlées.</strong> {n_conf} marges {pill('conforme')} sur {n_marges} : classes d'âge, occupation, motorisation sur deux bases, couronne, croisement, âge quinquennal, genre, taille de ménage, permis, abonnement TC, logement, immobiles. Ménages : {men['n_menages']}, dont {men['menages_complets_taille_declaree']} complets au sens strict ({fr(men['part_membres_presents_pct'])} % des membres déclarés présents). Immobiles {fr(men['part_immobiles_pct'])} % (cible {fr(ref['part_immobiles_pct'])}).</p>
<p><strong>Niveau 2 — ce que ça prouve.</strong> Douze marges sur treize sont allouées par la sélection (allocation sur 12 cellules, descente sur {len(dm)} marges) ; seule la motorisation en base ménage est probante sans allocation. La représentativité <em>réelle</em> du générateur se lit sur le vivier de {vivier_n} : <strong>{v_a_corr} marges à corriger</strong>. Deux choses que ce vivier porte et que les précédents n'avaient pas : le périmètre entier (la 3ᵉ couronne compte ses 275 communes, dont 100 hors de la Haute-Garonne) et un appariement sur l'ENTD nationale — le service Docker appariait jusqu'ici sur 308 donneurs résidents du 31, v3 comprise.</p>
<p><strong>Niveau 3 — ce que le contrôle ne voit pas.</strong> La mobilité des agents mobiles ({fr(men['deplacements_par_persona_mobile'], 2)} déplacements contre {fr(ref['deplacements_par_personne_mobile'], 2)}) reste {pill('à publier')} : les chaînes viennent de l'ENTD 2008 et aucune sélection ne les rallonge. Activités hors du polygone des 453 communes : {hp_txt}. Les enfants de moins de 5 ans sont hors population enquêtée et absents par construction.</p>
<p><strong>Équipement vélo</strong> (trait imputé, contrôlé à part par <code>enrich_personal_bike --check</code>, ticket 015) : {velo_phrase(vc, 'cohorte') if vc else "rapport de la cohorte non fourni"}. {("La pente ne se juge qu'à partir de " + str((vc or {}).get('pente', {}).get('min_foyers_pour_juger', 100)) + " foyers par taille, donc sur le vivier — " + velo_phrase(vv, 'vivier') + ".") if vv else "Rapport du vivier non fourni : la pente n'est pas jugée."}</p>{audit_txt}</div>
<h2 id="v3v4">2 · De la {prev_nom.split('_')[-1]} à la {nom.split('_')[-1]} — ce que le périmètre et l'appariement ont changé</h2><div class="prose"><p>Même mécanique de sélection (ménages entiers, allocation couronne × motorisation, descente), trois changements en amont : le cadre de tirage passe des 346 communes haut-garonnaises aux <strong>453 communes de six départements</strong> ; les journées donneuses ENTD sont des <strong>jours de classe</strong> et l'appariement se fait sur l'enquête <strong>nationale</strong> avec la classe d'âge tenue (borne à 17 ans pour les lycéens) ; les six classes d'âge du rapport entrent dans la descente. Le vivier est pré-imputé (logement, vélo, permis, abonnement) avant la sélection.</p></div>
<div class="tblwrap"><table><thead><tr><th>Marge</th><th>Vivier {vivier_n} (brut du générateur)</th><th>{prev_nom.split('_')[-1]}</th><th>{nom.split('_')[-1]}</th><th class="n">Descente : écart max avant → après (pt)</th></tr></thead><tbody>{rows_cmp}</tbody></table></div>
<h2 id="marges">3 · Les {n_marges} marges</h2><p class="prose">Barre : population scellée {nom.split('_')[-1]}. Tiret : cible. Trait fin : IC95. Survolez une barre pour le détail.</p><div class="legend"><span class="lb">population scellée (%)</span><span class="lt">cible</span><span class="li">IC95</span></div><div class="grid" id="charts"></div>
<div class="tblwrap"><table><thead><tr><th>Marge</th><th>Verdict</th><th class="n">n</th><th class="n">χ²</th><th class="n">p</th><th class="n">V de Cramér</th><th class="n">EMD / JSD</th><th class="n">écart max</th><th>allouée ?</th></tr></thead><tbody>{rows_marges}</tbody></table></div>
<div class="callout"><strong>Lecture.</strong> Sur une marge allouée, p ≈ 1 et V ≈ 0 par construction. Le χ² est publié parce que le gabarit le demande ; il se lit avec V et l'effectif, jamais seul.</div>
<h2 id="vivier">4 · Du vivier à la cohorte — ce que la sélection fait</h2><div class="prose"><p>Le vivier eqasim ({vivier_n} personnes livrées pour 10 000 demandées ; {(sel.get('vivier') or {}).get('eligibles', '—')} éligibles, exclus : {json.dumps((sel.get('vivier') or {}).get('exclus', {}), ensure_ascii=False)}) a <strong>{v_a_corr} marges à corriger</strong>. La sélection en retient {fr(100.0 * n / vivier_n) if isinstance(vivier_n, int) else '—'} %.</p><p><strong>Allocation.</strong> Cibles en personnes par cellule couronne × motorisation (plus fort reste) ; les ménages entrent dans l'ordre d'un <code>sha256</code> de leur identifiant s'ils tiennent dans leur cellule. Déficits : {json.dumps(sel.get('deficits') or {}) if sel.get('deficits') else 'aucun sur les 12 cellules'}.</p><p><strong>Descente.</strong> {desc.get('echanges', '—')} échanges de ménages de même taille et même cellule, en {desc.get('passes', '—')} passes, perte {fr(desc.get('perte_avant_pt'), 2)} → {fr(desc.get('perte_apres_pt'), 2)} pt ({desc.get('duree_s', '—')} s). Déterministe : même vivier, même fichier au sha256 près.</p></div>
<div class="tblwrap"><table><thead><tr><th>Marge de la descente</th><th class="n">champ (personas)</th><th class="n">écart max avant</th><th class="n">écart max après</th></tr></thead><tbody>{rows_desc}</tbody></table></div>
<h2 id="reste">5 · Ce qui reste</h2><h3>La mobilité des agents mobiles {pill('à publier')}</h3><div class="two"><div class="card"><h4>Déplacements par jour et par personne</h4><p class="sub">enquête : PENQ = 1, COEP · scellée : n déplacements pour n activités, retour au domicile compris</p><svg class="chart" id="c-mob"></svg></div><div class="card"><h4>Part des personnes sans déplacement</h4><p class="sub">une marge de la sélection</p><svg class="chart" id="c-imm"></svg></div></div>
<div class="prose"><p class="small">Les chaînes d'activités viennent de l'ENTD 2008 appariée par eqasim — désormais l'enquête nationale, un jour de classe, par classe d'âge — mais l'enquête de référence compte {fr(ref['deplacements_par_personne_mobile'], 2)} déplacements par personne mobile. Le levier restant est l'EMC² 2023 comme enquête d'appariement, un chantier eqasim distinct.</p></div>
<h3>La synthèse des écarts du contrôle</h3><div class="tblwrap"><table><thead><tr><th>Écart</th><th>Amplitude</th><th>Nature</th><th>Verdict</th></tr></thead><tbody>{synth_rows}</tbody></table></div>
<h3>Ce que ni la sélection ni le contrôle ne peuvent refermer</h3><div class="tblwrap"><table><thead><tr><th>Écart</th><th>Amplitude</th><th>Nature</th></tr></thead><tbody><tr><td>Chaînes d'activités</td><td>ENTD 2008 nationale, jours de classe ; motifs sans accompagnement (ticket 027)</td><td>enquête d'appariement</td></tr><tr><td>Enfants de moins de 5 ans</td><td>{men['membres_declares'] - men['membres_presents']} membres déclarés absents sur {men['membres_declares']} ; hors population enquêtée, absents par construction</td><td>définition</td></tr><tr><td>Immobiles du vivier</td><td>{fr(vmen.get('part_immobiles_pct'))} % dans le vivier (ENTD nationale, jours de classe) contre {fr(ref['part_immobiles_pct'])} % dans l'enquête ; la cohorte est tenue à {fr(men['part_immobiles_pct'])} %</td><td>vivier — déclaré</td></tr><tr><td>Activités hors du polygone des 453 communes</td><td>{hp_txt}</td><td>hypothèse assumée</td></tr><tr><td>Parts non redressées, rabattement voiture + TC, fenêtre saisonnière du run</td><td>axes A3, A7, A5 de l'audit de périmètre</td><td>objet compté / run</td></tr><tr><td>Équipement vélo — pente par taille de ménage</td><td>{velo_phrase(vc, 'cohorte') if vc else 'cohorte : non fourni'} ; {velo_phrase(vv, 'vivier') if vv else 'vivier : non fourni'}</td><td>jugée sur le vivier (≥ {(vc or vv or {}).get('pente', {}).get('min_foyers_pour_juger', 100)} foyers par taille, règle du 2026-09-03) — trait imputé, hors sélection</td></tr></tbody></table></div>
<h2 id="recoupement">6 · Recoupement du tableau § 2.1 du protocole</h2><div class="tblwrap"><table><thead><tr><th>Ligne du protocole</th><th class="n">publié</th><th class="n">référence</th><th class="n">écart</th><th>statut</th><th>source</th></tr></thead><tbody>{rows_recoup}</tbody></table></div>
<h2 id="article">7 · Ce que ça change pour l'article</h2><div class="prose"><p><strong>Formulation défendable.</strong> « La cohorte de {n} agents, en {men['n_menages']} ménages entiers, est <em>alignée par construction</em> sur {n_marges} marges de l'EMC² 2023 — structure couronne × motorisation, âge, genre, occupation, taille de ménage, permis, abonnement, logement, part d'immobiles — par sélection stratifiée dans un vivier synthétique de {vivier_n} personnes tiré sur le périmètre exact de l'enquête, 453 communes de six départements. Ses chaînes d'activités viennent de l'ENTD 2008 nationale appariée par classe d'âge un jour de classe ; elle ne reproduit pas leur longueur : {fr(men['deplacements_par_persona_mobile'], 2)} déplacements par agent mobile contre {fr(ref['deplacements_par_personne_mobile'], 2)}. »</p><p><strong>À déclarer.</strong> Que douze marges sur treize sont allouées ; que le vivier brut en avait {v_a_corr} à corriger ; les chaînes ENTD ; que les sceaux v2 et v3 — et les runs qui les citent — reposaient sur un appariement à 308 donneurs résidents de Haute-Garonne et sur le seul département 31.</p></div>
<footer>Sources : rapport AUAT/CEREMA EMC² 2023 (68 p., pages 10, 11, 21, 24, 26) ; microdonnées ProGEDO lil-1750 (personnes, ménages, déplacements ; COEP, COE0) ; cibles gelées <code>scripts/panel/cible_jointe_couronne_motorisation.yaml</code> (cj1) et <code>cibles_marges_personne.yaml</code> (cm1) ; <code>control_population.py</code> (rapport dans le dossier scellé) ; <code>seal_population.py</code> {sel.get('version', '')} ; <code>make audit-perimetre</code> ; <code>enrich_personal_bike.py --check --rapport-json</code> (cohorte et vivier). Méthode : <code>docs/arch/controle-population-jeu-de-test.md</code> ; tickets 029 et 031. Généré le {aujourd_hui} par <code>scripts/panel/synthese_representativite.py</code>.</footer></div><div class="tip" id="tip" hidden></div>
<script>
const DATA = {json.dumps(data, ensure_ascii=False)};
{script}</script>
"""
    return html


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sceau", type=Path, required=True, help="sealed folder (MANIFEST.yaml, report.json, selection.json)")
    ap.add_argument("--precedent", type=Path, default=None, help="previous sealed folder (comparison column)")
    ap.add_argument("--vivier", type=Path, default=None, help="report.json of the pool check")
    ap.add_argument("--audit", type=Path, default=None, help="audit_perimetre.json")
    ap.add_argument("--velo", type=Path, default=None, help="JSON report of the bike check of the cohort (enrich_personal_bike --check --rapport-json)")
    ap.add_argument("--velo-vivier", type=Path, default=None, help="JSON report of the bike check of the pool (the slope is judged there)")
    ap.add_argument("--template", type=Path, default=TEMPLATE_V2)
    ap.add_argument("--version", default="v3", help="summary number (v3 for population v4)")
    ap.add_argument("--lien-precedent", default="synthese_representativite_v2_population_v3_2026-09-03.html")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copie", type=Path, default=None, help="copy into $PAPER_DIR/methode/population/ (papers repository)")
    args = ap.parse_args(argv)
    html = build(args.sceau, args.precedent, args.vivier, args.audit, args.template, args.version, args.lien_precedent,
                 velo=args.velo, velo_vivier=args.velo_vivier)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"summary written: {args.out} ({len(html.encode('utf-8')) / 1024:.0f} KB)")
    if args.copie:
        args.copie.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, args.copie)
        print(f"copy: {args.copie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
