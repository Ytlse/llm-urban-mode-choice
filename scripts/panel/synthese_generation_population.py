"""synthese_generation_population.py — How the test-set population is built.

    services/llm-agents/.venv/bin/python -m scripts.panel.synthese_generation_population \\
        --sceau data/population/population_1000_PANEL_v4 \\
        --vivier docs/traces/<date>_controle_toulouse_population_10000/report.json \\
        --audit docs/traces/<date>_audit_perimetre_v4/audit_perimetre.json \\
        --velo docs/traces/<date>_…/velo_cohorte.json --velo-vivier docs/traces/<date>_…/velo_vivier.json \\
        --mesures-graphe docs/traces/<date>_mesures_graphe_perimetre_v4/mesures.json \\
        --out docs/traces/<date>_…/fabrication_population.html \\
        --copie $PAPER_DIR/methode/population/fabrication_population_v4_<date>.html

WHAT IT IS. The page that explains, end to end, where each agent of the test set comes from: the
public data and the matching survey that eqasim consumes, what the fork changes, the steps
of the notebook (day, PT, zone, selection, routing, export, traits, audit), the selection by
whole households, the check, the sealing — and the results measured at each stage. Companion to
the representativeness summary (`synthese_representativite.py`), which judges; this one narrates.

RULE. No figure is typed by hand: everything is read from the outputs of other scripts
(MANIFEST.yaml, report.json, selection.json of the seal; report.json of the pool check;
audit_perimetre.json; `enrich_personal_bike --rapport-json` reports; OSMnx graph metadata;
mesures.json of the graph; config_toulouse.yml and commune_couronne.json). The only figures that
have no structured file (eqasim generation log) are grouped in `JOURNAL`, with
their source, and marked with a dagger (†) in the page.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from scripts.panel.synthese_representativite import (TEMPLATE_V2, TITRES, fr, load, pill,
                                                     template_parts, velo_resume)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_EQASIM = REPO_ROOT / "services" / "eqasim-toulouse" / "config_toulouse.yml"
COMMUNES = REPO_ROOT / "packages" / "mobility_core" / "src" / "mobility_core" / "data" / "commune_couronne.json"
GRAPHE_META = REPO_ROOT / "data" / "cache" / "osmnx" / "graphs_444ca7e6a515.meta.json"

# Figures of the eqasim generation log and of the notebook, with no structured file. Each carries
# its source; the page marks them with a dagger (†). Replace them with a file read on the day
# the service writes its log as JSON.
JOURNAL = {
    "source": "ticket 031 (docs/tickets/ticket_031_perimetre_453_communes.md, § 1.2, § 1.2 bis, "
              "décisions et « Sceau v4 ») et CHANGELOG.md du fork eqasim-toulouse, journal de "
              "génération du 2026-09-03",
    "donneurs_avant": 15687, "donneurs_apres": 12392,
    "scolaires_trajet_ecole_avant_pct": 72.0, "scolaires_trajet_ecole_apres_pct": 90.8,
    "personnes_avant_ponderation": 17986, "part_3e_couronne_avant_ponderation_pct": 42.5,
    "part_sans_iris_dans_cadre_pct": {"31": 86.7, "32": 9.4, "81": 9.0, "82": 20.1, "09": 4.0, "11": 1.0},
    "donneurs_service_avant_correction": 308,
    "duree_eqasim_min_caches": 3, "duree_eqasim_min_froid": 8,
    "routage_paires": 3291, "routage_none": 17, "routage_s": 582, "routage_workers": 3,
    "activites_planifiees": 3335, "activites_desservies_tc": 2660,
}

EXTRA_CSS = """<style>
.flow{width:100%;height:auto;display:block;margin:14px 0 6px}
.flow text{font-family:var(--sans);font-size:12px;fill:var(--ink)}
.flow .t{font-weight:600;font-size:13px}
.flow .s{fill:var(--ink-3);font-size:11px}
.flow rect{fill:var(--card);stroke:var(--line)}
.flow .hl rect{stroke:var(--accent);stroke-width:1.5}
.flow path,.flow line{stroke:var(--ink-3);fill:none}
.steps{counter-reset:s;margin:0;padding:0;list-style:none}
.steps li{position:relative;padding:8px 0 8px 40px;border-top:1px solid var(--line-soft)}
.steps li::before{counter-increment:s;content:counter(s);position:absolute;left:0;top:9px;width:26px;height:26px;border-radius:13px;background:var(--accent-soft);color:var(--accent-ink);font:600 12px/26px var(--sans);text-align:center}
.steps li strong{color:var(--ink)}
.dag{color:var(--ink-3);font-size:12px}
sup.j{color:var(--pub);font-weight:600}
</style>"""


def dag_svg() -> str:
    """The four phases of the build, in a diagram that follows the theme (template colours)."""
    boxes = [
        (10, "1 · eqasim", "recensement + ENTD + BD TOPO/BAN", "vivier ≈ 11 000 personnes, JSON"),
        (215, "2 · notebook 1 → 3bis", "journée, TC, zone fine", "vivier pré-imputé"),
        (420, "3 · sélection v4", "ménages entiers, 12 cellules", "1 000 personas"),
        (625, "4 · notebook 4 → 9", "routage polygone, export, traits", "population complète"),
        (830, "5 · contrôle + sceau", "13 marges, audit, MANIFEST", "dossier immuable"),
    ]
    parts = ['<svg class="flow" viewBox="0 0 1035 120" role="img" aria-label="Les cinq temps de la fabrication">']
    for i, (x, t, s1, s2) in enumerate(boxes):
        cls = ' class="hl"' if i in (2, 4) else ""
        parts.append(f'<g{cls}><rect x="{x}" y="18" width="190" height="84" rx="8"/>'
                     f'<text class="t" x="{x + 12}" y="42">{t}</text>'
                     f'<text class="s" x="{x + 12}" y="62">{s1}</text>'
                     f'<text class="s" x="{x + 12}" y="80">{s2}</text></g>')
        if i < len(boxes) - 1:
            parts.append(f'<path d="M{x + 190} 60 L{x + 205} 60" stroke-width="1.5"/>'
                         f'<path d="M{x + 200} 55 L{x + 205} 60 L{x + 200} 65" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def j(v, nd=0) -> str:
    """A figure from the (unstructured) log, marked with a dagger."""
    return f"{fr(v, nd)}<sup class='j' title='lu dans le journal de génération, pas dans un fichier structuré'>†</sup>"


def build(sceau: Path, vivier: Optional[Path], audit: Optional[Path], velo: Optional[Path],
          velo_vivier: Optional[Path], config: Path, communes: Path, graphe_meta: Optional[Path],
          mesures_graphe: Optional[Path], template: Path, lien_synthese: str) -> str:
    head, _, _ = template_parts(template)
    manifest = yaml.safe_load((sceau / "MANIFEST.yaml").read_text(encoding="utf-8"))
    rep = load(sceau / "report.json") or {}
    sel = load(sceau / "selection.json") or {}
    pool = load(vivier)
    aud = load(audit)
    vc, vv = velo_resume(load(velo)), velo_resume(load(velo_vivier))
    cfg = (yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}) or {}
    conf = cfg.get("config") or {}
    com = load(communes) or {}
    gm = load(graphe_meta) or {}
    mg = load(mesures_graphe) or {}

    nom = sceau.name
    men = rep.get("menages_et_mobilite") or manifest.get("controle", {}).get("menages_et_mobilite") or {}
    ref = men.get("reference_enquete") or {}
    vmen = (pool or {}).get("menages_et_mobilite") or {}
    per = manifest.get("perimetre") or {}
    msel = manifest.get("selection") or {}
    viv = msel.get("vivier") or sel.get("vivier") or {}
    exclus = viv.get("exclus") or {}
    mr = msel.get("menages_retenus") or {}
    desc = msel.get("descente") or sel.get("descente") or {}
    dm = (sel.get("descente") or {}).get("marges") or {}
    verd = (manifest.get("controle") or {}).get("verdicts") or rep.get("verdicts") or {}
    n = (manifest.get("population") or {}).get("n", "—")
    sha = (manifest.get("population") or {}).get("sha256", "")
    deps = per.get("retenus_par_departement") or {}
    aujourd_hui = date.today().isoformat()
    M = {m["marge"]: m for m in rep.get("marges", [])}
    V = {m["marge"]: m for m in (pool or {}).get("marges", [])}

    # ── Phase 1: eqasim ──────────────────────────────────────────────────────────────────
    dep_list = conf.get("departments") or []
    par_dep = Counter(c["insee"][:2] for c in com.get("communes", []))
    par_cour = Counter(c["couronne"] for c in com.get("communes", []))
    attrs = conf.get("matching_attributes") or []
    bornes = conf.get("matching_age_boundaries") or []
    reglages = [
        ("Départements", ", ".join(dep_list) or "—", "les six départements du périmètre EMC² 2023 (ticket 031)"),
        ("Cadre de tirage", f"{com.get('n_communes', '—')} communes ({', '.join(f'{d} : {k}' for d, k in sorted(par_dep.items()))})",
         "liste des communes de l'enquête, croisée avec les départements ; une commune inconnue du référentiel IRIS fait échouer le stage"),
        ("Couronnes", ", ".join(f"{k} : {v}" for k, v in sorted(par_cour.items())) or "—", "table `commune_couronne.json` version " + str(com.get("version", "—"))),
        ("Taux de sondage", f"{conf.get('sampling_rate', '—')} (recalculé par le service sur la population RP 2022 du cadre)", "10 000 demandés → vivier de " + fr(viv.get("n"), 0) + " personnes"),
        ("Graine", str(conf.get("random_seed", "—")), "reproductibilité du tirage"),
        ("Enquête d'appariement", f"ENTD 2008 nationale (`filter_hts: {conf.get('filter_hts', '—')}`)", "sans ce réglage, le vivier de donneurs tombait à " + j(JOURNAL["donneurs_service_avant_correction"]) + " résidents de Haute-Garonne — c'était le cas de toutes les populations jusqu'à la v3"),
        ("Attributs d'appariement", " → ".join(attrs) or "—", "la dégradation retire les colonnes par la fin : l'âge est le dernier abandonné"),
        ("Bornes d'âge", ", ".join(str(b) for b in bornes) or "—", "la borne 17 sépare les lycéens des jeunes actifs (décision du 2026-09-03)"),
        ("Observations minimales", str(conf.get("matching_minimum_observations", "—")), "seuil sous lequel une strate se dégrade"),
        ("Journées donneuses", f"jours de classe (`hts_school_days_only: {conf.get('hts_school_days_only', '—')}`), mercredi exclu sous {conf.get('hts_exclude_wednesday_under_age', '—')} ans",
         "hors vacances scolaires (V2_VAC_SCOL) ; un donneur dont la journée est écartée sort du vivier"),
        ("Personnes à commune inconnue", f"pondérées (`census_undefined_reweighting: {conf.get('census_undefined_reweighting', '—')}`)",
         "poids RP × part de la population sans IRIS du département qui vit dans le cadre"),
    ]
    rows_reglages = "\n".join(f"<tr><td>{a}</td><td><code>{b}</code></td><td>{c}</td></tr>" for a, b, c in reglages)

    stages = [
        ("data.census.filtered", "Recensement INSEE (RP)", "individus et ménages des communes du cadre : âge, sexe, PCS, activité, voitures, taille du ménage",
         "filtre par liste de communes ; les personnes à commune « undefined » (communes sans IRIS) sont gardées et pondérées"),
        ("data.hts.entd.cleaned → filtered → selected", "ENTD 2008", "journées de référence, trajets, permis, abonnement, vélos des donneurs",
         "jours de classe seulement ; témoin « scolaires mobiles avec trajet vers l'école » (alarme sous 85 %)"),
        ("synthesis.population.sampled", "Tirage", "tirage des ménages au taux de sondage, poids RP",
         "—"),
        ("synthesis.population.income.selected", "Revenu (FILOSOFI)", "revenu du ménage par unité de consommation, par commune",
         "—"),
        ("synthesis.population.matched", "Appariement HTS", "à chaque personne un donneur ENTD de même strate (âge, sexe, motorisation, PCS, département)",
         "national, bornes d'âge configurables, seuil 5 observations"),
        ("synthesis.population.enriched", "Enrichissement", "permis, abonnement, vélos du donneur ; `car_availability`, `age_range`, PCS détaillée, secteur",
         "permis des mineurs neutralisés avant l'agrégation ; colonnes source du sceau (RP `TRANS`, identifiants des donneurs)"),
        ("synthesis.population.trips → activities", "Chaîne d'activités", "la journée du donneur ENTD recopiée : motifs, heures, durées",
         "—"),
        ("data.bdtopo · data.ban · spatial.home.zones → locations", "Domicile", "zone (IRIS ou commune) puis adresse : bâtiments BD TOPO pondérés par leurs logements, adresses BAN",
         "BD TOPO 2025-03-15 des six départements, lue dans les archives .7z ; BAN du 2026-09-03"),
        ("spatial.primary · spatial.secondary → spatial.locations", "Lieux d'activité", "travail et études par distance de navette et capacité (BPE, SIRENE) ; lieux secondaires (achats, loisirs) par distances ENTD",
         "—"),
        ("synthesis.population.llm_agents", "Export JSON pour GAMA", "personas (traits, occupation, revenu textualisé, nom Faker), activités avec coordonnées, `household`, `provenance`, `validation.commute_mode` à la racine, immobiles gardés",
         "stage du fork ; `household.commune_id` renseigné pour tous depuis le tirage de zone ; couronne de résidence posée ici"),
    ]
    rows_stages = "\n".join(f"<tr><td><code class='dag'>{s}</code></td><td><strong>{t}</strong></td><td>{q}</td><td>{f}</td></tr>" for s, t, q, f in stages)
    ponder = ", ".join(f"{d} : {j(v, 1)} %" for d, v in JOURNAL["part_sans_iris_dans_cadre_pct"].items())

    # ── Phase 2: notebook ────────────────────────────────────────────────────────────────
    etapes = [
        ("Génération eqasim", "POST /generate au service (port 8003) ; le vivier brut va dans <code>Temp/1_raw/</code>",
         f"le service refuse de générer si <code>config_toulouse.yml</code> manque ou si un département n'a pas ses données ; {j(JOURNAL['duree_eqasim_min_caches'])} min avec les caches BD TOPO/BAN, {j(JOURNAL['duree_eqasim_min_froid'])} min à froid"),
        ("Journée valide", "activités hors du polygone des 453 communes supprimées (jamais le domicile), comptées et alarmées ; chevauchements corrigés ; activités consécutives ou circulaires de même motif et même lieu fusionnées — le domicile du soir et celui du matin sont une seule activité",
         f"{(per.get('activites_hors_perimetre') or {}).get('activites_hors_perimetre_supprimees', '—')} activité hors polygone sur {(per.get('activites_hors_perimetre') or {}).get('personas_controles', '—')} personas contrôlés"),
        ("Desserte TC", "drapeau <code>public_transport</code> vrai si un arrêt Tisséo (GTFS) est à moins de 1 500 m du lieu", "c'est ce drapeau que l'agent lit pour envisager le TC"),
        ("Zone fine", "libellé de zone de chaque lieu (grille de densité INSEE, aire d'attraction), géocodage inverse BAN mis en cache", "—"),
        ("Pré-imputation puis sélection (3ter)", "les traits d'équipement sont posés sur le <em>vivier</em> (permis, abonnement, logement, vélo, deux passes de <code>align_minor_traits</code>) pour être des marges de sélection ; puis <code>seal_population select</code>, règle " + str(msel.get("version", "—")),
         f"{fr(viv.get('n'), 0)} personnes → {n} retenus en {mr.get('n', '—')} ménages"),
    ]
    rows_etapes = "\n".join(f"<li><strong>{t}.</strong> {d} <span class='small'>— {m}</span></li>" for t, d, m in etapes)

    # ── Phase 3: selection ───────────────────────────────────────────────────────────────
    cibles = sel.get("cibles") or {}
    retenus = sel.get("retenus_par_cellule") or {}
    vcell = viv.get("par_cellule") or (sel.get("vivier") or {}).get("par_cellule") or {}
    ordre = ["Toulouse", "1ere couronne", "2eme couronne", "3eme couronne"]
    motos = ["sans voiture", "une voiture", "deux voitures et +"]
    rows_cell = []
    for c in ordre:
        for m in motos:
            k = f"{c} × {m}"
            if k in cibles:
                rows_cell.append(f"<tr><td>{k}</td><td class='n'>{fr(vcell.get(k), 0)}</td><td class='n'>{cibles[k]}</td><td class='n'>{retenus.get(k, '—')}</td></tr>")
    rows_cell = "\n".join(rows_cell)
    rows_desc = "\n".join(
        f"<tr><td>{TITRES.get(k, k)}</td><td class='n'>{fr(v['ecart_max_avant_pt'], 2)} pt</td><td class='n'>{fr(v['ecart_max_apres_pt'], 2)} pt</td></tr>"
        for k, v in dm.items() if v.get("mesuree"))
    tailles = mr.get("par_taille") or {}
    tailles_txt = ", ".join(f"{k} : {v}" for k, v in tailles.items())

    # ── Phase 4: routing ─────────────────────────────────────────────────────────────────
    modes = (gm.get("graphes") or {}).get("modes") or {}
    rows_modes = "\n".join(
        f"<tr><td>{mode}</td><td class='n'>{fr(d.get('noeuds'), 0)}</td><td class='n'>{fr(d.get('aretes'), 0)}</td><td class='n'>{fr(d.get('part_aretes_vitesse_repli_pct'))} %</td><td class='n'>{fr(d.get('duree_s'), 0)} s</td></tr>"
        for mode, d in modes.items())
    zc = (gm.get("graphes") or {}).get("zones_congestion") or {}
    zc_txt = " · ".join(f"{mode} : {fr(z.get('city'), 0)} ville / {fr(z.get('agglo'), 0)} agglomération / {fr(z.get('outside'), 0)} dehors" for mode, z in zc.items())
    poly = (mg.get("paires") or {}).get("graphes", {}).get("polygone_453") or {}
    disque = (mg.get("paires") or {}).get("graphes", {}).get("disque_30km") or {}
    pc = poly.get("meme_noeud_par_couronne") or {}
    pd_ = disque.get("meme_noeud_par_couronne") or {}
    rows_couronne = "\n".join(
        f"<tr><td>{c}</td><td class='n'>{fr((pc.get(c) or {}).get('n'), 0)}</td><td class='n'>{fr((pd_.get(c) or {}).get('part_lointain_pct'), 2)} %</td><td class='n'>{fr((pc.get(c) or {}).get('part_lointain_pct'), 2)} %</td><td class='n'>{fr((pc.get(c) or {}).get('part_pct'))} %</td></tr>"
        for c in ordre if c in pc)
    rout = poly.get("routage") or {}
    rab = poly.get("distance_rabattement_m") or {}
    workers = mg.get("workers") or {}
    wpoly, wdisq = workers.get("polygone_453") or {}, workers.get("disque_30km") or {}

    # ── Phase 5: traits ──────────────────────────────────────────────────────────────────
    traits = [
        ("align_minor_traits", "permis → non sous 18 ans ; `work` → `education` pour scolaires et étudiants ; motifs recalculés ; `car_availability` par ménage sans les permis de mineurs", "règles idempotentes ; rejoué une seconde fois après `enrich_equipment`"),
        ("enrich_residence_zone", "couronne, commune et code INSEE du domicile, lus sur le découpage de l'enquête (trait observé, pas tiré)", "`hors périmètre` pour un domicile connu et dehors ; rien sans coordonnées"),
        ("enrich_housing_type", "type de logement tiré dans la loi EMC² des ménages de la zone fine, levier de taille de ménage, hachage de l'adresse", "ticket 019 ; refuse une ressource v1"),
        ("enrich_personal_bike", "vélo personnel par les trois étages appris sur EMC² : stock du ménage, attribution, VAE — tirage par foyer reconstitué à l'adresse", "ticket 015 ; contrôle `--check` à part (ci-dessous)"),
        ("enrich_equipment", "abonnement TC et permis par les lois apprises sur EMC² (tickets 016 et 017), hachage sur la personne", "recette en lecture seule après la seconde passe de `align_minor_traits`"),
    ]
    rows_traits = "\n".join(f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in traits)

    def velo_row(v: Optional[dict], quoi: str) -> str:
        if not v:
            return f"<tr><td>{quoi}</td><td colspan='4'>rapport non fourni</td></tr>"
        p = v["pente"]
        return (f"<tr><td>{quoi}</td><td class='n'>{fr(v.get('couverture_pct'))} %</td><td class='n'>{v['verdicts'].get('ok', 0)} ok · {v['verdicts'].get('echec', 0)} échec · {v['verdicts'].get('non_concluant', 0)} non concluant</td>"
                f"<td class='n'>{' / '.join(fr(t) for t in p.get('taux_pct', []))} % sur {' / '.join(str(f) for f in p.get('foyers', []))} foyers</td><td>{pill('conforme') if p.get('statut') == 'ok' else pill('non mesurable') if p.get('statut') == 'non concluant' else pill('à corriger')} {p.get('statut', '—')} — code de sortie {v.get('code_sortie', '—')}</td></tr>")
    rows_velo = velo_row(vc, f"cohorte {nom.split('_')[-1]}") + velo_row(vv, "vivier")

    # ── Phase 6: check, audit, seal ──────────────────────────────────────────────────────
    rows_marges = "\n".join(
        f"<tr><td>{TITRES.get(k, k)}</td><td>{pill(V[k]['verdict']) if k in V else '—'}</td><td>{pill(M[k]['verdict'])}</td><td class='n'>{fr(M[k].get('ecart_max_pt'), 2)} pt</td></tr>"
        for k in M)
    mob_rows = [
        ("Personnes", fr(vmen.get("membres_presents"), 0), fr(men.get("membres_presents"), 0), "—"),
        ("Ménages", fr(vmen.get("n_menages"), 0), f"{fr(men.get('n_menages'), 0)} entiers, {fr(men.get('menages_complets_taille_declaree'), 0)} complets au sens strict", "—"),
        ("Immobiles", f"{fr(vmen.get('part_immobiles_pct'))} %", f"{fr(men.get('part_immobiles_pct'))} %", f"{fr(ref.get('part_immobiles_pct'))} %"),
        ("Déplacements par personne", fr(vmen.get("deplacements_par_persona"), 2), fr(men.get("deplacements_par_persona"), 2), fr(ref.get("deplacements_par_personne"), 2)),
        ("Déplacements par personne mobile", fr(vmen.get("deplacements_par_persona_mobile"), 2), fr(men.get("deplacements_par_persona_mobile"), 2), fr(ref.get("deplacements_par_personne_mobile"), 2)),
        ("Scolaires (6-17 ans) mobiles avec activité d'études", f"{fr(vmen.get('part_scolaires_avec_etudes_pct'))} %", f"{fr(men.get('part_scolaires_avec_etudes_pct'))} % ({men.get('scolaires_avec_activite_etudes', '—')}/{men.get('scolaires_mobiles', '—')})", " à ".join(fr(x, 0) for x in (ref.get("part_scolaires_avec_etudes_pct") or [])) + " %"),
    ]
    rows_mob = "\n".join(f"<tr><td>{a}</td><td class='n'>{b}</td><td class='n'>{c}</td><td class='n'>{d}</td></tr>" for a, b, c, d in mob_rows)
    finds = (aud or {}).get("findings") or []
    rows_audit = "\n".join(f"<tr><td>{f.get('axe')}</td><td>{f.get('titre')}</td><td>{f.get('simule', '')}</td><td>{pill(f.get('verdict', ''))}</td></tr>" for f in finds if isinstance(f, dict))
    depot = manifest.get("depot") or {}
    synth = (manifest.get("controle") or {}).get("synthese_des_ecarts") or rep.get("synthese") or []
    ecarts_txt = " ; ".join(f"{e.get('ecart')} — {e.get('amplitude')} ({e.get('verdict')})" for e in synth) or "aucun"
    v_verd = (pool or {}).get("verdicts") or {}

    html = f"""<title>Fabrication de la population {nom.split('_')[-1]}</title>
{head}
{EXTRA_CSS}
<div class="wrap">
<div class="eyebrow">Article · jalon 0 du protocole · population scellée {nom.split('_')[-1]} · page générée le {aujourd_hui}</div>
<h1>Comment la population du jeu de test est fabriquée</h1>
<p class="lead prose">Mille agents en {mr.get('n', '—')} ménages entiers, tirés dans un vivier synthétique de {fr(viv.get('n'), 0)} personnes que le générateur eqasim a construit pour les {com.get('n_communes', '—')} communes de l'enquête EMC² Toulouse 2023, puis routés, dotés de leurs traits et scellés. Cette page suit le fichier de bout en bout : ce qu'eqasim consomme et ce que le fork y change, ce que chaque étape du notebook ajoute ou corrige, comment la sélection par ménages entiers choisit, et ce que le contrôle a mesuré à chaque étage. La page compagne <a href="{lien_synthese}">« La population scellée {nom.split('_')[-1]} est-elle représentative de l'enquête ? »</a> juge ; celle-ci raconte.</p>
<div class="meta"><span>data/population/{nom}/</span><span>sha256 {sha[:16]}…</span><span>scellée le {str(manifest.get('scelle_le', ''))[:16].replace('T', ' ')}</span><span>règle {msel.get('version', '—')}</span><span>dépôt {str(depot.get('git_head', ''))[:10]} ({depot.get('branche', '—')})</span></div>
<div class="tiles"><div class="tile"><div class="k">{fr(viv.get('n'), 0)}</div><div class="l">personnes dans le vivier eqasim — {fr(viv.get('menages'), 0)} ménages, {len(deps)} départements, {com.get('n_communes', '—')} communes du cadre</div></div><div class="tile"><div class="k">{n}</div><div class="l">personas retenus en {mr.get('n', '—')} ménages entiers — {desc.get('echanges', '—')} échanges de descente, perte {fr(desc.get('perte_avant_pt'))} → {fr(desc.get('perte_apres_pt'))} pt</div></div><div class="tile"><div class="k">{verd.get('conforme', '—')} / {len(M) or '—'}</div><div class="l">marges conformes au contrôle — {verd.get('à publier', 0)} à publier, {verd.get('à corriger', 0)} à corriger ; vivier brut : {v_verd.get('à corriger', '—')} à corriger</div></div><div class="tile"><div class="k">{fr(poly.get('meme_noeud_lointain_total_pct'), 2)} %</div><div class="l">de paires distantes rabattues sur un même nœud du graphe du polygone (disque de 30 km : {fr(disque.get('meme_noeud_lointain_total_pct'), 2)} %)</div></div></div>
<ul class="toc"><li><a href="#vue">1 · Vue d'ensemble</a></li><li><a href="#eqasim">2 · eqasim</a></li><li><a href="#notebook">3 · Le notebook, étapes 1 à 3ter</a></li><li><a href="#selection">4 · La sélection par ménages</a></li><li><a href="#routage">5 · Le routage sur le polygone</a></li><li><a href="#traits">6 · Les traits</a></li><li><a href="#controle">7 · Contrôle, audit, sceau</a></li><li><a href="#resultats">8 · Les résultats en un tableau</a></li><li><a href="#limites">9 · Ce que la fabrication ne fait pas</a></li><li><a href="#rejouer">10 · Rejouer</a></li></ul>

<h2 id="vue">1 · Vue d'ensemble — cinq temps, un fichier</h2>
{dag_svg()}
<div class="prose"><p>Le générateur <strong>eqasim</strong> (fork <code>eqasim-llm-toulouse</code>) tire des ménages dans le recensement, leur apparie une journée réelle de l'enquête nationale ENTD 2008, les loge à une adresse et place leurs activités ; il livre un vivier bien plus grand que la cible. Le <strong>notebook</strong> <code>generate_population.ipynb</code> rend chaque journée jouable (chevauchements, fusions, activités hors du polygone), pose la desserte TC et la zone fine, puis <strong>pré-impute</strong> le vivier et en <strong>sélectionne</strong> {n} personas par ménages entiers, sur les marges de l'enquête. Les retenus seuls sont <strong>routés</strong> sur le graphe du polygone des {com.get('n_communes', '—')} communes pour recaler leurs horaires, exportés, puis dotés des traits que la simulation consomme. Enfin le <strong>contrôle</strong> juge le fichier sur {len(M) or 13} marges, l'audit de périmètre le passe aux neuf axes du ticket 020, et le <strong>scellement</strong> le fige avec son empreinte, sa règle et son rapport.</p><p class="small">Croix (†) : chiffre lu dans le journal de génération ou le ticket, pas dans un fichier structuré ({JOURNAL['source']}). Tout le reste est lu dans les fichiers du sceau, des contrôles et des traces.</p></div>

<h2 id="eqasim">2 · eqasim — des données publiques à un vivier de {fr(viv.get('n'), 0)} personnes</h2>
<div class="prose"><p><strong>Ce qu'il fait, en une phrase.</strong> Il tire des individus et leurs ménages dans le recensement INSEE des communes du cadre, apparie à chacun une personne réellement enquêtée dans l'ENTD 2008 — dont il recopie la journée (motifs, heures) et l'équipement (permis, abonnement, vélos) —, lui affecte un revenu FILOSOFI, le loge dans un bâtiment BD TOPO à une adresse BAN, place ses lieux de travail, d'études et d'achats, puis exporte le tout en JSON pour GAMA.</p></div>
<h3>Les réglages scientifiques (<code>config_toulouse.yml</code>, source unique)</h3>
<div class="tblwrap"><table><thead><tr><th>Réglage</th><th>Valeur</th><th>Pourquoi</th></tr></thead><tbody>{rows_reglages}</tbody></table></div>
<h3>La chaîne des stages synpp</h3>
<div class="tblwrap"><table><thead><tr><th>Stage</th><th>Quoi</th><th>Ce qu'il produit</th><th>Ce que le fork y change</th></tr></thead><tbody>{rows_stages}</tbody></table></div>
<h3>Ce que le journal de génération a mesuré</h3>
<div class="prose"><ul>
<li><strong>Journées donneuses.</strong> Le filtre des jours de classe garde {j(JOURNAL['donneurs_apres'])} journées de référence sur {j(JOURNAL['donneurs_avant'])} ; la part des scolaires mobiles avec un trajet vers l'école passe de {j(JOURNAL['scolaires_trajet_ecole_avant_pct'], 1)} % à {j(JOURNAL['scolaires_trajet_ecole_apres_pct'], 1)} % chez les donneurs. Un donneur dont la journée est écartée sort du vivier : la première version le gardait comme immobile, et la population générée comptait alors 40 % d'immobiles.</li>
<li><strong>Personnes à commune inconnue.</strong> Le recensement ne nomme pas la commune des habitants des communes sans IRIS. Avec une liste de communes comme cadre, eqasim versait toute la population rurale du département dans les quelques villages du cadre : {j(JOURNAL['personnes_avant_ponderation'])} personnes livrées pour 10 000 demandées, {j(JOURNAL['part_3e_couronne_avant_ponderation_pct'], 1)} % en 3ᵉ couronne. Leur poids est désormais multiplié par la part de la population sans IRIS du département qui vit dans le cadre ({ponder}). Résultat : {fr(viv.get('n'), 0)} personnes, 3ᵉ couronne au niveau de l'enquête.</li>
<li><strong>Le vivier livré</strong> : {fr(viv.get('n'), 0)} personnes en {fr(viv.get('menages'), 0)} ménages ; {fr(viv.get('eligibles'), 0)} éligibles à la sélection après exclusion de {exclus.get('moins_de_5_ans', '—')} enfants de moins de 5 ans (hors population enquêtée), {exclus.get('sans_couronne', '—')} sans domicile localisable et {exclus.get('hors_perimetre', '—')} domicile hors des {com.get('n_communes', '—')} communes (adresse en limite de commune). Immobiles {fr(vmen.get('part_immobiles_pct'))} % (enquête {fr(ref.get('part_immobiles_pct'))} %) ; scolaires mobiles avec activité d'études {fr(vmen.get('part_scolaires_avec_etudes_pct'))} %.</li>
</ul></div>

<h2 id="notebook">3 · Le notebook — rendre la journée jouable, puis choisir</h2>
<div class="prose"><p>Chaque étape écrit un point de reprise dans <code>Temp/</code> et se saute si sa sortie existe ; une sélection n'est réutilisée que si son journal porte l'empreinte du vivier courant. Les cinq premières étapes :</p></div>
<ol class="steps">{rows_etapes}</ol>
<div class="callout"><strong>Pourquoi la sélection vient avant le routage.</strong> Le vivier ne coûte que la synthèse eqasim ; le routage et les traits ne tournent que sur les {n} retenus. La pré-imputation du vivier (étape 3ter-a) est ce qui permet au logement, au vélo, au permis et à l'abonnement d'être des <em>marges</em> de la sélection et non de simples constats.</div>

<h2 id="selection">4 · La sélection par ménages entiers (règle {msel.get('version', '—')})</h2>
<div class="prose"><p><strong>L'unité est le ménage.</strong> Un ménage a une couronne et une motorisation, donc une cellule ; ses membres de 5 ans et plus entrent ou sortent ensemble. <strong>Allocation</strong> : les {len(cibles)} cellules couronne × motorisation reçoivent un effectif cible par plus fort reste sur la cible jointe de l'enquête (base personne, version {(msel.get('cible_jointe') or sel.get('cible_jointe') or {}).get('version', 'cj1')}) ; les ménages entrent dans l'ordre d'un <code>sha256</code> de leur identifiant s'ils tiennent dans leur cellule. <strong>Descente</strong> : tant qu'un échange de deux ménages de même taille et même cellule réduit la somme des écarts absolus aux marges contrôlées, on l'applique — {desc.get('echanges', '—')} échanges en {desc.get('passes', '—')} passes, perte {fr(desc.get('perte_avant_pt'), 2)} → {fr(desc.get('perte_apres_pt'), 2)} pt en {desc.get('duree_s', '—')} s. Déterministe : même vivier, même fichier au sha256 près. Déficits : {json.dumps(msel.get('deficits') or {}, ensure_ascii=False) if msel.get('deficits') else 'aucun'} ; reports : {msel.get('reports', 0)}.</p><p>Ménages retenus par taille : {tailles_txt} — {fr(mr.get('membres_presents'), 0)} membres présents sur {fr(mr.get('membres_declares'), 0)} déclarés (les absents sont les enfants de moins de 5 ans). Départements de résidence : {', '.join(f'{k} : {v}' for k, v in deps.items())} ; {per.get('communes_distinctes', '—')} communes distinctes.</p></div>
<div class="two"><div class="card"><h4>Les 12 cellules</h4><div class="tblwrap"><table><thead><tr><th>Couronne × motorisation</th><th class="n">vivier</th><th class="n">cible</th><th class="n">retenus</th></tr></thead><tbody>{rows_cell}</tbody></table></div></div>
<div class="card"><h4>Ce que la descente referme</h4><div class="tblwrap"><table><thead><tr><th>Marge</th><th class="n">écart max avant</th><th class="n">après</th></tr></thead><tbody>{rows_desc}</tbody></table></div></div></div>

<h2 id="routage">5 · Le routage sur le polygone des {com.get('n_communes', '—')} communes (étapes 4 et 5)</h2>
<div class="prose"><p>Les horaires exportés par eqasim ignorent les temps de trajet. L'étape 4 calcule, pour chaque paire de lieux consécutifs d'un persona, un itinéraire OSMnx dans son <em>mode de planification</em> (voiture s'il en a une, vélo sinon), et l'étape 5 recale les heures de début pour absorber ces durées. Le graphe est celui du <strong>polygone des communes</strong> ({fr((gm.get('polygone') or {}).get('area_km2'), 0)} km²), construit sans téléchargement depuis les extraits OSM régionaux du fork (<code>{gm.get('label', '—')}</code>, clé <code>{gm.get('cache_key', '—')}</code>, OSMnx {gm.get('osmnx_version', '—')}), et non le disque de 30 km de la production : un domicile de 3ᵉ couronne y a ses propres nœuds. Construction {fr(gm.get('duree_totale_s'), 0)} s, {fr(gm.get('ram_pointe_mo'), 0)} Mo de pointe, pickle {fr((gm.get('cache') or {}).get('graphs_pkl_mo'))} Mo.</p></div>
<div class="two"><div class="card"><h4>Les trois graphes</h4><div class="tblwrap"><table><thead><tr><th>Mode</th><th class="n">nœuds</th><th class="n">arêtes</th><th class="n">arêtes en vitesse de repli</th><th class="n">construction</th></tr></thead><tbody>{rows_modes}</tbody></table></div><p class="small">Vitesses de repli (config/osmnx.yaml) : {', '.join(f'{k} {v} km/h' for k, v in ((gm.get('vitesses') or {}).get('fallbacks_kph') or {}).items())}. Zones de congestion posées sur les nœuds — {zc_txt}.</p></div>
<div class="card"><h4>Paires « même nœud » par couronne (cohorte {nom.split('_')[-1]}, {fr((mg.get('paires') or {}).get('n_paires'), 0)} paires)</h4><div class="tblwrap"><table><thead><tr><th>Couronne d'origine</th><th class="n">paires</th><th class="n">distantes > 500 m sur un même nœud — disque 30 km</th><th class="n">— polygone</th><th class="n">même nœud, toutes distances</th></tr></thead><tbody>{rows_couronne}</tbody></table></div><p class="small">Critère 3 du ticket 031 (≤ 0,5 % par couronne) : {'tenu' if poly.get('critere_3_ok') else 'non tenu'} — {fr(poly.get('meme_noeud_lointain_total_pct'), 2)} % au total. Rabattement médian {fr(rab.get('mediane'), 0)} m, p95 {fr(rab.get('p95'), 0)} m. Routage échantillonné : {rout.get('routes', '—')} routes, {fr(rout.get('part_none_pct'), 2)} % sans itinéraire, {fr(rout.get('ms_par_route_mediane'), 0)} ms par route (médiane). Worker : {fr(wpoly.get('ram_pointe_mo'), 0)} Mo, chargement {fr(wpoly.get('chargement_s'))} s (disque : {fr(wdisq.get('ram_pointe_mo'), 0)} Mo, {fr(wdisq.get('chargement_s'))} s).</p></div></div>
<div class="prose"><p><strong>Ce que le routage de la cohorte a donné</strong><sup class='j'>†</sup> : {j(JOURNAL['routage_paires'])} paires de planification, {j(JOURNAL['routage_none'])} sans itinéraire, {j(JOURNAL['routage_s'])} s avec {j(JOURNAL['routage_workers'])} workers ; {j(JOURNAL['activites_planifiees'])} activités planifiées, {j(JOURNAL['activites_desservies_tc'])} desservies en TC. Deux règles du 2026-09-03 : le facteur de congestion TomTom s'applique par <em>zone</em> de l'arête (ville, agglomération, dehors) et non plus à tout trajet ; une paire rabattue sur un même nœud reçoit une durée à la vitesse du mode (vol d'oiseau × 1,3), plus jamais 0 minute. Le réchauffage du cache d'itinéraires du runtime (étape 6) est optionnel et ne touche pas au fichier.</p></div>

<h2 id="traits">6 · Les traits (étape 8) — ce que la simulation consomme</h2>
<div class="prose"><p>Après l'export vers <code>data/population/</code> (étape 7, qui refuse d'écraser une population sans horaires ou amputée de moitié), sept post-traitements posent ou corrigent les traits, dans un ordre imposé : le vélo lit le logement, et <code>car_availability</code> dérive des permis que <code>enrich_equipment</code> réécrit — d'où la seconde passe. Chacun refuse de tourner si sa ressource d'accès restreint manque, plutôt que d'imputer à l'aveugle. L'étape 9 rend le verdict <strong>POPULATION COMPLÈTE</strong> quand les neuf traits consommés et les horaires recalés sont là.</p></div>
<div class="tblwrap"><table><thead><tr><th>Post-traitement</th><th>Ce qu'il pose</th><th>Garde-fou</th></tr></thead><tbody>{rows_traits}</tbody></table></div>
<h3>Le contrôle du vélo, à part</h3>
<div class="prose"><p>Le vélo est le trait dont la part modale est la plus scrutée : son imputation a son propre contrôle (<code>enrich_personal_bike --check</code>), qui compare le taux de porteurs, la part de VAE, le gradient par taille de ménage et par type d'habitat, la part de ménages équipés et les vélos par ménage à des cibles EMC², bruit d'échantillonnage compris. La <strong>pente</strong> du taux de porteurs par taille de ménage ne se juge qu'à partir de {(vc or vv or {}).get('pente', {}).get('min_foyers_pour_juger', 100)} foyers par taille : sur la cohorte elle s'affiche « non concluant » sans peser sur le verdict, et se lit sur le vivier.</p></div>
<div class="tblwrap"><table><thead><tr><th>Population</th><th class="n">couverture</th><th class="n">contrôles</th><th class="n">pente tailles 1 → 4</th><th>verdict de la pente</th></tr></thead><tbody>{rows_velo}</tbody></table></div>

<h2 id="controle">7 · Contrôle, audit, scellement</h2>
<div class="prose"><p><strong>Le contrôle</strong> (<code>control_population.py</code>) compare la population aux marges de l'EMC² 2023 : IC95 de Clopper–Pearson, TOST à ± {(rep.get('parametres') or {}).get('borne_pt', 1.0)} pt, χ² et V de Cramér, EMD ou JSD selon l'échelle ; cibles gelées <code>cj1</code> (jointe couronne × motorisation) et <code>cm1</code> (marges personne recalculées sur microdonnées). Un écart établi sur une marge que la sélection sait refermer est <em>à corriger</em> et fait refuser le scellement ; sur une marge qu'elle ne referme pas, il est <em>à publier</em>. La ligne « scolaires avec activité d'études » est un témoin des chaînes d'activités, pas une marge de sélection.</p></div>
<div class="two"><div class="card"><h4>Les {len(M) or 13} marges — vivier brut puis cohorte</h4><div class="tblwrap"><table><thead><tr><th>Marge</th><th>vivier</th><th>cohorte</th><th class="n">écart max</th></tr></thead><tbody>{rows_marges}</tbody></table></div></div>
<div class="card"><h4>Ménages et mobilité</h4><div class="tblwrap"><table><thead><tr><th></th><th class="n">vivier</th><th class="n">cohorte</th><th class="n">enquête</th></tr></thead><tbody>{rows_mob}</tbody></table></div><p class="small">Déplacements : n déplacements pour n activités, retour au domicile compris ; enquête : PENQ = 1, COEP.</p></div></div>
<h3>L'audit de périmètre (ticket 020)</h3>
<div class="tblwrap"><table><thead><tr><th>Axe</th><th>Question</th><th>Constat sur la cohorte</th><th>Verdict</th></tr></thead><tbody>{rows_audit or "<tr><td colspan='4'>audit non fourni</td></tr>"}</tbody></table></div>
<h3>Le sceau</h3>
<div class="prose"><p><code>seal_population.py seal</code> rejoue le contrôle, refuse s'il reste un « à corriger », sinon écrit un dossier qui ne se modifie plus : <code>population.json</code> (sha256 <code>{sha}</code>), <code>MANIFEST.yaml</code> (empreinte, règle, vivier et exclus, ménages retenus, descente, verdicts, cibles gelées et leurs empreintes, périmètre — définition, départements attendus et retenus, activités hors polygone —, révision git <code>{str(depot.get('git_head', ''))[:10]}</code>), <code>CONTROLE.md</code> et <code>report.json</code>, <code>selection.json</code>. Écarts déclarés au sceau : {ecarts_txt}. La sauvegarde <code>data/population/sauvegardes/</code> archive le dossier <em>et</em> le vivier (brut et pré-imputé) : sans le vivier, la sélection ne se rejoue pas ; avec, elle redonne le même fichier à l'octet.</p></div>

<h2 id="resultats">8 · Les résultats en un tableau</h2>
<div class="tblwrap"><table><thead><tr><th>Étage</th><th>Mesure</th></tr></thead><tbody>
<tr><td>eqasim</td><td>{fr(viv.get('n'), 0)} personnes, {fr(viv.get('menages'), 0)} ménages, {len(deps)} départements ; immobiles {fr(vmen.get('part_immobiles_pct'))} % ; scolaires avec études {fr(vmen.get('part_scolaires_avec_etudes_pct'))} % ; {fr(vmen.get('deplacements_par_persona_mobile'), 2)} déplacements par personne mobile ; {v_verd.get('à corriger', '—')} marges à corriger sur {len(V) or '—'}</td></tr>
<tr><td>Journée (étape 2)</td><td>{(per.get('activites_hors_perimetre') or {}).get('activites_hors_perimetre_supprimees', '—')} activité hors du polygone sur {(per.get('activites_hors_perimetre') or {}).get('personas_controles', '—')} personas</td></tr>
<tr><td>Sélection</td><td>{n} personas, {mr.get('n', '—')} ménages entiers, aucun déficit, {desc.get('echanges', '—')} échanges, perte {fr(desc.get('perte_avant_pt'), 1)} → {fr(desc.get('perte_apres_pt'), 1)} pt ; {len(deps)} départements représentés</td></tr>
<tr><td>Routage</td><td>{fr(poly.get('meme_noeud_lointain_total_pct'), 2)} % de paires distantes sur un même nœud (critère 3 {'tenu' if poly.get('critere_3_ok') else 'non tenu'}), {fr(rout.get('part_none_pct'), 2)} % sans itinéraire, {fr(rout.get('ms_par_route_mediane'), 0)} ms par route</td></tr>
<tr><td>Traits</td><td>vélo : cohorte {(vc or {}).get('verdicts', {}).get('ok', '—')} contrôles ok, pente {(vc or {}).get('pente', {}).get('statut', '—')} ; vivier {(vv or {}).get('verdicts', {}).get('ok', '—')} ok, pente {(vv or {}).get('pente', {}).get('statut', '—')}</td></tr>
<tr><td>Contrôle</td><td>{verd.get('conforme', '—')} conformes, {verd.get('à publier', 0)} à publier, {verd.get('à corriger', 0)} à corriger ; immobiles {fr(men.get('part_immobiles_pct'))} % ; scolaires {fr(men.get('part_scolaires_avec_etudes_pct'))} % ; {fr(men.get('deplacements_par_persona'), 2)} déplacements par persona, {fr(men.get('deplacements_par_persona_mobile'), 2)} par mobile (enquête {fr(ref.get('deplacements_par_personne'), 2)} / {fr(ref.get('deplacements_par_personne_mobile'), 2)})</td></tr>
<tr><td>Audit</td><td>{' · '.join(f"{f.get('axe')} {f.get('verdict')}" for f in finds if isinstance(f, dict)) or '—'}</td></tr>
</tbody></table></div>

<h2 id="limites">9 · Ce que la fabrication ne fait pas</h2>
<div class="prose"><ul>
<li><strong>Les chaînes d'activités restent celles de l'ENTD 2008</strong>, nationale, un jour de classe : {fr(men.get('deplacements_par_persona_mobile'), 2)} déplacements par agent mobile contre {fr(ref.get('deplacements_par_personne_mobile'), 2)} dans l'enquête, et des motifs sans accompagnement. Remplacer l'enquête d'appariement par l'EMC² 2023 est un chantier eqasim distinct.</li>
<li><strong>Le vivier porte plus d'immobiles que l'enquête</strong> ({fr(vmen.get('part_immobiles_pct'))} % contre {fr(ref.get('part_immobiles_pct'))} %) ; la sélection tient la cohorte à {fr(men.get('part_immobiles_pct'))} %, l'écart du vivier est déclaré.</li>
<li><strong>La motorisation en base ménage</strong> est la seule marge que la sélection n'alloue pas : {ecarts_txt}.</li>
<li><strong>Le transport scolaire</strong> n'est dans aucun GTFS ouvert : les écoliers de 3ᵉ couronne n'ont pas de TC dans la simulation tant que le ticket 030 n'est pas livré.</li>
<li><strong>Le runtime</strong> doit encore charger ce fichier par commune du domicile et router sur le même graphe du polygone (partie 2 du ticket 031) ; jusque-là, le sceau ne se charge pas entier.</li>
<li><strong>Les enfants de moins de 5 ans</strong> sont hors population enquêtée et absents par construction ({fr((mr.get('membres_declares') or 0) - (mr.get('membres_presents') or 0), 0)} membres déclarés absents).</li>
</ul></div>

<h2 id="rejouer">10 · Rejouer la fabrication</h2>
<div class="prose"><ol>
<li><code>docker compose build eqasim && docker compose up -d eqasim otp1</code> — le service part de <code>config_toulouse.yml</code> et des six départements.</li>
<li><code>make osmnx-perimeter-graph</code> — le graphe du polygone, s'il n'est pas déjà dans <code>data/cache/osmnx/</code>.</li>
<li>Notebook <code>scripts/data/population/generate_population.ipynb</code>, cellule « Paramètres » : <code>POPULATION_SIZES = [10000]</code>, <code>SELECT_N = 1000</code>, <code>FORCE_REGENERATE = True</code> ; exécuter tout.</li>
<li><code>make control-population POP=… TRACE=docs/traces/&lt;date&gt;_…</code> puis <code>make seal-population POP=… SELECTION=…</code> ; <code>make audit-perimetre</code>.</li>
<li><code>enrich_personal_bike … --dry-run --check --rapport-json</code> sur la cohorte et sur le vivier ; <code>make synthese-representativite … VELO=… VELO_VIVIER=…</code> ; cette page : <code>make synthese-generation-population …</code>.</li>
</ol><p class="small">Méthode : <code>docs/setup/population.md</code> (chaîne), <code>docs/arch/population-post-traitements.md</code> (d'où vient chaque attribut), <code>docs/arch/controle-population-jeu-de-test.md</code> (contrôle et scellement), <code>docs/arch/velo-equipement.md</code>, tickets 026, 029, 031.</p></div>
<footer>Sources lues : <code>{nom}/MANIFEST.yaml</code>, <code>report.json</code>, <code>selection.json</code> ; contrôle du vivier {('(' + str(vivier) + ')') if vivier else 'non fourni'} ; audit de périmètre ; rapports <code>enrich_personal_bike --rapport-json</code> ; <code>config_toulouse.yml</code> du fork ; <code>commune_couronne.json</code> ; méta du graphe <code>{gm.get('cache_key', '—')}</code> ; <code>mesures.json</code> du graphe. Chiffres marqués † : {JOURNAL['source']}. Généré le {aujourd_hui} par <code>scripts/panel/synthese_generation_population.py</code>.</footer></div>
"""
    return html


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sceau", type=Path, required=True)
    ap.add_argument("--vivier", type=Path, default=None, help="report.json of the pool check")
    ap.add_argument("--audit", type=Path, default=None, help="audit_perimetre.json")
    ap.add_argument("--velo", type=Path, default=None, help="JSON report of the cohort bike check")
    ap.add_argument("--velo-vivier", type=Path, default=None, help="JSON report of the pool bike check")
    ap.add_argument("--config", type=Path, default=CONFIG_EQASIM, help="config_toulouse.yml of the eqasim fork")
    ap.add_argument("--communes", type=Path, default=COMMUNES)
    ap.add_argument("--graphe-meta", type=Path, default=GRAPHE_META, help="JSON metadata of the polygon OSMnx graph")
    ap.add_argument("--mesures-graphe", type=Path, default=None, help="mesures.json of measure_osmnx_perimeter_graph.py")
    ap.add_argument("--template", type=Path, default=TEMPLATE_V2)
    ap.add_argument("--lien-synthese", default="synthese_representativite_v3_population_v4_2026-09-03.html")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copie", type=Path, default=None)
    args = ap.parse_args(argv)
    html = build(args.sceau, args.vivier, args.audit, args.velo, args.velo_vivier, args.config, args.communes,
                 args.graphe_meta, args.mesures_graphe, args.template, args.lien_synthese)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"page written: {args.out} ({len(html.encode('utf-8')) / 1024:.0f} KB)")
    if args.copie:
        args.copie.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, args.copie)
        print(f"copy: {args.copie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
