"""Assembly of the synthesis HTML page — self-contained, offline, themeable."""
from __future__ import annotations

from html import escape
from typing import Optional, Sequence

from . import charts
from .frames import DIMENSIONS, MODE_COLORS, MODE_LABELS, MODES, pretty_cat

AGE_ORDER = ["5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
             "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-130"]
DIST_ORDER = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", "20-50km", "plus_50km"]

CSS = """
:root{--ink:#1b1b1f;--ink2:#54545e;--ink3:#82828e;--bg:#fbfbfa;--card:#ffffff;
--line:#e3e3df;--line2:#cfcfc9;--accent:#4A6FE3;--warn:#C2571A;--warnbg:#fdf4ec;
--ok:#2E7D5B;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--ink:#ececef;--ink2:#a8a8b2;--ink3:#7c7c88;
--bg:#17171a;--card:#1e1e22;--line:#2e2e34;--line2:#3d3d45;--accent:#8AA4F2;
--warn:#E08B4A;--warnbg:#2a2018;--ok:#5FB894}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:400 15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{display:grid;grid-template-columns:220px minmax(0,1fr);gap:0;max-width:1240px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;padding:28px 18px;
border-right:1px solid var(--line)}
nav h1{font-size:14px;font-weight:500;margin:0 0 4px;letter-spacing:-.01em}
nav .sub{font-size:12px;color:var(--ink3);margin-bottom:20px}
nav a{display:block;padding:5px 0;font-size:13px;color:var(--ink2);text-decoration:none;
border-left:2px solid transparent;padding-left:10px;margin-left:-10px}
nav a:hover{color:var(--ink);border-left-color:var(--line2)}
nav .grp{font-size:11px;text-transform:none;color:var(--ink3);margin:16px 0 4px;font-weight:500}
main{padding:36px 40px 96px;min-width:0}
section{margin-bottom:52px;scroll-margin-top:20px}
h2{font-size:21px;font-weight:500;margin:0 0 6px;letter-spacing:-.015em}
h3{font-size:16px;font-weight:500;margin:26px 0 8px}
h4{font-size:14px;font-weight:500;margin:18px 0 6px;color:var(--ink2)}
p{margin:0 0 12px;color:var(--ink2);max-width:74ch}
.lede{color:var(--ink2);font-size:15px;margin-bottom:20px;max-width:74ch}
code,.mono{font-family:var(--mono);font-size:12.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{font-size:11.5px;color:var(--ink3);margin-bottom:6px}
.tile .v{font-size:25px;font-weight:500;letter-spacing:-.02em;line-height:1.1}
.tile .u{font-size:12px;color:var(--ink3);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th{text-align:left;font-weight:500;color:var(--ink3);font-size:11.5px;padding:6px 10px 6px 0;
border-bottom:1px solid var(--line2)}
td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line);color:var(--ink2)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td strong{font-weight:500;color:var(--ink)}
.missing{border:1px dashed var(--warn);background:var(--warnbg);border-radius:10px;
padding:16px 18px;margin:14px 0}
.missing .t{font-size:13.5px;font-weight:500;color:var(--warn);margin-bottom:6px}
.missing p{color:var(--ink2);font-size:13.5px;margin:0 0 8px}
.missing .paths{font-family:var(--mono);font-size:12px;color:var(--ink3);margin-top:8px}
.missing .act{font-size:12.5px;margin-top:8px;color:var(--ink2)}
/* Constat de lecture (par opposition à .missing : une donnée est là, mais elle a
   un périmètre de validité) — bordure pleine, accent neutre. */
.note{border:1px solid var(--line2);border-left:3px solid var(--ok);background:var(--card);
border-radius:10px;padding:14px 18px;margin:14px 0}
.note .t{font-size:13.5px;font-weight:500;color:var(--ink);margin-bottom:6px}
.note p{color:var(--ink2);font-size:13.5px;margin:0 0 8px}
.note ul{margin:0 0 8px 18px;padding:0;color:var(--ink2);font-size:13px}
.note li{margin:2px 0}
p.warn-note{color:var(--warn);font-size:12.5px;margin:6px 0}
.badge{display:inline-block;font-size:11px;padding:2px 7px;border-radius:99px;
border:1px solid var(--line2);color:var(--ink3);margin-left:6px;vertical-align:middle}
.badge.ok{color:var(--ok);border-color:var(--ok)}
.badge.warn{color:var(--warn);border-color:var(--warn)}
.cx-legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 2px}
.cx-chip{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;color:var(--ink2)}
.cx-chip i{width:9px;height:9px;border-radius:2px;display:inline-block}
.cx-chip-note{color:var(--ink3)}
svg text{font-family:inherit}
.cx-label{font-size:11.5px;fill:var(--ink2)}
.cx-sub{font-size:12px;fill:var(--ink2)}
.cx-value{font-size:11.5px;fill:var(--ink);font-variant-numeric:tabular-nums}
.cx-delta{font-size:11px;font-variant-numeric:tabular-nums}
.cx-delta-up{fill:var(--warn)}.cx-delta-down{fill:var(--ink3)}
.cx-axis-label,.cx-tick{font-size:10.5px;fill:var(--ink3)}
.cx-tick-low{opacity:.45}
.cx-track{fill:var(--line);opacity:.55}
.cx-target{stroke:var(--ink);stroke-width:2}
.cx-axis{stroke:var(--line2);stroke-width:1}
.cx-grid{stroke:var(--line);stroke-width:1}
.cx-ref-line{fill:none;stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:4 3}
.cx-grid-cells{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;margin-top:10px}
.cx-cell{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:12px 14px}
.cx-cell-head{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.cx-cat{font-size:13px;font-weight:500}
.cx-n,.cx-l1{font-size:11px;color:var(--ink3)}
.cx-n-low{color:var(--warn)}
.cx-empty{font-size:12px;color:var(--ink3);margin:6px 0}
.cx-cell-na{fill:var(--line);opacity:.4}
.cx-na{font-size:11px;fill:var(--ink3)}
.cx-heat-value{font-size:12px;fill:var(--ink);font-variant-numeric:tabular-nums}
.formula{font-family:var(--mono);font-size:13.5px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:12px 0;overflow-x:auto}
.scroll{overflow-x:auto}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--ink3);font-size:12.5px}
@media(max-width:880px){.wrap{grid-template-columns:1fr}nav{position:static;height:auto;
border-right:0;border-bottom:1px solid var(--line)}main{padding:24px 20px 64px}}
"""


def missing_card(title: str, why: str, paths: Sequence[str] = (),
                 action: str = "") -> str:
    paths_html = ""
    if paths:
        items = "<br>".join(escape(p) for p in paths)
        paths_html = f'<div class="paths">Attendu&nbsp;: {items}</div>'
    action_html = f'<div class="act">→ {escape(action)}</div>' if action else ""
    return (f'<div class="missing"><div class="t">Données manquantes — {escape(title)}</div>'
            f'<p>{escape(why)}</p>{paths_html}{action_html}</div>')


def tiles(items: Sequence[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="tile"><div class="k">{escape(k)}</div>'
        f'<div class="v">{escape(v)}</div><div class="u">{escape(u)}</div></div>'
        for k, v, u in items)
    return f'<div class="tiles">{cells}</div>'


def _num(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


# ── Sections ─────────────────────────────────────────────────────────────────

def section_definition(payload: dict) -> str:
    sd = payload["score_def"]
    rows = "".join(
        f'<tr><td><strong>{escape(d["dim"])}</strong></td><td>{escape(d["kind"])}</td>'
        f'<td>{escape(d["metric"])}</td><td class="num">{d["weight"]:.1f}</td>'
        f'<td>{escape(d["note"])}</td></tr>' for d in sd["dimensions"])
    return f"""
<section id="definition">
<h2>Comment le score est mesuré</h2>
<p class="lede">Le score compare une <em>distribution</em> de parts modales à la référence
EMC² 2023 (enquête CEREMA, agglomération toulousaine). Il ne mesure pas si un agent
a « bien choisi » : il mesure si la population simulée se répartit entre marche, vélo,
voiture et transports collectifs comme la population enquêtée — globalement, et à
l'intérieur de chaque sous-catégorie. <strong>Plus bas est meilleur ; 0 signifie
distribution identique à l'enquête.</strong></p>

<div class="formula">composite&nbsp;=&nbsp;Σ<sub>d</sub> w<sub>d</sub> ·
s<sub>d</sub>&nbsp;&nbsp;&nbsp;avec d ∈ {{ global, absent_penalty, âge, occupation,
genre, motif, distance }}</div>

<p>Chaque dimension est mesurée avec la métrique adaptée à sa nature. Les dimensions
<strong>ordinales</strong> (âge, distance) utilisent l'<em>Earth Mover's Distance</em> :
déplacer la préférence bus des 15-19 ans vers les 20-24 ans doit coûter moins cher que
la déplacer vers les 55-59 ans, ce qu'une simple erreur L1 ignore. Les dimensions
<strong>nominales</strong> utilisent la divergence de Jensen-Shannon, bornée et
symétrique, pondérée en continu par l'effectif de la strate.</p>

<div class="scroll"><table>
<thead><tr><th>Dimension</th><th>Nature</th><th>Métrique</th><th class="num">Poids</th>
<th>Remarque</th></tr></thead><tbody>{rows}</tbody></table></div>

<h3>Le composite comparable</h3>
<p>Le moteur de calibration inclut une <code>length_penalty</code> qui pénalise les
prompts longs. Cette dimension n'a aucun sens pour la simulation ni pour le modèle
statistique — aucun des deux n'a de prompt. La page utilise donc un
<strong>composite comparable</strong> : les mêmes dimensions et les mêmes poids, mais
<code>length_penalty</code> ramenée à 0. C'est la seule modification apportée à la
loss du moteur ; tout le reste du calcul est importé tel quel depuis
<code>{escape(sd["engine"])}</code>, pour qu'un score affiché ici soit exactement
celui que la calibration optimise.</p>

<p>Deux losses sont rapportées côte à côte&nbsp;: <code>{escape(sd["primary"])}</code>
(celle qu'optimise le moteur) et <code>{escape(sd["secondary"])}</code>, qui s'exprime
en points de pourcentage et se lit directement — un L1 global de 12 signifie que la
somme des écarts absolus de parts modales vaut 12 points.</p>

<h3>Ce que la référence couvre</h3>
<p>{escape(sd["cerema_note"])}</p>
</section>"""


def _frozen_sets_note(fz: dict) -> str:
    """Declares the divergence between the frozen sets and the population in service.

    Rendered only if it exists: a permanent warning stops being read.
    """
    if not fz or not fz.get("diverges"):
        return ""
    frozen = str(fz.get("frozen_population_sha256") or "")[:12]
    current = str(fz.get("run_population_sha256") or "")[:12]
    return f"""
<div class="note"><div class="t">Les jeux gelés du volet 2 ne portent plus la population
du run</div>
<p>Le manifeste des jeux gelés épingle une population d'empreinte
<code>{escape(frozen)}</code>&nbsp;; le run de cette page en porte une d'empreinte
<code>{escape(current)}</code>. Les jeux gelés ont été découpés dans un run antérieur, et
depuis, la population en service a changé — les tickets 016 et 017 y réécrivent
l'abonnement TC et le permis, que les jeux gelés gardent recopiés du donneur ENTD 2008.</p>
<p>Conséquence de lecture&nbsp;: les scores du volet 2 mesurés <strong>sur jeux gelés</strong>
décrivent une population que la simulation ne joue plus. Ils restent comparables
<em>entre eux</em>, ce qui est leur rôle — suivre une trajectoire de prompt — mais pas au
volet 1 ni au volet 3, qui portent la population courante. Le témoin d'échantillon du
volet 2 sur le jeu commun, lui, est calculé sur le run et n'est pas concerné.</p>
<p>La divergence est <strong>déclarée et non corrigée</strong>&nbsp;: refaire les jeux gelés
casserait la comparabilité de toute la trajectoire de calibration déjà mesurée.
Manifeste&nbsp;: <code>{escape(str(fz.get("manifest") or "—"))}</code>.</p></div>"""


def section_common_set(payload: dict) -> str:
    cs = payload["common_set"]
    if not cs.get("available"):
        return f"""
<section id="jeu-commun"><h2>Jeu d'évaluation commun</h2>
{missing_card("run de simulation introuvable", cs.get("reason", ""),
              cs.get("expected", []),
              "Renseigner common_set.run dans scripts/synthesis/sources.yaml")}
</section>"""

    cov_dims = [d for d in DIMENSIONS if d["key"] in cs["coverage"]]
    head = "".join(f"<th>{escape(d['label'])}</th>" for d in cov_dims)
    body = ""
    max_cats = max((len(cs["coverage"][d["key"]]) for d in cov_dims), default=0)
    cells_by_dim = {d["key"]: list(cs["coverage"][d["key"]].items()) for d in cov_dims}
    for i in range(max_cats):
        row = ""
        for d in cov_dims:
            items = cells_by_dim[d["key"]]
            if i >= len(items):
                row += "<td></td>"
                continue
            cat, n = items[i]
            cls = "" if n >= 5 else ' style="color:var(--warn)"'
            row += f'<td{cls}>{escape(pretty_cat(cat))} <span class="mono">{n}</span></td>'
        body += f"<tr>{row}</tr>"

    warn = ""
    if cs.get("warnings"):
        warn = "".join(f"<li>{escape(w)}</li>" for w in cs["warnings"])
        warn = f'<ul style="color:var(--ink2);font-size:13.5px">{warn}</ul>'

    # Vehicle chain constraints (ticket 008, A4). A decision taken on an already
    # restricted set of options is not invalid, but it does not measure the same
    # thing as a free choice: the reader must know what share of strand 1 it
    # represents. The block disappears on a run older than the column.
    _CONSTRAINT_LABEL = {
        "aucune": "Aucune contrainte — jeu de choix d'OTP",
        "retour_force": "Retour forcé — options restreintes au mode du véhicule garé",
        "passager": "Passager — voiture conduite par un adulte du foyer",
        "sortie_bloquee": "Sortie bloquée — véhicule possédé, mais garé ailleurs",
    }
    constraints = cs.get("chain_constraints") or {}
    chain = ""
    if set(constraints) - {"aucune"}:
        total_c = max(1, sum(constraints.values()))
        lines = "".join(
            f"<tr><td>{escape(_CONSTRAINT_LABEL.get(k, k))}</td>"
            f'<td class="mono">{v}</td>'
            f'<td class="mono">{100.0 * v / total_c:.1f}%</td></tr>'
            for k, v in sorted(constraints.items(), key=lambda kv: -kv[1]))
        chain = f"""
<h3>Contraintes de chaîne des véhicules</h3>
<p>Un véhicule personnel est un <em>lieu</em>&nbsp;: il reste où son propriétaire l'a
garé. Les décisions ci-dessous ont donc été prises sur un jeu d'options déjà restreint
par cette cohérence. Elles <strong>restent dans le score</strong> — c'est bien ce que la
simulation a joué — mais elles ne mesurent pas la même chose qu'un choix libre entre
tous les modes, et leur part doit être lue avec le reste.</p>
<div class="scroll"><table><thead><tr><th>Contrainte</th><th>Trajets</th>
<th>Part</th></tr></thead><tbody>{lines}</tbody></table></div>"""

    return f"""
<section id="jeu-commun">
<h2>Jeu d'évaluation commun</h2>
<p class="lede">Les trois volets sont évalués sur le <strong>même substrat</strong>&nbsp;:
un run de simulation. C'est le seul terrain où ils peuvent se rencontrer — il fournit
à la fois les personas complets, les jeux de choix OTP réellement proposés, et les
coordonnées origine/destination dont le modèle statistique a besoin. Les jeux gelés de
la calibration sont d'ailleurs eux-mêmes construits à partir d'un run de ce type — mais
pas forcément de <em>celui-ci</em>, cf. ci-dessous.</p>
{tiles([
    ("Run", cs["run_id"], "épinglé dans le manifeste" if cs.get("run_pinned")
     else "⚠ chemin non épinglé — la page suivra le prochain run"),
    ("Trajets retenus", f'{cs["n_trips"]:,}'.replace(",", " "), "après exclusions"),
    ("Personnes", f'{cs["n_persons"]:,}'.replace(",", " "), "identifiants distincts"),
    ("Avec distribution", f'{cs["pct_distribution"]:.0f}%', "des trajets"),
    ("Jour simulé", cs.get("sim_day") or "tous",
     f'{cs.get("n_excluded_day", 0)} lignes écartées hors de ce jour'),
])}
{warn}
{chain}
{_frozen_sets_note(cs.get("frozen_sets") or {})}
<h3>Couverture par sous-catégorie</h3>
<p>Effectif en <em>personnes distinctes</em>. Les catégories sous le seuil de 5 sont
signalées&nbsp;: elles restent affichées mais ne pèsent quasiment rien dans le score,
et leurs écarts ne sont pas interprétables.</p>
<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
</section>"""


def _dimension_blocks(details: dict, prefix: str) -> str:
    out = []
    for dim in DIMENSIONS:
        detail = details.get(dim["key"])
        anchor = f"{prefix}-{dim['key']}"
        badge = ('<span class="badge ok">dans le composite</span>' if dim["scored"]
                 else '<span class="badge">hors composite</span>')
        if not detail:
            out.append(f'<h3 id="{anchor}">{escape(dim["label"])} {badge}</h3>'
                       + missing_card(
                           dim["label"],
                           "Aucune donnée exploitable pour cette dimension dans le "
                           "jeu commun.",
                           [], "Voir la liste d'actions en bas de page"))
            continue
        if dim["kind"] == "ordinal":
            order = AGE_ORDER if dim["key"] == "age" else DIST_ORDER
            body = charts.ordinal_profiles(detail, order=order)
            body += charts.legend(
                [(MODE_LABELS[m], MODE_COLORS[m]) for m in MODES],
                'ligne pointillée = référence EMC²')
        else:
            body = charts.nominal_groups(detail)
        out.append(f'<h3 id="{anchor}">{escape(dim["label"])} {badge}</h3>{body}')
    return "".join(out)


def _score_tiles(scores: dict, label: str) -> str:
    items = []
    for name, values in scores.items():
        items.append((f"{label} · {name}", _num(values.get("composite")), "composite"))
    return tiles(items) if items else ""


def section_simulation(payload: dict) -> str:
    arm = payload["arms"]["simulation"]
    if arm.get("status") != "ok":
        return f"""
<section id="volet-simulation"><h2>Volet 1 — Simulation (LLM + tirage)</h2>
{missing_card("volet simulation", arm.get("reason", ""), arm.get("expected", []),
              arm.get("action", ""))}</section>"""

    expected, drawn = arm["variants"]["attendu"], arm["variants"]["tire"]
    gview = expected["global"]
    worst = "".join(
        f'<tr><td>{escape(w["dim"])}</td><td>{escape(pretty_cat(w["cat"]))}</td>'
        f'<td>{escape(MODE_LABELS[w["mode"]])}</td>'
        f'<td class="num">{w["actual"]:.1f}</td><td class="num">{w["target"]:.1f}</td>'
        f'<td class="num" style="color:var(--warn)">{w["diff"]:+.1f}</td>'
        f'<td class="num">{w["n"]}</td></tr>' for w in arm["worst_strata"])

    return f"""
<section id="volet-simulation">
<h2>Volet 1 — Simulation (LLM + tirage)</h2>
<p class="lede">Le mode est choisi comme aujourd'hui&nbsp;: le LLM attribue une
probabilité à chaque itinéraire proposé, puis la simulation tire au sort dans cette
distribution (<code>draw_index</code>, graine déterministe). Deux lectures du même run
sont rapportées — la <strong>masse de probabilité</strong> (ce que la calibration
optimise, indépendant du tirage) et le <strong>mode effectivement tiré</strong>.
L'écart entre les deux est le bruit d'échantillonnage introduit par le tirage.</p>

{tiles([
    ("Composite comparable", _num(expected["scores"].get(payload["score_def"]["primary"], {}).get("composite")), "masse de probabilité"),
    ("L1 global", f'{gview["l1"]:.1f}', "points de %"),
    ("Composite (tiré)", _num(drawn["scores"].get(payload["score_def"]["primary"], {}).get("composite")), "après tirage"),
    ("Masse hors périmètre", f'{arm["excluded_pct"]:.1f}%', "2-roues motorisé, autres"),
])}

<h3>Parts modales globales</h3>
{charts.global_bullet(gview)}
{charts.legend([(MODE_LABELS[m], MODE_COLORS[m]) for m in MODES],
               'trait vertical = cible EMC² · chiffre de droite = écart signé')}

<h3>Pires croisements</h3>
<p>Classés par impact (écart absolu × effectif) — les strates où corriger rapporterait
le plus.</p>
<div class="scroll"><table><thead><tr><th>Dimension</th><th>Catégorie</th><th>Mode</th>
<th class="num">Observé</th><th class="num">EMC²</th><th class="num">Écart</th>
<th class="num">n</th></tr></thead><tbody>{worst}</tbody></table></div>

<h3>Détail par sous-catégorie</h3>
<p>Aussi disponible en page dédiée, sans le reste de la synthèse&nbsp;:
<a href="detail_simulation.html">détail par sous-catégorie — simulation</a>.</p>
{_dimension_blocks(arm["details"], "sim")}
</section>"""


def _lineage_block(lineage: dict) -> str:
    """The pinned lineage, measured under each regime that covers it.

    Two levels of reading. Under **one** regime, the curve reads end to end as
    the effect of the prompt — that is what action A5 produces. Under **several**, the
    comparison of the curves answers another, more interesting question: does the
    calibration gain survive the change of instrument, or does it only exist
    in the instrument that served to optimise it?
    """
    regimes = lineage.get("regimes") or [lineage]
    series = [{"label": r["label"],
               "points": [{"score": s["score"],
                           "label": ("graine" if i == 0 else f'#{i} · {s["short"]}')}
                          for i, s in enumerate(r["steps"]) if s["score"] is not None]}
              for r in regimes]

    def verdict(r: dict) -> str:
        gain, seed = r.get("gain"), r.get("seed_score")
        if gain is None or not seed:
            return (f'<li><span class="mono">{escape(r["label"])}</span> — trop peu de '
                    f'nœuds mesurés pour conclure.</li>')
        sense = "gagne" if gain > 0 else "perd"
        partial = ("" if r["complete"] else
                   f' <span class="badge warn">{r["n_scored"]}/{lineage["n_nodes"]} '
                   f'nœuds</span>')
        return (f'<li><span class="mono">{escape(r["label"])}</span> — la lignée '
                f'{sense} <strong>{abs(gain):.2f}</strong> point(s)&nbsp;: '
                f'{r["seed_score"]:.2f} → {r["leaf_score"]:.2f} '
                f'({-100.0 * gain / seed:+.1f} %){partial}</li>')

    # Table columns: one composite score per regime, plus the gap to the seed of the
    # main regime (the one carrying the reference reading).
    heads = "".join(f'<th class="num">{escape(r["label"])}</th>' for r in regimes)
    rows = ""
    for i, step in enumerate(regimes[0]["steps"]):
        cells = ""
        for r in regimes:
            s = r["steps"][i]
            cells += f'<td class="num">{_num(s["score"])}</td>'
        rank = "graine" if i == 0 else f"mutation {i}"
        rows += (f'<tr><td class="mono">{escape(step["short"])}</td><td>{rank}</td>'
                 f'<td>{escape(step["branch"])}</td>{cells}</tr>')

    incomplete = "".join(
        f'<p class="warn-note">⚠ <span class="mono">{escape(r["label"])}</span>&nbsp;: '
        f'{lineage["n_nodes"] - r["n_scored"]} nœud(s) de la lignée sans évaluation sous '
        f'ce régime, trajectoire incomplète. <code>calibrate reeval</code> complète les '
        f'manquants (les autres sont servis par le cache).</p>'
        for r in regimes if not r["complete"])

    pin = ""
    if lineage.get("pinned_regime") and not lineage.get("is_pinned"):
        pin = (f'<p class="warn-note">⚠ Le régime épinglé dans <code>sources.yaml</code> '
               f'(<span class="mono">{escape(lineage["pinned_regime"])}</span>) ne couvre '
               f'pas encore cette lignée&nbsp;: la lecture de référence est assurée par '
               f'<span class="mono">{escape(regimes[0]["label"])}</span> à sa place.</p>')

    cross = ""
    if len(regimes) > 1:
        gains = [r["gain"] for r in regimes if r.get("gain") is not None]
        if len(gains) > 1:
            if all(g > 0 for g in gains):
                cross = (
                    '<p><strong>Tous les régimes voient la lignée s\'améliorer.</strong> '
                    'La calibration n\'a donc pas seulement flatté l\'instrument qui l\'a '
                    'guidée : son effet se retrouve sous un modèle et une politique de '
                    'décision différents. L\'<em>ampleur</em>, elle, dépend du régime et '
                    'ne se transporte pas.</p>')
            elif all(g < 0 for g in gains):
                cross = (
                    '<p class="warn-note">⚠ <strong>Tous les régimes voient la lignée se '
                    'dégrader.</strong> Les mutations acceptées l\'ont été sur une mesure '
                    'que la loss courante contredit : le gain d\'alors est un artefact de '
                    'la loss d\'alors, pas un progrès.</p>')
            else:
                cross = (
                    '<p class="warn-note">⚠ <strong>Les régimes ne s\'accordent pas sur le '
                    'sens de l\'évolution.</strong> Ce que la calibration a amélioré sous '
                    'un instrument se dégrade sous l\'autre : ce qui est mesuré est un '
                    'effet de l\'instrument, pas une propriété du prompt.</p>')

    return (f'<h3>Lignée mesurée sous un régime unique</h3>'
            f'<p>Les courbes ci-dessus suivent l\'ordre chronologique&nbsp;: elles mêlent '
            f'des branches et des nœuds sans parenté. Celle-ci suit une '
            f'<strong>lignée</strong> — la chaîne des mutations acceptées, de la graine à '
            f'la feuille <code>{escape(lineage["leaf"])}</code>, '
            f'{lineage["n_nodes"]}&nbsp;nœuds — mesurée entièrement sous un même régime. '
            f'C\'est la seule trajectoire de cette page qui se lise bout à bout comme '
            f'l\'effet du prompt.</p>'
            + charts.trajectory(series, caption="ordre de la lignée : graine → feuille →")
            + f'<ul>{"".join(verdict(r) for r in regimes)}</ul>'
            + cross + pin + incomplete
            + f'<div class="scroll"><table><thead><tr><th>Nœud</th><th>Rang</th>'
              f'<th>Branche</th>{heads}</tr></thead><tbody>{rows}</tbody></table></div>')


def _train_version_note(arm: dict) -> str:
    """Version-qualified train evals, present in the store but off the curve.

    They are not merged into the trajectory: two versions of frozen sets
    do not have the same weather, hence not the same substrate, and overlaying them in one
    curve is the defect that version qualification comes to fix. But
    a paid measurement that vanishes without a word gets the eval rerun a second
    time — hence this note.
    """
    skipped = arm.get("train_versioned_skipped") or {}
    if not skipped:
        return ""
    listing = "".join(
        f'<li><span class="mono">{escape(k)}</span> — {v} éval(s)</li>'
        for k, v in sorted(skipped.items()))
    return f"""
<div class="note"><div class="t">Des évals de train qualifiées par version sont dans le
store, hors de cette courbe</div>
<p>La trajectoire ci-dessous ne porte que le split <span class="mono">train</span> à nom
nu (jeux <span class="mono">v1</span>). Le store contient aussi&nbsp;:</p>
<ul>{listing}</ul>
<p>Elles sont écartées et non fondues&nbsp;: une autre version de jeux gelés, c'est un
autre régime météo, donc un autre substrat. Les mêler dans une seule courbe est
exactement ce que la qualification par version corrige. Pour lire la trajectoire sous
cette version, épingler la version correspondante dans le manifeste.</p></div>"""


def _regimes_note(arm: dict) -> str:
    """What mixing measurement regimes forbids — and what escapes it.

    The warning is about the **level** of the scores: a composite score is only
    comparable to those measured under the same model AND the same decision
    policy. Once a lineage is replayed under a single regime (action A5), the
    finding stops being a gap and becomes a reading instruction.
    """
    if not arm.get("mixed_models"):
        return ""
    regimes = sorted({r for s in arm["stores"] for r in (s.get("eval_models") or [])})
    listing = "".join(f"<li><span class=\"mono\">{escape(r)}</span></li>" for r in regimes)
    lineage = next((s["lineage"] for s in arm["stores"]
                    if s.get("lineage") and not s.get("subset_of")), None)
    if lineage:
        return f"""
<div class="note"><div class="t">Les niveaux de score se lisent régime par régime</div>
<p>Le store porte {len(regimes)} régimes de mesure&nbsp;: un score n'est comparable
qu'aux scores obtenus sous le même modèle <em>et</em> la même politique de décision
(mode élu contre masse de probabilité — la seconde change les décisions elles-mêmes,
donc aucun recalcul de loss ne les réconcilie).</p>
<ul>{listing}</ul>
<p>La <a href="#volet-calibration">lignée ci-dessous</a> échappe à cette limite&nbsp;:
ses {lineage["n_nodes"]} nœuds sont tous mesurés sous
<span class="mono">{escape(lineage["label"])}</span> — c'est le seul endroit de la page
où une trajectoire se lit bout à bout comme l'effet du prompt.</p></div>"""
    return missing_card(
        "effet du prompt et effet du régime de mesure sont confondus",
        "Une fois la loss unifiée, les familles de nœuds occupent la même plage de "
        "score. Mais les décisions sous-jacentes viennent de modèles et de politiques "
        "différents : ce qui reste ne se lit pas comme l'effet du prompt seul. Les "
        "courbes sont facettées par régime et ne doivent pas être lues bout à bout.",
        [], "Action A5 — rejouer une lignée sous un modèle d'évaluation unique")


def _common_set_block(arm: dict) -> str:
    """Strand 2 measured on the common set (action A3), against its frozen-set score.

    The whole difficulty of reading this page lies in the distinction this
    block makes explicit: a calibration composite score can be measured on the
    **frozen personas** — the set the optimisation loop worked on —
    or on the **common set**, the personas of the pinned run that the other
    strands also score. They are two different numbers for the same prompt, and only the
    second enters the final comparison.
    """
    common = arm.get("common_set") or {}
    if not common.get("available"):
        # A measurement set aside because it is about ANOTHER run is not
        # missing data: the file exists and it is right. The card must say
        # which of the two situations it describes, otherwise the reader reruns
        # a measurement already paid for.
        why = common.get("reason") or (
            "Aucun prompt n'a encore été ré-évalué sur le jeu commun issu de la "
            "simulation. Tant que ce n'est pas fait, le volet 2 ne peut pas entrer "
            "dans la comparaison finale : ses scores portent sur les personas gelés, "
            "pas sur le run.")
        return missing_card(
            "prompts évalués sur le jeu commun", why,
            arm.get("common_set_expected", []),
            common.get("action")
            or "Action A3 — ré-évaluer la graine et le meilleur prompt sur le jeu commun")

    sample = common.get("sample") or {}
    rows = ""
    for e in common["entries"]:
        frozen = e.get("frozen_composite")
        delta = (e["composite"] - frozen) if (frozen is not None
                                              and e["composite"] is not None) else None
        colour = "var(--warn)" if (delta or 0) > 0 else "var(--ok)"
        rows += (f'<tr><td><strong>{escape(e["label"])}</strong> '
                 f'<code>{escape(e["short"])}</code></td>'
                 f'<td>{escape(e.get("branch") or "—")}</td>'
                 f'<td class="num">{_num(e["composite"])}</td>'
                 f'<td class="num">{_num(frozen)}</td>'
                 f'<td class="num" style="color:{colour}">'
                 f'{"—" if delta is None else f"{delta:+.2f}"}</td></tr>')

    gain, frozen_gain = common.get("gain"), common.get("frozen_gain")
    seed_c = (common.get("seed") or {}).get("composite")
    pct = (100.0 * gain / seed_c) if (gain is not None and seed_c) else None
    verdict = ""
    if gain is not None and frozen_gain is not None:
        moved = "se transporte" if gain > 0 else "ne se transporte pas"
        verdict = (
            f'<p>Le gain de la lignée <strong>{escape(moved)}</strong> sur le jeu '
            f'commun&nbsp;: {gain:+.2f} point(s) de composite entre la graine et la '
            f'feuille, contre {frozen_gain:+.2f} sur les personas gelés, sous le même '
            f'régime de mesure. Un prompt optimisé sur un jeu peut parfaitement ne pas '
            f'rendre le même service ailleurs&nbsp;: c\'est ce que ce couple de '
            f'chiffres permet enfin de vérifier plutôt que de supposer. Ici le gain '
            f'survit presque à l\'identique&nbsp;— mais le <em>niveau</em>, lui, ne '
            f'survit pas.</p>')

    # The level shift between the two substrates is the most visible thing in the
    # table above. Part of it is explained by sample size alone,
    # and the control puts a number on it: not saying so would let a gap that comes
    # from the number of persons observed be attributed to the prompt.
    size = common.get("size_control") or {}
    size_html = ""
    if size.get("penalty") is not None:
        seed_e, leaf_e = common.get("seed") or {}, common.get("leaf") or {}
        shift = ((seed_e.get("composite") or 0) - (seed_e.get("frozen_composite") or 0))
        size_html = f"""
<p><strong>Combien de ce décalage vient de la taille de l'échantillon&nbsp;?</strong>
Les divergences par strate (JSD, EMD) sont biaisées vers le haut quand les effectifs
sont petits&nbsp;: mesurer sur 81 personnes n'est pas mesurer sur 881. Le témoin le
chiffre sur ce run, sans le moindre appel LLM&nbsp;— la simulation, restreinte
<em>aux mêmes {size.get("n_persons", "?")} personnes</em>, passe de
{_num(size.get("full_composite"))} à {_num(size.get("composite"))}, soit
<strong>{size["penalty"]:+.2f} point(s)</strong> pour la seule réduction d'effectif, à
décisions inchangées. C'est donc à {_num(size.get("composite"))} — la ligne
«&nbsp;Sim. (éch. V2)&nbsp;» de la matrice — que les deux colonnes de calibration
doivent être comparées, et non à {_num(size.get("full_composite"))}. Le décalage
graine gelé → graine jeu commun étant de {shift:+.2f}, l'effet de taille en explique
une part et le reste tient au substrat lui-même&nbsp;: d'autres trajets, d'autres
contextes, d'autres motifs que ceux sur lesquels la lignée a été optimisée.</p>"""

    warns = sample.get("coverage_warnings") or []
    warn_html = ""
    if warns:
        items = "".join(f"<li>{escape(w)}</li>" for w in warns)
        warn_html = (f'<p>Strates sous le seuil de 5 dans l\'échantillon&nbsp;:</p>'
                     f'<ul>{items}</ul>')
    splits = sample.get("splits") or {}
    splits_html = ", ".join(f"{escape(k)}&nbsp;{v}" for k, v in sorted(splits.items()))

    return f"""
<h3>Ré-évaluation sur le jeu commun</h3>
<p>Les deux extrémités de la lignée épinglée, rejouées sur un échantillon
<strong>du run</strong> — les mêmes personas que le volet&nbsp;1 — sous le régime
<span class="mono">{escape(common.get("regime") or "régime hétérogène")}</span>.
C'est <em>ce</em> chiffre qui entre dans la synthèse comparative&nbsp;; le composite
mesuré sur les personas gelés reste affiché en regard, et les deux ne se confondent
pas.</p>
{tiles([
    ("Graine", _num((common.get("seed") or {}).get("composite")), "composite, jeu commun"),
    ("Meilleur", _num((common.get("leaf") or {}).get("composite")), "composite, jeu commun"),
    ("Gain", "—" if gain is None else f"{gain:+.2f}",
     "—" if pct is None else f"{pct:+.1f} % du niveau de la graine"),
    ("Échantillon", f'{sample.get("n_records", "?")}',
     f'décisions · {sample.get("n_agents", "?")} personnes'),
])}
<div class="scroll"><table><thead><tr><th>Prompt</th><th>Branche</th>
<th class="num">Composite — jeu commun</th><th class="num">Composite — personas gelés</th>
<th class="num">Δ</th></tr></thead><tbody>{rows}</tbody></table></div>
{verdict}
{size_html}
<p>Échantillon gelé et reproductible&nbsp;: <span class="mono">{escape(sample.get("rule", "—"))}</span>
— tirage <strong>par personne</strong> (tous ses trajets, jamais coupés), même famille
de règle que les jeux gelés du moteur, mais dans un espace de hachage distinct pour que
l'échantillon ne soit pas un préfixe du split d'entraînement. Composition en splits
gelés&nbsp;: {splits_html or "—"} — la calibration ayant été optimisée sur le train d'un
run <em>antérieur</em>, une part de l'échantillon porte des personnes déjà vues, avec
d'autres trajets et d'autres contextes. Sur des personnes <em>jamais</em> vues, c'est le
bloc «&nbsp;Généralisation&nbsp;» ci-dessous qui répond (action A4).</p>
{warn_html}
<div class="formula">python -m scripts.synthesis.common_set_eval&nbsp;&nbsp;&nbsp;# ou : make common-set-eval</div>"""


def _generalization_block(arm: dict) -> str:
    """What calibration is worth outside the set that served to optimise it (action A4).

    The block carries three things, and none can be removed without making the
    figure misleading:

    - the **score on the held-out set**, the one the loop has never seen;
    - the **sample-size control**, because the two sets do not have the same size
      and the per-stratum divergences are biased upwards at small
      counts — without it, the raw gap would read as overfitting;
    - the **exact nature of the split**, established on the files themselves.
      "Generalisation" does not mean the same thing depending on whether the split
      is by person (never-seen individuals) or by trip (other trips
      of the same individuals).
    """
    gen = arm.get("generalization") or {}
    if not gen.get("available"):
        return missing_card(
            "score sur le jeu de retenue",
            gen.get("reason", "Aucune évaluation sur le jeu de test gelé."),
            [f'store · dataset « {gen.get("dataset", "test")} » · '
             f'régime {gen.get("regime") or "épinglé"}'],
            gen.get("action", "Action A4 — évaluer sur le jeu de test gelé"))

    ds = escape(gen["dataset"])

    # Different frozen-set regimes on the two sides: to be said BEFORE the table, because
    # everything that follows — raw gap, corrected gap, verdict — reads differently.
    version_warning = ""
    if not gen.get("versions_match", True):
        version_warning = f"""
<div class="constat"><p><strong>Les deux colonnes ne portent pas le même régime
météo.</strong> Le score d'entraînement vient des évaluations de la campagne, faites sur
les jeux gelés <code>{escape(gen.get("train_version", "v1"))}</code>&nbsp;; le score de
retenue vient de la ré-évaluation, faite sur
<code>{escape(gen.get("held_version", "?"))}</code>. Or les jeux
<code>{escape(gen.get("train_version", "v1"))}</code> ne contiennent que cinq valeurs
météo, toutes «&nbsp;ciel dégagé, pas de précipitations&nbsp;» — la fenêtre du run source
était sèche — là où <code>{escape(gen.get("held_version", "?"))}</code> tire la météo dans
l'année climatique complète (≈&nbsp;43&nbsp;% de records avec précipitations).</p>
<p>L'écart train&nbsp;→&nbsp;retenue ci-dessous ne mesure donc <strong>pas</strong> le seul
effet du découpage&nbsp;: il porte aussi ce changement de régime. Le témoin d'effectif ne
le neutralise pas — il est calculé par rééchantillonnage des décisions
<code>{escape(gen.get("train_version", "v1"))}</code>. Pour un écart de généralisation pur,
il faudrait mesurer aussi le train sur
<code>{escape(gen.get("held_version", "?"))}</code>&nbsp;; ce n'est pas fait, et ce n'est
pas gratuit.</p></div>"""

    rows = ""
    for step in gen["steps"]:
        ctl = step.get("control") or {}
        raw, corrected = step.get("gap_raw"), step.get("gap_controlled")
        # The colour follows the CORRECTED gap, never the raw one: it is the only one
        # of the two that speaks of the prompt rather than of the size of the sets.
        colour = ("var(--ink3)" if corrected is None
                  else "var(--warn)" if corrected > 0 else "var(--ok)")
        band = (f'{_num(ctl.get("mean"))} <span style="color:var(--ink3)">'
                f'[{_num(ctl.get("p05"))}&nbsp;;&nbsp;{_num(ctl.get("p95"))}]</span>'
                if ctl else "—")
        rows += (f'<tr><td><strong>{escape(step["label"])}</strong> '
                 f'<code>{escape(step["short"])}</code></td>'
                 f'<td class="num">{_num(step.get("train"))}</td>'
                 f'<td class="num">{_num(step.get("held"))}</td>'
                 f'<td class="num">{"—" if raw is None else f"{raw:+.2f}"}</td>'
                 f'<td class="num">{band}</td>'
                 f'<td class="num" style="color:{colour}">'
                 f'{"—" if corrected is None else f"{corrected:+.2f}"}</td></tr>')

    leaf, seed = gen.get("leaf") or {}, gen.get("seed") or {}
    ctl = leaf.get("control") or {}
    gain_train, gain_held = gen.get("gain_train"), gen.get("gain_held")

    # Verdict. It reads on the leaf's corrected gap, and it is bracketed by the
    # spread of the control: a gap smaller than the width of the band
    # demonstrates nothing, in either direction.
    verdict = ""
    corrected = leaf.get("gap_controlled")
    if corrected is not None and ctl:
        spread = ctl["p95"] - ctl["p05"]
        if corrected > 0:
            head = (f"la calibration est <strong>moins fidèle</strong> sur le jeu de "
                    f"retenue que sur un tirage de même taille du jeu "
                    f"d'entraînement&nbsp;: {corrected:+.2f} point(s)")
        else:
            head = (f"la calibration est <strong>au moins aussi fidèle</strong> sur le "
                    f"jeu de retenue que sur un tirage de même taille du jeu "
                    f"d'entraînement&nbsp;: {corrected:+.2f} point(s)")
        significance = (
            "et cet écart <strong>ne franchit pas</strong> la dispersion du témoin"
            if leaf.get("within_control") else
            "et cet écart <strong>sort</strong> de la dispersion du témoin")
        verdict = f"""
<p><strong>Verdict.</strong> À effectif neutralisé, {head} — {significance}
({_num(ctl.get("p05"))} à {_num(ctl.get("p95"))}, soit {spread:.1f} points de large sur
{ctl.get("n_draws", "?")} tirages). Autrement dit&nbsp;: <strong>aucun surapprentissage
détectable</strong>, mais un jeu de retenue trop petit pour trancher finement. L'écart
brut de {"—" if leaf.get("gap_raw") is None else f'{leaf["gap_raw"]:+.2f}'} point(s) que
montre la troisième colonne s'explique <em>entièrement</em> par la réduction d'effectif
de {gen.get("train_agents", "?")} à {gen.get("n_agents", "?")} personnes&nbsp;: le lire
comme un surapprentissage serait une erreur, et c'est exactement l'erreur que ce tableau
existe pour éviter.</p>"""

    gain_html = ""
    if gain_train is not None and gain_held is not None:
        stronger = gain_held > gain_train
        # The gain control is PAIRED — both prompts are scored on the same
        # drawn persons — hence much tighter than the per-node control. It is what
        # permits a conclusion, and it alone.
        gc = gen.get("gain_control") or {}
        gain_verdict = ""
        if gc:
            outside = not (gc["p05"] <= gain_held <= gc["p95"])
            width = gc["p95"] - gc["p05"]
            # The ratio of widths is COMPUTED, not asserted: pairing
            # narrows the band, but by how much depends on the data of the day.
            node_width = ((ctl["p95"] - ctl["p05"]) if ctl else None)
            narrower = (f" — soit {node_width / width:.1f} fois plus serrée que celle "
                        f"des niveaux" if (node_width and width) else "")
            gain_verdict = (
                f' À effectif neutralisé, le gain mesuré sur le train vaut '
                f'{gc["mean"]:+.2f} en moyenne sur {gc["n_draws"]} tirages appariés '
                f'(bande {gc["p05"]:+.2f} à {gc["p95"]:+.2f}, {width:.1f} points de '
                f'large{narrower}, parce que les deux prompts y sont scorés sur '
                f'<em>les mêmes</em> personnes). Le gain de {gain_held:+.2f} obtenu sur '
                f'le jeu de retenue '
                + ("<strong>sort de cette bande</strong>&nbsp;: l\'amplification n\'est "
                   "pas un accident d\'échantillonnage."
                   if outside else
                   "<strong>reste dans cette bande</strong>&nbsp;: l\'amplification "
                   "n\'est donc <em>pas</em> démontrée, et la page ne la revendique "
                   "pas. Ce qui est acquis, c\'est que le gain <em>survit</em> — il "
                   "n\'était pas un artefact du jeu qui a servi à l\'obtenir."))
        gain_html = f"""
<p><strong>Le gain de la lignée se transporte.</strong>
Entre la graine et la feuille, la calibration gagne {gain_train:+.2f} point(s) sur le jeu
d'entraînement et {gain_held:+.2f} sur le jeu de retenue&nbsp;—
{"davantage" if stronger else "moins"}, en valeur brute, là où elle n'a jamais été
optimisée.{gain_verdict}
La même tendance se lit sur le témoin par nœud&nbsp;: la graine paie plein tarif l'effet
d'effectif ({"—" if seed.get("gap_raw") is None else f'{seed["gap_raw"]:+.2f}'} brut), la
feuille presque pas ({"—" if leaf.get("gap_raw") is None else f'{leaf["gap_raw"]:+.2f}'}).
Le prompt calibré <em>paraît</em> donc plus stable d'une population à l'autre&nbsp;; à la
précision que permettent {gen.get("n_agents", "?")} personnes, c'est une tendance et non
une démonstration.</p>"""

    # Nature of the split. The point on which the reader must not be left to
    # assume: a page that says "generalisation" without saying of what implies
    # the strongest claim.
    if gen.get("by_person"):
        nature = (f"<strong>par personne</strong>&nbsp;: les "
                  f"{gen.get('n_agents', '?')} personnes du jeu «&nbsp;{ds}&nbsp;» "
                  f"n'apparaissent <em>dans aucun</em> des "
                  f"{gen.get('train_agents', '?')} personas du train — vérifié sur les "
                  f"fichiers, pas sur la foi de la règle déclarée. La généralisation "
                  f"mesurée ici est donc celle qu'on croit&nbsp;: des <em>individus "
                  f"jamais vus</em>, et non d'autres trajets des mêmes individus.")
    else:
        nature = (f"<strong>par déplacement</strong>&nbsp;: "
                  f"{gen.get('agents_shared_with_train')} personne(s) du jeu "
                  f"«&nbsp;{ds}&nbsp;» sont aussi dans le train. La généralisation "
                  f"mesurée porte sur des <em>trajets</em>, pas sur des individus — "
                  f"affirmation nettement plus faible, et la page ne doit pas laisser "
                  f"croire l'autre.")

    # Residual confounding, and it is real: the engine removes the « Historique »
    # section from the val/test sets (STM/LTM memory of the source run, not
    # reproducible) and keeps it in train. Keeping quiet about it would pass off as a pure
    # population effect what is also a change of input form.
    memory_html = ""
    m_train, m_held = gen.get("memory_train"), gen.get("memory_held")
    if m_train is not None and m_held is not None and abs(m_train - m_held) > 0.01:
        memory_html = f"""
<p><strong>Une confusion résiduelle, qui n'est pas une population.</strong> Les personas
du train portent une section <span class="mono">**Historique&nbsp;:**</span> — la mémoire
STM/LTM du run source — dans {m_train:.0%} de leurs records&nbsp;; ceux du jeu
«&nbsp;{ds}&nbsp;» dans {m_held:.0%}. Le moteur la retire délibérément des jeux de
retenue, cette mémoire n'étant pas reproductible d'un run à l'autre. Conséquence&nbsp;:
le prompt de test n'est pas seulement adressé à d'autres personnes, il est aussi plus
court d'une section. L'écart mesuré ci-dessus mêle donc deux effets, et rien dans les
données disponibles ne permet de les séparer&nbsp;— il faudrait une éval du train
lui-même privé de sa mémoire, soit une mesure de plus.</p>"""

    complete = ("" if gen.get("complete") else
                f'<p><span class="badge warn">{gen["n_measured"]}/{gen["n_nodes"]} '
                f'nœuds</span> de la lignée sont mesurés sur «&nbsp;{ds}&nbsp;»&nbsp;: '
                f'les autres restent sans score de retenue.</p>')

    return f"""
<h3>Généralisation — le jeu que la boucle n'a jamais vu</h3>
<p>Tout le reste de ce volet est mesuré sur <span class="mono">train</span>, c'est-à-dire
sur le jeu qui a servi à <em>optimiser</em> la lignée. Un composite d'entraînement ne
distingue pas un prompt qui a compris la population d'un prompt qui a mémorisé ses
{gen.get("train_agents", "?")} personas. Le tableau ci-dessous ajoute le chiffre qui le
distingue&nbsp;: la même lignée, sous le même régime
(<span class="mono">{escape(gen.get("regime") or "—")}</span>), évaluée sur le jeu
<span class="mono">{ds}</span>.</p>
{version_warning}
{tiles([
    ("Graine", _num(seed.get("held")), f"composite, jeu {gen['dataset']}"),
    ("Meilleur", _num(leaf.get("held")), f"composite, jeu {gen['dataset']}"),
    ("Écart corrigé", "—" if leaf.get("gap_controlled") is None
     else f'{leaf["gap_controlled"]:+.2f}', "feuille, effectif neutralisé"),
    ("Jeu de retenue", f'{gen.get("n_records", "?")}',
     f'décisions · {gen.get("n_agents", "?")} personnes'),
])}
<div class="scroll"><table><thead><tr><th>Prompt</th>
<th class="num">Train ({gen.get("train_agents", "?")} pers.)</th>
<th class="num">{escape(gen["dataset"].capitalize())} ({gen.get("n_agents", "?")} pers.)</th>
<th class="num">Écart brut</th>
<th class="num">Témoin&nbsp;: train ramené à {gen.get("n_agents", "?")} pers.</th>
<th class="num">Écart corrigé</th></tr></thead><tbody>{rows}</tbody></table></div>
<p style="font-size:12.5px;color:var(--ink3)">Le témoin ne coûte aucun appel LLM&nbsp;:
il rejoue le score des décisions <em>déjà stockées</em> du train sur
{(leaf.get("control") or {}).get("n_draws", "?")} sous-ensembles de
{gen.get("n_agents", "?")} personnes tirés au hasard (par personne, tous ses trajets
conservés&nbsp;; graine fixée, la page se régénère à l'identique). Il dit ce que
vaudrait le score d'entraînement <em>s'il était mesuré sur aussi peu de monde</em> —
c'est à lui, et non à la colonne «&nbsp;Train&nbsp;», que la colonne
«&nbsp;{escape(gen["dataset"].capitalize())}&nbsp;» doit être comparée.</p>
{verdict}
{gain_html}
<p><strong>De quoi parle-t-on quand on dit «&nbsp;généralisation&nbsp;»&nbsp;?</strong>
Le découpage des jeux gelés est
<span class="mono">{escape(gen.get("split_rule") or "règle non déclarée")}</span>,
c'est-à-dire {nature}</p>
{memory_html}
{complete}
<p style="font-size:12.5px;color:var(--ink3)">Ces évals ne rejoignent ni la trajectoire
ni la lignée affichées plus haut&nbsp;: y mêler un autre jeu superposerait deux
populations dans la même courbe. Elles ne rejoignent pas non plus la matrice de synthèse,
qui porte sur le jeu commun&nbsp;— le jeu de retenue est un troisième substrat, et le
faire voisiner d'une colonne du run rejouerait exactement la confusion que l'action A3 a
corrigée.</p>
<div class="formula">python -m scripts.synthesis.heldout_eval&nbsp;&nbsp;&nbsp;# ou : make heldout-eval</div>"""


def section_calibration(payload: dict) -> str:
    arm = payload["arms"]["calibration"]
    if arm.get("status") != "ok":
        return f"""
<section id="volet-calibration"><h2>Volet 2 — Calibration de prompt</h2>
{missing_card("volet calibration", arm.get("reason", ""), arm.get("expected", []),
              arm.get("action", ""))}</section>"""

    store_rows = "".join(
        f'<tr><td><strong>{escape(s["label"])}</strong></td>'
        f'<td class="num">{s["totals"]["nodes"]}</td>'
        f'<td class="num">{s["totals"]["evals"]}</td>'
        f'<td class="num">{s["kept"]}</td>'
        f'<td class="num">{("%.1f – %.1f" % tuple(s["span"])) if s.get("span") else "—"}</td>'
        f'<td class="mono">{escape(", ".join(s["eval_models"]))}</td></tr>'
        for s in arm["stores"])

    ba = ""
    store = next((s for s in arm["stores"] if s.get("best") and s.get("seed")), None)
    if store:
        seed_d, best_d = store["seed"]["dims"], store["best"]["dims"]
        keys = ["composite", "global", "age", "occupation", "genre", "motif", "distance"]
        rows = ""
        for k in keys:
            before, after = seed_d.get(k), best_d.get(k)
            delta = (after - before) if (before is not None and after is not None) else None
            colour = "var(--ok)" if (delta or 0) < 0 else "var(--warn)"
            rows += (f'<tr><td><strong>{escape(k)}</strong></td>'
                     f'<td class="num">{_num(before)}</td><td class="num">{_num(after)}</td>'
                     f'<td class="num" style="color:{colour}">'
                     f'{"—" if delta is None else f"{delta:+.2f}"}</td></tr>')
        ba = (f'<h3>Avant / après, par dimension — sur les personas gelés</h3>'
              f'<p>Graine <code>{escape(store["seed"]["short"])}</code> face au meilleur '
              f'nœud <code>{escape(store["best"]["short"])}</code> de {escape(store["label"])}, '
              f'tous deux recalculés avec la loss courante. Négatif = amélioration. '
              f'Ces chiffres portent sur les <strong>personas gelés</strong> du moteur '
              f'de calibration, pas sur le jeu commun&nbsp;: le tableau qui suit donne '
              f'les deux en regard.</p>'
              f'<div class="scroll"><table><thead><tr><th>Dimension</th>'
              f'<th class="num">Graine</th><th class="num">Meilleur</th>'
              f'<th class="num">Δ</th></tr></thead><tbody>{rows}</tbody></table></div>')

    traj = ""
    for store in arm["stores"]:
        if not store.get("series"):
            continue
        if store.get("subset_of"):
            traj += (f'<p><strong>{escape(store["label"])}</strong> — ses nœuds évalués '
                     f'sont tous présents dans {escape(store["subset_of"])} '
                     f'(rapatriement du cloud vers le local). Courbe non répétée.</p>')
            continue
        traj += (f'<h4>{escape(store["label"])}</h4>'
                 + charts.trajectory(store["series"]))

    lin = ""
    lineage = next((s["lineage"] for s in arm["stores"]
                    if s.get("lineage") and not s.get("subset_of")), None)
    if lineage:
        lin = _lineage_block(lineage)

    nodes_rows = "".join(
        f'<tr><td class="mono">{escape(n["short"])}</td><td>{escape(n["store"])}</td>'
        f'<td>{escape(n["branch"])}</td>'
        f'<td>{escape((n["created_at"] or "")[:10])}</td>'
        f'<td>{escape(n["verdict"])}</td>'
        f'<td class="mono" style="font-size:11px">'
        f'{escape(n.get("regime") or n["eval_model"] or "—")}</td>'
        f'<td class="num">{_num(n["recomputed"])}</td>'
        f'<td class="num">{_num(n["stored"])}</td></tr>'
        for n in arm["nodes_table"])

    variants = arm.get("prompt_variants") or {}
    var_rows = "".join(
        f'<tr><td class="mono">{escape(v["name"])}'
        + (' <span class="badge ok">actif</span>' if v["active"] else "")
        + f'</td><td>{escape(v.get("date") or "—")}</td>'
        f'<td class="num">{v["words"]}</td>'
        f'<td class="num">{_num(v.get("score_final"))}</td>'
        f'<td>{escape(v.get("seed") or "—")}</td></tr>'
        for v in variants.get("variants", []))

    return f"""
<section id="volet-calibration">
<h2>Volet 2 — Calibration de prompt</h2>
<p class="lede">Le moteur de calibration explore des variantes du prompt système par
recuit simulé, en conservant les décisions brutes de chaque évaluation. Cela rend tout
score <strong>recalculable rétroactivement</strong>, sans un seul appel LLM&nbsp;: les
scores affichés ici sont recalculés avec la loss courante et les poids comparables,
pas relus tels quels.</p>

<div class="scroll"><table><thead><tr><th>Store</th><th class="num">Nœuds</th>
<th class="num">Évals</th><th class="num">Retenus</th><th class="num">Composite recalculé</th>
<th>Régimes de mesure</th></tr></thead><tbody>{store_rows}</tbody></table></div>
<p style="font-size:12.5px;color:var(--ink3)">La plage « composite recalculé » porte
sur le <strong>régime de référence</strong> de chaque store — le plus fourni — et non
sur tous ses nœuds&nbsp;: mélanger les régimes dans une même plage n'aurait pas de sens.</p>

<p>Le recalcul change la lecture de l'historique. Les composites <em>stockés</em>
opposent deux régimes sans rapport — environ 176 pour les nœuds évalués par
mistral-small, environ 26 pour le même prompt évalué par gemini-3.1-flash-lite —
parce qu'ils ont été produits par des loss différentes. Ramenés à la loss courante,
les deux régimes se recouvrent largement&nbsp;: l'écart apparent était un changement
d'instrument de mesure, pas un progrès. Ce que le recalcul ne répare
<em>pas</em>&nbsp;: les décisions elles-mêmes, qui dépendent du modèle interrogé et
de la politique de décision.</p>

{_regimes_note(arm)}
{_train_version_note(arm)}

<h3>Trajectoire des prompts retenus</h3>
<p>Seuls les nœuds non rejetés (acceptés, importés, graine) sont tracés, par ordre
chronologique, avec le composite recalculé. Une courbe par <strong>régime de
mesure</strong>&nbsp;: elles ne se lisent pas bout à bout.</p>
{traj or missing_card("trajectoire", "Aucun nœud évalué exploitable dans les stores.", [], "")}

{lin}

{ba}

<h3>Prompts retenus, un par un</h3>
<div class="scroll"><table><thead><tr><th>Nœud</th><th>Store</th><th>Branche</th><th>Date</th>
<th>Verdict</th><th>Régime de mesure</th><th class="num">Composite recalculé</th>
<th class="num">Composite stocké</th></tr></thead><tbody>{nodes_rows}</tbody></table></div>

<h3>Variantes historiques de prompts.yaml</h3>
<p>Les prompts effectivement livrés à la simulation. Leurs scores archivés proviennent
d'une <em>ancienne</em> loss (L1) sur un échantillon de 100 personas&nbsp;: ils ne sont
pas comparables aux composites ci-dessus.</p>
<div class="scroll"><table><thead><tr><th>Variante</th><th>Date</th><th class="num">Mots</th>
<th class="num">Score archivé</th><th>Graine</th></tr></thead><tbody>{var_rows}</tbody></table></div>

{_common_set_block(arm)}

{_generalization_block(arm)}
</section>"""


# PROGEDO model classes (English, fixed by feature_spec.json) → page labels.
# MODE_LABELS does not fit: it indexes the simulation modes.
POLICY_MODE_LABELS = {"bike": "Vélo", "car": "Voiture",
                      "transit": "Transports collectifs", "walk": "Marche"}


def _trained_policy_block(t: dict) -> str:
    """What the model is worth on *its* test set — not on the common set.

    Distinction to hold: these figures come from the survey's test split, sealed at
    household level. They say the model holds, not that it entered the comparison —
    two things the page must above all not confuse.
    """
    rows = "".join(
        f'<tr><td>{escape(POLICY_MODE_LABELS.get(c, c))}</td>'
        f'<td class="num">{100 * o:.1f}%</td><td class="num">{100 * p:.1f}%</td>'
        f'<td class="num">{100 * (p - o):+.1f} pt</td></tr>'
        for c, o, p in zip(t["classes"], t["observed"], t["predicted"]))
    imps = ", ".join(f'{escape(f["name"])} ({f["gain"]:.0%})' for f in t["top_features"])

    return f"""
<h3>Le modèle entraîné</h3>
<p>Booster LightGBM multiclasse, pondéré par les coefficients de redressement de
l'enquête, arrêté par une validation détourée dans le train (jamais sur le test).
Les chiffres ci-dessous portent sur le <strong>split test de l'enquête</strong>,
étanche au ménage&nbsp;— pas sur le jeu commun.</p>
{tiles([
    ("Log-loss", f'{t["log_loss"]:.4f}', "test, pondéré"),
    ("Accuracy", f'{t["accuracy"]:.1%}', "test, pondéré"),
    ("L1 parts modales", f'{100 * t["l1_mass"]:.1f}', "points cumulés, masse de probabilité"),
    ("Arbres retenus", str(t["best_iteration"]), "arrêt anticipé"),
])}
<div class="scroll"><table><thead><tr><th>Mode</th><th class="num">Observé</th>
<th class="num">Prédit</th><th class="num">Écart</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Parts sur les {t["n_test"]} déplacements du split test, pondérées.
«&nbsp;Prédit&nbsp;» est la masse de probabilité, ce que le pipeline consomme
réellement&nbsp;; en mode élu (argmax) l'écart cumulé monte à
{100 * t["l1_argmax"]:.1f} points, le durcissement exagérant les parts dominantes.
Variables les plus contributrices (gain)&nbsp;: {imps}.</p>"""


def _renormalization_block(payload: dict, preds: dict) -> str:
    """Strand 3 on the common set: what the OTP renormalisation changes.

    The question this block answers is not "does the model predict well" but
    "what did restricting it to the offer correct". The before and after shares are
    therefore given side by side: without the "before", the correction is an assertion.
    """
    primary = payload["score_def"]["primary"]
    variants = preds.get("variants") or {}
    summary = preds.get("summary") or {}
    expected = variants.get("attendu") or {}
    elected = variants.get("elu") or {}
    raw = variants.get("brut") or {}

    before = (preds.get("meta") or {}).get("shares_before") or {}
    after = (preds.get("meta") or {}).get("shares_after") or {}
    gview = expected.get("global") or {}
    target = gview.get("target") or {}
    rows = "".join(
        f'<tr><td>{escape(MODE_LABELS[m])}</td>'
        f'<td class="num">{before.get(m, 0.0):.1f}</td>'
        f'<td class="num">{after.get(m, 0.0):.1f}</td>'
        f'<td class="num">{after.get(m, 0.0) - before.get(m, 0.0):+.1f}</td>'
        f'<td class="num">{target.get(m, 0.0):.1f}</td></tr>' for m in MODES)

    sizes = summary.get("offer_sizes") or {}
    sizes_txt = ", ".join(f'{sizes[k]} à {k} mode' + ("s" if int(k) > 1 else "")
                          for k in sorted(sizes, key=lambda x: int(x)))
    mass = summary.get("offered_mass_mean")
    excluded = summary.get("excluded_pct")
    statuses = summary.get("status_counts") or {}
    non_ok = {k: v for k, v in statuses.items() if k != "ok"}
    excluded_txt = (
        "Aucune décision du périmètre n'a dû être écartée&nbsp;: les deux extrémités "
        "de chaque trajet tombent dans la couche de zones fines, et chaque jeu de "
        "choix contient au moins un mode que la politique sait prédire."
        if not non_ok else
        "Décisions écartées du score, par cause&nbsp;: "
        + escape(", ".join(f"{k} — {v}" for k, v in sorted(non_ok.items()))) + ".")

    return f"""
<h3>Sur le jeu commun, renormalisé sur l'offre OTP</h3>
<p>La politique prédit sur 4 classes sans savoir ce qui était offert&nbsp;; la
simulation, elle, ne choisit que parmi les itinéraires qu'OTP a proposés. Chaque
prédiction est donc <strong>restreinte aux modes réellement proposés</strong> pour ce
trajet-là, puis renormalisée à 100 % (hypothèse IIA). Sans cette correction, on
reprocherait au LLM de n'avoir pas choisi un mode qu'on ne lui a jamais offert.</p>
{tiles([
    ("Composite comparable",
     _num((expected.get("scores") or {}).get(primary, {}).get("composite")),
     "masse de probabilité"),
    ("Composite (mode élu)",
     _num((elected.get("scores") or {}).get(primary, {}).get("composite")),
     "argmax"),
    ("Avant renormalisation",
     _num((raw.get("scores") or {}).get(primary, {}).get("composite")),
     "sans l'offre OTP"),
    ("Décisions scorées",
     f'{summary.get("n_scored", 0):,}'.replace(",", " "),
     f'sur {summary.get("n_moves", 0):,} — {100 - (excluded or 0.0):.1f} %'.replace(",", " ")),
])}
<div class="scroll"><table><thead><tr><th>Mode</th>
<th class="num">Avant renorm.</th><th class="num">Après renorm.</th>
<th class="num">Effet</th><th class="num">EMC²</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Parts en % de masse de probabilité, sur les décisions scorées. La renormalisation
n'est pas cosmétique&nbsp;: la masse tombant sur des modes offerts vaut
{("%.1f" % (100 * mass)) if mass is not None else "—"} % en moyenne, et elle déplace le
mode le plus probable sur <strong>{summary.get("n_argmax_shifted", 0)}</strong>
décision(s). Taille des jeux de choix&nbsp;: {escape(sizes_txt)} — les
{summary.get("n_single_offer", 0)} trajets à mode unique reçoivent une probabilité de
1,0, ce qui n'est pas une prédiction mais le constat qu'il n'y avait pas de choix. Ils
sont conservés parce que le volet 1 les conserve aussi.</p>
<p>{excluded_txt}</p>
{_renormalisation_warning(preds.get("renormalisation_bias") or {})}
<h3>Détail par sous-catégorie</h3>
<p>Aussi disponible en page dédiée, sans le reste de la synthèse&nbsp;:
<a href="detail_progedo.html">détail par sous-catégorie — arbre PROGEDO</a>.</p>
{_dimension_blocks(preds.get("details") or {}, "mod")}"""


def _renormalisation_warning(bias: dict) -> str:
    """Reading warning on the `attendu` composite score, computed and not hard-coded.

    Renormalisation over the OTP offer is not neutral: it takes mass away from the
    modes the model overpredicts and gives it to those OTP almost always offers. On
    some modes it moves closer to the target, on others further away — and for the
    latter, `attendu` **penalises** any correction that increases the mode, even a right one.
    Measured on 2026-08-27: the composite score went from 6.015 to 6.208 while the gap of the
    15-19 year-olds, the largest on the page, fell from 63.8 to 48.7 of `l1`.

    The text names the modes concerned instead of assuming them: the bias depends on
    the run's offer, and a fixed sentence would silently become false on the next run.
    """
    modes = (bias or {}).get("modes") or {}
    worsened, improved = [], []
    for mode, v in modes.items():
        raw_gap, ren_gap = v.get("raw_gap_pt"), v.get("renormalised_gap_pt")
        if raw_gap is None or ren_gap is None:
            continue
        (worsened if abs(ren_gap) > abs(raw_gap) + 1.0 else
         improved if abs(ren_gap) + 1.0 < abs(raw_gap) else []).append((mode, raw_gap, ren_gap))
    if not worsened:
        return ""
    def _fmt(items):
        return ", ".join(
            f"<strong>{escape(MODE_LABELS.get(m, m))}</strong> ({r:+.1f} → {n:+.1f})"
            for m, r, n in items)
    good = (f" À l'inverse elle en rapproche {_fmt(improved)}, "
            f"donc elle n'est pas à retirer&nbsp;: elle corrige la surprédiction de la "
            f"voiture que le modèle produit en brut." if improved else "")
    return f"""
<div class="note"><div class="t">Comment lire le composite « masse de probabilité »</div>
<p>La renormalisation <strong>éloigne</strong> de la cible EMC² la part de
{_fmt(worsened)} — écart avant puis après renormalisation.{good}</p>
<p>Conséquence sur ces modes-là&nbsp;: le composite « masse de probabilité »
<strong>pénalise toute correction qui augmente le mode, même juste</strong>, puisqu'il le
lit au-dessus de sa cible alors que le modèle l'y met presque. Un lot qui corrige la
sous-représentation d'un de ces modes se juge sur le <strong>détail par
sous-catégorie</strong> et sur les composites « mode élu » et « avant renormalisation »,
qui n'ont pas ce biais.</p></div>"""


def section_model(payload: dict) -> str:
    arm = payload["arms"]["model"]
    spec = arm.get("feature_spec") or {}
    trained = arm.get("trained")
    preds = arm.get("predictions") or {}
    trained_html = _trained_policy_block(trained) if trained else ""
    if preds.get("available"):
        missing_html = _renormalization_block(payload, preds)
    elif trained:
        missing_html = missing_card(
            "prédictions sur le jeu commun",
            "Le modèle est entraîné et sérialisé, mais il n'a encore été appliqué à "
            "aucune décision du jeu commun : il n'existe pas de fichier de "
            "probabilités prédites, donc aucun score comparable aux deux autres "
            "volets. Les chiffres ci-dessus portent sur le split test de l'enquête, "
            "pas sur le run.",
            arm.get("expected", []),
            "Action A8 — prédire sur le jeu commun et renormaliser sur l'offre OTP")
    else:
        missing_html = missing_card(
            "modèle entraîné",
            "Le jeu de données et la spécification des variables existent, mais aucun "
            "modèle n'a été entraîné ni sérialisé, et aucun évaluateur d'exécution "
            "n'existe. Le volet ne peut produire aucun score aujourd'hui.",
            arm.get("expected", []),
            "Actions A6 à A8 — entraîner, sérialiser, puis prédire sur le jeu commun")

    feats = ""
    if spec.get("features"):
        rows = "".join(
            f'<tr><td class="mono">{escape(f["name"])}</td><td>{escape(f["source"])}</td>'
            f'<td>{escape(f["status"])}</td></tr>' for f in spec["features"])
        feats = (f'<h3>Variables attendues par le modèle</h3>'
                 f'<p>Disponibilité de chaque variable sur le jeu commun — c\'est ce '
                 f'qui conditionne la faisabilité du volet.</p>'
                 f'<div class="scroll"><table><thead><tr><th>Variable</th>'
                 f'<th>Source dans le jeu commun</th><th>État</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table></div>')

    return f"""
<section id="volet-modele">
<h2>Volet 3 — Modèle de régression PROGEDO</h2>
<p class="lede">Une politique statistique entraînée directement sur les micro-données
de l'enquête (54 585 déplacements, dont 27 886 retenus). Elle sert de
<strong>référence haute</strong> plus que de concurrent loyal&nbsp;: entraînée sur la
même enquête qui sert de cible, elle est par construction proche de l'oracle sur les
parts modales. Son intérêt est de borner ce qu'un modèle purement statistique atteint,
pour situer les deux autres volets.</p>
{trained_html}

{missing_html}

{feats}
</section>"""


def section_synthesis(payload: dict) -> str:
    syn = payload["synthesis"]
    # Substrate of each column, stated under the matrix: two columns can
    # carry the same strand name and not describe the same population.
    basis = "".join(
        f'<li><strong>{escape(a["label"])}</strong> — {escape(a["basis"])}'
        + (f' <span class="mono" style="font-size:11px">{escape(a["note"])}</span>'
           if a.get("note") else "")
        + "</li>"
        for a in syn["arms"] if a.get("basis"))

    # Three states, and the text must say EXACTLY what remains. The two strands to
    # bring onto the common set are independent: A3 (calibration) and A8 (model).
    model_ok = syn.get("model_available")
    # The matrix is about the common set. The held-out score (action A4) is one
    # more substrate and does not appear in it: the reader is sent where it is read,
    # rather than sticking a 66-person column next to 881-person columns.
    gen_pointer = ""
    if syn.get("generalization_available"):
        gen_pointer = (
            f' Enfin, aucune de ces colonnes ne dit ce que vaut la calibration hors du '
            f'jeu qui a servi à l\'optimiser : ce chiffre existe désormais, sur le jeu '
            f'« {syn.get("generalization_dataset", "test")} », et se lit dans le bloc '
            f'« Généralisation » du volet 2 — sur un troisième substrat, donc hors de '
            f'cette matrice.')
    if syn.get("commensurable") and model_ok:
        card = missing_card(
            "type de logement, effectifs inégaux, et lecture des colonnes",
            "Les trois volets sont scorés sur le même jeu commun : les personas du run "
            "épinglé — le volet 2 sur un échantillon gelé de ce run, le volet 3 sur les "
            "décisions dont la géographie est reconstructible. Les colonnes se "
            "comparent enfin. Trois réserves subsistent, et la deuxième est la plus "
            "facile à lire de travers. (1) La dimension « type de logement » est "
            "instrumentée — la population porte le trait, le journal écrit la colonne — "
            "mais le run épinglé est antérieur, donc l'axe reste vide ici jusqu'au "
            "prochain run. (2) Les colonnes ne portent pas sur le même NOMBRE de "
            "décisions : le volet 2 est mesuré sur ~500 décisions et 81 personnes, les "
            "volets 1 et 3 sur 5 945. Les divergences par strate étant biaisées vers le "
            "haut à petit effectif, comparer directement surestime l'écart du volet 2 ; "
            "c'est pourquoi la colonne « Sim. (éch. V2) » est là — même run, mêmes "
            "personnes que la calibration, et c'est à elle que les deux colonnes de "
            "calibration doivent être comparées. (3) Le volet 3 n'est pas un concurrent "
            "loyal — voir l'avertissement ci-dessus." + gen_pointer,
            [], "Action A2 — le trait est produit, l'axe attend un nouveau run épinglé")
    elif model_ok:
        card = missing_card(
            "comparaison sur base strictement identique",
            "Les volets simulation et modèle sont scorés sur le même jeu commun : les "
            "personas du run épinglé. Ces colonnes-là se comparent. Le volet "
            "calibration, lui, reste scoré sur ses personas gelés — un sous-ensemble "
            "d'un run antérieur : sa ré-évaluation sur le jeu commun est outillée mais "
            "la mesure n'a pas encore été payée.",
            [], "Action A3 — rejouer graine et meilleur prompt sur le jeu commun")
    elif syn.get("commensurable"):
        # Strand 2 has reached the common set (action A3). The admission does not
        # disappear for all that: it now only concerns strand 3.
        card = missing_card(
            "comparaison sur base strictement identique",
            "Les volets simulation et calibration sont désormais scorés sur le même "
            "jeu commun : les personas du run épinglé, le volet 2 sur un échantillon "
            "gelé de ce run. Ces colonnes-là se comparent. Le volet modèle, lui, est "
            "entraîné mais n'a encore été appliqué à aucune décision du run : sa "
            "colonne reste vide, et rien dans cette page ne le situe encore face aux "
            "deux autres.",
            [], "Action A8 — appliquer le modèle au jeu commun")
    else:
        card = missing_card(
            "comparaison sur base strictement identique",
            "Seul le volet simulation est aujourd'hui scoré sur le jeu commun. Le volet "
            "calibration est scoré sur ses personas gelés — un sous-ensemble d'un run "
            "antérieur — et le volet modèle, désormais entraîné, n'a encore été appliqué "
            "à aucune décision du run. Les colonnes ne sont donc pas encore "
            "commensurables et la lecture doit rester qualitative.",
            [], "Actions A3 et A8 amènent les trois volets sur le même jeu")

    # The warning on strand 3 is placed ABOVE the matrix, not three
    # sections further down: the reader must know the « Modèle » column is an
    # upper reference at the very moment they see its figure.
    caveat = ""
    if model_ok:
        caveat = """
<div class="missing" style="border-color:var(--line)">
<div class="t">Comment lire la colonne « Modèle »</div>
<p>Le volet 3 est une <strong>référence haute</strong>, pas un concurrent loyal : il est
entraîné sur l'enquête EMC² qui sert ici de cible. Qu'il devance les deux autres colonnes
est attendu par construction, et ne dit rien de la qualité relative du LLM — cela borne
ce qu'un modèle purement statistique atteint sur ce jeu. Son propre entraînement porte
d'ailleurs sur une sous-population (le périmètre d'enquête, plus dense et plus marcheur
que l'agglomération), donc même cette borne est optimiste.</p></div>"""

    return f"""
<section id="synthese">
<h2>Synthèse comparative</h2>
<p class="lede">Score par dimension et par volet, sur le composite comparable. Plus la
case est foncée, plus l'écart à l'enquête est grand. Les cases « n. d. » attendent les
données décrites dans la liste d'actions.</p>
{caveat}
{charts.heatmap(syn["dims"], syn["arms"])}
<p>Sur quoi porte chaque colonne&nbsp;— deux volets peuvent porter le même nom et pas
la même population&nbsp;:</p>
<ul>{basis}</ul>
{card}
</section>"""


def section_provenance(payload: dict) -> str:
    rows = "".join(
        f'<tr><td class="mono">{escape(s["path"])}</td><td>{escape(s["role"])}</td>'
        + (f'<td><span class="badge ok">présent</span></td>'
           if s["exists"] else '<td><span class="badge warn">absent</span></td>')
        + f'<td>{escape((s.get("mtime") or "—")[:10])}</td>'
        f'<td class="mono" style="font-size:11px">{escape((s.get("sha256") or "—")[:12])}</td>'
        f'</tr>' for s in payload["sources"])
    # Done actions stay listed: the code and the tickets refer to the
    # identifiers, and the reader sees what has already been settled.
    #
    # Three states, not two. « Partiellement faite » exists because an action
    # whose tooling is delivered but whose measurement produced nothing is not done:
    # striking it through would claim a non-existent result. It therefore stays counted as
    # pending, with its cost, and states explicitly what is missing.
    def action_row(a: dict) -> str:
        done, progress = a.get("done"), a.get("progress")
        if done:
            tr = '<tr style="opacity:.62">'
            title = (f'<span style="text-decoration:line-through">{escape(a["title"])}'
                     '</span> <span class="badge ok">faite</span>')
            detail, cost = escape(done), "—"
        elif progress:
            tr = "<tr>"
            title = (f'<strong>{escape(a["title"])}</strong> '
                     '<span class="badge warn">partiellement faite</span>')
            detail = (f'<strong>Acquis&nbsp;:</strong> {escape(progress["acquis"])}'
                      f'<br><strong>Reste&nbsp;:</strong> {escape(progress["reste"])}')
            cost = a["cost"]
        else:
            tr = "<tr>"
            title = f'<strong>{escape(a["title"])}</strong>'
            detail, cost = escape(a["detail"]), a["cost"]
        return (f'{tr}<td class="mono">{escape(a["id"])}</td><td>{title}'
                f'<br><span style="color:var(--ink3);font-size:12.5px">'
                f'{detail}</span></td>'
                f'<td>{escape(cost)}</td><td>{escape(a["unlocks"])}</td></tr>')

    actions = "".join(action_row(a) for a in payload["actions"])
    remaining = sum(1 for a in payload["actions"] if not a.get("done"))
    partial = sum(1 for a in payload["actions"] if a.get("progress") and not a.get("done"))
    partial_badge = (f' <span class="badge warn">dont {partial} entamée'
                     f'{"s" if partial > 1 else ""}</span>' if partial else "")
    # The title announces a to-do list; when everything left is « entamé »,
    # there is no more work to start but an external condition to wait for.
    # Saying so avoids suggesting an open work item that no longer exists.
    if remaining == 0:
        lede = ('<p class="lede">Toutes les actions listées sont faites. La liste reste '
                'affichée&nbsp;: le code et les tickets y renvoient par numéro, et les '
                'identifiants ne sont jamais recyclés.</p>')
    elif remaining == partial:
        lede = (f'<p class="lede">Plus aucune action n\'attend qu\'on l\'engage&nbsp;: '
                f'{"la seule restante est entamée" if remaining == 1 else f"les {remaining} restantes sont entamées"}, '
                f'et {"elle attend" if remaining == 1 else "elles attendent"} une '
                f'condition extérieure — un nouveau run épinglé — et non un travail à '
                f'faire. Les actions faites restent listées&nbsp;: le code et les '
                f'tickets y renvoient par numéro.</p>')
    else:
        lede = ('<p class="lede">Les actions faites restent listées&nbsp;: le code et '
                'les tickets y renvoient par numéro, et les identifiants ne sont jamais '
                'recyclés.</p>')
    return f"""
<section id="provenance">
<h2>Provenance et régénération</h2>
<p class="lede">Chaque chiffre de cette page vient d'un fichier listé ci-dessous. Le
manifeste <code>scripts/synthesis/sources.yaml</code> dit où les chercher&nbsp;;
changer un chemin suffit à rejouer la synthèse sur un autre run ou un autre store. Le
run du jeu commun y est <strong>épinglé par chemin d'archive</strong>, jamais par le
symlink <code>experiments/current</code>&nbsp;: deux régénérations décrivent le même
run, et les empreintes ci-dessous permettent de le vérifier.</p>
<div class="formula">make synthesis&nbsp;&nbsp;&nbsp;# ou : make synthesis RUN=experiments/archive/&lt;run&gt;</div>
<div class="scroll"><table><thead><tr><th>Fichier</th><th>Rôle</th><th>État</th>
<th>Modifié</th><th>Empreinte</th></tr></thead><tbody>{rows}</tbody></table></div>

<h3>Ce qu'il reste à faire pour remplir la page <span class="badge">{remaining} en
attente</span>{partial_badge}</h3>
{lede}
<div class="scroll"><table><thead><tr><th>#</th><th>Action</th><th>Coût</th>
<th>Débloque</th></tr></thead><tbody>{actions}</tbody></table></div>
</section>"""


def render(payload: dict) -> str:
    nav_items = [
        ("grp", "Cadre"),
        ("definition", "Définition du score"),
        ("jeu-commun", "Jeu d'évaluation commun"),
        ("grp", "Volets"),
        ("volet-simulation", "1 · Simulation"),
        ("volet-calibration", "2 · Calibration de prompt"),
        ("volet-modele", "3 · Modèle PROGEDO"),
        ("grp", "Conclusion"),
        ("synthese", "Synthèse comparative"),
        ("provenance", "Provenance et régénération"),
        ("grp", "Pages dédiées"),
        ("detail_simulation.html", "Détail — Simulation"),
        ("detail_progedo.html", "Détail — PROGEDO"),
    ]
    nav = ""
    for key, label in nav_items:
        if key == "grp":
            nav += f'<div class="grp">{escape(label)}</div>'
        elif key.endswith(".html"):
            nav += f'<a href="{key}">{escape(label)}</a>'
        else:
            nav += f'<a href="#{key}">{escape(label)}</a>'

    body = (section_definition(payload) + section_common_set(payload)
            + section_simulation(payload) + section_calibration(payload)
            + section_model(payload) + section_synthesis(payload)
            + section_provenance(payload))

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthèse des scores — parts modales vs EMC² 2023</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<nav><h1>Synthèse des scores</h1>
<div class="sub">Parts modales simulées face à l'enquête EMC² 2023</div>
{nav}</nav>
<main>
<p style="color:var(--ink3);font-size:12.5px;margin-bottom:24px">
Généré le {escape(payload["generated_at"])} · {escape(payload["engine_note"])}</p>
{body}
<footer>Régénérer&nbsp;: <code>make synthesis</code> · Sources déclarées dans
<code>scripts/synthesis/sources.yaml</code> · Loss importée de
<code>prompt_calibration/calibration/metrics.py</code></footer>
</main></div></body></html>"""


# ── Dedicated « Détail par sous-catégorie » pages ─────────────────────────────
# The synthesis page stays the complete reference; these two pages extract from it
# only the « Détail par sous-catégorie » sub-chapter, strand by strand, so it
# can be read (and printed) without scrolling through the rest. They are
# regenerated by the same `make synthesis`: no figure is copied into them by
# hand, everything comes from the same payload.

DETAIL_PAGES = {
    "simulation": {
        "file": "detail_simulation.html",
        "title": "Détail par sous-catégorie — Simulation (LLM + tirage)",
        "nav_title": "Détail — Simulation",
        "heading": "Volet 1 — Simulation (LLM + tirage)",
        "back_label": "Volet 1 dans la synthèse",
        "back_anchor": "volet-simulation",
        "lede": (
            "Le mode est choisi comme dans le run&nbsp;: le LLM attribue une "
            "probabilité à chaque itinéraire proposé par OTP, puis la simulation tire "
            "au sort dans cette distribution. Les profils ci-dessous portent sur la "
            "<strong>masse de probabilité</strong> — la lecture indépendante du tirage, "
            "celle que la calibration optimise."),
        "cells": "parts modales de la population simulée",
    },
    "model": {
        "file": "detail_progedo.html",
        "title": "Détail par sous-catégorie — Modèle de régression PROGEDO",
        "nav_title": "Détail — PROGEDO",
        "heading": "Volet 3 — Modèle de régression PROGEDO",
        "back_label": "Volet 3 dans la synthèse",
        "back_anchor": "volet-modele",
        "lede": (
            "L'arbre de régression PROGEDO est entraîné sur les micro-données de "
            "l'enquête, puis appliqué au jeu commun&nbsp;: chaque prédiction est "
            "restreinte aux modes réellement proposés par OTP pour ce trajet-là, puis "
            "renormalisée à 100&nbsp;% (hypothèse IIA). Les profils ci-dessous portent "
            "sur cette masse de probabilité renormalisée, la seule comparable au "
            "volet 1."),
        "cells": "parts modales prédites par la politique",
    },
}


def _detail_details(payload: dict, arm_key: str) -> tuple[dict, str]:
    """The strand's details dictionary, and the reason if it is not available."""
    arm = (payload.get("arms") or {}).get(arm_key) or {}
    if arm_key == "simulation":
        if arm.get("status") != "ok":
            return {}, arm.get("reason", "Volet simulation indisponible.")
        return arm.get("details") or {}, ""
    preds = arm.get("predictions") or {}
    if not preds.get("available"):
        return {}, ("Le modèle n'a pas encore été appliqué au jeu commun : aucune "
                    "distribution prédite à ventiler par sous-catégorie.")
    return preds.get("details") or {}, ""


def render_detail(payload: dict, arm_key: str) -> str:
    """A standalone page carrying only the « Détail par sous-catégorie » sub-chapter.

    No side table of contents here, unlike the synthesis page: the page
    carries only one chapter, seven cells in a row, and a navigation column
    would only eat into the width of the charts.
    """
    spec = DETAIL_PAGES[arm_key]
    prefix = "sim" if arm_key == "simulation" else "mod"
    details, reason = _detail_details(payload, arm_key)
    other = DETAIL_PAGES["model" if arm_key == "simulation" else "simulation"]

    if reason:
        body = missing_card(spec["heading"], reason, [],
                            "Voir la page de synthèse pour la liste d'actions")
    else:
        body = _dimension_blocks(details, prefix)

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(spec["title"])}</title>
<style>{CSS}
.solo{{max-width:1040px;margin:0 auto;padding:32px 40px 96px}}
.solo h1{{font-size:21px;font-weight:500;margin:0 0 4px;letter-spacing:-.015em}}
.solo .sub{{font-size:13px;color:var(--ink3);margin-bottom:18px}}
.solo .links{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin-bottom:22px}}
@media(max-width:880px){{.solo{{padding:24px 20px 64px}}}}
</style></head>
<body><main class="solo">
<h1>{escape(spec["heading"])}</h1>
<div class="sub">Détail par sous-catégorie — parts modales face à l'enquête
EMC² 2023</div>
<div class="links"><a href="index.html">← Synthèse complète</a>
<a href="index.html#{spec["back_anchor"]}">{escape(spec["back_label"])}</a>
<a href="{other["file"]}">{escape(other["nav_title"])}</a></div>
<p style="color:var(--ink3);font-size:12.5px;margin-bottom:24px">
Généré le {escape(payload.get("generated_at", ""))} ·
{escape(payload.get("engine_note", ""))}</p>
<section id="detail">
<p class="lede">{spec["lede"]}</p>
<p>Une cellule par sous-catégorie&nbsp;: barres = {spec["cells"]}, repère de
référence = EMC² 2023. Les dimensions marquées
<span class="badge ok">dans le composite</span> entrent dans le score&nbsp;; celles
marquées <span class="badge">hors composite</span> sont rapportées pour lecture
seulement. <strong>Plus l'écart au repère est faible, meilleur c'est.</strong>
Le contexte de ce volet — parts globales, pires croisements, périmètre — reste dans
<a href="index.html#{spec["back_anchor"]}">{escape(spec["back_label"])}</a>.</p>
{body}
</section>
<footer>Régénérer&nbsp;: <code>make synthesis</code> · Page extraite de
<code>docs/synthesis/index.html</code>, qui reste la référence complète</footer>
</main></body></html>"""
