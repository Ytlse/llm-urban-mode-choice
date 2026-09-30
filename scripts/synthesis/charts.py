"""Inline SVG charts, with no dependency and no script.

The page must open from a ``file://`` and read offline: no CDN, no chart
library. Two forms only, chosen for what the data have to say:

* **bullet** — an observed share against its EMC² target. The target is a marker,
  not a second series: it gets a tick, not one more colour.
* **ordinal profile** — for age and distance, where the order of the categories
  carries meaning: a solid line (observed) against a dotted line (EMC²),
  in small multiples, one per mode.

The mode palette is the project's (`.claude/CLAUDE.md`): car red, bike purple,
public transport green, walking cyan. CVD-validated (minimum adjacent ΔE 18.2);
since cyan and green fall below 3:1 contrast, every value is written out next
to its bar.
"""
from __future__ import annotations

from html import escape
from typing import Optional, Sequence

from .frames import MODE_COLORS, MODE_LABELS, MODES, pretty_cat


def _fmt(value: Optional[float], unit: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.1f}{unit}"


def _tick_label(cat: str) -> str:
    """Compact axis label: the bounds read without the repeated unit."""
    if cat == "plus_50km":
        return "50+"
    if cat == "75-130":
        return "75+"
    return cat.replace("km", "")


def legend(items: Sequence[tuple[str, str]], extra: str = "") -> str:
    """Swatch + label, in text ink (never the series colour)."""
    chips = "".join(
        f'<span class="cx-chip"><i style="background:{color}"></i>{escape(label)}</span>'
        for label, color in items)
    if extra:
        chips += f'<span class="cx-chip cx-chip-note">{extra}</span>'
    return f'<div class="cx-legend">{chips}</div>'


def bullet_rows(rows: Sequence[dict], *, max_pct: float = 70.0,
                label_width: int = 138, row_height: int = 26) -> str:
    """Horizontal bars, observed vs target.

    ``rows``: ``{label, color, actual, target}``. The bar carries the observed
    value, the vertical tick the EMC² target, the right-hand label the signed gap.
    """
    if not rows:
        return ""
    pad_r, height = 96, row_height * len(rows) + 26
    track_w = 520 - label_width - pad_r
    scale = track_w / max_pct

    # The SVG does not exceed its natural width: beyond it, all the text would
    # grow with the drawing and the labels would outsize the page body text.
    parts = [f'<svg viewBox="0 0 520 {height}" width="100%" style="max-width:520px" '
             f'role="img" preserveAspectRatio="xMinYMin meet">']
    for i, row in enumerate(rows):
        y = i * row_height + 6
        actual = float(row.get("actual") or 0.0)
        target = row.get("target")
        bar_w = max(0.0, min(actual, max_pct)) * scale
        parts.append(
            f'<text class="cx-label" x="0" y="{y + 13}">{escape(str(row["label"]))}</text>')
        parts.append(f'<rect class="cx-track" x="{label_width}" y="{y + 3}" '
                     f'width="{track_w}" height="14" rx="4"/>')
        parts.append(f'<rect x="{label_width}" y="{y + 3}" width="{bar_w:.1f}" '
                     f'height="14" rx="4" fill="{row["color"]}"/>')
        if target is not None:
            tx = label_width + min(float(target), max_pct) * scale
            parts.append(f'<line class="cx-target" x1="{tx:.1f}" y1="{y}" '
                         f'x2="{tx:.1f}" y2="{y + 20}"/>')
        value_x = label_width + track_w + 8
        parts.append(f'<text class="cx-value" x="{value_x}" y="{y + 13}">'
                     f'{_fmt(actual)}</text>')
        if target is not None:
            delta = actual - float(target)
            # The alert is about the size of the gap, not its sign: underestimating
            # walking by 19 points is as serious as overestimating bike by 15.
            cls = "cx-delta-up" if abs(delta) >= 5.0 else "cx-delta-down"
            sign = "+" if delta > 0 else "−"
            parts.append(f'<text class="cx-delta {cls}" x="{value_x + 46}" y="{y + 13}">'
                         f'{sign}{abs(delta):.1f}</text>')
    axis_y = height - 14
    parts.append(f'<line class="cx-axis" x1="{label_width}" y1="{axis_y - 4}" '
                 f'x2="{label_width + track_w}" y2="{axis_y - 4}"/>')
    for tick in (0, max_pct / 2, max_pct):
        tx = label_width + tick * scale
        parts.append(f'<text class="cx-axis-label" x="{tx:.1f}" y="{axis_y + 7}" '
                     f'text-anchor="middle">{tick:.0f}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def global_bullet(view: dict) -> str:
    rows = [{"label": MODE_LABELS[m], "color": MODE_COLORS[m],
             "actual": view["actual"].get(m), "target": view["target"].get(m)}
            for m in MODES]
    return bullet_rows(rows, max_pct=70.0)


def ordinal_profiles(detail: Sequence[dict], *, order: Sequence[str]) -> str:
    """Small multiples: one mini-chart per mode, observed solid vs EMC² dotted."""
    cats = [c for c in order if any(d["cat"] == c for d in detail)]
    if not cats:
        return ""
    by_cat = {d["cat"]: d for d in detail}
    cell_w, cell_h, gap = 300, 124, 18
    cols = 2
    rows_n = (len(MODES) + cols - 1) // cols
    total_w = cols * cell_w + (cols - 1) * gap
    total_h = rows_n * cell_h + (rows_n - 1) * gap

    peak = 1.0
    for cat in cats:
        entry = by_cat[cat]
        for m in MODES:
            peak = max(peak, entry["actual"].get(m, 0.0) or 0.0,
                       entry["target"].get(m, 0.0) or 0.0)
    top = min(100.0, (int(peak / 10) + 1) * 10.0)

    parts = [f'<svg viewBox="0 0 {total_w} {total_h}" width="100%" '
             f'style="max-width:{total_w}px" role="img" '
             f'preserveAspectRatio="xMinYMin meet">']
    plot_l, plot_t, plot_b = 32, 20, 26
    for idx, mode in enumerate(MODES):
        ox = (idx % cols) * (cell_w + gap)
        oy = (idx // cols) * (cell_h + gap)
        plot_w = cell_w - plot_l - 22
        plot_h = cell_h - plot_t - plot_b
        step = plot_w / max(1, len(cats) - 1)

        parts.append(f'<text class="cx-sub" x="{ox}" y="{oy + 11}">'
                     f'{escape(MODE_LABELS[mode])}</text>')
        for frac in (0.0, 0.5, 1.0):
            gy = oy + plot_t + plot_h * (1 - frac)
            parts.append(f'<line class="cx-grid" x1="{ox + plot_l}" y1="{gy:.1f}" '
                         f'x2="{ox + plot_l + plot_w}" y2="{gy:.1f}"/>')
            parts.append(f'<text class="cx-axis-label" x="{ox + plot_l - 6}" '
                         f'y="{gy + 3:.1f}" text-anchor="end">{top * frac:.0f}</text>')

        def segments(kind: str) -> list[str]:
            # The observed series is cut at the uncovered bins (n < 5): a
            # point resting on a handful of decisions must not pull the
            # curve. The survey reference, for its part, is drawn in full.
            segs: list[list[str]] = []
            cur: list[str] = []
            for i, cat in enumerate(cats):
                entry = by_cat[cat]
                value = entry[kind].get(mode)
                if value is None or (kind == "actual" and not entry.get("covered")):
                    if len(cur) >= 2:
                        segs.append(cur)
                    cur = []
                    continue
                px = ox + plot_l + i * step
                py = oy + plot_t + plot_h * (1 - min(value, top) / top)
                cur.append(f"{px:.1f},{py:.1f}")
            if len(cur) >= 2:
                segs.append(cur)
            return [" ".join(s) for s in segs]

        for pts in segments("target"):
            parts.append(f'<polyline class="cx-ref-line" points="{pts}"/>')
        for pts in segments("actual"):
            parts.append(f'<polyline points="{pts}" fill="none" '
                         f'stroke="{MODE_COLORS[mode]}" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, cat in enumerate(cats):
            entry = by_cat[cat]
            value = entry["actual"].get(mode)
            if value is None or not entry.get("covered"):
                continue
            px = ox + plot_l + i * step
            py = oy + plot_t + plot_h * (1 - min(value, top) / top)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" '
                         f'fill="{MODE_COLORS[mode]}"/>')
        # One label every n, chosen so that two neighbouring labels never touch:
        # the age axis has 15 bins for 250 usable px.
        every = 3 if len(cats) > 10 else (2 if len(cats) > 7 else 1)
        for i, cat in enumerate(cats):
            if i % every:
                continue
            px = ox + plot_l + i * step
            low = "" if by_cat[cat].get("covered") else " cx-tick-low"
            parts.append(f'<text class="cx-tick{low}" x="{px:.1f}" '
                         f'y="{oy + plot_t + plot_h + 12}" text-anchor="middle">'
                         f'{escape(_tick_label(cat))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def nominal_groups(detail: Sequence[dict]) -> str:
    """One cluster of 4 bullet bars per nominal category."""
    blocks = []
    for entry in detail:
        n = entry.get("n") or 0
        badge = (f'<span class="cx-n">n={n}</span>' if entry.get("covered")
                 else f'<span class="cx-n cx-n-low">n={n} · sous le seuil</span>')
        if not entry.get("actual"):
            body = ('<p class="cx-empty">Aucune décision observée dans cette '
                    'catégorie sur le jeu commun.</p>')
        else:
            rows = [{"label": MODE_LABELS[m], "color": MODE_COLORS[m],
                     "actual": entry["actual"].get(m),
                     "target": entry["target"].get(m)} for m in MODES]
            body = bullet_rows(rows, max_pct=70.0, row_height=22)
        l1 = entry.get("l1")
        l1_txt = f'<span class="cx-l1">L1 {_fmt(l1)} pts</span>' if l1 is not None else ""
        blocks.append(
            f'<div class="cx-cell"><div class="cx-cell-head">'
            f'<span class="cx-cat">{escape(pretty_cat(entry["cat"]))}</span>'
            f'{badge}{l1_txt}</div>{body}</div>')
    return f'<div class="cx-grid-cells">{"".join(blocks)}</div>'


def trajectory(series: Sequence[dict],
               caption: str = "ordre chronologique des nœuds retenus →") -> str:
    """Composite score over iterations, one line per measurement regime.

    Faceting by regime is not a presentation detail: changing the evaluation
    model — or the decision policy, elected mode versus probability
    mass — changes the decisions, hence the level of the score. A single curve
    would read as progress what is only a change of instrument.
    """
    points = [p for s in series for p in s["points"]]
    if not points:
        return ""
    lo = min(p["score"] for p in points)
    hi = max(p["score"] for p in points)
    span = max(1.0, hi - lo)
    lo, hi = lo - span * 0.12, hi + span * 0.12
    w, h, pl, pt, pb = 620, 230, 44, 16, 30
    plot_w, plot_h = w - pl - 12, h - pt - pb
    n_max = max(len(s["points"]) for s in series)
    step = plot_w / max(1, n_max - 1)
    ramp = ["#4A6FE3", "#C2571A", "#7A7A85"]

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" '
             f'role="img" preserveAspectRatio="xMinYMin meet">']
    for frac in (0.0, 0.5, 1.0):
        gy = pt + plot_h * (1 - frac)
        parts.append(f'<line class="cx-grid" x1="{pl}" y1="{gy:.1f}" '
                     f'x2="{pl + plot_w}" y2="{gy:.1f}"/>')
        parts.append(f'<text class="cx-axis-label" x="{pl - 8}" y="{gy + 3:.1f}" '
                     f'text-anchor="end">{lo + (hi - lo) * frac:.0f}</text>')
    for si, serie in enumerate(series):
        color = ramp[si % len(ramp)]
        pts = []
        for i, point in enumerate(serie["points"]):
            px = pl + i * step
            py = pt + plot_h * (1 - (point["score"] - lo) / (hi - lo))
            pts.append((px, py, point))
        if len(pts) > 1:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
                         f'stroke-width="2" stroke-linejoin="round"/>')
        for x, y, point in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}">'
                         f'<title>{escape(point["label"])} — {point["score"]:.2f}</title>'
                         f'</circle>')
    parts.append(f'<text class="cx-axis-label" x="{pl}" y="{h - 8}">'
                 f'{escape(caption)}</text>')
    parts.append("</svg>")
    chips = [(s["label"], ramp[i % len(ramp)]) for i, s in enumerate(series)]
    return "".join(parts) + legend(chips)


def heatmap(dims: Sequence[dict], arms: Sequence[dict]) -> str:
    """Dimension × strand matrix. Single sequential ramp: darker = worse."""
    values = [c["value"] for a in arms for c in a["cells"] if c["value"] is not None]
    if not values:
        return ""
    # Row-wise normalisation: the dimensions do not share a scale (a gender JSD
    # and a composite score do not compare). The question the matrix asks is
    # "which strand does best on this dimension", not "which dimension is
    # the worst", so each row has its own ramp.
    row_max = []
    for i in range(len(dims)):
        row = [a["cells"][i]["value"] for a in arms
               if a["cells"][i]["value"] is not None]
        row_max.append(max(row) if row else 1.0)
    cell_w, cell_h, head_w = 108, 40, 150
    w = head_w + cell_w * len(arms)
    h = 30 + cell_h * len(dims)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" '
             f'role="img" preserveAspectRatio="xMinYMin meet">']
    for j, arm in enumerate(arms):
        parts.append(f'<text class="cx-sub" x="{head_w + j * cell_w + cell_w / 2}" '
                     f'y="18" text-anchor="middle">{escape(arm["label"])}</text>')
    for i, dim in enumerate(dims):
        y = 30 + i * cell_h
        parts.append(f'<text class="cx-label" x="0" y="{y + cell_h / 2 + 4}">'
                     f'{escape(dim["label"])}</text>')
        for j, arm in enumerate(arms):
            cell = arm["cells"][i]
            x = head_w + j * cell_w
            if cell["value"] is None:
                parts.append(f'<rect class="cx-cell-na" x="{x + 2}" y="{y + 2}" '
                             f'width="{cell_w - 4}" height="{cell_h - 4}" rx="4"/>')
                parts.append(f'<text class="cx-na" x="{x + cell_w / 2}" '
                             f'y="{y + cell_h / 2 + 4}" text-anchor="middle">n. d.</text>')
                continue
            alpha = 0.12 + 0.72 * (cell["value"] / (row_max[i] or 1.0))
            parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" '
                         f'height="{cell_h - 4}" rx="4" fill="#4A6FE3" '
                         f'fill-opacity="{alpha:.2f}"/>')
            parts.append(f'<text class="cx-heat-value" x="{x + cell_w / 2}" '
                         f'y="{y + cell_h / 2 + 4}" text-anchor="middle">'
                         f'{cell["value"]:.1f}</text>')
    parts.append("</svg>")
    return "".join(parts)
