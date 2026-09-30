"""Dashboard palette.

The mode colours are the ones imposed by `.claude/CLAUDE.md` (visual
consistency across notebooks / GAMA / Grafana). Each hue comes in two steps:
one for the light surface, one for the dark surface.

⚠️  Accepted constraint: the imposed hues (red / green / cyan / magenta)
do not reach the colour-blindness separation threshold (ΔE deutan ≈ 3–4, threshold 8).
This cannot be fixed without dropping the official palette. Colour therefore
NEVER carries identity in this dashboard: every mode chart is a
horizontal bar chart with the mode written on the axis and the value at the end
of the bar, paired with a table view. Colour only reinforces.
"""

from __future__ import annotations

# Modes as written in moves.csv → (light step, dark step)
MODE_COLORS: dict[str, tuple[str, str]] = {
    "Voiture Privée": ("#CE3B4B", "#E4646F"),  # red
    "Vélo": ("#7C4DDB", "#9C7BEE"),  # purple
    "Transports_collectifs": ("#178A3F", "#3BAE60"),  # green
    "Marche": ("#0B7A9B", "#329BB8"),  # cyan
    "Deux-roues motorisé": ("#B5259B", "#D658B0"),  # magenta
    "Train": ("#5B3AB8", "#7E68D8"),  # deep purple ("purple" family)
}

# Modes outside the official palette (non-trips, leftovers): neutral grey.
NEUTRAL = ("#6E6D69", "#98968E")

# States: reserved for health, never reused as a series colour.
STATUS = {
    "good": ("#178A3F", "#3BAE60"),
    "warning": ("#A66A00", "#D19A2E"),
    "critical": ("#C02A2A", "#E36A6A"),
    "muted": ("#6E6D69", "#98968E"),
}

MODE_LABELS = {
    "Voiture Privée": "Voiture",
    "Transports_collectifs": "Transports collectifs",
    "Deux-roues motorisé": "Deux-roues motorisé",
}


def mode_color(mode: str, dark: bool) -> str:
    pair = MODE_COLORS.get(mode, NEUTRAL)
    return pair[1] if dark else pair[0]


def status_color(kind: str, dark: bool) -> str:
    pair = STATUS.get(kind, STATUS["muted"])
    return pair[1] if dark else pair[0]


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)
